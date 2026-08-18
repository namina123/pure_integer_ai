from copy import deepcopy
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentDecision,
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.generation_plan import GenerationPlanningRequest
from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.cognition.shared.identity import (
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_connector import (
    ConflictSetConnectorCompileRequest,
    compile_conflict_set_connector,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_order import (
    install_conflict_set_order_course,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_postcheck import (
    parse_conflict_set_generation_plan,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_contract import (
    ConflictSetEvidence,
    build_conflict_set_plan,
)
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceRuntime,
)
from pure_integer_ai.storage.backend import DictBackend

from tests.test_g02_generation_structure_plan import (
    _content_protocol,
    _request,
)
from tests.test_ph2_generation_generalization_conflict_set_connector import (
    _bindings,
    _conflict_candidate,
)
from tests.test_g03_generation_surface import _alias_fixture
from tests.test_ph2_generation_generalization_conflict_set_connector import _protocol
from tests.test_s07_structure_order import _graphs


def _compile(count=2):
    base, _unused = _request(count=count)
    branch = language_branch_identity((92001, 1))
    base_planning = GenerationPlanningRequest(
        replace(base.goal, target_branch=branch),
        base.candidates,
    )
    candidates = tuple(
        _conflict_candidate(candidate, index)
        for index, candidate in enumerate(base_planning.candidates, start=1)
    )
    planning = GenerationPlanningRequest(base_planning.goal, candidates)
    claim_ids = tuple(f"claim-{index}" for index in range(1, count + 1))
    plan = build_conflict_set_plan(
        scope_id=41,
        claim_ids=claim_ids,
        evidence=tuple(
            ConflictSetEvidence(
                f"evidence-{claim_index}-{stance_index}",
                claim_id,
                f"source-{claim_index}-{stance_index}",
                41,
                1 if stance_index == 1 else 0,
                1 if stance_index == 2 else 0,
            )
            for claim_index, claim_id in enumerate(claim_ids, start=1)
            for stance_index in (1, 2)
        ),
    )
    compilation = compile_conflict_set_connector(
        ConflictSetConnectorCompileRequest(
            plan,
            _bindings(
                plan,
                planning.candidates,
                tuple(f"第{index}命题。" for index in range(1, count + 1)),
            ),
            branch,
            _protocol(92002),
            (92003,),
            planning.candidates[0].source,
            (92004,),
        )
    )
    return planning, compilation


def _selection(planning):
    protocol = _content_protocol(92010)

    class _Policy:
        def select(self, request, artifacts):
            del artifacts
            return AnswerContentDecision(
                protocol.conflict,
                minimal_instruction_identity((92011, 1)),
                request.candidate_keys(),
                (),
                (92012, 1),
            )

    return AnswerContentSelector(protocol, _Policy()).select(planning)


def _execute(planning, compilation):
    """执行真实 G-02/S-07/R-01，并在返回完整 plan 后释放测试 backend。"""
    backend = DictBackend()
    alias_fixture = None
    try:
        graphs = _graphs(backend)
        order = install_conflict_set_order_course(
            compilation,
            graphs.lifecycle,
            planning.candidates[0].source,
            document_scope(planning.candidates[0].source),
            (92005,),
        )
        alias_fixture = _alias_fixture(
            compilation.language_branch,
            compilation.alias_pairs,
        )
        structure = compilation.connector.structure_planner().plan(
            _selection(planning))
        builder = compilation.connector.surface_request_builder(
            order.execution_planner)
        surface_request = builder.build(structure)
        run = GenerationSurfaceRuntime(alias_fixture.runtime).plan(surface_request)
        assert run.complete
        return run.plan
    finally:
        if alias_fixture is not None:
            alias_fixture.close()
        backend.close()


def _replace_actual_candidates(plan, replacements, originals):
    """只篡改已完成 surface plan 的运行期候选地址，模拟 postcheck 漂移。"""
    request = plan.preview.request
    object.__setattr__(
        request.structure.selection.request,
        "candidates",
        tuple(replacements),
    )
    index_by_key = {
        item.stable_key(): index
        for index, item in enumerate(originals)
    }
    for sentence in request.structure.syntax.sentences:
        instance = sentence.instance
        index = index_by_key[instance.candidate_key]
        object.__setattr__(
            instance,
            "candidate_key",
            replacements[index].stable_key(),
        )


@pytest.mark.parametrize("count", (2, 3))
def test_conflict_set_runs_real_s07_r01_and_g04_projection(count):
    planning, compilation = _compile(count)
    run_plan = _execute(planning, compilation)
    assert len(run_plan.representations) == count
    parsed = parse_conflict_set_generation_plan(compilation, run_plan)
    assert parsed.status == "PASS"
    assert parsed.projection == compilation.plan.projection


def test_conflict_set_postcheck_reports_generation_unavailable_as_ne():
    _planning, compilation = _compile()
    parsed = parse_conflict_set_generation_plan(compilation, None)
    assert parsed.status == "NE"
    assert parsed.projection is None


def test_conflict_set_postcheck_rejects_unknown_actual_source_as_ne():
    planning, compilation = _compile()
    run_plan = _execute(planning, compilation)
    replacements = tuple(
        _conflict_candidate(candidate, index, source_seed=92921)
        for index, candidate in enumerate(planning.candidates, start=1)
    )
    _replace_actual_candidates(run_plan, replacements, planning.candidates)
    parsed = parse_conflict_set_generation_plan(compilation, run_plan)
    assert parsed.status == "NE"
    assert parsed.projection is None


def test_conflict_set_postcheck_rejects_missing_actual_source():
    planning, compilation = _compile()
    run_plan = _execute(planning, compilation)
    replacements = tuple(
        _conflict_candidate(candidate, index, shared_source=True)
        for index, candidate in enumerate(planning.candidates, start=1)
    )
    _replace_actual_candidates(run_plan, replacements, planning.candidates)
    parsed = parse_conflict_set_generation_plan(compilation, run_plan)
    assert parsed.status == "FAIL"
    assert parsed.projection is not None
    assert parsed.projection.cited_source_ids != (
        compilation.plan.projection.cited_source_ids)


def test_conflict_set_postcheck_rejects_actual_support_only_candidate():
    planning, compilation = _compile()
    run_plan = _execute(planning, compilation)
    replacements = tuple(
        _conflict_candidate(
            candidate,
            index,
            stances=(EVIDENCE_SUPPORT, EVIDENCE_SUPPORT),
        )
        for index, candidate in enumerate(planning.candidates, start=1)
    )
    _replace_actual_candidates(run_plan, replacements, planning.candidates)
    parsed = parse_conflict_set_generation_plan(compilation, run_plan)
    assert parsed.status == "FAIL"
    assert parsed.projection is not None
    assert all(
        state == (claim_id, 1, 0)
        for state, claim_id in zip(
            parsed.projection.claim_states,
            compilation.plan.claim_ids,
            strict=True,
        )
    )


def test_conflict_set_postcheck_rejects_reversed_actual_sentence_order():
    planning, compilation = _compile()
    run_plan = deepcopy(_execute(planning, compilation))
    syntax = run_plan.preview.request.structure.syntax
    object.__setattr__(syntax, "sentences", tuple(reversed(syntax.sentences)))
    parsed = parse_conflict_set_generation_plan(compilation, run_plan)
    assert parsed.status == "FAIL"
    assert parsed.projection is not None
    assert parsed.projection.claim_ids == tuple(
        reversed(compilation.plan.claim_ids))


def test_conflict_set_postcheck_reports_actual_slot_drift_as_ne():
    planning, compilation = _compile()
    run_plan = deepcopy(_execute(planning, compilation))
    sentence = run_plan.preview.request.structure.syntax.sentences[0]
    claim_slot = compilation.sentences[0].claim_slot
    claim_value = next(
        item for item in sentence.values if item.slot == claim_slot)
    object.__setattr__(claim_value, "filler",
                       compilation.sentences[1].claim_filler)
    parsed = parse_conflict_set_generation_plan(compilation, run_plan)
    assert parsed.status == "NE"
    assert parsed.projection is None
