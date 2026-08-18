from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.generation_plan import GenerationPlanningRequest
from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentDecision,
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_REPRESENTATION,
    language_branch_identity,
    minimal_instruction_identity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.generation_plan import GenerationCandidate
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.generation_surface import GenerationSurfaceProtocol
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_connector import (
    ConflictSetConnectorCompileRequest,
    ConflictSetConnectorCompileError,
    ConflictSetSentenceBinding,
    ConflictSetSourceBinding,
    compile_conflict_set_connector,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_contract import (
    ConflictSetEvidence,
    build_conflict_set_plan,
)

from tests.test_g02_generation_structure_plan import (
    _content_protocol,
    _request,
)


def _protocol(seed: int) -> GenerationSurfaceProtocol:
    return GenerationSurfaceProtocol(*tuple(
        minimal_instruction_identity((seed, index))
        for index in range(1, 10)
    ))


def _plan():
    return build_conflict_set_plan(
        scope_id=41,
        claim_ids=("claim-a", "claim-b"),
        evidence=(
            ConflictSetEvidence("e1", "claim-a", "source-1", 41, 1, 0),
            ConflictSetEvidence("e2", "claim-a", "source-2", 41, 0, 1),
            ConflictSetEvidence("e3", "claim-b", "source-3", 41, 1, 0),
            ConflictSetEvidence("e4", "claim-b", "source-4", 41, 0, 1),
        ),
    )


def _conflict_candidate(
        candidate,
        index: int,
        *,
        source_seed: int = 91021,
        stances: tuple[int, int] = (EVIDENCE_SUPPORT, EVIDENCE_REFUTE),
        shared_source: bool = False,
        ) -> GenerationCandidate:
    """把 G-02 候选提升为真实双来源 support/refute candidate。"""
    source = candidate.source
    hypothesis = HypothesisKey(
        (91020, 1),
        (91020, index),
        (91020, 100 + index),
        candidate.scope,
        source,
    )
    support_source = SourceRef(
        source_seed, index * 2 - 1, index, source.owner, source.versions)
    refute_source = SourceRef(
        source_seed, index * 2, index, source.owner, source.versions)
    if shared_source:
        refute_source = support_source
    return GenerationCandidate(
        candidate.proposition,
        LogicEvidenceState(
            EVIDENCE_SUPPORT in stances,
            EVIDENCE_REFUTE in stances,
        ),
        source,
        candidate.scope,
        (
            EvidenceRecord(
                91030 + index * 2,
                hypothesis,
                stances[0],
                (91031, index, 1),
                support_source,
                91030 + index * 2,
            ),
            EvidenceRecord(
                91030 + index * 2 + 1,
                hypothesis,
                stances[1],
                (91031, index, 2),
                refute_source,
                91030 + index * 2 + 1,
            ),
        ),
    )


def _bindings(plan, candidates, surfaces, claim_ids=None):
    claim_ids = plan.claim_ids if claim_ids is None else claim_ids
    claims = {item.claim_id: item for item in plan.claims}
    return tuple(
        ConflictSetSentenceBinding(
            claim_id,
            candidate,
            surface,
            tuple(
                ConflictSetSourceBinding(source_id, source)
                for source_id, source in zip(
                    claims[claim_id].source_ids,
                    candidate.citation_sources,
                    strict=True,
                )
            ),
        )
        for claim_id, candidate, surface in zip(
            claim_ids, candidates, surfaces, strict=True)
    )


def _conflict_selection(request):
    protocol = _content_protocol(91040)

    class _Policy:
        def select(self, current, artifacts):
            del artifacts
            return AnswerContentDecision(
                protocol.conflict,
                minimal_instruction_identity((91041, 1)),
                current.candidate_keys(),
                (),
                (91042, 1),
            )

    selector = AnswerContentSelector(protocol, _Policy())
    return selector.select(request)


def test_conflict_set_connector_compiles_ordered_multisentence_templates():
    base, _unused = _request(count=2)
    branch = language_branch_identity((91001, 1))
    base_request = GenerationPlanningRequest(
        replace(base.goal, target_branch=branch),
        base.candidates,
    )
    plan = _plan()
    candidates = tuple(
        _conflict_candidate(candidate, index)
        for index, candidate in enumerate(base_request.candidates, start=1)
    )
    compilation = compile_conflict_set_connector(
        ConflictSetConnectorCompileRequest(
            plan,
            _bindings(plan, candidates, ("第一命题。", "第二命题。")),
            branch,
            _protocol(91002),
            (91004,),
            candidates[0].source,
            (91003,),
        )
    )
    assert tuple(item.claim_id for item in compilation.sentences) == plan.claim_ids
    assert len(compilation.connector.registry.templates) == 2
    assert compilation.connector.discourse_declarations is not None
    assert all(
        len(template.slots) == 2
        and len(template.bindings) == 2
        and len(template.surface) == 2
        for template in compilation.connector.registry.templates
    )
    assert all(
        template.surface[0].action == compilation.connector.surface_protocol.silent_action
        and template.surface[1].action == compilation.connector.surface_protocol.emit_action
        for template in compilation.connector.registry.templates
    )
    generated = compilation.generate_surfaces(("A", "B"))
    assert tuple(item.claim_id for item in generated) == plan.claim_ids
    assert tuple(item.source_ids for item in generated) == (
        ("source-1", "source-2"),
        ("source-3", "source-4"),
    )
    assert len(compilation.alias_pairs) == 2
    assert all(pair[1].object_kind == OBJECT_REPRESENTATION
               for pair in compilation.alias_pairs)
    request = GenerationPlanningRequest(
        replace(base.goal, target_branch=branch), candidates)
    selection = _conflict_selection(request)
    structure = compilation.connector.structure_planner().plan(selection)
    assert tuple(
        sentence.instance.candidate_key
        for sentence in structure.syntax.sentences
    ) == tuple(candidate.stable_key() for candidate in request.candidates)
    assert tuple(
        sentence.values[1].filler
        for sentence in structure.syntax.sentences
    ) == tuple(
        item.template.bindings[1].constant for item in compilation.sentences
    )


def test_conflict_set_connector_rejects_claim_order_or_binding_drift():
    base, _unused = _request(count=2)
    branch = language_branch_identity((91011, 1))
    plan = _plan()
    candidates = tuple(
        _conflict_candidate(candidate, index)
        for index, candidate in enumerate(base.candidates, start=1)
    )
    with pytest.raises(ConflictSetConnectorCompileError):
        compile_conflict_set_connector(
            ConflictSetConnectorCompileRequest(
                plan,
                _bindings(
                    plan,
                    candidates,
                    ("x", "x"),
                    claim_ids=("claim-b", "claim-a"),
                ),
                branch,
                _protocol(91012),
                (91014,),
                candidates[0].source,
                (91013,),
            )
        )


def test_conflict_set_connector_rejects_per_source_stance_drift():
    base, _unused = _request(count=2)
    branch = language_branch_identity((91051, 1))
    plan = _plan()
    candidates = (
        _conflict_candidate(
            base.candidates[0],
            1,
            stances=(EVIDENCE_REFUTE, EVIDENCE_SUPPORT),
        ),
        _conflict_candidate(base.candidates[1], 2),
    )
    with pytest.raises(ConflictSetConnectorCompileError):
        compile_conflict_set_connector(
            ConflictSetConnectorCompileRequest(
                plan,
                _bindings(plan, candidates, ("x", "y")),
                branch,
                _protocol(91052),
                (91054,),
                candidates[0].source,
                (91053,),
            )
        )
