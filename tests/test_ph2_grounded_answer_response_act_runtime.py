"""grounded CLARIFY 经 response-act、S-07、R-01、parser 与 G-04 的专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentProtocol,
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructureLayerProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.question_answer import (
    EvidenceAnswerPolicy,
    EvidenceAnswerPolicyProtocol,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSourceRequirement,
    RecoveredGenerationProposition,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.experiments.ph2_grounded_answer_compile import (
    compile_grounded_answer_training_records,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    PATTERN_LITERAL,
    learn_grounded_answer_surface_model,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_compile import (
    GroundedResponseActCompileTarget,
    compile_grounded_response_act_patterns,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_choice_use import (
    GroundedResponseActLexicalAdoptionLedger,
    GroundedResponseActLexicalUseError,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_parser import (
    GroundedResponseActParserProtocol,
    GroundedResponseActSourceVerifier,
    GroundedResponseActStructureVerifier,
    GroundedResponseActTaskVerifier,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationSourceCheckRequest,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_runtime_factory import (
    GroundedResponseActRunLocalBuild,
    GroundedResponseActRunLocalComponents,
    GroundedResponseActRunLocalFactory,
)
from pure_integer_ai.experiments.conversation_context_runtime import (
    start_conversation_context,
)
from pure_integer_ai.experiments.ph2_generation_generalization_executable_train_course import (
    read_generation_generalization_executable_train_course,
)
from pure_integer_ai.experiments.ph2_generation_generalization_executable_train_rehearsal import (
    GenerationGeneralizationTrainRehearsal,
    default_source_conflict_protocol,
    rehearse_grounded_response_act_case,
)
from pure_integer_ai.experiments.ph2_generation_generalization_source_conflict import (
    GenerationGeneralizationSourceConflictInput,
    run_generation_generalization_source_conflict_verification,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    compile_grounded_response_act_planning,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerProtocol,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_NOT_APPLICABLE,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
)
from pure_integer_ai.storage.backend import DictBackend

from tests.test_g01_generation_content import _T, _request
from tests.test_g02_generation_structure_plan import _plan_protocol
from tests.test_g03_generation_surface import _alias_fixture, _surface_protocol
from tests.test_g04_generation_postcheck import (
    _StaticVerifier,
    _protocol as _postcheck_protocol,
)
from tests.test_s07_structure_order import _graphs


_BASE = 20963
_SAMPLE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")
_CASE_SAMPLE = Path(
    "data/ph2/generation_generalization_executable_train_case_v1.jsonl.sample")


class _AliasFactory:
    """按 selected variant 建立 stance 到 learned literal 的独占 R-01 图。"""

    def __init__(self, branch) -> None:
        self.branch = branch
        self.fixture = None

    def build(self, variant):
        """只物化当前 stance/Representation，不读取 pattern id 返回文本。"""
        if self.fixture is not None:
            raise RuntimeError("response-act alias factory 不得重复 build")
        self.fixture = _alias_fixture(
            self.branch,
            ((variant.template.stance, variant.representation),),
        )
        return self.fixture.runtime


def test_clarify_runs_as_learned_response_act_with_selected_candidates():
    """CLARIFY 保留歧义候选，但实际生成零命题并由 G-04 恢复任务。"""
    episodes = read_grounded_answer_episodes(_SAMPLE)
    clarify_question = next(
        item.question for item in episodes
        if item.question.answer_plan.response_act == "CLARIFY")
    model, _report = learn_grounded_answer_surface_model(
        compile_grounded_answer_training_records(_SAMPLE))
    branch = language_branch_identity((_BASE, 1))
    raw_planning, _source, _scope, _bound = _request(states=(_T, _T))
    planning = GenerationPlanningRequest(
        replace(raw_planning.goal, target_branch=branch),
        raw_planning.candidates,
    )
    content_protocol = AnswerContentProtocol(*tuple(
        minimal_instruction_identity((_BASE, 2, index))
        for index in range(1, 6)
    ))
    policy_protocol = EvidenceAnswerPolicyProtocol(*tuple(
        minimal_instruction_identity((_BASE, 3, index))
        for index in range(1, 5)
    ))
    selector = AnswerContentSelector(
        content_protocol,
        EvidenceAnswerPolicy(content_protocol, policy_protocol),
    )
    selection = selector.select(planning)
    assert selection.stance == content_protocol.clarify
    assert len(selection.selected_candidate_keys) == 2

    target = GroundedResponseActCompileTarget(
        "CLARIFY",
        content_protocol.clarify,
        branch,
        (_BASE, 4),
    )
    compilation = compile_grounded_response_act_patterns(model, target)
    selected_pattern = compilation.variants[0]
    learned_literal = next(
        pattern.parts[0].literal for pattern in model.patterns
        if (pattern.pattern_id == selected_pattern.pattern_id
            and pattern.parts[0].kind == PATTERN_LITERAL)
    )
    backend = DictBackend()
    alias_factory = _AliasFactory(branch)
    try:
        graphs = _graphs(backend)
        renderer_identity = minimal_instruction_identity((_BASE, 5, 1))
        renderer = UnicodeRepresentationRenderer(
            target.representation_family, renderer_identity)
        components = GroundedResponseActRunLocalComponents(
            selector,
            _plan_protocol(_BASE + 10),
            GenerationStructureLayerProtocol(*tuple(
                minimal_instruction_identity((_BASE, 6, index))
                for index in range(1, 4)
            )),
            _surface_protocol(_BASE + 20),
            alias_factory,
            renderer,
            renderer_identity,
            _postcheck_protocol(),
            GroundedResponseActStructureVerifier(
                minimal_instruction_identity((_BASE, 7, 1)),
                minimal_instruction_identity((_BASE, 7, 2)),
            ),
            _StaticVerifier(VERDICT_SUPPORT, 1),
            GroundedResponseActTaskVerifier(
                minimal_instruction_identity((_BASE, 8, 1)),
                minimal_instruction_identity((_BASE, 8, 2)),
            ),
            QuestionAnswerProtocol(*tuple(
                minimal_instruction_identity((_BASE, 9, index))
                for index in range(1, 4)
            )),
        )
        query_kind = minimal_instruction_identity((_BASE, 10, 1))
        route = minimal_instruction_identity((_BASE, 10, 2))
        factory = GroundedResponseActRunLocalFactory(
            graphs.lifecycle, components)
        installation = factory.build(GroundedResponseActRunLocalBuild(
            model,
            clarify_question,
            target,
            planning,
            selected_pattern.pattern_id,
            GroundedResponseActParserProtocol(*tuple(
                minimal_instruction_identity((_BASE, 11, index))
                for index in range(1, 4)
            )),
            query_kind,
            route,
            minimal_instruction_identity((_BASE, 12, 1)),
            (_BASE, 12, 2),
        ))
        question_request = QuestionRequest(
            query_kind,
            minimal_instruction_identity((_BASE, 13, 1)),
            planning.goal.goal_kind,
            planning.goal.proposition,
            planning.goal.required,
            planning.goal.scope,
            planning.goal.scope,
            (_BASE, 13, 2),
            branch,
            tuple(item.proposition for item in planning.candidates),
        )

        run = installation.runtime.run(question_request)
        ledger = GroundedResponseActLexicalAdoptionLedger(installation)
        lexical_use = ledger.adopt(run)

        assert run.complete
        assert run.selection is not None
        assert len(run.selection.selected_candidate_keys) == 2
        assert run.generation is not None and run.generation.complete
        structure = run.generation.surface.preview.request.structure
        assert structure.selection.selected_candidate_keys == (
            run.selection.selected_candidate_keys)
        assert {item.candidate_key for item in structure.propositions.propositions} == (
            set(run.selection.selected_candidate_keys))
        assert len(structure.syntax.sentences) == 1
        assert structure.syntax.sentences[0].proposition_keys == ()
        assert structure.syntax.suppressed_candidate_keys == (
            run.selection.selected_candidate_keys)
        assert structure.syntax.sentences[0].values[0].filler == (
            content_protocol.clarify)
        execution = run.generation.surface.preview.request.execution
        assert execution.complete
        assert execution.sentences[0].active_constraints == ()
        assert renderer.text(run.generation.rendered) == learned_literal
        assert len(run.generation.surface.adoptions) == 1
        assert lexical_use.choice_before == installation.lexical_choice
        assert lexical_use.adoptions == run.generation.surface.adoptions
        assert lexical_use.choice_after.exact_uses == (lexical_use.use,)
        assert lexical_use.use.scope == planning.goal.scope
        assert ledger.records == (lexical_use,)
        with pytest.raises(
                GroundedResponseActLexicalUseError,
                match="不得重复登记"):
            ledger.adopt(run)
        assert run.postcheck is not None and run.postcheck.complete
        observation = run.postcheck.parsed.observation
        assert observation is not None
        assert observation.stance == content_protocol.clarify
        assert observation.propositions == ()
        assert observation.representations == run.generation.representations
        assert observation.task_observations[0].result_key == (
            selected_pattern.task_result_key)
        applicable = run.postcheck.report.applicable_results()
        assert all(item.verdict == VERDICT_SUPPORT for item in applicable)
    finally:
        if alias_factory.fixture is not None:
            alias_factory.fixture.close()
        backend.close()


def test_clarify_competes_in_aggregate_scope_across_source_local_evidence():
    """来源、来源 scope、Evidence id 与输入顺序变化不得把 CLARIFY 私选成 ANSWER。"""
    base = next(
        item for item in read_grounded_answer_episodes(_SAMPLE)
        if item.question.answer_plan.response_act == "CLARIFY")
    proposition_ids = tuple(dict.fromkeys(
        item.proposition_id for item in base.question.evidence))
    second = proposition_ids[1]
    evidence = tuple(reversed(tuple(
        replace(
            item,
            evidence_id=f"public-cross-source-{index}",
            source_id=(
                "public-clarify-source-b"
                if item.proposition_id == second
                else "public-clarify-source-a"),
            scope_id=303,
        )
        for index, item in enumerate(base.question.evidence, start=1)
    )))
    question = replace(
        base.question,
        evidence_scope_id=303,
        response_scope_id=403,
        evidence=evidence,
    )
    episode = replace(
        base,
        episode_id="public-clarify-cross-source-scope-id-order-v1",
        question=question,
        dialogue=replace(base.dialogue, active_scope_ids=(303, 403)),
    )
    branch = language_branch_identity((_BASE, 300))
    planning = compile_grounded_response_act_planning(episode, branch)
    content = AnswerContentProtocol(*tuple(
        minimal_instruction_identity((_BASE, 301, index))
        for index in range(1, 6)))
    policy = EvidenceAnswerPolicyProtocol(*tuple(
        minimal_instruction_identity((_BASE, 302, index))
        for index in range(1, 5)))

    selection = AnswerContentSelector(
        content, EvidenceAnswerPolicy(content, policy)).select(
            planning.planning)

    assert len({
        hypothesis.observation
        for candidate in planning.planning.candidates
        for hypothesis in candidate.hypotheses
    }) == 2
    assert selection.stance == content.clarify
    assert len(selection.selected_candidate_keys) == 2


def _rehearse_response_act(requirement, response_act, offset):
    """用 catalog episode、aggregate planning 和真实 runtime 运行一项 rehearsal。"""
    course = read_generation_generalization_executable_train_course(
        _CASE_SAMPLE, _SAMPLE)
    case = course.case_for_requirement(requirement)
    episode = course.episode_for(case)
    model, _report = learn_grounded_answer_surface_model(
        compile_grounded_answer_training_records(_SAMPLE))
    branch = language_branch_identity((_BASE, 100, offset))
    planning_build = compile_grounded_response_act_planning(episode, branch)
    planning = planning_build.planning
    content_protocol = AnswerContentProtocol(*tuple(
        minimal_instruction_identity((_BASE, 101 + offset, index))
        for index in range(1, 6)
    ))
    policy_protocol = EvidenceAnswerPolicyProtocol(*tuple(
        minimal_instruction_identity((_BASE, 111 + offset, index))
        for index in range(1, 5)
    ))
    selector = AnswerContentSelector(
        content_protocol,
        EvidenceAnswerPolicy(content_protocol, policy_protocol),
    )
    stance = getattr(content_protocol, response_act.lower())
    target = GroundedResponseActCompileTarget(
        response_act,
        stance,
        branch,
        (_BASE, 120, offset),
    )
    compilation = compile_grounded_response_act_patterns(model, target)
    selected_pattern = compilation.variants[0]
    backend = DictBackend()
    alias_factory = _AliasFactory(branch)
    try:
        graphs = _graphs(backend)
        renderer_identity = minimal_instruction_identity(
            (_BASE, 121, offset))
        renderer = UnicodeRepresentationRenderer(
            target.representation_family, renderer_identity)
        components = GroundedResponseActRunLocalComponents(
            selector,
            _plan_protocol(_BASE + 130 + offset),
            GenerationStructureLayerProtocol(*tuple(
                minimal_instruction_identity(
                    (_BASE, 140 + offset, index))
                for index in range(1, 4)
            )),
            _surface_protocol(_BASE + 150 + offset),
            alias_factory,
            renderer,
            renderer_identity,
            _postcheck_protocol(),
            GroundedResponseActStructureVerifier(
                minimal_instruction_identity((_BASE, 160 + offset, 1)),
                minimal_instruction_identity((_BASE, 160 + offset, 2)),
            ),
            _StaticVerifier(VERDICT_SUPPORT, 20 + offset),
            GroundedResponseActTaskVerifier(
                minimal_instruction_identity((_BASE, 170 + offset, 1)),
                minimal_instruction_identity((_BASE, 170 + offset, 2)),
            ),
            QuestionAnswerProtocol(*tuple(
                minimal_instruction_identity(
                    (_BASE, 180 + offset, index))
                for index in range(1, 4)
            )),
        )
        query_kind = minimal_instruction_identity((_BASE, 190 + offset, 1))
        route = minimal_instruction_identity((_BASE, 190 + offset, 2))
        installation = GroundedResponseActRunLocalFactory(
            graphs.lifecycle, components).build(
                GroundedResponseActRunLocalBuild(
                    model,
                    episode.question,
                    target,
                    planning,
                    selected_pattern.pattern_id,
                    GroundedResponseActParserProtocol(*tuple(
                        minimal_instruction_identity(
                            (_BASE, 200 + offset, index))
                        for index in range(1, 4)
                    )),
                    query_kind,
                    route,
                    minimal_instruction_identity(
                        (_BASE, 210 + offset, 1)),
                    (_BASE, 210 + offset, 2),
                ))
        request = QuestionRequest(
            query_kind,
            minimal_instruction_identity((_BASE, 220 + offset, 1)),
            planning.goal.goal_kind,
            planning.goal.proposition,
            planning.goal.required,
            planning.goal.scope,
            planning.goal.scope,
            (_BASE, 220 + offset, 2),
            branch,
            tuple(item.proposition for item in planning.candidates),
        )
        item, run = rehearse_grounded_response_act_case(
            course, requirement, planning_build, installation, request)
        return course, planning_build, item, run
    finally:
        if alias_factory.fixture is not None:
            alias_factory.fixture.close()
        backend.close()


def test_response_act_source_verifier_requires_empty_non_answer_sources():
    """non-answer G-04 只能回读零命题、零引用和零来源 requirement。"""
    _course, planning_build, _item, run = _rehearse_response_act(
        "COMMUNICATIVE_TASK", "CLARIFY", 31)
    postcheck = run.postcheck
    assert postcheck is not None
    observation = postcheck.parsed.observation
    assert observation is not None
    goal = postcheck.request.execution.plan.request.goal
    verifier = GroundedResponseActSourceVerifier(
        minimal_instruction_identity((_BASE, 240, 1)),
        minimal_instruction_identity((_BASE, 240, 2)),
    )

    clean = verifier.verify(GenerationSourceCheckRequest(
        postcheck.request,
        observation,
        (),
        (),
    ))

    assert clean.verdict == VERDICT_SUPPORT
    assert clean.claim_keys == (postcheck.request.execution.stable_key(),)
    assert clean.source == goal.source
    assert clean.scope == goal.scope

    candidate = planning_build.planning.candidates[0]
    recovered = RecoveredGenerationProposition(
        candidate.stable_key(),
        candidate.proposition,
        candidate.source,
        candidate.scope,
        (_BASE, 241, 1),
    )
    requirement = GenerationSourceRequirement(
        candidate.stable_key(),
        candidate.source,
        candidate.scope,
        True,
        True,
        (_BASE, 241, 2),
        candidate.citation_sources,
    )
    violating_requests = (
        GenerationSourceCheckRequest(
            postcheck.request,
            replace(observation, cited_sources=(candidate.source,)),
            (),
            (),
        ),
        GenerationSourceCheckRequest(
            postcheck.request,
            observation,
            (requirement,),
            (),
        ),
        GenerationSourceCheckRequest(
            postcheck.request,
            replace(observation, propositions=(recovered,)),
            (),
            (),
        ),
        GenerationSourceCheckRequest(
            postcheck.request,
            observation,
            (),
            (recovered,),
        ),
    )

    for request in violating_requests:
        evaluation = verifier.verify(request)
        assert evaluation.verdict == VERDICT_REFUTE
        assert evaluation.source == goal.source
        assert evaluation.scope == goal.scope


def test_response_act_catalog_rehearses_task_and_cross_source_conflict():
    """CLARIFY 与真实跨来源 CONFLICT 各形成独立 actual requirement 结果。"""
    course, clarify_planning, clarify, clarify_run = _rehearse_response_act(
        "COMMUNICATIVE_TASK", "CLARIFY", 1)
    same_course, conflict_planning, conflict, conflict_run = (
        _rehearse_response_act(
            "SOURCE_UNCERTAINTY_CITATION", "CONFLICT", 2))
    assert same_course == course

    partial = GenerationGeneralizationTrainRehearsal(
        course, (clarify, conflict))
    assert clarify.passed == 1
    assert conflict.passed == 1
    assert partial.complete == 0
    assert clarify.verification.result.dimension == (
        clarify_run.postcheck.protocol.task_dimension)
    assert conflict.verification.result.dimension != (
        conflict_run.postcheck.protocol.source_dimension)
    g04_source = next(
        item for item in conflict_run.postcheck.report.results
        if item.dimension == conflict_run.postcheck.protocol.source_dimension)
    assert g04_source.applicability == APPLICABILITY_NOT_APPLICABLE
    assert len(conflict_planning.source_bindings) == 2
    candidate = conflict_planning.candidate_bindings[0].candidate
    assert candidate.state.support and candidate.state.refute
    assert len({
        item.hypothesis.observation for item in candidate.evidence}) == 2
    assert conflict_planning.aggregate_source in conflict.choice.forming_sources
    assert clarify_planning.aggregate_source in clarify.choice.forming_sources
    assert conflict.verification.input_key.components == (
        conflict.verification.result.claim_keys[0])

    observation = conflict.postcheck.parsed.observation
    assert observation is not None
    biased_postcheck = replace(
        conflict.postcheck,
        parsed=replace(
            conflict.postcheck.parsed,
            observation=replace(
                observation,
                cited_sources=(conflict_planning.source_bindings[0].source,),
            ),
        ),
    )
    protocol = default_source_conflict_protocol()
    biased_report = run_generation_generalization_source_conflict_verification(
        protocol,
        GenerationGeneralizationSourceConflictInput(
            conflict.source_episode,
            conflict_planning,
            conflict.choice,
            conflict.use,
            conflict.execution,
            conflict.parse_request,
            biased_postcheck,
        ),
    )
    assert biased_report.results[0].verdict == VERDICT_REFUTE


def test_unknown_without_evidence_runs_as_empty_candidate_response_act():
    """无 Evidence 的 UNKNOWN 仍完成 actual surface/readback，但不产生候选。"""
    episode = next(
        item for item in read_grounded_answer_episodes(_SAMPLE)
        if item.episode_id == "train-grounded-unknown-budget-v1")
    model, _report = learn_grounded_answer_surface_model(
        compile_grounded_answer_training_records(_SAMPLE))
    offset = 30
    branch = language_branch_identity((_BASE, 100, offset))
    planning_build = compile_grounded_response_act_planning(episode, branch)
    assert planning_build.planning.candidates == ()
    content_protocol = AnswerContentProtocol(*tuple(
        minimal_instruction_identity((_BASE, 101 + offset, index))
        for index in range(1, 6)
    ))
    selector = AnswerContentSelector(
        content_protocol,
        EvidenceAnswerPolicy(
            content_protocol,
            EvidenceAnswerPolicyProtocol(*tuple(
                minimal_instruction_identity((_BASE, 111 + offset, index))
                for index in range(1, 5)
            )),
        ),
    )
    target = GroundedResponseActCompileTarget(
        "UNKNOWN", content_protocol.unknown, branch, (_BASE, 120, offset))
    compilation = compile_grounded_response_act_patterns(model, target)
    backend = DictBackend()
    alias_factory = _AliasFactory(branch)
    try:
        graphs = _graphs(backend)
        renderer_identity = minimal_instruction_identity(
            (_BASE, 121, offset))
        renderer = UnicodeRepresentationRenderer(
            target.representation_family, renderer_identity)
        components = GroundedResponseActRunLocalComponents(
            selector,
            _plan_protocol(_BASE + 130 + offset),
            GenerationStructureLayerProtocol(*tuple(
                minimal_instruction_identity(
                    (_BASE, 140 + offset, index))
                for index in range(1, 4)
            )),
            _surface_protocol(_BASE + 150 + offset),
            alias_factory,
            renderer,
            renderer_identity,
            _postcheck_protocol(),
            GroundedResponseActStructureVerifier(
                minimal_instruction_identity((_BASE, 160 + offset, 1)),
                minimal_instruction_identity((_BASE, 160 + offset, 2)),
            ),
            _StaticVerifier(VERDICT_SUPPORT, 20 + offset),
            GroundedResponseActTaskVerifier(
                minimal_instruction_identity((_BASE, 170 + offset, 1)),
                minimal_instruction_identity((_BASE, 170 + offset, 2)),
            ),
            QuestionAnswerProtocol(*tuple(
                minimal_instruction_identity(
                    (_BASE, 180 + offset, index))
                for index in range(1, 4)
            )),
        )
        selected = compilation.variants[0]
        query_kind = minimal_instruction_identity((_BASE, 190 + offset, 1))
        installation = GroundedResponseActRunLocalFactory(
            graphs.lifecycle, components).build(
                GroundedResponseActRunLocalBuild(
                    model,
                    episode.question,
                    target,
                    planning_build.planning,
                    selected.pattern_id,
                    GroundedResponseActParserProtocol(*tuple(
                        minimal_instruction_identity(
                            (_BASE, 200 + offset, index))
                        for index in range(1, 4)
                    )),
                    query_kind,
                    minimal_instruction_identity((_BASE, 190 + offset, 2)),
                    minimal_instruction_identity((_BASE, 210 + offset, 1)),
                    (_BASE, 210 + offset, 2),
                ))
        request = QuestionRequest(
            query_kind,
            minimal_instruction_identity((_BASE, 220 + offset, 1)),
            planning_build.planning.goal.goal_kind,
            planning_build.planning.goal.proposition,
            planning_build.planning.goal.required,
            planning_build.planning.goal.scope,
            planning_build.planning.goal.scope,
            (_BASE, 220 + offset, 2),
            branch,
            (),
        )
        run = installation.runtime.run(request)
        assert run.status == content_protocol.unknown
        assert run.selection is not None
        assert run.selection.selected_candidate_keys == ()
        assert run.generation is not None and run.generation.complete
        assert run.generation.plan.request.candidates == ()
        assert run.generation.rendered is not None
        assert run.generation.rendered.units
        assert run.postcheck is not None and run.postcheck.complete
        assert run.postcheck.parsed.observation.stance == content_protocol.unknown
        context = start_conversation_context((_BASE, 230, offset))
        first_context = context.append(run)
        assert first_context.revision == 1
        first_turn = first_context.turns[0]
        assert first_turn.response_stance == content_protocol.unknown
        assert first_turn.selected_candidate_keys == ()
        assert first_turn.cited_sources == ()
        assert first_turn.context_read is not None
        assert first_turn.context_read.turns == ()
        assert not hasattr(first_turn, "surface")
        with pytest.raises(ValueError, match="必须显式绑定"):
            first_context.append(run)

        # 第二回合只把显式 typed read 键带入 request trace；不存在可读取的
        # 上一轮问题/回答 surface 字段。
        context_read = first_context.read(1)
        assert context_read.turns == (first_turn,)
        assert not hasattr(context_read, "surface")
        second_request = context_read.bind_request(request)
        second_run = installation.runtime.run(second_request)
        assert second_run.complete
        with pytest.raises(ValueError, match="过期或其他会话"):
            first_context.append_consumed(second_run, context.read(0))
        second_context = first_context.append_consumed(
            second_run, context_read)
        assert second_context.revision == 2
        assert second_context.previous_digest == first_context.digest()
        second_turn = second_context.turns[1]
        assert second_turn.context_read == context_read
        assert second_turn.response_stance == first_turn.response_stance
        assert second_turn.selected_candidate_keys == (
            first_turn.selected_candidate_keys)
        assert second_turn.cited_sources == first_turn.cited_sources
        assert second_turn.discourse_sentence_keys == (
            first_turn.discourse_sentence_keys)
        assert second_turn.parser_revision == first_turn.parser_revision
        assert second_context.visible_turns(1) == (second_turn,)
    finally:
        if alias_factory.fixture is not None:
            alias_factory.fixture.close()
        backend.close()
