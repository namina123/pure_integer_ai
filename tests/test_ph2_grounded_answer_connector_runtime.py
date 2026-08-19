"""grounded-answer pattern 经真实 connector、executor 与 G-04 的专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructureLayerProtocol,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.identity import (
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    EvidenceQuestionPostcheckMapper,
    QuestionAnswerProtocol,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerClaimInput,
    GroundedAnswerConnectorTarget,
    compile_grounded_answer_connectors,
)
from pure_integer_ai.experiments.ph2_grounded_answer_choice_use import (
    GroundedAnswerLexicalAdoptionLedger,
    GroundedAnswerLexicalUseError,
)
from pure_integer_ai.experiments.ph2_grounded_answer_choice import (
    build_grounded_answer_lexical_choice,
)
from pure_integer_ai.experiments.ph2_grounded_answer_compile import (
    compile_grounded_answer_training_records,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    learn_grounded_answer_surface_model,
    surface_pattern_structure_id,
)
from pure_integer_ai.experiments.ph2_grounded_answer_layer_choice_use import (
    GroundedAnswerContentTaskAdoptionLedger,
    GroundedAnswerLayerChoiceUseError,
)
from pure_integer_ai.experiments.ph2_grounded_answer_structure_choice_use import (
    GroundedAnswerStructureAdoptionLedger,
    GroundedAnswerStructureChoiceUseError,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserProtocol,
)
from pure_integer_ai.experiments.ph2_grounded_answer_runtime_factory import (
    GroundedAnswerRunLocalBuild,
    GroundedAnswerRunLocalComponents,
    GroundedAnswerRunLocalFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_verification import (
    GroundedAnswerEvidenceSourceVerifier,
    GroundedAnswerStructureVerifier,
)
from pure_integer_ai.experiments.ph2_generation_generalization_executable_train_course import (
    read_generation_generalization_executable_train_course,
)
from pure_integer_ai.experiments.ph2_generation_generalization_executable_train_rehearsal import (
    GenerationGeneralizationTrainRehearsal,
    default_answer_verification_protocol,
    rehearse_grounded_answer_case,
)
from pure_integer_ai.experiments.ph2_generation_generalization_answer_verification import (
    GenerationGeneralizationAnswerVerificationInput,
    run_generation_generalization_answer_verification,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    compile_grounded_answer_planning,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
)
from pure_integer_ai.storage.backend import DictBackend

from tests.test_g02_generation_structure_plan import (
    _plan_protocol,
    _selection,
)
from tests.test_g03_generation_surface import (
    _alias_fixture,
    _surface_protocol,
)
from tests.test_g04_generation_postcheck import (
    _StaticVerifier,
    _protocol as _postcheck_protocol,
)
from tests.test_ph2_grounded_answer_course import (
    _connector_question_and_candidate,
)
from tests.test_s07_structure_order import (
    _graphs,
)


_BASE = 20930
_GROUNDED_SAMPLE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")
_CASE_SAMPLE = Path(
    "data/ph2/generation_generalization_executable_train_case_v1.jsonl.sample")


def test_grounded_connector_accepts_typed_claim_input_without_course_episode():
    """generic ANSWER connector 只需本次 claim surface，不读取课程 episode。"""
    model, question, _planning, candidate, branch = (
        _connector_question_and_candidate())
    claim_text = next(
        item.claim_text
        for item in question.evidence
        if item.proposition_id == question.answer_plan.ordered_claim_ids[0]
    )
    target = GroundedAnswerConnectorTarget(
        candidate.proposition, branch, (_BASE, 1))
    from tests.test_g03_generation_surface import _surface_protocol
    generic = GroundedAnswerClaimInput(claim_text)
    course = compile_grounded_answer_connectors(
        model, question, target, _surface_protocol(_BASE + 2))
    detached = compile_grounded_answer_connectors(
        model, generic, target, _surface_protocol(_BASE + 2))
    assert detached.variants == course.variants
    assert detached.structures == course.structures


class _AliasFactory:
    """测试中按 factory 选中的 variant 建立独占 R-01 课程。"""

    def __init__(self, branch) -> None:
        self.branch = branch
        self.fixture = None

    def build(self, variant):
        """只从 variant alias requirement 建立所需 realization。"""
        if self.fixture is not None:
            raise RuntimeError("grounded alias factory 不得重复 build")
        self.fixture = _alias_fixture(
            self.branch,
            tuple((item.filler, item.representation)
                  for item in variant.aliases),
        )
        return self.fixture.runtime


def test_grounded_variant_runs_through_typed_executor_parser_and_g04():
    """已学 literal/claim 槽必须由真实 G-02/G-03 输出并经 G-04 恢复。"""
    model, question, planning, candidate, branch = (
        _connector_question_and_candidate())
    surface_protocol = _surface_protocol(_BASE + 10)
    family = (_BASE + 11, 1)
    selected_pattern = next(
        pattern for pattern in model.patterns
        if any(part.literal == "档案显示，" for part in pattern.parts)
    )
    backend = DictBackend()
    alias_factory = _AliasFactory(branch)
    try:
        selection, selector, _content_protocol = _selection(planning)
        graphs = _graphs(backend)
        target = GroundedAnswerConnectorTarget(
            candidate.proposition, branch, family)
        plan_protocol = _plan_protocol(_BASE + 12)
        structure_protocol = GenerationStructureLayerProtocol(*tuple(
            minimal_instruction_identity((_BASE + 13, index))
            for index in range(1, 4)
        ))
        renderer_identity = minimal_instruction_identity((_BASE + 14, 1))
        renderer = UnicodeRepresentationRenderer(family, renderer_identity)
        parser_protocol = GroundedAnswerParserProtocol(
            *tuple(minimal_instruction_identity((_BASE + 15, index))
                   for index in range(1, 6)),
            selection.stance,
        )
        query_kind = minimal_instruction_identity((_BASE + 16, 1))
        route = minimal_instruction_identity((_BASE + 16, 2))
        components = GroundedAnswerRunLocalComponents(
            selector,
            plan_protocol,
            structure_protocol,
            alias_factory,
            renderer,
            renderer_identity,
            _postcheck_protocol(),
            _StaticVerifier(VERDICT_SUPPORT, 1),
            _StaticVerifier(VERDICT_SUPPORT, 2),
            QuestionAnswerProtocol(*tuple(
                minimal_instruction_identity((_BASE + 17, index))
                for index in range(1, 4)
            )),
            EvidenceQuestionPostcheckMapper(
                (_BASE + 18, 1),
                citation_required=True,
                trust_required=True,
            ),
        )
        factory = GroundedAnswerRunLocalFactory(
            surface_protocol, graphs.lifecycle, components)
        installation = factory.build(GroundedAnswerRunLocalBuild(
            model,
            question,
            target,
            planning,
            candidate,
            surface_pattern_structure_id(selected_pattern),
            selected_pattern.pattern_id,
            parser_protocol,
            query_kind,
            route,
            minimal_instruction_identity((_BASE + 20, 1)),
            (_BASE + 20, 2),
        ))
        request = QuestionRequest(
            query_kind,
            minimal_instruction_identity((_BASE + 16, 3)),
            planning.goal.goal_kind,
            planning.goal.proposition,
            planning.goal.required,
            candidate.scope,
            candidate.scope,
            (_BASE + 16, 4),
            branch,
        )

        run = installation.runtime.run(request)
        ledger = GroundedAnswerLexicalAdoptionLedger(installation)
        lexical_use = ledger.adopt(run)
        layer_ledger = GroundedAnswerContentTaskAdoptionLedger(installation)
        layer_uses = layer_ledger.adopt(run)
        structure_ledger = GroundedAnswerStructureAdoptionLedger(installation)
        structure_use = structure_ledger.adopt(run)

        assert run.complete
        assert run.generation is not None
        assert run.generation.rendered is not None
        assert run.postcheck is not None and run.postcheck.complete
        assert run.postcheck.parsed.observation.representations == (
            run.generation.representations)
        assert installation.lexical_choice.choice_kind == (
            "LEXICAL_REALIZATION_CHOICE")
        assert installation.lexical_choice.selected_object == (
            installation.variant.template.connector)
        assert installation.lexical_choice.exact_uses == ()
        lexical_alternatives = tuple(
            build_grounded_answer_lexical_choice(variant, candidate)
            for variant in installation.compilation.variants
        )
        assert len(lexical_alternatives) == 3
        assert len({
            item.condition.context for item in lexical_alternatives}) == 1
        selected_structure_alternatives = tuple(
            item for item, variant in zip(
                lexical_alternatives,
                installation.compilation.variants,
                strict=True,
            )
            if variant.option.structure_id
            == installation.variant.option.structure_id
        )
        assert len(selected_structure_alternatives) == 2
        assert len({
            item.competition_key
            for item in selected_structure_alternatives}) == 1
        assert len({
            item.selected_object
            for item in selected_structure_alternatives}) == 2
        assert lexical_use.run == run
        assert lexical_use.adoptions == run.generation.surface.adoptions
        assert lexical_use.use.scope == candidate.scope
        assert lexical_use.use.selection_key.components != (
            selected_pattern.pattern_id,)
        assert lexical_use.choice_after.exact_uses == (lexical_use.use,)
        assert lexical_use.choice_after.typed_outcomes == ()
        assert ledger.records == (lexical_use,)
        with pytest.raises(
                GroundedAnswerLexicalUseError,
                match="不得重复登记"):
            ledger.adopt(run)
        assert layer_uses.content.choice_after.exact_uses == (
            layer_uses.content.use,)
        assert layer_uses.task.choice_after.exact_uses == (
            layer_uses.task.use,)
        assert layer_uses.content.choice_before.selected_object == (
            candidate.proposition.template)
        assert layer_uses.task.choice_before.selected_object == run.status
        assert layer_uses.content.choice_before.condition.context == (
            installation.lexical_choice.condition.context)
        assert layer_uses.task.choice_before.condition.context == (
            installation.lexical_choice.condition.context)
        assert len({
            lexical_use.use.use_key,
            layer_uses.content.use.use_key,
            layer_uses.task.use.use_key,
            structure_use.use.use_key,
        }) == 4
        assert structure_use.choice_before.selected_object == (
            installation.structure_selection.selected.structure)
        assert structure_use.choice_before.condition.context == (
            installation.lexical_choice.condition.context)
        assert structure_use.choice_after.exact_uses == (structure_use.use,)
        assert structure_ledger.records == (structure_use,)
        with pytest.raises(
                GroundedAnswerStructureChoiceUseError,
                match="不得重复登记"):
            structure_ledger.adopt(run)
        assert layer_ledger.records == (layer_uses,)
        with pytest.raises(
                GroundedAnswerLayerChoiceUseError,
                match="不得重复登记"):
            layer_ledger.adopt(run)
        assert installation.alias is alias_factory.fixture.runtime
        assert installation.order.evidence_count == (
            len(installation.variant.order_requirements)
            * len(installation.variant.option.support_teacher_keys)
        )
        assert renderer.text(run.generation.rendered) == (
            "档案显示，云岭站西门于2026年启用。")
    finally:
        if alias_factory.fixture is not None:
            alias_factory.fixture.close()
        backend.close()


def _rehearse_answer_requirement(requirement, offset):
    """用真实结构/来源 verifier 运行一项 catalog ANSWER rehearsal。"""
    course = read_generation_generalization_executable_train_course(
        _CASE_SAMPLE, _GROUNDED_SAMPLE)
    case = course.case_for_requirement(requirement)
    episode = course.episode_for(case)
    model, _report = learn_grounded_answer_surface_model(
        compile_grounded_answer_training_records(_GROUNDED_SAMPLE))
    branch = language_branch_identity((_BASE, 100, offset))
    planning_build = compile_grounded_answer_planning(episode, branch)
    planning = planning_build.planning
    candidate = planning.candidates[0]
    selection, selector, _content_protocol = _selection(planning)
    surface_protocol = _surface_protocol(_BASE + 110 + offset)
    family = (_BASE, 120, offset)
    target = GroundedAnswerConnectorTarget(
        candidate.proposition, branch, family)
    compilation = compile_grounded_answer_connectors(
        model, episode.question, target, surface_protocol)
    selected = compilation.variants[0]
    backend = DictBackend()
    alias_factory = _AliasFactory(branch)
    try:
        graphs = _graphs(backend)
        renderer_identity = minimal_instruction_identity(
            (_BASE, 130, offset))
        renderer = UnicodeRepresentationRenderer(family, renderer_identity)
        query_kind = minimal_instruction_identity((_BASE, 140 + offset, 1))
        route = minimal_instruction_identity((_BASE, 140 + offset, 2))
        components = GroundedAnswerRunLocalComponents(
            selector,
            _plan_protocol(_BASE + 150 + offset),
            GenerationStructureLayerProtocol(*tuple(
                minimal_instruction_identity(
                    (_BASE, 160 + offset, index))
                for index in range(1, 4)
            )),
            alias_factory,
            renderer,
            renderer_identity,
            _postcheck_protocol(),
            GroundedAnswerStructureVerifier(
                minimal_instruction_identity((_BASE, 170 + offset, 1)),
                minimal_instruction_identity((_BASE, 170 + offset, 2)),
            ),
            GroundedAnswerEvidenceSourceVerifier(
                minimal_instruction_identity((_BASE, 180 + offset, 1)),
                minimal_instruction_identity((_BASE, 180 + offset, 2)),
            ),
            QuestionAnswerProtocol(*tuple(
                minimal_instruction_identity(
                    (_BASE, 190 + offset, index))
                for index in range(1, 4)
            )),
            EvidenceQuestionPostcheckMapper(
                (_BASE, 200 + offset, 1),
                citation_required=True,
                trust_required=True,
            ),
        )
        installation = GroundedAnswerRunLocalFactory(
            surface_protocol, graphs.lifecycle, components).build(
                GroundedAnswerRunLocalBuild(
                    model,
                    episode.question,
                    target,
                    planning,
                    candidate,
                    selected.option.structure_id,
                    selected.option.pattern_id,
                    GroundedAnswerParserProtocol(
                        *tuple(minimal_instruction_identity(
                            (_BASE, 210 + offset, index))
                               for index in range(1, 6)),
                        selection.stance,
                    ),
                    query_kind,
                    route,
                    minimal_instruction_identity(
                        (_BASE, 220 + offset, 1)),
                    (_BASE, 220 + offset, 2),
                ))
        request = QuestionRequest(
            query_kind,
            minimal_instruction_identity((_BASE, 230 + offset, 1)),
            planning.goal.goal_kind,
            planning.goal.proposition,
            planning.goal.required,
            planning.goal.scope,
            planning.goal.scope,
            (_BASE, 230 + offset, 2),
            branch,
        )
        item, run = rehearse_grounded_answer_case(
            course, requirement, planning_build, installation, request)
        return course, planning_build, item, run
    finally:
        if alias_factory.fixture is not None:
            alias_factory.fixture.close()
        backend.close()


def test_answer_catalog_rehearses_readback_and_legal_composition():
    """单命题 ANSWER 的 readback 与 structure choice 使用独立 actual routes。"""
    course, readback_planning, readback, readback_run = (
        _rehearse_answer_requirement(
            "INDEPENDENT_UNDERSTANDING_READBACK", 1))
    same_course, legal_planning, legal, legal_run = (
        _rehearse_answer_requirement("LEGAL_OBJECT_COMPOSITION", 2))
    assert same_course == course

    partial = GenerationGeneralizationTrainRehearsal(
        course, (readback, legal))
    assert partial.complete == 0
    assert readback.passed == 1
    assert legal.passed == 1
    assert readback.choice.choice_kind == "LEXICAL_REALIZATION_CHOICE"
    assert legal.choice.choice_kind == "PROPOSITION_STRUCTURE_CHOICE"
    assert readback.verification.input_key != legal.verification.input_key
    assert (readback.verification.result.dimension,
            readback.verification.result.verifier) != (
                legal.verification.result.dimension,
                legal.verification.result.verifier)
    assert readback_run.complete and legal_run.complete
    assert readback_planning.aggregate_source in readback.choice.forming_sources
    assert legal_planning.aggregate_source in legal.choice.forming_sources
    for run in (readback_run, legal_run):
        source_result = next(
            item for item in run.postcheck.report.results
            if item.dimension == run.postcheck.protocol.source_dimension)
        structure_result = next(
            item for item in run.postcheck.report.results
            if item.dimension == run.postcheck.protocol.structure_dimension)
        assert source_result.verdict == VERDICT_SUPPORT
        assert structure_result.verdict == VERDICT_SUPPORT

    observation = readback.postcheck.parsed.observation
    assert observation is not None
    broken_postcheck = replace(
        readback.postcheck,
        parsed=replace(
            readback.postcheck.parsed,
            observation=replace(observation, propositions=()),
        ),
    )
    protocol = default_answer_verification_protocol()
    broken_report = run_generation_generalization_answer_verification(
        protocol,
        GenerationGeneralizationAnswerVerificationInput(
            "INDEPENDENT_UNDERSTANDING_READBACK",
            readback.source_episode,
            readback_planning,
            readback.choice,
            readback.use,
            readback.execution,
            readback.parse_request,
            broken_postcheck,
        ),
    )
    assert broken_report.results[0].verdict == VERDICT_REFUTE
