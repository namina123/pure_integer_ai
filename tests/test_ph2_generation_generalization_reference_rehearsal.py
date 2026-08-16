"""E-01 双命题 reference rehearsal 与最终六路 TRAIN 组合专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructureLayerProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationSourceCheckRequest,
    GenerationStructureCheckRequest,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_executable_train_course import (
    read_generation_generalization_executable_train_course,
)
from pure_integer_ai.experiments.ph2_generation_generalization_executable_train_rehearsal import (
    GenerationGeneralizationTrainRehearsal,
    rehearse_grounded_answer_reference_case,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    REFERENCE_STRATEGIES,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserProtocol,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_choice import (
    build_grounded_answer_reference_selection,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    GroundedAnswerClaimCandidateBinding,
    GroundedAnswerReferenceCompileRequest,
    compile_grounded_answer_reference_connector,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_postcheck import (
    GroundedAnswerReferenceEvidenceSourceVerifier,
    GroundedAnswerReferenceStructureVerifier,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_runtime_factory import (
    GroundedAnswerReferenceRunLocalBuild,
    GroundedAnswerReferenceRunLocalFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_runtime_factory import (
    GroundedAnswerRunLocalComponents,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    compile_grounded_answer_reference_planning,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    EvidenceQuestionPostcheckMapper,
    QuestionAnswerProtocol,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
)
from pure_integer_ai.storage.backend import DictBackend

from tests.test_g02_generation_structure_plan import _plan_protocol, _selection
from tests.test_g03_generation_surface import _surface_protocol
from tests.test_g04_generation_postcheck import _protocol as _postcheck_protocol
from tests.test_ph2_grounded_answer_connector_runtime import (
    _rehearse_answer_requirement,
)
from tests.test_ph2_grounded_answer_reference_compile import (
    _ReferenceAliasFactory,
)
from tests.test_ph2_grounded_answer_response_act_runtime import (
    _rehearse_response_act,
)
from tests.test_s07_structure_order import _graphs


_BASE = 21070
_GROUNDED_SAMPLE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")
_CASE_SAMPLE = Path(
    "data/ph2/generation_generalization_executable_train_case_v1.jsonl.sample")


def _rehearse_reference_requirement(requirement: str, offset: int):
    """从 catalog episode 建立双命题 planning 并运行一个 reference case。"""
    course = read_generation_generalization_executable_train_course(
        _CASE_SAMPLE, _GROUNDED_SAMPLE)
    case = course.case_for_requirement(requirement)
    episode = course.episode_for(case)
    branch = language_branch_identity((_BASE, 1, offset))
    planning_build = compile_grounded_answer_reference_planning(
        episode, branch)
    planning = planning_build.planning
    claims = tuple(
        GroundedAnswerClaimCandidateBinding(
            proposition_id,
            planning_build.candidate_for(proposition_id),
        )
        for proposition_id in episode.reference_course.ordered_proposition_ids
    )
    surface_protocol = _surface_protocol(_BASE + 10 + offset)
    family = (_BASE, 20, offset)
    compilations = tuple(
        compile_grounded_answer_reference_connector(
            GroundedAnswerReferenceCompileRequest(
                episode,
                planning,
                claims,
                branch,
                family,
                strategy,
                ((_BASE, 30, offset),),
            ),
            surface_protocol,
        )
        for strategy in REFERENCE_STRATEGIES
    )
    reference_selection = build_grounded_answer_reference_selection(
        compilations,
        case.reference_strategy,
        (_BASE, 40, offset),
    )
    content_selection, selector, _content_protocol = _selection(planning)
    backend = DictBackend()
    alias_factory = _ReferenceAliasFactory(branch)
    try:
        graphs = _graphs(backend)
        renderer_identity = minimal_instruction_identity(
            (_BASE, 50, offset))
        renderer = UnicodeRepresentationRenderer(family, renderer_identity)
        structure_verifier = GroundedAnswerReferenceStructureVerifier(
            minimal_instruction_identity((_BASE, 60, offset, 1)),
            minimal_instruction_identity((_BASE, 60, offset, 2)),
        )
        source_verifier = GroundedAnswerReferenceEvidenceSourceVerifier(
            minimal_instruction_identity((_BASE, 70, offset, 1)),
            minimal_instruction_identity((_BASE, 70, offset, 2)),
        )
        query_kind = minimal_instruction_identity((_BASE, 80, offset, 1))
        route = minimal_instruction_identity((_BASE, 80, offset, 2))
        components = GroundedAnswerRunLocalComponents(
            selector,
            _plan_protocol(_BASE + 90 + offset),
            GenerationStructureLayerProtocol(*tuple(
                minimal_instruction_identity((_BASE, 100, offset, index))
                for index in range(1, 4)
            )),
            alias_factory,
            renderer,
            renderer_identity,
            _postcheck_protocol(),
            structure_verifier,
            source_verifier,
            QuestionAnswerProtocol(*tuple(
                minimal_instruction_identity((_BASE, 110, offset, index))
                for index in range(1, 4)
            )),
            EvidenceQuestionPostcheckMapper(
                (_BASE, 120, offset),
                citation_required=True,
                trust_required=True,
            ),
        )
        installation = GroundedAnswerReferenceRunLocalFactory(
            graphs.lifecycle, components).build(
                GroundedAnswerReferenceRunLocalBuild(
                    reference_selection.compilation,
                    reference_selection,
                    GroundedAnswerParserProtocol(
                        *tuple(minimal_instruction_identity(
                            (_BASE, 130, offset, index))
                               for index in range(1, 6)),
                        content_selection.stance,
                    ),
                    query_kind,
                    route,
                    minimal_instruction_identity((_BASE, 140, offset, 1)),
                    (_BASE, 140, offset, 2),
                ))
        request = QuestionRequest(
            query_kind,
            minimal_instruction_identity((_BASE, 150, offset, 1)),
            planning.goal.goal_kind,
            planning.goal.proposition,
            planning.goal.required,
            planning.goal.scope,
            planning.goal.scope,
            (_BASE, 150, offset, 2),
            branch,
            tuple(item.proposition for item in planning.candidates),
        )
        item, run = rehearse_grounded_answer_reference_case(
            course, requirement, planning_build, installation, request)
        return (
            course,
            planning_build,
            item,
            run,
            structure_verifier,
            source_verifier,
        )
    finally:
        if alias_factory.fixture is not None:
            alias_factory.fixture.close()
        backend.close()


def test_reference_catalog_rehearses_two_routes_and_completes_train_course():
    """两种 reference 策略真实执行，六项按 catalog 顺序形成 6/6。"""
    (course, antecedent_planning, antecedent, antecedent_run,
     structure_verifier, source_verifier) = (
        _rehearse_reference_requirement("ADDRESSEE_RECOVERABILITY", 1))
    (same_course, explicit_planning, explicit, explicit_run,
     _unused_structure, _unused_source) = (
        _rehearse_reference_requirement("STRUCTURE_SLOT_ORDER", 2))
    assert same_course == course

    _, _, communicative, _ = _rehearse_response_act(
        "COMMUNICATIVE_TASK", "CLARIFY", 11)
    _, _, readback, _ = _rehearse_answer_requirement(
        "INDEPENDENT_UNDERSTANDING_READBACK", 11)
    _, _, legal, _ = _rehearse_answer_requirement(
        "LEGAL_OBJECT_COMPOSITION", 12)
    _, _, source, _ = _rehearse_response_act(
        "SOURCE_UNCERTAINTY_CITATION", "CONFLICT", 12)
    rehearsal = GenerationGeneralizationTrainRehearsal(
        course,
        (antecedent, communicative, readback, legal, source, explicit),
    )

    assert tuple(item.case.requirement for item in rehearsal.items) == (
        INDEPENDENT_VERIFIER_REQUIREMENTS)
    assert rehearsal.complete == 1
    assert all(item.passed for item in rehearsal.items)
    assert antecedent.choice.choice_kind == "DISCOURSE_REFERENCE_CHOICE"
    assert explicit.choice.choice_kind == "PROPOSITION_STRUCTURE_CHOICE"
    assert antecedent.verification.input_key != explicit.verification.input_key
    assert (antecedent.verification.result.dimension,
            antecedent.verification.result.verifier) != (
                explicit.verification.result.dimension,
                explicit.verification.result.verifier)
    assert tuple(
        binding.proposition_id
        for binding in antecedent_planning.candidate_bindings) == (
            "p-registration", "p-year")
    assert tuple(
        item.proposition.template
        for item in antecedent_planning.planning.candidates
    ) == tuple(
        antecedent_planning.candidate_for(item).proposition.template
        for item in ("p-year", "p-registration")
    )
    assert explicit_planning.planning.candidates != ()

    for run in (antecedent_run, explicit_run):
        assert run.complete
        structure_result = next(
            item for item in run.postcheck.report.results
            if item.dimension == run.postcheck.protocol.structure_dimension)
        source_result = next(
            item for item in run.postcheck.report.results
            if item.dimension == run.postcheck.protocol.source_dimension)
        assert structure_result.verdict == VERDICT_SUPPORT
        assert source_result.verdict == VERDICT_SUPPORT

    observation = antecedent_run.postcheck.parsed.observation
    assert observation is not None
    assert structure_verifier.verify(GenerationStructureCheckRequest(
        antecedent_run.postcheck.request,
        replace(observation, structure_payload=(_BASE, 999)),
    )).verdict == VERDICT_REFUTE
    assert source_verifier.verify(GenerationSourceCheckRequest(
        antecedent_run.postcheck.request,
        replace(observation, cited_sources=()),
        antecedent_run.postcheck.request.source_requirements,
        observation.propositions,
    )).verdict == VERDICT_REFUTE
