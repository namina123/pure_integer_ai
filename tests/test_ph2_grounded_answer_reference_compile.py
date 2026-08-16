from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructureLayerProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.experiments.ph2_grounded_answer_runtime_factory import (
    GroundedAnswerRunLocalComponents,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_runtime_factory import (
    GroundedAnswerReferenceRunLocalBuild,
    GroundedAnswerReferenceRunLocalFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_choice import (
    build_grounded_answer_reference_selection,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_episode_use import (
    GroundedAnswerReferenceEpisodeAdoptionLedger,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_verification import (
    build_grounded_answer_reference_verifier_protocol,
    run_grounded_answer_reference_gg02,
)
from pure_integer_ai.experiments.ph2_generation_choice_assessment_consumer import (
    GenerationChoiceAssessmentConsumer,
    GenerationChoiceAssessmentConsumerPolicy,
    assessment_input_stance,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceCandidateMapper,
)
from pure_integer_ai.experiments.ph2_generation_choice_outcome_bridge import (
    build_assessment_inputs,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserProtocol,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.question_answer_runtime import (
    EvidenceQuestionPostcheckMapper,
    QuestionAnswerProtocol,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_NOT_APPLICABLE,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    REFERENCE_STRATEGIES,
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    GroundedAnswerClaimCandidateBinding,
    GroundedAnswerReferenceCompileRequest,
    compile_grounded_answer_reference_connector,
)

from tests.test_g02_generation_structure_plan import (
    _request,
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
from tests.test_g02_generation_structure_plan import _plan_protocol
from tests.test_d02_gg01_generation_choice_contract import _candidate_protocol
from tests.test_r00_relation_closure import _projection_protocol, _verifier
from tests.test_s07_structure_order import _graphs


SAMPLE_PATH = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")
_BASE = 20970


class _ReferenceAliasFactory:
    """按 compilation 建立 direct realization 和唯一 reference fact。"""

    def __init__(self, branch) -> None:
        self.branch = branch
        self.fixture = None

    def build(self, compilation):
        """建立一次独占 alias owner，不读取 teacher surface。"""
        if self.fixture is not None:
            raise RuntimeError("reference alias factory 不得重复 build")
        aliases = tuple(
            alias
            for sentence in compilation.sentences
            for alias in sentence.aliases
        )
        self.fixture = _alias_fixture(
            self.branch,
            tuple((item.filler, item.representation) for item in aliases),
            ((
                compilation.reference_origin,
                compilation.claims[0].candidate.proposition.template,
            ),),
        )
        return self.fixture.runtime


def _reference_selection(selected_strategy="ANTECEDENT_REFERENCE"):
    """建立第五条课程的完整双策略竞争与先行选择。"""
    episode = read_grounded_answer_episodes(SAMPLE_PATH)[-1]
    request, _unused = _request(count=2)
    branch = language_branch_identity((_BASE, 1))
    planning = GenerationPlanningRequest(
        replace(request.goal, target_branch=branch),
        request.candidates,
    )
    claims = tuple(
        GroundedAnswerClaimCandidateBinding(proposition_id, candidate)
        for proposition_id, candidate in zip(
            episode.question.answer_plan.ordered_claim_ids,
            planning.candidates,
            strict=True,
        )
    )
    compilations = tuple(
        compile_grounded_answer_reference_connector(
            GroundedAnswerReferenceCompileRequest(
                episode,
                planning,
                claims,
                branch,
                (_BASE, 2),
                strategy,
                ((_BASE, 3),),
            ),
            _surface_protocol(_BASE + 1),
        )
        for strategy in REFERENCE_STRATEGIES
    )
    selection = build_grounded_answer_reference_selection(
        compilations,
        selected_strategy,
        (_BASE, 4, REFERENCE_STRATEGIES.index(selected_strategy) + 1),
    )
    return episode, planning, branch, selection


def test_reference_course_compiles_to_two_sentences_and_one_anaphora():
    """第五条 grounded 课程形成逐 Proposition 句和前序 antecedent。"""
    _episode, planning, _branch, reference_selection = _reference_selection()
    compilation = reference_selection.compilation
    selection, _unused_first, _unused_second = _selection(planning)

    plan = compilation.connector.structure_planner().plan(selection)

    assert len(compilation.sentences) == 2
    assert len(compilation.connector.registry.templates) == 2
    assert len(plan.syntax.sentences) == 2
    assert len(plan.syntax.anaphora) == 1
    assert tuple(
        sentence.instance.candidate_key
        for sentence in plan.syntax.sentences) == tuple(
            candidate.stable_key() for candidate in planning.candidates)
    requirement = plan.syntax.anaphora[0]
    assert requirement.address == plan.syntax.sentences[1].address
    assert requirement.slot == compilation.reference_slot
    assert requirement.antecedent_candidate_key == (
        planning.candidates[0].stable_key())
    assert compilation.connector.anaphora_declarations is not None
    explicit = _reference_selection("EXPLICIT_REPETITION")[3]
    assert tuple(
        item.strategy for item in reference_selection.options
    ) == REFERENCE_STRATEGIES
    assert (reference_selection.choice.competition_key
            == explicit.choice.competition_key)
    assert (reference_selection.choice.condition
            == explicit.choice.condition)
    assert (reference_selection.choice.selected_object
            != explicit.choice.selected_object)


def test_reference_compilation_runs_two_sentences_through_g04():
    """双句 compilation 经真实 G-02/G-03 输出，并由 parser 恢复两命题。"""
    _episode, planning, branch, reference_selection = _reference_selection()
    compilation = reference_selection.compilation
    selection, selector, _content_protocol = _selection(planning)
    backend = DictBackend()
    alias_factory = _ReferenceAliasFactory(branch)
    try:
        graphs = _graphs(backend)
        renderer_identity = minimal_instruction_identity((_BASE + 20, 1))
        renderer = UnicodeRepresentationRenderer(
            (_BASE, 2), renderer_identity)
        parser_protocol = GroundedAnswerParserProtocol(
            *tuple(minimal_instruction_identity((_BASE + 21, index))
                   for index in range(1, 6)),
            selection.stance,
        )
        query_kind = minimal_instruction_identity((_BASE + 22, 1))
        route = minimal_instruction_identity((_BASE + 22, 2))
        components = GroundedAnswerRunLocalComponents(
            selector,
            _plan_protocol(_BASE + 23),
            GenerationStructureLayerProtocol(*tuple(
                minimal_instruction_identity((_BASE + 24, index))
                for index in range(1, 4)
            )),
            alias_factory,
            renderer,
            renderer_identity,
            _postcheck_protocol(),
            _StaticVerifier(VERDICT_SUPPORT, 1),
            _StaticVerifier(VERDICT_SUPPORT, 2),
            QuestionAnswerProtocol(*tuple(
                minimal_instruction_identity((_BASE + 25, index))
                for index in range(1, 4)
            )),
            EvidenceQuestionPostcheckMapper(
                (_BASE + 26, 1),
                citation_required=True,
                trust_required=True,
            ),
        )
        factory = GroundedAnswerReferenceRunLocalFactory(
            graphs.lifecycle, components)
        installation = factory.build(
            GroundedAnswerReferenceRunLocalBuild(
                compilation,
                reference_selection,
                parser_protocol,
                query_kind,
                route,
                minimal_instruction_identity((_BASE + 27, 1)),
                (_BASE + 27, 2),
            ))
        candidate = planning.candidates[0]
        request = QuestionRequest(
            query_kind,
            minimal_instruction_identity((_BASE + 22, 3)),
            planning.goal.goal_kind,
            planning.goal.proposition,
            planning.goal.required,
            candidate.scope,
            candidate.scope,
            (_BASE + 22, 4),
            branch,
            tuple(item.proposition for item in planning.candidates),
        )

        run = installation.runtime.run(request)

        assert run.complete
        assert run.generation is not None
        assert run.generation.surface is not None
        syntax = run.generation.surface.preview.request.structure.syntax
        assert len(syntax.sentences) == 2
        assert len(syntax.anaphora) == 1
        assert len(run.generation.surface.adoptions) == 6
        assert run.postcheck is not None and run.postcheck.complete
        assert len(run.postcheck.parsed.observation.propositions) == 2
        assert installation.order.evidence_count == 3
        assert renderer.text(run.generation.rendered) == (
            "北川站东门于2024年启用。前述启用事项已登记入档。")
        bundle = GroundedAnswerReferenceEpisodeAdoptionLedger(
            installation).adopt(run)
        record = bundle.reference
        assert record.choice_after.exact_uses == (record.use,)
        assert record.reference_adoption is not None
        assert record.recovery.antecedent == (
            planning.candidates[0].proposition.template)
        assert len({item.use_key for item in bundle.uses}) == 5
    finally:
        if alias_factory.fixture is not None:
            alias_factory.fixture.close()
        backend.close()


def _run_reference_strategy(selected_strategy):
    """运行一个 selected strategy 并返回 actual reference exact Use。"""
    _episode, planning, branch, reference_selection = _reference_selection(
        selected_strategy)
    compilation = reference_selection.compilation
    selection, selector, _content_protocol = _selection(planning)
    backend = DictBackend()
    alias_factory = _ReferenceAliasFactory(branch)
    try:
        graphs = _graphs(backend)
        renderer_identity = minimal_instruction_identity((_BASE + 30, 1))
        renderer = UnicodeRepresentationRenderer(
            (_BASE, 2), renderer_identity)
        parser_protocol = GroundedAnswerParserProtocol(
            *tuple(minimal_instruction_identity((_BASE + 31, index))
                   for index in range(1, 6)),
            selection.stance,
        )
        query_kind = minimal_instruction_identity((_BASE + 32, 1))
        route = minimal_instruction_identity((_BASE + 32, 2))
        components = GroundedAnswerRunLocalComponents(
            selector,
            _plan_protocol(_BASE + 33),
            GenerationStructureLayerProtocol(*tuple(
                minimal_instruction_identity((_BASE + 34, index))
                for index in range(1, 4)
            )),
            alias_factory,
            renderer,
            renderer_identity,
            _postcheck_protocol(),
            _StaticVerifier(VERDICT_SUPPORT, 1),
            _StaticVerifier(VERDICT_SUPPORT, 2),
            QuestionAnswerProtocol(*tuple(
                minimal_instruction_identity((_BASE + 35, index))
                for index in range(1, 4)
            )),
            EvidenceQuestionPostcheckMapper(
                (_BASE + 36, 1),
                citation_required=True,
                trust_required=True,
            ),
        )
        installation = GroundedAnswerReferenceRunLocalFactory(
            graphs.lifecycle, components).build(
                GroundedAnswerReferenceRunLocalBuild(
                    compilation,
                    reference_selection,
                    parser_protocol,
                    query_kind,
                    route,
                    minimal_instruction_identity((_BASE + 37, 1)),
                    (_BASE + 37, 2),
                ))
        candidate = planning.candidates[0]
        request = QuestionRequest(
            query_kind,
            minimal_instruction_identity((_BASE + 32, 3)),
            planning.goal.goal_kind,
            planning.goal.proposition,
            planning.goal.required,
            candidate.scope,
            candidate.scope,
            (_BASE + 32, 4),
            branch,
            tuple(item.proposition for item in planning.candidates),
        )
        run = installation.runtime.run(request)
        record = GroundedAnswerReferenceEpisodeAdoptionLedger(
            installation).adopt(run)
        gg02 = run_grounded_answer_reference_gg02(
            build_grounded_answer_reference_verifier_protocol(
                (_BASE + 38, 1)),
            installation,
            run,
            record,
        )
        return (
            reference_selection,
            record,
            renderer.text(run.generation.rendered),
            len(run.generation.surface.adoptions),
            gg02,
        )
    finally:
        if alias_factory.fixture is not None:
            alias_factory.fixture.close()
        backend.close()


def test_reference_strategies_form_distinct_actual_uses():
    """同一竞争集的两种策略均真实执行，并形成互异 exact Use。"""
    antecedent = _run_reference_strategy("ANTECEDENT_REFERENCE")
    explicit = _run_reference_strategy("EXPLICIT_REPETITION")

    assert (antecedent[0].choice.competition_key
            == explicit[0].choice.competition_key)
    assert antecedent[1].reference.reference_adoption is not None
    assert explicit[1].reference.reference_adoption is None
    assert (antecedent[1].reference.use.use_key
            != explicit[1].reference.use.use_key)
    assert antecedent[1].reference.use.use_key.components[0] == 21010
    assert explicit[1].reference.use.use_key.components[0] == 21010
    assert len({item.use_key for item in antecedent[1].uses}) == 5
    assert len({item.use_key for item in explicit[1].uses}) == 5
    assert antecedent[2] == (
        "北川站东门于2024年启用。前述启用事项已登记入档。")
    assert explicit[2] == (
        "北川站东门于2024年启用。北川站东门的启用事项已登记入档。")
    assert antecedent[3] == 6
    assert explicit[3] == 5
    for completed in (antecedent[4], explicit[4]):
        assert len(completed.verification.report.results) == 10
        assert len(completed.attribution.choices) == 5
        assert len(completed.outcome.outcomes) == 10
        assert completed.outcome.host_learning_write_count == 0
        assert completed.outcome.teacher_call_count == 0
        assert completed.outcome.assessment_consumer_status == (
            "REQUIRED_NOT_CONNECTED")
        assert all(
            item.verdict == VERDICT_SUPPORT
            for item in completed.verification.report.results
            if item.applicability != APPLICABILITY_NOT_APPLICABLE)
    assert len(antecedent[4].verification.claims) == 10
    assert len(explicit[4].verification.claims) == 9
    explicit_route = explicit[4].verification.protocol.by_name()[
        "REFERENCE_UNIQUE_RESOLUTION"].route
    explicit_unique = next(
        item for item in explicit[4].verification.report.results
        if (item.dimension == explicit_route.dimension
            and item.verifier == explicit_route.verifier))
    assert explicit_unique.applicability == APPLICABILITY_NOT_APPLICABLE
    assert explicit_unique.claim_keys == ()


def test_reference_gg02_outcome_updates_h05_once_and_replays_without_writes():
    """真实五层 outcome 只执行一次 assessment，关闭层 prepare 保持零写。"""
    layered = _run_reference_strategy("ANTECEDENT_REFERENCE")[4].outcome
    explicit_layered = _run_reference_strategy("EXPLICIT_REPETITION")[4].outcome
    support_input = build_assessment_inputs(layered).inputs[0]
    support_outcome = support_input.outcomes[0]
    assert assessment_input_stance(support_input) == EVIDENCE_SUPPORT
    assert assessment_input_stance(replace(
        support_input,
        outcomes=(replace(support_outcome, verdict=VERDICT_REFUTE),),
    )) == EVIDENCE_REFUTE
    assert assessment_input_stance(replace(
        support_input,
        outcomes=(replace(support_outcome, verdict=VERDICT_UNKNOWN),),
    )) == EVIDENCE_UNKNOWN
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        aggregate = SourceRef(
            _BASE + 50, 1, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
        learning = CandidateLearningRuntime(
            EvidenceCandidateEngine(EvidenceCandidateProtocol(
                (_BASE + 50, 2),
                (_BASE + 50, 3),
                aggregate,
                document_scope(aggregate),
                1,
            )),
            CandidateProjectionGraph(
                context.graph_ontology, _projection_protocol()),
            _verifier(),
            CandidateProjectionMetadata(SOURCE_BARE_TEXT, EPI_STRUCTURED),
        )
        mapper = GenerationChoiceCandidateMapper(
            _candidate_protocol(_BASE + 50))
        verifier_source = SourceRef(
            _BASE + 50, 4, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
        consumer = GenerationChoiceAssessmentConsumer(
            mapper,
            learning,
            GenerationChoiceAssessmentConsumerPolicy(
                verifier_source, (_BASE + 50, 5)),
        )
        disabled = GenerationChoiceAssessmentConsumer(
            mapper,
            learning,
            GenerationChoiceAssessmentConsumerPolicy(
                verifier_source,
                (_BASE + 50, 6),
                ("DISCOURSE_REFERENCE_CHOICE",),
            ),
        )
        before_prepare_backend = backend.snapshot()
        before_prepare_state = learning.state_key()

        prepared = disabled.prepare(layered)

        assert len(prepared) == 4
        assert all(
            item.assessment.choice_kind != "DISCOURSE_REFERENCE_CHOICE"
            for item in prepared)
        assert backend.snapshot() == before_prepare_backend
        assert learning.state_key() == before_prepare_state

        first = consumer.apply(layered)

        assert len(first.records) == 5
        assert first.candidate_registrations == 5
        assert first.assessment_updates_executed == 5
        assert first.replayed_updates == 0
        assert first.teacher_call_count == 0
        assert first.support_records == 5
        assert first.refute_records == 0
        assert first.unknown_records == 0
        assert all(item.candidate_registered == 1 for item in first.records)
        assert all(
            item.learning.verification.stance == EVIDENCE_SUPPORT
            for item in first.records)
        assert all(item.learning.projection is not None for item in first.records)
        assert len({item.stable_key() for item in first.records}) == 5

        explicit = consumer.apply(explicit_layered)

        assert len(explicit.records) == 5
        assert explicit.candidate_registrations == 1
        assert explicit.assessment_updates_executed == 5
        assert explicit.replayed_updates == 0
        assert explicit.support_records == 5
        assert explicit.teacher_call_count == 0
        after_first_backend = backend.snapshot()
        after_first_state = learning.state_key()

        replay = consumer.apply(layered)

        assert replay.records == first.records
        assert replay.candidate_registrations == 0
        assert replay.assessment_updates_executed == 0
        assert replay.replayed_updates == 5
        assert replay.teacher_call_count == 0
        assert backend.snapshot() == after_first_backend
        assert learning.state_key() == after_first_state
    finally:
        backend.close()
