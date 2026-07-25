"""L-05B2B connector 理论候选生命周期、恢复和故障隔离对抗。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateHistoryUnavailableError,
    CandidateLearningRuntime,
)
from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionSelector,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateEngine,
    EVIDENCE_REFUTE,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.identity import (
    OwnerScope,
    VISIBILITY_SESSION,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    DiscourseDependency,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceAttribution,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationPostcheckRequest,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    proposition_identity,
    role_identity,
    semantic_source,
)
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderGraph,
    StructureOrderGraphPredicates,
)
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    StructureOrderLifecycleGraph,
    StructureOrderLifecycleProtocol,
)
from pure_integer_ai.cognition.shared.memory_hypothesis import (
    MemoryHypothesisEventSink,
)
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingHypothesisEventSink,
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureConsumer,
)
from pure_integer_ai.cognition.shared.relation_use import (
    RelationUseGraph,
    RelationUseGraphProtocol,
    RelationUseOwner,
    RelationUseWriteMetadata,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    document_scope,
    episode_scope,
    make_scope,
    query_scope,
)
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorDiscourseDeclaration,
    LanguageGenerationConnector,
    LanguageGenerationConnectorError,
    LanguageGenerationConnectorRegistry,
    LanguageConnectorSlotBinding,
    LanguageConnectorSurfaceDirective,
    StaticLanguageConnectorDiscourseDeclarations,
)
from pure_integer_ai.experiments.language_generation_connector_candidate import (
    CANDIDATE_PERSISTENCE_TRAINING,
    LanguageConnectorCandidateError,
    LanguageConnectorCandidateProtocol,
    LanguageConnectorCandidateRuntime,
)
from pure_integer_ai.experiments.language_generation_connector_factory import (
    ActiveLanguageConnectorFactory,
    LanguageConnectorProductionFactory,
    LanguageConnectorProductionRuntimeBinding,
    ScheduledLanguageConnectorFactory,
    TrialLanguageConnectorFactory,
)
from pure_integer_ai.experiments.language_generation_connector_scheduler import (
    ScheduledLanguageGenerationConnectorRegistry,
)
from pure_integer_ai.experiments.language_generation_connector_stage4 import (
    LanguageConnectorSignalRoute,
    LanguageConnectorStage4Policy,
    LanguageConnectorStage4Runtime,
)
from pure_integer_ai.experiments.language_generation_episode import (
    TypedLanguageEpisode,
)
from pure_integer_ai.experiments.language_generation_connector_graph import (
    LanguageConnectorGraphPredicates,
    LanguageGenerationConnectorGraph,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend as clone_storage_backend,
    clone_train_context,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.generation_production_runtime import (
    ProductionGenerationRequestDecision,
    ProductionGenerationRun,
)
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceRuntime,
)
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRelationRuntime,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureRuntime,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_NOT_APPLICABLE,
    APPLICABILITY_UNKNOWN,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
    VerificationReport,
    VerificationResult,
)
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run
from pure_integer_ai.storage.memory_event import MEMORY_EVENT_TABLE
from pure_integer_ai.storage.training_candidate_event import (
    TRAINING_CANDIDATE_EVENT_TABLE,
)

from tests.test_h05_language_candidate import (
    _projection_protocol,
    _reveal,
    _runtime,
    _source,
)
from tests.test_l05b2b_language_generation_connector_graph import _fixture
from tests.test_l05b2b_semantic_course_runtime import (
    _SemanticNoRequestFactory,
    _connector_generation_runtime,
)
from tests.test_g02_generation_structure_plan import _request
from tests.test_g04_generation_postcheck import (
    _ExecutionParser,
    _source_requirements,
)
from tests.test_l05b2b_language_generation_connector import (
    _execution_planner,
    _selection_with_role,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundRoleBinding
from tests.test_r00_relation_closure import (
    _candidate_runtime as _relation_candidate_runtime,
    _projection_protocol as _relation_projection_protocol,
    _r01_fixture,
    _relation_protocol,
    _semantic_graph,
)


_BASE = 16000


def _candidate_fixture(backend: DictBackend, *, variant: int):
    """在同一 ontology 上装配 connector 理论图与通用候选 owner。"""
    graphs, promotion, connector, predicate_identities, definition_graph = (
        _fixture(backend, variant=variant)
    )
    candidate_graph = CandidateProjectionGraph(
        graphs.context.graph_ontology,
        _projection_protocol(),
    )
    learning = _runtime(candidate_graph, kind_key=500 + variant)
    protocol = LanguageConnectorCandidateProtocol(
        concept_identity((_BASE, variant, 1)),
        concept_identity((_BASE, variant, 2)),
        concept_identity((_BASE, variant, 3)),
        (_BASE, variant, 4),
    )
    runtime = LanguageConnectorCandidateRuntime(
        definition_graph,
        learning,
        protocol,
    )
    return (
        graphs,
        promotion,
        connector.registry.templates[0],
        predicate_identities,
        candidate_graph,
        runtime,
    )


def _factory_fixture(backend: DictBackend, *, variant: int):
    """保留运行策略和 surface 协议，装配可供 active factory 使用的 owner。"""
    graphs, promotion, connector, predicate_identities, definition_graph = (
        _fixture(backend, variant=variant)
    )
    candidate_graph = CandidateProjectionGraph(
        graphs.context.graph_ontology,
        _projection_protocol(),
    )
    learning = _runtime(candidate_graph, kind_key=700 + variant)
    protocol = LanguageConnectorCandidateProtocol(
        concept_identity((_BASE + 20, variant, 1)),
        concept_identity((_BASE + 20, variant, 2)),
        concept_identity((_BASE + 20, variant, 3)),
        (_BASE + 20, variant, 4),
    )
    candidates = LanguageConnectorCandidateRuntime(
        definition_graph,
        learning,
        protocol,
    )
    return graphs, promotion, connector, predicate_identities, candidates


def _memory_candidate_fixture(backend: DictBackend, *, variant: int):
    """把普通 connector fixture 改接到当前 context 的 M-03 阅读事件层。"""
    graphs, promotion, template, predicate_identities, candidate_graph, base = (
        _candidate_fixture(backend, variant=variant)
    )
    sink = MemoryHypothesisEventSink(graphs.context.memory_read_events)
    base_learning = base.learning
    learning = CandidateLearningRuntime(
        EvidenceCandidateEngine(
            base_learning.engine.protocol,
            ledger=HypothesisLedger(sink),
        ),
        candidate_graph,
        base_learning.verifier,
        base_learning.metadata,
    )
    runtime = LanguageConnectorCandidateRuntime(
        base.definition_graph,
        learning,
        base.protocol,
    )
    return (
        graphs,
        promotion,
        template,
        predicate_identities,
        candidate_graph,
        runtime,
    )


def _memory_factory_fixture(backend: DictBackend, *, variant: int):
    """为 active factory fixture 安装独立的 M-03 connector owner。"""
    graphs, promotion, connector, predicate_identities, base = (
        _factory_fixture(backend, variant=variant)
    )
    sink = MemoryHypothesisEventSink(graphs.context.memory_read_events)
    base_learning = base.learning
    learning = CandidateLearningRuntime(
        EvidenceCandidateEngine(
            base_learning.engine.protocol,
            ledger=HypothesisLedger(sink),
        ),
        base_learning.graph,
        base_learning.verifier,
        base_learning.metadata,
    )
    candidates = LanguageConnectorCandidateRuntime(
        base.definition_graph,
        learning,
        base.protocol,
    )
    return graphs, promotion, connector, predicate_identities, candidates


def _training_candidate_fixture(backend: DictBackend, *, variant: int):
    """把 connector fixture 接到当前 Core 训练历史，且不创建 Memory 事件。"""
    graphs, promotion, template, predicate_identities, candidate_graph, base = (
        _candidate_fixture(backend, variant=variant)
    )
    base_learning = base.learning
    aggregate = base_learning.engine.protocol
    history_protocol = TrainingHypothesisHistoryProtocol(
        base.protocol.stable_key(),
        aggregate.hypothesis_kind_key,
        aggregate.aggregate_source,
        aggregate.aggregate_scope,
    )
    sink = TrainingHypothesisEventSink(
        graphs.context.training_candidate_history,
        history_protocol,
    )
    learning = CandidateLearningRuntime(
        EvidenceCandidateEngine(
            aggregate,
            ledger=HypothesisLedger(sink),
        ),
        candidate_graph,
        base_learning.verifier,
        base_learning.metadata,
    )
    runtime = LanguageConnectorCandidateRuntime(
        base.definition_graph,
        learning,
        base.protocol,
    )
    return (
        graphs,
        promotion,
        template,
        predicate_identities,
        candidate_graph,
        runtime,
    )


def _restore_memory_candidate(
        source_runtime: LanguageConnectorCandidateRuntime,
        source_graphs,
        target_context,
        predicate_identities,
        ) -> LanguageConnectorCandidateRuntime:
    """在目标 context 从 dump 后图与 M-03 event log 重建 connector owner。"""
    ontology = target_context.graph_ontology
    order_graph = StructureOrderGraph(
        ontology,
        StructureOrderGraphPredicates(*tuple(
            ontology.resolve(identity)
            for identity in source_graphs.order_predicate_identities
        )),
    )
    definition_graph = LanguageGenerationConnectorGraph(
        ontology,
        order_graph,
        LanguageConnectorGraphPredicates(*tuple(
            ontology.resolve(identity) for identity in predicate_identities
        )),
        source_runtime.definition_graph.value_protocol,
    )
    candidate_graph = CandidateProjectionGraph(
        ontology,
        source_runtime.learning.graph.protocol,
    )
    return source_runtime.restore_for_graphs(
        definition_graph,
        candidate_graph,
        target_context.memory_read_events,
    )


def _restore_training_candidate(
        source_runtime: LanguageConnectorCandidateRuntime,
        target_context,
        ) -> LanguageConnectorCandidateRuntime:
    """在目标 context 从 Core 图和训练历史重建 connector owner。"""
    ontology = target_context.graph_ontology
    source_ontology = source_runtime.definition_graph.ontology
    order_graph = StructureOrderGraph(
        ontology,
        StructureOrderGraphPredicates(*tuple(
            ontology.resolve(source_ontology.identity_of(item))
            for item in source_runtime.definition_graph.order_graph.predicates.refs()
        )),
    )
    definition_graph = LanguageGenerationConnectorGraph(
        ontology,
        order_graph,
        LanguageConnectorGraphPredicates(*tuple(
            ontology.resolve(source_ontology.identity_of(item))
            for item in source_runtime.definition_graph.predicates.refs()
        )),
        source_runtime.definition_graph.value_protocol,
    )
    candidate_graph = CandidateProjectionGraph(
        ontology,
        source_runtime.learning.graph.protocol,
    )
    return source_runtime.restore_for_training_graphs(
        definition_graph,
        candidate_graph,
        target_context.training_candidate_history,
    )


@pytest.mark.parametrize("state", ("forming", "active", "archived"))
def test_connector_m03_restore_preserves_forming_active_and_archived(
        state, tmp_path):
    """connector 三种候选状态可从 Dict dump 到 SQLite 继续恢复。"""
    host_backend = DictBackend()
    target_backend = SQLiteBackend()
    try:
        (graphs, promotion, template, predicate_identities, _candidate_graph,
         runtime) = _memory_candidate_fixture(
             host_backend, variant=240 + (0 if state == "forming" else 1))
        hypothesis = _register(runtime, promotion, template)
        if state == "active":
            _recognize(
                runtime, hypothesis, template, source_id=3, event=7,
                stance="support")
        elif state == "archived":
            _recognize(
                runtime, hypothesis, template, source_id=3, event=8,
                stance="refute", archive_refuted=True)

        dump_run(
            host_backend,
            str(tmp_path),
            f"connector_{state}",
            spaces=[
                graphs.context.space_id,
                graphs.context.memory_read.space_id,
                graphs.context.memory_interact.space_id,
            ],
            tables=DUMP_TABLES,
        )
        target_context = make_train_context(target_backend)
        assert load_run(
            target_backend, str(tmp_path), f"connector_{state}")
        target_context.graph_ontology.clear_runtime_caches()
        restored = _restore_memory_candidate(
            runtime,
            graphs,
            target_context,
            predicate_identities,
        )

        assert restored.memory_event_log is target_context.memory_read_events
        assert restored.memory_event_log is not runtime.memory_event_log
        assert restored.state_key() == runtime.state_key()
        assert restored.learning.next_timestamps(1) == (
            runtime.learning.next_timestamps(1))
        if state == "forming":
            assert restored.trial_template(hypothesis) == template
            assert restored.active_templates() == ()
        elif state == "active":
            assert restored.active_templates() == (template,)
        else:
            assert restored.active_templates() == ()
            with pytest.raises(LanguageConnectorCandidateError, match="forming"):
                restored.trial_template(hypothesis)
    finally:
        target_backend.close()
        host_backend.close()


def test_connector_unknown_decision_restores_without_projection_then_continues(
        tmp_path):
    """无图 unknown 决策也必须恢复 previous 链，并能在后续 support 时续写。"""
    host_backend = DictBackend()
    target_backend = SQLiteBackend()
    try:
        (graphs, promotion, template, predicate_identities, _candidate_graph,
         runtime) = _memory_candidate_fixture(host_backend, variant=244)
        hypothesis = _register(runtime, promotion, template)
        unknown = _recognize(
            runtime, hypothesis, template, source_id=3, event=9,
            stance="unknown")
        assert unknown.projection is None
        assert runtime.learning.engine.resolver.decision_history(hypothesis)

        dump_run(
            host_backend,
            str(tmp_path),
            "connector_unknown",
            spaces=[
                graphs.context.space_id,
                graphs.context.memory_read.space_id,
                graphs.context.memory_interact.space_id,
            ],
            tables=DUMP_TABLES,
        )
        target_context = make_train_context(target_backend)
        load_run(target_backend, str(tmp_path), "connector_unknown")
        target_context.graph_ontology.clear_runtime_caches()
        restored = _restore_memory_candidate(
            runtime,
            graphs,
            target_context,
            predicate_identities,
        )
        assert restored.state_key() == runtime.state_key()
        prior = restored.learning.engine.resolver.decision_history(
            hypothesis)[-1]

        host_outcome = _recognize(
            runtime, hypothesis, template, source_id=4, event=10,
            stance="support")
        restored_outcome = _recognize(
            restored, hypothesis, template, source_id=4, event=10,
            stance="support")
        assert host_outcome.decision.previous_decision_id == prior.decision_id
        assert restored_outcome.decision == host_outcome.decision
        assert restored.state_key() == runtime.state_key()
        assert restored.active_templates() == (template,)
    finally:
        target_backend.close()
        host_backend.close()


def test_connector_restore_rejects_stale_active_graph_after_m03_refute():
    """M-03 已产生新 refute 决策而图仍旧 active 时必须拒绝恢复采用。"""
    backend = DictBackend()
    try:
        (graphs, promotion, template, _predicate_identities, candidate_graph,
         runtime) = _memory_candidate_fixture(backend, variant=246)
        hypothesis = _register(runtime, promotion, template)
        _recognize(
            runtime, hypothesis, template, source_id=3, event=16,
            stance="support")
        observation = _source(4)
        event_key = (_BASE, 17)
        prediction = runtime.learning.engine.predict(
            hypothesis,
            observation=observation,
            scope=document_scope(observation),
            event_key=event_key,
            visible_inputs=(template.connector,),
            predicted=template.connector,
        )
        verification = runtime.learning.verifier.verify(
            prediction,
            _reveal(
                observation,
                event_key=event_key,
                refuted=(template.connector,),
            ),
        )
        runtime.learning.engine.reveal(
            prediction, verification, timestamp_seq=170)
        runtime.learning.engine.resolve(hypothesis, timestamp_seq=171)

        with pytest.raises(
                CandidateHistoryUnavailableError,
                match="图仍为 active|不同步"):
            runtime.restore_for_graphs(
                runtime.definition_graph,
                candidate_graph,
                graphs.context.memory_read_events,
            )
    finally:
        backend.close()


def test_connector_restore_filters_full_aggregate_protocol_before_local_ids():
    """相同 kind/裸 Evidence id 的其他 aggregate 来源不得混入 connector。"""
    backend = DictBackend()
    try:
        (graphs, promotion, template, _predicate_identities, candidate_graph,
         runtime) = _memory_candidate_fixture(backend, variant=247)
        hypothesis = _register(runtime, promotion, template)
        local_id = runtime.learning.engine.ledger.evidence_history(
            hypothesis)[0].evidence_id
        foreign_source = _source(777)
        foreign = HypothesisKey(
            runtime.learning.engine.protocol.hypothesis_kind_key,
            (999,),
            hypothesis.competition_key,
            document_scope(foreign_source),
            foreign_source,
        )
        foreign_ledger = HypothesisLedger(MemoryHypothesisEventSink(
            graphs.context.memory_read_events))
        foreign_ledger.register(foreign)
        foreign_ledger.append_evidence(EvidenceRecord(
            local_id,
            foreign,
            EVIDENCE_SUPPORT,
            (999, 1),
            foreign_source,
            1,
        ))

        restored = runtime.restore_for_graphs(
            runtime.definition_graph,
            candidate_graph,
            graphs.context.memory_read_events,
        )
        assert restored.learning.engine.ledger.hypotheses() == (hypothesis,)
        assert restored.state_key() == runtime.state_key()
    finally:
        backend.close()


def _register(runtime, promotion, template):
    """按统一来源和图元数据登记一个 connector forming 候选。"""
    return runtime.register(
        template,
        (_source(1), _source(2)),
        scope=promotion.constraint.scope,
        provenance_kind=31,
        epistemic_origin=32,
        content_version=33,
        qualifiers=(34,),
        timestamp_base=1,
    )


def _recognize(
        runtime, hypothesis, template, *, source_id: int, event: int,
        stance: str, archive_refuted: bool = False, replacement=None):
    """提交一次独立揭示，并用互异逻辑序同步 H-00/H-04 与图状态。"""
    observation = _source(source_id)
    event_key = (_BASE, event)
    if stance == "support":
        supported = (template.connector,)
        refuted = ()
    elif stance == "refute":
        supported = ()
        refuted = (template.connector,)
    elif stance == "unknown":
        supported = (concept_identity((_BASE, event, 99)),)
        refuted = ()
    else:
        raise ValueError("测试 stance 非法")
    return runtime.recognize(
        hypothesis,
        observation=observation,
        scope=document_scope(observation),
        event_key=event_key,
        visible_inputs=(template.connector,),
        predicted=template.connector,
        revealed=_reveal(
            observation,
            event_key=event_key,
            supported=supported,
            refuted=refuted,
        ),
        timestamp_seq=event * 10,
        resolve_timestamp_seq=event * 10 + 1,
        projection_timestamp_seq=event * 10 + 2,
        archive_refuted=archive_refuted,
        replacement=replacement,
    )


def _replacement_template(template, *, variant: int):
    """保持竞争边界不变，仅替换 connector 及其独占内部理论身份。"""
    bindings = tuple(
        replace(
            item,
            binding=structure_concept_identity(
                (_BASE + 1, variant, index)),
        )
        for index, item in enumerate(template.bindings, start=1)
    )
    surface = tuple(
        replace(
            item,
            directive=structure_concept_identity(
                (_BASE + 2, variant, index)),
            prefix_route=structure_concept_identity(
                (_BASE + 3, variant, index)),
        )
        for index, item in enumerate(template.surface, start=1)
    )
    return replace(
        template,
        connector=structure_concept_identity((_BASE + 4, variant)),
        bindings=bindings,
        constraint_set=structure_concept_identity((_BASE + 5, variant)),
        context_set=structure_concept_identity((_BASE + 6, variant)),
        surface=surface,
    )


def _typed_episode_candidate_fixture(
        *, variant: int, attributed: bool = True, trial: bool = False,
        training_history: bool = False):
    """运行最小真实 G-00 至 G-04 链，并由 active factory 注入候选归属。"""
    initial_request, _unresolved = _request(count=1)
    source = initial_request.goal.source
    branch = language_branch_identity((_BASE + 29, variant, 1))
    role = role_identity((_BASE + 29, variant, 2))
    filler = concept_identity((_BASE + 29, variant, 3))
    selection = _selection_with_role(branch, role, filler)
    formal_scope = episode_scope(
        variant,
        parent=document_scope(source),
    )
    runtime_scope = query_scope(1, parent=formal_scope)
    base_request = selection.request
    request = GenerationPlanningRequest(
        replace(base_request.goal, scope=runtime_scope),
        tuple(replace(item, scope=runtime_scope)
              for item in base_request.candidates),
    )
    backend = DictBackend()
    (
        production,
        _renderer,
        alias,
        promotion,
        _predicate_representation,
        _filler_representation,
        graphs,
        connector,
    ) = _connector_generation_runtime(request, backend)

    predicate_identities = tuple(
        concept_identity((_BASE + 30, variant, index))
        for index in range(21)
    )
    definition_graph = LanguageGenerationConnectorGraph(
        graphs.context.graph_ontology,
        graphs.order_graph,
        LanguageConnectorGraphPredicates(*tuple(
            graphs.context.graph_ontology.materialize(identity)
            for identity in predicate_identities
        )),
        connector.registry.value_protocol,
    )
    candidate_graph = CandidateProjectionGraph(
        graphs.context.graph_ontology,
        _projection_protocol(),
    )
    candidate_protocol = LanguageConnectorCandidateProtocol(
        concept_identity((_BASE + 31, variant, 1)),
        concept_identity((_BASE + 31, variant, 2)),
        concept_identity((_BASE + 31, variant, 3)),
        (_BASE + 31, variant, 4),
    )
    learning = _runtime(candidate_graph, kind_key=900 + variant)
    if training_history:
        aggregate = learning.engine.protocol
        sink = TrainingHypothesisEventSink(
            graphs.context.training_candidate_history,
            TrainingHypothesisHistoryProtocol(
                candidate_protocol.stable_key(),
                aggregate.hypothesis_kind_key,
                aggregate.aggregate_source,
                aggregate.aggregate_scope,
            ),
        )
        learning = CandidateLearningRuntime(
            EvidenceCandidateEngine(
                aggregate,
                ledger=HypothesisLedger(sink),
            ),
            candidate_graph,
            learning.verifier,
            learning.metadata,
        )
    candidates = LanguageConnectorCandidateRuntime(
        definition_graph,
        learning,
        candidate_protocol,
    )
    template = connector.registry.templates[0]
    hypothesis = _register(candidates, promotion, template)
    if not trial:
        _recognize(
            candidates,
            hypothesis,
            template,
            source_id=300 + variant,
            event=300 + variant,
            stance="support",
        )
    if attributed:
        if trial:
            assembly = TrialLanguageConnectorFactory(
                candidates,
                connector.runtime_policy,
                connector.surface_protocol,
                hypothesis,
                minimal_instruction_identity((_BASE + 33, variant, 2)),
            ).build(graphs.context)
        else:
            assembly = ActiveLanguageConnectorFactory(
                candidates,
                connector.runtime_policy,
                connector.surface_protocol,
                minimal_instruction_identity((_BASE + 33, variant, 1)),
            ).build(graphs.context)
        surface_resolver = (
            production._executor._planner._registrations[-1].resolver)
        surface_resolver._request_builder = (
            assembly.connector.surface_request_builder(
                _execution_planner(graphs, promotion)))
    execution = production._executor.execute(request)
    parser = production._postchecker.parser
    if not isinstance(parser, _ExecutionParser):
        raise TypeError("connector stage4 fixture 需要显式可登记的受限 parser")
    parser.record(execution)
    postcheck = production._postchecker.run(GenerationPostcheckRequest(
        execution,
        (),
        _source_requirements(execution),
    ))
    run = ProductionGenerationRun(
        ProductionGenerationRequestDecision(
            minimal_instruction_identity((_BASE + 29, variant, 4)),
            (_BASE + 29, variant, 5),
            request,
        ),
        execution,
        postcheck,
    )
    episode = TypedLanguageEpisode.from_production(
        variant,
        source,
        formal_scope,
        run,
        read_only=False,
    )
    return (
        backend,
        alias,
        episode,
        candidates,
        template,
        promotion,
        hypothesis,
        connector,
    )


def _cross_source_candidate(
        candidate,
        *,
        source,
        scope,
        variant: int,
        ):
    """把一个候选完整重绑到另一知识来源，同时保留当前 query scope。"""
    proposition = replace(
        candidate.proposition,
        template=proposition_identity(source, (_BASE + 78, variant, 1)),
        source_anchor=occurrence_identity(
            source,
            start=_BASE + 78 + variant,
            end=_BASE + 79 + variant,
            ordinal=0,
        ),
        context=context_scope_identity(source, (_BASE + 78, variant, 2)),
    )
    evidence = tuple(
        replace(
            item,
            hypothesis=HypothesisKey(
                item.hypothesis.hypothesis_kind,
                item.hypothesis.candidate_key,
                item.hypothesis.competition_key,
                document_scope(source),
                source,
            ),
            source=source,
        )
        for item in candidate.evidence
    )
    return replace(
        candidate,
        proposition=proposition,
        source=source,
        scope=scope,
        evidence=evidence,
    )


def _install_core_use_owner(alias, *, variant: int) -> None:
    """为真实 G-03 fixture 的 R-01 runtime 装配同图 PH2 Core Use owner。"""
    predicate_identities = tuple(
        concept_identity((_BASE + 78, variant, index))
        for index in range(1, 9)
    )
    use_owner = RelationUseOwner(
        RelationUseGraph(
            alias.semantic_graph.ontology,
            RelationUseGraphProtocol(
                *predicate_identities,
                (_BASE + 78, variant, 9),
            ),
        ),
        RelationUseWriteMetadata(
            SOURCE_BARE_TEXT,
            EPI_STRUCTURED,
            content_version=1,
            qualifiers=(_BASE + 78, variant, 10),
        ),
    )
    closure = RelationClosureRuntime(
        alias.closure.candidate_runtime,
        alias.semantic_graph,
        alias.closure.consumer,
        alias.closure.protocol,
        use_owner,
    )
    alias.closure = closure
    alias.runtime = AliasRelationRuntime(
        closure,
        AliasResolutionSelector(alias.protocol),
    )


def _multi_connector_episode_fixture(
        *, variant: int, same_template: bool,
        cross_source: bool = False,
        training_history: bool = False,
        runtime_owner: OwnerScope | None = None):
    """经真实 G-00 至 G-04 形成多句 episode，保留逐句理论和 Core Use 归属。"""
    base_request, _unresolved = _request(count=2)
    branch = language_branch_identity((_BASE + 70, variant, 1))
    role = role_identity((_BASE + 70, variant, 2))
    filler = concept_identity((_BASE + 70, variant, 3))
    first_raw, second_raw = base_request.candidates
    first = replace(
        first_raw,
        proposition=replace(
            first_raw.proposition,
            bindings=(BoundRoleBinding(role, filler),),
        ),
    )
    second_proposition = (
        first.proposition if same_template else replace(
            second_raw.proposition,
            predicate=first.proposition.predicate,
            bindings=(BoundRoleBinding(role, filler),),
        )
    )
    second = replace(second_raw, proposition=second_proposition)
    initial_request = GenerationPlanningRequest(
        replace(
            base_request.goal,
            proposition=first.proposition,
            target_branch=branch,
        ),
        (first,),
    )
    source = initial_request.goal.source
    backend = DictBackend()
    (
        production,
        _renderer,
        alias,
        promotion,
        _predicate_representation,
        _filler_representation,
        graphs,
        original_connector,
    ) = _connector_generation_runtime(initial_request, backend)
    _install_core_use_owner(alias, variant=variant)
    core_use_surface_runtime = GenerationSurfaceRuntime(alias.runtime)
    production._executor._committer = core_use_surface_runtime
    for registration in production._executor._planner._registrations:
        resolver = registration.resolver
        if hasattr(resolver, "_runtime"):
            resolver._runtime = core_use_surface_runtime
    if runtime_owner is None:
        formal_scope = episode_scope(
            variant,
            parent=document_scope(source),
        )
    else:
        formal_scope = episode_scope(
            variant,
            parent=make_scope(
                SCOPE_DOCUMENT,
                _BASE + 79 + variant,
                owner=runtime_owner,
            ),
        )
    runtime_scope = query_scope(1, parent=formal_scope)
    first = replace(first, scope=runtime_scope)
    second = (
        _cross_source_candidate(
            second,
            source=_source(740 + variant),
            scope=runtime_scope,
            variant=variant,
        )
        if cross_source else replace(second, scope=runtime_scope)
    )
    request = GenerationPlanningRequest(
        replace(initial_request.goal, scope=runtime_scope),
        (first, second),
    )

    predicate_identities = tuple(
        concept_identity((_BASE + 71, variant, index))
        for index in range(21)
    )
    definition_graph = LanguageGenerationConnectorGraph(
        graphs.context.graph_ontology,
        graphs.order_graph,
        LanguageConnectorGraphPredicates(*tuple(
            graphs.context.graph_ontology.materialize(identity)
            for identity in predicate_identities
        )),
        original_connector.registry.value_protocol,
    )
    candidate_graph = CandidateProjectionGraph(
        graphs.context.graph_ontology,
        _projection_protocol(),
    )
    candidate_protocol = LanguageConnectorCandidateProtocol(
        concept_identity((_BASE + 72, variant, 1)),
        concept_identity((_BASE + 72, variant, 2)),
        concept_identity((_BASE + 72, variant, 3)),
        (_BASE + 72, variant, 4),
    )
    learning = _runtime(candidate_graph, kind_key=950 + variant)
    if training_history:
        aggregate = learning.engine.protocol
        sink = TrainingHypothesisEventSink(
            graphs.context.training_candidate_history,
            TrainingHypothesisHistoryProtocol(
                candidate_protocol.stable_key(),
                aggregate.hypothesis_kind_key,
                aggregate.aggregate_source,
                aggregate.aggregate_scope,
            ),
        )
        learning = CandidateLearningRuntime(
            EvidenceCandidateEngine(
                aggregate,
                ledger=HypothesisLedger(sink),
            ),
            candidate_graph,
            learning.verifier,
            learning.metadata,
        )
    candidates = LanguageConnectorCandidateRuntime(
        definition_graph,
        learning,
        candidate_protocol,
    )
    first_template = original_connector.registry.templates[0]
    first_hypothesis = _register(candidates, promotion, first_template)
    _recognize(
        candidates,
        first_hypothesis,
        first_template,
        source_id=700 + variant,
        event=700 + variant,
        stance="support",
    )
    templates = (first_template,)
    runtime_policy = original_connector.runtime_policy
    purpose = minimal_instruction_identity((_BASE + 33, variant, 1))
    attributions = (GenerationSurfaceAttribution(
        first_template.connector,
        first_hypothesis,
        purpose,
    ),)
    hypotheses = (first_hypothesis,)
    if not same_template:
        second_template = replace(
            _replacement_template(first_template, variant=variant),
            sentence=structure_concept_identity((_BASE + 73, variant, 1)),
            proposition_structure=second.proposition.structure,
            predicate=second.proposition.predicate,
        )
        second_hypothesis = _register(
            candidates,
            promotion,
            second_template,
        )
        _recognize(
            candidates,
            second_hypothesis,
            second_template,
            source_id=710 + variant,
            event=710 + variant,
            stance="support",
        )
        templates = (first_template, second_template)
        runtime_policy = replace(
            original_connector.runtime_policy,
            templates=(
                original_connector.runtime_policy.templates[0],
                replace(
                    original_connector.runtime_policy.templates[0],
                    connector=second_template.connector,
                ),
            ),
        )
        attributions = (
            attributions[0],
            GenerationSurfaceAttribution(
                second_template.connector,
                second_hypothesis,
                purpose,
            ),
        )
        hypotheses = (first_hypothesis, second_hypothesis)
    registry = LanguageGenerationConnectorRegistry(
        original_connector.registry.value_protocol,
        templates,
    )
    dependency = DiscourseDependency(
        first.stable_key(),
        second.stable_key(),
        structure_concept_identity((_BASE + 74, variant, 1)),
        minimal_instruction_identity((_BASE + 74, variant, 2)),
        (_BASE + 74, variant, 3),
    )
    declarations = StaticLanguageConnectorDiscourseDeclarations((
        LanguageConnectorDiscourseDeclaration(
            request.candidate_keys(),
            (dependency,),
            _source(720 + variant),
            (_BASE + 74, variant, 4),
        ),
    ))
    connector = LanguageGenerationConnector(
        registry,
        runtime_policy,
        original_connector.surface_protocol,
        attributions,
        declarations,
    )
    structure_planner = connector.structure_planner()
    surface_builder = connector.surface_request_builder(
        _execution_planner(graphs, promotion))
    for registration in production._executor._planner._registrations:
        resolver = registration.resolver
        if hasattr(resolver, "_planner"):
            resolver._planner = structure_planner
        if hasattr(resolver, "_structure_planner"):
            resolver._structure_planner = structure_planner
        if hasattr(resolver, "_request_builder"):
            resolver._request_builder = surface_builder
    execution = production._executor.execute(request)
    parser = production._postchecker.parser
    if not isinstance(parser, _ExecutionParser):
        raise TypeError("多命题 fixture 需要显式可登记的受限 parser")
    parser.record(execution)
    postcheck = production._postchecker.run(GenerationPostcheckRequest(
        execution,
        (),
        _source_requirements(execution),
    ))
    run = ProductionGenerationRun(
        ProductionGenerationRequestDecision(
            minimal_instruction_identity((_BASE + 75, variant, 1)),
            (_BASE + 75, variant, 2),
            request,
        ),
        execution,
        postcheck,
    )
    episode = TypedLanguageEpisode.from_production(
        variant,
        source,
        formal_scope,
        run,
        read_only=False,
    )
    return (
        backend,
        alias,
        episode,
        candidates,
        templates,
        hypotheses,
        connector,
        declarations,
    )


def _stage4_policy(episode, *, variant: int):
    """只注册对本次全部 candidate 明确作出 verdict 的 G-04 维度。"""
    execution = episode.production.execution
    if execution is None or execution.surface is None:
        raise ValueError("stage4 fixture 缺少完整 typed surface")
    candidate_keys = {
        item.candidate_key
        for item in execution.surface.preview.request.structure
        .propositions.propositions
    }
    if not candidate_keys:
        raise ValueError("stage4 fixture 缺少 candidate claim")
    routes = tuple(
        LanguageConnectorSignalRoute(
            signal.dimension,
            signal.verifier,
            ((APPLICABILITY_APPLICABLE, VERDICT_SUPPORT),),
            ((APPLICABILITY_APPLICABLE, VERDICT_REFUTE),),
        )
        for signal in episode.signals
        if (signal.applicability == APPLICABILITY_APPLICABLE
            and candidate_keys.issubset(signal.claim_keys))
    )
    if not routes:
        raise ValueError("stage4 fixture 缺少 candidate 归属的 G-04 route")
    return LanguageConnectorStage4Policy(
        routes,
        _source(500 + variant),
        (_BASE + 32, variant),
        minimal_instruction_identity((_BASE + 33, variant, 1)),
        minimal_instruction_identity((_BASE + 33, variant, 2)),
    )


def _episode_with_connector_verdict(
        episode, verdict, *, round_id=None):
    """只改一条明确 claim 全部 candidate 的 G-04 verdict，用于真实反驳路径。"""
    execution = episode.production.execution
    postcheck = episode.production.postcheck
    if execution is None or execution.surface is None or postcheck is None:
        raise ValueError("stage4 fixture 缺少完整 G-04 execution")
    candidate_keys = {
        item.candidate_key
        for item in execution.surface.preview.request.structure
        .propositions.propositions
    }
    results = []
    changed = False
    for result in postcheck.report.results:
        if (not changed
                and result.applicability == APPLICABILITY_APPLICABLE
                and candidate_keys.issubset(result.claim_keys)):
            results.append(replace(result, verdict=verdict))
            changed = True
        else:
            results.append(result)
    if not changed:
        raise ValueError("stage4 fixture 无可改写的 candidate G-04 verdict")
    production = replace(
        episode.production,
        postcheck=replace(
            postcheck,
            report=replace(postcheck.report, results=tuple(results)),
        ),
    )
    return TypedLanguageEpisode.from_production(
        episode.round_id if round_id is None else round_id,
        episode.source,
        episode.scope,
        production,
        read_only=episode.read_only,
        supplemental_verification=episode.supplemental_verification,
    )


def _episode_without_route_candidate_claim(
        episode,
        policy: LanguageConnectorStage4Policy,
        candidate_key: tuple[int, ...],
        ):
    """只移除一个已路由 G-04 结果的 candidate claim，验证 stage4 零写拒绝。"""
    postcheck = episode.production.postcheck
    if postcheck is None:
        raise ValueError("stage4 fixture 缺少 G-04 postcheck")
    route = policy.routes[0]
    changed = False
    results = []
    for result in postcheck.report.results:
        if (not changed
                and result.dimension == route.dimension
                and result.verifier == route.verifier
                and candidate_key in result.claim_keys):
            remaining = tuple(
                item for item in result.claim_keys if item != candidate_key)
            if not remaining:
                raise ValueError("stage4 fixture 无法构造非空缺 claim 结果")
            results.append(replace(result, claim_keys=remaining))
            changed = True
        else:
            results.append(result)
    if not changed:
        raise ValueError("stage4 fixture 未找到可改写的 route claim")
    production = replace(
        episode.production,
        postcheck=replace(
            postcheck,
            report=VerificationReport(True, tuple(results)),
        ),
    )
    return TypedLanguageEpisode.from_production(
        episode.round_id,
        episode.source,
        episode.scope,
        production,
        read_only=episode.read_only,
        supplemental_verification=episode.supplemental_verification,
    )


class _ProductionBindingBuilder:
    """按目标 context 重建 S-07/R-01，并允许注入单项归属故障。"""

    VALID = 0
    REPLACE_CONNECTOR = 1
    DRIFT_ORDER_FACADE = 2
    FOREIGN_ALIAS = 3

    def __init__(
            self,
            graphs,
            alias_protocol,
            relation_schemas,
            *,
            mode: int = VALID,
            foreign_alias=None,
            runtime_key=(_BASE + 34, 1),
            ) -> None:
        ontology = graphs.context.graph_ontology
        self.lifecycle_predicates = tuple(
            ontology.identity_of(item)
            for item in graphs.lifecycle.protocol.predicate_refs()
        )
        self.lifecycle_states_and_kinds = (
            graphs.lifecycle.protocol.state_identities()
            + graphs.lifecycle.protocol.kind_identities()
        )
        self.lifecycle_event_namespace = (
            graphs.lifecycle.protocol.event_namespace_key)
        self.alias_protocol = alias_protocol
        self.relation_schemas = tuple(relation_schemas)
        self.mode = mode
        self.foreign_alias = foreign_alias
        self.runtime_key = tuple(runtime_key)
        self.bindings = []

    def build(self, ctx, assembly):
        """在当前图装配 runtime binding，并仅按 mode 制造一个可定位故障。"""
        refs = tuple(
            ctx.graph_ontology.resolve(identity)
            for identity in self.lifecycle_predicates
        )
        if any(item is None for item in refs):
            raise RuntimeError("测试 builder 无法恢复 S-07 lifecycle predicate")
        order_graph = assembly.order_graph
        if self.mode == self.DRIFT_ORDER_FACADE:
            order_graph = StructureOrderGraph(
                ctx.graph_ontology,
                assembly.order_graph.predicates,
            )
        lifecycle = StructureOrderLifecycleGraph(
            order_graph,
            StructureOrderLifecycleProtocol(
                *refs,
                *self.lifecycle_states_and_kinds,
                self.lifecycle_event_namespace,
            ),
        )
        alias = self._alias_for_context(ctx)
        if self.mode == self.FOREIGN_ALIAS:
            alias = self.foreign_alias
        connector = assembly.connector
        if self.mode == self.REPLACE_CONNECTOR:
            connector = LanguageGenerationConnector(
                connector.registry,
                connector.runtime_policy,
                connector.surface_protocol,
                tuple(connector.attribution_mapper.attributions.values()),
            )
        binding = LanguageConnectorProductionRuntimeBinding(
            _SemanticNoRequestFactory(self.runtime_key).build(ctx),
            connector,
            lifecycle,
            alias,
        )
        self.bindings.append(binding)
        return binding

    def _alias_for_context(self, ctx):
        """建立绑定当前 ontology 的空 R-01 owner，不共享 Use 或候选状态。"""
        semantic_graph = _semantic_graph(ctx.graph_ontology)
        candidate_graph = CandidateProjectionGraph(
            ctx.graph_ontology,
            _relation_projection_protocol(),
        )
        candidate_runtime = _relation_candidate_runtime(candidate_graph)
        relation_protocol = _relation_protocol()
        consumer = ActiveRelationClosureConsumer(
            semantic_graph,
            candidate_graph,
            relation_protocol,
            self.relation_schemas,
            engine=candidate_runtime.engine,
        )
        closure = RelationClosureRuntime(
            candidate_runtime,
            semantic_graph,
            consumer,
            relation_protocol,
        )
        return AliasRelationRuntime(
            closure,
            AliasResolutionSelector(self.alias_protocol),
        )

    def clone_for_evaluation(self):
        """复制配置但清空 runtime binding 记录和所有可变 owner。"""
        return _ProductionBindingBuilder(
            _BuilderGraphsProtocol(
                self.lifecycle_predicates,
                self.lifecycle_states_and_kinds,
                self.lifecycle_event_namespace,
            ),
            self.alias_protocol,
            self.relation_schemas,
            mode=self.mode,
            foreign_alias=self.foreign_alias,
            runtime_key=self.runtime_key,
        )

    def state_key(self):
        """返回 lifecycle、R-01、runtime 和故障注入的完整配置键。"""
        return (
            tuple(item.stable_key() for item in self.lifecycle_predicates),
            tuple(item.stable_key()
                  for item in self.lifecycle_states_and_kinds),
            self.lifecycle_event_namespace,
            self.alias_protocol.stable_key(),
            tuple(self._schema_key(item) for item in self.relation_schemas),
            self.mode,
            self.runtime_key,
        )

    @staticmethod
    def _schema_key(schema):
        """展开关系 schema 的 Role、类型、基数和同型约束配置。"""
        return (
            schema.schema.stable_key(),
            schema.relation.stable_key(),
            tuple((
                slot.role.stable_key(),
                tuple(sorted(slot.allowed_object_kinds)),
                slot.min_count,
                -1 if slot.max_count is None else slot.max_count,
            ) for slot in schema.slots),
            tuple((
                constraint.constraint.stable_key(),
                tuple(role.stable_key() for role in constraint.roles),
            ) for constraint in schema.same_kind_constraints),
        )


class _BuilderGraphsProtocol:
    """让 builder 克隆复用冻结 identity，而不携带宿主图 facade。"""

    def __init__(self, predicates, states_and_kinds, event_namespace) -> None:
        self.context = _BuilderOntologyProtocol(predicates)
        self.lifecycle = _BuilderLifecycleProtocol(
            predicates,
            states_and_kinds,
            event_namespace,
        )


class _BuilderOntologyProtocol:
    """只为 builder 构造器提供 identity_of 的冻结视图。"""

    def __init__(self, predicates) -> None:
        self.graph_ontology = self
        self.predicates = tuple(predicates)

    @staticmethod
    def identity_of(value):
        """冻结视图收到的值已经是 ObjectIdentity。"""
        return value


class _BuilderLifecycleProtocol:
    """只暴露 builder 构造器读取的 lifecycle 协议字段。"""

    def __init__(self, predicates, states_and_kinds, event_namespace) -> None:
        self.protocol = self
        self._predicates = tuple(predicates)
        self._states_and_kinds = tuple(states_and_kinds)
        self.event_namespace_key = tuple(event_namespace)

    def predicate_refs(self):
        """返回已冻结 predicate identity。"""
        return self._predicates

    def state_identities(self):
        """返回三个 lifecycle state identity。"""
        return self._states_and_kinds[:3]

    def kind_identities(self):
        """返回三个 lifecycle event kind identity。"""
        return self._states_and_kinds[3:]


def _production_stage4_policy(*, variant: int, active_purpose=None):
    """构造不解释名称的最小 connector stage4 路由协议。"""
    active = active_purpose or minimal_instruction_identity(
        (_BASE + 35, variant, 1))
    return LanguageConnectorStage4Policy(
        (LanguageConnectorSignalRoute(
            ProtocolKey((_BASE + 36, variant, 1)),
            ProtocolKey((_BASE + 36, variant, 2)),
            ((APPLICABILITY_APPLICABLE, VERDICT_SUPPORT),),
            ((APPLICABILITY_APPLICABLE, VERDICT_REFUTE),),
        ),),
        _source(600 + variant),
        (_BASE + 37, variant),
        active,
        minimal_instruction_identity((_BASE + 35, variant, 2)),
    )


def test_connector_forming_and_unknown_never_enter_active_registry():
    """forming 来源和独立 unknown 都只能留下候选历史，不能成为语言理论。"""
    backend = DictBackend()
    try:
        _graphs, promotion, template, _predicates, _graph, runtime = (
            _candidate_fixture(backend, variant=201)
        )
        hypothesis = _register(runtime, promotion, template)

        assert runtime.active_templates() == ()
        with pytest.raises(LanguageConnectorCandidateError, match="没有 active"):
            runtime.active_registry()

        outcome = _recognize(
            runtime,
            hypothesis,
            template,
            source_id=3,
            event=1,
            stance="unknown",
        )
        assert outcome.projection is None
        assert runtime.active_templates() == ()
    finally:
        backend.close()


def test_connector_support_activates_and_refute_deactivates_theory():
    """独立 support 后才可采用，定向 refute 归档后立即退出 registry。"""
    backend = DictBackend()
    try:
        _graphs, promotion, template, _predicates, _graph, runtime = (
            _candidate_fixture(backend, variant=202)
        )
        hypothesis = _register(runtime, promotion, template)

        supported = _recognize(
            runtime,
            hypothesis,
            template,
            source_id=3,
            event=2,
            stance="support",
        )
        assert supported.projection is not None
        assert runtime.active_templates() == (template,)
        assert runtime.active_registry().templates == (template,)

        refuted = _recognize(
            runtime,
            hypothesis,
            template,
            source_id=4,
            event=3,
            stance="refute",
            archive_refuted=True,
        )
        assert refuted.projection is not None
        assert refuted.projection.state == runtime.learning.graph.protocol.inactive_state
        assert runtime.active_templates() == ()
    finally:
        backend.close()


def test_connector_replacement_supersedes_old_theory_without_private_choice():
    """同竞争组的新理论被支持后，显式 replacement 只淘汰指定旧候选。"""
    backend = DictBackend()
    try:
        _graphs, promotion, first, _predicates, _graph, runtime = (
            _candidate_fixture(backend, variant=203)
        )
        second = _replacement_template(first, variant=203)
        first_hypothesis = _register(runtime, promotion, first)
        second_hypothesis = _register(runtime, promotion, second)
        _recognize(
            runtime,
            first_hypothesis,
            first,
            source_id=3,
            event=4,
            stance="support",
        )
        _recognize(
            runtime,
            second_hypothesis,
            second,
            source_id=4,
            event=5,
            stance="support",
        )

        outcome = _recognize(
            runtime,
            first_hypothesis,
            first,
            source_id=5,
            event=6,
            stance="refute",
            replacement=second_hypothesis,
        )

        assert outcome.projection is not None
        assert outcome.projection.state == (
            runtime.learning.graph.protocol.superseded_state)
        assert outcome.projection.replacement == second.connector
        assert runtime.active_templates() == (second,)
    finally:
        backend.close()


def test_connector_active_registry_rebuilds_from_both_graphs_after_cache_clear():
    """active registry 由恢复图和克隆 owner 重建，不依赖旧 registry 模板缓存。"""
    backend = DictBackend()
    try:
        (graphs, promotion, template, predicate_identities,
         _candidate_graph, runtime) = _candidate_fixture(
             backend, variant=204)
        hypothesis = _register(runtime, promotion, template)
        _recognize(
            runtime,
            hypothesis,
            template,
            source_id=3,
            event=7,
            stance="support",
        )
        before = backend.snapshot()

        graphs.context.graph_ontology.clear_runtime_caches()
        rebuilt_definition = LanguageGenerationConnectorGraph(
            graphs.context.graph_ontology,
            graphs.order_graph,
            LanguageConnectorGraphPredicates(*tuple(
                graphs.context.graph_ontology.resolve(identity)
                for identity in predicate_identities
            )),
            runtime.definition_graph.value_protocol,
        )
        rebuilt_candidate = CandidateProjectionGraph(
            graphs.context.graph_ontology,
            _projection_protocol(),
        )
        cloned = runtime.clone_for_graphs(
            rebuilt_definition,
            rebuilt_candidate,
        )

        assert cloned.active_registry().templates == (template,)
        assert backend.snapshot() == before
        assert cloned.state_key() == runtime.state_key()
    finally:
        backend.close()


def test_connector_v06_clone_uses_independent_backend_and_candidate_owner():
    """V-06 从独立后端恢复三张图，评测状态转换不得污染宿主。"""
    host_backend = DictBackend()
    clone_backend = None
    try:
        (graphs, promotion, template, connector_predicates,
         _candidate_graph, runtime) = _candidate_fixture(
             host_backend, variant=210)
        hypothesis = _register(runtime, promotion, template)
        _recognize(
            runtime,
            hypothesis,
            template,
            source_id=3,
            event=10,
            stance="support",
        )
        host_before = host_backend.snapshot()

        clone_backend = clone_storage_backend(host_backend)
        clone_context = make_train_context(clone_backend)
        clone_context.graph_ontology.clear_runtime_caches()
        order_graph = StructureOrderGraph(
            clone_context.graph_ontology,
            StructureOrderGraphPredicates(*tuple(
                clone_context.graph_ontology.resolve(identity)
                for identity in graphs.order_predicate_identities
            )),
        )
        host_lifecycle_protocol = graphs.lifecycle.protocol
        lifecycle = StructureOrderLifecycleGraph(
            order_graph,
            StructureOrderLifecycleProtocol(
                *tuple(
                    clone_context.graph_ontology.resolve(identity)
                    for identity in graphs.lifecycle_predicate_identities
                ),
                *graphs.states_and_kinds,
                host_lifecycle_protocol.event_namespace_key,
            ),
        )
        definition_graph = LanguageGenerationConnectorGraph(
            clone_context.graph_ontology,
            order_graph,
            LanguageConnectorGraphPredicates(*tuple(
                clone_context.graph_ontology.resolve(identity)
                for identity in connector_predicates
            )),
            runtime.definition_graph.value_protocol,
        )
        candidate_graph = CandidateProjectionGraph(
            clone_context.graph_ontology,
            runtime.learning.graph.protocol,
        )
        cloned = runtime.clone_for_graphs(definition_graph, candidate_graph)

        assert lifecycle.active_constraints(
            clone_context.graph_ontology.resolve(template.structure))
        assert cloned.active_templates() == (template,)
        _recognize(
            cloned,
            hypothesis,
            template,
            source_id=4,
            event=11,
            stance="refute",
            archive_refuted=True,
        )

        assert cloned.active_templates() == ()
        assert runtime.active_templates() == (template,)
        assert host_backend.snapshot() == host_before
        assert clone_backend.snapshot() != host_before
        assert cloned.learning.state_key() != runtime.learning.state_key()
    finally:
        if clone_backend is not None:
            clone_backend.close()
        host_backend.close()


def test_active_connector_factory_rebuilds_host_and_v06_from_graph_state():
    """工厂只采用 active 图理论，并在独立 context 重建而不共享 owner。"""
    host_backend = DictBackend()
    clone_backend = None
    try:
        graphs, promotion, connector, _predicates, candidates = (
            _factory_fixture(host_backend, variant=211)
        )
        template = connector.registry.templates[0]
        hypothesis = _register(candidates, promotion, template)
        _recognize(
            candidates,
            hypothesis,
            template,
            source_id=3,
            event=12,
            stance="support",
        )
        factory = ActiveLanguageConnectorFactory(
            candidates,
            connector.runtime_policy,
            connector.surface_protocol,
            minimal_instruction_identity((_BASE + 24, 211, 1)),
        )

        host = factory.build(graphs.context)
        assert host.candidates is candidates
        assert host.connector.registry.templates == (template,)
        assert host.connector.runtime_policy is connector.runtime_policy

        cloned_factory = factory.clone_for_evaluation()
        assert cloned_factory is not factory
        assert cloned_factory.state_key() == factory.state_key()
        clone_backend = clone_storage_backend(host_backend)
        clone_context = make_train_context(clone_backend)
        cloned = cloned_factory.build(clone_context)

        assert cloned.candidates is not candidates
        assert cloned.candidates.definition_graph.ontology is (
            clone_context.graph_ontology)
        assert cloned.connector.registry.templates == (template,)
        assert cloned.connector.stable_key() == host.connector.stable_key()
        assert cloned.order_graph.ontology is clone_context.graph_ontology

        policy = connector.runtime_policy.templates[0]
        drifting = replace(
            connector.runtime_policy,
            templates=(replace(
                policy,
                connector=structure_concept_identity(
                    (_BASE + 21, 211)),
            ),),
        )
        bad_factory = ActiveLanguageConnectorFactory(
            candidates,
            drifting,
            connector.surface_protocol,
            minimal_instruction_identity((_BASE + 24, 211, 1)),
        )
        with pytest.raises(ValueError, match="理论模板与运行策略"):
            bad_factory.build(graphs.context)
    finally:
        if clone_backend is not None:
            clone_backend.close()
        host_backend.close()


def _schedule_hypothesis(index: int) -> HypothesisKey:
    """为纯调度测试建立同竞争键但不同候选身份的 exact Hypothesis。"""
    source = _source(900 + index)
    return HypothesisKey(
        (_BASE + 41, 1),
        (_BASE + 41, 2, index),
        (_BASE + 41, 3),
        document_scope(source),
        source,
    )


def _selection_for_template(template, *, variant: int):
    """按模板声明的目标分支和 Role 建立同 match key 的单命题选择。"""
    role = next(item.role for item in template.bindings if item.role is not None)
    return _selection_with_role(
        template.language_branch,
        role,
        concept_identity((_BASE + 42, variant)),
    )


def test_scheduled_registry_prefers_active_and_rejects_multiple_forming():
    """active 同键优先；无 active 时多个 forming 必须拒绝而非稳定排序。"""
    backend = DictBackend()
    try:
        _graphs, _promotion, connector, _predicates, _candidates = (
            _factory_fixture(backend, variant=248)
        )
        active = connector.registry.templates[0]
        forming = replace(
            active,
            connector=structure_concept_identity((_BASE + 43, 1)),
        )
        other_forming = replace(
            active,
            connector=structure_concept_identity((_BASE + 43, 2)),
        )
        selection = _selection_for_template(active, variant=248)
        active_hypothesis = _schedule_hypothesis(1)
        first_forming = _schedule_hypothesis(2)
        second_forming = _schedule_hypothesis(3)

        mixed = ScheduledLanguageGenerationConnectorRegistry(
            connector.registry.value_protocol,
            ((active, active_hypothesis),),
            ((forming, first_forming),),
        )
        assert mixed.match(selection)[0] == active

        active_with_ambiguous_trials = (
            ScheduledLanguageGenerationConnectorRegistry(
                connector.registry.value_protocol,
                ((active, active_hypothesis),),
                (
                    (forming, first_forming),
                    (other_forming, second_forming),
                ),
            )
        )
        assert active_with_ambiguous_trials.match(selection)[0] == active

        ambiguous = ScheduledLanguageGenerationConnectorRegistry(
            connector.registry.value_protocol,
            (),
            (
                (forming, first_forming),
                (other_forming, second_forming),
            ),
        )
        with pytest.raises(
                LanguageGenerationConnectorError,
                match="多个 forming trial"):
            ambiguous.match(selection)

        ambiguous_active = ScheduledLanguageGenerationConnectorRegistry(
            connector.registry.value_protocol,
            (
                (active, active_hypothesis),
                (forming, first_forming),
            ),
            (),
        )
        with pytest.raises(
                LanguageGenerationConnectorError,
                match="多个 active 模板"):
            ambiguous_active.match(selection)

        with pytest.raises(ValueError, match="同一 Hypothesis"):
            ScheduledLanguageGenerationConnectorRegistry(
                connector.registry.value_protocol,
                ((active, active_hypothesis),),
                ((forming, active_hypothesis),),
            )
    finally:
        backend.close()


def test_scheduled_factory_indexes_once_and_v06_rebuilds_isolated_trial_owner():
    """forming 索引只在 owner 构建时扫描一次，V-06 从 clone 历史独立重建。"""
    host_backend = DictBackend()
    clone_backend = None
    relation_fixture = _r01_fixture()
    try:
        graphs, promotion, connector, _predicates, candidates = (
            _factory_fixture(host_backend, variant=249)
        )
        template = connector.registry.templates[0]
        hypothesis = _register(candidates, promotion, template)
        policy = _production_stage4_policy(variant=249)
        schedule = ScheduledLanguageConnectorFactory(
            candidates,
            connector.runtime_policy,
            connector.surface_protocol,
            policy.active_purpose,
            policy.trial_purpose,
        )
        definitions = candidates.learning.engine.definitions
        scans = []

        def counted_definitions():
            """记录启动恢复扫描次数，round 内 registry.match 不得再次调用。"""
            scans.append(1)
            return definitions()

        candidates.learning.engine.definitions = counted_definitions
        builder = _ProductionBindingBuilder(
            graphs,
            relation_fixture.protocol,
            relation_fixture.closure.consumer.schemas,
            runtime_key=(_BASE + 44, 249),
        )
        factory = LanguageConnectorProductionFactory(
            schedule,
            builder,
            policy,
        )
        installation = factory.build_installation(graphs.context)
        registry = builder.bindings[-1].connector.registry
        selection = _selection_for_template(template, variant=249)

        assert scans == [1]
        assert isinstance(
            registry, ScheduledLanguageGenerationConnectorRegistry)
        assert registry.active_entries == ()
        assert registry.trial_entries == ((template, hypothesis),)
        for _index in range(3):
            assert registry.match(selection)[0] == template
        assert scans == [1]
        attribution = builder.bindings[-1].connector.attribution_mapper \
            .attributions[template.connector]
        assert attribution.hypothesis == hypothesis
        assert attribution.purpose == policy.trial_purpose
        assert installation.stage4_runtime.candidates is candidates

        clone_backend = clone_storage_backend(host_backend)
        clone_context = make_train_context(clone_backend)
        cloned_factory = factory.clone_for_evaluation()
        cloned_installation = cloned_factory.build_installation(clone_context)
        cloned_builder = cloned_factory._runtime_builder
        cloned_registry = cloned_builder.bindings[-1].connector.registry

        assert cloned_registry.stable_key() == registry.stable_key()
        assert cloned_registry is not registry
        assert cloned_registry._trial_by_key is not registry._trial_by_key
        assert cloned_installation.stage4_runtime.candidates is not candidates
        assert cloned_installation.stage4_runtime.candidates.definition_graph \
            .ontology is clone_context.graph_ontology
    finally:
        relation_fixture.close()
        if clone_backend is not None:
            clone_backend.close()
        host_backend.close()


def test_active_factory_clones_discourse_declarations_for_v06():
    """active factory 必须把来源化篇章声明带入宿主和独立 V-06 connector。"""
    backend = DictBackend()
    clone_backend = None
    try:
        graphs, promotion, connector, _predicates, candidates = (
            _factory_fixture(backend, variant=255)
        )
        template = connector.registry.templates[0]
        hypothesis = _register(candidates, promotion, template)
        _recognize(
            candidates,
            hypothesis,
            template,
            source_id=755,
            event=755,
            stance="support",
        )
        selection = _selection_for_template(template, variant=255)
        declarations = StaticLanguageConnectorDiscourseDeclarations((
            LanguageConnectorDiscourseDeclaration(
                selection.selected_candidate_keys,
                (),
                _source(756),
                (_BASE + 78, 255, 1),
            ),
        ))
        factory = ActiveLanguageConnectorFactory(
            candidates,
            connector.runtime_policy,
            connector.surface_protocol,
            minimal_instruction_identity((_BASE + 78, 255, 2)),
            declarations,
        )
        host = factory.build(graphs.context)
        clone_backend = clone_storage_backend(backend)
        clone_context = clone_train_context(
            graphs.context,
            clone_backend,
            label="connector-discourse-declarations",
        )
        cloned_factory = factory.clone_for_evaluation()
        cloned = cloned_factory.build(clone_context)

        assert host.connector.discourse_declarations is declarations
        assert host.connector.structure_planner().plan(
            selection).discourse.topological_order == selection.selected_candidate_keys
        assert cloned.connector.discourse_declarations is not declarations
        assert cloned.connector.discourse_declarations.state_key() == (
            declarations.state_key())
        assert cloned.connector.structure_planner().plan(
            selection).discourse.topological_order == selection.selected_candidate_keys
        assert cloned_factory.state_key() == factory.state_key()
    finally:
        if clone_backend is not None:
            clone_backend.close()
        backend.close()


def test_v06_memory_connector_rebinds_clone_event_log_before_continuation():
    """V-06 connector 恢复后续写必须落到 clone event log，不能回流宿主。"""
    host_backend = DictBackend()
    clone_backend = None
    try:
        graphs, promotion, connector, _predicates, candidates = (
            _memory_factory_fixture(host_backend, variant=245)
        )
        template = connector.registry.templates[0]
        hypothesis = _register(candidates, promotion, template)
        _recognize(
            candidates, hypothesis, template, source_id=3, event=14,
            stance="support")
        factory = ActiveLanguageConnectorFactory(
            candidates,
            connector.runtime_policy,
            connector.surface_protocol,
            minimal_instruction_identity((_BASE + 24, 245, 1)),
        )
        host_before = host_backend.snapshot()
        clone_backend = clone_storage_backend(host_backend)
        clone_context = clone_train_context(
            graphs.context, clone_backend, label="memory-connector")
        cloned = factory.clone_for_evaluation().build(clone_context)

        assert cloned.candidates.memory_event_log is (
            clone_context.memory_read_events)
        assert cloned.candidates.memory_event_log is not (
            candidates.memory_event_log)
        _recognize(
            cloned.candidates,
            hypothesis,
            template,
            source_id=4,
            event=15,
            stance="refute",
            archive_refuted=True,
        )
        assert host_backend.snapshot() == host_before
        assert clone_backend.snapshot() != host_before
        assert candidates.active_templates() == (template,)
        assert cloned.candidates.active_templates() == ()
    finally:
        if clone_backend is not None:
            clone_backend.close()
        host_backend.close()


def test_production_factory_binds_current_owners_and_clones_without_sharing():
    """组合 factory 在宿主和克隆图分别重建 connector、S-07、R-01 和 stage4。"""
    host_backend = DictBackend()
    clone_backend = None
    relation_fixture = _r01_fixture()
    try:
        graphs, promotion, connector, _predicates, candidates = (
            _factory_fixture(host_backend, variant=212)
        )
        template = connector.registry.templates[0]
        hypothesis = _register(candidates, promotion, template)
        _recognize(
            candidates,
            hypothesis,
            template,
            source_id=3,
            event=13,
            stance="support",
        )
        policy = _production_stage4_policy(variant=212)
        connector_factory = ActiveLanguageConnectorFactory(
            candidates,
            connector.runtime_policy,
            connector.surface_protocol,
            policy.active_purpose,
        )
        builder = _ProductionBindingBuilder(
            graphs,
            relation_fixture.protocol,
            relation_fixture.closure.consumer.schemas,
            runtime_key=(_BASE + 38, 212),
        )
        factory = LanguageConnectorProductionFactory(
            connector_factory,
            builder,
            policy,
        )

        host_installation = factory.build_installation(graphs.context)
        host_binding = builder.bindings[-1]

        assert host_installation.runtime is host_binding.runtime
        assert host_installation.stage4_runtime.candidates is candidates
        assert host_binding.order_lifecycle.order_graph is graphs.order_graph
        assert host_binding.alias.closure.semantic_graph.ontology is (
            graphs.context.graph_ontology)
        cloned_factory = factory.clone_for_evaluation()
        assert cloned_factory is not factory
        assert cloned_factory.state_key() == factory.state_key()

        clone_backend = clone_storage_backend(host_backend)
        clone_context = make_train_context(clone_backend)
        clone_installation = cloned_factory.build_installation(clone_context)
        clone_builder = cloned_factory._runtime_builder
        clone_binding = clone_builder.bindings[-1]

        assert clone_installation.runtime is not host_installation.runtime
        assert clone_installation.runtime._mapper is not (
            host_installation.runtime._mapper)
        assert clone_installation.stage4_runtime is not (
            host_installation.stage4_runtime)
        assert clone_installation.stage4_runtime._processed is not (
            host_installation.stage4_runtime._processed)
        assert clone_installation.stage4_runtime.candidates is not candidates
        assert clone_installation.stage4_runtime.candidates.definition_graph.ontology is (
            clone_context.graph_ontology)
        assert clone_binding.order_lifecycle.order_graph.ontology is (
            clone_context.graph_ontology)
        assert clone_binding.alias is not host_binding.alias
        assert clone_binding.alias._uses is not host_binding.alias._uses
        assert clone_binding.alias.closure.semantic_graph.ontology is (
            clone_context.graph_ontology)
    finally:
        relation_fixture.close()
        if clone_backend is not None:
            clone_backend.close()
        host_backend.close()


@pytest.mark.parametrize(
    "mode,purpose_drift,match",
    (
        (_ProductionBindingBuilder.REPLACE_CONNECTOR, False,
         "未使用本次恢复的 connector"),
        (_ProductionBindingBuilder.DRIFT_ORDER_FACADE, False,
         "未使用本次恢复的 S-07 lifecycle"),
        (_ProductionBindingBuilder.FOREIGN_ALIAS, False,
         "R-01 owner 未绑定当前 context 图"),
        (_ProductionBindingBuilder.VALID, True,
         "purpose 与 stage4 policy 不一致"),
    ),
)
def test_production_factory_rejects_cross_owner_binding(
        mode, purpose_drift, match):
    """任一 connector、purpose、S-07 或 R-01 归属漂移都必须 fail closed。"""
    backend = DictBackend()
    relation_fixture = _r01_fixture()
    try:
        graphs, promotion, connector, _predicates, candidates = (
            _factory_fixture(backend, variant=213)
        )
        template = connector.registry.templates[0]
        hypothesis = _register(candidates, promotion, template)
        _recognize(
            candidates,
            hypothesis,
            template,
            source_id=3,
            event=14,
            stance="support",
        )
        production_purpose = minimal_instruction_identity(
            (_BASE + 39, 213, 1))
        policy = _production_stage4_policy(
            variant=213,
            active_purpose=(
                minimal_instruction_identity((_BASE + 39, 213, 2))
                if purpose_drift else production_purpose
            ),
        )
        builder = _ProductionBindingBuilder(
            graphs,
            relation_fixture.protocol,
            relation_fixture.closure.consumer.schemas,
            mode=mode,
            foreign_alias=relation_fixture.alias_runtime,
            runtime_key=(_BASE + 40, 213),
        )
        factory = LanguageConnectorProductionFactory(
            ActiveLanguageConnectorFactory(
                candidates,
                connector.runtime_policy,
                connector.surface_protocol,
                production_purpose,
            ),
            builder,
            policy,
        )

        with pytest.raises(ValueError, match=match):
            factory.build_installation(graphs.context)
    finally:
        relation_fixture.close()
        backend.close()


def test_connector_rejects_theory_or_candidate_definition_drift():
    """active lifecycle 任一侧出现竞争拓扑都必须拒绝采用，不按稳定序私选。"""
    theory_backend = DictBackend()
    try:
        _graphs, promotion, template, _predicates, _graph, runtime = (
            _candidate_fixture(theory_backend, variant=205)
        )
        hypothesis = _register(runtime, promotion, template)
        _recognize(
            runtime,
            hypothesis,
            template,
            source_id=3,
            event=8,
            stance="support",
        )
        binding = template.bindings[0]
        runtime.definition_graph.ontology.relate(
            runtime.definition_graph.predicates.binding_source,
            runtime.definition_graph.ontology.resolve(binding.binding),
            runtime.definition_graph.ontology.materialize(
                concept_identity((_BASE + 7, 205))),
            scope=promotion.constraint.scope,
            provenance_kind=31,
            epistemic_origin=32,
            content_version=33,
            qualifiers=(34,),
        )

        with pytest.raises(Exception, match="binding source"):
            runtime.active_templates()
    finally:
        theory_backend.close()

    candidate_backend = DictBackend()
    try:
        _graphs, promotion, template, _predicates, _graph, runtime = (
            _candidate_fixture(candidate_backend, variant=206)
        )
        hypothesis = _register(runtime, promotion, template)
        _recognize(
            runtime,
            hypothesis,
            template,
            source_id=3,
            event=9,
            stance="support",
        )
        definition = runtime.mapper.definition(
            template,
            (_source(1), _source(2)),
        )
        binding = next(
            item for item in definition.bindings
            if item.predicate == runtime.protocol.definition_member_predicate
        )
        runtime.learning.graph.ontology.relate(
            runtime.learning.graph.ontology.resolve(binding.predicate),
            runtime.learning.graph.ontology.resolve(definition.candidate),
            runtime.learning.graph.ontology.materialize(
                concept_identity((_BASE + 8, 206))),
            scope=hypothesis.scope,
            provenance_kind=runtime.learning.metadata.provenance_kind,
            epistemic_origin=runtime.learning.metadata.epistemic_origin,
            content_version=runtime.learning.metadata.content_version,
            qualifiers=(binding.ordinal, *runtime.learning.metadata.qualifiers),
        )

        with pytest.raises(Exception, match="竞争端点|定义外"):
            runtime.active_templates()
    finally:
        candidate_backend.close()


def test_connector_registration_preflight_and_faults_never_create_active_use(
        monkeypatch):
    """契约冲突零写；理论或 forming 写后故障也不能留下 active 候选。"""
    conflict_backend = DictBackend()
    try:
        _graphs, promotion, template, _predicates, _graph, runtime = (
            _candidate_fixture(conflict_backend, variant=207)
        )
        definition = runtime.mapper.definition(
            template,
            (_source(1), _source(2)),
        )
        binding = definition.bindings[0]
        runtime.learning.graph.ontology.relate(
            runtime.learning.graph.ontology.materialize(binding.predicate),
            runtime.learning.graph.ontology.materialize(definition.candidate),
            runtime.learning.graph.ontology.materialize(binding.value),
            scope=definition.hypothesis(runtime.learning.engine.protocol).scope,
            provenance_kind=runtime.learning.metadata.provenance_kind,
            epistemic_origin=runtime.learning.metadata.epistemic_origin,
            content_version=runtime.learning.metadata.content_version,
            qualifiers=(binding.ordinal, *runtime.learning.metadata.qualifiers),
        )
        before = conflict_backend.snapshot()

        with pytest.raises(Exception, match="部分图拓扑"):
            _register(runtime, promotion, template)

        assert conflict_backend.snapshot() == before
        assert runtime.active_templates() == ()
    finally:
        conflict_backend.close()

    theory_only_backend = DictBackend()
    try:
        _graphs, promotion, template, _predicates, _graph, runtime = (
            _candidate_fixture(theory_only_backend, variant=208)
        )

        def fail_learning(*args, **kwargs):
            """模拟双侧预检通过后的 candidate runtime 运行故障。"""
            raise RuntimeError("injected candidate failure")

        monkeypatch.setattr(runtime.learning, "register", fail_learning)
        with pytest.raises(RuntimeError, match="injected candidate failure"):
            _register(runtime, promotion, template)

        assert runtime.definition_graph.read(template.connector).definition == template
        assert runtime.active_templates() == ()
        assert runtime.learning.engine.state_key()[0] == ()
    finally:
        theory_only_backend.close()

    forming_only_backend = DictBackend()
    try:
        _graphs, promotion, template, _predicates, _graph, runtime = (
            _candidate_fixture(forming_only_backend, variant=209)
        )

        def fail_engine(*args, **kwargs):
            """模拟 candidate 图定义写入后的 H-00 运行故障。"""
            raise RuntimeError("injected H-00 failure")

        monkeypatch.setattr(runtime.learning.engine, "register", fail_engine)
        with pytest.raises(RuntimeError, match="injected H-00 failure"):
            _register(runtime, promotion, template)

        assert runtime.definition_graph.read(template.connector).definition == template
        assert runtime.active_templates() == ()
        assert runtime.learning.engine.state_key()[0] == ()
    finally:
        forming_only_backend.close()


def test_connector_stage4_support_consumes_real_typed_signals_and_refreshes():
    """全 support route 从真实 G-04 信号形成 prediction/Evidence/decision/refresh。"""
    (backend, alias, episode, candidates, template, _promotion, _hypothesis,
     _connector) = (
        _typed_episode_candidate_fixture(variant=220)
    )
    try:
        before = candidates.learning.projection_for_candidate(
            template.connector)
        runtime = LanguageConnectorStage4Runtime(
            candidates,
            _stage4_policy(
                episode,
                variant=220,
            ),
        )

        report = runtime.apply((episode,))

        assert report.complete is True
        assert report.changed_count == 1
        assert len(report.outcomes) == 1
        outcome = report.outcomes[0]
        assert outcome.learning.prediction.predicted == template.connector
        assert outcome.learning.evidence == (
            candidates.learning.engine.ledger.evidence_history(
                outcome.learning.prediction.hypothesis
            )[-1]
        )
        assert outcome.learning.decision == (
            candidates.learning.engine.resolver.decision_history(
                outcome.learning.prediction.hypothesis
            )[-1]
        )
        assert outcome.learning.projection is not None
        assert outcome.learning.projection.state == (
            candidates.learning.graph.protocol.active_state)
        assert len(outcome.learning.projection.history) == len(before.history) + 1
        assert candidates.active_templates() == (template,)
        assert runtime.apply((episode,)) == report
    finally:
        alias.close()
        backend.close()


def test_connector_training_history_restores_stage4_without_memory(
        tmp_path):
    """PH2 Core 历史跨 backend 恢复 stage4，且全程不写断奶后 Memory。"""
    host_backend = None
    target_backend = SQLiteBackend()
    alias = None
    try:
        (actual_backend, alias, episode, candidates, template, _promotion,
         _hypothesis, _connector) = _typed_episode_candidate_fixture(
             variant=250,
             training_history=True,
         )
        host_backend = actual_backend
        assert candidates.persistence_kind == CANDIDATE_PERSISTENCE_TRAINING
        assert host_backend.count(MEMORY_EVENT_TABLE) == 0
        policy = _stage4_policy(
            episode,
            variant=250,
        )
        runtime = LanguageConnectorStage4Runtime(candidates, policy)
        report = runtime.apply((episode,))
        state_after_apply = runtime.state_key()
        training_rows = host_backend.count(TRAINING_CANDIDATE_EVENT_TABLE)
        assert report.complete is True
        assert training_rows > 0
        assert host_backend.count(MEMORY_EVENT_TABLE) == 0

        core_space_id = candidates.definition_graph.ontology.space_id
        dump_run(
            host_backend,
            str(tmp_path),
            "connector_training_stage4",
            spaces=[core_space_id],
            tables=DUMP_TABLES,
        )
        target_context = make_train_context(target_backend)
        assert load_run(
            target_backend,
            str(tmp_path),
            "connector_training_stage4",
        ) == [core_space_id]
        target_context.graph_ontology.clear_runtime_caches()
        restored_candidates = _restore_training_candidate(
            candidates,
            target_context,
        )
        restored_runtime = LanguageConnectorStage4Runtime(
            restored_candidates,
            policy,
        )

        assert restored_candidates.persistence_kind == (
            CANDIDATE_PERSISTENCE_TRAINING)
        assert restored_runtime.state_key() == state_after_apply
        target_rows = target_backend.count(TRAINING_CANDIDATE_EVENT_TABLE)
        replayed = restored_runtime.apply((episode,))
        assert replayed.complete is True
        assert replayed.outcomes[0].connector == template
        assert target_backend.count(
            TRAINING_CANDIDATE_EVENT_TABLE) == target_rows
        assert target_backend.count(MEMORY_EVENT_TABLE) == 0

        execution = episode.production.execution
        postcheck = episode.production.postcheck
        assert execution is not None and execution.surface is not None
        assert postcheck is not None
        candidate_keys = {
            item.candidate_key
            for item in execution.surface.preview.request.structure
            .propositions.propositions
        }
        changed = False
        changed_results = []
        for item in postcheck.report.results:
            if (not changed
                    and item.applicability == APPLICABILITY_APPLICABLE
                    and candidate_keys.issubset(item.claim_keys)):
                changed_results.append(replace(
                    item,
                    detail=(*item.detail, _BASE + 250),
                ))
                changed = True
            else:
                changed_results.append(item)
        assert changed
        changed_postcheck = replace(
            postcheck,
            report=VerificationReport(
                True,
                tuple(changed_results),
            ),
        )
        changed_episode = TypedLanguageEpisode.from_production(
            episode.round_id,
            episode.source,
            episode.scope,
            replace(episode.production, postcheck=changed_postcheck),
            read_only=False,
        )
        with pytest.raises(RuntimeError, match="消费内容漂移"):
            restored_runtime.apply((changed_episode,))
        assert target_backend.count(
            TRAINING_CANDIDATE_EVENT_TABLE) == target_rows
    finally:
        if alias is not None:
            alias.close()
        if host_backend is not None:
            host_backend.close()
        target_backend.close()


def test_connector_stage4_event_namespace_is_not_prefix_matched():
    """较短 namespace 不得接管较长 namespace 已保存的 stage4 历史。"""
    (backend, alias, episode, candidates, _template, _promotion,
     _hypothesis, _connector) = _typed_episode_candidate_fixture(
         variant=251,
         training_history=True,
     )
    try:
        policy = _stage4_policy(
            episode,
            variant=251,
        )
        longer = replace(
            policy,
            event_namespace=(*policy.event_namespace, _BASE + 251),
        )
        report = LanguageConnectorStage4Runtime(
            candidates,
            longer,
        ).apply((episode,))
        assert report.complete is True

        shorter_runtime = LanguageConnectorStage4Runtime(candidates, policy)

        assert shorter_runtime.state_key()[2] == ()
    finally:
        alias.close()
        backend.close()


def test_typed_episode_merges_g04_and_supplemental_verifier_dimensions():
    """同次 G-04 与领域 verifier 逐维合并，重复维度不得被静默覆盖。"""
    (backend, alias, episode, _candidates, _template, _promotion, _hypothesis,
     _connector) = _typed_episode_candidate_fixture(variant=226)
    try:
        extra = VerificationResult(
            ProtocolKey((_BASE + 40, 1)),
            ProtocolKey((_BASE + 40, 2)),
            APPLICABILITY_APPLICABLE,
            VERDICT_SUPPORT,
            claim_keys=((_BASE + 40, 3),),
            detail=(_BASE + 40, 4),
            source=episode.source,
            scope=episode.scope,
        )
        supplemental = VerificationReport(False, (extra,))

        merged = TypedLanguageEpisode.from_production(
            episode.round_id,
            episode.source,
            episode.scope,
            episode.production,
            read_only=False,
            supplemental_verification=supplemental,
        )

        assert merged.supplemental_verification is supplemental
        assert len(merged.signals) == len(episode.signals) + 1
        assert any(
            signal.dimension == extra.dimension
            and signal.verifier == extra.verifier
            and signal.verdict == extra.verdict
            for signal in merged.signals
        )

        duplicate = VerificationReport(False, (
            replace(extra, dimension=episode.signals[0].dimension),
        ))
        with pytest.raises(ValueError, match="不得重复 reward dimension"):
            TypedLanguageEpisode.from_production(
                episode.round_id,
                episode.source,
                episode.scope,
                episode.production,
                read_only=False,
                supplemental_verification=duplicate,
            )
    finally:
        alias.close()
        backend.close()


def test_connector_stage4_refute_consumes_real_typed_signals_and_deactivates():
    """任一显式 refute route 形成定向反驳，并把 active connector 降为 inactive。"""
    (backend, alias, episode, candidates, template, _promotion, _hypothesis,
     _connector) = (
        _typed_episode_candidate_fixture(variant=221)
    )
    try:
        episode = _episode_with_connector_verdict(episode, VERDICT_REFUTE)
        runtime = LanguageConnectorStage4Runtime(
            candidates,
            _stage4_policy(
                episode,
                variant=221,
            ),
        )

        report = runtime.apply((episode,))

        assert report.complete is True
        outcome = report.outcomes[0]
        assert outcome.learning.projection is not None
        assert outcome.learning.projection.state == (
            candidates.learning.graph.protocol.inactive_state)
        assert outcome.learning.verification.stance == outcome.stance
        assert candidates.active_templates() == ()
        with pytest.raises(LanguageConnectorCandidateError, match="没有 active"):
            candidates.active_registry()
    finally:
        alias.close()
        backend.close()


def test_connector_stage4_aggregates_same_hypothesis_before_single_transition():
    """同一 Hypothesis 的 support/refute 批只形成一次顺序无关的反驳转换。"""
    (backend, alias, support_episode, candidates, template, _promotion,
     _hypothesis, _connector) = _typed_episode_candidate_fixture(variant=224)
    try:
        refute_episode = _episode_with_connector_verdict(
            support_episode,
            VERDICT_REFUTE,
            round_id=support_episode.round_id + 1,
        )
        runtime = LanguageConnectorStage4Runtime(
            candidates,
            _stage4_policy(
                support_episode,
                variant=224,
            ),
        )
        before = candidates.learning.projection_for_candidate(
            template.connector)

        report = runtime.apply((support_episode, refute_episode))

        assert report.complete is True
        assert report.changed_count == 1
        assert len(report.outcomes) == 1
        outcome = report.outcomes[0]
        assert outcome.stance == EVIDENCE_REFUTE
        assert outcome.learning.projection is not None
        assert len(outcome.learning.projection.history) == len(before.history) + 1
        assert candidates.active_templates() == ()
        assert runtime.apply((refute_episode, support_episode)) == report
    finally:
        alias.close()
        backend.close()


def test_connector_stage4_requires_surface_theory_attribution():
    """未由 lifecycle-aware factory 归属的 surface 不得产生候选写入。"""
    (backend, alias, episode, candidates, _template, _promotion, _hypothesis,
     _connector) = (
        _typed_episode_candidate_fixture(variant=222, attributed=False)
    )
    try:
        runtime = LanguageConnectorStage4Runtime(
            candidates,
            _stage4_policy(
                episode,
                variant=222,
            ),
        )
        before = candidates.state_key()
        with pytest.raises(ValueError, match="缺少显式理论归属"):
            runtime.apply((episode,))
        assert candidates.state_key() == before
    finally:
        alias.close()
        backend.close()


def test_connector_stage4_rejects_stale_attribution_after_replacement():
    """同 match key 的新理论 active 后不得继承旧 connector episode 的反馈。"""
    (backend, alias, episode, candidates, template, promotion, hypothesis,
     _connector) = (
        _typed_episode_candidate_fixture(variant=223)
    )
    try:
        replacement = _replacement_template(template, variant=223)
        replacement_hypothesis = _register(
            candidates,
            promotion,
            replacement,
        )
        _recognize(
            candidates,
            replacement_hypothesis,
            replacement,
            source_id=800,
            event=800,
            stance="support",
        )
        _recognize(
            candidates,
            hypothesis,
            template,
            source_id=801,
            event=801,
            stance="refute",
            replacement=replacement_hypothesis,
        )
        runtime = LanguageConnectorStage4Runtime(
            candidates,
            _stage4_policy(
                episode,
                variant=223,
            ),
        )
        before = candidates.state_key()
        with pytest.raises(ValueError, match="理论归属漂移"):
            runtime.apply((episode,))
        assert candidates.state_key() == before
    finally:
        alias.close()
        backend.close()


def test_connector_trial_support_promotes_exact_forming_hypothesis():
    """显式 trial 的独立 support 首次晋升 forming，随后只能走 active production。"""
    (backend, alias, episode, candidates, template, _promotion, hypothesis,
     connector) = _typed_episode_candidate_fixture(variant=224, trial=True)
    try:
        attribution = episode.production.execution.surface.preview.request \
            .attribution
        assert attribution.hypothesis == hypothesis
        assert attribution.purpose == minimal_instruction_identity(
            (_BASE + 33, 224, 2))
        assert candidates.active_templates() == ()

        report = LanguageConnectorStage4Runtime(
            candidates,
            _stage4_policy(
                episode,
                variant=224,
            ),
        ).apply((episode,))

        assert report.complete is True
        assert report.outcomes[0].learning.projection.state == (
            candidates.learning.graph.protocol.active_state)
        assert candidates.active_templates() == (template,)
        with pytest.raises(
                LanguageConnectorCandidateError,
                match="已有 lifecycle Event"):
            candidates.trial_template(hypothesis)
    finally:
        alias.close()
        backend.close()


def test_connector_trial_refute_archives_without_active_projection():
    """forming trial 被反驳时保留 H-00 证据但不得形成 active 图采用。"""
    (backend, alias, episode, candidates, _template, _promotion, hypothesis,
     _connector) = _typed_episode_candidate_fixture(variant=225, trial=True)
    try:
        episode = _episode_with_connector_verdict(episode, VERDICT_REFUTE)
        report = LanguageConnectorStage4Runtime(
            candidates,
            _stage4_policy(
                episode,
                variant=225,
            ),
        ).apply((episode,))

        assert report.complete is False
        assert report.changed_count == 0
        assert report.outcomes[0].learning.projection is None
        assert candidates.active_templates() == ()
        with pytest.raises(
                LanguageConnectorCandidateError,
                match="未采用且未归档"):
            candidates.trial_template(hypothesis)
    finally:
        alias.close()
        backend.close()


def test_connector_stage4_repeated_template_keeps_sentence_use_contexts_and_one_evidence():
    """同模板两次出现必须各有句实例和 Use context，但只形成一次理论 Evidence。"""
    (backend, alias, episode, candidates, templates, hypotheses, _connector,
     _declarations) = _multi_connector_episode_fixture(
         variant=252,
         same_template=True,
         training_history=True,
     )
    try:
        execution = episode.production.execution
        assert execution is not None and execution.surface is not None
        request = execution.surface.preview.request
        instances = tuple(
            item.instance for item in request.structure.syntax.sentences)
        assert len(instances) == 2
        assert instances[0] != instances[1]
        assert instances[0].template == instances[1].template
        assert request.attribution is None
        assert tuple(item.sentence for item in request.sentence_attributions) == (
            instances)
        assert {item.hypothesis for item in request.sentence_attributions} == {
            hypotheses[0]}
        uses = tuple(alias.runtime._uses.values())
        assert uses
        contexts = tuple(item.context for item in uses)
        assert all(item is not None for item in contexts)
        assert {item.sentence_instance_key for item in contexts} == {
            item.stable_key() for item in instances}
        assert {item.connector_hypothesis for item in contexts} == {
            hypotheses[0]}

        policy = _stage4_policy(episode, variant=252)
        before = len(candidates.learning.engine.ledger.evidence_history(
            hypotheses[0]))
        runtime = LanguageConnectorStage4Runtime(candidates, policy)
        report = runtime.apply((episode,))

        assert report.complete is True
        assert len(report.outcomes) == 1
        assert report.outcomes[0].connector == templates[0]
        assert len(candidates.learning.engine.ledger.evidence_history(
            hypotheses[0])) == before + 1
        assert runtime.apply((episode,)) == report
        restored = LanguageConnectorStage4Runtime(candidates, policy)
        assert restored.apply((episode,)) == report
    finally:
        alias.close()
        backend.close()


def test_connector_stage4_separates_multi_template_hypotheses_and_rejects_missing_claim():
    """两套理论分别结算，任一被路由 signal 缺当前 candidate claim 必须整批零写失败。"""
    (backend, alias, episode, candidates, templates, hypotheses, _connector,
     _declarations) = _multi_connector_episode_fixture(
         variant=253,
         same_template=False,
         training_history=True,
     )
    try:
        execution = episode.production.execution
        assert execution is not None and execution.surface is not None
        request = execution.surface.preview.request
        assert len(request.structure.syntax.sentences) == 2
        assert {item.theory for item in request.sentence_attributions} == {
            item.connector for item in templates}
        assert {item.hypothesis for item in request.sentence_attributions} == (
            set(hypotheses))
        uses = tuple(alias.runtime._uses.values())
        assert {item.context.connector_hypothesis for item in uses} == set(
            hypotheses)

        policy = _stage4_policy(episode, variant=253)
        before = candidates.state_key()
        missing_claim = _episode_without_route_candidate_claim(
            episode,
            policy,
            request.sentence_attributions[0].sentence.candidate_key,
        )
        with pytest.raises(ValueError, match="未显式 claim 当前 connector candidate"):
            LanguageConnectorStage4Runtime(candidates, policy).apply((
                missing_claim,
            ))
        assert candidates.state_key() == before

        runtime = LanguageConnectorStage4Runtime(candidates, policy)
        report = runtime.apply((episode,))
        assert report.complete is True
        assert len(report.outcomes) == 2
        assert {item.connector for item in report.outcomes} == set(templates)
        assert {
            item.learning.prediction.hypothesis for item in report.outcomes
        } == set(hypotheses)
        restored = LanguageConnectorStage4Runtime(candidates, policy)
        assert restored.apply((episode,)) == report
    finally:
        alias.close()
        backend.close()


def test_connector_stage4_keeps_cross_knowledge_sources_under_one_query_feedback():
    """跨知识来源的句子保留各自来源，但 G-04 与 R-01 一律绑定本次 query。"""
    (backend, alias, episode, candidates, templates, hypotheses, _connector,
     _declarations) = _multi_connector_episode_fixture(
         variant=255,
         same_template=False,
         cross_source=True,
         training_history=True,
     )
    try:
        execution = episode.production.execution
        assert execution is not None and execution.surface is not None
        request = execution.surface.preview.request
        goal = request.structure.selection.request.goal
        sentences = request.structure.syntax.sentences
        assert len(sentences) == 2
        assert {item.source for item in sentences} != {goal.source}
        assert any(item.source != goal.source for item in sentences)
        assert all(item.scope == goal.scope for item in sentences)
        assert all(
            item.source == item.instance.source
            and item.scope == item.instance.scope
            for item in sentences
        )
        policy = _stage4_policy(episode, variant=255)
        routed_keys = {
            (item.dimension, item.verifier) for item in policy.routes}
        routed_signals = tuple(
            item for item in episode.signals
            if (item.dimension, item.verifier) in routed_keys
        )
        assert len(routed_signals) == len(policy.routes)
        assert all(
            signal.source == goal.source and signal.scope == goal.scope
            for signal in routed_signals
        )

        uses = tuple(alias.runtime._uses.values())
        assert uses
        assert all(
            item.context is not None
            and item.context.source == goal.source
            and item.context.scope == goal.scope
            for item in uses
        )
        assert {item.context.sentence_instance_key for item in uses} == {
            item.instance.stable_key() for item in sentences
        }

        before = candidates.state_key()
        cross_knowledge_source = next(
            item.source for item in sentences
            if item.source != goal.source
        )
        drifted_results = tuple(
            replace(result, source=cross_knowledge_source)
            if (result.dimension == policy.routes[0].dimension
                and result.verifier == policy.routes[0].verifier)
            else result
            for result in episode.production.postcheck.report.results
        )
        drifted_production = replace(
            episode.production,
            postcheck=replace(
                episode.production.postcheck,
                report=replace(
                    episode.production.postcheck.report,
                    results=drifted_results,
                ),
            ),
        )
        drifted = TypedLanguageEpisode.from_production(
            episode.round_id,
            episode.source,
            episode.scope,
            drifted_production,
            read_only=False,
        )
        with pytest.raises(ValueError, match="未绑定当前 generation query"):
            LanguageConnectorStage4Runtime(candidates, policy).apply((
                drifted,
            ))
        assert candidates.state_key() == before

        runtime = LanguageConnectorStage4Runtime(candidates, policy)
        report = runtime.apply((episode,))
        assert report.complete is True
        assert {item.connector for item in report.outcomes} == set(templates)
        assert {
            item.learning.prediction.hypothesis for item in report.outcomes
        } == set(hypotheses)
        restored = LanguageConnectorStage4Runtime(candidates, policy)
        assert restored.apply((episode,)) == report
    finally:
        alias.close()
        backend.close()


def test_connector_stage4_accepts_v06_owner_only_query_scope():
    """真实多命题链在 V-06 独立 owner scope 下仍按知识来源归属 Use 与反馈。"""
    evaluation_owner = OwnerScope(
        1,
        1,
        255,
        VISIBILITY_SESSION,
    )
    (backend, alias, episode, candidates, templates, hypotheses, _connector,
     _declarations) = _multi_connector_episode_fixture(
         variant=256,
         same_template=False,
         cross_source=True,
         training_history=True,
         runtime_owner=evaluation_owner,
     )
    try:
        execution = episode.production.execution
        assert execution is not None and execution.surface is not None
        request = execution.surface.preview.request
        goal = request.structure.selection.request.goal
        assert episode.scope.owner == evaluation_owner
        assert episode.scope.source is None
        assert goal.scope.owner == evaluation_owner
        assert goal.scope.source is None
        assert goal.source == episode.source

        uses = tuple(alias.runtime._uses.values())
        assert uses
        assert all(
            item.context is not None
            and item.context.source == goal.source
            and item.context.scope == goal.scope
            for item in uses
        )
        assert {
            item.context.sentence_instance_key for item in uses
        } == {
            item.instance.stable_key()
            for item in request.structure.syntax.sentences
        }
        use_owner = alias.closure.use_owner
        assert use_owner is not None
        materialized_uses = use_owner.history()
        assert materialized_uses
        assert all(
            semantic_source(item.event) == goal.source
            and item.definition.context.scope == goal.scope
            for item in materialized_uses
        )
        assert all(
            statement.assertion.scope == goal.scope
            for item in materialized_uses
            for statement in alias.semantic_graph.ontology.statements(
                subject=item.event_ref)
        )

        policy = _stage4_policy(episode, variant=256)
        runtime = LanguageConnectorStage4Runtime(candidates, policy)
        report = runtime.apply((episode,))
        assert report.complete is True
        assert {item.connector for item in report.outcomes} == set(templates)
        assert {
            item.learning.prediction.hypothesis for item in report.outcomes
        } == set(hypotheses)
        restored = LanguageConnectorStage4Runtime(candidates, policy)
        assert restored.apply((episode,)) == report
    finally:
        alias.close()
        backend.close()


def test_connector_multi_sentence_rejects_missing_duplicate_and_nonunique_declarations():
    """多命题没有唯一来源化篇章声明时不得从 candidate 或稳定键推断句序。"""
    (backend, alias, episode, _candidates, _templates, _hypotheses, connector,
     _declarations) = _multi_connector_episode_fixture(
         variant=254,
         same_template=True,
     )
    try:
        execution = episode.production.execution
        assert execution is not None and execution.surface is not None
        selection = execution.surface.preview.request.structure.selection
        attributions = tuple(connector.attribution_mapper.attributions.values())
        without_declaration = LanguageGenerationConnector(
            connector.registry,
            connector.runtime_policy,
            connector.surface_protocol,
            attributions,
        )
        with pytest.raises(
                LanguageGenerationConnectorError,
                match="必须注入来源化篇章声明"):
            without_declaration.structure_planner().plan(selection)

        unordered = LanguageConnectorDiscourseDeclaration(
            selection.selected_candidate_keys,
            (),
            _source(730 + 254),
            (_BASE + 76, 254, 1),
        )
        with pytest.raises(ValueError, match="未唯一确定句间顺序"):
            LanguageGenerationConnector(
                connector.registry,
                connector.runtime_policy,
                connector.surface_protocol,
                attributions,
                StaticLanguageConnectorDiscourseDeclarations((unordered,)),
            ).structure_planner().plan(selection)
        with pytest.raises(ValueError, match="重复声明篇章顺序"):
            StaticLanguageConnectorDiscourseDeclarations((
                unordered,
                unordered,
            ))
    finally:
        alias.close()
        backend.close()


def test_connector_stage4_route_rejects_non_applicable_and_reinterpreted_outcomes():
    """connector route 只能消费原生 applicable/support 或 applicable/refute，不能重解释。"""
    dimension = ProtocolKey((_BASE + 77, 1))
    verifier = ProtocolKey((_BASE + 77, 2))
    with pytest.raises(ValueError, match="只能接受 applicable/support"):
        LanguageConnectorSignalRoute(
            dimension,
            verifier,
            ((APPLICABILITY_NOT_APPLICABLE, VERDICT_SUPPORT),),
            ((APPLICABILITY_APPLICABLE, VERDICT_REFUTE),),
        )
    with pytest.raises(ValueError, match="只能接受 applicable/refute"):
        LanguageConnectorSignalRoute(
            dimension,
            verifier,
            ((APPLICABILITY_APPLICABLE, VERDICT_SUPPORT),),
            ((APPLICABILITY_UNKNOWN, VERDICT_SUPPORT),),
        )
