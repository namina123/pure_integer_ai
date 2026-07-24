"""训练上下文类型和存储装配入口。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.cognition.shared.memory_overlay import (
    CoreIdentityCatalog,
    MemoryOverlay,
)
from pure_integer_ai.cognition.shared.memory_event_log import MemoryEventLog
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MemoryHypothesisAggregateIndex,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    SCOPE_DOCUMENT,
    document_scope as source_document_scope,
    episode_scope,
    make_scope,
)
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.cognition.shared.types import JudgeWeights, WEANING_PRE
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingCandidateHistoryLog,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.node_store import NodeStore
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_CORE,
    SpaceRegistry,
)
from pure_integer_ai.storage.word_form_index import register_word_form_index
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.corpus_identity import ensure_item_scope

if TYPE_CHECKING:
    from pure_integer_ai.experiments.evaluation_protocol import EvaluationPlan
    from pure_integer_ai.experiments.verification_orchestration import (
        VerificationReport,
    )
    from pure_integer_ai.teacher.probe_set import ProbeSet


@dataclass
class TrainContext:
    """集中保存一次训练运行的存储、图、学习协议和有界工作状态。"""

    backend: StorageBackend
    core_space: AbstractSpace
    edge_store: EdgeStore
    node_store: NodeStore
    concept_index: ConceptIndex
    concept_graph: ConceptGraph
    scoped_identity_store: ScopedIdentityStore
    graph_ontology: GraphOntology
    training_candidate_history: TrainingCandidateHistoryLog | None = None
    core_identity_catalog: CoreIdentityCatalog | None = None
    memory_read_overlay: MemoryOverlay | None = None
    memory_interact_overlay: MemoryOverlay | None = None
    memory_read_events: MemoryEventLog | None = None
    memory_interact_events: MemoryEventLog | None = None
    memory_read_aggregates: MemoryHypothesisAggregateIndex | None = None
    memory_interact_aggregates: MemoryHypothesisAggregateIndex | None = None
    memory_query_runtime: Any = None
    memory_resolver_runtime: Any = None
    memory_hot_set_runtime: Any = None
    attractor_runtime: Any = None
    memory_use_runtime: Any = None
    memory_maintenance_runtime: Any = None
    memory_read_intake: Any = None
    memory_interact_intake: Any = None
    tiered_segment_store: Any = None
    memory_batch_config: Any = None
    memory_batch_visibility: Any = None
    memory_batch_coordinator: Any = None
    memory_read_batch_runtime: Any = None
    memory_interact_batch_runtime: Any = None
    memory_forget_visibility: Any = None
    memory_isolation_runtime: Any = None
    unicode_intake: Any = None
    word_form_providers: Any = None
    word_form_course_report: Any = None
    occurrence_index: Any = None
    occurrence_order_reader: Any = None
    occurrence_order_writer: Any = None
    precedence_relation_runtime: Any = None
    precedence_relation_reports: list[Any] = field(default_factory=list)
    causal_relation_runtime: Any = None
    causal_relation_reports: list[Any] = field(default_factory=list)
    set_relation_runtime: Any = None
    set_relation_reports: list[Any] = field(default_factory=list)
    property_relation_runtime: Any = None
    property_relation_reports: list[Any] = field(default_factory=list)
    mereology_relation_runtime: Any = None
    mereology_relation_reports: list[Any] = field(default_factory=list)
    semantic_pair_runtime: Any = None
    semantic_pair_reports: list[Any] = field(default_factory=list)
    language_semantic_course_runtime: Any = None
    language_semantic_course_reports: list[Any] = field(default_factory=list)
    alias_relation_course_report: Any = None
    language_generation_course_report: Any = None
    language_generation_postcheck_course_report: Any = None
    language_generation_runtime_factory: Any = None
    language_generation_runtime: Any = None
    language_generation_stage4_runtime: Any = None
    span_index: Any = None
    segmentation_span_materializer: Any = None
    boundary_hypothesis_engine: Any = None
    boundary_span_materializer: Any = None
    language_prediction_runtime: Any = None
    language_prediction_reports: list[Any] = field(default_factory=list)
    candidate_projection_graph: Any = None
    structure_candidate_runtime: Any = None
    structure_candidate_consumer: Any = None
    structure_candidate_mapper: Any = None
    structure_candidate_reports: list[Any] = field(default_factory=list)
    structure_boundary_evidence_mapper: Any = None
    structure_boundary_report: Any = None
    sense_candidate_runtime: Any = None
    sense_candidate_consumer: Any = None
    sense_candidate_course_runtime: Any = None
    sense_candidate_reports: list[Any] = field(default_factory=list)
    verification_reports: list[VerificationReport] = field(
        default_factory=list)
    teacher: Any = None
    weights: JudgeWeights = field(default_factory=JudgeWeights)
    work_memory: WorkMemory = field(default_factory=WorkMemory)
    weaning_phase: int = WEANING_PRE
    judge_source_id: int | None = None
    judge_source_independent: bool = False
    evaluation_plan: EvaluationPlan | None = None
    evaluation_corpora: dict[Any, list] = field(default_factory=dict)
    probe_set: ProbeSet | None = None
    probe_corpus: list = field(default_factory=list)
    probe_content_disjoint: bool = False
    evaluation_strictly_isolated: bool = False
    probe_set_disjoint: bool = False
    holdout_retention: int = 0
    e2_eval_passed: bool = False
    memory_read: Any = None
    memory_interact: Any = None
    scope_owner: Any = None
    position_histogram_state: Any = None
    hub_degree_state: Any = None

    def __post_init__(self) -> None:
        """为每个训练或评测上下文装配独立的派生统计状态。"""
        if self.position_histogram_state is None:
            from pure_integer_ai.cognition.understanding.emergent_role import (
                PositionHistogramState,
            )
            self.position_histogram_state = PositionHistogramState(self.backend)
        if self.hub_degree_state is None:
            from pure_integer_ai.cognition.shared.hub_detect import HubDegreeState
            self.hub_degree_state = HubDegreeState(self.edge_store)

    @property
    def space_id(self) -> int:
        """返回当前核心空间的运行时整数标识。"""
        return self.core_space.space_id


def make_train_context(
        backend: StorageBackend, *, teacher: Any = None,
        weights: JudgeWeights | None = None,
        companion: bool = False,
        ) -> TrainContext:
    """注册训练所需表并装配核心、伴随和两层记忆空间。"""
    bootstrap(backend)

    from pure_integer_ai.cognition.understanding.emergent_role import (
        register_position_hist,
    )
    from pure_integer_ai.cognition.understanding.modification_direction import (
        register_modification_hist,
    )
    from pure_integer_ai.storage.abstract_mark import register_abstract_mark
    from pure_integer_ai.storage.chapter_seq import register_chapter_seq
    from pure_integer_ai.storage.composes_attr import register_composes_attr
    from pure_integer_ai.storage.concept_correspondence import (
        register_concept_correspondence,
    )
    from pure_integer_ai.storage.concept_identity import register_concept_identity
    from pure_integer_ai.storage.experience_count import register_experience_count
    from pure_integer_ai.storage.op_confidence import register_op_confidence
    from pure_integer_ai.storage.pronoun_resolution_count import (
        register_pronoun_resolution_count,
    )
    from pure_integer_ai.storage.selection_pref_count import (
        register_selection_pref_count,
    )
    from pure_integer_ai.storage.sense_candidates import register_sense_candidates
    from pure_integer_ai.storage.spaces.memory_space import MemorySpace
    from pure_integer_ai.storage.structure_match_count import (
        register_structure_match_count,
    )
    from pure_integer_ai.teacher.weaning_calibration import (
        register_weaning_calibration,
    )

    register_position_hist(backend)
    register_weaning_calibration(backend)
    register_composes_attr(backend)
    register_op_confidence(backend)
    register_concept_identity(backend)
    register_concept_correspondence(backend)
    register_experience_count(backend)
    register_structure_match_count(backend)
    register_chapter_seq(backend)
    register_selection_pref_count(backend)
    register_sense_candidates(backend)
    register_pronoun_resolution_count(backend)
    register_modification_hist(backend)
    register_abstract_mark(backend)
    register_word_form_index(backend)

    registry = SpaceRegistry(backend)
    core = AbstractSpace.create(registry, "core")
    companion_space = (
        CompanionSpace.create(registry, "companion") if companion else None
    )
    memory_read = MemorySpace.create(registry, "memory_read")
    memory_interact = MemorySpace.create(registry, "memory_interact")
    scoped_identities = ScopedIdentityStore(backend)
    ontology = GraphOntology(
        backend,
        space_id=core.space_id,
        space_identity=SpaceRegistry.identity_for(SPACE_TYPE_CORE, "core"),
        scoped_identities=scoped_identities,
    )
    core_identity_catalog = CoreIdentityCatalog((ontology,))
    memory_read_overlay = MemoryOverlay(
        registry,
        backend,
        memory_read.space_id,
        scoped_identities,
        core_identity_catalog,
    )
    memory_interact_overlay = MemoryOverlay(
        registry,
        backend,
        memory_interact.space_id,
        scoped_identities,
        core_identity_catalog,
    )
    memory_read_events = MemoryEventLog(
        registry,
        backend,
        memory_read.space_id,
        scoped_identities,
        core_identity_catalog,
    )
    memory_interact_events = MemoryEventLog(
        registry,
        backend,
        memory_interact.space_id,
        scoped_identities,
        core_identity_catalog,
    )
    memory_read_aggregates = MemoryHypothesisAggregateIndex(memory_read_events)
    memory_interact_aggregates = MemoryHypothesisAggregateIndex(
        memory_interact_events)
    ctx = TrainContext(
        backend=backend,
        core_space=core,
        edge_store=EdgeStore(backend),
        node_store=NodeStore(backend),
        concept_index=ConceptIndex(backend, companion_space),
        concept_graph=ConceptGraph(backend),
        scoped_identity_store=scoped_identities,
        graph_ontology=ontology,
        training_candidate_history=TrainingCandidateHistoryLog(
            backend, core.space_id),
        core_identity_catalog=core_identity_catalog,
        memory_read_overlay=memory_read_overlay,
        memory_interact_overlay=memory_interact_overlay,
        memory_read_events=memory_read_events,
        memory_interact_events=memory_interact_events,
        memory_read_aggregates=memory_read_aggregates,
        memory_interact_aggregates=memory_interact_aggregates,
        teacher=teacher,
        weights=weights or JudgeWeights(),
        work_memory=WorkMemory(),
        memory_read=memory_read,
        memory_interact=memory_interact,
    )
    if companion_space is not None:
        from pure_integer_ai.cognition.understanding.memory_intake import (
            install_memory_source_intakes,
        )
        install_memory_source_intakes(ctx, companion_space)
    return ctx


def _item_document_identity(
        ctx: TrainContext,
        item: CollectedItem,
        ) -> tuple[int, ScopeIdentity]:
    """返回经 registry 核验的 document scope 索引和完整身份。"""
    scope, scope_hash = ensure_item_scope(item, ctx.scoped_identity_store)
    if ctx.scope_owner is not None:
        # 评测运行 scope 不得复用正式训练的 document owner，也不得把 SourceRef 版本混入
        # 已打开的评测 session；来源版本由独立 occurrence/source scope 保留。
        eval_local_id = Hasher("formal_train.evaluation_document.v1").h63(
            (ctx.scope_owner.stable_key(), scope.stable_key()))
        if eval_local_id == 0:
            eval_local_id = 1
        active_session = ctx.work_memory.active_session_scope
        runtime_versions = (
            scope.versions
            if active_session is None
            else active_session.versions
        )
        scope = make_scope(
            SCOPE_DOCUMENT,
            eval_local_id,
            owner=ctx.scope_owner,
            versions=runtime_versions,
        )
        scope_hash = ctx.scoped_identity_store.register_scope(scope)
    return scope_hash, scope


def _item_observation_identity(
        ctx: TrainContext,
        item: CollectedItem,
        *, stage: int,
        round_id: int,
        ) -> tuple[int, ScopeIdentity]:
    """为一次训练观察构造 document 子 episode scope，并返回 item 桥键。"""
    item_key, document = _item_document_identity(ctx, item)
    local_id = Hasher("formal_train.episode_scope.v1").h63((stage, round_id))
    if local_id == 0:
        local_id = 1
    scope = episode_scope(local_id, parent=document)
    ctx.scoped_identity_store.register_scope(scope)
    return item_key, scope


def _item_occurrence_scope(
        ctx: TrainContext,
        item: CollectedItem,
        ) -> ScopeIdentity:
    """返回不随训练 stage、round 或评测 owner 改变的来源 document scope。"""
    if item.source_ref is None:
        raise ValueError("构造 occurrence scope 前必须先分配 SourceRef")
    scope = source_document_scope(item.source_ref)
    ctx.scoped_identity_store.register_scope(scope)
    return scope


__all__ = [
    "TrainContext",
    "_item_document_identity",
    "_item_observation_identity",
    "_item_occurrence_scope",
    "make_train_context",
]
