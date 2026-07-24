"""评测沙箱和训练状态隔离协议。

评测可以执行观察、建图和局部统计，但这些写入只能进入独立后端。正式训练上下文只
接收明确列入评测契约的结果，例如 H2 权重和校准专属台账，不接收评测图、Memory
或逻辑身份的副作用。
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Iterator

from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.identity import (
    OwnerScope,
    VISIBILITY_SESSION,
)
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.cognition.shared.memory_overlay import (
    CoreIdentityCatalog,
    MemoryOverlay,
)
from pure_integer_ai.cognition.shared.memory_event_log import MemoryEventLog
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MemoryHypothesisAggregateIndex,
)
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingCandidateHistoryLog,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.storage.backend import (
    DictBackend,
    SQLiteBackend,
    StorageBackend,
)
from pure_integer_ai.storage.node_store import NodeStore
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.telemetry import telemetry_scope
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.storage.spaces.memory_space import MemorySpace
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.experiments.train_context import TrainContext


class EvaluationIsolationError(RuntimeError):
    """评测沙箱无法建立或正式状态出现非预期变化。"""


_EVALUATION_OWNER_HASHER = Hasher("pure_integer_ai.evaluation_owner.v1")
_CHECKED_BACKEND_ATTRIBUTES = (
    "_id_pool",
    "_isa_edge_gen",
    "_legacy_observe_timestamp_seq",
)


def _copy_backend_schema(source: StorageBackend, target: StorageBackend) -> None:
    """复制后端表定义，使沙箱不依赖正式后端的可变引用。"""
    tables = getattr(source, "_tables", None)
    if not isinstance(tables, dict):
        raise EvaluationIsolationError(
            "评测沙箱要求后端暴露可复制的表协议；未知后端类型必须显式实现 clone")
    for table, metadata in tables.items():
        columns = [
            (name, metadata["col_types"][name])
            for name in metadata["columns"]
        ]
        target.register_table(
            table,
            columns,
            metadata["discipline"],
            list(metadata.get("indexes", ())),
            core=bool(metadata["core"]),
            recovery_key=tuple(metadata.get("recovery_key", ())),
        )


def clone_backend(source: StorageBackend) -> StorageBackend:
    """复制 Dict/SQLite 后端的 schema、数据和整数水位，返回独立沙箱。"""
    if isinstance(source, DictBackend):
        target: StorageBackend = DictBackend()
    elif isinstance(source, SQLiteBackend):
        target = SQLiteBackend()
    else:
        raise EvaluationIsolationError(
            f"后端类型 {type(source).__name__} 没有评测 clone 协议，拒绝复用正式后端")
    _copy_backend_schema(source, target)
    target.load_snapshot(source.snapshot())
    for attribute in ("_id_pool", "_isa_edge_gen"):
        if hasattr(source, attribute) and hasattr(target, attribute):
            setattr(target, attribute, dict(getattr(source, attribute)))
    if hasattr(source, "_legacy_observe_timestamp_seq"):
        target._legacy_observe_timestamp_seq = source._legacy_observe_timestamp_seq
    return target


def _backend_state(backend: StorageBackend) -> tuple[Any, dict[str, Any], Any]:
    """保存后端数据、整数水位和 schema，供无 TrainContext 的评测入口核验。"""
    attributes = {
        name: copy.deepcopy(getattr(backend, name))
        for name in _CHECKED_BACKEND_ATTRIBUTES
        if hasattr(backend, name)
    }
    return (
        backend.snapshot(),
        attributes,
        copy.deepcopy(getattr(backend, "_tables", None)),
    )


def _assert_backend_state(backend: StorageBackend,
                          baseline: tuple[Any, dict[str, Any], Any]) -> None:
    """核验无上下文评测没有改变宿主后端的数据、水位或 schema。"""
    if _backend_state(backend) != baseline:
        raise EvaluationIsolationError("评测改变了宿主后端状态，拒绝继续")


def _teacher_state(teacher: Any) -> tuple[Any, ...]:
    """提取教师计数、模式和限流器状态，避免复制不可序列化的调用对象。"""
    if teacher is None:
        return ()
    state = tuple(
        (name, copy.deepcopy(getattr(teacher, name)))
        for name in ("call_count", "_mode", "_source_id")
        if hasattr(teacher, name)
    )
    limiter = getattr(teacher, "_limiter", None)
    limiter_state = tuple(
        (name, copy.deepcopy(getattr(limiter, name)))
        for name in ("_window", "_in_flight")
        if limiter is not None and hasattr(limiter, name)
    )
    return state + (("_limiter", limiter_state),)


def _evaluation_owner(label: str, space_id: int,
                      parent_owner: OwnerScope | None = None) -> OwnerScope:
    """为一次评测生成稳定且不同于宿主上下文的 session owner。"""
    parent_key = parent_owner.stable_key() if parent_owner is not None else ()
    session_id = _EVALUATION_OWNER_HASHER.h63((label, space_id, parent_key))
    if session_id == 0:
        session_id = 1
    if parent_owner is not None and session_id == parent_owner.session_id:
        session_id = _EVALUATION_OWNER_HASHER.h63(
            ("nested", label, space_id, parent_key)) or 1
    tenant_id = parent_owner.tenant_id if parent_owner and parent_owner.tenant_id else 1
    user_id = parent_owner.user_id if parent_owner and parent_owner.user_id else 1
    return OwnerScope(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        visibility=VISIBILITY_SESSION,
    )


def _clone_teacher(teacher: Any, backend: StorageBackend) -> Any:
    """复制教师的录放读取状态，禁止评测递增正式教师计数或写正式录放表。"""
    if teacher is None:
        return None
    factory = getattr(teacher, "clone_for_evaluation", None)
    if callable(factory):
        cloned = factory(backend)
    elif hasattr(teacher, "_b"):
        cloned = copy.copy(teacher)
        cloned._b = backend
        if hasattr(cloned, "call_count"):
            cloned.call_count = 0
        if hasattr(cloned, "_limiter"):
            cloned._limiter = copy.copy(cloned._limiter)
    else:
        raise EvaluationIsolationError(
            "评测需要可复制的教师录放对象；未知教师不得共享正式可变状态")
    return cloned


def _generation_factory_state(factory: Any) -> Any:
    """读取可比较的 typed generation factory 状态，缺协议时分型失败。"""
    state_key = getattr(factory, "state_key", None)
    if not callable(state_key):
        raise EvaluationIsolationError(
            "typed generation factory 缺少评测 clone/state 协议")
    return copy.deepcopy(state_key())


def _clone_generation_factory(factory: Any) -> Any:
    """复制 typed generation 装配状态，并核验克隆前后配置完全一致。"""
    clone = getattr(factory, "clone_for_evaluation", None)
    if not callable(clone):
        raise EvaluationIsolationError(
            "typed generation factory 缺少评测 clone/state 协议")
    baseline = _generation_factory_state(factory)
    cloned = clone()
    cloned_state_key = getattr(cloned, "state_key", None)
    if not hasattr(cloned, "build") or not callable(cloned_state_key):
        raise EvaluationIsolationError(
            "typed generation factory 的评测 clone 协议非法")
    if copy.deepcopy(cloned_state_key()) != baseline:
        raise EvaluationIsolationError(
            "typed generation factory 的评测 clone 改变了装配状态")
    return cloned


def clone_train_context(ctx: Any, backend: StorageBackend, *, label: str) -> Any:
    """在独立后端上重建训练上下文及其图、身份和 Memory 读写入口。"""
    registry = SpaceRegistry(backend)
    core = AbstractSpace(registry, backend, ctx.core_space.space_id)
    original_companion = getattr(ctx.concept_index, "_companion", None)
    companion = None
    if original_companion is not None:
        companion = CompanionSpace(
            registry, backend, original_companion.space_id)
    memory_read = None
    if ctx.memory_read is not None:
        memory_read = MemorySpace(registry, backend, ctx.memory_read.space_id)
    memory_interact = None
    if ctx.memory_interact is not None:
        memory_interact = MemorySpace(
            registry, backend, ctx.memory_interact.space_id)
    active_session = ctx.work_memory.active_session_scope
    parent_owner = ctx.scope_owner
    if parent_owner is None and active_session is not None:
        parent_owner = active_session.owner
    eval_owner = _evaluation_owner(label, ctx.space_id, parent_owner)
    scoped_identities = ScopedIdentityStore(backend)
    graph_ontology = GraphOntology(
        backend,
        space_id=core.space_id,
        space_identity=ctx.graph_ontology.space_identity,
        scoped_identities=scoped_identities,
    )
    core_identity_catalog = CoreIdentityCatalog((graph_ontology,))
    memory_read_overlay = (
        None if memory_read is None else MemoryOverlay(
            registry,
            backend,
            memory_read.space_id,
            scoped_identities,
            core_identity_catalog,
        )
    )
    memory_interact_overlay = (
        None if memory_interact is None else MemoryOverlay(
            registry,
            backend,
            memory_interact.space_id,
            scoped_identities,
            core_identity_catalog,
        )
    )
    memory_read_events = (
        None if memory_read is None else MemoryEventLog(
            registry,
            backend,
            memory_read.space_id,
            scoped_identities,
            core_identity_catalog,
        )
    )
    memory_interact_events = (
        None if memory_interact is None else MemoryEventLog(
            registry,
            backend,
            memory_interact.space_id,
            scoped_identities,
            core_identity_catalog,
        )
    )
    memory_read_aggregates = (
        None if memory_read_events is None
        else MemoryHypothesisAggregateIndex(memory_read_events)
    )
    memory_interact_aggregates = (
        None if memory_interact_events is None
        else MemoryHypothesisAggregateIndex(memory_interact_events)
    )
    cloned = TrainContext(
        backend=backend,
        core_space=core,
        edge_store=EdgeStore(backend),
        node_store=NodeStore(backend),
        concept_index=ConceptIndex(backend, companion),
        concept_graph=ConceptGraph(backend),
        scoped_identity_store=scoped_identities,
        graph_ontology=graph_ontology,
        training_candidate_history=TrainingCandidateHistoryLog(
            backend,
            core.space_id,
        ),
        core_identity_catalog=core_identity_catalog,
        memory_read_overlay=memory_read_overlay,
        memory_interact_overlay=memory_interact_overlay,
        memory_read_events=memory_read_events,
        memory_interact_events=memory_interact_events,
        memory_read_aggregates=memory_read_aggregates,
        memory_interact_aggregates=memory_interact_aggregates,
        word_form_course_report=ctx.word_form_course_report,
        alias_relation_course_report=ctx.alias_relation_course_report,
        language_generation_course_report=(
            ctx.language_generation_course_report),
        language_generation_postcheck_course_report=(
            ctx.language_generation_postcheck_course_report),
        teacher=_clone_teacher(ctx.teacher, backend),
        weights=copy.deepcopy(ctx.weights),
        work_memory=WorkMemory(),
        weaning_phase=ctx.weaning_phase,
        judge_source_id=ctx.judge_source_id,
        judge_source_independent=ctx.judge_source_independent,
        evaluation_plan=ctx.evaluation_plan,
        evaluation_corpora={
            key: list(items)
            for key, items in ctx.evaluation_corpora.items()
        },
        probe_set=ctx.probe_set,
        probe_corpus=list(ctx.probe_corpus),
        probe_content_disjoint=ctx.probe_content_disjoint,
        evaluation_strictly_isolated=ctx.evaluation_strictly_isolated,
        probe_set_disjoint=ctx.probe_set_disjoint,
        verification_reports=[],
        holdout_retention=ctx.holdout_retention,
        e2_eval_passed=ctx.e2_eval_passed,
        memory_read=memory_read,
        memory_interact=memory_interact,
        scope_owner=eval_owner,
    )
    source_intake_enabled = (
        ctx.memory_read_intake is not None,
        ctx.memory_interact_intake is not None,
    )
    if source_intake_enabled[0] != source_intake_enabled[1]:
        raise EvaluationIsolationError(
            "M-05 阅读/交互摄入协议必须成对装配")
    if source_intake_enabled[0]:
        if companion is None:
            raise EvaluationIsolationError(
                "M-05 摄入协议缺少可克隆 Companion")
        from pure_integer_ai.cognition.understanding.memory_intake import (
            install_memory_source_intakes,
        )
        install_memory_source_intakes(cloned, companion)
    if ctx.memory_batch_config is not None:
        from pure_integer_ai.cognition.shared.memory_batch import (
            install_memory_batch_runtimes,
        )
        install_memory_batch_runtimes(
            cloned,
            ctx.memory_batch_config,
        )
    if ctx.memory_isolation_runtime is not None:
        if cloned.memory_batch_config is None:
            raise EvaluationIsolationError(
                "M-11 isolation clone 缺少 M-10 batch runtime")
        cloned.memory_isolation_runtime = (
            ctx.memory_isolation_runtime.clone_for_context(cloned))
    if ctx.memory_query_runtime is not None:
        cloned.memory_query_runtime = (
            ctx.memory_query_runtime.clone_for_context(cloned))
    if ctx.memory_resolver_runtime is not None:
        if cloned.memory_query_runtime is None:
            raise EvaluationIsolationError(
                "M-07 resolver clone 缺少 M-06 query runtime")
        cloned.memory_resolver_runtime = (
            ctx.memory_resolver_runtime.clone_for_context(cloned))
    if ctx.memory_hot_set_runtime is not None:
        if (cloned.memory_resolver_runtime is None
                or cloned.tiered_segment_store is None):
            raise EvaluationIsolationError(
                "K-04 hot-set clone 缺少 M-07 resolver 或 K-02 store")
        cloned.memory_hot_set_runtime = (
            ctx.memory_hot_set_runtime.clone_for_context(cloned))
    if ctx.attractor_runtime is not None:
        if cloned.memory_resolver_runtime is None:
            raise EvaluationIsolationError(
                "A-10 runtime clone 缺少 M-07 resolver runtime")
        cloned.attractor_runtime = ctx.attractor_runtime.clone_for_context(
            cloned)
    if ctx.memory_use_runtime is not None:
        if cloned.attractor_runtime is None:
            raise EvaluationIsolationError(
                "M-08 runtime clone 缺少 A-10 attractor runtime")
        cloned.memory_use_runtime = (
            ctx.memory_use_runtime.clone_for_context(cloned))
    if ctx.memory_maintenance_runtime is not None:
        if cloned.memory_use_runtime is None:
            raise EvaluationIsolationError(
                "M-09 runtime clone 缺少 M-08 runtime")
        cloned.memory_maintenance_runtime = (
            ctx.memory_maintenance_runtime.clone_for_context(cloned))
    if ctx.unicode_intake is not None:
        cloned.unicode_intake = ctx.unicode_intake.clone_for_ontology(
            cloned.graph_ontology)
    if ctx.word_form_providers is not None:
        cloned.word_form_providers = ctx.word_form_providers.clone_for_context(
            backend=backend,
            concept_index=cloned.concept_index,
            ontology=cloned.graph_ontology,
        )
    if ctx.occurrence_index is not None:
        cloned.occurrence_index = ctx.occurrence_index.clone_for_context(
            cloned.graph_ontology,
            cloned.scoped_identity_store,
        )
    if ctx.occurrence_order_writer is not None:
        from pure_integer_ai.cognition.shared.order_facts import OrderFactIndex
        cloned_order_facts = OrderFactIndex(
            cloned.graph_ontology,
            cloned.scoped_identity_store,
        )
        cloned.occurrence_order_writer = (
            ctx.occurrence_order_writer.clone_for_context(cloned_order_facts))
        if ctx.occurrence_order_reader is not None:
            cloned.occurrence_order_reader = (
                ctx.occurrence_order_reader.clone_for_context(
                    cloned_order_facts,
                    cloned.occurrence_index,
                ))
    if ctx.precedence_relation_runtime is not None:
        if cloned.occurrence_order_reader is None:
            raise EvaluationIsolationError(
                "R-06 runtime 缺少可克隆的 occurrence order reader")
        cloned.precedence_relation_runtime = (
            ctx.precedence_relation_runtime.clone_for_context(cloned))
    if ctx.causal_relation_runtime is not None:
        cloned.causal_relation_runtime = (
            ctx.causal_relation_runtime.clone_for_context(cloned))
    if ctx.set_relation_runtime is not None:
        cloned.set_relation_runtime = (
            ctx.set_relation_runtime.clone_for_context(cloned))
        cloned.set_relation_reports = copy.deepcopy(
            ctx.set_relation_reports)
    if ctx.property_relation_runtime is not None:
        cloned.property_relation_runtime = (
            ctx.property_relation_runtime.clone_for_context(cloned))
        cloned.property_relation_reports = copy.deepcopy(
            ctx.property_relation_reports)
    if ctx.span_index is not None:
        cloned.span_index = ctx.span_index.clone_for_context(
            cloned.graph_ontology,
            cloned.scoped_identity_store,
            cloned.occurrence_index,
        )
    if ctx.segmentation_span_materializer is not None:
        cloned.segmentation_span_materializer = (
            ctx.segmentation_span_materializer.clone_for_context(
                cloned.span_index))
        if cloned.word_form_providers is not None:
            cloned.word_form_providers.install_segmentation_span_materializer(
                cloned.segmentation_span_materializer)
    if ctx.boundary_hypothesis_engine is not None:
        cloned.boundary_hypothesis_engine = (
            ctx.boundary_hypothesis_engine.clone())
    if ctx.boundary_span_materializer is not None:
        cloned.boundary_span_materializer = (
            ctx.boundary_span_materializer.clone_for_context(
                cloned.span_index))
    if ctx.language_prediction_runtime is not None:
        if cloned.occurrence_index is None:
            raise EvaluationIsolationError(
                "H-01 预测 runtime 缺少可克隆的 occurrence 索引")
        cloned.language_prediction_runtime = (
            ctx.language_prediction_runtime.clone_for_context(
                cloned.graph_ontology,
                cloned.occurrence_index,
            ))
    if ctx.candidate_projection_graph is not None:
        from pure_integer_ai.cognition.shared.candidate_projection import (
            CandidateProjectionGraph,
        )
        from pure_integer_ai.cognition.understanding.language_candidate import (
            ActiveCueStructureConsumer,
            ActiveSenseConsumer,
        )
        from pure_integer_ai.experiments.language_structure_candidate_runtime import (
            StructureCandidateCourseMapper,
        )
        cloned.candidate_projection_graph = CandidateProjectionGraph(
            cloned.graph_ontology,
            ctx.candidate_projection_graph.protocol,
        )
        if ctx.structure_candidate_runtime is None:
            raise EvaluationIsolationError(
                "H-05 投影图缺少 structure candidate owner")
        if ctx.sense_candidate_runtime is None:
            raise EvaluationIsolationError(
                "H-05 投影图缺少 Sense candidate owner")
        cloned.structure_candidate_runtime = (
            ctx.structure_candidate_runtime.clone_for_graph(
                cloned.candidate_projection_graph))
        cloned.sense_candidate_runtime = (
            ctx.sense_candidate_runtime.clone_for_graph(
                cloned.candidate_projection_graph))
        cloned.structure_candidate_consumer = ActiveCueStructureConsumer(
            cloned.candidate_projection_graph,
            ctx.structure_candidate_consumer.protocol,
        )
        mapper_factory = getattr(
            ctx.structure_candidate_mapper,
            "clone_for_evaluation",
            None,
        )
        if not callable(mapper_factory):
            raise EvaluationIsolationError(
                "H-05 structure mapper 缺少评测 clone 协议")
        cloned.structure_candidate_mapper = mapper_factory()
        if not isinstance(
                cloned.structure_candidate_mapper,
                StructureCandidateCourseMapper):
            raise EvaluationIsolationError(
                "H-05 structure mapper 的评测 clone 类型非法")
        if ctx.structure_boundary_evidence_mapper is not None:
            from pure_integer_ai.experiments.language_structure_boundary_runtime import (
                StructureBoundaryEvidenceMapper,
            )
            boundary_mapper_factory = getattr(
                ctx.structure_boundary_evidence_mapper,
                "clone_for_evaluation",
                None,
            )
            if not callable(boundary_mapper_factory):
                raise EvaluationIsolationError(
                    "H-05 structure-boundary mapper 缺少评测 clone 协议")
            cloned.structure_boundary_evidence_mapper = (
                boundary_mapper_factory())
            if not isinstance(
                    cloned.structure_boundary_evidence_mapper,
                    StructureBoundaryEvidenceMapper):
                raise EvaluationIsolationError(
                    "H-05 structure-boundary mapper 的评测 clone 类型非法")
        cloned.structure_boundary_report = copy.deepcopy(
            ctx.structure_boundary_report)
        cloned.structure_candidate_reports = copy.deepcopy(
            ctx.structure_candidate_reports)
        cloned.sense_candidate_consumer = ActiveSenseConsumer(
            cloned.candidate_projection_graph,
            ctx.sense_candidate_consumer.protocol,
        )
        if ctx.sense_candidate_course_runtime is None:
            raise EvaluationIsolationError(
                "H-05 投影图缺少 Sense 课程 runtime")
        cloned.sense_candidate_course_runtime = (
            ctx.sense_candidate_course_runtime.clone_for_evaluation())
        cloned.sense_candidate_reports = copy.deepcopy(
            ctx.sense_candidate_reports)
    if ctx.language_semantic_course_runtime is not None:
        if cloned.span_index is None or cloned.occurrence_index is None:
            raise EvaluationIsolationError(
                "语义课程 runtime 缺少可克隆的 occurrence/span 地基")
        cloned.language_semantic_course_runtime = (
            ctx.language_semantic_course_runtime.clone_for_context(cloned))
        cloned.language_semantic_course_reports = copy.deepcopy(
            ctx.language_semantic_course_reports)
    if ctx.language_generation_runtime is not None:
        if ctx.language_generation_runtime_factory is None:
            raise EvaluationIsolationError(
                "typed generation runtime 缺少可重建 factory")
        from pure_integer_ai.experiments.generation_production_runtime import (
            install_production_generation_runtime,
        )
        install_production_generation_runtime(
            cloned,
            _clone_generation_factory(
                ctx.language_generation_runtime_factory),
        )
        if ((ctx.language_generation_stage4_runtime is None)
                != (cloned.language_generation_stage4_runtime is None)):
            raise EvaluationIsolationError(
                "typed generation 评测 clone 改变了 stage4 owner 协议")
    return cloned


def _host_state(ctx: Any) -> tuple[Any, ...]:
    """保存正式上下文的可观察状态，防止评测偷偷回写正式对象。"""
    backend = ctx.backend
    attributes = {
        name: copy.deepcopy(getattr(backend, name))
        for name in _CHECKED_BACKEND_ATTRIBUTES
        if hasattr(backend, name)
    }
    concept_index = (
        copy.deepcopy(ctx.concept_index._index),
        copy.deepcopy(ctx.concept_index._loaded_spaces),
    )
    scoped_store = (
        copy.deepcopy(ctx.scoped_identity_store._scope_hashes),
        copy.deepcopy(ctx.scoped_identity_store._clock_hashes),
        copy.deepcopy(ctx.scoped_identity_store._timestamp_hashes),
        copy.deepcopy(ctx.scoped_identity_store._assertion_hashes),
    )
    word_form_state = (
        () if ctx.word_form_providers is None
        else ctx.word_form_providers.segmentation_state()
    )
    boundary_state = (
        () if ctx.boundary_hypothesis_engine is None
        else ctx.boundary_hypothesis_engine.ledger.state_key()
    )
    prediction_state = (
        () if ctx.language_prediction_runtime is None
        else ctx.language_prediction_runtime.state_key()
    )
    memory_query_state = (
        () if ctx.memory_query_runtime is None
        else ctx.memory_query_runtime.state_key()
    )
    memory_resolver_state = (
        () if ctx.memory_resolver_runtime is None
        else ctx.memory_resolver_runtime.state_key()
    )
    memory_hot_set_state = (
        () if ctx.memory_hot_set_runtime is None
        else ctx.memory_hot_set_runtime.state_key()
    )
    attractor_state = (
        () if ctx.attractor_runtime is None
        else ctx.attractor_runtime.state_key()
    )
    memory_use_state = (
        () if ctx.memory_use_runtime is None
        else ctx.memory_use_runtime.state_key()
    )
    memory_maintenance_state = (
        () if ctx.memory_maintenance_runtime is None
        else ctx.memory_maintenance_runtime.state_key()
    )
    memory_isolation_state = (
        () if ctx.memory_isolation_runtime is None
        else ctx.memory_isolation_runtime.state_key()
    )
    prediction_reports = copy.deepcopy(ctx.language_prediction_reports)
    structure_candidate_state = (
        () if ctx.structure_candidate_runtime is None
        else ctx.structure_candidate_runtime.state_key()
    )
    sense_candidate_state = (
        () if ctx.sense_candidate_runtime is None
        else ctx.sense_candidate_runtime.state_key()
    )
    structure_mapper_state = (
        () if ctx.structure_candidate_mapper is None
        else ctx.structure_candidate_mapper.state_key()
    )
    sense_course_state = (
        () if ctx.sense_candidate_course_runtime is None
        else ctx.sense_candidate_course_runtime.state_key()
    )
    structure_boundary_mapper_state = (
        () if ctx.structure_boundary_evidence_mapper is None
        else ctx.structure_boundary_evidence_mapper.state_key()
    )
    precedence_state = (
        () if ctx.precedence_relation_runtime is None
        else ctx.precedence_relation_runtime.state_key()
    )
    causal_state = (
        () if ctx.causal_relation_runtime is None
        else ctx.causal_relation_runtime.state_key()
    )
    set_relation_state = (
        () if ctx.set_relation_runtime is None
        else ctx.set_relation_runtime.state_key()
    )
    property_relation_state = (
        () if ctx.property_relation_runtime is None
        else ctx.property_relation_runtime.state_key()
    )
    semantic_course_state = (
        () if ctx.language_semantic_course_runtime is None
        else ctx.language_semantic_course_runtime.state_key()
    )
    generation_factory_state = (
        () if ctx.language_generation_runtime_factory is None
        else _generation_factory_state(
            ctx.language_generation_runtime_factory)
    )
    generation_stage4_state = ()
    if ctx.language_generation_stage4_runtime is not None:
        stage4_state_key = getattr(
            ctx.language_generation_stage4_runtime,
            "state_key",
            None,
        )
        if not callable(stage4_state_key):
            raise EvaluationIsolationError(
                "typed generation stage4 owner 缺少状态协议")
        generation_stage4_state = copy.deepcopy(stage4_state_key())
    return (
        backend.snapshot(),
        attributes,
        concept_index,
        scoped_store,
        copy.deepcopy(ctx.work_memory),
        _teacher_state(ctx.teacher),
        word_form_state,
        boundary_state,
        prediction_state,
        memory_query_state,
        memory_resolver_state,
        memory_hot_set_state,
        attractor_state,
        memory_use_state,
        memory_maintenance_state,
        memory_isolation_state,
        prediction_reports,
        structure_candidate_state,
        sense_candidate_state,
        structure_mapper_state,
        structure_boundary_mapper_state,
        sense_course_state,
        precedence_state,
        causal_state,
        set_relation_state,
        property_relation_state,
        semantic_course_state,
        generation_factory_state,
        generation_stage4_state,
        copy.deepcopy(ctx.structure_candidate_reports),
        copy.deepcopy(ctx.structure_boundary_report),
        copy.deepcopy(ctx.sense_candidate_reports),
        copy.deepcopy(ctx.precedence_relation_reports),
        copy.deepcopy(ctx.causal_relation_reports),
        copy.deepcopy(ctx.set_relation_reports),
        copy.deepcopy(ctx.property_relation_reports),
        copy.deepcopy(ctx.language_semantic_course_reports),
        copy.deepcopy(ctx.verification_reports),
    )


def _assert_host_state(ctx: Any, baseline: tuple[Any, ...]) -> None:
    """核验正式后端、索引、身份缓存和 WorkMemory 均未被评测修改。"""
    current = _host_state(ctx)
    if current != baseline:
        raise EvaluationIsolationError("评测改变了正式训练状态，拒绝继续")


def _close_evaluation_session(eval_ctx: Any) -> None:
    """关闭沙箱生命周期；异常中的未闭合子 scope 统一按 episode 中止清理。"""
    work_memory = eval_ctx.work_memory
    has_episode_residue = any((
        work_memory.active_episode_scope,
        work_memory.active_query_scope,
        work_memory.active_generation_scope,
        work_memory.active_segment_index is not None,
    ))
    if has_episode_residue:
        work_memory.abort_episode()
    if work_memory.active_document_scope is not None:
        work_memory.end_document()
    if work_memory.active_session_scope is not None:
        work_memory.end_session()


@contextmanager
def isolated_evaluation(ctx: Any, *, label: str) -> Iterator[Any]:
    """给独立评测附加 caller 标记，并复用统一沙箱实现。"""
    with telemetry_scope(
            caller=f"evaluation:{label}",
            query=label,
            evaluation=True):
        with _isolated_evaluation_impl(ctx, label=label) as eval_ctx:
            yield eval_ctx


@contextmanager
def _isolated_evaluation_impl(ctx: Any, *, label: str) -> Iterator[Any]:
    """建立独立评测上下文，并在退出时核验正式状态仍 bit-identical。"""
    baseline = _host_state(ctx)
    backend = clone_backend(ctx.backend)
    try:
        eval_ctx = clone_train_context(ctx, backend, label=label)
        eval_ctx.work_memory.begin_session(session_scope(
            eval_ctx.space_id,
            owner=eval_ctx.scope_owner,
        ))
    except BaseException:
        try:
            _assert_host_state(ctx, baseline)
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
        raise
    try:
        yield eval_ctx
    finally:
        try:
            try:
                _close_evaluation_session(eval_ctx)
            finally:
                _assert_host_state(ctx, baseline)
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                close()


@contextmanager
def isolated_backend_evaluation(
        backend: StorageBackend,
        *,
        teacher: Any = None,
        label: str = "backend",
        ) -> Iterator[tuple[StorageBackend, Any]]:
    """给无上下文评测附加 caller 标记，并复用统一后端沙箱实现。"""
    with telemetry_scope(
            caller=f"evaluation:{label}",
            query=label,
            evaluation=True):
        with _isolated_backend_evaluation_impl(
                backend, teacher=teacher) as isolated:
            yield isolated


@contextmanager
def _isolated_backend_evaluation_impl(
        backend: StorageBackend,
        *,
        teacher: Any = None,
        ) -> Iterator[tuple[StorageBackend, Any]]:
    """为尚未构造 TrainContext 的统一考核入口提供独立 backend 和教师。"""
    baseline = (_backend_state(backend), _teacher_state(teacher))
    eval_backend = clone_backend(backend)
    try:
        eval_teacher = _clone_teacher(teacher, eval_backend)
    except BaseException:
        try:
            if (_backend_state(backend), _teacher_state(teacher)) != baseline:
                raise EvaluationIsolationError("评测初始化改变了宿主状态，拒绝继续")
        finally:
            close = getattr(eval_backend, "close", None)
            if callable(close):
                close()
        raise
    try:
        yield eval_backend, eval_teacher
    finally:
        try:
            _assert_backend_state(backend, baseline[0])
            if _teacher_state(teacher) != baseline[1]:
                raise EvaluationIsolationError("评测改变了宿主教师状态，拒绝继续")
        finally:
            close = getattr(eval_backend, "close", None)
            if callable(close):
                close()


__all__ = [
    "EvaluationIsolationError",
    "clone_backend",
    "clone_train_context",
    "isolated_backend_evaluation",
    "isolated_evaluation",
]
