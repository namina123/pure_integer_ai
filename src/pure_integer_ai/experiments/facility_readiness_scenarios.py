"""F-01 生产 adapter 使用的真实 Memory、恢复和隔离场景。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorActivationProposal,
    AttractorBudget,
    AttractorDependency,
    AttractorProtocol,
    AttractorRecomputeDecision,
    AttractorScoreReason,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    VISIBILITY_SESSION,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    span_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.memory_batch import (
    MemoryBatchRuntimeConfig,
    install_memory_batch_runtimes,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_OBSERVATION,
    MEMORY_EVENT_USE,
    MEMORY_EVENT_USE_OUTCOME,
    MEMORY_OBJECT_OBSERVATION,
    MemoryEvent,
    MemoryLinkedRef,
    ObservationPayload,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_hot_set import (
    StableTopKSourcePolicy,
)
from pure_integer_ai.cognition.shared.memory_hypothesis import (
    MemoryHypothesisEventSink,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_query import (
    MemoryCurrentQuery,
    MemoryQueryDefinition,
    MemoryQueryProtocol,
    MemoryQueryRoles,
)
from pure_integer_ai.cognition.shared.memory_resolver import (
    ActivationScore,
    ActivationScoreReason,
    CoreBaselineCandidate,
    MemoryAggregateFilter,
    MemoryResolution,
    SourceDiversityAssessment,
)
from pure_integer_ai.cognition.shared.memory_resolver_engine import (
    MemoryOverlayResolver,
)
from pure_integer_ai.cognition.shared.memory_generation import (
    MemoryGenerationSource,
)
from pure_integer_ai.cognition.shared.post_weaning import (
    PostWeaningFacilityCheck,
    PostWeaningFacilityProbe,
    PostWeaningIntakeRequest,
    PostWeaningResourceBudget,
    PostWeaningRouteProtocol,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
)
from pure_integer_ai.cognition.shared.reasoning_planner import (
    ReasoningObligation,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_OBSERVED,
    CLOCK_QUERY,
    LogicalClock,
    LogicalClockIdentity,
    document_scope,
    episode_scope,
    query_scope,
    session_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.cognition.shared.source_trust import (
    SOURCE_ADMISSION_ACCEPTED,
    SourceTrustAssessment,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft,
    ObservationIntakeDraft,
)
from pure_integer_ai.experiments.attractor_runtime import install_attractor_runtime
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.evaluation_protocol import CanonicalIdentity
from pure_integer_ai.experiments.facility_generation_scenario import (
    build_postcheck_owners,
    build_question_fixture,
)
from pure_integer_ai.experiments.memory_generation_outcome_runtime import (
    MemoryGenerationOutcomeProtocol,
    MemoryGenerationOutcomeRoute,
    MemoryGenerationOutcomeValueRoute,
    MemoryQuestionOutcomeCommitter,
)
from pure_integer_ai.experiments.memory_generation_runtime import (
    MemoryAwareQuestionDialogueRuntime,
    MemoryQuestionSelectionCommitter,
    ResolvedMemoryQuestionExecutor,
)
from pure_integer_ai.experiments.memory_hot_set_runtime import (
    MemoryCandidateProjectionManifest,
    MemoryCandidateProjectionPublisher,
    MemoryProjectionPublication,
    install_memory_hot_set_runtime,
)
from pure_integer_ai.experiments.memory_query_runtime import (
    install_memory_query_runtime,
)
from pure_integer_ai.experiments.memory_resolver_runtime import (
    install_memory_resolver_runtime,
)
from pure_integer_ai.experiments.memory_use_runtime import (
    install_memory_use_runtime,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    PostWeaningDryRunRuntime,
    build_post_weaning_dry_run_manifest,
)
from pure_integer_ai.experiments.source_trust_runtime import (
    install_source_admission_runtime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_NOT_APPLICABLE,
    APPLICABILITY_UNKNOWN,
    VERDICT_CONFLICTED,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.memory_aggregate import (
    MEMORY_AGGREGATE_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.memory_batch import (
    MEMORY_BATCH_CORE_DEPENDENCY_KEY,
)
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.placement import TemperatureProfile, TemperatureTier
from pure_integer_ai.storage.query_hot_set import (
    QueryHotSetPolicy,
    QueryPrefetchContext,
)
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from pure_integer_ai.storage.segment_dependency import SegmentDependency
from pure_integer_ai.storage.source_record import (
    SourceRecordMetadata,
    SourceRecordRepository,
)
from pure_integer_ai.training.cursor import (
    CursorState,
    cursor_state_from_payload,
    dump_run,
    load_run_package,
)


_ACCESS = MemoryAccessContext(1, 2, 3)
_PROFILE = TemperatureProfile(
    (920, 1),
    (TemperatureTier((920, 1), 0), TemperatureTier((920, 2), 1)),
)
_HOT = (920, 1)
_COLD = (920, 2)
_PROJECTION_KEY = (921, 1)
_KINDS = ((7201,), (7202,))
_DEPENDENCIES = (
    SegmentDependency(MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY, (1, 1), (2, 1)),
    SegmentDependency(
        MEMORY_AGGREGATE_STORAGE_DESCRIPTOR_KEY,
        (1, 2),
        (2, 2),
    ),
)


@dataclass(frozen=True)
class MemoryPathEvidence:
    """保存同一 clone 内来源、问答、生成和 Memory 闭环证据。"""

    positive_behavior: int
    negative_behavior: int
    admissions: int
    source_clusters: int
    candidates: int
    conflicts: int
    uses: int
    outcomes: int
    query_key: tuple[int, ...]
    query_before: CanonicalIdentity
    query_after: CanonicalIdentity
    resources_closed: bool
    result_identity: CanonicalIdentity
    observation_ref: Any
    source: SourceRef


def _versions() -> VersionBundle:
    """构造非零版本，避免设施只覆盖 legacy 默认版本。"""
    return VersionBundle(
        CorpusVersion(1),
        ParserVersion(2),
        PrimitiveVersion(3),
        CurriculumVersion(4),
    )


def _owner() -> OwnerScope:
    """构造设施场景独占的 session owner。"""
    return OwnerScope(1, 2, 3, VISIBILITY_SESSION)


def _memory_source(*, source_id: int, document_id: int) -> SourceRef:
    """构造与 Memory owner/version 对齐的来源。"""
    return SourceRef(1, source_id, document_id, _owner(), _versions())


def _query_source(document_id: int = 1) -> SourceRef:
    """构造 query 编译和 ACL 共用的 session 来源。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        6100,
        document_id,
        _owner(),
        VersionBundle(),
    )


def _instruction(source: SourceRef, value: int) -> ObjectIdentity:
    """构造 owner/version 对齐的一等最小指令。"""
    return minimal_instruction_identity(
        (value,), owner=source.owner, versions=source.versions)


def _core_refs(ctx: Any) -> tuple[Any, ...]:
    """在 Core 冻结前物化上下文、概念、结构、命题和 relation 端点。"""
    from pure_integer_ai.cognition.shared.graph_ontology import (
        relation_concept_identity,
    )

    ontology = ctx.graph_ontology
    return (
        ontology.materialize(concept_identity((301,))),
        ontology.materialize(concept_identity((302,))),
        ontology.materialize(concept_identity((303,))),
        ontology.materialize(concept_identity((304,))),
        ontology.materialize(relation_concept_identity((305,))),
    )


def _append_observation(
        ctx: Any,
        source: SourceRef,
        refs: tuple[Any, ...],
        ) -> Any:
    """追加字段齐全的 Observation 并返回真实物化事件。"""
    document = document_scope(source)
    episode = episode_scope(1, parent=document)
    context, concept, structure, proposition, relation = refs
    timestamp = LogicalClock(
        LogicalClockIdentity(episode, CLOCK_MEMORY_OBSERVED)).advance()
    payload = ObservationPayload(
        source,
        MemoryLinkedRef.core(context),
        (concept,),
        (concept, structure),
        structure,
        (proposition,),
        (MemoryLinkedRef.core(relation),),
        timestamp,
    )
    ref = memory_object_ref(
        ctx.memory_interact_events.memory_space_identity,
        MEMORY_OBJECT_OBSERVATION,
        payload.stable_key(),
        owner=source.owner,
        versions=source.versions,
    )
    return ctx.memory_interact_events.append(MemoryEvent(
        MEMORY_EVENT_OBSERVATION, ref, episode, payload))


def _query_protocol(source: SourceRef) -> MemoryQueryProtocol:
    """构造两个独立 Hypothesis kind 和分型预算。"""
    roles = MemoryQueryRoles(*tuple(
        _instruction(source, value) for value in range(8101, 8109)))
    return MemoryQueryProtocol(
        roles,
        (
            MemoryQueryDefinition(
                _instruction(source, 8201),
                (7201,),
                (roles.occurrence, roles.domain),
                3,
            ),
            MemoryQueryDefinition(
                _instruction(source, 8202),
                (7202,),
                (roles.span, roles.intent),
                1,
            ),
        ),
    )


def _hypothesis(
        source: SourceRef,
        *,
        kind: tuple[int, ...],
        candidate: int,
        competition: int,
        ) -> HypothesisKey:
    """构造来源化 Memory 候选。"""
    return HypothesisKey(
        kind,
        (candidate,),
        (competition,),
        document_scope(source),
        source,
    )


class _BaselineProvider:
    """为 overlay 提供只读 Core 基线候选。"""

    def __init__(self, core_ref: Any) -> None:
        self.core_ref = core_ref

    def candidates(self, request: Any) -> tuple[CoreBaselineCandidate, ...]:
        """为每个 request 返回同一 Core 身份和独立竞争组。"""
        return (CoreBaselineCandidate(
            self.core_ref,
            (8300, *request.hypothesis_kind),
            ActivationScore(50, (ActivationScoreReason(
                (8301, *request.hypothesis_kind), 50),)),
        ),)

    def clone_for_context(self, ctx: Any) -> "_BaselineProvider":
        """为 clone 建立无共享可变状态的 provider。"""
        del ctx
        return _BaselineProvider(self.core_ref)

    def state_key(self) -> tuple[int, ...]:
        """返回完整 Core 引用状态键。"""
        return 1, *self.core_ref.stable_key()


class _CurrentContextScorer:
    """只按当前 document 和支持事件数形成确定性整数分。"""

    def score(
            self,
            request: Any,
            hypothesis: HypothesisKey,
            aggregate: Any,
            sources: Any,
            ) -> ActivationScore:
        """当前 document 命中 candidate 时提高方向分。"""
        del sources
        candidate = hypothesis.candidate_key[0]
        value = (
            1000
            if candidate == request.source.document_id
            else aggregate.support_count * 100
        )
        return ActivationScore(
            value,
            (ActivationScoreReason(
                (8401, request.source.document_id, candidate), value),),
        )

    def clone_for_context(self, ctx: Any) -> "_CurrentContextScorer":
        """返回无共享状态的新评分器。"""
        del ctx
        return _CurrentContextScorer()

    def state_key(self) -> tuple[int, ...]:
        """返回评分协议版本。"""
        return (1,)


class _IndexFilterProvider:
    """按 request kind 提供无额外词面条件的索引分支。"""

    def filters(self, request: Any) -> tuple[MemoryAggregateFilter, ...]:
        """返回唯一空过滤分支，kind 由 resolver 自行注入。"""
        del request
        return (MemoryAggregateFilter(),)

    def clone_for_context(self, ctx: Any) -> "_IndexFilterProvider":
        """返回无状态 provider。"""
        del ctx
        return _IndexFilterProvider()

    def state_key(self) -> tuple[int, ...]:
        """返回索引协议版本。"""
        return (1, 0)


class _DistinctSourcePolicy:
    """按完整来源簇形成多样性分，并优先引入新来源。"""

    def assess(
            self,
            request: Any,
            hypothesis: Any,
            aggregate: Any,
            sources: Any,
            source_traces: Any,
            ) -> SourceDiversityAssessment:
        """每个独立来源簇增加十分并保留整数理由。"""
        del request, hypothesis, aggregate, sources
        count = len({item.source_cluster_key for item in source_traces})
        value = count * 10
        return SourceDiversityAssessment(
            count,
            value,
            (ActivationScoreReason((8501, count), value),),
        )

    def select(self, request: Any, candidates: Any, budget: int) -> tuple[Any, ...]:
        """优先选择能引入新来源簇的候选。"""
        del request
        selected = []
        pending = []
        seen = set()
        for candidate in candidates:
            keys = {
                trace.source_cluster_key
                for trace in candidate.memory_source_traces
            }
            if keys - seen:
                selected.append(candidate)
                seen.update(keys)
            else:
                pending.append(candidate)
            if len(selected) == budget:
                return tuple(selected)
        selected.extend(pending[:budget - len(selected)])
        return tuple(selected)

    def clone_for_context(self, ctx: Any) -> "_DistinctSourcePolicy":
        """返回无共享状态的新 policy。"""
        del ctx
        return _DistinctSourcePolicy()

    def state_key(self) -> tuple[int, ...]:
        """返回多样性策略版本。"""
        return (1,)


def _seed_memory(ctx: Any) -> None:
    """写入两个同 kind 候选和一个异 kind 候选，并形成真实冲突。"""
    source_a = _memory_source(source_id=10, document_id=11)
    source_b = _memory_source(source_id=20, document_id=12)
    first = _hypothesis(
        source_a, kind=(7201,), candidate=1, competition=8601)
    second = _hypothesis(
        source_b, kind=(7201,), candidate=2, competition=8601)
    other = _hypothesis(
        source_a, kind=(7202,), candidate=3, competition=8602)
    ledger = HypothesisLedger(
        MemoryHypothesisEventSink(ctx.memory_interact_events))
    for item in (first, second, other):
        ledger.register(item)
    ledger.append_evidence(EvidenceRecord(
        1, first, EVIDENCE_SUPPORT, (8701,), source_a, 1))
    ledger.append_evidence(EvidenceRecord(
        2, first, EVIDENCE_REFUTE, (8702,), source_b, 2))
    for evidence_id in range(3, 8):
        ledger.append_evidence(EvidenceRecord(
            evidence_id,
            second,
            EVIDENCE_SUPPORT,
            (8700 + evidence_id,),
            source_b,
            evidence_id,
        ))
    ledger.append_evidence(EvidenceRecord(
        8, other, EVIDENCE_SUPPORT, (8708,), source_a, 8))
    ctx.memory_interact_aggregates.rebuild_dirty(access=_ACCESS)


def _install_resolver(ctx: Any, source: SourceRef, core_ref: Any) -> tuple[Any, Any]:
    """安装同一 aggregate 上的 query compiler 和 overlay resolver。"""
    query_runtime = install_memory_query_runtime(
        ctx,
        _query_protocol(source),
        aggregates=ctx.memory_interact_aggregates,
    )
    resolver = MemoryOverlayResolver(
        ctx.memory_interact_aggregates,
        ctx.core_identity_catalog,
        _BaselineProvider(core_ref),
        _IndexFilterProvider(),
        _CurrentContextScorer(),
        StableTopKSourcePolicy(),
    )
    return query_runtime, install_memory_resolver_runtime(ctx, resolver)


def _batch_config() -> MemoryBatchRuntimeConfig:
    """构造 M-10/K-02 装配配置和显式资源预算。"""
    return MemoryBatchRuntimeConfig(
        _PROFILE,
        _HOT,
        SegmentDependency(
            MEMORY_BATCH_CORE_DEPENDENCY_KEY,
            (920, 10),
            (920, 11),
        ),
        SegmentBudget(8, 1_000_000),
        SegmentBudget(64, 2_000_000),
    )


def _publish_projection(ctx: Any, resolver: Any) -> MemoryCandidateProjectionManifest:
    """用小 segment 预算发布多页候选投影。"""
    return MemoryCandidateProjectionPublisher(
        resolver,
        ctx.tiered_segment_store,
    ).publish(
        _PROJECTION_KEY,
        access=_ACCESS,
        hypothesis_kinds=_KINDS,
        publication=MemoryProjectionPublication(
            (922, 1),
            _COLD,
            (923, 1),
            _DEPENDENCIES,
            SegmentBudget(1, 1_000_000),
            1,
        ),
    )


class _NoPrefetch:
    """冻结禁用预取的确定性策略。"""

    def should_prefetch(self, context: QueryPrefetchContext) -> bool:
        """核验上下文类型并返回固定禁用决策。"""
        if not isinstance(context, QueryPrefetchContext):
            raise TypeError("F-01 prefetch context 类型错误")
        return False

    def state_key(self) -> tuple[int, ...]:
        """返回固定策略身份。"""
        return 1, 2


def _open_query(ctx: Any, source: SourceRef, *, local_id: int = 1) -> Any:
    """打开 session/document/episode/query 生命周期。"""
    session = session_scope(
        1, owner=source.owner, versions=source.versions, source=source)
    document = document_scope(source, parent=session)
    episode = episode_scope(local_id, parent=document)
    query = query_scope(local_id, parent=episode)
    ctx.work_memory.begin_session(session)
    ctx.work_memory.begin_document(document)
    ctx.work_memory.begin_episode(episode)
    ctx.work_memory.begin_query(query)
    return query


def _current(ctx: Any, source: SourceRef, scope: Any) -> MemoryCurrentQuery:
    """物化当前 typed 输入并建立完整 query。"""
    ontology = ctx.graph_ontology
    occurrence = ontology.materialize(occurrence_identity(
        source, start=0, end=1, ordinal=0))
    span = ontology.materialize(span_identity(
        source, members=((0, 1),), ordinal=0))
    semantic = ontology.materialize(proposition_identity(source, (7301, 1)))
    structure = ontology.materialize(structure_concept_identity(
        (7302, 1), owner=source.owner, versions=source.versions))
    timestamp = LogicalClock(
        LogicalClockIdentity(scope, CLOCK_QUERY)).advance()
    return MemoryCurrentQuery(
        scope,
        source,
        timestamp,
        (occurrence,),
        (span,),
        (semantic,),
        (structure,),
        concept_identity((7303,), owner=source.owner, versions=source.versions),
        concept_identity((7304,), owner=source.owner, versions=source.versions),
        concept_identity((7305,), owner=source.owner, versions=source.versions),
        concept_identity((7306,), owner=source.owner, versions=source.versions),
    )


def _attractor_protocol(source: SourceRef) -> AttractorProtocol:
    """构造互异 agenda 生命周期状态身份。"""
    return AttractorProtocol(*tuple(
        _instruction(source, value) for value in range(9101, 9105)))


def _goals(
        source: SourceRef,
        scope: Any,
        *,
        count: int = 2,
        ) -> tuple[ReasoningObligation, ...]:
    """构造属于当前 query 的指定数量 typed 目标，不预置真值。"""
    if type(count) is not int or count <= 0:
        raise ValueError("goal count 必须是正严格整数")
    definitions = tuple(
        AtomicPropositionDefinition(
            proposition_identity(source, (9130, ordinal)),
            concept_identity(
                (9131, ordinal),
                owner=source.owner,
                versions=source.versions,
            ),
            occurrence_identity(
                source,
                start=ordinal,
                end=ordinal + 1,
                ordinal=ordinal,
            ),
            context_scope_identity(source, (9132, ordinal)),
            (),
        )
        for ordinal in range(1, count + 1)
    )
    graph = PropositionTemplateGraph(tuple(
        ScopedPropositionTemplate(
            definition,
            structure_concept_identity(
                (9133, ordinal),
                owner=source.owner,
                versions=source.versions,
            ),
        )
        for ordinal, definition in enumerate(definitions, start=1)
    ))
    failures = BindingFailureProtocol(*tuple(
        _instruction(source, value) for value in range(9111, 9120)))
    substituter = PropositionSubstituter(SubstitutionProtocol(
        _instruction(source, 9134), failures))
    state = LogicEvidenceState(True, False)
    return tuple(
        ReasoningObligation(
            substituter.substitute(
                definition.proposition, graph, BindingEnvironment()),
            state,
            source,
            scope,
        )
        for definition in definitions
    )


class _GoalMapper:
    """把两个 Memory 候选映射到两个当前目标。"""

    def project(
            self,
            request: Any,
            candidate: Any,
            obligations: tuple[ReasoningObligation, ...],
            ) -> tuple[AttractorActivationProposal, ...]:
        """只投影目标 kind，并稳定偏向第二个真实 Memory 候选。"""
        if candidate.hypothesis is None or request.hypothesis_kind != (7201,):
            return ()
        candidate_id = candidate.hypothesis.candidate_key[0]
        if candidate_id not in {1, 2}:
            return ()
        obligation = obligations[candidate_id - 1]
        dependency = AttractorDependency(
            request.query_kind, candidate.hypothesis)
        adjustment = 5000 if candidate_id == 2 else 0
        reason = AttractorScoreReason(
            _instruction(request.source, 9140 + candidate_id),
            adjustment,
            (dependency,),
        )
        return (AttractorActivationProposal(
            _instruction(request.source, 9143),
            obligation,
            adjustment,
            (reason,),
            (dependency,),
        ),)

    def clone_for_context(self, ctx: Any) -> "_GoalMapper":
        """返回无共享可变状态的 mapper。"""
        del ctx
        return _GoalMapper()

    def state_key(self) -> tuple[int, ...]:
        """返回 mapper 协议版本。"""
        return 1, 0


class _SupersedeChanged:
    """把命中依赖的未执行焦点替换为 superseded。"""

    def __init__(self, protocol: AttractorProtocol) -> None:
        self.protocol = protocol

    def recompute(self, activation: Any, update: Any) -> AttractorRecomputeDecision:
        """返回同一 activation 的低分 superseded 快照。"""
        reason = AttractorScoreReason(
            update.reason, -9000, update.changed_dependencies)
        return AttractorRecomputeDecision(
            -9000,
            (reason,),
            activation.dependencies,
            self.protocol.superseded,
        )

    def clone_for_context(self, ctx: Any) -> "_SupersedeChanged":
        """返回绑定同一不可变协议的新策略。"""
        del ctx
        return _SupersedeChanged(self.protocol)

    def state_key(self) -> tuple[int, ...]:
        """返回局部重算策略版本和协议身份。"""
        key = self.protocol.stable_key()
        return 1, len(key), *key


def _post_weaning_source(source_id: int) -> SourceRef:
    """构造受控阅读来源。"""
    return SourceRef(
        71,
        source_id,
        source_id,
        OwnerScope(),
        VersionBundle(
            CorpusVersion(1),
            ParserVersion(1),
            PrimitiveVersion(1),
            CurriculumVersion(1),
        ),
    )


class _PostWeaningParser:
    """把来源切片转换为一个来源化 Memory 候选。"""

    def __init__(self, source: SourceRef, candidate: int) -> None:
        self.source = source
        self.candidate = candidate

    def parse(self, source_slice: Any) -> ObservationIntakeDraft:
        """核验来源后返回 Observation/Hypothesis 草案。"""
        if source_slice.source != self.source:
            raise ValueError("F-01 parser 收到其他来源")
        context = MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_CONTEXT_SCOPE,
            (1000 + self.candidate,),
            self.source.owner,
            self.source.versions,
        ))
        signal = MemoryLinkedRef.object(_instruction(
            self.source, 2000 + self.candidate))
        return ObservationIntakeDraft(
            (3000 + self.candidate,),
            context,
            hypotheses=(HypothesisIntakeDraft(
                (4000 + self.candidate,),
                (5000 + self.candidate,),
                (6000 + self.candidate,),
                (7000 + self.candidate,),
                1,
                signal_ref=signal,
            ),),
        )


class _SourceTrustPolicy:
    """按显式 SourceRef 来源簇接受受控设施请求。"""

    def __init__(self, refs: tuple[Any, ...]) -> None:
        self.refs = refs

    def state_key(self) -> tuple[int, ...]:
        """返回全部图身份组成的固定状态。"""
        result = [1, len(self.refs)]
        for ref in self.refs:
            key = ref.stable_key()
            result.extend((len(key), *key))
        return tuple(result)

    def assess(self, request: Any) -> SourceTrustAssessment:
        """只按来源协议字段形成准入，不解释原文或许可词面。"""
        return SourceTrustAssessment(
            request.stable_key(),
            self.state_key(),
            SOURCE_ADMISSION_ACCEPTED,
            (request.source.source_kind, request.source.source_id),
            (
                request.source.versions.corpus.value,
                request.source.versions.parser.value,
            ),
            self.refs[0],
            self.refs[1],
            self.refs[2],
            (self.refs[4],),
            (),
            (self.refs[3].local_id,),
        )

    def clone_for_context(self, ctx: Any) -> "_SourceTrustPolicy":
        """核验 clone 图可回读相同 refs 后复用不可变 policy。"""
        for ref in self.refs:
            ctx.graph_ontology.identity_of(ref)
        return self


def _install_source_trust(ctx: Any, routes: PostWeaningRouteProtocol) -> Any:
    """物化 policy 概念并安装真实 A-05 source admission runtime。"""
    if ctx.source_trust_runtime is not None:
        return ctx.source_trust_runtime
    from pure_integer_ai.cognition.shared.graph_ontology import (
        relation_concept_identity,
    )

    refs = tuple(
        ctx.graph_ontology.materialize(relation_concept_identity((value,)))
        for value in range(20201, 20206)
    )
    return install_source_admission_runtime(
        ctx,
        _SourceTrustPolicy(refs),
        reading_route=routes.reading,
        interaction_route=routes.interaction,
        record_only_routes=(routes.external_define,),
    )


def _post_weaning_manifest(ctx: Any, source: SourceRef) -> tuple[Any, Any]:
    """从当前 K/M/A owner 形成可重建 dry-run manifest。"""
    routes = PostWeaningRouteProtocol(*tuple(
        _instruction(source, value) for value in range(20101, 20105)))
    checks = tuple(sorted((
        PostWeaningFacilityCheck(
            _instruction(source, 20110), True, (20111, 1)),
        PostWeaningFacilityCheck(
            _instruction(source, 20112), True, (20113, 1)),
    ), key=lambda item: item.requirement.stable_key()))
    probe = PostWeaningFacilityProbe(checks, (20114, 1))
    _install_source_trust(ctx, routes)
    manifest = build_post_weaning_dry_run_manifest(
        ctx,
        runtime_owner=_instruction(source, 20115),
        fixture_artifact_key=(20116, 1),
        routes=routes,
        probe=probe,
        budget=PostWeaningResourceBudget(8, 64, 1, 1),
        trace=(20117, 1),
    )
    return routes, manifest


def _install_post_weaning_consumers(
        ctx: Any,
        source: SourceRef,
        projection: MemoryCandidateProjectionManifest,
        ) -> None:
    """安装 K-04、A-10 和 M-08 的生产消费链。"""
    install_memory_hot_set_runtime(
        ctx,
        projection,
        QueryHotSetPolicy(
            SegmentBudget(4, 4_000_000),
            SegmentBudget(1, 1_000_000),
            _NoPrefetch(),
            8,
        ),
    )
    protocol = _attractor_protocol(source)
    install_attractor_runtime(
        ctx,
        protocol,
        AttractorBudget(2, 4, 4),
        _GoalMapper(),
        _SupersedeChanged(protocol),
    )
    install_memory_use_runtime(ctx)


def _close_outer_lifecycle(ctx: Any) -> None:
    """按逆序关闭 question caller 保留的全部 WorkMemory scope。"""
    work = ctx.work_memory
    if work.active_generation_scope is not None:
        work.end_generation()
    if work.active_query_scope is not None:
        work.end_query()
    if work.active_episode_scope is not None:
        work.end_episode()
    if work.active_document_scope is not None:
        work.end_document()
    if work.active_session_scope is not None:
        work.end_session()


def _complete_source(
        repository: SourceRecordRepository,
        trace: Any,
        ordinal: int,
        ) -> MemoryGenerationSource:
    """为一条 resolver 来源分账建立完整 SourceRecord。"""
    record = repository.find(trace.source.stable_key())
    if record is None:
        record = repository.put_complete(
            trace.source.stable_key(),
            f"设施来源{ordinal}",
            metadata=SourceRecordMetadata(
                "facility-license",
                ordinal,
                100 + ordinal,
                200 + ordinal,
                300 + ordinal,
            ),
        )
    elif not record.metadata_complete:
        raise ValueError("既有设施来源缺少完整 SourceRecord metadata")
    return MemoryGenerationSource.from_record(trace, record)


def _events(ctx: Any, event_kind: int) -> tuple[Any, ...]:
    """读取设施 owner 可见的指定交互 Memory 事件。"""
    return ctx.memory_interact_events.query(
        access=_ACCESS, event_kind=event_kind)


class _EmptyQuestionExecutor:
    """返回真实执行过但没有候选的 Memory OFF 对照 route。"""

    def __init__(self, reason: ObjectIdentity) -> None:
        self.reason = reason

    def execute(self, query: Any) -> QuestionExecutionResult:
        """返回同次 query 的空候选，不读取 Memory 或 expected。"""
        return QuestionExecutionResult(query, self.reason, (), (19811, 1))


class _UseBeforePostcheckMapper:
    """要求真实 Use 已提交后才委托同次 generation 复核。"""

    def __init__(self, ctx: Any, delegate: Any) -> None:
        self.ctx = ctx
        self.delegate = delegate

    def build(self, request: Any, query: Any, result: Any, generation: Any) -> Any:
        """核验事件日志存在 Use，再建立 G-04 请求。"""
        if not _events(self.ctx, MEMORY_EVENT_USE):
            raise RuntimeError("F-01 postcheck 前缺少真实 Memory Use")
        return self.delegate.build(request, query, result, generation)


def _outcome_protocol(source: SourceRef, postchecker: Any) -> Any:
    """按 G-04 维度注入全部 applicability/verdict outcome 路由。"""
    pairs = (
        (APPLICABILITY_APPLICABLE, VERDICT_SUPPORT),
        (APPLICABILITY_APPLICABLE, VERDICT_REFUTE),
        (APPLICABILITY_APPLICABLE, VERDICT_UNKNOWN),
        (APPLICABILITY_APPLICABLE, VERDICT_CONFLICTED),
        (APPLICABILITY_NOT_APPLICABLE, VERDICT_UNKNOWN),
        (APPLICABILITY_UNKNOWN, VERDICT_UNKNOWN),
    )
    routes = []
    for route_index, (dimension, verifier) in enumerate(
            postchecker.protocol.bindings(), start=1):
        values = tuple(
            MemoryGenerationOutcomeValueRoute(
                applicability,
                verdict,
                MemoryLinkedRef.object(_instruction(
                    source,
                    19_900 + route_index * 10 + value_index,
                )),
            )
            for value_index, (applicability, verdict) in enumerate(
                pairs, start=1)
        )
        routes.append(MemoryGenerationOutcomeRoute(
            dimension,
            verifier,
            MemoryLinkedRef.object(_instruction(source, 19_800 + route_index)),
            values,
        ))
    return MemoryGenerationOutcomeProtocol(tuple(routes))


def _question_dialogue(
        ctx: Any,
        source: SourceRef,
        observation: Any,
        *,
        target_index: int = 1,
        obligation_factory: Any = None,
        ) -> tuple[Any, Any]:
    """装配走 K-04、A-10、M-08、G-00..G-05 的完整 question caller。"""
    scope = _open_query(ctx, source)
    current = _current(ctx, source, scope)
    compilation = ctx.memory_query_runtime.compile(current, access=_ACCESS)
    resolution = ctx.memory_resolver_runtime.resolve(compilation)
    repository = SourceRecordRepository(ctx.backend)
    traces = {
        trace.source.stable_key(): trace
        for candidate_set in resolution.sets
        for candidate in candidate_set.candidates
        for trace in candidate.memory_source_traces
    }
    for ordinal, trace in enumerate(
            (traces[key] for key in sorted(traces)), start=1):
        _complete_source(repository, trace, ordinal)
    goals = (
        _goals(source, scope)
        if obligation_factory is None
        else obligation_factory(source, scope)
    )
    if (not isinstance(goals, tuple) or not goals
            or any(not isinstance(item, ReasoningObligation)
                   for item in goals)):
        raise TypeError("question obligations 类型错误")
    if any(item.source != source or item.scope != scope for item in goals):
        raise ValueError("question obligations 不属于当前 query")
    if (type(target_index) is not int
            or target_index < 0
            or target_index >= len(goals)):
        raise ValueError("question target_index 超出目标范围")
    ctx.work_memory.end_query()
    executor = ResolvedMemoryQuestionExecutor(
        ctx,
        current,
        _ACCESS,
        goals,
        executed_reason=_instruction(source, 20140),
        binding_reason=_instruction(source, 20141),
        trace_prefix=(20142, 1),
        source_records=repository,
    )
    committer = MemoryQuestionSelectionCommitter(
        ctx,
        consumer=_instruction(source, 20143),
        input_observation_ref=observation.event.object_ref,
        influence_kind=MemoryLinkedRef.object(_instruction(source, 20144)),
        trace_prefix=(20145, 1),
    )
    mapper, postchecker = build_postcheck_owners()
    fixture = build_question_fixture(
        executor_factory=lambda route: executor,
        world=(source, scope, goals[target_index].proposition),
        selection_committer=committer,
        postcheck_mapper=mapper,
        postchecker=postchecker,
    )
    dialogue = MemoryAwareQuestionDialogueRuntime(
        ctx,
        fixture.runtime,
        trace_prefix=(20146, 1),
        source_records=repository,
    )
    return fixture, dialogue


def _observation(ctx: Any, observation_ref: Any) -> Any:
    """读取恢复后唯一的真实 Observation 事件。"""
    matches = ctx.memory_interact_events.query(
        access=_ACCESS,
        event_kind=MEMORY_EVENT_OBSERVATION,
        object_ref=observation_ref,
    )
    if len(matches) != 1:
        raise RuntimeError("F-01 恢复场景缺少唯一 Observation")
    return matches[0]


def _run_question_once(
        ctx: Any,
        source: SourceRef,
        observation: Any,
        ) -> tuple[Any, ...]:
    """在同一恢复状态执行一次真实 J-G 问答并返回分账测量。"""
    fixture, dialogue = _question_dialogue(ctx, source, observation)
    try:
        _, manifest = _post_weaning_manifest(ctx, source)
        uses_before = len(_events(ctx, MEMORY_EVENT_USE))
        operation = PostWeaningDryRunRuntime(
            ctx, manifest).run_question(dialogue, fixture.request)
        uses_after = len(_events(ctx, MEMORY_EVENT_USE))
        question = operation.result.question
        return (
            100 if question.complete else 0,
            uses_after - uses_before,
            fixture.request.stable_key(),
            question.generation.rendered.stable_key(),
        )
    finally:
        fixture.close()
        _close_outer_lifecycle(ctx)


def _refresh_projection(ctx: Any) -> MemoryCandidateProjectionManifest:
    """重建脏 aggregate 并发布与当前 Memory 一致的投影。"""
    ctx.memory_read_aggregates.rebuild_dirty(access=_ACCESS)
    ctx.memory_interact_aggregates.rebuild_dirty(access=_ACCESS)
    projection = _publish_projection(
        ctx, ctx.memory_resolver_runtime.resolver)
    ctx.memory_hot_set_runtime.replace_projection(projection)
    return projection


def _event_count(ctx: Any, event_kind: int) -> int:
    """跨阅读和交互空间统计一种真实事件。"""
    return sum(
        len(log.query(access=_ACCESS, event_kind=event_kind))
        for log in (ctx.memory_read_events, ctx.memory_interact_events)
    )


def prepare_facility_context(eval_ctx: Any) -> None:
    """在 Core 冻结前安装主纵切所需的真实 M/K/A/PW-00 owner。"""
    eval_ctx.work_memory.end_session()
    install_memory_batch_runtimes(eval_ctx, _batch_config())
    _seed_memory(eval_ctx)
    source = _query_source(document_id=1)
    core_refs = _core_refs(eval_ctx)
    _, resolver_runtime = _install_resolver(eval_ctx, source, core_refs[1])
    observation = _append_observation(eval_ctx, source, core_refs)
    projection = _publish_projection(eval_ctx, resolver_runtime.resolver)
    _install_post_weaning_consumers(eval_ctx, source, projection)
    routes, _ = _post_weaning_manifest(eval_ctx, source)
    warm_fixture, _ = _question_dialogue(eval_ctx, source, observation)
    warm_fixture.close()
    _close_outer_lifecycle(eval_ctx)
    routes, manifest = _post_weaning_manifest(eval_ctx, source)
    eval_ctx.f01_source = source
    eval_ctx.f01_observation = observation
    eval_ctx.f01_projection = projection
    eval_ctx.f01_routes = routes
    eval_ctx.f01_manifest = manifest


def run_main_memory_path(ctx: Any) -> MemoryPathEvidence:
    """执行双来源准入、Memory OFF/ON、Use 和逐维 outcome。"""
    source = ctx.f01_source
    routes = ctx.f01_routes
    intake_runtime = PostWeaningDryRunRuntime(ctx, ctx.f01_manifest)
    first_source = _post_weaning_source(601)
    second_source = replace(first_source, document_id=602)
    for ordinal, admitted_source in enumerate(
            (first_source, second_source), start=1):
        intake_runtime.run_intake(PostWeaningIntakeRequest(
            routes.reading,
            admitted_source,
            f"F-01 来源 {ordinal}",
            f"license-f01-{ordinal}",
            52600 + ordinal,
            parser=_PostWeaningParser(admitted_source, 40 + ordinal),
            trace=(52601, ordinal),
        ))
    admission_records = tuple(
        ctx.source_trust_records.find(item.stable_key())
        for item in (first_source, second_source)
    )
    assessments = tuple(
        SourceTrustAssessment.from_stable_key(item.assessment_key)
        for item in admission_records
        if item is not None
    )
    admissions = sum(
        item.decision == SOURCE_ADMISSION_ACCEPTED for item in assessments)
    clusters = len({item.source_cluster_key for item in assessments})
    _close_outer_lifecycle(ctx)
    _refresh_projection(ctx)

    scope = _open_query(ctx, source)
    current = _current(ctx, source, scope)
    compilation = ctx.memory_query_runtime.compile(current, access=_ACCESS)
    resolution = ctx.memory_resolver_runtime.resolve(compilation)
    candidates = tuple(
        item
        for candidate_set in resolution.sets
        for item in candidate_set.candidates
    )
    conflicts = sum(
        item.aggregate is not None
        and item.aggregate.support_count > 0
        and item.aggregate.contradict_count > 0
        for item in candidates
    )
    repository = SourceRecordRepository(ctx.backend)
    traces = {
        trace.source.stable_key(): trace
        for item in candidates
        for trace in item.memory_source_traces
    }
    for ordinal, trace in enumerate(
            (traces[key] for key in sorted(traces)), start=1):
        _complete_source(repository, trace, ordinal)
    goals = _goals(source, scope)
    ctx.work_memory.end_query()
    target = goals[1].proposition
    off_fixture = build_question_fixture(
        executor_factory=lambda route: _EmptyQuestionExecutor(
            _instruction(source, 20140)),
        world=(source, current.scope, target),
    )
    on_fixture = None
    try:
        off_dialogue = MemoryAwareQuestionDialogueRuntime(
            ctx,
            off_fixture.runtime,
            trace_prefix=(20146, 1),
            source_records=repository,
        )
        off_run = off_dialogue.run(off_fixture.request)
        executor = ResolvedMemoryQuestionExecutor(
            ctx,
            current,
            _ACCESS,
            goals,
            executed_reason=_instruction(source, 20140),
            binding_reason=_instruction(source, 20141),
            trace_prefix=(20142, 1),
            source_records=repository,
        )
        committer = MemoryQuestionSelectionCommitter(
            ctx,
            consumer=_instruction(source, 20143),
            input_observation_ref=ctx.f01_observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(_instruction(source, 20144)),
            trace_prefix=(20145, 1),
        )
        mapper, postchecker = build_postcheck_owners()
        ordered_mapper = _UseBeforePostcheckMapper(ctx, mapper)
        outcome_committer = MemoryQuestionOutcomeCommitter(
            ctx.memory_use_runtime,
            _outcome_protocol(source, postchecker),
            trace_prefix=(52618, 1),
        )
        on_fixture = build_question_fixture(
            executor_factory=lambda route: executor,
            world=(source, current.scope, target),
            selection_committer=committer,
            postcheck_mapper=ordered_mapper,
            postchecker=postchecker,
            outcome_committer=outcome_committer,
        )
        on_dialogue = MemoryAwareQuestionDialogueRuntime(
            ctx,
            on_fixture.runtime,
            trace_prefix=(20146, 1),
            source_records=repository,
        )
        _, question_manifest = _post_weaning_manifest(ctx, source)
        operation = PostWeaningDryRunRuntime(
            ctx, question_manifest).run_question(
                on_dialogue, on_fixture.request)
        question = operation.result.question
        result_identity = CanonicalIdentity.from_value((
            question.status,
            question.query.request.target,
            question.generation.rendered.stable_key(),
        ))
        resources_closed = (
            operation.report.query_closed
            and ctx.memory_hot_set_runtime.query_resources_closed()
            and ctx.work_memory.active_query_scope is None
            and ctx.work_memory.attractor_state is None
        )
        _close_outer_lifecycle(ctx)
        return MemoryPathEvidence(
            positive_behavior=100 if question.complete else 0,
            negative_behavior=(
                100 if off_run.question.selection.selected_candidate_keys else 0),
            admissions=admissions,
            source_clusters=clusters,
            candidates=len(candidates),
            conflicts=conflicts,
            uses=_event_count(ctx, MEMORY_EVENT_USE),
            outcomes=_event_count(ctx, MEMORY_EVENT_USE_OUTCOME),
            query_key=off_fixture.request.stable_key(),
            query_before=CanonicalIdentity.from_value(
                off_fixture.request.stable_key()),
            query_after=CanonicalIdentity.from_value(
                on_fixture.request.stable_key()),
            resources_closed=(
                resources_closed
                and ctx.work_memory.active_session_scope is None),
            result_identity=result_identity,
            observation_ref=ctx.f01_observation.event.object_ref,
            source=source,
        )
    finally:
        off_fixture.close()
        if on_fixture is not None:
            on_fixture.close()


def run_clone_history_check(
        ctx: Any,
        evidence: MemoryPathEvidence,
        ) -> tuple[Any, Any, CanonicalIdentity, CanonicalIdentity]:
    """在两个 V-06 clone 扰动上一 episode 并核验宿主零写。"""
    _refresh_projection(ctx)
    host_before = CanonicalIdentity.from_value(
        ctx.backend.recovery_state_snapshot())
    measurements = []
    for variant in (1, 2):
        with isolated_evaluation(ctx, label=f"f01-history-{variant}") as clone:
            clone.work_memory.end_session()
            clone.work_memory.pr_vector[(52620,)] = variant
            measurements.append(_run_question_once(
                clone,
                evidence.source,
                _observation(clone, evidence.observation_ref),
            ))
    host_after = CanonicalIdentity.from_value(
        ctx.backend.recovery_state_snapshot())
    return measurements[0], measurements[1], host_before, host_after


def run_rollback_check(
        ctx: Any,
        evidence: MemoryPathEvidence,
        ) -> tuple[bool, CanonicalIdentity, CanonicalIdentity]:
    """在真实 question 已写 Use 后注入异常并比较完整恢复状态。"""
    _refresh_projection(ctx)
    with isolated_evaluation(ctx, label="f01-rollback") as clone:
        clone.work_memory.end_session()
        fixture, dialogue = _question_dialogue(
            clone,
            evidence.source,
            _observation(clone, evidence.observation_ref),
        )
        try:
            _, manifest = _post_weaning_manifest(clone, evidence.source)
            runtime = PostWeaningDryRunRuntime(clone, manifest)
            before = CanonicalIdentity.from_value(
                clone.backend.recovery_state_snapshot())
            original = dialogue.run

            def fail_after_question(request: Any) -> None:
                """先完成真实问答，再模拟调用边界故障。"""
                original(request)
                raise RuntimeError("F-01 rollback injection")

            dialogue.run = fail_after_question
            failed = False
            try:
                runtime.run_question(dialogue, fixture.request)
            except RuntimeError as exc:
                if str(exc) != "F-01 rollback injection":
                    raise
                failed = True
            after = CanonicalIdentity.from_value(
                clone.backend.recovery_state_snapshot())
            return failed, before, after
        finally:
            fixture.close()


def _restore_runtime(
        backend: Any,
        projection_key: tuple[int, ...],
        ) -> tuple[Any, SourceRef, Any]:
    """从已加载后端和投影 manifest 重装同一 PW-00 消费链。"""
    ctx = make_train_context(backend, companion=True)
    install_memory_batch_runtimes(ctx, _batch_config())
    source = _query_source(document_id=1)
    _install_resolver(ctx, source, _core_refs(ctx)[1])
    projection = MemoryCandidateProjectionManifest.from_stable_key(
        projection_key)
    projection.validate_store(ctx.tiered_segment_store)
    _install_post_weaning_consumers(ctx, source, projection)
    _, manifest = _post_weaning_manifest(ctx, source)
    return ctx, source, PostWeaningDryRunRuntime(ctx, manifest)


def run_cross_backend_migration(
        ctx: Any,
        evidence: MemoryPathEvidence,
        run_dir: Path,
        ) -> tuple[bool, CanonicalIdentity, CanonicalIdentity]:
    """把 Dict fresh 状态打包到 SQLite 并重跑同一 PW-00 question。"""
    projection = _refresh_projection(ctx)
    spaces = [
        row["space_id"]
        for row in ctx.backend.select("space", order_by="space_id")
    ]
    cursor = CursorState(
        base_run_id="f01-fixture",
        run_id="f01-migrate",
        completed={1},
        non_skippable={2},
    )
    dumped = dump_run(
        ctx.backend,
        str(run_dir),
        "f01-migrate",
        spaces=spaces,
        tables=None,
        require_all_spaces=True,
        versions=evidence.source.versions,
        cursor_state=cursor,
    )
    target_backend = SQLiteBackend(":memory:")
    fixture = None
    try:
        schema_ctx = make_train_context(target_backend, companion=True)
        install_memory_batch_runtimes(schema_ctx, _batch_config())
        loaded = load_run_package(
            target_backend,
            str(run_dir),
            "f01-migrate",
            expected_versions=evidence.source.versions,
            expected_dependencies=(),
            expected_publish_epoch=1,
        )
        restored_cursor = cursor_state_from_payload(
            loaded.cursor_payload,
            fallback_run_id="f01-migrate",
        )
        target, source, runtime = _restore_runtime(
            target_backend, projection.stable_key())
        observations = target.memory_interact_events.query(
            access=_ACCESS,
            event_kind=MEMORY_EVENT_OBSERVATION,
            object_ref=evidence.observation_ref,
        )
        fixture, dialogue = _question_dialogue(
            target, source, observations[0])
        resumed = runtime.run_question(dialogue, fixture.request)
        question = resumed.result.question
        restored_identity = CanonicalIdentity.from_value((
            question.status,
            question.query.request.target,
            question.generation.rendered.stable_key(),
        ))
        passed = (
            dumped == spaces
            and bool(loaded.loaded_tables)
            and restored_cursor == cursor
            and len(observations) == 1
            and resumed.report.core_unchanged
            and resumed.report.query_closed
        )
        return passed, evidence.result_identity, restored_identity
    finally:
        if fixture is not None:
            fixture.close()
        target_backend.close()


def run_reparse_evidence() -> Any:
    """委托独立 production A-08 场景执行重解析和 replay。"""
    from pure_integer_ai.experiments.facility_reparse_scenario import (
        run_reparse_evidence as execute,
    )

    return execute()


def run_capability_evidence() -> Any:
    """委托独立 production C-02 场景执行能力复用。"""
    from pure_integer_ai.experiments.facility_capability_scenario import (
        run_capability_evidence as execute,
    )

    return execute()


def published_worker_bytes(worker_count: int) -> tuple[Any, ...]:
    """委托独立 production K-03 场景发布 worker barrier 字节。"""
    from pure_integer_ai.experiments.facility_worker_scenario import (
        published_worker_bytes as execute,
    )

    return execute(worker_count)


__all__ = [
    "MemoryPathEvidence",
    "prepare_facility_context",
    "published_worker_bytes",
    "run_capability_evidence",
    "run_clone_history_check",
    "run_cross_backend_migration",
    "run_main_memory_path",
    "run_reparse_evidence",
    "run_rollback_check",
]
