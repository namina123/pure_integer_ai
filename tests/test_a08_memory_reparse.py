"""A-08 长期 Memory 重解析、多对多降阶和恢复对抗。"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pure_integer_ai.cognition.shared.formal_artifact import ArtifactSchema
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.memory_batch import (
    FAULT_MEMORY_BATCH_AFTER_EVENT,
    MemoryBatchRuntimeConfig,
    install_memory_batch_runtimes,
)
from pure_integer_ai.cognition.shared.memory_event import (
    EpisodePayload,
    MEMORY_EVENT_DERIVATION,
    MEMORY_EVENT_EPISODE,
    MEMORY_EVENT_USE,
    MEMORY_OBJECT_EPISODE,
    MEMORY_OBJECT_USE,
    MemoryEvent,
    MemoryLinkedRef,
    UsePayload,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_query import (
    MemoryQueryDefinition,
    MemoryQueryProtocol,
    MemoryQueryRoles,
)
from pure_integer_ai.cognition.shared.memory_resolver import (
    ActivationScore,
    ActivationScoreReason,
    MemoryAggregateFilter,
    SourceDiversityAssessment,
)
from pure_integer_ai.cognition.shared.memory_resolver_engine import (
    MemoryOverlayResolver,
)
from pure_integer_ai.cognition.shared.parser_revision import (
    ParserHypothesisRevision,
    ParserRevisionGraph,
    ParserRevisionProtocol,
    ParserRevisionRequest,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_CREATED,
    CLOCK_MEMORY_USED,
    LogicalClockIdentity,
    document_scope,
    episode_scope,
    session_scope,
)
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft,
    ObservationIntakeDraft,
)
from pure_integer_ai.experiments.memory_reparse_runtime import (
    MemoryParserRevisionError,
    MemoryParserRevisionRuntime,
)
from pure_integer_ai.experiments.memory_query_runtime import (
    install_memory_query_runtime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import (
    DictBackend,
    SQLiteBackend,
    StorageBackend,
)
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.memory_batch import (
    MEMORY_BATCH_CORE_DEPENDENCY_KEY,
)
from pure_integer_ai.storage.placement import (
    TemperatureProfile,
    TemperatureTier,
)
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from pure_integer_ai.storage.segment_dependency import SegmentDependency
from pure_integer_ai.storage.telemetry import collect_backend_telemetry

from test_m06_memory_query import (
    _close_query,
    _current,
    _open_query,
)


_TEXT = "甲乙丙丁"
_LICENSE = "license-a08"
_ACCESS = MemoryAccessContext(0, 0, 0)
_PROFILE = TemperatureProfile(
    (24800, 1),
    (
        TemperatureTier((24800, 1), 0),
        TemperatureTier((24800, 2), 1),
    ),
)


def _source(parser: int) -> SourceRef:
    """构造除 ParserVersion 外完全一致的全局来源。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        24801,
        1,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(parser)),
    )


def _batch_config() -> MemoryBatchRuntimeConfig:
    """构造 A-08 专项使用的 M-10 批次依赖和资源预算。"""
    return MemoryBatchRuntimeConfig(
        _PROFILE,
        (24800, 1),
        SegmentDependency(
            MEMORY_BATCH_CORE_DEPENDENCY_KEY,
            (24802, 1),
            (24802, 2),
        ),
        SegmentBudget(16, 2_000_000),
        SegmentBudget(128, 8_000_000),
    )


def _revision_protocol() -> ParserRevisionProtocol:
    """注入 A-03 Artifact 类型、图关系、元数据和映射预算。"""
    kinds = tuple(concept_identity((24810, item)) for item in range(1, 4))
    relations = tuple(
        relation_concept_identity((24811, item)) for item in range(1, 7))
    return ParserRevisionProtocol(
        kinds[0],
        kinds[1],
        kinds[2],
        ArtifactSchema(
            concept_identity((24812, 1)),
            concept_identity((24812, 2)),
        ),
        *relations,
        7,
        2,
        1,
        (13, 17),
        8,
        8,
        4,
    )


def _hypothesis(source: SourceRef, candidate: int) -> HypothesisKey:
    """按候选编号构造与 M-05 草案完全相同的 Hypothesis。"""
    return HypothesisKey(
        (24820, 1),
        (24821, candidate),
        (24822, candidate),
        document_scope(source),
        source,
    )


def _context_ref(source: SourceRef) -> MemoryLinkedRef:
    """构造来源版本化的上下文引用。"""
    return MemoryLinkedRef.object(minimal_instruction_identity(
        (24830, source.versions.parser.value),
        owner=source.owner,
        versions=source.versions,
    ))


def _signal_ref(source: SourceRef, candidate: int) -> MemoryLinkedRef:
    """构造每个候选独立的来源化 Evidence 信号。"""
    return MemoryLinkedRef.object(minimal_instruction_identity(
        (24831, candidate),
        owner=source.owner,
        versions=source.versions,
    ))


class _OldParser:
    """产生五个旧候选及彼此互异的 M-05 lineage。"""

    def __init__(self, source: SourceRef) -> None:
        """绑定旧 ParserVersion 来源。"""
        self.source = source

    def parse(self, source_slice) -> ObservationIntakeDraft:
        """返回覆盖唯一、拆分、合并和删除四类旧端点的草案。"""
        return ObservationIntakeDraft(
            (24840, 1),
            _context_ref(self.source),
            hypotheses=tuple(
                HypothesisIntakeDraft(
                    (24841, candidate),
                    (24820, 1),
                    (24821, candidate),
                    (24822, candidate),
                    1,
                    signal_ref=_signal_ref(self.source, candidate),
                )
                for candidate in range(1, 6)
            ),
        )


class _NewParser:
    """产生一对一、拆分和合并目标，并可注入错误 split lineage。"""

    def __init__(self, source: SourceRef, *, invalid_split: bool = False) -> None:
        """绑定新来源和可选的多目标私选故障。"""
        self.source = source
        self.invalid_split = invalid_split
        self.calls = 0

    def parse(self, source_slice) -> ObservationIntakeDraft:
        """返回四个新候选；只有双向唯一目标沿用旧 lineage。"""
        self.calls += 1
        lineages = (
            (24841, 1),
            ((24841, 2) if self.invalid_split else (24842, 6)),
            (24842, 7),
            (24842, 8),
        )
        return ObservationIntakeDraft(
            (24840, 2),
            _context_ref(self.source),
            hypotheses=tuple(
                HypothesisIntakeDraft(
                    lineages[index],
                    (24820, 1),
                    (24821, candidate),
                    (24822, candidate),
                    1,
                    signal_ref=_signal_ref(self.source, candidate),
                )
                for index, candidate in enumerate(range(6, 10))
            ),
        )


class _SingleParser:
    """V-02 manifest 链每个版本只产生一个同 lineage 候选。"""

    def __init__(self, source: SourceRef) -> None:
        """绑定当前 ParserVersion 来源。"""
        self.source = source

    def parse(self, source_slice) -> ObservationIntakeDraft:
        """返回固定派生 lineage 和版本化候选身份。"""
        del source_slice
        parser = self.source.versions.parser.value
        return ObservationIntakeDraft(
            (24900, 1),
            _context_ref(self.source),
            hypotheses=(HypothesisIntakeDraft(
                (24901, 1),
                (24902, 1),
                (24903, parser),
                (24904, 1),
                1,
                signal_ref=_signal_ref(self.source, 100 + parser),
            ),),
        )


class _RejectParser:
    """重放或 M-10 恢复期间被调用即失败。"""

    def parse(self, source_slice):
        """拒绝任何重复 parser 执行。"""
        pytest.fail("A-08 exact replay 不得重跑 parser")


class _FailOnce:
    """在指定 M-10 事件序号首次命中时中断。"""

    def __init__(self) -> None:
        """初始化尚未触发的故障注入器。"""
        self.triggered = False

    def hit(self, point: int, context: dict[str, int]) -> None:
        """只在第二个物理事件后中断一次。"""
        if (not self.triggered
                and point == FAULT_MEMORY_BATCH_AFTER_EVENT
                and context.get("event_ordinal") == 1):
            self.triggered = True
            raise RuntimeError("A-08 M-10 fault")


class _NoCoreBaseline:
    """A-08 消费者验收不注入额外 Core 基线候选。"""

    def candidates(self, request):
        """返回空 Core 候选集。"""
        del request
        return ()

    def state_key(self) -> tuple[int, ...]:
        """返回测试 baseline 协议版本。"""
        return (1,)


class _LifecycleFilter:
    """可省略或显式请求 lifecycle 的 M-07 索引 provider。"""

    def __init__(self, lifecycle_state: int | None) -> None:
        """保存调用方请求的可选 lifecycle。"""
        self.lifecycle_state = lifecycle_state

    def filters(self, request):
        """返回一个不附加词面、来源或领域常量的过滤分支。"""
        del request
        return (MemoryAggregateFilter(
            lifecycle_state=self.lifecycle_state),)

    def state_key(self) -> tuple[int, ...]:
        """返回 provider 版本和可选 lifecycle。"""
        return (
            1,
            0 if self.lifecycle_state is None else self.lifecycle_state,
        )


class _CandidateScorer:
    """按候选身份给出确定性整数分的测试 scorer。"""

    def score(self, request, hypothesis, aggregate, sources):
        """只用当前候选键形成分数，不读取历史输出或 reward。"""
        del request, aggregate, sources
        value = hypothesis.candidate_key[-1]
        return ActivationScore(
            value,
            (ActivationScoreReason((24880, value), value),),
        )

    def state_key(self) -> tuple[int, ...]:
        """返回 scorer 协议版本。"""
        return (1,)


class _StableDiversity:
    """保持候选集合和来源计数的无调整多样性策略。"""

    def assess(self, request, hypothesis, aggregate, sources, source_traces):
        """报告完整来源数，不改变基础分。"""
        del request, hypothesis, aggregate, sources
        source_count = len({item.source_cluster_key for item in source_traces})
        return SourceDiversityAssessment(source_count, 0, ())

    def select(self, request, candidates, budget):
        """按 resolver 已形成的稳定顺序选择精确 Top-K。"""
        del request
        return candidates[:min(budget, len(candidates))]

    def state_key(self) -> tuple[int, ...]:
        """返回多样性策略协议版本。"""
        return (1,)


@dataclass
class _World:
    """集中持有 A-08 图、Memory、请求和新旧候选。"""

    backend: StorageBackend
    ctx: object
    runtime: MemoryParserRevisionRuntime
    request: ParserRevisionRequest
    old_source: SourceRef
    new_source: SourceRef
    old_hypotheses: tuple[HypothesisKey, ...]
    new_hypotheses: tuple[HypothesisKey, ...]
    old_result: object
    use_ref: object


def _append_use(world_ctx, source: SourceRef, old_result) -> MemoryObjectRef:
    """为第一个旧候选写带输出 Episode 和一条显式 Use。"""
    event_log = world_ctx.memory_read_events
    document = document_scope(source)
    episode = episode_scope(1, parent=document)
    session = session_scope(
        1,
        owner=source.owner,
        versions=source.versions,
    )
    output = world_ctx.graph_ontology.materialize(
        minimal_instruction_identity((24850, 1)))
    created = event_log.scoped_identities.resume_clock(
        LogicalClockIdentity(episode, CLOCK_MEMORY_CREATED)).advance()
    episode_payload = EpisodePayload(
        old_result.observation_ref,
        None,
        (),
        None,
        MemoryLinkedRef.core(output),
        (old_result.hypothesis_refs[0],),
        (),
        None,
        1,
        session,
        created,
    )
    episode_ref = memory_object_ref(
        event_log.memory_space_identity,
        MEMORY_OBJECT_EPISODE,
        episode_payload.stable_key(),
        owner=source.owner,
        versions=source.versions,
    )
    event_log.append(MemoryEvent(
        MEMORY_EVENT_EPISODE,
        episode_ref,
        episode,
        episode_payload,
    ))
    used_at = event_log.scoped_identities.resume_clock(
        LogicalClockIdentity(episode, CLOCK_MEMORY_USED)).advance()
    use_payload = UsePayload(
        old_result.hypothesis_refs[0],
        episode_ref,
        MemoryLinkedRef.core(output),
        None,
        used_at,
    )
    use_ref = memory_object_ref(
        event_log.memory_space_identity,
        MEMORY_OBJECT_USE,
        use_payload.identity_key(),
        owner=source.owner,
        versions=source.versions,
    )
    event_log.append(MemoryEvent(
        MEMORY_EVENT_USE,
        use_ref,
        episode,
        use_payload,
    ))
    world_ctx.memory_read_aggregates.rebuild_dirty(access=_ACCESS)
    return use_ref


def _world(backend_type) -> _World:
    """建立完整 SourceRecord、旧 Memory、A-03 图和 M-10 runtime。"""
    backend = backend_type()
    ctx = make_train_context(backend, companion=True)
    install_memory_batch_runtimes(ctx, _batch_config())
    old_source = _source(1)
    new_source = _source(2)
    source_intake = ctx.memory_read_intake.source_intake
    source_intake.ensure(
        old_source,
        _TEXT,
        license_id=_LICENSE,
        batch_id=101,
    )
    source_intake.ensure(
        new_source,
        _TEXT,
        license_id=_LICENSE,
        batch_id=102,
    )
    old_result = ctx.memory_read_intake.ingest(
        old_source,
        _TEXT,
        license_id=_LICENSE,
        batch_id=101,
        parser=_OldParser(old_source),
    )
    use_ref = _append_use(ctx, old_source, old_result)
    old_hypotheses = tuple(_hypothesis(old_source, item)
                           for item in range(1, 6))
    new_hypotheses = tuple(_hypothesis(new_source, item)
                           for item in range(6, 10))
    graph = ParserRevisionGraph(ctx.graph_ontology, _revision_protocol())
    for hypothesis in (*old_hypotheses, *new_hypotheses):
        ctx.graph_ontology.materialize(hypothesis.object_identity())
    dimension = concept_identity((24860, 1))
    reason = minimal_instruction_identity((24861, 1))
    ctx.graph_ontology.materialize(dimension)
    ctx.graph_ontology.materialize(reason)
    replacements = (
        (new_hypotheses[0],),
        (new_hypotheses[1], new_hypotheses[2]),
        (new_hypotheses[3],),
        (new_hypotheses[3],),
        (),
    )
    request = ParserRevisionRequest(
        old_source,
        new_source,
        document_scope(old_source),
        document_scope(new_source),
        (24862, 1),
        (),
        tuple(
            ParserHypothesisRevision(
                hypothesis,
                replacements[index],
                EvidenceRecord(
                    24870 + index,
                    hypothesis,
                    EVIDENCE_REFUTE,
                    (24871, index),
                    new_source,
                    10 + index,
                ),
            )
            for index, hypothesis in enumerate(old_hypotheses)
        ),
        (dimension,),
        reason,
        20,
        (24872, 1),
    )
    graph.materialize(request)
    runtime = MemoryParserRevisionRuntime(
        ctx,
        graph,
        ctx.memory_read_intake,
    )
    return _World(
        backend,
        ctx,
        runtime,
        request,
        old_source,
        new_source,
        old_hypotheses,
        new_hypotheses,
        old_result,
        use_ref,
    )


def _query_protocol(source: SourceRef) -> MemoryQueryProtocol:
    """构造只查询 A-08 候选 kind 的开放 M-06 协议。"""
    roles = MemoryQueryRoles(*(
        minimal_instruction_identity(
            (24890, item),
            owner=source.owner,
            versions=source.versions,
        )
        for item in range(1, 9)
    ))
    return MemoryQueryProtocol(
        roles,
        (MemoryQueryDefinition(
            minimal_instruction_identity(
                (24891, 1),
                owner=source.owner,
                versions=source.versions,
            ),
            (24820, 1),
            (roles.occurrence,),
            16,
        ),),
    )


def _manifest_lookup_metrics(version_count: int) -> tuple[int, int, int]:
    """构造连续 manifest 链并返回定点 derivation 查询、后端 calls 和 rows。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        sources = tuple(_source(index)
                        for index in range(1, version_count + 1))
        source_intake = ctx.memory_read_intake.source_intake
        for index, source in enumerate(sources, start=1):
            source_intake.ensure(
                source,
                _TEXT,
                license_id=_LICENSE,
                batch_id=300 + index,
            )
        prior = None
        for index, source in enumerate(sources, start=1):
            ctx.memory_read_intake.ingest(
                source,
                _TEXT,
                license_id=_LICENSE,
                batch_id=300 + index,
                parser=_SingleParser(source),
                supersedes_source=prior,
            )
            prior = source

        event_log = ctx.memory_read_events
        original_query = event_log.query
        derivation_calls = 0

        def bounded_query(*, access, event_kind=None,
                          object_kind=None, object_ref=None):
            """拒绝 current lookup 对 derivation 事件做无对象全扫描。"""
            nonlocal derivation_calls
            if event_kind == MEMORY_EVENT_DERIVATION:
                assert object_ref is not None
                derivation_calls += 1
            return original_query(
                access=access,
                event_kind=event_kind,
                object_kind=object_kind,
                object_ref=object_ref,
            )

        event_log.query = bounded_query
        with collect_backend_telemetry() as collector:
            current = ctx.memory_read_intake.require_current_manifest(
                sources[-1])
        assert current.source == sources[-1]
        snapshot = collector.operation_snapshot()
        backend_calls = sum(item[0] for item in snapshot.values())
        returned_rows = sum(item[1] for item in snapshot.values())
        return derivation_calls, backend_calls, returned_rows
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_mapping_cardinality_preserves_use_and_exact_replay(backend_type):
    """双向唯一才 supersede，多对多 archive，旧 Use/输出保留且重放零增行。"""
    world = _world(backend_type)
    try:
        parser = _NewParser(world.new_source)
        result = world.runtime.apply(
            world.request,
            raw_text=_TEXT,
            license_id=_LICENSE,
            batch_id=102,
            parser=parser,
        )
        assert parser.calls == 1
        assert result.replayed is False
        assert result.preserved_use_refs == (world.use_ref,)
        assert len(result.old_hypothesis_refs) == 5
        assert len(result.new_hypothesis_refs) == 4
        old_states = tuple(
            world.ctx.memory_read_aggregates.read(ref, access=_ACCESS)
            for ref in world.old_result.hypothesis_refs
        )
        assert old_states[0].lifecycle_state == LIFECYCLE_SUPERSEDED
        assert all(item.lifecycle_state == LIFECYCLE_ARCHIVED
                   for item in old_states[1:])
        assert old_states[0].use_count == 1
        assert all(
            world.ctx.memory_read_aggregates.read(
                ref, access=_ACCESS).lifecycle_state == LIFECYCLE_ACTIVE
            for ref in result.new_hypothesis_refs
        )
        query_runtime = install_memory_query_runtime(
            world.ctx,
            _query_protocol(world.new_source),
            aggregates=world.ctx.memory_read_aggregates,
        )
        scope = _open_query(world.ctx, world.new_source, local_id=8)
        compilation = query_runtime.compile(
            _current(world.ctx, world.new_source, scope),
            access=_ACCESS,
        )
        resolver = MemoryOverlayResolver(
            world.ctx.memory_read_aggregates,
            world.ctx.core_identity_catalog,
            _NoCoreBaseline(),
            _LifecycleFilter(None),
            _CandidateScorer(),
            _StableDiversity(),
        )
        resolution = resolver.resolve(compilation)
        assert {
            item.memory_ref for item in resolution.sets[0].candidates
        } == set(result.new_hypothesis_refs)
        inactive_bundle = resolver.load_bundle(
            old_states[1], access=_ACCESS)
        with pytest.raises(ValueError, match="不得消费 inactive aggregate bundle"):
            resolver.candidate_from_bundle(
                compilation.requests[0], inactive_bundle)
        inactive_resolver = MemoryOverlayResolver(
            world.ctx.memory_read_aggregates,
            world.ctx.core_identity_catalog,
            _NoCoreBaseline(),
            _LifecycleFilter(LIFECYCLE_ARCHIVED),
            _CandidateScorer(),
            _StableDiversity(),
        )
        with pytest.raises(ValueError, match="不得请求 inactive lifecycle"):
            inactive_resolver.resolve(compilation)
        _close_query(world.ctx)
        event_count = len(world.ctx.memory_read_events.query(access=_ACCESS))
        replay = world.runtime.apply(
            world.request,
            raw_text=_TEXT,
            license_id=_LICENSE,
            batch_id=102,
            parser=_RejectParser(),
        )
        assert replay.replayed is True
        assert replay.preserved_use_refs == (world.use_ref,)
        assert len(world.ctx.memory_read_events.query(
            access=_ACCESS)) == event_count
    finally:
        world.backend.close()


def test_split_lineage_private_choice_fails_before_memory_write():
    """拆分若沿用旧 lineage 私选一个目标，必须在首个 Memory event 前拒绝。"""
    world = _world(DictBackend)
    try:
        before = world.ctx.memory_read_events.query(access=_ACCESS)
        with pytest.raises(
                MemoryParserRevisionError,
                match="不得私选单个 Memory replacement"):
            world.runtime.apply(
                world.request,
                raw_text=_TEXT,
                license_id=_LICENSE,
                batch_id=102,
                parser=_NewParser(
                    world.new_source,
                    invalid_split=True,
                ),
            )
        assert world.ctx.memory_read_events.query(access=_ACCESS) == before
        assert world.ctx.memory_read_intake.manifest(
            world.new_source) is None
        assert world.ctx.memory_read_intake.require_current_manifest(
            world.old_source).source == world.old_source
    finally:
        world.backend.close()


def test_m10_hidden_partial_recovers_without_reparsing():
    """M-10 中段故障只暴露旧视图，重试 roll-forward 且不再调用 parser。"""
    world = _world(DictBackend)
    try:
        before = world.ctx.memory_read_events.query(access=_ACCESS)
        parser = _NewParser(world.new_source)
        with pytest.raises(RuntimeError, match="A-08 M-10 fault"):
            world.runtime.apply(
                world.request,
                raw_text=_TEXT,
                license_id=_LICENSE,
                batch_id=102,
                parser=parser,
                batch_fault_injector=_FailOnce(),
            )
        assert parser.calls == 1
        assert world.ctx.memory_read_events.query(access=_ACCESS) == before
        assert world.ctx.memory_read_intake.require_current_manifest(
            world.old_source).source == world.old_source

        recovered = world.runtime.apply(
            world.request,
            raw_text=_TEXT,
            license_id=_LICENSE,
            batch_id=102,
            parser=_RejectParser(),
        )
        assert recovered.replayed is True
        assert recovered.preserved_use_refs == (world.use_ref,)
        assert world.ctx.memory_read_intake.require_current_manifest(
            world.new_source).source == world.new_source
    finally:
        world.backend.close()


def test_evaluation_clone_updates_only_cloned_memory():
    """V-06 clone 可完成真实 A-08，宿主 manifest、事件和 aggregate 不变。"""
    world = _world(DictBackend)
    cloned = None
    try:
        host_events = world.ctx.memory_read_events.query(access=_ACCESS)
        host_old = tuple(
            world.ctx.memory_read_aggregates.read(ref, access=_ACCESS)
            for ref in world.old_result.hypothesis_refs
        )
        cloned = world.runtime.clone_for_evaluation()
        result = cloned.apply(
            world.request,
            raw_text=_TEXT,
            license_id=_LICENSE,
            batch_id=102,
            parser=_NewParser(world.new_source),
        )
        assert result.new_hypothesis_refs
        assert cloned.intake.require_current_manifest(
            world.new_source).source == world.new_source
        assert world.ctx.memory_read_intake.manifest(
            world.new_source) is None
        assert world.ctx.memory_read_events.query(access=_ACCESS) == host_events
        assert tuple(
            world.ctx.memory_read_aggregates.read(ref, access=_ACCESS)
            for ref in world.old_result.hypothesis_refs
        ) == host_old
    finally:
        if cloned is not None:
            cloned.ctx.backend.close()
        world.backend.close()


def test_v02_manifest_current_lookup_is_targeted_and_linear():
    """2/4/8 版本 current lookup 只做定点查询，calls/rows 不出现二次增长。"""
    metrics = tuple(
        _manifest_lookup_metrics(count) for count in (2, 4, 8))
    assert tuple(item[0] for item in metrics) == (2, 4, 8)
    calls = tuple(item[1] for item in metrics)
    rows = tuple(item[2] for item in metrics)
    assert calls[2] - calls[1] <= (calls[1] - calls[0]) * 2
    assert rows[2] - rows[1] <= (rows[1] - rows[0]) * 2 + 2
