"""K-04 Memory 候选投影、query 热集和全热等价对抗测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisLedger,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MEMORY_EVIDENCE_CONFLICTED,
    MEMORY_EVIDENCE_PROVISIONAL,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_USE,
    MEMORY_EVENT_USE_OUTCOME,
    MemoryLinkedRef,
    RETENTION_CONSOLIDATED,
    RETENTION_EPISODIC,
)
from pure_integer_ai.cognition.shared.memory_decay import (
    LinearTimelineDecayPolicy,
    RetentionDecayCurve,
)
from pure_integer_ai.cognition.shared.memory_hot_set import (
    StableTopKSourcePolicy,
    decode_memory_candidate,
)
from pure_integer_ai.cognition.shared.memory_batch import (
    MemoryBatchRuntimeConfig,
    install_memory_batch_runtimes,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_maintenance import (
    MemoryMaintenanceService,
    MemoryPlacementHint,
    MemoryRetentionDecision,
)
from pure_integer_ai.cognition.shared.memory_hypothesis import (
    MemoryHypothesisEventSink,
)
from pure_integer_ai.cognition.shared.memory_resolver_engine import (
    MemoryOverlayResolver,
)
from pure_integer_ai.cognition.shared.memory_resolver import (
    MemoryAggregateFilter,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.experiments.memory_hot_set_runtime import (
    MemoryCandidateProjectionManifest,
    MemoryCandidateProjectionPublisher,
    MemoryProjectionError,
    MemoryProjectionPlacementPublication,
    MemoryProjectionPublication,
    install_memory_hot_set_runtime,
)
from pure_integer_ai.experiments.memory_query_runtime import (
    install_memory_query_runtime,
)
from pure_integer_ai.experiments.memory_maintenance_runtime import (
    install_memory_maintenance_runtime,
)
from pure_integer_ai.experiments.memory_resolver_runtime import (
    install_memory_resolver_runtime,
)
from pure_integer_ai.experiments.memory_use_runtime import (
    install_memory_use_runtime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_budget import (
    EDGE_BUDGET_MAXIMUM,
    EDGE_BUDGET_MINIMUM,
    EdgeBudgetLimit,
    EdgeBudgetProfile,
    EdgeMetricObservation,
    compare_pareto,
)
from pure_integer_ai.storage.memory_batch import (
    MEMORY_BATCH_CORE_DEPENDENCY_KEY,
)
from pure_integer_ai.storage.memory_aggregate import (
    MEMORY_AGGREGATE_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.memory_query_projection import (
    MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.placement import (
    TemperatureProfile,
    TemperatureTier,
)
from pure_integer_ai.storage.query_hot_set import QueryHotSetPolicy
from pure_integer_ai.storage.query_hot_set import (
    QueryHotSetBudgetExceeded,
    QueryPrefetchContext,
    QuerySegmentHotSet,
)
from pure_integer_ai.storage.sealed_segment import (
    SegmentBudget,
    SegmentRecord,
)
from pure_integer_ai.storage.segment_dependency import SegmentDependency
from pure_integer_ai.storage.segment_repository import OBJECT_KIND_SEGMENT
from pure_integer_ai.storage.tiered_segment_store import segment_copy_identity
from pure_integer_ai.storage.tiered_segment_store import (
    FAULT_RELEASE_AFTER_MANIFEST_PUBLISH,
    FAULT_RELEASE_AFTER_PREPARE,
    TieredSegmentStore,
)

from test_m03_memory_event import (
    _append_observation,
    _core_refs,
    _source as _memory_source,
)
from test_m06_memory_query import (
    _close_query,
    _current,
    _open_query,
    _source as _query_source,
)
from test_m07_memory_resolver import (
    _BaselineProvider,
    _CurrentContextScorer,
    _DistinctSourcePolicy,
    _IndexFilterProvider,
    _hypothesis,
    _protocol,
    _seed_memory,
)
from test_a10_attractor_state import (
    _goals,
    _install_a10,
    _instruction,
)
from test_m08_memory_use import _consume_head, _query_time


_PROFILE = TemperatureProfile(
    (920, 1),
    (
        TemperatureTier((920, 1), 0),
        TemperatureTier((920, 2), 1),
    ),
)
_HOT = (920, 1)
_COLD = (920, 2)
_ACCESS = MemoryAccessContext(1, 2, 3)
_PROJECTION_KEY = (921, 1)
_KINDS = ((7201,), (7202,))
_DEPENDENCIES = (
    SegmentDependency(
        MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
        (1, 1),
        (2, 1),
    ),
    SegmentDependency(
        MEMORY_AGGREGATE_STORAGE_DESCRIPTOR_KEY,
        (1, 2),
        (2, 2),
    ),
)


def _batch_config() -> MemoryBatchRuntimeConfig:
    """构造 K-04 测试依赖的正式 M-10/K-02 装配配置。"""
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


def _install_resolver(
        ctx,
        source,
        core_ref,
        *,
        filter_provider=None,
        diversity_policy=None,
        ):
    """安装使用同一 Stable Top-K 策略的全热基线和 K-04 resolver。"""
    query_runtime = install_memory_query_runtime(
        ctx,
        _protocol(source),
        aggregates=ctx.memory_interact_aggregates,
    )
    resolver = MemoryOverlayResolver(
        ctx.memory_interact_aggregates,
        ctx.core_identity_catalog,
        _BaselineProvider(core_ref),
        _IndexFilterProvider() if filter_provider is None else filter_provider,
        _CurrentContextScorer(),
        (StableTopKSourcePolicy()
         if diversity_policy is None else diversity_policy),
    )
    return query_runtime, install_memory_resolver_runtime(ctx, resolver)


def _seed_many(ctx, count: int) -> None:
    """写入指定数量的同 kind 独立候选，形成大于 query cache 的冷库。"""
    ledger = HypothesisLedger(
        MemoryHypothesisEventSink(ctx.memory_interact_events))
    for index in range(count):
        source = _memory_source(
            source_id=100 + index,
            document_id=100 + index,
        )
        hypothesis = _hypothesis(
            source,
            kind=(7201,),
            candidate=index + 1,
            competition=9301,
        )
        ledger.register(hypothesis)
        ledger.append_evidence(EvidenceRecord(
            10_000 + index,
            hypothesis,
            EVIDENCE_SUPPORT,
            (9400 + index,),
            source,
            index + 1,
        ))
    ctx.memory_interact_aggregates.rebuild_dirty(access=_ACCESS)


class _FixedFilterProvider:
    """返回调用方注入的 OR-of-AND 分支，供全热与冷页共用。"""

    def __init__(self, filters: tuple[MemoryAggregateFilter, ...]) -> None:
        """冻结非空且唯一的完整过滤分支。"""
        if not filters or len({item.stable_key() for item in filters}) != len(filters):
            raise ValueError("固定过滤分支必须非空且唯一")
        self._filters = filters

    def filters(self, request):
        """忽略 request 路由差异并返回冻结过滤分支。"""
        del request
        return self._filters

    def clone_for_context(self, ctx):
        """过滤分支不可变，clone 只创建新的 provider 容器。"""
        del ctx
        return _FixedFilterProvider(self._filters)

    def state_key(self):
        """返回全部过滤分支的长度分帧稳定状态。"""
        result = [1, len(self._filters)]
        for item in self._filters:
            key = item.stable_key()
            result.extend((len(key), *key))
        return tuple(result)


class _ReleaseFault:
    """在指定 release 边界首次中断，用于验证重启 roll-forward。"""

    def __init__(self, point: int) -> None:
        """绑定故障点并创建未触发状态。"""
        self.point = point
        self.triggered = False

    def hit(self, point: int, context: dict[str, object]) -> None:
        """忽略其他边界，在目标点首次抛出可识别异常。"""
        del context
        if point == self.point and not self.triggered:
            self.triggered = True
            raise RuntimeError(f"release fault {point}")


class _StaticPrefetchPolicy:
    """以注入布尔配置决定每个续页是否预取，供测试显式选择。"""

    def __init__(self, enabled: bool) -> None:
        """冻结预取开关，拒绝非严格布尔配置。"""
        if type(enabled) is not bool:
            raise TypeError("test prefetch enabled 必须是严格 bool")
        self.enabled = enabled

    def should_prefetch(self, context: QueryPrefetchContext) -> bool:
        """核验上下文类型后返回冻结决策。"""
        if not isinstance(context, QueryPrefetchContext):
            raise TypeError("test prefetch context 类型错误")
        return self.enabled

    def state_key(self) -> tuple[int, ...]:
        """返回区分启用和禁用配置的稳定整数键。"""
        return 1, 1 if self.enabled else 2


def _prefetch(enabled: bool) -> _StaticPrefetchPolicy:
    """构造一个不共享 query 可变状态的测试预取策略。"""
    return _StaticPrefetchPolicy(enabled)


class _PeriodicPrefetchPolicy:
    """按已消费页序注入周期决策，并记录收到的只读上下文。"""

    def __init__(self, interval: int) -> None:
        """冻结正周期间隔并创建观测列表。"""
        if type(interval) is not int or interval <= 0:
            raise ValueError("test prefetch interval 必须是正严格整数")
        self.interval = interval
        self.contexts: list[QueryPrefetchContext] = []

    def should_prefetch(self, context: QueryPrefetchContext) -> bool:
        """记录上下文，并只由已消费页序和冻结间隔形成决策。"""
        if not isinstance(context, QueryPrefetchContext):
            raise TypeError("test periodic prefetch context 类型错误")
        self.contexts.append(context)
        return context.consumed_pages % self.interval == 0

    def state_key(self) -> tuple[int, ...]:
        """返回不含观测列表的冻结策略身份。"""
        return 2, self.interval


class _NoRetentionPolicy:
    """测试中不执行巩固，只让 M-09 产生独立 placement hint。"""

    def state_key(self) -> tuple[int, ...]:
        """返回不巩固策略的稳定身份。"""
        return (3, 1)

    def assess(self, snapshot) -> MemoryRetentionDecision:
        """忽略快照内容并返回显式不巩固决定。"""
        del snapshot
        return MemoryRetentionDecision(
            False, (), (3, 2), self.state_key())


class _ProjectionPlacementHintPolicy:
    """把 M-09 对象级建议指向注入的 K-04 投影温层。"""

    def __init__(self, target_tier_key: tuple[int, ...]) -> None:
        """冻结目标温层，不从 retention 或事实状态猜测位置。"""
        self.target_tier_key = target_tier_key

    def state_key(self) -> tuple[int, ...]:
        """返回目标温层参与的稳定策略身份。"""
        return 4, len(self.target_tier_key), *self.target_tier_key

    def hints(self, snapshot) -> tuple[MemoryPlacementHint, ...]:
        """为当前完整 Hypothesis 引用产生一个投影 descriptor hint。"""
        return (MemoryPlacementHint(
            snapshot.hypothesis_ref.stable_key(),
            MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
            self.target_tier_key,
            _PROFILE.profile_key,
            (4, 1),
            self.state_key(),
            snapshot.as_of.seq,
        ),)


def _publish_projection(ctx, resolver):
    """用小 segment 预算发布多页候选投影，并返回冻结 manifest。"""
    publisher = MemoryCandidateProjectionPublisher(
        resolver,
        ctx.tiered_segment_store,
    )
    return publisher.publish(
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


def _projection_bundles(ctx, manifest):
    """经 K-02 稳定 reader 恢复 manifest 指向的全部 typed 候选。"""
    reader = ctx.tiered_segment_store.open_reader(
        (924, 1),
        MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
    )
    bundles = []
    try:
        for segment in manifest.segments:
            page = reader.page(
                budget=SegmentBudget(2, 2_000_000),
                lower_key=segment.lower_key,
                upper_key=segment.upper_key,
            )
            assert not page.has_more
            bundles.extend(decode_memory_candidate(
                manifest.projection_key, record) for record in page.records)
    finally:
        reader.close()
    return tuple(bundles)


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_projection_roundtrip_and_hot_resolution_match_full_baseline(
        backend_type):
    """多页冷投影恢复完整候选，hot-set 输出与同策略全热执行逐字段相同。"""
    backend = backend_type()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])

        scope = _open_query(ctx, query_source)
        full_compilation = query_runtime.compile(
            _current(ctx, query_source, scope),
            access=_ACCESS,
        )
        full = resolver_runtime.resolve(full_compilation)
        _close_query(ctx)

        source_state = resolver_runtime.resolver.aggregates.event_log.projection_state_key()
        manifest = _publish_projection(ctx, resolver_runtime.resolver)
        assert resolver_runtime.resolver.aggregates.event_log.projection_state_key() == source_state
        assert manifest.record_count == 3
        assert len(manifest.segments) == 3
        assert MemoryCandidateProjectionManifest.from_stable_key(
            manifest.stable_key()) == manifest
        manifest.validate_store(ctx.tiered_segment_store)

        expected = []
        for kind in _KINDS:
            for aggregate in ctx.memory_interact_aggregates.query(
                    access=_ACCESS, hypothesis_kind=kind):
                expected.append(resolver_runtime.resolver.load_bundle(
                    aggregate, access=_ACCESS).stable_key())
        restored = tuple(
            item.stable_key() for item in _projection_bundles(ctx, manifest))
        assert tuple(sorted(restored)) == tuple(sorted(expected))

        hot_runtime = install_memory_hot_set_runtime(
            ctx,
            manifest,
            QueryHotSetPolicy(
                SegmentBudget(4, 4_000_000),
                SegmentBudget(1, 1_000_000),
                _prefetch(True),
                8,
            ),
        )
        hot_scope = _open_query(ctx, query_source)
        hot_compilation = query_runtime.compile(
            _current(ctx, query_source, hot_scope),
            access=_ACCESS,
        )
        hot = resolver_runtime.resolve(hot_compilation)
        assert hot.stable_key() == full.stable_key()
        _close_query(ctx)

        metrics = hot_runtime.metrics()
        assert metrics is not None
        assert metrics.page_in_records == manifest.record_count
        assert metrics.prefetched_pages >= 1
        assert metrics.peak_hot_objects <= 4
        assert metrics.released_pins == 3
        assert ctx.tiered_segment_store.reader_epochs.snapshot() == ()
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_large_projection_keeps_peak_hot_set_bound_to_query_budget():
    """候选库增长到热预算之外时，精确输出不变且峰值常驻不随总量增长。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_many(ctx, 40)
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])

        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope), access=_ACCESS)
        full = resolver_runtime.resolve(compilation)
        _close_query(ctx)

        manifest = _publish_projection(ctx, resolver_runtime.resolver)
        assert manifest.record_count == 40
        hot_runtime = install_memory_hot_set_runtime(
            ctx,
            manifest,
            QueryHotSetPolicy(
                SegmentBudget(4, 4_000_000),
                SegmentBudget(1, 1_000_000),
                _prefetch(False),
                4,
            ),
        )
        hot_scope = _open_query(ctx, query_source)
        hot_compilation = query_runtime.compile(
            _current(ctx, query_source, hot_scope), access=_ACCESS)
        assert resolver_runtime.resolve(
            hot_compilation).stable_key() == full.stable_key()
        _close_query(ctx)

        metrics = hot_runtime.metrics()
        assert metrics is not None
        assert metrics.page_in_records == 40
        assert metrics.peak_hot_objects <= 4
        assert metrics.peak_hot_objects < manifest.record_count
        assert metrics.released_pins == 3

        external = (
            EdgeMetricObservation((10, 1), 12),
            EdgeMetricObservation((10, 2), 20),
            EdgeMetricObservation((10, 3), 30),
            EdgeMetricObservation((10, 4), 40),
            EdgeMetricObservation((10, 5), 50),
            EdgeMetricObservation((10, 6), 60),
            EdgeMetricObservation((10, 7), 70),
            EdgeMetricObservation((10, 8), 80),
            EdgeMetricObservation((10, 9), 90),
            EdgeMetricObservation((10, 10), 100),
        )
        limits = [EdgeBudgetLimit(
            item.metric_key,
            EDGE_BUDGET_MAXIMUM,
            item.value + 1,
        ) for item in metrics.observations()]
        for index, item in enumerate(external):
            direction = (
                EDGE_BUDGET_MINIMUM if index == 3
                else EDGE_BUDGET_MAXIMUM)
            hard_limit = item.value if direction == EDGE_BUDGET_MINIMUM else 200
            limits.append(EdgeBudgetLimit(
                item.metric_key, direction, hard_limit))
        profile = EdgeBudgetProfile(
            (11, 1),
            (11, 2),
            (11, 3),
            tuple(limits),
        )
        report = hot_runtime.evaluate_edge_budget(
            profile, external_observations=external)
        assert report.passed
        assert not hasattr(report, "score")

        failed_external = (
            EdgeMetricObservation(external[0].metric_key, 201),
            *external[1:3],
            EdgeMetricObservation(external[3].metric_key, 400),
            *external[4:],
        )
        failed = hot_runtime.evaluate_edge_budget(
            profile, external_observations=failed_external)
        assert not failed.passed
        assert tuple(
            item.limit.metric_key for item in failed.results
            if not item.passed) == (external[0].metric_key,)
        complete = (*metrics.observations(), *external)
        assert not profile.evaluate(complete[:-1]).passed
        assert not profile.evaluate((
            *complete,
            EdgeMetricObservation((10, 99), 1),
        )).passed

        changed = list(complete)
        latency_index = next(
            index for index, item in enumerate(changed)
            if item.metric_key == external[0].metric_key)
        storage_index = next(
            index for index, item in enumerate(changed)
            if item.metric_key == external[7].metric_key)
        changed[latency_index] = EdgeMetricObservation(
            external[0].metric_key, 11)
        changed[storage_index] = EdgeMetricObservation(
            external[7].metric_key, 81)
        pareto = compare_pareto(profile, complete, tuple(changed))
        assert pareto.has_regression
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_open_prefetch_policy_selects_continuations_from_integer_context():
    """开放预取策略可按 query 物理上下文动态决策，不需要固定 0/1 深度。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_many(ctx, 6)
        query_source = _query_source(document_id=1)
        _, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])
        manifest = _publish_projection(ctx, resolver_runtime.resolver)
        strategy = _PeriodicPrefetchPolicy(2)
        hot_set = QuerySegmentHotSet(
            ctx.tiered_segment_store,
            reader_key=(925, 1),
            descriptor_key=MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
            policy=QueryHotSetPolicy(
                SegmentBudget(2, 2_000_000),
                SegmentBudget(1, 1_000_000),
                strategy,
                8,
            ),
        )
        try:
            records = tuple(item.record for item in hot_set.iter_range())
        finally:
            hot_set.close()

        assert len(records) == manifest.record_count == 6
        assert tuple(item.consumed_pages for item in strategy.contexts) == (
            1, 2, 3, 4, 5)
        assert hot_set.metrics().prefetched_pages == 2
    finally:
        backend.close()


def test_hot_projection_drives_a10_use_outcome_and_stale_rebuild():
    """K-04 候选须被 A-10 真消费并形成 Use/outcome，随后使旧投影失效。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        observation = _append_observation(
            ctx, query_source, _core_refs(ctx))
        ctx.memory_interact_aggregates.rebuild_dirty(access=_ACCESS)
        query_runtime, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])
        publisher = MemoryCandidateProjectionPublisher(
            resolver_runtime.resolver, ctx.tiered_segment_store)
        first_manifest = _publish_projection(
            ctx, resolver_runtime.resolver)
        hot_runtime = install_memory_hot_set_runtime(
            ctx,
            first_manifest,
            QueryHotSetPolicy(
                SegmentBudget(4, 4_000_000),
                SegmentBudget(1, 1_000_000),
                _prefetch(True),
                8,
            ),
        )
        attractor, _ = _install_a10(
            ctx, query_source, prefer_matching_document=False)
        memory_use = install_memory_use_runtime(ctx)

        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope), access=_ACCESS)
        state = attractor.resolve_and_activate(
            compilation, _goals(query_source, scope))
        assert hot_runtime._hot_set is not None
        assert ctx.tiered_segment_store.reader_epochs.snapshot()
        trace = _consume_head(attractor, query_source)
        selected = memory_use.record_selection_use(
            trace,
            input_observation_ref=observation.event.object_ref,
            influence_kind=MemoryLinkedRef.object(
                _instruction(query_source, 9951)),
            used_at=_query_time(state, 2),
        )
        outcome = memory_use.record_outcome(
            selected.use.event.object_ref,
            scope=state.scope,
            outcome_kind=MemoryLinkedRef.object(
                _instruction(query_source, 9952)),
            outcome_ref=None,
            observed_at=_query_time(state, 3),
        )
        assert selected.use.event.payload.memory_ref == (
            trace.activation.candidate.memory_ref)
        assert ctx.memory_interact_events.query(
            access=_ACCESS,
            event_kind=MEMORY_EVENT_USE,
            object_ref=selected.use.event.object_ref,
        ) == (selected.use,)
        assert ctx.memory_interact_events.query(
            access=_ACCESS,
            event_kind=MEMORY_EVENT_USE_OUTCOME,
            object_ref=selected.use.event.object_ref,
        ) == (outcome,)
        assert (ctx.memory_interact_events.projection_state_key()
                != first_manifest.source_state_key)
        _close_query(ctx)
        assert ctx.tiered_segment_store.reader_epochs.snapshot() == ()

        stale_scope = _open_query(ctx, query_source, local_id=2)
        stale_compilation = query_runtime.compile(
            _current(ctx, query_source, stale_scope, ordinal=1),
            access=_ACCESS,
        )
        with pytest.raises(MemoryProjectionError, match="投影已因"):
            attractor.resolve_and_activate(
                stale_compilation, _goals(query_source, stale_scope))
        _close_query(ctx)

        ctx.memory_interact_aggregates.rebuild_dirty(access=_ACCESS)
        second_manifest = publisher.publish(
            _PROJECTION_KEY,
            access=_ACCESS,
            hypothesis_kinds=_KINDS,
            publication=MemoryProjectionPublication(
                (922, 41),
                _COLD,
                (923, 41),
                _DEPENDENCIES,
                SegmentBudget(1, 1_000_000),
                1,
            ),
        )
        hot_runtime.replace_projection(second_manifest)
        assert second_manifest.projection_key != first_manifest.projection_key

        maintenance = install_memory_maintenance_runtime(
            ctx,
            MemoryMaintenanceService(
                ctx.memory_interact_aggregates,
                LinearTimelineDecayPolicy(
                    (9960, 1),
                    (
                        RetentionDecayCurve(
                            RETENTION_EPISODIC, 100, 1, 0),
                        RetentionDecayCurve(
                            RETENTION_CONSOLIDATED, 80, 1, 0),
                    ),
                    (9960, 2),
                ),
                _NoRetentionPolicy(),
                _ProjectionPlacementHintPolicy(_HOT),
                ctx.tiered_segment_store.registry,
                _PROFILE,
            ),
        )
        assessment = maintenance.assess(
            selected.use.event.payload.memory_ref,
            access=_ACCESS,
        )
        placement = hot_runtime.apply_maintenance_assessments(
            (assessment,),
            publication=MemoryProjectionPlacementPublication(
                (9961, 1), SegmentBudget(1, 1_000_000)),
        )
        assert placement is not None
        assert len(placement.directives) == 1
        directive = placement.directives[0]
        assert directive.target_tier_key == _HOT
        moved_entry = next(
            item for item in ctx.tiered_segment_store.current_manifest().entries
            if item.segment_key == directive.segment_key)
        assert moved_entry.tier_key == _HOT

        rebuilt_scope = _open_query(ctx, query_source, local_id=3)
        rebuilt_compilation = query_runtime.compile(
            _current(ctx, query_source, rebuilt_scope, ordinal=2),
            access=_ACCESS,
        )
        rebuilt_state = attractor.resolve_and_activate(
            rebuilt_compilation, _goals(query_source, rebuilt_scope))
        assert rebuilt_state.next_activation() is not None
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_object_placement_hints_reject_conflicting_targets_for_one_segment():
    """同一投影 segment 的对象级 hint 目标冲突时不得迁移或猜首条。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        _, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])
        publisher = MemoryCandidateProjectionPublisher(
            resolver_runtime.resolver, ctx.tiered_segment_store)
        manifest = publisher.publish(
            _PROJECTION_KEY,
            access=_ACCESS,
            hypothesis_kinds=_KINDS,
            publication=MemoryProjectionPublication(
                (922, 51),
                _COLD,
                (923, 51),
                _DEPENDENCIES,
                SegmentBudget(10, 4_000_000),
                4,
            ),
        )
        shared_segment = next(
            item for item in manifest.segments if item.record_count >= 2)
        reader = ctx.tiered_segment_store.open_reader(
            (9970, 1), MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY)
        try:
            page = reader.page(
                budget=SegmentBudget(10, 4_000_000),
                lower_key=shared_segment.lower_key,
                upper_key=shared_segment.upper_key,
            )
            references = tuple(
                decode_memory_candidate(
                    manifest.projection_key, record).hypothesis_ref
                for record in page.records[:2]
            )
        finally:
            reader.close()
        before = ctx.tiered_segment_store.current_manifest().stable_key()
        hints = tuple(MemoryPlacementHint(
            reference.stable_key(),
            MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
            target,
            _PROFILE.profile_key,
            (9971, index),
            (9972, 1),
            manifest.source_fence,
        ) for index, (reference, target) in enumerate(zip(
            references, (_HOT, _COLD)), start=1))

        with pytest.raises(MemoryProjectionError, match="冲突目标 tier"):
            publisher.apply_placement_hints(
                manifest,
                hints,
                publication=MemoryProjectionPlacementPublication(
                    (9973, 1), SegmentBudget(1, 1_000_000)),
            )

        assert ctx.tiered_segment_store.current_manifest().stable_key() == before
        assert all(
            item.tier_key == _COLD
            for item in ctx.tiered_segment_store.current_manifest().entries
            if item.segment_key in {
                segment.segment_key for segment in manifest.segments})
    finally:
        backend.close()


def test_placement_retry_after_published_migration_does_not_advance_epoch(
        monkeypatch):
    """迁移已发布后调用边界失败，重试同一 hint plan 不得重复迁移。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        _, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])
        publisher = MemoryCandidateProjectionPublisher(
            resolver_runtime.resolver, ctx.tiered_segment_store)
        manifest = _publish_projection(ctx, resolver_runtime.resolver)
        reference = _projection_bundles(ctx, manifest)[0].hypothesis_ref
        hint = MemoryPlacementHint(
            reference.stable_key(),
            MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
            _HOT,
            _PROFILE.profile_key,
            (9981, 1),
            (9982, 1),
            manifest.source_fence,
        )
        publication = MemoryProjectionPlacementPublication(
            (9983, 1), SegmentBudget(1, 1_000_000))
        store = ctx.tiered_segment_store
        original_migrate = store.migrate
        calls = 0

        def migrate_then_fail(*args, **kwargs):
            """首次真实迁移完成后抛错，模拟 caller 未收到成功结果。"""
            nonlocal calls
            calls += 1
            result = original_migrate(*args, **kwargs)
            if calls == 1:
                raise RuntimeError("migration returned failure")
            return result

        monkeypatch.setattr(store, "migrate", migrate_then_fail)
        with pytest.raises(RuntimeError, match="migration returned failure"):
            publisher.apply_placement_hints(
                manifest, (hint,), publication=publication)
        epoch_after_failure = store.current_manifest().publish_epoch

        receipt = publisher.apply_placement_hints(
            manifest, (hint,), publication=publication)

        assert calls == 1
        assert receipt.publish_epoch == epoch_after_failure
        assert store.current_manifest().publish_epoch == epoch_after_failure
        directive = receipt.directives[0]
        entry = next(
            item for item in store.current_manifest().entries
            if item.segment_key == directive.segment_key)
        assert entry.tier_key == _HOT
    finally:
        backend.close()


def test_kind_hash_collision_keeps_full_kind_isolation(monkeypatch):
    """强制多个开放 kind 共用派生 hash 时，全热与冷页仍按完整 kind 隔离。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        monkeypatch.setattr(
            ctx.memory_interact_aggregates,
            "hypothesis_kind_hash",
            lambda key: 9501,
        )
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])

        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope), access=_ACCESS)
        full = resolver_runtime.resolve(compilation)
        _close_query(ctx)

        manifest = _publish_projection(ctx, resolver_runtime.resolver)
        install_memory_hot_set_runtime(
            ctx,
            manifest,
            QueryHotSetPolicy(
                SegmentBudget(4, 4_000_000),
                SegmentBudget(1, 1_000_000),
                _prefetch(False),
                4,
            ),
        )
        hot_scope = _open_query(ctx, query_source)
        hot_compilation = query_runtime.compile(
            _current(ctx, query_source, hot_scope), access=_ACCESS)
        hot = resolver_runtime.resolve(hot_compilation)
        assert hot.stable_key() == full.stable_key()
        for resolved_set in hot.sets:
            assert all(
                candidate.hypothesis is None
                or candidate.hypothesis.hypothesis_kind
                == resolved_set.request.hypothesis_kind
                for candidate in resolved_set.candidates
            )
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_context_source_and_status_filters_match_full_resolver():
    """冷页 OR-of-AND 必须与全热 context/source/三状态过滤逐候选等价。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        first, _, _ = _seed_memory(ctx)
        source_b = _memory_source(source_id=20, document_id=12)
        filters = _FixedFilterProvider((
            MemoryAggregateFilter(
                context=first.scope.stable_key(),
                evidence_state=MEMORY_EVIDENCE_CONFLICTED,
            ),
            MemoryAggregateFilter(
                source=source_b,
                evidence_state=MEMORY_EVIDENCE_PROVISIONAL,
                lifecycle_state=LIFECYCLE_ACTIVE,
                retention_state=RETENTION_EPISODIC,
            ),
        ))
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx,
            query_source,
            _core_refs(ctx)[1],
            filter_provider=filters,
        )

        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope), access=_ACCESS)
        full = resolver_runtime.resolve(compilation)
        _close_query(ctx)

        manifest = _publish_projection(ctx, resolver_runtime.resolver)
        install_memory_hot_set_runtime(
            ctx,
            manifest,
            QueryHotSetPolicy(
                SegmentBudget(4, 4_000_000),
                SegmentBudget(1, 1_000_000),
                _prefetch(False),
                4,
            ),
        )
        hot_scope = _open_query(ctx, query_source)
        hot_compilation = query_runtime.compile(
            _current(ctx, query_source, hot_scope), access=_ACCESS)
        assert resolver_runtime.resolve(
            hot_compilation).stable_key() == full.stable_key()
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_stale_projection_fails_before_opening_reader():
    """投影发布后新增 Memory 事件必须让新 query fail closed，且不泄漏 reader。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])
        manifest = _publish_projection(ctx, resolver_runtime.resolver)
        install_memory_hot_set_runtime(
            ctx,
            manifest,
            QueryHotSetPolicy(
                SegmentBudget(4, 4_000_000),
                SegmentBudget(1, 1_000_000),
                _prefetch(False),
                4,
            ),
        )
        _seed_many(ctx, 1)

        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope), access=_ACCESS)
        with pytest.raises(MemoryProjectionError, match="投影已因"):
            resolver_runtime.resolve(compilation)
        assert ctx.tiered_segment_store.reader_epochs.snapshot() == ()
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_unbounded_selector_is_rejected_for_limited_profile():
    """只有全集合 select 而无有界 accumulator 的策略不得安装 K-04。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        _, resolver_runtime = _install_resolver(
            ctx,
            query_source,
            _core_refs(ctx)[1],
            diversity_policy=_DistinctSourcePolicy(),
        )
        manifest = _publish_projection(ctx, resolver_runtime.resolver)
        with pytest.raises(TypeError, match="有界流式选择协议"):
            install_memory_hot_set_runtime(
                ctx,
                manifest,
                QueryHotSetPolicy(
                    SegmentBudget(4, 4_000_000),
                    SegmentBudget(1, 1_000_000),
                    _prefetch(False),
                    4,
                ),
            )
    finally:
        backend.close()


def test_insufficient_cache_budget_fails_without_truncating_candidates():
    """pinned 候选占满 cache 时必须显式失败，不能返回静默截断结果。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])
        manifest = _publish_projection(ctx, resolver_runtime.resolver)
        install_memory_hot_set_runtime(
            ctx,
            manifest,
            QueryHotSetPolicy(
                SegmentBudget(1, 1_000_000),
                SegmentBudget(1, 1_000_000),
                _prefetch(False),
                4,
            ),
        )
        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope), access=_ACCESS)
        with pytest.raises(QueryHotSetBudgetExceeded):
            resolver_runtime.resolve(compilation)
        _close_query(ctx)
        assert ctx.tiered_segment_store.reader_epochs.snapshot() == ()
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_dirty_flush_failure_is_retryable_and_success_closes_reader():
    """dirty flush 失败保留记录和 lease，重试成功后才能清空并关闭。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        _, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])
        _publish_projection(ctx, resolver_runtime.resolver)
        flushed = []

        def fail_flush(records):
            """模拟介质刷写失败，并验证调用方收到完整 dirty 批。"""
            flushed.append(records)
            raise RuntimeError("flush failed")

        hot_set = QuerySegmentHotSet(
            ctx.tiered_segment_store,
            reader_key=(9601,),
            descriptor_key=MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY,
            policy=QueryHotSetPolicy(
                SegmentBudget(2, 1_000_000),
                SegmentBudget(1, 1_000_000),
                _prefetch(False),
                2,
            ),
            flush=fail_flush,
        )
        hot_set.put_dirty(SegmentRecord((9602,), (9603,)))
        with pytest.raises(RuntimeError, match="flush failed"):
            hot_set.close()
        assert len(flushed) == 1
        assert ctx.tiered_segment_store.reader_epochs.snapshot()
        assert any(item.dirty for item in hot_set.cache.snapshot())

        persisted = []

        def succeed_flush(records):
            """记录重试批次，模拟介质已完整持久化。"""
            persisted.extend(records)

        hot_set.flush_callback = succeed_flush
        hot_set.close()
        assert len(persisted) == 1
        assert hot_set.metrics().dirty_flushes == 1
        assert ctx.tiered_segment_store.reader_epochs.snapshot() == ()
    finally:
        backend.close()


def test_old_query_reader_survives_location_migration_then_reclaims_source():
    """活动 query 固定旧 epoch，迁移不改输出，关闭后才回收旧物理副本。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])
        manifest = _publish_projection(ctx, resolver_runtime.resolver)
        hot_runtime = install_memory_hot_set_runtime(
            ctx,
            manifest,
            QueryHotSetPolicy(
                SegmentBudget(4, 4_000_000),
                SegmentBudget(1, 1_000_000),
                _prefetch(False),
                4,
            ),
        )
        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope), access=_ACCESS)
        before = resolver_runtime.resolve(compilation)
        segment = manifest.segments[0]
        source_identity = segment_copy_identity(_COLD, segment.segment_key)
        ctx.tiered_segment_store.migrate(
            segment.segment_key,
            target_tier_key=_HOT,
            manifest_key=(9701,),
            migration_key=(9702,),
        )
        ctx.tiered_segment_store.recover_pending_operations()
        assert hot_runtime._hot_set is not None and hot_runtime._hot_set.stale
        assert resolver_runtime.resolve(compilation).stable_key() == before.stable_key()
        assert ctx.tiered_segment_store.repository.get(
            OBJECT_KIND_SEGMENT, source_identity)
        _close_query(ctx)
        with pytest.raises(KeyError):
            ctx.tiered_segment_store.repository.get(
                OBJECT_KIND_SEGMENT, source_identity)

        next_scope = _open_query(ctx, query_source)
        next_compilation = query_runtime.compile(
            _current(ctx, query_source, next_scope), access=_ACCESS)
        assert resolver_runtime.resolve(
            next_compilation).stable_key() == before.stable_key()
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_v06_clone_uses_independent_store_reader_and_cache():
    """评测 clone 复用不可变投影身份，但 backend、reader 和 query cache 均独立。"""
    backend = DictBackend()
    cloned_backend = None
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])
        manifest = _publish_projection(ctx, resolver_runtime.resolver)
        host_runtime = install_memory_hot_set_runtime(
            ctx,
            manifest,
            QueryHotSetPolicy(
                SegmentBudget(4, 4_000_000),
                SegmentBudget(1, 1_000_000),
                _prefetch(False),
                4,
            ),
        )
        cloned_backend = clone_backend(backend)
        cloned = clone_train_context(ctx, cloned_backend, label="k04-clone")
        assert cloned.memory_hot_set_runtime is not host_runtime
        assert cloned.tiered_segment_store is not ctx.tiered_segment_store

        scope = _open_query(cloned, query_source)
        compilation = cloned.memory_query_runtime.compile(
            _current(cloned, query_source, scope), access=_ACCESS)
        cloned.memory_resolver_runtime.resolve(compilation)
        assert host_runtime._hot_set is None
        assert cloned.memory_hot_set_runtime._hot_set is not None
        assert cloned.memory_hot_set_runtime._hot_set.cache is not (
            getattr(host_runtime._hot_set, "cache", None))
        _close_query(cloned)
        assert cloned.tiered_segment_store.reader_epochs.snapshot() == ()
        assert ctx.tiered_segment_store.reader_epochs.snapshot() == ()
    finally:
        if cloned_backend is not None:
            cloned_backend.close()
        backend.close()


def test_new_projection_generation_switches_only_at_query_boundary():
    """Memory 变化后同一命名空间可发布新代，活动 query 禁止切换。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])
        first_manifest = _publish_projection(ctx, resolver_runtime.resolver)
        hot_runtime = install_memory_hot_set_runtime(
            ctx,
            first_manifest,
            QueryHotSetPolicy(
                SegmentBudget(4, 4_000_000),
                SegmentBudget(1, 1_000_000),
                _prefetch(False),
                4,
            ),
        )

        _seed_many(ctx, 1)
        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope), access=_ACCESS)
        full = resolver_runtime.resolver.resolve(compilation)
        _close_query(ctx)

        second_manifest = _publish_projection(ctx, resolver_runtime.resolver)
        assert second_manifest.projection_key != first_manifest.projection_key
        assert set(item.segment_key for item in first_manifest.segments).isdisjoint(
            item.segment_key for item in second_manifest.segments)

        active_scope = _open_query(ctx, query_source)
        with pytest.raises(RuntimeError, match="活动 query"):
            hot_runtime.replace_projection(second_manifest)
        _close_query(ctx)
        hot_runtime.replace_projection(second_manifest)

        next_scope = _open_query(ctx, query_source)
        next_compilation = query_runtime.compile(
            _current(ctx, query_source, next_scope), access=_ACCESS)
        assert resolver_runtime.resolve(
            next_compilation).stable_key() == full.stable_key()
        _close_query(ctx)
        current_keys = {
            item.segment_key
            for item in ctx.tiered_segment_store.current_manifest().entries
            if item.descriptor_key == MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY
        }
        assert set(item.segment_key for item in first_manifest.segments) <= current_keys
        assert set(item.segment_key for item in second_manifest.segments) <= current_keys

        old_reader = ctx.tiered_segment_store.open_reader(
            (9801,), MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY)
        publisher = MemoryCandidateProjectionPublisher(
            resolver_runtime.resolver, ctx.tiered_segment_store)
        publisher.release(
            first_manifest,
            release_key=(9802,),
            manifest_key=(9803,),
        )
        old_segment = first_manifest.segments[0]
        old_page = old_reader.page(
            budget=SegmentBudget(2, 2_000_000),
            lower_key=old_segment.lower_key,
            upper_key=old_segment.upper_key,
        )
        assert old_page.records
        new_reader = ctx.tiered_segment_store.open_reader(
            (9804,), MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY)
        try:
            new_page = new_reader.page(
                budget=SegmentBudget(2, 2_000_000),
                lower_key=old_segment.lower_key,
                upper_key=old_segment.upper_key,
            )
            assert new_page.records == ()
        finally:
            new_reader.close()
        for segment in first_manifest.segments:
            assert ctx.tiered_segment_store.repository.get(
                OBJECT_KIND_SEGMENT,
                segment_copy_identity(_COLD, segment.segment_key),
            )
        old_reader.close()
        for segment in first_manifest.segments:
            with pytest.raises(KeyError):
                ctx.tiered_segment_store.repository.get(
                    OBJECT_KIND_SEGMENT,
                    segment_copy_identity(_COLD, segment.segment_key),
                )
        remaining_keys = {
            item.segment_key
            for item in ctx.tiered_segment_store.current_manifest().entries
            if item.descriptor_key == MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY
        }
        assert remaining_keys == {
            item.segment_key for item in second_manifest.segments}
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_sqlite_restart_restores_projection_manifest_and_hot_output(tmp_path):
    """SQLite 重启后从纯整数 manifest 恢复 K-02 投影并保持 resolver 输出。"""
    path = str(tmp_path / "k04_projection.sqlite3")
    first_backend = SQLiteBackend(path)
    try:
        first = make_train_context(first_backend, companion=True)
        install_memory_batch_runtimes(first, _batch_config())
        _seed_memory(first)
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            first, query_source, _core_refs(first)[1])
        scope = _open_query(first, query_source)
        compilation = query_runtime.compile(
            _current(first, query_source, scope), access=_ACCESS)
        expected = resolver_runtime.resolve(compilation).stable_key()
        _close_query(first)
        manifest = _publish_projection(first, resolver_runtime.resolver)
        manifest_key = manifest.stable_key()
        first_backend.commit()
    finally:
        first_backend.close()

    second_backend = SQLiteBackend(path)
    try:
        second = make_train_context(second_backend, companion=True)
        install_memory_batch_runtimes(second, _batch_config())
        query_source = _query_source(document_id=1)
        query_runtime, resolver_runtime = _install_resolver(
            second, query_source, _core_refs(second)[1])
        restored = MemoryCandidateProjectionManifest.from_stable_key(
            manifest_key)
        restored.validate_store(second.tiered_segment_store)
        install_memory_hot_set_runtime(
            second,
            restored,
            QueryHotSetPolicy(
                SegmentBudget(4, 4_000_000),
                SegmentBudget(1, 1_000_000),
                _prefetch(False),
                4,
            ),
        )
        scope = _open_query(second, query_source)
        compilation = query_runtime.compile(
            _current(second, query_source, scope), access=_ACCESS)
        assert resolver_runtime.resolve(compilation).stable_key() == expected
    finally:
        if ("second" in locals()
                and second.work_memory.active_query_scope is not None):
            _close_query(second)
        second_backend.close()


def test_projection_source_drift_releases_every_attempted_segment(monkeypatch):
    """扫描结束时源状态漂移必须失败，并清除本 generation 的全部物理段。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        _, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])
        publisher = MemoryCandidateProjectionPublisher(
            resolver_runtime.resolver, ctx.tiered_segment_store)
        baseline_segments = tuple(
            item.identity_key for item in
            ctx.tiered_segment_store.repository.list_kind(OBJECT_KIND_SEGMENT)
        )
        source_state = publisher._source_state_key()
        calls = 0

        def drifting_source_state():
            """首次返回真实状态，终检返回不同物理水位模拟并发写。"""
            nonlocal calls
            calls += 1
            if calls == 1:
                return source_state
            return source_state[0] + 1, *source_state[1:]

        monkeypatch.setattr(
            publisher, "_source_state_key", drifting_source_state)
        with pytest.raises(MemoryProjectionError, match="扫描期间"):
            publisher.publish(
                _PROJECTION_KEY,
                access=_ACCESS,
                hypothesis_kinds=_KINDS,
                publication=MemoryProjectionPublication(
                    (922, 31),
                    _COLD,
                    (923, 31),
                    _DEPENDENCIES,
                    SegmentBudget(1, 1_000_000),
                    1,
                ),
            )

        assert not any(
            item.descriptor_key == MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY
            for item in ctx.tiered_segment_store.current_manifest().entries)
        assert tuple(
            item.identity_key for item in
            ctx.tiered_segment_store.repository.list_kind(OBJECT_KIND_SEGMENT)
        ) == baseline_segments
    finally:
        backend.close()


def test_projection_publish_error_after_manifest_is_cleaned(monkeypatch):
    """底层已发布 manifest 后再抛错时，publisher 仍须释放该部分 generation。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        _, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])
        publisher = MemoryCandidateProjectionPublisher(
            resolver_runtime.resolver, ctx.tiered_segment_store)
        store = ctx.tiered_segment_store
        baseline_segments = tuple(
            item.identity_key
            for item in store.repository.list_kind(OBJECT_KIND_SEGMENT)
        )
        original_publish = store.publish_segment

        def publish_then_fail(segment, **kwargs):
            """真实完成一次 segment 发布后抛错，模拟返回边界中断。"""
            original_publish(segment, **kwargs)
            raise RuntimeError("publish returned failure")

        monkeypatch.setattr(store, "publish_segment", publish_then_fail)
        with pytest.raises(RuntimeError, match="publish returned failure"):
            publisher.publish(
                _PROJECTION_KEY,
                access=_ACCESS,
                hypothesis_kinds=_KINDS,
                publication=MemoryProjectionPublication(
                    (922, 32),
                    _COLD,
                    (923, 32),
                    _DEPENDENCIES,
                    SegmentBudget(1, 1_000_000),
                    1,
                ),
            )

        assert not any(
            item.descriptor_key == MEMORY_QUERY_PROJECTION_DESCRIPTOR_KEY
            for item in store.current_manifest().entries)
        assert tuple(
            item.identity_key
            for item in store.repository.list_kind(OBJECT_KIND_SEGMENT)
        ) == baseline_segments
    finally:
        backend.close()


@pytest.mark.parametrize("point", (
    FAULT_RELEASE_AFTER_PREPARE,
    FAULT_RELEASE_AFTER_MANIFEST_PUBLISH,
))
def test_projection_release_restart_rolls_forward_to_complete_reclaim(point):
    """release 在 prepared 或 manifest 后中断，重启只能恢复到完整已释放状态。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        _seed_memory(ctx)
        query_source = _query_source(document_id=1)
        _, resolver_runtime = _install_resolver(
            ctx, query_source, _core_refs(ctx)[1])
        manifest = _publish_projection(ctx, resolver_runtime.resolver)
        store = ctx.tiered_segment_store
        with pytest.raises(RuntimeError, match="release fault"):
            store.release_rebuildable_segments(
                tuple(item.segment_key for item in manifest.segments),
                release_key=(9901, point),
                manifest_key=(9902, point),
                fault_injector=_ReleaseFault(point),
            )

        recovered = TieredSegmentStore(
            store.repository,
            store.registry,
            _PROFILE,
        )
        assert all(
            item.segment_key not in {
                segment.segment_key for segment in manifest.segments}
            for item in recovered.current_manifest().entries
        )
        for segment in manifest.segments:
            with pytest.raises(KeyError):
                recovered.repository.get(
                    OBJECT_KIND_SEGMENT,
                    segment_copy_identity(_COLD, segment.segment_key),
                )
    finally:
        backend.close()
