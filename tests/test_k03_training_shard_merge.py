"""K-03 稳定逻辑分片、worker artifact 和确定性 barrier 对抗。"""
from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from pathlib import Path

import pytest

from pure_integer_ai.experiments.training_shard_runtime import (
    SharedTrainingArtifactRepository,
    TrainingShardResourceBudget,
    TrainingShardResourceBudgetExceeded,
    TrainingShardRuntime,
)
from pure_integer_ai.storage import build_storage_role_registry
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.placement import (
    TemperatureProfile,
    TemperatureTier,
)
from pure_integer_ai.storage.sealed_segment import (
    OpenHotDelta,
    SegmentBudget,
    SegmentBudgetExceeded,
    SegmentRecord,
)
from pure_integer_ai.storage.segment_repository import (
    BackendObjectRepository,
    FAULT_OBJECT_AFTER_PART,
    InMemoryObjectRepository,
)
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore
from pure_integer_ai.training.sharded_delta import (
    FAULT_TRAINING_BARRIER_AFTER_MERGE,
    FAULT_TRAINING_BARRIER_AFTER_PUBLISH,
    FAULT_TRAINING_BARRIER_AFTER_RECEIPT,
    OBJECT_KIND_TRAINING_BARRIER_RECEIPT,
    FrozenTrainingManifest,
    LogicalShardPlan,
    LogicalTrainingShard,
    MergedTrainingRecord,
    TrainingAllocationFloor,
    TrainingBarrierCoordinator,
    TrainingBarrierPublishReceipt,
    TrainingBaseReadFence,
    TrainingDeltaRecord,
    TrainingExternalReference,
    TrainingManifestEntry,
    TrainingShardConflictError,
    TrainingShardIntegrityError,
    WorkerDeltaArtifact,
    WorkerLocalDelta,
    training_base_manifest_state_key,
)


_PROFILE = TemperatureProfile(
    (930, 1),
    (
        TemperatureTier((930, 1), 0),
        TemperatureTier((930, 2), 1),
    ),
)
_HOT = (930, 1)
_DESCRIPTOR = MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY
_VERSION = (931, 1)
_PRODUCER_KEY = (932, 1)
_EXECUTION_KEY = (933, 1)
_BARRIER_KEY = (934, 1)
_ALLOCATION_SCOPE = (935, 1)


class _InjectedFailure(RuntimeError):
    """测试在指定 worker 或 barrier 边界主动中断。"""


class _FailAt:
    """仅在指定故障点首次命中时抛出异常。"""

    def __init__(self, point: int) -> None:
        """绑定故障点并创建未触发状态。"""
        self.point = point
        self.triggered = False

    def hit(self, point: int, context: dict[str, object]) -> None:
        """忽略其他边界，在目标边界首次中断。"""
        if point == self.point and not self.triggered:
            self.triggered = True
            raise _InjectedFailure(f"fault point {point}: {context}")


class _SQLiteArtifactRepository:
    """每次操作在线程内打开窄 SQLite 对象仓库，避免连接跨线程。"""

    def __init__(self, path: Path) -> None:
        """绑定共享 SQLite 文件路径，不持有训练 backend。"""
        self.path = path

    def put(self, object_kind, identity_key, payload, *, fault_injector=None):
        """在线程内打开、写入并关闭 append-only 对象仓库。"""
        backend = SQLiteBackend(str(self.path))
        try:
            return BackendObjectRepository(backend).put(
                object_kind,
                identity_key,
                payload,
                fault_injector=fault_injector,
            )
        finally:
            backend.close()

    def get(self, object_kind, identity_key):
        """在线程内打开、读取并关闭 append-only 对象仓库。"""
        backend = SQLiteBackend(str(self.path))
        try:
            return BackendObjectRepository(backend).get(
                object_kind,
                identity_key,
            )
        finally:
            backend.close()

    def list_kind(self, object_kind):
        """在线程内列出指定对象类型后关闭仓库。"""
        backend = SQLiteBackend(str(self.path))
        try:
            return BackendObjectRepository(backend).list_kind(object_kind)
        finally:
            backend.close()

    def reclaim(self, object_kind, identity_key, *, fault_injector=None):
        """在线程内执行对象回收并关闭仓库。"""
        backend = SQLiteBackend(str(self.path))
        try:
            return BackendObjectRepository(backend).reclaim(
                object_kind,
                identity_key,
                fault_injector=fault_injector,
            )
        finally:
            backend.close()


class _Resolver:
    """测试用冻结外部身份 resolver。"""

    def __init__(self) -> None:
        """预装一个 base 身份并绑定稳定状态键。"""
        self.key = (936, 1)
        self.mapping = {(_ALLOCATION_SCOPE, (50,)): 5}

    def state_key(self) -> tuple[int, ...]:
        """返回当前 resolver 状态键。"""
        return self.key

    def resolve(
            self,
            allocation_scope_key: tuple[int, ...],
            external_key: tuple[int, ...],
            ) -> int | None:
        """按完整 scope 和外部身份查询 base local id。"""
        return self.mapping.get((allocation_scope_key, external_key))


class _Producer:
    """从测试映射流式发出一个 shard 的全部增量。"""

    def __init__(self, provider: _ProducerProvider, shard_key: tuple[int, ...]) -> None:
        """绑定 provider 和逻辑 shard 身份。"""
        self.provider = provider
        self.shard_key = shard_key

    def produce(self, request, emit) -> None:
        """按请求 entry 顺序发出增量并记录完成顺序。"""
        self.provider.before(self.shard_key)
        for entry in request.entries:
            for record in self.provider.records.get(entry.input_key, ()):
                emit(record)
        self.provider.completed.append(self.shard_key)
        self.provider.after(self.shard_key)


class _ProducerProvider:
    """提供可并行 producer、稳定逻辑键和调用记录。"""

    def __init__(
            self,
            records: dict[tuple[int, ...], tuple[TrainingDeltaRecord, ...]],
            *,
            state_key: tuple[int, ...] = _PRODUCER_KEY,
            ) -> None:
        """绑定输入到增量的确定性映射。"""
        self.records = records
        self.key = state_key
        self.completed: list[tuple[int, ...]] = []

    def state_key(self) -> tuple[int, ...]:
        """返回 producer 逻辑版本。"""
        return self.key

    def producer_for(self, shard_key: tuple[int, ...]) -> _Producer:
        """为每个 shard 返回独立轻量 producer。"""
        return _Producer(self, shard_key)

    def before(self, shard_key: tuple[int, ...]) -> None:
        """默认不改变 producer 调度。"""

    def after(self, shard_key: tuple[int, ...]) -> None:
        """默认不改变 producer 完成。"""


class _ReverseProducerProvider(_ProducerProvider):
    """用事件链强制四个 producer 按 shard 反序完成。"""

    def __init__(
            self,
            records: dict[tuple[int, ...], tuple[TrainingDeltaRecord, ...]],
            plan: LogicalShardPlan,
            ) -> None:
        """为每个 shard 创建独立完成事件和稳定序号。"""
        super().__init__(records)
        self.rank = {
            shard.shard_key: index for index, shard in enumerate(plan.shards)
        }
        self.events = tuple(threading.Event() for _ in plan.shards)

    def before(self, shard_key: tuple[int, ...]) -> None:
        """较低 shard 等待紧邻较高 shard 完成 producer。"""
        rank = self.rank[shard_key]
        if rank + 1 < len(self.events):
            assert self.events[rank + 1].wait(timeout=5)

    def after(self, shard_key: tuple[int, ...]) -> None:
        """唤醒紧邻较低 shard。"""
        self.events[self.rank[shard_key]].set()


@dataclass
class _Harness:
    """一组共享 K-03 冻结输入、运行时和介质。"""

    manifest: FrozenTrainingManifest
    plan: LogicalShardPlan
    resolver: _Resolver
    provider: _ProducerProvider
    coordinator: TrainingBarrierCoordinator
    runtime: TrainingShardRuntime
    artifact_repository: object
    receipt_repository: InMemoryObjectRepository
    store: TieredSegmentStore


def _manifest(*, checksum_offset: int = 0) -> FrozenTrainingManifest:
    """构造四项冻结 source/course manifest。"""
    entries = tuple(
        TrainingManifestEntry(
            (index,),
            (100 + index,),
            (600, 1),
            0,
            index - 1,
            (700 + checksum_offset, index),
        )
        for index in range(1, 5)
    )
    return FrozenTrainingManifest((937, 1), (938, 1), entries)


def _plan(manifest: FrozenTrainingManifest) -> LogicalShardPlan:
    """把四项输入冻结为四个与 worker 数无关的逻辑 shard。"""
    return LogicalShardPlan(
        (939, 1),
        manifest.manifest_key,
        tuple(
            LogicalTrainingShard((940, index), ((index,),))
            for index in range(1, 5)
        ),
    )


def _records(*, conflict: bool = False, empty: bool = False):
    """构造重复、base id、新 id 和跨 shard 引用混合 fixture。"""
    if empty:
        return {}
    first = TrainingDeltaRecord(
        (1,),
        (1,),
        _ALLOCATION_SCOPE,
        (100,),
        (TrainingExternalReference(_ALLOCATION_SCOPE, (200,)),),
        0,
        (1,),
    )
    duplicate = TrainingDeltaRecord(
        (3,),
        (1,),
        _ALLOCATION_SCOPE,
        (100,),
        (TrainingExternalReference(_ALLOCATION_SCOPE, (200,)),),
        0,
        (99,) if conflict else (1,),
    )
    return {
        (1,): (first,),
        (2,): (TrainingDeltaRecord(
            (2,), (2,), _ALLOCATION_SCOPE, (200,), (), 0, (2,)),),
        (3,): (duplicate,),
        (4,): (TrainingDeltaRecord(
            (4,),
            (3,),
            _ALLOCATION_SCOPE,
            (50,),
            (TrainingExternalReference(_ALLOCATION_SCOPE, (100,)),),
            0,
            (3,),
        ),),
    }


def _store(repository=None) -> TieredSegmentStore:
    """用 K-02 通用角色和两级温层构造介质。"""
    return TieredSegmentStore(
        repository or InMemoryObjectRepository(),
        build_storage_role_registry(),
        _PROFILE,
    )


def _coordinator(
        manifest: FrozenTrainingManifest,
        plan: LogicalShardPlan,
        resolver: _Resolver,
        store: TieredSegmentStore,
        *,
        producer_key: tuple[int, ...] = _PRODUCER_KEY,
        execution_key: tuple[int, ...] = _EXECUTION_KEY,
        barrier_key: tuple[int, ...] = _BARRIER_KEY,
        timeline_floor: int = 20,
        allocation_floor: int = 10,
        output_budget: SegmentBudget = SegmentBudget(16, 32768),
        ) -> TrainingBarrierCoordinator:
    """从 store 当前 manifest 冻结 base fence 并构造协调器。"""
    current = store.current_manifest()
    fence = TrainingBaseReadFence(
        0 if current is None else current.publish_epoch,
        training_base_manifest_state_key(current),
        (941, 1),
        resolver.state_key(),
        (TrainingAllocationFloor(_ALLOCATION_SCOPE, allocation_floor),),
        timeline_floor,
    )
    return TrainingBarrierCoordinator(
        manifest=manifest,
        shard_plan=plan,
        base_fence=fence,
        identity_resolver=resolver,
        producer_key=producer_key,
        execution_key=execution_key,
        barrier_key=barrier_key,
        descriptor_key=_DESCRIPTOR,
        version_key=_VERSION,
        dependencies=(),
        output_budget=output_budget,
        output_segment_key=(942, *barrier_key),
    )


def _harness(
        *,
        provider: _ProducerProvider | None = None,
        artifact_repository=None,
        store: TieredSegmentStore | None = None,
        receipt_repository: InMemoryObjectRepository | None = None,
        manifest: FrozenTrainingManifest | None = None,
        output_budget: SegmentBudget = SegmentBudget(16, 32768),
        telemetry_clock_ns=None,
        working_set_source=None,
        resource_budget: TrainingShardResourceBudget | None = None,
        ) -> _Harness:
    """构造一套默认 K-03 fixture，允许替换物理仓库和预算。"""
    manifest = manifest or _manifest()
    plan = _plan(manifest)
    resolver = _Resolver()
    store = store or _store()
    provider = provider or _ProducerProvider(_records())
    artifact_repository = artifact_repository or InMemoryObjectRepository()
    receipt_repository = receipt_repository or InMemoryObjectRepository()
    coordinator = _coordinator(
        manifest,
        plan,
        resolver,
        store,
        producer_key=provider.state_key(),
        output_budget=output_budget,
    )
    runtime = TrainingShardRuntime(
        coordinator,
        provider,
        SharedTrainingArtifactRepository(artifact_repository),
        SegmentBudget(16, 32768),
        telemetry_clock_ns=telemetry_clock_ns,
        working_set_source=working_set_source,
        resource_budget=resource_budget,
    )
    return _Harness(
        manifest,
        plan,
        resolver,
        provider,
        coordinator,
        runtime,
        artifact_repository,
        receipt_repository,
        store,
    )


def _run(harness: _Harness, worker_count: int, **kwargs):
    """使用稳定发布键执行一次 barrier。"""
    return harness.runtime.run(
        worker_count,
        store=harness.store,
        receipt_repository=harness.receipt_repository,
        tier_key=_HOT,
        manifest_key=(943, 1),
        migration_key=(944, 1),
        **kwargs,
    )


def _manual_artifacts(
        coordinator: TrainingBarrierCoordinator,
        records,
        ) -> tuple[WorkerDeltaArtifact, ...]:
    """不经 executor 构造完整 artifact 集，便于直接攻击 merge。"""
    result = []
    for shard in coordinator.shard_plan.shards:
        delta = WorkerLocalDelta(
            manifest=coordinator.manifest,
            shard_plan=coordinator.shard_plan,
            shard=shard,
            producer_key=coordinator.producer_key,
            execution_key=coordinator.execution_key,
            barrier_key=coordinator.barrier_key,
            base_fence=coordinator.base_fence,
            descriptor_key=coordinator.descriptor_key,
            version_key=coordinator.version_key,
            dependencies=coordinator.dependencies,
            budget=SegmentBudget(16, 32768),
        )
        for input_key in shard.input_keys:
            for record in records.get(input_key, ()):
                delta.append(record)
        result.append(delta.seal())
    return tuple(result)


def _published_bytes(worker_count: int):
    """在全新介质运行并返回 canonical、receipt 和 manifest 字节。"""
    harness = _harness()
    result = _run(harness, worker_count)
    return (
        result.barrier_result.segment.to_bytes(),
        result.receipt.to_bytes(),
        harness.store.current_manifest().to_bytes(),
        result.barrier_result.stable_key(),
    )


def test_worker_count_changes_only_scheduling_and_outputs_are_bit_identical():
    """同一冻结输入用 1、2、4 worker 必须产生完全相同的正式字节。"""
    assert _published_bytes(1) == _published_bytes(2) == _published_bytes(4)


def test_manifest_and_plan_state_keys_do_not_repeat_full_input_payload():
    """状态键长度不随 entry/shard 数增长，内容变化仍由 checksum 失效。"""
    small = _manifest()
    large_entries = tuple(
        TrainingManifestEntry(
            (index,),
            (1000 + index,),
            (600, 1),
            index // 10,
            index,
            (1100, index),
        )
        for index in range(1, 101)
    )
    large = FrozenTrainingManifest((937, 1), (938, 1), large_entries)
    large_plan = LogicalShardPlan(
        (939, 1),
        large.manifest_key,
        tuple(
            LogicalTrainingShard((1200, index), ((index,),))
            for index in range(1, 101)
        ),
    )

    assert len(small.state_key()) == len(large.state_key())
    assert len(_plan(small).state_key()) == len(large_plan.state_key())
    assert small.state_key() != large.state_key()


def test_reverse_producer_completion_does_not_change_canonical_order():
    """producer 反序完成时，timeline 和 canonical segment 仍按冻结顺序。"""
    manifest = _manifest()
    plan = _plan(manifest)
    provider = _ReverseProducerProvider(_records(), plan)
    harness = _harness(provider=provider, manifest=manifest)
    result = _run(harness, 4)

    assert provider.completed == [
        shard.shard_key for shard in reversed(harness.plan.shards)
    ]
    merged = tuple(
        MergedTrainingRecord.from_segment_record(record)
        for record in result.barrier_result.segment.records
    )
    assert tuple(item.timeline_seq for item in merged) == (21, 22, 23)


def test_duplicate_artifact_and_duplicate_record_are_idempotent():
    """精确重复 shard 和跨 shard 同内容记录只计重，不改变输出。"""
    harness = _harness()
    artifacts = _manual_artifacts(harness.coordinator, _records())
    baseline = harness.coordinator.merge(artifacts)
    replay = harness.coordinator.merge((*reversed(artifacts), artifacts[0]))

    assert replay.segment.to_bytes() == baseline.segment.to_bytes()
    assert replay.metrics.duplicate_records == 1
    assert replay.metrics.worker_artifacts == 4


def test_stream_merge_releases_each_worker_segment_before_loading_next():
    """协调器逐 shard 冷读，不能把全部 worker segment 重新堆回热内存。"""
    harness = _harness()
    artifacts = _manual_artifacts(harness.coordinator, _records())
    payloads = {
        artifact.shard_key: artifact.to_bytes() for artifact in artifacts
    }
    baseline = harness.coordinator.merge(artifacts)
    del artifacts
    previous = [None]

    def load(shard: LogicalTrainingShard) -> WorkerDeltaArtifact:
        """加载下一 shard 前确认上一完整 artifact 已释放。"""
        if previous[0] is not None:
            assert previous[0]() is None
        artifact = WorkerDeltaArtifact.from_bytes(payloads[shard.shard_key])
        previous[0] = weakref.ref(artifact)
        return artifact

    streamed = harness.coordinator.merge_stream(load)
    assert streamed.segment.to_bytes() == baseline.segment.to_bytes()


def test_same_merge_identity_with_different_content_hard_fails():
    """同 scope 和 merge key 的载荷冲突不得按完成顺序覆盖。"""
    harness = _harness()
    artifacts = _manual_artifacts(harness.coordinator, _records(conflict=True))
    with pytest.raises(TrainingShardConflictError):
        harness.coordinator.merge(artifacts)


def test_merge_reuses_base_id_assigns_new_ids_and_preserves_reference_scope():
    """merge 必须复用 base id、稳定分配新 id，并保留引用所属 scope。"""
    harness = _harness()
    result = _run(harness, 4)
    assignments = {
        item.external_key: (item.local_id, item.existed_in_base)
        for item in result.barrier_result.assignments
    }
    assert assignments == {
        (50,): (5, True),
        (100,): (11, False),
        (200,): (12, False),
    }
    merged = {
        item.merge_key: item
        for item in (
            MergedTrainingRecord.from_segment_record(record)
            for record in result.barrier_result.segment.records
        )
    }
    assert merged[(1,)].resolved_references[0].allocation_scope_key == (
        _ALLOCATION_SCOPE)
    assert merged[(1,)].resolved_references[0].local_id == 12
    assert merged[(3,)].assigned_local_id == 5
    assert merged[(3,)].resolved_references[0].local_id == 11


def test_missing_shard_and_resolver_drift_fail_before_publish():
    """缺 shard 或 resolver state 漂移均不得形成 canonical segment。"""
    harness = _harness()
    artifacts = _manual_artifacts(harness.coordinator, _records())
    with pytest.raises(TrainingShardIntegrityError):
        harness.coordinator.merge(artifacts[:-1])

    harness.resolver.key = (936, 2)
    with pytest.raises(TrainingShardIntegrityError):
        harness.coordinator.merge(artifacts)
    assert harness.store.current_manifest() is None


def test_empty_shards_persist_and_all_empty_barrier_writes_receipt_only():
    """每个空 shard 都有 artifact，全空 barrier 只发布正式 receipt。"""
    provider = _ProducerProvider(_records(empty=True))
    harness = _harness(provider=provider)
    result = _run(harness, 4)

    assert len(result.artifact_references) == 4
    assert all(not item.has_segment for item in result.artifact_references)
    assert result.barrier_result.segment is None
    assert result.receipt.manifest_epoch == 0
    assert harness.store.current_manifest() is None
    assert len(harness.receipt_repository.list_kind(
        OBJECT_KIND_TRAINING_BARRIER_RECEIPT)) == 1


def test_empty_artifact_binds_producer_and_manifest_full_state():
    """空 artifact 不能跨 producer 或 manifest 内容漂移继续复用。"""
    artifact_repository = InMemoryObjectRepository()
    receipt_repository = InMemoryObjectRepository()
    first_provider = _ProducerProvider(_records(empty=True))
    first = _harness(
        provider=first_provider,
        artifact_repository=artifact_repository,
        receipt_repository=receipt_repository,
    )
    _run(first, 1)

    changed_provider = _ProducerProvider(
        _records(empty=True), state_key=(932, 2))
    changed = _harness(
        provider=changed_provider,
        artifact_repository=artifact_repository,
        receipt_repository=receipt_repository,
    )
    with pytest.raises(TrainingShardIntegrityError):
        _run(changed, 1)

    changed_manifest = _harness(
        provider=first_provider,
        artifact_repository=artifact_repository,
        receipt_repository=receipt_repository,
        manifest=_manifest(checksum_offset=1),
    )
    with pytest.raises(TrainingShardIntegrityError):
        _run(changed_manifest, 1)


def test_worker_artifact_round_trip_and_sqlite_restart(tmp_path: Path):
    """worker artifact 规范往返，并可从 SQLite 文件重启后全部恢复。"""
    path = tmp_path / "k03-worker-artifact.sqlite3"
    repository = _SQLiteArtifactRepository(path)
    provider = _ProducerProvider(_records())
    harness = _harness(
        provider=provider,
        artifact_repository=repository,
    )
    artifact = _manual_artifacts(harness.coordinator, _records())[0]
    assert WorkerDeltaArtifact.from_bytes(
        artifact.to_bytes()) == artifact
    first = _run(harness, 1)
    assert TrainingBarrierPublishReceipt.from_bytes(
        first.receipt.to_bytes()) == first.receipt

    restored_repository = _SQLiteArtifactRepository(path)
    runtime = TrainingShardRuntime(
        harness.coordinator,
        provider,
        SharedTrainingArtifactRepository(restored_repository),
        SegmentBudget(16, 32768),
    )
    result = runtime.run(
        4,
        store=harness.store,
        receipt_repository=harness.receipt_repository,
        tier_key=_HOT,
        manifest_key=(943, 1),
        migration_key=(944, 1),
    )
    assert result.metrics.restored_shards == 4
    assert result.metrics.produced_shards == 0


@pytest.mark.parametrize("point", (
    FAULT_TRAINING_BARRIER_AFTER_MERGE,
    FAULT_TRAINING_BARRIER_AFTER_PUBLISH,
    FAULT_TRAINING_BARRIER_AFTER_RECEIPT,
))
def test_barrier_fault_resume_is_idempotent(point: int):
    """merge、publish、receipt 后崩溃均可恢复且不重复发布 epoch。"""
    harness = _harness()
    fault = _FailAt(point)
    with pytest.raises(_InjectedFailure):
        _run(harness, 4, barrier_fault_injector=fault)
    epoch_after_failure = (
        0 if harness.store.current_manifest() is None
        else harness.store.current_manifest().publish_epoch)
    completed_after_failure = len(harness.provider.completed)

    result = _run(harness, 2)
    assert result.metrics.restored_shards == 4
    assert len(harness.provider.completed) == completed_after_failure
    assert harness.store.current_manifest().publish_epoch == 1
    if point == FAULT_TRAINING_BARRIER_AFTER_MERGE:
        assert epoch_after_failure == 0
    else:
        assert epoch_after_failure == 1


def test_worker_part_failure_never_marks_barrier_complete_and_resume_reuses_successes():
    """worker artifact 半写时无 receipt，恢复后才允许 barrier 完成。"""
    harness = _harness()
    first_artifact = _manual_artifacts(
        harness.coordinator, _records())[0]
    first_artifact.persist(harness.artifact_repository)
    fault = _FailAt(FAULT_OBJECT_AFTER_PART)
    with pytest.raises(_InjectedFailure):
        _run(harness, 1, artifact_fault_injector=fault)
    assert harness.store.current_manifest() is None
    assert harness.receipt_repository.list_kind(
        OBJECT_KIND_TRAINING_BARRIER_RECEIPT) == ()

    result = _run(harness, 2)
    assert result.receipt.manifest_epoch == 1
    assert result.metrics.restored_shards >= 1


def test_output_budget_exceeded_does_not_publish_partial_segments():
    """单 barrier 超出 canonical 预算时硬失败，不自动拆成非原子多段。"""
    harness = _harness(output_budget=SegmentBudget(2, 32768))
    with pytest.raises(SegmentBudgetExceeded):
        _run(harness, 4)
    assert harness.store.current_manifest() is None
    assert harness.receipt_repository.list_kind(
        OBJECT_KIND_TRAINING_BARRIER_RECEIPT) == ()


def _publish_unrelated(store: TieredSegmentStore, ordinal: int) -> None:
    """发布不与 K-03 barrier key range 重叠的同 descriptor 段。"""
    delta = OpenHotDelta(
        _DESCRIPTOR,
        (950, ordinal),
        (),
        SegmentBudget(2, 4096),
    )
    delta.append(SegmentRecord((9999, ordinal), (ordinal,)))
    store.publish_delta(
        delta,
        segment_key=(951, ordinal),
        tier_key=_HOT,
        read_fence=ordinal,
        manifest_key=(952, ordinal),
        migration_key=(953, ordinal),
    )


def test_stale_base_fence_fails_before_new_barrier_publish():
    """base manifest 推进后不得沿旧 allocation floor 发布新 barrier。"""
    harness = _harness()
    _publish_unrelated(harness.store, 1)
    with pytest.raises(TrainingShardIntegrityError):
        _run(harness, 4)
    assert harness.store.current_manifest().publish_epoch == 1
    assert harness.receipt_repository.list_kind(
        OBJECT_KIND_TRAINING_BARRIER_RECEIPT) == ()


def test_receipt_replay_reads_historical_manifest_after_later_epoch():
    """当前 manifest 已推进时，receipt 必须回查其历史 epoch 完整状态。"""
    harness = _harness()
    first = _run(harness, 4)
    _publish_unrelated(harness.store, 2)
    assert harness.store.current_manifest().publish_epoch == 2

    replay = _run(harness, 1)
    assert replay.receipt == first.receipt
    assert harness.store.current_manifest().publish_epoch == 2


def test_distinct_barriers_have_non_overlapping_canonical_key_ranges():
    """barrier 前缀必须让同 descriptor 的连续批次可独立发布。"""
    first = _harness()
    first_result = _run(first, 4)
    first.resolver.mapping.update({
        (_ALLOCATION_SCOPE, (100,)): 11,
        (_ALLOCATION_SCOPE, (200,)): 12,
    })
    first.resolver.key = (936, 2)
    provider = _ProducerProvider(_records())
    second_coordinator = _coordinator(
        first.manifest,
        first.plan,
        first.resolver,
        first.store,
        producer_key=provider.state_key(),
        execution_key=(933, 2),
        barrier_key=(934, 2),
        timeline_floor=23,
        allocation_floor=12,
    )
    second_runtime = TrainingShardRuntime(
        second_coordinator,
        provider,
        SharedTrainingArtifactRepository(InMemoryObjectRepository()),
        SegmentBudget(16, 32768),
    )
    second_result = second_runtime.run(
        2,
        store=first.store,
        receipt_repository=InMemoryObjectRepository(),
        tier_key=_HOT,
        manifest_key=(943, 2),
        migration_key=(944, 2),
    )

    assert first.store.current_manifest().publish_epoch == 2
    assert len(first.store.current_manifest().entries) == 2
    assert (first_result.barrier_result.segment.upper_key
            < second_result.barrier_result.segment.lower_key)


def test_runtime_metrics_expose_bounded_hot_window_without_worker_identity():
    """worker 数只出现在诊断计量，artifact 和 canonical 身份均不含 worker index。"""
    harness = _harness()
    result = _run(harness, 3)
    baseline = _harness()
    baseline_result = _run(baseline, 1)
    assert result.metrics.requested_workers == 3
    assert result.metrics.in_flight_shard_limit == 3
    assert result.metrics.worker_object_limit == 16
    assert result.metrics.worker_byte_limit == 32768
    assert result.metrics.elapsed_ns == 0
    assert result.metrics.peak_working_set_bytes == 0
    assert result.metrics.sealed_cold_bytes > 0
    assert tuple(
        item.identity_key for item in result.artifact_references
    ) == tuple(
        item.identity_key for item in baseline_result.artifact_references)


def test_injected_resource_budget_checks_throughput_ram_cold_bytes_and_amplification():
    """K-03 资源预算使用外部采样逐维硬验，不进入 canonical 身份。"""
    clock_values = iter((100, 200))
    budget = TrainingShardResourceBudget(
        minimum_raw_records_per_window=1,
        throughput_window_ns=100,
        peak_working_set_byte_limit=500,
        sealed_cold_byte_limit=100000,
        write_amplification_numerator_limit=100,
        write_amplification_denominator=1,
    )
    harness = _harness(
        telemetry_clock_ns=lambda: next(clock_values),
        working_set_source=lambda: 500,
        resource_budget=budget,
    )
    result = _run(harness, 4)
    assert result.metrics.elapsed_ns == 100
    assert result.metrics.peak_working_set_bytes == 500

    failing_clock = iter((100, 200))
    failing = _harness(
        telemetry_clock_ns=lambda: next(failing_clock),
        working_set_source=lambda: 500,
        resource_budget=TrainingShardResourceBudget(
            minimum_raw_records_per_window=1,
            throughput_window_ns=100,
            peak_working_set_byte_limit=499,
            sealed_cold_byte_limit=100000,
            write_amplification_numerator_limit=100,
            write_amplification_denominator=1,
        ),
    )
    with pytest.raises(TrainingShardResourceBudgetExceeded):
        _run(failing, 4)
