"""K-02 sealed segment、迁移提交、稳定分页和故障恢复对抗。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.storage import (
    build_storage_role_registry,
    build_tiered_segment_store,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.location_manifest import LocationManifest
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.placement import (
    TemperatureProfile,
    TemperatureTier,
)
from pure_integer_ai.storage.sealed_segment import (
    OpenHotDelta,
    SealedSegment,
    SegmentBudget,
    SegmentBudgetExceeded,
    SegmentRecord,
)
from pure_integer_ai.storage.segment_cache import (
    SegmentCacheError,
    SegmentPageCache,
)
from pure_integer_ai.storage.segment_repository import (
    BackendObjectRepository,
    FAULT_OBJECT_AFTER_PART,
    InMemoryObjectRepository,
    OBJECT_KIND_SEGMENT,
    SEGMENT_OBJECT_SEAL_TABLE,
)
from pure_integer_ai.storage.tiered_segment_store import (
    FAULT_MIGRATION_AFTER_MANIFEST_PUBLISH,
    FAULT_MIGRATION_AFTER_SOURCE_RECLAIM,
    FAULT_MIGRATION_AFTER_TARGET_VERIFY,
    FAULT_MIGRATION_AFTER_TARGET_WRITE,
    FAULT_MIGRATION_BEFORE_SOURCE_RECLAIM,
    FAULT_MIGRATION_AFTER_PREPARE,
    TieredSegmentStore,
    TieredSegmentStoreError,
    segment_copy_identity,
)


_PROFILE = TemperatureProfile(
    (820, 1),
    (
        TemperatureTier((820, 1), 0),
        TemperatureTier((820, 2), 1),
        TemperatureTier((820, 3), 2),
    ),
)
_HOT = (820, 1)
_COLD = (820, 3)
_DESCRIPTOR = MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY


class _InjectedFailure(RuntimeError):
    """测试在指定 K-02 承重边界主动中断。"""


class _FailAt:
    """仅在指定故障点首次命中时抛出异常。"""

    def __init__(self, point: int) -> None:
        """绑定目标故障点并创建未触发状态。"""
        self.point = point
        self.triggered = False

    def hit(self, point: int, context: dict[str, object]) -> None:
        """忽略其他物理点，在目标点首次命中时中断。"""
        if point == self.point and not self.triggered:
            self.triggered = True
            raise _InjectedFailure(f"fault point {point}: {context}")


class _CountingRepository(InMemoryObjectRepository):
    """记录 segment 读取次数，验证范围读取不会隐式全扫冷层。"""

    def __init__(self) -> None:
        """创建空最小介质和零读取计数。"""
        super().__init__()
        self.segment_gets = 0

    def get(self, object_kind: int, identity_key: tuple[int, ...]) -> bytes:
        """读取对象，并只计数真实 segment 副本读取。"""
        if object_kind == OBJECT_KIND_SEGMENT:
            self.segment_gets += 1
        return super().get(object_kind, identity_key)


def _delta(
        start: int,
        stop: int,
        *,
        budget: SegmentBudget | None = None,
        ) -> OpenHotDelta:
    """构造指定闭区间稳定键的权威事件 hot delta。"""
    delta = OpenHotDelta(
        _DESCRIPTOR,
        (1, 1),
        (),
        budget or SegmentBudget(64, 4096),
    )
    for value in range(start, stop + 1):
        delta.append(SegmentRecord((value,), (value * 10, value * 100)))
    return delta


def _store(repository) -> TieredSegmentStore:
    """以同一角色和开放三温层 profile 构造测试 store。"""
    return TieredSegmentStore(
        repository,
        build_storage_role_registry(),
        _PROFILE,
    )


def _publish(
        store: TieredSegmentStore,
        start: int,
        stop: int,
        ordinal: int,
        ) -> LocationManifest:
    """发布一个非重叠 sealed segment 并返回新 manifest。"""
    return store.publish_delta(
        _delta(start, stop),
        segment_key=(830, ordinal),
        tier_key=_HOT,
        read_fence=stop,
        manifest_key=(840, ordinal),
        migration_key=(850, ordinal),
    )


def _read_all(store: TieredSegmentStore, reader_key: tuple[int, ...]):
    """沿稳定 continuation token 读取当前 descriptor 的全部记录。"""
    reader = store.open_reader(reader_key, _DESCRIPTOR)
    records = []
    token = None
    try:
        while True:
            page = reader.page(
                budget=SegmentBudget(2, 4096),
                continuation=token,
            )
            records.extend(page.records)
            if not page.has_more:
                return tuple(records)
            token = page.continuation
    finally:
        reader.close()


def test_sealed_segment_is_deterministic_and_delta_ack_is_post_publish():
    """封段排序和 checksum 确定，发布确认前 hot delta 不得被提前清空。"""
    delta = OpenHotDelta(
        _DESCRIPTOR, (1, 1), (), SegmentBudget(3, 4096))
    delta.append(SegmentRecord((3,), (30,)))
    delta.append(SegmentRecord((1,), (10,)))
    delta.append(SegmentRecord((2,), (20,)))
    segment = delta.seal((830, 1), 9)

    assert delta.object_count == 3
    assert tuple(item.record_key for item in segment.records) == (
        (1,), (2,), (3,))
    assert SealedSegment.from_bytes(segment.to_bytes()) == segment
    assert SealedSegment.from_bytes(segment.to_bytes()).checksum_key == (
        segment.checksum_key)
    delta.acknowledge(segment)
    assert delta.object_count == 0

    limited = _delta(1, 1, budget=SegmentBudget(1, 20))
    with pytest.raises(SegmentBudgetExceeded):
        limited.append(SegmentRecord((2,), tuple(range(30))))
    assert limited.object_count == 1


def test_manifest_round_trip_preserves_full_entries_and_dependencies():
    """K-01 manifest 持久格式必须逐字段往返而非只保存 stable hash。"""
    store = _store(InMemoryObjectRepository())
    manifest = _publish(store, 1, 3, 1)
    assert LocationManifest.from_bytes(manifest.to_bytes()) == manifest


def test_backend_repository_uses_hash_only_as_collision_tolerant_index():
    """强制相同候选索引时，不同完整身份仍必须独立存取。"""
    backend = DictBackend()
    try:
        repository = BackendObjectRepository(
            backend,
            index_key_fn=lambda object_kind, identity_key: 1,
        )
        first = repository.put(OBJECT_KIND_SEGMENT, (1, 2), b"first")
        second = repository.put(OBJECT_KIND_SEGMENT, (1, 3), b"second")
        assert first.identity_key != second.identity_key
        assert repository.get(OBJECT_KIND_SEGMENT, (1, 2)) == b"first"
        assert repository.get(OBJECT_KIND_SEGMENT, (1, 3)) == b"second"
    finally:
        backend.close()


def test_production_factory_builds_capability_negotiated_store():
    """项目正式构造入口必须装配 backend repository、角色和温层协议。"""
    backend = DictBackend()
    try:
        store = build_tiered_segment_store(
            backend,
            build_storage_role_registry(),
            _PROFILE,
        )
        assert store.repository.capability_report().capabilities
        _publish(store, 1, 2, 1)
        assert tuple(item.record_key for item in _read_all(
            store, (860, 8))) == ((1,), (2,))
    finally:
        backend.close()


def test_authoritative_segment_cannot_be_released_without_replacement():
    """权威事件段不得借可重建 release 协议从 manifest 和介质中移除。"""
    store = _store(InMemoryObjectRepository())
    _publish(store, 1, 2, 1)
    with pytest.raises(TieredSegmentStoreError, match="只有可重建"):
        store.release_rebuildable_segments(
            ((830, 1),),
            release_key=(865, 1),
            manifest_key=(866, 1),
        )


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_native_reclamation_removes_old_segment_seal_rows(backend_type):
    """具备 reclamation 能力的后端迁移后必须物理删除源 seal，而非只读时隐藏。"""
    backend = backend_type()
    try:
        store = _store(BackendObjectRepository(backend))
        _publish(store, 1, 3, 1)
        store.migrate(
            (830, 1),
            target_tier_key=_COLD,
            manifest_key=(840, 2),
            migration_key=(850, 2),
        )
        segment_seals = backend.select(
            SEGMENT_OBJECT_SEAL_TABLE,
            where={"object_kind": OBJECT_KIND_SEGMENT},
        )
        assert len(segment_seals) == 1
    finally:
        backend.close()


def test_mid_segment_write_is_invisible_and_restart_reclaims_orphan_parts():
    """段内 part 写入中断不得形成可见对象，重启后可从原 hot delta 重试。"""
    backend = DictBackend()
    try:
        repository = BackendObjectRepository(backend)
        store = _store(repository)
        delta = _delta(1, 5)
        fault = _FailAt(FAULT_OBJECT_AFTER_PART)
        with pytest.raises(_InjectedFailure):
            store.publish_delta(
                delta,
                segment_key=(830, 1),
                tier_key=_HOT,
                read_fence=5,
                manifest_key=(840, 1),
                migration_key=(850, 1),
                fault_injector=fault,
            )
        assert delta.object_count == 5
        assert repository.list_kind(OBJECT_KIND_SEGMENT) == ()

        restored = _store(BackendObjectRepository(backend))
        assert restored.current_manifest() is None
        restored.publish_delta(
            delta,
            segment_key=(830, 1),
            tier_key=_HOT,
            read_fence=5,
            manifest_key=(840, 1),
            migration_key=(850, 2),
        )
        assert delta.object_count == 0
        assert tuple(item.record_key for item in _read_all(
            restored, (860, 9))) == tuple((value,) for value in range(1, 6))
    finally:
        backend.close()


@pytest.mark.parametrize("backend_kind", ("dict", "sqlite_memory", "sqlite_file"))
def test_backend_adapters_restart_with_complete_manifest_and_rows(
        backend_kind: str,
        tmp_path: Path,
        ):
    """Dict、SQLite 内存和文件适配器重建 store 后读取结果一致。"""
    path = tmp_path / f"k02-{backend_kind}.sqlite3"
    if backend_kind == "dict":
        backend = DictBackend()
    elif backend_kind == "sqlite_memory":
        backend = SQLiteBackend()
    else:
        backend = SQLiteBackend(str(path))
    repository = BackendObjectRepository(backend)
    store = _store(repository)
    _publish(store, 1, 5, 1)

    if backend_kind == "dict":
        snapshot = backend.snapshot()
        backend.close()
        reopened = DictBackend()
        BackendObjectRepository(reopened)
        reopened.load_snapshot(snapshot)
        backend = reopened
    elif backend_kind == "sqlite_file":
        backend.close()
        backend = SQLiteBackend(str(path))
    try:
        restored = _store(BackendObjectRepository(backend))
        assert tuple(item.record_key for item in _read_all(
            restored, (860, 1))) == tuple((value,) for value in range(1, 6))
        assert restored.current_manifest().publish_epoch == 1
    finally:
        backend.close()


def test_continuation_is_stable_across_new_publish_and_old_reader_barrier():
    """旧 reader 只读旧 epoch，新发布不造成 offset 漂移，迁移回收等待更早 reader。"""
    repository = InMemoryObjectRepository()
    store = _store(repository)
    _publish(store, 1, 4, 1)
    old_reader = store.open_reader((861, 1), _DESCRIPTOR)
    first = old_reader.page(budget=SegmentBudget(2, 4096))

    _publish(store, 5, 8, 2)
    second = old_reader.page(
        budget=SegmentBudget(2, 4096),
        continuation=first.continuation,
    )
    assert tuple(item.record_key for item in (*first.records, *second.records)) == (
        (1,), (2,), (3,), (4,))
    assert second.has_more is False

    store.migrate(
        (830, 1),
        target_tier_key=_COLD,
        manifest_key=(840, 3),
        migration_key=(850, 3),
    )
    source_identity = segment_copy_identity(_HOT, (830, 1))
    target_identity = segment_copy_identity(_COLD, (830, 1))
    assert repository.get(OBJECT_KIND_SEGMENT, source_identity)
    assert repository.get(OBJECT_KIND_SEGMENT, target_identity)

    assert tuple(item.record_key for item in _read_all(
        store, (861, 2))) == tuple((value,) for value in range(1, 9))
    old_reader.close()
    with pytest.raises(KeyError):
        repository.get(OBJECT_KIND_SEGMENT, source_identity)


def test_range_reader_touches_only_manifest_segments_that_can_match():
    """冷层范围读取只 page-in 相交段，不因缺业务索引而全扫所有段。"""
    repository = _CountingRepository()
    store = _store(repository)
    _publish(store, 1, 3, 1)
    _publish(store, 10, 12, 2)
    _publish(store, 20, 22, 3)
    repository.segment_gets = 0
    reader = store.open_reader((862, 1), _DESCRIPTOR)
    try:
        page = reader.page(
            budget=SegmentBudget(10, 4096),
            lower_key=(10,),
            upper_key=(12,),
        )
        assert tuple(item.record_key for item in page.records) == (
            (10,), (11,), (12,))
        assert repository.segment_gets == 1
    finally:
        reader.close()


def test_page_cache_requires_dirty_flush_and_uses_clean_evict_for_budget():
    """热集预算只能自动淘汰 clean 对象，dirty 对象必须先批量 flush。"""
    cache = SegmentPageCache(SegmentBudget(2, 4096))
    first = SegmentRecord((1,), (10,))
    second = SegmentRecord((2,), (20,))
    third = SegmentRecord((3,), (30,))
    cache.page_in(_DESCRIPTOR, (first, second))
    cache.put_dirty(_DESCRIPTOR, third)
    assert cache.object_count == 2
    dirty_key = (_DESCRIPTOR, third.record_key)
    with pytest.raises(SegmentCacheError):
        cache.evict((dirty_key,))
    flushed = []
    assert cache.evict(
        (dirty_key,),
        flush=lambda records: flushed.extend(records),
    ) == 1
    assert tuple(item.record for item in flushed) == (third,)


def test_compaction_replaces_multiple_segments_with_budget_and_reader_barrier():
    """compaction 在硬预算内发布单一新段，并等待旧 reader 后回收全部源段。"""
    repository = InMemoryObjectRepository()
    store = _store(repository)
    _publish(store, 1, 3, 1)
    _publish(store, 4, 6, 2)
    old_reader = store.open_reader((863, 1), _DESCRIPTOR)

    manifest = store.compact(
        ((830, 1), (830, 2)),
        target_segment_key=(830, 9),
        target_tier_key=_COLD,
        version_key=(2, 1),
        read_fence=6,
        budget=SegmentBudget(6, 4096),
        manifest_key=(840, 9),
        migration_key=(850, 9),
    )
    assert manifest.publish_epoch == 3
    assert tuple(item.segment_key for item in manifest.entries) == ((830, 9),)
    assert tuple(item.record_key for item in _read_all(
        store, (863, 2))) == tuple((value,) for value in range(1, 7))
    assert repository.get(
        OBJECT_KIND_SEGMENT, segment_copy_identity(_HOT, (830, 1)))
    assert repository.get(
        OBJECT_KIND_SEGMENT, segment_copy_identity(_HOT, (830, 2)))

    old_page = old_reader.page(budget=SegmentBudget(6, 4096))
    assert tuple(item.record_key for item in old_page.records) == tuple(
        (value,) for value in range(1, 7))
    old_reader.close()
    for source_key in ((830, 1), (830, 2)):
        with pytest.raises(KeyError):
            repository.get(
                OBJECT_KIND_SEGMENT,
                segment_copy_identity(_HOT, source_key),
            )


def test_compaction_budget_failure_keeps_complete_old_epoch():
    """compaction 任一硬预算不足时不得写目标、发布 manifest 或清理源段。"""
    repository = InMemoryObjectRepository()
    store = _store(repository)
    _publish(store, 1, 3, 1)
    _publish(store, 4, 6, 2)
    with pytest.raises(SegmentBudgetExceeded):
        store.compact(
            ((830, 1), (830, 2)),
            target_segment_key=(830, 9),
            target_tier_key=_COLD,
            version_key=(2, 1),
            read_fence=6,
            budget=SegmentBudget(5, 4096),
            manifest_key=(840, 9),
            migration_key=(850, 9),
        )
    assert store.current_manifest().publish_epoch == 2
    assert tuple(item.record_key for item in _read_all(
        store, (864, 1))) == tuple((value,) for value in range(1, 7))
    assert len(repository.list_kind(OBJECT_KIND_SEGMENT)) == 2


def test_prepared_without_verified_target_rolls_back_to_complete_old_epoch():
    """prepared 后目标副本丢失时必须 abort 并保留旧 manifest 与源段。"""
    repository = InMemoryObjectRepository()
    store = _store(repository)
    _publish(store, 1, 5, 1)
    fault = _FailAt(FAULT_MIGRATION_AFTER_PREPARE)
    with pytest.raises(_InjectedFailure):
        store.migrate(
            (830, 1),
            target_tier_key=_COLD,
            manifest_key=(840, 2),
            migration_key=(850, 2),
            fault_injector=fault,
        )
    assert repository.reclaim(
        OBJECT_KIND_SEGMENT,
        segment_copy_identity(_COLD, (830, 1)),
    )

    restored = _store(repository)
    assert restored.current_manifest().publish_epoch == 1
    assert restored.current_manifest().entries[0].tier_key == _HOT
    assert tuple(item.record_key for item in _read_all(
        restored, (865, 1))) == tuple((value,) for value in range(1, 6))
    assert len(repository.list_kind(OBJECT_KIND_SEGMENT)) == 1


_FAULT_CASES = (
    (FAULT_MIGRATION_AFTER_TARGET_WRITE, 1),
    (FAULT_MIGRATION_AFTER_TARGET_VERIFY, 1),
    (FAULT_MIGRATION_AFTER_PREPARE, 2),
    (FAULT_MIGRATION_AFTER_MANIFEST_PUBLISH, 2),
    (FAULT_MIGRATION_BEFORE_SOURCE_RECLAIM, 2),
    (FAULT_MIGRATION_AFTER_SOURCE_RECLAIM, 2),
)


@pytest.mark.parametrize("fault_point, expected_epoch", _FAULT_CASES)
@pytest.mark.parametrize(
    "adapter_kind", ("minimal", "dict", "sqlite_memory", "sqlite_file"))
def test_faults_recover_to_complete_old_or_new_epoch(
        adapter_kind: str,
        fault_point: int,
        expected_epoch: int,
        tmp_path: Path,
        ):
    """写段、核验、发布和回收故障后只能读取完整旧 epoch 或完整新 epoch。"""
    path = tmp_path / f"k02-fault-{adapter_kind}-{fault_point}.sqlite3"
    backend = None
    if adapter_kind == "minimal":
        repository = InMemoryObjectRepository()
    else:
        if adapter_kind == "dict":
            backend = DictBackend()
        elif adapter_kind == "sqlite_memory":
            backend = SQLiteBackend()
        else:
            backend = SQLiteBackend(str(path))
        repository = BackendObjectRepository(backend)
    store = _store(repository)
    _publish(store, 1, 5, 1)
    fault = _FailAt(fault_point)
    with pytest.raises(_InjectedFailure):
        store.migrate(
            (830, 1),
            target_tier_key=_COLD,
            manifest_key=(840, 2),
            migration_key=(850, 2),
            fault_injector=fault,
        )
    assert fault.triggered is True

    if adapter_kind == "dict":
        snapshot = backend.snapshot()
        backend.close()
        backend = DictBackend()
        BackendObjectRepository(backend)
        backend.load_snapshot(snapshot)
        repository = BackendObjectRepository(backend)
    elif adapter_kind == "sqlite_file":
        backend.close()
        backend = SQLiteBackend(str(path))
        repository = BackendObjectRepository(backend)
    elif adapter_kind == "sqlite_memory":
        repository = BackendObjectRepository(backend)

    try:
        restored = _store(repository)
        manifest = restored.current_manifest()
        assert manifest is not None
        assert manifest.publish_epoch == expected_epoch
        assert tuple(item.record_key for item in _read_all(
            restored, (870, fault_point))) == (
                (1,), (2,), (3,), (4,), (5,))
        assert len(repository.list_kind(OBJECT_KIND_SEGMENT)) == 1
        tier = manifest.entries[0].tier_key
        assert tier == (_HOT if expected_epoch == 1 else _COLD)
    finally:
        if backend is not None:
            backend.close()
