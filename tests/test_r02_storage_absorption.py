"""R-02 write intent、single verified get 与单段 reader cache 生产测试。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_storage_absorption_catalog import (
    build_storage_absorption_manifest,
)
from pure_integer_ai.experiments.ph2_storage_absorption_contract import (
    StorageAbsorptionContractError,
    StorageEvidenceFile,
    read_storage_absorption_manifest,
    verify_storage_absorption_files,
    write_storage_absorption_manifest,
)
from pure_integer_ai.storage import build_storage_role_registry
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.placement import TemperatureProfile, TemperatureTier
from pure_integer_ai.storage.sealed_segment import (
    OpenHotDelta,
    SealedSegment,
    SegmentBudget,
    SegmentRecord,
)
from pure_integer_ai.storage.segment_commit import (
    MIGRATION_PHASE_PREPARED,
    MigrationCommitRecord,
)
from pure_integer_ai.storage.segment_repository import (
    BackendObjectRepository,
    InMemoryObjectRepository,
    OBJECT_KIND_LOCATION_MANIFEST,
    OBJECT_KIND_MIGRATION_COMMIT,
    OBJECT_KIND_SEGMENT,
    OBJECT_KIND_SEGMENT_RELEASE,
    OBJECT_KIND_SEGMENT_WRITE_INTENT,
)
from pure_integer_ai.storage.segment_write_intent import SegmentWriteIntent
from pure_integer_ai.storage.tiered_segment_store import (
    FAULT_MIGRATION_AFTER_MANIFEST_PUBLISH,
    FAULT_MIGRATION_AFTER_PREPARE,
    FAULT_MIGRATION_AFTER_TARGET_VERIFY,
    FAULT_MIGRATION_AFTER_TARGET_WRITE,
    TieredSegmentStore,
    segment_copy_identity,
)


_DESCRIPTOR = MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY
_HOT = (20260728, 201, 1)
_COLD = (20260728, 201, 2)
_PROFILE = TemperatureProfile(
    (20260728, 201, 3),
    (
        TemperatureTier(_HOT, 0),
        TemperatureTier(_COLD, 1),
    ),
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPOSITORY_ROOT.parent


class _InjectedFailure(RuntimeError):
    """在指定 migration 承重点模拟进程中断。"""


class _FailAt:
    """只在目标故障点首次命中时中断。"""

    def __init__(self, point: int) -> None:
        self.point = point
        self.triggered = False

    def hit(self, point: int, context: dict[str, object]) -> None:
        if point == self.point and not self.triggered:
            self.triggered = True
            raise _InjectedFailure(f"fault point {point}: {context}")


class _CountingBackendObjectRepository(BackendObjectRepository):
    """按 object kind 记录实际完整 payload 解码次数与字节。"""

    def __init__(self, backend) -> None:
        self.payload_gets: dict[int, int] = defaultdict(int)
        self.payload_bytes: dict[int, int] = defaultdict(int)
        super().__init__(backend)

    def _read_object_id(self, object_id: int):
        descriptor, payload = super()._read_object_id(object_id)
        self.payload_gets[descriptor.object_kind] += 1
        self.payload_bytes[descriptor.object_kind] += len(payload)
        return descriptor, payload

    def reset_metrics(self) -> None:
        self.payload_gets.clear()
        self.payload_bytes.clear()


def _store(repository) -> TieredSegmentStore:
    return TieredSegmentStore(
        repository,
        build_storage_role_registry(),
        _PROFILE,
    )


def _delta(start: int, stop: int) -> OpenHotDelta:
    delta = OpenHotDelta(
        _DESCRIPTOR,
        (20260728, 202, 1),
        (),
        SegmentBudget(max(1, stop - start + 1), 1_000_000),
    )
    for value in range(start, stop + 1):
        delta.append(SegmentRecord((value,), (value * 10, value * 100)))
    return delta


def _publish(
        store: TieredSegmentStore,
        start: int,
        stop: int,
        ordinal: int,
        *,
        fault_injector=None,
        ) -> None:
    store.publish_delta(
        _delta(start, stop),
        segment_key=(20260728, 203, ordinal),
        tier_key=_HOT,
        read_fence=stop,
        manifest_key=(20260728, 204, ordinal),
        migration_key=(20260728, 205, ordinal),
        fault_injector=fault_injector,
    )


def _read_range(
        store: TieredSegmentStore,
        reader_key: tuple[int, ...],
        *,
        lower: tuple[int, ...] | None = None,
        upper: tuple[int, ...] | None = None,
        page_size: int = 3,
        ) -> tuple[SegmentRecord, ...]:
    reader = store.open_reader(reader_key, _DESCRIPTOR)
    records = []
    continuation = None
    try:
        while True:
            page = reader.page(
                budget=SegmentBudget(page_size, 1_000_000),
                lower_key=lower,
                upper_key=upper,
                continuation=continuation,
            )
            records.extend(page.records)
            if not page.has_more:
                return tuple(records)
            continuation = page.continuation
    finally:
        reader.close()


def test_object_kind_extension_and_intent_round_trip_preserve_old_numbers():
    """write intent 只追加 kind=5，旧四类编号和 commit 编码不漂移。"""
    assert (
        OBJECT_KIND_SEGMENT,
        OBJECT_KIND_LOCATION_MANIFEST,
        OBJECT_KIND_MIGRATION_COMMIT,
        OBJECT_KIND_SEGMENT_RELEASE,
        OBJECT_KIND_SEGMENT_WRITE_INTENT,
    ) == (1, 2, 3, 4, 5)
    prepared = MigrationCommitRecord(
        (1, 2), MIGRATION_PHASE_PREPARED, _DESCRIPTOR, (3, 4), (),
        _COLD, (5, 6), (7, 8), 9, 0, 1, (10, 11))
    intent = SegmentWriteIntent.from_prepared(prepared)
    assert SegmentWriteIntent.from_bytes(intent.to_bytes()) == intent
    assert intent.identity_key() == (2, 1, 2)


def test_backend_get_and_idempotent_put_decode_matching_payload_once():
    """direct get 与已存在 put 均只做一次完整 checksum/identity 核验。"""
    backend = DictBackend()
    try:
        repository = _CountingBackendObjectRepository(backend)
        repository.put(OBJECT_KIND_SEGMENT, (1, 2), b"payload")
        repository.reset_metrics()
        assert repository.get(OBJECT_KIND_SEGMENT, (1, 2)) == b"payload"
        assert repository.payload_gets[OBJECT_KIND_SEGMENT] == 1
        repository.reset_metrics()
        repository.put(OBJECT_KIND_SEGMENT, (1, 2), b"payload")
        assert repository.payload_gets[OBJECT_KIND_SEGMENT] == 1
    finally:
        backend.close()


@pytest.mark.parametrize("point", (
    FAULT_MIGRATION_AFTER_TARGET_WRITE,
    FAULT_MIGRATION_AFTER_TARGET_VERIFY,
    FAULT_MIGRATION_AFTER_PREPARE,
    FAULT_MIGRATION_AFTER_MANIFEST_PUBLISH,
))
def test_write_intent_recovers_target_faults_and_clears_at_terminal_state(point):
    """target 写、核验、PREPARED 和 publish 中断均定点恢复且清空 intent。"""
    backend = DictBackend()
    try:
        repository = BackendObjectRepository(backend)
        store = _store(repository)
        with pytest.raises(_InjectedFailure):
            _publish(store, 1, 4, 1, fault_injector=_FailAt(point))
        assert len(repository.list_kind(
            OBJECT_KIND_SEGMENT_WRITE_INTENT)) == 1

        recovered_repository = BackendObjectRepository(backend)
        recovered = _store(recovered_repository)
        assert recovered.active_write_intent_count() == 0
        assert recovered_repository.list_kind(
            OBJECT_KIND_SEGMENT_WRITE_INTENT) == ()
        if point in {
                FAULT_MIGRATION_AFTER_TARGET_WRITE,
                FAULT_MIGRATION_AFTER_TARGET_VERIFY}:
            assert recovered.current_manifest() is None
            assert recovered_repository.list_kind(OBJECT_KIND_SEGMENT) == ()
        else:
            assert recovered.current_manifest() is not None
            assert tuple(item.record_key for item in _read_range(
                recovered, (20260728, 206, point))) == (
                    (1,), (2,), (3,), (4,))
    finally:
        backend.close()


def test_restart_reads_no_segment_payload_and_reader_cache_is_query_local(
        tmp_path: Path):
    """启动 0 payload；exact 每段一次；audit 每段一次且不跨 query 缓存。"""
    database = tmp_path / "r02-restart.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        store = _store(BackendObjectRepository(backend))
        for ordinal, start in enumerate((1, 5, 9), start=1):
            _publish(store, start, start + 3, ordinal)
        assert store.active_write_intent_count() == 0
    finally:
        backend.close()

    backend = SQLiteBackend(str(database))
    try:
        repository = _CountingBackendObjectRepository(backend)
        store = _store(repository)
        assert repository.payload_gets[OBJECT_KIND_SEGMENT] == 0
        assert repository.payload_bytes[OBJECT_KIND_SEGMENT] == 0

        repository.reset_metrics()
        for ordinal, target in enumerate((1, 5, 9), start=1):
            records = _read_range(
                store,
                (20260728, 207, ordinal),
                lower=(target,),
                upper=(target,),
                page_size=1,
            )
            assert tuple(item.record_key for item in records) == ((target,),)
        assert repository.payload_gets[OBJECT_KIND_SEGMENT] == 3

        repository.reset_metrics()
        assert len(_read_range(
            store, (20260728, 208, 1), page_size=3)) == 12
        assert repository.payload_gets[OBJECT_KIND_SEGMENT] == 3
        assert store.reader_epochs.snapshot() == ()

        _read_range(
            store, (20260728, 208, 2), lower=(1,), upper=(1,), page_size=1)
        assert repository.payload_gets[OBJECT_KIND_SEGMENT] == 4
        assert store.active_write_intent_count() == 0
    finally:
        backend.close()


def test_unknown_legacy_orphan_requires_explicit_full_media_audit():
    """无 intent 的旧 segment 不拖累启动，只由显式 maintenance 回收。"""
    repository = InMemoryObjectRepository()
    segment = SealedSegment(
        _DESCRIPTOR,
        (20260728, 209, 1),
        (20260728, 209, 2),
        (),
        1,
        (SegmentRecord((1,), (10,)),),
    )
    repository.put(
        OBJECT_KIND_SEGMENT,
        segment_copy_identity(_HOT, segment.segment_key),
        segment.to_bytes(),
    )
    store = _store(repository)
    assert store.current_manifest() is None
    assert len(repository.list_kind(OBJECT_KIND_SEGMENT)) == 1
    assert store.audit_and_reclaim_legacy_orphans() == 1
    assert repository.list_kind(OBJECT_KIND_SEGMENT) == ()


@pytest.fixture(scope="module")
def formal_storage_manifest():
    """从冻结的三进程 profile 与最终实现字节构建正式 R-02 证据。"""
    return build_storage_absorption_manifest(
        _REPOSITORY_ROOT,
        _WORKSPACE_ROOT,
    )


def test_formal_profile_manifest_round_trip_and_file_identity(
        tmp_path: Path,
        formal_storage_manifest):
    """正式 artifact 可规范回读，且仓内外证据逐字节闭合。"""
    target = tmp_path / "r02_storage_absorption_v1.json"
    assert write_storage_absorption_manifest(
        formal_storage_manifest, target) == target
    assert read_storage_absorption_manifest(target) == formal_storage_manifest
    verify_storage_absorption_files(
        formal_storage_manifest,
        repository_root=_REPOSITORY_ROOT,
        workspace_root=_WORKSPACE_ROOT,
    )


def test_formal_manifest_rejects_fake_profile_metric(formal_storage_manifest):
    """profile 硬门不能被非实测的启动 payload 指标冒充。"""
    metrics = formal_storage_manifest.profile_metrics.to_value()
    metrics["startup_segment_payload_gets"] = 1
    with pytest.raises(
            StorageAbsorptionContractError,
            match="十万级 profile 硬门未通过"):
        replace(
            formal_storage_manifest,
            profile_metrics=CanonicalJsonObject.from_value(metrics),
        )


def test_formal_manifest_rejects_evidence_identity_drift(
        formal_storage_manifest):
    """任一实现、测试、报告或 SQLite 字节漂移均拒绝回验。"""
    first = formal_storage_manifest.evidence_files[0]
    drifted = StorageEvidenceFile(
        first.root_key,
        first.relative_path,
        first.role,
        first.byte_count,
        "0" * 64,
    )
    manifest = replace(
        formal_storage_manifest,
        evidence_files=(drifted, *formal_storage_manifest.evidence_files[1:]),
    )
    with pytest.raises(
            StorageAbsorptionContractError,
            match="evidence 文件身份漂移"):
        verify_storage_absorption_files(
            manifest,
            repository_root=_REPOSITORY_ROOT,
            workspace_root=_WORKSPACE_ROOT,
        )


def test_formal_manifest_is_idempotent_but_non_overwritable(
        tmp_path: Path,
        formal_storage_manifest):
    """同字节可幂等重放，同版本不同字节禁止覆盖。"""
    target = tmp_path / "r02_storage_absorption_v1.json"
    write_storage_absorption_manifest(formal_storage_manifest, target)
    assert write_storage_absorption_manifest(
        formal_storage_manifest, target) == target
    first = formal_storage_manifest.evidence_files[0]
    drifted = replace(first, sha256="f" * 64)
    different = replace(
        formal_storage_manifest,
        evidence_files=(drifted, *formal_storage_manifest.evidence_files[1:]),
    )
    with pytest.raises(
            StorageAbsorptionContractError,
            match="已存在且内容不同"):
        write_storage_absorption_manifest(different, target)
