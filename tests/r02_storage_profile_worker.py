"""R-02 十万 record SQLite profile 的独立进程测试 worker。"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.storage import build_storage_role_registry
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.integer_codec import encode_integer_tuple
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.placement import TemperatureProfile, TemperatureTier
from pure_integer_ai.storage.sealed_segment import (
    OpenHotDelta,
    SegmentBudget,
    SegmentRecord,
)
from pure_integer_ai.storage.segment_repository import (
    BackendObjectRepository,
    OBJECT_KIND_SEGMENT,
)
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore


RECORD_COUNT = 100_000
RECORDS_PER_SEGMENT = 1_000
SEGMENT_COUNT = 100
TARGETS = (1, 1000, 1001, 25_000, 25_001, 50_000, 75_000, 99_000, 99_999, 100_000)
AUDIT_PAGE_OBJECT_LIMIT = 256
BYTE_LIMIT = 1_000_000
_DESCRIPTOR = MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY
_HOT = (20260728, 210, 1)
_COLD = (20260728, 210, 2)
_PROFILE = TemperatureProfile(
    (20260728, 210, 3),
    (
        TemperatureTier(_HOT, 0),
        TemperatureTier(_COLD, 1),
    ),
)


class CountingBackendObjectRepository(BackendObjectRepository):
    """记录真实完整 payload 解码次数，不改变 repository 行为。"""

    def __init__(self, backend: SQLiteBackend) -> None:
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


def _record(value: int) -> SegmentRecord:
    """构造只用于物理 profile 的确定整数记录，不进入生产语义。"""
    return SegmentRecord(
        (1, value),
        (
            value,
            value + 1,
            value + 2,
            value % 997 + 1,
            value % 991 + 1,
            value % 983 + 1,
            value % 977 + 1,
            value,
        ),
    )


def _update_digest(digest, record: SegmentRecord) -> None:
    values = (
        len(record.record_key),
        *record.record_key,
        len(record.payload),
        *record.payload,
    )
    digest.update(encode_integer_tuple(values))


def _store(repository: BackendObjectRepository) -> TieredSegmentStore:
    return TieredSegmentStore(
        repository,
        build_storage_role_registry(),
        _PROFILE,
    )


def _write_report(path: Path, value: dict[str, object]) -> None:
    payload = canonical_json_line(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError("R-02 profile report 已存在且内容不同")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def build(database: Path, report_path: Path) -> None:
    """在新 SQLite 中流式发布 100 个互不重叠 segment。"""
    if database.exists():
        raise RuntimeError("R-02 build database 已存在")
    backend = SQLiteBackend(str(database))
    digest = hashlib.sha256()
    maximum_segment_bytes = 0
    try:
        store = _store(BackendObjectRepository(backend))
        for ordinal in range(1, SEGMENT_COUNT + 1):
            start = (ordinal - 1) * RECORDS_PER_SEGMENT + 1
            stop = ordinal * RECORDS_PER_SEGMENT
            delta = OpenHotDelta(
                _DESCRIPTOR,
                (20260728, 211, 1),
                (),
                SegmentBudget(RECORDS_PER_SEGMENT, BYTE_LIMIT),
            )
            for value in range(start, stop + 1):
                record = _record(value)
                delta.append(record)
                _update_digest(digest, record)
            segment = delta.seal((20260728, 212, ordinal), stop)
            maximum_segment_bytes = max(
                maximum_segment_bytes, len(segment.to_bytes()))
            store.publish_delta(
                delta,
                segment_key=segment.segment_key,
                tier_key=_COLD,
                read_fence=stop,
                manifest_key=(20260728, 213, ordinal),
                migration_key=(20260728, 214, ordinal),
            )
        manifest = store.current_manifest()
        if manifest is None:
            raise RuntimeError("R-02 build 缺最终 manifest")
        report = {
            "active_write_intents": store.active_write_intent_count(),
            "content_sha256": digest.hexdigest(),
            "manifest_entry_count": len(manifest.entries),
            "mode": "BUILD",
            "record_count": RECORD_COUNT,
            "records_per_segment": RECORDS_PER_SEGMENT,
            "segment_count": SEGMENT_COUNT,
            "max_segment_bytes": maximum_segment_bytes,
        }
    finally:
        backend.close()
    _write_report(report_path, report)


def query(database: Path, report_path: Path) -> None:
    """冷启动后执行十个互相独立的 exact reader。"""
    backend = SQLiteBackend(str(database))
    try:
        repository = CountingBackendObjectRepository(backend)
        store = _store(repository)
        startup_gets = repository.payload_gets[OBJECT_KIND_SEGMENT]
        startup_bytes = repository.payload_bytes[OBJECT_KIND_SEGMENT]
        repository.reset_metrics()
        rows = []
        for ordinal, target in enumerate(TARGETS, start=1):
            before_gets = repository.payload_gets[OBJECT_KIND_SEGMENT]
            before_bytes = repository.payload_bytes[OBJECT_KIND_SEGMENT]
            reader = store.open_reader((20260728, 215, ordinal), _DESCRIPTOR)
            try:
                page = reader.page(
                    budget=SegmentBudget(1, BYTE_LIMIT),
                    lower_key=(1, target),
                    upper_key=(1, target),
                )
            finally:
                reader.close()
            if (tuple(item.record_key for item in page.records)
                    != ((1, target),)):
                raise RuntimeError("R-02 exact query 结果漂移")
            rows.append({
                "record_count": len(page.records),
                "segment_payload_bytes": (
                    repository.payload_bytes[OBJECT_KIND_SEGMENT]
                    - before_bytes),
                "segment_payload_gets": (
                    repository.payload_gets[OBJECT_KIND_SEGMENT]
                    - before_gets),
                "target": target,
            })
        manifest = store.current_manifest()
        if manifest is None:
            raise RuntimeError("R-02 query 缺 manifest")
        report = {
            "active_readers_after": len(store.reader_epochs.snapshot()),
            "active_write_intents": store.active_write_intent_count(),
            "manifest_entry_count": len(manifest.entries),
            "mode": "QUERY",
            "queries": rows,
            "query_segment_payload_bytes": repository.payload_bytes[
                OBJECT_KIND_SEGMENT],
            "query_segment_payload_gets": repository.payload_gets[
                OBJECT_KIND_SEGMENT],
            "startup_segment_payload_bytes": startup_bytes,
            "startup_segment_payload_gets": startup_gets,
        }
    finally:
        backend.close()
    _write_report(report_path, report)


def audit(database: Path, report_path: Path) -> None:
    """新进程以稳定 continuation 全量流式重算内容 digest。"""
    backend = SQLiteBackend(str(database))
    digest = hashlib.sha256()
    try:
        repository = CountingBackendObjectRepository(backend)
        store = _store(repository)
        startup_gets = repository.payload_gets[OBJECT_KIND_SEGMENT]
        startup_bytes = repository.payload_bytes[OBJECT_KIND_SEGMENT]
        repository.reset_metrics()
        reader = store.open_reader((20260728, 216, 1), _DESCRIPTOR)
        continuation = None
        count = 0
        pages = 0
        max_page_records = 0
        try:
            while True:
                page = reader.page(
                    budget=SegmentBudget(AUDIT_PAGE_OBJECT_LIMIT, BYTE_LIMIT),
                    continuation=continuation,
                )
                pages += 1
                max_page_records = max(max_page_records, len(page.records))
                for record in page.records:
                    _update_digest(digest, record)
                    count += 1
                if not page.has_more:
                    break
                continuation = page.continuation
        finally:
            reader.close()
        report = {
            "active_readers_after": len(store.reader_epochs.snapshot()),
            "active_write_intents": store.active_write_intent_count(),
            "audit_segment_payload_bytes": repository.payload_bytes[
                OBJECT_KIND_SEGMENT],
            "audit_segment_payload_gets": repository.payload_gets[
                OBJECT_KIND_SEGMENT],
            "content_sha256": digest.hexdigest(),
            "max_page_records": max_page_records,
            "mode": "AUDIT",
            "page_count": pages,
            "record_count": count,
            "startup_segment_payload_bytes": startup_bytes,
            "startup_segment_payload_gets": startup_gets,
        }
    finally:
        backend.close()
    _write_report(report_path, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "query", "audit"))
    parser.add_argument("database", type=Path)
    parser.add_argument("report", type=Path)
    arguments = parser.parse_args()
    if arguments.mode == "build":
        build(arguments.database, arguments.report)
    elif arguments.mode == "query":
        query(arguments.database, arguments.report)
    else:
        audit(arguments.database, arguments.report)


if __name__ == "__main__":
    main()
