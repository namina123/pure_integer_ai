"""PERF-P0 独立子进程 workload，输出只含整数的 canonical 指标。"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import sys
import time
import tracemalloc

try:
    import resource
except ImportError:  # pragma: no cover - Windows 运行时没有 resource
    resource = None

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.experiments.long_generation_checkpoint import (
    LongGenerationCheckpointStore,
    LongGenerationPlan,
    LongGenerationPlanItem,
    OBJECT_KIND_LONG_GENERATION_CHECKPOINT,
)
from pure_integer_ai.experiments.long_input_hierarchy import (
    LongInputChunk,
    LongInputHierarchyBuilder,
    LongInputHierarchySeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    StableRecordKey,
    canonical_json_bytes,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.integer_codec import encode_integer_tuple
from pure_integer_ai.storage.memory_query_projection import (
    MEMORY_QUERY_PROJECTION_DESCRIPTOR,
)
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.segment_dependency import SegmentDependency
from pure_integer_ai.storage.segment_repository import (
    BackendObjectRepository,
    InMemoryObjectRepository,
)
from pure_integer_ai.storage.sealed_segment import (
    OpenHotDelta,
    SegmentBudget,
    SegmentRecord,
)
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_MEMORY,
    SpaceIdentity,
)
from pure_integer_ai.storage.placement import TemperatureProfile, TemperatureTier
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore
from pure_integer_ai.storage import build_storage_role_registry


_BASE = 98000
_DESCRIPTOR = MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY
_HOT = (20260807, 1, 1)
_COLD = (20260807, 1, 2)
_PROFILE = TemperatureProfile(
    (20260807, 1, 3),
    (TemperatureTier(_HOT, 0), TemperatureTier(_COLD, 1)),
)


def _peak_rss_bytes() -> int:
    """返回当前独立 workload 进程的峰值驻留内存。"""
    if os.name == "nt":
        class _Counters(ctypes.Structure):
            _fields_ = (
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            )
        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        reader = ctypes.windll.psapi.GetProcessMemoryInfo
        reader.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Counters),
            ctypes.c_ulong,
        ]
        reader.restype = ctypes.c_int
        ok = reader(
            process, ctypes.byref(counters), counters.cb)
        if ok:
            return int(counters.peak_working_set_size)
        return 0
    if resource is None:
        return 0
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1024 if sys.platform != "darwin" else 1))


def _digest(value: object) -> str:
    """对只含稳定 JSON 值的结果摘要计算 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _metrics(
        scenario: str,
        scale: int,
        started_ns: int,
        stable: object,
        metrics: dict[str, object],
        ) -> dict[str, object]:
    """补齐 workload 的统一整数指标和稳定摘要。"""
    current, peak = tracemalloc.get_traced_memory()
    return {
        "contract": "PURE_INTEGER_AI_PERFORMANCE_WORKLOAD_V1",
        "scenario": scenario,
        "scale": scale,
        "duration_ns": time.perf_counter_ns() - started_ns,
        "peak_rss_bytes": _peak_rss_bytes(),
        "tracemalloc_current_bytes": current,
        "tracemalloc_peak_bytes": peak,
        "stable_digest": _digest(stable),
        "metrics": metrics,
    }


def _run_long_input(scale: int, started_ns: int) -> dict[str, object]:
    """测量长文本绝对 offset 分块、层级重组和摘要生成。"""
    record_count = max(1, scale)
    surface = "甲事实。"
    text = surface * record_count
    source = SourceRef(
        SOURCE_BARE_TEXT,
        _BASE + 1,
        1,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(1)),
    )
    scope = document_scope(source)
    seeds = []
    width = len(surface)
    for ordinal in range(record_count):
        start = ordinal * width
        end = start + width
        seeds.append(LongInputHierarchySeed(
            proposition_identity(source, (_BASE + 10, ordinal + 1)),
            context_scope_identity(source, (_BASE + 20, ordinal + 1)),
            ordinal,
            ((0, len(text)),),
            ((0, len(text)),),
            ((start, end),),
            ((start, end),),
            1,
            1,
            ordinal + 1,
            ordinal + 1,
        ))
    chunk_width = max(width, 97)
    chunks = tuple(
        LongInputChunk.from_text(source, start, text[start:start + chunk_width])
        for start in range(0, len(text), chunk_width)
    )
    hierarchy = LongInputHierarchyBuilder().build(
        tuple(reversed(chunks)), scope, tuple(seeds))
    return _metrics(
        "long_input_hierarchy", scale, started_ns, hierarchy.stable_key(), {
            "chunk_count": len(chunks),
            "record_count": len(hierarchy.records),
            "text_length": hierarchy.text_length,
        })


def _run_long_session(scale: int, started_ns: int) -> dict[str, object]:
    """测量长会话 continuation plan 的 metadata 创建和跨对象恢复。"""
    item_count = max(1, scale)
    answer_key = StableRecordKey((20260807, 2, 1))
    plan = LongGenerationPlan(
        answer_key,
        tuple(
            LongGenerationPlanItem(
                StableRecordKey((20260807, 2, ordinal + 2)),
                (20260807, 3, ordinal + 1),
            )
            for ordinal in range(item_count)
        ),
    )
    repository = InMemoryObjectRepository()
    store = LongGenerationCheckpointStore(repository, lambda: None)
    created = store.create(plan)
    loaded = store.load(answer_key)
    stable = {
        "created": created.stable_key(),
        "loaded": loaded.stable_key(),
        "plan": plan.stable_key(),
    }
    return _metrics(
        "long_session_checkpoint", scale, started_ns, stable, {
            "item_count": item_count,
            "checkpoint_revision": loaded.revision,
            "repository_object_count": len(
                repository.list_kind(OBJECT_KIND_LONG_GENERATION_CHECKPOINT)),
        })


def _run_long_memory(scale: int, started_ns: int) -> dict[str, object]:
    """测量长期 Memory 候选投影 manifest 的整数编码和恢复。"""
    segment_count = max(1, scale)
    dependencies = tuple(
        SegmentDependency(key, (1, 1), (1, 2, 3))
        for key in MEMORY_QUERY_PROJECTION_DESCRIPTOR.dependency_keys
    )
    from pure_integer_ai.experiments.memory_hot_set_runtime import (
        MemoryCandidateProjectionManifest,
        MemoryProjectionSegment,
    )
    segments = tuple(
        MemoryProjectionSegment(
            (20260807, 4, ordinal),
            (1, ordinal * 100 + 1),
            (1, ordinal * 100 + 100),
            (1, 2, 3),
            100,
            3200,
        )
        for ordinal in range(segment_count)
    )
    manifest = MemoryCandidateProjectionManifest(
        (20260807, 4, 0),
        SpaceIdentity(SPACE_TYPE_MEMORY, 11, 12),
        1,
        MemoryAccessContext(1, 1, 1),
        ((1, 1),),
        segment_count * 100,
        (segment_count * 100, 1, 2),
        (1, 1),
        dependencies,
        segments,
        1,
    )
    restored = MemoryCandidateProjectionManifest.from_stable_key(
        manifest.stable_key())
    return _metrics(
        "long_memory_projection", scale, started_ns, restored.stable_key(), {
            "segment_count": len(restored.segments),
            "record_count": restored.record_count,
            "stable_key_int_count": len(restored.stable_key()),
        })


def _record(value: int) -> SegmentRecord:
    """构造只用于 P0 存储 profile 的确定整数记录。"""
    return SegmentRecord((1, value), (value, value + 1, value % 997 + 1))


def _query(store: TieredSegmentStore, targets: tuple[int, ...]) -> str:
    """对固定键执行精确 reader 并返回结果摘要。"""
    digest = hashlib.sha256()
    for ordinal, target in enumerate(targets, start=1):
        reader = store.open_reader((20260807, 5, ordinal), _DESCRIPTOR)
        try:
            page = reader.page(
                budget=SegmentBudget(1, 1_000_000),
                lower_key=(1, target), upper_key=(1, target),
            )
        finally:
            reader.close()
        if tuple(item.record_key for item in page.records) != ((1, target),):
            raise RuntimeError("P0 storage exact query 结果漂移")
        digest.update(encode_integer_tuple((target, len(page.records))))
    return digest.hexdigest()


def _run_storage(
        scale: int,
        started_ns: int,
        backend_name: str,
        database: Path | None,
        ) -> dict[str, object]:
    """测量 Dict/SQLite segment 发布、冷读和热读，不写 Core。"""
    record_count = max(1, scale)
    per_segment = max(1, min(250, record_count))
    segment_count = (record_count + per_segment - 1) // per_segment
    targets = tuple(
        sorted({1, max(1, record_count // 2), record_count})
    )
    if backend_name == "sqlite":
        if database is None:
            raise ValueError("SQLite profile 缺 database 路径")
        if database.exists():
            raise ValueError("SQLite profile database 已存在")
        backend = SQLiteBackend(str(database))
        repository = BackendObjectRepository(backend)
    elif backend_name == "dict":
        backend = None
        repository = InMemoryObjectRepository()
    else:
        raise ValueError("未知 storage backend")
    store = TieredSegmentStore(
        repository, build_storage_role_registry(), _PROFILE)
    digest = hashlib.sha256()
    build_started = time.perf_counter_ns()
    try:
        for ordinal in range(segment_count):
            start = ordinal * per_segment + 1
            stop = min(record_count, start + per_segment - 1)
            delta = OpenHotDelta(
                _DESCRIPTOR,
                (20260807, 6, 1),
                (),
                SegmentBudget(stop - start + 1, 1_000_000),
            )
            for value in range(start, stop + 1):
                record = _record(value)
                delta.append(record)
                digest.update(encode_integer_tuple(
                    (*record.record_key, *record.payload)))
            sealed = delta.seal((20260807, 7, ordinal + 1), stop)
            store.publish_delta(
                delta,
                segment_key=sealed.segment_key,
                tier_key=_COLD,
                read_fence=stop,
                manifest_key=(20260807, 8, ordinal + 1),
                migration_key=(20260807, 9, ordinal + 1),
            )
        build_duration = time.perf_counter_ns() - build_started
        if backend is not None:
            backend.commit()
        if backend_name == "sqlite":
            backend.close()
            backend = SQLiteBackend(str(database))
            repository = BackendObjectRepository(backend)
            store = TieredSegmentStore(
                repository, build_storage_role_registry(), _PROFILE)
        cold_started = time.perf_counter_ns()
        cold_digest = _query(store, targets)
        cold_duration = time.perf_counter_ns() - cold_started
        warm_started = time.perf_counter_ns()
        warm_digest = _query(store, targets)
        warm_duration = time.perf_counter_ns() - warm_started
        if cold_digest != warm_digest:
            raise RuntimeError("P0 cold/warm query digest 漂移")
        stable = {
            "content_digest": digest.hexdigest(),
            "cold_query_digest": cold_digest,
            "record_count": record_count,
            "segment_count": segment_count,
        }
        disk_bytes = 0
        database_file_count = 0
        if database is not None and database.exists():
            database_files = tuple(
                path for path in (
                    database,
                    Path(str(database) + "-wal"),
                    Path(str(database) + "-shm"),
                )
                if path.exists()
            )
            disk_bytes = sum(path.stat().st_size for path in database_files)
            database_file_count = len(database_files)
        return _metrics(
            f"storage_{backend_name}", scale, started_ns, stable, {
                "backend": backend_name,
                "build_duration_ns": build_duration,
                "cold_query_duration_ns": cold_duration,
                "warm_query_duration_ns": warm_duration,
                "record_count": record_count,
                "segment_count": segment_count,
                "disk_bytes": disk_bytes,
                "database_file_count": database_file_count,
                "query_count": len(targets),
            })
    finally:
        if backend is not None:
            backend.close()


def run_scenario(
        scenario: str, scale: int, database: Path | None,
        ) -> dict[str, object]:
    """按固定场景名执行一次独立 workload。"""
    started_ns = time.perf_counter_ns()
    tracemalloc.start()
    if scenario == "long_input_hierarchy":
        result = _run_long_input(scale, started_ns)
    elif scenario == "long_session_checkpoint":
        result = _run_long_session(scale, started_ns)
    elif scenario == "long_memory_projection":
        result = _run_long_memory(scale, started_ns)
    elif scenario == "storage_dict":
        result = _run_storage(scale, started_ns, "dict", database)
    elif scenario == "storage_sqlite":
        result = _run_storage(scale, started_ns, "sqlite", database)
    else:
        raise ValueError(f"未知性能 workload: {scenario}")
    tracemalloc.stop()
    return result


def main() -> None:
    """解析一个有界 workload 命令并输出一条 canonical JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--database", type=Path)
    arguments = parser.parse_args()
    result = run_scenario(
        arguments.scenario, arguments.scale, arguments.database)
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")


if __name__ == "__main__":
    main()
