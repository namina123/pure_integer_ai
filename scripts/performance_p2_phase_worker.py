"""PERF-P2 独立阶段 profiler；只产生整数诊断，不发布 readiness。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Callable

_BASELINE_IMPORT_STARTED_NS = time.perf_counter_ns()
from scripts import performance_baseline_worker as baseline
_BASELINE_IMPORT_DURATION_NS = (
    time.perf_counter_ns() - _BASELINE_IMPORT_STARTED_NS
)

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


CONTRACT = "PURE_INTEGER_AI_PERFORMANCE_PHASE_PROFILE_V1"
SCHEMA_VERSION = 1
SCENARIOS = (
    "long_memory_projection",
    "storage_sqlite",
)
READINESS_TRANSITION = {
    "LANGUAGE_READINESS_REPUBLISHED": 0,
    "PW00A_STARTED": 0,
}
_PROFILE_ALREADY_RUN = False


def _record_phase(
        phases: dict[str, dict[str, int]],
        name: str,
        duration_ns: int,
        ) -> None:
    current = phases.setdefault(name, {"call_count": 0, "duration_ns": 0})
    current["call_count"] += 1
    current["duration_ns"] += duration_ns


def _timed_call(
        phases: dict[str, dict[str, int]],
        name: str,
        function: Callable[..., Any],
        *args: object,
        **kwargs: object,
        ) -> Any:
    started_ns = time.perf_counter_ns()
    try:
        return function(*args, **kwargs)
    finally:
        _record_phase(phases, name, time.perf_counter_ns() - started_ns)


def _wrap_method(
        owner: object,
        method_name: str,
        phase_name: str,
        phases: dict[str, dict[str, int]],
        ) -> None:
    original = getattr(owner, method_name)

    def timed(*args: object, **kwargs: object) -> Any:
        return _timed_call(phases, phase_name, original, *args, **kwargs)

    setattr(owner, method_name, timed)


def _install_sqlite_timers(
        phases: dict[str, dict[str, int]],
        ) -> None:
    for method_name, phase_name in (
            ("__init__", "sqlite_backend_init"),
            ("_do_create_table", "sqlite_create_table"),
            ("_do_ensure_index", "sqlite_ensure_index"),
            ("_do_insert", "sqlite_insert"),
            ("_do_update", "sqlite_update"),
            ("_do_select", "sqlite_select"),
            ("_do_count", "sqlite_count"),
            ("_do_delete", "sqlite_delete"),
            ("commit", "sqlite_commit"),
            ("close", "sqlite_close")):
        _wrap_method(
            baseline.SQLiteBackend, method_name, phase_name, phases)
    for owner, method_name, phase_name in (
            (baseline.BackendObjectRepository, "__init__", "repository_init"),
            (baseline.TieredSegmentStore, "__init__", "store_init"),
            (baseline.TieredSegmentStore, "publish_delta", "publish_delta"),
            (baseline.TieredSegmentStore, "publish_segment", "publish_segment"),
            (baseline.OpenHotDelta, "seal", "seal_delta")):
        _wrap_method(owner, method_name, phase_name, phases)
    original_query = baseline._query

    def timed_query(*args: object, **kwargs: object) -> Any:
        return _timed_call(
            phases, "query_total", original_query, *args, **kwargs)

    baseline._query = timed_query


def _profile_sqlite(
        scale: int,
        database: Path,
        phases: dict[str, dict[str, int]],
        ) -> dict[str, object]:
    if database.exists():
        raise ValueError("P2 SQLite profile database 已存在")
    _install_sqlite_timers(phases)
    return baseline.run_scenario("storage_sqlite", scale, database)


def _profile_long_memory(
        scale: int,
        phases: dict[str, dict[str, int]],
        ) -> dict[str, object]:
    started_ns = time.perf_counter_ns()
    tracemalloc.start()
    import_started_ns = time.perf_counter_ns()
    from pure_integer_ai.experiments.memory_hot_set_runtime import (
        MemoryCandidateProjectionManifest,
        MemoryProjectionSegment,
    )
    _record_phase(
        phases,
        "memory_runtime_import",
        time.perf_counter_ns() - import_started_ns,
    )
    segment_count = max(1, scale)
    phase_started_ns = time.perf_counter_ns()
    dependencies = tuple(
        baseline.SegmentDependency(key, (1, 1), (1, 2, 3))
        for key in baseline.MEMORY_QUERY_PROJECTION_DESCRIPTOR.dependency_keys
    )
    _record_phase(
        phases,
        "memory_dependencies",
        time.perf_counter_ns() - phase_started_ns,
    )
    phase_started_ns = time.perf_counter_ns()
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
    _record_phase(
        phases,
        "memory_segment_construction",
        time.perf_counter_ns() - phase_started_ns,
    )
    phase_started_ns = time.perf_counter_ns()
    manifest = MemoryCandidateProjectionManifest(
        (20260807, 4, 0),
        baseline.SpaceIdentity(baseline.SPACE_TYPE_MEMORY, 11, 12),
        1,
        baseline.MemoryAccessContext(1, 1, 1),
        ((1, 1),),
        segment_count * 100,
        (segment_count * 100, 1, 2),
        (1, 1),
        dependencies,
        segments,
        1,
    )
    _record_phase(
        phases,
        "memory_manifest_construction",
        time.perf_counter_ns() - phase_started_ns,
    )
    phase_started_ns = time.perf_counter_ns()
    stable_key = manifest.stable_key()
    _record_phase(
        phases,
        "memory_stable_key_encode",
        time.perf_counter_ns() - phase_started_ns,
    )
    phase_started_ns = time.perf_counter_ns()
    restored = MemoryCandidateProjectionManifest.from_stable_key(stable_key)
    _record_phase(
        phases,
        "memory_stable_key_restore",
        time.perf_counter_ns() - phase_started_ns,
    )
    phase_started_ns = time.perf_counter_ns()
    restored_key = restored.stable_key()
    _record_phase(
        phases,
        "memory_restored_key_encode",
        time.perf_counter_ns() - phase_started_ns,
    )
    result = baseline._metrics(
        "long_memory_projection", scale, started_ns, restored_key, {
            "segment_count": len(restored.segments),
            "record_count": restored.record_count,
            "stable_key_int_count": len(restored_key),
        })
    tracemalloc.stop()
    return result


def run_profile(
        scenario: str,
        scale: int,
        database: Path | None = None,
        ) -> dict[str, object]:
    global _PROFILE_ALREADY_RUN
    if _PROFILE_ALREADY_RUN:
        raise RuntimeError("P2 phase profile 每个进程只允许运行一次")
    if scenario not in SCENARIOS:
        raise ValueError("P2 phase profile 场景未注册")
    if type(scale) is not int or scale < 1:
        raise ValueError("P2 phase profile scale 必须是正严格整数")
    _PROFILE_ALREADY_RUN = True
    phases: dict[str, dict[str, int]] = {}
    profile_started_ns = time.perf_counter_ns()
    if scenario == "storage_sqlite":
        if database is None:
            raise ValueError("P2 SQLite phase profile 缺少 database")
        report = _profile_sqlite(scale, database, phases)
    else:
        if database is not None:
            raise ValueError("P2 memory phase profile 不接受 database")
        report = _profile_long_memory(scale, phases)
    profile_duration_ns = time.perf_counter_ns() - profile_started_ns
    return {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario,
        "scale": scale,
        "baseline_import_duration_ns": _BASELINE_IMPORT_DURATION_NS,
        "profile_duration_ns": profile_duration_ns,
        "phase_semantics": "OVERLAPPING_CALL_AGGREGATES",
        "phases": {name: phases[name] for name in sorted(phases)},
        "baseline_report": report,
        "readiness_transition": dict(READINESS_TRANSITION),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 PERF-P2 整数阶段 profile。")
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--database", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    arguments = _build_parser().parse_args(argv)
    try:
        result = run_profile(
            arguments.scenario,
            arguments.scale,
            arguments.database,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"performance_p2_phase_worker: ERROR: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTRACT", "READINESS_TRANSITION", "SCENARIOS", "SCHEMA_VERSION",
    "run_profile",
]
