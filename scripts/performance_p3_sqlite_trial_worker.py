"""PERF-P3 仅基准使用的 SQLite schema、初始化与读取试验。

该 worker 与生产后端实现保持分离，只测量既有 append-only repository 合同，
并验证读取路径能够经受回收、重启和写入中断。
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time
from typing import Any, Callable

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.integer_codec import encode_integer_tuple
from pure_integer_ai.storage.segment_repository import (
    FAULT_OBJECT_AFTER_PART,
    OBJECT_KIND_SEGMENT,
    BackendObjectRepository,
    SegmentRepositoryError,
    register_segment_repository_tables,
)


CONTRACT = "PURE_INTEGER_AI_PERFORMANCE_P3_SQLITE_TRIAL_V1"
SCHEMA_VERSION = 1
READINESS_TRANSITION = {
    "LANGUAGE_READINESS_REPUBLISHED": 0,
    "PW00A_STARTED": 0,
}
_BASE_KEY = (20260807, 3, 1)
_FAULT_KEY = (20260807, 3, 900000)


class _InjectedFailure(RuntimeError):
    """只用于触发写入中断恢复路径的注入故障。"""


class _FailAfterPart:
    def hit(self, point: int, context: dict[str, Any]) -> None:
        """在 part 已落盘而对象尚未发布时注入一次故障。"""
        if point == FAULT_OBJECT_AFTER_PART:
            raise _InjectedFailure("trial injected failure after part")


def _digest(value: object) -> str:
    """返回规范 JSON 数据的 SHA-256 摘要。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _record_phase(
        phases: dict[str, dict[str, int]],
        name: str,
        duration_ns: int,
        ) -> None:
    """把一次严格整数纳秒耗时累计到指定阶段。"""
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
    """调用目标函数并在成功或异常时都记录阶段耗时。"""
    started_ns = time.perf_counter_ns()
    try:
        return function(*args, **kwargs)
    finally:
        _record_phase(phases, name, time.perf_counter_ns() - started_ns)


def _install_backend_timers(
        backend: SQLiteBackend,
        phases: dict[str, dict[str, int]],
        ) -> None:
    """包装当前后端实例的计时边界，不修改生产实现。"""
    original_create = backend._do_create_table
    original_index = backend._do_ensure_index
    original_commit = backend.commit

    def timed_create(table: str, columns: list[tuple[str, str]]) -> None:
        _timed_call(phases, "sqlite_table_registration", original_create,
                    table, columns)

    def timed_index(
            table: str,
            columns: list[str] | tuple[str, ...],
            *,
            defer_indexes: bool = False,
            ) -> None:
        _timed_call(
            phases,
            "sqlite_index_registration",
            original_index,
            table,
            columns,
            defer_indexes=defer_indexes,
        )

    def timed_commit() -> None:
        _timed_call(phases, "sqlite_commit", original_commit)

    backend._do_create_table = timed_create
    backend._do_ensure_index = timed_index
    backend.commit = timed_commit


def _install_repository_read_timer(
        repository: BackendObjectRepository,
        phases: dict[str, dict[str, int]],
        ) -> None:
    """只在当前 repository 实例上安装读取阶段计时。"""
    original_read = repository._read_object_id

    def timed_read(object_id: int):
        return _timed_call(phases, "segment_decode", original_read, object_id)

    repository._read_object_id = timed_read


def _payload(ordinal: int) -> bytes:
    """形成一个可按序号确定重建的规范整数 payload。"""
    return encode_integer_tuple((_BASE_KEY[0], _BASE_KEY[1], ordinal, ordinal + 1))


def _identities(scale: int) -> tuple[tuple[int, ...], ...]:
    """返回当前规模下稳定有序的对象身份集合。"""
    return tuple((*_BASE_KEY, ordinal) for ordinal in range(1, scale + 1))


def _query_keys(scale: int) -> tuple[tuple[int, ...], ...]:
    """选择首、中、尾三个稳定查询键并去重。"""
    values = tuple(sorted({1, max(1, scale // 2), scale}))
    return tuple((*_BASE_KEY, value) for value in values)


def _visible_digest(repository: BackendObjectRepository) -> str:
    """摘要当前可见段对象的身份、校验值和尺寸。"""
    descriptors = repository.list_kind(OBJECT_KIND_SEGMENT)
    return _digest(tuple(
        (item.object_id, item.identity_key, item.checksum_key, item.size_bytes)
        for item in descriptors
    ))


def _query(
        repository: BackendObjectRepository,
        keys: tuple[tuple[int, ...], ...],
        phases: dict[str, dict[str, int]],
        phase_name: str,
        ) -> str:
    """读取固定键集合并返回与调用顺序无关的稳定结果摘要。"""
    values = []
    for key in keys:
        payload = _timed_call(phases, phase_name, repository.get,
                              OBJECT_KIND_SEGMENT, key)
        values.append((key, tuple(payload)))
    return _digest(tuple(values))


def _database_bytes(database: Path) -> tuple[int, int]:
    """汇总 SQLite 主文件及现存 WAL/SHM 文件的物理尺寸。"""
    files = tuple(
        path for path in (
            database,
            Path(str(database) + "-wal"),
            Path(str(database) + "-shm"),
        ) if path.exists()
    )
    return sum(path.stat().st_size for path in files), len(files)


def run_trial(scale: int, database: Path) -> dict[str, object]:
    """在全新 SQLite 文件上运行一次确定、有界且可恢复的试验。"""
    if type(scale) is not int or scale < 1:
        raise ValueError("P3 scale must be a positive strict integer")
    if database.exists():
        raise ValueError("P3 database already exists")
    phases: dict[str, dict[str, int]] = {}
    started_ns = time.perf_counter_ns()

    backend = SQLiteBackend(str(database))
    _install_backend_timers(backend, phases)
    schema_started_ns = time.perf_counter_ns()
    register_segment_repository_tables(backend)
    backend.commit()
    schema_duration_ns = time.perf_counter_ns() - schema_started_ns
    schema = backend.schema_snapshot()

    repository_started_ns = time.perf_counter_ns()
    repository = BackendObjectRepository(backend)
    repository_init_duration_ns = time.perf_counter_ns() - repository_started_ns

    publish_started_ns = time.perf_counter_ns()
    for identity in _identities(scale):
        repository.put(OBJECT_KIND_SEGMENT, identity, _payload(identity[-1]))
    publish_duration_ns = time.perf_counter_ns() - publish_started_ns
    publish_digest = _visible_digest(repository)
    expected_identities = _identities(scale)
    if tuple(item.identity_key for item in repository.list_kind(
            OBJECT_KIND_SEGMENT)) != expected_identities:
        raise RuntimeError("P3 publish identity order drift")
    backend.close()

    reopen_started_ns = time.perf_counter_ns()
    reopened_backend = SQLiteBackend(str(database))
    reopen_duration_ns = time.perf_counter_ns() - reopen_started_ns
    repository_reopen_started_ns = time.perf_counter_ns()
    reopened = BackendObjectRepository(reopened_backend)
    repository_reopen_duration_ns = (
        time.perf_counter_ns() - repository_reopen_started_ns)
    _install_repository_read_timer(reopened, phases)
    if _visible_digest(reopened) != publish_digest:
        raise RuntimeError("P3 restart visible digest drift")

    keys = _query_keys(scale)
    cold_query_digest = _query(reopened, keys, phases, "query_read_cold")
    warm_query_digest = _query(reopened, keys, phases, "query_read_warm")
    if cold_query_digest != warm_query_digest:
        raise RuntimeError("P3 cold/warm query digest drift")

    rollback_key = _identities(scale)[-1]
    rollback_started_ns = time.perf_counter_ns()
    if not reopened.reclaim(OBJECT_KIND_SEGMENT, rollback_key):
        raise RuntimeError("P3 expected reclaim target missing")
    rollback_duration_ns = time.perf_counter_ns() - rollback_started_ns
    try:
        reopened.get(OBJECT_KIND_SEGMENT, rollback_key)
    except KeyError:
        pass
    else:
        raise RuntimeError("P3 reclaimed object remained visible")
    rollback_digest = _visible_digest(reopened)
    reopened_backend.close()

    recovered_backend = SQLiteBackend(str(database))
    recovered = BackendObjectRepository(recovered_backend)
    if _visible_digest(recovered) != rollback_digest:
        raise RuntimeError("P3 rollback restart digest drift")
    try:
        recovered.get(OBJECT_KIND_SEGMENT, rollback_key)
    except KeyError:
        pass
    else:
        raise RuntimeError("P3 reclaimed object reappeared after restart")

    fault_key = _FAULT_KEY
    try:
        recovered.put(
            OBJECT_KIND_SEGMENT,
            fault_key,
            _payload(900000),
            fault_injector=_FailAfterPart(),
        )
    except _InjectedFailure:
        pass
    else:
        raise RuntimeError("P3 fault injector did not interrupt write")
    recovered_backend.close()

    fault_recovered_backend = SQLiteBackend(str(database))
    fault_recovered = BackendObjectRepository(fault_recovered_backend)
    if _visible_digest(fault_recovered) != rollback_digest:
        raise RuntimeError("P3 interrupted-write restart digest drift")
    try:
        fault_recovered.get(OBJECT_KIND_SEGMENT, fault_key)
    except KeyError:
        pass
    else:
        raise RuntimeError("P3 interrupted object became visible")
    final_visible = _visible_digest(fault_recovered)
    disk_bytes, database_file_count = _database_bytes(database)
    fault_recovered_backend.close()

    stable = {
        "schema_digest": _digest(schema),
        "publish_digest": publish_digest,
        "cold_query_digest": cold_query_digest,
        "warm_query_digest": warm_query_digest,
        "rollback_digest": rollback_digest,
        "final_visible_digest": final_visible,
        "rollback_restart_equivalent": int(final_visible == rollback_digest),
        "object_count_after_rollback": scale - 1,
    }
    return {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "scenario": "storage_sqlite_schema_init_read_trial",
        "scale": scale,
        "duration_ns": time.perf_counter_ns() - started_ns,
        "phases": {name: phases[name] for name in sorted(phases)},
        "metrics": {
            "schema_table_count": len(schema),
            "schema_index_count": sum(
                len(item["indexes"]) for item in schema.values()),
            "schema_duration_ns": schema_duration_ns,
            "repository_init_duration_ns": repository_init_duration_ns,
            "publish_duration_ns": publish_duration_ns,
            "reopen_duration_ns": reopen_duration_ns,
            "repository_reopen_duration_ns": repository_reopen_duration_ns,
            "cold_query_duration_ns": phases["query_read_cold"]["duration_ns"],
            "warm_query_duration_ns": phases["query_read_warm"]["duration_ns"],
            "rollback_duration_ns": rollback_duration_ns,
            "disk_bytes": disk_bytes,
            "database_file_count": database_file_count,
            "query_count": len(keys),
            "visible_object_count": scale - 1,
            "exception_path_verified": 1,
        },
        "stable": stable,
        "readiness_transition": dict(READINESS_TRANSITION),
        "phase_semantics": "OVERLAPPING_CALL_AGGREGATES",
    }


def main(argv: list[str] | None = None) -> int:
    """解析命令行并向 stdout 写出一份规范试验报告。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--database", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        report = run_trial(arguments.scale, arguments.database)
    except (OSError, RuntimeError, TypeError, ValueError, SegmentRepositoryError) as error:
        print(f"performance_p3_sqlite_trial_worker: ERROR: {error}",
              file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CONTRACT", "READINESS_TRANSITION", "SCHEMA_VERSION", "run_trial"]
