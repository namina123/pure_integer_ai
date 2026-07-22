"""正式训练的编排计数、外部时钟适配和独立遥测文件。"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.telemetry import (
    BackendTelemetryCollector,
    active_backend_telemetry,
    collect_backend_telemetry,
    current_telemetry_scope,
    suppress_backend_telemetry,
    telemetry_scope,
)


OperationSnapshot = dict[tuple[str, str], tuple[int, int, int]]


@dataclass(frozen=True)
class ExecutionSnapshot:
    """阶段边界的后端计数、表规模和工作集采样。"""

    operations: OperationSnapshot = field(default_factory=dict)
    table_sizes: dict[str, int] = field(default_factory=dict)
    working_set_bytes: int = 0


@dataclass
class ExecutionPhaseStats:
    """单个训练阶段或编排 phase 的可差分诊断。"""

    name: str
    stage: int | None = None
    elapsed_ns: int = 0
    item_count: int = 0
    candidate_count: int = 0
    dump_calls: int = 0
    working_set_before_bytes: int = 0
    working_set_after_bytes: int = 0
    peak_working_set_bytes: int = 0
    backend_operations: list[dict[str, Any]] = field(default_factory=list)
    table_growth: list[dict[str, int | str]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """导出稳定的纯整数阶段指标。"""
        return {
            "name": self.name,
            "stage": self.stage,
            "elapsed_ns": self.elapsed_ns,
            "item_count": self.item_count,
            "candidate_count": self.candidate_count,
            "dump_calls": self.dump_calls,
            "working_set_before_bytes": self.working_set_before_bytes,
            "working_set_after_bytes": self.working_set_after_bytes,
            "peak_working_set_bytes": self.peak_working_set_bytes,
            "backend_operations": self.backend_operations,
            "table_growth": self.table_growth,
        }


@dataclass
class FormalTrainExecutionStats:
    """供机器读取的编排计数与整数纳秒耗时。"""

    formal_train_calls: int = 1
    input_items: int = 0
    training_items: int = 0
    probe_items: int = 0
    active_relations: tuple[str, ...] | None = None
    boot_relations: tuple[str, ...] | None = None
    training_stages: tuple[int, ...] = ()
    stage_batch_calls: int = 0
    stage_item_runs: int = 0
    h2_item_runs: int = 0
    preflight_item_runs: int = 0
    graph_dump_calls: int = 0
    bootstrap_elapsed_ns: int = 0
    discovery_elapsed_ns: int = 0
    stage_loop_elapsed_ns: int = 0
    finalize_elapsed_ns: int = 0
    total_elapsed_ns: int = 0
    peak_working_set_bytes: int = 0
    phases: list[ExecutionPhaseStats] = field(default_factory=list)
    stages: list[ExecutionPhaseStats] = field(default_factory=list)
    run_table_growth: list[dict[str, int | str]] = field(default_factory=list)
    backend_telemetry: dict[str, Any] | None = None

    def record_phase(self, stats: ExecutionPhaseStats) -> None:
        """追加一个编排 phase，并提升全 run 峰值工作集。"""
        self.phases.append(stats)
        self.peak_working_set_bytes = max(
            self.peak_working_set_bytes,
            stats.peak_working_set_bytes,
        )

    def record_stage(self, stats: ExecutionPhaseStats) -> None:
        """追加一个训练阶段，并提升全 run 峰值工作集。"""
        if stats.stage is None:
            raise ValueError("训练阶段遥测必须提供 stage")
        self.stages.append(stats)
        self.peak_working_set_bytes = max(
            self.peak_working_set_bytes,
            stats.peak_working_set_bytes,
        )

    def to_json(self) -> dict[str, Any]:
        """导出 JSON 兼容的稳定字段集合。"""
        return {
            "formal_train_calls": self.formal_train_calls,
            "input_items": self.input_items,
            "training_items": self.training_items,
            "probe_items": self.probe_items,
            "active_relations": (None if self.active_relations is None
                                 else list(self.active_relations)),
            "boot_relations": (None if self.boot_relations is None
                               else list(self.boot_relations)),
            "training_stages": list(self.training_stages),
            "stage_batch_calls": self.stage_batch_calls,
            "stage_item_runs": self.stage_item_runs,
            "h2_item_runs": self.h2_item_runs,
            "preflight_item_runs": self.preflight_item_runs,
            "graph_dump_calls": self.graph_dump_calls,
            "bootstrap_elapsed_ns": self.bootstrap_elapsed_ns,
            "discovery_elapsed_ns": self.discovery_elapsed_ns,
            "stage_loop_elapsed_ns": self.stage_loop_elapsed_ns,
            "finalize_elapsed_ns": self.finalize_elapsed_ns,
            "total_elapsed_ns": self.total_elapsed_ns,
            "peak_working_set_bytes": self.peak_working_set_bytes,
            "phases": [phase.to_json() for phase in self.phases],
            "stages": [stage.to_json() for stage in self.stages],
            "run_table_growth": self.run_table_growth,
            "backend_telemetry": self.backend_telemetry,
        }


@dataclass(frozen=True)
class TelemetryClock:
    """实验边界注入的整数纳秒时钟；未注入时恒返 0。"""

    source: Callable[[], int] | None = None

    def now_ns(self) -> int:
        """读取一次外部时钟并执行纯整数守卫。"""
        if self.source is None:
            return 0
        value = self.source()
        assert_int(value, _where="TelemetryClock.now_ns")
        return value


def process_working_set_bytes() -> int:
    """读取宿主进程工作集；平台不支持或查询失败时返回 0。"""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                """Windows ``PROCESS_MEMORY_COUNTERS`` 的本地结构声明。"""

                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                process,
                ctypes.byref(counters),
                counters.cb,
            )
            return int(counters.PeakWorkingSetSize) if ok else 0
        except (AttributeError, OSError, TypeError, ValueError):
            return 0
    try:
        import resource

        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return maximum if sys.platform == "darwin" else maximum * 1024
    except (ImportError, OSError, TypeError, ValueError):
        return 0


def sample_working_set(source: Callable[[], int] | None) -> int:
    """读取注入的工作集源并守严格非负整数。"""
    if source is None:
        return 0
    value = source()
    assert_int(value, _where="sample_working_set")
    if type(value) is not int or value < 0:
        raise TypeError("工作集采样必须返回非负严格整数")
    return value


def capture_execution_snapshot(
        backend: Any,
        *,
        working_set_source: Callable[[], int] | None = None,
        ) -> ExecutionSnapshot:
    """读取阶段边界，不把表规模诊断查询计入后端调用成本。"""
    collector = active_backend_telemetry()
    operations = (
        {} if collector is None else collector.operation_snapshot())
    table_sizes: dict[str, int] = {}
    tables = getattr(backend, "_tables", None)
    if isinstance(tables, dict):
        with suppress_backend_telemetry():
            for table in sorted(tables):
                table_sizes[table] = backend.count(table)
    return ExecutionSnapshot(
        operations=operations,
        table_sizes=table_sizes,
        working_set_bytes=sample_working_set(working_set_source),
    )


def backend_operation_delta(
        before: OperationSnapshot,
        after: OperationSnapshot,
        ) -> list[dict[str, Any]]:
    """计算两个后端计数快照之间的非零差分。"""
    payload: list[dict[str, Any]] = []
    for operation, table in sorted(set(before) | set(after)):
        previous = before.get((operation, table), (0, 0, 0))
        current = after.get((operation, table), (0, 0, 0))
        calls = current[0] - previous[0]
        rows = current[1] - previous[1]
        failures = current[2] - previous[2]
        if calls == 0 and rows == 0 and failures == 0:
            continue
        payload.append({
            "operation": operation,
            "table": table,
            "calls": calls,
            "rows": rows,
            "failures": failures,
        })
    return payload


def table_growth_delta(before: dict[str, int],
                       after: dict[str, int]) -> list[dict[str, int | str]]:
    """计算表行数前后值和增长量，保留零增长已注册表。"""
    return [
        {
            "table": table,
            "before": before.get(table, 0),
            "after": after.get(table, 0),
            "growth": after.get(table, 0) - before.get(table, 0),
        }
        for table in sorted(set(before) | set(after))
    ]


def complete_execution_phase(
        name: str,
        before: ExecutionSnapshot,
        after: ExecutionSnapshot,
        *,
        elapsed_ns: int,
        stage: int | None = None,
        item_count: int = 0,
        candidate_count: int = 0,
        dump_calls: int = 0,
        ) -> ExecutionPhaseStats:
    """把阶段边界快照收敛为可序列化差分。"""
    assert_int(
        elapsed_ns, item_count, candidate_count, dump_calls,
        _where="complete_execution_phase",
    )
    if min(elapsed_ns, item_count, candidate_count, dump_calls) < 0:
        raise ValueError("阶段遥测计数不能为负")
    return ExecutionPhaseStats(
        name=name,
        stage=stage,
        elapsed_ns=elapsed_ns,
        item_count=item_count,
        candidate_count=candidate_count,
        dump_calls=dump_calls,
        working_set_before_bytes=before.working_set_bytes,
        working_set_after_bytes=after.working_set_bytes,
        peak_working_set_bytes=max(
            before.working_set_bytes, after.working_set_bytes),
        backend_operations=backend_operation_delta(
            before.operations, after.operations),
        table_growth=table_growth_delta(
            before.table_sizes, after.table_sizes),
    )


class ExecutionPhaseRecorder:
    """封装阶段边界采样，使训练 facade 只声明阶段而不实现遥测算法。"""

    def __init__(
            self,
            *,
            enabled: bool,
            backend: Any,
            execution: FormalTrainExecutionStats,
            working_set_source: Callable[[], int] | None = None,
            ) -> None:
        self.enabled = enabled
        self.backend = backend
        self.execution = execution
        self.working_set_source = (
            working_set_source or process_working_set_bytes)

    def snapshot(self) -> ExecutionSnapshot | None:
        """只在显式启用时读取阶段边界。"""
        if not self.enabled:
            return None
        return capture_execution_snapshot(
            self.backend,
            working_set_source=self.working_set_source,
        )

    def finish(
            self,
            name: str,
            before: ExecutionSnapshot | None,
            *,
            elapsed_ns: int,
            stage: int | None = None,
            item_count: int = 0,
            candidate_count: int = 0,
            dump_calls: int = 0,
            ) -> None:
        """完成一个阶段差分；关闭时保持严格空操作。"""
        if before is None:
            return
        after = self.snapshot()
        if after is None:
            return
        stats = complete_execution_phase(
            name,
            before,
            after,
            elapsed_ns=elapsed_ns,
            stage=stage,
            item_count=item_count,
            candidate_count=candidate_count,
            dump_calls=dump_calls,
        )
        if stage is None:
            self.execution.record_phase(stats)
        else:
            self.execution.record_stage(stats)


def run_with_execution_telemetry(
        *,
        enabled: bool,
        backend: Any,
        operation: Callable[[], Any],
        working_set_source: Callable[[], int] | None = None,
        default_caller: str = "training",
        ) -> Any:
    """在可选采集器内执行一次编排，并把 run 级差分附到结果。"""
    if not enabled:
        return operation()
    collector = BackendTelemetryCollector()
    source = working_set_source or process_working_set_bytes
    caller = current_telemetry_scope().caller or default_caller
    with collect_backend_telemetry(collector):
        with telemetry_scope(caller=caller, query="setup"):
            before = capture_execution_snapshot(
                backend,
                working_set_source=source,
            )
            result = operation()
            after = capture_execution_snapshot(
                backend,
                working_set_source=source,
            )
            result.execution.run_table_growth = table_growth_delta(
                before.table_sizes, after.table_sizes)
            result.execution.peak_working_set_bytes = max(
                result.execution.peak_working_set_bytes,
                before.working_set_bytes,
                after.working_set_bytes,
            )
            result.execution.backend_telemetry = collector.to_json()
            return result


def item_candidate_counts(item: Any) -> dict[str, int]:
    """按 item 字段发现公开 ``candidates`` 集合，不绑定候选领域枚举。"""
    values = vars(item) if hasattr(item, "__dict__") else {}
    counts: dict[str, int] = {}
    for field_name, value in values.items():
        candidates = getattr(value, "candidates", None)
        if isinstance(candidates, (tuple, list)):
            counts[field_name] = len(candidates)
    return counts


def item_candidate_total(item: Any) -> int:
    """汇总一个 item 当前公开的全部候选集合大小。"""
    return sum(item_candidate_counts(item).values())


def language_structure_payload(summary: Any) -> dict[str, Any] | None:
    """把语言结构汇总转换为遥测结构，不依赖 formal_train 类型。"""
    if summary is None:
        return None
    return {
        "total_held_out": summary.total_held_out,
        "recognized": summary.recognized,
        "verified": summary.verified,
        "expected_verified": summary.expected_verified,
        "routing": (None if summary.routing_stats is None
                    else summary.routing_stats.to_json()),
        "tally": (None if summary.tally_stats is None
                  else summary.tally_stats.to_json()),
        "state": (None if summary.structure_state is None
                  else summary.structure_state.to_json()),
    }


def execution_payload(*, run_id: str,
                      execution: FormalTrainExecutionStats,
                      language_summary: Any) -> dict[str, Any]:
    """构造与图快照分离的编排遥测对象。"""
    return {
        "run_id": run_id,
        "execution": execution.to_json(),
        "language_structure": language_structure_payload(language_summary),
    }


def save_execution_metrics(*, run_dir: str, run_id: str,
                           execution: FormalTrainExecutionStats,
                           language_summary: Any) -> str:
    """确定性写入 ``execution.json``，不混入图状态。"""
    path = os.path.join(run_dir, run_id, "execution.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = execution_payload(
        run_id=run_id,
        execution=execution,
        language_summary=language_summary,
    )
    with open(path, "w", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return path


__all__ = [
    "ExecutionPhaseRecorder",
    "ExecutionPhaseStats",
    "ExecutionSnapshot",
    "FormalTrainExecutionStats",
    "TelemetryClock",
    "backend_operation_delta",
    "capture_execution_snapshot",
    "complete_execution_phase",
    "execution_payload",
    "item_candidate_counts",
    "item_candidate_total",
    "language_structure_payload",
    "process_working_set_bytes",
    "run_with_execution_telemetry",
    "sample_working_set",
    "save_execution_metrics",
    "table_growth_delta",
]
