"""存储操作的 context-local 外层诊断遥测。

本模块只记录后端已经发生的调用、返回行数和调用语境，不解释表的领域含义，
也不向核心图或伴随表写入任何遥测状态。采集器未启用时，后端只读取一次空的
``ContextVar``；诊断快照使用 suppression，不能把自己的统计查询计入被测成本。
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any, Iterator


StableKey = tuple[int, ...]
_UNSET = object()


def _validate_stable_key(value: StableKey | None, *, where: str) -> None:
    """校验可选稳定键只含严格整数，避免把对象地址带入诊断身份。"""
    if value is None:
        return
    if not isinstance(value, tuple):
        raise TypeError(f"{where} 必须是整数元组或 None")
    if any(type(part) is not int for part in value):
        raise TypeError(f"{where} 只能包含严格整数")


@dataclass(frozen=True)
class TelemetryScope:
    """一次存储调用可继承和覆盖的开放诊断语境。"""

    caller: str | None = None
    query: str | None = None
    source_key: StableKey | None = None
    occurrence_key: StableKey | None = None
    assertion_key: StableKey | None = None
    scope_key: StableKey | None = None
    stage: int | None = None
    round_id: int | None = None
    item_index: int | None = None
    evaluation: bool = False

    def __post_init__(self) -> None:
        if self.caller is not None and not isinstance(self.caller, str):
            raise TypeError("TelemetryScope.caller 必须是字符串或 None")
        if self.query is not None and not isinstance(self.query, str):
            raise TypeError("TelemetryScope.query 必须是字符串或 None")
        _validate_stable_key(self.source_key, where="TelemetryScope.source_key")
        _validate_stable_key(
            self.occurrence_key, where="TelemetryScope.occurrence_key")
        _validate_stable_key(
            self.assertion_key, where="TelemetryScope.assertion_key")
        _validate_stable_key(self.scope_key, where="TelemetryScope.scope_key")
        for name in ("stage", "round_id", "item_index"):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise TypeError(f"TelemetryScope.{name} 必须是严格整数或 None")
        if type(self.evaluation) is not bool:
            raise TypeError("TelemetryScope.evaluation 必须是 bool")

    def to_json(self) -> dict[str, Any]:
        """导出稳定 JSON 字段，完整键保留为整数序列。"""
        return {
            "caller": self.caller,
            "query": self.query,
            "source": (None if self.source_key is None
                       else list(self.source_key)),
            "occurrence": (None if self.occurrence_key is None
                           else list(self.occurrence_key)),
            "assertion": (None if self.assertion_key is None
                          else list(self.assertion_key)),
            "scope": (None if self.scope_key is None
                      else list(self.scope_key)),
            "stage": self.stage,
            "round_id": self.round_id,
            "item_index": self.item_index,
            "evaluation": self.evaluation,
        }


@dataclass
class BackendOperationStats:
    """单类后端操作的调用、行数和失败计数。"""

    calls: int = 0
    rows: int = 0
    failures: int = 0

    def add(self, *, rows: int, failed: bool) -> None:
        """累加一次调用；失败调用不伪造返回或影响行数。"""
        self.calls += 1
        if failed:
            self.failures += 1
        else:
            self.rows += rows

    def merge(self, other: "BackendOperationStats") -> None:
        """合并另一个桶，供低基数维度聚合使用。"""
        self.calls += other.calls
        self.rows += other.rows
        self.failures += other.failures

    def to_json(self) -> dict[str, int]:
        """导出纯整数计数。"""
        return {
            "calls": self.calls,
            "rows": self.rows,
            "failures": self.failures,
        }


@dataclass
class CandidateStats:
    """候选集合被观察的次数和候选总量。"""

    observations: int = 0
    candidates: int = 0

    def add(self, count: int) -> None:
        """记录一次候选集合观察。"""
        self.observations += 1
        self.candidates += count

    def merge(self, other: "CandidateStats") -> None:
        """合并候选统计。"""
        self.observations += other.observations
        self.candidates += other.candidates

    def to_json(self) -> dict[str, int]:
        """导出纯整数候选统计。"""
        return {
            "observations": self.observations,
            "candidates": self.candidates,
        }


@dataclass
class DiagnosticEventStats:
    """开放诊断事件的严格整数累计次数。"""

    count: int = 0

    def add(self, count: int) -> None:
        """累加调用方显式报告的事件次数。"""
        self.count += count

    def merge(self, other: "DiagnosticEventStats") -> None:
        """合并另一个作用域桶。"""
        self.count += other.count

    def to_json(self) -> dict[str, int]:
        """导出纯整数事件计数。"""
        return {"count": self.count}


class BackendTelemetryCollector:
    """收集后端操作，并保留可按开放作用域定位的组合明细。"""

    def __init__(self) -> None:
        self._totals: dict[tuple[str, str], BackendOperationStats] = {}
        self._scoped: dict[
            TelemetryScope,
            dict[tuple[str, str], BackendOperationStats],
        ] = {}
        self._candidate_totals: dict[str, CandidateStats] = {}
        self._scoped_candidates: dict[
            TelemetryScope,
            dict[str, CandidateStats],
        ] = {}
        self._event_totals: dict[str, DiagnosticEventStats] = {}
        self._scoped_events: dict[
            TelemetryScope,
            dict[str, DiagnosticEventStats],
        ] = {}

    def record(self, operation: str, table: str, *,
               rows: int = 0, failed: bool = False,
               scope: TelemetryScope | None = None) -> None:
        """记录一次已完成或已失败的后端调用。"""
        if type(rows) is not int or rows < 0:
            raise TypeError("后端遥测 rows 必须是非负严格整数")
        active_scope = scope or current_telemetry_scope()
        key = (operation, table)
        self._totals.setdefault(key, BackendOperationStats()).add(
            rows=rows, failed=failed)
        scoped = self._scoped.setdefault(active_scope, {})
        scoped.setdefault(key, BackendOperationStats()).add(
            rows=rows, failed=failed)

    def record_candidates(self, kind: str, count: int, *,
                          scope: TelemetryScope | None = None) -> None:
        """记录调用方已生成的候选数，不解释候选的领域类型。"""
        if not isinstance(kind, str) or not kind:
            raise TypeError("候选遥测 kind 必须是非空字符串")
        if type(count) is not int or count < 0:
            raise TypeError("候选遥测 count 必须是非负严格整数")
        active_scope = scope or current_telemetry_scope()
        self._candidate_totals.setdefault(kind, CandidateStats()).add(count)
        scoped = self._scoped_candidates.setdefault(active_scope, {})
        scoped.setdefault(kind, CandidateStats()).add(count)

    def record_event(self, kind: str, count: int = 1, *,
                     scope: TelemetryScope | None = None) -> None:
        """记录开放诊断事件，不把函数类别固化为存储表或语义枚举。"""
        if not isinstance(kind, str) or not kind:
            raise TypeError("诊断事件 kind 必须是非空字符串")
        if type(count) is not int or count < 0:
            raise TypeError("诊断事件 count 必须是非负严格整数")
        active_scope = scope or current_telemetry_scope()
        self._event_totals.setdefault(
            kind, DiagnosticEventStats()).add(count)
        scoped = self._scoped_events.setdefault(active_scope, {})
        scoped.setdefault(kind, DiagnosticEventStats()).add(count)

    def operation_snapshot(self) -> dict[tuple[str, str], tuple[int, int, int]]:
        """返回可做前后差分的不可变计数投影。"""
        return {
            key: (stats.calls, stats.rows, stats.failures)
            for key, stats in self._totals.items()
        }

    @staticmethod
    def _operation_payload(
            operations: dict[tuple[str, str], BackendOperationStats],
            ) -> list[dict[str, Any]]:
        """把表级操作桶转换为确定顺序列表。"""
        payload: list[dict[str, Any]] = []
        for operation, table in sorted(operations):
            row: dict[str, Any] = {
                "operation": operation,
                "table": table,
            }
            row.update(operations[(operation, table)].to_json())
            payload.append(row)
        return payload

    @staticmethod
    def _candidate_payload(
            candidates: dict[str, CandidateStats],
            ) -> list[dict[str, Any]]:
        """把候选桶转换为确定顺序列表。"""
        payload: list[dict[str, Any]] = []
        for kind in sorted(candidates):
            row: dict[str, Any] = {"kind": kind}
            row.update(candidates[kind].to_json())
            payload.append(row)
        return payload

    @staticmethod
    def _event_payload(
            events: dict[str, DiagnosticEventStats],
            ) -> list[dict[str, Any]]:
        """把事件桶转换为确定顺序列表。"""
        payload: list[dict[str, Any]] = []
        for kind in sorted(events):
            row: dict[str, Any] = {"kind": kind}
            row.update(events[kind].to_json())
            payload.append(row)
        return payload

    @staticmethod
    def _merge_operations(
            target: dict[tuple[str, str], BackendOperationStats],
            source: dict[tuple[str, str], BackendOperationStats],
            ) -> None:
        """把一个组合 scope 的操作计数合入维度桶。"""
        for key, stats in source.items():
            target.setdefault(key, BackendOperationStats()).merge(stats)

    @staticmethod
    def _merge_candidates(target: dict[str, CandidateStats],
                          source: dict[str, CandidateStats]) -> None:
        """把一个组合 scope 的候选计数合入维度桶。"""
        for kind, stats in source.items():
            target.setdefault(kind, CandidateStats()).merge(stats)

    @staticmethod
    def _merge_events(target: dict[str, DiagnosticEventStats],
                      source: dict[str, DiagnosticEventStats]) -> None:
        """把一个组合 scope 的事件计数合入维度桶。"""
        for kind, stats in source.items():
            target.setdefault(kind, DiagnosticEventStats()).merge(stats)

    @staticmethod
    def _dimension_json(value: Any) -> Any:
        """把哈希维度值转换为 JSON 值。"""
        return list(value) if isinstance(value, tuple) else value

    def _dimension_payload(self, field: str) -> list[dict[str, Any]]:
        """按单一维度聚合操作和候选，避免报告使用不可读复合键。"""
        grouped_ops: dict[Any, dict[tuple[str, str], BackendOperationStats]] = {}
        grouped_candidates: dict[Any, dict[str, CandidateStats]] = {}
        grouped_events: dict[Any, dict[str, DiagnosticEventStats]] = {}
        scopes = (
            set(self._scoped)
            | set(self._scoped_candidates)
            | set(self._scoped_events)
        )
        for scope in scopes:
            value = getattr(scope, field)
            if value is None:
                continue
            self._merge_operations(
                grouped_ops.setdefault(value, {}),
                self._scoped.get(scope, {}),
            )
            self._merge_candidates(
                grouped_candidates.setdefault(value, {}),
                self._scoped_candidates.get(scope, {}),
            )
            self._merge_events(
                grouped_events.setdefault(value, {}),
                self._scoped_events.get(scope, {}),
            )
        payload: list[dict[str, Any]] = []
        values = set(grouped_ops) | set(grouped_candidates) | set(grouped_events)
        for value in sorted(values, key=repr):
            payload.append({
                "value": self._dimension_json(value),
                "operations": self._operation_payload(
                    grouped_ops.get(value, {})),
                "candidates": self._candidate_payload(
                    grouped_candidates.get(value, {})),
                "events": self._event_payload(
                    grouped_events.get(value, {})),
            })
        return payload

    def to_json(self) -> dict[str, Any]:
        """导出总计、单维分桶和可定位的组合 scope 明细。"""
        operation_totals: dict[str, BackendOperationStats] = {}
        for (operation, _table), stats in self._totals.items():
            operation_totals.setdefault(
                operation, BackendOperationStats()).merge(stats)
        total_payload = []
        for operation in sorted(operation_totals):
            row: dict[str, Any] = {"operation": operation}
            row.update(operation_totals[operation].to_json())
            total_payload.append(row)

        scope_payload: list[dict[str, Any]] = []
        scopes = (
            set(self._scoped)
            | set(self._scoped_candidates)
            | set(self._scoped_events)
        )
        for scope in sorted(scopes, key=lambda value: repr(value.to_json())):
            scope_payload.append({
                "scope": scope.to_json(),
                "operations": self._operation_payload(
                    self._scoped.get(scope, {})),
                "candidates": self._candidate_payload(
                    self._scoped_candidates.get(scope, {})),
                "events": self._event_payload(
                    self._scoped_events.get(scope, {})),
            })
        dimensions = {
            name: self._dimension_payload(field)
            for name, field in (
                ("caller", "caller"),
                ("query", "query"),
                ("source", "source_key"),
                ("occurrence", "occurrence_key"),
                ("assertion", "assertion_key"),
                ("scope", "scope_key"),
                ("stage", "stage"),
                ("round", "round_id"),
                ("item", "item_index"),
                ("evaluation", "evaluation"),
            )
        }
        return {
            "operation_totals": total_payload,
            "table_operations": self._operation_payload(self._totals),
            "candidate_totals": self._candidate_payload(
                self._candidate_totals),
            "event_totals": self._event_payload(self._event_totals),
            "by_dimension": dimensions,
            "scopes": scope_payload,
        }


_EMPTY_SCOPE = TelemetryScope()
_ACTIVE_COLLECTOR: ContextVar[BackendTelemetryCollector | None] = ContextVar(
    "pure_integer_ai.storage.active_telemetry_collector", default=None)
_ACTIVE_SCOPE: ContextVar[TelemetryScope] = ContextVar(
    "pure_integer_ai.storage.active_telemetry_scope", default=_EMPTY_SCOPE)


def active_backend_telemetry() -> BackendTelemetryCollector | None:
    """返回当前 context 的采集器；未启用时返回 None。"""
    return _ACTIVE_COLLECTOR.get()


def current_telemetry_scope() -> TelemetryScope:
    """返回当前 context 的不可变诊断作用域。"""
    return _ACTIVE_SCOPE.get()


def push_telemetry_scope(*, caller: str | None | object = _UNSET,
                         query: str | None | object = _UNSET,
                         source_key: StableKey | None | object = _UNSET,
                         occurrence_key: StableKey | None | object = _UNSET,
                         assertion_key: StableKey | None | object = _UNSET,
                         scope_key: StableKey | None | object = _UNSET,
                         stage: int | None | object = _UNSET,
                         round_id: int | None | object = _UNSET,
                         item_index: int | None | object = _UNSET,
                         evaluation: bool | object = _UNSET,
                         ) -> Token[TelemetryScope]:
    """在当前作用域上覆盖给定字段，并返回精确复位 token。"""
    current = current_telemetry_scope()
    updates = {
        name: value
        for name, value in {
            "caller": caller,
            "query": query,
            "source_key": source_key,
            "occurrence_key": occurrence_key,
            "assertion_key": assertion_key,
            "scope_key": scope_key,
            "stage": stage,
            "round_id": round_id,
            "item_index": item_index,
            "evaluation": evaluation,
        }.items()
        if value is not _UNSET
    }
    return _ACTIVE_SCOPE.set(replace(current, **updates))


def reset_telemetry_scope(token: Token[TelemetryScope]) -> None:
    """精确恢复 push 前的诊断作用域。"""
    _ACTIVE_SCOPE.reset(token)


@contextmanager
def telemetry_scope(**updates: Any) -> Iterator[TelemetryScope]:
    """在一个词法边界内叠加诊断作用域，异常时同样复位。"""
    token = push_telemetry_scope(**updates)
    try:
        yield current_telemetry_scope()
    finally:
        reset_telemetry_scope(token)


@contextmanager
def telemetry_scope_if_active(**updates: Any) -> Iterator[TelemetryScope]:
    """仅在采集器存在时叠加 scope，供默认热路径避免无效写 ContextVar。"""
    if active_backend_telemetry() is None:
        yield _EMPTY_SCOPE
        return
    with telemetry_scope(**updates) as active:
        yield active


@contextmanager
def collect_backend_telemetry(
        collector: BackendTelemetryCollector | None = None,
        ) -> Iterator[BackendTelemetryCollector]:
    """为当前 context 启用采集器，嵌套调用退出后恢复原采集器。"""
    active = collector or BackendTelemetryCollector()
    token = _ACTIVE_COLLECTOR.set(active)
    try:
        yield active
    finally:
        _ACTIVE_COLLECTOR.reset(token)


@contextmanager
def suppress_backend_telemetry() -> Iterator[None]:
    """临时关闭采集，供表规模快照等诊断自身查询使用。"""
    token = _ACTIVE_COLLECTOR.set(None)
    try:
        yield None
    finally:
        _ACTIVE_COLLECTOR.reset(token)


def record_candidate_count(kind: str, count: int) -> None:
    """若采集已启用，则把调用方候选数写入当前诊断 scope。"""
    collector = active_backend_telemetry()
    if collector is not None:
        collector.record_candidates(kind, count)


def record_diagnostic_event(kind: str, count: int = 1) -> None:
    """若采集已启用，则把开放诊断事件写入当前 scope。"""
    collector = active_backend_telemetry()
    if collector is not None:
        collector.record_event(kind, count)


__all__ = [
    "BackendOperationStats",
    "BackendTelemetryCollector",
    "CandidateStats",
    "DiagnosticEventStats",
    "TelemetryScope",
    "active_backend_telemetry",
    "collect_backend_telemetry",
    "current_telemetry_scope",
    "push_telemetry_scope",
    "record_candidate_count",
    "record_diagnostic_event",
    "reset_telemetry_scope",
    "suppress_backend_telemetry",
    "telemetry_scope",
    "telemetry_scope_if_active",
]
