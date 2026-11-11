"""Caller-supplied phase endpoints, independent from the live batch runner.

These in-memory records do not establish durable recovery, process ownership,
phase peaks, production call-site coverage, or language capability.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Callable, Iterator, Mapping


class BatchPhaseEvidenceError(ValueError):
    """Phase evidence is malformed, incomplete or outside the frozen contract."""


PHASES = (
    "identity",
    "training",
    "materialization",
    "worker_spawn",
    "worker_wait",
    "worker_result_read",
    "memory_init",
    "memory_recover",
    "memory_append",
    "query",
    "core",
    "companion",
    "response_plan",
    "generation",
    "delivery",
    "memory_commit",
    "finalize",
)
_PHASE_SET = frozenset(PHASES)


def _positive(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise BatchPhaseEvidenceError(f"{field} must be a positive integer")
    return value


def _nonnegative(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise BatchPhaseEvidenceError(f"{field} must be a nonnegative integer")
    return value


def _sample(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _nonnegative(value, field)


def _phase_name(value: object) -> str:
    if type(value) is not str or value not in _PHASE_SET:
        raise BatchPhaseEvidenceError("phase name is not frozen")
    return value


def _resource_pair(current: object, peak: object) -> tuple[int | None, int | None]:
    current = _sample(current, "working_set_bytes")
    peak = _sample(peak, "process_peak_working_set_bytes")
    if current is not None and peak is not None and peak < current:
        raise BatchPhaseEvidenceError("process peak is below current working set")
    return current, peak


def _qualification(present: int, expected: int) -> str:
    if not expected or not present:
        return "UNKNOWN"
    return "OBSERVED" if present == expected else "PARTIAL"


@dataclass(frozen=True, slots=True)
class PhaseEvidence:
    """One interval, with separately validated optional endpoint counters."""

    name: str
    ordinal: int
    started_ns: int
    ended_ns: int
    elapsed_ns: int
    cpu_started_ns: int | None
    cpu_ended_ns: int | None
    cpu_elapsed_ns: int | None
    working_set_started_bytes: int | None
    working_set_ended_bytes: int | None
    process_peak_started_bytes: int | None
    process_peak_ended_bytes: int | None
    status: str
    operation_failure_type: str | None = None

    def __post_init__(self) -> None:
        _phase_name(self.name)
        _positive(self.ordinal, "ordinal")
        _nonnegative(self.started_ns, "started_ns")
        _nonnegative(self.ended_ns, "ended_ns")
        _positive(self.elapsed_ns, "elapsed_ns")
        if self.ended_ns < self.started_ns or self.elapsed_ns != self.ended_ns - self.started_ns:
            raise BatchPhaseEvidenceError("phase wall interval is inconsistent")
        cpu_started = _sample(self.cpu_started_ns, "cpu_started_ns")
        cpu_ended = _sample(self.cpu_ended_ns, "cpu_ended_ns")
        cpu_elapsed = _sample(self.cpu_elapsed_ns, "cpu_elapsed_ns")
        if cpu_started is None or cpu_ended is None:
            if self.cpu_elapsed_ns is not None:
                raise BatchPhaseEvidenceError("CPU elapsed requires both CPU endpoints")
        else:
            if cpu_ended < cpu_started:
                raise BatchPhaseEvidenceError("CPU interval is reversed")
            if cpu_elapsed != cpu_ended - cpu_started:
                raise BatchPhaseEvidenceError("CPU interval is inconsistent")
        _, peak_started = _resource_pair(
            self.working_set_started_bytes, self.process_peak_started_bytes)
        _, peak_ended = _resource_pair(
            self.working_set_ended_bytes, self.process_peak_ended_bytes)
        if peak_started is not None and peak_ended is not None and peak_ended < peak_started:
            raise BatchPhaseEvidenceError("cumulative process peak decreased")
        if not isinstance(self.status, str) or self.status not in {"PASS", "FAIL"}:
            raise BatchPhaseEvidenceError("phase status must be PASS or FAIL")
        if self.operation_failure_type is not None:
            if type(self.operation_failure_type) is not str or not self.operation_failure_type:
                raise BatchPhaseEvidenceError("operation failure type must be a nonempty string")
            if self.status != "FAIL":
                raise BatchPhaseEvidenceError("operation failure cannot have PASS status")

    def to_value(self) -> dict[str, object]:
        return {
            "name": self.name, "ordinal": self.ordinal,
            "started_ns": self.started_ns, "ended_ns": self.ended_ns,
            "elapsed_ns": self.elapsed_ns,
            "cpu_started_ns": self.cpu_started_ns,
            "cpu_ended_ns": self.cpu_ended_ns,
            "cpu_elapsed_ns": self.cpu_elapsed_ns,
            "working_set_started_bytes": self.working_set_started_bytes,
            "working_set_ended_bytes": self.working_set_ended_bytes,
            "process_peak_started_bytes": self.process_peak_started_bytes,
            "process_peak_ended_bytes": self.process_peak_ended_bytes,
            "status": self.status,
            "operation_failure_type": self.operation_failure_type,
        }


@dataclass(frozen=True, slots=True)
class _OpenPhase:
    name: str
    ordinal: int
    started_ns: int
    cpu_started_ns: int | None
    working_set_bytes: int | None
    process_peak_working_set_bytes: int | None


class BatchPhaseRecorder:
    """Capture explicit phase intervals without reading application state."""

    def __init__(
            self, *, clock: Callable[[], int] | None = None,
            cpu_clock: Callable[[], int] | None = None,
            resource_snapshot: Callable[[], Mapping[str, object]] | None = None,
            required_phases: tuple[str, ...] | None = None,
            ) -> None:
        if required_phases is not None:
            if type(required_phases) is not tuple or not required_phases:
                raise BatchPhaseEvidenceError("required phases must be a nonempty tuple")
            for name in required_phases:
                _phase_name(name)
            if len(set(required_phases)) != len(required_phases):
                raise BatchPhaseEvidenceError("required phases contain duplicates")
        self._clock = clock or time.perf_counter_ns
        self._cpu_clock = cpu_clock
        self._resource_snapshot = resource_snapshot
        self._rows: list[PhaseEvidence] = []
        self._open: _OpenPhase | None = None
        self._ordinals: dict[str, int] = {}
        self._required_phases = required_phases
        self._collection_failures: list[dict[str, object]] = []

    def _now(self, source: Callable[[], int], field: str) -> int:
        return _nonnegative(source(), field)

    def _resources(self) -> tuple[int | None, int | None]:
        if self._resource_snapshot is None:
            return None, None
        value = self._resource_snapshot()
        if not isinstance(value, Mapping):
            raise BatchPhaseEvidenceError("resource snapshot must be a mapping")
        current = value.get("working_set_bytes")
        peak = value.get("process_peak_working_set_bytes")
        return _resource_pair(current, peak)

    def _collection_failure(
            self, *, name: str, ordinal: int, boundary: str,
            error: BaseException, samples: Mapping[str, object],
            operation_failure_type: str | None = None,
            ) -> None:
        self._collection_failures.append({
            "name": name, "ordinal": ordinal, "boundary": boundary,
            "status": "FAIL", "error_type": type(error).__name__,
            "operation_failure_type": operation_failure_type,
            "available_samples": dict(samples),
        })

    def begin(self, name: str) -> None:
        _phase_name(name)
        if self._open is not None:
            raise BatchPhaseEvidenceError("nested phase intervals are not allowed")
        ordinal = self._ordinals.get(name, 0) + 1
        self._ordinals[name] = ordinal
        samples: dict[str, object] = {}
        try:
            started = self._now(self._clock, "started_ns")
            samples["started_ns"] = started
            if self._rows and started < self._rows[-1].ended_ns:
                raise BatchPhaseEvidenceError("phase clock reversed between intervals")
            cpu_started = (None if self._cpu_clock is None
                           else self._now(self._cpu_clock, "cpu_started_ns"))
            samples["cpu_started_ns"] = cpu_started
            working, peak = self._resources()
            samples.update(working_set_started_bytes=working, process_peak_started_bytes=peak)
        except BaseException as error:
            self._collection_failure(name=name, ordinal=ordinal, boundary="begin",
                                     error=error, samples=samples)
            raise
        self._open = _OpenPhase(name, ordinal, started, cpu_started, working, peak)

    def end(self, *, status: str = "PASS",
            operation_failure_type: str | None = None) -> PhaseEvidence:
        opened = self._open
        if opened is None:
            raise BatchPhaseEvidenceError("no phase is open")
        samples: dict[str, object] = {
            "started_ns": opened.started_ns, "cpu_started_ns": opened.cpu_started_ns,
            "working_set_started_bytes": opened.working_set_bytes,
            "process_peak_started_bytes": opened.process_peak_working_set_bytes,
        }
        try:
            ended = self._now(self._clock, "ended_ns")
            samples["ended_ns"] = ended
            if ended <= opened.started_ns:
                raise BatchPhaseEvidenceError("phase clock did not advance")
            cpu_ended = (None if self._cpu_clock is None
                         else self._now(self._cpu_clock, "cpu_ended_ns"))
            samples["cpu_ended_ns"] = cpu_ended
            current, peak = self._resources()
            samples.update(working_set_ended_bytes=current, process_peak_ended_bytes=peak)
            row = PhaseEvidence(
                opened.name, opened.ordinal, opened.started_ns, ended,
                ended - opened.started_ns, opened.cpu_started_ns, cpu_ended,
                (None if opened.cpu_started_ns is None or cpu_ended is None
                 else cpu_ended - opened.cpu_started_ns),
                opened.working_set_bytes, current,
                opened.process_peak_working_set_bytes, peak, status, operation_failure_type)
        except BaseException as error:
            self._collection_failure(
                name=opened.name, ordinal=opened.ordinal, boundary="end",
                error=error, samples=samples, operation_failure_type=operation_failure_type)
            self._open = None
            raise
        self._rows.append(row)
        self._open = None
        return row

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self.begin(name)
        try:
            yield
        except BaseException as operation_error:
            try:
                self.end(status="FAIL", operation_failure_type=type(operation_error).__name__)
            except BaseException as collection_error:
                operation_error.add_note(
                    f"Phase endpoint collection also failed: {type(collection_error).__name__}")
            raise
        else:
            self.end()

    def report(self) -> dict[str, object]:
        if self._open is not None:
            raise BatchPhaseEvidenceError("cannot finalize with an open phase")
        rows = tuple(self._rows)
        observed = {row.name for row in rows}
        missing = ([] if self._required_phases is None else
                   [name for name in self._required_phases if name not in observed])
        coverage = ("UNKNOWN" if self._required_phases is None else
                    "FAIL" if missing else "PASS")
        intervals = ("FAIL" if self._collection_failures or any(
            row.status == "FAIL" for row in rows) else "PASS" if rows else "UNKNOWN")
        status = ("FAIL" if "FAIL" in (coverage, intervals) else
                  "PASS" if coverage == intervals == "PASS" else "UNKNOWN")
        cpu_intervals = sum(row.cpu_elapsed_ns is not None for row in rows)
        working_samples = sum(value is not None for row in rows for value in (
            row.working_set_started_bytes, row.working_set_ended_bytes))
        peak_samples = sum(value is not None for row in rows for value in (
            row.process_peak_started_bytes, row.process_peak_ended_bytes))
        return {
            "format": "PURE_INTEGER_BATCH_PHASE_EVIDENCE_V2",
            "status": status,
            "status_scope": "CALLER_DECLARED_PHASE_NAMES_AND_RECORDED_INTERVALS_ONLY",
            "recorded_intervals_status": intervals,
            "allowed_phase_names": list(PHASES),
            "required_phase_names": (None if self._required_phases is None else
                                     list(self._required_phases)),
            "observed_phase_names": [name for name in PHASES if name in observed],
            "missing_required_phase_names": missing,
            "required_phase_coverage": coverage,
            "phases": [row.to_value() for row in rows],
            "phase_count": len(rows),
            "collection_failures": [dict(value, available_samples=dict(value["available_samples"]))
                                    for value in self._collection_failures],
            "cpu_interval_count": cpu_intervals,
            "working_set_endpoint_count": working_samples,
            "process_peak_endpoint_count": peak_samples,
            "cpu_qualification": _qualification(cpu_intervals, len(rows)),
            "resource_qualification": _qualification(working_samples, 2 * len(rows)),
            "process_peak_qualification": _qualification(peak_samples, 2 * len(rows)),
            "measurement_scope": "CALLER_SUPPLIED_ENDPOINTS_OF_RECORDED_INTERVALS",
            "phase_peak_qualification": "UNKNOWN_endpoint_samples_do_not_measure_phase_peak",
            "owner_qualification": "UNKNOWN_no_process_identity_binding",
            "durable_recovery_qualification": "UNKNOWN_in_memory_only",
            "production_chain_qualification": "UNKNOWN_no_call_site_binding",
            "runtime_cost_causality": "UNKNOWN_requires_comparison_and_phase_profile",
            "language_qualification": "UNKNOWN_not_measured_by_timing",
            "free_dialogue_complete": 0,
        }
