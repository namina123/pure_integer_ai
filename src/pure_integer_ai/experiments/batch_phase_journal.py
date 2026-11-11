"""Fsynced, exclusive boundary files; never resume application work from them.

This independent journal is not wired into the live batch runner. A readable
boundary proves neither a committed Memory transaction nor a production call.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Iterator

from .batch_phase_evidence import (
    PHASES, BatchPhaseEvidenceError, BatchPhaseRecorder, PhaseEvidence,
)
from .batch_phase_host import ProcessResourceEndpoint, current_process_resource_endpoint


_FORMAT = "PURE_INTEGER_BATCH_PHASE_JOURNAL_V1"


def _context(run_identity: str, attempt_key: str, case_ordinal: int) -> dict[str, object]:
    for value in (run_identity, attempt_key):
        if type(value) is not str or not value.strip():
            raise BatchPhaseEvidenceError("run identity and attempt key must be nonempty strings")
    if type(case_ordinal) is not int or case_ordinal < 0:
        raise BatchPhaseEvidenceError("case ordinal must be a nonnegative integer")
    return dict(run_identity=run_identity, attempt_key=attempt_key, case_ordinal=case_ordinal)


def _phase(name: str) -> None:
    if type(name) is not str or name not in PHASES:
        raise BatchPhaseEvidenceError("phase name is not frozen")


def _required(phases: tuple[str, ...]) -> None:
    if type(phases) is not tuple or not phases:
        raise BatchPhaseEvidenceError("journal required phases must be a nonempty tuple")
    BatchPhaseRecorder(required_phases=phases)


class DurableBatchPhaseRecorder:
    """Require a persisted BEGIN before executing a caller-labelled interval.

    Each directory is one attempt and has one writer. Existing directories are
    refused, including failed or incomplete attempts. No files are overwritten.
    """

    def __init__(self, directory: Path, *, run_identity: str, attempt_key: str,
                 case_ordinal: int, required_phases: tuple[str, ...],
                 sampler: Callable[[], ProcessResourceEndpoint] = current_process_resource_endpoint,
                 ) -> None:
        self._context = _context(run_identity, attempt_key, case_ordinal)
        _required(required_phases)
        self._endpoint: ProcessResourceEndpoint | None = None
        self._sampling_error: BaseException | None = None
        self._raw_endpoint: ProcessResourceEndpoint | None = None
        self._last_endpoint: ProcessResourceEndpoint | None = None
        self._owner: tuple[int, int] | None = None
        self._sampler = sampler
        self._recorder = BatchPhaseRecorder(
            clock=lambda: self._sample().sampled_ended_ns,
            cpu_clock=lambda: self._sample().cpu_ns,
            resource_snapshot=lambda: asdict(self._sample()),
            required_phases=required_phases)
        self._required = required_phases
        self._active: tuple[str, int] | None = None
        self._ordinals: dict[str, int] = {}
        self._sequence = 0
        self._previous: str | None = None
        self._broken = False
        self.directory = Path(directory)
        # Parents must already exist; never create a new training run tree here.
        self.directory.mkdir(exist_ok=False)

    def _sample(self) -> ProcessResourceEndpoint:
        if self._sampling_error is not None:
            raise self._sampling_error
        if self._endpoint is None:
            raise BatchPhaseEvidenceError("endpoint is unavailable")
        return self._endpoint

    def _capture(self) -> None:
        self._endpoint = self._raw_endpoint = None
        self._sampling_error = None
        try:
            value = self._sampler()
            if not isinstance(value, ProcessResourceEndpoint):
                raise BatchPhaseEvidenceError("sampler must return a ProcessResourceEndpoint")
            self._raw_endpoint = value
            owner = (value.pid, value.creation_epoch_ns)
            if self._owner is not None and owner != self._owner:
                raise BatchPhaseEvidenceError("endpoint process creation identity changed")
            prior = self._last_endpoint
            if prior is not None:
                if value.sampled_started_ns < prior.sampled_ended_ns or value.cpu_ns < prior.cpu_ns:
                    raise BatchPhaseEvidenceError("endpoint clock/CPU reversed between boundaries")
                if (prior.process_peak_working_set_bytes is not None and
                        value.process_peak_working_set_bytes is not None and
                        value.process_peak_working_set_bytes < prior.process_peak_working_set_bytes):
                    raise BatchPhaseEvidenceError("cumulative process peak decreased between boundaries")
            self._owner = owner
            self._endpoint = value
            self._last_endpoint = value
        except BaseException as error:
            self._sampling_error = error

    def _write(self, event: str, name: str, ordinal: int, **data: object) -> None:
        if self._broken:
            raise BatchPhaseEvidenceError("journal write failed; attempt is not reusable")
        value = dict(format=_FORMAT, **self._context, sequence=self._sequence + 1,
                     previous_sha256=self._previous, event=event, phase=name,
                     ordinal=ordinal, **data)
        raw = (json.dumps(value, ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        path = self.directory / f"{self._sequence + 1:08d}.json"
        try:
            from .relation_resource_budget import before_resource_write
            before_resource_write(path, len(raw))
            with path.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            self._broken = True
            raise
        self._sequence += 1
        self._previous = hashlib.sha256(raw).hexdigest()

    def _failure(self, event: str, name: str, ordinal: int,
                 error: BaseException, operation_failure_type: str | None) -> None:
        try:
            self._write(event, name, ordinal, collection_error_type=type(error).__name__,
                        operation_failure_type=operation_failure_type,
                        endpoint=(None if self._raw_endpoint is None else
                                  self._raw_endpoint.to_value()),
                        collection_failures=self._recorder.report()["collection_failures"])
        except BaseException as persistence_error:
            error.add_note(f"Phase failure persistence also failed: {type(persistence_error).__name__}")

    def begin(self, name: str) -> None:
        _phase(name)
        if self._broken:
            raise BatchPhaseEvidenceError("journal write failed; attempt is not reusable")
        if self._active is not None:
            raise BatchPhaseEvidenceError("nested phase intervals are not allowed")
        ordinal = self._ordinals.get(name, 0) + 1
        self._ordinals[name] = ordinal
        self._capture()
        try:
            self._recorder.begin(name)
        except BaseException as error:
            self._failure("BEGIN_COLLECTION_FAILURE", name, ordinal, error, None)
            raise
        self._active = name, ordinal
        self._write("BEGIN", name, ordinal, endpoint=self._sample().to_value())

    def end(self, *, status: str = "PASS",
            operation_failure_type: str | None = None) -> PhaseEvidence:
        if self._broken:
            raise BatchPhaseEvidenceError("journal write failed; attempt is not reusable")
        if self._active is None:
            raise BatchPhaseEvidenceError("no phase is open")
        name, ordinal = self._active
        self._capture()
        try:
            row = self._recorder.end(status=status, operation_failure_type=operation_failure_type)
        except BaseException as error:
            self._active = None
            self._failure("END_COLLECTION_FAILURE", name, ordinal, error, operation_failure_type)
            raise
        self._active = None
        self._write("END", name, ordinal, endpoint=self._sample().to_value(), interval=row.to_value())
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
                    f"Phase endpoint persistence also failed: {type(collection_error).__name__}")
            raise
        else:
            self.end()

    def report(self) -> dict[str, object]:
        if self._broken:
            raise BatchPhaseEvidenceError("journal write failed; audit preserved files instead")
        return dict(in_memory_intervals=self._recorder.report(), journal=audit_phase_journal(
            self.directory, **self._context, required_phases=self._required))


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BatchPhaseEvidenceError("duplicate JSON field")
        result[key] = value
    return result


def audit_phase_journal(directory: Path, *, run_identity: str, attempt_key: str,
                        case_ordinal: int, required_phases: tuple[str, ...]) -> dict[str, object]:
    """Read original boundary files in a new instance, with no application calls.

    Hash chaining detects dropped/reordered predecessors, not authenticity or a
    removed suffix. An open last BEGIN remains an incomplete boundary; neither
    successful END nor fsync proves application commit or power-loss recovery.
    """
    context = _context(run_identity, attempt_key, case_ordinal)
    _required(required_phases)
    result: dict[str, object] = dict(
        format=_FORMAT, status="UNKNOWN", status_scope="PERSISTED_CALLER_LABELLED_BOUNDARIES_ONLY",
        **context, event_count=0, completed_intervals=[], collection_failures=[],
        incomplete_boundary=None, missing_required_phase_names=list(required_phases),
        owner_qualification="UNKNOWN_no_observed_endpoint", owner_identity=None,
        persistence_qualification="UNKNOWN_no_readable_boundaries",
        application_recovery_qualification="UNKNOWN_no_application_commit_or_resume",
        phase_peak_qualification="UNKNOWN_endpoint_samples_only",
        production_chain_qualification="UNKNOWN_no_call_site_binding",
        runtime_cost_causality="UNKNOWN_requires_comparison_and_phase_profile",
        language_qualification="UNKNOWN_not_measured_by_timing", free_dialogue_complete=0)
    previous = None
    active: dict[str, object] | None = None
    owner: tuple[int, int] | None = None
    prior_endpoint: ProcessResourceEndpoint | None = None
    ordinals: dict[str, int] = {}
    rows: list[PhaseEvidence] = []
    failures: list[dict[str, object]] = []
    try:
        paths = sorted(Path(directory).iterdir(), key=lambda path: path.name)
        for sequence, path in enumerate(paths, 1):
            if path.is_symlink() or not path.is_file() or path.name != f"{sequence:08d}.json":
                raise BatchPhaseEvidenceError("journal file sequence is incomplete or unexpected")
            raw = path.read_bytes()
            value = json.loads(raw, object_pairs_hook=_object)
            if type(value) is not dict or value.get("format") != _FORMAT:
                raise BatchPhaseEvidenceError("journal format drift")
            if any(value.get(key) != expected or type(value.get(key)) is not type(expected)
                   for key, expected in context.items()):
                raise BatchPhaseEvidenceError("journal run/attempt/case identity drift")
            if type(value.get("sequence")) is not int or value["sequence"] != sequence:
                raise BatchPhaseEvidenceError("journal sequence drift")
            if value.get("previous_sha256") != previous:
                raise BatchPhaseEvidenceError("journal predecessor digest drift")
            name, ordinal, event = value.get("phase"), value.get("ordinal"), value.get("event")
            _phase(name)
            if type(ordinal) is not int or ordinal <= 0:
                raise BatchPhaseEvidenceError("journal ordinal is invalid")
            endpoint = (None if value.get("endpoint") is None else
                        ProcessResourceEndpoint(**value["endpoint"]))
            if endpoint is not None:
                observed_owner = endpoint.pid, endpoint.creation_epoch_ns
                if owner is not None and owner != observed_owner:
                    raise BatchPhaseEvidenceError("journal endpoint creation identity drift")
                if prior_endpoint is not None:
                    if (endpoint.sampled_started_ns < prior_endpoint.sampled_ended_ns or
                            endpoint.cpu_ns < prior_endpoint.cpu_ns):
                        raise BatchPhaseEvidenceError("journal endpoint clock/CPU reversed")
                    before, after = (prior_endpoint.process_peak_working_set_bytes,
                                     endpoint.process_peak_working_set_bytes)
                    if before is not None and after is not None and after < before:
                        raise BatchPhaseEvidenceError("journal cumulative process peak decreased")
                owner = observed_owner
                prior_endpoint = endpoint
            if event in {"BEGIN", "BEGIN_COLLECTION_FAILURE"}:
                if active is not None or ordinal != ordinals.get(name, 0) + 1:
                    raise BatchPhaseEvidenceError("journal begin ordering drift")
                ordinals[name] = ordinal
                if event == "BEGIN":
                    if endpoint is None:
                        raise BatchPhaseEvidenceError("BEGIN endpoint is missing")
                    active = value
            elif event in {"END", "END_COLLECTION_FAILURE"}:
                if active is None or (name, ordinal) != (active["phase"], active["ordinal"]):
                    raise BatchPhaseEvidenceError("journal end has no matching BEGIN")
                if event == "END":
                    if endpoint is None:
                        raise BatchPhaseEvidenceError("END endpoint is missing")
                    started = ProcessResourceEndpoint(**active["endpoint"])
                    row = PhaseEvidence(**value["interval"])
                    expected = PhaseEvidence(
                        name, ordinal, started.sampled_ended_ns, endpoint.sampled_ended_ns,
                        endpoint.sampled_ended_ns - started.sampled_ended_ns,
                        started.cpu_ns, endpoint.cpu_ns, endpoint.cpu_ns - started.cpu_ns,
                        started.working_set_bytes, endpoint.working_set_bytes,
                        started.process_peak_working_set_bytes,
                        endpoint.process_peak_working_set_bytes, row.status,
                        row.operation_failure_type)
                    if row != expected or (rows and row.started_ns < rows[-1].ended_ns):
                        raise BatchPhaseEvidenceError("journal interval/endpoint drift")
                    rows.append(row)
                active = None
            else:
                raise BatchPhaseEvidenceError("journal event is unknown")
            if event.endswith("COLLECTION_FAILURE"):
                if type(value.get("collection_error_type")) is not str or not value["collection_error_type"]:
                    raise BatchPhaseEvidenceError("collection failure type is missing")
                failures.append(value)
            previous = hashlib.sha256(raw).hexdigest()
            result["event_count"] = sequence
        missing = [name for name in required_phases if not any(row.name == name for row in rows)]
        failed = bool(failures) or any(row.status == "FAIL" for row in rows)
        result.update(
            status="FAIL" if failed else "UNKNOWN" if active is not None or not paths
            else "FAIL" if missing else "PASS",
            completed_intervals=[row.to_value() for row in rows], collection_failures=failures,
            incomplete_boundary=active, missing_required_phase_names=missing,
            last_file_sha256=previous,
            owner_qualification=("UNKNOWN_no_observed_endpoint" if owner is None else
                                 "OBSERVED_recorded_endpoint_identity_consistency_only"),
            owner_identity=(None if owner is None else dict(pid=owner[0], creation_epoch_ns=owner[1])),
            persistence_qualification=("UNKNOWN_no_readable_boundaries" if not paths else
                                       "OBSERVED_readable_boundary_files_only"))
    except (OSError, ValueError, TypeError, KeyError) as error:
        result.update(status="FAIL", audit_error_type=type(error).__name__, audit_error=str(error))
    return result
