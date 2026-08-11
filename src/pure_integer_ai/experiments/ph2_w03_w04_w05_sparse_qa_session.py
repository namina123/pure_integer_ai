"""FT23 long-lived JSONL session over one reusable sparse QA runtime."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Iterator

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime import (
    run_sparse_qa_query,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime_contract import (
    SparseQAResult,
    SparseQARuntime,
)


SPARSE_QA_SESSION_KINDS = {"ERROR", "RESULT"}
SPARSE_QA_SESSION_ERROR_CODES = {
    "EMPTY_LINE",
    "INVALID_AUDIT",
    "INVALID_FIELDS",
    "INVALID_JSON",
    "INVALID_QUESTION",
    "INVALID_SOURCE_REF",
}


# object-model: exception
class W03W04W05SparseQASessionError(ValueError):
    """An FT23 session record or deterministic session probe drifted."""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W03W04W05SparseQASessionError(
            f"{where} is not a canonical SHA-256")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SparseQASessionRecord:
    """One ordered JSONL result or a typed input error."""

    line_ordinal: int
    runtime_identity_sha256: str
    kind: str
    result: SparseQAResult | None
    error_code: str | None

    def __post_init__(self) -> None:
        if type(self.line_ordinal) is not int or self.line_ordinal < 0:
            raise W03W04W05SparseQASessionError(
                "FT23 line ordinal is invalid")
        _sha256(self.runtime_identity_sha256, where="FT23 runtime")
        if self.kind not in SPARSE_QA_SESSION_KINDS:
            raise W03W04W05SparseQASessionError(
                "FT23 session record kind is invalid")
        if self.kind == "RESULT":
            if (not isinstance(self.result, SparseQAResult)
                    or self.result.runtime_identity_sha256
                    != self.runtime_identity_sha256
                    or self.error_code is not None):
                raise W03W04W05SparseQASessionError(
                    "FT23 result record escaped its runtime")
        elif (self.result is not None
                or self.error_code not in SPARSE_QA_SESSION_ERROR_CODES):
            raise W03W04W05SparseQASessionError(
                "FT23 error record is incomplete")

    def to_dict(self) -> dict[str, object]:
        return {
            "error_code": self.error_code,
            "kind": self.kind,
            "line_ordinal": self.line_ordinal,
            "result": None if self.result is None else self.result.to_dict(),
            "result_sha256": (
                None if self.result is None else self.result.sha256()),
            "runtime_identity_sha256": self.runtime_identity_sha256,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SparseQASessionProbe:
    """Session-wide build, isolation, audit, and repetition counts."""

    runtime_identity_sha256: str
    runtime_build_count: int
    input_line_count: int
    query_count: int
    error_count: int
    audit_projection_count: int
    created_sparse_trace_count: int
    created_audit_trace_count: int
    record_commitment_sha256: str
    result_commitment_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.runtime_identity_sha256, where="FT23 probe runtime")
        counts = (
            self.runtime_build_count,
            self.input_line_count,
            self.query_count,
            self.error_count,
            self.audit_projection_count,
            self.created_sparse_trace_count,
            self.created_audit_trace_count,
        )
        if any(type(item) is not int or item < 0 for item in counts):
            raise W03W04W05SparseQASessionError(
                "FT23 session probe count is invalid")
        if (self.runtime_build_count != 1
                or self.query_count + self.error_count
                != self.input_line_count
                or self.audit_projection_count > self.query_count):
            raise W03W04W05SparseQASessionError(
                "FT23 session counts diverged")
        _sha256(
            self.record_commitment_sha256,
            where="FT23 record commitment",
        )
        _sha256(
            self.result_commitment_sha256,
            where="FT23 result commitment",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_projection_count": self.audit_projection_count,
            "created_audit_trace_count": self.created_audit_trace_count,
            "created_sparse_trace_count": self.created_sparse_trace_count,
            "error_count": self.error_count,
            "input_line_count": self.input_line_count,
            "query_count": self.query_count,
            "record_commitment_sha256": self.record_commitment_sha256,
            "result_commitment_sha256": self.result_commitment_sha256,
            "runtime_build_count": self.runtime_build_count,
            "runtime_identity_sha256": self.runtime_identity_sha256,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class SparseQASessionState:
    """Constant-size rolling state for one streamed JSONL session."""

    runtime_identity_sha256: str
    runtime_build_count: int
    input_line_count: int
    query_count: int
    error_count: int
    audit_projection_count: int
    created_sparse_trace_count: int
    created_audit_trace_count: int
    record_commitment_sha256: str
    result_commitment_sha256: str

    def __post_init__(self) -> None:
        SparseQASessionProbe(
            self.runtime_identity_sha256,
            self.runtime_build_count,
            self.input_line_count,
            self.query_count,
            self.error_count,
            self.audit_projection_count,
            self.created_sparse_trace_count,
            self.created_audit_trace_count,
            self.record_commitment_sha256,
            self.result_commitment_sha256,
        )


def _error(
        runtime: SparseQARuntime,
        ordinal: int,
        code: str,
        ) -> SparseQASessionRecord:
    return SparseQASessionRecord(
        ordinal,
        runtime.identity_sha256,
        "ERROR",
        None,
        code,
    )


def _request(value: object) -> tuple[RawQuestionRequest, bool] | str:
    if not isinstance(value, dict):
        return "INVALID_FIELDS"
    allowed = {"audit", "question", "source_ref"}
    if (not set(value).issubset(allowed) or "question" not in value):
        return "INVALID_FIELDS"
    question = value["question"]
    if (not isinstance(question, str) or not question
            or question.strip() != question):
        return "INVALID_QUESTION"
    source_value = value.get("source_ref")
    source = None
    if source_value is not None:
        if (not isinstance(source_value, list) or not source_value
                or any(type(item) is not int for item in source_value)):
            return "INVALID_SOURCE_REF"
        source = tuple(source_value)
    audit = value.get("audit", False)
    if type(audit) is not bool:
        return "INVALID_AUDIT"
    return RawQuestionRequest(question, source), audit


def iter_sparse_qa_jsonl_session(
        runtime: SparseQARuntime,
        lines: Iterable[str],
        ) -> Iterator[SparseQASessionRecord]:
    """Yield one isolated record per input line without rebuilding runtime."""
    if not isinstance(runtime, SparseQARuntime):
        raise TypeError("FT23 session runtime is invalid")
    for ordinal, source_line in enumerate(lines):
        if not isinstance(source_line, str):
            yield _error(runtime, ordinal, "INVALID_JSON")
            continue
        line = source_line.rstrip("\r\n")
        if ordinal == 0:
            line = line.removeprefix("\ufeff")
        if not line:
            yield _error(runtime, ordinal, "EMPTY_LINE")
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            yield _error(runtime, ordinal, "INVALID_JSON")
            continue
        parsed = _request(value)
        if isinstance(parsed, str):
            yield _error(runtime, ordinal, parsed)
            continue
        request, audit = parsed
        result = run_sparse_qa_query(runtime, request, audit=audit)
        yield SparseQASessionRecord(
            ordinal,
            runtime.identity_sha256,
            "RESULT",
            result,
            None,
        )


def start_sparse_qa_session(
        runtime: SparseQARuntime,
        ) -> SparseQASessionState:
    """Create constant-size rolling state for an already built runtime."""
    if not isinstance(runtime, SparseQARuntime):
        raise TypeError("FT23 session runtime is invalid")
    empty = _sha([])
    return SparseQASessionState(
        runtime.identity_sha256,
        runtime.build_probe.runtime_build_count,
        0,
        0,
        0,
        0,
        0,
        0,
        empty,
        empty,
    )


def advance_sparse_qa_session(
        state: SparseQASessionState,
        record: SparseQASessionRecord,
        ) -> SparseQASessionState:
    """Advance counts and rolling commitments without retaining old results."""
    if (not isinstance(state, SparseQASessionState)
            or not isinstance(record, SparseQASessionRecord)
            or record.runtime_identity_sha256
            != state.runtime_identity_sha256
            or record.line_ordinal != state.input_line_count):
        raise TypeError("FT23 session advance inputs are invalid")
    result = record.result
    return SparseQASessionState(
        state.runtime_identity_sha256,
        state.runtime_build_count,
        state.input_line_count + 1,
        state.query_count + (result is not None),
        state.error_count + (result is None),
        state.audit_projection_count + (
            result is not None and result.audit_result is not None),
        state.created_sparse_trace_count + (
            0 if result is None else result.dispatch_probe.sparse_trace_count),
        state.created_audit_trace_count + (
            0 if result is None or result.audit_result is None
            else len(result.audit_result.traces)
        ),
        _sha({
            "prior_sha256": state.record_commitment_sha256,
            "record_sha256": record.sha256(),
        }),
        (
            state.result_commitment_sha256
            if result is None
            else _sha({
                "prior_sha256": state.result_commitment_sha256,
                "result_sha256": result.sha256(),
            })
        ),
    )


def finish_sparse_qa_session(
        state: SparseQASessionState,
        ) -> SparseQASessionProbe:
    """Freeze a session probe from constant-size rolling state."""
    if not isinstance(state, SparseQASessionState):
        raise TypeError("FT23 session state is invalid")
    return SparseQASessionProbe(
        state.runtime_identity_sha256,
        state.runtime_build_count,
        state.input_line_count,
        state.query_count,
        state.error_count,
        state.audit_projection_count,
        state.created_sparse_trace_count,
        state.created_audit_trace_count,
        state.record_commitment_sha256,
        state.result_commitment_sha256,
    )


def build_sparse_qa_session_probe(
        runtime: SparseQARuntime,
        records: tuple[SparseQASessionRecord, ...],
        ) -> SparseQASessionProbe:
    """Freeze deterministic session counts and repeated-request equality."""
    if (not isinstance(runtime, SparseQARuntime)
            or not isinstance(records, tuple)
            or any(not isinstance(item, SparseQASessionRecord)
                   for item in records)
            or tuple(item.line_ordinal for item in records)
            != tuple(range(len(records)))
            or any(item.runtime_identity_sha256 != runtime.identity_sha256
                   for item in records)):
        raise TypeError("FT23 session probe inputs are invalid")
    state = start_sparse_qa_session(runtime)
    for record in records:
        state = advance_sparse_qa_session(state, record)
    return finish_sparse_qa_session(state)


__all__ = [
    "SPARSE_QA_SESSION_ERROR_CODES",
    "SPARSE_QA_SESSION_KINDS",
    "SparseQASessionProbe",
    "SparseQASessionRecord",
    "SparseQASessionState",
    "W03W04W05SparseQASessionError",
    "advance_sparse_qa_session",
    "build_sparse_qa_session_probe",
    "finish_sparse_qa_session",
    "iter_sparse_qa_jsonl_session",
    "start_sparse_qa_session",
]
