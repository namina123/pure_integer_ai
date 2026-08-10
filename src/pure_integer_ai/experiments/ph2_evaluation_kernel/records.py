"""Small immutable records and fail-closed four-state evaluation results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03ContractError,
    enum_text,
    exact_dict,
    nonnegative,
    sha256_text,
    text,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_ZERO_CALL_WINDOW_COUNT,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2WriteAccount,
)


EVALUATION_RESULT_STATUSES = ("PASS", "FAIL", "NE", "BLOCKED")
EVALUATION_RESULT_ROLES = ("BEARING", "GENERATION", "SUPPORT", "ABLATION")


# object-model: exception
class EvaluationKernelContractError(D03ContractError):
    """A generic evaluation record or immutable binding drifted."""


def evaluation_status_from_counts(
        *,
        planned_count: int,
        passed_count: int,
        failed_count: int,
        not_evaluated_count: int,
        blocked_count: int,
        ) -> str:
    """Derive one status without averaging away a blocking or failed record."""
    values = (
        planned_count, passed_count, failed_count,
        not_evaluated_count, blocked_count,
    )
    for name, value in zip((
            "planned_count", "passed_count", "failed_count",
            "not_evaluated_count", "blocked_count"), values, strict=True):
        nonnegative(value, where=name)
    if planned_count <= 0:
        raise EvaluationKernelContractError("evaluation planned_count must be positive")
    if sum(values[1:]) != planned_count:
        raise EvaluationKernelContractError(
            "evaluation result counts must exactly consume planned_count")
    if blocked_count:
        return "BLOCKED"
    if failed_count:
        return "FAIL"
    if not_evaluated_count:
        return "NE"
    if passed_count == planned_count:
        return "PASS"
    raise EvaluationKernelContractError("evaluation result status is not derivable")


def evaluation_status_from_results(
        results: Iterable["EvaluationDimensionResult"],
        ) -> str:
    """Derive the hard-conjunct status using BLOCKED > FAIL > NE > PASS."""
    statuses = tuple(result.status for result in results)
    if not statuses:
        raise EvaluationKernelContractError("evaluation result set cannot be empty")
    for status in ("BLOCKED", "FAIL", "NE"):
        if status in statuses:
            return status
    if all(status == "PASS" for status in statuses):
        return "PASS"
    raise EvaluationKernelContractError("evaluation result set contains invalid status")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class EvaluationDimensionResult:
    """One preregistered result with independent PASS/FAIL/NE/BLOCKED counts."""

    result_key: str
    role: str
    status: str
    planned_count: int
    passed_count: int
    failed_count: int
    not_evaluated_count: int
    blocked_count: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        text(self.result_key, where="evaluation result_key")
        enum_text(self.role, EVALUATION_RESULT_ROLES, where="evaluation role")
        enum_text(self.status, EVALUATION_RESULT_STATUSES, where="evaluation status")
        expected = evaluation_status_from_counts(
            planned_count=self.planned_count,
            passed_count=self.passed_count,
            failed_count=self.failed_count,
            not_evaluated_count=self.not_evaluated_count,
            blocked_count=self.blocked_count,
        )
        if self.status != expected:
            raise EvaluationKernelContractError(
                "evaluation status does not match its independent counts")
        sha256_text(self.evidence_sha256, where="evaluation evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked_count": self.blocked_count,
            "evidence_sha256": self.evidence_sha256,
            "failed_count": self.failed_count,
            "not_evaluated_count": self.not_evaluated_count,
            "passed_count": self.passed_count,
            "planned_count": self.planned_count,
            "result_key": self.result_key,
            "role": self.role,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationDimensionResult":
        raw = exact_dict(value, {
            "blocked_count", "evidence_sha256", "failed_count",
            "not_evaluated_count", "passed_count", "planned_count",
            "result_key", "role", "status",
        }, where="EvaluationDimensionResult")
        return cls(
            str(raw["result_key"]), str(raw["role"]), str(raw["status"]),
            raw["planned_count"], raw["passed_count"], raw["failed_count"],
            raw["not_evaluated_count"], raw["blocked_count"],
            str(raw["evidence_sha256"]),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationResultSet:
    """Ordered plugin output; result identities may not repeat or reorder."""

    results: tuple[EvaluationDimensionResult, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.results, tuple) or not self.results
                or any(not isinstance(item, EvaluationDimensionResult)
                       for item in self.results)):
            raise EvaluationKernelContractError(
                "evaluation result set must contain immutable result structs")
        keys = tuple(item.result_key for item in self.results)
        if len(keys) != len(set(keys)):
            raise EvaluationKernelContractError("evaluation result keys must be unique")

    @property
    def status(self) -> str:
        return evaluation_status_from_results(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [item.to_dict() for item in self.results],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationResultSet":
        raw = exact_dict(value, {"results", "status"}, where="EvaluationResultSet")
        if not isinstance(raw["results"], list):
            raise EvaluationKernelContractError("evaluation results must be an array")
        result = cls(tuple(
            EvaluationDimensionResult.from_dict(item) for item in raw["results"]))
        if raw["status"] != result.status:
            raise EvaluationKernelContractError("evaluation result-set status drifted")
        return result


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationRunAudit:
    """Safe resource/read/write account emitted with every plugin outcome."""

    audit_state: str
    source_ref_count: int
    pair_count: int
    private_record_reads: int
    private_payload_gets: int
    transport_bytes_read: int
    logic_operations: int
    zero_call_window_count: int
    write_account: V2WriteAccount | None = V2WriteAccount()

    def __post_init__(self) -> None:
        if self.audit_state == "COMPLETE":
            for name in (
                    "source_ref_count", "pair_count", "private_record_reads",
                    "private_payload_gets", "transport_bytes_read",
                    "logic_operations", "zero_call_window_count"):
                nonnegative(getattr(self, name), where=f"evaluation audit {name}")
            if self.source_ref_count <= 0 or self.pair_count <= 0:
                raise EvaluationKernelContractError(
                    "evaluation audit must consume sources and pairs")
            if self.private_record_reads != self.source_ref_count + self.pair_count * 2:
                raise EvaluationKernelContractError(
                    "evaluation audit record reads must close source/observation/label")
            if self.zero_call_window_count != V2_ZERO_CALL_WINDOW_COUNT:
                raise EvaluationKernelContractError("evaluation zero-call windows drifted")
            if not isinstance(self.write_account, V2WriteAccount):
                raise EvaluationKernelContractError("evaluation write account type drifted")
            if not self.write_account.is_zero:
                raise EvaluationKernelContractError("evaluation runtime performed writes")
        elif self.audit_state == "BLOCKED_UNAVAILABLE":
            if any(getattr(self, name) != -1 for name in (
                    "source_ref_count", "pair_count", "private_record_reads",
                    "private_payload_gets", "transport_bytes_read",
                    "logic_operations", "zero_call_window_count")):
                raise EvaluationKernelContractError(
                    "blocked unavailable audit must use explicit -1 sentinels")
            if self.write_account is not None:
                raise EvaluationKernelContractError(
                    "blocked unavailable audit cannot claim a write account")
        else:
            raise EvaluationKernelContractError("evaluation audit state is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_state": self.audit_state,
            "logic_operations": self.logic_operations,
            "pair_count": self.pair_count,
            "private_payload_gets": self.private_payload_gets,
            "private_record_reads": self.private_record_reads,
            "source_ref_count": self.source_ref_count,
            "transport_bytes_read": self.transport_bytes_read,
            "write_account": (
                None if self.write_account is None else self.write_account.to_dict()),
            "zero_call_window_count": self.zero_call_window_count,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationRunAudit":
        raw = exact_dict(value, {
            "audit_state", "logic_operations", "pair_count", "private_payload_gets",
            "private_record_reads", "source_ref_count", "transport_bytes_read",
            "write_account", "zero_call_window_count",
        }, where="EvaluationRunAudit")
        writes = None
        if raw["write_account"] is not None:
            writes = exact_dict(
                raw["write_account"], set(V2WriteAccount().__dataclass_fields__),
                where="EvaluationRunAudit.write_account")
        return cls(
            str(raw["audit_state"]), raw["source_ref_count"], raw["pair_count"],
            raw["private_record_reads"], raw["private_payload_gets"],
            raw["transport_bytes_read"], raw["logic_operations"],
            raw["zero_call_window_count"],
            None if writes is None else V2WriteAccount(**writes),
        )

    @classmethod
    def blocked_unavailable(cls) -> "EvaluationRunAudit":
        """Represent an interrupted run without falsely asserting zero reads/writes."""
        return cls("BLOCKED_UNAVAILABLE", -1, -1, -1, -1, -1, -1, -1, None)


__all__ = [
    "EVALUATION_RESULT_ROLES",
    "EVALUATION_RESULT_STATUSES",
    "EvaluationDimensionResult",
    "EvaluationKernelContractError",
    "EvaluationRunAudit",
    "EvaluationResultSet",
    "evaluation_status_from_counts",
    "evaluation_status_from_results",
]
