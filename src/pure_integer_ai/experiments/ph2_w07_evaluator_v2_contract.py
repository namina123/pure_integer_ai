"""W-07 evaluator v2 的公开诊断游标与安全 aggregate 合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_PUBLIC_ABLATION_KEYS,
    W07_PUBLIC_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_contract import (
    W07_EVALUATOR_FAILURE_PHASES,
    W07PrivateDimensionResult,
    W07PrivateEvaluationError,
)


W07_V2_EVALUATOR_VERSION = 2
W07_V2_PRIVATE_AGGREGATE_NAME = "private_evaluation_v2_aggregate.json"
W07_V2_PRIVATE_RECOMMENDATION_NAME = "runtime_receipt_v2_recommendation.json"
W07_V2_NONE = "NONE"
W07_V2_OPERATIONS = (
    "ENTER_PHASE",
    "DECODE_DOCUMENTS",
    "CREATE_EXECUTION_ROOT",
    "LOAD_CLONE",
    "COPY_HOST_MANIFEST",
    "COMPARE_CLONE",
    "BUILD_SUITE",
    "COMMIT_LEDGERS",
    "AUDIT_LEDGERS",
    "CLOSE_LEDGERS",
    "EVALUATE_CASE",
    "ASSEMBLE_ABLATION",
    "AUDIT_OWNERS",
    "ASSEMBLE_REPORT",
    "ENCODE_REPORT",
    "PUBLISH_REPORT",
)
W07_V2_FAILURE_KINDS = (
    W07_V2_NONE,
    "INJECTED",
    "STORAGE",
    "RESOURCE",
    "DOMAIN_CONTRACT",
    "TYPE_VALUE",
    "OS",
    "MEMORY",
    "UNEXPECTED",
)


def _strict_sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W07PrivateEvaluationError(f"{label} is not canonical SHA-256")
    return value


def _validate_result_prefix(
        results: tuple[W07PrivateDimensionResult, ...],
        *,
        label: str,
        ) -> None:
    if (not isinstance(results, tuple)
            or any(not isinstance(item, W07PrivateDimensionResult)
                   for item in results)
            or tuple(item.dimension_key for item in results)
            != W07_PUBLIC_DIMENSION_KEYS[:len(results)]):
        raise W07PrivateEvaluationError(f"{label} result prefix drift")


@dataclass(frozen=True)
class W07V2DiagnosticCursor:
    phase: str
    operation: str
    ablation_key: str = W07_V2_NONE
    dimension_key: str = W07_V2_NONE

    def __post_init__(self) -> None:
        if self.phase not in W07_EVALUATOR_FAILURE_PHASES:
            raise W07PrivateEvaluationError("W-07 v2 cursor phase drift")
        if self.operation not in W07_V2_OPERATIONS:
            raise W07PrivateEvaluationError("W-07 v2 cursor operation drift")
        if self.ablation_key not in {W07_V2_NONE, *W07_PUBLIC_ABLATION_KEYS}:
            raise W07PrivateEvaluationError("W-07 v2 cursor ablation drift")
        if self.dimension_key not in {W07_V2_NONE, *W07_PUBLIC_DIMENSION_KEYS}:
            raise W07PrivateEvaluationError("W-07 v2 cursor dimension drift")

    def to_safe_dict(self) -> dict[str, str]:
        return {
            "ablation_key": self.ablation_key,
            "dimension_key": self.dimension_key,
            "operation": self.operation,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class W07V2AblationProgress:
    ablation_key: str
    results: tuple[W07PrivateDimensionResult, ...]

    def __post_init__(self) -> None:
        if self.ablation_key not in W07_PUBLIC_ABLATION_KEYS:
            raise W07PrivateEvaluationError("W-07 v2 ablation key drift")
        _validate_result_prefix(self.results, label="ablation")

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "ablation_key": self.ablation_key,
            "dimension_results": [
                item.to_safe_dict() for item in self.results],
        }


def public_safe_w07_v2_aggregate(
        baseline_results: tuple[W07PrivateDimensionResult, ...],
        ablation_results: tuple[W07V2AblationProgress, ...],
        *,
        family_commitment: str,
        payload_commitment: str,
        case_commitment: str,
        label_commitment: str,
        cluster_commitment: str,
        formal_run_count: int,
        host_writes: int,
        label_writes: int,
        public_repo_writes: int,
        failure_kind: str,
        cursor: W07V2DiagnosticCursor,
        ablation_gates_passed: bool,
        ) -> dict[str, object]:
    """保留安全结果前缀与精确公开游标，不导出 case、label 或消息。"""
    _validate_result_prefix(baseline_results, label="baseline")
    if (not isinstance(ablation_results, tuple)
            or any(not isinstance(item, W07V2AblationProgress)
                   for item in ablation_results)
            or tuple(item.ablation_key for item in ablation_results)
            != W07_PUBLIC_ABLATION_KEYS[:len(ablation_results)]):
        raise W07PrivateEvaluationError("W-07 v2 ablation progress drift")
    for label, value in (
            ("family", family_commitment), ("payload", payload_commitment),
            ("case", case_commitment), ("label", label_commitment),
            ("cluster", cluster_commitment)):
        _strict_sha256(value, label=label)
    if formal_run_count != 1 or type(formal_run_count) is not int:
        raise W07PrivateEvaluationError("W-07 v2 formal run count drift")
    if any(value not in {0, 1} for value in (
            host_writes, label_writes, public_repo_writes)):
        raise W07PrivateEvaluationError("W-07 v2 write count drift")
    if failure_kind not in W07_V2_FAILURE_KINDS:
        raise W07PrivateEvaluationError("W-07 v2 failure kind drift")
    if not isinstance(cursor, W07V2DiagnosticCursor):
        raise TypeError("W-07 v2 cursor type drift")
    if type(ablation_gates_passed) is not bool:
        raise TypeError("W-07 v2 gate state drift")

    completed = (
        len(baseline_results) == len(W07_PUBLIC_DIMENSION_KEYS)
        and len(ablation_results) == len(W07_PUBLIC_ABLATION_KEYS)
        and all(len(item.results) == len(W07_PUBLIC_DIMENSION_KEYS)
                for item in ablation_results)
    )
    baseline_passes = sum(item.passed for item in baseline_results)
    baseline_fails = sum(item.fail_count for item in baseline_results)
    if failure_kind != W07_V2_NONE:
        status, passed, fails, ne = "NE", 0, 0, 1
    elif not completed:
        raise W07PrivateEvaluationError(
            "W-07 v2 success aggregate is incomplete")
    elif (baseline_passes == len(W07_PUBLIC_DIMENSION_KEYS)
          and not baseline_fails and ablation_gates_passed
          and not any((host_writes, label_writes, public_repo_writes))):
        status, passed, fails, ne = (
            "PASS", baseline_passes, 0, 0)
    else:
        status, passed, fails, ne = (
            "FAIL", baseline_passes, max(1, baseline_fails), 0)
    return {
        "ablation_results": [
            item.to_safe_dict() for item in ablation_results],
        "artifact_kind": "PH2_W07_PRIVATE_EVALUATION_V2_AGGREGATE",
        "baseline_results": [
            item.to_safe_dict() for item in baseline_results],
        "case_commitment": case_commitment,
        "cluster_commitment": cluster_commitment,
        "diagnostic_cursor": cursor.to_safe_dict(),
        "evaluator_version": W07_V2_EVALUATOR_VERSION,
        "fail_count": fails,
        "failure_kind": failure_kind,
        "family_commitment": family_commitment,
        "formal_run_count": formal_run_count,
        "format_version": 2,
        "host_writes": host_writes,
        "infrastructure": {},
        "label_commitment": label_commitment,
        "label_writes": label_writes,
        "ne_count": ne,
        "pass_count": passed,
        "payload_commitment": payload_commitment,
        "public_repo_writes": public_repo_writes,
        "status": status,
    }


__all__ = [name for name in globals() if name.startswith("W07_V2_")] + [
    "W07V2AblationProgress",
    "W07V2DiagnosticCursor",
    "public_safe_w07_v2_aggregate",
]
