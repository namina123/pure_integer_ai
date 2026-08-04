"""W08-09 private evaluator 的结果、故障与公开安全 aggregate 合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w08_authority import (
    W08_ABLATION_KEYS,
    W08_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08_CANDIDATE_INFERENCE_INTERFACE_VERSION,
)


W08_PRIVATE_OWNER_KEY = "PH2_W08_PRIVATE_EVALUATOR_OWNER"
W08_PRIVATE_FAMILY_FREEZE_NAME = "private_family_freeze.json"
W08_PRIVATE_FIRST_RUN_GUARD_NAME = "formal_first_run_guard.json"
W08_PRIVATE_AGGREGATE_NAME = "private_evaluation_aggregate.json"
W08_PRIVATE_RECOMMENDATION_NAME = "runtime_receipt_recommendation.json"
W08_PRIVATE_DUMP_NAME = "private_evaluation_dump.json"
W08_PRIVATE_SOURCE_NAME = "private_source.json"
W08_PRIVATE_SCHEMA_NAME = "private_schema.json"
W08_PRIVATE_CASE_NAME = "private_cases.json"
W08_PRIVATE_LABEL_NAME = "private_labels.json"
W08_PRIVATE_CLUSTER_NAME = "private_clusters.json"
W08_PRIVATE_INFERENCE_INTERFACE_VERSION = W08_CANDIDATE_INFERENCE_INTERFACE_VERSION

W08_EVALUATOR_PHASES = (
    "CANDIDATE_VERIFY",
    "CANDIDATE_DUMP_READBACK",
    "PAYLOAD_READ",
    "PAYLOAD_PAIR",
    "BASELINE",
    "ABLATION_CHINESE_VARIATION",
    "ABLATION_DISCOURSE",
    "ABLATION_LOCAL_RECOMPUTE",
    "ABLATION_LONG_CONTEXT",
    "ABLATION_P3IA",
    "OPEN_GENERATION",
    "LC16",
    "DUMP_READBACK",
    "INTEGRITY",
    "REPORT_SAFETY",
)
W08_EVALUATOR_FAILURE_PHASES = ("NONE", *W08_EVALUATOR_PHASES)
W08_EVALUATOR_THRESHOLD = {
    "max_fail_count": 0,
    "min_pass_denominator": 1,
    "min_pass_numerator": 1,
    "ne_policy": "BLOCK",
}


class W08PrivateEvaluationError(RuntimeError):
    """W08 private family、结果、报告或隔离合同发生漂移。"""


def strict_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise W08PrivateEvaluationError(f"W08 {label} 不是规范 SHA-256")
    return value


def evidence_commitment(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class W08PrivateDimensionResult:
    """只含承重维状态、计数与 Evidence commitment 的安全结果。"""

    dimension_key: str
    status: str
    passed_count: int
    required_count: int
    fail_count: int
    ne_count: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        if (
            self.dimension_key not in W08_DIMENSION_KEYS
            or self.status not in {"PASS", "FAIL", "NE"}
            or type(self.required_count) is not int
            or self.required_count <= 0
            or any(
                type(value) is not int or value < 0
                for value in (self.passed_count, self.fail_count, self.ne_count)
            )
            or self.passed_count + self.fail_count + self.ne_count
            != self.required_count
        ):
            raise W08PrivateEvaluationError("W08 dimension result 字段非法")
        expected = (
            "PASS"
            if self.passed_count == self.required_count
            and self.fail_count == self.ne_count == 0
            else "FAIL"
            if self.fail_count
            else "NE"
        )
        if self.status != expected:
            raise W08PrivateEvaluationError("W08 dimension 状态与计数漂移")
        strict_sha256(self.evidence_sha256, label="dimension evidence")

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "dimension_key": self.dimension_key,
            "evidence_sha256": self.evidence_sha256,
            "fail_count": self.fail_count,
            "ne_count": self.ne_count,
            "passed_count": self.passed_count,
            "required_count": self.required_count,
            "status": self.status,
        }


def public_safe_w08_aggregate(
    results: tuple[W08PrivateDimensionResult, ...],
    *,
    family_commitment: str,
    payload_commitment: str,
    case_commitment: str,
    label_commitment: str,
    cluster_commitment: str,
    failure_phase: str,
    formal_run_count: int,
    write_counts: dict[str, int],
    ablation_results: list[dict[str, object]] | None = None,
    open_generation: dict[str, object] | None = None,
    lc16: dict[str, object] | None = None,
    infrastructure: dict[str, object] | None = None,
) -> dict[str, object]:
    """只发布 commitment/count/state，不发布 case、label、surface 或路径。"""
    commitments = {
        "case_commitment": strict_sha256(case_commitment, label="case commitment"),
        "cluster_commitment": strict_sha256(
            cluster_commitment, label="cluster commitment"
        ),
        "family_commitment": strict_sha256(
            family_commitment, label="family commitment"
        ),
        "label_commitment": strict_sha256(label_commitment, label="label commitment"),
        "payload_commitment": strict_sha256(
            payload_commitment, label="payload commitment"
        ),
    }
    if failure_phase not in W08_EVALUATOR_FAILURE_PHASES:
        raise W08PrivateEvaluationError("W08 failure phase 非法")
    if formal_run_count != 1 or type(formal_run_count) is not int:
        raise W08PrivateEvaluationError("W08 private formal run count 必须为一")
    expected_write_keys = {
        "candidate_writes",
        "label_writes",
        "public_writes",
    }
    if (
        set(write_counts) != expected_write_keys
        or any(type(value) is not int or value not in {0, 1} for value in write_counts.values())
    ):
        raise W08PrivateEvaluationError("W08 evaluator write account 非法")
    if failure_phase != "NONE":
        safe_results: tuple[W08PrivateDimensionResult, ...] = ()
        status = "NE"
        passed = fails = 0
        ne = 1
        ablations: list[dict[str, object]] = []
        open_safe: dict[str, object] = {}
        lc16_safe: dict[str, object] = {}
        infrastructure_safe = dict(infrastructure or {})
    else:
        if (
            tuple(item.dimension_key for item in results) != W08_DIMENSION_KEYS
            or any(not isinstance(item, W08PrivateDimensionResult) for item in results)
        ):
            raise W08PrivateEvaluationError("W08 五维结果顺序漂移")
        safe_results = results
        passed = sum(item.status == "PASS" for item in results)
        fails = sum(item.status == "FAIL" for item in results)
        ne = sum(item.status == "NE" for item in results)
        ablations = list(ablation_results or [])
        open_safe = dict(open_generation or {})
        lc16_safe = dict(lc16 or {})
        infrastructure_safe = dict(infrastructure or {})
        explicit_failure = any((
            bool(fails),
            any(item.get("status") == "FAIL" for item in ablations),
            open_safe.get("status") == "FAIL",
            lc16_safe.get("status") == "FAIL",
            any(write_counts.values()),
        ))
        incomplete = any((
            bool(ne),
            len(ablations) != len(W08_ABLATION_KEYS),
            any(item.get("status") != "PASS" for item in ablations),
            open_safe.get("status") != "PASS",
            lc16_safe.get("status") != "PASS",
        ))
        if explicit_failure:
            status = "FAIL"
            fails = max(fails, 1)
        elif incomplete:
            status = "NE"
            ne = max(ne, 1)
        else:
            status = "PASS"
    return {
        "ablation_results": ablations,
        "artifact_kind": "PH2_W08_PRIVATE_EVALUATION_AGGREGATE",
        **commitments,
        "dimension_results": [item.to_safe_dict() for item in safe_results],
        "fail_count": fails,
        "failure_phase": failure_phase,
        "formal_run_count": formal_run_count,
        "format_version": 1,
        "infrastructure": infrastructure_safe,
        "lc16": lc16_safe,
        "ne_count": ne,
        "open_generation": open_safe,
        "pass_count": passed,
        "status": status,
        "write_counts": dict(sorted(write_counts.items())),
    }


__all__ = [
    "W08_EVALUATOR_FAILURE_PHASES",
    "W08_EVALUATOR_PHASES",
    "W08_EVALUATOR_THRESHOLD",
    "W08_PRIVATE_AGGREGATE_NAME",
    "W08_PRIVATE_CASE_NAME",
    "W08_PRIVATE_CLUSTER_NAME",
    "W08_PRIVATE_DUMP_NAME",
    "W08_PRIVATE_FAMILY_FREEZE_NAME",
    "W08_PRIVATE_FIRST_RUN_GUARD_NAME",
    "W08_PRIVATE_LABEL_NAME",
    "W08_PRIVATE_INFERENCE_INTERFACE_VERSION",
    "W08_PRIVATE_OWNER_KEY",
    "W08_PRIVATE_RECOMMENDATION_NAME",
    "W08_PRIVATE_SCHEMA_NAME",
    "W08_PRIVATE_SOURCE_NAME",
    "W08PrivateDimensionResult",
    "W08PrivateEvaluationError",
    "evidence_commitment",
    "public_safe_w08_aggregate",
    "strict_sha256",
]
