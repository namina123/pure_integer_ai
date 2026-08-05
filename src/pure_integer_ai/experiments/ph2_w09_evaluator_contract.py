"""W09-10 private evaluator 的安全结果合同与不可泄漏 aggregate。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_ABLATION_KEYS,
    W09_ALL_DIMENSION_KEYS,
    W09_DIMENSION_KEYS,
    W09_WALL_DIMENSION_KEYS,
)

W09_PRIVATE_OWNER_KEY = "PH2_W09_PRIVATE_EVALUATOR_OWNER"
W09_PRIVATE_FAMILY_FREEZE_NAME = "w09_private_family_freeze.json"
W09_PRIVATE_FIRST_RUN_GUARD_NAME = "w09_private_formal_first_run_guard.json"
W09_PRIVATE_SOURCE_NAME = "w09_private_source.json"
W09_PRIVATE_SCHEMA_NAME = "w09_private_schema.json"
W09_PRIVATE_CASE_NAME = "w09_private_cases.json"
W09_PRIVATE_LABEL_NAME = "w09_private_labels.json"
W09_PRIVATE_CLUSTER_NAME = "w09_private_clusters.json"
W09_PRIVATE_AGGREGATE_NAME = "w09_private_evaluation_aggregate.json"
W09_PRIVATE_RECOMMENDATION_NAME = "w09_private_runtime_receipt_recommendation.json"
W09_PRIVATE_DUMP_NAME = "w09_private_evaluation_dump.json"
W09_PRIVATE_TERMINAL_SEAL_NAME = "w09_private_terminal_seal.json"
W09_PRIVATE_INFERENCE_INTERFACE_VERSION = "PH2-W09-INFERENCE-V1"

W09_EVALUATOR_PHASES = (
    "CANDIDATE_VERIFY",
    "CODE_FREEZE_VERIFY",
    "FAMILY_METADATA_VERIFY",
    "OBSERVATION_READ",
    "INFERENCE_INVENTORY",
    "LABEL_READ",
    "BASELINE_EVALUATION",
    "ABLATION_EVALUATION",
    "ZERO_CALL_WINDOWS",
    "J_LC_W09",
    "V06_CLONE",
    "ROLLBACK",
    "RESOURCE",
    "DUMP_READBACK",
    "INTEGRITY",
    "REPORT_SAFETY",
)
W09_EVALUATOR_FAILURE_PHASES = ("NONE", *W09_EVALUATOR_PHASES)
W09_EVALUATOR_THRESHOLD = {
    "max_fail_count": 0,
    "min_pass_denominator": 1,
    "min_pass_numerator": 1,
    "ne_policy": "BLOCK",
}
W09_PRIVATE_HARD_CONJUNCT_KEYS = (
    *W09_DIMENSION_KEYS,
    "OPEN-GENERATION",
    "W-09-TEACHER_ZERO_WINDOW",
    "J-LC-W09",
    "V-06-CLONE",
    "ROLLBACK-AUDIT",
    "RESOURCE",
)


class W09PrivateEvaluationError(RuntimeError):
    """W09 private family、评估结果或安全报告发生漂移。"""


def strict_sha256(value: object, *, label: str = "value") -> str:
    """校验公开 commitment 必须是小写 SHA-256。"""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise W09PrivateEvaluationError(f"W09 {label} 不是规范 SHA-256")
    return value


def strict_sha1(value: object, *, label: str = "value") -> str:
    """校验 public HEAD 必须是小写 Git SHA-1。"""
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise W09PrivateEvaluationError(f"W09 {label} 不是规范 Git SHA-1")
    return value


def evidence_commitment(value: object) -> str:
    """对安全 typed 对象生成确定性 SHA-256 commitment。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class W09PrivateDimensionResult:
    """单个承重维度的逐 case 计数与安全 Evidence commitment。"""

    dimension_key: str
    status: str
    passed_count: int
    required_count: int
    fail_count: int
    ne_count: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        """校验状态只能由计数硬合取产生。"""
        if self.dimension_key not in W09_DIMENSION_KEYS:
            raise W09PrivateEvaluationError("W09 承重维度未登记")
        if self.status not in {"PASS", "FAIL", "NE"}:
            raise W09PrivateEvaluationError("W09 dimension status 非法")
        values = (self.passed_count, self.required_count, self.fail_count, self.ne_count)
        if any(type(item) is not int or item < 0 for item in values) or self.required_count <= 0:
            raise W09PrivateEvaluationError("W09 dimension counts 非法")
        if self.passed_count + self.fail_count + self.ne_count != self.required_count:
            raise W09PrivateEvaluationError("W09 dimension counts 不闭合")
        expected = (
            "PASS" if self.passed_count == self.required_count and not self.fail_count and not self.ne_count
            else "FAIL" if self.fail_count else "NE"
        )
        if self.status != expected:
            raise W09PrivateEvaluationError("W09 dimension status 与计数漂移")
        strict_sha256(self.evidence_sha256, label="dimension evidence")

    def to_safe_dict(self) -> dict[str, object]:
        """返回不含 case key、label 或文本的公开结果。"""
        return {
            "dimension_key": self.dimension_key,
            "evidence_sha256": self.evidence_sha256,
            "fail_count": self.fail_count,
            "ne_count": self.ne_count,
            "passed_count": self.passed_count,
            "required_count": self.required_count,
            "status": self.status,
        }


_FORBIDDEN_REPORT_KEYS = frozenset({
    "case", "case_key", "expected", "expected_payload", "expected_state",
    "label", "label_key", "message", "path", "private_path", "relative_path",
    "surface", "surface_form", "text", "typed_payload", "raw_text",
    "raw_observation", "observed_surface", "exception", "error_message",
})


def validate_w09_safe_report(value: object) -> None:
    """递归检查 aggregate/receipt 不能携带私有字段或路径。"""
    if isinstance(value, dict):
        if any(str(key).lower() in _FORBIDDEN_REPORT_KEYS for key in value):
            raise W09PrivateEvaluationError("W09 aggregate 泄漏 private 字段")
        for item in value.values():
            validate_w09_safe_report(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_w09_safe_report(item)


def _safe_status(value: object, *, allow_public_bounded: bool = False) -> str:
    """校验辅助结果状态枚举。"""
    allowed = {"PASS", "FAIL", "NE"}
    if allow_public_bounded:
        allowed.update({"PUBLIC_BOUNDED_PASS", "PUBLIC_BOUNDED_NOT_FORMAL", "RETENTION_EVIDENCED"})
    if value not in allowed:
        raise W09PrivateEvaluationError("W09 auxiliary status 非法")
    return str(value)


def public_safe_w09_aggregate(
    results: tuple[W09PrivateDimensionResult, ...],
    *,
    family_commitment: str,
    payload_commitment: str,
    case_commitment: str,
    label_commitment: str,
    cluster_commitment: str,
    rotation_package_commitment: str,
    failure_phase: str,
    formal_run_count: int,
    write_counts: dict[str, int],
    ablation_results: list[dict[str, object]] | None = None,
    windows: list[dict[str, object]] | None = None,
    open_generation: dict[str, object] | None = None,
    j_lc: dict[str, object] | None = None,
    v06: dict[str, object] | None = None,
    rollback: dict[str, object] | None = None,
    resource: dict[str, object] | None = None,
    infrastructure: dict[str, object] | None = None,
) -> dict[str, object]:
    """生成只含 commitment/count/state 的 W09 aggregate。"""
    for label, value in {
        "family": family_commitment,
        "payload": payload_commitment,
        "case": case_commitment,
        "label": label_commitment,
        "cluster": cluster_commitment,
        "rotation package": rotation_package_commitment,
    }.items():
        strict_sha256(value, label=label)
    if failure_phase not in W09_EVALUATOR_FAILURE_PHASES:
        raise W09PrivateEvaluationError("W09 failure phase 未登记")
    if type(formal_run_count) is not int or formal_run_count != 1:
        raise W09PrivateEvaluationError("W09 private formal run count 必须为一")
    expected_writes = {
        "candidate_writes", "label_writes", "public_writes", "host_writes",
        "core_writes", "evidence_writes", "use_writes", "memory_writes",
        "assessment_writes", "clock_writes",
    }
    if set(write_counts) != expected_writes or any(type(item) is not int or item < 0 for item in write_counts.values()):
        raise W09PrivateEvaluationError("W09 evaluator write account 非法")
    failed_phase = failure_phase != "NONE"
    safe_results = () if failed_phase else results
    if not failed_phase and tuple(item.dimension_key for item in results) != W09_DIMENSION_KEYS:
        raise W09PrivateEvaluationError("W09 五维结果顺序漂移")
    dimension_status = "NE" if failed_phase else (
        "FAIL" if any(item.status == "FAIL" for item in results)
        else "NE" if any(item.status == "NE" for item in results)
        else "PASS"
    )
    safe_ablations = list(ablation_results or []) if not failed_phase else []
    safe_windows = list(windows or []) if not failed_phase else []
    safe_open_generation = dict(open_generation or {}) if not failed_phase else {}
    safe_jlc = dict(j_lc or {}) if not failed_phase else {}
    safe_v06 = dict(v06 or {}) if not failed_phase else {}
    safe_rollback = dict(rollback or {}) if not failed_phase else {}
    safe_resource = dict(resource or {}) if not failed_phase else {}
    safe_infrastructure = dict(infrastructure or {})
    if failed_phase:
        status = "NE"
    else:
        explicit_fail = any(item.get("status") == "FAIL" for item in (
            *safe_ablations,
            safe_open_generation,
            safe_jlc,
            safe_v06,
            safe_rollback,
            safe_resource,
        ))
        incomplete = (
            dimension_status != "PASS"
            or len(safe_ablations) != len(W09_ABLATION_KEYS)
            or len(safe_windows) != 3
            or safe_open_generation.get("status") != "PASS"
            or safe_jlc.get("status") != "PASS"
            or safe_v06.get("status") != "PASS"
            or safe_rollback.get("status") != "PASS"
            or safe_resource.get("status") != "PASS"
        )
        status = "FAIL" if explicit_fail else "NE" if incomplete else "PASS"
    result = {
        "ablation_results": safe_ablations,
        "artifact_kind": "PH2_W09_PRIVATE_EVALUATION_AGGREGATE",
        "case_commitment": case_commitment,
        "cluster_commitment": cluster_commitment,
        "dimension_results": [item.to_safe_dict() for item in safe_results],
        "fail_count": int(status == "FAIL"),
        "failure_phase": failure_phase,
        "family_commitment": family_commitment,
        "formal_run_count": formal_run_count,
        "format_version": 1,
        "infrastructure": safe_infrastructure,
        "j_lc": safe_jlc,
        "label_commitment": label_commitment,
        "ne_count": int(status == "NE"),
        "open_generation": safe_open_generation,
        "payload_commitment": payload_commitment,
        "pre_wean_language_learning_capability_evidenced": int(status == "PASS"),
        "resource": safe_resource,
        "rollback": safe_rollback,
        "rotation_package_commitment": rotation_package_commitment,
        "status": status,
        "v06": safe_v06,
        "windows": safe_windows,
        "write_counts": dict(sorted(write_counts.items())),
    }
    validate_w09_safe_report(result)
    return result


__all__ = [
    "W09_EVALUATOR_FAILURE_PHASES",
    "W09_EVALUATOR_PHASES",
    "W09_EVALUATOR_THRESHOLD",
    "W09_PRIVATE_AGGREGATE_NAME",
    "W09_PRIVATE_CASE_NAME",
    "W09_PRIVATE_CLUSTER_NAME",
    "W09_PRIVATE_DUMP_NAME",
    "W09_PRIVATE_FAMILY_FREEZE_NAME",
    "W09_PRIVATE_FIRST_RUN_GUARD_NAME",
    "W09_PRIVATE_INFERENCE_INTERFACE_VERSION",
    "W09_PRIVATE_LABEL_NAME",
    "W09_PRIVATE_OWNER_KEY",
    "W09_PRIVATE_RECOMMENDATION_NAME",
    "W09_PRIVATE_SCHEMA_NAME",
    "W09_PRIVATE_SOURCE_NAME",
    "W09_PRIVATE_TERMINAL_SEAL_NAME",
    "W09PrivateDimensionResult",
    "W09PrivateEvaluationError",
    "evidence_commitment",
    "public_safe_w09_aggregate",
    "strict_sha1",
    "strict_sha256",
    "validate_w09_safe_report",
]
