"""PH2 W-03 private evaluator 的 owner 文档、结果与安全 aggregate 合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w03_contract import W03_EVALUATION_ORDER


W03_PRIVATE_OWNER_KEY = "PH2_W03_PRIVATE_EVALUATOR_OWNER"
W03_PRIVATE_FAMILY_FREEZE_NAME = "private_family_freeze.json"
W03_PRIVATE_FIRST_RUN_GUARD_NAME = "formal_first_run_guard.json"
W03_PRIVATE_AGGREGATE_NAME = "private_evaluation_aggregate.json"
W03_PRIVATE_RECOMMENDATION_NAME = "runtime_receipt_recommendation.json"
W03_EVALUATOR_PHASES = (
    "PAYLOAD_DECODE",
    "CLONE_LOAD",
    "HOST_COPY",
    "CLONE_COMPARE",
    "BASELINE",
    "ABLATION_CONCEPT_SPLIT",
    "ABLATION_POLYSEMY_COMPETITION",
    "ABLATION_SOURCE_CONFLICT",
    "ABLATION_SUPERSEDE",
    "GENERATION",
    "INTEGRITY",
    "REPORT_SAFETY",
)
W03_EVALUATOR_FAILURE_PHASES = ("NONE", *W03_EVALUATOR_PHASES)


class W03PrivateEvaluationError(RuntimeError):
    """private 文档、能力结果或安全报告违反冻结合同。"""


def _strict_sha256(value: object, *, label: str) -> str:
    """验证规范小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise W03PrivateEvaluationError(f"{label} 不是规范 SHA-256")
    return value


def _strict_key(value: object, *, label: str) -> tuple[int, ...]:
    """验证非空纯整数稳定键。"""
    if (not isinstance(value, list) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise W03PrivateEvaluationError(f"{label} 不是非空纯整数键")
    return tuple(value)


def _decode(payload: bytes, *, label: str) -> dict[str, Any]:
    """严格解码 canonical JSON object，拒绝 float 和额外编码差异。"""
    if not isinstance(payload, bytes) or not payload:
        raise W03PrivateEvaluationError(f"{label} payload 为空")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise W03PrivateEvaluationError(f"{label} payload 无法解析") from exc
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise W03PrivateEvaluationError(f"{label} payload 含非法值") from exc
    if not isinstance(value, dict) or canonical != payload:
        raise W03PrivateEvaluationError(f"{label} payload 不是 canonical object")
    return value


@dataclass(frozen=True)
class W03PrivateSource:
    """独立 evaluator owner 的 CC0 source 与 candidate 后置身份。"""

    family_key: str
    source_key: tuple[int, ...]
    owner_key: str
    license_id: str
    candidate_contract_sha256: str
    candidate_host_freeze_sha256: str
    nonce_commitment: str

    def __post_init__(self) -> None:
        _strict_sha256(self.family_key, label="private family key")
        if (not isinstance(self.source_key, tuple) or not self.source_key
                or any(type(item) is not int or item < 0
                       for item in self.source_key)):
            raise W03PrivateEvaluationError("private source key 非法")
        if self.owner_key != W03_PRIVATE_OWNER_KEY or self.license_id != "CC0-1.0":
            raise W03PrivateEvaluationError("private source owner/license 漂移")
        _strict_sha256(
            self.candidate_contract_sha256, label="candidate contract")
        _strict_sha256(
            self.candidate_host_freeze_sha256, label="candidate host freeze")
        _strict_sha256(self.nonce_commitment, label="private nonce commitment")


@dataclass(frozen=True)
class W03PrivateCase:
    """不含 surface/expected 的一个 private dimension challenge。"""

    case_key: str
    dimension_key: str
    challenge_key: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_sha256(self.case_key, label="private case key")
        if self.dimension_key not in W03_EVALUATION_ORDER:
            raise W03PrivateEvaluationError("private case dimension 非法")
        if (not isinstance(self.challenge_key, tuple) or not self.challenge_key
                or any(type(item) is not int or item < 0
                       for item in self.challenge_key)):
            raise W03PrivateEvaluationError("private challenge key 非法")


@dataclass(frozen=True)
class W03PrivateLabel:
    """独立 label owner 的 1/1、fail=0、NE=BLOCK 合同。"""

    label_key: str
    case_key: str
    dimension_key: str
    expected_status: str
    required: int
    fail_allowed: int
    ne_policy: str

    def __post_init__(self) -> None:
        _strict_sha256(self.label_key, label="private label key")
        _strict_sha256(self.case_key, label="private label case key")
        if (self.dimension_key not in W03_EVALUATION_ORDER
                or self.expected_status != "PASS"
                or self.required != 1
                or self.fail_allowed != 0
                or self.ne_policy != "BLOCK"):
            raise W03PrivateEvaluationError("private label 1/1 阈值漂移")


@dataclass(frozen=True)
class W03PrivateClusterBinding:
    """一个 case 的 source/template/content/schema 物理隔离承诺。"""

    case_key: str
    source_cluster: str
    template_cluster: str
    content_cluster: str
    schema_cluster: str

    def __post_init__(self) -> None:
        _strict_sha256(self.case_key, label="private cluster case key")
        for label, value in (
                ("source", self.source_cluster),
                ("template", self.template_cluster),
                ("content", self.content_cluster),
                ("schema", self.schema_cluster)):
            _strict_sha256(value, label=f"private {label} cluster")


@dataclass(frozen=True)
class W03PrivatePayload:
    """五份 owner 文档解码后的完整 private family 合同。"""

    source: W03PrivateSource
    cases: tuple[W03PrivateCase, ...]
    labels: tuple[W03PrivateLabel, ...]
    cluster_bindings: tuple[W03PrivateClusterBinding, ...]
    schema_key: str
    candidate_contract_sha256: str
    candidate_host_freeze_sha256: str
    formal_run_count: int


@dataclass(frozen=True)
class W03PrivateDimensionResult:
    """只含维度状态、计数和安全 Evidence commitment 的能力结果。"""

    dimension_key: str
    status: str
    passed: int
    required: int
    fail_count: int
    ne_count: int
    evidence_commitment: str

    def __post_init__(self) -> None:
        if self.dimension_key not in W03_EVALUATION_ORDER:
            raise W03PrivateEvaluationError("dimension result key 非法")
        if (self.status not in {"PASS", "FAIL", "NE"}
                or self.passed not in {0, 1}
                or self.required != 1
                or self.fail_count not in {0, 1}
                or self.ne_count not in {0, 1}):
            raise W03PrivateEvaluationError("dimension result 计数非法")
        expected = (
            (1, 0, 0) if self.status == "PASS"
            else (0, 1, 0) if self.status == "FAIL"
            else (0, 0, 1)
        )
        if (self.passed, self.fail_count, self.ne_count) != expected:
            raise W03PrivateEvaluationError("dimension result 状态/计数漂移")
        _strict_sha256(
            self.evidence_commitment, label="dimension evidence commitment")

    def to_safe_dict(self) -> dict[str, object]:
        """只导出可公开的维度、计数和 commitment。"""
        return {
            "dimension_key": self.dimension_key,
            "evidence_commitment": self.evidence_commitment,
            "fail_count": self.fail_count,
            "ne_count": self.ne_count,
            "passed": self.passed,
            "required": self.required,
            "status": self.status,
        }


def decode_w03_private_documents(
        source_bytes: bytes,
        schema_bytes: bytes,
        case_bytes: bytes,
        label_bytes: bytes,
        cluster_bytes: bytes,
        ) -> W03PrivatePayload:
    """原子解码五份 owner 文档并闭合顺序、引用、schema 和 run-count=0。"""
    source = _decode(source_bytes, label="private source")
    schema = _decode(schema_bytes, label="private schema")
    cases_doc = _decode(case_bytes, label="private case")
    labels_doc = _decode(label_bytes, label="private label")
    clusters_doc = _decode(cluster_bytes, label="private cluster")
    if set(source) != {
            "artifact_kind", "candidate_contract_sha256",
            "candidate_host_freeze_sha256", "family_key", "format_version",
            "license_id", "nonce_commitment", "owner_key", "source_key"}:
        raise W03PrivateEvaluationError("private source 字段集合漂移")
    if (source["artifact_kind"] != "PH2_W03_PRIVATE_SOURCE"
            or source["format_version"] != 1):
        raise W03PrivateEvaluationError("private source 版本漂移")
    source_value = W03PrivateSource(
        source["family_key"],
        _strict_key(source["source_key"], label="private source key"),
        source["owner_key"],
        source["license_id"],
        source["candidate_contract_sha256"],
        source["candidate_host_freeze_sha256"],
        source["nonce_commitment"],
    )
    if set(schema) != {
            "artifact_kind", "case_fields", "cluster_fields",
            "evaluation_order", "failure_phases", "fault_registry",
            "format_version", "label_fields", "schema_key"}:
        raise W03PrivateEvaluationError("private schema 字段集合漂移")
    expected_fields = {
        "case_fields": ["case_key", "challenge_key", "dimension_key"],
        "label_fields": [
            "case_key", "dimension_key", "expected_status", "fail_allowed",
            "label_key", "ne_policy", "required"],
        "cluster_fields": [
            "case_key", "content_cluster", "schema_cluster",
            "source_cluster", "template_cluster"],
    }
    if (schema.get("artifact_kind") != "PH2_W03_PRIVATE_SCHEMA"
            or schema.get("format_version") != 1
            or schema.get("evaluation_order") != list(W03_EVALUATION_ORDER)
            or schema.get("failure_phases") != list(W03_EVALUATOR_FAILURE_PHASES)
            or schema.get("fault_registry") != list(W03_EVALUATOR_PHASES)
            or any(schema.get(key) != value
                   for key, value in expected_fields.items())):
        raise W03PrivateEvaluationError("private schema/evaluation 顺序漂移")
    schema_key = _strict_sha256(schema["schema_key"], label="private schema key")

    if set(cases_doc) != {
            "artifact_kind", "cases", "formal_run_count", "format_version"}:
        raise W03PrivateEvaluationError("private case document 字段漂移")
    if (cases_doc["artifact_kind"] != "PH2_W03_PRIVATE_CASES"
            or cases_doc["format_version"] != 1
            or cases_doc["formal_run_count"] != 0
            or not isinstance(cases_doc["cases"], list)):
        raise W03PrivateEvaluationError("private case document 状态漂移")
    case_values = []
    for row in cases_doc["cases"]:
        if not isinstance(row, dict) or set(row) != set(expected_fields["case_fields"]):
            raise W03PrivateEvaluationError("private case 行字段漂移")
        case_values.append(W03PrivateCase(
            row["case_key"],
            row["dimension_key"],
            _strict_key(row["challenge_key"], label="private challenge key"),
        ))
    cases = tuple(case_values)
    if (tuple(item.dimension_key for item in cases) != W03_EVALUATION_ORDER
            or len({item.case_key for item in cases}) != len(cases)):
        raise W03PrivateEvaluationError("private case dimension 顺序或 identity 漂移")

    if set(labels_doc) != {"artifact_kind", "format_version", "labels"}:
        raise W03PrivateEvaluationError("private label document 字段漂移")
    if (labels_doc["artifact_kind"] != "PH2_W03_PRIVATE_LABELS"
            or labels_doc["format_version"] != 1
            or not isinstance(labels_doc["labels"], list)):
        raise W03PrivateEvaluationError("private label document 状态漂移")
    label_values = []
    for row in labels_doc["labels"]:
        if not isinstance(row, dict) or set(row) != set(expected_fields["label_fields"]):
            raise W03PrivateEvaluationError("private label 行字段漂移")
        label_values.append(W03PrivateLabel(
            row["label_key"], row["case_key"], row["dimension_key"],
            row["expected_status"], row["required"], row["fail_allowed"],
            row["ne_policy"],
        ))
    labels = tuple(label_values)
    if (tuple(item.dimension_key for item in labels) != W03_EVALUATION_ORDER
            or tuple(item.case_key for item in labels)
            != tuple(item.case_key for item in cases)
            or len({item.label_key for item in labels}) != len(labels)):
        raise W03PrivateEvaluationError("private label 顺序、case 引用或 identity 漂移")

    if set(clusters_doc) != {"artifact_kind", "clusters", "format_version"}:
        raise W03PrivateEvaluationError("private cluster document 字段漂移")
    if (clusters_doc["artifact_kind"] != "PH2_W03_PRIVATE_CLUSTERS"
            or clusters_doc["format_version"] != 1
            or not isinstance(clusters_doc["clusters"], list)):
        raise W03PrivateEvaluationError("private cluster document 状态漂移")
    cluster_values = []
    for row in clusters_doc["clusters"]:
        if not isinstance(row, dict) or set(row) != set(expected_fields["cluster_fields"]):
            raise W03PrivateEvaluationError("private cluster 行字段漂移")
        cluster_values.append(W03PrivateClusterBinding(
            row["case_key"], row["source_cluster"], row["template_cluster"],
            row["content_cluster"], row["schema_cluster"],
        ))
    clusters = tuple(cluster_values)
    if (tuple(item.case_key for item in clusters)
            != tuple(item.case_key for item in cases)
            or len({item.source_cluster for item in clusters}) != len(clusters)):
        raise W03PrivateEvaluationError("private cluster 引用或 source 隔离漂移")
    for attribute in (
            "template_cluster", "content_cluster", "schema_cluster"):
        if len({getattr(item, attribute) for item in clusters}) != len(clusters):
            raise W03PrivateEvaluationError("private cluster family 未隔离")
    return W03PrivatePayload(
        source_value,
        cases,
        labels,
        clusters,
        schema_key,
        source_value.candidate_contract_sha256,
        source_value.candidate_host_freeze_sha256,
        cases_doc["formal_run_count"],
    )


def public_safe_w03_aggregate(
        results: tuple[W03PrivateDimensionResult, ...],
        *,
        family_commitment: str,
        payload_commitment: str,
        case_commitment: str,
        label_commitment: str,
        cluster_commitment: str,
        failure_phase: str,
        formal_run_count: int,
        host_writes: int,
        label_writes: int,
        ) -> dict[str, object]:
    """只发布安全 commitment、枚举 phase 和五维计数，不投影私有内容。"""
    if failure_phase not in W03_EVALUATOR_FAILURE_PHASES:
        raise W03PrivateEvaluationError("failure phase 未注册")
    for label, value in (
            ("family", family_commitment), ("payload", payload_commitment),
            ("case", case_commitment), ("label", label_commitment),
            ("cluster", cluster_commitment)):
        _strict_sha256(value, label=f"{label} commitment")
    if (type(formal_run_count) is not int or formal_run_count != 1
            or type(host_writes) is not int or host_writes < 0
            or type(label_writes) is not int or label_writes < 0):
        raise W03PrivateEvaluationError("aggregate run/write count 非法")
    if failure_phase == "NONE":
        if (not isinstance(results, tuple)
                or tuple(item.dimension_key for item in results)
                != W03_EVALUATION_ORDER):
            raise W03PrivateEvaluationError("aggregate 五维结果顺序漂移")
    elif results:
        raise W03PrivateEvaluationError("基础设施失败不得伪造 capability result")
    pass_count = sum(item.passed for item in results)
    fail_count = sum(item.fail_count for item in results)
    ne_count = sum(item.ne_count for item in results)
    if failure_phase != "NONE":
        status = "NE"
        ne_count = 1
    elif (pass_count == len(W03_EVALUATION_ORDER)
            and fail_count == 0 and ne_count == 0
            and host_writes == 0 and label_writes == 0):
        status = "PASS"
    elif ne_count:
        status = "NE"
    else:
        status = "FAIL"
    return {
        "artifact_kind": "PH2_W03_PRIVATE_EVALUATION_AGGREGATE",
        "case_commitment": case_commitment,
        "cluster_commitment": cluster_commitment,
        "dimension_results": [item.to_safe_dict() for item in results],
        "fail_count": fail_count,
        "failure_phase": failure_phase,
        "family_commitment": family_commitment,
        "formal_run_count": formal_run_count,
        "format_version": 1,
        "host_writes": host_writes,
        "label_commitment": label_commitment,
        "label_writes": label_writes,
        "ne_count": ne_count,
        "pass_count": pass_count,
        "payload_commitment": payload_commitment,
        "status": status,
    }


def evidence_commitment(value: object) -> str:
    """把不含表层文本的结构证据摘要为公开安全 commitment。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "W03_EVALUATOR_FAILURE_PHASES",
    "W03_EVALUATOR_PHASES",
    "W03_PRIVATE_AGGREGATE_NAME",
    "W03_PRIVATE_FAMILY_FREEZE_NAME",
    "W03_PRIVATE_FIRST_RUN_GUARD_NAME",
    "W03_PRIVATE_OWNER_KEY",
    "W03_PRIVATE_RECOMMENDATION_NAME",
    "W03PrivateCase",
    "W03PrivateDimensionResult",
    "W03PrivateEvaluationError",
    "W03PrivatePayload",
    "decode_w03_private_documents",
    "evidence_commitment",
    "public_safe_w03_aggregate",
]
