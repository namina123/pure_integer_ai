"""PH2 W-06 独立 private evaluator 的文档、结果和安全 aggregate 合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w06_contract import (
    W06_EVALUATION_ORDER,
    W06_GENERATION_ABLATION_KEY,
    W06_PRIVATE_ABLATION_KEYS,
)


W06_PRIVATE_OWNER_KEY = "PH2_W06_PRIVATE_EVALUATOR_OWNER"
W06_PRIVATE_FAMILY_FREEZE_NAME = "private_family_freeze.json"
W06_PRIVATE_FIRST_RUN_GUARD_NAME = "formal_first_run_guard.json"
W06_PRIVATE_AGGREGATE_NAME = "private_evaluation_aggregate.json"
W06_PRIVATE_RECOMMENDATION_NAME = "runtime_receipt_recommendation.json"
W06_PRIVATE_SOURCE_NAME = "private_source.json"
W06_PRIVATE_SCHEMA_NAME = "private_schema.json"
W06_PRIVATE_CASE_NAME = "private_cases.json"
W06_PRIVATE_LABEL_NAME = "private_labels.json"
W06_PRIVATE_CLUSTER_NAME = "private_clusters.json"
W06_EVALUATOR_PHASES = (
    "PAYLOAD_DECODE",
    "CLONE_LOAD",
    "HOST_COPY",
    "CLONE_COMPARE",
    "BASELINE",
    "ABLATION_CAUSES",
    "ABLATION_MEREOLOGY",
    "ABLATION_PRECEDES",
    "ABLATION_PROPERTY",
    "ABLATION_PURE_ALIAS_REFERS",
    "ABLATION_SIMILAR_ANTONYM",
    "ABLATION_SUBSET_MEMBER",
    "ABLATION_GENERATION",
    "INTEGRITY",
    "REPORT_SAFETY",
)
W06_EVALUATOR_FAILURE_PHASES = ("NONE", *W06_EVALUATOR_PHASES)


class W06PrivateEvaluationError(RuntimeError):
    """W-06 private 文档、结果或安全报告违反冻结合同。"""


def _strict_sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W06PrivateEvaluationError(f"{label} 不是规范 SHA-256")
    return value


def _strict_key(value: object, *, label: str) -> tuple[int, ...]:
    if (not isinstance(value, list) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise W06PrivateEvaluationError(f"{label} 不是非空整数键")
    return tuple(value)


def _decode(payload: bytes, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not payload:
        raise W06PrivateEvaluationError(f"{label} payload 为空")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise W06PrivateEvaluationError(f"{label} payload 无法解析") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W06PrivateEvaluationError(f"{label} payload 非 canonical object")
    return value


@dataclass(frozen=True)
class W06PrivateSource:
    """private family 的 source 与 candidate 后置绑定。"""

    family_key: str
    source_key: tuple[int, ...]
    owner_key: str
    license_id: str
    candidate_contract_sha256: str
    candidate_host_freeze_sha256: str
    nonce_commitment: str

    def __post_init__(self) -> None:
        _strict_sha256(self.family_key, label="private family key")
        if self.owner_key != W06_PRIVATE_OWNER_KEY or self.license_id != "CC0-1.0":
            raise W06PrivateEvaluationError("private source owner/license 漂移")
        if not self.source_key:
            raise W06PrivateEvaluationError("private source key 为空")
        _strict_sha256(self.candidate_contract_sha256, label="candidate contract")
        _strict_sha256(self.candidate_host_freeze_sha256, label="candidate host")
        _strict_sha256(self.nonce_commitment, label="nonce commitment")


@dataclass(frozen=True)
class W06PrivateCase:
    """不携带 surface/expected 的 private challenge。"""

    case_key: str
    dimension_key: str
    challenge_key: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_sha256(self.case_key, label="private case key")
        if self.dimension_key not in W06_EVALUATION_ORDER:
            raise W06PrivateEvaluationError("private case dimension 非法")
        if not self.challenge_key:
            raise W06PrivateEvaluationError("private challenge key 为空")


@dataclass(frozen=True)
class W06PrivateLabel:
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
        if (self.dimension_key not in W06_EVALUATION_ORDER
                or self.expected_status != "PASS"
                or self.required != 1
                or self.fail_allowed != 0
                or self.ne_policy != "BLOCK"):
            raise W06PrivateEvaluationError("private label threshold 漂移")


@dataclass(frozen=True)
class W06PrivateClusterBinding:
    """一个 case 的 source/template/content/schema 隔离承诺。"""

    case_key: str
    source_cluster: str
    template_cluster: str
    content_cluster: str
    schema_cluster: str

    def __post_init__(self) -> None:
        _strict_sha256(self.case_key, label="cluster case key")
        for label, value in (
                ("source", self.source_cluster),
                ("template", self.template_cluster),
                ("content", self.content_cluster),
                ("schema", self.schema_cluster)):
            _strict_sha256(value, label=f"private {label} cluster")


@dataclass(frozen=True)
class W06PrivatePayload:
    """五份 private 文档解码后的合同。"""

    source: W06PrivateSource
    cases: tuple[W06PrivateCase, ...]
    labels: tuple[W06PrivateLabel, ...]
    cluster_bindings: tuple[W06PrivateClusterBinding, ...]
    schema_key: str
    candidate_contract_sha256: str
    candidate_host_freeze_sha256: str
    formal_run_count: int


@dataclass(frozen=True)
class W06PrivateDimensionResult:
    """只含维度状态、计数和 Evidence commitment 的结果。"""

    dimension_key: str
    status: str
    passed: int
    required: int
    fail_count: int
    ne_count: int
    evidence_commitment: str

    def __post_init__(self) -> None:
        if (self.dimension_key not in W06_EVALUATION_ORDER
                or self.status not in {"PASS", "FAIL", "NE"}
                or self.required != 1
                or self.passed not in {0, 1}
                or self.fail_count not in {0, 1}
                or self.ne_count not in {0, 1}):
            raise W06PrivateEvaluationError("dimension result 字段非法")
        expected = (
            (1, 0, 0) if self.status == "PASS"
            else (0, 1, 0) if self.status == "FAIL"
            else (0, 0, 1)
        )
        if (self.passed, self.fail_count, self.ne_count) != expected:
            raise W06PrivateEvaluationError("dimension result 状态/计数漂移")
        _strict_sha256(self.evidence_commitment, label="evidence commitment")

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "dimension_key": self.dimension_key,
            "evidence_commitment": self.evidence_commitment,
            "fail_count": self.fail_count,
            "ne_count": self.ne_count,
            "passed": self.passed,
            "required": self.required,
            "status": self.status,
        }


def decode_w06_private_documents(
        source_bytes: bytes,
        schema_bytes: bytes,
        case_bytes: bytes,
        label_bytes: bytes,
        cluster_bytes: bytes,
        ) -> W06PrivatePayload:
    """原子解码五份 private owner 文档并校验完整引用闭合。"""
    source = _decode(source_bytes, label="private source")
    schema = _decode(schema_bytes, label="private schema")
    cases_doc = _decode(case_bytes, label="private cases")
    labels_doc = _decode(label_bytes, label="private labels")
    clusters_doc = _decode(cluster_bytes, label="private clusters")
    if (set(source) != {
            "artifact_kind", "candidate_contract_sha256",
            "candidate_host_freeze_sha256", "family_key", "format_version",
            "license_id", "nonce_commitment", "owner_key", "source_key"}
            or source.get("artifact_kind") != "PH2_W06_PRIVATE_SOURCE"
            or source.get("format_version") != 1):
        raise W06PrivateEvaluationError("private source 版本漂移")
    source_value = W06PrivateSource(
        source["family_key"],
        _strict_key(source["source_key"], label="source key"),
        source["owner_key"],
        source["license_id"],
        source["candidate_contract_sha256"],
        source["candidate_host_freeze_sha256"],
        source["nonce_commitment"],
    )
    if (set(schema) != {
            "ablation_order", "artifact_kind", "case_fields",
            "cluster_fields", "evaluation_order", "failure_phases",
            "fault_registry", "format_version", "label_fields", "schema_key"}
            or schema.get("artifact_kind") != "PH2_W06_PRIVATE_SCHEMA"
            or schema.get("format_version") != 1
            or schema.get("ablation_order") != list(W06_PRIVATE_ABLATION_KEYS)
            or schema.get("evaluation_order") != list(W06_EVALUATION_ORDER)
            or schema.get("failure_phases") != list(W06_EVALUATOR_FAILURE_PHASES)
            or schema.get("fault_registry") != list(W06_EVALUATOR_PHASES)):
        raise W06PrivateEvaluationError("private schema 漂移")
    schema_key = _strict_sha256(schema["schema_key"], label="schema key")
    if (set(cases_doc) != {"artifact_kind", "case_count", "cases", "format_version"}
            or cases_doc.get("artifact_kind") != "PH2_W06_PRIVATE_CASES"
            or cases_doc.get("format_version") != 1
            or cases_doc.get("case_count") != len(W06_EVALUATION_ORDER)
            or set(labels_doc) != {"artifact_kind", "format_version", "labels"}
            or labels_doc.get("artifact_kind") != "PH2_W06_PRIVATE_LABELS"
            or labels_doc.get("format_version") != 1
            or set(clusters_doc) != {"artifact_kind", "clusters", "format_version"}
            or clusters_doc.get("artifact_kind") != "PH2_W06_PRIVATE_CLUSTERS"
            or clusters_doc.get("format_version") != 1):
        raise W06PrivateEvaluationError("private case/label/cluster 文档漂移")
    raw_cases = cases_doc.get("cases")
    raw_labels = labels_doc.get("labels")
    raw_clusters = clusters_doc.get("clusters")
    if not all(isinstance(item, list) for item in (
            raw_cases, raw_labels, raw_clusters)):
        raise W06PrivateEvaluationError("private list 文档非法")
    cases = tuple(W06PrivateCase(
        item["case_key"],
        item["dimension_key"],
        _strict_key(item["challenge_key"], label="challenge key"),
    ) for item in raw_cases)
    labels = tuple(W06PrivateLabel(
        item["label_key"],
        item["case_key"],
        item["dimension_key"],
        item["expected_status"],
        item["required"],
        item["fail_allowed"],
        item["ne_policy"],
    ) for item in raw_labels)
    clusters = tuple(W06PrivateClusterBinding(
        item["case_key"],
        item["source_cluster"],
        item["template_cluster"],
        item["content_cluster"],
        item["schema_cluster"],
    ) for item in raw_clusters)
    if (tuple(item.dimension_key for item in cases) != W06_EVALUATION_ORDER
            or tuple(item.case_key for item in labels)
            != tuple(item.case_key for item in cases)
            or tuple(item.case_key for item in clusters)
            != tuple(item.case_key for item in cases)
            or len(labels) != len(cases)
            or len(clusters) != len(cases)):
        raise W06PrivateEvaluationError("private case 顺序或引用漂移")
    return W06PrivatePayload(
        source_value,
        cases,
        labels,
        clusters,
        schema_key,
        source_value.candidate_contract_sha256,
        source_value.candidate_host_freeze_sha256,
        0,
    )


def evidence_commitment(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def public_safe_w06_aggregate(
        results: tuple[W06PrivateDimensionResult, ...],
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
    """只导出 commitment/count，不导出 private case、label 或 surface。"""
    for label, value in (
            ("family", family_commitment),
            ("payload", payload_commitment),
            ("case", case_commitment),
            ("label", label_commitment),
            ("cluster", cluster_commitment)):
        _strict_sha256(value, label=label)
    if failure_phase not in W06_EVALUATOR_FAILURE_PHASES:
        raise W06PrivateEvaluationError("failure phase 非法")
    if type(formal_run_count) is not int or formal_run_count != 1:
        raise W06PrivateEvaluationError("formal run count 必须为 1")
    if host_writes not in {0, 1} or label_writes not in {0, 1}:
        raise W06PrivateEvaluationError("host/label write count 非法")
    if failure_phase != "NONE":
        safe_results: tuple[W06PrivateDimensionResult, ...] = ()
        status = "NE"
        passed = fails = 0
        ne = 1
    else:
        if (not isinstance(results, tuple)
                or tuple(item.dimension_key for item in results)
                != W06_EVALUATION_ORDER):
            raise W06PrivateEvaluationError("W-06 八维结果顺序漂移")
        safe_results = results
        passed = sum(item.passed for item in results)
        fails = sum(item.fail_count for item in results)
        ne = sum(item.ne_count for item in results)
        status = (
            "PASS" if passed == len(W06_EVALUATION_ORDER) and not fails and not ne
            else "FAIL" if fails else "NE"
        )
    if host_writes or label_writes:
        status = "FAIL"
        fails = max(fails, 1)
    return {
        "ablation_results": [],
        "artifact_kind": "PH2_W06_PRIVATE_EVALUATION_AGGREGATE",
        "case_commitment": case_commitment,
        "cluster_commitment": cluster_commitment,
        "dimension_results": [item.to_safe_dict() for item in safe_results],
        "fail_count": fails,
        "failure_phase": failure_phase,
        "family_commitment": family_commitment,
        "formal_run_count": formal_run_count,
        "format_version": 1,
        "generation_ablation_statuses": [],
        "host_writes": host_writes,
        "infrastructure": {},
        "label_commitment": label_commitment,
        "label_writes": label_writes,
        "ne_count": ne,
        "pass_count": passed,
        "payload_commitment": payload_commitment,
        "status": status,
    }


__all__ = [
    "W06_EVALUATOR_FAILURE_PHASES",
    "W06_EVALUATOR_PHASES",
    "W06_GENERATION_ABLATION_KEY",
    "W06_PRIVATE_ABLATION_KEYS",
    "W06_PRIVATE_AGGREGATE_NAME",
    "W06_PRIVATE_CASE_NAME",
    "W06_PRIVATE_CLUSTER_NAME",
    "W06_PRIVATE_FAMILY_FREEZE_NAME",
    "W06_PRIVATE_FIRST_RUN_GUARD_NAME",
    "W06_PRIVATE_LABEL_NAME",
    "W06_PRIVATE_OWNER_KEY",
    "W06_PRIVATE_RECOMMENDATION_NAME",
    "W06_PRIVATE_SCHEMA_NAME",
    "W06_PRIVATE_SOURCE_NAME",
    "W06PrivateCase",
    "W06PrivateClusterBinding",
    "W06PrivateDimensionResult",
    "W06PrivateEvaluationError",
    "W06PrivateLabel",
    "W06PrivatePayload",
    "W06PrivateSource",
    "decode_w06_private_documents",
    "evidence_commitment",
    "public_safe_w06_aggregate",
]
