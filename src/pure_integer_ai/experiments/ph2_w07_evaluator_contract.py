"""W-07 独立 private evaluator 文档与 public-safe 结果合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_GENERATION_ABLATION_KEY,
    W07_PUBLIC_ABLATION_KEYS,
    W07_PUBLIC_DIMENSION_KEYS,
)


W07_PRIVATE_OWNER_KEY = "PH2_W07_PRIVATE_EVALUATOR_OWNER"
W07_PRIVATE_FAMILY_FREEZE_NAME = "private_family_freeze.json"
W07_PRIVATE_FIRST_RUN_GUARD_NAME = "formal_first_run_guard.json"
W07_PRIVATE_AGGREGATE_NAME = "private_evaluation_aggregate.json"
W07_PRIVATE_RECOMMENDATION_NAME = "runtime_receipt_recommendation.json"
W07_PRIVATE_SOURCE_NAME = "private_source.json"
W07_PRIVATE_SCHEMA_NAME = "private_schema.json"
W07_PRIVATE_CASE_NAME = "private_cases.json"
W07_PRIVATE_LABEL_NAME = "private_labels.json"
W07_PRIVATE_CLUSTER_NAME = "private_clusters.json"
W07_PRIVATE_HARD_REQUIREMENTS = (
    "SCOPE_FLIP",
    "QUANTIFIER_EXCHANGE",
    "UNKNOWN_CONFLICT_PRESERVATION",
    "OPEN_DOMAIN_FAIL_CLOSED",
    "MODAL_CERTIFICATE_REQUIRED",
    "CONDITION_CAUSAL_ISOLATION",
    "NESTED_LAYER_TRACE",
    "GENERATION_STRUCTURE_POSTCHECK",
)
W07_EVALUATOR_PHASES = (
    "PAYLOAD_DECODE",
    "CLONE_LOAD",
    "HOST_COPY",
    "CLONE_COMPARE",
    "BASELINE",
    "ABLATION_AND_OR",
    "ABLATION_CONDITION",
    "ABLATION_EXISTS",
    "ABLATION_FORALL",
    "ABLATION_MODAL",
    "ABLATION_NESTED_SCOPE",
    "ABLATION_NOT",
    "ABLATION_GENERATION",
    "INTEGRITY",
    "REPORT_SAFETY",
)
W07_EVALUATOR_FAILURE_PHASES = ("NONE", *W07_EVALUATOR_PHASES)


class W07PrivateEvaluationError(RuntimeError):
    """private 文档或安全结果违反冻结的 W-07 合同。"""


def _strict_sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W07PrivateEvaluationError(f"{label} is not canonical SHA-256")
    return value


def _strict_sha1(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 40
            or any(char not in "0123456789abcdef" for char in value)):
        raise W07PrivateEvaluationError(f"{label} is not canonical SHA-1")
    return value


def _strict_key(value: object, *, label: str) -> tuple[int, ...]:
    if (not isinstance(value, list) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise W07PrivateEvaluationError(f"{label} is not an integer key")
    return tuple(value)


def _decode(payload: bytes, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not payload:
        raise W07PrivateEvaluationError(f"{label} payload is empty")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise W07PrivateEvaluationError(f"{label} payload is invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W07PrivateEvaluationError(f"{label} is not canonical JSON")
    return value


@dataclass(frozen=True)
class W07PrivateSource:
    family_key: str
    source_key: tuple[int, ...]
    owner_key: str
    license_id: str
    candidate_contract_sha256: str
    candidate_host_freeze_sha256: str
    evaluator_public_head_commit_sha1: str
    nonce_commitment: str

    def __post_init__(self) -> None:
        _strict_sha256(self.family_key, label="private family key")
        if self.owner_key != W07_PRIVATE_OWNER_KEY or self.license_id != "CC0-1.0":
            raise W07PrivateEvaluationError("private source owner/license drift")
        if not self.source_key:
            raise W07PrivateEvaluationError("private source key is empty")
        _strict_sha256(self.candidate_contract_sha256, label="candidate contract")
        _strict_sha256(self.candidate_host_freeze_sha256, label="candidate host")
        _strict_sha1(self.evaluator_public_head_commit_sha1, label="evaluator HEAD")
        _strict_sha256(self.nonce_commitment, label="nonce commitment")


@dataclass(frozen=True)
class W07PrivateCase:
    case_key: str
    dimension_key: str
    challenge_key: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_sha256(self.case_key, label="private case key")
        if self.dimension_key not in W07_PUBLIC_DIMENSION_KEYS:
            raise W07PrivateEvaluationError("private case dimension drift")
        if not self.challenge_key:
            raise W07PrivateEvaluationError("private challenge key is empty")


@dataclass(frozen=True)
class W07PrivateLabel:
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
        if (self.dimension_key not in W07_PUBLIC_DIMENSION_KEYS
                or self.expected_status != "PASS"
                or self.required != 1
                or self.fail_allowed != 0
                or self.ne_policy != "BLOCK"):
            raise W07PrivateEvaluationError("private label threshold drift")


@dataclass(frozen=True)
class W07PrivateClusterBinding:
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
class W07PrivatePayload:
    source: W07PrivateSource
    cases: tuple[W07PrivateCase, ...]
    labels: tuple[W07PrivateLabel, ...]
    cluster_bindings: tuple[W07PrivateClusterBinding, ...]
    schema_key: str
    candidate_contract_sha256: str
    candidate_host_freeze_sha256: str
    evaluator_public_head_commit_sha1: str
    formal_run_count: int


@dataclass(frozen=True)
class W07PrivateDimensionResult:
    dimension_key: str
    status: str
    passed: int
    required: int
    fail_count: int
    ne_count: int
    evidence_commitment: str

    def __post_init__(self) -> None:
        if (self.dimension_key not in W07_PUBLIC_DIMENSION_KEYS
                or self.status not in {"PASS", "FAIL", "NE"}
                or self.required != 1
                or self.passed not in {0, 1}
                or self.fail_count not in {0, 1}
                or self.ne_count not in {0, 1}):
            raise W07PrivateEvaluationError("dimension result fields are invalid")
        expected = (
            (1, 0, 0) if self.status == "PASS"
            else (0, 1, 0) if self.status == "FAIL"
            else (0, 0, 1)
        )
        if (self.passed, self.fail_count, self.ne_count) != expected:
            raise W07PrivateEvaluationError("dimension status/count drift")
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


def decode_w07_private_documents(
        source_bytes: bytes,
        schema_bytes: bytes,
        case_bytes: bytes,
        label_bytes: bytes,
        cluster_bytes: bytes,
        ) -> W07PrivatePayload:
    """原子解码五份 owner 隔离的 private 文档。"""
    source = _decode(source_bytes, label="private source")
    schema = _decode(schema_bytes, label="private schema")
    cases_doc = _decode(case_bytes, label="private cases")
    labels_doc = _decode(label_bytes, label="private labels")
    clusters_doc = _decode(cluster_bytes, label="private clusters")
    if (set(source) != {
            "artifact_kind", "candidate_contract_sha256",
            "candidate_host_freeze_sha256",
            "evaluator_public_head_commit_sha1", "family_key",
            "format_version", "license_id", "nonce_commitment",
            "owner_key", "source_key"}
            or source.get("artifact_kind") != "PH2_W07_PRIVATE_SOURCE"
            or source.get("format_version") != 1):
        raise W07PrivateEvaluationError("private source version drift")
    source_value = W07PrivateSource(
        source["family_key"],
        _strict_key(source["source_key"], label="source key"),
        source["owner_key"],
        source["license_id"],
        source["candidate_contract_sha256"],
        source["candidate_host_freeze_sha256"],
        source["evaluator_public_head_commit_sha1"],
        source["nonce_commitment"],
    )
    if (set(schema) != {
            "ablation_order", "artifact_kind", "case_fields",
            "cluster_fields", "evaluation_order", "failure_phases",
            "fault_registry", "format_version", "generation_contract",
            "hard_requirements", "label_fields", "schema_key"}
            or schema.get("artifact_kind") != "PH2_W07_PRIVATE_SCHEMA"
            or schema.get("format_version") != 1
            or schema.get("ablation_order") != list(W07_PUBLIC_ABLATION_KEYS)
            or schema.get("evaluation_order") != list(W07_PUBLIC_DIMENSION_KEYS)
            or schema.get("failure_phases") != list(W07_EVALUATOR_FAILURE_PHASES)
            or schema.get("fault_registry") != list(W07_EVALUATOR_PHASES)
            or schema.get("hard_requirements")
            != list(W07_PRIVATE_HARD_REQUIREMENTS)
            or schema.get("case_fields")
            != ["case_key", "challenge_key", "dimension_key"]
            or schema.get("cluster_fields") != [
                "case_key", "content_cluster", "schema_cluster",
                "source_cluster", "template_cluster",
            ]
            or schema.get("label_fields") != [
                "case_key", "dimension_key", "expected_status",
                "fail_allowed", "label_key", "ne_policy", "required",
            ]
            or schema.get("generation_contract") != {
                "ablation_key": W07_GENERATION_ABLATION_KEY,
                "choice_use_postcheck_required": 1,
                "dimension_key": W07_PUBLIC_DIMENSION_KEYS[-1],
                "substage_count": 7,
            }):
        raise W07PrivateEvaluationError("private schema drift")
    schema_key = _strict_sha256(schema["schema_key"], label="schema key")
    if (set(cases_doc) != {"artifact_kind", "case_count", "cases", "format_version"}
            or cases_doc.get("artifact_kind") != "PH2_W07_PRIVATE_CASES"
            or cases_doc.get("format_version") != 1
            or cases_doc.get("case_count") != len(W07_PUBLIC_DIMENSION_KEYS)
            or set(labels_doc) != {"artifact_kind", "format_version", "labels"}
            or labels_doc.get("artifact_kind") != "PH2_W07_PRIVATE_LABELS"
            or labels_doc.get("format_version") != 1
            or set(clusters_doc) != {"artifact_kind", "clusters", "format_version"}
            or clusters_doc.get("artifact_kind") != "PH2_W07_PRIVATE_CLUSTERS"
            or clusters_doc.get("format_version") != 1):
        raise W07PrivateEvaluationError("private case/label/cluster drift")
    raw_cases = cases_doc.get("cases")
    raw_labels = labels_doc.get("labels")
    raw_clusters = clusters_doc.get("clusters")
    if not all(isinstance(item, list) for item in (
            raw_cases, raw_labels, raw_clusters)):
        raise W07PrivateEvaluationError("private list document is invalid")
    if (any(not isinstance(item, dict) or set(item) != {
                "case_key", "challenge_key", "dimension_key"}
            for item in raw_cases)
            or any(not isinstance(item, dict) or set(item) != {
                "case_key", "dimension_key", "expected_status",
                "fail_allowed", "label_key", "ne_policy", "required"}
                for item in raw_labels)
            or any(not isinstance(item, dict) or set(item) != {
                "case_key", "content_cluster", "schema_cluster",
                "source_cluster", "template_cluster"}
                for item in raw_clusters)):
        raise W07PrivateEvaluationError("private list item schema drift")
    try:
        cases = tuple(W07PrivateCase(
            item["case_key"], item["dimension_key"],
            _strict_key(item["challenge_key"], label="challenge key"),
        ) for item in raw_cases)
        labels = tuple(W07PrivateLabel(
            item["label_key"], item["case_key"], item["dimension_key"],
            item["expected_status"], item["required"],
            item["fail_allowed"], item["ne_policy"],
        ) for item in raw_labels)
        clusters = tuple(W07PrivateClusterBinding(
            item["case_key"], item["source_cluster"],
            item["template_cluster"], item["content_cluster"],
            item["schema_cluster"],
        ) for item in raw_clusters)
    except (KeyError, TypeError) as error:
        raise W07PrivateEvaluationError("private list item fields drift") from error
    if (tuple(item.dimension_key for item in cases)
            != W07_PUBLIC_DIMENSION_KEYS
            or tuple(item.case_key for item in labels)
            != tuple(item.case_key for item in cases)
            or tuple(item.case_key for item in clusters)
            != tuple(item.case_key for item in cases)
            or len(labels) != len(cases) or len(clusters) != len(cases)):
        raise W07PrivateEvaluationError("private case order/reference drift")
    return W07PrivatePayload(
        source_value,
        cases,
        labels,
        clusters,
        schema_key,
        source_value.candidate_contract_sha256,
        source_value.candidate_host_freeze_sha256,
        source_value.evaluator_public_head_commit_sha1,
        0,
    )


def evidence_commitment(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def public_safe_w07_aggregate(
        results: tuple[W07PrivateDimensionResult, ...],
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
    """只导出 commitment 与计数，不导出 case、label 或 surface。"""
    for label, value in (
            ("family", family_commitment), ("payload", payload_commitment),
            ("case", case_commitment), ("label", label_commitment),
            ("cluster", cluster_commitment)):
        _strict_sha256(value, label=label)
    if failure_phase not in W07_EVALUATOR_FAILURE_PHASES:
        raise W07PrivateEvaluationError("failure phase drift")
    if type(formal_run_count) is not int or formal_run_count != 1:
        raise W07PrivateEvaluationError("formal run count must be one")
    if host_writes not in {0, 1} or label_writes not in {0, 1}:
        raise W07PrivateEvaluationError("host/label write count is invalid")
    if failure_phase != "NONE":
        safe_results: tuple[W07PrivateDimensionResult, ...] = ()
        status, passed, fails, ne = "NE", 0, 0, 1
    else:
        if (not isinstance(results, tuple)
                or tuple(item.dimension_key for item in results)
                != W07_PUBLIC_DIMENSION_KEYS):
            raise W07PrivateEvaluationError("W-07 dimension order drift")
        safe_results = results
        passed = sum(item.passed for item in results)
        fails = sum(item.fail_count for item in results)
        ne = sum(item.ne_count for item in results)
        status = (
            "PASS" if passed == len(W07_PUBLIC_DIMENSION_KEYS)
            and not fails and not ne else "FAIL" if fails else "NE")
    if host_writes or label_writes:
        status, fails = "FAIL", max(fails, 1)
    return {
        "ablation_results": [],
        "artifact_kind": "PH2_W07_PRIVATE_EVALUATION_AGGREGATE",
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


__all__ = [name for name in globals() if name.startswith("W07_")] + [
    "W07PrivateCase",
    "W07PrivateClusterBinding",
    "W07PrivateDimensionResult",
    "W07PrivateEvaluationError",
    "W07PrivateLabel",
    "W07PrivatePayload",
    "W07PrivateSource",
    "decode_w07_private_documents",
    "evidence_commitment",
    "public_safe_w07_aggregate",
]
