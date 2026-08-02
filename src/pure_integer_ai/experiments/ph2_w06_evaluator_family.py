"""W-06 private evaluator family 的独立生成、冻结与唯一 guard。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w06_contract import (
    W06_EVALUATION_ORDER,
    W06_PRIVATE_ABLATION_KEYS,
)
from pure_integer_ai.experiments.ph2_w06_evaluator_contract import (
    W06_EVALUATOR_FAILURE_PHASES,
    W06_EVALUATOR_PHASES,
    W06_PRIVATE_CASE_NAME,
    W06_PRIVATE_CLUSTER_NAME,
    W06_PRIVATE_FAMILY_FREEZE_NAME,
    W06_PRIVATE_FIRST_RUN_GUARD_NAME,
    W06_PRIVATE_LABEL_NAME,
    W06_PRIVATE_OWNER_KEY,
    W06_PRIVATE_SCHEMA_NAME,
    W06_PRIVATE_SOURCE_NAME,
    W06PrivateEvaluationError,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W06PrivateEvaluationError(f"{label} 不是规范 SHA-256")
    return value


def _digest(value: object) -> str:
    return _sha256(canonical_json_bytes(value))


def _nonce_bytes(nonce: tuple[int, ...]) -> bytes:
    if (not isinstance(nonce, tuple) or not nonce
            or any(type(item) is not int or not 0 <= item <= 255
                   for item in nonce)):
        raise W06PrivateEvaluationError("private nonce 必须是 byte tuple")
    return bytes(nonce)


@dataclass(frozen=True)
class W06PrivateFamilyDocuments:
    """五份 private 文档与其 commitment。"""

    source_bytes: bytes
    schema_bytes: bytes
    case_bytes: bytes
    label_bytes: bytes
    cluster_bytes: bytes
    family_key: str
    payload_commitment: str
    case_commitment: str
    label_commitment: str
    cluster_commitment: str
    candidate_contract_sha256: str
    candidate_host_freeze_sha256: str

    def files(self) -> tuple[tuple[str, bytes], ...]:
        return (
            (W06_PRIVATE_SOURCE_NAME, self.source_bytes),
            (W06_PRIVATE_SCHEMA_NAME, self.schema_bytes),
            (W06_PRIVATE_CASE_NAME, self.case_bytes),
            (W06_PRIVATE_LABEL_NAME, self.label_bytes),
            (W06_PRIVATE_CLUSTER_NAME, self.cluster_bytes),
        )


def build_w06_private_family_documents(
        *,
        candidate_contract_sha256: str,
        candidate_host_freeze_sha256: str,
        nonce: tuple[int, ...] = (6, 17, 31, 47),
        ) -> W06PrivateFamilyDocuments:
    """生成不含 expected surface/raw Observation 的新 private family。"""
    candidate_contract = _strict_sha256(
        candidate_contract_sha256, label="candidate contract")
    candidate_host = _strict_sha256(
        candidate_host_freeze_sha256, label="candidate host")
    nonce_commitment = _digest({"nonce": list(_nonce_bytes(nonce))})
    family_key = _digest({
        "candidate_contract_sha256": candidate_contract,
        "candidate_host_freeze_sha256": candidate_host,
        "nonce_commitment": nonce_commitment,
        "owner_key": W06_PRIVATE_OWNER_KEY,
    })
    source = {
        "artifact_kind": "PH2_W06_PRIVATE_SOURCE",
        "candidate_contract_sha256": candidate_contract,
        "candidate_host_freeze_sha256": candidate_host,
        "family_key": family_key,
        "format_version": 1,
        "license_id": "CC0-1.0",
        "nonce_commitment": nonce_commitment,
        "owner_key": W06_PRIVATE_OWNER_KEY,
        "source_key": [2026, 806, 1],
    }
    schema_key = _digest({
        "ablation_order": list(W06_PRIVATE_ABLATION_KEYS),
        "evaluation_order": list(W06_EVALUATION_ORDER),
        "failure_phases": list(W06_EVALUATOR_FAILURE_PHASES),
        "family_key": family_key,
    })
    schema = {
        "ablation_order": list(W06_PRIVATE_ABLATION_KEYS),
        "artifact_kind": "PH2_W06_PRIVATE_SCHEMA",
        "case_fields": ["case_key", "challenge_key", "dimension_key"],
        "cluster_fields": [
            "case_key", "source_cluster", "template_cluster",
            "content_cluster", "schema_cluster",
        ],
        "evaluation_order": list(W06_EVALUATION_ORDER),
        "failure_phases": list(W06_EVALUATOR_FAILURE_PHASES),
        "fault_registry": list(W06_EVALUATOR_PHASES),
        "format_version": 1,
        "label_fields": [
            "label_key", "case_key", "dimension_key", "expected_status",
            "required", "fail_allowed", "ne_policy",
        ],
        "schema_key": schema_key,
    }
    cases = []
    labels = []
    clusters = []
    for ordinal, dimension in enumerate(W06_EVALUATION_ORDER, start=1):
        case_key = _digest({
            "dimension": dimension,
            "family_key": family_key,
            "ordinal": ordinal,
        })
        label_key = _digest({"case_key": case_key, "kind": "label"})
        challenge = [2026, 806, ordinal, *list(_nonce_bytes(nonce))]
        cases.append({
            "case_key": case_key,
            "challenge_key": challenge,
            "dimension_key": dimension,
        })
        labels.append({
            "case_key": case_key,
            "dimension_key": dimension,
            "expected_status": "PASS",
            "fail_allowed": 0,
            "label_key": label_key,
            "ne_policy": "BLOCK",
            "required": 1,
        })
        clusters.append({
            "case_key": case_key,
            "content_cluster": _digest({"case": case_key, "axis": "content"}),
            "schema_cluster": _digest({"case": case_key, "axis": "schema"}),
            "source_cluster": _digest({"case": case_key, "axis": "source"}),
            "template_cluster": _digest({"case": case_key, "axis": "template"}),
        })
    source_bytes = canonical_json_bytes(source)
    schema_bytes = canonical_json_bytes(schema)
    case_bytes = canonical_json_bytes({
        "artifact_kind": "PH2_W06_PRIVATE_CASES",
        "case_count": len(cases),
        "cases": cases,
        "format_version": 1,
    })
    label_bytes = canonical_json_bytes({
        "artifact_kind": "PH2_W06_PRIVATE_LABELS",
        "format_version": 1,
        "labels": labels,
    })
    cluster_bytes = canonical_json_bytes({
        "artifact_kind": "PH2_W06_PRIVATE_CLUSTERS",
        "clusters": clusters,
        "format_version": 1,
    })
    return W06PrivateFamilyDocuments(
        source_bytes,
        schema_bytes,
        case_bytes,
        label_bytes,
        cluster_bytes,
        family_key,
        _digest({
            "source": _sha256(source_bytes),
            "schema": _sha256(schema_bytes),
        }),
        _sha256(case_bytes),
        _sha256(label_bytes),
        _sha256(cluster_bytes),
        candidate_contract,
        candidate_host,
    )


def _validate_root(root: Path, forbidden_roots: tuple[str | Path, ...]) -> None:
    for forbidden in forbidden_roots:
        path = Path(forbidden).resolve()
        if (root == path or root.is_relative_to(path)
                or path.is_relative_to(root)):
            raise W06PrivateEvaluationError("private family root 未隔离")


def publish_w06_private_family(
        artifact_root: str | Path,
        documents: W06PrivateFamilyDocuments,
        *,
        forbidden_roots: tuple[str | Path, ...] = (),
        ) -> tuple[Path, str]:
    """以排他写发布新 private family，不覆盖既有 family。"""
    if not isinstance(documents, W06PrivateFamilyDocuments):
        raise TypeError("private documents 类型非法")
    root = Path(artifact_root).resolve()
    _validate_root(root, forbidden_roots)
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in documents.files():
        target = root / name
        try:
            with target.open("xb") as handle:
                handle.write(payload)
        except FileExistsError as error:
            raise W06PrivateEvaluationError("private family 文件不可覆盖") from error
    inventory = [
        {"path": name, "sha256": _sha256(payload), "size_bytes": len(payload)}
        for name, payload in documents.files()
    ]
    freeze = {
        "ablation_order": list(W06_PRIVATE_ABLATION_KEYS),
        "artifact_kind": "PH2_W06_PRIVATE_FAMILY_FREEZE",
        "candidate_contract_sha256": documents.candidate_contract_sha256,
        "candidate_host_freeze_sha256": documents.candidate_host_freeze_sha256,
        "case_commitment": documents.case_commitment,
        "cluster_commitment": documents.cluster_commitment,
        "family_key": documents.family_key,
        "file_inventory": inventory,
        "formal_run_count": 0,
        "format_version": 1,
        "label_commitment": documents.label_commitment,
        "owner_key": W06_PRIVATE_OWNER_KEY,
        "payload_commitment": documents.payload_commitment,
        "self_excluded": 1,
    }
    target = root / W06_PRIVATE_FAMILY_FREEZE_NAME
    encoded = canonical_json_bytes(freeze)
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        raise W06PrivateEvaluationError("private family freeze 不可覆盖") from error
    return target, _sha256(encoded)


def consume_w06_private_first_run_guard(
        artifact_root: str | Path,
        *,
        family_freeze_sha256: str,
        ) -> tuple[Path, str]:
    """排他消费 private family 的唯一正式运行 guard。"""
    root = Path(artifact_root).resolve()
    expected = _strict_sha256(family_freeze_sha256, label="family freeze")
    freeze = root / W06_PRIVATE_FAMILY_FREEZE_NAME
    if (not freeze.is_file() or freeze.is_symlink()
            or _sha256(freeze.read_bytes()) != expected):
        raise W06PrivateEvaluationError("private family freeze SHA 漂移")
    try:
        value = json.loads(freeze.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise W06PrivateEvaluationError("private family freeze 无法读取") from error
    if value.get("formal_run_count") != 0:
        raise W06PrivateEvaluationError("private family 已运行")
    payload = canonical_json_bytes({
        "artifact_kind": "PH2_W06_PRIVATE_FIRST_RUN_GUARD",
        "family_freeze_sha256": expected,
        "formal_run_count_after": 1,
        "formal_run_count_before": 0,
        "format_version": 1,
    })
    target = root / W06_PRIVATE_FIRST_RUN_GUARD_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise W06PrivateEvaluationError(
            "private family guard 已消费，不可重跑") from error
    return target, _sha256(payload)


__all__ = [
    "W06PrivateFamilyDocuments",
    "build_w06_private_family_documents",
    "consume_w06_private_first_run_guard",
    "publish_w06_private_family",
]
