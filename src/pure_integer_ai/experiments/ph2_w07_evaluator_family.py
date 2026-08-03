"""W-07 private evaluator family 的生成、冻结与唯一 guard。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_GENERATION_ABLATION_KEY,
    W07_PUBLIC_ABLATION_KEYS,
    W07_PUBLIC_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_contract import (
    W07_EVALUATOR_FAILURE_PHASES,
    W07_EVALUATOR_PHASES,
    W07_PRIVATE_CASE_NAME,
    W07_PRIVATE_CLUSTER_NAME,
    W07_PRIVATE_FAMILY_FREEZE_NAME,
    W07_PRIVATE_FIRST_RUN_GUARD_NAME,
    W07_PRIVATE_HARD_REQUIREMENTS,
    W07_PRIVATE_LABEL_NAME,
    W07_PRIVATE_OWNER_KEY,
    W07_PRIVATE_SCHEMA_NAME,
    W07_PRIVATE_SOURCE_NAME,
    W07PrivateEvaluationError,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _digest(value: object) -> str:
    return _sha256(canonical_json_bytes(value))


def _nonce_bytes(nonce: tuple[int, ...]) -> bytes:
    if (not isinstance(nonce, tuple) or not nonce
            or any(type(item) is not int or not 0 <= item <= 255
                   for item in nonce)):
        raise W07PrivateEvaluationError("private nonce must be a byte tuple")
    return bytes(nonce)


@dataclass(frozen=True)
class W07PrivateFamilyDocuments:
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
    evaluator_public_head_commit_sha1: str

    def files(self) -> tuple[tuple[str, bytes], ...]:
        return (
            (W07_PRIVATE_SOURCE_NAME, self.source_bytes),
            (W07_PRIVATE_SCHEMA_NAME, self.schema_bytes),
            (W07_PRIVATE_CASE_NAME, self.case_bytes),
            (W07_PRIVATE_LABEL_NAME, self.label_bytes),
            (W07_PRIVATE_CLUSTER_NAME, self.cluster_bytes),
        )


def build_w07_private_family_documents(
        *,
        candidate_contract_sha256: str,
        candidate_host_freeze_sha256: str,
        evaluator_public_head_commit_sha1: str,
        nonce: tuple[int, ...] = (7, 19, 37, 73),
        ) -> W07PrivateFamilyDocuments:
    """建立不含 raw expected surface 的全新八维 family。"""
    candidate_contract = _strict_sha256(
        candidate_contract_sha256, label="candidate contract")
    candidate_host = _strict_sha256(
        candidate_host_freeze_sha256, label="candidate host")
    evaluator_head = _strict_sha1(
        evaluator_public_head_commit_sha1, label="evaluator HEAD")
    nonce_commitment = _digest({"nonce": list(_nonce_bytes(nonce))})
    family_key = _digest({
        "candidate_contract_sha256": candidate_contract,
        "candidate_host_freeze_sha256": candidate_host,
        "evaluator_public_head_commit_sha1": evaluator_head,
        "nonce_commitment": nonce_commitment,
        "owner_key": W07_PRIVATE_OWNER_KEY,
    })
    source = {
        "artifact_kind": "PH2_W07_PRIVATE_SOURCE",
        "candidate_contract_sha256": candidate_contract,
        "candidate_host_freeze_sha256": candidate_host,
        "evaluator_public_head_commit_sha1": evaluator_head,
        "family_key": family_key,
        "format_version": 1,
        "license_id": "CC0-1.0",
        "nonce_commitment": nonce_commitment,
        "owner_key": W07_PRIVATE_OWNER_KEY,
        "source_key": [2026, 807, 1],
    }
    schema_key = _digest({
        "ablation_order": list(W07_PUBLIC_ABLATION_KEYS),
        "evaluation_order": list(W07_PUBLIC_DIMENSION_KEYS),
        "failure_phases": list(W07_EVALUATOR_FAILURE_PHASES),
        "family_key": family_key,
        "hard_requirements": list(W07_PRIVATE_HARD_REQUIREMENTS),
    })
    schema = {
        "ablation_order": list(W07_PUBLIC_ABLATION_KEYS),
        "artifact_kind": "PH2_W07_PRIVATE_SCHEMA",
        "case_fields": ["case_key", "challenge_key", "dimension_key"],
        "cluster_fields": [
            "case_key", "content_cluster", "schema_cluster",
            "source_cluster", "template_cluster",
        ],
        "evaluation_order": list(W07_PUBLIC_DIMENSION_KEYS),
        "failure_phases": list(W07_EVALUATOR_FAILURE_PHASES),
        "fault_registry": list(W07_EVALUATOR_PHASES),
        "format_version": 1,
        "generation_contract": {
            "ablation_key": W07_GENERATION_ABLATION_KEY,
            "choice_use_postcheck_required": 1,
            "dimension_key": W07_PUBLIC_DIMENSION_KEYS[-1],
            "substage_count": 7,
        },
        "hard_requirements": list(W07_PRIVATE_HARD_REQUIREMENTS),
        "label_fields": [
            "case_key", "dimension_key", "expected_status",
            "fail_allowed", "label_key", "ne_policy", "required",
        ],
        "schema_key": schema_key,
    }
    cases = []
    labels = []
    clusters = []
    for ordinal, dimension in enumerate(W07_PUBLIC_DIMENSION_KEYS, start=1):
        challenge_digest = hashlib.sha256(canonical_json_bytes({
            "dimension": dimension,
            "family_key": family_key,
            "ordinal": ordinal,
            "schema_key": schema_key,
        })).digest()
        challenge_key = [ordinal, *challenge_digest[:16]]
        case_key = _digest({
            "challenge_key": challenge_key,
            "dimension_key": dimension,
            "family_key": family_key,
        })
        label_key = _digest({
            "case_key": case_key,
            "dimension_key": dimension,
            "expected_status": "PASS",
            "threshold": [1, 0, "BLOCK"],
        })
        cases.append({
            "case_key": case_key,
            "challenge_key": challenge_key,
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
            "content_cluster": _digest([family_key, dimension, "content"]),
            "schema_cluster": _digest([family_key, dimension, "schema"]),
            "source_cluster": _digest([family_key, dimension, "source"]),
            "template_cluster": _digest([family_key, dimension, "template"]),
        })
    source_bytes = canonical_json_bytes(source)
    schema_bytes = canonical_json_bytes(schema)
    case_bytes = canonical_json_bytes({
        "artifact_kind": "PH2_W07_PRIVATE_CASES",
        "case_count": len(cases),
        "cases": cases,
        "format_version": 1,
    })
    label_bytes = canonical_json_bytes({
        "artifact_kind": "PH2_W07_PRIVATE_LABELS",
        "format_version": 1,
        "labels": labels,
    })
    cluster_bytes = canonical_json_bytes({
        "artifact_kind": "PH2_W07_PRIVATE_CLUSTERS",
        "clusters": clusters,
        "format_version": 1,
    })
    return W07PrivateFamilyDocuments(
        source_bytes,
        schema_bytes,
        case_bytes,
        label_bytes,
        cluster_bytes,
        family_key,
        _digest({
            "source": _sha256(source_bytes),
            "schema": _sha256(schema_bytes),
            "cases": _sha256(case_bytes),
            "labels": _sha256(label_bytes),
            "clusters": _sha256(cluster_bytes),
        }),
        _sha256(case_bytes),
        _sha256(label_bytes),
        _sha256(cluster_bytes),
        candidate_contract,
        candidate_host,
        evaluator_head,
    )


def _validate_root(root: Path, forbidden_roots: tuple[str | Path, ...]) -> None:
    for value in forbidden_roots:
        forbidden = Path(value).resolve()
        if (root == forbidden or root.is_relative_to(forbidden)
                or forbidden.is_relative_to(root)):
            raise W07PrivateEvaluationError("private family root is not isolated")


def publish_w07_private_family(
        artifact_root: str | Path,
        documents: W07PrivateFamilyDocuments,
        *,
        forbidden_roots: tuple[str | Path, ...] = (),
        ) -> tuple[Path, str]:
    """在首次 private payload 读取前持久化全部五份 commitment。"""
    if not isinstance(documents, W07PrivateFamilyDocuments):
        raise TypeError("W-07 private family documents type drift")
    root = Path(artifact_root).resolve()
    _validate_root(root, forbidden_roots)
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in documents.files():
        try:
            with (root / name).open("xb") as handle:
                handle.write(payload)
        except FileExistsError as error:
            raise W07PrivateEvaluationError("private family file is immutable") from error
    inventory = [
        {"path": name, "sha256": _sha256(payload), "size_bytes": len(payload)}
        for name, payload in documents.files()
    ]
    freeze = {
        "ablation_order": list(W07_PUBLIC_ABLATION_KEYS),
        "artifact_kind": "PH2_W07_PRIVATE_FAMILY_FREEZE",
        "candidate_contract_sha256": documents.candidate_contract_sha256,
        "candidate_host_freeze_sha256": documents.candidate_host_freeze_sha256,
        "case_commitment": documents.case_commitment,
        "cluster_commitment": documents.cluster_commitment,
        "evaluator_public_head_commit_sha1": (
            documents.evaluator_public_head_commit_sha1),
        "family_key": documents.family_key,
        "file_inventory": inventory,
        "formal_run_count": 0,
        "format_version": 1,
        "hard_requirements": list(W07_PRIVATE_HARD_REQUIREMENTS),
        "label_commitment": documents.label_commitment,
        "owner_key": W07_PRIVATE_OWNER_KEY,
        "payload_commitment": documents.payload_commitment,
        "self_excluded": 1,
    }
    encoded = canonical_json_bytes(freeze)
    target = root / W07_PRIVATE_FAMILY_FREEZE_NAME
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        raise W07PrivateEvaluationError("private family freeze is immutable") from error
    return target, _sha256(encoded)


def consume_w07_private_first_run_guard(
        artifact_root: str | Path,
        *,
        family_freeze_sha256: str,
        ) -> tuple[Path, str]:
    """排他消费当前 family 只允许一次的 formal run。"""
    root = Path(artifact_root).resolve()
    expected = _strict_sha256(family_freeze_sha256, label="family freeze")
    freeze_path = root / W07_PRIVATE_FAMILY_FREEZE_NAME
    if (not freeze_path.is_file() or freeze_path.is_symlink()
            or _sha256(freeze_path.read_bytes()) != expected):
        raise W07PrivateEvaluationError("private family freeze SHA drift")
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise W07PrivateEvaluationError("private family freeze unreadable") from error
    if (freeze.get("formal_run_count") != 0
            or freeze.get("ablation_order") != list(W07_PUBLIC_ABLATION_KEYS)
            or freeze.get("hard_requirements")
            != list(W07_PRIVATE_HARD_REQUIREMENTS)):
        raise W07PrivateEvaluationError("private family already ran or drifted")
    payload = canonical_json_bytes({
        "artifact_kind": "PH2_W07_PRIVATE_FIRST_RUN_GUARD",
        "family_freeze_sha256": expected,
        "formal_run_count_after": 1,
        "formal_run_count_before": 0,
        "format_version": 1,
    })
    target = root / W07_PRIVATE_FIRST_RUN_GUARD_NAME
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise W07PrivateEvaluationError(
            "private family guard already consumed") from error
    return target, _sha256(payload)


__all__ = [
    "W07PrivateFamilyDocuments",
    "build_w07_private_family_documents",
    "consume_w07_private_first_run_guard",
    "publish_w07_private_family",
]
