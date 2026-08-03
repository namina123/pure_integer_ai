"""W-07 evaluator v2 family 冻结与唯一 guard。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_PUBLIC_ABLATION_KEYS,
    W07_PUBLIC_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_contract import (
    W07_EVALUATOR_FAILURE_PHASES,
    W07_PRIVATE_FAMILY_FREEZE_NAME,
    W07_PRIVATE_FIRST_RUN_GUARD_NAME,
    W07_PRIVATE_HARD_REQUIREMENTS,
    W07_PRIVATE_OWNER_KEY,
    W07PrivateEvaluationError,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_family import (
    W07PrivateFamilyDocuments,
    build_w07_private_family_documents,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_v2_contract import (
    W07_V2_EVALUATOR_VERSION,
    W07_V2_FAILURE_KINDS,
    W07_V2_OPERATIONS,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_root(root: Path, forbidden_roots: tuple[str | Path, ...]) -> None:
    for value in forbidden_roots:
        forbidden = Path(value).resolve()
        if (root == forbidden or root.is_relative_to(forbidden)
                or forbidden.is_relative_to(root)):
            raise W07PrivateEvaluationError(
                "W-07 v2 private family root is not isolated")


def build_w07_v2_private_family_documents(
        *,
        candidate_contract_sha256: str,
        candidate_host_freeze_sha256: str,
        evaluator_public_head_commit_sha1: str,
        nonce: tuple[int, ...],
        ) -> W07PrivateFamilyDocuments:
    """复用无 surface 文档格式，并由 v2 freeze 绑定诊断合同。"""
    return build_w07_private_family_documents(
        candidate_contract_sha256=candidate_contract_sha256,
        candidate_host_freeze_sha256=candidate_host_freeze_sha256,
        evaluator_public_head_commit_sha1=evaluator_public_head_commit_sha1,
        nonce=nonce,
    )


def publish_w07_v2_private_family(
        artifact_root: str | Path,
        documents: W07PrivateFamilyDocuments,
        *,
        forbidden_roots: tuple[str | Path, ...] = (),
        ) -> tuple[Path, str]:
    """先排他写五份 owner 文档，再冻结 v2 诊断与运行合同。"""
    if not isinstance(documents, W07PrivateFamilyDocuments):
        raise TypeError("W-07 v2 private family documents type drift")
    root = Path(artifact_root).resolve()
    _validate_root(root, forbidden_roots)
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in documents.files():
        try:
            with (root / name).open("xb") as handle:
                handle.write(payload)
        except FileExistsError as error:
            raise W07PrivateEvaluationError(
                "W-07 v2 private family file is immutable") from error
    inventory = [
        {"path": name, "sha256": _sha256(payload), "size_bytes": len(payload)}
        for name, payload in documents.files()
    ]
    freeze = {
        "ablation_order": list(W07_PUBLIC_ABLATION_KEYS),
        "artifact_kind": "PH2_W07_PRIVATE_FAMILY_V2_FREEZE",
        "candidate_contract_sha256": documents.candidate_contract_sha256,
        "candidate_host_freeze_sha256": documents.candidate_host_freeze_sha256,
        "case_commitment": documents.case_commitment,
        "cluster_commitment": documents.cluster_commitment,
        "diagnostic_contract": {
            "dimension_order": list(W07_PUBLIC_DIMENSION_KEYS),
            "failure_kinds": list(W07_V2_FAILURE_KINDS),
            "failure_phases": list(W07_EVALUATOR_FAILURE_PHASES),
            "operations": list(W07_V2_OPERATIONS),
        },
        "evaluator_public_head_commit_sha1": (
            documents.evaluator_public_head_commit_sha1),
        "evaluator_version": W07_V2_EVALUATOR_VERSION,
        "family_key": documents.family_key,
        "file_inventory": inventory,
        "formal_run_count": 0,
        "format_version": 2,
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
        raise W07PrivateEvaluationError(
            "W-07 v2 private family freeze is immutable") from error
    return target, _sha256(encoded)


def consume_w07_v2_private_first_run_guard(
        artifact_root: str | Path,
        *,
        family_freeze_sha256: str,
        ) -> tuple[Path, str]:
    """消费 v2 family 的唯一正式运行权。"""
    root = Path(artifact_root).resolve()
    freeze = root / W07_PRIVATE_FAMILY_FREEZE_NAME
    if (not freeze.is_file()
            or _sha256(freeze.read_bytes()) != family_freeze_sha256):
        raise W07PrivateEvaluationError("W-07 v2 family freeze SHA drift")
    guard = canonical_json_bytes({
        "artifact_kind": "PH2_W07_PRIVATE_V2_FIRST_RUN_GUARD",
        "evaluator_version": W07_V2_EVALUATOR_VERSION,
        "family_freeze_sha256": family_freeze_sha256,
        "formal_run_count_after": 1,
        "formal_run_count_before": 0,
        "format_version": 2,
    })
    target = root / W07_PRIVATE_FIRST_RUN_GUARD_NAME
    try:
        with target.open("xb") as handle:
            handle.write(guard)
    except FileExistsError as error:
        raise W07PrivateEvaluationError(
            "W-07 v2 private evaluator already consumed") from error
    return target, _sha256(guard)


__all__ = [
    "build_w07_v2_private_family_documents",
    "consume_w07_v2_private_first_run_guard",
    "publish_w07_v2_private_family",
]
