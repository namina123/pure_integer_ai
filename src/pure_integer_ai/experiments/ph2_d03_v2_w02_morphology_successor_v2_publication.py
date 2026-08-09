"""W-02 morphology successor V2 formal artifact 的公开 receipt。"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_contract import (
    read_w02_morphology_successor_v2_runtime_freeze,
    verify_w02_morphology_successor_v2_consumed_guard,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    W02_MORPH_V2_OVERLAY_DB,
    W02_MORPH_V2_OVERLAY_MANIFEST,
    read_w02_morphology_successor_v2_overlay_artifact,
)


W02_MORPH_V2_RECEIPT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V2-OVERLAY-RECEIPT-V1")
W02_MORPH_V2_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v2_overlay_artifact_v1.json")


# object-model: exception
class W02MorphologySuccessorV2PublicationError(RuntimeError):
    """V2 overlay artifact 不能投影为正式零泄漏 receipt。"""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def build_w02_morphology_successor_v2_receipt(
        repository_root: str | Path, artifact_root: str | Path,
        ) -> dict[str, object]:
    """严格回读 formal V2 overlay，只公开身份、计数与零写状态。"""
    repository = Path(repository_root).resolve()
    artifact = Path(artifact_root).resolve()
    freeze = read_w02_morphology_successor_v2_runtime_freeze(repository)
    manifest_path = artifact / W02_MORPH_V2_OVERLAY_MANIFEST
    manifest = read_canonical_object(manifest_path)
    result = read_w02_morphology_successor_v2_overlay_artifact(artifact)
    identity = manifest.get("run_identity")
    if (not isinstance(identity, dict)
            or manifest.get("run_scope") != "FORMAL_SUCCESSOR_V2_TRANSFORM"
            or identity.get("run_scope") != "FORMAL_SUCCESSOR_V2_TRANSFORM"
            or manifest.get("formal_successor_v2_transform_runs") != 1
            or manifest.get("formal_training_runs") != 0
            or manifest.get("formal_private_evaluation_runs") != 0
            or manifest.get("private_payload_reads") != 0
            or manifest.get("teacher_calls") != 0
            or manifest.get("candidate_writes") != 0
            or manifest.get("v1_overlay_writes") != 0
            or manifest.get("v2_overlay_writes") != 1):
        raise W02MorphologySuccessorV2PublicationError(
            "V2 artifact 不是正式零泄漏 transform")
    expected = {
        "accepted_lexeme_rows": 112,
        "accepted_support_count": 651,
        "logic_operations": 4_611,
        "rule_row_count": 383,
        "unsupported_lexeme_rows": 4,
        "unsupported_support_count": 4,
    }
    if (result.semantic_sha256 != freeze.expected_semantic_sha256
            or result.run_identity_sha256
            != manifest.get("run_identity_sha256")
            or identity.get("runtime_freeze_sha256") != freeze.sha256()
            or identity.get("parent_candidate_manifest_sha256")
            != freeze.parent_candidate_manifest_sha256
            or identity.get("parent_candidate_semantic_sha256")
            != freeze.parent_candidate_semantic_sha256
            or identity.get("parent_v1_overlay_manifest_sha256")
            != freeze.parent_v1_manifest_sha256
            or identity.get("parent_v1_overlay_semantic_sha256")
            != freeze.parent_v1_semantic_sha256
            or any(manifest.get(key) != value for key, value in expected.items())):
        raise W02MorphologySuccessorV2PublicationError(
            "V2 artifact 与 freeze/count 漂移")
    verify_w02_morphology_successor_v2_consumed_guard(
        artifact.parent.parent,
        expected_guard_sha256=freeze.first_run_guard_sha256,
        run_id=int(identity["run_id"]),
        run_identity_sha256=result.run_identity_sha256)
    db_rows = tuple(item for item in manifest["tree"]
                    if item.get("path") == W02_MORPH_V2_OVERLAY_DB)
    if len(db_rows) != 1:
        raise W02MorphologySuccessorV2PublicationError(
            "V2 overlay DB tree identity 不唯一")
    manifest_size, manifest_sha = _sha256_file(manifest_path)
    return {
        "accepted_lexeme_rows": result.accepted_lexeme_rows,
        "accepted_support_count": result.accepted_support_count,
        "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_RECEIPT",
        "artifact_version": W02_MORPH_V2_RECEIPT_VERSION,
        "candidate_writes": 0,
        "first_run_guard_sha256": freeze.first_run_guard_sha256,
        "formal_private_evaluation_runs": 0,
        "formal_successor_v2_transform_runs": 1,
        "formal_training_runs": 0,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "logic_operations": result.logic_operations,
        "logical_shard_count": result.shard_count,
        "next_action": "W02_SUCCESSOR_V2_DEV_CALIBRATION_FREEZE",
        "operation_kind": manifest["run_identity"]["operation_kind"],
        "parent_candidate_manifest_sha256": (
            freeze.parent_candidate_manifest_sha256),
        "parent_candidate_semantic_sha256": (
            freeze.parent_candidate_semantic_sha256),
        "parent_formal_successor_transform_runs": 1,
        "parent_formal_training_runs": 1,
        "parent_v1_overlay_manifest_sha256": freeze.parent_v1_manifest_sha256,
        "parent_v1_overlay_semantic_sha256": freeze.parent_v1_semantic_sha256,
        "private_family_registered": 0,
        "private_payload_reads": 0,
        "release_key": "PH2-D03-V2",
        "rule_row_count": result.rule_row_count,
        "run_identity_sha256": result.run_identity_sha256,
        "runtime_freeze_sha256": freeze.sha256(),
        "semantic_sha256": result.semantic_sha256,
        "shadow_started": 0,
        "stage_key": "W-02",
        "status": "W02_MORPHOLOGY_SUCCESSOR_V2_ARTIFACT_FROZEN",
        "teacher_calls": 0,
        "unsupported_lexeme_rows": result.unsupported_lexeme_rows,
        "unsupported_support_count": result.unsupported_support_count,
        "v1_overlay_writes": 0,
        "v2_overlay_artifact_manifest_sha256": manifest_sha,
        "v2_overlay_artifact_manifest_size_bytes": manifest_size,
        "v2_overlay_db_sha256": str(db_rows[0]["sha256"]),
        "v2_overlay_db_size_bytes": int(db_rows[0]["size_bytes"]),
        "v2_overlay_tree_commitment": hashlib.sha256(canonical_json_bytes(
            manifest["tree"])).hexdigest(),
        "v2_overlay_writes": 1,
        "worker_counts_supported": [1, 2, 4],
    }


def publish_w02_morphology_successor_v2_receipt(
        repository_root: str | Path, artifact_root: str | Path,
        ) -> Path:
    """不可覆盖发布 formal V2 successor receipt。"""
    repository = Path(repository_root).resolve()
    value = build_w02_morphology_successor_v2_receipt(
        repository, artifact_root)
    target = repository / Path(*PurePosixPath(W02_MORPH_V2_RECEIPT_PATH).parts)
    write_immutable_json(value, target)
    if target.read_bytes() != canonical_json_bytes(value) + b"\n":
        raise W02MorphologySuccessorV2PublicationError(
            "V2 receipt 发布字节漂移")
    return target


__all__ = [
    "W02_MORPH_V2_RECEIPT_PATH",
    "W02MorphologySuccessorV2PublicationError",
    "build_w02_morphology_successor_v2_receipt",
    "publish_w02_morphology_successor_v2_receipt",
]
