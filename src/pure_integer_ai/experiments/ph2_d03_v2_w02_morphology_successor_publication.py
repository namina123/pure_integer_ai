"""W-02 morphology successor formal artifact 的安全公开 receipt。"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    W02_MORPH_OVERLAY_DB_NAME,
    W02_MORPH_OVERLAY_MANIFEST_NAME,
    read_w02_morphology_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_contract import (
    read_w02_morphology_successor_runtime_freeze,
    verify_w02_morphology_successor_consumed_guard,
)


W02_MORPH_SUCCESSOR_RECEIPT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-RECEIPT-V1")
W02_MORPH_SUCCESSOR_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_artifact_v1.json"
)


# object-model: exception
class W02MorphologySuccessorPublicationError(RuntimeError):
    """successor artifact 不能投影为正式零泄漏 receipt。"""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def build_w02_morphology_successor_receipt(
        repository_root: str | Path,
        artifact_root: str | Path,
        ) -> dict[str, object]:
    """严格回读 formal overlay，只公开摘要、计数与状态。"""
    repository = Path(repository_root).resolve()
    artifact = Path(artifact_root).resolve()
    freeze = read_w02_morphology_successor_runtime_freeze(repository)
    manifest_path = artifact / W02_MORPH_OVERLAY_MANIFEST_NAME
    manifest = read_canonical_object(manifest_path)
    result = read_w02_morphology_overlay_artifact(artifact)
    if (manifest.get("run_scope") != "FORMAL_SUCCESSOR_TRANSFORM"
            or manifest.get("formal_successor_transform_runs") != 1
            or manifest.get("formal_training_runs") != 0
            or manifest.get("formal_private_evaluation_runs") != 0
            or manifest.get("private_payload_reads") != 0
            or manifest.get("teacher_calls") != 0
            or manifest.get("candidate_writes") != 0
            or manifest.get("overlay_writes") != 1):
        raise W02MorphologySuccessorPublicationError(
            "successor artifact 不是正式零泄漏 transform")
    expected = {
        "logic_operations": 1_574_251,
        "morphology_observation_count": 3_997,
        "morphology_token_count": 97_959,
        "rule_row_count": 47_975,
        "training_pair_count": 51_200,
    }
    if (result.overlay_semantic_sha256
            != freeze.expected_overlay_semantic_sha256
            or result.parent_candidate_semantic_sha256
            != freeze.parent_candidate_semantic_sha256
            or manifest.get("parent_candidate_manifest_sha256")
            != freeze.parent_candidate_manifest_sha256
            or manifest.get("run_identity", {}).get("runtime_freeze_sha256")
            != freeze.sha256()
            or any(manifest.get(key) != value for key, value in expected.items())):
        raise W02MorphologySuccessorPublicationError(
            "successor artifact 与 freeze/count 漂移")
    verify_w02_morphology_successor_consumed_guard(
        artifact.parent.parent,
        expected_guard_sha256=freeze.first_run_guard_sha256,
        run_id=int(manifest["run_identity"]["run_id"]),
        run_identity_sha256=result.run_identity_sha256,
    )
    db_rows = tuple(
        item for item in manifest["tree"]
        if item.get("path") == W02_MORPH_OVERLAY_DB_NAME)
    if len(db_rows) != 1:
        raise W02MorphologySuccessorPublicationError(
            "successor overlay DB tree identity 不唯一")
    manifest_size, manifest_sha = _sha256_file(manifest_path)
    return {
        "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_RECEIPT",
        "artifact_version": W02_MORPH_SUCCESSOR_RECEIPT_VERSION,
        "candidate_writes": 0,
        "first_run_guard_sha256": freeze.first_run_guard_sha256,
        "formal_private_evaluation_runs": 0,
        "formal_successor_transform_runs": 1,
        "formal_training_runs": 0,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "logic_operations": result.logic_operations,
        "logical_shard_count": result.shard_count,
        "morphology_observation_count": result.morphology_observation_count,
        "morphology_token_count": result.morphology_token_count,
        "next_action": "W02_SUCCESSOR_DEV_CALIBRATION_FREEZE",
        "operation_kind": manifest["operation_kind"],
        "overlay_artifact_manifest_sha256": manifest_sha,
        "overlay_artifact_manifest_size_bytes": manifest_size,
        "overlay_db_sha256": str(db_rows[0]["sha256"]),
        "overlay_db_size_bytes": int(db_rows[0]["size_bytes"]),
        "overlay_semantic_sha256": result.overlay_semantic_sha256,
        "overlay_tree_commitment": hashlib.sha256(canonical_json_bytes(
            manifest["tree"])).hexdigest(),
        "overlay_writes": 1,
        "parent_candidate_manifest_sha256": (
            freeze.parent_candidate_manifest_sha256),
        "parent_candidate_semantic_sha256": (
            freeze.parent_candidate_semantic_sha256),
        "parent_formal_training_runs": 1,
        "private_family_registered": 0,
        "private_payload_reads": 0,
        "release_key": "PH2-D03-V2",
        "rule_row_count": result.rule_row_count,
        "run_identity_sha256": result.run_identity_sha256,
        "runtime_freeze_sha256": freeze.sha256(),
        "shadow_started": 0,
        "stage_key": "W-02",
        "status": "W02_MORPHOLOGY_SUCCESSOR_ARTIFACT_FROZEN",
        "teacher_calls": 0,
        "training_pair_count": result.training_pair_count,
        "worker_counts_supported": [1, 2, 4],
    }


def publish_w02_morphology_successor_receipt(
        repository_root: str | Path,
        artifact_root: str | Path,
        ) -> Path:
    """不可覆盖发布 formal successor receipt。"""
    repository = Path(repository_root).resolve()
    value = build_w02_morphology_successor_receipt(repository, artifact_root)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_RECEIPT_PATH).parts)
    write_immutable_json(value, target)
    if target.read_bytes() != canonical_json_bytes(value) + b"\n":
        raise W02MorphologySuccessorPublicationError(
            "successor receipt 发布字节漂移")
    return target


__all__ = [
    "W02_MORPH_SUCCESSOR_RECEIPT_PATH",
    "W02MorphologySuccessorPublicationError",
    "build_w02_morphology_successor_receipt",
    "publish_w02_morphology_successor_receipt",
]
