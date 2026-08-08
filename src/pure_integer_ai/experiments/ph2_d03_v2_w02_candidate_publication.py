"""PH2-D03-V2 W-02 Candidate artifact 的安全公开 receipt。"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    W02_CANDIDATE_MANIFEST_NAME,
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    read_w02_compile_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_runtime_contract import (
    read_w02_candidate_runtime_freeze,
)


W02_CANDIDATE_RECEIPT_VERSION = "PH2-D03-V2-W02-CANDIDATE-RECEIPT-V1"
W02_CANDIDATE_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_candidate_artifact_v1.json"
)


# object-model: exception
class W02CandidatePublicationError(RuntimeError):
    """Candidate artifact 不能安全投影为正式公开 receipt。"""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def build_w02_candidate_receipt(
        repository_root: str | Path,
        artifact_root: str | Path,
        ) -> dict[str, object]:
    """回读正式 artifact，仅投影 hash、计数和零泄漏状态。"""
    repository = Path(repository_root).resolve()
    artifact = Path(artifact_root).resolve()
    manifest_path = artifact / W02_CANDIDATE_MANIFEST_NAME
    manifest = read_canonical_object(manifest_path)
    if (manifest.get("run_scope") != "FORMAL"
            or manifest.get("formal_training_runs") != 1
            or manifest.get("formal_private_evaluation_runs") != 0
            or manifest.get("private_payload_reads") != 0
            or manifest.get("teacher_calls") != 0):
        raise W02CandidatePublicationError("Candidate artifact 不是正式零泄漏运行")
    parent = read_w02_compile_freeze(repository)
    runtime = read_w02_candidate_runtime_freeze(repository)
    result = read_w02_candidate_artifact(artifact)
    if (result.pair_count != parent.plan.split_total("train")
            or manifest.get("pack_commitment") != parent.pack_commitment
            or manifest.get("compile_freeze_sha256") != parent.sha256()
            or manifest.get("runtime_freeze_sha256") != runtime.sha256()):
        raise W02CandidatePublicationError("Candidate artifact 与 parent/runtime freeze 漂移")
    db_rows = tuple(item for item in manifest["tree"]
                    if item.get("path") == "candidate.sqlite3")
    if len(db_rows) != 1:
        raise W02CandidatePublicationError("Candidate DB tree identity 不唯一")
    manifest_size, manifest_sha = _sha256_file(manifest_path)
    return {
        "artifact_kind": "PH2_D03_V2_W02_CANDIDATE_RECEIPT",
        "artifact_version": W02_CANDIDATE_RECEIPT_VERSION,
        "candidate_artifact_manifest_sha256": manifest_sha,
        "candidate_artifact_manifest_size_bytes": manifest_size,
        "candidate_db_sha256": str(db_rows[0]["sha256"]),
        "candidate_db_size_bytes": int(db_rows[0]["size_bytes"]),
        "candidate_semantic_sha256": result.candidate_semantic_sha256,
        "candidate_tree_commitment": hashlib.sha256(canonical_json_bytes(
            manifest["tree"])).hexdigest(),
        "candidate_writes": 1,
        "compile_freeze_sha256": parent.sha256(),
        "first_run_guard_sha256": parent.first_run_guard_sha256,
        "formal_private_evaluation_runs": 0,
        "formal_training_runs": 1,
        "generated_probe_sha256": result.generated_probe_sha256,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "logic_operations": result.logic_operations,
        "logical_shard_count": result.shard_count,
        "next_action": "W02_DEV_CALIBRATION",
        "pack_commitment": parent.pack_commitment,
        "pair_count": result.pair_count,
        "private_payload_reads": 0,
        "release_key": "PH2-D03-V2",
        "run_identity_sha256": result.run_identity_sha256,
        "runtime_freeze_sha256": runtime.sha256(),
        "source_count": result.source_count,
        "stage_key": "W-02",
        "status": "W02_CANDIDATE_ARTIFACT_FROZEN",
        "teacher_calls": 0,
        "worker_counts_supported": [1, 2, 4],
    }


def publish_w02_candidate_receipt(
        repository_root: str | Path,
        artifact_root: str | Path,
        ) -> Path:
    """不可覆盖发布正式 Candidate receipt。"""
    repository = Path(repository_root).resolve()
    value = build_w02_candidate_receipt(repository, artifact_root)
    target = repository / Path(*PurePosixPath(W02_CANDIDATE_RECEIPT_PATH).parts)
    write_immutable_json(value, target)
    if target.read_bytes() != canonical_json_bytes(value) + b"\n":
        raise W02CandidatePublicationError("Candidate receipt 发布字节漂移")
    return target


__all__ = [
    "W02_CANDIDATE_RECEIPT_PATH",
    "W02CandidatePublicationError",
    "build_w02_candidate_receipt",
    "publish_w02_candidate_receipt",
]
