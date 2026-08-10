"""Public facade for the successor V3 R4 Kyoto owner receipt."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import read_canonical_object
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v5 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V5_PATH,
    BLIND_PRIVATE_SOURCE_EXTENSION_V5_VERSION,
    read_blind_private_source_extension_v5_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r4_audit import (
    build_w02_morphology_successor_v3_private_owner_r4_receipt_from_metadata,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r4_contract import (
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02_MORPH_V3_PRIVATE_PATHS,
    W02_MORPH_V3_PRIVATE_R4_METADATA_SHA256,
    W02_MORPH_V3_PRIVATE_R4_METADATA_SIZE_BYTES,
    W02_MORPH_V3_PRIVATE_R4_OWNER_FAMILY_KEY,
    W02_MORPH_V3_PRIVATE_R4_OWNER_ID,
    W02_MORPH_V3_PRIVATE_R4_OWNER_RECEIPT_PATH,
    W02_MORPH_V3_PRIVATE_R4_OWNER_RECEIPT_VERSION,
    W02_MORPH_V3_PRIVATE_R4_PUBLIC_BASE_COMMIT,
    W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_CODE_SHA256,
    W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_MANIFEST_SHA256,
    W02_MORPH_V3_PRIVATE_R4_SOURCE_KEY,
    W02MorphologySuccessorV3PrivateOwnerR4Error,
    W02MorphologySuccessorV3PrivateR4FileIdentity,
    require_exact_dict,
    validate_r4_private_file_inventory,
)


_SOURCE_EXTENSION_V5_CODE_PATH = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_blind_private_source_extension_v5.py"
)


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (repository / Path(*pure.parts)).resolve()
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative or ".." in pure.parts
            or not target.is_relative_to(repository) or target.is_symlink()
            or not target.is_file()):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 owner receipt repository path is invalid")
    return target


def _sha256_file(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _metadata_projection(raw: dict[str, Any]) -> dict[str, object]:
    files = []
    for item in raw["files"]:
        row = dict(item)
        files.append({
            "content_sha256": row["content_sha256"],
            "content_size_bytes": row["content_size_bytes"],
            "first_record_key": row["first_record_key"],
            "kind": row["record_kind"],
            "last_record_key": row["last_record_key"],
            "license_ids": row["license_ids"],
            "record_count": row["record_count"],
            "relative_path": row["relative_path"],
            "split": row["split"] or None,
            "transport_sha256": row["transport_sha256"],
            "transport_size_bytes": row["transport_size_bytes"],
        })
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_SUCCESSOR_V3_R4_BLIND_PRIVATE_OWNER_METADATA"),
        "artifact_version": (
            "PH2-D03-V2-W02-SUCCESSOR-V3-R4-OWNER-METADATA-V1"),
        "audit_identity": raw["owner_audit_identity"],
        "commitments": raw["commitments"],
        "contamination_audit": raw["contamination_audit"],
        "dimension_denominators": raw["dimension_denominator_counts"],
        "domain_disjoint_audit": raw["domain_disjoint_audit"],
        "double_pass_audit": raw["double_pass_audit"],
        "duplicate_audit": raw["duplicate_audit"],
        "file_count": raw["file_count"],
        "files": files,
        "label_semantic_binding_audit": raw["label_semantic_binding_audit"],
        "next_action": (
            "BUILD_SUCCESSOR_V3_PRIVATE_R4_OWNER_RECEIPT_IO_AND_EVALUATOR_REVISION"),
        "owner_family_key": raw["owner_family_key"],
        "owner_id": raw["owner_id"],
        "pair_count": raw["pair_count"],
        "public_commit": raw["public_identity"]["public_repository_commit"],
        "resource_limits": raw["resource_limits"],
        "resource_usage": raw["resource_usage"],
        "seal_identity": raw["owner_seal_identity"],
        "source_count": raw["source_count"],
        "source_snapshot": raw["source_snapshot"],
        "source_validator_binding_audit": raw["source_validator_binding_audit"],
        "split_counts": raw["split_counts"],
        "status": "OWNER_METADATA_FROZEN_SOURCE_V5_LABEL_BINDING_VERIFIED",
        "v5_identity": {
            "code_sha256": raw["public_identity"][
                "source_extension_v5_code_sha256"],
            "manifest_sha256": raw["public_identity"][
                "source_extension_v5_manifest_sha256"],
            "validator_version": BLIND_PRIVATE_SOURCE_EXTENSION_V5_VERSION,
        },
        "zero_call_audit": raw["zero_call_audit"],
    }


def validate_w02_morphology_successor_v3_private_owner_r4_receipt(
        value: object, repository_root: str | Path,
        ) -> tuple[
            dict[str, Any],
            tuple[W02MorphologySuccessorV3PrivateR4FileIdentity, ...],
        ]:
    """Validate the R4 safe receipt without accepting a private root."""
    raw = require_exact_dict(value, {
        "artifact_kind", "artifact_version", "candidate_evaluation_runs",
        "commitments", "contamination_audit", "dimension_denominator_counts",
        "domain_disjoint_audit", "double_pass_audit", "duplicate_audit",
        "file_count", "files", "formal_private_evaluation_runs",
        "label_record_count", "label_semantic_binding_audit",
        "main_session_private_payload_reads", "main_session_source_payload_reads",
        "namespace_policy", "next_action", "owner_audit_identity",
        "owner_family_key", "owner_id", "owner_metadata_sha256",
        "owner_metadata_size_bytes", "owner_seal_identity", "pair_count",
        "public_identity", "resource_limits", "resource_usage", "source_count",
        "source_key", "source_snapshot", "source_snapshot_commitment",
        "source_validator_binding_audit", "split_counts", "status",
        "teacher_llm_provenance", "zero_call_audit",
    }, where="R4 owner receipt")
    if (raw["artifact_kind"]
            != "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_OWNER_R4_RECEIPT"
            or raw["artifact_version"]
            != W02_MORPH_V3_PRIVATE_R4_OWNER_RECEIPT_VERSION
            or raw["status"]
            != "OWNER_METADATA_INGESTED_SOURCE_V5_PAYLOAD_UNREAD"
            or raw["next_action"]
            != "BUILD_SUCCESSOR_V3_PRIVATE_R4_EVALUATOR_FAMILY_REVISION"
            or raw["owner_id"] != W02_MORPH_V3_PRIVATE_R4_OWNER_ID
            or raw["owner_family_key"]
            != W02_MORPH_V3_PRIVATE_R4_OWNER_FAMILY_KEY
            or raw["owner_metadata_sha256"]
            != W02_MORPH_V3_PRIVATE_R4_METADATA_SHA256
            or raw["owner_metadata_size_bytes"]
            != W02_MORPH_V3_PRIVATE_R4_METADATA_SIZE_BYTES
            or raw["source_key"] != W02_MORPH_V3_PRIVATE_R4_SOURCE_KEY
            or any(raw[name] != 0 for name in (
                "candidate_evaluation_runs", "formal_private_evaluation_runs",
                "main_session_private_payload_reads",
                "main_session_source_payload_reads"))
            or raw["teacher_llm_provenance"]
            != {"llm_calls": 0, "teacher_calls": 0}):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 owner receipt identity or zero-call state drifted")
    files = validate_r4_private_file_inventory(raw["files"])
    rebuilt = (
        build_w02_morphology_successor_v3_private_owner_r4_receipt_from_metadata(
            _metadata_projection(raw)))
    if rebuilt != raw:
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 owner receipt no longer matches its safe metadata projection")

    repository = Path(repository_root).resolve()
    extension = read_blind_private_source_extension_v5_manifest(repository)
    code_path = _repository_file(repository, _SOURCE_EXTENSION_V5_CODE_PATH)
    manifest_path = _repository_file(
        repository, BLIND_PRIVATE_SOURCE_EXTENSION_V5_PATH)
    code_size, code_sha = _sha256_file(code_path)
    manifest_size, manifest_sha = _sha256_file(manifest_path)
    public = raw["public_identity"]
    if (extension.get("status") != "BLIND_PRIVATE_SOURCE_EXTENSION_V5_APPROVED"
            or extension.get("sources", [{}])[0].get("source_key")
            != W02_MORPH_V3_PRIVATE_R4_SOURCE_KEY
            or extension.get("sources", [{}])[0].get("license_id")
            != "CC-BY-SA-4.0"
            or public != {
                "public_repository_commit":
                    W02_MORPH_V3_PRIVATE_R4_PUBLIC_BASE_COMMIT,
                "source_extension_v5_code_sha256":
                    W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_CODE_SHA256,
                "source_extension_v5_manifest_sha256":
                    W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_MANIFEST_SHA256,
                "source_extension_v5_status":
                    "BLIND_PRIVATE_SOURCE_EXTENSION_V5_APPROVED",
            }
            or code_sha != W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_CODE_SHA256
            or manifest_sha
            != W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_MANIFEST_SHA256
            or code_size != 13_913 or manifest_size != 6_287):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 owner live V5 binding drifted")
    return raw, files


def read_w02_morphology_successor_v3_private_owner_r4_receipt(
        repository_root: str | Path,
        ) -> tuple[
            dict[str, Any],
            tuple[W02MorphologySuccessorV3PrivateR4FileIdentity, ...],
        ]:
    repository = Path(repository_root).resolve()
    target = _repository_file(
        repository, W02_MORPH_V3_PRIVATE_R4_OWNER_RECEIPT_PATH)
    value = read_canonical_object(target)
    return validate_w02_morphology_successor_v3_private_owner_r4_receipt(
        value, repository)


__all__ = [
    "W02_MORPH_V3_PRIVATE_LAYOUTS",
    "W02_MORPH_V3_PRIVATE_PATHS",
    "W02_MORPH_V3_PRIVATE_R4_METADATA_SHA256",
    "W02_MORPH_V3_PRIVATE_R4_OWNER_FAMILY_KEY",
    "W02_MORPH_V3_PRIVATE_R4_OWNER_RECEIPT_PATH",
    "W02MorphologySuccessorV3PrivateOwnerR4Error",
    "W02MorphologySuccessorV3PrivateR4FileIdentity",
    "read_w02_morphology_successor_v3_private_owner_r4_receipt",
    "validate_w02_morphology_successor_v3_private_owner_r4_receipt",
]
