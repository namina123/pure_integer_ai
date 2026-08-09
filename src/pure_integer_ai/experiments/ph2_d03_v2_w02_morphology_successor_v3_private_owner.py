"""Public facade for the successor V3 PUD-news owner receipt."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import read_canonical_object
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v2 import (
    V3_SHADOW_REPORT_SHA256,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v3 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V3_PATH,
    PARENT_EXTENSION_V2_MANIFEST_SHA256,
    read_blind_private_source_extension_v3_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_audit import (
    validate_v3_private_owner_audits,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_contract import (
    W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS,
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02_MORPH_V3_PRIVATE_OWNER_FAMILY_KEY,
    W02_MORPH_V3_PRIVATE_OWNER_ID,
    W02_MORPH_V3_PRIVATE_OWNER_METADATA_SHA256,
    W02_MORPH_V3_PRIVATE_OWNER_METADATA_SIZE_BYTES,
    W02_MORPH_V3_PRIVATE_OWNER_RECEIPT_PATH,
    W02_MORPH_V3_PRIVATE_OWNER_RECEIPT_VERSION,
    W02_MORPH_V3_PRIVATE_PAIR_COUNT,
    W02_MORPH_V3_PRIVATE_PATHS,
    W02_MORPH_V3_PRIVATE_PUBLIC_BASE_COMMIT,
    W02_MORPH_V3_PRIVATE_SOURCE_COUNT,
    W02_MORPH_V3_PRIVATE_SOURCE_EXTENSION_V3_SHA256,
    W02_MORPH_V3_PRIVATE_SOURCE_KEY,
    W02_MORPH_V3_PRIVATE_SPLIT_COUNTS,
    W02MorphologySuccessorV3PrivateFileIdentity,
    W02MorphologySuccessorV3PrivateOwnerError,
    require_exact_dict,
    validate_v3_private_file_inventory,
)


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (repository / Path(*pure.parts)).resolve()
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative or ".." in pure.parts
            or not target.is_relative_to(repository) or target.is_symlink()
            or not target.is_file()):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner receipt repository path is invalid")
    return target


def validate_w02_morphology_successor_v3_private_owner_receipt(
        value: object,
        repository_root: str | Path,
        ) -> tuple[
            dict[str, Any],
            tuple[W02MorphologySuccessorV3PrivateFileIdentity, ...],
        ]:
    """Validate the safe receipt without accepting an external owner path."""
    raw = require_exact_dict(value, {
        "artifact_kind", "artifact_version", "audit_report",
        "blocked_owner_disjoint_audit", "candidate_evaluation_runs",
        "commitments", "contamination_audit", "dimension_denominator_counts",
        "double_pass_audit", "file_count", "files",
        "formal_private_evaluation_runs", "main_session_conllu_payload_reads",
        "main_session_private_payload_reads", "next_action",
        "old_private_disjoint_audit", "owner_family_key", "owner_id",
        "owner_metadata_sha256", "owner_metadata_size_bytes", "owner_seal",
        "owner_teacher_llm_provenance", "pair_count",
        "previous_blocked_owner", "public_repository", "pud_domain_audit",
        "resource_budget", "source_count", "source_key",
        "source_snapshot_commitment", "split_counts", "status",
        "wikipedia_accepted_count", "within_owner_duplicate_audit",
        "within_owner_filter_audit",
    }, where="V3 owner receipt")
    if (raw["artifact_kind"]
            != "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_OWNER_RECEIPT"
            or raw["artifact_version"]
            != W02_MORPH_V3_PRIVATE_OWNER_RECEIPT_VERSION
            or raw["status"] != "OWNER_METADATA_INGESTED_PAYLOAD_UNREAD"
            or raw["next_action"]
            != "BUILD_SUCCESSOR_V3_PRIVATE_FAMILY_REGISTRATION_FREEZE"
            or raw["owner_family_key"]
            != W02_MORPH_V3_PRIVATE_OWNER_FAMILY_KEY
            or raw["owner_id"] != W02_MORPH_V3_PRIVATE_OWNER_ID
            or raw["owner_metadata_sha256"]
            != W02_MORPH_V3_PRIVATE_OWNER_METADATA_SHA256
            or raw["owner_metadata_size_bytes"]
            != W02_MORPH_V3_PRIVATE_OWNER_METADATA_SIZE_BYTES
            or raw["file_count"] != len(W02_MORPH_V3_PRIVATE_LAYOUTS)
            or raw["source_count"] != W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            or raw["source_key"] != W02_MORPH_V3_PRIVATE_SOURCE_KEY
            or raw["pair_count"] != W02_MORPH_V3_PRIVATE_PAIR_COUNT
            or raw["split_counts"] != W02_MORPH_V3_PRIVATE_SPLIT_COUNTS
            or raw["dimension_denominator_counts"]
            != W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS
            or raw["wikipedia_accepted_count"] != 0
            or any(raw[key] != 0 for key in (
                "candidate_evaluation_runs", "formal_private_evaluation_runs",
                "main_session_conllu_payload_reads",
                "main_session_private_payload_reads"))):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner receipt identity or status drifted")

    files = validate_v3_private_file_inventory(raw["files"])
    validate_v3_private_owner_audits(raw)
    public = require_exact_dict(raw["public_repository"], {
        "head_commit_sha1", "source_extension_v2_sha256",
        "source_extension_v3_sha256", "v3_shadow_report_sha256",
    }, where="V3 owner public repository")
    if (public["head_commit_sha1"] != W02_MORPH_V3_PRIVATE_PUBLIC_BASE_COMMIT
            or public["source_extension_v2_sha256"]
            != PARENT_EXTENSION_V2_MANIFEST_SHA256
            or public["source_extension_v3_sha256"]
            != W02_MORPH_V3_PRIVATE_SOURCE_EXTENSION_V3_SHA256
            or public["v3_shadow_report_sha256"]
            != V3_SHADOW_REPORT_SHA256):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner public dependency binding drifted")

    repository = Path(repository_root).resolve()
    extension = read_blind_private_source_extension_v3_manifest(repository)
    extension_path = _repository_file(
        repository, BLIND_PRIVATE_SOURCE_EXTENSION_V3_PATH)
    source = extension["sources"][0]
    source_filter = source["data_file"]["owner_filter"]
    if (extension["status"] != "BLIND_PRIVATE_SOURCE_EXTENSION_V3_APPROVED"
            or hashlib.sha256(extension_path.read_bytes()).hexdigest()
            != public["source_extension_v3_sha256"]
            or source["source_key"] != W02_MORPH_V3_PRIVATE_SOURCE_KEY
            or source["license_id"] != "CC-BY-SA-3.0"
            or source_filter["fixed_blob_news_sentence_count"] != 500
            or source_filter["fixed_blob_wikipedia_sentence_count"] != 500):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner source extension binding drifted")
    return raw, files


def read_w02_morphology_successor_v3_private_owner_receipt(
        repository_root: str | Path,
        ) -> tuple[
            dict[str, Any],
            tuple[W02MorphologySuccessorV3PrivateFileIdentity, ...],
        ]:
    """Read the canonical public receipt without accepting a private path."""
    repository = Path(repository_root).resolve()
    target = _repository_file(repository, W02_MORPH_V3_PRIVATE_OWNER_RECEIPT_PATH)
    value = read_canonical_object(target)
    return validate_w02_morphology_successor_v3_private_owner_receipt(
        value, repository)


__all__ = [
    "W02_MORPH_V3_PRIVATE_LAYOUTS",
    "W02_MORPH_V3_PRIVATE_OWNER_FAMILY_KEY",
    "W02_MORPH_V3_PRIVATE_OWNER_METADATA_SHA256",
    "W02_MORPH_V3_PRIVATE_OWNER_METADATA_SIZE_BYTES",
    "W02_MORPH_V3_PRIVATE_OWNER_RECEIPT_PATH",
    "W02_MORPH_V3_PRIVATE_PAIR_COUNT",
    "W02_MORPH_V3_PRIVATE_PATHS",
    "W02_MORPH_V3_PRIVATE_SOURCE_COUNT",
    "W02MorphologySuccessorV3PrivateFileIdentity",
    "W02MorphologySuccessorV3PrivateOwnerError",
    "read_w02_morphology_successor_v3_private_owner_receipt",
    "validate_w02_morphology_successor_v3_private_owner_receipt",
]
