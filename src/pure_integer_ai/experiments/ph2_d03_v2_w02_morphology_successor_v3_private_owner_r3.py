"""Public facade for the successor V3 R3 PUD-Wikipedia owner receipt."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import read_canonical_object
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v4 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V4_PATH,
    PARENT_EXTENSION_V3_MANIFEST_SHA256,
    read_blind_private_source_extension_v4_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r3_audit import (
    validate_r3_owner_audits,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r3_contract import (
    W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS,
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02_MORPH_V3_PRIVATE_PAIR_COUNT,
    W02_MORPH_V3_PRIVATE_PATHS,
    W02_MORPH_V3_PRIVATE_R3_METADATA_SHA256,
    W02_MORPH_V3_PRIVATE_R3_METADATA_SIZE_BYTES,
    W02_MORPH_V3_PRIVATE_R3_OWNER_FAMILY_KEY,
    W02_MORPH_V3_PRIVATE_R3_OWNER_ID,
    W02_MORPH_V3_PRIVATE_R3_OWNER_RECEIPT_PATH,
    W02_MORPH_V3_PRIVATE_R3_OWNER_RECEIPT_VERSION,
    W02_MORPH_V3_PRIVATE_R3_PUBLIC_BASE_COMMIT,
    W02_MORPH_V3_PRIVATE_R3_SOURCE_EXTENSION_CODE_SHA256,
    W02_MORPH_V3_PRIVATE_R3_SOURCE_EXTENSION_MANIFEST_SHA256,
    W02_MORPH_V3_PRIVATE_R3_SOURCE_KEY,
    W02_MORPH_V3_PRIVATE_R3_SOURCE_SNAPSHOT_COMMITMENT,
    W02_MORPH_V3_PRIVATE_SOURCE_COUNT,
    W02_MORPH_V3_PRIVATE_SPLIT_COUNTS,
    W02MorphologySuccessorV3PrivateFileIdentity,
    W02MorphologySuccessorV3PrivateOwnerR3Error,
    require_exact_dict,
    require_sha256,
    validate_v3_private_file_inventory,
)


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (repository / Path(*pure.parts)).resolve()
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative or ".." in pure.parts
            or not target.is_relative_to(repository) or target.is_symlink()
            or not target.is_file()):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 owner receipt repository path is invalid")
    return target


def _identity(value: object, *, where: str,
              relative: str | None = None) -> dict[str, Any]:
    raw = require_exact_dict(value, {
        "relative_path", "sha256", "size_bytes",
    }, where=where)
    if (relative is not None and raw["relative_path"] != relative):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            f"{where} relative path drifted")
    require_sha256(raw["sha256"], where=f"{where} SHA")
    if type(raw["size_bytes"]) is not int or raw["size_bytes"] <= 0:
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            f"{where} size drifted")
    return raw


def validate_w02_morphology_successor_v3_private_owner_r3_receipt(
        value: object,
        repository_root: str | Path,
        ) -> tuple[
            dict[str, Any],
            tuple[W02MorphologySuccessorV3PrivateFileIdentity, ...],
        ]:
    """Validate the R3 safe receipt without accepting an owner path."""
    raw = require_exact_dict(value, {
        "artifact_kind", "artifact_version", "candidate_evaluation_runs",
        "commitments", "contamination", "dimension_denominator_counts",
        "domain_disjoint_audit", "double_pass_audit", "duplicate_audit",
        "file_count", "files", "formal_private_evaluation_runs",
        "input_rejections", "label_record_count",
        "label_semantic_binding_audit", "main_session_private_payload_reads",
        "main_session_source_payload_reads", "namespace_policy",
        "news_accepted_count", "next_action", "owner_audit_identity",
        "owner_family_key", "owner_id", "owner_metadata_sha256",
        "owner_metadata_size_bytes", "owner_seal_identity", "pair_count",
        "public_identity", "record_key_ranges", "resource_limits",
        "resource_usage", "source_count", "source_key",
        "source_snapshot_commitment", "source_snapshot_identity",
        "split_counts", "status", "teacher_llm_provenance",
        "zero_call_audit",
    }, where="R3 owner receipt")
    if (raw["artifact_kind"]
            != "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_OWNER_R3_RECEIPT"
            or raw["artifact_version"]
            != W02_MORPH_V3_PRIVATE_R3_OWNER_RECEIPT_VERSION
            or raw["status"] != "OWNER_METADATA_INGESTED_PAYLOAD_UNREAD"
            or raw["next_action"]
            != "BUILD_SUCCESSOR_V3_PRIVATE_R3_EVALUATOR_FAMILY_REVISION"
            or raw["owner_family_key"]
            != W02_MORPH_V3_PRIVATE_R3_OWNER_FAMILY_KEY
            or raw["owner_id"] != W02_MORPH_V3_PRIVATE_R3_OWNER_ID
            or raw["owner_metadata_sha256"]
            != W02_MORPH_V3_PRIVATE_R3_METADATA_SHA256
            or raw["owner_metadata_size_bytes"]
            != W02_MORPH_V3_PRIVATE_R3_METADATA_SIZE_BYTES
            or raw["source_key"] != W02_MORPH_V3_PRIVATE_R3_SOURCE_KEY
            or raw["source_count"] != W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            or raw["pair_count"] != W02_MORPH_V3_PRIVATE_PAIR_COUNT
            or raw["label_record_count"] != W02_MORPH_V3_PRIVATE_PAIR_COUNT
            or raw["split_counts"] != W02_MORPH_V3_PRIVATE_SPLIT_COUNTS
            or raw["dimension_denominator_counts"]
            != W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS
            or raw["source_snapshot_commitment"]
            != W02_MORPH_V3_PRIVATE_R3_SOURCE_SNAPSHOT_COMMITMENT
            or raw["file_count"] != len(W02_MORPH_V3_PRIVATE_LAYOUTS)
            or raw["news_accepted_count"] != 0
            or raw["input_rejections"] != {}
            or raw["teacher_llm_provenance"] != {
                "llm_calls": 0, "teacher_calls": 0}
            or any(raw[name] != 0 for name in (
                "candidate_evaluation_runs", "formal_private_evaluation_runs",
                "main_session_private_payload_reads",
                "main_session_source_payload_reads"))):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 owner receipt identity or zero-call state drifted")

    files = validate_v3_private_file_inventory(raw["files"])
    validate_r3_owner_audits(raw)
    commitments = require_exact_dict(raw["commitments"], {
        "case_commitment", "cluster_commitment", "label_commitment",
        "payload_commitment",
    }, where="R3 owner commitments")
    for name, item in commitments.items():
        require_sha256(item, where=f"R3 {name}")

    public = require_exact_dict(raw["public_identity"], {
        "public_repository_commit", "source_extension_v4_code",
        "source_extension_v4_manifest", "source_extension_v4_status",
    }, where="R3 public identity")
    code = require_exact_dict(public["source_extension_v4_code"], {
        "repository_path", "sha256", "size_bytes",
    }, where="R3 source extension code")
    manifest = require_exact_dict(public["source_extension_v4_manifest"], {
        "repository_path", "sha256", "size_bytes",
    }, where="R3 source extension manifest")
    if (public["public_repository_commit"]
            != W02_MORPH_V3_PRIVATE_R3_PUBLIC_BASE_COMMIT
            or public["source_extension_v4_status"]
            != "BLIND_PRIVATE_SOURCE_EXTENSION_V4_APPROVED"
            or code != {
                "repository_path": (
                    "src/pure_integer_ai/experiments/"
                    "ph2_d03_v2_blind_private_source_extension_v4.py"),
                "sha256": W02_MORPH_V3_PRIVATE_R3_SOURCE_EXTENSION_CODE_SHA256,
                "size_bytes": 12_141,
            }
            or manifest != {
                "repository_path": BLIND_PRIVATE_SOURCE_EXTENSION_V4_PATH,
                "sha256":
                    W02_MORPH_V3_PRIVATE_R3_SOURCE_EXTENSION_MANIFEST_SHA256,
                "size_bytes": 4_826,
            }):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 owner public source binding drifted")

    namespace = require_exact_dict(raw["namespace_policy"], {
        "artifact_key", "dataset_key", "evaluator_owner_key",
        "namespace_components", "policy", "record_kind_components",
    }, where="R3 namespace policy")
    prefix = [3, 15_937_461_156_557_176_020]
    if (namespace["namespace_components"] != prefix
            or namespace["dataset_key"] != [*prefix, 1]
            or namespace["artifact_key"] != [*prefix, 2]
            or namespace["evaluator_owner_key"] != [*prefix, 90, 1]
            or namespace["policy"]
            != "FRESH_R3_OWNER_PREFIX_WITH_DISJOINT_KIND_COMPONENTS"
            or namespace["record_kind_components"] != {
                "evaluator_label": 40, "observation": 20, "source_ref": 10}):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 namespace policy drifted")

    _identity(raw["owner_audit_identity"], where="R3 owner audit",
              relative="owner-audit-report.json")
    _identity(raw["owner_seal_identity"], where="R3 owner seal",
              relative="owner-seal.json")
    ranges = require_exact_dict(raw["record_key_ranges"], {
        "evaluator_label", "observation", "source_ref",
    }, where="R3 record key ranges")
    by_kind = {
        "source_ref": tuple(row for row in files if row.record_kind == "source_ref"),
        "observation": tuple(row for row in files if row.record_kind == "observation"),
        "evaluator_label": tuple(
            row for row in files if row.record_kind == "evaluator_label"),
    }
    for kind, rows in by_kind.items():
        item = require_exact_dict(
            ranges[kind], {"first_key", "last_key"},
            where=f"R3 {kind} key range")
        if (item["first_key"] != list(min(row.first_record_key for row in rows))
                or item["last_key"]
                != list(max(row.last_record_key for row in rows))):
            raise W02MorphologySuccessorV3PrivateOwnerR3Error(
                "R3 record key range drifted")

    repository = Path(repository_root).resolve()
    extension = read_blind_private_source_extension_v4_manifest(repository)
    extension_path = _repository_file(
        repository, BLIND_PRIVATE_SOURCE_EXTENSION_V4_PATH)
    source = extension["sources"][0]
    if (hashlib.sha256(extension_path.read_bytes()).hexdigest()
            != W02_MORPH_V3_PRIVATE_R3_SOURCE_EXTENSION_MANIFEST_SHA256
            or extension.get("parent_extension_v3_manifest_sha256")
            != PARENT_EXTENSION_V3_MANIFEST_SHA256
            or source.get("source_key") != W02_MORPH_V3_PRIVATE_R3_SOURCE_KEY
            or source.get("license_id") != "CC-BY-SA-3.0"):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 source extension live binding drifted")
    return raw, files


def read_w02_morphology_successor_v3_private_owner_r3_receipt(
        repository_root: str | Path,
        ) -> tuple[
            dict[str, Any],
            tuple[W02MorphologySuccessorV3PrivateFileIdentity, ...],
        ]:
    """Read the canonical public R3 receipt without a private path."""
    repository = Path(repository_root).resolve()
    target = _repository_file(
        repository, W02_MORPH_V3_PRIVATE_R3_OWNER_RECEIPT_PATH)
    value = read_canonical_object(target)
    return validate_w02_morphology_successor_v3_private_owner_r3_receipt(
        value, repository)


__all__ = [
    "W02_MORPH_V3_PRIVATE_LAYOUTS",
    "W02_MORPH_V3_PRIVATE_PATHS",
    "W02_MORPH_V3_PRIVATE_R3_METADATA_SHA256",
    "W02_MORPH_V3_PRIVATE_R3_OWNER_FAMILY_KEY",
    "W02_MORPH_V3_PRIVATE_R3_OWNER_RECEIPT_PATH",
    "W02_MORPH_V3_PRIVATE_R3_SOURCE_KEY",
    "W02MorphologySuccessorV3PrivateFileIdentity",
    "W02MorphologySuccessorV3PrivateOwnerR3Error",
    "read_w02_morphology_successor_v3_private_owner_r3_receipt",
    "validate_w02_morphology_successor_v3_private_owner_r3_receipt",
]
