"""successor V3 R5 owner receipt 的公开 facade。"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v6 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V6_PATH,
    read_blind_private_source_extension_v6_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r5_audit import (
    build_w02_morphology_successor_v3_private_owner_r5_receipt_from_metadata,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r5_contract import (
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02_MORPH_V3_PRIVATE_PATHS,
    W02_MORPH_V3_PRIVATE_R5_ADAPTER_CODE_SHA256,
    W02_MORPH_V3_PRIVATE_R5_METADATA_SHA256,
    W02_MORPH_V3_PRIVATE_R5_METADATA_SIZE_BYTES,
    W02_MORPH_V3_PRIVATE_R5_OWNER_FAMILY_KEY,
    W02_MORPH_V3_PRIVATE_R5_OWNER_ID,
    W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_PATH,
    W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_SHA256,
    W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_SIZE_BYTES,
    W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_VERSION,
    W02_MORPH_V3_PRIVATE_R5_PROBE_CODE_SHA256,
    W02_MORPH_V3_PRIVATE_R5_PROBE_REPORT_SHA256,
    W02_MORPH_V3_PRIVATE_R5_PUBLIC_BASE_COMMIT,
    W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_CODE_SHA256,
    W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_MANIFEST_SHA256,
    W02_MORPH_V3_PRIVATE_R5_SOURCE_KEY,
    W02MorphologySuccessorV3PrivateOwnerR5Error,
    W02MorphologySuccessorV3PrivateR5FileIdentity,
    require_exact_dict,
    validate_r5_private_file_inventory,
)


_V6_CODE_PATH = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_blind_private_source_extension_v6.py"
)
_ADAPTER_CODE_PATH = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_base_language_family_adapter.py"
)
_PROBE_CODE_PATH = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_base_language_family_probe.py"
)
_PROBE_REPORT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_base_language_family_probe_report_v1.json"
)


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (repository / Path(*pure.parts)).resolve()
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative or ".." in pure.parts
            or not target.is_relative_to(repository) or target.is_symlink()
            or not target.is_file()):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 owner receipt repository path is invalid")
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
    audit = dict(raw["owner_audit_identity"])
    audit.pop("relative_path")
    seal = dict(raw["owner_seal_identity"])
    seal.pop("relative_path")
    return {
        "adapter_probe_identity": raw["adapter_probe_identity"],
        "artifact_kind": (
            "PH2_D03_V2_W02_SUCCESSOR_V3_BLIND_PRIVATE_OWNER_R5_METADATA"),
        "artifact_version": (
            "PH2-D03-V2-W02-SUCCESSOR-V3-BLIND-PRIVATE-OWNER-R5-V1"),
        "audit_identity": audit,
        "commitments": raw["commitments"],
        "contamination_audit": raw["contamination_audit"],
        "dimension_denominators": raw["dimension_denominator_counts"],
        "dimension_split_counts": raw["dimension_split_counts"],
        "domain_disjoint_audit": raw["domain_disjoint_audit"],
        "double_pass_audit": raw["double_pass_audit"],
        "duplicate_audit": raw["duplicate_audit"],
        "files": files,
        "label_semantic_binding_audit": raw["label_semantic_binding_audit"],
        "namespace_audit": raw["namespace_audit"],
        "next_action": (
            "BUILD_SUCCESSOR_V3_PRIVATE_R5_OWNER_RECEIPT_IO_AND_EVALUATOR_REVISION"),
        "ordinal_slice_identity": raw["ordinal_slice_identity"],
        "owner_family_key": raw["owner_family_key"],
        "owner_id": raw["owner_id"],
        "pair_count": raw["pair_count"],
        "public_commit": raw["public_identity"]["public_repository_commit"],
        "public_git_write_count": 0,
        "resource_limits": raw["resource_limits"],
        "resource_usage": raw["resource_usage"],
        "seal_identity": seal,
        "source_count": raw["source_count"],
        "source_snapshot_identity": raw["source_snapshot_identity"],
        "source_validator_binding_audit": raw["source_validator_binding_audit"],
        "split_counts": raw["split_counts"],
        "status": "OWNER_METADATA_FROZEN_SOURCE_V6_LABEL_BINDING_VERIFIED",
        "v6_identity": raw["v6_identity"],
        "zero_call_audit": raw["zero_call_audit"],
    }


def validate_w02_morphology_successor_v3_private_owner_r5_receipt(
        value: object, repository_root: str | Path,
        ) -> tuple[
            dict[str, Any],
            tuple[W02MorphologySuccessorV3PrivateR5FileIdentity, ...],
        ]:
    """不接收私有 root，只验证 R5 安全 receipt。"""
    raw = require_exact_dict(value, {
        "adapter_probe_identity", "artifact_kind", "artifact_version",
        "candidate_evaluation_runs", "commitments", "contamination_audit",
        "dimension_denominator_counts", "dimension_split_counts",
        "domain_disjoint_audit", "double_pass_audit", "duplicate_audit",
        "file_count", "files", "formal_private_evaluation_runs",
        "label_record_count", "label_semantic_binding_audit",
        "main_session_private_payload_reads", "main_session_source_payload_reads",
        "namespace_audit", "next_action", "ordinal_slice_identity",
        "owner_audit_identity", "owner_family_key", "owner_id",
        "owner_metadata_sha256", "owner_metadata_size_bytes",
        "owner_seal_identity", "pair_count", "public_identity",
        "resource_limits", "resource_usage", "source_count", "source_key",
        "source_snapshot_identity", "source_validator_binding_audit",
        "split_counts", "status", "teacher_llm_provenance", "v6_identity",
        "zero_call_audit",
    }, where="R5 owner receipt")
    payload = canonical_json_bytes(raw) + b"\n"
    if (len(payload) != W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_SIZE_BYTES
            or hashlib.sha256(payload).hexdigest()
            != W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_SHA256):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 owner receipt byte identity drifted")
    if (raw["artifact_kind"]
            != "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_OWNER_R5_RECEIPT"
            or raw["artifact_version"]
            != W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_VERSION
            or raw["status"]
            != "OWNER_METADATA_INGESTED_SOURCE_V6_PAYLOAD_UNREAD"
            or raw["next_action"]
            != "BUILD_SUCCESSOR_V3_PRIVATE_R5_EVALUATOR_FAMILY_REVISION"
            or raw["owner_id"] != W02_MORPH_V3_PRIVATE_R5_OWNER_ID
            or raw["owner_family_key"]
            != W02_MORPH_V3_PRIVATE_R5_OWNER_FAMILY_KEY
            or raw["owner_metadata_sha256"]
            != W02_MORPH_V3_PRIVATE_R5_METADATA_SHA256
            or raw["owner_metadata_size_bytes"]
            != W02_MORPH_V3_PRIVATE_R5_METADATA_SIZE_BYTES
            or raw["source_key"] != W02_MORPH_V3_PRIVATE_R5_SOURCE_KEY
            or any(raw[name] != 0 for name in (
                "candidate_evaluation_runs", "formal_private_evaluation_runs",
                "main_session_private_payload_reads",
                "main_session_source_payload_reads"))
            or raw["teacher_llm_provenance"]
            != {"llm_calls": 0, "teacher_calls": 0}):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 owner receipt identity or zero-call state drifted")
    files = validate_r5_private_file_inventory(raw["files"])
    rebuilt = (
        build_w02_morphology_successor_v3_private_owner_r5_receipt_from_metadata(
            _metadata_projection(raw)))
    if rebuilt != raw:
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 owner receipt no longer matches its safe metadata projection")

    repository = Path(repository_root).resolve()
    extension = read_blind_private_source_extension_v6_manifest(repository)
    dependencies = {
        _V6_CODE_PATH: (17_890, W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_CODE_SHA256),
        BLIND_PRIVATE_SOURCE_EXTENSION_V6_PATH: (
            8_417, W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_MANIFEST_SHA256),
        _ADAPTER_CODE_PATH: (2_492, W02_MORPH_V3_PRIVATE_R5_ADAPTER_CODE_SHA256),
        _PROBE_CODE_PATH: (9_058, W02_MORPH_V3_PRIVATE_R5_PROBE_CODE_SHA256),
        _PROBE_REPORT_PATH: (1_471, W02_MORPH_V3_PRIVATE_R5_PROBE_REPORT_SHA256),
    }
    for relative, expected in dependencies.items():
        if _sha256_file(_repository_file(repository, relative)) != expected:
            raise W02MorphologySuccessorV3PrivateOwnerR5Error(
                "R5 owner public dependency drifted")
    public = raw["public_identity"]
    if (extension.get("status") != "BLIND_PRIVATE_SOURCE_EXTENSION_V6_APPROVED"
            or extension.get("sources", [{}])[0].get("source_key")
            != W02_MORPH_V3_PRIVATE_R5_SOURCE_KEY
            or extension.get("sources", [{}])[0].get("license_id")
            != "CC-BY-SA-4.0"
            or public != {
                "adapter_code_sha256":
                    W02_MORPH_V3_PRIVATE_R5_ADAPTER_CODE_SHA256,
                "probe_code_sha256": W02_MORPH_V3_PRIVATE_R5_PROBE_CODE_SHA256,
                "probe_report_sha256":
                    W02_MORPH_V3_PRIVATE_R5_PROBE_REPORT_SHA256,
                "public_repository_commit":
                    W02_MORPH_V3_PRIVATE_R5_PUBLIC_BASE_COMMIT,
                "source_extension_v6_code_sha256":
                    W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_CODE_SHA256,
                "source_extension_v6_manifest_sha256":
                    W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_MANIFEST_SHA256,
                "source_extension_v6_status":
                    "BLIND_PRIVATE_SOURCE_EXTENSION_V6_APPROVED",
            }):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 owner live V6/adapter binding drifted")
    return raw, files


def read_w02_morphology_successor_v3_private_owner_r5_receipt(
        repository_root: str | Path,
        ) -> tuple[
            dict[str, Any],
            tuple[W02MorphologySuccessorV3PrivateR5FileIdentity, ...],
        ]:
    repository = Path(repository_root).resolve()
    target = _repository_file(
        repository, W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_PATH)
    value = read_canonical_object(target)
    return validate_w02_morphology_successor_v3_private_owner_r5_receipt(
        value, repository)


__all__ = [
    "W02_MORPH_V3_PRIVATE_LAYOUTS",
    "W02_MORPH_V3_PRIVATE_PATHS",
    "W02_MORPH_V3_PRIVATE_R5_METADATA_SHA256",
    "W02_MORPH_V3_PRIVATE_R5_OWNER_FAMILY_KEY",
    "W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_PATH",
    "W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_SHA256",
    "W02MorphologySuccessorV3PrivateOwnerR5Error",
    "W02MorphologySuccessorV3PrivateR5FileIdentity",
    "read_w02_morphology_successor_v3_private_owner_r5_receipt",
    "validate_w02_morphology_successor_v3_private_owner_r5_receipt",
]
