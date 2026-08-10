"""Metadata-only audits for the successor V3 R4 Kyoto owner."""
from __future__ import annotations

from typing import Any

from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v5 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V5_VERSION,
    KYOTO_COMMIT_SHA1,
    KYOTO_DATA_BLOB_SHA1,
    KYOTO_DATA_SIZE_BYTES,
    KYOTO_SOURCE_KEY,
    KYOTO_TEST_SENTENCE_COUNT,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r4_contract import (
    W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS,
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02_MORPH_V3_PRIVATE_PAIR_COUNT,
    W02_MORPH_V3_PRIVATE_PATHS,
    W02_MORPH_V3_PRIVATE_R4_DIMENSION_BINDINGS,
    W02_MORPH_V3_PRIVATE_R4_DOUBLE_PASS_SHA256,
    W02_MORPH_V3_PRIVATE_R4_LABEL_BINDING_SHA256,
    W02_MORPH_V3_PRIVATE_R4_METADATA_SHA256,
    W02_MORPH_V3_PRIVATE_R4_METADATA_SIZE_BYTES,
    W02_MORPH_V3_PRIVATE_R4_OWNER_FAMILY_KEY,
    W02_MORPH_V3_PRIVATE_R4_OWNER_ID,
    W02_MORPH_V3_PRIVATE_R4_OWNER_RECEIPT_VERSION,
    W02_MORPH_V3_PRIVATE_R4_PUBLIC_BASE_COMMIT,
    W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_CODE_SHA256,
    W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_MANIFEST_SHA256,
    W02_MORPH_V3_PRIVATE_R4_SOURCE_KEY,
    W02_MORPH_V3_PRIVATE_R4_SOURCE_SNAPSHOT_COMMITMENT,
    W02_MORPH_V3_PRIVATE_SOURCE_COUNT,
    W02_MORPH_V3_PRIVATE_SPLIT_COUNTS,
    W02MorphologySuccessorV3PrivateOwnerR4Error,
    require_exact_dict,
    require_positive,
    require_sha256,
    validate_r4_private_file_inventory,
)


_LAYOUT_BY_PATH = {
    relative: layout for layout, relative in W02_MORPH_V3_PRIVATE_PATHS.items()
}


def _identity(value: object, *, where: str, relative: str) -> dict[str, Any]:
    raw = require_exact_dict(
        value, {"relative_path", "sha256", "size_bytes"}, where=where)
    if raw["relative_path"] != relative:
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            f"{where} relative path drifted")
    require_sha256(raw["sha256"], where=f"{where} SHA")
    require_positive(raw["size_bytes"], where=f"{where} size")
    return raw


def _validate_source_snapshot(value: object) -> dict[str, Any]:
    raw = require_exact_dict(value, {
        "commit_sha1", "data_blob_sha1", "data_size_bytes", "language",
        "license_id", "parallel", "repository", "script", "sentence_count",
        "snapshot_id", "source_key", "source_snapshot_commitment", "split",
        "tag",
    }, where="R4 source snapshot")
    if raw != {
            "commit_sha1": KYOTO_COMMIT_SHA1,
            "data_blob_sha1": KYOTO_DATA_BLOB_SHA1,
            "data_size_bytes": KYOTO_DATA_SIZE_BYTES,
            "language": "lzh",
            "license_id": "CC-BY-SA-4.0",
            "parallel": 0,
            "repository": (
                "https://github.com/UniversalDependencies/"
                "UD_Classical_Chinese-Kyoto"),
            "script": "Hant",
            "sentence_count": KYOTO_TEST_SENTENCE_COUNT,
            "snapshot_id": "ud-lzh-kyoto-r2.18-test",
            "source_key": KYOTO_SOURCE_KEY,
            "source_snapshot_commitment":
                W02_MORPH_V3_PRIVATE_R4_SOURCE_SNAPSHOT_COMMITMENT,
            "split": "test",
            "tag": "r2.18",
            }:
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 source snapshot identity drifted")
    return raw


def _validate_source_binding(value: object) -> dict[str, Any]:
    raw = require_exact_dict(value, {
        "consumed_source_key_count", "language_counts", "license_counts",
        "manifest_sha256", "source_key", "source_record_count", "status",
        "unknown_source_key_count", "validation_failure_count",
        "validation_pass_count", "validator_code_sha256", "validator_version",
    }, where="R4 V5 source binding")
    if (raw["status"] != "PASS"
            or raw["validator_version"]
            != BLIND_PRIVATE_SOURCE_EXTENSION_V5_VERSION
            or raw["validator_code_sha256"]
            != W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_CODE_SHA256
            or raw["manifest_sha256"]
            != W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_MANIFEST_SHA256
            or raw["source_key"] != W02_MORPH_V3_PRIVATE_R4_SOURCE_KEY
            or raw["source_record_count"] != W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            or raw["validation_pass_count"] != W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            or raw["language_counts"] != {"lzh": 500}
            or raw["license_counts"] != {"CC-BY-SA-4.0": 500}
            or any(raw[name] != 0 for name in (
                "consumed_source_key_count", "unknown_source_key_count",
                "validation_failure_count"))):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 V5 source binding did not close")
    return raw


def _validate_label_binding(value: object) -> dict[str, Any]:
    raw = require_exact_dict(value, {
        "binding_sha256", "dimensions", "double_pass_equal",
        "expected_true_count", "observation_mismatch_count",
        "owner_mode_mismatch_count", "second_pass_binding_sha256",
        "stage_mismatch_count", "status", "unknown_dimension_key_count",
        "unknown_expected_state_count",
    }, where="R4 label semantic binding")
    expected = [
        {"count": count, "dimension_key": list(key), "dimension_name": name}
        for name, key, count in W02_MORPH_V3_PRIVATE_R4_DIMENSION_BINDINGS
    ]
    if (raw["status"] != "PASS"
            or raw["dimensions"] != expected
            or raw["binding_sha256"]
            != W02_MORPH_V3_PRIVATE_R4_LABEL_BINDING_SHA256
            or raw["second_pass_binding_sha256"] != raw["binding_sha256"]
            or raw["double_pass_equal"] != 1
            or raw["expected_true_count"] != W02_MORPH_V3_PRIVATE_PAIR_COUNT
            or any(raw[name] != 0 for name in (
                "observation_mismatch_count", "owner_mode_mismatch_count",
                "stage_mismatch_count", "unknown_dimension_key_count",
                "unknown_expected_state_count"))):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 label semantic binding did not close")
    return raw


def _receipt_files(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != 7:
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 safe metadata file inventory drifted")
    rows = []
    for item in value:
        raw = require_exact_dict(item, {
            "content_sha256", "content_size_bytes", "first_record_key", "kind",
            "last_record_key", "license_ids", "record_count", "relative_path",
            "split", "transport_sha256", "transport_size_bytes",
        }, where="R4 safe metadata file")
        relative = raw["relative_path"]
        layout = _LAYOUT_BY_PATH.get(relative)
        if layout is None:
            raise W02MorphologySuccessorV3PrivateOwnerR4Error(
                "R4 safe metadata relative path drifted")
        rows.append({
            "content_sha256": raw["content_sha256"],
            "content_size_bytes": raw["content_size_bytes"],
            "first_record_key": raw["first_record_key"],
            "last_record_key": raw["last_record_key"],
            "layout_key": layout,
            "license_ids": raw["license_ids"],
            "record_count": raw["record_count"],
            "record_kind": raw["kind"],
            "relative_path": relative,
            "root_key": "PRIVATE_EVALUATOR_ROOT",
            "split": raw["split"] or "",
            "transport_sha256": raw["transport_sha256"],
            "transport_size_bytes": raw["transport_size_bytes"],
        })
    validate_r4_private_file_inventory(rows)
    return rows


def build_w02_morphology_successor_v3_private_owner_r4_receipt_from_metadata(
        value: object) -> dict[str, object]:
    """Validate the safe R4 metadata and project only payload-free fields."""
    raw = require_exact_dict(value, {
        "artifact_kind", "artifact_version", "audit_identity", "commitments",
        "contamination_audit", "dimension_denominators",
        "domain_disjoint_audit", "double_pass_audit", "duplicate_audit",
        "file_count", "files", "label_semantic_binding_audit", "next_action",
        "owner_family_key", "owner_id", "pair_count", "public_commit",
        "resource_limits", "resource_usage", "seal_identity", "source_count",
        "source_snapshot", "source_validator_binding_audit", "split_counts",
        "status", "v5_identity", "zero_call_audit",
    }, where="R4 safe owner metadata")
    if (raw["artifact_kind"]
            != "PH2_D03_V2_W02_SUCCESSOR_V3_R4_BLIND_PRIVATE_OWNER_METADATA"
            or raw["artifact_version"]
            != "PH2-D03-V2-W02-SUCCESSOR-V3-R4-OWNER-METADATA-V1"
            or raw["status"]
            != "OWNER_METADATA_FROZEN_SOURCE_V5_LABEL_BINDING_VERIFIED"
            or raw["next_action"]
            != "BUILD_SUCCESSOR_V3_PRIVATE_R4_OWNER_RECEIPT_IO_AND_EVALUATOR_REVISION"
            or raw["owner_id"] != W02_MORPH_V3_PRIVATE_R4_OWNER_ID
            or raw["owner_family_key"]
            != W02_MORPH_V3_PRIVATE_R4_OWNER_FAMILY_KEY
            or raw["public_commit"] != W02_MORPH_V3_PRIVATE_R4_PUBLIC_BASE_COMMIT
            or raw["source_count"] != W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            or raw["pair_count"] != W02_MORPH_V3_PRIVATE_PAIR_COUNT
            or raw["file_count"] != len(W02_MORPH_V3_PRIVATE_LAYOUTS)
            or raw["split_counts"] != W02_MORPH_V3_PRIVATE_SPLIT_COUNTS
            or raw["dimension_denominators"]
            != W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 safe owner identity drifted")

    v5 = require_exact_dict(raw["v5_identity"], {
        "code_sha256", "manifest_sha256", "validator_version",
    }, where="R4 V5 identity")
    if v5 != {
            "code_sha256": W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_CODE_SHA256,
            "manifest_sha256":
                W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_MANIFEST_SHA256,
            "validator_version": BLIND_PRIVATE_SOURCE_EXTENSION_V5_VERSION,
            }:
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 V5 identity drifted")
    source = _validate_source_snapshot(raw["source_snapshot"])
    source_binding = _validate_source_binding(
        raw["source_validator_binding_audit"])
    label_binding = _validate_label_binding(
        raw["label_semantic_binding_audit"])
    files = _receipt_files(raw["files"])

    commitments = require_exact_dict(raw["commitments"], {
        "case_commitment", "cluster_commitment", "label_commitment",
        "payload_commitment",
    }, where="R4 commitments")
    for name, digest in commitments.items():
        require_sha256(digest, where=f"R4 {name}")
    contamination = require_exact_dict(raw["contamination_audit"], {
        "exact_case_overlap", "exact_cluster_overlap", "exact_content_overlap",
        "normalized_content_overlap",
    }, where="R4 contamination audit")
    if any(value != 0 for value in contamination.values()):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 contamination audit did not close")
    duplicate = require_exact_dict(raw["duplicate_audit"], {
        "cluster_collision_count", "exact_duplicate_count",
        "label_key_duplicate_count", "near_duplicate_algorithm",
        "near_duplicate_count", "near_duplicate_pair_comparisons",
        "normalized_duplicate_count", "observation_key_duplicate_count",
    }, where="R4 duplicate audit")
    if (duplicate["near_duplicate_algorithm"]
            != "NFKC_CASEFOLD_DROP_UNICODE_Z_P_CC_CF_LENGTH_RATIO_0_8_CHARACTER_TRIGRAM_SET_JACCARD_0_82"
            or duplicate["near_duplicate_pair_comparisons"] != 44_826
            or any(duplicate[name] != 0 for name in (
                "cluster_collision_count", "exact_duplicate_count",
                "label_key_duplicate_count", "near_duplicate_count",
                "normalized_duplicate_count", "observation_key_duplicate_count"))):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 duplicate audit did not close")
    domain = require_exact_dict(raw["domain_disjoint_audit"], {
        "consumed_source_key_intersection_count", "family_identity_intersection_count",
        "namespace_prefix_intersection_count", "new_commitment_intersection_count",
        "old_private_payload_reads", "old_r2_formal_family_reads",
        "old_r2_owner_payload_reads", "old_r3_formal_family_reads",
        "old_r3_owner_payload_reads", "parallel",
        "safe_public_freeze_receipt_files_read",
        "source_repository_tag_blob_distinct_from_consumed",
    }, where="R4 domain disjoint audit")
    if (domain["safe_public_freeze_receipt_files_read"] != 8
            or domain["source_repository_tag_blob_distinct_from_consumed"] != 1
            or any(domain[name] != 0 for name in (
                "consumed_source_key_intersection_count",
                "family_identity_intersection_count",
                "namespace_prefix_intersection_count",
                "new_commitment_intersection_count", "old_private_payload_reads",
                "old_r2_formal_family_reads", "old_r2_owner_payload_reads",
                "old_r3_formal_family_reads", "old_r3_owner_payload_reads",
                "parallel"))):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 domain disjoint audit did not close")
    double_pass = require_exact_dict(raw["double_pass_audit"], {
        "equal", "first_pass_sha256", "second_pass_sha256",
    }, where="R4 double-pass audit")
    if (double_pass != {
            "equal": 1,
            "first_pass_sha256": W02_MORPH_V3_PRIVATE_R4_DOUBLE_PASS_SHA256,
            "second_pass_sha256": W02_MORPH_V3_PRIVATE_R4_DOUBLE_PASS_SHA256,
            }):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 double-pass audit drifted")
    limits = require_exact_dict(raw["resource_limits"], {
        "logic_operations", "payload_bytes", "payload_gets", "records",
    }, where="R4 resource limits")
    if limits != {
            "logic_operations": 9_000_000,
            "payload_bytes": 536_870_912,
            "payload_gets": 300_000,
            "records": 100_000,
            }:
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 resource limits drifted")
    usage = require_exact_dict(raw["resource_usage"], {
        "logic_operations", "payload_bytes", "payload_gets", "records",
    }, where="R4 resource usage")
    if (usage["records"] != 1_500
            or any(type(usage[name]) is not int or usage[name] <= 0
                   or usage[name] > limits[name] for name in limits)):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 resource usage drifted")
    zero_call = require_exact_dict(raw["zero_call_audit"], {
        "candidate_calls", "evaluator_calls", "formal_runs", "llm_calls",
        "main_session_content_reads", "old_private_reads", "teacher_calls",
        "v1_predictor_calls", "v2_predictor_calls", "v3_predictor_calls",
    }, where="R4 zero-call audit")
    if any(value != 0 for value in zero_call.values()):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 zero-call audit drifted")
    audit = _identity(
        raw["audit_identity"], where="R4 owner audit",
        relative="owner-audit-report.json")
    seal = _identity(
        raw["seal_identity"], where="R4 owner seal",
        relative="owner-seal.json")
    namespace_prefix = list(files[0]["first_record_key"][:2])
    if (namespace_prefix != [4, 1_799_101_329_316_159_300]
            or any(row["first_record_key"][:2] != namespace_prefix
                   or row["last_record_key"][:2] != namespace_prefix
                   for row in files)):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 namespace prefix drifted")

    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_OWNER_R4_RECEIPT"),
        "artifact_version": W02_MORPH_V3_PRIVATE_R4_OWNER_RECEIPT_VERSION,
        "candidate_evaluation_runs": 0,
        "commitments": dict(commitments),
        "contamination_audit": dict(contamination),
        "dimension_denominator_counts": dict(raw["dimension_denominators"]),
        "domain_disjoint_audit": dict(domain),
        "double_pass_audit": dict(double_pass),
        "duplicate_audit": dict(duplicate),
        "file_count": len(files),
        "files": files,
        "formal_private_evaluation_runs": 0,
        "label_record_count": W02_MORPH_V3_PRIVATE_PAIR_COUNT,
        "label_semantic_binding_audit": dict(label_binding),
        "main_session_private_payload_reads": 0,
        "main_session_source_payload_reads": 0,
        "namespace_policy": {
            "namespace_components": namespace_prefix,
            "policy": "FRESH_R4_OWNER_PREFIX_WITH_DISJOINT_KIND_COMPONENTS",
            "record_kind_components": {
                "evaluator_label": 40, "observation": 20, "source_ref": 10},
        },
        "next_action": "BUILD_SUCCESSOR_V3_PRIVATE_R4_EVALUATOR_FAMILY_REVISION",
        "owner_audit_identity": dict(audit),
        "owner_family_key": W02_MORPH_V3_PRIVATE_R4_OWNER_FAMILY_KEY,
        "owner_id": W02_MORPH_V3_PRIVATE_R4_OWNER_ID,
        "owner_metadata_sha256": W02_MORPH_V3_PRIVATE_R4_METADATA_SHA256,
        "owner_metadata_size_bytes": W02_MORPH_V3_PRIVATE_R4_METADATA_SIZE_BYTES,
        "owner_seal_identity": dict(seal),
        "pair_count": W02_MORPH_V3_PRIVATE_PAIR_COUNT,
        "public_identity": {
            "public_repository_commit": W02_MORPH_V3_PRIVATE_R4_PUBLIC_BASE_COMMIT,
            "source_extension_v5_code_sha256":
                W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_CODE_SHA256,
            "source_extension_v5_manifest_sha256":
                W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_MANIFEST_SHA256,
            "source_extension_v5_status":
                "BLIND_PRIVATE_SOURCE_EXTENSION_V5_APPROVED",
        },
        "resource_limits": dict(limits),
        "resource_usage": dict(usage),
        "source_count": W02_MORPH_V3_PRIVATE_SOURCE_COUNT,
        "source_key": W02_MORPH_V3_PRIVATE_R4_SOURCE_KEY,
        "source_snapshot": dict(source),
        "source_snapshot_commitment":
            W02_MORPH_V3_PRIVATE_R4_SOURCE_SNAPSHOT_COMMITMENT,
        "source_validator_binding_audit": dict(source_binding),
        "split_counts": dict(raw["split_counts"]),
        "status": "OWNER_METADATA_INGESTED_SOURCE_V5_PAYLOAD_UNREAD",
        "teacher_llm_provenance": {"llm_calls": 0, "teacher_calls": 0},
        "zero_call_audit": dict(zero_call),
    }


__all__ = [
    "build_w02_morphology_successor_v3_private_owner_r4_receipt_from_metadata",
]
