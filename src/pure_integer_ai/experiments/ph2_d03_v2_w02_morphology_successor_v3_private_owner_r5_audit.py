"""successor V3 R5 Kyoto remainder owner 的纯 metadata 审计。"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v5 import (
    KYOTO_COMMIT_SHA1,
    KYOTO_DATA_BLOB_SHA1,
    KYOTO_DATA_SIZE_BYTES,
    KYOTO_LICENSE_BLOB_SHA1,
    KYOTO_LICENSE_SHA256,
    KYOTO_README_BLOB_SHA1,
    KYOTO_README_SHA256,
    KYOTO_TEST_SENTENCE_COUNT,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v6 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V6_VERSION,
    KYOTO_REMAINDER_MINIMUM_ORDINAL,
    KYOTO_REMAINDER_SOURCE_KEY,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r5_contract import (
    W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS,
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02_MORPH_V3_PRIVATE_PAIR_COUNT,
    W02_MORPH_V3_PRIVATE_PATHS,
    W02_MORPH_V3_PRIVATE_R5_ADAPTER_CODE_SHA256,
    W02_MORPH_V3_PRIVATE_R5_DIMENSION_BINDINGS,
    W02_MORPH_V3_PRIVATE_R5_LABEL_BINDING_SHA256,
    W02_MORPH_V3_PRIVATE_R5_METADATA_SHA256,
    W02_MORPH_V3_PRIVATE_R5_METADATA_SIZE_BYTES,
    W02_MORPH_V3_PRIVATE_R5_ORDINAL_SLICE_COMMITMENT,
    W02_MORPH_V3_PRIVATE_R5_OWNER_FAMILY_KEY,
    W02_MORPH_V3_PRIVATE_R5_OWNER_ID,
    W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_VERSION,
    W02_MORPH_V3_PRIVATE_R5_PROBE_CODE_SHA256,
    W02_MORPH_V3_PRIVATE_R5_PROBE_REPORT_SHA256,
    W02_MORPH_V3_PRIVATE_R5_PUBLIC_BASE_COMMIT,
    W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_CODE_SHA256,
    W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_MANIFEST_SHA256,
    W02_MORPH_V3_PRIVATE_R5_SOURCE_KEY,
    W02_MORPH_V3_PRIVATE_SOURCE_COUNT,
    W02_MORPH_V3_PRIVATE_SPLIT_COUNTS,
    W02MorphologySuccessorV3PrivateOwnerR5Error,
    require_exact_dict,
    require_positive,
    require_sha256,
    validate_r5_private_file_inventory,
)


_LAYOUT_BY_PATH = {
    relative: layout for layout, relative in W02_MORPH_V3_PRIVATE_PATHS.items()
}
_NAMESPACE_PREFIX = (5, 2_007_340_265, 2_168_420_232, 1_803_141_975,
                     3_910_540_986)


def _identity(
        value: object, *, where: str, relative: str) -> dict[str, Any]:
    raw = require_exact_dict(value, {"sha256", "size_bytes"}, where=where)
    require_sha256(raw["sha256"], where=f"{where} SHA")
    require_positive(raw["size_bytes"], where=f"{where} size")
    return {
        "relative_path": relative,
        "sha256": raw["sha256"],
        "size_bytes": raw["size_bytes"],
    }


def _validate_adapter_probe(value: object) -> dict[str, Any]:
    raw = require_exact_dict(value, {
        "adapter_code_sha256", "probe_code_sha256", "probe_report_sha256",
        "status",
    }, where="R5 adapter probe identity")
    if raw != {
            "adapter_code_sha256": W02_MORPH_V3_PRIVATE_R5_ADAPTER_CODE_SHA256,
            "probe_code_sha256": W02_MORPH_V3_PRIVATE_R5_PROBE_CODE_SHA256,
            "probe_report_sha256": W02_MORPH_V3_PRIVATE_R5_PROBE_REPORT_SHA256,
            "status": "PASS",
            }:
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 adapter probe identity drifted")
    return raw


def _validate_source_snapshot(value: object) -> dict[str, Any]:
    raw = require_exact_dict(value, {
        "commit_sha1", "data_git_blob_sha1", "data_sha256",
        "data_size_bytes", "license_git_blob_sha1", "license_id",
        "license_sha256", "parallel", "readme_git_blob_sha1",
        "readme_sha256", "repository", "sentence_count",
        "snapshot_file_sha256", "snapshot_file_size_bytes", "snapshot_id",
        "source_key", "split", "tag",
    }, where="R5 source snapshot")
    expected = {
        "commit_sha1": KYOTO_COMMIT_SHA1,
        "data_git_blob_sha1": KYOTO_DATA_BLOB_SHA1,
        "data_sha256": (
            "e492ba5f5054ee560c33197e1681a5c18c3f21adff7dca82be3ed4af09cbf1e5"),
        "data_size_bytes": KYOTO_DATA_SIZE_BYTES,
        "license_git_blob_sha1": KYOTO_LICENSE_BLOB_SHA1,
        "license_id": "CC-BY-SA-4.0",
        "license_sha256": KYOTO_LICENSE_SHA256,
        "parallel": 0,
        "readme_git_blob_sha1": KYOTO_README_BLOB_SHA1,
        "readme_sha256": KYOTO_README_SHA256,
        "repository": (
            "https://github.com/UniversalDependencies/"
            "UD_Classical_Chinese-Kyoto"),
        "sentence_count": KYOTO_TEST_SENTENCE_COUNT,
        "snapshot_file_sha256": (
            "c30f727bdbff4c70c1ad8316e98d6d4de5e530aa70708bf0d0df11e482465900"),
        "snapshot_file_size_bytes": 823,
        "snapshot_id": "ud-lzh-kyoto-r2.18-test-remainder-r5",
        "source_key": KYOTO_REMAINDER_SOURCE_KEY,
        "split": "test",
        "tag": "r2.18",
    }
    if raw != expected:
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 source snapshot identity drifted")
    return raw


def _validate_source_binding(value: object) -> dict[str, Any]:
    raw = require_exact_dict(value, {
        "consumed_source_key_count", "exact_source_identity_match_count",
        "language_counts", "license_counts", "manifest_sha256",
        "maximum_sentence_ordinal", "minimum_sentence_ordinal",
        "selected_below_minimum_count", "source_key", "source_record_count",
        "status", "unknown_source_key_count", "validation_failure_count",
        "validation_pass_count", "validator_code_sha256", "validator_version",
    }, where="R5 V6 source binding")
    if (raw["status"] != "PASS"
            or raw["validator_version"]
            != BLIND_PRIVATE_SOURCE_EXTENSION_V6_VERSION
            or raw["validator_code_sha256"]
            != W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_CODE_SHA256
            or raw["manifest_sha256"]
            != W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_MANIFEST_SHA256
            or raw["source_key"] != W02_MORPH_V3_PRIVATE_R5_SOURCE_KEY
            or raw["source_record_count"] != W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            or raw["validation_pass_count"] != W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            or raw["exact_source_identity_match_count"]
            != W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            or raw["minimum_sentence_ordinal"]
            != KYOTO_REMAINDER_MINIMUM_ORDINAL
            or raw["maximum_sentence_ordinal"] != KYOTO_TEST_SENTENCE_COUNT
            or raw["language_counts"] != {"lzh": 500}
            or raw["license_counts"] != {"CC-BY-SA-4.0": 500}
            or any(raw[name] != 0 for name in (
                "consumed_source_key_count", "selected_below_minimum_count",
                "unknown_source_key_count", "validation_failure_count"))):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 V6 source binding did not close")
    return raw


def _validate_label_binding(value: object) -> dict[str, Any]:
    raw = require_exact_dict(value, {
        "binding_sha256", "dimension_bindings", "double_pass_equal",
        "expected_true_count", "observation_mismatch_count",
        "owner_mode_mismatch_count", "semantic_pass_1_sha256",
        "semantic_pass_2_sha256", "stage_mismatch_count", "status",
        "unknown_dimension_key_count", "unknown_expected_state_count",
    }, where="R5 label semantic binding")
    expected = {
        name: {"count": count, "integer_components": list(key)}
        for name, key, count in W02_MORPH_V3_PRIVATE_R5_DIMENSION_BINDINGS
    }
    if (raw["status"] != "PASS"
            or raw["dimension_bindings"] != expected
            or raw["binding_sha256"]
            != W02_MORPH_V3_PRIVATE_R5_LABEL_BINDING_SHA256
            or raw["semantic_pass_1_sha256"] != raw["binding_sha256"]
            or raw["semantic_pass_2_sha256"] != raw["binding_sha256"]
            or raw["double_pass_equal"] != 1
            or raw["expected_true_count"] != W02_MORPH_V3_PRIVATE_PAIR_COUNT
            or any(raw[name] != 0 for name in (
                "observation_mismatch_count", "owner_mode_mismatch_count",
                "stage_mismatch_count", "unknown_dimension_key_count",
                "unknown_expected_state_count"))):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 label semantic binding did not close")
    return raw


def _receipt_files(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != 7:
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 safe metadata file inventory drifted")
    rows = []
    for item in value:
        raw = require_exact_dict(item, {
            "content_sha256", "content_size_bytes", "first_record_key", "kind",
            "last_record_key", "license_ids", "record_count", "relative_path",
            "split", "transport_sha256", "transport_size_bytes",
        }, where="R5 safe metadata file")
        relative = raw["relative_path"]
        layout = _LAYOUT_BY_PATH.get(relative)
        if layout is None:
            raise W02MorphologySuccessorV3PrivateOwnerR5Error(
                "R5 safe metadata relative path drifted")
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
    validate_r5_private_file_inventory(rows)
    if any(tuple(row["first_record_key"][:5]) != _NAMESPACE_PREFIX
           or tuple(row["last_record_key"][:5]) != _NAMESPACE_PREFIX
           for row in rows):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 owner namespace prefix drifted")
    return rows


def _validate_dimension_splits(value: object) -> dict[str, Any]:
    raw = require_exact_dict(
        value, set(W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS),
        where="R5 dimension split counts")
    expected = {"adversarial": 20, "held_out": 70, "wall": 10}
    if any(item != expected for item in raw.values()):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 dimension split counts drifted")
    return raw


def _validate_namespace(value: object, files: list[dict[str, object]]) -> dict[str, Any]:
    raw = require_exact_dict(value, {
        "all_components_positive_integers",
        "dataset_artifact_source_observation_label_cluster_fresh",
        "key_ranges", "kind_components", "policy", "status",
    }, where="R5 namespace audit")
    ranges = [{
        "first_record_key": row["first_record_key"],
        "kind": row["record_kind"],
        "last_record_key": row["last_record_key"],
        "relative_path": row["relative_path"],
    } for row in files]
    if (raw["status"] != "PASS"
            or raw["all_components_positive_integers"] != 1
            or raw["dataset_artifact_source_observation_label_cluster_fresh"] != 1
            or raw["key_ranges"] != ranges
            or raw["kind_components"] != {
                "artifact": 2, "carrier_node": 80, "cluster_range": [50, 63],
                "dataset": 1, "evaluator_owner": 90, "label": 40,
                "observation": 20, "source": 10,
            }
            or raw["policy"]
            != "OWNER_ID_DERIVED_FRESH_POSITIVE_INTEGER_PREFIX_WITH_KIND_COMPONENT"):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 namespace audit did not close")
    return raw


def build_w02_morphology_successor_v3_private_owner_r5_receipt_from_metadata(
        value: object) -> dict[str, object]:
    """验证安全 R5 metadata，并且只投影不含 payload 的字段。"""
    raw = require_exact_dict(value, {
        "adapter_probe_identity", "artifact_kind", "artifact_version",
        "audit_identity", "commitments", "contamination_audit",
        "dimension_denominators", "dimension_split_counts",
        "domain_disjoint_audit", "double_pass_audit", "duplicate_audit",
        "files", "label_semantic_binding_audit", "namespace_audit",
        "next_action", "ordinal_slice_identity", "owner_family_key",
        "owner_id", "pair_count", "public_commit", "public_git_write_count",
        "resource_limits", "resource_usage", "seal_identity", "source_count",
        "source_snapshot_identity", "source_validator_binding_audit",
        "split_counts", "status", "v6_identity", "zero_call_audit",
    }, where="R5 safe owner metadata")
    if (raw["artifact_kind"]
            != "PH2_D03_V2_W02_SUCCESSOR_V3_BLIND_PRIVATE_OWNER_R5_METADATA"
            or raw["artifact_version"]
            != "PH2-D03-V2-W02-SUCCESSOR-V3-BLIND-PRIVATE-OWNER-R5-V1"
            or raw["status"]
            != "OWNER_METADATA_FROZEN_SOURCE_V6_LABEL_BINDING_VERIFIED"
            or raw["next_action"]
            != "BUILD_SUCCESSOR_V3_PRIVATE_R5_OWNER_RECEIPT_IO_AND_EVALUATOR_REVISION"
            or raw["owner_id"] != W02_MORPH_V3_PRIVATE_R5_OWNER_ID
            or raw["owner_family_key"]
            != W02_MORPH_V3_PRIVATE_R5_OWNER_FAMILY_KEY
            or raw["public_commit"] != W02_MORPH_V3_PRIVATE_R5_PUBLIC_BASE_COMMIT
            or raw["source_count"] != W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            or raw["pair_count"] != W02_MORPH_V3_PRIVATE_PAIR_COUNT
            or raw["split_counts"] != W02_MORPH_V3_PRIVATE_SPLIT_COUNTS
            or raw["dimension_denominators"]
            != W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS
            or raw["public_git_write_count"] != 0):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 safe owner identity drifted")

    adapter = _validate_adapter_probe(raw["adapter_probe_identity"])
    v6 = require_exact_dict(raw["v6_identity"], {
        "code_sha256", "manifest_sha256", "status",
    }, where="R5 V6 identity")
    if v6 != {
            "code_sha256": W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_CODE_SHA256,
            "manifest_sha256":
                W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_MANIFEST_SHA256,
            "status": "PASS",
            }:
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 V6 identity drifted")
    files = _receipt_files(raw["files"])
    source = _validate_source_snapshot(raw["source_snapshot_identity"])
    source_binding = _validate_source_binding(
        raw["source_validator_binding_audit"])
    label_binding = _validate_label_binding(
        raw["label_semantic_binding_audit"])
    dimension_splits = _validate_dimension_splits(
        raw["dimension_split_counts"])
    namespace = _validate_namespace(raw["namespace_audit"], files)

    commitments = require_exact_dict(raw["commitments"], {
        "case_commitment", "cluster_commitment", "label_commitment",
        "payload_commitment",
    }, where="R5 commitments")
    for name, digest in commitments.items():
        require_sha256(digest, where=f"R5 {name}")
    contamination = require_exact_dict(raw["contamination_audit"], {
        "exact_case_overlap_count", "exact_cluster_overlap_count",
        "exact_content_overlap_count", "normalized_content_overlap_count",
        "public_source_identity_overlap_count", "status",
    }, where="R5 contamination audit")
    if contamination["status"] != "PASS" or any(
            contamination[name] != 0 for name in contamination
            if name != "status"):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 contamination audit did not close")
    duplicate = require_exact_dict(raw["duplicate_audit"], {
        "cluster_collision_count", "exact_duplicate_count",
        "label_key_duplicate_count", "near_duplicate_algorithm",
        "near_duplicate_count", "near_duplicate_minimum_normalized_length",
        "near_duplicate_pair_comparison_count",
        "near_duplicate_threshold_basis_points", "normalized_duplicate_count",
        "observation_key_duplicate_count", "source_key_duplicate_count",
        "status",
    }, where="R5 duplicate audit")
    if (duplicate["status"] != "PASS"
            or duplicate["near_duplicate_algorithm"]
            != "NFKC_CASEFOLD_NO_WHITESPACE_CHAR_TRIGRAM_DICE"
            or duplicate["near_duplicate_threshold_basis_points"] != 9_000
            or duplicate["near_duplicate_minimum_normalized_length"] != 4
            or duplicate["near_duplicate_pair_comparison_count"] != 124_750
            or any(duplicate[name] != 0 for name in (
                "cluster_collision_count", "exact_duplicate_count",
                "label_key_duplicate_count", "near_duplicate_count",
                "normalized_duplicate_count", "observation_key_duplicate_count",
                "source_key_duplicate_count"))):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 duplicate audit did not close")
    domain = require_exact_dict(raw["domain_disjoint_audit"], {
        "consumed_source_key_intersection_count", "old_private_payload_reads",
        "old_r2_formal_family_reads", "old_r2_owner_payload_reads",
        "old_r3_formal_family_reads", "old_r3_owner_payload_reads",
        "old_r4_formal_family_reads", "old_r4_owner_payload_reads",
        "ordinal_gap", "parallel", "previous_r4_selection_scan_upper_bound",
        "r4_r5_authorized_ordinal_intersection_count",
        "r5_minimum_sentence_ordinal",
        "same_upstream_repository_tag_blob_as_r4",
        "selected_below_r5_minimum_count", "status",
    }, where="R5 domain disjoint audit")
    if (domain["status"] != "PASS"
            or domain["same_upstream_repository_tag_blob_as_r4"] != 1
            or domain["previous_r4_selection_scan_upper_bound"] != 592
            or domain["r5_minimum_sentence_ordinal"] != 1_001
            or domain["ordinal_gap"] != 408
            or any(domain[name] != 0 for name in (
                "consumed_source_key_intersection_count",
                "old_private_payload_reads", "old_r2_formal_family_reads",
                "old_r2_owner_payload_reads", "old_r3_formal_family_reads",
                "old_r3_owner_payload_reads", "old_r4_formal_family_reads",
                "old_r4_owner_payload_reads",
                "r4_r5_authorized_ordinal_intersection_count", "parallel",
                "selected_below_r5_minimum_count"))):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 domain disjoint audit did not close")
    double_pass = require_exact_dict(raw["double_pass_audit"], {
        "aggregate_equal", "file_count", "pass_count", "status",
    }, where="R5 double-pass audit")
    if double_pass != {
            "aggregate_equal": 1, "file_count": 7, "pass_count": 2,
            "status": "PASS",
            }:
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 double-pass audit drifted")
    ordinal = require_exact_dict(raw["ordinal_slice_identity"], {
        "commitment", "maximum_sentence_ordinal", "minimum_sentence_ordinal",
        "previous_r4_selection_scan_upper_bound",
        "selected_below_minimum_count",
    }, where="R5 ordinal slice identity")
    if ordinal != {
            "commitment": W02_MORPH_V3_PRIVATE_R5_ORDINAL_SLICE_COMMITMENT,
            "maximum_sentence_ordinal": 5_528,
            "minimum_sentence_ordinal": 1_001,
            "previous_r4_selection_scan_upper_bound": 592,
            "selected_below_minimum_count": 0,
            }:
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 ordinal slice identity drifted")
    limits = require_exact_dict(raw["resource_limits"], {
        "logic_operations", "payload_bytes", "payload_gets", "records",
    }, where="R5 resource limits")
    if limits != {
            "logic_operations": 9_000_000, "payload_bytes": 536_870_912,
            "payload_gets": 300_000, "records": 100_000,
            }:
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 resource limits drifted")
    usage = require_exact_dict(raw["resource_usage"], set(limits),
                               where="R5 resource usage")
    if (usage["records"] != 1_500
            or any(type(usage[name]) is not int or usage[name] <= 0
                   or usage[name] > limits[name] for name in limits)):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 resource usage drifted")
    zero_call = require_exact_dict(raw["zero_call_audit"], {
        "candidate_calls", "evaluator_calls", "formal_private_evaluation_runs",
        "llm_calls", "main_session_conllu_content_reads",
        "old_private_payload_reads", "old_r2_formal_family_reads",
        "old_r2_owner_payload_reads", "old_r3_formal_family_reads",
        "old_r3_owner_payload_reads", "old_r4_formal_family_reads",
        "old_r4_owner_payload_reads", "public_git_staging", "public_git_writes",
        "teacher_calls", "v1_calls", "v2_calls", "v3_calls",
    }, where="R5 zero-call audit")
    if any(item != 0 for item in zero_call.values()):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 zero-call audit drifted")
    audit = _identity(raw["audit_identity"], where="R5 owner audit",
                      relative="owner-audit-report.json")
    seal = _identity(raw["seal_identity"], where="R5 owner seal",
                     relative="owner-seal.json")

    return {
        "adapter_probe_identity": dict(adapter),
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_OWNER_R5_RECEIPT"),
        "artifact_version": W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_VERSION,
        "candidate_evaluation_runs": 0,
        "commitments": dict(commitments),
        "contamination_audit": dict(contamination),
        "dimension_denominator_counts": dict(raw["dimension_denominators"]),
        "dimension_split_counts": dict(dimension_splits),
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
        "namespace_audit": dict(namespace),
        "next_action": "BUILD_SUCCESSOR_V3_PRIVATE_R5_EVALUATOR_FAMILY_REVISION",
        "ordinal_slice_identity": dict(ordinal),
        "owner_audit_identity": dict(audit),
        "owner_family_key": W02_MORPH_V3_PRIVATE_R5_OWNER_FAMILY_KEY,
        "owner_id": W02_MORPH_V3_PRIVATE_R5_OWNER_ID,
        "owner_metadata_sha256": W02_MORPH_V3_PRIVATE_R5_METADATA_SHA256,
        "owner_metadata_size_bytes": W02_MORPH_V3_PRIVATE_R5_METADATA_SIZE_BYTES,
        "owner_seal_identity": dict(seal),
        "pair_count": W02_MORPH_V3_PRIVATE_PAIR_COUNT,
        "public_identity": {
            "adapter_code_sha256": W02_MORPH_V3_PRIVATE_R5_ADAPTER_CODE_SHA256,
            "probe_code_sha256": W02_MORPH_V3_PRIVATE_R5_PROBE_CODE_SHA256,
            "probe_report_sha256": W02_MORPH_V3_PRIVATE_R5_PROBE_REPORT_SHA256,
            "public_repository_commit": W02_MORPH_V3_PRIVATE_R5_PUBLIC_BASE_COMMIT,
            "source_extension_v6_code_sha256":
                W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_CODE_SHA256,
            "source_extension_v6_manifest_sha256":
                W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_MANIFEST_SHA256,
            "source_extension_v6_status":
                "BLIND_PRIVATE_SOURCE_EXTENSION_V6_APPROVED",
        },
        "resource_limits": dict(limits),
        "resource_usage": dict(usage),
        "source_count": W02_MORPH_V3_PRIVATE_SOURCE_COUNT,
        "source_key": W02_MORPH_V3_PRIVATE_R5_SOURCE_KEY,
        "source_snapshot_identity": dict(source),
        "source_validator_binding_audit": dict(source_binding),
        "split_counts": dict(raw["split_counts"]),
        "status": "OWNER_METADATA_INGESTED_SOURCE_V6_PAYLOAD_UNREAD",
        "teacher_llm_provenance": {"llm_calls": 0, "teacher_calls": 0},
        "v6_identity": dict(v6),
        "zero_call_audit": dict(zero_call),
    }


__all__ = [
    "build_w02_morphology_successor_v3_private_owner_r5_receipt_from_metadata",
]
