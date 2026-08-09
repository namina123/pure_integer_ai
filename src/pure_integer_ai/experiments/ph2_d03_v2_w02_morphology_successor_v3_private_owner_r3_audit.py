"""Pure audit validators for the successor V3 R3 owner receipt."""
from __future__ import annotations

import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r3_contract import (
    W02_MORPH_V3_PRIVATE_PAIR_COUNT,
    W02_MORPH_V3_PRIVATE_R3_DIMENSION_BINDINGS,
    W02_MORPH_V3_PRIVATE_R3_LABEL_BINDING_SHA256,
    W02_MORPH_V3_PRIVATE_R3_LABEL_BINDING_VERSION,
    W02_MORPH_V3_PRIVATE_R3_LABEL_PASS_SHA256,
    W02_MORPH_V3_PRIVATE_R3_SOURCE_KEY,
    W02MorphologySuccessorV3PrivateOwnerR3Error,
    require_exact_dict,
    require_sha256,
)


def _exact_zero_fields(
        value: object, fields: set[str], *, where: str,
        status: str = "PASS") -> dict[str, Any]:
    raw = require_exact_dict(value, {*fields, "status"}, where=where)
    if raw["status"] != status or any(raw[name] != 0 for name in fields):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            f"{where} did not close at zero")
    return raw


def validate_r3_label_semantic_binding(value: object) -> dict[str, Any]:
    raw = require_exact_dict(value, {
        "binding_version", "dimension_binding_sha256", "dimension_bindings",
        "double_pass_equal", "expected_state_counts", "first_pass_sha256",
        "observation_key_mismatch_count", "owner_mode_mismatch_count",
        "second_pass_sha256", "status", "unknown_dimension_key_count",
        "unknown_expected_state_count", "visible_stage_mismatch_count",
    }, where="R3 label semantic binding")
    expected_rows = [
        {
            "dimension_key": list(key),
            "dimension_name": name,
            "record_count": count,
        }
        for name, key, count in W02_MORPH_V3_PRIVATE_R3_DIMENSION_BINDINGS
    ]
    zero_fields = (
        "observation_key_mismatch_count", "owner_mode_mismatch_count",
        "unknown_dimension_key_count", "unknown_expected_state_count",
        "visible_stage_mismatch_count",
    )
    if (raw["binding_version"]
            != W02_MORPH_V3_PRIVATE_R3_LABEL_BINDING_VERSION
            or raw["dimension_bindings"] != expected_rows
            or raw["dimension_binding_sha256"]
            != W02_MORPH_V3_PRIVATE_R3_LABEL_BINDING_SHA256
            or hashlib.sha256(canonical_json_bytes(expected_rows)).hexdigest()
            != raw["dimension_binding_sha256"]
            or raw["expected_state_counts"] != {
                "TRUE": W02_MORPH_V3_PRIVATE_PAIR_COUNT}
            or raw["double_pass_equal"] != 1
            or raw["first_pass_sha256"]
            != W02_MORPH_V3_PRIVATE_R3_LABEL_PASS_SHA256
            or raw["second_pass_sha256"] != raw["first_pass_sha256"]
            or raw["status"] != "PASS"
            or any(raw[name] != 0 for name in zero_fields)):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 label semantic binding did not close")
    core = {
        key: raw[key] for key in (
            "binding_version", "dimension_binding_sha256",
            "dimension_bindings", "expected_state_counts",
            "observation_key_mismatch_count", "owner_mode_mismatch_count",
            "unknown_dimension_key_count", "unknown_expected_state_count",
            "visible_stage_mismatch_count",
        )
    }
    if hashlib.sha256(canonical_json_bytes(core)).hexdigest() != raw[
            "first_pass_sha256"]:
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 label semantic pass identity drifted")
    return raw


def validate_r3_owner_audits(value: dict[str, Any]) -> None:
    contamination = _exact_zero_fields(
        value.get("contamination"), {
            "exact_case_overlap_count", "exact_cluster_overlap_count",
            "exact_content_overlap_count", "normalized_content_overlap_count",
        }, where="R3 contamination")
    assert contamination["status"] == "PASS"

    domain = require_exact_dict(value.get("domain_disjoint_audit"), {
        "consumed_news_source_key_intersection_count", "consumed_prefix",
        "formal_private_family_reads", "news_accepted_count",
        "old_private_payload_reads", "r2_formal_family_reads",
        "r2_owner_payload_reads", "required_locator_prefix",
        "required_vs_consumed_prefix_domain_intersection_count",
        "source_prefix_counts", "status",
    }, where="R3 domain disjoint audit")
    if (domain["status"] != "PASS" or domain["consumed_prefix"] != "n"
            or domain["required_locator_prefix"] != "w"
            or domain["source_prefix_counts"] != {"n": 500, "w": 500}
            or any(domain[name] != 0 for name in (
                "consumed_news_source_key_intersection_count",
                "formal_private_family_reads", "news_accepted_count",
                "old_private_payload_reads", "r2_formal_family_reads",
                "r2_owner_payload_reads",
                "required_vs_consumed_prefix_domain_intersection_count"))):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 domain disjoint audit drifted")

    double_pass = require_exact_dict(value.get("double_pass_audit"), {
        "double_pass_equal", "first_pass_sha256", "second_pass_sha256",
        "seven_file_actual_decompression_passes", "status",
    }, where="R3 double-pass audit")
    if (double_pass["status"] != "PASS"
            or double_pass["double_pass_equal"] != 1
            or double_pass["seven_file_actual_decompression_passes"] != 2
            or require_sha256(
                double_pass["first_pass_sha256"], where="R3 first pass")
            != double_pass["second_pass_sha256"]):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 double-pass audit drifted")

    duplicate = require_exact_dict(value.get("duplicate_audit"), {
        "cluster_collision_count", "exact_case_overlap_count",
        "exact_cluster_overlap_count", "exact_content_overlap_count",
        "label_key_duplicate_count", "near_duplicate_algorithm",
        "near_duplicate_count", "near_duplicate_length_ratio",
        "near_duplicate_pair_comparisons", "near_duplicate_threshold",
        "normalized_content_overlap_count", "observation_key_duplicate_count",
        "owner_exact_duplicate_count", "owner_normalized_duplicate_count",
        "status",
    }, where="R3 duplicate audit")
    zero_names = (
        "cluster_collision_count", "exact_case_overlap_count",
        "exact_cluster_overlap_count", "exact_content_overlap_count",
        "label_key_duplicate_count", "near_duplicate_count",
        "normalized_content_overlap_count", "observation_key_duplicate_count",
        "owner_exact_duplicate_count", "owner_normalized_duplicate_count",
    )
    if (duplicate["status"] != "PASS"
            or duplicate["near_duplicate_algorithm"]
            != "UNICODE_NFKC_CASEFOLD_WHITESPACE_COLLAPSE_CHARACTER_TRIGRAM_SET_JACCARD_V1"
            or duplicate["near_duplicate_length_ratio"]
            != {"denominator": 10, "numerator": 9}
            or duplicate["near_duplicate_threshold"]
            != {"denominator": 10, "numerator": 9}
            or duplicate["near_duplicate_pair_comparisons"] != 124_750
            or any(duplicate[name] != 0 for name in zero_names)):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 duplicate audit drifted")

    zero_call = require_exact_dict(value.get("zero_call_audit"), {
        "candidate_evaluation_runs", "evaluator_runs",
        "formal_private_evaluation_runs", "llm_calls",
        "main_session_payload_reads", "old_private_payload_reads",
        "public_git_writes", "r2_formal_family_reads",
        "r2_owner_payload_reads", "teacher_calls",
    }, where="R3 zero-call audit")
    if any(item != 0 for item in zero_call.values()):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 zero-call audit drifted")

    resource_limits = require_exact_dict(value.get("resource_limits"), {
        "max_logic_operations", "max_payload_bytes", "max_payload_gets",
        "max_records",
    }, where="R3 resource limits")
    if resource_limits != {
            "max_logic_operations": 9_000_000,
            "max_payload_bytes": 536_870_912,
            "max_payload_gets": 300_000,
            "max_records": 100_000,
            }:
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 resource limits drifted")
    usage = require_exact_dict(value.get("resource_usage"), {
        "logic_operation_accounting_version", "logic_operations",
        "max_logic_operations", "max_payload_bytes", "max_payload_gets",
        "max_records", "payload_bytes", "payload_gets", "records",
        "transport_bytes",
    }, where="R3 resource usage")
    if (usage["logic_operation_accounting_version"]
            != "OWNER-R3-LOGIC-UNITS-V1"
            or any(usage[name] != resource_limits[name] for name in (
                "max_logic_operations", "max_payload_bytes",
                "max_payload_gets", "max_records"))
            or usage["logic_operations"] > usage["max_logic_operations"]
            or usage["payload_bytes"] > usage["max_payload_bytes"]
            or usage["payload_gets"] > usage["max_payload_gets"]
            or usage["records"] > usage["max_records"]
            or usage["records"] != 1_500
            or any(type(usage[name]) is not int or usage[name] <= 0 for name in (
                "logic_operations", "payload_bytes", "payload_gets",
                "records", "transport_bytes"))):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 resource usage drifted")

    source = require_exact_dict(value.get("source_snapshot_identity"), {
        "commit_sha1", "file_sha256", "file_size_bytes", "git_blob_sha1",
        "n_input_count", "n_rejected_count", "repository",
        "required_locator_prefix", "source_key", "tag",
        "total_sentence_count", "w_input_count",
        "wikipedia_accepted_count", "wikipedia_rejected_count",
    }, where="R3 source snapshot")
    if (source["source_key"] != W02_MORPH_V3_PRIVATE_R3_SOURCE_KEY
            or source["required_locator_prefix"] != "w"
            or source["n_input_count"] != 500
            or source["n_rejected_count"] != 500
            or source["w_input_count"] != 500
            or source["wikipedia_accepted_count"] != 500
            or source["wikipedia_rejected_count"] != 0
            or source["total_sentence_count"] != 1_000):
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(
            "R3 source snapshot domain drifted")
    require_sha256(source["file_sha256"], where="R3 source file")

    validate_r3_label_semantic_binding(value.get("label_semantic_binding_audit"))


__all__ = [
    "validate_r3_label_semantic_binding",
    "validate_r3_owner_audits",
]
