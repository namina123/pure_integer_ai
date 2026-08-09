"""Pure aggregate audits for the successor V3 private owner receipt."""
from __future__ import annotations

from typing import Any

from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v3 import (
    BLOCKED_OWNER_CODE,
    BLOCKED_OWNER_METADATA_SHA256,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_contract import (
    W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS,
    W02_MORPH_V3_PRIVATE_OWNER_ID,
    W02_MORPH_V3_PRIVATE_PAIR_COUNT,
    W02_MORPH_V3_PRIVATE_SOURCE_COUNT,
    W02_MORPH_V3_PRIVATE_SOURCE_SNAPSHOT_COMMITMENT,
    W02MorphologySuccessorV3PrivateOwnerError,
    require_exact_dict,
    require_positive,
    require_sha256,
)


def _validate_zero_overlap(value: object, *, where: str) -> None:
    row = require_exact_dict(value, {
        "exact_case_overlap_count", "exact_cluster_overlap_count",
        "exact_content_overlap_count", "normalized_content_overlap_count",
        "proof_basis",
    }, where=where)
    if (row["proof_basis"] != "CASE_AND_CLUSTER_BIND_NORMALIZED_CONTENT_SHA256"
            or any(row[key] != 0 for key in row if key != "proof_basis")):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            f"{where} contamination audit did not close")


def _validate_resource_budget(value: object) -> None:
    raw = require_exact_dict(
        value, {"limits", "usage"}, where="V3 owner resource budget")
    limits = require_exact_dict(raw["limits"], {
        "max_logic_operations", "max_payload_bytes", "max_payload_gets",
        "max_records",
    }, where="V3 owner resource limits")
    usage = require_exact_dict(raw["usage"], {
        "logic_operations", "payload_bytes", "payload_gets", "record_count",
    }, where="V3 owner resource usage")
    if limits != {
            "max_logic_operations": 9_000_000,
            "max_payload_bytes": 536_870_912,
            "max_payload_gets": 300_000,
            "max_records": 100_000,
            }:
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner resource limits drifted")
    for used, maximum in (
            ("logic_operations", "max_logic_operations"),
            ("payload_bytes", "max_payload_bytes"),
            ("payload_gets", "max_payload_gets"),
            ("record_count", "max_records")):
        if require_positive(
                usage[used], where=f"V3 owner {used}") > limits[maximum]:
            raise W02MorphologySuccessorV3PrivateOwnerError(
                "V3 owner resource usage exceeded its limit")
    if usage["record_count"] != (
            W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            + W02_MORPH_V3_PRIVATE_PAIR_COUNT * 2):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner resource record count drifted")


def _validate_filter_audit(value: object) -> None:
    audit = require_exact_dict(value, {
        "input_count", "near_duplicate_hit_count",
        "near_duplicate_thresholds", "public_baseline_counts",
        "public_baseline_mode", "rejected_counts", "retained_count",
    }, where="V3 owner filter audit")
    thresholds = require_exact_dict(audit["near_duplicate_thresholds"], {
        "character_sequence_ratio_milli", "length_ratio_milli",
        "trigram_jaccard_milli",
    }, where="V3 owner near-duplicate thresholds")
    baseline = require_exact_dict(audit["public_baseline_counts"], {
        "authored_dev_unknown_nonce_pattern_covered",
        "authored_train_family_count", "gsdsimp_dev_count",
        "gsdsimp_train_count", "wiktionary_adversarial_reconstructed",
        "wiktionary_dev_reconstructed", "wiktionary_held_out_reconstructed",
        "wiktionary_pages_scanned", "wiktionary_train_reconstructed",
    }, where="V3 owner public baseline counts")
    if (audit["input_count"] != 500 or audit["retained_count"] != 500
            or audit["near_duplicate_hit_count"] != 0
            or audit["rejected_counts"] != {}
            or audit["public_baseline_mode"]
            != "PUBLIC_RAW_RECONSTRUCTION_PLUS_AUTHORED_DEV_GENERATOR_PATTERN"
            or thresholds != {
                "character_sequence_ratio_milli": 960,
                "length_ratio_milli": 900,
                "trigram_jaccard_milli": 850,
            }
            or any(type(item) is not int or item <= 0
                   for item in baseline.values())):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner filter audit drifted")


def validate_v3_private_owner_audits(raw: dict[str, Any]) -> None:
    """Validate all pure aggregate evidence after top-level identity checks."""
    commitments = require_exact_dict(raw["commitments"], {
        "case_commitment", "cluster_commitment", "label_commitment",
        "payload_commitment",
    }, where="V3 owner commitments")
    for key, digest in commitments.items():
        require_sha256(digest, where=f"V3 owner {key}")
    if len(set(commitments.values())) != 4:
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner commitments are not independent")

    contamination = require_exact_dict(
        raw["contamination_audit"], {"dev", "shadow", "train"},
        where="V3 owner contamination audit")
    for split in ("train", "dev", "shadow"):
        _validate_zero_overlap(contamination[split], where=f"V3 owner {split}")
    duplicates = require_exact_dict(raw["within_owner_duplicate_audit"], {
        "exact_case_duplicate_count", "exact_cluster_collision_count",
        "exact_content_duplicate_count", "near_duplicate_pair_count",
        "normalized_content_duplicate_count",
    }, where="V3 owner duplicate audit")
    if any(item != 0 for item in duplicates.values()):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner duplicate audit did not close")

    old_private = require_exact_dict(raw["old_private_disjoint_audit"], {
        "old_private_payload_reads", "policy",
    }, where="V3 old private audit")
    if (old_private["old_private_payload_reads"] != 0
            or old_private["policy"]
            != "PUD_NEWS_SOURCE_IDENTITY_WITH_ZERO_OLD_PRIVATE_PAYLOAD_READS"):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 old private disjoint audit drifted")
    blocked = require_exact_dict(raw["blocked_owner_disjoint_audit"], {
        "blocked_owner_payload_reads", "new_opaque_owner_id", "policy",
        "previous_blocked_metadata_sha256", "previous_blocker_code",
    }, where="V3 blocked owner audit")
    if (blocked["blocked_owner_payload_reads"] != 0
            or blocked["new_opaque_owner_id"]
            != W02_MORPH_V3_PRIVATE_OWNER_ID
            or blocked["policy"]
            != "NEW_ROOT_ID_COMMITMENTS_WITH_ZERO_BLOCKED_OWNER_PAYLOAD_READS"
            or blocked["previous_blocked_metadata_sha256"]
            != BLOCKED_OWNER_METADATA_SHA256
            or blocked["previous_blocker_code"] != BLOCKED_OWNER_CODE):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 blocked owner disjoint audit drifted")
    previous = require_exact_dict(raw["previous_blocked_owner"], {
        "blocker_code", "metadata_sha256", "payload_reads",
    }, where="V3 previous blocked owner")
    if (previous["blocker_code"] != BLOCKED_OWNER_CODE
            or previous["metadata_sha256"] != BLOCKED_OWNER_METADATA_SHA256
            or previous["payload_reads"] != 0):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 previous blocked owner binding drifted")

    for key, relative in (
            ("audit_report", "owner-audit-report.json"),
            ("owner_seal", "owner-seal.json")):
        identity = require_exact_dict(raw[key], {
            "relative_path", "sha256", "size_bytes",
        }, where=f"V3 owner {key}")
        if identity["relative_path"] != relative:
            raise W02MorphologySuccessorV3PrivateOwnerError(
                f"V3 owner {key} path drifted")
        require_sha256(identity["sha256"], where=f"V3 owner {key}")
        require_positive(identity["size_bytes"], where=f"V3 owner {key} size")

    provenance = require_exact_dict(raw["owner_teacher_llm_provenance"], {
        "deterministic_adapter_runs", "llm_calls", "teacher_calls", "tool",
    }, where="V3 owner provenance")
    if (provenance["deterministic_adapter_runs"] != 2
            or provenance["llm_calls"] != 0
            or provenance["teacher_calls"] != 0
            or provenance["tool"] != "PYTHON_STANDARD_LIBRARY_DETERMINISTIC"):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner provenance drifted")

    double_pass = require_exact_dict(raw["double_pass_audit"], {
        "dimension_denominators", "pass_1_commitment", "pass_2_commitment",
        "stable_key_unique",
    }, where="V3 owner double-pass audit")
    pass_1 = require_sha256(
        double_pass["pass_1_commitment"], where="V3 owner pass 1")
    pass_2 = require_sha256(
        double_pass["pass_2_commitment"], where="V3 owner pass 2")
    if (pass_1 != pass_2 or double_pass["stable_key_unique"] != 1
            or double_pass["dimension_denominators"]
            != W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner double-pass audit drifted")

    pud = require_exact_dict(raw["pud_domain_audit"], {
        "n_accepted", "n_input", "n_rejected", "w_input", "w_rejected",
    }, where="V3 PUD domain audit")
    if pud != {
            "n_accepted": 500, "n_input": 500, "n_rejected": 0,
            "w_input": 500, "w_rejected": 500}:
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 PUD domain audit drifted")
    _validate_resource_budget(raw["resource_budget"])
    _validate_filter_audit(raw["within_owner_filter_audit"])
    if raw["source_snapshot_commitment"] != (
            W02_MORPH_V3_PRIVATE_SOURCE_SNAPSHOT_COMMITMENT):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 source snapshot commitment drifted")


__all__ = ["validate_v3_private_owner_audits"]
