"""冻结 recovery-v7 VLC held-out 分母与正式判分不变量。

publisher 只读取 VLC source pack 的规范 manifest，不打开 raw PO、source-files 或
identity roster。它必须在任何 v7 learner 改动前发布；个体 label 仍只能由 future
candidate/code/family freeze 后的不可逆 formal guard 物化。
"""
from __future__ import annotations

import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    sha256_hex,
    strict_json_equal,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_vlc_source_pack import (
    NORMALIZATION_RECOVERY_V7_VLC_SOURCE_PACK_KIND,
    NORMALIZATION_RECOVERY_V7_VLC_SOURCE_STATUS,
    VLC_ARCHIVE_SHA256,
    VLC_COMMIT,
    VLC_LICENSE_EXPRESSION,
    VLC_OFFICIAL_SUMMARY,
    VLC_PO_TREE,
    VLC_ROOT_TREE,
    VLC_SOURCE_FILES,
    VLC_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_V1")
NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_STATUS = (
    "LABEL_BLIND_DENOMINATOR_AND_GATES_FROZEN_BEFORE_V7_LEARNER_CHANGE")

NORMALIZATION_RECOVERY_V7_DIMENSIONS = {
    "CONTEXT_CONDITIONED_TRANSFER": {
        "applicable_must_equal_full_frozen_context_inventory": 1,
        "exact_output_must_equal_applicable": 1,
        "missing_facility_outcome": "NE",
        "wrong_output_count_max": 0,
    },
    "DEFEATER_REPRESENTATION_EXECUTABILITY": {
        "declared_must_equal_executable": 1,
        "identity_only_defeater_count_max": 0,
        "malformed_defeater_count_max": 0,
        "missing_facility_outcome": "NE",
    },
    "END_TO_END_COVERAGE": {
        "applicable_must_equal_full_inventory": 1,
        "exact_output_must_equal_applicable": 1,
        "false_accept_count_max": 0,
        "false_reject_count_max": 0,
        "length_mismatch_count_max": 0,
        "wrong_changed_position_count_max": 0,
    },
    "IDENTITY_PRESERVATION": {
        "applicable_must_equal_identity_inventory": 1,
        "exact_output_must_equal_applicable": 1,
        "false_change_count_max": 0,
        "hard_conjunct": 1,
    },
    "LOCAL_MAPPING_TRANSFER": {
        "applicable_must_equal_full_frozen_mapping_inventory": 1,
        "exact_output_must_equal_applicable": 1,
        "false_accept_count_max": 0,
        "false_reject_count_max": 0,
        "scope_mismatch_count_max": 0,
        "unscoped_rule_count_max": 0,
    },
    "RUNTIME_PRODUCTION_BEHAVIOR": {
        "all_inputs_execute_twice": 1,
        "exception_count_max": 0,
        "indexed_reference_mismatch_count_max": 0,
        "production_enabled_must_equal": 0,
        "target_policy_scope_required": 1,
    },
    "SOURCE_POLICY_CONFLICT": {
        "declared_conflict_must_equal_observed": 1,
        "identity_priority_conflict_block_counts_as_observed": 1,
        "missing_facility_outcome": "NE",
        "policy_specific_replay_must_equal_observation": 1,
        "unscoped_conflict_execution_count_max": 0,
    },
    "VARIABLE_LENGTH_WHOLE_INPUT_TRANSFER": {
        "applicable_must_equal_variable_length_inventory": 1,
        "exact_output_must_equal_applicable": 1,
        "length_mismatch_count_max": 0,
        "missing_facility_outcome": "NE",
        "wrong_output_count_max": 0,
    },
}


def _read_source_manifest_only(
        source_pack_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """只读 VLC manifest，并核对物理和规范 JSON identity。"""
    root = Path(source_pack_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "recovery-v7 VLC source manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or sha256_hex(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "recovery-v7 VLC source manifest identity/encoding 漂移")
    return {**stored, "manifest_sha256": expected_manifest_sha256}


def _validate_source_manifest(value: object) -> dict[str, object]:
    """核验先选源边界、held-out 分母 identity 与整包禁训状态。"""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError("recovery-v7 VLC source manifest 非对象")
    acquisition = value.get("archive_acquisition")
    evaluation = value.get("evaluation_state")
    exclusion = value.get("training_exclusion")
    selection = value.get("selection_boundary")
    summary = value.get("parser_summary")
    license_value = value.get("license")
    files = value.get("files")
    if (value.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V7_VLC_SOURCE_PACK_KIND
            or value.get("status")
            != NORMALIZATION_RECOVERY_V7_VLC_SOURCE_STATUS
            or not isinstance(acquisition, dict)
            or acquisition.get("commit") != VLC_COMMIT
            or acquisition.get("root_tree") != VLC_ROOT_TREE
            or acquisition.get("po_tree") != VLC_PO_TREE
            or value.get("source_policy_scope") != VLC_SOURCE_POLICY_SCOPE
            or not isinstance(selection, dict)
            or selection.get("source_selected_before_translation_label_parse")
            != 1
            or selection.get(
                "individual_translation_or_pair_read_before_selection") != 0
            or not isinstance(evaluation, dict)
            or evaluation.get("formal_label_jsonl_materialized") != 0
            or evaluation.get("inventory_identity_materialized") != 1
            or evaluation.get("formal_evaluation_run_count") != 0
            or not isinstance(exclusion, dict)
            or exclusion.get("derivative_message_or_pair_allowed_in_v7_train")
            != 0
            or exclusion.get(
                "learner_profiler_selector_case_browser_read_count") != 0
            or not isinstance(summary, dict)
            or any(not strict_json_equal(summary.get(key), expected)
                   for key, expected in VLC_OFFICIAL_SUMMARY.items())
            or not isinstance(license_value, dict)
            or license_value.get("expression") != VLC_LICENSE_EXPRESSION
            or license_value.get("copying_git_blob_sha1")
            != VLC_SOURCE_FILES["COPYING"]["git_blob_sha1"]
            or license_value.get("copying_sha256")
            != VLC_SOURCE_FILES["COPYING"]["sha256"]
            or not isinstance(files, list)):
        raise BroadQaExternalDataError(
            "recovery-v7 VLC held-out source commitment 漂移")
    raw = [item for item in files if isinstance(item, dict)
           and item.get("role") == "VLC_TRANSLATIONS_RAW_ARCHIVE"]
    identity = [item for item in files if isinstance(item, dict)
                and item.get("role")
                == "VLC_HELD_OUT_IDENTITY_WITHOUT_LABELS"]
    if (len(raw) != 1 or raw[0].get("sha256") != VLC_ARCHIVE_SHA256
            or len(identity) != 1
            or identity[0].get("record_count")
            != VLC_OFFICIAL_SUMMARY["plain_pair_count"]):
        raise BroadQaExternalDataError(
            "recovery-v7 VLC held-out physical identity 漂移")
    return value


def build_normalization_recovery_v7_evaluation_commitment(
        source_manifest: dict[str, object],
        ) -> dict[str, object]:
    """从 VLC manifest 构造零个体 label read 的 v7 分母合同。"""
    source = _validate_source_manifest(source_manifest)
    summary = source["parser_summary"]
    identity = next(
        item for item in source["files"]
        if item["role"] == "VLC_HELD_OUT_IDENTITY_WITHOUT_LABELS")
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_KIND,
        "candidate_or_code_read_count": 0,
        "denominator": {
            "aggregate_buckets": {
                "equal_length_count": summary["equal_length_pair_count"],
                "identity_count": summary["identity_pair_count"],
                "nonidentity_count": summary["nonidentity_pair_count"],
                "single_han_difference_count": summary[
                    "single_han_difference_count"],
                "structure_equal_count": summary["structure_equal_count"],
                "variable_length_count": summary[
                    "variable_length_pair_count"],
            },
            "identity_artifact": identity,
            "label_blind": 1,
            "record_count": summary["plain_pair_count"],
            "selection": (
                "ALL_COMMON_SINGULAR_NONFUZZY_NONOBSOLETE_NONEMPTY_"
                "MSGCTXT_MSGID_MSGID_PLURAL_IDENTITIES"),
            "structure_equal_required_for_denominator": 0,
        },
        "dimensions": NORMALIZATION_RECOVERY_V7_DIMENSIONS,
        "formal_contract": {
            "candidate_applicability_cannot_shrink_denominator": 1,
            "candidate_code_family_freeze_required_before_label_read": 1,
            "denominator_or_threshold_change_after_publish_allowed": 0,
            "formal_run_count_max": 1,
            "label_materialization_allowed_before_guard": 0,
            "overall_rule": "FAIL_DOMINATES_NE_DOMINATES_PASS",
            "production_enablement_during_evaluation": 0,
            "source_identity_reselection_allowed": 0,
        },
        "format_version": 1,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "source_exclusion": {
            "derivative_message_or_pair_allowed_in_v7_train": 0,
            "excluded_commit": VLC_COMMIT,
            "excluded_root_tree": VLC_ROOT_TREE,
            "excluded_source_pack_manifest_sha256": source[
                "manifest_sha256"],
            "exclusion_granularity": (
                "WHOLE_SOURCE_PACK_AND_ALL_DERIVATIVES"),
        },
        "source_manifest_read_count": 1,
        "source_non_manifest_file_read_count": 0,
        "status": NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_source_read_count": 0,
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v7 commitment run root 必须是 K 盘目录")
    return root


def publish_normalization_recovery_v7_evaluation_commitment(
        *,
        run_root: str | Path,
        vlc_source_pack_dir: str | Path,
        expected_vlc_source_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """在任何 v7 learner 改动前不可覆盖发布标签盲 commitment。"""
    root = _require_k_root(run_root)
    source = Path(vlc_source_pack_dir).resolve()
    target = Path(target_dir).resolve()
    if (not source.is_dir() or not source.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "normalization recovery v7 commitment path 越出 run root")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization recovery v7 commitment target 已存在")
    source_manifest = _read_source_manifest_only(
        source,
        expected_manifest_sha256=expected_vlc_source_manifest_sha256,
    )
    manifest = build_normalization_recovery_v7_evaluation_commitment(
        source_manifest)
    try:
        target.mkdir()
        encoded = canonical_json_line(manifest)
        with (target / "manifest.json").open("xb") as handle:
            handle.write(encoded)
    except OSError as error:
        raise BroadQaExternalDataError(
            "normalization recovery v7 commitment 发布失败") from error
    return {**manifest, "manifest_sha256": sha256_hex(encoded)}


def read_normalization_recovery_v7_evaluation_commitment(
        commitment_dir: str | Path,
        *,
        vlc_source_pack_dir: str | Path,
        expected_vlc_source_manifest_sha256: str,
        expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """严格回读 commitment，仍只读取 VLC manifest。"""
    root = Path(commitment_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization recovery v7 commitment 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or sha256_hex(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "normalization recovery v7 commitment identity/encoding 漂移")
    source_manifest = _read_source_manifest_only(
        vlc_source_pack_dir,
        expected_manifest_sha256=expected_vlc_source_manifest_sha256,
    )
    expected = build_normalization_recovery_v7_evaluation_commitment(
        source_manifest)
    if not strict_json_equal(stored, expected):
        raise BroadQaExternalDataError(
            "normalization recovery v7 commitment 字段漂移")
    return {**stored, "manifest_sha256": expected_manifest_sha256}


__all__ = [
    "NORMALIZATION_RECOVERY_V7_DIMENSIONS",
    "NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_KIND",
    "NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_STATUS",
    "build_normalization_recovery_v7_evaluation_commitment",
    "publish_normalization_recovery_v7_evaluation_commitment",
    "read_normalization_recovery_v7_evaluation_commitment",
]
