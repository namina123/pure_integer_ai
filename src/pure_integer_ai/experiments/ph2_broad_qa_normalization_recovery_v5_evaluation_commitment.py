"""冻结 recovery-v5 Qt held-out 分母与正式判分不变量。

publisher 只读取 Qt source pack 的规范 manifest，不打开 raw TS、source-files 或
identity JSONL。它必须在任何 LibreOffice learner read 前发布；个体 label 仍由
后续 candidate/code/family freeze 后的唯一 formal guard 才能物化。
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_qt_source_pack import (
    NORMALIZATION_RECOVERY_V5_QT_SOURCE_PACK_KIND,
    NORMALIZATION_RECOVERY_V5_QT_SOURCE_STATUS,
    QT_ARCHIVE_SHA256,
    QT_COMMIT,
    QT_LICENSE_EXPRESSION,
    QT_OFFICIAL_SUMMARY,
    QT_ROOT_TREE,
    QT_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_V1")
NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_STATUS = (
    "LABEL_BLIND_DENOMINATOR_AND_GATES_FROZEN_BEFORE_TRAIN")

NORMALIZATION_RECOVERY_V5_DIMENSIONS = {
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
    """只读 Qt manifest，并核对物理和规范 JSON identity。"""
    root = Path(source_pack_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "recovery-v5 Qt source manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or sha256_hex(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "recovery-v5 Qt source manifest identity/encoding 漂移")
    return {**stored, "manifest_sha256": expected_manifest_sha256}


def _validate_source_manifest(value: object) -> dict[str, object]:
    """核验 held-out source、分母 identity 与整包禁训边界。"""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError("recovery-v5 Qt source manifest 非对象")
    acquisition = value.get("archive_acquisition")
    evaluation = value.get("evaluation_state")
    exclusion = value.get("training_exclusion")
    summary = value.get("parser_summary")
    license_value = value.get("license")
    files = value.get("files")
    if (value.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V5_QT_SOURCE_PACK_KIND
            or value.get("status")
            != NORMALIZATION_RECOVERY_V5_QT_SOURCE_STATUS
            or not isinstance(acquisition, dict)
            or acquisition.get("commit") != QT_COMMIT
            or acquisition.get("root_tree") != QT_ROOT_TREE
            or value.get("source_policy_scope") != QT_SOURCE_POLICY_SCOPE
            or not isinstance(evaluation, dict)
            or evaluation.get("formal_label_jsonl_materialized") != 0
            or evaluation.get("inventory_identity_materialized") != 1
            or evaluation.get("formal_evaluation_run_count") != 0
            or not isinstance(exclusion, dict)
            or exclusion.get("derivative_message_or_pair_allowed_in_v5_train")
            != 0
            or exclusion.get("learner_read_count") != 0
            or not isinstance(summary, dict)
            or any(not strict_json_equal(summary.get(key), expected)
                   for key, expected in QT_OFFICIAL_SUMMARY.items())
            or not isinstance(license_value, dict)
            or license_value.get("expression") != QT_LICENSE_EXPRESSION
            or not isinstance(files, list)):
        raise BroadQaExternalDataError(
            "recovery-v5 Qt held-out source commitment 漂移")
    raw = [item for item in files if isinstance(item, dict)
           and item.get("role") == "QT_TRANSLATIONS_RAW_ARCHIVE"]
    identity = [item for item in files if isinstance(item, dict)
                and item.get("role")
                == "QT_HELD_OUT_IDENTITY_WITHOUT_LABELS"]
    if (len(raw) != 1 or raw[0].get("sha256") != QT_ARCHIVE_SHA256
            or len(identity) != 1
            or identity[0].get("record_count")
            != QT_OFFICIAL_SUMMARY["plain_pair_count"]):
        raise BroadQaExternalDataError(
            "recovery-v5 Qt held-out physical identity 漂移")
    return value


def build_normalization_recovery_v5_evaluation_commitment(
        source_manifest: dict[str, object],
        ) -> dict[str, object]:
    """从 Qt manifest 构造零个体 label read 的 v5 分母合同。"""
    source = _validate_source_manifest(source_manifest)
    identity = next(
        item for item in source["files"]
        if item["role"] == "QT_HELD_OUT_IDENTITY_WITHOUT_LABELS")
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_KIND,
        "candidate_or_code_read_count": 0,
        "denominator": {
            "aggregate_buckets": {
                "equal_length_count": 1_997,
                "identity_count": 337,
                "nonidentity_count": 3_194,
                "single_han_difference_count": 349,
                "variable_length_count": 1_534,
            },
            "identity_artifact": identity,
            "label_blind": 1,
            "record_count": 3_531,
            "selection": (
                "ALL_FIXED_MODULE_ACTIVE_NON_NUMERUS_NONEMPTY_COMMON_SOURCE_"
                "IDENTITIES"),
        },
        "dimensions": NORMALIZATION_RECOVERY_V5_DIMENSIONS,
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
            "derivative_message_or_pair_allowed_in_v5_train": 0,
            "excluded_commit": QT_COMMIT,
            "excluded_root_tree": QT_ROOT_TREE,
            "excluded_source_pack_manifest_sha256": source[
                "manifest_sha256"],
            "exclusion_granularity": (
                "WHOLE_SOURCE_PACK_AND_ALL_DERIVATIVES"),
        },
        "source_manifest_read_count": 1,
        "source_non_manifest_file_read_count": 0,
        "status": NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_source_read_count": 0,
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v5 commitment run root 必须是 K 盘目录")
    return root


def publish_normalization_recovery_v5_evaluation_commitment(
        *,
        run_root: str | Path,
        qt_source_pack_dir: str | Path,
        expected_qt_source_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """在任何 v5 TRAIN read 前不可覆盖发布标签盲 commitment。"""
    root = _require_k_root(run_root)
    source = Path(qt_source_pack_dir).resolve()
    target = Path(target_dir).resolve()
    if (not source.is_dir() or not source.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "normalization recovery v5 commitment path 越出 run root")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization recovery v5 commitment target 已存在")
    source_manifest = _read_source_manifest_only(
        source,
        expected_manifest_sha256=expected_qt_source_manifest_sha256,
    )
    manifest = build_normalization_recovery_v5_evaluation_commitment(
        source_manifest)
    try:
        target.mkdir()
        encoded = canonical_json_line(manifest)
        with (target / "manifest.json").open("xb") as handle:
            handle.write(encoded)
    except OSError as error:
        raise BroadQaExternalDataError(
            "normalization recovery v5 commitment 发布失败") from error
    return {**manifest, "manifest_sha256": sha256_hex(encoded)}


def read_normalization_recovery_v5_evaluation_commitment(
        commitment_dir: str | Path,
        *,
        qt_source_pack_dir: str | Path,
        expected_qt_source_manifest_sha256: str,
        expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """严格回读 commitment，仍只读取 Qt manifest。"""
    root = Path(commitment_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization recovery v5 commitment 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or sha256_hex(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "normalization recovery v5 commitment identity/encoding 漂移")
    source_manifest = _read_source_manifest_only(
        qt_source_pack_dir,
        expected_manifest_sha256=expected_qt_source_manifest_sha256,
    )
    expected = build_normalization_recovery_v5_evaluation_commitment(
        source_manifest)
    if not strict_json_equal(stored, expected):
        raise BroadQaExternalDataError(
            "normalization recovery v5 commitment 字段漂移")
    return {**stored, "manifest_sha256": expected_manifest_sha256}


__all__ = [
    "NORMALIZATION_RECOVERY_V5_DIMENSIONS",
    "NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_KIND",
    "NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_STATUS",
    "build_normalization_recovery_v5_evaluation_commitment",
    "publish_normalization_recovery_v5_evaluation_commitment",
    "read_normalization_recovery_v5_evaluation_commitment",
]
