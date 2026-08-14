"""冻结 normalization recovery v3 的标签盲评测分母。

本模块只读取已封存 v2 evaluation protocol 的 ``manifest.json``。它把未读
reserve 的完整 identity、Firefox 整包禁训边界和 v3 判分不变量写成不可覆盖
commitment；不会打开 evaluation/reserve JSONL，也不会构造任何 label。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND,
    read_normalization_recovery_evaluation_manifest_only,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_source_pack import (
    FIREFOX_L10N_COMMIT,
    FIREFOX_L10N_REPOSITORY_URL,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V3_EVALUATION_COMMITMENT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V3_EVALUATION_COMMITMENT_V1")
NORMALIZATION_RECOVERY_V3_EVALUATION_COMMITMENT_STATUS = (
    "FROZEN_LABEL_BLIND_BEFORE_V3_TRAINING")

PRIOR_EVALUATION_PROTOCOL_MANIFEST_SHA256 = (
    "9a1aa10f2b4285e74e62a8a265967caeefbb31779faf7af2bf8c6c29f15dfb70")
EXCLUDED_FIREFOX_SOURCE_PACK_MANIFEST_SHA256 = (
    "f529973c7a57c15c604f58bec03517a50d01f70e29d4d13c391be08dea117d29")
PRIOR_RESERVE_IDENTITY = {
    "bytes": 419_102,
    "record_count": 1_558,
    "relative_path": "reserve.identity.jsonl",
    "role": "RESERVE_IDENTITY_WITHOUT_LABELS",
    "sha256": "afc79daf060616233e74bd4567ed27842e604a5f32668012da200ddd16c32be2",
}
PRIOR_RESERVE_SUMMARY = {
    "context_reserve_count": 31,
    "identity_reserve_count": 52,
    "local_mapping_reserve_count": 80,
    "phrase_reserve_count": 1_505,
    "reserve_count": 1_558,
    "reserve_family_counts": {
        "END_TO_END_COVERAGE": 1_557,
        "INDEPENDENT_CONTEXT_TRANSFER": 31,
        "LOCAL_MAPPING_TRANSFER": 80,
    },
}

NORMALIZATION_RECOVERY_V3_DIMENSIONS = {
    "DEFEATER_REPRESENTATION_EXECUTABILITY": {
        "bearing": 1,
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
        "identity_false_accept_count_max": 0,
        "length_mismatch_count_max": 0,
        "report_equal_and_variable_length_separately": 1,
        "report_phrase_length_buckets": 1,
        "wrong_changed_position_count_max": 0,
    },
    "INDEPENDENT_CONTEXT_TRANSFER": {
        "applicable_must_equal_full_inventory": 1,
        "exact_output_must_equal_applicable": 1,
        "missing_facility_outcome": "NE",
        "wrong_output_count_max": 0,
    },
    "LOCAL_MAPPING_TRANSFER": {
        "applicable_must_equal_full_inventory": 1,
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
}


def _sha256(payload: bytes) -> str:
    """返回规范 artifact 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, list):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v3 run root 必须是 K 盘目录")
    return root


def _validate_prior_manifest(value: object) -> dict[str, object]:
    """核验旧 manifest 的 reserve identity 与整包来源身份。"""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError("recovery v3 prior manifest 非对象")
    summary = value.get("inventory_summary")
    if (value.get("artifact_kind")
            != NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND
            or value.get("manifest_sha256")
            != PRIOR_EVALUATION_PROTOCOL_MANIFEST_SHA256
            or value.get("source_pack_manifest_sha256")
            != EXCLUDED_FIREFOX_SOURCE_PACK_MANIFEST_SHA256
            or not _strict_equal(
                value.get("reserve_identity"), PRIOR_RESERVE_IDENTITY)
            or not isinstance(summary, dict)
            or any(summary.get(key) != expected
                   for key, expected in PRIOR_RESERVE_SUMMARY.items())):
        raise BroadQaExternalDataError(
            "recovery v3 prior reserve/source commitment 漂移")
    return value


def build_normalization_recovery_v3_evaluation_commitment(
        prior_manifest: dict[str, object],
        ) -> dict[str, object]:
    """从旧 manifest 构造零 label read 的 v3 分母合同。"""
    _validate_prior_manifest(prior_manifest)
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V3_EVALUATION_COMMITMENT_KIND,
        "candidate_or_code_read_count": 0,
        "denominator": {
            "context_count": 31,
            "coverage_count": 1_557,
            "identity_count": 52,
            "label_blind": 1,
            "local_mapping_count": 80,
            "phrase_count": 1_505,
            "record_count": 1_558,
            "selection": "ENTIRE_PRIOR_UNREAD_RESERVE_WITHOUT_RESELECTION",
        },
        "dimensions": NORMALIZATION_RECOVERY_V3_DIMENSIONS,
        "formal_contract": {
            "candidate_applicability_cannot_shrink_denominator": 1,
            "denominator_or_threshold_change_after_publish_allowed": 0,
            "formal_run_count_max": 1,
            "label_materialization_allowed_after_candidate_code_family_freeze": 1,
            "label_materialization_allowed_before_candidate_code_family_freeze": 0,
            "overall_rule": "FAIL_DOMINATES_NE_DOMINATES_PASS",
            "production_enablement_during_evaluation": 0,
            "reserve_identity_reselection_allowed": 0,
        },
        "format_version": 1,
        "mastery_claimed": 0,
        "prior_evaluation_protocol_manifest_sha256": (
            PRIOR_EVALUATION_PROTOCOL_MANIFEST_SHA256),
        "prior_reserve_identity": PRIOR_RESERVE_IDENTITY,
        "production_enabled": 0,
        "reserve_identity_read_count": 0,
        "reserve_payload_read_count": 0,
        "source_exclusion": {
            "derivative_message_or_pair_allowed_in_v3_train": 0,
            "excluded_commit": FIREFOX_L10N_COMMIT,
            "excluded_repository_url": FIREFOX_L10N_REPOSITORY_URL,
            "excluded_source_pack_manifest_sha256": (
                EXCLUDED_FIREFOX_SOURCE_PACK_MANIFEST_SHA256),
            "exclusion_granularity": "WHOLE_SOURCE_PACK_AND_ALL_DERIVATIVES",
        },
        "status": NORMALIZATION_RECOVERY_V3_EVALUATION_COMMITMENT_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_source_read_count": 0,
    }


def publish_normalization_recovery_v3_evaluation_commitment(
        *,
        run_root: str | Path,
        prior_evaluation_protocol_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """在 v3 TRAIN 读取前不可覆盖发布标签盲 commitment。"""
    root = _require_k_root(run_root)
    prior = Path(prior_evaluation_protocol_dir).resolve()
    target = Path(target_dir).resolve()
    if (not prior.is_dir() or not prior.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "normalization recovery v3 commitment path 越出 run root")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization recovery v3 commitment target 已存在")
    prior_manifest = read_normalization_recovery_evaluation_manifest_only(
        prior,
        expected_manifest_sha256=(
            PRIOR_EVALUATION_PROTOCOL_MANIFEST_SHA256),
    )
    manifest = build_normalization_recovery_v3_evaluation_commitment(
        prior_manifest)
    try:
        target.mkdir()
        encoded = canonical_json_line(manifest)
        with (target / "manifest.json").open("xb") as handle:
            handle.write(encoded)
    except OSError as error:
        raise BroadQaExternalDataError(
            "normalization recovery v3 commitment 发布失败") from error
    return {**manifest, "manifest_sha256": _sha256(encoded)}


def read_normalization_recovery_v3_evaluation_commitment(
        commitment_dir: str | Path,
        *,
        prior_evaluation_protocol_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """严格回读 commitment，仍只读取旧 manifest。"""
    root = Path(commitment_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization recovery v3 commitment 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "normalization recovery v3 commitment identity/encoding 漂移")
    prior_manifest = read_normalization_recovery_evaluation_manifest_only(
        prior_evaluation_protocol_dir,
        expected_manifest_sha256=(
            PRIOR_EVALUATION_PROTOCOL_MANIFEST_SHA256),
    )
    expected = build_normalization_recovery_v3_evaluation_commitment(
        prior_manifest)
    if not _strict_equal(stored, expected):
        raise BroadQaExternalDataError(
            "normalization recovery v3 commitment 字段漂移")
    return {**stored, "manifest_sha256": expected_manifest_sha256}


__all__ = [
    "EXCLUDED_FIREFOX_SOURCE_PACK_MANIFEST_SHA256",
    "NORMALIZATION_RECOVERY_V3_DIMENSIONS",
    "NORMALIZATION_RECOVERY_V3_EVALUATION_COMMITMENT_KIND",
    "NORMALIZATION_RECOVERY_V3_EVALUATION_COMMITMENT_STATUS",
    "PRIOR_EVALUATION_PROTOCOL_MANIFEST_SHA256",
    "PRIOR_RESERVE_IDENTITY",
    "build_normalization_recovery_v3_evaluation_commitment",
    "publish_normalization_recovery_v3_evaluation_commitment",
    "read_normalization_recovery_v3_evaluation_commitment",
]
