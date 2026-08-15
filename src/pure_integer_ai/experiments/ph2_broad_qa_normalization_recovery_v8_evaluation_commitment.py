"""冻结 recovery-v8 VLC 正式评测分母、维度与唯一运行合同。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    strict_json_equal,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_evaluation_commitment import (
    NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_KIND,
    NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_STATUS,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V8_EVALUATION_COMMITMENT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_EVALUATION_COMMITMENT_V1")
NORMALIZATION_RECOVERY_V8_EVALUATION_COMMITMENT_STATUS = (
    "VLC_FULL_DENOMINATOR_AND_SIX_DIMENSIONS_FROZEN_LABEL_BLIND_NOT_RUN")

NORMALIZATION_RECOVERY_V8_DIMENSION_ORDER = (
    "ORTHOGRAPHIC_ATOM_TRANSFER",
    "SOURCE_CONDITIONED_LEXICAL_TRANSFER",
    "IDENTITY_PRESERVATION",
    "STRUCTURE_AND_GENERATION_INTEGRITY",
    "RUNTIME_INDEXED_REFERENCE_EQUIVALENCE",
    "END_TO_END_COVERAGE",
)

NORMALIZATION_RECOVERY_V8_DIMENSIONS = {
    "ORTHOGRAPHIC_ATOM_TRANSFER": {
        "authorized_changed_exact_count_min": 1,
        "bearing": 1,
        "single_han_frozen_inventory_count": 350,
        "wrong_count_max": 0,
    },
    "SOURCE_CONDITIONED_LEXICAL_TRANSFER": {
        "authorized_changed_exact_count_min": 1,
        "bearing": 1,
        "official_source_exact_condition_required": 1,
        "unconditioned_execution_count_max": 0,
        "wrong_count_max": 0,
    },
    "IDENTITY_PRESERVATION": {
        "bearing": 1,
        "false_change_count_max": 0,
        "frozen_identity_inventory_count": 337,
        "identity_veto_exact_count_min": 1,
    },
    "STRUCTURE_AND_GENERATION_INTEGRITY": {
        "bearing": 1,
        "committed_structure_bearing_exact_count_min": 1,
        "exception_count_max": 0,
        "generation_hard_conjunct": 1,
        "partial_commit_count_max": 0,
        "structure_mismatch_count_max": 0,
    },
    "RUNTIME_INDEXED_REFERENCE_EQUIVALENCE": {
        "all_inputs_execute_twice": 1,
        "bearing": 1,
        "exception_count_max": 0,
        "indexed_reference_mismatch_count_max": 0,
        "production_enabled_must_equal": 0,
    },
    "END_TO_END_COVERAGE": {
        "bearing": 1,
        "changed_exact_count_min": 2,
        "exact_unknown_wrong_must_equal_full_inventory": 1,
        "full_frozen_inventory_count": 3_656,
        "wrong_count_max": 0,
    },
}


def _sha256(payload: bytes) -> str:
    """返回 commitment 或前驱 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise BroadQaExternalDataError(f"v8 commitment {label} 非法")
    return value


def _read_manifest_only(
        directory: str | Path, *, expected_sha256: str, label: str,
        ) -> dict[str, object]:
    """只读取一份规范 manifest，不打开同目录 payload。"""
    expected = _sha_value(expected_sha256, label=f"{label} SHA")
    try:
        encoded = (Path(directory).resolve() / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(f"v8 commitment {label} 不可读") from error
    if (_sha256(encoded) != expected or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(f"v8 commitment {label} identity 漂移")
    return {**stored, "manifest_sha256": expected}


def _validate_v7_commitment(value: object) -> dict[str, object]:
    """核对 VLC 先冻结分母，不接触 identity roster 或翻译文本。"""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError("v8 commitment v7 前驱非对象")
    denominator = value.get("denominator")
    buckets = denominator.get("aggregate_buckets") if isinstance(
        denominator, dict) else None
    source = value.get("source_exclusion")
    if (value.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_KIND
            or value.get("status")
            != NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_STATUS
            or not isinstance(denominator, dict)
            or denominator.get("label_blind") != 1
            or denominator.get("record_count") != 3_656
            or not isinstance(buckets, dict)
            or buckets != {
                "equal_length_count": 1_967,
                "identity_count": 337,
                "nonidentity_count": 3_319,
                "single_han_difference_count": 350,
                "structure_equal_count": 3_652,
                "variable_length_count": 1_689,
            }
            or not isinstance(source, dict)
            or not isinstance(source.get(
                "excluded_source_pack_manifest_sha256"), str)
            or len(str(source["excluded_source_pack_manifest_sha256"])) != 64
            or value.get("production_enabled") != 0
            or value.get("mastery_claimed") != 0):
        raise BroadQaExternalDataError("v8 commitment VLC 前驱边界漂移")
    return value


def build_normalization_recovery_v8_evaluation_commitment(
        v7_commitment: dict[str, object],
        ) -> dict[str, object]:
    """继承 VLC 分母并冻结六维 EXACT/UNKNOWN/WRONG 正式合同。"""
    predecessor = _validate_v7_commitment(v7_commitment)
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_EVALUATION_COMMITMENT_KIND,
        "candidate_or_code_read_count": 0,
        "denominator": predecessor["denominator"],
        "dimensions": NORMALIZATION_RECOVERY_V8_DIMENSIONS,
        "format_version": 1,
        "formal_contract": {
            "candidate_applicability_cannot_shrink_denominator": 1,
            "candidate_code_family_freeze_required_before_label_read": 1,
            "denominator_or_threshold_change_after_publish_allowed": 0,
            "formal_run_count_max": 1,
            "guard_write_required_before_archive_or_label_read": 1,
            "host_candidate_and_publication_write_isolation_required": 1,
            "individual_label_publication_allowed": 0,
            "missing_required_evidence_outcome": "NE",
            "overall_rule": "FAIL_DOMINATES_NE_DOMINATES_PASS",
            "retry_after_any_terminal_or_post_guard_exception_allowed": 0,
            "wrong_committed_output_outcome": "FAIL",
        },
        "judgements": ("EXACT", "UNKNOWN", "WRONG"),
        "mastery_claimed": 0,
        "production_enabled": 0,
        "source_exclusion": predecessor["source_exclusion"],
        "status": NORMALIZATION_RECOVERY_V8_EVALUATION_COMMITMENT_STATUS,
        "teacher_api_llm_call_count": 0,
        "v7_commitment_manifest_read_count": 1,
        "v7_evaluation_commitment_manifest_sha256": predecessor[
            "manifest_sha256"],
        "vlc_identity_raw_or_translation_read_count": 0,
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v8 commitment run root 必须在 K 盘")
    return root


def publish_normalization_recovery_v8_evaluation_commitment(
        *, run_root: str | Path, v7_commitment_dir: str | Path,
        expected_v7_commitment_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布标签盲 v8 正式 commitment。"""
    root = _require_k_root(run_root)
    predecessor = Path(v7_commitment_dir).resolve()
    target = Path(target_dir).resolve()
    if (not predecessor.is_dir() or not predecessor.is_relative_to(root)
            or not target.is_relative_to(root) or target.exists()
            or target == predecessor or target.is_relative_to(predecessor)
            or predecessor.is_relative_to(target)):
        raise BroadQaExternalDataError("v8 commitment path 非法")
    value = _read_manifest_only(
        predecessor,
        expected_sha256=expected_v7_commitment_manifest_sha256,
        label="v7 commitment",
    )
    manifest = build_normalization_recovery_v8_evaluation_commitment(value)
    target.mkdir()
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v8_evaluation_commitment(
        commitment_dir: str | Path, *, v7_commitment_dir: str | Path,
        expected_v7_commitment_manifest_sha256: str,
        expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """严格回读 v8 commitment，仍只读取两份 manifest。"""
    stored = _read_manifest_only(
        commitment_dir, expected_sha256=expected_manifest_sha256,
        label="v8 commitment")
    predecessor = _read_manifest_only(
        v7_commitment_dir,
        expected_sha256=expected_v7_commitment_manifest_sha256,
        label="v7 commitment")
    expected = build_normalization_recovery_v8_evaluation_commitment(
        predecessor)
    comparable = {key: item for key, item in stored.items()
                  if key != "manifest_sha256"}
    if not strict_json_equal(comparable, expected):
        raise BroadQaExternalDataError("v8 commitment fields 漂移")
    return stored


__all__ = [
    "NORMALIZATION_RECOVERY_V8_DIMENSIONS",
    "NORMALIZATION_RECOVERY_V8_DIMENSION_ORDER",
    "NORMALIZATION_RECOVERY_V8_EVALUATION_COMMITMENT_KIND",
    "NORMALIZATION_RECOVERY_V8_EVALUATION_COMMITMENT_STATUS",
    "build_normalization_recovery_v8_evaluation_commitment",
    "publish_normalization_recovery_v8_evaluation_commitment",
    "read_normalization_recovery_v8_evaluation_commitment",
]
