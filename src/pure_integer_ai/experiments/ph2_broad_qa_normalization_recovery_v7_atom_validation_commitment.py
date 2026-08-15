"""冻结独立 Audacity atom-validation 分母、门与一次性顺序。

publisher/reader 只读取 Audacity source-pack、TRAIN atom feasibility 与 VLC final
commitment 的 manifest。它不打开 Audacity raw/identity/translation，不创建 proposal、
candidate/runtime/formal，也不消耗 VLC one-shot held-out。
"""
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_identifiability_audit import (
    NORMALIZATION_RECOVERY_V7_ATOM_IDENTIFIABILITY_AUDIT_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_source_pack import (
    AUDACITY_COMMIT,
    AUDACITY_ROOT_TREE,
    AUDACITY_SOURCE_FAMILY,
    AUDACITY_SOURCE_POLICY_SCOPE,
    NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_PACK_KIND,
    NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_evaluation_commitment import (
    NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_KIND,
    NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_STATUS,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_"
    "AUDACITY_ATOM_VALIDATION_COMMITMENT_V1")
NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_STATUS = (
    "LABEL_BLIND_ATOM_VALIDATION_DENOMINATOR_AND_GATES_FROZEN_"
    "NOT_FAMILY_NOT_RUN")

_EXPECTED_ATOM_STATUS = (
    "TRAIN_ONLY_ATOM_IDENTIFIABILITY_FEASIBILITY_PASS_NOT_RUNTIME")
_EXPECTED_ATOM_OUTCOMES = {"EXACT": 2, "UNKNOWN": 12, "WRONG": 0}
_EXPECTED_VLC_RECORD_COUNT = 3_656


def _sha256(payload: bytes) -> str:
    """返回规范 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _read_manifest_only(
        directory: str | Path,
        *,
        expected_sha256: str,
        label: str,
        ) -> dict[str, object]:
    """只读取规范 manifest，不打开同目录其他 artifact 文件。"""
    root = Path(directory).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"Audacity atom-validation {label} manifest 不可读") from error
    if (not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or _sha256(encoded) != expected_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            f"Audacity atom-validation {label} manifest identity 漂移")
    return {**stored, "manifest_sha256": expected_sha256}


def _validate_source_manifest(value: object) -> dict[str, object]:
    """核对 Audacity selection、分母与零运行状态。"""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError(
            "Audacity atom-validation source manifest 非对象")
    acquisition = value.get("git_acquisition")
    selection = value.get("selection_boundary")
    summary = value.get("parser_summary")
    state = value.get("validation_state")
    files = value.get("files")
    if (value.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_PACK_KIND
            or value.get("status")
            != NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_STATUS
            or value.get("source_family") != AUDACITY_SOURCE_FAMILY
            or value.get("source_policy_scope")
            != AUDACITY_SOURCE_POLICY_SCOPE
            or not isinstance(acquisition, dict)
            or acquisition.get("commit") != AUDACITY_COMMIT
            or acquisition.get("root_tree") != AUDACITY_ROOT_TREE
            or not isinstance(selection, dict)
            or selection.get("source_selected_before_translation_blob_read")
            != 1
            or selection.get(
                "individual_translation_or_pair_read_before_selection") != 0
            or not isinstance(summary, dict)
            or type(summary.get("plain_pair_count")) is not int
            or int(summary["plain_pair_count"]) <= 0
            or not isinstance(files, list)
            or len([
                item for item in files if isinstance(item, dict)
                and item.get("role")
                == "AUDACITY_ATOM_VALIDATION_IDENTITY_WITHOUT_LABELS"
                and item.get("record_count") == summary["plain_pair_count"]
            ]) != 1
            or not isinstance(state, dict)
            or any(state.get(key) != 0 for key in (
                "candidate_or_runtime_read_count",
                "formal_label_jsonl_materialized",
                "individual_translation_surface_published_in_jsonl",
                "validation_run_count"))
            or state.get("raw_translation_surface_stored_on_k") != 1):
        raise BroadQaExternalDataError(
            "Audacity atom-validation source boundary 漂移")
    return value


def _validate_atom_manifest(value: object) -> dict[str, object]:
    """核对 section 90 TRAIN-only lower-bound identity 与边界。"""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError(
            "Audacity atom-validation atom manifest 非对象")
    summary = value.get("summary")
    identifiability = summary.get("identifiability") \
        if isinstance(summary, dict) else None
    scoring = identifiability.get("scoring") \
        if isinstance(identifiability, dict) else None
    if (value.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V7_ATOM_IDENTIFIABILITY_AUDIT_KIND
            or value.get("status") != _EXPECTED_ATOM_STATUS
            or not isinstance(scoring, dict)
            or not strict_json_equal(
                scoring.get("outcome_counts"), _EXPECTED_ATOM_OUTCOMES)
            or any(value.get(key) != 0 for key in (
                "candidate_family_formal_run_count", "mastery_claimed",
                "production_enabled", "runtime_program_published",
                "teacher_api_llm_call_count",
                "train_source_or_output_surface_published"))):
        raise BroadQaExternalDataError(
            "Audacity atom-validation atom predecessor 漂移")
    return value


def _validate_vlc_commitment(value: object) -> dict[str, object]:
    """证明 VLC final one-shot commitment 仍未被本 validation 消耗。"""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError(
            "Audacity atom-validation VLC commitment 非对象")
    denominator = value.get("denominator")
    formal = value.get("formal_contract")
    if (value.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_KIND
            or value.get("status")
            != NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_STATUS
            or not isinstance(denominator, dict)
            or denominator.get("record_count") != _EXPECTED_VLC_RECORD_COUNT
            or not isinstance(formal, dict)
            or formal.get("formal_run_count_max") != 1
            or value.get("source_non_manifest_file_read_count") != 0
            or any(value.get(key) != 0 for key in (
                "candidate_or_code_read_count", "mastery_claimed",
                "production_enabled", "teacher_api_llm_call_count",
                "training_source_read_count"))):
        raise BroadQaExternalDataError(
            "Audacity atom-validation VLC reserve boundary 漂移")
    return value


def build_audacity_atom_validation_commitment(
        *,
        source_manifest: dict[str, object],
        atom_manifest: dict[str, object],
        vlc_commitment: dict[str, object],
        ) -> dict[str, object]:
    """从三个 manifest 构造标签盲独立 validation 协议。"""
    source = _validate_source_manifest(source_manifest)
    atom = _validate_atom_manifest(atom_manifest)
    vlc = _validate_vlc_commitment(vlc_commitment)
    summary = source["parser_summary"]
    return {
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_KIND),
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
            "label_blind": 1,
            "record_count": summary["plain_pair_count"],
            "selection": source["selection_boundary"]["selection"],
            "source_family": AUDACITY_SOURCE_FAMILY,
            "source_policy_scope": AUDACITY_SOURCE_POLICY_SCOPE,
        },
        "format_version": 1,
        "gates": {
            "exception_count_max": 0,
            "exact_output_count_min": 1,
            "identity_false_change_count_max": 0,
            "indexed_reference_mismatch_count_max": 0,
            "label_read_before_authorization_freeze_max": 0,
            "partial_commit_count_max": 0,
            "selection_drift_count_max": 0,
            "structure_token_mismatch_count_max": 0,
            "wrong_output_count_max": 0,
        },
        "inputs": {
            "atom_identifiability_manifest_sha256": atom["manifest_sha256"],
            "audacity_source_pack_manifest_sha256": source[
                "manifest_sha256"],
            "vlc_final_commitment_manifest_sha256": vlc[
                "manifest_sha256"],
        },
        "mastery_claimed": 0,
        "outcome_rule": {
            "FAIL": "ANY_WRONG_OR_HARD_CONJUNCT_FAILURE",
            "NE": "ZERO_EXACT_ZERO_WRONG",
            "PASS": "NONZERO_EXACT_ZERO_WRONG_AND_ALL_HARD_CONJUNCTS",
            "precedence": "FAIL_DOMINATES_NE_DOMINATES_PASS",
        },
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": (
            NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_STATUS),
        "teacher_api_llm_call_count": 0,
        "validation_contract": {
            "candidate_runtime_formal_enabled": 0,
            "denominator_or_threshold_change_after_publish_allowed": 0,
            "family_code_freeze_required_before_translation_label_read": 1,
            "held_output_may_select_proposal_or_authorization": 0,
            "training_from_validation_family_allowed": 0,
            "validation_run_count_max": 1,
            "vlc_final_one_shot_consumed_by_this_validation": 0,
        },
        "validation_reads": {
            "atom_non_manifest_file_read_count": 0,
            "audacity_identity_raw_or_translation_read_count": 0,
            "source_manifest_read_count": 1,
            "vlc_commitment_manifest_read_count": 1,
            "vlc_identity_raw_or_translation_read_count": 0,
        },
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式 run root 是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "Audacity atom-validation commitment root 必须是 K 盘目录")
    return root


def _overlap(left: Path, right: Path) -> bool:
    """判断两个目录是否相同或互为祖先。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def publish_audacity_atom_validation_commitment(
        *,
        run_root: str | Path,
        source_pack_dir: str | Path,
        expected_source_manifest_sha256: str,
        atom_audit_dir: str | Path,
        expected_atom_manifest_sha256: str,
        vlc_commitment_dir: str | Path,
        expected_vlc_commitment_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布只读 manifest 的 atom-validation commitment。"""
    root = _require_k_root(run_root)
    paths = tuple(Path(value).resolve() for value in (
        source_pack_dir, atom_audit_dir, vlc_commitment_dir, target_dir))
    if (any(not path.is_relative_to(root) for path in paths)
            or any(not path.is_dir() for path in paths[:-1])
            or paths[-1].exists()
            or any(_overlap(paths[-1], path) for path in paths[:-1])):
        raise BroadQaExternalDataError(
            "Audacity atom-validation commitment path 非法")
    source = _read_manifest_only(
        paths[0], expected_sha256=expected_source_manifest_sha256,
        label="source pack")
    atom = _read_manifest_only(
        paths[1], expected_sha256=expected_atom_manifest_sha256,
        label="atom audit")
    vlc = _read_manifest_only(
        paths[2], expected_sha256=expected_vlc_commitment_manifest_sha256,
        label="VLC commitment")
    manifest = build_audacity_atom_validation_commitment(
        source_manifest=source,
        atom_manifest=atom,
        vlc_commitment=vlc,
    )
    paths[-1].mkdir()
    manifest_path = paths[-1] / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(
        manifest_path.read_bytes())}


def read_audacity_atom_validation_commitment(
        commitment_dir: str | Path,
        *,
        source_pack_dir: str | Path,
        expected_source_manifest_sha256: str,
        atom_audit_dir: str | Path,
        expected_atom_manifest_sha256: str,
        vlc_commitment_dir: str | Path,
        expected_vlc_commitment_manifest_sha256: str,
        expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """严格回读 commitment，仍只读取三个输入 manifest。"""
    root = Path(commitment_dir).resolve()
    stored = _read_manifest_only(
        root, expected_sha256=expected_manifest_sha256,
        label="commitment")
    source = _read_manifest_only(
        source_pack_dir, expected_sha256=expected_source_manifest_sha256,
        label="source pack")
    atom = _read_manifest_only(
        atom_audit_dir, expected_sha256=expected_atom_manifest_sha256,
        label="atom audit")
    vlc = _read_manifest_only(
        vlc_commitment_dir,
        expected_sha256=expected_vlc_commitment_manifest_sha256,
        label="VLC commitment")
    expected = build_audacity_atom_validation_commitment(
        source_manifest=source,
        atom_manifest=atom,
        vlc_commitment=vlc,
    )
    comparable = {
        key: value for key, value in stored.items()
        if key != "manifest_sha256"}
    if not strict_json_equal(comparable, expected):
        raise BroadQaExternalDataError(
            "Audacity atom-validation commitment fields 漂移")
    return stored


__all__ = [
    "NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_KIND",
    "NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_STATUS",
    "build_audacity_atom_validation_commitment",
    "publish_audacity_atom_validation_commitment",
    "read_audacity_atom_validation_commitment",
]
