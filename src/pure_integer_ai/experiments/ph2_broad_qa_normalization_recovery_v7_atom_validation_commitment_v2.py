"""无 label read 修正 Audacity atom-validation commitment v1 的结果门。

v2 只读取 v1 manifest。它保留来源、分母、input SHA 与全部零失败门，但要求
至少一条 authorized changed EXACT；identity-only EXACT 不再满足 transfer PASS。
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_commitment import (
    NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_KIND,
    NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_source_pack import (
    AUDACITY_SOURCE_FAMILY,
    AUDACITY_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_"
    "AUDACITY_ATOM_VALIDATION_COMMITMENT_V2")
NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_STATUS = (
    "LABEL_BLIND_AUTHORIZED_CHANGE_GATE_FROZEN_"
    "NOT_FAMILY_NOT_RUN")


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
            f"Audacity atom-validation v2 {label} 不可读") from error
    if (not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or _sha256(encoded) != expected_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            f"Audacity atom-validation v2 {label} identity 漂移")
    return {**stored, "manifest_sha256": expected_sha256}


def _validate_v1(value: object) -> dict[str, object]:
    """核对 v1 分母、门、input SHA 与零运行边界。"""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError(
            "Audacity atom-validation v1 commitment 非对象")
    denominator = value.get("denominator")
    gates = value.get("gates")
    inputs = value.get("inputs")
    contract = value.get("validation_contract")
    reads = value.get("validation_reads")
    if (value.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_KIND
            or value.get("status")
            != NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_STATUS
            or not isinstance(denominator, dict)
            or denominator.get("label_blind") != 1
            or denominator.get("source_family") != AUDACITY_SOURCE_FAMILY
            or denominator.get("source_policy_scope")
            != AUDACITY_SOURCE_POLICY_SCOPE
            or type(denominator.get("record_count")) is not int
            or int(denominator["record_count"]) <= 0
            or not isinstance(gates, dict)
            or gates.get("exact_output_count_min") != 1
            or gates.get("wrong_output_count_max") != 0
            or not isinstance(inputs, dict)
            or set(inputs) != {
                "atom_identifiability_manifest_sha256",
                "audacity_source_pack_manifest_sha256",
                "vlc_final_commitment_manifest_sha256",
            }
            or any(not isinstance(item, str) or len(item) != 64
                   for item in inputs.values())
            or not isinstance(contract, dict)
            or contract.get("validation_run_count_max") != 1
            or contract.get(
                "family_code_freeze_required_before_translation_label_read")
            != 1
            or not isinstance(reads, dict)
            or any(reads.get(key) != 0 for key in (
                "atom_non_manifest_file_read_count",
                "audacity_identity_raw_or_translation_read_count",
                "vlc_identity_raw_or_translation_read_count"))
            or any(value.get(key) != 0 for key in (
                "mastery_claimed", "production_enabled",
                "runtime_program_published", "teacher_api_llm_call_count"))):
        raise BroadQaExternalDataError(
            "Audacity atom-validation v1 commitment boundary 漂移")
    return value


def build_audacity_atom_validation_commitment_v2(
        v1_manifest: dict[str, object],
        ) -> dict[str, object]:
    """继承 v1 identity，并冻结 authorized-change 非零 PASS 门。"""
    v1 = _validate_v1(v1_manifest)
    gates = dict(v1["gates"])
    del gates["exact_output_count_min"]
    gates["authorized_changed_exact_output_count_min"] = 1
    contract = dict(v1["validation_contract"])
    contract["identity_only_exact_satisfies_transfer_pass"] = 0
    return {
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_KIND),
        "denominator": v1["denominator"],
        "format_version": 2,
        "gates": gates,
        "inputs": {
            **v1["inputs"],
            "superseded_v1_commitment_manifest_sha256": v1[
                "manifest_sha256"],
        },
        "mastery_claimed": 0,
        "outcome_rule": {
            "FAIL": "ANY_WRONG_OR_HARD_CONJUNCT_FAILURE",
            "NE": "ZERO_AUTHORIZED_CHANGED_EXACT_ZERO_WRONG",
            "PASS": (
                "NONZERO_AUTHORIZED_CHANGED_EXACT_ZERO_WRONG_"
                "AND_ALL_HARD_CONJUNCTS"),
            "precedence": "FAIL_DOMINATES_NE_DOMINATES_PASS",
        },
        "production_enabled": 0,
        "revision": {
            "denominator_changed": 0,
            "identity_exact_reported_but_not_transfer_bearing": 1,
            "label_or_individual_record_read_count": 0,
            "reason": "IDENTITY_ONLY_EXACT_COULD_SATISFY_V1_PASS",
            "threshold_relaxed": 0,
            "v1_superseded_for_family_freeze": 1,
        },
        "runtime_program_published": 0,
        "status": (
            NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_STATUS),
        "teacher_api_llm_call_count": 0,
        "validation_contract": contract,
        "validation_reads": {
            "audacity_identity_raw_or_translation_read_count": 0,
            "predecessor_non_manifest_file_read_count": 0,
            "v1_commitment_manifest_read_count": 1,
            "vlc_identity_raw_or_translation_read_count": 0,
        },
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式 run root 是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "Audacity atom-validation v2 root 必须是 K 盘目录")
    return root


def _overlap(left: Path, right: Path) -> bool:
    """判断两个目录是否相同或互为祖先。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def publish_audacity_atom_validation_commitment_v2(
        *,
        run_root: str | Path,
        v1_commitment_dir: str | Path,
        expected_v1_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布只读 v1 manifest 的 corrected commitment。"""
    root = _require_k_root(run_root)
    predecessor = Path(v1_commitment_dir).resolve()
    target = Path(target_dir).resolve()
    if (not predecessor.is_dir()
            or not predecessor.is_relative_to(root)
            or not target.is_relative_to(root)
            or target.exists()
            or _overlap(target, predecessor)):
        raise BroadQaExternalDataError(
            "Audacity atom-validation v2 path 非法")
    v1 = _read_manifest_only(
        predecessor,
        expected_sha256=expected_v1_manifest_sha256,
        label="v1 commitment manifest")
    manifest = build_audacity_atom_validation_commitment_v2(v1)
    target.mkdir()
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(
        manifest_path.read_bytes())}


def read_audacity_atom_validation_commitment_v2(
        commitment_dir: str | Path,
        *,
        v1_commitment_dir: str | Path,
        expected_v1_manifest_sha256: str,
        expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """严格回读 v2，仍只读取 v1 与 v2 两份 manifest。"""
    stored = _read_manifest_only(
        commitment_dir,
        expected_sha256=expected_manifest_sha256,
        label="v2 commitment manifest")
    v1 = _read_manifest_only(
        v1_commitment_dir,
        expected_sha256=expected_v1_manifest_sha256,
        label="v1 commitment manifest")
    expected = build_audacity_atom_validation_commitment_v2(v1)
    comparable = {
        key: value for key, value in stored.items()
        if key != "manifest_sha256"}
    if not strict_json_equal(comparable, expected):
        raise BroadQaExternalDataError(
            "Audacity atom-validation v2 commitment fields 漂移")
    return stored


__all__ = [
    "NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_KIND",
    "NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_STATUS",
    "build_audacity_atom_validation_commitment_v2",
    "publish_audacity_atom_validation_commitment_v2",
    "read_audacity_atom_validation_commitment_v2",
]
