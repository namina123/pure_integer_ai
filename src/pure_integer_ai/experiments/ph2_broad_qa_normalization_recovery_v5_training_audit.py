"""发布并严格回读 recovery-v5 TRAIN-only scoped audit artifact。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_rule_pack import (
    read_normalization_recovery_v5_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_audit_records import (
    derive_normalization_recovery_v5_training_audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_protocol import (
    read_normalization_recovery_v5_learner_input,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_V1")
NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_STATUS = (
    "TRAIN_ONLY_COMPLETE_NOT_FORMAL_NOT_DEPLOYED")

AUDIT_FILES = (
    ("runtime-audit.jsonl", "TRAIN_ONLY_SCOPED_RUNTIME_AUDIT", "case_id"),
    ("loso-audit.jsonl", "TRAIN_ONLY_FOUR_SOURCE_LOSO_AUDIT", "loso_id"),
)


def _sha256(payload: bytes) -> str:
    """返回 artifact 或文件 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(value, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(value, list):
        return (len(value) == len(expected)
                and all(_strict_equal(left, right)
                        for left, right in zip(value, expected)))
    return value == expected


def _require_k_root(value: str | Path) -> Path:
    """要求显式 audit 工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v5 audit run root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析 artifact 路径并拒绝逃出显式 K 盘根。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个 artifact 根是否相同或存在包含关系。"""
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _payload(values: tuple[dict[str, object], ...]) -> bytes:
    """把 audit records 编码为规范 JSONL。"""
    return b"".join(canonical_json_line(item) for item in values)


def _artifact(
        *,
        name: str,
        role: str,
        values: tuple[dict[str, object], ...],
        payload: bytes,
        ) -> dict[str, object]:
    """构造一个 audit 文件承诺。"""
    return {
        "bytes": len(payload),
        "record_count": len(values),
        "relative_path": name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _derive(
        *,
        protocol_dir: Path,
        expected_protocol_manifest_sha256: str,
        pack_dir: Path,
        expected_pack_manifest_sha256: str,
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]],
                   dict[str, bytes], list[dict[str, object]]]:
    """严格回读 protocol/pack 后重派生唯一 v5 audit material。"""
    protocol_values = read_normalization_recovery_v5_learner_input(
        protocol_dir,
        expected_manifest_sha256=expected_protocol_manifest_sha256,
    )
    protocol_manifest, observations, fragments, groups, _work = protocol_values
    pack_manifest, learned_outputs = read_normalization_recovery_v5_rule_pack(
        pack_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        expected_pack_manifest_sha256=expected_pack_manifest_sha256,
    )
    runtime_cases, loso_records, summary = (
        derive_normalization_recovery_v5_training_audit(
            protocol_manifest=protocol_manifest,
            observations=observations,
            fragments=fragments,
            groups=groups,
            pack_manifest=pack_manifest,
            outputs=learned_outputs,
        ))
    outputs = {
        "runtime-audit.jsonl": runtime_cases,
        "loso-audit.jsonl": loso_records,
    }
    payloads = {name: _payload(outputs[name])
                for name, _role, _identity in AUDIT_FILES}
    files = [_artifact(
        name=name, role=role, values=outputs[name], payload=payloads[name])
        for name, role, _identity in AUDIT_FILES]
    manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_KIND,
        "base_rule_pack_read_count": 0,
        "candidate_pack_read_count": 0,
        "evaluation_commitment_read_count": 0,
        "evaluation_payload_read_count": 0,
        "files": files,
        "formal_run_count": 0,
        "format_version": 1,
        "loso_contract": {
            "five_bucket_outcomes_required": 1,
            "full_pack_selection_reuse_allowed": 0,
            "held_out_observation_read_for_learning_allowed": 0,
            "identity_false_change_max": 0,
            "relearn_after_source_family_removal_required": 1,
            "required_exact_per_new_bucket_min": 1,
            "required_wrong_per_direction_max": 0,
            "source_scoped_rule_cross_family_execution_allowed": 0,
        },
        "mastery_claimed": 0,
        "pack_manifest_sha256": expected_pack_manifest_sha256,
        "predecessor_rule_pack_read_count": 0,
        "prior_formal_item_read_count": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": expected_protocol_manifest_sha256,
        "reserve_identity_read_count": 0,
        "reserve_payload_read_count": 0,
        "source_pack_read_count": 0,
        "status": NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
    }
    return manifest, outputs, payloads, files


def publish_normalization_recovery_v5_training_audit(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        pack_dir: str | Path,
        expected_pack_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 v5 TRAIN-only audit，并以 manifest-last 封口。"""
    root = _require_k_root(run_root)
    protocol_root = _within(root, protocol_dir, label="protocol_dir")
    pack_root = _within(root, pack_dir, label="pack_dir")
    target = _within(root, target_dir, label="target_dir")
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256,
        label="v5 audit protocol manifest")
    pack_sha = _sha_value(
        expected_pack_manifest_sha256,
        label="v5 audit pack manifest")
    if (not protocol_root.is_dir() or not pack_root.is_dir()
            or target.exists()
            or _overlap(protocol_root, pack_root)
            or _overlap(protocol_root, target)
            or _overlap(pack_root, target)):
        raise BroadQaExternalDataError(
            "v5 audit 输入缺失、artifact 混淆或 target 已存在")
    manifest, outputs, payloads, _files = _derive(
        protocol_dir=protocol_root,
        expected_protocol_manifest_sha256=protocol_sha,
        pack_dir=pack_root,
        expected_pack_manifest_sha256=pack_sha,
    )
    target.mkdir(parents=True)
    for name, _role, identity_key in AUDIT_FILES:
        values = outputs[name]
        identities = [str(item[identity_key]) for item in values]
        if len(set(identities)) != len(identities):
            raise BroadQaExternalDataError(
                f"v5 audit {name} identity 重复")
        with (target / name).open("xb") as handle:
            handle.write(payloads[name])
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def read_normalization_recovery_v5_training_audit(
        audit_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        pack_dir: str | Path,
        expected_pack_manifest_sha256: str,
        expected_audit_manifest_sha256: str,
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]]]:
    """以三外部 SHA 重派生并严格回读完整 v5 TRAIN-only audit。"""
    root = Path(audit_dir).resolve()
    protocol_root = Path(protocol_dir).resolve()
    pack_root = Path(pack_dir).resolve()
    if (_overlap(root, protocol_root) or _overlap(root, pack_root)
            or _overlap(protocol_root, pack_root)):
        raise BroadQaExternalDataError("v5 audit artifact 根混淆")
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256,
        label="v5 audit expected protocol manifest")
    pack_sha = _sha_value(
        expected_pack_manifest_sha256,
        label="v5 audit expected pack manifest")
    audit_sha = _sha_value(
        expected_audit_manifest_sha256,
        label="v5 audit expected manifest")
    expected, outputs, payloads, _files = _derive(
        protocol_dir=protocol_root,
        expected_protocol_manifest_sha256=protocol_sha,
        pack_dir=pack_root,
        expected_pack_manifest_sha256=pack_sha,
    )
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v5 audit manifest 不可读") from error
    if (_sha256(encoded) != audit_sha or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or not _strict_equal(stored, expected)):
        raise BroadQaExternalDataError(
            "v5 audit manifest identity/encoding/material 漂移")
    for name, _role, _identity in AUDIT_FILES:
        try:
            payload = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v5 audit {name} 不可读") from error
        if payload != payloads[name]:
            raise BroadQaExternalDataError(
                f"v5 audit {name} 与 protocol/pack 重派生漂移")
    return ({**stored, "manifest_sha256": audit_sha}, outputs)


__all__ = [
    "AUDIT_FILES",
    "NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_KIND",
    "NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_STATUS",
    "publish_normalization_recovery_v5_training_audit",
    "read_normalization_recovery_v5_training_audit",
]
