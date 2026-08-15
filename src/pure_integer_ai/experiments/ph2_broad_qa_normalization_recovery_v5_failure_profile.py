"""发布并严格回读 recovery-v5 TRAIN-only LOSO failure profile。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_failure_profile_records import (
    derive_normalization_recovery_v5_failure_profile,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_protocol import (
    read_normalization_recovery_v5_learner_input,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_V1")
NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_STATUS = (
    "TRAIN_ONLY_DIAGNOSTIC_COMPLETE_NOT_SELECTION_NOT_EVALUATION")

PROFILE_FILES = (
    ("wrong-cases.jsonl", "TRAIN_ONLY_LOSO_WRONG_CASES", "case_id"),
    ("rule-impacts.jsonl", "TRAIN_ONLY_LOSO_RULE_IMPACTS", "impact_id"),
    ("family-summaries.jsonl", "TRAIN_ONLY_LOSO_FAMILY_SUMMARIES",
     "family_summary_id"),
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
    """要求显式 profile 工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v5 failure profile root 必须是 K 盘目录")
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


def _read_audit_manifest_only(
        audit_dir: Path,
        *,
        expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """只打开 sealed training-audit manifest 并核验规范编码和外部 SHA。"""
    expected_sha = _sha_value(
        expected_manifest_sha256, label="v5 profile expected audit manifest")
    try:
        encoded = (audit_dir / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v5 failure profile audit manifest 不可读") from error
    if (_sha256(encoded) != expected_sha or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v5 failure profile audit manifest identity/encoding 漂移")
    return stored


def _payload(values: tuple[dict[str, object], ...]) -> bytes:
    """把 profile records 编码为规范 JSONL。"""
    return b"".join(canonical_json_line(item) for item in values)


def _artifact(
        *,
        name: str,
        role: str,
        values: tuple[dict[str, object], ...],
        payload: bytes,
        ) -> dict[str, object]:
    """构造一个 failure profile 文件承诺。"""
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
        audit_dir: Path,
        expected_audit_manifest_sha256: str,
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]],
                   dict[str, bytes]]:
    """严格回读 TRAIN/audit manifest 后重派生唯一 failure profile。"""
    protocol_values = read_normalization_recovery_v5_learner_input(
        protocol_dir,
        expected_manifest_sha256=expected_protocol_manifest_sha256,
    )
    protocol_manifest, observations, fragments, _groups, _work = protocol_values
    audit_manifest = _read_audit_manifest_only(
        audit_dir,
        expected_manifest_sha256=expected_audit_manifest_sha256,
    )
    wrong_cases, impacts, family_summaries, summary = (
        derive_normalization_recovery_v5_failure_profile(
            protocol_manifest=protocol_manifest,
            observations=observations,
            fragments=fragments,
            audit_manifest_sha256=expected_audit_manifest_sha256,
            audit_manifest=audit_manifest,
        ))
    outputs = {
        "wrong-cases.jsonl": wrong_cases,
        "rule-impacts.jsonl": impacts,
        "family-summaries.jsonl": family_summaries,
    }
    payloads = {name: _payload(outputs[name])
                for name, _role, _identity in PROFILE_FILES}
    files = [_artifact(
        name=name, role=role, values=outputs[name], payload=payloads[name])
        for name, role, _identity in PROFILE_FILES]
    manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_KIND,
        "candidate_pack_read_count": 0,
        "evaluation_commitment_read_count": 0,
        "evaluation_payload_read_count": 0,
        "files": files,
        "formal_run_count": 0,
        "format_version": 1,
        "mastery_claimed": 0,
        "prior_formal_item_read_count": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": expected_protocol_manifest_sha256,
        "reserve_identity_read_count": 0,
        "reserve_payload_read_count": 0,
        "selection_or_threshold_changed": 0,
        "status": NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "training_audit_manifest_only_read_count": 1,
        "training_audit_manifest_sha256": expected_audit_manifest_sha256,
        "training_audit_non_manifest_read_count": 0,
    }
    return manifest, outputs, payloads


def publish_normalization_recovery_v5_failure_profile(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        audit_dir: str | Path,
        expected_audit_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 TRAIN-only failure profile，并以 manifest-last 封口。"""
    root = _require_k_root(run_root)
    protocol_root = _within(root, protocol_dir, label="protocol_dir")
    audit_root = _within(root, audit_dir, label="audit_dir")
    target = _within(root, target_dir, label="target_dir")
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256,
        label="v5 profile protocol manifest")
    audit_sha = _sha_value(
        expected_audit_manifest_sha256,
        label="v5 profile audit manifest")
    if (not protocol_root.is_dir() or not audit_root.is_dir()
            or target.exists()
            or _overlap(protocol_root, audit_root)
            or _overlap(protocol_root, target)
            or _overlap(audit_root, target)):
        raise BroadQaExternalDataError(
            "v5 failure profile 输入缺失、artifact 混淆或 target 已存在")
    manifest, outputs, payloads = _derive(
        protocol_dir=protocol_root,
        expected_protocol_manifest_sha256=protocol_sha,
        audit_dir=audit_root,
        expected_audit_manifest_sha256=audit_sha,
    )
    target.mkdir(parents=True)
    for name, _role, identity_key in PROFILE_FILES:
        values = outputs[name]
        identities = [str(item[identity_key]) for item in values]
        if len(set(identities)) != len(identities):
            raise BroadQaExternalDataError(
                f"v5 failure profile {name} identity 重复")
        with (target / name).open("xb") as handle:
            handle.write(payloads[name])
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(
        manifest_path.read_bytes())}


def read_normalization_recovery_v5_failure_profile(
        profile_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        audit_dir: str | Path,
        expected_audit_manifest_sha256: str,
        expected_profile_manifest_sha256: str,
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]]]:
    """以三个外部 SHA 重派生并严格回读完整 failure profile。"""
    root = Path(profile_dir).resolve()
    protocol_root = Path(protocol_dir).resolve()
    audit_root = Path(audit_dir).resolve()
    roots = (root, protocol_root, audit_root)
    if any(_overlap(left, right)
           for index, left in enumerate(roots)
           for right in roots[index + 1:]):
        raise BroadQaExternalDataError(
            "v5 failure profile artifact 根混淆")
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256,
        label="v5 profile expected protocol manifest")
    audit_sha = _sha_value(
        expected_audit_manifest_sha256,
        label="v5 profile expected audit manifest")
    profile_sha = _sha_value(
        expected_profile_manifest_sha256,
        label="v5 profile expected manifest")
    expected, outputs, payloads = _derive(
        protocol_dir=protocol_root,
        expected_protocol_manifest_sha256=protocol_sha,
        audit_dir=audit_root,
        expected_audit_manifest_sha256=audit_sha,
    )
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v5 failure profile manifest 不可读") from error
    if (_sha256(encoded) != profile_sha or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or not _strict_equal(stored, expected)):
        raise BroadQaExternalDataError(
            "v5 failure profile manifest identity/encoding/material 漂移")
    for name, _role, _identity in PROFILE_FILES:
        try:
            payload = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v5 failure profile {name} 不可读") from error
        if payload != payloads[name]:
            raise BroadQaExternalDataError(
                f"v5 failure profile {name} 与 TRAIN 重派生漂移")
    return ({**stored, "manifest_sha256": profile_sha}, outputs)


__all__ = [
    "NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_KIND",
    "NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_STATUS",
    "PROFILE_FILES",
    "publish_normalization_recovery_v5_failure_profile",
    "read_normalization_recovery_v5_failure_profile",
]
