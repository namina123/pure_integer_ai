"""发布并严格回读 recovery-v6 strong-whole disabled rule pack。"""
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_learning_contract import (
    NORMALIZATION_RECOVERY_V6_OUTPUT_FILE_ROLES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_learning_records import (
    derive_normalization_recovery_v6_learning_outputs,
    normalization_recovery_v6_output_payloads,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V6_RULE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V6_RULE_PACK_V1")
NORMALIZATION_RECOVERY_V6_RULE_PACK_STATUS = (
    "FROZEN_POLICY_PROJECTED_NOT_EVALUATED_NOT_DEPLOYED")


def _sha256(payload: bytes) -> str:
    """返回 artifact、文件或语义结果 SHA-256。"""
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
    """要求显式 v6 pack 工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v6 rule pack root 必须是 K 盘目录")
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


def _artifact(
        *,
        name: str,
        role: str,
        values: tuple[dict[str, object], ...],
        payload: bytes,
        ) -> dict[str, object]:
    """构造一个 v6 pack 文件承诺。"""
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
        predecessor_pack_dir: Path,
        expected_predecessor_pack_manifest_sha256: str,
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]],
                   dict[str, bytes]]:
    """严格回读 v5 pack 后重派生唯一 v6 projection pack。"""
    predecessor_manifest, predecessor_outputs = (
        read_normalization_recovery_v5_rule_pack(
            predecessor_pack_dir,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=(
                expected_protocol_manifest_sha256),
            expected_pack_manifest_sha256=(
                expected_predecessor_pack_manifest_sha256),
        ))
    if (predecessor_manifest.get("fresh_resume_output_bytes_equal") != 1
            or predecessor_manifest.get("production_enabled") != 0
            or predecessor_manifest.get("mastery_claimed") != 0
            or predecessor_manifest.get("protocol_manifest_sha256")
            != expected_protocol_manifest_sha256):
        raise BroadQaExternalDataError(
            "v6 predecessor pack lineage/status 漂移")
    outputs, summary = derive_normalization_recovery_v6_learning_outputs(
        protocol_manifest_sha256=expected_protocol_manifest_sha256,
        predecessor_pack_manifest_sha256=(
            expected_predecessor_pack_manifest_sha256),
        predecessor_outputs=predecessor_outputs,
    )
    payloads = normalization_recovery_v6_output_payloads(outputs)
    files = [_artifact(
        name=name, role=role, values=outputs[name], payload=payloads[name])
        for name, role, _identity in NORMALIZATION_RECOVERY_V6_OUTPUT_FILE_ROLES]
    semantic_result_sha = _sha256(canonical_json_bytes({
        "files": files,
        "predecessor_rule_pack_manifest_sha256": (
            expected_predecessor_pack_manifest_sha256),
        "protocol_manifest_sha256": expected_protocol_manifest_sha256,
        "summary": summary,
    }))
    manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V6_RULE_PACK_KIND,
        "base_rule_pack_read_count": 0,
        "candidate_pack_read_count": 0,
        "evaluation_commitment_read_count": 0,
        "evaluation_payload_read_count": 0,
        "files": files,
        "formal_run_count": 0,
        "format_version": 1,
        "mastery_claimed": 0,
        "predecessor_fresh_resume_output_bytes_equal": 1,
        "predecessor_learner_lineages": predecessor_manifest[
            "learner_lineages"],
        "predecessor_rule_pack_manifest_sha256": (
            expected_predecessor_pack_manifest_sha256),
        "predecessor_rule_pack_read_count": 1,
        "prior_formal_item_read_count": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": expected_protocol_manifest_sha256,
        "reserve_identity_read_count": 0,
        "reserve_payload_read_count": 0,
        "runtime_state": "LEARNED_PACK_DISABLED",
        "semantic_result_sha256": semantic_result_sha,
        "status": NORMALIZATION_RECOVERY_V6_RULE_PACK_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
    }
    return manifest, outputs, payloads


def publish_normalization_recovery_v6_rule_pack(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        predecessor_pack_dir: str | Path,
        expected_predecessor_pack_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 v6 strong-whole pack，并以 manifest-last 封口。"""
    root = _require_k_root(run_root)
    protocol_root = _within(root, protocol_dir, label="protocol_dir")
    predecessor_root = _within(
        root, predecessor_pack_dir, label="predecessor_pack_dir")
    target = _within(root, target_dir, label="target_dir")
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256, label="v6 pack protocol manifest")
    predecessor_sha = _sha_value(
        expected_predecessor_pack_manifest_sha256,
        label="v6 predecessor pack manifest")
    roots = (protocol_root, predecessor_root, target)
    if (not protocol_root.is_dir() or not predecessor_root.is_dir()
            or target.exists()
            or any(_overlap(left, right)
                   for index, left in enumerate(roots)
                   for right in roots[index + 1:])):
        raise BroadQaExternalDataError(
            "v6 rule pack 输入缺失、artifact 混淆或 target 已存在")
    manifest, outputs, payloads = _derive(
        protocol_dir=protocol_root,
        expected_protocol_manifest_sha256=protocol_sha,
        predecessor_pack_dir=predecessor_root,
        expected_predecessor_pack_manifest_sha256=predecessor_sha,
    )
    target.mkdir(parents=True)
    for name, _role, identity_key in NORMALIZATION_RECOVERY_V6_OUTPUT_FILE_ROLES:
        values = outputs[name]
        identities = [str(item[identity_key]) for item in values]
        if len(set(identities)) != len(identities):
            raise BroadQaExternalDataError(
                f"v6 rule pack {name} identity 重复")
        with (target / name).open("xb") as handle:
            handle.write(payloads[name])
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(
        manifest_path.read_bytes())}


def read_normalization_recovery_v6_rule_pack(
        pack_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        predecessor_pack_dir: str | Path,
        expected_predecessor_pack_manifest_sha256: str,
        expected_pack_manifest_sha256: str,
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]]]:
    """以三外部 SHA 重派生并严格回读完整 v6 disabled pack。"""
    root = Path(pack_dir).resolve()
    protocol_root = Path(protocol_dir).resolve()
    predecessor_root = Path(predecessor_pack_dir).resolve()
    roots = (root, protocol_root, predecessor_root)
    if any(_overlap(left, right)
           for index, left in enumerate(roots)
           for right in roots[index + 1:]):
        raise BroadQaExternalDataError("v6 rule pack artifact 根混淆")
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256,
        label="v6 expected protocol manifest")
    predecessor_sha = _sha_value(
        expected_predecessor_pack_manifest_sha256,
        label="v6 expected predecessor pack manifest")
    pack_sha = _sha_value(
        expected_pack_manifest_sha256, label="v6 expected pack manifest")
    expected, outputs, payloads = _derive(
        protocol_dir=protocol_root,
        expected_protocol_manifest_sha256=protocol_sha,
        predecessor_pack_dir=predecessor_root,
        expected_predecessor_pack_manifest_sha256=predecessor_sha,
    )
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v6 rule pack manifest 不可读") from error
    if (_sha256(encoded) != pack_sha or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or not _strict_equal(stored, expected)):
        raise BroadQaExternalDataError(
            "v6 rule pack manifest identity/encoding/material 漂移")
    for name, _role, _identity in NORMALIZATION_RECOVERY_V6_OUTPUT_FILE_ROLES:
        try:
            payload = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v6 rule pack {name} 不可读") from error
        if payload != payloads[name]:
            raise BroadQaExternalDataError(
                f"v6 rule pack {name} 与 predecessor 重派生漂移")
    return ({**stored, "manifest_sha256": pack_sha}, outputs)


__all__ = [
    "NORMALIZATION_RECOVERY_V6_RULE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V6_RULE_PACK_STATUS",
    "publish_normalization_recovery_v6_rule_pack",
    "read_normalization_recovery_v6_rule_pack",
]
