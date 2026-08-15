"""Publish and strictly read the recovery-v8 TRAIN-only LOSO audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_rule_pack import (
    read_normalization_recovery_v8_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_audit_records import (
    V8_TRAIN_AUDIT_FILES,
    derive_normalization_recovery_v8_training_audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_protocol import (
    read_normalization_recovery_v8_learner_input,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V8_TRAINING_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_TRAINING_AUDIT_V1")
NORMALIZATION_RECOVERY_V8_TRAINING_AUDIT_STATUS = (
    "TRAIN_ONLY_FAMILY_LOSO_PASS_NOT_FORMAL_NOT_DEPLOYED")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise BroadQaExternalDataError(f"v8 audit {label} 非法")
    return value


def _strict_equal(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, list):
        return (len(value) == len(expected)
                and all(_strict_equal(left, right)
                        for left, right in zip(value, expected)))
    return value == expected


def _require_k_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v8 training audit run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"v8 training audit {label} 越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _payload(values: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(canonical_json_line(item) for item in values)


def _artifact(
        *, name: str, role: str, values: tuple[dict[str, object], ...],
        payload: bytes,
        ) -> dict[str, object]:
    return {
        "bytes": len(payload),
        "record_count": len(values),
        "relative_path": name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _manifest(
        *, protocol_sha: str, pack_sha: str,
        files: list[dict[str, object]], summary: dict[str, object],
        ) -> dict[str, object]:
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_TRAINING_AUDIT_KIND,
        "candidate_pack_read_count": 0,
        "evaluation_or_held_out_payload_read_count": 0,
        "evaluation_payload_read_count": 0,
        "files": files,
        "format_version": 1,
        "mastery_claimed": 0,
        "predecessor_rule_pack_read_count": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": protocol_sha,
        "reserve_payload_read_count": 0,
        "rule_pack_manifest_sha256": pack_sha,
        "rule_pack_read_count": 1,
        "source_pack_read_count": 0,
        "status": NORMALIZATION_RECOVERY_V8_TRAINING_AUDIT_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "training_protocol_read_count": 1,
        "vlc_final_read_count": 0,
    }


def _derive(
        *, protocol_dir: Path, protocol_sha: str,
        pack_dir: Path, pack_sha: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]],
            dict[str, bytes]]:
    protocol_manifest, protocol_outputs = (
        read_normalization_recovery_v8_learner_input(
            protocol_dir, expected_manifest_sha256=protocol_sha))
    if protocol_manifest["manifest_sha256"] != protocol_sha:
        raise BroadQaExternalDataError("v8 audit protocol identity 漂移")
    pack_manifest, rule_outputs = read_normalization_recovery_v8_rule_pack(
        pack_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        expected_pack_manifest_sha256=pack_sha,
    )
    if pack_manifest["manifest_sha256"] != pack_sha:
        raise BroadQaExternalDataError("v8 audit rule pack identity 漂移")
    outputs, summary = derive_normalization_recovery_v8_training_audit(
        protocol_outputs=protocol_outputs, rule_outputs=rule_outputs)
    if summary.get("hard_gates_pass") != 1:
        raise BroadQaExternalDataError("v8 audit TRAIN-only hard gate 未通过")
    payloads = {name: _payload(outputs[name])
                for name, _role, _identity in V8_TRAIN_AUDIT_FILES}
    files = [_artifact(
        name=name, role=role, values=outputs[name], payload=payloads[name])
        for name, role, _identity in V8_TRAIN_AUDIT_FILES]
    return _manifest(
        protocol_sha=protocol_sha, pack_sha=pack_sha,
        files=files, summary=summary), outputs, payloads


def publish_normalization_recovery_v8_training_audit(
        *, run_root: str | Path, protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        pack_dir: str | Path, expected_pack_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """Publish the PASS audit once after all frozen hard gates succeed."""
    root = _require_k_root(run_root)
    protocol_root = _within(root, protocol_dir, label="protocol_dir")
    pack_root = _within(root, pack_dir, label="pack_dir")
    target = _within(root, target_dir, label="target_dir")
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256, label="protocol SHA")
    pack_sha = _sha_value(expected_pack_manifest_sha256, label="pack SHA")
    if (target.exists() or not protocol_root.is_dir() or not pack_root.is_dir()
            or _overlap(protocol_root, pack_root)
            or _overlap(target, protocol_root) or _overlap(target, pack_root)):
        raise BroadQaExternalDataError("v8 training audit path 非法")
    manifest, outputs, payloads = _derive(
        protocol_dir=protocol_root, protocol_sha=protocol_sha,
        pack_dir=pack_root, pack_sha=pack_sha)
    target.mkdir()
    for name, _role, _identity in V8_TRAIN_AUDIT_FILES:
        with (target / name).open("xb") as handle:
            handle.write(payloads[name])
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def read_normalization_recovery_v8_training_audit(
        audit_dir: str | Path, *, protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        pack_dir: str | Path, expected_pack_manifest_sha256: str,
        expected_audit_manifest_sha256: str,
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]]]:
    """Strictly rederive the audit from protocol and disabled rule pack."""
    root = Path(audit_dir).resolve()
    protocol_root = Path(protocol_dir).resolve()
    pack_root = Path(pack_dir).resolve()
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256, label="protocol SHA")
    pack_sha = _sha_value(expected_pack_manifest_sha256, label="pack SHA")
    audit_sha = _sha_value(expected_audit_manifest_sha256, label="audit SHA")
    expected, outputs, payloads = _derive(
        protocol_dir=protocol_root, protocol_sha=protocol_sha,
        pack_dir=pack_root, pack_sha=pack_sha)
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v8 training audit manifest 不可读") from error
    if (_sha256(encoded) != audit_sha or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or not _strict_equal(stored, expected)):
        raise BroadQaExternalDataError("v8 training audit manifest 漂移")
    for name, _role, _identity in V8_TRAIN_AUDIT_FILES:
        try:
            payload = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v8 training audit {name} 不可读") from error
        if payload != payloads[name]:
            raise BroadQaExternalDataError(
                f"v8 training audit {name} 重派生漂移")
    return {**stored, "manifest_sha256": audit_sha}, outputs


__all__ = [
    "NORMALIZATION_RECOVERY_V8_TRAINING_AUDIT_KIND",
    "NORMALIZATION_RECOVERY_V8_TRAINING_AUDIT_STATUS",
    "publish_normalization_recovery_v8_training_audit",
    "read_normalization_recovery_v8_training_audit",
]
