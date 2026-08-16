"""不可覆盖发布并严格回读 recovery-v9 标签盲candidate pack。"""
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_candidate import (
    V8_CANDIDATE_RULE_COUNTS,
    derive_normalization_recovery_v8_candidate_preflight,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_evaluation_commitment import (
    read_normalization_recovery_v9_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_runtime_gate import (
    V8_CANDIDATE_PACK_MANIFEST_SHA256,
    V8_CANDIDATE_PROGRAM_FILE_SHA256,
    V8_CANDIDATE_PROGRAM_SHA256,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V9_CANDIDATE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_CANDIDATE_PACK_V1")
NORMALIZATION_RECOVERY_V9_CANDIDATE_PACK_STATUS = (
    "LABEL_BLIND_PREFLIGHT_PASS_GIMP_FORMAL_NOT_RUN")

V8_CANDIDATE_PREFLIGHT_FILE_SHA256 = (
    "7b45c1f3c479b44c06ad0fcabcb9f93d31ede2b36e412da7c3fd5665522111c2")
V9_EVALUATION_COMMITMENT_MANIFEST_SHA256 = (
    "65523888f7a89980c94fa087aa1316cbb67555f94432136ea68fc76b20274d12")

_FILES = (
    ("candidate-program.json", "FROZEN_FULL_TRAIN_CANDIDATE"),
    ("preflight.json", "LABEL_BLIND_DUAL_RUNTIME_PREFLIGHT"),
)
_BASE_NAMES = {"candidate-program.json", "manifest.json", "preflight.json"}


def _sha256(payload: bytes) -> str:
    """返回candidate文件、program或manifest的SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(path: Path, *, expected_sha256: str,
                    label: str) -> dict[str, object]:
    """读取规范单行JSON并核对固定SHA。"""
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v9 candidate pack {label}不可读") from error
    if (_sha256(encoded) != expected_sha256 or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            f"v9 candidate pack {label} identity漂移")
    return stored


def _base_candidate(
        base_candidate_dir: Path,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """严格读取旧full-train candidate与label-blind preflight。"""
    if (not base_candidate_dir.is_dir()
            or {item.name for item in base_candidate_dir.iterdir()}
            != _BASE_NAMES):
        raise BroadQaExternalDataError("v9 candidate base root漂移")
    manifest = _canonical_json(
        base_candidate_dir / "manifest.json",
        expected_sha256=V8_CANDIDATE_PACK_MANIFEST_SHA256,
        label="base manifest")
    candidate = _canonical_json(
        base_candidate_dir / "candidate-program.json",
        expected_sha256=V8_CANDIDATE_PROGRAM_FILE_SHA256,
        label="base candidate")
    preflight = _canonical_json(
        base_candidate_dir / "preflight.json",
        expected_sha256=V8_CANDIDATE_PREFLIGHT_FILE_SHA256,
        label="base preflight")
    files = manifest.get("files")
    by_name = {str(item.get("relative_path")): item for item in files
               if isinstance(item, dict)} if isinstance(files, list) else {}
    derived = derive_normalization_recovery_v8_candidate_preflight(candidate)
    if (manifest.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_CANDIDATE_PACK_V1"
            or manifest.get("status")
            != "LABEL_BLIND_PREFLIGHT_PASS_FORMAL_NOT_RUN"
            or manifest.get("candidate_program_sha256")
            != V8_CANDIDATE_PROGRAM_SHA256
            or manifest.get("production_enabled") != 0
            or manifest.get("mastery_claimed") != 0
            or manifest.get("vlc_final_read_count") != 0
            or by_name.get("candidate-program.json", {}).get("sha256")
            != V8_CANDIDATE_PROGRAM_FILE_SHA256
            or by_name.get("preflight.json", {}).get("sha256")
            != V8_CANDIDATE_PREFLIGHT_FILE_SHA256
            or candidate.get("candidate_program_sha256")
            != V8_CANDIDATE_PROGRAM_SHA256
            or candidate.get("inventories") is None
            or candidate.get("production_enabled") != 0
            or candidate.get("mastery_claimed") != 0
            or preflight != derived
            or preflight.get("failure_count") != 0
            or preflight.get("indexed_reference_mismatch_count") != 0
            or preflight.get("unknown_case_count") != 1):
        raise BroadQaExternalDataError("v9 candidate base state漂移")
    return manifest, candidate, preflight


def rebind_normalization_recovery_v9_candidate(
        base_candidate: dict[str, object], *,
        evaluation_commitment_manifest_sha256: str,
        ) -> dict[str, object]:
    """只替换正式commitment绑定，并重算candidate program identity。"""
    if (not isinstance(evaluation_commitment_manifest_sha256, str)
            or len(evaluation_commitment_manifest_sha256) != 64
            or any(char not in "0123456789abcdef"
                   for char in evaluation_commitment_manifest_sha256)
            or base_candidate.get("candidate_program_sha256")
            != V8_CANDIDATE_PROGRAM_SHA256
            or base_candidate.get("production_enabled") != 0
            or base_candidate.get("mastery_claimed") != 0):
        raise BroadQaExternalDataError("v9 candidate rebind输入漂移")
    program = {key: value for key, value in base_candidate.items()
               if key != "candidate_program_sha256"}
    program["evaluation_commitment_manifest_sha256"] = (
        evaluation_commitment_manifest_sha256)
    return {**program, "candidate_program_sha256": _sha256(
        canonical_json_bytes(program))}


def _artifact(name: str, role: str, payload: bytes) -> dict[str, object]:
    """构造一个candidate文件承诺。"""
    return {
        "bytes": len(payload),
        "relative_path": name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _derive(
        *, base_candidate_dir: Path,
        commitment_dir: Path,
        source_pack_dir: Path,
        runtime_gate_dir: Path,
        ) -> tuple[dict[str, object], dict[str, object],
                   dict[str, object], dict[str, bytes]]:
    """严格读取旧candidate与v9 commitment后形成新candidate pack。"""
    base_manifest, base_candidate, base_preflight = _base_candidate(
        base_candidate_dir)
    commitment = read_normalization_recovery_v9_evaluation_commitment(
        commitment_dir,
        source_pack_dir=source_pack_dir,
        runtime_gate_dir=runtime_gate_dir,
        expected_manifest_sha256=V9_EVALUATION_COMMITMENT_MANIFEST_SHA256)
    candidate = rebind_normalization_recovery_v9_candidate(
        base_candidate,
        evaluation_commitment_manifest_sha256=commitment["manifest_sha256"])
    preflight = derive_normalization_recovery_v8_candidate_preflight(candidate)
    if (preflight.get("case_count") != base_preflight.get("case_count")
            or preflight.get("failure_count") != 0
            or preflight.get("indexed_reference_mismatch_count") != 0
            or preflight.get("unknown_case_count") != 1
            or preflight.get("rule_counts") != V8_CANDIDATE_RULE_COUNTS
            or candidate.get("rule_pack_manifest_sha256")
            != base_candidate.get("rule_pack_manifest_sha256")):
        raise BroadQaExternalDataError("v9 candidate preflight未闭合")
    payloads = {
        "candidate-program.json": canonical_json_line(candidate),
        "preflight.json": canonical_json_line(preflight),
    }
    manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V9_CANDIDATE_PACK_KIND,
        "base_candidate_manifest_read_count": 1,
        "base_candidate_manifest_sha256": base_manifest["manifest_sha256"]
            if "manifest_sha256" in base_manifest
            else V8_CANDIDATE_PACK_MANIFEST_SHA256,
        "base_candidate_program_sha256": V8_CANDIDATE_PROGRAM_SHA256,
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "evaluation_commitment_manifest_sha256": commitment[
            "manifest_sha256"],
        "evaluation_or_held_out_payload_read_count": 0,
        "files": [_artifact(name, role, payloads[name])
                  for name, role in _FILES],
        "format_version": 1,
        "gimp_identity_raw_or_translation_read_count": 0,
        "mastery_claimed": 0,
        "preflight_case_count": preflight["case_count"],
        "preflight_failure_count": preflight["failure_count"],
        "production_enabled": 0,
        "rule_counts": V8_CANDIDATE_RULE_COUNTS,
        "rule_pack_manifest_sha256": candidate["rule_pack_manifest_sha256"],
        "status": NORMALIZATION_RECOVERY_V9_CANDIDATE_PACK_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_audit_manifest_sha256": candidate[
            "training_audit_manifest_sha256"],
    }
    return manifest, candidate, preflight, payloads


def _require_k_root(value: str | Path) -> Path:
    """要求显式candidate工作根位于已存在K盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v9 candidate run root必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析输入输出并限制其仍位于K盘run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"v9 candidate {label}越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个artifact根是否相同或互为祖先。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def publish_normalization_recovery_v9_candidate_pack(
        *, run_root: str | Path,
        base_candidate_dir: str | Path,
        evaluation_commitment_dir: str | Path,
        source_pack_dir: str | Path,
        runtime_gate_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布不读取GIMP payload的v9 candidate pack。"""
    root = _require_k_root(run_root)
    inputs = tuple(_within(root, value, label=str(index)) for index, value in
                   enumerate((base_candidate_dir, evaluation_commitment_dir,
                              source_pack_dir, runtime_gate_dir)))
    target = _within(root, target_dir, label="target")
    paths = (*inputs, target)
    if (target.exists() or any(not path.is_dir() for path in inputs)
            or any(_overlap(left, right)
                   for index, left in enumerate(paths)
                   for right in paths[index + 1:])):
        raise BroadQaExternalDataError("v9 candidate path非法")
    manifest, _candidate, _preflight, payloads = _derive(
        base_candidate_dir=inputs[0], commitment_dir=inputs[1],
        source_pack_dir=inputs[2], runtime_gate_dir=inputs[3])
    target.mkdir()
    for name, _role in _FILES:
        with (target / name).open("xb") as handle:
            handle.write(payloads[name])
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v9_candidate_pack(
        source_dir: str | Path, *,
        base_candidate_dir: str | Path,
        evaluation_commitment_dir: str | Path,
        source_pack_dir: str | Path,
        runtime_gate_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """重绑旧candidate并严格回读v9 candidate pack。"""
    root = Path(source_dir).resolve()
    expected, candidate, preflight, payloads = _derive(
        base_candidate_dir=Path(base_candidate_dir).resolve(),
        commitment_dir=Path(evaluation_commitment_dir).resolve(),
        source_pack_dir=Path(source_pack_dir).resolve(),
        runtime_gate_dir=Path(runtime_gate_dir).resolve())
    try:
        physical = {item.name for item in root.iterdir()}
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v9 candidate pack不可读") from error
    if (physical != {"manifest.json", *[name for name, _role in _FILES]}
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or not strict_json_equal(stored, expected)):
        raise BroadQaExternalDataError("v9 candidate manifest漂移")
    for name, _role in _FILES:
        try:
            payload = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v9 candidate {name}不可读") from error
        if payload != payloads[name]:
            raise BroadQaExternalDataError(f"v9 candidate {name}重派生漂移")
    return ({**stored, "manifest_sha256": expected_manifest_sha256},
            candidate, preflight)


__all__ = [
    "NORMALIZATION_RECOVERY_V9_CANDIDATE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V9_CANDIDATE_PACK_STATUS",
    "V9_EVALUATION_COMMITMENT_MANIFEST_SHA256",
    "publish_normalization_recovery_v9_candidate_pack",
    "read_normalization_recovery_v9_candidate_pack",
    "rebind_normalization_recovery_v9_candidate",
]
