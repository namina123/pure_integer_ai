"""不可覆盖发布并严格回读 recovery-v8 标签盲 candidate pack。"""
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
    compile_normalization_recovery_v8_candidate,
    derive_normalization_recovery_v8_candidate_preflight,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_evaluation_commitment import (
    read_normalization_recovery_v8_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_rule_pack import (
    read_normalization_recovery_v8_rule_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V8_CANDIDATE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_CANDIDATE_PACK_V1")
NORMALIZATION_RECOVERY_V8_CANDIDATE_PACK_STATUS = (
    "LABEL_BLIND_PREFLIGHT_PASS_FORMAL_NOT_RUN")

_FILES = (
    ("candidate-program.json", "FROZEN_FULL_TRAIN_CANDIDATE"),
    ("preflight.json", "LABEL_BLIND_DUAL_RUNTIME_PREFLIGHT"),
)


def _sha256(payload: bytes) -> str:
    """返回 candidate 文件或 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise BroadQaExternalDataError(f"v8 candidate pack {label} 非法")
    return value


def _require_k_root(value: str | Path) -> Path:
    """要求显式 candidate 工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v8 candidate pack run root 必须在 K 盘")
    return root


def _overlap(left: Path, right: Path) -> bool:
    """判断两个 artifact 根是否相同或存在包含关系。"""
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _read_audit_manifest_only(
        audit_dir: Path, *, expected_sha256: str,
        expected_protocol_sha256: str, expected_pack_sha256: str,
        ) -> dict[str, object]:
    """只读 TRAIN audit manifest，并要求所有硬门已通过。"""
    audit_sha = _sha_value(expected_sha256, label="audit SHA")
    try:
        encoded = (audit_dir / "manifest.json").read_bytes()
        manifest = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v8 candidate audit manifest 不可读") from error
    summary = manifest.get("summary") if isinstance(manifest, dict) else None
    if (_sha256(encoded) != audit_sha or not isinstance(manifest, dict)
            or canonical_json_line(manifest) != encoded
            or manifest.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_TRAINING_AUDIT_V1"
            or manifest.get("status")
            != "TRAIN_ONLY_FAMILY_LOSO_PASS_NOT_FORMAL_NOT_DEPLOYED"
            or manifest.get("protocol_manifest_sha256")
            != expected_protocol_sha256
            or manifest.get("rule_pack_manifest_sha256")
            != expected_pack_sha256
            or manifest.get("vlc_final_read_count") != 0
            or manifest.get("production_enabled") != 0
            or manifest.get("mastery_claimed") != 0
            or not isinstance(summary, dict)
            or summary.get("hard_gates_pass") != 1
            or summary.get("wrong_count") != 0
            or summary.get("indexed_reference_mismatch_count") != 0
            or summary.get("partial_commit_count") != 0
            or summary.get("structure_mismatch_count") != 0
            or summary.get("exception_count") != 0
            or summary.get("generation_hard_conjunct_pass") != 1):
        raise BroadQaExternalDataError("v8 candidate TRAIN audit 漂移")
    return {**manifest, "manifest_sha256": audit_sha}


def _artifact(name: str, role: str, payload: bytes) -> dict[str, object]:
    """构造一个 candidate 文件承诺。"""
    return {
        "bytes": len(payload),
        "relative_path": name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _derive(
        *, protocol_dir: Path, protocol_sha: str,
        pack_dir: Path, pack_sha: str,
        audit_dir: Path, audit_sha: str,
        v7_commitment_dir: Path, v7_commitment_sha: str,
        commitment_dir: Path, commitment_sha: str,
        ) -> tuple[dict[str, object], dict[str, object],
                   dict[str, object], dict[str, bytes]]:
    """严格重读 TRAIN pack、audit 与 commitment 后编译候选。"""
    pack_manifest, outputs = read_normalization_recovery_v8_rule_pack(
        pack_dir, protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        expected_pack_manifest_sha256=pack_sha)
    audit = _read_audit_manifest_only(
        audit_dir, expected_sha256=audit_sha,
        expected_protocol_sha256=protocol_sha,
        expected_pack_sha256=pack_sha)
    commitment = read_normalization_recovery_v8_evaluation_commitment(
        commitment_dir, v7_commitment_dir=v7_commitment_dir,
        expected_v7_commitment_manifest_sha256=v7_commitment_sha,
        expected_manifest_sha256=commitment_sha)
    candidate = compile_normalization_recovery_v8_candidate(
        rule_pack_manifest=pack_manifest, rule_outputs=outputs,
        training_audit_manifest_sha256=audit["manifest_sha256"],
        evaluation_commitment_manifest_sha256=commitment["manifest_sha256"])
    preflight = derive_normalization_recovery_v8_candidate_preflight(candidate)
    if (preflight.get("failure_count") != 0
            or preflight.get("indexed_reference_mismatch_count") != 0
            or preflight.get("unknown_case_count") != 1):
        raise BroadQaExternalDataError("v8 candidate preflight 未闭合")
    payloads = {
        "candidate-program.json": canonical_json_line(candidate),
        "preflight.json": canonical_json_line(preflight),
    }
    manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_CANDIDATE_PACK_KIND,
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "evaluation_commitment_manifest_sha256": commitment_sha,
        "evaluation_or_held_out_payload_read_count": 0,
        "files": [_artifact(name, role, payloads[name])
                  for name, role in _FILES],
        "format_version": 1,
        "mastery_claimed": 0,
        "preflight_case_count": preflight["case_count"],
        "preflight_failure_count": preflight["failure_count"],
        "production_enabled": 0,
        "rule_pack_manifest_sha256": pack_sha,
        "status": NORMALIZATION_RECOVERY_V8_CANDIDATE_PACK_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_audit_manifest_sha256": audit_sha,
        "training_protocol_manifest_sha256": protocol_sha,
        "v7_commitment_manifest_sha256": v7_commitment_sha,
        "vlc_final_read_count": 0,
    }
    return manifest, candidate, preflight, payloads


def publish_normalization_recovery_v8_candidate_pack(
        *, run_root: str | Path, protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        pack_dir: str | Path, expected_pack_manifest_sha256: str,
        audit_dir: str | Path, expected_audit_manifest_sha256: str,
        v7_commitment_dir: str | Path,
        expected_v7_commitment_manifest_sha256: str,
        evaluation_commitment_dir: str | Path,
        expected_evaluation_commitment_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布不读取 VLC payload 的正式 candidate pack。"""
    root = _require_k_root(run_root)
    paths = tuple(Path(value).resolve() for value in (
        protocol_dir, pack_dir, audit_dir, v7_commitment_dir,
        evaluation_commitment_dir, target_dir))
    if (any(not path.is_relative_to(root) for path in paths)
            or any(not path.is_dir() for path in paths[:-1])
            or paths[-1].exists()
            or any(_overlap(left, right)
                   for index, left in enumerate(paths)
                   for right in paths[index + 1:])):
        raise BroadQaExternalDataError("v8 candidate pack path 非法")
    shas = tuple(_sha_value(value, label="input SHA") for value in (
        expected_protocol_manifest_sha256,
        expected_pack_manifest_sha256,
        expected_audit_manifest_sha256,
        expected_v7_commitment_manifest_sha256,
        expected_evaluation_commitment_manifest_sha256))
    manifest, _candidate, _preflight, payloads = _derive(
        protocol_dir=paths[0], protocol_sha=shas[0],
        pack_dir=paths[1], pack_sha=shas[1],
        audit_dir=paths[2], audit_sha=shas[2],
        v7_commitment_dir=paths[3], v7_commitment_sha=shas[3],
        commitment_dir=paths[4], commitment_sha=shas[4])
    target = paths[-1]
    target.mkdir()
    for name, _role in _FILES:
        with (target / name).open("xb") as handle:
            handle.write(payloads[name])
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v8_candidate_pack(
        candidate_dir: str | Path, *, protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        pack_dir: str | Path, expected_pack_manifest_sha256: str,
        audit_dir: str | Path, expected_audit_manifest_sha256: str,
        v7_commitment_dir: str | Path,
        expected_v7_commitment_manifest_sha256: str,
        evaluation_commitment_dir: str | Path,
        expected_evaluation_commitment_manifest_sha256: str,
        expected_candidate_manifest_sha256: str,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """从五个冻结输入重编译并严格回读 candidate pack。"""
    roots = tuple(Path(value).resolve() for value in (
        candidate_dir, protocol_dir, pack_dir, audit_dir,
        v7_commitment_dir, evaluation_commitment_dir))
    if any(_overlap(left, right) for index, left in enumerate(roots)
           for right in roots[index + 1:]):
        raise BroadQaExternalDataError("v8 candidate pack artifact 根混淆")
    shas = tuple(_sha_value(value, label="expected SHA") for value in (
        expected_protocol_manifest_sha256,
        expected_pack_manifest_sha256,
        expected_audit_manifest_sha256,
        expected_v7_commitment_manifest_sha256,
        expected_evaluation_commitment_manifest_sha256,
        expected_candidate_manifest_sha256))
    expected, candidate, preflight, payloads = _derive(
        protocol_dir=roots[1], protocol_sha=shas[0],
        pack_dir=roots[2], pack_sha=shas[1],
        audit_dir=roots[3], audit_sha=shas[2],
        v7_commitment_dir=roots[4], v7_commitment_sha=shas[3],
        commitment_dir=roots[5], commitment_sha=shas[4])
    try:
        encoded = (roots[0] / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v8 candidate pack manifest 不可读") from error
    if (_sha256(encoded) != shas[5] or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or not strict_json_equal(stored, expected)):
        raise BroadQaExternalDataError("v8 candidate pack manifest 漂移")
    for name, _role in _FILES:
        try:
            payload = (roots[0] / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v8 candidate pack {name} 不可读") from error
        if payload != payloads[name]:
            raise BroadQaExternalDataError(
                f"v8 candidate pack {name} 重派生漂移")
    return {**stored, "manifest_sha256": shas[5]}, candidate, preflight


__all__ = [
    "NORMALIZATION_RECOVERY_V8_CANDIDATE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V8_CANDIDATE_PACK_STATUS",
    "publish_normalization_recovery_v8_candidate_pack",
    "read_normalization_recovery_v8_candidate_pack",
]
