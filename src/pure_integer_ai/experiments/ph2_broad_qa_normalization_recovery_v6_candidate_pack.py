"""不可覆盖发布并严格回读 recovery-v6 candidate pack。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_evaluation_commitment import (
    NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_KIND,
    NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_STATUS,
    read_normalization_recovery_v5_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_candidate import (
    compile_normalization_recovery_v6_candidate,
    derive_normalization_recovery_v6_candidate_preflight,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_phrase_runtime import (
    compile_normalization_recovery_v6_phrase_program,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_rule_pack import (
    read_normalization_recovery_v6_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_training_audit import (
    NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_KIND,
    NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_STATUS,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V6_CANDIDATE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V6_CANDIDATE_PACK_V1")
NORMALIZATION_RECOVERY_V6_CANDIDATE_PACK_STATUS = (
    "FROZEN_LABEL_BLIND_PREFLIGHT_PASS_FORMAL_NOT_RUN")

_FILES = (
    ("candidate-program.json", "FROZEN_WHOLE_ONLY_CANDIDATE"),
    ("preflight.json", "LABEL_BLIND_CANDIDATE_PREFLIGHT"),
)


def _sha256(payload: bytes) -> str:
    """返回 candidate artifact、文件或维度摘要。"""
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
    """要求显式 candidate 工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v6 candidate run root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析输入输出并拒绝逃出唯一 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个 artifact 根是否相同或存在包含关系。"""
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _artifact(name: str, role: str, payload: bytes) -> dict[str, object]:
    """构造一个 candidate 文件承诺。"""
    return {
        "bytes": len(payload),
        "relative_path": name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _read_audit_manifest(
        root: Path,
        *,
        expected_manifest_sha256: str,
        expected_protocol_manifest_sha256: str,
        expected_predecessor_pack_manifest_sha256: str,
        expected_pack_manifest_sha256: str,
        ) -> dict[str, object]:
    """只读 v6 audit manifest/file identity，并要求 facility/capability PASS。"""
    expected_sha = _sha_value(
        expected_manifest_sha256, label="v6 candidate audit manifest")
    try:
        encoded = (root / "manifest.json").read_bytes()
        manifest = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v6 candidate audit manifest 不可读") from error
    summary = manifest.get("summary") if isinstance(manifest, dict) else None
    if (_sha256(encoded) != expected_sha or not isinstance(manifest, dict)
            or canonical_json_line(manifest) != encoded
            or manifest.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_KIND
            or manifest.get("status")
            != NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_STATUS
            or manifest.get("protocol_manifest_sha256")
            != expected_protocol_manifest_sha256
            or manifest.get("predecessor_rule_pack_manifest_sha256")
            != expected_predecessor_pack_manifest_sha256
            or manifest.get("pack_manifest_sha256")
            != expected_pack_manifest_sha256
            or not isinstance(summary, dict)
            or summary.get("audit_outcome")
            != "FACILITY_PASS_CAPABILITY_PASS"
            or summary.get("capability_gate_pass") != 1
            or summary.get("facility_failure_count") != 0
            or summary.get("outcome_counts")
            != {"EXACT": 1510, "UNKNOWN": 36740, "WRONG": 0}
            or summary.get("identity_false_change_count") != 0
            or summary.get("non_identity_exact_count") != 18
            or summary.get("simulation_strategy_reference_equal") != 1
            or any(manifest.get(name) != 0 for name in (
                "candidate_pack_read_count", "evaluation_commitment_read_count",
                "evaluation_payload_read_count", "formal_run_count",
                "mastery_claimed", "prior_formal_item_read_count",
                "production_enabled", "reserve_identity_read_count",
                "reserve_payload_read_count", "teacher_api_llm_call_count"))):
        raise BroadQaExternalDataError("v6 candidate audit 冻结边界漂移")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise BroadQaExternalDataError("v6 candidate audit file roster 漂移")
    for record in files:
        if not isinstance(record, dict):
            raise BroadQaExternalDataError("v6 candidate audit file identity 非法")
        path = root / str(record.get("relative_path"))
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                "v6 candidate audit file 不可读") from error
        if (record.get("bytes") != len(payload)
                or record.get("sha256") != _sha256(payload)):
            raise BroadQaExternalDataError("v6 candidate audit file 漂移")
    return {**manifest, "manifest_sha256": expected_sha}


def _derive(
        *,
        protocol_dir: Path,
        expected_protocol_manifest_sha256: str,
        predecessor_pack_dir: Path,
        expected_predecessor_pack_manifest_sha256: str,
        pack_dir: Path,
        expected_pack_manifest_sha256: str,
        audit_dir: Path,
        expected_audit_manifest_sha256: str,
        qt_source_pack_dir: Path,
        expected_qt_source_manifest_sha256: str,
        evaluation_commitment_dir: Path,
        expected_evaluation_commitment_manifest_sha256: str,
        ) -> tuple[dict[str, object], dict[str, object],
                   dict[str, object], dict[str, bytes]]:
    """严格回读 pack/audit/commitment 后重编译 candidate 与 preflight。"""
    pack, outputs = read_normalization_recovery_v6_rule_pack(
        pack_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        predecessor_pack_dir=predecessor_pack_dir,
        expected_predecessor_pack_manifest_sha256=(
            expected_predecessor_pack_manifest_sha256),
        expected_pack_manifest_sha256=expected_pack_manifest_sha256,
    )
    audit = _read_audit_manifest(
        audit_dir,
        expected_manifest_sha256=expected_audit_manifest_sha256,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        expected_predecessor_pack_manifest_sha256=(
            expected_predecessor_pack_manifest_sha256),
        expected_pack_manifest_sha256=expected_pack_manifest_sha256,
    )
    commitment = read_normalization_recovery_v5_evaluation_commitment(
        evaluation_commitment_dir,
        qt_source_pack_dir=qt_source_pack_dir,
        expected_qt_source_manifest_sha256=(
            expected_qt_source_manifest_sha256),
        expected_manifest_sha256=(
            expected_evaluation_commitment_manifest_sha256),
    )
    if (commitment.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_KIND
            or commitment.get("status")
            != NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_STATUS
            or commitment.get("manifest_sha256")
            != expected_evaluation_commitment_manifest_sha256
            or commitment.get("production_enabled") != 0
            or commitment.get("mastery_claimed") != 0
            or commitment.get("source_non_manifest_file_read_count") != 0
            or commitment.get("training_source_read_count") != 0
            or commitment.get("denominator", {}).get("record_count") != 3531
            or commitment.get("formal_contract", {}).get(
                "candidate_applicability_cannot_shrink_denominator") != 1):
        raise BroadQaExternalDataError("v6 candidate commitment 漂移")
    phrase = compile_normalization_recovery_v6_phrase_program(
        rule_pack_manifest_sha256=pack["manifest_sha256"],
        target_whole_rules=outputs["target-whole-rules.jsonl"],
        defeaters=outputs["defeaters.jsonl"],
        identity_vetoes=outputs["identity-vetoes.jsonl"],
        conflict_vetoes=outputs["conflict-vetoes.jsonl"],
        target_index=outputs["target-index.jsonl"],
    )
    candidate = compile_normalization_recovery_v6_candidate(
        phrase_program=phrase,
        v6_training_audit_manifest_sha256=audit["manifest_sha256"],
        evaluation_commitment_manifest_sha256=commitment["manifest_sha256"],
    )
    preflight = derive_normalization_recovery_v6_candidate_preflight(candidate)
    if (preflight["failure_count"] != 0
            or preflight["indexed_reference_mismatch_count"] != 0
            or preflight["valid_scope_all_applicable"] != 1
            or preflight["invalid_scope_rejected"] != 1):
        raise BroadQaExternalDataError("v6 candidate preflight 未闭合")
    payloads = {
        "candidate-program.json": canonical_json_line(candidate),
        "preflight.json": canonical_json_line(preflight),
    }
    files = [_artifact(name, role, payloads[name]) for name, role in _FILES]
    manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V6_CANDIDATE_PACK_KIND,
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "denominator_record_count": commitment["denominator"]["record_count"],
        "dimensions_sha256": _sha256(canonical_json_bytes(
            commitment["dimensions"])),
        "evaluation_commitment_manifest_sha256": commitment[
            "manifest_sha256"],
        "evaluation_or_reserve_payload_read_count": 0,
        "files": files,
        "formal_run_count": 0,
        "format_version": 1,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "qt_source_manifest_only_read_count": 1,
        "qt_source_non_manifest_read_count": 0,
        "qt_source_pack_manifest_sha256": (
            expected_qt_source_manifest_sha256),
        "status": NORMALIZATION_RECOVERY_V6_CANDIDATE_PACK_STATUS,
        "summary": {
            "approved_target_rule_count": pack["summary"][
                "approved_target_rule_count"],
            "audit_exact_count": audit["summary"]["outcome_counts"]["EXACT"],
            "audit_unknown_count": audit["summary"]["outcome_counts"]["UNKNOWN"],
            "audit_wrong_count": audit["summary"]["outcome_counts"]["WRONG"],
            "preflight_case_count": preflight["case_count"],
            "preflight_failure_count": preflight["failure_count"],
        },
        "teacher_api_llm_call_count": 0,
        "v6_rule_pack_manifest_sha256": pack["manifest_sha256"],
        "v6_training_audit_manifest_sha256": audit["manifest_sha256"],
    }
    return manifest, candidate, preflight, payloads


def _paths(root: Path, values: tuple[tuple[object, str], ...]) -> tuple[Path, ...]:
    """核验全部 candidate 输入目录存在且彼此物理隔离。"""
    paths = tuple(_within(root, value, label=label)
                  for value, label in values)
    if (any(not path.is_dir() for path in paths)
            or any(_overlap(left, right)
                   for index, left in enumerate(paths)
                   for right in paths[index + 1:])):
        raise BroadQaExternalDataError("v6 candidate 输入缺失或 artifact 混淆")
    return paths


def publish_normalization_recovery_v6_candidate_pack(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        predecessor_pack_dir: str | Path,
        expected_predecessor_pack_manifest_sha256: str,
        pack_dir: str | Path,
        expected_pack_manifest_sha256: str,
        audit_dir: str | Path,
        expected_audit_manifest_sha256: str,
        qt_source_pack_dir: str | Path,
        expected_qt_source_manifest_sha256: str,
        evaluation_commitment_dir: str | Path,
        expected_evaluation_commitment_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布标签盲 v6 candidate pack，并以 manifest-last 封口。"""
    root = _require_k_root(run_root)
    inputs = _paths(root, (
        (protocol_dir, "protocol_dir"),
        (predecessor_pack_dir, "predecessor_pack_dir"),
        (pack_dir, "pack_dir"),
        (audit_dir, "audit_dir"),
        (qt_source_pack_dir, "qt_source_pack_dir"),
        (evaluation_commitment_dir, "evaluation_commitment_dir"),
    ))
    target = _within(root, target_dir, label="target_dir")
    if target.exists() or any(_overlap(target, path) for path in inputs):
        raise BroadQaExternalDataError("v6 candidate target 已存在或混淆")
    shas = tuple(_sha_value(value, label=label) for value, label in (
        (expected_protocol_manifest_sha256, "candidate protocol"),
        (expected_predecessor_pack_manifest_sha256, "candidate predecessor"),
        (expected_pack_manifest_sha256, "candidate pack"),
        (expected_audit_manifest_sha256, "candidate audit"),
        (expected_qt_source_manifest_sha256, "candidate Qt source"),
        (expected_evaluation_commitment_manifest_sha256,
         "candidate commitment"),
    ))
    manifest, candidate, preflight, payloads = _derive(
        protocol_dir=inputs[0],
        expected_protocol_manifest_sha256=shas[0],
        predecessor_pack_dir=inputs[1],
        expected_predecessor_pack_manifest_sha256=shas[1],
        pack_dir=inputs[2],
        expected_pack_manifest_sha256=shas[2],
        audit_dir=inputs[3],
        expected_audit_manifest_sha256=shas[3],
        qt_source_pack_dir=inputs[4],
        expected_qt_source_manifest_sha256=shas[4],
        evaluation_commitment_dir=inputs[5],
        expected_evaluation_commitment_manifest_sha256=shas[5],
    )
    del candidate, preflight
    target.mkdir(parents=True)
    for name, _role in _FILES:
        with (target / name).open("xb") as handle:
            handle.write(payloads[name])
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(
        manifest_path.read_bytes())}


def read_normalization_recovery_v6_candidate_pack(
        candidate_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        predecessor_pack_dir: str | Path,
        expected_predecessor_pack_manifest_sha256: str,
        pack_dir: str | Path,
        expected_pack_manifest_sha256: str,
        audit_dir: str | Path,
        expected_audit_manifest_sha256: str,
        qt_source_pack_dir: str | Path,
        expected_qt_source_manifest_sha256: str,
        evaluation_commitment_dir: str | Path,
        expected_evaluation_commitment_manifest_sha256: str,
        expected_candidate_manifest_sha256: str,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """从六个冻结输入重编译并严格回读完整 v6 candidate pack。"""
    root = Path(candidate_dir).resolve()
    inputs = tuple(Path(value).resolve() for value in (
        protocol_dir, predecessor_pack_dir, pack_dir, audit_dir,
        qt_source_pack_dir, evaluation_commitment_dir))
    roots = (root, *inputs)
    if any(_overlap(left, right)
           for index, left in enumerate(roots)
           for right in roots[index + 1:]):
        raise BroadQaExternalDataError("v6 candidate artifact 根混淆")
    expected, candidate, preflight, payloads = _derive(
        protocol_dir=inputs[0],
        expected_protocol_manifest_sha256=_sha_value(
            expected_protocol_manifest_sha256, label="expected protocol"),
        predecessor_pack_dir=inputs[1],
        expected_predecessor_pack_manifest_sha256=_sha_value(
            expected_predecessor_pack_manifest_sha256,
            label="expected predecessor"),
        pack_dir=inputs[2],
        expected_pack_manifest_sha256=_sha_value(
            expected_pack_manifest_sha256, label="expected pack"),
        audit_dir=inputs[3],
        expected_audit_manifest_sha256=_sha_value(
            expected_audit_manifest_sha256, label="expected audit"),
        qt_source_pack_dir=inputs[4],
        expected_qt_source_manifest_sha256=_sha_value(
            expected_qt_source_manifest_sha256, label="expected Qt source"),
        evaluation_commitment_dir=inputs[5],
        expected_evaluation_commitment_manifest_sha256=_sha_value(
            expected_evaluation_commitment_manifest_sha256,
            label="expected commitment"),
    )
    candidate_sha = _sha_value(
        expected_candidate_manifest_sha256, label="expected candidate")
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v6 candidate manifest 不可读") from error
    if (_sha256(encoded) != candidate_sha or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or not _strict_equal(stored, expected)):
        raise BroadQaExternalDataError(
            "v6 candidate manifest identity/encoding/material 漂移")
    for name, _role in _FILES:
        try:
            payload = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v6 candidate {name} 不可读") from error
        if payload != payloads[name]:
            raise BroadQaExternalDataError(
                f"v6 candidate {name} 与冻结输入重派生漂移")
    return ({**stored, "manifest_sha256": candidate_sha}, candidate, preflight)


__all__ = [
    "NORMALIZATION_RECOVERY_V6_CANDIDATE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V6_CANDIDATE_PACK_STATUS",
    "publish_normalization_recovery_v6_candidate_pack",
    "read_normalization_recovery_v6_candidate_pack",
]
