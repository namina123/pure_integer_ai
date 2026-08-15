"""发布 recovery-v7 variable structure TRAIN-only representation audit。

publisher 只读四来源 TRAIN observations、v7 commitment manifest 与 predecessor
feasibility manifest。它发布不可执行 plans 和四方向 LOSO；VLC/Qt label、learner、
runtime、candidate 与 formal 均不可读或创建。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    strict_json_equal,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_variable_structure_records import (
    derive_variable_structure_plans,
    run_variable_structure_loso,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_AUDIT_V1")
NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_AUDIT_STATUS = (
    "TRAIN_ONLY_REPRESENTATION_PASS_CAPABILITY_NE_NOT_RUNTIME")

V5_TRAINING_PROTOCOL_MANIFEST_SHA256 = (
    "3385e340705af3dd75bd30980f35152574bd967aa257c6d789ee8142d0e87480")
V7_EVALUATION_COMMITMENT_MANIFEST_SHA256 = (
    "a406598a134a0390e101419518f81bf9877a415e8b4b060c4982be0e1844a8d4")
V7_SUCCESSOR_FEASIBILITY_MANIFEST_SHA256 = (
    "649f42d8d60074eea017017152868b6aa17e478c7ee233ebfeb02759cb97d3a9")

_OBSERVATION_FILE = "train.pair-observations.jsonl"
_OBSERVATION_ROLE = "TRAIN_PAIR_OBSERVATIONS"
_OUTPUT_FILES = (
    ("structure-plans.jsonl", "VARIABLE_STRUCTURE_OBLIGATION_PLANS"),
    ("loso-audit.jsonl", "VARIABLE_STRUCTURE_FOUR_SOURCE_LOSO"),
)


def _sha256(payload: bytes) -> str:
    """返回文件或规范 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v7 variable root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制一个输入或输出路径位于 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"v7 variable structure {label} path 越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个目录是否相同或互为祖先。"""
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _read_manifest(
        directory: Path,
        *,
        expected_sha256: str,
        label: str,
        ) -> dict[str, object]:
    """读取规范 manifest 并核对固定 SHA。"""
    try:
        payload = (directory / "manifest.json").read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v7 variable structure {label} manifest 不可读") from error
    if (not isinstance(value, dict)
            or _sha256(payload) != expected_sha256
            or canonical_json_line(value) != payload):
        raise BroadQaExternalDataError(
            f"v7 variable structure {label} manifest identity 漂移")
    return value


def _observation_artifact(protocol: dict[str, object]) -> dict[str, object]:
    """取得 sealed observation JSONL 的唯一物理承诺。"""
    files = protocol.get("files")
    matches = [item for item in files if isinstance(item, dict)
               and item.get("relative_path") == _OBSERVATION_FILE
               and item.get("role") == _OBSERVATION_ROLE] \
        if isinstance(files, list) else []
    if len(matches) != 1:
        raise BroadQaExternalDataError(
            "v7 variable structure observation artifact 漂移")
    value = matches[0]
    if (type(value.get("bytes")) is not int or value["bytes"] <= 0
            or type(value.get("record_count")) is not int
            or value["record_count"] <= 0
            or not isinstance(value.get("sha256"), str)
            or len(value["sha256"]) != 64):
        raise BroadQaExternalDataError(
            "v7 variable structure observation artifact 非法")
    return value


def _iter_observations(
        path: Path,
        *,
        artifact: dict[str, object],
        ) -> Iterator[dict[str, object]]:
    """流式读取 observations，并在 EOF 核对完整 SHA/bytes/count。"""
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    try:
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                byte_count += len(line)
                record_count += 1
                try:
                    value = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise BroadQaExternalDataError(
                        "v7 variable structure observations JSONL 非法") from error
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        "v7 variable structure observations 非规范")
                yield value
    except OSError as error:
        raise BroadQaExternalDataError(
            "v7 variable structure observations 不可读") from error
    if (byte_count != artifact["bytes"]
            or record_count != artifact["record_count"]
            or digest.hexdigest() != artifact["sha256"]):
        raise BroadQaExternalDataError(
            "v7 variable structure observations 物理 identity 漂移")


def _validate_inputs(
        protocol: dict[str, object],
        commitment: dict[str, object],
        feasibility: dict[str, object],
        ) -> None:
    """核验 commitment 先冻结、feasibility 已完成且 held-out payload 为零。"""
    denominator = commitment.get("denominator")
    feasibility_summary = feasibility.get("summary")
    if (protocol.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_TRAINING_PROTOCOL_V1"
            or protocol.get("status") != "FROZEN_NOT_READ_NOT_LEARNED"
            or protocol.get("evaluation_or_held_out_payload_read_count") != 0
            or protocol.get("production_enabled") != 0
            or commitment.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_V1"
            or commitment.get("status")
            != "LABEL_BLIND_DENOMINATOR_AND_GATES_FROZEN_BEFORE_V7_LEARNER_CHANGE"
            or commitment.get("source_non_manifest_file_read_count") != 0
            or commitment.get("training_source_read_count") != 0
            or commitment.get("production_enabled") != 0
            or not isinstance(denominator, dict)
            or denominator.get("record_count") != 3_656
            or feasibility.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_SUCCESSOR_FEASIBILITY_V1"
            or feasibility.get("status")
            != "TRAIN_ONLY_FEASIBILITY_COMPLETE_NOT_LEARNER_NOT_RUNTIME"
            or feasibility.get("learner_or_selection_change_count") != 0
            or feasibility.get("runtime_program_published") != 0
            or not isinstance(feasibility_summary, dict)
            or feasibility_summary.get("overall_outcome")
            != "FEASIBILITY_CONFIRMED_NARROW_OR_PARTIAL_IMPLEMENTATION_REQUIRED"):
        raise BroadQaExternalDataError(
            "v7 variable structure sealed input contract 漂移")


def _input_state(
        protocol_dir: Path,
        commitment_dir: Path,
        feasibility_dir: Path,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """读取并核验三个 sealed input manifests。"""
    protocol = _read_manifest(
        protocol_dir,
        expected_sha256=V5_TRAINING_PROTOCOL_MANIFEST_SHA256,
        label="v5 protocol",
    )
    commitment = _read_manifest(
        commitment_dir,
        expected_sha256=V7_EVALUATION_COMMITMENT_MANIFEST_SHA256,
        label="v7 commitment",
    )
    feasibility = _read_manifest(
        feasibility_dir,
        expected_sha256=V7_SUCCESSOR_FEASIBILITY_MANIFEST_SHA256,
        label="v7 feasibility",
    )
    _validate_inputs(protocol, commitment, feasibility)
    return protocol, commitment, feasibility


def _derive(
        *,
        protocol_dir: Path,
        protocol: dict[str, object],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
        ]:
    """从 sealed observations 派生 plans、LOSO 与总分账。"""
    plans, plan_summary = derive_variable_structure_plans(_iter_observations(
        protocol_dir / _OBSERVATION_FILE,
        artifact=_observation_artifact(protocol),
    ))
    loso, loso_summary = run_variable_structure_loso(plans)
    if (loso_summary["facility_outcome"] != "PASS"
            or loso_summary["wrong_count"] != 0
            or loso_summary["partial_commit_count"] != 0
            or loso_summary["capability_outcome"]
            != "NE_SEGMENT_GENERATOR_NOT_IMPLEMENTED"):
        raise BroadQaExternalDataError(
            "v7 variable structure audit hard gate 漂移")
    return {
        _OUTPUT_FILES[0][0]: plans,
        _OUTPUT_FILES[1][0]: loso,
    }, {
        "audit_outcome": "REPRESENTATION_PASS_CAPABILITY_NE",
        "loso": loso_summary,
        "plans": plan_summary,
    }


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """独占写入规范 output JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """读取规范 output JSONL。"""
    try:
        payload = path.read_bytes()
        lines = payload.splitlines(keepends=True)
        values = tuple(json.loads(line) for line in lines)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v7 variable structure {label} 不可读") from error
    if (not lines or b"".join(lines) != payload
            or any(not isinstance(item, dict) for item in values)
            or b"".join(canonical_json_line(item) for item in values)
            != payload):
        raise BroadQaExternalDataError(
            f"v7 variable structure {label} 非规范")
    return values


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """构造 output 物理文件承诺。"""
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "record_count": count,
        "relative_path": path.name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> dict[str, object]:
    """构造 representation PASS、capability NE 的 audit manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_AUDIT_KIND,
        "candidate_family_formal_run_count": 0,
        "files": files,
        "format_version": 1,
        "held_out_boundary": {
            "consumed_qt_individual_or_derivative_read_count": 0,
            "vlc_commitment_manifest_read_count": 1,
            "vlc_identity_raw_or_translation_read_count": 0,
        },
        "inputs": {
            "v5_training_protocol_manifest_sha256": (
                V5_TRAINING_PROTOCOL_MANIFEST_SHA256),
            "v7_evaluation_commitment_manifest_sha256": (
                V7_EVALUATION_COMMITMENT_MANIFEST_SHA256),
            "v7_successor_feasibility_manifest_sha256": (
                V7_SUCCESSOR_FEASIBILITY_MANIFEST_SHA256),
        },
        "learner_or_selection_change_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_AUDIT_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_surface_published_in_audit": 0,
    }


def publish_normalization_recovery_v7_variable_structure_audit(
        *,
        run_root: str | Path,
        training_protocol_dir: str | Path,
        evaluation_commitment_dir: str | Path,
        successor_feasibility_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 variable structure representation audit。"""
    root = _require_k_root(run_root)
    protocol_dir = _within(root, training_protocol_dir, label="protocol")
    commitment_dir = _within(root, evaluation_commitment_dir, label="commitment")
    feasibility_dir = _within(
        root, successor_feasibility_dir, label="feasibility")
    target = _within(root, target_dir, label="target")
    inputs = (protocol_dir, commitment_dir, feasibility_dir)
    if (any(not path.is_dir() for path in inputs)
            or any(_overlap(target, path) for path in inputs)
            or target.exists()):
        raise BroadQaExternalDataError(
            "v7 variable structure input/target path 非法")
    protocol, _commitment, _feasibility = _input_state(
        protocol_dir, commitment_dir, feasibility_dir)
    outputs, summary = _derive(
        protocol_dir=protocol_dir, protocol=protocol)
    target.mkdir()
    files = []
    for name, role in _OUTPUT_FILES:
        path = target / name
        _write_jsonl(path, outputs[name])
        files.append(_artifact(path, role=role, count=len(outputs[name])))
    manifest = _manifest(files=files, summary=summary)
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(
        manifest_path.read_bytes())}


def read_normalization_recovery_v7_variable_structure_audit(
        audit_dir: str | Path,
        *,
        training_protocol_dir: str | Path,
        evaluation_commitment_dir: str | Path,
        successor_feasibility_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """从 sealed inputs 重派生并严格回读 representation audit。"""
    root = Path(audit_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v7 variable structure audit manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v7 variable structure audit manifest identity 漂移")
    protocol_dir = Path(training_protocol_dir).resolve()
    commitment_dir = Path(evaluation_commitment_dir).resolve()
    feasibility_dir = Path(successor_feasibility_dir).resolve()
    protocol, _commitment, _feasibility = _input_state(
        protocol_dir, commitment_dir, feasibility_dir)
    expected_outputs, summary = _derive(
        protocol_dir=protocol_dir, protocol=protocol)
    stored_outputs = {
        name: _read_jsonl(root / name, label=name)
        for name, _role in _OUTPUT_FILES
    }
    if any(not strict_json_equal(
            stored_outputs[name], expected_outputs[name])
           for name, _role in _OUTPUT_FILES):
        raise BroadQaExternalDataError(
            "v7 variable structure audit records/inputs 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    if not strict_json_equal(stored, _manifest(files=files, summary=summary)):
        raise BroadQaExternalDataError(
            "v7 variable structure audit manifest 字段漂移")
    return ({**stored, "manifest_sha256": expected_manifest_sha256},
            stored_outputs)


__all__ = [
    "NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_AUDIT_KIND",
    "NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_AUDIT_STATUS",
    "publish_normalization_recovery_v7_variable_structure_audit",
    "read_normalization_recovery_v7_variable_structure_audit",
]
