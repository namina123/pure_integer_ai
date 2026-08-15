"""发布 recovery-v7 context-scoped local TRAIN-only LOSO audit。

publisher 只读 sealed v5 TRAIN observations/fragments、v7 commitment、successor
feasibility 与 variable structure plans。四个方向各从另外三家重新学习一次；输出
只含规则/结果摘要，不发布 TRAIN surface、runtime program、candidate 或 formal。
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_context_local_records import (
    derive_context_scoped_local_loso,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V7_CONTEXT_LOCAL_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_CONTEXT_LOCAL_AUDIT_V1")

V5_TRAINING_PROTOCOL_MANIFEST_SHA256 = (
    "3385e340705af3dd75bd30980f35152574bd967aa257c6d789ee8142d0e87480")
V7_EVALUATION_COMMITMENT_MANIFEST_SHA256 = (
    "a406598a134a0390e101419518f81bf9877a415e8b4b060c4982be0e1844a8d4")
V7_SUCCESSOR_FEASIBILITY_MANIFEST_SHA256 = (
    "649f42d8d60074eea017017152868b6aa17e478c7ee233ebfeb02759cb97d3a9")
V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256 = (
    "a2e40ec5a4950bd167e66100e2b999122ace83a6348aeeddf862ab0d39f75a3e")

_INPUT_FILES = (
    ("train.pair-observations.jsonl", "TRAIN_PAIR_OBSERVATIONS"),
    ("train.phrase-fragments.jsonl", "TRAIN_PHRASE_FRAGMENTS"),
)
_PLAN_FILE = "structure-plans.jsonl"
_PLAN_ROLE = "VARIABLE_STRUCTURE_OBLIGATION_PLANS"
_OUTPUT_FILES = (
    ("loso-rule-representations.jsonl", "CONTEXT_LOCAL_LOSO_RULE_REPRESENTATIONS"),
    ("loso-audit.jsonl", "CONTEXT_LOCAL_FOUR_SOURCE_LOSO"),
)


def _sha256(payload: bytes) -> str:
    """返回文件或规范 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v7 context local root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入或输出位于显式 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"v7 context local {label} path 越界")
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
            f"v7 context local {label} manifest 不可读") from error
    if (not isinstance(value, dict)
            or _sha256(payload) != expected_sha256
            or canonical_json_line(value) != payload):
        raise BroadQaExternalDataError(
            f"v7 context local {label} manifest identity 漂移")
    return value


def _file_artifact(
        manifest: dict[str, object],
        *,
        relative_path: str,
        role: str,
        ) -> dict[str, object]:
    """从 sealed manifest 取得唯一文件物理承诺。"""
    files = manifest.get("files")
    matches = [
        item for item in files
        if isinstance(item, dict)
        and item.get("relative_path") == relative_path
        and item.get("role") == role
    ] if isinstance(files, list) else []
    if len(matches) != 1:
        raise BroadQaExternalDataError(
            f"v7 context local {relative_path} artifact 漂移")
    value = matches[0]
    if (type(value.get("bytes")) is not int or value["bytes"] <= 0
            or type(value.get("record_count")) is not int
            or value["record_count"] <= 0
            or not isinstance(value.get("sha256"), str)
            or len(value["sha256"]) != 64):
        raise BroadQaExternalDataError(
            f"v7 context local {relative_path} artifact 非法")
    return value


def _read_jsonl(
        path: Path,
        *,
        artifact: dict[str, object],
        label: str,
        ) -> tuple[dict[str, object], ...]:
    """读取规范 JSONL，并在 EOF 核对 bytes/count/SHA。"""
    digest = hashlib.sha256()
    byte_count = 0
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                byte_count += len(line)
                try:
                    value = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise BroadQaExternalDataError(
                        f"v7 context local {label} JSONL 非法") from error
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        f"v7 context local {label} 非规范")
                values.append(value)
    except OSError as error:
        raise BroadQaExternalDataError(
            f"v7 context local {label} 不可读") from error
    if (byte_count != artifact["bytes"]
            or len(values) != artifact["record_count"]
            or digest.hexdigest() != artifact["sha256"]):
        raise BroadQaExternalDataError(
            f"v7 context local {label} 物理 identity 漂移")
    return tuple(values)


def _validate_inputs(
        protocol: dict[str, object],
        commitment: dict[str, object],
        feasibility: dict[str, object],
        variable: dict[str, object],
        ) -> None:
    """核验 frozen 472/28 事实、structure PASS 与 held-out 禁读边界。"""
    denominator = commitment.get("denominator")
    feasibility_summary = feasibility.get("summary")
    context_summary = feasibility_summary.get(
        "context_scoped_local_transfer") \
        if isinstance(feasibility_summary, dict) else None
    variable_summary = variable.get("summary")
    variable_plans = variable_summary.get("plans") \
        if isinstance(variable_summary, dict) else None
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
            or not isinstance(denominator, dict)
            or denominator.get("record_count") != 3_656
            or feasibility.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_SUCCESSOR_FEASIBILITY_V1"
            or feasibility.get("status")
            != "TRAIN_ONLY_FEASIBILITY_COMPLETE_NOT_LEARNER_NOT_RUNTIME"
            or feasibility.get("learner_or_selection_change_count") != 0
            or feasibility.get("runtime_program_published") != 0
            or not isinstance(context_summary, dict)
            or context_summary.get("rule_count") != 500
            or context_summary.get("support_closed") != 500
            or context_summary.get("has_defeater") != 500
            or context_summary.get("positive_nonempty_context") != 472
            or context_summary.get("representation_feasible") != 472
            or context_summary.get("status")
            != "FEASIBLE_WITH_DEFER_AND_ATOMIC_COMMIT_IMPLEMENTATION_REQUIRED"
            or variable.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_AUDIT_V1"
            or variable.get("status")
            != "TRAIN_ONLY_REPRESENTATION_PASS_CAPABILITY_NE_NOT_RUNTIME"
            or variable.get("runtime_program_published") != 0
            or variable.get("candidate_family_formal_run_count") != 0
            or not isinstance(variable_plans, dict)
            or variable_plans.get("plan_count") != 3_460
            or variable_plans.get("representation_eligible_count") != 3_459
            or variable_plans.get("obligation_count") != 6_691):
        raise BroadQaExternalDataError(
            "v7 context local sealed input contract 漂移")


def _input_state(
        protocol_dir: Path,
        commitment_dir: Path,
        feasibility_dir: Path,
        variable_dir: Path,
        ) -> tuple[
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, object],
        ]:
    """读取并核验四个 sealed input manifests。"""
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
    variable = _read_manifest(
        variable_dir,
        expected_sha256=V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256,
        label="v7 variable structure",
    )
    _validate_inputs(protocol, commitment, feasibility, variable)
    return protocol, commitment, feasibility, variable


def _derive(
        *,
        protocol_dir: Path,
        protocol: dict[str, object],
        variable_dir: Path,
        variable: dict[str, object],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
        ]:
    """读取所需 TRAIN 文件并派生四向 LOSO 摘要。"""
    observations = _read_jsonl(
        protocol_dir / _INPUT_FILES[0][0],
        artifact=_file_artifact(
            protocol, relative_path=_INPUT_FILES[0][0],
            role=_INPUT_FILES[0][1]),
        label=_INPUT_FILES[0][1],
    )
    fragments = _read_jsonl(
        protocol_dir / _INPUT_FILES[1][0],
        artifact=_file_artifact(
            protocol, relative_path=_INPUT_FILES[1][0],
            role=_INPUT_FILES[1][1]),
        label=_INPUT_FILES[1][1],
    )
    plans = _read_jsonl(
        variable_dir / _PLAN_FILE,
        artifact=_file_artifact(
            variable, relative_path=_PLAN_FILE, role=_PLAN_ROLE),
        label=_PLAN_ROLE,
    )
    rules, loso, summary = derive_context_scoped_local_loso(
        protocol_manifest_sha256=V5_TRAINING_PROTOCOL_MANIFEST_SHA256,
        observations=observations,
        fragments=fragments,
        plans=plans,
    )
    return {
        _OUTPUT_FILES[0][0]: rules,
        _OUTPUT_FILES[1][0]: loso,
    }, {
        "audit_outcome": (
            "FACILITY_FAIL" if summary["facility_outcome"] != "PASS"
            else "REPRESENTATION_PASS_CAPABILITY_PASS"
            if summary["capability_outcome"]
            == "PASS_NONZERO_VARIABLE_EXACT"
            else "REPRESENTATION_PASS_CAPABILITY_NE"
            if summary["capability_outcome"] == "NE_ZERO_VARIABLE_EXACT"
            else "REPRESENTATION_PASS_CAPABILITY_FAIL"),
        "frozen_full_pack_context_rule_count": 500,
        "frozen_full_pack_deferred_no_surface_context_count": 28,
        "frozen_full_pack_representation_eligible_count": 472,
        "loso": summary,
        "loso_relearning_count": 4,
    }


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """以不可覆盖方式写入规范 JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _stored_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """读取已发布 output JSONL 并核对规范编码。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        f"v7 context local stored {label} 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v7 context local stored {label} 不可读") from error
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """形成 output file 的物理承诺。"""
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "record_count": count,
        "relative_path": path.name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _status(summary: dict[str, object]) -> str:
    """把 audit outcome 映射为诚实的非 runtime 状态。"""
    suffix = {
        "REPRESENTATION_PASS_CAPABILITY_PASS": "PASS",
        "REPRESENTATION_PASS_CAPABILITY_NE": "NE",
        "REPRESENTATION_PASS_CAPABILITY_FAIL": "FAIL",
        "FACILITY_FAIL": "FACILITY_FAIL",
    }.get(str(summary["audit_outcome"]))
    if suffix is None:
        raise BroadQaExternalDataError(
            "v7 context local audit outcome 非法")
    return f"TRAIN_ONLY_CONTEXT_LOCAL_{suffix}_NOT_RUNTIME"


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> dict[str, object]:
    """构造 TRAIN-only context local audit manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V7_CONTEXT_LOCAL_AUDIT_KIND,
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
            "v7_variable_structure_audit_manifest_sha256": (
                V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256),
        },
        "learner_or_selection_change_count": 0,
        "loso_relearning_count": 4,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": _status(summary),
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_surface_published_in_audit": 0,
    }


def publish_normalization_recovery_v7_context_local_audit(
        *,
        run_root: str | Path,
        training_protocol_dir: str | Path,
        evaluation_commitment_dir: str | Path,
        successor_feasibility_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 context-scoped local TRAIN-only audit。"""
    root = _require_k_root(run_root)
    protocol_dir = _within(root, training_protocol_dir, label="protocol")
    commitment_dir = _within(root, evaluation_commitment_dir, label="commitment")
    feasibility_dir = _within(root, successor_feasibility_dir, label="feasibility")
    variable_dir = _within(
        root, variable_structure_audit_dir, label="variable structure")
    target = _within(root, target_dir, label="target")
    inputs = (protocol_dir, commitment_dir, feasibility_dir, variable_dir)
    if (any(not path.is_dir() for path in inputs)
            or any(_overlap(target, path) for path in inputs)
            or target.exists()):
        raise BroadQaExternalDataError(
            "v7 context local input/target path 非法")
    protocol, _commitment, _feasibility, variable = _input_state(
        protocol_dir, commitment_dir, feasibility_dir, variable_dir)
    outputs, summary = _derive(
        protocol_dir=protocol_dir,
        protocol=protocol,
        variable_dir=variable_dir,
        variable=variable,
    )
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


def read_normalization_recovery_v7_context_local_audit(
        audit_dir: str | Path,
        *,
        training_protocol_dir: str | Path,
        evaluation_commitment_dir: str | Path,
        successor_feasibility_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """从四个 sealed inputs 重派生并严格回读 audit。"""
    root = Path(audit_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v7 context local audit manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v7 context local audit manifest identity 漂移")
    protocol_dir = Path(training_protocol_dir).resolve()
    commitment_dir = Path(evaluation_commitment_dir).resolve()
    feasibility_dir = Path(successor_feasibility_dir).resolve()
    variable_dir = Path(variable_structure_audit_dir).resolve()
    protocol, _commitment, _feasibility, variable = _input_state(
        protocol_dir, commitment_dir, feasibility_dir, variable_dir)
    expected_outputs, summary = _derive(
        protocol_dir=protocol_dir,
        protocol=protocol,
        variable_dir=variable_dir,
        variable=variable,
    )
    stored_outputs = {
        name: _stored_jsonl(root / name, label=name)
        for name, _role in _OUTPUT_FILES
    }
    if any(not strict_json_equal(
            stored_outputs[name], expected_outputs[name])
           for name, _role in _OUTPUT_FILES):
        raise BroadQaExternalDataError(
            "v7 context local audit records/inputs 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    if not strict_json_equal(stored, _manifest(files=files, summary=summary)):
        raise BroadQaExternalDataError(
            "v7 context local audit manifest 字段漂移")
    return ({**stored, "manifest_sha256": expected_manifest_sha256},
            stored_outputs)


__all__ = [
    "NORMALIZATION_RECOVERY_V7_CONTEXT_LOCAL_AUDIT_KIND",
    "publish_normalization_recovery_v7_context_local_audit",
    "read_normalization_recovery_v7_context_local_audit",
]
