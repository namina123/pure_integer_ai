"""发布 recovery-v7 source identity replay TRAIN-only audit。

artifact 同时报告已见 source commitment 的确定重放与 leave-one-family-out
迁移；前者只证明 replay facility，后者才是 unseen source capability。
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_context_local_audit import (
    _artifact,
    _file_artifact,
    _overlap,
    _read_jsonl,
    _read_manifest,
    _stored_jsonl,
    _write_jsonl,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_source_replay_records import (
    derive_source_replay_audit_records,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V7_SOURCE_REPLAY_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_SOURCE_REPLAY_AUDIT_V1")

V5_TRAINING_PROTOCOL_MANIFEST_SHA256 = (
    "3385e340705af3dd75bd30980f35152574bd967aa257c6d789ee8142d0e87480")
V7_EVALUATION_COMMITMENT_MANIFEST_SHA256 = (
    "a406598a134a0390e101419518f81bf9877a415e8b4b060c4982be0e1844a8d4")
V7_SUCCESSOR_FEASIBILITY_MANIFEST_SHA256 = (
    "649f42d8d60074eea017017152868b6aa17e478c7ee233ebfeb02759cb97d3a9")
V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256 = (
    "a2e40ec5a4950bd167e66100e2b999122ace83a6348aeeddf862ab0d39f75a3e")
V7_CONTEXT_LOCAL_AUDIT_MANIFEST_SHA256 = (
    "52d0e43510cb2647a14f5c63263dadc758abf789fd83b9588031a188556b8902")

_PROTOCOL_FILES = (
    ("train.pair-observations.jsonl", "TRAIN_PAIR_OBSERVATIONS"),
    ("train.phrase-fragments.jsonl", "TRAIN_PHRASE_FRAGMENTS"),
    ("train.phrase-groups.jsonl", "TRAIN_PHRASE_GROUPS"),
)
_PLAN_FILE = "structure-plans.jsonl"
_PLAN_ROLE = "VARIABLE_STRUCTURE_OBLIGATION_PLANS"
_OUTPUT_FILES = (
    ("full-pack-conflict-representations.jsonl",
     "SOURCE_IDENTITY_REPLAY_CONFLICT_REPRESENTATIONS"),
    ("loso-audit.jsonl", "SOURCE_IDENTITY_REPLAY_FOUR_SOURCE_LOSO"),
)


def _sha256(payload: bytes) -> str:
    """返回规范 manifest SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v7 source replay root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入或输出位于显式 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"v7 source replay {label} path 越界")
    return path


def _validate_inputs(
        protocol: dict[str, object],
        commitment: dict[str, object],
        feasibility: dict[str, object],
        variable: dict[str, object],
        context_local: dict[str, object],
        ) -> None:
    """核验 predecessor facts、NE 边界与 held-out 禁读合同。"""
    feasibility_summary = feasibility.get("summary")
    source_summary = feasibility_summary.get("source_policy_replay") \
        if isinstance(feasibility_summary, dict) else None
    variable_summary = variable.get("summary")
    variable_plans = variable_summary.get("plans") \
        if isinstance(variable_summary, dict) else None
    context_summary = context_local.get("summary")
    context_loso = context_summary.get("loso") \
        if isinstance(context_summary, dict) else None
    denominator = commitment.get("denominator")
    if (protocol.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_TRAINING_PROTOCOL_V1"
            or protocol.get("status") != "FROZEN_NOT_READ_NOT_LEARNED"
            or protocol.get("evaluation_or_held_out_payload_read_count") != 0
            or commitment.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_V1"
            or commitment.get("status")
            != "LABEL_BLIND_DENOMINATOR_AND_GATES_FROZEN_BEFORE_V7_LEARNER_CHANGE"
            or commitment.get("source_non_manifest_file_read_count") != 0
            or not isinstance(denominator, dict)
            or denominator.get("record_count") != 3_656
            or feasibility.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_SUCCESSOR_FEASIBILITY_V1"
            or feasibility.get("status")
            != "TRAIN_ONLY_FEASIBILITY_COMPLETE_NOT_LEARNER_NOT_RUNTIME"
            or not isinstance(source_summary, dict)
            or source_summary.get("train_output_conflict_count") != 6_687
            or source_summary.get("replayable_conflict_count") != 984
            or source_summary.get("context_or_source_identity_required_count")
            != 5_703
            or variable.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_AUDIT_V1"
            or variable.get("status")
            != "TRAIN_ONLY_REPRESENTATION_PASS_CAPABILITY_NE_NOT_RUNTIME"
            or not isinstance(variable_plans, dict)
            or variable_plans.get("plan_count") != 3_460
            or context_local.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_CONTEXT_LOCAL_AUDIT_V1"
            or context_local.get("status")
            != "TRAIN_ONLY_CONTEXT_LOCAL_NE_NOT_RUNTIME"
            or context_local.get("runtime_program_published") != 0
            or not isinstance(context_loso, dict)
            or context_loso.get("capability_outcome")
            != "NE_ZERO_VARIABLE_EXACT"
            or context_loso.get("wrong_count") != 0):
        raise BroadQaExternalDataError(
            "v7 source replay sealed input contract 漂移")


def _input_state(
        protocol_dir: Path,
        commitment_dir: Path,
        feasibility_dir: Path,
        variable_dir: Path,
        context_local_dir: Path,
        ) -> tuple[
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, object],
        ]:
    """读取并核验五个 sealed input manifests。"""
    protocol = _read_manifest(
        protocol_dir, expected_sha256=V5_TRAINING_PROTOCOL_MANIFEST_SHA256,
        label="v5 protocol")
    commitment = _read_manifest(
        commitment_dir,
        expected_sha256=V7_EVALUATION_COMMITMENT_MANIFEST_SHA256,
        label="v7 commitment")
    feasibility = _read_manifest(
        feasibility_dir,
        expected_sha256=V7_SUCCESSOR_FEASIBILITY_MANIFEST_SHA256,
        label="v7 feasibility")
    variable = _read_manifest(
        variable_dir,
        expected_sha256=V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256,
        label="v7 variable structure")
    context_local = _read_manifest(
        context_local_dir,
        expected_sha256=V7_CONTEXT_LOCAL_AUDIT_MANIFEST_SHA256,
        label="v7 context local")
    _validate_inputs(
        protocol, commitment, feasibility, variable, context_local)
    return protocol, commitment, feasibility, variable, context_local


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
    """读取 TRAIN material 并派生 full-pack routes 与四向 LOSO。"""
    values = []
    for name, role in _PROTOCOL_FILES:
        values.append(_read_jsonl(
            protocol_dir / name,
            artifact=_file_artifact(
                protocol, relative_path=name, role=role),
            label=role,
        ))
    observations, fragments, groups = values
    plans = _read_jsonl(
        variable_dir / _PLAN_FILE,
        artifact=_file_artifact(
            variable, relative_path=_PLAN_FILE, role=_PLAN_ROLE),
        label=_PLAN_ROLE,
    )
    representations, loso, summary = derive_source_replay_audit_records(
        observations=observations,
        fragments=fragments,
        groups=groups,
        plans=plans,
    )
    full = summary["full_pack"]
    if (full["conflict_count"] != 6_687
            or full["all_routes_unique_conflict_count"] != 6_483
            or full["ambiguous_conflict_count"] != 204
            or full["unique_route_count"] != 88_466
            or full["ambiguous_route_count"] != 370
            or full["source_commitment_schema_count"] != 4
            or full["source_commitment_schema_cross_family_count"] != 0
            or full["exact_commitment_cross_family_count"] != 0):
        raise BroadQaExternalDataError(
            "v7 source replay frozen feasibility facts 漂移")
    return {
        _OUTPUT_FILES[0][0]: representations,
        _OUTPUT_FILES[1][0]: loso,
    }, {
        "audit_outcome": (
            "FACILITY_FAIL" if summary["facility_outcome"] != "PASS"
            else "REPLAY_PASS_UNSEEN_TRANSFER_PASS"
            if summary["capability_outcome"]
            == "PASS_NONZERO_UNSEEN_FAMILY_EXACT"
            else "REPLAY_PASS_UNSEEN_TRANSFER_NE"
            if summary["capability_outcome"]
            == "NE_ZERO_UNSEEN_FAMILY_EXACT"
            else "REPLAY_PASS_UNSEEN_TRANSFER_FAIL"),
        "loso": summary,
        "loso_group_rederivation_count": 4,
    }


def _status(summary: dict[str, object]) -> str:
    """把两本账结果映射为诚实的非 runtime 状态。"""
    suffix = {
        "REPLAY_PASS_UNSEEN_TRANSFER_PASS": "PASS",
        "REPLAY_PASS_UNSEEN_TRANSFER_NE": "NE",
        "REPLAY_PASS_UNSEEN_TRANSFER_FAIL": "FAIL",
        "FACILITY_FAIL": "FACILITY_FAIL",
    }.get(str(summary["audit_outcome"]))
    if suffix is None:
        raise BroadQaExternalDataError(
            "v7 source replay audit outcome 非法")
    return f"TRAIN_ONLY_SOURCE_REPLAY_{suffix}_NOT_RUNTIME"


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> dict[str, object]:
    """构造 source replay TRAIN-only audit manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V7_SOURCE_REPLAY_AUDIT_KIND,
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
            "v7_context_local_audit_manifest_sha256": (
                V7_CONTEXT_LOCAL_AUDIT_MANIFEST_SHA256),
            "v7_evaluation_commitment_manifest_sha256": (
                V7_EVALUATION_COMMITMENT_MANIFEST_SHA256),
            "v7_successor_feasibility_manifest_sha256": (
                V7_SUCCESSOR_FEASIBILITY_MANIFEST_SHA256),
            "v7_variable_structure_audit_manifest_sha256": (
                V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256),
        },
        "learner_or_selection_change_count": 0,
        "loso_group_rederivation_count": 4,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": _status(summary),
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_surface_published_in_audit": 0,
    }


def publish_normalization_recovery_v7_source_replay_audit(
        *,
        run_root: str | Path,
        training_protocol_dir: str | Path,
        evaluation_commitment_dir: str | Path,
        successor_feasibility_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        context_local_audit_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 source identity replay TRAIN-only audit。"""
    root = _require_k_root(run_root)
    protocol_dir = _within(root, training_protocol_dir, label="protocol")
    commitment_dir = _within(root, evaluation_commitment_dir, label="commitment")
    feasibility_dir = _within(root, successor_feasibility_dir, label="feasibility")
    variable_dir = _within(
        root, variable_structure_audit_dir, label="variable structure")
    context_local_dir = _within(
        root, context_local_audit_dir, label="context local")
    target = _within(root, target_dir, label="target")
    inputs = (
        protocol_dir, commitment_dir, feasibility_dir,
        variable_dir, context_local_dir)
    if (any(not path.is_dir() for path in inputs)
            or any(_overlap(target, path) for path in inputs)
            or target.exists()):
        raise BroadQaExternalDataError(
            "v7 source replay input/target path 非法")
    protocol, _commitment, _feasibility, variable, _context = _input_state(
        protocol_dir, commitment_dir, feasibility_dir,
        variable_dir, context_local_dir)
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


def read_normalization_recovery_v7_source_replay_audit(
        audit_dir: str | Path,
        *,
        training_protocol_dir: str | Path,
        evaluation_commitment_dir: str | Path,
        successor_feasibility_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        context_local_audit_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """从五个 sealed inputs 重派生并严格回读 source replay audit。"""
    root = Path(audit_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v7 source replay audit manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v7 source replay audit manifest identity 漂移")
    protocol_dir = Path(training_protocol_dir).resolve()
    commitment_dir = Path(evaluation_commitment_dir).resolve()
    feasibility_dir = Path(successor_feasibility_dir).resolve()
    variable_dir = Path(variable_structure_audit_dir).resolve()
    context_local_dir = Path(context_local_audit_dir).resolve()
    protocol, _commitment, _feasibility, variable, _context = _input_state(
        protocol_dir, commitment_dir, feasibility_dir,
        variable_dir, context_local_dir)
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
            "v7 source replay audit records/inputs 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    if not strict_json_equal(stored, _manifest(files=files, summary=summary)):
        raise BroadQaExternalDataError(
            "v7 source replay audit manifest 字段漂移")
    return ({**stored, "manifest_sha256": expected_manifest_sha256},
            stored_outputs)


__all__ = [
    "NORMALIZATION_RECOVERY_V7_SOURCE_REPLAY_AUDIT_KIND",
    "publish_normalization_recovery_v7_source_replay_audit",
    "read_normalization_recovery_v7_source_replay_audit",
]
