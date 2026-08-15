"""发布 recovery-v7 cross-source transformation TRAIN-only feasibility。

artifact 只消费冻结 TRAIN material、variable plans 与 neutral projection，并读取
context-local/source-replay predecessor manifest 作为恢复链。它不创建 learner pack、
candidate、runtime 或 formal evaluation，也不发布任何 source/input/output surface。
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_cross_source_transformation_records import (
    derive_cross_source_transformation_feasibility,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V7_CROSS_SOURCE_TRANSFORMATION_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_"
    "CROSS_SOURCE_TRANSFORMATION_FEASIBILITY_V1")

V5_TRAINING_PROTOCOL_MANIFEST_SHA256 = (
    "3385e340705af3dd75bd30980f35152574bd967aa257c6d789ee8142d0e87480")
V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256 = (
    "a2e40ec5a4950bd167e66100e2b999122ace83a6348aeeddf862ab0d39f75a3e")
V7_CONTEXT_LOCAL_AUDIT_MANIFEST_SHA256 = (
    "52d0e43510cb2647a14f5c63263dadc758abf789fd83b9588031a188556b8902")
V7_SOURCE_REPLAY_AUDIT_MANIFEST_SHA256 = (
    "4fad3ca859d1b7349de4f5566e60fecb029a1751b62dba799b00ee589ff35b2a")
V7_NEUTRAL_SOURCE_PROJECTION_MANIFEST_SHA256 = (
    "5de3fe9a077703a0b915c64c93a5650bed79c4a3757c819340018f748b84ca23")

_PROTOCOL_FILES = (
    ("train.pair-observations.jsonl", "TRAIN_PAIR_OBSERVATIONS"),
    ("train.phrase-fragments.jsonl", "TRAIN_PHRASE_FRAGMENTS"),
)
_PLAN_FILE = "structure-plans.jsonl"
_PLAN_ROLE = "VARIABLE_STRUCTURE_OBLIGATION_PLANS"
_PROJECTION_FILE = "neutral-projections.jsonl"
_PROJECTION_ROLE = "NEUTRAL_SOURCE_COMMITMENT_PROJECTIONS"
_OUTPUT_FILES = (
    ("model-representations.jsonl",
     "CROSS_SOURCE_TRANSFORMATION_MODEL_REPRESENTATIONS"),
    ("stage-audit.jsonl", "CROSS_SOURCE_TRANSFORMATION_STAGE_AUDIT"),
    ("loso-audit.jsonl", "CROSS_SOURCE_TRANSFORMATION_FOUR_SOURCE_LOSO"),
)

_EXPECTED_MODEL_FACTS = {
    "GODOT_ENGINE_PROJECT": {
        "conflict_input_count": 602,
        "identity_veto_input_count": 182,
        "route_count": 2_798,
        "route_identity_set_sha256": (
            "342dfcb8e93dbc4550b87463e75d504dc6fe5026c950729e51d96f61addac381"),
        "stable_copy_scalar_count": 639,
        "stable_copy_scalar_set_sha256": (
            "cbab2504080696c42cf2fff70b42d1455aa58334a4f3ec2222c22f57230616fb"),
    },
    "LIBREOFFICE_PROJECT": {
        "conflict_input_count": 522,
        "identity_veto_input_count": 176,
        "route_count": 2_251,
        "route_identity_set_sha256": (
            "2bb93b133934c6499b1145903758a554cde9319231a5aa184a624d2ef4f386db"),
        "stable_copy_scalar_count": 656,
        "stable_copy_scalar_set_sha256": (
            "46f40b8b3e72f7492fe2816fed654f987b219283d4abd7a08f67a7fb7b026de5"),
    },
    "MICROSOFT_VSCODE_PROJECT": {
        "conflict_input_count": 1_852,
        "identity_veto_input_count": 434,
        "route_count": 8_063,
        "route_identity_set_sha256": (
            "ef2781b115d900cafb8ad836357b9869c29a33a21c506ffc53d44285f96e82ee"),
        "stable_copy_scalar_count": 757,
        "stable_copy_scalar_set_sha256": (
            "a35d06351d72fa8dee0c6df50031908c2826a0dca906dcfaa9e9ba1425eb0be2"),
    },
    "THUNDERBIRD_PROJECT": {
        "conflict_input_count": 383,
        "identity_veto_input_count": 147,
        "route_count": 1_736,
        "route_identity_set_sha256": (
            "95047ca9de948153349691247f345155d10d15b5730db7d45b71afd5f626253f"),
        "stable_copy_scalar_count": 573,
        "stable_copy_scalar_set_sha256": (
            "7c9885aba963c6d27487dd2f8390159682b286293e83809dc8e66df9053eafe9"),
    },
}
_EXPECTED_AUTHORITY_ROUTE_COUNTS = {
    "GODOT_ENGINE_PROJECT": 20,
    "LIBREOFFICE_PROJECT": 37,
    "MICROSOFT_VSCODE_PROJECT": 88,
    "THUNDERBIRD_PROJECT": 126,
}


def _sha256(payload: bytes) -> str:
    """返回规范 manifest SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v7 cross-source transformation root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入或输出位于显式 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"v7 cross-source transformation {label} path 越界")
    return path


def _validate_inputs(
        protocol: dict[str, object],
        variable: dict[str, object],
        context_local: dict[str, object],
        source_replay: dict[str, object],
        neutral_projection: dict[str, object],
        ) -> None:
    """核验五个 sealed predecessor 的状态与能力边界。"""
    variable_summary = variable.get("summary")
    variable_plans = variable_summary.get("plans") \
        if isinstance(variable_summary, dict) else None
    context_summary = context_local.get("summary")
    context_loso = context_summary.get("loso") \
        if isinstance(context_summary, dict) else None
    replay_summary = source_replay.get("summary")
    replay_loso = replay_summary.get("loso") \
        if isinstance(replay_summary, dict) else None
    neutral_summary = neutral_projection.get("summary")
    if (protocol.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_TRAINING_PROTOCOL_V1"
            or protocol.get("status") != "FROZEN_NOT_READ_NOT_LEARNED"
            or protocol.get("evaluation_or_held_out_payload_read_count") != 0
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
            or not isinstance(context_loso, dict)
            or context_loso.get("capability_outcome")
            != "NE_ZERO_VARIABLE_EXACT"
            or source_replay.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_SOURCE_REPLAY_AUDIT_V1"
            or source_replay.get("status")
            != "TRAIN_ONLY_SOURCE_REPLAY_NE_NOT_RUNTIME"
            or not isinstance(replay_loso, dict)
            or replay_loso.get("capability_outcome")
            != "NE_ZERO_UNSEEN_FAMILY_EXACT"
            or neutral_projection.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_"
            "NEUTRAL_UPSTREAM_SOURCE_PROJECTION_V1"
            or neutral_projection.get("status")
            != "TRAIN_ONLY_NEUTRAL_SOURCE_PROJECTION_"
            "PASS_NOT_CAPABILITY_NOT_RUNTIME"
            or neutral_projection.get("train_surface_published_in_audit") != 0
            or not isinstance(neutral_summary, dict)
            or neutral_summary.get("projection_record_count") != 33_746):
        raise BroadQaExternalDataError(
            "v7 cross-source transformation predecessor contract 漂移")


def _input_state(
        *,
        protocol_dir: Path,
        variable_dir: Path,
        context_local_dir: Path,
        source_replay_dir: Path,
        neutral_projection_dir: Path,
        ) -> tuple[
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, object],
        ]:
    """读取并核验五份 predecessor manifest。"""
    protocol = _read_manifest(
        protocol_dir,
        expected_sha256=V5_TRAINING_PROTOCOL_MANIFEST_SHA256,
        label="v5 protocol")
    variable = _read_manifest(
        variable_dir,
        expected_sha256=V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256,
        label="v7 variable structure")
    context_local = _read_manifest(
        context_local_dir,
        expected_sha256=V7_CONTEXT_LOCAL_AUDIT_MANIFEST_SHA256,
        label="v7 context local")
    source_replay = _read_manifest(
        source_replay_dir,
        expected_sha256=V7_SOURCE_REPLAY_AUDIT_MANIFEST_SHA256,
        label="v7 source replay")
    neutral_projection = _read_manifest(
        neutral_projection_dir,
        expected_sha256=V7_NEUTRAL_SOURCE_PROJECTION_MANIFEST_SHA256,
        label="v7 neutral source projection")
    _validate_inputs(
        protocol, variable, context_local,
        source_replay, neutral_projection)
    return (
        protocol, variable, context_local,
        source_replay, neutral_projection)


def _model_facts(
        models: tuple[dict[str, object], ...],
        ) -> dict[str, dict[str, object]]:
    """提取必须冻结的 family model identity。"""
    keys = (
        "conflict_input_count",
        "identity_veto_input_count",
        "route_count",
        "route_identity_set_sha256",
        "stable_copy_scalar_count",
        "stable_copy_scalar_set_sha256",
    )
    return {
        str(item["source_family"]): {
            key: item[key] for key in keys}
        for item in models}


def _derive(
        *,
        protocol_dir: Path,
        protocol: dict[str, object],
        variable_dir: Path,
        variable: dict[str, object],
        neutral_projection_dir: Path,
        neutral_projection: dict[str, object],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
        ]:
    """读取 TRAIN material 并派生固定 transformation feasibility。"""
    material = []
    for name, role in _PROTOCOL_FILES:
        material.append(_read_jsonl(
            protocol_dir / name,
            artifact=_file_artifact(
                protocol, relative_path=name, role=role),
            label=role,
        ))
    observations, fragments = material
    plans = _read_jsonl(
        variable_dir / _PLAN_FILE,
        artifact=_file_artifact(
            variable, relative_path=_PLAN_FILE, role=_PLAN_ROLE),
        label=_PLAN_ROLE,
    )
    projections = _read_jsonl(
        neutral_projection_dir / _PROJECTION_FILE,
        artifact=_file_artifact(
            neutral_projection,
            relative_path=_PROJECTION_FILE,
            role=_PROJECTION_ROLE),
        label=_PROJECTION_ROLE,
    )
    models, stages, loso, summary = (
        derive_cross_source_transformation_feasibility(
            observations=observations,
            fragments=fragments,
            plans=plans,
            neutral_projections=projections,
        ))
    authority_counts = {
        family: int(value["authority_route_count"])
        for family, value in summary[
            "neutral_authority_by_held_family"].items()}
    if (not strict_json_equal(_model_facts(models), _EXPECTED_MODEL_FACTS)
            or summary.get("capability_outcome")
            != "NE_ZERO_AUTHORIZED_EXACT"
            or summary.get("facility_outcome") != "PASS"
            or summary.get("variable_plan_count") != 3_460
            or summary.get("identity_probe_count") != 171
            or summary.get("indexed_reference_mismatch_count") != 0
            or summary.get("partial_commit_count") != 0
            or summary.get(
                "structure_token_execution_mismatch_count") != 0
            or summary.get("identity_false_change_count") != 0
            or summary.get("neutral_authorized_count") != 0
            or not strict_json_equal(
                summary.get("pre_authorization_outcome_counts"), {
                    "EXACT": 2, "UNKNOWN": 3_446, "WRONG": 12})
            or not strict_json_equal(
                summary.get("final_outcome_counts"), {
                    "EXACT": 0, "UNKNOWN": 3_460, "WRONG": 0})
            or not strict_json_equal(
                authority_counts, _EXPECTED_AUTHORITY_ROUTE_COUNTS)):
        raise BroadQaExternalDataError(
            "v7 cross-source transformation frozen feasibility 漂移")
    return {
        _OUTPUT_FILES[0][0]: models,
        _OUTPUT_FILES[1][0]: stages,
        _OUTPUT_FILES[2][0]: loso,
    }, {
        "audit_outcome": "FACILITY_PASS_CAPABILITY_NE",
        "transformation": summary,
    }


def _status(summary: dict[str, object]) -> str:
    """把 feasibility 两本账映射为诚实的非 runtime 状态。"""
    outcome = summary.get("audit_outcome")
    suffix = {
        "FACILITY_PASS_CAPABILITY_PASS": "PASS",
        "FACILITY_PASS_CAPABILITY_NE": "NE",
        "FACILITY_PASS_CAPABILITY_FAIL": "FAIL",
        "FACILITY_FAIL": "FACILITY_FAIL",
    }.get(str(outcome))
    if suffix is None:
        raise BroadQaExternalDataError(
            "v7 cross-source transformation audit outcome 非法")
    return f"TRAIN_ONLY_CROSS_SOURCE_TRANSFORMATION_{suffix}_NOT_RUNTIME"


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> dict[str, object]:
    """构造 cross-source transformation TRAIN-only manifest。"""
    return {
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V7_CROSS_SOURCE_TRANSFORMATION_AUDIT_KIND),
        "candidate_family_formal_run_count": 0,
        "files": files,
        "format_version": 1,
        "held_out_boundary": {
            "consumed_qt_individual_or_derivative_read_count": 0,
            "vlc_commitment_identity_raw_or_translation_read_count": 0,
        },
        "inputs": {
            "v5_training_protocol_manifest_sha256": (
                V5_TRAINING_PROTOCOL_MANIFEST_SHA256),
            "v7_context_local_audit_manifest_sha256": (
                V7_CONTEXT_LOCAL_AUDIT_MANIFEST_SHA256),
            "v7_neutral_source_projection_manifest_sha256": (
                V7_NEUTRAL_SOURCE_PROJECTION_MANIFEST_SHA256),
            "v7_source_replay_audit_manifest_sha256": (
                V7_SOURCE_REPLAY_AUDIT_MANIFEST_SHA256),
            "v7_variable_structure_audit_manifest_sha256": (
                V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256),
        },
        "learner_or_selection_change_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": _status(summary),
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_surface_published_in_audit": 0,
    }


def publish_normalization_recovery_v7_cross_source_transformation_audit(
        *,
        run_root: str | Path,
        training_protocol_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        context_local_audit_dir: str | Path,
        source_replay_audit_dir: str | Path,
        neutral_source_projection_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 cross-source transformation feasibility。"""
    root = _require_k_root(run_root)
    paths = [
        _within(root, value, label=label)
        for value, label in (
            (training_protocol_dir, "training protocol"),
            (variable_structure_audit_dir, "variable structure"),
            (context_local_audit_dir, "context local"),
            (source_replay_audit_dir, "source replay"),
            (neutral_source_projection_dir, "neutral projection"),
            (target_dir, "target"),
        )
    ]
    (protocol_dir, variable_dir, context_dir,
     replay_dir, neutral_dir, target) = paths
    inputs = paths[:-1]
    if (any(not path.is_dir() for path in inputs)
            or any(_overlap(target, path) for path in inputs)
            or target.exists()):
        raise BroadQaExternalDataError(
            "v7 cross-source transformation input/target path 非法")
    protocol, variable, _context, _replay, neutral = _input_state(
        protocol_dir=protocol_dir,
        variable_dir=variable_dir,
        context_local_dir=context_dir,
        source_replay_dir=replay_dir,
        neutral_projection_dir=neutral_dir,
    )
    outputs, summary = _derive(
        protocol_dir=protocol_dir,
        protocol=protocol,
        variable_dir=variable_dir,
        variable=variable,
        neutral_projection_dir=neutral_dir,
        neutral_projection=neutral,
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


def read_normalization_recovery_v7_cross_source_transformation_audit(
        audit_dir: str | Path,
        *,
        training_protocol_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        context_local_audit_dir: str | Path,
        source_replay_audit_dir: str | Path,
        neutral_source_projection_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """从 sealed inputs 重派生并严格回读 transformation feasibility。"""
    root = Path(audit_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v7 cross-source transformation manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v7 cross-source transformation manifest identity 漂移")
    protocol_dir = Path(training_protocol_dir).resolve()
    variable_dir = Path(variable_structure_audit_dir).resolve()
    context_dir = Path(context_local_audit_dir).resolve()
    replay_dir = Path(source_replay_audit_dir).resolve()
    neutral_dir = Path(neutral_source_projection_dir).resolve()
    protocol, variable, _context, _replay, neutral = _input_state(
        protocol_dir=protocol_dir,
        variable_dir=variable_dir,
        context_local_dir=context_dir,
        source_replay_dir=replay_dir,
        neutral_projection_dir=neutral_dir,
    )
    expected_outputs, summary = _derive(
        protocol_dir=protocol_dir,
        protocol=protocol,
        variable_dir=variable_dir,
        variable=variable,
        neutral_projection_dir=neutral_dir,
        neutral_projection=neutral,
    )
    stored_outputs = {
        name: _stored_jsonl(root / name, label=name)
        for name, _role in _OUTPUT_FILES
    }
    if any(not strict_json_equal(
            stored_outputs[name], expected_outputs[name])
           for name, _role in _OUTPUT_FILES):
        raise BroadQaExternalDataError(
            "v7 cross-source transformation records/inputs 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    if not strict_json_equal(stored, _manifest(files=files, summary=summary)):
        raise BroadQaExternalDataError(
            "v7 cross-source transformation manifest 字段漂移")
    return ({**stored, "manifest_sha256": expected_manifest_sha256},
            stored_outputs)


__all__ = [
    "NORMALIZATION_RECOVERY_V7_CROSS_SOURCE_TRANSFORMATION_AUDIT_KIND",
    "publish_normalization_recovery_v7_cross_source_transformation_audit",
    "read_normalization_recovery_v7_cross_source_transformation_audit",
]
