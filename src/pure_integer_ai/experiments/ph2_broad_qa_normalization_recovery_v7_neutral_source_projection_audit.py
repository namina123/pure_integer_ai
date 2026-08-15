"""发布 recovery-v7 neutral upstream source projection TRAIN-only artifact。

publisher 只读取四份冻结 TRAIN source pack，以及 v5 TRAIN protocol 和 v7
source replay 的 manifest。artifact 不发布任何 source/input/output 原文，不创建
learner、candidate、runtime 或 formal evaluation；reader 从同一组输入重派生并
逐字段核验全部记录。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_godot_source_pack import (
    read_normalization_recovery_v3_godot_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_thunderbird_source_pack import (
    read_normalization_recovery_v3_thunderbird_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_vscode_source_pack import (
    read_normalization_recovery_v4_vscode_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_libreoffice_source_pack import (
    read_normalization_recovery_v5_libreoffice_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    strict_json_equal,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_protocol import (
    V5_GODOT_SOURCE_PACK_MANIFEST_SHA256,
    V5_LIBREOFFICE_SOURCE_PACK_MANIFEST_SHA256,
    V5_THUNDERBIRD_SOURCE_PACK_MANIFEST_SHA256,
    V5_VSCODE_SOURCE_PACK_MANIFEST_SHA256,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
    derive_neutral_upstream_source_projection_records,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V7_NEUTRAL_SOURCE_PROJECTION_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_"
    "NEUTRAL_UPSTREAM_SOURCE_PROJECTION_V1")
NORMALIZATION_RECOVERY_V7_NEUTRAL_SOURCE_PROJECTION_STATUS = (
    "TRAIN_ONLY_NEUTRAL_SOURCE_PROJECTION_PASS_NOT_CAPABILITY_NOT_RUNTIME")

V5_TRAINING_PROTOCOL_MANIFEST_SHA256 = (
    "3385e340705af3dd75bd30980f35152574bd967aa257c6d789ee8142d0e87480")
V7_SOURCE_REPLAY_AUDIT_MANIFEST_SHA256 = (
    "4fad3ca859d1b7349de4f5566e60fecb029a1751b62dba799b00ee589ff35b2a")

_OUTPUT_FILES = (
    ("family-projections.jsonl", "NEUTRAL_SOURCE_FAMILY_PROJECTIONS"),
    ("neutral-projections.jsonl", "NEUTRAL_SOURCE_COMMITMENT_PROJECTIONS"),
    ("cross-family-support.jsonl", "NEUTRAL_SOURCE_CROSS_FAMILY_SUPPORT"),
)
_EXPECTED_SOURCE_PACKS = {
    GODOT_SOURCE_FAMILY: V5_GODOT_SOURCE_PACK_MANIFEST_SHA256,
    LIBREOFFICE_SOURCE_FAMILY: V5_LIBREOFFICE_SOURCE_PACK_MANIFEST_SHA256,
    VSCODE_SOURCE_FAMILY: V5_VSCODE_SOURCE_PACK_MANIFEST_SHA256,
    THUNDERBIRD_SOURCE_FAMILY: V5_THUNDERBIRD_SOURCE_PACK_MANIFEST_SHA256,
}
_EXPECTED_PAIRWISE = {
    f"{GODOT_SOURCE_FAMILY}__{LIBREOFFICE_SOURCE_FAMILY}": {
        "all_families_unique_output_count": 115,
        "common_exact_surface_count": 126,
        "conflict_count": 27,
        "consensus_all_variable_count": 4,
        "consensus_any_structured_count": 0,
        "consensus_any_variable_count": 18,
        "consensus_count": 88,
        "record_comparison_count": 294,
        "same_output_record_comparison_count": 209,
    },
    f"{GODOT_SOURCE_FAMILY}__{VSCODE_SOURCE_FAMILY}": {
        "all_families_unique_output_count": 44,
        "common_exact_surface_count": 47,
        "conflict_count": 7,
        "consensus_all_variable_count": 1,
        "consensus_any_structured_count": 0,
        "consensus_any_variable_count": 4,
        "consensus_count": 37,
        "record_comparison_count": 74,
        "same_output_record_comparison_count": 63,
    },
    f"{LIBREOFFICE_SOURCE_FAMILY}__{VSCODE_SOURCE_FAMILY}": {
        "all_families_unique_output_count": 23,
        "common_exact_surface_count": 24,
        "conflict_count": 3,
        "consensus_all_variable_count": 1,
        "consensus_any_structured_count": 0,
        "consensus_any_variable_count": 1,
        "consensus_count": 20,
        "record_comparison_count": 92,
        "same_output_record_comparison_count": 87,
    },
}
_EXPECTED_THREE_FAMILY = {
    "all_families_unique_output_count": 9,
    "common_exact_surface_count": 10,
    "conflict_count": 0,
    "consensus_all_variable_count": 0,
    "consensus_any_structured_count": 0,
    "consensus_any_variable_count": 0,
    "consensus_count": 9,
    "consensus_identity_set_sha256": (
        "3469b61d5ef73775a5f1b4c8fa0369cb46a75d40c8e6c014e31d8320a41ea46e"),
}


def _sha256(payload: bytes) -> str:
    """返回规范文件的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v7 neutral source projection root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入或输出位于显式 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"v7 neutral source projection {label} path 越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个已解析路径是否互为祖先或同一路径。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def _read_manifest(
        root: Path,
        *,
        expected_sha256: str,
        label: str,
        ) -> dict[str, object]:
    """只读取并核验一个 predecessor manifest。"""
    try:
        encoded = (root / "manifest.json").read_bytes()
        value = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(f"{label} manifest 不可读") from error
    if (_sha256(encoded) != expected_sha256
            or not isinstance(value, dict)
            or canonical_json_line(value) != encoded):
        raise BroadQaExternalDataError(f"{label} manifest identity 漂移")
    return value


def _validate_predecessors(
        protocol: dict[str, object],
        replay: dict[str, object],
        ) -> None:
    """核验 TRAIN-only predecessor 状态与禁止读取边界。"""
    if (protocol.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_TRAINING_PROTOCOL_V1"
            or protocol.get("status") != "FROZEN_NOT_READ_NOT_LEARNED"
            or protocol.get("evaluation_or_held_out_payload_read_count") != 0
            or not strict_json_equal(
                protocol.get("source_pack_manifests"),
                _EXPECTED_SOURCE_PACKS)
            or replay.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_SOURCE_REPLAY_AUDIT_V1"
            or replay.get("status")
            != "TRAIN_ONLY_SOURCE_REPLAY_NE_NOT_RUNTIME"
            or replay.get("train_surface_published_in_audit") != 0
            or replay.get("runtime_program_published") != 0
            or not isinstance(replay.get("inputs"), dict)
            or replay["inputs"].get("v5_training_protocol_manifest_sha256")
            != V5_TRAINING_PROTOCOL_MANIFEST_SHA256):
        raise BroadQaExternalDataError(
            "v7 neutral source projection predecessor contract 漂移")


def _source_material(
        *,
        godot_dir: Path,
        libreoffice_dir: Path,
        vscode_dir: Path,
        thunderbird_dir: Path,
        ) -> dict[str, tuple[
            dict[str, object], tuple[dict[str, object], ...]]]:
    """严格回读四套冻结 TRAIN source pack。"""
    godot, _files, godot_pairs = (
        read_normalization_recovery_v3_godot_source_pack(godot_dir))
    libreoffice, _files, libreoffice_pairs = (
        read_normalization_recovery_v5_libreoffice_source_pack(
            libreoffice_dir))
    vscode, _files, vscode_pairs = (
        read_normalization_recovery_v4_vscode_source_pack(vscode_dir))
    thunderbird, _files, thunderbird_pairs = (
        read_normalization_recovery_v3_thunderbird_source_pack(
            thunderbird_dir))
    values = {
        GODOT_SOURCE_FAMILY: (godot, godot_pairs),
        LIBREOFFICE_SOURCE_FAMILY: (libreoffice, libreoffice_pairs),
        VSCODE_SOURCE_FAMILY: (vscode, vscode_pairs),
        THUNDERBIRD_SOURCE_FAMILY: (thunderbird, thunderbird_pairs),
    }
    if any(values[family][0].get("manifest_sha256") != expected
           for family, expected in _EXPECTED_SOURCE_PACKS.items()):
        raise BroadQaExternalDataError(
            "v7 neutral source projection source pack identity 漂移")
    return values


def _input_state(
        *,
        protocol_dir: Path,
        source_replay_dir: Path,
        godot_dir: Path,
        libreoffice_dir: Path,
        vscode_dir: Path,
        thunderbird_dir: Path,
        ) -> tuple[
            dict[str, object],
            dict[str, object],
            dict[str, tuple[
                dict[str, object], tuple[dict[str, object], ...]]],
        ]:
    """读取两份 predecessor manifest 与四套 source pack。"""
    protocol = _read_manifest(
        protocol_dir,
        expected_sha256=V5_TRAINING_PROTOCOL_MANIFEST_SHA256,
        label="v5 training protocol",
    )
    replay = _read_manifest(
        source_replay_dir,
        expected_sha256=V7_SOURCE_REPLAY_AUDIT_MANIFEST_SHA256,
        label="v7 source replay",
    )
    _validate_predecessors(protocol, replay)
    sources = _source_material(
        godot_dir=godot_dir,
        libreoffice_dir=libreoffice_dir,
        vscode_dir=vscode_dir,
        thunderbird_dir=thunderbird_dir,
    )
    return protocol, replay, sources


def _derive(
        sources: dict[str, tuple[
            dict[str, object], tuple[dict[str, object], ...]]],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
        ]:
    """派生 projection records，并冻结 section-80 全部计数。"""
    family, projections, support, summary = (
        derive_neutral_upstream_source_projection_records(
            godot_manifest=sources[GODOT_SOURCE_FAMILY][0],
            godot_pairs=sources[GODOT_SOURCE_FAMILY][1],
            libreoffice_manifest=sources[LIBREOFFICE_SOURCE_FAMILY][0],
            libreoffice_pairs=sources[LIBREOFFICE_SOURCE_FAMILY][1],
            vscode_manifest=sources[VSCODE_SOURCE_FAMILY][0],
            vscode_pairs=sources[VSCODE_SOURCE_FAMILY][1],
            thunderbird_manifest=sources[THUNDERBIRD_SOURCE_FAMILY][0],
            thunderbird_pairs=sources[THUNDERBIRD_SOURCE_FAMILY][1],
        ))
    family_by_source = {
        str(item["source_family"]): item for item in family}
    support_counts = Counter(
        int(item["support_family_count"]) for item in support)
    if (len(family_by_source) != 4
            or summary.get("projection_record_count") != 33_746
            or summary.get("cross_family_support_record_count") != 177
            or summary.get("vscode_ascii_letter_leaf_count") != 24_270
            or summary.get("vscode_sentence_like_leaf_count") != 23_023
            or summary.get("surface_commitment_collision_count") != 0
            or summary.get(
                "raw_input_output_or_source_surface_published") != 0
            or not strict_json_equal(
                summary.get("source_family_projection_counts"), {
                    GODOT_SOURCE_FAMILY: 5_590,
                    LIBREOFFICE_SOURCE_FAMILY: 3_886,
                    VSCODE_SOURCE_FAMILY: 24_270,
                    THUNDERBIRD_SOURCE_FAMILY: 0,
                })
            or family_by_source.get(THUNDERBIRD_SOURCE_FAMILY, {}).get(
                "neutral_surface_availability")
            != "NEUTRAL_SURFACE_UNAVAILABLE"
            or not strict_json_equal(
                summary.get("pairwise_exact_key_overlap"),
                _EXPECTED_PAIRWISE)
            or not strict_json_equal(
                summary.get("three_family_exact_key_overlap"),
                _EXPECTED_THREE_FAMILY)
            or dict(sorted(support_counts.items())) != {2: 167, 3: 10}):
        raise BroadQaExternalDataError(
            "v7 neutral source projection frozen census 漂移")
    return {
        _OUTPUT_FILES[0][0]: family,
        _OUTPUT_FILES[1][0]: projections,
        _OUTPUT_FILES[2][0]: support,
    }, {
        **summary,
        "capability_claimed": 0,
        "projection_outcome": "PASS",
        "support_family_count_counts": {
            str(key): value for key, value in sorted(support_counts.items())},
    }


def _write_jsonl(
        path: Path,
        values: tuple[dict[str, object], ...],
        ) -> None:
    """不可覆盖写入规范 JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """严格读取规范 JSONL。"""
    try:
        payload = path.read_bytes()
        lines = payload.splitlines(keepends=True)
        values = tuple(json.loads(line) for line in lines)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(f"{label} 不可读") from error
    if (not lines or b"".join(lines) != payload
            or any(not isinstance(item, dict) for item in values)
            or b"".join(canonical_json_line(item) for item in values)
            != payload):
        raise BroadQaExternalDataError(f"{label} 非规范 JSONL")
    return values


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """形成一个输出文件的物理承诺。"""
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
    """构造无 capability/runtime 声明的 projection manifest。"""
    return {
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V7_NEUTRAL_SOURCE_PROJECTION_AUDIT_KIND),
        "candidate_family_formal_run_count": 0,
        "files": files,
        "format_version": 1,
        "held_out_boundary": {
            "consumed_qt_individual_or_derivative_read_count": 0,
            "vlc_commitment_identity_raw_or_translation_read_count": 0,
        },
        "inputs": {
            "source_pack_manifests": _EXPECTED_SOURCE_PACKS,
            "v5_training_protocol_manifest_sha256": (
                V5_TRAINING_PROTOCOL_MANIFEST_SHA256),
            "v7_source_replay_audit_manifest_sha256": (
                V7_SOURCE_REPLAY_AUDIT_MANIFEST_SHA256),
        },
        "learner_or_selection_change_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "source_pack_read_count": 4,
        "status": (
            NORMALIZATION_RECOVERY_V7_NEUTRAL_SOURCE_PROJECTION_STATUS),
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_surface_published_in_audit": 0,
    }


def publish_normalization_recovery_v7_neutral_source_projection_audit(
        *,
        run_root: str | Path,
        training_protocol_dir: str | Path,
        source_replay_audit_dir: str | Path,
        godot_source_pack_dir: str | Path,
        libreoffice_source_pack_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        thunderbird_source_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 neutral upstream source projection artifact。"""
    root = _require_k_root(run_root)
    paths = [
        _within(root, value, label=label)
        for value, label in (
            (training_protocol_dir, "training protocol"),
            (source_replay_audit_dir, "source replay"),
            (godot_source_pack_dir, "Godot source pack"),
            (libreoffice_source_pack_dir, "LibreOffice source pack"),
            (vscode_source_pack_dir, "VS Code source pack"),
            (thunderbird_source_pack_dir, "Thunderbird source pack"),
            (target_dir, "target"),
        )
    ]
    (protocol_dir, replay_dir, godot_dir, libreoffice_dir,
     vscode_dir, thunderbird_dir, target) = paths
    inputs = paths[:-1]
    if (any(not path.is_dir() for path in inputs)
            or any(_overlap(target, path) for path in inputs)
            or target.exists()):
        raise BroadQaExternalDataError(
            "v7 neutral source projection input/target path 非法")
    _protocol, _replay, sources = _input_state(
        protocol_dir=protocol_dir,
        source_replay_dir=replay_dir,
        godot_dir=godot_dir,
        libreoffice_dir=libreoffice_dir,
        vscode_dir=vscode_dir,
        thunderbird_dir=thunderbird_dir,
    )
    outputs, summary = _derive(sources)
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


def read_normalization_recovery_v7_neutral_source_projection_audit(
        audit_dir: str | Path,
        *,
        training_protocol_dir: str | Path,
        source_replay_audit_dir: str | Path,
        godot_source_pack_dir: str | Path,
        libreoffice_source_pack_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        thunderbird_source_pack_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """从 sealed inputs 重派生并严格回读 projection artifact。"""
    root = Path(audit_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v7 neutral source projection manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v7 neutral source projection manifest identity 漂移")
    _protocol, _replay, sources = _input_state(
        protocol_dir=Path(training_protocol_dir).resolve(),
        source_replay_dir=Path(source_replay_audit_dir).resolve(),
        godot_dir=Path(godot_source_pack_dir).resolve(),
        libreoffice_dir=Path(libreoffice_source_pack_dir).resolve(),
        vscode_dir=Path(vscode_source_pack_dir).resolve(),
        thunderbird_dir=Path(thunderbird_source_pack_dir).resolve(),
    )
    expected_outputs, summary = _derive(sources)
    stored_outputs = {
        name: _read_jsonl(root / name, label=name)
        for name, _role in _OUTPUT_FILES
    }
    if any(not strict_json_equal(
            stored_outputs[name], expected_outputs[name])
           for name, _role in _OUTPUT_FILES):
        raise BroadQaExternalDataError(
            "v7 neutral source projection records/inputs 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    if not strict_json_equal(stored, _manifest(files=files, summary=summary)):
        raise BroadQaExternalDataError(
            "v7 neutral source projection manifest 字段漂移")
    return ({**stored, "manifest_sha256": expected_manifest_sha256},
            stored_outputs)


__all__ = [
    "NORMALIZATION_RECOVERY_V7_NEUTRAL_SOURCE_PROJECTION_AUDIT_KIND",
    "NORMALIZATION_RECOVERY_V7_NEUTRAL_SOURCE_PROJECTION_STATUS",
    "publish_normalization_recovery_v7_neutral_source_projection_audit",
    "read_normalization_recovery_v7_neutral_source_projection_audit",
]
