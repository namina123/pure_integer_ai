"""发布 recovery-v7 intent/semantic alignment TRAIN-only census。

该 artifact 从 sealed TRAIN observations、variable plans、neutral source packs 与
compact ConceptNet alias pack 重派生离散事实。它不发布 surface，不创建 learner、
candidate、runtime 或 formal evaluation，也不读取 VLC/Qt 个体数据。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_conceptnet_alias_audit
    as alias_audit,
)
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_intent_semantic_alignment_records import (
    SIGNATURE_ALIAS_LEXICAL,
    SIGNATURE_ALIAS_PUNCTUATION,
    SIGNATURE_POS_ONLY,
    SIGNATURE_STRUCTURE_ONLY,
    derive_intent_semantic_alignment_feasibility,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
    derive_neutral_upstream_source_rows,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_raw_snapshot import (
    sha256_path,
)


NORMALIZATION_RECOVERY_V7_INTENT_SEMANTIC_ALIGNMENT_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_"
    "INTENT_SEMANTIC_ALIGNMENT_FEASIBILITY_V1")

V5_TRAINING_PROTOCOL_MANIFEST_SHA256 = (
    "3385e340705af3dd75bd30980f35152574bd967aa257c6d789ee8142d0e87480")
V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256 = (
    "a2e40ec5a4950bd167e66100e2b999122ace83a6348aeeddf862ab0d39f75a3e")
V7_NEUTRAL_SOURCE_PROJECTION_MANIFEST_SHA256 = (
    "5de3fe9a077703a0b915c64c93a5650bed79c4a3757c819340018f748b84ca23")
V7_CROSS_SOURCE_TRANSFORMATION_MANIFEST_SHA256 = (
    "acbab185d7f2facada5e72d2d81f3348c398a86b04b6c103192f1a36b045ca8e")
V7_CONCEPTNET_ALIAS_MANIFEST_SHA256 = (
    "9b783ce9c5e1f3b2535158812c41bd901e75aa97b04595e67625fda982109160")

_PROTOCOL_FILES = (
    ("train.pair-observations.jsonl", "TRAIN_PAIR_OBSERVATIONS"),
    ("train.phrase-fragments.jsonl", "TRAIN_PHRASE_FRAGMENTS"),
)
_PLAN_FILE = ("structure-plans.jsonl", "VARIABLE_STRUCTURE_OBLIGATION_PLANS")
_PROJECTION_FILE = (
    "neutral-projections.jsonl", "NEUTRAL_SOURCE_COMMITMENT_PROJECTIONS")
_ALIAS_FILES = (
    ("alias-evidence.jsonl", "CONCEPTNET_NEUTRAL_ALIAS_EVIDENCE"),
    ("english-alias-routes.jsonl", "CONCEPTNET_NEUTRAL_ALIAS_ROUTES"),
    ("family-coverage.jsonl", "CONCEPTNET_NEUTRAL_ALIAS_FAMILY_COVERAGE"),
)
_OUTPUT_FILES = (
    ("fact-families.jsonl", "INTENT_SEMANTIC_FACT_FAMILIES"),
    ("family-census.jsonl", "INTENT_SEMANTIC_SOURCE_FAMILY_CENSUS"),
    ("signature-census.jsonl", "INTENT_SEMANTIC_SIGNATURE_CENSUS"),
    ("loso-census.jsonl", "INTENT_SEMANTIC_FOUR_SOURCE_LOSO"),
)

_EXPECTED_ALIAS = {
    "clean_alias_evidence_count": 44_713,
    "clean_alias_route_count": 3_683,
    "clean_phrase_inventory_count": 164_892,
    "multi_specified_pos_route_count": 318,
    "pair_count": 33_746,
    "single_specified_pos_route_count": 1_943,
    "specified_pos_route_count": 2_261,
    "specified_pos_unique_chinese_route_count": 118,
    "structure_filtered_phrase_count": 164_892,
    "unique_chinese_route_count": 433,
}
_EXPECTED_FAMILIES = {
    GODOT_SOURCE_FAMILY: (
        5_590, 45_928, 2_024, 5_362, 228, 2_864, 5_029, 1_948,
        1_498, 398, 604, 623, 3_107),
    LIBREOFFICE_SOURCE_FAMILY: (
        3_886, 41_074, 1_900, 3_659, 227, 1_973, 3_323, 1_418,
        1_538, 145, 146, 163, 2_121),
    VSCODE_SOURCE_FAMILY: (
        24_270, 85_523, 2_388, 23_601, 669, 11_298, 22_478, 11_024,
        3_520, 485, 4_467, 432, 12_918),
    THUNDERBIRD_SOURCE_FAMILY: (0,) * 13,
}
_EXPECTED_SIGNATURES = {
    SIGNATURE_ALIAS_PUNCTUATION: (
        27_417, 429, 87, 342, 2_922,
        "1b0a08d822253d35a587123957a6cb74b49555ee239bc72170c29a507a7e1603"),
    SIGNATURE_ALIAS_LEXICAL: (
        26_772, 470, 58, 412, 3_488,
        "546d82767f7d013c35130f7d2a1d5cfdf16436b8cfbe37c1f73e3e19947ea910"),
    SIGNATURE_POS_ONLY: (
        11_288, 782, 1, 781, 17_191,
        "f0f3b266446721b2b0d532bae6d6a07f8651f09d12bc9ef0c1156d2e2057e493"),
    SIGNATURE_STRUCTURE_ONLY: (
        3_460, 285, 0, 285, 22_412,
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"),
}
_EXPECTED_LOSO = {
    GODOT_SOURCE_FAMILY: (21, 1, 1, 1, 1),
    LIBREOFFICE_SOURCE_FAMILY: (45, 2, 2, 1, 2),
    VSCODE_SOURCE_FAMILY: (61, 11, 10, 2, 7),
    THUNDERBIRD_SOURCE_FAMILY: (87, 0, 0, 0, 0),
}


def _sha256(payload: bytes) -> str:
    """返回 canonical artifact SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式 run root 是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v7 intent semantic root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入、source 与输出位于显式 K 盘 root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"v7 intent semantic {label} path 越界")
    return path


def _validate_manifests(
        protocol: dict[str, object],
        variable: dict[str, object],
        neutral: dict[str, object],
        transformation: dict[str, object],
        alias: dict[str, object],
        ) -> None:
    """核验五份直接 predecessor 的能力与禁读边界。"""
    alias_inputs = alias.get("inputs")
    if (protocol.get("status") != "FROZEN_NOT_READ_NOT_LEARNED"
            or protocol.get("evaluation_or_held_out_payload_read_count") != 0
            or variable.get("status")
            != "TRAIN_ONLY_REPRESENTATION_PASS_CAPABILITY_NE_NOT_RUNTIME"
            or neutral.get("status")
            != "TRAIN_ONLY_NEUTRAL_SOURCE_PROJECTION_"
            "PASS_NOT_CAPABILITY_NOT_RUNTIME"
            or transformation.get("status")
            != "TRAIN_ONLY_CROSS_SOURCE_TRANSFORMATION_NE_NOT_RUNTIME"
            or alias.get("status")
            != "TRAIN_ONLY_CONCEPTNET_NEUTRAL_ALIAS_"
            "PASS_NOT_CAPABILITY_NOT_RUNTIME"
            or not isinstance(alias_inputs, dict)
            or alias_inputs.get(
                "v7_cross_source_transformation_manifest_sha256")
            != V7_CROSS_SOURCE_TRANSFORMATION_MANIFEST_SHA256
            or alias_inputs.get(
                "v7_neutral_source_projection_manifest_sha256")
            != V7_NEUTRAL_SOURCE_PROJECTION_MANIFEST_SHA256
            or alias.get("candidate_family_formal_run_count") != 0
            or alias.get("runtime_program_published") != 0):
        raise BroadQaExternalDataError(
            "v7 intent semantic predecessor contract 漂移")


def _input_state(
        *,
        protocol_dir: Path,
        variable_dir: Path,
        source_replay_dir: Path,
        neutral_dir: Path,
        transformation_dir: Path,
        alias_dir: Path,
        godot_dir: Path,
        libreoffice_dir: Path,
        vscode_dir: Path,
        thunderbird_dir: Path,
        snapshot_manifest_path: Path,
        conceptnet_raw_path: Path,
        ) -> tuple[
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, tuple[
                dict[str, object], tuple[dict[str, object], ...]]],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """严格核验 predecessor、raw identity、source packs 与 compact alias。"""
    protocol = _read_manifest(
        protocol_dir,
        expected_sha256=V5_TRAINING_PROTOCOL_MANIFEST_SHA256,
        label="v5 training protocol")
    variable = _read_manifest(
        variable_dir,
        expected_sha256=V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256,
        label="v7 variable structure")
    neutral = _read_manifest(
        neutral_dir,
        expected_sha256=V7_NEUTRAL_SOURCE_PROJECTION_MANIFEST_SHA256,
        label="v7 neutral source projection")
    transformation = _read_manifest(
        transformation_dir,
        expected_sha256=V7_CROSS_SOURCE_TRANSFORMATION_MANIFEST_SHA256,
        label="v7 cross-source transformation")
    alias = _read_manifest(
        alias_dir,
        expected_sha256=V7_CONCEPTNET_ALIAS_MANIFEST_SHA256,
        label="v7 ConceptNet alias")
    _validate_manifests(protocol, variable, neutral, transformation, alias)
    alias_audit._read_snapshot_manifest(snapshot_manifest_path)
    if (not conceptnet_raw_path.is_file()
            or sha256_path(conceptnet_raw_path)
            != alias_audit.CONCEPTNET_RAW_SHA256):
        raise BroadQaExternalDataError(
            "v7 intent semantic ConceptNet raw identity 漂移")
    sources = alias_audit._input_state(
        protocol_dir=protocol_dir,
        source_replay_dir=source_replay_dir,
        neutral_projection_dir=neutral_dir,
        transformation_dir=transformation_dir,
        godot_dir=godot_dir,
        libreoffice_dir=libreoffice_dir,
        vscode_dir=vscode_dir,
        thunderbird_dir=thunderbird_dir,
    )
    alias_outputs = {
        name: _read_jsonl(
            alias_dir / name,
            artifact=_file_artifact(
                alias, relative_path=name, role=role),
            label=role,
        )
        for name, role in _ALIAS_FILES
    }
    alias_audit._validate_stored_outputs(
        stored_outputs=alias_outputs, sources=sources)
    return (
        protocol, variable, neutral, transformation, alias,
        sources, alias_outputs)


def _family_facts(
        records: tuple[dict[str, object], ...],
        ) -> dict[str, tuple[int, ...]]:
    """提取需冻结的四来源 structure-aware census。"""
    keys = (
        "projected_pair_count",
        "clean_phrase_inventory_count",
        "clean_matched_alias_phrase_count",
        "clean_alias_pair_count",
        "zero_alias_pair_count",
        "complete_alias_coverage_pair_count",
        "specified_pos_pair_count",
        "all_alias_specified_pos_pair_count",
        "any_unique_chinese_alias_pair_count",
        "digit_source_pair_count",
        "observation_structure_pair_count",
        "source_structure_pair_count",
        "punctuated_source_pair_count",
    )
    return {
        str(item["source_family"]): tuple(int(item[key]) for key in keys)
        for item in records}


def _signature_facts(
        records: tuple[dict[str, object], ...],
        ) -> dict[str, tuple[object, ...]]:
    """提取四种离散 signature 的冻结计数与共识 identity。"""
    return {
        str(item["signature_mode"]): (
            int(item["signature_count"]),
            int(item["cross_family_signature_count"]),
            int(item["cross_family_consensus_count"]),
            int(item["cross_family_conflict_count"]),
            int(item["cross_family_pair_count"]),
            str(item["cross_family_consensus_set_sha256"]),
        ) for item in records
    }


def _loso_facts(
        records: tuple[dict[str, object], ...],
        ) -> dict[str, tuple[int, ...]]:
    """提取四向 proposal coverage 与 alias/POS 可用性。"""
    return {
        str(item["held_out_source_family"]): (
            int(item["authority_route_count"]),
            int(item["proposal_count"]),
            int(item["clean_alias_available_count"]),
            int(item["complete_alias_coverage_count"]),
            int(item["specified_pos_available_count"]),
        ) for item in records
    }


def _derive(
        *,
        protocol_dir: Path,
        protocol: dict[str, object],
        variable_dir: Path,
        variable: dict[str, object],
        neutral_dir: Path,
        neutral: dict[str, object],
        sources: dict[str, tuple[
            dict[str, object], tuple[dict[str, object], ...]]],
        alias_outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
        ]:
    """读取 compact TRAIN material，派生并冻结 intent/semantic census。"""
    observations, fragments = tuple(
        _read_jsonl(
            protocol_dir / name,
            artifact=_file_artifact(
                protocol, relative_path=name, role=role),
            label=role,
        ) for name, role in _PROTOCOL_FILES)
    plans = _read_jsonl(
        variable_dir / _PLAN_FILE[0],
        artifact=_file_artifact(
            variable, relative_path=_PLAN_FILE[0], role=_PLAN_FILE[1]),
        label=_PLAN_FILE[1],
    )
    projections = _read_jsonl(
        neutral_dir / _PROJECTION_FILE[0],
        artifact=_file_artifact(
            neutral, relative_path=_PROJECTION_FILE[0],
            role=_PROJECTION_FILE[1]),
        label=_PROJECTION_FILE[1],
    )
    rows = derive_neutral_upstream_source_rows(
        godot_manifest=sources[GODOT_SOURCE_FAMILY][0],
        godot_pairs=sources[GODOT_SOURCE_FAMILY][1],
        libreoffice_manifest=sources[LIBREOFFICE_SOURCE_FAMILY][0],
        libreoffice_pairs=sources[LIBREOFFICE_SOURCE_FAMILY][1],
        vscode_manifest=sources[VSCODE_SOURCE_FAMILY][0],
        vscode_pairs=sources[VSCODE_SOURCE_FAMILY][1],
        thunderbird_manifest=sources[THUNDERBIRD_SOURCE_FAMILY][0],
        thunderbird_pairs=sources[THUNDERBIRD_SOURCE_FAMILY][1],
    )
    fact, family, signature, loso, summary = (
        derive_intent_semantic_alignment_feasibility(
            observations=observations,
            fragments=fragments,
            plans=plans,
            neutral_projections=projections,
            rows_by_family=rows,
            alias_evidence=alias_outputs[_ALIAS_FILES[0][0]],
            alias_routes=alias_outputs[_ALIAS_FILES[1][0]],
        ))
    if (len(fact) != 11 or len(family) != 4
            or len(signature) != 4 or len(loso) != 4
            or not strict_json_equal(summary.get("alias"), _EXPECTED_ALIAS)
            or _family_facts(family) != _EXPECTED_FAMILIES
            or _signature_facts(signature) != _EXPECTED_SIGNATURES
            or _loso_facts(loso) != _EXPECTED_LOSO
            or summary.get("facility_outcome") != "PASS"
            or summary.get("representation_outcome")
            != "PARTIAL_NONZERO_ALIAS_FACT_SUPPORT"
            or summary.get("capability_outcome")
            != "NE_ZERO_AUTHORIZED_EXACT"
            or not strict_json_equal(summary.get("loso"), {
                "authority_route_available_count": 0,
                "authorized_count": 0,
                "clean_alias_available_count": 13,
                "complete_alias_coverage_count": 4,
                "final_outcome_counts": {
                    "EXACT": 0, "UNKNOWN": 14, "WRONG": 0},
                "pre_outcome_counts": {
                    "EXACT": 2, "UNKNOWN": 0, "WRONG": 12},
                "proposal_count": 14,
                "specified_pos_available_count": 10,
            })):
        raise BroadQaExternalDataError(
            "v7 intent semantic frozen census 漂移")
    return {
        _OUTPUT_FILES[0][0]: fact,
        _OUTPUT_FILES[1][0]: family,
        _OUTPUT_FILES[2][0]: signature,
        _OUTPUT_FILES[3][0]: loso,
    }, {
        "alignment": summary,
        "audit_outcome": "FACILITY_PASS_REPRESENTATION_PARTIAL_CAPABILITY_NE",
        "raw_input_output_or_source_surface_published": 0,
    }


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> dict[str, object]:
    """构造 intent/semantic TRAIN-only artifact manifest。"""
    return {
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V7_INTENT_SEMANTIC_ALIGNMENT_AUDIT_KIND),
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
            "v7_conceptnet_alias_manifest_sha256": (
                V7_CONCEPTNET_ALIAS_MANIFEST_SHA256),
            "v7_cross_source_transformation_manifest_sha256": (
                V7_CROSS_SOURCE_TRANSFORMATION_MANIFEST_SHA256),
            "v7_neutral_source_projection_manifest_sha256": (
                V7_NEUTRAL_SOURCE_PROJECTION_MANIFEST_SHA256),
            "v7_variable_structure_audit_manifest_sha256": (
                V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256),
        },
        "learner_or_selection_change_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": "TRAIN_ONLY_INTENT_SEMANTIC_ALIGNMENT_NE_NOT_RUNTIME",
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_surface_published_in_audit": 0,
    }


def _paths(
        *,
        root: Path | None,
        values: tuple[tuple[str | Path, str], ...],
        ) -> tuple[Path, ...]:
    """统一解析 publisher 或 reader 的全部输入路径。"""
    return tuple(
        _within(root, value, label=label) if root is not None
        else Path(value).resolve()
        for value, label in values)


def publish_normalization_recovery_v7_intent_semantic_alignment_audit(
        *,
        run_root: str | Path,
        training_protocol_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        source_replay_audit_dir: str | Path,
        neutral_source_projection_dir: str | Path,
        cross_source_transformation_dir: str | Path,
        conceptnet_alias_audit_dir: str | Path,
        godot_source_pack_dir: str | Path,
        libreoffice_source_pack_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        thunderbird_source_pack_dir: str | Path,
        conceptnet_snapshot_manifest_path: str | Path,
        conceptnet_raw_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 structure-aware intent/semantic feasibility census。"""
    root = _require_k_root(run_root)
    values = (
        (training_protocol_dir, "training protocol"),
        (variable_structure_audit_dir, "variable structure"),
        (source_replay_audit_dir, "source replay"),
        (neutral_source_projection_dir, "neutral projection"),
        (cross_source_transformation_dir, "cross transformation"),
        (conceptnet_alias_audit_dir, "ConceptNet alias"),
        (godot_source_pack_dir, "Godot source"),
        (libreoffice_source_pack_dir, "LibreOffice source"),
        (vscode_source_pack_dir, "VS Code source"),
        (thunderbird_source_pack_dir, "Thunderbird source"),
        (target_dir, "target"),
    )
    paths = _paths(root=root, values=values)
    (*directories, target) = paths
    snapshot_path = Path(conceptnet_snapshot_manifest_path).resolve()
    raw_path = Path(conceptnet_raw_path).resolve()
    if (any(not path.is_dir() for path in directories)
            or not snapshot_path.is_file() or not raw_path.is_file()
            or any(_overlap(target, path) for path in directories)
            or _overlap(target, snapshot_path)
            or _overlap(target, raw_path)
            or target.exists()):
        raise BroadQaExternalDataError(
            "v7 intent semantic input/target path 非法")
    input_state = _input_state(
        protocol_dir=paths[0], variable_dir=paths[1],
        source_replay_dir=paths[2], neutral_dir=paths[3],
        transformation_dir=paths[4], alias_dir=paths[5],
        godot_dir=paths[6], libreoffice_dir=paths[7],
        vscode_dir=paths[8], thunderbird_dir=paths[9],
        snapshot_manifest_path=snapshot_path,
        conceptnet_raw_path=raw_path,
    )
    outputs, summary = _derive(
        protocol_dir=paths[0], protocol=input_state[0],
        variable_dir=paths[1], variable=input_state[1],
        neutral_dir=paths[3], neutral=input_state[2],
        sources=input_state[5], alias_outputs=input_state[6],
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


def read_normalization_recovery_v7_intent_semantic_alignment_audit(
        audit_dir: str | Path,
        *,
        training_protocol_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        source_replay_audit_dir: str | Path,
        neutral_source_projection_dir: str | Path,
        cross_source_transformation_dir: str | Path,
        conceptnet_alias_audit_dir: str | Path,
        godot_source_pack_dir: str | Path,
        libreoffice_source_pack_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        thunderbird_source_pack_dir: str | Path,
        conceptnet_snapshot_manifest_path: str | Path,
        conceptnet_raw_path: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """从 sealed inputs 重派生并严格回读 intent/semantic census。"""
    root = Path(audit_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v7 intent semantic manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v7 intent semantic manifest identity 漂移")
    values = (
        (training_protocol_dir, "training protocol"),
        (variable_structure_audit_dir, "variable structure"),
        (source_replay_audit_dir, "source replay"),
        (neutral_source_projection_dir, "neutral projection"),
        (cross_source_transformation_dir, "cross transformation"),
        (conceptnet_alias_audit_dir, "ConceptNet alias"),
        (godot_source_pack_dir, "Godot source"),
        (libreoffice_source_pack_dir, "LibreOffice source"),
        (vscode_source_pack_dir, "VS Code source"),
        (thunderbird_source_pack_dir, "Thunderbird source"),
        (conceptnet_snapshot_manifest_path, "ConceptNet snapshot"),
        (conceptnet_raw_path, "ConceptNet raw"),
    )
    paths = _paths(root=None, values=values)
    input_state = _input_state(
        protocol_dir=paths[0], variable_dir=paths[1],
        source_replay_dir=paths[2], neutral_dir=paths[3],
        transformation_dir=paths[4], alias_dir=paths[5],
        godot_dir=paths[6], libreoffice_dir=paths[7],
        vscode_dir=paths[8], thunderbird_dir=paths[9],
        snapshot_manifest_path=paths[10], conceptnet_raw_path=paths[11],
    )
    expected_outputs, summary = _derive(
        protocol_dir=paths[0], protocol=input_state[0],
        variable_dir=paths[1], variable=input_state[1],
        neutral_dir=paths[3], neutral=input_state[2],
        sources=input_state[5], alias_outputs=input_state[6],
    )
    stored_outputs = {
        name: _stored_jsonl(root / name, label=name)
        for name, _role in _OUTPUT_FILES
    }
    if any(not strict_json_equal(
            stored_outputs[name], expected_outputs[name])
           for name, _role in _OUTPUT_FILES):
        raise BroadQaExternalDataError(
            "v7 intent semantic records/inputs 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    if not strict_json_equal(stored, _manifest(files=files, summary=summary)):
        raise BroadQaExternalDataError(
            "v7 intent semantic manifest 字段漂移")
    return ({**stored, "manifest_sha256": expected_manifest_sha256},
            stored_outputs)


__all__ = [
    "NORMALIZATION_RECOVERY_V7_INTENT_SEMANTIC_ALIGNMENT_AUDIT_KIND",
    "publish_normalization_recovery_v7_intent_semantic_alignment_audit",
    "read_normalization_recovery_v7_intent_semantic_alignment_audit",
]
