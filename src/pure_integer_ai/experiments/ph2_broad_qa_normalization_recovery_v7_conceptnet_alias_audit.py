"""发布 recovery-v7 neutral source ConceptNet alias TRAIN-only artifact。

publisher 从四份冻结 TRAIN source pack 重建 neutral phrase inventory，并单遍扫描
已冻结 ConceptNet raw snapshot，只物化命中 inventory 的英中 ``/r/Synonym``。
reader 不重复 3,400 万行扫描；它核验 raw/snapshot identity，并从 sealed source packs
重建 inventory、由逐 assertion evidence 独立重聚合 routes 与 family coverage。
"""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    strict_json_equal,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_conceptnet_alias_records import (
    conceptnet_alias_evidence_record,
    derive_conceptnet_alias_routes,
    derive_neutral_phrase_inventory,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_audit import (
    _artifact,
    _input_state as _projection_input_state,
    _overlap,
    _read_jsonl,
    _read_manifest,
    _within,
    _write_jsonl,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
    _godot_rows,
    _libreoffice_rows,
    _vscode_rows,
)
from pure_integer_ai.experiments.ph2_conceptnet_adapter import (
    ConceptNetAdapterError,
    parse_conceptnet_assertion,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_raw_snapshot import (
    sha256_path,
)


NORMALIZATION_RECOVERY_V7_CONCEPTNET_ALIAS_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_"
    "CONCEPTNET_NEUTRAL_ALIAS_V1")
NORMALIZATION_RECOVERY_V7_CONCEPTNET_ALIAS_STATUS = (
    "TRAIN_ONLY_CONCEPTNET_NEUTRAL_ALIAS_PASS_NOT_CAPABILITY_NOT_RUNTIME")

CONCEPTNET_SNAPSHOT_MANIFEST_SHA256 = (
    "36195b2eee75bc3ef96722ffe3ed67de58e70c5067c00d4369db687dc3893e75")
CONCEPTNET_RAW_SHA256 = (
    "accd65fe94038584295574ddc26e1500c1919c8c4532bf771811cafd0948af7e")
V7_NEUTRAL_SOURCE_PROJECTION_MANIFEST_SHA256 = (
    "5de3fe9a077703a0b915c64c93a5650bed79c4a3757c819340018f748b84ca23")
V7_CROSS_SOURCE_TRANSFORMATION_MANIFEST_SHA256 = (
    "acbab185d7f2facada5e72d2d81f3348c398a86b04b6c103192f1a36b045ca8e")

_EXPECTED_SCAN = {
    "conceptnet_en_zh_synonym_count": 382_472,
    "conceptnet_line_count": 34_074_917,
    "conceptnet_synonym_count": 6_702_880,
    "matching_parse_anomaly_count": 0,
}
_EXPECTED_ALIAS = {
    "alias_evidence_count": 44_716,
    "ambiguous_english_route_count": 3_251,
    "english_route_count": 3_685,
    "matched_english_chinese_pair_count": 39_715,
    "matched_english_phrase_count": 3_685,
    "neutral_phrase_inventory_count": 169_833,
    "unique_english_route_count": 434,
}
_EXPECTED_LICENSE_COUNTS = {
    "CC-BY-4.0": 3_074,
    "CC-BY-SA-4.0": 41_642,
}
_EXPECTED_FAMILY_COVERAGE = {
    GODOT_SOURCE_FAMILY: {
        "matched_neutral_phrase_count": 2_026,
        "pair_any_alias_count": 5_382,
        "projected_pair_count": 5_590,
    },
    LIBREOFFICE_SOURCE_FAMILY: {
        "matched_neutral_phrase_count": 1_902,
        "pair_any_alias_count": 3_664,
        "projected_pair_count": 3_886,
    },
    VSCODE_SOURCE_FAMILY: {
        "matched_neutral_phrase_count": 2_388,
        "pair_any_alias_count": 23_640,
        "projected_pair_count": 24_270,
    },
    THUNDERBIRD_SOURCE_FAMILY: {
        "matched_neutral_phrase_count": 0,
        "pair_any_alias_count": 0,
        "projected_pair_count": 0,
    },
}
_OUTPUT_FILES = (
    ("alias-evidence.jsonl", "CONCEPTNET_NEUTRAL_ALIAS_EVIDENCE"),
    ("english-alias-routes.jsonl", "CONCEPTNET_NEUTRAL_ALIAS_ROUTES"),
    ("family-coverage.jsonl", "CONCEPTNET_NEUTRAL_ALIAS_FAMILY_COVERAGE"),
)


def _sha256(payload: bytes) -> str:
    """返回规范文件或 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式 artifact 工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias run root 必须是 K 盘目录")
    return root


def _read_snapshot_manifest(path: Path) -> dict[str, object]:
    """严格回读公开 ConceptNet raw snapshot manifest。"""
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias snapshot manifest 不可读") from error
    if (_sha256(encoded) != CONCEPTNET_SNAPSHOT_MANIFEST_SHA256
            or not isinstance(value, dict)
            or canonical_json_line(value) != encoded
            or value.get("source_key") != "CONCEPTNET_5_7_0"
            or value.get("snapshot_id") != "5.7.0"
            or value.get("raw_sha256") != CONCEPTNET_RAW_SHA256
            or value.get("parsed_record_count") != 34_074_915
            or value.get("release_eligible") != 1
            or value.get("redistribution_policy") != "PUBLIC"):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias snapshot manifest 漂移")
    return value


def _validate_predecessors(
        projection: dict[str, object],
        transformation: dict[str, object],
        ) -> None:
    """核验 projection PASS 与 transformation capability NE 边界。"""
    transformation_summary = transformation.get("summary")
    transformation_facts = transformation_summary.get("transformation") \
        if isinstance(transformation_summary, dict) else None
    if (projection.get("status")
            != "TRAIN_ONLY_NEUTRAL_SOURCE_PROJECTION_"
            "PASS_NOT_CAPABILITY_NOT_RUNTIME"
            or projection.get("train_surface_published_in_audit") != 0
            or transformation.get("status")
            != "TRAIN_ONLY_CROSS_SOURCE_TRANSFORMATION_NE_NOT_RUNTIME"
            or not isinstance(transformation_facts, dict)
            or transformation_facts.get("facility_outcome") != "PASS"
            or transformation_facts.get("capability_outcome")
            != "NE_ZERO_AUTHORIZED_EXACT"
            or transformation_facts.get("final_outcome_counts")
            != {"EXACT": 0, "UNKNOWN": 3_460, "WRONG": 0}):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias predecessor contract 漂移")


def _input_state(
        *,
        protocol_dir: Path,
        source_replay_dir: Path,
        neutral_projection_dir: Path,
        transformation_dir: Path,
        godot_dir: Path,
        libreoffice_dir: Path,
        vscode_dir: Path,
        thunderbird_dir: Path,
        ) -> dict[str, tuple[
            dict[str, object], tuple[dict[str, object], ...]]]:
    """严格回读两份 predecessor 与四套 TRAIN source pack。"""
    _protocol, _replay, sources = _projection_input_state(
        protocol_dir=protocol_dir,
        source_replay_dir=source_replay_dir,
        godot_dir=godot_dir,
        libreoffice_dir=libreoffice_dir,
        vscode_dir=vscode_dir,
        thunderbird_dir=thunderbird_dir,
    )
    projection = _read_manifest(
        neutral_projection_dir,
        expected_sha256=V7_NEUTRAL_SOURCE_PROJECTION_MANIFEST_SHA256,
        label="v7 neutral source projection",
    )
    transformation = _read_manifest(
        transformation_dir,
        expected_sha256=V7_CROSS_SOURCE_TRANSFORMATION_MANIFEST_SHA256,
        label="v7 cross-source transformation",
    )
    _validate_predecessors(projection, transformation)
    return sources


def _neutral_material(
        sources: dict[str, tuple[
            dict[str, object], tuple[dict[str, object], ...]]],
        ) -> tuple[
            dict[str, dict[str, set[str]]],
            dict[str, dict[str, tuple[str, ...]]],
            tuple[dict[str, object], ...],
        ]:
    """从冻结 source packs 重建含 transient raw neutral surface 的 inventory。"""
    rows = {
        GODOT_SOURCE_FAMILY: _godot_rows(
            *sources[GODOT_SOURCE_FAMILY]),
        LIBREOFFICE_SOURCE_FAMILY: _libreoffice_rows(
            *sources[LIBREOFFICE_SOURCE_FAMILY]),
        VSCODE_SOURCE_FAMILY: _vscode_rows(
            *sources[VSCODE_SOURCE_FAMILY]),
    }
    return derive_neutral_phrase_inventory(rows)


def _is_en_zh_synonym(columns: list[str]) -> bool:
    """在 rich parser 前只筛选明确的英中 Synonym endpoint 方向。"""
    if len(columns) != 5 or columns[1] != "/r/Synonym":
        return False
    left, right = columns[2], columns[3]
    return ((left.startswith("/c/en/") and right.startswith("/c/zh/"))
            or (left.startswith("/c/zh/") and right.startswith("/c/en/")))


def _scan_conceptnet_alias_evidence(
        raw_path: Path,
        *,
        phrase_support: dict[str, dict[str, set[str]]],
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """单遍扫描 full raw，并严格解析全部英中 Synonym 后选择 neutral 命中。"""
    if (not raw_path.is_file()
            or sha256_path(raw_path) != CONCEPTNET_RAW_SHA256):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias raw SHA-256 漂移")
    evidence = []
    synonym_count = 0
    en_zh_count = 0
    anomalies = Counter()
    line_count = 0
    try:
        with gzip.open(
                raw_path, "rt", encoding="utf-8",
                errors="strict", newline="") as handle:
            for line_number, raw in enumerate(handle, start=1):
                line_count = line_number
                line = raw.rstrip("\r\n")
                columns = line.split("\t", 4)
                if len(columns) == 5 and columns[1] == "/r/Synonym":
                    synonym_count += 1
                if not _is_en_zh_synonym(columns):
                    continue
                en_zh_count += 1
                try:
                    assertion = parse_conceptnet_assertion(
                        line, line_number=line_number)
                except ConceptNetAdapterError as error:
                    anomalies[error.code] += 1
                    continue
                record = conceptnet_alias_evidence_record(
                    assertion, phrase_support=phrase_support)
                if record is not None:
                    evidence.append(record)
    except (OSError, EOFError, UnicodeError) as error:
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias raw gzip/UTF-8 读取失败") from error
    if sha256_path(raw_path) != CONCEPTNET_RAW_SHA256:
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias raw mid-read 漂移")
    evidence.sort(key=lambda item: str(item["alias_evidence_id"]))
    summary = {
        "conceptnet_en_zh_synonym_count": en_zh_count,
        "conceptnet_line_count": line_count,
        "conceptnet_raw_bytes": raw_path.stat().st_size,
        "conceptnet_raw_sha256": CONCEPTNET_RAW_SHA256,
        "conceptnet_synonym_count": synonym_count,
        "matching_parse_anomaly_codes": dict(sorted(anomalies.items())),
        "matching_parse_anomaly_count": sum(anomalies.values()),
    }
    if not strict_json_equal(
            {key: summary[key] for key in _EXPECTED_SCAN},
            _EXPECTED_SCAN):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias full scan frozen census 漂移")
    return tuple(evidence), summary


def _family_facts(
        coverage: tuple[dict[str, object], ...],
        ) -> dict[str, dict[str, int]]:
    """提取必须冻结的 source family alias coverage。"""
    keys = (
        "matched_neutral_phrase_count",
        "pair_any_alias_count",
        "projected_pair_count",
    )
    return {
        str(item["source_family"]): {
            key: int(item[key]) for key in keys}
        for item in coverage
    }


def _validate_alias_summary(
        alias_summary: dict[str, object],
        coverage: tuple[dict[str, object], ...],
        ) -> None:
    """冻结 feasibility 命中、许可分账与 family coverage。"""
    if (any(alias_summary.get(key) != value
            for key, value in _EXPECTED_ALIAS.items())
            or not strict_json_equal(
                alias_summary.get("license_evidence_counts"),
                _EXPECTED_LICENSE_COUNTS)
            or not strict_json_equal(
                _family_facts(coverage), _EXPECTED_FAMILY_COVERAGE)):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias frozen feasibility 漂移")


def _derive(
        *,
        sources: dict[str, tuple[
            dict[str, object], tuple[dict[str, object], ...]]],
        raw_path: Path,
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
        ]:
    """重建 inventory、单遍选择 evidence 并聚合 routes/coverage。"""
    phrase_support, pair_phrases, family_records = _neutral_material(sources)
    projected = {
        str(item["source_family"]): int(item["projected_pair_count"])
        for item in family_records
    }
    if (len(phrase_support)
            != _EXPECTED_ALIAS["neutral_phrase_inventory_count"]
            or projected != {
                family: values["projected_pair_count"]
                for family, values in _EXPECTED_FAMILY_COVERAGE.items()
            }):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias neutral inventory 漂移")
    evidence, scan_summary = _scan_conceptnet_alias_evidence(
        raw_path, phrase_support=phrase_support)
    routes, coverage, alias_summary = derive_conceptnet_alias_routes(
        evidence,
        phrase_support=phrase_support,
        pair_phrases=pair_phrases,
        family_records=family_records,
    )
    _validate_alias_summary(alias_summary, coverage)
    return {
        _OUTPUT_FILES[0][0]: evidence,
        _OUTPUT_FILES[1][0]: routes,
        _OUTPUT_FILES[2][0]: coverage,
    }, {
        "alias": alias_summary,
        "capability_claimed": 0,
        "projection_outcome": "PASS",
        "scan": scan_summary,
    }


def _validate_stored_outputs(
        *,
        stored_outputs: dict[str, tuple[dict[str, object], ...]],
        sources: dict[str, tuple[
            dict[str, object], tuple[dict[str, object], ...]]],
        ) -> dict[str, object]:
    """不重扫 raw，从 sealed inventory 重聚合并严格核验 stored outputs。"""
    phrase_support, pair_phrases, family_records = _neutral_material(sources)
    evidence = stored_outputs[_OUTPUT_FILES[0][0]]
    routes, coverage, alias_summary = derive_conceptnet_alias_routes(
        evidence,
        phrase_support=phrase_support,
        pair_phrases=pair_phrases,
        family_records=family_records,
    )
    if (not strict_json_equal(
            routes, stored_outputs[_OUTPUT_FILES[1][0]])
            or not strict_json_equal(
                coverage, stored_outputs[_OUTPUT_FILES[2][0]])):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias stored aggregation 漂移")
    _validate_alias_summary(alias_summary, coverage)
    return alias_summary


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> dict[str, object]:
    """构造不声明 capability/runtime 的 ConceptNet alias manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V7_CONCEPTNET_ALIAS_AUDIT_KIND,
        "candidate_family_formal_run_count": 0,
        "files": files,
        "format_version": 1,
        "held_out_boundary": {
            "consumed_qt_individual_or_derivative_read_count": 0,
            "vlc_commitment_identity_raw_or_translation_read_count": 0,
        },
        "inputs": {
            "conceptnet_raw_sha256": CONCEPTNET_RAW_SHA256,
            "conceptnet_snapshot_manifest_sha256": (
                CONCEPTNET_SNAPSHOT_MANIFEST_SHA256),
            "v7_cross_source_transformation_manifest_sha256": (
                V7_CROSS_SOURCE_TRANSFORMATION_MANIFEST_SHA256),
            "v7_neutral_source_projection_manifest_sha256": (
                V7_NEUTRAL_SOURCE_PROJECTION_MANIFEST_SHA256),
        },
        "learner_or_selection_change_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "public_git_training_surface_published": 0,
        "runtime_program_published": 0,
        "status": NORMALIZATION_RECOVERY_V7_CONCEPTNET_ALIAS_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "training_surface_materialized_on_k": 1,
    }


def publish_normalization_recovery_v7_conceptnet_alias_audit(
        *,
        run_root: str | Path,
        training_protocol_dir: str | Path,
        source_replay_audit_dir: str | Path,
        neutral_source_projection_dir: str | Path,
        cross_source_transformation_dir: str | Path,
        godot_source_pack_dir: str | Path,
        libreoffice_source_pack_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        thunderbird_source_pack_dir: str | Path,
        conceptnet_snapshot_manifest_path: str | Path,
        conceptnet_raw_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 neutral-source-targeted ConceptNet alias pack。"""
    root = _require_k_root(run_root)
    paths = [
        _within(root, value, label=label)
        for value, label in (
            (training_protocol_dir, "training protocol"),
            (source_replay_audit_dir, "source replay"),
            (neutral_source_projection_dir, "neutral projection"),
            (cross_source_transformation_dir, "transformation"),
            (godot_source_pack_dir, "Godot source pack"),
            (libreoffice_source_pack_dir, "LibreOffice source pack"),
            (vscode_source_pack_dir, "VS Code source pack"),
            (thunderbird_source_pack_dir, "Thunderbird source pack"),
            (target_dir, "target"),
        )
    ]
    (protocol_dir, replay_dir, projection_dir, transformation_dir,
     godot_dir, libreoffice_dir, vscode_dir, thunderbird_dir, target) = paths
    inputs = paths[:-1]
    snapshot_path = Path(conceptnet_snapshot_manifest_path).resolve()
    raw_path = Path(conceptnet_raw_path).resolve()
    if (any(not path.is_dir() for path in inputs)
            or any(_overlap(target, path) for path in inputs)
            or target.exists()
            or not snapshot_path.is_file()
            or not raw_path.is_file()):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias input/target path 非法")
    _read_snapshot_manifest(snapshot_path)
    sources = _input_state(
        protocol_dir=protocol_dir,
        source_replay_dir=replay_dir,
        neutral_projection_dir=projection_dir,
        transformation_dir=transformation_dir,
        godot_dir=godot_dir,
        libreoffice_dir=libreoffice_dir,
        vscode_dir=vscode_dir,
        thunderbird_dir=thunderbird_dir,
    )
    outputs, summary = _derive(sources=sources, raw_path=raw_path)
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


def read_normalization_recovery_v7_conceptnet_alias_audit(
        audit_dir: str | Path,
        *,
        training_protocol_dir: str | Path,
        source_replay_audit_dir: str | Path,
        neutral_source_projection_dir: str | Path,
        cross_source_transformation_dir: str | Path,
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
    """核验 raw/source identities，并从 stored evidence 重聚合严格回读。"""
    root = Path(audit_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias manifest identity 漂移")
    _read_snapshot_manifest(Path(
        conceptnet_snapshot_manifest_path).resolve())
    raw_path = Path(conceptnet_raw_path).resolve()
    if (not raw_path.is_file()
            or sha256_path(raw_path) != CONCEPTNET_RAW_SHA256):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias reader raw identity 漂移")
    sources = _input_state(
        protocol_dir=Path(training_protocol_dir).resolve(),
        source_replay_dir=Path(source_replay_audit_dir).resolve(),
        neutral_projection_dir=Path(
            neutral_source_projection_dir).resolve(),
        transformation_dir=Path(
            cross_source_transformation_dir).resolve(),
        godot_dir=Path(godot_source_pack_dir).resolve(),
        libreoffice_dir=Path(libreoffice_source_pack_dir).resolve(),
        vscode_dir=Path(vscode_source_pack_dir).resolve(),
        thunderbird_dir=Path(thunderbird_source_pack_dir).resolve(),
    )
    stored_outputs = {
        name: _read_jsonl(root / name, label=name)
        for name, _role in _OUTPUT_FILES
    }
    alias_summary = _validate_stored_outputs(
        stored_outputs=stored_outputs, sources=sources)
    stored_summary = stored.get("summary")
    if (not isinstance(stored_summary, dict)
            or not strict_json_equal(stored_summary.get("alias"), alias_summary)
            or not strict_json_equal(
                {key: stored_summary.get("scan", {}).get(key)
                 for key in _EXPECTED_SCAN},
                _EXPECTED_SCAN)):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias stored summary 漂移")
    files = [
        _artifact(root / name, role=role,
                  count=len(stored_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    if not strict_json_equal(
            stored, _manifest(files=files, summary=stored_summary)):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias manifest 字段漂移")
    return ({**stored, "manifest_sha256": expected_manifest_sha256},
            stored_outputs)


__all__ = [
    "NORMALIZATION_RECOVERY_V7_CONCEPTNET_ALIAS_AUDIT_KIND",
    "NORMALIZATION_RECOVERY_V7_CONCEPTNET_ALIAS_STATUS",
    "publish_normalization_recovery_v7_conceptnet_alias_audit",
    "read_normalization_recovery_v7_conceptnet_alias_audit",
]
