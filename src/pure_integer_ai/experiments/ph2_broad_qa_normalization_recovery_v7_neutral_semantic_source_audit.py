"""发布 recovery-v7 neutral semantic source feasibility TRAIN-only artifact。

publisher 固定 OEWN/PropBank 来源与许可身份，重建四 product family 的 neutral
source rows 和 14 个 transformation proposal，只发布 aggregate/commitment 记录。
reader 从同一 raw source 与 sealed TRAIN inputs 独立重派生全部输出。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_audit
    as neutral_projection_audit,
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
    derive_cross_source_transformation_consensus_proposals,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_semantic_source_records import (
    OEWN_SOURCE_ID,
    PROPBANK_SOURCE_ID,
    SUPPORT_OEWN_ANY,
    SUPPORT_OEWN_ACTION_STATE,
    SUPPORT_PROPBANK_MODAL_CUE,
    SUPPORT_PROPBANK_NEGATION_CUE,
    SUPPORT_PROPBANK_PREDICATE,
    SUPPORT_PROPBANK_ROLE_INVENTORY,
    SUPPORT_TWO_SOURCE_ACTION_STATE,
    SUPPORT_TWO_SOURCE_LEXICAL,
    derive_neutral_semantic_source_feasibility,
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


NORMALIZATION_RECOVERY_V7_NEUTRAL_SEMANTIC_SOURCE_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_"
    "NEUTRAL_SEMANTIC_SOURCE_FEASIBILITY_V1")

V5_TRAINING_PROTOCOL_MANIFEST_SHA256 = (
    "3385e340705af3dd75bd30980f35152574bd967aa257c6d789ee8142d0e87480")
V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256 = (
    "a2e40ec5a4950bd167e66100e2b999122ace83a6348aeeddf862ab0d39f75a3e")
V7_INTENT_SEMANTIC_ALIGNMENT_MANIFEST_SHA256 = (
    "f6af1d650f088346589190b7a222439a1c2e8bf8cca2e9d02f0f2e6b76d6a376")

OEWN_RAW_BYTES = 11_363_503
OEWN_RAW_SHA256 = (
    "9ca6d1dcb75f822fdd66617f7d9da48142ace38dd544d6ad5e2feca1674ad3fe")
OEWN_LICENSE_BYTES = 19_863
OEWN_LICENSE_SHA256 = (
    "672cc8b5663e8dc74c4b07a9dcf477193853575b119908fd3dc0aeeb60a9dbbb")
OEWN_LICENSE_GIT_BLOB_SHA1 = "fe4d1dce8109caa7016fca97f48e49a9ced36ad4"

PROPBANK_ARCHIVE_BYTES = 9_170_910
PROPBANK_ARCHIVE_SHA256 = (
    "2252e0de43590ab93c50dbe92cbd2e61234cb477b504cea21454581eb05aab11")
PROPBANK_LICENSE_SHA256 = (
    "1fdd47d0526982b4e71694ca4faf1de06ef8957faec92e3d023a26d187ac9cd3")
PROPBANK_LICENSE_GIT_BLOB_SHA1 = "6f190a25fb43e284e58573d84e3e27c8f686ef6e"

_PROTOCOL_FILES = (
    ("train.pair-observations.jsonl", "TRAIN_PAIR_OBSERVATIONS"),
    ("train.phrase-fragments.jsonl", "TRAIN_PHRASE_FRAGMENTS"),
)
_PLAN_FILE = ("structure-plans.jsonl", "VARIABLE_STRUCTURE_OBLIGATION_PLANS")
_OUTPUT_FILES = (
    ("source-candidates.jsonl", "NEUTRAL_SEMANTIC_SOURCE_CANDIDATES"),
    ("source-census.jsonl", "NEUTRAL_SEMANTIC_SOURCE_CENSUS"),
    ("family-coverage.jsonl", "NEUTRAL_SEMANTIC_FAMILY_COVERAGE"),
    ("proposal-coverage.jsonl", "NEUTRAL_SEMANTIC_PROPOSAL_COVERAGE"),
    ("fact-families.jsonl", "NEUTRAL_SEMANTIC_FACT_FAMILIES"),
)

_EXPECTED_OEWN = {
    "lexical_entry_count": 135_969,
    "normalized_lemma_phrase_count": 126_726,
    "parse_anomaly_count": 0,
    "sense_count": 185_129,
    "synset_count": 107_519,
}
_EXPECTED_PROPBANK = {
    "alias_count": 17_408,
    "archive_member_count": 7_572,
    "archive_uncompressed_bytes": 30_378_324,
    "duplicate_roleset_id_count": 1,
    "empty_alias_count": 1,
    "empty_argument_text_count": 31,
    "example_argument_count": 56_853,
    "malformed_xml_file_count": 2,
    "modal_cue_occurrence_count": 1_325,
    "modal_cue_phrase_count": 23,
    "negation_cue_occurrence_count": 615,
    "negation_cue_phrase_count": 11,
    "parse_anomaly_count": 3,
    "predicate_alias_phrase_count": 13_562,
    "role_count": 28_597,
    "role_inventory_phrase_count": 13_562,
    "roleset_count": 11_198,
    "valid_xml_file_count": 7_564,
    "xml_file_count": 7_566,
}
_EXPECTED_PROPOSAL_COVERAGE = {
    SUPPORT_OEWN_ANY: 10,
    SUPPORT_OEWN_ACTION_STATE: 6,
    SUPPORT_PROPBANK_PREDICATE: 11,
    SUPPORT_PROPBANK_ROLE_INVENTORY: 11,
    SUPPORT_PROPBANK_MODAL_CUE: 0,
    SUPPORT_PROPBANK_NEGATION_CUE: 0,
    SUPPORT_TWO_SOURCE_LEXICAL: 10,
    SUPPORT_TWO_SOURCE_ACTION_STATE: 6,
}


def _sha256(payload: bytes) -> str:
    """返回文件或规范 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    """重算许可文件的 Git blob identity。"""
    prefix = b"blob " + str(len(payload)).encode("ascii") + b"\x00"
    return hashlib.sha1(prefix + payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式 run root 是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v7 neutral semantic source root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """限制 publisher 输入、source 与 target 位于显式 K 盘 root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"v7 neutral semantic source {label} path 越界")
    return path


def _physical_identity(
        path: Path,
        *,
        expected_bytes: int,
        expected_sha256: str,
        label: str,
        ) -> dict[str, object]:
    """流式核验一个固定 raw/license 文件的物理 identity。"""
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                byte_count += len(block)
    except OSError as error:
        raise BroadQaExternalDataError(
            f"v7 neutral semantic source {label} 不可读") from error
    if (byte_count != expected_bytes
            or digest.hexdigest() != expected_sha256):
        raise BroadQaExternalDataError(
            f"v7 neutral semantic source {label} identity 漂移")
    return {"bytes": byte_count, "sha256": expected_sha256}


def _validate_predecessors(
        protocol: dict[str, object],
        variable: dict[str, object],
        alignment: dict[str, object],
        ) -> None:
    """核验 TRAIN-only predecessor 与 candidate/runtime 禁入边界。"""
    summary = alignment.get("summary")
    alignment_summary = summary.get("alignment") \
        if isinstance(summary, dict) else None
    if (protocol.get("status") != "FROZEN_NOT_READ_NOT_LEARNED"
            or protocol.get("evaluation_or_held_out_payload_read_count") != 0
            or variable.get("status")
            != "TRAIN_ONLY_REPRESENTATION_PASS_CAPABILITY_NE_NOT_RUNTIME"
            or variable.get("candidate_family_formal_run_count") != 0
            or alignment.get("status")
            != "TRAIN_ONLY_INTENT_SEMANTIC_ALIGNMENT_NE_NOT_RUNTIME"
            or alignment.get("candidate_family_formal_run_count") != 0
            or alignment.get("runtime_program_published") != 0
            or not isinstance(alignment_summary, dict)
            or alignment_summary.get("capability_outcome")
            != "NE_ZERO_AUTHORIZED_EXACT"):
        raise BroadQaExternalDataError(
            "v7 neutral semantic source predecessor contract 漂移")


def _source_identities(
        *,
        oewn_raw: dict[str, object],
        oewn_license: dict[str, object],
        propbank_archive: dict[str, object],
        ) -> dict[str, dict[str, object]]:
    """构造 selected source 的 URL/revision/license/raw identity。"""
    return {
        OEWN_SOURCE_ID: {
            "download_url": (
                "https://en-word.net/static/english-wordnet-2025.xml.gz"),
            "license_attribution_required": [
                "PRINCETON_WORDNET", "OPEN_ENGLISH_WORDNET_TEAM"],
            "license_bytes": oewn_license["bytes"],
            "license_git_blob_sha1": OEWN_LICENSE_GIT_BLOB_SHA1,
            "license_id": "WORDNET_LICENSE_PLUS_CC-BY-4.0",
            "license_sha256": oewn_license["sha256"],
            "raw_bytes": oewn_raw["bytes"],
            "raw_sha256": oewn_raw["sha256"],
            "repository_commit": (
                "02ff9f3f5bc0a25592e7263ffdbc9bcb6564936b"),
            "repository_commit_date": "2026-07-22T09:53:49Z",
            "repository_tree": (
                "96ecd968b9f047b6e50e5d79b7e004e95864f715"),
            "version": "2025",
        },
        PROPBANK_SOURCE_ID: {
            "archive_bytes": propbank_archive["bytes"],
            "archive_selection": [
                "LICENSE", "README.md", "frames/.gitignore",
                "frames/README.txt", "frames/frameset.dtd",
                "frames/*.xml"],
            "archive_sha256": propbank_archive["sha256"],
            "excluded_repository_paths": [
                "AMR-UMR-91-rolesets.xml", "dtds/**", "scripts/**"],
            "license_git_blob_sha1": PROPBANK_LICENSE_GIT_BLOB_SHA1,
            "license_id": "CC-BY-SA-4.0",
            "license_sha256": PROPBANK_LICENSE_SHA256,
            "repository_commit": (
                "c66e0ccf28b53f00051b187db83e937b5bee2e32"),
            "repository_commit_date": "2025-12-28T19:47:29Z",
            "repository_tree": (
                "d1e1ef0c13c5ec6e06096b1448cb5f65d4e1b8c7"),
            "repository_url": (
                "https://github.com/propbank/propbank-frames"),
            "version": "3.4",
        },
    }


def _input_state(
        *,
        protocol_dir: Path,
        variable_dir: Path,
        alignment_dir: Path,
        godot_dir: Path,
        libreoffice_dir: Path,
        vscode_dir: Path,
        thunderbird_dir: Path,
        oewn_source_path: Path,
        oewn_license_path: Path,
        propbank_source_path: Path,
        ) -> tuple[
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, tuple[
                dict[str, object], tuple[dict[str, object], ...]]],
            dict[str, dict[str, object]],
        ]:
    """严格回读 predecessors、四 source packs 与两套新来源 identity。"""
    protocol = _read_manifest(
        protocol_dir,
        expected_sha256=V5_TRAINING_PROTOCOL_MANIFEST_SHA256,
        label="v5 training protocol")
    variable = _read_manifest(
        variable_dir,
        expected_sha256=V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256,
        label="v7 variable structure")
    alignment = _read_manifest(
        alignment_dir,
        expected_sha256=V7_INTENT_SEMANTIC_ALIGNMENT_MANIFEST_SHA256,
        label="v7 intent semantic alignment")
    _validate_predecessors(protocol, variable, alignment)
    sources = neutral_projection_audit._source_material(
        godot_dir=godot_dir,
        libreoffice_dir=libreoffice_dir,
        vscode_dir=vscode_dir,
        thunderbird_dir=thunderbird_dir,
    )
    oewn_raw = _physical_identity(
        oewn_source_path,
        expected_bytes=OEWN_RAW_BYTES,
        expected_sha256=OEWN_RAW_SHA256,
        label="OEWN raw")
    oewn_license = _physical_identity(
        oewn_license_path,
        expected_bytes=OEWN_LICENSE_BYTES,
        expected_sha256=OEWN_LICENSE_SHA256,
        label="OEWN license")
    try:
        license_payload = oewn_license_path.read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError(
            "v7 neutral semantic source OEWN license 不可读") from error
    if _git_blob_sha1(license_payload) != OEWN_LICENSE_GIT_BLOB_SHA1:
        raise BroadQaExternalDataError(
            "v7 neutral semantic source OEWN license blob 漂移")
    propbank_archive = _physical_identity(
        propbank_source_path,
        expected_bytes=PROPBANK_ARCHIVE_BYTES,
        expected_sha256=PROPBANK_ARCHIVE_SHA256,
        label="PropBank archive")
    return (
        protocol,
        variable,
        alignment,
        sources,
        _source_identities(
            oewn_raw=oewn_raw,
            oewn_license=oewn_license,
            propbank_archive=propbank_archive,
        ),
    )


def _subset_matches(
        value: dict[str, object],
        expected: dict[str, object],
        ) -> bool:
    """检查冻结 source census 的关键字段。"""
    return all(value.get(key) == expected_value
               for key, expected_value in expected.items())


def _derive(
        *,
        protocol_dir: Path,
        protocol: dict[str, object],
        variable_dir: Path,
        variable: dict[str, object],
        sources: dict[str, tuple[
            dict[str, object], tuple[dict[str, object], ...]]],
        source_identities: dict[str, dict[str, object]],
        oewn_source_path: Path,
        propbank_source_path: Path,
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
        ]:
    """重建 neutral rows/proposals，并派生 source feasibility outputs。"""
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
    proposals = derive_cross_source_transformation_consensus_proposals(
        observations=observations,
        fragments=fragments,
        plans=plans,
    )
    candidate, census, family, proposal, fact, summary = (
        derive_neutral_semantic_source_feasibility(
            oewn_source_path=oewn_source_path,
            propbank_source_path=propbank_source_path,
            propbank_license_sha256=PROPBANK_LICENSE_SHA256,
            propbank_license_git_blob_sha1=(
                PROPBANK_LICENSE_GIT_BLOB_SHA1),
            rows_by_family=rows,
            proposals=proposals,
            source_identities=source_identities,
        ))
    family_counts = {
        str(item["source_family"]): int(item["projected_pair_count"])
        for item in family}
    if (len(candidate) != 6 or len(census) != 2
            or len(family) != 4 or len(proposal) != 4 or len(fact) != 7
            or not _subset_matches(summary["oewn"], _EXPECTED_OEWN)
            or not _subset_matches(
                summary["propbank"], _EXPECTED_PROPBANK)
            or summary["propbank"].get("cross_link_counts") != {
                "FRAMENET": 11_314, "VERBNET": 39_788}
            or summary["propbank"].get("cross_link_roleset_counts") != {
                "FRAMENET": 3_889, "VERBNET": 5_135}
            or family_counts != {
                GODOT_SOURCE_FAMILY: 5_590,
                LIBREOFFICE_SOURCE_FAMILY: 3_886,
                VSCODE_SOURCE_FAMILY: 24_270,
                THUNDERBIRD_SOURCE_FAMILY: 0,
            }
            or summary["proposals"].get("proposal_count") != 14
            or summary["proposals"].get(
                "pre_authorization_outcome_counts") != {
                    "EXACT": 2, "UNKNOWN": 0, "WRONG": 12}
            or summary["proposals"].get(
                "coverage_available_counts")
            != _EXPECTED_PROPOSAL_COVERAGE
            or summary.get("selected_source_count") != 2
            or summary.get("feasibility_outcome")
            != "PASS_NONZERO_CROSS_FAMILY_DISCRETE_SUPPORT"
            or summary.get("capability_outcome")
            != "NE_SOURCE_FEASIBILITY_NOT_AUTHORIZATION"
            or summary.get("raw_or_lexical_surface_published") != 0
            or summary.get("placeholder_role_assignment_count") != 0
            or summary.get("lexical_match_assigns_semantic_sense") != 0):
        raise BroadQaExternalDataError(
            "v7 neutral semantic source frozen census 漂移")
    return {
        _OUTPUT_FILES[0][0]: candidate,
        _OUTPUT_FILES[1][0]: census,
        _OUTPUT_FILES[2][0]: family,
        _OUTPUT_FILES[3][0]: proposal,
        _OUTPUT_FILES[4][0]: fact,
    }, {
        "audit_outcome": "SOURCE_FEASIBILITY_PASS_CAPABILITY_NE",
        "semantic_source": summary,
        "source_surface_published": 0,
    }


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> dict[str, object]:
    """构造 neutral semantic source TRAIN-only manifest。"""
    return {
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V7_NEUTRAL_SEMANTIC_SOURCE_AUDIT_KIND),
        "candidate_family_formal_run_count": 0,
        "files": files,
        "format_version": 1,
        "held_out_boundary": {
            "consumed_qt_individual_or_derivative_read_count": 0,
            "vlc_commitment_identity_raw_or_translation_read_count": 0,
        },
        "inputs": {
            "oewn_license_sha256": OEWN_LICENSE_SHA256,
            "oewn_raw_sha256": OEWN_RAW_SHA256,
            "propbank_archive_sha256": PROPBANK_ARCHIVE_SHA256,
            "propbank_license_sha256": PROPBANK_LICENSE_SHA256,
            "v5_training_protocol_manifest_sha256": (
                V5_TRAINING_PROTOCOL_MANIFEST_SHA256),
            "v7_intent_semantic_alignment_manifest_sha256": (
                V7_INTENT_SEMANTIC_ALIGNMENT_MANIFEST_SHA256),
            "v7_variable_structure_audit_manifest_sha256": (
                V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256),
        },
        "learner_or_selection_change_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": (
            "TRAIN_ONLY_NEUTRAL_SEMANTIC_SOURCE_FEASIBILITY_"
            "PASS_NOT_RUNTIME"),
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_or_source_surface_published_in_audit": 0,
    }


def _paths(
        *,
        root: Path | None,
        values: tuple[tuple[str | Path, str], ...],
        ) -> tuple[Path, ...]:
    """统一解析 publisher 或 reader 的全部路径。"""
    return tuple(
        _within(root, value, label=label) if root is not None
        else Path(value).resolve()
        for value, label in values)


def publish_normalization_recovery_v7_neutral_semantic_source_audit(
        *,
        run_root: str | Path,
        training_protocol_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        intent_semantic_alignment_dir: str | Path,
        godot_source_pack_dir: str | Path,
        libreoffice_source_pack_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        thunderbird_source_pack_dir: str | Path,
        oewn_source_path: str | Path,
        oewn_license_path: str | Path,
        propbank_source_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 two-source semantic feasibility aggregate。"""
    root = _require_k_root(run_root)
    values = (
        (training_protocol_dir, "training protocol"),
        (variable_structure_audit_dir, "variable structure"),
        (intent_semantic_alignment_dir, "intent semantic alignment"),
        (godot_source_pack_dir, "Godot source"),
        (libreoffice_source_pack_dir, "LibreOffice source"),
        (vscode_source_pack_dir, "VS Code source"),
        (thunderbird_source_pack_dir, "Thunderbird source"),
        (oewn_source_path, "OEWN source"),
        (oewn_license_path, "OEWN license"),
        (propbank_source_path, "PropBank source"),
        (target_dir, "target"),
    )
    paths = _paths(root=root, values=values)
    directories = paths[:7]
    source_files = paths[7:10]
    target = paths[10]
    if (any(not path.is_dir() for path in directories)
            or any(not path.is_file() for path in source_files)
            or target.exists()
            or any(_overlap(target, path) for path in directories)
            or any(_overlap(target, path) for path in source_files)):
        raise BroadQaExternalDataError(
            "v7 neutral semantic source input/target path 非法")
    state = _input_state(
        protocol_dir=paths[0], variable_dir=paths[1],
        alignment_dir=paths[2], godot_dir=paths[3],
        libreoffice_dir=paths[4], vscode_dir=paths[5],
        thunderbird_dir=paths[6], oewn_source_path=paths[7],
        oewn_license_path=paths[8], propbank_source_path=paths[9],
    )
    outputs, summary = _derive(
        protocol_dir=paths[0], protocol=state[0],
        variable_dir=paths[1], variable=state[1],
        sources=state[3], source_identities=state[4],
        oewn_source_path=paths[7], propbank_source_path=paths[9],
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
    return {
        **manifest,
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
    }


def read_normalization_recovery_v7_neutral_semantic_source_audit(
        audit_dir: str | Path,
        *,
        training_protocol_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        intent_semantic_alignment_dir: str | Path,
        godot_source_pack_dir: str | Path,
        libreoffice_source_pack_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        thunderbird_source_pack_dir: str | Path,
        oewn_source_path: str | Path,
        oewn_license_path: str | Path,
        propbank_source_path: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """从 raw sources 与 sealed TRAIN inputs 独立重派生并严格回读。"""
    root = Path(audit_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v7 neutral semantic source manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v7 neutral semantic source manifest identity 漂移")
    paths = _paths(root=None, values=(
        (training_protocol_dir, "training protocol"),
        (variable_structure_audit_dir, "variable structure"),
        (intent_semantic_alignment_dir, "intent semantic alignment"),
        (godot_source_pack_dir, "Godot source"),
        (libreoffice_source_pack_dir, "LibreOffice source"),
        (vscode_source_pack_dir, "VS Code source"),
        (thunderbird_source_pack_dir, "Thunderbird source"),
        (oewn_source_path, "OEWN source"),
        (oewn_license_path, "OEWN license"),
        (propbank_source_path, "PropBank source"),
    ))
    state = _input_state(
        protocol_dir=paths[0], variable_dir=paths[1],
        alignment_dir=paths[2], godot_dir=paths[3],
        libreoffice_dir=paths[4], vscode_dir=paths[5],
        thunderbird_dir=paths[6], oewn_source_path=paths[7],
        oewn_license_path=paths[8], propbank_source_path=paths[9],
    )
    expected_outputs, summary = _derive(
        protocol_dir=paths[0], protocol=state[0],
        variable_dir=paths[1], variable=state[1],
        sources=state[3], source_identities=state[4],
        oewn_source_path=paths[7], propbank_source_path=paths[9],
    )
    stored_outputs = {
        name: _stored_jsonl(root / name, label=name)
        for name, _role in _OUTPUT_FILES
    }
    if any(not strict_json_equal(
            stored_outputs[name], expected_outputs[name])
           for name, _role in _OUTPUT_FILES):
        raise BroadQaExternalDataError(
            "v7 neutral semantic source records/inputs 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    if not strict_json_equal(
            stored, _manifest(files=files, summary=summary)):
        raise BroadQaExternalDataError(
            "v7 neutral semantic source manifest 字段漂移")
    return (
        {**stored, "manifest_sha256": expected_manifest_sha256},
        stored_outputs,
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V7_NEUTRAL_SEMANTIC_SOURCE_AUDIT_KIND",
    "OEWN_LICENSE_SHA256",
    "OEWN_RAW_SHA256",
    "PROPBANK_ARCHIVE_SHA256",
    "PROPBANK_LICENSE_SHA256",
    "publish_normalization_recovery_v7_neutral_semantic_source_audit",
    "read_normalization_recovery_v7_neutral_semantic_source_audit",
]
