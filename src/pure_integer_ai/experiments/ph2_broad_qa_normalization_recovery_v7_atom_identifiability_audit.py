"""发布 atom identifiable lower-bound TRAIN-only feasibility artifact。

publisher/reader 都从 sealed TRAIN、三家官方英文 source、OpenCC 与固定
UniMorph English 重派生。输出不含英文、中文或代码表面。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_identifiability_records import (
    derive_atom_identifiability_feasibility,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_identifiability_sources import (
    UNIMORPH_ENGLISH_COMMIT,
    UNIMORPH_ENGLISH_COMMIT_DATE,
    UNIMORPH_ENGLISH_DATA_BYTES,
    UNIMORPH_ENGLISH_DATA_SHA256,
    UNIMORPH_ENGLISH_LICENSE_ID,
    UNIMORPH_ENGLISH_REPOSITORY,
    UNIMORPH_ENGLISH_TREE,
    parse_unimorph_english,
    read_opencc_unique_t2s_routes,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_context_local_audit import (
    _artifact,
    _file_artifact,
    _overlap,
    _read_jsonl,
    _stored_jsonl,
    _write_jsonl,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_audit
    as projection_audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
    derive_neutral_upstream_source_rows,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_vscode_english_binding_audit
    as english_audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_vscode_english_binding_records import (
    transient_vscode_english_source_by_pair,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    strict_json_equal,
)


NORMALIZATION_RECOVERY_V7_ATOM_IDENTIFIABILITY_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_"
    "ATOM_IDENTIFIABILITY_AND_LAYOUT_FEASIBILITY_V1")

V5_TRAINING_PROTOCOL_MANIFEST_SHA256 = (
    "3385e340705af3dd75bd30980f35152574bd967aa257c6d789ee8142d0e87480")
V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256 = (
    "a2e40ec5a4950bd167e66100e2b999122ace83a6348aeeddf862ab0d39f75a3e")

_PROTOCOL_FILES = (
    ("train.pair-observations.jsonl", "TRAIN_PAIR_OBSERVATIONS"),
    ("train.phrase-fragments.jsonl", "TRAIN_PHRASE_FRAGMENTS"),
)
_PLAN_FILE = ("structure-plans.jsonl", "VARIABLE_STRUCTURE_OBLIGATION_PLANS")
_OUTPUT_FILES = (
    ("proposal-audit.jsonl", "ATOM_IDENTIFIABILITY_PROPOSAL_AUDIT"),
    ("census.jsonl", "ATOM_IDENTIFIABILITY_CENSUS"),
    ("source-census.jsonl", "ATOM_IDENTIFIABILITY_SOURCE_CENSUS"),
)


def _sha256(payload: bytes) -> str:
    """返回文件或 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式 run root 是已存在的 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v7 atom identifiability root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """限制所有大数据输入与输出位于显式 K 盘 root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"v7 atom identifiability {label} path 越界")
    return path


def _git_identity(root: Path) -> dict[str, str]:
    """核对 UniMorph checkout 的固定 commit/tree/remote。"""
    def read(*args: str) -> str:
        try:
            completed = subprocess.run(
                ("git", "-C", str(root), *args), check=True,
                capture_output=True, text=True, encoding="utf-8")
        except (OSError, subprocess.CalledProcessError) as error:
            raise BroadQaExternalDataError(
                "UniMorph English Git identity 不可读") from error
        return completed.stdout.strip()

    value = {
        "commit": read("rev-parse", "HEAD"),
        "remote": read("config", "--get", "remote.origin.url"),
        "tree": read("rev-parse", "HEAD^{tree}"),
    }
    if (value["commit"] != UNIMORPH_ENGLISH_COMMIT
            or value["tree"] != UNIMORPH_ENGLISH_TREE
            or value["remote"] not in {
                UNIMORPH_ENGLISH_REPOSITORY,
                UNIMORPH_ENGLISH_REPOSITORY + ".git"}):
        raise BroadQaExternalDataError("UniMorph English Git identity 漂移")
    return value


def _input_state(
        *,
        protocol_dir: Path,
        variable_dir: Path,
        semantic_dir: Path,
        godot_dir: Path,
        libreoffice_dir: Path,
        vscode_dir: Path,
        thunderbird_dir: Path,
        vscode_source_archive_path: Path,
        vscode_source_root: Path,
        parser_root: Path,
        opencc_dir: Path,
        unimorph_dir: Path,
        node_executable: str | Path,
        ) -> dict[str, object]:
    """严格回读 TRAIN、官方 source、OpenCC 与 UniMorph。"""
    english = english_audit._input_state(
        protocol_dir=protocol_dir, variable_dir=variable_dir,
        semantic_dir=semantic_dir, vscode_dir=vscode_dir,
        source_archive_path=vscode_source_archive_path,
        source_root=vscode_source_root, parser_root=parser_root,
        node_executable=node_executable)
    protocol = english[0]
    variable = english[1]
    observations, fragments = tuple(
        _read_jsonl(
            protocol_dir / name,
            artifact=_file_artifact(
                protocol, relative_path=name, role=role),
            label=role)
        for name, role in _PROTOCOL_FILES)
    plans = _read_jsonl(
        variable_dir / _PLAN_FILE[0],
        artifact=_file_artifact(
            variable, relative_path=_PLAN_FILE[0], role=_PLAN_FILE[1]),
        label=_PLAN_FILE[1])
    source_material = projection_audit._source_material(
        godot_dir=godot_dir, libreoffice_dir=libreoffice_dir,
        vscode_dir=vscode_dir, thunderbird_dir=thunderbird_dir)
    projected = derive_neutral_upstream_source_rows(
        godot_manifest=source_material[GODOT_SOURCE_FAMILY][0],
        godot_pairs=source_material[GODOT_SOURCE_FAMILY][1],
        libreoffice_manifest=source_material[LIBREOFFICE_SOURCE_FAMILY][0],
        libreoffice_pairs=source_material[LIBREOFFICE_SOURCE_FAMILY][1],
        vscode_manifest=source_material[VSCODE_SOURCE_FAMILY][0],
        vscode_pairs=source_material[VSCODE_SOURCE_FAMILY][1],
        thunderbird_manifest=source_material[THUNDERBIRD_SOURCE_FAMILY][0],
        thunderbird_pairs=source_material[THUNDERBIRD_SOURCE_FAMILY][1])
    official_source_by_pair = {
        str(item["pair_id"]): str(item["_neutral_surface"])
        for family in (GODOT_SOURCE_FAMILY, LIBREOFFICE_SOURCE_FAMILY)
        for item in projected[family]}
    official_source_by_pair.update(transient_vscode_english_source_by_pair(
        source_files=english[4], ast_result=english[5],
        vscode_pairs=english[3]))
    opencc_routes, opencc_census = read_opencc_unique_t2s_routes(opencc_dir)
    git_identity = _git_identity(unimorph_dir)
    morphology, morphology_census = parse_unimorph_english(
        data_path=unimorph_dir / "eng",
        readme_path=unimorph_dir / "README.md",
        license_path=unimorph_dir / "CC-BY-SA-3.0-legalcode.txt")
    return {
        "fragments": fragments,
        "morphology": morphology,
        "morphology_census": morphology_census,
        "observations": observations,
        "official_source_by_pair": official_source_by_pair,
        "opencc_census": opencc_census,
        "opencc_routes": opencc_routes,
        "plans": plans,
        "unimorph_git": git_identity,
    }


def _derive(
        state: dict[str, object],
        ) -> tuple[dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """派生 proposal/census，并形成不含 surface 的 source census。"""
    proposal, census, summary = derive_atom_identifiability_feasibility(
        observations=state["observations"], fragments=state["fragments"],
        plans=state["plans"],
        official_source_by_pair=state["official_source_by_pair"],
        opencc_routes=state["opencc_routes"],
        morphology_by_form=state["morphology"])
    scoring = summary["scoring"]
    if (len(proposal) != 14 or len(census) != 1
            or scoring["outcome_counts"]
            != {"EXACT": 2, "UNKNOWN": 12, "WRONG": 0}
            or scoring["feasibility_outcome"]
            != "PASS_NONZERO_EXACT_ZERO_WRONG"):
        raise BroadQaExternalDataError(
            "v7 atom identifiability frozen feasibility 漂移")
    source_census = ({
        "format_version": 1,
        "official_source_pair_count": len(state["official_source_by_pair"]),
        "record_kind": "ATOM_IDENTIFIABILITY_OFFICIAL_SOURCE_CENSUS_V1",
        "source_family_count": 3,
        "surface_published": 0,
    }, {
        "format_version": 1,
        "record_kind": "ATOM_IDENTIFIABILITY_OPENCC_SOURCE_CENSUS_V1",
        **state["opencc_census"],
        "surface_published": 0,
    }, {
        "format_version": 1,
        "record_kind": "ATOM_IDENTIFIABILITY_UNIMORPH_SOURCE_CENSUS_V1",
        **state["morphology_census"],
        "surface_published": 0,
    })
    return {
        _OUTPUT_FILES[0][0]: proposal,
        _OUTPUT_FILES[1][0]: census,
        _OUTPUT_FILES[2][0]: source_census,
    }, {
        "audit_outcome": (
            "ATOM_IDENTIFIABILITY_FACILITY_PASS_"
            "TRAIN_FEASIBILITY_NONZERO_EXACT_ZERO_WRONG"),
        "capability_claimed": 0,
        "identifiability": summary,
        "runtime_claimed": 0,
        "train_source_or_output_surface_published": 0,
    }


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        unimorph_git: dict[str, str],
        ) -> dict[str, object]:
    """构造 atom identifiability TRAIN-only manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V7_ATOM_IDENTIFIABILITY_AUDIT_KIND,
        "candidate_family_formal_run_count": 0,
        "files": files,
        "format_version": 1,
        "inputs": {
            "v5_training_protocol_manifest_sha256": (
                V5_TRAINING_PROTOCOL_MANIFEST_SHA256),
            "v7_variable_structure_manifest_sha256": (
                V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256),
        },
        "mastery_claimed": 0,
        "opencc": {
            "license_id": "Apache-2.0",
            "package": "opencc-python-reimplemented==0.1.7",
        },
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": (
            "TRAIN_ONLY_ATOM_IDENTIFIABILITY_FEASIBILITY_PASS_NOT_RUNTIME"),
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_source_or_output_surface_published": 0,
        "unimorph": {
            **unimorph_git,
            "commit_date": UNIMORPH_ENGLISH_COMMIT_DATE,
            "data_bytes": UNIMORPH_ENGLISH_DATA_BYTES,
            "data_sha256": UNIMORPH_ENGLISH_DATA_SHA256,
            "license_id": UNIMORPH_ENGLISH_LICENSE_ID,
        },
    }


def publish_normalization_recovery_v7_atom_identifiability_audit(
        *,
        run_root: str | Path,
        training_protocol_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        neutral_semantic_source_audit_dir: str | Path,
        godot_source_pack_dir: str | Path,
        libreoffice_source_pack_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        thunderbird_source_pack_dir: str | Path,
        vscode_source_archive_path: str | Path,
        vscode_source_root: str | Path,
        typescript_parser_root: str | Path,
        opencc_source_pack_dir: str | Path,
        unimorph_english_dir: str | Path,
        target_dir: str | Path,
        node_executable: str | Path = "node",
        ) -> dict[str, object]:
    """不可覆盖发布 atom identifiable lower-bound feasibility。"""
    root = _require_k_root(run_root)
    raw = (
        training_protocol_dir, variable_structure_audit_dir,
        neutral_semantic_source_audit_dir, godot_source_pack_dir,
        libreoffice_source_pack_dir, vscode_source_pack_dir,
        thunderbird_source_pack_dir, vscode_source_archive_path,
        vscode_source_root, typescript_parser_root, opencc_source_pack_dir,
        unimorph_english_dir, target_dir)
    paths = tuple(_within(root, value, label=str(index))
                  for index, value in enumerate(raw))
    target = paths[-1]
    if (any(not path.is_dir() for index, path in enumerate(paths[:-1])
            if index != 7)
            or not paths[7].is_file() or target.exists()
            or any(_overlap(target, path) for path in paths[:-1])):
        raise BroadQaExternalDataError(
            "v7 atom identifiability input/target path 非法")
    state = _input_state(
        protocol_dir=paths[0], variable_dir=paths[1], semantic_dir=paths[2],
        godot_dir=paths[3], libreoffice_dir=paths[4], vscode_dir=paths[5],
        thunderbird_dir=paths[6], vscode_source_archive_path=paths[7],
        vscode_source_root=paths[8], parser_root=paths[9],
        opencc_dir=paths[10], unimorph_dir=paths[11],
        node_executable=node_executable)
    outputs, summary = _derive(state)
    target.mkdir()
    files = []
    for name, role in _OUTPUT_FILES:
        path = target / name
        _write_jsonl(path, outputs[name])
        files.append(_artifact(path, role=role, count=len(outputs[name])))
    manifest = _manifest(
        files=files, summary=summary, unimorph_git=state["unimorph_git"])
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(
        manifest_path.read_bytes())}


def read_normalization_recovery_v7_atom_identifiability_audit_state(
        audit_dir: str | Path,
        *,
        training_protocol_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        neutral_semantic_source_audit_dir: str | Path,
        godot_source_pack_dir: str | Path,
        libreoffice_source_pack_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        thunderbird_source_pack_dir: str | Path,
        vscode_source_archive_path: str | Path,
        vscode_source_root: str | Path,
        typescript_parser_root: str | Path,
        opencc_source_pack_dir: str | Path,
        unimorph_english_dir: str | Path,
        expected_manifest_sha256: str,
        node_executable: str | Path = "node",
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
        ]:
    """严格重派生并返回已验证 TRAIN/OpenCC/UniMorph 内存 state。"""
    root = Path(audit_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v7 atom identifiability manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v7 atom identifiability manifest identity 漂移")
    paths = tuple(Path(value).resolve() for value in (
        training_protocol_dir, variable_structure_audit_dir,
        neutral_semantic_source_audit_dir, godot_source_pack_dir,
        libreoffice_source_pack_dir, vscode_source_pack_dir,
        thunderbird_source_pack_dir, vscode_source_archive_path,
        vscode_source_root, typescript_parser_root, opencc_source_pack_dir,
        unimorph_english_dir))
    state = _input_state(
        protocol_dir=paths[0], variable_dir=paths[1], semantic_dir=paths[2],
        godot_dir=paths[3], libreoffice_dir=paths[4], vscode_dir=paths[5],
        thunderbird_dir=paths[6], vscode_source_archive_path=paths[7],
        vscode_source_root=paths[8], parser_root=paths[9],
        opencc_dir=paths[10], unimorph_dir=paths[11],
        node_executable=node_executable)
    expected_outputs, summary = _derive(state)
    stored_outputs = {
        name: _stored_jsonl(root / name, label=name)
        for name, _role in _OUTPUT_FILES}
    if any(not strict_json_equal(stored_outputs[name], expected_outputs[name])
           for name, _role in _OUTPUT_FILES):
        raise BroadQaExternalDataError(
            "v7 atom identifiability records/inputs 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES]
    expected = _manifest(
        files=files, summary=summary, unimorph_git=state["unimorph_git"])
    if not strict_json_equal(stored, expected):
        raise BroadQaExternalDataError(
            "v7 atom identifiability manifest fields 漂移")
    return (
        {**stored, "manifest_sha256": _sha256(encoded)},
        stored_outputs,
        state,
    )


def read_normalization_recovery_v7_atom_identifiability_audit(
        audit_dir: str | Path,
        **arguments: object,
        ) -> tuple[dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """保持既有二元 strict-reader API，不重复重派生。"""
    manifest, outputs, _state = (
        read_normalization_recovery_v7_atom_identifiability_audit_state(
            audit_dir, **arguments))
    return manifest, outputs


__all__ = [
    "NORMALIZATION_RECOVERY_V7_ATOM_IDENTIFIABILITY_AUDIT_KIND",
    "publish_normalization_recovery_v7_atom_identifiability_audit",
    "read_normalization_recovery_v7_atom_identifiability_audit",
    "read_normalization_recovery_v7_atom_identifiability_audit_state",
]
