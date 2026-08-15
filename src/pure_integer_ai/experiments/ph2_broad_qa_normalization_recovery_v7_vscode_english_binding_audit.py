"""发布 VS Code 官方英文 source binding TRAIN-only feasibility artifact。

publisher/reader 都从固定 VS Code source ZIP、逐字节一致的 TypeScript source tree、
固定 parser install 与冻结本地化 source pack 重派生。输出不含源码、英文消息或
中文本地化表面，也不把 source binding 升级为 transformation authorization。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import BadZipFile, ZipFile

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_vscode_source_pack import (
    read_normalization_recovery_v4_vscode_source_pack,
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_vscode_english_binding_records import (
    BINDING_NOT_VSCODE,
    BINDING_UNIQUE,
    derive_vscode_english_binding_feasibility,
    inspect_vscode_typescript_source,
    run_vscode_typescript_ast_extractor,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V7_VSCODE_ENGLISH_BINDING_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_"
    "VSCODE_OFFICIAL_ENGLISH_SOURCE_BINDING_FEASIBILITY_V1")

V5_TRAINING_PROTOCOL_MANIFEST_SHA256 = (
    "3385e340705af3dd75bd30980f35152574bd967aa257c6d789ee8142d0e87480")
V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256 = (
    "a2e40ec5a4950bd167e66100e2b999122ace83a6348aeeddf862ab0d39f75a3e")
V7_NEUTRAL_SEMANTIC_SOURCE_MANIFEST_SHA256 = (
    "fef7eceed2855837a080f979cafd13173dc66c6ff199f99bbab9d3fdee938b01")
V4_VSCODE_SOURCE_PACK_MANIFEST_SHA256 = (
    "10fcdfd37503e3c2058b28dbd7a2d3cfa13ef4e301e4d1c64b1dbf563995b27c")

VSCODE_SOURCE_TAG = "1.131.0"
VSCODE_SOURCE_COMMIT = "3a03d6f72d628a7741c29f456b4ddbb5ae68502c"
VSCODE_SOURCE_TREE = "64560270f2b89eb3c5d86d517d67309113d96db8"
VSCODE_SOURCE_COMMIT_DATE = "2026-07-28T06:40:51Z"
VSCODE_SOURCE_ARCHIVE_BYTES = 60_285_383
VSCODE_SOURCE_ARCHIVE_SHA256 = (
    "fabd9d711113490565290629d96ec1086f2a82eb603fcfb5fec90bbdaacfb4c5")
VSCODE_SOURCE_ARCHIVE_PREFIX = f"vscode-{VSCODE_SOURCE_COMMIT}/"
VSCODE_SOURCE_LICENSE_BYTES = 1_109
VSCODE_SOURCE_LICENSE_SHA256 = (
    "cce33203a80863c22499035b1cfb6aba5df5f02e4ea2669cf5bc5730c1864236")
VSCODE_SOURCE_LICENSE_GIT_BLOB_SHA1 = (
    "7d58428a01c2086c98b81e50e3ec49d89bee52ef")

TYPESCRIPT_PACKAGE_JSON_BYTES = 67
TYPESCRIPT_PACKAGE_JSON_SHA256 = (
    "bd13f30497487a1d449a33932c124c1a149e3d05c4d41e5495b1c024b875c37a")
TYPESCRIPT_LOCK_BYTES = 1_118
TYPESCRIPT_LOCK_SHA256 = (
    "38a9812455fee75332971a0919a87e49eddf9a665a286f37c07c32caa1d5752c")
TYPESCRIPT_PACKAGE_VERSION = "6.0.2"
TYPESCRIPT_DELEGATED_VERSION = "6.0.3"
TYPESCRIPT_PACKAGE_INTEGRITY = (
    "sha512-mbCddXd+jm7hfx7w2YU64/Av4/NqqeG3GoRZgxPcgoTxYjhrcfJRw9ULch71SS4G+"
    "Q3bOXFhRvPqjguN0Hyp5w==")
TYPESCRIPT_DELEGATED_INTEGRITY = (
    "sha512-y2TvuxSZPDyQakkFRPZHKFm+KKVqIisdg9/CZwm9ftvKXLP8NRWj38/ODjNbr43Sso"
    "XqNuAisEf1GdCxqWcdBw==")
TYPESCRIPT_PACKAGE_INSTALL_IDENTITY = {
    "bytes": 10_548,
    "file_count": 9,
    "inventory_sha256": (
        "f198b6a0b54c56dd3fbcd9854f72fcd65ef8e0a63843b2bad413ec2122d8ef86"),
}
TYPESCRIPT_DELEGATED_INSTALL_IDENTITY = {
    "bytes": 24_346_827,
    "file_count": 140,
    "inventory_sha256": (
        "8eacb81b1eb3507c30c39267b2eff2c44da8b32097119e26fe5d2bfdaa8078bd"),
}

_PROTOCOL_FILES = (
    ("train.pair-observations.jsonl", "TRAIN_PAIR_OBSERVATIONS"),
    ("train.phrase-fragments.jsonl", "TRAIN_PHRASE_FRAGMENTS"),
)
_PLAN_FILE = ("structure-plans.jsonl", "VARIABLE_STRUCTURE_OBLIGATION_PLANS")
_OUTPUT_FILES = (
    ("source-files.jsonl", "VSCODE_ENGLISH_SOURCE_FILE_COMMITMENTS"),
    ("main-bindings.jsonl", "VSCODE_ENGLISH_MAIN_BINDING_COMMITMENTS"),
    ("proposal-bindings.jsonl", "VSCODE_ENGLISH_PROPOSAL_BINDINGS"),
    ("binding-census.jsonl", "VSCODE_ENGLISH_BINDING_CENSUS"),
)

_EXPECTED_SOURCE_SUMMARY = {
    "archive_member_count": 21_130,
    "archive_uncompressed_bytes": 237_767_098,
    "selected_source_bytes": 97_565_113,
    "selected_source_file_count": 7_871,
    "selected_source_inventory_sha256": (
        "23e5952e17952f48d5378c9de86775f8155f1fd7910279a26e37ad9e5f69b439"),
    "unselected_symlink_count": 1,
}
_EXPECTED_BINDING_SUMMARY = {
    "ast_binding_key_count": 22_544,
    "ast_call_count": 23_905,
    "ast_parse_diagnostic_count": 0,
    "ast_unsupported_count": 0,
    "eligible_extension_pair_count": 3_351,
    "eligible_main_pair_count": 20_919,
    "main_binding_conflict_count": 0,
    "main_binding_missing_count": 630,
    "main_binding_unique_count": 20_289,
    "main_binding_record_set_sha256": (
        "9ca85c733cef5d8cbe34856d544a453dd4ab9670fc1b1001e99fe420b92cbdbc"),
    "proposal_binding_outcome_counts": {
        BINDING_NOT_VSCODE: 3,
        BINDING_UNIQUE: 11,
    },
    "proposal_count": 14,
    "proposal_record_set_sha256": (
        "855c8d442963ab6fcac7c95288077520d558922fe2b5cdc1d8825d202ab51e79"),
    "source_binding_authorizes_transformation": 0,
    "source_file_record_count": 7_871,
    "source_file_record_set_sha256": (
        "9eb94421ca8d0fcc34d0394ea8d06c64c71896dc61cf24918e8ae93d7d0aa7c0"),
    "source_or_message_surface_published": 0,
    "typescript_version": TYPESCRIPT_DELEGATED_VERSION,
}


def _sha256(payload: bytes) -> str:
    """返回文件、代码或 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    """返回 license 的 Git blob identity。"""
    prefix = b"blob " + str(len(payload)).encode("ascii") + b"\x00"
    return hashlib.sha1(prefix + payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式 run root 是已存在的 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v7 VS Code English binding root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """限制 publisher 数据路径位于显式 K 盘 root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"v7 VS Code English binding {label} path 越界")
    return path


def _physical_identity(
        path: Path,
        *,
        expected_bytes: int,
        expected_sha256: str,
        label: str,
        ) -> dict[str, object]:
    """流式核验固定文件 identity。"""
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size += len(block)
    except OSError as error:
        raise BroadQaExternalDataError(
            f"v7 VS Code English binding {label} 不可读") from error
    if size != expected_bytes or digest.hexdigest() != expected_sha256:
        raise BroadQaExternalDataError(
            f"v7 VS Code English binding {label} identity 漂移")
    return {"bytes": size, "sha256": expected_sha256}


def _tree_identity(path: Path, *, label: str) -> dict[str, object]:
    """形成 parser install 的路径+bytes+SHA 文件树 identity。"""
    if not path.is_dir():
        raise BroadQaExternalDataError(
            f"v7 VS Code English binding {label} 目录缺失")
    records = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_symlink():
            raise BroadQaExternalDataError(
                f"v7 VS Code English binding {label} symlink 非法")
        if not file_path.is_file():
            continue
        try:
            payload = file_path.read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"v7 VS Code English binding {label} 文件不可读") from error
        records.append({
            "bytes": len(payload),
            "relative_path": file_path.relative_to(path).as_posix(),
            "sha256": _sha256(payload),
        })
    if not records:
        raise BroadQaExternalDataError(
            f"v7 VS Code English binding {label} 为空")
    return {
        "bytes": sum(int(item["bytes"]) for item in records),
        "file_count": len(records),
        "inventory_sha256": _sha256(canonical_json_bytes(records)),
    }


def _validate_parser(parser_root: Path) -> dict[str, object]:
    """核验 npm lock、版本、integrity 与实际安装树 identity。"""
    package_identity = _physical_identity(
        parser_root / "package.json",
        expected_bytes=TYPESCRIPT_PACKAGE_JSON_BYTES,
        expected_sha256=TYPESCRIPT_PACKAGE_JSON_SHA256,
        label="TypeScript package.json")
    lock_identity = _physical_identity(
        parser_root / "package-lock.json",
        expected_bytes=TYPESCRIPT_LOCK_BYTES,
        expected_sha256=TYPESCRIPT_LOCK_SHA256,
        label="TypeScript package-lock.json")
    try:
        package = json.loads((parser_root / "package.json").read_bytes())
        lock = json.loads((parser_root / "package-lock.json").read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v7 VS Code English binding parser metadata 非法") from error
    packages = lock.get("packages") if isinstance(lock, dict) else None
    direct = packages.get("node_modules/@typescript/typescript6") \
        if isinstance(packages, dict) else None
    delegated = packages.get("node_modules/@typescript/old") \
        if isinstance(packages, dict) else None
    if (package != {"dependencies": {
            "@typescript/typescript6": TYPESCRIPT_PACKAGE_VERSION}}
            or not isinstance(direct, dict)
            or direct.get("version") != TYPESCRIPT_PACKAGE_VERSION
            or direct.get("integrity") != TYPESCRIPT_PACKAGE_INTEGRITY
            or not isinstance(delegated, dict)
            or delegated.get("version") != TYPESCRIPT_DELEGATED_VERSION
            or delegated.get("integrity") != TYPESCRIPT_DELEGATED_INTEGRITY):
        raise BroadQaExternalDataError(
            "v7 VS Code English binding parser version/license 漂移")
    module_root = parser_root / "node_modules" / "@typescript"
    delegated_install = _tree_identity(
        module_root / "old", label="TypeScript delegated install")
    package_install = _tree_identity(
        module_root / "typescript6", label="TypeScript package install")
    if (delegated_install != TYPESCRIPT_DELEGATED_INSTALL_IDENTITY
            or package_install != TYPESCRIPT_PACKAGE_INSTALL_IDENTITY):
        raise BroadQaExternalDataError(
            "v7 VS Code English binding parser install 漂移")
    return {
        "delegated_install": delegated_install,
        "delegated_version": TYPESCRIPT_DELEGATED_VERSION,
        "package_install": package_install,
        "package_json": package_identity,
        "package_lock": lock_identity,
        "package_version": TYPESCRIPT_PACKAGE_VERSION,
    }


def _validate_source_archive(archive_path: Path) -> dict[str, object]:
    """在选择源码前核验 archive 与根 license identity。"""
    archive_identity = _physical_identity(
        archive_path,
        expected_bytes=VSCODE_SOURCE_ARCHIVE_BYTES,
        expected_sha256=VSCODE_SOURCE_ARCHIVE_SHA256,
        label="VS Code source archive")
    try:
        with ZipFile(archive_path) as archive:
            license_payload = archive.read(
                VSCODE_SOURCE_ARCHIVE_PREFIX + "LICENSE.txt")
    except (OSError, BadZipFile, KeyError, RuntimeError) as error:
        raise BroadQaExternalDataError(
            "v7 VS Code English binding source license 不可读") from error
    if (len(license_payload) != VSCODE_SOURCE_LICENSE_BYTES
            or _sha256(license_payload) != VSCODE_SOURCE_LICENSE_SHA256
            or _git_blob_sha1(license_payload)
            != VSCODE_SOURCE_LICENSE_GIT_BLOB_SHA1):
        raise BroadQaExternalDataError(
            "v7 VS Code English binding source license 漂移")
    return archive_identity


def _validate_predecessors(
        protocol: dict[str, object],
        variable: dict[str, object],
        semantic: dict[str, object],
        vscode: dict[str, object],
        ) -> None:
    """核验 TRAIN-only 顺序和 candidate/runtime 禁入边界。"""
    semantic_summary = semantic.get("summary")
    semantic_source = semantic_summary.get("semantic_source") \
        if isinstance(semantic_summary, dict) else None
    if (protocol.get("status") != "FROZEN_NOT_READ_NOT_LEARNED"
            or variable.get("status")
            != "TRAIN_ONLY_REPRESENTATION_PASS_CAPABILITY_NE_NOT_RUNTIME"
            or semantic.get("status")
            != "TRAIN_ONLY_NEUTRAL_SEMANTIC_SOURCE_FEASIBILITY_PASS_NOT_RUNTIME"
            or semantic.get("candidate_family_formal_run_count") != 0
            or not isinstance(semantic_source, dict)
            or semantic_source.get("capability_outcome")
            != "NE_SOURCE_FEASIBILITY_NOT_AUTHORIZATION"
            or vscode.get("production_enabled") != 0
            or vscode.get("training_read_count") != 0):
        raise BroadQaExternalDataError(
            "v7 VS Code English binding predecessor contract 漂移")


def _subset(value: dict[str, object], expected: dict[str, object]) -> bool:
    """核验冻结 census 的承重字段。"""
    return all(value.get(key) == expected_value
               for key, expected_value in expected.items())


def _input_state(
        *,
        protocol_dir: Path,
        variable_dir: Path,
        semantic_dir: Path,
        vscode_dir: Path,
        source_archive_path: Path,
        source_root: Path,
        parser_root: Path,
        node_executable: str | Path,
        ) -> tuple[
            dict[str, object], dict[str, object], dict[str, object],
            tuple[dict[str, object], ...], tuple[dict[str, object], ...],
            dict[str, object], dict[str, object], dict[str, object],
        ]:
    """严格回读 predecessors、source pack、源码与 AST parser。"""
    protocol = _read_manifest(
        protocol_dir,
        expected_sha256=V5_TRAINING_PROTOCOL_MANIFEST_SHA256,
        label="v5 training protocol")
    variable = _read_manifest(
        variable_dir,
        expected_sha256=V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256,
        label="v7 variable structure")
    semantic = _read_manifest(
        semantic_dir,
        expected_sha256=V7_NEUTRAL_SEMANTIC_SOURCE_MANIFEST_SHA256,
        label="v7 neutral semantic source")
    vscode, _files, pairs = (
        read_normalization_recovery_v4_vscode_source_pack(vscode_dir))
    if vscode.get("manifest_sha256") \
            != V4_VSCODE_SOURCE_PACK_MANIFEST_SHA256:
        raise BroadQaExternalDataError(
            "v7 VS Code English binding localization source 漂移")
    _validate_predecessors(protocol, variable, semantic, vscode)
    archive_identity = _validate_source_archive(source_archive_path)
    source_files, source_summary = inspect_vscode_typescript_source(
        archive_path=source_archive_path,
        source_root=source_root,
        archive_prefix=VSCODE_SOURCE_ARCHIVE_PREFIX,
    )
    if not _subset(source_summary, _EXPECTED_SOURCE_SUMMARY):
        raise BroadQaExternalDataError(
            "v7 VS Code English binding source census 漂移")
    parser_identity = _validate_parser(parser_root)
    ast_result = run_vscode_typescript_ast_extractor(
        source_root=source_root,
        parser_root=parser_root,
        source_files=source_files,
        node_executable=node_executable,
    )
    return (
        protocol, variable, semantic, pairs, source_files, ast_result,
        {"archive": archive_identity, "selection": source_summary},
        parser_identity,
    )


def _derive(
        *,
        protocol_dir: Path,
        protocol: dict[str, object],
        variable_dir: Path,
        variable: dict[str, object],
        vscode_pairs: tuple[dict[str, object], ...],
        source_files: tuple[dict[str, object], ...],
        ast_result: dict[str, object],
        ) -> tuple[dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """重建 14 proposal，并派生 main binding commitments。"""
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
    proposals = derive_cross_source_transformation_consensus_proposals(
        observations=observations, fragments=fragments, plans=plans)
    source_records, main, proposal, census, summary = (
        derive_vscode_english_binding_feasibility(
            source_files=source_files,
            ast_result=ast_result,
            vscode_pairs=vscode_pairs,
            proposals=proposals,
        ))
    if (len(source_records) != 7_871 or len(main) != 20_919
            or len(proposal) != 14 or len(census) != 2
            or not _subset(summary, _EXPECTED_BINDING_SUMMARY)):
        raise BroadQaExternalDataError(
            "v7 VS Code English binding frozen census 漂移")
    return {
        _OUTPUT_FILES[0][0]: source_records,
        _OUTPUT_FILES[1][0]: main,
        _OUTPUT_FILES[2][0]: proposal,
        _OUTPUT_FILES[3][0]: census,
    }, {
        "audit_outcome": (
            "OFFICIAL_ENGLISH_SOURCE_BINDING_FEASIBILITY_PASS_"
            "AUTHORIZATION_NE"),
        "binding": summary,
        "source_or_message_surface_published": 0,
    }


def _manifest(
        *,
        files: list[dict[str, object]],
        source_identity: dict[str, object],
        parser_identity: dict[str, object],
        extractor_sha256: str,
        summary: dict[str, object],
        ) -> dict[str, object]:
    """构造 official-English binding TRAIN-only manifest。"""
    return {
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V7_VSCODE_ENGLISH_BINDING_AUDIT_KIND),
        "candidate_family_formal_run_count": 0,
        "files": files,
        "format_version": 1,
        "held_out_boundary": {
            "consumed_qt_individual_or_derivative_read_count": 0,
            "vlc_commitment_identity_raw_or_translation_read_count": 0,
        },
        "inputs": {
            "v4_vscode_source_pack_manifest_sha256": (
                V4_VSCODE_SOURCE_PACK_MANIFEST_SHA256),
            "v5_training_protocol_manifest_sha256": (
                V5_TRAINING_PROTOCOL_MANIFEST_SHA256),
            "v7_neutral_semantic_source_manifest_sha256": (
                V7_NEUTRAL_SEMANTIC_SOURCE_MANIFEST_SHA256),
            "v7_variable_structure_audit_manifest_sha256": (
                V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256),
        },
        "learner_or_selection_change_count": 0,
        "mastery_claimed": 0,
        "parser": {
            **parser_identity,
            "extractor_sha256": extractor_sha256,
            "identity": "TYPESCRIPT_COMPILER_API_LOCALIZE_AST_V1",
        },
        "production_enabled": 0,
        "runtime_program_published": 0,
        "source": {
            **source_identity,
            "commit": VSCODE_SOURCE_COMMIT,
            "commit_date": VSCODE_SOURCE_COMMIT_DATE,
            "license": {
                "bytes": VSCODE_SOURCE_LICENSE_BYTES,
                "git_blob_sha1": VSCODE_SOURCE_LICENSE_GIT_BLOB_SHA1,
                "license_id": "MIT",
                "sha256": VSCODE_SOURCE_LICENSE_SHA256,
            },
            "repository_url": "https://github.com/microsoft/vscode",
            "tag": VSCODE_SOURCE_TAG,
            "tree": VSCODE_SOURCE_TREE,
        },
        "status": (
            "TRAIN_ONLY_VSCODE_OFFICIAL_ENGLISH_SOURCE_BINDING_"
            "FEASIBILITY_PASS_NOT_RUNTIME"),
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_source_or_message_surface_published": 0,
    }


def _extractor_sha256() -> str:
    """绑定公开 AST extractor 的逐字节代码 identity。"""
    path = Path(__file__).with_name(
        "ph2_broad_qa_normalization_recovery_v7_"
        "vscode_english_binding_ast.mjs")
    try:
        return _sha256(path.read_bytes())
    except OSError as error:
        raise BroadQaExternalDataError(
            "v7 VS Code English AST extractor 不可读") from error


def publish_normalization_recovery_v7_vscode_english_binding_audit(
        *,
        run_root: str | Path,
        training_protocol_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        neutral_semantic_source_audit_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        vscode_source_archive_path: str | Path,
        vscode_source_root: str | Path,
        typescript_parser_root: str | Path,
        target_dir: str | Path,
        node_executable: str | Path = "node",
        ) -> dict[str, object]:
    """不可覆盖发布 official-English binding feasibility。"""
    root = _require_k_root(run_root)
    values = (
        (training_protocol_dir, "training protocol"),
        (variable_structure_audit_dir, "variable structure"),
        (neutral_semantic_source_audit_dir, "neutral semantic source"),
        (vscode_source_pack_dir, "VS Code localization source"),
        (vscode_source_archive_path, "VS Code source archive"),
        (vscode_source_root, "VS Code source root"),
        (typescript_parser_root, "TypeScript parser"),
        (target_dir, "target"),
    )
    paths = tuple(_within(root, value, label=label)
                  for value, label in values)
    directories = (paths[0], paths[1], paths[2], paths[3], paths[5], paths[6])
    target = paths[7]
    if (any(not path.is_dir() for path in directories)
            or not paths[4].is_file() or target.exists()
            or any(_overlap(target, path) for path in paths[:7])):
        raise BroadQaExternalDataError(
            "v7 VS Code English binding input/target path 非法")
    state = _input_state(
        protocol_dir=paths[0], variable_dir=paths[1], semantic_dir=paths[2],
        vscode_dir=paths[3], source_archive_path=paths[4],
        source_root=paths[5], parser_root=paths[6],
        node_executable=node_executable,
    )
    outputs, summary = _derive(
        protocol_dir=paths[0], protocol=state[0],
        variable_dir=paths[1], variable=state[1], vscode_pairs=state[3],
        source_files=state[4], ast_result=state[5],
    )
    target.mkdir()
    files = []
    for name, role in _OUTPUT_FILES:
        path = target / name
        _write_jsonl(path, outputs[name])
        files.append(_artifact(path, role=role, count=len(outputs[name])))
    manifest = _manifest(
        files=files, source_identity=state[6], parser_identity=state[7],
        extractor_sha256=_extractor_sha256(), summary=summary)
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(
        manifest_path.read_bytes())}


def read_normalization_recovery_v7_vscode_english_binding_audit(
        audit_dir: str | Path,
        *,
        training_protocol_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        neutral_semantic_source_audit_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        vscode_source_archive_path: str | Path,
        vscode_source_root: str | Path,
        typescript_parser_root: str | Path,
        expected_manifest_sha256: str,
        node_executable: str | Path = "node",
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """从 raw source、parser 与 sealed TRAIN inputs 严格重派生。"""
    root = Path(audit_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v7 VS Code English binding manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v7 VS Code English binding manifest identity 漂移")
    paths = tuple(Path(value).resolve() for value in (
        training_protocol_dir, variable_structure_audit_dir,
        neutral_semantic_source_audit_dir, vscode_source_pack_dir,
        vscode_source_archive_path, vscode_source_root,
        typescript_parser_root,
    ))
    state = _input_state(
        protocol_dir=paths[0], variable_dir=paths[1], semantic_dir=paths[2],
        vscode_dir=paths[3], source_archive_path=paths[4],
        source_root=paths[5], parser_root=paths[6],
        node_executable=node_executable,
    )
    expected_outputs, summary = _derive(
        protocol_dir=paths[0], protocol=state[0],
        variable_dir=paths[1], variable=state[1], vscode_pairs=state[3],
        source_files=state[4], ast_result=state[5],
    )
    stored_outputs = {
        name: _stored_jsonl(root / name, label=name)
        for name, _role in _OUTPUT_FILES
    }
    if any(not strict_json_equal(
            stored_outputs[name], expected_outputs[name])
           for name, _role in _OUTPUT_FILES):
        raise BroadQaExternalDataError(
            "v7 VS Code English binding records/inputs 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    expected = _manifest(
        files=files, source_identity=state[6], parser_identity=state[7],
        extractor_sha256=_extractor_sha256(), summary=summary)
    if not strict_json_equal(stored, expected):
        raise BroadQaExternalDataError(
            "v7 VS Code English binding manifest 字段漂移")
    return (
        {**stored, "manifest_sha256": expected_manifest_sha256},
        stored_outputs,
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V7_VSCODE_ENGLISH_BINDING_AUDIT_KIND",
    "VSCODE_SOURCE_ARCHIVE_SHA256",
    "VSCODE_SOURCE_COMMIT",
    "publish_normalization_recovery_v7_vscode_english_binding_audit",
    "read_normalization_recovery_v7_vscode_english_binding_audit",
]
