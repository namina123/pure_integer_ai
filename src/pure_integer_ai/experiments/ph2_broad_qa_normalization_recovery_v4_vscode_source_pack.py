"""不可覆盖发布并严格回读 recovery-v4 VS Code 本地化来源。

publisher 只接受固定提交生成的 Git archive，保存原 archive 与纯 records 模块
派生的文件/pair 记录。reader 每次都从 raw archive 重派生，不信任同步修改后的
JSONL 或 manifest，也不读取 Firefox evaluation/reserve、learner 或 candidate。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_evaluation_commitment import (
    EXCLUDED_FIREFOX_SOURCE_PACK_MANIFEST_SHA256,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_vscode_source_records import (
    parse_normalization_recovery_v4_vscode_archive,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V4_VSCODE_SOURCE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V4_VSCODE_SOURCE_PACK_V1")
NORMALIZATION_RECOVERY_V4_VSCODE_SOURCE_STATUS = (
    "INDEPENDENT_SOURCE_FROZEN_NOT_SELECTED_NOT_TRAINED")

VSCODE_REPOSITORY_URL = "https://github.com/microsoft/vscode-loc"
VSCODE_COMMIT = "fa368b5a5774c74d7ee953934f1ea567c2c43cf6"
VSCODE_COMMIT_DATE = "2026-07-27T10:18:23-07:00"
VSCODE_ROOT_TREE = "be892e5c22ab853c1786d7428150b638014d6ea6"
VSCODE_ZH_HANS_TREE = "e8188cf4042a56e05f9094e78beafe622f1ae95a"
VSCODE_ZH_HANT_TREE = "34769826b1849cf275398b66cd3daad000498314"
VSCODE_ARCHIVE_NAME = "vscode-loc-fa368b5a5774c74-source-raw-v1.zip"
VSCODE_ARCHIVE_BYTES = 1_279_452
VSCODE_ARCHIVE_SHA256 = (
    "2b0ba416270d194426ec3015e12a06b789bf2b4b1849f833f225a710a4e294dc")
VSCODE_LICENSE_ID = "MIT"
VSCODE_LICENSE_URL = (
    "https://raw.githubusercontent.com/microsoft/vscode-loc/"
    f"{VSCODE_COMMIT}/LICENSE.md")
VSCODE_LICENSE_BYTES = 1_133
VSCODE_LICENSE_GIT_BLOB_SHA1 = "1adca5b75d5d39e0bf1e9a7c06c9a43a09b1f27e"
VSCODE_LICENSE_SHA256 = (
    "d8428ce0697ff754457dbebb25ff82da1a7f95b281f4fef4cc0cd2fa4586a144")
VSCODE_SOURCE_FAMILY = "MICROSOFT_VSCODE_PROJECT"
VSCODE_SOURCE_POLICY_SCOPE = "VSCODE_ZH_HANT_TO_ZH_HANS_FIXED_COMMIT_V1"

VSCODE_OFFICIAL_SUMMARY = {
    "all_equal_length_nonidentity_pair_count": 9_301,
    "all_identity_pair_count": 1_767,
    "all_nonidentity_pair_count": 24_084,
    "all_variable_length_nonidentity_pair_count": 14_783,
    "archive_file_count": 202,
    "both_han_equal_length_nonidentity_pair_count": 9_283,
    "both_han_identity_pair_count": 752,
    "both_han_nonidentity_pair_count": 23_974,
    "both_han_pair_count": 24_726,
    "both_han_structure_equal_nonidentity_pair_count": 23_524,
    "both_han_structure_equal_over_limit_count": 6,
    "both_han_structure_mismatch_nonidentity_pair_count": 450,
    "both_han_variable_length_nonidentity_pair_count": 14_691,
    "locale_file_counts": {"zh_Hans": 100, "zh_Hant": 100},
    "pair_count": 25_851,
    "training_eligible_pair_count": 24_270,
    "translation_json_file_count": 94,
}


def _sha256(payload: bytes) -> str:
    """返回 artifact 或来源文件的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, (list, tuple)):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def _write_jsonl(
        path: Path,
        values: tuple[dict[str, object], ...],
        ) -> None:
    """独占写入规范 JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(
        path: Path,
        *,
        label: str,
        ) -> tuple[dict[str, object], ...]:
    """读取规范 JSONL，并拒绝空行与非 object。"""
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


def _artifact(
        path: Path,
        *,
        role: str,
        count: int,
        ) -> dict[str, object]:
    """构造一个物理文件承诺。"""
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
        parser_summary: dict[str, object],
        ) -> dict[str, object]:
    """构造零训练、零评测读取的独立来源 manifest。"""
    return {
        "archive_acquisition": {
            "commit": VSCODE_COMMIT,
            "commit_date": VSCODE_COMMIT_DATE,
            "locale_trees": {
                "zh_Hans": VSCODE_ZH_HANS_TREE,
                "zh_Hant": VSCODE_ZH_HANT_TREE,
            },
            "repository_url": VSCODE_REPOSITORY_URL,
            "root_tree": VSCODE_ROOT_TREE,
        },
        "artifact_kind": NORMALIZATION_RECOVERY_V4_VSCODE_SOURCE_PACK_KIND,
        "evaluation_or_reserve_read_count": 0,
        "excluded_training_source": {
            "all_derivatives_excluded": 1,
            "source_pack_manifest_sha256": (
                EXCLUDED_FIREFOX_SOURCE_PACK_MANIFEST_SHA256),
        },
        "files": files,
        "format_version": 1,
        "license": {
            "attribution": (
                "Visual Studio Code Localization Packs; Microsoft "
                "Corporation and contributors"),
            "bytes": VSCODE_LICENSE_BYTES,
            "git_blob_sha1": VSCODE_LICENSE_GIT_BLOB_SHA1,
            "license_id": VSCODE_LICENSE_ID,
            "license_url": VSCODE_LICENSE_URL,
            "sha256": VSCODE_LICENSE_SHA256,
        },
        "mastery_claimed": 0,
        "parser": {
            "duplicate_key_policy": "REJECT_AT_EVERY_OBJECT_LEVEL",
            "identity": "PYTHON_STDLIB_JSON_FULL_KEY_PATH_V1",
            "locale_alignment": "RELATIVE_FILE_AND_COMPLETE_JSON_KEY_PATH",
            "non_string_leaf_policy": "REJECT",
        },
        "parser_summary": parser_summary,
        "production_enabled": 0,
        "source_family": VSCODE_SOURCE_FAMILY,
        "source_policy_scope": VSCODE_SOURCE_POLICY_SCOPE,
        "status": NORMALIZATION_RECOVERY_V4_VSCODE_SOURCE_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_read_count": 0,
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v4 source run root 必须是 K 盘目录")
    return root


def _validate_official_source(
        archive_payload: bytes,
        file_records: tuple[dict[str, object], ...],
        summary: dict[str, object],
        ) -> None:
    """核验固定 archive、许可 blob 与 parser 全量库存。"""
    license_records = tuple(
        item for item in file_records if item["relative_path"] == "LICENSE.md")
    if (len(archive_payload) != VSCODE_ARCHIVE_BYTES
            or _sha256(archive_payload) != VSCODE_ARCHIVE_SHA256
            or not _strict_equal(summary, VSCODE_OFFICIAL_SUMMARY)
            or len(license_records) != 1
            or license_records[0]["bytes"] != VSCODE_LICENSE_BYTES
            or license_records[0]["git_blob_sha1"]
            != VSCODE_LICENSE_GIT_BLOB_SHA1
            or license_records[0]["sha256"] != VSCODE_LICENSE_SHA256):
        raise BroadQaExternalDataError(
            "VS Code official source identity 漂移")


def _validate_official_archive(archive_payload: bytes) -> None:
    """在任何解压前核验固定 archive 的物理 identity。"""
    if (len(archive_payload) != VSCODE_ARCHIVE_BYTES
            or _sha256(archive_payload) != VSCODE_ARCHIVE_SHA256):
        raise BroadQaExternalDataError(
            "VS Code official archive identity 漂移")


def publish_normalization_recovery_v4_vscode_source_pack(
        *,
        run_root: str | Path,
        archive_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 VS Code recovery-v4 source pack。"""
    root = _require_k_root(run_root)
    source = Path(archive_path).resolve()
    target = Path(target_dir).resolve()
    if (not source.is_file() or not source.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "normalization recovery v4 VS Code source path 越界")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization recovery v4 VS Code source target 已存在")
    archive_payload = source.read_bytes()
    _validate_official_archive(archive_payload)
    file_records, pairs, summary = (
        parse_normalization_recovery_v4_vscode_archive(archive_payload))
    _validate_official_source(archive_payload, file_records, summary)
    target.mkdir(parents=True)
    archive_target = target / VSCODE_ARCHIVE_NAME
    with archive_target.open("xb") as handle:
        handle.write(archive_payload)
    file_path = target / "source-files.jsonl"
    pair_path = target / "translation-pairs.jsonl"
    _write_jsonl(file_path, file_records)
    _write_jsonl(pair_path, pairs)
    files = [
        _artifact(
            archive_target, role="VSCODE_LOCALIZATION_RAW_ARCHIVE", count=0),
        _artifact(
            file_path, role="VSCODE_LOCALIZATION_SOURCE_FILES",
            count=len(file_records)),
        _artifact(
            pair_path, role="VSCODE_LOCALIZATION_TRANSLATION_PAIRS",
            count=len(pairs)),
    ]
    manifest = _manifest(files=files, parser_summary=summary)
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def read_normalization_recovery_v4_vscode_source_pack(
        source_pack_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """从 raw Git archive 重派生并严格回读 source pack。"""
    root = Path(source_pack_dir).resolve()
    try:
        encoded_manifest = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded_manifest)
        archive_payload = (root / VSCODE_ARCHIVE_NAME).read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization recovery v4 VS Code source pack 不可读") from error
    if (not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded_manifest):
        raise BroadQaExternalDataError(
            "normalization recovery v4 VS Code source manifest 非规范")
    _validate_official_archive(archive_payload)
    derived_files, derived_pairs, summary = (
        parse_normalization_recovery_v4_vscode_archive(archive_payload))
    _validate_official_source(archive_payload, derived_files, summary)
    stored_files = _read_jsonl(
        root / "source-files.jsonl", label="VS Code source files")
    stored_pairs = _read_jsonl(
        root / "translation-pairs.jsonl", label="VS Code translation pairs")
    if (not _strict_equal(stored_files, derived_files)
            or not _strict_equal(stored_pairs, derived_pairs)):
        raise BroadQaExternalDataError(
            "normalization recovery v4 VS Code records/source 漂移")
    files = [
        _artifact(
            root / VSCODE_ARCHIVE_NAME,
            role="VSCODE_LOCALIZATION_RAW_ARCHIVE", count=0),
        _artifact(
            root / "source-files.jsonl",
            role="VSCODE_LOCALIZATION_SOURCE_FILES",
            count=len(derived_files)),
        _artifact(
            root / "translation-pairs.jsonl",
            role="VSCODE_LOCALIZATION_TRANSLATION_PAIRS",
            count=len(derived_pairs)),
    ]
    expected = _manifest(files=files, parser_summary=summary)
    if not _strict_equal(stored, expected):
        raise BroadQaExternalDataError(
            "normalization recovery v4 VS Code source manifest 漂移")
    return (
        {**stored, "manifest_sha256": _sha256(encoded_manifest)},
        derived_files,
        derived_pairs,
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V4_VSCODE_SOURCE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V4_VSCODE_SOURCE_STATUS",
    "VSCODE_ARCHIVE_NAME",
    "VSCODE_ARCHIVE_SHA256",
    "VSCODE_COMMIT",
    "VSCODE_LICENSE_ID",
    "VSCODE_SOURCE_FAMILY",
    "VSCODE_SOURCE_POLICY_SCOPE",
    "publish_normalization_recovery_v4_vscode_source_pack",
    "read_normalization_recovery_v4_vscode_source_pack",
]
