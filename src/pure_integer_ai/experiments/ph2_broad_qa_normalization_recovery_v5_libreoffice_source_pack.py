"""不可覆盖发布 recovery-v5 LibreOffice PO 训练候选来源。

publisher 保存固定 raw archive 与 records adapter 派生的文件/pair 记录；reader
每次从 raw archive 重派生，不信任同步修改后的 JSONL 或 manifest。source pack
阶段不启动训练，也不读取 Qt held-out pair、candidate 或 formal artifact。
"""
from __future__ import annotations

import json
from pathlib import Path

import polib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_libreoffice_source_records import (
    parse_normalization_recovery_v5_libreoffice_archive,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    sha256_hex,
    strict_json_equal,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V5_LIBREOFFICE_SOURCE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_LIBREOFFICE_SOURCE_PACK_V1")
NORMALIZATION_RECOVERY_V5_LIBREOFFICE_SOURCE_STATUS = (
    "INDEPENDENT_SOURCE_FROZEN_NOT_SELECTED_NOT_TRAINED")

LIBREOFFICE_REPOSITORY_URL = (
    "https://github.com/LibreOffice/translations.git")
LIBREOFFICE_COMMIT = "dc098e41c756c1b18ffc398e2ccc0e6966925faf"
LIBREOFFICE_COMMIT_DATE = "2026-08-11T13:35:21+02:00"
LIBREOFFICE_ROOT_TREE = "22b7bd224d031987769a73f9ab4ea999b9f3baf7"
LIBREOFFICE_ARCHIVE_NAME = (
    "libreoffice-translations-dc098e41-cui-zh-raw-v1.zip")
LIBREOFFICE_ARCHIVE_BYTES = 1_474_708
LIBREOFFICE_ARCHIVE_SHA256 = (
    "efef7d8280191230a33a675a07f645bef5f36844857658a55942165bfcef7986")
LIBREOFFICE_LICENSE_ID = "MPL-2.0"
LIBREOFFICE_LICENSE_URL = (
    "https://www.libreoffice.org/about-us/licenses/")
LIBREOFFICE_SOURCE_FAMILY = "LIBREOFFICE_PROJECT"
LIBREOFFICE_SOURCE_POLICY_SCOPE = (
    "LIBREOFFICE_CUI_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1")
POLIB_VERSION = "1.2.0"
POLIB_WHEEL_SHA256 = (
    "1c77ee1b81feb31df9bca258cbc58db1bbb32d10214b173882452c73af06d62d")

LIBREOFFICE_OFFICIAL_SUMMARY = {
    "archive_file_count": 3,
    "equal_length_pair_count": 1_971,
    "identity_pair_count": 228,
    "input_conflict_count": 61,
    "nonidentity_pair_count": 3_659,
    "plain_pair_count": 3_887,
    "single_han_difference_count": 331,
    "structure_equal_count": 3_886,
    "training_eligible_pair_count": 3_886,
    "variable_length_pair_count": 1_916,
}


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
        "sha256": sha256_hex(payload),
    }


def _manifest(
        *,
        files: list[dict[str, object]],
        file_records: tuple[dict[str, object], ...],
        parser_summary: dict[str, object],
        ) -> dict[str, object]:
    """构造零训练、零 held-out 读取的独立来源 manifest。"""
    return {
        "archive_acquisition": {
            "archive_builder": (
                "GIT_CAT_FILE_BLOB_DOTNET_ZIP_STORED_FIXED_1980_UTC_V1"),
            "commit": LIBREOFFICE_COMMIT,
            "commit_date": LIBREOFFICE_COMMIT_DATE,
            "member_git_blob_sha1": {
                str(item["relative_path"]): item["git_blob_sha1"]
                for item in file_records
            },
            "repository_url": LIBREOFFICE_REPOSITORY_URL,
            "root_tree": LIBREOFFICE_ROOT_TREE,
            "selected_paths": [
                str(item["relative_path"]) for item in file_records],
        },
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V5_LIBREOFFICE_SOURCE_PACK_KIND),
        "evaluation_or_reserve_read_count": 0,
        "files": files,
        "format_version": 1,
        "held_out_exclusion": {
            "all_qt_source_and_derivatives_excluded_from_train": 1,
            "qt_commit": "1ca7717d28e37a5e64ff5f55d6febb463e75943c",
            "qt_root_tree": "36fd68df240c952ac21dd9770dc70010991b861c",
        },
        "license": {
            "attribution": (
                "LibreOffice contributors; fixed cui localization files"),
            "license_id": LIBREOFFICE_LICENSE_ID,
            "license_url": LIBREOFFICE_LICENSE_URL,
        },
        "mastery_claimed": 0,
        "parser": {
            "polib_runtime_version": polib.__version__,
            "polib_version": POLIB_VERSION,
            "polib_wheel_sha256": POLIB_WHEEL_SHA256,
            "shared_structure_parser": (
                "RECOVERY_V5_LOCALIZATION_STRUCTURE_V1"),
        },
        "parser_summary": parser_summary,
        "production_enabled": 0,
        "source_family": LIBREOFFICE_SOURCE_FAMILY,
        "source_policy_scope": LIBREOFFICE_SOURCE_POLICY_SCOPE,
        "status": NORMALIZATION_RECOVERY_V5_LIBREOFFICE_SOURCE_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_read_count": 0,
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v5 source run root 必须是 K 盘目录")
    return root


def _validate_official_archive(archive_payload: bytes) -> None:
    """在任何解压前核验固定 archive 的物理 identity。"""
    if (len(archive_payload) != LIBREOFFICE_ARCHIVE_BYTES
            or sha256_hex(archive_payload) != LIBREOFFICE_ARCHIVE_SHA256):
        raise BroadQaExternalDataError(
            "LibreOffice official archive identity 漂移")


def _validate_official_source(
        summary: dict[str, object],
        ) -> None:
    """核验 parser 关键库存与固定版本。"""
    if (polib.__version__ != POLIB_VERSION
            or any(summary.get(key) != expected
                   for key, expected in LIBREOFFICE_OFFICIAL_SUMMARY.items())):
        raise BroadQaExternalDataError(
            "LibreOffice official source census 漂移")


def publish_normalization_recovery_v5_libreoffice_source_pack(
        *,
        run_root: str | Path,
        archive_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 LibreOffice recovery-v5 source pack。"""
    root = _require_k_root(run_root)
    source = Path(archive_path).resolve()
    target = Path(target_dir).resolve()
    if (not source.is_file() or not source.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "normalization recovery v5 LibreOffice source path 越界")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization recovery v5 LibreOffice source target 已存在")
    archive_payload = source.read_bytes()
    _validate_official_archive(archive_payload)
    file_records, pairs, summary = (
        parse_normalization_recovery_v5_libreoffice_archive(archive_payload))
    _validate_official_source(summary)
    target.mkdir(parents=True)
    archive_target = target / LIBREOFFICE_ARCHIVE_NAME
    with archive_target.open("xb") as handle:
        handle.write(archive_payload)
    file_path = target / "source-files.jsonl"
    pair_path = target / "translation-pairs.jsonl"
    _write_jsonl(file_path, file_records)
    _write_jsonl(pair_path, pairs)
    files = [
        _artifact(
            archive_target, role="LIBREOFFICE_CUI_RAW_ARCHIVE", count=0),
        _artifact(
            file_path, role="LIBREOFFICE_CUI_SOURCE_FILES",
            count=len(file_records)),
        _artifact(
            pair_path, role="LIBREOFFICE_CUI_TRANSLATION_PAIRS",
            count=len(pairs)),
    ]
    manifest = _manifest(
        files=files,
        file_records=file_records,
        parser_summary=summary,
    )
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": sha256_hex(
        manifest_path.read_bytes())}


def read_normalization_recovery_v5_libreoffice_source_pack(
        source_pack_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """从 raw archive 重派生并严格回读 LibreOffice source pack。"""
    root = Path(source_pack_dir).resolve()
    try:
        encoded_manifest = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded_manifest)
        archive_payload = (root / LIBREOFFICE_ARCHIVE_NAME).read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization recovery v5 LibreOffice source pack 不可读") \
            from error
    if (not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded_manifest):
        raise BroadQaExternalDataError(
            "normalization recovery v5 LibreOffice manifest 非规范")
    _validate_official_archive(archive_payload)
    derived_files, derived_pairs, summary = (
        parse_normalization_recovery_v5_libreoffice_archive(archive_payload))
    _validate_official_source(summary)
    stored_files = _read_jsonl(
        root / "source-files.jsonl", label="LibreOffice source files")
    stored_pairs = _read_jsonl(
        root / "translation-pairs.jsonl", label="LibreOffice pairs")
    if (not strict_json_equal(stored_files, derived_files)
            or not strict_json_equal(stored_pairs, derived_pairs)):
        raise BroadQaExternalDataError(
            "normalization recovery v5 LibreOffice records/source 漂移")
    files = [
        _artifact(
            root / LIBREOFFICE_ARCHIVE_NAME,
            role="LIBREOFFICE_CUI_RAW_ARCHIVE", count=0),
        _artifact(
            root / "source-files.jsonl",
            role="LIBREOFFICE_CUI_SOURCE_FILES", count=len(derived_files)),
        _artifact(
            root / "translation-pairs.jsonl",
            role="LIBREOFFICE_CUI_TRANSLATION_PAIRS", count=len(derived_pairs)),
    ]
    expected = _manifest(
        files=files,
        file_records=derived_files,
        parser_summary=summary,
    )
    if not strict_json_equal(stored, expected):
        raise BroadQaExternalDataError(
            "normalization recovery v5 LibreOffice manifest 漂移")
    return (
        {**stored, "manifest_sha256": sha256_hex(encoded_manifest)},
        derived_files,
        derived_pairs,
    )


__all__ = [
    "LIBREOFFICE_ARCHIVE_NAME",
    "LIBREOFFICE_ARCHIVE_SHA256",
    "LIBREOFFICE_COMMIT",
    "LIBREOFFICE_LICENSE_ID",
    "LIBREOFFICE_SOURCE_FAMILY",
    "LIBREOFFICE_SOURCE_POLICY_SCOPE",
    "NORMALIZATION_RECOVERY_V5_LIBREOFFICE_SOURCE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V5_LIBREOFFICE_SOURCE_STATUS",
    "publish_normalization_recovery_v5_libreoffice_source_pack",
    "read_normalization_recovery_v5_libreoffice_source_pack",
]
