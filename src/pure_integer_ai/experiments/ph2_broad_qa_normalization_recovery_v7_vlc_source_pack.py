"""不可覆盖发布 recovery-v7 VLC PO held-out 来源。

source pack 保存 fixed raw archive、source file records 与无翻译文本的 identity
roster。parser 只发布 aggregate census；个体 label 只能在 future family 冻结后的
不可逆 formal guard 之后物化。
"""
from __future__ import annotations

import json
from pathlib import Path

import polib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    sha256_hex,
    strict_json_equal,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_vlc_source_records import (
    VLC_ARCHIVE_FILES,
    VLC_LICENSE_EXPRESSION,
    parse_normalization_recovery_v7_vlc_archive,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V7_VLC_SOURCE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_VLC_HELD_OUT_SOURCE_PACK_V1")
NORMALIZATION_RECOVERY_V7_VLC_SOURCE_STATUS = (
    "HELD_OUT_SOURCE_FROZEN_LABEL_BLIND_NOT_EVALUATED")

VLC_REPOSITORY_URL = "https://github.com/videolan/vlc.git"
VLC_COMMIT = "11b36ad142cde9354864625c6d5cbde1c21c9538"
VLC_COMMIT_DATE = "2026-08-15T05:17:18Z"
VLC_ROOT_TREE = "a66d6f05456069990fb4279eec8fb9344b5f437f"
VLC_PO_TREE = "b6f23e66b0fe61203c617f41b989400871c28c0e"
VLC_ARCHIVE_NAME = "vlc-11b36ad142cde935-po-zh-raw-v1.zip"
VLC_ARCHIVE_BYTES = 2_020_433
VLC_ARCHIVE_SHA256 = (
    "eb0c2a4318925bef9dafd1afcf3491f41a2ae37edac3f6dac1553d0b9140c6ed")
VLC_SOURCE_FILES = {
    "COPYING": {
        "bytes": 18_092,
        "git_blob_sha1": "d159169d1050894d3ea3b98e1c965c4058208fe1",
        "sha256": (
            "8177f97513213526df2cf6184d8ff986c675afb514d4e68a404010521b880643"),
    },
    "po/zh_CN.po": {
        "bytes": 1_113_149,
        "git_blob_sha1": "e705bfcdd617b33520a55b8e248485aafbd11b7c",
        "sha256": (
            "4b8a6fdc49a37258a2a26961ae33e466047e4091e78fb2a0ec04915736973671"),
    },
    "po/zh_TW.po": {
        "bytes": 888_569,
        "git_blob_sha1": "f07ed86515d172873f0df8d726cf24ff7234357d",
        "sha256": (
            "83fa709fdda705bd59add8c6f3690b70b2e2471c73d59206e1b08fa867da19e5"),
    },
}
VLC_LICENSE_URL = (
    "https://github.com/videolan/vlc/blob/"
    f"{VLC_COMMIT}/COPYING")
VLC_SOURCE_FAMILY = "VLC_MEDIA_PLAYER_PROJECT"
VLC_SOURCE_POLICY_SCOPE = "VLC_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1"
POLIB_VERSION = "1.2.0"
POLIB_WHEEL_SHA256 = (
    "1c77ee1b81feb31df9bca258cbc58db1bbb32d10214b173882452c73af06d62d")

VLC_OFFICIAL_SUMMARY = {
    "archive_file_count": 3,
    "common_identity_count": 7_475,
    "equal_length_pair_count": 1_967,
    "excluded_common_pair_counts": {
        "any": 3_819,
        "empty": 2_124,
        "fuzzy": 1_340,
        "obsolete": 1_255,
        "plural": 2,
    },
    "identity_pair_count": 337,
    "input_conflict_count": 63,
    "locale_summaries": {
        "zh_Hans": {
            "empty_translation_count": 139,
            "entry_count": 8_454,
            "fuzzy_count": 1_562,
            "language": "zh_CN",
            "metadata_sha256": (
                "2ac0a9aa17951fade234066966eb67da906156d308c185858b2a32681bb215d6"),
            "obsolete_count": 2_234,
            "plural_count": 2,
            "same_package_license_notice_count": 1,
            "translated_count": 5_622,
        },
        "zh_Hant": {
            "empty_translation_count": 2_121,
            "entry_count": 8_017,
            "fuzzy_count": 1_645,
            "language": "zh_TW",
            "metadata_sha256": (
                "0359029b0ea96d95f105e2a3d81be43ca0acdc511ae17373eedeb8c194f93f93"),
            "obsolete_count": 1_797,
            "plural_count": 2,
            "same_package_license_notice_count": 1,
            "translated_count": 3_657,
        },
    },
    "nonidentity_pair_count": 3_319,
    "plain_pair_count": 3_656,
    "single_han_difference_count": 350,
    "structure_equal_count": 3_652,
    "training_eligible_pair_count": 3_651,
    "variable_length_pair_count": 1_689,
    "within_scalar_limit_count": 3_653,
}


def _inventory_identity(
        pairs: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """去除简繁文本和 label 特征，只保留冻结分母来源 identity。"""
    values = tuple({
        "format_version": 1,
        "pair_id": item["pair_id"],
        "record_kind": "VLC_HELD_OUT_IDENTITY_WITHOUT_LABEL_V1",
        "source_identity": item["source_identity"],
        "source_identity_sha256": item["source_identity_sha256"],
        "zh_hans_source_file_id": item["zh_hans"]["source_file_id"],
        "zh_hant_source_file_id": item["zh_hant"]["source_file_id"],
    } for item in pairs)
    if (not values or len({item["pair_id"] for item in values})
            != len(values)):
        raise BroadQaExternalDataError("VLC held-out identity roster 非法")
    return values


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
        inventory: tuple[dict[str, object], ...],
        parser_summary: dict[str, object],
        ) -> dict[str, object]:
    """构造先选源、整包禁训且未发布 label 的 held-out manifest。"""
    return {
        "archive_acquisition": {
            "archive_builder": (
                "GIT_BLOB_DOTNET_ZIP_STORED_FIXED_1980_UTC_V1"),
            "commit": VLC_COMMIT,
            "commit_date": VLC_COMMIT_DATE,
            "member_git_blob_sha1": {
                str(item["relative_path"]): item["git_blob_sha1"]
                for item in file_records
            },
            "po_tree": VLC_PO_TREE,
            "repository_url": VLC_REPOSITORY_URL,
            "root_tree": VLC_ROOT_TREE,
            "selected_paths": [
                str(item["relative_path"]) for item in file_records],
        },
        "artifact_kind": NORMALIZATION_RECOVERY_V7_VLC_SOURCE_PACK_KIND,
        "consumed_qt_exclusion": {
            "individual_or_derivative_label_read_count": 0,
            "whole_consumed_source_excluded": 1,
        },
        "evaluation_state": {
            "aggregate_census_pair_observation_count": len(inventory),
            "formal_evaluation_run_count": 0,
            "formal_label_jsonl_materialized": 0,
            "individual_label_print_count": 0,
            "inventory_identity_materialized": 1,
        },
        "files": files,
        "format_version": 1,
        "license": {
            "attribution": (
                "VideoLAN and VLC translation contributors; fixed zh_CN/zh_TW "
                "PO files"),
            "copying_bytes": VLC_SOURCE_FILES["COPYING"]["bytes"],
            "copying_git_blob_sha1": VLC_SOURCE_FILES[
                "COPYING"]["git_blob_sha1"],
            "copying_sha256": VLC_SOURCE_FILES["COPYING"]["sha256"],
            "copying_url": VLC_LICENSE_URL,
            "expression": VLC_LICENSE_EXPRESSION,
            "po_same_package_license_notice_required": 1,
        },
        "mastery_claimed": 0,
        "parser": {
            "polib_runtime_version": polib.__version__,
            "polib_version": POLIB_VERSION,
            "polib_wheel_sha256": POLIB_WHEEL_SHA256,
            "shared_structure_parser": (
                "RECOVERY_V5_LOCALIZATION_STRUCTURE_V1"),
            "source_identity": "MSGCTXT_MSGID_MSGID_PLURAL",
        },
        "parser_summary": parser_summary,
        "production_enabled": 0,
        "selection_boundary": {
            "candidate_product_independence_checked": 1,
            "individual_translation_or_pair_read_before_selection": 0,
            "selection_basis": [
                "OFFICIAL_REPOSITORY_IDENTITY",
                "FIXED_COMMIT_AND_TREE",
                "LOCALE_PATH_EXISTENCE_AND_BLOB_SIZE",
                "ROOT_LICENSE_FILE_METADATA",
                "PRODUCT_INDEPENDENCE",
            ],
            "source_selected_before_translation_label_parse": 1,
        },
        "source_family": VLC_SOURCE_FAMILY,
        "source_policy_scope": VLC_SOURCE_POLICY_SCOPE,
        "status": NORMALIZATION_RECOVERY_V7_VLC_SOURCE_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_exclusion": {
            "derivative_message_or_pair_allowed_in_v7_train": 0,
            "exclusion_granularity": (
                "WHOLE_SOURCE_PACK_AND_ALL_DERIVATIVES"),
            "learner_profiler_selector_case_browser_read_count": 0,
        },
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v7 source run root 必须是 K 盘目录")
    return root


def _validate_official_archive(archive_payload: bytes) -> None:
    """核对固定 VLC raw archive 的 bytes 与 SHA。"""
    if (len(archive_payload) != VLC_ARCHIVE_BYTES
            or sha256_hex(archive_payload) != VLC_ARCHIVE_SHA256):
        raise BroadQaExternalDataError("VLC official source archive 漂移")


def _validate_official_source(
        file_records: tuple[dict[str, object], ...],
        archive_payload: bytes,
        summary: dict[str, object],
        ) -> None:
    """核对 parser、固定 blobs 与 aggregate census。"""
    if polib.__version__ != POLIB_VERSION:
        raise BroadQaExternalDataError("VLC source polib 版本漂移")
    by_path = {
        str(item["relative_path"]): item for item in file_records}
    if (set(by_path) != set(VLC_ARCHIVE_FILES)
            or any(
                by_path[path].get(key) != expected
                for path, values in VLC_SOURCE_FILES.items()
                for key, expected in values.items())
            or any(not strict_json_equal(summary.get(key), expected)
                   for key, expected in VLC_OFFICIAL_SUMMARY.items())
            or len(archive_payload) != VLC_ARCHIVE_BYTES):
        raise BroadQaExternalDataError("VLC official source identity/census 漂移")


def publish_normalization_recovery_v7_vlc_source_pack(
        *,
        run_root: str | Path,
        archive_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 VLC recovery-v7 held-out source pack。"""
    root = _require_k_root(run_root)
    source = Path(archive_path).resolve()
    target = Path(target_dir).resolve()
    if (not source.is_file() or not source.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "normalization recovery v7 VLC source path 越界")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization recovery v7 VLC source target 已存在")
    archive_payload = source.read_bytes()
    _validate_official_archive(archive_payload)
    file_records, pairs, summary = (
        parse_normalization_recovery_v7_vlc_archive(archive_payload))
    _validate_official_source(file_records, archive_payload, summary)
    inventory = _inventory_identity(pairs)
    target.mkdir(parents=True)
    archive_target = target / VLC_ARCHIVE_NAME
    with archive_target.open("xb") as handle:
        handle.write(archive_payload)
    file_path = target / "source-files.jsonl"
    inventory_path = target / "evaluation-inventory.identity.jsonl"
    _write_jsonl(file_path, file_records)
    _write_jsonl(inventory_path, inventory)
    files = [
        _artifact(
            archive_target, role="VLC_TRANSLATIONS_RAW_ARCHIVE", count=0),
        _artifact(
            file_path, role="VLC_TRANSLATIONS_SOURCE_FILES",
            count=len(file_records)),
        _artifact(
            inventory_path, role="VLC_HELD_OUT_IDENTITY_WITHOUT_LABELS",
            count=len(inventory)),
    ]
    manifest = _manifest(
        files=files,
        file_records=file_records,
        inventory=inventory,
        parser_summary=summary,
    )
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": sha256_hex(
        manifest_path.read_bytes())}


def read_normalization_recovery_v7_vlc_source_pack(
        source_pack_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """从 raw archive 重派生并严格回读无 label source pack。"""
    root = Path(source_pack_dir).resolve()
    try:
        encoded_manifest = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded_manifest)
        archive_payload = (root / VLC_ARCHIVE_NAME).read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization recovery v7 VLC source pack 不可读") from error
    if (not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded_manifest):
        raise BroadQaExternalDataError(
            "normalization recovery v7 VLC manifest 非规范")
    _validate_official_archive(archive_payload)
    derived_files, pairs, summary = (
        parse_normalization_recovery_v7_vlc_archive(archive_payload))
    _validate_official_source(derived_files, archive_payload, summary)
    derived_inventory = _inventory_identity(pairs)
    stored_files = _read_jsonl(
        root / "source-files.jsonl", label="VLC source files")
    stored_inventory = _read_jsonl(
        root / "evaluation-inventory.identity.jsonl",
        label="VLC held-out identity",
    )
    if (not strict_json_equal(stored_files, derived_files)
            or not strict_json_equal(stored_inventory, derived_inventory)):
        raise BroadQaExternalDataError(
            "normalization recovery v7 VLC records/source 漂移")
    files = [
        _artifact(
            root / VLC_ARCHIVE_NAME,
            role="VLC_TRANSLATIONS_RAW_ARCHIVE", count=0),
        _artifact(
            root / "source-files.jsonl",
            role="VLC_TRANSLATIONS_SOURCE_FILES", count=len(derived_files)),
        _artifact(
            root / "evaluation-inventory.identity.jsonl",
            role="VLC_HELD_OUT_IDENTITY_WITHOUT_LABELS",
            count=len(derived_inventory)),
    ]
    expected = _manifest(
        files=files,
        file_records=derived_files,
        inventory=derived_inventory,
        parser_summary=summary,
    )
    if not strict_json_equal(stored, expected):
        raise BroadQaExternalDataError(
            "normalization recovery v7 VLC manifest 漂移")
    return (
        {**stored, "manifest_sha256": sha256_hex(encoded_manifest)},
        derived_files,
        derived_inventory,
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V7_VLC_SOURCE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V7_VLC_SOURCE_STATUS",
    "VLC_ARCHIVE_NAME",
    "VLC_ARCHIVE_SHA256",
    "VLC_COMMIT",
    "VLC_LICENSE_EXPRESSION",
    "VLC_OFFICIAL_SUMMARY",
    "VLC_ROOT_TREE",
    "VLC_SOURCE_FAMILY",
    "VLC_SOURCE_POLICY_SCOPE",
    "publish_normalization_recovery_v7_vlc_source_pack",
    "read_normalization_recovery_v7_vlc_source_pack",
]
