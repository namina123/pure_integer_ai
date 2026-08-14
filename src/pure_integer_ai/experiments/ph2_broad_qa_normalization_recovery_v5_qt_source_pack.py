"""不可覆盖发布 recovery-v5 Qt TS held-out 来源。

source pack 保存 fixed raw archive、source file records 与无翻译文本的 inventory
identity。parser 可以形成 aggregate census，但本阶段不发布 translation pair JSONL；
个体 label 只能由后续 guard 后的 formal materializer 从 raw archive 派生。
"""
from __future__ import annotations

import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    sha256_hex,
    strict_json_equal,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_qt_source_records import (
    QT_MODULES,
    parse_normalization_recovery_v5_qt_archive,
    validate_qt_license_rule,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V5_QT_SOURCE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_QT_HELD_OUT_SOURCE_PACK_V1")
NORMALIZATION_RECOVERY_V5_QT_SOURCE_STATUS = (
    "HELD_OUT_SOURCE_FROZEN_NOT_SELECTED_NOT_EVALUATED")

QT_REPOSITORY_URL = "https://github.com/qt/qttranslations.git"
QT_COMMIT = "1ca7717d28e37a5e64ff5f55d6febb463e75943c"
QT_COMMIT_DATE = "2026-08-04T17:02:13Z"
QT_ROOT_TREE = "36fd68df240c952ac21dd9770dc70010991b861c"
QT_ARCHIVE_NAME = "qttranslations-1ca7717d-selected-zh-raw-v1.zip"
QT_ARCHIVE_BYTES = 1_257_896
QT_ARCHIVE_SHA256 = (
    "04d6687dde75d17426ed9de2201fcc9c563a772c1bc34d01b0798c0813902235")
QT_LICENSE_RULE_BYTES = 2_401
QT_LICENSE_RULE_GIT_BLOB_SHA1 = "103d429b814908015e7e44f9c9d9636ff0f29eea"
QT_LICENSE_RULE_SHA256 = (
    "0087aedb53430fb236602d1965834ff1c902e905b672b759a22ce9665f8a836c")
QT_LICENSE_EXPRESSION = (
    "LicenseRef-Qt-Commercial OR LGPL-3.0-only OR GPL-2.0-only OR "
    "GPL-3.0-only")
QT_LICENSE_RULE_URL = (
    "https://github.com/qt/qttranslations/blob/"
    f"{QT_COMMIT}/licenseRule.json")
QT_SOURCE_FAMILY = "QT_TRANSLATIONS_PROJECT"
QT_SOURCE_POLICY_SCOPE = "QT_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1"

QT_OFFICIAL_SUMMARY = {
    "archive_file_count": 17,
    "equal_length_pair_count": 1_997,
    "identity_pair_count": 337,
    "input_conflict_count": 53,
    "module_pair_counts": {
        "assistant": 199,
        "designer": 1_239,
        "linguist": 381,
        "qt_help": 47,
        "qtbase": 1_561,
        "qtdeclarative": 5,
        "qtmultimedia": 99,
    },
    "nonidentity_pair_count": 3_194,
    "plain_pair_count": 3_531,
    "single_han_difference_count": 349,
    "structure_equal_count": 3_518,
    "training_eligible_pair_count": 3_517,
    "variable_length_pair_count": 1_534,
}


def _inventory_identity(
        pairs: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """去除翻译文本和特征，只保留冻结分母所需来源 identity。"""
    values = tuple({
        "format_version": 1,
        "pair_id": item["pair_id"],
        "record_kind": "QT_TRANSLATIONS_HELD_OUT_IDENTITY_V1",
        "source_identity": item["source_identity"],
        "source_identity_sha256": item["source_identity_sha256"],
        "zh_hans_source_file_id": item["zh_hans"]["source_file_id"],
        "zh_hant_source_file_id": item["zh_hant"]["source_file_id"],
    } for item in pairs)
    if (not values or len({item["pair_id"] for item in values})
            != len(values)):
        raise BroadQaExternalDataError("Qt held-out inventory identity 非法")
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
    """构造 held-out、整包禁训且未发布个体 label 的 manifest。"""
    return {
        "archive_acquisition": {
            "archive_builder": (
                "GIT_CAT_FILE_BLOB_DOTNET_ZIP_STORED_FIXED_1980_UTC_V1"),
            "commit": QT_COMMIT,
            "commit_date": QT_COMMIT_DATE,
            "member_git_blob_sha1": {
                str(item["relative_path"]): item["git_blob_sha1"]
                for item in file_records
            },
            "repository_url": QT_REPOSITORY_URL,
            "root_tree": QT_ROOT_TREE,
            "selected_modules": list(QT_MODULES),
            "selected_paths": [
                str(item["relative_path"]) for item in file_records],
        },
        "artifact_kind": NORMALIZATION_RECOVERY_V5_QT_SOURCE_PACK_KIND,
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
                "The Qt Company and Qt translation contributors; fixed "
                "qttranslations module files"),
            "expression": QT_LICENSE_EXPRESSION,
            "license_rule_bytes": QT_LICENSE_RULE_BYTES,
            "license_rule_git_blob_sha1": QT_LICENSE_RULE_GIT_BLOB_SHA1,
            "license_rule_sha256": QT_LICENSE_RULE_SHA256,
            "license_rule_url": QT_LICENSE_RULE_URL,
        },
        "mastery_claimed": 0,
        "parser": {
            "identity": "PYTHON_STDLIB_ELEMENTTREE_QT_TS_V1",
            "inactive_types": ["obsolete", "unfinished", "vanished"],
            "numerus_in_v1": 0,
            "shared_structure_parser": (
                "RECOVERY_V5_LOCALIZATION_STRUCTURE_V1"),
        },
        "parser_summary": parser_summary,
        "production_enabled": 0,
        "source_family": QT_SOURCE_FAMILY,
        "source_policy_scope": QT_SOURCE_POLICY_SCOPE,
        "status": NORMALIZATION_RECOVERY_V5_QT_SOURCE_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_exclusion": {
            "derivative_message_or_pair_allowed_in_v5_train": 0,
            "exclusion_granularity": (
                "WHOLE_SOURCE_PACK_AND_ALL_DERIVATIVES"),
            "learner_read_count": 0,
        },
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v5 held-out run root 必须是 K 盘目录")
    return root


def _validate_official_archive(archive_payload: bytes) -> None:
    """在任何解压前核验固定 archive 的物理 identity。"""
    if (len(archive_payload) != QT_ARCHIVE_BYTES
            or sha256_hex(archive_payload) != QT_ARCHIVE_SHA256):
        raise BroadQaExternalDataError("Qt official archive identity 漂移")


def _validate_official_source(
        file_records: tuple[dict[str, object], ...],
        archive_payload: bytes,
        summary: dict[str, object],
        ) -> None:
    """核验许可 blob 与 parser 关键库存。"""
    license_records = tuple(
        item for item in file_records
        if item["relative_path"] == "licenseRule.json")
    from zipfile import ZipFile
    from io import BytesIO
    try:
        with ZipFile(BytesIO(archive_payload), "r") as archive:
            license_payload = archive.read("licenseRule.json")
    except (OSError, RuntimeError, KeyError) as error:
        raise BroadQaExternalDataError("Qt licenseRule 不可读") from error
    validate_qt_license_rule(
        license_payload,
        expected_expression=QT_LICENSE_EXPRESSION,
    )
    if (len(license_records) != 1
            or license_records[0]["bytes"] != QT_LICENSE_RULE_BYTES
            or license_records[0]["git_blob_sha1"]
            != QT_LICENSE_RULE_GIT_BLOB_SHA1
            or license_records[0]["sha256"] != QT_LICENSE_RULE_SHA256
            or any(not strict_json_equal(summary.get(key), expected)
                   for key, expected in QT_OFFICIAL_SUMMARY.items())):
        raise BroadQaExternalDataError("Qt official source census 漂移")


def publish_normalization_recovery_v5_qt_source_pack(
        *,
        run_root: str | Path,
        archive_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 Qt recovery-v5 held-out source pack。"""
    root = _require_k_root(run_root)
    source = Path(archive_path).resolve()
    target = Path(target_dir).resolve()
    if (not source.is_file() or not source.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "normalization recovery v5 Qt source path 越界")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization recovery v5 Qt source target 已存在")
    archive_payload = source.read_bytes()
    _validate_official_archive(archive_payload)
    file_records, pairs, summary = (
        parse_normalization_recovery_v5_qt_archive(archive_payload))
    _validate_official_source(file_records, archive_payload, summary)
    inventory = _inventory_identity(pairs)
    target.mkdir(parents=True)
    archive_target = target / QT_ARCHIVE_NAME
    with archive_target.open("xb") as handle:
        handle.write(archive_payload)
    file_path = target / "source-files.jsonl"
    inventory_path = target / "evaluation-inventory.identity.jsonl"
    _write_jsonl(file_path, file_records)
    _write_jsonl(inventory_path, inventory)
    files = [
        _artifact(
            archive_target, role="QT_TRANSLATIONS_RAW_ARCHIVE", count=0),
        _artifact(
            file_path, role="QT_TRANSLATIONS_SOURCE_FILES",
            count=len(file_records)),
        _artifact(
            inventory_path, role="QT_HELD_OUT_IDENTITY_WITHOUT_LABELS",
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


def read_normalization_recovery_v5_qt_source_pack(
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
        archive_payload = (root / QT_ARCHIVE_NAME).read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization recovery v5 Qt source pack 不可读") from error
    if (not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded_manifest):
        raise BroadQaExternalDataError(
            "normalization recovery v5 Qt manifest 非规范")
    _validate_official_archive(archive_payload)
    derived_files, pairs, summary = (
        parse_normalization_recovery_v5_qt_archive(archive_payload))
    _validate_official_source(derived_files, archive_payload, summary)
    derived_inventory = _inventory_identity(pairs)
    stored_files = _read_jsonl(
        root / "source-files.jsonl", label="Qt source files")
    stored_inventory = _read_jsonl(
        root / "evaluation-inventory.identity.jsonl",
        label="Qt held-out identity",
    )
    if (not strict_json_equal(stored_files, derived_files)
            or not strict_json_equal(stored_inventory, derived_inventory)):
        raise BroadQaExternalDataError(
            "normalization recovery v5 Qt records/source 漂移")
    files = [
        _artifact(
            root / QT_ARCHIVE_NAME,
            role="QT_TRANSLATIONS_RAW_ARCHIVE", count=0),
        _artifact(
            root / "source-files.jsonl",
            role="QT_TRANSLATIONS_SOURCE_FILES", count=len(derived_files)),
        _artifact(
            root / "evaluation-inventory.identity.jsonl",
            role="QT_HELD_OUT_IDENTITY_WITHOUT_LABELS",
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
            "normalization recovery v5 Qt manifest 漂移")
    return (
        {**stored, "manifest_sha256": sha256_hex(encoded_manifest)},
        derived_files,
        derived_inventory,
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V5_QT_SOURCE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V5_QT_SOURCE_STATUS",
    "QT_ARCHIVE_NAME",
    "QT_ARCHIVE_SHA256",
    "QT_COMMIT",
    "QT_LICENSE_EXPRESSION",
    "QT_SOURCE_FAMILY",
    "QT_SOURCE_POLICY_SCOPE",
    "publish_normalization_recovery_v5_qt_source_pack",
    "read_normalization_recovery_v5_qt_source_pack",
]
