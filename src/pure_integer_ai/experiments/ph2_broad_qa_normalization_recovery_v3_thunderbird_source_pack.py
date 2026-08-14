"""冻结 recovery-v3 的独立 Thunderbird Fluent 训练来源。

source pack 固定 Thunderbird l10n 的一个 Git commit，只对 ``zh-CN`` 与
``zh-TW`` 的 Fluent AST 做结构对齐。DTD、properties 等原文件仍保存在 raw
archive 和 source-file inventory 中，但不会被本 parser 伪装成纯文本训练对。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_source_pack import (
    FLUENT_SYNTAX_VERSION,
    FLUENT_SYNTAX_WHEEL_SHA256,
    TYPING_EXTENSIONS_VERSION,
    TYPING_EXTENSIONS_WHEEL_SHA256,
    parse_normalization_recovery_firefox_archive,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_evaluation_commitment import (
    EXCLUDED_FIREFOX_SOURCE_PACK_MANIFEST_SHA256,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V3_THUNDERBIRD_SOURCE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V3_THUNDERBIRD_SOURCE_PACK_V1")
NORMALIZATION_RECOVERY_V3_THUNDERBIRD_SOURCE_STATUS = (
    "INDEPENDENT_SOURCE_FROZEN_NOT_SELECTED_NOT_TRAINED")
THUNDERBIRD_PATTERN_PAIR_RECORD_KIND = "THUNDERBIRD_L10N_PATTERN_PAIR_V1"

THUNDERBIRD_L10N_REPOSITORY_URL = (
    "https://github.com/thunderbird/thunderbird-l10n")
THUNDERBIRD_L10N_COMMIT = "89827d7f0faa7591324ac1330966f9174fb50773"
THUNDERBIRD_L10N_COMMIT_DATE = "2026-08-14T08:00:24Z"
THUNDERBIRD_L10N_ROOT_TREE = "b25b2731fce1aa4750750ab8d7ddef13b4d9928d"
THUNDERBIRD_L10N_LOCALE_TREES = {
    "zh-CN": "be1f22d08754ced5ddbe3ffd6895b659e01a5f5c",
    "zh-TW": "2a29a40aa27c1f205ee8cee88d36f4ad61083386",
}
THUNDERBIRD_L10N_ARCHIVE_NAME = (
    "thunderbird-l10n-89827d7f0faa7591-raw-v1.zip")
THUNDERBIRD_L10N_ARCHIVE_BYTES = 996_602
THUNDERBIRD_L10N_ARCHIVE_SHA256 = (
    "841d761cbef697e1f5d17353a31dee20a71c03f57d5a3bbf2e186c8dc5cdd0fa")
THUNDERBIRD_L10N_ARCHIVE_FILE_COUNT = 743
THUNDERBIRD_L10N_FILE_EXTENSION_COUNTS = {
    ".dtd": 226,
    ".ftl": 324,
    ".ini": 4,
    ".js": 4,
    ".properties": 182,
    ".txt": 2,
    "<none>": 1,
}
THUNDERBIRD_L10N_FTL_FILE_COUNTS = {"zh-CN": 162, "zh-TW": 162}
THUNDERBIRD_L10N_PATTERN_COUNTS = {"zh-CN": 6_637, "zh-TW": 6_730}
THUNDERBIRD_L10N_COMMON_PATTERN_COUNT = 6_637
THUNDERBIRD_L10N_STRUCTURE_EQUAL_COUNT = 6_507
THUNDERBIRD_L10N_PLAIN_PAIR_COUNT = 5_498
THUNDERBIRD_L10N_NONIDENTITY_PLAIN_PAIR_COUNT = 4_220
THUNDERBIRD_L10N_IDENTITY_PLAIN_PAIR_COUNT = 1_278

THUNDERBIRD_L10N_LICENSE_ID = "MPL-2.0"
THUNDERBIRD_L10N_LICENSE_URL = (
    "https://raw.githubusercontent.com/thunderbird/thunderbird-l10n/"
    f"{THUNDERBIRD_L10N_COMMIT}/LICENSE")
THUNDERBIRD_L10N_LICENSE_BLOB_SHA1 = (
    "a612ad9813b006ce81d1ee438dd784da99a54007")
THUNDERBIRD_L10N_LICENSE_BYTES = 16_725
THUNDERBIRD_L10N_LICENSE_SHA256 = (
    "1f256ecad192880510e84ad60474eab7589218784b9a50bc7ceee34c2b91f1d5")


def _sha256(payload: bytes) -> str:
    """返回来源或规范 artifact 的 SHA-256。"""
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


def parse_normalization_recovery_v3_thunderbird_archive(
        archive_payload: bytes,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """解析双 locale archive，并替换为 Thunderbird 来源身份。"""
    file_records, base_pairs, base_summary = (
        parse_normalization_recovery_firefox_archive(archive_payload))
    pair_records = []
    for item in base_pairs:
        identity = "\0".join((
            str(item["relative_path"]),
            str(item["entry_kind"]),
            str(item["message_id"]),
            str(item["attribute_id"]),
        ))
        pair_records.append({
            **item,
            "pair_id": _sha256((
                THUNDERBIRD_PATTERN_PAIR_RECORD_KIND + "\0" + identity
            ).encode("utf-8")),
            "record_kind": THUNDERBIRD_PATTERN_PAIR_RECORD_KIND,
        })
    extension_counts = Counter(
        PurePosixPath(str(item["relative_path"])).suffix or "<none>"
        for item in file_records)
    nonidentity_count = sum(
        item["plain_pair_eligible"] == 1
        and item["zh_cn"]["surface_text"]
        != item["zh_tw"]["surface_text"]
        for item in pair_records)
    identity_count = sum(
        item["plain_pair_eligible"] == 1
        and item["zh_cn"]["surface_text"]
        == item["zh_tw"]["surface_text"]
        for item in pair_records)
    summary = {
        **base_summary,
        "file_extension_counts": dict(sorted(extension_counts.items())),
        "identity_plain_pair_count": identity_count,
        "nonidentity_plain_pair_count": nonidentity_count,
        "source_format_policy": {
            "fluent_ast_aligned": 1,
            "non_fluent_files_preserved_not_parsed": 1,
            "non_fluent_plain_text_training_allowed": 0,
        },
    }
    return file_records, tuple(pair_records), summary


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范 JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """严格回读规范 JSONL。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(f"{label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(f"{label} JSONL 不可读") from error
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """返回 manifest 文件身份。"""
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
            "command_contract": (
                "git -c core.autocrlf=false archive --format=zip "
                "<commit> LICENSE zh-CN zh-TW"),
            "commit": THUNDERBIRD_L10N_COMMIT,
            "commit_date": THUNDERBIRD_L10N_COMMIT_DATE,
            "repository_url": THUNDERBIRD_L10N_REPOSITORY_URL,
            "root_tree": THUNDERBIRD_L10N_ROOT_TREE,
        },
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V3_THUNDERBIRD_SOURCE_PACK_KIND),
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
                "Thunderbird and Mozilla contributors; fixed l10n commit"),
            "blob_sha1": THUNDERBIRD_L10N_LICENSE_BLOB_SHA1,
            "license_id": THUNDERBIRD_L10N_LICENSE_ID,
            "license_url": THUNDERBIRD_L10N_LICENSE_URL,
            "sha256": THUNDERBIRD_L10N_LICENSE_SHA256,
        },
        "mastery_claimed": 0,
        "parser": {
            "fluent_syntax_version": FLUENT_SYNTAX_VERSION,
            "fluent_syntax_wheel_sha256": FLUENT_SYNTAX_WHEEL_SHA256,
            "typing_extensions_version": TYPING_EXTENSIONS_VERSION,
            "typing_extensions_wheel_sha256": (
                TYPING_EXTENSIONS_WHEEL_SHA256),
        },
        "parser_summary": parser_summary,
        "production_enabled": 0,
        "source_family": "THUNDERBIRD_PROJECT",
        "source_policy_scope": (
            "THUNDERBIRD_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1"),
        "status": NORMALIZATION_RECOVERY_V3_THUNDERBIRD_SOURCE_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_read_count": 0,
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v3 source run root 必须是 K 盘目录")
    return root


def _validate_official_source(
        archive_payload: bytes,
        summary: dict[str, object],
        ) -> None:
    """核验固定 archive、Git tree、许可、格式边界和 parser 规模。"""
    if (len(archive_payload) != THUNDERBIRD_L10N_ARCHIVE_BYTES
            or _sha256(archive_payload) != THUNDERBIRD_L10N_ARCHIVE_SHA256
            or summary.get("archive_file_count")
            != THUNDERBIRD_L10N_ARCHIVE_FILE_COUNT
            or summary.get("license_bytes") != THUNDERBIRD_L10N_LICENSE_BYTES
            or summary.get("license_sha256")
            != THUNDERBIRD_L10N_LICENSE_SHA256
            or summary.get("license_git_blob_sha1")
            != THUNDERBIRD_L10N_LICENSE_BLOB_SHA1
            or not _strict_equal(
                summary.get("locale_trees"), THUNDERBIRD_L10N_LOCALE_TREES)
            or not _strict_equal(
                summary.get("file_extension_counts"),
                THUNDERBIRD_L10N_FILE_EXTENSION_COUNTS)
            or summary.get("common_pattern_count")
            != THUNDERBIRD_L10N_COMMON_PATTERN_COUNT
            or summary.get("structure_equal_count")
            != THUNDERBIRD_L10N_STRUCTURE_EQUAL_COUNT
            or summary.get("plain_pair_count")
            != THUNDERBIRD_L10N_PLAIN_PAIR_COUNT
            or summary.get("nonidentity_plain_pair_count")
            != THUNDERBIRD_L10N_NONIDENTITY_PLAIN_PAIR_COUNT
            or summary.get("identity_plain_pair_count")
            != THUNDERBIRD_L10N_IDENTITY_PLAIN_PAIR_COUNT):
        raise BroadQaExternalDataError(
            "Thunderbird official source identity 漂移")
    locale_summaries = summary.get("locale_summaries")
    if not isinstance(locale_summaries, dict):
        raise BroadQaExternalDataError("Thunderbird locale summary 缺失")
    for locale in ("zh-CN", "zh-TW"):
        item = locale_summaries.get(locale)
        if (not isinstance(item, dict)
                or item.get("ftl_file_count")
                != THUNDERBIRD_L10N_FTL_FILE_COUNTS[locale]
                or item.get("pattern_count")
                != THUNDERBIRD_L10N_PATTERN_COUNTS[locale]):
            raise BroadQaExternalDataError(
                "Thunderbird parser inventory 漂移")


def publish_normalization_recovery_v3_thunderbird_source_pack(
        *,
        run_root: str | Path,
        archive_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 Thunderbird recovery-v3 source pack。"""
    root = _require_k_root(run_root)
    source = Path(archive_path).resolve()
    target = Path(target_dir).resolve()
    if (not source.is_file() or not source.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "normalization recovery v3 source path 越界")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization recovery v3 source target 已存在")
    archive_payload = source.read_bytes()
    file_records, pair_records, summary = (
        parse_normalization_recovery_v3_thunderbird_archive(archive_payload))
    _validate_official_source(archive_payload, summary)
    target.mkdir(parents=True)
    archive_target = target / THUNDERBIRD_L10N_ARCHIVE_NAME
    with archive_target.open("xb") as handle:
        handle.write(archive_payload)
    file_path = target / "source-files.jsonl"
    pair_path = target / "pattern-pairs.jsonl"
    _write_jsonl(file_path, file_records)
    _write_jsonl(pair_path, pair_records)
    files = [
        _artifact(archive_target, role="THUNDERBIRD_L10N_RAW_ARCHIVE", count=0),
        _artifact(file_path, role="THUNDERBIRD_L10N_SOURCE_FILES",
                  count=len(file_records)),
        _artifact(pair_path, role="THUNDERBIRD_L10N_PATTERN_PAIRS",
                  count=len(pair_records)),
    ]
    manifest = _manifest(files=files, parser_summary=summary)
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def read_normalization_recovery_v3_thunderbird_source_pack(
        source_pack_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """从固定 Git archive 重派生并严格回读 source pack。"""
    root = Path(source_pack_dir).resolve()
    try:
        encoded_manifest = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded_manifest)
        archive_payload = (root / THUNDERBIRD_L10N_ARCHIVE_NAME).read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization recovery v3 source pack 不可读") from error
    if (not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded_manifest):
        raise BroadQaExternalDataError(
            "normalization recovery v3 source manifest 非规范")
    derived_files, derived_pairs, summary = (
        parse_normalization_recovery_v3_thunderbird_archive(archive_payload))
    _validate_official_source(archive_payload, summary)
    stored_files = _read_jsonl(
        root / "source-files.jsonl", label="Thunderbird source files")
    stored_pairs = _read_jsonl(
        root / "pattern-pairs.jsonl", label="Thunderbird pattern pairs")
    if (not _strict_equal(stored_files, derived_files)
            or not _strict_equal(stored_pairs, derived_pairs)):
        raise BroadQaExternalDataError(
            "normalization recovery v3 records/source 漂移")
    files = [
        _artifact(root / THUNDERBIRD_L10N_ARCHIVE_NAME,
                  role="THUNDERBIRD_L10N_RAW_ARCHIVE", count=0),
        _artifact(root / "source-files.jsonl",
                  role="THUNDERBIRD_L10N_SOURCE_FILES",
                  count=len(derived_files)),
        _artifact(root / "pattern-pairs.jsonl",
                  role="THUNDERBIRD_L10N_PATTERN_PAIRS",
                  count=len(derived_pairs)),
    ]
    expected = _manifest(files=files, parser_summary=summary)
    if not _strict_equal(stored, expected):
        raise BroadQaExternalDataError(
            "normalization recovery v3 source manifest 漂移")
    return (
        {**stored, "manifest_sha256": _sha256(encoded_manifest)},
        derived_files,
        derived_pairs,
    )


def thunderbird_source_pack_commitment(value: dict[str, object]) -> str:
    """返回 source pack 的路径无关内容 commitment。"""
    return _sha256(canonical_json_bytes(value))


__all__ = [
    "NORMALIZATION_RECOVERY_V3_THUNDERBIRD_SOURCE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V3_THUNDERBIRD_SOURCE_STATUS",
    "THUNDERBIRD_L10N_ARCHIVE_NAME",
    "THUNDERBIRD_L10N_ARCHIVE_SHA256",
    "THUNDERBIRD_L10N_COMMIT",
    "THUNDERBIRD_L10N_LOCALE_TREES",
    "THUNDERBIRD_PATTERN_PAIR_RECORD_KIND",
    "parse_normalization_recovery_v3_thunderbird_archive",
    "publish_normalization_recovery_v3_thunderbird_source_pack",
    "read_normalization_recovery_v3_thunderbird_source_pack",
]
