"""冻结 recovery-v3 的独立 Godot 编辑器 PO 训练来源。

parser 使用固定 ``polib`` 版本读取 ``zh_Hans`` 与 ``zh_Hant``，按英文
msgctxt/msgid identity 对齐，并把占位符、BBCode 与 HTML 结构签名单独保存。
source pack 不读取 Firefox reserve、candidate、learner 或 formal artifact。
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import polib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_evaluation_commitment import (
    EXCLUDED_FIREFOX_SOURCE_PACK_MANIFEST_SHA256,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V3_GODOT_SOURCE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V3_GODOT_SOURCE_PACK_V1")
NORMALIZATION_RECOVERY_V3_GODOT_SOURCE_STATUS = (
    "INDEPENDENT_SOURCE_FROZEN_NOT_SELECTED_NOT_TRAINED")
GODOT_PO_PAIR_RECORD_KIND = "GODOT_EDITOR_PO_PAIR_V1"

GODOT_REPOSITORY_URL = "https://github.com/godotengine/godot"
GODOT_COMMIT = "3defa2466e4f2c767c347f74620ee86b23282902"
GODOT_COMMIT_DATE = "2026-08-13T23:54:55Z"
GODOT_ROOT_TREE = "e64875a0bc46ef99d1ec741478cd5a507f8ed210"
GODOT_ARCHIVE_NAME = (
    "godot-editor-translations-3defa2466e4f2c76-raw-v1.zip")
GODOT_ARCHIVE_BYTES = 317_045
GODOT_ARCHIVE_SHA256 = (
    "99aa8b9d06304b3346f7975f0794baf8ff6112ee2d2204f0e70e07b74f834074")
GODOT_SOURCE_FILES = {
    "LICENSE.txt": {
        "bytes": 1_149,
        "git_blob_sha1": "0e3ba08d6b2e8cf435241829c96f10b74e4356fe",
        "sha256": "b0435e3b3e4e55238f05f4b306f30524a1b2e20147810d436eaa554fa6855c80",
    },
    "editor/translations/editor/zh_Hans.po": {
        "bytes": 593_073,
        "git_blob_sha1": "58ebbbe09e4940195c344a64f60c1de1c5e3345a",
        "sha256": "2db5f576df0a78aee7d5f42c8ccb9cc18a73a5f9f15f67dd2f26c6f27dad8ac0",
    },
    "editor/translations/editor/zh_Hant.po": {
        "bytes": 508_744,
        "git_blob_sha1": "4c05d114730234ac57d3409675f9549bfd63e930",
        "sha256": "bc9dd2f23aad265a8c88bd2be38f8228fcc723d98e51f5442113348e005a351d",
    },
}
GODOT_ENTRY_COUNTS = {"zh_Hans": 6_419, "zh_Hant": 5_664}
GODOT_COMMON_ENTRY_COUNT = 5_664
GODOT_PLURAL_ENTRY_COUNT = 22
GODOT_SIMPLE_PAIR_COUNT = 5_642
GODOT_STRUCTURE_EQUAL_COUNT = 5_642
GODOT_TRAINING_ELIGIBLE_COUNT = 5_590
GODOT_NONIDENTITY_PAIR_COUNT = 5_406
GODOT_IDENTITY_PAIR_COUNT = 236
GODOT_EQUAL_LENGTH_PAIR_COUNT = 2_870
GODOT_VARIABLE_LENGTH_PAIR_COUNT = 2_772

GODOT_LICENSE_ID = "MIT"
GODOT_LICENSE_URL = (
    "https://raw.githubusercontent.com/godotengine/godot/"
    f"{GODOT_COMMIT}/LICENSE.txt")
POLIB_VERSION = "1.2.0"
POLIB_WHEEL_SHA256 = (
    "1c77ee1b81feb31df9bca258cbc58db1bbb32d10214b173882452c73af06d62d")

_PO_PATHS = {
    "zh_Hans": "editor/translations/editor/zh_Hans.po",
    "zh_Hant": "editor/translations/editor/zh_Hant.po",
}
_STRUCTURE_TOKEN = re.compile(
    r"%(?:L?\d+|n|[-+#0 ]*\d*(?:\.\d+)?[diouxXeEfFgGcrs%])"
    r"|\{[^{}\n]+\}"
    r"|\[(?:/?(?:b|i|u|s|code|url|img|color|font|font_size|hint|kbd|"
    r"center|right|fill|indent|ol|ul|table|cell|p|br)(?:=[^\]\n]*)?)\]"
    r"|</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?/?>"
    r"|&[A-Za-z][A-Za-z0-9]+;",
    re.IGNORECASE,
)
_HAN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def _sha256(payload: bytes) -> str:
    """返回来源或规范 artifact 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    """按 Git blob 编码重算 SHA-1。"""
    header = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(header + payload).hexdigest()


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


def _archive_files(archive_payload: bytes) -> dict[str, bytes]:
    """严格读取只含许可与两份 PO Git blob 的 archive。"""
    if not isinstance(archive_payload, bytes):
        raise BroadQaExternalDataError("Godot archive payload 非 bytes")
    files: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload), "r") as archive:
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                name = path.as_posix()
                if (info.is_dir() or path.is_absolute() or "\\" in name
                        or any(part in ("", ".", "..") for part in path.parts)
                        or name not in GODOT_SOURCE_FILES
                        or info.flag_bits & 0x1 or name in files):
                    raise BroadQaExternalDataError(
                        "Godot archive member 非法")
                files[name] = archive.read(info)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise BroadQaExternalDataError("Godot archive 非法") from error
    if set(files) != set(GODOT_SOURCE_FILES):
        raise BroadQaExternalDataError("Godot archive 来源不完整")
    return files


def _structure_tokens(value: str) -> tuple[str, ...]:
    """提取不承载自然语言表面的占位符与标记结构。"""
    tokens = []
    for match in _STRUCTURE_TOKEN.finditer(value):
        token = match.group()
        if token.startswith("["):
            body = token[1:-1]
            closing = body.startswith("/")
            name = body.lstrip("/").split("=", 1)[0].lower()
            tokens.append(("BBCODE_CLOSE:" if closing else "BBCODE_OPEN:")
                          + name)
        elif token.startswith("<"):
            body = token[1:-1].strip()
            closing = body.startswith("/")
            self_closing = body.endswith("/")
            name = body.lstrip("/").rstrip("/").split(None, 1)[0].lower()
            prefix = "HTML_CLOSE:" if closing else (
                "HTML_SELF:" if self_closing else "HTML_OPEN:")
            tokens.append(prefix + name)
        else:
            tokens.append(token)
    return tuple(tokens)


def _entry_key(entry: polib.POEntry) -> tuple[str, str, str]:
    """返回跨 locale 对齐的英文 source identity。"""
    return entry.msgctxt or "", entry.msgid, entry.msgid_plural or ""


def _entry_record(
        entry: polib.POEntry,
        *,
        locale: str,
        ordinal: int,
        source_file_sha256: str,
        ) -> dict[str, object]:
    """把 polib entry 变为带物理来源身份的规范记录。"""
    semantic = {
        "flags": sorted(entry.flags),
        "msgctxt": entry.msgctxt or "",
        "msgid": entry.msgid,
        "msgid_plural": entry.msgid_plural or "",
        "msgstr": entry.msgstr,
        "msgstr_plural": [
            {"plural_index": int(index), "value": value}
            for index, value in sorted(entry.msgstr_plural.items())
        ],
        "obsolete": int(entry.obsolete),
        "occurrences": [list(item) for item in sorted(entry.occurrences)],
    }
    return {
        **semantic,
        "entry_linenum": entry.linenum,
        "entry_ordinal": ordinal,
        "entry_semantic_sha256": _sha256(canonical_json_bytes(semantic)),
        "locale": locale,
        "source_file_sha256": source_file_sha256,
        "structure_tokens": list(_structure_tokens(entry.msgstr)),
    }


def _parse_po(payload: bytes, *, locale: str) -> tuple[
        dict[tuple[str, str, str], dict[str, object]], dict[str, object]]:
    """严格解析一份 UTF-8/LF PO，并拒绝重复 source identity。"""
    if payload.startswith(b"\xef\xbb\xbf") or b"\r" in payload \
            or not payload.endswith(b"\n"):
        raise BroadQaExternalDataError("Godot PO 编码或换行非法")
    try:
        text = payload.decode("utf-8")
        parsed = polib.pofile(text, wrapwidth=0)
    except (UnicodeDecodeError, OSError, ValueError) as error:
        raise BroadQaExternalDataError("Godot PO parser 失败") from error
    if parsed.metadata.get("Language") != locale:
        raise BroadQaExternalDataError("Godot PO locale 漂移")
    records = {}
    for ordinal, entry in enumerate(parsed):
        key = _entry_key(entry)
        if key in records:
            raise BroadQaExternalDataError("Godot PO source identity 重复")
        records[key] = _entry_record(
            entry,
            locale=locale,
            ordinal=ordinal,
            source_file_sha256=_sha256(payload),
        )
    summary = {
        "entry_count": len(parsed),
        "language": parsed.metadata.get("Language"),
        "metadata_sha256": _sha256(canonical_json_bytes(parsed.metadata)),
        "obsolete_count": sum(int(entry.obsolete) for entry in parsed),
        "plural_count": sum(bool(entry.msgid_plural) for entry in parsed),
        "translated_count": len(parsed.translated_entries()),
    }
    return records, summary


def parse_normalization_recovery_v3_godot_archive(
        archive_payload: bytes,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """重派生 source files、跨 locale pair 与完整库存摘要。"""
    files = _archive_files(archive_payload)
    file_records = tuple({
        "bytes": len(payload),
        "git_blob_sha1": _git_blob_sha1(payload),
        "relative_path": path,
        "sha256": _sha256(payload),
    } for path, payload in sorted(files.items()))
    locale_records = {}
    locale_summaries = {}
    for locale, path in _PO_PATHS.items():
        records, summary = _parse_po(files[path], locale=locale)
        locale_records[locale] = records
        locale_summaries[locale] = summary
    common = sorted(set(locale_records["zh_Hans"]).intersection(
        locale_records["zh_Hant"]))
    pairs = []
    counts = {
        "equal_length_pair_count": 0,
        "identity_pair_count": 0,
        "nonidentity_pair_count": 0,
        "plural_pair_count": 0,
        "simple_pair_count": 0,
        "structure_equal_count": 0,
        "training_eligible_count": 0,
        "variable_length_pair_count": 0,
    }
    for key in common:
        hans = locale_records["zh_Hans"][key]
        hant = locale_records["zh_Hant"][key]
        plural = int(bool(hans["msgid_plural"] or hant["msgid_plural"]))
        simple = int(
            plural == 0
            and "fuzzy" not in hans["flags"]
            and "fuzzy" not in hant["flags"]
            and bool(hans["msgstr"])
            and bool(hant["msgstr"])
        )
        structure_equal = int(
            simple == 1
            and hans["structure_tokens"] == hant["structure_tokens"])
        contains_han = int(bool(_HAN.search(
            str(hans["msgstr"]) + str(hant["msgstr"]))))
        eligible = int(structure_equal == 1 and contains_han == 1)
        identity = int(simple == 1 and hans["msgstr"] == hant["msgstr"])
        equal_length = int(
            simple == 1 and len(str(hans["msgstr"]))
            == len(str(hant["msgstr"])))
        counts["plural_pair_count"] += plural
        counts["simple_pair_count"] += simple
        counts["structure_equal_count"] += structure_equal
        counts["training_eligible_count"] += eligible
        counts["identity_pair_count"] += identity
        counts["nonidentity_pair_count"] += int(simple == 1 and identity == 0)
        counts["equal_length_pair_count"] += equal_length
        counts["variable_length_pair_count"] += int(
            simple == 1 and equal_length == 0)
        identity_payload = {
            "msgctxt": key[0],
            "msgid": key[1],
            "msgid_plural": key[2],
        }
        pairs.append({
            "contains_han": contains_han,
            "equal_length": equal_length,
            "format_version": 1,
            "identity_preservation": identity,
            "pair_id": _sha256(canonical_json_bytes({
                **identity_payload,
                "record_kind": GODOT_PO_PAIR_RECORD_KIND,
            })),
            "plural": plural,
            "record_kind": GODOT_PO_PAIR_RECORD_KIND,
            "source_identity": identity_payload,
            "structure_equal": structure_equal,
            "training_eligible": eligible,
            "zh_hans": hans,
            "zh_hant": hant,
        })
    summary = {
        "archive_file_count": len(file_records),
        "common_entry_count": len(pairs),
        **counts,
        "locale_summaries": locale_summaries,
        "source_format_policy": {
            "bbcode_html_placeholder_structure_preserved": 1,
            "fuzzy_or_empty_training_allowed": 0,
            "plural_training_allowed_in_v1": 0,
            "po_parser": "polib",
        },
    }
    return file_records, tuple(pairs), summary


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
            "commit": GODOT_COMMIT,
            "commit_date": GODOT_COMMIT_DATE,
            "member_git_blob_sha1": {
                key: value["git_blob_sha1"]
                for key, value in sorted(GODOT_SOURCE_FILES.items())
            },
            "repository_url": GODOT_REPOSITORY_URL,
            "root_tree": GODOT_ROOT_TREE,
        },
        "artifact_kind": NORMALIZATION_RECOVERY_V3_GODOT_SOURCE_PACK_KIND,
        "evaluation_or_reserve_read_count": 0,
        "excluded_training_source": {
            "all_derivatives_excluded": 1,
            "source_pack_manifest_sha256": (
                EXCLUDED_FIREFOX_SOURCE_PACK_MANIFEST_SHA256),
        },
        "files": files,
        "format_version": 1,
        "license": {
            "attribution": "Godot Engine contributors; fixed editor locale files",
            "license_id": GODOT_LICENSE_ID,
            "license_url": GODOT_LICENSE_URL,
            "sha256": GODOT_SOURCE_FILES["LICENSE.txt"]["sha256"],
        },
        "mastery_claimed": 0,
        "parser": {
            "polib_version": POLIB_VERSION,
            "polib_wheel_sha256": POLIB_WHEEL_SHA256,
        },
        "parser_summary": parser_summary,
        "production_enabled": 0,
        "source_family": "GODOT_ENGINE_PROJECT",
        "source_policy_scope": "GODOT_EDITOR_ZH_HANT_TO_ZH_HANS_V1",
        "status": NORMALIZATION_RECOVERY_V3_GODOT_SOURCE_STATUS,
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
        file_records: tuple[dict[str, object], ...],
        summary: dict[str, object],
        ) -> None:
    """核验固定 archive、Git blob、许可和 parser 全量库存。"""
    expected_files = tuple({
        "bytes": value["bytes"],
        "git_blob_sha1": value["git_blob_sha1"],
        "relative_path": path,
        "sha256": value["sha256"],
    } for path, value in sorted(GODOT_SOURCE_FILES.items()))
    locale_summaries = summary.get("locale_summaries")
    if (len(archive_payload) != GODOT_ARCHIVE_BYTES
            or _sha256(archive_payload) != GODOT_ARCHIVE_SHA256
            or not _strict_equal(file_records, expected_files)
            or summary.get("archive_file_count") != 3
            or summary.get("common_entry_count") != GODOT_COMMON_ENTRY_COUNT
            or summary.get("plural_pair_count") != GODOT_PLURAL_ENTRY_COUNT
            or summary.get("simple_pair_count") != GODOT_SIMPLE_PAIR_COUNT
            or summary.get("structure_equal_count")
            != GODOT_STRUCTURE_EQUAL_COUNT
            or summary.get("training_eligible_count")
            != GODOT_TRAINING_ELIGIBLE_COUNT
            or summary.get("nonidentity_pair_count")
            != GODOT_NONIDENTITY_PAIR_COUNT
            or summary.get("identity_pair_count") != GODOT_IDENTITY_PAIR_COUNT
            or summary.get("equal_length_pair_count")
            != GODOT_EQUAL_LENGTH_PAIR_COUNT
            or summary.get("variable_length_pair_count")
            != GODOT_VARIABLE_LENGTH_PAIR_COUNT
            or not isinstance(locale_summaries, dict)
            or any(locale_summaries.get(locale, {}).get("entry_count")
                   != count for locale, count in GODOT_ENTRY_COUNTS.items())):
        raise BroadQaExternalDataError("Godot official source identity 漂移")


def publish_normalization_recovery_v3_godot_source_pack(
        *,
        run_root: str | Path,
        archive_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 Godot recovery-v3 source pack。"""
    root = _require_k_root(run_root)
    source = Path(archive_path).resolve()
    target = Path(target_dir).resolve()
    if (not source.is_file() or not source.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "normalization recovery v3 Godot source path 越界")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization recovery v3 Godot source target 已存在")
    archive_payload = source.read_bytes()
    file_records, pairs, summary = (
        parse_normalization_recovery_v3_godot_archive(archive_payload))
    _validate_official_source(archive_payload, file_records, summary)
    target.mkdir(parents=True)
    archive_target = target / GODOT_ARCHIVE_NAME
    with archive_target.open("xb") as handle:
        handle.write(archive_payload)
    file_path = target / "source-files.jsonl"
    pair_path = target / "translation-pairs.jsonl"
    _write_jsonl(file_path, file_records)
    _write_jsonl(pair_path, pairs)
    files = [
        _artifact(archive_target, role="GODOT_EDITOR_RAW_ARCHIVE", count=0),
        _artifact(file_path, role="GODOT_EDITOR_SOURCE_FILES",
                  count=len(file_records)),
        _artifact(pair_path, role="GODOT_EDITOR_TRANSLATION_PAIRS",
                  count=len(pairs)),
    ]
    manifest = _manifest(files=files, parser_summary=summary)
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def read_normalization_recovery_v3_godot_source_pack(
        source_pack_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """从 raw Git blob archive 重派生并严格回读 source pack。"""
    root = Path(source_pack_dir).resolve()
    try:
        encoded_manifest = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded_manifest)
        archive_payload = (root / GODOT_ARCHIVE_NAME).read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization recovery v3 Godot source pack 不可读") from error
    if (not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded_manifest):
        raise BroadQaExternalDataError(
            "normalization recovery v3 Godot source manifest 非规范")
    derived_files, derived_pairs, summary = (
        parse_normalization_recovery_v3_godot_archive(archive_payload))
    _validate_official_source(archive_payload, derived_files, summary)
    stored_files = _read_jsonl(
        root / "source-files.jsonl", label="Godot source files")
    stored_pairs = _read_jsonl(
        root / "translation-pairs.jsonl", label="Godot translation pairs")
    if (not _strict_equal(stored_files, derived_files)
            or not _strict_equal(stored_pairs, derived_pairs)):
        raise BroadQaExternalDataError(
            "normalization recovery v3 Godot records/source 漂移")
    files = [
        _artifact(root / GODOT_ARCHIVE_NAME,
                  role="GODOT_EDITOR_RAW_ARCHIVE", count=0),
        _artifact(root / "source-files.jsonl",
                  role="GODOT_EDITOR_SOURCE_FILES", count=len(derived_files)),
        _artifact(root / "translation-pairs.jsonl",
                  role="GODOT_EDITOR_TRANSLATION_PAIRS",
                  count=len(derived_pairs)),
    ]
    expected = _manifest(files=files, parser_summary=summary)
    if not _strict_equal(stored, expected):
        raise BroadQaExternalDataError(
            "normalization recovery v3 Godot source manifest 漂移")
    return (
        {**stored, "manifest_sha256": _sha256(encoded_manifest)},
        derived_files,
        derived_pairs,
    )


__all__ = [
    "GODOT_ARCHIVE_NAME",
    "GODOT_ARCHIVE_SHA256",
    "GODOT_COMMIT",
    "GODOT_PO_PAIR_RECORD_KIND",
    "NORMALIZATION_RECOVERY_V3_GODOT_SOURCE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V3_GODOT_SOURCE_STATUS",
    "parse_normalization_recovery_v3_godot_archive",
    "publish_normalization_recovery_v3_godot_source_pack",
    "read_normalization_recovery_v3_godot_source_pack",
]
