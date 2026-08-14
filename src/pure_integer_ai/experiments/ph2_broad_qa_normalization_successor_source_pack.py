"""冻结 normalization successor 的 Unihan 与 MediaWiki 独立评测来源。

本模块只解析、发布和严格回读官方来源。它不读取 OpenCC/ICU learner、
candidate、已消费 formal report 或 reserve label，也不运行任何评测。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_SUCCESSOR_SOURCE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_SUCCESSOR_EVALUATION_SOURCE_PACK_V1")
NORMALIZATION_SUCCESSOR_SOURCE_STATUS = (
    "INDEPENDENT_SOURCES_FROZEN_NOT_SELECTED_NOT_EVALUATED")

UNIHAN_VERSION = "17.0.0"
UNIHAN_ARCHIVE_URL = "https://www.unicode.org/Public/17.0.0/ucd/Unihan.zip"
UNIHAN_LICENSE_URL = "https://www.unicode.org/license.txt"
UNIHAN_LICENSE_ID = "Unicode-3.0"
UNIHAN_ARCHIVE_SHA256 = (
    "f7a48b2b545acfaa77b2d607ae28747404ce02baefee16396c5d2d7a8ef34b5e")
UNIHAN_ARCHIVE_BYTES = 8_518_517
UNIHAN_LICENSE_SHA256 = (
    "e7a93b009565cfce55919a381437ac4db883e9da2126fa28b91d12732bc53d96")
UNIHAN_LICENSE_BYTES = 1_995
UNIHAN_VARIANTS_MEMBER = "Unihan_Variants.txt"

MEDIAWIKI_COMMIT = "a6cf530da1712483040708d5fa499eb2cf0d024c"
MEDIAWIKI_COMMIT_DATE = "2026-08-13T19:42:02Z"
MEDIAWIKI_TREE = "41c0df0edf3a673b04dba661dbb6508ff7309a84"
MEDIAWIKI_BLOB_SHA = "43d4dd07a761ab091421347a9b0a5c44a3ccdaba"
MEDIAWIKI_RULE_PATH = "includes/Languages/Data/ZhConversion.php"
MEDIAWIKI_RULE_URL = (
    "https://raw.githubusercontent.com/wikimedia/mediawiki/"
    f"{MEDIAWIKI_COMMIT}/{MEDIAWIKI_RULE_PATH}")
MEDIAWIKI_LICENSE_URL = (
    "https://raw.githubusercontent.com/wikimedia/mediawiki/"
    f"{MEDIAWIKI_COMMIT}/COPYING")
MEDIAWIKI_LICENSE_ID = "GPL-2.0-or-later"
MEDIAWIKI_RULE_SHA256 = (
    "963792970f8a15a37d299a93c91e63f5324f5b8cb06c0f4ec9249fa5489518a5")
MEDIAWIKI_RULE_BYTES = 500_733
MEDIAWIKI_LICENSE_SHA256 = (
    "ea5429175b5ed3b83131d2a3b848ef1bd0142a6192e1027be9e1efaf031a3928")
MEDIAWIKI_LICENSE_BYTES = 19_309

UNIHAN_VARIANT_RECORD_KIND = "UNICODE_UNIHAN_VARIANT_OBSERVATION_V1"
MEDIAWIKI_CONVERSION_RECORD_KIND = "MEDIAWIKI_ZH_CONVERSION_OBSERVATION_V1"
UNIHAN_VARIANT_PROPERTIES = (
    "kSemanticVariant",
    "kSimplifiedVariant",
    "kSpecializedSemanticVariant",
    "kSpoofingVariant",
    "kTraditionalVariant",
    "kZVariant",
)
MEDIAWIKI_CONVERSION_TABLES = (
    "ZH_TO_HANT", "ZH_TO_HANS", "ZH_TO_TW", "ZH_TO_HK", "ZH_TO_CN")

_UNIHAN_ROW = re.compile(
    r"^(U\+[0-9A-F]{4,6})\t(k[A-Za-z]+Variant)\t([^\t]+)$")
_UNIHAN_TARGET = re.compile(
    r"^(U\+[0-9A-F]{4,6})(?:<([ks][A-Za-z0-9]+(?::(?:T|Z|TZ))?"
    r"(?:,[ks][A-Za-z0-9]+(?::(?:T|Z|TZ))?)*))?$")
_MEDIAWIKI_TABLE_OPEN = re.compile(r"^\tpublic const ([A-Z_]+) = \[$")
_MEDIAWIKI_ENTRY = re.compile(r"^\t\t'([^'\\]*)' => '([^'\\]*)',$")


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
    if isinstance(expected, list):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def _require_scalar(value: str, *, label: str) -> tuple[int, str]:
    """解析 ``U+`` 标记并拒绝越界或 surrogate 码点。"""
    try:
        codepoint = int(value[2:], 16)
    except (TypeError, ValueError) as error:
        raise BroadQaExternalDataError(f"{label} 非法") from error
    if codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
        raise BroadQaExternalDataError(f"{label} 不是 Unicode scalar")
    return codepoint, chr(codepoint)


def _zip_member(payload: bytes) -> tuple[bytes, dict[str, object]]:
    """从 Unihan ZIP 严格读取唯一 variants 成员及其身份。"""
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            matches = [item for item in archive.infolist()
                       if item.filename == UNIHAN_VARIANTS_MEMBER]
            if len(matches) != 1 or matches[0].is_dir():
                raise BroadQaExternalDataError(
                    "Unihan archive 缺少唯一 variants member")
            info = matches[0]
            member = archive.read(info)
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise BroadQaExternalDataError("Unihan archive 非法") from error
    return member, {
        "compressed_bytes": info.compress_size,
        "crc32": f"{info.CRC:08x}",
        "member_path": info.filename,
        "sha256": _sha256(member),
        "uncompressed_bytes": info.file_size,
    }


def _encoded_lf_lines(payload: bytes, *, label: str) -> tuple[bytes, ...]:
    """要求来源为无 BOM、完整 LF 的 UTF-8 物理行。"""
    if payload.startswith(b"\xef\xbb\xbf"):
        raise BroadQaExternalDataError(f"{label} 不得含 UTF-8 BOM")
    lines = tuple(payload.splitlines(keepends=True))
    if (not lines or any(not item.endswith(b"\n") for item in lines)
            or any(item.endswith(b"\r\n") for item in lines)):
        raise BroadQaExternalDataError(f"{label} 必须是完整 LF 物理行")
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BroadQaExternalDataError(f"{label} 非 UTF-8") from error
    return lines


def parse_normalization_unihan_source(
        archive_payload: bytes,
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """解析 Unihan variant 行并显式投影唯一简化关系。"""
    if not isinstance(archive_payload, bytes):
        raise BroadQaExternalDataError("Unihan archive payload 非 bytes")
    member, member_identity = _zip_member(archive_payload)
    lines = _encoded_lf_lines(member, label="Unihan variants member")
    records = []
    byte_start = 0
    identities = set()
    property_counts: Counter[str] = Counter()
    for line_ordinal, encoded in enumerate(lines, start=1):
        byte_end = byte_start + len(encoded)
        text = encoded[:-1].decode("utf-8")
        if not text or text.startswith("#"):
            byte_start = byte_end
            continue
        match = _UNIHAN_ROW.fullmatch(text)
        if match is None or match.group(2) not in UNIHAN_VARIANT_PROPERTIES:
            raise BroadQaExternalDataError(
                f"Unihan variants row syntax 漂移: {line_ordinal}")
        source_token, property_name, raw_targets = match.groups()
        source_codepoint, source_text = _require_scalar(
            source_token, label="Unihan source codepoint")
        targets = []
        for raw_target in raw_targets.split(" "):
            target_match = _UNIHAN_TARGET.fullmatch(raw_target)
            if target_match is None:
                raise BroadQaExternalDataError(
                    f"Unihan target syntax 漂移: {line_ordinal}")
            target_codepoint, target_text = _require_scalar(
                target_match.group(1), label="Unihan target codepoint")
            sources = ([] if target_match.group(2) is None
                       else target_match.group(2).split(","))
            targets.append({
                "codepoint": target_codepoint,
                "source_tags": sources,
                "text": target_text,
                "uplus": target_match.group(1),
            })
        identity = (source_codepoint, property_name)
        if not targets or identity in identities:
            raise BroadQaExternalDataError(
                "Unihan variant identity 重复或无 target")
        identities.add(identity)
        eligible = int(
            property_name == "kSimplifiedVariant"
            and len(targets) == 1
            and not targets[0]["source_tags"]
            and targets[0]["codepoint"] != source_codepoint)
        records.append({
            "byte_end": byte_end,
            "byte_start": byte_start,
            "format_version": 1,
            "line_ordinal": line_ordinal,
            "line_sha256": _sha256(encoded),
            "property_name": property_name,
            "record_kind": UNIHAN_VARIANT_RECORD_KIND,
            "source_codepoint": source_codepoint,
            "source_text": source_text,
            "source_uplus": source_token,
            "t2s_expected_output": targets[0]["text"] if eligible else "",
            "t2s_input": source_text if eligible else "",
            "t2s_unambiguous_eligible": eligible,
            "targets": targets,
        })
        property_counts[property_name] += 1
        byte_start = byte_end
    if byte_start != len(member) or not records:
        raise BroadQaExternalDataError("Unihan variants 物理覆盖漂移")
    summary = {
        "member_identity": member_identity,
        "physical_line_count": len(lines),
        "property_counts": {
            key: property_counts[key] for key in UNIHAN_VARIANT_PROPERTIES},
        "record_count": len(records),
        "t2s_unambiguous_eligible_count": sum(
            item["t2s_unambiguous_eligible"] for item in records),
    }
    return tuple(records), summary


def _mediawiki_outside_line_allowed(
        text: str,
        *,
        in_comment: bool,
        ) -> tuple[bool, bool]:
    """核验生成文件数组外的有限 PHP 壳并推进注释状态。"""
    if in_comment:
        return True, not text.endswith("*/")
    if text == "/**":
        return True, True
    allowed = {
        "", "<?php", "namespace MediaWiki\\Languages\\Data;",
        "class ZhConversion {", "}",
    }
    return text in allowed, False


def parse_normalization_mediawiki_source(
        payload: bytes,
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """以严格生成格式解析 MediaWiki 五张中文转换表。"""
    if not isinstance(payload, bytes):
        raise BroadQaExternalDataError("MediaWiki conversion payload 非 bytes")
    lines = _encoded_lf_lines(payload, label="MediaWiki conversion source")
    records = []
    table_name: str | None = None
    table_order = []
    table_keys: dict[str, set[str]] = {}
    table_counts: Counter[str] = Counter()
    identity_counts: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()
    byte_start = 0
    in_comment = False
    for line_ordinal, encoded in enumerate(lines, start=1):
        byte_end = byte_start + len(encoded)
        text = encoded[:-1].decode("utf-8")
        if table_name is None:
            opened = _MEDIAWIKI_TABLE_OPEN.fullmatch(text)
            if opened is not None:
                value = opened.group(1)
                if (value not in MEDIAWIKI_CONVERSION_TABLES
                        or value in table_keys):
                    raise BroadQaExternalDataError(
                        "MediaWiki conversion table 重复或未知")
                table_name = value
                table_order.append(value)
                table_keys[value] = set()
            else:
                allowed, in_comment = _mediawiki_outside_line_allowed(
                    text, in_comment=in_comment)
                if not allowed:
                    raise BroadQaExternalDataError(
                        f"MediaWiki PHP 壳 syntax 漂移: {line_ordinal}")
        elif text == "\t];":
            table_name = None
        else:
            match = _MEDIAWIKI_ENTRY.fullmatch(text)
            if match is None:
                raise BroadQaExternalDataError(
                    f"MediaWiki conversion entry syntax 漂移: {line_ordinal}")
            input_text, expected_output = match.groups()
            if (not input_text or not expected_output
                    or any(0xD800 <= ord(value) <= 0xDFFF
                           for value in input_text + expected_output)
                    or input_text in table_keys[table_name]):
                raise BroadQaExternalDataError(
                    "MediaWiki conversion entry 内容非法或重复")
            table_keys[table_name].add(input_text)
            input_count = len(input_text)
            output_count = len(expected_output)
            records.append({
                "byte_end": byte_end,
                "byte_start": byte_start,
                "expected_output": expected_output,
                "format_version": 1,
                "input_scalar_count": input_count,
                "input_text": input_text,
                "is_identity": int(input_text == expected_output),
                "line_ordinal": line_ordinal,
                "line_sha256": _sha256(encoded),
                "output_scalar_count": output_count,
                "record_kind": MEDIAWIKI_CONVERSION_RECORD_KIND,
                "table_name": table_name,
            })
            table_counts[table_name] += 1
            identity_counts[table_name] += int(input_text == expected_output)
            phrase_counts[table_name] += int(input_count >= 2)
        byte_start = byte_end
    if (byte_start != len(payload) or table_name is not None or in_comment
            or tuple(table_order) != MEDIAWIKI_CONVERSION_TABLES
            or not records):
        raise BroadQaExternalDataError(
            "MediaWiki conversion source 结构或物理覆盖漂移")
    summary = {
        "identity_counts": {
            key: identity_counts[key] for key in MEDIAWIKI_CONVERSION_TABLES},
        "physical_line_count": len(lines),
        "phrase_counts": {
            key: phrase_counts[key] for key in MEDIAWIKI_CONVERSION_TABLES},
        "record_count": len(records),
        "table_counts": {
            key: table_counts[key] for key in MEDIAWIKI_CONVERSION_TABLES},
        "table_order": list(table_order),
    }
    return tuple(records), summary


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


def _file_record(path: Path, *, role: str, count: int) -> dict[str, object]:
    """返回一个已发布文件的规范身份。"""
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
        unihan_summary: dict[str, object],
        mediawiki_summary: dict[str, object],
        ) -> dict[str, object]:
    """构造零消费、可重算的 successor source manifest。"""
    return {
        "artifact_kind": NORMALIZATION_SUCCESSOR_SOURCE_PACK_KIND,
        "candidate_pack_read_count": 0,
        "evaluation_run_count": 0,
        "failed_icu_evaluation_read_count": 0,
        "files": files,
        "format_version": 1,
        "learned_pack_read_count": 0,
        "mediawiki": {
            "blob_sha": MEDIAWIKI_BLOB_SHA,
            "commit": MEDIAWIKI_COMMIT,
            "commit_date": MEDIAWIKI_COMMIT_DATE,
            "license_id": MEDIAWIKI_LICENSE_ID,
            "license_url": MEDIAWIKI_LICENSE_URL,
            "rule_path": MEDIAWIKI_RULE_PATH,
            "rule_url": MEDIAWIKI_RULE_URL,
            "summary": mediawiki_summary,
            "tree_sha": MEDIAWIKI_TREE,
        },
        "mastery_claimed": 0,
        "production_enabled": 0,
        "reserve_label_read_count": 0,
        "status": NORMALIZATION_SUCCESSOR_SOURCE_STATUS,
        "teacher_api_llm_call_count": 0,
        "unihan": {
            "archive_url": UNIHAN_ARCHIVE_URL,
            "license_id": UNIHAN_LICENSE_ID,
            "license_url": UNIHAN_LICENSE_URL,
            "summary": unihan_summary,
            "version": UNIHAN_VERSION,
        },
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization successor run root 必须是 K 盘目录")
    return root


def publish_normalization_successor_source_pack(
        *,
        run_root: str | Path,
        unihan_archive_path: str | Path,
        unihan_license_path: str | Path,
        mediawiki_rule_path: str | Path,
        mediawiki_license_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """从显式 K 盘官方字节不可覆盖发布 successor source pack。"""
    root = _require_k_root(run_root)
    paths = tuple(Path(value).resolve() for value in (
        unihan_archive_path, unihan_license_path,
        mediawiki_rule_path, mediawiki_license_path))
    target = Path(target_dir).resolve()
    if (any(not path.is_file() or not path.is_relative_to(root)
            for path in paths) or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "normalization successor source/target 越出 run root")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization successor source target 已存在")
    payloads = tuple(path.read_bytes() for path in paths)
    expected = (
        (UNIHAN_ARCHIVE_BYTES, UNIHAN_ARCHIVE_SHA256),
        (UNIHAN_LICENSE_BYTES, UNIHAN_LICENSE_SHA256),
        (MEDIAWIKI_RULE_BYTES, MEDIAWIKI_RULE_SHA256),
        (MEDIAWIKI_LICENSE_BYTES, MEDIAWIKI_LICENSE_SHA256),
    )
    if any(len(payload) != size or _sha256(payload) != digest
           for payload, (size, digest) in zip(payloads, expected)):
        raise BroadQaExternalDataError(
            "normalization successor official source identity 漂移")
    try:
        unihan_license = payloads[1].decode("utf-8")
        mediawiki_license = payloads[3].decode("utf-8")
    except UnicodeDecodeError as error:
        raise BroadQaExternalDataError(
            "normalization successor license 非 UTF-8") from error
    if ("UNICODE LICENSE V3" not in unihan_license
            or "GNU General Public License" not in mediawiki_license
            or "version 2 or later" not in mediawiki_license):
        raise BroadQaExternalDataError(
            "normalization successor license 文本漂移")
    unihan_records, unihan_summary = parse_normalization_unihan_source(
        payloads[0])
    mediawiki_records, mediawiki_summary = (
        parse_normalization_mediawiki_source(payloads[2]))
    target.mkdir(parents=True)
    outputs = (
        ("Unihan-17.0.0.zip", payloads[0]),
        ("UNICODE-LICENSE.txt", payloads[1]),
        ("MediaWiki-ZhConversion.php", payloads[2]),
        ("MediaWiki-COPYING", payloads[3]),
    )
    for name, payload in outputs:
        (target / name).write_bytes(payload)
    _write_jsonl(target / "unihan-variants.jsonl", unihan_records)
    _write_jsonl(target / "mediawiki-conversions.jsonl", mediawiki_records)
    files = [
        _file_record(target / name, role=role, count=count)
        for name, role, count in (
            ("MediaWiki-COPYING", "MEDIAWIKI_LICENSE", 0),
            ("MediaWiki-ZhConversion.php", "MEDIAWIKI_RULE_SOURCE", 0),
            ("UNICODE-LICENSE.txt", "UNIHAN_LICENSE", 0),
            ("Unihan-17.0.0.zip", "UNIHAN_ARCHIVE_SOURCE", 0),
            ("mediawiki-conversions.jsonl", "MEDIAWIKI_PARSED_RECORDS",
             len(mediawiki_records)),
            ("unihan-variants.jsonl", "UNIHAN_PARSED_RECORDS",
             len(unihan_records)),
        )
    ]
    manifest = _manifest(
        files=files, unihan_summary=unihan_summary,
        mediawiki_summary=mediawiki_summary)
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def read_normalization_successor_source_pack(
        source_pack_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """从原始官方字节重派生并严格回读 successor source pack。"""
    root = Path(source_pack_dir).resolve()
    manifest_path = root / "manifest.json"
    try:
        encoded_manifest = manifest_path.read_bytes()
        stored = json.loads(encoded_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization successor source manifest 不可读") from error
    if (not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded_manifest):
        raise BroadQaExternalDataError(
            "normalization successor source manifest 非规范")
    try:
        unihan_payload = (root / "Unihan-17.0.0.zip").read_bytes()
        unihan_license = (root / "UNICODE-LICENSE.txt").read_bytes()
        mediawiki_payload = (root / "MediaWiki-ZhConversion.php").read_bytes()
        mediawiki_license = (root / "MediaWiki-COPYING").read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError(
            "normalization successor source raw file 不可读") from error
    expected_identity = (
        (unihan_payload, UNIHAN_ARCHIVE_BYTES, UNIHAN_ARCHIVE_SHA256),
        (unihan_license, UNIHAN_LICENSE_BYTES, UNIHAN_LICENSE_SHA256),
        (mediawiki_payload, MEDIAWIKI_RULE_BYTES, MEDIAWIKI_RULE_SHA256),
        (mediawiki_license, MEDIAWIKI_LICENSE_BYTES,
         MEDIAWIKI_LICENSE_SHA256),
    )
    if any(len(payload) != size or _sha256(payload) != digest
           for payload, size, digest in expected_identity):
        raise BroadQaExternalDataError(
            "normalization successor source raw identity 漂移")
    derived_unihan, unihan_summary = parse_normalization_unihan_source(
        unihan_payload)
    derived_mediawiki, mediawiki_summary = (
        parse_normalization_mediawiki_source(mediawiki_payload))
    stored_unihan = _read_jsonl(
        root / "unihan-variants.jsonl", label="Unihan variants")
    stored_mediawiki = _read_jsonl(
        root / "mediawiki-conversions.jsonl", label="MediaWiki conversion")
    if (not _strict_equal(stored_unihan, derived_unihan)
            or not _strict_equal(stored_mediawiki, derived_mediawiki)):
        raise BroadQaExternalDataError(
            "normalization successor records/source 漂移")
    files = [
        _file_record(root / name, role=role, count=count)
        for name, role, count in (
            ("MediaWiki-COPYING", "MEDIAWIKI_LICENSE", 0),
            ("MediaWiki-ZhConversion.php", "MEDIAWIKI_RULE_SOURCE", 0),
            ("UNICODE-LICENSE.txt", "UNIHAN_LICENSE", 0),
            ("Unihan-17.0.0.zip", "UNIHAN_ARCHIVE_SOURCE", 0),
            ("mediawiki-conversions.jsonl", "MEDIAWIKI_PARSED_RECORDS",
             len(derived_mediawiki)),
            ("unihan-variants.jsonl", "UNIHAN_PARSED_RECORDS",
             len(derived_unihan)),
        )
    ]
    expected_manifest = _manifest(
        files=files, unihan_summary=unihan_summary,
        mediawiki_summary=mediawiki_summary)
    if not _strict_equal(stored, expected_manifest):
        raise BroadQaExternalDataError(
            "normalization successor source manifest 漂移")
    return (
        {**stored, "manifest_sha256": _sha256(encoded_manifest)},
        derived_unihan,
        derived_mediawiki,
    )


__all__ = [
    "MEDIAWIKI_COMMIT",
    "MEDIAWIKI_CONVERSION_RECORD_KIND",
    "MEDIAWIKI_LICENSE_BYTES",
    "MEDIAWIKI_LICENSE_SHA256",
    "MEDIAWIKI_RULE_BYTES",
    "MEDIAWIKI_RULE_SHA256",
    "NORMALIZATION_SUCCESSOR_SOURCE_PACK_KIND",
    "NORMALIZATION_SUCCESSOR_SOURCE_STATUS",
    "UNIHAN_ARCHIVE_BYTES",
    "UNIHAN_ARCHIVE_SHA256",
    "UNIHAN_LICENSE_BYTES",
    "UNIHAN_LICENSE_SHA256",
    "UNIHAN_VARIANT_RECORD_KIND",
    "UNIHAN_VERSION",
    "parse_normalization_mediawiki_source",
    "parse_normalization_unihan_source",
    "publish_normalization_successor_source_pack",
    "read_normalization_successor_source_pack",
]
