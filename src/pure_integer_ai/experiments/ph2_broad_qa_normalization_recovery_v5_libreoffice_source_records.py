"""从固定 LibreOffice `cui/messages.po` archive 派生来源记录。

adapter 使用 `polib` 按 `msgctxt/msgid/msgid_plural` 对齐 zh-TW/zh-CN，
保留原 entry、来源文件和嵌入结构。它不打开路径，也不选择训练 rule。
"""
from __future__ import annotations

from collections import Counter

import polib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    git_blob_sha1,
    localization_pair_features,
    localization_record_id,
    read_exact_localization_zip,
    sha256_hex,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


LIBREOFFICE_SOURCE_FILE_RECORD_KIND = (
    "LIBREOFFICE_LOCALIZATION_SOURCE_FILE_V1")
LIBREOFFICE_PO_PAIR_RECORD_KIND = "LIBREOFFICE_CUI_PO_PAIR_V1"
LIBREOFFICE_PAIR_TEXT_SCALAR_MAX = 320
LIBREOFFICE_SOURCE_PATHS = {
    "zh_Hans": "source/zh-CN/cui/messages.po",
    "zh_Hant": "source/zh-TW/cui/messages.po",
}
LIBREOFFICE_ARCHIVE_FILES = (
    "README",
    LIBREOFFICE_SOURCE_PATHS["zh_Hans"],
    LIBREOFFICE_SOURCE_PATHS["zh_Hant"],
)


def _source_file_records(
        files: dict[str, bytes],
        ) -> tuple[dict[str, object], ...]:
    """为 archive 中每个固定 Git blob 形成来源记录。"""
    values = []
    locale_by_path = {
        path: locale for locale, path in LIBREOFFICE_SOURCE_PATHS.items()}
    for relative_path, payload in sorted(files.items()):
        identity = {
            "git_blob_sha1": git_blob_sha1(payload),
            "relative_path": relative_path,
            "sha256": sha256_hex(payload),
        }
        values.append({
            **identity,
            "bytes": len(payload),
            "file_id": localization_record_id(identity),
            "format_version": 1,
            "locale": locale_by_path.get(relative_path, ""),
            "record_kind": LIBREOFFICE_SOURCE_FILE_RECORD_KIND,
            "role": (
                "TRANSLATION_PO"
                if relative_path in locale_by_path else "SOURCE_AUXILIARY"),
        })
    return tuple(values)


def _entry_key(entry: polib.POEntry) -> tuple[str, str, str]:
    """返回跨 locale 对齐的英文 source identity。"""
    return entry.msgctxt or "", entry.msgid, entry.msgid_plural or ""


def _entry_record(
        entry: polib.POEntry,
        *,
        locale: str,
        ordinal: int,
        source_file_id: str,
        source_file_sha256: str,
        ) -> dict[str, object]:
    """把 polib entry 变为带物理来源身份的规范记录。"""
    semantic = {
        "comment": entry.comment or "",
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
        "previous_msgctxt": entry.previous_msgctxt or "",
        "previous_msgid": entry.previous_msgid or "",
        "previous_msgid_plural": entry.previous_msgid_plural or "",
        "tcomment": entry.tcomment or "",
    }
    return {
        **semantic,
        "entry_linenum": entry.linenum,
        "entry_ordinal": ordinal,
        "entry_semantic_sha256": sha256_hex(canonical_json_bytes(semantic)),
        "locale": locale,
        "source_file_id": source_file_id,
        "source_file_sha256": source_file_sha256,
    }


def _parse_po(
        payload: bytes,
        *,
        locale: str,
        source_file_id: str,
        ) -> tuple[
            dict[tuple[str, str, str], dict[str, object]],
            dict[str, object],
        ]:
    """严格解析 UTF-8 PO，并拒绝重复 source identity。"""
    if (not isinstance(payload, bytes) or payload.startswith(b"\xef\xbb\xbf")
            or not payload.endswith(b"\n")):
        raise BroadQaExternalDataError(
            "LibreOffice PO 编码或结尾换行非法")
    try:
        text = payload.decode("utf-8")
        parsed = polib.pofile(text, wrapwidth=0)
    except (UnicodeDecodeError, OSError, ValueError) as error:
        raise BroadQaExternalDataError(
            "LibreOffice PO parser 失败") from error
    if parsed.metadata.get("Language") != locale:
        raise BroadQaExternalDataError("LibreOffice PO locale 漂移")
    records = {}
    for ordinal, entry in enumerate(parsed):
        key = _entry_key(entry)
        if key in records:
            raise BroadQaExternalDataError(
                "LibreOffice PO source identity 重复")
        records[key] = _entry_record(
            entry,
            locale=locale,
            ordinal=ordinal,
            source_file_id=source_file_id,
            source_file_sha256=sha256_hex(payload),
        )
    summary = {
        "empty_translation_count": sum(not entry.msgstr for entry in parsed),
        "entry_count": len(parsed),
        "fuzzy_count": sum("fuzzy" in entry.flags for entry in parsed),
        "language": parsed.metadata.get("Language"),
        "metadata_sha256": sha256_hex(canonical_json_bytes(parsed.metadata)),
        "obsolete_count": sum(int(entry.obsolete) for entry in parsed),
        "plural_count": sum(bool(entry.msgid_plural) for entry in parsed),
        "translated_count": len(parsed.translated_entries()),
    }
    return records, summary


def _plain_pair(
        zh_hans: dict[str, object],
        zh_hant: dict[str, object],
        ) -> bool:
    """判断 entry pair 是否进入固定 plain pair inventory。"""
    return bool(
        not zh_hans["msgid_plural"]
        and not zh_hant["msgid_plural"]
        and zh_hans["obsolete"] == 0
        and zh_hant["obsolete"] == 0
        and "fuzzy" not in zh_hans["flags"]
        and "fuzzy" not in zh_hant["flags"]
        and zh_hans["msgstr"]
        and zh_hant["msgstr"])


def _translation_pairs(
        locale_records: dict[
            str, dict[tuple[str, str, str], dict[str, object]]],
        ) -> tuple[dict[str, object], ...]:
    """按完整 source identity 派生 plain 简繁 pair。"""
    common = sorted(set(locale_records["zh_Hans"]).intersection(
        locale_records["zh_Hant"]))
    values = []
    for key in common:
        zh_hans = locale_records["zh_Hans"][key]
        zh_hant = locale_records["zh_Hant"][key]
        if not _plain_pair(zh_hans, zh_hant):
            continue
        source_identity = {
            "msgctxt": key[0],
            "msgid": key[1],
            "msgid_plural": key[2],
        }
        features = localization_pair_features(
            str(zh_hant["msgstr"]),
            str(zh_hans["msgstr"]),
            scalar_limit=LIBREOFFICE_PAIR_TEXT_SCALAR_MAX,
        )
        values.append({
            **features,
            "format_version": 1,
            "pair_id": localization_record_id({
                "record_kind": LIBREOFFICE_PO_PAIR_RECORD_KIND,
                "source_identity": source_identity,
            }),
            "record_kind": LIBREOFFICE_PO_PAIR_RECORD_KIND,
            "source_identity": source_identity,
            "source_identity_sha256": sha256_hex(
                canonical_json_bytes(source_identity)),
            "zh_hans": zh_hans,
            "zh_hant": zh_hant,
        })
    if (not values or len({item["pair_id"] for item in values})
            != len(values)):
        raise BroadQaExternalDataError(
            "LibreOffice translation pair identity 非法")
    return tuple(values)


def _summary(
        file_records: tuple[dict[str, object], ...],
        pairs: tuple[dict[str, object], ...],
        locale_summaries: dict[str, dict[str, object]],
        ) -> dict[str, object]:
    """汇总文件、entry、pair、冲突与结构资格库存。"""
    counts = Counter()
    outputs_by_input: dict[str, set[str]] = {}
    for item in pairs:
        identity = int(item["identity_preservation"])
        counts["identity_pair_count" if identity else
               "nonidentity_pair_count"] += 1
        counts["equal_length_pair_count" if item["equal_length"] else
               "variable_length_pair_count"] += 1
        counts["single_han_difference_count"] += int(
            item["single_han_difference"])
        counts["structure_equal_count"] += int(item["structure_equal"])
        counts["training_eligible_pair_count"] += int(
            item["training_eligible"])
        input_text = str(item["zh_hant"]["msgstr"])
        outputs_by_input.setdefault(input_text, set()).add(
            str(item["zh_hans"]["msgstr"]))
    return {
        "archive_file_count": len(file_records),
        "equal_length_pair_count": counts["equal_length_pair_count"],
        "identity_pair_count": counts["identity_pair_count"],
        "input_conflict_count": sum(
            len(outputs) > 1 for outputs in outputs_by_input.values()),
        "locale_summaries": locale_summaries,
        "nonidentity_pair_count": counts["nonidentity_pair_count"],
        "plain_pair_count": len(pairs),
        "single_han_difference_count": counts[
            "single_han_difference_count"],
        "source_format_policy": {
            "empty_fuzzy_obsolete_or_plural_pair_allowed": 0,
            "parser": "polib",
            "source_identity": "MSGCTXT_MSGID_MSGID_PLURAL",
            "structure_and_original_text_preserved": 1,
        },
        "structure_equal_count": counts["structure_equal_count"],
        "training_eligible_pair_count": counts[
            "training_eligible_pair_count"],
        "variable_length_pair_count": counts["variable_length_pair_count"],
    }


def parse_normalization_recovery_v5_libreoffice_archive(
        archive_payload: bytes,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """从固定 archive 派生来源文件、简繁 pair 与完整 census。"""
    files = read_exact_localization_zip(
        archive_payload,
        expected_files=LIBREOFFICE_ARCHIVE_FILES,
        label="LibreOffice recovery-v5 source",
        member_count_max=8,
        uncompressed_bytes_max=4 * 1024 * 1024,
    )
    file_records = _source_file_records(files)
    file_by_path = {
        str(item["relative_path"]): item for item in file_records}
    locale_records = {}
    locale_summaries = {}
    for locale, path in LIBREOFFICE_SOURCE_PATHS.items():
        records, summary = _parse_po(
            files[path],
            locale="zh-CN" if locale == "zh_Hans" else "zh-TW",
            source_file_id=str(file_by_path[path]["file_id"]),
        )
        locale_records[locale] = records
        locale_summaries[locale] = summary
    pairs = _translation_pairs(locale_records)
    return (
        file_records,
        pairs,
        _summary(file_records, pairs, locale_summaries),
    )


__all__ = [
    "LIBREOFFICE_ARCHIVE_FILES",
    "LIBREOFFICE_PAIR_TEXT_SCALAR_MAX",
    "LIBREOFFICE_PO_PAIR_RECORD_KIND",
    "LIBREOFFICE_SOURCE_FILE_RECORD_KIND",
    "LIBREOFFICE_SOURCE_PATHS",
    "parse_normalization_recovery_v5_libreoffice_archive",
]
