"""从固定 VLC PO archive 派生 v7 held-out 来源记录。

adapter 使用固定 ``polib`` 按 ``msgctxt/msgid/msgid_plural`` 对齐
zh-TW/zh-CN，保留原 entry、物理来源和嵌入结构。它不打开路径，不发布
translation label，也不选择 learner rule。
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


VLC_SOURCE_FILE_RECORD_KIND = "VLC_LOCALIZATION_SOURCE_FILE_V1"
VLC_PO_PAIR_RECORD_KIND = "VLC_HELD_OUT_PO_PAIR_V1"
VLC_PAIR_TEXT_SCALAR_MAX = 320
VLC_SOURCE_PATHS = {
    "zh_Hans": "po/zh_CN.po",
    "zh_Hant": "po/zh_TW.po",
}
VLC_ARCHIVE_FILES = (
    "COPYING",
    VLC_SOURCE_PATHS["zh_Hans"],
    VLC_SOURCE_PATHS["zh_Hant"],
)
VLC_LICENSE_EXPRESSION = "GPL-2.0-or-later"

_PO_LICENSE_NOTICE = (
    b"# This file is distributed under the same license as the VLC package.\n")
_GPL_V2_TITLE = (
    b"                    GNU GENERAL PUBLIC LICENSE\n"
    b"                       Version 2, June 1991\n")
_GPL_OR_LATER_NOTICE = (
    b"it under the terms of the GNU General Public License as published by\n"
    b"    the Free Software Foundation; either version 2 of the License, or\n"
    b"    (at your option) any later version.\n")


def validate_vlc_license(
        copying_payload: bytes,
        *,
        expected_expression: str,
        ) -> None:
    """核验固定 GPL v2 COPYING 与其标准 or-later notice。"""
    if (not isinstance(copying_payload, bytes)
            or expected_expression != VLC_LICENSE_EXPRESSION
            or not copying_payload.startswith(_GPL_V2_TITLE)
            or copying_payload.count(_GPL_V2_TITLE) != 1
            or _GPL_OR_LATER_NOTICE not in copying_payload):
        raise BroadQaExternalDataError("VLC COPYING license 表达式漂移")


def _source_file_records(
        files: dict[str, bytes],
        ) -> tuple[dict[str, object], ...]:
    """为 COPYING 与两份固定 PO Git blob 形成来源记录。"""
    locale_by_path = {
        path: locale for locale, path in VLC_SOURCE_PATHS.items()}
    values = []
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
            "record_kind": VLC_SOURCE_FILE_RECORD_KIND,
            "role": (
                "TRANSLATION_PO"
                if relative_path in locale_by_path else "LICENSE_TEXT"),
        })
    return tuple(values)


def _entry_key(entry: polib.POEntry) -> tuple[str, str, str]:
    """返回跨 locale 对齐的完整英文 source identity。"""
    return entry.msgctxt or "", entry.msgid, entry.msgid_plural or ""


def _entry_record(
        entry: polib.POEntry,
        *,
        locale: str,
        ordinal: int,
        source_file_id: str,
        source_file_sha256: str,
        ) -> dict[str, object]:
    """把 PO entry 变为带物理来源身份的规范记录。"""
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
    """严格解析固定 UTF-8/LF PO，并拒绝重复 source identity。"""
    if (not isinstance(payload, bytes)
            or payload.startswith(b"\xef\xbb\xbf")
            or b"\r" in payload
            or not payload.endswith(b"\n")
            or _PO_LICENSE_NOTICE not in payload[:4096]):
        raise BroadQaExternalDataError("VLC PO 编码、换行或许可声明非法")
    try:
        parsed = polib.pofile(payload.decode("utf-8"), wrapwidth=0)
    except (UnicodeDecodeError, OSError, ValueError) as error:
        raise BroadQaExternalDataError("VLC PO parser 失败") from error
    if (parsed.metadata.get("Language") != locale
            or parsed.metadata.get("Content-Type")
            != "text/plain; charset=UTF-8"):
        raise BroadQaExternalDataError("VLC PO locale/charset 漂移")
    records = {}
    for ordinal, entry in enumerate(parsed):
        key = _entry_key(entry)
        if key in records:
            raise BroadQaExternalDataError("VLC PO source identity 重复")
        records[key] = _entry_record(
            entry,
            locale=locale,
            ordinal=ordinal,
            source_file_id=source_file_id,
            source_file_sha256=sha256_hex(payload),
        )
    return records, {
        "empty_translation_count": sum(not entry.msgstr for entry in parsed),
        "entry_count": len(parsed),
        "fuzzy_count": sum("fuzzy" in entry.flags for entry in parsed),
        "language": parsed.metadata.get("Language"),
        "metadata_sha256": sha256_hex(canonical_json_bytes(parsed.metadata)),
        "obsolete_count": sum(int(entry.obsolete) for entry in parsed),
        "plural_count": sum(bool(entry.msgid_plural) for entry in parsed),
        "same_package_license_notice_count": payload[:4096].count(
            _PO_LICENSE_NOTICE),
        "translated_count": len(parsed.translated_entries()),
    }


def _pair_exclusion_reasons(
        zh_hans: dict[str, object],
        zh_hant: dict[str, object],
        ) -> tuple[str, ...]:
    """返回 common source identity 未进入 plain 分母的全部原因。"""
    reasons = []
    if zh_hans["msgid_plural"] or zh_hant["msgid_plural"]:
        reasons.append("plural")
    if zh_hans["obsolete"] or zh_hant["obsolete"]:
        reasons.append("obsolete")
    if "fuzzy" in zh_hans["flags"] or "fuzzy" in zh_hant["flags"]:
        reasons.append("fuzzy")
    if not zh_hans["msgstr"] or not zh_hant["msgstr"]:
        reasons.append("empty")
    return tuple(reasons)


def _translation_pairs(
        locale_records: dict[
            str, dict[tuple[str, str, str], dict[str, object]]],
        ) -> tuple[
            tuple[dict[str, object], ...],
            dict[str, int],
        ]:
    """按完整 source identity 派生 singular active nonempty pair。"""
    common = sorted(set(locale_records["zh_Hans"]).intersection(
        locale_records["zh_Hant"]))
    exclusions = Counter()
    values = []
    for key in common:
        zh_hans = locale_records["zh_Hans"][key]
        zh_hant = locale_records["zh_Hant"][key]
        reasons = _pair_exclusion_reasons(zh_hans, zh_hant)
        if reasons:
            exclusions["any"] += 1
            for reason in reasons:
                exclusions[reason] += 1
            continue
        source_identity = {
            "msgctxt": key[0],
            "msgid": key[1],
            "msgid_plural": key[2],
        }
        features = localization_pair_features(
            str(zh_hant["msgstr"]),
            str(zh_hans["msgstr"]),
            scalar_limit=VLC_PAIR_TEXT_SCALAR_MAX,
        )
        values.append({
            **features,
            "format_version": 1,
            "pair_id": localization_record_id({
                "record_kind": VLC_PO_PAIR_RECORD_KIND,
                "source_identity": source_identity,
            }),
            "record_kind": VLC_PO_PAIR_RECORD_KIND,
            "source_identity": source_identity,
            "source_identity_sha256": sha256_hex(
                canonical_json_bytes(source_identity)),
            "zh_hans": zh_hans,
            "zh_hant": zh_hant,
        })
    if (not values or len({item["pair_id"] for item in values})
            != len(values)):
        raise BroadQaExternalDataError("VLC translation pair identity 非法")
    return tuple(values), {
        name: exclusions[name]
        for name in ("any", "empty", "fuzzy", "obsolete", "plural")
    }


def _summary(
        file_records: tuple[dict[str, object], ...],
        pairs: tuple[dict[str, object], ...],
        locale_records: dict[
            str, dict[tuple[str, str, str], dict[str, object]]],
        locale_summaries: dict[str, dict[str, object]],
        exclusions: dict[str, int],
        ) -> dict[str, object]:
    """汇总固定文件、common identity、pair、结构和冲突库存。"""
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
        counts["within_scalar_limit_count"] += int(
            item["within_scalar_limit"])
        input_text = str(item["zh_hant"]["msgstr"])
        outputs_by_input.setdefault(input_text, set()).add(
            str(item["zh_hans"]["msgstr"]))
    common_count = len(set(locale_records["zh_Hans"]).intersection(
        locale_records["zh_Hant"]))
    if common_count != len(pairs) + exclusions["any"]:
        raise BroadQaExternalDataError("VLC common identity census 不闭合")
    return {
        "archive_file_count": len(file_records),
        "common_identity_count": common_count,
        "equal_length_pair_count": counts["equal_length_pair_count"],
        "excluded_common_pair_counts": exclusions,
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
            "structure_equal_required_for_denominator": 0,
        },
        "structure_equal_count": counts["structure_equal_count"],
        "training_eligible_pair_count": counts[
            "training_eligible_pair_count"],
        "variable_length_pair_count": counts["variable_length_pair_count"],
        "within_scalar_limit_count": counts["within_scalar_limit_count"],
    }


def parse_normalization_recovery_v7_vlc_archive(
        archive_payload: bytes,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """从固定 archive 派生文件、held-out pair 与 aggregate census。"""
    files = read_exact_localization_zip(
        archive_payload,
        expected_files=VLC_ARCHIVE_FILES,
        allowed_directories=("po",),
        label="VLC recovery-v7 held-out source",
        member_count_max=6,
        uncompressed_bytes_max=4 * 1024 * 1024,
    )
    validate_vlc_license(
        files["COPYING"], expected_expression=VLC_LICENSE_EXPRESSION)
    file_records = _source_file_records(files)
    file_by_path = {
        str(item["relative_path"]): item for item in file_records}
    locale_records = {}
    locale_summaries = {}
    for locale_key, path in VLC_SOURCE_PATHS.items():
        locale = "zh_CN" if locale_key == "zh_Hans" else "zh_TW"
        records, summary = _parse_po(
            files[path],
            locale=locale,
            source_file_id=str(file_by_path[path]["file_id"]),
        )
        locale_records[locale_key] = records
        locale_summaries[locale_key] = summary
    pairs, exclusions = _translation_pairs(locale_records)
    return (
        file_records,
        pairs,
        _summary(
            file_records,
            pairs,
            locale_records,
            locale_summaries,
            exclusions,
        ),
    )


__all__ = [
    "VLC_ARCHIVE_FILES",
    "VLC_LICENSE_EXPRESSION",
    "VLC_PAIR_TEXT_SCALAR_MAX",
    "VLC_PO_PAIR_RECORD_KIND",
    "VLC_SOURCE_FILE_RECORD_KIND",
    "VLC_SOURCE_PATHS",
    "parse_normalization_recovery_v7_vlc_archive",
    "validate_vlc_license",
]
