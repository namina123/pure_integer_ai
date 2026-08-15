"""解析固定 Audacity zh-CN/zh-TW PO atom-validation 来源。

selection 已在读取翻译 blob 前冻结。本模块只做确定性来源解析、全量分母
对齐与 aggregate census，不选择规则、不评分，也不发布个体翻译 label。
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
    sha256_hex,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


AUDACITY_VALIDATION_SOURCE_FILE_KIND = (
    "AUDACITY_ATOM_VALIDATION_SOURCE_FILE_V1")
AUDACITY_VALIDATION_PAIR_KIND = "AUDACITY_ATOM_VALIDATION_PO_PAIR_V1"
AUDACITY_PAIR_TEXT_SCALAR_MAX = 320
AUDACITY_LICENSE_PATH = "LICENSE.txt"
AUDACITY_SOURCE_PATHS = {
    "zh_Hans": "au3/locale/zh_CN.po",
    "zh_Hant": "au3/locale/zh_TW.po",
}
AUDACITY_SOURCE_FILES = (
    AUDACITY_LICENSE_PATH,
    AUDACITY_SOURCE_PATHS["zh_Hans"],
    AUDACITY_SOURCE_PATHS["zh_Hant"],
)
AUDACITY_TRANSLATION_LICENSE_EXPRESSION = "GPL-2.0-or-later"

_LICENSE_DECLARATION = (
    b"Audacity is released under the GNU General Public License version 3 "
    b"(GPLv3).")
_DEFAULT_FILE_LICENSE = (
    b"available under GPL version 2 (GPLv2) or (at your option) any later "
    b"version")
_GPL_V3_TITLE = b"GNU GENERAL PUBLIC LICENSE\n                       Version 3"


def validate_audacity_license(payload: bytes) -> None:
    """核对固定 Audacity project/default-file 许可声明。"""
    if (not isinstance(payload, bytes)
            or _LICENSE_DECLARATION not in payload[:2048]
            or _DEFAULT_FILE_LICENSE not in payload[:2048]
            or _GPL_V3_TITLE not in payload):
        raise BroadQaExternalDataError("Audacity atom-validation license 漂移")


def _source_file_records(
        files: dict[str, bytes],
        ) -> tuple[dict[str, object], ...]:
    """为 license 与两份 PO 形成稳定物理来源记录。"""
    locale_by_path = {
        path: locale for locale, path in AUDACITY_SOURCE_PATHS.items()}
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
            "record_kind": AUDACITY_VALIDATION_SOURCE_FILE_KIND,
            "role": (
                "TRANSLATION_PO"
                if relative_path in locale_by_path else "LICENSE_TEXT"),
        })
    return tuple(values)


def _entry_key(entry: polib.POEntry) -> tuple[str, str, str]:
    """返回跨 locale 对齐的完整官方 English source identity。"""
    return entry.msgctxt or "", entry.msgid, entry.msgid_plural or ""


def _entry_record(
        entry: polib.POEntry,
        *,
        locale: str,
        ordinal: int,
        source_file_id: str,
        source_file_sha256: str,
        ) -> dict[str, object]:
    """把单条 PO entry 变为仅驻内存的来源化记录。"""
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
    """严格解析一份固定 UTF-8/LF PO，并拒绝重复 source identity。"""
    if (not isinstance(payload, bytes)
            or payload.startswith(b"\xef\xbb\xbf")
            or b"\r" in payload
            or not payload.endswith(b"\n")):
        raise BroadQaExternalDataError(
            "Audacity atom-validation PO 编码或换行非法")
    try:
        parsed = polib.pofile(payload.decode("utf-8"), wrapwidth=0)
    except (UnicodeDecodeError, OSError, ValueError) as error:
        raise BroadQaExternalDataError(
            "Audacity atom-validation PO parser 失败") from error
    language = parsed.metadata.get("Language")
    if language not in {locale, locale.replace("_", "-")}:
        raise BroadQaExternalDataError(
            "Audacity atom-validation PO locale 漂移")
    records = {}
    for ordinal, entry in enumerate(parsed):
        key = _entry_key(entry)
        if key in records:
            raise BroadQaExternalDataError(
                "Audacity atom-validation source identity 重复")
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
        "language": language,
        "metadata_sha256": sha256_hex(canonical_json_bytes(parsed.metadata)),
        "obsolete_count": sum(int(entry.obsolete) for entry in parsed),
        "plural_count": sum(bool(entry.msgid_plural) for entry in parsed),
        "translated_count": len(parsed.translated_entries()),
    }


def parse_audacity_atom_validation_locale(
        payload: bytes,
        *,
        locale: str,
        source_file_id: str,
        ) -> tuple[
            dict[tuple[str, str, str], dict[str, object]],
            dict[str, object],
        ]:
    """单独解析一个冻结 locale，供 label 物理隔离 reader 使用。"""
    if locale not in {"zh_CN", "zh_TW"}:
        raise BroadQaExternalDataError(
            "Audacity atom-validation locale 非法")
    if not isinstance(source_file_id, str) or len(source_file_id) != 64:
        raise BroadQaExternalDataError(
            "Audacity atom-validation source file identity 非法")
    return _parse_po(
        payload, locale=locale, source_file_id=source_file_id)


def _pair_exclusion_reasons(
        zh_hans: dict[str, object],
        zh_hant: dict[str, object],
        ) -> tuple[str, ...]:
    """返回 common source identity 未进入冻结分母的全部原因。"""
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
        ) -> tuple[tuple[dict[str, object], ...], dict[str, int]]:
    """按冻结 selection 派生全部 singular active nonempty pair。"""
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
            scalar_limit=AUDACITY_PAIR_TEXT_SCALAR_MAX,
        )
        values.append({
            **features,
            "format_version": 1,
            "pair_id": localization_record_id({
                "record_kind": AUDACITY_VALIDATION_PAIR_KIND,
                "source_identity": source_identity,
            }),
            "record_kind": AUDACITY_VALIDATION_PAIR_KIND,
            "source_identity": source_identity,
            "source_identity_sha256": sha256_hex(
                canonical_json_bytes(source_identity)),
            "zh_hans": zh_hans,
            "zh_hant": zh_hant,
        })
    if (not values or len({item["pair_id"] for item in values})
            != len(values)):
        raise BroadQaExternalDataError(
            "Audacity atom-validation pair identity 非法")
    return tuple(values), {
        name: exclusions[name]
        for name in ("any", "empty", "fuzzy", "obsolete", "plural")}


def _summary(
        *,
        pairs: tuple[dict[str, object], ...],
        locale_records: dict[
            str, dict[tuple[str, str, str], dict[str, object]]],
        locale_summaries: dict[str, dict[str, object]],
        exclusions: dict[str, int],
        ) -> dict[str, object]:
    """汇总固定分母、结构与冲突库存，不发布翻译 surface。"""
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
        raise BroadQaExternalDataError(
            "Audacity atom-validation common identity census 不闭合")
    return {
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


def parse_audacity_atom_validation_files(
        files: dict[str, bytes],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """从固定三文件派生 source records、内存 pair 与 aggregate census。"""
    if (not isinstance(files, dict)
            or set(files) != set(AUDACITY_SOURCE_FILES)
            or any(not isinstance(payload, bytes)
                   for payload in files.values())):
        raise BroadQaExternalDataError(
            "Audacity atom-validation source inventory 漂移")
    validate_audacity_license(files[AUDACITY_LICENSE_PATH])
    file_records = _source_file_records(files)
    file_by_path = {
        str(item["relative_path"]): item for item in file_records}
    locale_records = {}
    locale_summaries = {}
    for locale_key, path in AUDACITY_SOURCE_PATHS.items():
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
            pairs=pairs,
            locale_records=locale_records,
            locale_summaries=locale_summaries,
            exclusions=exclusions,
        ),
    )


__all__ = [
    "AUDACITY_LICENSE_PATH",
    "AUDACITY_PAIR_TEXT_SCALAR_MAX",
    "AUDACITY_SOURCE_FILES",
    "AUDACITY_SOURCE_PATHS",
    "AUDACITY_TRANSLATION_LICENSE_EXPRESSION",
    "AUDACITY_VALIDATION_PAIR_KIND",
    "AUDACITY_VALIDATION_SOURCE_FILE_KIND",
    "parse_audacity_atom_validation_locale",
    "parse_audacity_atom_validation_files",
    "validate_audacity_license",
]
