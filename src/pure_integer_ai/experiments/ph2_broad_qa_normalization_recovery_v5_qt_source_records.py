"""从固定 Qt TS archive 派生 held-out 来源与对齐记录。

adapter 使用标准库 XML parser，按 module/context/source/comment/message id
对齐 zh-TW/zh-CN。它保存原文和结构，但不决定 formal 阈值或训练 rule。
"""
from __future__ import annotations

from collections import Counter
import json
import xml.etree.ElementTree as ElementTree

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


QT_SOURCE_FILE_RECORD_KIND = "QT_TRANSLATIONS_SOURCE_FILE_V1"
QT_TS_PAIR_RECORD_KIND = "QT_TRANSLATIONS_TS_PAIR_V1"
QT_PAIR_TEXT_SCALAR_MAX = 320
QT_MODULES = (
    "assistant",
    "designer",
    "linguist",
    "qt",
    "qt_help",
    "qtbase",
    "qtdeclarative",
    "qtmultimedia",
)
QT_LOCALES = {"zh_Hans": "zh_CN", "zh_Hant": "zh_TW"}
QT_ARCHIVE_FILES = (
    "licenseRule.json",
    *tuple(
        f"translations/{module}_{locale}.ts"
        for module in QT_MODULES
        for locale in ("zh_CN", "zh_TW")),
)
_INACTIVE_TRANSLATION_TYPES = {"unfinished", "vanished", "obsolete"}


def _reject_duplicate_json_keys(
        pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
    """构造 JSON object，并拒绝任意层级的重复 key。"""
    value = {}
    for key, item in pairs:
        if key in value:
            raise BroadQaExternalDataError("Qt licenseRule 含重复 key")
        value[key] = item
    return value


def validate_qt_license_rule(
        payload: bytes,
        *,
        expected_expression: str,
        ) -> None:
    """结构化核验固定 licenseRule 的 ordinary module 默认表达式。"""
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("Qt licenseRule JSON 非法") from error
    if (not isinstance(value, list) or not value
            or not isinstance(value[-1], dict)
            or "file_pattern_ending" in value[-1]):
        raise BroadQaExternalDataError("Qt licenseRule default rule 漂移")
    location = value[-1].get("location")
    default = location.get("") if isinstance(location, dict) else None
    spdx = default.get("spdx") if isinstance(default, dict) else None
    if spdx != [expected_expression]:
        raise BroadQaExternalDataError("Qt licenseRule SPDX 漂移")


def _source_file_records(
        files: dict[str, bytes],
        ) -> tuple[dict[str, object], ...]:
    """为许可规则和每份 TS Git blob 形成来源记录。"""
    values = []
    for relative_path, payload in sorted(files.items()):
        module = ""
        locale = ""
        role = "LICENSE_RULE"
        if relative_path.startswith("translations/"):
            filename = relative_path.removeprefix("translations/")
            for candidate in sorted(QT_MODULES, key=len, reverse=True):
                prefix = f"{candidate}_"
                if filename.startswith(prefix):
                    module = candidate
                    locale = filename[len(prefix):-3]
                    break
            if not module or locale not in QT_LOCALES.values():
                raise BroadQaExternalDataError(
                    "Qt TS source filename contract 漂移")
            role = "TRANSLATION_TS"
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
            "locale": locale,
            "module": module,
            "record_kind": QT_SOURCE_FILE_RECORD_KIND,
            "role": role,
        })
    return tuple(values)


def _element_text(
        element: ElementTree.Element | None,
        *,
        label: str,
        allow_empty: bool,
        ) -> str:
    """读取无子节点的 TS 文本元素并保持其完整字符内容。"""
    if element is None or len(element):
        raise BroadQaExternalDataError(f"Qt TS {label} 结构非法")
    value = element.text or ""
    if not allow_empty and not value:
        raise BroadQaExternalDataError(f"Qt TS {label} 不能为空")
    return value


def _message_source_identity(
        *,
        module: str,
        context_name: str,
        message: ElementTree.Element,
        ) -> dict[str, object]:
    """形成跨 locale 稳定的 TS message source identity。"""
    return {
        "comment": _element_text(
            message.find("comment"), label="comment", allow_empty=True)
        if message.find("comment") is not None else "",
        "context": context_name,
        "message_id": message.get("id") or "",
        "module": module,
        "source": _element_text(
            message.find("source"), label="source", allow_empty=False),
    }


def _parse_ts(
        payload: bytes,
        *,
        locale: str,
        module: str,
        source_file_id: str,
        ) -> tuple[dict[tuple[str, ...], dict[str, object]], dict[str, object]]:
    """解析一份 TS，并返回 active non-numerus message map。"""
    if (not isinstance(payload, bytes)
            or payload.count(b"<!DOCTYPE TS>") != 1
            or b"<!ENTITY" in payload.upper()):
        raise BroadQaExternalDataError("Qt TS declaration 非法")
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise BroadQaExternalDataError("Qt TS XML parser 失败") from error
    if (root.tag != "TS" or root.get("version") != "2.1"
            or root.get("language") != locale
            or set(root.attrib) != {"version", "language"}
            or any(child.tag not in {"context", "dependencies"}
                   for child in root)):
        raise BroadQaExternalDataError("Qt TS root schema 漂移")
    records = {}
    counts = Counter()
    for context in root.findall("context"):
        if (any(child.tag not in {"name", "message"} for child in context)
                or len(context.findall("name")) != 1):
            raise BroadQaExternalDataError("Qt TS context schema 漂移")
        context_name = _element_text(
            context.find("name"), label="context name", allow_empty=False)
        for message in context.findall("message"):
            counts["message_count"] += 1
            if set(message.attrib).difference({"id", "numerus"}):
                raise BroadQaExternalDataError("Qt TS message attrs 漂移")
            translation = message.find("translation")
            if (translation is None
                    or any(child.tag not in {
                        "source", "translation", "comment", "extracomment",
                        "translatorcomment", "location"}
                        for child in message)):
                raise BroadQaExternalDataError("Qt TS message schema 漂移")
            translation_type = translation.get("type") or ""
            if translation_type and (
                    translation_type not in _INACTIVE_TRANSLATION_TYPES
                    or set(translation.attrib) != {"type"}):
                raise BroadQaExternalDataError(
                    "Qt TS translation type 漂移")
            if message.get("numerus") == "yes":
                counts["numerus_count"] += 1
                continue
            if message.get("numerus") not in (None, ""):
                raise BroadQaExternalDataError("Qt TS numerus attr 漂移")
            if translation_type in _INACTIVE_TRANSLATION_TYPES:
                counts[f"{translation_type}_count"] += 1
                continue
            translation_text = _element_text(
                translation, label="translation", allow_empty=True)
            if not translation_text:
                counts["empty_active_translation_count"] += 1
                continue
            identity = _message_source_identity(
                module=module,
                context_name=context_name,
                message=message,
            )
            key = tuple(str(identity[name]) for name in (
                "module", "context", "source", "comment", "message_id"))
            if key in records:
                raise BroadQaExternalDataError(
                    "Qt TS active source identity 重复")
            semantic = {
                "extracomment": _element_text(
                    message.find("extracomment"),
                    label="extracomment",
                    allow_empty=True,
                ) if message.find("extracomment") is not None else "",
                "source_identity": identity,
                "translation": translation_text,
                "translatorcomment": _element_text(
                    message.find("translatorcomment"),
                    label="translatorcomment",
                    allow_empty=True,
                ) if message.find("translatorcomment") is not None else "",
            }
            records[key] = {
                **semantic,
                "locale": locale,
                "semantic_sha256": sha256_hex(canonical_json_bytes(semantic)),
                "source_file_id": source_file_id,
                "source_file_sha256": sha256_hex(payload),
            }
            counts["active_plain_count"] += 1
    return records, {
        "active_plain_count": counts["active_plain_count"],
        "empty_active_translation_count": counts[
            "empty_active_translation_count"],
        "message_count": counts["message_count"],
        "numerus_count": counts["numerus_count"],
        "obsolete_count": counts["obsolete_count"],
        "unfinished_count": counts["unfinished_count"],
        "vanished_count": counts["vanished_count"],
    }


def _translation_pairs(
        locale_records: dict[str, dict[tuple[str, ...], dict[str, object]]],
        ) -> tuple[dict[str, object], ...]:
    """按完整 source identity 对齐 active 简繁 TS message。"""
    common = sorted(set(locale_records["zh_Hans"]).intersection(
        locale_records["zh_Hant"]))
    values = []
    for key in common:
        zh_hans = locale_records["zh_Hans"][key]
        zh_hant = locale_records["zh_Hant"][key]
        source_identity = zh_hans["source_identity"]
        if source_identity != zh_hant["source_identity"]:
            raise BroadQaExternalDataError("Qt TS source identity 漂移")
        features = localization_pair_features(
            str(zh_hant["translation"]),
            str(zh_hans["translation"]),
            scalar_limit=QT_PAIR_TEXT_SCALAR_MAX,
        )
        values.append({
            **features,
            "format_version": 1,
            "pair_id": localization_record_id({
                "record_kind": QT_TS_PAIR_RECORD_KIND,
                "source_identity": source_identity,
            }),
            "record_kind": QT_TS_PAIR_RECORD_KIND,
            "source_identity": source_identity,
            "source_identity_sha256": sha256_hex(
                canonical_json_bytes(source_identity)),
            "zh_hans": zh_hans,
            "zh_hant": zh_hant,
        })
    if (not values or len({item["pair_id"] for item in values})
            != len(values)):
        raise BroadQaExternalDataError("Qt TS pair identity 非法")
    return tuple(values)


def _summary(
        file_records: tuple[dict[str, object], ...],
        pairs: tuple[dict[str, object], ...],
        locale_summaries: dict[str, dict[str, dict[str, object]]],
        ) -> dict[str, object]:
    """汇总固定文件、active inventory、结构资格与输入冲突。"""
    counts = Counter()
    outputs_by_input: dict[str, set[str]] = {}
    module_pair_counts = Counter()
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
        module_pair_counts[str(item["source_identity"]["module"])] += 1
        input_text = str(item["zh_hant"]["translation"])
        outputs_by_input.setdefault(input_text, set()).add(
            str(item["zh_hans"]["translation"]))
    return {
        "archive_file_count": len(file_records),
        "equal_length_pair_count": counts["equal_length_pair_count"],
        "identity_pair_count": counts["identity_pair_count"],
        "input_conflict_count": sum(
            len(outputs) > 1 for outputs in outputs_by_input.values()),
        "locale_module_summaries": locale_summaries,
        "module_pair_counts": dict(sorted(module_pair_counts.items())),
        "nonidentity_pair_count": counts["nonidentity_pair_count"],
        "plain_pair_count": len(pairs),
        "single_han_difference_count": counts[
            "single_han_difference_count"],
        "source_format_policy": {
            "empty_or_inactive_translation_allowed": 0,
            "numerus_allowed_in_v1": 0,
            "parser": "xml.etree.ElementTree",
            "source_identity": (
                "MODULE_CONTEXT_SOURCE_COMMENT_MESSAGE_ID"),
            "structure_and_original_text_preserved": 1,
        },
        "structure_equal_count": counts["structure_equal_count"],
        "training_eligible_pair_count": counts[
            "training_eligible_pair_count"],
        "variable_length_pair_count": counts["variable_length_pair_count"],
    }


def parse_normalization_recovery_v5_qt_archive(
        archive_payload: bytes,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """从 fixed archive 派生文件、held-out pair 与完整 census。"""
    files = read_exact_localization_zip(
        archive_payload,
        expected_files=QT_ARCHIVE_FILES,
        allowed_directories=("translations",),
        label="Qt recovery-v5 held-out source",
        member_count_max=32,
        uncompressed_bytes_max=4 * 1024 * 1024,
    )
    file_records = _source_file_records(files)
    file_by_path = {
        str(item["relative_path"]): item for item in file_records}
    locale_records = {"zh_Hans": {}, "zh_Hant": {}}
    locale_summaries: dict[str, dict[str, dict[str, object]]] = {
        "zh_Hans": {}, "zh_Hant": {}}
    for locale_key, locale in QT_LOCALES.items():
        for module in QT_MODULES:
            path = f"translations/{module}_{locale}.ts"
            records, summary = _parse_ts(
                files[path],
                locale=locale,
                module=module,
                source_file_id=str(file_by_path[path]["file_id"]),
            )
            overlap = set(locale_records[locale_key]).intersection(records)
            if overlap:
                raise BroadQaExternalDataError(
                    "Qt TS cross-module identity 重复")
            locale_records[locale_key].update(records)
            locale_summaries[locale_key][module] = summary
    pairs = _translation_pairs(locale_records)
    return (
        file_records,
        pairs,
        _summary(file_records, pairs, locale_summaries),
    )


__all__ = [
    "QT_ARCHIVE_FILES",
    "QT_LOCALES",
    "QT_MODULES",
    "QT_PAIR_TEXT_SCALAR_MAX",
    "QT_SOURCE_FILE_RECORD_KIND",
    "QT_TS_PAIR_RECORD_KIND",
    "parse_normalization_recovery_v5_qt_archive",
    "validate_qt_license_rule",
]
