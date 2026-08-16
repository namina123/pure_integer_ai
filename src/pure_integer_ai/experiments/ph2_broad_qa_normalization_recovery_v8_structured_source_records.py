"""派生 recovery-v8 参数化 Qt TS TRAIN source records。

本模块不绑定具体项目路径或来源数量。调用方先冻结 repository/tree/blob 与
locale pair spec，再把逐字节核验后的 payload 交给共享 parser。active common
pair 全量保留；结构、汉字与长度仅形成显式资格事实，不缩 source inventory。
"""
from __future__ import annotations

from collections import Counter
import xml.etree.ElementTree as ElementTree

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


V8_STRUCTURED_SOURCE_FILE_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V8_STRUCTURED_SOURCE_FILE_V1")
V8_QT_TS_PAIR_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V8_QT_TS_PAIR_V1")
V8_QT_TS_TEXT_SCALAR_MAX = 4_096

_INACTIVE_TRANSLATION_TYPES = {"obsolete", "unfinished", "vanished"}
_LOCALE_ROLES = ("zh_Hans", "zh_Hant")


def _element_text(
        element: ElementTree.Element | None,
        *,
        label: str,
        allow_empty: bool,
        ) -> str:
    """读取不含子节点的 TS 文本并保持原字符内容。"""
    if element is None or len(element):
        raise BroadQaExternalDataError(f"v8 Qt TS {label} 结构非法")
    value = element.text or ""
    if not allow_empty and not value:
        raise BroadQaExternalDataError(f"v8 Qt TS {label} 不能为空")
    return value


def _source_file_record(
        *,
        domain: str,
        locale_role: str,
        relative_path: str,
        payload: bytes,
        ) -> dict[str, object]:
    """形成一个逐字节来源文件 commitment。"""
    identity = {
        "git_blob_sha1": git_blob_sha1(payload),
        "relative_path": relative_path,
        "sha256": sha256_hex(payload),
    }
    return {
        **identity,
        "bytes": len(payload),
        "domain": domain,
        "file_id": localization_record_id(identity),
        "format_version": 1,
        "locale_role": locale_role,
        "record_kind": V8_STRUCTURED_SOURCE_FILE_RECORD_KIND,
        "role": "TRANSLATION_WITH_OFFICIAL_SOURCE_QT_TS",
    }


def _source_identity(
        *,
        domain: str,
        context_name: str,
        message: ElementTree.Element,
        ) -> dict[str, object]:
    """形成跨 locale 唯一对齐的 Qt message identity。"""
    return {
        "comment": _element_text(
            message.find("comment"), label="comment", allow_empty=True)
        if message.find("comment") is not None else "",
        "context": context_name,
        "domain": domain,
        "message_id": message.get("id") or "",
        "source": _element_text(
            message.find("source"), label="source", allow_empty=False),
    }


def _parse_qt_ts(
        payload: bytes,
        *,
        domain: str,
        expected_language: str,
        expected_source_language: str,
        locale_role: str,
        source_file_id: str,
        ) -> tuple[dict[tuple[str, ...], dict[str, object]], dict[str, int]]:
    """解析单份 TS，并保留 active、non-numerus、nonempty message。"""
    if (not isinstance(payload, bytes) or not payload
            or not isinstance(domain, str) or not domain
            or not isinstance(expected_language, str) or not expected_language
            or not isinstance(expected_source_language, str)
            or locale_role not in _LOCALE_ROLES
            or not isinstance(source_file_id, str)
            or len(source_file_id) != 64
            or payload.count(b"<!DOCTYPE TS>") > 1
            or b"<!ENTITY" in payload.upper()):
        raise BroadQaExternalDataError("v8 Qt TS input contract 漂移")
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise BroadQaExternalDataError("v8 Qt TS XML parser 失败") from error
    expected_attributes = {"version", "language"}
    if expected_source_language:
        expected_attributes.add("sourcelanguage")
    if (root.tag != "TS" or root.get("version") != "2.1"
            or root.get("language") != expected_language
            or root.get("sourcelanguage", "") != expected_source_language
            or set(root.attrib) != expected_attributes
            or any(child.tag not in {"context", "dependencies"}
                   for child in root)):
        raise BroadQaExternalDataError("v8 Qt TS root schema 漂移")
    records = {}
    counts = Counter()
    for context in root.findall("context"):
        if (any(child.tag not in {"name", "message"} for child in context)
                or len(context.findall("name")) != 1):
            raise BroadQaExternalDataError("v8 Qt TS context schema 漂移")
        context_name = _element_text(
            context.find("name"), label="context name", allow_empty=False)
        for message in context.findall("message"):
            counts["message_count"] += 1
            if (set(message.attrib).difference({"id", "numerus"})
                    or any(child.tag not in {
                        "comment", "extracomment", "location", "oldsource",
                        "source", "translation", "translatorcomment"}
                        for child in message)):
                raise BroadQaExternalDataError("v8 Qt TS message schema 漂移")
            old_sources = message.findall("oldsource")
            if len(old_sources) > 1:
                raise BroadQaExternalDataError(
                    "v8 Qt TS oldsource 数量漂移")
            if old_sources:
                _element_text(
                    old_sources[0], label="oldsource", allow_empty=False)
            translation = message.find("translation")
            if translation is None:
                raise BroadQaExternalDataError("v8 Qt TS translation 缺失")
            translation_type = translation.get("type") or ""
            if (translation_type and (
                    translation_type not in _INACTIVE_TRANSLATION_TYPES
                    or set(translation.attrib) != {"type"})):
                raise BroadQaExternalDataError(
                    "v8 Qt TS translation type 漂移")
            numerus = message.get("numerus")
            if numerus == "yes":
                counts["numerus_count"] += 1
                continue
            if numerus not in (None, ""):
                raise BroadQaExternalDataError("v8 Qt TS numerus attr 漂移")
            if translation_type in _INACTIVE_TRANSLATION_TYPES:
                counts[f"{translation_type}_count"] += 1
                continue
            translation_text = _element_text(
                translation, label="translation", allow_empty=True)
            if not translation_text:
                counts["empty_active_translation_count"] += 1
                continue
            identity = _source_identity(
                domain=domain, context_name=context_name, message=message)
            key = tuple(str(identity[name]) for name in (
                "domain", "context", "source", "comment", "message_id"))
            if key in records:
                raise BroadQaExternalDataError(
                    "v8 Qt TS active source identity 重复")
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
                "locale_role": locale_role,
                "root_language": expected_language,
                "semantic_sha256": sha256_hex(canonical_json_bytes(semantic)),
                "source_file_id": source_file_id,
                "source_file_sha256": sha256_hex(payload),
            }
            counts["active_plain_count"] += 1
    return records, {
        key: counts[key] for key in (
            "active_plain_count", "empty_active_translation_count",
            "message_count", "numerus_count", "obsolete_count",
            "unfinished_count", "vanished_count")}


def _pair_records(
        *,
        source_family: str,
        source_policy_scope: str,
        license_expression: str,
        by_locale: dict[
            str, dict[tuple[str, ...], dict[str, object]]],
        ) -> tuple[dict[str, object], ...]:
    """按完整 official-source identity 对齐 active 简繁 message。"""
    if set(by_locale) != set(_LOCALE_ROLES):
        raise BroadQaExternalDataError("v8 Qt TS locale roster 漂移")
    common = sorted(set(by_locale["zh_Hans"]).intersection(
        by_locale["zh_Hant"]))
    values = []
    for key in common:
        hans = by_locale["zh_Hans"][key]
        hant = by_locale["zh_Hant"][key]
        source_identity = hans["source_identity"]
        if source_identity != hant["source_identity"]:
            raise BroadQaExternalDataError("v8 Qt TS source identity 漂移")
        features = localization_pair_features(
            str(hant["translation"]),
            str(hans["translation"]),
            scalar_limit=V8_QT_TS_TEXT_SCALAR_MAX,
        )
        identity = {
            "record_kind": V8_QT_TS_PAIR_RECORD_KIND,
            "source_family": source_family,
            "source_identity": source_identity,
            "source_policy_scope": source_policy_scope,
        }
        values.append({
            **features,
            "format_version": 1,
            "license_expression": license_expression,
            "official_source_text": source_identity["source"],
            "pair_id": localization_record_id(identity),
            "record_kind": V8_QT_TS_PAIR_RECORD_KIND,
            "source_family": source_family,
            "source_identity": source_identity,
            "source_identity_sha256": sha256_hex(
                canonical_json_bytes(source_identity)),
            "source_policy_scope": source_policy_scope,
            "v8_training_eligible": int(
                features["training_eligible"] == 1
                and features["contains_han_both"] == 1),
            "zh_hans": hans,
            "zh_hant": hant,
        })
    if len({item["pair_id"] for item in values}) != len(values):
        raise BroadQaExternalDataError("v8 Qt TS pair identity 重复")
    return tuple(values)


def _summary(
        *,
        file_records: tuple[dict[str, object], ...],
        pairs: tuple[dict[str, object], ...],
        locale_summaries: dict[str, dict[str, dict[str, int]]],
        ) -> dict[str, object]:
    """汇总 active source inventory、pair资格与冲突。"""
    counts = Counter()
    outputs_by_input: dict[str, set[str]] = {}
    domain_counts = Counter()
    for item in pairs:
        counts["identity_pair_count"] += int(item["identity_preservation"])
        counts["nonidentity_pair_count"] += int(
            item["identity_preservation"] == 0)
        counts["equal_length_pair_count"] += int(item["equal_length"])
        counts["variable_length_pair_count"] += int(
            item["equal_length"] == 0)
        counts["single_han_difference_count"] += int(
            item["single_han_difference"])
        counts["structure_equal_count"] += int(item["structure_equal"])
        counts["contains_han_both_count"] += int(item["contains_han_both"])
        counts["v8_training_eligible_pair_count"] += int(
            item["v8_training_eligible"])
        domain_counts[str(item["source_identity"]["domain"])] += 1
        input_text = str(item["zh_hant"]["translation"])
        outputs_by_input.setdefault(input_text, set()).add(
            str(item["zh_hans"]["translation"]))
    pair_count = len(pairs)
    return {
        "content_outcome": (
            "PASS_NONZERO_ACTIVE_COMMON_PAIR" if pair_count
            else "REJECTED_ZERO_ACTIVE_COMMON_PAIR"),
        "contains_han_both_count": counts["contains_han_both_count"],
        "domain_pair_counts": dict(sorted(domain_counts.items())),
        "equal_length_pair_count": counts["equal_length_pair_count"],
        "identity_pair_count": counts["identity_pair_count"],
        "input_conflict_count": sum(
            len(outputs) > 1 for outputs in outputs_by_input.values()),
        "locale_summaries": locale_summaries,
        "nonidentity_pair_count": counts["nonidentity_pair_count"],
        "plain_pair_count": pair_count,
        "single_han_difference_count": counts[
            "single_han_difference_count"],
        "source_file_count": len(file_records),
        "source_format_policy": {
            "active_translation_required": 1,
            "empty_translation_allowed": 0,
            "inactive_translation_allowed": 0,
            "numerus_allowed": 0,
            "official_source_identity": (
                "DOMAIN_CONTEXT_SOURCE_COMMENT_MESSAGE_ID"),
            "parser": "xml.etree.ElementTree",
            "structure_unequal_pair_preserved": 1,
        },
        "structure_equal_count": counts["structure_equal_count"],
        "v8_training_eligible_pair_count": counts[
            "v8_training_eligible_pair_count"],
        "variable_length_pair_count": counts["variable_length_pair_count"],
    }


def derive_normalization_recovery_v8_qt_ts_source_records(
        *,
        source_family: str,
        source_policy_scope: str,
        license_expression: str,
        pair_specs: tuple[dict[str, object], ...],
        files: dict[str, bytes],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """从已冻结的成对 TS blob 派生共享 v8 source records。"""
    if (not isinstance(source_family, str) or not source_family
            or not isinstance(source_policy_scope, str)
            or not source_policy_scope
            or not isinstance(license_expression, str)
            or not license_expression
            or not isinstance(pair_specs, tuple) or not pair_specs
            or not isinstance(files, dict)):
        raise BroadQaExternalDataError("v8 Qt TS source contract 非法")
    expected_paths = []
    domains = set()
    file_records = []
    by_locale = {role: {} for role in _LOCALE_ROLES}
    locale_summaries: dict[str, dict[str, dict[str, int]]] = {
        role: {} for role in _LOCALE_ROLES}
    for spec in pair_specs:
        if not isinstance(spec, dict):
            raise BroadQaExternalDataError("v8 Qt TS pair spec 非对象")
        domain = spec.get("domain")
        if not isinstance(domain, str) or not domain or domain in domains:
            raise BroadQaExternalDataError("v8 Qt TS domain identity 漂移")
        domains.add(domain)
        for role in _LOCALE_ROLES:
            value = spec.get(role)
            allowed_keys = {
                "expected_language", "expected_source_language",
                "relative_path",
            }
            if (not isinstance(value, dict)
                    or set(value).difference(allowed_keys)
                    or not {"expected_language", "relative_path"}.issubset(value)
                    or not isinstance(value["expected_language"], str)
                    or not value["expected_language"]
                    or not isinstance(
                        value.get("expected_source_language", ""), str)
                    or not isinstance(value["relative_path"], str)
                    or not value["relative_path"]):
                raise BroadQaExternalDataError(
                    "v8 Qt TS locale pair spec 漂移")
            path = value["relative_path"]
            if path in expected_paths or path not in files:
                raise BroadQaExternalDataError(
                    "v8 Qt TS source file roster 漂移")
            expected_paths.append(path)
            payload = files[path]
            record = _source_file_record(
                domain=domain,
                locale_role=role,
                relative_path=path,
                payload=payload,
            )
            records, summary = _parse_qt_ts(
                payload,
                domain=domain,
                expected_language=value["expected_language"],
                expected_source_language=value.get(
                    "expected_source_language", ""),
                locale_role=role,
                source_file_id=str(record["file_id"]),
            )
            overlap = set(by_locale[role]).intersection(records)
            if overlap:
                raise BroadQaExternalDataError(
                    "v8 Qt TS cross-domain source identity 重复")
            by_locale[role].update(records)
            locale_summaries[role][domain] = summary
            file_records.append(record)
    if set(files) != set(expected_paths):
        raise BroadQaExternalDataError("v8 Qt TS unselected file 混入")
    file_records.sort(key=lambda item: str(item["relative_path"]))
    pairs = _pair_records(
        source_family=source_family,
        source_policy_scope=source_policy_scope,
        license_expression=license_expression,
        by_locale=by_locale,
    )
    return tuple(file_records), pairs, _summary(
        file_records=tuple(file_records),
        pairs=pairs,
        locale_summaries=locale_summaries,
    )


__all__ = [
    "V8_QT_TS_PAIR_RECORD_KIND",
    "V8_QT_TS_TEXT_SCALAR_MAX",
    "V8_STRUCTURED_SOURCE_FILE_RECORD_KIND",
    "derive_normalization_recovery_v8_qt_ts_source_records",
]
