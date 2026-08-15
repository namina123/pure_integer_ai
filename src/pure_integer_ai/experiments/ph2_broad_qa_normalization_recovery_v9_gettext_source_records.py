"""派生 recovery-v9 多 domain gettext held-out 来源记录。

v9 独立修正文档与实现不一致的 obsolete 漏筛问题，不改变任何已冻结的
v8 artifact。正式 plain 分母只接受双方 singular、nonfuzzy、nonobsolete、
nonempty 的完整 ``domain/msgctxt/msgid/msgid_plural`` source identity。
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


V9_GETTEXT_SOURCE_FILE_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V9_GETTEXT_SOURCE_FILE_V1")
V9_GETTEXT_PAIR_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V9_GETTEXT_PAIR_V1")
V9_GETTEXT_TEXT_SCALAR_MAX = 4_096

_LOCALE_ROLES = ("zh_Hans", "zh_Hant")
_LICENSE_NOTICE_CATEGORIES = {
    "FOREIGN_DISK_PACKAGE_NOTICE",
    "GIMP_FAMILY_PACKAGE_NOTICE",
    "NO_FILE_NOTICE_ROOT_LICENSE_DEFAULT",
    "TEMPLATE_PACKAGE_NOTICE",
}


def _license_notice_category(header: str) -> str:
    """分类 PO 文件级许可说明，同时保留根许可默认边界。"""
    lowered = header.lower()
    if "same license as the gimp" in lowered:
        return "GIMP_FAMILY_PACKAGE_NOTICE"
    if "same license as the package package" in lowered:
        return "TEMPLATE_PACKAGE_NOTICE"
    if "same license as the disk package" in lowered:
        return "FOREIGN_DISK_PACKAGE_NOTICE"
    if "same license as" in lowered:
        raise BroadQaExternalDataError("v9 gettext 未冻结的许可说明")
    return "NO_FILE_NOTICE_ROOT_LICENSE_DEFAULT"


def _entry_identity(
        domain: str,
        entry: polib.POEntry,
        ) -> tuple[str, str, str, str]:
    """返回跨 locale 稳定且跨 domain 不碰撞的 source identity。"""
    return domain, entry.msgctxt or "", entry.msgid, entry.msgid_plural or ""


def _entry_record(
        *,
        domain: str,
        entry: polib.POEntry,
        locale_role: str,
        ordinal: int,
        source_file_id: str,
        source_file_sha256: str,
        ) -> dict[str, object]:
    """保存完整 PO entry、来源和 obsolete 状态。"""
    source_identity = {
        "domain": domain,
        "msgctxt": entry.msgctxt or "",
        "msgid": entry.msgid,
        "msgid_plural": entry.msgid_plural or "",
    }
    semantic = {
        "comment": entry.comment or "",
        "flags": sorted(entry.flags),
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
        "source_identity": source_identity,
        "tcomment": entry.tcomment or "",
    }
    return {
        **semantic,
        "entry_linenum": entry.linenum,
        "entry_ordinal": ordinal,
        "locale_role": locale_role,
        "semantic_sha256": sha256_hex(canonical_json_bytes(semantic)),
        "source_file_id": source_file_id,
        "source_file_sha256": source_file_sha256,
    }


def _parse_po(
        payload: bytes,
        *,
        domain: str,
        expected_language: str,
        locale_role: str,
        source_file_id: str,
        ) -> tuple[
            dict[tuple[str, str, str, str], dict[str, object]],
            dict[str, object],
        ]:
    """严格解析 UTF-8/LF PO 并拒绝重复 source identity。"""
    if (not isinstance(payload, bytes) or not payload
            or payload.startswith(b"\xef\xbb\xbf") or b"\r" in payload
            or not payload.endswith(b"\n")
            or locale_role not in _LOCALE_ROLES
            or not isinstance(source_file_id, str)
            or len(source_file_id) != 64):
        raise BroadQaExternalDataError("v9 gettext 输入编码或合同漂移")
    try:
        parsed = polib.pofile(payload.decode("utf-8"), wrapwidth=0)
    except (UnicodeDecodeError, OSError, ValueError) as error:
        raise BroadQaExternalDataError("v9 gettext parser 失败") from error
    if (parsed.metadata.get("Language") != expected_language
            or parsed.metadata.get("Content-Type")
            != "text/plain; charset=UTF-8"):
        raise BroadQaExternalDataError("v9 gettext locale/charset 漂移")
    category = _license_notice_category(parsed.header or "")
    records = {}
    for ordinal, entry in enumerate(parsed):
        identity = _entry_identity(domain, entry)
        if identity in records:
            raise BroadQaExternalDataError("v9 gettext source identity 重复")
        records[identity] = _entry_record(
            domain=domain,
            entry=entry,
            locale_role=locale_role,
            ordinal=ordinal,
            source_file_id=source_file_id,
            source_file_sha256=sha256_hex(payload),
        )
    return records, {
        "empty_translation_count": sum(not entry.msgstr for entry in parsed),
        "entry_count": len(parsed),
        "fuzzy_count": sum("fuzzy" in entry.flags for entry in parsed),
        "header_sha256": sha256_hex((parsed.header or "").encode("utf-8")),
        "language": parsed.metadata.get("Language"),
        "license_notice_category": category,
        "metadata_sha256": sha256_hex(canonical_json_bytes(
            dict(sorted(parsed.metadata.items())))),
        "obsolete_count": sum(int(entry.obsolete) for entry in parsed),
        "plural_count": sum(bool(entry.msgid_plural) for entry in parsed),
        "translated_count": len(parsed.translated_entries()),
    }


def _source_file_record(
        *,
        domain: str,
        locale_role: str,
        relative_path: str,
        payload: bytes,
        parser_summary: dict[str, object],
        ) -> dict[str, object]:
    """形成一份 locale blob 与 header 元数据的 commitment。"""
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
        "header_sha256": parser_summary["header_sha256"],
        "license_notice_category": parser_summary[
            "license_notice_category"],
        "locale_role": locale_role,
        "record_kind": V9_GETTEXT_SOURCE_FILE_RECORD_KIND,
        "role": "TRANSLATION_WITH_OFFICIAL_SOURCE_GETTEXT_PO",
    }


def _pair_exclusion_reasons(
        zh_hans: dict[str, object],
        zh_hant: dict[str, object],
        ) -> tuple[str, ...]:
    """返回 common identity 未进入正式 plain 分母的全部原因。"""
    reasons = []
    hans_source = zh_hans["source_identity"]
    hant_source = zh_hant["source_identity"]
    if hans_source["msgid_plural"] or hant_source["msgid_plural"]:
        reasons.append("plural")
    if zh_hans["obsolete"] or zh_hant["obsolete"]:
        reasons.append("obsolete")
    if "fuzzy" in zh_hans["flags"] or "fuzzy" in zh_hant["flags"]:
        reasons.append("fuzzy")
    if not zh_hans["msgstr"] or not zh_hant["msgstr"]:
        reasons.append("empty")
    return tuple(reasons)


def _pair_records(
        *,
        source_family: str,
        source_policy_scope: str,
        license_expression: str,
        by_locale: dict[
            str, dict[tuple[str, str, str, str], dict[str, object]]],
        ) -> tuple[
            tuple[dict[str, object], ...], dict[str, object]]:
    """派生严格 plain pair，并完整分账所有排除原因。"""
    common = sorted(set(by_locale["zh_Hans"]).intersection(
        by_locale["zh_Hant"]))
    exclusions = Counter()
    domain_common = Counter(key[0] for key in common)
    values = []
    for key in common:
        zh_hans = by_locale["zh_Hans"][key]
        zh_hant = by_locale["zh_Hant"][key]
        reasons = _pair_exclusion_reasons(zh_hans, zh_hant)
        if reasons:
            exclusions["any"] += 1
            for reason in reasons:
                exclusions[reason] += 1
            continue
        source_identity = zh_hans["source_identity"]
        if source_identity != zh_hant["source_identity"]:
            raise BroadQaExternalDataError("v9 gettext pair identity 漂移")
        features = localization_pair_features(
            str(zh_hant["msgstr"]),
            str(zh_hans["msgstr"]),
            scalar_limit=V9_GETTEXT_TEXT_SCALAR_MAX,
        )
        identity = {
            "record_kind": V9_GETTEXT_PAIR_RECORD_KIND,
            "source_family": source_family,
            "source_identity": source_identity,
            "source_policy_scope": source_policy_scope,
        }
        values.append({
            **features,
            "format_version": 1,
            "license_expression": license_expression,
            "official_source_text": source_identity["msgid"],
            "pair_id": localization_record_id(identity),
            "record_kind": V9_GETTEXT_PAIR_RECORD_KIND,
            "source_family": source_family,
            "source_identity": source_identity,
            "source_identity_sha256": sha256_hex(
                canonical_json_bytes(source_identity)),
            "source_policy_scope": source_policy_scope,
            "v9_evaluation_eligible": int(
                features["training_eligible"] == 1
                and features["contains_han_both"] == 1),
            "zh_hans": zh_hans,
            "zh_hant": zh_hant,
        })
    if len({item["pair_id"] for item in values}) != len(values):
        raise BroadQaExternalDataError("v9 gettext pair identity 重复")
    return tuple(values), {
        "common_source_identity_count": len(common),
        "domain_common_source_identity_counts": dict(sorted(
            domain_common.items())),
        "excluded_any_count": exclusions["any"],
        "excluded_empty_count": exclusions["empty"],
        "excluded_fuzzy_count": exclusions["fuzzy"],
        "excluded_obsolete_count": exclusions["obsolete"],
        "excluded_plural_count": exclusions["plural"],
    }


def _summary(
        *,
        file_records: tuple[dict[str, object], ...],
        pairs: tuple[dict[str, object], ...],
        exclusion_summary: dict[str, object],
        locale_summaries: dict[str, dict[str, dict[str, object]]],
        ) -> dict[str, object]:
    """汇总真实 plain 分母、结构资格、冲突与许可 header 事实。"""
    counts = Counter()
    domain_counts = Counter()
    outputs_by_input: dict[str, set[str]] = {}
    for item in pairs:
        counts["identity_pair_count"] += int(item["identity_preservation"])
        counts["equal_length_pair_count"] += int(item["equal_length"])
        counts["single_han_difference_count"] += int(
            item["single_han_difference"])
        counts["structure_equal_count"] += int(item["structure_equal"])
        counts["contains_han_both_count"] += int(item["contains_han_both"])
        counts["v9_evaluation_eligible_pair_count"] += int(
            item["v9_evaluation_eligible"])
        domain_counts[str(item["source_identity"]["domain"])] += 1
        input_text = str(item["zh_hant"]["msgstr"])
        outputs_by_input.setdefault(input_text, set()).add(
            str(item["zh_hans"]["msgstr"]))
    notice_counts = Counter(
        str(item["license_notice_category"]) for item in file_records)
    if set(notice_counts).difference(_LICENSE_NOTICE_CATEGORIES):
        raise BroadQaExternalDataError("v9 gettext license notice 漂移")
    pair_count = len(pairs)
    return {
        **exclusion_summary,
        "contains_han_both_count": counts["contains_han_both_count"],
        "content_outcome": (
            "PASS_NONZERO_ACTIVE_COMMON_PAIR" if pair_count
            else "REJECTED_ZERO_ACTIVE_COMMON_PAIR"),
        "domain_pair_counts": dict(sorted(domain_counts.items())),
        "equal_length_pair_count": counts["equal_length_pair_count"],
        "identity_pair_count": counts["identity_pair_count"],
        "input_conflict_count": sum(
            len(outputs) > 1 for outputs in outputs_by_input.values()),
        "license_notice_census": dict(sorted(notice_counts.items())),
        "locale_summaries": locale_summaries,
        "nonidentity_pair_count": pair_count - counts["identity_pair_count"],
        "plain_pair_count": pair_count,
        "single_han_difference_count": counts[
            "single_han_difference_count"],
        "source_file_count": len(file_records),
        "source_format_policy": {
            "empty_fuzzy_obsolete_or_plural_pair_allowed": 0,
            "official_source_identity": (
                "DOMAIN_MSGCTXT_MSGID_MSGID_PLURAL"),
            "parser": f"polib {polib.__version__}",
            "root_license_expression": "GPL-3.0-or-later",
            "structure_unequal_pair_preserved": 1,
        },
        "structure_equal_count": counts["structure_equal_count"],
        "v9_evaluation_eligible_pair_count": counts[
            "v9_evaluation_eligible_pair_count"],
        "variable_length_pair_count": (
            pair_count - counts["equal_length_pair_count"]),
    }


def derive_normalization_recovery_v9_gettext_source_records(
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
    """从冻结的多 domain PO blob 派生 v9 held-out records。"""
    if (not source_family or not source_policy_scope
            or license_expression != "GPL-3.0-or-later"
            or not isinstance(pair_specs, tuple) or not pair_specs
            or not isinstance(files, dict)):
        raise BroadQaExternalDataError("v9 gettext source contract 非法")
    expected_paths = []
    domains = set()
    file_records = []
    by_locale = {role: {} for role in _LOCALE_ROLES}
    locale_summaries: dict[str, dict[str, dict[str, object]]] = {
        role: {} for role in _LOCALE_ROLES}
    for spec in pair_specs:
        if not isinstance(spec, dict):
            raise BroadQaExternalDataError("v9 gettext pair spec 非对象")
        domain = spec.get("domain")
        if not isinstance(domain, str) or not domain or domain in domains:
            raise BroadQaExternalDataError("v9 gettext domain identity 漂移")
        domains.add(domain)
        for role in _LOCALE_ROLES:
            value = spec.get(role)
            if (not isinstance(value, dict)
                    or set(value) != {"expected_language", "relative_path"}
                    or not isinstance(value["expected_language"], str)
                    or not isinstance(value["relative_path"], str)):
                raise BroadQaExternalDataError(
                    "v9 gettext locale pair spec 漂移")
            path = value["relative_path"]
            if path in expected_paths or path not in files:
                raise BroadQaExternalDataError(
                    "v9 gettext source file roster 漂移")
            expected_paths.append(path)
            payload = files[path]
            identity = {
                "git_blob_sha1": git_blob_sha1(payload),
                "relative_path": path,
                "sha256": sha256_hex(payload),
            }
            source_file_id = localization_record_id(identity)
            entries, parser_summary = _parse_po(
                payload,
                domain=domain,
                expected_language=value["expected_language"],
                locale_role=role,
                source_file_id=source_file_id,
            )
            overlap = set(by_locale[role]).intersection(entries)
            if overlap:
                raise BroadQaExternalDataError(
                    "v9 gettext cross-domain source identity 重复")
            by_locale[role].update(entries)
            locale_summaries[role][domain] = parser_summary
            file_records.append(_source_file_record(
                domain=domain,
                locale_role=role,
                relative_path=path,
                payload=payload,
                parser_summary=parser_summary,
            ))
    if set(files) != set(expected_paths):
        raise BroadQaExternalDataError("v9 gettext unselected file 混入")
    file_records.sort(key=lambda item: str(item["relative_path"]))
    pairs, exclusions = _pair_records(
        source_family=source_family,
        source_policy_scope=source_policy_scope,
        license_expression=license_expression,
        by_locale=by_locale,
    )
    return tuple(file_records), pairs, _summary(
        file_records=tuple(file_records),
        pairs=pairs,
        exclusion_summary=exclusions,
        locale_summaries=locale_summaries,
    )


__all__ = [
    "V9_GETTEXT_PAIR_RECORD_KIND",
    "V9_GETTEXT_SOURCE_FILE_RECORD_KIND",
    "V9_GETTEXT_TEXT_SCALAR_MAX",
    "derive_normalization_recovery_v9_gettext_source_records",
]
