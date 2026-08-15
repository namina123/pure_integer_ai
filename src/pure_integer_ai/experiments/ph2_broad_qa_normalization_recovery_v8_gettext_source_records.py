"""派生 recovery-v8 参数化 gettext PO TRAIN source records。

调用方先冻结 domain、locale path与Git blob，再把逐字节核验后的payload交给
本模块。parser按完整 ``domain/msgctxt/msgid/msgid_plural`` 对齐；active、
singular、nonfuzzy、nonobsolete、nonempty common pair全部保留。
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


V8_GETTEXT_SOURCE_FILE_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V8_GETTEXT_SOURCE_FILE_V1")
V8_GETTEXT_PAIR_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V8_GETTEXT_PAIR_V1")
V8_GETTEXT_TEXT_SCALAR_MAX = 4_096

_LOCALE_ROLES = ("zh_Hans", "zh_Hant")


def _source_file_record(
        *,
        domain: str,
        locale_role: str,
        relative_path: str,
        payload: bytes,
        ) -> dict[str, object]:
    """形成一份gettext locale文件的逐字节commitment。"""
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
        "record_kind": V8_GETTEXT_SOURCE_FILE_RECORD_KIND,
        "role": "TRANSLATION_WITH_OFFICIAL_SOURCE_GETTEXT_PO",
    }


def _entry_identity(
        domain: str,
        entry: polib.POEntry,
        ) -> tuple[str, str, str, str]:
    """返回跨locale稳定的完整gettext source identity。"""
    return (
        domain,
        entry.msgctxt or "",
        entry.msgid,
        entry.msgid_plural or "",
    )


def _entry_record(
        *,
        domain: str,
        entry: polib.POEntry,
        locale_role: str,
        source_file_id: str,
        source_file_sha256: str,
        ) -> dict[str, object]:
    """保存active entry的surface、来源注释与语义commitment。"""
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
        "occurrences": [
            [str(path), str(line)] for path, line in entry.occurrences],
        "previous_msgctxt": entry.previous_msgctxt or "",
        "previous_msgid": entry.previous_msgid or "",
        "previous_msgid_plural": entry.previous_msgid_plural or "",
        "source_identity": source_identity,
        "tcomment": entry.tcomment or "",
    }
    return {
        **semantic,
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
    """解析单份PO并返回全部entry map与locale census。"""
    if (not isinstance(payload, bytes) or not payload
            or not isinstance(domain, str) or not domain
            or not isinstance(expected_language, str) or not expected_language
            or locale_role not in _LOCALE_ROLES
            or not isinstance(source_file_id, str)
            or len(source_file_id) != 64):
        raise BroadQaExternalDataError("v8 gettext input contract 漂移")
    try:
        text = payload.decode("utf-8", errors="strict")
        parsed = polib.pofile(text, wrapwidth=0)
    except (UnicodeDecodeError, OSError, ValueError) as error:
        raise BroadQaExternalDataError("v8 gettext parser 失败") from error
    language = parsed.metadata.get("Language")
    if language != expected_language:
        raise BroadQaExternalDataError("v8 gettext language header 漂移")
    records = {}
    counts = Counter()
    source_sha256 = sha256_hex(payload)
    for entry in parsed:
        counts["entry_count"] += 1
        counts["obsolete_count"] += int(entry.obsolete)
        counts["fuzzy_count"] += int("fuzzy" in entry.flags)
        counts["plural_count"] += int(bool(entry.msgid_plural))
        counts["empty_translation_count"] += int(not bool(entry.msgstr))
        identity = _entry_identity(domain, entry)
        if identity in records:
            raise BroadQaExternalDataError(
                "v8 gettext source identity 重复")
        records[identity] = _entry_record(
            domain=domain,
            entry=entry,
            locale_role=locale_role,
            source_file_id=source_file_id,
            source_file_sha256=source_sha256,
        )
    return records, {
        "empty_translation_count": counts["empty_translation_count"],
        "entry_count": counts["entry_count"],
        "fuzzy_count": counts["fuzzy_count"],
        "language": language,
        "metadata_sha256": sha256_hex(canonical_json_bytes(
            dict(sorted(parsed.metadata.items())))),
        "obsolete_count": counts["obsolete_count"],
        "plural_count": counts["plural_count"],
    }


def _active(entry: dict[str, object]) -> bool:
    """判定entry可进入active common source inventory。"""
    source = entry["source_identity"]
    return bool(
        not source["msgid_plural"]
        and "fuzzy" not in entry["flags"]
        and entry["msgstr"])


def _pair_records(
        *,
        source_family: str,
        source_policy_scope: str,
        license_expression: str,
        by_locale: dict[
            str, dict[tuple[str, str, str, str], dict[str, object]]],
        ) -> tuple[dict[str, object], ...]:
    """按完整source identity保留所有双方active common pair。"""
    if set(by_locale) != set(_LOCALE_ROLES):
        raise BroadQaExternalDataError("v8 gettext locale roster 漂移")
    common = sorted(set(by_locale["zh_Hans"]).intersection(
        by_locale["zh_Hant"]))
    values = []
    for key in common:
        hans = by_locale["zh_Hans"][key]
        hant = by_locale["zh_Hant"][key]
        if not _active(hans) or not _active(hant):
            continue
        source_identity = hans["source_identity"]
        if source_identity != hant["source_identity"]:
            raise BroadQaExternalDataError(
                "v8 gettext source identity 漂移")
        features = localization_pair_features(
            str(hant["msgstr"]),
            str(hans["msgstr"]),
            scalar_limit=V8_GETTEXT_TEXT_SCALAR_MAX,
        )
        identity = {
            "record_kind": V8_GETTEXT_PAIR_RECORD_KIND,
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
            "record_kind": V8_GETTEXT_PAIR_RECORD_KIND,
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
        raise BroadQaExternalDataError("v8 gettext pair identity 重复")
    return tuple(values)


def _summary(
        *,
        file_records: tuple[dict[str, object], ...],
        pairs: tuple[dict[str, object], ...],
        locale_summaries: dict[str, dict[str, dict[str, object]]],
        ) -> dict[str, object]:
    """汇总gettext pair资格、domain分布和input conflict。"""
    counts = Counter()
    domain_counts = Counter()
    outputs_by_input: dict[str, set[str]] = {}
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
        input_text = str(item["zh_hant"]["msgstr"])
        outputs_by_input.setdefault(input_text, set()).add(
            str(item["zh_hans"]["msgstr"]))
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
            "active_singular_nonfuzzy_nonobsolete_nonempty_required": 1,
            "official_source_identity": (
                "DOMAIN_MSGCTXT_MSGID_MSGID_PLURAL"),
            "parser": f"polib {polib.__version__}",
            "structure_unequal_pair_preserved": 1,
        },
        "structure_equal_count": counts["structure_equal_count"],
        "v8_training_eligible_pair_count": counts[
            "v8_training_eligible_pair_count"],
        "variable_length_pair_count": counts["variable_length_pair_count"],
    }


def derive_normalization_recovery_v8_gettext_source_records(
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
    """从已冻结的成对PO blob派生共享v8 source records。"""
    if (not isinstance(source_family, str) or not source_family
            or not isinstance(source_policy_scope, str)
            or not source_policy_scope
            or not isinstance(license_expression, str)
            or not license_expression
            or not isinstance(pair_specs, tuple) or not pair_specs
            or not isinstance(files, dict)):
        raise BroadQaExternalDataError("v8 gettext source contract 非法")
    expected_paths = []
    domains = set()
    file_records = []
    by_locale = {role: {} for role in _LOCALE_ROLES}
    locale_summaries: dict[str, dict[str, dict[str, object]]] = {
        role: {} for role in _LOCALE_ROLES}
    for spec in pair_specs:
        if not isinstance(spec, dict):
            raise BroadQaExternalDataError("v8 gettext pair spec 非对象")
        domain = spec.get("domain")
        if not isinstance(domain, str) or not domain or domain in domains:
            raise BroadQaExternalDataError("v8 gettext domain identity 漂移")
        domains.add(domain)
        for role in _LOCALE_ROLES:
            value = spec.get(role)
            if (not isinstance(value, dict)
                    or set(value) != {"expected_language", "relative_path"}
                    or not isinstance(value["expected_language"], str)
                    or not value["expected_language"]
                    or not isinstance(value["relative_path"], str)
                    or not value["relative_path"]):
                raise BroadQaExternalDataError(
                    "v8 gettext locale pair spec 漂移")
            path = value["relative_path"]
            if path in expected_paths or path not in files:
                raise BroadQaExternalDataError(
                    "v8 gettext source file roster 漂移")
            expected_paths.append(path)
            payload = files[path]
            record = _source_file_record(
                domain=domain,
                locale_role=role,
                relative_path=path,
                payload=payload,
            )
            records, summary = _parse_po(
                payload,
                domain=domain,
                expected_language=value["expected_language"],
                locale_role=role,
                source_file_id=str(record["file_id"]),
            )
            overlap = set(by_locale[role]).intersection(records)
            if overlap:
                raise BroadQaExternalDataError(
                    "v8 gettext cross-domain source identity 重复")
            by_locale[role].update(records)
            locale_summaries[role][domain] = summary
            file_records.append(record)
    if set(files) != set(expected_paths):
        raise BroadQaExternalDataError("v8 gettext unselected file 混入")
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
    "V8_GETTEXT_PAIR_RECORD_KIND",
    "V8_GETTEXT_SOURCE_FILE_RECORD_KIND",
    "V8_GETTEXT_TEXT_SCALAR_MAX",
    "derive_normalization_recovery_v8_gettext_source_records",
]
