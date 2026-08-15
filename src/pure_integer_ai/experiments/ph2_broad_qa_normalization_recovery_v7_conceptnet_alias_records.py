"""派生 recovery-v7 neutral source 的 ConceptNet 英中 alias 记录。

本模块只做确定性的离散投影：从 TRAIN neutral source 提取最多四个 ASCII
词元的连续短语，并接收调用方已经严格解析的 ConceptNet ``/r/Synonym``
assertion。alias 保留逐 assertion 来源、许可和整数 weight，不把外部关系升级为
项目真值，也不创建 learner、candidate、runtime 或 formal evaluation。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import re
from urllib.parse import unquote_to_bytes

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
)
from pure_integer_ai.experiments.ph2_conceptnet_adapter import (
    ConceptNetAssertion,
    ENDPOINT_CONCEPT,
    LICENSE_PARTITIONS,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


CONCEPTNET_ALIAS_EVIDENCE_KIND = (
    "NORMALIZATION_RECOVERY_V7_CONCEPTNET_ALIAS_EVIDENCE_V1")
CONCEPTNET_ALIAS_ROUTE_KIND = (
    "NORMALIZATION_RECOVERY_V7_CONCEPTNET_ALIAS_ROUTE_V1")
CONCEPTNET_ALIAS_FAMILY_COVERAGE_KIND = (
    "NORMALIZATION_RECOVERY_V7_CONCEPTNET_ALIAS_FAMILY_COVERAGE_V1")

NEUTRAL_PHRASE_UNIT_MAX = 4

_SOURCE_UNIT = re.compile(r"[A-Za-z]+|\d+")
_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_ACRONYM_BOUNDARY = re.compile(r"([A-Z]+)([A-Z][a-z])")
_SOURCE_ORDER = (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
)


def _sha256(payload: bytes) -> str:
    """返回规范 identity 或 UTF-8 surface 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(identity))


def _surface_commitment(value: str) -> dict[str, object]:
    """形成训练 surface 的 UTF-8 长度与 SHA 承诺。"""
    if not isinstance(value, str) or not value:
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias surface 为空或非法")
    encoded = value.encode("utf-8")
    return {
        "bytes": len(encoded),
        "scalar_length": len(value),
        "sha256": _sha256(encoded),
    }


def neutral_source_units(value: str) -> tuple[str, ...]:
    """按边界提取 ASCII 英文字母或 Unicode 十进制数字单元。"""
    if not isinstance(value, str) or not value:
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias neutral source 为空或非法")
    separated = _CAMEL_BOUNDARY.sub(r"\1 \2", value)
    separated = _ACRONYM_BOUNDARY.sub(r"\1 \2", separated)
    return tuple(item.lower() for item in _SOURCE_UNIT.findall(separated))


def neutral_source_phrases(value: str) -> tuple[str, ...]:
    """形成一至四单元的有序连续 neutral 短语集合。"""
    units = neutral_source_units(value)
    values = set()
    maximum = min(NEUTRAL_PHRASE_UNIT_MAX, len(units))
    for length in range(1, maximum + 1):
        for start in range(0, len(units) - length + 1):
            values.add(" ".join(units[start:start + length]))
    return tuple(sorted(values))


def normalize_conceptnet_term(term: str, *, language: str) -> str:
    """按 ConceptNet URI term 规则严格恢复 UTF-8 surface。"""
    if (not isinstance(term, str) or not term
            or language not in {"en", "zh"}):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias endpoint term 非法")
    try:
        decoded = unquote_to_bytes(term).decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias endpoint UTF-8 非法") from error
    value = decoded.replace("_", " ")
    if not value.strip():
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias endpoint surface 为空")
    if language == "en":
        return " ".join(value.lower().split())
    return value


def _source_cluster_sha256(
        dataset: str,
        sources: list[dict[str, object]],
        ) -> str:
    """按 ConceptNet adapter 合同重算 dataset/source cluster identity。"""
    return _sha256(canonical_json_line({
        "dataset": dataset,
        "sources": sorted(sources, key=canonical_json_line),
    }))


def derive_neutral_phrase_inventory(
        rows_by_family: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[
            dict[str, dict[str, set[str]]],
            dict[str, dict[str, tuple[str, ...]]],
            tuple[dict[str, object], ...],
        ]:
    """从 transient neutral rows 派生 phrase->family/pair 与 pair inventory。"""
    if set(rows_by_family) != set(_SOURCE_ORDER[:-1]):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias neutral family inventory 漂移")
    phrase_support: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set))
    pair_phrases: dict[str, dict[str, tuple[str, ...]]] = {
        family: {} for family in _SOURCE_ORDER}
    family_records = []
    for family in _SOURCE_ORDER:
        rows = rows_by_family.get(family, ())
        unique_phrases = set()
        for row in rows:
            pair_id = row.get("pair_id") if isinstance(row, dict) else None
            surface = row.get("_neutral_surface") \
                if isinstance(row, dict) else None
            if (not isinstance(pair_id, str) or len(pair_id) != 64
                    or pair_id in pair_phrases[family]
                    or not isinstance(surface, str) or not surface):
                raise BroadQaExternalDataError(
                    "v7 ConceptNet alias neutral row 漂移")
            phrases = neutral_source_phrases(surface)
            pair_phrases[family][pair_id] = phrases
            unique_phrases.update(phrases)
            for phrase in phrases:
                phrase_support[phrase][family].add(pair_id)
        identity = {
            "source_family": family,
            "target_scope": "NEUTRAL_SOURCE_CONCEPTNET_ALIAS_V1",
        }
        family_records.append({
            **identity,
            "family_coverage_id": _record_id(identity),
            "format_version": 1,
            "matched_neutral_phrase_count": 0,
            "pair_any_alias_count": 0,
            "projected_pair_count": len(rows),
            "record_kind": CONCEPTNET_ALIAS_FAMILY_COVERAGE_KIND,
            "unique_neutral_phrase_count": len(unique_phrases),
        })
    return phrase_support, pair_phrases, tuple(family_records)


def conceptnet_alias_evidence_record(
        assertion: ConceptNetAssertion,
        *,
        phrase_support: dict[str, dict[str, set[str]]],
        ) -> dict[str, object] | None:
    """把命中 neutral phrase 的英中 Synonym assertion 投影为来源化证据。"""
    if not isinstance(assertion, ConceptNetAssertion):
        raise TypeError("v7 ConceptNet alias 需要 ConceptNetAssertion")
    if assertion.relation != "/r/Synonym":
        return None
    if (assertion.start.kind != ENDPOINT_CONCEPT
            or assertion.end.kind != ENDPOINT_CONCEPT):
        return None
    if assertion.start.language == "en" and assertion.end.language == "zh":
        english_endpoint, chinese_endpoint = assertion.start, assertion.end
    elif (assertion.start.language == "zh"
          and assertion.end.language == "en"):
        english_endpoint, chinese_endpoint = assertion.end, assertion.start
    else:
        return None
    english = normalize_conceptnet_term(
        english_endpoint.term, language="en")
    support = phrase_support.get(english)
    if not support:
        return None
    chinese = normalize_conceptnet_term(
        chinese_endpoint.term, language="zh")
    english_commitment = _surface_commitment(english)
    chinese_commitment = _surface_commitment(chinese)
    sources = [item.to_value() for item in assertion.sources]
    identity = {
        "assertion_uri": assertion.assertion_uri,
        "source_cluster_sha256": assertion.source_cluster_sha256,
    }
    return {
        "alias_evidence_id": _record_id(identity),
        "assertion_uri": assertion.assertion_uri,
        "chinese_suffix": list(chinese_endpoint.suffix),
        "chinese_surface": chinese,
        "chinese_surface_bytes": chinese_commitment["bytes"],
        "chinese_surface_scalar_length": chinese_commitment[
            "scalar_length"],
        "chinese_surface_sha256": chinese_commitment["sha256"],
        "dataset": assertion.dataset,
        "english_suffix": list(english_endpoint.suffix),
        "english_surface": english,
        "english_surface_bytes": english_commitment["bytes"],
        "english_surface_scalar_length": english_commitment[
            "scalar_length"],
        "english_surface_sha256": english_commitment["sha256"],
        "format_version": 1,
        "license_id": assertion.license_partition,
        "license_text": assertion.license_text,
        "line_number": assertion.line_number,
        "metadata_sha256": assertion.metadata_sha256,
        "neutral_source_families": sorted(support),
        "record_kind": CONCEPTNET_ALIAS_EVIDENCE_KIND,
        "source_cluster_sha256": assertion.source_cluster_sha256,
        "sources": sources,
        "weight_denominator": assertion.weight_denominator,
        "weight_numerator": assertion.weight_numerator,
    }


def _valid_digest(value: object) -> bool:
    """判断值是否为规范小写 SHA-256。"""
    return (isinstance(value, str) and len(value) == 64
            and all(item in "0123456789abcdef" for item in value))


def _validate_alias_evidence_record(
        item: dict[str, object],
        *,
        phrase_support: dict[str, dict[str, set[str]]],
        ) -> tuple[str, str, str, int]:
    """严格核验一个 stored assertion evidence 的全部可重算字段。"""
    evidence_id = item.get("alias_evidence_id")
    assertion_uri = item.get("assertion_uri")
    line_number = item.get("line_number")
    english = item.get("english_surface")
    chinese = item.get("chinese_surface")
    dataset = item.get("dataset")
    sources = item.get("sources")
    source_cluster = item.get("source_cluster_sha256")
    license_text = item.get("license_text")
    numerator = item.get("weight_numerator")
    denominator = item.get("weight_denominator")
    if (item.get("record_kind") != CONCEPTNET_ALIAS_EVIDENCE_KIND
            or item.get("format_version") != 1
            or not isinstance(assertion_uri, str) or not assertion_uri
            or type(line_number) is not int or line_number <= 0
            or not isinstance(english, str) or english not in phrase_support
            or english != " ".join(english.lower().split())
            or not isinstance(chinese, str) or not chinese.strip()
            or not isinstance(dataset, str) or not dataset.startswith("/d/")
            or not isinstance(sources, list) or not sources
            or any(not isinstance(source, dict) or not source
                   for source in sources)
            or not _valid_digest(source_cluster)
            or _source_cluster_sha256(dataset, sources) != source_cluster
            or not _valid_digest(item.get("metadata_sha256"))
            or not isinstance(license_text, str)
            or item.get("license_id") != LICENSE_PARTITIONS.get(license_text)
            or type(numerator) is not int or numerator <= 0
            or type(denominator) is not int or denominator <= 0
            or math.gcd(numerator, denominator) != 1
            or item.get("neutral_source_families")
            != sorted(phrase_support[english])):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias evidence schema/identity 漂移")
    english_commitment = _surface_commitment(english)
    chinese_commitment = _surface_commitment(chinese)
    if (item.get("english_surface_bytes") != english_commitment["bytes"]
            or item.get("english_surface_scalar_length")
            != english_commitment["scalar_length"]
            or item.get("english_surface_sha256")
            != english_commitment["sha256"]
            or item.get("chinese_surface_bytes")
            != chinese_commitment["bytes"]
            or item.get("chinese_surface_scalar_length")
            != chinese_commitment["scalar_length"]
            or item.get("chinese_surface_sha256")
            != chinese_commitment["sha256"]
            or not isinstance(item.get("english_suffix"), list)
            or not isinstance(item.get("chinese_suffix"), list)
            or any(not isinstance(value, str) or not value
                   for key in ("english_suffix", "chinese_suffix")
                   for value in item[key])
            or not _valid_digest(evidence_id)
            or evidence_id != _record_id({
                "assertion_uri": assertion_uri,
                "source_cluster_sha256": source_cluster,
            })):
        raise BroadQaExternalDataError(
            "v7 ConceptNet alias evidence commitment 漂移")
    return evidence_id, assertion_uri, english, line_number


def derive_conceptnet_alias_routes(
        evidence: tuple[dict[str, object], ...],
        *,
        phrase_support: dict[str, dict[str, set[str]]],
        pair_phrases: dict[str, dict[str, tuple[str, ...]]],
        family_records: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """聚合英文 route、更新 family coverage，并形成可回读摘要。"""
    by_english: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list))
    evidence_ids = set()
    assertion_uris = set()
    line_numbers = set()
    for item in evidence:
        if not isinstance(item, dict):
            raise BroadQaExternalDataError(
                "v7 ConceptNet alias evidence schema/identity 漂移")
        evidence_id, assertion_uri, english, line_number = (
            _validate_alias_evidence_record(
                item, phrase_support=phrase_support))
        chinese = item.get("chinese_surface") \
            if isinstance(item, dict) else None
        if (evidence_id in evidence_ids
                or assertion_uri in assertion_uris
                or line_number in line_numbers
                or not isinstance(chinese, str)):
            raise BroadQaExternalDataError(
                "v7 ConceptNet alias evidence schema/identity 漂移")
        evidence_ids.add(evidence_id)
        assertion_uris.add(assertion_uri)
        line_numbers.add(line_number)
        by_english[english][chinese].append(item)
    routes = []
    for english, variants in by_english.items():
        variant_records = []
        for chinese, items in sorted(variants.items()):
            variant_records.append({
                "chinese_surface": chinese,
                "chinese_surface_sha256": _surface_commitment(chinese)[
                    "sha256"],
                "evidence_count": len(items),
                "evidence_ids": sorted(str(item["alias_evidence_id"])
                                       for item in items),
                "license_ids": sorted({str(item["license_id"])
                                       for item in items}),
                "source_cluster_count": len({
                    str(item["source_cluster_sha256"]) for item in items}),
            })
        identity = {
            "english_surface": english,
            "target_scope": "NEUTRAL_SOURCE_CONCEPTNET_ALIAS_V1",
        }
        routes.append({
            **identity,
            "alias_route_id": _record_id(identity),
            "chinese_variant_count": len(variant_records),
            "chinese_variants": variant_records,
            "evidence_count": sum(item["evidence_count"]
                                  for item in variant_records),
            "format_version": 1,
            "neutral_source_families": sorted(phrase_support[english]),
            "record_kind": CONCEPTNET_ALIAS_ROUTE_KIND,
            "unique_chinese_surface": int(len(variant_records) == 1),
        })
    routes.sort(key=lambda item: str(item["alias_route_id"]))
    matched_phrases = set(by_english)
    coverage = []
    for record in family_records:
        family = str(record["source_family"])
        pairs = pair_phrases[family]
        coverage.append({
            **record,
            "matched_neutral_phrase_count": sum(
                1 for phrase in matched_phrases
                if family in phrase_support[phrase]),
            "pair_any_alias_count": sum(
                1 for phrases in pairs.values()
                if any(phrase in matched_phrases for phrase in phrases)),
        })
    variant_counts = Counter(
        int(item["chinese_variant_count"]) for item in routes)
    license_counts = Counter(str(item["license_id"]) for item in evidence)
    return tuple(routes), tuple(coverage), {
        "alias_evidence_count": len(evidence),
        "ambiguous_english_route_count": sum(
            1 for item in routes if item["unique_chinese_surface"] == 0),
        "chinese_variant_count_counts": {
            str(key): value for key, value in sorted(variant_counts.items())},
        "english_route_count": len(routes),
        "license_evidence_counts": dict(sorted(license_counts.items())),
        "matched_english_phrase_count": len(matched_phrases),
        "matched_english_chinese_pair_count": sum(
            int(item["chinese_variant_count"]) for item in routes),
        "neutral_phrase_inventory_count": len(phrase_support),
        "unique_english_route_count": sum(
            1 for item in routes if item["unique_chinese_surface"] == 1),
    }


__all__ = [
    "CONCEPTNET_ALIAS_EVIDENCE_KIND",
    "CONCEPTNET_ALIAS_FAMILY_COVERAGE_KIND",
    "CONCEPTNET_ALIAS_ROUTE_KIND",
    "NEUTRAL_PHRASE_UNIT_MAX",
    "conceptnet_alias_evidence_record",
    "derive_conceptnet_alias_routes",
    "derive_neutral_phrase_inventory",
    "neutral_source_phrases",
    "neutral_source_units",
    "normalize_conceptnet_term",
]
