"""Wikidata fixed-revision EntityData 的纯整数 statement parser。"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_wikidata_allowlist import (
    WikidataPropertyRule,
)


KNOWN_DATATYPES = frozenset({
    "commonsMedia",
    "entity-schema",
    "external-id",
    "geo-shape",
    "globe-coordinate",
    "math",
    "monolingualtext",
    "musical-notation",
    "quantity",
    "string",
    "tabular-data",
    "time",
    "url",
    "wikibase-form",
    "wikibase-item",
    "wikibase-lexeme",
    "wikibase-property",
    "wikibase-sense",
})
RANKS = frozenset({"deprecated", "normal", "preferred"})
SNAKTYPES = frozenset({"novalue", "somevalue", "value"})
_QID_RE = re.compile(r"Q[1-9][0-9]*")
_PROPERTY_RE = re.compile(r"P[1-9][0-9]*")
_DECIMAL_RE = re.compile(
    r"-?(?:0|[1-9][0-9]*)\.[0-9]+(?:[eE][+-]?[0-9]+)?"
    r"|-?(?:0|[1-9][0-9]*)[eE][+-]?[0-9]+"
)


class WikidataAdapterError(RuntimeError):
    """EntityData JSON、实体身份或 statement 合同不一致。"""


class WikidataStatementError(WikidataAdapterError):
    """单条 statement 可原子隔离的结构错误。"""

    def __init__(self, code: str, message: str) -> None:
        """保存稳定 anomaly code 与可读错误说明。"""
        super().__init__(message)
        self.code = code


class _RawDecimal(str):
    """保留上游 JSON 十进制原文，避免形成 binary float。"""


def _parse_decimal(text: str) -> _RawDecimal:
    """验证 JSON 十进制词元并以文本对象返回。"""
    if not _DECIMAL_RE.fullmatch(text):
        raise WikidataAdapterError("Wikidata JSON 十进制非法")
    return _RawDecimal(text)


def _reject_constant(text: str) -> None:
    """拒绝 NaN/Infinity 等非 JSON 数值。"""
    raise WikidataAdapterError(f"Wikidata JSON 常量非法: {text}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """构造 object 并拒绝重复 key。"""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WikidataAdapterError("Wikidata JSON object key 重复")
        result[key] = value
    return result


def _parse_json(payload: bytes) -> dict[str, Any]:
    """严格解析 UTF-8 EntityData，不经过浮点且拒绝重复 key。"""
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_float=_parse_decimal,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise WikidataAdapterError("Wikidata EntityData JSON 损坏") from error
    if not isinstance(value, dict):
        raise WikidataAdapterError("Wikidata EntityData 根必须是 object")
    return value


def _normalized(value: Any) -> Any:
    """把 raw decimal 显式标记为文本并递归生成无浮点 JSON 值。"""
    if isinstance(value, _RawDecimal):
        return {"raw_decimal_text": str(value)}
    if value is None or isinstance(value, (bool, str)) or type(value) is int:
        return value
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalized(item) for key, item in value.items()}
    raise WikidataAdapterError("Wikidata JSON 含不支持的值类型")


def _text(value: Any, *, where: str) -> str:
    """要求非空文本没有首尾空白。"""
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise WikidataStatementError("BAD_TEXT", f"{where} 必须是非空文本")
    return value


def _property_id(value: Any, *, where: str) -> str:
    """要求合法无前导零 property id。"""
    text = _text(value, where=where)
    if not _PROPERTY_RE.fullmatch(text):
        raise WikidataStatementError("BAD_PROPERTY", f"{where} 非法")
    return text


def _exact_keys(
        value: dict[str, Any],
        *,
        required: frozenset[str],
        optional: frozenset[str],
        code: str,
        where: str,
        ) -> None:
    """拒绝缺字段和未来未知字段，防止静默丢失上游结构。"""
    keys = frozenset(value)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise WikidataStatementError(code, f"{where} 字段集合非法")


@dataclass(frozen=True)
class WikidataValidatedStatement:
    """一个完整保留 mainsnak/qualifier/rank/reference 的 statement。"""

    qid: str
    property_id: str
    statement_id: str
    rank: str
    payload: CanonicalJsonObject
    qualifier_snak_count: int
    reference_count: int
    reference_snak_count: int
    snaktype_counts: CanonicalJsonObject
    datatype_counts: CanonicalJsonObject

    def to_event_dict(self) -> dict[str, Any]:
        """导出 parser event，完整 payload 而非只保留 mainsnak。"""
        return {
            "event_kind": "WIKIDATA_VALIDATED_STATEMENT_V1",
            "payload": self.payload.to_value(),
            "property_id": self.property_id,
            "qid": self.qid,
            "statement_id": self.statement_id,
        }


def _counter_add(counter: dict[str, int], key: str) -> None:
    """向纯整数计数器增加一次。"""
    counter[key] = counter.get(key, 0) + 1


def _validate_snak(
        value: Any,
        *,
        expected_property: str,
        datatype_counts: dict[str, int],
        snaktype_counts: dict[str, int],
        ) -> dict[str, Any]:
    """验证并完整规范化一个 mainsnak、qualifier 或 reference snak。"""
    if not isinstance(value, dict):
        raise WikidataStatementError("BAD_SNAK", "snak 必须是 object")
    _exact_keys(
        value,
        required=frozenset({"datatype", "property", "snaktype"}),
        optional=frozenset({"datavalue", "hash"}),
        code="BAD_SNAK_FIELDS",
        where="snak",
    )
    property_id = _property_id(value["property"], where="snak.property")
    if property_id != expected_property:
        raise WikidataStatementError(
            "BAD_SNAK_PROPERTY", "snak property 与所属位置不一致")
    snaktype = _text(value["snaktype"], where="snak.snaktype")
    if snaktype not in SNAKTYPES:
        raise WikidataStatementError("BAD_SNAKTYPE", "snaktype 非法")
    datatype = _text(value["datatype"], where="snak.datatype")
    if datatype not in KNOWN_DATATYPES:
        raise WikidataStatementError(
            "UNKNOWN_DATATYPE", "snak datatype 未注册")
    if snaktype == "value":
        datavalue = value.get("datavalue")
        if not isinstance(datavalue, dict):
            raise WikidataStatementError(
                "BAD_DATAVALUE", "value snak 缺 datavalue")
        _exact_keys(
            datavalue,
            required=frozenset({"type", "value"}),
            optional=frozenset(),
            code="BAD_DATAVALUE_FIELDS",
            where="datavalue",
        )
        _text(datavalue["type"], where="datavalue.type")
    elif "datavalue" in value:
        raise WikidataStatementError(
            "BAD_NONVALUE_DATAVALUE",
            "somevalue/novalue 不得携带 datavalue",
        )
    if "hash" in value:
        _text(value["hash"], where="snak.hash")
    _counter_add(snaktype_counts, snaktype)
    _counter_add(datatype_counts, datatype)
    normalized = _normalized(value)
    assert isinstance(normalized, dict)
    return normalized


def _validate_ordered_snak_map(
        snaks: Any,
        order: Any,
        *,
        datatype_counts: dict[str, int],
        snaktype_counts: dict[str, int],
        where: str,
        ) -> tuple[dict[str, Any], int]:
    """验证 qualifier/reference snak map 与显式 property 顺序完全一致。"""
    if not isinstance(snaks, dict) or not isinstance(order, list):
        raise WikidataStatementError(
            "BAD_SNAK_ORDER", f"{where} snaks/order 类型非法")
    ordered_properties = [
        _property_id(item, where=f"{where}.order") for item in order
    ]
    if (len(ordered_properties) != len(set(ordered_properties))
            or set(ordered_properties) != set(snaks)):
        raise WikidataStatementError(
            "BAD_SNAK_ORDER", f"{where} property 顺序不完整")
    normalized: dict[str, Any] = {}
    count = 0
    for property_id, values in snaks.items():
        _property_id(property_id, where=f"{where}.property")
        if not isinstance(values, list) or not values:
            raise WikidataStatementError(
                "BAD_SNAK_LIST", f"{where} snak list 不能为空")
        normalized[property_id] = [
            _validate_snak(
                item,
                expected_property=property_id,
                datatype_counts=datatype_counts,
                snaktype_counts=snaktype_counts,
            )
            for item in values
        ]
        count += len(values)
    return normalized, count


def parse_wikidata_statement(
        value: Any,
        *,
        qid: str,
        property_rule: WikidataPropertyRule,
        ) -> WikidataValidatedStatement:
    """验证一个 allowlisted statement 并保留所有承重子结构。"""
    if not _QID_RE.fullmatch(qid):
        raise WikidataStatementError("BAD_QID", "statement QID 非法")
    if not isinstance(property_rule, WikidataPropertyRule):
        raise WikidataStatementError(
            "BAD_PROPERTY_RULE", "property rule 类型非法")
    if not isinstance(value, dict):
        raise WikidataStatementError("BAD_STATEMENT", "statement 必须是 object")
    _exact_keys(
        value,
        required=frozenset({"id", "mainsnak", "rank", "type"}),
        optional=frozenset({
            "qualifiers", "qualifiers-order", "references",
        }),
        code="BAD_STATEMENT_FIELDS",
        where="statement",
    )
    if value["type"] != "statement":
        raise WikidataStatementError("BAD_STATEMENT_TYPE", "statement type 非法")
    statement_id = _text(value["id"], where="statement.id")
    rank = _text(value["rank"], where="statement.rank")
    if rank not in RANKS:
        raise WikidataStatementError("BAD_RANK", "statement rank 非法")

    datatype_counts: dict[str, int] = {}
    snaktype_counts: dict[str, int] = {}
    normalized_main = _validate_snak(
        value["mainsnak"],
        expected_property=property_rule.property_id,
        datatype_counts=datatype_counts,
        snaktype_counts=snaktype_counts,
    )
    if normalized_main["datatype"] not in property_rule.allowed_datatypes:
        raise WikidataStatementError(
            "BAD_MAIN_DATATYPE",
            "allowlisted property 的 mainsnak datatype 不匹配",
        )

    has_qualifiers = "qualifiers" in value or "qualifiers-order" in value
    if has_qualifiers and not {
            "qualifiers", "qualifiers-order"}.issubset(value):
        raise WikidataStatementError(
            "BAD_QUALIFIER_FIELDS", "qualifier map/order 必须同时存在")
    qualifiers: dict[str, Any] = {}
    qualifier_order: list[str] = []
    qualifier_count = 0
    if has_qualifiers:
        qualifiers, qualifier_count = _validate_ordered_snak_map(
            value["qualifiers"],
            value["qualifiers-order"],
            datatype_counts=datatype_counts,
            snaktype_counts=snaktype_counts,
            where="qualifiers",
        )
        qualifier_order = list(value["qualifiers-order"])

    references_value = value.get("references", [])
    if not isinstance(references_value, list):
        raise WikidataStatementError(
            "BAD_REFERENCES", "references 必须是 list")
    references: list[dict[str, Any]] = []
    reference_snak_count = 0
    for reference in references_value:
        if not isinstance(reference, dict):
            raise WikidataStatementError(
                "BAD_REFERENCE", "reference 必须是 object")
        _exact_keys(
            reference,
            required=frozenset({"hash", "snaks", "snaks-order"}),
            optional=frozenset(),
            code="BAD_REFERENCE_FIELDS",
            where="reference",
        )
        _text(reference["hash"], where="reference.hash")
        normalized_snaks, count = _validate_ordered_snak_map(
            reference["snaks"],
            reference["snaks-order"],
            datatype_counts=datatype_counts,
            snaktype_counts=snaktype_counts,
            where="reference",
        )
        references.append({
            "hash": reference["hash"],
            "snaks": normalized_snaks,
            "snaks-order": list(reference["snaks-order"]),
        })
        reference_snak_count += count

    payload = {
        "id": statement_id,
        "mainsnak": normalized_main,
        "rank": rank,
        "type": "statement",
    }
    if has_qualifiers:
        payload["qualifiers"] = qualifiers
        payload["qualifiers-order"] = qualifier_order
    if "references" in value:
        payload["references"] = references
    return WikidataValidatedStatement(
        qid=qid,
        property_id=property_rule.property_id,
        statement_id=statement_id,
        rank=rank,
        payload=CanonicalJsonObject.from_value(payload),
        qualifier_snak_count=qualifier_count,
        reference_count=len(references),
        reference_snak_count=reference_snak_count,
        snaktype_counts=CanonicalJsonObject.from_value(snaktype_counts),
        datatype_counts=CanonicalJsonObject.from_value(datatype_counts),
    )


def _validate_terms(value: Any, *, aliases: bool) -> dict[str, Any]:
    """验证 labels/descriptions/aliases 的 language/value 身份。"""
    if not isinstance(value, dict):
        raise WikidataAdapterError("Wikidata term map 非法")
    normalized: dict[str, Any] = {}
    for language, entry in value.items():
        if not isinstance(language, str) or not language:
            raise WikidataAdapterError("Wikidata term language 非法")
        entries = entry if aliases else [entry]
        if not isinstance(entries, list) or not entries:
            raise WikidataAdapterError("Wikidata alias list 非法")
        normalized_entries: list[dict[str, str]] = []
        for item in entries:
            if (not isinstance(item, dict)
                    or set(item) != {"language", "value"}
                    or item.get("language") != language
                    or not isinstance(item.get("value"), str)
                    or not item["value"]):
                raise WikidataAdapterError("Wikidata term entry 非法")
            normalized_entries.append({
                "language": language,
                "value": item["value"],
            })
        normalized[language] = (
            normalized_entries if aliases else normalized_entries[0])
    return normalized


@dataclass(frozen=True)
class WikidataEntityTerms:
    """严格回读的 fixed-revision label、description 与 alias。"""

    qid: str
    revision: int
    labels: CanonicalJsonObject
    descriptions: CanonicalJsonObject
    aliases: CanonicalJsonObject


def _parse_wikidata_entity(
        payload: bytes,
        *,
        expected_qid: str,
        expected_revision: int,
        ) -> tuple[dict[str, Any], WikidataEntityTerms]:
    """一次解析并闭合 entity identity，供 terms 与 statement 扫描共用。"""
    if (not isinstance(payload, bytes) or not payload
            or not _QID_RE.fullmatch(expected_qid)
            or type(expected_revision) is not int or expected_revision <= 0):
        raise WikidataAdapterError("Wikidata entity 输入非法")
    root = _parse_json(payload)
    if set(root) != {"entities"} or not isinstance(root["entities"], dict):
        raise WikidataAdapterError("Wikidata EntityData 根字段非法")
    if set(root["entities"]) != {expected_qid}:
        raise WikidataAdapterError("Wikidata EntityData QID 集不匹配")
    entity = root["entities"][expected_qid]
    required_entity = {
        "aliases", "claims", "descriptions", "id", "labels", "lastrevid",
        "modified", "ns", "pageid", "sitelinks", "title", "type",
    }
    if not isinstance(entity, dict) or not required_entity.issubset(entity):
        raise WikidataAdapterError("Wikidata entity 字段缺失")
    if (entity["id"] != expected_qid or entity["title"] != expected_qid
            or entity["type"] != "item"
            or entity["lastrevid"] != expected_revision):
        raise WikidataAdapterError("Wikidata entity 身份或 revision 不匹配")
    labels = _validate_terms(entity["labels"], aliases=False)
    descriptions = _validate_terms(entity["descriptions"], aliases=False)
    aliases = _validate_terms(entity["aliases"], aliases=True)
    return entity, WikidataEntityTerms(
        expected_qid,
        expected_revision,
        CanonicalJsonObject.from_value(labels),
        CanonicalJsonObject.from_value(descriptions),
        CanonicalJsonObject.from_value(aliases),
    )


def parse_wikidata_entity_terms_bytes(
        payload: bytes,
        *,
        expected_qid: str,
        expected_revision: int,
        ) -> WikidataEntityTerms:
    """公开回读完整 terms，不经浮点且不丢 language/value 身份。"""
    _, terms = _parse_wikidata_entity(
        payload,
        expected_qid=expected_qid,
        expected_revision=expected_revision,
    )
    return terms


@dataclass(frozen=True)
class WikidataEntityScanReport:
    """一个 fixed-revision entity 的身份、statement 与异常摘要。"""

    qid: str
    revision: int
    raw_sha256: str
    raw_size_bytes: int
    label_language_count: int
    description_language_count: int
    alias_language_count: int
    claim_property_count: int
    statement_count: int
    selected_statement_count: int
    valid_statement_count: int
    anomaly_count: int
    anomaly_codes: CanonicalJsonObject
    anomaly_evidence: CanonicalJsonObject
    selected_property_counts: CanonicalJsonObject
    rank_counts: CanonicalJsonObject
    snaktype_counts: CanonicalJsonObject
    datatype_counts: CanonicalJsonObject
    qualifier_snak_count: int
    reference_count: int
    reference_snak_count: int
    terminal_newline_present: int
    event_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """导出 entity scan 的规范 manifest object。"""
        return {
            "alias_language_count": self.alias_language_count,
            "anomaly_codes": self.anomaly_codes.to_value(),
            "anomaly_count": self.anomaly_count,
            "anomaly_evidence": self.anomaly_evidence.to_value()["items"],
            "claim_property_count": self.claim_property_count,
            "datatype_counts": self.datatype_counts.to_value(),
            "description_language_count": self.description_language_count,
            "event_sha256": self.event_sha256,
            "label_language_count": self.label_language_count,
            "qid": self.qid,
            "qualifier_snak_count": self.qualifier_snak_count,
            "rank_counts": self.rank_counts.to_value(),
            "raw_sha256": self.raw_sha256,
            "raw_size_bytes": self.raw_size_bytes,
            "reference_count": self.reference_count,
            "reference_snak_count": self.reference_snak_count,
            "revision": self.revision,
            "selected_property_counts": self.selected_property_counts.to_value(),
            "selected_statement_count": self.selected_statement_count,
            "snaktype_counts": self.snaktype_counts.to_value(),
            "statement_count": self.statement_count,
            "terminal_newline_present": self.terminal_newline_present,
            "valid_statement_count": self.valid_statement_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WikidataEntityScanReport":
        """从 snapshot manifest 恢复不可变 entity scan。"""
        return cls(
            str(value["qid"]),
            value["revision"],
            str(value["raw_sha256"]),
            value["raw_size_bytes"],
            value["label_language_count"],
            value["description_language_count"],
            value["alias_language_count"],
            value["claim_property_count"],
            value["statement_count"],
            value["selected_statement_count"],
            value["valid_statement_count"],
            value["anomaly_count"],
            CanonicalJsonObject.from_value(dict(value["anomaly_codes"])),
            CanonicalJsonObject.from_value({
                "items": list(value["anomaly_evidence"]),
            }),
            CanonicalJsonObject.from_value(
                dict(value["selected_property_counts"])),
            CanonicalJsonObject.from_value(dict(value["rank_counts"])),
            CanonicalJsonObject.from_value(dict(value["snaktype_counts"])),
            CanonicalJsonObject.from_value(dict(value["datatype_counts"])),
            value["qualifier_snak_count"],
            value["reference_count"],
            value["reference_snak_count"],
            value["terminal_newline_present"],
            str(value["event_sha256"]),
        )


def _merge_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    """合并只含非负严格整数的计数字典。"""
    for key, value in source.items():
        if type(value) is not int or value < 0:
            raise WikidataAdapterError("Wikidata statement 计数非法")
        target[key] = target.get(key, 0) + value


def scan_wikidata_entity_bytes(
        payload: bytes,
        *,
        expected_qid: str,
        expected_revision: int,
        property_rules: tuple[WikidataPropertyRule, ...],
        ) -> WikidataEntityScanReport:
    """扫描 fixed-revision EntityData，坏 statement 原子隔离且身份错误停线。"""
    rule_by_property = {rule.property_id: rule for rule in property_rules}
    if len(rule_by_property) != len(property_rules) or not rule_by_property:
        raise WikidataAdapterError("Wikidata property rules 非法")
    entity, terms = _parse_wikidata_entity(
        payload,
        expected_qid=expected_qid,
        expected_revision=expected_revision,
    )
    labels = terms.labels.to_value()
    descriptions = terms.descriptions.to_value()
    aliases = terms.aliases.to_value()
    claims = entity["claims"]
    if not isinstance(claims, dict):
        raise WikidataAdapterError("Wikidata claims 非法")

    event_digest = hashlib.sha256()
    event_digest.update(canonical_json_line({
        "aliases": aliases,
        "descriptions": descriptions,
        "event_kind": "WIKIDATA_ENTITY_TERMS_V1",
        "labels": labels,
        "qid": expected_qid,
        "revision": expected_revision,
    }))
    statement_count = 0
    selected_count = 0
    valid_count = 0
    anomaly_codes: dict[str, int] = {}
    anomaly_evidence: list[dict[str, Any]] = []
    selected_property_counts: dict[str, int] = {}
    rank_counts: dict[str, int] = {}
    snaktype_counts: dict[str, int] = {}
    datatype_counts: dict[str, int] = {}
    qualifier_count = 0
    reference_count = 0
    reference_snak_count = 0
    statement_ids: set[str] = set()
    for property_id, statements in claims.items():
        try:
            _property_id(property_id, where="claims property")
        except WikidataStatementError as error:
            raise WikidataAdapterError("Wikidata claims property 非法") from error
        if not isinstance(statements, list) or not statements:
            raise WikidataAdapterError("Wikidata statement list 不能为空")
        statement_count += len(statements)
        rule = rule_by_property.get(property_id)
        if rule is None:
            continue
        selected_property_counts[property_id] = len(statements)
        for statement_index, statement in enumerate(statements):
            selected_count += 1
            try:
                parsed = parse_wikidata_statement(
                    statement,
                    qid=expected_qid,
                    property_rule=rule,
                )
                if parsed.statement_id in statement_ids:
                    raise WikidataStatementError(
                        "DUPLICATE_STATEMENT_ID", "statement id 重复")
                statement_ids.add(parsed.statement_id)
            except WikidataStatementError as error:
                _counter_add(anomaly_codes, error.code)
                anomaly_evidence.append({
                    "code": error.code,
                    "property_id": property_id,
                    "statement_index": statement_index,
                    "statement_sha256": hashlib.sha256(
                        canonical_json_line(_normalized(statement))).hexdigest(),
                })
                continue
            valid_count += 1
            _counter_add(rank_counts, parsed.rank)
            _merge_counts(snaktype_counts, parsed.snaktype_counts.to_value())
            _merge_counts(datatype_counts, parsed.datatype_counts.to_value())
            qualifier_count += parsed.qualifier_snak_count
            reference_count += parsed.reference_count
            reference_snak_count += parsed.reference_snak_count
            event_digest.update(canonical_json_line(parsed.to_event_dict()))

    return WikidataEntityScanReport(
        qid=expected_qid,
        revision=expected_revision,
        raw_sha256=hashlib.sha256(payload).hexdigest(),
        raw_size_bytes=len(payload),
        label_language_count=len(labels),
        description_language_count=len(descriptions),
        alias_language_count=len(aliases),
        claim_property_count=len(claims),
        statement_count=statement_count,
        selected_statement_count=selected_count,
        valid_statement_count=valid_count,
        anomaly_count=len(anomaly_evidence),
        anomaly_codes=CanonicalJsonObject.from_value(anomaly_codes),
        anomaly_evidence=CanonicalJsonObject.from_value({
            "items": anomaly_evidence,
        }),
        selected_property_counts=CanonicalJsonObject.from_value(
            selected_property_counts),
        rank_counts=CanonicalJsonObject.from_value(rank_counts),
        snaktype_counts=CanonicalJsonObject.from_value(snaktype_counts),
        datatype_counts=CanonicalJsonObject.from_value(datatype_counts),
        qualifier_snak_count=qualifier_count,
        reference_count=reference_count,
        reference_snak_count=reference_snak_count,
        terminal_newline_present=int(payload.endswith(b"\n")),
        event_sha256=event_digest.hexdigest(),
    )


def scan_wikidata_entity_file(
        path: str | Path,
        *,
        expected_qid: str,
        expected_revision: int,
        property_rules: tuple[WikidataPropertyRule, ...],
        expected_sha256: str = "",
        ) -> WikidataEntityScanReport:
    """读取并扫描一个文件，可先核对冻结 SHA-256。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise WikidataAdapterError("Wikidata raw 文件无法读取") from error
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise WikidataAdapterError("Wikidata raw SHA-256 不匹配")
    return scan_wikidata_entity_bytes(
        payload,
        expected_qid=expected_qid,
        expected_revision=expected_revision,
        property_rules=property_rules,
    )


__all__ = [
    "KNOWN_DATATYPES",
    "RANKS",
    "SNAKTYPES",
    "WikidataAdapterError",
    "WikidataEntityScanReport",
    "WikidataEntityTerms",
    "WikidataStatementError",
    "WikidataValidatedStatement",
    "parse_wikidata_entity_terms_bytes",
    "parse_wikidata_statement",
    "scan_wikidata_entity_bytes",
    "scan_wikidata_entity_file",
]
