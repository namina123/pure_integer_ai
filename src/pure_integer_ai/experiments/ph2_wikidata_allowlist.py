"""Wikidata revision-pinned bounded QID/property allowlist 合同。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)


SOURCE_KEY = "WIKIDATA_REVISION_V1"
LICENSE_ID = "CC0-1.0"
FORMAT_VERSION = 1
CURRENT_ALLOWLIST_REVISION = 2
PREVIOUS_ALLOWLIST_SHA256 = (
    "c052a162ef203dda770e48524a150ef8da2446de20b6b9995de972c85aabae48"
)
ENTITY_BASE_URL = "https://www.wikidata.org/wiki/Special:EntityData"

REQUIRED_ENTITY_SEEDS = {
    1: (
        ("Q89", "apple-polysemy", "held_out",
         ("LABEL_ALIAS_DESCRIPTION", "POLYSEMY")),
        ("Q312", "apple-polysemy", "held_out",
         ("LABEL_ALIAS_DESCRIPTION", "POLYSEMY")),
        ("Q313", "celestial-alias", "train",
         ("ALIAS", "LABEL_ALIAS_DESCRIPTION")),
        ("Q361", "war-sequence", "held_out",
         ("LABEL_ALIAS_DESCRIPTION", "P155_P156")),
        ("Q362", "war-sequence", "held_out",
         ("LABEL_ALIAS_DESCRIPTION", "P155_P156")),
        ("Q446", "vehicle-mereology", "dev",
         ("LABEL_ALIAS_DESCRIPTION", "P361_P527")),
        ("Q1420", "vehicle-mereology", "dev",
         ("LABEL_ALIAS_DESCRIPTION", "P361_P527")),
        ("Q5113", "bird-taxonomy", "train",
         ("LABEL_ALIAS_DESCRIPTION", "P31_P279")),
        ("Q25364", "bird-taxonomy", "train",
         ("LABEL_ALIAS_DESCRIPTION", "P31_P279")),
        ("Q82069695", "pandemic-cause", "dev",
         ("LABEL_ALIAS_DESCRIPTION", "P828_P1542")),
        ("Q84263196", "pandemic-cause", "dev",
         ("LABEL_ALIAS_DESCRIPTION", "P828_P1542")),
    ),
    2: (
        ("Q89", "apple-polysemy", "held_out",
         ("LABEL_ALIAS_DESCRIPTION", "POLYSEMY")),
        ("Q312", "apple-polysemy", "held_out",
         ("LABEL_ALIAS_DESCRIPTION", "POLYSEMY")),
        ("Q313", "celestial-alias", "train",
         ("ALIAS", "LABEL_ALIAS_DESCRIPTION")),
        ("Q361", "war-sequence", "held_out",
         ("LABEL_ALIAS_DESCRIPTION", "P155_P156")),
        ("Q362", "war-sequence", "held_out",
         ("LABEL_ALIAS_DESCRIPTION", "P155_P156")),
        ("Q446", "vehicle-mereology", "dev",
         ("LABEL_ALIAS_DESCRIPTION", "P361_P527")),
        ("Q1420", "vehicle-mereology", "dev",
         ("LABEL_ALIAS_DESCRIPTION", "P361_P527")),
        ("Q5113", "animal-taxonomy", "train",
         ("LABEL_ALIAS_DESCRIPTION", "P31_P279")),
        ("Q25364", "animal-taxonomy", "train",
         ("LABEL_ALIAS_DESCRIPTION", "P31_P279")),
        ("Q82069695", "pandemic-cause", "dev",
         ("LABEL_ALIAS_DESCRIPTION", "P828_P1542")),
        ("Q84263196", "pandemic-cause", "dev",
         ("LABEL_ALIAS_DESCRIPTION", "P828_P1542")),
    ),
}

REQUIRED_PROPERTY_RULES = {
    "P31": ("MEMBER", "SUBJECT_TO_OBJECT", ("wikibase-item",)),
    "P279": ("SUBSET", "SUBJECT_TO_OBJECT", ("wikibase-item",)),
    "P361": ("PART_OF", "SUBJECT_TO_OBJECT", ("wikibase-item",)),
    "P527": ("HAS_PART", "SUBJECT_TO_OBJECT", ("wikibase-item",)),
    "P155": ("PRECEDES", "OBJECT_TO_SUBJECT", ("wikibase-item",)),
    "P156": ("PRECEDES", "SUBJECT_TO_OBJECT", ("wikibase-item",)),
    "P828": ("CAUSES", "OBJECT_TO_SUBJECT", ("wikibase-item",)),
    "P1542": ("CAUSES", "SUBJECT_TO_OBJECT", ("wikibase-item",)),
    "P17": ("PROPERTY", "TYPED_VALUE", ("wikibase-item",)),
    "P571": ("PROPERTY", "TYPED_VALUE", ("time",)),
    "P2048": ("PROPERTY", "TYPED_VALUE", ("quantity",)),
}

REQUIRED_CLAIM_CONTRACT = {
    "deprecated_statement_policy": "RETAIN_EXCLUDE_POSITIVE_EVIDENCE",
    "novalue_policy": "RETAIN_NONPOSITIVE",
    "preserve_mainsnak": 1,
    "preserve_qualifier_order": 1,
    "preserve_qualifiers": 1,
    "preserve_rank": 1,
    "preserve_reference_order": 1,
    "preserve_references": 1,
    "preserve_statement_id": 1,
    "property_allowlist_only": 1,
    "somevalue_policy": "RETAIN_NONPOSITIVE",
    "unknown_datatype_policy": "ANOMALY",
}


class WikidataAllowlistError(RuntimeError):
    """Wikidata QID/property allowlist、claim contract 或 URL 非法。"""


def _text(value: Any, *, where: str) -> str:
    """要求 allowlist 文本为无首尾空白的非空字符串。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise WikidataAllowlistError(f"{where} 必须是非空文本")
    return value


def _qid(value: Any) -> str:
    """校验 QID 为无前导零的正整数实体标识。"""
    text = _text(value, where="QID")
    if not re.fullmatch(r"Q[1-9][0-9]*", text):
        raise WikidataAllowlistError("QID 非法")
    return text


def _property_id(value: Any) -> str:
    """校验 property id 为无前导零的正整数属性标识。"""
    text = _text(value, where="property id")
    if not re.fullmatch(r"P[1-9][0-9]*", text):
        raise WikidataAllowlistError("property id 非法")
    return text


@dataclass(frozen=True, order=True)
class WikidataEntitySeed:
    """一个预注册 QID 的保守来源簇、split 和用途。"""

    qid: str
    cluster_id: str
    split: str
    purpose_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "qid", _qid(self.qid))
        _text(self.cluster_id, where="entity cluster_id")
        if self.split not in {"train", "dev", "held_out"}:
            raise WikidataAllowlistError("entity split 非法")
        if (not isinstance(self.purpose_keys, tuple) or not self.purpose_keys
                or any(not isinstance(item, str) or not item
                       for item in self.purpose_keys)
                or tuple(sorted(set(self.purpose_keys))) != self.purpose_keys):
            raise WikidataAllowlistError("entity purpose_keys 必须唯一有序")

    def to_dict(self) -> dict[str, Any]:
        """导出 QID seed。"""
        return {
            "cluster_id": self.cluster_id,
            "purpose_keys": list(self.purpose_keys),
            "qid": self.qid,
            "split": self.split,
        }


@dataclass(frozen=True, order=True)
class WikidataPropertyRule:
    """一个逐项注册的 Wikidata property 到项目 family 的方向合同。"""

    property_id: str
    project_relation_family: str
    direction: str
    allowed_datatypes: tuple[str, ...]
    purpose: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "property_id", _property_id(self.property_id))
        _text(self.project_relation_family, where="project relation family")
        if self.direction not in {
                "SUBJECT_TO_OBJECT", "OBJECT_TO_SUBJECT", "TYPED_VALUE"}:
            raise WikidataAllowlistError("property direction 非法")
        if (not isinstance(self.allowed_datatypes, tuple)
                or not self.allowed_datatypes
                or tuple(sorted(set(self.allowed_datatypes)))
                != self.allowed_datatypes):
            raise WikidataAllowlistError("allowed_datatypes 必须唯一有序")
        _text(self.purpose, where="property purpose")

    def to_dict(self) -> dict[str, Any]:
        """导出 property rule。"""
        return {
            "allowed_datatypes": list(self.allowed_datatypes),
            "direction": self.direction,
            "project_relation_family": self.project_relation_family,
            "property_id": self.property_id,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class WikidataRevisionAllowlist:
    """冻结 bounded QID、property mapping 与不可丢失 claim 字段。"""

    format_version: int
    source_key: str
    license_id: str
    entities: tuple[WikidataEntitySeed, ...]
    properties: tuple[WikidataPropertyRule, ...]
    claim_contract: dict[str, Any]
    allowlist_revision: int = 1
    supersedes_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != FORMAT_VERSION:
            raise WikidataAllowlistError("allowlist format_version 非法")
        if (type(self.allowlist_revision) is not int
                or self.allowlist_revision not in REQUIRED_ENTITY_SEEDS):
            raise WikidataAllowlistError("allowlist revision 非法")
        if self.allowlist_revision == 1:
            if self.supersedes_sha256 is not None:
                raise WikidataAllowlistError("首版 allowlist 不得 supersede")
        elif self.supersedes_sha256 != PREVIOUS_ALLOWLIST_SHA256:
            raise WikidataAllowlistError("allowlist supersede hash 非法")
        if self.source_key != SOURCE_KEY or self.license_id != LICENSE_ID:
            raise WikidataAllowlistError("allowlist source/license 非法")
        if not self.entities or not self.properties:
            raise WikidataAllowlistError("allowlist entities/properties 不能为空")
        object.__setattr__(
            self,
            "entities",
            tuple(sorted(self.entities, key=lambda item: int(item.qid[1:]))),
        )
        object.__setattr__(
            self,
            "properties",
            tuple(sorted(
                self.properties,
                key=lambda item: int(item.property_id[1:]),
            )),
        )
        qids = [item.qid for item in self.entities]
        if len(qids) != len(set(qids)):
            raise WikidataAllowlistError("allowlist QID 重复")
        cluster_splits: dict[str, str] = {}
        for item in self.entities:
            prior = cluster_splits.setdefault(item.cluster_id, item.split)
            if prior != item.split:
                raise WikidataAllowlistError("同一 QID cluster 跨 split")
        actual_entities = tuple(
            (item.qid, item.cluster_id, item.split, item.purpose_keys)
            for item in self.entities
        )
        if actual_entities != REQUIRED_ENTITY_SEEDS[self.allowlist_revision]:
            raise WikidataAllowlistError("entity allowlist 未逐项匹配注册表")
        property_ids = [item.property_id for item in self.properties]
        if len(property_ids) != len(set(property_ids)):
            raise WikidataAllowlistError("allowlist property 重复")
        actual_rules = {
            item.property_id: (
                item.project_relation_family,
                item.direction,
                item.allowed_datatypes,
            )
            for item in self.properties
        }
        if actual_rules != REQUIRED_PROPERTY_RULES:
            raise WikidataAllowlistError("property mapping 未逐项匹配注册表")
        if self.claim_contract != REQUIRED_CLAIM_CONTRACT:
            raise WikidataAllowlistError("claim contract 缺字段或被放宽")

    def to_dict(self) -> dict[str, Any]:
        """导出规范 allowlist object。"""
        value = {
            "claim_contract": dict(self.claim_contract),
            "entity_allowlist": [item.to_dict() for item in self.entities],
            "format_version": self.format_version,
            "license_id": self.license_id,
            "property_allowlist": [item.to_dict() for item in self.properties],
            "source_key": self.source_key,
        }
        if self.allowlist_revision > 1:
            value["allowlist_revision"] = self.allowlist_revision
            value["supersedes_sha256"] = self.supersedes_sha256
        return value

    def canonical_bytes(self) -> bytes:
        """返回唯一换行结尾的规范 JSON。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回 allowlist 规范 SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _entity_from_dict(value: dict[str, Any]) -> WikidataEntitySeed:
    """从 JSON object 恢复一个 QID seed。"""
    return WikidataEntitySeed(
        str(value["qid"]),
        str(value["cluster_id"]),
        str(value["split"]),
        tuple(str(item) for item in value["purpose_keys"]),
    )


def _property_from_dict(value: dict[str, Any]) -> WikidataPropertyRule:
    """从 JSON object 恢复一个 property rule。"""
    return WikidataPropertyRule(
        str(value["property_id"]),
        str(value["project_relation_family"]),
        str(value["direction"]),
        tuple(str(item) for item in value["allowed_datatypes"]),
        str(value["purpose"]),
    )


def read_wikidata_allowlist(path: str | Path) -> WikidataRevisionAllowlist:
    """严格读取规范 allowlist，拒绝 float、非规范字节和字段漂移。"""
    try:
        payload = Path(path).read_bytes()
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        allowlist = WikidataRevisionAllowlist(
            value["format_version"],
            str(value["source_key"]),
            str(value["license_id"]),
            tuple(_entity_from_dict(item) for item in value["entity_allowlist"]),
            tuple(_property_from_dict(item)
                  for item in value["property_allowlist"]),
            dict(value["claim_contract"]),
            value.get("allowlist_revision", 1),
            value.get("supersedes_sha256"),
        )
    except Exception as error:
        raise WikidataAllowlistError("Wikidata allowlist 无法恢复") from error
    if (not payload.endswith(b"\n") or payload.endswith(b"\n\n")
            or allowlist.canonical_bytes() != payload):
        raise WikidataAllowlistError("Wikidata allowlist 规范字节漂移")
    return allowlist


def wikidata_entity_url(qid: str, *, revision: int | None = None) -> str:
    """构造官方 entity JSON URL；正式获取必须提供正整数 revision。"""
    entity = _qid(qid)
    base = f"{ENTITY_BASE_URL}/{entity}.json"
    if revision is None:
        return base
    if type(revision) is not int or revision <= 0:
        raise WikidataAllowlistError("Wikidata revision 必须为正整数")
    return f"{base}?revision={revision}"


__all__ = [
    "ENTITY_BASE_URL",
    "FORMAT_VERSION",
    "CURRENT_ALLOWLIST_REVISION",
    "LICENSE_ID",
    "PREVIOUS_ALLOWLIST_SHA256",
    "REQUIRED_CLAIM_CONTRACT",
    "REQUIRED_ENTITY_SEEDS",
    "REQUIRED_PROPERTY_RULES",
    "SOURCE_KEY",
    "WikidataAllowlistError",
    "WikidataEntitySeed",
    "WikidataPropertyRule",
    "WikidataRevisionAllowlist",
    "read_wikidata_allowlist",
    "wikidata_entity_url",
]
