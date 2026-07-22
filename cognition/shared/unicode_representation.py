"""Unicode scalar sequence 表示概念和整数化 UCD 属性图接线。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    GraphStatement,
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_REPRESENTATION,
    ObjectIdentity,
    TypedRef,
    representation_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _strict_integer_key(values: tuple[int, ...], *,
                        where: str) -> tuple[int, ...]:
    """校验注入键为非空严格整数元组。"""
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{where} 必须是非空严格整数元组")
    assert_int(*values, _where=where)
    if any(type(value) is not int for value in values):
        raise ValueError(f"{where} 必须使用严格整数")
    return values


def validate_unicode_scalars(codepoints: tuple[int, ...]) -> tuple[int, ...]:
    """校验非空有序序列只含 Unicode scalar value，不执行归一化或拆分。"""
    _strict_integer_key(codepoints, where="unicode codepoints")
    for codepoint in codepoints:
        if (codepoint < 0 or codepoint > 0x10FFFF
                or 0xD800 <= codepoint <= 0xDFFF):
            raise ValueError("序列包含非 Unicode scalar value")
    return codepoints


@dataclass(frozen=True, order=True)
class UnicodePropertyEvidence:
    """由 UCD 边界压成整数的版本化外部属性证据。"""

    unicode_version: tuple[int, int, int]
    parser_version: int
    source_hash: int
    namespace_hash: int
    property_hash: int
    value_hash: int
    codepoint: int
    sequence_index: int

    def __post_init__(self) -> None:
        values = (
            *self.unicode_version,
            self.parser_version,
            self.source_hash,
            self.namespace_hash,
            self.property_hash,
            self.value_hash,
            self.codepoint,
            self.sequence_index,
        )
        assert_int(*values, _where="UnicodePropertyEvidence")
        if any(type(value) is not int for value in values):
            raise ValueError("UnicodePropertyEvidence 必须使用严格整数")
        if any(value < 0 for value in values):
            raise ValueError("UnicodePropertyEvidence 字段必须非负")
        validate_unicode_scalars((self.codepoint,))

    def stable_key(self) -> tuple[int, ...]:
        """返回含版本、来源和序列位置的完整整数 evidence 键。"""
        return (
            *self.unicode_version,
            self.parser_version,
            self.source_hash,
            self.namespace_hash,
            self.property_hash,
            self.value_hash,
            self.codepoint,
            self.sequence_index,
        )

    def property_identity(self) -> ObjectIdentity:
        """构造可跨码点复用、但不跨 UCD 版本混合的属性概念身份。"""
        return ObjectIdentity(OBJECT_CONCEPT, (
            *self.unicode_version,
            self.parser_version,
            self.source_hash,
            self.namespace_hash,
            self.property_hash,
            self.value_hash,
        ))

    @classmethod
    def from_integer_key(cls, key: tuple[int, ...]) -> "UnicodePropertyEvidence":
        """从 UCD adapter 的固定十元整数键恢复证据。"""
        if not isinstance(key, tuple) or len(key) != 10:
            raise ValueError("UnicodePropertyEvidence 整数键长度必须为 10")
        return cls(
            (key[0], key[1], key[2]),
            key[3], key[4], key[5], key[6], key[7], key[8], key[9],
        )


@dataclass(frozen=True)
class UnicodePropertyLink:
    """表示序列上的一个已核验 UCD 属性位置链接。"""

    evidence: UnicodePropertyEvidence
    property_ref: TypedRef
    statement: GraphStatement


class UnicodeSequenceMaterializer:
    """物化 Unicode sequence，并用动态 predicate 附加外部属性概念。"""

    def __init__(self, ontology: GraphOntology, *,
                 family_key: tuple[int, ...],
                 external_property_relation_key: tuple[int, ...]) -> None:
        self._ontology = ontology
        self._family_key = _strict_integer_key(
            family_key, where="Unicode family_key")
        relation_key = _strict_integer_key(
            external_property_relation_key,
            where="Unicode external_property_relation_key")
        self._property_predicate_identity = relation_concept_identity(
            relation_key)

    def identity(self, codepoints: tuple[int, ...]) -> ObjectIdentity:
        """构造不含 language、UCD 版本或最终结构作用的序列表示身份。"""
        return representation_identity(
            self._family_key, validate_unicode_scalars(codepoints))

    def materialize(self, codepoints: tuple[int, ...]) -> TypedRef:
        """幂等物化单码点或多码点 Unicode sequence 表示概念。"""
        return self._ontology.materialize(self.identity(codepoints))

    def materialize_with_properties(
            self, codepoints: tuple[int, ...],
            evidence_keys: tuple[tuple[int, ...], ...], *,
            scope: ScopeIdentity, provenance_kind: int,
            epistemic_origin: int = 0) -> tuple[TypedRef, tuple[UnicodePropertyLink, ...]]:
        """物化序列并把已验证 UCD evidence 作为位置化外部属性写入图。"""
        codepoints = validate_unicode_scalars(codepoints)
        sequence_ref = self.materialize(codepoints)
        predicate = self._ontology.materialize(
            self._property_predicate_identity)
        links: list[UnicodePropertyLink] = []
        for key in evidence_keys:
            evidence = UnicodePropertyEvidence.from_integer_key(key)
            if evidence.sequence_index >= len(codepoints):
                raise ValueError("UCD evidence 的 sequence_index 越界")
            if codepoints[evidence.sequence_index] != evidence.codepoint:
                raise ValueError("UCD evidence 与序列码点不一致")
            property_ref = self._ontology.materialize(
                evidence.property_identity())
            statement = self._ontology.relate(
                predicate,
                sequence_ref,
                property_ref,
                scope=scope,
                provenance_kind=provenance_kind,
                epistemic_origin=epistemic_origin,
                content_version=evidence.parser_version,
                qualifiers=(evidence.sequence_index, evidence.codepoint),
            )
            links.append(UnicodePropertyLink(
                evidence, property_ref, statement))
        return sequence_ref, tuple(sorted(
            links, key=lambda item: item.evidence.stable_key()))

    def property_links(self, sequence_ref: TypedRef) -> tuple[UnicodePropertyLink, ...]:
        """从图和 assertion qualifiers 恢复某序列的全部整数化属性链接。"""
        identity = self._ontology.identity_of(sequence_ref)
        if identity.object_kind != OBJECT_REPRESENTATION:
            raise ValueError("sequence_ref 不是 Representation")
        predicate = self._ontology.resolve(self._property_predicate_identity)
        if predicate is None:
            return ()
        links: list[UnicodePropertyLink] = []
        for statement in self._ontology.statements(
                predicate=predicate, subject=sequence_ref):
            property_identity = self._ontology.identity_of(statement.object)
            components = property_identity.components
            qualifiers = statement.assertion.qualifiers
            if len(components) != 8 or len(qualifiers) != 2:
                raise ValueError("Unicode 属性 statement 结构非法")
            if statement.assertion.content_version != components[3]:
                raise ValueError("Unicode 属性 parser version 不一致")
            evidence = UnicodePropertyEvidence(
                (components[0], components[1], components[2]),
                components[3], components[4], components[5],
                components[6], components[7],
                qualifiers[1], qualifiers[0],
            )
            links.append(UnicodePropertyLink(
                evidence, statement.object, statement))
        return tuple(sorted(
            links, key=lambda item: item.evidence.stable_key()))


__all__ = [
    "UnicodePropertyEvidence",
    "UnicodePropertyLink",
    "UnicodeSequenceMaterializer",
    "validate_unicode_scalars",
]
