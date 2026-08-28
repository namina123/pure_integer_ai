"""从权威图读取表面变体，不在核心查询中猜测语言关系。

调用方注入表示族、变体 predicate 和遍历方向。模块只执行通用的
Representation -> Representation 图读取，并把已核验的整数码点还原为
边缘文本；不包含语言名称、脚本名称、转换表或方向规则。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_REPRESENTATION,
    ObjectIdentity,
    representation_identity,
)
from pure_integer_ai.cognition.shared.unicode_representation import (
    representation_surface,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer.unicode_codec import encode


# Traversal direction is part of the injected graph protocol, not language data.
OUTGOING = 1
INCOMING = 2


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


@dataclass(frozen=True, slots=True)
class GraphSurfaceVariantProvider:
    """把一个注入的图 predicate 暴露为查询层可消费的表面候选器。"""

    ontology: GraphOntology
    family_key: tuple[int, ...]
    variant_relation_key: tuple[int, ...]
    directions: tuple[int, ...] = (OUTGOING, INCOMING)
    max_variants: int = 64

    def __post_init__(self) -> None:
        if not isinstance(self.ontology, GraphOntology):
            raise TypeError("GraphSurfaceVariantProvider.ontology 类型错误")
        _strict_key(self.family_key, where="surface variant family_key")
        _strict_key(
            self.variant_relation_key,
            where="surface variant relation_key",
        )
        if (not isinstance(self.directions, tuple) or not self.directions
                or any(type(item) is not int for item in self.directions)
                or any(item not in (OUTGOING, INCOMING)
                       for item in self.directions)
                or len(set(self.directions)) != len(self.directions)):
            raise ValueError("surface variant directions 非法")
        if type(self.max_variants) is not int or not 1 <= self.max_variants <= 4096:
            raise ValueError("surface variant max_variants 非法")

    def __call__(self, text: str) -> tuple[str, ...]:
        """返回图中全部可见竞争表面；无图证据时返回空 tuple。"""
        if not isinstance(text, str) or not text:
            return ()
        representation = self.ontology.resolve(
            self._representation_identity(text))
        if representation is None:
            return ()
        predicate = self.ontology.resolve(
            relation_concept_identity(self.variant_relation_key))
        if predicate is None:
            return ()
        refs: dict[tuple[int, ...], object] = {}
        for direction in self.directions:
            statements = (
                self.ontology.statements(
                    predicate=predicate, subject=representation)
                if direction == OUTGOING else
                self.ontology.statements(
                    predicate=predicate, object_ref=representation)
            )
            for statement in statements:
                target = (statement.object if direction == OUTGOING
                          else statement.subject)
                identity = self.ontology.identity_of(target)
                if identity.object_kind != OBJECT_REPRESENTATION:
                    continue
                surface = representation_surface(
                    identity, family_key=self.family_key)
                if surface is None or surface == text:
                    continue
                refs[identity.stable_key()] = surface
                if len(refs) > self.max_variants:
                    raise ValueError("图表面变体数量超过预算")
        return tuple(refs[key] for key in sorted(refs))

    def _representation_identity(self, text: str) -> ObjectIdentity:
        """按注入表示族构造原始文本的权威 Representation 身份。"""
        return representation_identity(self.family_key, encode(text))


__all__ = [
    "GraphSurfaceVariantProvider",
    "INCOMING",
    "OUTGOING",
]
