"""投影已训练 PROPERTY/EVENT_TIME 关系到共同 QueryState。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_AFTER,
    EVENT_TIME_BEFORE,
    EVENT_TIME_DIRECTION_UNKNOWN,
    EVENT_TIME_SAME,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.property_relation import PropertyClaim
from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry, SPACE_CORE

TYPED_RELATION_VERSION = 91600
TYPED_RELATION_NAMESPACE = 91601
TYPED_PROPERTY = 1
TYPED_EVENT_TIME = 2
PROPERTY_RELATION_KIND = 5
EVENT_TIME_RELATION_KINDS = (10, 11, 12, 13)


def _pack(key: tuple[int, ...]) -> tuple[int, ...]:
    return len(key), *key


def _key(value: tuple[int, ...], label: str) -> tuple[int, ...]:
    if type(value) is not tuple or not value or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{label} 必须是非空非负整数 tuple")
    return value


@dataclass(frozen=True, slots=True)
class TypedRelationProjection:
    """一个 active relation 的完整 typed role/filler 投影。"""
    relation_kind: int
    semantic_kind: int
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    fields: tuple[tuple[int, tuple[int, ...]], ...]
    polarity: int = 1
    temporal_direction: int = 0

    def __post_init__(self) -> None:
        if type(self.relation_kind) is not int or self.relation_kind <= 0:
            raise ValueError("typed relation kind 非法")
        if self.semantic_kind not in {TYPED_PROPERTY, TYPED_EVENT_TIME}:
            raise ValueError("typed relation semantic kind 未注册")
        if self.polarity not in {1, 3}:
            raise ValueError("typed relation polarity 只能为 SUPPORT 或 UNKNOWN")
        for label in ("proposition_key", "predicate_key", "source_ref", "scope_key"):
            _key(getattr(self, label), label)
        if type(self.fields) is not tuple or not self.fields:
            raise ValueError("typed relation fields 不能为空")
        if tuple(item[0] for item in self.fields) != tuple(sorted(item[0] for item in self.fields)):
            raise ValueError("typed relation fields 必须排序")
        if self.semantic_kind == TYPED_PROPERTY:
            if self.relation_kind != PROPERTY_RELATION_KIND or tuple(item[0] for item in self.fields) != tuple(range(9, 15)):
                raise ValueError("PROPERTY typed 字段不完整")
            if self.temporal_direction != 0:
                raise ValueError("PROPERTY 不得携带时间方向")
        else:
            if self.relation_kind not in EVENT_TIME_RELATION_KINDS or len(self.fields) != 2:
                raise ValueError("EVENT_TIME typed 字段不完整")
            expected_direction = {
                10: EVENT_TIME_BEFORE,
                11: EVENT_TIME_AFTER,
                12: EVENT_TIME_SAME,
                13: EVENT_TIME_DIRECTION_UNKNOWN,
            }[self.relation_kind]
            if self.temporal_direction != expected_direction:
                raise ValueError("EVENT_TIME predicate 与方向不一致")

    def stable_key(self) -> tuple[int, ...]:
        result = [TYPED_RELATION_VERSION, self.relation_kind, self.semantic_kind,
                  self.polarity, self.temporal_direction]
        for value in (self.proposition_key, self.predicate_key, self.source_ref, self.scope_key):
            result.extend(_pack(value))
        result.append(len(self.fields))
        for role_kind, filler in self.fields:
            result.extend((role_kind, *_pack(filler)))
        return tuple(result)

    def binding_entries(self) -> tuple[BindingEntry, ...]:
        return tuple(BindingEntry(
            role_key=(TYPED_RELATION_NAMESPACE, self.relation_kind, role_kind),
            filler_key=filler, space=SPACE_CORE, scope_key=self.scope_key,
        ) for role_kind, filler in self.fields)

    def evidence_entry(self) -> EvidenceEntry:
        return EvidenceEntry(
            source_ref=self.source_ref, hypothesis_key=self.proposition_key,
            space=SPACE_CORE, polarity=self.polarity, trust=1,
            evidence_key=(TYPED_RELATION_NAMESPACE, *self.stable_key()),
            scope_key=self.scope_key, payload_key=self.stable_key(),
        )

    def trace(self) -> dict[str, object]:
        return {
            "relation_kind": self.relation_kind, "semantic_kind": self.semantic_kind,
            "polarity": self.polarity, "temporal_direction": self.temporal_direction,
            "proposition": list(self.proposition_key), "predicate": list(self.predicate_key),
            "source_ref": list(self.source_ref), "scope": list(self.scope_key),
            "fields": [{"role_kind": role, "filler": list(filler)} for role, filler in self.fields],
            "stable_key": list(self.stable_key()),
        }


def project_typed_relation(fact: object, route: object) -> TypedRelationProjection | None:
    kind = getattr(route, "relation_kind", None)
    if kind == PROPERTY_RELATION_KIND:
        semantic, expected = TYPED_PROPERTY, tuple(range(9, 15))
    elif kind in EVENT_TIME_RELATION_KINDS:
        semantic, expected = TYPED_EVENT_TIME, {10: (23, 24), 11: (25, 26), 12: (27, 28), 13: (29, 30)}[kind]
    else:
        return None
    fields = []
    for binding in fact.bindings:
        role_key = binding.role.stable_key()
        fields.append((role_key[-1], binding.filler.stable_key()))
    fields.sort(key=lambda item: item[0])
    if tuple(item[0] for item in fields) != expected:
        raise ValueError("active typed relation Role 集合不完整")
    if kind == PROPERTY_RELATION_KIND:
        # 复用正式 R-03 六维 claim 合同，而不是把六条普通 RoleBinding
        # 误称为 PROPERTY。
        PropertyClaim(*(ObjectIdentity.from_stable_key(item[1]) for item in fields))
        direction = 0
    else:
        endpoints = tuple(ObjectIdentity.from_stable_key(item[1]) for item in fields)
        if any(item.object_kind not in {OBJECT_EVENT, OBJECT_PROPOSITION}
               for item in endpoints):
            raise ValueError("EVENT_TIME 端点必须是 Event 或 Proposition")
        direction = {
            10: EVENT_TIME_BEFORE,
            11: EVENT_TIME_AFTER,
            12: EVENT_TIME_SAME,
            13: EVENT_TIME_DIRECTION_UNKNOWN,
        }[kind]
    return TypedRelationProjection(
        kind, semantic, fact.proposition.stable_key(), fact.predicate.stable_key(),
        tuple(route.source_ref), tuple(route.scope_key), tuple(fields),
        polarity=1, temporal_direction=direction)


def project_open_typed_relation(candidate: object, route: object) -> TypedRelationProjection | None:
    """保留开放输入自身 span，只复用 schema，绝不借用样例 filler。"""
    kind = getattr(route, "relation_kind", None)
    if kind == PROPERTY_RELATION_KIND:
        semantic, expected, direction = TYPED_PROPERTY, tuple(range(9, 15)), 0
    elif kind in EVENT_TIME_RELATION_KINDS:
        semantic = TYPED_EVENT_TIME
        expected = {10: (23, 24), 11: (25, 26), 12: (27, 28), 13: (29, 30)}[kind]
        direction = {
            10: EVENT_TIME_BEFORE,
            11: EVENT_TIME_AFTER,
            12: EVENT_TIME_SAME,
            13: EVENT_TIME_DIRECTION_UNKNOWN,
        }[kind]
    else:
        return None
    fields = tuple(sorted(
        ((item.role[-1], item.stable_key()) for item in candidate.bindings),
        key=lambda item: item[0]))
    if tuple(item[0] for item in fields) != expected:
        raise ValueError("open typed relation Role 集合不完整")
    return TypedRelationProjection(
        kind, semantic, candidate.stable_key(), candidate.frame.predicate,
        candidate.source_ref, candidate.source_ref, fields,
        polarity=3, temporal_direction=direction)


__all__ = [
    "EVENT_TIME_RELATION_KINDS", "PROPERTY_RELATION_KIND", "TYPED_EVENT_TIME",
    "TYPED_PROPERTY", "TYPED_RELATION_NAMESPACE", "TYPED_RELATION_VERSION",
    "TypedRelationProjection", "project_open_typed_relation", "project_typed_relation",
]
