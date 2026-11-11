"""查询仲裁使用的来源无关声明键；不改写或合并图内对象。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_BINDER, OBJECT_CONTEXT_SCOPE, OBJECT_ENTITY, OBJECT_EVENT,
    OBJECT_PROPOSITION, OBJECT_SET_EXPR, ObjectIdentity,
)
from pure_integer_ai.cognition.shared.semantic_object import semantic_source


def semantic_candidate_identity_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """仅剥离已核验声明的物理来源，保留类型、owner、版本和完整局部声明。

    比较键只用于本轮候选分组；原对象稳定键与来源仍供证据和生成使用。
    两类编码带不同标签，非声明对象或非法声明绝不与规范化声明碰撞。
    """
    try:
        identity = ObjectIdentity.from_stable_key(value)
        if identity.object_kind not in {
                OBJECT_ENTITY, OBJECT_EVENT, OBJECT_PROPOSITION,
                OBJECT_SET_EXPR, OBJECT_BINDER, OBJECT_CONTEXT_SCOPE}:
            return (0, *value)
        semantic_source(identity)
    except (TypeError, ValueError):
        return (0, *value)
    return (
        1, identity.object_kind, *identity.owner.stable_key(),
        *identity.versions.stable_key(), identity.components[0],
        *identity.components[12:],
    )
