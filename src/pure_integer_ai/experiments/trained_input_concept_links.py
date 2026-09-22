"""把训练后稳定 PURE_ALIAS 本体接到通用输入关联，不扩充篇章指代规则。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_concept_links import TrainedConceptLink
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
    authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_PURE_ALIAS,
    ROLE_ALIAS_LEFT,
    ROLE_ALIAS_RIGHT,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime


def restore_input_concept_links(
        runtime: TrainedRelationGraphRuntime,
        ) -> tuple[TrainedConceptLink, ...]:
    """只恢复已通过 W-06 firewall 的稳定同类别名关系，双向边仍保留原命题。

    REFERS 没有传递许可，occurrence-bound 引用属于另外的篇章协议，均不能
    借本适配器扩大为全局别名。具体词形和节点完全来自训练后的 Span/RoleBinding。
    """
    predicate = authored_relation_identity(RELATION_PURE_ALIAS)
    left_role = authored_relation_role_identity(ROLE_ALIAS_LEFT)
    right_role = authored_relation_role_identity(ROLE_ALIAS_RIGHT)
    links = []
    for fact in runtime.active_surface_facts():
        if fact.predicate != predicate:
            continue
        bindings = {item.role: item.filler for item in fact.bindings}
        if len(fact.bindings) != 2 or set(bindings) != {left_role, right_role}:
            raise ValueError("稳定 PURE_ALIAS 的角色结构不闭合")
        left, right = bindings[left_role], bindings[right_role]
        if left.object_kind != right.object_kind:
            raise ValueError("稳定 PURE_ALIAS 不得合并不同对象类型")
        source = runtime.generation_input(fact.proposition)
        for origin, target in ((left, right), (right, left)):
            links.append(TrainedConceptLink(
                origin.stable_key(), target.stable_key(), fact.proposition.stable_key(),
                predicate.stable_key(), source.proposition.definition.source.stable_key(),
                source.hypothesis.scope.stable_key()))
    return tuple(sorted(set(links), key=lambda item: item.stable_key()))


__all__ = ["restore_input_concept_links"]
