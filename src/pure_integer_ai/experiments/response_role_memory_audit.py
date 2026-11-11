"""Check consumed prior-delivery roles without elevating UNKNOWN history to truth."""
from __future__ import annotations

from pure_integer_ai.cognition.shared.query_state import ROOT_EXPANDED
from pure_integer_ai.experiments.trained_response_context import (
    DeliveredFactualResponseContext, DeliveredGenericResponseContext,
)


def verify_response_role_memory(contexts, *, evidence, bindings, roots,
                                plan_keys, plan_refs, has_plan, error,
                                adopted_hypothesis_keys=None):
    """核对冷图中逐项恢复的事实及通用历史；仅查询的角色不扩大实际采用分支。"""
    if (type(contexts) is not tuple or len(contexts) > 32
            or any(not isinstance(item, (DeliveredFactualResponseContext, DeliveredGenericResponseContext))
                   for item in contexts)
            or len({item.stable_key() for item in contexts}) != len(contexts)):
        raise error("response role audit requires bounded cold factual contexts")
    roles = {role.stable_key(): (context, role) for context in contexts for role in context.role_occurrences}
    if len(roles) != sum(len(item.role_occurrences) for item in contexts):
        raise error("cold response role identity is duplicated")
    members = tuple(item for item in evidence if item.space == 2 and item.evidence_key in roles)
    for entry in members:
        context, role = roles[entry.evidence_key]
        if (entry != role.query_evidence() or role.query_binding().stable_key() not in bindings
                or not any(root.owner_space == 2 and root.status == ROOT_EXPANDED
                           and root.root_key == role.stable_key() and root.source_ref == role.source_ref
                           and root.scope_key == role.scope_key for root in roots)
                or not all(item.stable_key() in bindings for item in context.query_bindings())):
            raise error("response role Evidence/binding/expanded root/context differs from graph")
        adopted = adopted_hypothesis_keys is None or role.hypothesis.stable_key() in adopted_hypothesis_keys
        if has_plan and (entry.stable_key() not in plan_keys
                         or adopted and not {role.observation, role.hypothesis, *context.memory_refs()} <= plan_refs):
            raise error("ResponsePlan omits consumed prior response role Evidence/Memory")
    return members
