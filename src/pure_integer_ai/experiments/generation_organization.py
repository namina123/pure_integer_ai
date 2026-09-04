"""阶段 C：ResponsePlan 构造器与 token postcheck（把闭合查询组装为可验证组织）。

从训练后 Core 关系图恢复的闭合回答由多种输入汇聚：QueryState 的多跳路径
（Core/Memory/Companion）、闭合命题、已学框架 literal 间隔与角色表层。本模块
负责把这些纯整数证据组装为 ResponsePlan，并执行 token postcheck：每个必填
槽表层必须按槽位顺序作为连续子串出现在至少一个 realization 候选内；不满足
时 fail-closed（返回 None / 拒绝），绝不从宿主语言规则补洞。

realization 的选择顺序只由 style/carrier 与已学组合权重决定，不选择语言固定
连接词——literal 间隔一律来自训练后 Span 图的 RelationSurfaceFrame。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.response_plan import (
    FILLER_NODE_KINDS,
    ROLE_NODE_KINDS,
    ResponsePlan,
    ResponseRealization,
    ResponseSlot,
    response_act_identity,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface,
    GraphRelationGeneration,
    RelationSurfaceBinding,
)

_RESPONSE_ACT_ANSWER = response_act_identity((1, 1, 1))
_RESPONSE_ACT_CLARIFY = response_act_identity((1, 1, 2))


def answer_response_act() -> ObjectIdentity:
    """返回 answer response act 的一等指令身份。"""
    return _RESPONSE_ACT_ANSWER


def clarify_response_act() -> ObjectIdentity:
    """返回 clarify response act 的一等指令身份。"""
    return _RESPONSE_ACT_CLARIFY


@dataclass(frozen=True, slots=True)
class ClosedGenerationEvidence:
    """一次闭合查询可交给 ResponsePlan 构造器的纯证据视图。"""

    bindings: tuple[RelationSurfaceBinding, ...]
    generation: GraphRelationGeneration
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    source_hash: int
    anchor_surface: str = ""
    connector: ObjectIdentity | None = None


def _role_for_binding(binding: RelationSurfaceBinding) -> ObjectIdentity:
    """返回每个绑定的角色身份；角色必须是一等 Role/StructureConcept。"""
    role = binding.role
    if not isinstance(role, ObjectIdentity):
        raise TypeError("relation binding role 类型错误")
    if role.object_kind not in ROLE_NODE_KINDS:
        raise ValueError("relation binding role 不是一等 Role/StructureConcept")
    return role


def build_response_plan(
        evidence: ClosedGenerationEvidence,
        *,
        required_node_kinds: tuple[int, ...] | None = None,
        ) -> ResponsePlan | None:
    """从闭合查询证据构造 ResponsePlan；token postcheck 失败时返回 None。

    slot_sequence 按角色排序，必填角色来自所有 evidence bindings；每个槽位的
    表层即该角色在 RelationSurfaceBinding 中可核验的表层。realization 候选来自
    evidence.generation（其表层已经由同 predicate/role 的已学框架槽位重填）。
    """
    if not isinstance(evidence, ClosedGenerationEvidence):
        raise TypeError("closed generation evidence 类型错误")
    if required_node_kinds is None:
        required_node_kinds = tuple(sorted(FILLER_NODE_KINDS))
    if not isinstance(required_node_kinds, tuple) or not required_node_kinds or any(
            type(item) is not int or item <= 0 for item in required_node_kinds):
        raise ValueError("required_node_kinds 必须是非空正整数 tuple")
    if not evidence.bindings:
        return None
    generation = evidence.generation
    if not isinstance(generation, GraphRelationGeneration):
        raise TypeError("generation 类型错误")
    # 只接受由 Core 关系图恢复的生成；宿主注入的 connector 为空时仍可走
    # RelationSurfaceFrame 槽位重填路径（阶段 C 目标是图内组合）。
    seen_roles: dict[ObjectIdentity, RelationSurfaceBinding] = {}
    ordered_bindings: list[RelationSurfaceBinding] = []
    for binding in evidence.bindings:
        role = _role_for_binding(binding)
        prior = seen_roles.get(role)
        if prior is not None and prior.surface != binding.surface:
            # 同一 role 出现竞争表层：不能确定性组织回答。
            return None
        if role not in seen_roles:
            seen_roles[role] = binding
            # slot 顺序必须与 realization 的 Span/表层顺序一致（不是按 role
            # 身份排序），否则 token postcheck 会误判跨序槽位缺失。
            ordered_bindings.append(binding)
    slots = []
    claim_refs = []
    for binding in ordered_bindings:
        role = _role_for_binding(binding)
        allowed_kinds = (binding.filler.object_kind,)
        if binding.filler.object_kind not in required_node_kinds:
            allowed_kinds = (binding.filler.object_kind,)
        slots.append(ResponseSlot(
            role,
            binding.filler,
            binding.surface,
            True,
            binding.source_hash,
            allowed_node_kinds=allowed_kinds,
        ))
        if binding.filler not in claim_refs:
            claim_refs.append(binding.filler)
    realization = ResponseRealization(
        generation.surface,
        generation.frame_proposition,
        generation.frame_source_hash,
        generation.slot_count,
    )
    try:
        plan = ResponsePlan(
            _RESPONSE_ACT_ANSWER,
            tuple(claim_refs),
            tuple(slots),
            (realization,),
            ((evidence.source_hash,),),
            scope_and_time=(),
            discourse_links=(),
            style_ref=None,
            carrier_ref=None,
        )
    except (TypeError, ValueError):
        # token postcheck 未通过：无结构证据时不得输出表层。
        return None
    return plan


def plan_from_active_fact(
        fact: ActiveRelationSurface,
        generation: GraphRelationGeneration,
        ) -> ResponsePlan | None:
    """从单个 active Core fact 及其已学生成组装可验证 ResponsePlan。

    fact 的每个 RoleBinding 提供角色、filler 与可核验表层；generation 必须是
    同 predicate/role 框架槽位重填的新表层（由 _generate_surface 执行）。无
    bindings 或 token postcheck 失败返回 None，不补洞。
    """
    if not isinstance(fact, ActiveRelationSurface):
        raise TypeError("active relation fact 类型错误")
    if not isinstance(generation, GraphRelationGeneration):
        raise TypeError("generation 类型错误")
    evidence = ClosedGenerationEvidence(
        fact.bindings,
        generation,
        fact.proposition.stable_key(),
        fact.predicate.stable_key(),
        fact.source_hash,
        anchor_surface=fact.evidence_surface,
        connector=generation.connector,
    )
    return build_response_plan(evidence)


__all__ = [
    "ClosedGenerationEvidence",
    "answer_response_act",
    "build_response_plan",
    "clarify_response_act",
    "plan_from_active_fact",
]
