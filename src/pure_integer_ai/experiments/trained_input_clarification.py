"""完整输入未获事实资格时，在原查询中消费 UNKNOWN 图生成澄清回应。"""
from __future__ import annotations

from pure_integer_ai.experiments.input_projection_conflict import retained_projection_conflict_allowed

from pure_integer_ai.cognition.shared.query_driver import run_query, unvisited_frontier
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EMPTY, ROOT_PENDING, SPACE_CORE, SPACE_DIALOGUE, SPACE_MEMORY,
    TERMINATION_BUDGET_EXHAUSTED, TERMINATION_CLARIFY_CONFLICT,
    TERMINATION_CLARIFY_MISSING_BINDING,
    TERMINATION_CYCLE_GUARD, TERMINATION_OPEN,
    BindingEntry, QueryRoot, QueryState,
)


INPUT_CLARIFICATION_VERSION = 92371


def clarification_failure_termination(state: QueryState) -> int:
    """外层保留真实竞争、预算及循环原因，其它未绑定终止映射为缺槽。"""
    return (state.termination if state.termination in (
        TERMINATION_CLARIFY_CONFLICT, TERMINATION_BUDGET_EXHAUSTED,
        TERMINATION_CYCLE_GUARD) else TERMINATION_CLARIFY_MISSING_BINDING)


def unanchored_reference_question(parts, candidate_sets) -> bool:
    """只识别已经核验的唯一指代问式缺历史，不解释词面或猜测目标。"""
    return bool(parts is not None and parts[0].is_reference_slot
                and len(parts[4]) == 1 and not candidate_sets)


def unresolved_role_context_allowed(state: QueryState, input_structure=None) -> bool:
    """仅在同次 frontier 已完成且没有真值、冲突或超预算时允许 UNKNOWN 表达。

    此资格不解析动态角色，不关闭事实证据，也不授权可见图对象槽；当前来源、
    Observation 和 UNKNOWN 假设仍须由原六层生成核验。
    """
    closed = bool(
        tuple(state.active_spaces) == (SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE)
        and not state.conflict_open
        and not any(item.conflict_kept for item in state.bindings)
        and not any(root.status == ROOT_PENDING for root in state.roots)
        and not unvisited_frontier(state.frontier, state.visited)
        and state.termination not in (TERMINATION_BUDGET_EXHAUSTED, TERMINATION_CYCLE_GUARD)
        and state.node_count <= state.budget.max_nodes
        and state.edge_count <= state.budget.max_edges
        and state.read_count <= state.budget.max_reads
        and state.depth <= state.budget.max_depth)
    if not closed:
        return False
    from pure_integer_ai.cognition.understanding.query_open_role_state import open_role_query_nodes
    syntax_evidence = {
        node.query_evidence().stable_key()
        for node in open_role_query_nodes(
            () if input_structure is None else input_structure.open_relations)
        if node.space == SPACE_CORE
    }
    return all(
        item.polarity == 1 and item.stable_key() in syntax_evidence
        for item in state.evidence
        if item.space == SPACE_CORE and item.polarity in (1, 2))


def clarification_query_state(state: QueryState, frontier, hypotheses, nodes, *, input_role_occurrences=(), input_structure=None):
    """只授权当前观察和训练 connector，完整记录被拒绝的原根。

    保留原查询身份、输入投影、预算和来源；不把未展开的事实根标成已展开。
    Observation 的实际 O/H/E 仍由原扩展器读取，不能用摘要合成 Evidence。
    """
    if (not hypotheses or not nodes or state.read_count or state.visited
            or state.conflict_open
            or (any(item.conflict_kept for item in state.bindings)
                and not retained_projection_conflict_allowed(state, input_structure))
            or any(item.space == SPACE_CORE and item.polarity in (1, 2)
                   for item in state.evidence)):
        return None
    observation_keys = {item.observation_key for item in hypotheses}
    if len(observation_keys) != 1:
        return None
    admitted = {(SPACE_MEMORY, key) for key in observation_keys}
    if any(item.observation.stable_key() not in observation_keys for item in input_role_occurrences):
        return None
    admitted.update((SPACE_MEMORY, item.stable_key()) for item in input_role_occurrences)
    admitted.update((SPACE_DIALOGUE, node.key) for node in nodes)
    roots = tuple(root for root in state.roots
                  if (root.owner_space, root.root_key) in admitted)
    if not admitted <= {(root.owner_space, root.root_key) for root in roots}:
        return None
    core_weight = max((root.priority_weight for root in state.roots
                       if root.owner_space == SPACE_CORE), default=1)
    denied = tuple(BindingEntry(
        (INPUT_CLARIFICATION_VERSION, 1), root.stable_key(),
        space=SPACE_CORE, scope_key=state.query_key)
        for root in state.roots
        if (root.owner_space, root.root_key) not in admitted)
    return state.with_(
        roots=(*roots, QueryRoot(
            SPACE_CORE, (INPUT_CLARIFICATION_VERSION, 0), ROOT_EMPTY, core_weight)),
        anchors=tuple(item for item in state.anchors
                      if (item.space, item.ref_key) in admitted),
        frontier=tuple(item for item in frontier
                       if (item.owner_space, item.root_key) in admitted),
        bindings=(*state.bindings, *denied),
        required_slots_open=max(1, state.required_slots_open),
        evidence_closed=0, generation_ready=0, termination=TERMINATION_OPEN,
    ).canonical()


def run_input_clarification(state, frontier, hypotheses, nodes, *, expander, tracer,
                            input_role_occurrences=(), input_relation_conflict=False):
    """在唯一生产查询内运行受限 frontier，保持事实失败及所有原预算。

    生成器和 Memory 读取均复用当前扩展器；没有训练后生成结果时不补句子。
    搜索前派生的端点竞争只改变最终失败状态，不能解除已有冲突或真值门。
    """
    if type(input_relation_conflict) is not bool:
        raise TypeError("input_relation_conflict 必须是严格布尔值")
    structure = getattr(expander, "input_structure", None)
    projection_conflict = retained_projection_conflict_allowed(state, structure)
    admitted = clarification_query_state(state, frontier, hypotheses, nodes,
        input_role_occurrences=input_role_occurrences, input_structure=structure)
    if admitted is None:
        return state, None
    tracer.record(admitted, TERMINATION_OPEN)
    result = run_query(admitted, expander=expander, tracer=tracer)
    if result.termination in (TERMINATION_BUDGET_EXHAUSTED, TERMINATION_CYCLE_GUARD):
        return result, None
    if (unvisited_frontier(result.frontier, result.visited)
            or any(root.status == ROOT_PENDING for root in result.roots)):
        return result, None
    if any(item.space == SPACE_CORE and item.polarity in (1, 2)
           for item in result.evidence) or any(hop.space == SPACE_CORE for hop in expander.hops):
        raise ValueError("未绑定澄清不得消费事实根或获得 Core 真值证据")
    if ((result.conflict_open or any(item.conflict_kept for item in result.bindings))
            and not (projection_conflict and retained_projection_conflict_allowed(
                result, structure, restricted=True))):
        return result, None
    generated = tuple(expander.generic_response_generations)
    generic = generated[0] if len(generated) == 1 else None
    if generic is not None and (
            generic.response_plan.claim_refs
            or generic.response_plan.claim_evidence_refs
            or generic.graph_surfaces):
        raise ValueError("未绑定澄清只能消费 UNKNOWN 语境，不能交付事实或对象表面")
    retained_conflict = input_relation_conflict or projection_conflict
    termination = (TERMINATION_CLARIFY_CONFLICT if retained_conflict
                   else TERMINATION_CLARIFY_MISSING_BINDING)
    result = result.with_(
        termination=termination,
        conflict_open=int(retained_conflict),
        required_slots_open=max(1, result.required_slots_open),
        evidence_closed=0, generation_ready=int(generic is not None),
    ).canonical()
    tracer.record(result, termination)
    return result, generic


def clarification_generation_payload(generic, connector_key):
    """把真实六层生成转换为已有交付协议，不构造语言表面或事实槽。"""
    plan = generic.response_plan
    return {
        "kind": 3,
        "response_act": list(plan.response_act.stable_key()),
        "connector": list(connector_key),
        "representations": [list(item.stable_key()) for item in generic.representations],
        "trace": list(generic.trace),
        "surface_values": list(generic.rendered.units),
    }
