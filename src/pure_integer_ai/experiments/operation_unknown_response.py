"""完整请求缺关系连接时，只在原QueryState中交付有执行证明的UNKNOWN行为。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.integer_operation_execution import UNKNOWN
from pure_integer_ai.cognition.shared.query_driver import unvisited_frontier
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_PENDING, BindingEntry, EvidenceEntry,
    TERMINATION_BUDGET_EXHAUSTED, TERMINATION_CLARIFY_CONFLICT,
    TERMINATION_CYCLE_GUARD, TERMINATION_DIALOGUE_UNKNOWN,
)
from pure_integer_ai.experiments.context_only_plan_slots import language_only_plan_slots
from pure_integer_ai.experiments.operation_output_query import consumed_current_request_key
from pure_integer_ai.experiments.operation_program_query import OPERATION_EXECUTION_ROLE


def unknown_operation_permission(execution):
    """UNKNOWN只声明该完整程序缺证据，不改变任何前提事实的立场。"""
    return EvidenceEntry(execution.witness.source.stable_key(), execution.witness.program.operation_key,
        space=1, polarity=3, evidence_key=(92230, *execution.query_key),
        scope_key=execution.query_key, payload_key=execution.stable_key())


def unknown_operation_context_allowed(execution, state, structure, hypotheses, observation_key):
    """完整执行、当前请求O/H/E和原预算全部通过才授权无对象槽生成。"""
    if (execution is None or execution.status != UNKNOWN or execution.outputs
            or execution.query_key != state.query_key or structure.source_ref != state.query_key
            or len(structure.operation_request_bindings) != 1
            or tuple(state.active_spaces) != (1, 2, 3) or state.conflict_open
            or any(item.conflict_kept for item in state.bindings)
            or any(root.status == ROOT_PENDING for root in state.roots)
            or unvisited_frontier(state.frontier, state.visited)
            or state.termination in (TERMINATION_BUDGET_EXHAUSTED,
                TERMINATION_CYCLE_GUARD, TERMINATION_CLARIFY_CONFLICT)
            or state.node_count > state.budget.max_nodes
            or state.edge_count > state.budget.max_edges
            or state.read_count > state.budget.max_reads or state.depth > state.budget.max_depth):
        return False
    request, = structure.operation_request_bindings
    if (request.source_ref != state.query_key
            or request.operation_key != execution.witness.program.operation_key
            or BindingEntry(OPERATION_EXECUTION_ROLE, execution.stable_key(),
                space=1, scope_key=state.query_key) not in state.bindings
            or unknown_operation_permission(execution) not in state.evidence):
        return False
    return request.stable_key() == consumed_current_request_key(
        execution, state, hypotheses, observation_key)


def generate_unknown_operation_responses(runtime, execution, state, structure, hypotheses, nodes,
                                         observation, *, request_hypotheses):
    """复用同次已展开connector与原六层生成；READY或缺证明时不尝试回应。"""
    if observation is None or not unknown_operation_context_allowed(
            execution, state, structure, request_hypotheses, observation.stable_key()):
        return ()
    from pure_integer_ai.experiments.trained_open_response_runtime import generate_generic_responses
    responses = generate_generic_responses(runtime, state, hypotheses, nodes, observation,
                                           structure, context_only=True)
    if any(item.graph_surfaces or item.response_plan.claim_refs
           or item.response_plan.claim_evidence_refs
           or not language_only_plan_slots(item.response_plan.to_dict())
           for item in responses):
        raise ValueError("unknown operation response must have only trained language slots")
    return responses


def unknown_operation_plan_allowed(execution, state, plan, generation):
    """最终交付仍保留完整执行权限，不能用普通generic候选绕过程序失败。"""
    return bool(execution is not None and execution.status == UNKNOWN and not execution.outputs
        and state.query_key == execution.query_key
        and state.termination == TERMINATION_DIALOGUE_UNKNOWN
        and state.evidence_closed == 0 and state.generation_ready == 1 and not state.conflict_open
        and plan is not None and not plan.claim_refs and not plan.claim_evidence_refs
        and language_only_plan_slots(plan.to_dict())
        and unknown_operation_permission(execution).stable_key() in plan.evidence_refs
        and generation is not None and generation.get('kind') == 3
        and tuple(generation.get('surface_values', ())) == tuple(map(ord, plan.surface())))


def verify_unknown_operation_delivery(execution, value, trace):
    """冷核原执行证明和零事实交付形态；完整选择与Memory由原审计链另行核验。"""
    from pure_integer_ai.experiments.pending_composition_call_audit import _decode_input
    plan = _decode_input(value.get('response_plan'), trace)
    generation = _decode_input(value.get('generation'), trace)
    scalars = value['state_key']['scalars']
    if (execution.status != UNKNOWN or execution.outputs
            or value['termination'] != TERMINATION_DIALOGUE_UNKNOWN
            or scalars[7] != 0 or scalars[8] != 0 or scalars[15] != 1
            or not isinstance(plan, dict) or plan.get('claim_refs') or plan.get('claim_evidence_refs')
            or not language_only_plan_slots(plan)
            or list(unknown_operation_permission(execution).stable_key()) not in plan.get('evidence_refs', ())
            or not isinstance(generation, dict) or generation.get('kind') != 3
            or generation.get('unknown_operation_execution') != list(execution.stable_key())
            or not value.get('response_surface')
            or tuple(generation.get('surface_values', ())) != tuple(map(ord, value['response_surface']))):
        raise ValueError('unknown operation lacks its exact consumed execution and language-only delivery')
    return {'unknown_operation_status': 'MATCHED_CURRENT_EXECUTION_AND_LANGUAGE_ONLY_DELIVERY',
            'factual_claim_count': 0, 'query_generation_replays': 0}
