"""只读核对未落地请求的原始查询证明，不重新理解、查询或生成答案。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_operation_request_scope import PENDING_REQUEST_SCOPE
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.operation_request_scope_query import pending_request_query_records
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack
from pure_integer_ai.experiments.trained_input_clarification import INPUT_CLARIFICATION_VERSION
from pure_integer_ai.experiments.context_only_plan_slots import language_only_plan_slots


def audit_saved_pending_request_scope(value, report, prepared, frames):
    """完整比较当前重建的来源/变量/区间证明，并拒绝残留事实交付或对象槽。"""
    records = pending_request_query_records(frames, prepared)
    if not records:
        return {"pending_request_scope_count": 0}
    trace = GraphQueryTraceEvidence(value)
    bindings = tuple(trace.resolve_ref(ref) for ref in value['state_key']['bindings'])
    evidence = tuple(trace.resolve_ref(ref) for ref in value['state_key']['evidence'])
    actual_bindings = set()
    denied = []
    for key in bindings:
        role, filler, _scope = _unpack(key, 1, 3, tail=1)
        if role == (PENDING_REQUEST_SCOPE, 1):
            actual_bindings.add(key)
        if role == (INPUT_CLARIFICATION_VERSION, 1):
            denied.append(filler)
    actual_evidence = set()
    for key in evidence:
        parts = _unpack(key, 3, 6)
        if key[0] == 1 and key[1] in (1, 2):
            raise ValueError('pending request consumed Core truth instead of retaining its input failure')
        if parts[4][:1] == (PENDING_REQUEST_SCOPE,):
            actual_evidence.add(key)
    if (len(bindings) != len(set(bindings)) or len(evidence) != len(set(evidence))
            or actual_bindings != {binding.stable_key() for binding, _ in records}
            or actual_evidence != {entry.stable_key() for _, entry in records}
            or trace.resolve_ref(value['state_key']['query_ref']) != prepared.source_ref):
        raise ValueError('pending request differs from its exact current syntax/projections/source proof')
    scalars = value['state_key']['scalars']
    plan, generation = value.get('response_plan'), value.get('generation')
    if (value.get('termination') != 3 or report.get('termination') != 3
            or scalars[6] < 1 or scalars[7] != 0 or scalars[15] != 1
            or report.get('delivery_claim_count') != 0 or not plan or not generation
            or plan.get('claim_refs') or plan.get('claim_evidence_refs') or not language_only_plan_slots(plan)
            or generation.get('kind') != 3 or any(hop.space == 1 for hop in trace.hops())):
        raise ValueError('pending request lacks its original generated zero-claim missing-binding response')
    return {"pending_request_scope_count": len(records), "complete_scope_proofs_matched": len(records),
            "denied_original_roots": len(denied), "core_truth_consumed": 0,
            "factual_claim_count": 0, "query_generation_replays": 0}
