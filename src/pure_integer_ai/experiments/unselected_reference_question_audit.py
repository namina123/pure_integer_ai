"""核验未选择的受训指代备选保持开放，不能冒充已采用问式或事实。"""
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError, _unpack
from pure_integer_ai.experiments.query_pending_reference import pending_reference_relation_keys
from pure_integer_ai.experiments.trained_reference_resolution import REFERENCE_OPEN, REFERENCE_CONFLICT


def verify_unselected_reference_questions(trace, prepared, resolutions, *, reference_bridge):
    """严格核对零事实、零采用和完整输入不变；调用前须已独立恢复全部指代。

    候选只是待解义务，必须具有完整训练来源和当前开放解析；本合同不会
    为候选补造QueryState语法证明，也不会替代后续O/H/E及交付核验。
    """
    bindings = prepared.relation_question_bindings
    syntax = lambda item: (item.frame_key, item.operation_key, item.slot_span_refs, item.answer_role_key)
    if (reference_bridge is None or len(bindings) < 2
            or not all(item.answer_role_key for item in bindings)
            or any(syntax(item) != syntax(bindings[0]) for item in bindings)
            or prepared.discourse_compositions or prepared.discourse_gaps):
        raise QueryMemoryOheEvidenceError('unselected reference alternatives lack one complete question syntax')
    pending = pending_reference_relation_keys(prepared, reference_bridge.core_fact_index,
                                              reference_bridge.relation_question_frames)
    if pending != frozenset(item.relation_key for item in bindings) or len(pending) != len(bindings):
        raise QueryMemoryOheEvidenceError('unselected question lacks complete trained pending-reference evidence')
    if (len(resolutions) != len(bindings)
            or len({item.candidate_set.stable_key() for item in resolutions}) != len(resolutions)
            or any(not item.requires_resolution or item.status not in (REFERENCE_OPEN, REFERENCE_CONFLICT)
                   or item.winner_target_key or item.adoption_support_key for item in resolutions)):
        raise QueryMemoryOheEvidenceError('unselected question has missing, competing or adopted reference decisions')
    value = trace.trace
    plan = value.get('response_plan')
    if not isinstance(plan, dict) or plan.get('claim_refs') or plan.get('claim_slot_ranges'):
        raise QueryMemoryOheEvidenceError('unselected reference question cannot authorize any factual output')
    for ref in value['state_key']['evidence']:
        key = trace.resolve_ref(ref)
        if _unpack(key, 3, 6)[2][:2] == (91599, 7):
            raise QueryMemoryOheEvidenceError('unselected reference question contains a target adoption')
    for ref in value['state_key']['bindings']:
        key = trace.resolve_ref(ref)
        if _unpack(key, 1, 3, 1)[0] == (91599, 6):
            raise QueryMemoryOheEvidenceError('unselected reference question contains an adoption binding')
    final = value['input_structure']

    def complete_key(item):
        """只解析完整整数键或当前trace的明确引用，不接受摘要替身。"""
        if isinstance(item, dict) and set(item) == {'integer_key_ref'}:
            return trace.resolve_ref(item['integer_key_ref'])
        if isinstance(item, (tuple, list)) and all(type(value) is int and value >= 0 for value in item):
            return tuple(item)
        raise QueryMemoryOheEvidenceError('unselected input requires complete integer keys')

    if (complete_key(final['stable_key']) != prepared.stable_key()
            or tuple(complete_key(item) for item in final['relation_question_bindings'])
            != tuple(item.stable_key() for item in bindings)):
        raise QueryMemoryOheEvidenceError('unselected question changed its full input or candidate bindings')
    return dict(initial_question_proofs=0, unselected_question_candidates=len(bindings),
                target_adoption_count=0, full_input_transition_matches=1,
                query_call_count=0, generation_call_count=0)
