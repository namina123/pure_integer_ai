"""完整未决指代历史只供恢复候选，不能扩大实际读取或Plan采用权限。"""
from pure_integer_ai.experiments.memory_provenance_inputs import consumed_response_contexts
from pure_integer_ai.experiments.trained_reference_resolution import (
    CrossTurnReferenceResolution, REFERENCE_OPEN, REFERENCE_CONFLICT,
)
from pure_integer_ai.experiments.trained_response_context import (
    DeliveredFactualResponseContext, DeliveredGenericResponseContext,
)
from pure_integer_ai.experiments.trained_response_occurrence import ReferenceCandidateSet


def qualify_reference_history(contexts, complete, *, trace, evidence, bindings,
                              candidate_sets, resolutions, nonfactual_plan,
                              adopted_hypothesis_keys, generic_adoption, error):
    """分账须与原实际消费精确相等；额外历史仅准入零事实、全未选择澄清。

    调用方已从物理proof/Episode独立恢复完整context。本函数不授予额外
    O/H/E读取、角色Evidence或Plan引用，既有消费者继续只接收contexts。
    """
    if complete is None:
        return contexts
    if (type(complete) is not tuple or len(complete) > 32
            or any(not isinstance(item, (DeliveredFactualResponseContext, DeliveredGenericResponseContext))
                   for item in complete)
            or len({item.stable_key() for item in complete}) != len(complete)):
        raise error('complete reference history requires bounded unique cold contexts')
    role_keys = {item.evidence_key for item in evidence if item.space == 2}
    if consumed_response_contexts(complete, trace, role_keys) != contexts:
        raise error('reference history differs from the exact actually consumed contexts')
    if complete == contexts:
        return complete
    plan = trace.trace.get('response_plan')
    if (not nonfactual_plan or trace.trace.get('termination') != 3
            or not isinstance(plan, dict)
            or any(plan.get(key) for key in ('claim_refs', 'claim_evidence_refs', 'claim_slot_ranges', 'event_refs'))
            or generic_adoption.get('visible_graph_slot_count') != 0
            or not adopted_hypothesis_keys
            or type(candidate_sets) is not tuple or not candidate_sets or len(candidate_sets) > 64
            or any(not isinstance(item, ReferenceCandidateSet) for item in candidate_sets)
            or type(resolutions) is not tuple or len(resolutions) != len(candidate_sets)
            or any(not isinstance(item, CrossTurnReferenceResolution) for item in resolutions)
            or {item.candidate_set for item in resolutions} != set(candidate_sets)
            or len({item.candidate_set for item in resolutions}) != len(resolutions)
            or any(not item.requires_resolution or item.status not in (REFERENCE_OPEN, REFERENCE_CONFLICT)
                   or item.winner_target_key or item.adoption_support_key for item in resolutions)
            or any(item.reference.hypothesis.stable_key() in adopted_hypothesis_keys for item in candidate_sets)):
        raise error('unconsumed reference history requires complete zero-claim unselected clarification')
    from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack
    if (any(item.evidence_key[:2] == (91599, 7) for item in evidence)
            or any(_unpack(key, 1, 3, 1)[0] == (91599, 6) for key in bindings)):
        raise error('unconsumed reference history cannot authorize target adoption')
    return complete
