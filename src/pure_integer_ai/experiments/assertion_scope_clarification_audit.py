"""独立冷核验完整作用域回应，保留实际 Core 历史而不生产新回答。"""
from dataclasses import replace

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.query_state import (
    BindingEntry, EvidenceEntry, ROOT_PENDING, SPACE_CORE, SPACE_DIALOGUE,
)
from pure_integer_ai.cognition.understanding.query_assertion_scope import (
    SCOPE_QUERY_PROOF, match_assertion_scope_frames,
)
from pure_integer_ai.experiments.trained_query_evidence import core_query_evidence


def verify_assertion_scope_clarification(trace, prepared, frames, core_runtime, *,
                                         authorized_input, roots, evidence, bindings, error):
    """从物理课程、全输入及冷证据恢复资格；不把用户立场当世界真值。"""
    from pure_integer_ai.experiments.pending_composition_call_audit import _decode_input

    scopes = match_assertion_scope_frames(frames, prepared)
    state = trace.trace['state_key']['scalars']
    if (len(scopes) != 1 or scopes != prepared.assertion_scope_bindings
            or authorized_input is None or core_runtime is None
            or state[7:9] != [0, 1] or state[15:] != [1, 2]
            or trace.trace.get('termination') != 2):
        raise error('scope clarification lacks a unique cold mounted scope or nonfactual state')
    scope = scopes[0]
    relations = tuple(item for item in prepared.relation_candidates
                      if item.stable_key() == scope.relation_key)
    if len(relations) != 1:
        raise error('scope clarification lacks its exact current relation')
    relation = relations[0]
    # 独立恢复视图合同，不由最终 claim 或已保存的选中键倒选候选。
    spans = set(relation.span_refs)
    expected_input = replace(prepared, relation_candidates=relations,
        semantic_candidates=tuple(sorted((item for item in prepared.semantic_candidates
            if item.proposition_key == relation.proposition_key and item.span_ref in spans),
            key=lambda item: item.stable_key())),
        carrier_candidates=tuple(sorted((item for item in prepared.carrier_candidates
            if item.structure_key == relation.structure_key), key=lambda item: item.stable_key())),
        open_relations=(), discourse_compositions=(), discourse_gaps=())
    if (authorized_input != expected_input
            or _decode_input(trace.trace['input_structure'], trace) != expected_input.integer_trace()):
        raise error('scope clarification input differs from the complete authorized cold projection')
    query = trace.resolve_ref(trace.trace['query_ref'])
    proof = scope.stable_key()
    expected_binding = BindingEntry((SCOPE_QUERY_PROOF,), proof,
        space=SPACE_DIALOGUE, scope_key=query).stable_key()
    expected_evidence = EvidenceEntry(scope.source_ref, proof,
        space=SPACE_DIALOGUE, polarity=3, trust=1, evidence_key=proof,
        payload_key=scope.frame_key, scope_key=query)
    scoped_bindings = {key for key in bindings
                       if key[1:3] == (1, SCOPE_QUERY_PROOF)}
    scoped_evidence = tuple(item for item in evidence
        if item.hypothesis_key[:1] == proof[:1] or item.evidence_key[:1] == proof[:1])
    if (scoped_bindings != {expected_binding} or scoped_evidence != (expected_evidence,)
            or any(key[-1] for key in bindings)
            or any(root.status == ROOT_PENDING for root in roots)):
        raise error('scope clarification lost its exact QueryState proof or retained a conflict')
    truth = tuple(item for item in evidence if item.space == SPACE_CORE and item.polarity in (1, 2))
    for key in sorted({item.hypothesis_key for item in truth}):
        try:
            proposition = ObjectIdentity.from_stable_key(key)
            history = core_runtime.evidence_history(proposition)
            cold = core_query_evidence(proposition, history,
                core_runtime.context.scoped_identity_store.registry)
        except (ValueError, KeyError) as cause:
            raise error('scope clarification Core evidence cannot be cold restored') from cause
        consumed = tuple(item for item in truth if item.hypothesis_key == key)
        if (set(consumed) != {item for item in cold if item.polarity in (1, 2)}
                or any(item not in evidence for item in cold)):
            raise error('scope clarification Core evidence differs from the complete cold history')
    return dict(status='MATCHED_INDEPENDENT_ASSERTION_SCOPE_AND_NONFACTUAL_CLARIFICATION',
        retained_input_relation_count=1, scope_mode=scope.mode,
        mounted_scope_verified=1, scope_query_proof_verified=1,
        core_truth_evidence_count=len(truth),
        core_hop_count=sum(item.space == SPACE_CORE for item in trace.hops()),
        conflict_open=1, factual_evidence_closed=0)
