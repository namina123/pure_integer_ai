"""独立核验真实端点竞争产生的零事实澄清，不查询或生成新结果。"""
from pure_integer_ai.cognition.shared.query_state import ROOT_EMPTY, ROOT_PENDING, SPACE_CORE
from pure_integer_ai.cognition.understanding.query_relation_questions import relation_question_syntax_predicates
from pure_integer_ai.experiments.query_relation_arbitration import input_relation_competition
from pure_integer_ai.experiments.trained_input_clarification import INPUT_CLARIFICATION_VERSION
from pure_integer_ai.experiments.input_projection_conflict import projection_conflict_keys_match


def verify_input_conflict_clarification(trace, prepared, facts, question_frames, *,
                                      roots, evidence, bindings, error,
                                      assertion_scope_frames=(), core_runtime=None,
                                      authorized_input=None):
    """从冷训练图证明竞争和当前输入，保留冲突并拒绝任何事实消费资格。"""
    from pure_integer_ai.experiments.pending_composition_call_audit import _decode_input

    if prepared is not None and prepared.assertion_scope_bindings:
        from pure_integer_ai.experiments.assertion_scope_clarification_audit import verify_assertion_scope_clarification
        return verify_assertion_scope_clarification(trace, prepared, assertion_scope_frames,
            core_runtime, authorized_input=authorized_input, roots=roots, evidence=evidence,
            bindings=bindings, error=error)
    state = trace.trace['state_key']['scalars']
    if (prepared is None or facts is None
            or state[7] != 0 or state[8] != 1 or state[15] != 1 or state[16] != 2
            or trace.trace.get('termination') != 2):
        raise error('conflict clarification lacks its independent input or retained conflict state')
    if _decode_input(trace.trace['input_structure'], trace) != prepared.integer_trace():
        raise error('conflict clarification input differs from the complete cold projection')
    predicates = relation_question_syntax_predicates(question_frames, prepared)
    projection_conflict = projection_conflict_keys_match(prepared, prepared.source_ref,
        bindings, tuple(item.stable_key() for item in evidence))
    if projection_conflict and trace.resolve_ref(trace.trace['query_ref']) != prepared.source_ref:
        raise error('projection conflict query differs from the independent current input')
    if not projection_conflict and not input_relation_competition(
            prepared, facts, question_predicate_keys=frozenset(predicates) if predicates else None,
            question_frames=question_frames):
        raise error('conflict clarification lacks competing complete Core endpoint identities')
    core_roots = tuple(root for root in roots if root.owner_space == SPACE_CORE)
    if (len(core_roots) != 1
            or core_roots[0].root_key != (INPUT_CLARIFICATION_VERSION, 0)
            or core_roots[0].status != ROOT_EMPTY
            or any(root.status == ROOT_PENDING for root in roots)
            or any(item.space == SPACE_CORE for item in trace.hops())
            or any(item.space == SPACE_CORE and item.polarity in (1, 2) for item in evidence)
            or (any(key[-1] != 0 for key in bindings) and not projection_conflict)):
        raise error('conflict clarification consumed Core truth, conflicting bindings or pending roots')
    return dict(status=('MATCHED_RETAINED_INPUT_PROJECTION_CONFLICT_AND_NONFACTUAL_CLARIFICATION'
        if projection_conflict else 'MATCHED_INDEPENDENT_INPUT_COMPETITION_AND_NONFACTUAL_CLARIFICATION'),
        retained_input_relation_count=len(prepared.relation_candidates),
        conflict_open=1, factual_evidence_closed=0, core_hop_count=0)
