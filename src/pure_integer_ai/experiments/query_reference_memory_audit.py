"""逐项核对冷恢复的指代候选与 H-00/H-04 消费，不把结构历史升为内容真值。"""
from pure_integer_ai.cognition.shared.query_state import ROOT_EXPANDED
from pure_integer_ai.experiments.trained_reference_resolution import CrossTurnReferenceResolution
from pure_integer_ai.experiments.trained_response_occurrence import ReferenceCandidateSet


def verify_reference_memory(candidate_sets, resolutions, *, input_roles, contexts, evidence,
                            bindings, roots, plan_keys, plan_refs, has_plan, error,
                            adopted_hypothesis_keys=None):
    """调用者先从物理只读图独立恢复成员；本函数核验同次状态及 Plan 的完整引用。"""
    for items, kind, label in ((candidate_sets, ReferenceCandidateSet, 'candidate set'),
                               (resolutions, CrossTurnReferenceResolution, 'resolution')):
        if (type(items) is not tuple or len(items) > 64
                or any(not isinstance(item, kind) for item in items)
                or len({item.stable_key() for item in items}) != len(items)):
            raise error(f'reference {label} audit requires bounded unique cold records')
    inputs = {item.stable_key() for item in input_roles}
    roles = {role.stable_key(): role for context in contexts for role in context.role_occurrences}
    sets = {item.stable_key(): item for item in candidate_sets}
    expected = {}
    for candidate in candidate_sets:
        if candidate.reference.stable_key() not in inputs:
            raise error('reference candidate set lacks its current input role')
        for antecedent, depth in zip(candidate.antecedents, candidate.context_depths, strict=True):
            if (roles.get(antecedent.stable_key()) != antecedent
                    or not any(antecedent in context.role_occurrences and context.context_depth == depth
                               for context in contexts)):
                raise error('reference antecedent or depth differs from cold response graph')
        complete = tuple(roles[key] for key in sorted(roles)
                         if roles[key].source_ref != candidate.reference.source_ref)
        depths = tuple(min(context.context_depth for context in contexts
                           if role in context.role_occurrences) for role in complete)
        if candidate != ReferenceCandidateSet(candidate.reference, complete, depths):
            raise error('reference candidate set omits a cold antecedent or changes its depth')
        entry = candidate.query_evidence()
        expected[entry.evidence_key] = entry
        if entry in evidence and (candidate.query_binding().stable_key() not in bindings
                or not any(root.owner_space == 2 and root.status == ROOT_EXPANDED
                           and root.root_key == candidate.stable_key() and root.source_ref == candidate.source_ref
                           and root.scope_key == candidate.reference.scope_key for root in roots)):
            raise error('reference candidate set binding or expanded root differs')
    for resolution in resolutions:
        if sets.get(resolution.candidate_set.stable_key()) != resolution.candidate_set:
            raise error('reference resolution omits its complete candidate set')
        if any(item.stable_key() not in bindings for item in resolution.query_bindings()):
            raise error('QueryState omits complete reference decision bindings')
        for entry in resolution.query_evidence():
            if entry.evidence_key in expected and expected[entry.evidence_key] != entry:
                raise error('reference Evidence identity is ambiguous')
            expected[entry.evidence_key] = entry
            if entry not in evidence:
                raise error('QueryState omits consumed reference decision Evidence')
    members = tuple(item for item in evidence if item.space == 2 and item.evidence_key in expected)
    for entry in members:
        if entry != expected[entry.evidence_key]:
            raise error('reference Evidence differs from the complete cold member')
        if has_plan and entry.stable_key() not in plan_keys:
            raise error('ResponsePlan omits consumed reference Evidence')
    if has_plan and candidate_sets:
        adopted = tuple(candidate for candidate in candidate_sets if adopted_hypothesis_keys is None
                        or candidate.reference.hypothesis.stable_key() in adopted_hypothesis_keys)
        memory = {candidate.reference.observation for candidate in adopted}
        memory.update(candidate.reference.hypothesis for candidate in adopted)
        if not memory <= plan_refs:
            raise error('ResponsePlan omits current reference Observation/Hypothesis')
    return members
