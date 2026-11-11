"""从当前物理 O/H/E 与原生成记录恢复非事实交付的有界采用分支。"""
from collections.abc import Mapping

from pure_integer_ai.cognition.shared.memory_event import (
    INTAKE_DERIVED_HYPOTHESIS, HypothesisPayload, MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.query_state import ROOT_EXPANDED
from pure_integer_ai.cognition.understanding.query_memory_candidates import MEMORY_GENERIC_CANDIDATE_VERSION
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record


def verify_generic_plan_memory(trace, input_source, input_ref, evidence, roots, *,
                               declaration, manifest, error):
    """核验完整 UNKNOWN 候选、生成 context 和首分支，绝不把查询历史全部采用。"""
    generation, plan = trace.trace['generation'], trace.trace['response_plan']

    def integer_key(encoded):
        """按原 trace 的显式引用格式恢复整数键，不推断缺项。"""
        if isinstance(encoded, Mapping):
            if set(encoded) != {'integer_key_ref'}:
                raise error('generic generation trace uses an unregistered reference')
            return trace.resolve_ref(encoded['integer_key_ref'])
        if (not isinstance(encoded, (tuple, list))
                or any(type(value) is not int or value < 0 for value in encoded)):
            raise error('generic generation trace uses an unregistered reference')
        return tuple(encoded)

    fields = unpack_record(integer_key(generation.get('trace')), 91566)
    if len(fields) != 7:
        raise error('generic generation trace has an unregistered shape')
    node, query, context, act, hypotheses, members, graphs = fields
    if (query != trace.resolve_ref(trace.trace['query_ref'])
            or act != tuple(plan['response_act']) or act != integer_key(generation['response_act'])
            or unpack_record(graphs, 3) or plan.get('event_refs')
            or not any(root.owner_space == 3 and root.status == ROOT_EXPANDED
                       and root.root_key == node for root in roots)):
        raise error('generic generation query/act/root or visible graph contract differs')
    recorded = unpack_record(hypotheses, 1)
    current = []
    for binding in manifest(input_source).bindings:
        if binding.binding_kind != INTAKE_DERIVED_HYPOTHESIS:
            continue
        row = declaration(binding.object_ref)
        if not isinstance(row.event.payload, HypothesisPayload):
            raise error('generic hypothesis lacks its durable payload')
        hypothesis = row.event.payload.hypothesis
        if hypothesis.candidate_key[:1] != (MEMORY_GENERIC_CANDIDATE_VERSION,):
            continue
        key = binding.object_ref.stable_key()
        unknown = tuple(item for item in evidence if item.hypothesis_key == key
                        and item.space in (2, 3) and item.polarity == 3)
        if not unknown:
            continue
        if (hypothesis.observation != input_source
                or any(item.source_ref[1:] != input_source.stable_key()
                       or item.scope_key != hypothesis.scope.stable_key() for item in unknown)):
            raise error('generic hypothesis crosses the current source/scope')
        current.append((key, hypothesis.scope.stable_key()))
    current.sort()
    keys = tuple(key for key, _scope in current)
    scopes = {scope for _key, scope in current}
    if not keys or recorded != keys or len(scopes) != 1:
        raise error('generic generation omits or reorders a current UNKNOWN candidate')
    expected_context = pack_record(91565, query, input_ref.stable_key(),
        input_source.stable_key(), next(iter(scopes)), pack_record(1, *keys))
    if context != expected_context:
        raise error('generic generation context differs from durable current O/H/E')
    expected_members = tuple(item.evidence_key for item in evidence if item.space in (2, 3))
    if unpack_record(members, 2) != expected_members:
        raise error('generic generation omits or changes consumed Memory/Dialogue evidence')
    expected_evidence = tuple(item.stable_key() for item in evidence)
    plan_evidence = tuple(tuple(item) for item in plan['evidence_refs'])
    if plan_evidence != expected_evidence:
        raise error('generic Plan evidence differs from the complete QueryState')
    selected = keys[0]
    selected_evidence = tuple(item for item in evidence if item.space == 2
                              and item.hypothesis_key == selected)
    if not selected_evidence or any(item.polarity != 3 for item in selected_evidence):
        raise error('generic adopted branch is not the current UNKNOWN hypothesis')
    expected_refs = {MemoryObjectRef.from_stable_key(selected)}
    expected_refs.update(MemoryObjectRef.from_stable_key(item.evidence_key)
                         for item in selected_evidence if item.evidence_key[:1] == (1,))
    refs = tuple(MemoryObjectRef.from_stable_key(tuple(key)) for key in plan['memory_refs'])
    if refs != tuple(sorted(expected_refs, key=lambda item: item.stable_key())):
        raise error('generic Plan changes its deterministic bounded adopted H/E branch')
    return frozenset((selected,)), {
        'status':'MATCHED_COMPLETE_GENERIC_CONTEXT_AND_BOUNDED_ADOPTION',
        'generic_hypothesis_count':len(keys), 'adopted_memory_count':len(refs),
        'query_evidence_count':len(evidence), 'visible_graph_slot_count':0,
    }
