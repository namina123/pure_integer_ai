"""按有界成员恢复完整生成绑定，不把复合Plan当成可任意放大的原子键。"""
from collections.abc import Mapping

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.query_state import BindingEntry
from pure_integer_ai.cognition.shared.response_plan import ResponsePlan, ResponseSlot, ResponseRealization
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record


def verify_compound_generation_binding(value, trace, evidence, *, key_reader, rows_reader,
                                      max_state_members, max_ohe_declarations, error):
    """只恢复同次零事实生成的规范值，所有原子和成员仍走原预算读取器。"""
    from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack

    def key(item, label):
        """显式引用只指向原trace整数表，原子长度预算不变。"""
        if isinstance(item, Mapping):
            if set(item) != {'integer_key_ref'}:
                raise error('compound generation uses an unregistered integer reference')
            item = trace.resolve_ref(item['integer_key_ref'])
        return key_reader(item, label)

    def rows(item, label, maximum=max_state_members):
        """复用原状态/声明成员预算，不按复合长度扩大分母。"""
        return rows_reader(item, label, maximum)

    def identity(item):
        """恢复完整一等对象身份，不能只看名称或类型标签。"""
        return ObjectIdentity.from_stable_key(key(item, 'compound object identity'))

    def surface(item):
        """仅恢复原输出表面的码点，不执行新生成或读取来源正文。"""
        return ''.join(map(chr, key(item, 'compound generated surface')))

    def optional(item):
        """空身份保留None，其余必须恢复完整一等身份。"""
        if isinstance(item, Mapping):
            if set(item) != {'integer_key_ref'}:
                raise error('compound optional identity uses an unregistered reference')
            item = trace.resolve_ref(item['integer_key_ref'])
        return identity(item) if item else None

    original = trace.trace
    plan = original.get('response_plan')
    generation = original.get('generation')
    if (not isinstance(plan, Mapping) or not isinstance(generation, Mapping)
            or generation.get('kind') != 3 or plan.get('claim_refs')
            or plan.get('claim_evidence_refs') or plan.get('claim_slot_ranges')):
        raise error('compound binding requires the original zero-claim generic generation')
    from pure_integer_ai.experiments.context_only_plan_slots import language_only_plan_slots
    if not language_only_plan_slots(plan):
        raise error('compound context-only generation cannot contain factual object slots')
    if not isinstance(value, (tuple, list)) or len(value) < 6 or value[0] != 3 or value[-1] != 0:
        raise error('compound generation binding space or conflict marker differs')
    role, filler, scope = _unpack(tuple(value), 1, 3, 1)
    act = key(plan['response_act'], 'compound ResponseAct')
    query = trace.resolve_ref(original['query_ref'])
    if key(role, 'compound binding role') != act or key(scope, 'compound binding scope') != query:
        raise error('compound generation binding differs from the current act or query')
    slots = []
    for slot in rows(plan['slot_sequence'], 'compound Plan slots'):
        if type(slot['required']) is not int or slot['required'] not in (0, 1):
            raise error('compound slot required marker is not a canonical integer')
        slots.append(ResponseSlot(identity(slot['role']), identity(slot['filler']),
            surface(slot['filler_surface_values']), bool(slot['required']), slot['source_hash'],
            key(slot['token_digest'], 'compound slot digest'),
            key(slot['allowed_node_kinds'], 'compound slot kinds')))
    realizations = tuple(ResponseRealization(surface(item['surface_values']),
        identity(item['frame_proposition']), item['frame_source_hash'], item['slot_count'])
        for item in rows(plan['realization_candidates'], 'compound Plan realizations'))
    recorded_evidence = tuple(key(item, 'compound Plan evidence')
        for item in rows(plan['evidence_refs'], 'compound Plan evidence', max_state_members))
    expected_evidence = tuple(item.stable_key() for item in evidence)
    if recorded_evidence != expected_evidence:
        raise error('compound Plan evidence differs from the independently restored QueryState')
    restored = ResponsePlan(identity(plan['response_act']), (), tuple(slots), realizations, expected_evidence,
        scope_and_time=key(plan['scope_and_time'], 'compound Plan scope'),
        discourse_links=tuple(identity(item) for item in rows(plan['discourse_links'], 'compound discourse identities')),
        event_refs=tuple(identity(item) for item in rows(plan['event_refs'], 'compound event identities')),
        memory_refs=tuple(MemoryObjectRef.from_stable_key(key(item, 'compound Plan Memory ref'))
            for item in rows(plan['memory_refs'], 'compound Plan Memory refs', max_ohe_declarations)),
        style_ref=optional(plan['style_ref']), carrier_ref=optional(plan['carrier_ref']))
    saved_plan = plan['stable_key']
    saved_plan = trace.resolve_ref(saved_plan['integer_key_ref']) if isinstance(saved_plan, Mapping) else tuple(saved_plan)
    if restored.stable_key() != saved_plan:
        raise error('compound complete Plan differs from its canonical integer projection')
    encoded = key(generation['trace'], 'compound original generation trace')
    fields = unpack_record(encoded, 91566)
    if len(fields) != 7 or fields[1] != query or fields[3] != act:
        raise error('compound original generation trace has a different query or act')
    representations = tuple(key(item, 'compound Representation')
        for item in rows(generation['representations'], 'compound Representations'))
    expected = pack_record(91566, fields[0], fields[2], restored.stable_key(),
                           pack_record(1, *representations), encoded)
    recorded = tuple(trace.resolve_ref(item) for item in rows(
        original.get('generic_response_generation_refs'), 'compound generation records'))
    if len(recorded) != 1 or recorded[0] != expected or filler != expected:
        raise error('compound binding differs from the one complete original generation record')
    canonical = BindingEntry(act, expected, 3, query).stable_key()
    if canonical != tuple(value):
        raise error('compound generation binding is not canonical')
    return canonical
