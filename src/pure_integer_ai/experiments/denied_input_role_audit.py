"""冷核已完整声明但被同次首次读取前澄清明确拒绝的全部当前角色。"""
from dataclasses import replace

from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EMPTY, ROOT_PENDING, SPACE_CORE, SPACE_MEMORY, BindingEntry, QueryRoot,
)
from pure_integer_ai.experiments.trained_input_clarification import INPUT_CLARIFICATION_VERSION


def qualify_denied_input_roles(trace, occurrences, *, evidence, bindings, roots,
        input_ref, input_structure, nonfactual_plan, max_state_members, error):
    """只有完整拒绝根证明才可将声明与零消费分账，普通缺证据仍然失败。

    所有角色已经由调用者从完整物理来源恢复；本函数不删除声明、补造读操作
    或重放查询。原首两个零读取状态必须保存全部原根及精确准入变化，最终
    保留全部初始投影、拒绝Binding和同次当前Observation的非事实采用。
    """
    from pure_integer_ai.experiments.pending_composition_call_audit import _decode_input
    from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack

    plan = trace.trace.get('response_plan')
    scalars = trace.trace.get('state_key',{}).get('scalars',())
    if (not occurrences or not nonfactual_plan or trace.version != 4
            or trace.trace.get('termination') != 3 or not isinstance(plan, dict)
            or len(scalars)!=17 or scalars[8]!=0 or scalars[16]!=3
            or any(plan.get(name) for name in (
                'claim_refs','claim_evidence_refs','claim_slot_ranges','event_refs'))
            or trace.trace.get('generation', {}).get('kind') != 3
            or trace.trace.get('reference_candidate_set_refs')
            or trace.trace.get('reference_resolution_refs')
            or input_structure is None
            or _decode_input(trace.trace['input_structure'], trace) != input_structure.integer_trace()
            or any(item.space == SPACE_CORE for item in trace.hops())
            or any(item.space == SPACE_CORE and item.polarity in (1,2) for item in evidence)):
        raise error('unconsumed input roles lack complete same-call nonfactual clarification')
    query = trace.resolve_ref(trace.trace['query_ref'])
    if query != input_structure.source_ref:
        raise error('unconsumed input roles belong to another current query')
    core = tuple(root for root in roots if root.owner_space == SPACE_CORE)
    memory = tuple(root for root in roots if root.owner_space == SPACE_MEMORY)
    if (len(core) != 1 or core[0].root_key != (INPUT_CLARIFICATION_VERSION,0)
            or core[0].status != ROOT_EMPTY or len(memory) != 1
            or memory[0].root_key != input_ref.stable_key()
            or any(root.status == ROOT_PENDING for root in roots)):
        raise error('unconsumed input roles expanded facts, history or unresolved roots')
    role_keys = {item.stable_key() for item in occurrences}
    if (any(item.evidence_key in role_keys for item in evidence)
            or any(item.query_binding().stable_key() in bindings for item in occurrences)):
        raise error('denied input roles cannot also retain any consumed role evidence or binding')
    steps = trace.trace.get('frontier_trace',{}).get('steps')
    if not isinstance(steps,(list,tuple)) or len(steps)<2:
        raise error('unconsumed input roles lost original admission states')

    def keys(row, name):
        """有界完整解引用原步状态，禁止用最终状态补缺。"""
        refs = row.get(name)
        if not isinstance(refs,(list,tuple)) or len(refs)>max_state_members:
            raise error('unconsumed role step exceeds its original member budget')
        values = tuple(trace.resolve_ref(ref) for ref in refs)
        if len(set(values)) != len(values):
            raise error('unconsumed role step duplicates an original member')
        return values

    def read_root(key):
        """恢复既有规范root整数记录，不推断优先级或来源。"""
        root,source,scope = _unpack(key,3,3)
        return QueryRoot(key[0],root,key[1],key[2],source,scope)

    initial,admitted = steps[:2]
    if any(row.get('counters') != [0,0,0,0] or row.get('visited_refs')
           for row in (initial,admitted)):
        raise error('denied input role admission must precede every query read')
    original = tuple(read_root(key) for key in keys(initial,'root_refs'))
    selected = tuple(read_root(key) for key in keys(admitted,'root_refs'))
    if {replace(root,status=ROOT_PENDING).stable_key() for root in selected
        if root.status!=ROOT_EMPTY} != {
            replace(root,status=ROOT_PENDING).stable_key() for root in roots
            if root.status!=ROOT_EMPTY} or tuple(
                root.stable_key() for root in selected if root.status==ROOT_EMPTY) != tuple(
                root.stable_key() for root in roots if root.status==ROOT_EMPTY):
        raise error('final clarification roots differ from the original admitted roots')
    selected_ids = {(root.owner_space,root.root_key) for root in selected}
    expected = {BindingEntry((INPUT_CLARIFICATION_VERSION,1),root.stable_key(),
        space=SPACE_CORE,scope_key=query).stable_key() for root in original
        if (root.owner_space,root.root_key) not in selected_ids}
    actual = {key for key in bindings if _unpack(key,1,3,1)[0] == (INPUT_CLARIFICATION_VERSION,1)}
    if (not expected or actual != expected or any(key[-1] for key in bindings)
            or not expected <= set(keys(admitted,'binding_refs'))):
        raise error('complete denied roots differ from preserved same-call denial bindings')
    for occurrence in occurrences:
        matching = tuple(root for root in original
            if root.owner_space==SPACE_MEMORY and root.root_key==occurrence.stable_key())
        if (len(matching)!=1 or matching[0] != occurrence.seed(matching[0].priority_weight)[0]
                or (SPACE_MEMORY,occurrence.stable_key()) in selected_ids):
            raise error('declared input role lacks its exact physical unconsumed original root')
    final_evidence = {item.stable_key() for item in evidence}
    if (not set(keys(initial,'evidence_refs')) <= final_evidence
            or not set(keys(initial,'binding_refs')) <= bindings):
        raise error('clarification discarded original input projections or bindings')
    report = dict(status='MATCHED_COMPLETE_DECLARED_INPUT_ROLES_AND_EXPLICIT_DENIED_ROOTS',
        declared_role_count=len(occurrences),denied_role_count=len(occurrences),
        consumed_role_count=0,denied_root_count=len(expected))
    return report, tuple(root for root in original
        if (root.owner_space,root.root_key) not in selected_ids)
