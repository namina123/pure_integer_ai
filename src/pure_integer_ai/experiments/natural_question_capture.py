"""区分并列问题与运行内已消解的WH备选，验收不决定生产答案。"""
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack


def delivered_question_binding(structure, result):
    """返回同次当前QueryState实际采用的原绑定，保留并列问题完整分母。

    指代备选必须有唯一原claim、唯一最终绑定、精确的原语法/观察跨度与
    当前Core正证据；这里只恢复验收对象，不改查询、事实或交付状态。
    """
    bindings=structure.relation_question_bindings
    if not bindings:
        raise ValueError('delivered question lacks an original input binding')
    if len(bindings)==1:
        return bindings[0]
    trace=GraphQueryTraceEvidence(result)
    relations={item.stable_key():item for item in structure.relation_candidates}
    claims=tuple(tuple(key) for key in result['response_plan']['claim_refs'])
    bound={relations[item.relation_key].proposition_key for item in bindings}
    primary=trace.resolve_ref(result['state_key']['scalars'][11],'QueryState best claim')
    candidates=tuple(item for item in bindings if relations[item.relation_key].proposition_key==primary)
    if claims and set(claims)==bound:
        if len(candidates)!=1:
            raise ValueError('compound delivery lacks one original primary binding')
        return candidates[0]
    if (len(claims)!=1 or claims!=(primary,) or len(candidates)!=1
            or structure.discourse_compositions or structure.discourse_gaps
            or not all(item.answer_role_key for item in bindings)):
        raise ValueError('multi-question binding differs from the delivered claim set')
    selected,=candidates
    syntax=lambda item:(item.frame_key,item.operation_key,item.slot_span_refs,item.answer_role_key)
    if any(syntax(item)!=syntax(selected) for item in bindings):
        raise ValueError('resolved WH alternatives do not share one original syntax/span')
    current=result['input_structure']['relation_question_bindings']
    if len(current)!=1 or set(current[0])!={'integer_key_ref'}:
        raise ValueError('resolved WH lacks one exact final question binding')
    key=trace.resolve_ref(current[0]['integer_key_ref'])
    if key[:1]!=(92032,):
        raise ValueError('resolved WH has an unregistered binding format')
    fields,cursor=[],1
    for _ in range(3):
        size=key[cursor]
        cursor+=1
        if size<=0 or cursor+size>len(key):
            raise ValueError('resolved WH binding field is truncated')
        fields.append(key[cursor:cursor+size])
        cursor+=size
    if cursor>=len(key) or key[cursor]!=1:
        raise ValueError('resolved WH requires one original observed role span')
    cursor+=1
    for _ in range(2):
        size=key[cursor]
        cursor+=1
        if size<=0 or cursor+size>len(key):
            raise ValueError('resolved WH role/span field is truncated')
        fields.append(key[cursor:cursor+size])
        cursor+=size
    if (cursor!=len(key) or fields[0]!=selected.frame_key
            or fields[1]!=selected.operation_key or (fields[3],)!=selected.slot_span_refs
            or fields[4]!=selected.answer_role_key):
        raise ValueError('resolved WH differs from the complete original syntax/span')
    positive=tuple(_unpack(trace.resolve_ref(ref),3,6) for ref in result['state_key']['evidence']
                   if trace.resolve_ref(ref)[:2]==(1,1))
    if not any(item[1]==primary for item in positive):
        raise ValueError('resolved WH lacks current positive Core evidence')
    if len(result.get('reference_resolution_refs',()))!=1:
        raise ValueError('resolved WH lacks its unique current reference decision')
    reference=trace.resolve_ref(result['reference_resolution_refs'][0])
    if len(reference)<2 or reference[:1]!=(91536,) or reference[1]<=0:
        raise ValueError('resolved WH reference record is incomplete')
    offset=2+reference[1]
    if reference[offset:offset+2]!=(1,3):
        raise ValueError('resolved WH reference is not supported and resolved')
    return selected
