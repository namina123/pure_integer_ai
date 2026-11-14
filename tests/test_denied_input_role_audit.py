"""明确拒绝的完整当前角色可不消费；缺失、伪造及混合消费继续拒绝。"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ROLE, ObjectIdentity, occurrence_identity, span_identity,
)
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EMPTY, ROOT_EXPANDED, ROOT_PENDING, BindingEntry, EvidenceEntry,
    QueryAnchor, QueryRoot, seed_query,
)
from pure_integer_ai.experiments.denied_input_role_audit import qualify_denied_input_roles
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.trained_graph_query_bridge import _IntegerQueryTrace
from pure_integer_ai.experiments.trained_input_clarification import clarification_query_state
from pure_integer_ai.experiments.trained_response_occurrence import InputRoleOccurrence
from test_query_memory_ohe_audit import _fixture
from test_query_unlicensed_relation import _grammar


def _case():
    """真实角色seed及原澄清函数产生完整初始/准入状态，不手造拒绝证据。"""
    _, declarations, manifest, original_evidence = _fixture(3)
    source = manifest.source
    observation, hypothesis = (manifest.bindings[n].object_ref for n in (0,1))
    role = InputRoleOccurrence(occurrence_identity(source,start=0,end=1,ordinal=0),
        span_identity(source,members=((0,1),)),
        ObjectIdentity(OBJECT_ROLE,(81,),source.owner,source.versions), observation,hypothesis,
        declarations[observation].event.payload.context.value(),0,
        original_evidence.source_ref,original_evidence.scope_key,(65,))
    projector,_ = _grammar()
    prepared = projector.project((11,33,99))
    query = prepared.source_ref
    roots = (QueryRoot(2,observation.stable_key(),ROOT_PENDING,1),
        QueryRoot(3,(31,),ROOT_PENDING,1), QueryRoot(1,(41,),ROOT_PENDING,1), role.seed(1)[0])
    initial = seed_query(query,roots=roots,anchors=tuple(
        QueryAnchor(root.owner_space,root.root_key,kind=1) for root in roots)).with_(
            evidence=(EvidenceEntry((9,),(41,),space=1,polarity=3),),
            bindings=(BindingEntry((71,),(72,),space=1,scope_key=query),))
    admitted = clarification_query_state(initial,(),
        (SimpleNamespace(observation_key=observation.stable_key()),),
        (SimpleNamespace(key=(31,)),))
    assert admitted is not None
    final = admitted.with_(roots=tuple(replace(root,status=ROOT_EXPANDED)
        if root.status!=ROOT_EMPTY else root for root in admitted.roots))
    recorder = _IntegerQueryTrace(reference_mode=True)
    recorder.record(initial,0)
    recorder.record(admitted,0)
    frontier = recorder.export()
    scalars = [0]*17
    scalars[15],scalars[16] = 1,3
    value = dict(schema_version=4,integer_key_encoding='INTEGER_KEY_TABLE_V1',
        integer_key_table=frontier['integer_key_table'],query_ref=recorder._ref(query),
        frontier_trace=frontier,termination=3,response_plan=dict(claim_refs=[],
            claim_evidence_refs=[],claim_slot_ranges=[],event_refs=[]),generation=dict(kind=3),
        input_structure=prepared.integer_trace(),state_key=dict(scalars=scalars),hops=[])
    # query_ref首次登记可能扩展表，完整列表再次取得，保持所有原步引用。
    value['integer_key_table'] = recorder.export()['integer_key_table']
    return dict(trace=GraphQueryTraceEvidence(value),occurrences=(role,),evidence=final.evidence,
        bindings={item.stable_key() for item in final.bindings},roots=final.roots,
        input_ref=observation,input_structure=prepared,nonfactual_plan=True,max_state_members=256,
        error=ValueError)


def test_complete_denial_proof_preserves_declarations_and_consumes_no_roles():
    case = _case()
    before = deepcopy(case['trace'].trace)
    result,denied = qualify_denied_input_roles(**case)
    assert result['declared_role_count']==result['denied_role_count']==1
    assert result['consumed_role_count']==0 and result['denied_root_count']==2
    assert len(denied)==2 and all(root.status==ROOT_PENDING for root in denied)
    assert case['trace'].trace==before and len(case['occurrences'])==1


@pytest.mark.parametrize('mutation',(
    'missing_denial','extra_denial','role_evidence','role_binding','source_root',
    'prior_read','missing_original_projection','foreign_query','truth','fact_plan',
    'reference','history_root','pending','factual_mode',
))
def test_denial_qualification_rejects_missing_and_consumed_members(mutation):
    case = _case()
    trace,role = case['trace'],case['occurrences'][0]
    if mutation=='missing_denial':
        case['bindings']={key for key in case['bindings'] if key!=next(
            key for key in case['bindings'] if key[0]==1 and key[-1]==0 and len(key)>100)}
    elif mutation=='extra_denial':
        from pure_integer_ai.experiments.trained_input_clarification import INPUT_CLARIFICATION_VERSION
        case['bindings'].add(BindingEntry((INPUT_CLARIFICATION_VERSION,1),
            QueryRoot(1,(99,),ROOT_PENDING,1).stable_key(),space=1,
            scope_key=case['input_structure'].source_ref).stable_key())
    elif mutation=='role_evidence':case['evidence']=(*case['evidence'],role.query_evidence())
    elif mutation=='role_binding':case['bindings'].add(role.query_binding().stable_key())
    elif mutation=='source_root':
        original=role.seed(1)[0]
        index=trace.trace['integer_key_table'].index(list(original.stable_key()))
        trace.trace['integer_key_table'][index]=list(replace(original,source_ref=(99,)).stable_key())
    elif mutation=='prior_read':trace.trace['frontier_trace']['steps'][1]['counters'][3]=1
    elif mutation=='missing_original_projection':case['evidence']=()
    elif mutation=='foreign_query':trace.trace['query_ref']=0
    elif mutation=='truth':case['evidence']=(*case['evidence'],EvidenceEntry((9,),(41,),space=1,polarity=1))
    elif mutation=='fact_plan':trace.trace['response_plan']['claim_refs']=[[7]]
    elif mutation=='reference':trace.trace['reference_candidate_set_refs']=[1]
    elif mutation=='history_root':case['roots']=(*case['roots'],QueryRoot(2,(99,),ROOT_EXPANDED,1))
    elif mutation=='pending':case['roots']=tuple(replace(root,status=ROOT_PENDING)
        if root.owner_space==2 else root for root in case['roots'])
    elif mutation=='factual_mode':case['nonfactual_plan']=False
    # 表内容被改时重建严格只读视图，使引用按修改后的原artifact解释。
    case['trace']=GraphQueryTraceEvidence(trace.trace)
    with pytest.raises(ValueError):qualify_denied_input_roles(**case)


@pytest.mark.parametrize('mutation',(
    'valid','missing_denial','source','scope','expanded','present_root','present_binding',
    'duplicate','missing_declaration','budget',
))
def test_denied_recall_retains_physical_declaration_and_never_counts_a_read(mutation):
    """只有同一物理来源的被拒pending root可分账，不能隐藏任何实际消费。"""
    from pure_integer_ai.experiments.memory_provenance_inputs import verify_recalled_observations
    _,declarations,manifest,evidence = _fixture(3)
    observation = manifest.bindings[0].object_ref
    root = QueryRoot(2,observation.stable_key(),ROOT_PENDING,1,
        source_ref=evidence.source_ref,scope_key=evidence.scope_key)
    denied,roots,bindings = (root,),(),set()
    if mutation=='missing_denial':denied=()
    elif mutation=='source':denied=(replace(root,source_ref=(99,)),)
    elif mutation=='scope':denied=(replace(root,scope_key=(99,)),)
    elif mutation=='expanded':denied=(replace(root,status=ROOT_EXPANDED),)
    elif mutation=='present_root':roots=(replace(root,status=ROOT_EXPANDED),)
    elif mutation=='present_binding':bindings.add(BindingEntry((1,),observation.stable_key(),2,
        evidence.scope_key).stable_key())
    elif mutation=='duplicate':denied=(root,root)
    def verify():
        """完整声明读取在分账前执行；未返回hop不是未检查物理对象。"""
        return verify_recalled_observations((observation,),maximum=0 if mutation=='budget' else 256,
            denied_roots=denied,roots=roots,bindings=bindings,
            declaration=lambda ref: (_ for _ in ()).throw(ValueError('missing declaration'))
                if mutation=='missing_declaration' else declarations[ref],
            observation=lambda source: observation,source_hash=lambda source:19,error=ValueError)
    if mutation=='valid':assert verify()==set()
    else:
        with pytest.raises(ValueError):verify()
