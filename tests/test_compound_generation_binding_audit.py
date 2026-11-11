"""复合生成按原成员预算完整恢复，不能用超长平面键或漂移成员借道。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity, OBJECT_LANGUAGE_ATOM, OBJECT_ROLE, OBJECT_PROPOSITION, OBJECT_REPRESENTATION,
)
from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.cognition.shared.response_plan import ResponsePlan, ResponseSlot, ResponseRealization, response_act_identity
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record
from pure_integer_ai.experiments.compound_generation_binding_audit import verify_compound_generation_binding
from pure_integer_ai.experiments.query_memory_ohe_audit import _key, _rows, QueryMemoryOheEvidenceError
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge


def _fixture(*, large=False):
    """真实ResponsePlan及规范生成/绑定记录；大例只增加有界Evidence成员长度。"""
    act=response_act_identity((1, 3))
    slot=ResponseSlot(ObjectIdentity(OBJECT_ROLE,(11,)),ObjectIdentity(OBJECT_LANGUAGE_ATOM,(12,)),
        '未知',True,31,allowed_node_kinds=(OBJECT_LANGUAGE_ATOM,))
    evidence=tuple(EvidenceEntry((21,), (index,), space=2, polarity=3, evidence_key=(1,index),
        payload_key=(77,)*(550_000 if large else 3)) for index in (41,42))
    plan=ResponsePlan(act,(),(slot,),(ResponseRealization('未知',ObjectIdentity(OBJECT_PROPOSITION,(13,)),31,1),),
        tuple(item.stable_key() for item in evidence),scope_and_time=(1,))
    query=(1,2,3)
    node,context=(91564,1),(91565,1)
    representations=(ObjectIdentity(OBJECT_REPRESENTATION,(14,)).stable_key(),)
    encoded=pack_record(91566,node,query,context,act.stable_key(),pack_record(1),pack_record(2),pack_record(3))
    generated=pack_record(91566,node,context,plan.stable_key(),pack_record(1,*representations),encoded)
    binding=BindingEntry(act.stable_key(),generated,3,query).stable_key()
    value=dict(query_ref=query,response_plan=TrainedGraphQueryBridge._integer_response_plan(plan),
        generation=dict(kind=3,trace=encoded,representations=representations),generic_response_generation_refs=[generated])
    trace=SimpleNamespace(trace=value,resolve_ref=lambda key:tuple(key))
    kwargs=dict(key_reader=_key,rows_reader=_rows,max_state_members=4096,
                max_ohe_declarations=256,error=QueryMemoryOheEvidenceError)
    return binding,trace,evidence,kwargs


def test_compound_binding_larger_than_atom_limit_preserves_all_original_members():
    """总复合键超原子上限时完整相等；任一独立原子仍不得超限。"""
    binding,trace,evidence,kwargs=_fixture(large=True)
    assert len(binding)>1_000_000
    with pytest.raises(QueryMemoryOheEvidenceError):
        _key(binding,'untyped long binding')
    assert verify_compound_generation_binding(binding,trace,evidence,**kwargs)==binding
    assert kwargs['max_state_members']==4096 and kwargs['max_ohe_declarations']==256


@pytest.mark.parametrize('drift', ('kind','claim','range','factual_slot','space','role','scope',
    'conflict','missing_record','duplicate_record','representation','generation_trace','plan_key',
    'evidence','required','atom_budget','member_budget'))
def test_compound_binding_rejects_unregistered_or_changed_members(drift):
    """类型、完整来源成员和原预算都参与比较，不能只比较长度或hash。"""
    binding,trace,evidence,kwargs=_fixture()
    plan=trace.trace['response_plan']
    if drift=='kind':trace.trace['generation']['kind']=6
    elif drift=='claim':plan['claim_refs']=[[9]]
    elif drift=='range':plan['claim_slot_ranges']=[dict(claim=[9],start=0,end=1)]
    elif drift=='factual_slot':plan['slot_sequence'][0]['allowed_node_kinds']=[4]
    elif drift=='space':binding=(2,*binding[1:])
    elif drift=='role':binding=BindingEntry((9,),binding[1:],3,(1,2,3)).stable_key()
    elif drift=='scope':binding=BindingEntry(tuple(plan['response_act']),binding[1:],3,(9,)).stable_key()
    elif drift=='conflict':binding=(*binding[:-1],1)
    elif drift=='missing_record':trace.trace['generic_response_generation_refs']=[]
    elif drift=='duplicate_record':trace.trace['generic_response_generation_refs']*=2
    elif drift=='representation':trace.trace['generation']['representations']=((9,),)
    elif drift=='generation_trace':trace.trace['generation']['trace']=(91566,)
    elif drift=='plan_key':plan['stable_key']=[9]
    elif drift=='evidence':evidence=(replace(evidence[0],payload_key=(99,)),evidence[1])
    elif drift=='required':plan['slot_sequence'][0]['required']=True
    elif drift=='atom_budget':plan['evidence_refs'][0]=[1]*1_000_001
    else:kwargs['max_state_members']=1
    with pytest.raises((ValueError,TypeError)):
        verify_compound_generation_binding(binding,trace,evidence,**kwargs)
