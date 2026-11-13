"""交付取证须区分同槽指代备选和真正的并列问题。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.understanding.query_input_structure import InputRelationQuestionBinding
from pure_integer_ai.experiments.natural_question_capture import delivered_question_binding


def _case():
    """构造共享语法的两个来源备选及当前唯一Core/指代决定。"""
    first=InputRelationQuestionBinding((1,),(2,),(31,),((4,),),(5,))
    second=replace(first,relation_key=(32,))
    relations=tuple(SimpleNamespace(stable_key=lambda key=key:key,proposition_key=(claim,),predicate_key=(91,))
                    for key,claim in (((31,),71),((32,),72)))
    structure=SimpleNamespace(relation_question_bindings=(first,second),relation_candidates=relations,
                              discourse_compositions=(),discourse_gaps=())
    final=replace(first,relation_key=(33,))
    evidence=(1,1,1,1,8,1,71,0,0,0,0)
    reference=(91536,1,9,1,3)
    table=[(71,),final.stable_key(),evidence,reference]
    scalars=[0]*17
    scalars[11]=1
    value=dict(schema_version=4,integer_key_encoding='INTEGER_KEY_TABLE_V1',integer_key_table=table,
               state_key=dict(scalars=scalars,evidence=[3]),response_plan=dict(claim_refs=[[71]]),
               input_structure=dict(relation_question_bindings=[dict(integer_key_ref=2)]),
               reference_resolution_refs=[4])
    return structure,value,first


def test_resolved_alternatives_capture_only_the_current_claim():
    structure,value,first=_case()
    assert delivered_question_binding(structure,value)==first
    assert len(structure.relation_question_bindings)==2


def test_complete_compound_claims_keep_their_original_primary():
    structure,value,first=_case()
    value['response_plan']['claim_refs'].append([72])
    value['reference_resolution_refs']=[]
    assert delivered_question_binding(structure,value)==first


def test_unrelated_predicate_alternatives_are_outside_the_selected_slot_group():
    """Memory选中的答案角色槽可以与其它谓词候选共存，不能把它们误并为同一语法竞争。"""
    structure,value,first=_case()
    unrelated=replace(structure.relation_question_bindings[1],
                      answer_role_key=(99,), slot_span_refs=((98,),))
    structure.relation_question_bindings=(first,unrelated)
    structure.relation_candidates=(
        structure.relation_candidates[0],
        SimpleNamespace(stable_key=lambda:(32,), proposition_key=(72,), predicate_key=(92,)),
    )
    assert delivered_question_binding(structure,value)==first


@pytest.mark.parametrize('field',('frame_key','operation_key','slot_span_refs','answer_role_key'))
def test_distinct_question_slots_cannot_collapse_to_one_answer(field):
    structure,value,_=_case()
    second=structure.relation_question_bindings[1]
    changed=((99,),) if field=='slot_span_refs' else (99,)
    structure.relation_question_bindings=(structure.relation_question_bindings[0],replace(second,**{field:changed}))
    with pytest.raises(ValueError,match='one original syntax/span'):
        delivered_question_binding(structure,value)


@pytest.mark.parametrize('failure',('missing_core','missing_reference','conflict','final_span','compound_gap','multiple_claims'))
def test_missing_or_competing_current_evidence_rejects_capture(failure):
    structure,value,first=_case()
    if failure=='missing_core':
        value['state_key']['evidence']=[]
    elif failure=='missing_reference':
        value['reference_resolution_refs']=[]
    elif failure=='conflict':
        value['integer_key_table'][3]=(91536,1,9,1,4)
    elif failure=='final_span':
        value['integer_key_table'][1]=replace(first,slot_span_refs=((88,),)).stable_key()
    elif failure=='compound_gap':
        structure.discourse_gaps=((1,),)
    else:
        value['response_plan']['claim_refs'].append([73])
    with pytest.raises(ValueError):
        delivered_question_binding(structure,value)
