"""Capture the exact complete observed-role sequence after reference adoption."""
from dataclasses import replace

import pytest

from pure_integer_ai.experiments.natural_question_capture import delivered_question_binding
from test_natural_question_capture import _case


def _multi(count):
    structure,value,first=_case()
    spans=tuple((4,index) for index in range(1,count+1))
    original=tuple(replace(binding,slot_span_refs=spans) for binding in structure.relation_question_bindings)
    structure.relation_question_bindings=original
    final=replace(original[0],relation_key=(33,))
    value['integer_key_table'][1]=final.stable_key()
    return structure,value,original[0],final


@pytest.mark.parametrize('count',(2,5,6))
def test_multirole_reference_keeps_every_original_observed_span(count):
    structure,value,first,_final=_multi(count)
    assert delivered_question_binding(structure,value)==first
    assert len(first.slot_span_refs)==count
    assert len(structure.relation_question_bindings)==2


@pytest.mark.parametrize('drift',('omit','reorder','span','answer','truncated','trailing','count'))
def test_multirole_capture_rejects_incomplete_or_changed_binding(drift):
    structure,value,_first,final=_multi(5)
    if drift=='omit':final=replace(final,slot_span_refs=final.slot_span_refs[:-1])
    elif drift=='reorder':final=replace(final,slot_span_refs=tuple(reversed(final.slot_span_refs)))
    elif drift=='span':final=replace(final,slot_span_refs=((88,),*final.slot_span_refs[1:]))
    elif drift=='answer':final=replace(final,answer_role_key=(88,))
    key=final.stable_key()
    if drift=='truncated':key=key[:-1]
    elif drift=='trailing':key=(*key,1,9)
    elif drift=='count':
        cursor=1
        for _index in range(3):cursor+=1+key[cursor]
        key=(*key[:cursor],key[cursor]+1,*key[cursor+1:])
    value['integer_key_table'][1]=key
    with pytest.raises(ValueError):delivered_question_binding(structure,value)
