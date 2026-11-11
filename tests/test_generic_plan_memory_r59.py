"""完整查询证据和有界非事实采用必须同时成立，不能以放宽成员过门。"""
from copy import deepcopy
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.query_state import QueryRoot, ROOT_EXPANDED
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record
from pure_integer_ai.experiments.generic_plan_memory_audit import verify_generic_plan_memory
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError
from test_query_memory_ohe_audit import _fixture


def _case():
    """构造同一真实类型的 UNKNOWN 来源链与完整生成记录。"""
    captured, declarations, manifest, evidence = _fixture(3, candidate_key=pack_record(91503,(7,),(3,1)))
    raw = captured['trace']
    raw['integer_key_encoding'] = 'INTEGER_KEY_TABLE_V1'
    observation = manifest.bindings[0].object_ref
    hypothesis = manifest.bindings[1].object_ref
    eref = manifest.bindings[2].object_ref
    members = (evidence, replace(evidence, space=3))
    query, node, act = (65,), (91564,1), (15,1)
    context = pack_record(91565,query,observation.stable_key(),manifest.source.stable_key(),
                          evidence.scope_key,pack_record(1,hypothesis.stable_key()))
    fields = (node,query,context,act,pack_record(1,hypothesis.stable_key()),
              pack_record(2,*(item.evidence_key for item in members)),pack_record(3))
    raw['generation'] = dict(kind=3,response_act=list(act),trace=list(pack_record(91566,*fields)))
    raw['response_plan'].update(response_act=list(act),event_refs=[],memory_refs=[
        list(ref.stable_key()) for ref in sorted((hypothesis,eref),key=lambda item:item.stable_key())])
    roots = (QueryRoot(3,node,ROOT_EXPANDED,1),)
    return raw, declarations, manifest, observation, members, roots, fields


def _verify(case):
    """只调用被测的独立恢复器，不运行查询或生成器。"""
    raw, declarations, manifest, observation, members, roots, _fields = case
    return verify_generic_plan_memory(GraphQueryTraceEvidence(raw),manifest.source,observation,members,roots,
        declaration=lambda ref:declarations[ref],manifest=lambda source:manifest,error=QueryMemoryOheEvidenceError)


def test_bounded_adoption_retains_complete_query_evidence_without_adopting_observation():
    """来源 O/H/E 完整但交付只采用选中 H/E，查询证据没有被裁剪。"""
    case = _case()
    adopted, receipt = _verify(case)
    assert adopted == frozenset((case[2].bindings[1].object_ref.stable_key(),))
    assert receipt['adopted_memory_count'] == 2
    assert receipt['query_evidence_count'] == 2


@pytest.mark.parametrize('mutation',('missing_evidence','extra_adoption','missing_adoption','missing_candidate',
                                     'foreign_context','changed_query','visible_graph','changed_members'))
def test_saved_generic_branch_drift_fails_closed(mutation):
    """少证据、多采用、候选缺失、跨来源和未登记可见图槽不能获得资格。"""
    case = _case()
    raw, _declarations, _manifest, observation, _members, _roots, fields = case
    plan = raw['response_plan']
    fields = list(fields)
    if mutation == 'missing_evidence':
        plan['evidence_refs'].pop()
    elif mutation == 'extra_adoption':
        plan['memory_refs'].append(list(observation.stable_key()))
    elif mutation == 'missing_adoption':
        plan['memory_refs'].pop()
    elif mutation == 'missing_candidate':
        fields[4] = pack_record(1)
    elif mutation == 'foreign_context':
        fields[2] = pack_record(91565,(99,))
    elif mutation == 'changed_query':
        fields[1] = (99,)
    elif mutation == 'visible_graph':
        fields[6] = pack_record(3,(1,2))
    else:
        fields[5] = pack_record(2)
    raw['generation']['trace'] = list(pack_record(91566,*fields))
    with pytest.raises(QueryMemoryOheEvidenceError):
        _verify(case)
