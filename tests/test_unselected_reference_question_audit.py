"""未选择指代只证明完整输入保持开放，事实与目标采用不得借用此合同。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError
from pure_integer_ai.experiments.reference_question_transition_audit import verify_reference_question_transition
from pure_integer_ai.experiments.reference_question_transition_audit import restore_reference_question_transition
from test_query_relation_arbitration import _pending_reference_fixture


def _fixture():
    """复用真实训练帧和开放投影；只以合成决策隔离本模块的拒绝条件。"""
    bridge, prepared, marker = _pending_reference_fixture()
    value = dict(state_key=dict(evidence=[], bindings=[]), evidence=[],
        response_plan=dict(claim_refs=[], claim_slot_ranges=[]),
        input_structure=dict(stable_key=prepared.stable_key(),
            relation_question_bindings=[item.stable_key() for item in prepared.relation_question_bindings]))
    trace = SimpleNamespace(trace=value, resolve_ref=lambda key: tuple(key))
    resolutions = tuple(SimpleNamespace(candidate_set=SimpleNamespace(stable_key=lambda i=i: (i,)),
        requires_resolution=1, status=2, winner_target_key=(), adoption_support_key=())
        for i in range(len(prepared.relation_question_bindings)))
    return bridge, prepared, marker, trace, resolutions


@pytest.mark.parametrize('status', (2, 4))
def test_all_unselected_trained_alternatives_preserve_complete_input(status):
    """OPEN与CONFLICT均不产生目标采用或事实，原全部候选保持。"""
    bridge, prepared, _marker, trace, resolutions = _fixture()
    original = prepared.integer_trace()
    for item in resolutions:
        item.status = status
    result = verify_reference_question_transition(trace, prepared, resolutions, reference_bridge=bridge)
    assert result == dict(initial_question_proofs=0, unselected_question_candidates=2,
        target_adoption_count=0, full_input_transition_matches=1, query_call_count=0, generation_call_count=0)
    assert prepared.integer_trace() == original


@pytest.mark.parametrize('drift', (
    'missing_resolution', 'duplicate_resolution', 'resolved', 'not_required', 'winner', 'support',
    'marker', 'partial_proof', 'one_binding', 'claim', 'claim_range', 'adoption_evidence',
    'adoption_binding', 'final_input', 'final_candidates'))
def test_incomplete_or_adopted_questions_cannot_use_unselected_contract(drift):
    """每个拒绝条件覆盖语义成员，不用放宽正例合同消除原失败。"""
    bridge, prepared, _marker, trace, resolutions = _fixture()
    if drift == 'missing_resolution':
        resolutions = resolutions[:-1]
    elif drift == 'duplicate_resolution':
        resolutions = (resolutions[0], resolutions[0])
    elif drift == 'resolved':
        resolutions[0].status = 3
    elif drift == 'not_required':
        resolutions[0].requires_resolution = 0
    elif drift == 'winner':
        resolutions[0].winner_target_key = (71,)
    elif drift == 'support':
        resolutions[0].adoption_support_key = (72,)
    elif drift == 'marker':
        bridge.relation_question_frames = bridge.relation_question_frames[:-1]
    elif drift == 'partial_proof':
        key = prepared.relation_question_bindings[0].stable_key()
        trace.trace['evidence'] = [dict(evidence_ref=(len(key), *key))]
    elif drift == 'one_binding':
        prepared = replace(prepared, relation_question_bindings=prepared.relation_question_bindings[:1])
    elif drift == 'claim':
        trace.trace['response_plan']['claim_refs'] = [(73,)]
    elif drift == 'claim_range':
        trace.trace['response_plan']['claim_slot_ranges'] = [dict(claim=(73,), start=0, end=1)]
    elif drift == 'adoption_evidence':
        trace.trace['state_key']['evidence'] = [EvidenceEntry((1,), (2,), evidence_key=(91599, 7)).stable_key()]
    elif drift == 'adoption_binding':
        trace.trace['state_key']['bindings'] = [BindingEntry((91599, 6), (74,)).stable_key()]
    elif drift == 'final_input':
        trace.trace['input_structure']['stable_key'] = (75,)
    else:
        trace.trace['input_structure']['relation_question_bindings'].reverse()
    with pytest.raises(QueryMemoryOheEvidenceError):
        verify_reference_question_transition(trace, prepared, resolutions, reference_bridge=bridge)


def test_all_existing_question_proofs_keep_the_original_transition_contract():
    """已证明路径保留原证明计数，不能被替换为候选资格。"""
    bridge, prepared, _marker, trace, resolutions = _fixture()
    trace.trace['evidence'] = [dict(evidence_ref=(len(key), *key))
        for key in (item.stable_key() for item in prepared.relation_question_bindings)]
    result = verify_reference_question_transition(trace, prepared, resolutions, reference_bridge=bridge)
    assert result['initial_question_proofs'] == 2
    assert 'unselected_question_candidates' not in result


def test_full_transition_record_keeps_original_report_and_current_input_identity():
    """新同次值保留完整输入，不给无采用路径制造Core采用资格。"""
    bridge, prepared, _marker, trace, resolutions = _fixture()
    restored = restore_reference_question_transition(trace, prepared, resolutions, reference_bridge=bridge)
    assert restored.report() == verify_reference_question_transition(
        trace, prepared, resolutions, reference_bridge=bridge)
    assert restored.final_input_key == prepared.stable_key()
    assert restored.adoption_evidence_keys == ()
