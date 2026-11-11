"""指代历史、决策及 Plan 必须保留完整来源、深度、立场和成员。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.query_state import ROOT_EXPANDED
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError
from pure_integer_ai.experiments.query_reference_memory_audit import verify_reference_memory
from pure_integer_ai.experiments.saved_reference_context_audit import (
    reference_resolution_header, restore_reference_candidate_set,
)
from test_trained_reference_resolution_cache import _reference_fixture


def _records():
    """复用真实 H-00/H-04 解析夹具，所有正反例保持同一个完整决策。"""
    runtime, candidate, _target, proposition, _prepared = _reference_fixture()
    resolution = runtime.resolve(candidate, (), core_support=frozenset({proposition}), timestamp_seq=1)
    contexts = (SimpleNamespace(role_occurrences=candidate.antecedents, context_depth=0),)
    evidence = (candidate.query_evidence(), *resolution.query_evidence())
    root = replace(candidate.seed(1)[0], status=ROOT_EXPANDED)
    return candidate, resolution, dict(input_roles=(candidate.reference,), contexts=contexts,
        evidence=evidence, bindings={candidate.query_binding().stable_key(),
                                   *(item.stable_key() for item in resolution.query_bindings())},
        roots=(root,), plan_keys={item.stable_key() for item in evidence},
        plan_refs={candidate.reference.observation,candidate.reference.hypothesis}, has_plan=True,
        error=QueryMemoryOheEvidenceError)


def test_cold_reference_members_round_trip_and_match_actual_decision_state():
    """候选及决策使用规范整数全记录，结构历史不产生额外证据。"""
    candidate, resolution, kwargs = _records()
    assert verify_reference_memory((candidate,), (resolution,), **kwargs) == kwargs['evidence']
    inputs = {candidate.reference.stable_key():candidate.reference}
    roles = {item.stable_key():item for item in candidate.antecedents}
    restored = restore_reference_candidate_set(candidate.stable_key(), inputs, roles, kwargs['contexts'])
    assert restored == candidate
    assert reference_resolution_header(resolution.stable_key()) == (candidate.stable_key(), 1, 3, 1)
    with pytest.raises(QueryMemoryOheEvidenceError, match='depth'):
        restore_reference_candidate_set(replace(candidate,context_depths=(2,)).stable_key(),
                                         inputs,roles,kwargs['contexts'])
    with pytest.raises((ValueError,QueryMemoryOheEvidenceError)):
        reference_resolution_header(resolution.stable_key()[:-1])


def test_reference_set_cannot_omit_an_independently_recovered_competing_role():
    """已验证的候选集合必须覆盖全部冷历史角色，不能截掉另一候选换唯一 winner。"""
    candidate,resolution,kwargs = _records()
    first = candidate.antecedents[0]
    second = replace(first, occurrence=replace(first.occurrence,components=(994,)))
    kwargs['contexts'] = (SimpleNamespace(role_occurrences=tuple(sorted((first,second),key=lambda item:item.stable_key())),
                                         context_depth=0),)
    with pytest.raises(QueryMemoryOheEvidenceError,match='omits a cold antecedent'):
        verify_reference_memory((candidate,),(resolution,),**kwargs)
    with pytest.raises(QueryMemoryOheEvidenceError,match='omits a cold antecedent'):
        restore_reference_candidate_set(candidate.stable_key(), {candidate.reference.stable_key():candidate.reference},
            {item.stable_key():item for item in (first,second)}, kwargs['contexts'])


@pytest.mark.parametrize('field,value', (('polarity',2),('source_ref',(91,)),
                                       ('scope_key',(92,)),('supersedes_key',(93,))))
def test_changed_reference_evidence_never_borrows_the_original_decision(field,value):
    """同 evidence id 不能掩盖来源、作用域、立场或替代链漂移。"""
    candidate,resolution,kwargs = _records()
    changed = replace(kwargs['evidence'][-1], **{field:value})
    kwargs['evidence'] = (*kwargs['evidence'][:-1],changed)
    with pytest.raises(QueryMemoryOheEvidenceError):
        verify_reference_memory((candidate,),(resolution,),**kwargs)


@pytest.mark.parametrize('missing', ('candidate_binding','expanded_root','plan_member','current_memory'))
def test_missing_reference_consumers_fail_instead_of_counting_correct_answers(missing):
    """完整成员和当前实际消费者缺一项就拒绝，不用计数或答案补齐。"""
    candidate,resolution,kwargs = _records()
    if missing == 'candidate_binding':
        kwargs['bindings'].remove(candidate.query_binding().stable_key())
    elif missing == 'expanded_root':
        kwargs['roots'] = ()
    elif missing == 'plan_member':
        kwargs['plan_keys'].remove(kwargs['evidence'][-1].stable_key())
    else:
        kwargs['plan_refs'].remove(candidate.reference.observation)
    with pytest.raises(QueryMemoryOheEvidenceError):
        verify_reference_memory((candidate,),(resolution,),**kwargs)
