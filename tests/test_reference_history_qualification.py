"""完整未决历史可以恢复候选，但不能被解释为已读取或已采用。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError
from pure_integer_ai.experiments.query_reference_memory_audit import verify_reference_memory
from pure_integer_ai.experiments.reference_history_qualification import qualify_reference_history
from pure_integer_ai.experiments.saved_reference_context_audit import restore_reference_candidate_set
from pure_integer_ai.experiments.trained_response_context import DeliveredGenericResponseContext
from test_reference_memory_audit_r59 import _records
from test_trained_reference_resolution_cache import _reference_fixture


def _case():
    """以真实H-00/H-04未选择决策及有类型context隔离历史分账合同。"""
    runtime, candidate, _target, _proposition, _prepared = _reference_fixture()
    resolution = runtime.resolve(candidate, (), core_support=frozenset(), timestamp_seq=1)
    role = candidate.antecedents[0]
    context = DeliveredGenericResponseContext(role.observation, role.hypothesis,
        role.source_ref, role.scope_key, role.proof_ref, role.observation,
        role.delivery_context, (), 0, (role,))
    trace = SimpleNamespace(trace=dict(termination=3, response_plan=dict(claim_refs=[],
        claim_evidence_refs=[], claim_slot_ranges=[], event_refs=[])), hops=lambda: ())
    evidence = (candidate.query_evidence(), *resolution.query_evidence())
    kwargs = dict(trace=trace, evidence=evidence,
        bindings={candidate.query_binding().stable_key(),
                  *(item.stable_key() for item in resolution.query_bindings())},
        candidate_sets=(candidate,), resolutions=(resolution,), nonfactual_plan=True,
        adopted_hypothesis_keys=frozenset(((991,),)), generic_adoption=dict(visible_graph_slot_count=0),
        error=QueryMemoryOheEvidenceError)
    return context, kwargs


def test_complete_open_history_restores_every_antecedent_without_consuming_it():
    """零历史读取仍恢复完整候选及其深度，证据和实际消费集合都不改写。"""
    context, kwargs = _case()
    candidate, = kwargs['candidate_sets']
    complete = qualify_reference_history((), (context,), **kwargs)
    roles = {item.stable_key(): item for item in context.role_occurrences}
    restored = restore_reference_candidate_set(candidate.stable_key(),
        {candidate.reference.stable_key(): candidate.reference}, roles, complete)
    assert restored == candidate
    assert kwargs['trace'].hops() == ()
    assert kwargs['resolutions'][0].winner_target_key == ()
    assert kwargs['adopted_hypothesis_keys'] == frozenset(((991,),))


@pytest.mark.parametrize('drift', ('duplicate_context', 'missing_context', 'consumed_role', 'consumed_input',
    'factual', 'termination', 'claim', 'claim_evidence', 'claim_range', 'visible_event', 'visible_graph',
    'missing_resolution', 'duplicate_resolution', 'selected', 'adopted_reference',
    'adoption_evidence', 'adoption_binding'))
def test_missing_or_adopted_history_cannot_borrow_unselected_qualification(drift):
    """实际读取、事实、目标及可见图采用的原门不因保留候选而放宽。"""
    context, kwargs = _case()
    complete = (context,)
    if drift == 'duplicate_context':
        complete = (context, context)
    elif drift == 'missing_context':
        return _assert_missing_consumed_context(context, kwargs)
    elif drift == 'consumed_role':
        kwargs['evidence'] += (context.role_occurrences[0].query_evidence(),)
    elif drift == 'consumed_input':
        kwargs['trace'].hops = lambda: (SimpleNamespace(space=2, predicate_key=(1,),
            proposition_key=context.input_observation.stable_key()),)
    elif drift == 'factual':
        kwargs['nonfactual_plan'] = False
    elif drift == 'termination':
        kwargs['trace'].trace['termination'] = 1
    elif drift in ('claim', 'claim_evidence', 'claim_range', 'visible_event'):
        key = {'claim': 'claim_refs', 'claim_evidence': 'claim_evidence_refs',
               'claim_range': 'claim_slot_ranges', 'visible_event': 'event_refs'}[drift]
        kwargs['trace'].trace['response_plan'][key] = [(992,)]
    elif drift == 'visible_graph':
        kwargs['generic_adoption']['visible_graph_slot_count'] = 1
    elif drift == 'missing_resolution':
        kwargs['resolutions'] = ()
    elif drift == 'duplicate_resolution':
        kwargs['resolutions'] *= 2
    elif drift == 'selected':
        _candidate, selected, _unused = _records()
        kwargs['resolutions'] = (selected,)
    elif drift == 'adopted_reference':
        kwargs['adopted_hypothesis_keys'] = frozenset((kwargs['candidate_sets'][0].reference.hypothesis.stable_key(),))
    elif drift == 'adoption_evidence':
        kwargs['evidence'] += (EvidenceEntry((1,), (2,), evidence_key=(91599, 7)),)
    else:
        kwargs['bindings'].add(BindingEntry((91599, 6), (993,)).stable_key())
    with pytest.raises(QueryMemoryOheEvidenceError):
        qualify_reference_history((), complete, **kwargs)


def _assert_missing_consumed_context(context, kwargs):
    """完整历史不能删除实际已消费的context，空集合也不得绕过包含门。"""
    with pytest.raises(QueryMemoryOheEvidenceError, match='actually consumed'):
        qualify_reference_history((context,), (), **kwargs)


def test_complete_members_do_not_relax_the_existing_reference_evidence_gate():
    """分账后仍调用原逐项Evidence/binding/root核验，缺项保持失败。"""
    context, kwargs = _case()
    complete = qualify_reference_history((), (context,), **kwargs)
    candidate, = kwargs['candidate_sets']
    _old_candidate, _old_resolution, readers = _records()
    readers.update(contexts=complete, evidence=kwargs['evidence'], bindings=kwargs['bindings'],
        roots=(replace(candidate.seed(1)[0], status=2),),
        plan_keys={item.stable_key() for item in kwargs['evidence']})
    assert verify_reference_memory((candidate,), kwargs['resolutions'], **readers) == kwargs['evidence']
    readers['bindings'] = set()
    with pytest.raises(QueryMemoryOheEvidenceError):
        verify_reference_memory((candidate,), kwargs['resolutions'], **readers)


def test_no_additional_history_preserves_existing_factual_consumers():
    """不新增历史的调用保留原合同，不强制改为澄清。"""
    context, kwargs = _case()
    assert qualify_reference_history((context,), None, **kwargs) == (context,)
    kwargs['nonfactual_plan'] = False
    kwargs['trace'].hops = lambda: (SimpleNamespace(space=2, predicate_key=(1,),
        proposition_key=context.observation.stable_key()),)
    assert qualify_reference_history((context,), (context,), **kwargs) == (context,)
