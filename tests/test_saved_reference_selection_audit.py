"""Cold scope selection must examine every input candidate before trace roles."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.query_state import BindingEntry
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError
from pure_integer_ai.experiments import saved_reference_selection_audit as audit
from test_reference_question_memory_view import _fixture
from test_trained_reference_resolution_cache import _reference_fixture


def _records(monkeypatch, *, ambiguous=False):
    bridge, projector, roles = _fixture()
    bridge.input_projector, bridge.core_runtime = projector, None
    prepared = bridge._attach_relation_question_binding(projector.project((99,33,22,88)))
    _runtime, previous, _target, _prop, _state = _reference_fixture()
    original = previous.reference
    occurrences, relation_map = [], {}
    for index, relation in enumerate(prepared.open_relations):
        ref = original.hypothesis
        hypothesis = MemoryObjectRef(ref.memory_space,ref.owner,ref.versions,ref.object_kind,(11,index))
        occurrence = replace(original,role=roles[0],hypothesis=hypothesis,ordinal=0,values=(99,))
        occurrences.append(occurrence)
        relation_map[hypothesis.stable_key()] = relation
    occurrences = tuple(sorted(occurrences,key=lambda item:item.stable_key()))
    targets = tuple(bridge.core_fact_index[item.frame.proposition].bindings[0].filler
                    for item in prepared.open_relations)
    runtime = SimpleNamespace(semantic_objects_for=lambda *_args: targets if ambiguous else targets[:1])
    monkeypatch.setattr(audit,'TrainedCrossTurnReferenceRuntime',lambda *_args:runtime)
    monkeypatch.setattr(audit,'read_current_reference_scope',lambda *_args:None)
    expected = occurrences if ambiguous else occurrences[:1]
    value = dict(state_key=dict(bindings=[]),
        input_role_occurrence_refs=[item.stable_key() for item in expected])
    trace = SimpleNamespace(trace=value,resolve_ref=lambda key:tuple(key))
    report = dict(projection=dict(question_bindings=[list(item.stable_key())
        for item in prepared.relation_question_bindings]),observation_ref=list(original.observation.stable_key()))
    contexts = (SimpleNamespace(role_occurrences=previous.antecedents,context_depth=0),)
    return bridge,prepared,occurrences,relation_map,trace,report,contexts


def _restore(records):
    bridge,prepared,occurrences,relation_map,trace,report,contexts = records
    return audit.restore_saved_reference_selection(SimpleNamespace(occurrence_index=None),
        trace,report,prepared,occurrences,contexts,relation_map,reference_bridge=bridge)


def test_complete_cold_candidates_select_one_scope_without_reading_a_trace_winner(monkeypatch):
    records = _records(monkeypatch)
    prepared = records[1]
    view,roles,relations,result = _restore(records)
    assert len(prepared.relation_question_bindings)==2
    assert len(view.relation_question_bindings)==len(roles)==len(relations)==1
    assert result['initial_reference_candidate_sets']==2
    assert result['initial_input_roles']==2 and result['grounded_question_selected']==1
    assert result['query_call_count']==result['memory_write_count']==0
    assert prepared==records[1]


def test_distinct_supported_objects_keep_all_original_scopes(monkeypatch):
    records = _records(monkeypatch,ambiguous=True)
    view,roles,relations,result = _restore(records)
    assert view==records[1] and roles==records[2] and relations==records[3]
    assert result['grounded_question_selected']==0
    assert result['initial_reference_candidate_sets']==2


@pytest.mark.parametrize('drift',('omit_role','omit_candidate','original_question','trace_role','fake_consensus'))
def test_selection_cannot_discard_a_candidate_or_borrow_trace_authority(monkeypatch,drift):
    records = list(_records(monkeypatch))
    if drift=='omit_role':records[2]=records[2][:-1]
    elif drift=='omit_candidate':records[3]={}
    elif drift=='original_question':records[5]['projection']['question_bindings'].pop()
    elif drift=='trace_role':records[4].trace['input_role_occurrence_refs']=[records[2][1].stable_key()]
    else:
        records[4].trace['state_key']['bindings']=[BindingEntry((91656,1),(1,),space=2).stable_key()]
    with pytest.raises(QueryMemoryOheEvidenceError):_restore(records)
