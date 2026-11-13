"""Compound Dialogue Evidence keeps the per-atom limit and full source identity."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT, OBJECT_MINIMAL_INSTRUCTION, OBJECT_PROPOSITION,
    OBJECT_REPRESENTATION, OBJECT_ROLE, ObjectIdentity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.compound_dialogue_evidence_audit import verify_compound_dialogue_evidence
from pure_integer_ai.experiments.query_memory_ohe_audit import (
    QueryMemoryOheEvidenceError, _key, _rows, verify_query_memory_ohe,
)
from pure_integer_ai.experiments.trained_query_evidence import dialogue_query_evidence
from pure_integer_ai.experiments.trained_relation_graph_runtime import GraphRelationGeneration, RelationSurfaceFrame
from test_query_memory_ohe_audit import _fixture


@pytest.fixture(scope='module')
def records():
    _captured, _declarations, manifest, _evidence = _fixture()
    source = manifest.source

    def identity(kind, number):
        return ObjectIdentity(kind, (number,), source.owner, source.versions)

    claim = identity(OBJECT_PROPOSITION, 101)
    frame = RelationSurfaceFrame(claim, identity(OBJECT_CONCEPT, 102),
        (identity(OBJECT_ROLE, 103),), ('', ''), 19, 0, 1)
    atom = (1, 2, 3) * 117000
    generation_trace = (2, 3, *(value for part in (atom, atom, atom) for value in (len(part), *part)))
    generated = GraphRelationGeneration('x', claim, 19, 1,
        identity(OBJECT_MINIMAL_INSTRUCTION, 104), (identity(OBJECT_REPRESENTATION, 105),), generation_trace)
    physical = SimpleNamespace(proposition=SimpleNamespace(definition=SimpleNamespace(source=source, proposition=claim)),
        hypothesis=SimpleNamespace(scope=document_scope(source)))
    core = SimpleNamespace(active_surface_frames=lambda: (frame,), generation_input=lambda _claim: physical)
    entry = dialogue_query_evidence(claim, frame, physical, generated)
    value = dict(generation=dict(kind=1, frame_proposition=claim.stable_key(), slot_count=1,
        connector=generated.connector.stable_key(), representations=tuple(
            item.stable_key() for item in generated.representations),
        trace=generation_trace, surface_values=(120,)),
        response_plan=dict(claim_refs=(claim.stable_key(),)))
    return entry, value, core, frame


def _verify(entry, value, core, *, maximum=256):
    trace = SimpleNamespace(trace=value)
    return verify_compound_dialogue_evidence(entry.stable_key(), trace, core_runtime=core,
        key_reader=_key, rows_reader=_rows, max_state_members=maximum, error=QueryMemoryOheEvidenceError)


def test_whole_key_is_compound_and_all_original_atoms_remain_bounded(records):
    entry, value, core, _frame = records
    with pytest.raises(QueryMemoryOheEvidenceError, match='bounded integer'):
        _key(entry.stable_key(), 'original oversized key')
    assert _verify(entry, value, core) == entry.stable_key()


@pytest.mark.parametrize('field,new', (('source_ref', (99,)), ('hypothesis_key', (98,)),
    ('evidence_key', (97,)), ('scope_key', (96,)), ('payload_key', (95,)),
    ('supersedes_key', (94,)), ('space', 2), ('polarity', 3), ('trust', 2)))
def test_changed_complete_members_never_borrow_an_original_generation(records, field, new):
    entry, value, core, _frame = records
    with pytest.raises((ValueError, QueryMemoryOheEvidenceError)):
        _verify(replace(entry, **{field:new}), value, core)


@pytest.mark.parametrize('corruption', ('kind', 'trace_tag', 'trace_count', 'truncated',
    'trailing', 'huge_atom', 'slot_bool', 'frame', 'duplicate_frame', 'missing_core'))
def test_unregistered_or_unbounded_composites_fail_closed(records, corruption):
    entry, original, core, frame = records
    generation = dict(original['generation'])
    value = dict(original, generation=generation)
    if corruption == 'kind':
        generation['kind'] = 3
    elif corruption == 'trace_tag':
        generation['trace'] = (4, *generation['trace'][1:])
    elif corruption == 'trace_count':
        generation['trace'] = (2, 4, *generation['trace'][2:])
    elif corruption == 'truncated':
        generation['trace'] = generation['trace'][:-1]
    elif corruption == 'trailing':
        generation['trace'] = (*generation['trace'], 0)
    elif corruption == 'huge_atom':
        atom = (0,) * 1000001
        generation['trace'] = (2, 3, len(atom), *atom, 1, 0, 1, 0)
    elif corruption == 'slot_bool':
        generation['slot_count'] = True
    elif corruption == 'frame':
        core = SimpleNamespace(active_surface_frames=lambda: ())
    elif corruption == 'duplicate_frame':
        core = SimpleNamespace(active_surface_frames=lambda: (frame,frame))
    else:
        core = None
    with pytest.raises((ValueError, QueryMemoryOheEvidenceError)):
        _verify(entry, value, core)


def test_existing_member_count_budget_still_applies(records):
    entry, value, core, _frame = records
    with pytest.raises(QueryMemoryOheEvidenceError, match='member count'):
        _verify(entry, value, core, maximum=2)


def test_actual_ohe_reader_checks_both_state_and_plan_compound_keys(records):
    entry, original, core, _frame = records
    captured, declarations, manifest, _evidence = _fixture()
    value = captured['trace']
    value['generation'] = dict(original['generation'])
    value['integer_key_table'].append(list(entry.stable_key()))
    value['state_key']['evidence'].append(len(value['integer_key_table']))
    value['response_plan']['claim_refs'] = original['response_plan']['claim_refs']
    value['response_plan']['evidence_refs'].append(list(entry.stable_key()))
    result = verify_query_memory_ohe(captured,
        declaration_reader=lambda ref: (declarations[ref],) if ref in declarations else (),
        current_manifest_reader=lambda source: manifest, source_hash_reader=lambda source:19,
        core_runtime=core)
    assert result['status'] == 'MATCHED_DURABLE_INTAKE_OHE_QUERYSTATE_AND_PLAN_REFERENCES'
