"""Full trained requests must retain readers before ordinary WH narrowing."""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.identity import OwnerScope, SourceRef, VersionBundle
from pure_integer_ai.cognition.understanding.query_input_structure import TrainedInputStructureProjector
from pure_integer_ai.cognition.understanding.query_operation_requests import FRAME, OperationRequestFrame, match_operation_request_frames
from pure_integer_ai.cognition.understanding.query_operation_request_scope import match_pending_operation_request_scopes
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
from test_query_input_provenance import _structure
from test_reference_question_memory_view import _fixture


def _bridge(*, ambiguous=False):
    bridge, base, _roles = _fixture()
    questions = tuple(replace(item,gaps=((55,77,99,33),(88,)))
                      for item in base.slot_question_frames)
    open_frames = tuple(replace(item,gaps=questions[0].gaps)
                        for item in base.open_role_frames)
    bridge.relation_question_frames = questions
    other = _structure(101)
    # A request's prefix supplies its reader; its grammar also contains an
    # ordinary WH parse for a different predicate and proposition family.
    other = replace(other, members=tuple(replace(member, values=(77,))
        if member.projection_kind == 1 and member.member_ordinal == 0 else member
        for member in other.members))
    structures = (*base.structures, other)
    if ambiguous:
        rival = replace(other, members=tuple(replace(member,
            candidate_key=(*member.candidate_key[:-1],member.candidate_key[-1]+1))
            if member.projection_kind == 1 and member.member_ordinal == 0 else member
            for member in other.members))
        structures = (*structures, rival)
    bridge.input_projector = TrainedInputStructureProjector(structures,
        open_role_frames=open_frames, slot_question_frames=questions)
    bridge._relation_question_open_frames = open_frames
    digest = tuple(range(32))
    reader = other.members[0]
    frame = OperationRequestFrame((902,1), SourceRef(FRAME,int.from_bytes(bytes(digest[:8]),'big'),0,
        OwnerScope(),VersionBundle()),digest,(903,1),(904,1),
        ((1,reader.predicate_key,reader.role_key,(reader.object_kind,)),),
        ((55,),(99,33,22,88)))
    bridge.operation_request_frames = (frame,)
    bridge.input_discourse_operation_modes = {frame.operation_key:0}
    bridge.assertion_scope_frames = ()
    bridge._activate_kdconv_graph_objects = lambda keys: None
    bridge._augment_kdconv_input = lambda structure: structure
    bridge._aggregate_input_provenance = lambda structure: structure
    return bridge, frame


def test_complete_request_keeps_reader_lost_by_real_ordinary_wh_filter():
    bridge, frame = _bridge()
    raw = bridge.input_projector.project((55,77,99,33,22,88))
    original, = match_operation_request_frames((frame,),raw)
    narrowed = bridge._filter_relation_question_open_scope(raw)
    assert narrowed.open_relations
    assert not match_operation_request_frames((frame,),narrowed)
    prepared = bridge.prepare_input_structure(''.join(map(chr,raw.token_values)))
    assert prepared.operation_request_bindings == (original,)
    assert not prepared.relation_question_bindings and not prepared.open_relations
    assert prepared.tokens == raw.tokens and prepared.spans == raw.spans
    assert prepared.source_ref == raw.source_ref
    assert {member.stable_key() for member in prepared.semantic_candidates} == set(original.readings[0][3])
    # The query entry repeats binding against the prepared view.
    assert bridge._attach_operation_request_binding(prepared) == prepared
    assert bridge._attach_relation_question_binding(prepared) == prepared
    with pytest.raises(ValueError):
        bridge._attach_operation_request_binding(replace(prepared,semantic_candidates=()))


@pytest.mark.parametrize('ambiguous,reader',[(False,78),(True,77)])
def test_unknown_or_ambiguous_request_retains_complete_pending_scope(ambiguous,reader):
    bridge, frame = _bridge(ambiguous=ambiguous)
    values = (55,reader,99,33,22,88)
    raw = bridge.input_projector.project(values)
    expected = match_pending_operation_request_scopes((frame,),raw)
    assert expected and not match_operation_request_frames((frame,),raw)
    prepared = bridge.prepare_input_structure(''.join(map(chr,values)))
    assert not prepared.operation_request_bindings and not prepared.operation_keys
    assert not prepared.relation_question_bindings and not prepared.relation_candidates
    assert match_pending_operation_request_scopes((frame,),prepared) == expected
    assert prepared.tokens == raw.tokens and prepared.spans == raw.spans
    members = {key for scope in expected for row in scope.readings for key in row[3]}
    assert {member.stable_key() for member in prepared.semantic_candidates} == members
    assert len(members) == (2 if ambiguous else 0)


@pytest.mark.parametrize('values',[(5,55,77,99,33,22,88),(55,77,99,33,22,88,6)])
def test_unlicensed_outer_input_does_not_acquire_request_scope(values):
    bridge, frame = _bridge()
    prepared = bridge.prepare_input_structure(''.join(map(chr,values)))
    assert not prepared.operation_request_bindings
    assert not match_pending_operation_request_scopes((frame,),prepared)
