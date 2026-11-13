"""Teacher role substitutions preserve syntax, provenance and type limits."""
from copy import deepcopy
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record
from pure_integer_ai.cognition.understanding.query_reference_context import reference_marker_sources
from pure_integer_ai.cognition.understanding.query_relation_questions import SLOT_MODE_REFERENCE, TrainedRelationQuestionFrame
from pure_integer_ai.experiments.reference_role_course import (
    REFERENCE_ROLE_LEARNING_VERSION, compile_train_reference_role_record,
)
from pure_integer_ai.experiments.trained_relation_question_graph import (
    materialize_relation_question_graph, read_relation_question_graph,
)


def _course(index=0):
    parent = TrainedRelationQuestionFrame((701, 1), (702, 1), (703, 1),
        ((704, 1), (704, 2), (704, 3)), ((11,), (12,), (13,), (14,)))
    marker = TrainedRelationQuestionFrame((701, 2), (702, 1), (703, 2),
        ((704, 4),), ((19,), (15,)), (704, 5), (19,), SLOT_MODE_REFERENCE)
    record = dict(owner='teacher', split='train', license_id='CC0-1.0',
        parent_frame_key=list(parent.stable_key()), marker_frame_key=list(marker.stable_key()),
        reference_role_key=list(parent.roles[index]), allowed_object_kinds=[4, 16])
    return record, (parent, marker), ((parent.predicate_key, parent.roles[index], (4, 16)),)


@pytest.mark.parametrize('index', (0, 1, 2))
def test_single_role_substitution_keeps_all_other_roles_and_literal_intervals(tmp_path, index):
    record, frames, contracts = _course(index)
    original = deepcopy(record)
    parent, marker = frames
    result = compile_train_reference_role_record(record, trained_frames=frames, role_contracts=contracts)
    assert result.roles == parent.roles[:index] + parent.roles[index + 1:]
    assert result.gaps == (*parent.gaps[:index], parent.gaps[index] + marker.wh_values + parent.gaps[index + 1], *parent.gaps[index + 2:])
    assert result.predicate_key == parent.predicate_key and result.operation_key == parent.operation_key
    assert result.is_reference_slot and result.answer_role_key == parent.roles[index]
    fields = unpack_record(result.source_case_key, REFERENCE_ROLE_LEARNING_VERSION)
    assert len(fields[0]) == 8 and fields[1:] == (parent.source_case_key, marker.source_case_key, parent.roles[index], (4, 16))
    assert compile_train_reference_role_record(record, trained_frames=frames, role_contracts=contracts) == result
    assert record == original
    database = tmp_path / 'references.sqlite3'
    merged = tuple(sorted((*frames, result), key=lambda item: item.stable_key()))
    materialize_relation_question_graph(merged, database, source_identity=(705, 1), license_identity=(706, 1))
    restored, _meta = read_relation_question_graph(database)
    assert restored == merged
    assert reference_marker_sources(marker.wh_values, parent.roles[index], parent.predicate_key, restored) == (result.stable_key(),)
    assert not reference_marker_sources((99,), parent.roles[index], parent.predicate_key, restored)
    assert not reference_marker_sources(marker.wh_values, (704, 9), parent.predicate_key, restored)


@pytest.mark.parametrize('drift', ('owner', 'split', 'license', 'parent', 'marker', 'role', 'kinds', 'bool', 'extra'))
def test_unlicensed_or_changed_teacher_contract_is_rejected(drift):
    record, frames, contracts = _course()
    if drift == 'owner': record['owner'] = 'evaluator'
    elif drift == 'split': record['split'] = 'held_out'
    elif drift == 'license': record['license_id'] = 'unknown'
    elif drift == 'parent': record['parent_frame_key'] = [1]
    elif drift == 'marker': record['marker_frame_key'] = list(frames[0].stable_key())
    elif drift == 'role': record['reference_role_key'] = [704, 9]
    elif drift == 'kinds': record['allowed_object_kinds'].append(18)
    elif drift == 'bool': record['allowed_object_kinds'] = [True]
    else: record['target_answer'] = [9]
    with pytest.raises(ValueError):
        compile_train_reference_role_record(record, trained_frames=frames, role_contracts=contracts)


def test_missing_or_ambiguous_physical_role_contract_does_not_grant_permission():
    record, frames, contracts = _course()
    for values in ((), contracts + contracts):
        with pytest.raises(ValueError, match='mounted type contract'):
            compile_train_reference_role_record(record, trained_frames=frames, role_contracts=values)


def test_literal_marker_or_nonunique_parent_source_cannot_acquire_a_slot():
    record, frames, contracts = _course()
    parent, marker = frames
    repeated = replace(parent, gaps=((19,), *parent.gaps[1:]))
    record['parent_frame_key'] = list(repeated.stable_key())
    with pytest.raises(ValueError, match='another literal'):
        compile_train_reference_role_record(record, trained_frames=(repeated, marker), role_contracts=contracts)
    record['parent_frame_key'] = list(parent.stable_key())
    rival = replace(parent, operation_key=(702, 2))
    with pytest.raises(ValueError, match='uniquely recoverable'):
        compile_train_reference_role_record(record, trained_frames=(*frames, rival), role_contracts=contracts)
