"""Connected question scopes must bind facts without dropping any input."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, STRUCTURE_CLOSED, STRUCTURE_OPEN,
    TrainedInputMember, TrainedInputStructure, TrainedInputStructureProjector,
    scope_local_conjunction_candidate_keys,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    TrainedRelationQuestionFrame,
)
from pure_integer_ai.cognition.understanding.query_scoped_relation_questions import (
    match_scoped_relation_question_frames,
)
from pure_integer_ai.experiments.conjoined_relation_question_course import (
    MODE_CONJOINED_RELATION_QUERY, compile_train_conjoined_question,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
from pure_integer_ai.experiments.trained_input_discourse_conjunction_graph import (
    TrainedInputDiscourseConjunction,
    materialize_input_discourse_conjunction_graph,
)
from pure_integer_ai.experiments.trained_input_discourse_graph import (
    TrainedInputDiscourseOperator,
    materialize_input_discourse_operator_graph,
)
from pure_integer_ai.experiments.trained_relation_question_graph import (
    MODE_RELATION_PAIR_QUERY,
    materialize_relation_question_graph,
)


QUESTION_OPERATION = (92004, 1)
JOIN_OPERATION = (92004, 9)
JOIN = TrainedInputDiscourseOperator(JOIN_OPERATION, (99,))


def _fact(index, left, right, *, left_identity=None, right_identity=None):
    proposition, predicate, source = (1, index), (5, index), (9, index)
    left_identity = left_identity or (4, left)
    right_identity = right_identity or (4, right)
    members = (
        TrainedInputMember((left,), left_identity, 4, PROJECTION_FILLER,
                           proposition, predicate, source, 0, 1, 0, (8, 1)),
        TrainedInputMember((80 + index,), predicate, 5, PROJECTION_PREDICATE,
                           proposition, predicate, source, 1, 2, 1),
        TrainedInputMember((right,), right_identity, 4, PROJECTION_FILLER,
                           proposition, predicate, source, 2, 3, 2, (8, 2)),
    )
    return TrainedInputStructure(proposition, predicate, source, (7, index), members)


def _frame(index):
    return TrainedRelationQuestionFrame(
        (92002, index), QUESTION_OPERATION, (5, index), ((8, 1), (8, 2)),
        ((70,), (40 + index,), (60,)))


def _project(values=None, *, facts=None, frames=None):
    facts = facts or (_fact(1, 11, 22), _fact(2, 22, 33))
    frames = frames or tuple(_frame(index) for index in range(1, len(facts) + 1))
    if values is None:
        values = (70, 11, 41, 22, 60, 99, 70, 22, 42, 33, 60)
    projector = TrainedInputStructureProjector(facts, discourse_operators=(JOIN,))
    return projector.project(values), frames


def _bridge(frames):
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = frames
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_conjunctions = (TrainedInputDiscourseConjunction(JOIN_OPERATION),)
    bridge.input_discourse_operation_modes = {
        QUESTION_OPERATION: MODE_RELATION_PAIR_QUERY,
        JOIN_OPERATION: MODE_CONJOINED_RELATION_QUERY,
    }
    bridge.input_discourse_fact_query_modes = {}
    return bridge


def _match(projected, frames):
    return match_scoped_relation_question_frames(
        frames, projected, frozenset({JOIN_OPERATION}))


@pytest.mark.parametrize("count", [2, 3])
def test_two_and_three_connected_questions_keep_all_current_fact_bindings(count):
    facts = tuple(_fact(index, 11 * index, 11 * (index + 1))
                  for index in range(1, count + 1))
    values = tuple(value for index in range(1, count + 1)
                   for value in ((*((99,) if index > 1 else ()),
                                  70, 11 * index, 40 + index, 11 * (index + 1), 60)))
    projected, frames = _project(values, facts=facts)
    matches = _match(projected, frames)
    assert len(matches) == count
    bridge = _bridge(frames)
    prepared = bridge._attach_relation_question_binding(projected)
    assert prepared.relation_question_bindings == matches
    assert prepared.operation_keys == tuple(sorted({QUESTION_OPERATION, JOIN_OPERATION}))
    assert bridge._attach_relation_question_binding(prepared) == prepared
    memory_view = bridge.active_memory_input_structure(prepared)
    assert len(memory_view.relation_candidates) == count
    assert memory_view.relation_question_bindings == matches
    assert {item.proposition_key for item in memory_view.relation_candidates} == {
        item.proposition_key for item in facts}
    assert len(scope_local_conjunction_candidate_keys(
        prepared, frozenset({JOIN_OPERATION}))) == count - 1


@pytest.mark.parametrize("values", [
    (70, 22, 41, 11, 60, 99, 70, 22, 42, 33, 60),  # swapped roles
    (70, 11, 41, 22, 60, 99, 70, 22, 42, 60),  # missing role
    (70, 11, 41, 22, 60, 99, 70, 22, 42, 33, 60, 88),  # extra suffix
    (88, 70, 11, 41, 22, 60, 99, 70, 22, 42, 33, 60),  # extra prefix
    (70, 11, 41, 22, 60, 98, 70, 22, 42, 33, 60),  # untrained join
    (70, 11, 41, 22, 60, 99, 70, 33, 42, 44, 60),  # unadmitted fact
])
def test_partial_questions_cannot_fall_back_to_fact_conjunction(values):
    projected, frames = _project(values)
    assert _match(projected, frames) == ()
    bridge = _bridge(frames)
    prepared = bridge._attach_relation_question_binding(projected)
    assert not prepared.relation_question_bindings
    assert not scope_local_conjunction_candidate_keys(prepared, frozenset({JOIN_OPERATION}))
    if prepared.discourse_compositions:
        assert all(item.state == STRUCTURE_OPEN for item in prepared.discourse_compositions)
        memory = bridge.active_memory_input_structure(prepared)
        assert not memory.relation_candidates and not memory.semantic_candidates


def test_surface_equality_cannot_supply_shared_graph_identity():
    projected, frames = _project(facts=(
        _fact(1, 11, 22), _fact(2, 22, 33, left_identity=(4, 222))))
    assert _match(projected, frames) == ()
    assert not _bridge(frames)._attach_relation_question_binding(
        projected).relation_question_bindings


def test_competing_frame_and_source_keep_the_whole_question_unknown():
    projected, frames = _project()
    competing = replace(frames[1], source_case_key=(92002, 22))
    assert _match(projected, (*frames, competing)) == ()
    projected, frames = _project(facts=(
        _fact(1, 11, 22), _fact(2, 22, 33), _fact(3, 22, 33)))
    assert _match(projected, frames) == ()


def test_literal_fact_conjunction_still_uses_its_exact_trained_members():
    projected, frames = _project((11, 81, 22, 99, 22, 82, 33))
    prepared = _bridge(frames)._attach_relation_question_binding(projected)
    assert not prepared.relation_question_bindings
    assert all(item.state == STRUCTURE_CLOSED for item in prepared.discourse_compositions)
    assert len(scope_local_conjunction_candidate_keys(
        prepared, frozenset({JOIN_OPERATION}))) == 1


def test_input_binding_cannot_be_reused_after_tampering():
    projected, frames = _project()
    prepared = _bridge(frames)._attach_relation_question_binding(projected)
    changed = replace(frames[0], gaps=((71,), (41,), (60,)))
    with pytest.raises(ValueError, match="does not match"):
        _bridge((changed, frames[1]))._attach_relation_question_binding(prepared)


def _teacher(index, left, right):
    return SimpleNamespace(
        seed_id=f"teacher-{index}", split="train", label_owner="teacher",
        sample_role="support", license_id="CC0-1.0", relation_kind=14,
        endpoints=tuple(SimpleNamespace(surface_fragment=name, object_kind=6, local_id=number)
                        for name, number in (left, right)),
        bindings=(SimpleNamespace(role_kind=31), SimpleNamespace(role_kind=32)),
    )


def _compile(seeds, **overrides):
    arguments = {
        "source_case_key": (92002, 50), "question_operation_key": QUESTION_OPERATION,
        "conjunction_operation_key": JOIN_OPERATION, "conjunction_surface": "同时",
        "source_identity": (92026, 50),
        "license_identity": tuple(map(ord, "CC0-1.0")),
    }
    arguments.update(overrides)
    return compile_train_conjoined_question(
        seeds, "甲和乙有什么关系？同时乙和丙有什么关系？", **arguments)


def test_train_compiler_retains_syntax_sources_and_no_filler_answers():
    compiled = _compile((_teacher(1, ("甲", 1), ("乙", 2)),
                         _teacher(2, ("乙", 2), ("丙", 3))))
    assert len(compiled.frames) == 2
    assert compiled.operator.surface_values == tuple(map(ord, "同时"))
    assert compiled.contract.mode_key == MODE_CONJOINED_RELATION_QUERY
    assert {item.source_case_key for item in compiled.frames} == {
        (92002, 50, 1), (92002, 50, 2)}
    assert not {ord("甲"), ord("乙"), ord("丙")} & {
        value for frame in compiled.frames for gap in frame.gaps for value in gap}


@pytest.mark.parametrize("field,value", [
    ("split", "held_out"), ("label_owner", "evaluator"),
    ("sample_role", "refute"),
])
def test_train_compiler_rejects_non_teacher_input(field, value):
    first, second = _teacher(1, ("甲", 1), ("乙", 2)), _teacher(2, ("乙", 2), ("丙", 3))
    setattr(second, field, value)
    with pytest.raises(ValueError, match="licensed train teachers"):
        _compile((first, second))


def test_train_compiler_rejects_unlicensed_source_identity():
    with pytest.raises(ValueError, match="licensed source identity"):
        _compile((_teacher(1, ("甲", 1), ("乙", 2)),
                  _teacher(2, ("乙", 2), ("丙", 3))),
                 license_identity=(1,))


def test_train_compiler_rejects_surface_link_without_identity_link():
    with pytest.raises(ValueError, match="no shared graph endpoint"):
        _compile((_teacher(1, ("甲", 1), ("乙", 2)),
                  _teacher(2, ("乙", 20), ("丙", 3))))


def test_unknown_endpoint_ids_cannot_form_a_false_shared_graph_link():
    first, second = _teacher(1, ("甲", 1), ("乙", None)), _teacher(2, ("乙", None), ("丙", 3))
    with pytest.raises(ValueError, match="independently materialized"):
        _compile((first, second))


@pytest.mark.parametrize("changed", ["source", "license"])
def test_scoped_graph_provenance_mismatch_is_rejected_before_core_open(tmp_path, changed):
    source, license_key = (92026, 70), tuple(map(ord, "CC0-1.0"))
    materialize_relation_question_graph(
        (_frame(1), _frame(2)), tmp_path / "questions.sqlite3",
        source_identity=source, license_identity=license_key)
    materialize_input_discourse_operator_graph(
        (JOIN,), tmp_path / "operators.sqlite3",
        source_identity=source, license_identity=license_key)
    materialize_input_discourse_conjunction_graph(
        (TrainedInputDiscourseConjunction(JOIN_OPERATION),), tmp_path / "joins.sqlite3",
        source_identity=(92026, 71) if changed == "source" else source,
        license_identity=(1,) if changed == "license" else license_key)
    with pytest.raises(ValueError, match="different provenance"):
        TrainedGraphQueryBridge(
            tmp_path / "core_must_not_exist.sqlite3",
            relation_question_graph_database=tmp_path / "questions.sqlite3",
            input_discourse_graph_database=tmp_path / "operators.sqlite3",
            input_discourse_conjunction_graph_database=tmp_path / "joins.sqlite3")
    assert not (tmp_path / "core_must_not_exist.sqlite3").exists()
