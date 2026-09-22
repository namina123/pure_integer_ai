"""输入 token/span/结构投影的纯整数阶段验证。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
    STRUCTURE_CLOSED,
    STRUCTURE_OPEN,
    STRUCTURE_ORDER_CONFLICT,
    TrainedInputMember,
    TrainedInputStructure,
    TrainedInputStructureProjector,
)


def _structure(
        *,
        filler_values: tuple[int, ...] = (11, 22),
        filler_candidates: tuple[tuple[int, ...], ...] = (
            (4, 101), (4, 102)),
        proposition: tuple[int, ...] = (1, 50),
        ) -> TrainedInputStructure:
    predicate = (5, 60)
    source_ref = (9, 70)
    members = [
        TrainedInputMember(
            (33,), predicate, 5, PROJECTION_PREDICATE,
            proposition, predicate, source_ref, 0, 1, 0),
    ]
    for ordinal, (value, candidate) in enumerate(
            zip(filler_values, filler_candidates)):
        members.append(TrainedInputMember(
            (value,), candidate, 4, PROJECTION_FILLER,
            proposition, predicate, source_ref, ordinal + 2, ordinal + 3,
            ordinal + 1, (8, ordinal + 1)))
    members.sort(key=lambda item: item.trained_start)
    members = [
        TrainedInputMember(
            item.values, item.candidate_key, item.object_kind,
            item.projection_kind, item.proposition_key, item.predicate_key,
            item.source_ref, item.trained_start, item.trained_end,
            ordinal, item.role_key)
        for ordinal, item in enumerate(members)
    ]
    return TrainedInputStructure(
        proposition, predicate, source_ref, (7, 1), tuple(members))


def test_projector_closes_structure_with_gap_and_integer_trace():
    structure = _structure()
    projected = TrainedInputStructureProjector((structure,)).project(
        (33, 11, 99, 22))
    assert projected.relation_candidates
    candidate = projected.relation_candidates[0]
    assert candidate.state == STRUCTURE_CLOSED
    assert candidate.required_open == 0
    assert len(projected.carrier_candidates) == 1
    assert projected.precedes
    assert projected.contains

    def assert_integer_tree(value):
        if isinstance(value, dict):
            for item in value.values():
                assert_integer_tree(item)
        elif isinstance(value, list):
            for item in value:
                assert_integer_tree(item)
        else:
            assert type(value) is int

    assert_integer_tree(projected.integer_trace())


def test_projector_keeps_open_candidate_for_missing_member():
    projected = TrainedInputStructureProjector((_structure(),)).project(
        (33, 11))
    candidate = projected.relation_candidates[0]
    assert candidate.state == STRUCTURE_OPEN
    assert candidate.required_open == 1


def test_projector_marks_reverse_order_as_conflict():
    projected = TrainedInputStructureProjector((_structure(),)).project(
        (22, 11, 33))
    candidate = projected.relation_candidates[0]
    assert candidate.state == STRUCTURE_ORDER_CONFLICT
    assert candidate.conflict_count == 1


def test_projector_retains_parallel_filler_candidates():
    first = _structure(
        filler_values=(11, 22),
        filler_candidates=((4, 101), (4, 102)),
    )
    second = _structure(
        filler_values=(11, 22),
        filler_candidates=((4, 201), (4, 202)),
        proposition=(1, 51),
    )
    projected = TrainedInputStructureProjector((first, second)).project(
        (11, 33, 22))
    fillers = {
        item.candidate_key
        for item in projected.semantic_candidates
        if item.projection_kind == PROJECTION_FILLER
    }
    assert fillers == {(4, 101), (4, 102), (4, 201), (4, 202)}
    assert len(projected.relation_candidates) == 2
