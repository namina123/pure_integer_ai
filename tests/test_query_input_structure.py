"""输入 token/span/结构投影的纯整数阶段验证。"""
from __future__ import annotations

from dataclasses import replace

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
    STRUCTURE_CLOSED,
    STRUCTURE_OPEN,
    STRUCTURE_ORDER_CONFLICT,
    TrainedInputDiscourseOperator,
    TrainedInputMember,
    TrainedInputStructure,
    TrainedInputStructureProjector,
    scope_local_conjunction_members,
)
from pure_integer_ai.cognition.understanding.query_concept_links import (
    TrainedConceptLink,
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


def _single_pattern_structure(
        values: tuple[int, ...], ordinal: int,
        ) -> TrainedInputStructure:
    proposition = (1, 1000 + ordinal)
    predicate = (5, 2000 + ordinal)
    source_ref = (9, 3000 + ordinal)
    predicate_member = TrainedInputMember(
        (4000 + ordinal,), predicate, 5, PROJECTION_PREDICATE,
        proposition, predicate, source_ref, 0, 1, 0)
    filler_member = TrainedInputMember(
        values, (4, 5000 + ordinal), 4, PROJECTION_FILLER,
        proposition, predicate, source_ref, 1, len(values) + 1, 1,
        (8, 1))
    return TrainedInputStructure(
        proposition, predicate, source_ref, (7, 1),
        (predicate_member, filler_member))


class _FirstCodepointProjector(TrainedInputStructureProjector):
    def _matching_members(self, values):
        postings = {}
        for structure in self.structures:
            for member in structure.members:
                postings.setdefault(member.values[0], []).append(member)
        for start, value in enumerate(values):
            for member in postings.get(value, ()):
                end = start + len(member.values)
                if end <= len(values) and values[start:end] == member.values:
                    yield start, member


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


def test_projector_materializes_trained_binary_discourse_operation():
    operator = TrainedInputDiscourseOperator((91591, 1), (99,))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (33, 11, 22, 99, 33, 11, 22))

    assert len(projected.discourse_compositions) == 1
    composition = projected.discourse_compositions[0]
    spans = {item.ref_key: item for item in projected.spans}
    assert spans[composition.operation_span_ref].values == (99,)
    assert [spans[item].values for item in composition.operand_scope_refs] == [
        (33, 11, 22), (33, 11, 22)]
    assert composition.state in {
        STRUCTURE_OPEN, STRUCTURE_CLOSED, STRUCTURE_ORDER_CONFLICT}
    assert projected.integer_trace()["discourse_compositions"]
    assert projected.stable_key()[0] == 4


def test_discourse_operand_ignores_partial_competing_projection():
    complete = _structure()
    partial = _structure(
        filler_values=(11, 99),
        filler_candidates=((4, 301), (4, 302)),
        proposition=(1, 51),
    )
    projected = TrainedInputStructureProjector(
        (complete, partial), discourse_operators=(
            TrainedInputDiscourseOperator((91591, 1), (77,)),
        )).project((33, 11, 22, 77, 33, 11, 22))
    assert projected.discourse_compositions[0].state == STRUCTURE_CLOSED


def test_discourse_operand_preserves_local_order_conflict_after_global_closure():
    operator = TrainedInputDiscourseOperator((91591, 1), (77,))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (22, 11, 33, 77, 33, 11, 22))
    assert any(item.state == STRUCTURE_CLOSED
               for item in projected.relation_candidates)
    composition = projected.discourse_compositions[0]
    assert composition.operand_states == (
        STRUCTURE_ORDER_CONFLICT, STRUCTURE_CLOSED)
    assert composition.state == STRUCTURE_ORDER_CONFLICT


def test_untrained_surface_does_not_create_discourse_operation():
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(
            TrainedInputDiscourseOperator((91591, 1), (99,)),
        )).project((33, 11, 22, 100, 33, 11, 22))
    assert projected.discourse_compositions == ()


def test_untrained_boundary_between_two_closed_scopes_is_an_integer_gap():
    operator = TrainedInputDiscourseOperator((91591, 1), (99,))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (33, 11, 22, 100, 33, 11, 22))
    assert projected.discourse_compositions == ()
    assert len(projected.discourse_gaps) == 1
    gap = projected.discourse_gaps[0]
    assert gap.state == STRUCTURE_OPEN
    assert projected.integer_trace()["discourse_gaps"]


def test_scope_local_conjunction_requires_unique_closed_members_per_operand():
    left = _structure()
    right = _structure(
        filler_values=(12, 23),
        filler_candidates=((4, 201), (4, 202)),
        proposition=(1, 51),
    )
    operator = TrainedInputDiscourseOperator((91591, 1), (77,))
    projected = TrainedInputStructureProjector(
        (left, right), discourse_operators=(operator,)).project(
            (33, 11, 99, 22, 77, 33, 12, 99, 23))

    bindings = scope_local_conjunction_members(
        projected, frozenset({(91591, 1)}))
    assert len(bindings) == 1
    assert bindings[0][1] == ((1, 50), (1, 51))
    assert scope_local_conjunction_members(
        projected, frozenset({(91591, 99)})) == ()

    candidates = tuple(sorted(
        {item.stable_key() for item in projected.relation_candidates}))
    ambiguous = replace(
        projected,
        discourse_compositions=tuple(replace(
            item,
            operand_scope_refs=(projected.source_span_ref,
                                projected.source_span_ref),
            operand_relation_keys=(candidates, candidates),
        ) for item in projected.discourse_compositions),
    )
    assert scope_local_conjunction_members(
        ambiguous, frozenset({(91591, 1)})) == ()


def test_scope_local_conjunction_three_member_component_contract():
    from pure_integer_ai.experiments.trained_graph_query_bridge import (
        _closed_scope_conjunction_components,
    )
    a, b, c = (1, 50), (1, 51), (1, 52)
    bindings = (
        ((10,), (91599, 1), (a, b)),
        ((11,), (91599, 1), (b, c)),
    )
    assert _closed_scope_conjunction_components(
        bindings, {a, b, c}) == ((a, b, c),)
    assert _closed_scope_conjunction_components(
        bindings, {a, b}) == ((a, b),)
    assert _closed_scope_conjunction_components(
        (bindings[0], ((12,), (91599, 1), ((1, 60), (1, 61)))),
        {a, b, (1, 60), (1, 61)},
    ) == ()


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


def test_prefix_index_matches_legacy_exact_integer_spans():
    patterns = [(11, 22), (11, 22, 33), (11,)]
    patterns.extend((11, value) for value in range(100, 140))
    structures = tuple(
        _single_pattern_structure(pattern, ordinal)
        for ordinal, pattern in enumerate(patterns)
    )
    structures += (_single_pattern_structure((11, 22), len(structures)),)
    origin = structures[0]
    target = structures[1]
    link = TrainedConceptLink(
        origin.members[1].candidate_key,
        target.members[1].candidate_key,
        origin.proposition_key,
        origin.predicate_key,
        origin.source_ref,
        (6, 1),
    )
    projector = TrainedInputStructureProjector(
        structures, concept_links=(link,))
    reference = _FirstCodepointProjector(
        structures, concept_links=(link,))
    values = (11, 22, 33, 11)

    expected = tuple(sorted(
        (start, member.stable_key())
        for structure in projector.structures
        for member in structure.members
        for start in range(len(values))
        if values[start:start + len(member.values)] == member.values
    ))
    actual = tuple(sorted(
        (start, member.stable_key())
        for start, member in projector._matching_members(values)
    ))

    assert actual == expected
    assert projector.project(values) == reference.project(values)
    assert len(projector._prefix_postings[(11, 22)]) == 3
    assert sum(
        len(structure.members)
        for structure in projector.structures
        if any(member.values[0] == 11 for member in structure.members)
    ) > len(projector._prefix_postings[(11, 22)])
