"""输入 token/span/结构投影的纯整数阶段验证。"""
from __future__ import annotations

from dataclasses import replace

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
    STRUCTURE_CLOSED,
    STRUCTURE_OPEN,
    STRUCTURE_ORDER_CONFLICT,
    InputRelationQuestionBinding,
    TrainedInputDiscourseOperator,
    TrainedInputMember,
    TrainedInputStructure,
    TrainedInputStructureProjector,
    relation_candidate_semantic_groups,
    scope_local_conjunction_members,
    scope_local_fact_query_candidate_keys,
    scope_local_general_operation_candidate_keys,
)
from pure_integer_ai.cognition.understanding.query_concept_links import (
    TrainedConceptLink,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    _conflicting_input_filler_groups,
    _has_closed_bound_relation_question,
)


def _structure(
        *,
        predicate_values: tuple[int, ...] = (33,),
        filler_values: tuple[int, ...] = (11, 22),
        filler_candidates: tuple[tuple[int, ...], ...] = (
            (4, 101), (4, 102)),
        proposition: tuple[int, ...] = (1, 50),
        source_ref: tuple[int, ...] = (9, 70),
        ) -> TrainedInputStructure:
    predicate = (5, 60)
    members = [
        TrainedInputMember(
            predicate_values, predicate, 5, PROJECTION_PREDICATE,
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


def test_empty_active_relation_scope_keeps_unscoped_rank_candidates():
    """Non-composition queries must not turn an empty scope into deny-all."""
    from pure_integer_ai.experiments.trained_graph_query_bridge import (
        TrainedGraphQueryBridge,
    )

    projected = TrainedInputStructureProjector((_structure(),)).project(
        (33, 11, 22))
    candidate = projected.relation_candidates[0]
    bridge = TrainedGraphQueryBridge.__new__(TrainedGraphQueryBridge)
    best, second = bridge._rank_candidates(
        projected, (candidate.proposition_key,),
        active_relation_keys=frozenset())
    assert best == (candidate.score, candidate.proposition_key)
    assert second == (0, ())


def test_bound_question_keeps_closed_relation_without_predicate_span():
    """A question operation can close a binary relation with no predicate span."""
    projected = TrainedInputStructureProjector((_structure(),)).project(
        (33, 11, 22))
    relation = projected.relation_candidates[0]
    relation = replace(relation, predicate_coverage=0)
    binding = InputRelationQuestionBinding(
        frame_key=(92020, 1), operation_key=(92004, 1),
        relation_key=relation.stable_key(), slot_span_refs=relation.span_refs)
    prepared = replace(
        projected, relation_candidates=(relation,),
        relation_question_bindings=(binding,), operation_keys=((92004, 1),),
        open_relations=())
    assert _has_closed_bound_relation_question(prepared)

    open_relation = replace(
        relation, support_count=relation.required_count - 1, required_open=1,
        state=STRUCTURE_OPEN)
    open_binding = replace(binding, relation_key=open_relation.stable_key())
    assert not _has_closed_bound_relation_question(replace(
        prepared, relation_candidates=(open_relation,),
        relation_question_bindings=(open_binding,)))


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
    assert projected.operation_keys == ((91591, 1),)


def test_projector_does_not_invent_operation_for_untrained_integer_surface():
    operator = TrainedInputDiscourseOperator((91591, 1), (99,))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (33, 11, 22, 100))
    assert projected.operation_keys == ()
    assert projected.discourse_compositions == ()


def test_projector_prefers_longer_nested_surface_for_same_operation():
    operator = (91680, 1)
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(
            TrainedInputDiscourseOperator(operator, (99,)),
            TrainedInputDiscourseOperator(operator, (98, 99)),
        )).project((33, 11, 22, 98, 99))

    assert len(projected.discourse_compositions) == 1
    composition = projected.discourse_compositions[0]
    spans = {item.ref_key: item for item in projected.spans}
    assert spans[composition.operation_span_ref].values == (98, 99)


def test_projector_keeps_overlapping_distinct_operations_fail_closed():
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(
            TrainedInputDiscourseOperator((91680, 1), (99,)),
            TrainedInputDiscourseOperator((91680, 2), (98, 99)),
        )).project((33, 11, 22, 98, 99))

    assert len(projected.discourse_compositions) == 2
    assert scope_local_fact_query_candidate_keys(
        projected, frozenset({(91680, 1)})) == ()


def test_fact_query_contract_selects_only_closed_left_scope():
    operator = TrainedInputDiscourseOperator((91680, 1), (99,))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (33, 11, 22, 99))
    matches = scope_local_fact_query_candidate_keys(
        projected, frozenset({operator.operation_key}))
    assert len(matches) == 1
    assert matches[0][1] == operator.operation_key
    assert matches[0][2] == projected.relation_candidates[0].stable_key()


def test_fact_query_contract_selects_only_closed_right_scope():
    operator = TrainedInputDiscourseOperator((91680, 3), (98, 99))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (98, 99, 33, 11, 22))
    matches = scope_local_fact_query_candidate_keys(
        projected, {operator.operation_key: (91672, 2)})
    assert len(matches) == 1
    assert matches[0][1] == operator.operation_key
    assert matches[0][2] == projected.relation_candidates[0].stable_key()


def test_fact_query_contract_rejects_right_scope_for_left_mode():
    operator = TrainedInputDiscourseOperator((91680, 3), (98, 99))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (98, 99, 33, 11, 22))
    assert scope_local_fact_query_candidate_keys(
        projected, {operator.operation_key: (91672, 1)}) == ()


def test_fact_query_contract_rejects_right_scope_and_unauthorized_marker():
    operator = TrainedInputDiscourseOperator((91680, 1), (99,))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (33, 11, 22, 99, 33, 11, 22))
    assert scope_local_fact_query_candidate_keys(
        projected, frozenset({operator.operation_key})) == ()
    assert scope_local_fact_query_candidate_keys(
        projected, frozenset({(91680, 99)})) == ()


def test_fact_query_contract_rejects_multiple_operations():
    operator = TrainedInputDiscourseOperator((91680, 1), (99,))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (33, 11, 22, 99, 33, 11, 22, 99))
    assert len(projected.discourse_compositions) == 2
    assert scope_local_fact_query_candidate_keys(
        projected, frozenset({operator.operation_key})) == ()


def test_general_operation_contract_selects_unique_closed_right_scope():
    operator = TrainedInputDiscourseOperator((91680, 8), (98,))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (98, 33, 11, 22))
    matches = scope_local_general_operation_candidate_keys(
        projected, frozenset({operator.operation_key, (91680, 99)}))
    assert len(matches) == 1
    assert matches[0][1] == operator.operation_key
    assert matches[0][2] == projected.relation_candidates[0].stable_key()


def test_general_operation_memory_view_keeps_only_the_closed_scope():
    from pure_integer_ai.experiments.trained_graph_query_bridge import (
        TrainedGraphQueryBridge,
    )

    operator = TrainedInputDiscourseOperator((91680, 12), (98,))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (98, 33, 11, 22))
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.input_discourse_operation_modes = {
        operator.operation_key: (91672, 3)}
    bridge.input_discourse_fact_query_modes = {}
    bridge.input_discourse_conjunctions = ()
    bridge.relation_question_frames = ()

    memory_view = bridge.active_memory_input_structure(projected)

    assert memory_view.discourse_compositions == ()
    assert len(memory_view.relation_candidates) == 1
    assert memory_view.relation_candidates[0].state == STRUCTURE_CLOSED
    assert memory_view.relation_candidates[0].proposition_key == (1, 50)


def test_general_operation_contract_rejects_open_right_scope():
    operator = TrainedInputDiscourseOperator((91680, 9), (98,))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (98, 33, 11))
    assert scope_local_general_operation_candidate_keys(
        projected, frozenset({operator.operation_key})) == ()


def test_general_operation_contract_rejects_repeated_marker():
    operator = TrainedInputDiscourseOperator((91680, 10), (98,))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (98, 33, 11, 22, 98, 33, 11, 22))
    assert len(projected.discourse_compositions) == 2
    assert scope_local_general_operation_candidate_keys(
        projected, frozenset({operator.operation_key})) == ()


def test_general_operation_contract_rejects_conflicting_right_scope():
    operator = TrainedInputDiscourseOperator((91680, 11), (98,))
    projected = TrainedInputStructureProjector(
        (_structure(),), discourse_operators=(operator,)).project(
            (98, 33, 22, 11))
    assert projected.discourse_compositions[0].state == STRUCTURE_ORDER_CONFLICT
    assert scope_local_general_operation_candidate_keys(
        projected, frozenset({operator.operation_key})) == ()


def test_fact_query_and_conjunction_contracts_can_coexist_once_each():
    left = _structure()
    right = _structure(
        predicate_values=(44,), filler_values=(12, 23),
        filler_candidates=((4, 201), (4, 202)), proposition=(1, 51),
    )
    fact = TrainedInputDiscourseOperator((91680, 1), (98,))
    conjunction = TrainedInputDiscourseOperator((91591, 1), (99,))
    projected = TrainedInputStructureProjector(
        (left, right), discourse_operators=(fact, conjunction)).project(
            (33, 11, 22, 99, 44, 12, 23, 98))

    fact_matches = scope_local_fact_query_candidate_keys(
        projected, {fact.operation_key: (91672, 1)})
    conjunction_matches = scope_local_conjunction_members(
        projected, frozenset({conjunction.operation_key}))
    assert len(fact_matches) == 1
    assert len(conjunction_matches) == 1
    assert fact_matches[0][1] == fact.operation_key
    assert tuple(item[0] for item in conjunction_matches) == tuple(
        item.stable_key() for item in projected.discourse_compositions
        if item.operation_key == conjunction.operation_key)


def test_pair_closure_requires_distinct_ordered_filler_spans():
    left = _structure(predicate_values=(88,))
    right = _structure(
        predicate_values=(99,), filler_values=(12, 23),
        filler_candidates=((4, 201), (4, 202)), proposition=(1, 51),
    )
    projected = TrainedInputStructureProjector(
        (left, right), discourse_operators=(
            TrainedInputDiscourseOperator((91591, 1), (77,)),
        )).project((11, 22, 77, 12, 23))
    target = next(
        item for item in projected.relation_candidates
        if item.proposition_key == left.proposition_key)
    assert target.required_count == 2
    assert target.predicate_coverage == 0
    assert scope_local_conjunction_members(
        projected, frozenset({(91591, 1)}))

    target_semantics = [
        item for item in projected.semantic_candidates
        if item.proposition_key == target.proposition_key
    ]
    first_span = next(item.span_ref for item in target_semantics
                      if item.member_ordinal == 1)
    collapsed = tuple(sorted(
        (replace(item, span_ref=first_span)
         if item.proposition_key == target.proposition_key
         and item.member_ordinal == 2 else item
         for item in projected.semantic_candidates),
        key=lambda item: item.stable_key(),
    ))
    corrupted = replace(projected, semantic_candidates=collapsed)
    assert scope_local_conjunction_members(
        corrupted, frozenset({(91591, 1)})) == ()
    assert not corrupted.concept_evidence_closed(
        target.proposition_key, frozenset())


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


def test_provenance_duplicate_relation_rows_are_one_scope_candidate():
    first = _structure(proposition=(1, 501), source_ref=(9, 701))
    second = _structure(proposition=(1, 502), source_ref=(9, 702))
    operator = TrainedInputDiscourseOperator((91680, 1), (99,))
    projected = TrainedInputStructureProjector(
        (first, second), discourse_operators=(operator,)).project(
            (33, 11, 22, 99))

    assert len(projected.relation_candidates) == 2
    groups = relation_candidate_semantic_groups(projected)
    assert len(groups) == 1
    assert len(groups[0][1]) == 2
    matches = scope_local_fact_query_candidate_keys(
        projected, frozenset({operator.operation_key}))
    assert len(matches) == 1
    assert matches[0][2] in {
        item.stable_key() for item in projected.relation_candidates}


def test_input_filler_conflicts_are_scoped_to_proposition_source_and_role():
    from pure_integer_ai.cognition.understanding.query_input_structure import (
        InputSemanticCandidate,
    )

    role = (8, 1)
    span = (1, 900, 0, 1, 1)

    def candidate(filler, proposition, source):
        return InputSemanticCandidate(
            (4, filler), 4, PROJECTION_FILLER, span,
            (1, proposition), (5, 60), role, (9, source), 1)

    cross_proposition = (
        candidate(101, 50, 70),
        candidate(102, 51, 71),
    )
    assert _conflicting_input_filler_groups(cross_proposition) == frozenset()

    same_proposition = (
        candidate(101, 50, 70),
        candidate(102, 50, 70),
    )
    assert _conflicting_input_filler_groups(same_proposition) == frozenset({
        ((1, 50), (9, 70), role),
    })


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
