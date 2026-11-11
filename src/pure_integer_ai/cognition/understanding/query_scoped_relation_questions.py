"""Bind learned question syntax inside trained, closed conjunction scopes.

This recognizes an exact question over each independently admitted Core fact.
It does not infer a transitive or causal conclusion from the shared endpoint.
"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
    STRUCTURE_CLOSED,
    InputRelationQuestionBinding,
    QueryInputStructure,
    scope_local_conjunction_candidate_keys,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    TrainedRelationQuestionFrame,
    _bind_question_parses,
    _composed_frames,
)


def match_scoped_relation_question_frames(
        frames: tuple[TrainedRelationQuestionFrame, ...],
        structure: QueryInputStructure,
        authorized_conjunction_operations: frozenset[tuple[int, ...]], *,
        max_combinations: int = 4096,
        ) -> tuple[InputRelationQuestionBinding, ...]:
    """Require every operator and every full clause to have one graph witness."""
    if (not frames or not structure.tokens
            or not structure.discourse_compositions or structure.discourse_gaps
            or not authorized_conjunction_operations):
        return ()
    if len({item.open_frame().stable_key() for item in frames}) != len(frames):
        raise ValueError("question frames have duplicate parse identities")
    selected = scope_local_conjunction_candidate_keys(
        structure, authorized_conjunction_operations)
    if len(selected) != len(structure.discourse_compositions):
        return ()
    by_composition = {item.stable_key(): item
                      for item in structure.discourse_compositions}
    spans = {item.ref_key: item for item in structure.spans}
    scope_relation: dict[tuple[int, ...], tuple[int, ...]] = {}
    intervals = set()
    for composition_key, relation_keys in selected:
        composition = by_composition[composition_key]
        if (composition.state != STRUCTURE_CLOSED
                or composition.operation_key not in authorized_conjunction_operations):
            return ()
        operator_span = spans[composition.operation_span_ref]
        intervals.add((operator_span.start, operator_span.end))
        for scope_ref, relation_key in zip(
                composition.operand_scope_refs, relation_keys):
            prior = scope_relation.setdefault(scope_ref, relation_key)
            if prior != relation_key:
                return ()
    if len(scope_relation) < 2 or len(set(scope_relation.values())) != len(scope_relation):
        return ()
    for scope_ref in scope_relation:
        span = spans[scope_ref]
        intervals.add((span.start, span.end))
    # Exact partition prevents a valid pair of clauses from silently dropping
    # extra input outside its trained integer operator and question scopes.
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor or end <= start:
            return ()
        cursor = end
    if cursor != len(structure.token_values):
        return ()
    relation_by_key = {item.stable_key(): item
                       for item in structure.relation_candidates}
    bindings = []
    identities = []
    for scope_ref, relation_key in sorted(
            scope_relation.items(), key=lambda item: spans[item[0]].start):
        span = spans[scope_ref]
        values = structure.token_values[span.start:span.end]
        available = (*frames, *_composed_frames(
            frames, values, max_combinations=max_combinations))
        if len({item.open_frame().stable_key() for item in available}) != len(available):
            raise ValueError("question composition has duplicate parse identities")
        matches = _bind_question_parses(
            available, values, structure, scope_start=span.start,
            allowed_relation_keys=frozenset({relation_key}))
        if len(matches) != 1:
            return ()
        binding = matches[0]
        relation = relation_by_key[relation_key]
        slot_refs = set(binding.slot_span_refs)
        role_identities = {
            item.candidate_key for item in structure.semantic_candidates
            if (item.projection_kind == PROJECTION_FILLER
                and item.proposition_key == relation.proposition_key
                and item.predicate_key == relation.predicate_key
                and item.source_ref == relation.source_ref
                and item.span_ref in slot_refs)
        }
        if len(role_identities) != len(binding.slot_span_refs):
            return ()
        bindings.append(binding)
        identities.append(role_identities)
    # Shared graph identity, not a common character or proposition order,
    # establishes connected input.  This is a topology gate, not inference.
    reached = {0}
    while True:
        expanded = reached | {
            index for index, values in enumerate(identities)
            if any(values & identities[prior] for prior in reached)
        }
        if expanded == reached:
            break
        reached = expanded
    if len(reached) != len(identities):
        return ()
    return tuple(sorted(bindings, key=lambda item: item.stable_key()))


def conjunction_has_complete_literal_scopes(
        structure: QueryInputStructure,
        authorized_conjunction_operations: frozenset[tuple[int, ...]],
        ) -> bool:
    """Retain the existing fact route only when trained members cover each clause.

    An arbitrary question prefix or suffix must not become factual authority
    merely because its two entity mentions project to an existing Core pair.
    """
    selected = scope_local_conjunction_candidate_keys(
        structure, authorized_conjunction_operations)
    if not selected or len(selected) != len(structure.discourse_compositions):
        return False
    compositions = {item.stable_key(): item
                    for item in structure.discourse_compositions}
    relations = {item.stable_key(): item for item in structure.relation_candidates}
    spans = {item.ref_key: item for item in structure.spans}
    for composition_key, relation_keys in selected:
        for scope_ref, relation_key in zip(
                compositions[composition_key].operand_scope_refs, relation_keys):
            scope, relation = spans[scope_ref], relations[relation_key]
            members = tuple(item for item in structure.semantic_candidates
                            if item.proposition_key == relation.proposition_key
                            and item.predicate_key == relation.predicate_key
                            and item.source_ref == relation.source_ref
                            and item.span_ref in relation.span_refs
                            and scope.start <= spans[item.span_ref].start
                            and spans[item.span_ref].end <= scope.end)
            if (not any(item.projection_kind == PROJECTION_PREDICATE for item in members)
                    or len({item.member_ordinal for item in members}) != relation.required_count):
                return False
            cursor = scope.start
            for start, end in sorted({(spans[item.span_ref].start,
                                      spans[item.span_ref].end) for item in members}):
                if start != cursor:
                    return False
                cursor = end
            if cursor != scope.end:
                return False
    return True


__all__ = ["conjunction_has_complete_literal_scopes",
           "match_scoped_relation_question_frames"]
