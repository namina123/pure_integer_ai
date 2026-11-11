"""识别仍须由当前 Memory 消解的观察槽，不给来源端点授予事实资格。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, STRUCTURE_OPEN,
)
from pure_integer_ai.cognition.understanding.query_reference_context import (
    reference_marker_sources,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    _matching_open_relation_candidates,
)
from pure_integer_ai.cognition.understanding.query_slot_question_composition import (
    restore_slot_question_frame,
)


def pending_reference_relation_keys(structure, facts, trained_frames=()):
    """以完整受训问式、来源和唯一指代跨度分类待解关系，保留全部候选。

    返回值只用于输入端点竞争分账；它不能选事实、采用指代目标或关闭
    开放槽。独立组合、普通已绑定 WH 和缺少任一来源见证的候选不适用。
    """
    if (not trained_frames or structure.discourse_compositions
            or structure.discourse_gaps):
        return frozenset()
    relations = {item.stable_key(): item for item in structure.relation_candidates}
    spans = {item.ref_key: item for item in structure.spans}
    pending = set()
    for binding in structure.relation_question_bindings:
        try:
            frame = restore_slot_question_frame(
                binding.frame_key, trained_frames=trained_frames)
        except (TypeError, ValueError):
            continue
        relation = relations.get(binding.relation_key)
        if (frame.is_reference_slot or relation is None
                or binding.operation_key != frame.operation_key
                or binding.answer_role_key != frame.answer_role_key
                or relation.predicate_key != frame.predicate_key
                or relation.state != STRUCTURE_OPEN
                or relation.conflict_count or relation.predicate_coverage != 1
                or relation.required_open != 2
                or relation.role_coverage != len(frame.roles) - 1
                or relation.support_count != relation.role_coverage + 1
                or len(binding.slot_span_refs) != len(frame.roles)):
            continue
        fact = facts.get(relation.proposition_key)
        if (fact is None or fact.source_hash <= 0
                or relation.source_ref != (fact.source_hash,)
                or fact.predicate.stable_key() != frame.predicate_key
                or relation.required_count != len(fact.bindings) + 1):
            continue
        fillers = {item.role.stable_key(): item.filler.stable_key()
                   for item in fact.bindings}
        if (len(fillers) != len(fact.bindings)
                or frame.answer_role_key not in fillers
                or frame.roles != tuple(item.role.stable_key() for item in fact.bindings
                                         if item.role.stable_key() != frame.answer_role_key)
                or any(ref not in spans
                       or spans[ref].source_ref != structure.source_ref
                       for ref in binding.slot_span_refs)):
            continue
        parses = _matching_open_relation_candidates(
            structure, frame, relation.proposition_key,
            relation.source_ref, binding.slot_span_refs)
        if len(parses) != 1:
            continue
        members = tuple(item for item in structure.semantic_candidates
                        if item.proposition_key == relation.proposition_key
                        and item.predicate_key == relation.predicate_key
                        and item.source_ref == relation.source_ref
                        and item.span_ref in relation.span_refs)
        observed = tuple(tuple(item for item in members
                               if item.projection_kind == PROJECTION_FILLER
                               and item.role_key == role and item.span_ref == ref)
                         for role, ref in zip(frame.roles, binding.slot_span_refs))
        if (any(len(items) > 1 for items in observed)
                or sum(not items for items in observed) != 1
                or any(items and items[0].candidate_key != fillers[role]
                       for role, items in zip(frame.roles, observed))
                or {item.role_key for item in members
                    if item.projection_kind == PROJECTION_FILLER}
                != {role for role, items in zip(frame.roles, observed) if items}):
            continue
        predicate_refs = tuple(item.span_ref for item in members
                               if item.projection_kind == PROJECTION_PREDICATE)
        if (len(predicate_refs) > 1 or relation.span_refs != tuple(sorted((
                *(ref for ref, items in zip(binding.slot_span_refs, observed) if items),
                *predicate_refs)))):
            continue
        role, ref = next((role, ref) for role, ref, items in zip(
            frame.roles, binding.slot_span_refs, observed) if not items)
        if reference_marker_sources(
                spans[ref].values, role, frame.predicate_key, trained_frames):
            pending.add(binding.relation_key)
    return frozenset(pending)


def unselected_reference_question(structure, facts, trained_frames=()):
    """仅把同一完整受训问式的全部待解来源归为零事实澄清资格。

    调用方须先尝试现役Memory唯一绑定；本函数不选择来源、目标或事实，
    不改输入、候选和预算。混合语法、缺证据及部分待解关系继续原查询门。
    """
    bindings = structure.relation_question_bindings
    if len(bindings) < 2:
        return False
    syntax = lambda item: (item.frame_key, item.operation_key,
                           item.slot_span_refs, item.answer_role_key)
    if (not all(item.answer_role_key for item in bindings)
            or any(syntax(item) != syntax(bindings[0]) for item in bindings)):
        return False
    pending = pending_reference_relation_keys(structure, facts, trained_frames)
    return (len(pending) == len(bindings)
            and pending == frozenset(item.relation_key for item in bindings))
