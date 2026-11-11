"""以完整问式的当前槽序闭合证据，保留来源角色序及全部图身份。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, STRUCTURE_CLOSED,
    QueryInputStructure,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    QUESTION_FRAME_VERSION, ComposedRelationQuestionFrame,
    TrainedRelationQuestionFrame, _bind_question_parses,
)


def complete_question_evidence_closed(
        structure: QueryInputStructure,
        proposition: tuple[int, ...],
        evidence_claims: frozenset[tuple[int, ...]],
        closed_semantic_keys: frozenset[tuple[int, ...]],
        ) -> bool:
    """核验唯一完整问式、当前有序槽及每槽已经闭合的概念证据。

    问式绑定由既有精确 matcher 复核，闭合投影由原概念路径消费者提供；
    不把问式当事实，不将未知槽填为训练样例，也不重排来源成员。
    """
    if len(structure.relation_question_bindings) != 1 or proposition not in evidence_claims:
        return False
    question = structure.relation_question_bindings[0]
    if question.answer_role_key or question.operation_key not in structure.operation_keys:
        return False
    relation = next((item for item in structure.relation_candidates
                     if item.proposition_key == proposition
                     and item.stable_key() == question.relation_key), None)
    if (relation is None or relation.state != STRUCTURE_CLOSED
            or relation.required_open or relation.conflict_count
            or relation.support_count != relation.required_count
            or relation.role_coverage != len(question.slot_span_refs)
            or relation.required_count not in {relation.role_coverage,
                                               relation.role_coverage + 1}
            or relation.span_refs != tuple(sorted(question.slot_span_refs))):
        return False
    # 只处理谓词由问式 gap 承载的路径；独立谓词 Span 仍走原闭合算法。
    if any(item.projection_kind == PROJECTION_PREDICATE
           and item.proposition_key == proposition
           and item.span_ref in relation.span_refs
           for item in structure.semantic_candidates):
        return False
    try:
        frame_type = (TrainedRelationQuestionFrame
                      if question.frame_key[0] == QUESTION_FRAME_VERSION
                      else ComposedRelationQuestionFrame)
        frame = frame_type.from_stable_key(question.frame_key)
        matches = _bind_question_parses(
            (frame,), structure.token_values, structure, scope_start=0,
            allowed_relation_keys=frozenset({relation.stable_key()}))
    except (TypeError, ValueError):
        return False
    if matches != (question,):
        return False
    for role, ref in zip(frame.roles, question.slot_span_refs):
        members = tuple(item for item in structure.semantic_candidates
                        if item.projection_kind == PROJECTION_FILLER
                        and item.proposition_key == proposition
                        and item.predicate_key == relation.predicate_key
                        and item.source_ref == relation.source_ref
                        and item.role_key == role and item.span_ref == ref)
        if len(members) != 1 or members[0].stable_key() not in closed_semantic_keys:
            return False
    return True
