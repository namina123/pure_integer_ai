"""普通部分关系只能由完整当前角色或已受训的指代义务授权事实查询。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, STRUCTURE_OPEN, QueryInputStructure,
)
from pure_integer_ai.cognition.understanding.query_reference_context import (
    extended_reference_context_required,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    TrainedRelationQuestionFrame,
)


def unlicensed_incomplete_relation_input(
        structure: QueryInputStructure,
        trained_frames: tuple[TrainedRelationQuestionFrame, ...] = (),
        *, complete_relation: bool = False,
        ) -> bool:
    """在读取事实前识别缺当前角色且无历史指代许可的普通关系。

    此判断不消费真值、不修改投影，也不授予生成；只选择既有同次澄清分支。
    调用者已核验的完整事实路径以及有独立合同的请求不能被该判断覆盖。
    已观察角色必须属于原关系的同一命题、谓词和物理来源，标记许可复用原
    全跨度/角色合同，不能把未知名称或含额外字符的角色当作指代。
    """
    if type(complete_relation) is not bool:
        raise TypeError('complete_relation 必须是严格布尔值')
    if (complete_relation or not structure.token_values
            or structure.graph_object_keys or structure.operation_keys
            or structure.relation_question_bindings
            or structure.operation_request_bindings
            or structure.assertion_scope_bindings
            or structure.pending_relation_composition_bindings
            or structure.discourse_compositions or structure.discourse_gaps
            or not structure.relation_candidates):
        return False
    if not all(item.state == STRUCTURE_OPEN and item.required_open > 0
               for item in structure.relation_candidates):
        return False
    if extended_reference_context_required(structure, trained_frames):
        return False
    return any(
        relation.role_coverage > 0 and any(
            item.projection_kind == PROJECTION_FILLER
            and item.proposition_key == relation.proposition_key
            and item.predicate_key == relation.predicate_key
            and item.source_ref == relation.source_ref
            and item.span_ref in relation.span_refs
            for item in structure.semantic_candidates)
        for relation in structure.relation_candidates)


__all__ = ['unlicensed_incomplete_relation_input']
