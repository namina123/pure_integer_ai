"""问式对Core施加约束前的整数角色作用域校验，不选择事实或创建概念。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    QueryInputStructure,
)
from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate,
)


def exact_question_filler_intervals(
        structure: QueryInputStructure,
        ) -> frozenset[tuple[tuple[int, ...], int, int, tuple[int, ...]]]:
    """汇聚独立filler投影的完整区间，不依赖请求谓词、闭合或派生概念路由。"""
    filler_refs = {
        item.span_ref for item in structure.semantic_candidates
        if item.projection_kind == PROJECTION_FILLER
        and not item.concept_route_key
    }
    return frozenset(
        (span.source_ref, span.start, span.end, span.values)
        for span in structure.spans
        if span.ref_key in filler_refs
        and span.source_ref == structure.source_ref
    )


def question_parse_aligns_with_fillers(
        parsed: OpenRelationCandidate,
        intervals: frozenset[
            tuple[tuple[int, ...], int, int, tuple[int, ...]]],
        ) -> bool:
    """全部已观测角色须精确对齐，禁止仅用间隔命中覆盖现有Core输入作用域。"""
    return all(
        (slot.source_ref, slot.start, slot.end, slot.values) in intervals
        for slot in parsed.bindings
    )
