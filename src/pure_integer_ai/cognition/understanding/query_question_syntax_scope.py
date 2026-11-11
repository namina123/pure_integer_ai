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


def grounded_question_open_scope(candidates, structure, frames_by_shape, trained_frames):
    """完整落地WH存在时排除零锚点吞入解析，保留所有真实来源及指代竞争。

    没有全部角色对齐的plain WH见证时不改变输入。只排除一个独立角色也未
    对齐、且没有训练指代授权的plain候选；部分落地、Core及reference候选
    保持。此纯作用域门不选择来源、答案或事实，也不增加任何读取预算。
    """
    from pure_integer_ai.cognition.understanding.query_reference_context import reference_marker_sources

    candidates = tuple(candidates)
    intervals = exact_question_filler_intervals(structure)
    spans = {(span.source_ref, span.start, span.end, span.values): span.ref_key
             for span in structure.spans}
    members = {(item.proposition_key, item.predicate_key, item.source_ref,
                item.role_key, item.span_ref) for item in structure.semantic_candidates
               if item.projection_kind == PROJECTION_FILLER}
    classified = []
    grounded = False
    for candidate in candidates:
        frames = frames_by_shape.get((candidate.frame.predicate,
            candidate.frame.roles, candidate.frame.gaps), ())
        plain = bool(frames) and all(not frame.is_reference_slot for frame in frames)
        aligned = tuple(
            (slot.source_ref, slot.start, slot.end, slot.values) in intervals
            and (candidate.frame.proposition, candidate.frame.predicate,
                 candidate.frame.source_ref, slot.role,
                 spans.get((slot.source_ref, slot.start, slot.end, slot.values))) in members
            for slot in candidate.bindings)
        if plain and aligned and all(aligned):
            grounded = True
        classified.append((candidate, plain, aligned))
    if not grounded:
        return tuple(candidates)
    return tuple(candidate for candidate, plain, aligned in classified
        if not plain or any(aligned) or any(reference_marker_sources(
            slot.values, slot.role, candidate.frame.predicate, trained_frames)
            for slot in candidate.bindings))
