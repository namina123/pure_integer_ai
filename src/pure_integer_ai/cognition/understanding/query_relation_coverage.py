"""检查闭合关系是否完整消费当前整数输入，而不是只命中已知子串。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    STRUCTURE_CLOSED,
    QueryInputStructure,
)


def has_uncovered_closed_relation_modifier(structure: QueryInputStructure) -> bool:
    """阻止未解析的修饰、否定或载体区间借闭合子串取得事实资格。

    已绑定的问式、组合及操作继续由其独立作用域合同裁决。普通陈述至少需要
    一条关系的成员覆盖完整输入，或一个来源一致的全输入开放框架，其每个动态
    角色都精确绑定当前候选的已知 filler。框架的训练间隔可消费未单独投影的
    语言结构；未知区间不能被宽角色吞入后再冒充较短的已知实体。
    无独立谓词跨度的端点闭合也受同一覆盖合同约束。
    """
    if (structure.assertion_scope_bindings
            or structure.relation_question_bindings
            or structure.pending_relation_composition_bindings
            or structure.discourse_compositions
            or structure.discourse_gaps
            or structure.operation_request_bindings
            or not structure.token_values):
        return False
    closed = tuple(
        item for item in structure.relation_candidates
        if (item.state == STRUCTURE_CLOSED
            and item.required_open == 0
            and item.required_count == item.support_count
            and item.role_coverage > 0
            and item.conflict_count == 0))
    if not closed:
        return False
    spans = {item.ref_key: item for item in structure.spans}
    width = len(structure.token_values)
    for relation in closed:
        intervals = sorted(
            (spans[ref].start, spans[ref].end)
            for ref in relation.span_refs if ref in spans)
        cursor = 0
        for start, end in intervals:
            if start > cursor:
                break
            cursor = max(cursor, end)
        if cursor == width:
            return False
        # 一条全输入训练框架可消费合法间隔，但动态角色必须精确等于当前图
        # 的 filler span；不能把“主体+未解析否定”当作那个较短的主体。
        projections = {
            (item.role_key, spans[item.span_ref].start, spans[item.span_ref].end,
             spans[item.span_ref].values)
            for item in structure.semantic_candidates
            if (item.projection_kind == PROJECTION_FILLER
                and item.proposition_key == relation.proposition_key
                and item.predicate_key == relation.predicate_key
                and item.source_ref == relation.source_ref
                and item.span_ref in relation.span_refs
                and item.span_ref in spans)}
        for parsed in structure.open_relations:
            if (parsed.frame.proposition == relation.proposition_key
                    and parsed.frame.predicate == relation.predicate_key
                    and parsed.frame.source_ref == relation.source_ref
                    and parsed.source_ref == structure.source_ref
                    and parsed.values == structure.token_values
                    and len(parsed.bindings) == relation.role_coverage
                    and all((binding.role, binding.start, binding.end,
                             binding.values) in projections
                            for binding in parsed.bindings)):
                return False
    return True


__all__ = ["has_uncovered_closed_relation_modifier"]
