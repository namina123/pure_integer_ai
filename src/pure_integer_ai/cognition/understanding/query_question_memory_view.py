"""合并已逐项核验的指代作用域，仅交接候选，不裁决来源或实体。"""
from dataclasses import replace

from pure_integer_ai.cognition.understanding.query_input_structure import (
    QueryInputStructure,
)
from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate, OpenRoleFrame, OpenRoleSpan, match_open_roles,
)
from pure_integer_ai.cognition.understanding.query_slot_question_composition import (
    restore_slot_question_frame,
)


def reference_question_open_candidate(
        structure, frame, relation, binding, *, require_retained=False):
    """由已核验训练问式重建完整角色发生；既有投影必须与当前解析逐项相同。

    调用者核验当前挂载训练来源和 Core；此函数只验证语法、当前来源及跨度，
    不填实体、不构造事实证据。它同时用于首次交接和重复核验，避免两条路径漂移。
    """
    if type(require_retained) is not bool:
        raise TypeError("完整指代投影复核模式必须是布尔值")
    if (not frame.is_reference_slot or binding.frame_key != frame.stable_key()
            or binding.operation_key != frame.operation_key
            or binding.answer_role_key != frame.answer_role_key
            or binding.relation_key != relation.stable_key()
            or relation.predicate_key != frame.predicate_key
            or structure.discourse_compositions or structure.discourse_gaps):
        return None
    observed_frame = OpenRoleFrame(
        relation.proposition_key, frame.predicate_key, relation.source_ref,
        frame.roles, frame.gaps, (0, len(frame.roles) + sum(map(len, frame.gaps))))
    matching = []
    for parsed in match_open_roles((observed_frame,), structure.token_values,
                                   structure.source_ref):
        refs = []
        for item in parsed.bindings:
            spans = tuple(span for span in structure.spans
                          if span.start == item.start and span.end == item.end
                          and span.values == item.values
                          and span.source_ref == structure.source_ref)
            if len(spans) != 1:
                break
            refs.append(spans[0].ref_key)
        else:
            if tuple(refs) == binding.slot_span_refs:
                matching.append(parsed)
    if len(matching) != 1:
        return None
    values, marker = structure.token_values, frame.wh_values
    starts = tuple(index for index in range(len(values) - len(marker) + 1)
                   if values[index:index + len(marker)] == marker)
    if len(starts) != 1:
        return None
    ordered = tuple(sorted((
        *((item.start, item.end, item.values, item.role) for item in matching[0].bindings),
        (starts[0], starts[0] + len(marker), marker, frame.answer_role_key),
    ), key=lambda item: (item[0], item[1], item[3])))
    if (len({(item[0], item[1]) for item in ordered}) != len(ordered)
            or {item[3] for item in ordered} != {*frame.roles, frame.answer_role_key}):
        return None
    gaps, cursor = [], 0
    for start, end, _values, _role in ordered:
        if start < cursor:
            return None
        gaps.append(values[cursor:start])
        cursor = end
    gaps.append(values[cursor:])
    roles = tuple(item[3] for item in ordered)
    full_frame = OpenRoleFrame(
        relation.proposition_key, frame.predicate_key, relation.source_ref,
        roles, tuple(gaps), (0, len(roles) + sum(map(len, gaps))))
    spans = tuple(OpenRoleSpan(role, structure.source_ref, ordinal, start, end, part)
                  for ordinal, (start, end, part, role) in enumerate(ordered))
    expected = OpenRelationCandidate(full_frame, structure.source_ref, values, spans)
    # 首次交接已有精确观察角色解析，普通 Core 完整框架可能把前缀并入代词槽。
    # 只有当前训练问式、物理来源及全部观察绑定完全一致，才能替换这种普通切分。
    observed = tuple(item for item in structure.open_relations
                     if item.frame.proposition == relation.proposition_key
                     and item.frame.predicate == frame.predicate_key
                     and item.frame.source_ref == relation.source_ref
                     and item.frame.roles == frame.roles and item.frame.gaps == frame.gaps
                     and item.source_ref == structure.source_ref
                     and item.values == values and item.bindings == matching[0].bindings)
    if not require_retained and len(observed) == 1:
        return expected
    retained = tuple(item for item in structure.open_relations
                     if item.frame.proposition == relation.proposition_key
                     and item.frame.predicate == frame.predicate_key
                     and any(part.role == frame.answer_role_key for part in item.bindings))
    if len(retained) != 1 or retained[0] != expected:
        return None
    return expected


def retained_reference_question_gap_witness(structure, frame, relation, binding):
    """已绑定完整角色投影须精确重建，才能继续证明训练 gap 的谓词语法。"""
    if binding not in structure.relation_question_bindings:
        return False
    candidate = reference_question_open_candidate(
        structure, frame, relation, binding, require_retained=True)
    return candidate is not None and candidate in structure.open_relations


def merge_reference_question_views(
        structure: QueryInputStructure,
        views: tuple[QueryInputStructure, ...],
        ) -> QueryInputStructure:
    """按完整稳定键合并全部指代候选，保留输入身份与竞争绑定。

    调用者先用当前训练图及 Core 逐项验证各视图。本函数只接受每绑定一个
    同输入 reference 视图；不能用缺失、重复或跨输入视图隐式选择来源。
    """
    bindings = structure.relation_question_bindings
    if (len(bindings) < 2 or len(views) != len(bindings)
            or structure.discourse_compositions or structure.discourse_gaps):
        raise ValueError("指代交接必须覆盖全部无组合问式绑定")
    selected_fields = frozenset({
        "operation_keys", "carrier_candidates", "semantic_candidates",
        "relation_candidates", "relation_question_bindings", "open_relations",
    })
    seen = []
    for view in views:
        if len(view.relation_question_bindings) != 1:
            raise ValueError("每个指代视图必须只含一个已验证绑定")
        binding = view.relation_question_bindings[0]
        frame = restore_slot_question_frame(binding.frame_key)
        if (binding not in bindings or not frame.is_reference_slot
                or not view.open_relations):
            raise ValueError("指代视图缺少当前绑定或完整角色候选")
        for name in QueryInputStructure.__dataclass_fields__:
            if name not in selected_fields and getattr(view, name) != getattr(structure, name):
                raise ValueError("指代视图改变了当前输入或非问式合同")
        seen.append(binding)
    if tuple(sorted(seen, key=lambda item: item.stable_key())) != bindings:
        raise ValueError("指代交接丢失或重复了竞争绑定")
    combined = {}
    for name in selected_fields - {"operation_keys", "relation_question_bindings"}:
        members = {item.stable_key(): item for view in views
                   for item in getattr(view, name)}
        combined[name] = tuple(members[key] for key in sorted(members))
    return replace(structure, **combined)
