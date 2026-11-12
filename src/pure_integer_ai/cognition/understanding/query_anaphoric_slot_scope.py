"""从完整来源化问式保留纯语法候选，实体和事实仍等待当前指代证据。"""
from dataclasses import replace

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, STRUCTURE_OPEN, InputRelationCandidate,
)
from pure_integer_ai.cognition.understanding.query_reference_context import reference_marker_sources
from pure_integer_ai.cognition.understanding.query_slot_question_composition import available_slot_question_frames


def attach_anaphoric_slot_syntax(structure, trained_frames, core_structures):
    """仅给全输入单指代观察槽补语法候选，不投影对象或选择事实来源。

    谓词可完全位于受训gap内；没有当前语义成员时，空span集合明确表示
    尚未观察到任何Core成员。当前完整开放解析、同源Core角色拓扑及独立
    指代授权必须同时存在。后续WH、Memory和Core消费者继续核验全部来源。
    """
    if (not structure.tokens or not structure.open_relations
            or structure.relation_question_bindings or structure.assertion_scope_bindings
            or structure.discourse_compositions or structure.discourse_gaps):
        return structure
    frames = available_slot_question_frames(trained_frames, structure.token_values)
    by_shape = {}
    for frame in frames:
        if not frame.is_reference_slot and len(frame.roles) == 1:
            by_shape.setdefault((frame.predicate_key, frame.roles, frame.gaps), []).append(frame)
    cores = {}
    for core in core_structures:
        cores.setdefault((core.proposition_key, core.predicate_key, core.source_ref), []).append(core)
    existing = {(row.proposition_key, row.predicate_key, row.source_ref)
                for row in structure.relation_candidates}
    semantic = {(row.proposition_key, row.predicate_key, row.source_ref)
                for row in structure.semantic_candidates}
    spans = {(span.start, span.end, span.values, span.source_ref)
             for span in structure.spans}
    added = set()
    for parsed in structure.open_relations:
        frame = parsed.frame
        identity = (frame.proposition, frame.predicate, frame.source_ref)
        core_rows = cores.get(identity, ())
        if (identity in existing or identity in semantic or len(core_rows) != 1
                or parsed.source_ref != structure.source_ref
                or parsed.values != structure.token_values or len(parsed.bindings) != 1):
            continue
        observed, = parsed.bindings
        if ((observed.start, observed.end, observed.values, observed.source_ref) not in spans
                or observed.role not in frame.roles
                or not reference_marker_sources(observed.values, observed.role,
                                                frame.predicate, trained_frames)):
            continue
        core, = core_rows
        roles = tuple(member.role_key for member in core.members
                      if member.projection_kind == PROJECTION_FILLER)
        if (len(core.members) != len(roles) + 1 or len(set(roles)) != len(roles)
                or sum(member.projection_kind == PROJECTION_PREDICATE
                       for member in core.members) != 1):
            continue
        for question in by_shape.get((frame.predicate, frame.roles, frame.gaps), ()):
            if (set(roles) != {*question.roles, question.answer_role_key}
                    or len(roles) != len(question.roles) + 1):
                continue
            added.add(InputRelationCandidate(core.proposition_key, core.predicate_key,
                core.relation_structure_key(), core.carrier_key, core.source_ref, (),
                len(core.members), 1, 0, 1, len(core.members)-1, STRUCTURE_OPEN, 0))
    if not added:
        return structure
    return replace(structure, relation_candidates=tuple(sorted(
        {*structure.relation_candidates, *added}, key=lambda row: row.stable_key())))
