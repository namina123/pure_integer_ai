"""按当前整数问式和已观察角色选择有界历史窗口，不读取历史内容。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, QueryInputStructure,
)
from pure_integer_ai.cognition.understanding.query_slot_question_composition import (
    restore_slot_question_frame,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    TrainedRelationQuestionFrame,
)


def reference_marker_sources(
        values: tuple[int, ...], role_key: tuple[int, ...],
        predicate_key: tuple[int, ...],
        trained_frames: tuple[TrainedRelationQuestionFrame, ...],
        ) -> tuple[tuple[int, ...], ...]:
    """由当前模型的 reference 帧授权完整角色跨度，不把未知名称当作指代。"""
    return tuple(sorted({frame.stable_key() for frame in trained_frames
                         if frame.is_reference_slot
                         and frame.wh_values == values
                         and frame.answer_role_key == role_key
                         and frame.predicate_key == predicate_key}))


def extended_reference_context_required(
        structure: QueryInputStructure,
        trained_frames: tuple[TrainedRelationQuestionFrame, ...] = (),
        ) -> bool:
    """只有有来源的当前指代义务才能扩大历史，无锚点开放猜测保持单步。

    普通 WH 的观察槽均已绑定，不需要历史；显式指代槽或动态观察槽还须
    有当前模型的指代原子来源。无问式绑定时，至少一个已观察角色及唯一
    指代角色必须同属当前开放关系，禁止用未知名称或无关投影扩大召回。
    """
    if not structure.open_relations:
        return False
    relations = {item.stable_key(): item for item in structure.relation_candidates}
    spans = {item.ref_key: item for item in structure.spans}
    if structure.relation_question_bindings:
        for binding in structure.relation_question_bindings:
            if not binding.answer_role_key:
                continue
            relation = relations.get(binding.relation_key)
            if relation is None:
                continue
            try:
                frame = restore_slot_question_frame(binding.frame_key)
            except (TypeError, ValueError):
                continue
            if (frame.operation_key != binding.operation_key
                    or frame.predicate_key != relation.predicate_key
                    or frame.answer_role_key != binding.answer_role_key
                    or len(frame.roles) != len(binding.slot_span_refs)):
                continue
            if frame.is_reference_slot:
                if reference_marker_sources(frame.wh_values, frame.answer_role_key,
                                            frame.predicate_key, trained_frames):
                    return True
                continue
            observed = tuple(tuple(item for item in structure.semantic_candidates
                                  if item.projection_kind == PROJECTION_FILLER
                                  and item.proposition_key == relation.proposition_key
                                  and item.predicate_key == relation.predicate_key
                                  and item.source_ref == relation.source_ref
                                  and item.role_key == role and item.span_ref == ref)
                             for role, ref in zip(frame.roles, binding.slot_span_refs))
            if all(len(items) <= 1 for items in observed) and sum(
                    not items for items in observed) == 1:
                role, ref = next((role, ref) for role, ref, items in zip(
                    frame.roles, binding.slot_span_refs, observed) if not items)
                span = spans.get(ref)
                if span is not None and reference_marker_sources(
                        span.values, role, frame.predicate_key, trained_frames):
                    return True
        return False
    spans_by_interval = {
        (item.start, item.end, item.values): item.ref_key
        for item in structure.spans if item.source_ref == structure.source_ref}
    for candidate in structure.open_relations:
        observed_roles = set()
        ambiguous = False
        for binding in candidate.bindings:
            ref = spans_by_interval.get((binding.start, binding.end, binding.values))
            members = tuple(item for item in structure.semantic_candidates
                            if item.projection_kind == PROJECTION_FILLER
                            and item.proposition_key == candidate.frame.proposition
                            and item.predicate_key == candidate.frame.predicate
                            and item.source_ref == candidate.frame.source_ref
                            and item.role_key == binding.role and item.span_ref == ref)
            if len(members) == 1:
                observed_roles.add(binding.role)
            elif members:
                ambiguous = True
        if (not ambiguous and observed_roles
                and len(set(candidate.frame.roles) - observed_roles) == 1):
            unresolved = next(role for role in candidate.frame.roles
                              if role not in observed_roles)
            missing = tuple(item for item in candidate.bindings if item.role == unresolved)
            if len(missing) == 1 and reference_marker_sources(
                    missing[0].values, unresolved, candidate.frame.predicate, trained_frames):
                return True
    return False
