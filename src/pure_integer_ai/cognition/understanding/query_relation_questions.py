"""Graph-trained question syntax bound to current Core relation evidence."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pure_integer_ai.cognition.understanding.query_slot_question_composition import ComposedSlotQuestionFrame

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
    STRUCTURE_CLOSED,
    STRUCTURE_OPEN,
    InputRelationQuestionBinding,
    QueryInputStructure,
)
from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRoleFrame,
    match_open_roles,
    pack_record,
    unpack_record,
)


QUESTION_FRAME_VERSION = 92020
QUESTION_ROLE_LIST = 92021
QUESTION_GAP_LIST = 92022
QUESTION_SOURCE_REF = 92023
QUESTION_COMPOSED_FRAME_VERSION = 92027
QUESTION_GAP_SOURCE_LIST = 92028
QUESTION_ANSWER_ROLE = 92030
QUESTION_WH_VALUES = 92031
QUESTION_SLOT_MODE = 92032

# A slot question normally asks Core for an unknown endpoint.  A reference
# slot has the same structural shape, but its omitted endpoint is a discourse
# reference and must be resolved by the current Memory Observation first.
# This is a trained integer contract, not a runtime surface-word rule.
SLOT_MODE_PLAIN = 0
SLOT_MODE_REFERENCE = 1


class QuestionCompositionBudgetExceeded(ValueError):
    """The complete source-backed syntax search exceeds its declared budget."""


@dataclass(frozen=True, slots=True)
class TrainedRelationQuestionFrame:
    """A source-local syntax frame, not a fact or an answer template."""

    source_case_key: tuple[int, ...]
    operation_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    roles: tuple[tuple[int, ...], ...]
    gaps: tuple[tuple[int, ...], ...]
    answer_role_key: tuple[int, ...] = ()
    wh_values: tuple[int, ...] = ()
    slot_mode: int = SLOT_MODE_PLAIN

    def __post_init__(self) -> None:
        if not self.source_case_key or not self.operation_key or not self.predicate_key:
            raise ValueError("question frame has no source or semantic identity")
        if type(self.slot_mode) is not int or self.slot_mode not in {
                SLOT_MODE_PLAIN, SLOT_MODE_REFERENCE}:
            raise ValueError("question frame slot mode is invalid")
        is_slot_question = bool(self.answer_role_key)
        if (type(self.roles) is not tuple
                or len(self.roles) < (1 if is_slot_question else 2)
                or len(set(self.roles)) != len(self.roles)
                or any(not role for role in self.roles)
                or type(self.gaps) is not tuple
                or len(self.gaps) != len(self.roles) + 1):
            raise ValueError("question frame requires distinct roles and ordered gaps")
        if not any(self.gaps):
            raise ValueError("question frame has no learned syntax")
        if is_slot_question:
            if (type(self.answer_role_key) is not tuple
                    or not self.answer_role_key
                    or any(type(value) is not int or value < 0
                           for value in self.answer_role_key)
                    or self.answer_role_key in self.roles
                    or type(self.wh_values) is not tuple or not self.wh_values
                    or any(type(value) is not int or value < 0
                           for value in self.wh_values)
                    or not any(
                        gap[index:index + len(self.wh_values)] == self.wh_values
                        for gap in self.gaps
                        for index in range(len(gap) - len(self.wh_values) + 1))):
                raise ValueError(
                    "slot question requires a distinct answer role and learned WH marker")
        elif self.wh_values:
            raise ValueError("WH values require an answer role")
        elif self.slot_mode != SLOT_MODE_PLAIN:
            raise ValueError("slot mode requires an answer role")
        if self.slot_mode != SLOT_MODE_PLAIN and not is_slot_question:
            raise ValueError("slot mode requires a slot question")
        self.stable_key()

    def stable_key(self) -> tuple[int, ...]:
        if self.answer_role_key:
            fields = [
                QUESTION_FRAME_VERSION,
                self.source_case_key, self.operation_key, self.predicate_key,
                pack_record(QUESTION_ROLE_LIST, *self.roles),
                pack_record(QUESTION_GAP_LIST, *self.gaps),
                pack_record(QUESTION_ANSWER_ROLE, self.answer_role_key),
                pack_record(QUESTION_WH_VALUES, self.wh_values),
            ]
            if self.slot_mode != SLOT_MODE_PLAIN:
                fields.append(pack_record(QUESTION_SLOT_MODE, (self.slot_mode,)))
            return pack_record(*fields)
        return pack_record(
            QUESTION_FRAME_VERSION,
            self.source_case_key, self.operation_key, self.predicate_key,
            pack_record(QUESTION_ROLE_LIST, *self.roles),
            pack_record(QUESTION_GAP_LIST, *self.gaps),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> TrainedRelationQuestionFrame:
        fields = unpack_record(key, QUESTION_FRAME_VERSION)
        if len(fields) not in {5, 7, 8}:
            raise ValueError("question frame field count is invalid")
        answer_role = ()
        wh_values = ()
        slot_mode = SLOT_MODE_PLAIN
        if len(fields) in {7, 8}:
            answer_fields = unpack_record(fields[5], QUESTION_ANSWER_ROLE)
            wh_fields = unpack_record(fields[6], QUESTION_WH_VALUES)
            if len(answer_fields) != 1 or len(wh_fields) != 1:
                raise ValueError("slot question fields are invalid")
            answer_role, = answer_fields
            wh_values, = wh_fields
            if len(fields) == 8:
                mode_fields = unpack_record(fields[7], QUESTION_SLOT_MODE)
                if len(mode_fields) != 1 or len(mode_fields[0]) != 1:
                    raise ValueError("slot question mode is invalid")
                slot_mode, = mode_fields[0]
        return cls(
            fields[0], fields[1], fields[2],
            unpack_record(fields[3], QUESTION_ROLE_LIST),
            unpack_record(fields[4], QUESTION_GAP_LIST),
            answer_role, wh_values, slot_mode,
        )

    @property
    def is_slot_question(self) -> bool:
        return bool(self.answer_role_key)

    @property
    def is_reference_slot(self) -> bool:
        """Whether the omitted answer role is a trained discourse reference."""
        return self.is_slot_question and self.slot_mode == SLOT_MODE_REFERENCE

    def open_frame(self) -> OpenRoleFrame:
        return OpenRoleFrame(
            self.source_case_key, self.predicate_key,
            (QUESTION_SOURCE_REF, *self.source_case_key),
            self.roles, self.gaps,
            (0, len(self.roles) + sum(map(len, self.gaps))),
        )


@dataclass(frozen=True, slots=True)
class ComposedRelationQuestionFrame:
    """A new syntax assembled from train frames of one semantic contract.

    Every gap retains the complete identities of the train cases that supplied
    its exact integer interval.  This is a parse witness, never a Core fact.
    """

    operation_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    roles: tuple[tuple[int, ...], ...]
    gaps: tuple[tuple[int, ...], ...]
    gap_sources: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self) -> None:
        if (not self.operation_key or not self.predicate_key
                or len(self.roles) < 2
                or len(set(self.roles)) != len(self.roles)
                or len(self.gaps) != len(self.roles) + 1
                or len(self.gap_sources) != len(self.gaps)
                or not any(self.gaps)):
            raise ValueError("composed question contract is incomplete")
        if any(not sources or sources != tuple(sorted(set(sources)))
               or any(not source for source in sources)
               for sources in self.gap_sources):
            raise ValueError("composed question gaps need canonical train sources")
        self.stable_key()

    def stable_key(self) -> tuple[int, ...]:
        return pack_record(
            QUESTION_COMPOSED_FRAME_VERSION,
            self.operation_key, self.predicate_key,
            pack_record(QUESTION_ROLE_LIST, *self.roles),
            pack_record(QUESTION_GAP_LIST, *self.gaps),
            pack_record(QUESTION_GAP_SOURCE_LIST, *(
                pack_record(QUESTION_SOURCE_REF, *sources)
                for sources in self.gap_sources)),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> ComposedRelationQuestionFrame:
        fields = unpack_record(key, QUESTION_COMPOSED_FRAME_VERSION)
        if len(fields) != 5:
            raise ValueError("composed question field count is invalid")
        return cls(
            fields[0], fields[1],
            unpack_record(fields[2], QUESTION_ROLE_LIST),
            unpack_record(fields[3], QUESTION_GAP_LIST),
            tuple(unpack_record(item, QUESTION_SOURCE_REF)
                  for item in unpack_record(fields[4], QUESTION_GAP_SOURCE_LIST)),
        )

    def open_frame(self) -> OpenRoleFrame:
        key = self.stable_key()
        return OpenRoleFrame(
            key, self.predicate_key, (QUESTION_SOURCE_REF, *key),
            self.roles, self.gaps,
            (0, len(self.roles) + sum(map(len, self.gaps))),
        )


def _composed_frames(
        frames: tuple[TrainedRelationQuestionFrame, ...],
        values: tuple[int, ...],
        *, max_combinations: int,
        combination_counter: list[int] | None = None,
        ) -> tuple[ComposedRelationQuestionFrame | ComposedSlotQuestionFrame, ...]:
    """只枚举当前输入中的训练间隔，完整及 WH 合同分别组合。"""
    if type(max_combinations) is not int or max_combinations <= 0:
        raise ValueError("question composition budget must be positive")
    if combination_counter is not None and (type(combination_counter) is not list
            or len(combination_counter) != 1 or type(combination_counter[0]) is not int
            or combination_counter[0] != 0):
        raise ValueError("question composition counter must begin at zero")
    grouped: dict[tuple, list[TrainedRelationQuestionFrame]] = {}
    for frame in frames:
        wh_gap_indices = ()
        if frame.is_slot_question:
            from pure_integer_ai.cognition.understanding.query_slot_question_composition import _wh_gap_indices
            wh_gap_indices = _wh_gap_indices(frame)
        grouped.setdefault((frame.operation_key, frame.predicate_key,
                            frame.roles, frame.answer_role_key,
                            frame.wh_values, frame.slot_mode, wh_gap_indices), []).append(frame)
    result = []
    searched = 0
    for (operation, predicate, roles, answer, wh, mode, _wh_gap_indices), owners in sorted(grouped.items()):
        if len(owners) < 2:
            continue
        options = []
        for ordinal in range(len(roles) + 1):
            sources_by_gap: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
            for frame in owners:
                gap = frame.gaps[ordinal]
                if (not gap or any(values[start:start + len(gap)] == gap
                                   for start in range(len(values) - len(gap) + 1))):
                    sources_by_gap.setdefault(gap, set()).add(
                        frame.stable_key() if answer else frame.source_case_key)
            if not sources_by_gap:
                break
            options.append(tuple(sorted(
                (gap, tuple(sorted(sources)))
                for gap, sources in sources_by_gap.items())))
        if len(options) != len(roles) + 1:
            continue
        base_gaps = {frame.gaps for frame in owners}
        for selection in product(*options):
            searched += 1
            if combination_counter is not None:
                combination_counter[0] = searched
            if searched > max_combinations:
                raise QuestionCompositionBudgetExceeded(
                    "question syntax composition exceeds request budget")
            gaps = tuple(item[0] for item in selection)
            if gaps in base_gaps:
                continue
            sources = tuple(item[1] for item in selection)
            if len({source for group in sources for source in group}) < 2:
                continue
            if answer:
                from pure_integer_ai.cognition.understanding.query_slot_question_composition import ComposedSlotQuestionFrame
                result.append(ComposedSlotQuestionFrame(
                    operation, predicate, roles, gaps, answer, wh, mode, sources))
            else:
                result.append(ComposedRelationQuestionFrame(
                    operation, predicate, roles, gaps, sources))
    return tuple(sorted(result, key=lambda item: item.stable_key()))


def relation_question_syntax_predicates(
        frames: tuple[TrainedRelationQuestionFrame, ...],
        structure: QueryInputStructure,
        *, max_combinations: int = 4096,
        ) -> tuple[tuple[int, ...], ...]:
    """返回角色作用域已与独立filler对齐的问式谓词，不授予命题真值。

    请求关系仍可未闭合；精确角色跨度足以限制无关的反向回退。
    单独的训练间隔命中不能吞入连接词或切开filler来清空普通输入锚点。
    """
    if (not frames or not structure.tokens or structure.discourse_compositions
            or structure.discourse_gaps):
        return ()
    complete_frames = tuple(item for item in frames if not item.is_slot_question)
    from pure_integer_ai.cognition.understanding.query_slot_question_composition import available_slot_question_frames
    slot_frames = available_slot_question_frames(
        frames, structure.token_values, max_combinations=max_combinations)
    available = (*complete_frames, *_composed_frames(
        complete_frames, structure.token_values,
        max_combinations=max_combinations))
    available = (*available, *slot_frames)
    if not available:
        return ()
    open_frames = tuple(item.open_frame() for item in available)
    if len({item.stable_key() for item in open_frames}) != len(open_frames):
        raise ValueError("question syntax frames have duplicate parse identities")
    parses = match_open_roles(
        open_frames, structure.token_values, structure.source_ref)
    by_open_key = {
        item.open_frame().stable_key(): item for item in available}
    from pure_integer_ai.cognition.understanding.query_question_syntax_scope import (
        exact_question_filler_intervals,
        question_parse_aligns_with_fillers,
    )
    filler_intervals = exact_question_filler_intervals(structure)
    return tuple(sorted({
        by_open_key[item.frame.stable_key()].predicate_key
        for item in parses
        if question_parse_aligns_with_fillers(item, filler_intervals)
    }))


def match_relation_question_frames(
        frames: tuple[TrainedRelationQuestionFrame, ...],
        structure: QueryInputStructure,
        *, max_combinations: int = 4096,
        ) -> tuple[InputRelationQuestionBinding, ...]:
    """Bind full-input variable slots to one closed, role-correct relation.

    A syntax frame alone has no factual authority. Multiple possible frames,
    role spans, or propositions remain unresolved instead of choosing a score.
    """
    if not frames or not structure.tokens:
        return ()
    if structure.discourse_compositions or structure.discourse_gaps:
        return ()
    complete_frames = tuple(item for item in frames if not item.is_slot_question)
    if len({item.open_frame().stable_key() for item in complete_frames}) != len(complete_frames):
        raise ValueError("question frames have duplicate parse identities")
    available = (*complete_frames, *_composed_frames(
        complete_frames, structure.token_values, max_combinations=max_combinations))
    if len({item.open_frame().stable_key() for item in available}) != len(available):
        raise ValueError("question composition has duplicate parse identities")
    bindings = _bind_question_parses(
        available, structure.token_values, structure, scope_start=0,
        allowed_relation_keys=None)
    return bindings if len(bindings) == 1 else ()


def match_relation_slot_question_frames(
        frames: tuple[TrainedRelationQuestionFrame, ...],
        structure: QueryInputStructure,
        *, allow_dynamic_alternatives: bool = False,
        allow_exact_alternatives: bool = False,
        max_combinations: int = 4096,
        ) -> tuple[InputRelationQuestionBinding, ...]:
    """Bind one observed role to a trained WH question over an open relation.

    Observed slots omit the requested answer role. Each observed role must be
    grounded by an exact Core projection, except for one role whose full input
    span is retained by a matching trained open-role frame. That exception is
    a dynamic reference candidate, not a filler. The WH marker is syntax only
    and never becomes a fact identity. One exact Core resolution takes
    precedence over dynamic candidates; competing exact resolutions remain
    ambiguous.
    """
    if (not frames or not structure.tokens or structure.discourse_compositions
            or structure.discourse_gaps):
        return ()
    if (type(allow_dynamic_alternatives) is not bool
            or type(allow_exact_alternatives) is not bool):
        raise TypeError("question alternative modes must be boolean")
    from pure_integer_ai.cognition.understanding.query_open_roles import match_open_roles

    from pure_integer_ai.cognition.understanding.query_slot_question_composition import available_slot_question_frames
    slot_frames = available_slot_question_frames(
        frames, structure.token_values, max_combinations=max_combinations)
    if not slot_frames:
        return ()
    open_frames = tuple(item.open_frame() for item in slot_frames)
    if len({item.stable_key() for item in open_frames}) != len(open_frames):
        raise ValueError("slot question frames have duplicate parse identities")
    parses = match_open_roles(
        open_frames,
        structure.token_values, structure.source_ref)
    frame_by_key = {item.open_frame().stable_key(): item for item in slot_frames}
    exact_matches = set()
    dynamic_matches = set()
    for parsed in parses:
        frame = frame_by_key[parsed.frame.stable_key()]
        observed_roles = frame.roles
        slot_refs = tuple(
            next((span.ref_key for span in structure.spans
                  if span.start == binding.start and span.end == binding.end
                  and span.values == binding.values
                  and span.source_ref == structure.source_ref), ())
            for binding in parsed.bindings)
        if len(slot_refs) != len(observed_roles) or any(not item for item in slot_refs):
            continue
        for relation in structure.relation_candidates:
            if (relation.predicate_key != frame.predicate_key
                    or relation.state != STRUCTURE_OPEN
                    or relation.required_count != len(observed_roles) + 2
                    or relation.support_count != relation.role_coverage + 1
                    or relation.predicate_coverage != 1
                    or relation.conflict_count != 0):
                continue
            valid = True
            known_refs = []
            dynamic_roles = []
            for role, span_ref in zip(observed_roles, slot_refs):
                span = next((item for item in structure.spans
                             if item.ref_key == span_ref), None)
                if span is None or span.source_ref != structure.source_ref:
                    valid = False
                    break
                semantic = tuple(item for item in structure.semantic_candidates
                                 if item.projection_kind == PROJECTION_FILLER
                                 and item.proposition_key == relation.proposition_key
                                 and item.predicate_key == relation.predicate_key
                                 and item.source_ref == relation.source_ref
                                 and item.role_key == role
                                 and item.span_ref == span_ref
                                 and item.span_ref in relation.span_refs)
                if len(semantic) == 1:
                    known_refs.append(span_ref)
                elif not semantic and len(_matching_open_relation_candidates(
                        structure, frame, relation.proposition_key,
                        relation.source_ref, slot_refs)) == 1:
                    dynamic_roles.append(role)
                else:
                    valid = False
                    break
            if not valid:
                continue
            if (len(dynamic_roles) > 1
                    or relation.role_coverage != len(observed_roles) - len(dynamic_roles)
                    or relation.required_open != 1 + len(dynamic_roles)):
                continue
            proposition_members = tuple(
                item for item in structure.semantic_candidates
                if item.proposition_key == relation.proposition_key
                and item.predicate_key == relation.predicate_key
                and item.source_ref == relation.source_ref
                and item.span_ref in relation.span_refs)
            predicate_members = tuple(item for item in proposition_members
                                       if item.projection_kind == PROJECTION_PREDICATE)
            filler_members = tuple(item for item in proposition_members
                                   if item.projection_kind == PROJECTION_FILLER)
            predicate_gap_witness = False
            if not predicate_members and relation.predicate_coverage == 1:
                # A trained slot frame may carry the predicate entirely in a
                # learned gap (for example a reverse MEMBER surface).  The
                # bridge promotes that exact frame before this matcher runs;
                # require the same unique open-role parse here so a missing
                # predicate can never be guessed from endpoint order.
                predicate_gap_matches = _matching_open_relation_candidates(
                    structure, frame, relation.proposition_key,
                    relation.source_ref, slot_refs)
                predicate_gap_witness = (
                    len(predicate_gap_matches) == 1
                    and predicate_gap_matches[0].frame.roles == frame.roles
                    and predicate_gap_matches[0].frame.gaps == frame.gaps)
            if ((len(predicate_members) != 1 and not predicate_gap_witness)
                    or len(filler_members) != len(known_refs)
                    or {item.role_key for item in filler_members}
                    != {role for role, span_ref in zip(observed_roles, slot_refs)
                        if span_ref in known_refs}
                    or frame.answer_role_key in {
                        item.role_key for item in filler_members}):
                continue
            predicate_span = (predicate_members[0].span_ref
                              if predicate_members else None)
            expected_relation_spans = tuple(sorted({
                *known_refs, *( () if predicate_span is None
                                else (predicate_span,) ),
            }))
            if expected_relation_spans != relation.span_refs:
                continue
            if predicate_span is not None and predicate_span not in relation.span_refs:
                continue
            target = dynamic_matches if dynamic_roles else exact_matches
            target.add(InputRelationQuestionBinding(
                frame.stable_key(), frame.operation_key, relation.stable_key(),
                slot_refs, frame.answer_role_key))
    def _deduplicate_provenance(
            bindings: set[InputRelationQuestionBinding],
            ) -> tuple[InputRelationQuestionBinding, ...]:
        """按当前语义绑定合并同形训练来源，保留最小稳定来源键。

        同一个 operation、关系命题、已观测槽位和答案 Role 可以由多个
        独立 train case 物化。它们是同一输入合同的来源证据，不是两个
        竞争答案；把 source frame key 直接拿来计数会把重复课程来源误报
        为歧义。关系、槽位或答案 Role 不同的绑定仍然保留为冲突。
        """
        groups: dict[tuple[object, ...], list[InputRelationQuestionBinding]] = {}
        for binding in bindings:
            contract = (
                binding.operation_key,
                binding.relation_key,
                binding.slot_span_refs,
                binding.answer_role_key,
            )
            groups.setdefault(contract, []).append(binding)
        return tuple(sorted(
            (min(items, key=lambda item: item.frame_key)
             for items in groups.values()),
            key=lambda item: item.stable_key()))

    exact_result = _deduplicate_provenance(exact_matches)
    if exact_result:
        # 显式桥接模式只提交候选，不宣称多候选已经等价；消费者仍须
        # 校验完整 Core 端点和实际答案，默认调用继续拒绝精确歧义。
        return exact_result if len(exact_result) == 1 or allow_exact_alternatives else ()
    dynamic_result = _deduplicate_provenance(dynamic_matches)
    if len(dynamic_result) == 1 or (dynamic_result and allow_dynamic_alternatives):
        return dynamic_result
    return ()


def _matching_open_relation_candidates(
        structure: QueryInputStructure,
        frame: TrainedRelationQuestionFrame | ComposedSlotQuestionFrame,
        proposition_key: tuple[int, ...],
        relation_source_ref: tuple[int, ...],
        slot_span_refs: tuple[tuple[int, ...], ...],
        ):
    """Find exact trained open-role parses for one candidate Core proposition."""
    spans_by_interval = {
        (item.start, item.end, item.values, item.source_ref): item.ref_key
        for item in structure.spans
    }
    result = []
    for candidate in structure.open_relations:
        if (candidate.frame.proposition != proposition_key
                or candidate.frame.predicate != frame.predicate_key
                or candidate.frame.source_ref != relation_source_ref
                or candidate.frame.roles != frame.roles
                or candidate.frame.gaps != frame.gaps
                or candidate.source_ref != structure.source_ref
                or candidate.values != structure.token_values):
            continue
        candidate_refs = tuple(
            spans_by_interval.get((binding.start, binding.end,
                                   binding.values, binding.source_ref))
            for binding in candidate.bindings)
        if candidate_refs == slot_span_refs:
            result.append(candidate)
    return tuple(sorted(set(result), key=lambda item: item.stable_key()))


def _bind_question_parses(
        available: tuple[TrainedRelationQuestionFrame | ComposedRelationQuestionFrame, ...],
        values: tuple[int, ...], structure: QueryInputStructure, *,
        scope_start: int,
        allowed_relation_keys: frozenset[tuple[int, ...]] | None,
        ) -> tuple[InputRelationQuestionBinding, ...]:
    """Join exact syntax slots to current Core projections in one integer scope."""
    parses = match_open_roles(
        tuple(item.open_frame() for item in available),
        values, structure.source_ref,
    )
    by_open_key = {item.open_frame().stable_key(): item for item in available}
    spans_by_interval = {
        (span.start, span.end): span.ref_key for span in structure.spans
    }
    bindings = set()
    for parsed in parses:
        frame = by_open_key[parsed.frame.stable_key()]
        slot_refs = tuple(spans_by_interval.get((
            scope_start + slot.start, scope_start + slot.end))
                          for slot in parsed.bindings)
        if any(ref is None for ref in slot_refs):
            continue
        for relation in structure.relation_candidates:
            if (relation.state != STRUCTURE_CLOSED
                    or relation.predicate_key != frame.predicate_key
                    or relation.role_coverage != len(frame.roles)
                    or (allowed_relation_keys is not None
                        and relation.stable_key() not in allowed_relation_keys)):
                continue
            for role, span_ref in zip(frame.roles, slot_refs):
                matching = {
                    item.candidate_key
                    for item in structure.semantic_candidates
                    if item.projection_kind == PROJECTION_FILLER
                    and item.proposition_key == relation.proposition_key
                    and item.predicate_key == relation.predicate_key
                    and item.source_ref == relation.source_ref
                    and item.role_key == role
                    and item.span_ref == span_ref
                    and span_ref in relation.span_refs
                }
                if len(matching) != 1:
                    break
            else:
                bindings.add(InputRelationQuestionBinding(
                    frame.stable_key(), frame.operation_key,
                    relation.stable_key(), slot_refs,
                ))
    return tuple(sorted(bindings, key=lambda item: item.stable_key()))


__all__ = [
    "QUESTION_FRAME_VERSION", "QUESTION_COMPOSED_FRAME_VERSION",
    "ComposedRelationQuestionFrame", "QuestionCompositionBudgetExceeded",
    "TrainedRelationQuestionFrame",
    "relation_question_syntax_predicates",
    "match_relation_slot_question_frames",
    "match_relation_question_frames",
]
