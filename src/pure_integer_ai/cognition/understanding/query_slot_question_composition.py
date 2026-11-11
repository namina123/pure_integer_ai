"""有来源的 WH 间隔组合；只扩展语法，不提供事实或答案端点。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRoleFrame, pack_record, unpack_record,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    QUESTION_FRAME_VERSION, QUESTION_GAP_LIST, QUESTION_ROLE_LIST,
    QUESTION_SOURCE_REF, SLOT_MODE_REFERENCE, TrainedRelationQuestionFrame,
)

COMPOSED_SLOT_FRAME_VERSION = 92046
COMPOSED_SLOT_PARENT_LIST = 92047


def slot_contract(frame) -> tuple:
    """保留语法组合不能越过的全部角色及操作合同。"""
    return (frame.operation_key, frame.predicate_key, frame.roles,
            frame.answer_role_key, frame.wh_values, frame.slot_mode,
            _wh_gap_indices(frame))


def _wh_gap_indices(frame) -> tuple[int, ...]:
    """恢复 WH 原子所在的训练间隔序位，禁止跨序位拼接。"""
    result = []
    for ordinal, gap in enumerate(frame.gaps):
        if any(gap[index:index + len(frame.wh_values)] == frame.wh_values
               for index in range(len(gap) - len(frame.wh_values) + 1)):
            result.append(ordinal)
    if len(result) != 1:
        raise ValueError("slot question WH marker must occupy one gap")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ComposedSlotQuestionFrame:
    """每个间隔保留完整父帧的一等整数语法对象，不保存示例端点。"""

    operation_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    roles: tuple[tuple[int, ...], ...]
    gaps: tuple[tuple[int, ...], ...]
    answer_role_key: tuple[int, ...]
    wh_values: tuple[int, ...]
    slot_mode: int
    gap_sources: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self) -> None:
        """验证每个间隔的原始序位、完整合同和至少两个独立训练来源。"""
        syntax = TrainedRelationQuestionFrame(
            (COMPOSED_SLOT_FRAME_VERSION,), self.operation_key, self.predicate_key,
            self.roles, self.gaps, self.answer_role_key, self.wh_values, self.slot_mode)
        if not syntax.is_slot_question or len(self.gap_sources) != len(self.gaps):
            raise ValueError("composed WH syntax lacks answer or gap sources")
        parents = set()
        for ordinal, keys in enumerate(self.gap_sources):
            if not keys or keys != tuple(sorted(set(keys))):
                raise ValueError("composed WH gap sources are not canonical")
            for key in keys:
                parent = TrainedRelationQuestionFrame.from_stable_key(key)
                if (slot_contract(parent) != slot_contract(self)
                        or parent.gaps[ordinal] != self.gaps[ordinal]):
                    raise ValueError("composed WH gap source contract drifted")
                parents.add(key)
        if len(parents) < 2:
            raise ValueError("composed WH syntax requires multiple train frames")

    @property
    def is_slot_question(self) -> bool:
        """组合对象始终保留一个待 Core 或 Memory 解析的答案角色。"""
        return True

    @property
    def is_reference_slot(self) -> bool:
        """仅保留父帧显式训练的指代合同，不从表面推断。"""
        return self.slot_mode == SLOT_MODE_REFERENCE

    def stable_key(self) -> tuple[int, ...]:
        """完整编码语法及逐间隔父帧；不以签名替代来源成员。"""
        return pack_record(
            COMPOSED_SLOT_FRAME_VERSION, self.operation_key, self.predicate_key,
            pack_record(QUESTION_ROLE_LIST, *self.roles),
            pack_record(QUESTION_GAP_LIST, *self.gaps), self.answer_role_key,
            self.wh_values, (self.slot_mode,),
            pack_record(COMPOSED_SLOT_PARENT_LIST, *(
                pack_record(QUESTION_SOURCE_REF, *keys) for keys in self.gap_sources)))

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> ComposedSlotQuestionFrame:
        """从规范整数成员恢复组合对象，并重新核验完整父帧合同。"""
        fields = unpack_record(key, COMPOSED_SLOT_FRAME_VERSION)
        if len(fields) != 8 or len(fields[6]) != 1:
            raise ValueError("composed WH frame field count is invalid")
        return cls(fields[0], fields[1],
                   unpack_record(fields[2], QUESTION_ROLE_LIST),
                   unpack_record(fields[3], QUESTION_GAP_LIST),
                   fields[4], fields[5], fields[6][0],
                   tuple(unpack_record(item, QUESTION_SOURCE_REF) for item in
                         unpack_record(fields[7], COMPOSED_SLOT_PARENT_LIST)))

    def open_frame(self) -> OpenRoleFrame:
        """将同一来源对象交给通用开放角色解析器，不填充未知端点。"""
        key = self.stable_key()
        return OpenRoleFrame(key, self.predicate_key, (QUESTION_SOURCE_REF, *key),
                             self.roles, self.gaps,
                             (0, len(self.roles) + sum(map(len, self.gaps))))


def available_slot_question_frames(frames, values, *, max_combinations=4096):
    """在固定预算内返回原训练帧及只由当前整数间隔构成的组合。"""
    from pure_integer_ai.cognition.understanding.query_relation_questions import _composed_frames
    from pure_integer_ai.cognition.understanding.query_question_prefix import prefixed_slot_question_frames

    slots = tuple(frame for frame in frames if frame.is_slot_question)
    counter = [0]
    composed = _composed_frames(slots, values, max_combinations=max_combinations,
                                combination_counter=counter)
    prefixed = prefixed_slot_question_frames(frames, values, covered_frames=composed,
        max_combinations=max_combinations, initial_combinations=counter[0])
    return (*slots, *composed, *prefixed)


def restore_slot_question_frame(key, *, trained_frames=None):
    """恢复问式；生产答案验证额外要求所有完整父帧属于当前挂载模型。"""
    if not key:
        raise ValueError("slot question frame key is empty")
    if key[0] == QUESTION_FRAME_VERSION:
        frame = TrainedRelationQuestionFrame.from_stable_key(key)
        parents = (key,)
    else:
        from pure_integer_ai.cognition.understanding.query_question_prefix import (
            PREFIXED_SLOT_FRAME, PrefixedSlotQuestionFrame,
        )
        if key[0] == PREFIXED_SLOT_FRAME:
            frame = PrefixedSlotQuestionFrame.from_stable_key(key)
            parents = tuple(parent.stable_key() for parent in frame.source_frames)
        else:
            frame = ComposedSlotQuestionFrame.from_stable_key(key)
            parents = tuple(parent for group in frame.gap_sources for parent in group)
    if not frame.is_slot_question:
        raise ValueError("slot question frame has no answer role")
    if trained_frames is not None:
        allowed = {item.stable_key() for item in trained_frames}
        if any(parent not in allowed for parent in parents):
            raise ValueError("composed WH source is absent from the mounted model")
    return frame


def composed_slot_open_role_frames(frames, bound_frames, values):
    """将组合语法绑定到已有 Core 来源框架，保留其命题与物理来源身份。"""
    available = available_slot_question_frames(frames, values)
    from pure_integer_ai.cognition.understanding.query_question_prefix import PrefixedSlotQuestionFrame
    result = set()
    for frame in available:
        if isinstance(frame, PrefixedSlotQuestionFrame):
            parents = {frame.base}
        elif isinstance(frame, ComposedSlotQuestionFrame):
            parents = {TrainedRelationQuestionFrame.from_stable_key(key)
                       for keys in frame.gap_sources for key in keys}
        else:
            continue
        parent_gaps = {parent.gaps for parent in parents}
        for bound in bound_frames:
            if (bound.predicate != frame.predicate_key or bound.roles != frame.roles
                    or bound.gaps not in parent_gaps):
                continue
            result.add(OpenRoleFrame(
                bound.proposition, bound.predicate, bound.source_ref,
                frame.roles, frame.gaps,
                (0, len(frame.roles) + sum(map(len, frame.gaps)))))
    return tuple(sorted(result, key=lambda item: item.stable_key()))
