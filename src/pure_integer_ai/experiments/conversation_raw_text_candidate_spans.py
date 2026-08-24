"""T1-G16：从 raw UTF-8 整数序列提取机械句界候选。

该模块只做物理切分：按已冻结的标点 scalar 形成不重叠候选 span，不猜词义、角色、命题、
来源或对话行为。候选必须经过后续 evidence/qualification 才能进入训练 pack。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    ConversationRawIntake,
    intake_raw_conversation_vector,
)


RAW_TEXT_CANDIDATE_PROTOCOL_V1 = 1
RAW_TEXT_CANDIDATE_BOUNDARY_TERMINAL = 1
RAW_TEXT_CANDIDATE_BOUNDARY_RESIDUAL = 2

# CJK/ASCII sentence boundaries. This is a mechanical registry, not a language label.
_TERMINAL_SCALARS = frozenset({33, 46, 59, 63, 12290, 65307, 65281, 65311})


class RawTextCandidateError(ValueError):
    """机械候选 span 输入或不变量越界。"""


def _u8(value: tuple[int, ...], where: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or any(
            type(item) is not int or item < 0 or item > 255 for item in value):
        raise RawTextCandidateError(f"{where} 必须是 0..255 整数 tuple")
    return value


def _nonnegative(value: int, where: str) -> None:
    if type(value) is not int or value < 0:
        raise RawTextCandidateError(f"{where} 必须是非负严格整数")


# object-model: value; representation=struct; interop=T1-G16
@dataclass(frozen=True, slots=True)
class RawTextCandidateSpan:
    """一个只含物理位置和机械边界来源的候选 span。"""

    ordinal: int
    start_scalar: int
    end_scalar: int
    start_byte: int
    end_byte: int
    boundary_kind: int

    def __post_init__(self) -> None:
        for name in ("ordinal", "start_scalar", "end_scalar", "start_byte", "end_byte"):
            _nonnegative(getattr(self, name), f"candidate.{name}")
        if self.end_scalar <= self.start_scalar or self.end_byte <= self.start_byte:
            raise RawTextCandidateError("candidate span 必须非空")
        if self.boundary_kind not in {
                RAW_TEXT_CANDIDATE_BOUNDARY_TERMINAL,
                RAW_TEXT_CANDIDATE_BOUNDARY_RESIDUAL,
        }:
            raise RawTextCandidateError("candidate boundary_kind 未注册")

    def canonical_record(self) -> tuple[int, ...]:
        return (RAW_TEXT_CANDIDATE_PROTOCOL_V1, self.ordinal,
                self.start_scalar, self.end_scalar,
                self.start_byte, self.end_byte, self.boundary_kind)


@dataclass(frozen=True, slots=True)
class RawTextCandidateExtraction:
    """raw intake 与机械候选的只读结果；拒绝时不携带候选。"""

    intake: ConversationRawIntake
    candidates: tuple[RawTextCandidateSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.intake, ConversationRawIntake):
            raise TypeError("candidate extraction 需要 ConversationRawIntake")
        if not isinstance(self.candidates, tuple):
            raise TypeError("candidate extraction candidates 必须是 tuple")
        if not self.intake.accepted and self.candidates:
            raise RawTextCandidateError("raw 拒绝时不得携带候选")
        if tuple(item.ordinal for item in self.candidates) != tuple(
                range(len(self.candidates))):
            raise RawTextCandidateError("candidate ordinal 必须连续")

    @property
    def accepted(self) -> bool:
        return self.intake.result_code == DLG_RAW_ACCEPT

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_TEXT_CANDIDATE_PROTOCOL_V1, self.intake.result_code,
                  len(self.candidates)]
        for item in self.candidates:
            result.extend(item.canonical_record())
        return tuple(result)


def _byte_offsets(scalars: tuple[int, ...]) -> tuple[int, ...]:
    offsets = [0]
    for value in scalars:
        if value <= 0x7F:
            width = 1
        elif value <= 0x7FF:
            width = 2
        elif value <= 0xFFFF:
            width = 3
        else:
            width = 4
        offsets.append(offsets[-1] + width)
    return tuple(offsets)


def extract_raw_text_candidate_spans(
        raw_input_bytes: tuple[int, ...],
        ) -> RawTextCandidateExtraction:
    """按冻结句界标点切分 raw 输入，输出只含整数位置的候选。"""
    raw = _u8(raw_input_bytes, "candidate raw input")
    intake = intake_raw_conversation_vector(raw)
    if not intake.accepted:
        return RawTextCandidateExtraction(intake)
    scalars = intake.unicode_scalars
    offsets = _byte_offsets(scalars)
    candidates: list[RawTextCandidateSpan] = []
    start = 0
    index = 0
    while index < len(scalars):
        if scalars[index] in _TERMINAL_SCALARS:
            end = index + 1
            while end < len(scalars) and scalars[end] in _TERMINAL_SCALARS:
                end += 1
            if start < end:
                candidates.append(RawTextCandidateSpan(
                    len(candidates), start, end, offsets[start], offsets[end],
                    RAW_TEXT_CANDIDATE_BOUNDARY_TERMINAL))
            start = end
            index = end
            continue
        index += 1
    if start < len(scalars):
        candidates.append(RawTextCandidateSpan(
            len(candidates), start, len(scalars), offsets[start], offsets[-1],
            RAW_TEXT_CANDIDATE_BOUNDARY_RESIDUAL))
    return RawTextCandidateExtraction(intake, tuple(candidates))


__all__ = [
    "RAW_TEXT_CANDIDATE_BOUNDARY_RESIDUAL",
    "RAW_TEXT_CANDIDATE_BOUNDARY_TERMINAL",
    "RAW_TEXT_CANDIDATE_PROTOCOL_V1",
    "RawTextCandidateError",
    "RawTextCandidateExtraction",
    "RawTextCandidateSpan",
    "extract_raw_text_candidate_spans",
]
