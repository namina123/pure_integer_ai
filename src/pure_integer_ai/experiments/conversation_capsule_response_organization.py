"""M4 资格化命题到可读多段回答的组织接线。

本模块不产生事实、不判断资格，也不依赖 Python 字符串作为核心身份。它只消费
M3 的 response-act 与既有 raw dialogue 结果，把已有 claim、来源依据和限定/修复
载荷排列为可回放的 UTF-8 整数段。未知/澄清/冲突路径不会携带 claim segment。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.learning_input_capsule import digest_bytes
from pure_integer_ai.experiments.conversation_capsule_evidence_bridge import (
    CapsuleEvidenceDialogueTransition,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    decode_utf8_v1,
    encode_utf8_v1,
)
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


SEGMENT_CLAIM = 1
SEGMENT_SUPPORT = 2
SEGMENT_QUALIFIER = 3
SEGMENT_REPAIR = 4
_SEGMENT_KINDS = frozenset({
    SEGMENT_CLAIM, SEGMENT_SUPPORT, SEGMENT_QUALIFIER, SEGMENT_REPAIR,
})
_ACTS = frozenset({"ANSWER", "UNKNOWN", "CLARIFY"})
_ORDER = {
    SEGMENT_CLAIM: 1,
    SEGMENT_SUPPORT: 2,
    SEGMENT_QUALIFIER: 3,
    SEGMENT_REPAIR: 4,
}
RESPONSE_ORGANIZATION_PROTOCOL_V1 = 1


# object-model: exception; interop=portable
class ResponseOrganizationError(ValueError):
    """回答段、response-act 或 UTF-8 组织边界不闭合。"""


def _u8(value: Any, *, label: str, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise ResponseOrganizationError(f"{label} 必须是整数 tuple")
    if not allow_empty and not value:
        raise ResponseOrganizationError(f"{label} 不能为空")
    if any(type(item) is not int or item < 0 or item > 255 for item in value):
        raise ResponseOrganizationError(f"{label} 必须是 0..255 严格整数")
    if value and decode_utf8_v1(value) is None:
        raise ResponseOrganizationError(f"{label} 不是规范 UTF-8")
    return value


def _key(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise ResponseOrganizationError(f"{label} 必须是非空整数 tuple")
    if any(type(item) is not int or item < 0 for item in value):
        raise ResponseOrganizationError(f"{label} 必须是非负严格整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    result.extend((len(value), *value))


# object-model: value; representation=struct; interop=portable
@dataclass(frozen=True, slots=True)
class ResponseSegment:
    """一个可组织的已有 UTF-8 表层片段。"""

    segment_kind: int
    surface_bytes: tuple[int, ...]
    provenance_key: tuple[int, ...]
    ordinal: int

    def __post_init__(self) -> None:
        if type(self.segment_kind) is not int or self.segment_kind not in _SEGMENT_KINDS:
            raise ResponseOrganizationError("segment_kind 未注册")
        _u8(self.surface_bytes, label="segment.surface_bytes")
        _key(self.provenance_key, label="segment.provenance_key")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ResponseOrganizationError("segment.ordinal 必须是非负严格整数")

    def canonical_record(self) -> tuple[int, ...]:
        result = [RESPONSE_ORGANIZATION_PROTOCOL_V1, self.segment_kind,
                  self.ordinal]
        _pack(result, self.surface_bytes)
        _pack(result, self.provenance_key)
        return tuple(result)


# object-model: value; representation=struct; interop=portable
@dataclass(frozen=True, slots=True)
class ResponseOrganizationPlan:
    """response-act 守恒的确定性多段回答计划。"""

    response_act: str
    segments: tuple[ResponseSegment, ...]
    source_identity: tuple[int, ...]
    replay_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.response_act not in _ACTS:
            raise ResponseOrganizationError("response_act 未注册")
        _key(self.source_identity, label="organization.source_identity")
        _key(self.replay_key, label="organization.replay_key")
        if not isinstance(self.segments, tuple) or not self.segments:
            raise ResponseOrganizationError("organization.segments 不能为空")
        if any(not isinstance(item, ResponseSegment) for item in self.segments):
            raise TypeError("organization.segments 类型错误")
        ordered = tuple(sorted(
            self.segments,
            key=lambda item: (_ORDER[item.segment_kind], item.ordinal,
                              item.canonical_record()),
        ))
        if ordered != self.segments:
            raise ResponseOrganizationError("segments 未按组织顺序排列")
        claim_count = sum(item.segment_kind == SEGMENT_CLAIM
                          for item in self.segments)
        support_count = sum(item.segment_kind == SEGMENT_SUPPORT
                            for item in self.segments)
        if self.response_act == "ANSWER":
            if claim_count == 0 or support_count == 0:
                raise ResponseOrganizationError(
                    "ANSWER 必须同时包含 claim 和 support")
        elif claim_count != 0:
            raise ResponseOrganizationError(
                "UNKNOWN/CLARIFY 不得携带 claim segment")
        if self.response_act in {"UNKNOWN", "CLARIFY"} and not any(
                item.segment_kind in {SEGMENT_QUALIFIER, SEGMENT_REPAIR}
                for item in self.segments):
            raise ResponseOrganizationError(
                "UNKNOWN/CLARIFY 必须包含 qualifier 或 repair")

    @property
    def output_bytes(self) -> tuple[int, ...]:
        result: list[int] = []
        for index, segment in enumerate(self.segments):
            if index:
                result.append(10)
            result.extend(segment.surface_bytes)
        return tuple(result)

    @property
    def output_surface(self) -> str:
        scalars = decode_utf8_v1(self.output_bytes)
        if scalars is None:
            raise ResponseOrganizationError("组织结果 UTF-8 回读失败")
        return "".join(chr(item) for item in scalars)

    def canonical_record(self) -> tuple[int, ...]:
        result = [RESPONSE_ORGANIZATION_PROTOCOL_V1,
                  *map(ord, self.response_act), len(self.segments)]
        for segment in self.segments:
            _pack(result, segment.canonical_record())
        _pack(result, self.source_identity)
        _pack(result, self.replay_key)
        return tuple(result)


def _surface_bytes(value: str, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value.strip():
        raise ResponseOrganizationError(f"{label} 必须是非空文本")
    try:
        return tuple(encode_utf8_v1(tuple(ord(item) for item in value.strip())))
    except (TypeError, ValueError) as error:
        raise ResponseOrganizationError(f"{label} 不是合法 UTF-8 文本") from error


def organize_capsule_response(
        transition: CapsuleEvidenceDialogueTransition,
        *,
        support_surfaces: tuple[str, ...] = (),
        fallback_surfaces: tuple[str, ...] = (),
        ) -> ResponseOrganizationPlan:
    """把 M3 结果组织为多段回答，文本只来自既有结果或调用方载荷。"""
    if not isinstance(transition, CapsuleEvidenceDialogueTransition):
        raise TypeError("transition 类型错误")
    act = transition.response_act
    if act not in _ACTS:
        raise ResponseOrganizationError("M3 response_act 不可组织")
    if not isinstance(support_surfaces, tuple) or not isinstance(
            fallback_surfaces, tuple):
        raise TypeError("support/fallback surfaces 必须是 tuple")
    source_key = transition.capsule.identity_key
    segments: list[ResponseSegment] = []
    if act == "ANSWER":
        answer = transition.dual_plane.dialogue.dialogue_turn.answer
        if answer is None or not answer.accepted:
            raise ResponseOrganizationError("ANSWER 缺少既有 dialogue claim")
        segments.append(ResponseSegment(
            SEGMENT_CLAIM, _u8(answer.output_bytes, label="claim"),
            source_key, 0))
        for ordinal, support in enumerate(support_surfaces):
            segments.append(ResponseSegment(
                SEGMENT_SUPPORT, _surface_bytes(support, label="support"),
                source_key, ordinal))
    else:
        for ordinal, fallback in enumerate(fallback_surfaces):
            segments.append(ResponseSegment(
                SEGMENT_REPAIR if act == "CLARIFY" else SEGMENT_QUALIFIER,
                _surface_bytes(fallback, label="fallback"), source_key, ordinal))
    if not segments:
        raise ResponseOrganizationError("没有可组织的回答段")
    segments_tuple = tuple(sorted(
        segments,
        key=lambda item: (_ORDER[item.segment_kind], item.ordinal,
                          item.canonical_record()),
    ))
    replay_record: list[int] = [RESPONSE_ORGANIZATION_PROTOCOL_V1]
    for value in (transition.canonical_record(),
                  tuple(item.canonical_record() for item in segments_tuple)):
        if value and isinstance(value[0], tuple):
            replay_record.append(len(value))
            for nested in value:
                _pack(replay_record, nested)
        else:
            _pack(replay_record, value)
    return ResponseOrganizationPlan(
        act,
        segments_tuple,
        source_key,
        digest_bytes(encode_integer_tuple(tuple(replay_record))),
    )


__all__ = [
    "RESPONSE_ORGANIZATION_PROTOCOL_V1",
    "SEGMENT_CLAIM",
    "SEGMENT_QUALIFIER",
    "SEGMENT_REPAIR",
    "SEGMENT_SUPPORT",
    "ResponseOrganizationError",
    "ResponseOrganizationPlan",
    "ResponseSegment",
    "organize_capsule_response",
]
