"""T1-G6：source-qualified shadow 的多轮焦点与 response-act outer state。

本模块只消费已经完成 T1-G4 的 shadow result。它不解析 raw 文本、不写会话/Memory，
只验证同一 proposition 在 ANSWER -> UNKNOWN -> CLARIFY 序列中的焦点 revision、
上下文身份和零替换边界。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_t1_shadow_adapter import (
    RawT1ShadowAdapterResult,
)


RAW_T1_SHADOW_DIALOGUE_PROTOCOL_V1 = 1
FOCUS_NONE = 0
FOCUS_PRESENT = 1


class RawT1ShadowDialogueError(ValueError):
    """多轮 T1 shadow 的焦点、response-act 或 revision 不满足合同。"""


def _text(value: str, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RawT1ShadowDialogueError(f"{where} 必须是无首尾空白的非空字符串")
    return value


def _key(value: tuple[int, ...], where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise RawT1ShadowDialogueError(f"{where} 必须是非空非负整数 tuple")
    return value


@dataclass(frozen=True, slots=True)
class RawT1ShadowDialogueState:
    """多轮旁路状态；focus 只保存当前 qualified proposition identity。"""

    conversation_key: tuple[int, ...]
    next_operation_ordinal: int = 1
    focus_revision: int = 0
    proposition_id: str | None = None
    source_id: str | None = None
    context_id: str | None = None
    family_id: str | None = None
    last_response_act: str | None = None
    last_state: str | None = None

    def __post_init__(self) -> None:
        _key(self.conversation_key, "state.conversation_key")
        if type(self.next_operation_ordinal) is not int or self.next_operation_ordinal <= 0:
            raise RawT1ShadowDialogueError("state.next_operation_ordinal 非法")
        if type(self.focus_revision) is not int or self.focus_revision < 0:
            raise RawT1ShadowDialogueError("state.focus_revision 非法")
        fields = (self.proposition_id, self.source_id, self.context_id, self.family_id)
        if self.proposition_id is None:
            if any(item is not None for item in fields[1:]) or self.focus_revision != 0:
                raise RawT1ShadowDialogueError("空 focus state 携带了部分 identity")
            if self.last_response_act is not None or self.last_state is not None:
                raise RawT1ShadowDialogueError("空 focus state 携带了旧 response")
        else:
            for name, value in zip(
                    ("proposition_id", "source_id", "context_id", "family_id"), fields):
                _text(value, f"state.{name}")
            if self.focus_revision <= 0:
                raise RawT1ShadowDialogueError("有 focus state 的 revision 必须为正")
            if self.last_response_act not in {"ANSWER", "UNKNOWN", "CLARIFY"}:
                raise RawT1ShadowDialogueError("state.last_response_act 未注册")
            if self.last_state not in {"SUPPORTED", "UNKNOWN", "CONFLICT"}:
                raise RawT1ShadowDialogueError("state.last_state 未注册")

    @property
    def focus_present(self) -> bool:
        """返回当前是否有同一 proposition focus。"""
        return self.proposition_id is not None

    def canonical_record(self) -> tuple[int, ...]:
        """返回含 optional focus 的纯整数 state record。"""
        result = [RAW_T1_SHADOW_DIALOGUE_PROTOCOL_V1]
        result.extend((len(self.conversation_key), *self.conversation_key,
                       self.next_operation_ordinal, self.focus_revision))
        if self.proposition_id is None:
            result.append(FOCUS_NONE)
        else:
            result.append(FOCUS_PRESENT)
            for value in (
                    self.proposition_id, self.source_id, self.context_id,
                    self.family_id, self.last_response_act, self.last_state):
                scalars = tuple(ord(item) for item in value)
                result.extend((len(scalars), *scalars))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class RawT1ShadowDialogueTurn:
    """一轮 shadow transition 及其前后 focus state。"""

    before: RawT1ShadowDialogueState
    adapter_result: RawT1ShadowAdapterResult
    after: RawT1ShadowDialogueState

    def __post_init__(self) -> None:
        if not isinstance(self.before, RawT1ShadowDialogueState):
            raise TypeError("turn.before 类型错误")
        if not isinstance(self.adapter_result, RawT1ShadowAdapterResult):
            raise TypeError("turn.adapter_result 类型错误")
        if not isinstance(self.after, RawT1ShadowDialogueState):
            raise TypeError("turn.after 类型错误")
        if self.after.conversation_key != self.before.conversation_key:
            raise RawT1ShadowDialogueError("turn conversation key 漂移")
        if self.after.next_operation_ordinal != self.before.next_operation_ordinal + 1:
            raise RawT1ShadowDialogueError("turn operation ordinal 漂移")
        if self.adapter_result.replaced != 0:
            raise RawT1ShadowDialogueError("turn shadow 不得替换旧答案")

    def canonical_record(self) -> tuple[int, ...]:
        """返回 transition 的完整整数记录。"""
        before = self.before.canonical_record()
        adapter = self.adapter_result.canonical_record()
        after = self.after.canonical_record()
        return (
            RAW_T1_SHADOW_DIALOGUE_PROTOCOL_V1,
            len(before), *before, len(adapter), *adapter, len(after), *after,
        )


def start_raw_t1_shadow_dialogue(
        conversation_key: tuple[int, ...],
        ) -> RawT1ShadowDialogueState:
    """建立无 focus 的纯值 outer state。"""
    return RawT1ShadowDialogueState(_key(conversation_key, "conversation_key"))


def run_raw_t1_shadow_dialogue_turn(
        state: RawT1ShadowDialogueState,
        adapter_result: RawT1ShadowAdapterResult,
        ) -> RawT1ShadowDialogueTurn:
    """按固定 response-act 顺序推进 focus，不产生任何 surface。"""
    if not isinstance(state, RawT1ShadowDialogueState):
        raise TypeError("state 类型错误")
    if not isinstance(adapter_result, RawT1ShadowAdapterResult):
        raise TypeError("adapter_result 类型错误")
    consumer = adapter_result.consumer
    if not state.focus_present:
        if consumer.response_act != "ANSWER" or consumer.state != "SUPPORTED":
            raise RawT1ShadowDialogueError("无 focus 的首轮必须是 SUPPORTED ANSWER")
    else:
        if (consumer.proposition_id != state.proposition_id
                or consumer.context_id != state.context_id
                or consumer.family_id != state.family_id):
            raise RawT1ShadowDialogueError("follow-up proposition/context/family 漂移")
        allowed = {
            "ANSWER": {"UNKNOWN", "CLARIFY"},
            "UNKNOWN": {"CLARIFY"},
            "CLARIFY": set(),
        }[state.last_response_act]
        if consumer.response_act not in allowed:
            raise RawT1ShadowDialogueError("follow-up response-act 顺序不允许")
    after = RawT1ShadowDialogueState(
        state.conversation_key,
        state.next_operation_ordinal + 1,
        state.focus_revision + 1,
        consumer.proposition_id,
        consumer.source_id,
        consumer.context_id,
        consumer.family_id,
        consumer.response_act,
        consumer.state,
    )
    return RawT1ShadowDialogueTurn(state, adapter_result, after)


__all__ = [
    "FOCUS_NONE", "FOCUS_PRESENT", "RAW_T1_SHADOW_DIALOGUE_PROTOCOL_V1",
    "RawT1ShadowDialogueError", "RawT1ShadowDialogueState",
    "RawT1ShadowDialogueTurn", "run_raw_t1_shadow_dialogue_turn",
    "start_raw_t1_shadow_dialogue",
]
