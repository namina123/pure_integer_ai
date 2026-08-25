"""把广域对话用户输入显式追加到既有 Runtime Memory。

这里只记录用户输入的观察事件，不把 QA 答案、来源检索结果或训练输出写成
Memory proposition，也不触发 Core promotion。每个会话有稳定的 source/scope，
每一轮有独立 memory item key；重复回放走既有 DUPLICATE 语义。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.learning_input_capsule import (
    ADMISSION_ACCEPTED,
    ADMISSION_DUPLICATE,
    LearningInputCapsule,
    RuntimeMemoryEvent,
    RuntimeMemoryState,
    append_runtime_event,
    digest_bytes,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    DialogueTurn,
)
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


BROAD_DIALOGUE_RUNTIME_MEMORY_PROTOCOL_V1 = 1
BROAD_DIALOGUE_RUNTIME_SOURCE_KIND_V1 = 1
BROAD_DIALOGUE_RUNTIME_MEMORY_ITEM_KIND_V1 = 1
BROAD_DIALOGUE_RUNTIME_LANGUAGE_V1 = 1
BROAD_DIALOGUE_RUNTIME_MODALITY_TEXT_V1 = 1


class BroadDialogueRuntimeMemoryError(ValueError):
    """广域对话 Runtime Memory scope、capsule 或 append 边界不闭合。"""


def _strict_conversation_key(value: tuple[int, ...]) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise BroadDialogueRuntimeMemoryError("conversation key 必须是非空非负整数 tuple")
    return value


def _conversation_digest(conversation_key: tuple[int, ...]) -> tuple[int, ...]:
    return digest_bytes(encode_integer_tuple(
        (BROAD_DIALOGUE_RUNTIME_MEMORY_PROTOCOL_V1, *conversation_key)))


def _conversation_numeric_id(conversation_key: tuple[int, ...]) -> int:
    value = int.from_bytes(bytes(_conversation_digest(conversation_key)), "big")
    return max(1, value)


def runtime_memory_source_for_conversation(
        conversation_key: tuple[int, ...],
        ) -> SourceRef:
    """由会话整数 key 派生稳定 Runtime 观察来源。"""
    key = _strict_conversation_key(conversation_key)
    numeric = _conversation_numeric_id(key)
    return SourceRef(
        BROAD_DIALOGUE_RUNTIME_SOURCE_KIND_V1,
        numeric,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def runtime_memory_scope_for_conversation(
        conversation_key: tuple[int, ...],
        ):
    """返回绑定同一 source 的 session scope。"""
    key = _strict_conversation_key(conversation_key)
    source = runtime_memory_source_for_conversation(key)
    return session_scope(_conversation_numeric_id(key), source=source)


def empty_runtime_memory_for_conversation(
        conversation_key: tuple[int, ...],
        ) -> RuntimeMemoryState:
    """建立一个只属于该会话的空 Runtime Memory 账本。"""
    return RuntimeMemoryState(
        runtime_memory_scope_for_conversation(conversation_key).stable_key())


def _capsule_for_turn(
        conversation_key: tuple[int, ...], turn: DialogueTurn,
        ) -> LearningInputCapsule:
    key = _strict_conversation_key(conversation_key)
    if not isinstance(turn, DialogueTurn):
        raise TypeError("dialogue turn 类型错误")
    if type(turn.ordinal) is not int or turn.ordinal < 0:
        raise BroadDialogueRuntimeMemoryError("turn ordinal 非法")
    raw = turn.question.encode("utf-8")
    if not raw:
        raise BroadDialogueRuntimeMemoryError("turn question 不能为空")
    source = runtime_memory_source_for_conversation(key)
    scope = runtime_memory_scope_for_conversation(key)
    sequence = turn.ordinal + 1
    return LearningInputCapsule(
        source=source,
        scope=scope,
        version_key=(BROAD_DIALOGUE_RUNTIME_MEMORY_PROTOCOL_V1, *key, sequence),
        parent_version_key=(),
        language=BROAD_DIALOGUE_RUNTIME_LANGUAGE_V1,
        modality=BROAD_DIALOGUE_RUNTIME_MODALITY_TEXT_V1,
        raw_content_digest=digest_bytes(raw),
        structural_units=((1, len(raw)),),
        authority_key=(BROAD_DIALOGUE_RUNTIME_MEMORY_PROTOCOL_V1, *source.stable_key()),
        license_id="runtime-dialogue-v1",
        split=1,
        delta_sequence=sequence,
    )


def _memory_item_key(conversation_key: tuple[int, ...], ordinal: int) -> tuple[int, ...]:
    key = _strict_conversation_key(conversation_key)
    if type(ordinal) is not int or ordinal < 0:
        raise BroadDialogueRuntimeMemoryError("memory item ordinal 非法")
    return (BROAD_DIALOGUE_RUNTIME_MEMORY_ITEM_KIND_V1, *key, ordinal + 1)


@dataclass(frozen=True, slots=True)
class BroadDialogueRuntimeMemoryAppend:
    """一次用户输入进入 Runtime Memory 的可回放结果。"""

    turn: DialogueTurn
    event: RuntimeMemoryEvent
    memory_before: RuntimeMemoryState
    memory_after: RuntimeMemoryState
    admission_status: int

    def __post_init__(self) -> None:
        if not isinstance(self.turn, DialogueTurn):
            raise TypeError("append turn 类型错误")
        if not isinstance(self.event, RuntimeMemoryEvent):
            raise TypeError("append event 类型错误")
        if self.event.capsule.scope.stable_key() != self.memory_before.scope_key:
            raise BroadDialogueRuntimeMemoryError("event 与 memory_before scope 漂移")
        if self.event.capsule.scope.stable_key() != self.memory_after.scope_key:
            raise BroadDialogueRuntimeMemoryError("event 与 memory_after scope 漂移")
        if self.event.event_key not in tuple(item.event_key for item in self.memory_after.events):
            raise BroadDialogueRuntimeMemoryError("memory_after 缺少 event")
        if self.admission_status not in (ADMISSION_ACCEPTED, ADMISSION_DUPLICATE):
            raise BroadDialogueRuntimeMemoryError("append admission 未注册")


def append_dialogue_turn_to_runtime_memory(
        state: RuntimeMemoryState,
        conversation_key: tuple[int, ...],
        turn: DialogueTurn,
        ) -> BroadDialogueRuntimeMemoryAppend:
    """将一轮用户输入追加到 Runtime Memory，重复回放保持幂等。"""
    if not isinstance(state, RuntimeMemoryState):
        raise TypeError("runtime memory state 类型错误")
    key = _strict_conversation_key(conversation_key)
    if state.scope_key != runtime_memory_scope_for_conversation(key).stable_key():
        raise BroadDialogueRuntimeMemoryError("state 与 conversation scope 不一致")
    capsule = _capsule_for_turn(key, turn)
    event = RuntimeMemoryEvent(capsule, _memory_item_key(key, turn.ordinal))
    after, admission = append_runtime_event(state, event)
    if admission not in (ADMISSION_ACCEPTED, ADMISSION_DUPLICATE):
        raise BroadDialogueRuntimeMemoryError(
            f"Runtime Memory append 被拒绝: {admission}")
    return BroadDialogueRuntimeMemoryAppend(turn, event, state, after, admission)


def replay_dialogue_state_to_runtime_memory(
        state: BroadDialogueState,
        ) -> RuntimeMemoryState:
    """从当前 bounded dialogue state 重建空账本中的用户输入事件。"""
    if not isinstance(state, BroadDialogueState):
        raise TypeError("broad dialogue state 类型错误")
    memory = empty_runtime_memory_for_conversation(state.conversation_key)
    for turn in state.turns:
        memory = append_dialogue_turn_to_runtime_memory(
            memory, state.conversation_key, turn).memory_after
    return memory


__all__ = [
    "BROAD_DIALOGUE_RUNTIME_MEMORY_ITEM_KIND_V1",
    "BROAD_DIALOGUE_RUNTIME_MEMORY_PROTOCOL_V1",
    "BROAD_DIALOGUE_RUNTIME_SOURCE_KIND_V1",
    "BroadDialogueRuntimeMemoryAppend",
    "BroadDialogueRuntimeMemoryError",
    "append_dialogue_turn_to_runtime_memory",
    "empty_runtime_memory_for_conversation",
    "replay_dialogue_state_to_runtime_memory",
    "runtime_memory_scope_for_conversation",
    "runtime_memory_source_for_conversation",
]
