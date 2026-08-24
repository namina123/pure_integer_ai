"""B1 capsule 到公开 raw 对话 runtime 的最小纵切。

这个模块只负责把已经构造好的 ``LearningInputCapsule`` 接入两个既有平面：
Runtime Memory 的 append-only event，以及 ``DLG-RAW-04`` 对话 transition。
它不解析文本、不生成领域事实，也不把 Runtime Memory 自动晋升为 Core。
所有 bridge identity 都由规范整数 record 派生，宿主 bytes 只在入口被复制为
整数 tuple。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.learning_input_capsule import (
    ADMISSION_ACCEPTED,
    ADMISSION_DUPLICATE,
    LearningInputCapsule,
    LearningReplayReceipt,
    PROJECTION_RUNTIME,
    RuntimeMemoryEvent,
    RuntimeMemoryState,
    STATUS_OBSERVED,
    append_runtime_event,
    digest_bytes,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
)
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationPreflightCache,
)
from pure_integer_ai.experiments.conversation_raw_dialogue_session import (
    ConversationRawDialogueState,
    ConversationRawDialogueTurn,
    run_public_frame_dialogue_turn,
)
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


CAPSULE_DIALOGUE_BRIDGE_PROTOCOL_V1 = 1
CAPSULE_DIALOGUE_MEMORY_ITEM_KIND_V1 = 1


# object-model: exception; interop=portable
class CapsuleDialogueBridgeError(ValueError):
    """capsule、memory scope 或既有对话 runtime 不满足 bridge 契约。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    result.extend((len(value), *value))


def _strict_byte_vector(value: Any) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise CapsuleDialogueBridgeError("raw_input_bytes 必须是整数 tuple")
    if any(type(item) is not int or item < 0 or item > 255 for item in value):
        raise CapsuleDialogueBridgeError("raw_input_bytes 必须是 0..255 严格整数")
    return value


def _memory_item_key(capsule: LearningInputCapsule) -> tuple[int, ...]:
    """由 capsule identity 形成稳定 memory item key，不读取表层文本。"""
    return (CAPSULE_DIALOGUE_MEMORY_ITEM_KIND_V1, *capsule.identity_key)


def _state_record(state: RuntimeMemoryState) -> tuple[int, ...]:
    result = [CAPSULE_DIALOGUE_BRIDGE_PROTOCOL_V1]
    _pack(result, state.scope_key)
    result.append(len(state.events))
    for event in state.events:
        _pack(result, event.event_key)
    return tuple(result)


def _bridge_replay_key(
        capsule: LearningInputCapsule,
        event: RuntimeMemoryEvent,
        turn: ConversationRawDialogueTurn,
        ) -> tuple[int, ...]:
    """用完整整数 transition 派生可跨语言回放 key。"""
    # Admission (ACCEPTED versus DUPLICATE) is an operational outcome and is
    # deliberately excluded so an idempotent replay receives the same receipt.
    record: list[int] = [CAPSULE_DIALOGUE_BRIDGE_PROTOCOL_V1]
    for value in (
            capsule.canonical_record, event.event_key,
            turn.canonical_record(),
    ):
        _pack(record, value)
    return digest_bytes(encode_integer_tuple(tuple(record)))


# object-model: value; representation=struct; interop=portable
@dataclass(frozen=True, slots=True)
class CapsuleDialogueTransition:
    """一次 capsule 输入同时经过 Runtime Memory 与对话 runtime 的结果。"""

    capsule: LearningInputCapsule
    runtime_event: RuntimeMemoryEvent
    memory_before: RuntimeMemoryState
    memory_after: RuntimeMemoryState
    admission_status: int
    dialogue_turn: ConversationRawDialogueTurn
    receipt: LearningReplayReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.capsule, LearningInputCapsule):
            raise TypeError("transition capsule 类型错误")
        if not isinstance(self.runtime_event, RuntimeMemoryEvent):
            raise TypeError("transition runtime_event 类型错误")
        if self.runtime_event.capsule != self.capsule:
            raise CapsuleDialogueBridgeError("transition capsule/event 漂移")
        if not isinstance(self.memory_before, RuntimeMemoryState):
            raise TypeError("transition memory_before 类型错误")
        if not isinstance(self.memory_after, RuntimeMemoryState):
            raise TypeError("transition memory_after 类型错误")
        if self.memory_before.scope_key != self.memory_after.scope_key:
            raise CapsuleDialogueBridgeError("transition memory scope 漂移")
        if self.runtime_event.capsule.scope.stable_key() != self.memory_before.scope_key:
            raise CapsuleDialogueBridgeError("transition event 不在 memory scope 内")
        if self.admission_status not in (ADMISSION_ACCEPTED, ADMISSION_DUPLICATE):
            raise CapsuleDialogueBridgeError("transition admission status 未注册")
        if not isinstance(self.dialogue_turn, ConversationRawDialogueTurn):
            raise TypeError("transition dialogue_turn 类型错误")
        if not isinstance(self.receipt, LearningReplayReceipt):
            raise TypeError("transition receipt 类型错误")
        if self.receipt.projection_kind != PROJECTION_RUNTIME:
            raise CapsuleDialogueBridgeError("receipt projection kind 漂移")
        if self.runtime_event.event_key not in tuple(
                item.event_key for item in self.memory_after.events):
            raise CapsuleDialogueBridgeError("memory_after 缺少本次 runtime event")
        expected_output = digest_bytes(
            encode_integer_tuple(self.dialogue_turn.canonical_record()))
        if self.receipt.output_identity != expected_output:
            raise CapsuleDialogueBridgeError("receipt output identity 漂移")
        if self.receipt.input_identity != self.capsule.identity_key:
            raise CapsuleDialogueBridgeError("receipt input identity 漂移")

    def canonical_record(self) -> tuple[int, ...]:
        """导出 transition 的完整整数身份，供 replay 和跨语言核验。"""
        result = [CAPSULE_DIALOGUE_BRIDGE_PROTOCOL_V1, self.admission_status]
        for value in (
                self.capsule.canonical_record,
                self.runtime_event.event_key,
                _state_record(self.memory_before),
                _state_record(self.memory_after),
                self.dialogue_turn.canonical_record(),
                self.receipt.stable_key(),
        ):
            _pack(result, value)
        return tuple(result)


def run_capsule_dialogue_turn(
        capsule: LearningInputCapsule,
        raw_input_bytes: tuple[int, ...],
        dialogue_state: ConversationRawDialogueState,
        runtime_memory_state: RuntimeMemoryState,
        runtime: PublicDialogueRuntimeV1,
        *,
        preparation_cache: PublicCoursePreparationCache | None = None,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> CapsuleDialogueTransition:
    """执行一次 capsule -> Runtime Memory -> 对话的确定性 transition。

    同一 capsule 在同一 memory state 上重放时 admission 为 DUPLICATE，但仍会
    从给定 dialogue state 重新产生同一 dialogue turn 和 receipt；冲突、scope
    漂移和 raw digest 不匹配均 fail closed。
    """
    if not isinstance(capsule, LearningInputCapsule):
        raise TypeError("capsule 类型错误")
    if not isinstance(dialogue_state, ConversationRawDialogueState):
        raise TypeError("dialogue_state 类型错误")
    if not isinstance(runtime_memory_state, RuntimeMemoryState):
        raise TypeError("runtime_memory_state 类型错误")
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("runtime 类型错误")
    raw = _strict_byte_vector(raw_input_bytes)
    if digest_bytes(bytes(raw)) != capsule.raw_content_digest:
        raise CapsuleDialogueBridgeError("raw input digest 与 capsule 不匹配")
    if runtime_memory_state.scope_key != capsule.scope.stable_key():
        raise CapsuleDialogueBridgeError("capsule 与 Runtime Memory scope 不一致")

    event = RuntimeMemoryEvent(capsule, _memory_item_key(capsule))
    memory_after, admission_status = append_runtime_event(
        runtime_memory_state, event)
    if admission_status not in (ADMISSION_ACCEPTED, ADMISSION_DUPLICATE):
        raise CapsuleDialogueBridgeError(
            f"Runtime Memory append 被拒绝: {admission_status}")

    turn = run_public_frame_dialogue_turn(
        dialogue_state,
        raw,
        runtime,
        preparation_cache=preparation_cache,
        preflight_cache=preflight_cache,
    )
    output_identity = digest_bytes(
        encode_integer_tuple(turn.canonical_record()))
    receipt = LearningReplayReceipt(
        projection_kind=PROJECTION_RUNTIME,
        input_identity=capsule.identity_key,
        output_identity=output_identity,
        status=STATUS_OBSERVED,
        replay_key=_bridge_replay_key(
            capsule, event, turn),
    )
    return CapsuleDialogueTransition(
        capsule,
        event,
        runtime_memory_state,
        memory_after,
        admission_status,
        turn,
        receipt,
    )


__all__ = [
    "CAPSULE_DIALOGUE_BRIDGE_PROTOCOL_V1",
    "CAPSULE_DIALOGUE_MEMORY_ITEM_KIND_V1",
    "CapsuleDialogueBridgeError",
    "CapsuleDialogueTransition",
    "run_capsule_dialogue_turn",
]
