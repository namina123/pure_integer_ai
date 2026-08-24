"""B2 同一 capsule 的 Core/Runtime 双平面增量接线。

调用方必须显式提供 ``CoreDelta``，因此 Core 的 graph diff、基线和状态归属
仍由已有理解/学习链决定。本模块只负责把这个 delta 与 Runtime Memory 及
现有 raw dialogue turn 放进同一个不可变 transition；不会自动晋升、猜事实，
也不会让 Runtime event 反向写 Core。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.learning_input_capsule import (
    ADMISSION_ACCEPTED,
    ADMISSION_DUPLICATE,
    CoreDelta,
    CoreLearningState,
    LearningInputCapsule,
    LearningReplayReceipt,
    PROJECTION_CORE,
    consume_core_delta,
    digest_bytes,
)
from pure_integer_ai.experiments.conversation_capsule_dialogue_bridge import (
    CapsuleDialogueBridgeError,
    CapsuleDialogueTransition,
    run_capsule_dialogue_turn,
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
)
from pure_integer_ai.cognition.shared.learning_input_capsule import (
    RuntimeMemoryState,
)
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


CAPSULE_DUAL_PLANE_PROTOCOL_V1 = 1
CORE_STATE_RECORD_V1 = 1


# object-model: exception; interop=portable
class CapsuleDualPlaneError(ValueError):
    """Core/Runtime 双平面 transition 不满足边界。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    result.extend((len(value), *value))


def _core_state_record(state: CoreLearningState) -> tuple[int, ...]:
    """将 Core ledger 投影为可跨语言重建的整数 record。"""
    result: list[int] = [CORE_STATE_RECORD_V1]
    for value in (state.scope_key, state.base_state_identity):
        _pack(result, value)
    result.append(len(state.consumed_item_ledger))
    for value in state.consumed_item_ledger:
        _pack(result, value)
    result.append(len(state.deltas))
    for delta in state.deltas:
        _pack(result, delta.stable_key())
    return tuple(result)


def _core_replay_key(
        capsule: LearningInputCapsule,
        delta: CoreDelta,
        after: CoreLearningState,
        ) -> tuple[int, ...]:
    record: list[int] = [CAPSULE_DUAL_PLANE_PROTOCOL_V1]
    for value in (
            capsule.canonical_record,
            delta.stable_key(),
            _core_state_record(after),
    ):
        _pack(record, value)
    return digest_bytes(encode_integer_tuple(tuple(record)))


# object-model: value; representation=struct; interop=portable
@dataclass(frozen=True, slots=True)
class CapsuleDualPlaneTransition:
    """同一输入的 Core delta、Runtime event 和对话结果。"""

    capsule: LearningInputCapsule
    core_delta: CoreDelta
    core_before: CoreLearningState
    core_after: CoreLearningState
    core_admission_status: int
    core_receipt: LearningReplayReceipt
    dialogue: CapsuleDialogueTransition

    def __post_init__(self) -> None:
        if not isinstance(self.capsule, LearningInputCapsule):
            raise TypeError("dual transition capsule 类型错误")
        if not isinstance(self.core_delta, CoreDelta):
            raise TypeError("dual transition core_delta 类型错误")
        if self.core_delta.capsule != self.capsule:
            raise CapsuleDualPlaneError("Core delta 与 capsule 漂移")
        if not isinstance(self.core_before, CoreLearningState):
            raise TypeError("dual transition core_before 类型错误")
        if not isinstance(self.core_after, CoreLearningState):
            raise TypeError("dual transition core_after 类型错误")
        if self.core_before.scope_key != self.core_after.scope_key:
            raise CapsuleDualPlaneError("Core scope 漂移")
        if self.core_admission_status not in (ADMISSION_ACCEPTED, ADMISSION_DUPLICATE):
            raise CapsuleDualPlaneError("Core admission status 未注册")
        if not isinstance(self.core_receipt, LearningReplayReceipt):
            raise TypeError("dual transition core_receipt 类型错误")
        if self.core_receipt.projection_kind != PROJECTION_CORE:
            raise CapsuleDualPlaneError("Core receipt projection kind 漂移")
        if self.core_receipt.input_identity != self.capsule.identity_key:
            raise CapsuleDualPlaneError("Core receipt input identity 漂移")
        if self.core_receipt.output_identity != digest_bytes(
                encode_integer_tuple(_core_state_record(self.core_after))):
            raise CapsuleDualPlaneError("Core receipt output identity 漂移")
        if not isinstance(self.dialogue, CapsuleDialogueTransition):
            raise TypeError("dual transition dialogue 类型错误")
        if self.dialogue.capsule != self.capsule:
            raise CapsuleDualPlaneError("Runtime dialogue capsule 漂移")

    def canonical_record(self) -> tuple[int, ...]:
        result: list[int] = [
            CAPSULE_DUAL_PLANE_PROTOCOL_V1,
            self.core_admission_status,
        ]
        for value in (
                self.capsule.canonical_record,
                self.core_delta.stable_key(),
                _core_state_record(self.core_before),
                _core_state_record(self.core_after),
                self.core_receipt.stable_key(),
                self.dialogue.canonical_record(),
        ):
            _pack(result, value)
        return tuple(result)


def run_capsule_dual_plane_turn(
        capsule: LearningInputCapsule,
        core_delta: CoreDelta,
        core_state: CoreLearningState,
        raw_input_bytes: tuple[int, ...],
        dialogue_state: ConversationRawDialogueState,
        runtime_memory_state: RuntimeMemoryState,
        runtime: PublicDialogueRuntimeV1,
        *,
        preparation_cache: PublicCoursePreparationCache | None = None,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> CapsuleDualPlaneTransition:
    """显式消费同一 capsule 的 Core delta 与 Runtime/Dialogue 路径。"""
    if not isinstance(capsule, LearningInputCapsule):
        raise TypeError("capsule 类型错误")
    if not isinstance(core_delta, CoreDelta):
        raise TypeError("core_delta 类型错误")
    if core_delta.capsule != capsule:
        raise CapsuleDualPlaneError("Core delta 与 capsule 不一致")
    if not isinstance(core_state, CoreLearningState):
        raise TypeError("core_state 类型错误")
    if core_delta.base_state_identity != core_state.base_state_identity:
        raise CapsuleDualPlaneError("Core delta 基线与 state 不一致")
    if core_delta.capsule.scope.stable_key() != core_state.scope_key:
        raise CapsuleDualPlaneError("capsule 与 Core scope 不一致")

    core_after, core_status = consume_core_delta(core_state, core_delta)
    if core_status not in (ADMISSION_ACCEPTED, ADMISSION_DUPLICATE):
        raise CapsuleDualPlaneError(f"Core delta 被拒绝: {core_status}")

    try:
        dialogue = run_capsule_dialogue_turn(
            capsule,
            raw_input_bytes,
            dialogue_state,
            runtime_memory_state,
            runtime,
            preparation_cache=preparation_cache,
            preflight_cache=preflight_cache,
        )
    except CapsuleDialogueBridgeError as error:
        raise CapsuleDualPlaneError("Runtime/Dialogue 平面拒绝输入") from error

    core_receipt = LearningReplayReceipt(
        projection_kind=PROJECTION_CORE,
        input_identity=capsule.identity_key,
        output_identity=digest_bytes(
            encode_integer_tuple(_core_state_record(core_after))),
        status=core_delta.status,
        replay_key=_core_replay_key(capsule, core_delta, core_after),
    )
    return CapsuleDualPlaneTransition(
        capsule,
        core_delta,
        core_state,
        core_after,
        core_status,
        core_receipt,
        dialogue,
    )


__all__ = [
    "CAPSULE_DUAL_PLANE_PROTOCOL_V1",
    "CORE_STATE_RECORD_V1",
    "CapsuleDualPlaneError",
    "CapsuleDualPlaneTransition",
    "run_capsule_dual_plane_turn",
]
