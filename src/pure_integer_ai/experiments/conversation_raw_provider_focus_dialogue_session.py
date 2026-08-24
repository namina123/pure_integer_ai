"""DLG-RAW-12：在冻结 V4 session 外叠加可替换的话语焦点。

V2 mixed context 继续只保存真实 Frame/provider turn。这里的 outer state 只保存
一个严格绑定其当前 provider tail 的短暂 focus，因此 follow-up 的 answer 不会被
误写为独立事实或跨锚点长期记忆。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationPreflightCache,
)
from pure_integer_ai.experiments.conversation_provider_origin_context import (
    MIXED_CONTEXT_WRITE_ORIGIN_NONE,
    MixedContextReadV2,
)
from pure_integer_ai.experiments.conversation_provider_origin_focus_chain import (
    ProviderOriginDiscourseFocusV1,
    ProviderOriginFocusChainError,
    focus_from_provider_origin_followup_v1,
    run_provider_origin_focus_followup_v1,
)
from pure_integer_ai.experiments.conversation_provider_origin_followup import (
    ProviderOriginFollowupResultV1,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PublicProofSentenceProviderResultV1,
)
from pure_integer_ai.experiments.conversation_raw_answer_runtime import (
    ConversationRawAnswerResult,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    ConversationRawIntake,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_mixed_dialogue_session import (
    ConversationRawMixedDialogueStateV2,
    ConversationRawMixedDialogueTurnV4,
    run_public_mixed_frame_dialogue_turn_v4,
    start_public_mixed_frame_dialogue,
)


RAW_PROVIDER_FOCUS_DIALOGUE_STATE_RECORD_V1 = 1
RAW_PROVIDER_FOCUS_DIALOGUE_TURN_RECORD_V1 = 1
RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_NONE = 0
RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_PRESENT = 1


# object-model: exception; interop=DLG-RAW-12
class ConversationRawProviderFocusDialogueSessionError(ValueError):
    """V5 outer state、carrier 分型或焦点生命周期不闭合。"""


def _u8_vector(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 raw input 仅通过唯一的 u8 tuple 进入 session。"""
    if (type(value) is not tuple
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ConversationRawProviderFocusDialogueSessionError(
            f"{label} 必须是 0..255 严格整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """写入 count-framed finite integer segment。"""
    if (type(value) is not tuple
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawProviderFocusDialogueSessionError(
            "provider focus dialogue canonical segment 非法")
    result.extend((len(value), *value))


# object-model: value; representation=struct; interop=DLG-RAW-12
@dataclass(frozen=True, slots=True)
class ConversationRawProviderFocusDialogueStateV1:
    """V5 outer session：冻结 V4 state 加上可替换的同锚点焦点。"""

    mixed_state: ConversationRawMixedDialogueStateV2
    focus: ProviderOriginDiscourseFocusV1 | None = None

    def __post_init__(self) -> None:
        """仅接受已闭合 V4 state 与显式 focus optional 分支。"""
        if type(self.mixed_state) is not ConversationRawMixedDialogueStateV2:
            raise TypeError("provider focus dialogue mixed state 类型错误")
        if (self.focus is not None
                and type(self.focus) is not ProviderOriginDiscourseFocusV1):
            raise TypeError("provider focus dialogue focus 类型错误")

    @property
    def conversation_key(self) -> tuple[int, ...]:
        """暴露 V4 conversation key，避免 outer state 复制第二份 owner。"""
        return self.mixed_state.conversation_key

    @property
    def next_operation_ordinal(self) -> int:
        """暴露 V4 operation ordinal，唯一增量仍由 inner state 定义。"""
        return self.mixed_state.next_operation_ordinal

    @property
    def context(self):
        """提供只读 V2 context adapter；焦点不是 context 的一部分。"""
        return self.mixed_state.context

    def canonical_record(self) -> tuple[int, ...]:
        """导出含 optional tag 的完整外层状态记录。"""
        result = [RAW_PROVIDER_FOCUS_DIALOGUE_STATE_RECORD_V1]
        _pack(result, self.mixed_state.canonical_record())
        if self.focus is None:
            result.append(RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_NONE)
        else:
            result.append(RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_PRESENT)
            _pack(result, self.focus.canonical_record())
        return tuple(result)


def _new_success_from_v4(turn: ConversationRawMixedDialogueTurnV4) -> bool:
    """识别任何新的真实回答，防止旧 focus 穿过新的语义边界。"""
    return bool(
        (turn.answer is not None and turn.answer.accepted)
        or (turn.provider_answer is not None and turn.provider_answer.accepted))


def _focus_after_v4(
        before: ConversationRawProviderFocusDialogueStateV1,
        turn: ConversationRawMixedDialogueTurnV4,
        ) -> ProviderOriginDiscourseFocusV1 | None:
    """按 V4 carrier 分类决定创建、清除或保留焦点。"""
    if turn.provider_followup_answer is not None and turn.provider_followup_answer.accepted:
        try:
            return focus_from_provider_origin_followup_v1(
                turn.after.context,
                turn.provider_followup_answer,
            )
        except (ProviderOriginFocusChainError, TypeError, ValueError) as error:
            raise ConversationRawProviderFocusDialogueSessionError(
                "V4 follow-up 无法形成可审计 discourse focus") from error
    if _new_success_from_v4(turn) or turn.after.context != before.context:
        return None
    return before.focus


# object-model: value; representation=struct; interop=DLG-RAW-12
@dataclass(frozen=True, slots=True)
class ConversationRawProviderFocusDialogueTurnV1:
    """一次 V5 transition，严格区分 V4 fallback 与 active-focus reduction。"""

    before: ConversationRawProviderFocusDialogueStateV1
    intake: ConversationRawIntake
    focus_context_read: MixedContextReadV2 | None
    v4_turn: ConversationRawMixedDialogueTurnV4 | None
    provider_focus_followup_answer: ProviderOriginFollowupResultV1 | None
    after: ConversationRawProviderFocusDialogueStateV1

    def __post_init__(self) -> None:
        """验证 operation、inner state、carrier 与 focus lifecycle 的完整闭环。"""
        if type(self.before) is not ConversationRawProviderFocusDialogueStateV1:
            raise TypeError("provider focus dialogue turn before 类型错误")
        if type(self.intake) is not ConversationRawIntake:
            raise TypeError("provider focus dialogue turn intake 类型错误")
        if (self.focus_context_read is not None
                and type(self.focus_context_read) is not MixedContextReadV2):
            raise TypeError("provider focus dialogue turn read 类型错误")
        if (self.v4_turn is not None
                and type(self.v4_turn) is not ConversationRawMixedDialogueTurnV4):
            raise TypeError("provider focus dialogue turn V4 carrier 类型错误")
        if (self.provider_focus_followup_answer is not None
                and type(self.provider_focus_followup_answer)
                is not ProviderOriginFollowupResultV1):
            raise TypeError("provider focus dialogue turn focus carrier 类型错误")
        if type(self.after) is not ConversationRawProviderFocusDialogueStateV1:
            raise TypeError("provider focus dialogue turn after 类型错误")
        if ((self.v4_turn is None)
                == (self.provider_focus_followup_answer is None)):
            raise ConversationRawProviderFocusDialogueSessionError(
                "provider focus dialogue 每轮必须且只能有一种 carrier")
        if (self.after.conversation_key != self.before.conversation_key
                or self.after.next_operation_ordinal
                != self.before.next_operation_ordinal + 1):
            raise ConversationRawProviderFocusDialogueSessionError(
                "provider focus dialogue operation 序或 conversation key 漂移")
        if self.v4_turn is not None:
            base = self.v4_turn
            if (self.focus_context_read is not None
                    or base.before != self.before.mixed_state
                    or base.after != self.after.mixed_state
                    or base.intake != self.intake):
                raise ConversationRawProviderFocusDialogueSessionError(
                    "provider focus dialogue V4 projection 漂移")
            expected_focus = _focus_after_v4(self.before, base)
            if self.after.focus != expected_focus:
                raise ConversationRawProviderFocusDialogueSessionError(
                    "provider focus dialogue V4 后 focus lifecycle 漂移")
            return
        result = self.provider_focus_followup_answer
        if result is None:
            raise ConversationRawProviderFocusDialogueSessionError(
                "provider focus dialogue 缺 focus follow-up result")
        if (self.before.focus is None
                or result.intake != self.intake
                or not result.handled
                or result.context_read != self.focus_context_read
                or result.context_write_origin != MIXED_CONTEXT_WRITE_ORIGIN_NONE
                or self.after.context != self.before.context):
            raise ConversationRawProviderFocusDialogueSessionError(
                "provider focus dialogue focus carrier 或零写边界漂移")
        if result.accepted:
            try:
                expected_focus = focus_from_provider_origin_followup_v1(
                    self.after.context,
                    result,
                )
            except (ProviderOriginFocusChainError, TypeError, ValueError) as error:
                raise ConversationRawProviderFocusDialogueSessionError(
                    "provider focus dialogue accepted result 不能形成 focus") from error
            if self.after.focus != expected_focus:
                raise ConversationRawProviderFocusDialogueSessionError(
                    "provider focus dialogue accepted focus transition 漂移")
        elif self.after.focus != self.before.focus:
            raise ConversationRawProviderFocusDialogueSessionError(
                "provider focus dialogue rejected result 不得改变 focus")

    @property
    def answer(self) -> ConversationRawAnswerResult | None:
        """返回 V4 Frame carrier；focus reduction 不伪装为 Frame。"""
        return None if self.v4_turn is None else self.v4_turn.answer

    @property
    def provider_answer(self) -> PublicProofSentenceProviderResultV1 | None:
        """返回 V4 provider 首句 carrier；focus reduction 不重跑 provider。"""
        return None if self.v4_turn is None else self.v4_turn.provider_answer

    @property
    def provider_followup_answer(self) -> ProviderOriginFollowupResultV1 | None:
        """返回 V4 或 V5 的 source-bound follow-up carrier。"""
        if self.provider_focus_followup_answer is not None:
            return self.provider_focus_followup_answer
        return None if self.v4_turn is None else self.v4_turn.provider_followup_answer

    def canonical_record(self) -> tuple[int, ...]:
        """导出 before/after focus 与唯一 carrier 的完整 transition record。"""
        result = [RAW_PROVIDER_FOCUS_DIALOGUE_TURN_RECORD_V1]
        for value in (
                self.before.canonical_record(),
                self.intake.canonical_record(),
                (() if self.focus_context_read is None
                 else self.focus_context_read.canonical_record()),
                (() if self.v4_turn is None
                 else self.v4_turn.canonical_record()),
                (() if self.provider_focus_followup_answer is None
                 else self.provider_focus_followup_answer.canonical_record()),
                self.after.canonical_record()):
            _pack(result, value)
        return tuple(result)


def start_public_provider_focus_dialogue(
        conversation_key: tuple[int, ...],
        ) -> ConversationRawProviderFocusDialogueStateV1:
    """建立没有焦点的 V5 outer session，不升级已有 V4 snapshot。"""
    return ConversationRawProviderFocusDialogueStateV1(
        start_public_mixed_frame_dialogue(conversation_key),
    )


def run_public_provider_focus_dialogue_turn_v1(
        state: ConversationRawProviderFocusDialogueStateV1,
        raw_input_bytes: tuple[int, ...],
        runtime: PublicDialogueRuntimeV1,
        *,
        preparation_cache: PublicCoursePreparationCache | None = None,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> ConversationRawProviderFocusDialogueTurnV1:
    """运行默认 V5 dialogue：active focus form 先于且独占 V4 fallback。"""
    if type(state) is not ConversationRawProviderFocusDialogueStateV1:
        raise TypeError("provider focus dialogue state 类型错误")
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("provider focus dialogue runtime 类型错误")
    raw = _u8_vector(raw_input_bytes, label="provider focus dialogue raw input")
    intake = intake_raw_conversation_vector(raw)
    catalog = runtime.provider_origin_followup_catalog
    if state.focus is not None and catalog is not None:
        result = run_provider_origin_focus_followup_v1(
            intake,
            state.context,
            catalog,
            state.focus,
        )
        if result is not None:
            after_mixed = ConversationRawMixedDialogueStateV2(
                state.conversation_key,
                state.next_operation_ordinal + 1,
                state.context,
            )
            if result.accepted:
                try:
                    after_focus = focus_from_provider_origin_followup_v1(
                        state.context,
                        result,
                    )
                except (ProviderOriginFocusChainError, TypeError, ValueError) as error:
                    raise ConversationRawProviderFocusDialogueSessionError(
                        "provider focus dialogue accepted reduction 无法投影") from error
            else:
                after_focus = state.focus
            after = ConversationRawProviderFocusDialogueStateV1(
                after_mixed,
                after_focus,
            )
            return ConversationRawProviderFocusDialogueTurnV1(
                state,
                intake,
                result.context_read,
                None,
                result,
                after,
            )
    base = run_public_mixed_frame_dialogue_turn_v4(
        state.mixed_state,
        raw,
        runtime,
        preparation_cache=preparation_cache,
        preflight_cache=preflight_cache,
    )
    after = ConversationRawProviderFocusDialogueStateV1(
        base.after,
        _focus_after_v4(state, base),
    )
    return ConversationRawProviderFocusDialogueTurnV1(
        state,
        intake,
        None,
        base,
        None,
        after,
    )


__all__ = [
    "RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_NONE",
    "RAW_PROVIDER_FOCUS_DIALOGUE_FOCUS_PRESENT",
    "RAW_PROVIDER_FOCUS_DIALOGUE_STATE_RECORD_V1",
    "RAW_PROVIDER_FOCUS_DIALOGUE_TURN_RECORD_V1",
    "ConversationRawProviderFocusDialogueSessionError",
    "ConversationRawProviderFocusDialogueStateV1",
    "ConversationRawProviderFocusDialogueTurnV1",
    "run_public_provider_focus_dialogue_turn_v1",
    "start_public_provider_focus_dialogue",
]
