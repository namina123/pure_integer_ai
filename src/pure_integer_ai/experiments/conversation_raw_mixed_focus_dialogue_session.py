"""DLG-RAW-12：将冻结 V4 session 与 V3 连续焦点账本单向绑定。

这个模块是公开 runtime 的薄外层，不改变 V2 mixed context、V4 reducer 或旧的
cursor 原型。它同时保存 V4 mixed state 和 V3 append-only ledger，并逐轮证明
二者的 Frame/provider 投影完全相同。follow-up focus 只写入 V3 ledger；它不是
V2 context write，也不是 Python 可变游标。

Python dataclass 仅是当前 reference implementation 的结构体便利。所有可观察
state、turn、输入、read witness 与 carrier 都可导出为有序非负整数 record。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationPreflightCache,
)
from pure_integer_ai.experiments.conversation_provider_origin_context import (
    FrameQuestionAnswerTurnV2,
    MIXED_CONTEXT_WRITE_ORIGIN_NONE,
    ProviderOriginContextTurnV1,
)
from pure_integer_ai.experiments.conversation_provider_origin_focus_chain import (
    ProviderOriginDiscourseFocusV1,
    ProviderOriginFocusChainError,
    run_provider_origin_focus_followup_v1,
)
from pure_integer_ai.experiments.conversation_provider_origin_focus_context import (
    FocusContextAppendResultV1,
    FocusContextReadV3,
    FrameQuestionAnswerTurnV3,
    MixedConversationFocusContextStateV3,
    ProviderOriginContextTurnV3,
    ProviderOriginFollowupFocusTurnV1,
    ProviderOriginFocusContextError,
    provider_origin_focus_admission_from_followup_result_v1,
    start_mixed_conversation_focus_context_v3,
)
from pure_integer_ai.experiments.conversation_provider_origin_followup import (
    PROVIDER_ORIGIN_FOLLOWUP_STATUS_REJECTED,
    ProviderOriginFollowupCatalogV1,
    ProviderOriginFollowupResultV1,
    run_provider_origin_followup_v1,
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
    DLG_RAW_REJECT_RUNTIME,
    ConversationRawIntake,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_mixed_dialogue_session import (
    ConversationRawMixedDialogueStateV2,
    ConversationRawMixedDialogueTurnV4,
    run_public_mixed_frame_dialogue_turn_v4,
    start_public_mixed_frame_dialogue,
)


RAW_MIXED_FOCUS_DIALOGUE_STATE_RECORD_V1 = 1
RAW_MIXED_FOCUS_DIALOGUE_TURN_RECORD_V1 = 1


# object-model: exception; interop=DLG-RAW-12
class ConversationRawMixedFocusDialogueSessionError(ValueError):
    """V4/V3 projection、连续焦点 transition 或 outer state 不闭合。"""


def _u8_vector(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 raw input 使用唯一、可跨语言复现的 u8 vector。"""
    if (type(value) is not tuple
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ConversationRawMixedFocusDialogueSessionError(
            f"{label} 必须是 0..255 严格整数 tuple")
    return value


def _positive(value: int, *, label: str) -> int:
    """核验 outer operation ordinal 为显式正整数。"""
    if type(value) is not int or value < 1:
        raise ConversationRawMixedFocusDialogueSessionError(
            f"{label} 必须是正严格整数")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """以 count framing 写入一段有限非负整数 record。"""
    if (type(value) is not tuple
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawMixedFocusDialogueSessionError(
            "mixed focus dialogue canonical segment 非法")
    result.extend((len(value), *value))


def _projection_matches(
        mixed_state: ConversationRawMixedDialogueStateV2,
        focus_context: MixedConversationFocusContextStateV3,
        ) -> bool:
    """比较 V4/V3 的 Frame/provider 投影，焦点 event 不进入 V2 context。"""
    if (mixed_state.conversation_key != focus_context.conversation_key
            or type(mixed_state.context.turns) is not tuple
            or type(focus_context.turns) is not tuple):
        return False
    projected = tuple(
        item for item in focus_context.turns
        if (type(item) is FrameQuestionAnswerTurnV3
            or type(item) is ProviderOriginContextTurnV3))
    inner_turns = mixed_state.context.turns
    if len(projected) != len(inner_turns):
        return False
    for inner, outer in zip(inner_turns, projected):
        if (type(inner) is FrameQuestionAnswerTurnV2
                and type(outer) is FrameQuestionAnswerTurnV3):
            if inner.frame_turn.stable_key() != outer.frame_turn.stable_key():
                return False
            continue
        if (type(inner) is ProviderOriginContextTurnV1
                and type(outer) is ProviderOriginContextTurnV3):
            if (inner.anchor_projection.canonical_record()
                    != outer.anchor_projection.canonical_record()):
                return False
            continue
        return False
    return True


def _projection_or_raise(
        mixed_state: ConversationRawMixedDialogueStateV2,
        focus_context: MixedConversationFocusContextStateV3,
        ) -> None:
    """把跨层 projection 视为 state 可构造性的必要条件。"""
    if not _projection_matches(mixed_state, focus_context):
        raise ConversationRawMixedFocusDialogueSessionError(
            "mixed focus dialogue V4/V3 Frame-provider projection 漂移")


def _initial_or_tail_read(
        context: MixedConversationFocusContextStateV3,
        ) -> FocusContextReadV3:
    """首个写入严格消费 read(0)，之后只消费当前 read(1)。"""
    return context.read(0 if context.revision == 0 else 1)


def _anchor_for_focus_tail(
        turn: ProviderOriginContextTurnV3 | ProviderOriginFollowupFocusTurnV1,
        ):
    """从唯一 V3 可见尾轮取得完整 source-bound parent anchor。"""
    if type(turn) is ProviderOriginContextTurnV3:
        return turn.anchor_projection
    if type(turn) is ProviderOriginFollowupFocusTurnV1:
        return turn.parent_anchor_projection
    raise ConversationRawMixedFocusDialogueSessionError(
        "mixed focus dialogue visible tail 不是 provider 或 focus")


def _legacy_focus_from_v3_event(
        mixed_state: ConversationRawMixedDialogueStateV2,
        event: ProviderOriginFollowupFocusTurnV1,
        ) -> ProviderOriginDiscourseFocusV1:
    """把 V3 event 暂投影为旧 reducer 的候选计算器输入。

    这个对象不保存到 outer state，也不作为连续状态的权威。它只使既有、已验证的
    profile reducer 可以计算下一 candidate；真正的状态 transition 随后必须回写 V3。
    """
    read = mixed_state.context.read(1)
    if len(read.turns) != 1 or type(read.turns[0]) is not ProviderOriginContextTurnV1:
        raise ConversationRawMixedFocusDialogueSessionError(
            "mixed focus dialogue focus event 缺 V2 provider tail")
    provider_turn = read.turns[0]
    anchor = provider_turn.anchor_projection
    if (anchor.canonical_record()
            != event.parent_anchor_projection.canonical_record()):
        raise ConversationRawMixedFocusDialogueSessionError(
            "mixed focus dialogue V2 provider 与 V3 focus parent anchor 漂移")
    try:
        return ProviderOriginDiscourseFocusV1(
            mixed_state.context.revision,
            mixed_state.context.digest(),
            provider_turn.turn_identity_u8,
            anchor.provider_kind,
            anchor.provider_identity_u8,
            anchor.runtime_identity_u8,
            anchor.catalog_record_identity_u8,
            anchor.provider_result_identity_u8,
            anchor.anchor_identity_u8,
            anchor.source_record_key,
            anchor.source_ref_stable_key,
            anchor.source_commitment_u8,
            anchor.w03_observation_key,
            anchor.w04_observation_key,
            anchor.w05_observation_key,
            anchor.generation_construction_key,
            anchor.proposition_key,
            anchor.predicate_key,
            anchor.relation_kind_code,
            event.current_role_binding_key,
            event.current_role_key,
            event.current_filler_key,
            event.current_occurrence_key,
            event.current_start,
            event.current_end,
            event.admission.candidate_identity_u8,
            event.admission.form_identity_u8,
        )
    except (ProviderOriginFocusChainError, TypeError, ValueError) as error:
        raise ConversationRawMixedFocusDialogueSessionError(
            "mixed focus dialogue 无法构造临时 focus candidate input") from error


def _mirror_v4_context_append(
        state: "ConversationRawMixedFocusDialogueStateV1",
        base: ConversationRawMixedDialogueTurnV4,
        ) -> FocusContextAppendResultV1 | None:
    """把 V4 的真实 context admission 单向镜像为 V3 Frame/provider event。"""
    v4_append = base.context_append
    if v4_append is None:
        return None
    focus_context = state.focus_context
    if base.provider_anchor is not None:
        prior = (_initial_or_tail_read(focus_context)
                 if base.provider_anchor.accepted else None)
        append = focus_context.admit_provider_origin_projection(
            base.provider_anchor,
            prior,
        )
    elif (base.answer is not None and base.answer.accepted
          and v4_append.accepted
          and type(v4_append.appended_turn) is FrameQuestionAnswerTurnV2):
        append = focus_context.admit_frame_qa_run(
            v4_append.appended_turn.frame_turn,
            _initial_or_tail_read(focus_context),
        )
    else:
        raise ConversationRawMixedFocusDialogueSessionError(
            "mixed focus dialogue V4 admission 无法分型为 Frame/provider projection")
    if append.accepted != v4_append.accepted:
        raise ConversationRawMixedFocusDialogueSessionError(
            "mixed focus dialogue V4/V3 admission accept boundary 漂移")
    return append


def _runtime_reject_after_focus_admission_failure(
        result: ProviderOriginFollowupResultV1,
        ) -> ProviderOriginFollowupResultV1:
    """把 V3 admission 边界故障收束为零输出 ``REJECT_RUNTIME`` carrier。"""
    if result.form is None:
        raise ConversationRawMixedFocusDialogueSessionError(
            "mixed focus dialogue runtime reject 缺 matched form")
    return ProviderOriginFollowupResultV1(
        PROVIDER_ORIGIN_FOLLOWUP_STATUS_REJECTED,
        DLG_RAW_REJECT_RUNTIME,
        result.intake,
        result.catalog_identity_u8,
        1,
        0,
        result.form,
        result.context_read,
    )


def _validate_current_runtime_course(
        state: ConversationRawMixedFocusDialogueStateV1,
        runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """在 live dispatch 前重演已有 focus event，拒绝脱离当前公开课程的状态。

    验证器与 outer snapshot 共用同一确定性 course replay。延迟导入仅避免 Python
    reference implementation 的模块装载循环；它不改变整数 record 或 transition
    的可观察语义。
    """
    if not any(type(turn) is ProviderOriginFollowupFocusTurnV1
               for turn in state.focus_context.turns):
        return
    from pure_integer_ai.experiments.conversation_raw_mixed_focus_dialogue_snapshot import (
        ConversationRawMixedFocusDialogueSnapshotError,
        validate_public_mixed_focus_dialogue_runtime_v1,
    )

    try:
        validate_public_mixed_focus_dialogue_runtime_v1(state, runtime)
    except (ConversationRawMixedFocusDialogueSnapshotError,
            TypeError, ValueError) as error:
        raise ConversationRawMixedFocusDialogueSessionError(
            "mixed focus dialogue 当前状态未绑定 runtime course") from error


# object-model: value; representation=struct; interop=DLG-RAW-12
@dataclass(frozen=True, slots=True)
class ConversationRawMixedFocusDialogueStateV1:
    """V4 session 与 V3 append-only focus ledger 的冻结双层状态。"""

    mixed_state: ConversationRawMixedDialogueStateV2
    focus_context: MixedConversationFocusContextStateV3
    next_operation_ordinal: int

    def __post_init__(self) -> None:
        """冻结两个状态 owner、独立 operation 序及严格投影关系。"""
        if type(self.mixed_state) is not ConversationRawMixedDialogueStateV2:
            raise TypeError("mixed focus dialogue mixed state 类型错误")
        if type(self.focus_context) is not MixedConversationFocusContextStateV3:
            raise TypeError("mixed focus dialogue focus context 类型错误")
        operation = _positive(
            self.next_operation_ordinal,
            label="mixed focus dialogue next operation ordinal",
        )
        if operation != self.mixed_state.next_operation_ordinal:
            raise ConversationRawMixedFocusDialogueSessionError(
                "mixed focus dialogue outer/V4 operation ordinal 漂移")
        _projection_or_raise(self.mixed_state, self.focus_context)
        object.__setattr__(self, "next_operation_ordinal", operation)

    @property
    def conversation_key(self) -> tuple[int, ...]:
        """暴露唯一 conversation owner，避免复制第二份 key。"""
        return self.mixed_state.conversation_key

    @property
    def context(self):
        """提供冻结 V2 context 的只读 adapter；focus 永不伪装为 V2 write。"""
        return self.mixed_state.context

    def canonical_record(self) -> tuple[int, ...]:
        """导出 outer/V4/V3 三层均可跨语言重建的整数 state record。"""
        result = [
            RAW_MIXED_FOCUS_DIALOGUE_STATE_RECORD_V1,
            self.next_operation_ordinal,
        ]
        _pack(result, self.mixed_state.canonical_record())
        _pack(result, self.focus_context.canonical_record())
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-12
@dataclass(frozen=True, slots=True)
class ConversationRawMixedFocusDialogueTurnV1:
    """一次 outer transition：V4 fallback 或 V3 active-focus reduction 二选一。"""

    before: ConversationRawMixedFocusDialogueStateV1
    intake: ConversationRawIntake
    focus_context_read: FocusContextReadV3 | None
    v4_turn: ConversationRawMixedDialogueTurnV4 | None
    provider_focus_followup_answer: ProviderOriginFollowupResultV1 | None
    focus_context_append: FocusContextAppendResultV1 | None
    after: ConversationRawMixedFocusDialogueStateV1

    def __post_init__(self) -> None:
        """核验 carrier 分型、operation 连续性和 V4/V3 零写/追加边界。"""
        if type(self.before) is not ConversationRawMixedFocusDialogueStateV1:
            raise TypeError("mixed focus dialogue turn before 类型错误")
        if type(self.intake) is not ConversationRawIntake:
            raise TypeError("mixed focus dialogue turn intake 类型错误")
        if (self.focus_context_read is not None
                and type(self.focus_context_read) is not FocusContextReadV3):
            raise TypeError("mixed focus dialogue V3 read 类型错误")
        if (self.v4_turn is not None
                and type(self.v4_turn) is not ConversationRawMixedDialogueTurnV4):
            raise TypeError("mixed focus dialogue V4 carrier 类型错误")
        if (self.provider_focus_followup_answer is not None
                and type(self.provider_focus_followup_answer)
                is not ProviderOriginFollowupResultV1):
            raise TypeError("mixed focus dialogue focus follow-up carrier 类型错误")
        if (self.focus_context_append is not None
                and type(self.focus_context_append) is not FocusContextAppendResultV1):
            raise TypeError("mixed focus dialogue focus append 类型错误")
        if type(self.after) is not ConversationRawMixedFocusDialogueStateV1:
            raise TypeError("mixed focus dialogue turn after 类型错误")
        if ((self.v4_turn is None)
                == (self.provider_focus_followup_answer is None)):
            raise ConversationRawMixedFocusDialogueSessionError(
                "mixed focus dialogue 每轮必须且只能有一种 carrier")
        if (self.after.conversation_key != self.before.conversation_key
                or self.after.next_operation_ordinal
                != self.before.next_operation_ordinal + 1):
            raise ConversationRawMixedFocusDialogueSessionError(
                "mixed focus dialogue outer operation 或 conversation key 漂移")
        if self.v4_turn is not None:
            base = self.v4_turn
            if (self.focus_context_read is not None
                    or base.before != self.before.mixed_state
                    or base.after != self.after.mixed_state
                    or base.intake != self.intake):
                raise ConversationRawMixedFocusDialogueSessionError(
                    "mixed focus dialogue V4 carrier projection 漂移")
            if (base.provider_followup_answer is not None
                    and base.provider_followup_answer.accepted):
                raise ConversationRawMixedFocusDialogueSessionError(
                    "mixed focus dialogue accepted follow-up 不得绕过 V3 ledger")
            if base.context_append is None:
                if (self.focus_context_append is not None
                        or self.after.focus_context.canonical_record()
                        != self.before.focus_context.canonical_record()):
                    raise ConversationRawMixedFocusDialogueSessionError(
                        "mixed focus dialogue V4 zero-write 不得改变 V3 ledger")
                return
            append = self.focus_context_append
            if (append is None or append.before != self.before.focus_context
                    or append.after != self.after.focus_context
                    or append.accepted != base.context_append.accepted):
                raise ConversationRawMixedFocusDialogueSessionError(
                    "mixed focus dialogue V4/V3 admission mirror 漂移")
            return

        result = self.provider_focus_followup_answer
        read = self.focus_context_read
        if result is None or read is None:
            raise ConversationRawMixedFocusDialogueSessionError(
                "mixed focus dialogue active focus 缺 result 或 V3 read")
        expected_read = self.before.focus_context.read(1)
        if (read.canonical_record() != expected_read.canonical_record()
                or len(read.turns) != 1
                or (type(read.turns[0]) is not ProviderOriginContextTurnV3
                    and type(read.turns[0]) is not ProviderOriginFollowupFocusTurnV1)
                or result.intake != self.intake
                or not result.handled
                or result.context_write_origin != MIXED_CONTEXT_WRITE_ORIGIN_NONE
                or self.after.mixed_state.context
                != self.before.mixed_state.context):
            raise ConversationRawMixedFocusDialogueSessionError(
                "mixed focus dialogue active focus carrier 或 V2 zero-write 边界漂移")
        if (self.after.mixed_state.next_operation_ordinal
                != self.before.mixed_state.next_operation_ordinal + 1):
            raise ConversationRawMixedFocusDialogueSessionError(
                "mixed focus dialogue active focus 未推进 V4 operation 序")
        if result.accepted:
            append = self.focus_context_append
            if (append is None or not append.accepted
                    or append.before != self.before.focus_context
                    or append.after != self.after.focus_context):
                raise ConversationRawMixedFocusDialogueSessionError(
                    "mixed focus dialogue accepted follow-up 未追加 V3 focus event")
            return
        if (self.focus_context_append is not None
                or self.after.focus_context.canonical_record()
                != self.before.focus_context.canonical_record()):
            raise ConversationRawMixedFocusDialogueSessionError(
                "mixed focus dialogue rejected follow-up 必须为 V3 no-op")

    @property
    def answer(self) -> ConversationRawAnswerResult | None:
        """暴露 V4 Frame carrier；V3 follow-up 永不伪装成 Frame answer。"""
        return None if self.v4_turn is None else self.v4_turn.answer

    @property
    def provider_answer(self) -> PublicProofSentenceProviderResultV1 | None:
        """暴露 V4 provider 首句 carrier，不重跑 provider。"""
        return None if self.v4_turn is None else self.v4_turn.provider_answer

    @property
    def provider_followup_answer(self) -> ProviderOriginFollowupResultV1 | None:
        """返回 V4 reject 或 V3 active-focus 的唯一 source-bound follow-up carrier。"""
        if self.provider_focus_followup_answer is not None:
            return self.provider_focus_followup_answer
        return None if self.v4_turn is None else self.v4_turn.provider_followup_answer

    @property
    def context_write_origin(self) -> int:
        """返回 outer 可观察写入来源；拒绝与 V4 follow-up 固定 NONE。"""
        if self.focus_context_append is not None:
            return self.focus_context_append.context_write_origin
        return (MIXED_CONTEXT_WRITE_ORIGIN_NONE if self.v4_turn is None
                else self.v4_turn.context_write_origin)

    def canonical_record(self) -> tuple[int, ...]:
        """导出 before/read/carrier/admission/after 的完整整数 transition record。"""
        result = [RAW_MIXED_FOCUS_DIALOGUE_TURN_RECORD_V1, self.context_write_origin]
        for value in (
                self.before.canonical_record(),
                self.intake.canonical_record(),
                (() if self.focus_context_read is None
                 else self.focus_context_read.canonical_record()),
                (() if self.v4_turn is None else self.v4_turn.canonical_record()),
                (() if self.provider_focus_followup_answer is None
                 else self.provider_focus_followup_answer.canonical_record()),
                (() if self.focus_context_append is None
                 else self.focus_context_append.canonical_record()),
                self.after.canonical_record(),
        ):
            _pack(result, value)
        return tuple(result)


def start_public_mixed_focus_dialogue(
        conversation_key: tuple[int, ...],
        ) -> ConversationRawMixedFocusDialogueStateV1:
    """建立空 V4/V3 双层 session；两层首次写入都严格使用 read(0)。"""
    mixed = start_public_mixed_frame_dialogue(conversation_key)
    focus = start_mixed_conversation_focus_context_v3(conversation_key)
    return ConversationRawMixedFocusDialogueStateV1(
        mixed,
        focus,
        mixed.next_operation_ordinal,
    )


def _run_v3_followup_from_tail(
        state: ConversationRawMixedFocusDialogueStateV1,
        intake: ConversationRawIntake,
        catalog: ProviderOriginFollowupCatalogV1,
        read: FocusContextReadV3,
        ) -> tuple[ProviderOriginFollowupResultV1, FocusContextAppendResultV1 | None]:
    """用 V3 的当前唯一尾轮计算并接纳一次连续 source-bound follow-up。"""
    parent = read.turns[0]
    try:
        if type(parent) is ProviderOriginContextTurnV3:
            result = run_provider_origin_followup_v1(
                intake,
                state.mixed_state.context.read(1),
                catalog,
            )
        elif type(parent) is ProviderOriginFollowupFocusTurnV1:
            result = run_provider_origin_focus_followup_v1(
                intake,
                state.mixed_state.context,
                catalog,
                _legacy_focus_from_v3_event(state.mixed_state, parent),
            )
        else:
            raise ConversationRawMixedFocusDialogueSessionError(
                "mixed focus dialogue V3 tail 未登记为 follow-up parent")
    except (ProviderOriginFocusChainError, ProviderOriginFocusContextError,
            TypeError, ValueError) as error:
        raise ConversationRawMixedFocusDialogueSessionError(
            "mixed focus dialogue source-bound follow-up reducer 失败") from error
    if result is None or not result.handled:
        raise ConversationRawMixedFocusDialogueSessionError(
            "mixed focus dialogue 已匹配 form 不得回退 V4 reducer")
    if not result.accepted:
        return result, None
    try:
        admission = provider_origin_focus_admission_from_followup_result_v1(
            result,
            catalog,
            _anchor_for_focus_tail(parent),
        )
        append = state.focus_context.admit_provider_origin_followup_focus(
            admission,
            read,
        )
    except (ProviderOriginFocusContextError, TypeError, ValueError):
        return _runtime_reject_after_focus_admission_failure(result), None
    if not append.accepted:
        return _runtime_reject_after_focus_admission_failure(result), None
    return result, append


def run_public_mixed_focus_dialogue_turn_v1(
        state: ConversationRawMixedFocusDialogueStateV1,
        raw_input_bytes: tuple[int, ...],
        runtime: PublicDialogueRuntimeV1,
        *,
        preparation_cache: PublicCoursePreparationCache | None = None,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> ConversationRawMixedFocusDialogueTurnV1:
    """运行一轮公开对话；已匹配 focus form 只消费 V3 visible tail。

    没有匹配 follow-up form 的输入继续委托冻结 V4 session。V4 产生真实 Frame 或
    provider context write 时，函数立即以相同 payload 镜像 V3；因此 V3 账本始终
    是连续 focus 的唯一状态权威，而 V4 继续只承担其封存的兼容 context。
    """
    if type(state) is not ConversationRawMixedFocusDialogueStateV1:
        raise TypeError("mixed focus dialogue state 类型错误")
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("mixed focus dialogue runtime 类型错误")
    _validate_current_runtime_course(state, runtime)
    raw = _u8_vector(raw_input_bytes, label="mixed focus dialogue raw input")
    intake = intake_raw_conversation_vector(raw)
    catalog = runtime.provider_origin_followup_catalog
    read = state.focus_context.read(1)
    forms = (() if (catalog is None or not intake.accepted)
             else catalog.matching_forms(intake.unicode_scalars))
    if (forms and len(read.turns) == 1
            and (type(read.turns[0]) is ProviderOriginContextTurnV3
                 or type(read.turns[0]) is ProviderOriginFollowupFocusTurnV1)):
        if type(catalog) is not ProviderOriginFollowupCatalogV1:
            raise ConversationRawMixedFocusDialogueSessionError(
                "mixed focus dialogue runtime follow-up catalog 类型错误")
        result, append = _run_v3_followup_from_tail(
            state,
            intake,
            catalog,
            read,
        )
        after_focus = state.focus_context if append is None else append.after
        after_mixed = ConversationRawMixedDialogueStateV2(
            state.mixed_state.conversation_key,
            state.mixed_state.next_operation_ordinal + 1,
            state.mixed_state.context,
        )
        after = ConversationRawMixedFocusDialogueStateV1(
            after_mixed,
            after_focus,
            state.next_operation_ordinal + 1,
        )
        return ConversationRawMixedFocusDialogueTurnV1(
            state,
            intake,
            read,
            None,
            result,
            append,
            after,
        )

    base = run_public_mixed_frame_dialogue_turn_v4(
        state.mixed_state,
        raw,
        runtime,
        preparation_cache=preparation_cache,
        preflight_cache=preflight_cache,
    )
    if (base.provider_followup_answer is not None
            and base.provider_followup_answer.accepted):
        raise ConversationRawMixedFocusDialogueSessionError(
            "mixed focus dialogue V4 accepted follow-up 未经 V3 ledger")
    append = _mirror_v4_context_append(state, base)
    after_focus = state.focus_context if append is None else append.after
    after = ConversationRawMixedFocusDialogueStateV1(
        base.after,
        after_focus,
        state.next_operation_ordinal + 1,
    )
    return ConversationRawMixedFocusDialogueTurnV1(
        state,
        base.intake,
        None,
        base,
        None,
        append,
        after,
    )


__all__ = [
    "RAW_MIXED_FOCUS_DIALOGUE_STATE_RECORD_V1",
    "RAW_MIXED_FOCUS_DIALOGUE_TURN_RECORD_V1",
    "ConversationRawMixedFocusDialogueSessionError",
    "ConversationRawMixedFocusDialogueStateV1",
    "ConversationRawMixedFocusDialogueTurnV1",
    "run_public_mixed_focus_dialogue_turn_v1",
    "start_public_mixed_focus_dialogue",
]
