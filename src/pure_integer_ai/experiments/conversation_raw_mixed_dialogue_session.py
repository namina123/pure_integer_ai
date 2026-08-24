"""DLG-RAW-11B：Frame 与 proof provider 的混合会话执行桥。

本模块保留 DLG-RAW-04/V1 的状态和行为不变。它只把已经完成的
``QuestionAnswerRun`` 与同次、已验证的 provider proof 分别投影到 V2 tagged
context；任何 provider 结果都不会伪造成旧的 ``QuestionAnswerRun``。

Python 的 data class 仅是当前 reference 实现的结构体便利。可观察状态、输入、
read witness、写入来源和输出均可导出为有序非负整数 record。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationPreflightCache,
)
from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationContextRead,
    ConversationContextState,
    ConversationTurnState,
    start_conversation_context,
)
from pure_integer_ai.experiments.conversation_provider_origin_anchor import (
    ProviderOriginAnchorProjectionV1,
    project_provider_origin_anchor_v1,
    provider_origin_legacy_proof_from_same_dispatch_v1,
    provider_origin_provider_binding_from_public_provider_v1,
)
from pure_integer_ai.experiments.conversation_provider_origin_context import (
    MIXED_CONTEXT_WRITE_ORIGIN_NONE,
    FrameQuestionAnswerTurnV2,
    MixedContextAppendResultV1,
    MixedContextReadV2,
    MixedConversationContextStateV2,
    start_mixed_conversation_context_v2,
)
from pure_integer_ai.experiments.conversation_provider_origin_followup import (
    ProviderOriginFollowupResultV1,
    run_provider_origin_followup_v1,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_LEXICAL_MISS,
    PublicProofSentenceProviderResultV1,
    reject_public_proof_sentence_provider_runtime,
    run_public_proof_sentence_provider_vector_with_typed_proof,
    verify_public_proof_sentence_provider_result,
)
from pure_integer_ai.experiments.conversation_raw_answer_runtime import (
    ConversationRawAnswerResult,
    run_public_frame_answer,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_CONSTRUCTION_MISS,
    DLG_RAW_REJECT_CONTEXT,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_SOURCE_CONFLICT,
    ConversationRawIntake,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_lexical_ingress import (
    ConversationRawLexicalIngressResult,
    ingress_raw_lexical_frame,
)
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    resolve_source_bound_slot_composition,
)


RAW_MIXED_DIALOGUE_STATE_RECORD_V2 = 2
RAW_MIXED_DIALOGUE_TURN_RECORD_V3 = 3
RAW_MIXED_DIALOGUE_TURN_RECORD_V4 = 4


# object-model: exception; interop=DLG-RAW-11B
class ConversationRawMixedDialogueSessionError(ValueError):
    """混合对话 state、legacy compatibility chain 或 turn transition 不闭合。"""


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验会话 identity 只含非负严格整数且不可为空。"""
    if (type(value) is not tuple or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawMixedDialogueSessionError(
            f"{label} 必须是非空非负严格整数 tuple")
    return value


def _byte_vector(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 raw input 使用唯一的有限 u8 表示。"""
    if (type(value) is not tuple
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ConversationRawMixedDialogueSessionError(
            f"{label} 必须是 0..255 严格整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """将一段可变长度 canonical integer record 写为 count 加 payload。"""
    if (type(value) is not tuple
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawMixedDialogueSessionError(
            "mixed dialogue canonical segment 非法")
    result.extend((len(value), *value))


def _legacy_append(
        context: ConversationContextState,
        turn: ConversationTurnState,
        ) -> ConversationContextState:
    """用 V2 Frame payload 重放旧 typed context，不读取 provider turn。"""
    if not isinstance(context, ConversationContextState):
        raise TypeError("mixed dialogue legacy context 类型错误")
    if not isinstance(turn, ConversationTurnState):
        raise TypeError("mixed dialogue legacy turn 类型错误")
    read = turn.context_read
    if not isinstance(read, ConversationContextRead):
        raise ConversationRawMixedDialogueSessionError(
            "mixed dialogue Frame turn 缺 legacy context read")
    expected = context.read(len(read.turns))
    if (turn.turn_ordinal != context.revision or read != expected):
        raise ConversationRawMixedDialogueSessionError(
            "mixed dialogue Frame legacy read 与已恢复前缀漂移")
    return ConversationContextState(
        context.conversation_key,
        context.revision + 1,
        context.digest(),
        (*context.turns, turn),
    )


def _legacy_compatibility_context(
        context: MixedConversationContextStateV2,
        ) -> ConversationContextState:
    """从所有已写 Frame tagged turn 重放旧 context，provider 绝不进入其载荷。"""
    if type(context) is not MixedConversationContextStateV2:
        raise TypeError("mixed dialogue context 类型错误")
    legacy = start_conversation_context(context.conversation_key)
    for tagged_turn in context.turns:
        if type(tagged_turn) is FrameQuestionAnswerTurnV2:
            legacy = _legacy_append(legacy, tagged_turn.frame_turn)
    return legacy


# object-model: value; representation=struct; interop=DLG-RAW-11B
@dataclass(frozen=True, slots=True)
class ConversationRawMixedDialogueStateV2:
    """一个只保存 V2 tagged context 的可审计对话 state。"""

    conversation_key: tuple[int, ...]
    next_operation_ordinal: int
    context: MixedConversationContextStateV2

    def __post_init__(self) -> None:
        """冻结会话 key、operation 序和 V2 context owner 的一致性。"""
        key = _strict_key(self.conversation_key, label="mixed dialogue key")
        if (type(self.next_operation_ordinal) is not int
                or self.next_operation_ordinal < 1):
            raise ConversationRawMixedDialogueSessionError(
                "mixed dialogue next operation ordinal 非法")
        if type(self.context) is not MixedConversationContextStateV2:
            raise TypeError("mixed dialogue context 必须是 V2")
        if self.context.conversation_key != key:
            raise ConversationRawMixedDialogueSessionError(
                "mixed dialogue context conversation key 漂移")
        # 立即重放 compatibility chain；不允许构造一个无法被 V2 codec 恢复的 state。
        _legacy_compatibility_context(self.context)
        object.__setattr__(self, "conversation_key", key)

    def frame_compatibility_context(self) -> ConversationContextState:
        """提供由 V2 Frame 前缀重建的只读 legacy context adapter。"""
        return _legacy_compatibility_context(self.context)

    def canonical_record(self) -> tuple[int, ...]:
        """导出不依赖 Python object identity 的完整 session state。"""
        result = [RAW_MIXED_DIALOGUE_STATE_RECORD_V2]
        for value in (
                self.conversation_key,
                self.context.canonical_record(),
        ):
            _pack(result, value)
        result.append(self.next_operation_ordinal)
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-11B
@dataclass(frozen=True, slots=True)
class ConversationRawMixedDialogueTurnV3:
    """一次 raw input 的 V2 mixed-context state transition。"""

    before: ConversationRawMixedDialogueStateV2
    intake: ConversationRawIntake
    mixed_context_read: MixedContextReadV2 | None
    legacy_context_read: ConversationContextRead | None
    answer: ConversationRawAnswerResult | None
    provider_answer: PublicProofSentenceProviderResultV1 | None
    provider_anchor: ProviderOriginAnchorProjectionV1 | None
    context_append: MixedContextAppendResultV1 | None
    after: ConversationRawMixedDialogueStateV2

    def __post_init__(self) -> None:
        """核验回答 carrier 分账、operation 连续性与 V2 append 前后关系。"""
        if type(self.before) is not ConversationRawMixedDialogueStateV2:
            raise TypeError("mixed dialogue turn before 类型错误")
        if type(self.intake) is not ConversationRawIntake:
            raise TypeError("mixed dialogue turn intake 类型错误")
        if (self.mixed_context_read is not None
                and type(self.mixed_context_read) is not MixedContextReadV2):
            raise TypeError("mixed dialogue V2 context read 类型错误")
        if (self.legacy_context_read is not None
                and type(self.legacy_context_read) is not ConversationContextRead):
            raise TypeError("mixed dialogue legacy context read 类型错误")
        if (self.answer is not None
                and type(self.answer) is not ConversationRawAnswerResult):
            raise TypeError("mixed dialogue frame answer 类型错误")
        if (self.provider_answer is not None
                and type(self.provider_answer)
                is not PublicProofSentenceProviderResultV1):
            raise TypeError("mixed dialogue provider answer 类型错误")
        if (self.answer is None) == (self.provider_answer is None):
            raise ConversationRawMixedDialogueSessionError(
                "mixed dialogue 每轮必须且只能有一种回答 carrier")
        if (self.provider_anchor is not None
                and type(self.provider_anchor) is not ProviderOriginAnchorProjectionV1):
            raise TypeError("mixed dialogue provider anchor 类型错误")
        if (self.context_append is not None
                and type(self.context_append) is not MixedContextAppendResultV1):
            raise TypeError("mixed dialogue context append 类型错误")
        if type(self.after) is not ConversationRawMixedDialogueStateV2:
            raise TypeError("mixed dialogue turn after 类型错误")
        if (self.after.conversation_key != self.before.conversation_key
                or self.after.next_operation_ordinal
                != self.before.next_operation_ordinal + 1):
            raise ConversationRawMixedDialogueSessionError(
                "mixed dialogue operation 序或 conversation key 漂移")
        if self.context_append is None:
            if self.after.context != self.before.context:
                raise ConversationRawMixedDialogueSessionError(
                    "mixed dialogue 无 admission 不得改变 context")
        elif (self.context_append.before != self.before.context
                or self.context_append.after != self.after.context):
            raise ConversationRawMixedDialogueSessionError(
                "mixed dialogue admission 前后 context 漂移")
        if self.provider_answer is not None:
            if self.provider_answer.intake != self.intake:
                raise ConversationRawMixedDialogueSessionError(
                    "mixed dialogue provider intake 漂移")
            if self.provider_anchor is None:
                raise ConversationRawMixedDialogueSessionError(
                    "mixed dialogue provider turn 缺 anchor projection")
        elif self.provider_anchor is not None:
            raise ConversationRawMixedDialogueSessionError(
                "mixed dialogue Frame turn 不得携带 provider anchor")

    @property
    def context_write_origin(self) -> int:
        """返回本轮显式 V2 write origin，未产生 admission 时固定 NONE。"""
        return (MIXED_CONTEXT_WRITE_ORIGIN_NONE if self.context_append is None
                else self.context_append.context_write_origin)

    def canonical_record(self) -> tuple[int, ...]:
        """导出 raw input、read、carrier、anchor、admission 和 state 的完整记录。"""
        result = [RAW_MIXED_DIALOGUE_TURN_RECORD_V3, self.context_write_origin]
        values = (
            self.before.canonical_record(),
            self.intake.canonical_record(),
            (() if self.mixed_context_read is None
             else self.mixed_context_read.canonical_record()),
            (() if self.legacy_context_read is None
             else self.legacy_context_read.stable_key()),
            (() if self.answer is None else self.answer.canonical_record()),
            (() if self.provider_answer is None
             else self.provider_answer.canonical_record()),
            (() if self.provider_anchor is None
             else self.provider_anchor.canonical_record()),
            (() if self.context_append is None
             else self.context_append.canonical_record()),
            self.after.canonical_record(),
        )
        for value in values:
            _pack(result, value)
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-11C
@dataclass(frozen=True, slots=True)
class ConversationRawMixedDialogueTurnV4:
    """V4 transition：在 V3 的两类 carrier 外加入无写入来源内追问。"""

    before: ConversationRawMixedDialogueStateV2
    intake: ConversationRawIntake
    mixed_context_read: MixedContextReadV2 | None
    legacy_context_read: ConversationContextRead | None
    answer: ConversationRawAnswerResult | None
    provider_answer: PublicProofSentenceProviderResultV1 | None
    provider_followup_answer: ProviderOriginFollowupResultV1 | None
    provider_anchor: ProviderOriginAnchorProjectionV1 | None
    context_append: MixedContextAppendResultV1 | None
    after: ConversationRawMixedDialogueStateV2

    def __post_init__(self) -> None:
        """使 Frame/provider/follow-up 三种 carrier 严格分型且保持 V2 context 不变。"""
        if type(self.before) is not ConversationRawMixedDialogueStateV2:
            raise TypeError("mixed dialogue V4 before 类型错误")
        if type(self.intake) is not ConversationRawIntake:
            raise TypeError("mixed dialogue V4 intake 类型错误")
        if (self.mixed_context_read is not None
                and type(self.mixed_context_read) is not MixedContextReadV2):
            raise TypeError("mixed dialogue V4 context read 类型错误")
        if (self.legacy_context_read is not None
                and type(self.legacy_context_read) is not ConversationContextRead):
            raise TypeError("mixed dialogue V4 legacy context read 类型错误")
        if (self.answer is not None
                and type(self.answer) is not ConversationRawAnswerResult):
            raise TypeError("mixed dialogue V4 Frame answer 类型错误")
        if (self.provider_answer is not None
                and type(self.provider_answer)
                is not PublicProofSentenceProviderResultV1):
            raise TypeError("mixed dialogue V4 provider answer 类型错误")
        if (self.provider_followup_answer is not None
                and type(self.provider_followup_answer)
                is not ProviderOriginFollowupResultV1):
            raise TypeError("mixed dialogue V4 follow-up answer 类型错误")
        carrier_count = sum(item is not None for item in (
            self.answer,
            self.provider_answer,
            self.provider_followup_answer,
        ))
        if carrier_count != 1:
            raise ConversationRawMixedDialogueSessionError(
                "mixed dialogue V4 每轮必须且只能有一种回答 carrier")
        if (self.provider_anchor is not None
                and type(self.provider_anchor) is not ProviderOriginAnchorProjectionV1):
            raise TypeError("mixed dialogue V4 provider anchor 类型错误")
        if (self.context_append is not None
                and type(self.context_append) is not MixedContextAppendResultV1):
            raise TypeError("mixed dialogue V4 context append 类型错误")
        if type(self.after) is not ConversationRawMixedDialogueStateV2:
            raise TypeError("mixed dialogue V4 after 类型错误")
        if (self.after.conversation_key != self.before.conversation_key
                or self.after.next_operation_ordinal
                != self.before.next_operation_ordinal + 1):
            raise ConversationRawMixedDialogueSessionError(
                "mixed dialogue V4 operation 序或 conversation key 漂移")
        if self.context_append is None:
            if self.after.context != self.before.context:
                raise ConversationRawMixedDialogueSessionError(
                    "mixed dialogue V4 无 admission 不得改变 context")
        elif (self.context_append.before != self.before.context
                or self.context_append.after != self.after.context):
            raise ConversationRawMixedDialogueSessionError(
                "mixed dialogue V4 admission 前后 context 漂移")
        if self.provider_answer is not None:
            if self.provider_answer.intake != self.intake:
                raise ConversationRawMixedDialogueSessionError(
                    "mixed dialogue V4 provider intake 漂移")
            if self.provider_anchor is None:
                raise ConversationRawMixedDialogueSessionError(
                    "mixed dialogue V4 provider turn 缺 anchor")
        elif self.provider_anchor is not None:
            raise ConversationRawMixedDialogueSessionError(
                "mixed dialogue V4 非 provider 首轮不得携带 anchor")
        if self.provider_followup_answer is not None:
            followup = self.provider_followup_answer
            if (followup.intake != self.intake or not followup.handled
                    or followup.context_write_origin
                    != MIXED_CONTEXT_WRITE_ORIGIN_NONE
                    or self.context_append is not None
                    or followup.context_read != self.mixed_context_read):
                raise ConversationRawMixedDialogueSessionError(
                    "mixed dialogue V4 follow-up carrier 或无写入边界漂移")

    @property
    def context_write_origin(self) -> int:
        """返回本轮显式 write origin；follow-up 固定为 NONE。"""
        return (MIXED_CONTEXT_WRITE_ORIGIN_NONE if self.context_append is None
                else self.context_append.context_write_origin)

    def canonical_record(self) -> tuple[int, ...]:
        """导出三分 carrier、read witness 和 state transition 的完整 record。"""
        result = [RAW_MIXED_DIALOGUE_TURN_RECORD_V4, self.context_write_origin]
        values = (
            self.before.canonical_record(),
            self.intake.canonical_record(),
            (() if self.mixed_context_read is None
             else self.mixed_context_read.canonical_record()),
            (() if self.legacy_context_read is None
             else self.legacy_context_read.stable_key()),
            (() if self.answer is None else self.answer.canonical_record()),
            (() if self.provider_answer is None
             else self.provider_answer.canonical_record()),
            (() if self.provider_followup_answer is None
             else self.provider_followup_answer.canonical_record()),
            (() if self.provider_anchor is None
             else self.provider_anchor.canonical_record()),
            (() if self.context_append is None
             else self.context_append.canonical_record()),
            self.after.canonical_record(),
        )
        for value in values:
            _pack(result, value)
        return tuple(result)


def start_public_mixed_frame_dialogue(
        conversation_key: tuple[int, ...],
        ) -> ConversationRawMixedDialogueStateV2:
    """建立 revision 0 的 V2 mixed session；不隐式升级任何 V1 state。"""
    key = _strict_key(conversation_key, label="mixed dialogue key")
    return ConversationRawMixedDialogueStateV2(
        key,
        1,
        start_mixed_conversation_context_v2(key),
    )


def _target_contextual_ingress(
        ingress: ConversationRawLexicalIngressResult,
        intake: ConversationRawIntake,
        catalog,
        occurrence_key: tuple[int, ...],
        state: ConversationRawMixedDialogueStateV2,
        ) -> tuple[
            ConversationRawLexicalIngressResult,
            MixedContextReadV2 | None,
            ConversationContextRead | None,
        ]:
    """只让 V2 可见尾轮的同 target Frame 解锁旧 target-anchor ingress。"""
    frame = ingress.frame
    if (ingress.result_code != DLG_RAW_REJECT_CONTEXT
            or frame is None
            or frame.context_requirement != PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR):
        return ingress, None, None
    mixed_read = state.context.read(1)
    target = mixed_read.latest_frame_target_turn(frame.context_target_key)
    if target is None:
        # 关键闭锁：provider tail 不能跳过以读取较早 Frame，也不能 fallback provider。
        return ingress, None, None
    legacy = state.frame_compatibility_context()
    legacy_read = legacy.read(1)
    if (not legacy_read.turns or legacy_read.turns[-1] != target.frame_turn):
        raise ConversationRawMixedDialogueSessionError(
            "mixed dialogue V2 tail 与 legacy compatibility context 漂移")
    accepted = ingress_raw_lexical_frame(
        intake,
        catalog,
        occurrence_key,
        context_read=legacy_read,
    )
    if not accepted.accepted:
        raise ConversationRawMixedDialogueSessionError(
            "mixed dialogue verified target Frame 未形成 request")
    return accepted, mixed_read, legacy_read


def _contextual_ingress(
        intake: ConversationRawIntake,
        runtime: PublicDialogueRuntimeV1,
        occurrence_key: tuple[int, ...],
        state: ConversationRawMixedDialogueStateV2,
        ) -> tuple[
            ConversationRawLexicalIngressResult,
            MixedContextReadV2 | None,
            ConversationContextRead | None,
        ]:
    """保留 RAW-06 词汇入口，并在 target-anchor 前施加 V2 尾轮约束。"""
    catalog = runtime.active_catalog
    ingress = ingress_raw_lexical_frame(intake, catalog, occurrence_key)
    if ingress.result_code != DLG_RAW_REJECT_LEXICAL_MISS:
        return _target_contextual_ingress(
            ingress,
            intake,
            catalog,
            occurrence_key,
            state,
        )
    resolution = resolve_source_bound_slot_composition(
        runtime.source_bound_slot_catalog,
        runtime.base_catalog,
        catalog,
        intake.unicode_scalars,
        runtime.source_payload_closure,
    )
    if resolution.accepted:
        dynamic_catalog = resolution.public_frame_catalog
        if dynamic_catalog is None:
            raise ConversationRawMixedDialogueSessionError(
                "mixed dialogue accepted source composition 缺动态 catalog")
        dynamic = ingress_raw_lexical_frame(intake, dynamic_catalog, occurrence_key)
        return _target_contextual_ingress(
            dynamic,
            intake,
            dynamic_catalog,
            occurrence_key,
            state,
        )
    if resolution.result_code == DLG_RAW_REJECT_LEXICAL_MISS:
        return ingress, None, None
    if resolution.result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS:
        return ConversationRawLexicalIngressResult(
            resolution.result_code,
            intake,
            catalog,
            matched_frame_count=resolution.matched_frame_count,
        ), None, None
    if resolution.result_code == DLG_RAW_REJECT_CONSTRUCTION_MISS:
        frame = resolution.frame
        dynamic_catalog = resolution.public_frame_catalog
        if frame is None or dynamic_catalog is None:
            return ConversationRawLexicalIngressResult(
                resolution.result_code,
                intake,
                catalog,
                matched_frame_count=resolution.matched_frame_count,
            ), None, None
        return ConversationRawLexicalIngressResult(
            resolution.result_code,
            intake,
            dynamic_catalog,
            matched_frame_count=resolution.matched_frame_count,
            frame=frame,
            representations=tuple(route.representation for route in frame.routes),
            language_atoms=tuple(route.atom for route in frame.routes),
        ), None, None
    if resolution.result_code == DLG_RAW_REJECT_SOURCE_CONFLICT:
        return ConversationRawLexicalIngressResult(
            resolution.result_code,
            intake,
            catalog,
            matched_frame_count=resolution.matched_frame_count,
        ), None, None
    raise ConversationRawMixedDialogueSessionError(
        "mixed dialogue source composition result code 未注册")


def _provider_anchor_for_same_dispatch(
        runtime: PublicDialogueRuntimeV1,
        candidate: PublicProofSentenceProviderResultV1,
        same_dispatch,
        ) -> ProviderOriginAnchorProjectionV1:
    """只从同次 host proof 投影 provider 来源锚点，绝不按输出文本重查。"""
    provider = runtime.proof_sentence_provider
    if provider is None:
        raise ConversationRawMixedDialogueSessionError(
            "mixed dialogue provider anchor 缺 provider")
    carrier = None
    if (same_dispatch.provider_result.canonical_record()
            == candidate.canonical_record()
            and same_dispatch.demo_proof_projection is not None):
        carrier = provider_origin_legacy_proof_from_same_dispatch_v1(
            same_dispatch.demo_proof_projection.sparse_proof_projection)
    return project_provider_origin_anchor_v1(
        provider_origin_provider_binding_from_public_provider_v1(provider),
        candidate,
        carrier,
    )


def _admit_frame_answer(
        state: ConversationRawMixedDialogueStateV2,
        answer: ConversationRawAnswerResult,
        mixed_read: MixedContextReadV2 | None,
        legacy_read: ConversationContextRead | None,
        ) -> MixedContextAppendResultV1 | None:
    """只让实际完成的 Frame run 形成 V2 Frame tagged turn。"""
    if not answer.accepted:
        return None
    if answer.run is None:
        raise ConversationRawMixedDialogueSessionError(
            "mixed dialogue accepted Frame answer 缺 QuestionAnswerRun")
    legacy = state.frame_compatibility_context()
    if legacy.revision == 0:
        if legacy_read is not None:
            raise ConversationRawMixedDialogueSessionError(
                "mixed dialogue 初始 Frame 不得消费 legacy context")
        after_legacy = legacy.append(answer.run)
        prior = state.context.read(0)
    elif legacy_read is not None:
        if mixed_read is None:
            raise ConversationRawMixedDialogueSessionError(
                "mixed dialogue target Frame 缺 V2 read witness")
        after_legacy = legacy.append_consumed(answer.run, legacy_read)
        prior = mixed_read
    else:
        # 保持 RAW-04 语义：已有 context 后的 NONE Frame 可回答但不自动覆盖锚点。
        return None
    return state.context.admit_frame_qa_run(after_legacy.turns[-1], prior)


def run_public_mixed_frame_dialogue_turn(
        state: ConversationRawMixedDialogueStateV2,
        raw_input_bytes: tuple[int, ...],
        runtime: PublicDialogueRuntimeV1,
        *,
        preparation_cache: PublicCoursePreparationCache | None = None,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> ConversationRawMixedDialogueTurnV3:
    """运行一轮公开对话，并以 V2 tagged context 保存可接纳的真实结果。"""
    if type(state) is not ConversationRawMixedDialogueStateV2:
        raise TypeError("mixed dialogue state 类型错误")
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("mixed dialogue runtime 类型错误")
    raw = _byte_vector(raw_input_bytes, label="mixed dialogue raw input")
    intake = intake_raw_conversation_vector(raw)
    occurrence = (*state.conversation_key, state.next_operation_ordinal)
    ingress, mixed_read, legacy_read = _contextual_ingress(
        intake,
        runtime,
        occurrence,
        state,
    )
    answer: ConversationRawAnswerResult | None = None
    provider_answer: PublicProofSentenceProviderResultV1 | None = None
    provider_anchor: ProviderOriginAnchorProjectionV1 | None = None
    context_append: MixedContextAppendResultV1 | None = None
    context_after = state.context

    if (ingress.result_code == DLG_RAW_REJECT_LEXICAL_MISS
            and runtime.proof_sentence_provider is not None):
        same_dispatch = run_public_proof_sentence_provider_vector_with_typed_proof(
            runtime.proof_sentence_provider,
            raw,
        )
        candidate = same_dispatch.provider_result
        if not verify_public_proof_sentence_provider_result(
                runtime.proof_sentence_provider,
                raw,
                candidate):
            candidate = reject_public_proof_sentence_provider_runtime(
                runtime.proof_sentence_provider,
                intake,
            )
        if candidate.provider_status == (
                PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_LEXICAL_MISS):
            answer = run_public_frame_answer(
                ingress,
                source_payload_closure=runtime.source_payload_closure,
                preparation_cache=preparation_cache,
                preflight_cache=preflight_cache,
            )
        else:
            provider_answer = candidate
            provider_anchor = _provider_anchor_for_same_dispatch(
                runtime,
                candidate,
                same_dispatch,
            )
            prior = state.context.read(1) if provider_anchor.accepted else None
            context_append = state.context.admit_provider_origin_projection(
                provider_anchor,
                prior,
            )
            context_after = context_append.after
    else:
        answer = run_public_frame_answer(
            ingress,
            source_payload_closure=runtime.source_payload_closure,
            preparation_cache=preparation_cache,
            preflight_cache=preflight_cache,
        )
        context_append = _admit_frame_answer(
            state,
            answer,
            mixed_read,
            legacy_read,
        )
        if context_append is not None:
            context_after = context_append.after

    after = ConversationRawMixedDialogueStateV2(
        state.conversation_key,
        state.next_operation_ordinal + 1,
        context_after,
    )
    return ConversationRawMixedDialogueTurnV3(
        state,
        intake,
        mixed_read,
        legacy_read,
        answer,
        provider_answer,
        provider_anchor,
        context_append,
        after,
    )


def run_public_mixed_frame_dialogue_turn_v4(
        state: ConversationRawMixedDialogueStateV2,
        raw_input_bytes: tuple[int, ...],
        runtime: PublicDialogueRuntimeV1,
        *,
        preparation_cache: PublicCoursePreparationCache | None = None,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> ConversationRawMixedDialogueTurnV4:
    """运行默认 V4 session，并只在 lexical miss 前接入来源内 follow-up。

    对未学习 follow-up form、本轮 Frame 和 provider 首句，函数委托 V3 原路径并
    逐字段投影为 V4 carrier，避免修改 DLG-RAW-11B 已封存的状态转换。已命中
    follow-up form 则只读取相邻 V2 tail；它不调用 Frame/provider runtime，也不
    append context。
    """
    if type(state) is not ConversationRawMixedDialogueStateV2:
        raise TypeError("mixed dialogue V4 state 类型错误")
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("mixed dialogue V4 runtime 类型错误")
    raw = _byte_vector(raw_input_bytes, label="mixed dialogue V4 raw input")
    intake = intake_raw_conversation_vector(raw)
    catalog = runtime.provider_origin_followup_catalog
    if catalog is not None:
        occurrence = (*state.conversation_key, state.next_operation_ordinal)
        ingress, _prior_read, _legacy_read = _contextual_ingress(
            intake,
            runtime,
            occurrence,
            state,
        )
        if ingress.result_code == DLG_RAW_REJECT_LEXICAL_MISS:
            forms = catalog.matching_forms(intake.unicode_scalars)
            followup_read = state.context.read(1) if forms else None
            followup = run_provider_origin_followup_v1(
                intake,
                followup_read,
                catalog,
            )
            if followup.handled:
                after = ConversationRawMixedDialogueStateV2(
                    state.conversation_key,
                    state.next_operation_ordinal + 1,
                    state.context,
                )
                return ConversationRawMixedDialogueTurnV4(
                    state,
                    intake,
                    followup_read,
                    None,
                    None,
                    None,
                    followup,
                    None,
                    None,
                    after,
                )
    legacy = run_public_mixed_frame_dialogue_turn(
        state,
        raw,
        runtime,
        preparation_cache=preparation_cache,
        preflight_cache=preflight_cache,
    )
    return ConversationRawMixedDialogueTurnV4(
        legacy.before,
        legacy.intake,
        legacy.mixed_context_read,
        legacy.legacy_context_read,
        legacy.answer,
        legacy.provider_answer,
        None,
        legacy.provider_anchor,
        legacy.context_append,
        legacy.after,
    )


__all__ = [
    "RAW_MIXED_DIALOGUE_STATE_RECORD_V2",
    "RAW_MIXED_DIALOGUE_TURN_RECORD_V3",
    "RAW_MIXED_DIALOGUE_TURN_RECORD_V4",
    "ConversationRawMixedDialogueSessionError",
    "ConversationRawMixedDialogueStateV2",
    "ConversationRawMixedDialogueTurnV3",
    "ConversationRawMixedDialogueTurnV4",
    "run_public_mixed_frame_dialogue_turn",
    "run_public_mixed_frame_dialogue_turn_v4",
    "start_public_mixed_frame_dialogue",
]
