"""DLG-RAW-11B：真实 Frame run 与 provider-origin 的 mixed V3 会话。

V3 不修改 RAW-04。Frame 分支仍由现有 RAW-01/02 形成完整
``QuestionAnswerRun``，随后经 ``ConversationContextState.append`` 生成 legacy
typed turn，再包进 V2 tagged context。provider 分支只消费同次 proof 投影，绝不
把它伪造成 Frame run。
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
    PROVIDER_ORIGIN_ANCHOR_STATUS_NONE,
    ProviderOriginAnchorProjectionV1,
    project_provider_origin_anchor_v1,
    provider_origin_legacy_proof_from_same_dispatch_v1,
)
from pure_integer_ai.experiments.conversation_provider_origin_context import (
    MIXED_CONTEXT_APPEND_ACCEPTED,
    MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN,
    MIXED_CONTEXT_WRITE_ORIGIN_NONE,
    MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION,
    FrameQuestionAnswerTurnV2,
    MixedContextAppendResultV1,
    MixedContextReadV2,
    MixedConversationContextStateV2,
    start_mixed_conversation_context_v2,
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
from pure_integer_ai.experiments.conversation_public_dialogue_runtime_v3 import (
    PublicDialogueRuntimeV3,
)
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    resolve_source_bound_slot_composition,
)


RAW_DIALOGUE_STATE_RECORD_V3 = 3
RAW_DIALOGUE_TURN_RECORD_V3 = 3


# object-model: exception; interop=DLG-RAW-11B
class ConversationRawDialogueSessionV3Error(ValueError):
    """V3 mixed session、legacy compatibility context 或 admission 不闭合。"""


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证可被 V2 snapshot 编码的非空严格整数会话键。"""
    if (type(value) is not tuple or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawDialogueSessionV3Error(
            f"{label} 必须是非空非负严格整数 tuple")
    return value


def _u8_vector(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证 raw ingress 的规范有限 u8 vector。"""
    if (type(value) is not tuple
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ConversationRawDialogueSessionV3Error(
            f"{label} 必须是 0..255 严格整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """把一个有序非负整数段写入 V3 canonical record。"""
    result.extend((len(value), *value))


def rebuild_legacy_frame_context_from_mixed_v2(
        mixed_context: MixedConversationContextStateV2,
        ) -> ConversationContextState:
    """从 V2 的全部 Frame turn 重放唯一 legacy compatibility context。

    provider turn 被显式忽略而非删除：它们仍留在 V2 snapshot 的顺序链中。每个
    Frame 的 ``ConversationContextRead`` 必须与已恢复的 Frame 前缀完全相等，
    因此 V3 不会把 provider 后的旧 anchor 偷渡给新的 follow-up。
    """
    if type(mixed_context) is not MixedConversationContextStateV2:
        raise TypeError("V3 legacy rebuild 需要 MixedConversationContextStateV2")
    context = start_conversation_context(mixed_context.conversation_key)
    for mixed_turn in mixed_context.turns:
        if type(mixed_turn) is not FrameQuestionAnswerTurnV2:
            continue
        frame_turn = mixed_turn.frame_turn
        read = frame_turn.context_read
        if (type(frame_turn) is not ConversationTurnState
                or type(read) is not ConversationContextRead
                or frame_turn.turn_ordinal != context.revision):
            raise ConversationRawDialogueSessionV3Error(
                "V3 mixed Frame legacy ordinal 或 read 类型漂移")
        expected_read = context.read(len(read.turns))
        if read.stable_key() != expected_read.stable_key():
            raise ConversationRawDialogueSessionV3Error(
                "V3 mixed Frame legacy read 与前缀漂移")
        context = ConversationContextState(
            context.conversation_key,
            context.revision + 1,
            context.digest(),
            (*context.turns, frame_turn),
        )
    return context


# object-model: value; representation=struct; interop=DLG-RAW-11B
@dataclass(frozen=True, slots=True)
class ConversationRawDialogueStateV3:
    """V3 的 append-only mixed state 与可验证 Frame execution compatibility state。"""

    conversation_key: tuple[int, ...]
    next_operation_ordinal: int
    mixed_context: MixedConversationContextStateV2
    legacy_frame_context: ConversationContextState

    def __post_init__(self) -> None:
        """固定同一会话键，并证明 legacy context 可由 V2 Frame 前缀重建。"""
        key = _strict_key(self.conversation_key, label="V3 conversation key")
        if (type(self.next_operation_ordinal) is not int
                or self.next_operation_ordinal < 1):
            raise ConversationRawDialogueSessionV3Error(
                "V3 next operation ordinal 非法")
        if type(self.mixed_context) is not MixedConversationContextStateV2:
            raise TypeError("V3 mixed context 类型错误")
        if type(self.legacy_frame_context) is not ConversationContextState:
            raise TypeError("V3 legacy frame context 类型错误")
        if (self.mixed_context.conversation_key != key
                or self.legacy_frame_context.conversation_key != key):
            raise ConversationRawDialogueSessionV3Error(
                "V3 state conversation key 漂移")
        expected_legacy = rebuild_legacy_frame_context_from_mixed_v2(
            self.mixed_context)
        if (self.legacy_frame_context.stable_key()
                != expected_legacy.stable_key()):
            raise ConversationRawDialogueSessionV3Error(
                "V3 legacy frame context 不是 V2 Frame 前缀的重放结果")
        object.__setattr__(self, "conversation_key", key)

    def canonical_record(self) -> tuple[int, ...]:
        """导出 V3 state 的显式整数本体和 compatibility replay witness。"""
        result = [RAW_DIALOGUE_STATE_RECORD_V3]
        for segment in (
                self.conversation_key,
                self.mixed_context.canonical_record(),
                self.legacy_frame_context.stable_key()):
            _pack(result, segment)
        result.append(self.next_operation_ordinal)
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-11B
@dataclass(frozen=True, slots=True)
class ConversationRawDialogueTurnV3:
    """一次 V3 raw operation 的 frame/provider carrier 与 tagged context transition。"""

    before: ConversationRawDialogueStateV3
    intake: ConversationRawIntake
    legacy_context_read: ConversationContextRead | None
    mixed_context_read: MixedContextReadV2 | None
    frame_answer: ConversationRawAnswerResult | None
    provider_answer: PublicProofSentenceProviderResultV1 | None
    provider_anchor: ProviderOriginAnchorProjectionV1 | None
    mixed_admission: MixedContextAppendResultV1 | None
    after: ConversationRawDialogueStateV3

    def __post_init__(self) -> None:
        """验证 Frame/provider 分账、零写拒绝和所有 state transition。"""
        if type(self.before) is not ConversationRawDialogueStateV3:
            raise TypeError("V3 dialogue turn before 类型错误")
        if type(self.intake) is not ConversationRawIntake:
            raise TypeError("V3 dialogue turn intake 类型错误")
        if (self.legacy_context_read is not None
                and type(self.legacy_context_read) is not ConversationContextRead):
            raise TypeError("V3 dialogue legacy context read 类型错误")
        if (self.mixed_context_read is not None
                and type(self.mixed_context_read) is not MixedContextReadV2):
            raise TypeError("V3 dialogue mixed context read 类型错误")
        if (self.frame_answer is not None
                and type(self.frame_answer) is not ConversationRawAnswerResult):
            raise TypeError("V3 dialogue frame answer 类型错误")
        if (self.provider_answer is not None
                and type(self.provider_answer)
                is not PublicProofSentenceProviderResultV1):
            raise TypeError("V3 dialogue provider answer 类型错误")
        if (self.provider_anchor is not None
                and type(self.provider_anchor)
                is not ProviderOriginAnchorProjectionV1):
            raise TypeError("V3 dialogue provider anchor 类型错误")
        if (self.mixed_admission is not None
                and type(self.mixed_admission) is not MixedContextAppendResultV1):
            raise TypeError("V3 dialogue mixed admission 类型错误")
        if type(self.after) is not ConversationRawDialogueStateV3:
            raise TypeError("V3 dialogue turn after 类型错误")
        if (self.after.conversation_key != self.before.conversation_key
                or self.after.next_operation_ordinal
                != self.before.next_operation_ordinal + 1):
            raise ConversationRawDialogueSessionV3Error(
                "V3 dialogue turn operation ordinal 漂移")
        if (self.frame_answer is None) == (self.provider_answer is None):
            raise ConversationRawDialogueSessionV3Error(
                "V3 每轮必须且只能有一种回答 carrier")
        if self.frame_answer is not None:
            self._validate_frame_turn()
        else:
            self._validate_provider_turn()

    def _validate_frame_turn(self) -> None:
        """验证 frame 只经旧 append 形成 V2 wrapper，绝不手工伪造 typed turn。"""
        answer = self.frame_answer
        if answer is None:
            raise AssertionError("V3 frame validation 缺 answer")
        if (answer.ingress.intake != self.intake
                or answer.ingress.context_read != self.legacy_context_read
                or answer.persistent_state_delta):
            raise ConversationRawDialogueSessionV3Error(
                "V3 Frame ingress/read 或 state delta 漂移")
        if self.provider_anchor is not None:
            raise ConversationRawDialogueSessionV3Error(
                "V3 Frame turn 不得携带 provider anchor")
        if not answer.accepted:
            if (self.mixed_admission is not None
                    or self.after.mixed_context.canonical_record()
                    != self.before.mixed_context.canonical_record()
                    or self.after.legacy_frame_context.stable_key()
                    != self.before.legacy_frame_context.stable_key()):
                raise ConversationRawDialogueSessionV3Error(
                    "V3 rejected Frame 不得写任一 context")
            return
        if answer.run is None:
            raise ConversationRawDialogueSessionV3Error("V3 accepted Frame 缺完整 run")
        expected_legacy = self.before.legacy_frame_context
        did_legacy_append = False
        if expected_legacy.revision == 0:
            if self.legacy_context_read is not None:
                raise ConversationRawDialogueSessionV3Error(
                    "V3 first Frame 不得消费 legacy context read")
            expected_legacy = expected_legacy.append(answer.run)
            did_legacy_append = True
        elif self.legacy_context_read is not None:
            expected_legacy = expected_legacy.append_consumed(
                answer.run, self.legacy_context_read)
            did_legacy_append = True
        if not did_legacy_append:
            if (self.mixed_admission is not None
                    or self.after.legacy_frame_context.stable_key()
                    != self.before.legacy_frame_context.stable_key()
                    or self.after.mixed_context.canonical_record()
                    != self.before.mixed_context.canonical_record()):
                raise ConversationRawDialogueSessionV3Error(
                    "V3 standalone Frame 不得伪造 context append")
            return
        if self.mixed_context_read is None or self.mixed_admission is None:
            raise ConversationRawDialogueSessionV3Error(
                "V3 appended Frame 缺 mixed read 或 admission")
        appended = expected_legacy.turns[-1]
        expected_admission = self.before.mixed_context.admit_frame_qa_run(
            appended, self.mixed_context_read)
        if (not expected_admission.accepted
                or expected_admission.canonical_record()
                != self.mixed_admission.canonical_record()
                or self.mixed_admission.context_write_origin
                != MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN
                or self.after.legacy_frame_context.stable_key()
                != expected_legacy.stable_key()
                or self.after.mixed_context.canonical_record()
                != expected_admission.after.canonical_record()):
            raise ConversationRawDialogueSessionV3Error(
                "V3 Frame legacy append 或 V2 wrapper 漂移")

    def _validate_provider_turn(self) -> None:
        """验证 provider 仍为 V1 NONE_NO_WRITE，V2 投影独立且不能伪装 Frame。"""
        provider = self.provider_answer
        if provider is None:
            raise AssertionError("V3 provider validation 缺 answer")
        if provider.intake != self.intake:
            raise ConversationRawDialogueSessionV3Error(
                "V3 provider intake 与 turn 漂移")
        if self.frame_answer is not None or self.legacy_context_read is not None:
            raise ConversationRawDialogueSessionV3Error(
                "V3 provider turn 不得携带 Frame carrier/read")
        if (self.after.legacy_frame_context.stable_key()
                != self.before.legacy_frame_context.stable_key()):
            raise ConversationRawDialogueSessionV3Error(
                "V3 provider 不得写 legacy Frame context")
        anchor = self.provider_anchor
        if anchor is None:
            raise ConversationRawDialogueSessionV3Error("V3 provider turn 缺 anchor record")
        if not anchor.accepted:
            if (self.mixed_context_read is not None
                    or self.mixed_admission is None
                    or self.mixed_admission.accepted
                    or self.mixed_admission.context_write_origin
                    != MIXED_CONTEXT_WRITE_ORIGIN_NONE
                    or self.after.mixed_context.canonical_record()
                    != self.before.mixed_context.canonical_record()):
                raise ConversationRawDialogueSessionV3Error(
                    "V3 ANCHOR_NONE 必须为零 read、零写 no-op")
            return
        if self.mixed_context_read is None or self.mixed_admission is None:
            raise ConversationRawDialogueSessionV3Error(
                "V3 accepted provider anchor 缺 mixed read/admission")
        expected_admission = self.before.mixed_context.admit_provider_origin_projection(
            anchor, self.mixed_context_read)
        if (not expected_admission.accepted
                or expected_admission.canonical_record()
                != self.mixed_admission.canonical_record()
                or self.mixed_admission.context_write_origin
                != MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION
                or self.after.mixed_context.canonical_record()
                != expected_admission.after.canonical_record()):
            raise ConversationRawDialogueSessionV3Error(
                "V3 provider-origin admission 漂移")

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整 V3 transition，host-only typed proof 不进入该 record。"""
        result = [RAW_DIALOGUE_TURN_RECORD_V3]
        for segment in (
                self.before.canonical_record(),
                self.intake.canonical_record(),
                (() if self.legacy_context_read is None
                 else self.legacy_context_read.stable_key()),
                (() if self.mixed_context_read is None
                 else self.mixed_context_read.canonical_record()),
                (() if self.frame_answer is None
                 else self.frame_answer.canonical_record()),
                (() if self.provider_answer is None
                 else self.provider_answer.canonical_record()),
                (() if self.provider_anchor is None
                 else self.provider_anchor.canonical_record()),
                (() if self.mixed_admission is None
                 else self.mixed_admission.canonical_record()),
                self.after.canonical_record()):
            _pack(result, segment)
        return tuple(result)


def start_public_frame_dialogue_v3(
        conversation_key: tuple[int, ...],
        ) -> ConversationRawDialogueStateV3:
    """创建 V3 empty session；legacy context 从 empty V2 Frame 前缀唯一重建。"""
    key = _strict_key(conversation_key, label="V3 conversation key")
    mixed = start_mixed_conversation_context_v2(key)
    return ConversationRawDialogueStateV3(
        key,
        1,
        mixed,
        start_conversation_context(key),
    )


def _dynamic_or_static_ingress(
        intake: ConversationRawIntake,
        runtime: PublicDialogueRuntimeV3,
        occurrence_key: tuple[int, ...],
        state: ConversationRawDialogueStateV3,
        ) -> tuple[
            ConversationRawLexicalIngressResult,
            ConversationContextRead | None,
            MixedContextReadV2 | None,
        ]:
    """执行 RAW-01/06，并在 materialization 前强制 V2 tail 的 target gate。"""
    legacy_runtime = runtime.legacy_runtime
    catalog = legacy_runtime.active_catalog
    ingress = ingress_raw_lexical_frame(intake, catalog, occurrence_key)
    if ingress.result_code == DLG_RAW_REJECT_LEXICAL_MISS:
        resolution = resolve_source_bound_slot_composition(
            legacy_runtime.source_bound_slot_catalog,
            legacy_runtime.base_catalog,
            catalog,
            intake.unicode_scalars,
            legacy_runtime.source_payload_closure,
        )
        if resolution.accepted:
            dynamic_catalog = resolution.public_frame_catalog
            if dynamic_catalog is None:
                raise ConversationRawDialogueSessionV3Error(
                    "V3 accepted source-bound composition 缺 dynamic catalog")
            return (
                ingress_raw_lexical_frame(
                    intake,
                    dynamic_catalog,
                    occurrence_key,
                ),
                None,
                None,
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
        raise ConversationRawDialogueSessionV3Error(
            "V3 source-bound composition result code 未注册")
    if (ingress.result_code != DLG_RAW_REJECT_CONTEXT
            or ingress.frame is None
            or ingress.frame.context_requirement
            != PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR):
        return ingress, None, None
    # 这里必须先看 V2 tagged tail。provider turn 时 helper 返回 None，不能以
    # legacy context 回溯跳过它，更不能构造 QuestionRequest 或进入 RAW-02。
    target_turn = state.mixed_context.latest_frame_target_turn(
        ingress.frame.context_target_key)
    if target_turn is None:
        return ingress, None, None
    legacy_read = state.legacy_frame_context.read(1)
    if (not legacy_read.turns
            or legacy_read.turns[-1].stable_key()
            != target_turn.frame_turn.stable_key()):
        raise ConversationRawDialogueSessionV3Error(
            "V3 V2 tail 与 legacy Frame compatibility context 漂移")
    mixed_read = state.mixed_context.read(1)
    contextual = ingress_raw_lexical_frame(
        intake,
        catalog,
        occurrence_key,
        context_read=legacy_read,
    )
    return contextual, legacy_read, mixed_read


def _provider_anchor_from_same_dispatch(
        runtime: PublicDialogueRuntimeV3,
        provider_result: PublicProofSentenceProviderResultV1,
        same_dispatch: object | None,
        ) -> ProviderOriginAnchorProjectionV1:
    """从同一次 host proof 投影 anchor；绝不按 output 文本重新查询或匹配。"""
    carrier = None
    try:
        projection = getattr(same_dispatch, "demo_proof_projection", None)
        sparse_projection = getattr(projection, "sparse_proof_projection", None)
        carrier = provider_origin_legacy_proof_from_same_dispatch_v1(
            sparse_projection)
    except (AttributeError, TypeError, ValueError):
        carrier = None
    return project_provider_origin_anchor_v1(
        runtime.provider_origin_binding,
        provider_result,
        carrier,
    )


def run_public_frame_dialogue_turn_v3(
        state: ConversationRawDialogueStateV3,
        raw_input_bytes: tuple[int, ...],
        runtime: PublicDialogueRuntimeV3,
        *,
        preparation_cache: PublicCoursePreparationCache | None = None,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> ConversationRawDialogueTurnV3:
    """执行一轮 V3：Frame 实际运行或 provider 同次 proof 投影，二者严格分账。"""
    if type(state) is not ConversationRawDialogueStateV3:
        raise TypeError("V3 dialogue state 类型错误")
    if type(runtime) is not PublicDialogueRuntimeV3:
        raise TypeError("V3 dialogue runtime 类型错误")
    raw = _u8_vector(raw_input_bytes, label="V3 raw input")
    intake = intake_raw_conversation_vector(raw)
    occurrence = (*state.conversation_key, state.next_operation_ordinal)
    ingress, legacy_read, mixed_read = _dynamic_or_static_ingress(
        intake,
        runtime,
        occurrence,
        state,
    )
    frame_answer: ConversationRawAnswerResult | None = None
    provider_answer: PublicProofSentenceProviderResultV1 | None = None
    provider_anchor: ProviderOriginAnchorProjectionV1 | None = None
    mixed_admission: MixedContextAppendResultV1 | None = None
    mixed_after = state.mixed_context
    legacy_after = state.legacy_frame_context
    if ingress.result_code == DLG_RAW_REJECT_LEXICAL_MISS:
        same_dispatch = None
        try:
            same_dispatch = run_public_proof_sentence_provider_vector_with_typed_proof(
                runtime.provider,
                raw,
            )
            candidate = same_dispatch.provider_result
            if not verify_public_proof_sentence_provider_result(
                    runtime.provider,
                    raw,
                    candidate):
                candidate = reject_public_proof_sentence_provider_runtime(
                    runtime.provider,
                    intake,
                )
                same_dispatch = None
        except (AttributeError, TypeError, ValueError, RuntimeError):
            candidate = reject_public_proof_sentence_provider_runtime(
                runtime.provider,
                intake,
            )
        if candidate.provider_status == PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_LEXICAL_MISS:
            frame_answer = run_public_frame_answer(
                ingress,
                source_payload_closure=runtime.legacy_runtime.source_payload_closure,
                preparation_cache=preparation_cache,
                preflight_cache=preflight_cache,
            )
        else:
            provider_answer = candidate
            provider_anchor = _provider_anchor_from_same_dispatch(
                runtime,
                candidate,
                same_dispatch,
            )
            if provider_anchor.accepted:
                mixed_read = state.mixed_context.read(1)
                mixed_admission = state.mixed_context.admit_provider_origin_projection(
                    provider_anchor,
                    mixed_read,
                )
            else:
                mixed_read = None
                mixed_admission = state.mixed_context.admit_provider_origin_projection(
                    provider_anchor,
                )
            mixed_after = mixed_admission.after
    else:
        if ingress.result_code == DLG_RAW_REJECT_CONTEXT:
            # TARGET_ANCHOR 的 V2 tail gate 已在 lexical ingress 结束。这里
            # 只能复制零 request/零 output 的拒绝 carrier，不能触碰 G-01/G-03/G-04。
            frame_answer = ConversationRawAnswerResult(
                ingress.result_code,
                ingress,
            )
        else:
            frame_answer = run_public_frame_answer(
                ingress,
                source_payload_closure=runtime.legacy_runtime.source_payload_closure,
                preparation_cache=preparation_cache,
                preflight_cache=preflight_cache,
            )
    if frame_answer is not None and frame_answer.accepted:
        if legacy_after.revision == 0:
            if legacy_read is not None:
                raise ConversationRawDialogueSessionV3Error(
                    "V3 first Frame 不得已有 legacy context read")
            legacy_after = legacy_after.append(frame_answer.run)
            mixed_read = state.mixed_context.read(1)
            mixed_admission = state.mixed_context.admit_frame_qa_run(
                legacy_after.turns[-1],
                mixed_read,
            )
            mixed_after = mixed_admission.after
        elif legacy_read is not None:
            legacy_after = legacy_after.append_consumed(
                frame_answer.run,
                legacy_read,
            )
            if mixed_read is None:
                raise AssertionError("V3 target-anchor Frame 必须已有 mixed read")
            mixed_admission = state.mixed_context.admit_frame_qa_run(
                legacy_after.turns[-1],
                mixed_read,
            )
            mixed_after = mixed_admission.after
    after = ConversationRawDialogueStateV3(
        state.conversation_key,
        state.next_operation_ordinal + 1,
        mixed_after,
        legacy_after,
    )
    return ConversationRawDialogueTurnV3(
        state,
        intake,
        legacy_read,
        mixed_read,
        frame_answer,
        provider_answer,
        provider_anchor,
        mixed_admission,
        after,
    )


__all__ = [
    "RAW_DIALOGUE_STATE_RECORD_V3",
    "RAW_DIALOGUE_TURN_RECORD_V3",
    "ConversationRawDialogueSessionV3Error",
    "ConversationRawDialogueStateV3",
    "ConversationRawDialogueTurnV3",
    "rebuild_legacy_frame_context_from_mixed_v2",
    "run_public_frame_dialogue_turn_v3",
    "start_public_frame_dialogue_v3",
]
