"""DLG-RAW-04：公开 Frame 的显式上下文会话状态机。

本模块把每次 raw byte 输入转换为一次有序 turn transition。它不读取 terminal
history，也不持久化到 SQLite；当前 canonical integer record 用于审计身份和
确定性比较，跨进程恢复仍需独立冻结并实现完整的 context codec。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationContextRead,
    ConversationContextState,
    start_conversation_context,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CONTEXT_NONE,
    PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
)
from pure_integer_ai.experiments.conversation_raw_answer_runtime import (
    ConversationRawAnswerResult,
    run_public_frame_answer,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_LEXICAL_MISS,
    PublicProofSentenceProviderResultV1,
    reject_public_proof_sentence_provider_runtime,
    run_public_proof_sentence_provider_vector,
    verify_public_proof_sentence_provider_result,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
)
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationPreflightCache,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_CONTEXT,
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
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_CONSTRUCTION_MISS,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_SOURCE_CONFLICT,
)


RAW_DIALOGUE_STATE_RECORD_V1 = 1
RAW_DIALOGUE_TURN_RECORD_V1 = 1
RAW_DIALOGUE_TURN_RECORD_V2 = 2


# object-model: exception; interop=DLG-RAW-04
class ConversationRawDialogueSessionError(ValueError):
    """RAW-04 session、context append 或规范整数 transition 不闭合。"""


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验会话身份使用可无歧义编码的非负严格整数。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise ConversationRawDialogueSessionError(
            f"{label} 必须是非空非负严格整数 tuple")
    return value


def _byte_vector(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 core ingress 的原始有限 u8 vector。"""
    if (not isinstance(value, tuple)
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise ConversationRawDialogueSessionError(
            f"{label} 必须是 0..255 严格整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """将可变长整数段写入长度前缀 canonical record。"""
    result.extend((len(value), *value))


# object-model: value; representation=struct; interop=DLG-RAW-04
@dataclass(frozen=True, slots=True)
class ConversationRawDialogueState:
    """一个会话的显式整数 identity、下一 operation 序和 append-only context snapshot。"""

    conversation_key: tuple[int, ...]
    next_operation_ordinal: int
    context: ConversationContextState

    def __post_init__(self) -> None:
        """冻结 session key 与 context owner 的同一性，不允许隐式换会话。"""
        key = _strict_key(self.conversation_key, label="RAW-04 conversation key")
        if (type(self.next_operation_ordinal) is not int
                or self.next_operation_ordinal < 1):
            raise ConversationRawDialogueSessionError(
                "RAW-04 next operation ordinal 非法")
        if not isinstance(self.context, ConversationContextState):
            raise TypeError("RAW-04 context 类型错误")
        if self.context.conversation_key != key:
            raise ConversationRawDialogueSessionError(
                "RAW-04 context conversation key 漂移")
        object.__setattr__(self, "conversation_key", key)

    def canonical_record(self) -> tuple[int, ...]:
        """导出可审计的 session identity，不使用对象地址或 terminal history。"""
        result = [RAW_DIALOGUE_STATE_RECORD_V1]
        for value in (
                self.conversation_key,
                self.context.stable_key(),
        ):
            _pack(result, value)
        result.append(self.next_operation_ordinal)
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-04
@dataclass(frozen=True, slots=True)
class ConversationRawDialogueTurn:
    """一次 raw input 的完整状态转换；frame 与 proof provider 结果显式分账。"""

    before: ConversationRawDialogueState
    intake: ConversationRawIntake
    context_read: ConversationContextRead | None
    answer: ConversationRawAnswerResult | None
    provider_answer: PublicProofSentenceProviderResultV1 | None
    context_written: int
    after: ConversationRawDialogueState

    def __post_init__(self) -> None:
        """验证每次 operation 的序、RAW-02 result 和 context append 全部可重放。"""
        if not isinstance(self.before, ConversationRawDialogueState):
            raise TypeError("RAW-04 turn before 类型错误")
        if not isinstance(self.intake, ConversationRawIntake):
            raise TypeError("RAW-04 turn intake 类型错误")
        if (self.context_read is not None
                and not isinstance(self.context_read, ConversationContextRead)):
            raise TypeError("RAW-04 turn context read 类型错误")
        if (self.answer is not None
                and not isinstance(self.answer, ConversationRawAnswerResult)):
            raise TypeError("RAW-04 turn frame answer 类型错误")
        if (self.provider_answer is not None
                and type(self.provider_answer)
                is not PublicProofSentenceProviderResultV1):
            raise TypeError("RAW-04 turn provider answer 类型错误")
        if (self.answer is None) == (self.provider_answer is None):
            raise ConversationRawDialogueSessionError(
                "RAW-04 每轮必须且只能有一种回答 carrier")
        if type(self.context_written) is not int or self.context_written not in (0, 1):
            raise ConversationRawDialogueSessionError(
                "RAW-04 context written 必须是 0/1")
        if not isinstance(self.after, ConversationRawDialogueState):
            raise TypeError("RAW-04 turn after 类型错误")
        if (self.after.conversation_key != self.before.conversation_key
                or self.after.next_operation_ordinal
                != self.before.next_operation_ordinal + 1
                ):
            raise ConversationRawDialogueSessionError(
                "RAW-04 turn operation 或 RAW-02 零持久化约束漂移")
        if self.provider_answer is not None:
            provider = self.provider_answer
            if provider.intake != self.intake:
                raise ConversationRawDialogueSessionError(
                    "RAW-04 provider intake 与 turn 漂移")
            if (self.context_read is not None or self.context_written
                    or self.after.context != self.before.context):
                raise ConversationRawDialogueSessionError(
                    "DLG-RAW-10 provider 只能 NONE_NO_WRITE")
            return
        answer = self.answer
        if answer is None:
            raise ConversationRawDialogueSessionError("RAW-04 frame answer 缺失")
        if (answer.ingress.intake != self.intake
                or answer.ingress.context_read != self.context_read
                or answer.persistent_state_delta):
            raise ConversationRawDialogueSessionError(
                "RAW-04 frame answer ingress 或持久化状态漂移")
        if not answer.accepted:
            if self.context_written or self.after.context != self.before.context:
                raise ConversationRawDialogueSessionError(
                    "RAW-04 拒绝不得写 context")
            if (self.context_read is not None
                    and self.context_read != self.before.context.read(1)):
                raise ConversationRawDialogueSessionError(
                    "RAW-04 拒绝使用了非当前 context read")
            return
        if answer.run is None:
            raise ConversationRawDialogueSessionError("RAW-04 accepted answer 缺 run")
        if (answer.run.selection_commit is not None
                or answer.run.outcome_commit is not None):
            raise ConversationRawDialogueSessionError(
                "RAW-04 public session 不得接收 selection/outcome commit")
        ingress = answer.ingress
        frame = ingress.frame
        if frame is None:
            raise ConversationRawDialogueSessionError("RAW-04 accepted answer 缺 frame")
        if self.context_written:
            if self.before.context.revision == 0:
                if (self.context_read is not None or ingress.context_read is not None
                        or frame.context_requirement != PUBLIC_FRAME_CONTEXT_NONE):
                    raise ConversationRawDialogueSessionError(
                        "RAW-04 首次 append 必须来自 NONE frame")
                expected = self.before.context.append(answer.run)
            else:
                if (self.context_read is None
                        or ingress.context_read != self.context_read
                        or frame.context_requirement
                        != PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR):
                    raise ConversationRawDialogueSessionError(
                        "RAW-04 后续 append 必须消费 TARGET_ANCHOR read")
                expected = self.before.context.append_consumed(
                    answer.run,
                    self.context_read,
                )
            if self.after.context != expected:
                raise ConversationRawDialogueSessionError(
                    "RAW-04 context append result 漂移")
            return
        if (self.after.context != self.before.context
                or self.context_read is not None
                or frame.context_requirement != PUBLIC_FRAME_CONTEXT_NONE
                or self.before.context.revision == 0):
            raise ConversationRawDialogueSessionError(
                "RAW-04 非写成功只允许已有 context 后的独立 NONE frame")

    def canonical_record(self) -> tuple[int, ...]:
        """导出本次 input、read、answer 与前后 context 的完整整数 transition。"""
        if self.provider_answer is None:
            result = [RAW_DIALOGUE_TURN_RECORD_V1, self.context_written]
            values = (
                self.before.canonical_record(),
                self.intake.canonical_record(),
                (() if self.context_read is None
                 else self.context_read.stable_key()),
                (() if self.answer is None
                 else self.answer.canonical_record()),
                self.after.canonical_record(),
            )
        else:
            result = [RAW_DIALOGUE_TURN_RECORD_V2, self.context_written]
            values = (
                self.before.canonical_record(),
                self.intake.canonical_record(),
                (() if self.context_read is None
                 else self.context_read.stable_key()),
                self.provider_answer.canonical_record(),
                self.after.canonical_record(),
            )
        for value in values:
            _pack(result, value)
        return tuple(result)


def start_public_frame_dialogue(
        conversation_key: tuple[int, ...],
        ) -> ConversationRawDialogueState:
    """建立 revision 0 的 RAW-04 state；当前只提供可审计 canonical identity。"""
    key = _strict_key(conversation_key, label="RAW-04 conversation key")
    return ConversationRawDialogueState(
        key,
        1,
        start_conversation_context(key),
    )


def _contextual_ingress(
        intake: ConversationRawIntake,
        runtime: PublicDialogueRuntimeV1,
        occurrence_key: tuple[int, ...],
        state: ConversationRawDialogueState,
        ) -> tuple[ConversationRawLexicalIngressResult, ConversationContextRead | None]:
    """静态 exact 优先；仅静态 miss 后可解析来源绑定的 NONE 槽组合。"""
    catalog = runtime.active_catalog
    ingress = ingress_raw_lexical_frame(intake, catalog, occurrence_key)
    if ingress.result_code == DLG_RAW_REJECT_LEXICAL_MISS:
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
                raise ConversationRawDialogueSessionError(
                    "RAW-06 accepted composition 缺动态 catalog")
            return ingress_raw_lexical_frame(
                intake,
                dynamic_catalog,
                occurrence_key,
            ), None
        if resolution.result_code == DLG_RAW_REJECT_LEXICAL_MISS:
            return ingress, None
        if resolution.result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS:
            return ConversationRawLexicalIngressResult(
                resolution.result_code,
                intake,
                catalog,
                matched_frame_count=resolution.matched_frame_count,
            ), None
        if resolution.result_code == DLG_RAW_REJECT_CONSTRUCTION_MISS:
            frame = resolution.frame
            dynamic_catalog = resolution.public_frame_catalog
            if frame is None or dynamic_catalog is None:
                # RAW-01 明确接受唯一 pre-frame construction failure：没有
                # route/request/context，运行时只转发 REJECT:9，绝不伪造 frame。
                return ConversationRawLexicalIngressResult(
                    resolution.result_code,
                    intake,
                    catalog,
                    matched_frame_count=resolution.matched_frame_count,
                ), None
            return ConversationRawLexicalIngressResult(
                resolution.result_code,
                intake,
                dynamic_catalog,
                matched_frame_count=resolution.matched_frame_count,
                frame=frame,
                representations=tuple(
                    route.representation for route in frame.routes),
                language_atoms=tuple(route.atom for route in frame.routes),
            ), None
        if resolution.result_code == DLG_RAW_REJECT_SOURCE_CONFLICT:
            return ConversationRawLexicalIngressResult(
                resolution.result_code,
                intake,
                catalog,
                matched_frame_count=resolution.matched_frame_count,
            ), None
        raise ConversationRawDialogueSessionError(
            "RAW-06 composition result code 未注册")
    if (ingress.result_code != DLG_RAW_REJECT_CONTEXT
            or ingress.frame is None
            or ingress.frame.context_requirement != PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR):
        return ingress, None
    context_read = state.context.read(1)
    return (
        ingress_raw_lexical_frame(
            intake,
            catalog,
            occurrence_key,
            context_read=context_read,
        ),
        context_read,
    )


def run_public_frame_dialogue_turn(
        state: ConversationRawDialogueState,
        raw_input_bytes: tuple[int, ...],
        runtime: PublicDialogueRuntimeV1,
        *,
        preparation_cache: PublicCoursePreparationCache | None = None,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> ConversationRawDialogueTurn:
    """执行一轮 RAW-00 至 RAW-04；只有已完成真实回答才可能显式 append context。"""
    if not isinstance(state, ConversationRawDialogueState):
        raise TypeError("RAW-04 state 类型错误")
    if type(runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("RAW-04 public dialogue runtime 类型错误")
    raw = _byte_vector(raw_input_bytes, label="RAW-04 raw input")
    intake = intake_raw_conversation_vector(raw)
    occurrence = (*state.conversation_key, state.next_operation_ordinal)
    ingress, context_read = _contextual_ingress(
        intake,
        runtime,
        occurrence,
        state,
    )
    answer: ConversationRawAnswerResult | None
    provider_answer: PublicProofSentenceProviderResultV1 | None = None
    if (ingress.result_code == DLG_RAW_REJECT_LEXICAL_MISS
            and runtime.proof_sentence_provider is not None):
        candidate = run_public_proof_sentence_provider_vector(
            runtime.proof_sentence_provider,
            raw,
        )
        if not verify_public_proof_sentence_provider_result(
                runtime.proof_sentence_provider,
                raw,
                candidate,
        ):
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
            answer = None
            provider_answer = candidate
    else:
        answer = run_public_frame_answer(
            ingress,
            source_payload_closure=runtime.source_payload_closure,
            preparation_cache=preparation_cache,
            preflight_cache=preflight_cache,
        )
    context = state.context
    context_written = 0
    if answer is not None and answer.accepted:
        if context.revision == 0:
            if context_read is not None:
                raise ConversationRawDialogueSessionError(
                    "RAW-04 无首轮 target anchor 时不得接受 follow-up")
            context = context.append(answer.run)
            context_written = 1
        elif context_read is not None:
            context = context.append_consumed(answer.run, context_read)
            context_written = 1
    after = ConversationRawDialogueState(
        state.conversation_key,
        state.next_operation_ordinal + 1,
        context,
    )
    return ConversationRawDialogueTurn(
        state,
        intake,
        context_read,
        answer,
        provider_answer,
        context_written,
        after,
    )


__all__ = [
    "RAW_DIALOGUE_STATE_RECORD_V1",
    "RAW_DIALOGUE_TURN_RECORD_V1",
    "RAW_DIALOGUE_TURN_RECORD_V2",
    "ConversationRawDialogueSessionError",
    "ConversationRawDialogueState",
    "ConversationRawDialogueTurn",
    "run_public_frame_dialogue_turn",
    "start_public_frame_dialogue",
]
