"""M3 capsule 到命题/关系证据资格与回答义务的接线。

这里复用 T1-G0 至 T1-G3 已有合同：raw observation 只保存物理字节和显式
units，lexical evidence 只绑定 span，proposition relation 只引用 evidence，
qualification 决定 ``ANSWER/UNKNOWN/CLARIFY``。本模块把这条证据闭环接在
B2 双平面 transition 上；它不从表面猜关系，也不生成未经资格的事实句。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.learning_input_capsule import (
    LearningInputCapsule,
    digest_bytes,
)
from pure_integer_ai.cognition.shared.learning_input_capsule import (
    CoreDelta,
    CoreLearningState,
    RuntimeMemoryState,
)
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationPreflightCache,
)
from pure_integer_ai.experiments.conversation_capsule_dual_plane import (
    CapsuleDualPlaneError,
    CapsuleDualPlaneTransition,
    run_capsule_dual_plane_turn,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
)
from pure_integer_ai.experiments.conversation_raw_dialogue_session import (
    ConversationRawDialogueState,
)
from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    RawLexicalEvidence,
    RawLexicalEvidenceBinding,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionConsumerResult,
    RawPropositionQualification,
    consume_raw_proposition_relation,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    RawPropositionRelationBinding,
    RawPropositionRelationEvidence,
    bind_raw_proposition_relation,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservation,
)


CAPSULE_EVIDENCE_BRIDGE_PROTOCOL_V1 = 1


# object-model: exception; interop=portable
class CapsuleEvidenceBridgeError(ValueError):
    """capsule、raw observation 或命题资格链不闭合。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    result.extend((len(value), *value))


def _binding_record(binding: RawPropositionRelationBinding) -> tuple[int, ...]:
    result = [CAPSULE_EVIDENCE_BRIDGE_PROTOCOL_V1]
    _pack(result, binding.canonical_record)
    return tuple(result)


# object-model: value; representation=struct; interop=portable
@dataclass(frozen=True, slots=True)
class CapsuleEvidenceDialogueTransition:
    """双平面 transition 与已资格化 proposition consumer 结果。"""

    capsule: LearningInputCapsule
    observation: RawTextObservation
    lexical_bindings: tuple[RawLexicalEvidenceBinding, ...]
    proposition_binding: RawPropositionRelationBinding
    consumer_result: RawPropositionConsumerResult
    dual_plane: CapsuleDualPlaneTransition

    def __post_init__(self) -> None:
        if not isinstance(self.capsule, LearningInputCapsule):
            raise TypeError("evidence transition capsule 类型错误")
        if not isinstance(self.observation, RawTextObservation):
            raise TypeError("evidence transition observation 类型错误")
        if digest_bytes(bytes(self.observation.raw_bytes)) != (
                self.capsule.raw_content_digest):
            raise CapsuleEvidenceBridgeError("observation raw digest 与 capsule 不一致")
        if (not isinstance(self.lexical_bindings, tuple)
                or any(not isinstance(item, RawLexicalEvidenceBinding)
                       for item in self.lexical_bindings)):
            raise TypeError("evidence transition lexical_bindings 类型错误")
        if not isinstance(self.proposition_binding, RawPropositionRelationBinding):
            raise TypeError("evidence transition proposition_binding 类型错误")
        if not isinstance(self.consumer_result, RawPropositionConsumerResult):
            raise TypeError("evidence transition consumer_result 类型错误")
        if not isinstance(self.dual_plane, CapsuleDualPlaneTransition):
            raise TypeError("evidence transition dual_plane 类型错误")
        if self.dual_plane.capsule != self.capsule:
            raise CapsuleEvidenceBridgeError("evidence 与 dual capsule 漂移")
        expected_ids = tuple(item.evidence_id for item in self.lexical_bindings)
        bound_ids = tuple(item.evidence_id for item in self.proposition_binding.arguments)
        if expected_ids != bound_ids:
            raise CapsuleEvidenceBridgeError("lexical/proposition evidence 顺序漂移")
        if self.consumer_result.proposition_id != (
                self.proposition_binding.proposition_id):
            raise CapsuleEvidenceBridgeError("consumer/proposition identity 漂移")

    @property
    def response_act(self) -> str:
        """资格化结果给回答侧的唯一 response-act 义务。"""
        return self.consumer_result.response_act

    def canonical_record(self) -> tuple[int, ...]:
        result = [CAPSULE_EVIDENCE_BRIDGE_PROTOCOL_V1]
        values = (
            self.capsule.canonical_record,
            self.observation.canonical_record(),
            tuple(item.evidence_record for item in self.lexical_bindings),
            _binding_record(self.proposition_binding),
            self.consumer_result.integer_record,
            self.dual_plane.canonical_record(),
        )
        for value in values:
            if value and isinstance(value[0], tuple):
                result.append(len(value))
                for nested in value:
                    _pack(result, nested)
            else:
                _pack(result, value)
        return tuple(result)


def run_capsule_evidence_dialogue_turn(
        capsule: LearningInputCapsule,
        raw_input_bytes: tuple[int, ...],
        observation: RawTextObservation,
        lexical_evidence: tuple[RawLexicalEvidence, ...],
        proposition: RawPropositionRelationEvidence,
        qualification: RawPropositionQualification,
        core_delta: CoreDelta,
        core_state: CoreLearningState,
        dialogue_state: ConversationRawDialogueState,
        runtime_memory_state: RuntimeMemoryState,
        runtime: PublicDialogueRuntimeV1,
        *,
        preparation_cache: PublicCoursePreparationCache | None = None,
        preflight_cache: AliasRelationPreflightCache | None = None,
        ) -> CapsuleEvidenceDialogueTransition:
    """执行 M3 证据资格闭环并返回回答义务。

    ``qualification`` 是外部证据/标注侧显式输入；本函数只验证绑定和消费，
    因此状态不足时会稳定返回 UNKNOWN/CLARIFY，而不是猜测 surface。
    """
    if not isinstance(capsule, LearningInputCapsule):
        raise TypeError("capsule 类型错误")
    if not isinstance(observation, RawTextObservation):
        raise TypeError("observation 类型错误")
    if not isinstance(raw_input_bytes, tuple):
        raise CapsuleEvidenceBridgeError("raw_input_bytes 必须是整数 tuple")
    raw = tuple(raw_input_bytes)
    if raw != observation.raw_bytes:
        raise CapsuleEvidenceBridgeError("raw input 与 observation 字节不一致")
    if digest_bytes(bytes(raw)) != capsule.raw_content_digest:
        raise CapsuleEvidenceBridgeError("raw input digest 与 capsule 不一致")
    if not isinstance(lexical_evidence, tuple):
        raise TypeError("lexical_evidence 必须是 tuple")
    if not isinstance(proposition, RawPropositionRelationEvidence):
        raise TypeError("proposition 类型错误")
    if not isinstance(qualification, RawPropositionQualification):
        raise TypeError("qualification 类型错误")

    try:
        proposition_binding = bind_raw_proposition_relation(
            observation, lexical_evidence, proposition)
        consumer = consume_raw_proposition_relation(
            proposition_binding, qualification)
        dual = run_capsule_dual_plane_turn(
            capsule,
            core_delta,
            core_state,
            raw,
            dialogue_state,
            runtime_memory_state,
            runtime,
            preparation_cache=preparation_cache,
            preflight_cache=preflight_cache,
        )
    except (ValueError, TypeError, CapsuleDualPlaneError) as error:
        raise CapsuleEvidenceBridgeError("M3 证据或双平面 transition 被拒绝") from error

    bindings = tuple(
        item for argument in proposition_binding.arguments
        for item in (argument,))
    return CapsuleEvidenceDialogueTransition(
        capsule,
        observation,
        bindings,
        proposition_binding,
        consumer,
        dual,
    )


__all__ = [
    "CAPSULE_EVIDENCE_BRIDGE_PROTOCOL_V1",
    "CapsuleEvidenceBridgeError",
    "CapsuleEvidenceDialogueTransition",
    "run_capsule_evidence_dialogue_turn",
]
