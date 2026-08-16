"""E-01 SOURCE_UNCERTAINTY_CITATION 的独立跨来源冲突 verifier。"""
from __future__ import annotations

from dataclasses import dataclass, field

from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateDefinition,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceParseRequest,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckRun,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceHypothesis,
    GenerationChoiceUseRef,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    GroundedAnswerEpisode,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    GroundedResponseActPlanningBuild,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    MultiVerifierOrchestrator,
    VerificationEvaluation,
    VerificationReport,
    VerifierRegistration,
)


# object-model: exception
class GenerationGeneralizationSourceConflictError(ValueError):
    """source-conflict input、route 或 actual run 归属发生漂移。"""


def _instruction(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验 verifier reason 使用 MinimalInstruction。"""
    if (not isinstance(value, ObjectIdentity)
            or value.object_kind != OBJECT_MINIMAL_INSTRUCTION):
        raise GenerationGeneralizationSourceConflictError(
            f"{where} 必须是 MinimalInstruction")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给开放完整键增加长度边界。"""
    return len(value), *value


# object-model: protocol; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationSourceConflictProtocol:
    """注入 source-conflict 独立 dimension/verifier 与分型 reason。"""

    dimension: ProtocolKey
    verifier: ProtocolKey
    matched_reason: ObjectIdentity
    mismatched_reason: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, ProtocolKey):
            raise TypeError("source-conflict dimension 类型错误")
        if not isinstance(self.verifier, ProtocolKey):
            raise TypeError("source-conflict verifier 类型错误")
        _instruction(self.matched_reason, where="source-conflict matched reason")
        _instruction(
            self.mismatched_reason,
            where="source-conflict mismatched reason",
        )
        if self.matched_reason == self.mismatched_reason:
            raise GenerationGeneralizationSourceConflictError(
                "source-conflict reason 不得重复")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationSourceConflictInput:
    """课程冲突 Evidence 与一次 actual response-act 输出的完整只读输入。"""

    episode: GroundedAnswerEpisode
    planning: GroundedResponseActPlanningBuild
    choice: GenerationChoiceHypothesis
    use: GenerationChoiceUseRef
    execution: TypedGenerationExecution
    parse_request: GenerationSurfaceParseRequest
    postcheck: GenerationPostcheckRun
    _stable_key_cache: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        if not isinstance(self.episode, GroundedAnswerEpisode):
            raise TypeError("source-conflict episode 类型错误")
        if not isinstance(self.planning, GroundedResponseActPlanningBuild):
            raise TypeError("source-conflict planning 类型错误")
        if self.planning.episode != self.episode:
            raise GenerationGeneralizationSourceConflictError(
                "source-conflict episode/planning 漂移")
        if self.episode.question.answer_plan.response_act != "CONFLICT":
            raise GenerationGeneralizationSourceConflictError(
                "source-conflict 只接受 CONFLICT episode")
        if not isinstance(self.choice, GenerationChoiceHypothesis):
            raise TypeError("source-conflict choice 类型错误")
        if not isinstance(self.use, GenerationChoiceUseRef):
            raise TypeError("source-conflict use 类型错误")
        if self.use not in self.choice.exact_uses:
            raise GenerationGeneralizationSourceConflictError(
                "source-conflict Use 未回填 exact choice")
        if not isinstance(self.execution, TypedGenerationExecution):
            raise TypeError("source-conflict execution 类型错误")
        if (not self.execution.complete
                or self.execution.plan.request != self.planning.planning):
            raise GenerationGeneralizationSourceConflictError(
                "source-conflict execution 未绑定 typed planning")
        if not isinstance(self.parse_request, GenerationSurfaceParseRequest):
            raise TypeError("source-conflict parse request 类型错误")
        if self.parse_request != GenerationSurfaceParseRequest.from_execution(
                self.execution):
            raise GenerationGeneralizationSourceConflictError(
                "source-conflict parse request 未由 actual execution 派生")
        if not isinstance(self.postcheck, GenerationPostcheckRun):
            raise TypeError("source-conflict postcheck 类型错误")
        if self.postcheck.request.execution != self.execution:
            raise GenerationGeneralizationSourceConflictError(
                "source-conflict postcheck 替换了 actual execution")
        goal = self.execution.plan.request.goal
        if (goal.source != self.planning.aggregate_source
                or goal.scope != self.planning.response_scope
                or self.use.scope != goal.scope
                or goal.source not in self.choice.forming_sources):
            raise GenerationGeneralizationSourceConflictError(
                "source-conflict choice/Use/source/scope 归属漂移")
        object.__setattr__(self, "_stable_key_cache", self._build_stable_key())

    def stable_key(self) -> tuple[int, ...]:
        """返回课程 Evidence、actual output、readback 与 exact Use 的完整键。"""
        if not self._stable_key_cache:
            raise RuntimeError("source-conflict stable key 尚未构造")
        return self._stable_key_cache

    def _build_stable_key(self) -> tuple[int, ...]:
        """在构造完成时形成不依赖对象地址的内容引用。"""
        episode = self.episode.episode_id.encode("utf-8")
        values = [len(episode), *episode]
        for key in (
                self.planning.stable_key(),
                self.choice.stable_key(),
                self.use.stable_key(),
                self.execution.stable_key(),
                self.parse_request.stable_key(),
                self.postcheck.stable_key()):
            values.extend(_pack(key))
        return tuple(values)


# object-model: verifier; state=immutable
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationSourceConflictVerifier:
    """核验冲突来源均被保留，actual CONFLICT 未私选来源或命题。"""

    protocol: GenerationGeneralizationSourceConflictProtocol

    def __post_init__(self) -> None:
        if not isinstance(
                self.protocol, GenerationGeneralizationSourceConflictProtocol):
            raise TypeError("source-conflict verifier protocol 类型错误")

    @staticmethod
    def _matches(
            request: GenerationGeneralizationSourceConflictInput,
            ) -> bool:
        """逐点核验课程冲突、aggregate 归因、actual stance 与零偏置引用。"""
        question = request.episode.question
        proposition_ids = {item.proposition_id for item in question.evidence}
        source_ids = {item.source_id for item in question.evidence}
        support_source_ids = {
            item.source_id for item in question.evidence if item.support}
        refute_source_ids = {
            item.source_id for item in question.evidence if item.refute}
        if len(proposition_ids) != 1 or len(source_ids) < 2:
            return False
        if (not support_source_ids or not refute_source_ids
                or not support_source_ids.isdisjoint(refute_source_ids)):
            return False
        proposition_id = next(iter(proposition_ids))
        candidate = request.planning.candidate_for(proposition_id)
        expected_sources = {
            request.planning.source_for(source_id) for source_id in source_ids}
        if (not candidate.state.support or not candidate.state.refute
                or {item.hypothesis.observation for item in candidate.evidence}
                != expected_sources
                or {item.source for item in candidate.evidence}
                != expected_sources
                or {item.stance for item in candidate.evidence}
                != {EVIDENCE_SUPPORT, EVIDENCE_REFUTE}):
            return False
        for evidence in candidate.evidence:
            try:
                definition = EvidenceCandidateDefinition.from_stable_key(
                    evidence.hypothesis.candidate_key)
            except (TypeError, ValueError):
                return False
            if (definition.candidate != candidate.proposition.template
                    or set(definition.forming_sources) != expected_sources):
                return False
        structure = request.execution.surface.preview.request.structure
        selection = structure.selection
        if selection.selected_candidate_keys != (candidate.stable_key(),):
            return False
        parsed = request.postcheck.parsed
        if not parsed.succeeded or parsed.observation is None:
            return False
        observation = parsed.observation
        return (
            selection.stance == request.choice.target_obligation
            and observation.stance == request.choice.target_obligation
            and observation.source == request.planning.aggregate_source
            and observation.scope == request.planning.response_scope
            and observation.propositions == ()
            and observation.cited_sources == ()
            and request.postcheck.request.source_requirements == ()
        )

    def verify(
            self,
            request: GenerationGeneralizationSourceConflictInput,
            ) -> VerificationEvaluation:
        """返回独立 source-conflict support/refute，不提交 effect。"""
        if not isinstance(
                request, GenerationGeneralizationSourceConflictInput):
            raise TypeError("source-conflict verifier input 类型错误")
        matches = self._matches(request)
        reason = (
            self.protocol.matched_reason
            if matches else self.protocol.mismatched_reason)
        return VerificationEvaluation(
            VERDICT_SUPPORT if matches else VERDICT_REFUTE,
            (request.stable_key(),),
            detail=(1 if matches else 2, *reason.stable_key()),
            source=request.planning.aggregate_source,
            scope=request.planning.response_scope,
        )


def run_generation_generalization_source_conflict_verification(
        protocol: GenerationGeneralizationSourceConflictProtocol,
        request: GenerationGeneralizationSourceConflictInput,
        ) -> VerificationReport:
    """运行唯一只读 source-conflict route，并返回不可提交的 report。"""
    if not isinstance(
            protocol, GenerationGeneralizationSourceConflictProtocol):
        raise TypeError("source-conflict protocol 类型错误")
    verifier = GenerationGeneralizationSourceConflictVerifier(protocol)
    return MultiVerifierOrchestrator().run(
        request,
        (VerifierRegistration(
            protocol.dimension,
            protocol.verifier,
            lambda value: isinstance(
                value, GenerationGeneralizationSourceConflictInput),
            verifier.verify,
        ),),
        read_only=True,
    )


__all__ = [
    "GenerationGeneralizationSourceConflictError",
    "GenerationGeneralizationSourceConflictInput",
    "GenerationGeneralizationSourceConflictProtocol",
    "GenerationGeneralizationSourceConflictVerifier",
    "run_generation_generalization_source_conflict_verification",
]
