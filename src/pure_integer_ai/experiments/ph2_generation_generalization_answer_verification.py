"""E-01 单命题 ANSWER 的 readback 与 legal-composition 独立 verifier。"""
from __future__ import annotations

from dataclasses import dataclass, field

from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceParseRequest,
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
    APPLICABILITY_APPLICABLE,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    MultiVerifierOrchestrator,
    VerificationEvaluation,
    VerificationReport,
    VerifierRegistration,
)


ANSWER_REQUIREMENTS = (
    "INDEPENDENT_UNDERSTANDING_READBACK",
    "LEGAL_OBJECT_COMPOSITION",
)


# object-model: exception
class GenerationGeneralizationAnswerVerificationError(ValueError):
    """ANSWER requirement input、choice/Use 或 verifier route 发生漂移。"""


def _instruction(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验 verifier reason 使用 MinimalInstruction。"""
    if (not isinstance(value, ObjectIdentity)
            or value.object_kind != OBJECT_MINIMAL_INSTRUCTION):
        raise GenerationGeneralizationAnswerVerificationError(
            f"{where} 必须是 MinimalInstruction")
    return value


def _pack(key: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(key), *key


# object-model: protocol; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationAnswerVerificationProtocol:
    """注入 readback/legal 两条互异 route 和各自分型 reason。"""

    readback_dimension: ProtocolKey
    legal_dimension: ProtocolKey
    readback_verifier: ProtocolKey
    legal_verifier: ProtocolKey
    readback_match: ObjectIdentity
    readback_mismatch: ObjectIdentity
    legal_match: ObjectIdentity
    legal_mismatch: ObjectIdentity

    def __post_init__(self) -> None:
        routes = (
            self.readback_dimension,
            self.legal_dimension,
            self.readback_verifier,
            self.legal_verifier,
        )
        if any(not isinstance(item, ProtocolKey) for item in routes):
            raise TypeError("ANSWER verification route 类型错误")
        if len(set(routes)) != len(routes):
            raise GenerationGeneralizationAnswerVerificationError(
                "ANSWER verification routes 必须互异")
        reasons = (
            self.readback_match,
            self.readback_mismatch,
            self.legal_match,
            self.legal_mismatch,
        )
        for reason in reasons:
            _instruction(reason, where="ANSWER verification reason")
        if len(set(reasons)) != len(reasons):
            raise GenerationGeneralizationAnswerVerificationError(
                "ANSWER verification reasons 必须互异")

    def route(self, requirement: str) -> tuple[ProtocolKey, ProtocolKey]:
        """返回一个 ANSWER requirement 的唯一 dimension/verifier。"""
        if requirement == "INDEPENDENT_UNDERSTANDING_READBACK":
            return self.readback_dimension, self.readback_verifier
        if requirement == "LEGAL_OBJECT_COMPOSITION":
            return self.legal_dimension, self.legal_verifier
        raise GenerationGeneralizationAnswerVerificationError(
            "ANSWER requirement 未注册")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationAnswerVerificationInput:
    """一项 ANSWER case 的 actual choice/Use、execution 和 G-04 readback。"""

    requirement: str
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
        if self.requirement not in ANSWER_REQUIREMENTS:
            raise GenerationGeneralizationAnswerVerificationError(
                "ANSWER verification requirement 未注册")
        if (not isinstance(self.episode, GroundedAnswerEpisode)
                or self.episode.question.answer_plan.response_act != "ANSWER"):
            raise GenerationGeneralizationAnswerVerificationError(
                "ANSWER verification episode 非法")
        if (not isinstance(self.planning, GroundedResponseActPlanningBuild)
                or self.planning.episode != self.episode):
            raise GenerationGeneralizationAnswerVerificationError(
                "ANSWER verification episode/planning 漂移")
        if not isinstance(self.choice, GenerationChoiceHypothesis):
            raise TypeError("ANSWER verification choice 类型错误")
        if not isinstance(self.use, GenerationChoiceUseRef):
            raise TypeError("ANSWER verification use 类型错误")
        if self.use not in self.choice.exact_uses:
            raise GenerationGeneralizationAnswerVerificationError(
                "ANSWER verification Use 未回填 exact choice")
        if not isinstance(self.execution, TypedGenerationExecution):
            raise TypeError("ANSWER verification execution 类型错误")
        if (not self.execution.complete
                or self.execution.plan.request != self.planning.planning):
            raise GenerationGeneralizationAnswerVerificationError(
                "ANSWER verification execution 未绑定 planning")
        if (not isinstance(self.parse_request, GenerationSurfaceParseRequest)
                or self.parse_request
                != GenerationSurfaceParseRequest.from_execution(
                    self.execution)):
            raise GenerationGeneralizationAnswerVerificationError(
                "ANSWER verification parse request 漂移")
        if (not isinstance(self.postcheck, GenerationPostcheckRun)
                or self.postcheck.request.execution != self.execution):
            raise GenerationGeneralizationAnswerVerificationError(
                "ANSWER verification postcheck 漂移")
        goal = self.execution.plan.request.goal
        if (goal.source != self.planning.aggregate_source
                or goal.scope != self.planning.response_scope
                or goal.source not in self.choice.forming_sources
                or self.use.scope != goal.scope):
            raise GenerationGeneralizationAnswerVerificationError(
                "ANSWER verification choice/source/scope 漂移")
        expected_kind = (
            "LEXICAL_REALIZATION_CHOICE"
            if self.requirement == "INDEPENDENT_UNDERSTANDING_READBACK"
            else "PROPOSITION_STRUCTURE_CHOICE")
        if self.choice.choice_kind != expected_kind:
            raise GenerationGeneralizationAnswerVerificationError(
                "ANSWER requirement 未绑定对应 choice kind")
        object.__setattr__(self, "_stable_key_cache", self._build_stable_key())

    def stable_key(self) -> tuple[int, ...]:
        """返回 requirement-specific actual input identity。"""
        if not self._stable_key_cache:
            raise RuntimeError("ANSWER verification stable key 尚未构造")
        return self._stable_key_cache

    def _build_stable_key(self) -> tuple[int, ...]:
        """在构造完成时保存完整 choice/Use/output/readback 引用。"""
        values = [ANSWER_REQUIREMENTS.index(self.requirement) + 1]
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
class GenerationGeneralizationAnswerVerifier:
    """分别裁决 actual parser readback 与结构对象合法组合。"""

    protocol: GenerationGeneralizationAnswerVerificationProtocol

    def __post_init__(self) -> None:
        if not isinstance(
                self.protocol,
                GenerationGeneralizationAnswerVerificationProtocol):
            raise TypeError("ANSWER verifier protocol 类型错误")

    @staticmethod
    def _g04_result(request, dimension):
        """从同次 G-04 report 返回指定 dimension 的唯一结果。"""
        matches = tuple(
            item for item in request.postcheck.report.results
            if item.dimension == dimension)
        return matches[0] if len(matches) == 1 else None

    def _readback_matches(
            self,
            request: GenerationGeneralizationAnswerVerificationInput,
            ) -> bool:
        """核验 actual units 恢复唯一 planned Proposition 和 Representation。"""
        result = self._g04_result(
            request, request.postcheck.protocol.proposition_dimension)
        observation = request.postcheck.parsed.observation
        candidate = request.planning.candidate_bindings[0].candidate
        structure = request.execution.surface.preview.request.structure
        emitted = tuple(
            key for sentence in structure.syntax.sentences
            for key in sentence.proposition_keys)
        return (
            result is not None
            and result.applicability == APPLICABILITY_APPLICABLE
            and result.verdict == VERDICT_SUPPORT
            and request.postcheck.parsed.succeeded
            and observation is not None
            and observation.parse_request_key == request.parse_request.stable_key()
            and observation.representations == request.execution.representations
            and observation.source == request.planning.aggregate_source
            and observation.scope == request.planning.response_scope
            and len(observation.propositions) == 1
            and observation.propositions[0].candidate_key
            == candidate.stable_key()
            and observation.propositions[0].proposition
            == candidate.proposition
            and emitted == (candidate.stable_key(),)
        )

    def _legal_matches(
            self,
            request: GenerationGeneralizationAnswerVerificationInput,
            ) -> bool:
        """核验 selected structure 被 syntax/S-07 和 surface slots 实际消费。"""
        result = self._g04_result(
            request, request.postcheck.protocol.structure_dimension)
        candidate = request.planning.candidate_bindings[0].candidate
        generation = request.execution
        surface_request = generation.surface.preview.request
        structure = surface_request.structure
        sentences = structure.syntax.sentences
        if len(sentences) != 1:
            return False
        sentence = sentences[0]
        syntax_layers = tuple(
            item for item in generation.plan.layers
            if item.layer == generation.plan.protocol.syntax_layer)
        preview_slots = generation.surface.preview.slots
        return (
            result is not None
            and result.applicability == APPLICABILITY_APPLICABLE
            and result.verdict == VERDICT_SUPPORT
            and len(syntax_layers) == 1
            and syntax_layers[0].executed
            and syntax_layers[0].payload == structure.syntax.stable_key()
            and syntax_layers[0].selected_candidate_keys
            == (candidate.stable_key(),)
            and request.choice.target_obligation
            == candidate.proposition.template
            and request.choice.selected_object == sentence.structure
            and sentence.proposition_keys == (candidate.stable_key(),)
            and sentence.instance is not None
            and sentence.instance.candidate_key == candidate.stable_key()
            and sentence.instance.template == sentence.sentence
            and surface_request.execution.complete
            and surface_request.execution.request.syntax == structure.syntax
            and len(preview_slots) == len(sentence.values)
            and {item.value.slot for item in preview_slots}
            == {item.slot for item in sentence.values}
        )

    def verify(
            self,
            request: GenerationGeneralizationAnswerVerificationInput,
            ) -> VerificationEvaluation:
        """按 requirement 返回独立 support/refute，不写 effect。"""
        if not isinstance(
                request, GenerationGeneralizationAnswerVerificationInput):
            raise TypeError("ANSWER verifier input 类型错误")
        if request.requirement == "INDEPENDENT_UNDERSTANDING_READBACK":
            matches = self._readback_matches(request)
            reason = (
                self.protocol.readback_match
                if matches else self.protocol.readback_mismatch)
        else:
            matches = self._legal_matches(request)
            reason = (
                self.protocol.legal_match
                if matches else self.protocol.legal_mismatch)
        return VerificationEvaluation(
            VERDICT_SUPPORT if matches else VERDICT_REFUTE,
            (request.stable_key(),),
            detail=(1 if matches else 2, *reason.stable_key()),
            source=request.planning.aggregate_source,
            scope=request.planning.response_scope,
        )


def run_generation_generalization_answer_verification(
        protocol: GenerationGeneralizationAnswerVerificationProtocol,
        request: GenerationGeneralizationAnswerVerificationInput,
        ) -> VerificationReport:
    """只运行当前 ANSWER requirement 的唯一独立 route。"""
    if not isinstance(
            protocol, GenerationGeneralizationAnswerVerificationProtocol):
        raise TypeError("ANSWER verification protocol 类型错误")
    dimension, verifier = protocol.route(request.requirement)
    evaluator = GenerationGeneralizationAnswerVerifier(protocol)
    return MultiVerifierOrchestrator().run(
        request,
        (VerifierRegistration(
            dimension,
            verifier,
            lambda value: isinstance(
                value, GenerationGeneralizationAnswerVerificationInput),
            evaluator.verify,
        ),),
        read_only=True,
    )


__all__ = [
    "ANSWER_REQUIREMENTS",
    "GenerationGeneralizationAnswerVerificationError",
    "GenerationGeneralizationAnswerVerificationInput",
    "GenerationGeneralizationAnswerVerificationProtocol",
    "GenerationGeneralizationAnswerVerifier",
    "run_generation_generalization_answer_verification",
]
