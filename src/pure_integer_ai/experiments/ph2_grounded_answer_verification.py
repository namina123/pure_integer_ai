"""单命题 grounded ANSWER 的真实结构与 Evidence 来源 verifier。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationSourceCheckRequest,
    GenerationStructureCheckRequest,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VerificationEvaluation,
)


# object-model: exception
class GroundedAnswerVerificationError(ValueError):
    """grounded ANSWER structure/source verifier 输入或协议非法。"""


def _instruction(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验 verifier 分型原因使用 MinimalInstruction。"""
    if (not isinstance(value, ObjectIdentity)
            or value.object_kind != OBJECT_MINIMAL_INSTRUCTION):
        raise GroundedAnswerVerificationError(
            f"{where} 必须是 MinimalInstruction")
    return value


# object-model: verifier; state=immutable
@dataclass(frozen=True, slots=True)
class GroundedAnswerStructureVerifier:
    """比较单句 syntax/S-07 execution 与 parser structure readback。"""

    matched_reason: ObjectIdentity
    mismatched_reason: ObjectIdentity

    def __post_init__(self) -> None:
        _instruction(self.matched_reason, where="grounded structure match")
        _instruction(
            self.mismatched_reason, where="grounded structure mismatch")

    def verify(
            self,
            request: GenerationStructureCheckRequest,
            ) -> VerificationEvaluation:
        """核验候选只发射一次、无 suppression 且实际结构被执行。"""
        if not isinstance(request, GenerationStructureCheckRequest):
            raise TypeError("grounded structure verifier request 类型错误")
        execution = request.postcheck.execution
        structure = execution.surface.preview.request.structure
        sentences = structure.syntax.sentences
        selected = structure.selection.selected_candidate_keys
        emitted = tuple(
            key for sentence in sentences for key in sentence.proposition_keys)
        matches = (
            len(selected) == 1
            and len(sentences) == 1
            and emitted == selected
            and structure.syntax.suppressed_candidate_keys == ()
            and len(structure.propositions.propositions) == 1
            and structure.propositions.propositions[0].candidate_key
            == selected[0]
            and request.observation.structure_payload
            == sentences[0].structure.stable_key()
            and execution.surface.preview.request.execution.complete
        )
        goal = execution.plan.request.goal
        reason = self.matched_reason if matches else self.mismatched_reason
        return VerificationEvaluation(
            VERDICT_SUPPORT if matches else VERDICT_REFUTE,
            (execution.stable_key(),),
            detail=(1 if matches else 2, *reason.stable_key()),
            source=goal.source,
            scope=goal.scope,
        )


# object-model: verifier; state=immutable
@dataclass(frozen=True, slots=True)
class GroundedAnswerEvidenceSourceVerifier:
    """核验 citation requirement 来自 actual candidate Evidence 来源。"""

    matched_reason: ObjectIdentity
    mismatched_reason: ObjectIdentity

    def __post_init__(self) -> None:
        _instruction(self.matched_reason, where="grounded source match")
        _instruction(
            self.mismatched_reason, where="grounded source mismatch")

    def verify(
            self,
            request: GenerationSourceCheckRequest,
            ) -> VerificationEvaluation:
        """逐候选比较 Evidence source、citation、恢复命题和 scope。"""
        if not isinstance(request, GenerationSourceCheckRequest):
            raise TypeError("grounded source verifier request 类型错误")
        planning = request.postcheck.execution.plan.request
        candidates = {item.stable_key(): item for item in planning.candidates}
        recovered = {item.candidate_key: item for item in request.propositions}
        citations = set(request.observation.cited_sources)
        matches = (
            len(request.requirements) == 1
            and len(recovered) == 1
            and all(
                requirement.candidate_key in candidates
                and requirement.candidate_key in recovered
                and requirement.evidence_sources
                == candidates[requirement.candidate_key].citation_sources
                and set(requirement.evidence_sources).issubset(citations)
                and recovered[requirement.candidate_key].source
                == requirement.source
                and recovered[requirement.candidate_key].scope
                == requirement.scope
                for requirement in request.requirements
            )
        )
        goal = planning.goal
        reason = self.matched_reason if matches else self.mismatched_reason
        return VerificationEvaluation(
            VERDICT_SUPPORT if matches else VERDICT_REFUTE,
            tuple(item.candidate_key for item in request.requirements),
            detail=(1 if matches else 2, *reason.stable_key()),
            source=goal.source,
            scope=goal.scope,
        )


__all__ = [
    "GroundedAnswerEvidenceSourceVerifier",
    "GroundedAnswerStructureVerifier",
    "GroundedAnswerVerificationError",
]
