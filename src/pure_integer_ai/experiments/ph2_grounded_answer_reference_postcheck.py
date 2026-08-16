"""双命题 grounded reference ANSWER 的真实 G-04 结构与来源 verifier。"""
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
class GroundedAnswerReferencePostcheckError(ValueError):
    """reference structure/source verifier 输入或协议非法。"""


def _instruction(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验 verifier 分型原因使用 MinimalInstruction。"""
    if (not isinstance(value, ObjectIdentity)
            or value.object_kind != OBJECT_MINIMAL_INSTRUCTION):
        raise GroundedAnswerReferencePostcheckError(
            f"{where} 必须是 MinimalInstruction")
    return value


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(key), *key


# object-model: verifier; state=immutable
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceStructureVerifier:
    """比较双句 syntax/S-07 execution 与 parser structure readback。"""

    matched_reason: ObjectIdentity
    mismatched_reason: ObjectIdentity

    def __post_init__(self) -> None:
        _instruction(self.matched_reason, where="reference structure match")
        _instruction(
            self.mismatched_reason, where="reference structure mismatch")

    def verify(
            self,
            request: GenerationStructureCheckRequest,
            ) -> VerificationEvaluation:
        """核验两个候选按序各发射一次，且实际结构被完整恢复。"""
        if not isinstance(request, GenerationStructureCheckRequest):
            raise TypeError("reference structure verifier request 类型错误")
        execution = request.postcheck.execution
        structure = execution.surface.preview.request.structure
        planning = execution.plan.request
        expected = tuple(item.stable_key() for item in planning.candidates)
        sentences = structure.syntax.sentences
        emitted = tuple(
            key for sentence in sentences for key in sentence.proposition_keys)
        proposition_keys = tuple(
            item.candidate_key for item in structure.propositions.propositions)
        expected_payload = tuple(
            value
            for sentence in sentences
            for value in _packed(sentence.structure.stable_key())
        )
        matches = (
            len(expected) == 2
            and set(structure.selection.selected_candidate_keys)
            == set(expected)
            and len(sentences) == 2
            and all(len(item.proposition_keys) == 1 for item in sentences)
            and emitted == expected
            and proposition_keys == expected
            and structure.syntax.suppressed_candidate_keys == ()
            and request.observation.structure_payload == expected_payload
            and execution.surface.preview.request.execution.complete
        )
        reason = self.matched_reason if matches else self.mismatched_reason
        goal = planning.goal
        return VerificationEvaluation(
            VERDICT_SUPPORT if matches else VERDICT_REFUTE,
            expected,
            detail=(1 if matches else 2, *reason.stable_key()),
            source=goal.source,
            scope=goal.scope,
        )


# object-model: verifier; state=immutable
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceEvidenceSourceVerifier:
    """逐命题核验 aggregate 归属、Evidence 来源和实际 citation。"""

    matched_reason: ObjectIdentity
    mismatched_reason: ObjectIdentity

    def __post_init__(self) -> None:
        _instruction(self.matched_reason, where="reference source match")
        _instruction(self.mismatched_reason, where="reference source mismatch")

    def verify(
            self,
            request: GenerationSourceCheckRequest,
            ) -> VerificationEvaluation:
        """要求两个 emitted candidate 均能回读其 Evidence citation 与 scope。"""
        if not isinstance(request, GenerationSourceCheckRequest):
            raise TypeError("reference source verifier request 类型错误")
        planning = request.postcheck.execution.plan.request
        candidates = {item.stable_key(): item for item in planning.candidates}
        requirements = {
            item.candidate_key: item for item in request.requirements}
        recovered = {item.candidate_key: item for item in request.propositions}
        expected_keys = tuple(item.stable_key() for item in planning.candidates)
        expected_citations = {
            source
            for candidate in planning.candidates
            for source in candidate.citation_sources
        }
        matches = (
            len(expected_keys) == 2
            and set(requirements) == set(expected_keys)
            and set(recovered) == set(expected_keys)
            and set(request.observation.cited_sources) == expected_citations
            and all(
                requirement.source == candidates[key].source
                and requirement.scope == candidates[key].scope
                and requirement.evidence_sources
                == candidates[key].citation_sources
                and set(requirement.evidence_sources).issubset(
                    expected_citations)
                and recovered[key].source == requirement.source
                and recovered[key].scope == requirement.scope
                for key, requirement in requirements.items()
            )
        )
        reason = self.matched_reason if matches else self.mismatched_reason
        goal = planning.goal
        return VerificationEvaluation(
            VERDICT_SUPPORT if matches else VERDICT_REFUTE,
            expected_keys,
            detail=(1 if matches else 2, *reason.stable_key()),
            source=goal.source,
            scope=goal.scope,
        )


__all__ = [
    "GroundedAnswerReferenceEvidenceSourceVerifier",
    "GroundedAnswerReferencePostcheckError",
    "GroundedAnswerReferenceStructureVerifier",
]
