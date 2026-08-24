"""T1-G24：把 G23 ready consensus 投影为独立候选证据记录。

该记录与 G1 lexical evidence 分开：candidate span 可能跨越多个 lexical unit，不能静默降维。
它只保存 consensus 对 candidate 的显式判断，供后续粒度适配器读取，不自动进入 G14。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_t1_annotation_consensus import (
    CONSENSUS_ACCEPT,
    CONSENSUS_REJECT,
    RawT1AnnotationConsensus,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservation,
)


RAW_T1_CONSENSUS_CANDIDATE_EVIDENCE_PROTOCOL_V1 = 1
EVIDENCE_DECISION_ACCEPT = 1
EVIDENCE_DECISION_REJECT = 2


class RawT1ConsensusCandidateEvidenceError(ValueError):
    """ready consensus 无法安全投影为候选证据。"""


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RawT1ConsensusCandidateEvidenceError(f"{where} 必须是规范非空字符串")
    return value


def _nonnegative(value: object, where: str) -> int:
    if type(value) is not int or value < 0:
        raise RawT1ConsensusCandidateEvidenceError(f"{where} 必须是非负严格整数")
    return value


# object-model: value; representation=struct; interop=T1-G24
@dataclass(frozen=True, slots=True)
class RawT1ConsensusCandidateEvidence:
    """一条不降维的 candidate-level consensus evidence。"""

    evidence_id: str
    observation_id: str
    source_id: str
    context_id: str
    family_id: str
    source_namespace: str
    candidate_ordinal: int
    start_scalar: int
    end_scalar: int
    start_byte: int
    end_byte: int
    decision: int
    label_role: str
    reason_code: int
    reviewer_scopes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("evidence_id", "observation_id", "source_id", "context_id",
                     "family_id", "source_namespace"):
            _text(getattr(self, name), f"evidence.{name}")
        _nonnegative(self.candidate_ordinal, "evidence.candidate_ordinal")
        _nonnegative(self.start_scalar, "evidence.start_scalar")
        _nonnegative(self.start_byte, "evidence.start_byte")
        if type(self.end_scalar) is not int or self.end_scalar <= self.start_scalar:
            raise RawT1ConsensusCandidateEvidenceError("evidence scalar span 非法")
        if type(self.end_byte) is not int or self.end_byte <= self.start_byte:
            raise RawT1ConsensusCandidateEvidenceError("evidence byte span 非法")
        if self.decision not in {EVIDENCE_DECISION_ACCEPT, EVIDENCE_DECISION_REJECT}:
            raise RawT1ConsensusCandidateEvidenceError("evidence decision 未注册")
        _text(self.label_role, "evidence.label_role")
        _nonnegative(self.reason_code, "evidence.reason_code")
        if not isinstance(self.reviewer_scopes, tuple) or len(self.reviewer_scopes) < 2:
            raise RawT1ConsensusCandidateEvidenceError("reviewer scopes 不足")
        if tuple(sorted(set(self.reviewer_scopes))) != self.reviewer_scopes:
            raise RawT1ConsensusCandidateEvidenceError("reviewer scopes 必须排序且唯一")

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_T1_CONSENSUS_CANDIDATE_EVIDENCE_PROTOCOL_V1]
        for value in (self.evidence_id, self.observation_id, self.source_id,
                      self.context_id, self.family_id, self.source_namespace):
            result.extend((len(value), *(ord(item) for item in value)))
        result.extend((self.candidate_ordinal, self.start_scalar, self.end_scalar,
                       self.start_byte, self.end_byte, self.decision,
                       len(self.label_role), *(ord(item) for item in self.label_role),
                       self.reason_code, len(self.reviewer_scopes)))
        for scope in self.reviewer_scopes:
            result.extend((len(scope), *(ord(item) for item in scope)))
        return tuple(result)


def project_raw_t1_consensus_candidate_evidence(
        consensus: RawT1AnnotationConsensus,
        observation: RawTextObservation,
        *,
        evidence_namespace: str,
        ) -> tuple[RawT1ConsensusCandidateEvidence, ...]:
    """仅投影 ready ACCEPT/REJECT consensus；DEFER/CONFLICT/INCOMPLETE 拒绝。"""
    if type(consensus) is not RawT1AnnotationConsensus:
        raise TypeError("candidate evidence 需要 RawT1AnnotationConsensus")
    if type(observation) is not RawTextObservation:
        raise TypeError("candidate evidence 需要 RawTextObservation")
    _text(evidence_namespace, "evidence_namespace")
    if (consensus.observation_id != observation.observation_id
            or consensus.source_namespace != observation.source_namespace):
        raise RawT1ConsensusCandidateEvidenceError("consensus/observation identity 漂移")
    if not consensus.ready_for_training_review:
        raise RawT1ConsensusCandidateEvidenceError("consensus 尚未 ready")
    result = []
    for item in consensus.decisions:
        if item.status not in {CONSENSUS_ACCEPT, CONSENSUS_REJECT}:
            raise RawT1ConsensusCandidateEvidenceError("consensus 含未 ready decision")
        decision = (EVIDENCE_DECISION_ACCEPT
                    if item.status == CONSENSUS_ACCEPT
                    else EVIDENCE_DECISION_REJECT)
        evidence_id = f"{evidence_namespace}:{observation.observation_id}:{item.candidate_ordinal}"
        result.append(RawT1ConsensusCandidateEvidence(
            evidence_id, observation.observation_id, observation.source_id,
            observation.context_id, observation.family_id, observation.source_namespace,
            item.candidate_ordinal, item.start_scalar, item.end_scalar,
            item.start_byte, item.end_byte, decision, item.label_role,
            item.reason_code, consensus.reviewer_scopes,
        ))
    return tuple(result)


__all__ = [
    "EVIDENCE_DECISION_ACCEPT", "EVIDENCE_DECISION_REJECT",
    "RAW_T1_CONSENSUS_CANDIDATE_EVIDENCE_PROTOCOL_V1",
    "RawT1ConsensusCandidateEvidence",
    "RawT1ConsensusCandidateEvidenceError",
    "project_raw_t1_consensus_candidate_evidence",
]
