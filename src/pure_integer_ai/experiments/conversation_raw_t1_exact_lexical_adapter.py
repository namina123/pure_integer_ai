"""T1-G26：exact candidate consensus 到 G1 lexical evidence 的显式适配器。

这是唯一允许的 candidate→lexical 降维路径：必须是 G25 EXACT_UNIT，且 consensus decision
必须为 ACCEPT。REJECT 返回负例状态，不创建 lexical evidence；其他粒度 fail closed。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    RawLexicalEvidence,
)
from pure_integer_ai.experiments.conversation_raw_t1_candidate_granularity import (
    GRANULARITY_EXACT_UNIT,
    RawT1CandidateGranularityAudit,
)
from pure_integer_ai.experiments.conversation_raw_t1_consensus_candidate_evidence import (
    EVIDENCE_DECISION_ACCEPT,
    EVIDENCE_DECISION_REJECT,
    RawT1ConsensusCandidateEvidence,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservation,
)


RAW_T1_EXACT_LEXICAL_ADAPTER_PROTOCOL_V1 = 1
ADAPTER_ACCEPTED = 1
ADAPTER_REJECTED_NEGATIVE = 2
ADAPTER_GRANULARITY_MISMATCH = 3


class RawT1ExactLexicalAdapterError(ValueError):
    """exact lexical adapter 输入或 identity 不满足合同。"""


@dataclass(frozen=True, slots=True)
class RawT1ExactLexicalAdapterResult:
    """适配状态与可选 G1 lexical evidence。"""

    status: int
    candidate_evidence_id: str
    lexical_evidence: RawLexicalEvidence | None = None
    unit_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {
                ADAPTER_ACCEPTED, ADAPTER_REJECTED_NEGATIVE,
                ADAPTER_GRANULARITY_MISMATCH,
        }:
            raise RawT1ExactLexicalAdapterError("adapter status 未注册")
        if not isinstance(self.candidate_evidence_id, str) or not self.candidate_evidence_id:
            raise RawT1ExactLexicalAdapterError("candidate evidence id 非法")
        if self.status == ADAPTER_ACCEPTED:
            if type(self.lexical_evidence) is not RawLexicalEvidence or not self.unit_id:
                raise RawT1ExactLexicalAdapterError("accepted adapter result 不完整")
        elif self.lexical_evidence is not None or self.unit_id is not None:
            raise RawT1ExactLexicalAdapterError("非 accepted 不得携带 lexical evidence")

    @property
    def accepted(self) -> bool:
        return self.status == ADAPTER_ACCEPTED

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_T1_EXACT_LEXICAL_ADAPTER_PROTOCOL_V1, self.status,
                  len(self.candidate_evidence_id),
                  *(ord(item) for item in self.candidate_evidence_id)]
        if self.lexical_evidence is None:
            result.append(0)
        else:
            record = self.lexical_evidence.canonical_record()
            result.extend((len(record), *record))
        return tuple(result)


def adapt_exact_candidate_to_lexical_evidence(
        evidence: RawT1ConsensusCandidateEvidence,
        granularity: RawT1CandidateGranularityAudit,
        observation: RawTextObservation,
        *,
        authority: str,
        ) -> RawT1ExactLexicalAdapterResult:
    """执行唯一 candidate→G1 lexical 降维路径。"""
    if type(evidence) is not RawT1ConsensusCandidateEvidence:
        raise TypeError("exact adapter 需要 candidate evidence")
    if type(granularity) is not RawT1CandidateGranularityAudit:
        raise TypeError("exact adapter 需要 granularity audit")
    if type(observation) is not RawTextObservation:
        raise TypeError("exact adapter 需要 observation")
    if not isinstance(authority, str) or not authority or authority.strip() != authority:
        raise RawT1ExactLexicalAdapterError("authority 非法")
    if evidence.observation_id != observation.observation_id:
        raise RawT1ExactLexicalAdapterError("candidate evidence observation identity 漂移")
    if granularity.evidence_id != evidence.evidence_id:
        raise RawT1ExactLexicalAdapterError("granularity/evidence identity 漂移")
    if granularity.status != GRANULARITY_EXACT_UNIT:
        return RawT1ExactLexicalAdapterResult(
            ADAPTER_GRANULARITY_MISMATCH, evidence.evidence_id)
    if evidence.decision == EVIDENCE_DECISION_REJECT:
        return RawT1ExactLexicalAdapterResult(
            ADAPTER_REJECTED_NEGATIVE, evidence.evidence_id)
    if evidence.decision != EVIDENCE_DECISION_ACCEPT:
        raise RawT1ExactLexicalAdapterError("candidate evidence decision 未注册")
    unit_id = granularity.matched_unit_ids[0]
    lexical = RawLexicalEvidence(
        evidence.evidence_id, observation.observation_id,
        observation.source_id, observation.context_id, observation.family_id,
        observation.source_namespace, observation.split, unit_id,
        evidence.label_role, authority,
        evidence.start_scalar, evidence.end_scalar,
        evidence.start_byte, evidence.end_byte,
    )
    return RawT1ExactLexicalAdapterResult(
        ADAPTER_ACCEPTED, evidence.evidence_id, lexical, unit_id)


__all__ = [
    "ADAPTER_ACCEPTED", "ADAPTER_GRANULARITY_MISMATCH", "ADAPTER_REJECTED_NEGATIVE",
    "RAW_T1_EXACT_LEXICAL_ADAPTER_PROTOCOL_V1",
    "RawT1ExactLexicalAdapterError", "RawT1ExactLexicalAdapterResult",
    "adapt_exact_candidate_to_lexical_evidence",
]
