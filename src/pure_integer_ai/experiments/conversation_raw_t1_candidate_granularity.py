"""T1-G25：candidate-level evidence 降维到 lexical unit 的粒度审计。

默认拒绝降维。只有 candidate 与恰好一个 observation unit 的 scalar/byte span 完全相等时，
才报告 ``EXACT_UNIT``；覆盖多个 unit、部分覆盖或无 unit 均不生成 G1 evidence。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_t1_consensus_candidate_evidence import (
    RawT1ConsensusCandidateEvidence,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservation,
)


RAW_T1_CANDIDATE_GRANULARITY_PROTOCOL_V1 = 1
GRANULARITY_EXACT_UNIT = 1
GRANULARITY_COVERS_MULTIPLE_UNITS = 2
GRANULARITY_PARTIAL_UNIT = 3
GRANULARITY_NO_UNIT = 4


class RawT1CandidateGranularityError(ValueError):
    """candidate/unit 粒度输入或状态非法。"""


@dataclass(frozen=True, slots=True)
class RawT1CandidateGranularityAudit:
    """candidate-level evidence 的 lexical unit 降维审计。"""

    evidence_id: str
    candidate_ordinal: int
    status: int
    matched_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise RawT1CandidateGranularityError("evidence_id 非法")
        if type(self.candidate_ordinal) is not int or self.candidate_ordinal < 0:
            raise RawT1CandidateGranularityError("candidate ordinal 非法")
        if self.status not in {
                GRANULARITY_EXACT_UNIT, GRANULARITY_COVERS_MULTIPLE_UNITS,
                GRANULARITY_PARTIAL_UNIT, GRANULARITY_NO_UNIT,
        }:
            raise RawT1CandidateGranularityError("granularity status 未注册")
        if tuple(sorted(set(self.matched_unit_ids))) != self.matched_unit_ids:
            raise RawT1CandidateGranularityError("matched unit ids 必须排序且唯一")
        if self.status == GRANULARITY_EXACT_UNIT and len(self.matched_unit_ids) != 1:
            raise RawT1CandidateGranularityError("EXACT_UNIT 必须恰好一个 unit")
        if self.status == GRANULARITY_COVERS_MULTIPLE_UNITS and len(self.matched_unit_ids) < 2:
            raise RawT1CandidateGranularityError("MULTIPLE_UNITS 至少两个 unit")

    @property
    def can_downcast_to_lexical_unit(self) -> bool:
        return self.status == GRANULARITY_EXACT_UNIT

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_T1_CANDIDATE_GRANULARITY_PROTOCOL_V1,
                  self.candidate_ordinal, self.status,
                  len(self.evidence_id), *(ord(item) for item in self.evidence_id),
                  len(self.matched_unit_ids)]
        for unit_id in self.matched_unit_ids:
            result.extend((len(unit_id), *(ord(item) for item in unit_id)))
        return tuple(result)


def audit_raw_t1_candidate_granularity(
        evidence: RawT1ConsensusCandidateEvidence,
        observation: RawTextObservation,
        ) -> RawT1CandidateGranularityAudit:
    """比较 candidate span 与 observation units；不返回 lexical evidence。"""
    if type(evidence) is not RawT1ConsensusCandidateEvidence:
        raise TypeError("granularity audit 需要 candidate evidence")
    if type(observation) is not RawTextObservation:
        raise TypeError("granularity audit 需要 observation")
    if (evidence.observation_id != observation.observation_id
            or evidence.source_id != observation.source_id
            or evidence.context_id != observation.context_id
            or evidence.family_id != observation.family_id
            or evidence.source_namespace != observation.source_namespace):
        raise RawT1CandidateGranularityError("candidate evidence identity 漂移")
    exact = []
    containing = []
    intersects = []
    for unit in observation.units:
        unit_span = (unit.start_scalar, unit.end_scalar, unit.start_byte, unit.end_byte)
        candidate_span = (evidence.start_scalar, evidence.end_scalar,
                          evidence.start_byte, evidence.end_byte)
        if unit_span == candidate_span:
            exact.append(unit.unit_id)
        if (evidence.start_scalar <= unit.start_scalar
                and unit.end_scalar <= evidence.end_scalar):
            containing.append(unit.unit_id)
        if (evidence.start_scalar < unit.end_scalar
                and unit.start_scalar < evidence.end_scalar):
            intersects.append(unit.unit_id)
    if len(exact) == 1:
        status = GRANULARITY_EXACT_UNIT
        matched = tuple(sorted(exact))
    elif len(containing) >= 2:
        status = GRANULARITY_COVERS_MULTIPLE_UNITS
        matched = tuple(sorted(containing))
    elif intersects:
        status = GRANULARITY_PARTIAL_UNIT
        matched = tuple(sorted(set(intersects)))
    else:
        status = GRANULARITY_NO_UNIT
        matched = ()
    return RawT1CandidateGranularityAudit(
        evidence.evidence_id, evidence.candidate_ordinal, status, matched)


__all__ = [
    "GRANULARITY_COVERS_MULTIPLE_UNITS", "GRANULARITY_EXACT_UNIT",
    "GRANULARITY_NO_UNIT", "GRANULARITY_PARTIAL_UNIT",
    "RAW_T1_CANDIDATE_GRANULARITY_PROTOCOL_V1",
    "RawT1CandidateGranularityAudit", "RawT1CandidateGranularityError",
    "audit_raw_t1_candidate_granularity",
]
