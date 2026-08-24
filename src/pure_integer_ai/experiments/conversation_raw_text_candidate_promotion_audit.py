"""T1-G19：候选/词法证据到训练送审资格的只读闸门。

这是一个审计投影，不执行 promotion、不写训练库。只有 train/heldout observation 且同时
通过 G17 物理 unit 覆盖和 G18 lexical evidence 覆盖，才标为 ``ELIGIBLE_FOR_REVIEW``；
negative 永远只保留为 witness。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    RawLexicalEvidence,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_coverage import (
    RAW_TEXT_CANDIDATE_COVERAGE_COMPLETE,
    RawTextCandidateCoverageAudit,
    audit_raw_text_candidate_coverage,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_evidence import (
    RAW_TEXT_CANDIDATE_EVIDENCE_COMPLETE,
    RawTextCandidateEvidenceAudit,
    audit_raw_text_candidate_lexical_evidence,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    RawTextCandidateExtraction,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservation,
)


RAW_TEXT_CANDIDATE_PROMOTION_PROTOCOL_V1 = 1
PROMOTION_ELIGIBLE_FOR_REVIEW = 1
PROMOTION_NEGATIVE_WITNESS_ONLY = 2
PROMOTION_UNIT_COVERAGE_REJECTED = 3
PROMOTION_EVIDENCE_COVERAGE_REJECTED = 4


class RawTextCandidatePromotionError(ValueError):
    """candidate promotion audit 输入或状态越界。"""


@dataclass(frozen=True, slots=True)
class RawTextCandidatePromotionAudit:
    """候选进入后续人工/课程审核前的只读资格结果。"""

    observation_id: str
    split: str
    status: int
    candidate_count: int
    unit_count: int
    evidence_count: int
    unit_coverage_status: int
    evidence_coverage_status: int

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise RawTextCandidatePromotionError("observation_id 非法")
        if self.split not in {"train", "heldout", "negative"}:
            raise RawTextCandidatePromotionError("split 未注册")
        if self.status not in {
                PROMOTION_ELIGIBLE_FOR_REVIEW,
                PROMOTION_NEGATIVE_WITNESS_ONLY,
                PROMOTION_UNIT_COVERAGE_REJECTED,
                PROMOTION_EVIDENCE_COVERAGE_REJECTED,
        }:
            raise RawTextCandidatePromotionError("promotion status 未注册")
        values = (self.candidate_count, self.unit_count, self.evidence_count)
        if any(type(value) is not int or value < 0 for value in values):
            raise RawTextCandidatePromotionError("promotion count 非法")

    @property
    def eligible_for_review(self) -> bool:
        return self.status == PROMOTION_ELIGIBLE_FOR_REVIEW

    def canonical_record(self) -> tuple[int, ...]:
        return (RAW_TEXT_CANDIDATE_PROMOTION_PROTOCOL_V1,
                self.status, len(self.split),
                *(ord(item) for item in self.split),
                len(self.observation_id),
                *(ord(item) for item in self.observation_id),
                self.candidate_count, self.unit_count, self.evidence_count,
                self.unit_coverage_status, self.evidence_coverage_status)


def audit_raw_text_candidate_promotion(
        extraction: RawTextCandidateExtraction,
        observation: RawTextObservation,
        evidence: tuple[RawLexicalEvidence, ...],
        ) -> RawTextCandidatePromotionAudit:
    """合并 G17/G18 结果；不创建或写入任何训练数据。"""
    if type(extraction) is not RawTextCandidateExtraction:
        raise TypeError("promotion audit 需要 RawTextCandidateExtraction")
    if type(observation) is not RawTextObservation:
        raise TypeError("promotion audit 需要 RawTextObservation")
    unit_audit: RawTextCandidateCoverageAudit = audit_raw_text_candidate_coverage(
        extraction, observation)
    evidence_audit: RawTextCandidateEvidenceAudit = (
        audit_raw_text_candidate_lexical_evidence(extraction, observation, evidence))
    if unit_audit.status != RAW_TEXT_CANDIDATE_COVERAGE_COMPLETE:
        status = PROMOTION_UNIT_COVERAGE_REJECTED
    elif evidence_audit.status != RAW_TEXT_CANDIDATE_EVIDENCE_COMPLETE:
        status = PROMOTION_EVIDENCE_COVERAGE_REJECTED
    elif observation.split == "negative":
        status = PROMOTION_NEGATIVE_WITNESS_ONLY
    else:
        status = PROMOTION_ELIGIBLE_FOR_REVIEW
    return RawTextCandidatePromotionAudit(
        observation.observation_id, observation.split, status,
        len(extraction.candidates), len(observation.units), len(evidence),
        unit_audit.status, evidence_audit.status)


__all__ = [
    "PROMOTION_ELIGIBLE_FOR_REVIEW",
    "PROMOTION_EVIDENCE_COVERAGE_REJECTED",
    "PROMOTION_NEGATIVE_WITNESS_ONLY",
    "PROMOTION_UNIT_COVERAGE_REJECTED",
    "RAW_TEXT_CANDIDATE_PROMOTION_PROTOCOL_V1",
    "RawTextCandidatePromotionAudit",
    "RawTextCandidatePromotionError",
    "audit_raw_text_candidate_promotion",
]
