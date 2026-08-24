"""T1-G18：机械句界候选与显式 lexical evidence 的物理对齐审计。

通过该审计只说明 evidence span 位于某个机械候选内；``unit_kind``、词义、命题、关系和
来源资格仍由原有 G1-G3 合同负责。此模块不生成 evidence，也不把候选升级为训练标签。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    RawLexicalEvidence,
    RawLexicalEvidenceError,
    bind_raw_lexical_evidence,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    RawTextCandidateExtraction,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservation,
)


RAW_TEXT_CANDIDATE_EVIDENCE_PROTOCOL_V1 = 1
RAW_TEXT_CANDIDATE_EVIDENCE_COMPLETE = 1
RAW_TEXT_CANDIDATE_EVIDENCE_CROSSES_BOUNDARY = 2
RAW_TEXT_CANDIDATE_EVIDENCE_UNCOVERED = 3


class RawTextCandidateEvidenceError(ValueError):
    """候选/evidence 物理对齐输入越过合同。"""


@dataclass(frozen=True, slots=True)
class RawTextCandidateEvidenceAudit:
    """lexical evidence 对候选的只读覆盖摘要。"""

    status: int
    candidate_count: int
    evidence_count: int
    covered_evidence_count: int
    crossing_evidence_count: int
    uncovered_evidence_count: int
    selected_candidate_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.status not in {
                RAW_TEXT_CANDIDATE_EVIDENCE_COMPLETE,
                RAW_TEXT_CANDIDATE_EVIDENCE_CROSSES_BOUNDARY,
                RAW_TEXT_CANDIDATE_EVIDENCE_UNCOVERED,
        }:
            raise RawTextCandidateEvidenceError("evidence audit status 未注册")
        counts = (self.candidate_count, self.evidence_count,
                  self.covered_evidence_count, self.crossing_evidence_count,
                  self.uncovered_evidence_count)
        if any(type(value) is not int or value < 0 for value in counts):
            raise RawTextCandidateEvidenceError("evidence audit count 非法")
        if (self.covered_evidence_count + self.crossing_evidence_count
                + self.uncovered_evidence_count != self.evidence_count):
            raise RawTextCandidateEvidenceError("evidence audit count 不守恒")
        if len(self.selected_candidate_ordinals) != self.covered_evidence_count:
            raise RawTextCandidateEvidenceError("selected candidate count 漂移")
        if any(type(value) is not int or value < 0
               for value in self.selected_candidate_ordinals):
            raise RawTextCandidateEvidenceError("selected candidate ordinal 非法")

    @property
    def complete(self) -> bool:
        return self.status == RAW_TEXT_CANDIDATE_EVIDENCE_COMPLETE

    def canonical_record(self) -> tuple[int, ...]:
        return (RAW_TEXT_CANDIDATE_EVIDENCE_PROTOCOL_V1, self.status,
                self.candidate_count, self.evidence_count,
                self.covered_evidence_count, self.crossing_evidence_count,
                self.uncovered_evidence_count,
                len(self.selected_candidate_ordinals),
                *self.selected_candidate_ordinals)


def audit_raw_text_candidate_lexical_evidence(
        extraction: RawTextCandidateExtraction,
        observation: RawTextObservation,
        evidence: tuple[RawLexicalEvidence, ...],
        ) -> RawTextCandidateEvidenceAudit:
    """核对每条 lexical evidence 是否完整落在一个机械候选中。"""
    if type(extraction) is not RawTextCandidateExtraction:
        raise TypeError("evidence audit 需要 RawTextCandidateExtraction")
    if type(observation) is not RawTextObservation:
        raise TypeError("evidence audit 需要 RawTextObservation")
    if not isinstance(evidence, tuple):
        raise TypeError("evidence audit 需要 evidence tuple")
    if not evidence:
        raise RawTextCandidateEvidenceError("evidence 不能为空")
    if (not extraction.accepted
            or extraction.intake.raw_input_bytes != observation.raw_bytes):
        raise RawTextCandidateEvidenceError("candidate/observation raw input 不一致")
    if len({item.evidence_id for item in evidence}) != len(evidence):
        raise RawTextCandidateEvidenceError("evidence_id 不得重复")
    covered: list[int] = []
    crossing = 0
    uncovered = 0
    for item in evidence:
        try:
            bind_raw_lexical_evidence(observation, item)
        except (RawLexicalEvidenceError, TypeError) as error:
            raise RawTextCandidateEvidenceError("lexical evidence 物理绑定失败") from error
        unit = next(unit for unit in observation.units if unit.unit_id == item.unit_id)
        containing = tuple(
            candidate for candidate in extraction.candidates
            if candidate.start_scalar <= unit.start_scalar
            and unit.end_scalar <= candidate.end_scalar
            and candidate.start_byte <= unit.start_byte
            and unit.end_byte <= candidate.end_byte)
        if len(containing) == 1:
            covered.append(containing[0].ordinal)
            continue
        intersects = any(
            candidate.start_scalar < unit.end_scalar
            and unit.start_scalar < candidate.end_scalar
            for candidate in extraction.candidates)
        if intersects:
            crossing += 1
        else:
            uncovered += 1
    if crossing:
        status = RAW_TEXT_CANDIDATE_EVIDENCE_CROSSES_BOUNDARY
    elif uncovered:
        status = RAW_TEXT_CANDIDATE_EVIDENCE_UNCOVERED
    else:
        status = RAW_TEXT_CANDIDATE_EVIDENCE_COMPLETE
    return RawTextCandidateEvidenceAudit(
        status, len(extraction.candidates), len(evidence), len(covered),
        crossing, uncovered, tuple(covered))


__all__ = [
    "RAW_TEXT_CANDIDATE_EVIDENCE_COMPLETE",
    "RAW_TEXT_CANDIDATE_EVIDENCE_CROSSES_BOUNDARY",
    "RAW_TEXT_CANDIDATE_EVIDENCE_PROTOCOL_V1",
    "RAW_TEXT_CANDIDATE_EVIDENCE_UNCOVERED",
    "RawTextCandidateEvidenceAudit",
    "RawTextCandidateEvidenceError",
    "audit_raw_text_candidate_lexical_evidence",
]
