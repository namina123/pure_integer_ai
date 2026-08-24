"""T1-G17：机械句界候选与显式 observation unit 的物理覆盖审计。

本模块只比较 scalar/byte 坐标，不把候选自动标成词法、命题或关系。覆盖通过后仍需原有
authority/evidence 链，覆盖失败只产生可审计状态，不生成 fallback。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    RawTextCandidateExtraction,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservation,
)


RAW_TEXT_CANDIDATE_COVERAGE_PROTOCOL_V1 = 1
RAW_TEXT_CANDIDATE_COVERAGE_COMPLETE = 1
RAW_TEXT_CANDIDATE_COVERAGE_UNIT_CROSSES_BOUNDARY = 2
RAW_TEXT_CANDIDATE_COVERAGE_UNIT_UNCOVERED = 3


class RawTextCandidateCoverageError(ValueError):
    """候选/observation 物理输入不一致。"""


@dataclass(frozen=True, slots=True)
class RawTextCandidateCoverageAudit:
    """候选对显式 unit 的只读覆盖结果。"""

    status: int
    candidate_count: int
    unit_count: int
    covered_unit_count: int
    crossing_unit_count: int
    uncovered_unit_count: int
    selected_candidate_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.status not in {
                RAW_TEXT_CANDIDATE_COVERAGE_COMPLETE,
                RAW_TEXT_CANDIDATE_COVERAGE_UNIT_CROSSES_BOUNDARY,
                RAW_TEXT_CANDIDATE_COVERAGE_UNIT_UNCOVERED,
        }:
            raise RawTextCandidateCoverageError("coverage status 未注册")
        values = (self.candidate_count, self.unit_count, self.covered_unit_count,
                  self.crossing_unit_count, self.uncovered_unit_count)
        if any(type(value) is not int or value < 0 for value in values):
            raise RawTextCandidateCoverageError("coverage count 必须是非负严格整数")
        if self.covered_unit_count + self.crossing_unit_count + self.uncovered_unit_count != self.unit_count:
            raise RawTextCandidateCoverageError("coverage counts 不守恒")
        if len(self.selected_candidate_ordinals) != self.covered_unit_count:
            raise RawTextCandidateCoverageError("selected candidate count 漂移")
        if any(type(value) is not int or value < 0 for value in self.selected_candidate_ordinals):
            raise RawTextCandidateCoverageError("selected candidate ordinal 非法")

    @property
    def complete(self) -> bool:
        return self.status == RAW_TEXT_CANDIDATE_COVERAGE_COMPLETE

    def canonical_record(self) -> tuple[int, ...]:
        return (RAW_TEXT_CANDIDATE_COVERAGE_PROTOCOL_V1, self.status,
                self.candidate_count, self.unit_count, self.covered_unit_count,
                self.crossing_unit_count, self.uncovered_unit_count,
                len(self.selected_candidate_ordinals),
                *self.selected_candidate_ordinals)


def audit_raw_text_candidate_coverage(
        extraction: RawTextCandidateExtraction,
        observation: RawTextObservation,
        ) -> RawTextCandidateCoverageAudit:
    """比较候选与 observation 的物理 span；不解释 unit.role。"""
    if type(extraction) is not RawTextCandidateExtraction:
        raise TypeError("coverage 需要 RawTextCandidateExtraction")
    if type(observation) is not RawTextObservation:
        raise TypeError("coverage 需要 RawTextObservation")
    if (not extraction.accepted
            or extraction.intake.raw_input_bytes != observation.raw_bytes):
        raise RawTextCandidateCoverageError("candidate/observation raw input 不一致")
    candidates = extraction.candidates
    covered = []
    crossing = 0
    uncovered = 0
    for unit in observation.units:
        containing = tuple(
            item for item in candidates
            if item.start_scalar <= unit.start_scalar
            and unit.end_scalar <= item.end_scalar
            and item.start_byte <= unit.start_byte
            and unit.end_byte <= item.end_byte)
        if len(containing) == 1:
            covered.append(containing[0].ordinal)
            continue
        intersects = any(
            item.start_scalar < unit.end_scalar
            and unit.start_scalar < item.end_scalar
            for item in candidates)
        if intersects:
            crossing += 1
        else:
            uncovered += 1
    if crossing:
        status = RAW_TEXT_CANDIDATE_COVERAGE_UNIT_CROSSES_BOUNDARY
    elif uncovered:
        status = RAW_TEXT_CANDIDATE_COVERAGE_UNIT_UNCOVERED
    else:
        status = RAW_TEXT_CANDIDATE_COVERAGE_COMPLETE
    return RawTextCandidateCoverageAudit(
        status, len(candidates), len(observation.units), len(covered),
        crossing, uncovered, tuple(covered))


__all__ = [
    "RAW_TEXT_CANDIDATE_COVERAGE_COMPLETE",
    "RAW_TEXT_CANDIDATE_COVERAGE_PROTOCOL_V1",
    "RAW_TEXT_CANDIDATE_COVERAGE_UNIT_CROSSES_BOUNDARY",
    "RAW_TEXT_CANDIDATE_COVERAGE_UNIT_UNCOVERED",
    "RawTextCandidateCoverageAudit",
    "RawTextCandidateCoverageError",
    "audit_raw_text_candidate_coverage",
]
