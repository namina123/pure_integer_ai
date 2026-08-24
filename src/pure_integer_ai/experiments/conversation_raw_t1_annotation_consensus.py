"""T1-G23：多标注者提交的共识、冲突和缺标合并协议。

该模块只合并已绑定的 G22 submissions。它不裁决哪一个 label 在现实中正确，不写回 G14，
也不把冲突自动投票成“多数正确”。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_t1_annotation_submission import (
    ANNOTATION_ACCEPT,
    ANNOTATION_DEFER,
    ANNOTATION_REJECT,
    RawT1AnnotationDecision,
    RawT1AnnotationSubmission,
    validate_raw_t1_annotation_submission,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    RawTextCandidateExtraction,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservation,
)


RAW_T1_ANNOTATION_CONSENSUS_PROTOCOL_V1 = 1
CONSENSUS_ACCEPT = 1
CONSENSUS_REJECT = 2
CONSENSUS_DEFER = 3
CONSENSUS_CONFLICT = 4
CONSENSUS_INCOMPLETE = 5
CONSENSUS_READY = frozenset({CONSENSUS_ACCEPT, CONSENSUS_REJECT})


class RawT1AnnotationConsensusError(ValueError):
    """多标注者合并输入、身份或一致性状态非法。"""


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RawT1AnnotationConsensusError(f"{where} 必须是规范非空字符串")
    return value


@dataclass(frozen=True, slots=True)
class RawT1ConsensusDecision:
    """一个 candidate 的多标注者合并状态。"""

    candidate_ordinal: int
    start_scalar: int
    end_scalar: int
    start_byte: int
    end_byte: int
    status: int
    label_role: str
    reason_code: int
    reviewer_count: int

    def __post_init__(self) -> None:
        if type(self.candidate_ordinal) is not int or self.candidate_ordinal < 0:
            raise RawT1AnnotationConsensusError("consensus ordinal 非法")
        if type(self.start_scalar) is not int or type(self.end_scalar) is not int \
                or self.end_scalar <= self.start_scalar:
            raise RawT1AnnotationConsensusError("consensus scalar span 非法")
        if type(self.start_byte) is not int or type(self.end_byte) is not int \
                or self.end_byte <= self.start_byte:
            raise RawT1AnnotationConsensusError("consensus byte span 非法")
        if self.status not in {
                CONSENSUS_ACCEPT, CONSENSUS_REJECT, CONSENSUS_DEFER,
                CONSENSUS_CONFLICT, CONSENSUS_INCOMPLETE,
        }:
            raise RawT1AnnotationConsensusError("consensus status 未注册")
        if not isinstance(self.label_role, str) or self.label_role.strip() != self.label_role:
            raise RawT1AnnotationConsensusError("consensus label_role 非法")
        if type(self.reason_code) is not int or self.reason_code < 0:
            raise RawT1AnnotationConsensusError("consensus reason_code 非法")
        if type(self.reviewer_count) is not int or self.reviewer_count < 0:
            raise RawT1AnnotationConsensusError("consensus reviewer_count 非法")

    @property
    def ready(self) -> bool:
        return self.status in CONSENSUS_READY

    def canonical_record(self) -> tuple[int, ...]:
        return (RAW_T1_ANNOTATION_CONSENSUS_PROTOCOL_V1,
                self.candidate_ordinal, self.start_scalar, self.end_scalar,
                self.start_byte, self.end_byte, self.status,
                len(self.label_role), *(ord(item) for item in self.label_role),
                self.reason_code, self.reviewer_count)


@dataclass(frozen=True, slots=True)
class RawT1AnnotationConsensus:
    """同一 observation 的多标注者合并结果。"""

    observation_id: str
    source_namespace: str
    reviewer_scopes: tuple[str, ...]
    decisions: tuple[RawT1ConsensusDecision, ...]

    def __post_init__(self) -> None:
        _text(self.observation_id, "consensus.observation_id")
        _text(self.source_namespace, "consensus.source_namespace")
        if not isinstance(self.reviewer_scopes, tuple) or len(self.reviewer_scopes) < 2:
            raise RawT1AnnotationConsensusError("至少需要两个 reviewer scope")
        if tuple(sorted(set(self.reviewer_scopes))) != self.reviewer_scopes:
            raise RawT1AnnotationConsensusError("reviewer scopes 必须排序且唯一")
        if not isinstance(self.decisions, tuple) or not self.decisions:
            raise RawT1AnnotationConsensusError("consensus decisions 不能为空")
        if tuple(item.candidate_ordinal for item in self.decisions) != tuple(
                range(len(self.decisions))):
            raise RawT1AnnotationConsensusError("consensus candidates 必须完整连续")
        reviewer_count = len(self.reviewer_scopes)
        for item in self.decisions:
            if item.reviewer_count > reviewer_count:
                raise RawT1AnnotationConsensusError("consensus reviewer_count 超出 scope")
            if item.status != CONSENSUS_INCOMPLETE and item.reviewer_count != reviewer_count:
                raise RawT1AnnotationConsensusError(
                    "非 incomplete consensus 必须包含全部 reviewer")

    @property
    def ready_for_training_review(self) -> bool:
        return all(item.ready for item in self.decisions)

    @property
    def conflict_count(self) -> int:
        return sum(item.status == CONSENSUS_CONFLICT for item in self.decisions)

    @property
    def incomplete_count(self) -> int:
        return sum(item.status == CONSENSUS_INCOMPLETE for item in self.decisions)

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_T1_ANNOTATION_CONSENSUS_PROTOCOL_V1]
        for value in (self.observation_id, self.source_namespace):
            result.extend((len(value), *(ord(item) for item in value)))
        result.append(len(self.reviewer_scopes))
        for scope in self.reviewer_scopes:
            result.extend((len(scope), *(ord(item) for item in scope)))
        result.append(len(self.decisions))
        for decision in self.decisions:
            record = decision.canonical_record()
            result.extend((len(record), *record))
        return tuple(result)


def _decision_signature(item: RawT1AnnotationDecision) -> tuple[int, str, int]:
    return item.decision, item.label_role, item.reason_code


def merge_raw_t1_annotation_submissions(
        extraction: RawTextCandidateExtraction,
        observation: RawTextObservation,
        submissions: tuple[RawT1AnnotationSubmission, ...],
        ) -> RawT1AnnotationConsensus:
    """严格绑定并合并 submissions；不使用多数投票覆盖冲突。"""
    if not isinstance(submissions, tuple) or len(submissions) < 2:
        raise RawT1AnnotationConsensusError("至少需要两个 submissions")
    validated = tuple(validate_raw_t1_annotation_submission(
        extraction, observation, item) for item in submissions)
    scopes = tuple(sorted(item.reviewer_scope for item in validated))
    if len(set(scopes)) != len(scopes):
        raise RawT1AnnotationConsensusError("reviewer scope 不得重复")
    if tuple(item.observation_id for item in validated) != (
            observation.observation_id,) * len(validated):
        raise RawT1AnnotationConsensusError("submission observation identity 漂移")
    by_ordinal: dict[int, list[RawT1AnnotationDecision]] = {
        ordinal: [] for ordinal in range(len(extraction.candidates))
    }
    for submission in validated:
        for decision in submission.decisions:
            by_ordinal[decision.candidate_ordinal].append(decision)
    decisions = []
    for ordinal, candidate in enumerate(extraction.candidates):
        values = by_ordinal[ordinal]
        signatures = {_decision_signature(item) for item in values}
        if not values:
            status, label, reason = CONSENSUS_INCOMPLETE, "", 0
        elif len(values) != len(validated):
            status, label, reason = CONSENSUS_INCOMPLETE, "", 0
        elif len(signatures) != 1:
            status, label, reason = CONSENSUS_CONFLICT, "", 0
        else:
            decision, label, reason = next(iter(signatures))
            status = {
                ANNOTATION_ACCEPT: CONSENSUS_ACCEPT,
                ANNOTATION_REJECT: CONSENSUS_REJECT,
                ANNOTATION_DEFER: CONSENSUS_DEFER,
            }[decision]
        decisions.append(RawT1ConsensusDecision(
            ordinal, candidate.start_scalar, candidate.end_scalar,
            candidate.start_byte, candidate.end_byte, status, label, reason,
            len(values)))
    return RawT1AnnotationConsensus(
        observation.observation_id, observation.source_namespace,
        scopes, tuple(decisions))


__all__ = [
    "CONSENSUS_ACCEPT", "CONSENSUS_CONFLICT", "CONSENSUS_DEFER",
    "CONSENSUS_INCOMPLETE", "CONSENSUS_REJECT", "CONSENSUS_READY",
    "RAW_T1_ANNOTATION_CONSENSUS_PROTOCOL_V1",
    "RawT1AnnotationConsensus", "RawT1AnnotationConsensusError",
    "RawT1ConsensusDecision", "merge_raw_t1_annotation_submissions",
]
