"""T1-G22：独立标注提交的纯值、来源绑定与回放协议。

标注者只提交对机械 candidate 的结构判断；本模块验证提交是否绑定到原始 observation 和
candidate span，不决定 label 的现实真值，也不自动写入 G14 training pack。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    RawTextCandidateExtraction,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextObservation,
)


RAW_T1_ANNOTATION_PROTOCOL_V1 = 1
ANNOTATION_ACCEPT = 1
ANNOTATION_REJECT = 2
ANNOTATION_DEFER = 3
ANNOTATION_DECISIONS = frozenset({
    ANNOTATION_ACCEPT, ANNOTATION_REJECT, ANNOTATION_DEFER,
})


class RawT1AnnotationSubmissionError(ValueError):
    """独立标注提交越过 identity/span/decision 合同。"""


def _text(value: object, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise RawT1AnnotationSubmissionError(f"{where} 必须是规范字符串")
    if not allow_empty and not value:
        raise RawT1AnnotationSubmissionError(f"{where} 不能为空")
    if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
        raise RawT1AnnotationSubmissionError(f"{where} 含非法 Unicode scalar")
    return value


def _nonnegative(value: object, where: str) -> int:
    if type(value) is not int or value < 0:
        raise RawT1AnnotationSubmissionError(f"{where} 必须是非负严格整数")
    return value


# object-model: value; representation=struct; interop=T1-G22
@dataclass(frozen=True, slots=True)
class RawT1AnnotationDecision:
    """对一个 candidate 的独立结构判断。"""

    candidate_ordinal: int
    start_scalar: int
    end_scalar: int
    start_byte: int
    end_byte: int
    decision: int
    label_role: str
    reason_code: int

    def __post_init__(self) -> None:
        _nonnegative(self.candidate_ordinal, "decision.candidate_ordinal")
        _nonnegative(self.start_scalar, "decision.start_scalar")
        _nonnegative(self.start_byte, "decision.start_byte")
        if type(self.end_scalar) is not int or self.end_scalar <= self.start_scalar:
            raise RawT1AnnotationSubmissionError("decision scalar span 非法")
        if type(self.end_byte) is not int or self.end_byte <= self.start_byte:
            raise RawT1AnnotationSubmissionError("decision byte span 非法")
        if self.decision not in ANNOTATION_DECISIONS:
            raise RawT1AnnotationSubmissionError("decision 未注册")
        _text(self.label_role, "decision.label_role", allow_empty=self.decision != ANNOTATION_ACCEPT)
        _nonnegative(self.reason_code, "decision.reason_code")
        if self.decision == ANNOTATION_ACCEPT and not self.label_role:
            raise RawT1AnnotationSubmissionError("accepted decision 必须有 label_role")

    def canonical_record(self) -> tuple[int, ...]:
        return (RAW_T1_ANNOTATION_PROTOCOL_V1, self.candidate_ordinal,
                self.start_scalar, self.end_scalar, self.start_byte, self.end_byte,
                self.decision, len(self.label_role),
                *(ord(item) for item in self.label_role), self.reason_code)


@dataclass(frozen=True, slots=True)
class RawT1AnnotationSubmission:
    """绑定到单一 observation/candidate extraction 的只读提交。"""

    annotation_id: str
    reviewer_scope: str
    observation_id: str
    source_namespace: str
    decisions: tuple[RawT1AnnotationDecision, ...]

    def __post_init__(self) -> None:
        for name in ("annotation_id", "reviewer_scope", "observation_id", "source_namespace"):
            _text(getattr(self, name), f"submission.{name}")
        if not isinstance(self.decisions, tuple) or not self.decisions:
            raise RawT1AnnotationSubmissionError("submission decisions 不能为空")
        if any(type(item) is not RawT1AnnotationDecision for item in self.decisions):
            raise TypeError("submission decisions 类型错误")
        ordinals = tuple(item.candidate_ordinal for item in self.decisions)
        if ordinals != tuple(sorted(ordinals)) or len(set(ordinals)) != len(ordinals):
            raise RawT1AnnotationSubmissionError("candidate ordinal 必须排序且唯一")

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_T1_ANNOTATION_PROTOCOL_V1]
        for value in (self.annotation_id, self.reviewer_scope,
                      self.observation_id, self.source_namespace):
            result.extend((len(value), *(ord(item) for item in value)))
        result.append(len(self.decisions))
        for decision in self.decisions:
            record = decision.canonical_record()
            result.extend((len(record), *record))
        return tuple(result)


def validate_raw_t1_annotation_submission(
        extraction: RawTextCandidateExtraction,
        observation: RawTextObservation,
        submission: RawT1AnnotationSubmission,
        ) -> RawT1AnnotationSubmission:
    """验证 submission 绑定到 extraction/observation；返回同一不可变值。"""
    if type(extraction) is not RawTextCandidateExtraction:
        raise TypeError("annotation 需要 RawTextCandidateExtraction")
    if type(observation) is not RawTextObservation:
        raise TypeError("annotation 需要 RawTextObservation")
    if type(submission) is not RawT1AnnotationSubmission:
        raise TypeError("annotation 需要 RawT1AnnotationSubmission")
    if (not extraction.accepted
            or extraction.intake.raw_input_bytes != observation.raw_bytes):
        raise RawT1AnnotationSubmissionError("annotation raw identity 不一致")
    if submission.observation_id != observation.observation_id:
        raise RawT1AnnotationSubmissionError("annotation observation identity 不一致")
    if submission.source_namespace != observation.source_namespace:
        raise RawT1AnnotationSubmissionError("annotation source namespace 不一致")
    by_ordinal = {item.ordinal: item for item in extraction.candidates}
    for decision in submission.decisions:
        candidate = by_ordinal.get(decision.candidate_ordinal)
        if candidate is None:
            raise RawT1AnnotationSubmissionError("annotation candidate ordinal 未知")
        if (decision.start_scalar, decision.end_scalar,
                decision.start_byte, decision.end_byte) != (
                    candidate.start_scalar, candidate.end_scalar,
                    candidate.start_byte, candidate.end_byte):
            raise RawT1AnnotationSubmissionError("annotation candidate span 漂移")
    return submission


__all__ = [
    "ANNOTATION_ACCEPT", "ANNOTATION_DEFER", "ANNOTATION_REJECT",
    "RAW_T1_ANNOTATION_PROTOCOL_V1",
    "RawT1AnnotationDecision", "RawT1AnnotationSubmission",
    "RawT1AnnotationSubmissionError",
    "validate_raw_t1_annotation_submission",
]
