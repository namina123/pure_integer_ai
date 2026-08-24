"""T1-G21：把 training admission 投影为独立标注者可读的只读观察报告。

核心 row 保留 raw scalar/u8 与物理审计状态；中文文本 renderer 只是 host 侧观察工具，不会
回写 annotation、训练 pack 或默认 terminal。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_t1_training_admission import (
    RawT1TrainingAdmission,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_promotion_audit import (
    PROMOTION_ELIGIBLE_FOR_REVIEW,
    PROMOTION_NEGATIVE_WITNESS_ONLY,
    RawTextCandidatePromotionAudit,
)


RAW_T1_ADMISSION_REVIEW_PROTOCOL_V1 = 1
REVIEW_ELIGIBLE = 1
REVIEW_NEGATIVE = 2
REVIEW_REJECTED = 3


class RawT1AdmissionReviewError(ValueError):
    """admission review projection 输入或状态非法。"""


def _u8(value: tuple[int, ...], where: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or any(
            type(item) is not int or item < 0 or item > 255 for item in value):
        raise RawT1AdmissionReviewError(f"{where} 必须是 u8 tuple")
    return value


def _scalars(value: tuple[int, ...], where: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or any(type(item) is not int or item < 0
                                            for item in value):
        raise RawT1AdmissionReviewError(f"{where} 必须是非负整数 tuple")
    return value


# object-model: value; representation=struct; interop=T1-G21
@dataclass(frozen=True, slots=True)
class RawT1AdmissionReviewRow:
    """一条供独立标注者观察的原文与物理状态。"""

    observation_id: str
    split: str
    review_status: int
    raw_u8: tuple[int, ...]
    raw_scalars: tuple[int, ...]
    candidate_count: int
    unit_count: int
    evidence_count: int
    unit_coverage_status: int
    evidence_coverage_status: int

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise RawT1AdmissionReviewError("observation_id 非法")
        if self.split not in {"train", "heldout", "negative"}:
            raise RawT1AdmissionReviewError("split 未注册")
        if self.review_status not in {REVIEW_ELIGIBLE, REVIEW_NEGATIVE, REVIEW_REJECTED}:
            raise RawT1AdmissionReviewError("review_status 未注册")
        _u8(self.raw_u8, "review.raw_u8")
        _scalars(self.raw_scalars, "review.raw_scalars")
        if any(type(value) is not int or value < 0 for value in (
                self.candidate_count, self.unit_count, self.evidence_count,
                self.unit_coverage_status, self.evidence_coverage_status)):
            raise RawT1AdmissionReviewError("review count/status 非法")

    @property
    def text(self) -> str:
        return "".join(chr(value) for value in self.raw_scalars)

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_T1_ADMISSION_REVIEW_PROTOCOL_V1, self.review_status,
                  len(self.observation_id),
                  *(ord(item) for item in self.observation_id),
                  len(self.split), *(ord(item) for item in self.split),
                  len(self.raw_u8), *self.raw_u8,
                  len(self.raw_scalars), *self.raw_scalars,
                  self.candidate_count, self.unit_count, self.evidence_count,
                  self.unit_coverage_status, self.evidence_coverage_status]
        return tuple(result)


@dataclass(frozen=True, slots=True)
class RawT1AdmissionReview:
    """独立标注者观察报告，不包含写回或标注操作。"""

    rows: tuple[RawT1AdmissionReviewRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple) or not self.rows:
            raise RawT1AdmissionReviewError("review rows 不能为空")
        if any(type(item) is not RawT1AdmissionReviewRow for item in self.rows):
            raise TypeError("review row 类型错误")
        if tuple(item.observation_id for item in self.rows) != tuple(
                sorted(item.observation_id for item in self.rows)):
            raise RawT1AdmissionReviewError("review rows 必须按 observation_id 排序")

    @property
    def eligible_count(self) -> int:
        return sum(item.review_status == REVIEW_ELIGIBLE for item in self.rows)

    @property
    def negative_count(self) -> int:
        return sum(item.review_status == REVIEW_NEGATIVE for item in self.rows)

    @property
    def rejected_count(self) -> int:
        return sum(item.review_status == REVIEW_REJECTED for item in self.rows)

    def canonical_record(self) -> tuple[int, ...]:
        result = [RAW_T1_ADMISSION_REVIEW_PROTOCOL_V1, len(self.rows),
                  self.eligible_count, self.negative_count, self.rejected_count]
        for row in self.rows:
            record = row.canonical_record()
            result.extend((len(record), *record))
        return tuple(result)


def build_raw_t1_admission_review(
        admission: RawT1TrainingAdmission,
        ) -> RawT1AdmissionReview:
    """从已通过 admission 的 typed values 生成稳定 review rows。"""
    if type(admission) is not RawT1TrainingAdmission:
        raise TypeError("review 需要 RawT1TrainingAdmission")
    observations = {
        item.observation.observation_id: item.observation
        for item in (*admission.pack.cases, *admission.pack.negatives)
    }
    audits = {item.observation_id: item for item in admission.audits}
    if set(observations) != set(audits):
        raise RawT1AdmissionReviewError("review observation/audit identity 漂移")
    rows = []
    for observation_id in sorted(observations):
        observation = observations[observation_id]
        audit: RawTextCandidatePromotionAudit = audits[observation_id]
        if audit.status == PROMOTION_ELIGIBLE_FOR_REVIEW:
            status = REVIEW_ELIGIBLE
        elif audit.status == PROMOTION_NEGATIVE_WITNESS_ONLY:
            status = REVIEW_NEGATIVE
        else:
            status = REVIEW_REJECTED
        rows.append(RawT1AdmissionReviewRow(
            observation_id, observation.split, status,
            observation.raw_bytes, observation.scalars,
            audit.candidate_count, audit.unit_count, audit.evidence_count,
            audit.unit_coverage_status, audit.evidence_coverage_status,
        ))
    return RawT1AdmissionReview(tuple(rows))


def render_raw_t1_admission_review_zh(
        review: RawT1AdmissionReview,
        ) -> str:
    """生成供人类读取的中文观察文本；不产生任何可回写指令。"""
    if type(review) is not RawT1AdmissionReview:
        raise TypeError("renderer 需要 RawT1AdmissionReview")
    labels = {
        REVIEW_ELIGIBLE: "可送独立标注审核",
        REVIEW_NEGATIVE: "仅保留为负例见证",
        REVIEW_REJECTED: "拒绝，不能进入训练输入",
    }
    lines = [
        "T1 原始文本 admission 观察报告（只读）",
        f"总记录：{len(review.rows)}；可送审：{review.eligible_count}；"
        f"负例：{review.negative_count}；拒绝：{review.rejected_count}",
        "说明：以下状态只描述物理覆盖与课程边界，不代表词义、命题或现实真值。",
    ]
    for row in review.rows:
        lines.extend((
            f"[{row.observation_id}] split={row.split}：{labels[row.review_status]}",
            f"原文：{row.text}",
            f"候选={row.candidate_count}，unit={row.unit_count}，evidence={row.evidence_count}，"
            f"unit_status={row.unit_coverage_status}，evidence_status={row.evidence_coverage_status}",
        ))
    return "\n".join(lines)


__all__ = [
    "RAW_T1_ADMISSION_REVIEW_PROTOCOL_V1",
    "REVIEW_ELIGIBLE", "REVIEW_NEGATIVE", "REVIEW_REJECTED",
    "RawT1AdmissionReview", "RawT1AdmissionReviewError", "RawT1AdmissionReviewRow",
    "build_raw_t1_admission_review", "render_raw_t1_admission_review_zh",
]
