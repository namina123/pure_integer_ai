"""把已锁定关系证据窗口投影为可回溯的原文 value span。

投影只返回来源窗口中的原文子串，不生成句子、不补事实、不引用页面外
内容。候选边界来自通用标点结构，选择由公开 relation model 与问题表面
重叠共同授权；无法唯一确定时返回 ``None``。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_broad_qa_index import broad_qa_terms
from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    LearnedRelationEvidenceModel,
)


_SEPARATORS = frozenset("，,、；;。！？!?\n")
_OPENERS = frozenset("（(【[《〈「『")
_CLOSERS = frozenset("）)】]》〉」』")
_MIN_OVERLAP = 1


class BroadQaRelationEvidenceProjectionError(ValueError):
    """关系证据 value projection 输入或整数边界非法。"""


def _segments(text: str) -> tuple[tuple[int, int, str], ...]:
    """按外层标点切分，括号内分隔符不切断证据值。"""
    if not isinstance(text, str) or not text.strip():
        raise BroadQaRelationEvidenceProjectionError("evidence text 不能为空")
    result = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in _OPENERS:
            depth += 1
        elif char in _CLOSERS:
            depth = max(0, depth - 1)
        elif depth == 0 and char in _SEPARATORS:
            value = text[start:index].strip()
            if value:
                left = start + len(text[start:index]) - len(text[start:index].lstrip())
                result.append((left, index, value))
            start = index + 1
    value = text[start:].strip()
    if value:
        left = start + len(text[start:]) - len(text[start:].lstrip())
        result.append((left, len(text), value))
    return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RelationEvidenceValueProjection:
    """一个只引用来源窗口原文的 typed value span。"""

    family: str
    value: str
    start: int
    end: int
    overlap_count: int

    def __post_init__(self) -> None:
        if (not isinstance(self.family, str) or not self.family
                or not isinstance(self.value, str) or not self.value
                or type(self.start) is not int or self.start < 0
                or type(self.end) is not int or self.end <= self.start
                or type(self.overlap_count) is not int
                or self.overlap_count < _MIN_OVERLAP):
            raise BroadQaRelationEvidenceProjectionError(
                "relation value projection 非法")

    def canonical_record(self) -> tuple[int, ...]:
        result = [len(self.family), *map(ord, self.family),
                  len(self.value), *map(ord, self.value), self.start,
                  self.end, self.overlap_count]
        return tuple(result)


def project_relation_evidence_value(
        model: LearnedRelationEvidenceModel,
        question: str,
        evidence_text: str,
        answer_kinds: tuple[str, ...] | None = None,
        anchor_text: str | None = None,
        ) -> RelationEvidenceValueProjection | None:
    """选择唯一问题限定 value；失败闭合为 ``None``。"""
    if not isinstance(model, LearnedRelationEvidenceModel):
        raise TypeError("relation evidence model 类型错误")
    if not isinstance(question, str) or not question.strip():
        raise BroadQaRelationEvidenceProjectionError("question 不能为空")
    if not isinstance(evidence_text, str) or not evidence_text.strip():
        raise BroadQaRelationEvidenceProjectionError("evidence_text 不能为空")
    if answer_kinds is not None:
        if (not isinstance(answer_kinds, tuple)
                or any(not isinstance(item, str) or not item
                       for item in answer_kinds)):
            raise BroadQaRelationEvidenceProjectionError(
                "answer_kinds 类型错误")
        # relation value projection 目前只覆盖显式实体值问题；因果、时间、
        # 数量等类型必须保留完整来源证据，避免词项巧合截断答案。
        if "ENTITY" not in answer_kinds:
            return None
    if anchor_text is not None and (
            not isinstance(anchor_text, str) or not anchor_text.strip()):
        raise BroadQaRelationEvidenceProjectionError("anchor_text 类型错误")
    family = model.relation_family(question)
    if family is None:
        return None
    question_terms = set(broad_qa_terms(question))
    if anchor_text is not None:
        # 来源标题已由检索层锁定；其词项只负责找页，不应再次把主体段
        # 误选为关系值候选。
        question_terms.difference_update(broad_qa_terms(anchor_text))
    if not question_terms:
        return None
    ranked = []
    for start, end, value in _segments(evidence_text):
        overlap = question_terms.intersection(broad_qa_terms(value))
        if len(overlap) < _MIN_OVERLAP:
            continue
        # 长特征优先于其二元子串；其余排序键保证跨语言稳定回放。
        score = sum(len(term) for term in overlap)
        ranked.append((score, len(overlap), -len(value), -start, value,
                       start, end))
    ranked.sort(reverse=True)
    if not ranked:
        return None
    best = ranked[0]
    if len(ranked) > 1 and (best[0], best[1]) == (ranked[1][0], ranked[1][1]):
        return None
    return RelationEvidenceValueProjection(
        family, best[4], best[5], best[6], best[1])


__all__ = [
    "BroadQaRelationEvidenceProjectionError",
    "RelationEvidenceValueProjection",
    "project_relation_evidence_value",
]
