"""从公开对话课程学习证据窗口词项权重。

该模块只学习公开 ``DialogueTrainingPack`` 中词项的跨样本支持次数，并把
结果投影为整数权重。它不读取广域问答评测、答案标签或页面内容，也不携带
任何题目到答案的映射。查询侧只能把这些权重用于已确定来源页内的证据窗口
排序，来源、事实和拒答边界仍由原查询合同负责。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.conversation_training_pack import (
    DialogueTrainingPack,
)
from pure_integer_ai.experiments.ph2_broad_qa_index import broad_qa_terms


EVIDENCE_LEARNING_PROTOCOL_V1 = 1
EVIDENCE_LEARNING_DOMAIN = "pure_integer_ai.broad_qa.evidence_learning.v1"
_MIN_SUPPORT = 2
_BASE_WEIGHT = 10_000
_SUPPORT_WEIGHT = 50_000
_MAX_WEIGHT = 5_000_000


class BroadQaEvidenceLearningError(ValueError):
    """公开训练课程无法形成确定的证据权重模型。"""


@dataclass(frozen=True, slots=True)
class LearnedEvidenceTermWeights:
    """可跨语言重建的词项到整数权重表。"""

    weights: tuple[tuple[str, int], ...]
    training_pack_sha256: str
    case_count: int

    def __post_init__(self) -> None:
        if (not isinstance(self.weights, tuple)
                or tuple(sorted(self.weights)) != self.weights
                or len({term for term, _ in self.weights}) != len(self.weights)
                or any(not isinstance(term, str) or not term
                       or type(weight) is not int or weight <= 0
                       or weight > _MAX_WEIGHT
                       for term, weight in self.weights)
                or len(self.training_pack_sha256) != 64
                or any(item not in "0123456789abcdef"
                       for item in self.training_pack_sha256)
                or type(self.case_count) is not int or self.case_count <= 0):
            raise BroadQaEvidenceLearningError("证据词权重模型字段非法")

    def for_query(self, terms: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
        """只返回当前 query 命中的已学习词项，保持稳定排序。"""
        if (not isinstance(terms, tuple)
                or any(not isinstance(item, str) for item in terms)):
            raise TypeError("query terms 必须是字符串 tuple")
        selected = dict(self.weights)
        return tuple((term, selected[term]) for term in sorted(set(terms))
                     if term in selected)

    def canonical_record(self) -> tuple[int, ...]:
        """整数线协议；字符串按 Unicode scalar 顺序编码。"""
        result = [EVIDENCE_LEARNING_PROTOCOL_V1, self.case_count,
                  len(self.training_pack_sha256)]
        result.extend(ord(item) for item in self.training_pack_sha256)
        result.append(len(self.weights))
        for term, weight in self.weights:
            result.extend((len(term), *map(ord, term), weight))
        return tuple(result)


def learn_evidence_term_weights(
        pack: DialogueTrainingPack,
        ) -> LearnedEvidenceTermWeights:
    """从公开 train cases 学习跨样本词项支持次数。"""
    if not isinstance(pack, DialogueTrainingPack):
        raise TypeError("pack 必须是 DialogueTrainingPack")
    train_cases = tuple(item for item in pack.cases if item.split == "train")
    if not train_cases:
        raise BroadQaEvidenceLearningError("公开 train course 为空")
    support: Counter[str] = Counter()
    for case in train_cases:
        for term in set(broad_qa_terms(case.raw_text)):
            support[term] += 1
    weights = []
    for term, count in support.items():
        if count < _MIN_SUPPORT:
            continue
        weight = min(_MAX_WEIGHT, _BASE_WEIGHT + count * _SUPPORT_WEIGHT)
        weights.append((term, weight))
    if not weights:
        raise BroadQaEvidenceLearningError("公开 course 没有跨样本词项")
    return LearnedEvidenceTermWeights(
        tuple(sorted(weights)), pack.pack_sha256, len(train_cases))


def evidence_learning_sha256(model: LearnedEvidenceTermWeights) -> str:
    """返回整数模型线协议的 SHA-256。"""
    if not isinstance(model, LearnedEvidenceTermWeights):
        raise TypeError("model 必须是 LearnedEvidenceTermWeights")
    payload = b"".join(
        int(value).to_bytes(8, "big", signed=False)
        for value in model.canonical_record())
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "BroadQaEvidenceLearningError",
    "EVIDENCE_LEARNING_DOMAIN",
    "EVIDENCE_LEARNING_PROTOCOL_V1",
    "LearnedEvidenceTermWeights",
    "evidence_learning_sha256",
    "learn_evidence_term_weights",
]
