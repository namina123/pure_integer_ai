"""从公开关系课程学习问式到证据角色的纯整数评分。

该模块只消费严格字段的公开关系课程：问题表面、抽象关系 family 和证据
表面形状。它拒绝答案、gold、页面和实体字段，因此不会建立题目到答案的
映射。运行时只把确定的关系形状分数加到已锁定来源页内的证据窗口排序。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_index import broad_qa_terms


RELATION_EVIDENCE_PROTOCOL_V1 = 1
RELATION_EVIDENCE_DOMAIN = "pure_integer_ai.broad_qa.relation_evidence.v1"
_MIN_SUPPORT = 2
_MIN_QUERY_FEATURES = 2
_MIN_STRONG_QUERY_FEATURES = 2
_MIN_MARGIN = 1
_FEATURE_WEIGHT = 100
_SHAPE_FEATURE_WEIGHT = 200
_MAX_BONUS = 2_000_000
_ALLOWED_RECORD_FIELDS = frozenset({
    "evidence", "family", "item_id", "license_id", "question",
    "source_identity", "split",
})
_ALLOWED_QUESTION_FIELDS = frozenset({"question_surface", "typed_intent"})
_ALLOWED_EVIDENCE_FIELDS = frozenset({"evidence_surface"})
_SHAPE_RE = {
    "LIST_SEPARATOR": ("、", ",", "，"),
    "PARENTHESIS": ("（", "）", "(", ")"),
    "COLON": ("：", ":"),
    "QUOTE": ("“", "”", "\"", "「", "」"),
    "SEMICOLON": ("；", ";"),
}


class BroadQaRelationEvidenceLearningError(ValueError):
    """公开关系课程无法形成确定的关系证据模型。"""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise BroadQaRelationEvidenceLearningError(f"{label} 非法")
    if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
        raise BroadQaRelationEvidenceLearningError(f"{label} 含非法 Unicode scalar")
    return value


def _shape_features(value: str) -> tuple[str, ...]:
    """把证据表面投影为与内容无关的可迁移形状特征。"""
    features = []
    for name, chars in _SHAPE_RE.items():
        count = sum(value.count(char) for char in chars)
        if count:
            # 只保留结构是否出现，不把某个来源的具体标点数量当作
            # 证据模板；这样不同长度的列表/括号内容仍可迁移。
            features.append(f"{name}:PRESENT")
    clause_count = len(tuple(
        item for item in re.split(r"[。！？!?；;]", value) if item))
    if clause_count:
        features.append(f"CLAUSE_COUNT:{min(clause_count, 31)}")
    return tuple(sorted(features))


def _record_features(record: object) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    """严格读取公开 relation/evidence record，拒绝额外字段。"""
    if not isinstance(record, dict) or set(record) != _ALLOWED_RECORD_FIELDS:
        raise BroadQaRelationEvidenceLearningError(
            "relation evidence record 字段集合漂移")
    if record["license_id"] != "CC0-1.0":
        raise BroadQaRelationEvidenceLearningError("relation evidence 许可必须为 CC0")
    family = _text(record["family"], label="relation family")
    _text(record["item_id"], label="relation item_id")
    _text(record["source_identity"], label="relation source_identity")
    question = record["question"]
    evidence = record["evidence"]
    if (not isinstance(question, dict)
            or set(question) != _ALLOWED_QUESTION_FIELDS
            or not isinstance(evidence, dict)
            or set(evidence) != _ALLOWED_EVIDENCE_FIELDS):
        raise BroadQaRelationEvidenceLearningError(
            "relation question/evidence 字段集合漂移")
    question_surface = _text(
        question["question_surface"], label="question_surface")
    evidence_surface = _text(
        evidence["evidence_surface"], label="evidence_surface")
    question_terms = broad_qa_terms(question_surface)
    evidence_terms = broad_qa_terms(evidence_surface)
    if not question_terms or not evidence_terms:
        raise BroadQaRelationEvidenceLearningError(
            "relation question/evidence 缺少可学习特征")
    return family, question_surface, question_terms, (
        *evidence_terms, *_shape_features(evidence_surface))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LearnedRelationEvidenceFamily:
    """一个公开关系 family 的问式/证据特征统计。"""

    family: str
    question_support: tuple[tuple[str, int], ...]
    evidence_support: tuple[tuple[str, int], ...]
    case_count: int

    def __post_init__(self) -> None:
        if (not self.family
                or tuple(self.question_support) != tuple(sorted(self.question_support))
                or tuple(self.evidence_support) != tuple(sorted(self.evidence_support))
                or not self.question_support or not self.evidence_support
                or any(type(count) is not int or count <= 0
                       for _, count in (*self.question_support,
                                        *self.evidence_support))
                or type(self.case_count) is not int
                or self.case_count < _MIN_SUPPORT):
            raise BroadQaRelationEvidenceLearningError(
                "relation evidence family 非法")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LearnedRelationEvidenceModel:
    """可跨语言重建的 relation family 到 evidence-shape 模型。"""

    families: tuple[LearnedRelationEvidenceFamily, ...]
    source_sha256: str
    case_count: int

    def __post_init__(self) -> None:
        if (not self.families
                or tuple(item.family for item in self.families)
                != tuple(sorted(item.family for item in self.families))
                or len({item.family for item in self.families}) != len(self.families)
                or any(not isinstance(item, LearnedRelationEvidenceFamily)
                       for item in self.families)
                or len(self.source_sha256) != 64
                or any(item not in "0123456789abcdef"
                       for item in self.source_sha256)
                or type(self.case_count) is not int or self.case_count <= 0):
            raise BroadQaRelationEvidenceLearningError(
                "relation evidence model 非法")

    def _family_for_question(self, question: str) -> LearnedRelationEvidenceFamily | None:
        terms = set(broad_qa_terms(question))
        if not terms:
            return None
        ranked = []
        for family in self.families:
            support = dict(family.question_support)
            vote = sum(support.get(term, 0) for term in terms)
            overlap = sum(term in support for term in terms)
            strong_overlap = sum(
                support.get(term, 0) >= _MIN_SUPPORT for term in terms)
            ranked.append((
                vote, strong_overlap, overlap, -family.case_count,
                family.family, family))
        ranked.sort(reverse=True)
        best = ranked[0]
        # 一个通用疑问词（如“什么”）不足以确定关系 family；至少要有
        # 两个训练中出现过的问式特征，未知变体才会失败闭合。
        if (best[0] < _MIN_SUPPORT
                or best[1] < _MIN_STRONG_QUERY_FEATURES
                or best[2] < _MIN_QUERY_FEATURES):
            return None
        if len(ranked) > 1 and best[0] - ranked[1][0] < _MIN_MARGIN:
            return None
        return best[-1]

    def relation_family(self, question: str) -> str | None:
        """返回通过支持/边际闸门的 relation family，失败闭合。"""
        if not isinstance(question, str) or not question.strip():
            raise TypeError("relation evidence question 必须是非空字符串")
        family = self._family_for_question(question)
        return family.family if family is not None else None

    def evidence_bonus(self, question: str, evidence_text: str) -> int:
        """返回确定 relation family 对窗口的有限整数偏置。"""
        if not isinstance(question, str) or not question.strip():
            raise TypeError("relation evidence question 必须是非空字符串")
        if not isinstance(evidence_text, str) or not evidence_text.strip():
            raise TypeError("relation evidence window 必须是非空字符串")
        family = self._family_for_question(question)
        if family is None:
            return 0
        observed = set((*broad_qa_terms(evidence_text),
                        *_shape_features(evidence_text)))
        support = dict(family.evidence_support)
        bonus = 0
        for feature in observed:
            count = support.get(feature, 0)
            if count:
                # 同一 family 中越稀有的表面特征，区分证据角色的作用越大；
                # 权重只来自训练计数，不绑定任何实体或答案。
                rarity = max(1, family.case_count - count + 1)
                feature_weight = (
                    _SHAPE_FEATURE_WEIGHT
                    if feature.split(":", 1)[0] in _SHAPE_RE
                    else _FEATURE_WEIGHT)
                bonus += feature_weight * family.case_count * rarity
        return min(_MAX_BONUS, bonus)

    def canonical_record(self) -> tuple[int, ...]:
        result = [RELATION_EVIDENCE_PROTOCOL_V1, self.case_count,
                  len(self.source_sha256), *map(ord, self.source_sha256),
                  len(self.families)]
        for family in self.families:
            for value in (family.family,):
                result.extend((len(value), *map(ord, value)))
            result.append(family.case_count)
            for values in (family.question_support, family.evidence_support):
                result.append(len(values))
                for feature, count in values:
                    result.extend((len(feature), *map(ord, feature), count))
        return tuple(result)


def learn_relation_evidence_model(
        paths: Iterable[str | Path],
        ) -> LearnedRelationEvidenceModel:
    """从 train split 的公开 relation/evidence surface 学习模型。"""
    files = tuple(sorted(Path(item).resolve() for item in paths))
    if not files or len(files) != len(set(files)):
        raise BroadQaRelationEvidenceLearningError("课程 inventory 重复或为空")
    digest = hashlib.sha256()
    support: dict[str, list[tuple[tuple[str, ...], tuple[str, ...]]]] = defaultdict(list)
    total = 0
    for path in files:
        if not path.is_file():
            raise BroadQaRelationEvidenceLearningError(f"课程文件缺失: {path}")
        payload = path.read_bytes()
        digest.update(path.as_posix().encode("utf-8") + b"\0")
        digest.update(payload)
        for line_number, raw in enumerate(payload.splitlines(), 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BroadQaRelationEvidenceLearningError(
                    f"课程 JSONL 非法: {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise BroadQaRelationEvidenceLearningError(
                    f"relation evidence record 非法: {path}:{line_number}")
            split = record.get("split")
            family, _, question_terms, evidence_features = _record_features(record)
            if split != "train":
                if split != "heldout":
                    raise BroadQaRelationEvidenceLearningError(
                        f"relation evidence split 非法: {path}:{line_number}")
                continue
            support[family].append((question_terms, evidence_features))
            total += 1
    families = []
    for family, cases in sorted(support.items()):
        if len(cases) < _MIN_SUPPORT:
            continue
        question_support: Counter[str] = Counter()
        evidence_support: Counter[str] = Counter()
        for question_terms, evidence_features in cases:
            question_support.update(set(question_terms))
            evidence_support.update(set(evidence_features))
        families.append(LearnedRelationEvidenceFamily(
            family,
            tuple(sorted(question_support.items())),
            tuple(sorted(evidence_support.items())),
            len(cases),
        ))
    if not families:
        raise BroadQaRelationEvidenceLearningError(
            "没有达到支持阈值的 relation family")
    return LearnedRelationEvidenceModel(
        tuple(families), digest.hexdigest(), total)


def relation_evidence_sha256(model: LearnedRelationEvidenceModel) -> str:
    """返回纯整数 canonical record 的 SHA-256。"""
    if not isinstance(model, LearnedRelationEvidenceModel):
        raise TypeError("model 必须是 LearnedRelationEvidenceModel")
    payload = b"".join(
        int(value).to_bytes(8, "big", signed=False)
        for value in model.canonical_record())
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "BroadQaRelationEvidenceLearningError",
    "LearnedRelationEvidenceFamily",
    "LearnedRelationEvidenceModel",
    "RELATION_EVIDENCE_DOMAIN",
    "RELATION_EVIDENCE_PROTOCOL_V1",
    "learn_relation_evidence_model",
    "relation_evidence_sha256",
]
