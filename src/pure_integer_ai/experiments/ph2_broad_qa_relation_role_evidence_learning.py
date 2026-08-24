"""学习关系证据中的 ``value + qualifier`` 列表结构。

课程必须提供公开的 role span；运行时只返回已锁定来源窗口中的连续原文
子串。问题限定与 qualifier 无法唯一匹配时，模型不投影。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_index import broad_qa_terms
from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    LearnedRelationEvidenceModel,
)


ROLE_EVIDENCE_PROTOCOL_V1 = 1
_MIN_SUPPORT = 2
_RECORD_FIELDS = frozenset({
    "evidence", "family", "item_id", "license_id", "question", "roles",
    "source_identity", "split",
})
_QUESTION_FIELDS = frozenset({"question_surface", "typed_intent"})
_EVIDENCE_FIELDS = frozenset({"evidence_surface"})
_ROLE_FIELDS = frozenset({"end", "role", "start"})


class RelationRoleEvidenceLearningError(ValueError):
    """公开 role evidence 课程或投影输入非法。"""


def _text(value: object, label: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value
            or any(0xD800 <= ord(item) <= 0xDFFF for item in value)):
        raise RelationRoleEvidenceLearningError(f"{label} 非法")
    return value


def _record_shape(record: object) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        raise RelationRoleEvidenceLearningError("role evidence record 字段漂移")
    if record["license_id"] != "CC0-1.0":
        raise RelationRoleEvidenceLearningError("role evidence 必须是 CC0")
    family = _text(record["family"], "family")
    _text(record["item_id"], "item_id")
    _text(record["source_identity"], "source_identity")
    question = record["question"]
    evidence = record["evidence"]
    if (not isinstance(question, dict) or set(question) != _QUESTION_FIELDS
            or not isinstance(evidence, dict)
            or set(evidence) != _EVIDENCE_FIELDS):
        raise RelationRoleEvidenceLearningError("role evidence nested fields 漂移")
    question_surface = _text(question["question_surface"], "question_surface")
    evidence_surface = _text(evidence["evidence_surface"], "evidence_surface")
    roles = record["roles"]
    if not isinstance(roles, list) or not roles:
        raise RelationRoleEvidenceLearningError("roles 不能为空")
    spans = []
    for role in roles:
        if not isinstance(role, dict) or set(role) != _ROLE_FIELDS:
            raise RelationRoleEvidenceLearningError("role span 字段漂移")
        if (type(role["start"]) is not int or type(role["end"]) is not int
                or role["start"] < 0 or role["end"] <= role["start"]
                or role["end"] > len(evidence_surface)):
            raise RelationRoleEvidenceLearningError("role span 边界非法")
        _text(role["role"], "role")
        spans.append((role["start"], role["end"], role["role"]))
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise RelationRoleEvidenceLearningError("role span 重叠或未排序")
    if tuple(item[2] for item in spans) != (
            "value", "qualifier", "value", "qualifier"):
        raise RelationRoleEvidenceLearningError(
            "当前协议只接受 value/qualifier 双项列表")
    first_value, first_qualifier, second_value, second_qualifier = spans
    opener = evidence_surface[first_qualifier[0]]
    closer = evidence_surface[first_qualifier[1] - 1]
    separator = evidence_surface[first_qualifier[1]:second_value[0]]
    if not separator or not opener or not closer:
        raise RelationRoleEvidenceLearningError("role evidence 结构标记缺失")
    return family, question_surface, (opener, closer, separator)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LearnedRelationRoleFamily:
    family: str
    openers: tuple[tuple[str, int], ...]
    closers: tuple[tuple[str, int], ...]
    separators: tuple[tuple[str, int], ...]
    case_count: int

    def __post_init__(self) -> None:
        if (not self.family or type(self.case_count) is not int
                or self.case_count < _MIN_SUPPORT
                or not self.openers or not self.closers or not self.separators):
            raise RelationRoleEvidenceLearningError("learned role family 非法")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LearnedRelationRoleEvidenceModel:
    families: tuple[LearnedRelationRoleFamily, ...]
    source_sha256: str
    case_count: int

    def __post_init__(self) -> None:
        if (not self.families
                or tuple(item.family for item in self.families)
                != tuple(sorted(item.family for item in self.families))
                or len({item.family for item in self.families}) != len(self.families)
                or len(self.source_sha256) != 64
                or type(self.case_count) is not int or self.case_count <= 0):
            raise RelationRoleEvidenceLearningError("role evidence model 非法")

    def family(self, name: str) -> LearnedRelationRoleFamily | None:
        return next((item for item in self.families if item.family == name), None)

    def canonical_record(self) -> tuple[int, ...]:
        result = [ROLE_EVIDENCE_PROTOCOL_V1, self.case_count,
                  len(self.source_sha256), *map(ord, self.source_sha256),
                  len(self.families)]
        for family in self.families:
            result.extend((len(family.family), *map(ord, family.family),
                           family.case_count))
            for values in (family.openers, family.closers, family.separators):
                result.append(len(values))
                for value, count in values:
                    result.extend((len(value), *map(ord, value), count))
        return tuple(result)


def learn_relation_role_evidence_model(
        paths: Iterable[str | Path],
        ) -> LearnedRelationRoleEvidenceModel:
    files = tuple(sorted(Path(item).resolve() for item in paths))
    if not files or len(files) != len(set(files)):
        raise RelationRoleEvidenceLearningError("role evidence inventory 非法")
    digest = hashlib.sha256()
    support: dict[str, list[tuple[str, tuple[str, ...]]] ] = {}
    total = 0
    for path in files:
        if not path.is_file():
            raise RelationRoleEvidenceLearningError(f"课程文件缺失: {path}")
        payload = path.read_bytes()
        digest.update(path.as_posix().encode("utf-8") + b"\0" + payload)
        for line_number, raw in enumerate(payload.splitlines(), 1):
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RelationRoleEvidenceLearningError(
                    f"课程 JSONL 非法: {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise RelationRoleEvidenceLearningError("role evidence record 非法")
            family, question, shape = _record_shape(record)
            if record["split"] not in {"train", "heldout"}:
                raise RelationRoleEvidenceLearningError("role evidence split 非法")
            if record["split"] != "train":
                continue
            support.setdefault(family, []).append((question, shape))
            total += 1
    learned = []
    for family, cases in sorted(support.items()):
        if len(cases) < _MIN_SUPPORT:
            continue
        openers = Counter(item[1][0] for item in cases)
        closers = Counter(item[1][1] for item in cases)
        separators = Counter(item[1][2] for item in cases)
        learned.append(LearnedRelationRoleFamily(
            family,
            tuple(sorted(openers.items())),
            tuple(sorted(closers.items())),
            tuple(sorted(separators.items())),
            len(cases),
        ))
    if not learned:
        raise RelationRoleEvidenceLearningError("无达到阈值的 role family")
    return LearnedRelationRoleEvidenceModel(tuple(learned), digest.hexdigest(), total)


def relation_role_evidence_sha256(
        model: LearnedRelationRoleEvidenceModel) -> str:
    payload = b"".join(int(value).to_bytes(8, "big")
                       for value in model.canonical_record())
    return hashlib.sha256(payload).hexdigest()


def project_qualified_relation_value(
        relation_model: LearnedRelationEvidenceModel,
        role_model: LearnedRelationRoleEvidenceModel,
        question: str,
        evidence_text: str,
        *,
        anchor_text: str | None = None,
        ) -> tuple[str, int, int] | None:
    """按公开列表 role 结构选择唯一 qualifier 匹配的原文 item。"""
    if (not isinstance(relation_model, LearnedRelationEvidenceModel)
            or not isinstance(role_model, LearnedRelationRoleEvidenceModel)):
        raise TypeError("relation model 类型错误")
    if not isinstance(question, str) or not question.strip() \
            or not isinstance(evidence_text, str) or not evidence_text.strip():
        raise RelationRoleEvidenceLearningError("projection 输入不能为空")
    family_name = relation_model.relation_family(question)
    if family_name is None:
        return None
    family = role_model.family(family_name)
    if family is None:
        return None
    anchor_terms = set(broad_qa_terms(anchor_text)) if anchor_text else set()
    query_terms = set(broad_qa_terms(question)) - anchor_terms
    if not query_terms:
        return None
    openers = tuple(value for value, _ in family.openers)
    closers = tuple(value for value, _ in family.closers)
    separators = tuple(value for value, _ in family.separators)
    candidates = []
    for opener in openers:
        for closer in closers:
            start = 0
            while True:
                open_index = evidence_text.find(opener, start)
                if open_index < 0:
                    break
                close_index = evidence_text.find(closer, open_index + len(opener))
                if close_index < 0:
                    break
                qualifier = evidence_text[open_index + len(opener):close_index]
                overlap = query_terms.intersection(broad_qa_terms(qualifier))
                if overlap:
                    left = max(
                        evidence_text.rfind(separator, 0, open_index)
                        for separator in separators)
                    item_start = left + 1
                    item = evidence_text[item_start:close_index + len(closer)].strip()
                    trim_left = len(evidence_text[item_start:close_index + len(closer)]) - len(
                        evidence_text[item_start:close_index + len(closer)].lstrip())
                    absolute_start = item_start + trim_left
                    candidates.append((len(overlap), len(item), -absolute_start,
                                       item, absolute_start,
                                       close_index + len(closer)))
                start = close_index + len(closer)
    candidates.sort(reverse=True)
    if not candidates or (len(candidates) > 1
                          and candidates[0][:2] == candidates[1][:2]):
        return None
    return candidates[0][3], candidates[0][4], candidates[0][5]


__all__ = [
    "LearnedRelationRoleEvidenceModel",
    "LearnedRelationRoleFamily",
    "RelationRoleEvidenceLearningError",
    "learn_relation_role_evidence_model",
    "project_qualified_relation_value",
    "relation_role_evidence_sha256",
]
