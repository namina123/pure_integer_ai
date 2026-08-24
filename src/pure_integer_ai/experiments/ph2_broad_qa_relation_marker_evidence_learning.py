"""学习非列表关系证据中的 marker -> value 原文边界。"""
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


MARKER_EVIDENCE_PROTOCOL_V1 = 1
_MIN_SUPPORT = 2
_FIELDS = frozenset({
    "evidence", "family", "item_id", "license_id", "question", "roles",
    "source_identity", "split",
})
_NESTED = frozenset({"question_surface", "typed_intent"})
_EVIDENCE = frozenset({"evidence_surface"})
_ROLE = frozenset({"end", "role", "start"})


class RelationMarkerEvidenceLearningError(ValueError):
    """marker/value 课程或投影输入非法。"""


def _text(value: object, label: str) -> str:
    if (not isinstance(value, str) or not value or value.strip() != value
            or any(0xD800 <= ord(item) <= 0xDFFF for item in value)):
        raise RelationMarkerEvidenceLearningError(f"{label} 非法")
    return value


def _shape(record: object) -> tuple[str, str, str]:
    if not isinstance(record, dict) or set(record) != _FIELDS:
        raise RelationMarkerEvidenceLearningError("marker record 字段漂移")
    if record["license_id"] != "CC0-1.0":
        raise RelationMarkerEvidenceLearningError("marker 课程必须是 CC0")
    family = _text(record["family"], "family")
    _text(record["item_id"], "item_id")
    _text(record["source_identity"], "source_identity")
    question = record["question"]
    evidence = record["evidence"]
    if (not isinstance(question, dict) or set(question) != _NESTED
            or not isinstance(evidence, dict) or set(evidence) != _EVIDENCE):
        raise RelationMarkerEvidenceLearningError("marker nested fields 漂移")
    _text(question["question_surface"], "question_surface")
    surface = _text(evidence["evidence_surface"], "evidence_surface")
    roles = record["roles"]
    if (not isinstance(roles, list) or len(roles) != 2
            or any(not isinstance(item, dict) or set(item) != _ROLE
                   for item in roles)):
        raise RelationMarkerEvidenceLearningError("marker roles 必须为 marker/value")
    marker, value = roles
    if (marker["role"], value["role"]) != ("marker", "value"):
        raise RelationMarkerEvidenceLearningError("marker role 顺序非法")
    if (type(marker["start"]) is not int or type(marker["end"]) is not int
            or type(value["start"]) is not int or type(value["end"]) is not int
            or marker["start"] < 0 or marker["end"] <= marker["start"]
            or value["start"] != marker["end"]
            or value["end"] <= value["start"]
            or value["end"] > len(surface)):
        raise RelationMarkerEvidenceLearningError("marker/value span 非法")
    marker_text = surface[marker["start"]:marker["end"]]
    # The text after a value is learned as a tail.  It may be a sentence
    # terminator (``。``) or a continuation such as ``负责。``; neither is
    # treated as a fact or a fixed answer template.
    tail = surface[value["end"]:]
    return family, marker_text, tail


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LearnedRelationMarkerFamily:
    family: str
    markers: tuple[tuple[str, int], ...]
    tails: tuple[tuple[str, int], ...]
    marker_tails: tuple[tuple[str, str, int], ...]
    case_count: int

    def __post_init__(self) -> None:
        if (not self.family or not self.markers or not self.tails
                or not self.marker_tails
                or tuple(self.markers) != tuple(sorted(self.markers))
                or tuple(self.tails) != tuple(sorted(self.tails))
                or tuple(self.marker_tails) != tuple(sorted(self.marker_tails))
                or any(not value or type(count) is not int or count <= 0
                       for value, count in self.markers)
                or any(type(count) is not int or count <= 0
                       for value, count in self.tails)
                or any((not marker or type(tail) is not str
                        or type(count) is not int or count <= 0)
                       for marker, tail, count in self.marker_tails)
                or any(marker not in dict(self.markers)
                       or (marker, tail) not in {
                           (item[0], item[1]) for item in self.marker_tails}
                       for marker, tail, _ in self.marker_tails)
                or type(self.case_count) is not int
                or self.case_count < _MIN_SUPPORT):
            raise RelationMarkerEvidenceLearningError("marker family 非法")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LearnedRelationMarkerEvidenceModel:
    families: tuple[LearnedRelationMarkerFamily, ...]
    source_sha256: str
    case_count: int

    def __post_init__(self) -> None:
        if (not self.families
                or tuple(item.family for item in self.families)
                != tuple(sorted(item.family for item in self.families))
                or len({item.family for item in self.families}) != len(self.families)
                or any(not isinstance(item, LearnedRelationMarkerFamily)
                       for item in self.families)
                or len(self.source_sha256) != 64
                or any(item not in "0123456789abcdef"
                       for item in self.source_sha256)
                or type(self.case_count) is not int or self.case_count <= 0):
            raise RelationMarkerEvidenceLearningError("marker model 非法")

    def family(self, name: str) -> LearnedRelationMarkerFamily | None:
        return next((item for item in self.families if item.family == name), None)

    def canonical_record(self) -> tuple[int, ...]:
        result = [MARKER_EVIDENCE_PROTOCOL_V1, self.case_count,
                  len(self.source_sha256), *map(ord, self.source_sha256),
                  len(self.families)]
        for family in self.families:
            result.extend((len(family.family), *map(ord, family.family),
                           family.case_count))
            for values in (family.markers, family.tails):
                result.append(len(values))
                for value, count in values:
                    result.extend((len(value), *map(ord, value), count))
            result.append(len(family.marker_tails))
            for marker, tail, count in family.marker_tails:
                result.extend((len(marker), *map(ord, marker),
                               len(tail), *map(ord, tail), count))
        return tuple(result)


def learn_relation_marker_evidence_model(
        paths: Iterable[str | Path],
        ) -> LearnedRelationMarkerEvidenceModel:
    files = tuple(sorted(Path(item).resolve() for item in paths))
    if not files or len(files) != len(set(files)):
        raise RelationMarkerEvidenceLearningError("marker inventory 非法")
    digest = hashlib.sha256()
    support: dict[str, list[tuple[str, str]]] = {}
    total = 0
    for path in files:
        if not path.is_file():
            raise RelationMarkerEvidenceLearningError(f"课程缺失: {path}")
        payload = path.read_bytes()
        digest.update(path.as_posix().encode("utf-8") + b"\0" + payload)
        for line_number, raw in enumerate(payload.splitlines(), 1):
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RelationMarkerEvidenceLearningError(
                    f"课程 JSONL 非法: {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise RelationMarkerEvidenceLearningError("marker record 非法")
            family, marker, stop = _shape(record)
            if record["split"] not in {"train", "heldout"}:
                raise RelationMarkerEvidenceLearningError("marker split 非法")
            if record["split"] != "train":
                continue
            support.setdefault(family, []).append((marker, stop))
            total += 1
    families = []
    for family, cases in sorted(support.items()):
        if len(cases) < _MIN_SUPPORT:
            continue
        marker_tail_counts = Counter(cases)
        families.append(LearnedRelationMarkerFamily(
            family,
            tuple(sorted(Counter(item[0] for item in cases).items())),
            tuple(sorted(Counter(item[1] for item in cases).items())),
            tuple(sorted((marker, tail, count)
                         for (marker, tail), count in marker_tail_counts.items())),
            len(cases),
        ))
    if not families:
        raise RelationMarkerEvidenceLearningError("无达到阈值的 marker family")
    return LearnedRelationMarkerEvidenceModel(tuple(families), digest.hexdigest(), total)


def relation_marker_evidence_sha256(model: LearnedRelationMarkerEvidenceModel) -> str:
    if not isinstance(model, LearnedRelationMarkerEvidenceModel):
        raise TypeError("model 必须是 LearnedRelationMarkerEvidenceModel")
    payload = b"".join(int(value).to_bytes(8, "big")
                       for value in model.canonical_record())
    return hashlib.sha256(payload).hexdigest()


def project_marker_relation_value(
        relation_model: LearnedRelationEvidenceModel,
        marker_model: LearnedRelationMarkerEvidenceModel,
        question: str,
        evidence_text: str,
        *,
        anchor_text: str | None = None,
        ) -> tuple[str, int, int] | None:
    """按已学习 marker 与句末边界返回唯一来源 value。"""
    if (not isinstance(relation_model, LearnedRelationEvidenceModel)
            or not isinstance(marker_model, LearnedRelationMarkerEvidenceModel)):
        raise TypeError("marker projection model 类型错误")
    if not isinstance(question, str) or not question.strip() \
            or not isinstance(evidence_text, str) or not evidence_text.strip():
        raise RelationMarkerEvidenceLearningError("marker projection 输入为空")
    family_name = relation_model.relation_family(question)
    family = marker_model.family(family_name) if family_name else None
    if family is None:
        return None
    anchor = anchor_text or ""
    query_terms = set(broad_qa_terms(question))
    if anchor:
        query_terms.difference_update(broad_qa_terms(anchor))
    candidates = []
    for marker, tail, _ in family.marker_tails:
        start = 0
        while True:
            marker_start = evidence_text.find(marker, start)
            if marker_start < 0:
                break
            value_start = marker_start + len(marker)
            tail_start = (evidence_text.find(tail, value_start)
                          if tail else len(evidence_text))
            if tail_start < value_start:
                start = marker_start + len(marker)
                continue
            value_end = tail_start
            value = evidence_text[value_start:value_end].strip()
            if value:
                overlap = len(query_terms.intersection(
                    broad_qa_terms(value)))
                candidates.append((overlap, -len(value), -value_start,
                                   marker, tail, value, value_start, value_end))
            start = marker_start + len(marker)
    if not candidates:
        return None
    # Query overlap may disambiguate a source window only when exactly one
    # candidate carries it.  Otherwise the marker/boundary pair is not unique;
    # never choose by position or by an answer-shaped length heuristic.
    positive = [item for item in candidates if item[0] > 0]
    if len(positive) == 1:
        chosen = positive[0]
    elif positive or len(candidates) != 1:
        return None
    else:
        chosen = candidates[0]
    return chosen[5:]


__all__ = [
    "LearnedRelationMarkerEvidenceModel",
    "LearnedRelationMarkerFamily",
    "RelationMarkerEvidenceLearningError",
    "learn_relation_marker_evidence_model",
    "project_marker_relation_value",
    "relation_marker_evidence_sha256",
]
