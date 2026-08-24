"""公开多候选 marker/value 负例的独立协议。

负例只登记问题、来源证据和多个可能的原文 span，不登记 gold、正确候选或答案。
它可被独立审计器送入既有投影器；只要投影器返回确定值，负例审计就失败。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    LearnedRelationEvidenceModel,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_marker_evidence_learning import (
    LearnedRelationMarkerEvidenceModel,
    project_marker_relation_value,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)


MARKER_NEGATIVE_PROTOCOL_V1 = 1
MARKER_NEGATIVE_RECORD_KIND = "PH2_BROAD_QA_RELATION_MARKER_NEGATIVE_V1"
_LICENSE_ID = "CC0-1.0"
_FIELDS = frozenset({
    "candidates", "evidence_surface", "family", "item_id", "license_id",
    "protocol_version", "question_surface", "source_identity", "split",
    "typed_intent", "record_kind",
})
_CANDIDATE_FIELDS = frozenset({
    "marker_end", "marker_start", "value_end", "value_start",
})


class MarkerNegativeProtocolError(ValueError):
    """多候选负例的公开 schema、span 或审计不合法。"""


def _text(value: object, label: str) -> str:
    if (not isinstance(value, str) or not value or value.strip() != value
            or any(0xD800 <= ord(item) <= 0xDFFF for item in value)):
        raise MarkerNegativeProtocolError(f"{label} 非法")
    return value


@dataclass(frozen=True, slots=True)
class MarkerNegativeCandidate:
    marker_start: int
    marker_end: int
    value_start: int
    value_end: int

    def __post_init__(self) -> None:
        values = (self.marker_start, self.marker_end,
                  self.value_start, self.value_end)
        if (any(type(item) is not int for item in values)
                or self.marker_start < 0
                or self.marker_end <= self.marker_start
                or self.value_start != self.marker_end
                or self.value_end <= self.value_start):
            raise MarkerNegativeProtocolError("negative candidate span 非法")

    def canonical_record(self) -> tuple[int, ...]:
        return (self.marker_start, self.marker_end,
                self.value_start, self.value_end)


@dataclass(frozen=True, slots=True)
class MarkerNegativeCase:
    item_id: str
    family: str
    source_identity: str
    question_surface: str
    typed_intent: str
    evidence_surface: str
    candidates: tuple[MarkerNegativeCandidate, ...]

    def __post_init__(self) -> None:
        for value, label in (
                (self.item_id, "item_id"), (self.family, "family"),
                (self.source_identity, "source_identity"),
                (self.question_surface, "question_surface"),
                (self.typed_intent, "typed_intent"),
                (self.evidence_surface, "evidence_surface")):
            _text(value, label)
        if len(self.candidates) < 2:
            raise MarkerNegativeProtocolError("negative 必须至少含两个候选")
        if len({item.canonical_record() for item in self.candidates}) != len(
                self.candidates):
            raise MarkerNegativeProtocolError("negative candidate 不得重复")
        if any(item.value_end > len(self.evidence_surface)
               for item in self.candidates):
            raise MarkerNegativeProtocolError("negative candidate 超出 evidence")

    def canonical_record(self) -> tuple[int, ...]:
        values = [MARKER_NEGATIVE_PROTOCOL_V1]
        for text in (self.item_id, self.family, self.source_identity,
                     self.question_surface, self.typed_intent,
                     self.evidence_surface):
            values.extend((len(text), *map(ord, text)))
        values.append(len(self.candidates))
        for candidate in self.candidates:
            values.extend(candidate.canonical_record())
        return tuple(values)


def parse_marker_negative_record(value: object) -> MarkerNegativeCase:
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise MarkerNegativeProtocolError("negative record 字段集合漂移")
    if (value["record_kind"] != MARKER_NEGATIVE_RECORD_KIND
            or value["protocol_version"] != MARKER_NEGATIVE_PROTOCOL_V1
            or value["license_id"] != _LICENSE_ID
            or value["split"] != "negative"):
        raise MarkerNegativeProtocolError("negative record 身份或 split 非法")
    candidates = value["candidates"]
    if not isinstance(candidates, list):
        raise MarkerNegativeProtocolError("negative candidates 必须是列表")
    parsed = []
    for index, item in enumerate(candidates):
        if not isinstance(item, dict) or set(item) != _CANDIDATE_FIELDS:
            raise MarkerNegativeProtocolError(
                f"negative candidate[{index}] 字段集合漂移")
        parsed.append(MarkerNegativeCandidate(
            item["marker_start"], item["marker_end"],
            item["value_start"], item["value_end"],
        ))
    return MarkerNegativeCase(
        _text(value["item_id"], "item_id"), _text(value["family"], "family"),
        _text(value["source_identity"], "source_identity"),
        _text(value["question_surface"], "question_surface"),
        _text(value["typed_intent"], "typed_intent"),
        _text(value["evidence_surface"], "evidence_surface"),
        tuple(parsed),
    )


def load_marker_negative_jsonl(
        paths: Iterable[str | Path],
        ) -> tuple[MarkerNegativeCase, ...]:
    files = tuple(sorted(Path(item).resolve() for item in paths))
    if not files or len(files) != len(set(files)):
        raise MarkerNegativeProtocolError("negative inventory 非法")
    result = []
    for path in files:
        if not path.is_file():
            raise MarkerNegativeProtocolError(f"negative 课程缺失: {path}")
        payload = path.read_bytes()
        for line_number, line in enumerate(payload.splitlines(), 1):
            if not line:
                raise MarkerNegativeProtocolError(
                    f"negative JSONL 空行: {path}:{line_number}")
            try:
                value = parse_canonical_json_bytes(line, require_object=True)
            except (TypeError, ValueError) as error:
                raise MarkerNegativeProtocolError(
                    f"negative JSONL 非规范: {path}:{line_number}") from error
            if canonical_json_line(value).rstrip(b"\n") != line:
                raise MarkerNegativeProtocolError(
                    f"negative JSONL 非规范: {path}:{line_number}")
            result.append(parse_marker_negative_record(value))
    if len({item.item_id for item in result}) != len(result):
        raise MarkerNegativeProtocolError("negative item_id 重复")
    return tuple(result)


def marker_negative_source_sha256(
        paths: Iterable[str | Path],
        ) -> str:
    files = tuple(sorted(Path(item).resolve() for item in paths))
    digest = hashlib.sha256()
    for path in files:
        payload = path.read_bytes()
        digest.update(path.as_posix().encode("utf-8") + b"\0" + payload)
    return digest.hexdigest()


def audit_marker_negative_projection(
        cases: Iterable[MarkerNegativeCase],
        relation_model: LearnedRelationEvidenceModel,
        marker_model: LearnedRelationMarkerEvidenceModel,
        ) -> tuple[int, ...]:
    values = tuple(cases)
    if not all(isinstance(item, MarkerNegativeCase) for item in values):
        raise TypeError("negative cases 类型错误")
    for case in values:
        if project_marker_relation_value(
                relation_model, marker_model, case.question_surface,
                case.evidence_surface) is not None:
            raise MarkerNegativeProtocolError(
                f"negative projection unexpectedly selected: {case.item_id}")
    return (MARKER_NEGATIVE_PROTOCOL_V1, len(values),
            *sum((item.canonical_record() for item in values), ()))


__all__ = [
    "MARKER_NEGATIVE_PROTOCOL_V1", "MARKER_NEGATIVE_RECORD_KIND",
    "MarkerNegativeCandidate", "MarkerNegativeCase",
    "MarkerNegativeProtocolError", "audit_marker_negative_projection",
    "load_marker_negative_jsonl", "marker_negative_source_sha256",
    "parse_marker_negative_record",
]
