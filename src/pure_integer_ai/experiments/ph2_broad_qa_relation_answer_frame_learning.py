"""从公开 relation frame 课程学习 source-bound 完整句组织。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable


RELATION_ANSWER_FRAME_PROTOCOL_V1 = 1
_MIN_SUPPORT = 2
_FIELDS = frozenset({"family", "frame", "item_id", "license_id", "roles",
                     "source_identity", "split"})
_FRAME_FIELDS = frozenset({"surface"})
_ROLE_FIELDS = frozenset({"end", "role", "start"})


class RelationAnswerFrameLearningError(ValueError):
    """公开 relation answer frame 课程或投影输入非法。"""


def _text(value: object, label: str) -> str:
    if (not isinstance(value, str) or not value or value.strip() != value
            or any(0xD800 <= ord(item) <= 0xDFFF for item in value)):
        raise RelationAnswerFrameLearningError(f"{label} 非法")
    return value


def _shape(record: object) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    if not isinstance(record, dict) or set(record) != _FIELDS:
        raise RelationAnswerFrameLearningError("relation frame 字段漂移")
    if record["license_id"] != "CC0-1.0":
        raise RelationAnswerFrameLearningError("relation frame 必须是 CC0")
    family = _text(record["family"], "family")
    _text(record["item_id"], "item_id")
    source = _text(record["source_identity"], "source_identity")
    frame = record["frame"]
    if not isinstance(frame, dict) or set(frame) != _FRAME_FIELDS:
        raise RelationAnswerFrameLearningError("frame nested fields 漂移")
    surface = _text(frame["surface"], "frame.surface")
    roles = record["roles"]
    if (not isinstance(roles, list) or not roles
            or any(not isinstance(item, dict) or set(item) != _ROLE_FIELDS
                   for item in roles)):
        raise RelationAnswerFrameLearningError("relation frame roles 非法")
    spans: list[tuple[int, int, str]] = []
    for item in roles:
        start, end, role = item["start"], item["end"], item["role"]
        if (type(start) is not int or type(end) is not int
                or start < 0 or end <= start or end > len(surface)):
            raise RelationAnswerFrameLearningError("relation frame span 非法")
        spans.append((start, end, _text(role, "role")))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise RelationAnswerFrameLearningError("relation frame span 重叠")
    if tuple(item[2] for item in spans) != ("subject", "value"):
        raise RelationAnswerFrameLearningError("relation frame 必须是 subject/value")
    gaps = (surface[:spans[0][0]], surface[spans[0][1]:spans[1][0]],
            surface[spans[1][1]:])
    role_values = tuple(surface[start:end] for start, end, _ in spans)
    return family, source, gaps, role_values


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LearnedRelationAnswerFrame:
    family: str
    gaps: tuple[str, str, str]
    support_sources: tuple[str, ...]
    case_count: int

    def __post_init__(self) -> None:
        if (not self.family or len(self.gaps) != 3
                or len(self.support_sources) < _MIN_SUPPORT
                or tuple(sorted(set(self.support_sources))) != self.support_sources
                or type(self.case_count) is not int
                or self.case_count < _MIN_SUPPORT):
            raise RelationAnswerFrameLearningError("learned relation frame 非法")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class LearnedRelationAnswerFrameModel:
    frames: tuple[LearnedRelationAnswerFrame, ...]
    source_sha256: str
    case_count: int

    def __post_init__(self) -> None:
        if (not self.frames
                or tuple(item.family for item in self.frames)
                != tuple(sorted(item.family for item in self.frames))
                or len({item.family for item in self.frames}) != len(self.frames)
                or len(self.source_sha256) != 64
                or any(item not in "0123456789abcdef" for item in self.source_sha256)
                or type(self.case_count) is not int or self.case_count <= 0):
            raise RelationAnswerFrameLearningError("relation frame model 非法")

    def frame(self, family: str | None) -> LearnedRelationAnswerFrame | None:
        if not isinstance(family, str):
            return None
        return next((item for item in self.frames if item.family == family), None)

    def canonical_record(self) -> tuple[int, ...]:
        values = [RELATION_ANSWER_FRAME_PROTOCOL_V1, self.case_count,
                  len(self.source_sha256), *map(ord, self.source_sha256),
                  len(self.frames)]
        for frame in self.frames:
            values.extend((len(frame.family), *map(ord, frame.family),
                           frame.case_count, len(frame.support_sources)))
            for source in frame.support_sources:
                values.extend((len(source), *map(ord, source)))
            for gap in frame.gaps:
                values.extend((len(gap), *map(ord, gap)))
        return tuple(values)


def learn_relation_answer_frame_model(
        paths: Iterable[str | Path],
        ) -> LearnedRelationAnswerFrameModel:
    files = tuple(sorted(Path(item).resolve() for item in paths))
    if not files or len(files) != len(set(files)):
        raise RelationAnswerFrameLearningError("relation frame inventory 非法")
    digest = hashlib.sha256()
    support: dict[str, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
    total = 0
    for path in files:
        if not path.is_file():
            raise RelationAnswerFrameLearningError(f"课程缺失: {path}")
        payload = path.read_bytes()
        digest.update(path.as_posix().encode("utf-8") + b"\0" + payload)
        for line_number, raw in enumerate(payload.splitlines(), 1):
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RelationAnswerFrameLearningError(
                    f"课程 JSONL 非法: {path}:{line_number}") from error
            family, source, gaps, role_values = _shape(record)
            split = record["split"]
            if split not in {"train", "heldout"}:
                raise RelationAnswerFrameLearningError("relation frame split 非法")
            if split == "train":
                support[family].append((source, gaps))
                total += 1
    frames = []
    for family, cases in sorted(support.items()):
        sources = tuple(sorted({item[0] for item in cases}))
        gap_counts: dict[tuple[str, ...], int] = {}
        for _, gaps in cases:
            gap_counts[gaps] = gap_counts.get(gaps, 0) + 1
        if len(sources) < _MIN_SUPPORT or len(gap_counts) != 1:
            continue
        frames.append(LearnedRelationAnswerFrame(
            family, next(iter(gap_counts)), sources, len(cases)))
    if not frames:
        raise RelationAnswerFrameLearningError("无跨来源唯一 relation frame")
    return LearnedRelationAnswerFrameModel(tuple(frames), digest.hexdigest(), total)


def relation_answer_frame_sha256(model: LearnedRelationAnswerFrameModel) -> str:
    if not isinstance(model, LearnedRelationAnswerFrameModel):
        raise TypeError("model 必须是 LearnedRelationAnswerFrameModel")
    payload = b"".join(int(value).to_bytes(8, "big")
                       for value in model.canonical_record())
    return hashlib.sha256(payload).hexdigest()


def render_relation_answer_frame(
        model: LearnedRelationAnswerFrameModel,
        family: str | None,
        subject: str,
        value: str,
        ) -> str | None:
    """用来源标题和已投影来源值渲染唯一、已学习的完整句。"""
    if not isinstance(model, LearnedRelationAnswerFrameModel):
        raise TypeError("model 类型错误")
    subject = _text(subject, "subject")
    value = _text(value, "value")
    frame = model.frame(family)
    if frame is None:
        return None
    surface = frame.gaps[0] + subject + frame.gaps[1] + value + frame.gaps[2]
    return surface if surface.strip() else None


__all__ = [
    "LearnedRelationAnswerFrame", "LearnedRelationAnswerFrameModel",
    "RelationAnswerFrameLearningError", "learn_relation_answer_frame_model",
    "relation_answer_frame_sha256", "render_relation_answer_frame",
]
