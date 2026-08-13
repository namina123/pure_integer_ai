"""加载公开 CC0 问式变量课程，分离答案槽位与实体限定。"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sysconfig

from opencc import OpenCC

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


REPOSITORY = Path(__file__).resolve().parents[3]
QUESTION_SLOT_RELATIVE_PATH = Path("data/ph2/broad_qa_question_slots_v1.json")
QUESTION_SLOT_PATH = REPOSITORY / QUESTION_SLOT_RELATIVE_PATH
QUESTION_SLOT_DISTRIBUTION_SUBDIRECTORY = Path("share/pure_integer_ai")
QUESTION_SLOT_SHA256 = (
    "932f5ed9b95c4501e64dc06ca769f9fb6b5a79b529f2e09b4a2c5b322c59f2ee")
_ANSWER_KINDS = (
    "CAUSE", "ENTITY", "LOCATION", "MANNER", "QUANTITY", "TIME", "TYPE")
_TO_SIMPLIFIED = OpenCC("t2s")


def _aligned_simplified_surface(value: str) -> str:
    """构造与原文等长的简体视图，使问式 span 保持原坐标。"""
    converted = []
    for character in value:
        simplified = _TO_SIMPLIFIED.convert(character)
        converted.append(simplified if len(simplified) == 1 else character)
    return "".join(converted)


def _is_contextual_slot(
        question: str, start: int, kind: str, surface: str) -> bool:
    """拒绝侵入关系谓词的歧义槽，同时保留独立因果问式。"""
    return not (
        kind == "CAUSE"
        and surface == "为什么"
        and start > 0
        and question[start - 1] in {"称", "稱"}
    )


# object-model: exception
class BroadQaQuestionSlotError(RuntimeError):
    """问式 artifact 缺失、漂移、重叠或无法规范解析。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaQuestionSlots:
    """保存问式变量 surface、答案类型及规范 artifact 身份。"""

    entries: tuple[tuple[str, tuple[str, ...]], ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        """核验答案类型全集、artifact 身份及 surface 唯一性。"""
        if (tuple(item[0] for item in self.entries) != _ANSWER_KINDS
                or len(self.artifact_sha256) != 64):
            raise BroadQaQuestionSlotError("broad QA 问式 inventory 漂移")
        surfaces = tuple(
            surface for _, values in self.entries for surface in values)
        if (not surfaces or len(set(surfaces)) != len(surfaces)
                or any(not value or value.strip() != value
                       for value in surfaces)):
            raise BroadQaQuestionSlotError("broad QA 问式 surface 非规范")

    @property
    def surfaces(self) -> tuple[str, ...]:
        """按最长优先返回公开课程中的全部问式变量表面。"""
        return tuple(sorted(
            (surface for _, values in self.entries for surface in values),
            key=lambda item: (-len(item), item),
        ))

    def _selected_slots(
            self, question: str,
            ) -> tuple[tuple[int, int, str], ...]:
        """按上下文选择最长且互不重叠的问式槽 span。"""
        aligned_question = _aligned_simplified_surface(question)
        candidates = []
        for kind, surfaces in self.entries:
            for surface in surfaces:
                start = aligned_question.find(surface)
                while start >= 0:
                    end = start + len(surface)
                    if _is_contextual_slot(
                            aligned_question, start, kind, surface):
                        candidates.append((start, end, kind))
                    start = aligned_question.find(surface, start + 1)
        candidates.sort(key=lambda item: (-(item[1] - item[0]), item))
        selected = []
        for item in candidates:
            if any(item[0] < prior[1] and prior[0] < item[1]
                   for prior in selected):
                continue
            selected.append(item)
        return tuple(sorted(selected))

    def strip_slots(self, question: str) -> str:
        """移除显式问式变量但保留实体、属性和其他限定。"""
        parts = []
        cursor = 0
        for start, end, _ in self._selected_slots(question):
            parts.extend((question[cursor:start], "\n"))
            cursor = end
        parts.append(question[cursor:])
        return "".join(parts)

    def answer_kinds(self, question: str) -> tuple[str, ...]:
        """返回问题实际命中的答案槽类型，不依赖任何知识页。"""
        observed = {item[2] for item in self._selected_slots(question)}
        return tuple(kind for kind, _ in self.entries if kind in observed)


def _candidate_paths() -> tuple[Path, ...]:
    """枚举 checkout 与标准 data scheme 内的 artifact 位置。"""
    roots = [REPOSITORY]
    current = sysconfig.get_path("data")
    if current:
        roots.append(
            Path(current) / QUESTION_SLOT_DISTRIBUTION_SUBDIRECTORY)
    values = []
    seen = set()
    for root in roots:
        candidate = (root / QUESTION_SLOT_RELATIVE_PATH).resolve()
        if candidate not in seen:
            seen.add(candidate)
            values.append(candidate)
    return tuple(values)


@lru_cache(maxsize=8)
def load_broad_qa_question_slots(
        path: str | Path | None = None,
        ) -> BroadQaQuestionSlots:
    """严格加载 canonical artifact，并拒绝字段或 SHA 漂移。"""
    source = (Path(path).resolve() if path is not None else next(
        (item for item in _candidate_paths() if item.is_file()),
        QUESTION_SLOT_PATH,
    ))
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise BroadQaQuestionSlotError("broad QA 问式 artifact 缺失") from error
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != QUESTION_SLOT_SHA256:
        raise BroadQaQuestionSlotError("broad QA 问式 artifact SHA 漂移")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaQuestionSlotError("broad QA 问式 artifact 非法") from error
    if canonical_json_bytes(value) + b"\n" != payload:
        raise BroadQaQuestionSlotError("broad QA 问式 artifact 非 canonical JSON")
    keys = {
        "artifact_kind", "entries", "format_version", "language",
        "license_id", "source_identity",
    }
    if (not isinstance(value, dict) or set(value) != keys
            or value["artifact_kind"] != "PH2_BROAD_QA_QUESTION_SLOTS_V1"
            or value["format_version"] != 1 or value["language"] != "zh"
            or value["license_id"] != "CC0-1.0"
            or value["source_identity"]
            != "AUTHORED_CC0_BROAD_QA_QUESTION_SLOTS_V1"
            or not isinstance(value["entries"], list)):
        raise BroadQaQuestionSlotError("broad QA 问式 artifact envelope 漂移")
    entries = []
    for item in value["entries"]:
        if (not isinstance(item, dict)
                or set(item) != {"answer_kind", "surfaces"}
                or not isinstance(item["answer_kind"], str)
                or not isinstance(item["surfaces"], list)):
            raise BroadQaQuestionSlotError("broad QA 问式 entry 漂移")
        entries.append((
            item["answer_kind"], tuple(str(value) for value in item["surfaces"])))
    return BroadQaQuestionSlots(tuple(entries), actual_sha256)


__all__ = [
    "BroadQaQuestionSlotError",
    "BroadQaQuestionSlots",
    "QUESTION_SLOT_SHA256",
    "load_broad_qa_question_slots",
]
