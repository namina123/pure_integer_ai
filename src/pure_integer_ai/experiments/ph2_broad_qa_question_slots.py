"""加载公开 CC0 问式变量课程，分离答案槽位与实体限定。"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sysconfig

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


REPOSITORY = Path(__file__).resolve().parents[3]
QUESTION_SLOT_RELATIVE_PATH = Path("data/ph2/broad_qa_question_slots_v2.json")
QUESTION_SLOT_PATH = REPOSITORY / QUESTION_SLOT_RELATIVE_PATH
QUESTION_SLOT_DISTRIBUTION_SUBDIRECTORY = Path("share/pure_integer_ai")
QUESTION_SLOT_SHA256 = (
    "f3783c7c38bf05f9e099edb80e9e0e1ff90aa5d1b7dd868ac4285e3e2c65c7ca")
_ANSWER_KINDS = (
    "CAUSE", "ENTITY", "LOCATION", "MANNER", "QUANTITY", "TIME", "TYPE")
def _aligned_simplified_surface(value: str) -> str:
    """兼容旧公开问式 artifact 的外部坐标适配。"""
    from pure_integer_ai.experiments.ph2_broad_qa_question_slots_compat import (
        aligned_surface,
    )
    return aligned_surface(value)


def _is_contextual_slot(
        question: str, start: int, kind: str, surface: str) -> bool:
    """兼容旧公开问式 artifact 的外部上下文适配。"""
    from pure_integer_ai.experiments.ph2_broad_qa_question_slots_compat import (
        contextual_slot_allowed,
    )
    return contextual_slot_allowed(question, start, kind, surface)


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
            self, question: str, surface_variant_provider=None,
            ) -> tuple[tuple[int, int, str], ...]:
        """按上下文选择最长且互不重叠的问式槽 span。"""
        if not isinstance(question, str):
            raise TypeError("question 必须是字符串")
        # 注入 provider 是运行时语言关系的权威来源。它可能表达长度变化或
        # 多对多表面，不能再把问题投影到固定长度外部脚本视图后复用坐标。
        # provider 缺省时才使用旧问式 artifact 的外部坐标适配。
        search_question = (
            question if surface_variant_provider is not None
            else _aligned_simplified_surface(question))
        candidates = []
        for kind, surfaces in self.entries:
            for surface in surfaces:
                variants = (surface,)
                if surface_variant_provider is not None:
                    if not callable(surface_variant_provider):
                        raise TypeError(
                            "surface_variant_provider 必须是可调用对象")
                    variants = tuple(dict.fromkeys((
                        surface, *surface_variant_provider(surface))))
                for candidate_surface in variants:
                    if (not isinstance(candidate_surface, str)
                            or not candidate_surface):
                        raise ValueError(
                            "surface_variant_provider 返回非法问式表面")
                    start = search_question.find(candidate_surface)
                    while start >= 0:
                        end = start + len(candidate_surface)
                        if _is_contextual_slot(
                                search_question, start, kind,
                                candidate_surface):
                            candidates.append((start, end, kind))
                        start = search_question.find(
                            candidate_surface, start + 1)
        candidates.sort(key=lambda item: (-(item[1] - item[0]), item))
        selected = []
        for item in candidates:
            if any(item[0] < prior[1] and prior[0] < item[1]
                   for prior in selected):
                continue
            selected.append(item)
        return tuple(sorted(selected))

    def strip_slots(self, question: str, surface_variant_provider=None) -> str:
        """移除显式问式变量但保留实体、属性和其他限定。"""
        parts = []
        cursor = 0
        for start, end, _ in self._selected_slots(
                question, surface_variant_provider):
            parts.extend((question[cursor:start], "\n"))
            cursor = end
        parts.append(question[cursor:])
        return "".join(parts)

    def answer_kinds(
            self, question: str, surface_variant_provider=None,
            ) -> tuple[str, ...]:
        """返回问题实际命中的答案槽类型，不依赖任何知识页。"""
        observed = {item[2] for item in self._selected_slots(
            question, surface_variant_provider)}
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
            or value["artifact_kind"] != "PH2_BROAD_QA_QUESTION_SLOTS_V2"
            or value["format_version"] != 2 or value["language"] != "zh"
            or value["license_id"] != "CC0-1.0"
            or value["source_identity"]
            != "AUTHORED_CC0_BROAD_QA_QUESTION_SLOTS_V2"
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
