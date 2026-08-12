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
QUESTION_SLOT_RELATIVE_PATH = Path("data/ph2/broad_qa_question_slots_v1.json")
QUESTION_SLOT_PATH = REPOSITORY / QUESTION_SLOT_RELATIVE_PATH
QUESTION_SLOT_DISTRIBUTION_SUBDIRECTORY = Path("share/pure_integer_ai")
QUESTION_SLOT_SHA256 = (
    "932f5ed9b95c4501e64dc06ca769f9fb6b5a79b529f2e09b4a2c5b322c59f2ee")
_ANSWER_KINDS = (
    "CAUSE", "ENTITY", "LOCATION", "MANNER", "QUANTITY", "TIME", "TYPE")


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

    def strip_slots(self, question: str) -> str:
        """移除显式问式变量但保留实体、属性和其他限定。"""
        value = question
        for surface in self.surfaces:
            value = value.replace(surface, "\n")
        return value

    def answer_kinds(self, question: str) -> tuple[str, ...]:
        """返回问题实际命中的答案槽类型，不依赖任何知识页。"""
        return tuple(
            kind for kind, surfaces in self.entries
            if any(surface in question for surface in surfaces)
        )


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
