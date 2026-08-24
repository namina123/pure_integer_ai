"""DLG-RAW-16 表层组织课程的严格、可迁移数据合同。

本模块只定义公开课程切片的 record 形状与整数边界，不承担训练、生成或
默认终端路由。所有文本同时保留 Unicode scalar 与 UTF-8 ``u8`` 序列，
使非 Python 实现可以按同一记录重建语义和规范字节。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_core import DatasetContractError


SCHEMA_VERSION = 1
RECORD_KIND = "DLG_RAW16_SURFACE_ORGANIZATION_V1"
LICENSE_ID = "CC0-1.0"
DIALOGUE_ACTS = frozenset({"ANSWER", "CLARIFY", "UNKNOWN", "REPAIR"})
REGISTERS = frozenset({"neutral", "plain", "polite"})
SPLITS = frozenset({"course"})


class SurfaceOrganizationError(DatasetContractError):
    """表层组织课程记录、义务或隔离字段不满足冻结合同。"""


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SurfaceOrganizationError(f"{where} 必须是无首尾空白的非空字符串")
    return value


def _positive(value: Any, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise SurfaceOrganizationError(f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, where: str) -> int:
    if type(value) is not int or value < 0:
        raise SurfaceOrganizationError(f"{where} 必须是非负严格整数")
    return value


def _text_tuple(value: Any, where: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise SurfaceOrganizationError(f"{where} 必须是非空字符串列表")
    result = tuple(_text(item, f"{where}[]") for item in value)
    if len(set(result)) != len(result):
        raise SurfaceOrganizationError(f"{where} 不得重复")
    return result


@dataclass(frozen=True, slots=True)
class ClauseSlot:
    """一个有序的表层槽位；slot identity 不由文本猜测。"""

    slot_id: str
    role: str
    order: int
    required: int

    def __post_init__(self) -> None:
        _text(self.slot_id, "clause_slot.slot_id")
        _text(self.role, "clause_slot.role")
        _positive(self.order, "clause_slot.order")
        if self.required not in (0, 1) or type(self.required) is not int:
            raise SurfaceOrganizationError("clause_slot.required 必须是 0/1")


@dataclass(frozen=True, slots=True)
class SurfaceVariant:
    """一条可接受或应拒绝的表层实现。"""

    variant_id: str
    surface: str
    proposition_ids: tuple[str, ...]
    clause_order: tuple[str, ...]
    register: str
    violations: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.variant_id, "surface.variant_id")
        _text(self.surface, "surface.surface")
        _text_tuple(list(self.proposition_ids), "surface.proposition_ids", allow_empty=True)
        _text_tuple(list(self.clause_order), "surface.clause_order", allow_empty=True)
        if self.register not in REGISTERS:
            raise SurfaceOrganizationError("surface.register 非法")
        _text_tuple(list(self.violations), "surface.violations", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SurfaceOrganizationRecord:
    """经 schema 校验后的不可变课程 record。"""

    sample_id: str
    source_id: str
    context_id: str
    family_id: str
    template_family: str
    owner: str
    split: str
    dialogue_act: str
    proposition_id: str
    proposition_kind: str
    proposition_subject: str
    proposition_predicate: str
    proposition_object: str
    required_proposition_ids: tuple[str, ...]
    forbidden_proposition_ids: tuple[str, ...]
    clause_slots: tuple[ClauseSlot, ...]
    register: str
    min_chars: int
    max_chars: int
    accepted: tuple[SurfaceVariant, ...]
    rejected: tuple[SurfaceVariant, ...]

    def __post_init__(self) -> None:
        for name in ("sample_id", "source_id", "context_id", "family_id", "template_family", "owner",
                     "proposition_id", "proposition_kind", "proposition_subject",
                     "proposition_predicate", "proposition_object"):
            _text(getattr(self, name), f"record.{name}")
        if self.split not in SPLITS:
            raise SurfaceOrganizationError("record.split 非法")
        if self.dialogue_act not in DIALOGUE_ACTS:
            raise SurfaceOrganizationError("record.dialogue_act 非法")
        if self.register not in REGISTERS:
            raise SurfaceOrganizationError("record.register 非法")
        _positive(self.min_chars, "record.min_chars")
        _positive(self.max_chars, "record.max_chars")
        if self.max_chars < self.min_chars:
            raise SurfaceOrganizationError("长度预算倒置")
        if not self.clause_slots:
            raise SurfaceOrganizationError("clause_slots 不能为空")
        if not self.accepted or len(self.accepted) < 2:
            raise SurfaceOrganizationError("accepted 至少需要两条")
        if not self.rejected:
            raise SurfaceOrganizationError("rejected 至少需要一条")
        expected = self.required_proposition_ids
        if self.dialogue_act == "ANSWER" and expected != (self.proposition_id,):
            raise SurfaceOrganizationError("ANSWER 必须要求唯一主命题")
        if any(item.proposition_ids != expected for item in self.accepted):
            raise SurfaceOrganizationError("accepted 表层的语义命题漂移")
        order = tuple(item.slot_id for item in self.clause_slots)
        if any(item.clause_order != order for item in self.accepted):
            raise SurfaceOrganizationError("accepted 表层的有序槽位漂移")
        if any(not item.violations for item in self.rejected):
            raise SurfaceOrganizationError("rejected 必须声明至少一个错误")

    @property
    def canonical_integer_record(self) -> dict[str, Any]:
        """投影为跨语言可重建的整数/u8 记录；不依赖 Python 对象 identity。"""
        def surface(value: str) -> dict[str, Any]:
            return {"scalars": [ord(item) for item in value],
                    "utf8": list(value.encode("utf-8"))}
        def variant(item: SurfaceVariant) -> dict[str, Any]:
            return {"variant_id": surface(item.variant_id), "surface": surface(item.surface),
                    "proposition_ids": [surface(value) for value in item.proposition_ids],
                    "clause_order": [surface(value) for value in item.clause_order],
                    "register": item.register,
                    "violations": [surface(value) for value in item.violations]}
        return {
            "record_kind": RECORD_KIND, "schema_version": SCHEMA_VERSION,
            "sample_id": surface(self.sample_id), "source_id": surface(self.source_id),
            "context_id": surface(self.context_id), "family_id": surface(self.family_id),
            "template_family": surface(self.template_family), "owner": surface(self.owner),
            "split": self.split, "dialogue_act": self.dialogue_act,
            "semantic_proposition": {"id": surface(self.proposition_id), "kind": self.proposition_kind,
                            "subject": surface(self.proposition_subject),
                            "predicate": surface(self.proposition_predicate),
                            "object": surface(self.proposition_object)},
            "obligation": {"required_proposition_ids": [surface(value) for value in self.required_proposition_ids],
                            "forbidden_proposition_ids": [surface(value) for value in self.forbidden_proposition_ids],
                            "ordered_clause_slots": [
                                {"slot_id": surface(item.slot_id), "role": item.role,
                                 "order": item.order, "required": item.required}
                                for item in self.clause_slots]},
            "register": self.register, "length_budget": {"min_chars": self.min_chars,
                                                             "max_chars": self.max_chars},
            "accepted": [variant(item) for item in self.accepted],
            "rejected": [variant(item) for item in self.rejected],
        }


__all__ = ["ClauseSlot", "SurfaceVariant", "SurfaceOrganizationRecord",
           "SurfaceOrganizationError", "RECORD_KIND", "SCHEMA_VERSION", "LICENSE_ID"]
