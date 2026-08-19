"""DLG-05 未见关系结构的 typed 组合与训练分母边界。

本模块只验证关系结构组合的新颖度，不凭结构新颖度创造关系事实。held-out
关系必须由 TRAIN 已见的 predicate、construction 和 Role/filler-type 原语组成，
但完整组合不得出现在 TRAIN inventory；实际事实仍由 QuestionRequest 的来源和
Evidence 路径负责。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition


# object-model: exception
class ConversationRelationInputError(RuntimeError):
    """关系结构、TRAIN 分母或 held-out 组合不能严格闭合。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """把可变长度整数身份写入带长度边界的稳定键。"""
    result.extend((len(value), *value))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationRelationRoleSlot:
    """一个关系 Role 及其 filler 对象类型，不携带具体实体身份。"""

    role: ObjectIdentity
    filler_kind: int

    def __post_init__(self) -> None:
        """核验 Role 与严格正整数 filler kind。"""
        if (not isinstance(self.role, ObjectIdentity)
                or self.role.object_kind != OBJECT_ROLE):
            raise ConversationRelationInputError(
                "relation role slot 必须使用一等 Role")
        if type(self.filler_kind) is not int or self.filler_kind <= 0:
            raise ConversationRelationInputError(
                "relation role slot filler kind 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回 Role 与 filler type 的完整整数键。"""
        result: list[int] = [1]
        _pack(result, self.role.stable_key())
        result.append(self.filler_kind)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationRelationStructure:
    """从实际 BoundProposition 提取的关系行为结构。"""

    predicate: ObjectIdentity
    construction: ObjectIdentity
    slots: tuple[ConversationRelationRoleSlot, ...]

    def __post_init__(self) -> None:
        """拒绝无 Role、重复 Role 或非规范顺序的伪关系结构。"""
        if not isinstance(self.predicate, ObjectIdentity):
            raise TypeError("relation structure predicate 类型错误")
        if (not isinstance(self.construction, ObjectIdentity)
                or self.construction.object_kind != OBJECT_STRUCTURE_CONCEPT):
            raise ConversationRelationInputError(
                "relation structure construction 必须是 StructureConcept")
        if (not isinstance(self.slots, tuple) or len(self.slots) < 2
                or any(not isinstance(item, ConversationRelationRoleSlot)
                       for item in self.slots)):
            raise ConversationRelationInputError(
                "relation structure 至少需要两个 typed Role slot")
        ordered = tuple(sorted(
            self.slots, key=lambda item: item.role.stable_key()))
        if ordered != self.slots:
            raise ConversationRelationInputError(
                "relation structure Role slot 必须规范排序")
        if len({item.role for item in self.slots}) != len(self.slots):
            raise ConversationRelationInputError(
                "relation structure Role 不得重复")

    @classmethod
    def from_target(
            cls,
            target: BoundProposition,
            ) -> "ConversationRelationStructure":
        """从运行时命题读取 predicate、construction 与 Role/type 结构。"""
        if not isinstance(target, BoundProposition):
            raise TypeError("relation structure target 必须是 BoundProposition")
        slots = []
        for binding in target.bindings:
            filler_kind = (
                OBJECT_PROPOSITION
                if isinstance(binding.filler, BoundProposition)
                else binding.filler.object_kind
            )
            slots.append(ConversationRelationRoleSlot(
                binding.role, filler_kind))
        return cls(
            target.predicate,
            target.structure,
            tuple(sorted(slots, key=lambda item: item.role.stable_key())),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回不含具体实体 filler 的关系组合结构键。"""
        result: list[int] = [1]
        _pack(result, self.predicate.stable_key())
        _pack(result, self.construction.stable_key())
        result.append(len(self.slots))
        for slot in self.slots:
            _pack(result, slot.stable_key())
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationRelationTrainingInventory:
    """正式运行前冻结的 TRAIN 关系结构分母。"""

    structures: tuple[ConversationRelationStructure, ...]

    def __post_init__(self) -> None:
        """核验 TRAIN 结构非空、唯一且规范排序。"""
        if (not isinstance(self.structures, tuple) or not self.structures
                or any(not isinstance(item, ConversationRelationStructure)
                       for item in self.structures)):
            raise ConversationRelationInputError(
                "relation TRAIN inventory 必须包含 typed structures")
        ordered = tuple(sorted(
            self.structures, key=lambda item: item.stable_key()))
        if ordered != self.structures:
            raise ConversationRelationInputError(
                "relation TRAIN inventory 必须规范排序")
        if len({item.stable_key() for item in self.structures}) != len(
                self.structures):
            raise ConversationRelationInputError(
                "relation TRAIN structure 不得重复")

    def stable_key(self) -> tuple[int, ...]:
        """返回完整 TRAIN 关系结构分母键。"""
        result: list[int] = [1, len(self.structures)]
        for structure in self.structures:
            _pack(result, structure.stable_key())
        return tuple(result)


# object-model: runtime-owner; owns=immutable-train-structure-index
class ConversationUnseenRelationCompiler:
    """读取实际 QuestionRequest 并鉴权未见但原语已见的关系组合。"""

    def __init__(self, inventory: ConversationRelationTrainingInventory) -> None:
        """冻结 TRAIN 结构和各原语索引。"""
        if not isinstance(inventory, ConversationRelationTrainingInventory):
            raise TypeError("unseen relation compiler inventory 类型错误")
        self.inventory = inventory
        self._structures = {
            item.stable_key() for item in inventory.structures}
        self._predicates = {
            item.predicate for item in inventory.structures}
        self._constructions = {
            item.construction for item in inventory.structures}
        self._slots = {
            slot for item in inventory.structures for slot in item.slots}

    def compile(
            self,
            request: QuestionRequest,
            ) -> ConversationRelationStructure:
        """证明完整结构未见、全部承重原语已见，并返回实际结构。"""
        if not isinstance(request, QuestionRequest):
            raise TypeError("unseen relation compiler request 类型错误")
        structure = ConversationRelationStructure.from_target(request.target)
        if structure.stable_key() in self._structures:
            raise ConversationRelationInputError(
                "held-out relation 完整结构已出现在 TRAIN")
        if structure.predicate not in self._predicates:
            raise ConversationRelationInputError(
                "held-out relation predicate 未在 TRAIN 出现")
        if structure.construction not in self._constructions:
            raise ConversationRelationInputError(
                "held-out relation construction 未在 TRAIN 出现")
        missing = tuple(slot for slot in structure.slots
                        if slot not in self._slots)
        if missing:
            raise ConversationRelationInputError(
                "held-out relation Role/filler type 原语未在 TRAIN 出现")
        return structure


__all__ = [
    "ConversationRelationInputError",
    "ConversationRelationRoleSlot",
    "ConversationRelationStructure",
    "ConversationRelationTrainingInventory",
    "ConversationUnseenRelationCompiler",
]
