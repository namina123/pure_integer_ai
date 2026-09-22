"""G-02 完整命题到多个角色槽的无损绑定，不要求整句表示槽。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import OBJECT_ROLE, ObjectIdentity
from pure_integer_ai.cognition.shared.structure_order_consumer import StructureSlotValue
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition


def _pack(key: tuple[int, ...]) -> tuple[int, ...]:
    """保留完整可变长身份边界。"""
    return len(key), *key


@dataclass(frozen=True, slots=True)
class PropositionRoleSlotBinding:
    """命题的 Role+ordinal 与实际生成 slot/value 的显式连接。"""

    role: ObjectIdentity
    ordinal: int
    value: StructureSlotValue

    def __post_init__(self) -> None:
        """拒绝语言字符串、隐式序及非角色身份。"""
        if not isinstance(self.role, ObjectIdentity) or self.role.object_kind != OBJECT_ROLE:
            raise TypeError("生成角色绑定必须使用一等 Role")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("生成角色序必须是非负严格整数")
        if not isinstance(self.value, StructureSlotValue):
            raise TypeError("生成角色绑定必须引用实际 StructureSlotValue")

    def stable_key(self) -> tuple[int, ...]:
        """完整保留角色坐标、槽身份及填充者，不以内容摘要替代成员。"""
        return (*_pack(self.role.stable_key()), self.ordinal,
                *_pack(self.value.slot.stable_key()), *_pack(self.value.filler.stable_key()))


@dataclass(frozen=True, slots=True)
class PropositionRoleSlotFiller:
    """一个完整 BoundProposition 由全部角色槽共同承载，事实身份仍独立保留。"""

    candidate_key: tuple[int, ...]
    proposition: BoundProposition
    bindings: tuple[PropositionRoleSlotBinding, ...]

    def __post_init__(self) -> None:
        """要求每个原角色恰好对应一个真实槽，禁止缺角色、替换、重复或添加角色。"""
        if (type(self.candidate_key) is not tuple or not self.candidate_key
                or any(type(item) is not int for item in self.candidate_key)):
            raise TypeError("生成角色组候选键必须是非空严格整数 tuple")
        if not isinstance(self.proposition, BoundProposition):
            raise TypeError("生成角色组必须保留完整 BoundProposition")
        if (type(self.bindings) is not tuple or not self.bindings
                or any(not isinstance(item, PropositionRoleSlotBinding) for item in self.bindings)):
            raise TypeError("生成角色组必须具有非空角色连接")
        expected = {(item.role, item.ordinal): item.filler for item in self.proposition.bindings}
        actual = {(item.role, item.ordinal): item.value.filler for item in self.bindings}
        if (len(expected) != len(self.proposition.bindings)
                or len(actual) != len(self.bindings) or actual != expected
                or len({item.value.slot for item in self.bindings}) != len(self.bindings)):
            raise ValueError("生成角色组未完整且唯一地保留命题角色、序和填充者")
        object.__setattr__(self, "bindings", tuple(sorted(self.bindings, key=lambda item: item.stable_key())))

    @property
    def slot_values(self) -> tuple[StructureSlotValue, ...]:
        """提供句法消费者必须核验的全部实际槽值。"""
        return tuple(item.value for item in self.bindings)

    def stable_key(self) -> tuple[int, ...]:
        """冻结角色组 v1：候选、完整命题和全部显式角色连接。"""
        return (91517, 1, *_pack(self.candidate_key), *_pack(self.proposition.stable_key()),
                len(self.bindings), *(value for item in self.bindings for value in _pack(item.stable_key())))
