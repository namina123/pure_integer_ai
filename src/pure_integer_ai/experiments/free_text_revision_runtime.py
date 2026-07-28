"""复用 MD-02 dependency index 的自由文本局部失效薄适配。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.attractor_state import AttractorDependency
from pure_integer_ai.cognition.shared.situation_state import (
    SituationDependencyIndex,
    SituationDependencyLink,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey


DERIVED_KINDS = ("CENTER", "CLAIM", "HIERARCHY")


class FreeTextRevisionRuntimeError(RuntimeError):
    """自由文本派生项、dependency 或局部失效收据不闭合。"""


@dataclass(frozen=True, order=True)
class FreeTextDerivedDependency:
    """一个 hierarchy/center/claim 派生身份及其 typed 重算依赖。"""

    derived_key: StableRecordKey
    derived_kind: str
    dependencies: tuple[AttractorDependency, ...]

    def __post_init__(self) -> None:
        """要求派生 kind 已冻结且依赖稳定、非空、无重复。"""
        if not isinstance(self.derived_key, StableRecordKey):
            raise TypeError("derived dependency key 类型错误")
        if self.derived_kind not in DERIVED_KINDS:
            raise FreeTextRevisionRuntimeError("derived dependency kind 未登记")
        if (not isinstance(self.dependencies, tuple) or not self.dependencies
                or any(not isinstance(item, AttractorDependency)
                       for item in self.dependencies)):
            raise TypeError("derived dependencies 类型错误")
        keys = tuple(item.stable_key() for item in self.dependencies)
        if keys != tuple(sorted(set(keys))):
            raise FreeTextRevisionRuntimeError("derived dependencies 未排序去重")


@dataclass(frozen=True)
class FreeTextRevisionInvalidationReceipt:
    """一次 revision 精确命中三类派生项并保留其余身份的零写收据。"""

    invalidated_keys: tuple[StableRecordKey, ...]
    preserved_keys: tuple[StableRecordKey, ...]
    hierarchy_keys: tuple[StableRecordKey, ...]
    center_keys: tuple[StableRecordKey, ...]
    claim_keys: tuple[StableRecordKey, ...]
    unaffected_bit_identical: int
    host_learning_write_count: int

    def __post_init__(self) -> None:
        """核验分类并集、失效/保留互斥和零写事实。"""
        for name in (
                "invalidated_keys", "preserved_keys", "hierarchy_keys",
                "center_keys", "claim_keys"):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, StableRecordKey) for item in values)
                    or values != tuple(sorted(set(values)))):
                raise FreeTextRevisionRuntimeError(f"{name} 未排序去重")
        classified = tuple(sorted({
            *self.hierarchy_keys, *self.center_keys, *self.claim_keys}))
        if classified != self.invalidated_keys:
            raise FreeTextRevisionRuntimeError("三类失效键与总集不闭合")
        if set(self.invalidated_keys).intersection(self.preserved_keys):
            raise FreeTextRevisionRuntimeError("invalidated/preserved 重叠")
        if self.unaffected_bit_identical != 1:
            raise FreeTextRevisionRuntimeError("未受影响派生身份必须 bit-identical")
        if self.host_learning_write_count != 0:
            raise FreeTextRevisionRuntimeError("revision invalidator 不得学习写")


class FreeTextRevisionInvalidator:
    """把 typed dependency 交给现有 SituationDependencyIndex 精确求交。"""

    def __init__(self, bindings: tuple[FreeTextDerivedDependency, ...]) -> None:
        """冻结派生项并构造唯一可重建 dependency index。"""
        if (not isinstance(bindings, tuple) or not bindings
                or any(not isinstance(item, FreeTextDerivedDependency)
                       for item in bindings)):
            raise TypeError("revision bindings 类型错误")
        keys = tuple(item.derived_key for item in bindings)
        if keys != tuple(sorted(set(keys))):
            raise FreeTextRevisionRuntimeError("revision derived keys 未排序去重")
        mapping: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
        for binding in bindings:
            for dependency in binding.dependencies:
                mapping.setdefault(dependency.stable_key(), set()).add(
                    binding.derived_key.components)
        self.bindings = bindings
        self.index = SituationDependencyIndex(tuple(
            SituationDependencyLink(key, tuple(sorted(mapping[key])))
            for key in sorted(mapping)
        ))

    def invalidate(
            self,
            changed_dependencies: tuple[AttractorDependency, ...],
            ) -> FreeTextRevisionInvalidationReceipt:
        """返回仅命中 dependency 的派生键，不改写原文、Evidence 或 WorkMemory。"""
        affected_values = self.index.affected(changed_dependencies)
        affected = tuple(sorted(StableRecordKey(key) for key in affected_values))
        by_key = {item.derived_key: item for item in self.bindings}
        preserved = tuple(sorted(set(by_key) - set(affected)))

        def keys_for(kind: str) -> tuple[StableRecordKey, ...]:
            """返回受影响集合中指定派生种类的稳定键。"""
            return tuple(sorted(
                key for key in affected if by_key[key].derived_kind == kind))

        return FreeTextRevisionInvalidationReceipt(
            affected,
            preserved,
            keys_for("HIERARCHY"),
            keys_for("CENTER"),
            keys_for("CLAIM"),
            1,
            0,
        )


__all__ = [
    "DERIVED_KINDS",
    "FreeTextDerivedDependency",
    "FreeTextRevisionInvalidationReceipt",
    "FreeTextRevisionInvalidator",
    "FreeTextRevisionRuntimeError",
]
