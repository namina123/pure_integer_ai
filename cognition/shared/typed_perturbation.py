"""H-02B typed 扰动的定向影响范围和共享 trace 构造边界。

本模块不解释角色、cue 或逻辑算子，只负责把调用方给出的完整一等对象、来源、
作用域和前后影响对象接入 H-02A ``PerturbationTrace``。对象位置映射只按完整
``ObjectIdentity`` 建立，不能用摘要、surface 或 Python 对象身份替代。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.perturbation import (
    PerturbationTrace,
    build_replacement_trace,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity


def _object_tuple(value, *, where: str) -> tuple[ObjectIdentity, ...]:
    """核验 typed 扰动边界只接收完整一等对象序列。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空 ObjectIdentity tuple")
    if any(not isinstance(item, ObjectIdentity) for item in value):
        raise TypeError(f"{where} 只能包含 ObjectIdentity")
    return value


def _pack_objects(values: tuple[ObjectIdentity, ...]) -> tuple[int, ...]:
    """以长度前缀串联完整对象身份，供 typed trace 稳定键使用。"""
    packed: list[int] = [len(values)]
    for value in values:
        key = value.stable_key()
        packed.extend((len(key), *key))
    return tuple(packed)


def _identity_mapping(
        original: tuple[ObjectIdentity, ...],
        transformed: tuple[ObjectIdentity, ...],
        ) -> tuple[int, ...]:
    """确定性匹配前后相同对象；新增对象使用 ``-1``，且每个输入位置至多消费一次。"""
    positions: dict[ObjectIdentity, list[int]] = {}
    for index, identity in enumerate(original):
        positions.setdefault(identity, []).append(index)
    used: set[int] = set()
    mapping: list[int] = []
    for output_index, identity in enumerate(transformed):
        available = tuple(
            index for index in positions.get(identity, ())
            if index not in used
        )
        if not available:
            mapping.append(-1)
            continue
        input_index = (
            output_index if output_index in available else available[0]
        )
        used.add(input_index)
        mapping.append(input_index)
    return tuple(mapping)


@dataclass(frozen=True)
class TypedPerturbationTrace:
    """在 H-02A trace 外保存 typed 变换两侧的定向语义影响对象。"""

    trace: PerturbationTrace
    impact_original: tuple[ObjectIdentity, ...]
    impact_transformed: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        """核验影响范围存在于对应侧，并至少触及一个真实变化位置。"""
        if not isinstance(self.trace, PerturbationTrace):
            raise TypeError("typed perturbation trace 类型错误")
        original = _object_tuple(
            self.impact_original,
            where="TypedPerturbationTrace.impact_original",
        )
        transformed = _object_tuple(
            self.impact_transformed,
            where="TypedPerturbationTrace.impact_transformed",
        )
        if len(set(original)) != len(original):
            raise ValueError("impact_original 不得重复")
        if len(set(transformed)) != len(transformed):
            raise ValueError("impact_transformed 不得重复")
        if any(item not in self.trace.original for item in original):
            raise ValueError("impact_original 必须来自 trace.original")
        if any(item not in self.trace.transformed for item in transformed):
            raise ValueError("impact_transformed 必须来自 trace.transformed")
        affected_original = {
            self.trace.original[index]
            for index in self.trace.affected_input_positions
        }
        affected_transformed = {
            self.trace.transformed[index]
            for index in self.trace.affected_output_positions
        }
        if not set(original).intersection(affected_original):
            raise ValueError("impact_original 未覆盖输入侧真实变化")
        if not set(transformed).intersection(affected_transformed):
            raise ValueError("impact_transformed 未覆盖输出侧真实变化")

    def stable_key(self) -> tuple[int, ...]:
        """返回完整 H-02A trace 与双侧定向影响对象的稳定键。"""
        trace_key = self.trace.stable_key()
        original = _pack_objects(self.impact_original)
        transformed = _pack_objects(self.impact_transformed)
        return (
            len(trace_key),
            *trace_key,
            len(original),
            *original,
            len(transformed),
            *transformed,
        )


def build_typed_replacement_trace(
        original: tuple[ObjectIdentity, ...],
        transformed: tuple[ObjectIdentity, ...], *,
        transform_key: tuple[int, ...],
        source: SourceRef,
        scope: ScopeIdentity,
        impact_original: tuple[ObjectIdentity, ...],
        impact_transformed: tuple[ObjectIdentity, ...],
        metadata_keys: tuple[tuple[int, ...], ...] = (),
        ) -> TypedPerturbationTrace:
    """按完整身份建立通用替换 trace，并附加调用方声明的定向影响范围。"""
    original = _object_tuple(
        original, where="build_typed_replacement_trace.original")
    transformed = _object_tuple(
        transformed, where="build_typed_replacement_trace.transformed")
    trace = build_replacement_trace(
        original,
        transformed,
        output_to_input=_identity_mapping(original, transformed),
        transform_key=transform_key,
        source=source,
        scope=scope,
        metadata_keys=metadata_keys,
    )
    return TypedPerturbationTrace(
        trace,
        impact_original,
        impact_transformed,
    )


__all__ = [
    "TypedPerturbationTrace",
    "build_typed_replacement_trace",
]
