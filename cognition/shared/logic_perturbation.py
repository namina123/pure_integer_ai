"""H-02B 的 R-08 否定、量化和嵌套作用域 typed 扰动适配器。

各适配器只沿调用方注入的 ``LogicOperatorCandidateSpec``、开放 Role 槽和
``BoundProposition`` 路径核验变换，不检查 handler 类名，不内置 NOT/EXISTS/
FORALL 等固定算子编号，也不把结构存在本身当作语义变化证据。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_OCCURRENCE,
    OBJECT_SPAN,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_candidate import (
    LogicOperatorCandidateSpec,
)
from pure_integer_ai.cognition.shared.logic_executor import OperatorSlot
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
)
from pure_integer_ai.cognition.shared.typed_perturbation import (
    TypedPerturbationTrace,
    build_typed_replacement_trace,
)


def _require_scope(source: SourceRef, scope: ScopeIdentity) -> None:
    """核验逻辑扰动使用同一来源化运行 scope。"""
    if not isinstance(scope, ScopeIdentity):
        raise TypeError("logic perturbation scope 类型错误")
    if scope.source != source:
        raise ValueError("logic perturbation scope 必须绑定命题来源")


def _anchor_source(
        identity: ObjectIdentity, *, source_key_size: int,
        ) -> SourceRef:
    """按调用方来源键长度从 Occurrence/Span 身份前缀恢复 SourceRef。"""
    if identity.object_kind not in {OBJECT_OCCURRENCE, OBJECT_SPAN}:
        raise ValueError("bound proposition source_anchor 类型错误")
    return SourceRef.from_stable_key(identity.components[:source_key_size])


def _require_bound_source(
        proposition: BoundProposition, source: SourceRef,
        ) -> None:
    """递归核验 bound 命题、context、Binder、Variable 和嵌套子命题来源。"""
    if not isinstance(proposition, BoundProposition):
        raise TypeError("logic perturbation 需要 BoundProposition")
    if semantic_source(proposition.template) != source:
        raise ValueError("bound Proposition 与扰动来源不一致")
    if semantic_source(proposition.context) != source:
        raise ValueError("bound ContextScope 与扰动来源不一致")
    if _anchor_source(
            proposition.source_anchor,
            source_key_size=len(source.stable_key())) != source:
        raise ValueError("bound source_anchor 与扰动来源不一致")
    if any(semantic_source(item) != source
           for item in proposition.introduced_binders):
        raise ValueError("bound Binder 与扰动来源不一致")
    if any(semantic_source(item) != source
           for item in proposition.applied_variables):
        raise ValueError("bound Variable 与扰动来源不一致")
    for binding in proposition.bindings:
        if isinstance(binding.filler, BoundProposition):
            _require_bound_source(binding.filler, source)


def _slot_metadata(
        binding: BoundRoleBinding, *, depth: int, ordinal: int,
        ) -> tuple[int, ...]:
    """保存递归深度、局部序号和完整 Role/ordinal，不从 tuple 位置猜槽义。"""
    role_key = binding.role.stable_key()
    return (
        depth,
        ordinal,
        binding.ordinal,
        len(role_key),
        *role_key,
    )


def _bound_units(
        proposition: BoundProposition, *, depth: int = 0,
        ) -> tuple[tuple[ObjectIdentity, ...], tuple[tuple[int, ...], ...]]:
    """递归展开 bound view 中全部一等对象，并另存每个 RoleBinding 的槽位坐标。"""
    units: list[ObjectIdentity] = [
        proposition.template,
        proposition.instruction,
        proposition.predicate,
        proposition.structure,
        proposition.source_anchor,
        proposition.context,
        *proposition.introduced_binders,
    ]
    metadata: list[tuple[int, ...]] = []
    for ordinal, binding in enumerate(proposition.bindings):
        metadata.append(_slot_metadata(
            binding, depth=depth, ordinal=ordinal))
        units.append(binding.role)
        if isinstance(binding.filler, ObjectIdentity):
            units.append(binding.filler)
        else:
            nested_units, nested_metadata = _bound_units(
                binding.filler, depth=depth + 1)
            units.extend(nested_units)
            metadata.extend(nested_metadata)
    units.extend(proposition.applied_variables)
    return tuple(units), tuple(metadata)


def _spec_units(
        spec: LogicOperatorCandidateSpec,
        ) -> tuple[tuple[ObjectIdentity, ...], tuple[tuple[int, ...], ...]]:
    """展开 R-08 候选、结构、指令和全部开放槽位对象与 ordinal。"""
    units: list[ObjectIdentity] = [
        spec.candidate,
        spec.definition.structure,
        spec.definition.instruction,
    ]
    metadata: list[tuple[int, ...]] = [spec.competition_key]
    for ordinal, slot in enumerate(spec.definition.slots):
        units.append(slot.role)
        role_key = slot.role.stable_key()
        metadata.append((
            ordinal,
            slot.ordinal,
            len(role_key),
            *role_key,
        ))
    return tuple(units), tuple(metadata)


@dataclass(frozen=True)
class LogicScopeLayer:
    """一个 R-08 operator candidate 及其显式子命题槽。"""

    spec: LogicOperatorCandidateSpec
    child_slot: OperatorSlot

    def __post_init__(self) -> None:
        """核验子命题槽属于候选定义，禁止调用方旁路其 typed slot 协议。"""
        if not isinstance(self.spec, LogicOperatorCandidateSpec):
            raise TypeError("LogicScopeLayer.spec 类型错误")
        if not isinstance(self.child_slot, OperatorSlot):
            raise TypeError("LogicScopeLayer.child_slot 类型错误")
        if self.child_slot not in self.spec.definition.slots:
            raise ValueError("LogicScopeLayer 子槽不属于 operator candidate")

    def stable_key(self) -> tuple[int, ...]:
        """返回候选、结构、指令和子槽的 handler-free 完整键。"""
        candidate = self.spec.candidate.stable_key()
        definition = self.spec.definition.stable_key()
        slot = self.child_slot.stable_key()
        return (
            len(candidate),
            *candidate,
            len(definition),
            *definition,
            len(slot),
            *slot,
        )


def _layer_child(
        proposition: BoundProposition,
        layer: LogicScopeLayer,
        ) -> BoundProposition:
    """核验 bound 根采用该候选结构/指令，并返回显式 child slot 子命题。"""
    if proposition.structure != layer.spec.definition.structure:
        raise ValueError("bound structure 与 operator candidate 不一致")
    if proposition.instruction != layer.spec.definition.instruction:
        raise ValueError("bound instruction 与 operator candidate 不一致")
    matches = tuple(
        binding for binding in proposition.bindings
        if (binding.role == layer.child_slot.role
            and binding.ordinal == layer.child_slot.ordinal)
    )
    if len(matches) != 1:
        raise ValueError("operator child slot 必须恰有一个 bound binding")
    child = matches[0].filler
    if not isinstance(child, BoundProposition):
        raise ValueError("operator child slot 必须绑定嵌套 BoundProposition")
    return child


def _path_leaf(
        root: BoundProposition,
        path: tuple[LogicScopeLayer, ...],
        ) -> BoundProposition:
    """沿调用方显式 scope path 逐层核验并返回共同叶命题。"""
    current = root
    for layer in path:
        current = _layer_child(current, layer)
    return current


def _path_units(
        path: tuple[LogicScopeLayer, ...],
        ) -> tuple[tuple[ObjectIdentity, ...], tuple[tuple[int, ...], ...]]:
    """按作用域层序展开每个 operator candidate 及其完整 typed 定义。"""
    units: list[ObjectIdentity] = []
    metadata: list[tuple[int, ...]] = []
    for depth, layer in enumerate(path):
        layer_units, layer_metadata = _spec_units(layer.spec)
        units.extend(layer_units)
        metadata.append((depth, *layer.stable_key()))
        metadata.extend((depth, *item) for item in layer_metadata)
    return tuple(units), tuple(metadata)


@dataclass(frozen=True)
class NegationPerturbationAdapter:
    """把调用方声明的一层 unary 逻辑包裹变换作为否定维度 trace。"""

    transform_key: tuple[int, ...]

    def build(
            self,
            original: BoundProposition,
            transformed: BoundProposition,
            *,
            layer: LogicScopeLayer,
            source: SourceRef,
            scope: ScopeIdentity,
            ) -> TypedPerturbationTrace:
        """核验 transformed 经指定 child slot 精确包裹 original，并保留候选身份。"""
        if not isinstance(source, SourceRef):
            raise TypeError("negation perturbation source 类型错误")
        _require_scope(source, scope)
        _require_bound_source(original, source)
        _require_bound_source(transformed, source)
        child = _layer_child(transformed, layer)
        if child != original:
            raise ValueError("negation transformed child 必须是完整 original bound view")

        original_units, original_metadata = _bound_units(original)
        spec_units, spec_metadata = _spec_units(layer.spec)
        transformed_units, transformed_metadata = _bound_units(transformed)
        return build_typed_replacement_trace(
            original_units,
            (*spec_units, *transformed_units),
            transform_key=self.transform_key,
            source=source,
            scope=scope,
            impact_original=(original.template,),
            impact_transformed=(
                layer.spec.candidate,
                transformed.template,
            ),
            metadata_keys=(
                *original_metadata,
                *spec_metadata,
                *transformed_metadata,
            ),
        )


@dataclass(frozen=True)
class QuantifierPerturbationAdapter:
    """比较共享 Binder 与 body 的两个调用方注入量化 operator 应用。"""

    transform_key: tuple[int, ...]

    def build(
            self,
            original: BoundProposition,
            transformed: BoundProposition,
            *,
            original_layer: LogicScopeLayer,
            transformed_layer: LogicScopeLayer,
            source: SourceRef,
            scope: ScopeIdentity,
            ) -> TypedPerturbationTrace:
        """核验量化根只替换 operator，Binder 与完整 body 保持相同。"""
        if not isinstance(source, SourceRef):
            raise TypeError("quantifier perturbation source 类型错误")
        _require_scope(source, scope)
        _require_bound_source(original, source)
        _require_bound_source(transformed, source)
        if (not original.introduced_binders
                or original.introduced_binders
                != transformed.introduced_binders):
            raise ValueError("quantifier 变换必须保留同一非空 Binder 声明")
        original_child = _layer_child(original, original_layer)
        transformed_child = _layer_child(transformed, transformed_layer)
        if original_child != transformed_child:
            raise ValueError("quantifier 变换必须保留完整 body bound view")
        if (original_layer.spec.candidate
                == transformed_layer.spec.candidate):
            raise ValueError("quantifier 变换必须使用不同 operator candidate")
        if (original_layer.spec.definition.stable_key()
                == transformed_layer.spec.definition.stable_key()):
            raise ValueError("不同候选但 operator 定义相同不构成量化语义变换")

        original_spec, original_spec_metadata = _spec_units(
            original_layer.spec)
        transformed_spec, transformed_spec_metadata = _spec_units(
            transformed_layer.spec)
        original_units, original_metadata = _bound_units(original)
        transformed_units, transformed_metadata = _bound_units(transformed)
        binder_metadata = tuple(
            binder.stable_key() for binder in original.introduced_binders)
        return build_typed_replacement_trace(
            (*original_spec, *original_units),
            (*transformed_spec, *transformed_units),
            transform_key=self.transform_key,
            source=source,
            scope=scope,
            impact_original=(
                original_layer.spec.candidate,
                original.template,
            ),
            impact_transformed=(
                transformed_layer.spec.candidate,
                transformed.template,
            ),
            metadata_keys=(
                *binder_metadata,
                *original_spec_metadata,
                *original_metadata,
                *transformed_spec_metadata,
                *transformed_metadata,
            ),
        )


@dataclass(frozen=True)
class ScopeFlipPerturbationAdapter:
    """把同一组 R-08 operator 的嵌套层序翻转接入 H-02A。"""

    transform_key: tuple[int, ...]

    def build(
            self,
            original: BoundProposition,
            transformed: BoundProposition,
            *,
            original_path: tuple[LogicScopeLayer, ...],
            transformed_path: tuple[LogicScopeLayer, ...],
            source: SourceRef,
            scope: ScopeIdentity,
            ) -> TypedPerturbationTrace:
        """核验相同候选集合以不同层序包裹同一叶命题，且语义根对象真实改变。"""
        if not isinstance(source, SourceRef):
            raise TypeError("scope flip source 类型错误")
        _require_scope(source, scope)
        _require_bound_source(original, source)
        _require_bound_source(transformed, source)
        for path, label in (
                (original_path, "original_path"),
                (transformed_path, "transformed_path")):
            if (not isinstance(path, tuple) or len(path) < 2
                    or any(not isinstance(item, LogicScopeLayer)
                           for item in path)):
                raise TypeError(f"{label} 必须含至少两个 LogicScopeLayer")
            candidates = tuple(item.spec.candidate for item in path)
            if len(set(candidates)) != len(candidates):
                raise ValueError(f"{label} 不得重复 operator candidate")

        original_by_candidate = {
            item.spec.candidate: item for item in original_path
        }
        transformed_by_candidate = {
            item.spec.candidate: item for item in transformed_path
        }
        if set(original_by_candidate) != set(transformed_by_candidate):
            raise ValueError("scope flip 前后必须使用同一 operator candidate 集")
        if any(
                original_by_candidate[candidate] !=
                transformed_by_candidate[candidate]
                for candidate in original_by_candidate):
            raise ValueError("scope flip 不得替换候选定义或 child slot")
        original_order = tuple(item.spec.candidate for item in original_path)
        transformed_order = tuple(
            item.spec.candidate for item in transformed_path)
        if original_order == transformed_order:
            raise ValueError("scope flip 必须真实改变 operator 嵌套层序")
        original_leaf = _path_leaf(original, original_path)
        transformed_leaf = _path_leaf(transformed, transformed_path)
        if original_leaf != transformed_leaf:
            raise ValueError("scope flip 前后必须共享完整叶命题")
        if original == transformed:
            raise ValueError("scope flip 必须改变 BoundProposition 语义对象")

        original_path_units, original_path_metadata = _path_units(
            original_path)
        transformed_path_units, transformed_path_metadata = _path_units(
            transformed_path)
        original_units, original_metadata = _bound_units(original)
        transformed_units, transformed_metadata = _bound_units(transformed)
        return build_typed_replacement_trace(
            (*original_path_units, *original_units),
            (*transformed_path_units, *transformed_units),
            transform_key=self.transform_key,
            source=source,
            scope=scope,
            impact_original=(
                original.template,
                *original_order,
            ),
            impact_transformed=(
                transformed.template,
                *transformed_order,
            ),
            metadata_keys=(
                *original_path_metadata,
                *original_metadata,
                *transformed_path_metadata,
                *transformed_metadata,
            ),
        )


__all__ = [
    "LogicScopeLayer",
    "NegationPerturbationAdapter",
    "QuantifierPerturbationAdapter",
    "ScopeFlipPerturbationAdapter",
]
