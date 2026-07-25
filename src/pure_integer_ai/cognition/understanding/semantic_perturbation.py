"""H-02B 的 S-02 角色互换与 U-04 cue 错位 typed 适配器。

适配器只比较来源化命题、RoleBinding、cue 对象和图解析出的最小指令身份；不读取
surface，不猜固定 Role 名，也不在构造 trace 时判定命题真假或逻辑等价性。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_ROLE,
    ObjectIdentity,
    SourceRef,
    object_contracts_by_kind,
)
from pure_integer_ai.cognition.shared.language_signal import (
    LanguageSignalInstructionResolution,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    semantic_source,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.typed_perturbation import (
    TypedPerturbationTrace,
    build_typed_replacement_trace,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _require_scope(source: SourceRef, scope: ScopeIdentity) -> None:
    """核验 typed 语义扰动绑定同一来源化运行 scope。"""
    if not isinstance(scope, ScopeIdentity):
        raise TypeError("semantic perturbation scope 类型错误")
    if scope.source != source:
        raise ValueError("semantic perturbation scope 必须绑定命题来源")


def _slot_key(role: ObjectIdentity, ordinal: int) -> tuple[int, ...]:
    """把开放 Role 完整身份和 ordinal 编成无歧义元数据键。"""
    role_key = role.stable_key()
    return ordinal, len(role_key), *role_key


def _definition_units(
        definition: AtomicPropositionDefinition,
        ) -> tuple[ObjectIdentity, ...]:
    """展开命题及其完整 RoleBinding、Role 和 filler 身份供 trace 保存。"""
    units: list[ObjectIdentity] = [
        definition.proposition,
        definition.predicate,
        definition.source_anchor,
        definition.context,
    ]
    for binding in definition.canonical_bindings():
        units.extend((
            binding.identity_for(definition.proposition),
            binding.role,
            binding.filler,
        ))
    return tuple(units)


def _binding_map(
        definition: AtomicPropositionDefinition,
        ) -> dict[tuple[ObjectIdentity, int], AtomicRoleBinding]:
    """按完整 Role 与 ordinal 建立命题内唯一槽位映射。"""
    return {
        (binding.role, binding.ordinal): binding
        for binding in definition.canonical_bindings()
    }


def _require_authoritative(identity: ObjectIdentity, *, label: str) -> None:
    """拒绝 legacy 投影或未注册对象冒充 cue 的权威身份。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    contract = object_contracts_by_kind().get(identity.object_kind)
    if contract is None or not contract.authoritative_identity:
        raise ValueError(f"{label} 必须是权威一等对象")


@dataclass(frozen=True, order=True)
class SemanticRoleSlot:
    """由调用方指定的开放 Role/ordinal 槽，不承载固定角色意义。"""

    role: ObjectIdentity
    ordinal: int = 0

    def __post_init__(self) -> None:
        """核验槽位坐标使用完整 Role 身份和非负严格整数。"""
        if not isinstance(self.role, ObjectIdentity):
            raise TypeError("SemanticRoleSlot.role 类型错误")
        if self.role.object_kind != OBJECT_ROLE:
            raise ValueError("SemanticRoleSlot.role 必须是一等 Role")
        assert_int(self.ordinal, _where="SemanticRoleSlot.ordinal")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("SemanticRoleSlot.ordinal 必须是非负严格整数")

    def key(self) -> tuple[ObjectIdentity, int]:
        """返回命题 binding map 使用的精确槽位键。"""
        return self.role, self.ordinal


@dataclass(frozen=True)
class ResolvedCuePlacement:
    """保存一个 cue、U-04 唯一图指令和其当前语义作用目标。"""

    cue: ObjectIdentity
    instruction: ObjectIdentity
    target: ObjectIdentity
    resolution: LanguageSignalInstructionResolution

    def __post_init__(self) -> None:
        """核验 cue 权威身份、唯一图解析和目标来源化语义身份。"""
        _require_authoritative(self.cue, label="cue")
        if (not isinstance(self.instruction, ObjectIdentity)
                or self.instruction.object_kind
                != OBJECT_MINIMAL_INSTRUCTION):
            raise ValueError("cue instruction 必须是一等 MinimalInstruction")
        if not isinstance(
                self.resolution, LanguageSignalInstructionResolution):
            raise TypeError("cue resolution 类型错误")
        if (not self.resolution.has_evidence
                or self.resolution.instruction_key is None):
            raise ValueError("cue 必须有唯一且无冲突的 U-04 图指令")
        if self.instruction.components != self.resolution.instruction_key:
            raise ValueError("cue instruction 与 U-04 图解析不一致")
        if not isinstance(self.target, ObjectIdentity):
            raise TypeError("cue target 必须是 ObjectIdentity")
        semantic_source(self.target)

    def stable_key(self) -> tuple[int, ...]:
        """返回 cue、指令和目标的完整确定性键。"""
        result: list[int] = []
        for identity in (self.cue, self.instruction, self.target):
            key = identity.stable_key()
            result.extend((len(key), *key))
        return tuple(result)


@dataclass(frozen=True)
class RoleSwapPerturbationAdapter:
    """把两个 S-02 命题定义中的显式双槽 filler 互换接入 H-02A。"""

    transform_key: tuple[int, ...]

    def build(
            self,
            original: AtomicPropositionDefinition,
            transformed: AtomicPropositionDefinition,
            *,
            swapped_slots: tuple[SemanticRoleSlot, SemanticRoleSlot],
            scope: ScopeIdentity,
            ) -> TypedPerturbationTrace:
        """核验仅两个注入槽互换，并保存命题与 RoleBinding 的完整前后身份。"""
        if not isinstance(original, AtomicPropositionDefinition):
            raise TypeError("role swap original 类型错误")
        if not isinstance(transformed, AtomicPropositionDefinition):
            raise TypeError("role swap transformed 类型错误")
        if original.source != transformed.source:
            raise ValueError("role swap 前后命题必须来自同一 SourceRef")
        _require_scope(original.source, scope)
        if original.proposition == transformed.proposition:
            raise ValueError("role swap 必须使用不同 Proposition 身份")
        if (
                original.predicate != transformed.predicate
                or original.source_anchor != transformed.source_anchor
                or original.context != transformed.context):
            raise ValueError("role swap 不得同时改变 predicate、anchor 或 context")
        if (not isinstance(swapped_slots, tuple)
                or len(swapped_slots) != 2
                or any(not isinstance(item, SemanticRoleSlot)
                       for item in swapped_slots)):
            raise TypeError("swapped_slots 必须含两个 SemanticRoleSlot")
        first_slot, second_slot = swapped_slots
        if first_slot == second_slot:
            raise ValueError("role swap 两个槽位必须不同")

        original_map = _binding_map(original)
        transformed_map = _binding_map(transformed)
        if set(original_map) != set(transformed_map):
            raise ValueError("role swap 前后 Role/ordinal 槽集合必须相同")
        selected = {first_slot.key(), second_slot.key()}
        if not selected.issubset(original_map):
            raise ValueError("role swap 指定槽位不在命题定义中")
        for slot in set(original_map).difference(selected):
            if original_map[slot].filler != transformed_map[slot].filler:
                raise ValueError("role swap 不得改变指定双槽以外的 filler")
        first_before = original_map[first_slot.key()]
        second_before = original_map[second_slot.key()]
        first_after = transformed_map[first_slot.key()]
        second_after = transformed_map[second_slot.key()]
        if first_before.filler == second_before.filler:
            raise ValueError("相同 filler 的表面互换不构成角色扰动")
        if (first_after.filler != second_before.filler
                or second_after.filler != first_before.filler):
            raise ValueError("role swap 必须精确互换两个指定槽的 filler")

        impact_original = (
            first_before.identity_for(original.proposition),
            second_before.identity_for(original.proposition),
        )
        impact_transformed = (
            first_after.identity_for(transformed.proposition),
            second_after.identity_for(transformed.proposition),
        )
        metadata = tuple(
            _slot_key(item.role, item.ordinal)
            for item in sorted(swapped_slots)
        )
        return build_typed_replacement_trace(
            _definition_units(original),
            _definition_units(transformed),
            transform_key=self.transform_key,
            source=original.source,
            scope=scope,
            impact_original=impact_original,
            impact_transformed=impact_transformed,
            metadata_keys=metadata,
        )


@dataclass(frozen=True)
class CueMisalignmentPerturbationAdapter:
    """把同一组 U-04 resolved cue 的语义作用目标错位接入 H-02A。"""

    transform_key: tuple[int, ...]

    def build(
            self,
            original: tuple[ResolvedCuePlacement, ...],
            transformed: tuple[ResolvedCuePlacement, ...],
            *,
            source: SourceRef,
            scope: ScopeIdentity,
            ) -> TypedPerturbationTrace:
        """核验 cue/指令不变且目标只发生置换，保留所有三元对象身份。"""
        if not isinstance(source, SourceRef):
            raise TypeError("cue perturbation source 类型错误")
        _require_scope(source, scope)
        for values, label in (
                (original, "original"),
                (transformed, "transformed")):
            if (not isinstance(values, tuple) or len(values) < 2
                    or any(not isinstance(item, ResolvedCuePlacement)
                           for item in values)):
                raise TypeError(f"cue {label} 必须含至少两个 resolved placement")
            if len({item.cue for item in values}) != len(values):
                raise ValueError(f"cue {label} 不得重复 cue 身份")
            if any(semantic_source(item.target) != source for item in values):
                raise ValueError("cue target 必须与扰动 SourceRef 一致")

        original_map = {item.cue: item for item in original}
        transformed_map = {item.cue: item for item in transformed}
        if set(original_map) != set(transformed_map):
            raise ValueError("cue misalignment 前后 cue 集必须相同")
        if any(
                original_map[cue].instruction
                != transformed_map[cue].instruction
                for cue in original_map):
            raise ValueError("cue misalignment 不得替换 U-04 指令身份")
        original_targets = tuple(item.target for item in original_map.values())
        transformed_targets = tuple(
            item.target for item in transformed_map.values())
        if len(set(original_targets)) != len(original_targets):
            raise ValueError("cue misalignment 需要可审计的唯一原目标")
        if set(original_targets) != set(transformed_targets):
            raise ValueError("cue misalignment 只能置换既有目标，不能增删目标")
        changed_cues = tuple(sorted(
            (
                cue for cue in original_map
                if original_map[cue].target != transformed_map[cue].target
            ),
            key=ObjectIdentity.stable_key,
        ))
        if not changed_cues:
            raise ValueError("cue misalignment 必须真实改变至少一个作用目标")

        ordered_cues = tuple(sorted(
            original_map, key=ObjectIdentity.stable_key))
        original_ordered = tuple(original_map[cue] for cue in ordered_cues)
        transformed_ordered = tuple(
            transformed_map[cue] for cue in ordered_cues)
        original_units = tuple(
            identity
            for placement in original_ordered
            for identity in (
                placement.cue,
                placement.instruction,
                placement.target,
            )
        )
        transformed_units = tuple(
            identity
            for placement in transformed_ordered
            for identity in (
                placement.cue,
                placement.instruction,
                placement.target,
            )
        )
        impact_original = tuple(
            original_map[cue].target for cue in changed_cues)
        impact_transformed = tuple(
            transformed_map[cue].target for cue in changed_cues)
        metadata = tuple(cue.stable_key() for cue in changed_cues)
        return build_typed_replacement_trace(
            original_units,
            transformed_units,
            transform_key=self.transform_key,
            source=source,
            scope=scope,
            impact_original=impact_original,
            impact_transformed=impact_transformed,
            metadata_keys=metadata,
        )


__all__ = [
    "CueMisalignmentPerturbationAdapter",
    "ResolvedCuePlacement",
    "RoleSwapPerturbationAdapter",
    "SemanticRoleSlot",
]
