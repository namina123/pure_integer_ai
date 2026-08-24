"""DLG-RAW-16 G2A shadow consumer。

该边界把已确认的内容/来源计划与 G2A 表层结构 learner 接起来，但只返回
旁路 shadow 结果。它不替换 legacy surface、不推进会话、不写 Memory/SQLite，
也不把表层结果当作理解结果。当前允许已闭合的 ANSWER 与零 claim 的
CLARIFY/UNKNOWN/REPAIR 结构。非 ANSWER 的候选值由上游计划显式提供，
不由本层从文本猜测。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.determinism.fingerprint import integer_tuple_fingerprint
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    STRUCTURE_SELECTED,
    SurfaceStructureModel,
    SurfaceStructureRequest,
    SurfaceStructureResult,
    realize_surface_structure,
)


SHADOW_PROTOCOL_V1 = 1
SHADOW_SELECTED = 1
SHADOW_NOT_APPLICABLE = 2
SHADOW_REJECTED = 3
SHADOW_NO_PATTERN = 4
_SHADOW_DOMAIN = "pure_integer_ai.dlg_raw16.surface.shadow.v1"
_SEMANTIC_SURFACE_ROLES = frozenset({
    "subject", "topic", "cause", "predicate", "relation",
    "object", "claim", "effect",
})


class SurfaceShadowError(ValueError):
    """shadow plan 或 result 越过理解/来源授权边界。"""


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SurfaceShadowError(f"{where} 必须是无首尾空白的非空字符串")
    return value


@dataclass(frozen=True, slots=True)
class SurfaceShadowPlan:
    """理解/证据侧已经确认的 ANSWER/CLARIFY/UNKNOWN/REPAIR 内容计划。"""

    semantic: SurfaceSemantic
    dialogue_act: str
    register: str
    ordered_roles: tuple[str, ...]
    required_proposition_ids: tuple[str, ...]
    forbidden_proposition_ids: tuple[str, ...]
    authorized_source_ids: tuple[str, ...]
    context_id: str
    family_id: str
    legacy_surface: str
    min_chars: int = 1
    max_chars: int = 4096
    slot_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.semantic, SurfaceSemantic):
            raise TypeError("shadow semantic 类型错误")
        _text(self.dialogue_act, "shadow.dialogue_act")
        _text(self.register, "shadow.register")
        _text(self.context_id, "shadow.context_id")
        _text(self.family_id, "shadow.family_id")
        _text(self.legacy_surface, "shadow.legacy_surface")
        if not isinstance(self.slot_values, tuple):
            raise SurfaceShadowError("shadow.slot_values 必须是 tuple")
        if self.dialogue_act not in {"ANSWER", "CLARIFY", "UNKNOWN", "REPAIR"}:
            raise SurfaceShadowError(
                "当前 shadow 只允许已闭合 ANSWER 或零 claim CLARIFY/UNKNOWN/REPAIR")
        if not self.ordered_roles:
            raise SurfaceShadowError("shadow 缺少 ordered roles")
        if self.dialogue_act == "ANSWER":
            if not self.required_proposition_ids:
                raise SurfaceShadowError("ANSWER shadow 缺少 required proposition")
            if self.semantic.proposition_id not in self.required_proposition_ids:
                raise SurfaceShadowError("shadow required proposition 未绑定 semantic")
            if self.slot_values:
                if len(self.slot_values) != len(self.ordered_roles):
                    raise SurfaceShadowError(
                        "ANSWER slot_values 必须对应 ordered_roles")
                if all(role in _SEMANTIC_SURFACE_ROLES
                       for role in self.ordered_roles):
                    raise SurfaceShadowError(
                        "ANSWER 不得用 slot_values 绕过 semantic")
                if any(not isinstance(item, str) or not item
                       or item.strip() != item for item in self.slot_values):
                    raise SurfaceShadowError("ANSWER slot_values 含非法文本")
        else:
            if self.required_proposition_ids:
                raise SurfaceShadowError(
                    f"{self.dialogue_act} 不得携带 required proposition")
            if not self.slot_values:
                raise SurfaceShadowError(
                    f"{self.dialogue_act} shadow 缺少显式 slot_values")
            if len(self.slot_values) != len(self.ordered_roles):
                raise SurfaceShadowError(
                    f"{self.dialogue_act} slot_values 必须对应 ordered_roles")
            if any(not isinstance(item, str) or not item or item.strip() != item
                   for item in self.slot_values):
                raise SurfaceShadowError(
                    f"{self.dialogue_act} slot_values 含非法文本")
        if set(self.required_proposition_ids) & set(self.forbidden_proposition_ids):
            raise SurfaceShadowError("shadow required/forbidden proposition 冲突")
        if (not self.authorized_source_ids
                or any(not isinstance(item, str) or not item
                       for item in self.authorized_source_ids)):
            raise SurfaceShadowError("shadow 缺 authorized source")
        if type(self.min_chars) is not int or self.min_chars <= 0:
            raise SurfaceShadowError("shadow min_chars 非法")
        if type(self.max_chars) is not int or self.max_chars < self.min_chars:
            raise SurfaceShadowError("shadow length budget 非法")

    def structure_request(self) -> SurfaceStructureRequest:
        return SurfaceStructureRequest(
            self.semantic, self.dialogue_act, self.register, self.ordered_roles,
            self.min_chars, self.max_chars, 0,
            self.authorized_source_ids[0], self.context_id, self.family_id,
            self.slot_values,
        )

    def canonical_record(self) -> tuple[int, ...]:
        result = [SHADOW_PROTOCOL_V1, *self.structure_request().canonical_record()]
        for values in (self.required_proposition_ids,
                       self.forbidden_proposition_ids,
                       self.authorized_source_ids):
            result.append(len(values))
            for value in values:
                scalars = tuple(ord(item) for item in _text(value, "shadow.id"))
                result.extend((len(scalars), *scalars))
        scalars = tuple(ord(item) for item in self.legacy_surface)
        result.extend((len(scalars), *scalars))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class SurfaceShadowResult:
    plan: SurfaceShadowPlan
    status_code: int
    structure_result: SurfaceStructureResult | None
    shadow_surface: str | None
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.status_code not in {
                SHADOW_SELECTED, SHADOW_NOT_APPLICABLE, SHADOW_REJECTED,
                SHADOW_NO_PATTERN}:
            raise SurfaceShadowError("shadow status 未注册")
        if not self.trace or any(type(item) is not int or item < 0
                                 for item in self.trace):
            raise SurfaceShadowError("shadow trace 非法")
        if self.status_code == SHADOW_SELECTED:
            if (not isinstance(self.structure_result, SurfaceStructureResult)
                    or self.structure_result.status_code != STRUCTURE_SELECTED
                    or not isinstance(self.shadow_surface, str)
                    or not self.shadow_surface):
                raise SurfaceShadowError("shadow selected result 不完整")
        elif self.shadow_surface is not None:
            raise SurfaceShadowError("shadow 非 selected 不得携带输出")

    @property
    def replaced(self) -> int:
        """永远为 0：shadow 只旁路观察，不替换 legacy surface。"""
        return 0

    def canonical_record(self) -> tuple[int, ...]:
        result = [SHADOW_PROTOCOL_V1, self.status_code, self.replaced]
        result.extend(self.plan.canonical_record())
        if self.structure_result is None:
            result.append(0)
        else:
            record = self.structure_result.canonical_record()
            result.extend((len(record), *record))
        if self.shadow_surface is None:
            result.append(0)
        else:
            scalars = tuple(ord(item) for item in self.shadow_surface)
            result.extend((1, len(scalars), *scalars))
        result.extend((len(self.trace), *self.trace))
        return tuple(result)


def run_surface_shadow(
        model: SurfaceStructureModel,
        plan: SurfaceShadowPlan,
        ) -> SurfaceShadowResult:
    """只读消费结构 learner；任何授权/结构失败均不泄漏 shadow 文本。"""
    if not isinstance(model, SurfaceStructureModel):
        raise TypeError("shadow model 类型错误")
    if not isinstance(plan, SurfaceShadowPlan):
        raise TypeError("shadow plan 类型错误")
    structure = realize_surface_structure(model, plan.structure_request())
    if structure.status_code != STRUCTURE_SELECTED:
        status = SHADOW_NO_PATTERN
        shadow = None
    elif (plan.dialogue_act == "ANSWER"
          and plan.semantic.proposition_id not in plan.required_proposition_ids):
        status = SHADOW_REJECTED
        shadow = None
    else:
        status = SHADOW_SELECTED
        shadow = structure.surface
    values = [SHADOW_PROTOCOL_V1, status, *plan.canonical_record(),
              structure.status_code, structure.selected_pattern_id]
    trace = integer_tuple_fingerprint(tuple(values), domain=_SHADOW_DOMAIN)
    return SurfaceShadowResult(plan, status, structure, shadow, trace)


__all__ = [
    "SHADOW_NO_PATTERN", "SHADOW_NOT_APPLICABLE", "SHADOW_REJECTED",
    "SHADOW_SELECTED", "SurfaceShadowError", "SurfaceShadowPlan",
    "SurfaceShadowResult", "run_surface_shadow",
]
