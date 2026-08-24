"""T1-G10：统一 G7 gap 与 G9 role-order 的只读 surface plan。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.determinism.fingerprint import integer_tuple_fingerprint
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceStructureModel,
    SurfaceStructureRequest,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_schema import (
    DIALOGUE_ACTS,
    REGISTERS,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_adapter import (
    RawT1ShadowAdapterResult,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_dialogue import (
    RawT1ShadowDialogueState,
    RawT1ShadowDialogueTurn,
    run_raw_t1_shadow_dialogue_turn,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_order import (
    SurfaceOrderModel,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_variants import (
    SurfaceVariantModel,
)


T1_G10_SURFACE_PLAN_PROTOCOL_V1 = 1
PLAN_SELECTED = 1
PLAN_NO_PATTERN = 2
PLAN_AMBIGUOUS = 3
_PLAN_DOMAIN = "pure_integer_ai.t1.g10.surface-plan.v1"
ORIGIN_G7_GAP = 7
ORIGIN_G9_ORDER = 9
ORIGIN_G2A_STRUCTURE = 16


class SurfacePlanError(ValueError):
    """统一 surface plan 的模型、请求或 focus 边界不满足合同。"""


def _text(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value or (not allow_empty and not value):
        raise SurfacePlanError(f"{where} 必须是规范字符串")
    return value


def _positive(value: Any, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise SurfacePlanError(f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, where: str) -> int:
    if type(value) is not int or value < 0:
        raise SurfacePlanError(f"{where} 必须是非负严格整数")
    return value


def _pack_text(value: str, where: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    value = _text(value, where, allow_empty=allow_empty)
    scalars = tuple(ord(item) for item in value)
    return (len(scalars), *scalars)


def _pack_texts(values: tuple[str, ...], where: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    result = [len(values)]
    for index, value in enumerate(values):
        result.extend(_pack_text(value, f"{where}[{index}]", allow_empty=allow_empty))
    return tuple(result)


def _pack(values: tuple[int, ...]) -> tuple[int, ...]:
    if any(type(item) is not int or item < 0 for item in values):
        raise SurfacePlanError("integer record 含非法值")
    return (len(values), *values)


# object-model: value; representation=struct; interop=T1-G10
@dataclass(frozen=True, slots=True)
class SurfacePlanOption:
    roles: tuple[str, ...]
    gaps: tuple[str, ...]
    support_record_ids: tuple[str, ...]
    support_family_ids: tuple[str, ...]
    origins: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.roles or len(self.gaps) != len(self.roles) + 1:
            raise SurfacePlanError("plan option 形状非法")
        if tuple(sorted(set(self.support_record_ids))) != self.support_record_ids:
            raise SurfacePlanError("plan option record support 非规范")
        if tuple(sorted(set(self.support_family_ids))) != self.support_family_ids:
            raise SurfacePlanError("plan option family support 非规范")
        if len(self.support_family_ids) < 2:
            raise SurfacePlanError("plan option 至少需要两个 family")
        if not self.origins or any(item not in {
                ORIGIN_G2A_STRUCTURE, ORIGIN_G7_GAP, ORIGIN_G9_ORDER,
        } for item in self.origins):
            raise SurfacePlanError("plan option origin 未注册")

    def canonical_record(self) -> tuple[int, ...]:
        result = [T1_G10_SURFACE_PLAN_PROTOCOL_V1]
        result.extend(_pack_texts(self.roles, "option.roles"))
        result.extend(_pack_texts(self.gaps, "option.gaps", allow_empty=True))
        result.extend(_pack_texts(self.support_record_ids, "option.records"))
        result.extend(_pack_texts(self.support_family_ids, "option.families"))
        result.extend(self.origins)
        return tuple(result)


# object-model: value; representation=struct; interop=T1-G10
@dataclass(frozen=True, slots=True)
class SurfacePlanPattern:
    pattern_id: int
    dialogue_act: str
    register: str
    options: tuple[SurfacePlanOption, ...]

    def __post_init__(self) -> None:
        _positive(self.pattern_id, "pattern.pattern_id")
        if self.dialogue_act not in DIALOGUE_ACTS or self.register not in REGISTERS:
            raise SurfacePlanError("plan pattern act/register 非法")
        if not self.options:
            raise SurfacePlanError("plan pattern 不完整")
        if any(not isinstance(item, SurfacePlanOption) for item in self.options):
            raise SurfacePlanError("plan pattern option 类型错误")

    def canonical_record(self) -> tuple[int, ...]:
        result = [T1_G10_SURFACE_PLAN_PROTOCOL_V1, self.pattern_id]
        result.extend(_pack_text(self.dialogue_act, "pattern.act"))
        result.extend(_pack_text(self.register, "pattern.register"))
        result.append(len(self.options))
        for option in self.options:
            record = option.canonical_record()
            result.extend((len(record), *record))
        return tuple(result)


# object-model: value; representation=struct; interop=T1-G10
@dataclass(frozen=True, slots=True)
class SurfacePlanModel:
    patterns: tuple[SurfacePlanPattern, ...]

    def __post_init__(self) -> None:
        if (not self.patterns
                or self.patterns != tuple(sorted(self.patterns, key=lambda item: item.pattern_id))
                or len({item.pattern_id for item in self.patterns}) != len(self.patterns)):
            raise SurfacePlanError("plan model patterns 必须按 id 排序")


# object-model: value; representation=struct; interop=T1-G10
@dataclass(frozen=True, slots=True)
class SurfacePlanResult:
    request: SurfaceStructureRequest
    status_code: int
    candidate_count: int
    selected_pattern_id: int = 0
    selected_option_index: int = 0
    output_scalars: tuple[int, ...] = ()
    output_bytes: tuple[int, ...] = ()
    trace: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.status_code not in {PLAN_SELECTED, PLAN_NO_PATTERN, PLAN_AMBIGUOUS}:
            raise SurfacePlanError("plan result status 未注册")
        _nonnegative(self.candidate_count, "result.candidate_count")
        _nonnegative(self.selected_pattern_id, "result.selected_pattern_id")
        _nonnegative(self.selected_option_index, "result.selected_option_index")
        if not self.trace or any(type(item) is not int or item < 0 for item in self.trace):
            raise SurfacePlanError("plan result trace 非法")
        if self.status_code == PLAN_SELECTED:
            if self.candidate_count != 1 or not self.selected_pattern_id or not self.output_scalars:
                raise SurfacePlanError("selected plan result 不完整")
            if tuple("".join(chr(item) for item in self.output_scalars).encode("utf-8")) != self.output_bytes:
                raise SurfacePlanError("plan scalar/u8 漂移")
        elif self.selected_pattern_id or self.output_scalars or self.output_bytes:
            raise SurfacePlanError("非 selected plan result 不得携带输出")

    @property
    def surface(self) -> str | None:
        return ("".join(chr(item) for item in self.output_scalars)
                if self.status_code == PLAN_SELECTED else None)

    def canonical_record(self) -> tuple[int, ...]:
        result = [T1_G10_SURFACE_PLAN_PROTOCOL_V1, self.status_code, self.candidate_count,
                  self.selected_pattern_id, self.selected_option_index]
        result.extend(_pack(self.request.canonical_record()))
        result.extend(_pack(self.output_scalars))
        result.extend(_pack(self.output_bytes))
        result.extend(_pack(self.trace))
        return tuple(result)


def _pattern_id(act: str, register: str, options: tuple[SurfacePlanOption, ...]) -> int:
    values = [T1_G10_SURFACE_PLAN_PROTOCOL_V1, *_pack_text(act, "pattern.id.act"),
              *_pack_text(register, "pattern.id.register"), len(options)]
    for option in options:
        values.extend(option.canonical_record())
    digest = integer_tuple_fingerprint(tuple(values), domain=_PLAN_DOMAIN)
    result = 0
    for item in digest[2:]:
        result = (result << 8) | item
    return result or 1


def build_surface_plan_model(
        gap_model: SurfaceVariantModel,
        order_model: SurfaceOrderModel,
        structure_model: SurfaceStructureModel | None = None,
        ) -> SurfacePlanModel:
    """将 G7/G9 两种已验证模型归一为一个只读 plan。"""
    if (not isinstance(gap_model, SurfaceVariantModel)
            or not isinstance(order_model, SurfaceOrderModel)):
        raise TypeError("gap_model/order_model 类型错误")
    if structure_model is not None and not isinstance(structure_model, SurfaceStructureModel):
        raise TypeError("structure_model 类型错误")
    grouped: dict[tuple[str, str, tuple[str, ...], tuple[str, ...]], dict[str, set[str] | set[int]]] = {}
    for pattern in gap_model.patterns:
        for gaps in pattern.gap_options:
            key = (pattern.dialogue_act, pattern.register, pattern.roles, gaps)
            state = grouped.setdefault(key, {"records": set(), "families": set(), "origins": set()})
            state["records"].update(pattern.support_record_ids)  # type: ignore[union-attr]
            state["families"].update(pattern.support_family_ids)  # type: ignore[union-attr]
            state["origins"].add(ORIGIN_G7_GAP)  # type: ignore[union-attr]
    for pattern in order_model.patterns:
        for option in pattern.options:
            key = (pattern.dialogue_act, pattern.register, option.roles, option.gaps)
            state = grouped.setdefault(key, {"records": set(), "families": set(), "origins": set()})
            state["records"].update(option.support_record_ids)  # type: ignore[union-attr]
            state["families"].update(option.support_family_ids)  # type: ignore[union-attr]
            state["origins"].add(ORIGIN_G9_ORDER)  # type: ignore[union-attr]
    if structure_model is not None:
        for pattern in structure_model.patterns:
            key = (pattern.dialogue_act, pattern.register, pattern.roles, pattern.gaps)
            state = grouped.setdefault(key, {"records": set(), "families": set(), "origins": set()})
            state["records"].update(pattern.support_record_ids)  # type: ignore[union-attr]
            state["families"].update(pattern.support_family_ids)  # type: ignore[union-attr]
            state["origins"].add(ORIGIN_G2A_STRUCTURE)  # type: ignore[union-attr]
    by_act: dict[
        tuple[str, str],
        dict[tuple[str, ...], dict[tuple[str, ...], SurfacePlanOption]],
    ] = {}
    for (act, register, roles, gaps), state in grouped.items():
        records = tuple(sorted(state["records"]))  # type: ignore[arg-type]
        families = tuple(sorted(state["families"]))  # type: ignore[arg-type]
        origins = tuple(sorted(state["origins"]))  # type: ignore[arg-type]
        if len(families) < 2:
            continue
        option = SurfacePlanOption(roles, gaps, records, families, origins)
        by_act.setdefault((act, register), {}).setdefault(roles, {})[gaps] = option
    patterns = []
    for (act, register), role_groups in by_act.items():
        ordered_list: list[SurfacePlanOption] = []
        for roles in sorted(role_groups):
            ordered_list.extend(role_groups[roles][gaps] for gaps in sorted(role_groups[roles]))
        ordered = tuple(ordered_list)
        patterns.append(SurfacePlanPattern(
            _pattern_id(act, register, ordered), act, register, ordered))
    if not patterns:
        raise SurfacePlanError("G7/G9 没有可归一化的 plan")
    return SurfacePlanModel(tuple(sorted(patterns, key=lambda item: item.pattern_id)))


def _semantic_value(request: SurfaceStructureRequest, role: str, index: int) -> str | None:
    semantic = request.semantic
    if role in {"subject", "topic", "cause"}:
        return semantic.subject
    if role in {"predicate", "relation"}:
        return semantic.predicate
    if role in {"object", "claim", "effect"}:
        return semantic.object
    if request.slot_values and index < len(request.slot_values):
        return request.slot_values[index]
    return None


def realize_surface_plan(model: SurfacePlanModel, request: SurfaceStructureRequest) -> SurfacePlanResult:
    """以统一 plan 重组 typed surface；未见 act/order 仍 fail closed。"""
    if not isinstance(model, SurfacePlanModel) or not isinstance(request, SurfaceStructureRequest):
        raise TypeError("model/request 类型错误")
    patterns = tuple(item for item in model.patterns
                     if item.dialogue_act == request.dialogue_act
                     and item.register == request.register)
    if not patterns:
        return _result(request, PLAN_NO_PATTERN, ())
    if len(patterns) != 1:
        return _result(request, PLAN_AMBIGUOUS, patterns)
    pattern = patterns[0]
    options = tuple(item for item in pattern.options if item.roles == request.ordered_roles)
    if not options:
        return _result(request, PLAN_NO_PATTERN, ())
    option_index = request.selection_ordinal % len(options)
    option = options[option_index]
    values: list[str] = []
    for index, gap in enumerate(option.gaps):
        values.append(gap)
        if index < len(option.roles):
            value = _semantic_value(request, option.roles[index], index)
            if value is None:
                return _result(request, PLAN_NO_PATTERN, ())
            values.append(value)
    surface = "".join(values)
    scalars = tuple(ord(item) for item in surface)
    if not request.min_chars <= len(scalars) <= request.max_chars:
        return _result(request, PLAN_NO_PATTERN, ())
    output = tuple(surface.encode("utf-8"))
    return _result(request, PLAN_SELECTED, patterns,
                   selected_pattern_id=pattern.pattern_id,
                   selected_option_index=option_index,
                   output_scalars=scalars, output_bytes=output)


def _result(request: SurfaceStructureRequest, status: int,
            patterns: tuple[SurfacePlanPattern, ...], **kwargs: Any) -> SurfacePlanResult:
    values = [T1_G10_SURFACE_PLAN_PROTOCOL_V1, status, len(patterns), *request.canonical_record()]
    values.extend(item.pattern_id for item in patterns)
    trace = integer_tuple_fingerprint(tuple(values), domain=_PLAN_DOMAIN)
    return SurfacePlanResult(request, status, len(patterns), trace=trace, **kwargs)


# object-model: value; representation=struct; interop=T1-G10
@dataclass(frozen=True, slots=True)
class UnifiedFocusSurfaceResult:
    dialogue_turn: RawT1ShadowDialogueTurn
    plan_result: SurfacePlanResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.dialogue_turn, RawT1ShadowDialogueTurn):
            raise TypeError("dialogue_turn 类型错误")
        if self.dialogue_turn.adapter_result.replaced != 0:
            raise SurfacePlanError("统一 focus shadow 不得替换旧答案")
        act = self.dialogue_turn.adapter_result.consumer.response_act
        if act == "ANSWER":
            if (not isinstance(self.plan_result, SurfacePlanResult)
                    or self.plan_result.status_code != PLAN_SELECTED):
                raise SurfacePlanError("ANSWER 必须有 selected unified plan")
        elif self.plan_result is not None:
            raise SurfacePlanError("UNKNOWN/CLARIFY 不得携带 unified surface")

    @property
    def replaced(self) -> int:
        return 0

    @property
    def surface(self) -> str | None:
        return None if self.plan_result is None else self.plan_result.surface

    def canonical_record(self) -> tuple[int, ...]:
        turn = self.dialogue_turn.canonical_record()
        result = [T1_G10_SURFACE_PLAN_PROTOCOL_V1, self.replaced, len(turn), *turn]
        if self.plan_result is None:
            result.append(0)
        else:
            record = self.plan_result.canonical_record()
            result.extend((len(record), *record))
        return tuple(result)


def run_unified_focus_surface_turn(
        state: RawT1ShadowDialogueState,
        adapter_result: RawT1ShadowAdapterResult,
        model: SurfacePlanModel,
        *,
        selection_ordinal: int = 0,
        ) -> UnifiedFocusSurfaceResult:
    """推进 G6 focus，并以统一 G10 plan 只读生成 ANSWER surface。"""
    if not isinstance(model, SurfacePlanModel):
        raise TypeError("统一 surface model 类型错误")
    if type(selection_ordinal) is not int or selection_ordinal < 0:
        raise SurfacePlanError("selection_ordinal 必须是非负严格整数")
    turn = run_raw_t1_shadow_dialogue_turn(state, adapter_result)
    consumer = adapter_result.consumer
    if consumer.response_act == "ANSWER":
        plan = adapter_result.shadow.plan
        request = SurfaceStructureRequest(
            plan.semantic, plan.dialogue_act, plan.register, plan.ordered_roles,
            plan.min_chars, plan.max_chars, selection_ordinal,
            plan.authorized_source_ids[0], plan.context_id, plan.family_id,
            plan.slot_values,
        )
        return UnifiedFocusSurfaceResult(turn, realize_surface_plan(model, request))
    if consumer.response_act not in {"UNKNOWN", "CLARIFY"}:
        raise SurfacePlanError("统一 focus response-act 未注册")
    if adapter_result.shadow.plan.required_proposition_ids:
        raise SurfacePlanError("UNKNOWN/CLARIFY obligation 不得携带 claim")
    return UnifiedFocusSurfaceResult(turn, None)


__all__ = [
    "ORIGIN_G2A_STRUCTURE", "ORIGIN_G7_GAP", "ORIGIN_G9_ORDER",
    "PLAN_AMBIGUOUS", "PLAN_NO_PATTERN",
    "PLAN_SELECTED", "SurfacePlanError", "SurfacePlanModel", "SurfacePlanOption",
    "SurfacePlanPattern", "SurfacePlanResult", "T1_G10_SURFACE_PLAN_PROTOCOL_V1",
    "UnifiedFocusSurfaceResult", "build_surface_plan_model", "realize_surface_plan",
    "run_unified_focus_surface_turn",
]
