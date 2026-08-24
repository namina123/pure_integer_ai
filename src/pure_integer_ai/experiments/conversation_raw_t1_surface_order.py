"""T1-G9：显式 evidence 约束下的角色排列/语序 shadow。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pure_integer_ai.crosscut.determinism.fingerprint import integer_tuple_fingerprint
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceEvidencePack,
    SurfaceStructureLearningError,
    SurfaceStructureRequest,
    _validate_entry,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_schema import (
    DIALOGUE_ACTS,
    REGISTERS,
    SurfaceOrganizationRecord,
)


T1_G9_ORDER_PROTOCOL_V1 = 1
ORDER_SELECTED = 1
ORDER_NO_PATTERN = 2
ORDER_AMBIGUOUS = 3
_ORDER_DOMAIN = "pure_integer_ai.t1.g9.surface-order.v1"


class SurfaceOrderLearningError(SurfaceStructureLearningError):
    """角色排列 evidence、模型或 typed request 不满足合同。"""


def _text(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value or (not allow_empty and not value):
        raise SurfaceOrderLearningError(f"{where} 必须是规范字符串")
    return value


def _positive(value: Any, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise SurfaceOrderLearningError(f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, where: str) -> int:
    if type(value) is not int or value < 0:
        raise SurfaceOrderLearningError(f"{where} 必须是非负严格整数")
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
        raise SurfaceOrderLearningError("整数 record 含非法值")
    return (len(values), *values)


# object-model: value; representation=struct; interop=T1-G9
@dataclass(frozen=True, slots=True)
class SurfaceOrderOption:
    roles: tuple[str, ...]
    gaps: tuple[str, ...]
    support_record_ids: tuple[str, ...]
    support_family_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.roles or len(self.gaps) != len(self.roles) + 1:
            raise SurfaceOrderLearningError("order option 形状非法")
        if tuple(sorted(set(self.support_record_ids))) != self.support_record_ids:
            raise SurfaceOrderLearningError("order option record support 非规范")
        if tuple(sorted(set(self.support_family_ids))) != self.support_family_ids:
            raise SurfaceOrderLearningError("order option family support 非规范")
        if len(self.support_family_ids) < 2:
            raise SurfaceOrderLearningError("每个 role order 至少需要两个 family")

    def canonical_record(self) -> tuple[int, ...]:
        result = [T1_G9_ORDER_PROTOCOL_V1]
        result.extend(_pack_texts(self.roles, "order.roles"))
        result.extend(_pack_texts(self.gaps, "order.gaps", allow_empty=True))
        result.extend(_pack_texts(self.support_record_ids, "order.records"))
        result.extend(_pack_texts(self.support_family_ids, "order.families"))
        return tuple(result)


# object-model: value; representation=struct; interop=T1-G9
@dataclass(frozen=True, slots=True)
class SurfaceOrderPattern:
    pattern_id: int
    dialogue_act: str
    register: str
    options: tuple[SurfaceOrderOption, ...]

    def __post_init__(self) -> None:
        _positive(self.pattern_id, "pattern.pattern_id")
        if self.dialogue_act not in DIALOGUE_ACTS or self.register not in REGISTERS:
            raise SurfaceOrderLearningError("pattern act/register 非法")
        if len(self.options) < 2:
            raise SurfaceOrderLearningError("pattern 至少需要两个 role order")
        roles = tuple(item.roles for item in self.options)
        if len(set(roles)) != len(roles):
            raise SurfaceOrderLearningError("pattern role order 不得重复")

    def canonical_record(self) -> tuple[int, ...]:
        result = [T1_G9_ORDER_PROTOCOL_V1, self.pattern_id]
        result.extend(_pack_text(self.dialogue_act, "pattern.act"))
        result.extend(_pack_text(self.register, "pattern.register"))
        result.append(len(self.options))
        for option in self.options:
            record = option.canonical_record()
            result.extend((len(record), *record))
        return tuple(result)


# object-model: value; representation=struct; interop=T1-G9
@dataclass(frozen=True, slots=True)
class SurfaceOrderModel:
    patterns: tuple[SurfaceOrderPattern, ...]

    def __post_init__(self) -> None:
        if (not self.patterns
                or self.patterns != tuple(sorted(self.patterns, key=lambda item: item.pattern_id))):
            raise SurfaceOrderLearningError("order model patterns 必须按 id 排序")


# object-model: value; representation=struct; interop=T1-G9
@dataclass(frozen=True, slots=True)
class SurfaceOrderResult:
    request: SurfaceStructureRequest
    status_code: int
    candidate_count: int
    selected_pattern_id: int = 0
    selected_option_index: int = 0
    output_scalars: tuple[int, ...] = ()
    output_bytes: tuple[int, ...] = ()
    trace: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.status_code not in {ORDER_SELECTED, ORDER_NO_PATTERN, ORDER_AMBIGUOUS}:
            raise SurfaceOrderLearningError("order result status 未注册")
        _nonnegative(self.candidate_count, "result.candidate_count")
        _nonnegative(self.selected_pattern_id, "result.selected_pattern_id")
        _nonnegative(self.selected_option_index, "result.selected_option_index")
        if not self.trace or any(type(item) is not int or item < 0 for item in self.trace):
            raise SurfaceOrderLearningError("order result trace 非法")
        if self.status_code == ORDER_SELECTED:
            if self.candidate_count != 1 or not self.selected_pattern_id or not self.output_scalars:
                raise SurfaceOrderLearningError("selected order result 不完整")
            if tuple("".join(chr(item) for item in self.output_scalars).encode("utf-8")) != self.output_bytes:
                raise SurfaceOrderLearningError("order scalar/u8 漂移")
        elif self.selected_pattern_id or self.output_scalars or self.output_bytes:
            raise SurfaceOrderLearningError("非 selected order result 不得携带输出")

    @property
    def surface(self) -> str | None:
        return ("".join(chr(item) for item in self.output_scalars)
                if self.status_code == ORDER_SELECTED else None)

    def canonical_record(self) -> tuple[int, ...]:
        result = [T1_G9_ORDER_PROTOCOL_V1, self.status_code, self.candidate_count,
                  self.selected_pattern_id, self.selected_option_index]
        result.extend(_pack(self.request.canonical_record()))
        result.extend(_pack(self.output_scalars))
        result.extend(_pack(self.output_bytes))
        result.extend(_pack(self.trace))
        return tuple(result)


def _pattern_id(act: str, register: str, options: tuple[SurfaceOrderOption, ...]) -> int:
    values = [T1_G9_ORDER_PROTOCOL_V1, *_pack_text(act, "pattern.id.act"),
              *_pack_text(register, "pattern.id.register"), len(options)]
    for option in options:
        values.extend(option.canonical_record())
    digest = integer_tuple_fingerprint(tuple(values), domain=_ORDER_DOMAIN)
    result = 0
    for item in digest[2:]:
        result = (result << 8) | item
    return result or 1


def learn_surface_order_model(
        records: Iterable[SurfaceOrganizationRecord],
        evidence_pack: SurfaceEvidencePack,
        ) -> SurfaceOrderModel:
    """从 accepted span evidence 学习多个角色排列；不从表面猜 role。"""
    rows = tuple(records)
    if (not rows or any(not isinstance(item, SurfaceOrganizationRecord) for item in rows)
            or not isinstance(evidence_pack, SurfaceEvidencePack)):
        raise TypeError("records/evidence_pack 类型错误")
    grouped: dict[
        tuple[str, str, tuple[str, ...]],
        dict[tuple[str, ...], dict[str, set[str]]],
    ] = {}
    for record in rows:
        for variant in record.accepted:
            entries = tuple(item for item in evidence_pack.entries
                            if item.record_id == record.sample_id
                            and item.variant_id == variant.variant_id)
            if not entries:
                continue
            gaps, roles = _validate_entry(record, variant, entries)
            key = (record.dialogue_act, record.register, roles)
            # A role order is supported independently; gaps remain typed tuple
            # keys, never a string that must be interpreted again.
            state = grouped.setdefault(key, {})
            option_state = state.setdefault(gaps, {"records": set(), "families": set()})
            option_state["records"].add(record.sample_id)
            option_state["families"].add(record.family_id)
    grouped_by_act: dict[tuple[str, str], list[SurfaceOrderOption]] = {}
    for (act, register, roles), state in grouped.items():
        for gaps, option_state in sorted(state.items()):
            option_records = tuple(sorted(option_state["records"]))
            option_families = tuple(sorted(option_state["families"]))
            if len(option_families) < 2:
                continue
            grouped_by_act.setdefault((act, register), []).append(
                SurfaceOrderOption(roles, gaps, option_records, option_families))
    patterns = []
    for (act, register), options in grouped_by_act.items():
        unique: dict[tuple[str, ...], SurfaceOrderOption] = {}
        for item in options:
            prior = unique.get(item.roles)
            if prior is None:
                unique[item.roles] = item
            elif (item.gaps, item.support_record_ids) < (prior.gaps, prior.support_record_ids):
                unique[item.roles] = item
        ordered = tuple(unique[key] for key in sorted(unique))
        if len(ordered) < 2:
            continue
        patterns.append(SurfaceOrderPattern(
            _pattern_id(act, register, ordered), act, register, ordered))
    if not patterns:
        raise SurfaceOrderLearningError(
            "没有得到由两个独立 family 支持的多个 role order")
    return SurfaceOrderModel(tuple(sorted(patterns, key=lambda item: item.pattern_id)))


def _semantic_value(semantic: SurfaceSemantic, role: str) -> str | None:
    if role in {"subject", "topic", "cause"}:
        return semantic.subject
    if role in {"predicate", "relation"}:
        return semantic.predicate
    if role in {"object", "claim", "effect"}:
        return semantic.object
    return None


def realize_surface_order(model: SurfaceOrderModel, request: SurfaceStructureRequest) -> SurfaceOrderResult:
    """按 typed ordered roles 重组一个已学语序；未知排列 fail closed。"""
    if not isinstance(model, SurfaceOrderModel) or not isinstance(request, SurfaceStructureRequest):
        raise TypeError("model/request 类型错误")
    patterns = tuple(item for item in model.patterns
                     if item.dialogue_act == request.dialogue_act
                     and item.register == request.register)
    if not patterns:
        return _result(request, ORDER_NO_PATTERN, ())
    if len(patterns) != 1:
        return _result(request, ORDER_AMBIGUOUS, patterns)
    pattern = patterns[0]
    options = tuple(item for item in pattern.options if item.roles == request.ordered_roles)
    if not options:
        return _result(request, ORDER_NO_PATTERN, ())
    option = options[0]
    values: list[str] = []
    for index, gap in enumerate(option.gaps):
        values.append(gap)
        if index < len(option.roles):
            value = _semantic_value(request.semantic, option.roles[index])
            if value is None:
                if not request.slot_values or index >= len(request.slot_values):
                    return _result(request, ORDER_NO_PATTERN, ())
                value = request.slot_values[index]
            values.append(value)
    surface = "".join(values)
    scalars = tuple(ord(item) for item in surface)
    if not request.min_chars <= len(scalars) <= request.max_chars:
        return _result(request, ORDER_NO_PATTERN, ())
    output = tuple(surface.encode("utf-8"))
    return _result(request, ORDER_SELECTED, patterns,
                   selected_pattern_id=pattern.pattern_id,
                   output_scalars=scalars, output_bytes=output)


def _result(request: SurfaceStructureRequest, status: int,
            patterns: tuple[SurfaceOrderPattern, ...], **kwargs: Any) -> SurfaceOrderResult:
    values = [T1_G9_ORDER_PROTOCOL_V1, status, len(patterns), *request.canonical_record()]
    values.extend(item.pattern_id for item in patterns)
    trace = integer_tuple_fingerprint(tuple(values), domain=_ORDER_DOMAIN)
    return SurfaceOrderResult(request, status, len(patterns), trace=trace, **kwargs)


__all__ = [
    "ORDER_AMBIGUOUS", "ORDER_NO_PATTERN", "ORDER_SELECTED",
    "SurfaceOrderLearningError", "SurfaceOrderModel", "SurfaceOrderOption",
    "SurfaceOrderPattern", "SurfaceOrderResult", "T1_G9_ORDER_PROTOCOL_V1",
    "learn_surface_order_model", "realize_surface_order",
]
