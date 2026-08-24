"""T1-G7：带显式 slot evidence 的多表层结构 shadow。

该模块只学习 accepted surface 的 literal gap 变体；命题字段和非语义槽值仍由
运行时显式提供。训练身份不进入模型，输出只保留 canonical integer trace，因而
不能把模板回放或自由文本补全冒充语言泛化。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pure_integer_ai.crosscut.determinism.fingerprint import integer_tuple_fingerprint
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    DLG_RAW16_STRUCTURE_PROTOCOL_V1,
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


T1_G7_VARIANT_PROTOCOL_V1 = 1
VARIANT_SELECTED = 1
VARIANT_NO_PATTERN = 2
VARIANT_AMBIGUOUS = 3
_VARIANT_DOMAIN = "pure_integer_ai.t1.g7.surface-variants.v1"


class SurfaceVariantLearningError(SurfaceStructureLearningError):
    """多表层结构的 evidence、模型或请求不满足合同。"""


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SurfaceVariantLearningError(f"{where} 必须是无首尾空白的非空字符串")
    return value


def _positive(value: Any, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise SurfaceVariantLearningError(f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, where: str) -> int:
    if type(value) is not int or value < 0:
        raise SurfaceVariantLearningError(f"{where} 必须是非负严格整数")
    return value


def _pack_text(value: str, where: str) -> tuple[int, ...]:
    scalars = tuple(ord(item) for item in _text(value, where))
    return (len(scalars), *scalars)


def _pack_text_value(value: str, where: str) -> tuple[int, ...]:
    if not isinstance(value, str) or value.strip() != value:
        raise SurfaceVariantLearningError(f"{where} 必须是无首尾空白字符串")
    scalars = tuple(ord(item) for item in value)
    return (len(scalars), *scalars)


def _pack_texts(values: tuple[str, ...], where: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    result = [len(values)]
    for index, value in enumerate(values):
        result.extend((_pack_text_value if allow_empty else _pack_text)(
            value, f"{where}[{index}]"))
    return tuple(result)


def _pack(values: tuple[int, ...]) -> tuple[int, ...]:
    if any(type(item) is not int or item < 0 for item in values):
        raise SurfaceVariantLearningError("整数 record 含非法值")
    return (len(values), *values)


def _semantic_value(semantic: SurfaceSemantic, role: str) -> str | None:
    if role in {"subject", "topic", "cause"}:
        return semantic.subject
    if role in {"predicate", "relation"}:
        return semantic.predicate
    if role in {"object", "claim", "effect"}:
        return semantic.object
    return None


# object-model: value; representation=struct; interop=T1-G7
@dataclass(frozen=True, slots=True)
class SurfaceVariantPattern:
    """由两个独立 family 支持、且至少有两个 literal gap 选项的结构。"""

    pattern_id: int
    dialogue_act: str
    register: str
    roles: tuple[str, ...]
    gap_options: tuple[tuple[str, ...], ...]
    support_record_ids: tuple[str, ...]
    support_family_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _positive(self.pattern_id, "pattern.pattern_id")
        if self.dialogue_act not in DIALOGUE_ACTS or self.register not in REGISTERS:
            raise SurfaceVariantLearningError("pattern act/register 非法")
        if not self.roles or any(not isinstance(item, str) or not item for item in self.roles):
            raise SurfaceVariantLearningError("pattern.roles 非法")
        if len(self.gap_options) < 2:
            raise SurfaceVariantLearningError("pattern 必须保留至少两个 gap option")
        if len(set(self.gap_options)) != len(self.gap_options):
            raise SurfaceVariantLearningError("pattern gap option 不得重复")
        if any(len(item) != len(self.roles) + 1 for item in self.gap_options):
            raise SurfaceVariantLearningError("pattern gap option 形状非法")
        if len(self.support_family_ids) < 2:
            raise SurfaceVariantLearningError("pattern 必须由两个独立 family 支持")
        if tuple(sorted(set(self.support_record_ids))) != self.support_record_ids:
            raise SurfaceVariantLearningError("pattern support records 非规范")
        if tuple(sorted(set(self.support_family_ids))) != self.support_family_ids:
            raise SurfaceVariantLearningError("pattern support families 非规范")

    def canonical_record(self) -> tuple[int, ...]:
        result = [T1_G7_VARIANT_PROTOCOL_V1, self.pattern_id]
        for value in (self.dialogue_act, self.register):
            result.extend(_pack_text(value, "pattern.text"))
        result.extend(_pack_texts(self.roles, "pattern.roles"))
        result.append(len(self.gap_options))
        for option in self.gap_options:
            result.extend(_pack_texts(option, "pattern.gaps", allow_empty=True))
        result.extend(_pack_texts(self.support_record_ids, "pattern.records"))
        result.extend(_pack_texts(self.support_family_ids, "pattern.families"))
        return tuple(result)


# object-model: value; representation=struct; interop=T1-G7
@dataclass(frozen=True, slots=True)
class SurfaceVariantModel:
    patterns: tuple[SurfaceVariantPattern, ...]

    def __post_init__(self) -> None:
        if (not self.patterns
                or self.patterns != tuple(sorted(self.patterns, key=lambda item: item.pattern_id))
                or len({item.pattern_id for item in self.patterns}) != len(self.patterns)):
            raise SurfaceVariantLearningError("model patterns 必须按 id 排序且唯一")


# object-model: value; representation=struct; interop=T1-G7
@dataclass(frozen=True, slots=True)
class SurfaceVariantResult:
    request: SurfaceStructureRequest
    status_code: int
    candidate_count: int
    selected_pattern_id: int = 0
    selected_option_index: int = 0
    output_scalars: tuple[int, ...] = ()
    output_bytes: tuple[int, ...] = ()
    trace: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.status_code not in {VARIANT_SELECTED, VARIANT_NO_PATTERN, VARIANT_AMBIGUOUS}:
            raise SurfaceVariantLearningError("result status 未注册")
        _nonnegative(self.candidate_count, "result.candidate_count")
        _nonnegative(self.selected_pattern_id, "result.selected_pattern_id")
        _nonnegative(self.selected_option_index, "result.selected_option_index")
        if not self.trace or any(type(item) is not int or item < 0 for item in self.trace):
            raise SurfaceVariantLearningError("result trace 非法")
        if self.status_code == VARIANT_SELECTED:
            if (self.candidate_count != 1 or not self.selected_pattern_id
                    or not self.output_scalars):
                raise SurfaceVariantLearningError("selected result 不完整")
            if tuple("".join(chr(item) for item in self.output_scalars).encode("utf-8")) != self.output_bytes:
                raise SurfaceVariantLearningError("result scalar/u8 漂移")
        elif self.selected_pattern_id or self.output_scalars or self.output_bytes:
            raise SurfaceVariantLearningError("非 selected result 不得携带输出")

    @property
    def surface(self) -> str | None:
        return ("".join(chr(item) for item in self.output_scalars)
                if self.status_code == VARIANT_SELECTED else None)

    def canonical_record(self) -> tuple[int, ...]:
        result = [T1_G7_VARIANT_PROTOCOL_V1, self.status_code, self.candidate_count,
                  self.selected_pattern_id, self.selected_option_index]
        result.extend(_pack(self.request.canonical_record()))
        result.extend(_pack(self.output_scalars))
        result.extend(_pack(self.output_bytes))
        result.extend(_pack(self.trace))
        return tuple(result)


def _pattern_id(dialogue_act: str, register: str, roles: tuple[str, ...],
                options: tuple[tuple[str, ...], ...]) -> int:
    values = [T1_G7_VARIANT_PROTOCOL_V1]
    for value in (dialogue_act, register):
        values.extend(_pack_text(value, "pattern.id"))
    values.extend(_pack_texts(roles, "pattern.id.roles"))
    values.append(len(options))
    for option in options:
        values.extend(_pack_texts(option, "pattern.id.gaps", allow_empty=True))
    digest = integer_tuple_fingerprint(tuple(values), domain=_VARIANT_DOMAIN)
    result = 0
    for item in digest[2:]:
        result = (result << 8) | item
    return result or 1


def learn_surface_variant_model(
        records: Iterable[SurfaceOrganizationRecord],
        evidence_pack: SurfaceEvidencePack,
        ) -> SurfaceVariantModel:
    """从显式 evidence 学习 literal gap 选项，不保留训练实体。"""
    rows = tuple(records)
    if (not rows or any(not isinstance(item, SurfaceOrganizationRecord) for item in rows)
            or not isinstance(evidence_pack, SurfaceEvidencePack)):
        raise TypeError("records/evidence_pack 类型错误")
    grouped: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    for record in rows:
        expected_roles = tuple(item.role for item in record.clause_slots)
        for variant in record.accepted:
            entries = tuple(item for item in evidence_pack.entries
                            if item.record_id == record.sample_id
                            and item.variant_id == variant.variant_id)
            if not entries:
                continue
            gaps, roles = _validate_entry(record, variant, entries)
            key = (record.dialogue_act, record.register, roles)
            if roles != expected_roles:
                raise SurfaceVariantLearningError("evidence roles 与 clause slots 漂移")
            state = grouped.setdefault(key, {"options": {}, "records": set(), "families": set()})
            option_state = state["options"].setdefault(gaps, {"records": set(), "families": set()})
            option_state["records"].add(record.sample_id)
            option_state["families"].add(record.family_id)
            state["records"].add(record.sample_id)
            state["families"].add(record.family_id)
    patterns = []
    for (act, register, roles), state in grouped.items():
        options = tuple(sorted(state["options"]))
        if len(options) < 2 or len(state["families"]) < 2:
            continue
        patterns.append(SurfaceVariantPattern(
            _pattern_id(act, register, roles, options), act, register, roles, options,
            tuple(sorted(state["records"])), tuple(sorted(state["families"])),
        ))
    if not patterns:
        raise SurfaceVariantLearningError(
            "没有得到由两个独立 family 支持的多 gap structure")
    return SurfaceVariantModel(tuple(sorted(patterns, key=lambda item: item.pattern_id)))


def _request_value(request: SurfaceStructureRequest, role: str, index: int) -> str | None:
    value = _semantic_value(request.semantic, role)
    if value is not None:
        return value
    if request.slot_values:
        return request.slot_values[index]
    return None


def realize_surface_variants(
        model: SurfaceVariantModel,
        request: SurfaceStructureRequest,
        ) -> SurfaceVariantResult:
    """按 selection ordinal 选择一个已学 gap 变体并填入 typed values。"""
    if not isinstance(model, SurfaceVariantModel) or not isinstance(request, SurfaceStructureRequest):
        raise TypeError("model/request 类型错误")
    candidates = tuple(item for item in model.patterns
                       if item.dialogue_act == request.dialogue_act
                       and item.register == request.register
                       and item.roles == request.ordered_roles)
    if not candidates:
        return _result(request, VARIANT_NO_PATTERN, ())
    if len(candidates) != 1:
        return _result(request, VARIANT_AMBIGUOUS, candidates)
    pattern = candidates[0]
    option_index = request.selection_ordinal % len(pattern.gap_options)
    option = pattern.gap_options[option_index]
    values: list[str] = []
    for index, gap in enumerate(option):
        values.append(gap)
        if index < len(pattern.roles):
            value = _request_value(request, pattern.roles[index], index)
            if value is None:
                return _result(request, VARIANT_NO_PATTERN, ())
            values.append(value)
    surface = "".join(values)
    scalars = tuple(ord(item) for item in surface)
    if not request.min_chars <= len(scalars) <= request.max_chars:
        return _result(request, VARIANT_NO_PATTERN, ())
    output = tuple(surface.encode("utf-8"))
    return _result(request, VARIANT_SELECTED, candidates,
                   selected_pattern_id=pattern.pattern_id,
                   selected_option_index=option_index,
                   output_scalars=scalars, output_bytes=output)


def _result(request: SurfaceStructureRequest, status: int,
            candidates: tuple[SurfaceVariantPattern, ...], **kwargs: Any) -> SurfaceVariantResult:
    values = [T1_G7_VARIANT_PROTOCOL_V1, status, len(candidates),
              *request.canonical_record()]
    values.extend(item.pattern_id for item in candidates)
    trace = integer_tuple_fingerprint(tuple(values), domain=_VARIANT_DOMAIN)
    return SurfaceVariantResult(request, status, len(candidates), trace=trace, **kwargs)


__all__ = [
    "T1_G7_VARIANT_PROTOCOL_V1", "VARIANT_AMBIGUOUS", "VARIANT_NO_PATTERN",
    "VARIANT_SELECTED", "SurfaceVariantLearningError", "SurfaceVariantModel",
    "SurfaceVariantPattern", "SurfaceVariantResult", "learn_surface_variant_model",
    "realize_surface_variants",
]
