"""DLG-RAW-16 G1：只读、确定性的表层候选选择适配器。

本模块故意不接入公开 terminal，也不调用理解、记忆、网络、LLM、SQLite 或
任何生成 factory。调用方必须先提供已经确认的语义命题、response-act、槽位
顺序和义务；selector 只在公开课程的 accepted candidates 中做严格过滤，然后
按显式 ``selection_ordinal`` 在规范排序后的合法候选中选择。

核心状态全部可以投影为整数与 UTF-8 ``u8``。Python 字符串只是 I/O 边界，
不参与对象 identity、哈希容器顺序或选择决策。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_schema import (
    DIALOGUE_ACTS,
    REGISTERS,
    ClauseSlot,
    SurfaceOrganizationRecord,
    SurfaceVariant,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    CompiledSurfaceOrganizationRecord,
)


DLG_RAW16_SURFACE_SELECTOR_PROTOCOL_V1 = 1
SURFACE_SELECTED = 1
SURFACE_NO_MATCH = 2
SURFACE_AMBIGUOUS_RECORD = 3
SURFACE_NO_LEGAL_VARIANT = 4

SURFACE_SELECTOR_STATUS_NAMES = {
    SURFACE_SELECTED: "SELECTED",
    SURFACE_NO_MATCH: "NO_MATCH",
    SURFACE_AMBIGUOUS_RECORD: "AMBIGUOUS_RECORD",
    SURFACE_NO_LEGAL_VARIANT: "NO_LEGAL_VARIANT",
}
_SURFACE_SELECTOR_TRACE_DOMAIN = (
    "pure_integer_ai.dlg_raw16.surface_selector.trace.v1")


class SurfaceSelectorError(ValueError):
    """DLG-RAW-16 selector 的输入、课程或不变量不满足合同。"""


def _strict_int(value: Any, where: str, *, nonnegative: bool = False) -> int:
    if type(value) is not int or (value < 0 if nonnegative else value <= 0):
        expected = "非负" if nonnegative else "正"
        raise SurfaceSelectorError(f"{where} 必须是{expected}严格整数")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SurfaceSelectorError(f"{where} 必须是无首尾空白的非空字符串")
    # UTF-16 surrogate code points are not Unicode scalar values and cannot be
    # part of the portable surface contract.
    if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
        raise SurfaceSelectorError(f"{where} 含非 Unicode scalar")
    return value


def _scalars(value: str, where: str) -> tuple[int, ...]:
    value = _text(value, where)
    return tuple(ord(item) for item in value)


def _u8(value: str, where: str) -> tuple[int, ...]:
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:  # defensive; _text rejects surrogates
        raise SurfaceSelectorError(f"{where} UTF-8 编码失败") from error
    return tuple(encoded)


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return (len(value), *value)


def _pack_optional_text(value: str | None, where: str) -> tuple[int, ...]:
    if value is None:
        return (0,)
    return (1, *_pack(_scalars(value, where)))


def _tuple_text(value: Any, where: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise SurfaceSelectorError(f"{where} 必须是字符串 tuple")
    result = tuple(_text(item, f"{where}[]") for item in value)
    if len(result) != len(set(result)):
        raise SurfaceSelectorError(f"{where} 不得重复")
    return result


def _slot_ids(value: tuple[ClauseSlot, ...], where: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise SurfaceSelectorError(f"{where} 不能为空")
    if any(not isinstance(item, ClauseSlot) for item in value):
        raise SurfaceSelectorError(f"{where} 含非法 ClauseSlot")
    ids = tuple(item.slot_id for item in value)
    if len(ids) != len(set(ids)):
        raise SurfaceSelectorError(f"{where} slot_id 不得重复")
    orders = tuple(item.order for item in value)
    if orders != tuple(range(1, len(value) + 1)):
        raise SurfaceSelectorError(f"{where} order 必须从 1 连续递增")
    return ids


def _variant_key(item: SurfaceVariant) -> tuple[Any, ...]:
    """只用显式 scalar/u8 内容建立跨语言稳定排序键。"""
    return (
        _scalars(item.variant_id, "surface.variant_id"),
        _scalars(item.surface, "surface.surface"),
        _scalars(item.register, "surface.register"),
        tuple(_scalars(value, "surface.proposition_ids[]")
              for value in item.proposition_ids),
        tuple(_scalars(value, "surface.clause_order[]")
              for value in item.clause_order),
    )


@dataclass(frozen=True, slots=True)
class SurfaceSemantic:
    """已由理解/证据侧确认的单一命题，不在 selector 内推断。"""

    proposition_id: str
    kind: str
    subject: str
    predicate: str
    object: str

    def __post_init__(self) -> None:
        for name in ("proposition_id", "kind", "subject", "predicate", "object"):
            _text(getattr(self, name), f"semantic.{name}")

    def canonical_record(self) -> tuple[int, ...]:
        result = [DLG_RAW16_SURFACE_SELECTOR_PROTOCOL_V1]
        for value in (self.proposition_id, self.kind, self.subject,
                      self.predicate, self.object):
            result.extend(_pack(_scalars(value, "semantic")))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class SurfaceSelectionRequest:
    """G1 selector 输入：语义、response-act、顺序槽位和明确预算。"""

    dialogue_act: str
    semantic: SurfaceSemantic
    ordered_clause_slots: tuple[str, ...]
    register: str
    required_proposition_ids: tuple[str, ...]
    forbidden_proposition_ids: tuple[str, ...] = ()
    min_chars: int = 1
    max_chars: int = 4096
    selection_ordinal: int = 0
    source_id: str | None = None
    context_id: str | None = None
    family_id: str | None = None
    template_family: str | None = None

    def __post_init__(self) -> None:
        if self.dialogue_act not in DIALOGUE_ACTS:
            raise SurfaceSelectorError("request.dialogue_act 未注册")
        if not isinstance(self.semantic, SurfaceSemantic):
            raise TypeError("request.semantic 类型错误")
        _tuple_text(self.ordered_clause_slots, "request.ordered_clause_slots")
        _tuple_text(self.required_proposition_ids,
                    "request.required_proposition_ids", allow_empty=True)
        _tuple_text(self.forbidden_proposition_ids,
                    "request.forbidden_proposition_ids", allow_empty=True)
        if self.register not in REGISTERS:
            raise SurfaceSelectorError("request.register 未注册")
        _strict_int(self.min_chars, "request.min_chars")
        _strict_int(self.max_chars, "request.max_chars")
        if self.max_chars < self.min_chars:
            raise SurfaceSelectorError("request length budget 倒置")
        _strict_int(self.selection_ordinal,
                    "request.selection_ordinal", nonnegative=True)
        for name in ("source_id", "context_id", "family_id", "template_family"):
            value = getattr(self, name)
            if value is not None:
                _text(value, f"request.{name}")
        # ANSWER must never omit the proposition it claims to answer.
        if (self.dialogue_act == "ANSWER"
                and self.required_proposition_ids !=
                (self.semantic.proposition_id,)):
            raise SurfaceSelectorError(
                "ANSWER request.required_proposition_ids 必须唯一绑定 semantic")
        if set(self.required_proposition_ids) & set(self.forbidden_proposition_ids):
            raise SurfaceSelectorError("required/forbidden proposition 交集非空")

    def canonical_record(self) -> tuple[int, ...]:
        result = [DLG_RAW16_SURFACE_SELECTOR_PROTOCOL_V1]
        result.extend(_pack(_scalars(self.dialogue_act, "request.dialogue_act")))
        result.extend(_pack(self.semantic.canonical_record()))
        result.extend(_pack(tuple(
            value for item in self.ordered_clause_slots
            for value in (len(_scalars(item, "request.slot")),
                          *_scalars(item, "request.slot")))))
        result.extend(_pack(_scalars(self.register, "request.register")))
        for values in (self.required_proposition_ids,
                       self.forbidden_proposition_ids):
            result.append(len(values))
            for value in values:
                result.extend(_pack(_scalars(value, "request.proposition_id")))
        result.extend((self.min_chars, self.max_chars, self.selection_ordinal))
        for value in (self.source_id, self.context_id, self.family_id,
                      self.template_family):
            result.extend(_pack_optional_text(value, "request.optional"))
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-16
@dataclass(frozen=True, slots=True)
class SurfaceSelectionResult:
    """一次只读选择的完整整数/u8 可重放结果。"""

    request: SurfaceSelectionRequest
    status_code: int
    candidate_count: int
    selected_index: int
    record_id: str | None = None
    variant_id: str | None = None
    output_scalars: tuple[int, ...] = ()
    output_bytes: tuple[int, ...] = ()
    trace: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.request, SurfaceSelectionRequest):
            raise TypeError("selection result request 类型错误")
        if self.status_code not in SURFACE_SELECTOR_STATUS_NAMES:
            raise SurfaceSelectorError("selection status 未注册")
        _strict_int(self.candidate_count, "result.candidate_count", nonnegative=True)
        _strict_int(self.selected_index, "result.selected_index", nonnegative=True)
        if not isinstance(self.trace, tuple) or not self.trace or any(
                type(item) is not int or item < 0 for item in self.trace):
            raise SurfaceSelectorError("result.trace 必须是非空非负整数 tuple")
        if self.status_code == SURFACE_SELECTED:
            if (self.candidate_count < 1
                    or self.selected_index >= self.candidate_count
                    or self.record_id is None or self.variant_id is None
                    or not self.output_scalars or not self.output_bytes):
                raise SurfaceSelectorError("SELECTED result 不完整")
            _text(self.record_id, "result.record_id")
            _text(self.variant_id, "result.variant_id")
            if any(type(item) is not int or item < 0 or item > 0x10FFFF
                   for item in self.output_scalars):
                raise SurfaceSelectorError("result.output_scalars 非法")
            if any(type(item) is not int or item < 0 or item > 255
                   for item in self.output_bytes):
                raise SurfaceSelectorError("result.output_bytes 非法")
            expected = tuple(
                item for item in self.output_scalars
                if not 0xD800 <= item <= 0xDFFF)
            if expected != self.output_scalars:
                raise SurfaceSelectorError("result.output_scalars 含 surrogate")
            try:
                if tuple("".join(chr(item) for item in self.output_scalars)
                         .encode("utf-8")) != self.output_bytes:
                    raise SurfaceSelectorError("result scalar/u8 不一致")
            except (ValueError, UnicodeEncodeError) as error:
                raise SurfaceSelectorError("result scalar 无法 UTF-8 回读") from error
        else:
            if (self.record_id is not None or self.variant_id is not None
                    or self.output_scalars or self.output_bytes
                    or self.selected_index != 0):
                raise SurfaceSelectorError("非 SELECTED result 不得携带输出")

    @property
    def status(self) -> str:
        return SURFACE_SELECTOR_STATUS_NAMES[self.status_code]

    @property
    def surface(self) -> str | None:
        if self.status_code != SURFACE_SELECTED:
            return None
        return "".join(chr(item) for item in self.output_scalars)

    def canonical_record(self) -> tuple[int, ...]:
        result = [
            DLG_RAW16_SURFACE_SELECTOR_PROTOCOL_V1,
            self.status_code,
            self.candidate_count,
            self.selected_index,
        ]
        result.extend(_pack(self.request.canonical_record()))
        for value in (self.record_id, self.variant_id):
            result.extend(_pack_optional_text(value, "result.identity"))
        result.extend(_pack(self.output_scalars))
        result.extend(_pack(self.output_bytes))
        result.extend(_pack(self.trace))
        return tuple(result)


def _record_semantic(record: SurfaceOrganizationRecord) -> SurfaceSemantic:
    return SurfaceSemantic(
        record.proposition_id,
        record.proposition_kind,
        record.proposition_subject,
        record.proposition_predicate,
        record.proposition_object,
    )


def _record_key(record: SurfaceOrganizationRecord) -> tuple[Any, ...]:
    return (
        _scalars(record.sample_id, "record.sample_id"),
        _scalars(record.source_id, "record.source_id"),
        _scalars(record.context_id, "record.context_id"),
        _scalars(record.family_id, "record.family_id"),
        _scalars(record.template_family, "record.template_family"),
    )


def _candidate_is_legal(
        record: SurfaceOrganizationRecord,
        variant: SurfaceVariant,
        request: SurfaceSelectionRequest,
        slot_ids: tuple[str, ...],
        ) -> bool:
    """只依据已冻结 obligations 判断 accepted candidate 是否可输出。"""
    if variant.violations:
        return False
    if variant.proposition_ids != request.required_proposition_ids:
        return False
    if variant.clause_order != slot_ids:
        return False
    if variant.register != request.register:
        return False
    scalars = _scalars(variant.surface, "accepted.surface")
    if not (record.min_chars <= len(scalars) <= record.max_chars):
        return False
    if not (request.min_chars <= len(scalars) <= request.max_chars):
        return False
    # An UNKNOWN/CLARIFY/REPAIR surface cannot smuggle a proposition claim.
    if record.dialogue_act in {"UNKNOWN", "CLARIFY", "REPAIR"}:
        if variant.proposition_ids or set(variant.proposition_ids) & set(
                request.forbidden_proposition_ids):
            return False
    return True


# object-model: value; representation=struct; interop=DLG-RAW-16
@dataclass(frozen=True, slots=True)
class SurfaceOrganizationSelector:
    """DLG-RAW-16 G1 immutable course selector；不改变默认 terminal。"""

    records: tuple[SurfaceOrganizationRecord, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.records, tuple) or not self.records
                or any(type(item) is not SurfaceOrganizationRecord
                       for item in self.records)):
            raise SurfaceSelectorError("selector records 必须是非空 record tuple")
        if len({item.sample_id for item in self.records}) != len(self.records):
            raise SurfaceSelectorError("selector sample_id 必须唯一")
        ordered = tuple(sorted(self.records, key=_record_key))
        object.__setattr__(self, "records", ordered)

    def select(self, request: SurfaceSelectionRequest) -> SurfaceSelectionResult:
        """严格过滤 record/candidate，并按 ordinal 选择一个合法 surface。"""
        if not isinstance(request, SurfaceSelectionRequest):
            raise TypeError("surface selector request 类型错误")
        records = tuple(
            record for record in self.records
            if (record.dialogue_act == request.dialogue_act
                and _record_semantic(record) == request.semantic
                and tuple(item.slot_id for item in record.clause_slots)
                == request.ordered_clause_slots
                and record.register == request.register
                and record.required_proposition_ids
                == request.required_proposition_ids
                and record.forbidden_proposition_ids
                == request.forbidden_proposition_ids
                and (request.source_id is None
                     or record.source_id == request.source_id)
                and (request.context_id is None
                     or record.context_id == request.context_id)
                and (request.family_id is None
                     or record.family_id == request.family_id)
                and (request.template_family is None
                     or record.template_family == request.template_family))
        )
        if not records:
            return self._result(request, SURFACE_NO_MATCH, (), ())
        if len(records) != 1:
            return self._result(request, SURFACE_AMBIGUOUS_RECORD, (), records)
        record = records[0]
        slot_ids = _slot_ids(record.clause_slots, "record.clause_slots")
        legal = tuple(sorted(
            (variant for variant in record.accepted
             if _candidate_is_legal(record, variant, request, slot_ids)),
            key=_variant_key,
        ))
        if not legal:
            return self._result(request, SURFACE_NO_LEGAL_VARIANT, (), (record,))
        selected_index = request.selection_ordinal % len(legal)
        selected = legal[selected_index]
        scalars = _scalars(selected.surface, "selected.surface")
        output = _u8(selected.surface, "selected.surface")
        return self._result(
            request,
            SURFACE_SELECTED,
            legal,
            (record,),
            record_id=record.sample_id,
            variant_id=selected.variant_id,
            output_scalars=scalars,
            output_bytes=output,
            selected_index=selected_index,
        )

    def _result(
            self,
            request: SurfaceSelectionRequest,
            status_code: int,
            candidates: tuple[SurfaceVariant, ...],
            records: tuple[SurfaceOrganizationRecord, ...],
            *,
            record_id: str | None = None,
            variant_id: str | None = None,
            output_scalars: tuple[int, ...] = (),
            output_bytes: tuple[int, ...] = (),
            selected_index: int = 0,
            ) -> SurfaceSelectionResult:
        if records and isinstance(records[0], SurfaceOrganizationRecord):
            # For an ambiguous result, include all record identities; for a
            # selected result, include the legal candidate keys as well.
            record_values = tuple(
                value
                for record in records
                for value in _pack(_scalars(record.sample_id, "record.sample_id"))
            )
        else:
            record_values = ()
        candidate_values = tuple(
            value
            for item in candidates
            for value in _pack(_scalars(item.variant_id, "variant.variant_id"))
        )
        # ``candidate_count`` is the number of selectable alternatives for a
        # successful result; on record ambiguity it is the number of matching
        # records, so callers can distinguish an empty miss from an unresolved
        # identity without inspecting Python objects.
        candidate_count = (
            len(records) if status_code == SURFACE_AMBIGUOUS_RECORD
            else len(candidates))
        trace_values = (
            status_code,
            candidate_count,
            selected_index,
            *request.canonical_record(),
            *record_values,
            *candidate_values,
        )
        trace = integer_tuple_fingerprint(
            tuple(trace_values), domain=_SURFACE_SELECTOR_TRACE_DOMAIN)
        return SurfaceSelectionResult(
            request,
            status_code,
            candidate_count,
            selected_index if status_code == SURFACE_SELECTED else 0,
            record_id,
            variant_id,
            output_scalars,
            output_bytes,
            trace,
        )


def build_surface_organization_selector(
        values: Iterable[SurfaceOrganizationRecord | CompiledSurfaceOrganizationRecord],
        ) -> SurfaceOrganizationSelector:
    """从 raw/compiled G0 course 建立不可变 G1 selector。"""
    records: list[SurfaceOrganizationRecord] = []
    for index, value in enumerate(values):
        if isinstance(value, SurfaceOrganizationRecord):
            records.append(value)
        elif isinstance(value, CompiledSurfaceOrganizationRecord):
            if value.integer_record != value.record.canonical_integer_record:
                raise SurfaceSelectorError(
                    f"selector values[{index}] integer projection 漂移")
            records.append(value.record)
        else:
            raise TypeError(f"selector values[{index}] 类型错误")
    return SurfaceOrganizationSelector(tuple(records))


__all__ = [
    "DLG_RAW16_SURFACE_SELECTOR_PROTOCOL_V1",
    "SURFACE_AMBIGUOUS_RECORD",
    "SURFACE_NO_LEGAL_VARIANT",
    "SURFACE_NO_MATCH",
    "SURFACE_SELECTED",
    "SURFACE_SELECTOR_STATUS_NAMES",
    "SurfaceOrganizationSelector",
    "SurfaceSelectionRequest",
    "SurfaceSelectionResult",
    "SurfaceSemantic",
    "SurfaceSelectorError",
    "build_surface_organization_selector",
]
