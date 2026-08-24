"""DLG-RAW-16 G2A：带证据的表层结构学习与 held-out 重建。

该模块把“学习一句话的组织方式”和“复制一条完整句子”分开。训练输入必须
为 G0 accepted surface 的显式 slot/span evidence；学习器只保存 literal gap、
slot role/order、response-act 和语域，不从文本猜槽位，也不保存训练实体。
运行输入是新的 typed semantic plan，输出可重放的 scalar/u8/integer trace。
这是可迁移的纯值协议，不调用 LLM、网络、SQLite 或默认 terminal。
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
    SurfaceOrganizationRecord,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)


DLG_RAW16_STRUCTURE_PROTOCOL_V1 = 1
EVIDENCE_RECORD_KIND = "DLG_RAW16_SURFACE_SLOT_EVIDENCE_V1"
STRUCTURE_SELECTED = 1
STRUCTURE_NO_PATTERN = 2
STRUCTURE_AMBIGUOUS = 3
STRUCTURE_SELECTOR_DOMAIN = "pure_integer_ai.dlg_raw16.structure.v1"


class SurfaceStructureLearningError(ValueError):
    """slot evidence、学习模型或 held-out 请求不满足合同。"""


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SurfaceStructureLearningError(
            f"{where} 必须是无首尾空白的非空字符串")
    if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
        raise SurfaceStructureLearningError(f"{where} 含非 Unicode scalar")
    return value


def _positive(value: Any, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise SurfaceStructureLearningError(f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, where: str) -> int:
    if type(value) is not int or value < 0:
        raise SurfaceStructureLearningError(f"{where} 必须是非负严格整数")
    return value


def _pack_text(value: str, where: str) -> tuple[int, ...]:
    scalars = tuple(ord(item) for item in _text(value, where))
    return (len(scalars), *scalars)


def _pack_texts(values: tuple[str, ...], where: str) -> tuple[int, ...]:
    result = [len(values)]
    for index, value in enumerate(values):
        result.extend(_pack_text(value, f"{where}[{index}]"))
    return tuple(result)


def _pack(values: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(values, tuple) or any(type(item) is not int or item < 0
                                             for item in values):
        raise SurfaceStructureLearningError("整数 record 不是非负 tuple")
    return (len(values), *values)


@dataclass(frozen=True, slots=True)
class SurfaceSlotEvidence:
    """一个 accepted surface 中的 typed slot span。"""

    record_id: str
    variant_id: str
    slot_id: str
    role: str
    start: int
    end: int
    surface_text: str = ""

    def __post_init__(self) -> None:
        for name in ("record_id", "variant_id", "slot_id", "role"):
            _text(getattr(self, name), f"evidence.{name}")
        _nonnegative(self.start, "evidence.start")
        _positive(self.end, "evidence.end")
        if self.end <= self.start:
            raise SurfaceStructureLearningError("evidence.end 必须大于 start")
        if self.surface_text:
            _text(self.surface_text, "evidence.surface_text")

    def canonical_record(self) -> tuple[int, ...]:
        result = [DLG_RAW16_STRUCTURE_PROTOCOL_V1]
        for value in (self.record_id, self.variant_id, self.slot_id, self.role):
            result.extend(_pack_text(value, "evidence.text"))
        result.extend((self.start, self.end))
        result.extend(_pack_text(self.surface_text, "evidence.surface_text")
                      if self.surface_text else (0,))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class SurfaceEvidencePack:
    """训练证据包；不携带 hidden evaluator label。"""

    source_namespace: str
    license_id: str
    entries: tuple[SurfaceSlotEvidence, ...]

    def __post_init__(self) -> None:
        _text(self.source_namespace, "pack.source_namespace")
        if self.license_id != "CC0-1.0":
            raise SurfaceStructureLearningError("pack 必须是 CC0-1.0")
        if (not isinstance(self.entries, tuple) or not self.entries
                or any(not isinstance(item, SurfaceSlotEvidence)
                       for item in self.entries)):
            raise SurfaceStructureLearningError("pack.entries 不能为空")
        keys = tuple((item.record_id, item.variant_id, item.slot_id)
                     for item in self.entries)
        if len(keys) != len(set(keys)):
            raise SurfaceStructureLearningError("slot evidence key 重复")

    def canonical_record(self) -> tuple[int, ...]:
        result = [DLG_RAW16_STRUCTURE_PROTOCOL_V1]
        result.extend(_pack_text(self.source_namespace, "pack.namespace"))
        result.extend(_pack_text(self.license_id, "pack.license"))
        for item in self.entries:
            entry = item.canonical_record()
            result.extend((len(entry), *entry))
        return tuple(result)


def load_surface_evidence_jsonl(payload: bytes) -> SurfaceEvidencePack:
    """严格回读公开 slot evidence JSONL；不从表面字符串推导 span。"""
    if type(payload) is not bytes or not payload or not payload.endswith(b"\n"):
        raise SurfaceStructureLearningError(
            "evidence JSONL 必须是非空 bytes 并以换行结束")
    rows = payload.splitlines(keepends=True)
    fields = {"record_kind", "schema_version", "license_id",
              "source_namespace", "record_id", "variant_id", "slot_id",
              "role", "start", "end"}
    fields_with_surface_text = {*fields, "surface_text"}
    entries = []
    namespace = None
    license_id = None
    for index, line in enumerate(rows, start=1):
        if line == b"\n" or not line.endswith(b"\n"):
            raise SurfaceStructureLearningError(
                f"evidence 第 {index} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except Exception as error:
            raise SurfaceStructureLearningError(
                f"evidence 第 {index} 行不是规范 JSON") from error
        if (canonical_json_line(value) != line
                or set(value) not in (fields, fields_with_surface_text)):
            raise SurfaceStructureLearningError(
                f"evidence 第 {index} 行字段或规范字节漂移")
        if (value["record_kind"] != EVIDENCE_RECORD_KIND
                or value["schema_version"] != 1):
            raise SurfaceStructureLearningError(
                "evidence record kind/version 非法")
        row_namespace = _text(value["source_namespace"],
                               "evidence.source_namespace")
        row_license = _text(value["license_id"], "evidence.license_id")
        if row_license != "CC0-1.0":
            raise SurfaceStructureLearningError("evidence 必须是 CC0-1.0")
        if namespace is None:
            namespace, license_id = row_namespace, row_license
        elif (row_namespace, row_license) != (namespace, license_id):
            raise SurfaceStructureLearningError("evidence namespace/license 漂移")
        _nonnegative(value["start"], "evidence.start")
        _nonnegative(value["end"], "evidence.end")
        entries.append(SurfaceSlotEvidence(
            _text(value["record_id"], "evidence.record_id"),
            _text(value["variant_id"], "evidence.variant_id"),
            _text(value["slot_id"], "evidence.slot_id"),
            _text(value["role"], "evidence.role"),
            value["start"], value["end"],
            value.get("surface_text", ""),
        ))
    if namespace is None or license_id is None:
        raise SurfaceStructureLearningError("evidence JSONL 没有记录")
    return SurfaceEvidencePack(namespace, license_id, tuple(entries))


def _role_value(record: SurfaceOrganizationRecord, role: str) -> str:
    """把受限公开 role 映射到已确认命题字段；不使用表面猜测。"""
    if role in {"subject", "topic", "cause"}:
        return record.proposition_subject
    if role in {"predicate", "relation"}:
        return record.proposition_predicate
    if role in {"object", "claim", "effect"}:
        return record.proposition_object
    raise SurfaceStructureLearningError(f"不支持的结构 slot role: {role}")


def _semantic_role_value(semantic: SurfaceSemantic, role: str) -> str | None:
    """返回可由 semantic 直接授权的表层值；非语义限定槽返回 None。"""
    if role in {"subject", "topic", "cause"}:
        return semantic.subject
    if role in {"predicate", "relation"}:
        return semantic.predicate
    if role in {"object", "claim", "effect"}:
        return semantic.object
    return None


def _validate_entry(
        record: SurfaceOrganizationRecord,
        variant,
        entries: tuple[SurfaceSlotEvidence, ...],
        ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """验证 spans 与权威 proposition 字段一致，并提取 gaps/roles。"""
    expected_slots = tuple(item.slot_id for item in record.clause_slots)
    by_slot = {item.slot_id: item for item in entries}
    if (not entries or tuple(item.slot_id for item in entries) != expected_slots
            or set(by_slot) != set(expected_slots)):
        raise SurfaceStructureLearningError("slot evidence 未完整覆盖 ordered slots")
    if any(item.record_id != record.sample_id or item.variant_id != variant.variant_id
           for item in entries):
        raise SurfaceStructureLearningError("slot evidence identity 漂移")
    spans = []
    for item in entries:
        if item.role != next(slot.role for slot in record.clause_slots
                             if slot.slot_id == item.slot_id):
            raise SurfaceStructureLearningError("slot evidence role 漂移")
        if item.end > len(variant.surface):
            raise SurfaceStructureLearningError("slot evidence 越过 surface")
        value = (item.surface_text if item.surface_text
                 else _role_value(record, item.role))
        if variant.surface[item.start:item.end] != value:
            raise SurfaceStructureLearningError("slot span 未绑定 proposition 字段")
        spans.append((item.start, item.end))
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise SurfaceStructureLearningError("slot spans 重叠或未按表面顺序排列")
    gaps = [variant.surface[:spans[0][0]]]
    for left, right in zip(spans, spans[1:]):
        gaps.append(variant.surface[left[1]:right[0]])
    gaps.append(variant.surface[spans[-1][1]:])
    roles = tuple(item.role for item in entries)
    return tuple(gaps), roles


@dataclass(frozen=True, slots=True)
class SurfaceStructurePattern:
    """由至少两个独立训练 family 支持的 literal + typed slot 结构。"""

    pattern_id: int
    dialogue_act: str
    register: str
    roles: tuple[str, ...]
    gaps: tuple[str, ...]
    support_record_ids: tuple[str, ...]
    support_family_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _positive(self.pattern_id, "pattern.pattern_id")
        if self.dialogue_act not in DIALOGUE_ACTS or self.register not in REGISTERS:
            raise SurfaceStructureLearningError("pattern act/register 非法")
        if not self.roles or len(self.gaps) != len(self.roles) + 1:
            raise SurfaceStructureLearningError("pattern roles/gaps 形状非法")
        if len(self.support_family_ids) < 2:
            raise SurfaceStructureLearningError("pattern 必须由两个独立 family 支持")
        if not self.support_record_ids or tuple(sorted(set(self.support_record_ids))) != self.support_record_ids:
            raise SurfaceStructureLearningError("pattern support records 非规范")
        if tuple(sorted(set(self.support_family_ids))) != self.support_family_ids:
            raise SurfaceStructureLearningError("pattern support families 非规范")

    def canonical_record(self) -> tuple[int, ...]:
        result = [DLG_RAW16_STRUCTURE_PROTOCOL_V1, self.pattern_id]
        for value in (self.dialogue_act, self.register):
            result.extend(_pack_text(value, "pattern.text"))
        result.extend(_pack_texts(self.roles, "pattern.roles"))
        result.extend(_pack_texts(self.gaps, "pattern.gaps"))
        result.extend(_pack_texts(self.support_record_ids, "pattern.records"))
        result.extend(_pack_texts(self.support_family_ids, "pattern.families"))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class SurfaceStructureModel:
    patterns: tuple[SurfaceStructurePattern, ...]

    def __post_init__(self) -> None:
        if (not self.patterns
                or tuple(sorted(self.patterns, key=lambda x: x.pattern_id))
                != self.patterns
                or len({item.pattern_id for item in self.patterns})
                != len(self.patterns)):
            raise SurfaceStructureLearningError("model patterns 必须按 id 排序且非空")


@dataclass(frozen=True, slots=True)
class SurfaceStructureRequest:
    semantic: SurfaceSemantic
    dialogue_act: str
    register: str
    ordered_roles: tuple[str, ...]
    min_chars: int = 1
    max_chars: int = 4096
    selection_ordinal: int = 0
    source_id: str = ""
    context_id: str = ""
    family_id: str = ""
    # Non-semantic dialogue acts (for example CLARIFY) may carry explicitly
    # typed surface slot values.  Keeping this at the end preserves all prior
    # positional constructors and keeps the semantic proposition authoritative
    # for ANSWER.
    slot_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.semantic, SurfaceSemantic):
            raise TypeError("request.semantic 类型错误")
        if self.dialogue_act not in DIALOGUE_ACTS or self.register not in REGISTERS:
            raise SurfaceStructureLearningError("request act/register 非法")
        if not self.ordered_roles or any(not isinstance(item, str) or not item for item in self.ordered_roles):
            raise SurfaceStructureLearningError("request.ordered_roles 非法")
        _positive(self.min_chars, "request.min_chars")
        _positive(self.max_chars, "request.max_chars")
        _nonnegative(self.selection_ordinal, "request.selection_ordinal")
        if self.max_chars < self.min_chars:
            raise SurfaceStructureLearningError("request budget 倒置")
        for name in ("source_id", "context_id", "family_id"):
            _text(getattr(self, name), f"request.{name}")
        if not isinstance(self.slot_values, tuple):
            raise SurfaceStructureLearningError("request.slot_values 必须是 tuple")
        if self.slot_values and len(self.slot_values) != len(self.ordered_roles):
            raise SurfaceStructureLearningError(
                "request.slot_values 必须与 ordered_roles 一一对应")
        for index, value in enumerate(self.slot_values):
            _text(value, f"request.slot_values[{index}]")
        for index, role in enumerate(self.ordered_roles):
            semantic_value = _semantic_role_value(self.semantic, role)
            if (semantic_value is not None and self.slot_values
                    and self.slot_values[index] != semantic_value):
                raise SurfaceStructureLearningError(
                    f"request.slot_values[{index}] 绕过 semantic.{role}")
        if self.dialogue_act == "CLARIFY" and not self.slot_values:
            raise SurfaceStructureLearningError(
                "CLARIFY request 必须显式提供 slot_values")

    def canonical_record(self) -> tuple[int, ...]:
        result = [DLG_RAW16_STRUCTURE_PROTOCOL_V1]
        result.extend(_pack_text(self.dialogue_act, "request.act"))
        result.extend(_pack_text(self.register, "request.register"))
        result.extend(_pack_texts(self.ordered_roles, "request.roles"))
        result.extend((self.min_chars, self.max_chars, self.selection_ordinal))
        for value in (self.source_id, self.context_id, self.family_id):
            result.extend(_pack_text(value, "request.identity"))
        result.extend(_pack_texts(self.slot_values, "request.slot_values")
                      if self.slot_values else (0,))
        result.extend(_pack(self.semantic.canonical_record()))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class SurfaceStructureResult:
    request: SurfaceStructureRequest
    status_code: int
    candidate_count: int
    selected_pattern_id: int = 0
    output_scalars: tuple[int, ...] = ()
    output_bytes: tuple[int, ...] = ()
    trace: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.status_code not in {STRUCTURE_SELECTED, STRUCTURE_NO_PATTERN, STRUCTURE_AMBIGUOUS}:
            raise SurfaceStructureLearningError("result status 未注册")
        _nonnegative(self.candidate_count, "result.candidate_count")
        _nonnegative(self.selected_pattern_id, "result.selected_pattern_id")
        if not self.trace or any(type(item) is not int or item < 0 for item in self.trace):
            raise SurfaceStructureLearningError("result trace 非法")
        if self.status_code == STRUCTURE_SELECTED:
            if self.candidate_count < 1 or not self.selected_pattern_id or not self.output_scalars:
                raise SurfaceStructureLearningError("selected result 不完整")
            if any(type(item) is not int or item < 0 or item > 0x10FFFF
                   or 0xD800 <= item <= 0xDFFF for item in self.output_scalars):
                raise SurfaceStructureLearningError("result scalars 非法")
            if tuple("".join(chr(item) for item in self.output_scalars).encode("utf-8")) != self.output_bytes:
                raise SurfaceStructureLearningError("result scalar/u8 漂移")
        elif self.selected_pattern_id or self.output_scalars or self.output_bytes:
            raise SurfaceStructureLearningError("拒绝 result 不得携带输出")

    @property
    def surface(self) -> str | None:
        return ("".join(chr(item) for item in self.output_scalars)
                if self.status_code == STRUCTURE_SELECTED else None)

    def canonical_record(self) -> tuple[int, ...]:
        result = [DLG_RAW16_STRUCTURE_PROTOCOL_V1, self.status_code,
                  self.candidate_count, self.selected_pattern_id]
        result.extend(_pack(self.request.canonical_record()))
        result.extend(_pack(self.output_scalars))
        result.extend(_pack(self.output_bytes))
        result.extend(_pack(self.trace))
        return tuple(result)


def _pattern_id(dialogue_act: str, register: str, roles: tuple[str, ...], gaps: tuple[str, ...]) -> int:
    # Explicit scalar packing keeps the identity reproducible outside Python;
    # repr/list formatting is intentionally not part of the wire protocol.
    values = [DLG_RAW16_STRUCTURE_PROTOCOL_V1]
    for value in (dialogue_act, register):
        scalars = tuple(ord(item) for item in value)
        values.extend((len(scalars), *scalars))
    for group in (roles, gaps):
        values.append(len(group))
        for value in group:
            scalars = tuple(ord(item) for item in value)
            values.extend((len(scalars), *scalars))
    raw = integer_tuple_fingerprint(
        tuple(values), domain=STRUCTURE_SELECTOR_DOMAIN)
    value = 0
    for byte in raw[2:]:
        value = (value << 8) | byte
    return value if value > 0 else 1


def learn_surface_structure_model(
        records: Iterable[SurfaceOrganizationRecord],
        evidence_pack: SurfaceEvidencePack,
        ) -> SurfaceStructureModel:
    """从显式证据学习结构；同一结构必须跨两个独立 family 出现。"""
    rows = tuple(records)
    if not rows or any(not isinstance(item, SurfaceOrganizationRecord) for item in rows):
        raise TypeError("records 必须是非空 SurfaceOrganizationRecord tuple")
    if not isinstance(evidence_pack, SurfaceEvidencePack):
        raise TypeError("evidence_pack 类型错误")
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in rows:
        variants = {item.variant_id: item for item in record.accepted}
        for variant_id in sorted(variants):
            entries = tuple(item for item in evidence_pack.entries
                            if item.record_id == record.sample_id
                            and item.variant_id == variant_id)
            if not entries:
                continue
            gaps, roles = _validate_entry(record, variants[variant_id], entries)
            key = (record.dialogue_act, record.register, roles, gaps)
            state = by_key.setdefault(key, {"records": set(), "families": set()})
            state["records"].add(record.sample_id)
            state["families"].add(record.family_id)
    patterns = []
    for (act, register, roles, gaps), state in by_key.items():
        if len(state["families"]) < 2:
            continue
        patterns.append(SurfaceStructurePattern(
            _pattern_id(act, register, roles, gaps), act, register, roles, gaps,
            tuple(sorted(state["records"])), tuple(sorted(state["families"]))))
    if not patterns:
        raise SurfaceStructureLearningError(
            "没有跨独立 family 的可学习结构；需要两个独立 family")
    return SurfaceStructureModel(tuple(sorted(patterns, key=lambda item: item.pattern_id)))


def _request_value(
        semantic: SurfaceSemantic,
        role: str,
        slot_values: tuple[str, ...],
        index: int,
        ) -> str:
    semantic_value = _semantic_role_value(semantic, role)
    if semantic_value is not None:
        return semantic_value
    if slot_values:
        return slot_values[index]
    raise SurfaceStructureLearningError(f"不支持的 request role: {role}")


def realize_surface_structure(
        model: SurfaceStructureModel,
        request: SurfaceStructureRequest,
        ) -> SurfaceStructureResult:
    """以新 typed semantic/source plan 填入已学结构，不回读训练实体。"""
    if not isinstance(model, SurfaceStructureModel) or not isinstance(request, SurfaceStructureRequest):
        raise TypeError("model/request 类型错误")
    candidates = tuple(item for item in model.patterns
                       if item.dialogue_act == request.dialogue_act
                       and item.register == request.register
                       and item.roles == request.ordered_roles)
    if not candidates:
        return _result(request, STRUCTURE_NO_PATTERN, ())
    if len(candidates) > 1:
        return _result(request, STRUCTURE_AMBIGUOUS, candidates)
    pattern = candidates[request.selection_ordinal % len(candidates)]
    values = []
    for index, gap in enumerate(pattern.gaps):
        values.append(gap)
        if index < len(pattern.roles):
            try:
                values.append(_request_value(
                    request.semantic, pattern.roles[index],
                    request.slot_values, index))
            except SurfaceStructureLearningError:
                # A matching pattern with an unauthorized non-semantic slot is
                # a normal fail-closed runtime outcome, not a host exception.
                return _result(request, STRUCTURE_NO_PATTERN, ())
    surface = "".join(values)
    scalars = tuple(ord(item) for item in surface)
    if not surface or not request.min_chars <= len(scalars) <= request.max_chars:
        return _result(request, STRUCTURE_NO_PATTERN, ())
    output = tuple(surface.encode("utf-8"))
    return _result(request, STRUCTURE_SELECTED, candidates,
                   selected_pattern_id=pattern.pattern_id,
                   output_scalars=scalars, output_bytes=output)


def _result(request: SurfaceStructureRequest, status: int, candidates: tuple[SurfaceStructurePattern, ...], **kwargs: Any) -> SurfaceStructureResult:
    values = [DLG_RAW16_STRUCTURE_PROTOCOL_V1, status, len(candidates),
              *request.canonical_record()]
    values.extend(item.pattern_id for item in candidates)
    trace = integer_tuple_fingerprint(tuple(values), domain=STRUCTURE_SELECTOR_DOMAIN)
    return SurfaceStructureResult(request, status, len(candidates), trace=trace, **kwargs)


__all__ = [
    "DLG_RAW16_STRUCTURE_PROTOCOL_V1", "STRUCTURE_AMBIGUOUS", "STRUCTURE_NO_PATTERN",
    "STRUCTURE_SELECTED", "SurfaceEvidencePack", "SurfaceSlotEvidence",
    "SurfaceStructureLearningError", "SurfaceStructureModel", "SurfaceStructurePattern",
    "SurfaceStructureRequest", "SurfaceStructureResult", "learn_surface_structure_model",
    "realize_surface_structure", "load_surface_evidence_jsonl",
]
