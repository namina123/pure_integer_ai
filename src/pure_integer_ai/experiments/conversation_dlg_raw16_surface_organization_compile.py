"""DLG-RAW-16 表层组织课程的严格解析、编译与隔离校验。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_schema import (
    ClauseSlot,
    LICENSE_ID,
    RECORD_KIND,
    SCHEMA_VERSION,
    SurfaceOrganizationError,
    SurfaceOrganizationRecord,
    SurfaceVariant,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
)


_FIELDS = frozenset({
    "record_kind", "schema_version", "license_id", "sample_id", "source_id",
    "context_id", "family_id", "template_family", "owner", "split",
    "dialogue_act", "semantic_proposition", "obligation", "register",
    "length_budget", "accepted", "rejected",
})
_PROP_FIELDS = frozenset({"id", "kind", "subject", "predicate", "object"})
_OBL_FIELDS = frozenset({"required_proposition_ids", "forbidden_proposition_ids", "ordered_clause_slots"})
_SLOT_FIELDS = frozenset({"slot_id", "role", "order", "required"})
_VARIANT_FIELDS = frozenset({"variant_id", "surface", "proposition_ids", "clause_order", "register"})
_REJECT_FIELDS = frozenset({"variant_id", "surface", "proposition_ids", "clause_order", "register", "violations"})


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SurfaceOrganizationError(f"{where} 必须是无首尾空白的非空字符串")
    return value


def _text_list(value: Any, where: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise SurfaceOrganizationError(f"{where} 必须是字符串列表")
    result = tuple(_text(item, f"{where}[]") for item in value)
    if len(result) != len(set(result)):
        raise SurfaceOrganizationError(f"{where} 不得重复")
    return result


def _int(value: Any, where: str, *, positive: bool = False) -> int:
    if type(value) is not int or (value <= 0 if positive else value < 0):
        raise SurfaceOrganizationError(f"{where} 必须是严格整数")
    return value


def _surface_variant(value: Any, where: str, *, rejected: bool) -> SurfaceVariant:
    fields = _REJECT_FIELDS if rejected else _VARIANT_FIELDS
    if not isinstance(value, dict) or set(value) != fields:
        raise SurfaceOrganizationError(f"{where} 字段集合漂移")
    violations = _text_list(value.get("violations", []), f"{where}.violations", allow_empty=not rejected)
    return SurfaceVariant(
        _text(value["variant_id"], f"{where}.variant_id"),
        _text(value["surface"], f"{where}.surface"),
        _text_list(value["proposition_ids"], f"{where}.proposition_ids", allow_empty=True),
        _text_list(value["clause_order"], f"{where}.clause_order", allow_empty=True),
        _text(value["register"], f"{where}.register"),
        violations,
    )


def parse_surface_organization_record(value: Any) -> SurfaceOrganizationRecord:
    """严格恢复单条课程记录，并在恢复阶段执行语义与表层不变量。"""
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise SurfaceOrganizationError("DLG-RAW-16 记录字段集合漂移")
    if value["record_kind"] != RECORD_KIND or value["schema_version"] != SCHEMA_VERSION:
        raise SurfaceOrganizationError("DLG-RAW-16 record kind/version 不匹配")
    if value["license_id"] != LICENSE_ID:
        raise SurfaceOrganizationError("DLG-RAW-16 公开样例必须是 CC0-1.0")
    proposition = value["semantic_proposition"]
    if not isinstance(proposition, dict) or set(proposition) != _PROP_FIELDS:
        raise SurfaceOrganizationError("semantic_proposition 字段集合漂移")
    obligation = value["obligation"]
    if not isinstance(obligation, dict) or set(obligation) != _OBL_FIELDS:
        raise SurfaceOrganizationError("obligation 字段集合漂移")
    slots_raw = obligation["ordered_clause_slots"]
    if not isinstance(slots_raw, list) or not slots_raw:
        raise SurfaceOrganizationError("ordered_clause_slots 不能为空")
    slots = []
    for index, item in enumerate(slots_raw):
        if not isinstance(item, dict) or set(item) != _SLOT_FIELDS:
            raise SurfaceOrganizationError(f"ordered_clause_slots[{index}] 字段集合漂移")
        slots.append(ClauseSlot(
            _text(item["slot_id"], f"slot[{index}].slot_id"),
            _text(item["role"], f"slot[{index}].role"),
            _int(item["order"], f"slot[{index}].order", positive=True),
            item["required"],
        ))
    budget = value["length_budget"]
    if not isinstance(budget, dict) or set(budget) != {"min_chars", "max_chars"}:
        raise SurfaceOrganizationError("length_budget 字段集合漂移")
    accepted_raw = value["accepted"]
    rejected_raw = value["rejected"]
    if not isinstance(accepted_raw, list) or not isinstance(rejected_raw, list):
        raise SurfaceOrganizationError("accepted/rejected 必须为列表")
    return SurfaceOrganizationRecord(
        _text(value["sample_id"], "sample_id"), _text(value["source_id"], "source_id"),
        _text(value["context_id"], "context_id"), _text(value["family_id"], "family_id"),
        _text(value["template_family"], "template_family"), _text(value["owner"], "owner"),
        _text(value["split"], "split"), _text(value["dialogue_act"], "dialogue_act"),
        _text(proposition["id"], "proposition.id"), _text(proposition["kind"], "proposition.kind"),
        _text(proposition["subject"], "proposition.subject"), _text(proposition["predicate"], "proposition.predicate"),
        _text(proposition["object"], "proposition.object"),
        _text_list(obligation["required_proposition_ids"], "required_proposition_ids", allow_empty=True),
        _text_list(obligation["forbidden_proposition_ids"], "forbidden_proposition_ids", allow_empty=True),
        tuple(slots), _text(value["register"], "register"),
        _int(budget["min_chars"], "length_budget.min_chars", positive=True),
        _int(budget["max_chars"], "length_budget.max_chars", positive=True),
        tuple(_surface_variant(item, f"accepted[{index}]", rejected=False) for index, item in enumerate(accepted_raw)),
        tuple(_surface_variant(item, f"rejected[{index}]", rejected=True) for index, item in enumerate(rejected_raw)),
    )


@dataclass(frozen=True, slots=True)
class CompiledSurfaceOrganizationRecord:
    """单条课程的 typed record、规范 JSON 和整数投影。"""

    record: SurfaceOrganizationRecord
    canonical_json: bytes
    integer_record: dict[str, Any]


def compile_surface_organization_record(record: SurfaceOrganizationRecord) -> CompiledSurfaceOrganizationRecord:
    """将不可变课程 record 编译为稳定规范 JSONL 与整数/u8 记录。"""
    if not isinstance(record, SurfaceOrganizationRecord):
        raise TypeError("需要 SurfaceOrganizationRecord")
    integer_payload = record.canonical_integer_record
    # canonical_json 是可回读的 schema wire record；整数/u8 投影单独保留，
    # 避免把 Python 的字符串或 dataclass 形状误当作跨语言语义来源。
    wire = {
        "accepted": [
            {"clause_order": list(item.clause_order),
             "proposition_ids": list(item.proposition_ids),
             "register": item.register, "surface": item.surface,
             "variant_id": item.variant_id}
            for item in record.accepted
        ],
        "context_id": record.context_id,
        "dialogue_act": record.dialogue_act,
        "family_id": record.family_id,
        "length_budget": {"max_chars": record.max_chars, "min_chars": record.min_chars},
        "license_id": LICENSE_ID,
        "obligation": {
            "forbidden_proposition_ids": list(record.forbidden_proposition_ids),
            "ordered_clause_slots": [
                {"order": item.order, "required": item.required, "role": item.role,
                 "slot_id": item.slot_id}
                for item in record.clause_slots
            ],
            "required_proposition_ids": list(record.required_proposition_ids),
        },
        "owner": record.owner,
        "record_kind": RECORD_KIND,
        "register": record.register,
        "rejected": [
            {"clause_order": list(item.clause_order),
             "proposition_ids": list(item.proposition_ids),
             "register": item.register, "surface": item.surface,
             "variant_id": item.variant_id, "violations": list(item.violations)}
            for item in record.rejected
        ],
        "sample_id": record.sample_id,
        "schema_version": SCHEMA_VERSION,
        "semantic_proposition": {
            "id": record.proposition_id, "kind": record.proposition_kind,
            "object": record.proposition_object, "predicate": record.proposition_predicate,
            "subject": record.proposition_subject,
        },
        "source_id": record.source_id,
        "split": record.split,
        "template_family": record.template_family,
    }
    encoded = canonical_json_line(wire)
    return CompiledSurfaceOrganizationRecord(record, encoded, integer_payload)


def compile_surface_organization_course(values: Iterable[Any]) -> tuple[CompiledSurfaceOrganizationRecord, ...]:
    """编译整个课程并执行 source/context/family/template 的物理隔离。"""
    records = tuple(value if isinstance(value, SurfaceOrganizationRecord)
                    else parse_surface_organization_record(value) for value in values)
    if len({item.sample_id for item in records}) != len(records):
        raise SurfaceOrganizationError("sample_id 必须唯一")
    # source/context/family 是物理隔离键；template 可在不同 family 复用，
    # 但同一 family 不得出现两个模板或跨 family 共享其身份。
    for field in ("source_id", "context_id", "family_id"):
        values_for_field = [getattr(item, field) for item in records]
        if len(values_for_field) != len(set(values_for_field)):
            raise SurfaceOrganizationError(f"课程内 {field} 必须物理隔离")
    family_template = [(item.family_id, item.template_family) for item in records]
    if len(family_template) != len(set(family_template)):
        raise SurfaceOrganizationError("同一 family 的 template 必须唯一")
    if any(item.owner != "public-course" or item.split != "course" for item in records):
        raise SurfaceOrganizationError("课程 owner/split 必须固定为公开 course")
    return tuple(compile_surface_organization_record(item) for item in records)


def load_surface_organization_jsonl(payload: bytes) -> tuple[CompiledSurfaceOrganizationRecord, ...]:
    """从公开 JSONL bytes 严格回读并编译；拒绝非规范行和空行。"""
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise SurfaceOrganizationError("JSONL 必须以换行结束")
    rows = payload.splitlines(keepends=True)
    if not rows:
        raise SurfaceOrganizationError("JSONL 不能为空")
    values = []
    for index, line in enumerate(rows):
        if not line.endswith(b"\n") or line == b"\n":
            raise SurfaceOrganizationError(f"第 {index} 行为空或无换行")
        value = parse_canonical_json_bytes(line[:-1], require_object=True)
        if canonical_json_line(value) != line:
            raise SurfaceOrganizationError(f"第 {index} 行不是规范 JSONL")
        values.append(value)
    return compile_surface_organization_course(values)


__all__ = ["CompiledSurfaceOrganizationRecord", "parse_surface_organization_record",
           "compile_surface_organization_record", "compile_surface_organization_course",
           "load_surface_organization_jsonl"]
