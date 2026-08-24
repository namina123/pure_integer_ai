"""DLG-RAW-16 表层组织课程的有界合同测试。"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    compile_surface_organization_course,
    load_surface_organization_jsonl,
    parse_surface_organization_record,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_schema import (
    SurfaceOrganizationError,
)
from pure_integer_ai.experiments.ph2_dataset_core import parse_canonical_json_bytes


_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "data" / "ph2" / "dlg_raw16_surface_organization_v1.jsonl.sample"


def _values() -> list[dict]:
    return [parse_canonical_json_bytes(line[:-1], require_object=True)
            for line in _SAMPLE.read_bytes().splitlines(keepends=True)]


def test_public_course_compiles_to_immutable_integer_records() -> None:
    compiled = load_surface_organization_jsonl(_SAMPLE.read_bytes())
    assert len(compiled) == 16
    assert [item.record.dialogue_act for item in compiled] == [
        "ANSWER", "ANSWER", "ANSWER", "ANSWER", "ANSWER", "CLARIFY", "UNKNOWN", "REPAIR",
        "CLARIFY", "CLARIFY", "ANSWER", "ANSWER", "UNKNOWN", "UNKNOWN",
        "REPAIR", "REPAIR",
    ]
    assert all(item.integer_record["record_kind"] == "DLG_RAW16_SURFACE_ORGANIZATION_V1"
               for item in compiled)
    assert all(isinstance(item.canonical_json, bytes) and item.canonical_json.endswith(b"\n")
               for item in compiled)
    assert all(item.integer_record["accepted"][0]["surface"]["utf8"]
               for item in compiled)
    # 编译产物的规范 JSON 仍是同一 schema wire record，可跨实现回读。
    assert parse_surface_organization_record(
        parse_canonical_json_bytes(compiled[0].canonical_json[:-1], require_object=True)
    ).sample_id == "s01"


def test_semantic_proposition_and_clause_order_drift_fail_closed() -> None:
    rows = _values()
    rows[0]["accepted"][0]["proposition_ids"] = []
    with pytest.raises(SurfaceOrganizationError, match="语义命题漂移"):
        compile_surface_organization_course(rows)
    rows = _values()
    rows[0]["accepted"][0]["clause_order"] = ["s2", "s1", "s3"]
    with pytest.raises(SurfaceOrganizationError, match="有序槽位漂移"):
        compile_surface_organization_course(rows)


def test_rejected_variant_requires_explicit_surface_violation() -> None:
    rows = _values()
    rows[1]["rejected"][0]["violations"] = []
    with pytest.raises(SurfaceOrganizationError, match="rejected"):
        compile_surface_organization_course(rows)


@pytest.mark.parametrize("field", ["source_id", "context_id", "family_id"])
def test_physical_identity_collision_is_rejected(field: str) -> None:
    rows = _values()
    rows[1][field] = rows[0][field]
    with pytest.raises(SurfaceOrganizationError, match="物理隔离"):
        compile_surface_organization_course(rows)


def test_owner_split_and_family_template_are_frozen() -> None:
    rows = _values()
    rows[0]["owner"] = "private"
    with pytest.raises(SurfaceOrganizationError, match="owner/split"):
        compile_surface_organization_course(rows)
    rows = _values()
    rows[1]["family_id"] = rows[0]["family_id"]
    rows[1]["template_family"] = rows[0]["template_family"]
    with pytest.raises(SurfaceOrganizationError, match="物理隔离"):
        compile_surface_organization_course(rows)
