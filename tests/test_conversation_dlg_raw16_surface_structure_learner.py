"""DLG-RAW-16 G2A 带证据结构学习的窄回归。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    STRUCTURE_NO_PATTERN,
    STRUCTURE_SELECTED,
    SurfaceEvidencePack,
    SurfaceSlotEvidence,
    SurfaceStructureLearningError,
    SurfaceStructureRequest,
    load_surface_evidence_jsonl,
    learn_surface_structure_model,
    realize_surface_structure,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)


_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "data/ph2/dlg_raw16_surface_organization_v1.jsonl.sample"
_EVIDENCE = _ROOT / "data/ph2/dlg_raw16_surface_slot_evidence_v1.jsonl.sample"


def _records():
    return tuple(item.record for item in
                 load_surface_organization_jsonl(_SAMPLE.read_bytes()))


def _evidence(record, variant_id: str) -> tuple[SurfaceSlotEvidence, ...]:
    variant = next(item for item in record.accepted
                   if item.variant_id == variant_id)
    # s03/s04 a01 are causal records with the same typed role order.
    values = {
        "cause": record.proposition_subject,
        "relation": record.proposition_predicate,
        "effect": record.proposition_object,
    }
    result = []
    cursor = 0
    for slot in record.clause_slots:
        value = values[slot.role]
        start = variant.surface.find(value, cursor)
        assert start >= 0
        end = start + len(value)
        result.append(SurfaceSlotEvidence(
            record.sample_id, variant_id, slot.slot_id, slot.role, start, end))
        cursor = end
    return tuple(result)


@pytest.fixture(scope="module")
def model():
    records = _records()
    s03 = next(item for item in records if item.sample_id == "s03")
    s04 = next(item for item in records if item.sample_id == "s04")
    pack = load_surface_evidence_jsonl(_EVIDENCE.read_bytes())
    return learn_surface_structure_model((s03, s04), pack)


def test_public_evidence_pack_is_canonical_and_integer_replayable() -> None:
    pack = load_surface_evidence_jsonl(_EVIDENCE.read_bytes())
    assert pack.source_namespace == "dlg-raw16-g2a-cc0-v1"
    assert len(pack.entries) == 46
    assert pack.canonical_record()


def test_learns_literal_slot_structure_across_two_families(model) -> None:
    assert len(model.patterns) == 1
    pattern = model.patterns[0]
    assert pattern.roles == ("cause", "relation", "effect")
    assert len(pattern.support_family_ids) == 2
    request = SurfaceStructureRequest(
        SurfaceSemantic("p-new-causal", "causal", "台风", "导致", "港口封闭"),
        "ANSWER", "neutral", pattern.roles,
        2, 80, 0, "heldout-src", "heldout-ctx", "heldout-family",
    )
    result = realize_surface_structure(model, request)
    assert result.status_code == STRUCTURE_SELECTED
    assert result.surface == "台风导致港口封闭。"
    assert "暴雨" not in result.surface and "寒潮" not in result.surface
    assert result.output_bytes == tuple(result.surface.encode("utf-8"))
    assert result.canonical_record()


def test_unseen_role_order_fails_closed(model) -> None:
    request = SurfaceStructureRequest(
        SurfaceSemantic("p-new", "fact", "新设施", "状态", "已启用"),
        "ANSWER", "neutral", ("subject", "predicate", "object"),
        1, 80, 0, "heldout-src-2", "heldout-ctx-2", "heldout-family-2",
    )
    result = realize_surface_structure(model, request)
    assert result.status_code == STRUCTURE_NO_PATTERN
    assert result.surface is None
    assert result.output_bytes == ()


def test_slot_evidence_cannot_claim_a_different_proposition() -> None:
    records = _records()
    record = next(item for item in records if item.sample_id == "s03")
    entries = list(_evidence(record, "a01"))
    bad = entries[0]
    entries[0] = SurfaceSlotEvidence(
        bad.record_id, bad.variant_id, bad.slot_id, bad.role,
        bad.start + 1, bad.end + 1)
    with pytest.raises(SurfaceStructureLearningError, match="slot span"):
        learn_surface_structure_model(
            (record,), SurfaceEvidencePack(
                "dlg-raw16-g2a-bad", "CC0-1.0", tuple(entries)))


def test_one_family_does_not_count_as_learned_structure() -> None:
    records = _records()
    record = next(item for item in records if item.sample_id == "s03")
    with pytest.raises(SurfaceStructureLearningError, match="两个独立 family"):
        learn_surface_structure_model(
            (record,), SurfaceEvidencePack(
                "dlg-raw16-g2a-single", "CC0-1.0", _evidence(record, "a01")))


def test_clarify_structure_rebuilds_new_explicit_slot_values() -> None:
    records = _records()
    selected = tuple(item for item in records if item.sample_id in {"s09", "s10"})
    model = learn_surface_structure_model(
        selected, load_surface_evidence_jsonl(_EVIDENCE.read_bytes()))
    assert len(model.patterns) == 1
    pattern = model.patterns[0]
    assert pattern.dialogue_act == "CLARIFY"
    assert pattern.roles == ("choice", "target")
    request = SurfaceStructureRequest(
        SurfaceSemantic("p-heldout-clarify", "scope", "数量", "指向", "待选区域"),
        "CLARIFY", "polite", pattern.roles,
        2, 80, 0, "heldout-src", "heldout-ctx", "heldout-family",
        ("甲区还是乙区", "数量"),
    )
    result = realize_surface_structure(model, request)
    assert result.status_code == STRUCTURE_SELECTED
    assert result.surface == "请先选择甲区还是乙区，再说明要查询的数量。"
    assert "东区" not in result.surface and "北区" not in result.surface
    assert result.output_bytes == tuple(result.surface.encode("utf-8"))


def test_qualified_answer_preserves_explicit_qualifier_slot() -> None:
    records = _records()
    selected = tuple(item for item in records if item.sample_id in {"s11", "s12"})
    model = learn_surface_structure_model(
        selected, load_surface_evidence_jsonl(_EVIDENCE.read_bytes()))
    assert len(model.patterns) == 1
    pattern = model.patterns[0]
    assert pattern.roles == ("subject", "predicate", "qualifier", "object")
    request = SurfaceStructureRequest(
        SurfaceSemantic("p-heldout-qualified", "qualified_fact", "新入口",
                        "启用时间", "2030年1月"),
        "ANSWER", "polite", pattern.roles,
        2, 80, 0, "heldout-src-q", "heldout-ctx-q", "heldout-family-q",
        ("新入口", "启用时间", "审计记录", "2030年1月"),
    )
    result = realize_surface_structure(model, request)
    assert result.status_code == STRUCTURE_SELECTED
    assert result.surface == "新入口的启用时间（审计记录）为2030年1月。"
    assert "档案" not in result.surface and "现场" not in result.surface
    assert result.output_bytes == tuple(result.surface.encode("utf-8"))


def test_unknown_structure_rebuilds_zero_claim_surface() -> None:
    records = _records()
    selected = tuple(item for item in records if item.sample_id in {"s13", "s14"})
    model = learn_surface_structure_model(
        selected, load_surface_evidence_jsonl(_EVIDENCE.read_bytes()))
    pattern = model.patterns[0]
    assert pattern.dialogue_act == "UNKNOWN"
    request = SurfaceStructureRequest(
        SurfaceSemantic("p-heldout-unknown", "unknown", "青石台",
                        "运行预算", "未提供"),
        "UNKNOWN", "neutral", pattern.roles,
        2, 80, 0, "heldout-src-u", "heldout-ctx-u", "heldout-family-u",
        ("当前", "青石台的运行预算"),
    )
    result = realize_surface_structure(model, request)
    assert result.status_code == STRUCTURE_SELECTED
    assert result.surface == "当前资料没有提供青石台的运行预算。"
    assert "十万元" not in result.surface


def test_repair_structure_rebuilds_minimal_request() -> None:
    records = _records()
    selected = tuple(item for item in records if item.sample_id in {"s15", "s16"})
    model = learn_surface_structure_model(
        selected, load_surface_evidence_jsonl(_EVIDENCE.read_bytes()))
    pattern = model.patterns[0]
    assert pattern.dialogue_act == "REPAIR"
    request = SurfaceStructureRequest(
        SurfaceSemantic("p-heldout-repair", "repair", "先前问题",
                        "需要", "完整限定"),
        "REPAIR", "polite", pattern.roles,
        2, 80, 0, "heldout-src-r", "heldout-ctx-r", "heldout-family-r",
        ("前面的条件不够明确", "具体时间"),
    )
    result = realize_surface_structure(model, request)
    assert result.status_code == STRUCTURE_SELECTED
    assert result.surface == "前面的条件不够明确，请说明具体时间。"
