"""DLG-RAW-16 G2A shadow consumer 的只读边界回归。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_shadow import (
    SHADOW_NO_PATTERN,
    SHADOW_SELECTED,
    SurfaceShadowError,
    SurfaceShadowPlan,
    run_surface_shadow,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    load_surface_evidence_jsonl,
    learn_surface_structure_model,
)


_ROOT = Path(__file__).resolve().parents[1]
_COURSE = _ROOT / "data/ph2/dlg_raw16_surface_organization_v1.jsonl.sample"
_EVIDENCE = _ROOT / "data/ph2/dlg_raw16_surface_slot_evidence_v1.jsonl.sample"


@pytest.fixture(scope="module")
def model():
    records = tuple(item.record for item in
                    load_surface_organization_jsonl(_COURSE.read_bytes()))
    selected = tuple(item for item in records if item.sample_id in {"s03", "s04"})
    return learn_surface_structure_model(
        selected, load_surface_evidence_jsonl(_EVIDENCE.read_bytes()))


def _plan(*, legacy_surface: str = "旧答案") -> SurfaceShadowPlan:
    return SurfaceShadowPlan(
        SurfaceSemantic("p-shadow-new", "causal", "台风", "导致", "港口封闭"),
        "ANSWER", "neutral", ("cause", "relation", "effect"),
        ("p-shadow-new",), (), ("src-shadow-new",),
        "ctx-shadow-new", "family-shadow-new", legacy_surface, 2, 80,
    )


def test_shadow_generates_new_surface_without_replacing_legacy(model) -> None:
    result = run_surface_shadow(model, _plan())
    assert result.status_code == SHADOW_SELECTED
    assert result.shadow_surface == "台风导致港口封闭。"
    assert result.plan.legacy_surface == "旧答案"
    assert result.replaced == 0
    assert result.canonical_record()


def test_shadow_unknown_structure_has_no_output(model) -> None:
    plan = SurfaceShadowPlan(
        SurfaceSemantic("p-shadow-unknown", "causal", "台风", "影响", "港口"),
        "ANSWER", "neutral", ("topic", "relation", "condition", "effect"),
        ("p-shadow-unknown",), (), ("src-shadow-unknown",),
        "ctx-shadow-unknown", "family-shadow-unknown", "旧答案", 2, 80,
    )
    result = run_surface_shadow(model, plan)
    assert result.status_code == SHADOW_NO_PATTERN
    assert result.shadow_surface is None
    assert result.replaced == 0


def test_shadow_rejects_nonanswer_before_runtime() -> None:
    with pytest.raises(SurfaceShadowError, match="slot_values"):
        SurfaceShadowPlan(
            SurfaceSemantic("p", "unknown", "对象", "预算", "未提供"),
            "UNKNOWN", "neutral", ("scope", "epistemic"),
            (), (), ("src",), "ctx", "family", "旧答案", 1, 80,
        )


def test_clarify_shadow_rebuilds_candidates_without_claim(model) -> None:
    records = tuple(item.record for item in
                    load_surface_organization_jsonl(_COURSE.read_bytes()))
    selected = tuple(item for item in records if item.sample_id in {"s09", "s10"})
    clarify_model = learn_surface_structure_model(
        selected, load_surface_evidence_jsonl(_EVIDENCE.read_bytes()))
    plan = SurfaceShadowPlan(
        SurfaceSemantic("p-shadow-clarify", "scope", "数量", "指向", "待选区域"),
        "CLARIFY", "polite", ("choice", "target"),
        (), (), ("src-shadow-clarify",), "ctx-shadow-clarify",
        "family-shadow-clarify", "旧答案", 2, 80,
        ("甲区还是乙区", "数量"),
    )
    result = run_surface_shadow(clarify_model, plan)
    assert result.status_code == SHADOW_SELECTED
    assert result.shadow_surface == "请先选择甲区还是乙区，再说明要查询的数量。"
    assert "万元" not in result.shadow_surface
    assert result.replaced == 0
    assert result.canonical_record()


def test_clarify_shadow_fails_closed_on_claim_or_missing_slots() -> None:
    base = dict(
        semantic=SurfaceSemantic("p", "scope", "预算", "指向", "待选区域"),
        dialogue_act="CLARIFY", register="polite",
        ordered_roles=("choice", "target"), required_proposition_ids=(),
        forbidden_proposition_ids=(), authorized_source_ids=("src",),
        context_id="ctx", family_id="family", legacy_surface="旧答案",
    )
    with pytest.raises(SurfaceShadowError, match="slot_values"):
        SurfaceShadowPlan(**base)
    with pytest.raises(SurfaceShadowError, match="required proposition"):
        SurfaceShadowPlan(**{**base, "required_proposition_ids": ("p",),
                             "slot_values": ("东区还是西区", "预算")})


def test_qualified_answer_shadow_preserves_nonsemantic_qualifier(model) -> None:
    records = tuple(item.record for item in
                    load_surface_organization_jsonl(_COURSE.read_bytes()))
    selected = tuple(item for item in records if item.sample_id in {"s11", "s12"})
    qualified_model = learn_surface_structure_model(
        selected, load_surface_evidence_jsonl(_EVIDENCE.read_bytes()))
    plan = SurfaceShadowPlan(
        SurfaceSemantic("p-shadow-qualified", "qualified_fact", "新入口",
                        "启用时间", "2030年1月"),
        "ANSWER", "polite", ("subject", "predicate", "qualifier", "object"),
        ("p-shadow-qualified",), (), ("src-shadow-qualified",),
        "ctx-shadow-qualified", "family-shadow-qualified", "旧答案", 2, 80,
        ("新入口", "启用时间", "审计记录", "2030年1月"),
    )
    result = run_surface_shadow(qualified_model, plan)
    assert result.status_code == SHADOW_SELECTED
    assert result.shadow_surface == "新入口的启用时间（审计记录）为2030年1月。"
    assert "档案" not in result.shadow_surface
    assert result.replaced == 0


def test_qualified_answer_shadow_requires_explicit_qualifier(model) -> None:
    records = tuple(item.record for item in
                    load_surface_organization_jsonl(_COURSE.read_bytes()))
    selected = tuple(item for item in records if item.sample_id in {"s11", "s12"})
    qualified_model = learn_surface_structure_model(
        selected, load_surface_evidence_jsonl(_EVIDENCE.read_bytes()))
    plan = SurfaceShadowPlan(
        SurfaceSemantic("p", "qualified_fact", "入口", "启用时间", "2030年1月"),
        "ANSWER", "polite", ("subject", "predicate", "qualifier", "object"),
        ("p",), (), ("src",), "ctx", "family", "旧答案", 2, 80,
    )
    result = run_surface_shadow(qualified_model, plan)
    assert result.status_code == SHADOW_NO_PATTERN
    assert result.shadow_surface is None
    assert result.replaced == 0


def test_unknown_and_repair_shadow_remain_zero_claim(model) -> None:
    records = tuple(item.record for item in
                    load_surface_organization_jsonl(_COURSE.read_bytes()))
    unknown_model = learn_surface_structure_model(
        tuple(item for item in records if item.sample_id in {"s13", "s14"}),
        load_surface_evidence_jsonl(_EVIDENCE.read_bytes()))
    unknown = SurfaceShadowPlan(
        SurfaceSemantic("p-shadow-unknown", "unknown", "青石台",
                        "运行预算", "未提供"),
        "UNKNOWN", "neutral", ("source", "scope"),
        (), ("p-invented-shadow",), ("src-shadow-unknown",),
        "ctx-shadow-unknown", "family-shadow-unknown", "旧答案", 2, 80,
        ("当前", "青石台的运行预算"),
    )
    unknown_result = run_surface_shadow(unknown_model, unknown)
    assert unknown_result.status_code == SHADOW_SELECTED
    assert unknown_result.shadow_surface == "当前资料没有提供青石台的运行预算。"
    assert unknown_result.replaced == 0

    repair_model = learn_surface_structure_model(
        tuple(item for item in records if item.sample_id in {"s15", "s16"}),
        load_surface_evidence_jsonl(_EVIDENCE.read_bytes()))
    repair = SurfaceShadowPlan(
        SurfaceSemantic("p-shadow-repair", "repair", "先前问题",
                        "需要", "完整限定"),
        "REPAIR", "polite", ("acknowledge", "request"),
        (), (), ("src-shadow-repair",), "ctx-shadow-repair",
        "family-shadow-repair", "旧答案", 2, 80,
        ("前面的条件不够明确", "具体时间"),
    )
    repair_result = run_surface_shadow(repair_model, repair)
    assert repair_result.status_code == SHADOW_SELECTED
    assert repair_result.shadow_surface == "前面的条件不够明确，请说明具体时间。"
    assert repair_result.replaced == 0
