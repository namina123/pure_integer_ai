"""T1-G13 joint audit: qualifier ANSWER -> UNKNOWN focus chain."""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_shadow import (
    SurfaceShadowPlan,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceEvidencePack,
    learn_surface_structure_model,
    load_surface_evidence_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionConsumerResult,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_adapter import (
    run_raw_t1_shadow_adapter,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_dialogue import (
    start_raw_t1_shadow_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_order import (
    learn_surface_order_model,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_plan import (
    build_surface_plan_model,
    run_unified_focus_surface_turn,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_variants import (
    learn_surface_variant_model,
)


_ROOT = Path(__file__).resolve().parents[1]
_BASE_COURSE = _ROOT / "data/ph2/dlg_raw16_surface_organization_v1.jsonl.sample"
_BASE_EVIDENCE = _ROOT / "data/ph2/dlg_raw16_surface_slot_evidence_v1.jsonl.sample"
_G7_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_course_v1.jsonl.sample"
_G7_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_evidence_v1.jsonl.sample"
_G9_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g9_surface_order_course_v1.jsonl.sample"
_G9_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g9_surface_order_evidence_v1.jsonl.sample"


def _models():
    base_rows = tuple(item.record for item in load_surface_organization_jsonl(_BASE_COURSE.read_bytes()))
    base_evidence = load_surface_evidence_jsonl(_BASE_EVIDENCE.read_bytes())
    g7_rows = tuple(item.record for item in load_surface_organization_jsonl(_G7_COURSE.read_bytes()))
    g7_evidence = load_surface_evidence_jsonl(_G7_EVIDENCE.read_bytes())
    g9_rows = tuple(item.record for item in load_surface_organization_jsonl(_G9_COURSE.read_bytes()))
    g9_evidence = load_surface_evidence_jsonl(_G9_EVIDENCE.read_bytes())
    qualifier_rows = tuple(item for item in base_rows if item.sample_id in {"s11", "s12"})
    qualifier_model = learn_surface_structure_model(qualifier_rows, base_evidence)
    gap_model = learn_surface_variant_model(g7_rows, g7_evidence)
    order_model = learn_surface_order_model(g9_rows, g9_evidence)
    return qualifier_model, build_surface_plan_model(gap_model, order_model, qualifier_model)


def _consumer(state: str) -> RawPropositionConsumerResult:
    return RawPropositionConsumerResult(
        "p-g13-heldout", f"q-g13-{state.lower()}", "obs-g13-heldout",
        "src-g13-heldout", "ctx-g13-heldout", "fam-g13-heldout",
        state, {"SUPPORTED": "ANSWER", "UNKNOWN": "UNKNOWN"}[state],
        ("e-g13",), (1, 13),
    )


def _plan(state: str) -> SurfaceShadowPlan:
    semantic = SurfaceSemantic(
        "p-g13-heldout", "fact", "西岸入口", "启用时间", "2031年4月")
    if state == "SUPPORTED":
        return SurfaceShadowPlan(
            semantic, "ANSWER", "polite",
            ("subject", "predicate", "qualifier", "object"),
            ("p-g13-heldout",), (), ("src-g13-heldout",),
            "ctx-g13-heldout", "fam-g13-heldout", "legacy", 2, 80,
            ("西岸入口", "启用时间", "审计记录", "2031年4月"),
        )
    return SurfaceShadowPlan(
        semantic, "UNKNOWN", "neutral", ("source", "scope"),
        (), ("p-g13-heldout",), ("src-g13-heldout",),
        "ctx-g13-heldout", "fam-g13-heldout", "legacy", 2, 80,
        ("当前资料", "该入口启用时间"),
    )


def test_g13_qualified_answer_then_unknown_is_zero_claim_focus_chain() -> None:
    qualifier_model, unified_model = _models()
    answer_adapter = run_raw_t1_shadow_adapter(
        qualifier_model, _consumer("SUPPORTED"), _plan("SUPPORTED"))
    unknown_adapter = run_raw_t1_shadow_adapter(
        qualifier_model, _consumer("UNKNOWN"), _plan("UNKNOWN"))
    first = run_unified_focus_surface_turn(
        start_raw_t1_shadow_dialogue((65013, 13, 1)),
        answer_adapter, unified_model,
    )
    second = run_unified_focus_surface_turn(
        first.dialogue_turn.after, unknown_adapter, unified_model,
    )
    assert first.surface == "西岸入口的启用时间（审计记录）为2031年4月。"
    assert second.surface is None
    assert second.plan_result is None
    assert first.dialogue_turn.after.focus_revision == 1
    assert second.dialogue_turn.after.focus_revision == 2
    assert first.replaced == second.replaced == 0
