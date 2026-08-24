"""T1-G8 G7 surface variants through the G6 focus shadow."""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_shadow import (
    SurfaceShadowPlan,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceEvidencePack,
    learn_surface_structure_model,
    load_surface_evidence_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionConsumerResult,
)
from pure_integer_ai.experiments.conversation_raw_t1_g8_focus_surface import (
    run_focus_surface_shadow_turn,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_adapter import (
    run_raw_t1_shadow_adapter,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_dialogue import (
    start_raw_t1_shadow_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_variants import (
    learn_surface_variant_model,
)


_ROOT = Path(__file__).resolve().parents[1]
_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_course_v1.jsonl.sample"
_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_evidence_v1.jsonl.sample"


def _models():
    rows = tuple(item.record for item in load_surface_organization_jsonl(_COURSE.read_bytes()))
    evidence = load_surface_evidence_jsonl(_EVIDENCE.read_bytes())
    a01 = SurfaceEvidencePack(
        evidence.source_namespace,
        evidence.license_id,
        tuple(item for item in evidence.entries if item.variant_id == "a01"),
    )
    return (
        learn_surface_structure_model(rows, a01),
        learn_surface_variant_model(rows, evidence),
    )


def _consumer(state: str) -> RawPropositionConsumerResult:
    return RawPropositionConsumerResult(
        "p-g8-heldout", f"q-g8-{state.lower()}", "obs-g8-heldout",
        "src-g8-heldout", "ctx-g8-heldout", "fam-g8-heldout",
        state, {"SUPPORTED": "ANSWER", "UNKNOWN": "UNKNOWN"}[state],
        ("e-g8-heldout",), (1, 8),
    )


def _plan(state: str) -> SurfaceShadowPlan:
    semantic = SurfaceSemantic(
        "p-g8-heldout", "state", "设备C", "状态", "待机")
    if state == "SUPPORTED":
        return SurfaceShadowPlan(
            semantic, "ANSWER", "neutral", ("subject", "predicate", "object"),
            ("p-g8-heldout",), (), ("src-g8-heldout",),
            "ctx-g8-heldout", "fam-g8-heldout", "legacy", 2, 40,
        )
    return SurfaceShadowPlan(
        semantic, "UNKNOWN", "neutral", ("source", "scope"),
        (), ("p-g8-heldout",), ("src-g8-heldout",),
        "ctx-g8-heldout", "fam-g8-heldout", "legacy", 2, 40,
        ("当前资料", "待机状态"),
    )


def test_g8_answer_variant_and_unknown_focus_are_one_shadow_chain() -> None:
    old_model, variant_model = _models()
    answer_adapter = run_raw_t1_shadow_adapter(
        old_model, _consumer("SUPPORTED"), _plan("SUPPORTED"))
    unknown_adapter = run_raw_t1_shadow_adapter(
        old_model, _consumer("UNKNOWN"), _plan("UNKNOWN"))
    first = run_focus_surface_shadow_turn(
        start_raw_t1_shadow_dialogue((65008, 8, 1)),
        answer_adapter, variant_model, selection_ordinal=1,
    )
    second = run_focus_surface_shadow_turn(
        first.dialogue_turn.after, unknown_adapter, variant_model,
    )
    assert first.surface == "设备C的状态是待机。"
    assert first.replaced == second.replaced == 0
    assert first.dialogue_turn.after.focus_revision == 1
    assert second.dialogue_turn.after.focus_revision == 2
    assert second.surface is None
    assert second.variant_result is None
    assert second.dialogue_turn.adapter_result.consumer.response_act == "UNKNOWN"
    assert second.dialogue_turn.adapter_result.shadow.plan.required_proposition_ids == ()


def test_g8_records_are_deterministic() -> None:
    old_model, variant_model = _models()
    adapter = run_raw_t1_shadow_adapter(
        old_model, _consumer("SUPPORTED"), _plan("SUPPORTED"))
    state = start_raw_t1_shadow_dialogue((65008, 8, 2))
    first = run_focus_surface_shadow_turn(state, adapter, variant_model)
    second = run_focus_surface_shadow_turn(state, adapter, variant_model)
    assert first.canonical_record() == second.canonical_record()
