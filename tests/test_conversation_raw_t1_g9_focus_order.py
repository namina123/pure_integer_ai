"""T1-G9 role-order surface through the G6 focus shadow."""
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
from pure_integer_ai.experiments.conversation_raw_t1_g8_focus_surface import (
    run_focus_order_shadow_turn,
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


_ROOT = Path(__file__).resolve().parents[1]
_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g9_surface_order_course_v1.jsonl.sample"
_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g9_surface_order_evidence_v1.jsonl.sample"


def _models():
    rows = tuple(item.record for item in load_surface_organization_jsonl(_COURSE.read_bytes()))
    evidence = load_surface_evidence_jsonl(_EVIDENCE.read_bytes())
    forward_rows = tuple(item for item in rows if item.sample_id in {"g9-af", "g9-bf"})
    forward_evidence = SurfaceEvidencePack(
        evidence.source_namespace,
        evidence.license_id,
        tuple(item for item in evidence.entries
              if item.record_id in {"g9-af", "g9-bf"} and item.variant_id == "a01"),
    )
    return (
        learn_surface_structure_model(forward_rows, forward_evidence),
        learn_surface_order_model(rows, evidence),
    )


def _consumer(state: str) -> RawPropositionConsumerResult:
    return RawPropositionConsumerResult(
        "p-g9-heldout", f"q-g9-{state.lower()}", "obs-g9-heldout",
        "src-g9-heldout", "ctx-g9-heldout", "fam-g9-heldout",
        state, {"SUPPORTED": "ANSWER", "UNKNOWN": "UNKNOWN"}[state],
        ("e-g9-heldout",), (1, 9),
    )


def _plan(state: str) -> SurfaceShadowPlan:
    semantic = SurfaceSemantic("p-g9-heldout", "state", "装置C", "状态", "待机")
    if state == "SUPPORTED":
        return SurfaceShadowPlan(
            semantic, "ANSWER", "neutral", ("object", "predicate", "subject"),
            ("p-g9-heldout",), (), ("src-g9-heldout",),
            "ctx-g9-heldout", "fam-g9-heldout", "legacy", 2, 40,
        )
    return SurfaceShadowPlan(
        semantic, "UNKNOWN", "neutral", ("source", "scope"),
        (), ("p-g9-heldout",), ("src-g9-heldout",),
        "ctx-g9-heldout", "fam-g9-heldout", "legacy", 2, 40,
        ("当前资料", "待机状态"),
    )


def test_g9_role_order_and_unknown_focus_chain() -> None:
    old_model, order_model = _models()
    answer_adapter = run_raw_t1_shadow_adapter(
        old_model, _consumer("SUPPORTED"), _plan("SUPPORTED"))
    unknown_adapter = run_raw_t1_shadow_adapter(
        old_model, _consumer("UNKNOWN"), _plan("UNKNOWN"))
    first = run_focus_order_shadow_turn(
        start_raw_t1_shadow_dialogue((65009, 9, 1)),
        answer_adapter, order_model,
    )
    second = run_focus_order_shadow_turn(
        first.dialogue_turn.after, unknown_adapter, order_model,
    )
    assert first.surface == "待机状态属于装置C。"
    assert first.dialogue_turn.after.focus_revision == 1
    assert second.dialogue_turn.after.focus_revision == 2
    assert second.surface is None
    assert second.order_result is None
    assert first.replaced == second.replaced == 0
