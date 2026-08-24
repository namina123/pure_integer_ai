"""T1-G10 unified G7/G9 surface plan and focus boundary."""
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
    SurfaceStructureRequest,
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
    PLAN_NO_PATTERN,
    PLAN_SELECTED,
    build_surface_plan_model,
    realize_surface_plan,
    run_unified_focus_surface_turn,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_variants import (
    learn_surface_variant_model,
)


_ROOT = Path(__file__).resolve().parents[1]
_G7_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_course_v1.jsonl.sample"
_G7_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_evidence_v1.jsonl.sample"
_G9_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g9_surface_order_course_v1.jsonl.sample"
_G9_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g9_surface_order_evidence_v1.jsonl.sample"
_BASE_COURSE = _ROOT / "data/ph2/dlg_raw16_surface_organization_v1.jsonl.sample"
_BASE_EVIDENCE = _ROOT / "data/ph2/dlg_raw16_surface_slot_evidence_v1.jsonl.sample"


def _models():
    g7_rows = tuple(item.record for item in load_surface_organization_jsonl(_G7_COURSE.read_bytes()))
    g7_evidence = load_surface_evidence_jsonl(_G7_EVIDENCE.read_bytes())
    g9_rows = tuple(item.record for item in load_surface_organization_jsonl(_G9_COURSE.read_bytes()))
    g9_evidence = load_surface_evidence_jsonl(_G9_EVIDENCE.read_bytes())
    gap_model = learn_surface_variant_model(g7_rows, g7_evidence)
    order_model = learn_surface_order_model(g9_rows, g9_evidence)
    return build_surface_plan_model(gap_model, order_model), g7_rows, g7_evidence


def _models_with_qualifier():
    g7_rows = tuple(item.record for item in load_surface_organization_jsonl(_G7_COURSE.read_bytes()))
    g7_evidence = load_surface_evidence_jsonl(_G7_EVIDENCE.read_bytes())
    g9_rows = tuple(item.record for item in load_surface_organization_jsonl(_G9_COURSE.read_bytes()))
    g9_evidence = load_surface_evidence_jsonl(_G9_EVIDENCE.read_bytes())
    base_rows = tuple(item.record for item in load_surface_organization_jsonl(_BASE_COURSE.read_bytes()))
    base_evidence = load_surface_evidence_jsonl(_BASE_EVIDENCE.read_bytes())
    gap_model = learn_surface_variant_model(g7_rows, g7_evidence)
    order_model = learn_surface_order_model(g9_rows, g9_evidence)
    qualifier_rows = tuple(item for item in base_rows if item.sample_id in {"s11", "s12"})
    qualifier_model = learn_surface_structure_model(qualifier_rows, base_evidence)
    return build_surface_plan_model(gap_model, order_model, qualifier_model)


def _request(roles: tuple[str, ...], ordinal: int = 0):
    return SurfaceStructureRequest(
        SurfaceSemantic("p-g10-heldout", "state", "设备C", "状态", "待机"),
        "ANSWER", "neutral", roles, 2, 50, ordinal,
        "src-g10-heldout", "ctx-g10-heldout", "fam-g10-heldout",
    )


def test_g10_normalizes_gap_and_role_order_options_without_loss() -> None:
    model, _, _ = _models()
    forward = realize_surface_plan(model, _request(("subject", "predicate", "object"), 0))
    forward_alt = realize_surface_plan(model, _request(("subject", "predicate", "object"), 2))
    reverse = realize_surface_plan(model, _request(("object", "predicate", "subject"), 0))
    assert forward.status_code == forward_alt.status_code == reverse.status_code == PLAN_SELECTED
    assert {forward.surface, forward_alt.surface} >= {
        "设备C状态为待机。", "设备C的状态是待机。",
    }
    assert reverse.surface == "待机状态属于设备C。"
    assert forward.surface != reverse.surface
    assert forward.output_bytes == tuple(forward.surface.encode("utf-8"))


def test_g10_unknown_role_order_fails_closed_and_trace_is_deterministic() -> None:
    model, _, _ = _models()
    request = _request(("object", "subject", "predicate"), 0)
    first = realize_surface_plan(model, request)
    second = realize_surface_plan(model, request)
    assert first.status_code == PLAN_NO_PATTERN
    assert first.surface is None
    assert first.canonical_record() == second.canonical_record()


def test_g10_unified_plan_runs_through_g6_focus() -> None:
    model, g7_rows, g7_evidence = _models()
    # The old adapter is deliberately built from a single known forward surface;
    # G10's reverse output remains the independent shadow under test.
    forward_rows = tuple(item for item in g7_rows if item.sample_id in {"g7-a", "g7-b"})
    a01 = SurfaceEvidencePack(
        g7_evidence.source_namespace, g7_evidence.license_id,
        tuple(item for item in g7_evidence.entries if item.variant_id == "a01"),
    )
    old_model = learn_surface_structure_model(forward_rows, a01)
    consumer = RawPropositionConsumerResult(
        "p-g10-heldout", "q-g10-supported", "obs-g10-heldout",
        "src-g10-heldout", "ctx-g10-heldout", "fam-g10-heldout",
        "SUPPORTED", "ANSWER", ("e-g10",), (1, 10),
    )
    plan = SurfaceShadowPlan(
        SurfaceSemantic("p-g10-heldout", "state", "设备C", "状态", "待机"),
        "ANSWER", "neutral", ("object", "predicate", "subject"),
        ("p-g10-heldout",), (), ("src-g10-heldout",),
        "ctx-g10-heldout", "fam-g10-heldout", "legacy", 2, 50,
    )
    adapter = run_raw_t1_shadow_adapter(old_model, consumer, plan)
    result = run_unified_focus_surface_turn(
        start_raw_t1_shadow_dialogue((65010, 10, 1)), adapter, model)
    assert result.surface == "待机状态属于设备C。"
    assert result.dialogue_turn.after.focus_revision == 1
    assert result.replaced == 0


def test_g11_qualifier_pattern_is_preserved_for_new_identity() -> None:
    model = _models_with_qualifier()
    request = SurfaceStructureRequest(
        SurfaceSemantic("p-g11-heldout", "fact", "西岸入口", "启用时间", "2031年4月"),
        "ANSWER", "polite", ("subject", "predicate", "qualifier", "object"),
        2, 80, 0, "src-g11-heldout", "ctx-g11-heldout", "fam-g11-heldout",
        ("西岸入口", "启用时间", "审计记录", "2031年4月"),
    )
    result = realize_surface_plan(model, request)
    assert result.status_code == PLAN_SELECTED
    assert result.surface == "西岸入口的启用时间（审计记录）为2031年4月。"
    assert "东岸入口" not in result.surface
    assert "档案记载" not in result.surface
    assert result.output_bytes == tuple(result.surface.encode("utf-8"))
    qualifier_options = tuple(
        option for pattern in model.patterns for option in pattern.options
        if "qualifier" in option.roles
    )
    assert qualifier_options
    assert all(16 in option.origins for option in qualifier_options)


def test_g11_qualifier_register_drift_fails_closed() -> None:
    model = _models_with_qualifier()
    request = SurfaceStructureRequest(
        SurfaceSemantic("p-g11-drift", "fact", "西岸入口", "启用时间", "2031年4月"),
        "ANSWER", "neutral", ("subject", "predicate", "qualifier", "object"),
        2, 80, 0, "src-g11-drift", "ctx-g11-drift", "fam-g11-drift",
        ("西岸入口", "启用时间", "审计记录", "2031年4月"),
    )
    result = realize_surface_plan(model, request)
    assert result.status_code == PLAN_NO_PATTERN
    assert result.surface is None
