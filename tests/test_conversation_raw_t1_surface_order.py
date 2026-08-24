"""T1-G9 held-out role-order shadow."""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceStructureRequest,
    load_surface_evidence_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_order import (
    ORDER_NO_PATTERN,
    ORDER_SELECTED,
    learn_surface_order_model,
    realize_surface_order,
)


_ROOT = Path(__file__).resolve().parents[1]
_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g9_surface_order_course_v1.jsonl.sample"
_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g9_surface_order_evidence_v1.jsonl.sample"


def _model():
    records = tuple(item.record for item in load_surface_organization_jsonl(_COURSE.read_bytes()))
    evidence = load_surface_evidence_jsonl(_EVIDENCE.read_bytes())
    return learn_surface_order_model(records, evidence)


def _request(roles: tuple[str, ...]) -> SurfaceStructureRequest:
    return SurfaceStructureRequest(
        SurfaceSemantic("p-g9-heldout", "state", "装置C", "状态", "待机"),
        "ANSWER", "neutral", roles, 2, 40, 0,
        "src-g9-heldout", "ctx-g9-heldout", "fam-g9-heldout",
    )


def test_g9_learns_two_role_orders_from_independent_families() -> None:
    model = _model()
    assert len(model.patterns) == 1
    options = model.patterns[0].options
    assert {item.roles for item in options} == {
        ("subject", "predicate", "object"),
        ("object", "predicate", "subject"),
    }
    assert all(len(item.support_family_ids) >= 2 for item in options)


def test_g9_recomposes_unseen_identity_in_both_orders() -> None:
    model = _model()
    forward = realize_surface_order(model, _request(("subject", "predicate", "object")))
    reverse = realize_surface_order(model, _request(("object", "predicate", "subject")))
    assert forward.status_code == reverse.status_code == ORDER_SELECTED
    assert forward.surface in {"装置C状态是待机。", "装置C的状态为待机。"}
    assert reverse.surface in {"待机状态属于装置C。", "待机的状态属于装置C。"}
    assert forward.surface != reverse.surface
    assert forward.output_bytes == tuple(forward.surface.encode("utf-8"))
    assert forward.request.family_id == reverse.request.family_id == "fam-g9-heldout"


def test_g9_unlearned_role_order_fails_closed() -> None:
    result = realize_surface_order(
        _model(), _request(("subject", "object", "predicate")))
    assert result.status_code == ORDER_NO_PATTERN
    assert result.surface is None
