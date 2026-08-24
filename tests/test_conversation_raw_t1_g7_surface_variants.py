"""T1-G7 held-out surface gap alternatives with explicit evidence."""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    load_surface_evidence_jsonl,
    SurfaceStructureRequest,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_variants import (
    VARIANT_NO_PATTERN,
    VARIANT_SELECTED,
    learn_surface_variant_model,
    realize_surface_variants,
)


_ROOT = Path(__file__).resolve().parents[1]
_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_course_v1.jsonl.sample"
_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_evidence_v1.jsonl.sample"


def _model():
    records = tuple(item.record for item in load_surface_organization_jsonl(_COURSE.read_bytes()))
    evidence = load_surface_evidence_jsonl(_EVIDENCE.read_bytes())
    return learn_surface_variant_model(records, evidence)


def _request(selection_ordinal: int = 0, *, roles=("subject", "predicate", "object")):
    return SurfaceStructureRequest(
        SurfaceSemantic("p-g7-heldout", "state", "设备C", "状态", "待机"),
        "ANSWER", "neutral", tuple(roles), 2, 40, selection_ordinal,
        "src-g7-heldout", "ctx-g7-heldout", "fam-g7-heldout",
    )


def test_g7_learns_two_gap_options_from_two_independent_families() -> None:
    model = _model()
    assert len(model.patterns) == 1
    pattern = model.patterns[0]
    assert pattern.support_family_ids == ("fam-g7-a", "fam-g7-b")
    assert len(pattern.gap_options) == 2
    assert set(pattern.support_record_ids) == {"g7-a", "g7-b"}


def test_g7_recomposes_both_readable_forms_for_unseen_identity() -> None:
    model = _model()
    first = realize_surface_variants(model, _request(0))
    second = realize_surface_variants(model, _request(1))
    assert first.status_code == second.status_code == VARIANT_SELECTED
    assert first.surface == "设备C状态为待机。"
    assert second.surface == "设备C的状态是待机。"
    assert first.surface != second.surface
    assert first.output_bytes == tuple(first.surface.encode("utf-8"))
    assert first.request.source_id == "src-g7-heldout"
    assert first.request.family_id == "fam-g7-heldout"
    assert first.selected_pattern_id == second.selected_pattern_id
    assert first.canonical_record() != second.canonical_record()


def test_g7_unknown_structure_fails_closed_and_records_are_deterministic() -> None:
    model = _model()
    request = _request(3)
    first = realize_surface_variants(model, request)
    second = realize_surface_variants(model, request)
    assert first.canonical_record() == second.canonical_record()
    miss = realize_surface_variants(model, _request(0, roles=("object", "predicate", "subject")))
    assert miss.status_code == VARIANT_NO_PATTERN
    assert miss.surface is None
