"""T1-G12 negative audit: rejected surface evidence never enters the plan."""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceSlotEvidence,
    load_surface_evidence_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_order import (
    learn_surface_order_model,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_plan import (
    build_surface_plan_model,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_variants import (
    learn_surface_variant_model,
)


_ROOT = Path(__file__).resolve().parents[1]
_G7_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_course_v1.jsonl.sample"
_G7_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_evidence_v1.jsonl.sample"
_G9_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g9_surface_order_course_v1.jsonl.sample"
_G9_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g9_surface_order_evidence_v1.jsonl.sample"


def _rejected_entries(record) -> tuple[SurfaceSlotEvidence, ...]:
    values = {
        "subject": record.proposition_subject,
        "predicate": record.proposition_predicate,
        "object": record.proposition_object,
    }
    entries = []
    for variant in record.rejected:
        accepted_surface = record.accepted[0].surface
        for slot in record.clause_slots:
            value = values[slot.role]
            start = accepted_surface.index(value)
            end = start + len(value)
            entries.append(SurfaceSlotEvidence(
                record.sample_id, variant.variant_id, slot.slot_id, slot.role,
                start, end, variant.surface[start:end],
            ))
    return tuple(entries)


def test_rejected_variants_are_ignored_by_gap_and_order_learners() -> None:
    g7_rows = tuple(item.record for item in load_surface_organization_jsonl(_G7_COURSE.read_bytes()))
    g7_evidence = load_surface_evidence_jsonl(_G7_EVIDENCE.read_bytes())
    augmented_g7 = type(g7_evidence)(
        g7_evidence.source_namespace, g7_evidence.license_id,
        g7_evidence.entries
        + _rejected_entries(g7_rows[0])
        + _rejected_entries(g7_rows[1]),
    )
    original_gap = learn_surface_variant_model(g7_rows, g7_evidence)
    augmented_gap = learn_surface_variant_model(g7_rows, augmented_g7)
    assert original_gap.patterns == augmented_gap.patterns
    assert all("r01" not in pattern.support_record_ids for pattern in augmented_gap.patterns)

    g9_rows = tuple(item.record for item in load_surface_organization_jsonl(_G9_COURSE.read_bytes()))
    g9_evidence = load_surface_evidence_jsonl(_G9_EVIDENCE.read_bytes())
    augmented_g9 = type(g9_evidence)(
        g9_evidence.source_namespace, g9_evidence.license_id,
        g9_evidence.entries
        + tuple(entry for record in g9_rows for entry in _rejected_entries(record)),
    )
    original_order = learn_surface_order_model(g9_rows, g9_evidence)
    augmented_order = learn_surface_order_model(g9_rows, augmented_g9)
    assert original_order.patterns == augmented_order.patterns

    original_plan = build_surface_plan_model(original_gap, original_order)
    augmented_plan = build_surface_plan_model(augmented_gap, augmented_order)
    assert original_plan.patterns == augmented_plan.patterns
