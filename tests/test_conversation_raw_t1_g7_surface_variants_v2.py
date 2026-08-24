"""G7-v2：两个方向都具有唯一 literal slot 边界的公开课程。"""
from pathlib import Path

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceStructureRequest,
    load_surface_evidence_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_variants import (
    VARIANT_SELECTED,
    learn_surface_variant_model,
    realize_surface_variants,
)


_ROOT = Path(__file__).resolve().parents[1]
_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_course_v2.jsonl.sample"
_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_evidence_v2.jsonl.sample"


def test_g7_v2_has_two_uniquely_extractable_gap_options() -> None:
    records = tuple(item.record for item in
                    load_surface_organization_jsonl(_COURSE.read_bytes()))
    evidence = load_surface_evidence_jsonl(_EVIDENCE.read_bytes())
    model = learn_surface_variant_model(records, evidence)
    assert len(model.patterns) == 1
    assert model.patterns[0].support_family_ids == ("fam-g7-c", "fam-g7-d")
    assert model.patterns[0].gap_options == (
        ("", "的", "为", "。"),
        ("", "的", "是", "。"),
    )
    for ordinal, expected in enumerate((
            "设备C的状态为待机。", "设备C的状态是待机。")):
        request = SurfaceStructureRequest(
            SurfaceSemantic("g7-v2-heldout", "state", "设备C", "状态", "待机"),
            "ANSWER", "neutral", ("subject", "predicate", "object"),
            2, 40, ordinal, "src-g7-v2-heldout", "ctx-g7-v2-heldout",
            "fam-g7-v2-heldout",
        )
        result = realize_surface_variants(model, request)
        assert result.status_code == VARIANT_SELECTED
        assert result.surface == expected
        assert result.canonical_record()
