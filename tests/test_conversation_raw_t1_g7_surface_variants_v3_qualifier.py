"""G7-v3 qualifier slot：验证非三元组槽位不会被丢弃。"""
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
_COURSE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_course_v3_qualifier.jsonl.sample"
_EVIDENCE = _ROOT / "data/ph2/dlg_raw_t1_g7_surface_variant_evidence_v3_qualifier.jsonl.sample"


def test_g7_v3_preserves_explicit_qualifier_slot() -> None:
    rows = tuple(item.record for item in
                 load_surface_organization_jsonl(_COURSE.read_bytes()))
    evidence = load_surface_evidence_jsonl(_EVIDENCE.read_bytes())
    model = learn_surface_variant_model(rows, evidence)
    assert model.patterns[0].roles == (
        "subject", "predicate", "qualifier", "object")
    for ordinal, expected in enumerate((
            "设备C的状态（审计记录）为待机。",
            "设备C的状态（审计记录）是待机。")):
        result = realize_surface_variants(
            model,
            SurfaceStructureRequest(
                SurfaceSemantic("g7-v3", "qualified_state", "设备C", "状态", "待机"),
                "ANSWER", "neutral",
                ("subject", "predicate", "qualifier", "object"),
                2, 48, ordinal, "src-g7-v3", "ctx-g7-v3", "fam-g7-v3",
                ("设备C", "状态", "审计记录", "待机"),
            ),
        )
        assert result.status_code == VARIANT_SELECTED
        assert result.surface == expected
