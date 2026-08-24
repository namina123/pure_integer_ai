"""T1-G28：G27 consumer 到 G10/G6 surface shadow 的组合测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

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
    SurfaceStructureModel,
    learn_surface_structure_model,
    load_surface_evidence_jsonl,
)
from pure_integer_ai.experiments.conversation_raw_t1_annotation_consensus import (
    merge_raw_t1_annotation_submissions,
)
from pure_integer_ai.experiments.conversation_raw_t1_annotation_submission import (
    ANNOTATION_ACCEPT,
    RawT1AnnotationDecision,
    RawT1AnnotationSubmission,
)
from pure_integer_ai.experiments.conversation_raw_t1_candidate_granularity import (
    audit_raw_t1_candidate_granularity,
)
from pure_integer_ai.experiments.conversation_raw_t1_consensus_candidate_evidence import (
    project_raw_t1_consensus_candidate_evidence,
)
from pure_integer_ai.experiments.conversation_raw_t1_exact_relation_admission import (
    admit_exact_candidate_relation,
)
from pure_integer_ai.experiments.conversation_raw_t1_relation_surface_shadow import (
    RawT1RelationSurfaceShadowError,
    run_exact_relation_surface_shadow,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_observer import (
    observe_raw_t1_shadow_result,
    render_raw_t1_shadow_observation_zh,
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
from pure_integer_ai.experiments.conversation_raw_t1_shadow_dialogue import (
    RawT1ShadowDialogueState,
    start_raw_t1_shadow_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionConsumerResult,
    RawPropositionQualification,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    RawPropositionRelationEvidence,
    RawRelationArgument,
)
from pure_integer_ai.experiments.conversation_raw_text_candidate_spans import (
    extract_raw_text_candidate_spans,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextSpanUnit,
    compile_raw_text_observation,
)
from pure_integer_ai.experiments.ph2_dataset_core import parse_canonical_json_bytes


_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data/ph2"
_BASE = _DATA / "dlg_raw16_surface_organization_v1.jsonl.sample"
_BASE_E = _DATA / "dlg_raw16_surface_slot_evidence_v1.jsonl.sample"
_G7 = _DATA / "dlg_raw_t1_g7_surface_variant_course_v1.jsonl.sample"
_G7_E = _DATA / "dlg_raw_t1_g7_surface_variant_evidence_v1.jsonl.sample"
_G9 = _DATA / "dlg_raw_t1_g9_surface_order_course_v1.jsonl.sample"
_G9_E = _DATA / "dlg_raw_t1_g9_surface_order_evidence_v1.jsonl.sample"


def _admission(state: str = "SUPPORTED"):
    raw = tuple("设备C状态为待机。".encode("utf-8"))
    observation = compile_raw_text_observation(
        raw, observation_id="obs-g28", source_id="src-g28",
        context_id="ctx-g28", family_id="fam-g28",
        source_namespace="t1-g28-public-v1", split="heldout",
        units=(RawTextSpanUnit("u1", "sentence", 0, len("设备C状态为待机。"), 0, len(raw)),))
    extraction = extract_raw_text_candidate_spans(raw)
    candidate = extraction.candidates[0]

    def submission(scope: str):
        item = RawT1AnnotationDecision(candidate.ordinal, candidate.start_scalar,
            candidate.end_scalar, candidate.start_byte, candidate.end_byte,
            ANNOTATION_ACCEPT, "sentence", 1)
        return RawT1AnnotationSubmission(
            f"ann-{scope}", scope, observation.observation_id,
            observation.source_namespace, (item,))
    consensus = merge_raw_t1_annotation_submissions(
        extraction, observation, (submission("a"), submission("b")))
    candidate_evidence = project_raw_t1_consensus_candidate_evidence(
        consensus, observation, evidence_namespace="g28-public")
    audits = tuple(audit_raw_t1_candidate_granularity(item, observation)
                   for item in candidate_evidence)
    eid = candidate_evidence[0].evidence_id
    proposition = RawPropositionRelationEvidence(
        "p-g28", observation.observation_id, observation.source_id,
        observation.context_id, observation.family_id, observation.source_namespace,
        observation.split, "annotated_sentence", "g28-proposition-v1",
        (RawRelationArgument(eid, "u1", "sentence", 1),))
    qualification = RawPropositionQualification(
        "q-g28", proposition.proposition_id, observation.observation_id,
        observation.source_id, observation.context_id, observation.family_id,
        observation.source_namespace, observation.split, state, "g28-v1",
        (eid,), "g28-qualification-v1")
    return admit_exact_candidate_relation(
        observation, candidate_evidence, audits, proposition, qualification,
        authority="g26-adapter-v1")


def _surface_models():
    base_rows = tuple(item.record for item in load_surface_organization_jsonl(_BASE.read_bytes()))
    base_evidence = load_surface_evidence_jsonl(_BASE_E.read_bytes())
    # G7/G9 models supply the already verified unified plan; old structure model is a thin
    # source-qualified shadow consumer and does not infer from the new sentence.
    g7_rows = tuple(item.record for item in load_surface_organization_jsonl(_G7.read_bytes()))
    g7_evidence = load_surface_evidence_jsonl(_G7_E.read_bytes())
    g9_rows = tuple(item.record for item in load_surface_organization_jsonl(_G9.read_bytes()))
    g9_evidence = load_surface_evidence_jsonl(_G9_E.read_bytes())
    old_rows = tuple(item for item in g7_rows if item.sample_id in {"g7-a", "g7-b"})
    old_pack = SurfaceEvidencePack(
        g7_evidence.source_namespace, g7_evidence.license_id,
        tuple(item for item in g7_evidence.entries if item.variant_id == "a01"))
    old_model = learn_surface_structure_model(old_rows, old_pack)
    gap = learn_surface_variant_model(g7_rows, g7_evidence)
    order = learn_surface_order_model(g9_rows, g9_evidence)
    return old_model, build_surface_plan_model(gap, order)


def test_exact_relation_consumer_reaches_readable_unified_shadow() -> None:
    admission = _admission()
    old_model, unified = _surface_models()
    plan = SurfaceShadowPlan(
        SurfaceSemantic("p-g28", "state", "设备C", "状态", "待机"),
        "ANSWER", "neutral", ("subject", "predicate", "object"),
        ("p-g28",), (), ("src-g28",), "ctx-g28", "fam-g28", "g28",
        2, 50,
    )
    result = run_exact_relation_surface_shadow(
        admission, old_model, plan, start_raw_t1_shadow_dialogue((65028, 28, 1)), unified)

    assert result.surface in {"设备C状态为待机。", "设备C的状态是待机。"}
    assert result.replaced == 0
    assert result.dialogue_turn.after.focus_revision == 1
    observed = observe_raw_t1_shadow_result(result)
    assert observed.surface == result.surface
    assert observed.plan_selected == 1
    assert "developer-only shadow" in render_raw_t1_shadow_observation_zh(observed)


def test_shadow_plan_identity_drift_is_not_fallback() -> None:
    admission = _admission()
    old_model, unified = _surface_models()
    plan = SurfaceShadowPlan(
        SurfaceSemantic("p-other", "state", "设备C", "状态", "待机"),
        "ANSWER", "neutral", ("subject", "predicate", "object"),
        ("p-other",), (), ("src-g28",), "ctx-g28", "fam-g28", "g28", 2, 50)
    with pytest.raises(RawT1RelationSurfaceShadowError, match="失败"):
        run_exact_relation_surface_shadow(
            admission, old_model, plan,
            start_raw_t1_shadow_dialogue((65028, 28, 2)), unified)


def test_unknown_relation_reaches_focus_without_surface_or_claim() -> None:
    admission = _admission("UNKNOWN")
    base_rows = tuple(item.record for item in load_surface_organization_jsonl(_BASE.read_bytes()))
    base_evidence = load_surface_evidence_jsonl(_BASE_E.read_bytes())
    zero_model = learn_surface_structure_model(
        tuple(item for item in base_rows if item.sample_id in {"s13", "s14"}),
        base_evidence,
    )
    _, unified = _surface_models()
    plan = SurfaceShadowPlan(
        SurfaceSemantic("p-g28", "state", "设备C", "状态", "待机"),
        "UNKNOWN", "neutral", ("source", "scope"),
        (), ("p-g28",), ("src-g28",), "ctx-g28", "fam-g28", "g28",
        2, 50, ("当前资料", "该状态"),
    )
    focused_state = RawT1ShadowDialogueState(
        (65028, 28, 3), 2, 1, "p-g28", "src-g28", "ctx-g28", "fam-g28",
        "ANSWER", "SUPPORTED")
    result = run_exact_relation_surface_shadow(
        admission, zero_model, plan, focused_state, unified)

    assert result.surface is None
    assert result.plan_result is None
    assert result.replaced == 0
    assert result.dialogue_turn.after.focus_revision == 2


def test_conflict_followup_reaches_clarify_without_surface_or_claim() -> None:
    base_rows = tuple(item.record for item in load_surface_organization_jsonl(_BASE.read_bytes()))
    base_evidence = load_surface_evidence_jsonl(_BASE_E.read_bytes())
    clarify_model = learn_surface_structure_model(
        tuple(item for item in base_rows if item.sample_id in {"s09", "s10"}),
        base_evidence,
    )
    _, unified = _surface_models()
    consumer = RawPropositionConsumerResult(
        "p-g28", "q-g28-conflict", "obs-g28", "src-g28", "ctx-g28", "fam-g28",
        "CONFLICT", "CLARIFY", ("e-g28-a", "e-g28-b"), (1, 30),
    )
    plan = SurfaceShadowPlan(
        SurfaceSemantic("p-g28", "state", "设备C", "状态", "待机"),
        "CLARIFY", "polite", ("choice", "target"),
        (), ("p-g28",), ("src-g28",), "ctx-g28", "fam-g28", "g28",
        2, 50, ("哪一项", "设备C"),
    )
    from pure_integer_ai.experiments.conversation_raw_t1_shadow_adapter import (
        run_raw_t1_shadow_adapter,
    )
    adapter = run_raw_t1_shadow_adapter(clarify_model, consumer, plan)
    focused_state = RawT1ShadowDialogueState(
        (65028, 28, 4), 3, 2, "p-g28", "src-g28", "ctx-g28", "fam-g28",
        "UNKNOWN", "UNKNOWN")
    result = run_unified_focus_surface_turn(
        focused_state, adapter, unified)

    assert result.surface is None
    assert result.plan_result is None
    assert result.replaced == 0
    assert result.dialogue_turn.after.focus_revision == 3
    observed = observe_raw_t1_shadow_result(result)
    assert observed.surface is None
    assert observed.plan_selected == 0
    assert "无表层输出" in render_raw_t1_shadow_observation_zh(observed)
