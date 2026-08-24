"""T1-G5 未见 source/context/family 的 relation shadow conformance。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    load_surface_evidence_jsonl,
    learn_surface_structure_model,
)
from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    parse_raw_lexical_evidence_record,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    parse_raw_qualification_record,
    consume_raw_proposition_relation,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    bind_raw_proposition_relation,
    parse_raw_proposition_record,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_adapter import (
    RawT1ShadowAdapterError,
    run_raw_t1_shadow_adapter,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    parse_raw_text_observation_record,
)
from pure_integer_ai.experiments.ph2_dataset_core import parse_canonical_json_bytes
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_shadow import (
    SurfaceShadowPlan,
)


_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "tests/fixtures/dlg_raw_t1_g5_heldout_shadow_v1.json"
_COURSE = _ROOT / "data/ph2/dlg_raw16_surface_organization_v1.jsonl.sample"
_SURFACE_EVIDENCE = _ROOT / "data/ph2/dlg_raw16_surface_slot_evidence_v1.jsonl.sample"


@pytest.fixture(scope="module")
def fixture_rows():
    payload = _FIXTURE.read_bytes()
    assert payload.endswith(b"\n")
    root = parse_canonical_json_bytes(payload[:-1], require_object=True)
    assert root["record_kind"] == "DLG_RAW_T1_G5_HELDOUT_SHADOW_V1"
    assert root["schema_version"] == 1
    assert root["license_id"] == "CC0-1.0"
    assert isinstance(root["cases"], list) and len(root["cases"]) == 3
    return tuple(root["cases"])


@pytest.fixture(scope="module")
def models():
    records = tuple(item.record for item in
                    load_surface_organization_jsonl(_COURSE.read_bytes()))
    evidence = load_surface_evidence_jsonl(_SURFACE_EVIDENCE.read_bytes())
    return {
        "ANSWER": learn_surface_structure_model(
            tuple(item for item in records if item.sample_id in {"s03", "s04"}),
            evidence),
        "CLARIFY": learn_surface_structure_model(
            tuple(item for item in records if item.sample_id in {"s09", "s10"}),
            evidence),
        "UNKNOWN": learn_surface_structure_model(
            tuple(item for item in records if item.sample_id in {"s13", "s14"}),
            evidence),
    }


def _build_case(case: dict):
    observation = parse_raw_text_observation_record(case["observation"])
    lexical = tuple(parse_raw_lexical_evidence_record(item)
                    for item in case["lexical_evidence"])
    proposition = parse_raw_proposition_record(case["proposition"])
    qualification = parse_raw_qualification_record(case["qualification"])
    binding = bind_raw_proposition_relation(observation, lexical, proposition)
    consumer = consume_raw_proposition_relation(binding, qualification)
    semantic = case["plan"]["semantic"]
    plan = case["plan"]
    shadow_plan = SurfaceShadowPlan(
        SurfaceSemantic(
            semantic["proposition_id"], semantic["kind"], semantic["subject"],
            semantic["predicate"], semantic["object"],
        ),
        plan["dialogue_act"], plan["register"], tuple(plan["ordered_roles"]),
        tuple(plan["required_proposition_ids"]),
        tuple(plan["forbidden_proposition_ids"]),
        tuple(plan["authorized_source_ids"]), plan["context_id"],
        plan["family_id"], plan["legacy_surface"], plan["min_chars"],
        plan["max_chars"], tuple(plan["slot_values"]),
    )
    return observation, consumer, shadow_plan


def test_fixture_is_three_way_heldout_and_identity_disjoint(fixture_rows) -> None:
    identities = []
    for case in fixture_rows:
        observation, consumer, plan = _build_case(case)
        assert observation.split == "heldout"
        assert consumer.source_id.startswith("src-g5-")
        assert consumer.context_id.startswith("ctx-g5-")
        assert consumer.family_id.startswith("fam-g5-")
        assert plan.context_id == consumer.context_id
        identities.append((consumer.source_id, consumer.context_id, consumer.family_id))
    assert len(set(identities)) == 3


def test_unseen_supported_relation_recomposes_answer_shadow(fixture_rows, models) -> None:
    observation, consumer, plan = _build_case(fixture_rows[0])
    result = run_raw_t1_shadow_adapter(models["ANSWER"], consumer, plan)
    assert result.shadow.shadow_surface is not None
    assert "C1" in result.shadow.shadow_surface
    assert "dock-freeze" in result.shadow.shadow_surface
    assert result.replaced == 0
    assert observation.observation_id not in {"obs-causal-01", "obs-g0-train-01"}


@pytest.mark.parametrize("index,model_key,expected_act", [
    (1, "UNKNOWN", "UNKNOWN"),
    (2, "CLARIFY", "CLARIFY"),
])
def test_unseen_unknown_and_conflict_keep_zero_claim_obligation(
        fixture_rows, models, index, model_key, expected_act) -> None:
    _, consumer, plan = _build_case(fixture_rows[index])
    assert consumer.response_act == expected_act
    result = run_raw_t1_shadow_adapter(models[model_key], consumer, plan)
    assert result.shadow.shadow_surface is not None
    assert result.shadow.plan.required_proposition_ids == ()
    assert result.replaced == 0


def test_heldout_plan_drift_is_rejected_before_shadow(fixture_rows, models) -> None:
    _, consumer, plan = _build_case(fixture_rows[0])
    drifted = SurfaceShadowPlan(
        plan.semantic, plan.dialogue_act, plan.register, plan.ordered_roles,
        plan.required_proposition_ids, plan.forbidden_proposition_ids,
        ("different-source",), plan.context_id, plan.family_id,
        plan.legacy_surface, plan.min_chars, plan.max_chars, plan.slot_values,
    )
    with pytest.raises(RawT1ShadowAdapterError, match="source"):
        run_raw_t1_shadow_adapter(models["ANSWER"], consumer, drifted)
