"""T1-G6 source-qualified shadow 的连续 ANSWER/UNKNOWN/CLARIFY focus 链。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_selector import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_shadow import (
    SurfaceShadowPlan,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    load_surface_evidence_jsonl,
    learn_surface_structure_model,
)
from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    parse_raw_lexical_evidence_record,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionQualification,
    consume_raw_proposition_relation,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    bind_raw_proposition_relation,
    parse_raw_proposition_record,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_adapter import (
    run_raw_t1_shadow_adapter,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_dialogue import (
    RawT1ShadowDialogueError,
    run_raw_t1_shadow_dialogue_turn,
    start_raw_t1_shadow_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    parse_raw_text_observation_record,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


_ROOT = Path(__file__).resolve().parents[1]
_G5_FIXTURE = _ROOT / "tests/fixtures/dlg_raw_t1_g5_heldout_shadow_v1.json"
_G6_FIXTURE = _ROOT / "tests/fixtures/dlg_raw_t1_g6_focus_chain_v1.json"
_COURSE = _ROOT / "data/ph2/dlg_raw16_surface_organization_v1.jsonl.sample"
_SURFACE_EVIDENCE = _ROOT / "data/ph2/dlg_raw16_surface_slot_evidence_v1.jsonl.sample"


def _models():
    records = tuple(item.record for item in
                    load_surface_organization_jsonl(_COURSE.read_bytes()))
    evidence = load_surface_evidence_jsonl(_SURFACE_EVIDENCE.read_bytes())
    return {
        "ANSWER": learn_surface_structure_model(
            tuple(item for item in records if item.sample_id in {"s03", "s04"}), evidence),
        "CLARIFY": learn_surface_structure_model(
            tuple(item for item in records if item.sample_id in {"s09", "s10"}), evidence),
        "UNKNOWN": learn_surface_structure_model(
            tuple(item for item in records if item.sample_id in {"s13", "s14"}), evidence),
    }


def _base_binding_and_identity():
    root = parse_canonical_json_bytes(_G5_FIXTURE.read_bytes()[:-1], require_object=True)
    case = root["cases"][0]
    observation = parse_raw_text_observation_record(case["observation"])
    lexical = tuple(parse_raw_lexical_evidence_record(item)
                    for item in case["lexical_evidence"])
    proposition = parse_raw_proposition_record(case["proposition"])
    binding = bind_raw_proposition_relation(observation, lexical, proposition)
    return observation, binding


def _adapter_result(state: str, models):
    observation, binding = _base_binding_and_identity()
    evidence_ids = tuple(item.evidence_id for item in binding.arguments)
    qualification = RawPropositionQualification(
        f"q-g6-{state.lower()}", binding.proposition_id, observation.observation_id,
        observation.source_id, observation.context_id, observation.family_id,
        observation.source_namespace, observation.split, state,
        f"{state.lower()}-g6", evidence_ids, "g6-public-qualification-v1",
    )
    consumer = consume_raw_proposition_relation(binding, qualification)
    if state == "SUPPORTED":
        plan = SurfaceShadowPlan(
            SurfaceSemantic(binding.proposition_id, "g6", "C1", "causes", "dock-freeze"),
            "ANSWER", "neutral", ("cause", "relation", "effect"),
            (binding.proposition_id,), (), (observation.source_id,),
            observation.context_id, observation.family_id, "legacy", 2, 120,
        )
        model = models["ANSWER"]
    elif state == "UNKNOWN":
        plan = SurfaceShadowPlan(
            SurfaceSemantic(binding.proposition_id, "g6", "U1", "lacks", "dock-budget"),
            "UNKNOWN", "neutral", ("source", "scope"),
            (), (binding.proposition_id,), (observation.source_id,),
            observation.context_id, observation.family_id, "legacy", 2, 120,
            ("current", "dock-budget"),
        )
        model = models["UNKNOWN"]
    else:
        plan = SurfaceShadowPlan(
            SurfaceSemantic(binding.proposition_id, "g6", "L1", "causes", "bank-freeze"),
            "CLARIFY", "polite", ("choice", "target"),
            (), (), (observation.source_id,), observation.context_id,
            observation.family_id, "legacy", 2, 120,
            ("site-A-or-site-B", "scope"),
        )
        model = models["CLARIFY"]
    return run_raw_t1_shadow_adapter(model, consumer, plan)


def test_g6_fixture_replays_three_focus_revisions() -> None:
    fixture = parse_canonical_json_bytes(_G6_FIXTURE.read_bytes()[:-1], require_object=True)
    assert fixture["record_kind"] == "DLG_RAW_T1_G6_FOCUS_CHAIN_V1"
    assert fixture["schema_version"] == 1
    models = _models()
    state = start_raw_t1_shadow_dialogue((65001, 6, 1))
    for expected in fixture["steps"]:
        adapter = _adapter_result(expected["state"], models)
        turn = run_raw_t1_shadow_dialogue_turn(state, adapter)
        assert turn.after.focus_revision == expected["focus_revision"]
        assert adapter.consumer.response_act == expected["response_act"]
        assert turn.after.last_state == expected["state"]
        assert turn.after.proposition_id == adapter.consumer.proposition_id
        assert adapter.replaced == 0
        state = turn.after
    assert state.next_operation_ordinal == 4


def test_g6_first_turn_requires_supported_answer() -> None:
    models = _models()
    with pytest.raises(RawT1ShadowDialogueError, match="首轮"):
        run_raw_t1_shadow_dialogue_turn(
            start_raw_t1_shadow_dialogue((65001, 6, 2)),
            _adapter_result("UNKNOWN", models),
        )


def test_g6_followup_focus_and_act_drift_are_rejected() -> None:
    models = _models()
    first = start_raw_t1_shadow_dialogue((65001, 6, 3))
    answer = _adapter_result("SUPPORTED", models)
    after_answer = run_raw_t1_shadow_dialogue_turn(first, answer).after
    with pytest.raises(RawT1ShadowDialogueError, match="顺序"):
        run_raw_t1_shadow_dialogue_turn(after_answer, answer)
    unknown = _adapter_result("UNKNOWN", models)
    after_unknown = run_raw_t1_shadow_dialogue_turn(after_answer, unknown).after
    with pytest.raises(RawT1ShadowDialogueError, match="顺序"):
        run_raw_t1_shadow_dialogue_turn(after_unknown, unknown)


def test_g6_state_and_turn_records_are_deterministic() -> None:
    models = _models()
    state = start_raw_t1_shadow_dialogue((65001, 6, 4))
    answer = _adapter_result("SUPPORTED", models)
    first = run_raw_t1_shadow_dialogue_turn(state, answer)
    second = run_raw_t1_shadow_dialogue_turn(state, answer)
    assert first.canonical_record() == second.canonical_record()
    assert all(type(item) is int for item in first.canonical_record())
