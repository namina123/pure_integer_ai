"""W08-09 NE 恢复：真实 Candidate inference state 与逐 case adapter。"""
from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import tempfile

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_contract import (
    make_w08_request,
    open_w08_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w08_firewall import W08PayloadFirewall
from pure_integer_ai.experiments.ph2_w08_inference import (
    W08CandidateInferenceAdapter,
    validate_w08_inference_outcome,
)
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08_CANDIDATE_INFERENCE_INTERFACE_VERSION,
    W08CandidateInferenceError,
    W08CandidateInferenceState,
)
from pure_integer_ai.experiments.ph2_w08_inference_training import (
    compile_w08_candidate_inference_state,
)
from pure_integer_ai.experiments.ph2_w08_runtime import (
    load_w08_candidate_inference_state,
    load_w08_public_dump,
    run_language_stage8_public,
)
from pure_integer_ai.experiments.ph2_w08_runtime_contract import W08RuntimeConfig


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def training_fixture():
    contract = open_w08_frozen_contract(ROOT)
    payload = W08PayloadFirewall.open(
        ROOT, contract, make_w08_request(contract)
    ).read_training_payload()
    state = compile_w08_candidate_inference_state(payload)
    return payload, state


def _held_out(observation):
    return replace(observation, split="held_out")


def _course_observation(payload, *, kind: str):
    return next(
        item
        for item in payload.observations
        if item.payload_kind == "OpenSetClarificationCandidateV1"
        and item.typed_payload.to_value()["candidate_kind"] == kind
    )


def test_train_state_is_versioned_executable_and_contains_no_answer_or_text(
    training_fixture,
):
    payload, state = training_fixture
    encoded = canonical_json_bytes(state.to_dict())
    assert state.interface_version == W08_CANDIDATE_INFERENCE_INTERFACE_VERSION
    assert len(state.rules) == 60
    assert W08CandidateInferenceState.from_dict(state.to_dict()) == state
    assert all(token not in encoded for token in (
        b'"expected',
        b'"label',
        b'"surface',
        b'"observed_surface"',
        b'"accepted_surfaces"',
    ))
    visible_texts = [
        item.typed_payload.to_value().get("observed_surface", {}).get("text")
        for item in payload.observations
    ]
    assert all(
        not text or text.encode("utf-8") not in encoded for text in visible_texts
    )
    signature = inspect.signature(W08CandidateInferenceAdapter.infer)
    assert tuple(signature.parameters) == (
        "self", "observation", "dimension_key", "disabled_components"
    )


def test_every_train_shape_executes_as_a_held_out_case_without_owner_or_shortcut(
    training_fixture,
):
    payload, state = training_fixture
    adapter = W08CandidateInferenceAdapter(state)
    outcomes = tuple(
        adapter.infer(_held_out(observation), dimension_key=dimension)
        for observation in payload.observations
        for dimension in W08_DIMENSION_KEYS
    )
    assert len(outcomes) == 315
    assert len({item.invocation_key for item in outcomes}) == len(outcomes)
    assert all(item.component_state == "ACTIVE" for item in outcomes)
    assert all(not any(value for _, value in item.owner_counts) for item in outcomes)
    assert all(not any(value for _, value in item.shortcut_counts) for item in outcomes)
    by_key = {
        item.stable_key.components: _held_out(item) for item in payload.observations
    }
    assert all(validate_w08_inference_outcome(by_key[item.observation_key], item)
               for item in outcomes)


def test_unknown_clarify_budget_and_schema_failure_remain_explicit(training_fixture):
    payload, state = training_fixture
    adapter = W08CandidateInferenceAdapter(state)
    unknown = _held_out(_course_observation(payload, kind="UNKNOWN"))
    outcome = adapter.infer(unknown, dimension_key=W08_DIMENSION_KEYS[1])
    assert outcome.actual_state == "UNKNOWN"
    assert outcome.actual_payload.to_value()["resolution_state"] == "CLARIFY"
    assert tuple(value for _, value in outcome.consumer_states) == ("RESOLVED",) * 3

    raw = unknown.typed_payload.to_value()
    raw["resource_budget"]["max_output_units"] = 161
    exhausted = replace(
        unknown,
        typed_payload=CanonicalJsonObject.from_value(raw),
    )
    budget = adapter.infer(exhausted, dimension_key=W08_DIMENSION_KEYS[4])
    assert budget.component_state == "FAIL_CLOSED"
    assert tuple(value for _, value in budget.consumer_states) == (
        "BUDGET_EXHAUSTED",
    ) * 3
    assert budget.actual_payload.to_value()["publication"] == 0

    drift = unknown.typed_payload.to_value()
    drift["unregistered_field"] = 1
    with pytest.raises(W08CandidateInferenceError, match="schema 漂移"):
        adapter.infer(
            replace(unknown, typed_payload=CanonicalJsonObject.from_value(drift)),
            dimension_key=W08_DIMENSION_KEYS[0],
        )


def test_each_ablation_disables_a_real_component_and_reinvokes(training_fixture):
    payload, state = training_fixture
    observation = _held_out(_course_observation(payload, kind="NEW_USAGE_DETECTION"))
    adapter = W08CandidateInferenceAdapter(state)
    baseline = tuple(
        adapter.infer(observation, dimension_key=dimension)
        for dimension in W08_DIMENSION_KEYS
    )
    assert all(item.component_state == "ACTIVE" for item in baseline)
    for target in W08_DIMENSION_KEYS:
        rerun = tuple(
            adapter.infer(
                observation,
                dimension_key=dimension,
                disabled_components=(target,),
            )
            for dimension in W08_DIMENSION_KEYS
        )
        assert len({item.invocation_key for item in (*baseline, *rerun)}) == 10
        assert tuple(item.component_state for item in rerun) == tuple(
            "DISABLED" if dimension == target else "ACTIVE"
            for dimension in W08_DIMENSION_KEYS
        )
        assert tuple(dict(item.consumer_states)["REASONING"] for item in rerun) == tuple(
            "FAIL_CLOSED" if dimension == target else "RESOLVED"
            for dimension in W08_DIMENSION_KEYS
        )


def test_dump_restart_and_adapter_replay_are_bit_identical(training_fixture):
    payload, _ = training_fixture
    observation = _held_out(_course_observation(payload, kind="NEW_USAGE_DETECTION"))
    with tempfile.TemporaryDirectory(
        prefix="w08-inference-replay-", dir=ROOT.parent
    ) as value:
        run_root = Path(value)
        config = W08RuntimeConfig(
            ROOT,
            run_root,
            run_root / "coordinator.sqlite",
            worker_count=4,
            mode="fresh",
        )
        outcome = run_language_stage8_public(config)
        readback = load_w08_public_dump(config)
        state = load_w08_candidate_inference_state(config)
        first = W08CandidateInferenceAdapter(state).infer(
            observation, dimension_key=W08_DIMENSION_KEYS[0]
        )
        second = W08CandidateInferenceAdapter(
            load_w08_candidate_inference_state(config)
        ).infer(observation, dimension_key=W08_DIMENSION_KEYS[0])
    assert outcome.inference_state_key == readback.inference_state_key == state.state_key
    assert outcome.inference_state_sha256 == readback.inference_state_sha256 == state.sha256()
    assert first == second
