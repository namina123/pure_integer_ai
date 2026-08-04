"""W08-01 五维 registry、train schema 与 Evidence 覆盖专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w08_authority import W08_SUBTASK_ORDER
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CARRIER_KEYS,
    W08_CONSUMER_KEYS,
    W08_LEARNING_PACK_KEYS,
    W08_RESOURCE_BUDGET,
    W08_STOP_STATES,
    make_w08_request,
    open_w08_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w08_firewall import W08PayloadFirewall
from pure_integer_ai.experiments.ph2_w08_payload import W08TrainingPayload
from pure_integer_ai.experiments.ph2_w08_registry import (
    W08_DIMENSION_REGISTRY,
    W08_PACK_REGISTRY,
    W08RegistryError,
    audit_w08_registry_payload,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def payload():
    context = open_w08_frozen_contract(ROOT)
    return W08PayloadFirewall.open(
        ROOT, context, make_w08_request(context)
    ).read_training_payload()


def test_w08_registry_freezes_five_schema_and_one_shared_carrier_projection():
    assert tuple(W08_DIMENSION_REGISTRY) == W08_SUBTASK_ORDER
    assert tuple(W08_PACK_REGISTRY) == W08_LEARNING_PACK_KEYS
    for schema in W08_DIMENSION_REGISTRY.values():
        assert schema.request_kind and schema.result_kind
        assert schema.consumer_keys == W08_CONSUMER_KEYS
        assert schema.outcome_states == W08_STOP_STATES
        assert set(schema.resource_fields) == set(W08_RESOURCE_BUDGET)
        assert {"use_key", "outcome_key", "logical_clock"}.issubset(
            schema.trace_fields
        )
        assert schema.allowed_write_owners == ("PH2_W08_TRANSACTION_OWNER",)
    assert len(W08_CARRIER_KEYS) == 9
    assert len(set(W08_CARRIER_KEYS)) == 9
    assert all("carrier" not in schema.request_kind.lower()
               for schema in W08_DIMENSION_REGISTRY.values())


def test_w08_registry_audits_current_train_payload(payload):
    report = audit_w08_registry_payload(payload)
    assert report.observation_count == report.teacher_evidence_count == 63
    assert report.source_ref_count == 120
    assert report.pack_counts == (
        (W08_LEARNING_PACK_KEYS[0], 9),
        (W08_LEARNING_PACK_KEYS[1], 16),
        (W08_LEARNING_PACK_KEYS[2], 17),
        (W08_LEARNING_PACK_KEYS[3], 17),
        (W08_LEARNING_PACK_KEYS[4], 2),
        (W08_LEARNING_PACK_KEYS[5], 2),
    )
    assert {"support", "refute", "conflict", "supersede"}.issubset(
        report.sample_roles
    )
    assert report.definitive_truth_claims == 0
    assert report.expected_or_label_fields == 0


def test_w08_registry_rejects_stage_spoof_without_state_write(payload):
    first = payload.observations[0]
    invalid = W08TrainingPayload(
        payload.source_refs,
        (replace(first, w_stage="W-07"), *payload.observations[1:]),
        payload.teacher_evidence,
    )
    with pytest.raises(W08RegistryError, match="pack registry"):
        audit_w08_registry_payload(invalid)


def test_w08_registry_rejects_unbound_evidence(payload):
    invalid = W08TrainingPayload(
        payload.source_refs,
        payload.observations,
        payload.teacher_evidence[:-1],
    )
    with pytest.raises(W08RegistryError, match="coverage"):
        audit_w08_registry_payload(invalid)
