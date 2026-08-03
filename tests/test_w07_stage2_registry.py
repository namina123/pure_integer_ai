"""W07-01 operator registry、四态课程与 train 证据覆盖专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_BASELINE_COMMIT_SHA1,
    W07_FORMAL_RUN_ID,
    W07_RESOURCE_BUDGET,
    W07_RUNNER_KEY,
    W07_STAGE_KEY,
    W07_SUBSTAGE_ORDER,
    W07_W06_BASE_RUN_ID,
    W07RunRequest,
    open_w07_frozen_context,
)
from pure_integer_ai.experiments.ph2_w07_firewall import W07PayloadFirewall
from pure_integer_ai.experiments.ph2_w07_payload import W07TrainingPayload
from pure_integer_ai.experiments.ph2_w07_registry import (
    W07_CONSUMER_KEYS,
    W07_EXPECTED_STATES,
    W07_OPERATOR_REGISTRY,
    W07_REQUIRED_SAMPLE_ROLES,
    W07RegistryError,
    audit_w07_registry_payload,
)
from pure_integer_ai.storage.backend import SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    """经 public firewall 一次性交付真实 W-07 train payload。"""
    path = tmp_path_factory.mktemp("w07-registry") / "probe.sqlite"
    backend = SQLiteBackend(str(path))
    try:
        context = open_w07_frozen_context(
            ROOT,
            baseline_commit_sha1=W07_BASELINE_COMMIT_SHA1,
            backend_profile_key=backend.storage_capabilities().stable_key(),
        )
        request = W07RunRequest(
            W07_FORMAL_RUN_ID,
            W07_W06_BASE_RUN_ID,
            W07_W06_BASE_RUN_ID,
            W07_STAGE_KEY,
            context.owner_key,
            W07_RUNNER_KEY,
            context.baseline_commit_sha1,
            context.stable_key(),
            context.backend_profile_key,
            context.base_fence_key,
            1,
            "fresh",
            tuple(sorted(W07_RESOURCE_BUDGET.items())),
            tuple(item.relative_path for item in context.candidate_payload_bindings),
            tuple(item.relative_path for item in context.teacher_evidence_bindings),
        )
        return W07PayloadFirewall.open(
            ROOT, context, request).read_training_payload()
    finally:
        backend.close()


def test_w07_registry_binds_seven_substages_schema_certificates_and_consumers():
    """七子阶段逐项声明 operator、Role arity、Evidence、P2-G 与 U/R/G。"""
    assert tuple(W07_OPERATOR_REGISTRY) == W07_SUBSTAGE_ORDER
    for entry in W07_OPERATOR_REGISTRY.values():
        assert entry.operator_families
        assert entry.payload_kind and entry.query_kind
        assert entry.teacher_evidence_kind
        assert entry.required_perturbations
        assert entry.certificate_kinds
        assert entry.consumer_keys == W07_CONSUMER_KEYS
        assert entry.teacher_withdrawal_level == 0
    assert W07_REQUIRED_SAMPLE_ROLES == (
        "support", "refute", "conflict", "supersede")
    assert set(W07_EXPECTED_STATES) == {"TRUE", "FALSE", "UNKNOWN", "CONFLICT"}


def test_w07_registry_audits_current_train_payload(payload):
    """当前七 pack 的 schema、四态、负例和 lifecycle 全部命中 registry。"""
    report = audit_w07_registry_payload(payload)
    assert report.observation_count == report.teacher_evidence_count == 66
    assert report.substage_counts == (
        ("NOT", 9),
        ("AND_OR", 11),
        ("CONDITION", 9),
        ("EXISTS", 9),
        ("FORALL", 9),
        ("MODAL", 10),
        ("NESTED_SCOPE", 9),
    )
    assert {item[0] for item in report.operator_family_counts} == {
        "NOT", "AND", "OR", "CONDITION", "EXISTS", "FORALL", "MODAL"}
    assert report.source_keys == ("AUTHORED_CC0_V1",)


def test_w07_registry_rejects_non_w07_observation_without_state_write(payload):
    """其他 stage train record 即使结构相似也不能形成 W07 operator。"""
    first = payload.observations[0]
    mutated = replace(first, w_stage="W-06")
    invalid = W07TrainingPayload(
        payload.source_refs,
        (mutated, *payload.observations[1:]),
        payload.teacher_evidence,
    )
    with pytest.raises(W07RegistryError, match="未匹配 registry"):
        audit_w07_registry_payload(invalid)
