"""W05-02 typed adapter 与 H-05/H-04 lifecycle 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ENTITY,
    OBJECT_EVENT,
    ObjectIdentity,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_w05_adapter import (
    W05TypedAdapterError,
    adapt_w05_training_payload,
)
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_FORMAL_RUN_ID,
    W05_RESOURCE_BUDGET,
    W05_RUNNER_KEY,
    W05_STAGE_KEY,
    W05_W04_BASE_RUN_ID,
    W05RunRequest,
    digest_value,
    open_w05_frozen_context,
)
from pure_integer_ai.experiments.ph2_w05_firewall import W05PayloadFirewall
from pure_integer_ai.experiments.ph2_w05_learning import (
    build_w05_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w05_payload import W05TrainingPayload
from pure_integer_ai.storage.backend import SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]
HEAD = "693867db349e0ce05782fbaf6fa2b9206b26b4dc"


def _context_and_payload(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "context.sqlite"))
    try:
        context = open_w05_frozen_context(
            ROOT,
            current_remote_commit_sha1=HEAD,
            backend_profile_key=backend.storage_capabilities().stable_key(),
        )
    finally:
        backend.close()
    request = W05RunRequest(
        run_id=W05_FORMAL_RUN_ID,
        parent_run_id=W05_W04_BASE_RUN_ID,
        base_run_id=W05_W04_BASE_RUN_ID,
        stage_key=W05_STAGE_KEY,
        owner_key=context.owner_key,
        runner_key=W05_RUNNER_KEY,
        current_remote_commit_sha1=context.current_remote_commit_sha1,
        pre_w04_gate_key=context.pre_w04_gate_key,
        w04_receipt_key=digest_value(context.w04_receipt_identity.to_dict()),
        d03_context_key=context.stable_key(),
        backend_profile_key=context.backend_profile_key,
        base_fence_key=context.base_fence_key,
        worker_count=1,
        mode="fresh",
        resource_budget=tuple(sorted(W05_RESOURCE_BUDGET.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )
    payload = W05PayloadFirewall.open(
        ROOT, context, request).read_training_payload()
    return context, payload


def _replace_atomic_observation(payload, mutate):
    observations = list(payload.observations)
    index = next(
        index for index, item in enumerate(observations)
        if item.w_stage == "W-05"
        and item.payload_kind == "AtomicPropositionQuery"
    )
    value = observations[index].typed_payload.to_value()
    mutate(value)
    observations[index] = replace(
        observations[index],
        typed_payload=CanonicalJsonObject.from_value(value),
    )
    return W05TrainingPayload(
        payload.source_refs,
        tuple(observations),
        payload.teacher_evidence,
    )


def test_w05_adapter_restores_occurrence_role_and_proposition_without_cues(
        tmp_path):
    """只消费 atomic typed payload，并保留同 surface、order 与五种扰动。"""
    _context, payload = _context_and_payload(tmp_path)
    adapted = adapt_w05_training_payload(payload)
    assert len(adapted.candidates) == 6
    assert len(adapted.observations) == 6
    assert all(item.observation.w_stage == "W-05" for item in adapted.candidates)
    assert all(
        item.observation.payload_kind == "AtomicPropositionQuery"
        for item in adapted.candidates
    )
    assert {item.perturbation_kind for item in adapted.candidates} == {
        "NONE",
        "ROLE_SWAP",
        "ORDER_REVERSAL",
        "SCOPE_SHIFT",
        "OCCURRENCE_OMISSION",
        "OCCURRENCE_RESTORE",
    }
    assert {item.perturbation_kind for item in adapted.evidence} == {
        "NONE",
        "ROLE_SWAP",
        "ORDER_REVERSAL",
        "SCOPE_SHIFT",
        "OCCURRENCE_OMISSION",
        "OCCURRENCE_RESTORE",
    }

    same_surface = adapted.candidates_for_surface("小猫追逐小鸟。")
    assert len(same_surface) == 2
    first_occurrences = {item.identity for item in same_surface[0].occurrences}
    second_occurrences = {item.identity for item in same_surface[1].occurrences}
    assert first_occurrences.isdisjoint(second_occurrences)
    assert same_surface[0].proposition_definition.bindings != (
        same_surface[1].proposition_definition.bindings)

    reversed_order = adapted.candidates_for_perturbation("ORDER_REVERSAL")[0]
    assert [item.start for item in reversed_order.occurrences] == [4, 2, 0]
    for candidate in adapted.candidates:
        semantic_objects = set(candidate.semantic_objects())
        occurrence_ids = {item.identity for item in candidate.occurrences}
        assert candidate.proposition_definition.source_anchor in occurrence_ids
        assert all(
            binding.filler in semantic_objects
            for binding in candidate.proposition_definition.bindings
        )
        assert all(
            item.semantic_object.object_kind in {OBJECT_ENTITY, OBJECT_EVENT}
            for item in candidate.occurrences
        )
        assert all(
            ObjectIdentity.from_stable_key(item.identity.stable_key())
            == item.identity
            for item in candidate.occurrences
        )
        assert candidate.selection_state == "UNSELECTED"
    assert dict(adapted.execution_state)["W05_STARTED"] == 0
    assert dict(adapted.execution_state)["formal_w05_training_runs"] == 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["occurrences"][0].__setitem__("end", 1),
            "surface_fragment",
        ),
        (
            lambda value: value["occurrence_order"].pop(),
            "occurrence_order",
        ),
        (
            lambda value: value["candidate_definition"].__setitem__(
                "source_anchor_key",
                value["candidate_definition"]["context_key"],
            ),
            "source anchor",
        ),
        (
            lambda value: value["candidate_definition"]["role_bindings"][0]
            .__setitem__(
                "filler_key",
                value["candidate_definition"]["predicate_key"],
            ),
            "Role filler",
        ),
    ],
)
def test_w05_adapter_fails_closed_on_span_order_anchor_and_filler(
        tmp_path, mutate, message):
    """坏 occurrence、order、anchor 或 filler 不得进入候选 owner。"""
    _context, payload = _context_and_payload(tmp_path)
    invalid = _replace_atomic_observation(payload, mutate)
    with pytest.raises(W05TypedAdapterError, match=message):
        adapt_w05_training_payload(invalid)


def test_w05_learning_uses_h05_h04_and_supersedes_omission(tmp_path):
    """四态 Evidence 真写 H-05/H-04，restore 真实退出 omission。"""
    _context, payload = _context_and_payload(tmp_path)
    adapted = adapt_w05_training_payload(payload)
    backend = SQLiteBackend(str(tmp_path / "w05-learning.sqlite"))
    try:
        learning = build_w05_learning_runtime(backend, adapted)
        report = learning.report()
        assert report.candidate_count == 6
        assert report.evidence_application_count == 6
        assert report.account_count == 8
        assert report.active_candidate_count == 2
        assert report.superseded_candidate_count == 1
        assert report.conflict_candidate_count == 1
        assert report.unknown_candidate_count == 1
        assert report.occurrence_count == 19
        assert report.role_binding_count == 11

        assert {item.perturbation_kind for item in learning.active_candidates()} == {
            "NONE", "OCCURRENCE_RESTORE"}
        superseded = learning.superseded_candidates()
        assert len(superseded) == 1
        assert superseded[0].perturbation_kind == "OCCURRENCE_OMISSION"
        hypothesis = learning.hypothesis_for(superseded[0].candidate)
        snapshot = learning.learning.engine.ledger.snapshot(hypothesis)
        assert snapshot.lifecycle == LIFECYCLE_SUPERSEDED

        accounts = [
            account
            for application in learning.applications()
            for account in application.accounts
        ]
        by_perturbation = {
            application.binding.perturbation_kind: application
            for application in learning.applications()
        }
        assert {item.stance for item in by_perturbation["NONE"].accounts} == {
            EVIDENCE_SUPPORT}
        assert {item.stance for item in by_perturbation["ROLE_SWAP"].accounts} == {
            EVIDENCE_REFUTE}
        assert {item.stance for item in by_perturbation["ORDER_REVERSAL"].accounts} == {
            EVIDENCE_REFUTE}
        assert {
            item.stance for item in by_perturbation["SCOPE_SHIFT"].accounts
        } == {EVIDENCE_SUPPORT, EVIDENCE_REFUTE}
        assert {
            item.stance for item in by_perturbation["OCCURRENCE_OMISSION"].accounts
        } == {EVIDENCE_UNKNOWN}
        assert sum(item.derived_supersede for item in accounts) == 1
        assert all(
            item.teacher_record.visible_from_stage == "W-05"
            for item in accounts
        )
    finally:
        backend.close()
