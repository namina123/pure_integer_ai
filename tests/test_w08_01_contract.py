"""W08-01 public contract、registry inventory 与可见性防火墙专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w08_authority import (
    W08_ABLATION_KEYS,
    W08_DIMENSION_KEYS,
    W08_FUTURE_PACK_KEYS,
    W08_SUBTASK_ORDER,
)
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_ALLOWED_WORKER_COUNTS,
    W08_CARRIER_KEYS,
    W08_CONSUMER_KEYS,
    W08_FAILURE_POINT_KEYS,
    W08_LEARNING_PACK_KEYS,
    W08_RESOURCE_BUDGET,
    W08_STOP_STATES,
    W08ContractError,
    W08PayloadAudit,
    make_w08_request,
    open_w08_frozen_contract,
    validate_w08_request,
)
from pure_integer_ai.experiments.ph2_w08_firewall import (
    W08PayloadFirewall,
    W08VisibilityFirewall,
    _payload_file,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def context():
    return open_w08_frozen_contract(ROOT)


def test_w08_contract_binds_authority_five_schemas_resources_and_zero_state(context):
    assert context.authority_sha256 == (
        "1236c34b3076bee29d16508361d8405f80b0382a1403bc54acac0b2c4a15688a"
    )
    assert context.baseline_public_head_commit_sha1 == (
        "bbf610b7e05c66f5d2930cdeb3d66bc26e822010"
    )
    assert context.subtask_order == W08_SUBTASK_ORDER
    assert context.dimension_keys == W08_DIMENSION_KEYS
    assert context.ablation_keys == W08_ABLATION_KEYS
    assert context.consumer_keys == W08_CONSUMER_KEYS
    assert context.stop_states == W08_STOP_STATES
    assert context.carrier_keys == W08_CARRIER_KEYS
    assert context.discourse_state_components == (
        "APPEND_ONLY_EVENT_LOG",
        "CURRENT_PROJECTION",
        "DEPENDENCY_INDEX",
    )
    assert context.allowed_worker_counts == W08_ALLOWED_WORKER_COUNTS
    assert context.failure_point_keys == W08_FAILURE_POINT_KEYS
    assert context.logical_shard_count == 16
    assert dict(context.resource_budget) == W08_RESOURCE_BUDGET
    assert context.future_pack_keys == W08_FUTURE_PACK_KEYS
    assert context.learning_pack_keys == W08_LEARNING_PACK_KEYS
    assert len(context.stage_train_pack_keys) == 30
    assert len(context.stage_dev_pack_keys) == 1
    assert len(context.stage_held_out_pack_keys) == 33
    assert len(context.stage_evaluator_pack_keys) == 33
    assert len(context.candidate_bindings) == 12
    assert len(context.teacher_bindings) == 6
    assert len(context.evaluator_bindings) == 66
    assert len(context.forbidden_bindings) == 8
    assert len(context.future_forbidden_paths) == 20
    assert dict(context.execution_state)["W08_STARTED"] == 0
    assert dict(context.execution_state)["formal_w08_training_runs"] == 0
    assert len(context.stable_key()) == len(context.base_fence_key) == 32


@pytest.mark.parametrize(
    "mutator",
    [
        lambda request, context: replace(request, worker_count=3),
        lambda request, context: replace(request, mode="formal"),
        lambda request, context: replace(request, owner_key="PH2_W07_TRANSACTION_OWNER"),
        lambda request, context: replace(request, base_fence_key=(1,)),
        lambda request, context: replace(
            request,
            resource_budget=request.resource_budget + (("max_workers", 4),),
        ),
        lambda request, context: replace(
            request,
            candidate_payload_paths=request.candidate_payload_paths + (
                context.evaluator_bindings[0].relative_path,
            ),
        ),
        lambda request, context: replace(
            request,
            candidate_payload_paths=(
                "archive.zip!observations/train.jsonl.gz",
                *request.candidate_payload_paths[1:],
            ),
        ),
        lambda request, context: replace(
            request,
            forbidden_payload_paths=(context.future_forbidden_paths[0],),
        ),
    ],
)
def test_w08_request_attacks_fail_before_transport(context, mutator):
    request = mutator(make_w08_request(context), context)
    audit = W08PayloadAudit()
    with pytest.raises(W08ContractError):
        W08PayloadFirewall.open(ROOT, context, request, audit=audit)
    assert audit.transport_attempts == audit.transport_bytes == 0
    assert audit.future_payload_reads == audit.evaluator_label_reads == 0


def test_w08_firewall_delivers_six_train_packs_once_without_private_reads(context):
    firewall = W08PayloadFirewall.open(ROOT, context, make_w08_request(context))
    payload = firewall.read_training_payload()
    assert len(payload.source_refs) == 120
    assert len(payload.observations) == len(payload.teacher_evidence) == 63
    assert all(item.split == "train" for item in payload.observations)
    assert firewall.audit.transport_attempts == firewall.audit.payload_gets == 18
    assert firewall.audit.payload_bytes == firewall.audit.transport_bytes
    assert firewall.audit.evaluator_label_reads == 0
    assert firewall.audit.held_out_reads == 0
    assert firewall.audit.future_payload_reads == 0
    assert firewall.audit.teacher_calls == 0
    assert firewall.audit.learning_writes == 0
    assert firewall.audit.memory_learning_writes == 0
    with pytest.raises(W08ContractError, match="replay"):
        firewall.read_training_payload()


def test_w08_visibility_is_phase_separated_and_future_fails_closed(context):
    audit = W08PayloadAudit()
    firewall = W08VisibilityFirewall(context, audit)
    candidate_path = context.candidate_bindings[0].relative_path
    evaluator_path = context.evaluator_bindings[0].relative_path
    future_path = context.future_forbidden_paths[0]
    assert firewall.authorize_candidate(candidate_path).access_phase == "candidate"
    with pytest.raises(W08ContractError):
        firewall.authorize_candidate(evaluator_path)
    with pytest.raises(W08ContractError, match="sealed"):
        firewall.authorize_evaluator(evaluator_path, candidate_sealed=0)
    assert firewall.authorize_evaluator(
        evaluator_path, candidate_sealed=1
    ).access_phase == "evaluator"
    with pytest.raises(W08ContractError, match="future"):
        firewall.authorize_candidate(future_path)
    with pytest.raises(W08ContractError, match="future"):
        firewall.authorize_evaluator(future_path, candidate_sealed=1)
    assert audit.transport_attempts == audit.future_payload_reads == 0


def test_w08_schedule_changes_do_not_change_execution_identity(context):
    request = make_w08_request(context)
    rescheduled = replace(request, worker_count=4, mode="resume")
    assert validate_w08_request(context, rescheduled) is rescheduled
    assert request.execution_identity_key() == rescheduled.execution_identity_key()
    assert request.scheduling_key() != rescheduled.scheduling_key()


@pytest.mark.parametrize("link_method", ["is_symlink", "is_junction"])
def test_w08_payload_rejects_link_components_before_read(
    tmp_path: Path, monkeypatch, link_method: str
):
    root = tmp_path.resolve()
    linked = root / "linked"
    linked.mkdir()
    (linked / "payload.jsonl.gz").write_bytes(b"not-read")
    original = getattr(Path, link_method)

    def reports_link(path: Path) -> bool:
        return path.name == "linked" or original(path)

    monkeypatch.setattr(Path, link_method, reports_link)
    with pytest.raises(W08ContractError, match="link"):
        _payload_file(root, "linked/payload.jsonl.gz")


def test_w08_binding_owner_cannot_impersonate_evaluator(context):
    with pytest.raises(W08ContractError, match="owner/split"):
        replace(context.candidate_bindings[0], access_phase="evaluator")
