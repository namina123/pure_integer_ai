"""W05-00/W05-01 public contract、gate 和 train-only firewall 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_carrier_projection_runtime_contract import (
    CarrierProjectionRuntimeContractError,
)
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_ALLOWED_WORKER_COUNTS,
    W05_EVALUATION_ORDER,
    W05_FORMAL_RUN_ID,
    W05_OPEN_GENERATION_STATE,
    W05_RESOURCE_BUDGET,
    W05_RUNNER_KEY,
    W05_STAGE_KEY,
    W05_W04_BASE_RUN_ID,
    W05ContractError,
    W05RunRequest,
    digest_value,
    open_w05_frozen_context,
    validate_w05_request,
)
from pure_integer_ai.experiments.ph2_w05_firewall import W05PayloadFirewall
from pure_integer_ai.storage.backend import SQLiteBackend
from tests.w05_historical_context import open_historical_w05_context


ROOT = Path(__file__).resolve().parents[1]
HEAD = "693867db349e0ce05782fbaf6fa2b9206b26b4dc"


@pytest.fixture(scope="module")
def backend_key(tmp_path_factory):
    """用临时 SQLite profile 形成真实 backend capability key。"""
    path = tmp_path_factory.mktemp("w05-contract") / "probe.sqlite"
    backend = SQLiteBackend(str(path))
    try:
        return backend.storage_capabilities().stable_key()
    finally:
        backend.close()


@pytest.fixture(scope="module")
def context(backend_key):
    return open_historical_w05_context(
        ROOT,
        current_remote_commit_sha1=HEAD,
        backend_profile_key=backend_key,
    )


def test_current_authority_open_rejects_historical_parent_drift(backend_key):
    """生产 opener 必须拒绝历史 parent 漂移，行为 harness 不得改变它。"""
    with pytest.raises(CarrierProjectionRuntimeContractError, match="身份漂移"):
        open_w05_frozen_context(
            ROOT,
            current_remote_commit_sha1=HEAD,
            backend_profile_key=backend_key,
        )


def _request(context, *, mode="fresh", worker_count=1):
    return W05RunRequest(
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
        worker_count=worker_count,
        mode=mode,
        resource_budget=tuple(sorted(W05_RESOURCE_BUDGET.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )


def test_open_w05_context_binds_w04_lc16_stage_and_train_visibility(context):
    """W05-00 现场 context 只见 train owner/split，且不启动 W-05。"""
    assert context.stage_key == "W-05"
    assert context.stage_ordinal == 5
    assert context.pre_w04_gate_sha256 == (
        "c37bab6f02bd3adab2c546b5f79f070e3d232c481c70ef751727ea1edeff8c82"
    )
    assert context.stage_manifest_identity.sha256 == (
        "8dcc0d3d82c9bba9ebcb65533318f9977f7fd631c73e946e46ed4b83c448c453"
    )
    assert context.w04_receipt_identity.sha256 == (
        "153db3d7f3c0fca04642f4198df16e3c1adb0f5c78e4d6c7c59d35122989727b"
    )
    assert context.lc16_overlay_identity.sha256 == (
        "6cb9ab991ff41ecd87905f446ed5d75b2ad83e9d6f43124e2a69e15e7135083d"
    )
    assert len(context.train_pack_keys) == 11
    assert any("atomic-v1" in item.pack_key for item in context.pack_bindings)
    assert context.allowed_worker_counts == W05_ALLOWED_WORKER_COUNTS
    assert context.logical_shard_count == 16
    assert len(context.failure_point_keys) == 6
    assert context.evaluation_order == W05_EVALUATION_ORDER
    assert context.execution_state["W05_STARTED"] == 0
    assert context.execution_state["formal_w05_training_runs"] == 0
    assert context.execution_state["teacher_calls"] == 0
    assert context.open_generation_state == W05_OPEN_GENERATION_STATE
    assert context.payload_gets == 0
    assert context.payload_bytes == 0
    assert all(
        (item.owner_kind, item.split) in {("source", None), ("observation", "train")}
        for item in context.candidate_payload_bindings
    )
    assert all(
        (item.owner_kind, item.split) == ("teacher", "train")
        for item in context.teacher_evidence_bindings
    )
    assert all(
        (item.owner_kind, item.split) != ("evaluator", "held_out")
        for item in context.candidate_payload_bindings
    )


def test_w05_request_rejects_private_or_future_payload_before_transport(context):
    """request 中混入 held_out/evaluator/dev 路径时必须在 payload 前拒绝。"""
    valid = _request(context)
    assert validate_w05_request(context, valid) is valid
    private_path = next(
        item.relative_path for item in context.evaluator_visible_bindings
        if item.owner_kind == "evaluator"
    )
    invalid = replace(
        valid,
        candidate_payload_paths=valid.candidate_payload_paths + (private_path,),
    )
    with pytest.raises(W05ContractError, match="exact train whitelist"):
        validate_w05_request(context, invalid)
    drift = replace(valid, worker_count=3)
    with pytest.raises(W05ContractError, match="worker count"):
        validate_w05_request(context, drift)
    parent_drift = replace(valid, w04_receipt_key=(1,))
    with pytest.raises(W05ContractError, match="W-04 receipt"):
        validate_w05_request(context, parent_drift)


def test_w05_firewall_reads_train_once_and_never_labels(context):
    """firewall 成功交付后只包含 source、train Observation 与 train Evidence。"""
    request = _request(context)
    firewall = W05PayloadFirewall.open(ROOT, context, request)
    payload = firewall.read_training_payload()
    assert payload.source_refs
    assert payload.observations
    assert payload.teacher_evidence
    assert all(item.split == "train" for item in payload.observations)
    assert all(item.visible_from_stage in {"W-02", "W-03", "W-04", "W-05"}
               for item in payload.teacher_evidence)
    assert firewall.audit.payload_gets == (
        len(context.candidate_payload_bindings)
        + len(context.teacher_evidence_bindings)
    )
    assert firewall.audit.teacher_calls == 0
    assert firewall.audit.learning_writes == 0
    with pytest.raises(W05ContractError, match="cannot be replayed"):
        firewall.read_training_payload()
