"""W07-01 public parent、运行合同与 train-only firewall 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_ALLOWED_WORKER_COUNTS,
    W07_BASELINE_COMMIT_SHA1,
    W07_FAILURE_POINT_KEYS,
    W07_FORMAL_RUN_ID,
    W07_FORMING_PACK_KEYS,
    W07_GENERATION_ABLATION_KEY,
    W07_GENERATION_HARD_CONJUNCT,
    W07_HISTORICAL_ABLATION_KEYS,
    W07_HISTORICAL_DIMENSION_KEYS,
    W07_OPEN_GENERATION_STATE,
    W07_PUBLIC_ABLATION_KEYS,
    W07_PUBLIC_DIMENSION_KEYS,
    W07_RESOURCE_BUDGET,
    W07_RUNNER_KEY,
    W07_STAGE_KEY,
    W07_SUBSTAGE_ORDER,
    W07_W06_BASE_RUN_ID,
    W07ContractError,
    W07PayloadAudit,
    W07RunRequest,
    open_w07_frozen_context,
    validate_w07_request,
)
from pure_integer_ai.experiments.ph2_w07_firewall import W07PayloadFirewall
from pure_integer_ai.storage.backend import SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def backend_key(tmp_path_factory):
    """以临时 SQLite 实例形成真实 backend capability 身份。"""
    path = tmp_path_factory.mktemp("w07-contract") / "probe.sqlite"
    backend = SQLiteBackend(str(path))
    try:
        return backend.storage_capabilities().stable_key()
    finally:
        backend.close()


@pytest.fixture(scope="module")
def context(backend_key):
    """只读打开当前 public W-07 parent 与七个 forming pack。"""
    return open_w07_frozen_context(
        ROOT,
        baseline_commit_sha1=W07_BASELINE_COMMIT_SHA1,
        backend_profile_key=backend_key,
    )


def _request(context, *, mode="fresh", worker_count=1):
    """构造不含 evaluator/future 字段的精确 candidate 请求。"""
    return W07RunRequest(
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
        worker_count,
        mode,
        tuple(sorted(W07_RESOURCE_BUDGET.items())),
        tuple(item.relative_path for item in context.candidate_payload_bindings),
        tuple(item.relative_path for item in context.teacher_evidence_bindings),
    )


def test_w07_context_binds_public_parents_orders_generation_and_zero_state(context):
    """训练子序、历史评测序与追加 W07-G 独立冻结且零执行。"""
    assert context.baseline_commit_sha1 == W07_BASELINE_COMMIT_SHA1
    assert context.substage_order == W07_SUBSTAGE_ORDER
    assert context.historical_dimension_keys == W07_HISTORICAL_DIMENSION_KEYS
    assert context.historical_ablation_keys == W07_HISTORICAL_ABLATION_KEYS
    assert context.public_dimension_keys == W07_PUBLIC_DIMENSION_KEYS
    assert context.public_ablation_keys == W07_PUBLIC_ABLATION_KEYS
    assert context.generation_hard_conjunct == W07_GENERATION_HARD_CONJUNCT
    assert context.public_dimension_keys[-1] == W07_GENERATION_HARD_CONJUNCT
    assert context.public_ablation_keys[-1] == W07_GENERATION_ABLATION_KEY
    assert context.substage_order != tuple(
        item.removeprefix("W-07-") for item in context.historical_dimension_keys)
    assert len(context.parent_sha256) == 9
    assert len(context.stage_train_pack_keys) == 25
    assert context.forming_pack_keys == W07_FORMING_PACK_KEYS
    assert len(context.nonforming_train_pack_keys) == 18
    assert len(context.pack_bindings) == 7
    assert len(context.candidate_payload_bindings) == 14
    assert len(context.teacher_evidence_bindings) == 7
    assert len(context.forbidden_payload_bindings) == 14
    assert context.allowed_worker_counts == W07_ALLOWED_WORKER_COUNTS
    assert context.failure_point_keys == W07_FAILURE_POINT_KEYS
    assert context.logical_shard_count == 16
    assert dict(context.resource_budget) == W07_RESOURCE_BUDGET
    assert (context.run_id, context.parent_run_id, context.base_run_id) == (8, 7, 7)
    assert dict(context.execution_state) == {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W06_RUNTIME_EVIDENCED": 1,
        "W06_STARTED": 1,
        "W07_STARTED": 0,
        "W08_STARTED": 0,
        "formal_w07_training_runs": 0,
        "teacher_calls": 0,
    }
    assert context.open_generation_state == W07_OPEN_GENERATION_STATE
    assert context.payload_gets == context.payload_bytes == context.learning_writes == 0


def test_w07_request_rejects_non_train_and_resource_drift_before_transport(context):
    """forbidden path、worker、base 与资源漂移必须在 transport 前失败。"""
    valid = _request(context)
    assert validate_w07_request(context, valid) is valid
    assert valid.execution_identity_key() == replace(
        valid, worker_count=4, mode="resume").execution_identity_key()
    assert valid.scheduling_key() != replace(
        valid, worker_count=4, mode="resume").scheduling_key()
    forbidden_path = context.forbidden_payload_bindings[0].relative_path
    invalid_requests = (
        (replace(valid, worker_count=3), "worker count"),
        (replace(
            valid,
            candidate_payload_paths=valid.candidate_payload_paths + (
                forbidden_path,),
        ), "train whitelist"),
        (replace(valid, base_run_id=6), "run/parent/base"),
        (replace(valid, baseline_commit_sha1="0" * 40), "baseline commit"),
        (replace(
            valid,
            resource_budget=valid.resource_budget + (("max_workers", 4),),
        ), "resource budget"),
    )
    for invalid, message in invalid_requests:
        audit = W07PayloadAudit()
        with pytest.raises(W07ContractError, match=message):
            W07PayloadFirewall.open(ROOT, context, invalid, audit=audit)
        assert audit.transport_attempts == audit.transport_bytes == 0


def test_w07_firewall_reads_seven_train_packs_once_without_labels(context):
    """firewall 只交付七 pack 的 train 记录并拒绝同实例 replay。"""
    firewall = W07PayloadFirewall.open(ROOT, context, _request(context))
    payload = firewall.read_training_payload()
    assert len(payload.source_refs) == 90
    assert len(payload.observations) == 66
    assert len(payload.teacher_evidence) == 66
    assert all(item.w_stage == "W-07" and item.split == "train"
               for item in payload.observations)
    assert all(item.visible_from_stage == "W-07"
               for item in payload.teacher_evidence)
    binding_count = (
        len(context.candidate_payload_bindings)
        + len(context.teacher_evidence_bindings))
    assert firewall.audit.transport_attempts == binding_count == 21
    assert firewall.audit.payload_gets == binding_count
    assert firewall.audit.payload_bytes == firewall.audit.transport_bytes
    assert firewall.audit.teacher_calls == firewall.audit.learning_writes == 0
    with pytest.raises(W07ContractError, match="禁止重放"):
        firewall.read_training_payload()
