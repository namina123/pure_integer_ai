"""W06-01 public contract、有效 pack 替换与 train-only firewall 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w06_contract import (
    W06_ALLOWED_WORKER_COUNTS,
    W06_ALLOWED_WRITE_OWNERS,
    W06_EVALUATION_ORDER,
    W06_FAILURE_POINT_KEYS,
    W06_FORBIDDEN_WRITE_OWNERS,
    W06_FORMAL_RUN_ID,
    W06_GENERATION_ABLATION_KEY,
    W06_LOGICAL_CLOCK_VERSION,
    W06_OPEN_GENERATION_STATE,
    W06_PRIVATE_ABLATION_KEYS,
    W06_RESOURCE_BUDGET,
    W06_RUNNER_KEY,
    W06_STAGE_KEY,
    W06_V1_PACK_KEY,
    W06_V2_PACK_KEY,
    W06_W05_BASE_RUN_ID,
    W06ContractError,
    W06PayloadAudit,
    W06RunRequest,
    open_w06_frozen_context,
    validate_w06_request,
)
from pure_integer_ai.experiments.ph2_w06_firewall import W06PayloadFirewall
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_GENERATION_HARD_CONJUNCT,
    W06_RELATION_SUBSTAGE_ORDER,
)
from pure_integer_ai.storage.backend import SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]
HEAD = "4d57305bc4474081c9304a05287ab4783f49a849"
OVERLAY_SHA256 = "f5cae297254191dffb5bcacdafbdc461dcd1cf3a1340de27d9a8c98c598bfbbc"


@pytest.fixture(scope="module")
def backend_key(tmp_path_factory):
    """以临时 SQLite 实例形成真实 backend capability 身份。"""
    path = tmp_path_factory.mktemp("w06-contract") / "probe.sqlite"
    backend = SQLiteBackend(str(path))
    try:
        return backend.storage_capabilities().stable_key()
    finally:
        backend.close()


@pytest.fixture(scope="module")
def context(backend_key):
    """只读打开当前 public W-06 parent 与 18 个有效 train pack。"""
    return open_w06_frozen_context(
        ROOT,
        current_remote_commit_sha1=HEAD,
        backend_profile_key=backend_key,
    )


def _request(context, *, mode="fresh", worker_count=1):
    """构造不含 evaluator/private 字段的精确 W-06 candidate 请求。"""
    return W06RunRequest(
        run_id=W06_FORMAL_RUN_ID,
        parent_run_id=W06_W05_BASE_RUN_ID,
        base_run_id=W06_W05_BASE_RUN_ID,
        stage_key=W06_STAGE_KEY,
        owner_key=context.owner_key,
        runner_key=W06_RUNNER_KEY,
        current_remote_commit_sha1=context.current_remote_commit_sha1,
        source_overlay_sha256=context.source_overlay_sha256,
        context_key=context.stable_key(),
        backend_profile_key=context.backend_profile_key,
        base_fence_key=context.base_fence_key,
        worker_count=worker_count,
        mode=mode,
        resource_budget=tuple(sorted(W06_RESOURCE_BUDGET.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )


def test_w06_context_binds_v2_overlay_relations_recovery_and_zero_state(context):
    """context 保留 v1 父坐标但只向候选暴露 stable v2 与冻结执行合同。"""
    assert context.current_remote_commit_sha1 == HEAD
    assert context.source_overlay_sha256 == OVERLAY_SHA256
    assert len(context.stage_train_pack_keys) == 18
    assert len(context.effective_train_pack_keys) == 18
    assert context.stage_train_pack_keys[0] == W06_V1_PACK_KEY
    assert context.effective_train_pack_keys[0] == W06_V2_PACK_KEY
    assert sum(item.overlay_replacement for item in context.pack_bindings) == 1
    assert len(context.candidate_payload_bindings) == 36
    assert len(context.teacher_evidence_bindings) == 18
    assert len(context.evaluator_visible_bindings) == 38
    assert all(
        item.pack_key != W06_V1_PACK_KEY
        and "alias-refers-v1" not in item.relative_path
        for item in (
            *context.candidate_payload_bindings,
            *context.teacher_evidence_bindings,
            *context.evaluator_visible_bindings,
        )
    )
    assert context.relation_substage_order == W06_RELATION_SUBSTAGE_ORDER
    assert len(context.dimension_keys) == len(context.ablation_keys) == 7
    assert context.private_ablation_keys == W06_PRIVATE_ABLATION_KEYS
    assert context.private_ablation_keys[-1] == W06_GENERATION_ABLATION_KEY
    assert context.generation_hard_conjunct == W06_GENERATION_HARD_CONJUNCT
    assert context.evaluation_order == W06_EVALUATION_ORDER
    assert context.allowed_worker_counts == W06_ALLOWED_WORKER_COUNTS
    assert context.failure_point_keys == W06_FAILURE_POINT_KEYS
    assert context.logical_shard_count == 16
    assert context.logical_clock_version == W06_LOGICAL_CLOCK_VERSION
    assert dict(context.resource_budget) == W06_RESOURCE_BUDGET
    assert context.allowed_write_owners == W06_ALLOWED_WRITE_OWNERS
    assert context.forbidden_write_owners == W06_FORBIDDEN_WRITE_OWNERS
    assert dict(context.execution_state) == {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W06_STARTED": 0,
        "W07_STARTED": 0,
        "formal_w06_training_runs": 0,
        "teacher_calls": 0,
    }
    assert context.open_generation_state == W06_OPEN_GENERATION_STATE
    assert context.payload_gets == context.payload_bytes == context.learning_writes == 0


def test_w06_request_rejects_drift_before_transport(context):
    """worker、路径、overlay、base 和资源漂移必须在首次文件读取前失败。"""
    valid = _request(context)
    assert validate_w06_request(context, valid) is valid
    assert valid.execution_identity_key() == replace(
        valid, worker_count=4, mode="resume").execution_identity_key()
    assert valid.scheduling_key() != replace(
        valid, worker_count=4, mode="resume").scheduling_key()

    private_path = context.evaluator_visible_bindings[0].relative_path
    invalid_requests = (
        (replace(valid, worker_count=3), "worker count"),
        (replace(
            valid,
            candidate_payload_paths=valid.candidate_payload_paths + (private_path,),
        ), "train whitelist"),
        (replace(valid, source_overlay_sha256="0" * 64), "source overlay"),
        (replace(valid, base_fence_key=(1,)), "context/backend/base fence"),
        (replace(
            valid,
            resource_budget=valid.resource_budget + (("max_workers", 4),),
        ), "resource budget"),
    )
    for invalid, message in invalid_requests:
        audit = W06PayloadAudit()
        with pytest.raises(W06ContractError, match=message):
            W06PayloadFirewall.open(ROOT, context, invalid, audit=audit)
        assert audit.transport_attempts == 0
        assert audit.transport_bytes == 0


def test_w06_firewall_reads_effective_train_payload_once_without_labels(context):
    """firewall 只交付有效 18-pack 的 train 记录，并保持调用和学习写为零。"""
    firewall = W06PayloadFirewall.open(ROOT, context, _request(context))
    payload = firewall.read_training_payload()
    assert len(payload.source_refs) == 259
    assert len(payload.observations) == 141
    assert len(payload.teacher_evidence) == 141
    assert all(item.split == "train" for item in payload.observations)
    assert all(
        item.visible_from_stage in {"W-02", "W-03", "W-04", "W-05", "W-06"}
        for item in payload.teacher_evidence
    )
    binding_count = (
        len(context.candidate_payload_bindings)
        + len(context.teacher_evidence_bindings)
    )
    assert firewall.audit.transport_attempts == binding_count == 54
    assert firewall.audit.payload_gets == binding_count
    assert firewall.audit.payload_bytes == firewall.audit.transport_bytes
    assert firewall.audit.teacher_calls == 0
    assert firewall.audit.learning_writes == 0
    with pytest.raises(W06ContractError, match="禁止重放"):
        firewall.read_training_payload()
