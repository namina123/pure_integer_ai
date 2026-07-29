"""W-02 v2 正式 runtime 的 Evidence 归因、dump 和事务恢复测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w02_contract import (
    D03_GLOBAL_MANIFEST_PATH,
    W02_OWNER_KEY,
    W02_RUNNER_KEY,
    W02PayloadFirewall,
    W02RunRequest,
    open_w02_frozen_context,
)
from pure_integer_ai.experiments.ph2_w02_faults import (
    W02FaultPoint,
    W02InjectedFault,
)
from pure_integer_ai.experiments.ph2_w02_learning_v2 import (
    open_w02_learning_runtime_v2,
)
from pure_integer_ai.experiments.ph2_w02_runtime import W02RuntimeConfig
from pure_integer_ai.experiments.ph2_w02_runtime_v2 import (
    load_w02_candidate_dump_v2,
    run_language_stage1_v2,
    train_generation_probe_target_v2,
)
from pure_integer_ai.experiments.ph2_w02_use import DIRECTION_GENERATION
from pure_integer_ai.storage.backend import SQLiteBackend


_REPOSITORY = Path(__file__).resolve().parents[1]
_REMOTE_COMMIT = "5d00b703ada7d41bfa96e466a06af61026da3a64"


def _config(
        root: Path,
        *,
        mode: str = "fresh",
        worker_count: int = 1,
        fault_point: str | None = None,
        ) -> W02RuntimeConfig:
    """构造绑定公开 W-02 v2 切片身份的临时正式 host 配置。"""
    return W02RuntimeConfig(
        repository_root=_REPOSITORY,
        global_manifest_path=D03_GLOBAL_MANIFEST_PATH,
        run_root=root / "runs",
        sqlite_path=root / "w02-v2.sqlite3",
        run_id=203,
        parent_run_id=1,
        base_run_id=1,
        base_fence_key=(2, 5, 20260729),
        worker_count=worker_count,
        mode=mode,
        current_remote_commit_sha1=_REMOTE_COMMIT,
        fault_point=fault_point,
    )


def _training_payload():
    """通过正式 firewall 回读 public train payload，用于核 generation request key。"""
    context = open_w02_frozen_context(
        _REPOSITORY,
        D03_GLOBAL_MANIFEST_PATH,
        current_remote_commit_sha1=_REMOTE_COMMIT,
    )
    request = W02RunRequest(
        203, 1, 1, context.stage_key, W02_OWNER_KEY, W02_RUNNER_KEY,
        context.current_remote_commit_sha1, context.stable_key(),
        context.w01_receipt_sha256, (1, 20260729), (2, 5, 20260729),
        1, "fresh",
        tuple(item.relative_path
              for item in context.candidate_payload_bindings),
        tuple(item.relative_path
              for item in context.teacher_evidence_bindings),
    )
    return W02PayloadFirewall.open(
        _REPOSITORY, context, request).read_training_payload()


def _logical_projection(outcome):
    """只取恢复方式和物理介质不得改变的 v2 规范结果。"""
    return (
        outcome.logical_state_digest,
        outcome.core_digest,
        outcome.memory_digest,
        outcome.use_digest,
        outcome.cursor_digest,
        outcome.artifact_digest,
        outcome.dump_manifest_sha256,
    )


def test_v2_formal_runtime_records_evidence_target_and_dump_readback(tmp_path):
    """正式 generation Use 绑定完整 Evidence target，fresh dump 恢复同一状态。"""
    root = tmp_path / "candidate"
    config = _config(root)
    outcome = run_language_stage1_v2(config)
    assert outcome.execution_state["W02_STARTED"] == 1
    assert outcome.execution_state["W03_STARTED"] == 0
    assert outcome.execution_state["LANGUAGE_CAPABILITY_MASTERED"] == 0
    assert outcome.attribution_report.use_count_by_direction == (
        ("GENERATION", 1), ("UNDERSTANDING", 1))

    payload = _training_payload()
    target = train_generation_probe_target_v2(payload)
    backend = SQLiteBackend(str(config.sqlite_path))
    try:
        learning = open_w02_learning_runtime_v2(backend, mode="resume")
        generation = tuple(
            item for item in learning.use_outcomes.attributions()
            if item.direction == DIRECTION_GENERATION)
        assert len(generation) == 1
        assert generation[0].request_key == target.stable_key()
        assert learning.word_forms.lookup(
            target.stem_surface, branch=learning.branch) is None
        assert learning.generate(target).surfaces == ("清晰化",)
    finally:
        backend.close()

    readback = load_w02_candidate_dump_v2(
        config,
        target_sqlite_path=tmp_path / "readback.sqlite3",
    )
    assert _logical_projection(readback) == _logical_projection(outcome)
    assert readback.dump_readback is True


def test_v2_consumer_fault_restart_and_resume_match_independent_fresh(tmp_path):
    """consumer 后、commit 前中断不得留下部分 v2 Use 或重复 adoption。"""
    baseline = run_language_stage1_v2(_config(tmp_path / "baseline"))
    root = tmp_path / "fault"
    with pytest.raises(W02InjectedFault, match="AFTER_MERGE_BEFORE_COMMIT"):
        run_language_stage1_v2(_config(
            root,
            fault_point=W02FaultPoint.AFTER_MERGE_BEFORE_COMMIT,
        ))
    recovered = run_language_stage1_v2(_config(
        root,
        mode="restart",
        worker_count=4,
    ))
    replay = run_language_stage1_v2(_config(
        root,
        mode="resume",
        worker_count=2,
    ))
    assert _logical_projection(recovered) == _logical_projection(baseline)
    assert _logical_projection(replay) == _logical_projection(baseline)
    assert recovered.transaction_event_count == replay.transaction_event_count == 4
    assert recovered.attribution_report.outcome_count == 2
    assert replay.learning_report.replayed is True
