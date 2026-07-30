"""PH2 W-03 新 owner 的 shard、事务、恢复、dump 与 retention。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
)
from pure_integer_ai.experiments.ph2_w03_continuity import (
    formal_w03_publication_baseline,
    verify_formal_w02_continuity,
)
from pure_integer_ai.experiments.ph2_w03_faults import (
    W03FaultPoint,
    W03InjectedFault,
)
from pure_integer_ai.experiments.ph2_w03_runtime import (
    W03RuntimeConfig,
    load_w03_candidate_dump,
    run_language_stage2,
)


REPOSITORY = Path(__file__).resolve().parents[1]
W02_ARTIFACTS = REPOSITORY.parent / "w02_artifacts"


def _w02_identity() -> tuple[tuple[str, int, str], ...]:
    """固定 W-02 public receipt/candidate 物理树，不打开 private family。"""
    continuity = verify_formal_w02_continuity(REPOSITORY, W02_ARTIFACTS)
    return tuple(
        (item.relative_path, item.size_bytes, item.sha256)
        for item in (
            continuity.receipt_identity,
            continuity.candidate_freeze_identity,
            continuity.candidate_attestation_identity,
            continuity.aggregate_identity,
            *continuity.capability_code_identities,
            *continuity.host_artifact_identities,
        )
    )


def _config(root: Path, **changes) -> W03RuntimeConfig:
    """构造 test-local run 4；root 永远与 W-02 物理根分离。"""
    baseline = formal_w03_publication_baseline()
    continuity = verify_formal_w02_continuity(REPOSITORY, W02_ARTIFACTS)
    config = W03RuntimeConfig(
        repository_root=REPOSITORY,
        global_manifest_path=FORMAL_GLOBAL_MANIFEST_PATH,
        w02_artifacts_root=W02_ARTIFACTS,
        run_root=root / "runs",
        sqlite_path=root / "candidate.sqlite3",
        run_id=4,
        parent_run_id=3,
        base_run_id=3,
        base_fence_key=continuity.base_fence_key(),
        worker_count=1,
        mode="fresh",
        current_remote_commit_sha1=baseline.head_sha1,
    )
    return replace(config, **changes)


def _logical(outcome):
    """排除调度计数与 readback 标志的 worker/mode 无关投影。"""
    return (
        outcome.logical_state_digest,
        outcome.candidate_history_digest,
        outcome.projection_digest,
        outcome.generation_digest,
        outcome.cursor_digest,
        outcome.artifact_digest,
        outcome.dump_manifest_sha256,
        outcome.retention_digest,
        outcome.artifact_counts,
        outcome.execution_state,
    )


def test_one_two_four_workers_are_bit_identical_and_resource_bounded(tmp_path):
    """16 shard 的三种物理并发形成相同 host、cursor、dump 与可见结果。"""
    outcomes = tuple(
        run_language_stage2(_config(
            tmp_path / f"w{workers}", worker_count=workers))
        for workers in (1, 2, 4)
    )

    assert _logical(outcomes[0]) == _logical(outcomes[1]) == _logical(outcomes[2])
    assert {item.resource_report["requested_workers"] for item in outcomes} == {
        1, 2, 4,
    }
    assert all(item.resource_report["logical_shards"] == 16 for item in outcomes)
    assert all(item.resource_report["merged_records"] == 163 for item in outcomes)
    assert all(item.resource_report["actual_payload_gets"] == 18
               for item in outcomes)
    assert all(item.resource_report["actual_payload_bytes"] == 134763
               for item in outcomes)
    assert all(
        item.resource_report[f"actual_{key[4:]}"] <= value
        for item in outcomes
        for key, value in item.resource_budget.items()
    )
    assert all(item.execution_state == {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W03_STARTED": 0,
        "W04_STARTED": 0,
        "formal_w03_training_runs": 0,
        "teacher_calls": 0,
    } for item in outcomes)


@pytest.mark.parametrize("point", W03FaultPoint.injectable_points())
def test_all_six_faults_recover_without_duplicate_commit_or_publish(
        tmp_path, point):
    """六点故障后只允许精确恢复，事务、barrier、Use/outcome 各发布一次。"""
    baseline = run_language_stage2(_config(tmp_path / "baseline"))
    root = tmp_path / point.lower()
    with pytest.raises(W03InjectedFault, match=point):
        run_language_stage2(_config(root, fault_point=point))
    mode = (
        "resume" if point == W03FaultPoint.AFTER_MANIFEST_PUBLISH
        else "restart"
    )
    recovered = run_language_stage2(_config(
        root, mode=mode, worker_count=4))
    replay = run_language_stage2(_config(
        root, mode="resume", worker_count=2))

    assert _logical(recovered) == _logical(replay) == _logical(baseline)
    assert recovered.transaction_event_count == replay.transaction_event_count == 4
    assert recovered.merge_publication_count == replay.merge_publication_count == 1
    assert recovered.adopted_manifest_count == replay.adopted_manifest_count == 1
    assert recovered.artifact_counts == replay.artifact_counts
    assert replay.new_learning_write_count == 0


def test_dump_fresh_connection_restores_history_projection_generation_and_cursor(
        tmp_path):
    """fresh SQLite 从 package 回读 envelope、H-00/H-04、projection、Use/outcome。"""
    root = tmp_path / "run"
    outcome = run_language_stage2(_config(root))
    readback = load_w03_candidate_dump(
        _config(root, mode="resume"),
        target_sqlite_path=tmp_path / "readback.sqlite3",
    )

    assert _logical(readback) == _logical(outcome)
    assert readback.dump_readback is True
    assert readback.new_learning_write_count == 0
    assert dict(readback.artifact_counts) == {
        "EVIDENCE_ACCOUNT": 64,
        "GENERATION_CHOICE": 2,
        "GENERATION_DECISION": 3,
        "GENERATION_OUTCOME": 4,
        "GENERATION_USE": 3,
        "PROJECTION": 59,
        "TRAIN_ENVELOPE": 163,
        "W02_RETENTION": 1,
    }


def test_w02_base_and_host_remain_read_only_and_physically_isolated(tmp_path):
    """run 4 只绑定 W-02 fence，不能复用其 root、表 owner 或覆写 artifact。"""
    before = _w02_identity()
    root = tmp_path / "w03"
    outcome = run_language_stage2(_config(root))
    after = _w02_identity()

    assert after == before
    assert outcome.w02_host_write_count == 0
    assert outcome.w02_retention_passed is True
    assert Path(outcome.sqlite_path).is_relative_to(root.resolve())
    assert not Path(outcome.sqlite_path).is_relative_to(W02_ARTIFACTS.resolve())
    assert all(not name.startswith("ph2_w02_") for name in outcome.owned_tables)

    with pytest.raises(RuntimeError, match="W-02|隔离|root"):
        run_language_stage2(_config(
            tmp_path / "invalid",
            sqlite_path=(W02_ARTIFACTS / "formal_candidate_v2" /
                         "candidate.sqlite3"),
        ))


def test_identity_drift_fresh_overwrite_and_duplicate_publish_fail_closed(tmp_path):
    """同 run fence 漂移、fresh 覆盖和二次 manifest 发布均不可降级为重放。"""
    root = tmp_path / "run"
    run_language_stage2(_config(root))
    with pytest.raises(RuntimeError, match="fresh"):
        run_language_stage2(_config(root))
    with pytest.raises(RuntimeError, match="identity|fence"):
        run_language_stage2(_config(
            root,
            mode="resume",
            base_fence_key=(9, 9, 20260730),
        ))
    manifest = root / "runs" / "4" / "run.manifest.json"
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    replay = run_language_stage2(_config(root, mode="resume"))
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before
    assert replay.adopted_manifest_count == 1
