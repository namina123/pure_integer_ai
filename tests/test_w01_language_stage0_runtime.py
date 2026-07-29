"""W-01 事务、稳定 shard/merge、故障恢复和 SQLite 重启反例。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w01_contract import D03_GLOBAL_MANIFEST_PATH
from pure_integer_ai.experiments.ph2_w01_runtime import (
    W01FaultPoint,
    W01InjectedFault,
    W01RuntimeConfig,
    read_w01_run,
    run_language_stage0,
)


_REPOSITORY = Path(__file__).resolve().parents[1]


def _config(
        root: Path,
        *,
        worker_count: int = 1,
        mode: str = "fresh",
        fault_point: str | None = None,
        base_fence_key: tuple[int, ...] = (1, 0, 20260729),
        ) -> W01RuntimeConfig:
    """构造只绑定正式 D-03 和独立 SQLite 文件的阶段 0 配置。"""
    return W01RuntimeConfig(
        repository_root=_REPOSITORY,
        global_manifest_path=D03_GLOBAL_MANIFEST_PATH,
        run_root=root / "runs",
        sqlite_path=root / "w01.sqlite3",
        run_id=101,
        parent_run_id=0,
        base_run_id=0,
        base_fence_key=base_fence_key,
        worker_count=worker_count,
        mode=mode,
        fault_point=fault_point,
    )


def _semantic_projection(outcome):
    """只取 worker 数不应改变的规范状态、报告、cursor 和 artifact。"""
    return (
        outcome.logical_state_digest,
        outcome.report_digest,
        outcome.cursor_digest,
        outcome.artifact_digest,
    )


def test_one_two_four_workers_have_identical_logical_outputs(tmp_path: Path):
    """同一 frozen logical manifest 的 1/2/4 worker 只允许资源观测不同。"""
    outcomes = tuple(
        run_language_stage0(_config(tmp_path / f"w{workers}", worker_count=workers))
        for workers in (1, 2, 4)
    )

    assert _semantic_projection(outcomes[0]) == _semantic_projection(
        outcomes[1]) == _semantic_projection(outcomes[2])
    assert {item.resource_report["requested_workers"] for item in outcomes} == {
        1, 2, 4}
    assert all(item.report["status"] == "W01_PROTOCOL_VERIFIED"
               for item in outcomes)
    assert all(item.report["execution_state"]["formal_training_runs"] == 0
               for item in outcomes)


@pytest.mark.parametrize("point", W01FaultPoint.injectable_points())
def test_each_fault_resumes_without_partial_adoption_or_duplicate_merge(
        tmp_path: Path, point: str):
    """六个显式故障点恢复后与 fresh 规范结果一致且只发布一个 adopted manifest。"""
    baseline = run_language_stage0(_config(tmp_path / "baseline"))
    root = tmp_path / point.lower()
    with pytest.raises(W01InjectedFault, match=point):
        run_language_stage0(_config(root, fault_point=point))

    resumed = run_language_stage0(_config(
        root,
        worker_count=4,
        mode=("resume" if point == W01FaultPoint.AFTER_MANIFEST_PUBLISH
              else "restart"),
    ))
    replay = run_language_stage0(_config(
        root,
        worker_count=2,
        mode="resume",
    ))

    assert _semantic_projection(resumed) == _semantic_projection(baseline)
    assert _semantic_projection(replay) == _semantic_projection(baseline)
    assert resumed.run_manifest_path == replay.run_manifest_path
    assert resumed.adopted_manifest_count == 1
    assert replay.merge_publication_count == 1
    assert resumed.report["execution_state"]["W02_STARTED"] == 0


def test_same_run_replay_is_idempotent_but_identity_drift_fails_closed(
        tmp_path: Path):
    """同 identity 重放零新增；同 run 的 base fence 漂移不得续用旧 cursor。"""
    root = tmp_path / "identity"
    first = run_language_stage0(_config(root))
    second = run_language_stage0(_config(root, worker_count=4, mode="resume"))
    assert _semantic_projection(first) == _semantic_projection(second)
    assert second.transaction_event_count == first.transaction_event_count

    with pytest.raises(Exception, match="identity|身份|base fence"):
        run_language_stage0(_config(
            root,
            mode="resume",
            base_fence_key=(1, 9, 20260729),
        ))


def test_sqlite_close_does_not_commit_and_fresh_process_state_can_resume(
        tmp_path: Path):
    """commit 后 manifest 前故障跨新连接恢复，证明不依赖 close 隐式提交。"""
    root = tmp_path / "sqlite-restart"
    point = W01FaultPoint.AFTER_COMMIT_BEFORE_CURSOR
    with pytest.raises(W01InjectedFault, match=point):
        run_language_stage0(_config(root, fault_point=point))

    resumed = run_language_stage0(_config(root, mode="restart", worker_count=4))
    restored = read_w01_run(resumed.run_manifest_path.parent)
    assert restored.report_digest == resumed.report_digest
    assert restored.cursor_digest == resumed.cursor_digest
    assert resumed.report["fault_contract"]["sqlite_process_restart"] == 1
    assert resumed.report["execution_state"]["LANGUAGE_CAPABILITY_MASTERED"] == 0
