"""W-02 真学习 shard/merge、事务、故障恢复与 dump readback。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pure_integer_ai.experiments.ph2_w02_contract import D03_GLOBAL_MANIFEST_PATH
from pure_integer_ai.experiments.ph2_w02_runtime import (
    W02FaultPoint,
    W02InjectedFault,
    W02RuntimeConfig,
    load_w02_candidate_dump,
    run_language_stage1,
)


_REPOSITORY = Path(__file__).resolve().parents[1]
_REMOTE_COMMIT = "6322ed3d6aedf1a0fceeaffd1990ed5c9015e3f8"

_SUBPROCESS_RUNNER = r"""
import json
import sys
from pathlib import Path

from pure_integer_ai.experiments.ph2_w02_contract import D03_GLOBAL_MANIFEST_PATH
from pure_integer_ai.experiments.ph2_w02_runtime import (
    W02InjectedFault,
    W02RuntimeConfig,
    run_language_stage1,
)

root = Path(sys.argv[1])
mode = sys.argv[2]
workers = int(sys.argv[3])
fault = None if sys.argv[4] == "-" else sys.argv[4]
config = W02RuntimeConfig(
    repository_root=Path.cwd(),
    global_manifest_path=D03_GLOBAL_MANIFEST_PATH,
    run_root=root / "runs",
    sqlite_path=root / "w02.sqlite3",
    run_id=202,
    parent_run_id=1,
    base_run_id=1,
    base_fence_key=(1, 1, 20260729),
    worker_count=workers,
    mode=mode,
    current_remote_commit_sha1="6322ed3d6aedf1a0fceeaffd1990ed5c9015e3f8",
    fault_point=fault,
)
try:
    outcome = run_language_stage1(config)
except W02InjectedFault as exc:
    print(json.dumps({"fault": str(exc)}, sort_keys=True))
else:
    print(json.dumps({
        "artifact": outcome.artifact_digest,
        "core": outcome.core_digest,
        "cursor": outcome.cursor_digest,
        "dump": outcome.dump_manifest_sha256,
        "logical": outcome.logical_state_digest,
        "memory": outcome.memory_digest,
        "transaction_events": outcome.transaction_event_count,
        "use": outcome.use_digest,
    }, sort_keys=True))
"""


def _config(
        root: Path,
        *,
        worker_count: int = 1,
        mode: str = "fresh",
        fault_point: str | None = None,
        base_fence_key: tuple[int, ...] = (1, 1, 20260729),
        ) -> W02RuntimeConfig:
    """构造只消费正式 LC-01/02 train payload 的临时阶段运行。"""
    return W02RuntimeConfig(
        repository_root=_REPOSITORY,
        global_manifest_path=D03_GLOBAL_MANIFEST_PATH,
        run_root=root / "runs",
        sqlite_path=root / "w02.sqlite3",
        run_id=202,
        parent_run_id=1,
        base_run_id=1,
        base_fence_key=base_fence_key,
        worker_count=worker_count,
        mode=mode,
        current_remote_commit_sha1=_REMOTE_COMMIT,
        fault_point=fault_point,
    )


def _logical_projection(outcome):
    """只取 worker、恢复方式和物理资源不得改变的规范结果。"""
    return (
        outcome.logical_state_digest,
        outcome.core_digest,
        outcome.memory_digest,
        outcome.use_digest,
        outcome.cursor_digest,
        outcome.artifact_digest,
        outcome.dump_manifest_sha256,
    )


def _subprocess_run(
        root: Path,
        *,
        mode: str,
        workers: int,
        hash_seed: int,
        fault_point: str | None = None,
        ) -> dict[str, object]:
    """在全新解释器和显式 hash seed 中执行一次 W-02 runtime。"""
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(hash_seed)
    source_root = str(_REPOSITORY / "src")
    prior_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not prior_path else source_root + os.pathsep + prior_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _SUBPROCESS_RUNNER,
            str(root),
            mode,
            str(workers),
            "-" if fault_point is None else fault_point,
        ],
        cwd=_REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_one_two_four_workers_produce_identical_learning_and_dump(tmp_path):
    """16 logical shard 在 1/2/4 worker 下形成 bit-identical host 与 dump。"""
    outcomes = tuple(
        run_language_stage1(_config(tmp_path / f"w{workers}", worker_count=workers))
        for workers in (1, 2, 4)
    )
    assert _logical_projection(outcomes[0]) == _logical_projection(
        outcomes[1]) == _logical_projection(outcomes[2])
    assert {item.resource_report["requested_workers"] for item in outcomes} == {
        1, 2, 4}
    assert all(item.resource_report["logical_shards"] == 16
               for item in outcomes)
    assert all(item.resource_report["merged_records"] == 76
               for item in outcomes)
    assert all(item.execution_state == {
        "W02_STARTED": 1,
        "formal_training_runs": 1,
        "teacher_evidence_reads": 19,
        "teacher_calls": 0,
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "mastered_claims": 0,
        "readiness_claims": 0,
        "W03_STARTED": 0,
    } for item in outcomes)


@pytest.mark.parametrize("point", W02FaultPoint.injectable_points())
def test_each_fault_recovers_without_partial_learning_or_duplicate_adoption(
        tmp_path, point):
    """六故障点恢复后与 fresh 相同，manifest、merge、Use/outcome 均不重复。"""
    baseline = run_language_stage1(_config(tmp_path / "baseline"))
    root = tmp_path / point.lower()
    with pytest.raises(W02InjectedFault, match=point):
        run_language_stage1(_config(root, fault_point=point))
    mode = ("resume" if point == W02FaultPoint.AFTER_MANIFEST_PUBLISH
            else "restart")
    recovered = run_language_stage1(_config(
        root, worker_count=4, mode=mode))
    replay = run_language_stage1(_config(
        root, worker_count=2, mode="resume"))
    assert _logical_projection(recovered) == _logical_projection(baseline)
    assert _logical_projection(replay) == _logical_projection(baseline)
    assert recovered.adopted_manifest_count == 1
    assert replay.adopted_manifest_count == 1
    assert recovered.transaction_event_count == 4
    assert replay.transaction_event_count == 4
    assert recovered.merge_publication_count == 1
    assert replay.merge_publication_count == 1
    assert recovered.learning_report.candidate_count == 19
    assert replay.learning_report.replayed is True
    assert replay.learning_report.core_learning_writes == 0
    assert replay.attribution_report.outcome_count == 2


def test_dump_loads_into_fresh_sqlite_and_rejects_same_run_identity_drift(
        tmp_path):
    """权威 dump 在新 SQLite 回读同状态；同 run 的 base fence 漂移被事务拒绝。"""
    root = tmp_path / "run"
    outcome = run_language_stage1(_config(root))
    readback = load_w02_candidate_dump(
        _config(root, mode="resume"),
        target_sqlite_path=tmp_path / "readback.sqlite3",
    )
    assert _logical_projection(readback) == _logical_projection(outcome)
    assert readback.dump_readback is True
    with pytest.raises(RuntimeError, match="identity"):
        run_language_stage1(_config(
            root,
            mode="resume",
            base_fence_key=(9, 9, 20260729),
        ))


def test_sqlite_fresh_restart_resume_crosses_real_process_boundaries(tmp_path):
    """commit 后崩溃由新解释器 restart/resume，结果与独立 fresh 完全相同。"""
    baseline = _subprocess_run(
        tmp_path / "baseline",
        mode="fresh",
        workers=1,
        hash_seed=11,
    )
    root = tmp_path / "restart"
    failed = _subprocess_run(
        root,
        mode="fresh",
        workers=1,
        hash_seed=23,
        fault_point=W02FaultPoint.AFTER_COMMIT_BEFORE_CURSOR,
    )
    assert W02FaultPoint.AFTER_COMMIT_BEFORE_CURSOR in failed["fault"]
    recovered = _subprocess_run(
        root,
        mode="restart",
        workers=4,
        hash_seed=37,
    )
    replay = _subprocess_run(
        root,
        mode="resume",
        workers=2,
        hash_seed=41,
    )
    assert recovered == replay == baseline
    assert recovered["transaction_events"] == 4


def test_dual_pythonhashseed_produces_identical_host_and_dump(tmp_path):
    """两个 hash seed、不同物理 worker 不改变 host、artifact 或 manifest 字节。"""
    first = _subprocess_run(
        tmp_path / "seed_a",
        mode="fresh",
        workers=1,
        hash_seed=1,
    )
    second = _subprocess_run(
        tmp_path / "seed_b",
        mode="fresh",
        workers=4,
        hash_seed=987654,
    )
    assert first == second
