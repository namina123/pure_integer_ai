"""W04-03/W04-04 runtime、worker、fault 和 dump/readback 专项。"""
from __future__ import annotations

from pathlib import Path

import pytest

import pure_integer_ai.experiments.ph2_w04_runtime as runtime_owner
from pure_integer_ai.experiments.ph2_w04_contract import (
    W04_FORMAL_RUN_ID,
    W04_W03_BASE_RUN_ID,
)
from pure_integer_ai.experiments.ph2_w04_faults import (
    W04_FAILURE_POINT_KEYS,
    W04InjectedFault,
)
from pure_integer_ai.experiments.ph2_w04_runtime import (
    W04RuntimeConfig,
    load_w04_candidate_dump,
    run_language_stage4,
)
from tests.w04_historical_context import open_historical_w04_context


ROOT = Path(__file__).resolve().parents[1]
HEAD = "da69958c1f149a2f264053f7b7407a53f575cd93"
GLOBAL = "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"


def _config(tmp_path, *, worker=1, mode="fresh", fault=None):
    return W04RuntimeConfig(
        repository_root=ROOT,
        global_manifest_path=GLOBAL,
        run_root=tmp_path / f"run-{worker}-{mode}",
        sqlite_path=tmp_path / f"host-{worker}-{mode}.sqlite",
        run_id=W04_FORMAL_RUN_ID,
        parent_run_id=W04_W03_BASE_RUN_ID,
        base_run_id=W04_W03_BASE_RUN_ID,
        base_fence_key=None,
        worker_count=worker,
        mode=mode,
        current_remote_commit_sha1=HEAD,
        fault_point=fault,
    )


@pytest.fixture(autouse=True)
def _historical_runtime_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """历史 runtime 行为消费冻结 gate，生产 opener 仍由 contract 专项审计。"""
    monkeypatch.setattr(
        runtime_owner,
        "open_w04_frozen_context",
        open_historical_w04_context,
    )


def _logic_key(outcome):
    return (
        outcome.logical_state_digest,
        outcome.candidate_digest,
        outcome.understanding_digest,
        outcome.reasoning_digest,
        outcome.generation_digest,
        outcome.active_candidate_count,
        outcome.artifact_counts,
    )


def test_w04_runtime_worker_and_mode_are_bit_identical(tmp_path):
    """1/2/4 worker 与 fresh/restart/resume 逻辑输出 bit-identical。"""
    outcomes = [
        run_language_stage4(_config(tmp_path, worker=worker, mode=mode))
        for worker in (1, 2, 4)
        for mode in ("fresh", "restart", "resume")
    ]
    assert len({_logic_key(item) for item in outcomes}) == 1
    for outcome in outcomes:
        assert outcome.active_candidate_count == 1
        assert outcome.transaction_event_count == 4
        assert outcome.resource_report["teacher_calls"] == 0
        assert outcome.resource_report["actual_payload_gets"] > 0
        assert outcome.resource_report["actual_payload_bytes"] <= (
            outcome.resource_budget["max_payload_bytes"])


def test_w04_runtime_dump_readback_matches_logical_state(tmp_path):
    """dump/readback 保持逻辑 digest 一致且不写 teacher/private label。"""
    config = _config(tmp_path)
    outcome = run_language_stage4(config)
    readback = load_w04_candidate_dump(config)
    assert readback.dump_readback
    assert _logic_key(readback) == _logic_key(outcome)
    assert readback.teacher_calls == 0


@pytest.mark.parametrize("fault", W04_FAILURE_POINT_KEYS)
def test_w04_runtime_fault_points_recover_with_restart(tmp_path, fault):
    """六故障点命中后，同一 root 重新 restart 可恢复到同一逻辑状态。"""
    with pytest.raises(W04InjectedFault):
        run_language_stage4(_config(tmp_path, mode="fresh", fault=fault))
    recovered = run_language_stage4(_config(tmp_path, mode="restart"))
    clean = run_language_stage4(_config(tmp_path / "clean", mode="fresh"))
    assert _logic_key(recovered) == _logic_key(clean)
