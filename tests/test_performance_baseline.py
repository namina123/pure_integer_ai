"""PERF-P0 基线合同、身份冻结、workload 和 Git 外 checkpoint 测试。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    parse_canonical_json_bytes,
)
from scripts.performance_baseline_contract import (
    IMPLEMENTED_SCENARIOS,
    PerformanceBaselineError,
    build_initial_state,
    build_manifest,
    canonical_bytes,
    read_state,
    require_external_state_root,
    write_state,
)
from scripts.performance_baseline_runner import init_state, run_state


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_has_unique_implemented_scenarios_and_readiness_zero():
    """P0 场景必须有唯一身份且不得改变语言 readiness。"""
    manifest = build_manifest(ROOT, scale=4, repetitions=1)
    assert tuple(item["name"] for item in manifest["scenarios"]) == (
        *IMPLEMENTED_SCENARIOS,
    )
    assert manifest["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }
    for item in manifest["scenarios"]:
        assert item["source_bindings"]


def test_state_is_canonical_and_external_root_is_required(tmp_path: Path):
    """checkpoint 必须 canonical 且不能落入公开 Git。"""
    state_root = tmp_path / "state"
    manifest = build_manifest(ROOT, scale=2, repetitions=1,
                              scenarios=("long_input_hierarchy",))
    write_state(state_root, build_initial_state(manifest))
    target = state_root / "state.json"
    assert target.read_bytes() == canonical_bytes(read_state(state_root))
    with pytest.raises(PerformanceBaselineError, match="公开 Git 根之外"):
        require_external_state_root(ROOT, ROOT / "state")


def test_state_write_does_not_overwrite_guard_is_runner_level(
        tmp_path: Path, monkeypatch):
    """同一 state 可重读，但 init 不得覆盖已有 checkpoint。"""
    state_root = tmp_path / "state"
    monkeypatch.setattr(
        "scripts.performance_baseline_runner.require_clean_repository",
        lambda root: None,
    )
    init_state(state_root, scale=2, repetitions=1,
               scenarios=("long_input_hierarchy",))
    with pytest.raises(PerformanceBaselineError, match="已存在"):
        init_state(state_root, scale=2, repetitions=1,
                   scenarios=("long_input_hierarchy",))


@pytest.mark.parametrize("scenario", (
    "long_input_hierarchy",
    "long_session_checkpoint",
    "long_memory_projection",
))
def test_small_workloads_emit_integer_canonical_report(scenario: str):
    """前三个非 SQLite workload 必须能在极小规模独立运行。"""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.performance_baseline_worker",
            "--scenario",
            scenario,
            "--scale",
            "2",
        ],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONHASHSEED": "0",
             "PYTHONPATH": str(ROOT / "src")},
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    value = parse_canonical_json_bytes(
        completed.stdout.rstrip(), require_object=True)
    assert value["scenario"] == scenario
    assert value["scale"] == 2
    assert value["peak_rss_bytes"] > 0
    assert all(type(item) is int for item in value["metrics"].values()
               if type(item) is int)


def test_runner_can_complete_one_scenario_and_resume(
        tmp_path: Path, monkeypatch):
    """runner 逐场景写 checkpoint，已 PASS 场景恢复时不重复执行。"""
    state_root = tmp_path / "state"
    monkeypatch.setattr(
        "scripts.performance_baseline_runner.require_clean_repository",
        lambda root: None,
    )
    init_state(state_root, scale=2, repetitions=1,
               scenarios=("long_input_hierarchy",))
    state = run_state(state_root, timeout_seconds=30)
    assert state["aggregate_status"] == "PASS"
    attempts = state["results"]["long_input_hierarchy"]["attempts"]
    assert len(attempts) == 1
    frozen = read_state(state_root)
    resumed = run_state(state_root, timeout_seconds=30)
    assert resumed == frozen


def test_runner_keeps_timeout_and_retries_only_when_explicit(
        tmp_path: Path, monkeypatch):
    """TIMEOUT 必须封存，只有显式 retry 才追加新尝试。"""
    state_root = tmp_path / "state"
    monkeypatch.setattr(
        "scripts.performance_baseline_runner.require_clean_repository",
        lambda root: None,
    )
    init_state(state_root, scale=2, repetitions=1,
               scenarios=("long_input_hierarchy",))
    monkeypatch.setattr(
        "scripts.performance_baseline_runner._run_one",
        lambda *args, **kwargs: {
            "status": "TIMEOUT", "attempt": 1, "duration_ns": 1,
            "log_sha256": "0" * 64,
        },
    )
    timed_out = run_state(state_root, timeout_seconds=1)
    assert timed_out["aggregate_status"] == "FAIL"
    unchanged = run_state(state_root, timeout_seconds=1)
    assert unchanged == read_state(state_root)
    monkeypatch.setattr(
        "scripts.performance_baseline_runner._run_one",
        lambda *args, **kwargs: {
            "status": "PASS", "attempt": 2, "duration_ns": 1,
            "log_sha256": "1" * 64,
            "report": {"scenario": "long_input_hierarchy"},
        },
    )
    recovered = run_state(
        state_root, timeout_seconds=1, retry_failed=True)
    assert recovered["aggregate_status"] == "PASS"
    assert tuple(
        item["status"] for item in
        recovered["results"]["long_input_hierarchy"]["attempts"]
    ) == ("TIMEOUT", "PASS")


def test_runner_marks_interrupted_attempt_before_resume(
        tmp_path: Path, monkeypatch):
    """崩溃留下的 RUNNING 尝试必须转为 INTERRUPTED 后再执行。"""
    state_root = tmp_path / "state"
    monkeypatch.setattr(
        "scripts.performance_baseline_runner.require_clean_repository",
        lambda root: None,
    )
    init_state(state_root, scale=2, repetitions=1,
               scenarios=("long_input_hierarchy",))
    state = read_state(state_root)
    state["results"]["long_input_hierarchy"] = {
        "status": "RUNNING",
        "attempts": ({"status": "RUNNING", "attempt": 1},),
    }
    write_state(state_root, state)
    monkeypatch.setattr(
        "scripts.performance_baseline_runner._run_one",
        lambda *args, **kwargs: {
            "status": "PASS", "attempt": 2, "duration_ns": 1,
            "log_sha256": "2" * 64,
            "report": {"scenario": "long_input_hierarchy"},
        },
    )
    resumed = run_state(state_root, timeout_seconds=1)
    assert tuple(
        item["status"] for item in
        resumed["results"]["long_input_hierarchy"]["attempts"]
    ) == ("INTERRUPTED", "PASS")


def test_worker_report_does_not_contain_float_or_free_text_payloads(tmp_path):
    """基线报告不允许浮点指标，稳定指标只使用 canonical JSON 数据。"""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.performance_baseline_worker",
            "--scenario",
            "storage_dict",
            "--scale",
            "2",
        ],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONHASHSEED": "0",
             "PYTHONPATH": str(ROOT / "src")},
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    value = json.loads(completed.stdout)
    assert value["metrics"]["backend"] == "dict"
    assert all(not isinstance(item, float)
               for item in value["metrics"].values())


def test_sqlite_workload_records_restart_and_disk_bytes(tmp_path: Path):
    """SQLite workload 必须完成重启冷读、热读并记录实际磁盘字节。"""
    database = tmp_path / "baseline.sqlite3"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.performance_baseline_worker",
            "--scenario",
            "storage_sqlite",
            "--scale",
            "2",
            "--database",
            str(database),
        ],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONHASHSEED": "0",
             "PYTHONPATH": str(ROOT / "src")},
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    value = parse_canonical_json_bytes(
        completed.stdout.rstrip(), require_object=True)
    metrics = value["metrics"]
    assert metrics["backend"] == "sqlite"
    assert metrics["disk_bytes"] >= database.stat().st_size > 0
    assert metrics["database_file_count"] >= 1
    assert metrics["cold_query_duration_ns"] > 0
    assert metrics["warm_query_duration_ns"] > 0
