"""PERF-P2 阶段 profiler 的整数指标与能力边界。"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    parse_canonical_json_bytes,
)
from scripts import performance_p2_phase_worker


ROOT = Path(__file__).resolve().parents[1]


def _run_profile(
        scenario: str,
        tmp_path: Path,
        ) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "scripts.performance_p2_phase_worker",
        "--scenario",
        scenario,
        "--scale",
        "2",
    ]
    if scenario == "storage_sqlite":
        command.extend(("--database", str(tmp_path / "profile.sqlite3")))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(ROOT / "src"),
        },
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    value = parse_canonical_json_bytes(
        completed.stdout.rstrip(), require_object=True)
    assert isinstance(value, dict)
    return value


def _assert_no_float(value: object) -> None:
    assert not isinstance(value, float)
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_no_float(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_float(item)


def test_sqlite_phase_profile_is_integer_and_readiness_neutral(
        tmp_path: Path,
        ) -> None:
    value = _run_profile("storage_sqlite", tmp_path)
    assert value["baseline_report"]["stable_digest"]
    assert value["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }
    phases = value["phases"]
    assert phases["sqlite_commit"]["call_count"] > 0
    assert phases["sqlite_insert"]["duration_ns"] > 0
    assert phases["publish_delta"]["call_count"] == 1
    assert phases["query_total"]["call_count"] == 2
    _assert_no_float(value)


def test_memory_phase_profile_matches_baseline_digest(tmp_path: Path) -> None:
    profile = _run_profile("long_memory_projection", tmp_path)
    baseline = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.performance_baseline_worker",
            "--scenario",
            "long_memory_projection",
            "--scale",
            "2",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(ROOT / "src"),
        },
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    baseline_value = parse_canonical_json_bytes(
        baseline.stdout.rstrip(), require_object=True)
    assert (
        profile["baseline_report"]["stable_digest"]
        == baseline_value["stable_digest"]
    )
    phases = profile["phases"]
    assert phases["memory_segment_construction"]["duration_ns"] > 0
    assert phases["memory_stable_key_restore"]["duration_ns"] > 0
    _assert_no_float(profile)


def test_phase_profile_rejects_second_in_process_run(monkeypatch) -> None:
    monkeypatch.setattr(
        performance_p2_phase_worker, "_PROFILE_ALREADY_RUN", True)
    with pytest.raises(RuntimeError, match="每个进程只允许运行一次"):
        performance_p2_phase_worker.run_profile(
            "long_memory_projection", 1)
