"""Focused checks for the benchmark-only PERF-P3 SQLite trial."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from pure_integer_ai.experiments.ph2_dataset_contract import parse_canonical_json_bytes
from scripts.performance_p3_sqlite_trial_worker import CONTRACT, run_trial


ROOT = Path(__file__).resolve().parents[1]


def _assert_no_float(value: object) -> None:
    assert not isinstance(value, float)
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_no_float(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_float(item)


def test_trial_preserves_digests_through_reclaim_and_restart(tmp_path: Path) -> None:
    report = run_trial(2, tmp_path / "trial.sqlite3")
    assert report["contract"] == CONTRACT
    assert report["readiness_transition"] == {
        "LANGUAGE_READINESS_REPUBLISHED": 0,
        "PW00A_STARTED": 0,
    }
    assert report["stable"]["cold_query_digest"] == report["stable"]["warm_query_digest"]
    assert report["stable"]["rollback_digest"] == report["stable"]["final_visible_digest"]
    assert report["metrics"]["exception_path_verified"] == 1
    assert report["metrics"]["visible_object_count"] == 1
    assert report["phases"]["sqlite_table_registration"]["call_count"] == 4
    assert report["phases"]["sqlite_index_registration"]["call_count"] == 8
    assert report["phases"]["segment_decode"]["call_count"] >= 6
    _assert_no_float(report)


def test_trial_subprocess_emits_canonical_integer_report(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.performance_p3_sqlite_trial_worker",
            "--scale",
            "2",
            "--database",
            str(tmp_path / "subprocess.sqlite3"),
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
    report = parse_canonical_json_bytes(completed.stdout.rstrip(), require_object=True)
    assert report["contract"] == CONTRACT
    assert report["scenario"] == "storage_sqlite_schema_init_read_trial"
    _assert_no_float(report)


def test_trial_rejects_existing_database(tmp_path: Path) -> None:
    database = tmp_path / "existing.sqlite3"
    database.write_bytes(b"occupied")
    try:
        run_trial(1, database)
    except ValueError as error:
        assert "already exists" in str(error)
    else:
        raise AssertionError("existing SQLite trial database was accepted")
