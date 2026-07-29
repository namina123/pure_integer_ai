"""W-01 fresh/restart/resume 和 SQLite 真跨进程恢复验收。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w01_contract import D03_GLOBAL_MANIFEST_PATH
from pure_integer_ai.experiments.ph2_w01_faults import W01FaultPoint
from pure_integer_ai.experiments.ph2_w01_runtime import (
    W01RuntimeConfig,
    run_language_stage0,
)
from pure_integer_ai.experiments.ph2_w01_transaction import (
    W01_TRANSACTION_EVENT_TABLE,
    register_w01_transaction_table,
)
from pure_integer_ai.storage.backend import SQLiteBackend


_REPOSITORY = Path(__file__).resolve().parents[1]


def _config(root: Path, *, mode: str, fault_point: str | None = None):
    """构造状态机直接测试配置。"""
    return W01RuntimeConfig(
        repository_root=_REPOSITORY,
        global_manifest_path=D03_GLOBAL_MANIFEST_PATH,
        run_root=root / "runs",
        sqlite_path=root / "w01.sqlite3",
        run_id=202,
        parent_run_id=0,
        base_run_id=0,
        base_fence_key=(1, 0, 20260729),
        worker_count=1,
        mode=mode,
        fault_point=fault_point,
    )


def test_fresh_restart_resume_are_distinct_states(tmp_path: Path):
    """空状态只能 fresh；未 adopted 状态可 restart；完成后只能 resume。"""
    root = tmp_path / "states"
    with pytest.raises(RuntimeError, match="restart mode"):
        run_language_stage0(_config(root, mode="restart"))
    with pytest.raises(RuntimeError, match="resume mode"):
        run_language_stage0(_config(root, mode="resume"))

    with pytest.raises(Exception, match=W01FaultPoint.BEFORE_MERGE_PREVIEW):
        run_language_stage0(_config(
            root,
            mode="fresh",
            fault_point=W01FaultPoint.BEFORE_MERGE_PREVIEW,
        ))
    with pytest.raises(RuntimeError, match="resume mode"):
        run_language_stage0(_config(root, mode="resume"))
    completed = run_language_stage0(_config(root, mode="restart"))
    assert completed.report["status"] == "W01_PROTOCOL_VERIFIED"
    resumed = run_language_stage0(_config(root, mode="resume"))
    assert resumed.report_digest == completed.report_digest
    with pytest.raises(RuntimeError, match="fresh mode"):
        run_language_stage0(_config(root, mode="fresh"))


def test_sqlite_close_never_implicitly_commits(tmp_path: Path):
    """绕过事务 owner 的未提交 event 在 close 后必须消失。"""
    path = tmp_path / "close.sqlite3"
    backend = SQLiteBackend(str(path))
    register_w01_transaction_table(backend)
    backend.commit()
    backend.insert(W01_TRANSACTION_EVENT_TABLE, {
        "run_id": 1,
        "event_seq": 1,
        "event_kind": 1,
        "identity_sha256": "0" * 64,
        "payload_sha256": "0" * 64,
        "payload_json": "{}\n",
    })
    backend.close()

    reopened = SQLiteBackend(str(path))
    try:
        register_w01_transaction_table(reopened)
        assert reopened.select(W01_TRANSACTION_EVENT_TABLE) == []
    finally:
        reopened.close()


def _command(root: Path, *, mode: str, workers: int, fault: str | None = None):
    """构造独立 Python 进程的正式 CLI 命令。"""
    command = [
        sys.executable,
        "-m",
        "pure_integer_ai.experiments.run_ph2_language_stage0",
        "--repository-root", str(_REPOSITORY),
        "--run-root", str(root / "runs"),
        "--sqlite-path", str(root / "w01.sqlite3"),
        "--run-id", "303",
        "--worker-count", str(workers),
        "--mode", mode,
    ]
    if fault is not None:
        command.extend(("--fault-point", fault))
    return command


def test_sqlite_commit_recovers_in_fresh_python_process(tmp_path: Path):
    """第一个进程 commit 后中断，第二个 fresh interpreter 从 SQLite restart 完成。"""
    root = tmp_path / "process"
    failed = subprocess.run(
        _command(
            root,
            mode="fresh",
            workers=1,
            fault=W01FaultPoint.AFTER_COMMIT_BEFORE_CURSOR,
        ),
        cwd=_REPOSITORY,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert failed.returncode == 3
    assert W01FaultPoint.AFTER_COMMIT_BEFORE_CURSOR in failed.stderr

    restarted = subprocess.run(
        _command(root, mode="restart", workers=4),
        cwd=_REPOSITORY,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert restarted.returncode == 0, restarted.stderr
    result = json.loads(restarted.stdout)
    assert result["status"] == "W01_PROTOCOL_VERIFIED"
    assert Path(result["run_manifest_path"]).is_file()

    resumed = subprocess.run(
        _command(root, mode="resume", workers=2),
        cwd=_REPOSITORY,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout) == result


def test_hash_seed_zero_one_produce_identical_fresh_results(tmp_path: Path):
    """两个独立 hash seed 的 fresh 进程必须产生相同规范逻辑输出。"""
    results = []
    for seed in (0, 1):
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONHASHSEED"] = str(seed)
        completed = subprocess.run(
            _command(tmp_path / f"seed-{seed}", mode="fresh", workers=4),
            cwd=_REPOSITORY,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        result.pop("run_manifest_path")
        results.append(result)
    assert results[0] == results[1]
