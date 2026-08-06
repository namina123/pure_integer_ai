"""执行文件隔离 T3 子进程，并逐文件更新 checkpoint。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from .t3_shard_checkpoint import utc_now, write_state
from .t3_shard_contract import (
    HASH_SEED,
    RUNNER_CONTRACT,
    T3ShardRunnerError,
    build_inventory,
    canonical_bytes,
    read_head,
    require_clean_repository,
    verify_file_identity,
)


_SUMMARY_PATTERN = re.compile(
    r"(?:^|\s)(?:\d+\s+)?(?:passed|failed|error|errors|skipped|xfailed|xpassed)\b",
    re.IGNORECASE,
)
_TERMINAL_RESULTS = frozenset({"PASS", "FAIL", "TIMEOUT", "ERROR"})


def _log_slug(relative_path: str) -> str:
    """把测试相对路径压成稳定且适合 Windows 的日志文件名。"""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", relative_path).strip(".-")
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
    return f"{stem[:96] or 'test'}-{digest}"


def _log_relative_path(relative_path: str, ordinal: int, attempt: int) -> Path:
    """为一次文件尝试形成不会与前次尝试冲突的相对日志路径。"""
    return Path("logs") / (
        f"{ordinal:04d}-{_log_slug(relative_path)}-attempt-{attempt:03d}.log"
    )


def _extract_pytest_summary(log_path: Path) -> str | None:
    """从日志尾部提取 pytest 正式汇总行。"""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines[-120:]):
        normalized = line.strip().strip("=").strip()
        if _SUMMARY_PATTERN.search(normalized):
            return normalized
    return None


def _pytest_environment() -> dict[str, str]:
    """构造不继承外部 pytest 参数和插件自动加载的固定环境。"""
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = HASH_SEED
    environment["PYTHONUTF8"] = "1"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment.pop("PYTEST_ADDOPTS", None)
    return environment


def _spawn_pytest(
    command: list[str],
    repository_root: Path,
    log_stream: Any,
) -> subprocess.Popen[bytes]:
    """在独立进程组中启动 pytest，便于 timeout 时清理完整子树。"""
    options: dict[str, Any] = {
        "cwd": repository_root,
        "env": _pytest_environment(),
        "stdout": log_stream,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    return subprocess.Popen(command, **options)


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """强制终止超时或中断的 pytest 进程组及其后代。"""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_test_file(
    repository_root: Path,
    state_root: Path,
    relative_path: str,
    *,
    ordinal: int,
    attempt: int,
    timeout_seconds: int,
    expected_head: str,
) -> dict[str, Any]:
    """在 fresh pytest 子进程中运行一个测试文件并形成独立日志记录。"""
    log_directory = state_root / "logs"
    temporary_directory = state_root / "tmp" / f"{ordinal:04d}-{attempt:03d}"
    log_directory.mkdir(parents=True, exist_ok=True)
    temporary_directory.parent.mkdir(parents=True, exist_ok=True)
    log_relative = _log_relative_path(relative_path, ordinal, attempt)
    log_path = state_root / log_relative
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-ra",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(temporary_directory),
        relative_path,
    ]
    started_at = utc_now()
    started_ns = time.monotonic_ns()
    status = "ERROR"
    return_code: int | None = None
    with log_path.open("xb") as log_stream:
        header = {
            "contract": RUNNER_CONTRACT,
            "head": expected_head,
            "test_file": relative_path,
            "attempt": attempt,
            "command": command,
            "pythonhashseed": HASH_SEED,
        }
        log_stream.write(canonical_bytes(header))
        log_stream.flush()
        process = _spawn_pytest(command, repository_root, log_stream)
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            return_code = process.returncode
            status = "TIMEOUT"
        except KeyboardInterrupt:
            _terminate_process_tree(process)
            return_code = process.returncode
            status = "INTERRUPTED"
        else:
            status = "PASS" if return_code == 0 else "FAIL"
    duration_ms = (time.monotonic_ns() - started_ns) // 1_000_000
    summary = _extract_pytest_summary(log_path)
    if status == "PASS" and summary is None:
        status = "ERROR"
    shutil.rmtree(temporary_directory, ignore_errors=True)
    return {
        "status": status,
        "attempt": attempt,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "duration_ms": duration_ms,
        "return_code": return_code,
        "pytest_summary": summary,
        "log_path": log_relative.as_posix(),
    }


def _inventory_by_path(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """建立 checkpoint 内逐文件身份的只读索引。"""
    return {item["path"]: item for item in state["inventory"]["files"]}


def _refresh_aggregate(state: dict[str, Any]) -> None:
    """只从选中文件的最新状态重算 aggregate，不信任旧汇总字段。"""
    selected = state["selected_files"]
    results = state["results"]
    statuses = [results.get(path, {}).get("status", "PENDING") for path in selected]
    if statuses and all(status == "PASS" for status in statuses):
        aggregate = "PASS"
    elif any(status in {"FAIL", "TIMEOUT", "ERROR"} for status in statuses):
        aggregate = "FAIL"
    elif any(status == "RUNNING" for status in statuses):
        aggregate = "RUNNING"
    else:
        aggregate = "INCOMPLETE"
    state["aggregate_status"] = aggregate
    state["updated_at_utc"] = utc_now()


def run_state(
    repository_root: Path,
    state_root: Path,
    state: dict[str, Any],
    *,
    retry_failed: bool,
    max_files: int | None,
) -> dict[str, Any]:
    """执行 checkpoint 中尚未闭合的文件，并在每次尝试后原子保存。"""
    if max_files is not None and max_files < 1:
        raise T3ShardRunnerError("max_files 必须大于零")
    repository_root = repository_root.resolve()
    state_root = state_root.resolve()
    identity = state["identity"]
    expected_head = identity["head"]
    if read_head(repository_root) != expected_head:
        raise T3ShardRunnerError("运行开始前 HEAD 已漂移")
    inventory = _inventory_by_path(state)
    attempted = 0
    for ordinal, relative in enumerate(state["selected_files"]):
        existing = state["results"].get(relative, {})
        current_status = existing.get("status")
        if current_status == "PASS":
            continue
        if current_status in _TERMINAL_RESULTS and not retry_failed:
            if not identity["continue_on_failure"]:
                break
            continue
        if max_files is not None and attempted >= max_files:
            break
        if read_head(repository_root) != expected_head:
            raise T3ShardRunnerError("运行期间 HEAD 已漂移")
        verify_file_identity(repository_root, inventory[relative])
        attempts = list(existing.get("attempts", []))
        if current_status == "RUNNING" and attempts:
            interrupted = dict(attempts[-1])
            interrupted["status"] = "INTERRUPTED"
            interrupted["finished_at_utc"] = utc_now()
            attempts[-1] = interrupted
        attempt = len(attempts) + 1
        running_attempt = {
            "status": "RUNNING",
            "attempt": attempt,
            "started_at_utc": utc_now(),
            "log_path": _log_relative_path(relative, ordinal, attempt).as_posix(),
        }
        attempts.append(running_attempt)
        state["results"][relative] = {
            "status": "RUNNING",
            "attempts": list(attempts),
        }
        _refresh_aggregate(state)
        write_state(state_root, state)
        print(
            f"[{ordinal + 1}/{len(state['selected_files'])}] RUN {relative} attempt={attempt}",
            flush=True,
        )
        result = run_test_file(
            repository_root,
            state_root,
            relative,
            ordinal=ordinal,
            attempt=attempt,
            timeout_seconds=identity["file_timeout_seconds"],
            expected_head=expected_head,
        )
        attempts[-1] = result
        state["results"][relative] = {
            "status": result["status"],
            "attempts": attempts,
        }
        attempted += 1
        _refresh_aggregate(state)
        write_state(state_root, state)
        print(
            f"[{ordinal + 1}/{len(state['selected_files'])}] {result['status']} "
            f"{relative} duration_ms={result['duration_ms']}",
            flush=True,
        )
        if result["status"] == "INTERRUPTED":
            raise KeyboardInterrupt
        if result["status"] != "PASS" and not identity["continue_on_failure"]:
            break
    require_clean_repository(repository_root)
    if read_head(repository_root) != expected_head:
        raise T3ShardRunnerError("运行结束时 HEAD 已漂移")
    final_inventory = build_inventory(repository_root)
    if final_inventory != state["inventory"]:
        raise T3ShardRunnerError("运行结束时测试 inventory 已漂移")
    _refresh_aggregate(state)
    write_state(state_root, state)
    return state


def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    """生成不依赖缓存 aggregate 的当前覆盖摘要。"""
    counts: dict[str, int] = {}
    for relative in state["selected_files"]:
        status = state["results"].get(relative, {}).get("status", "PENDING")
        counts[status] = counts.get(status, 0) + 1
    return {
        "aggregate_status": state["aggregate_status"],
        "head": state["identity"]["head"],
        "inventory_sha256": state["identity"]["inventory_sha256"],
        "selection_sha256": state["identity"]["selection_sha256"],
        "selected_file_count": len(state["selected_files"]),
        "status_counts": dict(sorted(counts.items())),
    }
