"""PERF-P0 基线 runner：逐场景超时、checkpoint、resume 和结果身份。"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from scripts.performance_baseline_contract import (
    CONTRACT,
    IMPLEMENTED_SCENARIOS,
    PerformanceBaselineError,
    build_initial_state,
    build_manifest,
    canonical_bytes,
    file_identity,
    read_head,
    read_state,
    require_clean_repository,
    require_external_state_root,
    state_path,
    write_state,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    parse_canonical_json_bytes,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _aggregate(state: dict[str, object]) -> None:
    """从逐场景 status 重新计算 aggregate，不信任旧汇总。"""
    statuses = tuple(
        item.get("status", "PENDING")
        for item in state["results"].values()
    )
    if statuses and all(status == "PASS" for status in statuses):
        state["aggregate_status"] = "PASS"
    elif any(status in {"FAIL", "TIMEOUT", "ERROR"} for status in statuses):
        state["aggregate_status"] = "FAIL"
    elif any(status == "RUNNING" for status in statuses):
        state["aggregate_status"] = "RUNNING"
    else:
        state["aggregate_status"] = "INCOMPLETE"


def _state_source_bindings(state: dict[str, object]) -> tuple[dict[str, object], ...]:
    """返回 checkpoint manifest 内排序后的 source identity。"""
    entries = state["manifest"]["scenarios"]
    result = []
    for entry in entries:
        result.extend(entry["source_bindings"])
    return tuple(sorted(result, key=lambda item: item["path"]))


def _verify_frozen_identity(state: dict[str, object]) -> None:
    """检查 HEAD、源文件和 readiness 禁止项仍与初始 checkpoint 一致。"""
    require_clean_repository(REPOSITORY_ROOT)
    if read_head(REPOSITORY_ROOT) != state["manifest"]["head"]:
        raise PerformanceBaselineError("性能 checkpoint 运行期间 HEAD 漂移")
    if state["manifest"]["readiness_transition"] != {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
    }:
        raise PerformanceBaselineError("性能 checkpoint readiness 禁止项漂移")
    for binding in _state_source_bindings(state):
        current = file_identity(REPOSITORY_ROOT, binding["path"])
        if current != binding:
            raise PerformanceBaselineError(
                f"性能 checkpoint source identity 漂移: {binding['path']}")


def _environment() -> dict[str, str]:
    """构造不继承外部 pytest 参数的固定 workload 环境。"""
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONUTF8"] = "1"
    source = str(REPOSITORY_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source if not existing else source + os.pathsep + existing)
    environment.pop("PYTEST_ADDOPTS", None)
    return environment


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """清理超时 workload 的完整进程组。"""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ("taskkill", "/PID", str(process.pid), "/T", "/F"),
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


def _run_one(
        state_root: Path,
        scenario: str,
        *,
        scale: int,
        timeout_seconds: int,
        attempt: int,
        ) -> dict[str, object]:
    """运行一个独立 workload 并返回不含自由文本异常的结果。"""
    log_dir = state_root / "logs"
    database_dir = state_root / "databases"
    log_dir.mkdir(parents=True, exist_ok=True)
    database_dir.mkdir(parents=True, exist_ok=True)
    slug = scenario.replace("_", "-")
    log_path = log_dir / f"{slug}-attempt-{attempt:03d}.log"
    database = database_dir / f"{slug}-attempt-{attempt:03d}.sqlite3"
    command = [
        sys.executable,
        "-m",
        "scripts.performance_baseline_worker",
        "--scenario",
        scenario,
        "--scale",
        str(scale),
    ]
    if scenario == "storage_sqlite":
        command.extend(("--database", str(database)))
    started = time.perf_counter_ns()
    options: dict[str, object] = {
        "cwd": REPOSITORY_ROOT,
        "env": _environment(),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **options)
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate(process)
            stdout, stderr = process.communicate()
            log_path.write_bytes(stdout + b"\n[stderr]\n" + stderr)
            return {
                "status": "TIMEOUT",
                "attempt": attempt,
                "duration_ns": time.perf_counter_ns() - started,
                "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
            }
    except OSError as error:
        raise PerformanceBaselineError("无法启动性能 workload") from error
    log_path.write_bytes(stdout + b"\n[stderr]\n" + stderr)
    if process.returncode != 0:
        return {
            "status": "ERROR",
            "attempt": attempt,
            "duration_ns": time.perf_counter_ns() - started,
            "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        }
    try:
        report = parse_canonical_json_bytes(
            stdout.rstrip(), require_object=True)
    except (ValueError, TypeError) as error:
        raise PerformanceBaselineError(
            f"workload 输出不是 canonical JSON: {scenario}") from error
    if not isinstance(report, dict) or report.get("scenario") != scenario:
        raise PerformanceBaselineError("workload 场景身份不一致")
    return {
        "status": "PASS",
        "attempt": attempt,
        "duration_ns": time.perf_counter_ns() - started,
        "report": report,
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
    }


def init_state(
        state_root: Path,
        *,
        scale: int,
        repetitions: int,
        scenarios: tuple[str, ...] = IMPLEMENTED_SCENARIOS,
        ) -> dict[str, object]:
    """在 Git 外创建一个独占 PERF-P0 checkpoint。"""
    require_external_state_root(REPOSITORY_ROOT, state_root)
    require_clean_repository(REPOSITORY_ROOT)
    target = state_path(state_root)
    if target.exists():
        raise PerformanceBaselineError("性能 checkpoint 已存在，不能覆盖")
    manifest = build_manifest(
        REPOSITORY_ROOT,
        scale=scale,
        repetitions=repetitions,
        scenarios=scenarios,
    )
    state = build_initial_state(manifest)
    write_state(state_root, state)
    return state


def run_state(
        state_root: Path,
        *,
        timeout_seconds: int,
        max_scenarios: int | None = None,
        retry_failed: bool = False,
        ) -> dict[str, object]:
    """按 checkpoint 顺序运行场景，并在每次尝试后保存状态。"""
    if timeout_seconds < 1:
        raise PerformanceBaselineError("timeout_seconds 必须大于零")
    state = read_state(state_root)
    _verify_frozen_identity(state)
    scale = state["manifest"]["scale"]
    repetitions = state["manifest"]["repetitions"]
    selected = tuple(state["selected_scenarios"])
    attempted = 0
    for scenario in selected:
        current = state["results"][scenario]
        if current.get("status") == "PASS":
            continue
        if (current.get("status") in {"ERROR", "TIMEOUT"}
                and not retry_failed):
            continue
        attempts = list(current.get("attempts", ()))
        if attempts and attempts[-1].get("status") == "RUNNING":
            attempts[-1] = {
                **attempts[-1],
                "status": "INTERRUPTED",
            }
        passed_count = sum(
            item.get("status") == "PASS" for item in attempts)
        while passed_count < repetitions:
            if max_scenarios is not None and attempted >= max_scenarios:
                break
            _verify_frozen_identity(state)
            attempt = len(attempts) + 1
            attempts.append({"status": "RUNNING", "attempt": attempt})
            state["results"][scenario] = {
                "status": "RUNNING",
                "attempts": tuple(attempts),
            }
            _aggregate(state)
            write_state(state_root, state)
            result = _run_one(
                state_root,
                scenario,
                scale=scale,
                timeout_seconds=timeout_seconds,
                attempt=attempt,
            )
            attempts[-1] = result
            if result["status"] != "PASS":
                state["results"][scenario] = {
                    "status": result["status"],
                    "attempts": tuple(attempts),
                }
                _aggregate(state)
                write_state(state_root, state)
                break
            passed_count += 1
            state["results"][scenario] = {
                "status": (
                    "PASS" if passed_count == repetitions else "RUNNING"),
                "attempts": tuple(attempts),
            }
            _aggregate(state)
            write_state(state_root, state)
            attempted += 1
        if max_scenarios is not None and attempted >= max_scenarios:
            break
    _aggregate(state)
    write_state(state_root, state)
    return state


def _print_json(value: object) -> None:
    """向终端输出稳定 JSON，供人工或脚本读取。"""
    sys.stdout.buffer.write(canonical_bytes(value))


def main() -> None:
    """提供 list/init/run/read 四个有界命令。"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--state-root", type=Path, required=True)
    init_parser.add_argument("--scale", type=int, default=64)
    init_parser.add_argument("--repetitions", type=int, default=1)
    init_parser.add_argument("--scenario", action="append", dest="scenarios")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--state-root", type=Path, required=True)
    run_parser.add_argument("--timeout-seconds", type=int, default=120)
    run_parser.add_argument("--max-scenarios", type=int)
    run_parser.add_argument("--retry-failed", action="store_true")
    read_parser = subparsers.add_parser("read")
    read_parser.add_argument("--state-root", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "list":
        _print_json({"contract": CONTRACT, "scenarios": IMPLEMENTED_SCENARIOS})
        return
    if arguments.command == "init":
        scenarios = tuple(arguments.scenarios or IMPLEMENTED_SCENARIOS)
        _print_json(init_state(
            arguments.state_root,
            scale=arguments.scale,
            repetitions=arguments.repetitions,
            scenarios=scenarios,
        ))
        return
    if arguments.command == "read":
        _print_json(read_state(arguments.state_root))
        return
    _print_json(run_state(
        arguments.state_root,
        timeout_seconds=arguments.timeout_seconds,
        max_scenarios=arguments.max_scenarios,
        retry_failed=arguments.retry_failed,
    ))


if __name__ == "__main__":
    main()
