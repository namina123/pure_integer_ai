"""定义文件隔离 T3 的 Git、inventory、selection 与运行身份合同。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any


RUNNER_CONTRACT = "FILE_ISOLATED_COMPOSITE_T3_V1"
HASH_SEED = "0"


class T3ShardRunnerError(RuntimeError):
    """T3 分片的身份、状态或执行边界不成立。"""


def canonical_bytes(value: dict[str, Any]) -> bytes:
    """把状态写成稳定 JSON，保证 checkpoint 可逐字节审计。"""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    """返回字节串的十六进制 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """流式计算文件摘要，避免把较大测试材料整体载入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run_git(repository_root: Path, *arguments: str) -> bytes:
    """在指定仓库运行只读 Git 命令并返回原始标准输出。"""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise T3ShardRunnerError(f"Git 命令失败: {detail or arguments!r}")
    return completed.stdout


def read_head(repository_root: Path) -> str:
    """读取当前提交身份并要求它是完整十六进制对象名。"""
    head = run_git(repository_root, "rev-parse", "HEAD").decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", head):
        raise T3ShardRunnerError("HEAD 身份不是完整十六进制对象名")
    return head


def require_clean_repository(repository_root: Path) -> None:
    """拒绝在含跟踪或未跟踪改动的公开仓库上形成 T3 证据。"""
    status = run_git(
        repository_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if status:
        raise T3ShardRunnerError("公开仓库不是 clean 状态")


def _safe_test_path(relative_path: str) -> str:
    """校验 Git 返回的测试路径不会逃逸仓库或混入非测试文件。"""
    if not relative_path or "\\" in relative_path or ":" in relative_path:
        raise T3ShardRunnerError("测试路径不是 canonical POSIX 相对路径")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise T3ShardRunnerError("测试路径逃逸公开仓库")
    if not relative.parts or relative.parts[0] != "tests":
        raise T3ShardRunnerError("测试路径不属于 tests")
    if not relative.name.startswith("test_") or relative.suffix != ".py":
        raise T3ShardRunnerError("测试 inventory 混入非 test_*.py 文件")
    return relative.as_posix()


def build_inventory(repository_root: Path) -> dict[str, Any]:
    """冻结全部受 Git 跟踪的测试文件及逐文件字节身份。"""
    output = run_git(repository_root, "ls-files", "-z", "--", "tests")
    raw_paths = output.decode("utf-8").split("\0")
    paths = sorted(
        _safe_test_path(value)
        for value in raw_paths
        if value
        and PurePosixPath(value).name.startswith("test_")
        and PurePosixPath(value).suffix == ".py"
    )
    if not paths or len(paths) != len(set(paths)):
        raise T3ShardRunnerError("测试 inventory 为空或含重复路径")
    files: list[dict[str, Any]] = []
    for relative in paths:
        path = repository_root / Path(*PurePosixPath(relative).parts)
        if not path.is_file():
            raise T3ShardRunnerError(f"跟踪测试文件不存在: {relative}")
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {"files": files}
    return {
        "file_count": len(files),
        "sha256": sha256_bytes(canonical_bytes(payload)),
        "files": files,
    }


def select_files(
    inventory: dict[str, Any],
    *,
    shard_count: int,
    shard_index: int,
    start_at: str | None,
    end_at: str | None,
) -> tuple[str, ...]:
    """按稳定文件序应用闭区间边界，再用模分片形成不重叠选择。"""
    if shard_count < 1:
        raise T3ShardRunnerError("shard_count 必须大于零")
    if shard_index < 0 or shard_index >= shard_count:
        raise T3ShardRunnerError("shard_index 超出 shard_count")
    paths = tuple(item["path"] for item in inventory["files"])
    start = 0
    stop = len(paths)
    if start_at is not None:
        canonical_start = _safe_test_path(start_at)
        try:
            start = paths.index(canonical_start)
        except ValueError as error:
            raise T3ShardRunnerError(f"start_at 不在 inventory: {canonical_start}") from error
    if end_at is not None:
        canonical_end = _safe_test_path(end_at)
        try:
            stop = paths.index(canonical_end) + 1
        except ValueError as error:
            raise T3ShardRunnerError(f"end_at 不在 inventory: {canonical_end}") from error
    if start >= stop:
        raise T3ShardRunnerError("测试选择边界为空或倒置")
    bounded = paths[start:stop]
    selected = tuple(
        path for ordinal, path in enumerate(bounded) if ordinal % shard_count == shard_index
    )
    if not selected:
        raise T3ShardRunnerError("当前分片没有测试文件")
    return selected


def read_pytest_identity(repository_root: Path) -> str:
    """读取当前解释器实际使用的 pytest 版本身份。"""
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = HASH_SEED
    environment["PYTHONUTF8"] = "1"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment.pop("PYTEST_ADDOPTS", None)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--version"],
        cwd=repository_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0 or not output:
        raise T3ShardRunnerError("无法读取 pytest 版本")
    return output


def build_identity(
    repository_root: Path,
    inventory: dict[str, Any],
    selected_files: tuple[str, ...],
    *,
    shard_count: int,
    shard_index: int,
    start_at: str | None,
    end_at: str | None,
    file_timeout_seconds: int,
    continue_on_failure: bool,
) -> dict[str, Any]:
    """构造恢复时必须逐字段相等的运行身份。"""
    selection_payload = {"files": list(selected_files)}
    return {
        "contract": RUNNER_CONTRACT,
        "head": read_head(repository_root),
        "inventory_sha256": inventory["sha256"],
        "inventory_file_count": inventory["file_count"],
        "selection_sha256": sha256_bytes(canonical_bytes(selection_payload)),
        "selected_file_count": len(selected_files),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "start_at": start_at,
        "end_at": end_at,
        "file_timeout_seconds": file_timeout_seconds,
        "continue_on_failure": continue_on_failure,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "pytest_identity": read_pytest_identity(repository_root),
        "pytest_arguments": ["-q", "-ra", "--tb=short", "-p", "no:cacheprovider"],
        "pythonhashseed": HASH_SEED,
    }


def verify_file_identity(repository_root: Path, expected: dict[str, Any]) -> None:
    """在启动子进程前核对当前测试文件仍与冻结 inventory 相同。"""
    relative = expected["path"]
    path = repository_root / Path(*PurePosixPath(relative).parts)
    if not path.is_file():
        raise T3ShardRunnerError(f"测试文件在运行前消失: {relative}")
    if path.stat().st_size != expected["size"] or sha256_file(path) != expected["sha256"]:
        raise T3ShardRunnerError(f"测试文件在运行前漂移: {relative}")
