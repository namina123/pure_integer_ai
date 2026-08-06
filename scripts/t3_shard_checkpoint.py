"""持久化并恢复文件隔离 T3 的 canonical checkpoint。"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from .t3_shard_contract import (
    T3ShardRunnerError,
    build_identity,
    build_inventory,
    canonical_bytes,
    require_clean_repository,
    select_files,
)


SCHEMA_VERSION = 1
STATE_FILE_NAME = "state.json"


def utc_now() -> str:
    """返回只用于外部验证记录的 UTC 时间，不进入认知核心。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _state_path(state_root: Path) -> Path:
    """返回固定 checkpoint 路径。"""
    return state_root / STATE_FILE_NAME


def _require_external_state_root(repository_root: Path, state_root: Path) -> None:
    """禁止把状态和日志写进公开仓库。"""
    repository = repository_root.resolve()
    target = state_root.resolve()
    try:
        target.relative_to(repository)
    except ValueError:
        return
    raise T3ShardRunnerError("state_root 必须位于公开 Git 根之外")


def write_state(state_root: Path, state: dict[str, Any]) -> None:
    """使用同目录原子替换写入 canonical checkpoint。"""
    state_root.mkdir(parents=True, exist_ok=True)
    target = _state_path(state_root)
    temporary = state_root / f".{STATE_FILE_NAME}.{os.getpid()}.tmp"
    payload = canonical_bytes(state)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_state(state_root: Path) -> dict[str, Any]:
    """严格读取既有 checkpoint，拒绝非 canonical 或错误 schema。"""
    path = _state_path(state_root)
    try:
        payload = path.read_bytes()
        state = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise T3ShardRunnerError("无法读取恢复 checkpoint") from error
    if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
        raise T3ShardRunnerError("恢复 checkpoint schema 不受支持")
    if payload != canonical_bytes(state):
        raise T3ShardRunnerError("恢复 checkpoint 不是 canonical JSON")
    return state


def prepare_state(
    repository_root: Path,
    state_root: Path,
    *,
    resume: bool,
    shard_count: int,
    shard_index: int,
    start_at: str | None,
    end_at: str | None,
    file_timeout_seconds: int,
    continue_on_failure: bool,
) -> dict[str, Any]:
    """冻结新运行或严格核对可恢复运行的全部身份。"""
    repository_root = repository_root.resolve()
    state_root = state_root.resolve()
    if file_timeout_seconds < 1:
        raise T3ShardRunnerError("file_timeout_seconds 必须大于零")
    _require_external_state_root(repository_root, state_root)
    require_clean_repository(repository_root)
    inventory = build_inventory(repository_root)
    selected_files = select_files(
        inventory,
        shard_count=shard_count,
        shard_index=shard_index,
        start_at=start_at,
        end_at=end_at,
    )
    identity = build_identity(
        repository_root,
        inventory,
        selected_files,
        shard_count=shard_count,
        shard_index=shard_index,
        start_at=start_at,
        end_at=end_at,
        file_timeout_seconds=file_timeout_seconds,
        continue_on_failure=continue_on_failure,
    )
    state_path = _state_path(state_root)
    if resume:
        state = read_state(state_root)
        if state.get("identity") != identity:
            raise T3ShardRunnerError("HEAD、inventory、selection 或执行参数已漂移")
        if state.get("inventory") != inventory:
            raise T3ShardRunnerError("checkpoint 的逐文件 inventory 已漂移")
        if state.get("selected_files") != list(selected_files):
            raise T3ShardRunnerError("checkpoint 的选中文件序已漂移")
        return state
    if state_path.exists():
        raise T3ShardRunnerError("checkpoint 已存在；恢复必须显式使用 --resume")
    state = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "inventory": inventory,
        "selected_files": list(selected_files),
        "results": {},
        "aggregate_status": "PENDING",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
    }
    write_state(state_root, state)
    return state
