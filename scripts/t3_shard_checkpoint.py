"""持久化并恢复文件隔离 T3 的 canonical checkpoint。"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any

from .t3_shard_contract import (
    RUNNER_CONTRACT,
    T3ShardRunnerError,
    build_identity,
    build_inventory,
    canonical_bytes,
    require_clean_repository,
    run_git,
    select_files,
    sha256_bytes,
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


def _state_sha256(state_root: Path) -> str:
    """返回来源 checkpoint 的字节身份。"""
    return hashlib.sha256(_state_path(state_root).read_bytes()).hexdigest()


def _changed_paths(repository_root: Path, source_head: str) -> tuple[str, ...]:
    """读取来源 HEAD 到当前 HEAD 的 canonical changed-path 集合。"""
    if not isinstance(source_head, str) or not re.fullmatch(
        r"[0-9a-f]{40,64}", source_head
    ):
        raise T3ShardRunnerError("来源 HEAD 身份无效")
    merge_base = run_git(
        repository_root, "merge-base", source_head, "HEAD"
    ).decode("ascii").strip()
    if merge_base != source_head:
        raise T3ShardRunnerError("来源 HEAD 不是当前 HEAD 的祖先")
    payload = run_git(
        repository_root,
        "diff",
        "--name-only",
        "-z",
        source_head,
        "HEAD",
        "--",
    )
    values = tuple(sorted(item for item in payload.decode("utf-8").split("\0") if item))
    return values


def _module_paths(repository_root: Path, module: str) -> tuple[str, ...]:
    """把 tests 命名空间模块转换为仓库内现存 Python 路径。"""
    if module != "tests" and not module.startswith("tests."):
        return ()
    relative = PurePosixPath(*module.split("."))
    candidates = (
        relative.with_suffix(".py"),
        relative / "__init__.py",
    )
    return tuple(
        candidate.as_posix()
        for candidate in candidates
        if (repository_root / Path(*candidate.parts)).is_file()
    )


def _imported_modules(current: str, tree: ast.AST) -> set[str]:
    """解析绝对和相对 import，并保守加入 from-import 子模块候选。"""
    current_module = list(PurePosixPath(current).with_suffix("").parts)
    package = current_module[:-1]
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            keep = len(package) - (node.level - 1)
            if keep < 1:
                continue
            prefix = package[:keep]
            if node.module:
                prefix.extend(node.module.split("."))
            module = ".".join(prefix)
        else:
            module = node.module or ""
        if module:
            modules.add(module)
            modules.update(
                f"{module}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )
    return modules


def _test_dependency_closure(repository_root: Path, relative: str) -> set[str]:
    """解析测试文件直接/递归引用的 tests.* 辅助模块。"""
    pending = [relative]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        path = repository_root / Path(*PurePosixPath(current).parts)
        if not path.is_file() or path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=current)
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            raise T3ShardRunnerError(
                f"无法解析 PASS 测试依赖: {current}") from error
        for module in sorted(_imported_modules(current, tree)):
            for dependency in _module_paths(repository_root, module):
                if dependency not in seen:
                    pending.append(dependency)
    return seen


def _carryable_changed_paths(
    changed_paths: tuple[str, ...],
) -> None:
    """检查后继改动不会改变 pytest 全局环境或被继承测试的依赖。"""
    forbidden = {
        "conftest.py",
        "tests/conftest.py",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
    }
    allowed_runner_prefixes = (
        "scripts/run_t3_file_shards.py",
        "scripts/t3_shard_",
        "tests/test_t3_shard_runner.py",
    )
    for changed in changed_paths:
        changed_path = PurePosixPath(changed)
        if (
            changed in forbidden
            or changed_path.name == "conftest.py"
            or changed.startswith("src/")
            or changed.startswith("data/")
        ):
            raise T3ShardRunnerError(
                f"后继 PASS 继承遇到全局/生产改动: {changed}")
        if changed.startswith(allowed_runner_prefixes):
            continue
        if not changed.startswith("tests/") or not changed.endswith(".py"):
            raise T3ShardRunnerError(f"后继 PASS 继承遇到未授权改动: {changed}")


def _validate_source_state(source: dict[str, Any]) -> None:
    """重算来源 checkpoint 的 inventory、selection 与 runner 身份。"""
    identity = source.get("identity")
    inventory = source.get("inventory")
    selected = source.get("selected_files")
    if not isinstance(identity, dict) or identity.get("contract") != RUNNER_CONTRACT:
        raise T3ShardRunnerError("来源 checkpoint runner contract 无效")
    if not isinstance(inventory, dict) or not isinstance(inventory.get("files"), list):
        raise T3ShardRunnerError("来源 checkpoint inventory 无效")
    files = inventory["files"]
    expected_inventory_sha = sha256_bytes(canonical_bytes({"files": files}))
    if (
        inventory.get("file_count") != len(files)
        or inventory.get("sha256") != expected_inventory_sha
        or identity.get("inventory_file_count") != len(files)
        or identity.get("inventory_sha256") != expected_inventory_sha
    ):
        raise T3ShardRunnerError("来源 checkpoint inventory 身份不闭合")
    if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
        raise T3ShardRunnerError("来源 checkpoint selection 无效")
    try:
        expected_selection = select_files(
            inventory,
            shard_count=identity["shard_count"],
            shard_index=identity["shard_index"],
            start_at=identity["start_at"],
            end_at=identity["end_at"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise T3ShardRunnerError("来源 checkpoint selection 参数无效") from error
    selection_sha = sha256_bytes(canonical_bytes({"files": selected}))
    if (
        selected != list(expected_selection)
        or identity.get("selected_file_count") != len(selected)
        or identity.get("selection_sha256") != selection_sha
    ):
        raise T3ShardRunnerError("来源 checkpoint selection 身份不闭合")


def _source_log_payload(
    source_root: Path,
    relative: str,
    *,
    source_head: str,
    test_file: str,
    attempt: dict[str, Any],
    expected_attempt: int | None = None,
) -> bytes:
    """严格回读来源 PASS 日志，并拒绝路径逃逸或伪造 header。"""
    if "\\" in relative or ":" in relative:
        raise T3ShardRunnerError(f"来源 PASS 日志路径无效: {test_file}")
    log_relative = PurePosixPath(relative)
    if (
        log_relative.is_absolute()
        or not log_relative.parts
        or log_relative.parts[0] != "logs"
        or any(part in {"", ".", ".."} for part in log_relative.parts)
    ):
        raise T3ShardRunnerError(f"来源 PASS 日志路径无效: {test_file}")
    log_path = (source_root / Path(*log_relative.parts)).resolve()
    try:
        log_path.relative_to(source_root)
    except ValueError as error:
        raise T3ShardRunnerError(f"来源 PASS 日志逃逸: {test_file}") from error
    if not log_path.is_file():
        raise T3ShardRunnerError(f"来源 PASS 日志缺失: {test_file}")
    payload = log_path.read_bytes()
    header_line, separator, _ = payload.partition(b"\n")
    try:
        header = json.loads(header_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise T3ShardRunnerError(f"来源 PASS 日志 header 无效: {test_file}") from error
    if (
        separator != b"\n"
        or not isinstance(header, dict)
        or canonical_bytes(header) != header_line + b"\n"
        or header.get("contract") != RUNNER_CONTRACT
        or header.get("head") != source_head
        or header.get("test_file") != test_file
        or (
            expected_attempt is not None
            and header.get("attempt") != expected_attempt
        )
        or attempt.get("return_code") != 0
        or not isinstance(attempt.get("pytest_summary"), str)
        or attempt["pytest_summary"].encode("utf-8") not in payload
    ):
        raise T3ShardRunnerError(f"来源 PASS 日志证明不闭合: {test_file}")
    return payload


def _import_carried_passes(
    repository_root: Path,
    state: dict[str, Any],
    source_roots: tuple[Path, ...],
) -> None:
    """把满足祖先、字节和依赖条件的 PASS 作为可审计 provenance 导入新 state。"""
    if not source_roots:
        return
    current_head = state["identity"]["head"]
    selected = tuple(state["selected_files"])
    by_path = {item["path"]: item for item in state["inventory"]["files"]}
    imported: set[str] = set()
    for source_root in source_roots:
        source_root = source_root.resolve()
        _require_external_state_root(repository_root, source_root)
        source = read_state(source_root)
        _validate_source_state(source)
        source_head = source["identity"]["head"]
        if source_head == current_head:
            raise T3ShardRunnerError("后继 PASS 来源必须是较早 HEAD")
        changed = _changed_paths(repository_root, source_head)
        _carryable_changed_paths(changed)
        relay_source_state_sha = _state_sha256(source_root)
        source_files = {item["path"]: item for item in source["inventory"]["files"]}
        for relative in selected:
            if relative in imported or relative not in source_files:
                continue
            if source_files[relative] != by_path[relative]:
                continue
            if set(changed) & _test_dependency_closure(repository_root, relative):
                continue
            result = source.get("results", {}).get(relative, {})
            if result.get("status") != "PASS":
                continue
            attempts = result.get("attempts", [])
            carried = result.get("carried_pass")
            relay_root: Path | None = None
            relay_sha: str | None = None
            if isinstance(attempts, list) and attempts:
                latest = attempts[-1]
                if not isinstance(latest, dict) or latest.get("status") != "PASS":
                    raise T3ShardRunnerError(f"来源 PASS attempt 不闭合: {relative}")
                provenance_root = source_root
                provenance_head = source_head
                log_relative = latest.get("log_path")
                if not isinstance(log_relative, str):
                    raise T3ShardRunnerError(f"来源 PASS 缺少日志: {relative}")
                expected_attempt = latest.get("attempt")
                expected_log_sha = None
                expected_log_size = None
                source_test_sha = source_files[relative]["sha256"]
                provenance_state_sha = relay_source_state_sha
                provenance_summary = latest.get("pytest_summary")
                provenance_changed = changed
            elif isinstance(carried, dict):
                required = {
                    "source_head", "source_state_root", "source_state_sha256",
                    "source_test_sha256", "source_log_path", "source_log_sha256",
                    "source_log_size", "pytest_summary",
                }
                if not required.issubset(carried):
                    raise T3ShardRunnerError(f"来源 carried PASS provenance 不完整: {relative}")
                if not isinstance(carried["source_state_root"], str):
                    raise T3ShardRunnerError(f"来源 carried PASS state 路径无效: {relative}")
                provenance_head = carried["source_head"]
                provenance_root = Path(carried["source_state_root"]).resolve()
                _require_external_state_root(repository_root, provenance_root)
                relay_root = source_root
                relay_sha = relay_source_state_sha
                log_relative = carried["source_log_path"]
                expected_attempt = None
                expected_log_sha = carried["source_log_sha256"]
                expected_log_size = carried["source_log_size"]
                source_test_sha = carried["source_test_sha256"]
                provenance_state_sha = carried["source_state_sha256"]
                provenance_summary = carried["pytest_summary"]
                if source_test_sha != source_files[relative]["sha256"]:
                    raise T3ShardRunnerError(f"来源 carried PASS 测试身份不闭合: {relative}")
                if (
                    not isinstance(provenance_state_sha, str)
                    or _state_sha256(provenance_root) != provenance_state_sha
                ):
                    raise T3ShardRunnerError(f"来源 carried PASS state 身份不闭合: {relative}")
                provenance_changed = _changed_paths(repository_root, provenance_head)
                _carryable_changed_paths(provenance_changed)
                if set(provenance_changed) & _test_dependency_closure(
                    repository_root, relative
                ):
                    continue
            else:
                raise T3ShardRunnerError(f"来源 PASS 缺少 attempt 或 provenance: {relative}")
            log_payload = _source_log_payload(
                provenance_root,
                log_relative,
                source_head=provenance_head,
                test_file=relative,
                attempt={"return_code": 0, "pytest_summary": provenance_summary},
                expected_attempt=expected_attempt,
            )
            log_sha = hashlib.sha256(log_payload).hexdigest()
            if (
                expected_log_sha is not None
                and (log_sha != expected_log_sha or len(log_payload) != expected_log_size)
            ):
                raise T3ShardRunnerError(f"来源 PASS 日志身份不闭合: {relative}")
            provenance_changed_sha = sha256_bytes(canonical_bytes({
                "paths": list(provenance_changed),
            }))
            state["results"][relative] = {
                "status": "PASS",
                "attempts": [],
                "carried_pass": {
                    "source_head": provenance_head,
                    "source_state_root": str(provenance_root),
                    "source_state_sha256": provenance_state_sha,
                    "source_test_sha256": source_test_sha,
                    "source_log_path": log_relative,
                    "source_log_sha256": log_sha,
                    "source_log_size": len(log_payload),
                    "changed_path_count": len(provenance_changed),
                    "changed_paths_sha256": provenance_changed_sha,
                    "pytest_summary": provenance_summary,
                    **({
                        "relay_state_root": str(relay_root),
                        "relay_state_sha256": relay_sha,
                    } if relay_root is not None else {}),
                },
            }
            imported.add(relative)
    state["carried_pass_count"] = len(imported)


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
    carry_forward_from: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """冻结新运行或严格核对可恢复运行的全部身份。"""
    repository_root = repository_root.resolve()
    state_root = state_root.resolve()
    if file_timeout_seconds < 1:
        raise T3ShardRunnerError("file_timeout_seconds 必须大于零")
    if resume and carry_forward_from:
        raise T3ShardRunnerError("--resume 不得同时指定 PASS 继承来源")
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
    _import_carried_passes(repository_root, state, carry_forward_from)
    write_state(state_root, state)
    return state
