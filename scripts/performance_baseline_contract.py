"""PERF-P0 基线 manifest、身份和 Git 外 checkpoint 合同。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


CONTRACT = "PURE_INTEGER_AI_PERFORMANCE_BASELINE_V1"
SCHEMA_VERSION = 1
STATE_FILE_NAME = "state.json"
IMPLEMENTED_SCENARIOS = (
    "long_input_hierarchy",
    "long_session_checkpoint",
    "long_memory_projection",
    "storage_dict",
    "storage_sqlite",
)
_RUNNER_SOURCE_PATHS = (
    "scripts/performance_baseline_contract.py",
    "scripts/performance_baseline_runner.py",
    "scripts/performance_baseline_worker.py",
)
SCENARIO_DEFINITIONS = {
    "long_input_hierarchy": {
        "category": "long_text",
        "status": "IMPLEMENTED",
        "description": "来源化绝对 Span 层级重组和 digest",
        "source_paths": (
            "src/pure_integer_ai/experiments/long_input_hierarchy.py",
            "src/pure_integer_ai/cognition/shared/identity.py",
            "src/pure_integer_ai/cognition/shared/scope_identity.py",
            "src/pure_integer_ai/storage/integer_codec.py",
        ),
    },
    "long_session_checkpoint": {
        "category": "long_session",
        "status": "IMPLEMENTED",
        "description": "长生成 continuation metadata 的创建和恢复",
        "source_paths": (
            "src/pure_integer_ai/experiments/long_generation_checkpoint.py",
            "src/pure_integer_ai/storage/integer_codec.py",
            "src/pure_integer_ai/storage/segment_repository.py",
        ),
    },
    "long_memory_projection": {
        "category": "long_memory",
        "status": "IMPLEMENTED",
        "description": "Memory 候选冷投影 manifest 的整数编码和恢复",
        "source_paths": (
            "src/pure_integer_ai/experiments/memory_hot_set_runtime.py",
            "src/pure_integer_ai/storage/memory_query_projection.py",
            "src/pure_integer_ai/storage/segment_dependency.py",
            "src/pure_integer_ai/cognition/shared/memory_overlay.py",
        ),
    },
    "storage_dict": {
        "category": "storage",
        "status": "IMPLEMENTED",
        "description": "Dict backend segment 发布、冷读和热读",
        "source_paths": (
            "src/pure_integer_ai/storage/backend.py",
            "src/pure_integer_ai/storage/sealed_segment.py",
            "src/pure_integer_ai/storage/segment_repository.py",
            "src/pure_integer_ai/storage/tiered_segment_store.py",
        ),
    },
    "storage_sqlite": {
        "category": "storage",
        "status": "IMPLEMENTED",
        "description": "SQLite backend segment 发布、重启冷读和热读",
        "source_paths": (
            "src/pure_integer_ai/storage/backend.py",
            "src/pure_integer_ai/storage/sealed_segment.py",
            "src/pure_integer_ai/storage/segment_repository.py",
            "src/pure_integer_ai/storage/tiered_segment_store.py",
        ),
    },
}


class PerformanceBaselineError(RuntimeError):
    """性能基线身份、状态或输出违反合同。"""


def canonical_bytes(value: object) -> bytes:
    """把基线状态编码为 canonical JSON 单行并补一个稳定换行。"""
    return canonical_json_bytes(value) + b"\n"


def sha256_bytes(payload: bytes) -> str:
    """返回字节内容的 SHA-256 小写摘要。"""
    return hashlib.sha256(payload).hexdigest()


def read_head(repository_root: Path) -> str:
    """读取当前 Git HEAD，拒绝无法解析的仓库。"""
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PerformanceBaselineError("无法读取 Git HEAD") from error
    head = completed.stdout.strip()
    if len(head) not in (40, 64) or any(
            char not in "0123456789abcdef" for char in head):
        raise PerformanceBaselineError("Git HEAD 身份无效")
    return head


def require_clean_repository(repository_root: Path) -> None:
    """要求 baseline 在没有未提交差异的公开仓库上开始。"""
    try:
        completed = subprocess.run(
            ("git", "status", "--porcelain=v1"),
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PerformanceBaselineError("无法读取 Git 工作树") from error
    if completed.stdout:
        raise PerformanceBaselineError("性能基线要求 clean Git 工作树")


def _safe_relative_path(relative_path: str) -> PurePosixPath:
    """把源文件路径规范为仓库内 POSIX 路径。"""
    path = PurePosixPath(relative_path)
    if (not relative_path or path.is_absolute()
            or ".." in path.parts or "\\" in relative_path):
        raise PerformanceBaselineError("source path 必须是仓库内 POSIX 路径")
    return path


def file_identity(repository_root: Path, relative_path: str) -> dict[str, object]:
    """读取一个绑定源文件的字节数和 SHA-256。"""
    relative = _safe_relative_path(relative_path)
    path = repository_root.joinpath(*relative.parts)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PerformanceBaselineError(
            f"无法读取基线 source file: {relative_path}") from error
    return {
        "path": relative.as_posix(),
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def scenario_definition(name: str) -> dict[str, object]:
    """返回一个场景的只读定义副本，避免调用方改变全局合同。"""
    definition = SCENARIO_DEFINITIONS.get(name)
    if definition is None:
        raise PerformanceBaselineError(f"未知性能基线场景: {name}")
    return {
        **definition,
        "source_paths": tuple(definition["source_paths"]),
    }


def build_manifest(
        repository_root: Path,
        *,
        scale: int,
        repetitions: int,
        scenarios: tuple[str, ...] = IMPLEMENTED_SCENARIOS,
        ) -> dict[str, object]:
    """冻结 P0 场景、源文件身份和有界规模配置。"""
    if type(scale) is not int or scale < 1:
        raise PerformanceBaselineError("scale 必须为正严格整数")
    if type(repetitions) is not int or repetitions < 1:
        raise PerformanceBaselineError("repetitions 必须为正严格整数")
    normalized = tuple(dict.fromkeys(scenarios))
    if not normalized:
        raise PerformanceBaselineError("至少需要一个性能基线场景")
    entries = []
    for name in normalized:
        definition = scenario_definition(name)
        entries.append({
            "category": definition["category"],
            "description": definition["description"],
            "name": name,
            "source_bindings": tuple(
                file_identity(repository_root, path)
                for path in dict.fromkeys((
                    *_RUNNER_SOURCE_PATHS,
                    *definition["source_paths"],
                ))
            ),
            "status": definition["status"],
        })
    return {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "head": read_head(repository_root),
        "readiness_transition": {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
        },
        "scale": scale,
        "repetitions": repetitions,
        "scenarios": tuple(entries),
    }


def state_path(state_root: Path) -> Path:
    """返回 Git 外 checkpoint 的固定路径。"""
    return state_root / STATE_FILE_NAME


def require_external_state_root(
        repository_root: Path, state_root: Path,
        ) -> None:
    """禁止把可变 checkpoint、日志或 SQLite 写进公开仓库。"""
    repository = repository_root.resolve()
    target = state_root.resolve()
    try:
        target.relative_to(repository)
    except ValueError:
        return
    raise PerformanceBaselineError("state_root 必须位于公开 Git 根之外")


def write_state(state_root: Path, state: dict[str, object]) -> None:
    """以同目录临时文件原子替换写入 canonical checkpoint。"""
    state_root.mkdir(parents=True, exist_ok=True)
    target = state_path(state_root)
    temporary = state_root / f".{STATE_FILE_NAME}.{os.getpid()}.tmp"
    payload = canonical_bytes(state)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def read_state(state_root: Path) -> dict[str, object]:
    """读取并严格验证 canonical 性能 checkpoint。"""
    target = state_path(state_root)
    try:
        payload = target.read_bytes()
        value = parse_canonical_json_bytes(
            payload[:-1] if payload.endswith(b"\n") else payload,
            require_object=True,
        )
    except (OSError, ValueError, TypeError) as error:
        raise PerformanceBaselineError("性能 checkpoint 无法读取") from error
    if payload != canonical_bytes(value):
        raise PerformanceBaselineError("性能 checkpoint 不是 canonical JSON")
    if (not isinstance(value, dict)
            or value.get("contract") != CONTRACT
            or value.get("schema_version") != SCHEMA_VERSION):
        raise PerformanceBaselineError("性能 checkpoint contract 不受支持")
    return value


def build_initial_state(manifest: dict[str, object]) -> dict[str, object]:
    """从冻结 manifest 形成尚未运行的逐场景 checkpoint。"""
    if not isinstance(manifest, dict) or manifest.get("contract") != CONTRACT:
        raise PerformanceBaselineError("baseline manifest contract 无效")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, (list, tuple)):
        raise PerformanceBaselineError("baseline manifest scenarios 无效")
    selected = tuple(item["name"] for item in scenarios)
    return {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest,
        "selected_scenarios": selected,
        "results": {
            name: {"status": "PENDING", "attempts": ()}
            for name in selected
        },
        "aggregate_status": "INCOMPLETE",
    }


__all__ = [
    "CONTRACT",
    "IMPLEMENTED_SCENARIOS",
    "PerformanceBaselineError",
    "SCENARIO_DEFINITIONS",
    "SCHEMA_VERSION",
    "build_initial_state",
    "build_manifest",
    "canonical_bytes",
    "file_identity",
    "read_head",
    "read_state",
    "require_clean_repository",
    "require_external_state_root",
    "scenario_definition",
    "sha256_bytes",
    "state_path",
    "write_state",
]
