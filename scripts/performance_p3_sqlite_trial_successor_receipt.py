"""发布 PERF-P3 worker 中文说明修正的 append-only 后继 receipt。"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PURE_INTEGER_AI_PERFORMANCE_P3_SQLITE_TRIAL_SUCCESSOR_RECEIPT"
ARTIFACT_VERSION = "PERFORMANCE-P3-SQLITE-TRIAL-20260807-B"
STATUS = "PERFORMANCE_P3_SQLITE_TRIAL_SUCCESSOR_EVIDENCED"
RECEIPT_PATH = "data/ph2/manifests/performance_p3_sqlite_trial_receipt_v2.json"
PARENT_RECEIPT_PATH = (
    "data/ph2/manifests/performance_p3_sqlite_trial_receipt_v1.json"
)
PARENT_RECEIPT_SHA256 = (
    "fe82107dd792ac868564999f5f5dfde130319db81dcbec5a6a3420aeffb3f605"
)
WORKER_PATH = "scripts/performance_p3_sqlite_trial_worker.py"
PARENT_WORKER_SIZE = 11813
PARENT_WORKER_SHA256 = (
    "db4ba584d994882475d1318fdef85dcc5a917bf39cae8eb6491a678905b4b6d4"
)
EXECUTABLE_AST_SHA256 = (
    "86f72aed8ddad65aea3e14a43b3c26afef0b0a8e871de8442bc9eed1d703aa0a"
)
READINESS_TRANSITION = {
    "LANGUAGE_READINESS_REPUBLISHED": 0,
    "PW00A_STARTED": 0,
}


def _relative(value: str, *, where: str) -> str:
    """核验路径是仓库内不可逃逸的 POSIX 相对路径。"""
    path = PurePosixPath(value)
    if (not isinstance(value, str) or not value or path.is_absolute()
            or ".." in path.parts or "\\" in value):
        raise ValueError(f"{where} 相对路径非法")
    return value


def _target(root: Path, relative_path: str) -> Path:
    """解析并限制一个仓库相对文件路径。"""
    _relative(relative_path, where="P3 successor")
    target = (root / Path(*relative_path.split("/"))).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"P3 successor 文件缺失: {relative_path}")
    return target


def _identity(root: Path, relative_path: str) -> tuple[int, str]:
    """返回仓库相对文件的字节尺寸和 SHA-256。"""
    payload = _target(root, relative_path).read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _strip_docstrings(node: ast.AST) -> ast.AST:
    """递归移除 AST 中不参与执行的模块、类和函数说明。"""
    for child in ast.iter_child_nodes(node):
        _strip_docstrings(child)
    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                         ast.AsyncFunctionDef)):
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:]
    return node


def _executable_ast_sha256(payload: bytes) -> str:
    """计算忽略说明文字和源码位置后的可执行 AST 摘要。"""
    try:
        tree = ast.parse(payload.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ValueError("P3 worker 不是合法 UTF-8 Python 源码") from error
    stripped = _strip_docstrings(tree)
    encoded = ast.dump(
        stripped,
        annotate_fields=True,
        include_attributes=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    """把 receipt 编码为带单个末尾换行的规范 JSON 字节。"""
    return canonical_json_bytes(value) + b"\n"


def build_performance_p3_sqlite_trial_successor_receipt(
        repository_root: str | Path,
        ) -> dict[str, Any]:
    """构建只授权中文说明变化的 P3 v2 后继 receipt。"""
    root = Path(repository_root).resolve()
    if _identity(root, PARENT_RECEIPT_PATH)[1] != PARENT_RECEIPT_SHA256:
        raise ValueError("P3 v1 receipt 历史身份漂移")
    current_size, current_sha = _identity(root, WORKER_PATH)
    if current_sha == PARENT_WORKER_SHA256:
        raise ValueError("P3 worker 尚未形成后继源码")
    executable_sha = _executable_ast_sha256(
        _target(root, WORKER_PATH).read_bytes())
    if executable_sha != EXECUTABLE_AST_SHA256:
        raise ValueError("P3 worker 可执行 AST 已变化，不能按说明修正授权")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "format_version": FORMAT_VERSION,
        "parent_receipt": {
            "relative_path": PARENT_RECEIPT_PATH,
            "sha256": PARENT_RECEIPT_SHA256,
            "status": "HISTORICAL_PREDECESSOR",
        },
        "readiness_transition": dict(READINESS_TRANSITION),
        "receipt_relative_path": RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "source_binding": {
            "current_sha256": current_sha,
            "current_size_bytes": current_size,
            "parent_sha256": PARENT_WORKER_SHA256,
            "parent_size_bytes": PARENT_WORKER_SIZE,
            "relative_path": WORKER_PATH,
        },
        "status": STATUS,
        "transformation": {
            "documentation_language": "ZH_CN",
            "executable_ast_changed": 0,
            "executable_ast_sha256": EXECUTABLE_AST_SHA256,
            "external_evidence_rerun": 0,
            "runtime_contract_changed": 0,
        },
    }


def _validate(value: dict[str, Any]) -> None:
    """严格核验 P3 v2 后继 receipt 的固定字段与转移边界。"""
    if set(value) != {
            "artifact_kind", "artifact_version", "format_version",
            "parent_receipt", "readiness_transition", "receipt_relative_path",
            "receipt_self_excluded", "source_binding", "status",
            "transformation"}:
        raise ValueError("P3 v2 receipt 字段不精确")
    if (value["artifact_kind"] != ARTIFACT_KIND
            or value["artifact_version"] != ARTIFACT_VERSION
            or value["format_version"] != FORMAT_VERSION
            or value["receipt_relative_path"] != RECEIPT_PATH
            or value["receipt_self_excluded"] != 1
            or value["status"] != STATUS):
        raise ValueError("P3 v2 receipt 固定身份漂移")
    if value["parent_receipt"] != {
            "relative_path": PARENT_RECEIPT_PATH,
            "sha256": PARENT_RECEIPT_SHA256,
            "status": "HISTORICAL_PREDECESSOR"}:
        raise ValueError("P3 v2 parent receipt 身份漂移")
    if value["readiness_transition"] != READINESS_TRANSITION:
        raise ValueError("P3 v2 receipt 不得转移 readiness")
    binding = value["source_binding"]
    if (not isinstance(binding, dict) or set(binding) != {
            "current_sha256", "current_size_bytes", "parent_sha256",
            "parent_size_bytes", "relative_path"}
            or binding["relative_path"] != WORKER_PATH
            or binding["parent_size_bytes"] != PARENT_WORKER_SIZE
            or binding["parent_sha256"] != PARENT_WORKER_SHA256
            or type(binding["current_size_bytes"]) is not int
            or binding["current_size_bytes"] < 1
            or not isinstance(binding["current_sha256"], str)
            or len(binding["current_sha256"]) != 64):
        raise ValueError("P3 v2 worker binding 漂移")
    if value["transformation"] != {
            "documentation_language": "ZH_CN",
            "executable_ast_changed": 0,
            "executable_ast_sha256": EXECUTABLE_AST_SHA256,
            "external_evidence_rerun": 0,
            "runtime_contract_changed": 0}:
        raise ValueError("P3 v2 transformation 漂移")


def read_performance_p3_sqlite_trial_successor_receipt(
        repository_root: str | Path,
        path: str | Path = RECEIPT_PATH,
        *,
        verify_current_source: bool = True,
        ) -> dict[str, Any]:
    """规范回读 P3 v2 receipt，并可严格核验当前 worker 身份。"""
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("P3 v2 receipt 末尾换行非法")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if _canonical_bytes(value) != payload:
        raise ValueError("P3 v2 receipt 不是规范字节")
    _validate(value)
    if _identity(root, PARENT_RECEIPT_PATH)[1] != PARENT_RECEIPT_SHA256:
        raise ValueError("P3 v1 receipt 当前历史身份漂移")
    if verify_current_source:
        binding = value["source_binding"]
        if _identity(root, WORKER_PATH) != (
                binding["current_size_bytes"], binding["current_sha256"]):
            raise ValueError("P3 v2 worker 当前身份漂移")
        if _executable_ast_sha256(_target(root, WORKER_PATH).read_bytes()) != (
                EXECUTABLE_AST_SHA256):
            raise ValueError("P3 v2 worker 当前可执行 AST 漂移")
    return value


def publish_performance_p3_sqlite_trial_successor_receipt(
        repository_root: str | Path,
        *,
        target: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    """独占发布 P3 v2 receipt，已存在时禁止覆盖。"""
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("P3 v2 receipt 已发布，禁止覆盖")
    value = build_performance_p3_sqlite_trial_successor_receipt(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(_canonical_bytes(value))
    except FileExistsError as error:
        raise ValueError("P3 v2 receipt 已发布，禁止覆盖") from error
    restored = read_performance_p3_sqlite_trial_successor_receipt(
        root, destination)
    if restored != value:
        raise ValueError("P3 v2 receipt 发布回读不一致")
    return restored


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_VERSION",
    "RECEIPT_PATH",
    "STATUS",
    "build_performance_p3_sqlite_trial_successor_receipt",
    "publish_performance_p3_sqlite_trial_successor_receipt",
    "read_performance_p3_sqlite_trial_successor_receipt",
]
