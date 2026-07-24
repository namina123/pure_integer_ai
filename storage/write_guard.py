"""上下文局部的后端写禁区，阻止只读回调隐藏修改正式状态。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


class RuntimeWriteGuardError(PermissionError):
    """当前调用链声明只读却尝试执行后端写入。"""


_WRITE_BLOCK_DEPTH: ContextVar[int] = ContextVar(
    "pure_integer_ai_write_block_depth",
    default=0,
)


@contextmanager
def forbid_backend_writes() -> Iterator[None]:
    """在当前调用链禁止所有 backend insert/update/delete，支持安全嵌套。"""
    depth = _WRITE_BLOCK_DEPTH.get()
    token = _WRITE_BLOCK_DEPTH.set(depth + 1)
    try:
        yield
    finally:
        _WRITE_BLOCK_DEPTH.reset(token)


def require_write_allowed(table: str, operation: str) -> None:
    """在实际写入前核验当前调用链没有进入只读禁区。"""
    if _WRITE_BLOCK_DEPTH.get() <= 0:
        return
    raise RuntimeWriteGuardError(
        f"只读调用链禁止 {operation} 表 {table!r}")


__all__ = [
    "RuntimeWriteGuardError",
    "forbid_backend_writes",
    "require_write_allowed",
]
