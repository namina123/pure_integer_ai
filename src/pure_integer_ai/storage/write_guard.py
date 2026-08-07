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
_WRITE_BLOCKED_TABLE_STACK: ContextVar[tuple[frozenset[str], ...]] = ContextVar(
    "pure_integer_ai_write_blocked_table_stack",
    default=(),
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


@contextmanager
def forbid_backend_table_writes(tables: tuple[str, ...]) -> Iterator[None]:
    """只禁止指定表写入，供大只读物理资产跳过回滚复制。"""
    if (not isinstance(tables, tuple)
            or not tables
            or any(not isinstance(item, str) or not item for item in tables)
            or len(set(tables)) != len(tables)):
        raise ValueError("局部写禁区 tables 必须是非空唯一 str tuple")
    blocked = frozenset(tables)
    stack = _WRITE_BLOCKED_TABLE_STACK.get()
    token = _WRITE_BLOCKED_TABLE_STACK.set((*stack, blocked))
    try:
        yield
    finally:
        _WRITE_BLOCKED_TABLE_STACK.reset(token)


def require_write_allowed(table: str, operation: str) -> None:
    """在实际写入前核验当前调用链没有进入只读禁区。"""
    if (_WRITE_BLOCK_DEPTH.get() <= 0
            and not any(table in blocked
                        for blocked in _WRITE_BLOCKED_TABLE_STACK.get())):
        return
    raise RuntimeWriteGuardError(
        f"只读调用链禁止 {operation} 表 {table!r}")


__all__ = [
    "RuntimeWriteGuardError",
    "forbid_backend_table_writes",
    "forbid_backend_writes",
    "require_write_allowed",
]
