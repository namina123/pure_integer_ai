"""SQLite 有界 case 事务：原生主库页上限与显式成功提交。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pure_integer_ai.storage.backend import SQLiteBackend


# object-model: resource_owner; representation=host; interop=sqlite-case-budget-v1
class SQLiteCaseBudget:
    """持有一次可回滚事务；失败后调用方必须废弃全部相关内存 owner。"""

    __slots__ = ("backend", "maximum", "previous_pages", "actual_bytes", "committed", "entered", "active")

    def __init__(self, backend: SQLiteBackend, *, max_database_bytes: int) -> None:
        """只接受严格整数字节数；不通过宿主异常文本决定语义状态。"""
        if type(max_database_bytes) is not int or max_database_bytes <= 0:
            raise ValueError("database byte limit must be a positive strict integer")
        self.backend, self.maximum = backend, max_database_bytes
        self.previous_pages = self.actual_bytes = 0
        self.committed = self.entered = self.active = False

    def __enter__(self) -> SQLiteCaseBudget:
        """要求已有完整 checkpoint 与 durable journal，再开启独占 case 事务。"""
        backend = self.backend
        backend._require_writable()
        connection = backend._conn
        journal = connection.execute("PRAGMA journal_mode").fetchone()[0]
        if (self.entered or backend._performance_mode != "durable"
                or journal.lower() != "delete" or connection.in_transaction
                or getattr(backend, "_suppress_commit", False)):
            raise RuntimeError("bounded case requires an idle durable DELETE-journal owner")
        size = connection.execute("PRAGMA page_size").fetchone()[0]
        count = connection.execute("PRAGMA page_count").fetchone()[0]
        pages = self.maximum // size
        if not 1 <= pages <= 4294967294 or count > pages:
            raise ValueError("database already exceeds the main-file page budget")
        self.previous_pages = connection.execute("PRAGMA max_page_count").fetchone()[0]
        actual = connection.execute(f"PRAGMA max_page_count={pages}").fetchone()[0]
        if actual != pages:
            connection.execute(f"PRAGMA max_page_count={self.previous_pages}")
            raise RuntimeError("SQLite did not install the exact main-file page cap")
        self.actual_bytes = pages * size
        backend._suppress_commit = True
        try:
            connection.execute("BEGIN IMMEDIATE")
        except BaseException:
            backend._suppress_commit = False
            connection.execute(f"PRAGMA max_page_count={self.previous_pages}")
            raise
        self.entered = self.active = True
        return self

    def usage(self) -> dict[str, int | str]:
        """读取当前事务逻辑页数；磁盘 journal 必须由宿主另行计入。"""
        if not self.active:
            raise RuntimeError("SQLite case budget is not active")
        connection = self.backend._conn
        size = connection.execute("PRAGMA page_size").fetchone()[0]
        count = connection.execute("PRAGMA page_count").fetchone()[0]
        return {"page_size": size, "page_count": count, "logical_database_bytes": size * count,
                "hard_main_database_bytes": self.actual_bytes,
                "hard_bound_scope": "MAIN_DATABASE_ONLY_EXCLUDES_JOURNAL_AND_TEMP"}

    def commit(self) -> None:
        """仅由完整 case 成功边界调用；中途 backend.commit 一直被抑制。"""
        if not self.active or self.committed or not self.backend._conn.in_transaction:
            raise RuntimeError("bounded SQLite case cannot commit twice or outside its scope")
        self.backend._suppress_commit = False
        try:
            self.backend.commit()
        except BaseException as error:
            try:
                self.backend._conn.rollback()
            except BaseException as cleanup_error:
                error.add_note("SQLite rollback failed: " + type(cleanup_error).__name__)
            raise
        finally:
            self.backend._suppress_commit = True
        self.committed = True

    def __exit__(self, error_type, error, traceback) -> None:
        """未明确提交必回滚；恢复连接物理上限不构成业务重启。"""
        self.backend._suppress_commit = False
        self.active = False
        try:
            if not self.committed:
                self.backend._conn.rollback()
        except BaseException as cleanup_error:
            if error is None:
                raise
            error.add_note("SQLite rollback failed: " + type(cleanup_error).__name__)
        finally:
            try:
                self.backend._conn.execute(f"PRAGMA max_page_count={self.previous_pages}")
            except BaseException as cleanup_error:
                if error is None:
                    raise
                error.add_note("SQLite page-cap restore failed: " + type(cleanup_error).__name__)
