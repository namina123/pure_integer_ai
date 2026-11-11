"""原生主库页上限和完整 case 提交/回滚的真实双连接验证。"""
import sqlite3

import pytest

from pure_integer_ai.storage.backend import SQLiteBackend


def _database(tmp_path, **options):
    """创建一个已提交 checkpoint，独立连接能够验证未提交状态不可见。"""
    path = tmp_path / "memory.sqlite3"
    backend = SQLiteBackend(str(path), **options)
    backend._conn.execute("CREATE TABLE probe (value BLOB)")
    backend._conn.execute("INSERT INTO probe VALUES (?)", (b"checkpoint",))
    backend.commit()
    return path, backend


def test_intermediate_commit_is_invisible_until_complete_case(tmp_path):
    """中途 backend.commit 不泄漏；显式完整提交才对独立 reader 可见。"""
    path, backend = _database(tmp_path)
    try:
        with sqlite3.connect(path) as reader:
            previous = backend._conn.execute("PRAGMA max_page_count").fetchone()[0]
            with backend.bounded_case(max_database_bytes=65536) as budget:
                backend._conn.execute("INSERT INTO probe VALUES (?)", (b"new",))
                backend.commit()
                assert reader.execute("SELECT value FROM probe").fetchall() == [(b"checkpoint",)]
                assert budget.usage()["hard_main_database_bytes"] == 65536
                budget.commit()
                assert reader.execute("SELECT count(*) FROM probe").fetchone()[0] == 2
                with pytest.raises(RuntimeError):
                    budget.commit()
            assert backend._conn.execute("PRAGMA max_page_count").fetchone()[0] == previous
            with pytest.raises(RuntimeError):
                budget.usage()
    finally:
        backend.close()


@pytest.mark.parametrize("failure", ["exception", "not_committed", "sqlite_full"])
def test_failure_restores_checkpoint_and_main_file_limit(tmp_path, failure):
    """异常、无提交和 SQLITE_FULL 都保留旧行，主库实际长度不越页预算。"""
    path, backend = _database(tmp_path)
    previous = backend._conn.execute("PRAGMA max_page_count").fetchone()[0]
    try:
        try:
            with backend.bounded_case(max_database_bytes=16384) as budget:
                backend._conn.execute("INSERT INTO probe VALUES (?)", (b"partial",))
                backend.commit()
                if failure == "exception":
                    raise LookupError("business failure")
                if failure == "sqlite_full":
                    with pytest.raises(sqlite3.OperationalError) as caught:
                        backend._conn.execute("INSERT INTO probe VALUES (?)", (b"x" * 65536,))
                    assert caught.value.sqlite_errorcode == sqlite3.SQLITE_FULL
                    with pytest.raises(RuntimeError):
                        budget.commit()
        except LookupError:
            assert failure == "exception"
        with sqlite3.connect(path) as reader:
            assert reader.execute("SELECT value FROM probe").fetchall() == [(b"checkpoint",)]
        assert path.stat().st_size <= 16384
        assert backend._conn.execute("PRAGMA max_page_count").fetchone()[0] == previous
        with pytest.raises(RuntimeError):
            budget.commit()
    finally:
        backend.close()


@pytest.mark.parametrize("mode", ["bulk", "wal", "transaction", "read_only"])
def test_unsafe_or_uncheckpointed_owner_is_rejected(tmp_path, mode):
    """不允许无 journal、WAL、已有事务或只读连接偷偷进入资源合同。"""
    path, backend = _database(tmp_path, performance_mode="bulk" if mode == "bulk" else "durable")
    if mode == "read_only":
        backend.close()
        backend = SQLiteBackend(str(path), read_only=True)
    try:
        if mode == "wal":
            backend._conn.execute("PRAGMA journal_mode=WAL")
        elif mode == "transaction":
            backend._conn.execute("BEGIN")
        with pytest.raises(RuntimeError):
            with backend.bounded_case(max_database_bytes=65536):
                pytest.fail("unsafe owner entered the bounded transaction")
    finally:
        backend.close()
