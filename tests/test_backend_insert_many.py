"""StorageBackend 有界批量插入的跨后端等价性专项。"""
from __future__ import annotations

import pytest

from pure_integer_ai.crosscut.guards.float_guard import FloatViolation
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    DictBackend,
    SQLiteBackend,
    TYPE_INT,
)


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_insert_many_matches_ordered_single_insert_and_prevalidates(
        backend_type):
    """批量路径须保序、保索引查询，并在非法行前保持整批零写。"""
    backend = backend_type()
    sequential = backend_type()
    try:
        for target in (backend, sequential):
            target.register_table(
                "bulk_rows",
                [("group_id", TYPE_INT), ("ordinal", TYPE_INT)],
                disc.DISC_APPEND_ONLY,
                [("group_id",)],
                core=True,
            )
        rows = tuple(
            {"group_id": ordinal % 3, "ordinal": ordinal}
            for ordinal in range(10_000)
        )
        backend.insert_many("bulk_rows", rows)
        for row in rows:
            sequential.insert("bulk_rows", row)
        assert backend.snapshot() == sequential.snapshot()
        assert backend.select(
            "bulk_rows", {"group_id": 1}) == sequential.select(
                "bulk_rows", {"group_id": 1})

        before = backend.snapshot()
        with pytest.raises(FloatViolation):
            backend.insert_many("bulk_rows", (
                {"group_id": 4, "ordinal": 1},
                {"group_id": 4, "ordinal": 2.5},
            ))
        assert backend.snapshot() == before
    finally:
        sequential.close()
        backend.close()
