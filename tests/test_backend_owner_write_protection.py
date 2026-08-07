"""PW-00A Core owner 主动写保护的双后端边界测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    CoreOwnerWriteProtectionError,
    DictBackend,
    SQLiteBackend,
    TYPE_INT,
)


_OWNER_COLUMNS = [
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("value", TYPE_INT),
]
_GLOBAL_COLUMNS = [("identity_id", TYPE_INT), ("value", TYPE_INT)]


def _backend(backend_type):
    """注册一个共享 owner 表和一个无 owner 的全局设施表。"""
    backend = backend_type()
    backend.register_table(
        "owner_state",
        _OWNER_COLUMNS,
        disc.DISC_NONE,
        [("space_id", "local_id")],
        core=False,
    )
    backend.register_table(
        "global_identity_state",
        _GLOBAL_COLUMNS,
        disc.DISC_NONE,
        [],
        core=True,
    )
    return backend


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_owner_write_protection_rejects_only_protected_owner(backend_type):
    """受保护 owner 的三种写法均拒绝，Memory owner 与全局身份池仍可增长。"""
    backend = _backend(backend_type)
    try:
        backend.insert(
            "owner_state", {"space_id": 1, "local_id": 1, "value": 10})
        backend.insert(
            "owner_state", {"space_id": 2, "local_id": 1, "value": 20})
        backend.protect_owner_space(1)

        backend.insert(
            "owner_state", {"space_id": 2, "local_id": 2, "value": 21})
        backend.update(
            "owner_state", {"space_id": 2, "local_id": 1}, {"value": 22})
        assert backend.delete(
            "owner_state", {"space_id": 2, "local_id": 2}) == 1
        backend.insert(
            "global_identity_state", {"identity_id": 1, "value": 30})

        with pytest.raises(CoreOwnerWriteProtectionError):
            backend.insert(
                "owner_state", {"space_id": 1, "local_id": 2, "value": 11})
        with pytest.raises(CoreOwnerWriteProtectionError):
            backend.update(
                "owner_state", {"space_id": 1, "local_id": 1}, {"value": 12})
        with pytest.raises(CoreOwnerWriteProtectionError):
            backend.delete(
                "owner_state", {"space_id": 1, "local_id": 1})

        assert backend.select("owner_state", where={"space_id": 1}) == [
            {"space_id": 1, "local_id": 1, "value": 10},
        ]
        assert backend.select("owner_state", where={"space_id": 2}) == [
            {"space_id": 2, "local_id": 1, "value": 22},
        ]
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_owner_write_protection_blocks_owner_transfer(backend_type):
    """更新不能把行移入或移出受保护 owner。"""
    backend = _backend(backend_type)
    try:
        backend.insert(
            "owner_state", {"space_id": 1, "local_id": 1, "value": 10})
        backend.insert(
            "owner_state", {"space_id": 2, "local_id": 1, "value": 20})
        backend.protect_owner_space(1)

        with pytest.raises(CoreOwnerWriteProtectionError):
            backend.update(
                "owner_state", {"space_id": 2}, {"space_id": 1})
        with pytest.raises(CoreOwnerWriteProtectionError):
            backend.update(
                "owner_state", {"space_id": 1}, {"space_id": 2})
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_owner_write_protection_round_trips_with_recovery_state(backend_type):
    """启动事务恢复时，表、ID 水位与写保护集合必须一并回滚。"""
    backend = _backend(backend_type)
    try:
        backend.protect_owner_space(1)
        expected = backend.recovery_state_snapshot()
        backend.protect_owner_space(2)
        backend.insert(
            "owner_state", {"space_id": 3, "local_id": 1, "value": 30})

        backend.restore_recovery_state(expected)

        assert backend.owner_write_protection_state() == (1,)
        assert backend.select("owner_state", where=None) == []
        with pytest.raises(CoreOwnerWriteProtectionError):
            backend.insert(
                "owner_state", {"space_id": 1, "local_id": 1, "value": 10})
    finally:
        backend.close()
