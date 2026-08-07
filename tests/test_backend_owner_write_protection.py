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
from pure_integer_ai.storage.write_guard import (
    RuntimeWriteGuardError,
    forbid_backend_table_writes,
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


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_selective_recovery_snapshot_excludes_write_protected_table(
        backend_type):
    """被局部写保护的大表可不复制，其他表仍须精确回滚。"""
    backend = _backend(backend_type)
    try:
        backend.insert(
            "owner_state", {"space_id": 2, "local_id": 1, "value": 20})
        backend.insert(
            "global_identity_state", {"identity_id": 1, "value": 30})
        state = backend.recovery_state_snapshot(
            excluded_tables=("global_identity_state",))

        assert "global_identity_state" not in state["tables"]
        assert state["excluded_table_fences"] == (
            ("global_identity_state", 1, 1),)
        with forbid_backend_table_writes(("global_identity_state",)):
            backend.update(
                "owner_state", {"space_id": 2}, {"value": 21})
            with pytest.raises(RuntimeWriteGuardError):
                backend.insert(
                    "global_identity_state", {"identity_id": 2, "value": 31})

        assert backend.recovery_state_exclusions_unchanged(state)
        backend.restore_recovery_state(state)
        assert backend.select("owner_state", where=None) == [
            {"space_id": 2, "local_id": 1, "value": 20},
        ]
        assert backend.select("global_identity_state", where=None) == [
            {"identity_id": 1, "value": 30},
        ]
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_selective_recovery_rejects_changed_excluded_table(backend_type):
    """调用方漏加写保护时，等行数更新也不得伪造完整回滚。"""
    backend = _backend(backend_type)
    try:
        backend.insert(
            "global_identity_state", {"identity_id": 1, "value": 30})
        state = backend.recovery_state_snapshot(
            excluded_tables=("global_identity_state",))
        backend.update(
            "global_identity_state", {"identity_id": 1}, {"value": 31})

        assert not backend.recovery_state_exclusions_unchanged(state)
        with pytest.raises(RuntimeError, match="排除表"):
            backend.restore_recovery_state(state)
    finally:
        backend.close()
