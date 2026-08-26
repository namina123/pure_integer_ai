"""K-00 后端能力契约、一致性 workload 和降级边界。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    DictBackend,
    SQLiteBackend,
    TYPE_INT,
)
from pure_integer_ai.storage.backend_capability import (
    BackendCapabilityError,
    BackendCapabilityProfile,
    BackendCapabilityRequirement,
    BackendCapabilitySupport,
    BackendDeviceBudget,
    CAPABILITY_ATOMIC_MANIFEST_PUBLISH,
    CAPABILITY_DURABLE_COMMIT,
    CAPABILITY_MODE_FALLBACK,
    CAPABILITY_MODE_NATIVE,
    CAPABILITY_MODE_UNSUPPORTED,
    CAPABILITY_PERSISTENCE,
    CAPABILITY_RANGE_SCAN,
    CAPABILITY_RECLAMATION,
    CAPABILITY_SNAPSHOT_EXPORT,
    capability_profile,
    negotiate_backend_capabilities,
)


_COLUMNS = [
    ("owner", TYPE_INT),
    ("scope", TYPE_INT),
    ("record_key", TYPE_INT),
    ("value", TYPE_INT),
]


class _BackendProxy:
    """只转发公开后端行为和 capability 协议，不暴露具体实现类型。"""

    def __init__(self, backend, profile: BackendCapabilityProfile | None = None):
        """绑定被代理后端及可选的显式能力视图。"""
        self._backend = backend
        self._profile = profile or capability_profile(backend)

    def storage_capabilities(self) -> BackendCapabilityProfile:
        """返回代理声明的实例能力，不泄露被代理对象类型。"""
        return self._profile

    def __getattr__(self, name: str):
        """把领域无关 CRUD 调用转发给被代理后端。"""
        return getattr(self._backend, name)


def _register_tables(backend) -> None:
    """注册 conformance workload 使用的自由表和只增冲突表。"""
    backend.register_table(
        "k00_rows",
        _COLUMNS,
        disc.DISC_NONE,
        [("owner", "scope"), ("record_key",)],
    )
    backend.register_table(
        "k00_events",
        [("event_key", TYPE_INT), ("value", TYPE_INT)],
        disc.DISC_APPEND_ONLY,
        [("event_key",)],
    )


def _conformance_workload(backend) -> dict[str, object]:
    """执行同一 CRUD、范围、有序和写纪律 workload 并返回稳定结果。"""
    _register_tables(backend)
    for row in (
            {"owner": 2, "scope": 8, "record_key": 30, "value": 300},
            {"owner": 1, "scope": 8, "record_key": 10, "value": 100},
            {"owner": 1, "scope": 9, "record_key": 20, "value": 200},
            ):
        backend.insert("k00_rows", row)
    ordered = backend.select("k00_rows", order_by="record_key")
    ranged = backend.select(
        "k00_rows",
        where_gt={"record_key": 10},
        order_by="record_key",
        limit=2,
    )
    updated = backend.update(
        "k00_rows",
        {"owner": 1, "scope": 8},
        {"value": ("+=", 7)},
    )
    deleted = backend.delete("k00_rows", {"record_key": 20})
    backend.insert("k00_events", {"event_key": 1, "value": 1})
    failures = []
    for operation in (
            lambda: backend.update(
                "k00_events", {"event_key": 1}, {"value": 2}),
            lambda: backend.delete("k00_events", {"event_key": 1}),
            ):
        with pytest.raises(disc.AppendOnlyViolation) as caught:
            operation()
        failures.append(type(caught.value).__name__)
    return {
        "ordered": ordered,
        "ranged": ranged,
        "updated": updated,
        "deleted": deleted,
        "remaining": backend.select("k00_rows", order_by="record_key"),
        "count": backend.count("k00_rows"),
        "next_ids": (backend.next_id(4), backend.next_id(4)),
        "failures": tuple(failures),
    }


def _without_capability(
        profile: BackendCapabilityProfile,
        capability: int,
        ) -> BackendCapabilityProfile:
    """复制 profile 并只撤下一项能力，保留其余声明和预算。"""
    supports = tuple(
        BackendCapabilitySupport(
            item.capability,
            (CAPABILITY_MODE_UNSUPPORTED
             if item.capability == capability
             else item.mode),
            item.detail_key,
        )
        for item in profile.capabilities
    )
    return BackendCapabilityProfile(
        (91, capability),
        supports,
        profile.device_budget,
    )


def test_supported_backends_share_complete_conformance_workload(tmp_path: Path):
    """Dict、SQLite 内存和 SQLite 文件必须给出相同领域 CRUD 结果。"""
    backends = (
        DictBackend(),
        SQLiteBackend(),
        SQLiteBackend(str(tmp_path / "k00-conformance.sqlite3")),
    )
    try:
        results = tuple(_conformance_workload(backend) for backend in backends)
        assert results[1:] == results[:1] * 2
    finally:
        for backend in backends:
            backend.close()


def test_sqlite_file_commit_recovers_but_memory_modes_do_not_claim_durability(
        tmp_path: Path,
        ):
    """文件库提交后可重开恢复，内存和临时库不得声明持久或耐久提交。"""
    path = tmp_path / "k00-reopen.sqlite3"
    backend = SQLiteBackend(str(path))
    _register_tables(backend)
    backend.insert(
        "k00_rows",
        {"owner": 1, "scope": 2, "record_key": 3, "value": 4},
    )
    backend.commit()
    profile = capability_profile(backend)
    assert profile.mode(CAPABILITY_PERSISTENCE) == CAPABILITY_MODE_NATIVE
    assert profile.mode(CAPABILITY_DURABLE_COMMIT) == CAPABILITY_MODE_NATIVE
    backend.close()

    reopened = SQLiteBackend(str(path))
    try:
        _register_tables(reopened)
        assert reopened.select("k00_rows") == [
            {"owner": 1, "scope": 2, "record_key": 3, "value": 4},
        ]
    finally:
        reopened.close()

    for memory_backend in (SQLiteBackend(), SQLiteBackend(""), DictBackend()):
        try:
            memory_profile = capability_profile(memory_backend)
            assert memory_profile.mode(
                CAPABILITY_PERSISTENCE) == CAPABILITY_MODE_UNSUPPORTED
            assert memory_profile.mode(
                CAPABILITY_DURABLE_COMMIT) == CAPABILITY_MODE_UNSUPPORTED
        finally:
            memory_backend.close()


def test_sqlite_bulk_mode_is_explicit_and_does_not_claim_durable_commit(
        tmp_path: Path,
        ):
    """可重建训练 bulk 档位降低同步开销，但能力声明诚实降级。"""
    backend = SQLiteBackend(
        str(tmp_path / "k00-bulk.sqlite3"), performance_mode="bulk")
    try:
        assert backend.performance_mode == "bulk"
        profile = capability_profile(backend)
        assert profile.mode(CAPABILITY_PERSISTENCE) == CAPABILITY_MODE_NATIVE
        assert profile.mode(CAPABILITY_DURABLE_COMMIT) == (
            CAPABILITY_MODE_UNSUPPORTED)
    finally:
        backend.close()


def test_sqlite_performance_mode_rejects_unknown_value(tmp_path: Path):
    with pytest.raises(ValueError, match="durable 或 bulk"):
        SQLiteBackend(str(tmp_path / "bad.sqlite3"), performance_mode="fast")


@pytest.mark.parametrize("backend_kind", ("dict", "sqlite"))
def test_integer_increment_updates_known_counter_without_changing_schema(
        tmp_path: Path, backend_kind: str):
    """计数器增量保持整数和后端一致，供批量性能路径复用。"""
    backend = (DictBackend() if backend_kind == "dict" else
               SQLiteBackend(str(tmp_path / "increment.sqlite3")))
    try:
        backend.register_table(
            "increment_counter",
            [("key", TYPE_INT), ("value", TYPE_INT)],
            disc.DISC_MUTABLE_MONOTONE,
            [("key",)],
        )
        backend.insert("increment_counter", {"key": 1, "value": 1})
        assert backend.increment(
            "increment_counter", {"key": 1}, {"value": 2}) == 1
        assert backend.select("increment_counter") == [{"key": 1, "value": 3}]
    finally:
        backend.close()


def test_supported_backends_declare_physical_reclamation():
    """Dict 和 SQLite 均能在逻辑墓碑可见后删除 K-02 非核心物理行。"""
    for backend in (DictBackend(), SQLiteBackend()):
        try:
            assert capability_profile(backend).mode(
                CAPABILITY_RECLAMATION) == CAPABILITY_MODE_NATIVE
        finally:
            backend.close()


@pytest.mark.parametrize("capability", (
    CAPABILITY_SNAPSHOT_EXPORT,
    CAPABILITY_RANGE_SCAN,
    CAPABILITY_ATOMIC_MANIFEST_PUBLISH,
    CAPABILITY_DURABLE_COMMIT,
))
def test_missing_required_capability_fails_closed(capability: int):
    """缺少关键能力且未声明 fallback 时，协商必须拒绝弱一致性。"""
    backend = DictBackend()
    proxy = _BackendProxy(
        backend,
        _without_capability(capability_profile(backend), capability),
    )
    with pytest.raises(BackendCapabilityError):
        negotiate_backend_capabilities(
            proxy,
            (BackendCapabilityRequirement(capability),),
        )


def test_explicit_fallback_is_traced_in_negotiation_report():
    """显式 fallback 可被采用，但结果必须保存其稳定身份且标明非原生。"""
    backend = DictBackend()
    report = negotiate_backend_capabilities(
        backend,
        (BackendCapabilityRequirement(
            CAPABILITY_ATOMIC_MANIFEST_PUBLISH,
            fallback_key=(7, 11, 13),
        ),),
    )
    assert report.capabilities[0].mode == CAPABILITY_MODE_FALLBACK
    assert report.capabilities[0].fallback_key == (7, 11, 13)


def test_proxy_and_profile_change_do_not_change_domain_workload():
    """包装代理或改变预算 profile 只能改变物理协商，不得改变领域结果。"""
    budget = BackendDeviceBudget(
        working_bytes=4096,
        batch_bytes=512,
        concurrent_readers=2,
        concurrent_writers=1,
    )
    direct = DictBackend()
    proxied_backend = DictBackend(device_budget=budget)
    proxy = _BackendProxy(proxied_backend)
    try:
        assert _conformance_workload(proxy) == _conformance_workload(direct)
        assert capability_profile(proxy).device_budget == budget
    finally:
        direct.close()
        proxied_backend.close()


def test_backend_without_capability_protocol_is_rejected():
    """只有 CRUD 而未声明 capability 的旧代理不得被静默推断能力。"""
    class _CrudOnlyProxy:
        def __init__(self, backend):
            self._backend = backend

        def __getattr__(self, name: str):
            return getattr(self._backend, name)

    with pytest.raises(BackendCapabilityError):
        capability_profile(_CrudOnlyProxy(DictBackend()))
