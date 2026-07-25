"""Scope/Clock/Assertion 与 storage 整数身份索引之间的薄桥。

领域对象仍由 cognition 拥有；storage 只看到整数键。删除 registry 后不会删除图内本体，
而删除图内本体也不能靠 registry 伪装为仍可推理。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.cognition.shared.scope_identity import (
    AssertionIdentity,
    LogicalClock,
    LogicalClockIdentity,
    LogicalTimestamp,
    ScopeIdentity,
    concept_assertion,
)
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_ASSERTION,
    IDENTITY_CLOCK,
    IDENTITY_SCOPE,
    IDENTITY_TIMESTAMP,
    IdentityCollisionError,
    IntegerHasher,
    IntegerIdentityRegistry,
)
from pure_integer_ai.storage.assertion_record import (
    ASSERTION_ROLE_GENERIC,
    AssertionRecord,
    AssertionRecordStore,
)
from pure_integer_ai.storage.telemetry import telemetry_scope_if_active
from pure_integer_ai.storage.backend import StorageBackend


class ScopedIdentityStore:
    """按领域类型递归登记、恢复和 supersede 完整身份。"""

    def __init__(self, backend: StorageBackend, *,
                 hasher: IntegerHasher | None = None) -> None:
        """绑定后端并初始化 scope、时钟、断言及其恢复缓存。"""
        self._backend = backend
        self._registry = IntegerIdentityRegistry(backend, hasher=hasher)
        self._assertion_records = AssertionRecordStore(backend)
        self._scope_hashes: dict[ScopeIdentity, int] = {}
        self._clock_hashes: dict[LogicalClockIdentity, int] = {}
        self._timestamp_hashes: dict[LogicalTimestamp, int] = {}
        self._assertion_hashes: dict[tuple[int, ...], int] = {}
        self._assertion_roles_by_hash: dict[int, int] = {}
        self._fresh_assertion_namespace: bool | None = None
        self._scopes_by_hash: dict[int, ScopeIdentity] = {}
        self._clocks_by_hash: dict[int, LogicalClockIdentity] = {}
        self._timestamps_by_hash: dict[int, LogicalTimestamp] = {}
        self._assertions_by_hash: dict[int, AssertionIdentity] = {}

    @property
    def registry(self) -> IntegerIdentityRegistry:
        """暴露通用索引供完整性审计，不授予领域本体职责。"""
        return self._registry

    @property
    def backend(self) -> StorageBackend:
        """返回 scope identity store 绑定的存储后端。"""
        return self._backend

    @property
    def assertion_records(self) -> AssertionRecordStore:
        """暴露共享正规化记录 facade，供图投影复用同一核验缓存。"""
        return self._assertion_records

    def register_scope(self, scope: ScopeIdentity) -> int:
        """递归登记 parent 后登记 scope。"""
        cached = self._scope_hashes.get(scope)
        if cached is not None:
            return cached
        parent_hash = 0
        if scope.parent is not None:
            parent_hash = self.register_scope(scope.parent)
        identity_hash = self._registry.register(
            IDENTITY_SCOPE, scope.stable_key(), parent_hash=parent_hash)
        self._scope_hashes[scope] = identity_hash
        self._scopes_by_hash[identity_hash] = scope
        return identity_hash

    def load_scope(self, identity_hash: int) -> ScopeIdentity:
        """从完整整数键恢复 scope。"""
        cached = self._scopes_by_hash.get(identity_hash)
        if cached is not None:
            return cached
        scope = ScopeIdentity.from_stable_key(
            self._registry.read_key(IDENTITY_SCOPE, identity_hash))
        self._scope_hashes[scope] = identity_hash
        self._scopes_by_hash[identity_hash] = scope
        return scope

    def register_clock(self, clock: LogicalClockIdentity) -> int:
        """登记时钟并把 owning scope 作为 parent 索引。"""
        cached = self._clock_hashes.get(clock)
        if cached is not None:
            return cached
        scope_hash = self.register_scope(clock.scope)
        identity_hash = self._registry.register(
            IDENTITY_CLOCK, clock.stable_key(), parent_hash=scope_hash)
        self._clock_hashes[clock] = identity_hash
        self._clocks_by_hash[identity_hash] = clock
        return identity_hash

    def load_clock(self, identity_hash: int) -> LogicalClockIdentity:
        """从完整整数键恢复逻辑时钟身份。"""
        cached = self._clocks_by_hash.get(identity_hash)
        if cached is not None:
            return cached
        clock = LogicalClockIdentity.from_stable_key(
            self._registry.read_key(IDENTITY_CLOCK, identity_hash))
        self._clock_hashes[clock] = identity_hash
        self._clocks_by_hash[identity_hash] = clock
        return clock

    def register_timestamp(self, timestamp: LogicalTimestamp) -> int:
        """登记时间戳，并以 clock 和 seq 建恢复索引。"""
        cached = self._timestamp_hashes.get(timestamp)
        if cached is not None:
            return cached
        clock_hash = self.register_clock(timestamp.clock)
        identity_hash = self._registry.register(
            IDENTITY_TIMESTAMP,
            timestamp.stable_key(),
            parent_hash=clock_hash,
            ordinal=timestamp.seq,
        )
        self._timestamp_hashes[timestamp] = identity_hash
        self._timestamps_by_hash[identity_hash] = timestamp
        return identity_hash

    def load_timestamp(self, identity_hash: int) -> LogicalTimestamp:
        """从完整整数键恢复逻辑时间戳。"""
        cached = self._timestamps_by_hash.get(identity_hash)
        if cached is not None:
            return cached
        timestamp = LogicalTimestamp.from_stable_key(
            self._registry.read_key(IDENTITY_TIMESTAMP, identity_hash))
        self._timestamp_hashes[timestamp] = identity_hash
        self._timestamps_by_hash[identity_hash] = timestamp
        return timestamp

    def resume_clock(self, identity: LogicalClockIdentity) -> LogicalClock:
        """从同一 clock identity 的最大已登记 seq 恢复，不跨 scope 串值。"""
        clock_hash = self.register_clock(identity)
        current_seq = self._registry.max_ordinal(
            IDENTITY_TIMESTAMP, clock_hash)
        return LogicalClock(identity, current_seq)

    def register_assertion(
            self, assertion: AssertionIdentity, *,
            assertion_role: int = ASSERTION_ROLE_GENERIC) -> int:
        """按显式物理角色登记断言，并把完整 scope 作为 parent 索引。"""
        if assertion_role not in {0, 1} or type(assertion_role) is not int:
            raise ValueError("assertion_role 未注册")
        assertion_key = assertion.stable_key()
        cached = self._assertion_hashes.get(assertion_key)
        if cached is not None:
            if self._assertion_roles_by_hash[cached] != assertion_role:
                raise ValueError("同一 assertion 不得跨物理角色重复登记")
            cached_assertion = self._assertions_by_hash.get(cached)
            if cached_assertion is not None and cached_assertion != assertion:
                raise IdentityCollisionError("完整稳定键命中不同 assertion 对象")
            return cached
        with telemetry_scope_if_active(
                assertion_key=assertion_key,
                scope_key=assertion.scope.stable_key(),
                query="assertion.register"):
            scope_hash = self.register_scope(assertion.scope)
            subject_key = assertion.subject.stable_key()
            object_key = assertion.object.stable_key()

            def build_record(identity_hash: int) -> AssertionRecord:
                """把领域断言拆为固定端点、scope 引用和有序限定项。"""
                return AssertionRecord(
                    identity_hash=identity_hash,
                    assertion_role=assertion_role,
                    key_version=assertion_key[0],
                    relation_kind=assertion.relation_kind,
                    subject_key=subject_key,
                    object_key=object_key,
                    scope_hash=scope_hash,
                    provenance_kind=assertion.provenance_kind,
                    epistemic_origin=assertion.epistemic_origin,
                    content_version=assertion.content_version,
                    qualifiers=assertion.qualifiers,
                )

            if self._fresh_assertion_namespace is None:
                self._fresh_assertion_namespace = (
                    self._registry.assertion_namespace_is_empty())
            if self._fresh_assertion_namespace:
                def append_record(new_hash: int) -> None:
                    """在已核验空命名空间内直接追加一条新正规化记录。"""
                    self._assertion_records.append_new(build_record(new_hash))

                try:
                    identity_hash = (
                        self._registry.append_new_resolved_in_empty_namespace(
                            IDENTITY_ASSERTION,
                            assertion_key,
                            parent_hash=scope_hash,
                            ordinal=assertion_role,
                            writer=append_record,
                        ))
                except BaseException:
                    self._fresh_assertion_namespace = False
                    raise
            else:
                def write_record(existing_hash: int) -> None:
                    """在恢复或既有命名空间中执行逐项读回的严格登记。"""
                    self._assertion_records.register(
                        build_record(existing_hash))

                identity_hash = self._registry.register_resolved(
                    IDENTITY_ASSERTION,
                    assertion_key,
                    parent_hash=scope_hash,
                    ordinal=assertion_role,
                    writer=write_record,
                )
            self._assertion_roles_by_hash[identity_hash] = assertion_role
            self._assertion_hashes[assertion_key] = identity_hash
            self._assertions_by_hash[identity_hash] = assertion
            return identity_hash

    def load_assertion(self, identity_hash: int) -> AssertionIdentity:
        """从完整整数键恢复断言。"""
        cached = self._assertions_by_hash.get(identity_hash)
        if cached is not None:
            return cached
        assertion_key = self._registry.read_key(
            IDENTITY_ASSERTION, identity_hash)
        assertion = AssertionIdentity.from_stable_key(assertion_key)
        record = self._assertion_records.read(identity_hash)
        self._assertion_hashes[assertion_key] = identity_hash
        self._assertion_roles_by_hash[identity_hash] = record.assertion_role
        self._assertions_by_hash[identity_hash] = assertion
        return assertion

    def clear_runtime_caches(self) -> None:
        """外部 load 或完整性审计前清空运行期身份缓存。"""
        self._scope_hashes.clear()
        self._clock_hashes.clear()
        self._timestamp_hashes.clear()
        self._assertion_hashes.clear()
        self._assertion_roles_by_hash.clear()
        self._assertion_records.clear_runtime_caches()
        self._fresh_assertion_namespace = None
        self._scopes_by_hash.clear()
        self._clocks_by_hash.clear()
        self._timestamps_by_hash.clear()
        self._assertions_by_hash.clear()

    def supersede(self, old: AssertionIdentity,
                  new: AssertionIdentity,
                  timestamp: LogicalTimestamp) -> int:
        """登记旧、新断言和时间戳，再追加不可变替代事件。"""
        old_hash = self._register_existing_or_generic(old)
        new_hash = self._register_existing_or_generic(new)
        timestamp_hash = self.register_timestamp(timestamp)
        return self._registry.append_supersede(
            old_hash, new_hash, timestamp_hash)

    def _register_existing_or_generic(self, assertion: AssertionIdentity) -> int:
        """替代事件优先保持既有物理角色，仅为全新断言使用 generic。"""
        assertion_key = assertion.stable_key()
        cached = self._assertion_hashes.get(assertion_key)
        if cached is not None:
            return cached
        existing = self._registry.find(
            IDENTITY_ASSERTION,
            assertion_key,
        )
        if existing is None:
            return self.register_assertion(assertion)
        restored = self.load_assertion(existing)
        if restored != assertion:
            raise ValueError("既有 assertion hash 与完整键不一致")
        return existing

    def superseding_events(self, assertion_hash: int
                           ) -> tuple[dict[str, int], ...]:
        """读取并核验一个断言的全部 append-only 替代事件。"""
        return self._registry.superseding_events(assertion_hash)

    def assertion_is_superseded(self, assertion_hash: int) -> bool:
        """判断断言是否已有替代事件，不删除或隐藏历史记录本体。"""
        return bool(self.superseding_events(assertion_hash))

    def assertion_from_legacy_edge(
            self, *, scope: ScopeIdentity,
            subject_ref: tuple[int, int],
            object_ref: tuple[int, int],
            relation_kind: int,
            qualifiers: tuple[int, ...] = ()) -> AssertionIdentity | None:
        """把唯一旧宽边转换为候选断言；多行歧义由 storage 层拒绝。"""
        row = self._registry.select_unique_legacy_edge(
            space_id_from=subject_ref[0],
            local_id_from=subject_ref[1],
            space_id_to=object_ref[0],
            local_id_to=object_ref[1],
            edge_type=relation_kind,
            scope_key=scope.stable_key(),
        )
        if row is None:
            return None
        epistemic = row.get("epistemic_origin")
        return concept_assertion(
            relation_kind,
            subject_ref,
            object_ref,
            scope=scope,
            provenance_kind=row["source"],
            epistemic_origin=0 if epistemic is None else epistemic,
            content_version=row.get("content_version") or 0,
            qualifiers=qualifiers,
        )


__all__ = ["ScopedIdentityStore"]
