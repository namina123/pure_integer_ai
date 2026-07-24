"""append-only 纯整数身份索引和 supersede 事件。

storage 层不导入 cognition 领域类型，只保存调用方给出的完整整数稳定键。哈希仅是
查询索引；每次命中都必须回读并逐项核对完整键。显式授权的外部记录可自行承载
header 元数据，其余身份仍要求通用 header。碰撞、重复 header、缺 part、孤儿 part
和冲突 supersede 一律 fail closed。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.assertion_record import (
    ASSERTION_ROLE_GENERIC,
    AssertionRecordCollisionError,
    AssertionRecordIncompleteError,
    AssertionRecordStore,
)
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT


IDENTITY_SCOPE = 1
IDENTITY_CLOCK = 2
IDENTITY_TIMESTAMP = 3
IDENTITY_ASSERTION = 4
IDENTITY_GRAPH_OBJECT = 5
IDENTITY_SOURCE_RECORD = 6
IDENTITY_SPAN_ROLE = 7
IDENTITY_MEMORY_OVERLAY = 8
IDENTITY_MEMORY_OBJECT = 9
IDENTITY_MEMORY_EVENT = 10

IDENTITY_HEADER_TABLE = "identity_header"
IDENTITY_PART_TABLE = "identity_part"
ASSERTION_SUPERSEDE_TABLE = "assertion_supersede"

IDENTITY_HEADER_COLUMNS = [
    ("identity_kind", TYPE_INT),
    ("identity_hash", TYPE_INT),
    ("key_size", TYPE_INT),
    ("parent_hash", TYPE_INT),
    ("ordinal", TYPE_INT),
]
IDENTITY_PART_COLUMNS = [
    ("identity_kind", TYPE_INT),
    ("identity_hash", TYPE_INT),
    ("part_index", TYPE_INT),
    ("part_value", TYPE_INT),
]
ASSERTION_SUPERSEDE_COLUMNS = [
    ("event_hash", TYPE_INT),
    ("old_assertion_hash", TYPE_INT),
    ("new_assertion_hash", TYPE_INT),
    ("timestamp_hash", TYPE_INT),
]


class IntegerHasher(Protocol):
    """身份索引所需的最小确定性哈希协议。"""

    def h63(self, value: Any) -> int: ...


class IdentityRegistryError(RuntimeError):
    """身份 registry 完整性错误基类。"""


class IdentityCollisionError(IdentityRegistryError):
    """同整数索引命中不同完整稳定键。"""


class IdentityIncompleteError(IdentityRegistryError):
    """header/part 半写、重复或缺失。"""


class SupersedeConflictError(IdentityRegistryError):
    """同一旧断言出现竞争替代或事件哈希碰撞。"""


class LegacyAssertionAmbiguity(IdentityRegistryError):
    """旧宽边无法唯一迁移为 scoped assertion。"""


@dataclass(frozen=True)
class IdentityMetadata:
    """身份 header 中可索引但不替代完整键的元数据。"""

    identity_kind: int
    identity_hash: int
    key_size: int
    parent_hash: int
    ordinal: int


@dataclass(frozen=True)
class ExternalIdentityKey:
    """由正规化外部记录恢复的完整键及其 header 约束。"""

    key: tuple[int, ...]
    parent_hash: int | None = None
    ordinal: int | None = None


ExternalKeyResolver = Callable[[int], ExternalIdentityKey | None]


def register_assertion_identity_tables(backend: StorageBackend) -> None:
    """注册三个核心 append-only 身份表。"""
    backend.register_table(
        IDENTITY_HEADER_TABLE,
        IDENTITY_HEADER_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("identity_kind", "identity_hash"),
            ("identity_kind", "parent_hash"),
            ("parent_hash", "ordinal"),
        ],
        core=True,
        recovery_key=("identity_kind", "identity_hash"),
    )
    backend.register_table(
        IDENTITY_PART_TABLE,
        IDENTITY_PART_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("identity_kind", "identity_hash"),
            ("identity_kind", "identity_hash", "part_index"),
        ],
        core=True,
        recovery_key=(
            "identity_kind", "identity_hash", "part_index"),
    )
    backend.register_table(
        ASSERTION_SUPERSEDE_TABLE,
        ASSERTION_SUPERSEDE_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("event_hash",),
            ("old_assertion_hash",),
            ("new_assertion_hash",),
        ],
        core=True,
        recovery_key=("event_hash",),
    )


def _strict_int(value: int, *, where: str,
                nonnegative: bool = False,
                positive: bool = False) -> int:
    """校验 registry 字段是严格整数，并按需限制范围。"""
    if type(value) is not int:
        assert_int(value, _where=where)
        raise ValueError(f"{where} 必须为严格整数")
    if positive and value <= 0:
        raise ValueError(f"{where} 必须为正整数")
    if nonnegative and value < 0:
        raise ValueError(f"{where} 必须为非负整数")
    return value


def _stable_key(key: tuple[int, ...]) -> tuple[int, ...]:
    """校验并返回非空纯整数稳定键。"""
    if not isinstance(key, tuple) or not key:
        raise ValueError("identity key 必须是非空整数元组")
    for index, value in enumerate(key):
        if type(value) is not int:
            assert_int(value, _where=f"identity key[{index}]")
            raise ValueError(f"identity key[{index}] 必须为严格整数")
    return key


class IntegerIdentityRegistry:
    """保存完整整数键并以固定哈希作可核验索引。"""

    def __init__(self, backend: StorageBackend, *,
                 hasher: IntegerHasher | None = None) -> None:
        self._backend = backend
        self._hasher = hasher or Hasher("identity_registry.v1")
        self._event_hasher = hasher or Hasher("assertion_supersede.v1")
        self._identity_hash_cache: dict[
            tuple[int, tuple[int, ...]], int
        ] = {}
        self._assertion_records = AssertionRecordStore(backend)
        self._external_key_resolvers: dict[int, ExternalKeyResolver] = {
            IDENTITY_ASSERTION: self._resolve_assertion_record,
        }
        self._self_headed_external_kinds: set[int] = {
            IDENTITY_ASSERTION,
        }

    def _identity_hash(self, identity_kind: int,
                       key: tuple[int, ...]) -> int:
        """生成非零 63-bit 索引，0 保留给“无 parent”。"""
        cache_key = (identity_kind, key)
        cached = self._identity_hash_cache.get(cache_key)
        if cached is not None:
            return cached
        if type(self._hasher) is Hasher:
            value = self._hasher.h63_tagged_int_tuple(identity_kind, key)
        else:
            value = self._hasher.h63((identity_kind, key))
        _strict_int(value, where="identity hash", nonnegative=True)
        identity_hash = value if value > 0 else 1
        self._identity_hash_cache[cache_key] = identity_hash
        return identity_hash

    def identity_hash(self, identity_kind: int,
                      key: tuple[int, ...]) -> int:
        """公开计算与 register 完全相同的稳定身份 hash，不执行写入。"""
        _strict_int(identity_kind, where="identity_kind", positive=True)
        return self._identity_hash(identity_kind, _stable_key(key))

    def register_external_key_resolver(
            self, identity_kind: int, resolver: ExternalKeyResolver, *,
            self_headed: bool = False) -> None:
        """注册外部键恢复器，并显式声明外部记录能否自行承载 header。"""
        _strict_int(identity_kind, where="identity_kind", positive=True)
        if type(self_headed) is not bool:
            raise ValueError("self_headed 必须为严格布尔值")
        if identity_kind in self._external_key_resolvers:
            raise ValueError(f"identity kind={identity_kind} 已有外部键恢复器")
        self._external_key_resolvers[identity_kind] = resolver
        if self_headed:
            self._self_headed_external_kinds.add(identity_kind)

    def _event_hash(self, old_hash: int, new_hash: int,
                    timestamp_hash: int) -> int:
        """生成非零 supersede 事件索引。"""
        value = self._event_hasher.h63(
            (old_hash, new_hash, timestamp_hash))
        _strict_int(value, where="supersede event hash", nonnegative=True)
        return value if value > 0 else 1

    def register(self, identity_kind: int, key: tuple[int, ...], *,
                 parent_hash: int = 0, ordinal: int = 0) -> int:
        """幂等登记完整键；碰撞和任何既有半写状态均拒绝。"""
        _strict_int(identity_kind, where="identity_kind", positive=True)
        _strict_int(parent_hash, where="parent_hash", nonnegative=True)
        _strict_int(ordinal, where="ordinal", nonnegative=True)
        key = _stable_key(key)
        identity_hash = self._identity_hash(identity_kind, key)
        headers = self._header_rows(identity_kind, identity_hash)
        parts = self._part_rows(identity_kind, identity_hash)
        if headers:
            existing_key = self._read_existing(
                identity_kind, identity_hash, headers, parts)
            if existing_key != key:
                raise IdentityCollisionError(
                    f"identity hash={identity_hash} 命中不同完整键")
            metadata = self._metadata_from_header(headers[0])
            if (metadata.parent_hash != parent_hash
                    or metadata.ordinal != ordinal):
                raise IdentityIncompleteError(
                    "同一完整身份键出现不一致 parent/ordinal")
            return identity_hash
        if parts:
            raise IdentityIncompleteError(
                f"identity hash={identity_hash} 存在孤儿 part")
        if self._external_key(identity_kind, identity_hash) is not None:
            raise IdentityIncompleteError(
                f"identity hash={identity_hash} 存在无 header 的外部记录")

        for index, value in enumerate(key):
            self._backend.insert(IDENTITY_PART_TABLE, {
                "identity_kind": identity_kind,
                "identity_hash": identity_hash,
                "part_index": index,
                "part_value": value,
            })
        self._backend.insert(IDENTITY_HEADER_TABLE, {
            "identity_kind": identity_kind,
            "identity_hash": identity_hash,
            "key_size": len(key),
            "parent_hash": parent_hash,
            "ordinal": ordinal,
        })
        if self.read_key(identity_kind, identity_hash) != key:
            raise IdentityIncompleteError("identity 写后核验失败")
        return identity_hash

    def register_resolved(self, identity_kind: int, key: tuple[int, ...], *,
                          parent_hash: int = 0, ordinal: int = 0,
                          writer: Callable[[int], None]) -> int:
        """登记外部记录承载的完整键，按 kind 配置决定是否另写 header。"""
        _strict_int(identity_kind, where="identity_kind", positive=True)
        _strict_int(parent_hash, where="parent_hash", nonnegative=True)
        _strict_int(ordinal, where="ordinal", nonnegative=True)
        key = _stable_key(key)
        resolver = self._external_key_resolvers.get(identity_kind)
        if resolver is None:
            raise ValueError(f"identity kind={identity_kind} 未注册外部键恢复器")
        identity_hash = self._identity_hash(identity_kind, key)
        headers = self._header_rows(identity_kind, identity_hash)
        parts = self._part_rows(identity_kind, identity_hash)
        if headers:
            existing_key = self._read_existing(
                identity_kind, identity_hash, headers, parts)
            if existing_key != key:
                raise IdentityCollisionError(
                    f"identity hash={identity_hash} 命中不同完整键")
            metadata = self._metadata_from_header(headers[0])
            if (metadata.parent_hash != parent_hash
                    or metadata.ordinal != ordinal):
                raise IdentityIncompleteError(
                    "同一完整身份键出现不一致 parent/ordinal")
            return identity_hash
        if parts:
            raise IdentityIncompleteError(
                f"identity hash={identity_hash} 存在孤儿 part")
        external = self._external_key(identity_kind, identity_hash)
        if external is not None:
            external = self._require_self_headed_external(
                identity_kind, identity_hash, external)
            self._verify_external_registration(
                identity_hash, external, key, parent_hash, ordinal)
            return identity_hash

        try:
            writer(identity_hash)
        except AssertionRecordCollisionError as exc:
            raise IdentityCollisionError(str(exc)) from exc
        except AssertionRecordIncompleteError as exc:
            raise IdentityIncompleteError(str(exc)) from exc
        external = self._external_key(identity_kind, identity_hash)
        if external is None:
            raise IdentityIncompleteError("外部身份 writer 未生成可恢复完整键")
        self._verify_external_registration(
            identity_hash, external, key, parent_hash, ordinal)
        if identity_kind in self._self_headed_external_kinds:
            self._require_self_headed_external(
                identity_kind, identity_hash, external)
            if self.read_key(identity_kind, identity_hash) != key:
                raise IdentityIncompleteError("外部 identity 写后核验失败")
            return identity_hash
        self._backend.insert(IDENTITY_HEADER_TABLE, {
            "identity_kind": identity_kind,
            "identity_hash": identity_hash,
            "key_size": len(key),
            "parent_hash": parent_hash,
            "ordinal": ordinal,
        })
        if self.read_key(identity_kind, identity_hash) != key:
            raise IdentityIncompleteError("外部 identity 写后核验失败")
        return identity_hash

    def assertion_namespace_is_empty(self) -> bool:
        """检查 assertion 的 header、旧 parts 和正规化记录均为空。"""
        return (
            self._backend.count(
                IDENTITY_HEADER_TABLE,
                where={"identity_kind": IDENTITY_ASSERTION}) == 0
            and self._backend.count(
                IDENTITY_PART_TABLE,
                where={"identity_kind": IDENTITY_ASSERTION}) == 0
            and self._assertion_records.is_empty()
        )

    def append_new_resolved_in_empty_namespace(
            self, identity_kind: int, key: tuple[int, ...], *,
            parent_hash: int = 0, ordinal: int = 0,
            writer: Callable[[int], None]) -> int:
        """在空命名空间追加外部记录，仅非 self-headed kind 另写 header。"""
        _strict_int(identity_kind, where="identity_kind", positive=True)
        _strict_int(parent_hash, where="parent_hash", nonnegative=True)
        _strict_int(ordinal, where="ordinal", nonnegative=True)
        key = _stable_key(key)
        if identity_kind not in self._external_key_resolvers:
            raise ValueError(f"identity kind={identity_kind} 未注册外部键恢复器")
        identity_hash = self._identity_hash(identity_kind, key)
        try:
            writer(identity_hash)
        except AssertionRecordCollisionError as exc:
            raise IdentityCollisionError(str(exc)) from exc
        except AssertionRecordIncompleteError as exc:
            raise IdentityIncompleteError(str(exc)) from exc
        if identity_kind in self._self_headed_external_kinds:
            return identity_hash
        self._backend.insert(IDENTITY_HEADER_TABLE, {
            "identity_kind": identity_kind,
            "identity_hash": identity_hash,
            "key_size": len(key),
            "parent_hash": parent_hash,
            "ordinal": ordinal,
        })
        return identity_hash

    def find(self, identity_kind: int,
             key: tuple[int, ...]) -> int | None:
        """只读查找完整身份键；不存在返回空，碰撞或半写状态拒绝。"""
        _strict_int(identity_kind, where="identity_kind", positive=True)
        key = _stable_key(key)
        identity_hash = self._identity_hash(identity_kind, key)
        headers = self._header_rows(identity_kind, identity_hash)
        parts = self._part_rows(identity_kind, identity_hash)
        if not headers:
            if parts:
                raise IdentityIncompleteError(
                    f"identity hash={identity_hash} 存在孤儿 part")
            external = self._external_key(identity_kind, identity_hash)
            if external is None:
                return None
            external = self._require_self_headed_external(
                identity_kind, identity_hash, external)
            if external.key != key:
                raise IdentityCollisionError(
                    f"identity hash={identity_hash} 命中不同完整键")
            return identity_hash
        existing_key = self._read_existing(
            identity_kind, identity_hash, headers, parts)
        if existing_key != key:
            raise IdentityCollisionError(
                f"identity hash={identity_hash} 命中不同完整键")
        return identity_hash

    def read_key(self, identity_kind: int,
                 identity_hash: int) -> tuple[int, ...]:
        """按索引回读完整键，并重新计算哈希防止静默损坏。"""
        _strict_int(identity_kind, where="identity_kind", positive=True)
        _strict_int(identity_hash, where="identity_hash", positive=True)
        headers = self._header_rows(identity_kind, identity_hash)
        parts = self._part_rows(identity_kind, identity_hash)
        if not headers:
            if parts:
                raise IdentityIncompleteError(
                    f"identity hash={identity_hash} 存在孤儿 part")
            external = self._external_key(identity_kind, identity_hash)
            if external is None:
                raise KeyError(
                    f"identity kind={identity_kind} hash={identity_hash} 不存在")
            key = self._require_self_headed_external(
                identity_kind, identity_hash, external).key
        else:
            key = self._read_existing(
                identity_kind, identity_hash, headers, parts)
        expected = self._identity_hash(identity_kind, key)
        if expected != identity_hash:
            raise IdentityCollisionError(
                f"identity hash={identity_hash} 与完整键重算结果不一致")
        return key

    def metadata(self, identity_kind: int,
                 identity_hash: int) -> IdentityMetadata:
        """读取并核验身份 header 元数据。"""
        key = self.read_key(identity_kind, identity_hash)
        headers = self._header_rows(identity_kind, identity_hash)
        if headers:
            return self._metadata_from_header(headers[0])
        external = self._external_key(identity_kind, identity_hash)
        if external is None:
            raise IdentityIncompleteError("self-headed identity 缺少外部记录")
        external = self._require_self_headed_external(
            identity_kind, identity_hash, external)
        return IdentityMetadata(
            identity_kind,
            identity_hash,
            len(key),
            external.parent_hash,
            external.ordinal,
        )

    def max_ordinal(self, identity_kind: int, parent_hash: int) -> int:
        """返回某 parent 下已核验身份的最大序号，无记录时返回 0。"""
        _strict_int(identity_kind, where="identity_kind", positive=True)
        _strict_int(parent_hash, where="parent_hash", positive=True)
        rows = self._backend.select(IDENTITY_HEADER_TABLE, where={
            "identity_kind": identity_kind,
            "parent_hash": parent_hash,
        })
        maximum = 0
        seen: set[int] = set()
        for row in rows:
            identity_hash = row["identity_hash"]
            if identity_hash in seen:
                raise IdentityIncompleteError("parent 查询命中重复 identity header")
            seen.add(identity_hash)
            self.read_key(identity_kind, identity_hash)
            ordinal = row["ordinal"]
            if ordinal > maximum:
                maximum = ordinal
        return maximum

    def append_supersede(self, old_assertion_hash: int,
                         new_assertion_hash: int,
                         timestamp_hash: int) -> int:
        """追加唯一替代事件；同一旧断言不允许竞争分叉。"""
        for label, value in (
                ("old_assertion_hash", old_assertion_hash),
                ("new_assertion_hash", new_assertion_hash),
                ("timestamp_hash", timestamp_hash)):
            _strict_int(value, where=label, positive=True)
        if old_assertion_hash == new_assertion_hash:
            raise SupersedeConflictError("断言不得 supersede 自身")
        self.read_key(IDENTITY_ASSERTION, old_assertion_hash)
        self.read_key(IDENTITY_ASSERTION, new_assertion_hash)
        self.read_key(IDENTITY_TIMESTAMP, timestamp_hash)

        existing_for_old = self._backend.select(
            ASSERTION_SUPERSEDE_TABLE,
            where={"old_assertion_hash": old_assertion_hash},
        )
        for row in existing_for_old:
            if (row["new_assertion_hash"] == new_assertion_hash
                    and row["timestamp_hash"] == timestamp_hash):
                return row["event_hash"]
            raise SupersedeConflictError(
                "同一旧断言已有不同 supersede 目标或时间戳")

        event_hash = self._event_hash(
            old_assertion_hash, new_assertion_hash, timestamp_hash)
        collisions = self._backend.select(
            ASSERTION_SUPERSEDE_TABLE,
            where={"event_hash": event_hash},
        )
        if collisions:
            raise SupersedeConflictError(
                f"supersede event hash={event_hash} 发生碰撞或重复行")
        self._backend.insert(ASSERTION_SUPERSEDE_TABLE, {
            "event_hash": event_hash,
            "old_assertion_hash": old_assertion_hash,
            "new_assertion_hash": new_assertion_hash,
            "timestamp_hash": timestamp_hash,
        })
        return event_hash

    def superseding_events(self, old_assertion_hash: int
                           ) -> tuple[dict[str, int], ...]:
        """读取并核验某旧断言的 append-only 替代事件。"""
        self.read_key(IDENTITY_ASSERTION, old_assertion_hash)
        rows = self._backend.select(
            ASSERTION_SUPERSEDE_TABLE,
            where={"old_assertion_hash": old_assertion_hash},
        )
        events: list[dict[str, int]] = []
        seen: set[int] = set()
        for row in rows:
            event_hash = row["event_hash"]
            if event_hash in seen:
                raise IdentityIncompleteError("supersede 事件重复")
            seen.add(event_hash)
            self.read_key(IDENTITY_ASSERTION, row["new_assertion_hash"])
            self.read_key(IDENTITY_TIMESTAMP, row["timestamp_hash"])
            expected = self._event_hash(
                row["old_assertion_hash"],
                row["new_assertion_hash"],
                row["timestamp_hash"],
            )
            if expected != event_hash:
                raise IdentityCollisionError("supersede 事件完整键核验失败")
            events.append(dict(row))
        return tuple(events)

    def select_unique_legacy_edge(self, *,
                                  space_id_from: int,
                                  local_id_from: int,
                                  space_id_to: int,
                                  local_id_to: int,
                                  edge_type: int,
                                  scope_key: tuple[int, ...]
                                  ) -> dict[str, Any] | None:
        """在调用方显式给出 scope 后读取唯一旧宽边，多行时拒绝猜测。"""
        _stable_key(scope_key)
        for label, value in (
                ("space_id_from", space_id_from),
                ("local_id_from", local_id_from),
                ("space_id_to", space_id_to),
                ("local_id_to", local_id_to),
                ("edge_type", edge_type)):
            _strict_int(value, where=label, positive=True)
        rows = self._backend.select("edge", where={
            "space_id_from": space_id_from,
            "local_id_from": local_id_from,
            "space_id_to": space_id_to,
            "local_id_to": local_id_to,
            "edge_type": edge_type,
        })
        if len(rows) > 1:
            raise LegacyAssertionAmbiguity(
                "旧宽边命中多行；缺 source/parser/scope 唯一性，禁止 limit 1")
        return rows[0] if rows else None

    def _header_rows(self, identity_kind: int,
                     identity_hash: int) -> list[dict[str, Any]]:
        """读取同类同哈希的所有 header，不使用 limit 掩盖重复。"""
        return self._backend.select(IDENTITY_HEADER_TABLE, where={
            "identity_kind": identity_kind,
            "identity_hash": identity_hash,
        })

    def _part_rows(self, identity_kind: int,
                   identity_hash: int) -> list[dict[str, Any]]:
        """按 part_index 读取完整键组成行。"""
        return self._backend.select(
            IDENTITY_PART_TABLE,
            where={
                "identity_kind": identity_kind,
                "identity_hash": identity_hash,
            },
            order_by="part_index",
        )

    def _read_existing(self, identity_kind: int, identity_hash: int,
                       headers: list[dict[str, Any]],
                       parts: list[dict[str, Any]]) -> tuple[int, ...]:
        """从旧 flat parts 或唯一正规化记录重建完整键，禁止静默混读。"""
        if len(headers) != 1:
            raise IdentityIncompleteError(
                f"identity hash={identity_hash} header 数量={len(headers)}")
        metadata = self._metadata_from_header(headers[0])
        if metadata.identity_kind != identity_kind:
            raise IdentityIncompleteError("identity header kind 不一致")
        external = self._external_key(identity_kind, identity_hash)
        if parts and external is not None:
            raise IdentityIncompleteError(
                f"identity hash={identity_hash} 同时存在 flat part 和外部记录")
        if external is not None:
            if len(external.key) != metadata.key_size:
                raise IdentityIncompleteError(
                    f"identity hash={identity_hash} 外部键长度与 key_size 不一致")
            if (external.parent_hash is not None
                    and external.parent_hash != metadata.parent_hash):
                raise IdentityIncompleteError(
                    f"identity hash={identity_hash} 外部 parent_hash 不一致")
            if (external.ordinal is not None
                    and external.ordinal != metadata.ordinal):
                raise IdentityIncompleteError(
                    f"identity hash={identity_hash} 外部 ordinal 不一致")
            return external.key
        if not parts:
            raise IdentityIncompleteError(
                f"identity hash={identity_hash} header 没有完整键载荷")
        if len(parts) != metadata.key_size:
            raise IdentityIncompleteError(
                f"identity hash={identity_hash} part 数量与 key_size 不一致")
        values: list[int] = []
        for expected_index, row in enumerate(parts):
            if row["part_index"] != expected_index:
                raise IdentityIncompleteError(
                    f"identity hash={identity_hash} part_index 不连续")
            value = row["part_value"]
            _strict_int(value, where="identity part value")
            values.append(value)
        return tuple(values)

    def _external_key(self, identity_kind: int,
                      identity_hash: int) -> ExternalIdentityKey | None:
        """调用已注册恢复器，并把物理记录错误统一映射为 registry 完整性错误。"""
        resolver = self._external_key_resolvers.get(identity_kind)
        if resolver is None:
            return None
        try:
            resolved = resolver(identity_hash)
        except AssertionRecordCollisionError as exc:
            raise IdentityCollisionError(str(exc)) from exc
        except AssertionRecordIncompleteError as exc:
            raise IdentityIncompleteError(str(exc)) from exc
        if resolved is None:
            return None
        key = _stable_key(resolved.key)
        if resolved.parent_hash is not None:
            _strict_int(
                resolved.parent_hash, where="external parent_hash",
                nonnegative=True)
        if resolved.ordinal is not None:
            _strict_int(
                resolved.ordinal, where="external ordinal", nonnegative=True)
        return ExternalIdentityKey(
            key, resolved.parent_hash, resolved.ordinal)

    def _require_self_headed_external(
            self, identity_kind: int, identity_hash: int,
            external: ExternalIdentityKey) -> ExternalIdentityKey:
        """核验无通用 header 的外部记录已获授权且能完整恢复 header 元数据。"""
        if identity_kind not in self._self_headed_external_kinds:
            raise IdentityIncompleteError(
                f"identity hash={identity_hash} 存在无 header 的外部记录")
        if external.parent_hash is None or external.ordinal is None:
            raise IdentityIncompleteError(
                f"identity hash={identity_hash} 外部记录不能完整充当 header")
        return external

    @staticmethod
    def _verify_external_registration(
            identity_hash: int, external: ExternalIdentityKey,
            key: tuple[int, ...], parent_hash: int, ordinal: int) -> None:
        """核验外部记录恢复的键、parent 与 ordinal 均匹配登记请求。"""
        if external.key != key:
            raise IdentityCollisionError(
                f"identity hash={identity_hash} 外部完整键不一致")
        if (external.parent_hash is not None
                and external.parent_hash != parent_hash):
            raise IdentityIncompleteError("外部身份 parent_hash 与登记请求不一致")
        if external.ordinal is not None and external.ordinal != ordinal:
            raise IdentityIncompleteError("外部身份 ordinal 与登记请求不一致")

    def _resolve_assertion_record(
            self, identity_hash: int) -> ExternalIdentityKey | None:
        """从正规化 assertion record 和权威 scope registry 恢复原完整键。"""
        record = self._assertion_records.read_optional(identity_hash)
        if record is None:
            return None
        try:
            scope_key = self.read_key(IDENTITY_SCOPE, record.scope_hash)
        except KeyError as exc:
            raise IdentityIncompleteError(
                f"assertion hash={identity_hash} 引用的 scope 不存在") from exc
        return ExternalIdentityKey(
            record.stable_key(scope_key),
            parent_hash=record.scope_hash,
            ordinal=record.assertion_role,
        )

    @staticmethod
    def _metadata_from_header(row: dict[str, Any]) -> IdentityMetadata:
        """把已读取 header 转成强类型元数据并校验范围。"""
        return IdentityMetadata(
            _strict_int(row["identity_kind"], where="identity_kind", positive=True),
            _strict_int(row["identity_hash"], where="identity_hash", positive=True),
            _strict_int(row["key_size"], where="key_size", positive=True),
            _strict_int(row["parent_hash"], where="parent_hash", nonnegative=True),
            _strict_int(row["ordinal"], where="ordinal", nonnegative=True),
        )


__all__ = [
    "ASSERTION_SUPERSEDE_TABLE",
    "IDENTITY_ASSERTION",
    "IDENTITY_CLOCK",
    "IDENTITY_GRAPH_OBJECT",
    "IDENTITY_HEADER_TABLE",
    "IDENTITY_PART_TABLE",
    "IDENTITY_SCOPE",
    "IDENTITY_SOURCE_RECORD",
    "IDENTITY_SPAN_ROLE",
    "IDENTITY_TIMESTAMP",
    "IdentityCollisionError",
    "ExternalIdentityKey",
    "IdentityIncompleteError",
    "IdentityMetadata",
    "IdentityRegistryError",
    "IntegerIdentityRegistry",
    "LegacyAssertionAmbiguity",
    "SupersedeConflictError",
    "register_assertion_identity_tables",
]
