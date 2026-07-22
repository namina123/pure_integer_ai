"""一等图对象的完整身份到运行时节点映射。

新记录由 ``graph_object`` 主表自行承载 identity header 和固定 codec 字段，开放组件与
Hypothesis 竞争组由独立物理模块保存。旧通用 header/parts 仍可只读恢复；identity hash
始终只是可核验索引，不保存 surface，也不定义具体语言或结构语义。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_GRAPH_OBJECT,
    IDENTITY_HEADER_TABLE,
    IDENTITY_PART_TABLE,
    ExternalIdentityKey,
    IdentityCollisionError,
    IdentityIncompleteError,
    IntegerIdentityRegistry,
)
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT
from pure_integer_ai.storage.graph_object_identity import (
    GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE,
    GRAPH_HYPOTHESIS_GROUP_TABLE,
    GRAPH_OBJECT_COMPONENT_TABLE,
    GRAPH_OBJECT_IDENTITY_COLUMNS,
    GraphObjectIdentityCollisionError,
    GraphObjectIdentityIncompleteError,
    GraphObjectIdentitySpec,
    GraphObjectIdentityStore,
    register_graph_object_identity_tables,
)


GRAPH_OBJECT_TABLE = "graph_object"
GRAPH_OBJECT_COLUMNS = [
    ("identity_hash", TYPE_INT),
    ("object_kind", TYPE_INT),
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("space_type", TYPE_INT),
    ("space_type_hash", TYPE_INT),
    ("space_name_hash", TYPE_INT),
    *GRAPH_OBJECT_IDENTITY_COLUMNS,
]


class GraphObjectIntegrityError(RuntimeError):
    """图对象身份、节点编址或稳定空间映射不一致。"""


@dataclass(frozen=True, order=True)
class GraphObjectRecord:
    """已核验的一等图对象物化记录。"""

    identity_hash: int
    object_kind: int
    space_id: int
    local_id: int
    space_identity_key: tuple[int, int, int]


def register_graph_object_table(backend: StorageBackend) -> None:
    """注册一等图对象主表和版本化 identity codec 子表。"""
    backend.register_table(
        GRAPH_OBJECT_TABLE,
        GRAPH_OBJECT_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("identity_hash",),
            ("space_id", "local_id"),
            ("object_kind",),
        ],
        core=True,
    )
    register_graph_object_identity_tables(backend)


def _strict_positive(value: int, *, where: str) -> int:
    """校验图对象索引和编址字段是严格正整数。"""
    assert_int(value, _where=where)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{where} 必须为严格正整数")
    return value


def _strict_nonnegative(value: int, *, where: str) -> int:
    """校验稳定哈希字段是严格非负整数。"""
    assert_int(value, _where=where)
    if type(value) is not int or value < 0:
        raise ValueError(f"{where} 必须为严格非负整数")
    return value


def _space_identity_key(key: tuple[int, ...]) -> tuple[int, int, int]:
    """校验固定三元稳定空间身份；hash 可为零，空间类型必须为正。"""
    if not isinstance(key, tuple) or len(key) != 3:
        raise ValueError("space_identity_key 必须是三个严格整数")
    _strict_positive(key[0], where="space_identity_key[0]")
    _strict_nonnegative(key[1], where="space_identity_key[1]")
    _strict_nonnegative(key[2], where="space_identity_key[2]")
    return key


class GraphObjectRepository:
    """维护 self-headed 完整身份与唯一图节点之间的 append-only 映射。"""

    def __init__(self, backend: StorageBackend,
                 registry: IntegerIdentityRegistry | None = None) -> None:
        self._backend = backend
        self._registry = registry or IntegerIdentityRegistry(backend)
        self._identity_store = GraphObjectIdentityStore(
            backend, self._registry.read_key)
        self._keys_by_hash: dict[int, tuple[int, ...]] = {}
        self._records_by_hash: dict[int, GraphObjectRecord] = {}
        self._records_by_ref: dict[tuple[int, int], GraphObjectRecord] = {}
        self._fresh_identity_namespace: bool | None = None
        self._registry.register_external_key_resolver(
            IDENTITY_GRAPH_OBJECT,
            self._resolve_external_key,
            self_headed=True,
        )

    @property
    def registry(self) -> IntegerIdentityRegistry:
        """暴露共享 registry 供领域 facade 核验完整键。"""
        return self._registry

    def find_identity(self, key: tuple[int, ...]) -> int | None:
        """只读查找对象完整键对应的可核验 hash。"""
        GraphObjectIdentitySpec.generic(key)
        if self._fresh_identity_namespace is None:
            self._fresh_identity_namespace = self._namespace_is_empty()
        if self._fresh_identity_namespace:
            identity_hash = self._registry.identity_hash(
                IDENTITY_GRAPH_OBJECT, key)
            existing = self._keys_by_hash.get(identity_hash)
            if existing is None:
                return None
            if existing != key:
                raise IdentityCollisionError(
                    f"identity hash={identity_hash} 命中不同完整键")
            return identity_hash
        return self._registry.find(IDENTITY_GRAPH_OBJECT, key)

    def add(self, spec: GraphObjectIdentitySpec, *, object_kind: int,
            space_id: int, local_id: int,
            space_identity_key: tuple[int, ...]) -> GraphObjectRecord:
        """把完整身份和 codec 载荷一次性绑定到唯一节点。"""
        if not isinstance(spec, GraphObjectIdentitySpec):
            raise TypeError("GraphObjectRepository.add 需要 identity spec")
        _strict_positive(object_kind, where="object_kind")
        _strict_positive(space_id, where="space_id")
        _strict_positive(local_id, where="local_id")
        stable_space = _space_identity_key(space_identity_key)
        if spec.stable_key[0] != object_kind:
            raise GraphObjectIntegrityError("对象完整键与 object_kind 不一致")
        if self._fresh_identity_namespace is None:
            self._fresh_identity_namespace = self._namespace_is_empty()

        def write_record(identity_hash: int, *, fast: bool) -> None:
            """先写 codec 子载荷，再追加包含节点映射的 self-headed 主记录。"""
            if fast:
                self._identity_store.append_new(identity_hash, spec)
            else:
                self._reject_existing_mapping(
                    identity_hash, space_id, local_id)
                self._identity_store.register(identity_hash, spec)
            row = {
                "identity_hash": identity_hash,
                "object_kind": object_kind,
                "space_id": space_id,
                "local_id": local_id,
                "space_type": stable_space[0],
                "space_type_hash": stable_space[1],
                "space_name_hash": stable_space[2],
                **spec.main_row_fields(),
            }
            self._backend.insert(GRAPH_OBJECT_TABLE, row)
            self._keys_by_hash[identity_hash] = spec.stable_key

        if self._fresh_identity_namespace:
            identity_hash = self._registry.identity_hash(
                IDENTITY_GRAPH_OBJECT, spec.stable_key)
            existing_key = self._keys_by_hash.get(identity_hash)
            if existing_key is not None:
                if existing_key != spec.stable_key:
                    raise IdentityCollisionError(
                        f"identity hash={identity_hash} 命中不同完整键")
                return self._validate_existing_mapping(
                    identity_hash, object_kind, space_id,
                    local_id, stable_space)
            identity_hash = (
                self._registry.append_new_resolved_in_empty_namespace(
                    IDENTITY_GRAPH_OBJECT,
                    spec.stable_key,
                    writer=lambda new_hash: write_record(
                        new_hash, fast=True),
                ))
            record = GraphObjectRecord(
                identity_hash, object_kind, space_id,
                local_id, stable_space)
            self._cache_record(record)
            return record

        identity_hash = self._registry.register_resolved(
            IDENTITY_GRAPH_OBJECT,
            spec.stable_key,
            writer=lambda new_hash: write_record(new_hash, fast=False),
        )
        return self._validate_existing_mapping(
            identity_hash, object_kind, space_id,
            local_id, stable_space)

    def read(self, identity_hash: int) -> GraphObjectRecord:
        """按身份 hash 回读唯一映射，并核对完整身份键。"""
        _strict_positive(identity_hash, where="identity_hash")
        cached = self._records_by_hash.get(identity_hash)
        if cached is not None:
            return cached
        identity_key = self._registry.read_key(
            IDENTITY_GRAPH_OBJECT, identity_hash)
        rows = self._backend.select(
            GRAPH_OBJECT_TABLE, where={"identity_hash": identity_hash})
        if len(rows) != 1:
            raise GraphObjectIntegrityError("对象身份没有唯一物化记录")
        record = self._record_from_row(rows[0])
        if identity_key[0] != record.object_kind:
            raise GraphObjectIntegrityError("物化记录 object_kind 与完整键不一致")
        self._cache_record(record)
        return record

    def read_by_ref(self, space_id: int,
                    local_id: int) -> GraphObjectRecord:
        """按运行时节点编址回读唯一对象身份。"""
        _strict_positive(space_id, where="space_id")
        _strict_positive(local_id, where="local_id")
        cached = self._records_by_ref.get((space_id, local_id))
        if cached is not None:
            return cached
        rows = self._backend.select(GRAPH_OBJECT_TABLE, where={
            "space_id": space_id,
            "local_id": local_id,
        })
        if len(rows) != 1:
            raise GraphObjectIntegrityError("节点没有唯一对象身份映射")
        record = self._record_from_row(rows[0])
        self._registry.read_key(IDENTITY_GRAPH_OBJECT, record.identity_hash)
        self._cache_record(record)
        return record

    def clear_runtime_caches(self) -> None:
        """外部 load、迁移或故障注入后清空映射和 codec 运行期缓存。"""
        self._keys_by_hash.clear()
        self._records_by_hash.clear()
        self._records_by_ref.clear()
        self._fresh_identity_namespace = None
        self._identity_store.clear_runtime_caches()

    def _namespace_is_empty(self) -> bool:
        """核验新旧 GraphObject identity 主载荷和子载荷均为空。"""
        return (
            self._backend.count(GRAPH_OBJECT_TABLE) == 0
            and self._backend.count(
                IDENTITY_HEADER_TABLE,
                where={"identity_kind": IDENTITY_GRAPH_OBJECT}) == 0
            and self._backend.count(
                IDENTITY_PART_TABLE,
                where={"identity_kind": IDENTITY_GRAPH_OBJECT}) == 0
            and self._identity_store.auxiliary_namespace_is_empty()
        )

    def _resolve_external_key(
            self, identity_hash: int) -> ExternalIdentityKey | None:
        """从新 graph_object 主记录和 codec 子表恢复 self-headed 完整键。"""
        rows = self._backend.select(
            GRAPH_OBJECT_TABLE, where={"identity_hash": identity_hash})
        if not rows:
            return None
        resolved: list[tuple[int, ...] | None] = []
        try:
            for row in rows:
                resolved.append(
                    self._identity_store.read_optional(identity_hash, row))
        except GraphObjectIdentityCollisionError as exc:
            raise IdentityCollisionError(str(exc)) from exc
        except GraphObjectIdentityIncompleteError as exc:
            raise IdentityIncompleteError(str(exc)) from exc
        if all(key is None for key in resolved):
            return None
        if len(rows) != 1 or resolved[0] is None:
            raise IdentityIncompleteError(
                f"graph object hash={identity_hash} 新旧主记录混存或重复")
        return ExternalIdentityKey(resolved[0], parent_hash=0, ordinal=0)

    def _reject_existing_mapping(
            self, identity_hash: int,
            space_id: int, local_id: int) -> None:
        """严格新写前拒绝同身份或同节点已有任何映射。"""
        by_identity = self._backend.select(
            GRAPH_OBJECT_TABLE, where={"identity_hash": identity_hash})
        by_ref = self._backend.select(GRAPH_OBJECT_TABLE, where={
            "space_id": space_id,
            "local_id": local_id,
        })
        if by_identity or by_ref:
            raise GraphObjectIntegrityError("身份或节点已有图对象映射")

    def _validate_existing_mapping(
            self, identity_hash: int, object_kind: int,
            space_id: int, local_id: int,
            stable_space: tuple[int, int, int]) -> GraphObjectRecord:
        """核验同 hash 和同节点查询唯一指向请求中的完整映射。"""
        by_identity = self._backend.select(
            GRAPH_OBJECT_TABLE, where={"identity_hash": identity_hash})
        by_ref = self._backend.select(GRAPH_OBJECT_TABLE, where={
            "space_id": space_id,
            "local_id": local_id,
        })
        expected = GraphObjectRecord(
            identity_hash, object_kind, space_id, local_id, stable_space)
        if len(by_identity) != 1 or len(by_ref) != 1:
            raise GraphObjectIntegrityError("图对象映射重复或半写")
        existing = self._record_from_row(by_identity[0])
        if existing != expected or by_identity[0] != by_ref[0]:
            raise GraphObjectIntegrityError("身份或节点已绑定到其他图对象")
        self._cache_record(existing)
        return existing

    def _cache_record(self, record: GraphObjectRecord) -> None:
        """缓存已经完整核验的身份和节点双向映射。"""
        self._records_by_hash[record.identity_hash] = record
        self._records_by_ref[(record.space_id, record.local_id)] = record

    @staticmethod
    def _record_from_row(row: dict[str, int]) -> GraphObjectRecord:
        """把存储行转换为不可变记录并执行范围校验。"""
        return GraphObjectRecord(
            _strict_positive(row["identity_hash"], where="identity_hash"),
            _strict_positive(row["object_kind"], where="object_kind"),
            _strict_positive(row["space_id"], where="space_id"),
            _strict_positive(row["local_id"], where="local_id"),
            _space_identity_key((
                row["space_type"],
                row["space_type_hash"],
                row["space_name_hash"],
            )),
        )


__all__ = [
    "GRAPH_OBJECT_TABLE",
    "GraphObjectIntegrityError",
    "GraphObjectIdentitySpec",
    "GraphObjectRecord",
    "GraphObjectRepository",
    "register_graph_object_table",
]
