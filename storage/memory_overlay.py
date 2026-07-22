"""Memory overlay relation 的纯整数物理记录。

本模块只负责完整整数键、固定物理列和 append-only 记录。Core identity、关系 owner、
scope 以及可见性判断由 cognition 层 facade 负责，storage 不从端点空间推断关系所有权。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT


MEMORY_OVERLAY_TABLE = "memory_overlay_relation"
TYPED_REF_KEY_SIZE = 11
SPACE_KEY_SIZE = 3
OWNER_KEY_SIZE = 4
VERSION_KEY_SIZE = 4

_REF_FIELD_NAMES = (
    "object_kind",
    "space_id",
    "local_id",
    "tenant_id",
    "user_id",
    "session_id",
    "visibility",
    "corpus_version",
    "parser_version",
    "primitive_version",
    "curriculum_version",
)

MEMORY_OVERLAY_COLUMNS = [
    ("identity_hash", TYPE_INT),
    ("space_id", TYPE_INT),
    ("space_type", TYPE_INT),
    ("space_type_hash", TYPE_INT),
    ("space_name_hash", TYPE_INT),
    ("owner_tenant_id", TYPE_INT),
    ("owner_user_id", TYPE_INT),
    ("owner_session_id", TYPE_INT),
    ("owner_visibility", TYPE_INT),
    ("corpus_version", TYPE_INT),
    ("parser_version", TYPE_INT),
    ("primitive_version", TYPE_INT),
    ("curriculum_version", TYPE_INT),
    *[(f"predicate_{name}", TYPE_INT) for name in _REF_FIELD_NAMES],
    *[(f"subject_{name}", TYPE_INT) for name in _REF_FIELD_NAMES],
    *[(f"object_{name}", TYPE_INT) for name in _REF_FIELD_NAMES],
    ("scope_hash", TYPE_INT),
    ("provenance_kind", TYPE_INT),
    ("epistemic_origin", TYPE_INT),
    ("content_version", TYPE_INT),
]


class MemoryOverlayIntegrityError(RuntimeError):
    """Memory overlay 记录出现重复、半写或完整键不一致。"""


def _strict_int(value: int, *, where: str,
                nonnegative: bool = False,
                positive: bool = False) -> int:
    """校验物理字段使用严格整数，并按列职责限制取值范围。"""
    if type(value) is not int:
        assert_int(value, _where=where)
        raise ValueError(f"{where} 必须为严格整数")
    if positive and value <= 0:
        raise ValueError(f"{where} 必须为正整数")
    if nonnegative and value < 0:
        raise ValueError(f"{where} 必须为非负整数")
    return value


def _fixed_key(value: tuple[int, ...], *, size: int,
               where: str) -> tuple[int, ...]:
    """校验固定长度整数键，拒绝截断、尾随和布尔值。"""
    if not isinstance(value, tuple) or len(value) != size:
        raise ValueError(f"{where} 长度必须为 {size}")
    for index, item in enumerate(value):
        _strict_int(item, where=f"{where}[{index}]")
    return value


def _ref_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """校验 TypedRef 固定键的类型、编址、owner 和版本字段。"""
    key = _fixed_key(value, size=TYPED_REF_KEY_SIZE, where=where)
    for index in (0, 1, 2, 6):
        _strict_int(key[index], where=f"{where}[{index}]", positive=True)
    for index in (3, 4, 5, 7, 8, 9, 10):
        _strict_int(key[index], where=f"{where}[{index}]", nonnegative=True)
    return key


def register_memory_overlay_table(backend: StorageBackend) -> None:
    """注册不写 Core 图的 Memory overlay 关系表。"""
    backend.register_table(
        MEMORY_OVERLAY_TABLE,
        MEMORY_OVERLAY_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("identity_hash",),
            ("space_id",),
            ("space_id", "owner_visibility", "owner_tenant_id",
             "owner_user_id", "owner_session_id"),
            ("space_id", "predicate_space_id", "predicate_local_id"),
            ("space_id", "subject_space_id", "subject_local_id"),
            ("space_id", "object_space_id", "object_local_id"),
        ],
        core=True,
    )


@dataclass(frozen=True)
class MemoryOverlayRecord:
    """已完成物理校验的 Memory overlay 关系索引记录。"""

    identity_hash: int
    space_id: int
    space_identity_key: tuple[int, int, int]
    owner_key: tuple[int, int, int, int]
    version_key: tuple[int, int, int, int]
    predicate_key: tuple[int, ...]
    subject_key: tuple[int, ...]
    object_key: tuple[int, ...]
    scope_hash: int
    provenance_kind: int
    epistemic_origin: int
    content_version: int

    def __post_init__(self) -> None:
        """校验关系索引的固定列，确保物理记录可被领域层完整恢复。"""
        _strict_int(self.identity_hash, where="identity_hash", positive=True)
        _strict_int(self.space_id, where="space_id", positive=True)
        _fixed_key(
            self.space_identity_key, size=SPACE_KEY_SIZE,
            where="space_identity_key")
        _strict_int(self.space_identity_key[0], where="space_type",
                    positive=True)
        _strict_int(self.space_identity_key[1], where="space_type_hash",
                    nonnegative=True)
        _strict_int(self.space_identity_key[2], where="space_name_hash",
                    nonnegative=True)
        owner = _fixed_key(self.owner_key, size=OWNER_KEY_SIZE,
                           where="owner_key")
        for index, value in enumerate(owner):
            _strict_int(
                value, where=f"owner_key[{index}]",
                positive=index == 3, nonnegative=index != 3)
        _fixed_key(self.version_key, size=VERSION_KEY_SIZE,
                   where="version_key")
        for index, value in enumerate(self.version_key):
            _strict_int(value, where=f"version_key[{index}]", nonnegative=True)
        _ref_key(self.predicate_key, where="predicate_key")
        _ref_key(self.subject_key, where="subject_key")
        _ref_key(self.object_key, where="object_key")
        _strict_int(self.scope_hash, where="scope_hash", positive=True)
        _strict_int(
            self.provenance_kind, where="provenance_kind", positive=True)
        _strict_int(
            self.epistemic_origin, where="epistemic_origin", nonnegative=True)
        _strict_int(
            self.content_version, where="content_version", nonnegative=True)

    def to_row(self) -> dict[str, int]:
        """把完整关系索引展平为纯整数物理行。"""
        row: dict[str, int] = {
            "identity_hash": self.identity_hash,
            "space_id": self.space_id,
            "space_type": self.space_identity_key[0],
            "space_type_hash": self.space_identity_key[1],
            "space_name_hash": self.space_identity_key[2],
            "owner_tenant_id": self.owner_key[0],
            "owner_user_id": self.owner_key[1],
            "owner_session_id": self.owner_key[2],
            "owner_visibility": self.owner_key[3],
            "corpus_version": self.version_key[0],
            "parser_version": self.version_key[1],
            "primitive_version": self.version_key[2],
            "curriculum_version": self.version_key[3],
            "scope_hash": self.scope_hash,
            "provenance_kind": self.provenance_kind,
            "epistemic_origin": self.epistemic_origin,
            "content_version": self.content_version,
        }
        for prefix, key in (
                ("predicate", self.predicate_key),
                ("subject", self.subject_key),
                ("object", self.object_key)):
            row.update({
                f"{prefix}_{name}": value
                for name, value in zip(_REF_FIELD_NAMES, key)
            })
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MemoryOverlayRecord":
        """从物理行恢复关系索引，并拒绝缺列或错误字段。"""
        try:
            return cls(
                row["identity_hash"],
                row["space_id"],
                (row["space_type"], row["space_type_hash"],
                 row["space_name_hash"]),
                (row["owner_tenant_id"], row["owner_user_id"],
                 row["owner_session_id"], row["owner_visibility"]),
                (row["corpus_version"], row["parser_version"],
                 row["primitive_version"], row["curriculum_version"]),
                tuple(row[f"predicate_{name}"] for name in _REF_FIELD_NAMES),
                tuple(row[f"subject_{name}"] for name in _REF_FIELD_NAMES),
                tuple(row[f"object_{name}"] for name in _REF_FIELD_NAMES),
                row["scope_hash"],
                row["provenance_kind"],
                row["epistemic_origin"],
                row["content_version"],
            )
        except KeyError as exc:
            raise MemoryOverlayIntegrityError(
                f"Memory overlay 物理行缺少字段: {exc.args[0]}") from exc


class MemoryOverlayRecordStore:
    """幂等追加并恢复 Memory overlay 的纯整数关系索引。"""

    def __init__(self, backend: StorageBackend) -> None:
        """绑定已经注册 overlay 表的存储后端。"""
        self._backend = backend
        self._records: dict[int, MemoryOverlayRecord] = {}

    def add(self, record: MemoryOverlayRecord) -> MemoryOverlayRecord:
        """追加完整关系索引；同 hash 不同内容和重复物理行均拒绝。"""
        cached = self._records.get(record.identity_hash)
        if cached is not None:
            if cached != record:
                raise MemoryOverlayIntegrityError("overlay identity hash 命中不同记录")
            return cached
        rows = self._backend.select(
            MEMORY_OVERLAY_TABLE,
            where={"identity_hash": record.identity_hash},
        )
        if len(rows) > 1:
            raise MemoryOverlayIntegrityError("overlay identity hash 存在重复物理行")
        if rows:
            existing = MemoryOverlayRecord.from_row(rows[0])
            if existing != record:
                raise MemoryOverlayIntegrityError("overlay identity hash 命中不同记录")
            self._records[record.identity_hash] = existing
            return existing
        self._backend.insert(MEMORY_OVERLAY_TABLE, record.to_row())
        restored = self.read(record.identity_hash)
        if restored != record:
            raise MemoryOverlayIntegrityError("overlay 写后回读不一致")
        return restored

    def read(self, identity_hash: int) -> MemoryOverlayRecord:
        """按稳定 hash 回读唯一 overlay 物理记录。"""
        _strict_int(identity_hash, where="identity_hash", positive=True)
        cached = self._records.get(identity_hash)
        if cached is not None:
            return cached
        rows = self._backend.select(
            MEMORY_OVERLAY_TABLE,
            where={"identity_hash": identity_hash},
        )
        if len(rows) != 1:
            raise MemoryOverlayIntegrityError(
                f"overlay identity hash={identity_hash} 物理行数量={len(rows)}")
        record = MemoryOverlayRecord.from_row(rows[0])
        self._records[identity_hash] = record
        return record

    def query(self, *, space_id: int) -> tuple[MemoryOverlayRecord, ...]:
        """按 Memory 运行空间读取关系索引，返回稳定 hash 顺序。"""
        _strict_int(space_id, where="space_id", positive=True)
        rows = self._backend.select(
            MEMORY_OVERLAY_TABLE,
            where={"space_id": space_id},
            order_by="identity_hash",
        )
        records: list[MemoryOverlayRecord] = []
        seen: set[int] = set()
        for row in rows:
            record = self._record_from_row(row)
            if record.identity_hash in seen:
                raise MemoryOverlayIntegrityError(
                    "overlay query 命中重复 identity 物理行")
            seen.add(record.identity_hash)
            records.append(record)
        return tuple(records)

    def clear_runtime_caches(self) -> None:
        """清空物理记录缓存，使 dump/load 或损坏注入重新走完整回读。"""
        self._records.clear()

    def _record_from_row(self, row: dict[str, Any]) -> MemoryOverlayRecord:
        """恢复单行并确认缓存中的同 hash 内容没有漂移。"""
        record = MemoryOverlayRecord.from_row(row)
        cached = self._records.get(record.identity_hash)
        if cached is not None and cached != record:
            raise MemoryOverlayIntegrityError("overlay 缓存与物理行不一致")
        self._records[record.identity_hash] = record
        return record


__all__ = [
    "MEMORY_OVERLAY_COLUMNS",
    "MEMORY_OVERLAY_TABLE",
    "MemoryOverlayIntegrityError",
    "MemoryOverlayRecord",
    "MemoryOverlayRecordStore",
    "register_memory_overlay_table",
]
