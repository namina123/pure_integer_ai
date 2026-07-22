"""AssertionIdentity 的正规化纯整数物理记录。

本模块不导入 cognition 领域类型。调用方把两个 TypedRef 稳定键、scope hash 和
开放限定项拆入固定记录；读取时再由 identity registry 提供完整 scope 键，重建原
AssertionIdentity 稳定键。表只负责物理去重，不能删减或改写领域身份语义。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT


ASSERTION_RECORD_TABLE = "assertion_record"
ASSERTION_QUALIFIER_TABLE = "assertion_qualifier"

ASSERTION_ROLE_GENERIC = 0
ASSERTION_ROLE_GRAPH_STATEMENT = 1

TYPED_REF_KEY_SIZE = 11
INLINE_QUALIFIER_COUNT = 3

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

ASSERTION_RECORD_COLUMNS = [
    ("identity_hash", TYPE_INT),
    ("assertion_role", TYPE_INT),
    ("key_version", TYPE_INT),
    ("relation_kind", TYPE_INT),
    *[(f"subject_{name}", TYPE_INT) for name in _REF_FIELD_NAMES],
    *[(f"object_{name}", TYPE_INT) for name in _REF_FIELD_NAMES],
    ("scope_hash", TYPE_INT),
    ("provenance_kind", TYPE_INT),
    ("epistemic_origin", TYPE_INT),
    ("content_version", TYPE_INT),
    *[(f"qualifier_{index}", TYPE_INT)
      for index in range(INLINE_QUALIFIER_COUNT)],
    ("qualifier_size", TYPE_INT),
]

ASSERTION_QUALIFIER_COLUMNS = [
    ("identity_hash", TYPE_INT),
    ("ordinal", TYPE_INT),
    ("qualifier_value", TYPE_INT),
]


class AssertionRecordError(RuntimeError):
    """正规化 assertion 记录错误基类。"""


class AssertionRecordCollisionError(AssertionRecordError):
    """同一 assertion hash 命中不同正规化内容。"""


class AssertionRecordIncompleteError(AssertionRecordError):
    """assertion record 或 qualifier 半写、重复、缺失。"""


def register_assertion_record_tables(backend: StorageBackend) -> None:
    """注册 assertion 主记录和有序限定项两个核心 append-only 表。"""
    backend.register_table(
        ASSERTION_RECORD_TABLE,
        ASSERTION_RECORD_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("identity_hash",),
            ("assertion_role",),
            ("scope_hash",),
            ("relation_kind",),
            ("assertion_role", "relation_kind"),
            ("assertion_role", "subject_space_id", "subject_local_id"),
            ("assertion_role", "object_space_id", "object_local_id"),
        ],
        core=True,
    )
    backend.register_table(
        ASSERTION_QUALIFIER_TABLE,
        ASSERTION_QUALIFIER_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("identity_hash",),
            ("identity_hash", "ordinal"),
        ],
        core=True,
    )


def _strict_int(value: int, *, where: str,
                nonnegative: bool = False,
                positive: bool = False) -> int:
    """校验物理记录字段是严格整数，并按字段职责限制范围。"""
    if type(value) is not int:
        assert_int(value, _where=where)
        raise ValueError(f"{where} 必须为严格整数")
    if positive and value <= 0:
        raise ValueError(f"{where} 必须为正整数")
    if nonnegative and value < 0:
        raise ValueError(f"{where} 必须为非负整数")
    return value


def _typed_ref_key(key: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """校验当前物理版本使用的固定长度 TypedRef 稳定键。"""
    if not isinstance(key, tuple) or len(key) != TYPED_REF_KEY_SIZE:
        raise ValueError(f"{where} 长度必须为 {TYPED_REF_KEY_SIZE}")
    for index, value in enumerate(key):
        _strict_int(value, where=f"{where}[{index}]")
    for index in (0, 1, 2, 6):
        _strict_int(key[index], where=f"{where}[{index}]", positive=True)
    for index in (3, 4, 5, 7, 8, 9, 10):
        _strict_int(key[index], where=f"{where}[{index}]", nonnegative=True)
    return key


def _qualifier_key(values: tuple[int, ...]) -> tuple[int, ...]:
    """校验限定项保持调用方给定的完整整数顺序。"""
    if not isinstance(values, tuple):
        raise ValueError("assertion qualifiers 必须是整数元组")
    for index, value in enumerate(values):
        _strict_int(value, where=f"assertion qualifiers[{index}]")
    return values


def _row_ref_key(row: dict[str, Any], prefix: str) -> tuple[int, ...]:
    """从主记录固定列恢复一个 TypedRef 稳定键。"""
    return _typed_ref_key(
        tuple(row[f"{prefix}_{name}"] for name in _REF_FIELD_NAMES),
        where=f"assertion_record.{prefix}",
    )


@dataclass(frozen=True)
class AssertionRecord:
    """已完整核验的 assertion 主记录和有序限定项。"""

    identity_hash: int
    assertion_role: int
    key_version: int
    relation_kind: int
    subject_key: tuple[int, ...]
    object_key: tuple[int, ...]
    scope_hash: int
    provenance_kind: int
    epistemic_origin: int
    content_version: int
    qualifiers: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_int(self.identity_hash, where="identity_hash", positive=True)
        _assertion_role(self.assertion_role)
        _strict_int(self.key_version, where="key_version", positive=True)
        _strict_int(self.relation_kind, where="relation_kind", positive=True)
        _typed_ref_key(self.subject_key, where="subject_key")
        _typed_ref_key(self.object_key, where="object_key")
        _strict_int(self.scope_hash, where="scope_hash", positive=True)
        _strict_int(
            self.provenance_kind, where="provenance_kind", positive=True)
        _strict_int(
            self.epistemic_origin, where="epistemic_origin", nonnegative=True)
        _strict_int(
            self.content_version, where="content_version", nonnegative=True)
        _qualifier_key(self.qualifiers)

    def stable_key(self, scope_key: tuple[int, ...]) -> tuple[int, ...]:
        """用权威 scope 完整键重建原 AssertionIdentity 稳定键。"""
        if not isinstance(scope_key, tuple) or not scope_key:
            raise ValueError("scope_key 必须是非空整数元组")
        for index, value in enumerate(scope_key):
            _strict_int(value, where=f"scope_key[{index}]")
        return (
            self.key_version,
            self.relation_kind,
            len(self.subject_key),
            *self.subject_key,
            len(self.object_key),
            *self.object_key,
            len(scope_key),
            *scope_key,
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            len(self.qualifiers),
            *self.qualifiers,
        )

    def to_row(self) -> dict[str, int]:
        """生成 scope 只存引用、前三项限定内联的主表行。"""
        row = {
            "identity_hash": self.identity_hash,
            "assertion_role": self.assertion_role,
            "key_version": self.key_version,
            "relation_kind": self.relation_kind,
            "scope_hash": self.scope_hash,
            "provenance_kind": self.provenance_kind,
            "epistemic_origin": self.epistemic_origin,
            "content_version": self.content_version,
            "qualifier_size": len(self.qualifiers),
        }
        for index in range(INLINE_QUALIFIER_COUNT):
            row[f"qualifier_{index}"] = (
                self.qualifiers[index]
                if index < len(self.qualifiers)
                else 0
            )
        for prefix, key in (
                ("subject", self.subject_key),
                ("object", self.object_key)):
            for name, value in zip(_REF_FIELD_NAMES, key):
                row[f"{prefix}_{name}"] = value
        return row


@dataclass(frozen=True)
class AssertionStatementProjection:
    """从 assertion 主记录投影的图关系固定字段，不复制限定项。"""

    identity_hash: int
    assertion_role: int
    relation_kind: int
    subject_key: tuple[int, ...]
    object_key: tuple[int, ...]
    scope_hash: int


def _assertion_role(value: int) -> int:
    """校验 assertion 物理角色，角色不进入领域稳定键。"""
    _strict_int(value, where="assertion_role", nonnegative=True)
    if value not in {ASSERTION_ROLE_GENERIC, ASSERTION_ROLE_GRAPH_STATEMENT}:
        raise ValueError("assertion_role 未注册")
    return value


class AssertionRecordStore:
    """幂等写入并完整回读正规化 assertion 物理记录。"""

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend
        self._records_by_hash: dict[int, AssertionRecord] = {}

    def register(self, record: AssertionRecord) -> None:
        """追加一条完整记录；碰撞、孤儿限定项和既有半写状态均拒绝。"""
        cached = self._records_by_hash.get(record.identity_hash)
        if cached is not None:
            if cached != record:
                raise AssertionRecordCollisionError(
                    f"assertion hash={record.identity_hash} 命中不同正规化记录")
            return
        rows = self._record_rows(record.identity_hash)
        qualifier_rows = self._qualifier_rows(record.identity_hash)
        if rows:
            existing = self._read_rows(
                record.identity_hash, rows, qualifier_rows)
            if existing != record:
                raise AssertionRecordCollisionError(
                    f"assertion hash={record.identity_hash} 命中不同正规化记录")
            self._records_by_hash[record.identity_hash] = existing
            return
        if qualifier_rows:
            raise AssertionRecordIncompleteError(
                f"assertion hash={record.identity_hash} 存在孤儿 qualifier")

        for ordinal, value in self._overflow_qualifiers(record.qualifiers):
            self._backend.insert(ASSERTION_QUALIFIER_TABLE, {
                "identity_hash": record.identity_hash,
                "ordinal": ordinal,
                "qualifier_value": value,
            })
        self._backend.insert(ASSERTION_RECORD_TABLE, record.to_row())
        if self.read(record.identity_hash) != record:
            raise AssertionRecordIncompleteError("assertion record 写后核验失败")

    def is_empty(self) -> bool:
        """核验主记录和限定项表都为空，供单写者首次批量追加声明使用。"""
        return (
            self._backend.count(ASSERTION_RECORD_TABLE) == 0
            and self._backend.count(ASSERTION_QUALIFIER_TABLE) == 0
        )

    def append_new(self, record: AssertionRecord) -> None:
        """在调用方已声明命名空间为空时追加新记录，不执行重复读回。"""
        cached = self._records_by_hash.get(record.identity_hash)
        if cached is not None:
            if cached != record:
                raise AssertionRecordCollisionError(
                    f"assertion hash={record.identity_hash} 命中不同正规化记录")
            return
        for ordinal, value in self._overflow_qualifiers(record.qualifiers):
            self._backend.insert(ASSERTION_QUALIFIER_TABLE, {
                "identity_hash": record.identity_hash,
                "ordinal": ordinal,
                "qualifier_value": value,
            })
        self._backend.insert(ASSERTION_RECORD_TABLE, record.to_row())
        self._records_by_hash[record.identity_hash] = record

    def read_optional(self, identity_hash: int) -> AssertionRecord | None:
        """读取可选记录；完全不存在返回空，任一孤儿或半写状态失败。"""
        _strict_int(identity_hash, where="identity_hash", positive=True)
        cached = self._records_by_hash.get(identity_hash)
        if cached is not None:
            return cached
        rows = self._record_rows(identity_hash)
        qualifier_rows = self._qualifier_rows(identity_hash)
        if not rows:
            if qualifier_rows:
                raise AssertionRecordIncompleteError(
                    f"assertion hash={identity_hash} 存在孤儿 qualifier")
            return None
        record = self._read_rows(identity_hash, rows, qualifier_rows)
        self._records_by_hash[identity_hash] = record
        return record

    def read(self, identity_hash: int) -> AssertionRecord:
        """读取唯一完整记录，不存在或损坏时 fail closed。"""
        record = self.read_optional(identity_hash)
        if record is None:
            raise KeyError(f"assertion record hash={identity_hash} 不存在")
        return record

    def projection(self, identity_hash: int) -> AssertionStatementProjection:
        """按 hash 读取唯一主记录投影；限定项完整性由随后 assertion 恢复核验。"""
        _strict_int(identity_hash, where="identity_hash", positive=True)
        cached = self._records_by_hash.get(identity_hash)
        if cached is not None:
            return self._projection_from_record(cached)
        rows = self._record_rows(identity_hash)
        if len(rows) != 1:
            raise AssertionRecordIncompleteError(
                f"assertion hash={identity_hash} record 数量={len(rows)}")
        return self._projection_from_row(rows[0])

    def clear_runtime_caches(self) -> None:
        """外部 load 或故障注入后清空正规化 assertion 运行期缓存。"""
        self._records_by_hash.clear()

    def statement_projections(
            self, *, relation_kind: int | None = None,
            subject_ref: tuple[int, int] | None = None,
            object_ref: tuple[int, int] | None = None,
            ) -> tuple[AssertionStatementProjection, ...]:
        """按图查询字段读取 role=graph-statement 的唯一主记录投影。"""
        where: dict[str, int] = {
            "assertion_role": ASSERTION_ROLE_GRAPH_STATEMENT,
        }
        if relation_kind is not None:
            where["relation_kind"] = _strict_int(
                relation_kind, where="relation_kind", positive=True)
        for label, ref, prefix in (
                ("subject_ref", subject_ref, "subject"),
                ("object_ref", object_ref, "object")):
            if ref is None:
                continue
            if not isinstance(ref, tuple) or len(ref) != 2:
                raise ValueError(f"{label} 必须是二元节点编址")
            where[f"{prefix}_space_id"] = _strict_int(
                ref[0], where=f"{label}.space_id", positive=True)
            where[f"{prefix}_local_id"] = _strict_int(
                ref[1], where=f"{label}.local_id", positive=True)
        rows = self._backend.select(ASSERTION_RECORD_TABLE, where=where)
        projections: list[AssertionStatementProjection] = []
        seen: set[int] = set()
        for row in rows:
            projection = self._projection_from_row(row)
            if projection.identity_hash in seen:
                raise AssertionRecordIncompleteError(
                    "graph statement projection 命中重复 assertion")
            seen.add(projection.identity_hash)
            projections.append(projection)
        return tuple(sorted(
            projections,
            key=lambda item: item.identity_hash,
        ))

    def _record_rows(self, identity_hash: int) -> list[dict[str, Any]]:
        """读取同 hash 的全部主记录，不用 limit 掩盖重复。"""
        return self._backend.select(
            ASSERTION_RECORD_TABLE,
            where={"identity_hash": identity_hash},
        )

    def _qualifier_rows(self, identity_hash: int) -> list[dict[str, Any]]:
        """按 ordinal 读取同一 assertion 的全部限定项。"""
        return self._backend.select(
            ASSERTION_QUALIFIER_TABLE,
            where={"identity_hash": identity_hash},
            order_by="ordinal",
        )

    def _read_rows(self, identity_hash: int,
                   rows: list[dict[str, Any]],
                   qualifier_rows: list[dict[str, Any]]) -> AssertionRecord:
        """区分新旧限定项 codec，核验唯一性与连续性后恢复记录。"""
        if len(rows) != 1:
            raise AssertionRecordIncompleteError(
                f"assertion hash={identity_hash} record 数量={len(rows)}")
        row = rows[0]
        row_hash = _strict_int(
            row["identity_hash"], where="assertion_record.identity_hash",
            positive=True)
        if row_hash != identity_hash:
            raise AssertionRecordIncompleteError("assertion record hash 列不一致")
        qualifier_size = _strict_int(
            row["qualifier_size"], where="assertion_record.qualifier_size",
            nonnegative=True)
        inline_values = tuple(
            row.get(f"qualifier_{index}")
            for index in range(INLINE_QUALIFIER_COUNT)
        )
        legacy_codec = all(value is None for value in inline_values)
        if not legacy_codec and any(value is None for value in inline_values):
            raise AssertionRecordIncompleteError(
                f"assertion hash={identity_hash} qualifier 内联列不完整")

        qualifiers: list[int] = []
        if legacy_codec:
            expected_overflow_size = qualifier_size
            first_overflow_ordinal = 0
        else:
            inline_size = min(qualifier_size, INLINE_QUALIFIER_COUNT)
            for index, raw_value in enumerate(inline_values):
                value = _strict_int(
                    raw_value, where=f"assertion_record.qualifier_{index}")
                if index < inline_size:
                    qualifiers.append(value)
                elif value != 0:
                    raise AssertionRecordIncompleteError(
                        f"assertion hash={identity_hash} 未使用 qualifier 内联槽非零")
            expected_overflow_size = max(
                qualifier_size - INLINE_QUALIFIER_COUNT, 0)
            first_overflow_ordinal = INLINE_QUALIFIER_COUNT

        if len(qualifier_rows) != expected_overflow_size:
            raise AssertionRecordIncompleteError(
                f"assertion hash={identity_hash} qualifier 数量不一致")
        for offset, qualifier_row in enumerate(qualifier_rows):
            expected_ordinal = first_overflow_ordinal + offset
            ordinal = _strict_int(
                qualifier_row["ordinal"], where="assertion_qualifier.ordinal",
                nonnegative=True)
            if ordinal != expected_ordinal:
                raise AssertionRecordIncompleteError(
                    f"assertion hash={identity_hash} qualifier ordinal 不连续")
            qualifiers.append(_strict_int(
                qualifier_row["qualifier_value"],
                where="assertion_qualifier.qualifier_value"))
        return AssertionRecord(
            row_hash,
            _assertion_role(row.get(
                "assertion_role", ASSERTION_ROLE_GENERIC)),
            _strict_int(
                row["key_version"], where="assertion_record.key_version",
                positive=True),
            _strict_int(
                row["relation_kind"], where="assertion_record.relation_kind",
                positive=True),
            _row_ref_key(row, "subject"),
            _row_ref_key(row, "object"),
            _strict_int(
                row["scope_hash"], where="assertion_record.scope_hash",
                positive=True),
            _strict_int(
                row["provenance_kind"],
                where="assertion_record.provenance_kind", positive=True),
            _strict_int(
                row["epistemic_origin"],
                where="assertion_record.epistemic_origin", nonnegative=True),
            _strict_int(
                row["content_version"],
                where="assertion_record.content_version", nonnegative=True),
            tuple(qualifiers),
        )

    @staticmethod
    def _overflow_qualifiers(
            qualifiers: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        """返回保留原 ordinal 的限定项溢出段，前三项由主记录承载。"""
        return tuple(enumerate(
            qualifiers[INLINE_QUALIFIER_COUNT:],
            start=INLINE_QUALIFIER_COUNT,
        ))

    @staticmethod
    def _projection_from_row(row: dict[str, Any]) -> AssertionStatementProjection:
        """从主表行恢复关系投影，并校验所有参与图查询的固定字段。"""
        return AssertionStatementProjection(
            _strict_int(
                row["identity_hash"], where="identity_hash", positive=True),
            _assertion_role(row.get(
                "assertion_role", ASSERTION_ROLE_GENERIC)),
            _strict_int(
                row["relation_kind"], where="relation_kind", positive=True),
            _row_ref_key(row, "subject"),
            _row_ref_key(row, "object"),
            _strict_int(
                row["scope_hash"], where="scope_hash", positive=True),
        )

    @staticmethod
    def _projection_from_record(
            record: AssertionRecord) -> AssertionStatementProjection:
        """从已核验缓存记录直接生成关系投影，不重复访问后端。"""
        return AssertionStatementProjection(
            record.identity_hash,
            record.assertion_role,
            record.relation_kind,
            record.subject_key,
            record.object_key,
            record.scope_hash,
        )


__all__ = [
    "ASSERTION_QUALIFIER_TABLE",
    "ASSERTION_RECORD_TABLE",
    "ASSERTION_ROLE_GENERIC",
    "ASSERTION_ROLE_GRAPH_STATEMENT",
    "INLINE_QUALIFIER_COUNT",
    "AssertionRecord",
    "AssertionRecordCollisionError",
    "AssertionRecordError",
    "AssertionRecordIncompleteError",
    "AssertionRecordStore",
    "AssertionStatementProjection",
    "TYPED_REF_KEY_SIZE",
    "register_assertion_record_tables",
]
