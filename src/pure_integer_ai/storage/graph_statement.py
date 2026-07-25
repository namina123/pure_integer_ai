"""一等图 statement 的正规化投影和旧物理行兼容读取。

本层不定义 HAS_REPRESENTATION、成员、角色、顺序或作用域等宿主枚举。所有语义由
predicate 图节点给出；新 statement 由带物理角色的 assertion record 投影，旧行只读核验。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.assertion_record import (
    ASSERTION_ROLE_GENERIC,
    ASSERTION_ROLE_GRAPH_STATEMENT,
    AssertionRecordError,
    AssertionRecordStore,
    AssertionStatementProjection,
)
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT
from pure_integer_ai.storage.graph_object import (
    GraphObjectRecord,
    GraphObjectIntegrityError,
    GraphObjectRepository,
)


GRAPH_STATEMENT_TABLE = "graph_statement"
GRAPH_STATEMENT_COLUMNS = [
    ("assertion_hash", TYPE_INT),
    ("predicate_identity_hash", TYPE_INT),
    ("predicate_space_id", TYPE_INT),
    ("predicate_local_id", TYPE_INT),
    ("object_kind_from", TYPE_INT),
    ("space_id_from", TYPE_INT),
    ("local_id_from", TYPE_INT),
    ("object_kind_to", TYPE_INT),
    ("space_id_to", TYPE_INT),
    ("local_id_to", TYPE_INT),
    ("scope_hash", TYPE_INT),
]


class GraphStatementIntegrityError(RuntimeError):
    """statement 出现重复、半写或与 assertion 不一致。"""


@dataclass(frozen=True, order=True)
class GraphStatementRecord:
    """storage 层不解释语义的纯整数 statement。"""

    assertion_hash: int
    predicate_identity_hash: int
    predicate_ref: tuple[int, int]
    subject_ref: tuple[int, int, int]
    object_ref: tuple[int, int, int]
    scope_hash: int


def register_graph_statement_table(backend: StorageBackend) -> None:
    """注册通用 typed statement 核心表。"""
    backend.register_table(
        GRAPH_STATEMENT_TABLE,
        GRAPH_STATEMENT_COLUMNS,
        disc.DISC_APPEND_ONLY,
        [
            ("assertion_hash",),
            ("predicate_identity_hash",),
            ("space_id_from", "local_id_from"),
            ("space_id_to", "local_id_to"),
        ],
        core=True,
    )


def _strict_positive(value: int, *, where: str) -> int:
    """校验 statement 中的索引、类型和编址是严格正整数。"""
    assert_int(value, _where=where)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{where} 必须为严格正整数")
    return value


class GraphStatementStore:
    """从 assertion 角色投影 statement，并严格兼容旧冗余物理行。"""

    def __init__(self, backend: StorageBackend,
                 objects: GraphObjectRepository,
                 assertions: AssertionRecordStore) -> None:
        if not isinstance(objects, GraphObjectRepository):
            raise TypeError("objects 必须是 GraphObjectRepository")
        if not isinstance(assertions, AssertionRecordStore):
            raise TypeError("assertions 必须是 AssertionRecordStore")
        self._backend = backend
        self._assertions = assertions
        self._objects = objects
        self._records_by_hash: dict[int, GraphStatementRecord] = {}
        self._predicates_by_hash: dict[int, GraphObjectRecord] = {}
        self._legacy_namespace_empty: bool | None = None

    def add(self, record: GraphStatementRecord) -> GraphStatementRecord:
        """核验 graph-statement assertion 投影；新格式不再追加冗余旧行。"""
        record = self._validate(record)
        cached = self._records_by_hash.get(record.assertion_hash)
        if cached is not None:
            if cached != record:
                raise GraphStatementIntegrityError(
                    "assertion 已在运行期绑定到不同 statement")
            return cached
        projection = self._projection(record.assertion_hash)
        if projection.assertion_role != ASSERTION_ROLE_GRAPH_STATEMENT:
            raise GraphStatementIntegrityError(
                "只有 graph-statement 角色 assertion 可以形成新 statement")
        expected = self._from_projection(projection)
        if expected != record:
            raise GraphStatementIntegrityError("assertion 投影与 statement 不一致")
        if self._legacy_rows(record.assertion_hash):
            raise GraphStatementIntegrityError(
                "新 graph-statement 角色不得与旧冗余行混存")
        self._records_by_hash[record.assertion_hash] = expected
        return expected

    def read(self, assertion_hash: int) -> GraphStatementRecord:
        """按 assertion hash 回读新投影或唯一匹配的旧 statement。"""
        _strict_positive(assertion_hash, where="assertion_hash")
        cached = self._records_by_hash.get(assertion_hash)
        if cached is not None:
            return cached
        projection = self._projection(assertion_hash)
        projected = self._from_projection(projection)
        legacy_rows = self._legacy_rows(assertion_hash)
        if projection.assertion_role == ASSERTION_ROLE_GRAPH_STATEMENT:
            if legacy_rows:
                raise GraphStatementIntegrityError(
                    "新 graph-statement 角色不得与旧冗余行混存")
            record = projected
        else:
            if len(legacy_rows) != 1:
                raise GraphStatementIntegrityError("旧 assertion 没有唯一 statement")
            record = self._from_row(legacy_rows[0])
            if record != projected:
                raise GraphStatementIntegrityError("旧 statement 与 assertion 投影不一致")
        self._records_by_hash[assertion_hash] = record
        return record

    def clear_runtime_caches(self) -> None:
        """外部 load 或故障注入后清空 statement 投影核验缓存。"""
        self._records_by_hash.clear()
        self._predicates_by_hash.clear()
        self._legacy_namespace_empty = None

    def query(self, *, predicate_identity_hash: int | None = None,
              subject_ref: tuple[int, int] | None = None,
              object_ref: tuple[int, int] | None = None
              ) -> tuple[GraphStatementRecord, ...]:
        """按 predicate、subject 和 object 合并新投影与唯一旧兼容行。"""
        where: dict[str, int] = {}
        if predicate_identity_hash is not None:
            where["predicate_identity_hash"] = _strict_positive(
                predicate_identity_hash, where="predicate_identity_hash")
        if subject_ref is not None:
            where["space_id_from"] = _strict_positive(
                subject_ref[0], where="subject_ref.space_id")
            where["local_id_from"] = _strict_positive(
                subject_ref[1], where="subject_ref.local_id")
        if object_ref is not None:
            where["space_id_to"] = _strict_positive(
                object_ref[0], where="object_ref.space_id")
            where["local_id_to"] = _strict_positive(
                object_ref[1], where="object_ref.local_id")
        try:
            projections = self._assertions.statement_projections(
                relation_kind=predicate_identity_hash,
                subject_ref=subject_ref,
                object_ref=object_ref,
            )
        except AssertionRecordError as exc:
            raise GraphStatementIntegrityError(str(exc)) from exc
        seen: dict[int, GraphStatementRecord] = {}
        for projection in projections:
            record = self._from_projection(projection)
            seen[record.assertion_hash] = record
        legacy_rows = (
            [] if self._legacy_is_empty()
            else self._backend.select(
                GRAPH_STATEMENT_TABLE, where=where or None)
        )
        legacy_seen: set[int] = set()
        for row in legacy_rows:
            legacy = self._from_row(row)
            if legacy.assertion_hash in legacy_seen:
                raise GraphStatementIntegrityError(
                    "query 命中重复 assertion statement")
            legacy_seen.add(legacy.assertion_hash)
            projection = self._projection(legacy.assertion_hash)
            if projection.assertion_role != ASSERTION_ROLE_GENERIC:
                raise GraphStatementIntegrityError(
                    "旧 statement 与新 graph-statement 角色混存")
            projected = self._from_projection(projection)
            if projected != legacy:
                raise GraphStatementIntegrityError(
                    "旧 statement 与 assertion 投影不一致")
            existing = seen.get(legacy.assertion_hash)
            if existing is not None and existing != legacy:
                raise GraphStatementIntegrityError(
                    "同 assertion 命中冲突的新旧 statement")
            seen[legacy.assertion_hash] = legacy
        return tuple(sorted(
            seen.values(),
            key=lambda item: item.assertion_hash,
        ))

    def _projection(self, assertion_hash: int) -> AssertionStatementProjection:
        """读取 assertion 主记录投影，并统一转换存储完整性错误。"""
        try:
            return self._assertions.projection(assertion_hash)
        except (AssertionRecordError, KeyError) as exc:
            raise GraphStatementIntegrityError(
                "statement 没有可恢复的 assertion 主记录") from exc

    def _from_projection(
            self, projection: AssertionStatementProjection
            ) -> GraphStatementRecord:
        """联合 predicate 图对象映射，把 assertion 固定字段恢复为 statement。"""
        predicate = self._predicates_by_hash.get(projection.relation_kind)
        if predicate is None:
            try:
                predicate = self._objects.read(projection.relation_kind)
            except (GraphObjectIntegrityError, KeyError) as exc:
                raise GraphStatementIntegrityError(
                    "statement predicate 没有唯一图对象映射") from exc
            self._predicates_by_hash[projection.relation_kind] = predicate
        return self._validate(GraphStatementRecord(
            projection.identity_hash,
            projection.relation_kind,
            (predicate.space_id, predicate.local_id),
            (
                projection.subject_key[0],
                projection.subject_key[1],
                projection.subject_key[2],
            ),
            (
                projection.object_key[0],
                projection.object_key[1],
                projection.object_key[2],
            ),
            projection.scope_hash,
        ))

    def _legacy_rows(self, assertion_hash: int) -> list[dict[str, int]]:
        """读取旧冗余行全集，不用 limit 掩盖重复或半写。"""
        if self._legacy_is_empty():
            return []
        return self._backend.select(
            GRAPH_STATEMENT_TABLE,
            where={"assertion_hash": assertion_hash},
        )

    def _legacy_is_empty(self) -> bool:
        """一次性核验旧表是否为空；清缓存后必须重新检查外部载入或故障注入。"""
        if self._legacy_namespace_empty is None:
            self._legacy_namespace_empty = (
                self._backend.count(GRAPH_STATEMENT_TABLE) == 0)
        return self._legacy_namespace_empty

    @staticmethod
    def _validate(record: GraphStatementRecord) -> GraphStatementRecord:
        """校验 record 的全部字段为严格正整数。"""
        _strict_positive(record.assertion_hash, where="assertion_hash")
        _strict_positive(
            record.predicate_identity_hash, where="predicate_identity_hash")
        for label, values in (
                ("predicate_ref", record.predicate_ref),
                ("subject_ref", record.subject_ref),
                ("object_ref", record.object_ref)):
            if not isinstance(values, tuple):
                raise ValueError(f"{label} 必须为整数元组")
            for index, value in enumerate(values):
                _strict_positive(value, where=f"{label}[{index}]")
        if len(record.predicate_ref) != 2:
            raise ValueError("predicate_ref 必须为二元节点编址")
        if len(record.subject_ref) != 3 or len(record.object_ref) != 3:
            raise ValueError("statement 端点必须为 object_kind + 二元节点编址")
        _strict_positive(record.scope_hash, where="scope_hash")
        return record

    @staticmethod
    def _to_row(record: GraphStatementRecord) -> dict[str, int]:
        """把 statement record 展平为纯整数存储行。"""
        return {
            "assertion_hash": record.assertion_hash,
            "predicate_identity_hash": record.predicate_identity_hash,
            "predicate_space_id": record.predicate_ref[0],
            "predicate_local_id": record.predicate_ref[1],
            "object_kind_from": record.subject_ref[0],
            "space_id_from": record.subject_ref[1],
            "local_id_from": record.subject_ref[2],
            "object_kind_to": record.object_ref[0],
            "space_id_to": record.object_ref[1],
            "local_id_to": record.object_ref[2],
            "scope_hash": record.scope_hash,
        }

    @classmethod
    def _from_row(cls, row: dict[str, int]) -> GraphStatementRecord:
        """从存储行重建并校验 statement record。"""
        return cls._validate(GraphStatementRecord(
            row["assertion_hash"],
            row["predicate_identity_hash"],
            (row["predicate_space_id"], row["predicate_local_id"]),
            (row["object_kind_from"], row["space_id_from"],
             row["local_id_from"]),
            (row["object_kind_to"], row["space_id_to"],
             row["local_id_to"]),
            row["scope_hash"],
        ))


__all__ = [
    "GRAPH_STATEMENT_TABLE",
    "GraphStatementIntegrityError",
    "GraphStatementRecord",
    "GraphStatementStore",
    "register_graph_statement_table",
]
