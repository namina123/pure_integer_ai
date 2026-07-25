"""A-05 来源准入裁决与独立来源簇的 append-only 映射。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_SOURCE_CLUSTER,
    IDENTITY_SOURCE_RECORD,
    IDENTITY_SOURCE_TRUST_ASSESSMENT,
    IntegerIdentityRegistry,
)
from pure_integer_ai.storage.backend import (
    StorageBackend,
    TYPE_INT,
    register_extension_table,
)
from pure_integer_ai.storage.source_record import SOURCE_RECORD_TABLE


SOURCE_TRUST_ASSESSMENT_TABLE = "source_trust_assessment"
SOURCE_TRUST_ASSESSMENT_COLUMNS = [
    ("source_hash", TYPE_INT),
    ("assessment_hash", TYPE_INT),
    ("cluster_hash", TYPE_INT),
]


class SourceTrustStorageIntegrityError(RuntimeError):
    """来源准入映射缺失、冲突、半写或指向不存在的来源。"""


def _key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验身份 registry 接收的开放键为非空严格整数元组。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空严格整数元组")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


@dataclass(frozen=True)
class SourceTrustStorageRecord:
    """一个完整 SourceRef 对应的准入裁决与来源簇身份。"""

    source_hash: int
    source_key: tuple[int, ...]
    assessment_hash: int
    assessment_key: tuple[int, ...]
    cluster_hash: int
    cluster_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验三个身份索引为正整数且完整键非空。"""
        assert_int(
            self.source_hash,
            self.assessment_hash,
            self.cluster_hash,
            _where="SourceTrustStorageRecord",
        )
        if any(type(item) is not int or item <= 0 for item in (
                self.source_hash, self.assessment_hash, self.cluster_hash)):
            raise ValueError("来源准入存储 hash 必须是严格正整数")
        _key(self.source_key, label="SourceTrustStorageRecord.source_key")
        _key(
            self.assessment_key,
            label="SourceTrustStorageRecord.assessment_key",
        )
        _key(self.cluster_key, label="SourceTrustStorageRecord.cluster_key")


def register_source_trust_table(backend: StorageBackend) -> None:
    """注册来源到 assessment/cluster 身份的 append-only 映射表。"""
    register_extension_table(
        backend,
        SOURCE_TRUST_ASSESSMENT_TABLE,
        SOURCE_TRUST_ASSESSMENT_COLUMNS,
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[
            ("source_hash",),
            ("assessment_hash",),
            ("cluster_hash",),
        ],
        recovery_key=("source_hash",),
    )


class SourceTrustStorageRepository:
    """维护 SourceRef、准入 assessment 与独立来源簇的三轴绑定。"""

    def __init__(
            self,
            backend: StorageBackend,
            *,
            registry: IntegerIdentityRegistry | None = None,
            ) -> None:
        """绑定后端与共享身份 registry，不创建任何来源语义。"""
        self._backend = backend
        self._registry = registry or IntegerIdentityRegistry(backend)
        self._cache: dict[tuple[int, ...], SourceTrustStorageRecord] = {}

    def put(
            self,
            source_key: tuple[int, ...],
            assessment_key: tuple[int, ...],
            cluster_key: tuple[int, ...],
            ) -> SourceTrustStorageRecord:
        """幂等保存三轴绑定；同一 SourceRef 的任一身份漂移均拒绝。"""
        source_key = _key(source_key, label="source trust source_key")
        assessment_key = _key(
            assessment_key, label="source trust assessment_key")
        cluster_key = _key(cluster_key, label="source trust cluster_key")
        source_hash = self._registry.find(IDENTITY_SOURCE_RECORD, source_key)
        if source_hash is None:
            raise SourceTrustStorageIntegrityError(
                "来源准入映射要求 SourceRecord 已先持久化")
        source_rows = self._backend.select(
            SOURCE_RECORD_TABLE, where={"source_hash": source_hash})
        if len(source_rows) != 1:
            raise SourceTrustStorageIntegrityError(
                "来源准入映射没有唯一 SourceRecord")
        assessment_hash = self._registry.register(
            IDENTITY_SOURCE_TRUST_ASSESSMENT, assessment_key)
        cluster_hash = self._registry.register(
            IDENTITY_SOURCE_CLUSTER, cluster_key)
        expected = SourceTrustStorageRecord(
            source_hash,
            source_key,
            assessment_hash,
            assessment_key,
            cluster_hash,
            cluster_key,
        )
        rows = self._backend.select(
            SOURCE_TRUST_ASSESSMENT_TABLE,
            where={"source_hash": source_hash},
        )
        if rows:
            if len(rows) != 1:
                raise SourceTrustStorageIntegrityError(
                    "同一来源存在多个准入映射")
            actual = self._from_row(rows[0])
            if actual != expected:
                raise SourceTrustStorageIntegrityError(
                    "同一来源的准入 assessment 或来源簇发生漂移")
            self._cache[source_key] = actual
            return actual
        self._backend.insert(SOURCE_TRUST_ASSESSMENT_TABLE, {
            "source_hash": source_hash,
            "assessment_hash": assessment_hash,
            "cluster_hash": cluster_hash,
        })
        actual = self.read(source_hash)
        if actual != expected:
            raise SourceTrustStorageIntegrityError("来源准入写后回读不一致")
        return actual

    def find(
            self,
            source_key: tuple[int, ...],
            ) -> SourceTrustStorageRecord | None:
        """按完整 SourceRef 只读返回准入映射，不登记新身份。"""
        source_key = _key(source_key, label="source trust find source_key")
        cached = self._cache.get(source_key)
        if cached is not None:
            return cached
        source_hash = self._registry.find(IDENTITY_SOURCE_RECORD, source_key)
        if source_hash is None:
            return None
        rows = self._backend.select(
            SOURCE_TRUST_ASSESSMENT_TABLE,
            where={"source_hash": source_hash},
        )
        if not rows:
            return None
        if len(rows) != 1:
            raise SourceTrustStorageIntegrityError(
                "同一来源存在多个准入映射")
        record = self._from_row(rows[0])
        if record.source_key != source_key:
            raise SourceTrustStorageIntegrityError(
                "来源准入映射命中其他 SourceRef")
        self._cache[source_key] = record
        return record

    def read(self, source_hash: int) -> SourceTrustStorageRecord:
        """按来源 hash 回读三轴完整键，并核验各自 registry 身份。"""
        assert_int(source_hash, _where="SourceTrustStorageRepository.read")
        if type(source_hash) is not int or source_hash <= 0:
            raise ValueError("source trust source_hash 必须是严格正整数")
        source_key = self._registry.read_key(
            IDENTITY_SOURCE_RECORD, source_hash)
        cached = self._cache.get(source_key)
        if cached is not None:
            return cached
        rows = self._backend.select(
            SOURCE_TRUST_ASSESSMENT_TABLE,
            where={"source_hash": source_hash},
        )
        if len(rows) != 1:
            raise SourceTrustStorageIntegrityError(
                "来源没有唯一准入映射")
        record = self._from_row(rows[0])
        self._cache[source_key] = record
        return record

    def clear_runtime_caches(self) -> None:
        """外部恢复、克隆或故障回装后清空只读缓存。"""
        self._cache.clear()

    def _from_row(self, row) -> SourceTrustStorageRecord:
        """从固定映射行恢复三个完整身份，任一缺失均 fail closed。"""
        try:
            source_hash = row["source_hash"]
            assessment_hash = row["assessment_hash"]
            cluster_hash = row["cluster_hash"]
        except KeyError as exc:
            raise SourceTrustStorageIntegrityError(
                f"来源准入映射缺少字段 {exc.args[0]}") from exc
        assert_int(
            source_hash,
            assessment_hash,
            cluster_hash,
            _where="source trust storage row",
        )
        if any(type(item) is not int or item <= 0 for item in (
                source_hash, assessment_hash, cluster_hash)):
            raise SourceTrustStorageIntegrityError(
                "来源准入映射包含非法身份 hash")
        return SourceTrustStorageRecord(
            source_hash,
            self._registry.read_key(IDENTITY_SOURCE_RECORD, source_hash),
            assessment_hash,
            self._registry.read_key(
                IDENTITY_SOURCE_TRUST_ASSESSMENT, assessment_hash),
            cluster_hash,
            self._registry.read_key(IDENTITY_SOURCE_CLUSTER, cluster_hash),
        )


__all__ = [
    "SOURCE_TRUST_ASSESSMENT_TABLE",
    "SourceTrustStorageIntegrityError",
    "SourceTrustStorageRecord",
    "SourceTrustStorageRepository",
    "register_source_trust_table",
]
