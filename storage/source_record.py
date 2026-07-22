"""来源完整身份与原始文本的 append-only 留档。

SourceRef 完整整数键保存在通用 identity registry；本表只保存可核验 hash、原文和
码点长度。原文属于非核心 TEXT 介质，不参与纯整数热路径或语义推理。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_SOURCE_RECORD,
    IntegerIdentityRegistry,
)
from pure_integer_ai.storage.backend import (
    StorageBackend,
    TYPE_INT,
    TYPE_TEXT,
    register_extension_table,
)


SOURCE_RECORD_TABLE = "source_record"
SOURCE_RECORD_COLUMNS = [
    ("source_hash", TYPE_INT),
    ("text_hash", TYPE_INT),
    ("codepoint_count", TYPE_INT),
    ("source_kind", TYPE_INT),
    ("source_id", TYPE_INT),
    ("document_id", TYPE_INT),
    ("corpus_version", TYPE_INT),
    ("parser_version", TYPE_INT),
    ("license_id", TYPE_TEXT),
    ("batch_id", TYPE_INT),
    ("companion_type_hash", TYPE_INT),
    ("companion_name_hash", TYPE_INT),
    ("companion_assoc_id", TYPE_INT),
    ("raw_text", TYPE_TEXT),
]
_TEXT_HASHER = Hasher("pure_integer_ai.source_record.text.v1")


class SourceRecordIntegrityError(RuntimeError):
    """同一来源身份绑定了冲突原文或存储出现重复、半写。"""


@dataclass(frozen=True, order=True)
class SourceRecordMetadata:
    """来源许可、批次和稳定 Companion assoc 绑定。"""

    license_id: str = ""
    batch_id: int = 0
    companion_type_hash: int = 0
    companion_name_hash: int = 0
    companion_assoc_id: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.license_id, str):
            raise TypeError("SourceRecord license_id 必须是字符串")
        values = (
            self.batch_id,
            self.companion_type_hash,
            self.companion_name_hash,
            self.companion_assoc_id,
        )
        assert_int(*values, _where="SourceRecordMetadata")
        if any(type(value) is not int for value in values) or self.batch_id < 0:
            raise ValueError("SourceRecord metadata 必须使用合法严格整数")
        companion_values = values[1:]
        if any(companion_values):
            if (not self.license_id
                    or any(value <= 0 for value in companion_values)):
                raise ValueError("完整来源 metadata 缺少许可或 Companion assoc")
        elif self.license_id:
            raise ValueError("声明许可时必须同时绑定 Companion assoc")
        elif self.batch_id != 0:
            raise ValueError("legacy 来源不得单独声明 batch")

    @property
    def complete(self) -> bool:
        """返回该 metadata 是否满足断奶后来源链要求。"""
        return bool(self.license_id and self.companion_assoc_id > 0)


@dataclass(frozen=True)
class SourceRecordStorage:
    """已从 registry 和 TEXT 表双向核验的来源记录。"""

    source_hash: int
    source_key: tuple[int, ...]
    text_hash: int
    codepoint_count: int
    source_kind: int
    source_id: int
    document_id: int
    corpus_version: int
    parser_version: int
    license_id: str
    batch_id: int
    companion_type_hash: int
    companion_name_hash: int
    companion_assoc_id: int
    raw_text: str

    @property
    def metadata(self) -> SourceRecordMetadata:
        """返回来源记录中的许可、批次和 Companion 绑定。"""
        return SourceRecordMetadata(
            self.license_id,
            self.batch_id,
            self.companion_type_hash,
            self.companion_name_hash,
            self.companion_assoc_id,
        )

    @property
    def metadata_complete(self) -> bool:
        """返回该记录是否满足断奶后完整来源要求。"""
        return self.metadata.complete


def register_source_record_table(backend: StorageBackend) -> None:
    """注册全局来源原文表；TEXT 只存在于非核心扩展层。"""
    register_extension_table(
        backend,
        SOURCE_RECORD_TABLE,
        SOURCE_RECORD_COLUMNS,
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[
            ("source_hash",),
            ("text_hash",),
            ("source_kind", "source_id", "document_id"),
        ],
    )


def _source_key(key: tuple[int, ...]) -> tuple[int, ...]:
    """校验 storage 接收的来源完整键只含严格整数。"""
    if not isinstance(key, tuple) or len(key) != 11:
        raise ValueError("source_key 必须是 SourceRef 的完整 11 整数键")
    assert_int(*key, _where="SourceRecordRepository.source_key")
    if any(type(value) is not int for value in key):
        raise ValueError("source_key 必须使用严格整数")
    return key


class SourceRecordRepository:
    """维护 SourceRef 完整键到唯一原始文本的 append-only 映射。"""

    def __init__(self, backend: StorageBackend, *,
                 registry: IntegerIdentityRegistry | None = None) -> None:
        self._backend = backend
        self._registry = registry or IntegerIdentityRegistry(backend)
        self._records_by_key: dict[tuple[int, ...], SourceRecordStorage] = {}
        self._records_by_hash: dict[int, SourceRecordStorage] = {}

    def put(
            self, source_key: tuple[int, ...], raw_text: str, *,
            metadata: SourceRecordMetadata | None = None,
            ) -> SourceRecordStorage:
        """幂等保存来源；显式 metadata 必须与既有许可和 Companion 绑定一致。"""
        stable_key = _source_key(source_key)
        if not isinstance(raw_text, str):
            raise TypeError("SourceRecord.raw_text 必须是字符串")
        if metadata is not None and not isinstance(metadata, SourceRecordMetadata):
            raise TypeError("metadata 必须是 SourceRecordMetadata 或 None")
        cached = self._records_by_key.get(stable_key)
        if cached is not None:
            if cached.raw_text != raw_text:
                raise SourceRecordIntegrityError("同一来源身份绑定了不同原文")
            if metadata is not None and cached.metadata != metadata:
                raise SourceRecordIntegrityError("同一来源身份绑定了不同 metadata")
            return cached
        source_hash = self._registry.register(
            IDENTITY_SOURCE_RECORD, stable_key)
        rows = self._backend.select(
            SOURCE_RECORD_TABLE, where={"source_hash": source_hash})
        if rows:
            if len(rows) != 1:
                raise SourceRecordIntegrityError("来源身份已绑定冲突原文或重复记录")
            existing = self._from_row(rows[0], source_key=stable_key)
            if (existing.raw_text != raw_text
                    or existing.text_hash != _TEXT_HASHER.h63(raw_text)
                    or (metadata is not None and existing.metadata != metadata)):
                raise SourceRecordIntegrityError("来源身份已绑定冲突原文或 metadata")
            self._cache(existing)
            return existing
        actual_metadata = metadata or SourceRecordMetadata()
        expected = self._build_record(
            source_hash, stable_key, raw_text, actual_metadata)
        self._backend.insert(SOURCE_RECORD_TABLE, self._to_row(expected))
        record = self.read(source_hash)
        self._cache(record)
        return record

    def put_complete(
            self, source_key: tuple[int, ...], raw_text: str, *,
            metadata: SourceRecordMetadata,
            ) -> SourceRecordStorage:
        """保存满足断奶后许可、批次和 Companion 依赖的完整来源记录。"""
        if not isinstance(metadata, SourceRecordMetadata) or not metadata.complete:
            raise ValueError("断奶后 SourceRecord 必须携带完整 metadata")
        record = self.put(source_key, raw_text, metadata=metadata)
        if not record.metadata_complete:
            raise SourceRecordIntegrityError("既有来源记录缺少断奶后 metadata")
        return record

    def find(self, source_key: tuple[int, ...]) -> SourceRecordStorage | None:
        """按来源完整键只读查找，不存在时不登记身份。"""
        stable_key = _source_key(source_key)
        cached = self._records_by_key.get(stable_key)
        if cached is not None:
            return cached
        source_hash = self._registry.find(
            IDENTITY_SOURCE_RECORD, stable_key)
        return None if source_hash is None else self.read(source_hash)

    def versions_for(
            self, source_key: tuple[int, ...],
            ) -> tuple[SourceRecordStorage, ...]:
        """返回同来源、owner 和非 parser 版本坐标下的全部解析版本。"""
        stable_key = _source_key(source_key)
        lineage = (*stable_key[:8], *stable_key[9:])
        records: list[SourceRecordStorage] = []
        for row in self._backend.select(SOURCE_RECORD_TABLE, where={
                "source_kind": stable_key[0],
                "source_id": stable_key[1],
                "document_id": stable_key[2],
        }):
            record = self.read(row["source_hash"])
            if (*record.source_key[:8], *record.source_key[9:]) == lineage:
                records.append(record)
        return tuple(sorted(
            records,
            key=lambda item: (item.parser_version, item.source_key),
        ))

    def read(self, source_hash: int) -> SourceRecordStorage:
        """按 hash 回读唯一原文，并核对 registry 完整键和文本 hash。"""
        assert_int(source_hash, _where="SourceRecordRepository.read")
        if type(source_hash) is not int or source_hash <= 0:
            raise ValueError("source_hash 必须为严格正整数")
        cached = self._records_by_hash.get(source_hash)
        if cached is not None:
            return cached
        stable_key = self._registry.read_key(
            IDENTITY_SOURCE_RECORD, source_hash)
        rows = self._backend.select(
            SOURCE_RECORD_TABLE, where={"source_hash": source_hash})
        if len(rows) != 1:
            raise SourceRecordIntegrityError("来源原文没有唯一记录")
        record = self._from_row(rows[0], source_key=stable_key)
        if record.text_hash != _TEXT_HASHER.h63(record.raw_text):
            raise SourceRecordIntegrityError("来源原文 hash 与内容不一致")
        self._cache(record)
        return record

    def clear_runtime_caches(self) -> None:
        """外部 load、迁移或故障注入后清空来源双向核验缓存。"""
        self._records_by_key.clear()
        self._records_by_hash.clear()

    def source_count(self) -> int:
        """返回当前后端的唯一来源原文行数。"""
        return len(self._backend.select(SOURCE_RECORD_TABLE, where=None))

    @property
    def backend(self) -> StorageBackend:
        """返回承载来源记录和身份表的后端。"""
        return self._backend

    def _cache(self, record: SourceRecordStorage) -> None:
        """缓存已同时核验 registry 完整键和原文 hash 的来源记录。"""
        self._records_by_key[record.source_key] = record
        self._records_by_hash[record.source_hash] = record

    @staticmethod
    def _from_row(row, *, source_key: tuple[int, ...] = ()) -> SourceRecordStorage:
        """把存储行转换为不可变记录并执行范围校验。"""
        values = (
            row["source_hash"],
            row["text_hash"],
            row["codepoint_count"],
            row["source_kind"],
            row["source_id"],
            row["document_id"],
            row["corpus_version"],
            row["parser_version"],
            row["batch_id"],
            row["companion_type_hash"],
            row["companion_name_hash"],
            row["companion_assoc_id"],
        )
        assert_int(*values, _where="SourceRecordStorage")
        raw_text = row["raw_text"]
        license_id = row["license_id"]
        if not isinstance(raw_text, str) or not isinstance(license_id, str):
            raise SourceRecordIntegrityError("来源原文或许可存储类型非法")
        if (values[0] <= 0 or values[2] < 0 or values[2] != len(raw_text)
                or len(source_key) != 11
                or values[3:8] != (
                    source_key[0], source_key[1], source_key[2],
                    source_key[7], source_key[8])):
            raise SourceRecordIntegrityError("来源原文索引或码点长度非法")
        try:
            metadata = SourceRecordMetadata(
                license_id,
                values[8],
                values[9],
                values[10],
                values[11],
            )
        except (TypeError, ValueError) as exc:
            raise SourceRecordIntegrityError("来源 metadata 非法") from exc
        return SourceRecordStorage(
            values[0], source_key, values[1], values[2],
            values[3], values[4], values[5], values[6], values[7],
            metadata.license_id, metadata.batch_id,
            metadata.companion_type_hash, metadata.companion_name_hash,
            metadata.companion_assoc_id, raw_text)

    @staticmethod
    def _build_record(
            source_hash: int, source_key: tuple[int, ...], raw_text: str,
            metadata: SourceRecordMetadata,
            ) -> SourceRecordStorage:
        """从完整 SourceRef 键和 metadata 构造规范来源记录。"""
        return SourceRecordStorage(
            source_hash,
            source_key,
            _TEXT_HASHER.h63(raw_text),
            len(raw_text),
            source_key[0],
            source_key[1],
            source_key[2],
            source_key[7],
            source_key[8],
            metadata.license_id,
            metadata.batch_id,
            metadata.companion_type_hash,
            metadata.companion_name_hash,
            metadata.companion_assoc_id,
            raw_text,
        )

    @staticmethod
    def _to_row(record: SourceRecordStorage) -> dict[str, int | str]:
        """把规范来源记录转换为 append-only 存储行。"""
        return {
            "source_hash": record.source_hash,
            "text_hash": record.text_hash,
            "codepoint_count": record.codepoint_count,
            "source_kind": record.source_kind,
            "source_id": record.source_id,
            "document_id": record.document_id,
            "corpus_version": record.corpus_version,
            "parser_version": record.parser_version,
            "license_id": record.license_id,
            "batch_id": record.batch_id,
            "companion_type_hash": record.companion_type_hash,
            "companion_name_hash": record.companion_name_hash,
            "companion_assoc_id": record.companion_assoc_id,
            "raw_text": record.raw_text,
        }


__all__ = [
    "SOURCE_RECORD_TABLE",
    "SourceRecordIntegrityError",
    "SourceRecordMetadata",
    "SourceRecordRepository",
    "SourceRecordStorage",
    "register_source_record_table",
]
