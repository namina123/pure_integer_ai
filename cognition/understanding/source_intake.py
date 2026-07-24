"""断奶后来源进入 Companion、SourceRecord 和精确原文切片的统一入口。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.source_record import (
    SourceRecordIntegrityError,
    SourceRecordMetadata,
    SourceRecordRepository,
    SourceRecordStorage,
)
from pure_integer_ai.storage.memory_forget import MemoryForgetVisibility
from pure_integer_ai.storage.spaces.companion import CompanionSpace


class SourceIntakeIntegrityError(RuntimeError):
    """来源、Companion 留档和精确切片之间出现不一致。"""


@dataclass(frozen=True)
class SourceSlice:
    """可回到唯一 SourceRef 和原文位置的只读切片。"""

    source: SourceRef
    source_hash: int
    start: int
    end: int
    text: str


class SourceIntake:
    """把断奶后外部来源先留档于 Companion，再绑定完整 SourceRecord。"""

    def __init__(
            self, repository: SourceRecordRepository,
            companion: CompanionSpace | None,
            ) -> None:
        if not isinstance(repository, SourceRecordRepository):
            raise TypeError("repository 必须是 SourceRecordRepository")
        if not isinstance(companion, CompanionSpace):
            raise RuntimeError("断奶后来源摄入缺少 Companion")
        if repository.backend is not companion.backend:
            raise RuntimeError("SourceRecord 与 Companion 必须使用同一 backend")
        self.repository = repository
        self.companion = companion
        self._forget_visibility: MemoryForgetVisibility | None = None

    def attach_forget_visibility(
            self,
            visibility: MemoryForgetVisibility,
            ) -> None:
        """安装 M-11 可见性，使来源重学不复用已遗忘完整键和伴随项。"""
        if not isinstance(visibility, MemoryForgetVisibility):
            raise TypeError("visibility 必须是 MemoryForgetVisibility")
        repository_backend = getattr(
            visibility.store.store.repository, "backend", None)
        if repository_backend is not None and repository_backend is not self.repository.backend:
            raise ValueError("forget visibility 与 SourceIntake backend 不一致")
        if (self._forget_visibility is not None
                and self._forget_visibility is not visibility):
            raise ValueError("SourceIntake 已绑定其他 forget visibility")
        self._forget_visibility = visibility

    def ensure(
            self, source: SourceRef, raw_text: str, *,
            license_id: str, batch_id: int,
            ) -> SourceRecordStorage:
        """幂等保存完整来源；同来源的文本、许可、批次或 assoc 漂移均拒绝。"""
        if not isinstance(source, SourceRef):
            raise TypeError("source 必须是 SourceRef")
        if not isinstance(raw_text, str):
            raise TypeError("raw_text 必须是字符串")
        if not isinstance(license_id, str) or not license_id:
            raise ValueError("断奶后来源必须声明非空 license_id")
        assert_int(batch_id, _where="SourceIntake.ensure.batch_id")
        if type(batch_id) is not int or batch_id < 0:
            raise ValueError("batch_id 必须是非负严格整数")

        existing = self.repository.find(source.stable_key())
        if existing is not None:
            if self._source_is_forgotten(existing):
                raise SourceIntakeIntegrityError(
                    "已遗忘的完整 SourceRef 不得按同一身份重放")
            self._verify_existing(
                existing, raw_text=raw_text,
                license_id=license_id, batch_id=batch_id)
            return existing

        prior_versions = tuple(
            item for item in self.repository.versions_for(source.stable_key())
            if not self._source_is_forgotten(item)
        )
        if prior_versions:
            prior = prior_versions[-1]
            if (prior.raw_text != raw_text
                    or prior.license_id != license_id
                    or not prior.metadata_complete):
                raise SourceIntakeIntegrityError(
                    "同来源谱系的 parser 版本绑定了不同原文或许可")
            self._verify_companion(prior)
            prior_assoc = (
                prior.companion_type_hash,
                prior.companion_name_hash,
                prior.companion_assoc_id,
            )
            if any(
                    item.raw_text != raw_text
                    or item.license_id != license_id
                    or (
                        item.companion_type_hash,
                        item.companion_name_hash,
                        item.companion_assoc_id,
                    ) != prior_assoc
                    for item in prior_versions):
                raise SourceIntakeIntegrityError(
                    "同来源谱系的 parser 版本没有共享唯一 Companion 原文")
            assoc_type_hash, assoc_name_hash, assoc_id = prior_assoc
        else:
            assoc_id = self.companion.put_text(
                raw_text, meta=source.source_kind)
            assoc = self.companion.assoc_identity(assoc_id)
            assoc_type_hash = assoc.space.type_hash
            assoc_name_hash = assoc.space.name_hash
        metadata = SourceRecordMetadata(
            license_id,
            batch_id,
            assoc_type_hash,
            assoc_name_hash,
            assoc_id,
        )
        return self.repository.put_complete(
            source.stable_key(), raw_text, metadata=metadata)

    def read_slice(
            self, source: SourceRef, start: int, end: int,
            ) -> SourceSlice:
        """回读完整来源并返回精确码点区间，禁止越界和不完整来源。"""
        if not isinstance(source, SourceRef):
            raise TypeError("source 必须是 SourceRef")
        assert_int(start, end, _where="SourceIntake.read_slice")
        if (type(start) is not int or type(end) is not int
                or start < 0 or end < start):
            raise ValueError("来源切片区间非法")
        record = self.repository.find(source.stable_key())
        if (record is None or not record.metadata_complete
                or self._source_is_forgotten(record)):
            raise SourceIntakeIntegrityError("来源没有完整 Companion 记录")
        self._verify_companion(record)
        if end > len(record.raw_text):
            raise ValueError("来源切片超出原文")
        return SourceSlice(
            source, record.source_hash, start, end,
            record.raw_text[start:end])

    def _verify_existing(
            self, record: SourceRecordStorage, *, raw_text: str,
            license_id: str, batch_id: int,
            ) -> None:
        """核验已存在来源的完整 metadata 和 Companion 原文未漂移。"""
        if not record.metadata_complete:
            raise SourceRecordIntegrityError(
                "既有 legacy SourceRecord 不能静默补写断奶后 metadata")
        if (record.raw_text != raw_text
                or record.license_id != license_id
                or record.batch_id != batch_id):
            raise SourceIntakeIntegrityError("来源文本、许可或批次发生漂移")
        self._verify_companion(record)

    def _verify_companion(self, record: SourceRecordStorage) -> None:
        """核验 SourceRecord 指向当前稳定 Companion 空间内的唯一同文 assoc。"""
        identity = self.companion.identity
        if (record.companion_type_hash != identity.type_hash
                or record.companion_name_hash != identity.name_hash):
            raise SourceIntakeIntegrityError("SourceRecord 指向不同 Companion 空间")
        row = self.companion.read(record.companion_assoc_id)
        if (row["text"] != record.raw_text
                or row["meta"] != record.source_kind):
            raise SourceIntakeIntegrityError(
                "Companion assoc 与 SourceRecord 来源内容不一致")

    def _source_is_forgotten(self, record: SourceRecordStorage) -> bool:
        """判断来源或其绑定的 Companion assoc 是否已退出正式逻辑视图。"""
        visibility = self._forget_visibility
        if visibility is None:
            return False
        assoc_key = (
            record.companion_type_hash,
            record.companion_name_hash,
            record.companion_assoc_id,
        )
        return (
            visibility.source_is_forgotten(record.source_key)
            or visibility.companion_is_forgotten(assoc_key)
        )


__all__ = [
    "SourceIntake",
    "SourceIntakeIntegrityError",
    "SourceSlice",
]
