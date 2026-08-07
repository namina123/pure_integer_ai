"""跨 Memory Use 的 append-only 最小桥索引存储。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    TYPE_INT,
    StorageBackend,
    register_extension_table,
)


CROSS_MEMORY_USE_TABLE = "cross_memory_use_index"


def register_cross_memory_use_table(backend: StorageBackend) -> None:
    """注册不归入任一 Memory owner 的 append-only 派生桥表。"""
    register_extension_table(
        backend,
        CROSS_MEMORY_USE_TABLE,
        [
            ("source_space_id", TYPE_INT),
            ("use_event_hash", TYPE_INT),
            ("use_object_hash", TYPE_INT),
            ("source_timeline_seq", TYPE_INT),
            ("target_space_id", TYPE_INT),
            ("target_object_hash", TYPE_INT),
            ("target_tenant_id", TYPE_INT),
            ("target_user_id", TYPE_INT),
            ("target_session_id", TYPE_INT),
            ("target_visibility", TYPE_INT),
        ],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[
            ("source_space_id", "use_event_hash"),
            ("target_space_id", "target_object_hash"),
            ("target_tenant_id", "target_user_id", "target_session_id"),
        ],
        recovery_key=("source_space_id", "use_event_hash"),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class CrossMemoryUseRecord:
    """一条 interaction Use 指向其他 Memory 空间对象的最小事实。"""

    source_space_id: int
    use_event_hash: int
    use_object_hash: int
    source_timeline_seq: int
    target_space_id: int
    target_object_hash: int
    target_owner_key: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        """核验空间、事件、对象和时间线均为正整数且空间互异。"""
        values = (
            self.source_space_id,
            self.use_event_hash,
            self.use_object_hash,
            self.source_timeline_seq,
            self.target_space_id,
            self.target_object_hash,
        )
        if any(type(item) is not int or item <= 0 for item in values):
            raise ValueError("CrossMemoryUseRecord 索引字段必须为正严格整数")
        if self.source_space_id == self.target_space_id:
            raise ValueError("CrossMemoryUseRecord 只保存跨空间 Use")
        if (not isinstance(self.target_owner_key, tuple)
                or len(self.target_owner_key) != 4
                or any(type(item) is not int or item < 0
                       for item in self.target_owner_key)):
            raise TypeError("CrossMemoryUseRecord target_owner_key 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含 interaction payload 的完整桥事实键。"""
        return (
            1,
            self.source_space_id,
            self.use_event_hash,
            self.use_object_hash,
            self.source_timeline_seq,
            self.target_space_id,
            self.target_object_hash,
            *self.target_owner_key,
        )


# object-model: lifecycle; owner=train-context; cleanup=backend-close
class CrossMemoryUseRepository:
    """维护 Use event 到跨空间目标的唯一 append-only 派生行。"""

    def __init__(self, backend: StorageBackend) -> None:
        """绑定 backend 并注册桥表。"""
        self.backend = backend
        register_cross_memory_use_table(backend)

    def put(self, record: CrossMemoryUseRecord) -> tuple[CrossMemoryUseRecord, bool]:
        """幂等追加一条事实，既有同 event 内容漂移时拒绝。"""
        if not isinstance(record, CrossMemoryUseRecord):
            raise TypeError("cross Memory use record 类型错误")
        rows = self.backend.select(CROSS_MEMORY_USE_TABLE, where={
            "source_space_id": record.source_space_id,
            "use_event_hash": record.use_event_hash,
        })
        if rows:
            if len(rows) != 1:
                raise ValueError("cross Memory Use event 存在重复索引")
            existing = self._decode(rows[0])
            if existing != record:
                raise ValueError("cross Memory Use event 索引内容漂移")
            return existing, False
        self.backend.insert(CROSS_MEMORY_USE_TABLE, self._encode(record))
        return record, True

    def for_target(
            self,
            *,
            target_space_id: int,
            target_object_hash: int,
            ) -> tuple[CrossMemoryUseRecord, ...]:
        """按目标索引返回稳定有序物理事实；ACL 由上层完整身份 facade 执行。"""
        if (type(target_space_id) is not int or target_space_id <= 0
                or type(target_object_hash) is not int
                or target_object_hash <= 0):
            raise ValueError("cross Memory Use target 索引非法")
        records = tuple(
            self._decode(row)
            for row in self.backend.select(CROSS_MEMORY_USE_TABLE, where={
                "target_space_id": target_space_id,
                "target_object_hash": target_object_hash,
            })
        )
        return tuple(sorted(
            records,
            key=lambda item: (
                item.source_space_id,
                item.source_timeline_seq,
                item.use_event_hash,
            ),
        ))

    def all_records(self) -> tuple[CrossMemoryUseRecord, ...]:
        """供恢复审计读取全部桥行，不作为普通 ACL 查询入口。"""
        records = tuple(
            self._decode(row)
            for row in self.backend.select(CROSS_MEMORY_USE_TABLE, where=None)
        )
        keys = tuple(
            (item.source_space_id, item.use_event_hash) for item in records)
        if len(set(keys)) != len(keys):
            raise ValueError("cross Memory Use 全表含重复 event")
        return tuple(sorted(
            records,
            key=lambda item: (
                item.source_space_id,
                item.source_timeline_seq,
                item.use_event_hash,
            ),
        ))

    @staticmethod
    def _encode(record: CrossMemoryUseRecord) -> dict[str, int]:
        """把结构体转换为固定列。"""
        owner = record.target_owner_key
        return {
            "source_space_id": record.source_space_id,
            "use_event_hash": record.use_event_hash,
            "use_object_hash": record.use_object_hash,
            "source_timeline_seq": record.source_timeline_seq,
            "target_space_id": record.target_space_id,
            "target_object_hash": record.target_object_hash,
            "target_tenant_id": owner[0],
            "target_user_id": owner[1],
            "target_session_id": owner[2],
            "target_visibility": owner[3],
        }

    @staticmethod
    def _decode(row: dict[str, int]) -> CrossMemoryUseRecord:
        """从固定列恢复结构体并重新执行全部范围校验。"""
        expected = {
            "source_space_id", "use_event_hash", "use_object_hash",
            "source_timeline_seq", "target_space_id", "target_object_hash",
            "target_tenant_id", "target_user_id", "target_session_id",
            "target_visibility",
        }
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("cross Memory Use storage row 字段漂移")
        return CrossMemoryUseRecord(
            row["source_space_id"],
            row["use_event_hash"],
            row["use_object_hash"],
            row["source_timeline_seq"],
            row["target_space_id"],
            row["target_object_hash"],
            (
                row["target_tenant_id"],
                row["target_user_id"],
                row["target_session_id"],
                row["target_visibility"],
            ),
        )


__all__ = [
    "CROSS_MEMORY_USE_TABLE",
    "CrossMemoryUseRecord",
    "CrossMemoryUseRepository",
    "register_cross_memory_use_table",
]
