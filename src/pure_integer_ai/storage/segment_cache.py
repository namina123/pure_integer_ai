"""K-02 page-in/prefetch 热集与 clean/dirty 淘汰协议。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pure_integer_ai.storage.integer_codec import strict_integer_tuple
from pure_integer_ai.storage.sealed_segment import SegmentBudget, SegmentRecord


class SegmentCacheError(RuntimeError):
    """热集身份漂移、预算不足或脏对象未刷写。"""


@dataclass(frozen=True)
class CachedSegmentRecord:
    """热集中一条记录的描述、载荷、dirty 状态和逻辑访问序。"""

    descriptor_key: tuple[int, ...]
    record: SegmentRecord
    dirty: bool
    access_seq: int
    pin_count: int = 0

    def __post_init__(self) -> None:
        """核验描述身份、记录类型、dirty 标志、逻辑序和 pin 计数。"""
        strict_integer_tuple(
            self.descriptor_key, label="cached segment descriptor_key")
        if not isinstance(self.record, SegmentRecord):
            raise TypeError("cached segment record 类型错误")
        if type(self.dirty) is not bool:
            raise TypeError("cached segment dirty 必须是 bool")
        if type(self.access_seq) is not int or self.access_seq <= 0:
            raise ValueError("cached segment access_seq 必须是正严格整数")
        if type(self.pin_count) is not int or self.pin_count < 0:
            raise ValueError("cached segment pin_count 必须是非负严格整数")

    @property
    def cache_key(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """返回由逻辑描述和完整记录身份组成的热集键。"""
        return self.descriptor_key, self.record.record_key

    @property
    def pinned(self) -> bool:
        """判断当前记录是否仍被查询消费者持有。"""
        return self.pin_count > 0


FlushRecords = Callable[[tuple[CachedSegmentRecord, ...]], None]


class SegmentPageCache:
    """使用逻辑访问序和注入预算管理 query-scoped page-in 热集。"""

    def __init__(self, budget: SegmentBudget) -> None:
        """绑定对象/字节硬预算并创建空热集。"""
        if not isinstance(budget, SegmentBudget):
            raise TypeError("segment page cache budget 类型错误")
        self.budget = budget
        self._entries: dict[
            tuple[tuple[int, ...], tuple[int, ...]], CachedSegmentRecord
        ] = {}
        self._access_seq = 0
        self._size_bytes = 0

    @property
    def object_count(self) -> int:
        """返回当前热集对象数。"""
        return len(self._entries)

    @property
    def size_bytes(self) -> int:
        """返回当前热集记录规范字节数之和。"""
        return self._size_bytes

    @property
    def pinned_object_count(self) -> int:
        """返回当前至少持有一个 pin 的记录数。"""
        return sum(1 for item in self._entries.values() if item.pinned)

    @property
    def pinned_size_bytes(self) -> int:
        """返回所有 pinned 记录的规范字节数。"""
        return sum(
            item.record.size_bytes()
            for item in self._entries.values()
            if item.pinned
        )

    def page_in(
            self,
            descriptor_key: tuple[int, ...],
            records: tuple[SegmentRecord, ...],
            ) -> tuple[CachedSegmentRecord, ...]:
        """把冷层读取结果作为 clean 对象装入热集，并执行有界 clean evict。"""
        descriptor = strict_integer_tuple(
            descriptor_key, label="page cache descriptor_key")
        if (not isinstance(records, tuple)
                or any(not isinstance(item, SegmentRecord) for item in records)):
            raise TypeError("page cache records 必须是 SegmentRecord tuple")
        state = (dict(self._entries), self._access_seq, self._size_bytes)
        try:
            loaded = []
            for record in records:
                loaded.append(self._put(descriptor, record, dirty=False))
            return tuple(loaded)
        except BaseException:
            self._entries, self._access_seq, self._size_bytes = state
            raise

    def prefetch(
            self,
            descriptor_key: tuple[int, ...],
            records: tuple[SegmentRecord, ...],
            ) -> tuple[CachedSegmentRecord, ...]:
        """按与 page-in 相同的一致性和预算规则预取 clean 记录。"""
        return self.page_in(descriptor_key, records)

    def put_dirty(
            self,
            descriptor_key: tuple[int, ...],
            record: SegmentRecord,
            ) -> CachedSegmentRecord:
        """写入或替换一条 dirty 热记录，禁止无预算地堆积。"""
        descriptor = strict_integer_tuple(
            descriptor_key, label="dirty cache descriptor_key")
        if not isinstance(record, SegmentRecord):
            raise TypeError("dirty cache record 类型错误")
        return self._put(descriptor, record, dirty=True)

    def get(
            self,
            descriptor_key: tuple[int, ...],
            record_key: tuple[int, ...],
            ) -> CachedSegmentRecord | None:
        """读取一条热记录并推进确定性逻辑访问序。"""
        key = (
            strict_integer_tuple(
                descriptor_key, label="cache get descriptor_key"),
            strict_integer_tuple(record_key, label="cache get record_key"),
        )
        previous = self._entries.get(key)
        if previous is None:
            return None
        current = CachedSegmentRecord(
            previous.descriptor_key,
            previous.record,
            previous.dirty,
            self._next_access_seq(),
            previous.pin_count,
        )
        self._entries[key] = current
        return current

    def pin(
            self,
            descriptor_key: tuple[int, ...],
            record_key: tuple[int, ...],
            ) -> CachedSegmentRecord:
        """为一条已在热集中的记录增加查询级持有计数。"""
        key = (
            strict_integer_tuple(
                descriptor_key, label="cache pin descriptor_key"),
            strict_integer_tuple(record_key, label="cache pin record_key"),
        )
        previous = self._entries.get(key)
        if previous is None:
            raise SegmentCacheError("不能 pin 未 page-in 的记录")
        current = CachedSegmentRecord(
            previous.descriptor_key,
            previous.record,
            previous.dirty,
            self._next_access_seq(),
            previous.pin_count + 1,
        )
        self._entries[key] = current
        return current

    def unpin(
            self,
            descriptor_key: tuple[int, ...],
            record_key: tuple[int, ...],
            ) -> CachedSegmentRecord:
        """释放一条记录的一个查询级持有计数，拒绝无配对释放。"""
        key = (
            strict_integer_tuple(
                descriptor_key, label="cache unpin descriptor_key"),
            strict_integer_tuple(record_key, label="cache unpin record_key"),
        )
        previous = self._entries.get(key)
        if previous is None or previous.pin_count <= 0:
            raise SegmentCacheError("cache unpin 缺少配对 pin")
        current = CachedSegmentRecord(
            previous.descriptor_key,
            previous.record,
            previous.dirty,
            self._next_access_seq(),
            previous.pin_count - 1,
        )
        self._entries[key] = current
        return current

    def release_all_pins(self) -> int:
        """在 query 结束时一次释放全部 pin，并返回被释放的持有总数。"""
        released = 0
        for key in sorted(self._entries):
            previous = self._entries[key]
            if previous.pin_count <= 0:
                continue
            released += previous.pin_count
            self._entries[key] = CachedSegmentRecord(
                previous.descriptor_key,
                previous.record,
                previous.dirty,
                self._next_access_seq(),
                0,
            )
        return released

    def flush_dirty(self, flush: FlushRecords) -> int:
        """批量刷写全部 dirty 记录，成功后原地转为 clean 状态。"""
        if not callable(flush):
            raise TypeError("cache dirty flush 必须可调用")
        dirty = tuple(sorted(
            (item for item in self._entries.values() if item.dirty),
            key=lambda item: item.cache_key,
        ))
        if not dirty:
            return 0
        flush(dirty)
        for item in dirty:
            current = self._entries.get(item.cache_key)
            if current != item:
                raise SegmentCacheError("dirty flush 期间热集状态发生漂移")
            self._entries[item.cache_key] = CachedSegmentRecord(
                item.descriptor_key,
                item.record,
                False,
                self._next_access_seq(),
                item.pin_count,
            )
        return len(dirty)

    def clear(self, *, flush: FlushRecords | None = None) -> int:
        """清空 query 热集；dirty 必须先刷写，pinned 必须先释放。"""
        if any(item.pinned for item in self._entries.values()):
            raise SegmentCacheError("清空热集前必须释放全部 pin")
        if any(item.dirty for item in self._entries.values()):
            if flush is None:
                raise SegmentCacheError("清空含 dirty 记录的热集必须提供 flush")
            self.flush_dirty(flush)
        return self.evict(tuple(sorted(self._entries)))

    def evict(
            self,
            keys: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
            *,
            flush: FlushRecords | None = None,
            ) -> int:
        """批量淘汰指定记录；任何 dirty 对象必须先由回调完整刷写。"""
        if not isinstance(keys, tuple):
            raise TypeError("cache evict keys 必须是 tuple")
        selected = []
        normalized = []
        for descriptor_key, record_key in keys:
            key = (
                strict_integer_tuple(
                    descriptor_key, label="cache evict descriptor_key"),
                strict_integer_tuple(
                    record_key, label="cache evict record_key"),
            )
            normalized.append(key)
            item = self._entries.get(key)
            if item is not None:
                selected.append(item)
        dirty = tuple(item for item in selected if item.dirty)
        if any(item.pinned for item in selected):
            raise SegmentCacheError("pinned 对象不得淘汰")
        if dirty:
            if flush is None:
                raise SegmentCacheError("dirty 对象淘汰前必须提供 flush")
            flush(dirty)
        removed = 0
        for key in normalized:
            item = self._entries.pop(key, None)
            if item is None:
                continue
            self._size_bytes -= item.record.size_bytes()
            removed += 1
        return removed

    def evict_clean(self, object_limit: int) -> int:
        """按最早逻辑访问序淘汰至多指定数量的 unpinned clean 记录。"""
        if type(object_limit) is not int or object_limit < 0:
            raise ValueError("clean evict object_limit 必须是非负严格整数")
        candidates = sorted(
            (item for item in self._entries.values()
             if not item.dirty and not item.pinned),
            key=lambda item: (item.access_seq, item.cache_key),
        )[:object_limit]
        return self.evict(tuple(item.cache_key for item in candidates))

    def snapshot(self) -> tuple[CachedSegmentRecord, ...]:
        """按完整 cache key 返回当前热集确定性快照。"""
        return tuple(self._entries[key] for key in sorted(self._entries))

    def _put(
            self,
            descriptor_key: tuple[int, ...],
            record: SegmentRecord,
            *,
            dirty: bool,
            ) -> CachedSegmentRecord:
        """统一执行新增、幂等 page-in、dirty 替换和预算回收。"""
        key = (descriptor_key, record.record_key)
        previous = self._entries.get(key)
        previous_size = 0 if previous is None else previous.record.size_bytes()
        if previous is not None and not dirty and previous.record != record:
            raise SegmentCacheError("clean page-in 命中不同载荷")
        new_size = record.size_bytes()
        object_delta = 1 if previous is None else 0
        byte_delta = new_size - previous_size
        self._make_room(object_delta, byte_delta, protected_key=key)
        current = CachedSegmentRecord(
            descriptor_key,
            record,
            dirty or (False if previous is None else previous.dirty),
            self._next_access_seq(),
            0 if previous is None else previous.pin_count,
        )
        self._entries[key] = current
        self._size_bytes += byte_delta
        return current

    def _make_room(
            self,
            object_delta: int,
            byte_delta: int,
            *,
            protected_key: tuple[tuple[int, ...], tuple[int, ...]],
            ) -> None:
        """只淘汰 clean 最旧项，直到新增对象同时满足两类硬预算。"""
        while (len(self._entries) + object_delta > self.budget.object_limit
               or self._size_bytes + byte_delta > self.budget.byte_limit):
            candidates = sorted(
                (item for key, item in self._entries.items()
                 if key != protected_key and not item.dirty and not item.pinned),
                key=lambda item: (item.access_seq, item.cache_key),
            )
            if not candidates:
                raise SegmentCacheError("热集预算不足且没有可淘汰 clean 对象")
            self.evict((candidates[0].cache_key,))

    def _next_access_seq(self) -> int:
        """推进进程内逻辑访问序，不读取墙钟。"""
        self._access_seq += 1
        return self._access_seq


__all__ = [
    "CachedSegmentRecord",
    "FlushRecords",
    "SegmentCacheError",
    "SegmentPageCache",
]
