"""K-04 query-scoped page-in、预取、pin、flush 和 epoch 隔离。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, runtime_checkable

from pure_integer_ai.storage.integer_codec import strict_integer_tuple
from pure_integer_ai.storage.edge_budget import EdgeMetricObservation
from pure_integer_ai.storage.sealed_segment import (
    SegmentBudget,
    SegmentBudgetExceeded,
    SegmentRecord,
)
from pure_integer_ai.storage.segment_cache import (
    CachedSegmentRecord,
    FlushRecords,
    SegmentCacheError,
    SegmentPageCache,
)
from pure_integer_ai.storage.tiered_segment_store import (
    BoundedSegmentReader,
    StablePageResult,
    TieredSegmentStore,
)


class QueryHotSetError(RuntimeError):
    """query 热集的 scope、epoch、预算或关闭协议不一致。"""


class QueryHotSetBudgetExceeded(QueryHotSetError):
    """pinned/dirty 工作集已占满预算，无法继续精确 page-in。"""


QUERY_HOT_METRIC_PAGE_FAULTS = (1, 1)
QUERY_HOT_METRIC_PREFETCHED_PAGES = (1, 2)
QUERY_HOT_METRIC_PAGE_IN_RECORDS = (1, 3)
QUERY_HOT_METRIC_COLD_READ_BYTES = (1, 4)
QUERY_HOT_METRIC_CACHE_HITS = (1, 5)
QUERY_HOT_METRIC_CLEAN_EVICTIONS = (1, 6)
QUERY_HOT_METRIC_DIRTY_FLUSHES = (1, 7)
QUERY_HOT_METRIC_PEAK_OBJECTS = (1, 8)
QUERY_HOT_METRIC_PEAK_BYTES = (1, 9)
QUERY_HOT_METRIC_RELEASED_PINS = (1, 10)
QUERY_HOT_METRIC_OMITTED_FAULT_REPORTS = (1, 11)


@dataclass(frozen=True)
class QueryPrefetchContext:
    """交给注入式预取策略的 query 进度、I/O 和剩余热集容量。"""

    consumed_pages: int
    page_faults: int
    prefetched_pages: int
    page_in_records: int
    cold_read_bytes: int
    cache_objects: int
    cache_bytes: int
    available_objects: int
    available_bytes: int

    def __post_init__(self) -> None:
        """核验预取决策上下文只包含非负严格整数计数。"""
        for label, value in (
                ("consumed_pages", self.consumed_pages),
                ("page_faults", self.page_faults),
                ("prefetched_pages", self.prefetched_pages),
                ("page_in_records", self.page_in_records),
                ("cold_read_bytes", self.cold_read_bytes),
                ("cache_objects", self.cache_objects),
                ("cache_bytes", self.cache_bytes),
                ("available_objects", self.available_objects),
                ("available_bytes", self.available_bytes)):
            if type(value) is not int or value < 0:
                raise ValueError(f"query prefetch {label} 必须是非负严格整数")


@runtime_checkable
class QueryPrefetchPolicy(Protocol):
    """由设备 profile 注入的开放续页预取决策协议。"""

    def should_prefetch(self, context: QueryPrefetchContext) -> bool:
        """依据当前纯整数物理状态决定是否提前读取下一页。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回策略配置的稳定纯整数身份，不包含 query 可变计数。"""
        ...


@dataclass(frozen=True)
class QueryHotSetPolicy:
    """由设备 profile 注入的 cache、分页、预取和报告上限。"""

    cache_budget: SegmentBudget
    page_budget: SegmentBudget
    prefetch_policy: QueryPrefetchPolicy
    fault_report_limit: int

    def __post_init__(self) -> None:
        """核验两类硬预算以及非负预取/报告配置。"""
        if not isinstance(self.cache_budget, SegmentBudget):
            raise TypeError("query hot-set cache_budget 类型错误")
        if not isinstance(self.page_budget, SegmentBudget):
            raise TypeError("query hot-set page_budget 类型错误")
        if not isinstance(self.prefetch_policy, QueryPrefetchPolicy):
            raise TypeError("query hot-set prefetch_policy 协议错误")
        self.prefetch_state_key()
        if (type(self.fault_report_limit) is not int
                or self.fault_report_limit < 0):
            raise ValueError("query hot-set fault_report_limit 必须是非负严格整数")
        if self.page_budget.object_limit > self.cache_budget.object_limit:
            raise ValueError("page 对象预算不得大于 cache 对象预算")
        if self.page_budget.byte_limit > self.cache_budget.byte_limit:
            raise ValueError("page 字节预算不得大于 cache 字节预算")

    def prefetch_state_key(self) -> tuple[int, ...]:
        """读取并核验注入式预取策略的稳定纯整数身份。"""
        key = strict_integer_tuple(
            self.prefetch_policy.state_key(),
            label="query hot-set prefetch policy state_key",
        )
        if not key:
            raise ValueError("query hot-set prefetch policy state_key 不得为空")
        return key


@dataclass(frozen=True)
class QueryPageFaultReport:
    """一次冷页读取的固定 read view、范围、规模和预取标记。"""

    publish_epoch: int
    descriptor_key: tuple[int, ...]
    lower_key: tuple[int, ...] | None
    upper_key: tuple[int, ...] | None
    record_count: int
    record_bytes: int
    prefetched: bool

    def __post_init__(self) -> None:
        """核验缺页报告只保留稳定整数范围和非负规模。"""
        if type(self.publish_epoch) is not int or self.publish_epoch <= 0:
            raise ValueError("page fault publish_epoch 必须是正严格整数")
        strict_integer_tuple(
            self.descriptor_key, label="page fault descriptor_key")
        for label, value in (
                ("lower_key", self.lower_key),
                ("upper_key", self.upper_key)):
            if value is not None:
                strict_integer_tuple(value, label=f"page fault {label}")
        if (type(self.record_count) is not int or self.record_count < 0
                or type(self.record_bytes) is not int or self.record_bytes < 0):
            raise ValueError("page fault 规模必须是非负严格整数")
        if type(self.prefetched) is not bool:
            raise TypeError("page fault prefetched 必须是 bool")


@dataclass(frozen=True)
class QueryHotSetMetrics:
    """一次 query 的逐维 page/cache/flush 物理计数。"""

    page_faults: int
    prefetched_pages: int
    page_in_records: int
    cold_read_bytes: int
    cache_hits: int
    clean_evictions: int
    dirty_flushes: int
    peak_hot_objects: int
    peak_hot_bytes: int
    released_pins: int
    omitted_fault_reports: int

    def observations(self) -> tuple[EdgeMetricObservation, ...]:
        """把真实 query 计数转换为开放逐维边缘预算观测。"""
        return tuple(EdgeMetricObservation(key, value) for key, value in (
            (QUERY_HOT_METRIC_PAGE_FAULTS, self.page_faults),
            (QUERY_HOT_METRIC_PREFETCHED_PAGES, self.prefetched_pages),
            (QUERY_HOT_METRIC_PAGE_IN_RECORDS, self.page_in_records),
            (QUERY_HOT_METRIC_COLD_READ_BYTES, self.cold_read_bytes),
            (QUERY_HOT_METRIC_CACHE_HITS, self.cache_hits),
            (QUERY_HOT_METRIC_CLEAN_EVICTIONS, self.clean_evictions),
            (QUERY_HOT_METRIC_DIRTY_FLUSHES, self.dirty_flushes),
            (QUERY_HOT_METRIC_PEAK_OBJECTS, self.peak_hot_objects),
            (QUERY_HOT_METRIC_PEAK_BYTES, self.peak_hot_bytes),
            (QUERY_HOT_METRIC_RELEASED_PINS, self.released_pins),
            (QUERY_HOT_METRIC_OMITTED_FAULT_REPORTS,
             self.omitted_fault_reports),
        ))


class QuerySegmentHotSet:
    """在固定 K-02 reader epoch 内管理一个 descriptor 的查询热集。"""

    def __init__(
            self,
            store: TieredSegmentStore,
            *,
            reader_key: tuple[int, ...],
            descriptor_key: tuple[int, ...],
            policy: QueryHotSetPolicy,
            flush: FlushRecords | None = None,
            ) -> None:
        """打开稳定 reader，并创建不与其他 query 共享的 page cache。"""
        if not isinstance(store, TieredSegmentStore):
            raise TypeError("query hot-set store 类型错误")
        if not isinstance(policy, QueryHotSetPolicy):
            raise TypeError("query hot-set policy 类型错误")
        if flush is not None and not callable(flush):
            raise TypeError("query hot-set flush 必须可调用")
        self.store = store
        self.reader_key = strict_integer_tuple(
            reader_key, label="query hot-set reader_key")
        self.descriptor_key = strict_integer_tuple(
            descriptor_key, label="query hot-set descriptor_key")
        self.policy = policy
        self.flush_callback = flush
        self.cache = SegmentPageCache(policy.cache_budget)
        self.reader: BoundedSegmentReader = store.open_reader(
            self.reader_key, self.descriptor_key)
        self._closed = False
        self._page_faults = 0
        self._prefetched_pages = 0
        self._page_in_records = 0
        self._cold_read_bytes = 0
        self._cache_hits = 0
        self._clean_evictions = 0
        self._dirty_flushes = 0
        self._peak_hot_objects = 0
        self._peak_hot_bytes = 0
        self._released_pins = 0
        self._omitted_fault_reports = 0
        self._consumed_pages = 0
        self._fault_reports: list[QueryPageFaultReport] = []

    @property
    def publish_epoch(self) -> int:
        """返回当前 query 固定的 location manifest epoch。"""
        return self.reader.manifest.publish_epoch

    @property
    def stale(self) -> bool:
        """判断 store 是否已发布更新 epoch；旧 reader 仍可完成当前 query。"""
        manifest = self.store.current_manifest()
        return (
            manifest is None
            or manifest.publish_epoch != self.reader.manifest.publish_epoch
        )

    @property
    def fault_reports(self) -> tuple[QueryPageFaultReport, ...]:
        """返回受上限约束的逐页缺页报告。"""
        return tuple(self._fault_reports)

    def require_current_epoch(self) -> None:
        """拒绝把已固定的旧 query cache 复用于新 location epoch。"""
        self._ensure_open()
        if self.stale:
            raise QueryHotSetError("query hot-set epoch 已失效，必须新建上下文")

    def iter_range(
            self,
            *,
            lower_key: tuple[int, ...] | None = None,
            upper_key: tuple[int, ...] | None = None,
            ) -> Iterator[CachedSegmentRecord]:
        """沿稳定 continuation 流式读取范围，并只保留 pinned/dirty 工作集。"""
        self._ensure_open()
        lower = None if lower_key is None else strict_integer_tuple(
            lower_key, label="query hot-set lower_key")
        upper = None if upper_key is None else strict_integer_tuple(
            upper_key, label="query hot-set upper_key")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("query hot-set range 不能反向")
        continuation = None
        prepared: StablePageResult | None = None
        while True:
            if prepared is None:
                result = self._read_page(
                    lower, upper, continuation, prefetch=False)
            else:
                result = prepared
                prepared = None
            for record in result.records:
                cached = self.cache.get(
                    self.descriptor_key, record.record_key)
                if cached is None:
                    raise QueryHotSetError("page-in 后记录未保留在 query cache")
                yield cached
            self._clean_evictions += self.cache.evict_clean(
                self.cache.object_count)
            self._consumed_pages += 1
            if not result.has_more:
                break
            continuation = result.continuation
            if self._should_prefetch():
                prepared = self._read_page(
                    lower, upper, continuation, prefetch=True)

    def get(
            self,
            record_key: tuple[int, ...],
            ) -> CachedSegmentRecord | None:
        """读取已驻留记录并记录真实 cache hit，不触发隐式冷扫。"""
        self._ensure_open()
        cached = self.cache.get(self.descriptor_key, record_key)
        if cached is not None:
            self._cache_hits += 1
        return cached

    def pin(self, record_key: tuple[int, ...]) -> CachedSegmentRecord:
        """持有一个已 page-in 记录，使后续 clean evict 不会移除它。"""
        self._ensure_open()
        current = self.cache.pin(self.descriptor_key, record_key)
        self._sample_peak()
        return current

    def unpin(self, record_key: tuple[int, ...]) -> CachedSegmentRecord:
        """释放一个已配对 pin，使记录重新具备 clean evict 资格。"""
        self._ensure_open()
        return self.cache.unpin(self.descriptor_key, record_key)

    def put_dirty(self, record: SegmentRecord) -> CachedSegmentRecord:
        """把调用方生成的同 descriptor 新记录登记为 dirty query 状态。"""
        self._ensure_open()
        current = self.cache.put_dirty(self.descriptor_key, record)
        self._sample_peak()
        return current

    def flush_dirty(self) -> int:
        """使用安装时回调批量刷写全部 dirty 记录并转为 clean。"""
        self._ensure_open()
        if self.flush_callback is None:
            if any(item.dirty for item in self.cache.snapshot()):
                raise QueryHotSetError("query hot-set dirty flush 缺少回调")
            return 0
        flushed = self.cache.flush_dirty(self.flush_callback)
        self._dirty_flushes += flushed
        return flushed

    def close(self) -> None:
        """释放全部 pin，完整刷写 dirty，清空 cache 并释放 reader lease。"""
        if self._closed:
            return
        self._released_pins += self.cache.release_all_pins()
        self.flush_dirty()
        self._clean_evictions += self.cache.clear()
        self.reader.close()
        self._closed = True

    def metrics(self) -> QueryHotSetMetrics:
        """返回不进入 canonical 身份的 query 物理计数快照。"""
        return QueryHotSetMetrics(
            self._page_faults,
            self._prefetched_pages,
            self._page_in_records,
            self._cold_read_bytes,
            self._cache_hits,
            self._clean_evictions,
            self._dirty_flushes,
            self._peak_hot_objects,
            self._peak_hot_bytes,
            self._released_pins,
            self._omitted_fault_reports,
        )

    def _read_page(
            self,
            lower_key: tuple[int, ...] | None,
            upper_key: tuple[int, ...] | None,
            continuation,
            *,
            prefetch: bool,
            ) -> StablePageResult:
        """按 pinned/dirty 剩余容量读取一页，并形成有界缺页报告。"""
        budget = self._available_page_budget()
        try:
            result = self.reader.page(
                budget=budget,
                lower_key=lower_key,
                upper_key=upper_key,
                continuation=continuation,
            )
            self.cache.prefetch(self.descriptor_key, result.records)
        except (SegmentBudgetExceeded, SegmentCacheError) as exc:
            raise QueryHotSetBudgetExceeded(
                "query 热集剩余预算不足以恢复下一完整记录") from exc
        record_bytes = sum(item.size_bytes() for item in result.records)
        self._page_faults += 1
        self._prefetched_pages += int(prefetch)
        self._page_in_records += len(result.records)
        self._cold_read_bytes += record_bytes
        self._record_fault(QueryPageFaultReport(
            result.publish_epoch,
            self.descriptor_key,
            lower_key,
            upper_key,
            len(result.records),
            record_bytes,
            prefetch,
        ))
        self._sample_peak()
        return result

    def _available_page_budget(self) -> SegmentBudget:
        """先释放所有 unpinned clean，再按剩余对象/字节容量裁剪单页预算。"""
        self._clean_evictions += self.cache.evict_clean(
            self.cache.object_count)
        object_limit = min(
            self.policy.page_budget.object_limit,
            self.policy.cache_budget.object_limit - self.cache.object_count,
        )
        byte_limit = min(
            self.policy.page_budget.byte_limit,
            self.policy.cache_budget.byte_limit - self.cache.size_bytes,
        )
        if object_limit <= 0 or byte_limit <= 0:
            raise QueryHotSetBudgetExceeded(
                "query 热集已被 pinned/dirty 记录占满")
        return SegmentBudget(object_limit, byte_limit)

    def _should_prefetch(self) -> bool:
        """用当前物理计数调用注入策略，并拒绝非布尔或配置漂移结果。"""
        self.policy.prefetch_state_key()
        available_objects = max(
            0, self.policy.cache_budget.object_limit - self.cache.object_count)
        available_bytes = max(
            0, self.policy.cache_budget.byte_limit - self.cache.size_bytes)
        decision = self.policy.prefetch_policy.should_prefetch(
            QueryPrefetchContext(
                self._consumed_pages,
                self._page_faults,
                self._prefetched_pages,
                self._page_in_records,
                self._cold_read_bytes,
                self.cache.object_count,
                self.cache.size_bytes,
                available_objects,
                available_bytes,
            )
        )
        if type(decision) is not bool:
            raise TypeError("query prefetch policy 必须返回严格 bool")
        return decision

    def _record_fault(self, report: QueryPageFaultReport) -> None:
        """在注入上限内保留逐页报告，超出部分只累计数量。"""
        if len(self._fault_reports) < self.policy.fault_report_limit:
            self._fault_reports.append(report)
        else:
            self._omitted_fault_reports += 1

    def _sample_peak(self) -> None:
        """更新 query cache 自身的对象数和规范字节峰值。"""
        self._peak_hot_objects = max(
            self._peak_hot_objects, self.cache.object_count)
        self._peak_hot_bytes = max(
            self._peak_hot_bytes, self.cache.size_bytes)

    def _ensure_open(self) -> None:
        """拒绝 query 关闭后继续读取或修改其热集。"""
        if self._closed:
            raise QueryHotSetError("query hot-set 已关闭")


__all__ = [
    "QUERY_HOT_METRIC_CACHE_HITS",
    "QUERY_HOT_METRIC_CLEAN_EVICTIONS",
    "QUERY_HOT_METRIC_COLD_READ_BYTES",
    "QUERY_HOT_METRIC_DIRTY_FLUSHES",
    "QUERY_HOT_METRIC_OMITTED_FAULT_REPORTS",
    "QUERY_HOT_METRIC_PAGE_FAULTS",
    "QUERY_HOT_METRIC_PAGE_IN_RECORDS",
    "QUERY_HOT_METRIC_PEAK_BYTES",
    "QUERY_HOT_METRIC_PEAK_OBJECTS",
    "QUERY_HOT_METRIC_PREFETCHED_PAGES",
    "QUERY_HOT_METRIC_RELEASED_PINS",
    "QueryHotSetBudgetExceeded",
    "QueryHotSetError",
    "QueryHotSetMetrics",
    "QueryHotSetPolicy",
    "QueryPageFaultReport",
    "QueryPrefetchContext",
    "QueryPrefetchPolicy",
    "QuerySegmentHotSet",
]
