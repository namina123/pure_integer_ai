"""Memory 事件摄入单元的 staged、物化、activation、恢复与整批 rollback。"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.identity import OwnerScope, SourceRef
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MemoryHypothesisAggregateIndex,
)
from pure_integer_ai.cognition.shared.memory_event import MemoryEvent
from pure_integer_ai.cognition.shared.memory_event_log import (
    MaterializedMemoryEvent,
    MemoryEventLog,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.storage.memory_batch import (
    MEMORY_BATCH_CORE_DEPENDENCY_KEY,
    MEMORY_BATCH_SOURCE_DEPENDENCY_KEY,
    MemoryBatchIntegrityError,
    MemoryBatchReceiptStore,
    MemoryBatchVisibility,
    MemoryEventBatchLink,
    MemoryEventBatchLinkStore,
    StagedMemoryBatch,
    memory_batch_hash,
)
from pure_integer_ai.storage import (
    build_storage_role_registry,
    build_tiered_segment_store,
)
from pure_integer_ai.storage.placement import TemperatureProfile
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_TABLE,
    MemoryEventRecord,
)
from pure_integer_ai.storage.assertion_identity import IDENTITY_MEMORY_EVENT
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from pure_integer_ai.storage.segment_dependency import SegmentDependency


FAULT_MEMORY_BATCH_AFTER_STAGE = 1
FAULT_MEMORY_BATCH_AFTER_LINK = 2
FAULT_MEMORY_BATCH_AFTER_EVENT = 3
FAULT_MEMORY_BATCH_AFTER_PROJECTION = 4
FAULT_MEMORY_BATCH_BEFORE_ACTIVATION = 5
FAULT_MEMORY_BATCH_AFTER_ACTIVATION = 6
FAULT_MEMORY_BATCH_AFTER_ROLLBACK_PROJECTION = 7
FAULT_MEMORY_BATCH_AFTER_ROLLBACK_RECEIPT = 8
FAULT_MEMORY_GROUP_AFTER_UNIT = 9
FAULT_MEMORY_GROUP_AFTER_PROJECTION = 10
FAULT_MEMORY_GROUP_AFTER_COMMIT = 11


@runtime_checkable
class MemoryBatchFaultInjector(Protocol):
    """在 M-10 批次承重边界注入故障的最小协议。"""

    def hit(self, point: int, context: dict[str, int]) -> None:
        """观察边界；测试需要中断时直接抛出异常。"""
        ...


def _hit(
        injector: MemoryBatchFaultInjector | None,
        point: int,
        context: dict[str, int],
        ) -> None:
    """调用可选故障注入器并复制纯整数上下文。"""
    if injector is None:
        return
    if not isinstance(injector, MemoryBatchFaultInjector):
        raise TypeError("memory batch fault injector 协议错误")
    injector.hit(point, dict(context))


@dataclass(frozen=True)
class MemoryBatchPublishResult:
    """一个摄入单元 activation 后的稳定身份和物化事件。"""

    batch_hash: int
    materialized: tuple[MaterializedMemoryEvent, ...]


@dataclass(frozen=True)
class MemoryBatchRuntimeConfig:
    """装配两个 Memory 空间批次 runtime 所需的 K-02 注入配置。"""

    temperature_profile: TemperatureProfile
    tier_key: tuple[int, ...]
    core_dependency: SegmentDependency
    read_budget: SegmentBudget
    write_budget: SegmentBudget

    def __post_init__(self) -> None:
        """核验温层、Core 依赖和读写预算均由调用方完整提供。"""
        if not isinstance(self.temperature_profile, TemperatureProfile):
            raise TypeError("temperature_profile 类型错误")
        if not self.temperature_profile.has(self.tier_key):
            raise ValueError("tier_key 不属于 temperature profile")
        if (not isinstance(self.core_dependency, SegmentDependency)
                or self.core_dependency.descriptor_key
                != MEMORY_BATCH_CORE_DEPENDENCY_KEY):
            raise ValueError("core_dependency descriptor 非法")
        if (not isinstance(self.read_budget, SegmentBudget)
                or not isinstance(self.write_budget, SegmentBudget)):
            raise TypeError("read/write budget 必须是 SegmentBudget")


class MemoryBatchRuntime:
    """把 K-02 receipt、MemoryEventLog 和 M-04 派生投影组合为单元提交。"""

    def __init__(
            self,
            event_log: MemoryEventLog,
            aggregates: MemoryHypothesisAggregateIndex,
            visibility: MemoryBatchVisibility,
            receipts: MemoryBatchReceiptStore,
            *,
            core_dependency: SegmentDependency,
            write_budget: SegmentBudget,
            ) -> None:
        """绑定同一 Memory 空间，并安装可见性与注入式批次预算。"""
        if not isinstance(event_log, MemoryEventLog):
            raise TypeError("event_log 类型错误")
        if not isinstance(aggregates, MemoryHypothesisAggregateIndex):
            raise TypeError("aggregates 类型错误")
        if aggregates.event_log is not event_log:
            raise ValueError("aggregate 与 event log 不一致")
        if visibility.receipts is not receipts:
            raise ValueError("visibility 与 receipt store 不一致")
        if (not isinstance(core_dependency, SegmentDependency)
                or core_dependency.descriptor_key
                != MEMORY_BATCH_CORE_DEPENDENCY_KEY):
            raise ValueError("core_dependency descriptor 非法")
        if not isinstance(write_budget, SegmentBudget):
            raise TypeError("write_budget 必须是 SegmentBudget")
        self.event_log = event_log
        self.aggregates = aggregates
        self.visibility = visibility
        self.receipts = receipts
        self.core_dependency = core_dependency
        self.write_budget = write_budget
        event_log.attach_batch_visibility(visibility)

    def publish(
            self,
            source: SourceRef,
            batch_id: int,
            events: tuple[MemoryEvent, ...],
            *,
            source_dependency: SegmentDependency,
            fault_injector: MemoryBatchFaultInjector | None = None,
            ) -> MemoryBatchPublishResult:
        """封存确定事件集，补齐物理行和投影，最后发布 activation receipt。"""
        batch = self._batch(
            source,
            batch_id,
            events,
            source_dependency=source_dependency,
        )
        self._validate_group_write(batch)
        previous = self.receipts.staged(batch.batch_hash)
        if previous is not None and previous != batch:
            raise MemoryBatchIntegrityError("同一摄入单元 staged 内容漂移")
        if self.receipts.is_rolled_back(batch.batch_hash):
            raise MemoryBatchIntegrityError("已 rollback 摄入单元不得重新 activation")
        self.receipts.stage(batch)
        _hit(fault_injector, FAULT_MEMORY_BATCH_AFTER_STAGE, {
            "batch_hash": batch.batch_hash,
            "event_count": batch.event_count,
        })
        return self._complete(batch, fault_injector=fault_injector)

    def recover_unit(
            self,
            source: SourceRef,
            batch_id: int,
            *,
            source_dependency: SegmentDependency,
            fault_injector: MemoryBatchFaultInjector | None = None,
            ) -> MemoryBatchPublishResult | None:
        """若指定来源单元已有 staged segment，则不重跑 parser 直接 roll-forward。"""
        batch_hash = memory_batch_hash(
            self.event_log.memory_space_identity.stable_key(),
            source.stable_key(),
            batch_id,
        )
        staged = self.receipts.staged(batch_hash)
        if staged is None:
            if (self.receipts.has_group_intent(batch_id)
                    and (self.receipts.is_group_committed(batch_id)
                         or self.receipts.is_group_rolled_back(batch_id))):
                raise MemoryBatchIntegrityError(
                    "已关闭来源批次不得追加新摄入单元")
            return None
        self._validate_group_write(staged)
        self._validate_stage_dependencies(staged, source_dependency)
        if self.receipts.is_rolled_back(batch_hash):
            return None
        return self._complete(staged, fault_injector=fault_injector)

    def recover_pending(
            self,
            *,
            fault_injector: MemoryBatchFaultInjector | None = None,
            ) -> tuple[MemoryBatchPublishResult, ...]:
        """扫描当前 Memory 空间全部未完成 staged 单元并确定性 roll-forward。"""
        results = []
        for staged in self.receipts.staged_batches():
            if staged.space_key != self.event_log.memory_space_identity.stable_key():
                continue
            if (self.receipts.is_active(staged.batch_hash)
                    or self.receipts.is_rolled_back(staged.batch_hash)):
                continue
            if self.receipts.is_group_rolled_back(staged.batch_id):
                continue
            self._validate_group_write(staged)
            if staged.dependencies[0] != self.core_dependency:
                raise MemoryBatchIntegrityError(
                    "pending batch Core dependency 与当前 runtime 漂移")
            results.append(self._complete(
                staged, fault_injector=fault_injector))
        return tuple(results)

    def rollback_batch(
            self,
            batch_id: int,
            *,
            owner: OwnerScope | None = None,
            fault_injector: MemoryBatchFaultInjector | None = None,
            ) -> tuple[int, ...]:
        """隐藏当前空间指定来源批次的全部活动单元，并从剩余事件重建派生表。"""
        if type(batch_id) is not int or batch_id <= 0:
            raise ValueError("rollback batch_id 必须是正严格整数")
        if owner is not None and not isinstance(owner, OwnerScope):
            raise TypeError("owner 必须是 OwnerScope 或 None")
        candidates = []
        for batch_hash in self.visibility.links.hashes_for_source_batch(
                batch_id, space_id=self.event_log.memory_space_id):
            staged = self.receipts.staged(batch_hash)
            if staged is None:
                raise MemoryBatchIntegrityError(
                    "batch link 缺少 staged segment")
            if (staged.space_key
                    != self.event_log.memory_space_identity.stable_key()
                    or not self.receipts.is_active(staged.batch_hash)
                    or self.receipts.is_rolled_back(staged.batch_hash)):
                continue
            source = SourceRef.from_stable_key(staged.source_key)
            if owner is not None and source.owner != owner:
                continue
            candidates.append((staged, source))
        if not candidates:
            return ()
        owners = tuple(sorted({source.owner for _, source in candidates}))
        try:
            with ExitStack() as stack:
                for staged, _ in candidates:
                    stack.enter_context(
                        self.visibility.suppress(staged.batch_hash))
                for current_owner in owners:
                    self.aggregates.rebuild_all(
                        access=self._access(current_owner))
            _hit(fault_injector, FAULT_MEMORY_BATCH_AFTER_ROLLBACK_PROJECTION, {
                "batch_id": batch_id,
                "unit_count": len(candidates),
            })
            for staged, _ in candidates:
                self.receipts.rollback(staged.batch_hash)
            _hit(fault_injector, FAULT_MEMORY_BATCH_AFTER_ROLLBACK_RECEIPT, {
                "batch_id": batch_id,
                "unit_count": len(candidates),
            })
        except BaseException:
            for current_owner in owners:
                self.aggregates.rebuild_all(
                    access=self._access(current_owner))
            raise
        return tuple(sorted(staged.batch_hash for staged, _ in candidates))

    def staged_owners(
            self,
            batch_id: int,
            ) -> tuple[OwnerScope, ...]:
        """返回当前 Memory 空间指定来源批次涉及的稳定 owner 集。"""
        owners = {
            SourceRef.from_stable_key(staged.source_key).owner
            for batch_hash in self.visibility.links.hashes_for_source_batch(
                batch_id, space_id=self.event_log.memory_space_id)
            for staged in (self.receipts.staged(batch_hash),)
            if staged is not None
        }
        return tuple(sorted(owners, key=lambda item: item.stable_key()))

    def _batch(
            self,
            source: SourceRef,
            batch_id: int,
            events: tuple[MemoryEvent, ...],
            *,
            source_dependency: SegmentDependency,
            ) -> StagedMemoryBatch:
        """从完整来源和有序事件形成稳定 staged batch，并执行预算核验。"""
        if not isinstance(source, SourceRef):
            raise TypeError("source 必须是 SourceRef")
        if (not isinstance(events, tuple)
                or not events
                or any(not isinstance(item, MemoryEvent) for item in events)):
            raise TypeError("events 必须是非空 MemoryEvent tuple")
        if (not isinstance(source_dependency, SegmentDependency)
                or source_dependency.descriptor_key
                != MEMORY_BATCH_SOURCE_DEPENDENCY_KEY):
            raise ValueError("source_dependency descriptor 非法")
        if (len(source_dependency.version_key) < 2
                or source_dependency.version_key[-1] != batch_id):
            raise MemoryBatchIntegrityError(
                "source_dependency 未绑定当前来源批次")
        batch_hash = memory_batch_hash(
            self.event_log.memory_space_identity.stable_key(),
            source.stable_key(),
            batch_id,
        )
        staged = StagedMemoryBatch(
            batch_hash,
            self.event_log.memory_space_identity.stable_key(),
            source.stable_key(),
            batch_id,
            tuple(item.stable_key() for item in events),
            (self.core_dependency, source_dependency),
        )
        segment = staged.to_segment()
        if len(segment.records) > self.write_budget.object_limit:
            raise MemoryBatchIntegrityError("staged batch 超过对象数预算")
        if segment.size_bytes > self.write_budget.byte_limit:
            raise MemoryBatchIntegrityError("staged batch 超过字节预算")
        return staged

    def _validate_stage_dependencies(
            self,
            staged: StagedMemoryBatch,
            source_dependency: SegmentDependency,
            ) -> None:
        """核验重试时 Core 与 Source 依赖未发生漂移。"""
        if staged.dependencies != (
                self.core_dependency, source_dependency):
            raise MemoryBatchIntegrityError(
                "staged batch dependency 与当前来源状态漂移")

    def _validate_group_write(self, staged: StagedMemoryBatch) -> None:
        """拒绝向已提交或已回滚的组批次追加未登记单元。"""
        if not self.receipts.has_group_intent(staged.batch_id):
            return
        if self.receipts.is_group_rolled_back(staged.batch_id):
            raise MemoryBatchIntegrityError("已 rollback 来源批次不得重新摄入")
        if not self.receipts.is_group_committed(staged.batch_id):
            return
        members = self.receipts.group_members(staged.batch_id)
        if members is None or staged.batch_hash not in members:
            raise MemoryBatchIntegrityError("已提交来源批次不得追加新单元")

    def _complete(
            self,
            staged: StagedMemoryBatch,
            *,
            fault_injector: MemoryBatchFaultInjector | None,
            ) -> MemoryBatchPublishResult:
        """幂等补齐 link、事件和投影，activation 已存在时只恢复结果。"""
        source = SourceRef.from_stable_key(staged.source_key)
        events = tuple(
            MemoryEvent.from_stable_key(key) for key in staged.event_keys)
        if any(
                item.object_ref.memory_space
                != self.event_log.memory_space_identity
                for item in events):
            raise MemoryBatchIntegrityError("staged event 跨 Memory 空间")
        registry = self.event_log.scoped_identities.registry
        event_hashes = tuple(
            registry.identity_hash(
                IDENTITY_MEMORY_EVENT, item.stable_key())
            for item in events
        )
        for ordinal, event_hash in enumerate(event_hashes):
            self.visibility.links.add(MemoryEventBatchLink(
                staged.batch_hash,
                staged.batch_id,
                event_hash,
                self.event_log.memory_space_id,
                ordinal,
            ))
        _hit(fault_injector, FAULT_MEMORY_BATCH_AFTER_LINK, {
            "batch_hash": staged.batch_hash,
            "event_count": staged.event_count,
        })
        materialized: list[MaterializedMemoryEvent] = []
        watermark = self.event_log.physical_timeline_watermark()
        next_timeline = 1 if watermark is None else watermark.seq + 1
        existing_links = self.visibility.links.for_batch(staged.batch_hash)
        for link in existing_links:
            rows = self.event_log.backend.select(
                MEMORY_EVENT_TABLE,
                where={"event_hash": link.event_hash},
            )
            if len(rows) > 1:
                raise MemoryBatchIntegrityError("batch event 存在重复物理行")
            if rows:
                next_timeline = max(
                    next_timeline,
                    MemoryEventRecord.from_row(rows[0]).timeline_seq + 1,
                )
        try:
            for ordinal, event in enumerate(events):
                restored = self.event_log.append(
                    event,
                    batch_hash=staged.batch_hash,
                    source_batch_id=staged.batch_id,
                    batch_ordinal=ordinal,
                    timeline_seq=next_timeline,
                    notify_listener=False,
                )
                materialized.append(restored)
                next_timeline = max(
                    next_timeline,
                    restored.timeline.seq + 1,
                )
                _hit(fault_injector, FAULT_MEMORY_BATCH_AFTER_EVENT, {
                    "batch_hash": staged.batch_hash,
                    "event_ordinal": ordinal,
                })
            if not self.visibility.group_is_pending(staged.batch_id):
                with self.visibility.preview(staged.batch_hash):
                    for item in materialized:
                        self.event_log.notify_appended(item)
                    self.aggregates.rebuild_dirty(
                        access=self._access(source.owner))
            _hit(fault_injector, FAULT_MEMORY_BATCH_AFTER_PROJECTION, {
                "batch_hash": staged.batch_hash,
                "event_count": staged.event_count,
            })
            _hit(fault_injector, FAULT_MEMORY_BATCH_BEFORE_ACTIVATION, {
                "batch_hash": staged.batch_hash,
                "event_count": staged.event_count,
            })
            if not self.receipts.is_active(staged.batch_hash):
                self.receipts.activate(staged)
            _hit(fault_injector, FAULT_MEMORY_BATCH_AFTER_ACTIVATION, {
                "batch_hash": staged.batch_hash,
                "event_count": staged.event_count,
            })
        except BaseException:
            self.aggregates.rebuild_all(access=self._access(source.owner))
            raise
        return MemoryBatchPublishResult(
            staged.batch_hash,
            tuple(materialized),
        )

    @staticmethod
    def _access(owner: OwnerScope) -> MemoryAccessContext:
        """从来源 owner 构造同层 Memory ACL 上下文。"""
        return MemoryAccessContext(
            owner.tenant_id,
            owner.user_id,
            owner.session_id,
        )


class MemoryBatchCoordinator:
    """串行执行同一来源批次的多个摄入单元，并把 cursor 确认放在最后。"""

    def __init__(self, runtimes: tuple[MemoryBatchRuntime, ...]) -> None:
        """绑定至少一个共享 K-02 visibility 的 Memory runtime。"""
        if (not isinstance(runtimes, tuple)
                or not runtimes
                or any(not isinstance(item, MemoryBatchRuntime)
                       for item in runtimes)):
            raise TypeError("runtimes 必须是非空 MemoryBatchRuntime tuple")
        visibility = runtimes[0].visibility
        if any(item.visibility is not visibility for item in runtimes):
            raise ValueError("batch coordinator runtimes 未共享同一 visibility")
        self.runtimes = runtimes
        self.visibility = visibility
        self.receipts = visibility.receipts

    def execute(
            self,
            batch_id: int,
            unit_actions: tuple[Callable[[], object], ...],
            *,
            cursor_commit: Callable[[], None],
            fault_injector: MemoryBatchFaultInjector | None = None,
            ) -> tuple[object, ...]:
        """以组 commit 原子发布全部单元，成功后才确认 cursor。"""
        if type(batch_id) is not int or batch_id <= 0:
            raise ValueError("coordinator batch_id 必须是正严格整数")
        if (not isinstance(unit_actions, tuple)
                or not unit_actions
                or any(not callable(item) for item in unit_actions)):
            raise TypeError("unit_actions 必须是非空 callable tuple")
        if not callable(cursor_commit):
            raise TypeError("cursor_commit 必须可调用")
        had_intent = self.receipts.has_group_intent(batch_id)
        if not had_intent and self._staged_hashes(batch_id):
            raise MemoryBatchIntegrityError(
                "已有独立摄入单元的 batch_id 不得改作组批次")
        self.receipts.begin_group(batch_id)
        if self.receipts.is_group_rolled_back(batch_id):
            raise MemoryBatchIntegrityError("已 rollback 来源批次不得重新执行")
        already_committed = self.receipts.is_group_committed(batch_id)
        results = []
        try:
            for ordinal, action in enumerate(unit_actions):
                result = action()
                record = getattr(result, "source_record", None)
                if record is None or record.batch_id != batch_id:
                    raise MemoryBatchIntegrityError(
                        "摄入单元结果未绑定 coordinator batch_id")
                results.append(result)
                _hit(fault_injector, FAULT_MEMORY_GROUP_AFTER_UNIT, {
                    "batch_id": batch_id,
                    "unit_ordinal": ordinal,
                })
            members = tuple(sorted(
                self._result_batch_hash(item, batch_id)
                for item in results
            ))
            if len(set(members)) != len(members):
                raise MemoryBatchIntegrityError("coordinator 返回重复摄入单元")
            if self._staged_hashes(batch_id) != members:
                raise MemoryBatchIntegrityError(
                    "组批次实际 staged 单元与执行结果不闭合")
            if already_committed:
                if self.receipts.group_members(batch_id) != members:
                    raise MemoryBatchIntegrityError(
                        "已提交来源批次的成员清单发生漂移")
            else:
                self.receipts.finalize_group(batch_id, members)
                with self.visibility.preview_many(members):
                    self._rebuild_results(results)
                _hit(fault_injector, FAULT_MEMORY_GROUP_AFTER_PROJECTION, {
                    "batch_id": batch_id,
                    "unit_count": len(members),
                })
                self.receipts.commit_group(batch_id)
                _hit(fault_injector, FAULT_MEMORY_GROUP_AFTER_COMMIT, {
                    "batch_id": batch_id,
                    "unit_count": len(members),
                })
        except BaseException:
            if not self.receipts.is_group_committed(batch_id):
                self.receipts.rollback_group(batch_id)
                for runtime in self.runtimes:
                    runtime.rollback_batch(batch_id)
            raise
        cursor_commit()
        return tuple(results)

    def rollback_batch(self, batch_id: int) -> tuple[int, ...]:
        """先发布组回滚可见点，再清理各 Memory 空间单元投影。"""
        if type(batch_id) is not int or batch_id <= 0:
            raise ValueError("coordinator rollback batch_id 必须是正严格整数")
        if self.receipts.has_group_intent(batch_id):
            self.receipts.rollback_group(batch_id)
        rolled = []
        for runtime in self.runtimes:
            rolled.extend(runtime.rollback_batch(batch_id))
        return tuple(sorted(rolled))

    def recover_groups(self) -> None:
        """核验持久化组状态，并一次性重建受影响 owner 的派生投影。"""
        affected: set[tuple[int, OwnerScope]] = set()
        for batch_id in self.receipts.group_batch_ids():
            staged_hashes = self._staged_hashes(batch_id)
            if self.receipts.is_group_rolled_back(batch_id):
                for runtime in self.runtimes:
                    runtime.rollback_batch(batch_id)
            elif self.receipts.is_group_committed(batch_id):
                members = self.receipts.group_members(batch_id)
                if members is None or members != staged_hashes:
                    raise MemoryBatchIntegrityError(
                        "恢复时组成员与 staged 单元不闭合")
                if any(
                        not self.receipts.is_active(batch_hash)
                        or self.receipts.is_rolled_back(batch_hash)
                        for batch_hash in members):
                    raise MemoryBatchIntegrityError(
                        "恢复时已提交组含非活动单元")
            for index, runtime in enumerate(self.runtimes):
                for owner in runtime.staged_owners(batch_id):
                    affected.add((index, owner))
        for index, owner in sorted(
                affected,
                key=lambda item: (item[0], item[1].stable_key())):
            runtime = self.runtimes[index]
            runtime.aggregates.rebuild_all(access=runtime._access(owner))

    def _staged_hashes(self, batch_id: int) -> tuple[int, ...]:
        """返回跨 Memory 空间的完整 staged 单元集合。"""
        return tuple(sorted(
            self.visibility.links.hashes_for_source_batch(batch_id)
        ))

    def _result_batch_hash(self, result: object, batch_id: int) -> int:
        """从 intake 结果的空间与来源恢复组成员身份。"""
        record = getattr(result, "source_record", None)
        manifest_ref = getattr(result, "manifest_ref", None)
        memory_space = getattr(manifest_ref, "memory_space", None)
        if record is None or memory_space is None:
            raise MemoryBatchIntegrityError("摄入结果缺少来源或 Memory 空间")
        source = SourceRef.from_stable_key(record.source_key)
        batch_hash = memory_batch_hash(
            memory_space.stable_key(),
            source.stable_key(),
            batch_id,
        )
        staged = self.receipts.staged(batch_hash)
        if staged is None:
            raise MemoryBatchIntegrityError("摄入结果没有对应 staged 单元")
        return batch_hash

    def _rebuild_results(self, results: list[object]) -> None:
        """按结果涉及的空间和 owner 各重建一次完整派生投影。"""
        affected: set[tuple[int, OwnerScope]] = set()
        for result in results:
            record = getattr(result, "source_record", None)
            manifest_ref = getattr(result, "manifest_ref", None)
            memory_space = getattr(manifest_ref, "memory_space", None)
            if record is None or memory_space is None:
                raise MemoryBatchIntegrityError("摄入结果缺少投影重建身份")
            owner = SourceRef.from_stable_key(record.source_key).owner
            matches = tuple(
                index for index, runtime in enumerate(self.runtimes)
                if runtime.event_log.memory_space_identity == memory_space
            )
            if len(matches) != 1:
                raise MemoryBatchIntegrityError("摄入结果没有唯一 Memory runtime")
            affected.add((matches[0], owner))
        for index, owner in sorted(
                affected,
                key=lambda item: (item[0], item[1].stable_key())):
            runtime = self.runtimes[index]
            runtime.aggregates.rebuild_all(access=runtime._access(owner))


def install_memory_batch_runtimes(
        ctx,
        config: MemoryBatchRuntimeConfig,
        ) -> None:
    """在 TrainContext 上装配共享 K-02 store、可见性和两个 Memory writer。"""
    if not isinstance(config, MemoryBatchRuntimeConfig):
        raise TypeError("config 必须是 MemoryBatchRuntimeConfig")
    if (ctx.memory_read_events is None
            or ctx.memory_interact_events is None
            or ctx.memory_read_aggregates is None
            or ctx.memory_interact_aggregates is None):
        raise MemoryBatchIntegrityError("TrainContext 缺少双层 Memory event/aggregate")
    store = build_tiered_segment_store(
        ctx.backend,
        build_storage_role_registry(),
        config.temperature_profile,
    )
    receipts = MemoryBatchReceiptStore(
        store,
        tier_key=config.tier_key,
        read_budget=config.read_budget,
    )
    visibility = MemoryBatchVisibility(
        MemoryEventBatchLinkStore(ctx.backend),
        receipts,
    )
    read_runtime = MemoryBatchRuntime(
        ctx.memory_read_events,
        ctx.memory_read_aggregates,
        visibility,
        receipts,
        core_dependency=config.core_dependency,
        write_budget=config.write_budget,
    )
    interact_runtime = MemoryBatchRuntime(
        ctx.memory_interact_events,
        ctx.memory_interact_aggregates,
        visibility,
        receipts,
        core_dependency=config.core_dependency,
        write_budget=config.write_budget,
    )
    ctx.tiered_segment_store = store
    ctx.memory_batch_config = config
    ctx.memory_batch_visibility = visibility
    ctx.memory_read_batch_runtime = read_runtime
    ctx.memory_interact_batch_runtime = interact_runtime
    ctx.memory_batch_coordinator = MemoryBatchCoordinator((
        read_runtime,
        interact_runtime,
    ))
    if ctx.memory_read_intake is not None:
        ctx.memory_read_intake.attach_batch_runtime(read_runtime)
    if ctx.memory_interact_intake is not None:
        ctx.memory_interact_intake.attach_batch_runtime(interact_runtime)
    read_runtime.recover_pending()
    interact_runtime.recover_pending()
    ctx.memory_batch_coordinator.recover_groups()


__all__ = [
    "FAULT_MEMORY_BATCH_AFTER_ACTIVATION",
    "FAULT_MEMORY_BATCH_AFTER_EVENT",
    "FAULT_MEMORY_BATCH_AFTER_LINK",
    "FAULT_MEMORY_BATCH_AFTER_PROJECTION",
    "FAULT_MEMORY_BATCH_AFTER_ROLLBACK_PROJECTION",
    "FAULT_MEMORY_BATCH_AFTER_ROLLBACK_RECEIPT",
    "FAULT_MEMORY_BATCH_AFTER_STAGE",
    "FAULT_MEMORY_BATCH_BEFORE_ACTIVATION",
    "FAULT_MEMORY_GROUP_AFTER_COMMIT",
    "FAULT_MEMORY_GROUP_AFTER_PROJECTION",
    "FAULT_MEMORY_GROUP_AFTER_UNIT",
    "MemoryBatchFaultInjector",
    "MemoryBatchCoordinator",
    "MemoryBatchPublishResult",
    "MemoryBatchRuntime",
    "MemoryBatchRuntimeConfig",
    "install_memory_batch_runtimes",
]
