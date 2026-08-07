"""interaction Use 到其他 Memory 空间对象的桥索引生命周期。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import OwnerScope
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_USE,
    MEMORY_OBJECT_HYPOTHESIS,
    UsePayload,
    MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.memory_event_log import (
    MaterializedMemoryEvent,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.storage.assertion_identity import IDENTITY_MEMORY_OBJECT
from pure_integer_ai.storage.cross_memory_use import (
    CrossMemoryUseRecord,
    CrossMemoryUseRepository,
)
from pure_integer_ai.storage.spaces.registry import SpaceRegistry


# object-model: lifecycle; owner=train-context; cleanup=backend-close
class CrossMemoryUseRuntime:
    """同步记录、ACL 查询并按显式 access 修复跨空间 Use 桥事实。"""

    def __init__(self, ctx: TrainContext) -> None:
        """绑定双 Memory event log、共享 identity registry 和独立桥表。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("cross Memory Use ctx 必须是 TrainContext")
        if ctx.memory_read_events is None or ctx.memory_interact_events is None:
            raise ValueError("cross Memory Use 缺少双 Memory event log")
        self._ctx = ctx
        self.repository = CrossMemoryUseRepository(ctx.backend)
        self._spaces = SpaceRegistry(ctx.backend)
        self._identities = ctx.scoped_identity_store.registry
        for record in self.repository.all_records():
            OwnerScope(*record.target_owner_key)

    def require_target(self, target_ref: MemoryObjectRef) -> None:
        """在写 Use 前核验目标是当前上下文已声明的跨空间 Hypothesis。"""
        if (not isinstance(target_ref, MemoryObjectRef)
                or target_ref.object_kind != MEMORY_OBJECT_HYPOTHESIS):
            raise ValueError("cross Memory Use target 必须是 Hypothesis")
        if target_ref.memory_space == (
                self._ctx.memory_interact_events.memory_space_identity):
            raise ValueError("cross Memory Use target 必须属于其他 Memory 空间")
        if target_ref.memory_space != (
                self._ctx.memory_read_events.memory_space_identity):
            raise ValueError("cross Memory Use target 不属于当前阅读空间")
        if self._identities.find(
                IDENTITY_MEMORY_OBJECT, target_ref.stable_key()) is None:
            raise ValueError("cross Memory Use target 没有已登记完整身份")

    def record(self, materialized: MaterializedMemoryEvent) -> CrossMemoryUseRecord:
        """从真实 interaction Use 追加唯一桥行，不复制 payload/context/outcome。"""
        if not isinstance(materialized, MaterializedMemoryEvent):
            raise TypeError("cross Memory Use record 需要 materialized event")
        event = materialized.event
        if (event.event_kind != MEMORY_EVENT_USE
                or not isinstance(event.payload, UsePayload)):
            raise ValueError("cross Memory Use record 只接受 Use event")
        if event.object_ref.memory_space != (
                self._ctx.memory_interact_events.memory_space_identity):
            raise ValueError("cross Memory Use source 必须是 interaction")
        target_ref = event.payload.memory_ref
        self.require_target(target_ref)
        target_space_id = self._spaces.lookup_by_hash(
            target_ref.memory_space.type_hash,
            target_ref.memory_space.name_hash,
        )
        target_object_hash = self._identities.find(
            IDENTITY_MEMORY_OBJECT, target_ref.stable_key())
        if target_space_id is None or target_object_hash is None:
            raise ValueError("cross Memory Use target 空间或对象身份缺失")
        if self._spaces.identity(target_space_id) != target_ref.memory_space:
            raise ValueError("cross Memory Use target 运行时空间漂移")
        record = CrossMemoryUseRecord(
            self._ctx.memory_interact_events.memory_space_id,
            materialized.event_hash,
            materialized.object_hash,
            materialized.timeline.seq,
            target_space_id,
            target_object_hash,
            target_ref.owner.stable_key(),
        )
        return self.repository.put(record)[0]

    def uses_for(
            self,
            target_ref: MemoryObjectRef,
            *,
            access: MemoryAccessContext,
            ) -> tuple[CrossMemoryUseRecord, ...]:
        """按目标 owner ACL 返回最小桥事实，不提升 interaction payload 权限。"""
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("cross Memory Use query 缺少 MemoryAccessContext")
        self.require_target(target_ref)
        if not access.can_read(target_ref.owner):
            return ()
        target_space_id = self._spaces.lookup_by_hash(
            target_ref.memory_space.type_hash,
            target_ref.memory_space.name_hash,
        )
        target_object_hash = self._identities.find(
            IDENTITY_MEMORY_OBJECT, target_ref.stable_key())
        if target_space_id is None or target_object_hash is None:
            raise ValueError("cross Memory Use query 目标身份缺失")
        records = self.repository.for_target(
            target_space_id=target_space_id,
            target_object_hash=target_object_hash,
        )
        for record in records:
            if OwnerScope(*record.target_owner_key) != target_ref.owner:
                raise ValueError("cross Memory Use row 目标 owner 漂移")
        return records

    def audit_use(
            self,
            record: CrossMemoryUseRecord,
            *,
            access: MemoryAccessContext,
            ) -> MaterializedMemoryEvent | None:
        """用原 interaction ACL 回读完整 Use；无 session 权限只返回空。"""
        if not isinstance(record, CrossMemoryUseRecord):
            raise TypeError("cross Memory Use audit record 类型错误")
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("cross Memory Use audit access 类型错误")
        event = self._ctx.memory_interact_events.read(
            record.use_event_hash, access=access)
        if event is None:
            return None
        if (event.object_hash != record.use_object_hash
                or event.timeline.seq != record.source_timeline_seq
                or not isinstance(event.event.payload, UsePayload)):
            raise ValueError("cross Memory Use audit 与原事件漂移")
        target_ref = event.event.payload.memory_ref
        target_space_id = self._spaces.lookup_by_hash(
            target_ref.memory_space.type_hash,
            target_ref.memory_space.name_hash,
        )
        target_object_hash = self._identities.find(
            IDENTITY_MEMORY_OBJECT, target_ref.stable_key())
        if (target_space_id != record.target_space_id
                or target_object_hash != record.target_object_hash
                or target_ref.owner.stable_key() != record.target_owner_key):
            raise ValueError("cross Memory Use audit target 与桥行漂移")
        return event

    def recover(self, *, access: MemoryAccessContext) -> int:
        """扫描调用方有权读取的 interaction Use，幂等补齐可能缺失的桥行。"""
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("cross Memory Use recover 缺少 MemoryAccessContext")
        inserted = 0
        known = {
            (item.source_space_id, item.use_event_hash)
            for item in self.repository.all_records()
        }
        for event in self._ctx.memory_interact_events.query(
                access=access, event_kind=MEMORY_EVENT_USE):
            payload = event.event.payload
            if (not isinstance(payload, UsePayload)
                    or payload.memory_ref.memory_space
                    == self._ctx.memory_interact_events.memory_space_identity):
                continue
            self.require_target(payload.memory_ref)
            key = (
                self._ctx.memory_interact_events.memory_space_id,
                event.event_hash,
            )
            self.record(event)
            if key not in known:
                inserted += 1
                known.add(key)
        return inserted

    def clone_for_context(self, ctx: TrainContext) -> "CrossMemoryUseRuntime":
        """为 V-06 clone 重绑独立 backend，并核对协议身份不变。"""
        result = CrossMemoryUseRuntime(ctx)
        if result.state_key() != self.state_key():
            raise ValueError("cross Memory Use clone 改变了协议状态")
        return result

    def state_key(self) -> tuple[int, ...]:
        """返回桥协议和双空间完整稳定身份。"""
        return (
            1,
            *self._ctx.memory_interact_events.memory_space_identity.stable_key(),
            *self._ctx.memory_read_events.memory_space_identity.stable_key(),
        )


def install_cross_memory_use_runtime(
        ctx: TrainContext,
        *,
        recovery_access: MemoryAccessContext | None = None,
        ) -> CrossMemoryUseRuntime:
    """安装唯一桥 runtime，并可按显式 ACL 幂等修复既有 Use。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("cross Memory Use ctx 必须是 TrainContext")
    if ctx.cross_memory_use_runtime is not None:
        raise ValueError("TrainContext 已安装 cross Memory Use runtime")
    runtime = CrossMemoryUseRuntime(ctx)
    ctx.cross_memory_use_runtime = runtime
    if recovery_access is not None:
        runtime.recover(access=recovery_access)
    return runtime


__all__ = [
    "CrossMemoryUseRuntime",
    "install_cross_memory_use_runtime",
]
