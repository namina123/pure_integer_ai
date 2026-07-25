"""M-11 Memory 逻辑导出、管理授权和可恢复遗忘 runtime。"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.identity import OwnerScope, SourceRef
from pure_integer_ai.cognition.shared.memory_event import MemoryEvent
from pure_integer_ai.cognition.shared.memory_event_log import MemoryEventLog
from pure_integer_ai.cognition.shared.memory_overlay import (
    MemoryAccessContext,
    MemoryOverlay,
    MemoryOverlayRelation,
)
from pure_integer_ai.cognition.shared.memory_owner import (
    MemoryManagementContext,
    MemoryOwnerAuthorizer,
    MemoryOwnerSelector,
)
from pure_integer_ai.storage.memory_forget import (
    FORGET_TARGET_COMPANION,
    FORGET_TARGET_EVENT,
    FORGET_TARGET_OVERLAY,
    FORGET_TARGET_SOURCE,
    MemoryForgetIntegrityError,
    MemoryForgetStore,
    MemoryForgetTarget,
    MemoryForgetVisibility,
    StagedMemoryForget,
    memory_forget_hash,
)
from pure_integer_ai.storage.source_record import (
    SourceRecordRepository,
    SourceRecordStorage,
)
from pure_integer_ai.storage.spaces.companion import CompanionSpace


FAULT_MEMORY_FORGET_AFTER_STAGE = 1
FAULT_MEMORY_FORGET_AFTER_PROJECTION = 2
FAULT_MEMORY_FORGET_AFTER_COMMIT = 3


class MemoryIsolationError(RuntimeError):
    """Memory 管理授权、导出闭包或遗忘事务不一致。"""


@runtime_checkable
class MemoryForgetFaultInjector(Protocol):
    """在 M-11 遗忘承重边界注入故障的最小协议。"""

    def hit(self, point: int, context: dict[str, int]) -> None:
        """观察边界；测试需要中断时直接抛出异常。"""
        ...


def _hit(
        injector: MemoryForgetFaultInjector | None,
        point: int,
        context: dict[str, int],
        ) -> None:
    """调用可选故障注入器并复制纯整数上下文。"""
    if injector is None:
        return
    if not isinstance(injector, MemoryForgetFaultInjector):
        raise TypeError("forget fault injector 协议错误")
    injector.hit(point, dict(context))


@dataclass(frozen=True, order=True)
class ExportedMemoryEvent:
    """逻辑导出中的 Memory event 完整身份和领域对象。"""

    space_id: int
    event_hash: int
    event: MemoryEvent


@dataclass(frozen=True, order=True)
class ExportedMemoryOverlay:
    """逻辑导出中的 Memory overlay 完整身份和关系。"""

    space_id: int
    identity_hash: int
    relation: MemoryOverlayRelation


@dataclass(frozen=True, order=True)
class ExportedCompanionAssoc:
    """逻辑导出中的稳定 Companion assoc 和原文。"""

    type_hash: int
    name_hash: int
    assoc_id: int
    text_hash: int
    text: str
    meta: int

    def stable_assoc_key(self) -> tuple[int, ...]:
        """返回 Companion 空间身份和局部 assoc 的完整键。"""
        return self.type_hash, self.name_hash, self.assoc_id


@dataclass(frozen=True)
class MemoryLogicalExport:
    """不含 K-02 物理元数据的可审计 Memory 逻辑闭包。"""

    selection_key: tuple[int, ...]
    events: tuple[ExportedMemoryEvent, ...]
    overlays: tuple[ExportedMemoryOverlay, ...]
    sources: tuple[SourceRecordStorage, ...]
    companions: tuple[ExportedCompanionAssoc, ...]

    def __post_init__(self) -> None:
        """核验选择键非空且四类导出集合均稳定唯一。"""
        if not isinstance(self.selection_key, tuple) or not self.selection_key:
            raise ValueError("export selection_key 必须是非空 tuple")
        keyed_collections = (
            (self.events, lambda item: (item.space_id, item.event_hash)),
            (self.overlays,
             lambda item: (item.space_id, item.identity_hash)),
            (self.sources, lambda item: item.source_key),
            (self.companions, lambda item: item.stable_assoc_key()),
        )
        for items, identity_key in keyed_collections:
            if not isinstance(items, tuple):
                raise TypeError("export 集合必须是 tuple")
            keys = tuple(identity_key(item) for item in items)
            if len(set(keys)) != len(keys):
                raise MemoryIsolationError("export 集合含重复完整对象")


@dataclass(frozen=True)
class MemoryForgetResult:
    """一个已提交遗忘操作及其逻辑闭包计数。"""

    operation_hash: int
    event_count: int
    overlay_count: int
    source_count: int
    companion_count: int


class MemoryIsolationRuntime:
    """组合 ACL 导出、管理授权、K-02 遗忘和派生重建。"""

    def __init__(
            self,
            event_logs: tuple[MemoryEventLog, ...],
            overlays: tuple[MemoryOverlay, ...],
            aggregates: tuple[Any, ...],
            source_repository: SourceRecordRepository,
            companions: tuple[CompanionSpace, ...],
            forget_store: MemoryForgetStore,
            forget_visibility: MemoryForgetVisibility,
            authorizer: MemoryOwnerAuthorizer,
            ) -> None:
        """绑定双 Memory facade、来源闭包、遗忘存储和授权策略。"""
        if (not isinstance(event_logs, tuple)
                or not event_logs
                or any(not isinstance(item, MemoryEventLog)
                       for item in event_logs)):
            raise TypeError("event_logs 必须是非空 MemoryEventLog tuple")
        if (not isinstance(overlays, tuple)
                or len(overlays) != len(event_logs)
                or any(not isinstance(item, MemoryOverlay)
                       for item in overlays)):
            raise TypeError("overlays 必须与 event_logs 一一对应")
        if not isinstance(aggregates, tuple) or len(aggregates) != len(event_logs):
            raise TypeError("aggregates 必须与 event_logs 一一对应")
        if not isinstance(source_repository, SourceRecordRepository):
            raise TypeError("source_repository 类型错误")
        if (not isinstance(companions, tuple)
                or not companions
                or any(not isinstance(item, CompanionSpace)
                       for item in companions)):
            raise TypeError("companions 必须是非空 CompanionSpace tuple")
        if not isinstance(forget_store, MemoryForgetStore):
            raise TypeError("forget_store 类型错误")
        if (not isinstance(forget_visibility, MemoryForgetVisibility)
                or forget_visibility.store is not forget_store):
            raise TypeError("forget_visibility 与 store 不一致")
        if not isinstance(authorizer, MemoryOwnerAuthorizer):
            raise TypeError("authorizer 必须实现 MemoryOwnerAuthorizer")
        self.event_logs = event_logs
        self.overlays = overlays
        self.aggregates = aggregates
        self.source_repository = source_repository
        self.companions = companions
        self.forget_store = forget_store
        self.forget_visibility = forget_visibility
        self.authorizer = authorizer
        self._authorizer_key = self._validate_authorizer_key(
            authorizer.state_key())
        backends = {
            id(item.backend) for item in (*event_logs, *overlays, *companions)
        }
        backends.add(id(source_repository.backend))
        if len(backends) != 1:
            raise MemoryIsolationError("M-11 facade 未绑定同一 backend")

    def state_key(self) -> tuple[int, ...]:
        """返回授权配置和遗忘物理策略的稳定 runtime 键。"""
        return (
            *self._authorizer_key,
            len(self.forget_store.tier_key),
            *self.forget_store.tier_key,
            self.forget_store.read_budget.object_limit,
            self.forget_store.read_budget.byte_limit,
            self.forget_store.write_budget.object_limit,
            self.forget_store.write_budget.byte_limit,
        )

    def export(self, access: MemoryAccessContext) -> MemoryLogicalExport:
        """按普通 ACL 导出当前可见且未遗忘的逻辑闭包。"""
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("access 必须是 MemoryAccessContext")
        events = tuple(
            ExportedMemoryEvent(log.memory_space_id, item.event_hash, item.event)
            for log in self.event_logs
            for item in log.query(access=access)
        )
        overlays = tuple(
            ExportedMemoryOverlay(
                overlay.memory_space_id,
                item.identity_hash,
                item.relation,
            )
            for overlay in self.overlays
            for item in overlay.query(access=access)
        )
        return self._export_closure(
            (1, *access.stable_key()),
            events,
            overlays,
            source_visible=access.can_read,
        )

    def export_managed(
            self,
            context: MemoryManagementContext,
            ) -> MemoryLogicalExport:
        """经独立授权后按 exact/subtree owner 导出逻辑闭包。"""
        selector = self._authorize(context)
        events = tuple(
            ExportedMemoryEvent(log.memory_space_id, item.event_hash, item.event)
            for log in self.event_logs
            for item in log.query_owned(selector)
        )
        overlays = tuple(
            ExportedMemoryOverlay(
                overlay.memory_space_id,
                item.identity_hash,
                item.relation,
            )
            for overlay in self.overlays
            for item in overlay.query_owned(selector)
        )
        actor_access = MemoryAccessContext(
            context.actor.tenant_id,
            context.actor.user_id,
            context.actor.session_id,
        )
        return self._export_closure(
            (2, *context.stable_key()),
            events,
            overlays,
            source_visible=lambda owner: (
                selector.matches(owner) or actor_access.can_read(owner)),
        )

    def forget(
            self,
            context: MemoryManagementContext,
            *,
            reason_key: tuple[int, ...],
            fault_injector: MemoryForgetFaultInjector | None = None,
            ) -> MemoryForgetResult:
        """封存授权范围逻辑闭包，重建投影后发布唯一 forget commit。"""
        selector = self._authorize(context)
        export = self.export_managed(context)
        targets = self._forget_targets(export, selector)
        if not targets:
            raise MemoryIsolationError("遗忘目标范围当前没有可见对象")
        operation_hash = memory_forget_hash(
            selector.target.stable_key(),
            selector.selection_kind,
            reason_key,
            targets,
        )
        operation = StagedMemoryForget(
            operation_hash,
            selector.target.stable_key(),
            selector.selection_kind,
            reason_key,
            targets,
        )
        affected_owners = self._owners_for_operation(operation, selector)
        previous = self.forget_store.staged(operation_hash)
        if previous is not None and previous != operation:
            raise MemoryForgetIntegrityError("同一遗忘操作 staged 内容漂移")
        self.forget_store.stage(operation)
        _hit(fault_injector, FAULT_MEMORY_FORGET_AFTER_STAGE, {
            "operation_hash": operation_hash,
            "target_count": len(targets),
        })
        try:
            with self.forget_visibility.preview(operation_hash):
                self._rebuild_owners(affected_owners)
            _hit(fault_injector, FAULT_MEMORY_FORGET_AFTER_PROJECTION, {
                "operation_hash": operation_hash,
                "target_count": len(targets),
            })
            if not self.forget_store.is_committed(operation_hash):
                self.forget_store.commit(operation)
            _hit(fault_injector, FAULT_MEMORY_FORGET_AFTER_COMMIT, {
                "operation_hash": operation_hash,
                "target_count": len(targets),
            })
        except BaseException:
            self._rebuild_owners(affected_owners)
            raise
        return MemoryForgetResult(
            operation_hash,
            len(export.events),
            len(export.overlays),
            len(tuple(
                item for item in export.sources
                if selector.matches(SourceRef.from_stable_key(
                    item.source_key).owner))),
            len(tuple(
                item for item in export.companions
                if MemoryForgetTarget(
                    FORGET_TARGET_COMPANION,
                    item.stable_assoc_key()) in targets)),
        )

    def recover_pending(self) -> tuple[int, ...]:
        """启动时从完整 staged set 重建投影并 roll-forward commit。"""
        operations = self.forget_store.staged_operations()
        pending = tuple(
            operation for operation in operations
            if not self.forget_store.is_committed(operation.operation_hash)
        )
        recovered = []
        final_owners = set()
        for operation in pending:
            selector = MemoryOwnerSelector(
                OwnerScope(*operation.owner_key),
                operation.selection_kind,
            )
            affected_owners = self._owners_for_operation(operation, selector)
            with self.forget_visibility.preview(operation.operation_hash):
                self._rebuild_owners(affected_owners)
            self.forget_store.commit(operation)
            recovered.append(operation.operation_hash)
            final_owners.update(affected_owners)
        if final_owners:
            self._rebuild_owners(tuple(sorted(
                final_owners, key=lambda item: item.stable_key())))
        return tuple(sorted(recovered))

    def clone_for_context(self, ctx) -> "MemoryIsolationRuntime":
        """在 V-06 clone 上重建独立 K-02 可见性并克隆授权器。"""
        authorizer = self.authorizer
        clone = getattr(authorizer, "clone_for_context", None)
        if callable(clone):
            authorizer = clone(ctx)
        return install_memory_isolation_runtime(ctx, authorizer)

    def _authorize(
            self,
            context: MemoryManagementContext,
            ) -> MemoryOwnerSelector:
        """核验管理请求绑定当前授权配置且策略明确允许。"""
        if not isinstance(context, MemoryManagementContext):
            raise TypeError("context 必须是 MemoryManagementContext")
        if context.authorizer_state_key != self._authorizer_key:
            raise MemoryIsolationError("管理请求 authorizer state key 漂移")
        if not self.authorizer.authorize(context.actor, context.selector):
            raise MemoryIsolationError("管理请求未获授权")
        return context.selector

    def _export_closure(
            self,
            selection_key: tuple[int, ...],
            events: tuple[ExportedMemoryEvent, ...],
            overlays: tuple[ExportedMemoryOverlay, ...],
            *,
            source_visible,
            ) -> MemoryLogicalExport:
        """从可见事件递归收集 SourceRef，再定点读取来源和伴随原文。"""
        source_refs = set()
        for item in events:
            self._collect_sources(item.event, source_refs, set())
        visible_sources = tuple(sorted(
            source for source in source_refs
            if (source_visible(source.owner)
                and not self.forget_visibility.source_is_forgotten(
                    source.stable_key()))
        ))
        source_records = []
        companion_keys = set()
        for source in visible_sources:
            record = self.source_repository.find(source.stable_key())
            if record is None:
                raise MemoryIsolationError("导出事件引用了缺失 SourceRecord")
            source_records.append(record)
            companion_keys.add((
                record.companion_type_hash,
                record.companion_name_hash,
                record.companion_assoc_id,
            ))
        companion_by_identity = {
            (item.identity.type_hash, item.identity.name_hash): item
            for item in self.companions
        }
        companions = []
        for assoc_key in sorted(companion_keys):
            if self.forget_visibility.companion_is_forgotten(assoc_key):
                continue
            companion = companion_by_identity.get(assoc_key[:2])
            if companion is None:
                raise MemoryIsolationError("导出来源指向未装配 Companion")
            row = companion.read(assoc_key[2])
            companions.append(ExportedCompanionAssoc(
                assoc_key[0],
                assoc_key[1],
                assoc_key[2],
                row["text_hash"],
                row["text"],
                row["meta"],
            ))
        return MemoryLogicalExport(
            selection_key,
            tuple(sorted(events)),
            tuple(sorted(overlays)),
            tuple(sorted(source_records, key=lambda item: item.source_key)),
            tuple(companions),
        )

    def _forget_targets(
            self,
            export: MemoryLogicalExport,
            selector: MemoryOwnerSelector,
            ) -> tuple[MemoryForgetTarget, ...]:
        """只把管理范围自身对象纳入遗忘，保留其引用的更广共享来源。"""
        targets = [
            MemoryForgetTarget(
                FORGET_TARGET_EVENT,
                (
                    item.space_id,
                    item.event_hash,
                    *item.event.object_ref.owner.stable_key(),
                ),
            )
            for item in export.events
            if selector.matches(item.event.object_ref.owner)
        ]
        targets.extend(
            MemoryForgetTarget(
                FORGET_TARGET_OVERLAY,
                (
                    item.space_id,
                    item.identity_hash,
                    *item.relation.owner.stable_key(),
                ),
            )
            for item in export.overlays
            if selector.matches(item.relation.owner)
        )
        selected_sources = tuple(
            item for item in export.sources
            if selector.matches(SourceRef.from_stable_key(item.source_key).owner)
        )
        targets.extend(
            MemoryForgetTarget(FORGET_TARGET_SOURCE, item.source_key)
            for item in selected_sources
        )
        companion_keys = {
            (
                item.companion_type_hash,
                item.companion_name_hash,
                item.companion_assoc_id,
            )
            for item in selected_sources
        }
        targets.extend(
            MemoryForgetTarget(FORGET_TARGET_COMPANION, item)
            for item in companion_keys
        )
        return tuple(sorted(set(targets)))

    @staticmethod
    def _owners_for_operation(
            operation: StagedMemoryForget,
            selector: MemoryOwnerSelector,
            ) -> tuple[OwnerScope, ...]:
        """从 staged 完整目标恢复受影响 owner，不依赖已被预览隐藏的记录。"""
        owners = set()
        for target in operation.targets:
            owner_key = target.owner_key()
            if owner_key is None:
                continue
            owner = OwnerScope(*owner_key)
            if not selector.matches(owner):
                raise MemoryForgetIntegrityError(
                    "forget target owner 超出 staged 管理范围")
            owners.add(owner)
        if not owners:
            owners.add(selector.target)
        return tuple(sorted(owners, key=lambda item: item.stable_key()))

    def _rebuild_owners(self, owners: tuple[OwnerScope, ...]) -> None:
        """按显式 owner 集重建各 Memory 空间派生投影。"""
        if (not isinstance(owners, tuple) or not owners
                or any(not isinstance(item, OwnerScope) for item in owners)):
            raise TypeError("owners 必须是非空 OwnerScope tuple")
        for aggregate in self.aggregates:
            for owner in owners:
                aggregate.rebuild_all(access=MemoryAccessContext(
                    owner.tenant_id,
                    owner.user_id,
                    owner.session_id,
                ))

    @classmethod
    def _collect_sources(
            cls,
            value: Any,
            result: set[SourceRef],
            visited: set[int],
            ) -> None:
        """沿结构化 dataclass/tuple 字段递归收集 SourceRef。"""
        if isinstance(value, SourceRef):
            result.add(value)
            return
        if value is None or isinstance(value, (int, str, bytes)):
            return
        object_id = id(value)
        if object_id in visited:
            return
        visited.add(object_id)
        if is_dataclass(value):
            for field in fields(value):
                cls._collect_sources(
                    getattr(value, field.name), result, visited)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                cls._collect_sources(item, result, visited)

    @staticmethod
    def _validate_authorizer_key(value: tuple[int, ...]) -> tuple[int, ...]:
        """核验授权器 state key 为非空严格整数 tuple。"""
        if not isinstance(value, tuple) or not value:
            raise MemoryIsolationError("authorizer state_key 必须是非空 tuple")
        if any(type(item) is not int for item in value):
            raise MemoryIsolationError("authorizer state_key 必须使用严格整数")
        return value


def install_memory_isolation_runtime(
        ctx,
        authorizer: MemoryOwnerAuthorizer,
        ) -> MemoryIsolationRuntime:
    """在已安装 M-10 的 TrainContext 上装配 M-11 runtime。"""
    if ctx.tiered_segment_store is None or ctx.memory_batch_config is None:
        raise MemoryIsolationError("M-11 依赖已安装的 M-10 K-02 runtime")
    if (ctx.memory_read_events is None
            or ctx.memory_interact_events is None
            or ctx.memory_read_overlay is None
            or ctx.memory_interact_overlay is None
            or ctx.memory_read_aggregates is None
            or ctx.memory_interact_aggregates is None
            or ctx.memory_read_intake is None
            or ctx.memory_interact_intake is None):
        raise MemoryIsolationError("M-11 缺少双 Memory/Source facade")
    source_intake = ctx.memory_read_intake.source_intake
    source_repository = source_intake.repository
    if ctx.memory_interact_intake.source_intake is not source_intake:
        raise MemoryIsolationError("双 Memory intake 未共享 SourceRecordRepository")
    companion_by_key = {}
    for companion in (
            ctx.memory_read_intake.source_intake.companion,
            ctx.memory_interact_intake.source_intake.companion):
        companion_by_key.setdefault(companion.identity.stable_key(), companion)
    companions = tuple(
        companion_by_key[key] for key in sorted(companion_by_key))
    config = ctx.memory_batch_config
    forget_store = MemoryForgetStore(
        ctx.tiered_segment_store,
        tier_key=config.tier_key,
        read_budget=config.read_budget,
        write_budget=config.write_budget,
    )
    visibility = MemoryForgetVisibility(forget_store)
    for event_log in (
            ctx.memory_read_events,
            ctx.memory_interact_events):
        event_log.attach_forget_visibility(visibility)
    for overlay in (
            ctx.memory_read_overlay,
            ctx.memory_interact_overlay):
        overlay.attach_forget_visibility(visibility)
    source_intake.attach_forget_visibility(visibility)
    runtime = MemoryIsolationRuntime(
        (ctx.memory_read_events, ctx.memory_interact_events),
        (ctx.memory_read_overlay, ctx.memory_interact_overlay),
        (ctx.memory_read_aggregates, ctx.memory_interact_aggregates),
        source_repository,
        companions,
        forget_store,
        visibility,
        authorizer,
    )
    ctx.memory_forget_visibility = visibility
    ctx.memory_isolation_runtime = runtime
    runtime.recover_pending()
    return runtime


__all__ = [
    "FAULT_MEMORY_FORGET_AFTER_COMMIT",
    "FAULT_MEMORY_FORGET_AFTER_PROJECTION",
    "FAULT_MEMORY_FORGET_AFTER_STAGE",
    "ExportedCompanionAssoc",
    "ExportedMemoryEvent",
    "ExportedMemoryOverlay",
    "MemoryForgetFaultInjector",
    "MemoryForgetResult",
    "MemoryIsolationError",
    "MemoryIsolationRuntime",
    "MemoryLogicalExport",
    "install_memory_isolation_runtime",
]
