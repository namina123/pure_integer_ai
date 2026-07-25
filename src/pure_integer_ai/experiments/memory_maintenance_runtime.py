"""M-09 维护服务的 TrainContext 生命周期和评测隔离装配。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.memory_maintenance import (
    MemoryMaintenanceService,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.experiments.train_context import TrainContext


class MemoryMaintenanceRuntime:
    """把 M-09 维护限制在当前上下文的 Memory 和 Use 边界内。"""

    def __init__(
            self,
            ctx: TrainContext,
            service: MemoryMaintenanceService,
            ) -> None:
        """绑定上下文和 M-09 service，拒绝脱离真实 M-08 consumer 的装配。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        if not isinstance(service, MemoryMaintenanceService):
            raise TypeError("service 必须是 MemoryMaintenanceService")
        if ctx.memory_use_runtime is None:
            raise ValueError("安装 M-09 前必须先安装 M-08 runtime")
        if all(service.aggregates is not item for item in (
                ctx.memory_read_aggregates,
                ctx.memory_interact_aggregates,
        )):
            raise ValueError("M-09 aggregate 不属于当前 TrainContext")
        if service.event_log is not ctx.memory_use_runtime.event_log:
            raise ValueError("M-09 必须绑定 M-08 使用的 Memory event log")
        self._ctx = ctx
        self.service = service

    def assess(
            self,
            hypothesis_ref: MemoryObjectRef,
            *,
            access: MemoryAccessContext,
            ):
        """在不写事件的前提下返回当前 Hypothesis 的 M-09 评估。"""
        return self.service.assess(hypothesis_ref, access=access)

    def consolidate(
            self,
            hypothesis_ref: MemoryObjectRef,
            *,
            access: MemoryAccessContext,
            ):
        """按注入 retention 规则执行一次巩固尝试。"""
        return self.service.consolidate(hypothesis_ref, access=access)

    def resolve_lifecycle(self, *args, **kwargs):
        """把生命周期修正转发给已绑定 H-04 的 M-09 service。"""
        return self.service.resolve_lifecycle(*args, **kwargs)

    def clone_for_context(self, ctx: TrainContext) -> "MemoryMaintenanceRuntime":
        """为 V-06 克隆重绑同 identity Memory，并复制可克隆策略。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        matches = tuple(
            item for item in (
                ctx.memory_read_aggregates,
                ctx.memory_interact_aggregates,
            )
            if (isinstance(item, type(self.service.aggregates))
                and item.event_log.memory_space_identity
                == self.service.event_log.memory_space_identity)
        )
        if len(matches) != 1:
            raise ValueError("评测上下文缺少唯一同 identity aggregate")

        def cloned(component: object) -> object:
            """调用策略可选的上下文克隆协议，否则复用只读策略。"""
            method = getattr(component, "clone_for_context", None)
            if method is None:
                return component
            if not callable(method):
                raise TypeError("M-09 策略 clone_for_context 不可调用")
            return method(ctx)

        service = MemoryMaintenanceService(
            matches[0],
            cloned(self.service.activation_policy),
            cloned(self.service.retention_policy),
            cloned(self.service.placement_policy),
            self.service.storage_roles,
            self.service.temperature_profile,
        )
        if service.state_key() != self.service.state_key():
            raise ValueError("M-09 clone 改变了协议状态")
        return MemoryMaintenanceRuntime(ctx, service)

    def state_key(self) -> tuple[int, ...]:
        """返回 M-09 service 的完整注入状态。"""
        return self.service.state_key()


def install_memory_maintenance_runtime(
        ctx: TrainContext,
        service: MemoryMaintenanceService,
        ) -> MemoryMaintenanceRuntime:
    """在已装配 M-08 的 TrainContext 上安装唯一 M-09 runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("ctx 必须是 TrainContext")
    if not isinstance(service, MemoryMaintenanceService):
        raise TypeError("service 必须是 MemoryMaintenanceService")
    if ctx.memory_maintenance_runtime is not None:
        raise ValueError("TrainContext 已安装 M-09 runtime")
    runtime = MemoryMaintenanceRuntime(ctx, service)
    ctx.memory_maintenance_runtime = runtime
    return runtime


__all__ = [
    "MemoryMaintenanceRuntime",
    "install_memory_maintenance_runtime",
]
