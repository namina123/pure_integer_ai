"""C-02 verified Capability 的原子保存与恢复运行边界。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.capability_memory import (
    RecoveredCapability,
    VerifiedCapabilityContract,
    recover_verified_capability,
)
from pure_integer_ai.cognition.shared.capability_verification import (
    CapabilityVerificationReport,
)
from pure_integer_ai.cognition.shared.memory_batch import (
    MemoryBatchFaultInjector,
    MemoryBatchPublishResult,
    MemoryBatchRuntime,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_ARTIFACT,
    MEMORY_EVENT_CAPABILITY,
    MEMORY_OBJECT_ARTIFACT,
    MEMORY_OBJECT_CAPABILITY,
    ArtifactPayload,
    CapabilityPayload,
    MemoryEvent,
    MemoryObjectRef,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.scope_identity import LogicalTimestamp
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.storage.segment_dependency import SegmentDependency


@dataclass(frozen=True)
class CapabilityMemoryPublishResult:
    """一次原子发布产生的 batch、Capability 引用和恢复结果。"""

    report: CapabilityVerificationReport
    batch: MemoryBatchPublishResult
    capability_ref: MemoryObjectRef
    recovered: RecoveredCapability


class CapabilityMemoryRuntime:
    """只通过 M-10/K-02 batch 发布和恢复 verified Capability。"""

    def __init__(
            self,
            ctx: TrainContext,
            batch_runtime: MemoryBatchRuntime | None = None,
            ) -> None:
        """绑定当前 TrainContext 的 interact Memory batch，不创建旁路 writer。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        runtime = batch_runtime or ctx.memory_interact_batch_runtime
        if not isinstance(runtime, MemoryBatchRuntime):
            raise ValueError("C-02 保存前必须安装 M-10 interact batch runtime")
        if runtime.event_log is not ctx.memory_interact_events:
            raise ValueError("C-02 batch runtime 不属于当前 interact Memory")
        self._ctx = ctx
        self.batch_runtime = runtime
        self.event_log = runtime.event_log

    def publish_verified(
            self,
            report: CapabilityVerificationReport,
            *,
            batch_id: int,
            source_dependency: SegmentDependency,
            created_at: LogicalTimestamp,
            evidence_refs: tuple[MemoryObjectRef, ...] = (),
            fault_injector: MemoryBatchFaultInjector | None = None,
            ) -> CapabilityMemoryPublishResult:
        """原子发布 program Artifact 与 Capability，activation 后才允许恢复。"""
        contract = VerifiedCapabilityContract.from_report(report)
        source = contract.candidate_source
        if not isinstance(created_at, LogicalTimestamp):
            raise TypeError("created_at 必须是 LogicalTimestamp")
        scope = created_at.clock.scope
        if (scope.owner != source.owner
                or scope.versions != source.versions
                or (scope.source is not None and scope.source != source)):
            raise ValueError("Capability created_at scope 与 candidate source 漂移")
        if (not isinstance(evidence_refs, tuple)
                or any(not isinstance(item, MemoryObjectRef)
                       for item in evidence_refs)):
            raise TypeError("evidence_refs 必须是 MemoryObjectRef tuple")
        definition = contract.definition
        artifact_payload = ArtifactPayload(
            definition.program.identity,
            None,
            created_at,
        )
        artifact_ref = memory_object_ref(
            self.event_log.memory_space_identity,
            MEMORY_OBJECT_ARTIFACT,
            definition.program.identity.stable_key(),
            owner=source.owner,
            versions=source.versions,
        )
        capability_payload = CapabilityPayload(
            contract.capability_kind,
            artifact_ref,
            contract.stable_key(),
            evidence_refs,
            created_at,
        )
        capability_ref = memory_object_ref(
            self.event_log.memory_space_identity,
            MEMORY_OBJECT_CAPABILITY,
            capability_payload.stable_key(),
            owner=source.owner,
            versions=source.versions,
        )
        events = (
            MemoryEvent(
                MEMORY_EVENT_ARTIFACT,
                artifact_ref,
                scope,
                artifact_payload,
            ),
            MemoryEvent(
                MEMORY_EVENT_CAPABILITY,
                capability_ref,
                scope,
                capability_payload,
            ),
        )
        batch = self.batch_runtime.publish(
            source,
            batch_id,
            events,
            source_dependency=source_dependency,
            fault_injector=fault_injector,
        )
        recovered = self.recover(
            capability_ref,
            access=MemoryAccessContext(
                source.owner.tenant_id,
                source.owner.user_id,
                source.owner.session_id,
            ),
        )
        return CapabilityMemoryPublishResult(
            report, batch, capability_ref, recovered)

    def recover(
            self,
            capability_ref: MemoryObjectRef,
            *,
            access: MemoryAccessContext,
            ) -> RecoveredCapability:
        """经当前 interact event log 只读恢复 verified Capability。"""
        return recover_verified_capability(
            self.event_log, capability_ref, access=access)

    def state_key(self) -> tuple[int, ...]:
        """返回绑定 Memory 空间和 batch 可见性状态的诊断键。"""
        return (
            1,
            *self.event_log.memory_space_identity.stable_key(),
            *self.batch_runtime.visibility.state_key(),
        )


__all__ = [
    "CapabilityMemoryPublishResult",
    "CapabilityMemoryRuntime",
]
