"""C-02 从 A-10 frontier Capability 到 A-06、Use 或失败修正的纵切。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.artifact_binding import (
    ArtifactBindingRequest,
    ArtifactBindingRun,
)
from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorConsumptionDecision,
    AttractorProcessingTrace,
)
from pure_integer_ai.cognition.shared.capability_memory import (
    recover_verified_capability,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_CAPABILITY_ATTEMPT_OUTCOME,
    MEMORY_OBJECT_CAPABILITY,
    CapabilityAttemptOutcomePayload,
    MemoryEvent,
    MemoryLinkedRef,
    MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.memory_event_log import (
    MaterializedMemoryEvent,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.scope_identity import LogicalTimestamp
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.artifact_binding_runtime import (
    ArtifactBindingRuntime,
)
from pure_integer_ai.experiments.memory_use_runtime import (
    MemoryUseAttributionResult,
    MemoryUseRuntime,
)
from pure_integer_ai.experiments.train_context import TrainContext


_BINDING_RUN_DOMAIN = "capability.memory.binding_run.v1"


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给执行协议的开放稳定键增加长度边界。"""
    return len(value), *value


def _require_instruction(value: ObjectIdentity, *, label: str) -> None:
    """核验 Capability consumer 是图内一等最小指令。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")


@dataclass(frozen=True)
class CapabilityExecutionProtocol:
    """Capability 消费、成功影响和失败结果使用的注入式身份。"""

    consumer: ObjectIdentity
    influence_kind: MemoryLinkedRef
    failure_outcome_kind: MemoryLinkedRef

    def __post_init__(self) -> None:
        """核验三项协议均为一等身份，不在运行时解释文字。"""
        _require_instruction(self.consumer, label="Capability consumer")
        if not isinstance(self.influence_kind, MemoryLinkedRef):
            raise TypeError("Capability influence_kind 必须是一等引用")
        if not isinstance(self.failure_outcome_kind, MemoryLinkedRef):
            raise TypeError("Capability failure_outcome_kind 必须是一等引用")

    def stable_key(self) -> tuple[int, ...]:
        """返回 consumer、成功影响和失败类型的稳定协议键。"""
        return (
            1,
            *_packed(self.consumer.stable_key()),
            *_packed(self.influence_kind.stable_key()),
            *_packed(self.failure_outcome_kind.stable_key()),
        )


@dataclass(frozen=True)
class CapabilityExecutionResult:
    """一次 frontier 尝试的 A-06 run、A-10 trace 与唯一归因分支。"""

    binding_run: ArtifactBindingRun
    processing: AttractorProcessingTrace
    use: MemoryUseAttributionResult | None
    failure: MaterializedMemoryEvent | None

    def __post_init__(self) -> None:
        """核验成功只带 Use，失败只带 CapabilityAttemptOutcome。"""
        if not isinstance(self.binding_run, ArtifactBindingRun):
            raise TypeError("Capability execution binding_run 类型错误")
        if not isinstance(self.processing, AttractorProcessingTrace):
            raise TypeError("Capability execution processing 类型错误")
        if self.binding_run.succeeded:
            if not isinstance(self.use, MemoryUseAttributionResult):
                raise TypeError("成功 Capability execution 必须带 M-08 Use")
            if self.failure is not None:
                raise ValueError("成功 Capability execution 不得带失败事件")
        else:
            if self.use is not None:
                raise ValueError("失败 Capability execution 不得形成 Use")
            if (not isinstance(self.failure, MaterializedMemoryEvent)
                    or self.failure.event.event_kind
                    != MEMORY_EVENT_CAPABILITY_ATTEMPT_OUTCOME):
                raise TypeError("失败 Capability execution 必须带精确 outcome")


class CapabilityExecutionRuntime:
    """只处理当前 A-10 frontier Capability，并提交成功或失败唯一分支。"""

    def __init__(
            self,
            ctx: TrainContext,
            binding_runtime: ArtifactBindingRuntime,
            memory_use_runtime: MemoryUseRuntime,
            protocol: CapabilityExecutionProtocol,
            ) -> None:
        """绑定同一 WorkMemory、interact event log、A-06 和 M-08 runtime。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        if not isinstance(binding_runtime, ArtifactBindingRuntime):
            raise TypeError("binding_runtime 必须是 ArtifactBindingRuntime")
        if not isinstance(memory_use_runtime, MemoryUseRuntime):
            raise TypeError("memory_use_runtime 必须是 MemoryUseRuntime")
        if not isinstance(protocol, CapabilityExecutionProtocol):
            raise TypeError("protocol 必须是 CapabilityExecutionProtocol")
        if ctx.attractor_runtime is None:
            raise ValueError("C-02 execution 前必须安装 A-10 runtime")
        if binding_runtime.work_memory is not ctx.work_memory:
            raise ValueError("A-06 runtime 未绑定当前 WorkMemory")
        if memory_use_runtime.event_log is not ctx.memory_interact_events:
            raise ValueError("M-08 runtime 未绑定当前 interact Memory")
        self._ctx = ctx
        self.binding_runtime = binding_runtime
        self.memory_use_runtime = memory_use_runtime
        self.protocol = protocol

    def execute_frontier(
            self,
            request: ArtifactBindingRequest,
            *,
            input_observation_ref: MemoryObjectRef,
            used_at: LogicalTimestamp,
            failed_at: LogicalTimestamp,
            failure_outcome_ref: MemoryLinkedRef | None = None,
            ) -> CapabilityExecutionResult:
        """执行 frontier A-06；成功 consumed+Use，形式失败 suspended+精确 outcome。"""
        state = self._ctx.work_memory.require_attractor_state()
        activation = state.next_activation()
        if activation is None:
            raise RuntimeError("当前 A-10 agenda 没有 frontier activation")
        candidate_ref = activation.candidate.memory_ref
        if (candidate_ref is None
                or candidate_ref.object_kind != MEMORY_OBJECT_CAPABILITY
                or activation.candidate.capability is None):
            raise ValueError("当前 frontier 不是 Capability")
        access = MemoryAccessContext(
            activation.request.access.tenant_id,
            activation.request.access.user_id,
            activation.request.access.session_id,
        )
        recovered = recover_verified_capability(
            self.memory_use_runtime.event_log,
            candidate_ref,
            access=access,
        )
        self._validate_request(request, activation, recovered.definition)
        run = self.binding_runtime.run(request)
        disposition = (
            state.protocol.consumed if run.succeeded
            else state.protocol.suspended)
        decision = AttractorConsumptionDecision(
            activation.identity_key(),
            self.protocol.consumer,
            disposition,
            integer_tuple_fingerprint(
                run.stable_key(), domain=_BINDING_RUN_DOMAIN),
        )
        processing = state.commit_consumption(decision)
        if run.succeeded:
            use = self.memory_use_runtime.record_selection_use(
                processing,
                input_observation_ref=input_observation_ref,
                influence_kind=self.protocol.influence_kind,
                used_at=used_at,
            )
            return CapabilityExecutionResult(run, processing, use, None)
        failure = self._record_failure(
            processing,
            run,
            failed_at=failed_at,
            outcome_ref=failure_outcome_ref,
        )
        return CapabilityExecutionResult(run, processing, None, failure)

    def _validate_request(
            self,
            request: ArtifactBindingRequest,
            activation,
            definition,
            ) -> None:
        """核验 A-06 请求绑定恢复 definition、当前 query 和当前义务。"""
        if not isinstance(request, ArtifactBindingRequest):
            raise TypeError("request 必须是 ArtifactBindingRequest")
        if request.definition != definition:
            raise ValueError("A-06 binding definition 与恢复 Capability 漂移")
        if (request.scope != activation.request.scope
                or request.source != activation.request.source):
            raise ValueError("A-06 binding request 不属于 Capability 当前 query")
        if request.proposition != activation.obligation.proposition.template:
            raise ValueError("A-06 binding proposition 与当前 obligation 漂移")

    def _record_failure(
            self,
            processing: AttractorProcessingTrace,
            run: ArtifactBindingRun,
            *,
            failed_at: LogicalTimestamp,
            outcome_ref: MemoryLinkedRef | None,
            ) -> MaterializedMemoryEvent:
        """把形式失败追加到目标 Capability，不创建 Use 或修改其他候选。"""
        candidate_ref = processing.activation.candidate.memory_ref
        assert candidate_ref is not None
        if not isinstance(failed_at, LogicalTimestamp):
            raise TypeError("failed_at 必须是 LogicalTimestamp")
        scope = failed_at.clock.scope
        if (scope.owner != candidate_ref.owner
                or scope.versions != candidate_ref.versions):
            raise ValueError("Capability failure 时钟与目标 owner/version 漂移")
        if outcome_ref is not None and not isinstance(
                outcome_ref, MemoryLinkedRef):
            raise TypeError("failure_outcome_ref 必须是一等引用或 None")
        payload = CapabilityAttemptOutcomePayload(
            candidate_ref,
            MemoryLinkedRef.object(
                processing.activation.request.query_kind),
            integer_tuple_fingerprint(
                run.stable_key(), domain=_BINDING_RUN_DOMAIN),
            processing.stable_key(),
            self.protocol.failure_outcome_kind,
            outcome_ref,
            failed_at,
        )
        return self.memory_use_runtime.event_log.append(MemoryEvent(
            MEMORY_EVENT_CAPABILITY_ATTEMPT_OUTCOME,
            candidate_ref,
            scope,
            payload,
        ))

    def state_key(self) -> tuple[int, ...]:
        """返回执行协议、目标 Memory 空间和 A-10 状态协议键。"""
        return (
            1,
            *self.memory_use_runtime.event_log.memory_space_identity.stable_key(),
            *self.protocol.stable_key(),
            *self._ctx.attractor_runtime.protocol.stable_key(),
        )


__all__ = [
    "CapabilityExecutionProtocol",
    "CapabilityExecutionResult",
    "CapabilityExecutionRuntime",
]
