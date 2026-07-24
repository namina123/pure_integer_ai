"""M-08 从 A-10 真实选择到 Use 和延迟结果归因的运行边界。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorProcessingTrace,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_EPISODE,
    MEMORY_EVENT_OBSERVATION,
    MEMORY_EVENT_USE,
    MEMORY_EVENT_USE_OUTCOME,
    MEMORY_OBJECT_EPISODE,
    MEMORY_OBJECT_HYPOTHESIS,
    MEMORY_OBJECT_USE,
    EpisodePayload,
    MemoryEvent,
    MemoryLinkedRef,
    MemoryObjectRef,
    ObservationPayload,
    UseOutcomePayload,
    UsePayload,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_event_log import (
    MaterializedMemoryEvent,
    MemoryEventIntegrityError,
    MemoryEventLog,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.scope_identity import (
    LogicalTimestamp,
    SCOPE_QUERY,
    SCOPE_SESSION,
    ScopeIdentity,
)
from pure_integer_ai.experiments.train_context import TrainContext


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(value), *value


def _request_context_key(processing: AttractorProcessingTrace) -> tuple[int, ...]:
    """从完整 request 构造不含 reward、输出和历史回放的 context 键。"""
    request = processing.activation.request
    return (1, *_packed(request.stable_key()))


def _access_for(ref: MemoryObjectRef) -> MemoryAccessContext:
    """从目标 Memory owner 构造不提升 visibility 的读取上下文。"""
    return MemoryAccessContext(
        ref.owner.tenant_id,
        ref.owner.user_id,
        ref.owner.session_id,
    )


def _session_ancestor(scope: ScopeIdentity) -> ScopeIdentity:
    """沿显式 parent 链恢复当前 query 所属 session。"""
    current: ScopeIdentity | None = scope
    while current is not None:
        if current.scope_kind == SCOPE_SESSION:
            return current
        current = current.parent
    raise ValueError("query scope 缺少 session 祖先")


@dataclass(frozen=True)
class MemoryUseAttributionResult:
    """一个真实 selection Use 的 Episode、Use 和 A-10 处理链接。"""

    processing: AttractorProcessingTrace
    episode: MaterializedMemoryEvent
    use: MaterializedMemoryEvent

    def __post_init__(self) -> None:
        """核验两个声明事件和 processing trace 指向同一 Memory 候选。"""
        if not isinstance(self.processing, AttractorProcessingTrace):
            raise TypeError("processing 类型错误")
        if not isinstance(self.episode, MaterializedMemoryEvent):
            raise TypeError("episode 类型错误")
        if self.episode.event.event_kind != MEMORY_EVENT_EPISODE:
            raise ValueError("episode 不是 Episode event")
        if not isinstance(self.use, MaterializedMemoryEvent):
            raise TypeError("use 类型错误")
        if self.use.event.event_kind != MEMORY_EVENT_USE:
            raise ValueError("use 不是 Use event")
        payload = self.use.event.payload
        candidate = self.processing.activation.candidate.memory_ref
        if not isinstance(payload, UsePayload) or payload.memory_ref != candidate:
            raise ValueError("Use 没有引用 processing 的 Memory 候选")
        if payload.episode_ref != self.episode.event.object_ref:
            raise ValueError("Use 没有引用当前 Episode")
        if payload.decision_trace_key != self.processing.stable_key():
            raise ValueError("Use 没有链接完整 A-10 processing trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回 processing、Episode 和 Use 事件的完整稳定键。"""
        return (
            *_packed(self.processing.stable_key()),
            *_packed(self.episode.event.stable_key()),
            *_packed(self.use.event.stable_key()),
        )


class MemoryUseRuntime:
    """只为当前 A-10 frontier 真实选择写 Use，并支持延迟 outcome。"""

    def __init__(self, ctx: TrainContext, event_log: MemoryEventLog) -> None:
        """绑定 A-10 所属 Memory event log，拒绝另建私有写空间。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        if not isinstance(event_log, MemoryEventLog):
            raise TypeError("event_log 必须是 MemoryEventLog")
        if ctx.attractor_runtime is None:
            raise ValueError("安装 M-08 前必须先安装 A-10 runtime")
        resolver_log = (
            ctx.memory_resolver_runtime.resolver.aggregates.event_log)
        if event_log is not resolver_log:
            raise ValueError("M-08 必须写入 M-07 当前候选所属 Memory event log")
        self._ctx = ctx
        self.event_log = event_log
        self.protocol = ctx.attractor_runtime.protocol

    def record_selection_use(
            self,
            processing: AttractorProcessingTrace,
            *,
            input_observation_ref: MemoryObjectRef,
            influence_kind: MemoryLinkedRef,
            used_at: LogicalTimestamp,
            ) -> MemoryUseAttributionResult:
        """为实际消费的 frontier head 写唯一 Episode 和 Use，不接受仅命中或暂停项。"""
        state = self._ctx.work_memory.require_attractor_state()
        if not isinstance(processing, AttractorProcessingTrace):
            raise TypeError("processing 必须是 AttractorProcessingTrace")
        if processing not in state.processing_traces():
            raise ValueError("processing 不属于当前 WorkMemory query")
        if processing.decision.disposition != self.protocol.consumed:
            raise ValueError("只有 consumed activation 可以形成 Use")
        activation = processing.activation
        candidate_ref = activation.candidate.memory_ref
        if (candidate_ref is None
                or candidate_ref.object_kind != MEMORY_OBJECT_HYPOTHESIS):
            raise ValueError("M-08 只为实际使用的 Memory Hypothesis 写 Use")
        if candidate_ref.memory_space != self.event_log.memory_space_identity:
            raise ValueError("processing candidate 属于其他 Memory 空间")
        if not isinstance(influence_kind, MemoryLinkedRef):
            raise TypeError("influence_kind 必须是一等引用")
        if not isinstance(used_at, LogicalTimestamp):
            raise TypeError("used_at 必须是 LogicalTimestamp")
        if (used_at.clock.scope != state.scope
                or state.scope.scope_kind != SCOPE_QUERY
                or used_at.seq <= state.current_timestamp.seq):
            raise ValueError("Use 时钟必须在当前 query 内晚于最新 A-10 状态")
        observation = self._observation(input_observation_ref)
        if observation.source != activation.request.source:
            raise ValueError("Use 输入 Observation 与当前 query 来源不一致")

        by_key = {
            item.identity_key(): item for item in state.activations()}
        frontier = tuple(by_key[key] for key in processing.frontier_before)
        candidate_refs = tuple(
            MemoryLinkedRef.core(item.candidate.core_ref)
            if item.candidate.core_ref is not None
            else MemoryLinkedRef.memory(item.candidate.memory_ref)
            for item in frontier
        )
        selected_ref = MemoryLinkedRef.memory(candidate_ref)
        episode_payload = EpisodePayload(
            input_observation_ref,
            None,
            candidate_refs,
            selected_ref,
            None,
            (candidate_ref,),
            (),
            None,
            processing.ordinal,
            _session_ancestor(state.scope),
            used_at,
        )
        episode_ref = memory_object_ref(
            self.event_log.memory_space_identity,
            MEMORY_OBJECT_EPISODE,
            episode_payload.stable_key(),
            owner=state.scope.owner,
            versions=state.scope.versions,
        )
        query_kind = MemoryLinkedRef.object(activation.request.query_kind)
        context_key = _request_context_key(processing)
        use_payload = UsePayload(
            candidate_ref,
            episode_ref,
            influence_kind,
            None,
            used_at,
            processing.stable_key(),
            query_kind,
            context_key,
        )
        use_ref = memory_object_ref(
            self.event_log.memory_space_identity,
            MEMORY_OBJECT_USE,
            use_payload.identity_key(),
            owner=state.scope.owner,
            versions=state.scope.versions,
        )
        self._reject_competing_use(use_ref, use_payload)
        episode = self.event_log.append(MemoryEvent(
            MEMORY_EVENT_EPISODE, episode_ref, state.scope, episode_payload))
        use = self.event_log.append(MemoryEvent(
            MEMORY_EVENT_USE, use_ref, state.scope, use_payload))
        return MemoryUseAttributionResult(processing, episode, use)

    def record_outcome(
            self,
            use_ref: MemoryObjectRef,
            *,
            scope: ScopeIdentity,
            outcome_kind: MemoryLinkedRef,
            outcome_ref: MemoryLinkedRef | None,
            observed_at: LogicalTimestamp,
            ) -> MaterializedMemoryEvent:
        """把延迟结果追加到一个精确 Use，不向同 query 的其他候选扩散。"""
        use = self._use_payload(use_ref)
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        if (scope.owner != use_ref.owner
                or scope.versions != use_ref.versions):
            raise ValueError("outcome scope 与 Use owner/version 不一致")
        if not isinstance(outcome_kind, MemoryLinkedRef):
            raise TypeError("outcome_kind 必须是一等引用")
        if outcome_ref is not None and not isinstance(
                outcome_ref, MemoryLinkedRef):
            raise TypeError("outcome_ref 必须是一等引用或 None")
        if not isinstance(observed_at, LogicalTimestamp):
            raise TypeError("observed_at 必须是 LogicalTimestamp")
        if (observed_at.clock.scope.owner != scope.owner
                or observed_at.clock.scope.versions != scope.versions):
            raise ValueError("outcome 时钟与 scope owner/version 不一致")
        payload = UseOutcomePayload(
            use_ref,
            use.decision_trace_key,
            use.query_kind,
            use.context_key,
            outcome_kind,
            outcome_ref,
            observed_at,
        )
        return self.event_log.append(MemoryEvent(
            MEMORY_EVENT_USE_OUTCOME, use_ref, scope, payload))

    def _observation(self, ref: MemoryObjectRef) -> ObservationPayload:
        """恢复唯一可见 Observation 声明并核验对象种类。"""
        if not isinstance(ref, MemoryObjectRef):
            raise TypeError("input_observation_ref 必须是 MemoryObjectRef")
        events = self.event_log.query(
            access=_access_for(ref),
            event_kind=MEMORY_EVENT_OBSERVATION,
            object_ref=ref,
        )
        if len(events) != 1 or not isinstance(
                events[0].event.payload, ObservationPayload):
            raise ValueError("input_observation_ref 没有唯一 Observation 声明")
        return events[0].event.payload

    def _use_payload(self, ref: MemoryObjectRef) -> UsePayload:
        """恢复一个带完整 M-08 trace 的唯一 Use 声明。"""
        if (not isinstance(ref, MemoryObjectRef)
                or ref.object_kind != MEMORY_OBJECT_USE):
            raise ValueError("use_ref 必须指向 Use")
        events = self.event_log.query(
            access=_access_for(ref),
            event_kind=MEMORY_EVENT_USE,
            object_ref=ref,
        )
        if len(events) != 1 or not isinstance(
                events[0].event.payload, UsePayload):
            raise ValueError("use_ref 没有唯一 Use 声明")
        payload = events[0].event.payload
        if (not payload.decision_trace_key
                or payload.query_kind is None
                or not payload.context_key):
            raise ValueError("use_ref 只指向兼容 Use，不能接收 M-08 outcome")
        return payload

    def _reject_competing_use(
            self,
            use_ref: MemoryObjectRef,
            payload: UsePayload,
            ) -> None:
        """按唯一 Use 对象索引拒绝同 processing 的竞争归因，不扫描历史。"""
        events = self.event_log.query(
            access=_access_for(use_ref),
            event_kind=MEMORY_EVENT_USE,
            object_ref=use_ref,
        )
        if not events:
            return
        if len(events) != 1 or not isinstance(
                events[0].event.payload, UsePayload):
            raise MemoryEventIntegrityError("Use 对象没有唯一声明")
        if events[0].event.payload != payload:
            raise MemoryEventIntegrityError(
                "同一 processing trace 存在竞争 Use 归因")

    def state_key(self) -> tuple[int, ...]:
        """返回绑定 Memory 空间和 A-10 状态协议的配置键。"""
        return (
            2,
            *self.event_log.memory_space_identity.stable_key(),
            *self.protocol.stable_key(),
        )

    def clone_for_context(self, ctx: TrainContext) -> "MemoryUseRuntime":
        """为 V-06 重绑同 identity 的独立 Memory event log。"""
        matches = tuple(
            item for item in (
                ctx.memory_read_events,
                ctx.memory_interact_events,
            )
            if (item is not None
                and item.memory_space_identity
                == self.event_log.memory_space_identity)
        )
        if len(matches) != 1:
            raise ValueError("评测上下文缺少唯一同 identity Memory event log")
        cloned = MemoryUseRuntime(ctx, matches[0])
        if cloned.state_key() != self.state_key():
            raise ValueError("M-08 runtime clone 改变了协议状态")
        return cloned


def install_memory_use_runtime(ctx: TrainContext) -> MemoryUseRuntime:
    """在已安装 A-10 的上下文上装配唯一 M-08 Use runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("ctx 必须是 TrainContext")
    if ctx.memory_use_runtime is not None:
        raise ValueError("TrainContext 已安装 M-08 runtime")
    event_log = ctx.memory_resolver_runtime.resolver.aggregates.event_log
    runtime = MemoryUseRuntime(ctx, event_log)
    ctx.memory_use_runtime = runtime
    return runtime


__all__ = [
    "MemoryUseAttributionResult",
    "MemoryUseRuntime",
    "install_memory_use_runtime",
]
