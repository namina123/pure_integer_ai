"""A-10 当前 query 的动态方向状态、agenda 和局部重算协议。

长期 Memory 只通过 M-07 ``MemoryResolution`` 提供可追溯候选。本模块要求
调用方把候选映射到当前 ``ReasoningObligation``，并保存评分理由、依赖和处理
状态；它不读取 PR、replay、reward，也不判断命题真值。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    SourceRef,
    TypedRef,
)
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.memory_query import MemoryActivationRequest
from pure_integer_ai.cognition.shared.memory_resolver import (
    ActivationScoreReason,
    MemoryResolution,
    ResolvedCandidate,
)
from pure_integer_ai.cognition.shared.reasoning_planner import ReasoningObligation
from pure_integer_ai.cognition.shared.scope_identity import (
    LogicalTimestamp,
    SCOPE_QUERY,
    ScopeIdentity,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


ATTRACTOR_PROTOCOL_VERSION = 1


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界，避免不同字段拼接碰撞。"""
    return len(value), *value


def _strict_key(
        value: tuple[int, ...], *, label: str, allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """校验开放整数键，拒绝 bool、浮点和其他非整数值。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是整数 tuple")
    if not value and not allow_empty:
        raise ValueError(f"{label} 不得为空")
    if value:
        assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _require_instruction(value: ObjectIdentity, *, label: str) -> None:
    """要求状态、动作和理由均由图内最小指令身份注入。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")


@dataclass(frozen=True)
class AttractorProtocol:
    """注入 agenda、已消费、暂停和替代四个状态身份。"""

    agenda: ObjectIdentity
    consumed: ObjectIdentity
    suspended: ObjectIdentity
    superseded: ObjectIdentity

    def __post_init__(self) -> None:
        """核验四个状态均为互异的一等最小指令。"""
        for label, value in zip(
                ("agenda", "consumed", "suspended", "superseded"),
                self.values()):
            _require_instruction(value, label=f"AttractorProtocol.{label}")
        if len(set(self.values())) != len(self.values()):
            raise ValueError("AttractorProtocol 状态身份不得重复")

    def values(self) -> tuple[ObjectIdentity, ...]:
        """按状态机固定槽位顺序返回全部注入身份。"""
        return self.agenda, self.consumed, self.suspended, self.superseded

    def stable_key(self) -> tuple[int, ...]:
        """返回全部图内状态身份的稳定协议键。"""
        result = [ATTRACTOR_PROTOCOL_VERSION]
        for value in self.values():
            result.extend(_packed(value.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class AttractorBudget:
    """限制进入 agenda、实际处理和局部重算的 query 级预算。"""

    max_agenda_entries: int
    max_consumptions: int
    max_recomputations: int

    def __post_init__(self) -> None:
        """核验预算为严格整数，并允许显式关闭局部重算。"""
        assert_int(
            self.max_agenda_entries,
            self.max_consumptions,
            self.max_recomputations,
            _where="AttractorBudget",
        )
        values = (
            self.max_agenda_entries,
            self.max_consumptions,
            self.max_recomputations,
        )
        if any(type(value) is not int for value in values):
            raise ValueError("AttractorBudget 必须使用严格整数")
        if self.max_agenda_entries <= 0 or self.max_consumptions <= 0:
            raise ValueError("agenda 和 consumption 预算必须为正整数")
        if self.max_recomputations < 0:
            raise ValueError("recomputation 预算不得为负数")

    def stable_key(self) -> tuple[int, ...]:
        """返回冻结预算键，供运行期和隔离克隆核验。"""
        return (
            self.max_agenda_entries,
            self.max_consumptions,
            self.max_recomputations,
        )


_DEPENDENCY_OBJECT = 1
_DEPENDENCY_TYPED_REF = 2
_DEPENDENCY_SOURCE = 3
_DEPENDENCY_SCOPE = 4
_DEPENDENCY_HYPOTHESIS = 5
_DEPENDENCY_MEMORY_REF = 6


AttractorDependencyValue = (
    ObjectIdentity
    | TypedRef
    | SourceRef
    | ScopeIdentity
    | HypothesisKey
    | MemoryObjectRef
)


@dataclass(frozen=True)
class AttractorDependency:
    """一个带图内角色的 typed 局部重算依赖。"""

    role: ObjectIdentity
    value: AttractorDependencyValue

    def __post_init__(self) -> None:
        """拒绝裸整数、字符串和无角色派生 hash 进入依赖集合。"""
        _require_instruction(self.role, label="AttractorDependency.role")
        if not isinstance(self.value, (
                ObjectIdentity,
                TypedRef,
                SourceRef,
                ScopeIdentity,
                HypothesisKey,
                MemoryObjectRef,
                )):
            raise TypeError("AttractorDependency.value 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回角色、值类型和完整身份的无歧义依赖键。"""
        if isinstance(self.value, TypedRef):
            kind = _DEPENDENCY_TYPED_REF
        elif isinstance(self.value, SourceRef):
            kind = _DEPENDENCY_SOURCE
        elif isinstance(self.value, ScopeIdentity):
            kind = _DEPENDENCY_SCOPE
        elif isinstance(self.value, HypothesisKey):
            kind = _DEPENDENCY_HYPOTHESIS
        elif isinstance(self.value, MemoryObjectRef):
            kind = _DEPENDENCY_MEMORY_REF
        else:
            kind = _DEPENDENCY_OBJECT
        return (
            *_packed(self.role.stable_key()),
            kind,
            *_packed(self.value.stable_key()),
        )


def _normalized_dependencies(
        values: tuple[AttractorDependency, ...], *, label: str,
        ) -> tuple[AttractorDependency, ...]:
    """校验依赖非空、无重复并规范为稳定顺序。"""
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{label} 必须是非空 tuple")
    if any(not isinstance(item, AttractorDependency) for item in values):
        raise TypeError(f"{label} 含非法依赖")
    by_key = {item.stable_key(): item for item in values}
    if len(by_key) != len(values):
        raise ValueError(f"{label} 不得重复")
    return tuple(by_key[key] for key in sorted(by_key))


@dataclass(frozen=True)
class AttractorScoreReason:
    """图内理由身份及其对当前目标方向分的整数调整。"""

    reason: ObjectIdentity
    value: int
    dependencies: tuple[AttractorDependency, ...]

    def __post_init__(self) -> None:
        """核验理由身份、整数值和可追溯依赖。"""
        _require_instruction(self.reason, label="AttractorScoreReason.reason")
        assert_int(self.value, _where="AttractorScoreReason.value")
        if type(self.value) is not int:
            raise ValueError("AttractorScoreReason.value 必须是严格整数")
        object.__setattr__(self, "dependencies", _normalized_dependencies(
            self.dependencies, label="AttractorScoreReason.dependencies"))

    def stable_key(self) -> tuple[int, ...]:
        """返回图内理由、调整值和全部依赖的稳定键。"""
        result = [*_packed(self.reason.stable_key()), self.value,
                  len(self.dependencies)]
        for dependency in self.dependencies:
            result.extend(_packed(dependency.stable_key()))
        return tuple(result)


def _normalized_reasons(
        reasons: tuple[AttractorScoreReason, ...], *, label: str,
        ) -> tuple[AttractorScoreReason, ...]:
    """校验方向理由非空、身份完整且不重复。"""
    if not isinstance(reasons, tuple) or not reasons:
        raise ValueError(f"{label} 必须是非空 tuple")
    if any(not isinstance(item, AttractorScoreReason) for item in reasons):
        raise TypeError(f"{label} 含非法理由")
    keys = tuple(item.stable_key() for item in reasons)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} 不得重复")
    return tuple(sorted(reasons, key=lambda item: item.stable_key()))


@dataclass(frozen=True)
class AttractorActivationProposal:
    """mapper 将一个 M-07 候选投影到当前推理目标的提案。"""

    activation_kind: ObjectIdentity
    obligation: ReasoningObligation
    score_adjustment: int
    score_reasons: tuple[AttractorScoreReason, ...]
    dependencies: tuple[AttractorDependency, ...]

    def __post_init__(self) -> None:
        """核验提案携带一等作用身份、typed 目标和完整评分依据。"""
        _require_instruction(
            self.activation_kind,
            label="AttractorActivationProposal.activation_kind",
        )
        if not isinstance(self.obligation, ReasoningObligation):
            raise TypeError("AttractorActivationProposal.obligation 类型错误")
        assert_int(
            self.score_adjustment,
            _where="AttractorActivationProposal.score_adjustment",
        )
        if type(self.score_adjustment) is not int:
            raise ValueError("score_adjustment 必须是严格整数")
        object.__setattr__(self, "score_reasons", _normalized_reasons(
            self.score_reasons,
            label="AttractorActivationProposal.score_reasons",
        ))
        object.__setattr__(self, "dependencies", _normalized_dependencies(
            self.dependencies,
            label="AttractorActivationProposal.dependencies",
        ))

    def stable_key(self) -> tuple[int, ...]:
        """返回不含来源候选的目标投影稳定键。"""
        result = [
            *_packed(self.activation_kind.stable_key()),
            *_packed(self.obligation.stable_key()),
            self.score_adjustment,
            len(self.score_reasons),
        ]
        for reason in self.score_reasons:
            result.extend(_packed(reason.stable_key()))
        result.append(len(self.dependencies))
        for dependency in self.dependencies:
            result.extend(_packed(dependency.stable_key()))
        return tuple(result)


class AttractorActivationMapper(Protocol):
    """把 M-07 候选与当前目标义务关联，不在状态机中写语言规则。"""

    def project(
            self,
            request: MemoryActivationRequest,
            candidate: ResolvedCandidate,
            obligations: tuple[ReasoningObligation, ...],
            ) -> tuple[AttractorActivationProposal, ...]:
        """返回零个或多个当前 query 提案，不得改写候选和目标。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回 mapper 版本和注入配置的非空整数键。"""
        ...


@dataclass(frozen=True)
class AttractorActivation:
    """进入当前 agenda 的完整 M-07 候选、推理目标和动态状态。"""

    request: MemoryActivationRequest
    candidate: ResolvedCandidate
    activation_kind: ObjectIdentity
    obligation: ReasoningObligation
    score_adjustment: int
    score_reasons: tuple[AttractorScoreReason, ...]
    dependencies: tuple[AttractorDependency, ...]
    status: ObjectIdentity
    revision: int = 0

    def __post_init__(self) -> None:
        """核验候选、目标、scope、理由和动态状态没有身份漂移。"""
        if not isinstance(self.request, MemoryActivationRequest):
            raise TypeError("AttractorActivation.request 类型错误")
        if not isinstance(self.candidate, ResolvedCandidate):
            raise TypeError("AttractorActivation.candidate 类型错误")
        _require_instruction(
            self.activation_kind, label="AttractorActivation.activation_kind")
        if not isinstance(self.obligation, ReasoningObligation):
            raise TypeError("AttractorActivation.obligation 类型错误")
        if self.candidate.query_scope != self.request.scope:
            raise ValueError("AttractorActivation 候选与 request scope 不一致")
        if self.obligation.scope != self.request.scope:
            raise ValueError("AttractorActivation obligation 必须属于当前 query")
        if self.obligation.source != self.request.source:
            raise ValueError("AttractorActivation obligation 必须属于当前输入来源")
        assert_int(
            self.score_adjustment,
            self.revision,
            _where="AttractorActivation",
        )
        if type(self.score_adjustment) is not int:
            raise ValueError("AttractorActivation.score_adjustment 必须是严格整数")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("AttractorActivation.revision 必须是非负严格整数")
        object.__setattr__(self, "score_reasons", _normalized_reasons(
            self.score_reasons, label="AttractorActivation.score_reasons"))
        object.__setattr__(self, "dependencies", _normalized_dependencies(
            self.dependencies, label="AttractorActivation.dependencies"))
        _require_instruction(self.status, label="AttractorActivation.status")

    @property
    def score(self) -> int:
        """合并 M-07 基础分和当前目标调整，不改写 resolver trace。"""
        return self.candidate.score + self.score_adjustment

    def identity_key(self) -> tuple[int, ...]:
        """返回不随重算变化的 request、候选、作用和目标身份。"""
        return (
            ATTRACTOR_PROTOCOL_VERSION,
            *_packed(self.request.stable_key()),
            *_packed(self.candidate.stable_key()),
            *_packed(self.activation_kind.stable_key()),
            *_packed(self.obligation.stable_key()),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回当前评分、理由、依赖、状态和 revision 的完整快照键。"""
        result = [
            *_packed(self.identity_key()),
            self.candidate.score,
            self.score_adjustment,
            self.score,
            self.revision,
            *_packed(self.status.stable_key()),
            len(self.candidate.score_reasons),
        ]
        for reason in self.candidate.score_reasons:
            if not isinstance(reason, ActivationScoreReason):
                raise TypeError("M-07 score reason 类型错误")
            result.extend(_packed(reason.stable_key()))
        result.append(len(self.score_reasons))
        for reason in self.score_reasons:
            result.extend(_packed(reason.stable_key()))
        result.append(len(self.dependencies))
        for dependency in self.dependencies:
            result.extend(_packed(dependency.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class AttractorContextUpdate:
    """同一 query 内由后文或新 typed 上下文触发的局部变化。"""

    scope: ScopeIdentity
    logical_timestamp: LogicalTimestamp
    reason: ObjectIdentity
    changed_dependencies: tuple[AttractorDependency, ...]

    def __post_init__(self) -> None:
        """核验更新属于 query 时钟，并至少声明一个具体变化依赖。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("AttractorContextUpdate.scope 类型错误")
        if self.scope.scope_kind != SCOPE_QUERY:
            raise ValueError("AttractorContextUpdate 必须绑定 query scope")
        if not isinstance(self.logical_timestamp, LogicalTimestamp):
            raise TypeError("AttractorContextUpdate.logical_timestamp 类型错误")
        if self.logical_timestamp.clock.scope != self.scope:
            raise ValueError("AttractorContextUpdate 时钟不属于当前 query")
        _require_instruction(self.reason, label="AttractorContextUpdate.reason")
        object.__setattr__(
            self,
            "changed_dependencies",
            _normalized_dependencies(
                self.changed_dependencies,
                label="AttractorContextUpdate.changed_dependencies",
            ),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 scope、逻辑时间、理由和变化依赖的完整更新键。"""
        result = [
            *_packed(self.scope.stable_key()),
            *_packed(self.logical_timestamp.stable_key()),
            *_packed(self.reason.stable_key()),
            len(self.changed_dependencies),
        ]
        for dependency in self.changed_dependencies:
            result.extend(_packed(dependency.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class AttractorRecomputeDecision:
    """局部策略对一个未执行 activation 给出的新方向快照。"""

    score_adjustment: int
    score_reasons: tuple[AttractorScoreReason, ...]
    dependencies: tuple[AttractorDependency, ...]
    disposition: ObjectIdentity

    def __post_init__(self) -> None:
        """核验新评分、依赖和 disposition 都是完整注入值。"""
        assert_int(
            self.score_adjustment,
            _where="AttractorRecomputeDecision.score_adjustment",
        )
        if type(self.score_adjustment) is not int:
            raise ValueError("recompute score_adjustment 必须是严格整数")
        object.__setattr__(self, "score_reasons", _normalized_reasons(
            self.score_reasons,
            label="AttractorRecomputeDecision.score_reasons",
        ))
        object.__setattr__(self, "dependencies", _normalized_dependencies(
            self.dependencies,
            label="AttractorRecomputeDecision.dependencies",
        ))
        _require_instruction(
            self.disposition,
            label="AttractorRecomputeDecision.disposition",
        )


class AttractorRecomputeStrategy(Protocol):
    """只重算依赖命中的未执行 activation，不读取 reward 或 replay。"""

    def recompute(
            self,
            activation: AttractorActivation,
            update: AttractorContextUpdate,
            ) -> AttractorRecomputeDecision:
        """返回同一 activation 的新评分和状态，不得伪造新候选。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回局部重算协议版本和注入配置键。"""
        ...


@dataclass(frozen=True)
class AttractorConsumptionDecision:
    """真实 consumer 处理一个 agenda 项后提交的不可变决策链接。"""

    activation_key: tuple[int, ...]
    consumer: ObjectIdentity
    disposition: ObjectIdentity
    decision_trace_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 activation、consumer、状态和下游 trace 均可稳定追溯。"""
        _strict_key(
            self.activation_key,
            label="AttractorConsumptionDecision.activation_key",
        )
        _require_instruction(
            self.consumer, label="AttractorConsumptionDecision.consumer")
        _require_instruction(
            self.disposition,
            label="AttractorConsumptionDecision.disposition",
        )
        _strict_key(
            self.decision_trace_key,
            label="AttractorConsumptionDecision.decision_trace_key",
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 activation、consumer、状态和下游 trace 的稳定键。"""
        return (
            *_packed(self.activation_key),
            *_packed(self.consumer.stable_key()),
            *_packed(self.disposition.stable_key()),
            *_packed(self.decision_trace_key),
        )


@dataclass(frozen=True)
class AttractorProcessingTrace:
    """供 M-08 区分“进入 agenda”和“已实际处理”的边界记录。"""

    ordinal: int
    activation: AttractorActivation
    frontier_before: tuple[tuple[int, ...], ...]
    decision: AttractorConsumptionDecision

    def __post_init__(self) -> None:
        """核验消费确实来自处理前 frontier head，且决策引用同一 activation。"""
        assert_int(self.ordinal, _where="AttractorProcessingTrace.ordinal")
        if type(self.ordinal) is not int or self.ordinal <= 0:
            raise ValueError("AttractorProcessingTrace.ordinal 必须为正整数")
        if not isinstance(self.activation, AttractorActivation):
            raise TypeError("AttractorProcessingTrace.activation 类型错误")
        if not isinstance(self.frontier_before, tuple) or not self.frontier_before:
            raise ValueError("AttractorProcessingTrace.frontier_before 不得为空")
        for key in self.frontier_before:
            _strict_key(key, label="AttractorProcessingTrace.frontier_before")
        if len(set(self.frontier_before)) != len(self.frontier_before):
            raise ValueError("AttractorProcessingTrace.frontier_before 不得重复")
        if self.frontier_before[0] != self.activation.identity_key():
            raise ValueError("processing trace 必须记录被选中的 frontier head")
        if not isinstance(self.decision, AttractorConsumptionDecision):
            raise TypeError("AttractorProcessingTrace.decision 类型错误")
        if self.decision.activation_key != self.activation.identity_key():
            raise ValueError("processing trace 决策引用了其他 activation")

    def stable_key(self) -> tuple[int, ...]:
        """返回处理顺序、处理前 frontier、activation 和 consumer 决策键。"""
        result = [
            self.ordinal,
            len(self.frontier_before),
        ]
        for key in self.frontier_before:
            result.extend(_packed(key))
        result.extend((
            *_packed(self.activation.stable_key()),
            *_packed(self.decision.stable_key()),
        ))
        return tuple(result)


@dataclass(frozen=True)
class AttractorRecomputeTrace:
    """一个局部 activation 重算前后的可审计差异。"""

    before: AttractorActivation
    after: AttractorActivation

    def __post_init__(self) -> None:
        """要求重算只改变动态字段，不能替换候选或目标身份。"""
        if not isinstance(self.before, AttractorActivation):
            raise TypeError("AttractorRecomputeTrace.before 类型错误")
        if not isinstance(self.after, AttractorActivation):
            raise TypeError("AttractorRecomputeTrace.after 类型错误")
        if self.before.identity_key() != self.after.identity_key():
            raise ValueError("局部重算不得改变 activation 身份")
        if self.after.revision != self.before.revision + 1:
            raise ValueError("局部重算 revision 必须单步推进")

    def stable_key(self) -> tuple[int, ...]:
        """返回同一 activation 的重算前后快照键。"""
        return (
            *_packed(self.before.stable_key()),
            *_packed(self.after.stable_key()),
        )


@dataclass(frozen=True)
class AttractorUpdateTrace:
    """一次后文更新命中的局部重算及保持不可变的已执行项。"""

    update: AttractorContextUpdate
    recomputed: tuple[AttractorRecomputeTrace, ...]
    immutable_activation_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """核验重算和不可变键稳定有序且互不重复。"""
        if not isinstance(self.update, AttractorContextUpdate):
            raise TypeError("AttractorUpdateTrace.update 类型错误")
        if any(not isinstance(item, AttractorRecomputeTrace)
               for item in self.recomputed):
            raise TypeError("AttractorUpdateTrace.recomputed 类型错误")
        keys = tuple(item.before.identity_key() for item in self.recomputed)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("局部重算 trace 必须稳定有序且不重复")
        for key in self.immutable_activation_keys:
            _strict_key(key, label="immutable_activation_key")
        if self.immutable_activation_keys != tuple(sorted(
                self.immutable_activation_keys)):
            raise ValueError("immutable activation keys 必须稳定有序")
        if len(set(self.immutable_activation_keys)) != len(
                self.immutable_activation_keys):
            raise ValueError("immutable activation keys 不得重复")

    def stable_key(self) -> tuple[int, ...]:
        """返回更新、局部差异和不可变已执行项的稳定键。"""
        result = [*_packed(self.update.stable_key()), len(self.recomputed)]
        for item in self.recomputed:
            result.extend(_packed(item.stable_key()))
        result.append(len(self.immutable_activation_keys))
        for key in self.immutable_activation_keys:
            result.extend(_packed(key))
        return tuple(result)


@dataclass
class AttractorState:
    """每个 query 重建并由 WorkMemory 持有的动态 agenda 状态。"""

    scope: ScopeIdentity
    source: SourceRef
    initial_timestamp: LogicalTimestamp
    protocol: AttractorProtocol
    budget: AttractorBudget
    current_timestamp: LogicalTimestamp = field(init=False)
    mapper_state_key: tuple[int, ...] = field(default=(), init=False)
    considered_activation_count: int = field(default=0, init=False)
    dropped_activation_keys: tuple[tuple[int, ...], ...] = field(
        default=(), init=False)
    _activations: dict[tuple[int, ...], AttractorActivation] = field(
        default_factory=dict, init=False, repr=False)
    _processing_traces: tuple[AttractorProcessingTrace, ...] = field(
        default=(), init=False, repr=False)
    _update_traces: tuple[AttractorUpdateTrace, ...] = field(
        default=(), init=False, repr=False)
    _recomputations_used: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """核验状态只属于一个 query/source/clock，并冻结初始逻辑时间。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("AttractorState.scope 类型错误")
        if self.scope.scope_kind != SCOPE_QUERY:
            raise ValueError("AttractorState 必须绑定 query scope")
        if not isinstance(self.source, SourceRef):
            raise TypeError("AttractorState.source 类型错误")
        if (self.scope.source != self.source
                or self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions):
            raise ValueError("AttractorState scope 与 source 身份不一致")
        if not isinstance(self.initial_timestamp, LogicalTimestamp):
            raise TypeError("AttractorState.initial_timestamp 类型错误")
        if self.initial_timestamp.clock.scope != self.scope:
            raise ValueError("AttractorState 逻辑时钟不属于当前 query")
        if not isinstance(self.protocol, AttractorProtocol):
            raise TypeError("AttractorState.protocol 类型错误")
        if not isinstance(self.budget, AttractorBudget):
            raise TypeError("AttractorState.budget 类型错误")
        self.current_timestamp = self.initial_timestamp

    @property
    def activated(self) -> bool:
        """返回 M-07 resolution 是否已经投影过，空结果也算已投影。"""
        return bool(self.mapper_state_key)

    @property
    def recomputations_used(self) -> int:
        """返回已实际调用局部重算策略的 activation 数量。"""
        return self._recomputations_used

    @property
    def remaining_consumptions(self) -> int:
        """返回调用真实 consumer 前可预检的剩余处理预算。"""
        return self.budget.max_consumptions - len(self._processing_traces)

    def activations(self) -> tuple[AttractorActivation, ...]:
        """按不变身份返回全部进入状态机的 activation 快照。"""
        return tuple(self._activations[key] for key in sorted(self._activations))

    def processing_traces(self) -> tuple[AttractorProcessingTrace, ...]:
        """返回真实 consumer 已处理项，供 M-08 后续判断是否形成 Use。"""
        return self._processing_traces

    def update_traces(self) -> tuple[AttractorUpdateTrace, ...]:
        """返回后文局部重算记录，不包含长期 Memory 改写。"""
        return self._update_traces

    def activate(
            self,
            resolution: MemoryResolution,
            obligations: tuple[ReasoningObligation, ...],
            mapper: AttractorActivationMapper,
            ) -> tuple[AttractorActivation, ...]:
        """将 M-07 候选投影为当前目标 agenda，并按注入方向分截取预算。"""
        if self.activated:
            raise RuntimeError("AttractorState 已完成 activation，不能重复投影")
        if not isinstance(resolution, MemoryResolution):
            raise TypeError("resolution 必须是 MemoryResolution")
        current = resolution.compilation.current
        if (current.scope != self.scope
                or current.source != self.source
                or current.logical_timestamp != self.initial_timestamp):
            raise ValueError("MemoryResolution 不属于当前 AttractorState")
        if not isinstance(obligations, tuple) or not obligations:
            raise ValueError("AttractorState 至少需要一个目标义务")
        if any(not isinstance(item, ReasoningObligation)
               for item in obligations):
            raise TypeError("obligations 含非法项")
        obligation_keys = tuple(item.stable_key() for item in obligations)
        if len(set(obligation_keys)) != len(obligation_keys):
            raise ValueError("AttractorState 目标义务不得重复")
        for obligation in obligations:
            if obligation.scope != self.scope or obligation.source != self.source:
                raise ValueError("目标义务必须属于当前 query 和输入来源")
        if not hasattr(mapper, "project"):
            raise TypeError("mapper 必须实现 project")
        state_key = getattr(mapper, "state_key", None)
        if not callable(state_key):
            raise TypeError("mapper 必须实现 state_key")
        mapper_key = _strict_key(
            state_key(), label="AttractorActivationMapper.state_key")

        proposed: dict[tuple[int, ...], AttractorActivation] = {}
        for candidate_set in resolution.sets:
            request = candidate_set.request
            for candidate in candidate_set.candidates:
                proposals = mapper.project(request, candidate, obligations)
                if not isinstance(proposals, tuple):
                    raise TypeError("Attractor mapper 必须返回 tuple")
                for proposal in proposals:
                    if not isinstance(proposal, AttractorActivationProposal):
                        raise TypeError("Attractor mapper 返回非法 proposal")
                    if proposal.obligation not in obligations:
                        raise ValueError("Attractor mapper 伪造了未声明目标")
                    activation = AttractorActivation(
                        request,
                        candidate,
                        proposal.activation_kind,
                        proposal.obligation,
                        proposal.score_adjustment,
                        proposal.score_reasons,
                        proposal.dependencies,
                        self.protocol.agenda,
                    )
                    key = activation.identity_key()
                    previous = proposed.get(key)
                    if previous is not None and previous != activation:
                        raise ValueError("Attractor activation 稳定身份碰撞")
                    proposed[key] = activation

        ordered = sorted(
            proposed.values(),
            key=lambda item: (-item.score, item.identity_key()),
        )
        selected = ordered[:self.budget.max_agenda_entries]
        dropped = ordered[self.budget.max_agenda_entries:]
        self.mapper_state_key = mapper_key
        self.considered_activation_count = len(ordered)
        self.dropped_activation_keys = tuple(sorted(
            item.identity_key() for item in dropped))
        self._activations = {
            item.identity_key(): item for item in selected
        }
        return self.frontier()

    def frontier(self, *, limit: int | None = None) -> tuple[AttractorActivation, ...]:
        """按当前整数方向分返回可消费 agenda，不包含暂停或终态项。"""
        if limit is not None:
            assert_int(limit, _where="AttractorState.frontier.limit")
            if type(limit) is not int or limit <= 0:
                raise ValueError("frontier limit 必须为正严格整数")
        entries = tuple(sorted(
            (
                item for item in self._activations.values()
                if item.status == self.protocol.agenda
            ),
            key=lambda item: (-item.score, item.identity_key()),
        ))
        return entries if limit is None else entries[:limit]

    def next_activation(self) -> AttractorActivation | None:
        """返回当前 agenda 第一项；无可处理项时返回空。"""
        frontier = self.frontier(limit=1)
        return None if not frontier else frontier[0]

    def commit_consumption(
            self,
            decision: AttractorConsumptionDecision,
            ) -> AttractorProcessingTrace:
        """原子提交 consumer 结果，只允许 agenda 转为 consumed 或 suspended。"""
        if not isinstance(decision, AttractorConsumptionDecision):
            raise TypeError("decision 必须是 AttractorConsumptionDecision")
        if self.remaining_consumptions <= 0:
            raise RuntimeError("AttractorState consumption 预算已耗尽")
        activation = self._activations.get(decision.activation_key)
        if activation is None:
            raise ValueError("consumer decision 引用了未进入 agenda 的 activation")
        if activation.status != self.protocol.agenda:
            raise RuntimeError("consumer 只能处理当前 agenda 项")
        frontier = self.frontier()
        if not frontier or frontier[0].identity_key() != decision.activation_key:
            raise ValueError("consumer decision 必须提交当前 frontier head")
        if decision.disposition not in {
                self.protocol.consumed, self.protocol.suspended}:
            raise ValueError("consumer 只能提交 consumed 或 suspended")
        updated = replace(
            activation,
            status=decision.disposition,
            revision=activation.revision + 1,
        )
        trace = AttractorProcessingTrace(
            len(self._processing_traces) + 1,
            activation,
            tuple(item.identity_key() for item in frontier),
            decision,
        )
        self._activations[activation.identity_key()] = updated
        self._processing_traces = (*self._processing_traces, trace)
        return trace

    def apply_update(
            self,
            update: AttractorContextUpdate,
            strategy: AttractorRecomputeStrategy,
            ) -> AttractorUpdateTrace:
        """只对依赖命中的非终态项做原子重算，已执行项保持不可变。"""
        if not isinstance(update, AttractorContextUpdate):
            raise TypeError("update 必须是 AttractorContextUpdate")
        if update.scope != self.scope:
            raise ValueError("AttractorContextUpdate 属于其他 query")
        if update.logical_timestamp.clock != self.current_timestamp.clock:
            raise ValueError("AttractorContextUpdate 属于其他逻辑时钟")
        if update.logical_timestamp.seq <= self.current_timestamp.seq:
            raise ValueError("AttractorContextUpdate 逻辑时间必须单调推进")
        if not hasattr(strategy, "recompute"):
            raise TypeError("strategy 必须实现 recompute")
        strategy_state_key = getattr(strategy, "state_key", None)
        if not callable(strategy_state_key):
            raise TypeError("strategy 必须实现 state_key")
        _strict_key(
            strategy_state_key(),
            label="AttractorRecomputeStrategy.state_key",
        )

        changed = {
            item.stable_key() for item in update.changed_dependencies}
        affected = []
        immutable = []
        for activation in self.activations():
            dependency_keys = {
                item.stable_key() for item in activation.dependencies}
            if not changed.intersection(dependency_keys):
                continue
            if activation.status in {
                    self.protocol.consumed, self.protocol.superseded}:
                immutable.append(activation.identity_key())
            else:
                affected.append(activation)
        remaining = (
            self.budget.max_recomputations - self._recomputations_used)
        if len(affected) > remaining:
            raise RuntimeError("AttractorState recomputation 预算不足")

        pending: list[AttractorRecomputeTrace] = []
        for activation in affected:
            decision = strategy.recompute(activation, update)
            if not isinstance(decision, AttractorRecomputeDecision):
                raise TypeError("recompute strategy 返回非法 decision")
            if decision.disposition not in {
                    self.protocol.agenda,
                    self.protocol.suspended,
                    self.protocol.superseded,
                    }:
                raise ValueError("recompute disposition 不属于允许的未执行状态")
            updated = replace(
                activation,
                score_adjustment=decision.score_adjustment,
                score_reasons=decision.score_reasons,
                dependencies=decision.dependencies,
                status=decision.disposition,
                revision=activation.revision + 1,
            )
            pending.append(AttractorRecomputeTrace(activation, updated))

        for item in pending:
            self._activations[item.before.identity_key()] = item.after
        self._recomputations_used += len(pending)
        self.current_timestamp = update.logical_timestamp
        trace = AttractorUpdateTrace(
            update,
            tuple(sorted(
                pending, key=lambda item: item.before.identity_key())),
            tuple(sorted(immutable)),
        )
        self._update_traces = (*self._update_traces, trace)
        return trace

    def stable_key(self) -> tuple[int, ...]:
        """返回 query 输入、配置、agenda、处理和局部重算的完整状态键。"""
        result = [
            ATTRACTOR_PROTOCOL_VERSION,
            *_packed(self.scope.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.initial_timestamp.stable_key()),
            *_packed(self.current_timestamp.stable_key()),
            *_packed(self.protocol.stable_key()),
            *self.budget.stable_key(),
            *_packed(self.mapper_state_key),
            self.considered_activation_count,
            self._recomputations_used,
            len(self.dropped_activation_keys),
        ]
        for key in self.dropped_activation_keys:
            result.extend(_packed(key))
        activations = self.activations()
        result.append(len(activations))
        for activation in activations:
            result.extend(_packed(activation.stable_key()))
        result.append(len(self._processing_traces))
        for trace in self._processing_traces:
            result.extend(_packed(trace.stable_key()))
        result.append(len(self._update_traces))
        for trace in self._update_traces:
            result.extend(_packed(trace.stable_key()))
        return tuple(result)


__all__ = [
    "ATTRACTOR_PROTOCOL_VERSION",
    "AttractorActivation",
    "AttractorActivationMapper",
    "AttractorActivationProposal",
    "AttractorBudget",
    "AttractorConsumptionDecision",
    "AttractorContextUpdate",
    "AttractorDependency",
    "AttractorDependencyValue",
    "AttractorProcessingTrace",
    "AttractorProtocol",
    "AttractorRecomputeDecision",
    "AttractorRecomputeStrategy",
    "AttractorRecomputeTrace",
    "AttractorScoreReason",
    "AttractorState",
    "AttractorUpdateTrace",
]
