"""M-09 记忆激活衰减协议及 M-07 评分适配。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MemoryHypothesisAggregateIndex,
)
from pure_integer_ai.cognition.shared.memory_query import MemoryActivationRequest
from pure_integer_ai.cognition.shared.memory_resolver import (
    ActivationScore,
    ActivationScoreReason,
)
from pure_integer_ai.cognition.shared.scope_identity import LogicalTimestamp
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.memory_aggregate import (
    MemoryHypothesisAggregateRecord,
)


_DECAY_PROTOCOL_VERSION = 1


def _key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """校验衰减协议中的开放整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _component_key(component: object, *, label: str) -> tuple[int, ...]:
    """读取注入组件的稳定状态键，拒绝未版本化实现。"""
    method = getattr(component, "state_key", None)
    if not callable(method):
        raise TypeError(f"{label} 缺少 state_key")
    return _key(method(), label=f"{label}.state_key")


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


@dataclass(frozen=True)
class MemoryActivationSnapshot:
    """一个 Hypothesis 在统一 Memory 时间线上的激活输入。"""

    hypothesis: HypothesisKey
    aggregate: MemoryHypothesisAggregateRecord
    as_of: LogicalTimestamp

    def __post_init__(self) -> None:
        """核验候选、聚合和时间线水位没有倒退。"""
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("activation snapshot 缺少 HypothesisKey")
        if not isinstance(
                self.aggregate, MemoryHypothesisAggregateRecord):
            raise TypeError("activation snapshot 缺少 aggregate")
        if not isinstance(self.as_of, LogicalTimestamp):
            raise TypeError("activation snapshot.as_of 类型错误")
        if self.as_of.seq < self.activity_seq:
            raise ValueError("activation as_of 早于候选最近活动")

    @property
    def activity_seq(self) -> int:
        """返回创建、观察和真实使用中的最近统一时间线序。"""
        return max(
            self.aggregate.created_seq,
            self.aggregate.last_observed_seq,
            self.aggregate.last_used_seq,
        )

    @property
    def age(self) -> int:
        """返回统一时间线上的非负活动年龄。"""
        return self.as_of.seq - self.activity_seq

    def stable_key(self) -> tuple[int, ...]:
        """返回足以重算衰减结果的候选状态键。"""
        aggregate = self.aggregate
        return (
            _DECAY_PROTOCOL_VERSION,
            *_packed(self.hypothesis.stable_key()),
            aggregate.hypothesis_hash,
            aggregate.created_seq,
            aggregate.last_observed_seq,
            aggregate.last_used_seq,
            aggregate.retention_state,
            aggregate.lifecycle_state,
            aggregate.evidence_state,
            *_packed(self.as_of.stable_key()),
        )


@dataclass(frozen=True)
class ActivationDecayAssessment:
    """注入衰减策略返回的纯整数激活值和审计理由。"""

    value: int
    activity_seq: int
    as_of_seq: int
    reason_key: tuple[int, ...]
    policy_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验激活值、时间线边界和策略身份。"""
        assert_int(
            self.value,
            self.activity_seq,
            self.as_of_seq,
            _where="ActivationDecayAssessment",
        )
        if type(self.value) is not int or self.value < 0:
            raise ValueError("activation value 必须是非负严格整数")
        if (type(self.activity_seq) is not int
                or type(self.as_of_seq) is not int
                or self.activity_seq <= 0
                or self.as_of_seq < self.activity_seq):
            raise ValueError("activation 时间线边界非法")
        _key(self.reason_key, label="activation reason_key")
        _key(self.policy_key, label="activation policy_key")


class ActivationDecayPolicy(Protocol):
    """按完整激活快照计算纯整数值的注入协议。"""

    def state_key(self) -> tuple[int, ...]:
        """返回策略版本、参数和曲线的稳定身份。"""
        ...

    def assess(
            self,
            snapshot: MemoryActivationSnapshot,
            ) -> ActivationDecayAssessment:
        """计算激活值，不写 Evidence、事件或物理位置。"""
        ...


@dataclass(frozen=True, order=True)
class RetentionDecayCurve:
    """一个开放 retention state 的线性纯整数衰减参数。"""

    retention_state: int
    initial_value: int
    decay_per_step: int
    floor_value: int

    def __post_init__(self) -> None:
        """要求每条曲线真实衰减且下界不高于初值。"""
        assert_int(
            self.retention_state,
            self.initial_value,
            self.decay_per_step,
            self.floor_value,
            _where="RetentionDecayCurve",
        )
        if type(self.retention_state) is not int or self.retention_state <= 0:
            raise ValueError("retention_state 必须是正严格整数")
        if any(type(value) is not int or value < 0 for value in (
                self.initial_value, self.floor_value)):
            raise ValueError("activation curve 值必须是非负严格整数")
        if type(self.decay_per_step) is not int or self.decay_per_step <= 0:
            raise ValueError("每种 retention curve 都必须具有正整数衰减")
        if self.floor_value > self.initial_value:
            raise ValueError("activation floor 不得高于 initial value")

    def stable_key(self) -> tuple[int, ...]:
        """返回 retention state 和全部整数曲线参数。"""
        return (
            self.retention_state,
            self.initial_value,
            self.decay_per_step,
            self.floor_value,
        )


@dataclass(frozen=True)
class LinearTimelineDecayPolicy:
    """按注入 retention 曲线在统一时间线上执行线性衰减。"""

    policy_key: tuple[int, ...]
    curves: tuple[RetentionDecayCurve, ...]
    reason_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验策略身份、理由和 retention curve 全集没有重复。"""
        _key(self.policy_key, label="linear decay policy_key")
        _key(self.reason_key, label="linear decay reason_key")
        if (not isinstance(self.curves, tuple)
                or not self.curves
                or any(not isinstance(item, RetentionDecayCurve)
                       for item in self.curves)):
            raise TypeError("linear decay curves 类型错误")
        states = tuple(item.retention_state for item in self.curves)
        if len(set(states)) != len(states):
            raise ValueError("linear decay retention state 不得重复")
        object.__setattr__(
            self, "curves",
            tuple(sorted(self.curves, key=lambda item: item.retention_state)),
        )

    def state_key(self) -> tuple[int, ...]:
        """返回策略身份、理由和全部曲线参数。"""
        result = [
            _DECAY_PROTOCOL_VERSION,
            *_packed(self.policy_key),
            *_packed(self.reason_key),
            len(self.curves),
        ]
        for curve in self.curves:
            result.extend(curve.stable_key())
        return tuple(result)

    def assess(
            self,
            snapshot: MemoryActivationSnapshot,
            ) -> ActivationDecayAssessment:
        """按最近活动年龄计算下界受限的线性激活值。"""
        if not isinstance(snapshot, MemoryActivationSnapshot):
            raise TypeError("linear decay 需要 MemoryActivationSnapshot")
        curve = next(
            (item for item in self.curves
             if item.retention_state == snapshot.aggregate.retention_state),
            None,
        )
        if curve is None:
            raise ValueError("当前 retention state 没有注入 decay curve")
        value = max(
            curve.floor_value,
            curve.initial_value - snapshot.age * curve.decay_per_step,
        )
        return ActivationDecayAssessment(
            value,
            snapshot.activity_seq,
            snapshot.as_of.seq,
            self.reason_key,
            self.state_key(),
        )


class MemoryDecayScoreProvider:
    """在 M-07 原评分后追加 M-09 时间线激活分量。"""

    def __init__(
            self,
            aggregates: MemoryHypothesisAggregateIndex,
            base_provider: object,
            decay_policy: ActivationDecayPolicy,
            ) -> None:
        """绑定同一 M-04 索引、原评分器和版本化衰减策略。"""
        if not isinstance(aggregates, MemoryHypothesisAggregateIndex):
            raise TypeError("decay score provider 缺少 aggregate")
        if not callable(getattr(base_provider, "score", None)):
            raise TypeError("base score provider 缺少 score")
        if not callable(getattr(decay_policy, "assess", None)):
            raise TypeError("decay policy 缺少 assess")
        _component_key(base_provider, label="base score provider")
        self._policy_key = _component_key(
            decay_policy, label="activation decay policy")
        self.aggregates = aggregates
        self.base_provider = base_provider
        self.decay_policy = decay_policy

    def state_key(self) -> tuple[int, ...]:
        """返回空间、原评分器和衰减策略的稳定组合身份。"""
        return (
            _DECAY_PROTOCOL_VERSION,
            *_packed(
                self.aggregates.event_log.memory_space_identity.stable_key()),
            *_packed(_component_key(
                self.base_provider, label="base score provider")),
            *_packed(self._policy_key),
        )

    def score(
            self,
            request: MemoryActivationRequest,
            hypothesis: HypothesisKey,
            aggregate: MemoryHypothesisAggregateRecord,
            sources: tuple[SourceRef, ...],
            ) -> ActivationScore:
        """组合原始相关性和可审计衰减值，不改变证据或生命周期。"""
        base = self.base_provider.score(
            request, hypothesis, aggregate, sources)
        if not isinstance(base, ActivationScore):
            raise TypeError("base score provider 返回了错误评分")
        as_of = self.aggregates.event_log.timeline_watermark()
        if as_of is None:
            raise RuntimeError("存在 aggregate 但 Memory timeline 为空")
        snapshot = MemoryActivationSnapshot(hypothesis, aggregate, as_of)
        assessment = self.decay_policy.assess(snapshot)
        if not isinstance(assessment, ActivationDecayAssessment):
            raise TypeError("decay policy 返回了错误 assessment")
        if (assessment.policy_key != self._policy_key
                or assessment.activity_seq != snapshot.activity_seq
                or assessment.as_of_seq != snapshot.as_of.seq):
            raise ValueError("decay assessment 身份或时间线漂移")
        reason_key = (
            _DECAY_PROTOCOL_VERSION,
            *_packed(assessment.policy_key),
            *_packed(assessment.reason_key),
            assessment.activity_seq,
            assessment.as_of_seq,
        )
        return ActivationScore(
            base.value + assessment.value,
            (*base.reasons, ActivationScoreReason(
                reason_key, assessment.value)),
        )

    def clone_for_context(self, ctx) -> "MemoryDecayScoreProvider":
        """为 V-06 克隆重绑同 identity aggregate 和可选策略组件。"""
        matches = tuple(
            item for item in (
                ctx.memory_read_aggregates,
                ctx.memory_interact_aggregates,
            )
            if (isinstance(item, MemoryHypothesisAggregateIndex)
                and item.event_log.memory_space_identity
                == self.aggregates.event_log.memory_space_identity)
        )
        if len(matches) != 1:
            raise ValueError("评测上下文缺少唯一同 identity aggregate")

        def cloned(component: object) -> object:
            """调用组件可选 clone_for_context，否则复用只读实例。"""
            method = getattr(component, "clone_for_context", None)
            if method is None:
                return component
            if not callable(method):
                raise TypeError("衰减组件 clone_for_context 不可调用")
            return method(ctx)

        result = MemoryDecayScoreProvider(
            matches[0],
            cloned(self.base_provider),
            cloned(self.decay_policy),
        )
        if result.state_key() != self.state_key():
            raise ValueError("M-09 decay clone 改变了协议状态")
        return result


__all__ = [
    "ActivationDecayAssessment",
    "ActivationDecayPolicy",
    "LinearTimelineDecayPolicy",
    "MemoryActivationSnapshot",
    "MemoryDecayScoreProvider",
    "RetentionDecayCurve",
]
