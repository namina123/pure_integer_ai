"""C-02 Capability 的只读召回、整数评分和确定性 Top-K route。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.capability_memory import (
    RecoveredCapability,
    recover_verified_capability,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_CAPABILITY,
    MEMORY_EVENT_CAPABILITY_ATTEMPT_OUTCOME,
    MEMORY_EVENT_USE,
    MEMORY_OBJECT_CAPABILITY,
    CapabilityAttemptOutcomePayload,
    UsePayload,
)
from pure_integer_ai.cognition.shared.memory_event_log import MemoryEventLog
from pure_integer_ai.cognition.shared.memory_query import MemoryActivationRequest
from pure_integer_ai.cognition.shared.memory_resolver import (
    RESOLUTION_ORIGIN_MEMORY,
    ActivationScore,
    ResolvedCandidate,
    ResolvedCandidateSet,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_CAPABILITY_RESOLVER_VERSION = 1


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长整数键增加长度边界。"""
    return len(value), *value


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 route 和评分协议键为非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须只含严格整数")
    return value


@dataclass(frozen=True)
class CapabilityUsageSummary:
    """只属于一个 Capability 的成功 Use 与失败尝试统计。"""

    use_count: int
    failure_count: int
    last_used_seq: int
    last_failed_seq: int

    def __post_init__(self) -> None:
        """核验计数和逻辑序均为非负严格整数。"""
        values = (
            self.use_count,
            self.failure_count,
            self.last_used_seq,
            self.last_failed_seq,
        )
        assert_int(*values, _where="CapabilityUsageSummary")
        if any(type(item) is not int or item < 0 for item in values):
            raise ValueError("Capability usage summary 必须使用非负严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回成功、失败计数及各自最后逻辑序。"""
        return (
            self.use_count,
            self.failure_count,
            self.last_used_seq,
            self.last_failed_seq,
        )


@dataclass(frozen=True)
class CapabilityActivationAssessment:
    """注入评分器给出的竞争键和纯整数评分。"""

    competition_key: tuple[int, ...]
    score: ActivationScore

    def __post_init__(self) -> None:
        """核验竞争键和评分均可进入确定性 resolver trace。"""
        _strict_key(
            self.competition_key,
            label="CapabilityActivationAssessment.competition_key",
        )
        if not isinstance(self.score, ActivationScore):
            raise TypeError("Capability activation score 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回竞争键和完整评分理由。"""
        return (
            *_packed(self.competition_key),
            *_packed(self.score.stable_key()),
        )


class CapabilityScoreProvider(Protocol):
    """按当前 query、完整能力和自身使用统计提供整数评分。"""

    def assess(
            self,
            request: MemoryActivationRequest,
            capability: RecoveredCapability,
            usage: CapabilityUsageSummary,
            ) -> CapabilityActivationAssessment:
        """返回竞争键和评分，不得写 Memory 或读取其他候选结果。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回非空版本化协议状态。"""
        ...


class CapabilityResolverRoute:
    """从已激活 Capability 声明形成确定性 Top-K typed 结果。"""

    memory_object_kind = MEMORY_OBJECT_CAPABILITY

    def __init__(
            self,
            event_log: MemoryEventLog,
            score_provider: CapabilityScoreProvider,
            ) -> None:
        """绑定只读 event log 和注入式评分器。"""
        if not isinstance(event_log, MemoryEventLog):
            raise TypeError("event_log 必须是 MemoryEventLog")
        if not callable(getattr(score_provider, "assess", None)):
            raise TypeError("Capability score provider 缺少 assess")
        state_key = getattr(score_provider, "state_key", None)
        if not callable(state_key):
            raise TypeError("Capability score provider 缺少 state_key")
        _strict_key(state_key(), label="Capability score provider state_key")
        self.event_log = event_log
        self.score_provider = score_provider

    def resolve(
            self,
            request: MemoryActivationRequest,
            ) -> ResolvedCandidateSet:
        """按 ACL、能力类型和注入评分返回当前 request 的稳定 Top-K。"""
        if not isinstance(request, MemoryActivationRequest):
            raise TypeError("Capability route request 类型错误")
        if request.memory_object_kind != MEMORY_OBJECT_CAPABILITY:
            raise ValueError("Capability route 只处理 Capability request")
        if request.memory_space != self.event_log.memory_space_identity:
            raise ValueError("Capability request 属于其他 Memory 空间")
        usage = self._usage_by_capability(request)
        candidates = []
        declarations = self.event_log.query(
            access=request.access,
            event_kind=MEMORY_EVENT_CAPABILITY,
            object_kind=MEMORY_OBJECT_CAPABILITY,
        )
        for declaration in declarations:
            recovered = recover_verified_capability(
                self.event_log,
                declaration.event.object_ref,
                access=request.access,
            )
            if (recovered.payload.capability_kind.stable_key()
                    != request.hypothesis_kind):
                continue
            sources = tuple(sorted({
                recovered.contract.candidate_source,
                *recovered.contract.candidate_example_sources,
                recovered.contract.held_out_source,
            }, key=lambda item: item.stable_key()))
            if any(not request.access.can_read(source.owner)
                   for source in sources):
                raise PermissionError("Capability provenance 来源超出当前 ACL")
            summary = usage.get(
                declaration.event.object_ref.stable_key(),
                CapabilityUsageSummary(0, 0, 0, 0),
            )
            assessment = self.score_provider.assess(
                request, recovered, summary)
            if not isinstance(assessment, CapabilityActivationAssessment):
                raise TypeError("Capability score provider 返回类型错误")
            candidates.append(ResolvedCandidate(
                RESOLUTION_ORIGIN_MEMORY,
                declaration.event.object_ref.stable_key(),
                assessment.competition_key,
                request.scope,
                declaration.event.scope,
                None,
                declaration.event.object_ref,
                None,
                None,
                sources,
                (),
                assessment.score.value,
                assessment.score.reasons,
                None,
                recovered.payload,
            ))
        ordered = tuple(sorted(
            candidates,
            key=lambda item: (
                -item.score,
                item.competition_key,
                item.candidate_key,
            ),
        ))
        return ResolvedCandidateSet(
            request,
            ordered[:request.budget],
            len(ordered),
        )

    def _usage_by_capability(
            self,
            request: MemoryActivationRequest,
            ) -> dict[tuple[int, ...], CapabilityUsageSummary]:
        """一次索引读取聚合每个 Capability 自己的 Use 和失败逻辑序。"""
        counts: dict[tuple[int, ...], list[int]] = {}
        uses = self.event_log.query(
            access=request.access,
            event_kind=MEMORY_EVENT_USE,
        )
        for materialized in uses:
            payload = materialized.event.payload
            if (not isinstance(payload, UsePayload)
                    or payload.memory_ref.object_kind
                    != MEMORY_OBJECT_CAPABILITY):
                continue
            key = payload.memory_ref.stable_key()
            values = counts.setdefault(key, [0, 0, 0, 0])
            values[0] += 1
            values[2] = max(values[2], payload.used_at.seq)
        failures = self.event_log.query(
            access=request.access,
            event_kind=MEMORY_EVENT_CAPABILITY_ATTEMPT_OUTCOME,
        )
        for materialized in failures:
            payload = materialized.event.payload
            if not isinstance(payload, CapabilityAttemptOutcomePayload):
                raise RuntimeError("Capability attempt outcome payload 漂移")
            key = payload.target_ref.stable_key()
            values = counts.setdefault(key, [0, 0, 0, 0])
            values[1] += 1
            values[3] = max(values[3], payload.observed_at.seq)
        return {
            key: CapabilityUsageSummary(*values)
            for key, values in counts.items()
        }

    def clone_for_context(self, ctx) -> "CapabilityResolverRoute":
        """为 V-06 重绑同 identity 的独立 event log，并克隆可选评分器。"""
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
            raise ValueError("评测上下文缺少唯一同 identity Capability event log")
        clone = getattr(self.score_provider, "clone_for_context", None)
        provider = self.score_provider if clone is None else clone(ctx)
        result = CapabilityResolverRoute(matches[0], provider)
        if result.state_key() != self.state_key():
            raise ValueError("Capability resolver clone 改变了协议状态")
        return result

    def state_key(self) -> tuple[int, ...]:
        """返回 route、Memory 空间和评分协议的版本化状态键。"""
        return (
            _CAPABILITY_RESOLVER_VERSION,
            MEMORY_OBJECT_CAPABILITY,
            *_packed(self.event_log.memory_space_identity.stable_key()),
            *_packed(self.score_provider.state_key()),
        )


__all__ = [
    "CapabilityActivationAssessment",
    "CapabilityResolverRoute",
    "CapabilityScoreProvider",
    "CapabilityUsageSummary",
]
