"""把 typed Event/Proposition 时间核验接入 R-09 独立维度。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_CONFLICTED,
    EVENT_TIME_CONSISTENT,
    EVENT_TIME_EMPTY,
    EventTimeVerificationResult,
    EventTimeVerifier,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_CONFLICTED,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
    VerificationEvaluation,
    VerifierRegistration,
)


@dataclass(frozen=True)
class EventTimeVerificationProtocol:
    """声明 event-time 独立 dimension 和 verifier 身份。"""

    dimension: ProtocolKey
    verifier: ProtocolKey

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, ProtocolKey):
            raise TypeError("event-time dimension 必须是 ProtocolKey")
        if not isinstance(self.verifier, ProtocolKey):
            raise TypeError("event-time verifier 必须是 ProtocolKey")


@dataclass(frozen=True)
class EventTimeVerificationRequest:
    """一次精确 scope 和 relation 集的 typed 时间核验请求。"""

    scope: ScopeIdentity
    relations: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("event-time request scope 必须是 ScopeIdentity")
        if not isinstance(self.relations, tuple) or not self.relations:
            raise ValueError("event-time request relations 必须是非空 tuple")
        if any(not isinstance(item, ObjectIdentity) for item in self.relations):
            raise TypeError("event-time request relation 类型非法")
        if len(set(self.relations)) != len(self.relations):
            raise ValueError("event-time request relations 不得重复")


class EventTimeVerificationAdapter:
    """生成不提交 CAUSES 或其他维度 effect 的 R-09 注册项。"""

    def __init__(
            self,
            verifier: EventTimeVerifier,
            protocol: EventTimeVerificationProtocol,
            ) -> None:
        if not isinstance(verifier, EventTimeVerifier):
            raise TypeError("verifier 必须是 EventTimeVerifier")
        if not isinstance(protocol, EventTimeVerificationProtocol):
            raise TypeError("protocol 必须是 EventTimeVerificationProtocol")
        self.verifier = verifier
        self.protocol = protocol

    def registration(self) -> VerifierRegistration:
        """返回只读评估型注册；该 adapter 永不提交跨维 effect。"""
        return VerifierRegistration(
            dimension=self.protocol.dimension,
            verifier=self.protocol.verifier,
            applies=self._applies,
            evaluate=self._evaluate,
        )

    def _applies(self, request: object) -> bool:
        """只有请求合法且精确 scope 内存在 typed 事实时才适用。"""
        if not isinstance(request, EventTimeVerificationRequest):
            return False
        facts = self.verifier.facts.read(
            request.relations,
            scope=request.scope,
        )
        return bool(facts.facts)

    def _evaluate(self, request: object) -> VerificationEvaluation:
        """把 event-time 四态映射到 R-09 verdict 并保留完整 artifact。"""
        if not isinstance(request, EventTimeVerificationRequest):
            raise TypeError("event-time verifier 收到错误 request")
        result = self.verifier.verify(
            request.relations,
            scope=request.scope,
        )
        verdict = self._verdict(result)
        claims = tuple(
            fact.statement.assertion.stable_key()
            for fact in result.fact_set.facts
        )
        detail = (
            result.status,
            len(result.before_edges),
            len(result.same_groups),
            len(result.unknown_relations),
            len(result.conflict_assertion_hashes),
        )
        return VerificationEvaluation(
            verdict=verdict,
            claim_keys=claims,
            detail=detail,
            source=request.scope.source,
            scope=request.scope,
            artifact=result,
        )

    @staticmethod
    def _verdict(result: EventTimeVerificationResult) -> int:
        """映射一致、冲突和未知，不把 empty 伪装成 support。"""
        if result.status == EVENT_TIME_CONSISTENT:
            return VERDICT_SUPPORT
        if result.status == EVENT_TIME_CONFLICTED:
            return VERDICT_CONFLICTED
        if result.status == EVENT_TIME_EMPTY:
            return VERDICT_UNKNOWN
        return VERDICT_UNKNOWN


__all__ = [
    "EventTimeVerificationAdapter",
    "EventTimeVerificationProtocol",
    "EventTimeVerificationRequest",
]
