"""W-05 Reasoning consumer：只授权完整 Proposition/Role/Scope/Evidence。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE_BINDING,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w05_adapter import (
    W05AtomicPropositionCandidate,
)
from pure_integer_ai.experiments.ph2_w05_learning import (
    W05AtomicPropositionLearningRuntime,
)


W05_REASONING_AUTHORIZED = "AUTHORIZED"
W05_REASONING_REJECTED = "REJECTED"
W05_REASONING_UNKNOWN = "UNKNOWN"
W05_REASONING_CONFLICT = "CONFLICT"
W05_REASONING_SUPERSEDED = "SUPERSEDED"
W05_REASONING_STATUSES = (
    W05_REASONING_AUTHORIZED,
    W05_REASONING_REJECTED,
    W05_REASONING_UNKNOWN,
    W05_REASONING_CONFLICT,
    W05_REASONING_SUPERSEDED,
)
W05_REASONING_OUTCOME_SUPPORT = "SUPPORT"
W05_REASONING_OUTCOME_REFUTE = "REFUTE"

_NAMESPACE = 50514
_STATUS_CODE = {
    value: ordinal for ordinal, value in enumerate(W05_REASONING_STATUSES, start=1)
}


class W05ReasoningError(RuntimeError):
    """W-05 Reasoning 请求、Evidence 或 exact Use 不闭合。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _strict_bool(value: bool, *, where: str) -> bool:
    if type(value) is not bool:
        raise W05ReasoningError(f"{where} 必须是严格 bool")
    return value


def _identity_tuple(
        value: tuple[ObjectIdentity, ...],
        *,
        kind: int,
        preserve_order: bool,
        where: str,
        ) -> tuple[ObjectIdentity, ...]:
    if (not isinstance(value, tuple) or not value
            or any(not isinstance(item, ObjectIdentity)
                   or item.object_kind != kind for item in value)):
        raise W05ReasoningError(f"{where} identity 类型非法")
    if len(set(value)) != len(value):
        raise W05ReasoningError(f"{where} identity 重复")
    if not preserve_order and value != tuple(sorted(
            value, key=ObjectIdentity.stable_key)):
        raise W05ReasoningError(f"{where} 必须规范排序")
    return value


@dataclass(frozen=True)
class W05ReasoningProtocol:
    """Reasoning 侧四条结构桥与独立 Evidence consumer。"""

    occurrence_identity_connected: bool = True
    proposition_consumer_connected: bool = True
    role_bridge_connected: bool = True
    scope_projection_connected: bool = True
    evidence_consumer_connected: bool = True

    def __post_init__(self) -> None:
        for name in (
                "occurrence_identity_connected",
                "proposition_consumer_connected",
                "role_bridge_connected",
                "scope_projection_connected",
                "evidence_consumer_connected"):
            _strict_bool(getattr(self, name), where=name)


@dataclass(frozen=True)
class W05ReasoningRequest:
    """对完整 Proposition 结构的授权请求，不接受 predicate/token 快捷坐标。"""

    request_key: LosslessIntegerKey
    proposition: ObjectIdentity
    occurrence_order: tuple[ObjectIdentity, ...]
    role_bindings: tuple[ObjectIdentity, ...]
    context: ObjectIdentity
    source: SourceRef

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W05ReasoningError("reasoning request_key 类型非法")
        if (not isinstance(self.proposition, ObjectIdentity)
                or self.proposition.object_kind != OBJECT_PROPOSITION):
            raise W05ReasoningError("reasoning proposition identity 非法")
        _identity_tuple(
            self.occurrence_order,
            kind=OBJECT_OCCURRENCE,
            preserve_order=True,
            where="reasoning occurrence_order",
        )
        _identity_tuple(
            self.role_bindings,
            kind=OBJECT_ROLE_BINDING,
            preserve_order=False,
            where="reasoning role_bindings",
        )
        if (not isinstance(self.context, ObjectIdentity)
                or self.context.object_kind != OBJECT_CONTEXT_SCOPE):
            raise W05ReasoningError("reasoning context 非 ContextScope")
        if not isinstance(self.source, SourceRef):
            raise W05ReasoningError("reasoning source 类型非法")
        if semantic_source(self.proposition) != self.source:
            raise W05ReasoningError("reasoning Proposition 与 source 漂移")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1,
            *_pack(self.request_key.components),
            *_pack(self.proposition.stable_key()),
            len(self.occurrence_order),
        ]
        for item in self.occurrence_order:
            values.extend(_pack(item.stable_key()))
        values.append(len(self.role_bindings))
        for item in self.role_bindings:
            values.extend(_pack(item.stable_key()))
        values.extend((
            *_pack(self.context.stable_key()),
            *_pack(self.source.stable_key()),
        ))
        return tuple(values)


@dataclass(frozen=True)
class W05ReasoningUse:
    """Reasoning 对一个 exact Proposition 的 Evidence 授权记录。"""

    request: W05ReasoningRequest
    status: str
    candidate: W05AtomicPropositionCandidate | None
    evidence_keys: tuple[tuple[int, ...], ...]
    use_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.request, W05ReasoningRequest):
            raise W05ReasoningError("reasoning Use request 类型非法")
        if self.status not in W05_REASONING_STATUSES:
            raise W05ReasoningError("reasoning status 非法")
        if self.status == W05_REASONING_AUTHORIZED:
            if self.candidate is None or not self.evidence_keys:
                raise W05ReasoningError("AUTHORIZED 必须绑定 candidate/Evidence")
        elif self.status == W05_REASONING_UNKNOWN and self.candidate is not None:
            raise W05ReasoningError("UNKNOWN 不得采用 Proposition candidate")
        if (not isinstance(self.evidence_keys, tuple)
                or any(not isinstance(item, tuple) or not item
                       or any(type(value) is not int for value in item)
                       for item in self.evidence_keys)):
            raise W05ReasoningError("reasoning evidence key 非法")
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise W05ReasoningError("reasoning use_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        candidate_key = (
            () if self.candidate is None else self.candidate.candidate.stable_key())
        values = [
            1,
            _STATUS_CODE[self.status],
            *_pack(self.request.stable_key()),
            *_pack(candidate_key),
            len(self.evidence_keys),
        ]
        for item in self.evidence_keys:
            values.extend(_pack(item))
        values.extend(_pack(self.use_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W05ReasoningOutcome:
    """当前 Evidence 对 exact Reasoning Use 的复核结果。"""

    use: W05ReasoningUse
    verdict: str
    current_status: str
    outcome_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.use, W05ReasoningUse):
            raise W05ReasoningError("reasoning outcome Use 非法")
        if self.verdict not in {
                W05_REASONING_OUTCOME_SUPPORT,
                W05_REASONING_OUTCOME_REFUTE}:
            raise W05ReasoningError("reasoning outcome verdict 非法")
        if self.current_status not in W05_REASONING_STATUSES:
            raise W05ReasoningError("reasoning current status 非法")
        if not isinstance(self.outcome_key, LosslessIntegerKey):
            raise W05ReasoningError("reasoning outcome_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            1 if self.verdict == W05_REASONING_OUTCOME_SUPPORT else 2,
            _STATUS_CODE[self.current_status],
            *_pack(self.use.stable_key()),
            *_pack(self.outcome_key.components),
        )


def reasoning_request_for_candidate(
        candidate: W05AtomicPropositionCandidate,
        *,
        request_key: LosslessIntegerKey,
        ) -> W05ReasoningRequest:
    """从 typed candidate 建立完整 Proposition/Role/Scope 请求。"""
    if not isinstance(candidate, W05AtomicPropositionCandidate):
        raise TypeError("candidate 类型非法")
    return W05ReasoningRequest(
        request_key,
        candidate.candidate,
        candidate.occurrence_order,
        tuple(sorted(
            candidate.role_binding_identities(),
            key=ObjectIdentity.stable_key,
        )),
        candidate.proposition_definition.context,
        candidate.source_ref,
    )


class W05ReasoningRuntime:
    """不把 predicate kind、token 顺序或 dependency label 当事实。"""

    def __init__(
            self,
            learning: W05AtomicPropositionLearningRuntime,
            *,
            protocol: W05ReasoningProtocol = W05ReasoningProtocol(),
            ) -> None:
        if not isinstance(learning, W05AtomicPropositionLearningRuntime):
            raise TypeError("learning 必须是 W05AtomicPropositionLearningRuntime")
        if not isinstance(protocol, W05ReasoningProtocol):
            raise TypeError("reasoning protocol 类型非法")
        self.learning = learning
        self.protocol = protocol
        self._candidate_by_proposition = {
            item.candidate: item for item in learning.registered_candidates()
        }
        if len(self._candidate_by_proposition) != len(
                learning.registered_candidates()):
            raise W05ReasoningError("Reasoning inventory Proposition identity 重复")
        self._request_keys: set[LosslessIntegerKey] = set()
        self._uses: list[W05ReasoningUse] = []
        self._outcomes: list[W05ReasoningOutcome] = []

    @property
    def uses(self) -> tuple[W05ReasoningUse, ...]:
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W05ReasoningOutcome, ...]:
        return tuple(self._outcomes)

    @staticmethod
    def _role_bindings(
            candidate: W05AtomicPropositionCandidate,
            ) -> tuple[ObjectIdentity, ...]:
        return tuple(sorted(
            candidate.role_binding_identities(),
            key=ObjectIdentity.stable_key,
        ))

    def _structural_match(
            self,
            request: W05ReasoningRequest,
            candidate: W05AtomicPropositionCandidate,
            ) -> bool:
        return (
            self.protocol.occurrence_identity_connected
            and self.protocol.proposition_consumer_connected
            and self.protocol.role_bridge_connected
            and self.protocol.scope_projection_connected
            and candidate.occurrence_order == request.occurrence_order
            and self._role_bindings(candidate) == request.role_bindings
            and candidate.proposition_definition.context == request.context
            and candidate.source_ref == request.source
        )

    def _state(
            self,
            request: W05ReasoningRequest,
            ) -> tuple[str, W05AtomicPropositionCandidate | None,
                       tuple[tuple[int, ...], ...]]:
        if not self.protocol.proposition_consumer_connected:
            return W05_REASONING_UNKNOWN, None, ()
        candidate = self._candidate_by_proposition.get(request.proposition)
        if candidate is None:
            return W05_REASONING_UNKNOWN, None, ()
        if not self._structural_match(request, candidate):
            return W05_REASONING_REJECTED, candidate, ()
        if not self.protocol.evidence_consumer_connected:
            return W05_REASONING_UNKNOWN, None, ()
        hypothesis = self.learning.hypothesis_for(candidate.candidate)
        records = self.learning.learning.engine.recognition_history(hypothesis)
        evidence = tuple(item.evidence.stable_key() for item in records)
        stances = {item.verification.stance for item in records}
        snapshot = self.learning.learning.engine.ledger.snapshot(hypothesis)
        if snapshot.lifecycle == LIFECYCLE_SUPERSEDED:
            status = W05_REASONING_SUPERSEDED
        elif {EVIDENCE_SUPPORT, EVIDENCE_REFUTE}.issubset(stances):
            status = W05_REASONING_CONFLICT
        elif self.learning.learning.engine.active(hypothesis) is not None:
            status = W05_REASONING_AUTHORIZED
        elif EVIDENCE_REFUTE in stances:
            status = W05_REASONING_REJECTED
        elif EVIDENCE_UNKNOWN in stances:
            status = W05_REASONING_UNKNOWN
        else:
            status = W05_REASONING_UNKNOWN
        return status, candidate, tuple(sorted(evidence))

    def authorize(self, request: W05ReasoningRequest) -> W05ReasoningUse:
        """消费 Proposition/Role/Scope/Evidence 并形成 exact Reasoning Use。"""
        if not isinstance(request, W05ReasoningRequest):
            raise TypeError("reasoning request 类型非法")
        if request.request_key in self._request_keys:
            raise W05ReasoningError("重复 reasoning request key")
        status, candidate, evidence = self._state(request)
        if status == W05_REASONING_UNKNOWN:
            candidate = None
        use = W05ReasoningUse(
            request,
            status,
            candidate,
            evidence,
            LosslessIntegerKey((
                _NAMESPACE,
                100,
                len(self._uses) + 1,
                _STATUS_CODE[status],
                *_pack(request.stable_key()),
            )),
        )
        self._request_keys.add(request.request_key)
        self._uses.append(use)
        return use

    def verify_use(self, use: W05ReasoningUse) -> W05ReasoningOutcome:
        """按当前 Evidence 重验 exact Use，不覆盖既有 Use。"""
        if use not in self._uses:
            raise W05ReasoningError("Reasoning Use 不属于当前 runtime")
        status, candidate, evidence = self._state(use.request)
        supported = (
            use.status == W05_REASONING_AUTHORIZED
            and status == W05_REASONING_AUTHORIZED
            and candidate == use.candidate
            and evidence == use.evidence_keys
        )
        outcome = W05ReasoningOutcome(
            use,
            W05_REASONING_OUTCOME_SUPPORT if supported
            else W05_REASONING_OUTCOME_REFUTE,
            status,
            LosslessIntegerKey((
                _NAMESPACE,
                200,
                len(self._outcomes) + 1,
                *_pack(use.use_key.components),
            )),
        )
        self._outcomes.append(outcome)
        return outcome


def build_w05_reasoning_runtime(
        learning: W05AtomicPropositionLearningRuntime,
        *,
        protocol: W05ReasoningProtocol = W05ReasoningProtocol(),
        ) -> W05ReasoningRuntime:
    """建立 W05-03 Reasoning consumer，不启动正式训练。"""
    return W05ReasoningRuntime(learning, protocol=protocol)


__all__ = [
    "W05ReasoningError",
    "W05ReasoningOutcome",
    "W05ReasoningProtocol",
    "W05ReasoningRequest",
    "W05ReasoningRuntime",
    "W05ReasoningUse",
    "W05_REASONING_AUTHORIZED",
    "W05_REASONING_CONFLICT",
    "W05_REASONING_OUTCOME_REFUTE",
    "W05_REASONING_OUTCOME_SUPPORT",
    "W05_REASONING_REJECTED",
    "W05_REASONING_SUPERSEDED",
    "W05_REASONING_UNKNOWN",
    "build_w05_reasoning_runtime",
    "reasoning_request_for_candidate",
]
