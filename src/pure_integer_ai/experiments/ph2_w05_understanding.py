"""W-05 Understanding consumer：typed occurrence/Role/Scope 到 Proposition。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    OBJECT_OCCURRENCE,
    OBJECT_ROLE_BINDING,
    ObjectIdentity,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w05_adapter import (
    W05AtomicPropositionCandidate,
)
from pure_integer_ai.experiments.ph2_w05_learning import (
    W05AtomicPropositionLearningRuntime,
)


W05_UNDERSTANDING_UNIQUE = "UNIQUE"
W05_UNDERSTANDING_MULTI = "MULTI"
W05_UNDERSTANDING_UNKNOWN = "UNKNOWN"
W05_UNDERSTANDING_CONFLICT = "CONFLICT"
W05_UNDERSTANDING_CLARIFY = "CLARIFY"
W05_UNDERSTANDING_STATUSES = (
    W05_UNDERSTANDING_UNIQUE,
    W05_UNDERSTANDING_MULTI,
    W05_UNDERSTANDING_UNKNOWN,
    W05_UNDERSTANDING_CONFLICT,
    W05_UNDERSTANDING_CLARIFY,
)
W05_UNDERSTANDING_OUTCOME_SUPPORT = "SUPPORT"
W05_UNDERSTANDING_OUTCOME_REFUTE = "REFUTE"

_NAMESPACE = 50513
_STATUS_CODE = {
    value: ordinal
    for ordinal, value in enumerate(W05_UNDERSTANDING_STATUSES, start=1)
}


class W05UnderstandingError(RuntimeError):
    """W-05 Understanding 请求、状态或 exact Use 不闭合。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _strict_bool(value: bool, *, where: str) -> bool:
    if type(value) is not bool:
        raise W05UnderstandingError(f"{where} 必须是严格 bool")
    return value


def _identities(
        value: tuple[ObjectIdentity, ...],
        *,
        kind: int,
        where: str,
        preserve_order: bool,
        ) -> tuple[ObjectIdentity, ...]:
    if (not isinstance(value, tuple) or not value
            or any(not isinstance(item, ObjectIdentity)
                   or item.object_kind != kind for item in value)):
        raise W05UnderstandingError(f"{where} identity 类型非法")
    if len(set(value)) != len(value):
        raise W05UnderstandingError(f"{where} identity 重复")
    if not preserve_order and value != tuple(sorted(
            value, key=ObjectIdentity.stable_key)):
        raise W05UnderstandingError(f"{where} 必须规范排序")
    return value


@dataclass(frozen=True)
class W05UnderstandingProtocol:
    """四条独立结构桥；后续 ablation 只能关闭其中一条。"""

    occurrence_identity_connected: bool = True
    proposition_consumer_connected: bool = True
    role_bridge_connected: bool = True
    scope_projection_connected: bool = True

    def __post_init__(self) -> None:
        for name in (
                "occurrence_identity_connected",
                "proposition_consumer_connected",
                "role_bridge_connected",
                "scope_projection_connected"):
            _strict_bool(getattr(self, name), where=name)

    def all_connected(self) -> bool:
        return all((
            self.occurrence_identity_connected,
            self.proposition_consumer_connected,
            self.role_bridge_connected,
            self.scope_projection_connected,
        ))


@dataclass(frozen=True)
class W05UnderstandingRequest:
    """不含 expected label 的 surface/occurrence/Role/ContextScope 查询。"""

    request_key: LosslessIntegerKey
    surface: str
    occurrence_order: tuple[ObjectIdentity, ...]
    role_bindings: tuple[ObjectIdentity, ...]
    context: ObjectIdentity
    allow_multiple: bool

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W05UnderstandingError("understanding request_key 类型非法")
        if not isinstance(self.surface, str) or not self.surface:
            raise W05UnderstandingError("understanding surface 非法")
        _identities(
            self.occurrence_order,
            kind=OBJECT_OCCURRENCE,
            where="understanding occurrence_order",
            preserve_order=True,
        )
        _identities(
            self.role_bindings,
            kind=OBJECT_ROLE_BINDING,
            where="understanding role_bindings",
            preserve_order=False,
        )
        if (not isinstance(self.context, ObjectIdentity)
                or self.context.object_kind != OBJECT_CONTEXT_SCOPE):
            raise W05UnderstandingError("understanding context 非 ContextScope")
        _strict_bool(self.allow_multiple, where="allow_multiple")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1,
            *_pack(self.request_key.components),
            len(self.surface),
            *(ord(character) for character in self.surface),
            len(self.occurrence_order),
        ]
        for item in self.occurrence_order:
            values.extend(_pack(item.stable_key()))
        values.append(len(self.role_bindings))
        for item in self.role_bindings:
            values.extend(_pack(item.stable_key()))
        values.extend((
            *_pack(self.context.stable_key()),
            int(self.allow_multiple),
        ))
        return tuple(values)


@dataclass(frozen=True)
class W05PropositionResolution:
    """Understanding 的完整候选集与五态结果。"""

    request: W05UnderstandingRequest
    status: str
    candidates: tuple[W05AtomicPropositionCandidate, ...]
    selected: W05AtomicPropositionCandidate | None
    evidence_keys: tuple[tuple[int, ...], ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.request, W05UnderstandingRequest):
            raise W05UnderstandingError("resolution request 类型非法")
        if self.status not in W05_UNDERSTANDING_STATUSES:
            raise W05UnderstandingError("understanding status 非法")
        if (not isinstance(self.candidates, tuple)
                or any(not isinstance(item, W05AtomicPropositionCandidate)
                       for item in self.candidates)):
            raise W05UnderstandingError("resolution candidates 类型非法")
        normalized = tuple(sorted(
            self.candidates, key=lambda item: item.candidate.stable_key()))
        if self.candidates != normalized or len(set(self.candidates)) != len(
                self.candidates):
            raise W05UnderstandingError("resolution candidates 未规范化")
        if self.status == W05_UNDERSTANDING_UNIQUE:
            if len(self.candidates) != 1 or self.selected != self.candidates[0]:
                raise W05UnderstandingError("UNIQUE 必须采用唯一 Proposition")
        elif self.selected is not None:
            raise W05UnderstandingError("非 UNIQUE resolution 不得私选 Proposition")
        if (self.status in {
                W05_UNDERSTANDING_UNKNOWN,
                W05_UNDERSTANDING_CONFLICT,
                W05_UNDERSTANDING_CLARIFY}
                and self.status != W05_UNDERSTANDING_CONFLICT
                and self.candidates):
            raise W05UnderstandingError("UNKNOWN/CLARIFY 不得泄漏 candidate")
        if (not isinstance(self.evidence_keys, tuple)
                or any(not isinstance(item, tuple) or not item
                       or any(type(value) is not int for value in item)
                       for item in self.evidence_keys)):
            raise W05UnderstandingError("resolution evidence key 非法")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W05UnderstandingError("resolution reason key 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1,
            _STATUS_CODE[self.status],
            *_pack(self.request.stable_key()),
            len(self.candidates),
        ]
        for item in self.candidates:
            values.extend(_pack(item.candidate.stable_key()))
        selected = () if self.selected is None else self.selected.candidate.stable_key()
        values.extend(_pack(selected))
        values.append(len(self.evidence_keys))
        for item in self.evidence_keys:
            values.extend(_pack(item))
        values.extend(_pack(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W05UnderstandingUse:
    """调用方对一个精确 Proposition resolution 的采用。"""

    resolution: W05PropositionResolution
    candidate: W05AtomicPropositionCandidate
    use_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.resolution, W05PropositionResolution):
            raise W05UnderstandingError("understanding Use resolution 非法")
        if self.resolution.status not in {
                W05_UNDERSTANDING_UNIQUE, W05_UNDERSTANDING_MULTI}:
            raise W05UnderstandingError("只有 UNIQUE/MULTI 可形成 Understanding Use")
        if self.candidate not in self.resolution.candidates:
            raise W05UnderstandingError("Understanding Use 未绑定 resolution candidate")
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise W05UnderstandingError("understanding use_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            *_pack(self.resolution.stable_key()),
            *_pack(self.candidate.candidate.stable_key()),
            *_pack(self.use_key.components),
        )


@dataclass(frozen=True)
class W05UnderstandingOutcome:
    """当前 active projection 对 exact Understanding Use 的结果。"""

    use: W05UnderstandingUse
    verdict: str
    current_status: str
    outcome_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.use, W05UnderstandingUse):
            raise W05UnderstandingError("understanding outcome Use 非法")
        if self.verdict not in {
                W05_UNDERSTANDING_OUTCOME_SUPPORT,
                W05_UNDERSTANDING_OUTCOME_REFUTE}:
            raise W05UnderstandingError("understanding outcome verdict 非法")
        if self.current_status not in W05_UNDERSTANDING_STATUSES:
            raise W05UnderstandingError("understanding current status 非法")
        if not isinstance(self.outcome_key, LosslessIntegerKey):
            raise W05UnderstandingError("understanding outcome_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            1 if self.verdict == W05_UNDERSTANDING_OUTCOME_SUPPORT else 2,
            _STATUS_CODE[self.current_status],
            *_pack(self.use.stable_key()),
            *_pack(self.outcome_key.components),
        )


def understanding_request_for_candidate(
        candidate: W05AtomicPropositionCandidate,
        *,
        request_key: LosslessIntegerKey,
        allow_multiple: bool = False,
        ) -> W05UnderstandingRequest:
    """从 typed query candidate 建立不含 Proposition expected 的 Understanding 请求。"""
    if not isinstance(candidate, W05AtomicPropositionCandidate):
        raise TypeError("candidate 类型非法")
    return W05UnderstandingRequest(
        request_key,
        candidate.surface,
        candidate.occurrence_order,
        tuple(sorted(
            candidate.role_binding_identities(),
            key=ObjectIdentity.stable_key,
        )),
        candidate.proposition_definition.context,
        allow_multiple,
    )


class W05UnderstandingRuntime:
    """只消费 W-05 typed structure 与当前 Evidence，不用 cue/token 顺序猜 Role。"""

    def __init__(
            self,
            learning: W05AtomicPropositionLearningRuntime,
            *,
            protocol: W05UnderstandingProtocol = W05UnderstandingProtocol(),
            ) -> None:
        if not isinstance(learning, W05AtomicPropositionLearningRuntime):
            raise TypeError("learning 必须是 W05AtomicPropositionLearningRuntime")
        if not isinstance(protocol, W05UnderstandingProtocol):
            raise TypeError("understanding protocol 类型非法")
        self.learning = learning
        self.protocol = protocol
        self._request_keys: set[LosslessIntegerKey] = set()
        self._resolutions: list[W05PropositionResolution] = []
        self._uses: list[W05UnderstandingUse] = []
        self._outcomes: list[W05UnderstandingOutcome] = []

    @property
    def resolutions(self) -> tuple[W05PropositionResolution, ...]:
        return tuple(self._resolutions)

    @property
    def uses(self) -> tuple[W05UnderstandingUse, ...]:
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W05UnderstandingOutcome, ...]:
        return tuple(self._outcomes)

    @staticmethod
    def _role_bindings(
            candidate: W05AtomicPropositionCandidate,
            ) -> tuple[ObjectIdentity, ...]:
        return tuple(sorted(
            candidate.role_binding_identities(),
            key=ObjectIdentity.stable_key,
        ))

    def _matches(
            self,
            request: W05UnderstandingRequest,
            ) -> tuple[W05AtomicPropositionCandidate, ...]:
        if not self.protocol.all_connected():
            return ()
        return tuple(sorted((
            item for item in self.learning.registered_candidates()
            if item.surface == request.surface
            and item.occurrence_order == request.occurrence_order
            and self._role_bindings(item) == request.role_bindings
            and item.proposition_definition.context == request.context
        ), key=lambda item: item.candidate.stable_key()))

    def _candidate_stances(
            self,
            candidate: W05AtomicPropositionCandidate,
            ) -> tuple[int, ...]:
        hypothesis = self.learning.hypothesis_for(candidate.candidate)
        return tuple(
            item.verification.stance
            for item in self.learning.learning.engine.recognition_history(
                hypothesis)
        )

    def _evidence_keys(
            self,
            candidates: tuple[W05AtomicPropositionCandidate, ...],
            ) -> tuple[tuple[int, ...], ...]:
        values = []
        for candidate in candidates:
            hypothesis = self.learning.hypothesis_for(candidate.candidate)
            values.extend(
                item.evidence.stable_key()
                for item in self.learning.learning.engine.recognition_history(
                    hypothesis)
            )
        return tuple(sorted(set(values)))

    def _resolve_value(
            self,
            request: W05UnderstandingRequest,
            ) -> W05PropositionResolution:
        matches = self._matches(request)
        conflicts = tuple(
            item for item in matches
            if {EVIDENCE_SUPPORT, EVIDENCE_REFUTE}.issubset(
                set(self._candidate_stances(item)))
        )
        active_set = set(self.learning.active_candidates())
        active = tuple(item for item in matches if item in active_set)
        if conflicts:
            status = W05_UNDERSTANDING_CONFLICT
            candidates = conflicts
            selected = None
        elif len(active) == 1:
            status = W05_UNDERSTANDING_UNIQUE
            candidates = active
            selected = active[0]
        elif len(active) > 1 and request.allow_multiple:
            status = W05_UNDERSTANDING_MULTI
            candidates = active
            selected = None
        elif len(active) > 1:
            status = W05_UNDERSTANDING_CLARIFY
            candidates = ()
            selected = None
        else:
            status = W05_UNDERSTANDING_UNKNOWN
            candidates = ()
            selected = None
        reason = LosslessIntegerKey((
            _NAMESPACE,
            100 + _STATUS_CODE[status],
            int(self.protocol.occurrence_identity_connected),
            int(self.protocol.proposition_consumer_connected),
            int(self.protocol.role_bridge_connected),
            int(self.protocol.scope_projection_connected),
        ))
        evidence_source = conflicts or active or matches
        return W05PropositionResolution(
            request,
            status,
            candidates,
            selected,
            self._evidence_keys(evidence_source),
            reason,
        )

    def resolve(
            self,
            request: W05UnderstandingRequest,
            ) -> W05PropositionResolution:
        """返回 unique/multi/unknown/conflict/clarify，不私选多候选。"""
        if not isinstance(request, W05UnderstandingRequest):
            raise TypeError("understanding request 类型非法")
        if request.request_key in self._request_keys:
            raise W05UnderstandingError("重复 understanding request key")
        resolution = self._resolve_value(request)
        self._request_keys.add(request.request_key)
        self._resolutions.append(resolution)
        return resolution

    def adopt(
            self,
            resolution: W05PropositionResolution,
            candidate: W05AtomicPropositionCandidate,
            ) -> W05UnderstandingUse:
        """显式采用一个 exact Proposition，不从 stable order 自动选首项。"""
        if resolution not in self._resolutions:
            raise W05UnderstandingError("resolution 不属于当前 Understanding runtime")
        use = W05UnderstandingUse(
            resolution,
            candidate,
            LosslessIntegerKey((
                _NAMESPACE,
                200,
                len(self._uses) + 1,
                *_pack(resolution.stable_key()),
                *_pack(candidate.candidate.stable_key()),
            )),
        )
        self._uses.append(use)
        return use

    def verify_use(self, use: W05UnderstandingUse) -> W05UnderstandingOutcome:
        """按当前 lifecycle 重验 exact Use，历史 outcome append-only 保留。"""
        if use not in self._uses:
            raise W05UnderstandingError("Understanding Use 不属于当前 runtime")
        current = self._resolve_value(use.resolution.request)
        supported = use.candidate in current.candidates
        verdict = (
            W05_UNDERSTANDING_OUTCOME_SUPPORT
            if supported else W05_UNDERSTANDING_OUTCOME_REFUTE
        )
        outcome = W05UnderstandingOutcome(
            use,
            verdict,
            current.status,
            LosslessIntegerKey((
                _NAMESPACE,
                300,
                len(self._outcomes) + 1,
                *_pack(use.use_key.components),
            )),
        )
        self._outcomes.append(outcome)
        return outcome


def build_w05_understanding_runtime(
        learning: W05AtomicPropositionLearningRuntime,
        *,
        protocol: W05UnderstandingProtocol = W05UnderstandingProtocol(),
        ) -> W05UnderstandingRuntime:
    """建立 W05-03 Understanding consumer，不启动正式训练。"""
    return W05UnderstandingRuntime(learning, protocol=protocol)


__all__ = [
    "W05PropositionResolution",
    "W05UnderstandingError",
    "W05UnderstandingOutcome",
    "W05UnderstandingProtocol",
    "W05UnderstandingRequest",
    "W05UnderstandingRuntime",
    "W05UnderstandingUse",
    "W05_UNDERSTANDING_CLARIFY",
    "W05_UNDERSTANDING_CONFLICT",
    "W05_UNDERSTANDING_MULTI",
    "W05_UNDERSTANDING_OUTCOME_REFUTE",
    "W05_UNDERSTANDING_OUTCOME_SUPPORT",
    "W05_UNDERSTANDING_UNIQUE",
    "W05_UNDERSTANDING_UNKNOWN",
    "build_w05_understanding_runtime",
    "understanding_request_for_candidate",
]
