"""LC-16 九载体方向专属 consumer。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.experiments.ph2_carrier_directional_contract import (
    GENERATION_NEEDS_NARROWING,
    GENERATION_READY,
    GENERATION_REFUSED,
    REASONING_REFUTED,
    REASONING_SUPPORTED,
    REASONING_UNRESOLVED,
    UNDERSTANDING_REJECTED,
    UNDERSTANDING_UNCERTAIN,
    UNDERSTANDING_UNDERSTOOD,
    DirectionalProjectionContext,
    GenerationRequest,
    GenerationResult,
    GenerationUse,
    ReasoningRequest,
    ReasoningResult,
    ReasoningUse,
    UnderstandingRequest,
    UnderstandingResult,
    UnderstandingUse,
)
from pure_integer_ai.experiments.ph2_carrier_projection_runtime import (
    default_carrier_projection_protocol,
)


_NAMESPACE = 16618000
_ASSESS_SUPPORTED = 1
_ASSESS_REFUTED = 2
_ASSESS_UNKNOWN = 3


class CarrierDirectionalRuntimeError(RuntimeError):
    """方向 ledger、当前 projection 或结果重验失败。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


@dataclass(frozen=True)
class CarrierDirectionalProtocol:
    """方向消费只读依赖的候选生命周期身份。"""

    active_state: ObjectIdentity
    superseded_state: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.active_state, ObjectIdentity):
            raise TypeError("directional active_state 类型非法")
        if not isinstance(self.superseded_state, ObjectIdentity):
            raise TypeError("directional superseded_state 类型非法")
        if self.active_state == self.superseded_state:
            raise CarrierDirectionalRuntimeError(
                "directional lifecycle state 不得重合")


def default_carrier_directional_protocol() -> CarrierDirectionalProtocol:
    lifecycle = default_carrier_projection_protocol().lifecycle
    return CarrierDirectionalProtocol(
        lifecycle.active_state,
        lifecycle.superseded_state,
    )


def _assessment(
        context: DirectionalProjectionContext,
        protocol: CarrierDirectionalProtocol,
        ) -> int:
    projection = context.projection
    if projection.lifecycle_state != protocol.active_state:
        return _ASSESS_REFUTED
    latest = max(
        projection.evidence,
        key=lambda item: (item.timestamp_seq, item.evidence_id),
    )
    if latest.stance == EVIDENCE_SUPPORT:
        return _ASSESS_SUPPORTED
    if latest.stance == EVIDENCE_REFUTE:
        return _ASSESS_REFUTED
    return _ASSESS_UNKNOWN


def _result_key(kind: int, request_key: tuple[int, ...], context_key: tuple[int, ...]):
    return _NAMESPACE, kind, *_pack(request_key), *_pack(context_key)


def _use_key(kind: int, result_key: tuple[int, ...]) -> tuple[int, ...]:
    return _NAMESPACE, kind, *_pack(result_key)


def _understanding_result(
        request: UnderstandingRequest,
        protocol: CarrierDirectionalProtocol,
        ) -> UnderstandingResult:
    assessment = _assessment(request.context, protocol)
    status = {
        _ASSESS_SUPPORTED: UNDERSTANDING_UNDERSTOOD,
        _ASSESS_REFUTED: UNDERSTANDING_REJECTED,
        _ASSESS_UNKNOWN: UNDERSTANDING_UNCERTAIN,
    }[assessment]
    context = request.context
    return UnderstandingResult(
        request,
        status,
        context.projection.semantic_object,
        context.projection.projection_kind,
        context.carrier_input.envelope.carrier_family,
        context.subject_identities,
        context.carrier_input.feature_identities,
        context.evidence_keys,
        _result_key(101, request.request_key, context.stable_key()),
    )


def _reasoning_result(
        request: ReasoningRequest,
        protocol: CarrierDirectionalProtocol,
        ) -> ReasoningResult:
    assessment = _assessment(request.context, protocol)
    if request.claim != request.context.projection.semantic_object:
        conclusion = REASONING_UNRESOLVED
    else:
        conclusion = {
            _ASSESS_SUPPORTED: REASONING_SUPPORTED,
            _ASSESS_REFUTED: REASONING_REFUTED,
            _ASSESS_UNKNOWN: REASONING_UNRESOLVED,
        }[assessment]
    context = request.context
    premises = tuple(sorted(
        set(context.subject_identities)
        | {
            context.projection.projection_kind,
            context.projection.semantic_object,
            *context.carrier_input.feature_identities,
        },
        key=ObjectIdentity.stable_key,
    ))
    return ReasoningResult(
        request,
        conclusion,
        premises,
        context.evidence_keys,
        _result_key(201, request.request_key, context.stable_key()),
    )


def _generation_result(
        request: GenerationRequest,
        protocol: CarrierDirectionalProtocol,
        ) -> GenerationResult:
    context = request.context
    raw_units = context.carrier_input.envelope.raw_units
    assessment = _assessment(context, protocol)
    if (assessment != _ASSESS_SUPPORTED
            or request.goal != context.projection.semantic_object
            or not raw_units):
        status = GENERATION_REFUSED
        surface_units: tuple[int, ...] = ()
    elif len(raw_units) > request.max_surface_units:
        status = GENERATION_NEEDS_NARROWING
        surface_units = ()
    else:
        status = GENERATION_READY
        surface_units = raw_units
    return GenerationResult(
        request,
        status,
        context.carrier_input.envelope.raw_unit_kind,
        surface_units,
        context.subject_identities,
        context.evidence_keys,
        _result_key(301, request.request_key, context.stable_key()),
    )


class UnderstandingConsumer:
    """把 projection 解析为结构化语义框架并记录 exact Use。"""

    def __init__(self, protocol: CarrierDirectionalProtocol | None = None) -> None:
        self.protocol = (
            default_carrier_directional_protocol()
            if protocol is None else protocol)
        if not isinstance(self.protocol, CarrierDirectionalProtocol):
            raise TypeError("understanding protocol 类型非法")
        self._results: list[UnderstandingResult] = []
        self._uses: list[UnderstandingUse] = []
        self._request_keys: set[tuple[int, ...]] = set()
        self._used_result_keys: set[tuple[int, ...]] = set()

    def consume(self, request: UnderstandingRequest) -> UnderstandingResult:
        if not isinstance(request, UnderstandingRequest):
            raise TypeError("understanding request 类型非法")
        if request.request_key in self._request_keys:
            raise CarrierDirectionalRuntimeError(
                "重复 understanding request key")
        result = _understanding_result(request, self.protocol)
        self._request_keys.add(request.request_key)
        self._results.append(result)
        return result

    def use(self, result: UnderstandingResult, action: str) -> UnderstandingUse:
        if result not in self._results:
            raise CarrierDirectionalRuntimeError(
                "understanding result 不属于当前 consumer")
        if result.result_key in self._used_result_keys:
            raise CarrierDirectionalRuntimeError(
                "同一 understanding result 不得重复 Use")
        use = UnderstandingUse(
            result,
            action,
            _use_key(102, result.result_key),
        )
        self._used_result_keys.add(result.result_key)
        self._uses.append(use)
        return use


class ReasoningConsumer:
    """基于当前 Evidence/lifecycle 对明确 claim 形成结论。"""

    def __init__(self, protocol: CarrierDirectionalProtocol | None = None) -> None:
        self.protocol = (
            default_carrier_directional_protocol()
            if protocol is None else protocol)
        if not isinstance(self.protocol, CarrierDirectionalProtocol):
            raise TypeError("reasoning protocol 类型非法")
        self._results: list[ReasoningResult] = []
        self._uses: list[ReasoningUse] = []
        self._request_keys: set[tuple[int, ...]] = set()
        self._used_result_keys: set[tuple[int, ...]] = set()

    def consume(self, request: ReasoningRequest) -> ReasoningResult:
        if not isinstance(request, ReasoningRequest):
            raise TypeError("reasoning request 类型非法")
        if request.request_key in self._request_keys:
            raise CarrierDirectionalRuntimeError("重复 reasoning request key")
        result = _reasoning_result(request, self.protocol)
        self._request_keys.add(request.request_key)
        self._results.append(result)
        return result

    def use(self, result: ReasoningResult, action: str) -> ReasoningUse:
        if result not in self._results:
            raise CarrierDirectionalRuntimeError(
                "reasoning result 不属于当前 consumer")
        if result.result_key in self._used_result_keys:
            raise CarrierDirectionalRuntimeError(
                "同一 reasoning result 不得重复 Use")
        use = ReasoningUse(result, action, _use_key(202, result.result_key))
        self._used_result_keys.add(result.result_key)
        self._uses.append(use)
        return use


class GenerationConsumer:
    """由语义 goal 和当前授权生成实际整数 surface。"""

    def __init__(self, protocol: CarrierDirectionalProtocol | None = None) -> None:
        self.protocol = (
            default_carrier_directional_protocol()
            if protocol is None else protocol)
        if not isinstance(self.protocol, CarrierDirectionalProtocol):
            raise TypeError("generation protocol 类型非法")
        self._results: list[GenerationResult] = []
        self._uses: list[GenerationUse] = []
        self._request_keys: set[tuple[int, ...]] = set()
        self._used_result_keys: set[tuple[int, ...]] = set()

    def consume(self, request: GenerationRequest) -> GenerationResult:
        if not isinstance(request, GenerationRequest):
            raise TypeError("generation request 类型非法")
        if request.request_key in self._request_keys:
            raise CarrierDirectionalRuntimeError("重复 generation request key")
        result = _generation_result(request, self.protocol)
        self._request_keys.add(request.request_key)
        self._results.append(result)
        return result

    def use(self, result: GenerationResult, action: str) -> GenerationUse:
        if result not in self._results:
            raise CarrierDirectionalRuntimeError(
                "generation result 不属于当前 consumer")
        if result.result_key in self._used_result_keys:
            raise CarrierDirectionalRuntimeError(
                "同一 generation result 不得重复 Use")
        use = GenerationUse(result, action, _use_key(302, result.result_key))
        self._used_result_keys.add(result.result_key)
        self._uses.append(use)
        return use


__all__ = [
    "CarrierDirectionalProtocol", "CarrierDirectionalRuntimeError",
    "GenerationConsumer", "ReasoningConsumer", "UnderstandingConsumer",
    "default_carrier_directional_protocol",
]
