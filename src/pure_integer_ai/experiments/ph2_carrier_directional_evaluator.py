"""LC-16 三方向独立 evaluator，不复用 consumer 结果构造路径。"""
from __future__ import annotations

from dataclasses import replace

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.experiments.ph2_carrier_directional_contract import (
    GENERATION_ADOPT,
    GENERATION_DEFER,
    GENERATION_NEEDS_NARROWING,
    GENERATION_READY,
    GENERATION_REFUSED,
    GENERATION_REJECT,
    OUTCOME_REFUTE,
    OUTCOME_SUPPORT,
    REASONING_ACCEPT,
    REASONING_DEFER,
    REASONING_REFUTED,
    REASONING_REJECT,
    REASONING_SUPPORTED,
    REASONING_UNRESOLVED,
    UNDERSTANDING_ADOPT,
    UNDERSTANDING_DEFER,
    UNDERSTANDING_REJECT,
    UNDERSTANDING_REJECTED,
    UNDERSTANDING_UNCERTAIN,
    UNDERSTANDING_UNDERSTOOD,
    DirectionalProjectionContext,
    GenerationOutcome,
    GenerationResult,
    GenerationUse,
    ReasoningOutcome,
    ReasoningResult,
    ReasoningUse,
    UnderstandingOutcome,
    UnderstandingResult,
    UnderstandingUse,
)
from pure_integer_ai.experiments.ph2_carrier_directional_runtime import (
    CarrierDirectionalProtocol,
    default_carrier_directional_protocol,
)


_NAMESPACE = 16618000
_ASSESS_SUPPORTED = 1
_ASSESS_REFUTED = 2
_ASSESS_UNKNOWN = 3


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _protocol(
        value: CarrierDirectionalProtocol | None,
        ) -> CarrierDirectionalProtocol:
    result = default_carrier_directional_protocol() if value is None else value
    if not isinstance(result, CarrierDirectionalProtocol):
        raise TypeError("directional evaluator protocol 类型非法")
    return result


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


def _same_route(
        original: DirectionalProjectionContext,
        current: DirectionalProjectionContext,
        ) -> bool:
    return (
        original.carrier_input.carrier_key
        == current.carrier_input.carrier_key
        and original.carrier_input.source == current.carrier_input.source
        and original.carrier_input.scope == current.carrier_input.scope
        and original.carrier_input.envelope.identity
        == current.carrier_input.envelope.identity
        and original.carrier_input.input_key == current.carrier_input.input_key
        and original.projection.hypothesis.candidate_key
        == current.projection.hypothesis.candidate_key
        and original.projection.hypothesis.competition_key
        == current.projection.hypothesis.competition_key
    )


def _result_key(kind: int, request_key: tuple[int, ...], context_key: tuple[int, ...]):
    return _NAMESPACE, kind, *_pack(request_key), *_pack(context_key)


def _outcome_key(
        kind: int, use_key: tuple[int, ...], context_key: tuple[int, ...],
        ) -> tuple[int, ...]:
    return _NAMESPACE, kind, *_pack(use_key), *_pack(context_key)


def _expected_understanding(
        use: UnderstandingUse,
        current: DirectionalProjectionContext,
        protocol: CarrierDirectionalProtocol,
        ) -> UnderstandingResult:
    request = replace(use.result.request, context=current)
    status = {
        _ASSESS_SUPPORTED: UNDERSTANDING_UNDERSTOOD,
        _ASSESS_REFUTED: UNDERSTANDING_REJECTED,
        _ASSESS_UNKNOWN: UNDERSTANDING_UNCERTAIN,
    }[_assessment(current, protocol)]
    return UnderstandingResult(
        request,
        status,
        current.projection.semantic_object,
        current.projection.projection_kind,
        current.carrier_input.envelope.carrier_family,
        current.subject_identities,
        current.carrier_input.feature_identities,
        current.evidence_keys,
        _result_key(101, request.request_key, current.stable_key()),
    )


def _expected_reasoning(
        use: ReasoningUse,
        current: DirectionalProjectionContext,
        protocol: CarrierDirectionalProtocol,
        ) -> ReasoningResult:
    request = replace(use.result.request, context=current)
    assessment = _assessment(current, protocol)
    if request.claim != current.projection.semantic_object:
        conclusion = REASONING_UNRESOLVED
    else:
        conclusion = {
            _ASSESS_SUPPORTED: REASONING_SUPPORTED,
            _ASSESS_REFUTED: REASONING_REFUTED,
            _ASSESS_UNKNOWN: REASONING_UNRESOLVED,
        }[assessment]
    premises = tuple(sorted(
        set(current.subject_identities)
        | {
            current.projection.projection_kind,
            current.projection.semantic_object,
            *current.carrier_input.feature_identities,
        },
        key=ObjectIdentity.stable_key,
    ))
    return ReasoningResult(
        request,
        conclusion,
        premises,
        current.evidence_keys,
        _result_key(201, request.request_key, current.stable_key()),
    )


def _expected_generation(
        use: GenerationUse,
        current: DirectionalProjectionContext,
        protocol: CarrierDirectionalProtocol,
        ) -> GenerationResult:
    request = replace(use.result.request, context=current)
    raw_units = current.carrier_input.envelope.raw_units
    assessment = _assessment(current, protocol)
    if (assessment != _ASSESS_SUPPORTED
            or request.goal != current.projection.semantic_object
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
        current.carrier_input.envelope.raw_unit_kind,
        surface_units,
        current.subject_identities,
        current.evidence_keys,
        _result_key(301, request.request_key, current.stable_key()),
    )


class UnderstandingEvaluator:
    """独立重验结构化语义框架与 exact understanding Use。"""

    def __init__(self, protocol: CarrierDirectionalProtocol | None = None) -> None:
        self.protocol = _protocol(protocol)

    def evaluate(
            self,
            use: UnderstandingUse,
            current: DirectionalProjectionContext,
            ) -> UnderstandingOutcome:
        if not isinstance(use, UnderstandingUse):
            raise TypeError("understanding evaluator Use 类型非法")
        expected = _expected_understanding(use, current, self.protocol)
        expected_action = {
            UNDERSTANDING_UNDERSTOOD: UNDERSTANDING_ADOPT,
            UNDERSTANDING_REJECTED: UNDERSTANDING_REJECT,
            UNDERSTANDING_UNCERTAIN: UNDERSTANDING_DEFER,
        }[expected.status]
        exact = (
            _same_route(use.result.request.context, current)
            and use.result == expected
            and use.action == expected_action
        )
        return UnderstandingOutcome(
            use,
            OUTCOME_SUPPORT if exact else OUTCOME_REFUTE,
            current,
            _outcome_key(103, use.use_key, current.stable_key()),
        )


class ReasoningEvaluator:
    """独立重验 premise、Evidence、结论与 exact reasoning Use。"""

    def __init__(self, protocol: CarrierDirectionalProtocol | None = None) -> None:
        self.protocol = _protocol(protocol)

    def evaluate(
            self,
            use: ReasoningUse,
            current: DirectionalProjectionContext,
            ) -> ReasoningOutcome:
        if not isinstance(use, ReasoningUse):
            raise TypeError("reasoning evaluator Use 类型非法")
        expected = _expected_reasoning(use, current, self.protocol)
        expected_action = {
            REASONING_SUPPORTED: REASONING_ACCEPT,
            REASONING_REFUTED: REASONING_REJECT,
            REASONING_UNRESOLVED: REASONING_DEFER,
        }[expected.conclusion]
        exact = (
            _same_route(use.result.request.context, current)
            and use.result == expected
            and use.action == expected_action
        )
        return ReasoningOutcome(
            use,
            OUTCOME_SUPPORT if exact else OUTCOME_REFUTE,
            current,
            _outcome_key(203, use.use_key, current.stable_key()),
        )


class GenerationEvaluator:
    """独立重验 goal、surface、授权证据与 exact generation Use。"""

    def __init__(self, protocol: CarrierDirectionalProtocol | None = None) -> None:
        self.protocol = _protocol(protocol)

    def evaluate(
            self,
            use: GenerationUse,
            current: DirectionalProjectionContext,
            ) -> GenerationOutcome:
        if not isinstance(use, GenerationUse):
            raise TypeError("generation evaluator Use 类型非法")
        expected = _expected_generation(use, current, self.protocol)
        expected_action = {
            GENERATION_READY: GENERATION_ADOPT,
            GENERATION_REFUSED: GENERATION_REJECT,
            GENERATION_NEEDS_NARROWING: GENERATION_DEFER,
        }[expected.status]
        exact = (
            _same_route(use.result.request.context, current)
            and use.result == expected
            and use.action == expected_action
        )
        return GenerationOutcome(
            use,
            OUTCOME_SUPPORT if exact else OUTCOME_REFUTE,
            current,
            _outcome_key(303, use.use_key, current.stable_key()),
        )


__all__ = [
    "GenerationEvaluator", "ReasoningEvaluator", "UnderstandingEvaluator",
]
