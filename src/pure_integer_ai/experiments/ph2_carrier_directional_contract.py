"""LC-16 九载体方向专属 request/result/Use/outcome 合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.artifact_envelope import (
    PROJECTION_GENERATION,
    PROJECTION_REASONING,
    PROJECTION_UNDERSTANDING,
    ArtifactSemanticProjection,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.experiments.ph2_carrier_projection_mapper import (
    CarrierProjectionInput,
)


UNDERSTANDING_UNDERSTOOD = "UNDERSTOOD"
UNDERSTANDING_REJECTED = "REJECTED"
UNDERSTANDING_UNCERTAIN = "UNCERTAIN"
UNDERSTANDING_STATUSES = (
    UNDERSTANDING_UNDERSTOOD,
    UNDERSTANDING_REJECTED,
    UNDERSTANDING_UNCERTAIN,
)
UNDERSTANDING_ADOPT = "ADOPT"
UNDERSTANDING_REJECT = "REJECT"
UNDERSTANDING_DEFER = "DEFER"
UNDERSTANDING_ACTIONS = (
    UNDERSTANDING_ADOPT,
    UNDERSTANDING_REJECT,
    UNDERSTANDING_DEFER,
)

REASONING_SUPPORTED = "SUPPORTED"
REASONING_REFUTED = "REFUTED"
REASONING_UNRESOLVED = "UNRESOLVED"
REASONING_CONCLUSIONS = (
    REASONING_SUPPORTED,
    REASONING_REFUTED,
    REASONING_UNRESOLVED,
)
REASONING_ACCEPT = "ACCEPT"
REASONING_REJECT = "REJECT"
REASONING_DEFER = "DEFER"
REASONING_ACTIONS = (
    REASONING_ACCEPT,
    REASONING_REJECT,
    REASONING_DEFER,
)

GENERATION_READY = "READY"
GENERATION_REFUSED = "REFUSED"
GENERATION_NEEDS_NARROWING = "NEEDS_NARROWING"
GENERATION_STATUSES = (
    GENERATION_READY,
    GENERATION_REFUSED,
    GENERATION_NEEDS_NARROWING,
)
GENERATION_ADOPT = "ADOPT"
GENERATION_REJECT = "REJECT"
GENERATION_DEFER = "DEFER"
GENERATION_ACTIONS = (
    GENERATION_ADOPT,
    GENERATION_REJECT,
    GENERATION_DEFER,
)

OUTCOME_SUPPORT = "SUPPORT"
OUTCOME_REFUTE = "REFUTE"
OUTCOME_NEUTRAL = "NEUTRAL"
OUTCOME_VERDICTS = (
    OUTCOME_SUPPORT,
    OUTCOME_REFUTE,
    OUTCOME_NEUTRAL,
)

_UNDERSTANDING_STATUS_CODE = {
    value: index for index, value in enumerate(UNDERSTANDING_STATUSES, start=1)
}
_UNDERSTANDING_ACTION_CODE = {
    value: index for index, value in enumerate(UNDERSTANDING_ACTIONS, start=1)
}
_REASONING_CONCLUSION_CODE = {
    value: index for index, value in enumerate(REASONING_CONCLUSIONS, start=1)
}
_REASONING_ACTION_CODE = {
    value: index for index, value in enumerate(REASONING_ACTIONS, start=1)
}
_GENERATION_STATUS_CODE = {
    value: index for index, value in enumerate(GENERATION_STATUSES, start=1)
}
_GENERATION_ACTION_CODE = {
    value: index for index, value in enumerate(GENERATION_ACTIONS, start=1)
}
_OUTCOME_CODE = {
    value: index for index, value in enumerate(OUTCOME_VERDICTS, start=1)
}


class CarrierDirectionalContractError(RuntimeError):
    """方向、投影上下文或 exact Use/outcome 未精确闭合。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise CarrierDirectionalContractError(
            f"{where} 必须是非负严格整数 tuple")
    return value


def _objects(
        value: tuple[ObjectIdentity, ...], *, where: str,
        allow_empty: bool = False,
        ) -> tuple[ObjectIdentity, ...]:
    if (not isinstance(value, tuple) or (not allow_empty and not value)
            or any(not isinstance(item, ObjectIdentity) for item in value)
            or value != tuple(sorted(
                set(value), key=ObjectIdentity.stable_key))):
        raise CarrierDirectionalContractError(
            f"{where} 必须是排序去重的 ObjectIdentity tuple")
    return value


def _integer_tuple(
        value: tuple[int, ...], *, where: str, allow_empty: bool = False,
        ) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise CarrierDirectionalContractError(
            f"{where} 必须是非负严格整数 tuple")
    return value


def _nested_keys(
        value: tuple[tuple[int, ...], ...], *, where: str,
        ) -> tuple[tuple[int, ...], ...]:
    if (not isinstance(value, tuple) or not value
            or any(not isinstance(item, tuple) or not item
                   or any(type(part) is not int for part in item)
                   for item in value)
            or value != tuple(sorted(set(value)))):
        raise CarrierDirectionalContractError(
            f"{where} 必须是排序去重的整数键 tuple")
    return value


@dataclass(frozen=True)
class DirectionalProjectionContext:
    """共享候选投影与其 carrier-local 输入的无损方向消费上下文。"""

    projection: ArtifactSemanticProjection
    carrier_input: CarrierProjectionInput

    def __post_init__(self) -> None:
        if not isinstance(self.projection, ArtifactSemanticProjection):
            raise TypeError("directional projection 类型非法")
        if not isinstance(self.carrier_input, CarrierProjectionInput):
            raise TypeError("directional carrier input 类型非法")
        projection = self.projection
        carrier_input = self.carrier_input
        if (projection.source != carrier_input.source
                or projection.scope != carrier_input.scope
                or projection.envelope_identity
                != carrier_input.envelope.identity
                or projection.anchor_identities
                != carrier_input.anchor_identities
                or projection.structure_node_identities
                != carrier_input.structure_node_identities
                or projection.projection_key != carrier_input.input_key):
            raise CarrierDirectionalContractError(
                "directional context 未精确绑定 mapper input")

    @property
    def subject_identities(self) -> tuple[ObjectIdentity, ...]:
        return tuple(sorted(
            set(self.projection.anchor_identities)
            | set(self.projection.structure_node_identities),
            key=ObjectIdentity.stable_key,
        ))

    @property
    def evidence_keys(self) -> tuple[tuple[int, ...], ...]:
        return tuple(sorted(
            item.stable_key() for item in self.projection.evidence))

    def require_direction(self, direction: int) -> None:
        if direction not in self.projection.directions:
            raise CarrierDirectionalContractError(
                "projection 未授权当前方向")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            *_pack(self.projection.stable_key()),
            *_pack(self.carrier_input.stable_key()),
        )


@dataclass(frozen=True)
class UnderstandingRequest:
    context: DirectionalProjectionContext
    request_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.context, DirectionalProjectionContext):
            raise TypeError("understanding request context 类型非法")
        self.context.require_direction(PROJECTION_UNDERSTANDING)
        _key(self.request_key, where="understanding request_key")

    def stable_key(self) -> tuple[int, ...]:
        return 1, *_pack(self.context.stable_key()), *_pack(self.request_key)


@dataclass(frozen=True)
class UnderstandingResult:
    """带 carrier subject、共享 feature 和证据路径的语义框架。"""

    request: UnderstandingRequest
    status: str
    semantic_object: ObjectIdentity
    projection_kind: ObjectIdentity
    carrier_family: ObjectIdentity
    subject_identities: tuple[ObjectIdentity, ...]
    feature_identities: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    result_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, UnderstandingRequest):
            raise TypeError("understanding result request 类型非法")
        if self.status not in UNDERSTANDING_STATUSES:
            raise CarrierDirectionalContractError(
                "understanding result status 非法")
        for name in ("semantic_object", "projection_kind", "carrier_family"):
            if not isinstance(getattr(self, name), ObjectIdentity):
                raise TypeError(f"understanding result {name} 类型非法")
        _objects(self.subject_identities, where="understanding subjects")
        _objects(self.feature_identities, where="understanding features")
        _nested_keys(self.evidence_keys, where="understanding evidence")
        _key(self.result_key, where="understanding result_key")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1,
            _UNDERSTANDING_STATUS_CODE[self.status],
            *_pack(self.request.stable_key()),
            *_pack(self.semantic_object.stable_key()),
            *_pack(self.projection_kind.stable_key()),
            *_pack(self.carrier_family.stable_key()),
            len(self.subject_identities),
        ]
        for item in self.subject_identities:
            values.extend(_pack(item.stable_key()))
        values.append(len(self.feature_identities))
        for item in self.feature_identities:
            values.extend(_pack(item.stable_key()))
        values.append(len(self.evidence_keys))
        for item in self.evidence_keys:
            values.extend(_pack(item))
        values.extend(_pack(self.result_key))
        return tuple(values)


@dataclass(frozen=True)
class UnderstandingUse:
    result: UnderstandingResult
    action: str
    use_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.result, UnderstandingResult):
            raise TypeError("understanding Use result 类型非法")
        if self.action not in UNDERSTANDING_ACTIONS:
            raise CarrierDirectionalContractError(
                "understanding Use action 非法")
        _key(self.use_key, where="understanding use_key")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            _UNDERSTANDING_ACTION_CODE[self.action],
            *_pack(self.result.stable_key()),
            *_pack(self.use_key),
        )


@dataclass(frozen=True)
class UnderstandingOutcome:
    use: UnderstandingUse
    verdict: str
    current_context: DirectionalProjectionContext
    outcome_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.use, UnderstandingUse):
            raise TypeError("understanding outcome Use 类型非法")
        if self.verdict not in OUTCOME_VERDICTS:
            raise CarrierDirectionalContractError(
                "understanding outcome verdict 非法")
        if not isinstance(self.current_context, DirectionalProjectionContext):
            raise TypeError("understanding current context 类型非法")
        self.current_context.require_direction(PROJECTION_UNDERSTANDING)
        _key(self.outcome_key, where="understanding outcome_key")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            _OUTCOME_CODE[self.verdict],
            *_pack(self.use.stable_key()),
            *_pack(self.current_context.stable_key()),
            *_pack(self.outcome_key),
        )


@dataclass(frozen=True)
class ReasoningRequest:
    context: DirectionalProjectionContext
    claim: ObjectIdentity
    request_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.context, DirectionalProjectionContext):
            raise TypeError("reasoning request context 类型非法")
        self.context.require_direction(PROJECTION_REASONING)
        if not isinstance(self.claim, ObjectIdentity):
            raise TypeError("reasoning request claim 类型非法")
        _key(self.request_key, where="reasoning request_key")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            *_pack(self.context.stable_key()),
            *_pack(self.claim.stable_key()),
            *_pack(self.request_key),
        )


@dataclass(frozen=True)
class ReasoningResult:
    """从 carrier-grounded premise 与 Evidence 得出的显式结论。"""

    request: ReasoningRequest
    conclusion: str
    premise_identities: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    result_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, ReasoningRequest):
            raise TypeError("reasoning result request 类型非法")
        if self.conclusion not in REASONING_CONCLUSIONS:
            raise CarrierDirectionalContractError(
                "reasoning conclusion 非法")
        _objects(self.premise_identities, where="reasoning premises")
        _nested_keys(self.evidence_keys, where="reasoning evidence")
        _key(self.result_key, where="reasoning result_key")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1,
            _REASONING_CONCLUSION_CODE[self.conclusion],
            *_pack(self.request.stable_key()),
            len(self.premise_identities),
        ]
        for item in self.premise_identities:
            values.extend(_pack(item.stable_key()))
        values.append(len(self.evidence_keys))
        for item in self.evidence_keys:
            values.extend(_pack(item))
        values.extend(_pack(self.result_key))
        return tuple(values)


@dataclass(frozen=True)
class ReasoningUse:
    result: ReasoningResult
    action: str
    use_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.result, ReasoningResult):
            raise TypeError("reasoning Use result 类型非法")
        if self.action not in REASONING_ACTIONS:
            raise CarrierDirectionalContractError("reasoning Use action 非法")
        _key(self.use_key, where="reasoning use_key")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            _REASONING_ACTION_CODE[self.action],
            *_pack(self.result.stable_key()),
            *_pack(self.use_key),
        )


@dataclass(frozen=True)
class ReasoningOutcome:
    use: ReasoningUse
    verdict: str
    current_context: DirectionalProjectionContext
    outcome_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.use, ReasoningUse):
            raise TypeError("reasoning outcome Use 类型非法")
        if self.verdict not in OUTCOME_VERDICTS:
            raise CarrierDirectionalContractError("reasoning verdict 非法")
        if not isinstance(self.current_context, DirectionalProjectionContext):
            raise TypeError("reasoning current context 类型非法")
        self.current_context.require_direction(PROJECTION_REASONING)
        _key(self.outcome_key, where="reasoning outcome_key")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            _OUTCOME_CODE[self.verdict],
            *_pack(self.use.stable_key()),
            *_pack(self.current_context.stable_key()),
            *_pack(self.outcome_key),
        )


@dataclass(frozen=True)
class GenerationRequest:
    """只声明语义 goal 与整数资源边界，不携带期望 surface。"""

    context: DirectionalProjectionContext
    goal: ObjectIdentity
    max_surface_units: int
    request_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.context, DirectionalProjectionContext):
            raise TypeError("generation request context 类型非法")
        self.context.require_direction(PROJECTION_GENERATION)
        if not isinstance(self.goal, ObjectIdentity):
            raise TypeError("generation request goal 类型非法")
        if type(self.max_surface_units) is not int or self.max_surface_units <= 0:
            raise CarrierDirectionalContractError(
                "generation max_surface_units 必须是正严格整数")
        _key(self.request_key, where="generation request_key")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            *_pack(self.context.stable_key()),
            *_pack(self.goal.stable_key()),
            self.max_surface_units,
            *_pack(self.request_key),
        )


@dataclass(frozen=True)
class GenerationResult:
    """由当前投影授权的实际整数 surface，而非 envelope 身份占位。"""

    request: GenerationRequest
    status: str
    raw_unit_kind: int
    surface_units: tuple[int, ...]
    source_subjects: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    result_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, GenerationRequest):
            raise TypeError("generation result request 类型非法")
        if self.status not in GENERATION_STATUSES:
            raise CarrierDirectionalContractError("generation status 非法")
        if type(self.raw_unit_kind) is not int or self.raw_unit_kind <= 0:
            raise CarrierDirectionalContractError(
                "generation raw_unit_kind 必须是正严格整数")
        _integer_tuple(
            self.surface_units,
            where="generation surface_units",
            allow_empty=self.status != GENERATION_READY,
        )
        if self.status == GENERATION_READY and not self.surface_units:
            raise CarrierDirectionalContractError(
                "READY generation 必须含实际 surface units")
        if (self.status != GENERATION_READY and self.surface_units):
            raise CarrierDirectionalContractError(
                "非 READY generation 不得夹带 surface units")
        _objects(self.source_subjects, where="generation source subjects")
        _nested_keys(self.evidence_keys, where="generation evidence")
        _key(self.result_key, where="generation result_key")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1,
            _GENERATION_STATUS_CODE[self.status],
            *_pack(self.request.stable_key()),
            self.raw_unit_kind,
            *_pack(self.surface_units),
            len(self.source_subjects),
        ]
        for item in self.source_subjects:
            values.extend(_pack(item.stable_key()))
        values.append(len(self.evidence_keys))
        for item in self.evidence_keys:
            values.extend(_pack(item))
        values.extend(_pack(self.result_key))
        return tuple(values)


@dataclass(frozen=True)
class GenerationUse:
    result: GenerationResult
    action: str
    use_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.result, GenerationResult):
            raise TypeError("generation Use result 类型非法")
        if self.action not in GENERATION_ACTIONS:
            raise CarrierDirectionalContractError("generation Use action 非法")
        _key(self.use_key, where="generation use_key")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            _GENERATION_ACTION_CODE[self.action],
            *_pack(self.result.stable_key()),
            *_pack(self.use_key),
        )


@dataclass(frozen=True)
class GenerationOutcome:
    use: GenerationUse
    verdict: str
    current_context: DirectionalProjectionContext
    outcome_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.use, GenerationUse):
            raise TypeError("generation outcome Use 类型非法")
        if self.verdict not in OUTCOME_VERDICTS:
            raise CarrierDirectionalContractError("generation verdict 非法")
        if not isinstance(self.current_context, DirectionalProjectionContext):
            raise TypeError("generation current context 类型非法")
        self.current_context.require_direction(PROJECTION_GENERATION)
        _key(self.outcome_key, where="generation outcome_key")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            _OUTCOME_CODE[self.verdict],
            *_pack(self.use.stable_key()),
            *_pack(self.current_context.stable_key()),
            *_pack(self.outcome_key),
        )


__all__ = [
    "CarrierDirectionalContractError", "DirectionalProjectionContext",
    "GENERATION_ACTIONS", "GENERATION_ADOPT", "GENERATION_DEFER",
    "GENERATION_NEEDS_NARROWING", "GENERATION_READY", "GENERATION_REFUSED",
    "GENERATION_REJECT", "GENERATION_STATUSES", "GenerationOutcome",
    "GenerationRequest", "GenerationResult", "GenerationUse",
    "OUTCOME_NEUTRAL", "OUTCOME_REFUTE", "OUTCOME_SUPPORT",
    "OUTCOME_VERDICTS", "REASONING_ACCEPT", "REASONING_ACTIONS",
    "REASONING_CONCLUSIONS", "REASONING_DEFER", "REASONING_REFUTED",
    "REASONING_REJECT", "REASONING_SUPPORTED", "REASONING_UNRESOLVED",
    "ReasoningOutcome", "ReasoningRequest", "ReasoningResult", "ReasoningUse",
    "UNDERSTANDING_ACTIONS", "UNDERSTANDING_ADOPT", "UNDERSTANDING_DEFER",
    "UNDERSTANDING_REJECT", "UNDERSTANDING_REJECTED",
    "UNDERSTANDING_STATUSES", "UNDERSTANDING_UNCERTAIN",
    "UNDERSTANDING_UNDERSTOOD", "UnderstandingOutcome",
    "UnderstandingRequest", "UnderstandingResult", "UnderstandingUse",
]
