"""W05-G Proposition/Role/Scope construction choice、Use 与 outcome 合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE_BINDING,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w05_adapter import W05OccurrenceBinding


W05_GENERATION_READY = "READY"
W05_GENERATION_CLARIFY = "CLARIFY"
W05_GENERATION_UNKNOWN = "UNKNOWN"
W05_GENERATION_REJECTED = "REJECTED"
W05_GENERATION_STATUSES = (
    W05_GENERATION_READY,
    W05_GENERATION_CLARIFY,
    W05_GENERATION_UNKNOWN,
    W05_GENERATION_REJECTED,
)
W05_GENERATION_ADOPTED = "ADOPTED"
W05_GENERATION_OPTION_REJECTED = "REJECTED"
W05_GENERATION_ACTIONS = (
    W05_GENERATION_ADOPTED,
    W05_GENERATION_OPTION_REJECTED,
)
W05_GENERATION_OUTCOME_SUPPORT = "SUPPORT"
W05_GENERATION_OUTCOME_REFUTE = "REFUTE"
W05_GENERATION_OUTCOME_NEUTRAL = "NEUTRAL"
W05_GENERATION_OUTCOMES = (
    W05_GENERATION_OUTCOME_SUPPORT,
    W05_GENERATION_OUTCOME_REFUTE,
    W05_GENERATION_OUTCOME_NEUTRAL,
)
W05_GENERATION_HARD_CASES = (
    "TARGET_PROPOSITION_LEGAL_CONSTRUCTION",
    "OCCURRENCE_ORDER_PRESERVATION",
    "ROLE_BINDING_PRESERVATION",
    "CONTEXT_SCOPE_PRESERVATION",
    "INDEPENDENT_UNDERSTANDING_RECOVERY",
    "EXACT_CHOICE_USE_OUTCOME",
)

_STATUS_CODE = {
    value: ordinal for ordinal, value in enumerate(W05_GENERATION_STATUSES, start=1)
}
_ACTION_CODE = {
    value: ordinal for ordinal, value in enumerate(W05_GENERATION_ACTIONS, start=1)
}
_OUTCOME_CODE = {
    value: ordinal for ordinal, value in enumerate(W05_GENERATION_OUTCOMES, start=1)
}


class W05GenerationError(RuntimeError):
    """W05-G request、choice、Use、outcome 或 hard conjunct 非法。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _strict_bool(value: bool, *, where: str) -> bool:
    if type(value) is not bool:
        raise W05GenerationError(f"{where} 必须是严格 bool")
    return value


def _definition_key(value: AtomicPropositionDefinition) -> tuple[int, ...]:
    values = [
        1,
        *_pack(value.proposition.stable_key()),
        *_pack(value.predicate.stable_key()),
        *_pack(value.source_anchor.stable_key()),
        *_pack(value.context.stable_key()),
        len(value.bindings),
    ]
    for item in value.canonical_bindings():
        values.extend(_pack(item.identity_for(value.proposition).stable_key()))
    return tuple(values)


@dataclass(frozen=True)
class W05GenerationProtocol:
    """Generation 的结构、choice 与独立回读连接状态。"""

    occurrence_identity_connected: bool = True
    proposition_consumer_connected: bool = True
    role_bridge_connected: bool = True
    scope_projection_connected: bool = True
    choice_bridge_connected: bool = True
    generation_bridge_connected: bool = True
    independent_understanding_connected: bool = True

    def __post_init__(self) -> None:
        for name in (
                "occurrence_identity_connected",
                "proposition_consumer_connected",
                "role_bridge_connected",
                "scope_projection_connected",
                "choice_bridge_connected",
                "generation_bridge_connected",
                "independent_understanding_connected"):
            _strict_bool(getattr(self, name), where=name)

    def structure_connected(self) -> bool:
        return all((
            self.occurrence_identity_connected,
            self.proposition_consumer_connected,
            self.role_bridge_connected,
            self.scope_projection_connected,
        ))


@dataclass(frozen=True)
class W05GenerationRequest:
    """目标 Proposition/Role/Scope、source/uncertainty 与表达约束。"""

    request_key: LosslessIntegerKey
    target: AtomicPropositionDefinition
    occurrences: tuple[W05OccurrenceBinding, ...]
    occurrence_order: tuple[ObjectIdentity, ...]
    role_bindings: tuple[ObjectIdentity, ...]
    source: SourceRef
    scope: ScopeIdentity
    uncertainty: ObjectIdentity
    constraints: GenerationExpressionConstraints

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W05GenerationError("generation request_key 类型非法")
        if not isinstance(self.target, AtomicPropositionDefinition):
            raise W05GenerationError("generation target 非 AtomicPropositionDefinition")
        if (not isinstance(self.occurrences, tuple) or not self.occurrences
                or any(not isinstance(item, W05OccurrenceBinding)
                       for item in self.occurrences)):
            raise W05GenerationError("generation occurrences 类型非法")
        if len({item.identity for item in self.occurrences}) != len(
                self.occurrences):
            raise W05GenerationError("generation occurrence identity 重复")
        if (not isinstance(self.occurrence_order, tuple)
                or any(not isinstance(item, ObjectIdentity)
                       or item.object_kind != OBJECT_OCCURRENCE
                       for item in self.occurrence_order)
                or len(set(self.occurrence_order)) != len(self.occurrence_order)
                or self.occurrence_order
                != tuple(item.identity for item in self.occurrences)):
            raise W05GenerationError("generation occurrence_order 漂移")
        if (not isinstance(self.role_bindings, tuple) or not self.role_bindings
                or any(not isinstance(item, ObjectIdentity)
                       or item.object_kind != OBJECT_ROLE_BINDING
                       for item in self.role_bindings)):
            raise W05GenerationError("generation role_bindings 类型非法")
        expected_roles = tuple(sorted((
            item.identity_for(self.target.proposition)
            for item in self.target.canonical_bindings()
        ), key=ObjectIdentity.stable_key))
        if self.role_bindings != expected_roles:
            raise W05GenerationError("generation role_bindings 与 target 漂移")
        if not isinstance(self.source, SourceRef) or self.source != self.target.source:
            raise W05GenerationError("generation source 与 target 漂移")
        if (not isinstance(self.scope, ScopeIdentity)
                or self.scope.source != self.source):
            raise W05GenerationError("generation authorization scope 漂移")
        if not isinstance(self.uncertainty, ObjectIdentity):
            raise W05GenerationError("generation uncertainty identity 非法")
        if not isinstance(self.constraints, GenerationExpressionConstraints):
            raise W05GenerationError("generation expression constraints 非法")
        if self.constraints.target_language.object_kind != OBJECT_LANGUAGE_BRANCH:
            raise W05GenerationError("generation target language 非 LanguageBranch")
        if self.target.source_anchor not in self.occurrence_order:
            raise W05GenerationError("generation source anchor 不在 occurrence_order")
        if any(item.source != self.source for item in self.occurrences):
            raise W05GenerationError("generation occurrences 与 target source 漂移")
        semantic_objects = {item.semantic_object for item in self.occurrences}
        if any(item.filler not in semantic_objects
               for item in self.target.canonical_bindings()):
            raise W05GenerationError("generation Role filler 缺 occurrence realization")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1,
            *_pack(self.request_key.components),
            *_pack(_definition_key(self.target)),
            len(self.occurrences),
        ]
        for item in self.occurrences:
            values.extend(_pack((
                *_pack(item.identity.stable_key()),
                *_pack(item.semantic_object.stable_key()),
                item.start,
                item.end,
                item.ordinal,
                len(item.surface_fragment),
                *(ord(character) for character in item.surface_fragment),
            )))
        values.append(len(self.role_bindings))
        for item in self.role_bindings:
            values.extend(_pack(item.stable_key()))
        values.extend((
            *_pack(self.source.stable_key()),
            *_pack(self.scope.stable_key()),
            *_pack(self.uncertainty.stable_key()),
            *_pack(self.constraints.stable_key()),
        ))
        return tuple(values)


@dataclass(frozen=True)
class W05GenerationOption:
    """active construction 对目标 Proposition 的一个合法 surface 实例。"""

    surface: str
    construction: ObjectIdentity
    construction_source_candidate: ObjectIdentity
    target_proposition: ObjectIdentity
    target_predicate: ObjectIdentity
    occurrence_order: tuple[ObjectIdentity, ...]
    role_bindings: tuple[ObjectIdentity, ...]
    context: ObjectIdentity
    source: SourceRef
    uncertainty: ObjectIdentity
    branch: ObjectIdentity
    authorization_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface:
            raise W05GenerationError("generation option surface 非法")
        if (not isinstance(self.construction, ObjectIdentity)
                or self.construction.object_kind != OBJECT_CONCEPT):
            raise W05GenerationError("generation construction 非 Concept")
        if (not isinstance(self.construction_source_candidate, ObjectIdentity)
                or self.construction_source_candidate.object_kind
                != OBJECT_PROPOSITION):
            raise W05GenerationError("construction source candidate 非 Proposition")
        if (not isinstance(self.target_proposition, ObjectIdentity)
                or self.target_proposition.object_kind != OBJECT_PROPOSITION):
            raise W05GenerationError("generation target proposition 非法")
        if (not isinstance(self.context, ObjectIdentity)
                or self.context.object_kind != OBJECT_CONTEXT_SCOPE):
            raise W05GenerationError("generation option context 非法")
        if not isinstance(self.source, SourceRef):
            raise W05GenerationError("generation option source 非法")
        if not isinstance(self.uncertainty, ObjectIdentity):
            raise W05GenerationError("generation option uncertainty 非法")
        if (not isinstance(self.branch, ObjectIdentity)
                or self.branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise W05GenerationError("generation option branch 非法")
        if (not isinstance(self.occurrence_order, tuple)
                or not self.occurrence_order
                or any(item.object_kind != OBJECT_OCCURRENCE
                       for item in self.occurrence_order)):
            raise W05GenerationError("generation option occurrence_order 非法")
        if (not isinstance(self.role_bindings, tuple)
                or not self.role_bindings
                or any(item.object_kind != OBJECT_ROLE_BINDING
                       for item in self.role_bindings)):
            raise W05GenerationError("generation option role_bindings 非法")
        if not isinstance(self.authorization_key, LosslessIntegerKey):
            raise W05GenerationError("generation authorization key 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1,
            len(self.surface),
            *(ord(character) for character in self.surface),
            *_pack(self.construction.stable_key()),
            *_pack(self.construction_source_candidate.stable_key()),
            *_pack(self.target_proposition.stable_key()),
            *_pack(self.target_predicate.stable_key()),
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
            *_pack(self.uncertainty.stable_key()),
            *_pack(self.branch.stable_key()),
            *_pack(self.authorization_key.components),
        ))
        return tuple(values)


@dataclass(frozen=True)
class W05GenerationChoice:
    """不比较唯一 expected 字符串、不私选首项的 construction option 集。"""

    request: W05GenerationRequest
    status: str
    options: tuple[W05GenerationOption, ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.request, W05GenerationRequest):
            raise W05GenerationError("generation choice request 非法")
        if self.status not in W05_GENERATION_STATUSES:
            raise W05GenerationError("generation choice status 非法")
        if (not isinstance(self.options, tuple)
                or any(not isinstance(item, W05GenerationOption)
                       for item in self.options)):
            raise W05GenerationError("generation choice options 非法")
        normalized = tuple(sorted(self.options, key=lambda item: item.stable_key()))
        if self.options != normalized or len(set(self.options)) != len(self.options):
            raise W05GenerationError("generation options 未规范化")
        if self.status == W05_GENERATION_READY:
            if not self.options:
                raise W05GenerationError("READY choice 缺 construction option")
        elif self.options:
            raise W05GenerationError("非 READY choice 不得泄漏 option")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W05GenerationError("generation choice reason key 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1,
            _STATUS_CODE[self.status],
            *_pack(self.request.stable_key()),
            len(self.options),
        ]
        for item in self.options:
            values.extend(_pack(item.stable_key()))
        values.extend(_pack(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W05GenerationDecision:
    """对一个 exact construction option 的采用或拒绝。"""

    choice: W05GenerationChoice
    option: W05GenerationOption
    action: str
    decision_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.choice, W05GenerationChoice):
            raise W05GenerationError("generation decision choice 非法")
        if self.option not in self.choice.options:
            raise W05GenerationError("generation decision option 不属于 choice")
        if self.action not in W05_GENERATION_ACTIONS:
            raise W05GenerationError("generation decision action 非法")
        if not isinstance(self.decision_key, LosslessIntegerKey):
            raise W05GenerationError("generation decision key 非法")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            _ACTION_CODE[self.action],
            *_pack(self.choice.stable_key()),
            *_pack(self.option.stable_key()),
            *_pack(self.decision_key.components),
        )


@dataclass(frozen=True)
class W05GenerationUse:
    """一次 exact Proposition/Structure construction Use。"""

    decision: W05GenerationDecision
    ref: GenerationChoiceUseRef

    def __post_init__(self) -> None:
        if not isinstance(self.decision, W05GenerationDecision):
            raise W05GenerationError("generation Use decision 非法")
        if not isinstance(self.ref, GenerationChoiceUseRef):
            raise W05GenerationError("generation Use ref 非法")
        if self.ref.scope != self.decision.choice.request.scope:
            raise W05GenerationError("generation Use scope 漂移")
        if self.ref.selection_key.components != self.decision.option.stable_key():
            raise W05GenerationError("generation Use 未绑定 exact option")

    def stable_key(self) -> tuple[int, ...]:
        return 1, *_pack(self.decision.stable_key()), *_pack(self.ref.stable_key())


@dataclass(frozen=True)
class W05GenerationOutcome:
    """独立 Understanding 回读和当前 active authorization 的分层结果。"""

    use: W05GenerationUse
    verdict: str
    ref: GenerationChoiceOutcomeRef
    current_authorization_key: LosslessIntegerKey | None
    understanding_status: str
    occurrence_preserved: bool
    role_preserved: bool
    scope_preserved: bool
    proposition_preserved: bool

    def __post_init__(self) -> None:
        if not isinstance(self.use, W05GenerationUse):
            raise W05GenerationError("generation outcome Use 非法")
        if self.verdict not in W05_GENERATION_OUTCOMES:
            raise W05GenerationError("generation outcome verdict 非法")
        if not isinstance(self.ref, GenerationChoiceOutcomeRef):
            raise W05GenerationError("generation outcome ref 非法")
        if self.ref.use_key != self.use.ref.use_key:
            raise W05GenerationError("generation outcome 未绑定 exact Use")
        if (self.current_authorization_key is not None
                and not isinstance(self.current_authorization_key,
                                   LosslessIntegerKey)):
            raise W05GenerationError("current authorization key 非法")
        if not isinstance(self.understanding_status, str):
            raise W05GenerationError("understanding status 非法")
        for name in (
                "occurrence_preserved",
                "role_preserved",
                "scope_preserved",
                "proposition_preserved"):
            _strict_bool(getattr(self, name), where=name)

    def stable_key(self) -> tuple[int, ...]:
        authorization = (
            () if self.current_authorization_key is None
            else self.current_authorization_key.components
        )
        return (
            1,
            _OUTCOME_CODE[self.verdict],
            *_pack(self.use.stable_key()),
            *_pack(self.ref.stable_key()),
            *_pack(authorization),
            len(self.understanding_status),
            *(ord(character) for character in self.understanding_status),
            int(self.occurrence_preserved),
            int(self.role_preserved),
            int(self.scope_preserved),
            int(self.proposition_preserved),
        )


@dataclass(frozen=True)
class W05GenerationCaseResult:
    """W05-G 一个公开逻辑 case 的结果。"""

    case_name: str
    passed: bool
    evidence_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if self.case_name not in W05_GENERATION_HARD_CASES:
            raise W05GenerationError("generation hard case 未注册")
        _strict_bool(self.passed, where="generation case passed")
        if not isinstance(self.evidence_key, LosslessIntegerKey):
            raise W05GenerationError("generation case evidence key 非法")


@dataclass(frozen=True)
class W05GenerationHardConjunctReport:
    """W-05-G Proposition/Structure 的 1/1 硬合取。"""

    status: str
    passed: int
    required: int
    fail_count: int
    ne_count: int
    cases: tuple[W05GenerationCaseResult, ...]
    protocol: W05GenerationProtocol

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "FAIL"}:
            raise W05GenerationError("generation hard conjunct status 非法")
        for name in ("passed", "required", "fail_count", "ne_count"):
            assert_int(getattr(self, name), _where=f"hard conjunct {name}")
        if self.required != 1 or self.passed not in {0, 1} or self.ne_count != 0:
            raise W05GenerationError("generation hard conjunct score 非法")
        if (not isinstance(self.cases, tuple)
                or tuple(item.case_name for item in self.cases)
                != W05_GENERATION_HARD_CASES):
            raise W05GenerationError("generation hard cases 缺失或顺序漂移")
        if not isinstance(self.protocol, W05GenerationProtocol):
            raise W05GenerationError("generation hard conjunct protocol 非法")
        expected = int(
            all(item.passed for item in self.cases)
            and self.protocol.structure_connected()
            and self.protocol.choice_bridge_connected
            and self.protocol.generation_bridge_connected
            and self.protocol.independent_understanding_connected
        )
        if (self.passed != expected
                or self.fail_count != 1 - expected
                or self.status != ("PASS" if expected else "FAIL")):
            raise W05GenerationError("generation hard conjunct aggregate 漂移")


def run_w05_generation_hard_conjunct(
        cases: tuple[W05GenerationCaseResult, ...],
        *,
        protocol: W05GenerationProtocol,
        ) -> W05GenerationHardConjunctReport:
    """聚合 W05-G 逻辑 cases；异常由上层区分 FAIL 与基础设施 NE。"""
    if (not isinstance(cases, tuple)
            or any(not isinstance(item, W05GenerationCaseResult)
                   for item in cases)):
        raise W05GenerationError("generation hard cases 类型非法")
    if not isinstance(protocol, W05GenerationProtocol):
        raise TypeError("generation protocol 类型非法")
    passed = int(
        tuple(item.case_name for item in cases) == W05_GENERATION_HARD_CASES
        and all(item.passed for item in cases)
        and protocol.structure_connected()
        and protocol.choice_bridge_connected
        and protocol.generation_bridge_connected
        and protocol.independent_understanding_connected
    )
    return W05GenerationHardConjunctReport(
        "PASS" if passed else "FAIL",
        passed,
        1,
        1 - passed,
        0,
        cases,
        protocol,
    )


__all__ = [
    "W05GenerationCaseResult",
    "W05GenerationChoice",
    "W05GenerationDecision",
    "W05GenerationError",
    "W05GenerationHardConjunctReport",
    "W05GenerationOption",
    "W05GenerationOutcome",
    "W05GenerationProtocol",
    "W05GenerationRequest",
    "W05GenerationUse",
    "W05_GENERATION_ACTIONS",
    "W05_GENERATION_ADOPTED",
    "W05_GENERATION_CLARIFY",
    "W05_GENERATION_HARD_CASES",
    "W05_GENERATION_OPTION_REJECTED",
    "W05_GENERATION_OUTCOME_NEUTRAL",
    "W05_GENERATION_OUTCOME_REFUTE",
    "W05_GENERATION_OUTCOME_SUPPORT",
    "W05_GENERATION_READY",
    "W05_GENERATION_REJECTED",
    "W05_GENERATION_UNKNOWN",
    "run_w05_generation_hard_conjunct",
]
