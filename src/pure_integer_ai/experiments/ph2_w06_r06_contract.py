"""W06-R06 PRECEDES/event-time 查询与生成消费合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_AFTER,
    EVENT_TIME_BEFORE,
    EVENT_TIME_DIRECTION_UNKNOWN,
    EVENT_TIME_SAME,
    ResolvedEventTimeRelation,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureUse,
)


W06_R06_RUNTIME_NAMESPACE = 50606
W06_R06_SUBSTAGE = "PRECEDES"
W06_R06_RELATION_FAMILIES = (
    "EVENT_BEFORE",
    "EVENT_AFTER",
    "EVENT_SAME",
    "EVENT_UNKNOWN",
)
W06_R06_CONSUMERS = ("UNDERSTANDING", "REASONING")

W06_R06_SUPPORTED = "SUPPORTED"
W06_R06_REFUTED = "REFUTED"
W06_R06_CONFLICT = "CONFLICT"
W06_R06_UNKNOWN = "UNKNOWN"
W06_R06_QUERY_STATUSES = (
    W06_R06_SUPPORTED,
    W06_R06_REFUTED,
    W06_R06_CONFLICT,
    W06_R06_UNKNOWN,
)

W06_R06_GENERATION_READY = "READY"
W06_R06_GENERATION_UNKNOWN = "UNKNOWN"
W06_R06_GENERATION_REJECTED = "REJECTED"
W06_R06_GENERATION_STATUSES = (
    W06_R06_GENERATION_READY,
    W06_R06_GENERATION_UNKNOWN,
    W06_R06_GENERATION_REJECTED,
)

W06_R06_OUTCOME_SUPPORT = "SUPPORT"
W06_R06_OUTCOME_REFUTE = "REFUTE"
W06_R06_OUTCOMES = (W06_R06_OUTCOME_SUPPORT, W06_R06_OUTCOME_REFUTE)

_DIRECTION_BY_FAMILY = {
    "EVENT_BEFORE": EVENT_TIME_BEFORE,
    "EVENT_AFTER": EVENT_TIME_AFTER,
    "EVENT_SAME": EVENT_TIME_SAME,
    "EVENT_UNKNOWN": EVENT_TIME_DIRECTION_UNKNOWN,
}
_ENDPOINT_KINDS = frozenset({OBJECT_EVENT, OBJECT_PROPOSITION})


class W06R06ContractError(ValueError):
    """R06 请求、限定、Use 或生成归因不满足冻结合同。"""


class W06R06BudgetExceeded(RuntimeError):
    """R06 查询在产生部分结果前耗尽显式预算。"""


def pack_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def _strict_bool(value: bool, *, where: str) -> bool:
    if type(value) is not bool:
        raise W06R06ContractError(f"{where} 必须是严格 bool")
    return value


def _identity(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    if not isinstance(value, ObjectIdentity):
        raise W06R06ContractError(f"{where} 必须是 ObjectIdentity")
    return value


def _identities(
        values: tuple[ObjectIdentity, ...], *, where: str,
        allow_empty: bool = False,
        ) -> tuple[ObjectIdentity, ...]:
    if (not isinstance(values, tuple)
            or (not values and not allow_empty)
            or any(not isinstance(item, ObjectIdentity) for item in values)):
        raise W06R06ContractError(f"{where} identity 集合非法")
    if (values != tuple(sorted(values, key=ObjectIdentity.stable_key))
            or len(set(values)) != len(values)):
        raise W06R06ContractError(f"{where} 必须规范排序且无重复")
    return values


def _evidence_keys(
        values: tuple[tuple[int, ...], ...], *, where: str,
        ) -> tuple[tuple[int, ...], ...]:
    if not isinstance(values, tuple):
        raise W06R06ContractError(f"{where} 必须是 tuple")
    for item in values:
        if not isinstance(item, tuple) or not item:
            raise W06R06ContractError(f"{where} Evidence key 非法")
        assert_int(*item, _where=where)
        if any(type(value) is not int for value in item):
            raise W06R06ContractError(f"{where} 必须使用严格整数")
    if values != tuple(sorted(values)) or len(set(values)) != len(values):
        raise W06R06ContractError(f"{where} 必须规范排序且无重复")
    return values


def _role_fillers(
        values: tuple[tuple[ObjectIdentity, ObjectIdentity], ...],
        *, where: str,
        ) -> tuple[tuple[ObjectIdentity, ObjectIdentity], ...]:
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(item, tuple) or len(item) != 2
                   for item in values)):
        raise W06R06ContractError(f"{where} Role/filler 集合非法")
    for role, filler in values:
        _identity(role, where=f"{where}.role")
        _identity(filler, where=f"{where}.filler")
        if role.object_kind != OBJECT_ROLE:
            raise W06R06ContractError(f"{where} Role 类型非法")
    if (values != tuple(sorted(values, key=lambda item: item[0].stable_key()))
            or len({item[0] for item in values}) != len(values)):
        raise W06R06ContractError(f"{where} Role 必须规范排序且唯一")
    return values


def _endpoints(subject: ObjectIdentity, object_identity: ObjectIdentity) -> None:
    _identity(subject, where="event-time subject")
    _identity(object_identity, where="event-time object")
    if (subject.object_kind not in _ENDPOINT_KINDS
            or object_identity.object_kind not in _ENDPOINT_KINDS):
        raise W06R06ContractError("event-time 端点必须是 Event/Proposition")
    if subject == object_identity:
        raise W06R06ContractError("event-time 查询不得自环")


def _qualifier(
        family: str,
        value: ResolvedEventTimeRelation,
        ) -> ResolvedEventTimeRelation:
    if family not in W06_R06_RELATION_FAMILIES:
        raise W06R06ContractError("R06 relation family 未注册")
    if not isinstance(value, ResolvedEventTimeRelation):
        raise W06R06ContractError("event-time qualifier 类型非法")
    if value.direction != _DIRECTION_BY_FAMILY[family]:
        raise W06R06ContractError("event-time family/direction 漂移")
    return value


def _resolved_key(value: ResolvedEventTimeRelation) -> tuple[int, ...]:
    return (
        *pack_key(value.relation.stable_key()),
        value.direction,
        *pack_key(value.detail_key),
    )


@dataclass(frozen=True)
class W06R06Budget:
    """限制一次查询扫描的 candidate 和 Evidence 数。"""

    max_candidates: int
    max_evidence: int

    def __post_init__(self) -> None:
        assert_int(self.max_candidates, self.max_evidence, _where="R06 budget")
        if (type(self.max_candidates) is not int
                or type(self.max_evidence) is not int
                or self.max_candidates <= 0 or self.max_evidence <= 0):
            raise W06R06ContractError("R06 budget 必须是严格正整数")

    def stable_key(self) -> tuple[int, ...]:
        return self.max_candidates, self.max_evidence


@dataclass(frozen=True)
class W06R06ConsumerProtocol:
    """PRECEDES 四种 state、三向 consumer 与 postcheck 的正交门。"""

    before_connected: bool = True
    after_connected: bool = True
    same_connected: bool = True
    unknown_connected: bool = True
    qualifier_connected: bool = True
    generation_connected: bool = True
    source_scope_connected: bool = True
    postcheck_connected: bool = True

    def __post_init__(self) -> None:
        for name in (
                "before_connected", "after_connected", "same_connected",
                "unknown_connected", "qualifier_connected",
                "generation_connected", "source_scope_connected",
                "postcheck_connected"):
            _strict_bool(getattr(self, name), where=name)

    def family_ready(self, family: str) -> bool:
        if family not in W06_R06_RELATION_FAMILIES:
            raise W06R06ContractError("R06 protocol family 未注册")
        return {
            "EVENT_BEFORE": self.before_connected,
            "EVENT_AFTER": self.after_connected,
            "EVENT_SAME": self.same_connected,
            "EVENT_UNKNOWN": self.unknown_connected,
        }[family]

    def query_ready(self, consumer: str, family: str) -> bool:
        if consumer not in W06_R06_CONSUMERS:
            raise W06R06ContractError("R06 consumer 未注册")
        return self.family_ready(family) and self.qualifier_connected

    def generation_ready(self, family: str) -> bool:
        return (
            self.family_ready(family)
            and self.qualifier_connected
            and self.generation_connected
            and self.source_scope_connected
        )

    def stable_key(self) -> tuple[int, ...]:
        return tuple(int(getattr(self, name)) for name in (
            "before_connected", "after_connected", "same_connected",
            "unknown_connected", "qualifier_connected",
            "generation_connected", "source_scope_connected",
            "postcheck_connected",
        ))


@dataclass(frozen=True)
class W06R06EventTimeQuery:
    """保留 raw relation family、端点和时间限定的一次查询。"""

    request_key: LosslessIntegerKey
    relation_family: str
    subject: ObjectIdentity
    object_identity: ObjectIdentity
    qualifier: ResolvedEventTimeRelation
    budget: W06R06Budget
    source: SourceRef

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R06ContractError("R06 request_key 非法")
        _endpoints(self.subject, self.object_identity)
        _qualifier(self.relation_family, self.qualifier)
        if not isinstance(self.budget, W06R06Budget):
            raise W06R06ContractError("R06 query budget 非法")
        if not isinstance(self.source, SourceRef):
            raise W06R06ContractError("R06 query source 非法")

    def stable_key(self) -> tuple[int, ...]:
        return (
            *pack_key(self.request_key.components),
            W06_R06_RELATION_FAMILIES.index(self.relation_family) + 1,
            *pack_key(self.subject.stable_key()),
            *pack_key(self.object_identity.stable_key()),
            *_resolved_key(self.qualifier),
            *self.budget.stable_key(),
            *pack_key(self.source.stable_key()),
        )


@dataclass(frozen=True)
class W06R06EventTimeEvaluation:
    """W06 active truth 经 event-time direction facade 的四态结果。"""

    query: W06R06EventTimeQuery
    state: LogicEvidenceState
    explicit_unknown: bool
    normalized_before_edge: tuple[ObjectIdentity, ...]
    same_group: tuple[ObjectIdentity, ...]
    active_propositions: tuple[ObjectIdentity, ...]
    matched_propositions: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query, W06R06EventTimeQuery):
            raise W06R06ContractError("event-time evaluation query 非法")
        if not isinstance(self.state, LogicEvidenceState):
            raise W06R06ContractError("event-time evaluation state 非法")
        _strict_bool(self.explicit_unknown, where="explicit_unknown")
        for name in ("normalized_before_edge", "same_group"):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, ObjectIdentity)
                           for item in values)):
                raise W06R06ContractError(f"{name} 类型非法")
        if self.normalized_before_edge and len(self.normalized_before_edge) != 2:
            raise W06R06ContractError("normalized before edge 必须恰有两端")
        if self.same_group and (
                len(self.same_group) != 2
                or self.same_group != tuple(sorted(
                    self.same_group, key=ObjectIdentity.stable_key))):
            raise W06R06ContractError("same group 必须是 canonical pair")
        _identities(
            self.active_propositions,
            where="active_propositions", allow_empty=True)
        _identities(
            self.matched_propositions,
            where="matched_propositions", allow_empty=True)
        if not set(self.active_propositions).issubset(self.matched_propositions):
            raise W06R06ContractError("active propositions 未属于 matched 集")
        _evidence_keys(self.evidence_keys, where="event-time evidence")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *pack_key(self.query.stable_key()),
            int(self.state.support), int(self.state.refute),
            int(self.explicit_unknown),
            len(self.normalized_before_edge),
        ]
        for item in self.normalized_before_edge:
            values.extend(pack_key(item.stable_key()))
        values.append(len(self.same_group))
        for item in self.same_group:
            values.extend(pack_key(item.stable_key()))
        for group in (self.active_propositions, self.matched_propositions):
            values.append(len(group))
            for item in group:
                values.extend(pack_key(item.stable_key()))
        values.append(len(self.evidence_keys))
        for item in self.evidence_keys:
            values.extend(pack_key(item))
        return tuple(values)


@dataclass(frozen=True)
class W06R06EventTimeResolution:
    """Understanding/Reasoning 的 event-time 四态裁决。"""

    consumer: str
    request: W06R06EventTimeQuery
    status: str
    evaluation: W06R06EventTimeEvaluation
    propositions: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if self.consumer not in W06_R06_CONSUMERS:
            raise W06R06ContractError("event-time resolution consumer 非法")
        if self.status not in W06_R06_QUERY_STATUSES:
            raise W06R06ContractError("event-time resolution status 非法")
        if (not isinstance(self.evaluation, W06R06EventTimeEvaluation)
                or self.evaluation.query != self.request):
            raise W06R06ContractError("event-time resolution evaluation 漂移")
        _identities(self.propositions, where="resolution propositions", allow_empty=True)
        _evidence_keys(self.evidence_keys, where="resolution evidence")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R06ContractError("event-time resolution reason 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            W06_R06_CONSUMERS.index(self.consumer) + 1,
            *pack_key(self.request.stable_key()),
            W06_R06_QUERY_STATUSES.index(self.status) + 1,
            *pack_key(self.evaluation.stable_key()),
            len(self.propositions),
        ]
        for item in self.propositions:
            values.extend(pack_key(item.stable_key()))
        values.append(len(self.evidence_keys))
        for item in self.evidence_keys:
            values.extend(pack_key(item))
        values.extend(pack_key(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W06R06EventTimeUse:
    """一次 SUPPORTED event-time resolution 的 exact active-premise Use。"""

    resolution: W06R06EventTimeResolution
    relation_uses: tuple[RelationClosureUse, ...]
    use_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if (not isinstance(self.resolution, W06R06EventTimeResolution)
                or self.resolution.status != W06_R06_SUPPORTED):
            raise W06R06ContractError("event-time Use 必须来自 SUPPORTED resolution")
        if (not isinstance(self.relation_uses, tuple)
                or not self.relation_uses
                or any(not isinstance(item, RelationClosureUse)
                       for item in self.relation_uses)):
            raise W06R06ContractError("event-time relation Uses 非法")
        if {item.proposition for item in self.relation_uses} != set(
                self.resolution.propositions):
            raise W06R06ContractError("event-time Use 未精确归因 Proposition")
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise W06R06ContractError("event-time use_key 非法")


@dataclass(frozen=True)
class W06R06EventTimeOutcome:
    """按 current event-time view 重验历史 Use 的结果。"""

    use: W06R06EventTimeUse
    verdict: str
    current_status: str
    outcome_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.use, W06R06EventTimeUse):
            raise W06R06ContractError("event-time outcome Use 非法")
        if self.verdict not in W06_R06_OUTCOMES:
            raise W06R06ContractError("event-time outcome verdict 非法")
        if self.current_status not in W06_R06_QUERY_STATUSES:
            raise W06R06ContractError("event-time outcome status 非法")
        if not isinstance(self.outcome_key, LosslessIntegerKey):
            raise W06R06ContractError("event-time outcome key 非法")


@dataclass(frozen=True)
class W06R06GenerationRequest:
    """以 direct active PRECEDES fact 为目标且不含 expected surface。"""

    request_key: LosslessIntegerKey
    target_proposition: ObjectIdentity
    relation_family: str
    directionality: int
    role_fillers: tuple[tuple[ObjectIdentity, ObjectIdentity], ...]
    qualifier: ResolvedEventTimeRelation
    context: ObjectIdentity
    source: SourceRef
    uncertainty_units: int | None
    constraints: GenerationExpressionConstraints

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R06ContractError("generation request_key 非法")
        _identity(self.target_proposition, where="generation target")
        if self.target_proposition.object_kind != OBJECT_PROPOSITION:
            raise W06R06ContractError("generation target 必须是 Proposition")
        _qualifier(self.relation_family, self.qualifier)
        if self.directionality != DIRECTION_FORWARD:
            raise W06R06ContractError("R06 generation relation 必须有向")
        _role_fillers(self.role_fillers, where="generation role_fillers")
        _identity(self.context, where="generation context")
        if self.context.object_kind != OBJECT_CONTEXT_SCOPE:
            raise W06R06ContractError("generation context 必须是 ContextScope")
        if not isinstance(self.source, SourceRef):
            raise W06R06ContractError("generation source 非法")
        if self.uncertainty_units is not None and (
                type(self.uncertainty_units) is not int
                or self.uncertainty_units < 0):
            raise W06R06ContractError("generation uncertainty 非法")
        if not isinstance(self.constraints, GenerationExpressionConstraints):
            raise W06R06ContractError("generation constraints 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *pack_key(self.request_key.components),
            *pack_key(self.target_proposition.stable_key()),
            W06_R06_RELATION_FAMILIES.index(self.relation_family) + 1,
            self.directionality,
            len(self.role_fillers),
        ]
        for role, filler in self.role_fillers:
            values.extend(pack_key(role.stable_key()))
            values.extend(pack_key(filler.stable_key()))
        values.extend((
            *_resolved_key(self.qualifier),
            *pack_key(self.context.stable_key()),
            *pack_key(self.source.stable_key()),
            -1 if self.uncertainty_units is None else self.uncertainty_units,
            *pack_key(self.constraints.stable_key()),
        ))
        return tuple(values)


@dataclass(frozen=True)
class W06R06GenerationOption:
    """由 direct active event-time fact 授权的来源化表面实现。"""

    surface: str
    construction: ObjectIdentity
    target_proposition: ObjectIdentity
    relation_family: str
    directionality: int
    role_fillers: tuple[tuple[ObjectIdentity, ObjectIdentity], ...]
    qualifier: ResolvedEventTimeRelation
    canonical_endpoints: tuple[ObjectIdentity, ObjectIdentity]
    context: ObjectIdentity
    source: SourceRef
    uncertainty_units: int | None
    language_branch: ObjectIdentity
    authorization_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface:
            raise W06R06ContractError("generation option surface 非法")
        _identity(self.construction, where="generation construction")
        _identity(self.target_proposition, where="generation option target")
        _qualifier(self.relation_family, self.qualifier)
        if self.directionality != DIRECTION_FORWARD:
            raise W06R06ContractError("generation option direction 漂移")
        _role_fillers(self.role_fillers, where="generation option roles")
        if (not isinstance(self.canonical_endpoints, tuple)
                or len(self.canonical_endpoints) != 2):
            raise W06R06ContractError("generation canonical endpoints 非法")
        _endpoints(*self.canonical_endpoints)
        _identity(self.context, where="generation option context")
        if not isinstance(self.source, SourceRef):
            raise W06R06ContractError("generation option source 非法")
        if self.uncertainty_units is not None and (
                type(self.uncertainty_units) is not int
                or self.uncertainty_units < 0):
            raise W06R06ContractError("generation option uncertainty 非法")
        _identity(self.language_branch, where="generation language branch")
        if not isinstance(self.authorization_key, LosslessIntegerKey):
            raise W06R06ContractError("generation authorization key 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            len(self.surface), *(ord(item) for item in self.surface),
            *pack_key(self.construction.stable_key()),
            *pack_key(self.target_proposition.stable_key()),
            W06_R06_RELATION_FAMILIES.index(self.relation_family) + 1,
            self.directionality,
            len(self.role_fillers),
        ]
        for role, filler in self.role_fillers:
            values.extend(pack_key(role.stable_key()))
            values.extend(pack_key(filler.stable_key()))
        values.extend(_resolved_key(self.qualifier))
        for endpoint in self.canonical_endpoints:
            values.extend(pack_key(endpoint.stable_key()))
        values.extend((
            *pack_key(self.context.stable_key()),
            *pack_key(self.source.stable_key()),
            -1 if self.uncertainty_units is None else self.uncertainty_units,
            *pack_key(self.language_branch.stable_key()),
            *pack_key(self.authorization_key.components),
        ))
        return tuple(values)


@dataclass(frozen=True)
class W06R06GenerationChoice:
    request: W06R06GenerationRequest
    status: str
    options: tuple[W06R06GenerationOption, ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.request, W06R06GenerationRequest):
            raise W06R06ContractError("generation choice request 非法")
        if self.status not in W06_R06_GENERATION_STATUSES:
            raise W06R06ContractError("generation choice status 非法")
        if (not isinstance(self.options, tuple)
                or any(not isinstance(item, W06R06GenerationOption)
                       for item in self.options)):
            raise W06R06ContractError("generation choice options 非法")
        if (self.options != tuple(sorted(
                self.options, key=W06R06GenerationOption.stable_key))
                or len(set(self.options)) != len(self.options)):
            raise W06R06ContractError("generation options 未规范化")
        if (self.status == W06_R06_GENERATION_READY) != bool(self.options):
            raise W06R06ContractError("generation READY 与 option 数不一致")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R06ContractError("generation reason_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *pack_key(self.request.stable_key()),
            W06_R06_GENERATION_STATUSES.index(self.status) + 1,
            len(self.options),
        ]
        for item in self.options:
            values.extend(pack_key(item.stable_key()))
        values.extend(pack_key(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W06R06GenerationUse:
    choice: W06R06GenerationChoice
    option: W06R06GenerationOption
    relation_uses: tuple[RelationClosureUse, ...]
    ref: GenerationChoiceUseRef

    def __post_init__(self) -> None:
        if (not isinstance(self.choice, W06R06GenerationChoice)
                or self.choice.status != W06_R06_GENERATION_READY
                or self.option not in self.choice.options):
            raise W06R06ContractError("generation Use 未采用 READY option")
        if (not isinstance(self.relation_uses, tuple)
                or len(self.relation_uses) != 1
                or any(not isinstance(item, RelationClosureUse)
                       for item in self.relation_uses)):
            raise W06R06ContractError("generation 必须精确提交一个 direct Use")
        if {item.proposition for item in self.relation_uses} != {
                self.option.target_proposition}:
            raise W06R06ContractError("generation Use 未精确归因目标关系")
        if not isinstance(self.ref, GenerationChoiceUseRef):
            raise W06R06ContractError("generation Use ref 非法")


@dataclass(frozen=True)
class W06R06GenerationOutcome:
    use: W06R06GenerationUse
    verdict: str
    ref: GenerationChoiceOutcomeRef
    authorization_current: bool
    relation_qualifier_preserved: bool
    source_scope_preserved: bool
    surface_structure_valid: bool
    event_time_query_status: str
    recovered_target: bool

    def __post_init__(self) -> None:
        if not isinstance(self.use, W06R06GenerationUse):
            raise W06R06ContractError("generation outcome Use 非法")
        if self.verdict not in W06_R06_OUTCOMES:
            raise W06R06ContractError("generation outcome verdict 非法")
        if not isinstance(self.ref, GenerationChoiceOutcomeRef):
            raise W06R06ContractError("generation outcome ref 非法")
        for name in (
                "authorization_current", "relation_qualifier_preserved",
                "source_scope_preserved", "surface_structure_valid",
                "recovered_target"):
            _strict_bool(getattr(self, name), where=name)
        if self.event_time_query_status not in W06_R06_QUERY_STATUSES:
            raise W06R06ContractError("generation event-time status 非法")
        expected = all((
            self.authorization_current,
            self.relation_qualifier_preserved,
            self.source_scope_preserved,
            self.surface_structure_valid,
            self.event_time_query_status == W06_R06_SUPPORTED,
            self.recovered_target,
        ))
        if (self.verdict == W06_R06_OUTCOME_SUPPORT) != expected:
            raise W06R06ContractError("generation verdict 未匹配分维 postcheck")


__all__ = [
    "W06R06Budget",
    "W06R06BudgetExceeded",
    "W06R06ConsumerProtocol",
    "W06R06ContractError",
    "W06R06EventTimeEvaluation",
    "W06R06EventTimeOutcome",
    "W06R06EventTimeQuery",
    "W06R06EventTimeResolution",
    "W06R06EventTimeUse",
    "W06R06GenerationChoice",
    "W06R06GenerationOption",
    "W06R06GenerationOutcome",
    "W06R06GenerationRequest",
    "W06R06GenerationUse",
    "W06_R06_CONFLICT",
    "W06_R06_CONSUMERS",
    "W06_R06_GENERATION_READY",
    "W06_R06_GENERATION_REJECTED",
    "W06_R06_GENERATION_UNKNOWN",
    "W06_R06_OUTCOME_REFUTE",
    "W06_R06_OUTCOME_SUPPORT",
    "W06_R06_QUERY_STATUSES",
    "W06_R06_REFUTED",
    "W06_R06_RELATION_FAMILIES",
    "W06_R06_RUNTIME_NAMESPACE",
    "W06_R06_SUBSTAGE",
    "W06_R06_SUPPORTED",
    "W06_R06_UNKNOWN",
    "pack_key",
]
