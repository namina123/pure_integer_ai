"""W06-R02 SUBSET/MEMBER 的集合查询与生成消费合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ENTITY,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_SET_EXPR,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.set_relation import (
    SetRelationBudget,
    SetRelationEvaluation,
)
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


W06_R02_SUBSTAGE = "SUBSET_MEMBER"
W06_R02_RELATION_FAMILIES = ("SUBSET", "MEMBER")
W06_R02_CONSUMERS = ("UNDERSTANDING", "REASONING")

W06_R02_SUPPORTED = "SUPPORTED"
W06_R02_REFUTED = "REFUTED"
W06_R02_CONFLICT = "CONFLICT"
W06_R02_UNKNOWN = "UNKNOWN"
W06_R02_QUERY_STATUSES = (
    W06_R02_SUPPORTED,
    W06_R02_REFUTED,
    W06_R02_CONFLICT,
    W06_R02_UNKNOWN,
)

W06_R02_GENERATION_READY = "READY"
W06_R02_GENERATION_UNKNOWN = "UNKNOWN"
W06_R02_GENERATION_REJECTED = "REJECTED"
W06_R02_GENERATION_STATUSES = (
    W06_R02_GENERATION_READY,
    W06_R02_GENERATION_UNKNOWN,
    W06_R02_GENERATION_REJECTED,
)

W06_R02_OUTCOME_SUPPORT = "SUPPORT"
W06_R02_OUTCOME_REFUTE = "REFUTE"
W06_R02_OUTCOMES = (
    W06_R02_OUTCOME_SUPPORT,
    W06_R02_OUTCOME_REFUTE,
)


class W06R02ContractError(ValueError):
    """R02 请求、状态、Use 或生成归因不满足冻结合同。"""


def pack_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def _strict_bool(value: bool, *, where: str) -> bool:
    if type(value) is not bool:
        raise W06R02ContractError(f"{where} 必须是严格 bool")
    return value


def _identity(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    if not isinstance(value, ObjectIdentity):
        raise W06R02ContractError(f"{where} 必须是 ObjectIdentity")
    return value


def _identities(
        values: tuple[ObjectIdentity, ...], *, where: str,
        allow_empty: bool = False,
        ) -> tuple[ObjectIdentity, ...]:
    if (not isinstance(values, tuple)
            or (not values and not allow_empty)
            or any(not isinstance(item, ObjectIdentity) for item in values)):
        raise W06R02ContractError(f"{where} identity 集合非法")
    normalized = tuple(sorted(values, key=ObjectIdentity.stable_key))
    if values != normalized or len(set(values)) != len(values):
        raise W06R02ContractError(f"{where} 必须规范排序且无重复")
    return values


def _evidence_keys(
        values: tuple[tuple[int, ...], ...], *, where: str,
        ) -> tuple[tuple[int, ...], ...]:
    if (not isinstance(values, tuple)
            or any(not isinstance(item, tuple) or not item for item in values)):
        raise W06R02ContractError(f"{where} Evidence key 集合非法")
    for item in values:
        assert_int(*item, _where=where)
        if any(type(value) is not int for value in item):
            raise W06R02ContractError(f"{where} 必须使用严格整数")
    if values != tuple(sorted(values)) or len(set(values)) != len(values):
        raise W06R02ContractError(f"{where} 必须规范排序且无重复")
    return values


def _role_fillers(
        values: tuple[tuple[ObjectIdentity, ObjectIdentity], ...],
        *, where: str,
        ) -> tuple[tuple[ObjectIdentity, ObjectIdentity], ...]:
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(item, tuple) or len(item) != 2
                   for item in values)):
        raise W06R02ContractError(f"{where} Role/filler 集合非法")
    for role, filler in values:
        _identity(role, where=f"{where}.role")
        _identity(filler, where=f"{where}.filler")
        if role.object_kind != OBJECT_ROLE:
            raise W06R02ContractError(f"{where} Role 类型非法")
    normalized = tuple(sorted(values, key=lambda item: item[0].stable_key()))
    if values != normalized or len({item[0] for item in values}) != len(values):
        raise W06R02ContractError(f"{where} Role 必须规范排序且唯一")
    return values


def _relation_endpoints(
        relation_family: str,
        left: ObjectIdentity,
        right: ObjectIdentity,
        ) -> None:
    if relation_family not in W06_R02_RELATION_FAMILIES:
        raise W06R02ContractError("W06-R02 relation family 未注册")
    _identity(left, where="set relation left")
    _identity(right, where="set relation right")
    if relation_family == "SUBSET":
        if {left.object_kind, right.object_kind} != {OBJECT_SET_EXPR}:
            raise W06R02ContractError("SUBSET 两端必须都是 SetExpr")
    elif (left.object_kind not in {OBJECT_ENTITY, OBJECT_SET_EXPR}
          or right.object_kind != OBJECT_SET_EXPR):
        raise W06R02ContractError("MEMBER 必须是权威对象到 SetExpr")


@dataclass(frozen=True)
class W06R02ConsumerProtocol:
    """R02 集合桥与三向 consumer 的正交开关。"""

    set_relation_bridge_connected: bool = True
    understanding_connected: bool = True
    reasoning_connected: bool = True
    generation_connected: bool = True
    direction_connected: bool = True
    source_scope_connected: bool = True
    postcheck_connected: bool = True

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _strict_bool(getattr(self, name), where=name)

    def query_ready(self, consumer: str) -> bool:
        """返回指定 U/R consumer 是否可读取集合关系闭包。"""
        if consumer not in W06_R02_CONSUMERS:
            raise W06R02ContractError("R02 query consumer 未注册")
        connected = (
            self.understanding_connected
            if consumer == "UNDERSTANDING"
            else self.reasoning_connected
        )
        return all((
            self.set_relation_bridge_connected,
            connected,
            self.direction_connected,
            self.source_scope_connected,
        ))

    def generation_ready(self) -> bool:
        """返回生成侧是否可消费集合关系结构、方向和来源。"""
        return all((
            self.set_relation_bridge_connected,
            self.generation_connected,
            self.direction_connected,
            self.source_scope_connected,
        ))

    def stable_key(self) -> tuple[int, ...]:
        return tuple(int(getattr(self, name)) for name in self.__dataclass_fields__)


@dataclass(frozen=True)
class W06R02SetQuery:
    """不含 surface cue 或 expected label 的 typed 集合关系查询。"""

    request_key: LosslessIntegerKey
    relation_family: str
    left: ObjectIdentity
    right: ObjectIdentity
    budget: SetRelationBudget

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R02ContractError("set query request_key 非法")
        _relation_endpoints(self.relation_family, self.left, self.right)
        if not isinstance(self.budget, SetRelationBudget):
            raise W06R02ContractError("set query budget 非法")

    def stable_key(self) -> tuple[int, ...]:
        return (
            *pack_key(self.request_key.components),
            W06_R02_RELATION_FAMILIES.index(self.relation_family) + 1,
            *pack_key(self.left.stable_key()),
            *pack_key(self.right.stable_key()),
            *self.budget.stable_key(),
        )


@dataclass(frozen=True)
class W06R02SetResolution:
    """一次 U/R 集合查询的四态结果、proof 和 Evidence 归因。"""

    consumer: str
    request: W06R02SetQuery
    status: str
    evaluation: SetRelationEvaluation
    propositions: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if self.consumer not in W06_R02_CONSUMERS:
            raise W06R02ContractError("set resolution consumer 未注册")
        if not isinstance(self.request, W06R02SetQuery):
            raise W06R02ContractError("set resolution request 非法")
        if self.status not in W06_R02_QUERY_STATUSES:
            raise W06R02ContractError("set resolution status 未注册")
        if not isinstance(self.evaluation, SetRelationEvaluation):
            raise W06R02ContractError("set resolution evaluation 非法")
        _identities(self.propositions, where="set propositions", allow_empty=True)
        _evidence_keys(self.evidence_keys, where="set evidence")
        if self.status == W06_R02_UNKNOWN and (
                self.propositions or self.evidence_keys):
            raise W06R02ContractError("UNKNOWN 不得泄漏候选或 Evidence")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R02ContractError("set resolution reason_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            W06_R02_CONSUMERS.index(self.consumer) + 1,
            *pack_key(self.request.stable_key()),
            W06_R02_QUERY_STATUSES.index(self.status) + 1,
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
class W06R02SetUse:
    """一次 SUPPORTED 集合结论的全部 exact R-00 Use。"""

    resolution: W06R02SetResolution
    relation_uses: tuple[RelationClosureUse, ...]
    use_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if (not isinstance(self.resolution, W06R02SetResolution)
                or self.resolution.status != W06_R02_SUPPORTED):
            raise W06R02ContractError("set Use 只能采用 SUPPORTED 结论")
        if (not isinstance(self.relation_uses, tuple)
                or not self.relation_uses
                or any(not isinstance(item, RelationClosureUse)
                       for item in self.relation_uses)):
            raise W06R02ContractError("set relation Uses 非法")
        if {item.proposition for item in self.relation_uses} != set(
                self.resolution.propositions):
            raise W06R02ContractError("set Use 未精确覆盖 proof 前提")
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise W06R02ContractError("set use_key 非法")


@dataclass(frozen=True)
class W06R02SetOutcome:
    """按 current closure 重验 U/R exact Use 的结果。"""

    use: W06R02SetUse
    verdict: str
    current_status: str
    outcome_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.use, W06R02SetUse):
            raise W06R02ContractError("set outcome Use 非法")
        if self.verdict not in W06_R02_OUTCOMES:
            raise W06R02ContractError("set outcome verdict 非法")
        if self.current_status not in W06_R02_QUERY_STATUSES:
            raise W06R02ContractError("set outcome current status 非法")
        if not isinstance(self.outcome_key, LosslessIntegerKey):
            raise W06R02ContractError("set outcome key 非法")


@dataclass(frozen=True)
class W06R02GenerationRequest:
    """以 direct active set relation 为目标且不含 expected surface 的请求。"""

    request_key: LosslessIntegerKey
    target_proposition: ObjectIdentity
    relation_family: str
    directionality: int
    role_fillers: tuple[tuple[ObjectIdentity, ObjectIdentity], ...]
    context: ObjectIdentity
    source: SourceRef
    uncertainty_units: int | None
    constraints: GenerationExpressionConstraints

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R02ContractError("generation request_key 非法")
        _identity(self.target_proposition, where="generation target")
        if self.target_proposition.object_kind != OBJECT_PROPOSITION:
            raise W06R02ContractError("generation target 必须是 Proposition")
        if self.relation_family not in W06_R02_RELATION_FAMILIES:
            raise W06R02ContractError("generation relation family 未注册")
        if self.directionality != DIRECTION_FORWARD:
            raise W06R02ContractError("R02 generation relation 必须有向")
        _role_fillers(self.role_fillers, where="generation role_fillers")
        _identity(self.context, where="generation context")
        if self.context.object_kind != OBJECT_CONTEXT_SCOPE:
            raise W06R02ContractError("generation context 必须是 ContextScope")
        if not isinstance(self.source, SourceRef):
            raise W06R02ContractError("generation source 非法")
        if self.uncertainty_units is not None:
            assert_int(self.uncertainty_units, _where="generation uncertainty")
            if type(self.uncertainty_units) is not int or self.uncertainty_units < 0:
                raise W06R02ContractError("generation uncertainty 非法")
        if not isinstance(self.constraints, GenerationExpressionConstraints):
            raise W06R02ContractError("generation constraints 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *pack_key(self.request_key.components),
            *pack_key(self.target_proposition.stable_key()),
            W06_R02_RELATION_FAMILIES.index(self.relation_family) + 1,
            self.directionality,
            len(self.role_fillers),
        ]
        for role, filler in self.role_fillers:
            values.extend(pack_key(role.stable_key()))
            values.extend(pack_key(filler.stable_key()))
        values.extend((
            *pack_key(self.context.stable_key()),
            *pack_key(self.source.stable_key()),
            -1 if self.uncertainty_units is None else self.uncertainty_units,
            *pack_key(self.constraints.stable_key()),
        ))
        return tuple(values)


@dataclass(frozen=True)
class W06R02GenerationOption:
    """一个由 direct active set fact 授权的来源化表面实现。"""

    surface: str
    construction: ObjectIdentity
    target_proposition: ObjectIdentity
    relation_family: str
    directionality: int
    role_fillers: tuple[tuple[ObjectIdentity, ObjectIdentity], ...]
    context: ObjectIdentity
    source: SourceRef
    uncertainty_units: int | None
    language_branch: ObjectIdentity
    authorization_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface:
            raise W06R02ContractError("generation option surface 非法")
        _identity(self.construction, where="generation construction")
        _identity(self.target_proposition, where="generation option target")
        if self.relation_family not in W06_R02_RELATION_FAMILIES:
            raise W06R02ContractError("generation option relation 未注册")
        if self.directionality != DIRECTION_FORWARD:
            raise W06R02ContractError("generation option direction 漂移")
        _role_fillers(self.role_fillers, where="generation option roles")
        _identity(self.context, where="generation option context")
        if not isinstance(self.source, SourceRef):
            raise W06R02ContractError("generation option source 非法")
        if self.uncertainty_units is not None and (
                type(self.uncertainty_units) is not int
                or self.uncertainty_units < 0):
            raise W06R02ContractError("generation option uncertainty 非法")
        _identity(self.language_branch, where="generation language branch")
        if not isinstance(self.authorization_key, LosslessIntegerKey):
            raise W06R02ContractError("generation authorization key 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            len(self.surface), *(ord(item) for item in self.surface),
            *pack_key(self.construction.stable_key()),
            *pack_key(self.target_proposition.stable_key()),
            W06_R02_RELATION_FAMILIES.index(self.relation_family) + 1,
            self.directionality,
            len(self.role_fillers),
        ]
        for role, filler in self.role_fillers:
            values.extend(pack_key(role.stable_key()))
            values.extend(pack_key(filler.stable_key()))
        values.extend((
            *pack_key(self.context.stable_key()),
            *pack_key(self.source.stable_key()),
            -1 if self.uncertainty_units is None else self.uncertainty_units,
            *pack_key(self.language_branch.stable_key()),
            *pack_key(self.authorization_key.components),
        ))
        return tuple(values)


@dataclass(frozen=True)
class W06R02GenerationChoice:
    """生成集合关系结构的全部合法 option。"""

    request: W06R02GenerationRequest
    status: str
    options: tuple[W06R02GenerationOption, ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.request, W06R02GenerationRequest):
            raise W06R02ContractError("generation choice request 非法")
        if self.status not in W06_R02_GENERATION_STATUSES:
            raise W06R02ContractError("generation choice status 非法")
        if (not isinstance(self.options, tuple)
                or any(not isinstance(item, W06R02GenerationOption)
                       for item in self.options)):
            raise W06R02ContractError("generation choice options 非法")
        normalized = tuple(sorted(
            self.options, key=W06R02GenerationOption.stable_key))
        if self.options != normalized or len(set(self.options)) != len(self.options):
            raise W06R02ContractError("generation options 未规范化")
        if (self.status == W06_R02_GENERATION_READY) != bool(self.options):
            raise W06R02ContractError("generation READY 与 option 数不一致")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R02ContractError("generation reason_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *pack_key(self.request.stable_key()),
            W06_R02_GENERATION_STATUSES.index(self.status) + 1,
            len(self.options),
        ]
        for item in self.options:
            values.extend(pack_key(item.stable_key()))
        values.extend(pack_key(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W06R02GenerationUse:
    """一个 generation option 对应的集合 proof 前提与 Use。"""

    choice: W06R02GenerationChoice
    option: W06R02GenerationOption
    relation_uses: tuple[RelationClosureUse, ...]
    ref: GenerationChoiceUseRef

    def __post_init__(self) -> None:
        if (not isinstance(self.choice, W06R02GenerationChoice)
                or self.choice.status != W06_R02_GENERATION_READY
                or self.option not in self.choice.options):
            raise W06R02ContractError("generation Use 未采用 READY option")
        if (not isinstance(self.relation_uses, tuple)
                or not self.relation_uses
                or any(not isinstance(item, RelationClosureUse)
                       for item in self.relation_uses)):
            raise W06R02ContractError("generation relation Uses 非法")
        if {item.proposition for item in self.relation_uses} != {
                self.option.target_proposition}:
            raise W06R02ContractError("generation Use 未精确归因目标关系")
        if not isinstance(self.ref, GenerationChoiceUseRef):
            raise W06R02ContractError("generation Use ref 非法")


@dataclass(frozen=True)
class W06R02GenerationOutcome:
    """独立集合查询与 surface 结构 postcheck 的分维结果。"""

    use: W06R02GenerationUse
    verdict: str
    ref: GenerationChoiceOutcomeRef
    authorization_current: bool
    relation_structure_preserved: bool
    source_scope_preserved: bool
    surface_structure_valid: bool
    set_query_status: str
    recovered_target: bool

    def __post_init__(self) -> None:
        if not isinstance(self.use, W06R02GenerationUse):
            raise W06R02ContractError("generation outcome Use 非法")
        if self.verdict not in W06_R02_OUTCOMES:
            raise W06R02ContractError("generation outcome verdict 非法")
        if not isinstance(self.ref, GenerationChoiceOutcomeRef):
            raise W06R02ContractError("generation outcome ref 非法")
        for name in (
                "authorization_current", "relation_structure_preserved",
                "source_scope_preserved", "surface_structure_valid",
                "recovered_target"):
            _strict_bool(getattr(self, name), where=name)
        if self.set_query_status not in W06_R02_QUERY_STATUSES:
            raise W06R02ContractError("generation set query status 非法")
        expected = all((
            self.authorization_current,
            self.relation_structure_preserved,
            self.source_scope_preserved,
            self.surface_structure_valid,
            self.set_query_status == W06_R02_SUPPORTED,
            self.recovered_target,
        ))
        if (self.verdict == W06_R02_OUTCOME_SUPPORT) != expected:
            raise W06R02ContractError("generation verdict 未匹配分维 postcheck")


__all__ = [
    "W06R02ConsumerProtocol",
    "W06R02ContractError",
    "W06R02GenerationChoice",
    "W06R02GenerationOption",
    "W06R02GenerationOutcome",
    "W06R02GenerationRequest",
    "W06R02GenerationUse",
    "W06R02SetOutcome",
    "W06R02SetQuery",
    "W06R02SetResolution",
    "W06R02SetUse",
    "W06_R02_CONFLICT",
    "W06_R02_CONSUMERS",
    "W06_R02_GENERATION_READY",
    "W06_R02_GENERATION_REJECTED",
    "W06_R02_GENERATION_UNKNOWN",
    "W06_R02_OUTCOME_REFUTE",
    "W06_R02_OUTCOME_SUPPORT",
    "W06_R02_QUERY_STATUSES",
    "W06_R02_REFUTED",
    "W06_R02_RELATION_FAMILIES",
    "W06_R02_SUBSTAGE",
    "W06_R02_SUPPORTED",
    "W06_R02_UNKNOWN",
    "pack_key",
]
