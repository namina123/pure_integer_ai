"""W06-R04 PART_OF/HAS_PART 的部分整体查询与生成消费合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ENTITY,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.mereology_relation import (
    MereologyBudget,
    MereologyEvaluation,
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


W06_R04_SUBSTAGE = "MEREOLOGY"
W06_R04_RELATION_FAMILIES = ("PART_OF", "HAS_PART")
W06_R04_CONSUMERS = ("UNDERSTANDING", "REASONING")

W06_R04_SUPPORTED = "SUPPORTED"
W06_R04_REFUTED = "REFUTED"
W06_R04_CONFLICT = "CONFLICT"
W06_R04_UNKNOWN = "UNKNOWN"
W06_R04_QUERY_STATUSES = (
    W06_R04_SUPPORTED,
    W06_R04_REFUTED,
    W06_R04_CONFLICT,
    W06_R04_UNKNOWN,
)

W06_R04_GENERATION_READY = "READY"
W06_R04_GENERATION_UNKNOWN = "UNKNOWN"
W06_R04_GENERATION_REJECTED = "REJECTED"
W06_R04_GENERATION_STATUSES = (
    W06_R04_GENERATION_READY,
    W06_R04_GENERATION_UNKNOWN,
    W06_R04_GENERATION_REJECTED,
)

W06_R04_OUTCOME_SUPPORT = "SUPPORT"
W06_R04_OUTCOME_REFUTE = "REFUTE"
W06_R04_OUTCOMES = (
    W06_R04_OUTCOME_SUPPORT,
    W06_R04_OUTCOME_REFUTE,
)


class W06R04ContractError(ValueError):
    """R04 请求、状态、Use 或生成归因不满足冻结合同。"""


def pack_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def _strict_bool(value: bool, *, where: str) -> bool:
    """拒绝用整数伪装布尔协议位。"""
    if type(value) is not bool:
        raise W06R04ContractError(f"{where} 必须是严格 bool")
    return value


def _identity(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验开放对象字段保留完整 identity。"""
    if not isinstance(value, ObjectIdentity):
        raise W06R04ContractError(f"{where} 必须是 ObjectIdentity")
    return value


def _identities(
        values: tuple[ObjectIdentity, ...], *, where: str,
        allow_empty: bool = False,
        ) -> tuple[ObjectIdentity, ...]:
    """核验 identity 集规范排序、无重复且按合同决定是否可空。"""
    if (not isinstance(values, tuple)
            or (not values and not allow_empty)
            or any(not isinstance(item, ObjectIdentity) for item in values)):
        raise W06R04ContractError(f"{where} identity 集合非法")
    normalized = tuple(sorted(values, key=ObjectIdentity.stable_key))
    if values != normalized or len(set(values)) != len(values):
        raise W06R04ContractError(f"{where} 必须规范排序且无重复")
    return values


def _evidence_keys(
        values: tuple[tuple[int, ...], ...], *, where: str,
        ) -> tuple[tuple[int, ...], ...]:
    """核验 Evidence stable key 集只含严格整数并规范排序。"""
    if (not isinstance(values, tuple)
            or any(not isinstance(item, tuple) or not item for item in values)):
        raise W06R04ContractError(f"{where} Evidence key 集合非法")
    for item in values:
        assert_int(*item, _where=where)
        if any(type(value) is not int for value in item):
            raise W06R04ContractError(f"{where} 必须使用严格整数")
    if values != tuple(sorted(values)) or len(set(values)) != len(values):
        raise W06R04ContractError(f"{where} 必须规范排序且无重复")
    return values


def _role_fillers(
        values: tuple[tuple[ObjectIdentity, ObjectIdentity], ...],
        *, where: str,
        ) -> tuple[tuple[ObjectIdentity, ObjectIdentity], ...]:
    """核验 Role/filler 集规范排序且 Role 唯一。"""
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(item, tuple) or len(item) != 2
                   for item in values)):
        raise W06R04ContractError(f"{where} Role/filler 集合非法")
    for role, filler in values:
        _identity(role, where=f"{where}.role")
        _identity(filler, where=f"{where}.filler")
        if role.object_kind != OBJECT_ROLE:
            raise W06R04ContractError(f"{where} Role 类型非法")
    normalized = tuple(sorted(values, key=lambda item: item[0].stable_key()))
    if values != normalized or len({item[0] for item in values}) != len(values):
        raise W06R04ContractError(f"{where} Role 必须规范排序且唯一")
    return values


def _relation_endpoints(
        relation_family: str,
        part: ObjectIdentity,
        whole: ObjectIdentity,
        ) -> None:
    """核验 PART_OF/HAS_PART 均使用 canonical part/whole 实体端点。"""
    if relation_family not in W06_R04_RELATION_FAMILIES:
        raise W06R04ContractError("W06-R04 relation family 未注册")
    _identity(part, where="mereology part")
    _identity(whole, where="mereology whole")
    if part.object_kind != OBJECT_ENTITY or whole.object_kind != OBJECT_ENTITY:
        raise W06R04ContractError("MEREOLOGY 两端必须都是 Entity")


def _evaluation_key(evaluation: MereologyEvaluation) -> tuple[int, ...]:
    """展开 MereologyEvaluation 的直接证据、proof 和规则反证。"""
    if not isinstance(evaluation, MereologyEvaluation):
        raise W06R04ContractError("mereology evaluation 类型非法")
    values = [
        *pack_key(evaluation.statement.stable_key()),
        *evaluation.state.stable_key(),
        len(evaluation.direct_evidence),
    ]
    for item in evaluation.direct_evidence:
        values.extend(pack_key(item.stable_key()))
    proof = ()
    if evaluation.support_proof is not None:
        proof = evaluation.support_proof.stable_key()
    values.extend(pack_key(proof))
    values.append(len(evaluation.rule_refutes))
    for item in evaluation.rule_refutes:
        values.extend(pack_key(item.stable_key()))
    return tuple(values)


@dataclass(frozen=True)
class W06R04ConsumerProtocol:
    """R04 mereology 桥与三向 consumer 的正交开关。"""

    mereology_bridge_connected: bool = True
    understanding_connected: bool = True
    reasoning_connected: bool = True
    generation_connected: bool = True
    direction_connected: bool = True
    source_scope_connected: bool = True
    postcheck_connected: bool = True

    def __post_init__(self) -> None:
        """核验所有连接位均为严格布尔值。"""
        for name in self.__dataclass_fields__:
            _strict_bool(getattr(self, name), where=name)

    def query_ready(self, consumer: str) -> bool:
        """返回指定 U/R consumer 是否可读取部分整体闭包。"""
        if consumer not in W06_R04_CONSUMERS:
            raise W06R04ContractError("R04 query consumer 未注册")
        connected = (
            self.understanding_connected
            if consumer == "UNDERSTANDING"
            else self.reasoning_connected
        )
        return all((
            self.mereology_bridge_connected,
            connected,
            self.direction_connected,
            self.source_scope_connected,
        ))

    def generation_ready(self) -> bool:
        """返回生成侧是否可消费部分整体结构、方向和来源。"""
        return all((
            self.mereology_bridge_connected,
            self.generation_connected,
            self.direction_connected,
            self.source_scope_connected,
        ))

    def stable_key(self) -> tuple[int, ...]:
        """返回全部连接位的稳定整数键。"""
        return tuple(int(getattr(self, name)) for name in self.__dataclass_fields__)


@dataclass(frozen=True)
class W06R04MereologyQuery:
    """不含 surface cue 或 expected label 的 typed 部分整体查询。"""

    request_key: LosslessIntegerKey
    relation_family: str
    part: ObjectIdentity
    whole: ObjectIdentity
    budget: MereologyBudget

    def __post_init__(self) -> None:
        """核验请求键、关系族、实体端点和预算。"""
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R04ContractError("mereology query request_key 非法")
        _relation_endpoints(self.relation_family, self.part, self.whole)
        if not isinstance(self.budget, MereologyBudget):
            raise W06R04ContractError("mereology query budget 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回查询的完整稳定键。"""
        return (
            *pack_key(self.request_key.components),
            W06_R04_RELATION_FAMILIES.index(self.relation_family) + 1,
            *pack_key(self.part.stable_key()),
            *pack_key(self.whole.stable_key()),
            *self.budget.stable_key(),
        )


@dataclass(frozen=True)
class W06R04MereologyResolution:
    """一次 U/R 部分整体查询的四态结果、proof 和 Evidence 归因。"""

    consumer: str
    request: W06R04MereologyQuery
    status: str
    evaluation: MereologyEvaluation
    propositions: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """核验状态、evaluation 与归因字段。"""
        if self.consumer not in W06_R04_CONSUMERS:
            raise W06R04ContractError("mereology resolution consumer 未注册")
        if not isinstance(self.request, W06R04MereologyQuery):
            raise W06R04ContractError("mereology resolution request 非法")
        if self.status not in W06_R04_QUERY_STATUSES:
            raise W06R04ContractError("mereology resolution status 未注册")
        if not isinstance(self.evaluation, MereologyEvaluation):
            raise W06R04ContractError("mereology resolution evaluation 非法")
        _identities(self.propositions, where="mereology propositions",
                    allow_empty=True)
        _evidence_keys(self.evidence_keys, where="mereology evidence")
        if self.status == W06_R04_UNKNOWN and (
                self.propositions or self.evidence_keys):
            raise W06R04ContractError("UNKNOWN 不得泄漏候选或 Evidence")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R04ContractError("mereology resolution reason_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回 resolution 的完整稳定键。"""
        values = [
            W06_R04_CONSUMERS.index(self.consumer) + 1,
            *pack_key(self.request.stable_key()),
            W06_R04_QUERY_STATUSES.index(self.status) + 1,
            *pack_key(_evaluation_key(self.evaluation)),
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
class W06R04MereologyUse:
    """一次 SUPPORTED 部分整体结论的全部 exact R-00 Use。"""

    resolution: W06R04MereologyResolution
    relation_uses: tuple[RelationClosureUse, ...]
    use_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """要求 Use 精确覆盖支持 proof 的全部直接前提。"""
        if (not isinstance(self.resolution, W06R04MereologyResolution)
                or self.resolution.status != W06_R04_SUPPORTED):
            raise W06R04ContractError("mereology Use 只能采用 SUPPORTED 结论")
        if (not isinstance(self.relation_uses, tuple)
                or not self.relation_uses
                or any(not isinstance(item, RelationClosureUse)
                       for item in self.relation_uses)):
            raise W06R04ContractError("mereology relation Uses 非法")
        if {item.proposition for item in self.relation_uses} != set(
                self.resolution.propositions):
            raise W06R04ContractError("mereology Use 未精确覆盖 proof 前提")
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise W06R04ContractError("mereology use_key 非法")


@dataclass(frozen=True)
class W06R04MereologyOutcome:
    """按 current closure 重验 U/R exact Use 的结果。"""

    use: W06R04MereologyUse
    verdict: str
    current_status: str
    outcome_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """核验 outcome 引用和四态结果。"""
        if not isinstance(self.use, W06R04MereologyUse):
            raise W06R04ContractError("mereology outcome Use 非法")
        if self.verdict not in W06_R04_OUTCOMES:
            raise W06R04ContractError("mereology outcome verdict 非法")
        if self.current_status not in W06_R04_QUERY_STATUSES:
            raise W06R04ContractError("mereology outcome current status 非法")
        if not isinstance(self.outcome_key, LosslessIntegerKey):
            raise W06R04ContractError("mereology outcome key 非法")


@dataclass(frozen=True)
class W06R04GenerationRequest:
    """以 direct active mereology fact 为目标且不含 expected surface 的请求。"""

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
        """核验生成请求只携带 typed target 与输出约束。"""
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R04ContractError("generation request_key 非法")
        _identity(self.target_proposition, where="generation target")
        if self.target_proposition.object_kind != OBJECT_PROPOSITION:
            raise W06R04ContractError("generation target 必须是 Proposition")
        if self.relation_family not in W06_R04_RELATION_FAMILIES:
            raise W06R04ContractError("generation relation family 未注册")
        if self.directionality != DIRECTION_FORWARD:
            raise W06R04ContractError("R04 generation relation 必须有向")
        _role_fillers(self.role_fillers, where="generation role_fillers")
        _identity(self.context, where="generation context")
        if self.context.object_kind != OBJECT_CONTEXT_SCOPE:
            raise W06R04ContractError("generation context 必须是 ContextScope")
        if not isinstance(self.source, SourceRef):
            raise W06R04ContractError("generation source 非法")
        if self.uncertainty_units is not None:
            assert_int(self.uncertainty_units, _where="generation uncertainty")
            if type(self.uncertainty_units) is not int or self.uncertainty_units < 0:
                raise W06R04ContractError("generation uncertainty 非法")
        if not isinstance(self.constraints, GenerationExpressionConstraints):
            raise W06R04ContractError("generation constraints 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回 generation request 的完整稳定键。"""
        values = [
            *pack_key(self.request_key.components),
            *pack_key(self.target_proposition.stable_key()),
            W06_R04_RELATION_FAMILIES.index(self.relation_family) + 1,
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
class W06R04GenerationOption:
    """一个由 direct active mereology fact 授权的来源化表面实现。"""

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
        """核验 option 没有脱离授权 target、结构和来源。"""
        if not isinstance(self.surface, str) or not self.surface:
            raise W06R04ContractError("generation option surface 非法")
        _identity(self.construction, where="generation construction")
        _identity(self.target_proposition, where="generation option target")
        if self.relation_family not in W06_R04_RELATION_FAMILIES:
            raise W06R04ContractError("generation option relation 未注册")
        if self.directionality != DIRECTION_FORWARD:
            raise W06R04ContractError("generation option direction 漂移")
        _role_fillers(self.role_fillers, where="generation option roles")
        _identity(self.context, where="generation option context")
        if not isinstance(self.source, SourceRef):
            raise W06R04ContractError("generation option source 非法")
        if self.uncertainty_units is not None and (
                type(self.uncertainty_units) is not int
                or self.uncertainty_units < 0):
            raise W06R04ContractError("generation option uncertainty 非法")
        _identity(self.language_branch, where="generation language branch")
        if not isinstance(self.authorization_key, LosslessIntegerKey):
            raise W06R04ContractError("generation authorization key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回 generation option 的完整稳定键。"""
        values = [
            len(self.surface), *(ord(item) for item in self.surface),
            *pack_key(self.construction.stable_key()),
            *pack_key(self.target_proposition.stable_key()),
            W06_R04_RELATION_FAMILIES.index(self.relation_family) + 1,
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
class W06R04GenerationChoice:
    """生成部分整体关系结构的全部合法 option。"""

    request: W06R04GenerationRequest
    status: str
    options: tuple[W06R04GenerationOption, ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """核验 READY 与 options 非空状态一致。"""
        if not isinstance(self.request, W06R04GenerationRequest):
            raise W06R04ContractError("generation choice request 非法")
        if self.status not in W06_R04_GENERATION_STATUSES:
            raise W06R04ContractError("generation choice status 非法")
        if (not isinstance(self.options, tuple)
                or any(not isinstance(item, W06R04GenerationOption)
                       for item in self.options)):
            raise W06R04ContractError("generation choice options 非法")
        normalized = tuple(sorted(
            self.options, key=W06R04GenerationOption.stable_key))
        if self.options != normalized or len(set(self.options)) != len(self.options):
            raise W06R04ContractError("generation options 未规范化")
        if (self.status == W06_R04_GENERATION_READY) != bool(self.options):
            raise W06R04ContractError("generation READY 与 option 数不一致")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R04ContractError("generation reason_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回 generation choice 的完整稳定键。"""
        values = [
            *pack_key(self.request.stable_key()),
            W06_R04_GENERATION_STATUSES.index(self.status) + 1,
            len(self.options),
        ]
        for item in self.options:
            values.extend(pack_key(item.stable_key()))
        values.extend(pack_key(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W06R04GenerationUse:
    """一个 generation option 对应的 mereology proof 前提与 Use。"""

    choice: W06R04GenerationChoice
    option: W06R04GenerationOption
    relation_uses: tuple[RelationClosureUse, ...]
    ref: GenerationChoiceUseRef

    def __post_init__(self) -> None:
        """要求 generation Use 精确归因到 target direct fact。"""
        if (not isinstance(self.choice, W06R04GenerationChoice)
                or self.choice.status != W06_R04_GENERATION_READY
                or self.option not in self.choice.options):
            raise W06R04ContractError("generation Use 未采用 READY option")
        if (not isinstance(self.relation_uses, tuple)
                or not self.relation_uses
                or any(not isinstance(item, RelationClosureUse)
                       for item in self.relation_uses)):
            raise W06R04ContractError("generation relation Uses 非法")
        if {item.proposition for item in self.relation_uses} != {
                self.option.target_proposition}:
            raise W06R04ContractError("generation Use 未精确归因目标关系")
        if not isinstance(self.ref, GenerationChoiceUseRef):
            raise W06R04ContractError("generation Use ref 非法")


@dataclass(frozen=True)
class W06R04GenerationOutcome:
    """独立部分整体查询与 surface 结构 postcheck 的分维结果。"""

    use: W06R04GenerationUse
    verdict: str
    ref: GenerationChoiceOutcomeRef
    authorization_current: bool
    relation_structure_preserved: bool
    source_scope_preserved: bool
    surface_structure_valid: bool
    mereology_query_status: str
    recovered_target: bool

    def __post_init__(self) -> None:
        """核验 postcheck 分维结果与最终 verdict 一致。"""
        if not isinstance(self.use, W06R04GenerationUse):
            raise W06R04ContractError("generation outcome Use 非法")
        if self.verdict not in W06_R04_OUTCOMES:
            raise W06R04ContractError("generation outcome verdict 非法")
        if not isinstance(self.ref, GenerationChoiceOutcomeRef):
            raise W06R04ContractError("generation outcome ref 非法")
        for name in (
                "authorization_current", "relation_structure_preserved",
                "source_scope_preserved", "surface_structure_valid",
                "recovered_target"):
            _strict_bool(getattr(self, name), where=name)
        if self.mereology_query_status not in W06_R04_QUERY_STATUSES:
            raise W06R04ContractError("generation mereology query status 非法")
        expected = all((
            self.authorization_current,
            self.relation_structure_preserved,
            self.source_scope_preserved,
            self.surface_structure_valid,
            self.mereology_query_status == W06_R04_SUPPORTED,
            self.recovered_target,
        ))
        if (self.verdict == W06_R04_OUTCOME_SUPPORT) != expected:
            raise W06R04ContractError("generation verdict 未匹配分维 postcheck")


__all__ = [
    "W06R04ConsumerProtocol",
    "W06R04ContractError",
    "W06R04GenerationChoice",
    "W06R04GenerationOption",
    "W06R04GenerationOutcome",
    "W06R04GenerationRequest",
    "W06R04GenerationUse",
    "W06R04MereologyOutcome",
    "W06R04MereologyQuery",
    "W06R04MereologyResolution",
    "W06R04MereologyUse",
    "W06_R04_CONFLICT",
    "W06_R04_CONSUMERS",
    "W06_R04_GENERATION_READY",
    "W06_R04_GENERATION_REJECTED",
    "W06_R04_GENERATION_UNKNOWN",
    "W06_R04_OUTCOME_REFUTE",
    "W06_R04_OUTCOME_SUPPORT",
    "W06_R04_QUERY_STATUSES",
    "W06_R04_REFUTED",
    "W06_R04_RELATION_FAMILIES",
    "W06_R04_SUBSTAGE",
    "W06_R04_SUPPORTED",
    "W06_R04_UNKNOWN",
    "pack_key",
]
