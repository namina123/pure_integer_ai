"""W06-R05 SIMILAR/ANTONYM pair+channel 查询与生成消费合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.symmetric_relation import (
    SymmetricPairEvaluation,
    SymmetricRelationBudget,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_SYMMETRIC,
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


W06_R05_SUBSTAGE = "SIMILAR_ANTONYM"
W06_R05_RELATION_FAMILIES = ("SIMILAR", "ANTONYM")
W06_R05_CONSUMERS = ("UNDERSTANDING", "REASONING")

W06_R05_SUPPORTED = "SUPPORTED"
W06_R05_REFUTED = "REFUTED"
W06_R05_CONFLICT = "CONFLICT"
W06_R05_UNKNOWN = "UNKNOWN"
W06_R05_QUERY_STATUSES = (
    W06_R05_SUPPORTED,
    W06_R05_REFUTED,
    W06_R05_CONFLICT,
    W06_R05_UNKNOWN,
)

W06_R05_GENERATION_READY = "READY"
W06_R05_GENERATION_UNKNOWN = "UNKNOWN"
W06_R05_GENERATION_REJECTED = "REJECTED"
W06_R05_GENERATION_STATUSES = (
    W06_R05_GENERATION_READY,
    W06_R05_GENERATION_UNKNOWN,
    W06_R05_GENERATION_REJECTED,
)

W06_R05_OUTCOME_SUPPORT = "SUPPORT"
W06_R05_OUTCOME_REFUTE = "REFUTE"
W06_R05_OUTCOMES = (
    W06_R05_OUTCOME_SUPPORT,
    W06_R05_OUTCOME_REFUTE,
)


class W06R05ContractError(ValueError):
    """R05 请求、状态、Use 或生成归因不满足冻结合同。"""


def pack_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def _strict_bool(value: bool, *, where: str) -> bool:
    """拒绝用整数伪装布尔协议位。"""
    if type(value) is not bool:
        raise W06R05ContractError(f"{where} 必须是严格 bool")
    return value


def _identity(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验开放对象字段保留完整 identity。"""
    if not isinstance(value, ObjectIdentity):
        raise W06R05ContractError(f"{where} 必须是 ObjectIdentity")
    return value


def _identities(
        values: tuple[ObjectIdentity, ...], *, where: str,
        allow_empty: bool = False,
        ) -> tuple[ObjectIdentity, ...]:
    """核验 identity 集规范排序、无重复且按合同决定是否可空。"""
    if (not isinstance(values, tuple)
            or (not values and not allow_empty)
            or any(not isinstance(item, ObjectIdentity) for item in values)):
        raise W06R05ContractError(f"{where} identity 集合非法")
    normalized = tuple(sorted(values, key=ObjectIdentity.stable_key))
    if values != normalized or len(set(values)) != len(values):
        raise W06R05ContractError(f"{where} 必须规范排序且无重复")
    return values


def _evidence_keys(
        values: tuple[tuple[int, ...], ...], *, where: str,
        ) -> tuple[tuple[int, ...], ...]:
    """核验 Evidence stable key 集只含严格整数并规范排序。"""
    if (not isinstance(values, tuple)
            or any(not isinstance(item, tuple) or not item for item in values)):
        raise W06R05ContractError(f"{where} Evidence key 集合非法")
    for item in values:
        assert_int(*item, _where=where)
        if any(type(value) is not int for value in item):
            raise W06R05ContractError(f"{where} 必须使用严格整数")
    if values != tuple(sorted(values)) or len(set(values)) != len(values):
        raise W06R05ContractError(f"{where} 必须规范排序且无重复")
    return values


def _role_fillers(
        values: tuple[tuple[ObjectIdentity, ObjectIdentity], ...],
        *, where: str,
        ) -> tuple[tuple[ObjectIdentity, ObjectIdentity], ...]:
    """核验 Role/filler 集规范排序且 Role 唯一。"""
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(item, tuple) or len(item) != 2
                   for item in values)):
        raise W06R05ContractError(f"{where} Role/filler 集合非法")
    for role, filler in values:
        _identity(role, where=f"{where}.role")
        _identity(filler, where=f"{where}.filler")
        if role.object_kind != OBJECT_ROLE:
            raise W06R05ContractError(f"{where} Role 类型非法")
    normalized = tuple(sorted(values, key=lambda item: item[0].stable_key()))
    if values != normalized or len({item[0] for item in values}) != len(values):
        raise W06R05ContractError(f"{where} Role 必须规范排序且唯一")
    return values


def _pair_endpoints(
        channel: str,
        left: ObjectIdentity,
        right: ObjectIdentity,
        ) -> None:
    """核验两个 channel 均使用完整 Concept endpoint。"""
    if channel not in W06_R05_RELATION_FAMILIES:
        raise W06R05ContractError("W06-R05 channel 未注册")
    _identity(left, where="semantic pair left")
    _identity(right, where="semantic pair right")
    if left.object_kind != OBJECT_CONCEPT or right.object_kind != OBJECT_CONCEPT:
        raise W06R05ContractError("SIMILAR/ANTONYM 两端必须都是 Concept")


def _evaluation_key(evaluation: SymmetricPairEvaluation) -> tuple[int, ...]:
    """展开 pair 四态、全部直接 Evidence 和规则反证。"""
    if not isinstance(evaluation, SymmetricPairEvaluation):
        raise W06R05ContractError("symmetric evaluation 类型非法")
    values = [
        *pack_key(evaluation.pair.stable_key()),
        *evaluation.state.stable_key(),
        len(evaluation.evidence),
    ]
    for item in evaluation.evidence:
        values.extend(pack_key(item.stable_key()))
    values.append(len(evaluation.rule_refutes))
    for item in evaluation.rule_refutes:
        values.extend(pack_key(item.stable_key()))
    return tuple(values)


@dataclass(frozen=True)
class W06R05ConsumerProtocol:
    """R05 双 channel、U/R/G 和 postcheck 的正交开关。"""

    similar_connected: bool = True
    antonym_connected: bool = True
    understanding_connected: bool = True
    reasoning_connected: bool = True
    generation_connected: bool = True
    channel_identity_connected: bool = True
    source_scope_connected: bool = True
    postcheck_connected: bool = True

    def __post_init__(self) -> None:
        """核验所有连接位均为严格布尔值。"""
        for name in self.__dataclass_fields__:
            _strict_bool(getattr(self, name), where=name)

    def channel_ready(self, channel: str) -> bool:
        """返回指定 relation channel 是否可被消费。"""
        if channel == "SIMILAR":
            return self.similar_connected
        if channel == "ANTONYM":
            return self.antonym_connected
        raise W06R05ContractError("R05 channel 未注册")

    def query_ready(self, consumer: str, channel: str) -> bool:
        """返回指定 U/R consumer 是否可读取 pair+channel。"""
        if consumer not in W06_R05_CONSUMERS:
            raise W06R05ContractError("R05 query consumer 未注册")
        connected = (
            self.understanding_connected
            if consumer == "UNDERSTANDING"
            else self.reasoning_connected
        )
        return all((
            self.channel_ready(channel),
            connected,
            self.channel_identity_connected,
            self.source_scope_connected,
        ))

    def generation_ready(self, channel: str) -> bool:
        """返回生成侧是否可消费指定 channel 的 active pair。"""
        return all((
            self.channel_ready(channel),
            self.generation_connected,
            self.channel_identity_connected,
            self.source_scope_connected,
        ))

    def stable_key(self) -> tuple[int, ...]:
        """返回全部连接位的稳定整数键。"""
        return tuple(int(getattr(self, name)) for name in self.__dataclass_fields__)


@dataclass(frozen=True)
class W06R05PairQuery:
    """不含 surface cue 或 expected label 的 typed pair+channel 查询。"""

    request_key: LosslessIntegerKey
    channel: str
    left: ObjectIdentity
    right: ObjectIdentity
    budget: SymmetricRelationBudget
    source: SourceRef

    def __post_init__(self) -> None:
        """核验请求键、channel、Concept 端点、预算和来源。"""
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R05ContractError("pair query request_key 非法")
        _pair_endpoints(self.channel, self.left, self.right)
        if not isinstance(self.budget, SymmetricRelationBudget):
            raise W06R05ContractError("pair query budget 非法")
        if not isinstance(self.source, SourceRef):
            raise W06R05ContractError("pair query source 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回查询的完整稳定键。"""
        return (
            *pack_key(self.request_key.components),
            W06_R05_RELATION_FAMILIES.index(self.channel) + 1,
            *pack_key(self.left.stable_key()),
            *pack_key(self.right.stable_key()),
            *self.budget.stable_key(),
            *pack_key(self.source.stable_key()),
        )


@dataclass(frozen=True)
class W06R05PairResolution:
    """一次 U/R pair+channel 查询的四态结果和 Evidence 归因。"""

    consumer: str
    request: W06R05PairQuery
    status: str
    evaluation: SymmetricPairEvaluation
    propositions: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """核验状态、evaluation 与归因字段。"""
        if self.consumer not in W06_R05_CONSUMERS:
            raise W06R05ContractError("pair resolution consumer 未注册")
        if not isinstance(self.request, W06R05PairQuery):
            raise W06R05ContractError("pair resolution request 非法")
        if self.status not in W06_R05_QUERY_STATUSES:
            raise W06R05ContractError("pair resolution status 未注册")
        if not isinstance(self.evaluation, SymmetricPairEvaluation):
            raise W06R05ContractError("pair resolution evaluation 非法")
        _identities(self.propositions, where="pair propositions", allow_empty=True)
        _evidence_keys(self.evidence_keys, where="pair evidence")
        if self.status == W06_R05_UNKNOWN and (
                self.propositions or self.evidence_keys):
            raise W06R05ContractError("UNKNOWN 不得泄漏候选或 Evidence")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R05ContractError("pair resolution reason_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回 resolution 的完整稳定键。"""
        values = [
            W06_R05_CONSUMERS.index(self.consumer) + 1,
            *pack_key(self.request.stable_key()),
            W06_R05_QUERY_STATUSES.index(self.status) + 1,
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
class W06R05PairUse:
    """一次 SUPPORTED pair+channel 结论的全部 exact R-00 Use。"""

    resolution: W06R05PairResolution
    relation_uses: tuple[RelationClosureUse, ...]
    use_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """要求 Use 精确覆盖支持 pair 的全部 active premise。"""
        if (not isinstance(self.resolution, W06R05PairResolution)
                or self.resolution.status != W06_R05_SUPPORTED):
            raise W06R05ContractError("pair Use 只能采用 SUPPORTED 结论")
        if (not isinstance(self.relation_uses, tuple)
                or not self.relation_uses
                or any(not isinstance(item, RelationClosureUse)
                       for item in self.relation_uses)):
            raise W06R05ContractError("pair relation Uses 非法")
        if {item.proposition for item in self.relation_uses} != set(
                self.resolution.propositions):
            raise W06R05ContractError("pair Use 未精确覆盖 active premise")
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise W06R05ContractError("pair use_key 非法")


@dataclass(frozen=True)
class W06R05PairOutcome:
    """按 current channel 快照重验 U/R exact Use 的结果。"""

    use: W06R05PairUse
    verdict: str
    current_status: str
    outcome_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """核验 outcome 引用和四态结果。"""
        if not isinstance(self.use, W06R05PairUse):
            raise W06R05ContractError("pair outcome Use 非法")
        if self.verdict not in W06_R05_OUTCOMES:
            raise W06R05ContractError("pair outcome verdict 非法")
        if self.current_status not in W06_R05_QUERY_STATUSES:
            raise W06R05ContractError("pair outcome current status 非法")
        if not isinstance(self.outcome_key, LosslessIntegerKey):
            raise W06R05ContractError("pair outcome key 非法")


@dataclass(frozen=True)
class W06R05GenerationRequest:
    """以 direct active symmetric fact 为目标且不含 expected surface 的请求。"""

    request_key: LosslessIntegerKey
    target_proposition: ObjectIdentity
    channel: str
    directionality: int
    role_fillers: tuple[tuple[ObjectIdentity, ObjectIdentity], ...]
    context: ObjectIdentity
    source: SourceRef
    uncertainty_units: int | None
    constraints: GenerationExpressionConstraints

    def __post_init__(self) -> None:
        """核验生成请求只携带 typed target、channel 与输出约束。"""
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R05ContractError("generation request_key 非法")
        _identity(self.target_proposition, where="generation target")
        if self.target_proposition.object_kind != OBJECT_PROPOSITION:
            raise W06R05ContractError("generation target 必须是 Proposition")
        if self.channel not in W06_R05_RELATION_FAMILIES:
            raise W06R05ContractError("generation channel 未注册")
        if self.directionality != DIRECTION_SYMMETRIC:
            raise W06R05ContractError("R05 generation relation 必须对称")
        _role_fillers(self.role_fillers, where="generation role_fillers")
        _identity(self.context, where="generation context")
        if self.context.object_kind != OBJECT_CONTEXT_SCOPE:
            raise W06R05ContractError("generation context 必须是 ContextScope")
        if not isinstance(self.source, SourceRef):
            raise W06R05ContractError("generation source 非法")
        if self.uncertainty_units is not None:
            assert_int(self.uncertainty_units, _where="generation uncertainty")
            if type(self.uncertainty_units) is not int or self.uncertainty_units < 0:
                raise W06R05ContractError("generation uncertainty 非法")
        if not isinstance(self.constraints, GenerationExpressionConstraints):
            raise W06R05ContractError("generation constraints 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回 generation request 的完整稳定键。"""
        values = [
            *pack_key(self.request_key.components),
            *pack_key(self.target_proposition.stable_key()),
            W06_R05_RELATION_FAMILIES.index(self.channel) + 1,
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
class W06R05GenerationOption:
    """一个由 direct active pair fact 授权的来源化表面实现。"""

    surface: str
    construction: ObjectIdentity
    target_proposition: ObjectIdentity
    channel: str
    directionality: int
    role_fillers: tuple[tuple[ObjectIdentity, ObjectIdentity], ...]
    pair: tuple[ObjectIdentity, ObjectIdentity]
    context: ObjectIdentity
    source: SourceRef
    uncertainty_units: int | None
    language_branch: ObjectIdentity
    authorization_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """核验 option 没有脱离授权 target、pair/channel 和来源。"""
        if not isinstance(self.surface, str) or not self.surface:
            raise W06R05ContractError("generation option surface 非法")
        _identity(self.construction, where="generation construction")
        _identity(self.target_proposition, where="generation option target")
        if self.channel not in W06_R05_RELATION_FAMILIES:
            raise W06R05ContractError("generation option channel 未注册")
        if self.directionality != DIRECTION_SYMMETRIC:
            raise W06R05ContractError("generation option direction 漂移")
        _role_fillers(self.role_fillers, where="generation option roles")
        if not isinstance(self.pair, tuple) or len(self.pair) != 2:
            raise W06R05ContractError("generation option pair 非法")
        _pair_endpoints(self.channel, *self.pair)
        if self.pair != tuple(sorted(self.pair, key=ObjectIdentity.stable_key)):
            raise W06R05ContractError("generation option pair 未 canonical 排序")
        _identity(self.context, where="generation option context")
        if not isinstance(self.source, SourceRef):
            raise W06R05ContractError("generation option source 非法")
        if self.uncertainty_units is not None and (
                type(self.uncertainty_units) is not int
                or self.uncertainty_units < 0):
            raise W06R05ContractError("generation option uncertainty 非法")
        _identity(self.language_branch, where="generation language branch")
        if not isinstance(self.authorization_key, LosslessIntegerKey):
            raise W06R05ContractError("generation authorization key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回 generation option 的完整稳定键。"""
        values = [
            len(self.surface), *(ord(item) for item in self.surface),
            *pack_key(self.construction.stable_key()),
            *pack_key(self.target_proposition.stable_key()),
            W06_R05_RELATION_FAMILIES.index(self.channel) + 1,
            self.directionality,
            len(self.role_fillers),
        ]
        for role, filler in self.role_fillers:
            values.extend(pack_key(role.stable_key()))
            values.extend(pack_key(filler.stable_key()))
        for endpoint in self.pair:
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
class W06R05GenerationChoice:
    """生成指定 pair/channel 结构的全部合法 option。"""

    request: W06R05GenerationRequest
    status: str
    options: tuple[W06R05GenerationOption, ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """核验 READY 与 options 非空状态一致。"""
        if not isinstance(self.request, W06R05GenerationRequest):
            raise W06R05ContractError("generation choice request 非法")
        if self.status not in W06_R05_GENERATION_STATUSES:
            raise W06R05ContractError("generation choice status 非法")
        if (not isinstance(self.options, tuple)
                or any(not isinstance(item, W06R05GenerationOption)
                       for item in self.options)):
            raise W06R05ContractError("generation choice options 非法")
        normalized = tuple(sorted(
            self.options, key=W06R05GenerationOption.stable_key))
        if self.options != normalized or len(set(self.options)) != len(self.options):
            raise W06R05ContractError("generation options 未规范化")
        if (self.status == W06_R05_GENERATION_READY) != bool(self.options):
            raise W06R05ContractError("generation READY 与 option 数不一致")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R05ContractError("generation reason_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回 generation choice 的完整稳定键。"""
        values = [
            *pack_key(self.request.stable_key()),
            W06_R05_GENERATION_STATUSES.index(self.status) + 1,
            len(self.options),
        ]
        for item in self.options:
            values.extend(pack_key(item.stable_key()))
        values.extend(pack_key(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W06R05GenerationUse:
    """一个 generation option 对应的 direct target Use。"""

    choice: W06R05GenerationChoice
    option: W06R05GenerationOption
    relation_uses: tuple[RelationClosureUse, ...]
    ref: GenerationChoiceUseRef

    def __post_init__(self) -> None:
        """要求 generation Use 精确归因到一个 target direct fact。"""
        if (not isinstance(self.choice, W06R05GenerationChoice)
                or self.choice.status != W06_R05_GENERATION_READY
                or self.option not in self.choice.options):
            raise W06R05ContractError("generation Use 未采用 READY option")
        if (not isinstance(self.relation_uses, tuple)
                or len(self.relation_uses) != 1
                or any(not isinstance(item, RelationClosureUse)
                       for item in self.relation_uses)):
            raise W06R05ContractError("generation 必须精确提交一个 direct Use")
        if {item.proposition for item in self.relation_uses} != {
                self.option.target_proposition}:
            raise W06R05ContractError("generation Use 未精确归因目标关系")
        if not isinstance(self.ref, GenerationChoiceUseRef):
            raise W06R05ContractError("generation Use ref 非法")


@dataclass(frozen=True)
class W06R05GenerationOutcome:
    """独立 pair+channel 查询与 surface 结构 postcheck 的分维结果。"""

    use: W06R05GenerationUse
    verdict: str
    ref: GenerationChoiceOutcomeRef
    authorization_current: bool
    pair_channel_preserved: bool
    source_scope_preserved: bool
    surface_structure_valid: bool
    pair_query_status: str
    recovered_target: bool

    def __post_init__(self) -> None:
        """核验 postcheck 分维结果与最终 verdict 一致。"""
        if not isinstance(self.use, W06R05GenerationUse):
            raise W06R05ContractError("generation outcome Use 非法")
        if self.verdict not in W06_R05_OUTCOMES:
            raise W06R05ContractError("generation outcome verdict 非法")
        if not isinstance(self.ref, GenerationChoiceOutcomeRef):
            raise W06R05ContractError("generation outcome ref 非法")
        for name in (
                "authorization_current", "pair_channel_preserved",
                "source_scope_preserved", "surface_structure_valid",
                "recovered_target"):
            _strict_bool(getattr(self, name), where=name)
        if self.pair_query_status not in W06_R05_QUERY_STATUSES:
            raise W06R05ContractError("generation pair query status 非法")
        expected = all((
            self.authorization_current,
            self.pair_channel_preserved,
            self.source_scope_preserved,
            self.surface_structure_valid,
            self.pair_query_status == W06_R05_SUPPORTED,
            self.recovered_target,
        ))
        if (self.verdict == W06_R05_OUTCOME_SUPPORT) != expected:
            raise W06R05ContractError("generation verdict 未匹配分维 postcheck")


__all__ = [
    "W06R05ConsumerProtocol",
    "W06R05ContractError",
    "W06R05GenerationChoice",
    "W06R05GenerationOption",
    "W06R05GenerationOutcome",
    "W06R05GenerationRequest",
    "W06R05GenerationUse",
    "W06R05PairOutcome",
    "W06R05PairQuery",
    "W06R05PairResolution",
    "W06R05PairUse",
    "W06_R05_CONFLICT",
    "W06_R05_CONSUMERS",
    "W06_R05_GENERATION_READY",
    "W06_R05_GENERATION_REJECTED",
    "W06_R05_GENERATION_UNKNOWN",
    "W06_R05_OUTCOME_REFUTE",
    "W06_R05_OUTCOME_SUPPORT",
    "W06_R05_QUERY_STATUSES",
    "W06_R05_REFUTED",
    "W06_R05_RELATION_FAMILIES",
    "W06_R05_SUBSTAGE",
    "W06_R05_SUPPORTED",
    "W06_R05_UNKNOWN",
    "pack_key",
]
