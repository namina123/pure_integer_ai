"""W06-R03 PROPERTY 的六维查询、三向消费与生成 postcheck 合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.property_relation import (
    PropertyClaim,
    PropertyQueryBudget,
    PropertySelection,
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


W06_R03_SUBSTAGE = "PROPERTY"

W06_R03_UNDERSTANDING_UNIQUE = "UNIQUE"
W06_R03_UNDERSTANDING_MULTI = "MULTI"
W06_R03_UNDERSTANDING_CLARIFY = "CLARIFY"
W06_R03_UNDERSTANDING_CONFLICT = "CONFLICT"
W06_R03_UNDERSTANDING_UNKNOWN = "UNKNOWN"
W06_R03_UNDERSTANDING_STATUSES = (
    W06_R03_UNDERSTANDING_UNIQUE,
    W06_R03_UNDERSTANDING_MULTI,
    W06_R03_UNDERSTANDING_CLARIFY,
    W06_R03_UNDERSTANDING_CONFLICT,
    W06_R03_UNDERSTANDING_UNKNOWN,
)

W06_R03_REASONING_SUPPORTED = "SUPPORTED"
W06_R03_REASONING_REFUTED = "REFUTED"
W06_R03_REASONING_CONFLICT = "CONFLICT"
W06_R03_REASONING_UNRESOLVED = "UNRESOLVED"
W06_R03_REASONING_STATUSES = (
    W06_R03_REASONING_SUPPORTED,
    W06_R03_REASONING_REFUTED,
    W06_R03_REASONING_CONFLICT,
    W06_R03_REASONING_UNRESOLVED,
)

W06_R03_GENERATION_READY = "READY"
W06_R03_GENERATION_UNKNOWN = "UNKNOWN"
W06_R03_GENERATION_REJECTED = "REJECTED"
W06_R03_GENERATION_STATUSES = (
    W06_R03_GENERATION_READY,
    W06_R03_GENERATION_UNKNOWN,
    W06_R03_GENERATION_REJECTED,
)

W06_R03_OUTCOME_SUPPORT = "SUPPORT"
W06_R03_OUTCOME_REFUTE = "REFUTE"
W06_R03_OUTCOMES = (
    W06_R03_OUTCOME_SUPPORT,
    W06_R03_OUTCOME_REFUTE,
)


class W06R03ContractError(ValueError):
    """R03 请求、选择、Use 或生成归因不满足冻结合同。"""


def pack_key(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def _strict_bool(value: bool, *, where: str) -> bool:
    """拒绝以整数伪装布尔协议位。"""
    if type(value) is not bool:
        raise W06R03ContractError(f"{where} 必须是严格 bool")
    return value


def _identity(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验开放对象字段保留完整 identity。"""
    if not isinstance(value, ObjectIdentity):
        raise W06R03ContractError(f"{where} 必须是 ObjectIdentity")
    return value


def _claims(
        values: tuple[PropertyClaim, ...], *, where: str,
        allow_empty: bool = False,
        ) -> tuple[PropertyClaim, ...]:
    """核验 claim 集规范排序、无重复且按合同决定是否可空。"""
    if (not isinstance(values, tuple)
            or (not values and not allow_empty)
            or any(not isinstance(item, PropertyClaim) for item in values)):
        raise W06R03ContractError(f"{where} claim 集非法")
    normalized = tuple(sorted(values, key=PropertyClaim.stable_key))
    if values != normalized or len(set(values)) != len(values):
        raise W06R03ContractError(f"{where} claim 集未规范化")
    return values


def _identities(
        values: tuple[ObjectIdentity, ...], *, where: str,
        allow_empty: bool = False,
        ) -> tuple[ObjectIdentity, ...]:
    """核验 identity 集规范排序、无重复且按合同决定是否可空。"""
    if (not isinstance(values, tuple)
            or (not values and not allow_empty)
            or any(not isinstance(item, ObjectIdentity) for item in values)):
        raise W06R03ContractError(f"{where} identity 集非法")
    normalized = tuple(sorted(values, key=ObjectIdentity.stable_key))
    if values != normalized or len(set(values)) != len(values):
        raise W06R03ContractError(f"{where} identity 集未规范化")
    return values


def _evidence_keys(
        values: tuple[tuple[int, ...], ...], *, where: str,
        ) -> tuple[tuple[int, ...], ...]:
    """核验 Evidence stable key 集只含严格整数并规范排序。"""
    if (not isinstance(values, tuple)
            or any(not isinstance(item, tuple) or not item for item in values)):
        raise W06R03ContractError(f"{where} Evidence key 集非法")
    for item in values:
        assert_int(*item, _where=where)
        if any(type(value) is not int for value in item):
            raise W06R03ContractError(f"{where} 必须使用严格整数")
    if values != tuple(sorted(values)) or len(set(values)) != len(values):
        raise W06R03ContractError(f"{where} 必须规范排序且无重复")
    return values


def _role_fillers(
        values: tuple[tuple[ObjectIdentity, ObjectIdentity], ...],
        *, where: str,
        ) -> tuple[tuple[ObjectIdentity, ObjectIdentity], ...]:
    """核验六 Role/filler 结构完整且 Role 唯一。"""
    if (not isinstance(values, tuple) or len(values) != 6
            or any(not isinstance(item, tuple) or len(item) != 2
                   for item in values)):
        raise W06R03ContractError(f"{where} 必须含六个 Role/filler")
    for role, filler in values:
        _identity(role, where=f"{where}.role")
        _identity(filler, where=f"{where}.filler")
        if role.object_kind != OBJECT_ROLE:
            raise W06R03ContractError(f"{where} Role 类型非法")
    normalized = tuple(sorted(values, key=lambda item: item[0].stable_key()))
    if values != normalized or len({item[0] for item in values}) != len(values):
        raise W06R03ContractError(f"{where} Role 必须规范排序且唯一")
    return values


def _relation_uses(
        values: tuple[RelationClosureUse, ...],
        propositions: tuple[ObjectIdentity, ...],
        *, where: str,
        ) -> tuple[RelationClosureUse, ...]:
    """要求 exact Use 与 resolution 前提 Proposition 一一对应。"""
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(item, RelationClosureUse) for item in values)):
        raise W06R03ContractError(f"{where} relation Uses 非法")
    if {item.proposition for item in values} != set(propositions):
        raise W06R03ContractError(f"{where} 未精确覆盖 PROPERTY 前提")
    return values


@dataclass(frozen=True)
class W06R03ConsumerProtocol:
    """R03 direct fact 与三个 consumer 的正交连接开关。"""

    property_bridge_connected: bool = True
    understanding_connected: bool = True
    reasoning_connected: bool = True
    generation_connected: bool = True
    role_structure_connected: bool = True
    intensity_connected: bool = True
    source_scope_connected: bool = True
    postcheck_connected: bool = True

    def __post_init__(self) -> None:
        """核验所有连接位均为严格布尔值。"""
        for name in self.__dataclass_fields__:
            _strict_bool(getattr(self, name), where=name)

    def understanding_ready(self) -> bool:
        """返回 Understanding 是否可读取完整六维 direct fact。"""
        return all((
            self.property_bridge_connected,
            self.understanding_connected,
            self.role_structure_connected,
            self.intensity_connected,
            self.source_scope_connected,
        ))

    def reasoning_ready(self) -> bool:
        """返回 Reasoning 是否可裁决完整六维 direct fact。"""
        return all((
            self.property_bridge_connected,
            self.reasoning_connected,
            self.role_structure_connected,
            self.intensity_connected,
            self.source_scope_connected,
        ))

    def generation_ready(self) -> bool:
        """返回 Generation 是否可消费完整 PROPERTY 结构。"""
        return all((
            self.property_bridge_connected,
            self.generation_connected,
            self.role_structure_connected,
            self.intensity_connected,
            self.source_scope_connected,
        ))

    def stable_key(self) -> tuple[int, ...]:
        """返回全部连接位的稳定整数键。"""
        return tuple(int(getattr(self, name)) for name in self.__dataclass_fields__)


@dataclass(frozen=True)
class W06R03UnderstandingRequest:
    """以 subject/attribute 锚定 PROPERTY 值的无标签查询。"""

    request_key: LosslessIntegerKey
    subject: ObjectIdentity
    attribute: ObjectIdentity
    budget: PropertyQueryBudget
    allow_multiple: bool = False

    def __post_init__(self) -> None:
        """核验 typed 锚、预算和多值策略。"""
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R03ContractError("understanding request_key 非法")
        _identity(self.subject, where="property subject")
        _identity(self.attribute, where="property attribute")
        if not isinstance(self.budget, PropertyQueryBudget):
            raise W06R03ContractError("property understanding budget 非法")
        _strict_bool(self.allow_multiple, where="allow_multiple")

    def stable_key(self) -> tuple[int, ...]:
        """返回请求锚、预算和策略的完整稳定键。"""
        return (
            *pack_key(self.request_key.components),
            *pack_key(self.subject.stable_key()),
            *pack_key(self.attribute.stable_key()),
            *self.budget.stable_key(),
            int(self.allow_multiple),
        )


@dataclass(frozen=True)
class W06R03UnderstandingResolution:
    """PROPERTY 值查询的五态结果、直接选择和 Evidence 归因。"""

    request: W06R03UnderstandingRequest
    status: str
    options: tuple[PropertyClaim, ...]
    selected: PropertyClaim | None
    selection: PropertySelection
    propositions: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """核验状态、选项、唯一选择和归因保持一致。"""
        if not isinstance(self.request, W06R03UnderstandingRequest):
            raise W06R03ContractError("understanding resolution request 非法")
        if self.status not in W06_R03_UNDERSTANDING_STATUSES:
            raise W06R03ContractError("understanding status 未注册")
        _claims(self.options, where="understanding options", allow_empty=True)
        if self.selected is not None and not isinstance(
                self.selected, PropertyClaim):
            raise W06R03ContractError("understanding selected 非法")
        if (self.status == W06_R03_UNDERSTANDING_UNIQUE) != (
                self.selected is not None):
            raise W06R03ContractError("UNIQUE 与 selected 不一致")
        if self.selected is not None and self.selected not in self.options:
            raise W06R03ContractError("selected 不属于 options")
        if not isinstance(self.selection, PropertySelection):
            raise W06R03ContractError("understanding selection 非法")
        _identities(
            self.propositions, where="understanding propositions",
            allow_empty=True)
        _evidence_keys(self.evidence_keys, where="understanding evidence")
        if self.status in {
                W06_R03_UNDERSTANDING_UNKNOWN,
                W06_R03_UNDERSTANDING_CLARIFY,
                } and (self.options or self.propositions or self.evidence_keys):
            raise W06R03ContractError("UNKNOWN/CLARIFY 不得泄漏候选归因")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R03ContractError("understanding reason_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回状态、候选、选择、原始 Evidence 与 reason 的完整键。"""
        values = [
            *pack_key(self.request.stable_key()),
            W06_R03_UNDERSTANDING_STATUSES.index(self.status) + 1,
            len(self.options),
        ]
        for item in self.options:
            values.extend(pack_key(item.stable_key()))
        selected = () if self.selected is None else self.selected.stable_key()
        values.extend(pack_key(selected))
        values.extend((len(self.propositions),))
        for item in self.propositions:
            values.extend(pack_key(item.stable_key()))
        values.append(len(self.evidence_keys))
        for item in self.evidence_keys:
            values.extend(pack_key(item))
        values.extend(pack_key(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W06R03UnderstandingUse:
    """一次 UNIQUE PROPERTY 选择的 exact R-00 Use。"""

    resolution: W06R03UnderstandingResolution
    relation_uses: tuple[RelationClosureUse, ...]
    use_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """要求 Use 只采用 UNIQUE 结果并覆盖全部直接前提。"""
        if (not isinstance(self.resolution, W06R03UnderstandingResolution)
                or self.resolution.status != W06_R03_UNDERSTANDING_UNIQUE):
            raise W06R03ContractError("understanding Use 只能采用 UNIQUE")
        _relation_uses(
            self.relation_uses,
            self.resolution.propositions,
            where="understanding Use",
        )
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise W06R03ContractError("understanding use_key 非法")


@dataclass(frozen=True)
class W06R03UnderstandingOutcome:
    """按 current direct facts 重验 Understanding exact Use 的结果。"""

    use: W06R03UnderstandingUse
    verdict: str
    current_status: str
    outcome_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """核验 verdict、current 状态和 outcome key。"""
        if not isinstance(self.use, W06R03UnderstandingUse):
            raise W06R03ContractError("understanding outcome Use 非法")
        if self.verdict not in W06_R03_OUTCOMES:
            raise W06R03ContractError("understanding verdict 非法")
        if self.current_status not in W06_R03_UNDERSTANDING_STATUSES:
            raise W06R03ContractError("understanding current status 非法")
        if not isinstance(self.outcome_key, LosslessIntegerKey):
            raise W06R03ContractError("understanding outcome key 非法")


@dataclass(frozen=True)
class W06R03ReasoningRequest:
    """对一个完整六维 PROPERTY claim 的 typed 裁决请求。"""

    request_key: LosslessIntegerKey
    claim: PropertyClaim
    budget: PropertyQueryBudget

    def __post_init__(self) -> None:
        """核验请求键、完整 claim 和有界预算。"""
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R03ContractError("reasoning request_key 非法")
        if not isinstance(self.claim, PropertyClaim):
            raise W06R03ContractError("reasoning claim 非法")
        if not isinstance(self.budget, PropertyQueryBudget):
            raise W06R03ContractError("reasoning budget 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回请求键、六维 claim 和预算的稳定键。"""
        return (
            *pack_key(self.request_key.components),
            *pack_key(self.claim.stable_key()),
            *self.budget.stable_key(),
        )


@dataclass(frozen=True)
class W06R03ReasoningResolution:
    """完整 PROPERTY claim 的支持、反驳、冲突或未决结论。"""

    request: W06R03ReasoningRequest
    status: str
    selection: PropertySelection
    propositions: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """核验四态结论与可审计归因字段。"""
        if not isinstance(self.request, W06R03ReasoningRequest):
            raise W06R03ContractError("reasoning resolution request 非法")
        if self.status not in W06_R03_REASONING_STATUSES:
            raise W06R03ContractError("reasoning status 未注册")
        if not isinstance(self.selection, PropertySelection):
            raise W06R03ContractError("reasoning selection 非法")
        _identities(
            self.propositions, where="reasoning propositions",
            allow_empty=True)
        _evidence_keys(self.evidence_keys, where="reasoning evidence")
        if self.status == W06_R03_REASONING_UNRESOLVED and (
                self.propositions or self.evidence_keys):
            raise W06R03ContractError("UNRESOLVED 不得泄漏归因")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R03ContractError("reasoning reason_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回裁决、选择归因与 reason 的完整稳定键。"""
        values = [
            *pack_key(self.request.stable_key()),
            W06_R03_REASONING_STATUSES.index(self.status) + 1,
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
class W06R03ReasoningUse:
    """一次 SUPPORTED PROPERTY 裁决的 exact R-00 Use。"""

    resolution: W06R03ReasoningResolution
    relation_uses: tuple[RelationClosureUse, ...]
    use_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """要求 Use 只采用 SUPPORTED 结论并覆盖全部前提。"""
        if (not isinstance(self.resolution, W06R03ReasoningResolution)
                or self.resolution.status != W06_R03_REASONING_SUPPORTED):
            raise W06R03ContractError("reasoning Use 只能采用 SUPPORTED")
        _relation_uses(
            self.relation_uses,
            self.resolution.propositions,
            where="reasoning Use",
        )
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise W06R03ContractError("reasoning use_key 非法")


@dataclass(frozen=True)
class W06R03ReasoningOutcome:
    """按 current direct facts 重验 Reasoning exact Use 的结果。"""

    use: W06R03ReasoningUse
    verdict: str
    current_status: str
    outcome_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """核验 verdict、current 状态和 outcome key。"""
        if not isinstance(self.use, W06R03ReasoningUse):
            raise W06R03ContractError("reasoning outcome Use 非法")
        if self.verdict not in W06_R03_OUTCOMES:
            raise W06R03ContractError("reasoning verdict 非法")
        if self.current_status not in W06_R03_REASONING_STATUSES:
            raise W06R03ContractError("reasoning current status 非法")
        if not isinstance(self.outcome_key, LosslessIntegerKey):
            raise W06R03ContractError("reasoning outcome key 非法")


@dataclass(frozen=True)
class W06R03GenerationRequest:
    """以 direct active PROPERTY 为目标且不含 expected surface 的请求。"""

    request_key: LosslessIntegerKey
    target_proposition: ObjectIdentity
    claim: PropertyClaim
    directionality: int
    role_fillers: tuple[tuple[ObjectIdentity, ObjectIdentity], ...]
    context: ObjectIdentity
    source: SourceRef
    uncertainty_units: int | None
    constraints: GenerationExpressionConstraints

    def __post_init__(self) -> None:
        """核验目标、六维结构、来源和表达约束。"""
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R03ContractError("generation request_key 非法")
        _identity(self.target_proposition, where="generation target")
        if self.target_proposition.object_kind != OBJECT_PROPOSITION:
            raise W06R03ContractError("generation target 必须是 Proposition")
        if not isinstance(self.claim, PropertyClaim):
            raise W06R03ContractError("generation claim 非法")
        if self.directionality != DIRECTION_FORWARD:
            raise W06R03ContractError("PROPERTY generation 必须有向")
        _role_fillers(self.role_fillers, where="generation role_fillers")
        _identity(self.context, where="generation context")
        if self.context.object_kind != OBJECT_CONTEXT_SCOPE:
            raise W06R03ContractError("generation context 必须是 ContextScope")
        if not isinstance(self.source, SourceRef):
            raise W06R03ContractError("generation source 非法")
        if self.uncertainty_units is not None:
            assert_int(self.uncertainty_units, _where="generation uncertainty")
            if type(self.uncertainty_units) is not int or self.uncertainty_units < 0:
                raise W06R03ContractError("generation uncertainty 非法")
        if not isinstance(self.constraints, GenerationExpressionConstraints):
            raise W06R03ContractError("generation constraints 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回目标、claim、六 Role、来源和约束的完整键。"""
        values = [
            *pack_key(self.request_key.components),
            *pack_key(self.target_proposition.stable_key()),
            *pack_key(self.claim.stable_key()),
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
class W06R03GenerationOption:
    """一个由 direct active PROPERTY 授权的来源化表面实现。"""

    surface: str
    construction: ObjectIdentity
    target_proposition: ObjectIdentity
    claim: PropertyClaim
    directionality: int
    role_fillers: tuple[tuple[ObjectIdentity, ObjectIdentity], ...]
    context: ObjectIdentity
    source: SourceRef
    uncertainty_units: int | None
    language_branch: ObjectIdentity
    authorization_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """核验 option 保留完整结构、来源和授权键。"""
        if not isinstance(self.surface, str) or not self.surface:
            raise W06R03ContractError("generation option surface 非法")
        _identity(self.construction, where="generation construction")
        _identity(self.target_proposition, where="generation option target")
        if not isinstance(self.claim, PropertyClaim):
            raise W06R03ContractError("generation option claim 非法")
        if self.directionality != DIRECTION_FORWARD:
            raise W06R03ContractError("generation option direction 漂移")
        _role_fillers(self.role_fillers, where="generation option roles")
        _identity(self.context, where="generation option context")
        if not isinstance(self.source, SourceRef):
            raise W06R03ContractError("generation option source 非法")
        if self.uncertainty_units is not None and (
                type(self.uncertainty_units) is not int
                or self.uncertainty_units < 0):
            raise W06R03ContractError("generation option uncertainty 非法")
        _identity(self.language_branch, where="generation language branch")
        if not isinstance(self.authorization_key, LosslessIntegerKey):
            raise W06R03ContractError("generation authorization key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回 surface、结构、claim、来源与授权的完整键。"""
        values = [
            len(self.surface), *(ord(item) for item in self.surface),
            *pack_key(self.construction.stable_key()),
            *pack_key(self.target_proposition.stable_key()),
            *pack_key(self.claim.stable_key()),
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
class W06R03GenerationChoice:
    """PROPERTY 生成请求的全部合法 direct-fact option。"""

    request: W06R03GenerationRequest
    status: str
    options: tuple[W06R03GenerationOption, ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        """核验 READY 与 option 数、顺序和唯一性一致。"""
        if not isinstance(self.request, W06R03GenerationRequest):
            raise W06R03ContractError("generation choice request 非法")
        if self.status not in W06_R03_GENERATION_STATUSES:
            raise W06R03ContractError("generation choice status 非法")
        if (not isinstance(self.options, tuple)
                or any(not isinstance(item, W06R03GenerationOption)
                       for item in self.options)):
            raise W06R03ContractError("generation choice options 非法")
        normalized = tuple(sorted(
            self.options, key=W06R03GenerationOption.stable_key))
        if self.options != normalized or len(set(self.options)) != len(self.options):
            raise W06R03ContractError("generation options 未规范化")
        if (self.status == W06_R03_GENERATION_READY) != bool(self.options):
            raise W06R03ContractError("generation READY 与 option 数不一致")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R03ContractError("generation reason_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回请求、状态、options 与 reason 的完整键。"""
        values = [
            *pack_key(self.request.stable_key()),
            W06_R03_GENERATION_STATUSES.index(self.status) + 1,
            len(self.options),
        ]
        for item in self.options:
            values.extend(pack_key(item.stable_key()))
        values.extend(pack_key(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W06R03GenerationUse:
    """一个 generation option 对应的 direct PROPERTY Use。"""

    choice: W06R03GenerationChoice
    option: W06R03GenerationOption
    relation_uses: tuple[RelationClosureUse, ...]
    ref: GenerationChoiceUseRef

    def __post_init__(self) -> None:
        """要求 Use 精确归因被采用的 READY target Proposition。"""
        if (not isinstance(self.choice, W06R03GenerationChoice)
                or self.choice.status != W06_R03_GENERATION_READY
                or self.option not in self.choice.options):
            raise W06R03ContractError("generation Use 未采用 READY option")
        _relation_uses(
            self.relation_uses,
            (self.option.target_proposition,),
            where="generation Use",
        )
        if not isinstance(self.ref, GenerationChoiceUseRef):
            raise W06R03ContractError("generation Use ref 非法")


@dataclass(frozen=True)
class W06R03GenerationOutcome:
    """独立 exact PROPERTY 查询与 surface 结构 postcheck 结果。"""

    use: W06R03GenerationUse
    verdict: str
    ref: GenerationChoiceOutcomeRef
    authorization_current: bool
    relation_structure_preserved: bool
    source_scope_preserved: bool
    surface_structure_valid: bool
    property_query_status: str
    recovered_target: bool

    def __post_init__(self) -> None:
        """核验 verdict 必须等价于所有 postcheck 分维合取。"""
        if not isinstance(self.use, W06R03GenerationUse):
            raise W06R03ContractError("generation outcome Use 非法")
        if self.verdict not in W06_R03_OUTCOMES:
            raise W06R03ContractError("generation outcome verdict 非法")
        if not isinstance(self.ref, GenerationChoiceOutcomeRef):
            raise W06R03ContractError("generation outcome ref 非法")
        for name in (
                "authorization_current", "relation_structure_preserved",
                "source_scope_preserved", "surface_structure_valid",
                "recovered_target"):
            _strict_bool(getattr(self, name), where=name)
        if self.property_query_status not in W06_R03_REASONING_STATUSES:
            raise W06R03ContractError("generation property query status 非法")
        expected = all((
            self.authorization_current,
            self.relation_structure_preserved,
            self.source_scope_preserved,
            self.surface_structure_valid,
            self.property_query_status == W06_R03_REASONING_SUPPORTED,
            self.recovered_target,
        ))
        if (self.verdict == W06_R03_OUTCOME_SUPPORT) != expected:
            raise W06R03ContractError("generation verdict 未匹配分维 postcheck")


__all__ = [
    "W06R03ConsumerProtocol",
    "W06R03ContractError",
    "W06R03GenerationChoice",
    "W06R03GenerationOption",
    "W06R03GenerationOutcome",
    "W06R03GenerationRequest",
    "W06R03GenerationUse",
    "W06R03ReasoningOutcome",
    "W06R03ReasoningRequest",
    "W06R03ReasoningResolution",
    "W06R03ReasoningUse",
    "W06R03UnderstandingOutcome",
    "W06R03UnderstandingRequest",
    "W06R03UnderstandingResolution",
    "W06R03UnderstandingUse",
    "W06_R03_GENERATION_READY",
    "W06_R03_GENERATION_REJECTED",
    "W06_R03_GENERATION_UNKNOWN",
    "W06_R03_OUTCOME_REFUTE",
    "W06_R03_OUTCOME_SUPPORT",
    "W06_R03_REASONING_CONFLICT",
    "W06_R03_REASONING_REFUTED",
    "W06_R03_REASONING_STATUSES",
    "W06_R03_REASONING_SUPPORTED",
    "W06_R03_REASONING_UNRESOLVED",
    "W06_R03_SUBSTAGE",
    "W06_R03_UNDERSTANDING_CLARIFY",
    "W06_R03_UNDERSTANDING_CONFLICT",
    "W06_R03_UNDERSTANDING_MULTI",
    "W06_R03_UNDERSTANDING_STATUSES",
    "W06_R03_UNDERSTANDING_UNIQUE",
    "W06_R03_UNDERSTANDING_UNKNOWN",
    "pack_key",
]
