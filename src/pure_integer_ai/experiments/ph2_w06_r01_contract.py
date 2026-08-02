"""W06-R01 稳定 PURE_ALIAS/REFERS 的三向消费与有界验收合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionProposal,
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.alias_relation_runtime import AliasResolutionUse
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
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


W06_R01_SUBSTAGE = "PURE_ALIAS_REFERS"
W06_R01_RELATION_FAMILIES = ("PURE_ALIAS", "REFERS")

W06_R01_UNDERSTANDING_UNIQUE = "UNIQUE"
W06_R01_UNDERSTANDING_MULTI = "MULTI"
W06_R01_UNDERSTANDING_UNKNOWN = "UNKNOWN"
W06_R01_UNDERSTANDING_CONFLICT = "CONFLICT"
W06_R01_UNDERSTANDING_CLARIFY = "CLARIFY"
W06_R01_UNDERSTANDING_STATUSES = (
    W06_R01_UNDERSTANDING_UNIQUE,
    W06_R01_UNDERSTANDING_MULTI,
    W06_R01_UNDERSTANDING_UNKNOWN,
    W06_R01_UNDERSTANDING_CONFLICT,
    W06_R01_UNDERSTANDING_CLARIFY,
)

W06_R01_REASONING_SUPPORTED = "SUPPORTED"
W06_R01_REASONING_REFUTED = "REFUTED"
W06_R01_REASONING_CONFLICT = "CONFLICT"
W06_R01_REASONING_UNRESOLVED = "UNRESOLVED"
W06_R01_REASONING_STATUSES = (
    W06_R01_REASONING_SUPPORTED,
    W06_R01_REASONING_REFUTED,
    W06_R01_REASONING_CONFLICT,
    W06_R01_REASONING_UNRESOLVED,
)

W06_R01_GENERATION_READY = "READY"
W06_R01_GENERATION_UNKNOWN = "UNKNOWN"
W06_R01_GENERATION_REJECTED = "REJECTED"
W06_R01_GENERATION_STATUSES = (
    W06_R01_GENERATION_READY,
    W06_R01_GENERATION_UNKNOWN,
    W06_R01_GENERATION_REJECTED,
)

W06_R01_OUTCOME_SUPPORT = "SUPPORT"
W06_R01_OUTCOME_REFUTE = "REFUTE"
W06_R01_OUTCOMES = (
    W06_R01_OUTCOME_SUPPORT,
    W06_R01_OUTCOME_REFUTE,
)


class W06R01ContractError(ValueError):
    """W06-R01 请求、选择、归因或状态不满足冻结合同。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def _strict_bool(value: bool, *, where: str) -> bool:
    """拒绝以整数一或零冒充协议开关。"""
    if type(value) is not bool:
        raise W06R01ContractError(f"{where} 必须是严格 bool")
    return value


def _identity(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验一等对象身份。"""
    if not isinstance(value, ObjectIdentity):
        raise W06R01ContractError(f"{where} 必须是 ObjectIdentity")
    return value


def _identity_tuple(
        values: tuple[ObjectIdentity, ...], *, where: str,
        allow_empty: bool = False,
        ) -> tuple[ObjectIdentity, ...]:
    """核验规范排序、无重复的一等身份集合。"""
    if (not isinstance(values, tuple)
            or (not values and not allow_empty)
            or any(not isinstance(item, ObjectIdentity) for item in values)):
        raise W06R01ContractError(f"{where} identity 集合非法")
    normalized = tuple(sorted(values, key=ObjectIdentity.stable_key))
    if values != normalized or len(set(values)) != len(values):
        raise W06R01ContractError(f"{where} 必须规范排序且无重复")
    return values


def _evidence_keys(
        values: tuple[tuple[int, ...], ...], *, where: str,
        allow_empty: bool = True,
        ) -> tuple[tuple[int, ...], ...]:
    """核验 Evidence 稳定键集合，不接受布尔值或重复键。"""
    if (not isinstance(values, tuple)
            or (not values and not allow_empty)
            or any(not isinstance(item, tuple) or not item for item in values)):
        raise W06R01ContractError(f"{where} Evidence key 集合非法")
    for item in values:
        assert_int(*item, _where=where)
        if any(type(value) is not int for value in item):
            raise W06R01ContractError(f"{where} 必须使用严格整数")
    if values != tuple(sorted(values)) or len(set(values)) != len(values):
        raise W06R01ContractError(f"{where} 必须规范排序且无重复")
    return values


def _role_fillers(
        values: tuple[tuple[ObjectIdentity, ObjectIdentity], ...],
        *, where: str,
        ) -> tuple[tuple[ObjectIdentity, ObjectIdentity], ...]:
    """核验一个关系请求的 Role 到 filler 一一映射。"""
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(item, tuple) or len(item) != 2
                   for item in values)):
        raise W06R01ContractError(f"{where} Role/filler 集合非法")
    for role, filler in values:
        _identity(role, where=f"{where}.role")
        _identity(filler, where=f"{where}.filler")
        if role.object_kind != OBJECT_ROLE:
            raise W06R01ContractError(f"{where} Role 类型非法")
    normalized = tuple(sorted(values, key=lambda item: item[0].stable_key()))
    if values != normalized or len({item[0] for item in values}) != len(values):
        raise W06R01ContractError(f"{where} Role 必须规范排序且唯一")
    return values


def _relation_direction(relation_family: str, directionality: int) -> None:
    """冻结 PURE_ALIAS 对称与 REFERS 有向的语义边界。"""
    expected = {
        "PURE_ALIAS": DIRECTION_SYMMETRIC,
        "REFERS": DIRECTION_FORWARD,
    }
    if relation_family not in expected:
        raise W06R01ContractError("W06-R01 relation family 未注册")
    if type(directionality) is not int or directionality != expected[
            relation_family]:
        raise W06R01ContractError("W06-R01 relation direction 漂移")


@dataclass(frozen=True)
class W06R01ConsumerProtocol:
    """R01 关系桥与三向 consumer 的正交开关。"""

    alias_refers_bridge_connected: bool = True
    understanding_connected: bool = True
    reasoning_connected: bool = True
    generation_connected: bool = True
    direction_connected: bool = True
    source_scope_connected: bool = True
    postcheck_connected: bool = True

    def __post_init__(self) -> None:
        for name in (
                "alias_refers_bridge_connected",
                "understanding_connected",
                "reasoning_connected",
                "generation_connected",
                "direction_connected",
                "source_scope_connected",
                "postcheck_connected"):
            _strict_bool(getattr(self, name), where=name)

    def understanding_ready(self) -> bool:
        """返回理解侧是否可读取有向 active relation route。"""
        return all((
            self.alias_refers_bridge_connected,
            self.understanding_connected,
            self.direction_connected,
            self.source_scope_connected,
        ))

    def reasoning_ready(self) -> bool:
        """返回推理侧是否可消费 active relation Evidence。"""
        return all((
            self.alias_refers_bridge_connected,
            self.reasoning_connected,
            self.direction_connected,
            self.source_scope_connected,
        ))

    def generation_ready(self) -> bool:
        """返回生成侧是否可消费关系结构、方向和来源。"""
        return all((
            self.alias_refers_bridge_connected,
            self.generation_connected,
            self.direction_connected,
            self.source_scope_connected,
        ))

    def stable_key(self) -> tuple[int, ...]:
        """返回全部正交桥接开关。"""
        return tuple(int(getattr(self, name)) for name in (
            "alias_refers_bridge_connected",
            "understanding_connected",
            "reasoning_connected",
            "generation_connected",
            "direction_connected",
            "source_scope_connected",
            "postcheck_connected",
        ))


@dataclass(frozen=True)
class W06R01UnderstandingRequest:
    """不含 surface cue 或 expected label 的稳定指向查询。"""

    request_key: LosslessIntegerKey
    origin: ObjectIdentity
    target_object_kinds: tuple[int, ...]
    budget: AliasRouteSearchBudget
    allow_multiple: bool

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R01ContractError("understanding request_key 非法")
        _identity(self.origin, where="understanding origin")
        if (not isinstance(self.target_object_kinds, tuple)
                or not self.target_object_kinds):
            raise W06R01ContractError("understanding target kinds 为空")
        assert_int(*self.target_object_kinds, _where="W06R01 target kinds")
        if (any(type(item) is not int or item <= 0
                for item in self.target_object_kinds)
                or self.target_object_kinds != tuple(sorted(
                    set(self.target_object_kinds)))):
            raise W06R01ContractError("understanding target kinds 非法")
        if not isinstance(self.budget, AliasRouteSearchBudget):
            raise W06R01ContractError("understanding budget 非法")
        _strict_bool(self.allow_multiple, where="allow_multiple")

    def stable_key(self) -> tuple[int, ...]:
        """返回查询、方向目标类型和预算的完整键。"""
        return (
            *_pack(self.request_key.components),
            *_pack(self.origin.stable_key()),
            len(self.target_object_kinds),
            *self.target_object_kinds,
            *self.budget.stable_key(),
            int(self.allow_multiple),
        )


@dataclass(frozen=True)
class W06R01UnderstandingResolution:
    """稳定指向查询的五态结果、完整 route 和 Evidence 归因。"""

    request: W06R01UnderstandingRequest
    status: str
    options: tuple[ObjectIdentity, ...]
    selected: ObjectIdentity | None
    proposal: AliasResolutionProposal | None
    propositions: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.request, W06R01UnderstandingRequest):
            raise W06R01ContractError("understanding resolution request 非法")
        if self.status not in W06_R01_UNDERSTANDING_STATUSES:
            raise W06R01ContractError("understanding status 未注册")
        _identity_tuple(self.options, where="understanding options", allow_empty=True)
        _identity_tuple(
            self.propositions,
            where="understanding propositions",
            allow_empty=True,
        )
        _evidence_keys(self.evidence_keys, where="understanding evidence")
        if self.selected is not None:
            _identity(self.selected, where="understanding selected")
        if self.proposal is not None and not isinstance(
                self.proposal, AliasResolutionProposal):
            raise W06R01ContractError("understanding proposal 非法")
        if self.status == W06_R01_UNDERSTANDING_UNIQUE:
            if len(self.options) != 1 or self.selected != self.options[0]:
                raise W06R01ContractError("UNIQUE 必须采用唯一稳定指向")
        elif self.selected is not None:
            raise W06R01ContractError("非 UNIQUE 不得私选稳定指向")
        if self.status == W06_R01_UNDERSTANDING_MULTI and len(self.options) <= 1:
            raise W06R01ContractError("MULTI 必须保留多个稳定指向")
        if self.status in {
                W06_R01_UNDERSTANDING_UNKNOWN,
                W06_R01_UNDERSTANDING_CLARIFY,
                } and self.options:
            raise W06R01ContractError("UNKNOWN/CLARIFY 不得泄漏候选")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R01ContractError("understanding reason_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回五态、候选、route 和 Evidence 的完整稳定键。"""
        status = W06_R01_UNDERSTANDING_STATUSES.index(self.status) + 1
        values = [*_pack(self.request.stable_key()), status, len(self.options)]
        for item in self.options:
            values.extend(_pack(item.stable_key()))
        values.extend(_pack(
            () if self.selected is None else self.selected.stable_key()))
        values.extend(_pack(
            () if self.proposal is None else self.proposal.stable_key()))
        values.append(len(self.propositions))
        for item in self.propositions:
            values.extend(_pack(item.stable_key()))
        values.append(len(self.evidence_keys))
        for item in self.evidence_keys:
            values.extend(_pack(item))
        values.extend(_pack(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W06R01UnderstandingUse:
    """一次唯一稳定指向选择及其全部 active relation Use。"""

    resolution: W06R01UnderstandingResolution
    selection: ObjectIdentity
    alias_use: AliasResolutionUse
    use_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if (not isinstance(self.resolution, W06R01UnderstandingResolution)
                or self.resolution.status != W06_R01_UNDERSTANDING_UNIQUE
                or self.resolution.selected != self.selection):
            raise W06R01ContractError("understanding Use 未绑定 UNIQUE 选择")
        _identity(self.selection, where="understanding selection")
        if not isinstance(self.alias_use, AliasResolutionUse):
            raise W06R01ContractError("understanding alias Use 非法")
        if (self.alias_use.result.selected is None
                or self.alias_use.result.selected.value != self.selection):
            raise W06R01ContractError("understanding alias Use 选择漂移")
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise W06R01ContractError("understanding use_key 非法")


@dataclass(frozen=True)
class W06R01UnderstandingOutcome:
    """按当前 active route 重验 Understanding exact Use 的结果。"""

    use: W06R01UnderstandingUse
    verdict: str
    current_status: str
    outcome_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.use, W06R01UnderstandingUse):
            raise W06R01ContractError("understanding outcome Use 非法")
        if self.verdict not in W06_R01_OUTCOMES:
            raise W06R01ContractError("understanding outcome verdict 非法")
        if self.current_status not in W06_R01_UNDERSTANDING_STATUSES:
            raise W06R01ContractError("understanding outcome current status 非法")
        if not isinstance(self.outcome_key, LosslessIntegerKey):
            raise W06R01ContractError("understanding outcome key 非法")


@dataclass(frozen=True)
class W06R01ReasoningRequest:
    """以 typed relation family、方向端点查询当前关系结论。"""

    request_key: LosslessIntegerKey
    relation_family: str
    source: ObjectIdentity
    target: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R01ContractError("reasoning request_key 非法")
        _relation_direction(
            self.relation_family,
            DIRECTION_SYMMETRIC
            if self.relation_family == "PURE_ALIAS" else DIRECTION_FORWARD,
        )
        _identity(self.source, where="reasoning source")
        _identity(self.target, where="reasoning target")
        if self.source == self.target:
            raise W06R01ContractError("reasoning relation 不得自环")

    def stable_key(self) -> tuple[int, ...]:
        """返回 relation family 与有向端点的完整键。"""
        family = W06_R01_RELATION_FAMILIES.index(self.relation_family) + 1
        return (
            *_pack(self.request_key.components),
            family,
            *_pack(self.source.stable_key()),
            *_pack(self.target.stable_key()),
        )


@dataclass(frozen=True)
class W06R01ReasoningResolution:
    """关系结论及其命题和 Evidence 集，不以排序私选。"""

    request: W06R01ReasoningRequest
    status: str
    propositions: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.request, W06R01ReasoningRequest):
            raise W06R01ContractError("reasoning resolution request 非法")
        if self.status not in W06_R01_REASONING_STATUSES:
            raise W06R01ContractError("reasoning status 未注册")
        _identity_tuple(
            self.propositions,
            where="reasoning propositions",
            allow_empty=True,
        )
        _evidence_keys(self.evidence_keys, where="reasoning evidence")
        if (self.status == W06_R01_REASONING_UNRESOLVED
                and self.propositions):
            raise W06R01ContractError("UNRESOLVED 不得泄漏 relation candidate")
        if (self.status != W06_R01_REASONING_UNRESOLVED
                and not self.propositions):
            raise W06R01ContractError("已判 relation 必须保留命题")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R01ContractError("reasoning reason_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回结论、命题和 Evidence 的完整稳定键。"""
        status = W06_R01_REASONING_STATUSES.index(self.status) + 1
        values = [*_pack(self.request.stable_key()), status, len(self.propositions)]
        for item in self.propositions:
            values.extend(_pack(item.stable_key()))
        values.append(len(self.evidence_keys))
        for item in self.evidence_keys:
            values.extend(_pack(item))
        values.extend(_pack(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W06R01ReasoningUse:
    """一次 SUPPORTED 关系结论的全部 exact Relation Use。"""

    resolution: W06R01ReasoningResolution
    relation_uses: tuple[RelationClosureUse, ...]
    use_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if (not isinstance(self.resolution, W06R01ReasoningResolution)
                or self.resolution.status != W06_R01_REASONING_SUPPORTED):
            raise W06R01ContractError("reasoning Use 只能采用 SUPPORTED 结论")
        if (not isinstance(self.relation_uses, tuple)
                or not self.relation_uses
                or any(not isinstance(item, RelationClosureUse)
                       for item in self.relation_uses)):
            raise W06R01ContractError("reasoning relation Uses 非法")
        if {item.proposition for item in self.relation_uses} != set(
                self.resolution.propositions):
            raise W06R01ContractError("reasoning Use 未精确覆盖命题")
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise W06R01ContractError("reasoning use_key 非法")


@dataclass(frozen=True)
class W06R01ReasoningOutcome:
    """按当前 lifecycle 重验 reasoning exact Use 的结果。"""

    use: W06R01ReasoningUse
    verdict: str
    current_status: str
    outcome_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.use, W06R01ReasoningUse):
            raise W06R01ContractError("reasoning outcome Use 非法")
        if self.verdict not in W06_R01_OUTCOMES:
            raise W06R01ContractError("reasoning outcome verdict 非法")
        if self.current_status not in W06_R01_REASONING_STATUSES:
            raise W06R01ContractError("reasoning outcome current status 非法")
        if not isinstance(self.outcome_key, LosslessIntegerKey):
            raise W06R01ContractError("reasoning outcome key 非法")


@dataclass(frozen=True)
class W06R01GenerationRequest:
    """以 active typed relation 为目标且不含 expected surface 的生成请求。"""

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
            raise W06R01ContractError("generation request_key 非法")
        _identity(self.target_proposition, where="generation target proposition")
        if self.target_proposition.object_kind != OBJECT_PROPOSITION:
            raise W06R01ContractError("generation target 必须是 Proposition")
        _relation_direction(self.relation_family, self.directionality)
        _role_fillers(self.role_fillers, where="generation role_fillers")
        _identity(self.context, where="generation context")
        if self.context.object_kind != OBJECT_CONTEXT_SCOPE:
            raise W06R01ContractError("generation context 必须是 ContextScope")
        if not isinstance(self.source, SourceRef):
            raise W06R01ContractError("generation source 非法")
        if self.uncertainty_units is not None:
            assert_int(self.uncertainty_units, _where="generation uncertainty")
            if type(self.uncertainty_units) is not int or self.uncertainty_units < 0:
                raise W06R01ContractError("generation uncertainty 非负整数")
        if not isinstance(self.constraints, GenerationExpressionConstraints):
            raise W06R01ContractError("generation constraints 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回关系结构、方向、scope、source 和语言约束键。"""
        family = W06_R01_RELATION_FAMILIES.index(self.relation_family) + 1
        values = [
            *_pack(self.request_key.components),
            *_pack(self.target_proposition.stable_key()),
            family,
            self.directionality,
            len(self.role_fillers),
        ]
        for role, filler in self.role_fillers:
            values.extend((_pack(role.stable_key())))
            values.extend((_pack(filler.stable_key())))
        values.extend((
            *_pack(self.context.stable_key()),
            *_pack(self.source.stable_key()),
            -1 if self.uncertainty_units is None else self.uncertainty_units,
            *_pack(self.constraints.stable_key()),
        ))
        return tuple(values)


@dataclass(frozen=True)
class W06R01GenerationOption:
    """一个由 active relation fact 授权的来源化表面实现。"""

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
            raise W06R01ContractError("generation option surface 非法")
        _identity(self.construction, where="generation construction")
        _identity(self.target_proposition, where="generation option target")
        _relation_direction(self.relation_family, self.directionality)
        _role_fillers(self.role_fillers, where="generation option role_fillers")
        _identity(self.context, where="generation option context")
        if not isinstance(self.source, SourceRef):
            raise W06R01ContractError("generation option source 非法")
        if self.uncertainty_units is not None and (
                type(self.uncertainty_units) is not int
                or self.uncertainty_units < 0):
            raise W06R01ContractError("generation option uncertainty 非法")
        _identity(self.language_branch, where="generation language branch")
        if not isinstance(self.authorization_key, LosslessIntegerKey):
            raise W06R01ContractError("generation authorization key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回表面、结构和 active authorization 的完整键。"""
        family = W06_R01_RELATION_FAMILIES.index(self.relation_family) + 1
        values = [
            len(self.surface),
            *(ord(character) for character in self.surface),
            *_pack(self.construction.stable_key()),
            *_pack(self.target_proposition.stable_key()),
            family,
            self.directionality,
            len(self.role_fillers),
        ]
        for role, filler in self.role_fillers:
            values.extend(_pack(role.stable_key()))
            values.extend(_pack(filler.stable_key()))
        values.extend((
            *_pack(self.context.stable_key()),
            *_pack(self.source.stable_key()),
            -1 if self.uncertainty_units is None else self.uncertainty_units,
            *_pack(self.language_branch.stable_key()),
            *_pack(self.authorization_key.components),
        ))
        return tuple(values)


@dataclass(frozen=True)
class W06R01GenerationChoice:
    """生成关系结构的全部合法 option，不比较唯一 expected 字符串。"""

    request: W06R01GenerationRequest
    status: str
    options: tuple[W06R01GenerationOption, ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.request, W06R01GenerationRequest):
            raise W06R01ContractError("generation choice request 非法")
        if self.status not in W06_R01_GENERATION_STATUSES:
            raise W06R01ContractError("generation choice status 非法")
        if (not isinstance(self.options, tuple)
                or any(not isinstance(item, W06R01GenerationOption)
                       for item in self.options)):
            raise W06R01ContractError("generation choice options 非法")
        if (self.options != tuple(sorted(
                self.options, key=W06R01GenerationOption.stable_key))
                or len({item.stable_key() for item in self.options})
                != len(self.options)):
            raise W06R01ContractError("generation options 未规范化")
        if (self.status == W06_R01_GENERATION_READY) != bool(self.options):
            raise W06R01ContractError("generation READY 与 option 数不一致")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R01ContractError("generation reason_key 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回请求、状态和全部合法实现的稳定键。"""
        status = W06_R01_GENERATION_STATUSES.index(self.status) + 1
        values = [*_pack(self.request.stable_key()), status, len(self.options)]
        for item in self.options:
            values.extend(_pack(item.stable_key()))
        values.extend(_pack(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W06R01GenerationUse:
    """一个采用 option 对应的 relation truth 和 generation exact Use。"""

    choice: W06R01GenerationChoice
    option: W06R01GenerationOption
    relation_use: RelationClosureUse
    ref: GenerationChoiceUseRef

    def __post_init__(self) -> None:
        if (not isinstance(self.choice, W06R01GenerationChoice)
                or self.choice.status != W06_R01_GENERATION_READY
                or self.option not in self.choice.options):
            raise W06R01ContractError("generation Use 未采用 READY option")
        if not isinstance(self.relation_use, RelationClosureUse):
            raise W06R01ContractError("generation relation Use 非法")
        if self.relation_use.proposition != self.option.target_proposition:
            raise W06R01ContractError("generation Use 未精确归因目标关系")
        if not isinstance(self.ref, GenerationChoiceUseRef):
            raise W06R01ContractError("generation Use ref 非法")


@dataclass(frozen=True)
class W06R01GenerationOutcome:
    """独立 route 与 surface 结构 postcheck 的分维结果。"""

    use: W06R01GenerationUse
    verdict: str
    ref: GenerationChoiceOutcomeRef
    authorization_current: bool
    relation_structure_preserved: bool
    source_scope_preserved: bool
    surface_structure_valid: bool
    understanding_status: str
    recovered_target: bool

    def __post_init__(self) -> None:
        if not isinstance(self.use, W06R01GenerationUse):
            raise W06R01ContractError("generation outcome Use 非法")
        if self.verdict not in W06_R01_OUTCOMES:
            raise W06R01ContractError("generation outcome verdict 非法")
        if not isinstance(self.ref, GenerationChoiceOutcomeRef):
            raise W06R01ContractError("generation outcome ref 非法")
        for name in (
                "authorization_current",
                "relation_structure_preserved",
                "source_scope_preserved",
                "surface_structure_valid",
                "recovered_target"):
            _strict_bool(getattr(self, name), where=name)
        if self.understanding_status not in W06_R01_UNDERSTANDING_STATUSES:
            raise W06R01ContractError("generation postcheck understanding 非法")
        expected = all((
            self.authorization_current,
            self.relation_structure_preserved,
            self.source_scope_preserved,
            self.surface_structure_valid,
            self.understanding_status == W06_R01_UNDERSTANDING_UNIQUE,
            self.recovered_target,
        ))
        if (self.verdict == W06_R01_OUTCOME_SUPPORT) != expected:
            raise W06R01ContractError("generation verdict 未匹配分维 postcheck")


__all__ = [
    "W06R01ConsumerProtocol",
    "W06R01ContractError",
    "W06R01GenerationChoice",
    "W06R01GenerationOption",
    "W06R01GenerationOutcome",
    "W06R01GenerationRequest",
    "W06R01GenerationUse",
    "W06R01ReasoningOutcome",
    "W06R01ReasoningRequest",
    "W06R01ReasoningResolution",
    "W06R01ReasoningUse",
    "W06R01UnderstandingOutcome",
    "W06R01UnderstandingRequest",
    "W06R01UnderstandingResolution",
    "W06R01UnderstandingUse",
    "W06_R01_GENERATION_READY",
    "W06_R01_GENERATION_REJECTED",
    "W06_R01_GENERATION_UNKNOWN",
    "W06_R01_OUTCOME_REFUTE",
    "W06_R01_OUTCOME_SUPPORT",
    "W06_R01_REASONING_CONFLICT",
    "W06_R01_REASONING_REFUTED",
    "W06_R01_REASONING_SUPPORTED",
    "W06_R01_REASONING_UNRESOLVED",
    "W06_R01_RELATION_FAMILIES",
    "W06_R01_SUBSTAGE",
    "W06_R01_UNDERSTANDING_CLARIFY",
    "W06_R01_UNDERSTANDING_CONFLICT",
    "W06_R01_UNDERSTANDING_MULTI",
    "W06_R01_UNDERSTANDING_UNIQUE",
    "W06_R01_UNDERSTANDING_UNKNOWN",
]
