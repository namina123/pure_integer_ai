"""W-07 public logic consumers 的严格请求、Use 与 postcheck 合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_PROPOSITION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_candidate import (
    LogicOperatorAdoption,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvaluation
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w07_contract import W07_SUBSTAGE_ORDER


W07_LOGIC_CONSUMERS = ("UNDERSTANDING", "REASONING", "GENERATION")
W07_LOGIC_STATUSES = (
    "SUPPORTED", "REFUTED", "UNKNOWN", "CONFLICT", "NO_ADOPTION")
W07_LOGIC_OUTCOMES = ("SUPPORT", "REFUTE")
W07_GENERATION_STATUSES = ("READY", "UNKNOWN", "REJECTED")


class W07LogicContractError(RuntimeError):
    """W-07 public logic facade 发现 scope、Use 或结构合同漂移。"""


def pack_key(value: tuple[int, ...]) -> tuple[int, ...]:
    if (not isinstance(value, tuple)
            or any(type(item) is not int for item in value)):
        raise W07LogicContractError("W-07 logic key 必须是严格整数 tuple")
    return len(value), *value


def _identity(value, *, where: str) -> ObjectIdentity:
    if not isinstance(value, ObjectIdentity):
        raise W07LogicContractError(f"{where} 必须是一等对象")
    return value


def _strict_key(value, *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W07LogicContractError(f"{where} 必须是非空严格整数 tuple")
    return value


@dataclass(frozen=True)
class W07LogicConsumerProtocol:
    """U/R/G、postcheck 与目标 operator 的独立连接位。"""

    enabled_substages: tuple[str, ...]
    disabled_substages: tuple[str, ...] = ()
    understanding_connected: bool = True
    reasoning_connected: bool = True
    generation_connected: bool = True
    postcheck_connected: bool = True
    disabled_operator_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.enabled_substages, tuple)
                or not self.enabled_substages
                or len(set(self.enabled_substages))
                != len(self.enabled_substages)):
            raise W07LogicContractError("W-07 enabled substages 非法")
        expected = tuple(
            item for item in W07_SUBSTAGE_ORDER
            if item in set(self.enabled_substages))
        if self.enabled_substages != expected:
            raise W07LogicContractError("W-07 enabled substages 未保持冻结顺序")
        if (not isinstance(self.disabled_substages, tuple)
                or any(item not in self.enabled_substages
                       for item in self.disabled_substages)
                or len(set(self.disabled_substages))
                != len(self.disabled_substages)):
            raise W07LogicContractError("W-07 disabled substages 非法")
        for name in (
                "understanding_connected", "reasoning_connected",
                "generation_connected", "postcheck_connected"):
            if type(getattr(self, name)) is not bool:
                raise W07LogicContractError(f"{name} 必须是严格 bool")
        if (not isinstance(self.disabled_operator_families, tuple)
                or any(not isinstance(item, str) or not item
                       for item in self.disabled_operator_families)
                or len(set(self.disabled_operator_families))
                != len(self.disabled_operator_families)):
            raise W07LogicContractError("disabled operator families 非法")

    def connected(self, consumer: str) -> bool:
        if consumer not in W07_LOGIC_CONSUMERS:
            raise W07LogicContractError("W-07 logic consumer 未注册")
        return {
            "UNDERSTANDING": self.understanding_connected,
            "REASONING": self.reasoning_connected,
            "GENERATION": self.generation_connected,
        }[consumer]

    def stable_key(self) -> tuple[int, ...]:
        return (
            *(W07_SUBSTAGE_ORDER.index(item) + 1
              for item in self.enabled_substages),
            0,
            *(W07_SUBSTAGE_ORDER.index(item) + 1
              for item in self.disabled_substages),
            0,
            int(self.understanding_connected),
            int(self.reasoning_connected),
            int(self.generation_connected),
            int(self.postcheck_connected),
            *(value for item in self.disabled_operator_families
              for value in (len(item), *(ord(char) for char in item))),
        )


@dataclass(frozen=True)
class W07LogicBudget:
    """每次 consumer 执行的递归、分支和步骤硬上限。"""

    max_depth: int = 8
    max_branches: int = 32
    max_steps: int = 128
    max_resolver_calls: int = 32

    def __post_init__(self) -> None:
        for name in (
                "max_depth", "max_branches", "max_steps",
                "max_resolver_calls"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise W07LogicContractError(f"{name} 必须是正严格整数")

    def stable_key(self) -> tuple[int, ...]:
        return (
            self.max_depth, self.max_branches, self.max_steps,
            self.max_resolver_calls,
        )


@dataclass(frozen=True)
class W07LogicRequest:
    """只携 target/source/scope，不携 expected state、label 或 surface。"""

    request_key: LosslessIntegerKey
    substage: str
    target_proposition: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    budget: W07LogicBudget = W07LogicBudget()

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W07LogicContractError("logic request key 非法")
        if self.substage not in W07_SUBSTAGE_ORDER:
            raise W07LogicContractError("logic request substage 未注册")
        _identity(self.target_proposition, where="logic target")
        if self.target_proposition.object_kind != OBJECT_PROPOSITION:
            raise W07LogicContractError("logic target 必须是 Proposition")
        if not isinstance(self.source, SourceRef):
            raise W07LogicContractError("logic request source 非法")
        if not isinstance(self.scope, ScopeIdentity):
            raise W07LogicContractError("logic request scope 非法")
        if self.scope.source != self.source:
            raise W07LogicContractError("logic request source/scope 漂移")
        if not isinstance(self.budget, W07LogicBudget):
            raise W07LogicContractError("logic request budget 非法")

    def stable_key(self) -> tuple[int, ...]:
        return (
            *pack_key(self.request_key.components),
            W07_SUBSTAGE_ORDER.index(self.substage) + 1,
            *pack_key(self.target_proposition.stable_key()),
            *pack_key(self.source.stable_key()),
            *pack_key(self.scope.stable_key()),
            *self.budget.stable_key(),
        )


@dataclass(frozen=True)
class W07LogicExecution:
    """一次真实 S-04 结果与 operator/content premise 分账。"""

    request: W07LogicRequest
    evaluation: LogicEvaluation
    operator_adoptions: tuple[LogicOperatorAdoption, ...]
    operator_premise_keys: tuple[tuple[int, ...], ...]
    content_premise_keys: tuple[tuple[int, ...], ...]
    executed_structures: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, W07LogicRequest):
            raise W07LogicContractError("logic execution request 非法")
        if (not isinstance(self.evaluation, LogicEvaluation)
                or self.evaluation.proposition.template
                != self.request.target_proposition):
            raise W07LogicContractError("logic execution target 漂移")
        if (not self.operator_adoptions
                or any(not isinstance(item, LogicOperatorAdoption)
                       for item in self.operator_adoptions)):
            raise W07LogicContractError("logic execution 缺 learned adoption")
        for name in ("operator_premise_keys", "content_premise_keys"):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, tuple) or not item
                           or any(type(value) is not int for value in item)
                           for item in values)
                    or values != tuple(sorted(set(values)))):
                raise W07LogicContractError(f"{name} 未规范化")
        if set(self.operator_premise_keys) & set(self.content_premise_keys):
            raise W07LogicContractError("operator/content Evidence 不得混账")
        if (not self.executed_structures
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.executed_structures)
                or self.executed_structures != tuple(sorted(
                    set(self.executed_structures),
                    key=ObjectIdentity.stable_key))):
            raise W07LogicContractError("executed structures 未规范化")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *pack_key(self.request.stable_key()),
            *pack_key(self.evaluation.stable_key()),
            len(self.operator_adoptions),
        ]
        for item in self.operator_adoptions:
            values.extend(pack_key(item.stable_key()))
        for group in (self.operator_premise_keys, self.content_premise_keys):
            values.append(len(group))
            for item in group:
                values.extend(pack_key(item))
        values.append(len(self.executed_structures))
        for item in self.executed_structures:
            values.extend(pack_key(item.stable_key()))
        return tuple(values)


@dataclass(frozen=True)
class W07LogicResolution:
    """Understanding 或 Reasoning 的独立四态 resolution。"""

    consumer: str
    request: W07LogicRequest
    status: str
    execution: W07LogicExecution | None
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if self.consumer not in W07_LOGIC_CONSUMERS[:2]:
            raise W07LogicContractError("logic resolution consumer 非法")
        if not isinstance(self.request, W07LogicRequest):
            raise W07LogicContractError("logic resolution request 非法")
        if self.status not in W07_LOGIC_STATUSES:
            raise W07LogicContractError("logic resolution status 非法")
        if (self.status == "NO_ADOPTION") != (self.execution is None):
            raise W07LogicContractError("NO_ADOPTION 与 execution 不一致")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W07LogicContractError("logic resolution reason 非法")

    def stable_key(self) -> tuple[int, ...]:
        execution = () if self.execution is None else self.execution.stable_key()
        return (
            W07_LOGIC_CONSUMERS.index(self.consumer) + 1,
            *pack_key(self.request.stable_key()),
            W07_LOGIC_STATUSES.index(self.status) + 1,
            *pack_key(execution),
            *pack_key(self.reason_key.components),
        )


@dataclass(frozen=True)
class W07LogicUse:
    """一个 consumer 独占的 exact execution Use。"""

    resolution: W07LogicResolution
    use_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if (not isinstance(self.resolution, W07LogicResolution)
                or self.resolution.execution is None):
            raise W07LogicContractError("logic Use 必须来自已执行 resolution")
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise W07LogicContractError("logic use key 非法")


@dataclass(frozen=True)
class W07LogicOutcome:
    """按 current adoption/source/scope 重验历史 Use。"""

    use: W07LogicUse
    verdict: str
    current_status: str
    outcome_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.use, W07LogicUse):
            raise W07LogicContractError("logic outcome Use 非法")
        if self.verdict not in W07_LOGIC_OUTCOMES:
            raise W07LogicContractError("logic outcome verdict 非法")
        if self.current_status not in W07_LOGIC_STATUSES:
            raise W07LogicContractError("logic outcome status 非法")
        if not isinstance(self.outcome_key, LosslessIntegerKey):
            raise W07LogicContractError("logic outcome key 非法")


@dataclass(frozen=True)
class W07LogicGenerationRequest:
    """生成请求只声明 structural target 和表达约束。"""

    request_key: LosslessIntegerKey
    logic_request: W07LogicRequest
    constraints: GenerationExpressionConstraints

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W07LogicContractError("generation request key 非法")
        if not isinstance(self.logic_request, W07LogicRequest):
            raise W07LogicContractError("generation logic request 非法")
        if not isinstance(self.constraints, GenerationExpressionConstraints):
            raise W07LogicContractError("generation constraints 非法")

    def stable_key(self) -> tuple[int, ...]:
        return (
            *pack_key(self.request_key.components),
            *pack_key(self.logic_request.stable_key()),
            *pack_key(self.constraints.stable_key()),
        )


@dataclass(frozen=True)
class W07LogicGenerationOption:
    """保留完整 tree/Role/source/scope/four-state 的可生成 option。"""

    surface: str
    target_proposition: ObjectIdentity
    operator_families: tuple[str, ...]
    structure_tree_key: tuple[int, ...]
    role_tree_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    state_key: tuple[int, int]
    operator_premise_keys: tuple[tuple[int, ...], ...]
    content_premise_keys: tuple[tuple[int, ...], ...]
    language_branch: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface:
            raise W07LogicContractError("generation surface 非法")
        _identity(self.target_proposition, where="generation target")
        if (not self.operator_families
                or any(not isinstance(item, str) or not item
                       for item in self.operator_families)):
            raise W07LogicContractError("generation operator families 非法")
        _strict_key(self.structure_tree_key, where="structure tree")
        _strict_key(self.role_tree_key, where="role tree")
        if not isinstance(self.source, SourceRef):
            raise W07LogicContractError("generation source 非法")
        if not isinstance(self.scope, ScopeIdentity):
            raise W07LogicContractError("generation scope 非法")
        if (not isinstance(self.state_key, tuple) or len(self.state_key) != 2
                or any(type(item) is not int or item not in {0, 1}
                       for item in self.state_key)):
            raise W07LogicContractError("generation four-state 非法")
        _identity(self.language_branch, where="generation language branch")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            len(self.surface), *(ord(item) for item in self.surface),
            *pack_key(self.target_proposition.stable_key()),
            len(self.operator_families),
        ]
        for item in self.operator_families:
            values.extend((len(item), *(ord(char) for char in item)))
        values.extend((
            *pack_key(self.structure_tree_key),
            *pack_key(self.role_tree_key),
            *pack_key(self.source.stable_key()),
            *pack_key(self.scope.stable_key()),
            *self.state_key,
        ))
        for group in (self.operator_premise_keys, self.content_premise_keys):
            values.append(len(group))
            for item in group:
                values.extend(pack_key(item))
        values.extend(pack_key(self.language_branch.stable_key()))
        return tuple(values)


@dataclass(frozen=True)
class W07LogicGenerationChoice:
    request: W07LogicGenerationRequest
    status: str
    options: tuple[W07LogicGenerationOption, ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.request, W07LogicGenerationRequest):
            raise W07LogicContractError("generation choice request 非法")
        if self.status not in W07_GENERATION_STATUSES:
            raise W07LogicContractError("generation choice status 非法")
        if (any(not isinstance(item, W07LogicGenerationOption)
                for item in self.options)
                or self.options != tuple(sorted(
                    self.options, key=W07LogicGenerationOption.stable_key))):
            raise W07LogicContractError("generation options 未规范化")
        if (self.status == "READY") != bool(self.options):
            raise W07LogicContractError("generation READY/options 不一致")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W07LogicContractError("generation reason 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *pack_key(self.request.stable_key()),
            W07_GENERATION_STATUSES.index(self.status) + 1,
            len(self.options),
        ]
        for item in self.options:
            values.extend(pack_key(item.stable_key()))
        values.extend(pack_key(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W07LogicGenerationUse:
    choice: W07LogicGenerationChoice
    option: W07LogicGenerationOption
    execution: W07LogicExecution
    ref: GenerationChoiceUseRef

    def __post_init__(self) -> None:
        if (not isinstance(self.choice, W07LogicGenerationChoice)
                or self.choice.status != "READY"
                or self.option not in self.choice.options):
            raise W07LogicContractError("generation Use 未采用 READY option")
        if not isinstance(self.execution, W07LogicExecution):
            raise W07LogicContractError("generation Use execution 非法")
        if not isinstance(self.ref, GenerationChoiceUseRef):
            raise W07LogicContractError("generation Use ref 非法")


@dataclass(frozen=True)
class W07LogicGenerationOutcome:
    use: W07LogicGenerationUse
    verdict: str
    ref: GenerationChoiceOutcomeRef
    adoption_current: bool
    structure_preserved: bool
    role_order_preserved: bool
    state_preserved: bool
    source_scope_preserved: bool
    surface_valid: bool
    recovered_target: bool

    def __post_init__(self) -> None:
        if not isinstance(self.use, W07LogicGenerationUse):
            raise W07LogicContractError("generation outcome Use 非法")
        if self.verdict not in W07_LOGIC_OUTCOMES:
            raise W07LogicContractError("generation outcome verdict 非法")
        if not isinstance(self.ref, GenerationChoiceOutcomeRef):
            raise W07LogicContractError("generation outcome ref 非法")
        checks = (
            self.adoption_current, self.structure_preserved,
            self.role_order_preserved, self.state_preserved,
            self.source_scope_preserved, self.surface_valid,
            self.recovered_target,
        )
        if any(type(item) is not bool for item in checks):
            raise W07LogicContractError("generation postcheck 必须是严格 bool")
        if (self.verdict == "SUPPORT") != all(checks):
            raise W07LogicContractError("generation verdict 与 postcheck 不一致")


__all__ = [name for name in globals() if name.startswith("W07")]
