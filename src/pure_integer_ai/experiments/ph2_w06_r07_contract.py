"""W06-R07 direct CAUSES 查询、独立 witness 与生成消费合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.causal_execution import (
    CausalEndpointProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
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
from pure_integer_ai.experiments.causal_relation_runtime import (
    CausalIndependentWitness,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_r07_endpoint_projection import (
    W06_R07_RUNTIME_NAMESPACE,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureUse,
)


W06_R07_SUBSTAGE = "CAUSES"
W06_R07_CONSUMERS = ("UNDERSTANDING", "REASONING")

W06_R07_SUPPORTED = "SUPPORTED"
W06_R07_REFUTED = "REFUTED"
W06_R07_CONFLICT = "CONFLICT"
W06_R07_UNKNOWN = "UNKNOWN"
W06_R07_QUERY_STATUSES = (
    W06_R07_SUPPORTED,
    W06_R07_REFUTED,
    W06_R07_CONFLICT,
    W06_R07_UNKNOWN,
)
W06_R07_GENERATION_READY = "READY"
W06_R07_GENERATION_UNKNOWN = "UNKNOWN"
W06_R07_GENERATION_REJECTED = "REJECTED"
W06_R07_GENERATION_STATUSES = (
    W06_R07_GENERATION_READY,
    W06_R07_GENERATION_UNKNOWN,
    W06_R07_GENERATION_REJECTED,
)
W06_R07_OUTCOME_SUPPORT = "SUPPORT"
W06_R07_OUTCOME_REFUTE = "REFUTE"
W06_R07_OUTCOMES = (W06_R07_OUTCOME_SUPPORT, W06_R07_OUTCOME_REFUTE)
_ENDPOINT_KINDS = frozenset({OBJECT_EVENT, OBJECT_PROPOSITION})


class W06R07ContractError(ValueError):
    """R07 请求、witness、Use 或生成归因不满足冻结合同。"""


class W06R07BudgetExceeded(RuntimeError):
    """R07 查询在产生部分结果前耗尽显式预算。"""


def pack_key(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _strict_bool(value: bool, *, where: str) -> bool:
    if type(value) is not bool:
        raise W06R07ContractError(f"{where} 必须是严格 bool")
    return value


def _identity(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    if not isinstance(value, ObjectIdentity):
        raise W06R07ContractError(f"{where} 必须是 ObjectIdentity")
    return value


def _identities(
        values: tuple[ObjectIdentity, ...], *, where: str,
        allow_empty: bool = False,
        ) -> tuple[ObjectIdentity, ...]:
    if (not isinstance(values, tuple)
            or (not values and not allow_empty)
            or any(not isinstance(item, ObjectIdentity) for item in values)):
        raise W06R07ContractError(f"{where} identity 集合非法")
    if (values != tuple(sorted(values, key=ObjectIdentity.stable_key))
            or len(set(values)) != len(values)):
        raise W06R07ContractError(f"{where} 必须规范排序且无重复")
    return values


def _sources(
        values: tuple[SourceRef, ...], *, where: str,
        ) -> tuple[SourceRef, ...]:
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(item, SourceRef) for item in values)):
        raise W06R07ContractError(f"{where} source 集合非法")
    if (values != tuple(sorted(values, key=SourceRef.stable_key))
            or len(set(values)) != len(values)):
        raise W06R07ContractError(f"{where} 必须规范排序且无重复")
    return values


def _evidence_keys(
        values: tuple[tuple[int, ...], ...], *, where: str,
        ) -> tuple[tuple[int, ...], ...]:
    if not isinstance(values, tuple):
        raise W06R07ContractError(f"{where} 必须是 tuple")
    for item in values:
        if not isinstance(item, tuple) or not item:
            raise W06R07ContractError(f"{where} key 非法")
        assert_int(*item, _where=where)
        if any(type(value) is not int for value in item):
            raise W06R07ContractError(f"{where} 必须使用严格整数")
    if values != tuple(sorted(values)) or len(set(values)) != len(values):
        raise W06R07ContractError(f"{where} 必须规范排序且无重复")
    return values


def _role_fillers(
        values: tuple[tuple[ObjectIdentity, ObjectIdentity], ...],
        *, where: str,
        ) -> tuple[tuple[ObjectIdentity, ObjectIdentity], ...]:
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(item, tuple) or len(item) != 2
                   for item in values)):
        raise W06R07ContractError(f"{where} Role/filler 集合非法")
    for role, filler in values:
        _identity(role, where=f"{where}.role")
        _identity(filler, where=f"{where}.filler")
        if role.object_kind != OBJECT_ROLE:
            raise W06R07ContractError(f"{where} Role 类型非法")
    if (values != tuple(sorted(values, key=lambda item: item[0].stable_key()))
            or len({item[0] for item in values}) != len(values)):
        raise W06R07ContractError(f"{where} Role 必须规范排序且唯一")
    return values


def _endpoints(cause: ObjectIdentity, effect: ObjectIdentity) -> None:
    _identity(cause, where="causal cause")
    _identity(effect, where="causal effect")
    if cause.object_kind not in _ENDPOINT_KINDS or effect.object_kind not in (
            _ENDPOINT_KINDS):
        raise W06R07ContractError("causal 端点必须是 Event/Proposition")
    if cause == effect:
        raise W06R07ContractError("causal 查询不得自环")


def _endpoint_protocol_key(value: CausalEndpointProtocol) -> tuple[int, ...]:
    if not isinstance(value, CausalEndpointProtocol):
        raise W06R07ContractError("causal endpoint protocol 类型非法")
    return (
        *pack_key(value.relation.stable_key()),
        *pack_key(value.cause_role.stable_key()),
        *pack_key(value.effect_role.stable_key()),
        *pack_key(value.execution_instruction.stable_key()),
    )


def _witness_key(value: CausalIndependentWitness) -> tuple[int, ...]:
    if not isinstance(value, CausalIndependentWitness):
        raise W06R07ContractError("causal witness 类型非法")
    values = [
        value.stance,
        *pack_key(value.verifier_source.stable_key()),
        len(value.input_objects),
    ]
    for item in value.input_objects:
        values.extend(pack_key(item.stable_key()))
    values.extend(pack_key(value.trace))
    return tuple(values)


@dataclass(frozen=True)
class W06R07Budget:
    max_candidates: int
    max_evidence: int
    max_witness_inputs: int

    def __post_init__(self) -> None:
        assert_int(
            self.max_candidates, self.max_evidence, self.max_witness_inputs,
            _where="R07 budget",
        )
        if any(type(item) is not int or item <= 0 for item in (
                self.max_candidates,
                self.max_evidence,
                self.max_witness_inputs)):
            raise W06R07ContractError("R07 budget 必须是严格正整数")

    def stable_key(self) -> tuple[int, ...]:
        return self.max_candidates, self.max_evidence, self.max_witness_inputs


@dataclass(frozen=True)
class W06R07ConsumerProtocol:
    """direct CAUSES、witness、temporal boundary 和 G postcheck 的门。"""

    causes_connected: bool = True
    witness_connected: bool = True
    temporal_boundary_connected: bool = True
    generation_connected: bool = True
    source_scope_connected: bool = True
    postcheck_connected: bool = True

    def __post_init__(self) -> None:
        for name in (
                "causes_connected", "witness_connected",
                "temporal_boundary_connected", "generation_connected",
                "source_scope_connected", "postcheck_connected"):
            _strict_bool(getattr(self, name), where=name)

    def query_ready(self, consumer: str) -> bool:
        if consumer not in W06_R07_CONSUMERS:
            raise W06R07ContractError("R07 consumer 未注册")
        return (
            self.causes_connected
            and self.witness_connected
            and self.temporal_boundary_connected
        )

    def generation_ready(self) -> bool:
        return (
            self.causes_connected
            and self.witness_connected
            and self.temporal_boundary_connected
            and self.generation_connected
            and self.source_scope_connected
        )

    def stable_key(self) -> tuple[int, ...]:
        return tuple(int(getattr(self, name)) for name in (
            "causes_connected", "witness_connected",
            "temporal_boundary_connected", "generation_connected",
            "source_scope_connected", "postcheck_connected",
        ))


@dataclass(frozen=True)
class W06R07CausalQuery:
    request_key: LosslessIntegerKey
    cause: ObjectIdentity
    effect: ObjectIdentity
    budget: W06R07Budget
    source: SourceRef

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R07ContractError("causal request_key 非法")
        _endpoints(self.cause, self.effect)
        if not isinstance(self.budget, W06R07Budget):
            raise W06R07ContractError("causal query budget 非法")
        if not isinstance(self.source, SourceRef):
            raise W06R07ContractError("causal query source 非法")

    def stable_key(self) -> tuple[int, ...]:
        return (
            *pack_key(self.request_key.components),
            *pack_key(self.cause.stable_key()),
            *pack_key(self.effect.stable_key()),
            *self.budget.stable_key(),
            *pack_key(self.source.stable_key()),
        )


@dataclass(frozen=True)
class W06R07WitnessAccount:
    """一个 CAUSES Proposition 的独立 teacher witness 归因。"""

    proposition: ObjectIdentity
    witness: CausalIndependentWitness
    forming_sources: tuple[SourceRef, ...]
    observation_source: SourceRef

    def __post_init__(self) -> None:
        _identity(self.proposition, where="witness proposition")
        if self.proposition.object_kind != OBJECT_PROPOSITION:
            raise W06R07ContractError("witness target 必须是 Proposition")
        _witness_key(self.witness)
        _sources(self.forming_sources, where="forming sources")
        if not isinstance(self.observation_source, SourceRef):
            raise W06R07ContractError("witness observation source 非法")
        if self.witness.verifier_source in {
                self.observation_source, *self.forming_sources}:
            raise W06R07ContractError(
                "causal witness source 必须与 forming observation 分离")
        if self.proposition in self.witness.input_objects:
            raise W06R07ContractError("causal witness 不得读取 Proposition 自身")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *pack_key(self.proposition.stable_key()),
            *pack_key(_witness_key(self.witness)),
            len(self.forming_sources),
        ]
        for item in self.forming_sources:
            values.extend(pack_key(item.stable_key()))
        values.extend(pack_key(self.observation_source.stable_key()))
        return tuple(values)


@dataclass(frozen=True)
class W06R07CausalEvaluation:
    query: W06R07CausalQuery
    state: LogicEvidenceState
    active_propositions: tuple[ObjectIdentity, ...]
    matched_propositions: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    witnesses: tuple[W06R07WitnessAccount, ...]
    direct_only: bool
    effect_execution_authorized: bool

    def __post_init__(self) -> None:
        if not isinstance(self.query, W06R07CausalQuery):
            raise W06R07ContractError("causal evaluation query 非法")
        if not isinstance(self.state, LogicEvidenceState):
            raise W06R07ContractError("causal evaluation state 非法")
        _identities(
            self.active_propositions,
            where="active propositions", allow_empty=True)
        _identities(
            self.matched_propositions,
            where="matched propositions", allow_empty=True)
        if not set(self.active_propositions).issubset(self.matched_propositions):
            raise W06R07ContractError("active propositions 未属于 matched 集")
        _evidence_keys(self.evidence_keys, where="causal evidence")
        if (not isinstance(self.witnesses, tuple)
                or any(not isinstance(item, W06R07WitnessAccount)
                       for item in self.witnesses)
                or self.witnesses != tuple(sorted(
                    self.witnesses, key=W06R07WitnessAccount.stable_key))):
            raise W06R07ContractError("causal witnesses 未规范化")
        _strict_bool(self.direct_only, where="direct_only")
        _strict_bool(
            self.effect_execution_authorized,
            where="effect_execution_authorized")
        if not self.direct_only or self.effect_execution_authorized:
            raise W06R07ContractError(
                "R07 train 无 temporal assertion，只允许 direct relation query")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *pack_key(self.query.stable_key()),
            int(self.state.support), int(self.state.refute),
        ]
        for group in (self.active_propositions, self.matched_propositions):
            values.append(len(group))
            for item in group:
                values.extend(pack_key(item.stable_key()))
        values.append(len(self.evidence_keys))
        for item in self.evidence_keys:
            values.extend(pack_key(item))
        values.append(len(self.witnesses))
        for item in self.witnesses:
            values.extend(pack_key(item.stable_key()))
        values.extend((int(self.direct_only), int(self.effect_execution_authorized)))
        return tuple(values)


@dataclass(frozen=True)
class W06R07CausalResolution:
    consumer: str
    request: W06R07CausalQuery
    status: str
    evaluation: W06R07CausalEvaluation
    propositions: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if self.consumer not in W06_R07_CONSUMERS:
            raise W06R07ContractError("causal resolution consumer 非法")
        if self.status not in W06_R07_QUERY_STATUSES:
            raise W06R07ContractError("causal resolution status 非法")
        if (not isinstance(self.evaluation, W06R07CausalEvaluation)
                or self.evaluation.query != self.request):
            raise W06R07ContractError("causal resolution evaluation 漂移")
        _identities(self.propositions, where="resolution propositions", allow_empty=True)
        _evidence_keys(self.evidence_keys, where="resolution evidence")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R07ContractError("causal resolution reason 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            W06_R07_CONSUMERS.index(self.consumer) + 1,
            *pack_key(self.request.stable_key()),
            W06_R07_QUERY_STATUSES.index(self.status) + 1,
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
class W06R07CausalUse:
    resolution: W06R07CausalResolution
    relation_uses: tuple[RelationClosureUse, ...]
    use_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if (not isinstance(self.resolution, W06R07CausalResolution)
                or self.resolution.status != W06_R07_SUPPORTED):
            raise W06R07ContractError("causal Use 必须来自 SUPPORTED resolution")
        if (not isinstance(self.relation_uses, tuple)
                or not self.relation_uses
                or any(not isinstance(item, RelationClosureUse)
                       for item in self.relation_uses)):
            raise W06R07ContractError("causal relation Uses 非法")
        if {item.proposition for item in self.relation_uses} != set(
                self.resolution.propositions):
            raise W06R07ContractError("causal Use 未精确归因 Proposition")
        if not isinstance(self.use_key, LosslessIntegerKey):
            raise W06R07ContractError("causal use_key 非法")


@dataclass(frozen=True)
class W06R07CausalOutcome:
    use: W06R07CausalUse
    verdict: str
    current_status: str
    outcome_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.use, W06R07CausalUse):
            raise W06R07ContractError("causal outcome Use 非法")
        if self.verdict not in W06_R07_OUTCOMES:
            raise W06R07ContractError("causal outcome verdict 非法")
        if self.current_status not in W06_R07_QUERY_STATUSES:
            raise W06R07ContractError("causal outcome status 非法")
        if not isinstance(self.outcome_key, LosslessIntegerKey):
            raise W06R07ContractError("causal outcome key 非法")


@dataclass(frozen=True)
class W06R07GenerationRequest:
    request_key: LosslessIntegerKey
    target_proposition: ObjectIdentity
    directionality: int
    role_fillers: tuple[tuple[ObjectIdentity, ObjectIdentity], ...]
    endpoints: CausalEndpointProtocol
    context: ObjectIdentity
    source: SourceRef
    uncertainty_units: int | None
    constraints: GenerationExpressionConstraints

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W06R07ContractError("generation request_key 非法")
        _identity(self.target_proposition, where="generation target")
        if self.target_proposition.object_kind != OBJECT_PROPOSITION:
            raise W06R07ContractError("generation target 必须是 Proposition")
        if self.directionality != DIRECTION_FORWARD:
            raise W06R07ContractError("R07 generation relation 必须有向")
        _role_fillers(self.role_fillers, where="generation role_fillers")
        _endpoint_protocol_key(self.endpoints)
        _identity(self.context, where="generation context")
        if self.context.object_kind != OBJECT_CONTEXT_SCOPE:
            raise W06R07ContractError("generation context 必须是 ContextScope")
        if not isinstance(self.source, SourceRef):
            raise W06R07ContractError("generation source 非法")
        if self.uncertainty_units is not None and (
                type(self.uncertainty_units) is not int
                or self.uncertainty_units < 0):
            raise W06R07ContractError("generation uncertainty 非法")
        if not isinstance(self.constraints, GenerationExpressionConstraints):
            raise W06R07ContractError("generation constraints 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *pack_key(self.request_key.components),
            *pack_key(self.target_proposition.stable_key()),
            self.directionality,
            len(self.role_fillers),
        ]
        for role, filler in self.role_fillers:
            values.extend(pack_key(role.stable_key()))
            values.extend(pack_key(filler.stable_key()))
        values.extend((
            *_endpoint_protocol_key(self.endpoints),
            *pack_key(self.context.stable_key()),
            *pack_key(self.source.stable_key()),
            -1 if self.uncertainty_units is None else self.uncertainty_units,
            *pack_key(self.constraints.stable_key()),
        ))
        return tuple(values)


@dataclass(frozen=True)
class W06R07GenerationOption:
    surface: str
    construction: ObjectIdentity
    target_proposition: ObjectIdentity
    directionality: int
    role_fillers: tuple[tuple[ObjectIdentity, ObjectIdentity], ...]
    endpoints: CausalEndpointProtocol
    canonical_pair: tuple[ObjectIdentity, ObjectIdentity]
    witness_keys: tuple[tuple[int, ...], ...]
    context: ObjectIdentity
    source: SourceRef
    uncertainty_units: int | None
    language_branch: ObjectIdentity
    authorization_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface:
            raise W06R07ContractError("generation option surface 非法")
        _identity(self.construction, where="generation construction")
        _identity(self.target_proposition, where="generation target")
        if self.directionality != DIRECTION_FORWARD:
            raise W06R07ContractError("generation option direction 漂移")
        _role_fillers(self.role_fillers, where="generation option roles")
        _endpoint_protocol_key(self.endpoints)
        if not isinstance(self.canonical_pair, tuple) or len(self.canonical_pair) != 2:
            raise W06R07ContractError("generation causal pair 非法")
        _endpoints(*self.canonical_pair)
        _evidence_keys(self.witness_keys, where="generation witness keys")
        if not self.witness_keys:
            raise W06R07ContractError("generation option 缺少 independent witness")
        _identity(self.context, where="generation context")
        if not isinstance(self.source, SourceRef):
            raise W06R07ContractError("generation source 非法")
        if self.uncertainty_units is not None and (
                type(self.uncertainty_units) is not int
                or self.uncertainty_units < 0):
            raise W06R07ContractError("generation uncertainty 非法")
        _identity(self.language_branch, where="generation language branch")
        if not isinstance(self.authorization_key, LosslessIntegerKey):
            raise W06R07ContractError("generation authorization 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            len(self.surface), *(ord(item) for item in self.surface),
            *pack_key(self.construction.stable_key()),
            *pack_key(self.target_proposition.stable_key()),
            self.directionality,
            len(self.role_fillers),
        ]
        for role, filler in self.role_fillers:
            values.extend(pack_key(role.stable_key()))
            values.extend(pack_key(filler.stable_key()))
        values.extend(_endpoint_protocol_key(self.endpoints))
        for item in self.canonical_pair:
            values.extend(pack_key(item.stable_key()))
        values.append(len(self.witness_keys))
        for item in self.witness_keys:
            values.extend(pack_key(item))
        values.extend((
            *pack_key(self.context.stable_key()),
            *pack_key(self.source.stable_key()),
            -1 if self.uncertainty_units is None else self.uncertainty_units,
            *pack_key(self.language_branch.stable_key()),
            *pack_key(self.authorization_key.components),
        ))
        return tuple(values)


@dataclass(frozen=True)
class W06R07GenerationChoice:
    request: W06R07GenerationRequest
    status: str
    options: tuple[W06R07GenerationOption, ...]
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.request, W06R07GenerationRequest):
            raise W06R07ContractError("generation choice request 非法")
        if self.status not in W06_R07_GENERATION_STATUSES:
            raise W06R07ContractError("generation choice status 非法")
        if (not isinstance(self.options, tuple)
                or any(not isinstance(item, W06R07GenerationOption)
                       for item in self.options)):
            raise W06R07ContractError("generation options 非法")
        if (self.options != tuple(sorted(
                self.options, key=W06R07GenerationOption.stable_key))
                or len(set(self.options)) != len(self.options)):
            raise W06R07ContractError("generation options 未规范化")
        if (self.status == W06_R07_GENERATION_READY) != bool(self.options):
            raise W06R07ContractError("generation READY 与 option 数不一致")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W06R07ContractError("generation reason 非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *pack_key(self.request.stable_key()),
            W06_R07_GENERATION_STATUSES.index(self.status) + 1,
            len(self.options),
        ]
        for item in self.options:
            values.extend(pack_key(item.stable_key()))
        values.extend(pack_key(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W06R07GenerationUse:
    choice: W06R07GenerationChoice
    option: W06R07GenerationOption
    relation_uses: tuple[RelationClosureUse, ...]
    ref: GenerationChoiceUseRef

    def __post_init__(self) -> None:
        if (not isinstance(self.choice, W06R07GenerationChoice)
                or self.choice.status != W06_R07_GENERATION_READY
                or self.option not in self.choice.options):
            raise W06R07ContractError("generation Use 未采用 READY option")
        if (not isinstance(self.relation_uses, tuple)
                or len(self.relation_uses) != 1
                or any(not isinstance(item, RelationClosureUse)
                       for item in self.relation_uses)):
            raise W06R07ContractError("generation 必须精确提交一个 direct Use")
        if {item.proposition for item in self.relation_uses} != {
                self.option.target_proposition}:
            raise W06R07ContractError("generation Use 未归因 target")
        if not isinstance(self.ref, GenerationChoiceUseRef):
            raise W06R07ContractError("generation Use ref 非法")


@dataclass(frozen=True)
class W06R07GenerationOutcome:
    use: W06R07GenerationUse
    verdict: str
    ref: GenerationChoiceOutcomeRef
    authorization_current: bool
    witness_current: bool
    causal_structure_preserved: bool
    source_scope_preserved: bool
    surface_structure_valid: bool
    causal_query_status: str
    recovered_target: bool
    effect_execution_authorized: bool

    def __post_init__(self) -> None:
        if not isinstance(self.use, W06R07GenerationUse):
            raise W06R07ContractError("generation outcome Use 非法")
        if self.verdict not in W06_R07_OUTCOMES:
            raise W06R07ContractError("generation outcome verdict 非法")
        if not isinstance(self.ref, GenerationChoiceOutcomeRef):
            raise W06R07ContractError("generation outcome ref 非法")
        for name in (
                "authorization_current", "witness_current",
                "causal_structure_preserved", "source_scope_preserved",
                "surface_structure_valid", "recovered_target",
                "effect_execution_authorized"):
            _strict_bool(getattr(self, name), where=name)
        if self.causal_query_status not in W06_R07_QUERY_STATUSES:
            raise W06R07ContractError("generation causal status 非法")
        if self.effect_execution_authorized:
            raise W06R07ContractError("R07 relation generation 不得执行 effect")
        expected = all((
            self.authorization_current,
            self.witness_current,
            self.causal_structure_preserved,
            self.source_scope_preserved,
            self.surface_structure_valid,
            self.causal_query_status == W06_R07_SUPPORTED,
            self.recovered_target,
        ))
        if (self.verdict == W06_R07_OUTCOME_SUPPORT) != expected:
            raise W06R07ContractError("generation verdict 未匹配 postcheck")


__all__ = [
    "W06R07Budget",
    "W06R07BudgetExceeded",
    "W06R07CausalEvaluation",
    "W06R07CausalOutcome",
    "W06R07CausalQuery",
    "W06R07CausalResolution",
    "W06R07CausalUse",
    "W06R07ConsumerProtocol",
    "W06R07ContractError",
    "W06R07GenerationChoice",
    "W06R07GenerationOption",
    "W06R07GenerationOutcome",
    "W06R07GenerationRequest",
    "W06R07GenerationUse",
    "W06R07WitnessAccount",
    "W06_R07_CONFLICT",
    "W06_R07_CONSUMERS",
    "W06_R07_GENERATION_READY",
    "W06_R07_GENERATION_REJECTED",
    "W06_R07_GENERATION_UNKNOWN",
    "W06_R07_OUTCOME_REFUTE",
    "W06_R07_OUTCOME_SUPPORT",
    "W06_R07_QUERY_STATUSES",
    "W06_R07_REFUTED",
    "W06_R07_RUNTIME_NAMESPACE",
    "W06_R07_SUBSTAGE",
    "W06_R07_SUPPORTED",
    "W06_R07_UNKNOWN",
    "pack_key",
]
