"""P2-G 五类专属证明的生产输入、预算和只读结果合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.causal_execution import (
    CausalExecutionResult,
)
from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_AFTER,
    EVENT_TIME_BEFORE,
    EVENT_TIME_SAME,
    EventTimeVerificationResult,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OBJECT_SET_EXPR,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    LogicEvaluation,
    LogicEvidenceState,
)
from pure_integer_ai.cognition.shared.modal_primitives import (
    MODAL_KIND_BOX_NECESSITY,
    MODAL_KIND_BOX_POSSIBILITY,
    MODAL_KIND_DEONTIC_NECESSITY,
    MODAL_KIND_DEONTIC_POSSIBILITY,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.causal_relation_runtime import (
    CausalIndependentWitness,
)


PROOF_FAMILY_TEMPORAL = 1
PROOF_FAMILY_CAUSAL_COUNTERFACTUAL = 2
PROOF_FAMILY_CONDITION = 3
PROOF_FAMILY_NOT = 4
PROOF_FAMILY_MODAL = 5
PROOF_FAMILIES = frozenset({
    PROOF_FAMILY_TEMPORAL,
    PROOF_FAMILY_CAUSAL_COUNTERFACTUAL,
    PROOF_FAMILY_CONDITION,
    PROOF_FAMILY_NOT,
    PROOF_FAMILY_MODAL,
})

PROOF_ACCEPTED = 1
PROOF_REJECTED = 2
PROOF_UNKNOWN = 3
PROOF_CONFLICTED = 4
PROOF_BUDGET_EXHAUSTED = 5
PROOF_FAMILY_MISMATCH = 6
PROOF_FAIL_CLOSED = 7
PROOF_STATUSES = frozenset({
    PROOF_ACCEPTED,
    PROOF_REJECTED,
    PROOF_UNKNOWN,
    PROOF_CONFLICTED,
    PROOF_BUDGET_EXHAUSTED,
    PROOF_FAMILY_MISMATCH,
    PROOF_FAIL_CLOSED,
})

CAUSAL_DIRECTION_PROMOTING = 1
CAUSAL_DIRECTION_INHIBITING = 2

CONDITION_MATERIAL = 1
CONDITION_SUFFICIENT = 2
CONDITION_NECESSARY = 3
CONDITION_KINDS = frozenset({
    CONDITION_MATERIAL,
    CONDITION_SUFFICIENT,
    CONDITION_NECESSARY,
})

CONDITION_ASSERTION = 1
CONDITION_AFFIRMING_CONSEQUENT = 2

MODAL_FRAME_EPISTEMIC = 1
MODAL_FRAME_NORMATIVE = 2

_ENDPOINT_KINDS = frozenset({OBJECT_EVENT, OBJECT_PROPOSITION})
_TEMPORAL_DIRECTIONS = frozenset({
    EVENT_TIME_BEFORE,
    EVENT_TIME_AFTER,
    EVENT_TIME_SAME,
})
_MODAL_KINDS = frozenset({
    MODAL_KIND_BOX_NECESSITY,
    MODAL_KIND_BOX_POSSIBILITY,
    MODAL_KIND_DEONTIC_NECESSITY,
    MODAL_KIND_DEONTIC_POSSIBILITY,
})


def _strict_key(
        value: tuple[int, ...], *, where: str, allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """核验稳定 trace/evidence key 只含严格整数。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{where} 必须是整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise TypeError(f"{where} 只能包含严格 int")
    return value


def _identity(value: ObjectIdentity, kind: int, *, where: str) -> None:
    """要求一等身份具有调用方声明的对象类型。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{where} 必须是 ObjectIdentity")
    if value.object_kind != kind:
        raise ValueError(f"{where} 对象类型不匹配")


def _endpoint(value: ObjectIdentity, *, where: str) -> None:
    """要求 temporal/causal 端点是 Event 或 Proposition。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{where} 必须是 ObjectIdentity")
    if value.object_kind not in _ENDPOINT_KINDS:
        raise ValueError(f"{where} 必须是 Event 或 Proposition")


def _family(value: int) -> None:
    """核验声明 family 已注册；是否匹配由 dispatcher 独立裁决。"""
    assert_int(value, _where="typed proof family")
    if type(value) is not int or value not in PROOF_FAMILIES:
        raise ValueError("typed proof family 未注册")


def _logic(value: LogicEvaluation, *, where: str) -> None:
    """要求输入是现役 S-04 typed 求值结果。"""
    if not isinstance(value, LogicEvaluation):
        raise TypeError(f"{where} 必须是 LogicEvaluation")


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键加长度前缀。"""
    return len(value), *value


def _event_time_key(result: EventTimeVerificationResult) -> tuple[int, ...]:
    """编码 event-time 原始断言、scope、方向图和冲突归因。"""
    values: list[int] = [
        result.status,
        *_pack(result.fact_set.scope.stable_key()),
        len(result.fact_set.relations),
    ]
    for relation in result.fact_set.relations:
        values.extend(_pack(relation.stable_key()))
    values.append(len(result.fact_set.facts))
    for fact in result.fact_set.facts:
        values.extend(_pack(fact.statement.assertion.stable_key()))
        values.append(fact.assertion_hash)
    values.append(len(result.before_edges))
    for first, second in result.before_edges:
        values.extend(_pack(first.stable_key()))
        values.extend(_pack(second.stable_key()))
    values.append(len(result.same_groups))
    for group in result.same_groups:
        values.append(len(group))
        for item in group:
            values.extend(_pack(item.stable_key()))
    values.append(len(result.unknown_relations))
    for relation in result.unknown_relations:
        values.extend(_pack(relation.stable_key()))
    values.extend(_pack(result.conflict_assertion_hashes))
    values.append(len(result.detail_keys))
    for key in result.detail_keys:
        values.extend(_pack(key))
    return tuple(values)


@dataclass(frozen=True)
class ProofWorkBudget:
    """一次或多次 dispatcher 调用共享的严格工作单元预算。"""

    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        """预算必须为正，已用量必须位于闭区间内。"""
        assert_int(self.limit, self.used, _where="ProofWorkBudget")
        if type(self.limit) is not int or self.limit <= 0:
            raise ValueError("proof budget limit 必须为严格正整数")
        if type(self.used) is not int or not 0 <= self.used <= self.limit:
            raise ValueError("proof budget used 越界")

    @property
    def remaining(self) -> int:
        """返回尚未消费的工作单元。"""
        return self.limit - self.used

    def reserve(self, units: int) -> "ProofWorkBudget | None":
        """原子预留工作单元；不足时不改变原预算。"""
        assert_int(units, _where="ProofWorkBudget.reserve")
        if type(units) is not int or units <= 0:
            raise ValueError("proof work units 必须为严格正整数")
        if units > self.remaining:
            return None
        return ProofWorkBudget(self.limit, self.used + units)


@dataclass(frozen=True)
class ProofCheckResult:
    """一个 family checker 的稳定四态/fail-closed 结果和预算收据。"""

    family: int
    status: int
    work_units: int
    trace: tuple[int, ...]
    certificate_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 family、状态、工作量和无文本 trace。"""
        _family(self.family)
        assert_int(self.status, self.work_units, _where="ProofCheckResult")
        if type(self.status) is not int or self.status not in PROOF_STATUSES:
            raise ValueError("proof status 未注册")
        if type(self.work_units) is not int or self.work_units < 0:
            raise ValueError("proof work_units 必须为非负严格整数")
        _strict_key(self.trace, where="ProofCheckResult.trace")
        _strict_key(
            self.certificate_key,
            where="ProofCheckResult.certificate_key",
            allow_empty=True,
        )

    @property
    def accepted(self) -> bool:
        """只把专属 checker 的 provisional 通过解释为 accepted。"""
        return self.status == PROOF_ACCEPTED

    def stable_key(self) -> tuple[int, ...]:
        """返回 family、状态、预算和全部归因的稳定键。"""
        return (
            self.family,
            self.status,
            self.work_units,
            *_pack(self.trace),
            *_pack(self.certificate_key),
        )


@dataclass(frozen=True)
class TemporalProofCertificate:
    """显式端点、方向、scope 和原始时间前提的 temporal certificate。"""

    declared_family: int
    result: EventTimeVerificationResult
    relation: ObjectIdentity
    first: ObjectIdentity
    second: ObjectIdentity
    direction: int
    premise_assertion_hashes: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity

    def __post_init__(self) -> None:
        """核验 typed 形状；是否得到方向由 checker 裁决。"""
        _family(self.declared_family)
        if not isinstance(self.result, EventTimeVerificationResult):
            raise TypeError("temporal result 类型非法")
        _identity(self.relation, OBJECT_CONCEPT, where="temporal relation")
        _endpoint(self.first, where="temporal first")
        _endpoint(self.second, where="temporal second")
        if self.first == self.second:
            raise ValueError("temporal certificate 端点不得相同")
        assert_int(self.direction, _where="temporal direction")
        if self.direction not in _TEMPORAL_DIRECTIONS:
            raise ValueError("temporal direction 未注册")
        _strict_key(
            self.premise_assertion_hashes,
            where="temporal premise assertions",
        )
        if (any(item <= 0 for item in self.premise_assertion_hashes)
                or len(set(self.premise_assertion_hashes))
                != len(self.premise_assertion_hashes)):
            raise ValueError("temporal premise assertion 必须唯一且为正")
        if not isinstance(self.source, SourceRef):
            raise TypeError("temporal source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("temporal scope 必须是 ScopeIdentity")

    def stable_key(self) -> tuple[int, ...]:
        """返回全部端点、前提与 verifier 结果的稳定键。"""
        return (
            self.declared_family,
            *_pack(_event_time_key(self.result)),
            *_pack(self.relation.stable_key()),
            *_pack(self.first.stable_key()),
            *_pack(self.second.stable_key()),
            self.direction,
            *_pack(self.premise_assertion_hashes),
            *_pack(self.source.stable_key()),
            *_pack(self.scope.stable_key()),
        )


@dataclass(frozen=True)
class CounterfactualState:
    """一个独立来源下 cause/effect 的 baseline 或 intervention 状态。"""

    cause: ObjectIdentity
    effect: ObjectIdentity
    cause_state: LogicEvidenceState
    effect_state: LogicEvidenceState
    source: SourceRef
    scope: ScopeIdentity
    evidence_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验端点、S-04 四态、独立来源和显式 Evidence。"""
        _endpoint(self.cause, where="counterfactual cause")
        _endpoint(self.effect, where="counterfactual effect")
        if self.cause == self.effect:
            raise ValueError("counterfactual 端点不得相同")
        if not isinstance(self.cause_state, LogicEvidenceState):
            raise TypeError("counterfactual cause_state 类型非法")
        if not isinstance(self.effect_state, LogicEvidenceState):
            raise TypeError("counterfactual effect_state 类型非法")
        if not isinstance(self.source, SourceRef):
            raise TypeError("counterfactual source 类型非法")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("counterfactual scope 类型非法")
        if self.scope.source != self.source:
            raise ValueError("counterfactual scope 未绑定 source")
        _strict_key(self.evidence_ids, where="counterfactual evidence_ids")
        if (any(item <= 0 for item in self.evidence_ids)
                or len(set(self.evidence_ids)) != len(self.evidence_ids)):
            raise ValueError("counterfactual Evidence id 必须唯一且为正")

    def stable_key(self) -> tuple[int, ...]:
        """返回端点、四态、来源和 Evidence 的稳定键。"""
        return (
            *_pack(self.cause.stable_key()),
            *_pack(self.effect.stable_key()),
            *self.cause_state.stable_key(),
            *self.effect_state.stable_key(),
            *_pack(self.source.stable_key()),
            *_pack(self.scope.stable_key()),
            *_pack(self.evidence_ids),
        )


@dataclass(frozen=True)
class CounterfactualPair:
    """同一独立 verifier source 下显式区分的 baseline/intervention。"""

    baseline: CounterfactualState
    intervention: CounterfactualState

    def __post_init__(self) -> None:
        """要求两侧端点和来源相同，但 scope 必须互异。"""
        if not isinstance(self.baseline, CounterfactualState):
            raise TypeError("baseline 类型非法")
        if not isinstance(self.intervention, CounterfactualState):
            raise TypeError("intervention 类型非法")
        if (self.baseline.cause != self.intervention.cause
                or self.baseline.effect != self.intervention.effect):
            raise ValueError("counterfactual pair 替换了 cause/effect")
        if self.baseline.source != self.intervention.source:
            raise ValueError("counterfactual pair source 不一致")
        if self.baseline.scope == self.intervention.scope:
            raise ValueError("baseline/intervention scope 必须互异")

    def stable_key(self) -> tuple[int, ...]:
        """返回有序 baseline/intervention 稳定键。"""
        return (
            *_pack(self.baseline.stable_key()),
            *_pack(self.intervention.stable_key()),
        )


@dataclass(frozen=True)
class CausalCounterfactualProofCertificate:
    """绑定 active causal execution、独立 witness 和反事实配对的证书。"""

    declared_family: int
    execution: CausalExecutionResult
    witness: CausalIndependentWitness
    direction: int
    pair: CounterfactualPair

    def __post_init__(self) -> None:
        """核验输入类型和 promoting/inhibiting 方向枚举。"""
        _family(self.declared_family)
        if not isinstance(self.execution, CausalExecutionResult):
            raise TypeError("causal execution 类型非法")
        if not isinstance(self.witness, CausalIndependentWitness):
            raise TypeError("causal witness 类型非法")
        assert_int(self.direction, _where="causal direction")
        if self.direction not in {
                CAUSAL_DIRECTION_PROMOTING,
                CAUSAL_DIRECTION_INHIBITING}:
            raise ValueError("causal direction 未注册")
        if not isinstance(self.pair, CounterfactualPair):
            raise TypeError("counterfactual pair 类型非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回 execution、witness、方向和反事实配对完整键。"""
        witness_values: list[int] = [self.witness.stance]
        witness_values.extend(_pack(self.witness.verifier_source.stable_key()))
        witness_values.append(len(self.witness.input_objects))
        for item in self.witness.input_objects:
            witness_values.extend(_pack(item.stable_key()))
        witness_values.extend(_pack(self.witness.trace))
        return (
            self.declared_family,
            *_pack(self.execution.stable_key()),
            *_pack(tuple(witness_values)),
            self.direction,
            *_pack(self.pair.stable_key()),
        )


@dataclass(frozen=True)
class ConditionProofCertificate:
    """区分 material/sufficient/necessary 且保留方向的 condition 证书。"""

    declared_family: int
    condition_kind: int
    structure: ObjectIdentity
    condition: LogicEvaluation
    conditioned: LogicEvaluation
    evaluation: LogicEvaluation
    inference_kind: int
    kind_evidence_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验显式结构、三项 S-04 求值和条件类型 Evidence。"""
        _family(self.declared_family)
        assert_int(self.condition_kind, _where="condition kind")
        if self.condition_kind not in CONDITION_KINDS:
            raise ValueError("condition kind 未注册")
        _identity(
            self.structure,
            OBJECT_STRUCTURE_CONCEPT,
            where="condition structure",
        )
        _logic(self.condition, where="condition premise")
        _logic(self.conditioned, where="conditioned premise")
        _logic(self.evaluation, where="condition evaluation")
        assert_int(self.inference_kind, _where="condition inference kind")
        if self.inference_kind not in {
                CONDITION_ASSERTION,
                CONDITION_AFFIRMING_CONSEQUENT}:
            raise ValueError("condition inference kind 未注册")
        _strict_key(
            self.kind_evidence_ids,
            where="condition kind evidence_ids",
            allow_empty=True,
        )
        if (any(item <= 0 for item in self.kind_evidence_ids)
                or len(set(self.kind_evidence_ids))
                != len(self.kind_evidence_ids)):
            raise ValueError("condition kind Evidence id 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回结构、方向前提、求值和类型 Evidence 的稳定键。"""
        return (
            self.declared_family,
            self.condition_kind,
            *_pack(self.structure.stable_key()),
            *_pack(self.condition.stable_key()),
            *_pack(self.conditioned.stable_key()),
            *_pack(self.evaluation.stable_key()),
            self.inference_kind,
            *_pack(self.kind_evidence_ids),
        )


@dataclass(frozen=True)
class NotProofCertificate:
    """显式 structural NOT 的子命题和根求值证书。"""

    declared_family: int
    structure: ObjectIdentity
    child: LogicEvaluation
    evaluation: LogicEvaluation

    def __post_init__(self) -> None:
        """核验 NOT 结构身份和两项现役 S-04 求值。"""
        _family(self.declared_family)
        _identity(
            self.structure,
            OBJECT_STRUCTURE_CONCEPT,
            where="NOT structure",
        )
        _logic(self.child, where="NOT child")
        _logic(self.evaluation, where="NOT evaluation")

    def stable_key(self) -> tuple[int, ...]:
        """返回 structural NOT 的完整稳定键。"""
        return (
            self.declared_family,
            *_pack(self.structure.stable_key()),
            *_pack(self.child.stable_key()),
            *_pack(self.evaluation.stable_key()),
        )


@dataclass(frozen=True)
class ModalProofCertificate:
    """绑定 modal kind、frame、world/domain 和 resolver Evidence 的证书。"""

    declared_family: int
    structure: ObjectIdentity
    child: LogicEvaluation
    evaluation: LogicEvaluation
    modal_kind: int
    frame: int
    worlds: tuple[ObjectIdentity, ...]
    domains: tuple[ObjectIdentity, ...]
    worlds_complete: bool
    domains_complete: bool
    resolver_evidence_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验显式 modal 元数据和完整性声明，不推断其语义。"""
        _family(self.declared_family)
        _identity(
            self.structure,
            OBJECT_STRUCTURE_CONCEPT,
            where="modal structure",
        )
        _logic(self.child, where="modal child")
        _logic(self.evaluation, where="modal evaluation")
        assert_int(self.modal_kind, self.frame, _where="modal kind/frame")
        if self.modal_kind not in _MODAL_KINDS:
            raise ValueError("modal kind 未注册")
        if self.frame not in {MODAL_FRAME_EPISTEMIC, MODAL_FRAME_NORMATIVE}:
            raise ValueError("modal frame 未注册")
        if not isinstance(self.worlds, tuple) or not self.worlds:
            raise ValueError("modal worlds 必须是非空 tuple")
        for world in self.worlds:
            _identity(world, OBJECT_CONTEXT_SCOPE, where="modal world")
        if len(set(self.worlds)) != len(self.worlds):
            raise ValueError("modal world 不得重复")
        if not isinstance(self.domains, tuple) or not self.domains:
            raise ValueError("modal domains 必须是非空 tuple")
        for domain in self.domains:
            _identity(domain, OBJECT_SET_EXPR, where="modal domain")
        if len(set(self.domains)) != len(self.domains):
            raise ValueError("modal domain 不得重复")
        if type(self.worlds_complete) is not bool:
            raise TypeError("worlds_complete 必须是严格 bool")
        if type(self.domains_complete) is not bool:
            raise TypeError("domains_complete 必须是严格 bool")
        _strict_key(
            self.resolver_evidence_ids,
            where="modal resolver evidence_ids",
            allow_empty=True,
        )
        if (any(item <= 0 for item in self.resolver_evidence_ids)
                or len(set(self.resolver_evidence_ids))
                != len(self.resolver_evidence_ids)):
            raise ValueError("modal resolver Evidence id 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回 modal kind、frame、world/domain 和求值的完整键。"""
        values: list[int] = [
            self.declared_family,
            *_pack(self.structure.stable_key()),
            *_pack(self.child.stable_key()),
            *_pack(self.evaluation.stable_key()),
            self.modal_kind,
            self.frame,
            len(self.worlds),
        ]
        for world in self.worlds:
            values.extend(_pack(world.stable_key()))
        values.append(len(self.domains))
        for domain in self.domains:
            values.extend(_pack(domain.stable_key()))
        values.extend((
            int(self.worlds_complete),
            int(self.domains_complete),
            *_pack(self.resolver_evidence_ids),
        ))
        return tuple(values)


__all__ = [
    "CAUSAL_DIRECTION_INHIBITING",
    "CAUSAL_DIRECTION_PROMOTING",
    "CONDITION_AFFIRMING_CONSEQUENT",
    "CONDITION_ASSERTION",
    "CONDITION_MATERIAL",
    "CONDITION_NECESSARY",
    "CONDITION_SUFFICIENT",
    "CounterfactualPair",
    "CounterfactualState",
    "CausalCounterfactualProofCertificate",
    "ConditionProofCertificate",
    "MODAL_FRAME_EPISTEMIC",
    "MODAL_FRAME_NORMATIVE",
    "ModalProofCertificate",
    "NotProofCertificate",
    "PROOF_ACCEPTED",
    "PROOF_BUDGET_EXHAUSTED",
    "PROOF_CONFLICTED",
    "PROOF_FAIL_CLOSED",
    "PROOF_FAMILIES",
    "PROOF_FAMILY_CAUSAL_COUNTERFACTUAL",
    "PROOF_FAMILY_CONDITION",
    "PROOF_FAMILY_MISMATCH",
    "PROOF_FAMILY_MODAL",
    "PROOF_FAMILY_NOT",
    "PROOF_FAMILY_TEMPORAL",
    "PROOF_REJECTED",
    "PROOF_STATUSES",
    "PROOF_UNKNOWN",
    "ProofCheckResult",
    "ProofWorkBudget",
    "TemporalProofCertificate",
]
