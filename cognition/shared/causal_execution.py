"""active typed CAUSES 事实的时间约束和 provisional 执行消费。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_CONFLICTED,
    EVENT_TIME_CONSISTENT,
    EVENT_TIME_EMPTY,
    EVENT_TIME_UNKNOWN,
    EventTimeVerificationResult,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_EVENT,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    LogicEvaluation,
    LogicEvidenceState,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureFact,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


CAUSAL_TEMPORAL_ACCEPTED = 1
CAUSAL_TEMPORAL_REJECTED = 2
CAUSAL_TEMPORAL_UNKNOWN = 3
CAUSAL_TEMPORAL_CONFLICTED = 4
_TEMPORAL_STATES = frozenset({
    CAUSAL_TEMPORAL_ACCEPTED,
    CAUSAL_TEMPORAL_REJECTED,
    CAUSAL_TEMPORAL_UNKNOWN,
    CAUSAL_TEMPORAL_CONFLICTED,
})

CAUSAL_EXECUTION_UNKNOWN = 0
CAUSAL_EXECUTION_PREDICTED = 1
CAUSAL_EXECUTION_CONFLICTED = 2
CAUSAL_EXECUTION_TEMPORAL_REJECTED = 3
_EXECUTION_STATES = frozenset({
    CAUSAL_EXECUTION_UNKNOWN,
    CAUSAL_EXECUTION_PREDICTED,
    CAUSAL_EXECUTION_CONFLICTED,
    CAUSAL_EXECUTION_TEMPORAL_REJECTED,
})
_ENDPOINT_KINDS = frozenset({OBJECT_EVENT, OBJECT_PROPOSITION})


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """校验并返回非空严格整数协议键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise TypeError(f"{where} 必须只含严格 int")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长整数键添加长度前缀，避免连续拼接产生歧义。"""
    return len(value), *value


def _event_time_result_key(
        result: EventTimeVerificationResult,
        ) -> tuple[int, ...]:
    """展开 event-time 事实、规范方向、冲突和 resolver 归因。"""
    values: list[int] = [
        result.status,
        *_pack(result.fact_set.scope.stable_key()),
        len(result.fact_set.relations),
    ]
    for relation in result.fact_set.relations:
        values.extend(_pack(relation.stable_key()))
    values.append(len(result.fact_set.facts))
    for fact in result.fact_set.facts:
        statement = fact.statement
        values.extend((
            statement.assertion_hash,
            statement.predicate_identity_hash,
            *_pack(statement.predicate.stable_key()),
            *_pack(statement.subject.stable_key()),
            *_pack(statement.object.stable_key()),
            statement.scope_hash,
            *_pack(statement.assertion.stable_key()),
        ))
    values.append(len(result.before_edges))
    for before, after in result.before_edges:
        values.extend((
            *_pack(before.stable_key()),
            *_pack(after.stable_key()),
        ))
    values.append(len(result.same_groups))
    for group in result.same_groups:
        values.append(len(group))
        for endpoint in group:
            values.extend(_pack(endpoint.stable_key()))
    values.append(len(result.unknown_relations))
    for relation in result.unknown_relations:
        values.extend(_pack(relation.stable_key()))
    values.extend(_pack(result.conflict_assertion_hashes))
    values.append(len(result.detail_keys))
    for key in result.detail_keys:
        values.extend(_pack(key))
    return tuple(values)


def _identity(
        value: ObjectIdentity, expected_kind: int, *, where: str,
        ) -> ObjectIdentity:
    """核验一等身份的宿主对象类型，不解释其图中名称。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{where} 必须是 ObjectIdentity")
    if value.object_kind != expected_kind:
        raise ValueError(f"{where} 对象类型不匹配")
    return value


@dataclass(frozen=True)
class CausalEndpointProtocol:
    """声明 cause/effect Role 和 causal execution 最小指令。"""

    relation: ObjectIdentity
    cause_role: ObjectIdentity
    effect_role: ObjectIdentity
    execution_instruction: ObjectIdentity

    def __post_init__(self) -> None:
        """核验 relation、两个 Role 和执行指令均为正确的一等对象。"""
        _identity(self.relation, OBJECT_CONCEPT, where="causal relation")
        _identity(self.cause_role, OBJECT_ROLE, where="cause_role")
        _identity(self.effect_role, OBJECT_ROLE, where="effect_role")
        _identity(
            self.execution_instruction,
            OBJECT_MINIMAL_INSTRUCTION,
            where="execution_instruction",
        )
        if self.cause_role == self.effect_role:
            raise ValueError("cause/effect Role 必须不同")


def causal_endpoints(
        definition: AtomicPropositionDefinition,
        protocol: CausalEndpointProtocol,
        ) -> tuple[ObjectIdentity, ObjectIdentity]:
    """从 typed CAUSES 命题恢复唯一 cause/effect Event 或 Proposition。"""
    if not isinstance(definition, AtomicPropositionDefinition):
        raise TypeError("definition 必须是 AtomicPropositionDefinition")
    if not isinstance(protocol, CausalEndpointProtocol):
        raise TypeError("protocol 必须是 CausalEndpointProtocol")
    if definition.predicate != protocol.relation:
        raise ValueError("命题 predicate 与 causal relation 协议不一致")
    cause = tuple(
        binding.filler
        for binding in definition.bindings
        if binding.role == protocol.cause_role
    )
    effect = tuple(
        binding.filler
        for binding in definition.bindings
        if binding.role == protocol.effect_role
    )
    if len(cause) != 1 or len(effect) != 1:
        raise ValueError("causal 命题必须各有一个 cause/effect filler")
    if cause[0].object_kind not in _ENDPOINT_KINDS:
        raise ValueError("cause filler 必须是 Event 或 Proposition")
    if effect[0].object_kind not in _ENDPOINT_KINDS:
        raise ValueError("effect filler 必须是 Event 或 Proposition")
    if cause[0] == effect[0]:
        raise ValueError("causal 命题不得使用相同 cause/effect 端点")
    return cause[0], effect[0]


@dataclass(frozen=True)
class CausalTemporalAssessment:
    """调用方对 R-06 时间核验结果给出的 causal 约束裁决。"""

    status: int
    reason: ObjectIdentity
    trace: tuple[int, ...]
    supporting_assertion_hashes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验时间裁决状态、原因、trace 和事实引用完整合法。"""
        assert_int(self.status, _where="CausalTemporalAssessment.status")
        if self.status not in _TEMPORAL_STATES:
            raise ValueError("causal temporal status 未注册")
        _identity(
            self.reason,
            OBJECT_MINIMAL_INSTRUCTION,
            where="causal temporal reason",
        )
        _strict_key(self.trace, where="CausalTemporalAssessment.trace")
        if not isinstance(self.supporting_assertion_hashes, tuple):
            raise TypeError("supporting_assertion_hashes 必须是 tuple")
        assert_int(
            *self.supporting_assertion_hashes,
            _where="CausalTemporalAssessment.assertions",
        )
        if any(type(item) is not int or item <= 0
               for item in self.supporting_assertion_hashes):
            raise ValueError("supporting assertion hash 必须为正整数")
        if len(set(self.supporting_assertion_hashes)) != len(
                self.supporting_assertion_hashes):
            raise ValueError("supporting assertion hash 不得重复")
        if (self.status in {
                CAUSAL_TEMPORAL_ACCEPTED,
                CAUSAL_TEMPORAL_REJECTED,
            } and not self.supporting_assertion_hashes):
            raise ValueError("accepted/rejected temporal 裁决必须引用事实")

    def stable_key(self) -> tuple[int, ...]:
        """返回时间裁决、原因、trace 和事实引用的完整比较键。"""
        return (
            self.status,
            *_pack(self.reason.stable_key()),
            *_pack(self.trace),
            *_pack(self.supporting_assertion_hashes),
        )


class CausalTemporalResolver(Protocol):
    """把 typed event-time artifact 解释为当前 causal 约束裁决。"""

    def resolve(
            self,
            cause: ObjectIdentity,
            effect: ObjectIdentity,
            result: EventTimeVerificationResult,
            ) -> CausalTemporalAssessment: ...


def validate_temporal_assessment(
        result: EventTimeVerificationResult,
        assessment: CausalTemporalAssessment,
        ) -> CausalTemporalAssessment:
    """防止无事实、未知或冲突时间图被 resolver 错升为 accepted。"""
    if not isinstance(result, EventTimeVerificationResult):
        raise TypeError("result 必须是 EventTimeVerificationResult")
    if not isinstance(assessment, CausalTemporalAssessment):
        raise TypeError("assessment 必须是 CausalTemporalAssessment")
    available_hashes = {
        fact.assertion_hash for fact in result.fact_set.facts
    }
    if not set(assessment.supporting_assertion_hashes).issubset(
            available_hashes):
        raise ValueError("temporal assessment 引用了结果外 assertion")
    if result.status in {EVENT_TIME_EMPTY, EVENT_TIME_UNKNOWN}:
        if assessment.status != CAUSAL_TEMPORAL_UNKNOWN:
            raise ValueError("无事实或未知时间关系只能保持 temporal unknown")
    elif result.status == EVENT_TIME_CONFLICTED:
        if assessment.status != CAUSAL_TEMPORAL_CONFLICTED:
            raise ValueError("冲突时间图必须保持 temporal conflicted")
    elif result.status == EVENT_TIME_CONSISTENT:
        if assessment.status == CAUSAL_TEMPORAL_CONFLICTED:
            raise ValueError("一致时间图不得伪造 temporal conflict")
    else:
        raise ValueError("event-time result status 未注册")
    return assessment


@dataclass(frozen=True)
class CausalEndpointEvaluation:
    """把 causal 端点显式映射到当前 query 的 S-04 四态求值。"""

    endpoint: ObjectIdentity
    evaluation: LogicEvaluation
    binding_instruction: ObjectIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验端点类型、S-04 求值、绑定指令和映射 trace。"""
        if not isinstance(self.endpoint, ObjectIdentity):
            raise TypeError("causal endpoint 必须是 ObjectIdentity")
        if self.endpoint.object_kind not in _ENDPOINT_KINDS:
            raise ValueError("causal endpoint 必须是 Event 或 Proposition")
        if not isinstance(self.evaluation, LogicEvaluation):
            raise TypeError("evaluation 必须是 LogicEvaluation")
        _identity(
            self.binding_instruction,
            OBJECT_MINIMAL_INSTRUCTION,
            where="endpoint binding instruction",
        )
        _strict_key(self.trace, where="CausalEndpointEvaluation.trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回端点、S-04 求值、绑定指令和映射 trace 的完整键。"""
        return (
            *_pack(self.endpoint.stable_key()),
            *_pack(self.evaluation.stable_key()),
            *_pack(self.binding_instruction.stable_key()),
            *_pack(self.trace),
        )


@dataclass(frozen=True)
class CausalExecutionResult:
    """保留 active relation、时间约束和 S-04 trace 的 provisional 预测。"""

    fact: ActiveRelationClosureFact
    temporal_result: EventTimeVerificationResult
    temporal_assessment: CausalTemporalAssessment
    cause: CausalEndpointEvaluation
    effect: CausalEndpointEvaluation
    status: int
    effect_state: LogicEvidenceState
    predicted_effect: bool
    execution_instruction: ObjectIdentity

    def __post_init__(self) -> None:
        """核验执行结果的关系、时间、四态和预测状态彼此一致。"""
        if not isinstance(self.fact, ActiveRelationClosureFact):
            raise TypeError("fact 必须是 ActiveRelationClosureFact")
        if not isinstance(self.temporal_result, EventTimeVerificationResult):
            raise TypeError("temporal_result 类型非法")
        if not isinstance(self.temporal_assessment, CausalTemporalAssessment):
            raise TypeError("temporal_assessment 类型非法")
        if not isinstance(self.cause, CausalEndpointEvaluation):
            raise TypeError("cause evaluation 类型非法")
        if not isinstance(self.effect, CausalEndpointEvaluation):
            raise TypeError("effect evaluation 类型非法")
        assert_int(self.status, _where="CausalExecutionResult.status")
        if self.status not in _EXECUTION_STATES:
            raise ValueError("causal execution status 未注册")
        if not isinstance(self.effect_state, LogicEvidenceState):
            raise TypeError("effect_state 必须是 LogicEvidenceState")
        if type(self.predicted_effect) is not bool:
            raise TypeError("predicted_effect 必须是 bool")
        _identity(
            self.execution_instruction,
            OBJECT_MINIMAL_INSTRUCTION,
            where="execution instruction",
        )
        validate_temporal_assessment(
            self.temporal_result,
            self.temporal_assessment,
        )
        if (self.cause.evaluation.source != self.effect.evaluation.source
                or self.cause.evaluation.scope != self.effect.evaluation.scope):
            raise ValueError("cause/effect S-04 evaluation 必须属于同一 query")
        if self.predicted_effect != self.effect_state.support:
            raise ValueError("predicted_effect 与 effect support 位不一致")
        if self.predicted_effect and self.status not in {
                CAUSAL_EXECUTION_PREDICTED,
                CAUSAL_EXECUTION_CONFLICTED,
        }:
            raise ValueError("产生 effect prediction 时 execution 状态非法")
        if (not self.predicted_effect
                and self.status == CAUSAL_EXECUTION_PREDICTED):
            raise ValueError("predicted execution 必须实际产生 effect prediction")

    def stable_key(self) -> tuple[int, ...]:
        """返回关系归因、时间 artifact 和 query 执行的完整比较键。"""
        proposition = self.fact.proposition.proposition.stable_key()
        hypothesis = self.fact.hypothesis.stable_key()
        values: list[int] = [
            *_pack(proposition),
            *_pack(hypothesis),
            len(self.fact.evidence_keys),
        ]
        for key in self.fact.evidence_keys:
            values.extend(_pack(key))
        values.extend((
            *_pack(self.fact.decision_key),
            *_pack(_event_time_result_key(self.temporal_result)),
            *_pack(self.temporal_assessment.stable_key()),
            *_pack(self.cause.stable_key()),
            *_pack(self.effect.stable_key()),
            self.status,
            *self.effect_state.stable_key(),
            int(self.predicted_effect),
        ))
        values.extend(_pack(self.execution_instruction.stable_key()))
        return tuple(values)


class CausalExecutor:
    """消费 active CAUSES 和 S-04 四态结果，产生可废止 effect prediction。"""

    def __init__(
            self,
            protocol: CausalEndpointProtocol,
            temporal_resolver: CausalTemporalResolver,
            ) -> None:
        """绑定注入式端点协议和 event-time 到 causal 的裁决器。"""
        if not isinstance(protocol, CausalEndpointProtocol):
            raise TypeError("protocol 必须是 CausalEndpointProtocol")
        if not callable(getattr(temporal_resolver, "resolve", None)):
            raise TypeError("temporal_resolver 必须实现 resolve")
        self.protocol = protocol
        self.temporal_resolver = temporal_resolver

    def execute(
            self,
            fact: ActiveRelationClosureFact,
            temporal_result: EventTimeVerificationResult,
            cause: CausalEndpointEvaluation,
            effect: CausalEndpointEvaluation,
            ) -> CausalExecutionResult:
        """核验端点映射和时间约束后执行一次 provisional 因果传播。"""
        if not isinstance(fact, ActiveRelationClosureFact):
            raise TypeError("fact 必须是 ActiveRelationClosureFact")
        if not isinstance(cause, CausalEndpointEvaluation):
            raise TypeError("cause 必须是 CausalEndpointEvaluation")
        if not isinstance(effect, CausalEndpointEvaluation):
            raise TypeError("effect 必须是 CausalEndpointEvaluation")
        expected_cause, expected_effect = causal_endpoints(
            fact.proposition,
            self.protocol,
        )
        if cause.endpoint != expected_cause or effect.endpoint != expected_effect:
            raise ValueError("query endpoint binding 替换了 active CAUSES 端点")
        if (cause.evaluation.source != effect.evaluation.source
                or cause.evaluation.scope != effect.evaluation.scope):
            raise ValueError("cause/effect S-04 evaluation 必须属于同一 query")
        assessment = self.temporal_resolver.resolve(
            expected_cause,
            expected_effect,
            temporal_result,
        )
        assessment = validate_temporal_assessment(
            temporal_result,
            assessment,
        )

        predicted = False
        effect_state = LogicEvidenceState(False, False)
        if assessment.status == CAUSAL_TEMPORAL_CONFLICTED:
            status = CAUSAL_EXECUTION_CONFLICTED
        elif assessment.status == CAUSAL_TEMPORAL_REJECTED:
            status = CAUSAL_EXECUTION_TEMPORAL_REJECTED
        elif assessment.status != CAUSAL_TEMPORAL_ACCEPTED:
            status = CAUSAL_EXECUTION_UNKNOWN
        elif cause.evaluation.state.support and cause.evaluation.state.refute:
            status = CAUSAL_EXECUTION_CONFLICTED
        elif not cause.evaluation.state.support:
            status = CAUSAL_EXECUTION_UNKNOWN
        else:
            predicted = True
            effect_state = LogicEvidenceState(
                True,
                effect.evaluation.state.refute,
            )
            status = (
                CAUSAL_EXECUTION_CONFLICTED
                if effect_state.refute
                else CAUSAL_EXECUTION_PREDICTED
            )
        return CausalExecutionResult(
            fact,
            temporal_result,
            assessment,
            cause,
            effect,
            status,
            effect_state,
            predicted,
            self.protocol.execution_instruction,
        )


__all__ = [
    "CAUSAL_EXECUTION_CONFLICTED",
    "CAUSAL_EXECUTION_PREDICTED",
    "CAUSAL_EXECUTION_TEMPORAL_REJECTED",
    "CAUSAL_EXECUTION_UNKNOWN",
    "CAUSAL_TEMPORAL_ACCEPTED",
    "CAUSAL_TEMPORAL_CONFLICTED",
    "CAUSAL_TEMPORAL_REJECTED",
    "CAUSAL_TEMPORAL_UNKNOWN",
    "CausalEndpointEvaluation",
    "CausalEndpointProtocol",
    "CausalExecutionResult",
    "CausalExecutor",
    "CausalTemporalAssessment",
    "CausalTemporalResolver",
    "causal_endpoints",
    "validate_temporal_assessment",
]
