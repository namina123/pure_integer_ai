"""S-05 多步目标分解、竞争分支和 source-bearing reasoning trace。

本模块把候选检索、规则验证和规划调度分开。dag_path、PR 或其他启发式只能经
CandidateRetriever 提交候选，不能决定逻辑有效性；InferenceVerifier 必须对每个
候选给出带来源的四态结果。Planner 不保存 query activation，也不写 Core/Memory。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_PROPOSITION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    LogicEvaluation,
    LogicEvidenceState,
    LogicExecutor,
    ModalResolver,
    QuantifierResolver,
    STATE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BoundProposition,
    PropositionTemplateGraph,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _require_kind(
        identity: ObjectIdentity, expected: int, *, label: str,
        ) -> ObjectIdentity:
    """核验一等对象类型，不从名称或 tuple 位置猜语义。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != expected:
        raise ValueError(f"{label} 对象类型不匹配")
    return identity


def _strict_int_tuple(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验开放整数 tuple，禁止 bool 和其他标量混入稳定键。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键加长度前缀，避免拼接边界碰撞。"""
    return len(key), *key


def _identity_key(identity: ObjectIdentity | None) -> tuple[int, ...]:
    """把可选一等身份转换为带空值边界的稳定键。"""
    return () if identity is None else identity.stable_key()


def _state_merge(states: tuple[LogicEvidenceState, ...]) -> LogicEvidenceState:
    """合并同一目标的竞争证据，任一支持和任一反驳都必须保留。"""
    return LogicEvidenceState(
        any(state.support for state in states),
        any(state.refute for state in states),
    )


@dataclass(frozen=True)
class ReasoningBudget:
    """一次规划允许的逻辑求值、规则展开和递归深度上限。"""

    max_evaluations: int
    max_expansions: int
    max_depth: int

    def __post_init__(self) -> None:
        assert_int(
            self.max_evaluations,
            self.max_expansions,
            self.max_depth,
            _where="ReasoningBudget",
        )
        if any(type(value) is not int for value in (
                self.max_evaluations,
                self.max_expansions,
                self.max_depth)):
            raise ValueError("reasoning budget 必须使用严格整数")
        if self.max_evaluations <= 0:
            raise ValueError("max_evaluations 必须为正整数")
        if self.max_expansions < 0 or self.max_depth < 0:
            raise ValueError("max_expansions/max_depth 必须为非负整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回冻结预算键，供结果和回放核验。"""
        return self.max_evaluations, self.max_expansions, self.max_depth


@dataclass(frozen=True)
class ReasoningTerminationProtocol:
    """注入规划指令和终止原因，不冻结 reason 数值或文字。"""

    evaluate_instruction: ObjectIdentity
    resolved: ObjectIdentity
    unresolved: ObjectIdentity
    budget_exhausted: ObjectIdentity
    cycle: ObjectIdentity
    verifier_unknown: ObjectIdentity

    def __post_init__(self) -> None:
        identities = self.identities()
        if len(set(identities)) != len(identities):
            raise ValueError("reasoning instruction/reason 必须互不相同")
        for identity in identities:
            _require_kind(
                identity,
                OBJECT_MINIMAL_INSTRUCTION,
                label="reasoning instruction/reason",
            )

    def identities(self) -> tuple[ObjectIdentity, ...]:
        """返回全部协议槽位，供台账和测试统一检查。"""
        return (
            self.evaluate_instruction,
            self.resolved,
            self.unresolved,
            self.budget_exhausted,
            self.cycle,
            self.verifier_unknown,
        )


@dataclass(frozen=True)
class ReasoningObligation:
    """一个 typed 目标命题及所需 support/refute 证据方向。"""

    proposition: BoundProposition
    required: LogicEvidenceState
    source: SourceRef
    scope: ScopeIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.proposition, BoundProposition):
            raise TypeError("obligation proposition 必须是 BoundProposition")
        if not isinstance(self.required, LogicEvidenceState):
            raise TypeError("obligation required 必须是 LogicEvidenceState")
        if not self.required.support and not self.required.refute:
            raise ValueError("obligation 至少要求一个证据方向")
        if not isinstance(self.source, SourceRef):
            raise TypeError("obligation source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("obligation scope 必须是 ScopeIdentity")
        if semantic_source(self.proposition.template) != self.source:
            raise ValueError("obligation Proposition 与 source 不一致")

    def is_satisfied_by(self, state: LogicEvidenceState) -> bool:
        """判断状态是否包含本 obligation 要求的全部证据位。"""
        if not isinstance(state, LogicEvidenceState):
            raise TypeError("state 必须是 LogicEvidenceState")
        return (
            (not self.required.support or state.support)
            and (not self.required.refute or state.refute)
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 bound goal、所需证据、source 和 scope 的完整运行期键。"""
        return (
            *_packed(self.proposition.stable_key()),
            *self.required.stable_key(),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
        )


@dataclass(frozen=True)
class ReasoningCandidate:
    """候选检索器提交的规则应用，不因路径或 rank 存在而自动成立。"""

    conclusion: ReasoningObligation
    premises: tuple[ReasoningObligation, ...]
    rule: ObjectIdentity
    instruction: ObjectIdentity
    hypothesis: HypothesisKey
    evidence_ids: tuple[int, ...]
    assumptions: tuple[HypothesisKey, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.conclusion, ReasoningObligation):
            raise TypeError("candidate conclusion 必须是 ReasoningObligation")
        if not isinstance(self.premises, tuple):
            raise TypeError("candidate premises 必须是 ReasoningObligation tuple")
        if any(not isinstance(item, ReasoningObligation)
               for item in self.premises):
            raise TypeError("candidate premises 含非法项")
        _require_kind(self.rule, OBJECT_STRUCTURE_CONCEPT, label="candidate rule")
        _require_kind(
            self.instruction,
            OBJECT_MINIMAL_INSTRUCTION,
            label="candidate instruction",
        )
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("candidate hypothesis 必须是 HypothesisKey")
        _strict_int_tuple(self.evidence_ids, label="candidate evidence_ids")
        if not self.evidence_ids or any(item <= 0 for item in self.evidence_ids):
            raise ValueError("candidate 必须带正整数 Evidence id")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("candidate Evidence id 不得重复")
        if not isinstance(self.assumptions, tuple):
            raise TypeError("candidate assumptions 必须是 HypothesisKey tuple")
        if any(not isinstance(item, HypothesisKey) for item in self.assumptions):
            raise TypeError("candidate assumptions 含非法项")
        if len(set(self.assumptions)) != len(self.assumptions):
            raise ValueError("candidate assumptions 不得重复")
        hypotheses = (self.hypothesis, *self.assumptions)
        for hypothesis in hypotheses:
            if hypothesis.observation != self.conclusion.source:
                raise ValueError("candidate hypothesis 与 conclusion source 不一致")
            if hypothesis.scope != self.conclusion.scope:
                raise ValueError("candidate hypothesis 与 conclusion scope 不一致")
        object.__setattr__(self, "evidence_ids", tuple(sorted(self.evidence_ids)))
        object.__setattr__(self, "assumptions", tuple(sorted(self.assumptions)))

    def stable_key(self) -> tuple[int, ...]:
        """返回结论、前提、规则和全部来源证据的稳定候选键。"""
        result = [
            *_packed(self.conclusion.stable_key()),
            *_packed(self.rule.stable_key()),
            *_packed(self.instruction.stable_key()),
            *_packed(self.hypothesis.stable_key()),
            *_packed(self.evidence_ids),
            len(self.premises),
        ]
        for premise in self.premises:
            result.extend(_packed(premise.stable_key()))
        result.append(len(self.assumptions))
        for assumption in self.assumptions:
            result.extend(_packed(assumption.stable_key()))
        return tuple(result)


class ReasoningCandidateRetriever(Protocol):
    """按完整 obligation 返回候选；路径和 PR adapter 只能实现本协议。"""

    def retrieve(
            self, obligation: ReasoningObligation,
            ) -> tuple[ReasoningCandidate, ...]:
        """返回全部竞争候选，不能先选一条后丢弃其他分支。"""
        ...


class CompositeCandidateRetriever:
    """合并多个候选设施并按完整键去重，不读取 rank 作为有效性。"""

    def __init__(self, providers: tuple[ReasoningCandidateRetriever, ...]) -> None:
        if not isinstance(providers, tuple):
            raise TypeError("candidate providers 必须是 tuple")
        if any(not hasattr(provider, "retrieve") for provider in providers):
            raise TypeError("candidate provider 必须实现 retrieve")
        self._providers = providers

    def retrieve(
            self, obligation: ReasoningObligation,
            ) -> tuple[ReasoningCandidate, ...]:
        """汇总完整候选；稳定键碰撞但内容不同视为协议错误。"""
        by_key: dict[tuple[int, ...], ReasoningCandidate] = {}
        for provider in self._providers:
            candidates = provider.retrieve(obligation)
            if not isinstance(candidates, tuple):
                raise TypeError("candidate provider 必须返回 tuple")
            for candidate in candidates:
                if not isinstance(candidate, ReasoningCandidate):
                    raise TypeError("candidate provider 返回非法项")
                if candidate.conclusion != obligation:
                    raise ValueError("candidate conclusion 与查询 obligation 不一致")
                key = candidate.stable_key()
                previous = by_key.get(key)
                if previous is not None and previous != candidate:
                    raise ValueError("reasoning candidate 稳定键碰撞")
                by_key[key] = candidate
        return tuple(by_key[key] for key in sorted(by_key))


@dataclass(frozen=True)
class RuleVerification:
    """InferenceVerifier 对一个精确候选给出的受限四态结果。"""

    candidate: HypothesisKey
    state: LogicEvidenceState
    source: SourceRef
    scope: ScopeIdentity
    evidence_ids: tuple[int, ...] = ()
    hypotheses: tuple[HypothesisKey, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, HypothesisKey):
            raise TypeError("verification candidate 必须是 HypothesisKey")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("verification state 必须是 LogicEvidenceState")
        if not isinstance(self.source, SourceRef):
            raise TypeError("verification source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("verification scope 必须是 ScopeIdentity")
        _strict_int_tuple(self.evidence_ids, label="verification evidence_ids")
        if any(item <= 0 for item in self.evidence_ids):
            raise ValueError("verification Evidence id 必须为正整数")
        if self.state.status != STATE_UNKNOWN and not self.evidence_ids:
            raise ValueError("非 unknown verification 必须带 Evidence id")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("verification Evidence id 不得重复")
        if not isinstance(self.hypotheses, tuple):
            raise TypeError("verification hypotheses 必须是 tuple")
        if any(not isinstance(item, HypothesisKey) for item in self.hypotheses):
            raise TypeError("verification hypotheses 含非法项")
        for hypothesis in (self.candidate, *self.hypotheses):
            if hypothesis.observation != self.source:
                raise ValueError("verification hypothesis 与 source 不一致")
            if hypothesis.scope != self.scope:
                raise ValueError("verification hypothesis 与 scope 不一致")
        object.__setattr__(
            self, "evidence_ids", tuple(sorted(self.evidence_ids)))
        object.__setattr__(
            self, "hypotheses", tuple(sorted(set(self.hypotheses))))


class InferenceVerifier(Protocol):
    """验证候选规则应用，不接受路径到达或 salience 代替有效性。"""

    def verify(
            self,
            candidate: ReasoningCandidate,
            premises: tuple["ReasoningPlanResult", ...],
            ) -> RuleVerification:
        """返回与精确 candidate/source/scope 对齐的四态验证结果。"""
        ...


class ObligationEvaluator(Protocol):
    """把一个 obligation 交给 S-04 或同构 typed evaluator。"""

    def evaluate(self, obligation: ReasoningObligation) -> LogicEvaluation:
        """返回直接 Evidence 视图；命题未知时仍返回 S-04 unknown。"""
        ...


class LogicObligationEvaluator:
    """把固定运行期 graph/environment 注入 S-04 LogicExecutor。"""

    def __init__(
            self,
            executor: LogicExecutor,
            graph: PropositionTemplateGraph,
            environment: BindingEnvironment,
            *,
            quantifier_resolver: QuantifierResolver | None = None,
            modal_resolver: ModalResolver | None = None,
            inherited_binders: tuple[ObjectIdentity, ...] = (),
            ) -> None:
        if not isinstance(executor, LogicExecutor):
            raise TypeError("executor 必须是 LogicExecutor")
        if not isinstance(graph, PropositionTemplateGraph):
            raise TypeError("graph 必须是 PropositionTemplateGraph")
        if not isinstance(environment, BindingEnvironment):
            raise TypeError("environment 必须是 BindingEnvironment")
        if not isinstance(inherited_binders, tuple):
            raise TypeError("inherited_binders 必须是 Binder tuple")
        self._executor = executor
        self._graph = graph
        self._environment = environment
        self._quantifier_resolver = quantifier_resolver
        self._modal_resolver = modal_resolver
        self._inherited_binders = inherited_binders

    def evaluate(self, obligation: ReasoningObligation) -> LogicEvaluation:
        """使用 obligation 自身 source/scope 求值，不缓存 query activation。"""
        if not isinstance(obligation, ReasoningObligation):
            raise TypeError("obligation 类型错误")
        return self._executor.evaluate(
            obligation.proposition,
            source=obligation.source,
            scope=obligation.scope,
            graph=self._graph,
            environment=self._environment,
            quantifier_resolver=self._quantifier_resolver,
            modal_resolver=self._modal_resolver,
            inherited_binders=self._inherited_binders,
        )


@dataclass(frozen=True)
class ReasoningStep:
    """一次直接求值、规则应用或循环终止的顺序化推理步骤。"""

    ordinal: int
    depth: int
    instruction: ObjectIdentity
    obligation: ReasoningObligation
    state: LogicEvidenceState
    source: SourceRef
    scope: ScopeIdentity
    candidate: HypothesisKey | None = None
    rule: ObjectIdentity | None = None
    premises: tuple[ObjectIdentity, ...] = ()
    evidence_ids: tuple[int, ...] = ()
    hypotheses: tuple[HypothesisKey, ...] = ()
    stack_keys: tuple[tuple[int, ...], ...] = ()
    logic_evaluation: LogicEvaluation | None = None

    def __post_init__(self) -> None:
        assert_int(self.ordinal, self.depth, _where="ReasoningStep")
        if type(self.ordinal) is not int or self.ordinal <= 0:
            raise ValueError("reasoning step ordinal 必须为正整数")
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError("reasoning step depth 必须为非负整数")
        _require_kind(
            self.instruction,
            OBJECT_MINIMAL_INSTRUCTION,
            label="reasoning step instruction",
        )
        if not isinstance(self.obligation, ReasoningObligation):
            raise TypeError("reasoning step obligation 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("reasoning step state 类型错误")
        if self.source != self.obligation.source or self.scope != self.obligation.scope:
            raise ValueError("reasoning step source/scope 与 obligation 不一致")
        if self.candidate is not None and not isinstance(
                self.candidate, HypothesisKey):
            raise TypeError("reasoning step candidate 类型错误")
        if self.rule is not None:
            _require_kind(
                self.rule,
                OBJECT_STRUCTURE_CONCEPT,
                label="reasoning step rule",
            )
        if not isinstance(self.premises, tuple):
            raise TypeError("reasoning step premises 必须是 tuple")
        for premise in self.premises:
            _require_kind(premise, OBJECT_PROPOSITION, label="reasoning premise")
        _strict_int_tuple(self.evidence_ids, label="reasoning step evidence_ids")
        if any(item <= 0 for item in self.evidence_ids):
            raise ValueError("reasoning step Evidence id 必须为正整数")
        if any(not isinstance(item, HypothesisKey) for item in self.hypotheses):
            raise TypeError("reasoning step hypotheses 含非法项")
        if not isinstance(self.stack_keys, tuple):
            raise TypeError("reasoning step stack_keys 必须是 tuple")
        for key in self.stack_keys:
            _strict_int_tuple(key, label="reasoning stack key")
        if self.logic_evaluation is not None:
            if not isinstance(self.logic_evaluation, LogicEvaluation):
                raise TypeError("logic_evaluation 类型错误")
            if self.logic_evaluation.proposition != self.obligation.proposition:
                raise ValueError("logic_evaluation 与 step obligation 不一致")
            if (self.logic_evaluation.state != self.state
                    or self.logic_evaluation.source != self.source
                    or self.logic_evaluation.scope != self.scope):
                raise ValueError("logic_evaluation 与 step 状态或来源不一致")
        object.__setattr__(
            self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        object.__setattr__(
            self, "hypotheses", tuple(sorted(set(self.hypotheses))))

    def stable_key(self) -> tuple[int, ...]:
        """返回顺序、栈、规则和全部来源证据的稳定步骤键。"""
        result = [
            self.ordinal,
            self.depth,
            *_packed(self.instruction.stable_key()),
            *_packed(self.obligation.stable_key()),
            *self.state.stable_key(),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(() if self.candidate is None else self.candidate.stable_key()),
            *_packed(_identity_key(self.rule)),
            len(self.premises),
        ]
        for premise in self.premises:
            result.extend(_packed(premise.stable_key()))
        result.extend(_packed(self.evidence_ids))
        result.append(len(self.hypotheses))
        for hypothesis in self.hypotheses:
            result.extend(_packed(hypothesis.stable_key()))
        result.append(len(self.stack_keys))
        for key in self.stack_keys:
            result.extend(_packed(key))
        result.extend(_packed(
            () if self.logic_evaluation is None
            else self.logic_evaluation.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class ReasoningPremiseOutcome:
    """一个候选前提的状态、完成性和终止原因摘要。"""

    obligation: ReasoningObligation
    state: LogicEvidenceState
    complete: bool
    termination: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.obligation, ReasoningObligation):
            raise TypeError("premise outcome obligation 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("premise outcome state 类型错误")
        if type(self.complete) is not bool:
            raise TypeError("premise outcome complete 必须是严格 bool")
        _require_kind(
            self.termination,
            OBJECT_MINIMAL_INSTRUCTION,
            label="premise outcome termination",
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回前提结果摘要的稳定键。"""
        return (
            *_packed(self.obligation.stable_key()),
            *self.state.stable_key(),
            int(self.complete),
            *_packed(self.termination.stable_key()),
        )


@dataclass(frozen=True)
class ReasoningBranchResult:
    """一个竞争候选的全部前提结果、验证状态和来源账。"""

    candidate: ReasoningCandidate
    premises: tuple[ReasoningPremiseOutcome, ...]
    state: LogicEvidenceState
    complete: bool
    termination: ObjectIdentity
    evidence_ids: tuple[int, ...] = ()
    hypotheses: tuple[HypothesisKey, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ReasoningCandidate):
            raise TypeError("branch candidate 类型错误")
        if not isinstance(self.premises, tuple):
            raise TypeError("branch premises 必须是 tuple")
        if any(not isinstance(item, ReasoningPremiseOutcome)
               for item in self.premises):
            raise TypeError("branch premises 含非法项")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("branch state 类型错误")
        if type(self.complete) is not bool:
            raise TypeError("branch complete 必须是严格 bool")
        _require_kind(
            self.termination,
            OBJECT_MINIMAL_INSTRUCTION,
            label="branch termination",
        )
        _strict_int_tuple(self.evidence_ids, label="branch evidence_ids")
        if any(item <= 0 for item in self.evidence_ids):
            raise ValueError("branch Evidence id 必须为正整数")
        if any(not isinstance(item, HypothesisKey) for item in self.hypotheses):
            raise TypeError("branch hypotheses 含非法项")
        object.__setattr__(
            self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        object.__setattr__(
            self, "hypotheses", tuple(sorted(set(self.hypotheses))))

    def stable_key(self) -> tuple[int, ...]:
        """返回候选、前提、状态、终止原因和来源账的稳定键。"""
        result = [
            *_packed(self.candidate.stable_key()),
            *self.state.stable_key(),
            int(self.complete),
            *_packed(self.termination.stable_key()),
            len(self.premises),
        ]
        for premise in self.premises:
            result.extend(_packed(premise.stable_key()))
        result.extend(_packed(self.evidence_ids))
        result.append(len(self.hypotheses))
        for hypothesis in self.hypotheses:
            result.extend(_packed(hypothesis.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class ReasoningPlanResult:
    """根或子目标的最终四态、未解义务、竞争分支和有序 trace。"""

    obligation: ReasoningObligation
    state: LogicEvidenceState
    complete: bool
    termination: ObjectIdentity
    budget: ReasoningBudget
    evaluations_used: int
    expansions_used: int
    steps: tuple[ReasoningStep, ...] = ()
    branches: tuple[ReasoningBranchResult, ...] = ()
    unresolved: tuple[ReasoningObligation, ...] = ()
    evidence_ids: tuple[int, ...] = ()
    hypotheses: tuple[HypothesisKey, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.obligation, ReasoningObligation):
            raise TypeError("plan obligation 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("plan state 类型错误")
        if type(self.complete) is not bool:
            raise TypeError("plan complete 必须是严格 bool")
        _require_kind(
            self.termination,
            OBJECT_MINIMAL_INSTRUCTION,
            label="plan termination",
        )
        if not isinstance(self.budget, ReasoningBudget):
            raise TypeError("plan budget 类型错误")
        assert_int(
            self.evaluations_used,
            self.expansions_used,
            _where="ReasoningPlanResult",
        )
        if (type(self.evaluations_used) is not int
                or type(self.expansions_used) is not int
                or self.evaluations_used < 0
                or self.expansions_used < 0):
            raise ValueError("plan 使用量必须为非负严格整数")
        if self.evaluations_used > self.budget.max_evaluations:
            raise ValueError("plan evaluations_used 超过冻结预算")
        if self.expansions_used > self.budget.max_expansions:
            raise ValueError("plan expansions_used 超过冻结预算")
        if any(not isinstance(item, ReasoningStep) for item in self.steps):
            raise TypeError("plan steps 含非法项")
        if any(not isinstance(item, ReasoningBranchResult)
               for item in self.branches):
            raise TypeError("plan branches 含非法项")
        if any(not isinstance(item, ReasoningObligation)
               for item in self.unresolved):
            raise TypeError("plan unresolved 含非法项")
        _strict_int_tuple(self.evidence_ids, label="plan evidence_ids")
        if any(item <= 0 for item in self.evidence_ids):
            raise ValueError("plan Evidence id 必须为正整数")
        if any(not isinstance(item, HypothesisKey) for item in self.hypotheses):
            raise TypeError("plan hypotheses 含非法项")
        ordinals = tuple(step.ordinal for step in self.steps)
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("plan step ordinal 不得重复")
        if ordinals != tuple(sorted(ordinals)):
            raise ValueError("plan steps 必须保持实际执行顺序")
        object.__setattr__(
            self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        object.__setattr__(
            self, "hypotheses", tuple(sorted(set(self.hypotheses))))
        object.__setattr__(self, "unresolved", tuple(sorted(
            set(self.unresolved), key=lambda item: item.stable_key())))

    @property
    def goal_satisfied(self) -> bool:
        """返回最终状态是否包含根 obligation 要求的证据位。"""
        return self.obligation.is_satisfied_by(self.state)

    def stable_key(self) -> tuple[int, ...]:
        """返回本次纯规划结果的完整稳定键，不作为 Core identity。"""
        result = [
            *_packed(self.obligation.stable_key()),
            *self.state.stable_key(),
            int(self.complete),
            *_packed(self.termination.stable_key()),
            *self.budget.stable_key(),
            self.evaluations_used,
            self.expansions_used,
            len(self.steps),
        ]
        for step in self.steps:
            result.extend(_packed(step.stable_key()))
        result.append(len(self.branches))
        for branch in self.branches:
            result.extend(_packed(branch.stable_key()))
        result.append(len(self.unresolved))
        for obligation in self.unresolved:
            result.extend(_packed(obligation.stable_key()))
        result.extend(_packed(self.evidence_ids))
        result.append(len(self.hypotheses))
        for hypothesis in self.hypotheses:
            result.extend(_packed(hypothesis.stable_key()))
        return tuple(result)


class ReasoningPlanner:
    """在冻结预算内穷尽竞争候选并组合多步 typed obligation。"""

    def __init__(
            self,
            evaluator: ObligationEvaluator,
            retriever: ReasoningCandidateRetriever,
            verifier: InferenceVerifier,
            protocol: ReasoningTerminationProtocol,
            ) -> None:
        if not hasattr(evaluator, "evaluate"):
            raise TypeError("evaluator 必须实现 evaluate")
        if not hasattr(retriever, "retrieve"):
            raise TypeError("retriever 必须实现 retrieve")
        if not hasattr(verifier, "verify"):
            raise TypeError("verifier 必须实现 verify")
        if not isinstance(protocol, ReasoningTerminationProtocol):
            raise TypeError("protocol 类型错误")
        self._evaluator = evaluator
        self._retriever = retriever
        self._verifier = verifier
        self._protocol = protocol

    def plan(
            self,
            goal: ReasoningObligation,
            budget: ReasoningBudget,
            ) -> ReasoningPlanResult:
        """建立局部 session；预算、活动栈和 step 序不跨 plan 调用复用。"""
        if not isinstance(goal, ReasoningObligation):
            raise TypeError("goal 必须是 ReasoningObligation")
        if not isinstance(budget, ReasoningBudget):
            raise TypeError("budget 必须是 ReasoningBudget")
        session = _PlanningSession(
            self._evaluator,
            self._retriever,
            self._verifier,
            self._protocol,
            budget,
        )
        return session.solve(goal, depth=0, stack=())


class _PlanningSession:
    """单次 plan 的局部预算计数和深度优先执行状态。"""

    def __init__(
            self,
            evaluator: ObligationEvaluator,
            retriever: ReasoningCandidateRetriever,
            verifier: InferenceVerifier,
            protocol: ReasoningTerminationProtocol,
            budget: ReasoningBudget,
            ) -> None:
        self.evaluator = evaluator
        self.retriever = retriever
        self.verifier = verifier
        self.protocol = protocol
        self.budget = budget
        self.evaluations_used = 0
        self.expansions_used = 0
        self.next_ordinal = 1

    def _step(
            self,
            *,
            depth: int,
            instruction: ObjectIdentity,
            obligation: ReasoningObligation,
            state: LogicEvidenceState,
            candidate: HypothesisKey | None = None,
            rule: ObjectIdentity | None = None,
            premises: tuple[ObjectIdentity, ...] = (),
            evidence_ids: tuple[int, ...] = (),
            hypotheses: tuple[HypothesisKey, ...] = (),
            stack_keys: tuple[tuple[int, ...], ...] = (),
            logic_evaluation: LogicEvaluation | None = None,
            ) -> ReasoningStep:
        """分配本 session 单调 step ordinal 并构造不可变 trace 项。"""
        step = ReasoningStep(
            self.next_ordinal,
            depth,
            instruction,
            obligation,
            state,
            obligation.source,
            obligation.scope,
            candidate,
            rule,
            premises,
            evidence_ids,
            hypotheses,
            stack_keys,
            logic_evaluation,
        )
        self.next_ordinal += 1
        return step

    def _result(
            self,
            obligation: ReasoningObligation,
            state: LogicEvidenceState,
            complete: bool,
            termination: ObjectIdentity,
            *,
            steps: tuple[ReasoningStep, ...] = (),
            branches: tuple[ReasoningBranchResult, ...] = (),
            unresolved: tuple[ReasoningObligation, ...] = (),
            evidence_ids: tuple[int, ...] = (),
            hypotheses: tuple[HypothesisKey, ...] = (),
            ) -> ReasoningPlanResult:
        """使用当前全局预算计数构造根或子目标结果快照。"""
        return ReasoningPlanResult(
            obligation,
            state,
            complete,
            termination,
            self.budget,
            self.evaluations_used,
            self.expansions_used,
            steps,
            branches,
            unresolved,
            evidence_ids,
            hypotheses,
        )

    def _budget_result(
            self,
            obligation: ReasoningObligation,
            *,
            steps: tuple[ReasoningStep, ...] = (),
            branches: tuple[ReasoningBranchResult, ...] = (),
            unresolved: tuple[ReasoningObligation, ...] = (),
            evidence_ids: tuple[int, ...] = (),
            hypotheses: tuple[HypothesisKey, ...] = (),
            ) -> ReasoningPlanResult:
        """预算未覆盖全部竞争边界时强制 unknown，保留已完成的部分 trace。"""
        return self._result(
            obligation,
            LogicEvidenceState(False, False),
            False,
            self.protocol.budget_exhausted,
            steps=steps,
            branches=branches,
            unresolved=(obligation, *unresolved),
            evidence_ids=evidence_ids,
            hypotheses=hypotheses,
        )

    def solve(
            self,
            obligation: ReasoningObligation,
            *,
            depth: int,
            stack: tuple[tuple[int, ...], ...],
            ) -> ReasoningPlanResult:
        """深度优先展开全部候选；循环局部失败不阻断同层其他竞争分支。"""
        obligation_key = obligation.stable_key()
        if obligation_key in stack:
            step = self._step(
                depth=depth,
                instruction=self.protocol.cycle,
                obligation=obligation,
                state=LogicEvidenceState(False, False),
                stack_keys=stack,
            )
            return self._result(
                obligation,
                LogicEvidenceState(False, False),
                True,
                self.protocol.cycle,
                steps=(step,),
                unresolved=(obligation,),
            )
        if self.evaluations_used >= self.budget.max_evaluations:
            return self._budget_result(obligation)

        evaluation = self.evaluator.evaluate(obligation)
        if not isinstance(evaluation, LogicEvaluation):
            raise TypeError("obligation evaluator 必须返回 LogicEvaluation")
        if evaluation.proposition != obligation.proposition:
            raise ValueError("logic evaluation 返回了其他 Bound Proposition")
        if evaluation.source != obligation.source or evaluation.scope != obligation.scope:
            raise ValueError("logic evaluation 与 obligation source/scope 不一致")
        self.evaluations_used += 1
        direct_step = self._step(
            depth=depth,
            instruction=self.protocol.evaluate_instruction,
            obligation=obligation,
            state=evaluation.state,
            evidence_ids=evaluation.evidence_ids,
            hypotheses=evaluation.hypotheses,
            stack_keys=stack,
            logic_evaluation=evaluation,
        )
        steps: list[ReasoningStep] = [direct_step]
        branches: list[ReasoningBranchResult] = []
        all_evidence = set(evaluation.evidence_ids)
        all_hypotheses = set(evaluation.hypotheses)
        all_unresolved: set[ReasoningObligation] = set()
        candidate_states: list[LogicEvidenceState] = [evaluation.state]

        candidates = self.retriever.retrieve(obligation)
        if not isinstance(candidates, tuple):
            raise TypeError("candidate retriever 必须返回 tuple")
        normalized = CompositeCandidateRetriever(
            (_StaticCandidateProvider(candidates),)).retrieve(obligation)
        if not normalized:
            termination = (
                self.protocol.resolved
                if evaluation.state.status != STATE_UNKNOWN
                else self.protocol.unresolved)
            unresolved = (
                () if evaluation.state.status != STATE_UNKNOWN
                else (obligation,))
            return self._result(
                obligation,
                evaluation.state,
                True,
                termination,
                steps=tuple(steps),
                unresolved=unresolved,
                evidence_ids=tuple(all_evidence),
                hypotheses=tuple(all_hypotheses),
            )
        if depth >= self.budget.max_depth:
            return self._budget_result(
                obligation,
                steps=tuple(steps),
                evidence_ids=tuple(all_evidence),
                hypotheses=tuple(all_hypotheses),
            )

        next_stack = (*stack, obligation_key)
        for candidate in normalized:
            if self.expansions_used >= self.budget.max_expansions:
                return self._budget_result(
                    obligation,
                    steps=tuple(steps),
                    branches=tuple(branches),
                    evidence_ids=tuple(all_evidence),
                    hypotheses=tuple(all_hypotheses),
                )
            self.expansions_used += 1
            premise_results: list[ReasoningPlanResult] = []
            premise_outcomes: list[ReasoningPremiseOutcome] = []
            for premise in candidate.premises:
                result = self.solve(
                    premise,
                    depth=depth + 1,
                    stack=next_stack,
                )
                premise_results.append(result)
                premise_outcomes.append(ReasoningPremiseOutcome(
                    result.obligation,
                    result.state,
                    result.complete,
                    result.termination,
                ))
                steps.extend(result.steps)
                all_evidence.update(result.evidence_ids)
                all_hypotheses.update(result.hypotheses)
                all_unresolved.update(result.unresolved)
                if not result.complete:
                    break
            if any(not result.complete for result in premise_results):
                branch_evidence = set(candidate.evidence_ids)
                branch_hypotheses = {
                    candidate.hypothesis,
                    *candidate.assumptions,
                }
                for result in premise_results:
                    branch_evidence.update(result.evidence_ids)
                    branch_hypotheses.update(result.hypotheses)
                branch = ReasoningBranchResult(
                    candidate,
                    tuple(premise_outcomes),
                    LogicEvidenceState(False, False),
                    False,
                    self.protocol.budget_exhausted,
                    tuple(branch_evidence),
                    tuple(branch_hypotheses),
                )
                branches.append(branch)
                all_evidence.update(branch_evidence)
                all_hypotheses.update(branch_hypotheses)
                return self._budget_result(
                    obligation,
                    steps=tuple(steps),
                    branches=tuple(branches),
                    unresolved=tuple(all_unresolved),
                    evidence_ids=tuple(all_evidence),
                    hypotheses=tuple(all_hypotheses),
                )

            verification = self.verifier.verify(
                candidate, tuple(premise_results))
            if not isinstance(verification, RuleVerification):
                raise TypeError("inference verifier 必须返回 RuleVerification")
            if verification.candidate != candidate.hypothesis:
                raise ValueError("verification 返回了其他 candidate")
            if (verification.source != obligation.source
                    or verification.scope != obligation.scope):
                raise ValueError("verification 与 obligation source/scope 不一致")
            branch_evidence = {
                *candidate.evidence_ids,
                *verification.evidence_ids,
            }
            branch_hypotheses = {
                candidate.hypothesis,
                *candidate.assumptions,
                *verification.hypotheses,
            }
            for result in premise_results:
                branch_evidence.update(result.evidence_ids)
                branch_hypotheses.update(result.hypotheses)
            termination = (
                self.protocol.verifier_unknown
                if verification.state.status == STATE_UNKNOWN
                else self.protocol.resolved)
            branch = ReasoningBranchResult(
                candidate,
                tuple(premise_outcomes),
                verification.state,
                True,
                termination,
                tuple(branch_evidence),
                tuple(branch_hypotheses),
            )
            branches.append(branch)
            candidate_states.append(verification.state)
            all_evidence.update(branch_evidence)
            all_hypotheses.update(branch_hypotheses)
            steps.append(self._step(
                depth=depth,
                instruction=candidate.instruction,
                obligation=obligation,
                state=verification.state,
                candidate=candidate.hypothesis,
                rule=candidate.rule,
                premises=tuple(
                    premise.proposition.template
                    for premise in candidate.premises),
                evidence_ids=tuple(branch_evidence),
                hypotheses=tuple(branch_hypotheses),
                stack_keys=stack,
            ))

        state = _state_merge(tuple(candidate_states))
        termination = (
            self.protocol.resolved
            if state.status != STATE_UNKNOWN
            else self.protocol.unresolved)
        if state.status == STATE_UNKNOWN:
            all_unresolved.add(obligation)
        return self._result(
            obligation,
            state,
            True,
            termination,
            steps=tuple(steps),
            branches=tuple(branches),
            unresolved=tuple(all_unresolved),
            evidence_ids=tuple(all_evidence),
            hypotheses=tuple(all_hypotheses),
        )


class _StaticCandidateProvider:
    """把一次 retriever 返回值送入统一去重校验，不参与生产注册。"""

    def __init__(self, candidates: tuple[ReasoningCandidate, ...]) -> None:
        self._candidates = candidates

    def retrieve(
            self, obligation: ReasoningObligation,
            ) -> tuple[ReasoningCandidate, ...]:
        """返回已取得候选；obligation 一致性由组合器统一核验。"""
        del obligation
        return self._candidates


__all__ = [
    "CompositeCandidateRetriever",
    "InferenceVerifier",
    "LogicObligationEvaluator",
    "ObligationEvaluator",
    "ReasoningBranchResult",
    "ReasoningBudget",
    "ReasoningCandidate",
    "ReasoningCandidateRetriever",
    "ReasoningObligation",
    "ReasoningPlanResult",
    "ReasoningPlanner",
    "ReasoningPremiseOutcome",
    "ReasoningStep",
    "ReasoningTerminationProtocol",
    "RuleVerification",
]
