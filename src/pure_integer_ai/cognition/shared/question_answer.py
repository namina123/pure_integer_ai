"""F-00 typed 问题请求、查询执行和 Evidence 驱动回答选择协议。

本模块不解析 surface，也不拥有生成器。问题必须先携带 S-02/S-03 已形成的
``BoundProposition``、查询/意图指令和来源 scope；查询 executor 只返回真实
``GenerationCandidate`` 与完整 trace，最终 answer/unknown/conflict/clarify 仍由
G-01 ``AnswerContentSelector`` 的共享不变量裁决。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentDecision,
    AnswerContentProtocol,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationCandidate,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    HypothesisLedger,
    HypothesisSnapshot,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长整数键增加长度前缀。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验问题和查询 trace 使用非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _require_instruction(
        identity: ObjectIdentity, *, label: str,
        ) -> ObjectIdentity:
    """核验开放协议身份是一等 MinimalInstruction。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return identity


def _snapshot_key(snapshot: HypothesisSnapshot) -> tuple[int, ...]:
    """编码 H-00 派生快照，保留候选、生命周期、四态和全部 active Evidence。"""
    if not isinstance(snapshot, HypothesisSnapshot):
        raise TypeError("question snapshot 类型错误")
    result = [
        *_packed(snapshot.hypothesis.stable_key()),
        snapshot.lifecycle,
        snapshot.epistemic_status,
    ]
    for evidence_ids in (
            snapshot.support_evidence_ids,
            snapshot.refute_evidence_ids,
            snapshot.unknown_evidence_ids):
        result.append(len(evidence_ids))
        result.extend(evidence_ids)
    return tuple(result)


@dataclass(frozen=True)
class QuestionRequest:
    """一个已完成 typed 语义解析的事实/关系/逻辑问题请求。"""

    query_kind: ObjectIdentity
    intent: ObjectIdentity
    goal_kind: ObjectIdentity
    target: BoundProposition
    required: LogicEvidenceState
    evidence_scope: ScopeIdentity
    response_scope: ScopeIdentity
    trace: tuple[int, ...]
    target_branch: ObjectIdentity | None = None
    authorized_candidate_targets: tuple[BoundProposition, ...] = ()

    def __post_init__(self) -> None:
        """核验查询身份、目标来源和知识/回答 scope 没有被混用。"""
        for identity, label in (
                (self.query_kind, "question query_kind"),
                (self.intent, "question intent"),
                (self.goal_kind, "question goal_kind")):
            _require_instruction(identity, label=label)
        if not isinstance(self.target, BoundProposition):
            raise TypeError("question target 必须是 BoundProposition")
        if not isinstance(self.required, LogicEvidenceState):
            raise TypeError("question required 类型错误")
        if not self.required.support and not self.required.refute:
            raise ValueError("question required 至少需要一个 Evidence 方向")
        if not isinstance(self.evidence_scope, ScopeIdentity):
            raise TypeError("question evidence_scope 类型错误")
        if not isinstance(self.response_scope, ScopeIdentity):
            raise TypeError("question response_scope 类型错误")
        source = semantic_source(self.target.template)
        if self.evidence_scope.source != source:
            raise ValueError("question evidence_scope 必须绑定目标知识来源")
        if self.response_scope.source != source:
            raise ValueError("question response_scope 必须绑定目标知识来源")
        _strict_key(self.trace, label="question request trace")
        if self.target_branch is not None:
            if (not isinstance(self.target_branch, ObjectIdentity)
                    or self.target_branch.object_kind
                    != OBJECT_LANGUAGE_BRANCH):
                raise ValueError("question target_branch 必须是 LanguageBranch")
        if (not isinstance(self.authorized_candidate_targets, tuple)
                or any(not isinstance(item, BoundProposition)
                       for item in self.authorized_candidate_targets)):
            raise TypeError(
                "question authorized candidate targets 类型错误")
        if self.authorized_candidate_targets:
            if self.target not in self.authorized_candidate_targets:
                raise ValueError(
                    "question target 必须属于 authorized candidate targets")
            if len(set(self.authorized_candidate_targets)) != len(
                    self.authorized_candidate_targets):
                raise ValueError(
                    "question authorized candidate targets 不得重复")
            if any(semantic_source(item.template) != source
                   for item in self.authorized_candidate_targets):
                raise ValueError(
                    "question authorized candidate target 来源漂移")
            object.__setattr__(self, "authorized_candidate_targets", tuple(
                sorted(
                    self.authorized_candidate_targets,
                    key=lambda item: item.stable_key(),
                )
            ))

    @property
    def source(self):
        """返回目标 Proposition 的完整知识来源。"""
        return semantic_source(self.target.template)

    def stable_key(self) -> tuple[int, ...]:
        """返回查询/意图、目标、scope、trace 和语言分支的完整稳定键。"""
        result = [
            *_packed(self.query_kind.stable_key()),
            *_packed(self.intent.stable_key()),
            *_packed(self.goal_kind.stable_key()),
            *_packed(self.target.stable_key()),
            *self.required.stable_key(),
            *_packed(self.evidence_scope.stable_key()),
            *_packed(self.response_scope.stable_key()),
            *_packed(self.trace),
            0 if self.target_branch is None else 1,
        ]
        if self.target_branch is not None:
            result.extend(_packed(self.target_branch.stable_key()))
        result.append(len(self.authorized_candidate_targets))
        for target in self.authorized_candidate_targets:
            result.extend(_packed(target.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class QuestionQuery:
    """QuestionRequest 经开放 route 编译后的可执行 Query/Intent。"""

    request: QuestionRequest
    route: ObjectIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 query 保留原请求并用一等指令标识执行 route。"""
        if not isinstance(self.request, QuestionRequest):
            raise TypeError("question query request 类型错误")
        _require_instruction(self.route, label="question query route")
        _strict_key(self.trace, label="question query trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回原请求、route 和编译 trace 的完整稳定键。"""
        request = self.request.stable_key()
        route = self.route.stable_key()
        return (
            *_packed(request),
            *_packed(route),
            *_packed(self.trace),
        )


@dataclass(frozen=True)
class QuestionExecutionResult:
    """一次查询执行返回的 typed 候选、来源化原因和实际检索/执行 trace。"""

    query: QuestionQuery
    reason: ObjectIdentity
    candidates: tuple[GenerationCandidate, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验执行结果只携带当前回答 scope 的真实 G-00 候选。"""
        if not isinstance(self.query, QuestionQuery):
            raise TypeError("question execution query 类型错误")
        _require_instruction(self.reason, label="question execution reason")
        if (not isinstance(self.candidates, tuple)
                or any(not isinstance(item, GenerationCandidate)
                       for item in self.candidates)):
            raise TypeError("question execution candidates 类型错误")
        keys = tuple(item.stable_key() for item in self.candidates)
        if len(set(keys)) != len(keys):
            raise ValueError("question execution 不得重复 GenerationCandidate")
        if any(item.scope != self.query.request.response_scope
               for item in self.candidates):
            raise ValueError("question execution candidate scope 被替换")
        if self.candidates:
            authorized = (
                self.query.request.authorized_candidate_targets
                or (self.query.request.target,)
            )
            actual = tuple(item.proposition for item in self.candidates)
            if len(actual) != len(authorized) or set(actual) != set(authorized):
                raise ValueError(
                    "question execution candidate target 未获完整授权")
        _strict_key(self.trace, label="question execution trace")
        object.__setattr__(self, "candidates", tuple(sorted(
            self.candidates, key=lambda item: item.stable_key())))

    def planning_request(self) -> GenerationPlanningRequest:
        """用同一查询目标和执行候选形成 G-00 请求，不读取 expected 或 teacher。"""
        request = self.query.request
        from pure_integer_ai.cognition.shared.generation_plan import (
            AnswerGenerationGoal,
        )
        goal = AnswerGenerationGoal(
            request.goal_kind,
            request.target,
            request.required,
            request.source,
            request.response_scope,
            request.target_branch,
        )
        return GenerationPlanningRequest(goal, self.candidates)

    def stable_key(self) -> tuple[int, ...]:
        """返回 query、候选和执行 trace 的内容引用稳定键。"""
        result = [
            *_packed(integer_tuple_fingerprint(
                self.query.stable_key(), domain="question.execution.query.v1")),
            *_packed(self.reason.stable_key()),
            len(self.candidates),
        ]
        for candidate in self.candidates:
            result.extend(_packed(integer_tuple_fingerprint(
                candidate.stable_key(),
                domain="question.execution.candidate.v1",
            )))
        result.extend(_packed(integer_tuple_fingerprint(
            self.trace, domain="question.execution.trace.v1")))
        return tuple(result)


class QuestionExecutor(Protocol):
    """一个由 route 注册的 typed 查询、关系或逻辑执行 adapter。"""

    def execute(self, query: QuestionQuery) -> QuestionExecutionResult:
        """执行真实检索或推理并返回 G-00 候选和完整 trace。"""
        ...


class FactQuestionExecutor:
    """从既有 H-00 ledger 只读恢复一个显式 Proposition 的 active Evidence。"""

    def __init__(
            self,
            ledger: HypothesisLedger,
            *,
            route: ObjectIdentity,
            hypothesis_kind: tuple[int, ...],
            executed_reason: ObjectIdentity,
            ) -> None:
        """绑定真实候选 owner 和调用方注入 route/reason，不复制 ledger。"""
        if not isinstance(ledger, HypothesisLedger):
            raise TypeError("fact question ledger 类型错误")
        self.ledger = ledger
        self.route = _require_instruction(route, label="fact question route")
        self.hypothesis_kind = _strict_key(
            hypothesis_kind, label="fact question hypothesis_kind")
        self.executed_reason = _require_instruction(
            executed_reason, label="fact question executed_reason")

    def execute(self, query: QuestionQuery) -> QuestionExecutionResult:
        """读取目标命题 active 快照及未被替代 Evidence，形成回答 scope 候选。"""
        if not isinstance(query, QuestionQuery):
            raise TypeError("fact question query 类型错误")
        if query.route != self.route:
            raise ValueError("fact question executor 收到未注册 route")
        request = query.request
        snapshots = self.ledger.candidate_snapshots(
            request.target.template.stable_key(),
            observation=request.source,
            scope=request.evidence_scope,
            hypothesis_kind=self.hypothesis_kind,
        )
        candidates = []
        trace: list[int] = [1, *_packed(query.stable_key()), len(snapshots)]
        for snapshot in snapshots:
            trace.extend(_packed(_snapshot_key(snapshot)))
            if snapshot.lifecycle != LIFECYCLE_ACTIVE:
                continue
            active_ids = set(
                snapshot.support_evidence_ids
                + snapshot.refute_evidence_ids
                + snapshot.unknown_evidence_ids
            )
            evidence = tuple(
                item for item in self.ledger.evidence_history(
                    snapshot.hypothesis)
                if item.evidence_id in active_ids
            )
            if not evidence:
                continue
            candidates.append(GenerationCandidate(
                request.target,
                LogicEvidenceState.from_status(snapshot.epistemic_status),
                request.source,
                request.response_scope,
                evidence,
            ))
        trace.append(len(candidates))
        return QuestionExecutionResult(
            query,
            self.executed_reason,
            tuple(candidates),
            tuple(trace),
        )


@dataclass(frozen=True)
class EvidenceAnswerPolicyProtocol:
    """为 Evidence 自动选择注入 answer/clarify/unknown/conflict 原因。"""

    answer_reason: ObjectIdentity
    clarify_reason: ObjectIdentity
    unknown_reason: ObjectIdentity
    conflict_reason: ObjectIdentity

    def __post_init__(self) -> None:
        """核验四种原因是一等且互不相同的 MinimalInstruction。"""
        reasons = (
            self.answer_reason,
            self.clarify_reason,
            self.unknown_reason,
            self.conflict_reason,
        )
        if len(set(reasons)) != len(reasons):
            raise ValueError("Evidence answer policy reason 必须互不相同")
        for reason in reasons:
            _require_instruction(reason, label="Evidence answer policy reason")


def _satisfies(
        state: LogicEvidenceState,
        required: LogicEvidenceState,
        ) -> bool:
    """判断候选四态是否覆盖目标所需证据方向。"""
    return (
        (not required.support or state.support)
        and (not required.refute or state.refute)
    )


def _grouped_states(
        request: GenerationPlanningRequest,
        ) -> dict[tuple[int, ...], LogicEvidenceState]:
    """按完整 Proposition bound key 聚合候选四态，保留跨 Hypothesis 冲突。"""
    grouped: dict[tuple[int, ...], LogicEvidenceState] = {}
    for candidate in request.candidates:
        key = candidate.proposition.stable_key()
        prior = grouped.get(key, LogicEvidenceState(False, False))
        grouped[key] = LogicEvidenceState(
            prior.support or candidate.state.support,
            prior.refute or candidate.state.refute,
        )
    return grouped


def _ambiguous_propositions(
        request: GenerationPlanningRequest,
        ) -> frozenset[tuple[int, ...]]:
    """在 aggregate query 边界内按 Hypothesis competition 识别多命题歧义。"""
    grouped: dict[tuple, set[tuple[int, ...]]] = {}
    for candidate in request.candidates:
        if not _satisfies(candidate.state, request.goal.required):
            continue
        proposition = candidate.proposition.stable_key()
        for hypothesis in candidate.hypotheses:
            competition = (
                hypothesis.hypothesis_kind,
                hypothesis.competition_key,
                candidate.source,
                candidate.scope,
            )
            grouped.setdefault(competition, set()).add(proposition)
    ambiguous: set[tuple[int, ...]] = set()
    for propositions in grouped.values():
        if len(propositions) > 1:
            ambiguous.update(propositions)
    return frozenset(ambiguous)


class EvidenceAnswerPolicy:
    """只按 G-00 候选 Evidence 四态和竞争边界提出 G-01 内容选择。"""

    def __init__(
            self,
            content_protocol: AnswerContentProtocol,
            protocol: EvidenceAnswerPolicyProtocol,
            ) -> None:
        """绑定开放 stance 和原因，不接收 expected、teacher 或 fixture 答案。"""
        if not isinstance(content_protocol, AnswerContentProtocol):
            raise TypeError("Evidence answer content protocol 类型错误")
        if not isinstance(protocol, EvidenceAnswerPolicyProtocol):
            raise TypeError("Evidence answer policy protocol 类型错误")
        self.content_protocol = content_protocol
        self.protocol = protocol

    def select(self, request, artifacts) -> AnswerContentDecision:
        """按冲突、歧义、可回答、unknown 顺序生成可由共享 selector 复核的决定。"""
        if not isinstance(request, GenerationPlanningRequest):
            raise TypeError("Evidence answer policy request 类型错误")
        if artifacts:
            raise ValueError("Evidence answer policy 不私自采用 Artifact")
        grouped = _grouped_states(request)
        conflict_props = {
            proposition for proposition, state in grouped.items()
            if state.support and state.refute
        }
        candidate_keys = request.candidate_keys()
        by_key = {
            candidate.stable_key(): candidate
            for candidate in request.candidates
        }
        if conflict_props:
            selected = tuple(
                key for key in candidate_keys
                if by_key[key].proposition.stable_key() in conflict_props
            )
            return self._decision(
                request,
                self.content_protocol.conflict,
                self.protocol.conflict_reason,
                selected,
            )
        ambiguous = _ambiguous_propositions(request)
        if ambiguous:
            selected = tuple(
                key for key in candidate_keys
                if by_key[key].proposition.stable_key() in ambiguous
            )
            return self._decision(
                request,
                self.content_protocol.clarify,
                self.protocol.clarify_reason,
                selected,
            )
        selected = tuple(
            key for key in candidate_keys
            if _satisfies(by_key[key].state, request.goal.required)
            and not (
                by_key[key].state.support
                and by_key[key].state.refute)
        )
        if selected:
            return self._decision(
                request,
                self.content_protocol.answer,
                self.protocol.answer_reason,
                selected,
            )
        return self._decision(
            request,
            self.content_protocol.unknown,
            self.protocol.unknown_reason,
            (),
        )

    def _decision(
            self,
            request: GenerationPlanningRequest,
            stance: ObjectIdentity,
            reason: ObjectIdentity,
            selected: tuple[tuple[int, ...], ...],
            ) -> AnswerContentDecision:
        """保存完整请求、stance 和采用候选键作为可重放 policy trace。"""
        trace: list[int] = [
            1,
            *_packed(request.stable_key()),
            *_packed(stance.stable_key()),
            len(selected),
        ]
        for key in selected:
            trace.extend(_packed(key))
        return AnswerContentDecision(
            stance,
            reason,
            selected,
            (),
            tuple(trace),
        )


__all__ = [
    "EvidenceAnswerPolicy",
    "EvidenceAnswerPolicyProtocol",
    "FactQuestionExecutor",
    "QuestionExecutionResult",
    "QuestionExecutor",
    "QuestionQuery",
    "QuestionRequest",
]
