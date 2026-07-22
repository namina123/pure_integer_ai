"""L-05B2B 从 S-02 图和 active H-00 Evidence 只读恢复 G-00 请求。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.generation_plan import (
    AnswerGenerationGoal,
    GenerationCandidate,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    HypothesisSnapshot,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_BINDER,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_PROPOSITION,
    OBJECT_VARIABLE,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.scope_identity import query_scope
from pure_integer_ai.cognition.shared.semantic_template_scope import (
    SemanticTemplateScopeGraph,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
)
from pure_integer_ai.cognition.understanding.semantic_builder_graph import (
    MaterializedSemanticCandidate,
    SemanticCandidateGraphAdapter,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.language_semantic_course import (
    LanguageSemanticCourseInput,
)


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 query 决策中的非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _require_instruction(value: ObjectIdentity, *, label: str) -> None:
    """核验 query 原因是一等 MinimalInstruction。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")


@dataclass(frozen=True)
class RecoveredSemanticCandidate:
    """从权威图和 H-00 账本汇合出的一个 active 候选只读视图。"""

    materialized: MaterializedSemanticCandidate
    snapshot: HypothesisSnapshot
    evidence: tuple[EvidenceRecord, ...]
    ground: bool
    recoverable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.materialized, MaterializedSemanticCandidate):
            raise TypeError("recovered semantic candidate 图视图类型错误")
        if not isinstance(self.snapshot, HypothesisSnapshot):
            raise TypeError("recovered semantic candidate snapshot 类型错误")
        hypothesis = self.snapshot.hypothesis
        definition = self.materialized.atomic.definition
        if hypothesis.candidate_key != definition.proposition.stable_key():
            raise ValueError("recovered Hypothesis 未绑定当前 Proposition")
        if (hypothesis.observation != definition.source
                or hypothesis.scope != self.materialized.atomic.scope):
            raise ValueError("recovered Hypothesis 来源或 scope 不一致")
        if self.snapshot.lifecycle != LIFECYCLE_ACTIVE:
            raise ValueError("recovered semantic candidate 必须为 active")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("recovered semantic candidate 缺 active Evidence")
        if any(not isinstance(item, EvidenceRecord)
               or item.hypothesis != hypothesis for item in self.evidence):
            raise ValueError("recovered Evidence 未精确绑定当前 Hypothesis")
        active_ids = set(
            self.snapshot.support_evidence_ids
            + self.snapshot.refute_evidence_ids
            + self.snapshot.unknown_evidence_ids
        )
        if {item.evidence_id for item in self.evidence} != active_ids:
            raise ValueError("recovered Evidence 未精确覆盖 active 快照")
        if type(self.ground) is not bool:
            raise TypeError("recovered ground 必须是严格 bool")
        if type(self.recoverable) is not bool:
            raise TypeError("recovered recoverable 必须是严格 bool")
        object.__setattr__(self, "evidence", tuple(sorted(
            self.evidence, key=lambda item: item.evidence_id)))

    @property
    def hypothesis(self) -> HypothesisKey:
        """返回 mapper 必须显式选择的完整 Hypothesis key。"""
        return self.snapshot.hypothesis

    def stable_key(self) -> tuple[int, ...]:
        """返回图定义、候选状态、active Evidence 和 ground 标志。"""
        result = [
            *_packed(self.materialized.atomic.definition.proposition.stable_key()),
            *_packed(self.hypothesis.stable_key()),
            *self.snapshot.stable_key(),
            int(self.ground),
            int(self.recoverable),
            len(self.evidence),
        ]
        for item in self.evidence:
            result.extend(_packed(item.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class LanguageSemanticQueryInput:
    """向只读 mapper 暴露当前 typed 输入和全部可恢复候选。"""

    current: LanguageSemanticCourseInput
    candidates: tuple[RecoveredSemanticCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.current, LanguageSemanticCourseInput):
            raise TypeError("semantic query current 输入类型错误")
        if not self.current.read_only:
            raise ValueError("semantic query 只允许 read-only 输入")
        if not isinstance(self.candidates, tuple) or any(
                not isinstance(item, RecoveredSemanticCandidate)
                for item in self.candidates):
            raise TypeError("semantic query candidates 类型错误")
        keys = tuple(item.hypothesis for item in self.candidates)
        if len(set(keys)) != len(keys):
            raise ValueError("semantic query 不得重复 Hypothesis")


@dataclass(frozen=True)
class LanguageSemanticQueryDecision:
    """mapper 对 recovered 候选的显式采用或 typed 无请求决定。"""

    reason: ObjectIdentity
    trace: tuple[int, ...]
    goal: HypothesisKey | None = None
    candidates: tuple[HypothesisKey, ...] = ()
    goal_kind: ObjectIdentity | None = None
    required: LogicEvidenceState | None = None
    target_branch: ObjectIdentity | None = None
    binding_environment: BindingEnvironment = BindingEnvironment()

    def __post_init__(self) -> None:
        _require_instruction(self.reason, label="semantic query reason")
        _strict_key(self.trace, label="semantic query trace")
        if not isinstance(self.binding_environment, BindingEnvironment):
            raise TypeError("semantic query binding environment 类型错误")
        if self.goal is None:
            if (self.candidates or self.goal_kind is not None
                    or self.required is not None
                    or self.target_branch is not None):
                raise ValueError("semantic query 无 goal 时不得携带部分请求字段")
            return
        if not isinstance(self.goal, HypothesisKey):
            raise TypeError("semantic query goal 必须是 HypothesisKey")
        if (not isinstance(self.candidates, tuple) or not self.candidates
                or any(not isinstance(item, HypothesisKey)
                       for item in self.candidates)):
            raise ValueError("semantic query candidates 必须是非空 Hypothesis tuple")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("semantic query candidates 不得重复")
        if self.goal not in self.candidates:
            raise ValueError("semantic query goal 必须属于 candidates")
        if self.goal_kind is None:
            raise ValueError("semantic query 有 goal 时必须提供 goal_kind")
        _require_instruction(self.goal_kind, label="semantic query goal kind")
        if not isinstance(self.required, LogicEvidenceState):
            raise TypeError("semantic query required 类型错误")
        if not self.required.support and not self.required.refute:
            raise ValueError("semantic query required 至少需要一个 Evidence 方向")
        if self.target_branch is not None and (
                not isinstance(self.target_branch, ObjectIdentity)
                or self.target_branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise ValueError("semantic query target_branch 必须是 LanguageBranch")
        object.__setattr__(self, "candidates", tuple(sorted(self.candidates)))


@runtime_checkable
class LanguageSemanticQueryMapper(Protocol):
    """从全部 recovered 候选显式选择 query goal，不解释词面或存储顺序。"""

    def map(
            self, input_value: LanguageSemanticQueryInput,
            ) -> LanguageSemanticQueryDecision:
        """返回 exact Hypothesis 选择或 typed 无请求。"""
        ...

    def clone_for_evaluation(self) -> "LanguageSemanticQueryMapper":
        """返回不共享可变调用状态的评测 mapper。"""
        ...

    def state_key(self) -> tuple:
        """返回 mapper 的完整可比较状态。"""
        ...


@dataclass(frozen=True)
class LanguageSemanticQueryProtocol:
    """注入只读 recovered 候选选择 mapper。"""

    mapper: LanguageSemanticQueryMapper

    def __post_init__(self) -> None:
        if not isinstance(self.mapper, LanguageSemanticQueryMapper):
            raise TypeError("semantic query mapper 未实现完整协议")

    def clone_for_evaluation(self) -> "LanguageSemanticQueryProtocol":
        """克隆 mapper 状态，图和 ledger 由外层评测上下文重建。"""
        return LanguageSemanticQueryProtocol(
            self.mapper.clone_for_evaluation())

    def state_key(self) -> tuple:
        """返回 query mapper 状态供 V-06 宿主隔离核验。"""
        return self.mapper.state_key()


@dataclass(frozen=True)
class LanguageSemanticQueryRun:
    """一次只读候选恢复、mapper 决策和可选 G-00 请求。"""

    input_value: LanguageSemanticQueryInput
    decision: LanguageSemanticQueryDecision
    request: GenerationPlanningRequest | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_value, LanguageSemanticQueryInput):
            raise TypeError("semantic query run 输入类型错误")
        if not isinstance(self.decision, LanguageSemanticQueryDecision):
            raise TypeError("semantic query run 决策类型错误")
        if (self.decision.goal is None) != (self.request is None):
            raise ValueError("semantic query decision 与 request 状态不一致")


class LanguageSemanticQueryRuntime:
    """只读汇合 S-02 图和 H-00 Evidence，并重建 ground Proposition 请求。"""

    def __init__(
            self,
            graph: SemanticCandidateGraphAdapter,
            ledger: HypothesisLedger,
            substituter: PropositionSubstituter,
            protocol: LanguageSemanticQueryProtocol,
            *,
            template_scopes: SemanticTemplateScopeGraph,
            hypothesis_kind: tuple[int, ...],
            builder: ObjectIdentity,
            ) -> None:
        if not isinstance(graph, SemanticCandidateGraphAdapter):
            raise TypeError("semantic query graph 类型错误")
        if not isinstance(ledger, HypothesisLedger):
            raise TypeError("semantic query ledger 类型错误")
        if not isinstance(substituter, PropositionSubstituter):
            raise TypeError("semantic query substituter 类型错误")
        if not isinstance(protocol, LanguageSemanticQueryProtocol):
            raise TypeError("semantic query protocol 类型错误")
        if not isinstance(template_scopes, SemanticTemplateScopeGraph):
            raise TypeError("semantic query template scope graph 类型错误")
        _strict_key(hypothesis_kind, label="semantic query hypothesis kind")
        _require_instruction(builder, label="semantic query builder")
        self.graph = graph
        self.ledger = ledger
        self.substituter = substituter
        self.protocol = protocol
        self.template_scopes = template_scopes
        self.hypothesis_kind = hypothesis_kind
        self.builder = builder

    def process(
            self, current: LanguageSemanticCourseInput,
            ) -> LanguageSemanticQueryRun:
        """按当前 anchor/Sense 反查全部候选，再执行显式 mapper 和 ground 重建。"""
        if not isinstance(current, LanguageSemanticCourseInput):
            raise TypeError("semantic query current 类型错误")
        if not current.read_only:
            raise ValueError("semantic query runtime 只允许 read-only 调用")
        anchors = tuple(
            self.graph.ontology.identity_of(item)
            for item in current.occurrences + current.spans
        )
        fillers = tuple(sorted(
            {item.concept for item in current.active_senses},
            key=ObjectIdentity.stable_key,
        ))
        materialized = self.graph.lookup(
            anchors=anchors,
            fillers=fillers,
        )
        recovered = []
        for item in materialized:
            if item.builder != self.builder:
                continue
            definition = item.atomic.definition
            snapshots = self.ledger.candidate_snapshots(
                definition.proposition.stable_key(),
                observation=definition.source,
                scope=item.atomic.scope,
                hypothesis_kind=self.hypothesis_kind,
            )
            for snapshot in snapshots:
                if snapshot.lifecycle != LIFECYCLE_ACTIVE:
                    continue
                active_ids = set(
                    snapshot.support_evidence_ids
                    + snapshot.refute_evidence_ids
                    + snapshot.unknown_evidence_ids
                )
                evidence = tuple(
                    evidence
                    for evidence in self.ledger.evidence_history(
                        snapshot.hypothesis)
                    if evidence.evidence_id in active_ids
                )
                if not evidence:
                    continue
                _, ground, recoverable = self._template_closure(item)
                recovered.append(RecoveredSemanticCandidate(
                    item,
                    snapshot,
                    evidence,
                    ground,
                    recoverable,
                ))
        recovered.sort(key=lambda item: item.hypothesis.stable_key())
        input_value = LanguageSemanticQueryInput(current, tuple(recovered))
        decision = self.protocol.mapper.map(input_value)
        if not isinstance(decision, LanguageSemanticQueryDecision):
            raise TypeError("semantic query mapper 返回类型错误")
        if decision.goal is not None:
            by_hypothesis = {
                item.hypothesis: item for item in input_value.candidates}
            if any(item not in by_hypothesis for item in decision.candidates):
                raise ValueError(
                    "semantic query mapper 选择了 recovered 输入外的 Hypothesis")
            if any(not by_hypothesis[item].recoverable
                   for item in decision.candidates):
                decision = LanguageSemanticQueryDecision(
                    decision.reason,
                    decision.trace,
                )
        request = self._request(input_value, decision)
        return LanguageSemanticQueryRun(input_value, decision, request)

    def _request(
            self,
            input_value: LanguageSemanticQueryInput,
            decision: LanguageSemanticQueryDecision,
            ) -> GenerationPlanningRequest | None:
        """只从 mapper 显式选择、图定义和 active Evidence 重建当前 query 请求。"""
        if decision.goal is None:
            return None
        by_hypothesis = {
            item.hypothesis: item for item in input_value.candidates}
        if any(item not in by_hypothesis for item in decision.candidates):
            raise ValueError("semantic query mapper 选择了 recovered 输入外的 Hypothesis")
        selected = tuple(by_hypothesis[item] for item in decision.candidates)
        definitions = {}
        for item in selected:
            closure, _, recoverable = self._template_closure(item.materialized)
            if not recoverable:
                raise RuntimeError("不可恢复 semantic candidate 越过请求归一边界")
            for materialized in closure:
                definition = materialized.atomic.definition
                scope_definition = self.template_scopes.read(
                    definition.proposition).definition
                template = ScopedPropositionTemplate(
                    definition,
                    materialized.structure,
                    scope_definition.introduced_binders,
                )
                existing = definitions.get(definition.proposition)
                if existing is not None and existing != template:
                    raise RuntimeError("同一嵌套 Proposition 恢复出竞争 template")
                definitions[definition.proposition] = template
        templates = PropositionTemplateGraph(tuple(definitions.values()))
        environment = decision.binding_environment
        bound = {
            proposition: self.substituter.substitute(
                proposition,
                templates,
                environment,
            )
            for proposition in definitions
        }
        request_scope = query_scope(1, parent=input_value.current.runtime_scope)
        generation_candidates = tuple(
            GenerationCandidate(
                bound[item.materialized.atomic.definition.proposition],
                LogicEvidenceState.from_status(
                    item.snapshot.epistemic_status),
                item.materialized.atomic.definition.source,
                request_scope,
                item.evidence,
            )
            for item in selected
        )
        goal_view = by_hypothesis[decision.goal]
        goal_definition = goal_view.materialized.atomic.definition
        goal = AnswerGenerationGoal(
            decision.goal_kind,
            bound[goal_definition.proposition],
            decision.required,
            goal_definition.source,
            request_scope,
            decision.target_branch,
        )
        return GenerationPlanningRequest(goal, generation_candidates)

    def _template_closure(
            self, root: MaterializedSemanticCandidate,
            ) -> tuple[tuple[MaterializedSemanticCandidate, ...], bool, bool]:
        """递归恢复 S-02 template，并分别判定 ground 与可绑定状态。"""
        templates: dict[ObjectIdentity, MaterializedSemanticCandidate] = {}
        active: set[ObjectIdentity] = set()
        ground = True

        def visit(candidate: MaterializedSemanticCandidate) -> bool:
            """遍历嵌套命题；opaque Proposition 保持常量，部分拓扑继续失败。"""
            nonlocal ground
            proposition = candidate.atomic.definition.proposition
            if proposition in active:
                return False
            existing = templates.get(proposition)
            if existing is not None:
                if existing != candidate:
                    raise RuntimeError("同一 Proposition 恢复出竞争 S-02 template")
                return True
            if candidate.builder != self.builder:
                return False
            self.template_scopes.read(proposition)
            active.add(proposition)
            templates[proposition] = candidate
            try:
                for binding in candidate.atomic.definition.bindings:
                    filler = binding.filler
                    if filler.object_kind in {OBJECT_VARIABLE, OBJECT_BINDER}:
                        ground = False
                        continue
                    if filler.object_kind != OBJECT_PROPOSITION:
                        continue
                    nested = self.graph.read_if_defined(filler)
                    if nested is not None and not visit(nested):
                        return False
            finally:
                active.remove(proposition)
            return True

        recoverable = visit(root)
        ordered = tuple(
            templates[key]
            for key in sorted(templates, key=ObjectIdentity.stable_key)
        )
        return ordered, ground, recoverable


__all__ = [
    "LanguageSemanticQueryDecision",
    "LanguageSemanticQueryInput",
    "LanguageSemanticQueryMapper",
    "LanguageSemanticQueryProtocol",
    "LanguageSemanticQueryRun",
    "LanguageSemanticQueryRuntime",
    "RecoveredSemanticCandidate",
]
