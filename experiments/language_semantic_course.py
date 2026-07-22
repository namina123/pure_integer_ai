"""正式语言课程到 S-02 语义计划的无词面协议。

课程 mapper 只能看到来源身份、typed Span/Occurrence 和已采用 Sense 投影。具体
predicate、Role、StructureConcept、Evidence 与生成目标均由课程显式声明；本模块
不解释 surface、Unicode、旧 token 序列或词典类别。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateGraphProjection,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_BINDER,
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    SourceRef,
    TypedRef,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
)
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.shared.semantic_template_scope import (
    SemanticTemplateScopePredicates,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    SubstitutionProtocol,
)
from pure_integer_ai.cognition.understanding.semantic_builder import (
    LocalSemanticRef,
    SemanticBuildPlan,
    SemanticBuilderProtocol,
)
from pure_integer_ai.cognition.understanding.semantic_builder_graph import (
    SemanticBuilderTracePredicates,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_EVIDENCE_STANCES = frozenset({
    EVIDENCE_SUPPORT,
    EVIDENCE_REFUTE,
    EVIDENCE_UNKNOWN,
})


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验课程协议中的非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _require_instruction(value: ObjectIdentity, *, label: str) -> None:
    """核验课程调度身份是一等 MinimalInstruction。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")


@dataclass(frozen=True)
class ActiveSenseCourseView:
    """向语义 mapper 暴露的 occurrence、Sense、Concept 和 active 投影。"""

    occurrence: ObjectIdentity
    sense: ObjectIdentity
    concept: ObjectIdentity
    hypothesis: HypothesisKey
    projection: CandidateGraphProjection

    def __post_init__(self) -> None:
        if (not isinstance(self.occurrence, ObjectIdentity)
                or self.occurrence.object_kind != OBJECT_OCCURRENCE):
            raise ValueError("active Sense view 必须绑定 Occurrence")
        if not isinstance(self.sense, ObjectIdentity):
            raise TypeError("active Sense view 的 Sense 身份非法")
        if (not isinstance(self.concept, ObjectIdentity)
                or self.concept.object_kind != OBJECT_CONCEPT):
            raise ValueError("active Sense view 的目标必须是 Concept")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("active Sense view 的 Hypothesis 非法")
        if not isinstance(self.projection, CandidateGraphProjection):
            raise TypeError("active Sense view 的 lifecycle 投影非法")
        definition = self.projection.candidate
        if (definition.definition.candidate != self.sense
                or definition.hypothesis != self.hypothesis):
            raise ValueError("active Sense view 与图内候选投影不一致")


@dataclass(frozen=True)
class LanguageSemanticCourseInput:
    """不含词面和旧顺序的当前来源 typed 输入全集。"""

    source: SourceRef
    occurrence_scope: ScopeIdentity
    runtime_scope: ScopeIdentity
    occurrences: tuple[TypedRef, ...]
    spans: tuple[TypedRef, ...]
    active_senses: tuple[ActiveSenseCourseView, ...] = ()
    read_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceRef):
            raise TypeError("semantic course source 必须是 SourceRef")
        if (not isinstance(self.occurrence_scope, ScopeIdentity)
                or self.occurrence_scope.source != self.source):
            raise ValueError("semantic course occurrence scope 必须指向来源")
        if not isinstance(self.runtime_scope, ScopeIdentity):
            raise TypeError("semantic course runtime scope 类型错误")
        for name, values in (("occurrences", self.occurrences),
                             ("spans", self.spans)):
            if not isinstance(values, tuple):
                raise TypeError(f"semantic course {name} 必须是 TypedRef tuple")
            if any(not isinstance(item, TypedRef) for item in values):
                raise TypeError(f"semantic course {name} 含非法引用")
            if len(set(values)) != len(values):
                raise ValueError(f"semantic course {name} 不得重复")
        if (not isinstance(self.active_senses, tuple)
                or any(not isinstance(item, ActiveSenseCourseView)
                       for item in self.active_senses)):
            raise TypeError("active_senses 必须是 ActiveSenseCourseView tuple")
        if type(self.read_only) is not bool:
            raise TypeError("semantic course read_only 必须是 bool")


@dataclass(frozen=True)
class SemanticCourseEvidenceSpec:
    """课程对一个局部 Proposition 提交的独立来源 Evidence。"""

    proposition: LocalSemanticRef
    evidence_id: int
    stance: int
    reason_key: tuple[int, ...]
    source: SourceRef
    timestamp_seq: int
    payload: tuple[int, ...] = ()
    supersedes_evidence_id: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.proposition, LocalSemanticRef)
                or self.proposition.object_kind != OBJECT_PROPOSITION):
            raise ValueError("课程 Evidence 必须指向局部 Proposition")
        assert_int(
            self.evidence_id,
            self.stance,
            self.timestamp_seq,
            self.supersedes_evidence_id,
            *self.payload,
            _where="SemanticCourseEvidenceSpec",
        )
        if type(self.evidence_id) is not int or self.evidence_id <= 0:
            raise ValueError("课程 evidence_id 必须为严格正整数")
        if self.stance not in _EVIDENCE_STANCES:
            raise ValueError("课程 Evidence stance 未注册")
        if type(self.timestamp_seq) is not int or self.timestamp_seq < 0:
            raise ValueError("课程 Evidence timestamp 必须为非负严格整数")
        if (type(self.supersedes_evidence_id) is not int
                or self.supersedes_evidence_id < 0
                or self.supersedes_evidence_id == self.evidence_id):
            raise ValueError("课程 Evidence supersede id 非法")
        _strict_key(self.reason_key, label="semantic course evidence reason")
        if not isinstance(self.source, SourceRef):
            raise TypeError("课程 Evidence source 必须是 SourceRef")
        if not isinstance(self.payload, tuple) or any(
                type(item) is not int for item in self.payload):
            raise TypeError("课程 Evidence payload 必须是严格整数 tuple")


@dataclass(frozen=True)
class SemanticCourseTemplateScope:
    """声明一个局部 Proposition template 显式引入的 Binder。"""

    proposition: LocalSemanticRef
    scope: ObjectIdentity
    introduced_binders: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.proposition, LocalSemanticRef)
                or self.proposition.object_kind != OBJECT_PROPOSITION):
            raise ValueError("template scope 必须指向局部 Proposition")
        if (not isinstance(self.scope, ObjectIdentity)
                or self.scope.object_kind != OBJECT_CONTEXT_SCOPE):
            raise ValueError("template scope 必须注入一等 ContextScope")
        if not isinstance(self.introduced_binders, tuple):
            raise TypeError("introduced_binders 必须是 ObjectIdentity tuple")
        if any(not isinstance(item, ObjectIdentity)
               or item.object_kind != OBJECT_BINDER
               for item in self.introduced_binders):
            raise TypeError("introduced_binders 只能包含 Binder")
        if len(set(self.introduced_binders)) != len(self.introduced_binders):
            raise ValueError("template scope 不得重复 Binder")
        if any(semantic_source(item) != semantic_source(self.scope)
               for item in self.introduced_binders):
            raise ValueError("template scope 与 Binder 来源不一致")


@dataclass(frozen=True)
class LanguageSemanticLesson:
    """一次课程声明的 S-02 计划、证据、绑定和生成目标。"""

    root_anchor: TypedRef
    plan: SemanticBuildPlan
    evidence: tuple[SemanticCourseEvidenceSpec, ...]
    template_scopes: tuple[SemanticCourseTemplateScope, ...]
    binding_environment: BindingEnvironment
    goal_proposition: LocalSemanticRef
    generation_candidates: tuple[LocalSemanticRef, ...]
    goal_kind: ObjectIdentity
    required: LogicEvidenceState
    target_branch: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.root_anchor, TypedRef):
            raise TypeError("semantic lesson root_anchor 必须是 TypedRef")
        if not isinstance(self.plan, SemanticBuildPlan):
            raise TypeError("semantic lesson plan 必须是 SemanticBuildPlan")
        if (not isinstance(self.evidence, tuple)
                or any(not isinstance(item, SemanticCourseEvidenceSpec)
                       for item in self.evidence)):
            raise TypeError("semantic lesson evidence 类型错误")
        if (not isinstance(self.template_scopes, tuple)
                or any(not isinstance(item, SemanticCourseTemplateScope)
                       for item in self.template_scopes)):
            raise TypeError("semantic lesson template_scopes 类型错误")
        if not isinstance(self.binding_environment, BindingEnvironment):
            raise TypeError("semantic lesson binding environment 类型错误")
        if (not isinstance(self.goal_proposition, LocalSemanticRef)
                or self.goal_proposition.object_kind != OBJECT_PROPOSITION):
            raise ValueError("semantic lesson goal 必须是局部 Proposition")
        if (not isinstance(self.generation_candidates, tuple)
                or not self.generation_candidates
                or any(not isinstance(item, LocalSemanticRef)
                       or item.object_kind != OBJECT_PROPOSITION
                       for item in self.generation_candidates)):
            raise ValueError("generation_candidates 必须是非空 Proposition tuple")
        if len(set(self.generation_candidates)) != len(
                self.generation_candidates):
            raise ValueError("generation_candidates 不得重复")
        if self.goal_proposition not in self.generation_candidates:
            raise ValueError("generation_candidates 必须包含 goal Proposition")
        _require_instruction(self.goal_kind, label="semantic lesson goal kind")
        if not isinstance(self.required, LogicEvidenceState):
            raise TypeError("semantic lesson required 类型错误")
        if not self.required.support and not self.required.refute:
            raise ValueError("semantic lesson goal 至少要求一个 Evidence 方向")
        if self.target_branch is not None:
            if (not isinstance(self.target_branch, ObjectIdentity)
                    or self.target_branch.object_kind != OBJECT_LANGUAGE_BRANCH):
                raise ValueError("semantic lesson target_branch 类型错误")


@dataclass(frozen=True)
class LanguageSemanticCourseDecision:
    """课程 mapper 对当前来源返回的显式 lesson 或无 lesson 结果。"""

    reason: ObjectIdentity
    trace: tuple[int, ...]
    lesson: LanguageSemanticLesson | None = None

    def __post_init__(self) -> None:
        _require_instruction(self.reason, label="semantic course decision reason")
        _strict_key(self.trace, label="semantic course decision trace")
        if self.lesson is not None and not isinstance(
                self.lesson, LanguageSemanticLesson):
            raise TypeError("semantic course decision lesson 类型错误")


@runtime_checkable
class LanguageSemanticCourseMapper(Protocol):
    """按来源和 typed 输入形成课程语义声明的可替换 mapper。"""

    def map(
            self, input_value: LanguageSemanticCourseInput,
            ) -> LanguageSemanticCourseDecision:
        """返回完整 lesson；不得读取词面、Unicode 或旧顺序缓存。"""
        ...

    def clone_for_evaluation(self) -> "LanguageSemanticCourseMapper":
        """返回不共享可变课程状态的评测 mapper。"""
        ...

    def state_key(self) -> tuple:
        """返回课程 mapper 的完整可比较状态。"""
        ...


@dataclass(frozen=True)
class LanguageSemanticCourseProtocol:
    """注入 S-02/S-03 图协议、来源元数据和正式课程 mapper。"""

    builder: SemanticBuilderProtocol
    atomic_predicates: tuple[ObjectIdentity, ...]
    trace_predicates: tuple[ObjectIdentity, ...]
    scope_predicates: tuple[ObjectIdentity, ...]
    substitution: SubstitutionProtocol
    mapper: LanguageSemanticCourseMapper
    provenance_kind: int
    epistemic_origin: int = 0
    content_version: int = 0
    qualifiers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.builder, SemanticBuilderProtocol):
            raise TypeError("semantic course builder protocol 类型错误")
        if (not isinstance(self.atomic_predicates, tuple)
                or len(self.atomic_predicates) != 6):
            raise ValueError("semantic course 必须注入六个原子命题 predicate")
        if (not isinstance(self.trace_predicates, tuple)
                or len(self.trace_predicates) != 3):
            raise ValueError("semantic course 必须注入三个 builder trace predicate")
        if (not isinstance(self.scope_predicates, tuple)
                or len(self.scope_predicates) != 2):
            raise ValueError("semantic course 必须注入两个 template scope predicate")
        predicates = (
            self.atomic_predicates
            + self.trace_predicates
            + self.scope_predicates
        )
        if any(not isinstance(item, ObjectIdentity)
               or item.object_kind != OBJECT_CONCEPT for item in predicates):
            raise ValueError("semantic course predicate 必须是一等 Concept")
        if len(set(predicates)) != len(predicates):
            raise ValueError("semantic course predicate 必须互不相同")
        if not isinstance(self.substitution, SubstitutionProtocol):
            raise TypeError("semantic course substitution protocol 类型错误")
        if not isinstance(self.mapper, LanguageSemanticCourseMapper):
            raise TypeError("semantic course mapper 未实现完整协议")
        if not isinstance(self.qualifiers, tuple):
            raise TypeError("semantic course qualifiers 必须是整数 tuple")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            *self.qualifiers,
            _where="LanguageSemanticCourseProtocol",
        )
        if type(self.provenance_kind) is not int or self.provenance_kind <= 0:
            raise ValueError("semantic course provenance_kind 必须为严格正整数")
        if type(self.epistemic_origin) is not int or self.epistemic_origin < 0:
            raise ValueError("semantic course epistemic_origin 必须为非负整数")
        if type(self.content_version) is not int or self.content_version < 0:
            raise ValueError("semantic course content_version 必须为非负整数")
        if any(type(item) is not int for item in self.qualifiers):
            raise ValueError("semantic course qualifiers 必须使用严格整数")

    def graph_protocols(
            self, ontology,
            ) -> tuple[AtomicPropositionPredicates,
                       SemanticBuilderTracePredicates,
                       SemanticTemplateScopePredicates]:
        """物化开放 predicate 并构造 S-00/S-02/S-03 图协议。"""
        refs = tuple(ontology.materialize(item) for item in (
            self.atomic_predicates
            + self.trace_predicates
            + self.scope_predicates
        ))
        return (
            AtomicPropositionPredicates(*refs[:6]),
            SemanticBuilderTracePredicates(*refs[6:9]),
            SemanticTemplateScopePredicates(*refs[9:]),
        )


__all__ = [
    "ActiveSenseCourseView",
    "LanguageSemanticCourseDecision",
    "LanguageSemanticCourseInput",
    "LanguageSemanticCourseMapper",
    "LanguageSemanticCourseProtocol",
    "LanguageSemanticLesson",
    "SemanticCourseEvidenceSpec",
    "SemanticCourseTemplateScope",
]
