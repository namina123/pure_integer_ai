"""R-02 集合关系的 R-00 适配、课程编排和 V-06 重建协议。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.hypothesis import (
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
)
from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_PROPOSITION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.relation_closure import (
    RelationClosureCandidateSpec,
)
from pure_integer_ai.cognition.shared.relation_use import RelationUseContext
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
)
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
    SemanticGraph,
)
from pure_integer_ai.cognition.shared.set_relation import (
    MemberTypeResolver,
    SetBinaryRelationProtocol,
    SetQuantifierBranch,
    SetQuantifierEvaluation,
    SetRelationBudget,
    SetRelationDomainResult,
    SetRelationEngine,
    SetRelationError,
    SetRelationEvaluation,
    SetRelationEvidence,
    SetRelationKnowledge,
    SetRelationProtocol,
    SetRelationStatement,
    SetUnaryRelationProtocol,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureFormationTrace,
    RelationClosureRecognitionInput,
    RelationClosureRecognitionTrace,
    RelationClosureRuntime,
    RelationClosureUse,
)
from pure_integer_ai.experiments.train_context import TrainContext


class SetRelationRuntimeError(RuntimeError):
    """R-02 运行 owner、课程或旧边映射不满足完整协议。"""


def _strict_key(
        value: tuple[int, ...], *, label: str,
        allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """校验调用方注入的纯整数稳定键。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{label} 必须是整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def _single_filler(
        definition: AtomicPropositionDefinition,
        role: ObjectIdentity,
        ) -> ObjectIdentity:
    """读取一个命题中指定 Role 的唯一 filler，竞争或缺失均失败。"""
    matches = tuple(
        binding.filler
        for binding in definition.bindings
        if binding.role == role
    )
    if len(matches) != 1:
        raise SetRelationRuntimeError("set relation Role 必须恰有一个 filler")
    return matches[0]


@dataclass(frozen=True)
class SetRelationQuery:
    """一次集合关系查询及可选的来源化采用路由。"""

    statement: SetRelationStatement
    use_key: tuple[int, ...] | None = None
    context: RelationUseContext | None = None

    def __post_init__(self) -> None:
        """要求 use_key 与 context 的出现满足 R-00 Use owner 边界。"""
        if not isinstance(self.statement, SetRelationStatement):
            raise TypeError("set relation query statement 类型错误")
        if self.use_key is not None:
            _strict_key(self.use_key, label="SetRelationQuery.use_key")
        if self.context is not None and not isinstance(
                self.context, RelationUseContext):
            raise TypeError("set relation query context 类型错误")
        if self.context is not None and self.use_key is None:
            raise ValueError("set relation query context 必须配套 use_key")


@dataclass(frozen=True)
class SetRelationRuntimeResult:
    """一次集合关系查询的四态结果和实际 R-00 Use。"""

    evaluation: SetRelationEvaluation
    uses: tuple[RelationClosureUse, ...]

    def __post_init__(self) -> None:
        """核验结果容器不混入其他类型的采用记录。"""
        if not isinstance(self.evaluation, SetRelationEvaluation):
            raise TypeError("set relation runtime evaluation 类型错误")
        if not isinstance(self.uses, tuple):
            raise TypeError("set relation runtime uses 必须是 tuple")
        if any(not isinstance(item, RelationClosureUse) for item in self.uses):
            raise TypeError("set relation runtime use 类型错误")


@dataclass(frozen=True)
class SetRelationDomainQuery:
    """一次有限域构造及可选的闭域/MEMBER 采用路由。"""

    domain: ObjectIdentity
    use_key: tuple[int, ...] | None = None
    context: RelationUseContext | None = None

    def __post_init__(self) -> None:
        """核验域身份和 Use 路由；SetExpr 类型由纯语义层再次校验。"""
        if not isinstance(self.domain, ObjectIdentity):
            raise TypeError("set relation domain query 类型错误")
        if self.use_key is not None:
            _strict_key(self.use_key, label="SetRelationDomainQuery.use_key")
        if self.context is not None and not isinstance(
                self.context, RelationUseContext):
            raise TypeError("set relation domain query context 类型错误")
        if self.context is not None and self.use_key is None:
            raise ValueError("set relation domain context 必须配套 use_key")


@dataclass(frozen=True)
class SetRelationDomainRuntimeResult:
    """一次有限域构造结果和闭域/MEMBER 的实际 R-00 Use。"""

    domain: SetRelationDomainResult
    uses: tuple[RelationClosureUse, ...]

    def __post_init__(self) -> None:
        """核验有限域结果和采用记录类型。"""
        if not isinstance(self.domain, SetRelationDomainResult):
            raise TypeError("set relation domain result 类型错误")
        if not isinstance(self.uses, tuple):
            raise TypeError("set relation domain uses 必须是 tuple")
        if any(not isinstance(item, RelationClosureUse) for item in self.uses):
            raise TypeError("set relation domain use 类型错误")


@dataclass(frozen=True)
class LegacySetRelationRecord:
    """旧关系存储的一条不带新语义的来源化整数记录。"""

    relation_key: tuple[int, ...]
    subject_key: tuple[int, ...]
    object_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """保留旧记录原始键和来源，不在宿主中解释 EDGE_IS_A 数值。"""
        _strict_key(self.relation_key, label="legacy relation key")
        _strict_key(self.subject_key, label="legacy subject key")
        _strict_key(self.object_key, label="legacy object key")
        _strict_key(self.trace, label="legacy relation trace")
        if not isinstance(self.source, SourceRef):
            raise TypeError("legacy relation source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("legacy relation scope 类型错误")
        if self.scope.source != self.source:
            raise ValueError("legacy relation scope 必须绑定记录来源")


@dataclass(frozen=True)
class MappedLegacySetRelation:
    """旧记录经 mapper 补全后的 typed SUBSET_EQ forming 和 recognition。"""

    formation: "SetRelationFormationRequest"
    recognition: RelationClosureRecognitionInput

    def __post_init__(self) -> None:
        """要求 forming 与 recognition 指向同一完整 Proposition。"""
        if not isinstance(self.formation, SetRelationFormationRequest):
            raise TypeError("mapped legacy formation 类型错误")
        if not isinstance(self.recognition, RelationClosureRecognitionInput):
            raise TypeError("mapped legacy recognition 类型错误")
        if (self.formation.spec.proposition.proposition
                != self.recognition.proposition):
            raise ValueError("mapped legacy forming 与 recognition 不一致")


@runtime_checkable
class LegacySetRelationMapper(Protocol):
    """把旧整数记录显式映射为完整 typed SUBSET_EQ 候选。"""

    def map(
            self, record: LegacySetRelationRecord,
            ) -> MappedLegacySetRelation | None:
        """无法补全 SetExpr、来源、scope、版本或 Evidence 时返回 None。"""
        ...

    def clone_for_evaluation(self) -> "LegacySetRelationMapper":
        """返回不共享可变状态的评测 mapper。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回 mapper 版本、规则和来源策略的完整整数键。"""
        ...


class SetRelationRuntime:
    """把 R-00 当前关系快照适配为有界集合闭包和来源化 Use。"""

    def __init__(
            self,
            relation_runtime: RelationClosureRuntime,
            protocol: SetRelationProtocol,
            budget: SetRelationBudget,
            member_type_resolver: MemberTypeResolver,
            ) -> None:
        """绑定同一 R-00 owner，并核验五类 schema 均由 consumer 注册。"""
        if not isinstance(relation_runtime, RelationClosureRuntime):
            raise TypeError("set relation closure runtime 类型错误")
        if not isinstance(protocol, SetRelationProtocol):
            raise TypeError("set relation protocol 类型错误")
        if not isinstance(budget, SetRelationBudget):
            raise TypeError("set relation budget 类型错误")
        if not isinstance(member_type_resolver, MemberTypeResolver):
            raise TypeError("member type resolver 协议不完整")
        resolver_key = member_type_resolver.state_key()
        _strict_key(
            resolver_key,
            label="MemberTypeResolver.state_key",
            allow_empty=True,
        )
        registered = {item.schema for item in relation_runtime.consumer.schemas}
        required = {
            protocol.subset_eq.schema.schema,
            protocol.proper_subset.schema.schema,
            protocol.member.schema.schema,
            protocol.equal.schema.schema,
            protocol.closed_domain.schema.schema,
        }
        if not required.issubset(registered):
            raise SetRelationRuntimeError("R-00 consumer 缺少 R-02 typed schema")
        self.relation_runtime = relation_runtime
        self.protocol = protocol
        self.budget = budget
        self.member_type_resolver = member_type_resolver

    @property
    def semantic_graph(self):
        """返回底层 R-00 使用的 S-00 SemanticGraph。"""
        return self.relation_runtime.semantic_graph

    def knowledge(self) -> SetRelationKnowledge:
        """从 R-00 forming 列表读取当前 active lifecycle 的五类关系快照。"""
        evidence: list[SetRelationEvidence] = []
        for snapshot in self.relation_runtime.epistemic_snapshots():
            spec = snapshot.formation.spec
            relation_protocol = self.protocol.relation_protocol(
                spec.proposition.predicate)
            if relation_protocol is None:
                continue
            if spec.schema != relation_protocol.schema:
                raise SetRelationRuntimeError(
                    "R-02 relation 使用了协议外或陈旧 schema")
            if snapshot.snapshot.lifecycle == LIFECYCLE_ACTIVE:
                state = LogicEvidenceState.from_status(
                    snapshot.snapshot.epistemic_status)
                current_evidence = snapshot.evidence
            elif (snapshot.snapshot.lifecycle == LIFECYCLE_ARCHIVED
                    and snapshot.snapshot.refute_evidence_ids):
                refute_ids = frozenset(snapshot.snapshot.refute_evidence_ids)
                state = LogicEvidenceState(False, True)
                current_evidence = tuple(
                    item for item in snapshot.evidence
                    if item.evidence_id in refute_ids
                )
            else:
                continue
            statement = self._statement_from_definition(
                spec.proposition,
                relation_protocol,
            )
            evidence.append(SetRelationEvidence(
                statement,
                spec.proposition.proposition,
                snapshot.snapshot.hypothesis,
                state,
                current_evidence,
                spec.forming_sources,
                snapshot.active_fact is not None,
            ))
        return SetRelationKnowledge(tuple(evidence))

    def engine(self) -> SetRelationEngine:
        """在当前不可变读取切片上构造一次有界纯集合闭包。"""
        return SetRelationEngine(
            self.protocol,
            self.budget,
            self.knowledge(),
            self.member_type_resolver,
        )

    def evaluate_many(
            self,
            queries: tuple[SetRelationQuery, ...],
            ) -> tuple[SetRelationRuntimeResult, ...]:
        """共享一次闭包评估多个 statement，并原子提交全部 active 前提 Use。"""
        if not isinstance(queries, tuple) or not queries:
            raise ValueError("set relation queries 必须是非空 tuple")
        if any(not isinstance(item, SetRelationQuery) for item in queries):
            raise TypeError("set relation query 类型错误")
        engine = self.engine()
        evaluations = tuple(engine.evaluate(item.statement) for item in queries)
        uses = self._consume_groups(tuple(
            (
                query.use_key,
                query.context,
                evaluation.active_premises(),
            )
            for query, evaluation in zip(queries, evaluations, strict=True)
        ))
        return tuple(
            SetRelationRuntimeResult(evaluation, group)
            for evaluation, group in zip(evaluations, uses, strict=True)
        )

    def evaluate(
            self, query: SetRelationQuery,
            ) -> SetRelationRuntimeResult:
        """评估一个集合关系查询，并按请求写入实际采用前提。"""
        return self.evaluate_many((query,))[0]

    def finite_domains_many(
            self,
            queries: tuple[SetRelationDomainQuery, ...],
            ) -> tuple[SetRelationDomainRuntimeResult, ...]:
        """共享一次闭包构造多个有限域，并原子提交闭域/MEMBER Use。"""
        if not isinstance(queries, tuple) or not queries:
            raise ValueError("set relation domain queries 必须是非空 tuple")
        if any(not isinstance(item, SetRelationDomainQuery) for item in queries):
            raise TypeError("set relation domain query 类型错误")
        engine = self.engine()
        domains = tuple(engine.finite_domain(item.domain) for item in queries)
        uses = self._consume_groups(tuple(
            (
                query.use_key,
                query.context,
                domain.active_premises(),
            )
            for query, domain in zip(queries, domains, strict=True)
        ))
        return tuple(
            SetRelationDomainRuntimeResult(domain, group)
            for domain, group in zip(domains, uses, strict=True)
        )

    def finite_domain(
            self, query: SetRelationDomainQuery,
            ) -> SetRelationDomainRuntimeResult:
        """构造一个来源化有限域并记录实际采用前提。"""
        return self.finite_domains_many((query,))[0]

    def quantify_exists(
            self,
            domain: SetRelationDomainResult,
            branches: tuple[SetQuantifierBranch, ...],
            ) -> SetQuantifierEvaluation:
        """使用当前协议的 EXISTS MinimalInstruction 聚合有限域分支。"""
        return self.engine().quantify_exists(domain, branches)

    def quantify_forall(
            self,
            domain: SetRelationDomainResult,
            branches: tuple[SetQuantifierBranch, ...],
            ) -> SetQuantifierEvaluation:
        """使用当前协议的 FORALL MinimalInstruction 聚合有限域分支。"""
        return self.engine().quantify_forall(domain, branches)

    def map_legacy(
            self,
            record: LegacySetRelationRecord,
            mapper: LegacySetRelationMapper,
            ) -> MappedLegacySetRelation | None:
        """调用显式 mapper，并强制旧记录只能进入 typed SUBSET_EQ 路径。"""
        if not isinstance(record, LegacySetRelationRecord):
            raise TypeError("legacy set relation record 类型错误")
        if not isinstance(mapper, LegacySetRelationMapper):
            raise TypeError("legacy set relation mapper 协议不完整")
        mapped = mapper.map(record)
        if mapped is None:
            return None
        if not isinstance(mapped, MappedLegacySetRelation):
            raise TypeError("legacy mapper 返回类型错误")
        spec = mapped.formation.spec
        if spec.proposition.predicate != self.protocol.subset_eq.relation:
            raise SetRelationRuntimeError(
                "旧关系记录只能映射为 typed SUBSET_EQ 候选")
        if spec.schema != self.protocol.subset_eq.schema:
            raise SetRelationRuntimeError("旧关系映射未使用当前 SUBSET_EQ schema")
        self.protocol.validate_statement(self._statement_from_definition(
            spec.proposition,
            self.protocol.subset_eq,
        ))
        return mapped

    def state_key(self) -> tuple:
        """返回 R-00 owner、R-02 协议、预算和类型 resolver 的完整状态。"""
        return (
            self.relation_runtime.state_key(),
            self.protocol.stable_key(),
            self.budget.stable_key(),
            self.member_type_resolver.state_key(),
        )

    def clone_for_context(self, ctx: TrainContext) -> "SetRelationRuntime":
        """复制 H-00 owner，并在评测 TrainContext 的克隆图上重建 R-02 facade。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("set relation clone ctx 类型错误")
        predicate_identities = tuple(
            self.semantic_graph.ontology.identity_of(ref)
            for ref in self.semantic_graph.predicates.refs()
        )
        semantic_graph = SemanticGraph(
            ctx.graph_ontology,
            AtomicPropositionPredicates(*tuple(
                ctx.graph_ontology.materialize(identity)
                for identity in predicate_identities
            )),
        )
        candidate_graph = CandidateProjectionGraph(
            ctx.graph_ontology,
            self.relation_runtime.candidate_runtime.graph.protocol,
        )
        resolver = self.member_type_resolver.clone_for_evaluation()
        if not isinstance(resolver, MemberTypeResolver):
            raise TypeError("member type resolver clone 协议不完整")
        if resolver.state_key() != self.member_type_resolver.state_key():
            raise ValueError("member type resolver clone 改变了映射状态")
        return SetRelationRuntime(
            self.relation_runtime.clone_for_evaluation(
                semantic_graph,
                candidate_graph,
            ),
            self.protocol,
            self.budget,
            resolver,
        )

    def _statement_from_definition(
            self,
            definition: AtomicPropositionDefinition,
            relation_protocol: (
                SetBinaryRelationProtocol | SetUnaryRelationProtocol),
            ) -> SetRelationStatement:
        """按注入 Role 从 S-00 定义恢复集合关系内容，不依赖 binding 顺序。"""
        if isinstance(relation_protocol, SetUnaryRelationProtocol):
            statement = SetRelationStatement(
                relation_protocol.relation,
                _single_filler(definition, relation_protocol.value_role),
            )
        else:
            statement = SetRelationStatement(
                relation_protocol.relation,
                _single_filler(definition, relation_protocol.left_role),
                _single_filler(definition, relation_protocol.right_role),
            )
        return self.protocol.validate_statement(statement)

    def _consume_groups(self, groups: tuple[tuple, ...]) -> tuple[tuple, ...]:
        """把多个查询的 active 前提合并为一次 R-00 原子 Use 提交。"""
        requests = []
        owners = []
        sizes = []
        for group_index, group in enumerate(groups):
            use_key, context, premises = group
            if use_key is None:
                sizes.append(0)
                continue
            _strict_key(use_key, label="set relation consume use_key")
            sizes.append(len(premises))
            for ordinal, premise in enumerate(premises):
                requests.append((
                    premise.proposition,
                    (*use_key, ordinal),
                    context,
                ))
                owners.append(group_index)
        if not requests:
            return tuple(() for _group in groups)
        committed = self.relation_runtime.consume_many(tuple(requests))
        result: list[list[RelationClosureUse]] = [
            [] for _group in groups
        ]
        for owner, use in zip(owners, committed, strict=True):
            result[owner].append(use)
        if any(len(result[index]) != size for index, size in enumerate(sizes)):
            raise SetRelationRuntimeError("R-00 Use 提交数量与查询前提不一致")
        return tuple(tuple(items) for items in result)


@dataclass(frozen=True)
class SetRelationFormationRequest:
    """课程显式提交的一次 S-00 定义和 R-00 forming 请求。"""

    spec: RelationClosureCandidateSpec
    scope: ScopeIdentity
    provenance_kind: int
    epistemic_origin: int = 0
    content_version: int = 0
    qualifiers: tuple[int, ...] = ()
    timestamp_base: int = 0

    def __post_init__(self) -> None:
        """核验 typed spec、来源 scope、写入元数据和 forming 逻辑序。"""
        if not isinstance(self.spec, RelationClosureCandidateSpec):
            raise TypeError("set relation formation spec 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("set relation formation scope 类型错误")
        if self.scope.source != self.spec.proposition.source:
            raise ValueError("set relation formation scope 必须绑定 Proposition 来源")
        if not isinstance(self.qualifiers, tuple):
            raise TypeError("set relation formation qualifiers 必须是 tuple")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            self.timestamp_base,
            *self.qualifiers,
            _where="SetRelationFormationRequest",
        )
        if (type(self.provenance_kind) is not int
                or self.provenance_kind <= 0
                or type(self.epistemic_origin) is not int
                or self.epistemic_origin < 0
                or type(self.content_version) is not int
                or self.content_version < 0
                or type(self.timestamp_base) is not int
                or self.timestamp_base < 0
                or any(type(item) is not int for item in self.qualifiers)):
            raise ValueError("set relation formation 元数据非法")

    def metadata(self) -> dict:
        """返回 SemanticGraph.define_atomic 使用的显式来源元数据。"""
        return {
            "provenance_kind": self.provenance_kind,
            "epistemic_origin": self.epistemic_origin,
            "content_version": self.content_version,
            "qualifiers": self.qualifiers,
        }


@dataclass(frozen=True)
class SetRelationRoundRequest:
    """一个来源 scope 中互斥的学习写入轮或关系查询采用轮。"""

    scope: ScopeIdentity
    legacy_records: tuple[LegacySetRelationRecord, ...] = ()
    formations: tuple[SetRelationFormationRequest, ...] = ()
    recognitions: tuple[RelationClosureRecognitionInput, ...] = ()
    queries: tuple[SetRelationQuery, ...] = ()
    domain_queries: tuple[SetRelationDomainQuery, ...] = ()

    def __post_init__(self) -> None:
        """拒绝写入与查询混轮、错误 scope 和重复幂等路由。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("set relation round scope 类型错误")
        groups = (
            (self.legacy_records, LegacySetRelationRecord, "legacy_records"),
            (self.formations, SetRelationFormationRequest, "formations"),
            (self.recognitions, RelationClosureRecognitionInput, "recognitions"),
            (self.queries, SetRelationQuery, "queries"),
            (self.domain_queries, SetRelationDomainQuery, "domain_queries"),
        )
        for values, expected, label in groups:
            if not isinstance(values, tuple):
                raise TypeError(f"set relation round {label} 必须是 tuple")
            if any(not isinstance(item, expected) for item in values):
                raise TypeError(f"set relation round {label} 元素类型错误")
        writes = bool(self.legacy_records or self.formations or self.recognitions)
        reads = bool(self.queries or self.domain_queries)
        if writes and reads:
            raise ValueError("set relation 学习写入轮与查询采用轮必须分开")
        if any(item.scope != self.scope for item in self.legacy_records):
            raise ValueError("legacy set relation record 必须绑定当前 round scope")
        if any(item.scope != self.scope for item in self.formations):
            raise ValueError("set relation formation 必须绑定当前 round scope")
        if any(item.scope != self.scope for item in self.recognitions):
            raise ValueError("set relation recognition 必须绑定当前 round scope")
        query_contexts = tuple(
            item.context
            for item in (*self.queries, *self.domain_queries)
            if item.context is not None
        )
        if any(item.scope != self.scope for item in query_contexts):
            raise ValueError("set relation query context 必须绑定当前 round scope")
        propositions = tuple(
            item.spec.proposition.proposition for item in self.formations)
        if len(set(propositions)) != len(propositions):
            raise ValueError("set relation round 不得重复 forming Proposition")
        routes = tuple(item.route_key() for item in self.recognitions)
        if len(set(routes)) != len(routes):
            raise ValueError("set relation round 不得重复 recognition 路由")


@dataclass(frozen=True)
class SetRelationRoundReport:
    """一次 R-02 课程轮的写入、查询和有限域结果。"""

    scope: ScopeIdentity
    read_only: bool
    formations: tuple[RelationClosureFormationTrace, ...]
    recognitions: tuple[RelationClosureRecognitionTrace, ...]
    query_results: tuple[SetRelationRuntimeResult, ...]
    domain_results: tuple[SetRelationDomainRuntimeResult, ...]

    def __post_init__(self) -> None:
        """核验报告容器完整，read-only 轮不得包含学习写入。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("set relation report scope 类型错误")
        if type(self.read_only) is not bool:
            raise TypeError("set relation report read_only 必须是严格 bool")
        checks = (
            (self.formations, RelationClosureFormationTrace),
            (self.recognitions, RelationClosureRecognitionTrace),
            (self.query_results, SetRelationRuntimeResult),
            (self.domain_results, SetRelationDomainRuntimeResult),
        )
        for values, expected in checks:
            if not isinstance(values, tuple):
                raise TypeError("set relation report 字段必须是 tuple")
            if any(not isinstance(item, expected) for item in values):
                raise TypeError("set relation report 元素类型错误")
        if self.read_only and (self.formations or self.recognitions):
            raise ValueError("read-only set relation report 不得含学习写入")


@runtime_checkable
class SetRelationRuntimeBuilder(Protocol):
    """由项目课程注入完整 R-00 owner 和 R-02 纯语义组件。"""

    def build(self, ctx: TrainContext) -> SetRelationRuntime:
        """在指定 TrainContext 图上构造可恢复的集合关系 owner。"""
        ...

    def clone_for_evaluation(self) -> "SetRelationRuntimeBuilder":
        """返回清除宿主可变引用的评测 builder。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回协议、schema、预算和课程版本的完整整数键。"""
        ...


@runtime_checkable
class SetRelationCourse(Protocol):
    """把来源 scope 映射为 typed 集合关系学习轮或查询轮。"""

    def request(
            self, scope: ScopeIdentity, *, read_only: bool,
            ) -> SetRelationRoundRequest:
        """返回当前来源的 typed R-02 请求，不向宿主暴露表层规则。"""
        ...

    def legacy_mapper(self) -> LegacySetRelationMapper | None:
        """需要迁移旧边时返回显式 mapper，否则返回 None。"""
        ...

    def clone_for_evaluation(self) -> "SetRelationCourse":
        """返回不共享可变课程状态的评测副本。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回课程 mapper、版本和来源策略的完整整数键。"""
        ...


class SetRelationCourseRuntime:
    """让 formal round 只提交 scope，由课程完成 R-02 学习或查询。"""

    def __init__(
            self,
            ctx: TrainContext,
            owner: SetRelationRuntime,
            builder: SetRelationRuntimeBuilder,
            course: SetRelationCourse,
            ) -> None:
        """绑定当前 TrainContext、唯一 owner、可克隆 builder 和课程。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("set relation course ctx 类型错误")
        if not isinstance(owner, SetRelationRuntime):
            raise TypeError("set relation course owner 类型错误")
        if not isinstance(builder, SetRelationRuntimeBuilder):
            raise TypeError("set relation builder 协议不完整")
        if not isinstance(course, SetRelationCourse):
            raise TypeError("set relation course 协议不完整")
        _strict_key(builder.state_key(), label="SetRelationRuntimeBuilder.state_key")
        _strict_key(course.state_key(), label="SetRelationCourse.state_key")
        self.ctx = ctx
        self.owner = owner
        self.builder = builder
        self.course = course

    def process(
            self, scope: ScopeIdentity, *, read_only: bool,
            ) -> SetRelationRoundReport:
        """全量预检后执行一个纯学习写入轮或纯查询采用轮。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("set relation process scope 类型错误")
        if type(read_only) is not bool:
            raise TypeError("set relation process read_only 必须是严格 bool")
        request = self.course.request(scope, read_only=read_only)
        if not isinstance(request, SetRelationRoundRequest):
            raise TypeError("set relation course.request 返回类型错误")
        if request.scope != scope:
            raise ValueError("set relation course.request 替换了 round scope")
        if read_only and (
                request.legacy_records
                or request.formations
                or request.recognitions):
            raise ValueError("read-only set relation 请求不得学习或迁移旧边")

        formations = list(request.formations)
        recognitions = list(request.recognitions)
        if request.legacy_records:
            mapper = self.course.legacy_mapper()
            if mapper is None:
                raise SetRelationRuntimeError("legacy 记录缺少显式 mapper")
            if not isinstance(mapper, LegacySetRelationMapper):
                raise TypeError("legacy mapper 协议不完整")
            for record in request.legacy_records:
                mapped = self.owner.map_legacy(record, mapper)
                if mapped is None:
                    raise SetRelationRuntimeError(
                        "legacy mapper 无法补全 typed SUBSET_EQ")
                formations.append(mapped.formation)
                recognitions.append(mapped.recognition)
        self._validate_combined(formations, recognitions)

        formation_requests = tuple(
            (item.spec, item.timestamp_base) for item in formations)
        timestamps = ()
        if recognitions:
            next_timestamp = (
                self.owner.relation_runtime.candidate_runtime.next_timestamps(1)[0]
            )
            formation_end = max((
                item.timestamp_base + len(item.spec.forming_sources) - 1
                for item in formations
            ), default=next_timestamp - 1)
            start = max(next_timestamp, formation_end + 1)
            timestamps = tuple(range(
                start,
                start + len(recognitions) * 3,
            ))
        recognition_requests = tuple(
            (
                recognition,
                timestamps[index * 3],
                timestamps[index * 3 + 1],
                timestamps[index * 3 + 2],
            )
            for index, recognition in enumerate(recognitions)
        )
        for item in formations:
            self.owner.semantic_graph.preflight_atomic(
                item.spec.proposition,
                scope=item.scope,
                **item.metadata(),
            )
            relation_protocol = self.owner.protocol.relation_protocol(
                item.spec.proposition.predicate)
            if relation_protocol is None or item.spec.schema != relation_protocol.schema:
                raise SetRelationRuntimeError(
                    "set relation formation 未使用当前 typed schema")
        if formation_requests or recognition_requests:
            self.owner.relation_runtime.preflight_many(
                formation_requests,
                recognition_requests,
            )
        for item in formations:
            self.owner.semantic_graph.define_atomic(
                item.spec.proposition,
                scope=item.scope,
                **item.metadata(),
            )
        formation_traces = (
            self.owner.relation_runtime.form_many(formation_requests)
            if formation_requests else ()
        )
        recognition_traces = (
            self.owner.relation_runtime.recognize_many_at(
                recognition_requests)
            if recognition_requests else ()
        )
        query_results = (
            self.owner.evaluate_many(request.queries)
            if request.queries else ()
        )
        domain_results = (
            self.owner.finite_domains_many(request.domain_queries)
            if request.domain_queries else ()
        )
        return SetRelationRoundReport(
            scope,
            read_only,
            formation_traces,
            recognition_traces,
            query_results,
            domain_results,
        )

    def clone_for_context(
            self, ctx: TrainContext,
            ) -> "SetRelationCourseRuntime":
        """用克隆 builder/course 在评测 context 上重建独立 R-02 owner。"""
        cloned_builder = self.builder.clone_for_evaluation()
        cloned_course = self.course.clone_for_evaluation()
        if not isinstance(cloned_builder, SetRelationRuntimeBuilder):
            raise TypeError("set relation builder clone 协议不完整")
        if not isinstance(cloned_course, SetRelationCourse):
            raise TypeError("set relation course clone 协议不完整")
        return SetRelationCourseRuntime(
            ctx,
            self.owner.clone_for_context(ctx),
            cloned_builder,
            cloned_course,
        )

    def state_key(self) -> tuple:
        """返回 builder、课程、owner 和可选 legacy mapper 的完整状态。"""
        mapper = self.course.legacy_mapper()
        mapper_key = ()
        if mapper is not None:
            if not isinstance(mapper, LegacySetRelationMapper):
                raise TypeError("legacy mapper 协议不完整")
            mapper_key = mapper.state_key()
            _strict_key(mapper_key, label="LegacySetRelationMapper.state_key")
        return (
            self.builder.state_key(),
            self.course.state_key(),
            mapper_key,
            self.owner.state_key(),
        )

    @staticmethod
    def _validate_combined(
            formations: list[SetRelationFormationRequest],
            recognitions: list[RelationClosureRecognitionInput],
            ) -> None:
        """在 legacy 映射后重新拒绝重复 Proposition 和 recognition 路由。"""
        propositions = tuple(
            item.spec.proposition.proposition for item in formations)
        if len(set(propositions)) != len(propositions):
            raise ValueError("set relation combined forming Proposition 重复")
        routes = tuple(item.route_key() for item in recognitions)
        if len(set(routes)) != len(routes):
            raise ValueError("set relation combined recognition 路由重复")


def install_set_relation_runtime(
        ctx: TrainContext,
        builder: SetRelationRuntimeBuilder,
        course: SetRelationCourse,
        ) -> SetRelationCourseRuntime:
    """在 TrainContext 上安装显式成对注入且默认关闭的 R-02 runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("install set relation ctx 类型错误")
    if not isinstance(builder, SetRelationRuntimeBuilder):
        raise TypeError("set relation builder 协议不完整")
    if not isinstance(course, SetRelationCourse):
        raise TypeError("set relation course 协议不完整")
    if getattr(ctx, "set_relation_runtime", None) is not None:
        raise ValueError("TrainContext 已安装 set relation runtime")
    _strict_key(builder.state_key(), label="SetRelationRuntimeBuilder.state_key")
    _strict_key(course.state_key(), label="SetRelationCourse.state_key")
    owner = builder.build(ctx)
    if not isinstance(owner, SetRelationRuntime):
        raise TypeError("set relation builder.build 返回类型错误")
    if owner.semantic_graph.ontology is not ctx.graph_ontology:
        raise ValueError("set relation owner 未绑定当前 TrainContext 图")
    runtime = SetRelationCourseRuntime(ctx, owner, builder, course)
    ctx.set_relation_runtime = runtime
    return runtime


__all__ = [
    "LegacySetRelationMapper",
    "LegacySetRelationRecord",
    "MappedLegacySetRelation",
    "SetRelationCourse",
    "SetRelationCourseRuntime",
    "SetRelationDomainQuery",
    "SetRelationDomainRuntimeResult",
    "SetRelationFormationRequest",
    "SetRelationQuery",
    "SetRelationRoundReport",
    "SetRelationRoundRequest",
    "SetRelationRuntime",
    "SetRelationRuntimeBuilder",
    "SetRelationRuntimeError",
    "SetRelationRuntimeResult",
    "install_set_relation_runtime",
]
