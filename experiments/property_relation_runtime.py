"""R-03 PROPERTY 的 R-00 适配、课程编排和 V-06 重建协议。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
)
from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.property_relation import (
    PropertyClaim,
    PropertyEvidence,
    PropertyIntensityResolver,
    PropertyKnowledge,
    PropertyPattern,
    PropertyQueryBudget,
    PropertyRelationBudgetExceeded,
    PropertyRelationEngine,
    PropertyRelationProtocol,
    PropertySelection,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    RelationClosureCandidateSpec,
)
from pure_integer_ai.cognition.shared.relation_use import RelationUseContext
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
    SemanticGraph,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    semantic_source,
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


class PropertyRelationRuntimeError(RuntimeError):
    """R-03 owner、课程或旧边映射不满足完整协议。"""


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


def _single_filler(
        definition: AtomicPropositionDefinition,
        role: ObjectIdentity,
        ) -> ObjectIdentity:
    """读取命题中指定 Role 的唯一 filler，缺失或竞争均失败。"""
    matches = tuple(
        binding.filler
        for binding in definition.bindings
        if binding.role == role
    )
    if len(matches) != 1:
        raise PropertyRelationRuntimeError(
            "property Role 必须恰有一个 filler")
    return matches[0]


@dataclass(frozen=True)
class PropertyQuery:
    """一次 PROPERTY 模式查询及可选来源化采用路由。"""

    pattern: PropertyPattern
    use_key: tuple[int, ...] | None = None
    context: RelationUseContext | None = None

    def __post_init__(self) -> None:
        """核验查询模式和 R-00 Use 路由配对关系。"""
        if not isinstance(self.pattern, PropertyPattern):
            raise TypeError("property query pattern 类型错误")
        if self.use_key is not None:
            _strict_key(self.use_key, label="PropertyQuery.use_key")
        if self.context is not None and not isinstance(
                self.context, RelationUseContext):
            raise TypeError("property query context 类型错误")
        if self.context is not None and self.use_key is None:
            raise ValueError("property query context 必须配套 use_key")


@dataclass(frozen=True)
class PropertyRuntimeResult:
    """一次 PROPERTY 选择结果和实际提交的 R-00 Use。"""

    selection: PropertySelection
    uses: tuple[RelationClosureUse, ...]

    def __post_init__(self) -> None:
        """核验选择结果与采用记录类型。"""
        if not isinstance(self.selection, PropertySelection):
            raise TypeError("property runtime selection 类型错误")
        if not isinstance(self.uses, tuple) or any(
                not isinstance(item, RelationClosureUse) for item in self.uses):
            raise TypeError("property runtime uses 类型错误")
        if self.selection.selected() is None and self.uses:
            raise ValueError("property 非唯一选择不得写 Use")


@dataclass(frozen=True)
class LegacyPropertyRecord:
    """旧 PROPERTY 存储的一条不带新 typed 语义的来源化记录。"""

    relation_key: tuple[int, ...]
    subject_key: tuple[int, ...]
    attribute_key: tuple[int, ...]
    value_key: tuple[int, ...]
    polarity_key: tuple[int, ...]
    modality_key: tuple[int, ...]
    intensity_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """保留旧键和来源，不解释 EDGE_PROPERTY 或 ATTR 数值。"""
        for label, value in (
                ("relation", self.relation_key),
                ("subject", self.subject_key),
                ("attribute", self.attribute_key),
                ("value", self.value_key),
                ("polarity", self.polarity_key),
                ("modality", self.modality_key),
                ("intensity", self.intensity_key),
                ("trace", self.trace)):
            _strict_key(value, label=f"legacy property {label}")
        if not isinstance(self.source, SourceRef):
            raise TypeError("legacy property source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("legacy property scope 类型错误")
        if self.scope.source != self.source:
            raise ValueError("legacy property scope 必须绑定记录来源")


@dataclass(frozen=True)
class PropertyFormationRequest:
    """课程提交的一次 S-00 PROPERTY 定义和 R-00 forming 请求。"""

    spec: RelationClosureCandidateSpec
    scope: ScopeIdentity
    provenance_kind: int
    epistemic_origin: int = 0
    content_version: int = 0
    qualifiers: tuple[int, ...] = ()
    timestamp_base: int = 0

    def __post_init__(self) -> None:
        """核验 typed spec、来源 scope、元数据和 forming 逻辑序。"""
        if not isinstance(self.spec, RelationClosureCandidateSpec):
            raise TypeError("property formation spec 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("property formation scope 类型错误")
        if self.scope.source != self.spec.proposition.source:
            raise ValueError("property formation scope 必须绑定 Proposition 来源")
        if not isinstance(self.qualifiers, tuple):
            raise TypeError("property formation qualifiers 必须是 tuple")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            self.timestamp_base,
            *self.qualifiers,
            _where="PropertyFormationRequest",
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
            raise ValueError("property formation 元数据非法")

    def metadata(self) -> dict:
        """返回 SemanticGraph 定义写入所需显式来源元数据。"""
        return {
            "provenance_kind": self.provenance_kind,
            "epistemic_origin": self.epistemic_origin,
            "content_version": self.content_version,
            "qualifiers": self.qualifiers,
        }


@dataclass(frozen=True)
class MappedLegacyProperty:
    """旧记录经 mapper 补全后的 typed PROPERTY forming 和 recognition。"""

    formation: PropertyFormationRequest
    recognition: RelationClosureRecognitionInput

    def __post_init__(self) -> None:
        """要求 forming 与 recognition 指向同一完整 Proposition。"""
        if not isinstance(self.formation, PropertyFormationRequest):
            raise TypeError("mapped property formation 类型错误")
        if not isinstance(self.recognition, RelationClosureRecognitionInput):
            raise TypeError("mapped property recognition 类型错误")
        if (self.formation.spec.proposition.proposition
                != self.recognition.proposition):
            raise ValueError("mapped property forming 与 recognition 不一致")


@runtime_checkable
class LegacyPropertyMapper(Protocol):
    """把旧 PROPERTY/ATTR 记录显式映射为完整 typed 候选。"""

    def map(self, record: LegacyPropertyRecord) -> MappedLegacyProperty | None:
        """无法补全六维 filler、来源或 Evidence 时返回 None。"""
        ...

    def clone_for_evaluation(self) -> "LegacyPropertyMapper":
        """返回不共享可变状态的评测 mapper。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回 mapper 版本和来源策略的完整整数键。"""
        ...


class PropertyRelationRuntime:
    """把 R-00 当前关系快照适配为有界 PROPERTY 选择和 Use。"""

    def __init__(
            self,
            relation_runtime: RelationClosureRuntime,
            protocol: PropertyRelationProtocol,
            budget: PropertyQueryBudget,
            intensity_resolver: PropertyIntensityResolver,
            ) -> None:
        """绑定同一 R-00 owner，并核验 PROPERTY schema 与 resolver。"""
        if not isinstance(relation_runtime, RelationClosureRuntime):
            raise TypeError("property relation closure runtime 类型错误")
        if not isinstance(protocol, PropertyRelationProtocol):
            raise TypeError("property relation protocol 类型错误")
        if not isinstance(budget, PropertyQueryBudget):
            raise TypeError("property query budget 类型错误")
        if not isinstance(intensity_resolver, PropertyIntensityResolver):
            raise TypeError("property intensity resolver 协议不完整")
        resolver_key = intensity_resolver.state_key()
        _strict_key(
            resolver_key,
            label="PropertyIntensityResolver.state_key",
            allow_empty=True,
        )
        registered = {item.schema for item in relation_runtime.consumer.schemas}
        if protocol.schema.schema not in registered:
            raise PropertyRelationRuntimeError(
                "R-00 consumer 缺少 R-03 PROPERTY schema")
        self.relation_runtime = relation_runtime
        self.protocol = protocol
        self.budget = budget
        self.intensity_resolver = intensity_resolver

    @property
    def semantic_graph(self) -> SemanticGraph:
        """返回底层 R-00 使用的 S-00 SemanticGraph。"""
        return self.relation_runtime.semantic_graph

    def claim_from_definition(
            self, definition: AtomicPropositionDefinition,
            ) -> PropertyClaim:
        """按注入 Role 恢复六维 claim，不依赖 binding 顺序。"""
        self.protocol.schema.validate_definition(definition)
        return PropertyClaim(*tuple(
            _single_filler(definition, role)
            for role in self.protocol.roles()
        ))

    def knowledge(self) -> PropertyKnowledge:
        """从 R-00 当前 lifecycle 恢复 PROPERTY 四态 Evidence 快照。"""
        evidence = []
        for formation in self.relation_runtime.formation_traces():
            spec = formation.spec
            if spec.proposition.predicate != self.protocol.relation:
                continue
            snapshot = self.relation_runtime.snapshot_for_proposition(
                spec.proposition.proposition)
            if spec.schema != self.protocol.schema:
                raise PropertyRelationRuntimeError(
                    "PROPERTY relation 使用了协议外或陈旧 schema")
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
            proposition_ref = self.semantic_graph.ontology.resolve(
                spec.proposition.proposition)
            if proposition_ref is None:
                raise PropertyRelationRuntimeError(
                    "PROPERTY forming Proposition 未进入权威语义图")
            materialized = self.semantic_graph.read_atomic(proposition_ref)
            if materialized.definition != spec.proposition:
                raise PropertyRelationRuntimeError(
                    "PROPERTY forming 与语义图定义不一致")
            evidence.append(PropertyEvidence(
                self.claim_from_definition(spec.proposition),
                spec.proposition.proposition,
                materialized.definition.context,
                materialized.scope,
                snapshot.snapshot.hypothesis,
                state,
                current_evidence,
                spec.forming_sources,
                snapshot.active_fact is not None,
            ))
            if len(evidence) > self.budget.max_direct_facts:
                raise PropertyRelationBudgetExceeded(
                    "property 直接事实预算耗尽")
        return PropertyKnowledge(tuple(evidence))

    def engine(self) -> PropertyRelationEngine:
        """在当前不可变快照上构造一次有界 PROPERTY engine。"""
        return PropertyRelationEngine(
            self.protocol,
            self.budget,
            self.knowledge(),
            self.intensity_resolver,
        )

    def select_many(
            self,
            queries: tuple[PropertyQuery, ...],
            ) -> tuple[PropertyRuntimeResult, ...]:
        """共享一次快照查询多个模式，并原子采用唯一选项前提。"""
        if not isinstance(queries, tuple) or not queries:
            raise ValueError("property queries 必须是非空 tuple")
        if any(not isinstance(item, PropertyQuery) for item in queries):
            raise TypeError("property query 类型错误")
        engine = self.engine()
        selections = tuple(engine.select(item.pattern) for item in queries)
        groups = []
        for query, selection in zip(queries, selections, strict=True):
            selected = selection.selected()
            premises = (
                () if selected is None
                else selected.evaluation.active_premises()
            )
            if selected is not None and not premises:
                raise PropertyRelationRuntimeError(
                    "唯一 PROPERTY 选择缺少 active supported 前提")
            groups.append((query.use_key, query.context, premises))
        uses = self._consume_groups(tuple(groups))
        return tuple(
            PropertyRuntimeResult(selection, group)
            for selection, group in zip(selections, uses, strict=True)
        )

    def select(self, query: PropertyQuery) -> PropertyRuntimeResult:
        """执行一个 PROPERTY 模式查询并按需写 Use。"""
        return self.select_many((query,))[0]

    def map_legacy(
            self,
            record: LegacyPropertyRecord,
            mapper: LegacyPropertyMapper,
            ) -> MappedLegacyProperty | None:
        """调用显式 mapper，并核验结果为当前完整 PROPERTY schema。"""
        if not isinstance(record, LegacyPropertyRecord):
            raise TypeError("legacy property record 类型错误")
        if not isinstance(mapper, LegacyPropertyMapper):
            raise TypeError("legacy property mapper 协议不完整")
        mapped = mapper.map(record)
        if mapped is None:
            return None
        if not isinstance(mapped, MappedLegacyProperty):
            raise TypeError("legacy property mapper 返回类型错误")
        spec = mapped.formation.spec
        if (spec.proposition.predicate != self.protocol.relation
                or spec.schema != self.protocol.schema):
            raise PropertyRelationRuntimeError(
                "旧 PROPERTY 记录未映射为当前 typed schema")
        if (semantic_source(spec.proposition.proposition) != record.source
                or mapped.formation.scope != record.scope):
            raise PropertyRelationRuntimeError(
                "旧 PROPERTY mapper 不得替换记录来源或 scope")
        claim = self.claim_from_definition(spec.proposition)
        if self.intensity_resolver.resolve(claim.intensity) is None:
            raise PropertyRelationRuntimeError(
                "旧 PROPERTY intensity 缺少 Rational 解释")
        return mapped

    def state_key(self) -> tuple:
        """返回 R-00 owner、PROPERTY 协议、预算和 resolver 状态。"""
        return (
            self.relation_runtime.state_key(),
            self.protocol.stable_key(),
            self.budget.stable_key(),
            self.intensity_resolver.state_key(),
        )

    def clone_for_context(self, ctx: TrainContext) -> "PropertyRelationRuntime":
        """复制 H-00 owner，并在评测克隆图上重建 R-03 facade。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("property clone ctx 类型错误")
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
        resolver = self.intensity_resolver.clone_for_evaluation()
        if not isinstance(resolver, PropertyIntensityResolver):
            raise TypeError("property intensity resolver clone 协议不完整")
        if resolver.state_key() != self.intensity_resolver.state_key():
            raise ValueError("property intensity resolver clone 改变映射状态")
        return PropertyRelationRuntime(
            self.relation_runtime.clone_for_evaluation(
                semantic_graph,
                candidate_graph,
            ),
            self.protocol,
            self.budget,
            resolver,
        )

    def _consume_groups(self, groups: tuple[tuple, ...]) -> tuple[tuple, ...]:
        """把多个查询的唯一 active 前提合并为一次 R-00 Use 提交。"""
        requests = []
        owners = []
        sizes = []
        for group_index, (use_key, context, premises) in enumerate(groups):
            if use_key is None:
                sizes.append(0)
                continue
            _strict_key(use_key, label="property consume use_key")
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
            raise PropertyRelationRuntimeError(
                "PROPERTY Use 提交数量与选择前提不一致")
        return tuple(tuple(items) for items in result)


@dataclass(frozen=True)
class PropertyRoundRequest:
    """一个来源 scope 中互斥的 PROPERTY 学习轮或查询轮。"""

    scope: ScopeIdentity
    legacy_records: tuple[LegacyPropertyRecord, ...] = ()
    formations: tuple[PropertyFormationRequest, ...] = ()
    recognitions: tuple[RelationClosureRecognitionInput, ...] = ()
    queries: tuple[PropertyQuery, ...] = ()

    def __post_init__(self) -> None:
        """拒绝混轮、错误 scope、重复 Proposition 和 recognition 路由。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("property round scope 类型错误")
        groups = (
            (self.legacy_records, LegacyPropertyRecord, "legacy_records"),
            (self.formations, PropertyFormationRequest, "formations"),
            (self.recognitions, RelationClosureRecognitionInput, "recognitions"),
            (self.queries, PropertyQuery, "queries"),
        )
        for values, expected, label in groups:
            if not isinstance(values, tuple):
                raise TypeError(f"property round {label} 必须是 tuple")
            if any(not isinstance(item, expected) for item in values):
                raise TypeError(f"property round {label} 元素类型错误")
        writes = bool(self.legacy_records or self.formations or self.recognitions)
        if writes and self.queries:
            raise ValueError("property 学习写入轮与查询采用轮必须分开")
        if any(item.scope != self.scope for item in self.legacy_records):
            raise ValueError("legacy property record 必须绑定当前 scope")
        if any(item.scope != self.scope for item in self.formations):
            raise ValueError("property formation 必须绑定当前 scope")
        if any(item.scope != self.scope for item in self.recognitions):
            raise ValueError("property recognition 必须绑定当前 scope")
        contexts = tuple(
            item.context for item in self.queries if item.context is not None
        )
        if any(item.scope != self.scope for item in contexts):
            raise ValueError("property query context 必须绑定当前 scope")
        propositions = tuple(
            item.spec.proposition.proposition for item in self.formations)
        if len(set(propositions)) != len(propositions):
            raise ValueError("property round 不得重复 forming Proposition")
        routes = tuple(item.route_key() for item in self.recognitions)
        if len(set(routes)) != len(routes):
            raise ValueError("property round 不得重复 recognition 路由")


@dataclass(frozen=True)
class PropertyRoundReport:
    """一次 R-03 课程轮的 forming、recognition 和查询结果。"""

    scope: ScopeIdentity
    read_only: bool
    formations: tuple[RelationClosureFormationTrace, ...]
    recognitions: tuple[RelationClosureRecognitionTrace, ...]
    query_results: tuple[PropertyRuntimeResult, ...]

    def __post_init__(self) -> None:
        """核验报告类型，read-only 轮不得包含学习写入。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("property report scope 类型错误")
        if type(self.read_only) is not bool:
            raise TypeError("property report read_only 必须是严格 bool")
        checks = (
            (self.formations, RelationClosureFormationTrace),
            (self.recognitions, RelationClosureRecognitionTrace),
            (self.query_results, PropertyRuntimeResult),
        )
        for values, expected in checks:
            if not isinstance(values, tuple) or any(
                    not isinstance(item, expected) for item in values):
                raise TypeError("property report 字段类型错误")
        if self.read_only and (self.formations or self.recognitions):
            raise ValueError("read-only property report 不得含学习写入")


@runtime_checkable
class PropertyRuntimeBuilder(Protocol):
    """由项目课程注入完整 R-00 owner 和 R-03 纯语义组件。"""

    def build(self, ctx: TrainContext) -> PropertyRelationRuntime:
        """在指定 TrainContext 图上构造 PROPERTY owner。"""
        ...

    def clone_for_evaluation(self) -> "PropertyRuntimeBuilder":
        """返回清除宿主可变引用的评测 builder。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回协议、schema、预算和 resolver 版本完整键。"""
        ...


@runtime_checkable
class PropertyCourse(Protocol):
    """把来源 scope 映射为 typed PROPERTY 学习轮或查询轮。"""

    def request(
            self, scope: ScopeIdentity, *, read_only: bool,
            ) -> PropertyRoundRequest:
        """返回当前来源的 typed R-03 请求。"""
        ...

    def legacy_mapper(self) -> LegacyPropertyMapper | None:
        """需要迁移旧记录时返回显式 mapper，否则返回 None。"""
        ...

    def clone_for_evaluation(self) -> "PropertyCourse":
        """返回不共享可变课程状态的评测副本。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回课程、mapper 和来源策略的完整整数键。"""
        ...


class PropertyCourseRuntime:
    """让 formal round 只提交 scope，由课程完成 R-03 学习或查询。"""

    def __init__(
            self,
            ctx: TrainContext,
            owner: PropertyRelationRuntime,
            builder: PropertyRuntimeBuilder,
            course: PropertyCourse,
            ) -> None:
        """绑定当前 context、唯一 owner、可克隆 builder 和课程。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("property course ctx 类型错误")
        if not isinstance(owner, PropertyRelationRuntime):
            raise TypeError("property course owner 类型错误")
        if not isinstance(builder, PropertyRuntimeBuilder):
            raise TypeError("property builder 协议不完整")
        if not isinstance(course, PropertyCourse):
            raise TypeError("property course 协议不完整")
        _strict_key(builder.state_key(), label="PropertyRuntimeBuilder.state_key")
        _strict_key(course.state_key(), label="PropertyCourse.state_key")
        self.ctx = ctx
        self.owner = owner
        self.builder = builder
        self.course = course

    def process(
            self, scope: ScopeIdentity, *, read_only: bool,
            ) -> PropertyRoundReport:
        """全量预检后执行一个纯学习写入轮或纯查询采用轮。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("property process scope 类型错误")
        if type(read_only) is not bool:
            raise TypeError("property process read_only 必须是严格 bool")
        request = self.course.request(scope, read_only=read_only)
        if not isinstance(request, PropertyRoundRequest):
            raise TypeError("property course.request 返回类型错误")
        if request.scope != scope:
            raise ValueError("property course.request 替换了 round scope")
        if read_only and (
                request.legacy_records
                or request.formations
                or request.recognitions):
            raise ValueError("read-only property 请求不得学习或迁移旧边")

        formations = list(request.formations)
        recognitions = list(request.recognitions)
        if request.legacy_records:
            mapper = self.course.legacy_mapper()
            if mapper is None:
                raise PropertyRelationRuntimeError(
                    "legacy property 记录缺少显式 mapper")
            if not isinstance(mapper, LegacyPropertyMapper):
                raise TypeError("legacy property mapper 协议不完整")
            for record in request.legacy_records:
                mapped = self.owner.map_legacy(record, mapper)
                if mapped is None:
                    raise PropertyRelationRuntimeError(
                        "legacy property mapper 无法补全 typed PROPERTY")
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
            self.owner.claim_from_definition(item.spec.proposition)
            if item.spec.schema != self.owner.protocol.schema:
                raise PropertyRelationRuntimeError(
                    "property formation 未使用当前 typed schema")
            self.owner.semantic_graph.preflight_atomic(
                item.spec.proposition,
                scope=item.scope,
                **item.metadata(),
            )
            claim = self.owner.claim_from_definition(item.spec.proposition)
            if self.owner.intensity_resolver.resolve(claim.intensity) is None:
                raise PropertyRelationRuntimeError(
                    "property formation intensity 缺少 Rational 解释")
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
            self.owner.select_many(request.queries)
            if request.queries else ()
        )
        return PropertyRoundReport(
            scope,
            read_only,
            formation_traces,
            recognition_traces,
            query_results,
        )

    def clone_for_context(self, ctx: TrainContext) -> "PropertyCourseRuntime":
        """用克隆 builder/course 在评测 context 上重建独立 R-03 owner。"""
        cloned_builder = self.builder.clone_for_evaluation()
        cloned_course = self.course.clone_for_evaluation()
        if not isinstance(cloned_builder, PropertyRuntimeBuilder):
            raise TypeError("property builder clone 协议不完整")
        if not isinstance(cloned_course, PropertyCourse):
            raise TypeError("property course clone 协议不完整")
        if cloned_builder.state_key() != self.builder.state_key():
            raise ValueError("property builder clone 改变协议状态")
        if cloned_course.state_key() != self.course.state_key():
            raise ValueError("property course clone 改变课程状态")
        return PropertyCourseRuntime(
            ctx,
            self.owner.clone_for_context(ctx),
            cloned_builder,
            cloned_course,
        )

    def state_key(self) -> tuple:
        """返回 builder、课程、owner 和可选 legacy mapper 完整状态。"""
        mapper = self.course.legacy_mapper()
        mapper_key = ()
        if mapper is not None:
            if not isinstance(mapper, LegacyPropertyMapper):
                raise TypeError("legacy property mapper 协议不完整")
            mapper_key = mapper.state_key()
            _strict_key(
                mapper_key,
                label="LegacyPropertyMapper.state_key",
            )
        return (
            self.builder.state_key(),
            self.course.state_key(),
            mapper_key,
            self.owner.state_key(),
        )

    @staticmethod
    def _validate_combined(
            formations: list[PropertyFormationRequest],
            recognitions: list[RelationClosureRecognitionInput],
            ) -> None:
        """在 legacy 映射后重新拒绝重复 Proposition 和 recognition 路由。"""
        propositions = tuple(
            item.spec.proposition.proposition for item in formations)
        if len(set(propositions)) != len(propositions):
            raise ValueError("property combined forming Proposition 重复")
        routes = tuple(item.route_key() for item in recognitions)
        if len(set(routes)) != len(routes):
            raise ValueError("property combined recognition 路由重复")


def install_property_relation_runtime(
        ctx: TrainContext,
        builder: PropertyRuntimeBuilder,
        course: PropertyCourse,
        ) -> PropertyCourseRuntime:
    """在 TrainContext 上安装显式成对注入且默认关闭的 R-03 runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("install property relation ctx 类型错误")
    if not isinstance(builder, PropertyRuntimeBuilder):
        raise TypeError("property builder 协议不完整")
    if not isinstance(course, PropertyCourse):
        raise TypeError("property course 协议不完整")
    if getattr(ctx, "property_relation_runtime", None) is not None:
        raise ValueError("TrainContext 已安装 property relation runtime")
    _strict_key(builder.state_key(), label="PropertyRuntimeBuilder.state_key")
    _strict_key(course.state_key(), label="PropertyCourse.state_key")
    owner = builder.build(ctx)
    if not isinstance(owner, PropertyRelationRuntime):
        raise TypeError("property builder.build 返回类型错误")
    if owner.semantic_graph.ontology is not ctx.graph_ontology:
        raise ValueError("property owner 未绑定当前 TrainContext 图")
    runtime = PropertyCourseRuntime(ctx, owner, builder, course)
    ctx.property_relation_runtime = runtime
    return runtime


__all__ = [
    "LegacyPropertyMapper",
    "LegacyPropertyRecord",
    "MappedLegacyProperty",
    "PropertyCourse",
    "PropertyCourseRuntime",
    "PropertyFormationRequest",
    "PropertyQuery",
    "PropertyRelationRuntime",
    "PropertyRelationRuntimeError",
    "PropertyRoundReport",
    "PropertyRoundRequest",
    "PropertyRuntimeBuilder",
    "PropertyRuntimeResult",
    "install_property_relation_runtime",
]
