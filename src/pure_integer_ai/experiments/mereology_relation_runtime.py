"""R-04 部分整体关系族的 R-00 适配、课程编排和 V-06 重建协议。"""
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
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.mereology_relation import (
    MereologyBudget,
    MereologyBudgetExceeded,
    MereologyEvidence,
    MereologyKnowledge,
    MereologyPattern,
    MereologyProtocol,
    MereologyRelationEngine,
    MereologySelection,
    MereologyStatement,
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


class MereologyRuntimeError(RuntimeError):
    """R-04 owner、课程或旧边映射不满足完整协议。"""


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
        raise MereologyRuntimeError("mereology Role 必须恰有一个 filler")
    return matches[0]


@dataclass(frozen=True)
class MereologyQuery:
    """一次部分整体精确或发现查询及可选来源化采用路由。"""

    pattern: MereologyPattern
    use_key: tuple[int, ...] | None = None
    context: RelationUseContext | None = None

    def __post_init__(self) -> None:
        """核验查询模式和 R-00 Use 路由配对关系。"""
        if not isinstance(self.pattern, MereologyPattern):
            raise TypeError("mereology query pattern 类型错误")
        if self.use_key is not None:
            _strict_key(self.use_key, label="MereologyQuery.use_key")
        if self.context is not None and not isinstance(
                self.context, RelationUseContext):
            raise TypeError("mereology query context 类型错误")
        if self.context is not None and self.use_key is None:
            raise ValueError("mereology query context 必须配套 use_key")


@dataclass(frozen=True)
class MereologyRuntimeResult:
    """一次部分整体查询结果和实际提交的 R-00 Use。"""

    selection: MereologySelection
    uses: tuple[RelationClosureUse, ...]

    def __post_init__(self) -> None:
        """核验查询结果与采用记录类型。"""
        if not isinstance(self.selection, MereologySelection):
            raise TypeError("mereology runtime selection 类型错误")
        if not isinstance(self.uses, tuple) or any(
                not isinstance(item, RelationClosureUse) for item in self.uses):
            raise TypeError("mereology runtime uses 类型错误")
        if not self.selection.pure_supported() and self.uses:
            raise ValueError("mereology 非纯支持查询不得写 Use")


@dataclass(frozen=True)
class LegacyMereologyRecord:
    """旧 EDGE_MEREOLOGY 或词典摄入的一条来源化记录。"""

    relation_key: tuple[int, ...]
    part_key: tuple[int, ...]
    whole_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """保留旧键和来源，不在宿主中解释其关系分类。"""
        for label, value in (
                ("relation", self.relation_key),
                ("part", self.part_key),
                ("whole", self.whole_key),
                ("trace", self.trace)):
            _strict_key(value, label=f"legacy mereology {label}")
        if not isinstance(self.source, SourceRef):
            raise TypeError("legacy mereology source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("legacy mereology scope 类型错误")
        if self.scope.source != self.source:
            raise ValueError("legacy mereology scope 必须绑定记录来源")


@dataclass(frozen=True)
class MereologyFormationRequest:
    """课程提交的一次 S-00 部分整体定义和 R-00 forming 请求。"""

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
            raise TypeError("mereology formation spec 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("mereology formation scope 类型错误")
        if self.scope.source != self.spec.proposition.source:
            raise ValueError("mereology formation scope 必须绑定 Proposition 来源")
        if not isinstance(self.qualifiers, tuple):
            raise TypeError("mereology formation qualifiers 必须是 tuple")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            self.timestamp_base,
            *self.qualifiers,
            _where="MereologyFormationRequest",
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
            raise ValueError("mereology formation 元数据非法")

    def metadata(self) -> dict:
        """返回 SemanticGraph 定义写入所需显式来源元数据。"""
        return {
            "provenance_kind": self.provenance_kind,
            "epistemic_origin": self.epistemic_origin,
            "content_version": self.content_version,
            "qualifiers": self.qualifiers,
        }


@dataclass(frozen=True)
class MappedLegacyMereology:
    """旧记录经 mapper 补全后的 typed forming 和 recognition。"""

    formation: MereologyFormationRequest
    recognition: RelationClosureRecognitionInput

    def __post_init__(self) -> None:
        """要求 forming 与 recognition 指向同一完整 Proposition。"""
        if not isinstance(self.formation, MereologyFormationRequest):
            raise TypeError("mapped mereology formation 类型错误")
        if not isinstance(self.recognition, RelationClosureRecognitionInput):
            raise TypeError("mapped mereology recognition 类型错误")
        if (self.formation.spec.proposition.proposition
                != self.recognition.proposition):
            raise ValueError("mapped mereology forming 与 recognition 不一致")


@runtime_checkable
class LegacyMereologyMapper(Protocol):
    """把旧边、词典或 cue 记录显式映射为当前 typed 候选。"""

    def map(
            self, record: LegacyMereologyRecord,
            ) -> MappedLegacyMereology | None:
        """无法选择 relation/schema/Role、来源或 Evidence 时返回 None。"""
        ...

    def clone_for_evaluation(self) -> "LegacyMereologyMapper":
        """返回不共享可变状态的评测 mapper。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回 mapper 版本和来源策略的完整整数键。"""
        ...


class MereologyRelationRuntime:
    """把 R-00 当前关系快照适配为有界部分整体闭包和 Use。"""

    def __init__(
            self,
            relation_runtime: RelationClosureRuntime,
            protocol: MereologyProtocol,
            budget: MereologyBudget,
            ) -> None:
        """绑定同一 R-00 owner，并核验关系族全部 schema 已注册。"""
        if not isinstance(relation_runtime, RelationClosureRuntime):
            raise TypeError("mereology relation closure runtime 类型错误")
        if not isinstance(protocol, MereologyProtocol):
            raise TypeError("mereology protocol 类型错误")
        if not isinstance(budget, MereologyBudget):
            raise TypeError("mereology budget 类型错误")
        registered = {item.schema for item in relation_runtime.consumer.schemas}
        required = {item.schema.schema for item in protocol.relations}
        if not required.issubset(registered):
            raise MereologyRuntimeError("R-00 consumer 缺少 R-04 relation schema")
        self.relation_runtime = relation_runtime
        self.protocol = protocol
        self.budget = budget

    @property
    def semantic_graph(self) -> SemanticGraph:
        """返回底层 R-00 使用的 S-00 SemanticGraph。"""
        return self.relation_runtime.semantic_graph

    def statement_from_definition(
            self, definition: AtomicPropositionDefinition,
            ) -> MereologyStatement:
        """按 relation 的有名 Role 恢复 canonical part/whole，不看绑定顺序。"""
        relation = self.protocol.require_relation(definition.predicate)
        relation.schema.validate_definition(definition)
        return MereologyStatement(
            relation.relation,
            _single_filler(definition, relation.part_role),
            _single_filler(definition, relation.whole_role),
        )

    def knowledge(self) -> MereologyKnowledge:
        """从 R-00 当前 lifecycle 恢复全部 relation variant 的四态 Evidence。"""
        evidence = []
        relations = {item.relation for item in self.protocol.relations}
        for formation in self.relation_runtime.formation_traces():
            spec = formation.spec
            if spec.proposition.predicate not in relations:
                continue
            relation = self.protocol.require_relation(spec.proposition.predicate)
            if spec.schema != relation.schema:
                raise MereologyRuntimeError(
                    "mereology relation 使用了协议外或陈旧 schema")
            snapshot = self.relation_runtime.snapshot_for_proposition(
                spec.proposition.proposition)
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
                raise MereologyRuntimeError(
                    "mereology forming Proposition 未进入权威语义图")
            materialized = self.semantic_graph.read_atomic(proposition_ref)
            if materialized.definition != spec.proposition:
                raise MereologyRuntimeError(
                    "mereology forming 与语义图定义不一致")
            evidence.append(MereologyEvidence(
                self.statement_from_definition(spec.proposition),
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
                raise MereologyBudgetExceeded(
                    "mereology 直接事实预算耗尽")
        return MereologyKnowledge(tuple(evidence))

    def engine(self) -> MereologyRelationEngine:
        """在当前不可变快照上构造一次有界 fixpoint engine。"""
        return MereologyRelationEngine(
            self.protocol,
            self.budget,
            self.knowledge(),
        )

    def query_many(
            self,
            queries: tuple[MereologyQuery, ...],
            ) -> tuple[MereologyRuntimeResult, ...]:
        """共享一次闭包查询多个模式，并原子采用全部纯支持结果前提。"""
        if not isinstance(queries, tuple) or not queries:
            raise ValueError("mereology queries 必须是非空 tuple")
        if any(not isinstance(item, MereologyQuery) for item in queries):
            raise TypeError("mereology query 类型错误")
        engine = self.engine()
        selections = tuple(engine.select(item.pattern) for item in queries)
        groups = []
        for query, selection in zip(queries, selections, strict=True):
            options = []
            for evaluation in selection.pure_supported():
                premises = evaluation.active_premises()
                if not premises:
                    raise MereologyRuntimeError(
                        "纯支持 mereology 结果缺少 active 直接前提")
                options.append(premises)
            groups.append((query.use_key, query.context, tuple(options)))
        uses = self._consume_groups(tuple(groups))
        return tuple(
            MereologyRuntimeResult(selection, group)
            for selection, group in zip(selections, uses, strict=True)
        )

    def query(self, query: MereologyQuery) -> MereologyRuntimeResult:
        """执行一个部分整体精确或发现查询并按需写 Use。"""
        return self.query_many((query,))[0]

    def map_legacy(
            self,
            record: LegacyMereologyRecord,
            mapper: LegacyMereologyMapper,
            ) -> MappedLegacyMereology | None:
        """调用显式 mapper，并核验结果属于当前 relation variant。"""
        if not isinstance(record, LegacyMereologyRecord):
            raise TypeError("legacy mereology record 类型错误")
        if not isinstance(mapper, LegacyMereologyMapper):
            raise TypeError("legacy mereology mapper 协议不完整")
        mapped = mapper.map(record)
        if mapped is None:
            return None
        if not isinstance(mapped, MappedLegacyMereology):
            raise TypeError("legacy mereology mapper 返回类型错误")
        spec = mapped.formation.spec
        relation = self.protocol.relation_protocol(spec.proposition.predicate)
        if relation is None or spec.schema != relation.schema:
            raise MereologyRuntimeError(
                "旧记录未映射为当前 typed mereology schema")
        if (semantic_source(spec.proposition.proposition) != record.source
                or mapped.formation.scope != record.scope):
            raise MereologyRuntimeError(
                "旧 mereology mapper 不得替换记录来源或 scope")
        self.statement_from_definition(spec.proposition)
        return mapped

    def state_key(self) -> tuple:
        """返回 R-00 owner、部分整体协议和预算完整状态。"""
        return (
            self.relation_runtime.state_key(),
            self.protocol.stable_key(),
            self.budget.stable_key(),
        )

    def clone_for_context(self, ctx: TrainContext) -> "MereologyRelationRuntime":
        """复制 H-00 owner，并在评测克隆图上重建 R-04 facade。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("mereology clone ctx 类型错误")
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
        return MereologyRelationRuntime(
            self.relation_runtime.clone_for_evaluation(
                semantic_graph,
                candidate_graph,
            ),
            self.protocol,
            self.budget,
        )

    def _consume_groups(self, groups: tuple[tuple, ...]) -> tuple[tuple, ...]:
        """把各查询纯支持选项的 active 前提合并为一次 R-00 Use 提交。"""
        requests = []
        owners = []
        sizes = []
        for group_index, (use_key, context, options) in enumerate(groups):
            if use_key is None:
                sizes.append(0)
                continue
            _strict_key(use_key, label="mereology consume use_key")
            size = 0
            for option_index, premises in enumerate(options):
                for premise_index, premise in enumerate(premises):
                    requests.append((
                        premise.proposition,
                        (*use_key, option_index, premise_index),
                        context,
                    ))
                    owners.append(group_index)
                    size += 1
            sizes.append(size)
        if not requests:
            return tuple(() for _group in groups)
        committed = self.relation_runtime.consume_many(tuple(requests))
        result: list[list[RelationClosureUse]] = [[] for _group in groups]
        for owner, use in zip(owners, committed, strict=True):
            result[owner].append(use)
        if any(len(result[index]) != size for index, size in enumerate(sizes)):
            raise MereologyRuntimeError(
                "mereology Use 提交数量与支持前提不一致")
        return tuple(tuple(items) for items in result)


@dataclass(frozen=True)
class MereologyRoundRequest:
    """一个来源 scope 中互斥的部分整体学习轮或查询轮。"""

    scope: ScopeIdentity
    legacy_records: tuple[LegacyMereologyRecord, ...] = ()
    formations: tuple[MereologyFormationRequest, ...] = ()
    recognitions: tuple[RelationClosureRecognitionInput, ...] = ()
    queries: tuple[MereologyQuery, ...] = ()

    def __post_init__(self) -> None:
        """拒绝混轮、错误 scope、重复 Proposition 和 recognition 路由。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("mereology round scope 类型错误")
        groups = (
            (self.legacy_records, LegacyMereologyRecord, "legacy_records"),
            (self.formations, MereologyFormationRequest, "formations"),
            (self.recognitions, RelationClosureRecognitionInput, "recognitions"),
            (self.queries, MereologyQuery, "queries"),
        )
        for values, expected, label in groups:
            if not isinstance(values, tuple):
                raise TypeError(f"mereology round {label} 必须是 tuple")
            if any(not isinstance(item, expected) for item in values):
                raise TypeError(f"mereology round {label} 元素类型错误")
        writes = bool(self.legacy_records or self.formations or self.recognitions)
        if writes and self.queries:
            raise ValueError("mereology 学习写入轮与查询采用轮必须分开")
        if any(item.scope != self.scope for item in self.legacy_records):
            raise ValueError("legacy mereology record 必须绑定当前 scope")
        if any(item.scope != self.scope for item in self.formations):
            raise ValueError("mereology formation 必须绑定当前 scope")
        if any(item.scope != self.scope for item in self.recognitions):
            raise ValueError("mereology recognition 必须绑定当前 scope")
        contexts = tuple(
            item.context for item in self.queries if item.context is not None)
        if any(item.scope != self.scope for item in contexts):
            raise ValueError("mereology query context 必须绑定当前 scope")
        propositions = tuple(
            item.spec.proposition.proposition for item in self.formations)
        if len(set(propositions)) != len(propositions):
            raise ValueError("mereology round 不得重复 forming Proposition")
        routes = tuple(item.route_key() for item in self.recognitions)
        if len(set(routes)) != len(routes):
            raise ValueError("mereology round 不得重复 recognition 路由")


@dataclass(frozen=True)
class MereologyRoundReport:
    """一次 R-04 课程轮的 forming、recognition 和查询结果。"""

    scope: ScopeIdentity
    read_only: bool
    formations: tuple[RelationClosureFormationTrace, ...]
    recognitions: tuple[RelationClosureRecognitionTrace, ...]
    query_results: tuple[MereologyRuntimeResult, ...]

    def __post_init__(self) -> None:
        """核验报告类型，read-only 轮不得包含学习写入。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("mereology report scope 类型错误")
        if type(self.read_only) is not bool:
            raise TypeError("mereology report read_only 必须是严格 bool")
        checks = (
            (self.formations, RelationClosureFormationTrace),
            (self.recognitions, RelationClosureRecognitionTrace),
            (self.query_results, MereologyRuntimeResult),
        )
        for values, expected in checks:
            if not isinstance(values, tuple) or any(
                    not isinstance(item, expected) for item in values):
                raise TypeError("mereology report 字段类型错误")
        if self.read_only and (self.formations or self.recognitions):
            raise ValueError("read-only mereology report 不得含学习写入")


@runtime_checkable
class MereologyRuntimeBuilder(Protocol):
    """由项目课程注入完整 R-00 owner 和 R-04 纯语义组件。"""

    def build(self, ctx: TrainContext) -> MereologyRelationRuntime:
        """在指定 TrainContext 图上构造部分整体 owner。"""
        ...

    def clone_for_evaluation(self) -> "MereologyRuntimeBuilder":
        """返回清除宿主可变引用的评测 builder。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回协议、schema、规则和预算版本完整键。"""
        ...


@runtime_checkable
class MereologyCourse(Protocol):
    """把来源 scope 映射为 typed 部分整体学习轮或查询轮。"""

    def request(
            self, scope: ScopeIdentity, *, read_only: bool,
            ) -> MereologyRoundRequest:
        """返回当前来源的 typed R-04 请求。"""
        ...

    def legacy_mapper(self) -> LegacyMereologyMapper | None:
        """需要迁移旧记录时返回显式 mapper，否则返回 None。"""
        ...

    def clone_for_evaluation(self) -> "MereologyCourse":
        """返回不共享可变课程状态的评测副本。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回课程、mapper 和来源策略的完整整数键。"""
        ...


class MereologyCourseRuntime:
    """让 formal round 只提交 scope，由课程完成 R-04 学习或查询。"""

    def __init__(
            self,
            ctx: TrainContext,
            owner: MereologyRelationRuntime,
            builder: MereologyRuntimeBuilder,
            course: MereologyCourse,
            ) -> None:
        """绑定当前 context、唯一 owner、可克隆 builder 和课程。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("mereology course ctx 类型错误")
        if not isinstance(owner, MereologyRelationRuntime):
            raise TypeError("mereology course owner 类型错误")
        if not isinstance(builder, MereologyRuntimeBuilder):
            raise TypeError("mereology builder 协议不完整")
        if not isinstance(course, MereologyCourse):
            raise TypeError("mereology course 协议不完整")
        _strict_key(builder.state_key(), label="MereologyRuntimeBuilder.state_key")
        _strict_key(course.state_key(), label="MereologyCourse.state_key")
        self.ctx = ctx
        self.owner = owner
        self.builder = builder
        self.course = course

    def process(
            self, scope: ScopeIdentity, *, read_only: bool,
            ) -> MereologyRoundReport:
        """全量预检后执行一个纯学习写入轮或纯查询采用轮。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("mereology process scope 类型错误")
        if type(read_only) is not bool:
            raise TypeError("mereology process read_only 必须是严格 bool")
        request = self.course.request(scope, read_only=read_only)
        if not isinstance(request, MereologyRoundRequest):
            raise TypeError("mereology course.request 返回类型错误")
        if request.scope != scope:
            raise ValueError("mereology course.request 替换了 round scope")
        if read_only and (
                request.legacy_records
                or request.formations
                or request.recognitions):
            raise ValueError("read-only mereology 请求不得学习或迁移旧边")

        formations = list(request.formations)
        recognitions = list(request.recognitions)
        if request.legacy_records:
            mapper = self.course.legacy_mapper()
            if mapper is None:
                raise MereologyRuntimeError("legacy mereology 记录缺少显式 mapper")
            if not isinstance(mapper, LegacyMereologyMapper):
                raise TypeError("legacy mereology mapper 协议不完整")
            for record in request.legacy_records:
                mapped = self.owner.map_legacy(record, mapper)
                if mapped is None:
                    raise MereologyRuntimeError(
                        "legacy mereology mapper 无法补全 typed relation")
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
            timestamps = tuple(range(start, start + len(recognitions) * 3))
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
            statement = self.owner.statement_from_definition(
                item.spec.proposition)
            relation = self.owner.protocol.require_relation(statement.relation)
            if item.spec.schema != relation.schema:
                raise MereologyRuntimeError(
                    "mereology formation 未使用当前 typed schema")
            self.owner.semantic_graph.preflight_atomic(
                item.spec.proposition,
                scope=item.scope,
                **item.metadata(),
            )
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
            self.owner.query_many(request.queries)
            if request.queries else ()
        )
        return MereologyRoundReport(
            scope,
            read_only,
            formation_traces,
            recognition_traces,
            query_results,
        )

    def clone_for_context(self, ctx: TrainContext) -> "MereologyCourseRuntime":
        """用克隆 builder/course 在评测 context 上重建独立 R-04 owner。"""
        cloned_builder = self.builder.clone_for_evaluation()
        cloned_course = self.course.clone_for_evaluation()
        if not isinstance(cloned_builder, MereologyRuntimeBuilder):
            raise TypeError("mereology builder clone 协议不完整")
        if not isinstance(cloned_course, MereologyCourse):
            raise TypeError("mereology course clone 协议不完整")
        if cloned_builder.state_key() != self.builder.state_key():
            raise ValueError("mereology builder clone 改变协议状态")
        if cloned_course.state_key() != self.course.state_key():
            raise ValueError("mereology course clone 改变课程状态")
        return MereologyCourseRuntime(
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
            if not isinstance(mapper, LegacyMereologyMapper):
                raise TypeError("legacy mereology mapper 协议不完整")
            mapper_key = mapper.state_key()
            _strict_key(mapper_key, label="LegacyMereologyMapper.state_key")
        return (
            self.builder.state_key(),
            self.course.state_key(),
            mapper_key,
            self.owner.state_key(),
        )

    @staticmethod
    def _validate_combined(
            formations: list[MereologyFormationRequest],
            recognitions: list[RelationClosureRecognitionInput],
            ) -> None:
        """在 legacy 映射后重新拒绝重复 Proposition 和 recognition 路由。"""
        propositions = tuple(
            item.spec.proposition.proposition for item in formations)
        if len(set(propositions)) != len(propositions):
            raise ValueError("mereology combined forming Proposition 重复")
        routes = tuple(item.route_key() for item in recognitions)
        if len(set(routes)) != len(routes):
            raise ValueError("mereology combined recognition 路由重复")


def install_mereology_relation_runtime(
        ctx: TrainContext,
        builder: MereologyRuntimeBuilder,
        course: MereologyCourse,
        ) -> MereologyCourseRuntime:
    """在 TrainContext 上安装显式成对注入且默认关闭的 R-04 runtime。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("install mereology relation ctx 类型错误")
    if not isinstance(builder, MereologyRuntimeBuilder):
        raise TypeError("mereology builder 协议不完整")
    if not isinstance(course, MereologyCourse):
        raise TypeError("mereology course 协议不完整")
    if getattr(ctx, "mereology_relation_runtime", None) is not None:
        raise ValueError("TrainContext 已安装 mereology relation runtime")
    _strict_key(builder.state_key(), label="MereologyRuntimeBuilder.state_key")
    _strict_key(course.state_key(), label="MereologyCourse.state_key")
    owner = builder.build(ctx)
    if not isinstance(owner, MereologyRelationRuntime):
        raise TypeError("mereology builder.build 返回类型错误")
    if owner.semantic_graph.ontology is not ctx.graph_ontology:
        raise ValueError("mereology owner 未绑定当前 TrainContext 图")
    runtime = MereologyCourseRuntime(ctx, owner, builder, course)
    ctx.mereology_relation_runtime = runtime
    return runtime


__all__ = [
    "LegacyMereologyMapper",
    "LegacyMereologyRecord",
    "MappedLegacyMereology",
    "MereologyCourse",
    "MereologyCourseRuntime",
    "MereologyFormationRequest",
    "MereologyQuery",
    "MereologyRelationRuntime",
    "MereologyRoundReport",
    "MereologyRoundRequest",
    "MereologyRuntimeBuilder",
    "MereologyRuntimeError",
    "MereologyRuntimeResult",
    "install_mereology_relation_runtime",
]
