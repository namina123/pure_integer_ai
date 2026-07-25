"""单个对称 relation channel 的 R-00 适配、查询、Use 和 legacy 映射。"""
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
from pure_integer_ai.cognition.shared.symmetric_relation import (
    SymmetricPair,
    SymmetricPairEvidence,
    SymmetricPairKnowledge,
    SymmetricPairPattern,
    SymmetricPairSelection,
    SymmetricRelationBudget,
    SymmetricRelationBudgetExceeded,
    SymmetricRelationEngine,
    SymmetricRelationProtocol,
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


class SymmetricRelationRuntimeError(RuntimeError):
    """单个对称 channel 的 owner、查询或旧边映射不完整。"""


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
        raise SymmetricRelationRuntimeError(
            "symmetric endpoint Role 必须恰有一个 filler")
    return matches[0]


@dataclass(frozen=True)
class SymmetricPairQuery:
    """一次单 channel 精确或发现查询及可选来源化采用路由。"""

    pattern: SymmetricPairPattern
    use_key: tuple[int, ...] | None = None
    context: RelationUseContext | None = None

    def __post_init__(self) -> None:
        """只允许完整 pair 查询携带成对 use_key/context。"""
        if not isinstance(self.pattern, SymmetricPairPattern):
            raise TypeError("symmetric query pattern 类型错误")
        if self.use_key is not None:
            _strict_key(self.use_key, label="SymmetricPairQuery.use_key")
        if self.context is not None and not isinstance(
                self.context, RelationUseContext):
            raise TypeError("symmetric query context 类型错误")
        if (self.use_key is None) != (self.context is None):
            raise ValueError("symmetric query use_key 与 context 必须成对提供")
        if self.use_key is not None and self.pattern.counterpart is None:
            raise ValueError("发现查询不得写 Use，采用必须完整指定 pair")


@dataclass(frozen=True)
class SymmetricPairRuntimeResult:
    """一次对称 pair 查询结果和实际提交的 R-00 Use。"""

    selection: SymmetricPairSelection
    uses: tuple[RelationClosureUse, ...]

    def __post_init__(self) -> None:
        """核验选择结果与采用记录类型。"""
        if not isinstance(self.selection, SymmetricPairSelection):
            raise TypeError("symmetric runtime selection 类型错误")
        if not isinstance(self.uses, tuple) or any(
                not isinstance(item, RelationClosureUse) for item in self.uses):
            raise TypeError("symmetric runtime uses 类型错误")
        if not self.selection.pure_supported() and self.uses:
            raise ValueError("symmetric 非纯支持查询不得写 Use")


@dataclass(frozen=True)
class LegacySymmetricPairRecord:
    """旧 EDGE、词典或 cue 摄入的一条来源化对称 pair 记录。"""

    relation_key: tuple[int, ...]
    left_key: tuple[int, ...]
    right_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """保留旧 relation/endpoint 键和来源，不解释具体 channel 语义。"""
        for label, value in (
                ("relation", self.relation_key),
                ("left", self.left_key),
                ("right", self.right_key),
                ("trace", self.trace)):
            _strict_key(value, label=f"legacy symmetric {label}")
        if not isinstance(self.source, SourceRef):
            raise TypeError("legacy symmetric source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("legacy symmetric scope 类型错误")
        if self.scope.source != self.source:
            raise ValueError("legacy symmetric scope 必须绑定记录来源")


@dataclass(frozen=True)
class SymmetricPairFormationRequest:
    """课程提交的一次 S-00 pair 定义和 R-00 forming 请求。"""

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
            raise TypeError("symmetric formation spec 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("symmetric formation scope 类型错误")
        if self.scope.source != self.spec.proposition.source:
            raise ValueError("symmetric formation scope 必须绑定 Proposition 来源")
        if not isinstance(self.qualifiers, tuple):
            raise TypeError("symmetric formation qualifiers 必须是 tuple")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            self.timestamp_base,
            *self.qualifiers,
            _where="SymmetricPairFormationRequest",
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
            raise ValueError("symmetric formation 元数据非法")

    def metadata(self) -> dict:
        """返回 SemanticGraph 定义写入所需显式来源元数据。"""
        return {
            "provenance_kind": self.provenance_kind,
            "epistemic_origin": self.epistemic_origin,
            "content_version": self.content_version,
            "qualifiers": self.qualifiers,
        }


@dataclass(frozen=True)
class MappedLegacySymmetricPair:
    """旧记录经 mapper 补全后的 typed forming 和 recognition。"""

    formation: SymmetricPairFormationRequest
    recognition: RelationClosureRecognitionInput

    def __post_init__(self) -> None:
        """要求 forming 与 recognition 指向同一完整 Proposition。"""
        if not isinstance(self.formation, SymmetricPairFormationRequest):
            raise TypeError("mapped symmetric formation 类型错误")
        if not isinstance(self.recognition, RelationClosureRecognitionInput):
            raise TypeError("mapped symmetric recognition 类型错误")
        if (self.formation.spec.proposition.proposition
                != self.recognition.proposition):
            raise ValueError("mapped symmetric forming 与 recognition 不一致")


@runtime_checkable
class LegacySymmetricPairMapper(Protocol):
    """把旧边、词典或 cue 记录显式映射为当前 channel typed 候选。"""

    def map(
            self, record: LegacySymmetricPairRecord,
            ) -> MappedLegacySymmetricPair | None:
        """无法补全 relation/schema/Role、来源或 Evidence 时返回 None。"""
        ...

    def clone_for_evaluation(self) -> "LegacySymmetricPairMapper":
        """返回不共享可变状态的评测 mapper。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回 mapper 版本和来源策略的完整整数键。"""
        ...


class SymmetricRelationChannelRuntime:
    """把单个 R-00 owner 适配为有界对称 pair 查询和 context Use。"""

    def __init__(
            self,
            relation_runtime: RelationClosureRuntime,
            protocol: SymmetricRelationProtocol,
            budget: SymmetricRelationBudget,
            ) -> None:
        """绑定唯一 owner，并核验 schema 和候选 hypothesis kind 可读取。"""
        if not isinstance(relation_runtime, RelationClosureRuntime):
            raise TypeError("symmetric relation closure runtime 类型错误")
        if not isinstance(protocol, SymmetricRelationProtocol):
            raise TypeError("symmetric relation protocol 类型错误")
        if not isinstance(budget, SymmetricRelationBudget):
            raise TypeError("symmetric relation budget 类型错误")
        registered = {item.schema for item in relation_runtime.consumer.schemas}
        if protocol.schema.schema not in registered:
            raise SymmetricRelationRuntimeError(
                "R-00 consumer 缺少 symmetric relation schema")
        _strict_key(
            relation_runtime.candidate_runtime.engine.protocol.hypothesis_kind_key,
            label="symmetric channel hypothesis kind",
        )
        self.relation_runtime = relation_runtime
        self.protocol = protocol
        self.budget = budget

    @property
    def semantic_graph(self) -> SemanticGraph:
        """返回底层 R-00 使用的 S-00 SemanticGraph。"""
        return self.relation_runtime.semantic_graph

    @property
    def hypothesis_kind(self) -> tuple[int, ...]:
        """返回该 channel 独占的 EvidenceCandidate hypothesis kind。"""
        return (
            self.relation_runtime.candidate_runtime.engine.protocol
            .hypothesis_kind_key
        )

    def pair_from_definition(
            self, definition: AtomicPropositionDefinition,
            ) -> tuple[SymmetricPair, ObjectIdentity, ObjectIdentity]:
        """按有名 Role 恢复 canonical pair 和原始 left/right 方向。"""
        self.protocol.schema.validate_definition(definition)
        left = _single_filler(definition, self.protocol.left_role)
        right = _single_filler(definition, self.protocol.right_role)
        return self.protocol.pair(left, right), left, right

    def knowledge(self) -> SymmetricPairKnowledge:
        """从 R-00 当前 lifecycle 恢复该 channel 的多方向四态 Evidence。"""
        evidence = []
        for formation in self.relation_runtime.formation_traces():
            spec = formation.spec
            if spec.proposition.predicate != self.protocol.relation:
                continue
            if spec.schema != self.protocol.schema:
                raise SymmetricRelationRuntimeError(
                    "symmetric relation 使用了协议外或陈旧 schema")
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
                raise SymmetricRelationRuntimeError(
                    "symmetric forming Proposition 未进入权威语义图")
            materialized = self.semantic_graph.read_atomic(proposition_ref)
            if materialized.definition != spec.proposition:
                raise SymmetricRelationRuntimeError(
                    "symmetric forming 与语义图定义不一致")
            pair, left, right = self.pair_from_definition(spec.proposition)
            evidence.append(SymmetricPairEvidence(
                pair,
                left,
                right,
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
                raise SymmetricRelationBudgetExceeded(
                    "symmetric relation 直接事实预算耗尽")
        return SymmetricPairKnowledge(tuple(evidence))

    def engine(
            self,
            knowledge: SymmetricPairKnowledge | None = None,
            ) -> SymmetricRelationEngine:
        """在当前或调用方预读快照上构造一次有界 pair engine。"""
        return SymmetricRelationEngine(
            self.protocol,
            self.budget,
            self.knowledge() if knowledge is None else knowledge,
        )

    def select_many(
            self,
            queries: tuple[SymmetricPairQuery, ...],
            *,
            knowledge: SymmetricPairKnowledge | None = None,
            ) -> tuple[SymmetricPairSelection, ...]:
        """共享一次不可变快照执行精确或发现查询，不提交 Use。"""
        if not isinstance(queries, tuple) or not queries:
            raise ValueError("symmetric queries 必须是非空 tuple")
        if any(not isinstance(item, SymmetricPairQuery) for item in queries):
            raise TypeError("symmetric query 类型错误")
        engine = self.engine(knowledge)
        return tuple(engine.select(item.pattern) for item in queries)

    def consume_selections(
            self,
            queries: tuple[SymmetricPairQuery, ...],
            selections: tuple[SymmetricPairSelection, ...],
            ) -> tuple[tuple[RelationClosureUse, ...], ...]:
        """仅为完整 context 精确查询的唯一 pure-supported pair 写直接前提 Use。"""
        if (not isinstance(queries, tuple)
                or not isinstance(selections, tuple)
                or len(queries) != len(selections)):
            raise ValueError("symmetric queries/selections 数量不一致")
        requests = []
        owners = []
        sizes = []
        for index, (query, selection) in enumerate(
                zip(queries, selections, strict=True)):
            if not isinstance(query, SymmetricPairQuery):
                raise TypeError("symmetric consume query 类型错误")
            if not isinstance(selection, SymmetricPairSelection):
                raise TypeError("symmetric consume selection 类型错误")
            if query.use_key is None:
                sizes.append(0)
                continue
            if query.context is None:
                raise SymmetricRelationRuntimeError(
                    "symmetric 采用缺少 RelationUseContext")
            pure = selection.pure_supported()
            if len(selection.evaluations) != 1 or len(pure) != 1:
                sizes.append(0)
                continue
            premises = pure[0].active_premises()
            if not premises:
                raise SymmetricRelationRuntimeError(
                    "pure-supported symmetric pair 缺少 active 直接前提")
            sizes.append(len(premises))
            for ordinal, premise in enumerate(premises):
                requests.append((
                    premise.proposition,
                    (*query.use_key, ordinal),
                    query.context,
                ))
                owners.append(index)
        if not requests:
            return tuple(() for _query in queries)
        committed = self.relation_runtime.consume_many(tuple(requests))
        result: list[list[RelationClosureUse]] = [[] for _query in queries]
        for owner, use in zip(owners, committed, strict=True):
            result[owner].append(use)
        if any(len(result[index]) != size for index, size in enumerate(sizes)):
            raise SymmetricRelationRuntimeError(
                "symmetric Use 提交数量与支持前提不一致")
        return tuple(tuple(items) for items in result)

    def query_many(
            self,
            queries: tuple[SymmetricPairQuery, ...],
            ) -> tuple[SymmetricPairRuntimeResult, ...]:
        """共享一次快照查询多个 pair，并按完整 context 原子写本 channel Use。"""
        selections = self.select_many(queries)
        uses = self.consume_selections(queries, selections)
        return tuple(
            SymmetricPairRuntimeResult(selection, group)
            for selection, group in zip(selections, uses, strict=True)
        )

    def query(self, query: SymmetricPairQuery) -> SymmetricPairRuntimeResult:
        """执行一个对称 pair 查询并按需写 context Use。"""
        return self.query_many((query,))[0]

    def map_legacy(
            self,
            record: LegacySymmetricPairRecord,
            mapper: LegacySymmetricPairMapper,
            ) -> MappedLegacySymmetricPair | None:
        """调用显式 mapper，并核验结果属于当前 relation/schema/source/scope。"""
        if not isinstance(record, LegacySymmetricPairRecord):
            raise TypeError("legacy symmetric record 类型错误")
        if not isinstance(mapper, LegacySymmetricPairMapper):
            raise TypeError("legacy symmetric mapper 协议不完整")
        mapped = mapper.map(record)
        if mapped is None:
            return None
        if not isinstance(mapped, MappedLegacySymmetricPair):
            raise TypeError("legacy symmetric mapper 返回类型错误")
        spec = mapped.formation.spec
        if (spec.proposition.predicate != self.protocol.relation
                or spec.schema != self.protocol.schema):
            raise SymmetricRelationRuntimeError(
                "旧记录未映射为当前 symmetric typed schema")
        if (semantic_source(spec.proposition.proposition) != record.source
                or mapped.formation.scope != record.scope):
            raise SymmetricRelationRuntimeError(
                "旧 symmetric mapper 不得替换记录来源或 scope")
        self.pair_from_definition(spec.proposition)
        return mapped

    def state_key(self) -> tuple:
        """返回 R-00 owner、对称关系协议、kind 和预算完整状态。"""
        return (
            self.relation_runtime.state_key(),
            self.protocol.stable_key(),
            self.hypothesis_kind,
            self.budget.stable_key(),
        )

    def clone_for_context(
            self, ctx: TrainContext,
            ) -> "SymmetricRelationChannelRuntime":
        """复制 H-00 owner，并在评测克隆图上重建单 channel facade。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("symmetric clone ctx 类型错误")
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
        cloned = SymmetricRelationChannelRuntime(
            self.relation_runtime.clone_for_evaluation(
                semantic_graph,
                candidate_graph,
            ),
            self.protocol,
            self.budget,
        )
        if cloned.hypothesis_kind != self.hypothesis_kind:
            raise ValueError("symmetric clone 改变 hypothesis kind")
        return cloned


@dataclass(frozen=True)
class SymmetricChannelBatch:
    """单 channel 在一个来源 scope 中互斥的学习写入或查询批次。"""

    legacy_records: tuple[LegacySymmetricPairRecord, ...] = ()
    formations: tuple[SymmetricPairFormationRequest, ...] = ()
    recognitions: tuple[RelationClosureRecognitionInput, ...] = ()
    queries: tuple[SymmetricPairQuery, ...] = ()

    def __post_init__(self) -> None:
        """拒绝混批、重复 Proposition 和重复 recognition 路由。"""
        groups = (
            (self.legacy_records, LegacySymmetricPairRecord, "legacy_records"),
            (self.formations, SymmetricPairFormationRequest, "formations"),
            (self.recognitions, RelationClosureRecognitionInput, "recognitions"),
            (self.queries, SymmetricPairQuery, "queries"),
        )
        for values, expected, label in groups:
            if not isinstance(values, tuple):
                raise TypeError(f"symmetric batch {label} 必须是 tuple")
            if any(not isinstance(item, expected) for item in values):
                raise TypeError(f"symmetric batch {label} 元素类型错误")
        writes = bool(self.legacy_records or self.formations or self.recognitions)
        if writes and self.queries:
            raise ValueError("symmetric 学习写入与查询采用必须分批")
        propositions = tuple(
            item.spec.proposition.proposition for item in self.formations)
        if len(set(propositions)) != len(propositions):
            raise ValueError("symmetric batch 不得重复 forming Proposition")
        routes = tuple(item.route_key() for item in self.recognitions)
        if len(set(routes)) != len(routes):
            raise ValueError("symmetric batch 不得重复 recognition 路由")


@dataclass(frozen=True)
class SymmetricChannelReport:
    """单个 channel 一轮的 forming、recognition 和查询结果。"""

    formations: tuple[RelationClosureFormationTrace, ...]
    recognitions: tuple[RelationClosureRecognitionTrace, ...]
    query_results: tuple[SymmetricPairRuntimeResult, ...]

    def __post_init__(self) -> None:
        """核验单 channel 报告三个结果集合的元素类型。"""
        checks = (
            (self.formations, RelationClosureFormationTrace),
            (self.recognitions, RelationClosureRecognitionTrace),
            (self.query_results, SymmetricPairRuntimeResult),
        )
        for values, expected in checks:
            if not isinstance(values, tuple) or any(
                    not isinstance(item, expected) for item in values):
                raise TypeError("symmetric channel report 字段类型错误")


__all__ = [
    "LegacySymmetricPairMapper",
    "LegacySymmetricPairRecord",
    "MappedLegacySymmetricPair",
    "SymmetricChannelBatch",
    "SymmetricChannelReport",
    "SymmetricPairFormationRequest",
    "SymmetricPairQuery",
    "SymmetricPairRuntimeResult",
    "SymmetricRelationChannelRuntime",
    "SymmetricRelationRuntimeError",
]
