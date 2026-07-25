"""R-00 关系候选的 typed 定义、active 投影和只读消费者。

本模块只统一关系学习闭环的对象边界，不解释任何具体 relation、Role、反例或
代数律。原子命题拓扑继续以 ``SemanticGraph`` 为真源，候选状态继续以 H-05
Evidence/H-04/投影事件为真源；这里不建立第二套 relation strength 或真值表。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateGraphProjection,
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    CANDIDATE_AS_OBJECT,
    CANDIDATE_AS_SUBJECT,
    ActiveEvidenceCandidate,
    CandidateBinding,
    EvidenceCandidateDefinition,
    EvidenceCandidateEngine,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.semantic_graph import SemanticGraph
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    ActiveSupportedRelationFact,
    RelationSchema,
    RelationSchemaError,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class RelationClosureError(RuntimeError):
    """关系候选定义、状态真源或 typed 消费结果不一致。"""


def _strict_key(value, *, where: str) -> tuple[int, ...]:
    """校验由课程或评测协议注入的非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


@dataclass(frozen=True)
class RelationClosureField:
    """候选对象与一个 typed 字段值之间的动态图绑定槽。"""

    predicate: ObjectIdentity
    ordinal: int = 0
    candidate_endpoint: int = CANDIDATE_AS_SUBJECT

    def __post_init__(self) -> None:
        if not isinstance(self.predicate, ObjectIdentity):
            raise TypeError("field predicate 必须是 ObjectIdentity")
        if self.predicate.object_kind != OBJECT_CONCEPT:
            raise ValueError("field predicate 必须是一等 Concept")
        assert_int(
            self.ordinal,
            self.candidate_endpoint,
            _where="RelationClosureField",
        )
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("field ordinal 必须为非负严格整数")
        if self.candidate_endpoint not in {
                CANDIDATE_AS_SUBJECT, CANDIDATE_AS_OBJECT}:
            raise ValueError("field candidate_endpoint 非法")

    def binding(self, value: ObjectIdentity) -> CandidateBinding:
        """把调用方给出的一等字段值转换为 H-05 完整 binding。"""
        return CandidateBinding(
            self.predicate,
            value,
            self.ordinal,
            self.candidate_endpoint,
        )

    def slot_key(self) -> tuple[ObjectIdentity, int, int]:
        """返回可检测协议槽冲突的 predicate、方向和序号。"""
        return self.predicate, self.candidate_endpoint, self.ordinal


@dataclass(frozen=True)
class RelationClosureProtocol:
    """关系候选图中 relation 与 schema 字段的注入式协议。"""

    relation: RelationClosureField
    schema: RelationClosureField

    def __post_init__(self) -> None:
        if not isinstance(self.relation, RelationClosureField):
            raise TypeError("relation field 必须是 RelationClosureField")
        if not isinstance(self.schema, RelationClosureField):
            raise TypeError("schema field 必须是 RelationClosureField")
        if self.relation.slot_key() == self.schema.slot_key():
            raise ValueError("relation 与 schema 不得复用同一候选字段槽")


@dataclass(frozen=True)
class RelationClosureCandidateSpec:
    """一个 typed 原子关系候选及其竞争组、形成来源和领域附加绑定。"""

    proposition: AtomicPropositionDefinition
    schema: RelationSchema
    competition_key: tuple[int, ...]
    forming_sources: tuple[SourceRef, ...]
    domain_bindings: tuple[CandidateBinding, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.proposition, AtomicPropositionDefinition):
            raise TypeError("proposition 必须是 AtomicPropositionDefinition")
        if not isinstance(self.schema, RelationSchema):
            raise TypeError("schema 必须是 RelationSchema")
        self.schema.validate_definition(self.proposition)
        _strict_key(
            self.competition_key,
            where="RelationClosureCandidateSpec.competition_key",
        )
        if not isinstance(self.forming_sources, tuple):
            raise TypeError("forming_sources 必须是 SourceRef tuple")
        if any(not isinstance(item, SourceRef)
               for item in self.forming_sources):
            raise TypeError("forming_sources 只能包含 SourceRef")
        if len(set(self.forming_sources)) != len(self.forming_sources):
            raise ValueError("forming_sources 不得重复")
        if not isinstance(self.domain_bindings, tuple):
            raise TypeError("domain_bindings 必须是 CandidateBinding tuple")
        if any(not isinstance(item, CandidateBinding)
               for item in self.domain_bindings):
            raise TypeError("domain_bindings 只能包含 CandidateBinding")
        object.__setattr__(self, "forming_sources", tuple(sorted(
            self.forming_sources, key=SourceRef.stable_key)))
        object.__setattr__(self, "domain_bindings", tuple(sorted(
            self.domain_bindings, key=CandidateBinding.stable_key)))

    def candidate_definition(
            self, protocol: RelationClosureProtocol,
            ) -> EvidenceCandidateDefinition:
        """生成保留 relation/schema 和领域绑定的 H-05 完整候选定义。"""
        if not isinstance(protocol, RelationClosureProtocol):
            raise TypeError("protocol 必须是 RelationClosureProtocol")
        bindings = (
            protocol.relation.binding(self.proposition.predicate),
            protocol.schema.binding(self.schema.schema),
            *self.domain_bindings,
        )
        return EvidenceCandidateDefinition(
            self.proposition.proposition,
            self.competition_key,
            tuple(bindings),
            self.forming_sources,
        )


@dataclass(frozen=True)
class ActiveRelationClosureFact:
    """可回溯到原子命题和 H-05 active 事件的 typed 关系事实。"""

    proposition: AtomicPropositionDefinition
    schema: RelationSchema
    projection: CandidateGraphProjection
    active_candidate: ActiveEvidenceCandidate | None

    def __post_init__(self) -> None:
        if not isinstance(self.proposition, AtomicPropositionDefinition):
            raise TypeError("proposition 必须是 AtomicPropositionDefinition")
        if not isinstance(self.schema, RelationSchema):
            raise TypeError("schema 必须是 RelationSchema")
        if not isinstance(self.projection, CandidateGraphProjection):
            raise TypeError("projection 必须是 CandidateGraphProjection")
        self.schema.validate_definition(self.proposition)
        materialized = self.projection.candidate
        if materialized.definition.candidate != self.proposition.proposition:
            raise RelationClosureError("active 投影未绑定当前 Proposition")
        if self.active_candidate is not None:
            if not isinstance(self.active_candidate, ActiveEvidenceCandidate):
                raise TypeError("active_candidate 类型非法")
            if (self.active_candidate.definition != materialized.definition
                    or self.active_candidate.hypothesis
                    != materialized.hypothesis):
                raise RelationClosureError("H-05 active 状态与图投影定义不一致")

    @property
    def hypothesis(self):
        """返回候选图保存的完整 H-05 Hypothesis。"""
        return self.projection.candidate.hypothesis

    @property
    def evidence_keys(self) -> tuple[tuple[int, ...], ...]:
        """返回最新 active/refresh 事件引用的当前 Evidence 全键。"""
        return self.projection.history[-1].definition.evidence_keys

    @property
    def decision_key(self) -> tuple[int, ...]:
        """返回最新 active/refresh 事件引用的 H-04 decision 全键。"""
        return self.projection.history[-1].definition.decision_key

    @property
    def read_only_recovered(self) -> bool:
        """指出当前事实是否仅依赖图恢复而没有可续写 H-00 owner。"""
        return self.active_candidate is None

    def as_algebra_fact(self) -> ActiveSupportedRelationFact:
        """把在线 H-05 状态投影为 S-01 代数输入，恢复只读状态不得伪造 snapshot。"""
        if self.active_candidate is None:
            raise RelationClosureError(
                "M-03 前只读恢复事实不能伪造 relation algebra snapshot")
        return ActiveSupportedRelationFact(
            self.proposition,
            self.active_candidate.snapshot,
            self.projection.candidate.definition,
        )


class ActiveRelationClosureConsumer:
    """只消费 active typed 投影，并从 SemanticGraph 恢复关系完整定义。"""

    def __init__(
            self, semantic_graph: SemanticGraph,
            candidate_graph: CandidateProjectionGraph,
            protocol: RelationClosureProtocol,
            schemas: tuple[RelationSchema, ...], *,
            engine: EvidenceCandidateEngine | None = None) -> None:
        if not isinstance(semantic_graph, SemanticGraph):
            raise TypeError("semantic_graph 必须是 SemanticGraph")
        if not isinstance(candidate_graph, CandidateProjectionGraph):
            raise TypeError("candidate_graph 必须是 CandidateProjectionGraph")
        if semantic_graph.ontology is not candidate_graph.ontology:
            raise ValueError("语义图和候选图必须绑定同一 GraphOntology")
        if not isinstance(protocol, RelationClosureProtocol):
            raise TypeError("protocol 必须是 RelationClosureProtocol")
        if not isinstance(schemas, tuple) or not schemas:
            raise ValueError("schemas 必须是非空 RelationSchema tuple")
        if any(not isinstance(item, RelationSchema) for item in schemas):
            raise TypeError("schemas 只能包含 RelationSchema")
        by_identity = {item.schema: item for item in schemas}
        if len(by_identity) != len(schemas):
            raise ValueError("schemas 不得重复一等 schema 身份")
        if engine is not None and not isinstance(engine, EvidenceCandidateEngine):
            raise TypeError("engine 必须是 EvidenceCandidateEngine 或 None")
        self.semantic_graph = semantic_graph
        self.candidate_graph = candidate_graph
        self.protocol = protocol
        self.schemas = tuple(sorted(
            schemas, key=lambda item: item.schema.stable_key()))
        self.engine = engine
        self._schemas = by_identity

    def lookup_relation(
            self, relation: ObjectIdentity, *,
            schema: ObjectIdentity | None = None,
            ) -> tuple[ActiveRelationClosureFact, ...]:
        """按完整 relation 查询全部 active typed 事实，可再按 schema 过滤。"""
        if not isinstance(relation, ObjectIdentity):
            raise TypeError("relation 必须是 ObjectIdentity")
        if relation.object_kind != OBJECT_CONCEPT:
            raise ValueError("relation 必须是一等 Concept")
        if schema is not None and not isinstance(schema, ObjectIdentity):
            raise TypeError("schema 必须是 ObjectIdentity 或 None")
        projections = self.candidate_graph.active_for_binding(
            self.protocol.relation.binding(relation))
        facts = tuple(
            self._fact(projection)
            for projection in projections
            if (schema is None
                or self._field_value(
                    projection, self.protocol.schema) == schema)
        )
        return tuple(sorted(
            facts,
            key=lambda item: item.proposition.proposition.stable_key(),
        ))

    def lookup_proposition(
            self, proposition: ObjectIdentity,
            ) -> tuple[ActiveRelationClosureFact, ...]:
        """按完整 Proposition 读取 active 事实；无定义或无 lifecycle 返回空。"""
        if not isinstance(proposition, ObjectIdentity):
            raise TypeError("proposition 必须是 ObjectIdentity")
        if proposition.object_kind != OBJECT_PROPOSITION:
            raise ValueError("proposition 必须是一等 Proposition")
        candidate = self.candidate_graph.ontology.resolve(proposition)
        if candidate is None or not self.candidate_graph.history(candidate):
            return ()
        projection = self.candidate_graph.project(candidate)
        if projection.state != self.candidate_graph.protocol.active_state:
            return ()
        return (self._fact(projection),)

    def lookup_role_filler(
            self,
            relation: ObjectIdentity,
            schemas: tuple[ObjectIdentity, ...],
            role: ObjectIdentity,
            filler: ObjectIdentity,
            ) -> tuple[ActiveRelationClosureFact, ...]:
        """按 S-00 predicate/Role/filler 局部反查当前 active typed 事实。"""
        if not isinstance(relation, ObjectIdentity):
            raise TypeError("relation 必须是 ObjectIdentity")
        if relation.object_kind != OBJECT_CONCEPT:
            raise ValueError("relation 必须是一等 Concept")
        if not isinstance(schemas, tuple) or not schemas:
            raise ValueError("schemas 必须是非空 ObjectIdentity tuple")
        if any(not isinstance(item, ObjectIdentity) for item in schemas):
            raise TypeError("schemas 只能包含 ObjectIdentity")
        if len(set(schemas)) != len(schemas):
            raise ValueError("schemas 不得重复")
        if any(item not in self._schemas for item in schemas):
            raise ValueError("schemas 含 consumer 未注册 schema")
        if not isinstance(role, ObjectIdentity):
            raise TypeError("role 必须是 ObjectIdentity")
        if not isinstance(filler, ObjectIdentity):
            raise TypeError("filler 必须是 ObjectIdentity")
        allowed = frozenset(schemas)
        facts: list[ActiveRelationClosureFact] = []
        for restored in self.semantic_graph.lookup_atomic_by_binding(
                relation, role, filler):
            active = self.lookup_proposition(restored.definition.proposition)
            if not active:
                continue
            fact = active[0]
            if fact.schema.schema not in allowed:
                continue
            facts.append(fact)
        return tuple(sorted(
            facts,
            key=lambda item: item.proposition.proposition.stable_key(),
        ))

    def require_proposition(
            self, proposition: ObjectIdentity) -> ActiveRelationClosureFact:
        """要求指定 Proposition 恰有一个 active typed 投影，否则 fail closed。"""
        facts = self.lookup_proposition(proposition)
        if len(facts) != 1:
            raise LookupError("当前 Proposition 没有唯一 active typed 关系事实")
        return facts[0]

    def clone_for_graphs(
            self, semantic_graph: SemanticGraph,
            candidate_graph: CandidateProjectionGraph, *,
            engine: EvidenceCandidateEngine | None,
            ) -> "ActiveRelationClosureConsumer":
        """把同一字段/schema 协议绑定到隔离图和可选克隆 H-00 owner。"""
        return ActiveRelationClosureConsumer(
            semantic_graph,
            candidate_graph,
            self.protocol,
            self.schemas,
            engine=engine,
        )

    def _fact(
            self, projection: CandidateGraphProjection,
            ) -> ActiveRelationClosureFact:
        """交叉核验候选字段、SemanticGraph、schema 和可选 H-05 当前状态。"""
        if projection.state != self.candidate_graph.protocol.active_state:
            raise RelationClosureError("关系消费者只能接收 active 投影")
        candidate = projection.candidate
        if candidate.definition.candidate.object_kind != OBJECT_PROPOSITION:
            raise RelationClosureError("关系候选必须使用 Proposition 身份")
        relation = self._field_value(projection, self.protocol.relation)
        schema_identity = self._field_value(projection, self.protocol.schema)
        schema = self._schemas.get(schema_identity)
        if schema is None:
            raise RelationClosureError("active 关系候选引用了未注册 schema")
        restored = self.semantic_graph.read_atomic(candidate.candidate)
        definition = restored.definition
        if definition.predicate != relation:
            raise RelationClosureError("候选 relation 字段与原子命题 predicate 不一致")
        try:
            schema.validate_definition(definition)
        except RelationSchemaError as exc:
            raise RelationClosureError("active 关系事实不满足 typed schema") from exc

        active = None
        if self.engine is not None:
            active = self.engine.active(candidate.hypothesis)
            if active is None:
                raise RelationClosureError("图为 active 但 H-05 owner 当前不可采用")
            latest = projection.history[-1].definition
            history = self.engine.ledger.evidence_history(candidate.hypothesis)
            active_ids = frozenset((
                *active.snapshot.support_evidence_ids,
                *active.snapshot.refute_evidence_ids,
                *active.snapshot.unknown_evidence_ids,
            ))
            evidence_keys = tuple(
                item.stable_key() for item in history
                if item.evidence_id in active_ids
            )
            if (latest.evidence_keys != evidence_keys
                    or latest.decision_key != active.decision.stable_key()):
                raise RelationClosureError("图投影引用的 Evidence/H-04 决策已陈旧")
        return ActiveRelationClosureFact(
            definition,
            schema,
            projection,
            active,
        )

    @staticmethod
    def _field_value(
            projection: CandidateGraphProjection,
            field: RelationClosureField) -> ObjectIdentity:
        """按完整字段槽读取唯一值，缺失或竞争值均拒绝。"""
        matches = tuple(
            binding.value
            for binding in projection.candidate.definition.bindings
            if (binding.predicate == field.predicate
                and binding.ordinal == field.ordinal
                and binding.candidate_endpoint == field.candidate_endpoint)
        )
        if len(matches) != 1:
            raise RelationClosureError("关系候选字段必须恰有一个 typed 值")
        return matches[0]


__all__ = [
    "ActiveRelationClosureConsumer",
    "ActiveRelationClosureFact",
    "RelationClosureCandidateSpec",
    "RelationClosureError",
    "RelationClosureField",
    "RelationClosureProtocol",
]
