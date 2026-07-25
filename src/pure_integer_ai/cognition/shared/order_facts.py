"""分型顺序事实的通用图写入、查询和替代入口。

具体是来源位置序、篇章叙述序、事件时间、程序依赖还是生成呈现序，由调用方注入的
一等 predicate 身份区分。本模块不维护顺序类型枚举，也不把 ``order_index``、strength
或共享概念 PRECEDES 当作权威事实。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    GraphStatement,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    ObjectIdentity,
    TypedRef,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    LogicalTimestamp,
    ScopeIdentity,
)
from pure_integer_ai.cognition.shared.scoped_persistence import (
    ScopedIdentityStore,
)


@dataclass(frozen=True)
class OrderFact:
    """一个已由图对象、scope 和 assertion registry 完整核验的顺序事实。"""

    statement: GraphStatement

    @property
    def assertion_hash(self) -> int:
        """返回顺序事实的稳定断言索引。"""
        return self.statement.assertion_hash


class OrderFactIndex:
    """以动态 predicate 管理互不混型的 scoped 顺序事实。"""

    def __init__(self, ontology: GraphOntology,
                 scoped_identities: ScopedIdentityStore) -> None:
        if not isinstance(ontology, GraphOntology):
            raise TypeError("ontology 必须是 GraphOntology")
        if not isinstance(scoped_identities, ScopedIdentityStore):
            raise TypeError("scoped_identities 必须是 ScopedIdentityStore")
        self.ontology = ontology
        self.scoped_identities = scoped_identities
        self._predicate_cache: dict[ObjectIdentity, TypedRef] = {}

    def record(
            self, relation: ObjectIdentity,
            subject: TypedRef,
            object_ref: TypedRef, *,
            scope: ScopeIdentity,
            provenance_kind: int,
            epistemic_origin: int = 0,
            content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> OrderFact:
        """追加分型顺序事实；同一完整 AssertionIdentity 重放时保持幂等。"""
        predicate = self._predicate(relation)
        self.ontology.identity_of(subject)
        self.ontology.identity_of(object_ref)
        if subject == object_ref:
            raise ValueError("顺序事实不得形成自环")
        statement = self.ontology.relate(
            predicate,
            subject,
            object_ref,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        return OrderFact(statement)

    def facts(
            self, relation: ObjectIdentity, *,
            scope: ScopeIdentity | None = None,
            active_only: bool = True,
            ) -> tuple[OrderFact, ...]:
        """读取一种 predicate 下的顺序事实，并可按精确 scope 和替代状态过滤。"""
        predicate = self.ontology.resolve(self._relation(relation))
        if predicate is None:
            return ()
        facts = [
            OrderFact(statement)
            for statement in self.ontology.statements(predicate=predicate)
            if scope is None or statement.assertion.scope == scope
        ]
        if active_only:
            facts = [
                fact for fact in facts
                if not self.scoped_identities.assertion_is_superseded(
                    fact.assertion_hash)
            ]
        return tuple(sorted(facts, key=self._sort_key))

    def supersede(
            self, old: OrderFact, new: OrderFact,
            timestamp: LogicalTimestamp) -> int:
        """用 append-only 事件替代旧顺序事实，不删除旧 statement。"""
        self._validate_fact(old)
        self._validate_fact(new)
        if old.statement.predicate != new.statement.predicate:
            raise ValueError("不同顺序 predicate 的事实不得互相替代")
        return self.scoped_identities.supersede(
            old.statement.assertion,
            new.statement.assertion,
            timestamp,
        )

    def count(
            self, relation: ObjectIdentity, *,
            scope: ScopeIdentity | None = None,
            active_only: bool = True,
            ) -> int:
        """返回一种 predicate 在指定 scope 内的顺序事实数。"""
        return len(self.facts(
            relation, scope=scope, active_only=active_only))

    def clone_for_context(
            self, ontology: GraphOntology,
            scoped_identities: ScopedIdentityStore,
            ) -> "OrderFactIndex":
        """在评测 clone 的图和身份 registry 上重建独立 facade。"""
        return OrderFactIndex(ontology, scoped_identities)

    def _predicate(self, relation: ObjectIdentity) -> TypedRef:
        """幂等物化调用方注入的一等关系 predicate。"""
        relation = self._relation(relation)
        predicate = self._predicate_cache.get(relation)
        if predicate is None:
            predicate = self.ontology.materialize(relation)
            self._predicate_cache[relation] = predicate
        return predicate

    @staticmethod
    def _relation(relation: ObjectIdentity) -> ObjectIdentity:
        """校验顺序关系由一等通用概念承载，而不是宿主枚举。"""
        if not isinstance(relation, ObjectIdentity):
            raise TypeError("relation 必须是 ObjectIdentity")
        if relation.object_kind != OBJECT_CONCEPT:
            raise ValueError("顺序 predicate 必须是一等通用概念")
        return relation

    def _validate_fact(self, fact: OrderFact) -> None:
        """核验待替代事实确实属于当前图和 assertion registry。"""
        if not isinstance(fact, OrderFact):
            raise TypeError("supersede 需要 OrderFact")
        statement = fact.statement
        self.ontology.identity_of(statement.predicate)
        self.ontology.identity_of(statement.subject)
        self.ontology.identity_of(statement.object)
        if self.scoped_identities.load_assertion(
                statement.assertion_hash) != statement.assertion:
            raise ValueError("顺序事实与 assertion registry 不一致")
        matches = tuple(
            candidate
            for candidate in self.ontology.statements(
                predicate=statement.predicate,
                subject=statement.subject,
                object_ref=statement.object,
            )
            if candidate.assertion_hash == statement.assertion_hash
        )
        if matches != (statement,):
            raise ValueError("顺序事实必须在当前图中唯一存在")

    @staticmethod
    def _sort_key(fact: OrderFact) -> tuple:
        """按 scope、限定项和分型端点确定性排列，不赋予跨 scope 全局序。"""
        assertion = fact.statement.assertion
        return (
            assertion.scope.stable_key(),
            assertion.qualifiers,
            assertion.subject.stable_key(),
            assertion.object.stable_key(),
            fact.assertion_hash,
        )


__all__ = ["OrderFact", "OrderFactIndex"]
