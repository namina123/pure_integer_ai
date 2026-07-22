"""S-03 Proposition template 的显式词法 scope 图协议。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    GraphStatement,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_BINDER,
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    TypedRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class SemanticTemplateScopeError(RuntimeError):
    """template scope 声明缺失、部分、竞争或来源不一致。"""


@dataclass(frozen=True)
class SemanticTemplateScopePredicates:
    """由课程注入的 Proposition-scope 与 scope-Binder 关系。"""

    proposition_scope: TypedRef
    scope_binder: TypedRef

    def refs(self) -> tuple[TypedRef, TypedRef]:
        """按协议槽位返回两个互异 predicate。"""
        return self.proposition_scope, self.scope_binder


@dataclass(frozen=True)
class SemanticTemplateScopeDefinition:
    """一个 Proposition 的显式 ContextScope 和有序无关 Binder 集。"""

    proposition: ObjectIdentity
    scope: ObjectIdentity
    introduced_binders: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.proposition, ObjectIdentity)
                or self.proposition.object_kind != OBJECT_PROPOSITION):
            raise ValueError("template scope proposition 类型错误")
        if (not isinstance(self.scope, ObjectIdentity)
                or self.scope.object_kind != OBJECT_CONTEXT_SCOPE):
            raise ValueError("template scope 必须是一等 ContextScope")
        if not isinstance(self.introduced_binders, tuple):
            raise TypeError("template scope binders 必须是 tuple")
        if any(not isinstance(item, ObjectIdentity)
               or item.object_kind != OBJECT_BINDER
               for item in self.introduced_binders):
            raise ValueError("template scope 只能引入 Binder")
        if len(set(self.introduced_binders)) != len(self.introduced_binders):
            raise ValueError("template scope 不得重复 Binder")
        source = semantic_source(self.proposition)
        if semantic_source(self.scope) != source or any(
                semantic_source(item) != source
                for item in self.introduced_binders):
            raise ValueError("template scope、Binder 与 Proposition 来源不一致")
        object.__setattr__(self, "introduced_binders", tuple(sorted(
            self.introduced_binders,
            key=ObjectIdentity.stable_key,
        )))


@dataclass(frozen=True)
class MaterializedSemanticTemplateScope:
    """从图恢复的 scope 定义、来源元数据和断言集合。"""

    definition: SemanticTemplateScopeDefinition
    scope_identity: ScopeIdentity
    provenance_kind: int
    epistemic_origin: int
    content_version: int
    qualifiers: tuple[int, ...]
    assertion_hashes: tuple[int, ...]


class SemanticTemplateScopeGraph:
    """显式保存空/非空 template scope，不承担运行期变量赋值。"""

    def __init__(
            self,
            ontology: GraphOntology,
            predicates: SemanticTemplateScopePredicates,
            ) -> None:
        if not isinstance(ontology, GraphOntology):
            raise TypeError("template scope ontology 类型错误")
        if not isinstance(predicates, SemanticTemplateScopePredicates):
            raise TypeError("template scope predicates 类型错误")
        self.ontology = ontology
        self.predicates = predicates
        refs = predicates.refs()
        if len(set(refs)) != 2 or any(
                ontology.identity_of(ref).object_kind != OBJECT_CONCEPT
                for ref in refs):
            raise SemanticTemplateScopeError(
                "template scope predicate 必须是互异 Concept")

    def preflight_many(
            self,
            definitions: tuple[SemanticTemplateScopeDefinition, ...],
            *,
            scope: ScopeIdentity,
            provenance_kind: int,
            epistemic_origin: int = 0,
            content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> None:
        """整批核验声明、已有拓扑和来源元数据，任何写入前拒绝冲突。"""
        self._validate_batch(
            definitions, scope, provenance_kind,
            epistemic_origin, content_version, qualifiers)
        expected = (
            scope, provenance_kind, epistemic_origin,
            content_version, qualifiers,
        )
        for definition in definitions:
            proposition = self.ontology.resolve(definition.proposition)
            scope_ref = self.ontology.resolve(definition.scope)
            outgoing = () if proposition is None else self.ontology.statements(
                predicate=self.predicates.proposition_scope,
                subject=proposition,
            )
            incoming = () if scope_ref is None else self.ontology.statements(
                predicate=self.predicates.proposition_scope,
                object_ref=scope_ref,
            )
            binders = () if scope_ref is None else self.ontology.statements(
                predicate=self.predicates.scope_binder,
                subject=scope_ref,
            )
            if not outgoing and not incoming and not binders:
                continue
            if len(outgoing) != 1 or len(incoming) != 1:
                raise SemanticTemplateScopeError(
                    "已有 template scope 是部分拓扑或被多个 Proposition 复用")
            restored = self.read(definition.proposition)
            if restored.definition != definition:
                raise SemanticTemplateScopeError(
                    "已有 template scope 定义与课程声明不一致")
            if self._materialized_metadata(restored) != expected:
                raise SemanticTemplateScopeError(
                    "已有 template scope 来源元数据不一致")

    def materialize_many(
            self,
            definitions: tuple[SemanticTemplateScopeDefinition, ...],
            *,
            scope: ScopeIdentity,
            provenance_kind: int,
            epistemic_origin: int = 0,
            content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> tuple[MaterializedSemanticTemplateScope, ...]:
        """预检整批后写入显式 scope 边；空 Binder 集仍保留主声明边。"""
        self.preflight_many(
            definitions,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        for definition in definitions:
            proposition = self.ontology.materialize(definition.proposition)
            scope_ref = self.ontology.materialize(definition.scope)
            self.ontology.relate(
                self.predicates.proposition_scope,
                proposition,
                scope_ref,
                scope=scope,
                provenance_kind=provenance_kind,
                epistemic_origin=epistemic_origin,
                content_version=content_version,
                qualifiers=qualifiers,
            )
            for binder in definition.introduced_binders:
                self.ontology.relate(
                    self.predicates.scope_binder,
                    scope_ref,
                    self.ontology.materialize(binder),
                    scope=scope,
                    provenance_kind=provenance_kind,
                    epistemic_origin=epistemic_origin,
                    content_version=content_version,
                    qualifiers=qualifiers,
                )
        return tuple(self.read(item.proposition) for item in definitions)

    def read(
            self, proposition: ObjectIdentity,
            ) -> MaterializedSemanticTemplateScope:
        """按完整 Proposition 恢复唯一 scope，并区分空声明和缺失声明。"""
        if (not isinstance(proposition, ObjectIdentity)
                or proposition.object_kind != OBJECT_PROPOSITION):
            raise ValueError("template scope read 必须使用 Proposition")
        proposition_ref = self.ontology.resolve(proposition)
        if proposition_ref is None:
            raise SemanticTemplateScopeError("Proposition 尚未物化")
        links = self.ontology.statements(
            predicate=self.predicates.proposition_scope,
            subject=proposition_ref,
        )
        if len(links) != 1:
            raise SemanticTemplateScopeError(
                f"Proposition template scope 必须恰有一条，实际 {len(links)} 条")
        scope_ref = links[0].object
        scope_identity = self.ontology.identity_of(scope_ref)
        incoming = self.ontology.statements(
            predicate=self.predicates.proposition_scope,
            object_ref=scope_ref,
        )
        if len(incoming) != 1 or incoming[0].subject != proposition_ref:
            raise SemanticTemplateScopeError("template scope 被跨 Proposition 复用")
        binder_links = self.ontology.statements(
            predicate=self.predicates.scope_binder,
            subject=scope_ref,
        )
        binders = tuple(sorted(
            (self.ontology.identity_of(item.object) for item in binder_links),
            key=ObjectIdentity.stable_key,
        ))
        definition = SemanticTemplateScopeDefinition(
            proposition, scope_identity, binders)
        statements = (links[0], *binder_links)
        metadata = self._statement_metadata(statements[0])
        if any(self._statement_metadata(item) != metadata
               for item in statements[1:]):
            raise SemanticTemplateScopeError(
                "template scope 与 Binder 边来源元数据不一致")
        return MaterializedSemanticTemplateScope(
            definition,
            *metadata,
            tuple(sorted(item.assertion_hash for item in statements)),
        )

    @staticmethod
    def _statement_metadata(
            statement: GraphStatement,
            ) -> tuple[ScopeIdentity, int, int, int, tuple[int, ...]]:
        """提取 statement 的完整 scope 和来源元数据。"""
        assertion = statement.assertion
        return (
            assertion.scope,
            assertion.provenance_kind,
            assertion.epistemic_origin,
            assertion.content_version,
            assertion.qualifiers,
        )

    @staticmethod
    def _materialized_metadata(
            value: MaterializedSemanticTemplateScope,
            ) -> tuple[ScopeIdentity, int, int, int, tuple[int, ...]]:
        """提取已恢复声明的完整来源元数据。"""
        return (
            value.scope_identity,
            value.provenance_kind,
            value.epistemic_origin,
            value.content_version,
            value.qualifiers,
        )

    @staticmethod
    def _validate_batch(
            definitions: tuple[SemanticTemplateScopeDefinition, ...],
            scope: ScopeIdentity,
            provenance_kind: int,
            epistemic_origin: int,
            content_version: int,
            qualifiers: tuple[int, ...],
            ) -> None:
        """核验批次完整覆盖键和严格整数来源元数据。"""
        if not isinstance(definitions, tuple) or not definitions or any(
                not isinstance(item, SemanticTemplateScopeDefinition)
                for item in definitions):
            raise TypeError("template scope definitions 必须是非空定义 tuple")
        propositions = tuple(item.proposition for item in definitions)
        scopes = tuple(item.scope for item in definitions)
        if len(set(propositions)) != len(propositions):
            raise SemanticTemplateScopeError("同批次重复 Proposition scope 声明")
        if len(set(scopes)) != len(scopes):
            raise SemanticTemplateScopeError("同批次不得复用 template scope 身份")
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("template scope statement scope 类型错误")
        if any(semantic_source(item.proposition) != scope.source
               for item in definitions):
            raise SemanticTemplateScopeError("template scope 批次来源不一致")
        if not isinstance(qualifiers, tuple):
            raise TypeError("template scope qualifiers 必须是 tuple")
        assert_int(
            provenance_kind, epistemic_origin, content_version,
            *qualifiers, _where="SemanticTemplateScopeGraph",
        )
        if (type(provenance_kind) is not int or provenance_kind <= 0
                or type(epistemic_origin) is not int
                or epistemic_origin < 0
                or type(content_version) is not int
                or content_version < 0
                or any(type(item) is not int for item in qualifiers)):
            raise ValueError("template scope 来源元数据非法")


__all__ = [
    "MaterializedSemanticTemplateScope",
    "SemanticTemplateScopeDefinition",
    "SemanticTemplateScopeError",
    "SemanticTemplateScopeGraph",
    "SemanticTemplateScopePredicates",
]
