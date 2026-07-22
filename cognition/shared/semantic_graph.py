"""S-00 原子命题图拓扑的严格写入、恢复和完整性核验。

所有关系 predicate 都由调用方提供的一等 Concept 引用。本模块固定的只是原子命题
协议槽位，不固定任何自然语言角色、逻辑关系或 Unicode 表示；n 元参与关系通过一等
RoleBinding 节点保存，不能退化为命题到 filler 的裸边。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    GraphStatement,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_ROLE_BINDING,
    ObjectIdentity,
    TypedRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    role_binding_ordinal,
    semantic_source,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class SemanticTopologyError(RuntimeError):
    """语义图缺边、多值、身份不一致或协议竞争时的完整性错误。"""


@dataclass(frozen=True)
class AtomicPropositionPredicates:
    """原子命题协议使用的六个开放 predicate；具体概念身份由外部注入。"""

    proposition_predicate: TypedRef
    proposition_anchor: TypedRef
    proposition_context: TypedRef
    proposition_binding: TypedRef
    binding_role: TypedRef
    binding_filler: TypedRef

    def refs(self) -> tuple[TypedRef, ...]:
        """按协议槽位顺序返回 predicate，供统一分型和唯一性核验。"""
        return (
            self.proposition_predicate,
            self.proposition_anchor,
            self.proposition_context,
            self.proposition_binding,
            self.binding_role,
            self.binding_filler,
        )


@dataclass(frozen=True)
class MaterializedRoleBinding:
    """从完整图拓扑核验得到的 RoleBinding 及其分型端点。"""

    identity: ObjectIdentity
    ref: TypedRef
    role: TypedRef
    filler: TypedRef
    ordinal: int
    assertion_hashes: tuple[int, int, int]


@dataclass(frozen=True)
class MaterializedAtomicProposition:
    """原子命题定义、图引用、来源 scope 和定义 statement 元数据的只读视图。"""

    definition: AtomicPropositionDefinition
    proposition: TypedRef
    predicate: TypedRef
    source_anchor: TypedRef
    context: TypedRef
    bindings: tuple[MaterializedRoleBinding, ...]
    scope: ScopeIdentity
    provenance_kind: int
    epistemic_origin: int
    content_version: int
    qualifiers: tuple[int, ...]
    assertion_hashes: tuple[int, ...]


class SemanticGraph:
    """以 GraphOntology 为唯一真源物化和读取 S-00 语义对象拓扑。"""

    def __init__(self, ontology: GraphOntology,
                 predicates: AtomicPropositionPredicates) -> None:
        if not isinstance(ontology, GraphOntology):
            raise TypeError("ontology 必须是 GraphOntology")
        if not isinstance(predicates, AtomicPropositionPredicates):
            raise TypeError("predicates 必须是 AtomicPropositionPredicates")
        self._ontology = ontology
        self._predicates = predicates
        self._validate_predicates()

    @property
    def ontology(self) -> GraphOntology:
        """返回当前语义 facade 绑定的权威图，只开放已有图协议。"""
        return self._ontology

    @property
    def predicates(self) -> AtomicPropositionPredicates:
        """返回调用方注入的原子命题协议 predicate 集。"""
        return self._predicates

    def preflight_atomic(
            self, definition: AtomicPropositionDefinition, *,
            scope: ScopeIdentity, provenance_kind: int,
            epistemic_origin: int = 0, content_version: int = 0,
            qualifiers: tuple[int, ...] = ()
            ) -> MaterializedAtomicProposition | None:
        """只核验一次原子定义可安全新写或精确重放，不物化对象和 statement。"""
        if not isinstance(definition, AtomicPropositionDefinition):
            raise TypeError("definition 必须是 AtomicPropositionDefinition")
        self._validate_write_metadata(
            definition, scope=scope, provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version, qualifiers=qualifiers)
        existing = self._preflight_existing(definition)
        if existing is not None:
            self._require_replay_metadata(
                existing, scope=scope, provenance_kind=provenance_kind,
                epistemic_origin=epistemic_origin,
                content_version=content_version, qualifiers=qualifiers)
            return existing
        self._preflight_bindings(
            definition,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        return None

    def define_atomic(
            self, definition: AtomicPropositionDefinition, *,
            scope: ScopeIdentity, provenance_kind: int,
            epistemic_origin: int = 0, content_version: int = 0,
            qualifiers: tuple[int, ...] = ()
            ) -> MaterializedAtomicProposition:
        """幂等物化原子命题；已有部分或竞争拓扑时先失败，不追加修补边。"""
        existing = self.preflight_atomic(
            definition,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        if existing is not None:
            return existing

        proposition = self._ontology.materialize(definition.proposition)
        predicate = self._ontology.materialize(definition.predicate)
        source_anchor = self._ontology.materialize(definition.source_anchor)
        context = self._ontology.materialize(definition.context)

        statements: list[GraphStatement] = []
        statements.append(self._relate(
            self._predicates.proposition_predicate,
            proposition, predicate,
            scope=scope, provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version, qualifiers=qualifiers))
        statements.append(self._relate(
            self._predicates.proposition_anchor,
            proposition, source_anchor,
            scope=scope, provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version, qualifiers=qualifiers))
        statements.append(self._relate(
            self._predicates.proposition_context,
            proposition, context,
            scope=scope, provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version, qualifiers=qualifiers))

        for binding in definition.canonical_bindings():
            binding_ref = self._ontology.materialize(
                binding.identity_for(definition.proposition))
            role_ref = self._ontology.materialize(binding.role)
            filler_ref = self._ontology.materialize(binding.filler)
            statements.append(self._relate(
                self._predicates.proposition_binding,
                proposition, binding_ref,
                scope=scope, provenance_kind=provenance_kind,
                epistemic_origin=epistemic_origin,
                content_version=content_version, qualifiers=qualifiers))
            statements.append(self._relate(
                self._predicates.binding_role,
                binding_ref, role_ref,
                scope=scope, provenance_kind=provenance_kind,
                epistemic_origin=epistemic_origin,
                content_version=content_version, qualifiers=qualifiers))
            statements.append(self._relate(
                self._predicates.binding_filler,
                binding_ref, filler_ref,
                scope=scope, provenance_kind=provenance_kind,
                epistemic_origin=epistemic_origin,
                content_version=content_version, qualifiers=qualifiers))

        restored = self.read_atomic(proposition)
        expected_hashes = tuple(sorted(
            statement.assertion_hash for statement in statements))
        if restored.assertion_hashes != expected_hashes:
            raise SemanticTopologyError("写后恢复的 statement 集合与本次定义不一致")
        return restored

    def read_atomic(self, proposition: TypedRef) -> MaterializedAtomicProposition:
        """从权威图恢复原子命题，并严格核验基槽、RoleBinding 和统一来源元数据。"""
        proposition_identity = self._ontology.identity_of(proposition)
        if proposition_identity.object_kind != OBJECT_PROPOSITION:
            raise ValueError("read_atomic 需要 Proposition 引用")

        predicate_statement = self._single(
            self._predicates.proposition_predicate, proposition,
            label="predicate")
        anchor_statement = self._single(
            self._predicates.proposition_anchor, proposition,
            label="source anchor")
        context_statement = self._single(
            self._predicates.proposition_context, proposition,
            label="semantic context")
        binding_statements = self._ontology.statements(
            predicate=self._predicates.proposition_binding,
            subject=proposition,
        )

        statements: list[GraphStatement] = [
            predicate_statement, anchor_statement, context_statement]
        restored_bindings: list[MaterializedRoleBinding] = []
        domain_bindings: list[AtomicRoleBinding] = []
        seen_slots: set[tuple[ObjectIdentity, int]] = set()
        for link in binding_statements:
            binding_ref = link.object
            binding_identity = self._ontology.identity_of(binding_ref)
            if binding_identity.object_kind != OBJECT_ROLE_BINDING:
                raise SemanticTopologyError(
                    "proposition binding 端点不是 RoleBinding")
            role_statement = self._single(
                self._predicates.binding_role, binding_ref,
                label="binding role")
            filler_statement = self._single(
                self._predicates.binding_filler, binding_ref,
                label="binding filler")
            role_identity = self._ontology.identity_of(role_statement.object)
            filler_identity = self._ontology.identity_of(
                filler_statement.object)
            if role_identity.object_kind != OBJECT_ROLE:
                raise SemanticTopologyError("RoleBinding 的 role 端点不是 Role")
            ordinal = role_binding_ordinal(binding_identity)
            domain_binding = AtomicRoleBinding(
                role_identity, filler_identity, ordinal)
            expected_identity = domain_binding.identity_for(
                proposition_identity)
            if expected_identity != binding_identity:
                raise SemanticTopologyError(
                    "RoleBinding 完整身份与 proposition/role/filler 不一致")
            slot = role_identity, ordinal
            if slot in seen_slots:
                raise SemanticTopologyError(
                    "同一 Role 和 ordinal 出现竞争 filler")
            seen_slots.add(slot)
            statements.extend((link, role_statement, filler_statement))
            domain_bindings.append(domain_binding)
            restored_bindings.append(MaterializedRoleBinding(
                binding_identity,
                binding_ref,
                role_statement.object,
                filler_statement.object,
                ordinal,
                (
                    link.assertion_hash,
                    role_statement.assertion_hash,
                    filler_statement.assertion_hash,
                ),
            ))

        metadata = self._uniform_metadata(statements)
        source = semantic_source(proposition_identity)
        if metadata[0].source != source:
            raise SemanticTopologyError(
                "原子命题 statement scope 未绑定命题 SourceRef")
        definition = AtomicPropositionDefinition(
            proposition_identity,
            self._ontology.identity_of(predicate_statement.object),
            self._ontology.identity_of(anchor_statement.object),
            self._ontology.identity_of(context_statement.object),
            tuple(domain_bindings),
        )
        restored_bindings.sort(key=lambda item: item.identity.stable_key())
        assertion_hashes = tuple(sorted(
            statement.assertion_hash for statement in statements))
        return MaterializedAtomicProposition(
            definition,
            proposition,
            predicate_statement.object,
            anchor_statement.object,
            context_statement.object,
            tuple(restored_bindings),
            metadata[0],
            metadata[1],
            metadata[2],
            metadata[3],
            metadata[4],
            assertion_hashes,
        )

    def lookup_atomic_by_anchor(
            self, anchor: ObjectIdentity,
            ) -> tuple[MaterializedAtomicProposition, ...]:
        """按完整 Occurrence/Span anchor 反查全部原子命题，不按存储顺序私选。"""
        if not isinstance(anchor, ObjectIdentity):
            raise TypeError("lookup anchor 必须是 ObjectIdentity")
        anchor_ref = self._ontology.resolve(anchor)
        if anchor_ref is None:
            return ()
        propositions: dict[ObjectIdentity, MaterializedAtomicProposition] = {}
        for statement in self._ontology.statements(
                predicate=self._predicates.proposition_anchor,
                object_ref=anchor_ref):
            restored = self.read_atomic(statement.subject)
            if restored.definition.source_anchor != anchor:
                raise SemanticTopologyError(
                    "anchor 反向索引命中的 Proposition 未保留请求 anchor")
            propositions[restored.definition.proposition] = restored
        return tuple(
            propositions[key]
            for key in sorted(propositions, key=ObjectIdentity.stable_key)
        )

    def lookup_atomic_by_filler(
            self, filler: ObjectIdentity,
            ) -> tuple[MaterializedAtomicProposition, ...]:
        """按任意完整 Role filler 反查全部原子命题，Role 与 predicate 仍由结果携带。"""
        if not isinstance(filler, ObjectIdentity):
            raise TypeError("lookup filler 必须是 ObjectIdentity")
        filler_ref = self._ontology.resolve(filler)
        if filler_ref is None:
            return ()
        propositions: dict[ObjectIdentity, MaterializedAtomicProposition] = {}
        for filler_link in self._ontology.statements(
                predicate=self._predicates.binding_filler,
                object_ref=filler_ref):
            binding_ref = filler_link.subject
            proposition_links = self._ontology.statements(
                predicate=self._predicates.proposition_binding,
                object_ref=binding_ref,
            )
            if len(proposition_links) != 1:
                raise SemanticTopologyError(
                    "RoleBinding 必须恰属于一个 Proposition")
            restored = self.read_atomic(proposition_links[0].subject)
            if not any(
                    binding.filler == filler_ref
                    for binding in restored.bindings):
                raise SemanticTopologyError(
                    "filler 反向索引命中的 Proposition 未保留请求 filler")
            propositions[restored.definition.proposition] = restored
        return tuple(
            propositions[key]
            for key in sorted(propositions, key=ObjectIdentity.stable_key)
        )

    def lookup_atomic_by_binding(
            self,
            predicate: ObjectIdentity,
            role: ObjectIdentity,
            filler: ObjectIdentity,
            ) -> tuple[MaterializedAtomicProposition, ...]:
        """按完整 predicate/Role/filler 反查并核验原子命题，不扫描全图。"""
        if not isinstance(predicate, ObjectIdentity):
            raise TypeError("lookup predicate 必须是 ObjectIdentity")
        if predicate.object_kind != OBJECT_CONCEPT:
            raise ValueError("lookup predicate 必须是一等 Concept")
        if not isinstance(role, ObjectIdentity):
            raise TypeError("lookup role 必须是 ObjectIdentity")
        if role.object_kind != OBJECT_ROLE:
            raise ValueError("lookup role 必须是一等 Role")
        if not isinstance(filler, ObjectIdentity):
            raise TypeError("lookup filler 必须是 ObjectIdentity")
        predicate_ref = self._ontology.resolve(predicate)
        role_ref = self._ontology.resolve(role)
        filler_ref = self._ontology.resolve(filler)
        if predicate_ref is None or role_ref is None or filler_ref is None:
            return ()

        propositions: dict[ObjectIdentity, MaterializedAtomicProposition] = {}
        filler_links = self._ontology.statements(
            predicate=self._predicates.binding_filler,
            object_ref=filler_ref,
        )
        for filler_link in filler_links:
            binding_ref = filler_link.subject
            binding_identity = self._ontology.identity_of(binding_ref)
            if binding_identity.object_kind != OBJECT_ROLE_BINDING:
                raise SemanticTopologyError(
                    "binding_filler 的 subject 不是 RoleBinding")
            role_links = self._ontology.statements(
                predicate=self._predicates.binding_role,
                subject=binding_ref,
            )
            if len(role_links) != 1:
                raise SemanticTopologyError(
                    "RoleBinding 必须恰有一个 role statement")
            if role_links[0].object != role_ref:
                continue
            proposition_links = self._ontology.statements(
                predicate=self._predicates.proposition_binding,
                object_ref=binding_ref,
            )
            if len(proposition_links) != 1:
                raise SemanticTopologyError(
                    "RoleBinding 必须恰属于一个 Proposition")
            proposition_ref = proposition_links[0].subject
            restored = self.read_atomic(proposition_ref)
            if restored.predicate != predicate_ref:
                continue
            if not any(
                    item.role == role_ref and item.filler == filler_ref
                    for item in restored.bindings):
                raise SemanticTopologyError(
                    "反向索引命中的 Proposition 未保留请求 binding")
            identity = restored.definition.proposition
            existing = propositions.get(identity)
            if existing is not None and existing != restored:
                raise SemanticTopologyError(
                    "同一 Proposition 反向恢复结果不一致")
            propositions[identity] = restored
        return tuple(
            propositions[key]
            for key in sorted(propositions, key=ObjectIdentity.stable_key)
        )

    def _validate_predicates(self) -> None:
        """核验六个协议 predicate 均为图内 Concept 且互不复用。"""
        refs = self._predicates.refs()
        if any(not isinstance(ref, TypedRef) for ref in refs):
            raise TypeError("原子命题 predicate 必须全部是 TypedRef")
        if len({ref.stable_key() for ref in refs}) != len(refs):
            raise ValueError("原子命题协议的六个 predicate 必须互不相同")
        for ref in refs:
            identity = self._ontology.identity_of(ref)
            if identity.object_kind != OBJECT_CONCEPT:
                raise ValueError("原子命题协议 predicate 必须是 Concept")

    @staticmethod
    def _validate_write_metadata(
            definition: AtomicPropositionDefinition, *,
            scope: ScopeIdentity, provenance_kind: int,
            epistemic_origin: int, content_version: int,
            qualifiers: tuple[int, ...]) -> None:
        """校验定义 statement 的来源 scope 和开放整数元数据。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        if scope.source != definition.source:
            raise ValueError("定义 statement scope 必须绑定 Proposition SourceRef")
        assert_int(
            provenance_kind, epistemic_origin, content_version,
            *qualifiers, _where="SemanticGraph.define_atomic")
        if type(provenance_kind) is not int or provenance_kind <= 0:
            raise ValueError("provenance_kind 必须为严格正整数")
        if type(epistemic_origin) is not int or epistemic_origin < 0:
            raise ValueError("epistemic_origin 必须为非负严格整数")
        if type(content_version) is not int or content_version < 0:
            raise ValueError("content_version 必须为非负严格整数")
        if not isinstance(qualifiers, tuple):
            raise TypeError("qualifiers 必须是整数 tuple")
        if any(type(value) is not int for value in qualifiers):
            raise ValueError("qualifiers 必须使用严格整数")

    def _preflight_existing(
            self, definition: AtomicPropositionDefinition
            ) -> MaterializedAtomicProposition | None:
        """在写边前识别完整重放、空壳对象或部分/竞争拓扑。"""
        proposition = self._ontology.resolve(definition.proposition)
        if proposition is None:
            return None
        topology = tuple(
            statement
            for predicate in self._predicates.refs()[:4]
            for statement in self._ontology.statements(
                predicate=predicate, subject=proposition)
        )
        if not topology:
            return None
        restored = self.read_atomic(proposition)
        if restored.definition != definition:
            raise SemanticTopologyError(
                "已有 Proposition 身份绑定了不同原子命题拓扑")
        return restored

    def _preflight_bindings(
            self, definition: AtomicPropositionDefinition, *,
            scope: ScopeIdentity, provenance_kind: int,
            epistemic_origin: int, content_version: int,
            qualifiers: tuple[int, ...]) -> None:
        """在写命题边前核验既有 RoleBinding 不是部分定义或竞争定义。"""
        expected_metadata = (
            scope,
            provenance_kind,
            epistemic_origin,
            content_version,
            qualifiers,
        )
        for binding in definition.canonical_bindings():
            identity = binding.identity_for(definition.proposition)
            ref = self._ontology.resolve(identity)
            if ref is None:
                continue
            role_statements = self._ontology.statements(
                predicate=self._predicates.binding_role, subject=ref)
            filler_statements = self._ontology.statements(
                predicate=self._predicates.binding_filler, subject=ref)
            if not role_statements and not filler_statements:
                continue
            if len(role_statements) != 1 or len(filler_statements) != 1:
                raise SemanticTopologyError(
                    "已有 RoleBinding 是部分定义或存在竞争端点")
            role_statement = role_statements[0]
            filler_statement = filler_statements[0]
            if (self._ontology.identity_of(role_statement.object)
                    != binding.role
                    or self._ontology.identity_of(filler_statement.object)
                    != binding.filler):
                raise SemanticTopologyError(
                    "已有 RoleBinding 端点与完整身份不一致")
            actual_metadata = self._uniform_metadata(
                [role_statement, filler_statement])
            if actual_metadata != expected_metadata:
                raise SemanticTopologyError(
                    "已有 RoleBinding 的 scope 或来源元数据不一致")

    @staticmethod
    def _require_replay_metadata(
            existing: MaterializedAtomicProposition, *,
            scope: ScopeIdentity, provenance_kind: int,
            epistemic_origin: int, content_version: int,
            qualifiers: tuple[int, ...]) -> None:
        """精确重放必须保留同一 scope 和 statement 元数据，禁止暗增平行定义。"""
        expected = (
            scope,
            provenance_kind,
            epistemic_origin,
            content_version,
            qualifiers,
        )
        actual = (
            existing.scope,
            existing.provenance_kind,
            existing.epistemic_origin,
            existing.content_version,
            existing.qualifiers,
        )
        if actual != expected:
            raise SemanticTopologyError(
                "同一 Proposition 的定义重放元数据不一致")

    def _relate(
            self, predicate: TypedRef, subject: TypedRef,
            object_ref: TypedRef, *, scope: ScopeIdentity,
            provenance_kind: int, epistemic_origin: int,
            content_version: int, qualifiers: tuple[int, ...]
            ) -> GraphStatement:
        """统一追加定义 statement，保持六类拓扑边使用完全相同的来源元数据。"""
        return self._ontology.relate(
            predicate,
            subject,
            object_ref,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )

    def _single(self, predicate: TypedRef, subject: TypedRef, *,
                label: str) -> GraphStatement:
        """读取单值协议槽；缺失或多值均拒绝，不能靠排序选择首行。"""
        statements = self._ontology.statements(
            predicate=predicate, subject=subject)
        if len(statements) != 1:
            raise SemanticTopologyError(
                f"{label} statement 必须恰有一条，实际 {len(statements)} 条")
        return statements[0]

    @staticmethod
    def _uniform_metadata(
            statements: list[GraphStatement]
            ) -> tuple[ScopeIdentity, int, int, int, tuple[int, ...]]:
        """核验同一原子定义的全部 statement 使用一致 scope 和来源元数据。"""
        if not statements:
            raise SemanticTopologyError("原子命题定义没有 statement")
        first = statements[0].assertion
        expected = (
            first.scope,
            first.provenance_kind,
            first.epistemic_origin,
            first.content_version,
            first.qualifiers,
        )
        for statement in statements[1:]:
            assertion = statement.assertion
            actual = (
                assertion.scope,
                assertion.provenance_kind,
                assertion.epistemic_origin,
                assertion.content_version,
                assertion.qualifiers,
            )
            if actual != expected:
                raise SemanticTopologyError(
                    "原子命题定义 statement 的 scope 或来源元数据不一致")
        return expected


__all__ = [
    "AtomicPropositionPredicates",
    "MaterializedAtomicProposition",
    "MaterializedRoleBinding",
    "SemanticGraph",
    "SemanticTopologyError",
]
