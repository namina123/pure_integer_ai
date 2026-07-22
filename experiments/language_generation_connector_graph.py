"""持久化并严格恢复 connector 语言理论，不保存运行搜索策略。"""
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
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.structure_order import StructureOrderGraph
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorSlotBinding,
    LanguageConnectorSurfaceDirective,
    LanguageConnectorValueProtocol,
    LanguageGenerationConnectorRegistry,
    LanguageGenerationConnectorTemplate,
)


class LanguageConnectorGraphError(RuntimeError):
    """connector 理论图出现缺边、竞争端点、复用或来源漂移。"""


@dataclass(frozen=True)
class LanguageConnectorGraphPredicates:
    """connector 定义图使用的全部开放 predicate 槽位。"""

    connector_language: TypedRef
    connector_proposition_structure: TypedRef
    connector_predicate: TypedRef
    connector_sentence: TypedRef
    connector_surface_structure: TypedRef
    connector_binding: TypedRef
    binding_slot: TypedRef
    binding_source: TypedRef
    binding_role: TypedRef
    binding_ordinal: TypedRef
    binding_constant: TypedRef
    connector_constraint_set: TypedRef
    connector_context_set: TypedRef
    collection_member: TypedRef
    connector_boundary: TypedRef
    connector_linearization_reason: TypedRef
    connector_surface: TypedRef
    surface_slot: TypedRef
    surface_action: TypedRef
    surface_instruction: TypedRef
    surface_prefix_route: TypedRef

    def refs(self) -> tuple[TypedRef, ...]:
        """按协议槽位顺序返回全部 predicate。"""
        return (
            self.connector_language,
            self.connector_proposition_structure,
            self.connector_predicate,
            self.connector_sentence,
            self.connector_surface_structure,
            self.connector_binding,
            self.binding_slot,
            self.binding_source,
            self.binding_role,
            self.binding_ordinal,
            self.binding_constant,
            self.connector_constraint_set,
            self.connector_context_set,
            self.collection_member,
            self.connector_boundary,
            self.connector_linearization_reason,
            self.connector_surface,
            self.surface_slot,
            self.surface_action,
            self.surface_instruction,
            self.surface_prefix_route,
        )


@dataclass(frozen=True)
class MaterializedLanguageConnectorTemplate:
    """从图恢复的 connector 理论及其完整来源 statement。"""

    definition: LanguageGenerationConnectorTemplate
    scope: ScopeIdentity
    provenance_kind: int
    epistemic_origin: int
    content_version: int
    qualifiers: tuple[int, ...]
    assertion_hashes: tuple[int, ...]


class LanguageGenerationConnectorGraph:
    """在 GraphOntology 中保存一等 binding、set、directive 和 route 拓扑。"""

    def __init__(
            self,
            ontology: GraphOntology,
            order_graph: StructureOrderGraph,
            predicates: LanguageConnectorGraphPredicates,
            value_protocol: LanguageConnectorValueProtocol,
            ) -> None:
        if not isinstance(ontology, GraphOntology):
            raise TypeError("connector graph ontology 类型错误")
        if not isinstance(order_graph, StructureOrderGraph):
            raise TypeError("connector graph order_graph 类型错误")
        if order_graph.ontology is not ontology:
            raise ValueError("connector graph 与 S-07 必须共享 GraphOntology")
        if not isinstance(predicates, LanguageConnectorGraphPredicates):
            raise TypeError("connector graph predicates 类型错误")
        if not isinstance(value_protocol, LanguageConnectorValueProtocol):
            raise TypeError("connector graph value protocol 类型错误")
        refs = predicates.refs()
        if len(set(refs)) != len(refs) or any(
                ontology.identity_of(ref).object_kind != OBJECT_CONCEPT
                for ref in refs):
            raise LanguageConnectorGraphError(
                "connector graph predicate 必须是互异一等 Concept")
        self.ontology = ontology
        self.order_graph = order_graph
        self.predicates = predicates
        self.value_protocol = value_protocol

    def preflight(
            self,
            definition: LanguageGenerationConnectorTemplate,
            *,
            scope: ScopeIdentity,
            provenance_kind: int,
            epistemic_origin: int = 0,
            content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> None:
        """在首条写入前核验理论、S-07 引用、已有拓扑和来源元数据。"""
        self._validate_definition(definition)
        expected = self._validate_metadata(
            scope,
            provenance_kind,
            epistemic_origin,
            content_version,
            qualifiers,
        )
        connector = self.ontology.resolve(definition.connector)
        if connector is not None and self._protocol_rows(connector):
            restored = self.read(definition.connector)
            if restored.definition != definition:
                raise LanguageConnectorGraphError(
                    "已有 connector 图定义与课程声明不一致")
            if self._materialized_metadata(restored) != expected:
                raise LanguageConnectorGraphError(
                    "已有 connector 图来源元数据不一致")
            return
        for identity in self._owned_structures(definition):
            ref = self.ontology.resolve(identity)
            if ref is not None and self._protocol_rows(ref):
                raise LanguageConnectorGraphError(
                    "connector 根缺失但内部结构已有部分拓扑")

    def materialize(
            self,
            definition: LanguageGenerationConnectorTemplate,
            *,
            scope: ScopeIdentity,
            provenance_kind: int,
            epistemic_origin: int = 0,
            content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> MaterializedLanguageConnectorTemplate:
        """预检后幂等写入完整理论；空集合和空 prefix 仍保留一等集合节点。"""
        self.preflight(
            definition,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        connector = self.ontology.materialize(definition.connector)
        metadata = (
            scope,
            provenance_kind,
            epistemic_origin,
            content_version,
            qualifiers,
        )
        root_targets = (
            (self.predicates.connector_language,
             definition.language_branch),
            (self.predicates.connector_proposition_structure,
             definition.proposition_structure),
            (self.predicates.connector_predicate, definition.predicate),
            (self.predicates.connector_sentence, definition.sentence),
            (self.predicates.connector_surface_structure,
             definition.structure),
            (self.predicates.connector_constraint_set,
             definition.constraint_set),
            (self.predicates.connector_context_set,
             definition.context_set),
            (self.predicates.connector_boundary, definition.boundary),
            (self.predicates.connector_linearization_reason,
             definition.linearization_reason),
        )
        for predicate, target in root_targets:
            self._relate(predicate, connector, target, metadata)
        for binding in definition.bindings:
            binding_ref = self._relate(
                self.predicates.connector_binding,
                connector,
                binding.binding,
                metadata,
            ).object
            for predicate, target in (
                    (self.predicates.binding_slot, binding.slot),
                    (self.predicates.binding_source, binding.source)):
                self._relate(predicate, binding_ref, target, metadata)
            for predicate, target in (
                    (self.predicates.binding_role, binding.role),
                    (self.predicates.binding_ordinal, binding.ordinal),
                    (self.predicates.binding_constant, binding.constant)):
                if target is not None:
                    self._relate(predicate, binding_ref, target, metadata)
        for collection, members in (
                (definition.constraint_set, definition.constraints),
                (definition.context_set, definition.context)):
            collection_ref = self.ontology.materialize(collection)
            for member in members:
                self._relate(
                    self.predicates.collection_member,
                    collection_ref,
                    member,
                    metadata,
                )
        for surface in definition.surface:
            surface_ref = self._relate(
                self.predicates.connector_surface,
                connector,
                surface.directive,
                metadata,
            ).object
            for predicate, target in (
                    (self.predicates.surface_slot, surface.slot),
                    (self.predicates.surface_action, surface.action),
                    (self.predicates.surface_instruction,
                     surface.instruction),
                    (self.predicates.surface_prefix_route,
                     surface.prefix_route)):
                self._relate(predicate, surface_ref, target, metadata)
            route_ref = self.ontology.materialize(surface.prefix_route)
            for step in surface.surface_prefix_steps:
                self._relate(
                    self.predicates.collection_member,
                    route_ref,
                    step,
                    metadata,
                )
        restored = self.read(definition.connector)
        if restored.definition != definition:
            raise LanguageConnectorGraphError("connector 理论写后恢复不一致")
        return restored

    def read(
            self,
            connector: ObjectIdentity,
            ) -> MaterializedLanguageConnectorTemplate:
        """从 connector 根和 S-07 图严格恢复完整语言理论定义。"""
        connector_ref = self.ontology.resolve(connector)
        if connector_ref is None:
            raise LanguageConnectorGraphError("connector 尚未物化")
        statements: list[GraphStatement] = []

        def one(predicate: TypedRef, label: str) -> ObjectIdentity:
            row = self._one(connector_ref, predicate, label)
            statements.append(row)
            return self.ontology.identity_of(row.object)

        language_branch = one(
            self.predicates.connector_language, "language branch")
        proposition_structure = one(
            self.predicates.connector_proposition_structure,
            "proposition structure",
        )
        semantic_predicate = one(
            self.predicates.connector_predicate, "semantic predicate")
        sentence = one(self.predicates.connector_sentence, "sentence")
        structure = one(
            self.predicates.connector_surface_structure,
            "surface structure",
        )
        constraint_set = one(
            self.predicates.connector_constraint_set, "constraint set")
        context_set = one(
            self.predicates.connector_context_set, "context set")
        boundary = one(self.predicates.connector_boundary, "boundary")
        reason = one(
            self.predicates.connector_linearization_reason,
            "linearization reason",
        )
        materialized_structure = self.order_graph.read_structure(
            self.ontology.resolve(structure))
        slots = tuple(item.definition for item in materialized_structure.slots)
        bindings = tuple(
            self._read_binding(row, statements)
            for row in self._many(
                connector_ref,
                self.predicates.connector_binding,
                "binding",
                require_nonempty=True,
            )
        )
        constraints = self._read_collection(
            connector_ref,
            self.predicates.connector_constraint_set,
            constraint_set,
            statements,
        )
        context = self._read_collection(
            connector_ref,
            self.predicates.connector_context_set,
            context_set,
            statements,
        )
        surface = tuple(
            self._read_surface(row, statements)
            for row in self._many(
                connector_ref,
                self.predicates.connector_surface,
                "surface directive",
                require_nonempty=True,
            )
        )
        definition = LanguageGenerationConnectorTemplate(
            connector,
            language_branch,
            proposition_structure,
            semantic_predicate,
            sentence,
            structure,
            slots,
            bindings,
            constraint_set,
            constraints,
            context_set,
            context,
            boundary,
            reason,
            surface,
        )
        LanguageGenerationConnectorRegistry(
            self.value_protocol, (definition,))
        self._validate_definition(definition)
        metadata = self._uniform_metadata(tuple(statements))
        return MaterializedLanguageConnectorTemplate(
            definition,
            *metadata,
            tuple(sorted(item.assertion_hash for item in statements)),
        )

    def _read_binding(
            self,
            root: GraphStatement,
            statements: list[GraphStatement],
            ) -> LanguageConnectorSlotBinding:
        """从一等 binding 节点恢复 slot/source 及可选 Role/序/常量。"""
        statements.append(root)
        binding_ref = root.object
        binding = self.ontology.identity_of(binding_ref)
        slot_row = self._one(
            binding_ref, self.predicates.binding_slot, "binding slot")
        source_row = self._one(
            binding_ref, self.predicates.binding_source, "binding source")
        statements.extend((slot_row, source_row))

        def optional(predicate: TypedRef, label: str) -> ObjectIdentity | None:
            rows = self._many(binding_ref, predicate, label)
            if len(rows) > 1:
                raise LanguageConnectorGraphError(
                    f"connector {label} 存在竞争端点")
            statements.extend(rows)
            return (
                None if not rows else self.ontology.identity_of(rows[0].object))

        return LanguageConnectorSlotBinding(
            binding,
            self.ontology.identity_of(slot_row.object),
            self.ontology.identity_of(source_row.object),
            optional(self.predicates.binding_role, "binding role"),
            optional(self.predicates.binding_ordinal, "binding ordinal"),
            optional(self.predicates.binding_constant, "binding constant"),
        )

    def _read_surface(
            self,
            root: GraphStatement,
            statements: list[GraphStatement],
            ) -> LanguageConnectorSurfaceDirective:
        """从一等 directive 和 route 节点恢复 emit/silent 理论。"""
        statements.append(root)
        directive_ref = root.object
        rows = tuple(
            self._one(directive_ref, predicate, label)
            for predicate, label in (
                (self.predicates.surface_slot, "surface slot"),
                (self.predicates.surface_action, "surface action"),
                (self.predicates.surface_instruction, "surface instruction"),
                (self.predicates.surface_prefix_route, "surface prefix route"),
            )
        )
        statements.extend(rows)
        route = self.ontology.identity_of(rows[3].object)
        prefix_rows = self._many(
            rows[3].object,
            self.predicates.collection_member,
            "surface prefix member",
        )
        statements.extend(prefix_rows)
        self._require_unique_incoming(
            rows[3].object,
            self.predicates.surface_prefix_route,
            directive_ref,
            "surface prefix route",
        )
        return LanguageConnectorSurfaceDirective(
            self.ontology.identity_of(directive_ref),
            self.ontology.identity_of(rows[0].object),
            self.ontology.identity_of(rows[1].object),
            self.ontology.identity_of(rows[2].object),
            route,
            tuple(self.ontology.identity_of(row.object)
                  for row in prefix_rows),
        )

    def _read_collection(
            self,
            connector: TypedRef,
            root_predicate: TypedRef,
            collection: ObjectIdentity,
            statements: list[GraphStatement],
            ) -> tuple[ObjectIdentity, ...]:
        """恢复显式集合成员，并核验集合节点未被其他 connector 复用。"""
        collection_ref = self.ontology.resolve(collection)
        if collection_ref is None:
            raise LanguageConnectorGraphError("connector 集合节点尚未物化")
        self._require_unique_incoming(
            collection_ref,
            root_predicate,
            connector,
            "connector collection",
        )
        rows = self._many(
            collection_ref,
            self.predicates.collection_member,
            "collection member",
        )
        statements.extend(rows)
        return tuple(self.ontology.identity_of(row.object) for row in rows)

    def _validate_definition(
            self,
            definition: LanguageGenerationConnectorTemplate,
            ) -> None:
        """核验 connector 定义与已持久化 S-07 structure/constraint 完整一致。"""
        if not isinstance(definition, LanguageGenerationConnectorTemplate):
            raise TypeError("connector graph definition 类型错误")
        LanguageGenerationConnectorRegistry(
            self.value_protocol, (definition,))
        structure_ref = self.ontology.resolve(definition.structure)
        if structure_ref is None:
            raise LanguageConnectorGraphError("connector 引用的 S-07 structure 缺失")
        materialized = self.order_graph.read_structure(structure_ref)
        slots = tuple(item.definition for item in materialized.slots)
        if slots != definition.slots:
            raise LanguageConnectorGraphError(
                "connector slot schema 与 S-07 图不一致")
        for constraint in definition.constraints:
            constraint_ref = self.ontology.resolve(constraint)
            if constraint_ref is None:
                raise LanguageConnectorGraphError(
                    "connector 引用的 S-07 constraint 缺失")
            restored = self.order_graph.read_constraint(constraint_ref)
            if restored.definition.structure != definition.structure:
                raise LanguageConnectorGraphError(
                    "connector constraint 不属于目标 surface structure")

    def _relate(
            self,
            predicate: TypedRef,
            subject: TypedRef,
            target: ObjectIdentity,
            metadata: tuple[ScopeIdentity, int, int, int, tuple[int, ...]],
            ) -> GraphStatement:
        """按统一来源元数据追加一条 connector 理论 statement。"""
        scope, provenance, epistemic, version, qualifiers = metadata
        return self.ontology.relate(
            predicate,
            subject,
            self.ontology.materialize(target),
            scope=scope,
            provenance_kind=provenance,
            epistemic_origin=epistemic,
            content_version=version,
            qualifiers=qualifiers,
        )

    def _one(
            self,
            subject: TypedRef,
            predicate: TypedRef,
            label: str,
            ) -> GraphStatement:
        """读取唯一出边；缺失或竞争时 fail closed。"""
        rows = self._many(subject, predicate, label)
        if len(rows) != 1:
            raise LanguageConnectorGraphError(
                f"connector {label} 必须恰有一条，实际 {len(rows)} 条")
        return rows[0]

    def _many(
            self,
            subject: TypedRef,
            predicate: TypedRef,
            label: str,
            *,
            require_nonempty: bool = False,
            ) -> tuple[GraphStatement, ...]:
        """按对象稳定键规范化读取同一 predicate 的全部出边。"""
        rows = self.ontology.statements(predicate=predicate, subject=subject)
        if require_nonempty and not rows:
            raise LanguageConnectorGraphError(f"connector {label} 不得为空")
        return tuple(sorted(
            rows,
            key=lambda item: self.ontology.identity_of(
                item.object).stable_key(),
        ))

    def _require_unique_incoming(
            self,
            object_ref: TypedRef,
            predicate: TypedRef,
            expected_subject: TypedRef,
            label: str,
            ) -> None:
        """禁止内部一等结构被多个父节点复用。"""
        incoming = self.ontology.statements(
            predicate=predicate,
            object_ref=object_ref,
        )
        if len(incoming) != 1 or incoming[0].subject != expected_subject:
            raise LanguageConnectorGraphError(
                f"connector {label} 被复用或缺少唯一父节点")

    def _protocol_rows(self, ref: TypedRef) -> tuple[GraphStatement, ...]:
        """返回节点在 connector 协议下的全部入边和出边。"""
        rows: dict[int, GraphStatement] = {}
        for predicate in self.predicates.refs():
            for row in self.ontology.statements(
                    predicate=predicate, subject=ref):
                rows[row.assertion_hash] = row
            for row in self.ontology.statements(
                    predicate=predicate, object_ref=ref):
                rows[row.assertion_hash] = row
        return tuple(rows[key] for key in sorted(rows))

    @staticmethod
    def _owned_structures(
            definition: LanguageGenerationConnectorTemplate,
            ) -> tuple[ObjectIdentity, ...]:
        """返回 connector 独占的一等内部结构节点。"""
        return (
            definition.constraint_set,
            definition.context_set,
            *(item.binding for item in definition.bindings),
            *(item.directive for item in definition.surface),
            *(item.prefix_route for item in definition.surface),
        )

    @staticmethod
    def _validate_metadata(
            scope: ScopeIdentity,
            provenance_kind: int,
            epistemic_origin: int,
            content_version: int,
            qualifiers: tuple[int, ...],
            ) -> tuple[ScopeIdentity, int, int, int, tuple[int, ...]]:
        """核验 connector statement 的 scope 和严格整数来源元数据。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("connector graph scope 类型错误")
        if not isinstance(qualifiers, tuple):
            raise TypeError("connector graph qualifiers 必须是 tuple")
        assert_int(
            provenance_kind,
            epistemic_origin,
            content_version,
            *qualifiers,
            _where="LanguageGenerationConnectorGraph",
        )
        if (type(provenance_kind) is not int or provenance_kind <= 0
                or type(epistemic_origin) is not int
                or epistemic_origin < 0
                or type(content_version) is not int
                or content_version < 0
                or any(type(item) is not int for item in qualifiers)):
            raise ValueError("connector graph 来源元数据非法")
        return (
            scope,
            provenance_kind,
            epistemic_origin,
            content_version,
            qualifiers,
        )

    @staticmethod
    def _statement_metadata(
            statement: GraphStatement,
            ) -> tuple[ScopeIdentity, int, int, int, tuple[int, ...]]:
        """提取一条 connector statement 的完整来源元数据。"""
        assertion = statement.assertion
        return (
            assertion.scope,
            assertion.provenance_kind,
            assertion.epistemic_origin,
            assertion.content_version,
            assertion.qualifiers,
        )

    def _uniform_metadata(
            self,
            statements: tuple[GraphStatement, ...],
            ) -> tuple[ScopeIdentity, int, int, int, tuple[int, ...]]:
        """要求 connector 理论全部 statement 使用同一来源元数据。"""
        if not statements:
            raise LanguageConnectorGraphError("connector 理论没有 statement")
        expected = self._statement_metadata(statements[0])
        if any(self._statement_metadata(item) != expected
               for item in statements[1:]):
            raise LanguageConnectorGraphError(
                "connector 理论 statement 来源元数据漂移")
        return expected

    @staticmethod
    def _materialized_metadata(
            value: MaterializedLanguageConnectorTemplate,
            ) -> tuple[ScopeIdentity, int, int, int, tuple[int, ...]]:
        """提取已恢复 connector 定义的来源元数据。"""
        return (
            value.scope,
            value.provenance_kind,
            value.epistemic_origin,
            value.content_version,
            value.qualifiers,
        )


__all__ = [
    "LanguageConnectorGraphError",
    "LanguageConnectorGraphPredicates",
    "LanguageGenerationConnectorGraph",
    "MaterializedLanguageConnectorTemplate",
]
