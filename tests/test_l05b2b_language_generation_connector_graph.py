"""L-05B2B connector 理论图持久化和损坏拓扑对抗。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderHypothesisEngine,
)
from pure_integer_ai.cognition.understanding.order_constraint_promotion import (
    OrderConstraintPromoter,
)
from pure_integer_ai.experiments.language_generation_connector_graph import (
    LanguageConnectorGraphError,
    LanguageConnectorGraphPredicates,
    LanguageGenerationConnectorGraph,
)
from pure_integer_ai.storage.backend import DictBackend

from tests.test_l05b2b_language_generation_connector import (
    _connector,
    _selection_with_role,
)
from tests.test_s07_structure_order import (
    _active_plan,
    _domain,
    _graphs,
    _learning_protocol,
    _pattern,
)


_BASE = 15800


def _fixture(backend: DictBackend, *, variant: int):
    """建立已晋升 S-07 constraint 和尚未入图的 connector 理论。"""
    graphs = _graphs(backend)
    engine = OrderHypothesisEngine(_learning_protocol())
    domain = _domain(variant=variant, slot_count=3)
    promotion = _active_plan(
        engine,
        OrderConstraintPromoter(
            engine, graphs.order_graph, graphs.lifecycle),
        domain,
        _pattern(domain, first=1, second=2, kind=variant),
        event=variant,
        instance=variant,
    )
    role = promotion.slots[2].role
    selection = _selection_with_role(
        domain.language,
        role,
        concept_identity((_BASE + 1, variant)),
    )
    connector = _connector(promotion, selection, role)
    predicate_identities = tuple(
        concept_identity((_BASE + 2, variant, index))
        for index in range(21)
    )
    predicates = LanguageConnectorGraphPredicates(*tuple(
        graphs.context.graph_ontology.materialize(identity)
        for identity in predicate_identities
    ))
    graph = LanguageGenerationConnectorGraph(
        graphs.context.graph_ontology,
        graphs.order_graph,
        predicates,
        connector.registry.value_protocol,
    )
    return graphs, promotion, connector, predicate_identities, graph


def test_connector_theory_round_trips_after_runtime_cache_rebuild():
    """理论图清缓存后仍恢复完整模板，空集合和空 route 也有声明节点。"""
    backend = DictBackend()
    try:
        graphs, promotion, connector, predicate_identities, graph = _fixture(
            backend, variant=101)
        definition = connector.registry.templates[0]
        scope = promotion.constraint.scope

        written = graph.materialize(
            definition,
            scope=scope,
            provenance_kind=7,
            epistemic_origin=8,
            content_version=9,
            qualifiers=(10,),
        )

        assert written.definition == definition
        assert written.scope == scope
        assert written.qualifiers == (10,)
        assert graph.ontology.resolve(definition.context_set) is not None
        assert all(
            graph.ontology.resolve(item.prefix_route) is not None
            for item in definition.surface
        )
        before = backend.snapshot()
        graphs.context.graph_ontology.clear_runtime_caches()
        rebuilt = LanguageGenerationConnectorGraph(
            graphs.context.graph_ontology,
            graphs.order_graph,
            LanguageConnectorGraphPredicates(*tuple(
                graphs.context.graph_ontology.resolve(identity)
                for identity in predicate_identities
            )),
            connector.registry.value_protocol,
        )
        restored = rebuilt.read(definition.connector)

        assert restored == written
        assert backend.snapshot() == before
    finally:
        backend.close()


def test_connector_ordinal_is_graph_object_not_statement_qualifier():
    """Role 序由图内最小指令承担，statement qualifier 只保留来源限定。"""
    backend = DictBackend()
    try:
        _graphs_value, promotion, connector, _identities, graph = _fixture(
            backend, variant=102)
        definition = connector.registry.templates[0]
        graph.materialize(
            definition,
            scope=promotion.constraint.scope,
            provenance_kind=11,
            qualifiers=(12,),
        )
        role_binding = next(
            item for item in definition.bindings
            if item.ordinal is not None
        )
        binding_ref = graph.ontology.resolve(role_binding.binding)
        rows = graph.ontology.statements(
            predicate=graph.predicates.binding_ordinal,
            subject=binding_ref,
        )

        assert len(rows) == 1
        assert graph.ontology.identity_of(rows[0].object) == role_binding.ordinal
        assert rows[0].assertion.qualifiers == (12,)
        assert connector.registry.value_protocol.ordinal_value(
            role_binding.ordinal) == 0
    finally:
        backend.close()


def test_connector_graph_rejects_partial_root_and_competing_binding_source():
    """内部孤立边和 binding 竞争端点都不得被规范排序掩盖。"""
    partial_backend = DictBackend()
    try:
        _graphs_value, promotion, connector, _identities, graph = _fixture(
            partial_backend, variant=103)
        definition = connector.registry.templates[0]
        binding = definition.bindings[0]
        graph.ontology.relate(
            graph.predicates.binding_slot,
            graph.ontology.materialize(binding.binding),
            graph.ontology.materialize(binding.slot),
            scope=promotion.constraint.scope,
            provenance_kind=13,
        )

        with pytest.raises(LanguageConnectorGraphError, match="部分拓扑"):
            graph.preflight(
                definition,
                scope=promotion.constraint.scope,
                provenance_kind=13,
            )
    finally:
        partial_backend.close()

    conflict_backend = DictBackend()
    try:
        _graphs_value, promotion, connector, _identities, graph = _fixture(
            conflict_backend, variant=104)
        definition = connector.registry.templates[0]
        scope = promotion.constraint.scope
        graph.materialize(
            definition,
            scope=scope,
            provenance_kind=14,
        )
        binding = definition.bindings[0]
        graph.ontology.relate(
            graph.predicates.binding_source,
            graph.ontology.resolve(binding.binding),
            graph.ontology.materialize(
                minimal_instruction_identity((_BASE + 3, 104))),
            scope=scope,
            provenance_kind=14,
        )

        with pytest.raises(LanguageConnectorGraphError, match="binding source"):
            graph.read(definition.connector)
    finally:
        conflict_backend.close()
