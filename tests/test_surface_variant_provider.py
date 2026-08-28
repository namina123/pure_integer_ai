"""图内表面变体读取器的跨语言、无旁路规则契约。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    representation_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.cognition.shared.unicode_representation import (
    representation_surface,
)
from pure_integer_ai.cognition.understanding.surface_variant_provider import (
    GraphSurfaceVariantProvider,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT


@pytest.mark.parametrize("backend_kind", ["dict", "sqlite"])
def test_graph_provider_reads_only_injected_representation_relation(
        backend_kind: str) -> None:
    backend = DictBackend() if backend_kind == "dict" else SQLiteBackend()
    try:
        context = make_train_context(backend)
        ontology = context.graph_ontology
        family = (88001, 99002)
        relation_key = (88003, 1)
        predicate = ontology.materialize(
            relation_concept_identity(relation_key))
        source = session_scope(41)
        original = ontology.materialize(
            representation_identity(family, (0x7532,)))
        outgoing = ontology.materialize(
            representation_identity(family, (0x4E59,)))
        incoming = ontology.materialize(
            representation_identity(family, (0x4E19,)))
        unrelated = ontology.materialize(
            representation_identity((88004,), (0x7532,)))
        ontology.relate(
            predicate, original, outgoing, scope=source,
            provenance_kind=SOURCE_BARE_TEXT, epistemic_origin=EPI_STRUCTURED)
        ontology.relate(
            predicate, incoming, original, scope=source,
            provenance_kind=SOURCE_BARE_TEXT, epistemic_origin=EPI_STRUCTURED)
        ontology.relate(
            predicate, original, unrelated, scope=source,
            provenance_kind=SOURCE_BARE_TEXT, epistemic_origin=EPI_STRUCTURED)

        provider = GraphSurfaceVariantProvider(
            ontology, family, relation_key)
        assert provider("甲") == ("丙", "乙")
        assert provider("未知") == ()
        assert representation_surface(
            ontology.identity_of(outgoing), family_key=family) == "乙"
    finally:
        backend.close()
