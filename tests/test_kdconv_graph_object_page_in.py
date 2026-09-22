from __future__ import annotations

import json

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import entity_identity
from pure_integer_ai.experiments.kdconv_structure_bridge import (
    NAMESPACE,
    SOURCE_KIND,
    KdConvStructure,
    KdConvStructureRuntime,
    _positive,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import SQLiteBackend


def _codes(value: str) -> tuple[int, ...]:
    return tuple(map(ord, value))


def test_lazy_runtime_pages_in_complete_graph_object_identities(tmp_path) -> None:
    subject = _codes("同一主题")
    first_value = _codes("第一个值")
    rows = (
        KdConvStructure(1, subject, _codes("属性甲"), first_value, 2),
        KdConvStructure(1, subject, _codes("属性乙"), _codes("第二个值"), 1),
    )
    course = tmp_path / "kdconv.int.jsonl"
    course.write_text("".join(
        json.dumps(
            item.to_dict(), ensure_ascii=True, sort_keys=True,
            separators=(",", ":")) + "\n"
        for item in rows
    ), encoding="ascii")

    backend = SQLiteBackend(":memory:")
    try:
        context = make_train_context(backend)
        KdConvStructureRuntime(context, course)
        runtime = KdConvStructureRuntime(context, None, materialize=False)

        entity = entity_identity(
            SourceRef(
                SOURCE_KIND, 1, 1,
                GLOBAL_OWNER_SCOPE, VersionBundle()),
            (1, _positive(subject)),
        )
        concept = concept_identity((
            NAMESPACE, 2, 1, _positive(first_value)))

        assert runtime.facts == ()
        entity_facts = runtime.page_in_for_graph_object_keys(
            (entity.stable_key(),))
        assert len(entity_facts) == 2
        assert all(any(
            binding.filler == entity for binding in fact.bindings)
                   for fact in entity_facts)

        concept_facts = runtime.page_in_for_graph_object_keys(
            (concept.stable_key(),))
        assert len(concept_facts) == 1
        assert any(
            binding.filler == concept
            for binding in concept_facts[0].bindings)
    finally:
        backend.close()
