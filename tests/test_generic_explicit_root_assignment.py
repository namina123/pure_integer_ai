from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_LANGUAGE_ATOM,
    OBJECT_PROPOSITION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.query_state import (
    SPACE_CORE,
    SPACE_DIALOGUE,
    BindingEntry,
    EvidenceEntry,
)
from pure_integer_ai.experiments.response_generation_graph import (
    GENERIC_RESPONSE_GRAPH_CONCEPT,
    GENERIC_RESPONSE_GRAPH_ENTITY,
    GENERIC_RESPONSE_GRAPH_EVENT,
    GENERIC_RESPONSE_GRAPH_TOPIC,
)
from pure_integer_ai.experiments.trained_open_response_runtime import (
    _GenericGraphSlot,
    _GenericResponseContract,
    _generic_explicit_root_assignment,
    _select_generic_graph_assignment,
)


KINDS = {
    GENERIC_RESPONSE_GRAPH_ENTITY: OBJECT_ENTITY,
    GENERIC_RESPONSE_GRAPH_EVENT: OBJECT_EVENT,
    GENERIC_RESPONSE_GRAPH_CONCEPT: OBJECT_CONCEPT,
    GENERIC_RESPONSE_GRAPH_TOPIC: OBJECT_PROPOSITION,
}


def _fixture(*, ordinals=(1, 1, 1, 1), extra_entity=False, evidence=True):
    categories = (
        GENERIC_RESPONSE_GRAPH_ENTITY,
        GENERIC_RESPONSE_GRAPH_EVENT,
        GENERIC_RESPONSE_GRAPH_CONCEPT,
        GENERIC_RESPONSE_GRAPH_TOPIC,
    )
    slots = tuple(_GenericGraphSlot(
        SimpleNamespace(slot=ObjectIdentity(OBJECT_LANGUAGE_ATOM, (900 + i,))),
        category,
        ordinal,
    ) for i, (category, ordinal) in enumerate(zip(categories, ordinals)))
    contract = _GenericResponseContract(None, slots)
    roots = [ObjectIdentity(kind, (100 + i,))
             for i, kind in enumerate(KINDS.values())]
    if extra_entity:
        roots.append(ObjectIdentity(OBJECT_ENTITY, (999,)))

    bindings = []
    evidence_rows = []
    by_category = {category: {} for category in categories}
    for i, (category, root) in enumerate(zip(categories, roots)):
        space = SPACE_DIALOGUE if category == GENERIC_RESPONSE_GRAPH_TOPIC else SPACE_CORE
        scope = (500 + i,)
        binding = BindingEntry((700 + i,), root.stable_key(), space, scope)
        bindings.append(binding)
        by_category[category][root] = [binding]
        if evidence:
            evidence_rows.append(EvidenceEntry(
                (800 + i,), root.stable_key(), space=space, scope_key=scope))
    state = SimpleNamespace(bindings=tuple(bindings), evidence=tuple(evidence_rows))
    keys = tuple(root.stable_key() for root in roots)
    return state, contract, keys, by_category


def test_unique_evidenced_explicit_roots_bind_first_ordinal_typed_slots():
    state, contract, keys, by_category = _fixture()

    result = _generic_explicit_root_assignment(
        state, contract, keys, by_category, KINDS)

    assert result is not None
    assert len(result) == 4
    assert {filler.object_kind for filler, _bindings in result.values()} == {
        OBJECT_ENTITY, OBJECT_EVENT, OBJECT_CONCEPT, OBJECT_PROPOSITION,
    }


@pytest.mark.parametrize("kwargs", [
    {"extra_entity": True},
    {"evidence": False},
    {"ordinals": (1, 1, 2, 1)},
])
def test_explicit_root_binding_rejects_ambiguous_or_unqualified_slots(kwargs):
    state, contract, keys, by_category = _fixture(**kwargs)

    assert _generic_explicit_root_assignment(
        state, contract, keys, by_category, KINDS) is None


def test_explicit_root_binding_rejects_binding_outside_current_query_state():
    state, contract, keys, by_category = _fixture()
    state.bindings = ()

    assert _generic_explicit_root_assignment(
        state, contract, keys, by_category, KINDS) is None


def test_relation_assignment_precedes_explicit_roots_and_conflicts_stay_open():
    relation = {"relation": 1}
    explicit = {"explicit": 1}

    assert _select_generic_graph_assignment({(1,): relation}, explicit) == relation
    assert _select_generic_graph_assignment({}, explicit) == explicit
    assert _select_generic_graph_assignment(
        {(1,): relation, (2,): {"relation": 2}}, explicit) is None
