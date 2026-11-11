"""共享端点的真实 Core 展开不得因 frontier 去重丢失命题读取证明。"""
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity, OBJECT_PROPOSITION, concept_identity, structure_concept_identity,
)
from pure_integer_ai.cognition.shared.query_state import (
    EvidenceEntry, FrontierEntry, QueryState, SPACE_CORE, VisitedKey,
)
from pure_integer_ai.experiments import trained_graph_query_bridge as bridge_module
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge, _CrossSpaceExpander,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface, GraphRelationGeneration, RelationSurfaceBinding,
)


@pytest.mark.parametrize("target_visited", (False, True))
def test_shared_endpoint_keeps_each_consumed_fact_hop(monkeypatch, target_visited):
    """实际消费两来源时保留两 hop，后继最多一个；缓存本身仍不能授权回答。"""
    origin, target = (concept_identity((761, index)) for index in (1, 2))
    predicate = concept_identity((762, 1))
    roles = tuple(structure_concept_identity((763, index)) for index in (1, 2))
    facts = tuple(ActiveRelationSurface(
        ObjectIdentity(OBJECT_PROPOSITION, (764, index)), predicate, "",
        (RelationSurfaceBinding(roles[0], origin, "甲", index, 0, 1),
         RelationSurfaceBinding(roles[1], target, "乙", index, 1, 2)),
        "", index, 0, 0) for index in (1, 2))
    owner = SimpleNamespace(
        filler_edges={origin.stable_key(): facts},
        _partial_property_core_roots=frozenset(), _kdconv_query_propositions=frozenset(),
        _generic_graph_input_keys=frozenset(), _event_time_graph_input_keys=frozenset(),
        _event_time_query_edge_keys=frozenset(), _slot_question_core_roots=frozenset(),
        event_time_graph_adjacency={}, kdconv_fact_index={}, space_weights=(3, 2, 1),
        core_runtime=SimpleNamespace(context=SimpleNamespace(
            scoped_identity_store=SimpleNamespace(registry=object()))),
    )
    expander = object.__new__(_CrossSpaceExpander)
    expander.owner, expander.hops = owner, []
    expander._generation_input = lambda _prop: SimpleNamespace(
        hypothesis=SimpleNamespace(scope=SimpleNamespace(stable_key=lambda: (765,))))
    expander._route = lambda _key: object()
    expander._record_relation_capability = lambda *_args: None
    expander._record_typed_relation = lambda *_args: None
    expander._evidence_history = lambda _prop: ()
    expander._with_query_status = lambda state, _edge, **_kwargs: state
    monkeypatch.setattr(bridge_module, "core_query_evidence", lambda prop, *_args: (
        EvidenceEntry((766,), prop.stable_key(), space=SPACE_CORE, polarity=1),))
    edge = FrontierEntry((SPACE_CORE, *origin.stable_key()), SPACE_CORE,
                         target_key=origin.stable_key(), root_key=origin.stable_key())
    state = QueryState((767,), visited=(
        (VisitedKey(SPACE_CORE, target.stable_key()),) if target_visited else ()))
    outcome = expander._expand_core(state, (), edge, origin.stable_key())
    assert {hop.proposition for hop in expander.hops} == {
        fact.proposition.stable_key() for fact in facts}
    assert len(expander.hops) == 2
    assert len(outcome.state.frontier) == (0 if target_visited else 1)
    assert outcome.state.node_count == outcome.state.read_count == 2
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.core_fact_index = {fact.proposition.stable_key(): fact for fact in facts}
    bridge.generation_by_proposition = {fact.proposition.stable_key(): GraphRelationGeneration(
        "甲乙", fact.proposition, fact.source_hash, 2) for fact in facts}
    bridge.typed_relation_projections = {}
    bridge._event_time_query_bindings = ()
    for fact in facts:
        key = fact.proposition.stable_key()
        assert bridge._best_closed_fact(tuple(expander.hops), key)[0] == fact
        assert bridge._best_closed_fact((), key) is None
