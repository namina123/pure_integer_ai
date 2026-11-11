"""Response connector features must follow the current input's graph links."""
from __future__ import annotations

from types import SimpleNamespace

from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)
from pure_integer_ai.experiments.trained_reference_resolution import (
    REFERENCE_RESOLVED,
)
from pure_integer_ai.experiments.unknown_response_course import (
    RESPONSE_FEATURE_COLD_CONTEXT,
    RESPONSE_FEATURE_CONVERSATIONAL_OPENING,
    RESPONSE_FEATURE_HOT_CONTEXT,
    RESPONSE_FEATURE_REFERENCE_RESOLVED,
    RESPONSE_FEATURE_TOPIC,
    RESPONSE_FEATURE_TOPIC_GRAPH,
    RESPONSE_FEATURE_UNKNOWN,
)


class _FeatureProbe:
    _reference_generation_contexts = (
        TrainedGraphQueryBridge._reference_generation_contexts)
    _generic_response_feature_mask = (
        TrainedGraphQueryBridge._generic_response_feature_mask)
    _discourse_relation_feature_state = lambda self, _structure: (0, 0, 0, 0, 0, 0, 0)

    def __init__(self):
        self._reference_resolutions = ()
        self._response_contexts = ()
        self._current_discourse_topic_candidates = ()
        self._current_discourse_topic_graph_identity_keys = ()
        self._discourse_topic_candidates = ()
        self._discourse_topic_graph_identity_keys = ()
        self._reference_candidate_sets = ()
        self._discourse_graph_input_keys = ()
        self._event_time_query_bindings = ()
        self._event_time_graph_input_keys = ()
        self._core_graph_input_keys = ()
        self._generic_graph_input_keys = ()
        self._discourse_graph_input_keys = ()
        self.core_fact_index = {}


def _context(key: tuple[int, ...], depth: int):
    return SimpleNamespace(
        context_depth=depth,
        stable_key=lambda: key,
    )


def test_unrelated_restored_context_and_topics_do_not_condition_unknown_reply():
    bridge = _FeatureProbe()
    bridge._response_contexts = (_context((80,), 2),)
    bridge._discourse_topic_candidates = (
        SimpleNamespace(proposition_key=(90,)),)
    bridge._discourse_topic_graph_identity_keys = ((90,),)

    mask = bridge._generic_response_feature_mask((), None)

    historical_features = (
        RESPONSE_FEATURE_HOT_CONTEXT
        | RESPONSE_FEATURE_COLD_CONTEXT
        | RESPONSE_FEATURE_TOPIC
        | RESPONSE_FEATURE_TOPIC_GRAPH
    )
    assert mask == RESPONSE_FEATURE_UNKNOWN
    assert mask & historical_features == 0


def test_trained_dialogue_act_graph_adds_only_its_response_condition():
    bridge = _FeatureProbe()
    bridge._input_dialogue_act_match = SimpleNamespace(
        response_feature_mask=RESPONSE_FEATURE_CONVERSATIONAL_OPENING)

    mask = bridge._generic_response_feature_mask((), None)

    assert mask == (
        RESPONSE_FEATURE_UNKNOWN
        | RESPONSE_FEATURE_CONVERSATIONAL_OPENING)


def test_current_graph_topic_still_conditions_unknown_reply():
    bridge = _FeatureProbe()
    bridge._current_discourse_topic_candidates = (
        SimpleNamespace(proposition_key=(90,)),)
    bridge._current_discourse_topic_graph_identity_keys = ((90,),)

    mask = bridge._generic_response_feature_mask((), None)

    assert mask & RESPONSE_FEATURE_TOPIC
    assert mask & RESPONSE_FEATURE_TOPIC_GRAPH


def test_resolved_reference_uses_its_supported_cold_context():
    bridge = _FeatureProbe()
    context = _context((80,), 2)
    bridge._response_contexts = (context,)
    bridge._reference_resolutions = (SimpleNamespace(
        requires_resolution=True,
        status=REFERENCE_RESOLVED,
        support_context_keys=(context.stable_key(),),
    ),)

    mask = bridge._generic_response_feature_mask((), None)

    assert mask & RESPONSE_FEATURE_COLD_CONTEXT
    assert not mask & RESPONSE_FEATURE_HOT_CONTEXT
    assert mask & RESPONSE_FEATURE_REFERENCE_RESOLVED


def test_discourse_relation_features_ignore_restored_but_unlinked_topics():
    bridge = _FeatureProbe()
    bridge.discourse_topic_topology = SimpleNamespace(
        frames=(),
        relations=(SimpleNamespace(subject_key=(90,), object_key=(91,)),),
    )
    bridge._discourse_topic_candidates = (
        SimpleNamespace(proposition_key=(90,)),)

    assert TrainedGraphQueryBridge._discourse_relation_feature_state(
        bridge) == (0, 0, 0, 0, 0, 0, 0)

    bridge._current_discourse_topic_candidates = (
        SimpleNamespace(proposition_key=(90,)),)
    assert TrainedGraphQueryBridge._discourse_relation_feature_state(
        bridge)[0] == 1
