"""Restore Memory's grounded input scope from every original candidate."""
from dataclasses import replace
from types import SimpleNamespace

from pure_integer_ai.cognition.shared.query_state import BindingEntry
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION, MEMORY_CATEGORY_OPEN_RELATION, frame_key,
)
from pure_integer_ai.cognition.understanding.query_reference_context import reference_marker_sources
from pure_integer_ai.experiments.current_reference_scope import read_current_reference_scope
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError, _unpack
from pure_integer_ai.experiments.query_reference_source_consensus import REFERENCE_SOURCE_CONSENSUS_VERSION
from pure_integer_ai.experiments.trained_reference_resolution import TrainedCrossTurnReferenceRuntime
from pure_integer_ai.experiments.trained_response_occurrence import ReferenceCandidateSet


def restore_saved_reference_selection(memory, trace, report, prepared, occurrences,
                                      contexts, relation_map, *, reference_bridge):
    """Rebuild full grounding before comparing the trace's selected Role set.

    Physical input roles and historical contexts must already have been cold
    restored by the caller. Neither the final answer nor trace's winner is
    an input to the selection. This never invokes query, delivery or writes.
    """
    from pure_integer_ai.experiments.trained_graph_query_bridge import (
        TrainedGraphQueryBridge, _observed_input_role_keys,
    )
    projection = report.get('projection')
    original = tuple(item.stable_key() for item in prepared.relation_question_bindings)
    if (reference_bridge is None or not isinstance(projection, dict)
            or tuple(tuple(key) for key in projection.get('question_bindings', ())) != original):
        raise QueryMemoryOheEvidenceError('initial reference selection lacks its complete cold question set')
    if len({item.stable_key() for item in occurrences}) != len(occurrences):
        raise QueryMemoryOheEvidenceError('initial reference selection repeats an input Role')
    if ({item.stable_key() for item in relation_map.values()}
            != {item.stable_key() for item in prepared.open_relations}
            or set(relation_map) != {item.hypothesis.stable_key() for item in occurrences}):
        raise QueryMemoryOheEvidenceError('initial reference selection omits a cold input candidate or its Roles')
    antecedents = {role.stable_key(): role for context in contexts for role in context.role_occurrences}
    depths = {key: min(context.context_depth for context in contexts if role in context.role_occurrences)
              for key, role in antecedents.items()}
    observed = _observed_input_role_keys(prepared)
    sets = []
    observation = None
    for role in occurrences:
        if observation is None:
            observation = role.observation
        if role.observation != observation or role.hypothesis.stable_key() not in relation_map:
            raise QueryMemoryOheEvidenceError('initial reference Role is outside its original Observation/Hypothesis')
        relation = relation_map[role.hypothesis.stable_key()]
        if relation not in prepared.open_relations:
            raise QueryMemoryOheEvidenceError('initial reference Hypothesis is outside the complete input')
        if role.role.stable_key() in observed or not reference_marker_sources(
                role.values, role.role.stable_key(), relation.frame.predicate,
                reference_bridge.relation_question_frames):
            continue
        prior = tuple(antecedents[key] for key in sorted(antecedents)
                      if antecedents[key].source_ref != role.source_ref)
        if prior:
            sets.append(ReferenceCandidateSet(role, prior, tuple(depths[item.stable_key()] for item in prior)))
    if observation is None or observation.stable_key() != tuple(report['observation_ref']):
        raise QueryMemoryOheEvidenceError('reference selection lacks original input Roles')
    state = SimpleNamespace(_input_observation_ref=observation,
        _reference_candidate_sets=tuple(sorted(sets, key=lambda item: item.stable_key())),
        _reference_runtime=TrainedCrossTurnReferenceRuntime(reference_bridge.core_runtime,
            reference_bridge.input_projector, memory.occurrence_index),
        _response_contexts=contexts,
        _current_reference_scope=read_current_reference_scope(memory, observation, contexts),
        _input_open_relations_by_hypothesis=relation_map,
        _memory_grounded_reference_source_consensus=None,
        _relation_slot_question_parts=reference_bridge._relation_slot_question_parts)
    state._reference_relation_context = lambda candidate, structure: (
        TrainedGraphQueryBridge._reference_relation_context(state, candidate, structure))
    selected = TrainedGraphQueryBridge._memory_grounded_slot_binding(state, prepared)
    consensus = state._memory_grounded_reference_source_consensus
    expected = (() if consensus is None else (BindingEntry(
        (REFERENCE_SOURCE_CONSENSUS_VERSION, 1), consensus.stable_key(),
        space=2, scope_key=prepared.source_ref).stable_key(),))
    actual = tuple(trace.resolve_ref(ref) for ref in trace.trace['state_key']['bindings']
        if _unpack(trace.resolve_ref(ref), 1, 3, 1)[0] == (REFERENCE_SOURCE_CONSENSUS_VERSION, 1))
    if actual != expected:
        raise QueryMemoryOheEvidenceError('reference source consensus differs from full cold selection')
    view = prepared
    retained_roles, retained_map = occurrences, relation_map
    if selected is not None:
        view = reference_bridge._relation_slot_question_view(
            replace(prepared, relation_question_bindings=(selected,)), selected)
        if view is None:
            raise QueryMemoryOheEvidenceError('cold grounded reference selection has no valid full input scope')
        candidate_keys = {frame_key(MEMORY_CANDIDATE_VERSION, (MEMORY_CATEGORY_OPEN_RELATION,), item.stable_key())
                          for item in view.open_relations}
        retained_map = {key: item for key, item in relation_map.items()
            if frame_key(MEMORY_CANDIDATE_VERSION, (MEMORY_CATEGORY_OPEN_RELATION,), item.stable_key()) in candidate_keys}
        retained_roles = tuple(item for item in occurrences if item.hypothesis.stable_key() in retained_map)
    retained_roles = tuple(sorted(retained_roles, key=lambda item: item.stable_key()))
    if tuple(item.stable_key() for item in retained_roles) != tuple(
            trace.resolve_ref(ref) for ref in trace.trace['input_role_occurrence_refs']):
        raise QueryMemoryOheEvidenceError('cold grounded reference scope differs from actual Memory Roles')
    return view, retained_roles, retained_map, dict(initial_question_candidates=len(original),
        initial_input_roles=len(occurrences), initial_reference_candidate_sets=len(sets),
        grounded_question_selected=int(selected is not None), selected_input_roles=len(retained_roles),
        source_consensus_verified=int(consensus is not None), query_call_count=0, memory_write_count=0)
