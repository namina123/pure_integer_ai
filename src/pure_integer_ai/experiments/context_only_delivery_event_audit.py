"""无对象 UNKNOWN 回应的持久交付审计；只读同次 Episode/Use/Outcome。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_EPISODE, MEMORY_EVENT_USE, MEMORY_EVENT_USE_OUTCOME,
    EpisodePayload, MemoryLinkedRef, MemoryObjectRef, UsePayload, UseOutcomePayload,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import frame_key
from pure_integer_ai.experiments.factual_delivery_event_audit import (
    _declaration, _event, _hashes, _observation,
)
from pure_integer_ai.experiments.trained_response_delivery import DELIVERY_PROOF_IDENTITY, _DELIVERY_USE
from pure_integer_ai.experiments.context_only_plan_slots import language_only_plan_slots


def verify_context_only_delivery_events(value, report, summary, proof, *, event_reader,
                                        declaration_reader, max_memory_refs=256):
    """逐项核验原输入、完整采用集合、输出及全部归因，缺任意末项也拒绝。"""
    from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
    trace = GraphQueryTraceEvidence(value)
    plan, generation = value.get('response_plan'), value.get('generation')
    if (not plan or not generation or plan.get('claim_refs') or plan.get('claim_evidence_refs')
            or not language_only_plan_slots(plan) or generation.get('kind') != 3
            or report.get('delivery_claim_count') != 0 or not report.get('response_surface')):
        raise ValueError('context-only delivery requires the original zero-claim no-object generated Plan')
    input_source = SourceRef.from_stable_key(tuple(report['input_source']))
    input_ref = MemoryObjectRef.from_stable_key(tuple(report['observation_ref']))
    proof_ref = ObjectIdentity.from_stable_key(tuple(summary['proof_ref']))
    output_source = SourceRef.from_stable_key(tuple(summary['output_source']))
    act = MemoryLinkedRef.object(ObjectIdentity.from_stable_key(tuple(plan['response_act'])))
    proof_link = MemoryLinkedRef.object(proof_ref)
    if (type(max_memory_refs) is not int or max_memory_refs <= 0
            or not isinstance(plan.get('memory_refs'), list) or not plan['memory_refs']
            or len(plan['memory_refs']) > max_memory_refs or summary.get('delivered') != 1
            or summary.get('fact_claims_added') != 0 or output_source == input_source
            or any(item.owner != input_source.owner or item.versions != input_source.versions
                   for item in (input_ref, proof_ref, output_source))):
        raise ValueError('context-only delivery source, owner, version or declared budget differs')
    refs = tuple(MemoryObjectRef.from_stable_key(tuple(key)) for key in plan['memory_refs'])
    used_refs = tuple(sorted(set(refs), key=lambda ref: ref.stable_key()))
    if len(refs) != len(used_refs):
        raise ValueError('context-only Plan repeats a Memory adoption')
    expected = frame_key(DELIVERY_PROOF_IDENTITY, (3,), input_source.stable_key(), input_ref.stable_key(),
        tuple(plan['response_act']), frame_key(1, *(ref.stable_key() for ref in used_refs)), frame_key(2))
    if proof != expected:
        raise ValueError('context-only physical proof differs from the complete current Plan')
    _observation(declaration_reader, input_ref, input_source)
    episode_hash, = _hashes((summary['episode_event_hash'],), 'Episode', max_memory_refs)
    episode_row = _event(event_reader, episode_hash, MEMORY_EVENT_EPISODE)
    episode, episode_ref = episode_row.event.payload, episode_row.event.object_ref
    links = tuple(MemoryLinkedRef.memory(ref) for ref in used_refs)
    if (not isinstance(episode, EpisodePayload) or episode.input_observation_ref != input_ref
            or episode.selected_path_ref != proof_link or episode.intent_ref != act
            or episode.failure_ref is not None or episode.used_memory_refs != used_refs
            or episode.candidate_refs != links or episode.result_signal_refs
            or episode_ref.owner != input_ref.owner or episode_ref.versions != input_ref.versions
            or episode_ref.memory_space != input_ref.memory_space
            or episode.session_ref.owner != input_ref.owner or episode.session_ref.versions != input_ref.versions
            or episode.session_ref.local_id != input_ref.owner.session_id
            or episode.round_seq != output_source.document_id or episode.output_ref is None):
        raise ValueError('context-only Episode differs from its current input, adoption, proof or session')
    output_refs = episode.output_ref.memory_refs()
    if len(output_refs) != 1:
        raise ValueError('context-only Episode must reference one output Observation')
    output_ref, = output_refs
    if (output_ref.owner != input_ref.owner or output_ref.versions != input_ref.versions
            or output_ref.memory_space != input_ref.memory_space):
        raise ValueError('context-only output crosses owner/version/space')
    output = _observation(declaration_reader, output_ref, output_source)
    expected_links = {proof_link, *links}
    if (set(output.event.payload.relation_occurrences) != expected_links
            or len(output.event.payload.relation_occurrences) != len(expected_links)):
        raise ValueError('context-only output loses the exact current proof and Memory links')
    for ref in used_refs:
        if (ref.owner != input_ref.owner or ref.versions != input_ref.versions
                or ref.memory_space != input_ref.memory_space):
            raise ValueError('context-only adoption crosses owner/version/space')
        _declaration(declaration_reader, ref)
    uses = _hashes(summary['use_event_hashes'], 'Use', max_memory_refs)
    outcomes = _hashes(summary['outcome_event_hashes'], 'UseOutcome', max_memory_refs)
    if len(uses) != len(used_refs) or len(outcomes) != len(used_refs):
        raise ValueError('context-only delivery does not cover every adopted Memory ref')
    units = generation['surface_values']
    if isinstance(units, dict):
        units = trace.resolve_ref(units['integer_key_ref'])
    units = tuple(units)
    if units != tuple(map(ord, report['response_surface'])):
        raise ValueError('context-only emitted units differ from the original natural output')
    context = frame_key(DELIVERY_PROOF_IDENTITY, (2,), proof_ref.stable_key(), input_source.stable_key())
    for ordinal, (ref, use_hash, outcome_hash) in enumerate(zip(used_refs, uses, outcomes)):
        use_row = _event(event_reader, use_hash, MEMORY_EVENT_USE)
        use = use_row.event.payload
        decision = frame_key(_DELIVERY_USE, proof_ref.stable_key(), (3, ordinal))
        if (not isinstance(use, UsePayload) or use.memory_ref != ref or use.episode_ref != episode_ref
                or use.influence_kind != act or use.query_kind != act or use.context_key != context
                or use.decision_trace_key != decision or use.outcome_ref is not None
                or use_row.event.scope != episode_row.event.scope):
            raise ValueError('context-only Use differs from the exact original adoption')
        outcome_row = _event(event_reader, outcome_hash, MEMORY_EVENT_USE_OUTCOME)
        outcome = outcome_row.event.payload
        if (not isinstance(outcome, UseOutcomePayload) or outcome.target_ref != use_row.event.object_ref
                or outcome.decision_trace_key != decision or outcome.query_kind != act
                or outcome.context_key != context or outcome.outcome_kind != act
                or outcome_row.event.scope != episode_row.event.scope
                or outcome.outcome_ref != MemoryLinkedRef.memory(output_ref)
                or outcome.outcome_trace_key != frame_key(_DELIVERY_USE, proof_ref.stable_key(), (3, ordinal), units)):
            raise ValueError('context-only UseOutcome differs from current output or complete emitted units')
    return {'status': 'MATCHED_CURRENT_CONTEXT_ONLY_DELIVERY_EVENTS', 'episode_count': 1,
            'used_memory_refs': len(used_refs), 'use_count': len(uses), 'outcome_count': len(outcomes),
            'physical_delivery': 1, 'factual_claims': 0, 'query_generation_replays': 0}
