"""Read the Episode, Observations and Use events for one captured delivery.

Readers must be bound to the completed run's authorized, immutable Memory.
They use MemoryEventLog.read/query and perform no query generation or writes.
The audit qualifies durable delivery links only, never O/H/E closure or novelty.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_EPISODE, MEMORY_EVENT_OBSERVATION, MEMORY_EVENT_USE,
    MEMORY_EVENT_USE_OUTCOME, MEMORY_OBJECT_OBSERVATION,
    EpisodePayload, MemoryLinkedRef, MemoryObjectRef, ObservationPayload,
    UseOutcomePayload, UsePayload,
)
from pure_integer_ai.cognition.shared.memory_event_log import MaterializedMemoryEvent
from pure_integer_ai.cognition.understanding.query_memory_candidates import _parts, frame_key
from pure_integer_ai.experiments.factual_delivery_proof_audit import (
    verify_factual_delivery_proof,
)
from pure_integer_ai.experiments.trained_response_delivery import (
    DELIVERY_PROOF_IDENTITY, _DELIVERY_USE,
)


class FactualDeliveryEventEvidenceError(ValueError):
    """A required durable delivery link is absent or belongs to another call."""


EventReader = Callable[[int], MaterializedMemoryEvent | None]
DeclarationReader = Callable[[MemoryObjectRef], Sequence[MaterializedMemoryEvent]]


def _key(value: object, label: str) -> tuple[int, ...]:
    if (not isinstance(value, (list, tuple)) or not value or len(value) > 1_000_000
            or any(type(item) is not int or item < 0 for item in value)):
        raise FactualDeliveryEventEvidenceError(f"{label} requires a bounded integer key")
    return tuple(value)


def _hashes(value: object, label: str, maximum: int = 128) -> tuple[int, ...]:
    """核验本次显式预算内的完整 event hash 列表，拒绝重复和非严格整数。"""
    if (not isinstance(value, (list, tuple)) or len(value) > maximum
            or any(type(item) is not int or item <= 0 for item in value)
            or len(set(value)) != len(value)):
        raise FactualDeliveryEventEvidenceError(f"{label} requires unique bounded event hashes")
    return tuple(value)


def _event(reader: EventReader, event_hash: int, kind: int) -> MaterializedMemoryEvent:
    result = reader(event_hash)
    if (not isinstance(result, MaterializedMemoryEvent)
            or type(result.event_hash) is not int or result.event_hash != event_hash
            or result.event.event_kind != kind):
        raise FactualDeliveryEventEvidenceError("durable event hash/kind is missing or differs")
    return result


def _declaration(reader: DeclarationReader, ref: MemoryObjectRef) -> MaterializedMemoryEvent:
    rows = reader(ref)
    if not isinstance(rows, (tuple, list)) or len(rows) != 1:
        raise FactualDeliveryEventEvidenceError("Memory reference lacks one unique declaration")
    row = rows[0]
    if (not isinstance(row, MaterializedMemoryEvent)
            or type(row.event_hash) is not int or row.event_hash <= 0
            or not row.event.is_declaration or row.event.object_ref != ref):
        raise FactualDeliveryEventEvidenceError("Memory declaration identity differs")
    return row


def _observation(reader: DeclarationReader, ref: MemoryObjectRef,
                 source: SourceRef) -> MaterializedMemoryEvent:
    row = _declaration(reader, ref)
    if (ref.object_kind != MEMORY_OBJECT_OBSERVATION
            or row.event.event_kind != MEMORY_EVENT_OBSERVATION
            or not isinstance(row.event.payload, ObservationPayload)
            or row.event.payload.source != source):
        raise FactualDeliveryEventEvidenceError("Observation source/identity differs")
    return row


def verify_factual_delivery_events(
        provenance: Mapping[str, Any], *,
        proof_reader: Callable[[tuple[int, ...]], tuple[int, ...]],
        evidence_reader: Callable[[tuple[int, ...]], tuple[int, ...]],
        event_reader: EventReader, declaration_reader: DeclarationReader,
        max_memory_refs: int = 128,
        ) -> dict[str, Any]:
    """在显式采用引用预算内核对原物理事件，禁止跨调用采用或截断 Use 全集。"""
    if type(max_memory_refs) is not int or not 0 < max_memory_refs <= 4096:
        raise FactualDeliveryEventEvidenceError("Memory ref audit budget must be a strict integer from 1 to 4096")
    if not isinstance(provenance, Mapping):
        raise FactualDeliveryEventEvidenceError("captured provenance is missing")
    proof_key = _key(proof_reader(_key(provenance.get("delivery_proof_ref"),
                                    "proof ref")), "durable proof")
    # Keep one immutable proof read for all joins in this audit.
    proof_audit = verify_factual_delivery_proof(
        provenance, proof_reader=lambda ref: proof_key,
        evidence_reader=evidence_reader)
    summary = provenance.get("delivery_summary")
    if not isinstance(summary, Mapping):
        raise FactualDeliveryEventEvidenceError("delivery summary is missing")
    episode_hashes = _hashes((provenance.get("delivery_episode_event_hash"),), "Episode")
    episode_hash = episode_hashes[0]
    if (type(summary.get("episode_event_hash")) is not int
            or summary["episode_event_hash"] != episode_hash
            or summary.get("proof_ref") != provenance.get("delivery_proof_ref")
            or summary.get("output_source") != provenance.get("delivery_output_source")
            or type(summary.get("delivered")) is not int or summary["delivered"] != 1
            or type(summary.get("fact_claims_added")) is not int
            or summary["fact_claims_added"] != 0):
        raise FactualDeliveryEventEvidenceError("captured delivery references/counts differ")
    input_source = SourceRef.from_stable_key(_key(provenance.get("input_source"), "input source"))
    output_source = SourceRef.from_stable_key(_key(provenance.get("delivery_output_source"),
                                                "output source"))
    input_ref = MemoryObjectRef.from_stable_key(_key(provenance.get("input_observation"),
                                                  "input Observation"))
    proof_ref = ObjectIdentity.from_stable_key(_key(provenance.get("delivery_proof_ref"), "proof ref"))
    plan = provenance["trace"]["response_plan"]
    act = MemoryLinkedRef.object(ObjectIdentity.from_stable_key(_key(plan.get("response_act"), "act")))
    proof_link = MemoryLinkedRef.object(proof_ref)
    if (output_source == input_source or output_source.owner != input_source.owner
            or output_source.versions != input_source.versions
            or input_ref.owner != input_source.owner or input_ref.versions != input_source.versions
            or proof_ref.owner != input_source.owner or proof_ref.versions != input_source.versions):
        raise FactualDeliveryEventEvidenceError("delivery crosses input/output owner or version")
    input_row = _observation(declaration_reader, input_ref, input_source)
    episode_row = _event(event_reader, episode_hash, MEMORY_EVENT_EPISODE)
    episode_ref = episode_row.event.object_ref
    episode = episode_row.event.payload
    if (not isinstance(episode, EpisodePayload)
            or episode.input_observation_ref != input_ref
            or episode.selected_path_ref != proof_link or episode.intent_ref != act
            or episode.failure_ref is not None
            or episode_ref.owner != input_ref.owner or episode_ref.versions != input_ref.versions
            or episode_ref.memory_space != input_ref.memory_space
            or episode.session_ref.owner != input_ref.owner
            or episode.session_ref.versions != input_ref.versions
            or episode.session_ref.local_id != input_ref.owner.session_id
            or episode.round_seq != output_source.document_id
            or episode.output_ref is None):
        raise FactualDeliveryEventEvidenceError("Episode input/proof/act/session differs")
    output_refs = episode.output_ref.memory_refs()
    if len(output_refs) != 1:
        raise FactualDeliveryEventEvidenceError("Episode output is not one Observation")
    output_ref = output_refs[0]
    if (output_ref.owner != input_ref.owner or output_ref.versions != input_ref.versions
            or output_ref.memory_space != input_ref.memory_space):
        raise FactualDeliveryEventEvidenceError("output Observation crosses owner/version/space")
    output_row = _observation(declaration_reader, output_ref, output_source)
    outer = _parts(proof_key, DELIVERY_PROOF_IDENTITY)
    body = _parts(outer[4], 4)
    locators = _parts(body[5], 3)
    slot_refs = []
    for ordinal, record in enumerate(_parts(body[4], 2), 1):
        fields = _parts(record, ordinal)
        if len(fields) == 6:
            slot_refs.append(MemoryObjectRef.from_stable_key(fields[5]))
    planned_memory = plan.get("memory_refs")
    if not isinstance(planned_memory, list) or len(planned_memory) > max_memory_refs:
        raise FactualDeliveryEventEvidenceError("plan Memory refs are absent or exceed budget")
    planned_refs = tuple(MemoryObjectRef.from_stable_key(_key(ref, "plan Memory ref"))
                         for ref in planned_memory)
    if len(set(planned_refs)) != len(planned_refs):
        raise FactualDeliveryEventEvidenceError("plan Memory refs are duplicated")
    used_refs = tuple(sorted({
        *planned_refs,
        *slot_refs,
    }, key=lambda ref: ref.stable_key()))
    if len(used_refs) > max_memory_refs:
        raise FactualDeliveryEventEvidenceError("adopted Memory refs exceed explicit audit budget")
    if not used_refs or episode.used_memory_refs != used_refs:
        raise FactualDeliveryEventEvidenceError("Episode adopts Memory outside captured plan/proof")
    for ref in used_refs:
        if (ref.owner != input_ref.owner or ref.versions != input_ref.versions
                or ref.memory_space != input_ref.memory_space):
            raise FactualDeliveryEventEvidenceError("adopted Memory crosses owner/version/space")
        _declaration(declaration_reader, ref)
    signals = tuple(MemoryLinkedRef.object(ObjectIdentity.from_stable_key(key)) for key in locators)
    if episode.result_signal_refs != signals:
        raise FactualDeliveryEventEvidenceError("Episode evidence locators differ from proof")
    object_keys = {
        *(_key(ref, "plan claim") for ref in plan["claim_refs"]),
        *(_key(slot[name], "plan slot identity")
          for slot in plan["slot_sequence"] for name in ("role", "filler")),
        *locators,
    }
    object_links = {MemoryLinkedRef.object(ObjectIdentity.from_stable_key(key)) for key in object_keys}
    memory_links = {MemoryLinkedRef.memory(ref) for ref in used_refs}
    if (set(episode.candidate_refs) != object_links | memory_links
            or len(episode.candidate_refs) != len(object_links | memory_links)):
        raise FactualDeliveryEventEvidenceError("Episode candidate refs differ from plan")
    expected_links = object_links | memory_links | {proof_link, MemoryLinkedRef.memory(input_ref)}
    output_links = output_row.event.payload.relation_occurrences
    if set(output_links) != expected_links or len(output_links) != len(expected_links):
        raise FactualDeliveryEventEvidenceError("output Observation lacks exact proof/plan/Memory links")
    use_hashes = _hashes(summary.get("use_event_hashes"), "Use", max_memory_refs)
    outcome_hashes = _hashes(summary.get("outcome_event_hashes"), "UseOutcome", max_memory_refs)
    if len(use_hashes) != len(used_refs) or len(outcome_hashes) != len(used_refs):
        raise FactualDeliveryEventEvidenceError("Use/UseOutcome do not cover every adopted Memory ref")
    units = _key(provenance.get("output_values"), "delivered units")
    context = frame_key(DELIVERY_PROOF_IDENTITY, (4,), proof_ref.stable_key(), input_source.stable_key())
    for ordinal, (ref, use_hash, outcome_hash) in enumerate(zip(used_refs, use_hashes, outcome_hashes)):
        use_row = _event(event_reader, use_hash, MEMORY_EVENT_USE)
        use = use_row.event.payload
        decision = frame_key(_DELIVERY_USE, proof_ref.stable_key(), (4, ordinal))
        if (not isinstance(use, UsePayload) or use.memory_ref != ref or use.episode_ref != episode_ref
                or use.influence_kind != act or use.query_kind != act or use.context_key != context
                or use.decision_trace_key != decision or use.outcome_ref is not None
                or use_row.event.scope != episode_row.event.scope):
            raise FactualDeliveryEventEvidenceError("Use attribution differs from Episode/proof")
        outcome_row = _event(event_reader, outcome_hash, MEMORY_EVENT_USE_OUTCOME)
        outcome = outcome_row.event.payload
        if (not isinstance(outcome, UseOutcomePayload) or outcome.target_ref != use_row.event.object_ref
                or outcome.decision_trace_key != decision or outcome.query_kind != act
                or outcome.context_key != context or outcome.outcome_kind != act
                or outcome_row.event.scope != episode_row.event.scope
                or outcome.outcome_ref != MemoryLinkedRef.memory(output_ref)
                or outcome.outcome_trace_key != frame_key(
                    _DELIVERY_USE, proof_ref.stable_key(), (4, ordinal), units)):
            raise FactualDeliveryEventEvidenceError("UseOutcome output/units/attribution differs")
    return {
        "format": "PURE_INTEGER_FACTUAL_DELIVERY_EVENT_AUDIT_V1",
        "status": "MATCHED_DURABLE_DELIVERY_EPISODE_AND_USES",
        "proof_audit_status": proof_audit["status"],
        "input_observation_event_hash": input_row.event_hash,
        "output_observation_event_hash": output_row.event_hash,
        "episode_event_hash": episode_hash,
        "used_memory_ref_count": len(used_refs),
        "max_memory_refs": max_memory_refs,
        "use_event_hashes": list(use_hashes), "outcome_event_hashes": list(outcome_hashes),
        "recorded_delivery_units": "MATCHED_TO_CAPTURED_OUTPUT",
        "ohe_exact_join": "UNKNOWN_requires_Memory_declaration_and_QueryState_audit",
        "source_manifest_qualification": "UNKNOWN_requires_current_manifest_audit",
        "host_delivery_qualification": "UNKNOWN_requires_host_write_flush_receipt",
        "free_dialogue_complete": 0,
    }
