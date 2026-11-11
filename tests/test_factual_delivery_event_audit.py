"""Reject semantically mismatched but otherwise valid durable Memory events."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion, CurriculumVersion, OBJECT_CONTEXT_SCOPE, ObjectIdentity,
    OwnerScope, ParserVersion, PrimitiveVersion, SourceRef, VersionBundle,
    VISIBILITY_SESSION,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_EPISODE, MEMORY_EVENT_OBSERVATION, MEMORY_EVENT_USE,
    MEMORY_EVENT_USE_OUTCOME, MEMORY_OBJECT_EPISODE, MEMORY_OBJECT_OBSERVATION,
    MEMORY_OBJECT_USE, EpisodePayload, MemoryEvent, MemoryLinkedRef,
    ObservationPayload, UseOutcomePayload, UsePayload, memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_event_log import MaterializedMemoryEvent
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_OBSERVED, CLOCK_MEMORY_TIMELINE, CLOCK_MEMORY_USED,
    LogicalClockIdentity, LogicalTimestamp, query_scope, session_scope,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import frame_key
from pure_integer_ai.experiments.factual_delivery_event_audit import (
    FactualDeliveryEventEvidenceError, verify_factual_delivery_events,
)
from pure_integer_ai.storage.spaces.registry import SPACE_TYPE_MEMORY, SpaceIdentity


def _fixture(memory_count=1) -> tuple[dict, tuple[int, ...], dict[int, MaterializedMemoryEvent]]:
    owner = OwnerScope(1, 2, 3, VISIBILITY_SESSION)
    versions = VersionBundle(CorpusVersion(1), ParserVersion(2),
                             PrimitiveVersion(3), CurriculumVersion(4))
    space = SpaceIdentity(SPACE_TYPE_MEMORY, 31, 32)
    source = SourceRef(91501, 10, 1, owner, versions)
    output_source = replace(source, document_id=2)
    session = session_scope(owner.session_id, owner=owner, versions=versions)
    scope = query_scope(source.document_id, parent=session, source=source)
    output_scope = query_scope(output_source.document_id, parent=session, source=output_source)

    def obj(value: int) -> ObjectIdentity:
        return ObjectIdentity(OBJECT_CONTEXT_SCOPE, (value,), owner, versions)

    def timestamp(kind: int, seq: int, *, where=scope) -> LogicalTimestamp:
        return LogicalTimestamp(LogicalClockIdentity(where, kind), seq)

    def row(event_hash: int, kind: int, object_kind: int, payload,
            *, where=scope) -> MaterializedMemoryEvent:
        object_key = payload.identity_key() if isinstance(payload, UsePayload) else payload.stable_key()
        ref = memory_object_ref(space, object_kind, object_key, owner=owner, versions=versions)
        event = MemoryEvent(kind, ref, where, payload)
        return MaterializedMemoryEvent(event_hash, event_hash + 100, event,
                                       timestamp(CLOCK_MEMORY_TIMELINE, event_hash, where=session))

    act, claim, role, locator = obj(33), obj(7), obj(1), obj(555)
    proof_ref = ObjectIdentity(OBJECT_CONTEXT_SCOPE, (91530, 999), owner, versions)
    act_link, proof_link = MemoryLinkedRef.object(act), MemoryLinkedRef.object(proof_ref)
    input_payload = ObservationPayload(source, act_link, (), (), None, (), (),
                                       timestamp(CLOCK_MEMORY_OBSERVED, 1))
    input_row = row(101, MEMORY_EVENT_OBSERVATION, MEMORY_OBJECT_OBSERVATION, input_payload)
    input_ref = input_row.event.object_ref
    extra_rows = tuple(row(1000 + index, MEMORY_EVENT_OBSERVATION, MEMORY_OBJECT_OBSERVATION,
                          replace(input_payload, source=replace(source, document_id=100 + index)))
                       for index in range(memory_count - 1))
    used_refs = tuple(sorted((input_ref, *(item.event.object_ref for item in extra_rows)),
                             key=lambda item: item.stable_key()))
    object_links = {MemoryLinkedRef.object(item) for item in (claim, role, locator)}
    memory_links = {MemoryLinkedRef.memory(ref) for ref in used_refs}
    output_payload = ObservationPayload(
        output_source, act_link, (), (), None, (),
        tuple(sorted(object_links | memory_links | {proof_link})),
        timestamp(CLOCK_MEMORY_OBSERVED, 1, where=output_scope))
    output_row = row(102, MEMORY_EVENT_OBSERVATION, MEMORY_OBJECT_OBSERVATION,
                     output_payload, where=output_scope)
    output_ref = output_row.event.object_ref
    episode_payload = EpisodePayload(
        input_ref, act_link, tuple(sorted(object_links | memory_links)), proof_link,
        MemoryLinkedRef.memory(output_ref), used_refs, (MemoryLinkedRef.object(locator),),
        None, output_source.document_id, session, timestamp(CLOCK_MEMORY_USED, 1))
    episode_row = row(103, MEMORY_EVENT_EPISODE, MEMORY_OBJECT_EPISODE, episode_payload)
    context = frame_key(91530, (4,), proof_ref.stable_key(), source.stable_key())
    use_rows, outcome_rows = [], []
    for ordinal, ref in enumerate(used_refs):
        decision = frame_key(91531, proof_ref.stable_key(), (4, ordinal))
        use_payload = UsePayload(ref, episode_row.event.object_ref, act_link, None,
                                 timestamp(CLOCK_MEMORY_USED, 2), decision, act_link, context)
        use_row = row(104 + 2 * ordinal, MEMORY_EVENT_USE, MEMORY_OBJECT_USE, use_payload)
        outcome_payload = UseOutcomePayload(
            use_row.event.object_ref, decision, act_link, context, act_link,
            MemoryLinkedRef.memory(output_ref), timestamp(CLOCK_MEMORY_OBSERVED, 2),
            frame_key(91531, proof_ref.stable_key(), (4, ordinal), (88,)))
        outcome_hash = 105 + 2 * ordinal
        outcome_row = MaterializedMemoryEvent(
            outcome_hash, use_row.object_hash,
            MemoryEvent(MEMORY_EVENT_USE_OUTCOME, use_row.event.object_ref, scope, outcome_payload),
            timestamp(CLOCK_MEMORY_TIMELINE, outcome_hash, where=session))
        use_rows.append(use_row)
        outcome_rows.append(outcome_row)
    plan = {
        "response_act": list(act.stable_key()), "claim_refs": [list(claim.stable_key())],
        "claim_evidence_refs": [{"claim": list(claim.stable_key()), "evidence": [[90]]}],
        "evidence_refs": [[90]], "memory_refs": [list(ref.stable_key()) for ref in used_refs],
        "slot_sequence": [{
            "role": list(role.stable_key()), "filler": list(claim.stable_key()),
            "required": 1, "source_hash": 12, "token_digest": [77],
            "allowed_node_kinds": [4],
        }],
    }
    summary = {
        "proof_ref": list(proof_ref.stable_key()), "output_source": list(output_source.stable_key()),
        "episode_event_hash": 103, "use_event_hashes": [row.event_hash for row in use_rows],
        "outcome_event_hashes": [row.event_hash for row in outcome_rows],
        "claim_count": 1, "evidence_ref_count": 1, "adopted_slots": 1,
        "delivered": 1, "fact_claims_added": 0,
    }
    captured = {
        "status": "CAPTURED_QUERY_PLAN_AND_DELIVERY_REFS",
        "delivery_proof_ref": summary["proof_ref"],
        "delivery_output_source": summary["output_source"], "delivery_episode_event_hash": 103,
        "input_source": list(source.stable_key()), "input_observation": list(input_ref.stable_key()),
        "expected_claim": list(claim.stable_key()), "output_values": [88],
        "trace": {"response_plan": plan}, "delivery_summary": summary,
    }
    key = claim.stable_key()
    ordered = frame_key(5, frame_key(1, key))
    claims = frame_key(1, key)
    audit = frame_key(5, frame_key(1, key, frame_key(1, (90,))))
    slots = frame_key(2, frame_key(1, role.stable_key(), key, (1, 12), (77,), (4,)))
    evidence = frame_key(3, locator.stable_key())
    proof = frame_key(91530, (4,), source.stable_key(), input_ref.stable_key(), act.stable_key(),
                      frame_key(4, key, ordered, claims, audit, slots, evidence))
    return captured, proof, {item.event_hash: item for item in (
        input_row, output_row, episode_row, *extra_rows, *use_rows, *outcome_rows)}


def _verify(captured: dict, proof: tuple[int, ...], events: dict, **budgets) -> dict:
    return verify_factual_delivery_events(
        captured, proof_reader=lambda ref: proof, evidence_reader=lambda ref: (90,),
        event_reader=events.get,
        declaration_reader=lambda ref: tuple(row for row in events.values()
                                             if row.event.is_declaration and row.event.object_ref == ref), **budgets)


def _replace_declaration(row: MaterializedMemoryEvent, payload) -> MaterializedMemoryEvent:
    ref = replace(row.event.object_ref, object_key=payload.stable_key())
    return replace(row, event=MemoryEvent(row.event.event_kind, ref, row.event.scope, payload))


def test_exact_durable_episode_use_and_output_units_match() -> None:
    captured, proof, events = _fixture()
    original = deepcopy(captured)
    result = _verify(captured, proof, events)
    assert result["status"] == "MATCHED_DURABLE_DELIVERY_EPISODE_AND_USES"
    assert result["used_memory_ref_count"] == 1
    assert result["recorded_delivery_units"] == "MATCHED_TO_CAPTURED_OUTPUT"
    assert result["ohe_exact_join"].startswith("UNKNOWN")
    assert result["host_delivery_qualification"].startswith("UNKNOWN")
    assert captured == original


@pytest.mark.parametrize("missing", [101, 102, 103, 104, 105])
def test_missing_observation_episode_use_or_outcome_fails(missing: int) -> None:
    captured, proof, events = _fixture()
    events.pop(missing)
    with pytest.raises(FactualDeliveryEventEvidenceError):
        _verify(captured, proof, events)


@pytest.mark.parametrize("field,value", [
    ("use_event_hashes", []), ("outcome_event_hashes", []),
    ("use_event_hashes", [104, 104]), ("episode_event_hash", True),
])
def test_incomplete_or_boolean_summary_cannot_adopt_memory(field: str, value: object) -> None:
    captured, proof, events = _fixture()
    captured["delivery_summary"][field] = value
    with pytest.raises(FactualDeliveryEventEvidenceError):
        _verify(captured, proof, events)


def test_valid_episode_with_another_proof_is_rejected() -> None:
    captured, proof, events = _fixture()
    row = events[103]
    original = row.event.payload.selected_path_ref.value()
    wrong = MemoryLinkedRef.object(replace(original, components=(91530, 1000)))
    events[103] = _replace_declaration(row, replace(row.event.payload, selected_path_ref=wrong))
    with pytest.raises(FactualDeliveryEventEvidenceError, match="Episode input/proof"):
        _verify(captured, proof, events)


def test_valid_output_observation_without_proof_link_is_rejected() -> None:
    captured, proof, events = _fixture()
    row = events[102]
    payload = row.event.payload
    links = tuple(ref for ref in payload.relation_occurrences
                  if not (ref.reference_kind == 3 and ref.value().components[0] == 91530))
    events[102] = _replace_declaration(row, replace(payload, relation_occurrences=links))
    episode = events[103]
    events[103] = _replace_declaration(episode, replace(
        episode.event.payload, output_ref=MemoryLinkedRef.memory(events[102].event.object_ref)))
    with pytest.raises(FactualDeliveryEventEvidenceError, match="output Observation lacks exact"):
        _verify(captured, proof, events)


def test_recorded_units_cannot_be_relabelled_as_another_output() -> None:
    captured, proof, events = _fixture()
    captured["output_values"] = [89]
    with pytest.raises(FactualDeliveryEventEvidenceError, match="UseOutcome output/units"):
        _verify(captured, proof, events)


def test_undeclared_or_unplanned_memory_cannot_be_adopted() -> None:
    captured, proof, events = _fixture()
    captured["trace"]["response_plan"]["memory_refs"] = []
    with pytest.raises(FactualDeliveryEventEvidenceError, match="outside captured plan"):
        _verify(captured, proof, events)


def test_explicit_roster_capacity_checks_all_original_episode_uses_and_outcomes():
    """默认门不变；显式 256 预算完整核验 130 个实际采用及后页 Outcome。"""
    captured, proof, events = _fixture(memory_count=130)
    with pytest.raises(FactualDeliveryEventEvidenceError, match="plan Memory refs.*budget"):
        _verify(captured, proof, events)
    result = _verify(captured, proof, events, max_memory_refs=256)
    assert result["used_memory_ref_count"] == 130
    assert len(result["use_event_hashes"]) == len(result["outcome_event_hashes"]) == 130
    events.pop(captured["delivery_summary"]["outcome_event_hashes"][-1])
    with pytest.raises(FactualDeliveryEventEvidenceError, match="event hash/kind"):
        _verify(captured, proof, events, max_memory_refs=256)


@pytest.mark.parametrize("value", [True, 0, -1, 4097, None])
def test_delivery_capacity_remains_explicit_and_bounded(value):
    """非严格或无界采用预算不能进入声明读取。"""
    captured, proof, events = _fixture()
    with pytest.raises(FactualDeliveryEventEvidenceError, match="Memory ref audit budget"):
        _verify(captured, proof, events, max_memory_refs=value)
