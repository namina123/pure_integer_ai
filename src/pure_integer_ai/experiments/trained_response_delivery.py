"""Attribute delivered graph responses to session Memory without replay or Core writes."""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_UNKNOWN
from pure_integer_ai.cognition.shared.generation_observed_surface import (
    DeliveredGraphSurface,
    ObservedGraphSurfaceProposal,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE, ObjectIdentity, SourceRef,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity, semantic_source,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_EPISODE, MEMORY_EVENT_OBSERVATION, MEMORY_OBJECT_EPISODE,
    MEMORY_OBJECT_OBSERVATION,
    EpisodePayload, MemoryEvent, MemoryLinkedRef, MemoryObjectRef, ObservationPayload,
    UsePayload, memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_event_log import MaterializedMemoryEvent
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.query_driver import unvisited_frontier
from pure_integer_ai.cognition.shared.query_state import ROOT_PENDING, QueryState
from pure_integer_ai.cognition.shared.response_plan import ResponsePlan
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_OBSERVED, CLOCK_MEMORY_USED, LogicalClockIdentity, query_scope, session_scope,
)
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft, ObservationIntakeDraft,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION, MEMORY_CATEGORY_RESPONSE, _parts, frame_key,
)
from pure_integer_ai.cognition.understanding.query_open_role_generation import OpenRoleGenerationContext
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record
from pure_integer_ai.experiments.memory_use_runtime import (
    append_memory_use, append_memory_use_outcome,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    _DIALOGUE_MEMORY_NAMESPACE, DialogueMemoryAppend, TrainedDialogueMemoryGraph,
)
from pure_integer_ai.experiments.trained_open_response_runtime import OpenResponseGeneration
from pure_integer_ai.experiments.trained_response_occurrence import (
    ResponseRoleOccurrence, materialize_response_roles,
)


DELIVERY_PROOF_IDENTITY = 91530
_DELIVERY_USE = 91531
FACTUAL_EVIDENCE_REF_IDENTITY = 91536


def _delivery_proof_identity(source: SourceRef, proof_hash: int) -> ObjectIdentity:
    """Build a source-valid ContextScope for an append-only delivery proof."""
    if type(proof_hash) is not int or proof_hash <= 0:
        raise ValueError("delivery proof hash must be a positive integer")
    return context_scope_identity(source, (DELIVERY_PROOF_IDENTITY, proof_hash))


def _delivery_proof_hash(memory: TrainedDialogueMemoryGraph,
                         ref: ObjectIdentity) -> int:
    """Recover the registry hash from either the current or legacy proof key."""
    if (not isinstance(ref, ObjectIdentity)
            or ref.object_kind != OBJECT_CONTEXT_SCOPE
            or ref.owner != memory.owner):
        raise ValueError("Delivery proof is not owned by this session")
    try:
        source = semantic_source(ref)
    except (TypeError, ValueError):
        source = None
    if source is not None:
        prefix = 1 + len(source.stable_key())
        components = ref.components
        if prefix < len(components):
            size = components[prefix]
            context = components[prefix + 1:prefix + 1 + size]
            if (type(size) is int and size == len(context)
                    and len(context) == 2
                    and context[0] == DELIVERY_PROOF_IDENTITY
                    and context[1] > 0):
                return context[1]
    # Preserve read compatibility with pre-fix package-local proof identities.
    if len(ref.components) == 2 and ref.components[0] == DELIVERY_PROOF_IDENTITY:
        return ref.components[1]
    raise ValueError("Delivery proof identity schema mismatch")


@dataclass(frozen=True, slots=True)
class ResponseDeliveryReceipt:
    """Exact persisted objects; hashes are lookup indexes, never semantic evidence."""

    proof_ref: ObjectIdentity
    output: DialogueMemoryAppend
    episode: MaterializedMemoryEvent
    uses: tuple[MaterializedMemoryEvent, ...]
    outcomes: tuple[MaterializedMemoryEvent, ...]
    slot_count: int
    observed_slot_count: int
    role_occurrences: tuple[ResponseRoleOccurrence, ...]
    input_context: OpenRoleGenerationContext

    def summary(self) -> dict[str, object]:
        return {
            "proof_ref": list(self.proof_ref.stable_key()),
            "output_source": list(self.output.source.stable_key()),
            "episode_event_hash": self.episode.event_hash,
            "use_event_hashes": [item.event_hash for item in self.uses],
            "outcome_event_hashes": [item.event_hash for item in self.outcomes],
            "adopted_slots": self.slot_count,
            "observed_slots": self.observed_slot_count,
            "role_occurrences": [list(item.stable_key()) for item in self.role_occurrences],
            "delivered": 1,
            "fact_claims_added": 0,
        }


@dataclass(frozen=True, slots=True)
class GenericResponseGeneration:
    """A selected graph response act with no Core proposition claim."""

    response_plan: ResponsePlan
    input_observation: MemoryObjectRef
    trace: tuple[int, ...]
    emitted_units: tuple[int, ...]
    graph_slots: tuple[tuple[int, ObservedGraphSurfaceProposal], ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.response_plan, ResponsePlan)
                or self.response_plan.claim_refs
                or not self.response_plan.memory_refs):
            raise ValueError("generic delivery requires a non-claim Memory plan")
        if (not isinstance(self.input_observation, MemoryObjectRef)
                or self.input_observation.object_kind != MEMORY_OBJECT_OBSERVATION):
            raise ValueError("generic delivery input must be an Observation")
        if (type(self.trace) is not tuple or not self.trace
                or any(type(value) is not int for value in self.trace)):
            raise ValueError("generic delivery trace must be integer-only")
        if self.emitted_units != tuple(map(ord, self.response_plan.surface())):
            raise ValueError("generic delivery units differ from ResponsePlan")
        if (type(self.graph_slots) is not tuple
                or any(type(item) is not tuple or len(item) != 2
                       or type(item[0]) is not int or item[0] not in range(1, 6)
                       or not isinstance(item[1], ObservedGraphSurfaceProposal)
                       for item in self.graph_slots)
                or len({(item[0], item[1].target) for item in self.graph_slots})
                != len(self.graph_slots)
                or any(item[1].observation != self.input_observation
                       for item in self.graph_slots)
                or any(item[1].target not in {
                    slot.filler for slot in self.response_plan.slot_sequence}
                       for item in self.graph_slots)):
            raise ValueError("generic delivery graph slots are incomplete")

    def stable_key(self) -> tuple[int, ...]:
        return frame_key(DELIVERY_PROOF_IDENTITY, (2,),
                         self.input_observation.stable_key(), self.trace,
                         self.response_plan.stable_key(), self.emitted_units,
                         pack_record(
                             2,
                             *(pack_record(
                                 1, (category,),
                                 DeliveredGraphSurface.from_proposal(
                                     proposal).stable_key())
                               for category, proposal in self.graph_slots),
                         ))


@dataclass(frozen=True, slots=True)
class FactualResponseGeneration:
    """A closed Core claim selected and rendered by the current QueryState."""

    response_plan: ResponsePlan
    input_observation: MemoryObjectRef
    proposition_key: tuple[int, ...]
    trace: tuple[int, ...]
    emitted_units: tuple[int, ...]

    def __post_init__(self) -> None:
        response = self.response_plan
        if (not isinstance(response, ResponsePlan) or not response.claim_refs
                or not response.memory_refs or not response.evidence_refs):
            raise ValueError("factual delivery requires a fully bound claim plan")
        if (not isinstance(self.input_observation, MemoryObjectRef)
                or self.input_observation.object_kind != MEMORY_OBJECT_OBSERVATION):
            raise ValueError("factual delivery input must be an Observation")
        if (type(self.proposition_key) is not tuple or not self.proposition_key
                or any(type(value) is not int for value in self.proposition_key)
                or self.proposition_key not in {
                    item.stable_key() for item in response.claim_refs}):
            raise ValueError("factual delivery proposition is not the selected claim")
        if (type(self.trace) is not tuple
                or any(type(value) is not int or value < 0 for value in self.trace)):
            raise ValueError("factual delivery trace must be integer-only")
        if self.emitted_units != tuple(map(ord, response.surface())):
            raise ValueError("factual delivery units differ from ResponsePlan")

    def stable_key(self) -> tuple[int, ...]:
        # Keep the in-process adoption key bounded.  Full evidence is
        # normalized once by record_factual_response_delivery below.
        slots = frame_key(2, *(frame_key(
            index + 1, item.role.stable_key(), item.filler.stable_key())
            for index, item in enumerate(self.response_plan.slot_sequence)))
        return frame_key(
            DELIVERY_PROOF_IDENTITY, (4,),
            self.input_observation.stable_key(), self.proposition_key,
            self.response_plan.response_act.stable_key(), slots, self.trace)


@dataclass(frozen=True, slots=True)
class GenericResponseDeliveryReceipt:
    """Persisted Episode/Use/Outcome for a generic non-factual response."""

    proof_ref: ObjectIdentity
    output: DialogueMemoryAppend
    episode: MaterializedMemoryEvent
    uses: tuple[MaterializedMemoryEvent, ...]
    outcomes: tuple[MaterializedMemoryEvent, ...]
    slot_count: int

    def summary(self) -> dict[str, object]:
        return {
            "proof_ref": list(self.proof_ref.stable_key()),
            "output_source": list(self.output.source.stable_key()),
            "episode_event_hash": self.episode.event_hash,
            "use_event_hashes": [item.event_hash for item in self.uses],
            "outcome_event_hashes": [item.event_hash for item in self.outcomes],
            "adopted_slots": self.slot_count,
            "observed_slots": 0,
            "role_occurrences": [],
            "delivered": 1,
            "fact_claims_added": 0,
        }


@dataclass(frozen=True, slots=True)
class FactualResponseDeliveryReceipt:
    """Persisted O/H/E, Episode and exact normalized factual plan members."""

    proof_ref: ObjectIdentity
    output: DialogueMemoryAppend
    episode: MaterializedMemoryEvent
    uses: tuple[MaterializedMemoryEvent, ...]
    outcomes: tuple[MaterializedMemoryEvent, ...]
    slot_count: int
    claim_count: int
    evidence_ref_count: int

    def summary(self) -> dict[str, object]:
        return {
            "proof_ref": list(self.proof_ref.stable_key()),
            "output_source": list(self.output.source.stable_key()),
            "episode_event_hash": self.episode.event_hash,
            "use_event_hashes": [item.event_hash for item in self.uses],
            "outcome_event_hashes": [item.event_hash for item in self.outcomes],
            "adopted_slots": self.slot_count,
            "observed_slots": self.slot_count,
            "claim_count": self.claim_count,
            "evidence_ref_count": self.evidence_ref_count,
            "role_occurrences": [],
            "delivered": 1,
            # Delivery observes a selected Core claim; it never creates one.
            "fact_claims_added": 0,
        }


def read_delivery_proof(memory: TrainedDialogueMemoryGraph,
                        ref: ObjectIdentity) -> tuple[int, ...]:
    """Recover every integer of the original query/generation/delivery proof."""
    proof_hash = _delivery_proof_hash(memory, ref)
    key = memory.intake.event_log.scoped_identities.registry.read_key(
        DELIVERY_PROOF_IDENTITY, proof_hash)
    fields = _parts(key, DELIVERY_PROOF_IDENTITY)
    if (len(fields) not in {5, 6}
            or fields[0] not in {(1,), (2,), (3,), (4,)}
            or (len(fields) == 6 and fields[0] != (3,))):
        raise ValueError("Delivery proof schema mismatch")
    source = SourceRef.from_stable_key(fields[1])
    if source.owner != ref.owner or source.versions != ref.versions:
        raise ValueError("Delivery proof source does not belong to its reference")
    return key


def read_factual_evidence_ref(
        memory: TrainedDialogueMemoryGraph,
        ref: ObjectIdentity,
        ) -> tuple[int, ...]:
    """Resolve one normalized QueryState Evidence reference exactly."""
    if (not isinstance(ref, ObjectIdentity)
            or ref.object_kind != OBJECT_CONTEXT_SCOPE
            or ref.owner != memory.owner or len(ref.components) != 2
            or ref.components[0] != FACTUAL_EVIDENCE_REF_IDENTITY):
        raise ValueError("factual evidence locator is not owned by this session")
    return memory.intake.event_log.scoped_identities.registry.read_key(
        FACTUAL_EVIDENCE_REF_IDENTITY, ref.components[1])


def record_factual_response_delivery(
        memory: TrainedDialogueMemoryGraph,
        state: QueryState,
        generation: FactualResponseGeneration,
        ) -> FactualResponseDeliveryReceipt:
    """Persist a bounded factual ResponsePlan after successful host delivery.

    The output surface remains a SourceRecord boundary observation.  Claim,
    role, filler and evidence identities come from the already completed
    three-graph ResponsePlan; no output text is reparsed into facts.
    """
    if (not isinstance(memory, TrainedDialogueMemoryGraph)
            or not isinstance(state, QueryState)
            or not isinstance(generation, FactualResponseGeneration)):
        raise TypeError("factual delivery requires typed Memory/query/generation")
    response = generation.response_plan
    if (state.termination != 1
            or any(root.status == ROOT_PENDING for root in state.roots)
            or {root.owner_space for root in state.roots} != {1, 2, 3}
            or state.best_candidate_key != generation.proposition_key
            or not response.claim_refs
            or not {item.stable_key() for item in state.evidence}.issubset(
                set(response.evidence_refs))):
        raise ValueError("factual delivery is not the closed current three-graph result")

    events = memory.intake.event_log
    access = MemoryAccessContext(
        memory.owner.tenant_id, memory.owner.user_id, memory.session_id)
    input_events = events.query(
        access=access, event_kind=MEMORY_EVENT_OBSERVATION,
        object_ref=generation.input_observation)
    if (len(input_events) != 1
            or not isinstance(input_events[0].event.payload, ObservationPayload)):
        raise ValueError("factual delivery input has no unique Observation")
    source = input_events[0].event.payload.source
    if source.owner != memory.owner:
        raise ValueError("factual delivery input belongs to another owner")
    memory.intake.require_current_manifest(source)

    used_refs = tuple(sorted(set(response.memory_refs),
                             key=lambda item: item.stable_key()))
    state_memory_keys = {
        key for item in state.evidence if item.space == 2
        for key in (item.hypothesis_key, item.evidence_key) if key
    } | {
        item.root_key for item in state.roots if item.owner_space == 2
    }
    if (not used_refs
            or any(ref.stable_key() not in state_memory_keys for ref in used_refs)):
        raise ValueError("factual delivery adopts Memory outside the current QueryState")
    for ref in used_refs:
        declarations = tuple(
            item for item in events.query(access=access, object_ref=ref)
            if item.event.is_declaration)
        if len(declarations) != 1 or declarations[0].event.object_ref != ref:
            raise ValueError("factual delivery Memory reference has no unique declaration")

    evidence_locators = []
    for evidence_key in response.evidence_refs:
        evidence_hash = events.scoped_identities.registry.register(
            FACTUAL_EVIDENCE_REF_IDENTITY, evidence_key)
        locator = ObjectIdentity(
            OBJECT_CONTEXT_SCOPE,
            (FACTUAL_EVIDENCE_REF_IDENTITY, evidence_hash),
            source.owner,
            source.versions,
        )
        if read_factual_evidence_ref(memory, locator) != evidence_key:
            raise RuntimeError("factual delivery evidence locator drifted")
        evidence_locators.append(locator)
    evidence_locators = tuple(evidence_locators)

    claim_record = frame_key(
        1, *(item.stable_key() for item in response.claim_refs))
    slot_record = frame_key(2, *(frame_key(
        ordinal + 1,
        slot.role.stable_key(),
        slot.filler.stable_key(),
        (1 if slot.required else 0, slot.source_hash),
        slot.token_digest,
        slot.allowed_node_kinds,
    ) for ordinal, slot in enumerate(response.slot_sequence)))
    evidence_record = frame_key(
        3, *(item.stable_key() for item in evidence_locators))
    proof = frame_key(
        DELIVERY_PROOF_IDENTITY,
        (4,),
        source.stable_key(),
        generation.input_observation.stable_key(),
        response.response_act.stable_key(),
        frame_key(
            4,
            generation.proposition_key,
            claim_record,
            slot_record,
            evidence_record,
        ),
    )
    proof_hash = events.scoped_identities.registry.register(
        DELIVERY_PROOF_IDENTITY, proof)
    proof_ref = _delivery_proof_identity(source, proof_hash)
    proof_link = MemoryLinkedRef.object(proof_ref)
    act_link = MemoryLinkedRef.object(response.response_act)
    object_links = tuple(MemoryLinkedRef.object(item) for item in sorted({
        *response.claim_refs,
        *(slot.role for slot in response.slot_sequence),
        *(slot.filler for slot in response.slot_sequence),
        *evidence_locators,
    }, key=lambda item: item.stable_key()))
    memory_links = tuple(MemoryLinkedRef.memory(item) for item in sorted({
        generation.input_observation, *used_refs,
    }, key=lambda item: item.stable_key()))
    candidate_key = frame_key(
        MEMORY_CANDIDATE_VERSION,
        (MEMORY_CATEGORY_RESPONSE,),
        proof_ref.stable_key(),
        response.response_act.stable_key(),
    )
    output_draft = ObservationIntakeDraft(
        frame_key(
            DELIVERY_PROOF_IDENTITY, (4,),
            source.stable_key(), proof_ref.stable_key()),
        MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_CONTEXT_SCOPE,
            (_DIALOGUE_MEMORY_NAMESPACE, memory.session_id, 2),
            memory.owner,
            source.versions,
        )),
        relation_occurrences=tuple(sorted({
            proof_link, *object_links, *memory_links,
        }, key=lambda item: item.stable_key())),
        hypotheses=(HypothesisIntakeDraft(
            candidate_key,
            (DELIVERY_PROOF_IDENTITY, MEMORY_CATEGORY_RESPONSE),
            candidate_key,
            frame_key(
                DELIVERY_PROOF_IDENTITY, (4,),
                generation.input_observation.stable_key(),
                generation.proposition_key),
            EVIDENCE_UNKNOWN,
            signal_ref=act_link,
            detail=frame_key(
                DELIVERY_PROOF_IDENTITY, (4,),
                proof_ref.stable_key(),
                (len(response.claim_refs), len(response.slot_sequence),
                 len(evidence_locators)),
            ),
        ),),
    )
    output = memory.append(
        "".join(map(chr, generation.emitted_units)),
        speaker_kind=2,
        stance=EVIDENCE_UNKNOWN,
        observation_draft=output_draft,
    )
    output_ref = memory.intake.result_for_source(output.source).observation_ref
    if output_ref is None:
        raise RuntimeError("factual delivery output Observation was not materialized")

    session = session_scope(
        memory.session_id, owner=memory.owner, versions=source.versions)
    scope = query_scope(
        source.document_id, parent=session, source=source)
    used_clock = events.scoped_identities.resume_clock(
        LogicalClockIdentity(scope, CLOCK_MEMORY_USED))
    observed_clock = events.scoped_identities.resume_clock(
        LogicalClockIdentity(scope, CLOCK_MEMORY_OBSERVED))
    candidate_links = tuple(sorted({
        *object_links,
        *(MemoryLinkedRef.memory(item) for item in used_refs),
    }, key=lambda item: item.stable_key()))
    episode_payload = EpisodePayload(
        generation.input_observation,
        act_link,
        candidate_links,
        proof_link,
        MemoryLinkedRef.memory(output_ref),
        used_refs,
        tuple(MemoryLinkedRef.object(item) for item in evidence_locators),
        None,
        output.turn_seq,
        session,
        used_clock.advance(),
    )
    episode_ref = memory_object_ref(
        events.memory_space_identity,
        MEMORY_OBJECT_EPISODE,
        episode_payload.stable_key(),
        owner=memory.owner,
        versions=source.versions,
    )
    episode = events.append(MemoryEvent(
        MEMORY_EVENT_EPISODE, episode_ref, scope, episode_payload))
    uses = []
    outcomes = []
    for ordinal, ref in enumerate(used_refs):
        locator = (4, ordinal)
        use = append_memory_use(events, scope, UsePayload(
            ref,
            episode_ref,
            act_link,
            None,
            used_clock.advance(),
            frame_key(_DELIVERY_USE, proof_ref.stable_key(), locator),
            act_link,
            frame_key(
                DELIVERY_PROOF_IDENTITY, (4,),
                proof_ref.stable_key(), source.stable_key()),
        ))
        outcome = append_memory_use_outcome(
            events,
            use.event.object_ref,
            scope=scope,
            outcome_kind=act_link,
            outcome_ref=MemoryLinkedRef.memory(output_ref),
            observed_at=observed_clock.advance(),
            outcome_trace_key=frame_key(
                _DELIVERY_USE, proof_ref.stable_key(), locator,
                generation.emitted_units),
        )
        uses.append(use)
        outcomes.append(outcome)
    memory.candidate_index.synchronize()
    memory.backend.commit()
    return FactualResponseDeliveryReceipt(
        proof_ref,
        output,
        episode,
        tuple(uses),
        tuple(outcomes),
        len(response.slot_sequence),
        len(response.claim_refs),
        len(evidence_locators),
    )


def record_open_response_delivery(
        memory: TrainedDialogueMemoryGraph, state: QueryState,
        generation: OpenResponseGeneration, emitted_units: tuple[int, ...],
        ) -> ResponseDeliveryReceipt:
    """Called only after successful output write/flush, for the selected current result.

    The complete proof is normalized in the existing integer registry, not
    replaced by its hash. Each Use carries an exact locator into that proof.
    No A-10 processing, M-07 resolution or Core R-01 learning is fabricated.
    """
    if not isinstance(generation, OpenResponseGeneration) or not isinstance(state, QueryState):
        raise TypeError("Delivery needs the original typed query and generation")
    if (type(emitted_units) is not tuple or any(type(value) is not int for value in emitted_units)
            or emitted_units != generation.rendered.units):
        raise ValueError("Delivered units differ from the completed graph generation")
    response = generation.response_plan
    context = generation.context
    preview = generation.preview
    selection = preview.request.structure.selection
    if (unvisited_frontier(state.frontier, state.visited)
            or any(root.status == ROOT_PENDING for root in state.roots)
            or {root.owner_space for root in state.roots} != {1, 2, 3}
            or response.claim_refs or context.source.owner != memory.owner
            or response.evidence_refs != tuple(item.stable_key() for item in state.evidence)
            or selection.request.goal.source != context.source):
        raise ValueError("Delivery is not the complete non-factual three-graph result")
    events = memory.intake.event_log
    access = MemoryAccessContext(memory.owner.tenant_id, memory.owner.user_id, memory.session_id)
    input_events = events.query(access=access, event_kind=MEMORY_EVENT_OBSERVATION,
                                object_ref=context.observation)
    if (len(input_events) != 1 or not isinstance(input_events[0].event.payload, ObservationPayload)
            or input_events[0].event.payload.source != context.source):
        raise ValueError("Delivery input does not resolve to its original Observation")
    memory.intake.require_current_manifest(context.source)
    used_refs = set(response.memory_refs)
    for candidate in selection.request.candidates:
        for evidence in candidate.observation_evidence:
            used_refs.update((evidence.observation, evidence.hypothesis))
    used_refs = tuple(sorted(used_refs, key=lambda item: item.stable_key()))
    for ref in used_refs:
        declarations = tuple(item for item in events.query(access=access, object_ref=ref)
                             if item.event.is_declaration)
        if len(declarations) != 1 or declarations[0].event.object_ref != ref:
            raise ValueError("An adopted Memory reference has no unique declaration")

    proof = frame_key(DELIVERY_PROOF_IDENTITY, (1,), context.source.stable_key(), state.stable_key(),
                      generation.stable_key(), emitted_units)
    proof_hash = events.scoped_identities.registry.register(DELIVERY_PROOF_IDENTITY, proof)
    proof_ref = _delivery_proof_identity(context.source, proof_hash)
    proof_link = MemoryLinkedRef.object(proof_ref)
    candidate_key = frame_key(MEMORY_CANDIDATE_VERSION, (MEMORY_CATEGORY_RESPONSE,),
                              proof_ref.stable_key(), response.response_act.stable_key())
    output_draft = ObservationIntakeDraft(
        frame_key(DELIVERY_PROOF_IDENTITY, context.source.stable_key(), proof_ref.stable_key()),
        MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_CONTEXT_SCOPE, (_DIALOGUE_MEMORY_NAMESPACE, memory.session_id, 2),
            memory.owner, context.source.versions)),
        relation_occurrences=(proof_link, *(MemoryLinkedRef.memory(ref) for ref in used_refs)),
        hypotheses=(HypothesisIntakeDraft(
            candidate_key, (DELIVERY_PROOF_IDENTITY, MEMORY_CATEGORY_RESPONSE),
            candidate_key, frame_key(DELIVERY_PROOF_IDENTITY, context.observation.stable_key()),
            EVIDENCE_UNKNOWN, signal_ref=MemoryLinkedRef.object(selection.stance),
            detail=frame_key(DELIVERY_PROOF_IDENTITY, proof_ref.stable_key(),
                             context.observation.stable_key())),),
    )
    output = memory.append("".join(map(chr, emitted_units)), speaker_kind=2,
                           stance=EVIDENCE_UNKNOWN, observation_draft=output_draft)
    output_result = memory.intake.result_for_source(output.source)
    output_ref = output_result.observation_ref
    session = session_scope(memory.session_id, owner=memory.owner, versions=context.source.versions)
    scope = query_scope(context.source.document_id, parent=session, source=context.source)
    used_clock = events.scoped_identities.resume_clock(LogicalClockIdentity(scope, CLOCK_MEMORY_USED))
    observed_clock = events.scoped_identities.resume_clock(
        LogicalClockIdentity(scope, CLOCK_MEMORY_OBSERVED))
    query_kind = MemoryLinkedRef.object(selection.request.goal.goal_kind)
    candidate_refs = tuple(sorted({
        MemoryLinkedRef.memory(evidence.hypothesis)
        for candidate in selection.request.candidates for evidence in candidate.observation_evidence
    }))
    episode_payload = EpisodePayload(
        context.observation, query_kind, candidate_refs, proof_link,
        MemoryLinkedRef.memory(output_ref), used_refs, (), None,
        output.turn_seq, session, used_clock.advance())
    episode_ref = memory_object_ref(events.memory_space_identity, MEMORY_OBJECT_EPISODE,
                                    episode_payload.stable_key(), owner=memory.owner,
                                    versions=context.source.versions)
    episode = events.append(MemoryEvent(MEMORY_EVENT_EPISODE, episode_ref, scope, episode_payload))
    role_occurrences = materialize_response_roles(memory, context, proof_ref, output.source)
    uses = []
    outcomes = []
    adoptions = [(ref, MemoryLinkedRef.object(selection.stance), (1, ordinal))
                 for ordinal, ref in enumerate(used_refs)]
    for ordinal, slot in enumerate(preview.slots):
        if slot.observed_surface is not None:
            adoptions.append((slot.observed_surface.value.observation,
                              MemoryLinkedRef.object(slot.directive.instruction), (2, ordinal)))
    for ref, influence, locator in adoptions:
        use = append_memory_use(events, scope, UsePayload(
            ref, episode_ref, influence, None, used_clock.advance(),
            frame_key(_DELIVERY_USE, proof_ref.stable_key(), locator), query_kind,
            frame_key(DELIVERY_PROOF_IDENTITY, proof_ref.stable_key(), context.source.stable_key())))
        # This is transport emission, not a factual support or training reward.
        outcome = append_memory_use_outcome(
            events, use.event.object_ref, scope=scope,
            outcome_kind=MemoryLinkedRef.object(preview.request.protocol.emit_action),
            outcome_ref=MemoryLinkedRef.memory(output_ref), observed_at=observed_clock.advance(),
            outcome_trace_key=frame_key(_DELIVERY_USE, proof_ref.stable_key(), locator, emitted_units))
        uses.append(use)
        outcomes.append(outcome)
    memory.candidate_index.synchronize()
    memory.backend.commit()
    return ResponseDeliveryReceipt(
        proof_ref, output, episode, tuple(uses), tuple(outcomes), len(preview.slots),
        sum(slot.observed_surface is not None for slot in preview.slots), role_occurrences, context)


def record_generic_response_delivery(
        memory: TrainedDialogueMemoryGraph, state: QueryState,
        generation: GenericResponseGeneration,
        ) -> GenericResponseDeliveryReceipt:
    """Adopt one current generic response without creating a Core fact."""
    if (not isinstance(memory, TrainedDialogueMemoryGraph)
            or not isinstance(state, QueryState)
            or not isinstance(generation, GenericResponseGeneration)):
        raise TypeError("generic delivery requires typed Memory/query/generation")
    response = generation.response_plan
    if (unvisited_frontier(state.frontier, state.visited)
            or any(root.status == ROOT_PENDING for root in state.roots)
            or {root.owner_space for root in state.roots} != {1, 2, 3}
            or response.claim_refs):
        raise ValueError("generic delivery is not a complete non-factual three-graph result")
    events = memory.intake.event_log
    access = MemoryAccessContext(
        memory.owner.tenant_id, memory.owner.user_id, memory.session_id)
    input_events = events.query(
        access=access, event_kind=MEMORY_EVENT_OBSERVATION,
        object_ref=generation.input_observation)
    if len(input_events) != 1 or not isinstance(
            input_events[0].event.payload, ObservationPayload):
        raise ValueError("generic delivery input has no unique Observation")
    source = input_events[0].event.payload.source
    if source.owner != memory.owner:
        raise ValueError("generic delivery input belongs to another owner")
    memory.intake.require_current_manifest(source)

    used_refs = tuple(sorted(set(response.memory_refs),
                             key=lambda item: item.stable_key()))
    state_memory_keys = {
        key for item in state.evidence if item.space == 2
        for key in (item.hypothesis_key, item.evidence_key) if key
    }
    if not used_refs or any(ref.stable_key() not in state_memory_keys for ref in used_refs):
        raise ValueError("generic delivery adopts Memory outside the current QueryState")
    for ref in used_refs:
        declarations = tuple(item for item in events.query(access=access, object_ref=ref)
                             if item.event.is_declaration)
        if len(declarations) != 1 or declarations[0].event.object_ref != ref:
            raise ValueError("generic delivery Memory reference has no unique declaration")

    # Keep the proof complete for delivery validation without recursively
    # embedding the full QueryState and planner trace.  Those are already
    # represented by the current O/H/E references and the shared query trace;
    # copying them here made append-only identity_part grow quadratically on
    # every subsequent turn.
    graph_slot_record = pack_record(
        2,
        *(pack_record(
            1,
            (category,),
            DeliveredGraphSurface.from_proposal(proposal).stable_key())
          for category, proposal in generation.graph_slots),
    )
    proof = frame_key(
        DELIVERY_PROOF_IDENTITY, (3,), source.stable_key(),
        generation.input_observation.stable_key(),
        response.response_act.stable_key(),
        frame_key(1, *(ref.stable_key() for ref in used_refs)),
        graph_slot_record)
    proof_hash = events.scoped_identities.registry.register(
        DELIVERY_PROOF_IDENTITY, proof)
    proof_ref = _delivery_proof_identity(source, proof_hash)
    proof_link = MemoryLinkedRef.object(proof_ref)
    act_link = MemoryLinkedRef.object(response.response_act)
    candidate_key = frame_key(
        MEMORY_CANDIDATE_VERSION, (MEMORY_CATEGORY_RESPONSE,),
        proof_ref.stable_key(), response.response_act.stable_key())
    output_draft = ObservationIntakeDraft(
        frame_key(DELIVERY_PROOF_IDENTITY, (2,), source.stable_key(),
                  proof_ref.stable_key()),
        MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_CONTEXT_SCOPE, (_DIALOGUE_MEMORY_NAMESPACE, memory.session_id, 2),
            memory.owner, source.versions)),
        relation_occurrences=(
            proof_link,
            *(MemoryLinkedRef.memory(ref) for ref in used_refs),
            *(MemoryLinkedRef.object(proposal.target)
              for _category, proposal in generation.graph_slots)),
        hypotheses=(HypothesisIntakeDraft(
            candidate_key, (DELIVERY_PROOF_IDENTITY, MEMORY_CATEGORY_RESPONSE),
            candidate_key,
            frame_key(DELIVERY_PROOF_IDENTITY, (2,),
                      generation.input_observation.stable_key()),
            EVIDENCE_UNKNOWN, signal_ref=act_link,
            # Trace is deliberately bounded: full planner/state traces remain
            # in the response envelope, never inside the next Memory key.
            detail=frame_key(DELIVERY_PROOF_IDENTITY, (3,),
                             proof_ref.stable_key(), generation.input_observation.stable_key(),
                             response.response_act.stable_key())),),
    )
    output = memory.append(
        "".join(map(chr, generation.emitted_units)), speaker_kind=2,
        stance=EVIDENCE_UNKNOWN, observation_draft=output_draft)
    output_ref = memory.intake.result_for_source(output.source).observation_ref
    if output_ref is None:
        raise RuntimeError("generic delivery output Observation was not materialized")
    session = session_scope(memory.session_id, owner=memory.owner,
                            versions=source.versions)
    scope = query_scope(source.document_id, parent=session, source=source)
    used_clock = events.scoped_identities.resume_clock(
        LogicalClockIdentity(scope, CLOCK_MEMORY_USED))
    observed_clock = events.scoped_identities.resume_clock(
        LogicalClockIdentity(scope, CLOCK_MEMORY_OBSERVED))
    episode_payload = EpisodePayload(
        generation.input_observation, act_link,
        tuple(MemoryLinkedRef.memory(ref) for ref in used_refs), proof_link,
        MemoryLinkedRef.memory(output_ref), used_refs, (), None,
        output.turn_seq, session, used_clock.advance())
    episode_ref = memory_object_ref(
        events.memory_space_identity, MEMORY_OBJECT_EPISODE,
        episode_payload.stable_key(), owner=memory.owner, versions=source.versions)
    episode = events.append(MemoryEvent(
        MEMORY_EVENT_EPISODE, episode_ref, scope, episode_payload))
    uses = []
    outcomes = []
    for ordinal, ref in enumerate(used_refs):
        locator = (3, ordinal)
        use = append_memory_use(events, scope, UsePayload(
            ref, episode_ref, act_link, None, used_clock.advance(),
            frame_key(_DELIVERY_USE, proof_ref.stable_key(), locator), act_link,
            frame_key(DELIVERY_PROOF_IDENTITY, (2,), proof_ref.stable_key(),
                      source.stable_key())))
        outcome = append_memory_use_outcome(
            events, use.event.object_ref, scope=scope, outcome_kind=act_link,
            outcome_ref=MemoryLinkedRef.memory(output_ref),
            observed_at=observed_clock.advance(),
            outcome_trace_key=frame_key(
                _DELIVERY_USE, proof_ref.stable_key(), locator,
                generation.emitted_units))
        uses.append(use)
        outcomes.append(outcome)
    memory.candidate_index.synchronize()
    memory.backend.commit()
    return GenericResponseDeliveryReceipt(
        proof_ref, output, episode, tuple(uses), tuple(outcomes),
        len(response.slot_sequence))
