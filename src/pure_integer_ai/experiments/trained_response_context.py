"""Recover delivered dialogue roles from Memory edges, not from response text."""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.generation_observed_surface import (
    DeliveredGraphSurface,
    ObservedGraphSurfaceProposal,
    delivered_graph_surface_role,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_EPISODE, MEMORY_EVENT_OBSERVATION, MEMORY_OBJECT_OBSERVATION,
    EpisodePayload, MemoryObjectRef, ObservationPayload,
)
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_PENDING,
    BindingEntry,
    EvidenceEntry,
    FrontierEntry,
    QueryAnchor,
    QueryRoot,
    QueryState,
    SPACE_DIALOGUE,
    SPACE_MEMORY,
    VisitedKey,
)
from pure_integer_ai.cognition.shared.representation_rendering import representation_parts
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION, MEMORY_CATEGORY_RESPONSE, memory_candidate_category,
)
from pure_integer_ai.cognition.understanding.query_open_role_generation import OpenRoleGenerationContext
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record
from pure_integer_ai.experiments.trained_response_delivery import (
    DELIVERY_PROOF_IDENTITY,
    FACTUAL_EVIDENCE_REF_IDENTITY,
    read_delivery_proof,
    read_factual_evidence_ref,
)
from pure_integer_ai.experiments.trained_response_occurrence import ResponseRoleOccurrence, read_response_roles


RESPONSE_CONTEXT_VERSION = 91532
RESPONSE_CONTEXT_ROLE = (RESPONSE_CONTEXT_VERSION, 1)


@dataclass(frozen=True, slots=True)
class DeliveredGraphSlotProjection:
    """One acknowledged semantic slot projected through Memory O/H/E."""

    category: int
    target: ObjectIdentity
    observation: MemoryObjectRef
    hypothesis: MemoryObjectRef
    proof_ref: ObjectIdentity
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    context_depth: int

    def __post_init__(self) -> None:
        if (type(self.category) is not int or self.category not in range(1, 6)
                or not isinstance(self.target, ObjectIdentity)
                or not isinstance(self.observation, MemoryObjectRef)
                or not isinstance(self.hypothesis, MemoryObjectRef)
                or not isinstance(self.proof_ref, ObjectIdentity)
                or type(self.source_ref) is not tuple or not self.source_ref
                or type(self.scope_key) is not tuple or not self.scope_key
                or type(self.evidence_keys) is not tuple
                or not self.evidence_keys
                or type(self.context_depth) is not int
                or self.context_depth < 0):
            raise ValueError("delivered graph-slot projection is incomplete")

    def stable_key(self) -> tuple[int, ...]:
        return pack_record(
            91572,
            (self.category,),
            self.target.stable_key(),
            self.proof_ref.stable_key(),
            self.source_ref,
            self.scope_key,
            (self.context_depth,),
        )

    def query_binding(self) -> BindingEntry:
        return BindingEntry(
            delivered_graph_surface_role(self.category, self.proof_ref),
            self.target.stable_key(),
            space=SPACE_MEMORY,
            scope_key=self.scope_key,
        )

    def query_evidence(self) -> EvidenceEntry:
        return EvidenceEntry(
            self.source_ref,
            self.hypothesis.stable_key(),
            space=SPACE_MEMORY,
            polarity=3,
            trust=1,
            evidence_key=self.stable_key(),
            scope_key=self.scope_key,
            payload_key=pack_record(
                91572,
                self.proof_ref.stable_key(),
                self.target.stable_key(),
            ),
        )

    def seed(self, weight: int):
        key = self.stable_key()
        return (
            QueryRoot(
                SPACE_MEMORY,
                key,
                ROOT_PENDING,
                weight,
                source_ref=self.source_ref,
                scope_key=self.scope_key,
            ),
            QueryAnchor(
                SPACE_MEMORY,
                key,
                kind=4,
                support=len(self.evidence_keys),
                scope_key=self.scope_key,
            ),
            FrontierEntry(
                (SPACE_MEMORY, *key),
                SPACE_MEMORY,
                target_key=key,
                root_key=key,
                source_ref=self.source_ref,
                scope_key=self.scope_key,
                owner_weight=weight,
                required_slot_gain=1,
                evidence_support=len(self.evidence_keys),
                discourse_fit=2 if self.context_depth <= 1 else 1,
                recency_weight=1 if self.context_depth <= 1 else 0,
            ),
        )

    def expand(self, state: QueryState, edge: FrontierEntry) -> QueryState:
        key = self.stable_key()
        if edge.owner_space != SPACE_MEMORY or edge.target_key != key:
            raise ValueError("delivered graph-slot frontier differs")
        return state.with_(
            frontier=tuple(item for item in state.frontier if item != edge),
            depth=max(state.depth, edge.depth + 1),
            bindings=tuple(sorted(
                {*state.bindings, self.query_binding()},
                key=lambda item: item.stable_key(),
            )),
            evidence=tuple(sorted(
                {*state.evidence, self.query_evidence()},
                key=lambda item: item.stable_key(),
            )),
            visited=tuple(sorted(
                {*state.visited,
                 VisitedKey(SPACE_MEMORY, key, direction=edge.direction)},
                key=lambda item: item.stable_key(),
            )),
            node_count=state.node_count + 1,
            edge_count=state.edge_count + 1,
            read_count=state.read_count + 1,
        )


def delivered_graph_slot_projections(contexts) -> tuple[
        DeliveredGraphSlotProjection, ...]:
    """Normalize factual and generic deliveries into one Memory projection."""
    category_by_kind = {
        16: 1,
        17: 2,
        7: 4,
        4: 5,
    }
    result = []
    for context in contexts:
        if isinstance(context, DeliveredFactualResponseContext):
            members = tuple(
                (category_by_kind[slot.filler.object_kind], slot.filler)
                for slot in context.slots
                if slot.filler.object_kind in category_by_kind
            )
        elif isinstance(context, DeliveredGenericResponseContext):
            members = tuple(
                (category, proposal.target)
                for category, proposal in context.graph_surfaces
            )
        else:
            members = ()
        result.extend(
            DeliveredGraphSlotProjection(
                category,
                target,
                context.observation,
                context.hypothesis,
                context.proof_ref,
                context.source_ref,
                context.scope_key,
                context.evidence_keys,
                context.context_depth,
            )
            for category, target in members
        )
    return tuple(sorted(
        {item.stable_key(): item for item in result}.values(),
        key=lambda item: item.stable_key(),
    ))


@dataclass(frozen=True, slots=True)
class DeliveredResponseContext:
    """A prior dialogue branch and its observed roles, not adopted coreference."""

    observation: MemoryObjectRef
    hypothesis: MemoryObjectRef
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    proof_ref: ObjectIdentity
    context: OpenRoleGenerationContext
    response_act: ObjectIdentity
    connector: ObjectIdentity
    representations: tuple[ObjectIdentity, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    context_depth: int
    role_occurrences: tuple[ResponseRoleOccurrence, ...] = ()

    def stable_key(self) -> tuple[int, ...]:
        original = pack_record(
            RESPONSE_CONTEXT_VERSION, self.observation.stable_key(), self.hypothesis.stable_key(),
            self.source_ref, self.scope_key, self.proof_ref.stable_key(), self.context.stable_key(),
            self.response_act.stable_key(), self.connector.stable_key(),
            pack_record(1, *(item.stable_key() for item in self.representations)),
            pack_record(1, *self.evidence_keys), (self.context_depth,))
        if not self.role_occurrences:
            return original
        return pack_record(91535, original,
                           pack_record(1, *(item.stable_key() for item in self.role_occurrences)))

    def query_bindings(self) -> tuple[BindingEntry, ...]:
        return (
            BindingEntry(RESPONSE_CONTEXT_ROLE, self.stable_key(), space=SPACE_DIALOGUE,
                         scope_key=self.scope_key),
            *(BindingEntry(pack_record(RESPONSE_CONTEXT_VERSION, (2,), role.role.stable_key(),
                                      self.proof_ref.stable_key()), role.origin.stable_key(),
                           space=SPACE_DIALOGUE, scope_key=self.context.scope.stable_key())
              for role in self.context.observed_spans()),
        )

    def discourse_objects(self) -> tuple[ObjectIdentity, ...]:
        return (self.proof_ref, *(span.origin for span in self.context.observed_spans()),
                *(item.occurrence for item in self.role_occurrences))


@dataclass(frozen=True, slots=True)
class DeliveredGenericResponseContext:
    """A delivered non-propositional act available to later discourse queries."""

    observation: MemoryObjectRef
    hypothesis: MemoryObjectRef
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    proof_ref: ObjectIdentity
    input_observation: MemoryObjectRef
    response_act: ObjectIdentity
    evidence_keys: tuple[tuple[int, ...], ...]
    context_depth: int
    role_occurrences: tuple[ResponseRoleOccurrence, ...] = ()
    graph_surfaces: tuple[tuple[int, DeliveredGraphSurface], ...] = ()

    def stable_key(self) -> tuple[int, ...]:
        original = pack_record(
            RESPONSE_CONTEXT_VERSION, (2,), self.observation.stable_key(),
            self.hypothesis.stable_key(), self.source_ref, self.scope_key,
            self.proof_ref.stable_key(), self.input_observation.stable_key(),
            self.response_act.stable_key(), pack_record(1, *self.evidence_keys),
            (self.context_depth,))
        if not self.graph_surfaces:
            return original
        return pack_record(
            91573,
            original,
            pack_record(
                2,
                *(pack_record(1, (category,), proposal.stable_key())
                  for category, proposal in self.graph_surfaces),
            ),
        )

    def query_bindings(self) -> tuple[BindingEntry, ...]:
        return (
            BindingEntry(
                RESPONSE_CONTEXT_ROLE,
                self.stable_key(),
                space=SPACE_DIALOGUE,
                scope_key=self.scope_key,
            ),
        )

    def discourse_objects(self) -> tuple[ObjectIdentity, ...]:
        return tuple(sorted({
            self.proof_ref,
            self.response_act,
            *(proposal.target for _category, proposal in self.graph_surfaces),
        }, key=lambda item: item.stable_key()))


@dataclass(frozen=True, slots=True)
class DeliveredFactualSlot:
    """One exact role/filler member recovered from a factual delivery proof."""

    role: ObjectIdentity
    filler: ObjectIdentity
    required: int
    source_hash: int
    token_digest: tuple[int, ...]
    allowed_node_kinds: tuple[int, ...]

    def stable_key(self) -> tuple[int, ...]:
        return pack_record(
            RESPONSE_CONTEXT_VERSION,
            (3,),
            self.role.stable_key(),
            self.filler.stable_key(),
            (self.required, self.source_hash),
            self.token_digest,
            self.allowed_node_kinds,
        )


@dataclass(frozen=True, slots=True)
class DeliveredFactualResponseContext:
    """A prior closed claim and its ResponsePlan members, never response text."""

    observation: MemoryObjectRef
    hypothesis: MemoryObjectRef
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    proof_ref: ObjectIdentity
    input_observation: MemoryObjectRef
    response_act: ObjectIdentity
    proposition_key: tuple[int, ...]
    claim_refs: tuple[ObjectIdentity, ...]
    slots: tuple[DeliveredFactualSlot, ...]
    evidence_locators: tuple[ObjectIdentity, ...]
    evidence_refs: tuple[tuple[int, ...], ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    context_depth: int
    role_occurrences: tuple[ResponseRoleOccurrence, ...] = ()

    def stable_key(self) -> tuple[int, ...]:
        return pack_record(
            RESPONSE_CONTEXT_VERSION,
            (4,),
            self.observation.stable_key(),
            self.hypothesis.stable_key(),
            self.source_ref,
            self.scope_key,
            self.proof_ref.stable_key(),
            self.input_observation.stable_key(),
            self.response_act.stable_key(),
            self.proposition_key,
            pack_record(1, *(item.stable_key() for item in self.claim_refs)),
            pack_record(2, *(item.stable_key() for item in self.slots)),
            pack_record(3, *(item.stable_key()
                             for item in self.evidence_locators)),
            pack_record(4, *self.evidence_keys),
            (self.context_depth,),
        )

    def query_bindings(self) -> tuple[BindingEntry, ...]:
        return (
            BindingEntry(
                RESPONSE_CONTEXT_ROLE,
                self.stable_key(),
                space=SPACE_DIALOGUE,
                scope_key=self.scope_key,
            ),
            *(BindingEntry(
                pack_record(
                    RESPONSE_CONTEXT_VERSION,
                    (4,),
                    self.proof_ref.stable_key(),
                    slot.role.stable_key(),
                ),
                slot.filler.stable_key(),
                space=SPACE_DIALOGUE,
                scope_key=self.scope_key,
            ) for slot in self.slots),
        )

    def discourse_objects(self) -> tuple[ObjectIdentity, ...]:
        return tuple(sorted({
            self.proof_ref,
            self.response_act,
            *self.claim_refs,
            *(item.role for item in self.slots),
            *(item.filler for item in self.slots),
            *self.evidence_locators,
        }, key=lambda item: item.stable_key()))


def restore_response_contexts(memory, structures) -> tuple[
        DeliveredResponseContext | DeliveredGenericResponseContext |
        DeliveredFactualResponseContext, ...]:
    """Follow only recovered response hypotheses and verify their actual delivery Episodes."""
    hypotheses = tuple(item for item in structures.hypotheses
                       if item.candidate_key[0] == MEMORY_CANDIDATE_VERSION
                       and memory_candidate_category(item.candidate_key) == MEMORY_CATEGORY_RESPONSE)
    if not hypotheses:
        return ()
    observations = {item.observation_key: item for item in structures.observations}
    episodes = {}
    for event in memory.intake.event_log.query(
            access=memory.candidate_index.access, event_kind=MEMORY_EVENT_EPISODE):
        payload = event.event.payload
        if not isinstance(payload, EpisodePayload) or payload.output_ref is None:
            continue
        output = payload.output_ref.value()
        if isinstance(output, MemoryObjectRef):
            episodes.setdefault(output.stable_key(), []).append(payload)
    result = []
    for hypothesis in hypotheses:
        observation = observations[hypothesis.observation_key]
        fields = unpack_record(hypothesis.candidate_key, MEMORY_CANDIDATE_VERSION)
        if len(fields) != 3:
            raise ValueError("Delivered response candidate schema mismatch")
        proof_ref = ObjectIdentity.from_stable_key(fields[1])
        response_act = ObjectIdentity.from_stable_key(fields[2])
        proof = unpack_record(read_delivery_proof(memory, proof_ref), DELIVERY_PROOF_IDENTITY)
        declarations = episodes.get(observation.observation_key, ())
        if proof[0] == (4,):
            proof_source = SourceRef.from_stable_key(proof[1])
            input_observation = MemoryObjectRef.from_stable_key(proof[2])
            proof_act = ObjectIdentity.from_stable_key(proof[3])
            factual = unpack_record(proof[4], 4)
            if len(factual) != 4:
                raise ValueError("Bounded factual response proof is incomplete")
            proposition_key = factual[0]
            claim_refs = tuple(ObjectIdentity.from_stable_key(item)
                               for item in unpack_record(factual[1], 1))
            slots = []
            for ordinal, slot_key in enumerate(unpack_record(factual[2], 2), 1):
                if not slot_key or slot_key[0] != ordinal:
                    raise ValueError("Bounded factual response slots are not ordered")
                slot = unpack_record(slot_key, ordinal)
                if (len(slot) != 5 or len(slot[2]) != 2
                        or slot[2][0] not in {0, 1}
                        or slot[2][1] <= 0 or not slot[3] or not slot[4]):
                    raise ValueError("Bounded factual response slot is incomplete")
                role = ObjectIdentity.from_stable_key(slot[0])
                filler = ObjectIdentity.from_stable_key(slot[1])
                if filler.object_kind not in slot[4]:
                    raise ValueError("Bounded factual filler kind is not allowed")
                slots.append(DeliveredFactualSlot(
                    role, filler, slot[2][0], slot[2][1], slot[3], slot[4]))
            evidence_locators = tuple(
                ObjectIdentity.from_stable_key(item)
                for item in unpack_record(factual[3], 3))
            evidence_refs = tuple(
                read_factual_evidence_ref(memory, item)
                for item in evidence_locators)
            input_events = memory.intake.event_log.query(
                access=memory.candidate_index.access,
                event_kind=MEMORY_EVENT_OBSERVATION,
                object_ref=input_observation,
            )
            relation_objects = {
                link.value() for link in observation.relation_refs}
            expected_objects = {
                proof_ref,
                input_observation,
                *claim_refs,
                *(item.role for item in slots),
                *(item.filler for item in slots),
                *evidence_locators,
            }
            if (not claim_refs or proposition_key not in {
                    item.stable_key() for item in claim_refs}
                    or not slots or not evidence_refs
                    or any(item.components[0] != FACTUAL_EVIDENCE_REF_IDENTITY
                           for item in evidence_locators)
                    or len(input_events) != 1
                    or not isinstance(input_events[0].event.payload,
                                      ObservationPayload)
                    or input_events[0].event.payload.source != proof_source
                    or proof_act != response_act
                    or not expected_objects <= relation_objects
                    or len(declarations) != 1
                    or declarations[0].selected_path_ref is None
                    or declarations[0].selected_path_ref.value() != proof_ref
                    or declarations[0].input_observation_ref != input_observation):
                raise ValueError(
                    "Bounded factual response context has no complete delivery proof")
            result.append(DeliveredFactualResponseContext(
                MemoryObjectRef.from_stable_key(observation.observation_key),
                MemoryObjectRef.from_stable_key(hypothesis.hypothesis_key),
                observation.source_ref,
                observation.scope_key,
                proof_ref,
                input_observation,
                response_act,
                proposition_key,
                claim_refs,
                tuple(slots),
                evidence_locators,
                evidence_refs,
                hypothesis.evidence_keys,
                observation.context_depth,
            ))
            continue
        if proof[0] in {(2,), (3,)}:
            # Kind 3 is the bounded generic-delivery proof.  Unlike the
            # legacy kind 2 record it carries only source, input Observation,
            # response act and the exact Memory references; the full planner
            # trace is kept in the current query envelope, not recursively in
            # append-only identity storage.
            if proof[0] == (3,):
                if len(proof) not in {5, 6}:
                    raise ValueError("Bounded generic response proof is incomplete")
                proof_source = SourceRef.from_stable_key(proof[1])
                input_observation = MemoryObjectRef.from_stable_key(proof[2])
                proof_act = ObjectIdentity.from_stable_key(proof[3])
                input_events = memory.intake.event_log.query(
                    access=memory.candidate_index.access,
                    event_kind=MEMORY_EVENT_OBSERVATION,
                    object_ref=input_observation,
                )
                graph_surfaces = ()
                if len(proof) == 6:
                    slots = unpack_record(proof[5], 2)
                    restored = []
                    for slot in slots:
                        fields = unpack_record(slot, 1)
                        if (len(fields) != 2 or len(fields[0]) != 1
                                or fields[0][0] not in range(1, 6)):
                            raise ValueError(
                                "Bounded generic graph slot is incomplete")
                        restored.append((
                            fields[0][0],
                            DeliveredGraphSurface.from_stable_key(
                                fields[1]),
                        ))
                    graph_surfaces = tuple(sorted(
                        restored,
                        key=lambda item: (item[0], item[1].stable_key()),
                    ))
                    if (len({(category, proposal.target)
                             for category, proposal in graph_surfaces})
                            != len(graph_surfaces)
                            ):
                        raise ValueError(
                            "Bounded generic graph slots conflict")
                relation_objects = {
                    link.value() for link in observation.relation_refs}
                expected_targets = {
                    proposal.target
                    for _category, proposal in graph_surfaces}
                if (len(input_events) != 1
                        or not isinstance(input_events[0].event.payload,
                                          ObservationPayload)
                        or input_events[0].event.payload.source != proof_source
                        or proof_act.stable_key() != response_act.stable_key()
                        or input_observation.object_kind != MEMORY_OBJECT_OBSERVATION
                        or len(declarations) != 1
                        or declarations[0].selected_path_ref is None
                        or declarations[0].selected_path_ref.value() != proof_ref
                        or declarations[0].input_observation_ref != input_observation
                        or proof_ref not in relation_objects
                        or not expected_targets <= relation_objects):
                    raise ValueError("Bounded generic response context has no complete delivery proof")
                result.append(DeliveredGenericResponseContext(
                    MemoryObjectRef.from_stable_key(observation.observation_key),
                    MemoryObjectRef.from_stable_key(hypothesis.hypothesis_key),
                    observation.source_ref, observation.scope_key, proof_ref,
                    input_observation, response_act, hypothesis.evidence_keys,
                    observation.context_depth,
                    graph_surfaces=graph_surfaces))
                continue
            generated = unpack_record(proof[3], DELIVERY_PROOF_IDENTITY)
            if len(generated) != 5 or generated[0] != (2,):
                raise ValueError("Generic response generation record is incomplete")
            act_key = response_act.stable_key()
            input_observation = MemoryObjectRef.from_stable_key(generated[1])
            if (generated[3][:len(act_key) + 1] != (len(act_key), *act_key)
                    or generated[4] != proof[4]
                    or len(declarations) != 1
                    or declarations[0].selected_path_ref is None
                    or declarations[0].selected_path_ref.value() != proof_ref
                    or declarations[0].input_observation_ref != input_observation
                    or proof_ref not in tuple(
                        link.value() for link in observation.relation_refs)):
                raise ValueError("Generic response context has no complete delivery proof")
            result.append(DeliveredGenericResponseContext(
                MemoryObjectRef.from_stable_key(observation.observation_key),
                MemoryObjectRef.from_stable_key(hypothesis.hypothesis_key),
                observation.source_ref, observation.scope_key, proof_ref,
                input_observation, response_act, hypothesis.evidence_keys,
                observation.context_depth))
            continue
        if proof[0] != (1,):
            raise ValueError("Delivered response proof kind is not registered")
        generated = unpack_record(proof[3], 91529)
        if len(generated) != 5:
            raise ValueError("Delivered generation record is incomplete")
        context = OpenRoleGenerationContext.from_stable_key(generated[1])
        node = unpack_record(generated[0], 91527)
        act_key = response_act.stable_key()
        if (len(node) != 2 or node[1] != context.candidate.stable_key()
                or generated[2][:len(act_key) + 1] != (len(act_key), *act_key)
                or context.source != SourceRef.from_stable_key(proof[1])):
            raise ValueError("Delivered response source, action or role frame drifted")
        connector = ObjectIdentity.from_stable_key(node[0])
        representations = tuple(ObjectIdentity.from_stable_key(key)
                                for key in unpack_record(generated[3], 1))
        if tuple(value for item in representations for value in representation_parts(item)[1]) != proof[4]:
            raise ValueError("Delivered representations differ from acknowledged units")
        if (len(declarations) != 1 or declarations[0].selected_path_ref is None
                or declarations[0].selected_path_ref.value() != proof_ref
                or declarations[0].input_observation_ref != context.observation
                or proof_ref not in tuple(link.value() for link in observation.relation_refs)):
            raise ValueError("Response context has no unique source-backed delivery Episode")
        result.append(DeliveredResponseContext(
            MemoryObjectRef.from_stable_key(observation.observation_key),
            MemoryObjectRef.from_stable_key(hypothesis.hypothesis_key), observation.source_ref,
            observation.scope_key, proof_ref, context, response_act, connector,
            representations, hypothesis.evidence_keys, observation.context_depth))
        occurrences = read_response_roles(memory, context, proof_ref, SourceRef.from_stable_key(observation.source_ref[1:]))
        if occurrences:
            result[-1] = replace(result[-1], role_occurrences=occurrences)
    return tuple(sorted(result, key=lambda item: item.stable_key()))
