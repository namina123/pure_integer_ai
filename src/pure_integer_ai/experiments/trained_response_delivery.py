"""Attribute delivered graph responses to session Memory without replay or Core writes."""
from __future__ import annotations

from pure_integer_ai.experiments.relation_runtime_phase import observed_relation_phase

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
    MEMORY_EVENT_EPISODE, MEMORY_EVENT_OBSERVATION,
    MEMORY_OBJECT_HYPOTHESIS, MEMORY_OBJECT_OBSERVATION,
    EpisodePayload, MemoryEvent, MemoryLinkedRef, MemoryObjectRef, ObservationPayload,
    UsePayload,
)
from pure_integer_ai.cognition.shared.memory_event_log import MaterializedMemoryEvent
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.query_driver import unvisited_frontier
from pure_integer_ai.cognition.shared.query_state import ROOT_PENDING, QueryState
from pure_integer_ai.cognition.shared.response_plan import ResponsePlan
from pure_integer_ai.cognition.understanding.query_input_structure import (
    STRUCTURE_CLOSED,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_OBSERVED, CLOCK_MEMORY_USED, LogicalClockIdentity, query_scope, session_scope,
)
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft, ObservationIntakeDraft,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION, MEMORY_CATEGORY_OPEN_RELATION,
    MEMORY_CATEGORY_RESPONSE, _parts, frame_key, memory_candidate_category,
)
from pure_integer_ai.cognition.understanding.query_open_role_generation import OpenRoleGenerationContext
from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate, pack_record,
)
from pure_integer_ai.experiments.memory_use_runtime import (
    append_memory_use, append_memory_use_outcome,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    _DIALOGUE_MEMORY_NAMESPACE, DialogueMemoryAppend, TrainedDialogueMemoryGraph,
)
from pure_integer_ai.experiments.trained_open_response_runtime import OpenResponseGeneration
from pure_integer_ai.experiments.trained_response_occurrence import (
    ResponseRoleOccurrence, materialize_factual_response_roles,
    materialize_response_roles,
)


DELIVERY_PROOF_IDENTITY = 91530
_DELIVERY_USE = 91531
FACTUAL_EVIDENCE_REF_IDENTITY = 91536


def _current_open_hypothesis(
        memory: TrainedDialogueMemoryGraph,
        observation: MemoryObjectRef,
        ) -> MemoryObjectRef | None:
    """Recover the unique current-turn open-role H-00 Memory object.

    QueryState Memory evidence also carries cross-turn H-00 keys, whose
    ``hypothesis_key`` is a ``HypothesisKey`` record rather than a
    ``MemoryObjectRef``.  Delivery must select the current input's declared
    open-role object from its bounded Observation closure instead of relying
    on evidence ordering.
    """
    selected = memory.active_structures(
        candidate_keys=(), observation_refs=(observation,),
        context_depth_limit=0,
    )
    candidates = []
    for item in selected.hypotheses:
        if item.observation_key != observation.stable_key():
            continue
        try:
            if memory_candidate_category(item.candidate_key) != (
                    MEMORY_CATEGORY_OPEN_RELATION):
                continue
            ref = MemoryObjectRef.from_stable_key(item.hypothesis_key)
        except (TypeError, ValueError):
            continue
        if ref.object_kind != MEMORY_OBJECT_HYPOTHESIS:
            continue
        candidates.append(ref)
    unique = {item.stable_key(): item for item in candidates}
    if len(unique) == 1:
        return next(iter(unique.values()))

    # Context-only Observations have no role occurrence to materialize.  Keep
    # a deterministic typed Memory hypothesis only when the bounded closure
    # contains exactly one recoverable object; ambiguity remains absent rather
    # than being turned into an arbitrary provenance claim.
    typed = []
    for item in selected.hypotheses:
        try:
            ref = MemoryObjectRef.from_stable_key(item.hypothesis_key)
        except (TypeError, ValueError):
            continue
        if ref.object_kind == MEMORY_OBJECT_HYPOTHESIS:
            typed.append(ref)
    typed_unique = {item.stable_key(): item for item in typed}
    return next(iter(typed_unique.values())) if len(typed_unique) == 1 else None


def resolve_factual_claim_input_hypotheses(
        memory: TrainedDialogueMemoryGraph,
        observation: MemoryObjectRef,
        claim_keys: tuple[tuple[int, ...], ...],
        ) -> tuple[tuple[tuple[int, ...], MemoryObjectRef], ...]:
    """Match each closed claim to one current-input relation hypothesis.

    A closed relation question is admitted to Memory as category 5, while an
    open-role query is admitted as category 8.  Both are source-local
    hypotheses for the same delivery contract; the proposition identity is
    read from the candidate key and every claim must still have exactly one
    matching hypothesis in this Observation closure.
    """
    if (not isinstance(memory, TrainedDialogueMemoryGraph)
            or not isinstance(observation, MemoryObjectRef)
            or observation.object_kind != MEMORY_OBJECT_OBSERVATION
            or type(claim_keys) is not tuple or not claim_keys
            or any(type(key) is not tuple or not key
                   or any(type(value) is not int for value in key)
                   for key in claim_keys)
            or len(set(claim_keys)) != len(claim_keys)):
        raise TypeError("factual claim/H provenance input is invalid")
    selected = memory.active_structures(
        candidate_keys=(), observation_refs=(observation,),
        context_depth_limit=0,
    )
    by_proposition: dict[tuple[int, ...], dict[tuple[int, ...], MemoryObjectRef]] = {}
    for item in selected.hypotheses:
        if item.observation_key != observation.stable_key():
            continue
        try:
            category = memory_candidate_category(item.candidate_key)
            if category == MEMORY_CATEGORY_OPEN_RELATION:
                fields = _parts(item.candidate_key, MEMORY_CANDIDATE_VERSION)
                if len(fields) != 2:
                    continue
                proposition = OpenRelationCandidate.from_stable_key(
                    fields[1]).frame.proposition
            elif category == 5:
                fields = _parts(item.candidate_key, MEMORY_CANDIDATE_VERSION)
                if (len(fields) != 6 or not fields[1]
                        or not fields[2] or not fields[3] or not fields[4]
                        or len(fields[5]) != 7
                        or fields[5][0] < 2
                        or fields[5][0] != fields[5][1]
                        or fields[5][5] != STRUCTURE_CLOSED):
                    continue
                proposition = fields[1]
            else:
                continue
            hypothesis = MemoryObjectRef.from_stable_key(item.hypothesis_key)
        except (TypeError, ValueError):
            continue
        if hypothesis.object_kind != MEMORY_OBJECT_HYPOTHESIS:
            continue
        by_proposition.setdefault(proposition, {})[
            hypothesis.stable_key()] = hypothesis
    result = []
    for claim_key in claim_keys:
        matches = by_proposition.get(claim_key, {})
        if len(matches) != 1:
            return ()
        result.append((claim_key, next(iter(matches.values()))))
    if len({item[1].stable_key() for item in result}) != len(result):
        return ()
    return tuple(result)


def _slot_input_hypotheses(
        response: ResponsePlan,
        claim_input_hypotheses: tuple[
            tuple[tuple[int, ...], MemoryObjectRef], ...],
        ) -> tuple[MemoryObjectRef, ...]:
    """Expand exact claim provenance through contiguous ResponsePlan ranges."""
    if not claim_input_hypotheses:
        return ()
    claims = {item.stable_key() for item in response.claim_refs}
    mapping = {claim: hypothesis for claim, hypothesis in claim_input_hypotheses}
    ranges = {claim: (start, end) for claim, start, end in response.claim_slot_ranges}
    if not ranges and len(claims) == 1:
        only_claim = next(iter(claims))
        ranges = {only_claim: (0, len(response.slot_sequence))}
    if (set(mapping) != claims or set(ranges) != claims
            or len(mapping) != len(claim_input_hypotheses)
            or any(not isinstance(hypothesis, MemoryObjectRef)
                   or hypothesis.object_kind != MEMORY_OBJECT_HYPOTHESIS
                   for hypothesis in mapping.values())):
        raise ValueError("factual claim/H provenance does not cover the plan")
    slots = [None] * len(response.slot_sequence)
    for claim, (start, end) in ranges.items():
        for ordinal in range(start, end):
            if slots[ordinal] is not None:
                raise ValueError("factual claim/H slot provenance overlaps")
            slots[ordinal] = mapping[claim]
    if any(item is None for item in slots):
        raise ValueError("factual claim/H slot provenance has a gap")
    return tuple(slots)


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
            "role_occurrence_hypotheses": [
                list(item.hypothesis.stable_key())
                for item in self.role_occurrences],
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
    # The legacy proposition_key remains the first selected claim.  New
    # composition generations provide the complete ordered claim set here.
    claim_keys: tuple[tuple[int, ...], ...] = ()
    # Exact current-input category-8 H provenance, keyed by closed claim.
    # Empty preserves compatibility with legacy single-H callers; production
    # composition fills this from the current Observation closure.
    claim_input_hypotheses: tuple[
        tuple[tuple[int, ...], MemoryObjectRef], ...] = ()

    def __post_init__(self) -> None:
        response = self.response_plan
        if (not isinstance(response, ResponsePlan) or not response.claim_refs
                or not response.memory_refs or not response.evidence_refs):
            raise ValueError("factual delivery requires a fully bound claim plan")
        claim_keys = self.claim_keys or (self.proposition_key,)
        if (type(claim_keys) is not tuple or not claim_keys
                or any(type(key) is not tuple or not key
                       or any(type(value) is not int for value in key)
                       for key in claim_keys)
                or len(set(claim_keys)) != len(claim_keys)
                or set(claim_keys) != {
                    item.stable_key() for item in response.claim_refs}
                or self.proposition_key not in claim_keys):
            raise ValueError("factual delivery claim set differs from ResponsePlan")
        if len(claim_keys) > 1 and (
                not response.claim_evidence_refs
                or not response.claim_slot_ranges):
            raise ValueError("multi-claim delivery lacks per-claim audit records")
        if (type(self.claim_input_hypotheses) is not tuple
                or any(type(item) is not tuple or len(item) != 2
                       or type(item[0]) is not tuple or not item[0]
                       or any(type(value) is not int for value in item[0])
                       or not isinstance(item[1], MemoryObjectRef)
                       or item[1].object_kind != MEMORY_OBJECT_HYPOTHESIS
                       for item in self.claim_input_hypotheses)):
            raise TypeError("factual delivery claim/H provenance is invalid")
        if self.claim_input_hypotheses:
            if (len({claim for claim, _hypothesis in self.claim_input_hypotheses})
                    != len(self.claim_input_hypotheses)
                    or {claim for claim, _hypothesis in self.claim_input_hypotheses}
                    != set(claim_keys)
                    or len({hypothesis.stable_key()
                            for _claim, hypothesis in self.claim_input_hypotheses})
                    != len(self.claim_input_hypotheses)):
                raise ValueError("factual delivery claim/H provenance is incomplete")
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
            frame_key(5, *(frame_key(ordinal + 1, key)
                          for ordinal, key in enumerate(
                              self.claim_keys or (self.proposition_key,)))),
            frame_key(6, *(frame_key(1, claim, hypothesis.stable_key())
                          for claim, hypothesis in self.claim_input_hypotheses)),
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
    role_occurrences: tuple[ResponseRoleOccurrence, ...] = ()

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
            "role_occurrences": [list(item.stable_key()) for item in self.role_occurrences],
            "role_occurrence_hypotheses": [
                list(item.hypothesis.stable_key())
                for item in self.role_occurrences],
            "delivered": 1,
            # Delivery observes a selected Core claim; it never creates one.
            "fact_claims_added": 0,
        }


@observed_relation_phase("memory_commit")
def _commit_response_memory(memory: TrainedDialogueMemoryGraph) -> None:
    """记录实际 delivery 的事务提交边界；日志不替代数据库恢复验收。"""
    memory.backend.commit()


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
    identity_hash = _factual_evidence_ref_hash(memory, ref)
    return memory.intake.event_log.scoped_identities.registry.read_key(
        FACTUAL_EVIDENCE_REF_IDENTITY, identity_hash)


def read_factual_evidence_refs(
        memory: TrainedDialogueMemoryGraph,
        refs: tuple[ObjectIdentity, ...],
        ) -> tuple[tuple[int, ...], ...]:
    """Resolve normalized Evidence references with batched checked reads."""
    if type(refs) is not tuple:
        raise TypeError("factual evidence locators must be a tuple")
    hashes = tuple(_factual_evidence_ref_hash(memory, ref) for ref in refs)
    return memory.intake.event_log.scoped_identities.registry.read_keys(
        FACTUAL_EVIDENCE_REF_IDENTITY, hashes)


def _factual_evidence_ref_hash(
        memory: TrainedDialogueMemoryGraph,
        ref: ObjectIdentity,
        ) -> int:
    """Validate the session-owned stable locator before registry access."""
    if (not isinstance(ref, ObjectIdentity)
            or ref.object_kind != OBJECT_CONTEXT_SCOPE
            or ref.owner != memory.owner or len(ref.components) != 2
            or ref.components[0] != FACTUAL_EVIDENCE_REF_IDENTITY):
        raise ValueError("factual evidence locator is not owned by this session")
    return ref.components[1]


@observed_relation_phase("memory_append")
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
    claim_keys = generation.claim_keys or (generation.proposition_key,)
    slot_hypotheses = _slot_input_hypotheses(
        response, generation.claim_input_hypotheses)
    if not slot_hypotheses:
        legacy_hypothesis = _current_open_hypothesis(
            memory, generation.input_observation)
        if legacy_hypothesis is not None:
            slot_hypotheses = (legacy_hypothesis,) * len(response.slot_sequence)
    claim_ref_keys = {item.stable_key() for item in response.claim_refs}
    claim_audit = {
        claim: refs for claim, refs in response.claim_evidence_refs}
    if (state.termination != 1
            or any(root.status == ROOT_PENDING for root in state.roots)
            or {root.owner_space for root in state.roots} != {1, 2, 3}
            or state.best_candidate_key not in claim_keys
            or set(claim_keys) != claim_ref_keys
            or not response.claim_refs
            or not {item.stable_key() for item in state.evidence}.issubset(
                set(response.evidence_refs))
            or (len(claim_keys) > 1 and (
                set(claim_audit) != claim_ref_keys
                or any(ref not in response.evidence_refs
                       for refs in claim_audit.values() for ref in refs)
                or len(response.claim_slot_ranges) != len(claim_keys)))):
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

    used_refs = tuple(sorted(set(response.memory_refs) | set(slot_hypotheses),
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
    claim_audit_record = frame_key(
        5,
        *(frame_key(
            1,
            claim,
            frame_key(1, *claim_audit.get(claim, ())),
        ) for claim in sorted(claim_audit)))
    slot_record = frame_key(2, *(frame_key(
        ordinal + 1,
        slot.role.stable_key(),
        slot.filler.stable_key(),
        (1 if slot.required else 0, slot.source_hash),
        slot.token_digest,
        slot.allowed_node_kinds,
        *(slot_hypotheses[ordinal].stable_key(),)
        if slot_hypotheses else (),
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
            frame_key(5, *(frame_key(ordinal + 1, key)
                          for ordinal, key in enumerate(claim_keys))),
            claim_record,
            claim_audit_record,
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
    role_occurrences = ()
    if not slot_hypotheses:
        # A closed operation request can deliver a factual claim without a
        # category-5/8 input relation hypothesis.  Anchor its emitted roles to
        # the response H just written by this exact proof and response act.
        # This preserves durable role occurrences without inventing an input
        # fact or choosing among unrelated input hypotheses.
        response_hypotheses = tuple(
            item for item in memory.active_structures(
                candidate_keys=(candidate_key,),
                observation_refs=(output_ref,),
                context_depth_limit=0,
            ).hypotheses
            if (item.candidate_key == candidate_key
                and item.observation_key == output_ref.stable_key())
        )
        if len(response_hypotheses) != 1:
            raise RuntimeError(
                "factual response roles have no unique proof-bound response hypothesis")
        response_hypothesis = MemoryObjectRef.from_stable_key(
            response_hypotheses[0].hypothesis_key)
        if response_hypothesis.object_kind != MEMORY_OBJECT_HYPOTHESIS:
            raise RuntimeError("factual response role hypothesis has the wrong kind")
        slot_hypotheses = (response_hypothesis,) * len(response.slot_sequence)
    if slot_hypotheses:
        role_occurrences = materialize_factual_response_roles(
            memory,
            input_observation=generation.input_observation,
            hypothesis=slot_hypotheses[0],
            slot_hypotheses=slot_hypotheses,
            scope_key=next(
                (item.scope_key for item in state.evidence if item.space == 2),
                source.stable_key(),
            ),
            proof_ref=proof_ref,
            output_observation=output_ref,
            output_source=output.source,
            slots=response.slot_sequence,
        )

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
    episode_ref = events.declaration_ref(MEMORY_EVENT_EPISODE, episode_payload, scope=scope)
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
    _commit_response_memory(memory)
    return FactualResponseDeliveryReceipt(
        proof_ref,
        output,
        episode,
        tuple(uses),
        tuple(outcomes),
        len(response.slot_sequence),
        len(response.claim_refs),
        len(evidence_locators),
        role_occurrences,
    )


@observed_relation_phase("memory_append")
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
    episode_ref = events.declaration_ref(MEMORY_EVENT_EPISODE, episode_payload, scope=scope)
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
    _commit_response_memory(memory)
    return ResponseDeliveryReceipt(
        proof_ref, output, episode, tuple(uses), tuple(outcomes), len(preview.slots),
        sum(slot.observed_surface is not None for slot in preview.slots), role_occurrences, context)


@observed_relation_phase("memory_append")
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
    } | {
        # A delivered cross-turn context can contribute an Observation or
        # Hypothesis root without upgrading that structural O/H into a
        # support EvidenceEntry.  Those roots are nevertheless part of this
        # completed QueryState and are validated below through their exact
        # Memory declarations, matching factual delivery's contract.
        item.root_key for item in state.roots if item.owner_space == 2
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
    episode_ref = events.declaration_ref(MEMORY_EVENT_EPISODE, episode_payload, scope=scope)
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
    _commit_response_memory(memory)
    return GenericResponseDeliveryReceipt(
        proof_ref, output, episode, tuple(uses), tuple(outcomes),
        len(response.slot_sequence))
