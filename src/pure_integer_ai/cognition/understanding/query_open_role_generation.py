"""Bind open input roles to source-local generation spans inside one query."""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_content import AnswerContentSelection
from pure_integer_ai.cognition.shared.generation_observed_surface import ObservedGenerationSpan
from pure_integer_ai.cognition.shared.generation_response import (
    ResponseActGenerationBinding, ResponseActGenerationTemplate,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef, span_identity
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_OBJECT_OBSERVATION, MEMORY_OBJECT_HYPOTHESIS, MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EXPANDED, SPACE_CORE, SPACE_DIALOGUE, SPACE_MEMORY, BindingEntry, QueryState,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.structure_order_consumer import StructureSlotValue
from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate, pack_record, unpack_record,
)

OPEN_GENERATION_CONTEXT_VERSION = 91518
OPEN_GENERATION_CONTEXT_ROLE = (91519, 1)


@dataclass(frozen=True, slots=True)
class OpenRoleGenerationContext:
    """An unresolved parse and its complete three-graph generation provenance."""

    query_key: tuple[int, ...]
    candidate: OpenRelationCandidate
    observation: MemoryObjectRef
    hypothesis: MemoryObjectRef
    competition_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    scope: ScopeIdentity
    evidence: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, OpenRelationCandidate):
            raise TypeError("open generation candidate must be an open relation")
        if self.query_key != self.candidate.source_ref:
            raise ValueError("open generation query and input provenance differ")
        source = self.source
        for ref, kind in ((self.observation, MEMORY_OBJECT_OBSERVATION),
                          (self.hypothesis, MEMORY_OBJECT_HYPOTHESIS)):
            if (not isinstance(ref, MemoryObjectRef) or ref.object_kind != kind
                    or ref.owner != source.owner or ref.versions != source.versions):
                raise ValueError("open generation Memory reference ownership differs")
        if (not isinstance(self.scope, ScopeIdentity) or self.scope.owner != source.owner
                or self.scope.versions != source.versions
                or self.observation.memory_space != self.hypothesis.memory_space):
            raise ValueError("open generation source, scope and Memory space differ")
        if not self.competition_key or not self.evidence:
            raise ValueError("open generation requires competition and evidence")
        if self.evidence != tuple(sorted(set(self.evidence))):
            raise ValueError("open generation evidence must be complete and canonical")
        if {key[0] for key in self.evidence} != {SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE}:
            raise ValueError("open generation requires evidence from all three graphs")
        self.stable_key()

    @property
    def source(self) -> SourceRef:
        if not self.source_ref or type(self.source_ref[0]) is not int or self.source_ref[0] <= 0:
            raise ValueError("open generation requires a registered Memory source")
        return SourceRef.from_stable_key(self.source_ref[1:])

    def role_span(self, ordinal: int) -> ObjectIdentity:
        """Name the observed span, without assigning a Core concept or truth value."""
        if type(ordinal) is not int or not 0 <= ordinal < len(self.candidate.bindings):
            raise ValueError("open generation role ordinal is outside the parse")
        role = self.candidate.bindings[ordinal]
        return span_identity(self.source, members=((role.start, role.end),))

    def stable_key(self) -> tuple[int, ...]:
        return pack_record(
            OPEN_GENERATION_CONTEXT_VERSION, self.query_key, self.candidate.stable_key(),
            self.observation.stable_key(), self.hypothesis.stable_key(), self.competition_key,
            self.source_ref, self.scope.stable_key(), pack_record(1, *self.evidence),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> OpenRoleGenerationContext:
        fields = unpack_record(key, OPEN_GENERATION_CONTEXT_VERSION)
        if len(fields) != 8:
            raise ValueError("open generation context field count differs")
        return cls(fields[0], OpenRelationCandidate.from_stable_key(fields[1]),
                   MemoryObjectRef.from_stable_key(fields[2]),
                   MemoryObjectRef.from_stable_key(fields[3]), fields[4], fields[5],
                   ScopeIdentity.from_stable_key(fields[6]), unpack_record(fields[7], 1))

    def query_bindings(self) -> tuple[BindingEntry, ...]:
        """Keep both the original parse and source-local typed span in the query."""
        return (
            BindingEntry(OPEN_GENERATION_CONTEXT_ROLE, self.stable_key(), space=SPACE_DIALOGUE,
                         scope_key=self.scope.stable_key()),
            *(BindingEntry(role.role, self.role_span(role.ordinal).stable_key(),
                           space=SPACE_DIALOGUE, scope_key=self.scope.stable_key())
              for role in self.candidate.bindings),
        )

    def observed_spans(self) -> tuple[ObservedGenerationSpan, ...]:
        """Pass graph-derived role tokens and complete provenance to existing G-03."""
        return tuple(ObservedGenerationSpan(
            self.source, self.scope, self.observation, self.hypothesis,
            ObjectIdentity.from_stable_key(role.role), role.ordinal, role.start, role.end,
            role.values, self.stable_key(), self.evidence,
        ) for role in self.candidate.bindings)

    def bind_response_act(
            self, selection: AnswerContentSelection, template: ResponseActGenerationTemplate,
            role_slots: tuple[tuple[ObjectIdentity, int], ...],
            constant_values: tuple[StructureSlotValue, ...] = (),
            forming_evidence: tuple[tuple[int, ...], ...] = (),
            ) -> ResponseActGenerationBinding:
        """Supply every observed role to existing G-02 without asserting its relation."""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("open generation requires the actual G-01 selection")
        if not isinstance(template, ResponseActGenerationTemplate):
            raise TypeError("open generation requires a trained response-act template")
        if (selection.stance == selection.protocol.answer
                or template.stance != selection.stance
                or template.branch != selection.request.goal.target_branch):
            raise ValueError("an open role context cannot authorize a factual answer")
        goal = selection.request.goal
        if goal.source != self.source or goal.scope != self.scope:
            raise ValueError("response-act goal must retain its Memory source and scope")
        if (type(role_slots) is not tuple or len(role_slots) != len(self.candidate.bindings)
                or any(type(item) is not tuple or len(item) != 2 for item in role_slots)):
            raise ValueError("response-act role mapping must cover the complete parse")
        slots = {item.slot: item for item in template.content_slots}
        if (type(constant_values) is not tuple
                or any(not isinstance(item, StructureSlotValue) for item in constant_values)
                or len({item.slot for item in constant_values}) != len(constant_values)
                or (constant_values and not forming_evidence)):
            raise ValueError("对话语言原子必须具有完整槽值及形成证据")
        constant_slots = {item.slot for item in constant_values}
        if (len({slot for slot, _ in role_slots}) != len(role_slots)
                or constant_slots & {slot for slot, _ in role_slots}
                or constant_slots | {slot for slot, _ in role_slots} != set(slots)
                or any(type(index) is not int for _, index in role_slots)
                or {index for _, index in role_slots} != set(range(len(self.candidate.bindings)))):
            raise ValueError("response-act mapping omits or repeats a slot or role")
        if any(slots[slot].role.stable_key() != self.candidate.bindings[index].role
               for slot, index in role_slots):
            raise ValueError("response-act slot role differs from the observed role")
        return ResponseActGenerationBinding(
            selection.stable_key(), template.stable_key(), self.source, self.scope,
            (*tuple(StructureSlotValue(slot, self.role_span(index)) for slot, index in role_slots),
             *constant_values),
            tuple(sorted({self.stable_key(), *forming_evidence})),
        )


def prepare_open_role_generation(
        state: QueryState, candidate: OpenRelationCandidate, *, observation_key: tuple[int, ...],
        hypothesis_key: tuple[int, ...], competition_key: tuple[int, ...],
        source_ref: tuple[int, ...], scope_key: tuple[int, ...],
        evidence_keys: tuple[tuple[int, ...], ...],
        ) -> OpenRoleGenerationContext | None:
    """Wait for shared frontier evidence, not for failure of a different graph."""
    required_roots = {
        (SPACE_CORE, candidate.frame.schema_key()), (SPACE_DIALOGUE, candidate.stable_key()),
        (SPACE_MEMORY, observation_key), (SPACE_DIALOGUE, hypothesis_key),
    }
    expanded = {(item.owner_space, item.root_key) for item in state.roots
                if item.status == ROOT_EXPANDED}
    if not required_roots <= expanded:
        return None
    memory_roots = tuple(item for item in state.roots
                         if (item.owner_space, item.root_key) in {
                             (SPACE_MEMORY, observation_key), (SPACE_DIALOGUE, hypothesis_key)})
    if any(item.source_ref != source_ref or item.scope_key != scope_key for item in memory_roots):
        raise ValueError("open generation source or scope differs from the expanded roots")
    bound = {(item.space, item.role_key, item.filler_key, item.scope_key)
             for item in state.bindings}
    if any((SPACE_DIALOGUE, item.role, item.stable_key(), candidate.source_ref) not in bound
           for item in candidate.bindings):
        return None
    required_evidence = {
        (SPACE_CORE, candidate.frame.schema_key()), (SPACE_DIALOGUE, candidate.stable_key()),
        *((space, key) for key in evidence_keys for space in (SPACE_MEMORY, SPACE_DIALOGUE)),
    }
    evidence = tuple(item for item in state.evidence
                     if (item.space, item.evidence_key) in required_evidence)
    if {(item.space, item.evidence_key) for item in evidence} != required_evidence:
        return None
    if not evidence_keys:
        raise ValueError("open generation Memory hypothesis has no Evidence")
    for item in evidence:
        if item.space == SPACE_CORE and (item.polarity != 1
                or item.payload_key != candidate.frame.stable_key()):
            raise ValueError("open generation frame evidence differs")
        if item.evidence_key == candidate.stable_key() and (
                item.polarity != 3 or item.payload_key != candidate.stable_key()
                or item.scope_key != candidate.source_ref):
            raise ValueError("open generation input parse evidence differs")
        if item.evidence_key in evidence_keys and item.hypothesis_key != hypothesis_key:
            raise ValueError("open generation Memory Evidence points to a different hypothesis")
    return OpenRoleGenerationContext(
        state.query_key, candidate, MemoryObjectRef.from_stable_key(observation_key),
        MemoryObjectRef.from_stable_key(hypothesis_key), competition_key, source_ref,
        ScopeIdentity.from_stable_key(scope_key), tuple(sorted(item.stable_key() for item in evidence)),
    )
