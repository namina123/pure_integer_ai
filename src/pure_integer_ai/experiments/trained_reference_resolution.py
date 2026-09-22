"""Resolve cross-turn role occurrences from trained graph evidence.

The source-local A-01 runtime deliberately rejects occurrences from different
``SourceRef`` values.  Dialogue turns are different sources, so this adapter
keeps that boundary intact and reuses H-00/H-04 for a separate discourse
competition.  A unique winner requires an active trained ``REFERS`` fact that
has already entered the shared ``QueryState`` as Core support.  Entity, event,
property and time observations remain separate evidence dimensions and never
become a recency or stable-order tiebreak.
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_SUPPORTED,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    HypothesisResolver,
    ResolverDecision,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ENTITY,
    OBJECT_EVENT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.query_state import (
    SPACE_MEMORY,
    BindingEntry,
    EvidenceEntry,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.understanding.occurrence_index import OccurrenceIndex
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    TrainedInputStructureProjector,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
    authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_EVENT_AFTER,
    RELATION_EVENT_BEFORE,
    RELATION_EVENT_SAME,
    RELATION_EVENT_UNKNOWN,
    RELATION_PROPERTY,
    RELATION_REFERS,
    ROLE_EVENT_AFTER_OBJECT,
    ROLE_EVENT_AFTER_SUBJECT,
    ROLE_EVENT_BEFORE_OBJECT,
    ROLE_EVENT_BEFORE_SUBJECT,
    ROLE_EVENT_SAME_OBJECT,
    ROLE_EVENT_SAME_SUBJECT,
    ROLE_EVENT_UNKNOWN_OBJECT,
    ROLE_EVENT_UNKNOWN_SUBJECT,
    ROLE_PROPERTY_ATTRIBUTE,
    ROLE_PROPERTY_INTENSITY,
    ROLE_PROPERTY_MODALITY,
    ROLE_PROPERTY_POLARITY,
    ROLE_PROPERTY_SUBJECT,
    ROLE_PROPERTY_VALUE,
    ROLE_REFERS_FROM,
    ROLE_REFERS_TO,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface,
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.experiments.trained_response_context import DeliveredResponseContext
from pure_integer_ai.experiments.trained_response_occurrence import (
    ReferenceCandidateSet,
    ResponseRoleOccurrence,
)


REFERENCE_RESOLUTION_VERSION = 91536

REFERENCE_DIMENSION_ENTITY = 1
REFERENCE_DIMENSION_EVENT = 2
REFERENCE_DIMENSION_PROPERTY = 3
REFERENCE_DIMENSION_TIME = 4
REFERENCE_DIMENSION_REFERS = 5

REFERENCE_NOT_APPLICABLE = 1
REFERENCE_OPEN = 2
REFERENCE_RESOLVED = 3
REFERENCE_CONFLICT = 4

_DIMENSIONS = frozenset({
    REFERENCE_DIMENSION_ENTITY,
    REFERENCE_DIMENSION_EVENT,
    REFERENCE_DIMENSION_PROPERTY,
    REFERENCE_DIMENSION_TIME,
    REFERENCE_DIMENSION_REFERS,
})
_STANCES = frozenset({EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN})
_EVIDENCE_HASHER = Hasher("trained_reference_resolution.evidence.v1")


def _pack(key: tuple[int, ...]) -> tuple[int, ...]:
    if type(key) is not tuple or any(type(value) is not int for value in key):
        raise ValueError("reference resolution keys must contain strict integers")
    return len(key), *key


def _records(tag: int, records: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    values = [tag, len(records)]
    for record in records:
        values.extend(_pack(record))
    return tuple(values)


@dataclass(frozen=True, slots=True)
class _CoreFact:
    proposition: ObjectIdentity
    predicate: ObjectIdentity
    bindings: tuple[tuple[ObjectIdentity, ObjectIdentity], ...]
    source: SourceRef
    source_hash: int

    def __post_init__(self) -> None:
        if (not isinstance(self.proposition, ObjectIdentity)
                or not isinstance(self.predicate, ObjectIdentity)
                or not isinstance(self.source, SourceRef)):
            raise TypeError("reference Core fact identity types differ")
        if type(self.source_hash) is not int or self.source_hash <= 0:
            raise ValueError("reference Core fact requires a positive source hash")
        if (type(self.bindings) is not tuple or not self.bindings
                or any(len(item) != 2
                       or not isinstance(item[0], ObjectIdentity)
                       or not isinstance(item[1], ObjectIdentity)
                       for item in self.bindings)):
            raise TypeError("reference Core fact bindings are incomplete")

    def filler(self, role: ObjectIdentity) -> ObjectIdentity:
        matches = tuple(value for candidate, value in self.bindings if candidate == role)
        if len(matches) != 1:
            raise ValueError("reference Core fact Role does not have one filler")
        return matches[0]

    def stable_key(self) -> tuple[int, ...]:
        return (
            REFERENCE_RESOLUTION_VERSION,
            *_pack(self.proposition.stable_key()),
            *_pack(self.predicate.stable_key()),
            self.source_hash,
            *_pack(self.source.stable_key()),
            len(self.bindings),
            *(value for role, filler in self.bindings
              for value in (*_pack(role.stable_key()), *_pack(filler.stable_key()))),
        )


@dataclass(frozen=True, slots=True)
class TrainedReferenceEvidence:
    """One candidate-specific evidence item derived from active graph facts."""

    antecedent_key: tuple[int, ...]
    target: ObjectIdentity
    dimension: int
    stance: int
    decisive: int
    proposition_keys: tuple[tuple[int, ...], ...]
    predicate_keys: tuple[tuple[int, ...], ...]
    source: SourceRef
    source_hash: int
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        _pack(self.antecedent_key)
        if not isinstance(self.target, ObjectIdentity):
            raise TypeError("reference evidence target must be an ObjectIdentity")
        if self.dimension not in _DIMENSIONS or self.stance not in _STANCES:
            raise ValueError("reference evidence dimension or stance is not registered")
        if self.decisive not in {0, 1}:
            raise ValueError("reference evidence decisive flag must be 0 or 1")
        if self.decisive and (self.dimension != REFERENCE_DIMENSION_REFERS
                              or self.stance != EVIDENCE_SUPPORT):
            raise ValueError("only a supported REFERS fact can be decisive")
        for name in ("proposition_keys", "predicate_keys"):
            records = getattr(self, name)
            if (type(records) is not tuple or not records
                    or records != tuple(sorted(set(records)))):
                raise ValueError(f"reference evidence {name} must be canonical")
            for key in records:
                if not key:
                    raise ValueError(f"reference evidence {name} cannot contain empty keys")
                _pack(key)
        if not isinstance(self.source, SourceRef):
            raise TypeError("reference evidence source must be a SourceRef")
        if type(self.source_hash) is not int or self.source_hash <= 0:
            raise ValueError("reference evidence source hash must be positive")
        _pack(self.trace)

    def available(self, core_support: frozenset[tuple[int, ...]]) -> bool:
        return set(self.proposition_keys) <= core_support

    def stable_key(self) -> tuple[int, ...]:
        return (
            REFERENCE_RESOLUTION_VERSION,
            *_pack(self.antecedent_key),
            *_pack(self.target.stable_key()),
            self.dimension,
            self.stance,
            self.decisive,
            *_pack(_records(1, self.proposition_keys)),
            *_pack(_records(2, self.predicate_keys)),
            self.source_hash,
            *_pack(self.source.stable_key()),
            *_pack(self.trace),
        )


@dataclass(frozen=True, slots=True)
class CrossTurnReferenceCandidate:
    """A historical occurrence and the trained semantic objects found for its span."""

    occurrence: ResponseRoleOccurrence
    context_key: tuple[int, ...]
    context_depth: int
    semantic_objects: tuple[ObjectIdentity, ...]
    hypothesis: HypothesisKey

    def __post_init__(self) -> None:
        if not isinstance(self.occurrence, ResponseRoleOccurrence):
            raise TypeError("reference candidate occurrence type differs")
        _pack(self.context_key)
        if type(self.context_depth) is not int or self.context_depth < 0:
            raise ValueError("reference candidate context depth must be nonnegative")
        if (type(self.semantic_objects) is not tuple
                or any(not isinstance(item, ObjectIdentity) for item in self.semantic_objects)
                or self.semantic_objects != tuple(sorted(
                    set(self.semantic_objects), key=lambda item: item.stable_key()))):
            raise ValueError("reference candidate semantic objects must be canonical")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("reference candidate requires an H-00 key")

    def stable_key(self) -> tuple[int, ...]:
        return (
            REFERENCE_RESOLUTION_VERSION,
            *_pack(self.occurrence.stable_key()),
            *_pack(self.context_key),
            self.context_depth,
            len(self.semantic_objects),
            *(value for item in self.semantic_objects
              for value in _pack(item.stable_key())),
            *_pack(self.hypothesis.stable_key()),
        )


@dataclass(frozen=True, slots=True)
class CrossTurnReferenceResolution:
    """H-00/H-04 result with an evidence-qualified singleton winner."""

    candidate_set: ReferenceCandidateSet
    candidates: tuple[CrossTurnReferenceCandidate, ...]
    evidence: tuple[TrainedReferenceEvidence, ...]
    evidence_records: tuple[EvidenceRecord, ...]
    decision: ResolverDecision
    requires_resolution: int
    status: int
    winner_hypothesis_key: tuple[int, ...] = ()
    winner_target_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_set, ReferenceCandidateSet):
            raise TypeError("cross-turn resolution candidate set type differs")
        if (type(self.candidates) is not tuple or not self.candidates
                or any(not isinstance(item, CrossTurnReferenceCandidate)
                       for item in self.candidates)):
            raise TypeError("cross-turn resolution candidates are incomplete")
        if (type(self.evidence) is not tuple
                or any(not isinstance(item, TrainedReferenceEvidence)
                       for item in self.evidence)):
            raise TypeError("cross-turn resolution evidence type differs")
        if (type(self.evidence_records) is not tuple
                or any(not isinstance(item, EvidenceRecord)
                       for item in self.evidence_records)):
            raise TypeError("cross-turn H-00 evidence type differs")
        if not isinstance(self.decision, ResolverDecision):
            raise TypeError("cross-turn resolution requires an H-04 decision")
        if self.requires_resolution not in {0, 1}:
            raise ValueError("requires_resolution must be 0 or 1")
        if self.status not in {
                REFERENCE_NOT_APPLICABLE, REFERENCE_OPEN,
                REFERENCE_RESOLVED, REFERENCE_CONFLICT}:
            raise ValueError("cross-turn resolution status is not registered")
        _pack(self.winner_hypothesis_key)
        _pack(self.winner_target_key)
        if (self.status == REFERENCE_RESOLVED) != bool(self.winner_hypothesis_key):
            raise ValueError("resolved status and winner hypothesis differ")
        if bool(self.winner_hypothesis_key) != bool(self.winner_target_key):
            raise ValueError("winner hypothesis and target must be paired")
        if not self.requires_resolution and self.status != REFERENCE_NOT_APPLICABLE:
            raise ValueError("non-reference role cannot expose a reference decision")
        if self.requires_resolution and self.status == REFERENCE_NOT_APPLICABLE:
            raise ValueError("reference role cannot be marked not applicable")

    @property
    def winner(self) -> CrossTurnReferenceCandidate | None:
        if not self.winner_hypothesis_key:
            return None
        matches = tuple(item for item in self.candidates
                        if item.hypothesis.stable_key() == self.winner_hypothesis_key)
        if len(matches) != 1:
            raise ValueError("reference winner does not map to one candidate")
        return matches[0]

    @property
    def winner_target(self) -> ObjectIdentity | None:
        return (None if not self.winner_target_key
                else ObjectIdentity.from_stable_key(self.winner_target_key))

    @property
    def winner_context_key(self) -> tuple[int, ...]:
        winner = self.winner
        return () if winner is None else winner.context_key

    @property
    def hot_winner(self) -> int:
        winner = self.winner
        return int(winner is not None and winner.context_depth <= 1)

    def query_bindings(self) -> tuple[BindingEntry, ...]:
        winner = self.winner
        target = self.winner_target
        if winner is None or target is None:
            if self.status != REFERENCE_CONFLICT:
                return ()
            return (BindingEntry(
                (REFERENCE_RESOLUTION_VERSION, 9),
                self.candidate_set.stable_key(),
                space=SPACE_MEMORY,
                scope_key=self.candidate_set.reference.scope_key,
                conflict_kept=1,
            ),)
        return (
            BindingEntry(
                self.candidate_set.reference.role.stable_key(),
                target.stable_key(),
                space=SPACE_MEMORY,
                scope_key=self.candidate_set.reference.scope_key,
            ),
            BindingEntry(
                (REFERENCE_RESOLUTION_VERSION, 3),
                winner.occurrence.occurrence.stable_key(),
                space=SPACE_MEMORY,
                scope_key=self.candidate_set.reference.scope_key,
            ),
            BindingEntry(
                (REFERENCE_RESOLUTION_VERSION, 4),
                winner.context_key,
                space=SPACE_MEMORY,
                scope_key=self.candidate_set.reference.scope_key,
            ),
        )

    def query_evidence(self) -> tuple[EvidenceEntry, ...]:
        by_key = {item.stable_key(): item for item in self.evidence}
        result = []
        for record in self.evidence_records:
            if record.stance == EVIDENCE_UNKNOWN:
                continue
            graph = tuple(item for item in by_key.values()
                          if item.stable_key() == record.payload)
            if len(graph) != 1:
                raise ValueError("cross-turn H-00 evidence lost its graph payload")
            item = graph[0]
            result.append(EvidenceEntry(
                (item.source_hash, *item.source.stable_key()),
                record.hypothesis.stable_key(),
                space=SPACE_MEMORY,
                polarity=record.stance,
                trust=1,
                evidence_key=(REFERENCE_RESOLUTION_VERSION, record.evidence_id),
                scope_key=self.candidate_set.reference.scope_key,
                payload_key=item.stable_key(),
            ))
        return tuple(sorted(result, key=lambda item: item.stable_key()))

    def discourse_objects(self) -> tuple[ObjectIdentity, ...]:
        winner = self.winner
        target = self.winner_target
        if winner is None or target is None:
            return ()
        return tuple(sorted({
            target,
            winner.occurrence.occurrence,
            winner.occurrence.proof_ref,
            winner.occurrence.delivery_context,
        }, key=lambda item: item.stable_key()))

    def stable_key(self) -> tuple[int, ...]:
        return (
            REFERENCE_RESOLUTION_VERSION,
            *_pack(self.candidate_set.stable_key()),
            self.requires_resolution,
            self.status,
            len(self.candidates),
            *(value for item in self.candidates
              for value in _pack(item.stable_key())),
            len(self.evidence),
            *(value for item in self.evidence
              for value in _pack(item.stable_key())),
            *_pack(self.decision.stable_key()),
            *_pack(self.winner_hypothesis_key),
            *_pack(self.winner_target_key),
        )


@dataclass(frozen=True, slots=True)
class _PreparedReference:
    candidate_set: ReferenceCandidateSet
    candidates: tuple[CrossTurnReferenceCandidate, ...]
    evidence: tuple[TrainedReferenceEvidence, ...]
    requires_resolution: int


class TrainedCrossTurnReferenceRuntime:
    """Build and resolve discourse candidates from immutable trained graph facts."""

    def __init__(
            self,
            core: TrainedRelationGraphRuntime,
            projector: TrainedInputStructureProjector,
            occurrence_index: OccurrenceIndex,
            ) -> None:
        if not isinstance(core, TrainedRelationGraphRuntime):
            raise TypeError("cross-turn reference runtime requires trained Core")
        if not isinstance(projector, TrainedInputStructureProjector):
            raise TypeError("cross-turn reference runtime requires the input projector")
        if not isinstance(occurrence_index, OccurrenceIndex):
            raise TypeError("cross-turn reference runtime requires the Memory occurrence index")
        self.core = core
        self.projector = projector
        self.occurrence_index = occurrence_index
        self._facts = tuple(self._fact(item) for item in core.active_surface_facts())
        self._by_predicate: dict[ObjectIdentity, tuple[_CoreFact, ...]] = {}
        for fact in self._facts:
            self._by_predicate.setdefault(fact.predicate, ())
            self._by_predicate[fact.predicate] += (fact,)
        self._prepared: dict[tuple[int, ...], _PreparedReference] = {}

    def _fact(self, fact: ActiveRelationSurface) -> _CoreFact:
        source = self.core.generation_input(fact.proposition).proposition.definition.source
        bindings = tuple(sorted(
            ((item.role, item.filler) for item in fact.bindings),
            key=lambda item: (item[0].stable_key(), item[1].stable_key()),
        ))
        return _CoreFact(fact.proposition, fact.predicate, bindings, source, fact.source_hash)

    def _semantic_objects(self, values: tuple[int, ...]) -> tuple[ObjectIdentity, ...]:
        projected = self.projector.project(values)
        complete = {
            item.ref_key for item in projected.spans
            if item.start == 0 and item.end == len(values)
        }
        identities = {
            ObjectIdentity.from_stable_key(item.candidate_key)
            for item in projected.semantic_candidates
            if item.projection_kind == PROJECTION_FILLER and item.span_ref in complete
        }
        return tuple(sorted(identities, key=lambda item: item.stable_key()))

    def _occurrence_values(self, occurrence: ResponseRoleOccurrence) -> tuple[int, ...]:
        ref = self.occurrence_index.ontology.resolve(occurrence.occurrence)
        if ref is None:
            raise ValueError("cross-turn candidate occurrence is missing from Memory graph")
        record = self.occurrence_index.read(ref)
        if self.occurrence_index.ontology.identity_of(ref) != occurrence.occurrence:
            raise ValueError("cross-turn candidate occurrence identity drifted")
        return tuple(map(ord, record.surface))

    @staticmethod
    def _candidate_key(reference: ObjectIdentity, antecedent: ObjectIdentity) -> tuple[int, ...]:
        return (REFERENCE_RESOLUTION_VERSION, 1,
                *_pack(reference.stable_key()), *_pack(antecedent.stable_key()))

    @staticmethod
    def _competition_key(reference: ObjectIdentity) -> tuple[int, ...]:
        return (REFERENCE_RESOLUTION_VERSION, 2, *_pack(reference.stable_key()))

    def _context_for(
            self,
            occurrence: ResponseRoleOccurrence,
            contexts: tuple[DeliveredResponseContext, ...],
            ) -> DeliveredResponseContext:
        matches = tuple(item for item in contexts if occurrence in item.role_occurrences)
        if len(matches) != 1:
            raise ValueError("cross-turn candidate does not have one delivered context")
        return matches[0]

    def _property_evidence(
            self,
            antecedent_key: tuple[int, ...],
            current: tuple[ObjectIdentity, ...],
            target: tuple[ObjectIdentity, ...],
            source: SourceRef,
            source_hash: int,
            ) -> tuple[TrainedReferenceEvidence, ...]:
        predicate = authored_relation_identity(RELATION_PROPERTY)
        subject_role = authored_relation_role_identity(ROLE_PROPERTY_SUBJECT)
        signature_roles = tuple(authored_relation_role_identity(item) for item in (
            ROLE_PROPERTY_ATTRIBUTE, ROLE_PROPERTY_VALUE, ROLE_PROPERTY_POLARITY,
            ROLE_PROPERTY_MODALITY, ROLE_PROPERTY_INTENSITY,
        ))
        profiles: dict[ObjectIdentity, dict[tuple[int, ...], list[_CoreFact]]] = {}
        for fact in self._by_predicate.get(predicate, ()):
            subject = fact.filler(subject_role)
            signature = tuple(value for role in signature_roles
                              for value in _pack(fact.filler(role).stable_key()))
            profiles.setdefault(subject, {}).setdefault(signature, []).append(fact)
        result = []
        for left in current:
            for right in target:
                for signature in sorted(set(profiles.get(left, ()))
                                        & set(profiles.get(right, ()))):
                    for left_fact in profiles[left][signature]:
                        for right_fact in profiles[right][signature]:
                            propositions = tuple(sorted({
                                left_fact.proposition.stable_key(),
                                right_fact.proposition.stable_key(),
                            }))
                            result.append(TrainedReferenceEvidence(
                                antecedent_key, right, REFERENCE_DIMENSION_PROPERTY,
                                EVIDENCE_UNKNOWN, 0, propositions,
                                (predicate.stable_key(),), source, source_hash,
                                (REFERENCE_RESOLUTION_VERSION, 31,
                                 *_pack(left.stable_key()), *_pack(right.stable_key()),
                                 *_pack(signature)),
                            ))
        return tuple(result)

    def _time_evidence(
            self,
            antecedent_key: tuple[int, ...],
            current: tuple[ObjectIdentity, ...],
            target: tuple[ObjectIdentity, ...],
            ) -> tuple[TrainedReferenceEvidence, ...]:
        protocols = (
            (RELATION_EVENT_BEFORE, ROLE_EVENT_BEFORE_SUBJECT,
             ROLE_EVENT_BEFORE_OBJECT, EVIDENCE_REFUTE),
            (RELATION_EVENT_AFTER, ROLE_EVENT_AFTER_SUBJECT,
             ROLE_EVENT_AFTER_OBJECT, EVIDENCE_REFUTE),
            (RELATION_EVENT_SAME, ROLE_EVENT_SAME_SUBJECT,
             ROLE_EVENT_SAME_OBJECT, EVIDENCE_UNKNOWN),
            (RELATION_EVENT_UNKNOWN, ROLE_EVENT_UNKNOWN_SUBJECT,
             ROLE_EVENT_UNKNOWN_OBJECT, EVIDENCE_UNKNOWN),
        )
        current_set = set(current)
        target_set = set(target)
        result = []
        for relation, subject_kind, object_kind, stance in protocols:
            predicate = authored_relation_identity(relation)
            subject_role = authored_relation_role_identity(subject_kind)
            object_role = authored_relation_role_identity(object_kind)
            for fact in self._by_predicate.get(predicate, ()):
                subject = fact.filler(subject_role)
                object_ref = fact.filler(object_role)
                if subject not in current_set or object_ref not in target_set:
                    continue
                result.append(TrainedReferenceEvidence(
                    antecedent_key, object_ref, REFERENCE_DIMENSION_TIME,
                    stance, 0, (fact.proposition.stable_key(),),
                    (predicate.stable_key(),), fact.source, fact.source_hash,
                    (REFERENCE_RESOLUTION_VERSION, 41, relation,
                     *_pack(subject.stable_key()), *_pack(object_ref.stable_key())),
                ))
        return tuple(result)

    def _prepare(
            self,
            candidate_set: ReferenceCandidateSet,
            contexts: tuple[DeliveredResponseContext, ...],
            ) -> _PreparedReference:
        key = candidate_set.stable_key()
        cached = self._prepared.get(key)
        if cached is not None:
            return cached
        reference = candidate_set.reference
        source = SourceRef.from_stable_key(reference.source_ref[1:])
        scope = ScopeIdentity.from_stable_key(reference.scope_key)
        current_objects = self._semantic_objects(reference.values)
        refers_predicate = authored_relation_identity(RELATION_REFERS)
        from_role = authored_relation_role_identity(ROLE_REFERS_FROM)
        to_role = authored_relation_role_identity(ROLE_REFERS_TO)
        outgoing = tuple(
            fact for fact in self._by_predicate.get(refers_predicate, ())
            if fact.filler(from_role) in set(current_objects)
        )
        candidates = []
        evidence = []
        for occurrence, depth in zip(
                candidate_set.antecedents, candidate_set.context_depths, strict=True):
            context = self._context_for(occurrence, contexts)
            objects = self._semantic_objects(self._occurrence_values(occurrence))
            hypothesis = HypothesisKey(
                (REFERENCE_RESOLUTION_VERSION, 1),
                self._candidate_key(reference.occurrence, occurrence.occurrence),
                self._competition_key(reference.occurrence),
                scope,
                source,
            )
            candidates.append(CrossTurnReferenceCandidate(
                occurrence, context.stable_key(), depth, objects, hypothesis))
            targets = set(objects)
            for fact in outgoing:
                target = fact.filler(to_role)
                if target not in targets:
                    continue
                evidence.append(TrainedReferenceEvidence(
                    occurrence.stable_key(), target, REFERENCE_DIMENSION_REFERS,
                    EVIDENCE_SUPPORT, 1, (fact.proposition.stable_key(),),
                    (fact.predicate.stable_key(),), fact.source, fact.source_hash,
                    (REFERENCE_RESOLUTION_VERSION, 51,
                     *_pack(fact.stable_key())),
                ))
                if target.object_kind in {OBJECT_ENTITY, OBJECT_EVENT}:
                    dimension = (REFERENCE_DIMENSION_ENTITY
                                 if target.object_kind == OBJECT_ENTITY
                                 else REFERENCE_DIMENSION_EVENT)
                    evidence.append(TrainedReferenceEvidence(
                        occurrence.stable_key(), target, dimension,
                        EVIDENCE_UNKNOWN, 0, (fact.proposition.stable_key(),),
                        (fact.predicate.stable_key(),), fact.source, fact.source_hash,
                        (REFERENCE_RESOLUTION_VERSION, 11 + dimension,
                         *_pack(target.stable_key())),
                    ))
            evidence.extend(self._property_evidence(
                occurrence.stable_key(), current_objects, objects,
                source, reference.source_ref[0]))
            evidence.extend(self._time_evidence(
                occurrence.stable_key(), current_objects, objects))
        prepared = _PreparedReference(
            candidate_set,
            tuple(sorted(candidates, key=lambda item: item.stable_key())),
            tuple(sorted(set(evidence), key=lambda item: item.stable_key())),
            int(bool(outgoing)),
        )
        self._prepared[key] = prepared
        return prepared

    @staticmethod
    def _formation_evidence(candidate: CrossTurnReferenceCandidate,
                            timestamp_seq: int) -> EvidenceRecord:
        source = candidate.hypothesis.observation
        payload = (REFERENCE_RESOLUTION_VERSION, 6,
                   *_pack(candidate.occurrence.stable_key()))
        evidence_id = _EVIDENCE_HASHER.h63((
            candidate.hypothesis.stable_key(), EVIDENCE_UNKNOWN,
            (REFERENCE_RESOLUTION_VERSION, 6), source.stable_key(),
            timestamp_seq, payload,
        )) or 1
        return EvidenceRecord(
            evidence_id, candidate.hypothesis, EVIDENCE_UNKNOWN,
            (REFERENCE_RESOLUTION_VERSION, 6), source, timestamp_seq, payload)

    @staticmethod
    def _graph_evidence(candidate: CrossTurnReferenceCandidate,
                        graph: TrainedReferenceEvidence,
                        timestamp_seq: int) -> EvidenceRecord:
        evidence_id = _EVIDENCE_HASHER.h63((
            candidate.hypothesis.stable_key(), graph.stance,
            (REFERENCE_RESOLUTION_VERSION, graph.dimension, graph.decisive),
            graph.source.stable_key(), timestamp_seq, graph.stable_key(),
        )) or 1
        return EvidenceRecord(
            evidence_id, candidate.hypothesis, graph.stance,
            (REFERENCE_RESOLUTION_VERSION, graph.dimension, graph.decisive),
            graph.source, timestamp_seq, graph.stable_key())

    def resolve(
            self,
            candidate_set: ReferenceCandidateSet,
            contexts: tuple[DeliveredResponseContext, ...],
            *,
            core_support: frozenset[tuple[int, ...]],
            timestamp_seq: int,
            ) -> CrossTurnReferenceResolution:
        if type(core_support) is not frozenset:
            raise TypeError("cross-turn reference Core support must be a frozenset")
        if type(timestamp_seq) is not int or timestamp_seq < 0:
            raise ValueError("cross-turn reference logical sequence must be nonnegative")
        prepared = self._prepare(candidate_set, contexts)
        available = tuple(item for item in prepared.evidence
                          if item.available(core_support))
        by_antecedent = {item.occurrence.stable_key(): item
                         for item in prepared.candidates}
        ledger = HypothesisLedger()
        records = []
        for candidate in prepared.candidates:
            ledger.register(candidate.hypothesis)
            records.append(ledger.append_evidence(
                self._formation_evidence(candidate, timestamp_seq)))
        for graph in available:
            candidate = by_antecedent[graph.antecedent_key]
            records.append(ledger.append_evidence(
                self._graph_evidence(candidate, graph, timestamp_seq)))
        decision = HypothesisResolver(ledger).resolve(
            prepared.candidates[0].hypothesis,
            timestamp_seq=timestamp_seq,
        )
        decisive_targets: dict[HypothesisKey, set[tuple[int, ...]]] = {}
        for graph in available:
            if graph.decisive:
                candidate = by_antecedent[graph.antecedent_key]
                decisive_targets.setdefault(candidate.hypothesis, set()).add(
                    graph.target.stable_key())
        supported = tuple(
            candidate for candidate in prepared.candidates
            if ledger.snapshot(candidate.hypothesis).epistemic_status
            == EPISTEMIC_SUPPORTED
        )
        conflicted = any(
            ledger.snapshot(candidate.hypothesis).epistemic_status
            == EPISTEMIC_CONFLICTED
            for candidate in prepared.candidates)
        winner_hypothesis = ()
        winner_target = ()
        if not prepared.requires_resolution:
            status = REFERENCE_NOT_APPLICABLE
        elif conflicted or len(supported) > 1:
            status = REFERENCE_CONFLICT
        elif len(supported) == 1:
            targets = decisive_targets.get(supported[0].hypothesis, set())
            if (len(targets) == 1
                    and decision.adopted_hypotheses == (supported[0].hypothesis,)):
                status = REFERENCE_RESOLVED
                winner_hypothesis = supported[0].hypothesis.stable_key()
                winner_target = next(iter(targets))
            else:
                status = REFERENCE_CONFLICT if len(targets) > 1 else REFERENCE_OPEN
        else:
            status = REFERENCE_OPEN
        return CrossTurnReferenceResolution(
            candidate_set,
            prepared.candidates,
            available,
            tuple(sorted(records, key=lambda item: item.stable_key())),
            decision,
            prepared.requires_resolution,
            status,
            winner_hypothesis,
            winner_target,
        )


__all__ = [
    "CrossTurnReferenceCandidate",
    "CrossTurnReferenceResolution",
    "REFERENCE_CONFLICT",
    "REFERENCE_DIMENSION_ENTITY",
    "REFERENCE_DIMENSION_EVENT",
    "REFERENCE_DIMENSION_PROPERTY",
    "REFERENCE_DIMENSION_REFERS",
    "REFERENCE_DIMENSION_TIME",
    "REFERENCE_NOT_APPLICABLE",
    "REFERENCE_OPEN",
    "REFERENCE_RESOLVED",
    "REFERENCE_RESOLUTION_VERSION",
    "TrainedCrossTurnReferenceRuntime",
    "TrainedReferenceEvidence",
]
