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

from collections import OrderedDict
from dataclasses import dataclass, field

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
from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate,
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
from pure_integer_ai.experiments.trained_response_context import (
    DeliveredResponseContext, DeliveredFactualResponseContext,
)
from pure_integer_ai.experiments.reference_target_support import (
    REFERENCE_TARGET_SUPPORT_VERSION, ReferenceTargetSupport, reference_target_support,
)
from pure_integer_ai.experiments.trained_response_occurrence import (
    ReferenceCandidateSet,
    ResponseRoleOccurrence,
)


REFERENCE_RESOLUTION_VERSION = 91536
_MAX_SEMANTIC_OBJECT_CACHE_ENTRIES = 512
_MAX_SEMANTIC_OBJECT_CACHE_INTS = 131072

REFERENCE_DIMENSION_ENTITY = 1
REFERENCE_DIMENSION_EVENT = 2
REFERENCE_DIMENSION_PROPERTY = 3
REFERENCE_DIMENSION_TIME = 4
REFERENCE_DIMENSION_REFERS = 5
REFERENCE_DIMENSION_RELATION = 6

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
    REFERENCE_DIMENSION_RELATION,
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
    _stable_key_cache: tuple[int, ...] | None = field(
        default=None, init=False, repr=False, compare=False, hash=False)

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
        key = self._stable_key_cache
        if key is None:
            key = (
                REFERENCE_RESOLUTION_VERSION,
                *_pack(self.proposition.stable_key()),
                *_pack(self.predicate.stable_key()),
                self.source_hash,
                *_pack(self.source.stable_key()),
                len(self.bindings),
                *(value for role, filler in self.bindings
                  for value in (*_pack(role.stable_key()),
                                *_pack(filler.stable_key()))),
            )
            object.__setattr__(self, "_stable_key_cache", key)
        return key


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
    _stable_key_cache: tuple[int, ...] | None = field(
        default=None, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        _pack(self.antecedent_key)
        if not isinstance(self.target, ObjectIdentity):
            raise TypeError("reference evidence target must be an ObjectIdentity")
        if self.dimension not in _DIMENSIONS or self.stance not in _STANCES:
            raise ValueError("reference evidence dimension or stance is not registered")
        if self.decisive not in {0, 1}:
            raise ValueError("reference evidence decisive flag must be 0 or 1")
        if self.decisive and (
                self.dimension not in {
                    REFERENCE_DIMENSION_REFERS,
                    REFERENCE_DIMENSION_RELATION,
                }
                or self.stance != EVIDENCE_SUPPORT):
            raise ValueError("only supported relation evidence can be decisive")
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
        return all(key in core_support for key in self.proposition_keys)

    def stable_key(self) -> tuple[int, ...]:
        key = self._stable_key_cache
        if key is None:
            key = (
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
            object.__setattr__(self, "_stable_key_cache", key)
        return key


@dataclass(frozen=True, slots=True)
class CrossTurnReferenceCandidate:
    """A historical occurrence and the trained semantic objects found for its span."""

    occurrence: ResponseRoleOccurrence
    context_key: tuple[int, ...]
    context_depth: int
    semantic_objects: tuple[ObjectIdentity, ...]
    hypothesis: HypothesisKey
    _stable_key_cache: tuple[int, ...] | None = field(
        default=None, init=False, repr=False, compare=False, hash=False)

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
        key = self._stable_key_cache
        if key is None:
            key = (
                REFERENCE_RESOLUTION_VERSION,
                *_pack(self.occurrence.stable_key()),
                *_pack(self.context_key),
                self.context_depth,
                len(self.semantic_objects),
                *(value for item in self.semantic_objects
                  for value in _pack(item.stable_key())),
                *_pack(self.hypothesis.stable_key()),
            )
            object.__setattr__(self, "_stable_key_cache", key)
        return key


@dataclass(frozen=True, slots=True)
class CrossTurnReferenceResolution:
    """H-00/H-04 裁决：单历史 winner 或保留全部历史成员的唯一目标支持。"""

    candidate_set: ReferenceCandidateSet
    candidates: tuple[CrossTurnReferenceCandidate, ...]
    evidence: tuple[TrainedReferenceEvidence, ...]
    evidence_records: tuple[EvidenceRecord, ...]
    decision: ResolverDecision
    requires_resolution: int
    status: int
    winner_hypothesis_key: tuple[int, ...] = ()
    winner_target_key: tuple[int, ...] = ()
    target_support: ReferenceTargetSupport | None = None
    _stable_key_cache: tuple[int, ...] | None = field(
        default=None, init=False, repr=False, compare=False, hash=False)

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
        if self.target_support is not None:
            if (not isinstance(self.target_support, ReferenceTargetSupport)
                    or self.winner_hypothesis_key
                    or self.target_support.target_key != self.winner_target_key
                    or self.target_support != reference_target_support(
                        self.candidate_set, self.candidates, self.evidence,
                        self.evidence_records, self.decision)):
                raise ValueError("目标支持与当前完整候选、H-00/H-04 证据不一致")
        has_support = bool(self.winner_hypothesis_key or self.target_support is not None)
        if (self.status == REFERENCE_RESOLVED) != has_support:
            raise ValueError("resolved status and winner hypothesis differ")
        if has_support != bool(self.winner_target_key):
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
    def supporting_candidates(self) -> tuple[CrossTurnReferenceCandidate, ...]:
        """返回目标的全部采用成员；多个历史出现不伪装成某个单历史 winner。"""
        if self.target_support is None:
            winner = self.winner
            return () if winner is None else (winner,)
        keys = {item.candidate_key for item in self.target_support.members}
        return tuple(item for item in self.candidates if item.stable_key() in keys)

    @property
    def support_context_keys(self) -> tuple[tuple[int, ...], ...]:
        """生成必须恢复全部支持上下文，不能选择最近或稳定键最小的历史。"""
        return tuple(sorted({item.context_key for item in self.supporting_candidates}))

    @property
    def adoption_support_key(self) -> tuple[int, ...]:
        """目标采用证明绑定单历史 Hypothesis 或整项完整目标支持记录。"""
        return (self.winner_hypothesis_key if self.target_support is None
                else self.target_support.stable_key())

    @property
    def winner_context_key(self) -> tuple[int, ...]:
        winner = self.winner
        return () if winner is None else winner.context_key

    @property
    def hot_winner(self) -> int:
        return int(any(item.context_depth <= 1 for item in self.supporting_candidates))

    def query_bindings(self) -> tuple[BindingEntry, ...]:
        """把唯一目标、全部历史 occurrence/上下文及完整支持记录送入当前状态。"""
        supporters = self.supporting_candidates
        target = self.winner_target
        if not supporters or target is None:
            if self.status != REFERENCE_CONFLICT:
                return ()
            return (BindingEntry(
                (REFERENCE_RESOLUTION_VERSION, 9),
                self.candidate_set.stable_key(),
                space=SPACE_MEMORY,
                scope_key=self.candidate_set.reference.scope_key,
                conflict_kept=1,
            ),)
        bindings = [
            BindingEntry(
                self.candidate_set.reference.role.stable_key(),
                target.stable_key(),
                space=SPACE_MEMORY,
                scope_key=self.candidate_set.reference.scope_key,
            )]
        for supporter in supporters:
            bindings.extend((BindingEntry(
                (REFERENCE_RESOLUTION_VERSION, 3),
                supporter.occurrence.occurrence.stable_key(),
                space=SPACE_MEMORY,
                scope_key=self.candidate_set.reference.scope_key,
            ),
            BindingEntry(
                (REFERENCE_RESOLUTION_VERSION, 4),
                supporter.context_key,
                space=SPACE_MEMORY,
                scope_key=self.candidate_set.reference.scope_key,
            )))
        if self.target_support is None:
            return tuple(bindings)
        bindings.append(BindingEntry(
            (REFERENCE_TARGET_SUPPORT_VERSION, 1), self.target_support.stable_key(),
            space=SPACE_MEMORY, scope_key=self.candidate_set.reference.scope_key))
        return tuple(sorted(set(bindings), key=lambda item: item.stable_key()))

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
        """显式引用全部目标支持历史对象，不能仅保留某个历史 occurrence。"""
        supporters = self.supporting_candidates
        target = self.winner_target
        if not supporters or target is None:
            return ()
        return tuple(sorted({
            target,
            *(obj for item in supporters for obj in (
                item.occurrence.occurrence, item.occurrence.proof_ref,
                item.occurrence.delivery_context)),
        }, key=lambda item: item.stable_key()))

    def stable_key(self) -> tuple[int, ...]:
        key = self._stable_key_cache
        if key is None:
            key = (
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
                *((REFERENCE_TARGET_SUPPORT_VERSION, *_pack(self.target_support.stable_key()))
                  if self.target_support is not None else ()),
            )
            object.__setattr__(self, "_stable_key_cache", key)
        return key


@dataclass(frozen=True, slots=True)
class _PreparedReference:
    candidate_set: ReferenceCandidateSet
    candidates: tuple[CrossTurnReferenceCandidate, ...]
    evidence: tuple[TrainedReferenceEvidence, ...]
    requires_resolution: int
    contexts: tuple[DeliveredResponseContext, ...] = field(
        repr=False, compare=False, hash=False)


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
        # The same unresolved input role can be resolved against different
        # current-role projections when a caller supplies the relation's
        # observed fillers.  Keep that projection in the cache key; otherwise
        # a fallback value-based preparation could silently be reused.
        self._prepared: OrderedDict[object, _PreparedReference] = OrderedDict()
        self._resolved: OrderedDict[
            tuple[int, tuple[int, ...], int],
            tuple[_PreparedReference, CrossTurnReferenceResolution],
        ] = OrderedDict()
        self._semantic_objects_cache: OrderedDict[
            tuple[int, ...], tuple[tuple[ObjectIdentity, ...], int]
        ] = OrderedDict()
        self._semantic_objects_cache_ints = 0

    def clear_context_cache(self) -> None:
        """Drop all preparations tied to the previous Memory dialogue view."""
        self._prepared.clear()
        self._resolved.clear()
        self._semantic_objects_cache.clear()
        self._semantic_objects_cache_ints = 0

    @staticmethod
    def _trim_cache(cache: OrderedDict, limit: int) -> None:
        while len(cache) > limit:
            cache.popitem(last=False)

    def _fact(self, fact: ActiveRelationSurface) -> _CoreFact:
        source = self.core.generation_input(fact.proposition).proposition.definition.source
        bindings = tuple(sorted(
            ((item.role, item.filler) for item in fact.bindings),
            key=lambda item: (item[0].stable_key(), item[1].stable_key()),
        ))
        return _CoreFact(fact.proposition, fact.predicate, bindings, source, fact.source_hash)

    def _semantic_objects(self, values: tuple[int, ...]) -> tuple[ObjectIdentity, ...]:
        cached = self._semantic_objects_cache.get(values)
        if cached is not None:
            self._semantic_objects_cache.move_to_end(values)
            return cached[0]
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
        result = tuple(sorted(identities, key=lambda item: item.stable_key()))
        cache_ints = len(values) + sum(
            len(item.stable_key()) for item in result)
        if cache_ints <= _MAX_SEMANTIC_OBJECT_CACHE_INTS:
            while (self._semantic_objects_cache
                   and (len(self._semantic_objects_cache)
                        >= _MAX_SEMANTIC_OBJECT_CACHE_ENTRIES
                        or self._semantic_objects_cache_ints + cache_ints
                        > _MAX_SEMANTIC_OBJECT_CACHE_INTS)):
                _old_values, (_old_result, old_ints) = (
                    self._semantic_objects_cache.popitem(last=False))
                self._semantic_objects_cache_ints -= old_ints
            self._semantic_objects_cache[values] = (result, cache_ints)
            self._semantic_objects_cache_ints += cache_ints
        return result

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

    def semantic_objects_for(self, occurrence, contexts):
        """事实角色消费原交付整数槽；通用角色保留原受训词形投影。

        已有事实槽缺失或proof/序/角色/Hypothesis漂移时拒绝，不用表面
        同名对象集合弥补。完整角色来源仍由原Occurrence及交付图验证。
        """
        context = self._context_for(occurrence, contexts)
        if not isinstance(context, DeliveredFactualResponseContext):
            return self._semantic_objects(self._occurrence_values(occurrence))
        if not 0 <= occurrence.ordinal < len(context.slots):
            raise ValueError('factual antecedent has no original delivery slot')
        slot = context.slots[occurrence.ordinal]
        if (slot.role != occurrence.role or context.proof_ref != occurrence.proof_ref
                or context.source_ref != occurrence.source_ref or context.scope_key != occurrence.scope_key
                or context.input_observation != occurrence.observation
                or (slot.hypothesis or context.hypothesis) != occurrence.hypothesis):
            raise ValueError('factual antecedent differs from its original typed slot')
        return (slot.filler,)

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
            current_objects: tuple[ObjectIdentity, ...] | None = None,
            relation_frame: OpenRelationCandidate | None = None,
            current_role_fillers: tuple[
                tuple[tuple[int, ...], ObjectIdentity], ...] = (),
            relation_answer_role: tuple[int, ...] = (),
            reference_scope=None,
            ) -> _PreparedReference:
        current_key = None if current_objects is None else tuple(
            item.stable_key() for item in current_objects)
        if relation_frame is not None and not isinstance(
                relation_frame, OpenRelationCandidate):
            raise TypeError("reference relation frame must be an open Core frame")
        if (type(current_role_fillers) is not tuple
                or any(type(item) is not tuple or len(item) != 2
                       or type(item[0]) is not tuple
                       or not isinstance(item[1], ObjectIdentity)
                       for item in current_role_fillers)
                or current_role_fillers != tuple(sorted(
                    set(current_role_fillers),
                    key=lambda item: (item[0], item[1].stable_key())))):
            raise ValueError("reference role fillers must be canonical")
        relation_key = (None if relation_frame is None
                        else relation_frame.stable_key())
        filler_key = tuple((role, filler.stable_key())
                           for role, filler in current_role_fillers)
        if (type(relation_answer_role) is not tuple
                or any(type(value) is not int or value < 0
                       for value in relation_answer_role)):
            raise ValueError("reference relation answer role must be an integer key")
        key = (id(contexts), candidate_set.stable_key(), current_key,
               relation_key, filler_key, relation_answer_role,
               None if reference_scope is None else reference_scope.stable_key())
        cached = self._prepared.get(key)
        if cached is not None:
            self._prepared.move_to_end(key)
            return cached
        reference = candidate_set.reference
        source = SourceRef.from_stable_key(reference.source_ref[1:])
        scope = ScopeIdentity.from_stable_key(reference.scope_key)
        if current_objects is None:
            current_objects = self._semantic_objects(reference.values)
        else:
            current_objects = tuple(sorted(set(current_objects),
                                           key=lambda item: item.stable_key()))
        refers_predicate = authored_relation_identity(RELATION_REFERS)
        from_role = authored_relation_role_identity(ROLE_REFERS_FROM)
        to_role = authored_relation_role_identity(ROLE_REFERS_TO)
        outgoing = tuple(
            fact for fact in self._by_predicate.get(refers_predicate, ())
            if fact.filler(from_role) in set(current_objects)
        )
        relation_role_fillers = dict(current_role_fillers)
        relation_support = []
        if relation_frame is not None:
            relation_predicate = ObjectIdentity.from_stable_key(
                relation_frame.frame.predicate)
            relation_roles = set(relation_frame.frame.roles)
            reference_role = candidate_set.reference.role.stable_key()
            expected_fact_roles = relation_roles | (
                {relation_answer_role} if relation_answer_role else set())
            if (reference_role in relation_roles
                    and (not relation_answer_role
                         or relation_answer_role not in relation_roles)
                    and set(relation_role_fillers) == relation_roles - {reference_role}):
                for fact in self._by_predicate.get(relation_predicate, ()):
                    if (fact.proposition.stable_key()
                            != relation_frame.frame.proposition):
                        continue
                    fact_roles = {role.stable_key(): filler
                                  for role, filler in fact.bindings}
                    if (set(fact_roles) != expected_fact_roles
                            or any(fact_roles.get(role) != filler
                                   for role, filler in relation_role_fillers.items())):
                        continue
                    target = fact_roles[reference_role]
                    predicate_key = fact.predicate.stable_key()
                    for occurrence, _depth in zip(
                            candidate_set.antecedents,
                            candidate_set.context_depths, strict=True):
                        if reference_scope is not None and not reference_scope.permits(
                                self._context_for(occurrence, contexts)):
                            continue
                        antecedent_objects = set(self.semantic_objects_for(occurrence, contexts))
                        if target not in antecedent_objects:
                            continue
                        relation_support.append(TrainedReferenceEvidence(
                            occurrence.stable_key(), target,
                            REFERENCE_DIMENSION_RELATION,
                            EVIDENCE_SUPPORT, 1,
                            (fact.proposition.stable_key(),),
                            (predicate_key,), fact.source, fact.source_hash,
                            (REFERENCE_RESOLUTION_VERSION, 61,
                             *_pack(relation_frame.stable_key()),
                             *_pack(target.stable_key()),
                             *(() if reference_scope is None else
                               _pack(reference_scope.stable_key()))),
                        ))
        candidates = []
        evidence = []
        for occurrence, depth in zip(
                candidate_set.antecedents, candidate_set.context_depths, strict=True):
            context = self._context_for(occurrence, contexts)
            objects = self.semantic_objects_for(occurrence, contexts)
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
                if (target not in targets
                        or occurrence.role != candidate_set.reference.role):
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
        evidence.extend(relation_support)
        relation_scope_authorized = bool(
            relation_frame is not None
            and candidate_set.reference.role.stable_key()
            in set(relation_frame.frame.roles)
            and (not relation_answer_role
                 or relation_answer_role not in relation_frame.frame.roles)
            and set(relation_role_fillers)
            == set(relation_frame.frame.roles)
            - {candidate_set.reference.role.stable_key()})
        prepared = _PreparedReference(
            candidate_set,
            tuple(sorted(candidates, key=lambda item: item.stable_key())),
            tuple(sorted(set(evidence), key=lambda item: item.stable_key())),
            int(bool(outgoing or relation_scope_authorized)),
            contexts,
        )
        self._prepared[key] = prepared
        self._prepared.move_to_end(key)
        self._trim_cache(self._prepared, 32)
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
            current_objects: tuple[ObjectIdentity, ...] | None = None,
            relation_frame: OpenRelationCandidate | None = None,
            current_role_fillers: tuple[
                tuple[tuple[int, ...], ObjectIdentity], ...] = (),
            relation_answer_role: tuple[int, ...] = (),
            reference_scope=None,
            ) -> CrossTurnReferenceResolution:
        if type(core_support) is not frozenset:
            raise TypeError("cross-turn reference Core support must be a frozenset")
        if type(timestamp_seq) is not int or timestamp_seq < 0:
            raise ValueError("cross-turn reference logical sequence must be nonnegative")
        prepared = self._prepare(
            candidate_set, contexts, current_objects,
            relation_frame, current_role_fillers, relation_answer_role,
            reference_scope)
        available_indices = tuple(
            index for index, item in enumerate(prepared.evidence)
            if item.available(core_support))
        resolution_key = (id(prepared), available_indices, timestamp_seq)
        cached_resolution = self._resolved.get(resolution_key)
        if (cached_resolution is not None
                and cached_resolution[0] is prepared):
            self._resolved.move_to_end(resolution_key)
            return cached_resolution[1]
        available = tuple(prepared.evidence[index]
                          for index in available_indices)
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
        target_support = None
        if not prepared.requires_resolution:
            status = REFERENCE_NOT_APPLICABLE
        elif conflicted:
            status = REFERENCE_CONFLICT
        elif len(supported) > 1:
            target_support = reference_target_support(
                candidate_set, prepared.candidates, available, tuple(records), decision)
            status = REFERENCE_CONFLICT if target_support is None else REFERENCE_RESOLVED
            if target_support is not None:
                winner_target = target_support.target_key
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
        resolution = CrossTurnReferenceResolution(
            candidate_set,
            prepared.candidates,
            available,
            tuple(sorted(records, key=lambda item: item.stable_key())),
            decision,
            prepared.requires_resolution,
            status,
            winner_hypothesis,
            winner_target,
            target_support,
        )
        self._resolved[resolution_key] = (prepared, resolution)
        self._resolved.move_to_end(resolution_key)
        self._trim_cache(self._resolved, 64)
        return resolution


__all__ = [
    "CrossTurnReferenceCandidate",
    "CrossTurnReferenceResolution",
    "REFERENCE_CONFLICT",
    "REFERENCE_DIMENSION_ENTITY",
    "REFERENCE_DIMENSION_EVENT",
    "REFERENCE_DIMENSION_PROPERTY",
    "REFERENCE_DIMENSION_RELATION",
    "REFERENCE_DIMENSION_REFERS",
    "REFERENCE_DIMENSION_TIME",
    "REFERENCE_NOT_APPLICABLE",
    "REFERENCE_OPEN",
    "REFERENCE_RESOLVED",
    "REFERENCE_RESOLUTION_VERSION",
    "TrainedCrossTurnReferenceRuntime",
    "TrainedReferenceEvidence",
]
