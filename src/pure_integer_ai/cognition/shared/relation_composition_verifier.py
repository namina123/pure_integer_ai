"""Independent, licensed join verification followed by request-local H-05.

The verifier never calls RelationAlgebra, consumes a teacher conclusion or
writes the frozen Core. Its inputs are exact current premise facts/evidence
and an independently restored set of authorized integer rule witnesses.
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.evidence_candidate import (
    ActiveEvidenceCandidate, CandidateBinding, CandidatePrediction,
    CandidateVerification, EvidenceCandidateDefinition, EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord, HypothesisKey, HypothesisLedger
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ResolverDecision, _snapshot_key, _snapshot_from_key,
)
from pure_integer_ai.cognition.shared.identity import SourceRef, concept_identity
from pure_integer_ai.cognition.shared.relation_composition import (
    PendingRelationComposition, RelationCompositionWitness, _entry_fields,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import proposition_identity
from pure_integer_ai.cognition.shared.typed_relation import ActiveSupportedRelationFact
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record


COMPOSITION_VERIFIED = 92180
COMPOSITION_DECLARATION = 92181
COMPOSITION_VERIFIER_TRACE = 92182
COMPOSITION_VERIFICATION_ROLE = (92183, 1)
COMPOSITION_DECLARATION_V2 = 92186
COMPOSITION_VERIFIED_V2 = 92190


@dataclass(frozen=True, slots=True)
class VerifiedRelationComposition:
    """A derived claim with full H-00/H-04 evidence, local to this query."""

    pending: PendingRelationComposition
    query_key: tuple[int, ...]
    active: ActiveEvidenceCandidate
    history: tuple[EvidenceRecord, ...]
    verifier_trace: tuple[int, ...]

    def __post_init__(self):
        definition = composition_candidate_definition(self.pending, self.query_key,
                                                       version=self.declaration_version)
        protocol = _candidate_protocol(self.pending.witness.source)
        if (self.active.definition != definition
                or self.active.hypothesis != definition.hypothesis(protocol)
                or type(self.history) is not tuple or len(self.history) != 2
                or self.verifier_trace != pack_record(COMPOSITION_VERIFIER_TRACE,
                    self.query_key, self.pending.stable_key(), (0, 1))):
            raise ValueError("derived H-05 declaration or verifier trace differs")
        ledger = HypothesisLedger()
        ledger.register(self.active.hypothesis)
        for record in self.history:
            ledger.append_evidence(record)
        restored = EvidenceCandidateEngine.from_history(protocol,
            definitions=(definition,), ledger=ledger, decisions=(self.active.decision,))
        if restored.active(self.active.hypothesis) != self.active:
            raise ValueError("derived H-05 snapshot does not restore from its full history")
        recognition = restored.recognition_history(self.active.hypothesis)
        if (len(recognition) != 1 or recognition[0].verification.stance != 1
                or recognition[0].verification.trace != self.verifier_trace
                or recognition[0].prediction.event_key != self.query_key
                or recognition[0].prediction.predicted != self.claim):
            raise ValueError("derived H-05 support lacks exact independent recognition")

    @property
    def claim(self):
        return self.active.definition.candidate

    @property
    def declaration_version(self):
        if self.active.definition == composition_candidate_definition(self.pending, self.query_key, version=2):
            return 2
        if self.active.definition == composition_candidate_definition(self.pending, self.query_key, version=1):
            return 1
        raise ValueError("derived declaration identity version is not recognized")

    def stable_key(self) -> tuple[int, ...]:
        return pack_record(
            COMPOSITION_VERIFIED if self.declaration_version == 1 else COMPOSITION_VERIFIED_V2,
            self.pending.stable_key(), self.query_key,
            self.active.definition.stable_key(), self.active.hypothesis.stable_key(),
            _snapshot_key(self.active.snapshot), self.active.decision.stable_key(),
            pack_record(COMPOSITION_VERIFIED, *(item.stable_key() for item in self.history)),
            self.verifier_trace,
        )

    @classmethod
    def from_stable_key(cls, key):
        if key[:1] not in {(COMPOSITION_VERIFIED,), (COMPOSITION_VERIFIED_V2,)}:
            raise ValueError("verified composition proof version is unrecognized")
        fields = unpack_record(key, key[0])
        if len(fields) != 8:
            raise ValueError("verified composition proof is incomplete")
        result = cls(PendingRelationComposition.from_stable_key(fields[0]), fields[1],
            ActiveEvidenceCandidate(EvidenceCandidateDefinition.from_stable_key(fields[2]),
                HypothesisKey.from_stable_key(fields[3]), _snapshot_from_key(fields[4]),
                ResolverDecision.from_stable_key(fields[5])),
            tuple(EvidenceRecord.from_stable_key(item)
                  for item in unpack_record(fields[6], COMPOSITION_VERIFIED)), fields[7])
        if result.stable_key() != key:
            raise ValueError("verified composition proof is not canonical")
        return result


def composition_candidate_definition(pending, query_key, *, version=2):
    """Declare complete result roles; the declaration itself supplies no truth."""
    witness, candidate = pending.witness, pending.candidate
    relation_bindings = tuple(sorted(candidate.bindings, key=lambda item: item.role.stable_key()))
    result_schema = next(item for item in witness.schemas if item.relation == candidate.relation)
    content = pack_record(COMPOSITION_DECLARATION, candidate.relation.stable_key(), *(
        pack_record(COMPOSITION_DECLARATION, item.role.stable_key(), item.filler.stable_key(),
                    (item.ordinal,)) for item in relation_bindings))
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("composition declaration version is unrecognized")
    # V1 is retained solely for immutable historical evidence. V2 declares
    # the request-local object without putting role content/premise bodies
    # into Proposition identity; those remain complete candidate/proof fields.
    declaration = (pack_record(COMPOSITION_DECLARATION, query_key, witness.rule.rule.stable_key(), content,
        *(item.proposition.stable_key() for item in candidate.premises)) if version == 1
        else pack_record(COMPOSITION_DECLARATION_V2, query_key, witness.rule.rule.stable_key()))
    claim = proposition_identity(witness.source, declaration)
    fields = tuple(concept_identity((COMPOSITION_DECLARATION, index),
        owner=witness.source.owner, versions=witness.source.versions) for index in range(1, 5))
    bindings = [CandidateBinding(fields[0], candidate.relation),
                CandidateBinding(fields[1], result_schema.schema)]
    for index, item in enumerate(relation_bindings):
        bindings.extend((CandidateBinding(fields[2], item.role, index),
                         CandidateBinding(fields[3], item.filler, index)))
    return EvidenceCandidateDefinition(claim, content, tuple(bindings),
                                       (witness.source,))


class IndependentCompositionVerifier:
    """Check both premise roles and the instruction directly, never its candidate generator."""

    def __init__(self, authorized: tuple[RelationCompositionWitness, ...]):
        if (type(authorized) is not tuple or not authorized
                or any(not isinstance(item, RelationCompositionWitness) for item in authorized)
                or len(set(item.stable_key() for item in authorized)) != len(authorized)):
            raise ValueError("composition verifier needs exact unique authorized witnesses")
        self.authorized = authorized

    def verify(self, prediction: CandidatePrediction, pending: PendingRelationComposition,
               query_key: tuple[int, ...], premises: tuple[ActiveSupportedRelationFact, ...],
               evidence_keys: tuple[tuple[tuple[int, ...], ...], ...],
               *, conclusion_refuted: bool = False, declaration_version=2) -> CandidateVerification:
        witness, candidate, rule = pending.witness, pending.candidate, pending.witness.rule
        if len(premises) != 2 or len(evidence_keys) != 2:
            raise ValueError("independent composition requires exactly two current premises")
        valid = (witness in self.authorized
                 and evidence_keys == pending.premise_evidence_keys
                 and prediction.predicted == composition_candidate_definition(
                     pending, query_key, version=declaration_version).candidate
                 and prediction.hypothesis.candidate_key
                     == composition_candidate_definition(pending, query_key, version=declaration_version).stable_key()
                 and prediction.event_key == query_key
                 and prediction.visible_inputs == tuple(item.proposition for item in candidate.premises)
                 and not conclusion_refuted)
        schemas = {item.relation: item for item in witness.schemas}
        for fact, premise, group in zip(premises, candidate.premises, evidence_keys, strict=True):
            valid = valid and (
                fact.definition.proposition == premise.proposition
                and fact.snapshot.hypothesis == premise.hypothesis
                and fact.snapshot.support_evidence_ids == premise.support_evidence_ids
                and fact.candidate_definition is not None
                and bool(fact.candidate_definition.forming_sources)
                and set(fact.candidate_definition.forming_sources)
                    <= set(witness.authorized_forming_sources))
            if fact.definition.predicate not in schemas:
                valid = False
                continue
            schemas[fact.definition.predicate].validate_definition(fact.definition)
            records = tuple(EvidenceRecord.from_stable_key(_entry_fields(key)[4]) for key in group)
            valid = valid and all(record.hypothesis == premise.hypothesis for record in records)
            for stance, ids in ((1, fact.snapshot.support_evidence_ids),
                               (2, fact.snapshot.refute_evidence_ids),
                               (3, fact.snapshot.unknown_evidence_ids)):
                valid = valid and tuple(sorted(record.evidence_id for record in records
                    if record.stance == stance)) == tuple(sorted(ids))
        if len(premises) == 2:
            first, second = (item.definition for item in premises)

            def filler(definition, role):
                values = tuple(item.filler for item in definition.bindings
                               if item.role == role and item.ordinal == 0)
                return values[0] if len(values) == 1 else None

            start = filler(first, rule.first_input_role)
            join = filler(first, rule.first_join_role)
            end = filler(second, rule.second_output_role)
            valid = valid and (
                first.predicate == rule.first_relation
                and second.predicate == rule.second_relation
                and join is not None and join == filler(second, rule.second_join_role)
                and start is not None and end is not None
                and {item.role: (item.filler, item.ordinal) for item in candidate.bindings} == {
                    rule.result_input_role: (start, 0), rule.result_output_role: (end, 0)})
        expected_schemas = tuple(schemas[relation].schema for relation in
            (rule.first_relation, rule.second_relation, rule.result_relation))
        # RelationAlgebra records the two premise schemas plus result schema.
        valid = valid and candidate.schemas == expected_schemas
        schemas[candidate.relation].validate_bindings(candidate.bindings)
        trace = pack_record(COMPOSITION_VERIFIER_TRACE, query_key, pending.stable_key(),
                            (int(conclusion_refuted), int(valid)))
        authority = concept_identity((COMPOSITION_VERIFIER_TRACE, 1),
            owner=witness.source.owner, versions=witness.source.versions)
        return CandidateVerification(1 if valid else 3,
            (COMPOSITION_VERIFIER_TRACE, 1 if valid else 3), prediction.observation,
            authority, (COMPOSITION_VERIFIER_TRACE, 1), trace)


def _candidate_protocol(source):
    aggregate = SourceRef(COMPOSITION_VERIFIED, source.source_id, 0, source.owner, source.versions)
    return EvidenceCandidateProtocol((COMPOSITION_VERIFIED, 1), (COMPOSITION_VERIFIED, 3),
                                     aggregate, document_scope(aggregate), 1)


def admit_verified_composition(pending, query_key, premises, evidence_keys, authorized,
                              *, conclusion_refuted=False, declaration_version=2):
    """Freeze an UNKNOWN declaration, independently verify, then run real H-04."""
    definition = composition_candidate_definition(pending, query_key, version=declaration_version)
    source = pending.witness.source
    observation = SourceRef(COMPOSITION_VERIFIER_TRACE, source.source_id, 0,
                            source.owner, source.versions)
    engine = EvidenceCandidateEngine(_candidate_protocol(source))
    hypothesis = engine.register(definition)
    prediction = engine.predict(hypothesis, observation=observation,
        scope=document_scope(observation), event_key=query_key,
        visible_inputs=tuple(item.proposition for item in pending.candidate.premises),
        predicted=definition.candidate)
    verdict = IndependentCompositionVerifier(authorized).verify(
        prediction, pending, query_key, premises, evidence_keys,
        conclusion_refuted=conclusion_refuted, declaration_version=declaration_version)
    engine.reveal(prediction, verdict, timestamp_seq=2)
    engine.resolve(hypothesis, timestamp_seq=3)
    active = engine.active(hypothesis)
    if active is None:
        return None
    return VerifiedRelationComposition(pending, query_key, active,
        engine.ledger.evidence_history(hypothesis), verdict.trace)
