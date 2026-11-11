"""Read-only join from active Core H-05 premises to unresolved rule candidates."""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.relation_algebra import RelationAlgebra
from pure_integer_ai.cognition.shared.relation_closure import ActiveRelationClosureConsumer
from pure_integer_ai.cognition.shared.relation_composition import (
    PendingRelationComposition, RelationCompositionWitness,
)
from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord
from pure_integer_ai.cognition.shared.query_state import EvidenceEntry, QueryState, SPACE_CORE
from pure_integer_ai.cognition.shared.typed_relation import ActiveSupportedRelationFact
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.trained_query_evidence import (
    core_query_evidence, current_query_evidence,
)


@dataclass(frozen=True, slots=True)
class RelationCompositionBudget:
    max_witnesses: int = 64
    max_active_facts: int = 512
    max_rule_pairs: int = 4096
    max_candidates: int = 512
    max_history_events_per_fact: int = 256
    max_evidence_key_values: int = 65536
    max_current_evidence_per_fact: int = 64
    max_pending_proof_values: int = 131072
    max_verified_proof_values: int = 262144

    def __post_init__(self) -> None:
        if any(type(getattr(self, field)) is not int or getattr(self, field) <= 0
               for field in self.__dataclass_fields__):
            raise ValueError("composition budgets must be positive integers")


def validate_relation_composition_provenance(
        witnesses: tuple[RelationCompositionWitness, ...], metadata: dict | None) -> None:
    if metadata is not None and any(
            item.source_identity != metadata["source_identity"]
            or item.license_identity != metadata["license_identity"] for item in witnesses):
        raise ValueError("composition and question graphs have different source permission")


def pending_core_compositions(
        core: TrainedRelationGraphRuntime,
        witnesses: tuple[RelationCompositionWitness, ...],
        *, budget: RelationCompositionBudget = RelationCompositionBudget(),
        ) -> tuple[PendingRelationComposition, ...]:
    """Use current active snapshots, not graph-only recovery or teacher rows."""
    if not isinstance(core, TrainedRelationGraphRuntime):
        raise TypeError("composition requires the actual read-only Core runtime")
    if (type(witnesses) is not tuple
            or any(not isinstance(item, RelationCompositionWitness) for item in witnesses)):
        raise TypeError("composition witnesses must be typed integer graph records")
    if not isinstance(budget, RelationCompositionBudget):
        raise TypeError("composition budget must be explicit typed limits")
    if len(witnesses) > budget.max_witnesses:
        raise ValueError("composition witness budget exceeded")
    if not witnesses:
        return ()
    allowed_relations = frozenset(
        relation for witness in witnesses
        for relation in (witness.rule.first_relation, witness.rule.second_relation))
    relevant = []
    for item in core.active_propositions():
        if item.definition.predicate not in allowed_relations:
            continue
        relevant.append(item)
        if len(relevant) > budget.max_active_facts:
            raise ValueError("composition active fact budget exceeded")
    active = tuple(sorted((ActiveSupportedRelationFact(
        item.definition, source.snapshot,
        core.owner.learning.engine.definition(source.hypothesis))
        for item in relevant
        for source in (core.generation_input(item.definition.proposition),)
    ), key=lambda item: item.definition.proposition.stable_key()))
    available = {item.definition.proposition: item for item in active}
    registry = core.context.scoped_identity_store.registry
    current_evidence = {}
    for item in active:
        proposition = item.definition.proposition
        history = core.evidence_history(proposition)
        if len(history) > budget.max_history_events_per_fact:
            raise ValueError("composition premise history budget exceeded")
        projected = current_query_evidence(
            core_query_evidence(proposition, history, registry))
        if len(projected) > budget.max_current_evidence_per_fact:
            raise ValueError("composition premise current Evidence count budget exceeded")
        by_stance = {
            stance: tuple(sorted(record.evidence_id
                                 for entry in projected if entry.polarity == stance
                                 for record in (EvidenceRecord.from_stable_key(
                                     entry.payload_key),)))
            for stance in (1, 2, 3)
        }
        if (by_stance[1] != tuple(sorted(item.snapshot.support_evidence_ids))
                or by_stance[2] != tuple(sorted(item.snapshot.refute_evidence_ids))
                or by_stance[3] != tuple(sorted(item.snapshot.unknown_evidence_ids))):
            raise RuntimeError("Core active snapshot and projected Evidence disagree")
        entry_keys = tuple(sorted(entry.stable_key() for entry in projected))
        if any(len(key) > budget.max_evidence_key_values for key in entry_keys):
            raise ValueError("composition premise Evidence key budget exceeded")
        current_evidence[proposition] = entry_keys
    pending = {}
    pair_checks = 0
    for witness in witnesses:
        consumer = ActiveRelationClosureConsumer(
            core.owner.semantic_graph, core.owner.candidate_graph,
            core.owner.relation_protocol, witness.schemas,
            engine=core.owner.learning.engine)
        selected, provenance = [], {}
        for item in active:
            if item.definition.predicate not in {
                    witness.rule.first_relation, witness.rule.second_relation}:
                continue
            if not set(item.candidate_definition.forming_sources) <= set(witness.authorized_forming_sources):
                continue
            restored = consumer.require_proposition(item.definition.proposition)
            if restored.as_algebra_fact() != available[item.definition.proposition]:
                raise RuntimeError("Core and R-00 active snapshots disagree")
            selected.append(item)
            provenance[item.definition.proposition] = item.candidate_definition.forming_sources
        first_count = sum(item.definition.predicate == witness.rule.first_relation
                          for item in selected)
        second_count = sum(item.definition.predicate == witness.rule.second_relation
                           for item in selected)
        pair_checks += first_count * second_count
        if pair_checks > budget.max_rule_pairs:
            raise ValueError("composition rule pair budget exceeded")
        for candidate in RelationAlgebra(witness.schemas).derive_active_candidates(
                tuple(selected), (witness.rule,)):
            if len(candidate.premises) != 2:
                continue
            if any(not set(provenance[item.proposition])
                   <= set(witness.authorized_forming_sources)
                   for item in candidate.premises):
                continue
            pending_item = PendingRelationComposition(
                witness, candidate, tuple(current_evidence[item.proposition]
                                          for item in candidate.premises))
            if len(pending_item.stable_key()) > budget.max_pending_proof_values:
                raise ValueError("composition pending proof size budget exceeded")
            pending[pending_item.stable_key()] = pending_item
            if len(pending) > budget.max_candidates:
                raise ValueError("composition candidate budget exceeded")
    return tuple(pending[key] for key in sorted(pending))


def pending_composition_query_evidence(
        pending: tuple[PendingRelationComposition, ...], state: QueryState,
        ) -> tuple[EvidenceEntry, ...]:
    """Record a candidate only after its exact support premises enter this frontier."""
    current = current_query_evidence(state.evidence)
    by_proposition = {}
    for entry in current:
        # Language projections can retain UNKNOWN rows for the same
        # proposition. They do not belong to its H-00 ledger and cannot
        # supply or supersede premise truth. Any additional support/refute
        # row still makes the exact ledger comparison fail closed.
        if (entry.space == SPACE_CORE
                and (entry.evidence_key[:1] == (1,) or entry.polarity != 3)):
            by_proposition.setdefault(entry.hypothesis_key, []).append(entry.stable_key())
    result = []
    for item in pending:
        for premise, expected in zip(item.candidate.premises,
                                     item.premise_evidence_keys, strict=True):
            if tuple(sorted(by_proposition.get(premise.proposition.stable_key(), ()))) != expected:
                break
        else:
            proof = item.stable_key()
            result.append(EvidenceEntry(
                item.witness.source.stable_key(), proof, space=SPACE_CORE,
                polarity=3, trust=1, evidence_key=proof,
                scope_key=state.query_key, payload_key=proof))
    return tuple(sorted(result, key=lambda item: item.stable_key()))
