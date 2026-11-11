"""Query-local derived admission; frozen Core and its evidence remain read-only."""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.cognition.shared.relation_composition_verifier import (
    COMPOSITION_VERIFICATION_ROLE, VerifiedRelationComposition, admit_verified_composition,
)
from pure_integer_ai.cognition.shared.typed_relation import ActiveSupportedRelationFact
from pure_integer_ai.experiments.trained_core_relation_composition import pending_composition_query_evidence
from pure_integer_ai.experiments.trained_query_evidence import core_query_evidence, current_query_evidence
from pure_integer_ai.experiments.relation_runtime_phase import observed_relation_phase


@dataclass(frozen=True, slots=True)
class CompositionQueryOutput:
    verified: VerifiedRelationComposition
    fact: object
    generated: object
    conclusion_evidence: tuple[EvidenceEntry, ...]

    @observed_relation_phase("response_plan")
    def response_plan(self, state):
        from pure_integer_ai.experiments.generation_organization import (
            bind_query_evidence, plan_from_active_fact,
        )
        plan = plan_from_active_fact(self.fact, self.generated)
        if plan is None:
            return None
        return bind_query_evidence(plan, state)

    def generation_value(self):
        value = self.generated
        return {
            "kind": 5, "surface_values": list(map(ord, value.surface)),
            "frame_proposition": list(value.frame_proposition.stable_key()),
            "slot_count": value.slot_count, "connector": list(value.connector.stable_key()),
            "representations": [list(item.stable_key()) for item in value.representations],
            "trace": list(value.trace), "structure_plan": list(value.structure_plan.stable_key()),
            "verified_composition": list(self.verified.stable_key()),
            "core_claims_written": 0, "request_local_derivation": 1,
        }


def _conclusion_candidates(core, pending, budget):
    """按结论的谓词和完整角色内容找全部来源，不预裁掉反证。"""
    graph, conclusion = core.owner.semantic_graph, pending.candidate
    bindings = tuple(sorted(conclusion.bindings, key=lambda row: row.role.stable_key()))
    rows = (graph.lookup_atomic_by_binding(conclusion.relation,
            bindings[0].role, bindings[0].filler) if bindings else
            graph.lookup_atomic_by_predicate(conclusion.relation))
    if len(rows) > budget.max_active_facts:
        raise ValueError("composition conclusion check fact budget exceeded")
    return rows


def _conclusion_refuted(core, pending, budget, *, candidates=None):
    """当前任一来源的同内容反证都阻断派生，不借 active-only 视图绕过。"""
    engine = core.owner.learning.engine
    rows = _conclusion_candidates(core, pending, budget) if candidates is None else candidates
    result = frozenset((item.role, item.filler, item.ordinal) for item in pending.candidate.bindings)
    for row in rows:
        atomic = row.definition
        if frozenset((item.role, item.filler, item.ordinal) for item in atomic.bindings) != result:
            continue
        hypothesis = core.owner.learning.hypothesis_for_candidate(atomic.proposition)
        if engine.ledger.snapshot(hypothesis).refute_evidence_ids:
            return True
    return False


@observed_relation_phase("core")
def consume_verified_composition(core, authorized, pending, state, generator, budget):
    """Recheck cold authority/current ledger before any positive query projection."""
    if not pending_composition_query_evidence((pending,), state):
        return state, None
    if generator is None or not callable(getattr(generator, "generate_verified_composition", None)):
        return state, None
    conclusions = _conclusion_candidates(core, pending, budget)
    expected_reads = len(conclusions) + len(pending.candidate.premises)
    if (state.read_count + expected_reads > state.budget.max_reads
            or state.node_count + 1 > state.budget.max_nodes
            or state.edge_count + 4 > state.budget.max_edges):
        return state, None
    sources = tuple(core.generation_input(item.proposition) for item in pending.candidate.premises)
    facts = tuple(ActiveSupportedRelationFact(source.proposition.definition, source.snapshot,
        core.owner.learning.engine.definition(source.hypothesis)) for source in sources)
    registry = core.context.scoped_identity_store.registry
    keys = tuple(tuple(sorted(entry.stable_key() for entry in current_query_evidence(
        core_query_evidence(premise.proposition, core.evidence_history(premise.proposition), registry))))
        for premise in pending.candidate.premises)
    verified = admit_verified_composition(pending, state.query_key, facts, keys, authorized,
        conclusion_refuted=_conclusion_refuted(core, pending, budget, candidates=conclusions))
    if verified is None:
        return state, None
    if len(verified.stable_key()) > budget.max_verified_proof_values:
        raise ValueError("composition verified proof budget exceeded")
    try:
        fact, generated = generator.generate_verified_composition(verified, core)
    except (ValueError, RuntimeError):
        return state, None
    claim, proof = verified.claim.stable_key(), verified.stable_key()
    entries = core_query_evidence(verified.claim, verified.history, registry)
    dialogue = EvidenceEntry(pending.witness.source.stable_key(), claim, space=3, polarity=1,
        evidence_key=(92185, *claim), scope_key=state.query_key,
        payload_key=proof + (len(generated.trace), *generated.trace))
    bindings = tuple(BindingEntry(item.role.stable_key(), item.filler.stable_key(),
        space=1, scope_key=state.query_key) for item in pending.candidate.bindings)
    marker = BindingEntry(COMPOSITION_VERIFICATION_ROLE, proof, space=1, scope_key=state.query_key)
    evidence = {item.stable_key(): item for item in (*state.evidence, *entries, dialogue)}
    binding_set = {item.stable_key(): item for item in (*state.bindings, *bindings, marker)}
    state = state.with_(evidence=tuple(evidence[key] for key in sorted(evidence)),
        bindings=tuple(binding_set[key] for key in sorted(binding_set)),
        read_count=state.read_count + expected_reads,
        node_count=state.node_count + 1, edge_count=state.edge_count + 4,
        best_candidate_key=claim, second_candidate_key=(),
        best_score=max(1, state.best_score), second_score=0)
    return state, CompositionQueryOutput(verified, fact, generated, (*entries, dialogue))


def composition_output_is_current(output, state):
    """Require the exact final proof, full premise ledger and derived support."""
    if output is None or not pending_composition_query_evidence((output.verified.pending,), state):
        return False
    proof, claim = output.verified.stable_key(), output.verified.claim.stable_key()
    current = current_query_evidence(state.evidence)
    expected = tuple(sorted(item.stable_key() for item in output.conclusion_evidence))
    actual = tuple(sorted(item.stable_key() for item in current
                         if item.space in {1, 3} and item.hypothesis_key == claim))
    return (BindingEntry(COMPOSITION_VERIFICATION_ROLE, proof, space=1,
                         scope_key=state.query_key) in state.bindings
            and actual == expected)
