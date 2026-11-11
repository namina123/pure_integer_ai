"""Generate only the current claims and role order selected by the integer program."""
from __future__ import annotations

from dataclasses import replace

from pure_integer_ai.cognition.shared.integer_operation_execution import READY
from pure_integer_ai.cognition.shared.query_state import ROOT_PENDING, TERMINATION_ANSWER_CLOSED
from pure_integer_ai.cognition.shared.query_driver import unvisited_frontier
from pure_integer_ai.experiments.generation_organization import (
    ClosedGenerationEvidence, bind_query_evidence, build_multi_response_plan, plan_from_active_fact,
)
from pure_integer_ai.experiments.operation_program_query import operation_plan_matches
from pure_integer_ai.experiments.relation_runtime_phase import observed_relation_phase
from pure_integer_ai.experiments.trained_multi_claim_composer import GRAPH_KIND, PART_CLAIM_PREDICATE
from pure_integer_ai.experiments.operation_sequence_graph import SEQUENCE_KIND


def _select_operation_realization_variant(generator, execution, predicates):
    """按实际 claim 谓词序列从物理许可集合中选择唯一 variant。"""
    program, witness = execution.witness.program, execution.witness
    key = program.realization_key
    if (len(key) < 2 or key[0] != GRAPH_KIND
            or (program.path_query is None and len(key) != 2)):
        raise ValueError("operation has no complete graph connector permission")
    permitted_ids = frozenset(key[1:])
    matches = []
    for variant in getattr(generator, "variants", ()):
        guarded = tuple(part[2] for part in sorted((part for part in variant.parts
            if part[0] == PART_CLAIM_PREDICATE), key=lambda part: part[1]))
        if (variant.variant_id in permitted_ids and variant.claim_count == len(predicates)
                and variant.source_identity == witness.source_identity
                and variant.license_identity == witness.license_identity
                and guarded == predicates):
            matches.append(variant)
    if len(matches) != 1:
        raise ValueError("operation graph connector variant is missing or ambiguous")
    return matches[0]


@observed_relation_phase("response_plan")
def generate_operation_response(core, generator, execution, state, *, output_witnesses=None,
                                output_proof=None, output_programs=()):
    """The program supplies selection; trained connectors supply every output token."""
    if (execution is None or execution.status != READY or generator is None
            or state.conflict_open or state.required_slots_open
            or unvisited_frontier(state.frontier, state.visited)
            or any(root.status == ROOT_PENDING for root in state.roots)):
        return state, None, None
    from pure_integer_ai.experiments.operation_companion_query import operation_companion_consumed
    if not operation_companion_consumed(execution, state):
        return state, None, None
    if output_witnesses is not None:
        from pure_integer_ai.cognition.shared.integer_operation_output_verifier import verify_operation_output
        from pure_integer_ai.experiments.operation_output_query import output_proof_consumed
        from pure_integer_ai.experiments.operation_program_query import read_operation_core_context
        if not output_proof_consumed(output_proof, execution, state):
            return state, None, None
        try:
            verify_operation_output(output_proof, read_operation_core_context(core,
                                    witness=execution.witness, variables=execution.variables),
                                    output_programs, output_witnesses)
        except ValueError:
            return state, None, None
    facts = {fact.proposition.stable_key(): fact for fact in core.active_surface_facts()}
    components, evidences = [], []
    for key, roles in execution.outputs:
        fact = facts.get(key)
        if fact is None:
            return state, None, None
        by_role = {binding.role.stable_key(): binding for binding in fact.bindings}
        if set(roles) != set(by_role):
            return state, None, None
        fact = replace(fact, bindings=tuple(by_role[role] for role in roles))
        source = core.generation_input(fact.proposition)
        generated = generator.generate_relation(source, fact)
        if generated.connector is None or not generated.representations:
            return state, None, None
        components.append((source, fact, generated))
        evidences.append(ClosedGenerationEvidence(fact.bindings, generated, key,
            fact.predicate.stable_key(), fact.source_hash, connector=generated.connector))
    if not components:
        return state, None, None
    ready = state.with_(termination=TERMINATION_ANSWER_CLOSED, evidence_closed=1, generation_ready=1,
        best_candidate_key=execution.outputs[0][0], second_candidate_key=(), second_score=0,
        best_score=max(state.best_score, 1))
    program = execution.witness.program
    composition_witness = None
    sequence_witness = None
    try:
        if not program.realization_key:
            if len(components) != 1:
                raise ValueError("multi-claim operation requires an explicit graph connector variant")
            composed = components[0][2]
            plan = plan_from_active_fact(components[0][1], composed)
        else:
            predicates = tuple(item[1].predicate.stable_key() for item in components)
            if program.realization_key[0] == SEQUENCE_KIND:
                select_sequence = getattr(generator, "select_operation_sequence", None)
                compose_sequence = getattr(generator, "generate_operation_sequence", None)
                if not callable(select_sequence) or not callable(compose_sequence):
                    raise ValueError("operation realization needs the learned native sequence runtime")
                sequence_witness = select_sequence(execution, predicates)
                composed = compose_sequence(tuple(components), ready,
                    execution=execution, rule_id=sequence_witness.rule_id)
            else:
                variant = _select_operation_realization_variant(generator, execution, predicates)
                compose = getattr(generator, "generate_operation_composition", None)
                if compose is None:
                    raise ValueError("operation realization needs native graph generation with a complete role contract")
                composed = compose(tuple(components), ready, execution=execution,
                                   variant_id=variant.variant_id)
                composition_witness = generator.operation_composition_witness_for(variant.variant_id)
            plan = build_multi_response_plan(tuple(evidences), composed,
                                             claim_slot_contract=execution.outputs)
        if not operation_plan_matches(execution, plan):
            return state, None, None
        plan = bind_query_evidence(plan, ready)
    except (RuntimeError, TypeError, ValueError):
        return state, None, None
    value = {
        "kind": 6, "operation_execution": list(execution.stable_key()),
        "surface_values": list(map(ord, plan.surface())),
        "connector": list(composed.connector.stable_key()),
        "representations": [list(item.stable_key()) for item in composed.representations],
        "trace": list(composed.trace), "operation_scope": 1, "operation_truth_added": 0,
        "claim_generations": [{"claim": list(component[1].proposition.stable_key()),
            "connector": list(component[2].connector.stable_key()),
            "representations": [list(item.stable_key()) for item in component[2].representations],
            "trace": list(component[2].trace)} for component in components],
    }
    if composition_witness is not None:
        value["native_composition_witness"] = list(composition_witness.stable_key())
    if sequence_witness is not None:
        value["native_sequence_witness"] = list(sequence_witness.stable_key())
    if output_witnesses is not None:
        value["operation_output_proof"] = list(output_proof.stable_key())
    return ready, plan, value
