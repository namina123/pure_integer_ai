"""Independent cold audit of original natural graph-program counterfactual calls."""
from __future__ import annotations

from pure_integer_ai.experiments.batch_trace_storage import load_case_query_trace

from pure_integer_ai.experiments.batch_case_storage import load_case_provenance

from dataclasses import replace
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.integer_operation_execution import OperationExecution
from pure_integer_ai.cognition.shared.integer_operation_verifier import verify_integer_operation
from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.cognition.shared.representation_rendering import representation_parts
from pure_integer_ai.cognition.understanding.query_input_structure import STRUCTURE_CLOSED, TrainedInputStructureProjector
from pure_integer_ai.cognition.understanding.query_relation_questions import match_relation_question_frames
from pure_integer_ai.cognition.understanding.query_structure_adapter import RelationSurfaceStructureAdapter
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.connected_relation_question_course import compile_train_relation_question
from pure_integer_ai.experiments.independent_relation_fact_source import read_independent_relation_fact_source
from pure_integer_ai.experiments.integer_relation_connected_course import _FACT_FIELDS, _read_lines, _seed
from pure_integer_ai.experiments.natural_relation_preflight import file_sha256
from pure_integer_ai.experiments.operation_program_course import compile_train_operation_program
from pure_integer_ai.experiments.operation_program_query import OPERATION_EXECUTION_ROLE, _request_variables, read_operation_core_context
from pure_integer_ai.experiments.pending_composition_call_audit import _decode_input
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack
from pure_integer_ai.experiments.saved_relation_memory_audit import audit_saved_relation_memory
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime, _bound_relation_proposition, _generation_protocols, _slot_ordinal,
)
from pure_integer_ai.experiments.trained_input_concept_links import restore_input_concept_links
from pure_integer_ai.experiments.trained_open_role_frames import restore_open_role_frames
from pure_integer_ai.experiments.trained_operation_program_graph import read_operation_program_graph
from pure_integer_ai.experiments.trained_query_evidence import current_query_evidence
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.trained_relation_question_graph import MODE_RELATION_PAIR_QUERY, read_relation_question_graph


def _json(path):
    return json.loads(path.read_bytes())


def _hashes(root):
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in root.rglob("*") if path.is_file()}


def restore_saved_operation_question(value, report, projector, frames):
    """Cold-check full syntax and the selected scope, without invoking QueryBridge."""
    trace = GraphQueryTraceEvidence(value)
    projected = projector.project(tuple(report["input_values"]))
    bindings = match_relation_question_frames(frames, projected)
    if len(bindings) != 1:
        raise ValueError("operation ordinary input has no unique learned full question binding")
    question = bindings[0]
    relations = tuple(item for item in projected.relation_candidates
                      if item.stable_key() == question.relation_key)
    if len(relations) != 1 or relations[0].state != STRUCTURE_CLOSED:
        raise ValueError("operation question did not select one complete cold relation")
    relation = relations[0]
    # Preserve every original token/span/link. Only graph candidates outside the
    # learned question's relation scope are excluded, as in the actual intake.
    prepared = replace(projected, operation_keys=(question.operation_key,),
        relation_question_bindings=bindings,
        relation_candidates=relations, open_relations=(),
        carrier_candidates=tuple(item for item in projected.carrier_candidates
                                 if item.structure_key == relation.structure_key),
        semantic_candidates=tuple(item for item in projected.semantic_candidates
            if item.proposition_key == relation.proposition_key and item.span_ref in relation.span_refs))
    if (_decode_input(value["input_structure"], trace) != prepared.integer_trace()
            or trace.resolve_ref(value["query_ref"]) != prepared.source_ref
            or trace.operation_contracts() != ((question.operation_key, MODE_RELATION_PAIR_QUERY),)):
        raise ValueError("operation saved input differs from cold syntax/role graph")
    return prepared


def verify_removed_operation_trace(value, report):
    """An empty program graph must leave no proof, factual closure or delivery."""
    trace = GraphQueryTraceEvidence(value)
    state = value["state_key"]
    if (value["termination"] != report["termination"] or report["termination"] == 1
            or state["scalars"][7] != 0 or state["scalars"][15] != 0
            or value.get("response_plan") or value.get("generation") or value.get("response_surface")
            or report["response_plan_present"] or report["generation_present"]
            or report["delivery_claim_count"] or report["response_surface"]):
        raise ValueError("removed operation acquired factual closure, generation or delivery")
    if any(_unpack(trace.resolve_ref(ref), 1, 3, tail=1)[0] == OPERATION_EXECUTION_ROLE
           for ref in state["bindings"]):
        raise ValueError("removed operation retained a program execution proof")
    if any(_unpack(trace.resolve_ref(ref), 3, 6)[2][:1] == (92230,)
           for ref in state["evidence"]):
        raise ValueError("removed operation retained program permission Evidence")


def _verify_role_generation(core, generator, execution, value, trace):
    """Read selected role/atom representations; do not run G planners or queries."""
    if len(execution.outputs) != 1:
        raise ValueError("this cold auditor qualifies single-claim projection only")
    key, _roles = execution.outputs[0]
    fact = next(fact for fact in core.active_surface_facts() if fact.proposition.stable_key() == key)
    template = generator.generation_template_for(fact)
    protocols = _generation_protocols(generator.context, template.language_branch)
    bound = _bound_relation_proposition(core.generation_input(fact.proposition), template, protocols.content.answer)
    recovered = generator.connector_for(bound, branch=template.language_branch, connector_identity=template.connector)
    if any(binding.source == recovered.connector.registry.value_protocol.proposition_source for binding in template.bindings):
        raise ValueError("operation generation contains a whole-proposition replay slot")
    slots = recovered.connector.registry.values(template, bound)
    selected = {}
    for slot in slots:
        route = recovered.surface_for(slot.filler)
        if route.result.selected is None or len(route.result.options) != 1:
            raise ValueError("operation generation graph has ambiguous role realization")
        selected[slot.slot] = route.result.selected.value
    representations = tuple(selected[slot.slot] for slot in sorted(template.slots, key=lambda slot: _slot_ordinal(slot.slot)))
    actual = tuple(trace.resolve_ref(row["integer_key_ref"]) for row in value["generation"]["representations"])
    if actual != tuple(item.stable_key() for item in representations):
        raise ValueError("operation original generation representations differ from cold graph atoms")
    units = tuple(unit for item in representations for unit in representation_parts(item)[1])
    if (units != tuple(map(ord, value["response_surface"]))
            or units != trace.resolve_ref(value["generation"]["surface_values"]["integer_key_ref"])):
        raise ValueError("operation original surface differs from role-level graph representation sequence")
    if trace.resolve_ref(value["generation"]["connector"]["integer_key_ref"]) != template.connector.stable_key():
        raise ValueError("operation original connector differs from the selected Core frame")


def audit_saved_operation_counterfactuals(source, core_source):
    """Each graph delta is physically cold-read; metadata does not authorize mutation."""
    source, core_source = Path(source).resolve(strict=True), Path(core_source).resolve(strict=True)
    before, original = _hashes(source), _hashes(core_source)
    identity = _json(source / "identity.json")
    if (identity["source_files"] != original or identity["readonly_core_source"] != str(core_source)
            or identity["query_count_planned"] != 3):
        raise ValueError("operation source identity or expected original calls drifted")
    checkpoint = _json(source / "checkpoint.json")
    if checkpoint["completed_calls"] != 3 or checkpoint["stage"] != "ORIGINAL_GRAPH_COUNTERFACTUAL_CALLS_ENDED":
        raise ValueError("operation original calls have no complete frozen checkpoint")
    variants = identity["program_variants"]
    if (tuple(row["name"] for row in variants) != ("baseline", "swapped", "removed")
            or any(row["program_sha256"] != file_sha256(source / (row["name"] + ".sqlite3")) for row in variants)
            or any(row["source_sha256"] != file_sha256(source / (row["name"] + ".train.json")) for row in variants[:2])):
        raise ValueError("operation physical counterfactual graph or instruction identity drifted")
    manifests = tuple((core_source / "teachers").rglob("manifest.json"))
    if len(manifests) != 1:
        raise ValueError("operation Core source has no unique physical licensed fact pack")
    manifest = manifests[0]
    teachers = read_independent_relation_fact_source(core_source / "data/ph2/facts.jsonl", manifest.parent)
    witnesses = []
    for name in ("baseline", "swapped"):
        compiled = compile_train_operation_program(source / (name + ".train.json"), teachers)
        if read_operation_program_graph(source / (name + ".sqlite3")) != (compiled,):
            raise ValueError("operation runtime graph differs from physical train instruction")
        witnesses.append(compiled)
    if read_operation_program_graph(source / "removed.sqlite3") != ():
        raise ValueError("removed operation counterfactual still has a program relation")
    baseline, swapped = (witness.program for witness in witnesses)
    if (replace(baseline, projections=swapped.projections) != swapped
            or baseline.projections == swapped.projections
            or witnesses[0].authorized_forming_sources != witnesses[1].authorized_forming_sources
            or witnesses[0].source_identity != witnesses[1].source_identity
            or witnesses[0].license_identity != witnesses[1].license_identity):
        raise ValueError("counterfactual changes more than the actual output projection relation")
    frames, metadata = read_relation_question_graph(core_source / "questions.sqlite3")
    if metadata["source_identity"] != witnesses[0].source_identity or metadata["license_identity"] != witnesses[0].license_identity:
        raise ValueError("operation syntax and physical program permission drifted")
    syntax = _json(core_source / "question.train.json")
    if (syntax["owner"], syntax["split"], syntax["license_id"]) != ("teacher", "train", "CC0-1.0"):
        raise ValueError("operation question source is not a licensed train teacher")
    seeds = tuple(_seed(row, number) for number, row in enumerate(
        _read_lines(core_source / "data/ph2/facts.jsonl", _FACT_FIELDS), 1) if row["seed_id"] == syntax["seed_id"])
    if len(seeds) != 1 or frames != (compile_train_relation_question(seeds[0], syntax["question"],
            source_case_key=tuple(syntax["source_case_key"]), operation_key=tuple(syntax["operation_key"])),):
        raise ValueError("operation learned question differs from the physical train instruction")
    executions, cases = [], []
    with TrainedRelationGraphRuntime(core_source / "core.sqlite3") as core, TrainedGenerationConnectorRuntime(core_source / "core.sqlite3") as generator:
        facts = read_operation_core_context(core)
        surfaces = core.active_surface_facts()
        projector = TrainedInputStructureProjector(RelationSurfaceStructureAdapter.from_facts(surfaces),
            concept_links=restore_input_concept_links(core), open_role_frames=restore_open_role_frames(core.active_surface_frames(), surfaces))
        for number, name in enumerate(("baseline", "swapped", "removed"), 1):
            root = source / f"cases/{number:03d}"
            value, report = load_case_query_trace(root / "trace.int.json"), _json(root / "call_receipt.json")
            trace = GraphQueryTraceEvidence(value)
            if (report["input_values"] != identity["input_values"] or report.get("trace_descriptor_sha256", report["trace_sha256"]) != file_sha256(root / "trace.int.json")
                    or report["trace_counts"] != trace.counts() or report["phase_boundary_status"] != "PASS"
                    or report["phase_receipt_sha256"] != file_sha256(root / "phase_attempt/receipt.json")):
                raise ValueError("operation original call/phase receipt does not match saved artifacts")
            prepared = restore_saved_operation_question(value, report, projector, frames)
            captured = None
            if number <= 2:
                execution = OperationExecution.from_stable_key(trace.resolve_ref(value["generation"]["operation_execution"]["integer_key_ref"]))
                verify_integer_operation(execution, facts, (witnesses[number - 1],))
                if (execution.variables != _request_variables(execution.witness, prepared, facts)
                        or execution.query_key != trace.resolve_ref(value["query_ref"])):
                    raise ValueError("operation execution did not consume the exact ordinary input roles")
                state = value["state_key"]
                binding = BindingEntry(OPERATION_EXECUTION_ROLE, execution.stable_key(), space=1, scope_key=execution.query_key)
                if binding.stable_key() not in tuple(trace.resolve_ref(ref) for ref in state["bindings"]):
                    raise ValueError("operation actual shared QueryState lacks the complete program proof")
                evidence = []
                for ref in state["evidence"]:
                    key = trace.resolve_ref(ref)
                    fields = _unpack(key, 3, 6)
                    evidence.append(EvidenceEntry(fields[0], fields[1], key[0], key[1], key[2], *fields[2:]))
                current = current_query_evidence(tuple(evidence))
                used = {key for match in execution.matches for key in match}
                for fact in facts:
                    if fact.proposition_key in used and fact.evidence_keys != tuple(sorted(entry.stable_key() for entry in current
                        if entry.space == 1 and entry.hypothesis_key == fact.proposition_key
                        and (entry.evidence_key[:1] == (1,) or entry.polarity != 3))):
                        raise ValueError("operation original shared QueryState lacks exact premise Evidence")
                plan = _decode_input(value["response_plan"], trace)
                if (value["termination"] != 1 or report["termination"] != 1
                        or plan["claim_refs"] != [list(key) for key, _ in execution.outputs]
                        or state["scalars"][6:9] != [0, 1, 0]
                        or trace.resolve_ref(state["scalars"][11]) != execution.outputs[0][0]):
                    raise ValueError("operation final closure or actual Plan selection drifted")
                for key, roles in execution.outputs:
                    fact = next(fact for fact in facts if fact.proposition_key == key)
                    fillers = {row.role.stable_key(): row.filler.stable_key() for row in fact.definition.bindings}
                    if tuple((tuple(slot["role"]), tuple(slot["filler"])) for slot in plan["slot_sequence"]) != tuple((role, fillers[role]) for role in roles):
                        raise ValueError("operation final Plan role/filler order drifted")
                _verify_role_generation(core, generator, execution, value, trace)
                captured = load_case_provenance(root / "case_provenance.json")
                executions.append(execution)
            else:
                verify_removed_operation_trace(value, report)
            memory = audit_saved_relation_memory(source / f"memory-{number:03d}.sqlite3", value, report, prepared, captured=captured)
            cases.append({"name": name, "termination": report["termination"], "response_surface": report["response_surface"],
                "trace_counts": trace.counts(), "wall_ns": report["wall_ns"], "cpu_ns": report["cpu_ns"],
                "resource_started": report["resource_started"], "resource_ended": report["resource_ended"],
                "peak_scope": report["peak_scope"], "trace_sha256": report["trace_sha256"],
                "phase_receipt_sha256": report["phase_receipt_sha256"], **memory})
    if (executions[0].context_identity != executions[1].context_identity or executions[0].variables != executions[1].variables
            or executions[0].matches != executions[1].matches or executions[0].outputs == executions[1].outputs
            or cases[0]["response_surface"] == cases[1]["response_surface"]):
        raise ValueError("physical operation projection edit does not explain a same-context output difference")
    if before != _hashes(source) or original != _hashes(core_source):
        raise ValueError("operation cold audit changed an immutable original source")
    return {"status": "PASS_DEVELOPMENT_PHYSICAL_PROGRAM_PROJECTION_CORE_OHE_PLAN_DELIVERY_COUNTERFACTUAL",
        "cases": cases, "source_sha256": before, "core_source_sha256": original,
        "source_write_count": 0, "query_reexecution_count": 0, "generation_reexecution_count": 0,
        "qualified_semantic_operation_families": 0, "operation_family_qualification": "UNKNOWN_NO_EIGHT_FAMILY_COURSE",
        "held_out_novelty": "UNKNOWN_TRAIN_QUESTION", "free_dialogue_complete": 0}
