"""Cold-audit actual output duties and their original removed-obligation calls."""
from __future__ import annotations

from pure_integer_ai.experiments.batch_trace_storage import load_case_query_trace

from pure_integer_ai.experiments.batch_case_storage import load_case_provenance

import json
from pathlib import Path

from pure_integer_ai.cognition.shared.integer_operation_execution import READY
from pure_integer_ai.cognition.shared.query_state import TERMINATION_ANSWER_CLOSED
from pure_integer_ai.cognition.understanding.query_input_structure import TrainedInputStructureProjector
from pure_integer_ai.cognition.understanding.query_structure_adapter import RelationSurfaceStructureAdapter
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.independent_relation_fact_source import read_independent_relation_fact_source
from pure_integer_ai.experiments.native_operation_call_audit import _hashes, verify_native_operation_generation
from pure_integer_ai.experiments.natural_relation_preflight import file_sha256
from pure_integer_ai.experiments.operation_composition_course import compile_train_operation_composition
from pure_integer_ai.experiments.operation_composition_materialization import read_native_operation_composition_graph
from pure_integer_ai.experiments.operation_output_course import compile_train_operation_output, read_operation_output_graph
from pure_integer_ai.experiments.operation_program_course import compile_train_operation_program
from pure_integer_ai.experiments.operation_output_query import OPERATION_OUTPUT_ROLE
from pure_integer_ai.experiments.operation_program_query import OPERATION_EXECUTION_ROLE, read_operation_core_context
from pure_integer_ai.experiments.operation_request_call_audit import (
    audit_operation_request_roots, restore_saved_operation_request_projection,
)
from pure_integer_ai.experiments.operation_request_course import compile_train_operation_request, read_operation_request_graph
from pure_integer_ai.experiments.operation_semantic_trace_audit import (
    FAIL_CLOSED, read_consumed_operation_execution, read_consumed_operation_selection,
    read_consumed_operation_output, read_consumed_operation_output_proof,
    read_saved_operation_output_proofs,
)
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack
from pure_integer_ai.experiments.relation_composition_course import compile_train_relation_composition
from pure_integer_ai.experiments.saved_relation_memory_audit import audit_saved_relation_memory
from pure_integer_ai.experiments.trained_generation_connector_runtime import TrainedGenerationConnectorRuntime
from pure_integer_ai.experiments.trained_input_concept_links import restore_input_concept_links
from pure_integer_ai.experiments.trained_open_role_frames import restore_open_role_frames
from pure_integer_ai.experiments.trained_operation_program_graph import read_operation_program_graph
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime


def _json(path):
    return json.loads(path.read_bytes())


def cold_development_output_sources(root):
    """No runtime, writes, training or generation; verify all physical licenses."""
    root = Path(root).resolve(strict=True)
    matrix = _json(root / "acceptance_matrix.json")
    manifests = tuple((root / "teachers").rglob("manifest.json"))
    if len(manifests) != 1 or len(matrix["cases"]) != 8:
        raise ValueError("output development bundle is incomplete")
    teachers = read_independent_relation_fact_source(root / "data/ph2/facts.jsonl", manifests[0].parent)
    programs = read_operation_program_graph(root / "programs.sqlite3")
    outputs = read_operation_output_graph(root / "outputs.sqlite3", programs)
    frames = read_operation_request_graph(root / "requests.sqlite3")
    if read_operation_output_graph(root / "outputs-removed.sqlite3", programs):
        raise ValueError("removed-obligation graph retains output permission")
    physical_programs, physical_outputs, physical_frames = [], [], []
    for number, case in enumerate(matrix["cases"], 1):
        stem = f"instructions/{number:02d}"
        program = compile_train_operation_program(root / (stem + ".program.train.json"), teachers)
        rules = (() if number != 2 else (compile_train_relation_composition(
            root / (stem + ".rule.train.json"), teachers),))
        output = compile_train_operation_output(root / (stem + ".output.train.json"), programs, rules=rules)
        frame = compile_train_operation_request(root / (stem + ".request.train.json"), teachers, program)
        if (case["program_witness_key"] != list(program.stable_key())
                or case["output_witness_key"] != list(output.stable_key())
                or case["input_values"] != list(map(ord, _json(root / (stem + ".request.train.json"))["question"]))):
            raise ValueError("output acceptance matrix differs from actual train instructions")
        physical_programs.append(program)
        physical_outputs.append(output)
        physical_frames.append(frame)
    if set(programs) != set(physical_programs) or set(outputs) != set(physical_outputs) or set(frames) != set(physical_frames):
        raise ValueError("physical train instructions differ from recovered integer graphs")
    return teachers, tuple(physical_programs), tuple(physical_outputs), frames


def _refused_execution(value, report, facts, program, prepared, core):
    if (value["termination"] not in FAIL_CLOSED or value.get("response_plan") or value.get("generation")
            or value.get("response_surface") or value["state_key"]["scalars"][7]
            or value["state_key"]["scalars"][15] or report["delivery_claim_count"]
            or report["capture_present"] or report["positive_chain_phase_coverage"]):
        raise ValueError("failed output duty retained factual Plan/generation/delivery")
    if not _binding_role_count(value, GraphQueryTraceEvidence(value), OPERATION_EXECUTION_ROLE):
        return None
    return read_consumed_operation_selection(value, facts, program, prepared, core=core)


def _binding_role_count(value, trace, role):
    """输出权限恢复完整类型；其它角色只用于既有执行存在性计数。"""
    if role == OPERATION_OUTPUT_ROLE:
        return len(read_saved_operation_output_proofs(value, trace))
    return sum(_unpack(trace.resolve_ref(ref), 1, 3, tail=1)[0] == role
               for ref in value["state_key"]["bindings"])


def audit_development_output_calls(root):
    """Read original calls and delivery O/H/E/Episode/Use; never invoke query/G."""
    root = Path(root).resolve(strict=True)
    before = _hashes(root)
    teachers, programs, outputs, frames = cold_development_output_sources(root)
    matrix, checkpoint = _json(root / "acceptance_matrix.json"), _json(root / "checkpoint.json")
    if checkpoint["stage"] != "ORIGINAL_EIGHT_OUTPUT_CALLS_ENDED" or checkpoint["completed_calls"] != 16:
        raise ValueError("eight output duties have incomplete original calls; admission UNKNOWN")
    frozen = _json(root / "readonly_production_sources.json")
    if frozen != {name: file_sha256(root / name) for name in frozen}:
        raise ValueError("development output production source identity drifted")
    variants, native, _metadata = read_native_operation_composition_graph(root / "composition.sqlite3")
    cases = []
    with TrainedRelationGraphRuntime(root / "core.sqlite3") as core, TrainedGenerationConnectorRuntime(root / "core.sqlite3") as generator:
        courses = tuple(compile_train_operation_composition(root / f"instructions/{number:02d}.composition.train.json",
            teachers, core, generator) for number in range(2, 9))
        if set(variants) != {course.variant for course in courses} or set(native) != {course.witness for course in courses}:
            raise ValueError("native output organization differs from physical train permission")
        facts, surfaces = read_operation_core_context(core), core.active_surface_facts()
        projector = TrainedInputStructureProjector(RelationSurfaceStructureAdapter.from_facts(surfaces),
            concept_links=restore_input_concept_links(core),
            open_role_frames=restore_open_role_frames(core.active_surface_frames(), surfaces))
        for number, (case, program, obligation) in enumerate(zip(matrix["cases"], programs, outputs, strict=True), 1):
            pair = []
            for arm in ("baseline", "removed"):
                path = root / f"calls/{arm}/{number:02d}"
                value, report = load_case_query_trace(path / "trace.int.json"), _json(path / "call_receipt.json")
                trace = GraphQueryTraceEvidence(value)
                if (report["input_values"] != case["input_values"]
                        or report.get("trace_descriptor_sha256", report["trace_sha256"]) != file_sha256(path / "trace.int.json")
                        or report["trace_counts"] != trace.counts() or report["phase_boundary_status"] != "PASS"
                        or report["termination"] != value["termination"]
                        or report["response_surface"] != value["response_surface"]
                        or report["phase_receipt_sha256"] != file_sha256(path / "phase_attempt/receipt.json")):
                    raise ValueError("output original call/phase receipt drifted")
                prepared = restore_saved_operation_request_projection(value, report, projector, frames)
                request_roots = audit_operation_request_roots(value, prepared)
                generation, proof, captured = {}, None, None
                delivered = arm == "baseline" and value["termination"] == TERMINATION_ANSWER_CLOSED
                output_count = _binding_role_count(value, trace, OPERATION_OUTPUT_ROLE)
                if arm == "baseline":
                    execution = (read_consumed_operation_execution(value, facts, program, prepared, core=core)
                                 if delivered else _refused_execution(value, report, facts, program, prepared, core))
                    # The current reference requirement is joined below to the
                    # independent durable declaration reader before admission.
                    if output_count and execution is None:
                        raise ValueError("consumed output proof lacks its current execution")
                    if not obligation.contract.current_request_required and (delivered or output_count):
                        reader = read_consumed_operation_output if delivered else read_consumed_operation_output_proof
                        proof = reader(value, facts, programs, (obligation,), execution)
                    if delivered:
                        captured = load_case_provenance(path / "case_provenance.json")
                        if report["positive_chain_phase_coverage"] != 1 or report["delivery_claim_count"] != len(execution.outputs):
                            raise ValueError("output duty lacks complete production stages/delivery")
                    if delivered and case["variant_id"]:
                        variant = next(row for row in variants if row.variant_id == case["variant_id"])
                        witness = next(row for row in native if row.variant_id == case["variant_id"])
                        generation = verify_native_operation_generation(core, generator, execution, value, trace, variant, witness)
                else:
                    execution = _refused_execution(value, report, facts, program, prepared, core)
                    if output_count:
                        raise ValueError("removed output gate retained a consumed output proof")
                memory = audit_saved_relation_memory(root / f"memory-{arm}.sqlite3", value, report, prepared,
                    captured=captured, require_operation_request_memory=True, report_incomplete_ohe=True)
                if arm == "baseline" and obligation.contract.current_request_required and (delivered or output_count):
                    proof = _read_current_reference_output(root / f"memory-{arm}.sqlite3", value, report,
                        prepared, facts, programs, obligation, execution, delivered=delivered)
                pair.append({"arm": arm, "operation_status": execution.status if execution is not None else None,
                    "execution_present": int(execution is not None), "termination": report["termination"],
                    "response_surface": report["response_surface"], "output_proof_present": int(proof is not None),
                    "selected_claim_count": len(execution.outputs) if execution is not None else None,
                    "delivered_claim_count": report["delivery_claim_count"],
                    "delivery_status": "PASS" if delivered else "FAIL" if arm == "baseline" else "PASS_REMOVAL_REFUSAL",
                    "removal_isolation_status": ("PASS" if execution is not None and execution.status == READY
                        and request_roots["request_core_root_status"] == "PASS" else "UNKNOWN_CURRENT_SELECTION_NOT_PROVEN")
                        if arm == "removed" else None,
                    "failure_boundary": (None if delivered else "OUTPUT_PROOF_CONSUMED_WITHOUT_PLAN_OR_GENERATION"
                        if proof is not None else "CURRENT_SELECTION_WITHOUT_OUTPUT_PROOF_OR_DELIVERY"
                        if execution is not None else "NO_CONSUMED_CURRENT_SELECTION_OR_OUTPUT_PROOF"),
                    "wall_ns": report["wall_ns"], "cpu_ns": report["cpu_ns"],
                    "trace_counts": trace.counts(), "resource_started": report["resource_started"],
                    "resource_ended": report["resource_ended"], **request_roots, **generation, **memory})
            cases.append({"audit_family": case["audit_family"], "original_calls": pair,
                "ordinal": number,
                "development_output_gate": ("PASS_CURRENT_OBLIGATION_AND_REMOVAL_ONLY"
                    if pair[0]["delivery_status"] == "PASS" and pair[0]["request_core_root_status"] == "PASS"
                    and pair[1]["removal_isolation_status"] == "PASS"
                    and not any(row["memory_ohe"]["status"].startswith("UNKNOWN") for row in pair)
                    else "FAIL_CURRENT_OUTPUT_DUTY_NOT_DELIVERED" if pair[0]["delivery_status"] == "FAIL"
                    else "UNKNOWN_REQUEST_ROOT_MEMORY_OHE_OR_REMOVAL_ISOLATION_GAP"),
                "semantic_graph_edit_counterfactual": "UNKNOWN_NOT_ESTABLISHED_BY_WHOLE_OBLIGATION_REMOVAL",
                "held_out_novelty": "UNKNOWN_TRAIN_INPUT_ONLY"})
    if before != _hashes(root):
        raise ValueError("cold output audit changed original evidence")
    passed = sum(case["development_output_gate"] == "PASS_CURRENT_OBLIGATION_AND_REMOVAL_ONLY" for case in cases)
    return {"status": ("PASS_DEVELOPMENT_EIGHT_OUTPUT_GATES_ONLY" if passed == 8
        else "FAIL_DEVELOPMENT_OUTPUT_GATES_INCOMPLETE"), "cases": cases,
        "verified_original_call_count": 16, "verified_current_output_duty_count": passed,
        "actual_positive_delivery_count": sum(case["original_calls"][0]["delivery_status"] == "PASS" for case in cases),
        "failed_positive_ordinals": [case["ordinal"] for case in cases if case["original_calls"][0]["delivery_status"] == "FAIL"],
        "verified_removed_refusal_count": 8, "qualified_semantic_operation_families": 0,
        "verified_removal_isolation_count": sum(case["original_calls"][1]["removal_isolation_status"] == "PASS" for case in cases),
        "unknown_positive_ordinals": [case["ordinal"] for case in cases if case["development_output_gate"].startswith("UNKNOWN")],
        "source_write_count": 0, "query_reexecution_count": 0, "generation_reexecution_count": 0,
        "held_out_novelty": "UNKNOWN_NO_HELD_OUT", "wave1_production_authorized": 0,
        "session_resource_tail": checkpoint["session_resource_tail"], "free_dialogue_complete": 0}


def _read_current_reference_output(database, value, report, prepared, facts, programs, obligation, execution, *, delivered=True):
    from pure_integer_ai.cognition.shared.identity import SourceRef
    from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
    from pure_integer_ai.experiments.trained_dialogue_memory_graph import _build_interaction_memory_intake
    from pure_integer_ai.storage.backend import SQLiteBackend
    backend = SQLiteBackend(str(database), read_only=True)
    try:
        intake = _build_interaction_memory_intake(backend)
        owner = SourceRef.from_stable_key(tuple(report["input_source"])).owner
        access = MemoryAccessContext(owner.tenant_id, owner.user_id, owner.session_id)
        declarations = lambda ref: tuple(row for row in intake.event_log.query(access=access, object_ref=ref)
                                         if row.event.is_declaration)
        reader = read_consumed_operation_output if delivered else read_consumed_operation_output_proof
        return reader(value, facts, programs, (obligation,), execution,
            prepared=prepared, report=report, declaration_reader=declarations)
    finally:
        backend.close()
