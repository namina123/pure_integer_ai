"""物理程序编辑到原自然调用的独立冷审计；复用图生成和 O/H/E 审计内核。"""
from __future__ import annotations

from pure_integer_ai.experiments.batch_trace_storage import load_case_query_trace

from pure_integer_ai.experiments.batch_case_storage import load_case_provenance

from pathlib import Path

from pure_integer_ai.cognition.shared.integer_operation_behavior import OperationSemanticEditReceipt
from pure_integer_ai.cognition.understanding.query_input_structure import TrainedInputStructureProjector
from pure_integer_ai.cognition.understanding.query_structure_adapter import RelationSurfaceStructureAdapter
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.independent_relation_fact_source import read_independent_relation_fact_source
from pure_integer_ai.experiments.native_operation_call_audit import _hashes, _json, verify_native_operation_generation
from pure_integer_ai.experiments.natural_relation_preflight import file_sha256
from pure_integer_ai.experiments.operation_composition_course import compile_train_operation_composition
from pure_integer_ai.experiments.operation_composition_materialization import read_native_operation_composition_graph
from pure_integer_ai.experiments.operation_program_course import compile_train_operation_program
from pure_integer_ai.experiments.operation_program_query import read_operation_core_context
from pure_integer_ai.experiments.operation_request_call_audit import restore_saved_operation_request
from pure_integer_ai.experiments.operation_request_consumer import validate_operation_request_frames
from pure_integer_ai.experiments.operation_request_course import compile_train_operation_request, read_operation_request_graph
from pure_integer_ai.experiments.operation_semantic_trace_audit import (
    audit_consumed_semantic_pair, read_consumed_operation_execution,
)
from pure_integer_ai.experiments.saved_relation_memory_audit import audit_saved_relation_memory
from pure_integer_ai.experiments.trained_generation_connector_runtime import TrainedGenerationConnectorRuntime
from pure_integer_ai.experiments.trained_input_concept_links import restore_input_concept_links
from pure_integer_ai.experiments.trained_open_role_frames import restore_open_role_frames
from pure_integer_ai.experiments.trained_operation_program_graph import read_operation_program_graph
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime


def audit_saved_semantic_operation_pair(source, core_source):
    """核物理许可来源、确切图 delta、两份原始消费 proof 和真实交付/拒答。"""
    source, core_source = Path(source).resolve(strict=True), Path(core_source).resolve(strict=True)
    before, original = _hashes(source), _hashes(core_source)
    identity, checkpoint = _json(source / "identity.json"), _json(source / "checkpoint.json")
    if (identity["source_files"] != original or identity["readonly_core_source"] != str(core_source)
            or identity["query_count_planned"] != 2
            or checkpoint["stage"] != "ORIGINAL_SEMANTIC_COUNTERFACTUAL_CALLS_ENDED"
            or checkpoint["completed_calls"] != 2):
        raise ValueError("semantic original source or checkpoint drifted")
    manifests = tuple((core_source / "teachers").rglob("manifest.json"))
    if len(manifests) != 1:
        raise ValueError("semantic Core lacks one physical licensed fact pack")
    teachers = read_independent_relation_fact_source(core_source / "data/ph2/facts.jsonl", manifests[0].parent)
    witnesses = []
    for name in ("baseline", "counterfactual"):
        witness = compile_train_operation_program(source / f"{name}.train.json", teachers)
        if read_operation_program_graph(source / f"{name}.sqlite3") != (witness,):
            raise ValueError("semantic physical graph differs from licensed source")
        witnesses.append(witness)
    edit = OperationSemanticEditReceipt.from_stable_key(tuple(_json(source / "semantic_edit.int.json")["receipt_key"]))
    if (edit.before != witnesses[0] or edit.after != witnesses[1]
            or edit.before_graph_sha256 != tuple(bytes.fromhex(file_sha256(source / "baseline.sqlite3")))
            or edit.after_graph_sha256 != tuple(bytes.fromhex(file_sha256(source / "counterfactual.sqlite3")))):
        raise ValueError("semantic physical edit provenance differs from frozen receipt")
    if (identity["semantic_edit_sha256"] != file_sha256(source / "semantic_edit.int.json")
            or identity["request_source_sha256"] != file_sha256(source / "request.train.json")
            or identity["request_graph_sha256"] != file_sha256(source / "requests.sqlite3")):
        raise ValueError("semantic frozen edit or request physical identity drifted")
    frames = read_operation_request_graph(source / "requests.sqlite3")
    if frames != (compile_train_operation_request(source / "request.train.json", teachers, edit.before),):
        raise ValueError("semantic request graph differs from physical train permission")
    for witness in witnesses:
        validate_operation_request_frames(frames, (witness,))
    cases, values, prepared_inputs = [], [], []
    with TrainedRelationGraphRuntime(core_source / "core.sqlite3") as core, TrainedGenerationConnectorRuntime(
            core_source / "core.sqlite3") as generator:
        course = compile_train_operation_composition(core_source / "composition.train.json", teachers, core, generator)
        variants, native_witnesses, _metadata = read_native_operation_composition_graph(core_source / "composition.sqlite3")
        if variants != (course.variant,) or native_witnesses != (course.witness,):
            raise ValueError("semantic native grammar differs from physical train course")
        facts, surfaces = read_operation_core_context(core), core.active_surface_facts()
        projector = TrainedInputStructureProjector(RelationSurfaceStructureAdapter.from_facts(surfaces),
            concept_links=restore_input_concept_links(core),
            open_role_frames=restore_open_role_frames(core.active_surface_frames(), surfaces))
        for number, witness in enumerate(witnesses, 1):
            root = source / f"cases/{number:03d}"
            value, report = load_case_query_trace(root / "trace.int.json"), _json(root / "call_receipt.json")
            trace = GraphQueryTraceEvidence(value)
            if (report["input_values"] != identity["input_values"]
                    or report.get("trace_descriptor_sha256", report["trace_sha256"]) != file_sha256(root / "trace.int.json")
                    or report["trace_counts"] != trace.counts()
                    or report["termination"] != value["termination"]
                    or report["response_surface"] != value["response_surface"]
                    or report["phase_boundary_status"] != "PASS"
                    or report["phase_receipt_sha256"] != file_sha256(root / "phase_attempt/receipt.json")):
                raise ValueError("semantic original call/phase receipt differs from trace")
            prepared = restore_saved_operation_request(value, report, projector, frames)
            execution = read_consumed_operation_execution(value, facts, witness, prepared, core=core)
            captured, generation = None, {}
            if execution.status == 1:
                if report["delivery_claim_count"] != len(execution.outputs) or report["positive_chain_phase_coverage"] != 1:
                    raise ValueError("semantic positive original call lacks complete delivery stages")
                generation = verify_native_operation_generation(core, generator, execution, value, trace,
                                                                variants[0], native_witnesses[0])
                captured = load_case_provenance(root / "case_provenance.json")
            elif report["delivery_claim_count"] or report["capture_present"]:
                raise ValueError("semantic failed obligation obtained factual delivery")
            memory = audit_saved_relation_memory(source / f"memory-{number:03d}.sqlite3", value, report, prepared,
                captured=captured, require_operation_request_memory=True)
            values.append(value)
            prepared_inputs.append(prepared)
            cases.append({"name": "baseline" if number == 1 else "counterfactual",
                "operation_status": execution.status, "termination": report["termination"],
                "response_surface": report["response_surface"], "trace_counts": trace.counts(),
                "wall_ns": report["wall_ns"], "cpu_ns": report["cpu_ns"],
                "resource_started": report["resource_started"], "resource_ended": report["resource_ended"],
                "peak_scope": report["peak_scope"], "trace_sha256": report["trace_sha256"],
                "phase_receipt_sha256": report["phase_receipt_sha256"], **generation, **memory})
        if prepared_inputs[0] != prepared_inputs[1]:
            raise ValueError("semantic paired current input structures differ")
        result = audit_consumed_semantic_pair(edit, *values, facts, prepared_inputs[0], core=core)
    if before != _hashes(source) or original != _hashes(core_source):
        raise ValueError("semantic cold audit changed immutable original artifacts")
    return {**result, "cases": cases, "source_sha256": before, "core_source_sha256": original,
        "physical_graph_edit_qualification": "MATCHED_COMPLETE_CURRENT_TRAIN_PROGRAM_RELATION_DELTA",
        "source_write_count": 0, "query_reexecution_count": 0, "generation_reexecution_count": 0,
        "session_tail_budget": "UNKNOWN_TWO_INDEPENDENT_EMPTY_MEMORY_CALLS"}
