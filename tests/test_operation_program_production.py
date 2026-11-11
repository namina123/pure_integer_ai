"""One original natural call per genuine graph program counterfactual."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.integer_operation_execution import OperationExecution
from pure_integer_ai.cognition.shared.integer_operation_program import (
    IntegerOperationProgram, OperationAtom, OperationProjection,
)
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.independent_relation_fact_source import read_independent_relation_fact_source
from pure_integer_ai.experiments.natural_relation_preflight import file_sha256, run_natural_relation_case, write_preflight_json
from pure_integer_ai.experiments.operation_program_course import compile_train_operation_program
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.trained_generation_connector_runtime import TrainedGenerationConnectorRuntime
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
from pure_integer_ai.experiments.trained_operation_program_graph import materialize_operation_program_graph
from pure_integer_ai.experiments.trained_relation_composition_graph import read_relation_composition_graph
from pure_integer_ai.experiments.trained_relation_question_graph import read_relation_question_graph


def test_real_natural_program_projection_swap_and_removed_instruction(tmp_path):
    saved = os.environ.get("PIA_SAVED_COMPOSITION_SOURCE")
    if saved is None:
        pytest.skip("requires the explicitly supplied immutable two-premise Core")
    source = Path(saved).resolve(strict=True)
    evidence_root = os.environ.get("PIA_RELATION_VALIDATION_EVIDENCE_ROOT")
    if evidence_root:
        tmp_path = Path(evidence_root) / "development_operation_counterfactual_artifacts"
        tmp_path.mkdir(exist_ok=False)
    frozen = {path.relative_to(source).as_posix(): file_sha256(path)
              for path in source.rglob("*") if path.is_file()}
    manifest = tuple((source / "teachers").rglob("manifest.json"))[0]
    teachers = read_independent_relation_fact_source(source / "data/ph2/facts.jsonl", manifest.parent)
    composition = read_relation_composition_graph(source / "rules.sqlite3")[0]
    rule, schemas = composition.rule, {schema.relation: schema for schema in composition.schemas}
    frames, _ = read_relation_question_graph(source / "questions.sqlite3")
    atoms = (
        OperationAtom(schemas[rule.first_relation], tuple(sorted(((rule.first_input_role, 1),
            (rule.first_join_role, 2)), key=lambda row: row[0].stable_key()))),
        OperationAtom(schemas[rule.second_relation], tuple(sorted(((rule.second_join_role, 2),
            (rule.second_output_role, 3)), key=lambda row: row[0].stable_key()))),
    )
    baseline = IntegerOperationProgram(frames[0].operation_key, (1, 2), atoms, (),
        (OperationProjection(1, tuple(ObjectIdentity.from_stable_key(role) for role in frames[0].roles)),))
    swapped = replace(baseline, projections=(OperationProjection(2,
        (rule.second_join_role, rule.second_output_role)),))
    rows = []
    for name, program in (("baseline", baseline), ("swapped", swapped)):
        path = tmp_path / (name + ".train.json")
        path.write_bytes(canonical_json_bytes({"license_id": "CC0-1.0", "owner": "teacher", "split": "train",
                                               "program_key": list(program.stable_key())}) + b"\n")
        witness = compile_train_operation_program(path, teachers)
        graph = tmp_path / (name + ".sqlite3")
        materialize_operation_program_graph((witness,), graph)
        rows.append({"name": name, "program_sha256": file_sha256(graph), "source_sha256": file_sha256(path)})
    materialize_operation_program_graph((), tmp_path / "removed.sqlite3")
    rows.append({"name": "removed", "program_sha256": file_sha256(tmp_path / "removed.sqlite3")})
    surface = json.loads((source / "question.train.json").read_bytes())["question"]
    if evidence_root:
        write_preflight_json(tmp_path, "identity.json", {
            "stage": "THREE_PROGRAM_COUNTERFACTUALS_FROZEN_NOT_WAVE1_TRAINING",
            "readonly_core_source": str(source), "source_files": frozen, "program_variants": rows,
            "input_values": list(map(ord, surface)), "four_way_isolation": "UNKNOWN_NOT_A_COURSE",
            "held_out_novelty": "UNKNOWN_TRAIN_QUESTION", "operation_family_qualification": "UNKNOWN",
            "query_count_planned": 3, "restart_authorized": 0, "free_dialogue_complete": 0,
        })
        write_preflight_json(tmp_path, "checkpoint.json", {
            "stage": "FROZEN_BEFORE_ORIGINAL_CALLS", "completed_calls": 0, "free_dialogue_complete": 0})
    reports, executions = [], []
    for number, name in enumerate(("baseline", "swapped", "removed"), 1):
        with TrainedGenerationConnectorRuntime(source / "core.sqlite3") as generator, TrainedGraphQueryBridge(
                source / "core.sqlite3", memory_database=tmp_path / f"memory-{number:03d}.sqlite3",
                session_id=92240, surface_generator=generator,
                relation_question_graph_database=source / "questions.sqlite3",
                operation_program_graph_database=tmp_path / (name + ".sqlite3")) as bridge:
            report = run_natural_relation_case(bridge, surface, tmp_path, f"cases/{number:03d}", case_ordinal=number,
                max_depth=24, max_nodes=512, max_edges=1024, max_reads=2048, max_case_wall_seconds=300)
        value = json.loads((tmp_path / f"cases/{number:03d}/trace.int.json").read_bytes())
        reports.append(report)
        if number <= 2:
            assert report["termination"] == 1 and report["delivery_claim_count"] == 1
            assert report["capture_present"] == report["positive_chain_phase_coverage"] == 1
            trace = GraphQueryTraceEvidence(value)
            execution = OperationExecution.from_stable_key(trace.resolve_ref(
                value["generation"]["operation_execution"]["integer_key_ref"]))
            executions.append(execution)
            assert value["generation"]["operation_truth_added"] == 0
            assert value["response_plan"]["claim_refs"] == [list(execution.outputs[0][0])]
        else:
            assert report["termination"] != 1 and report["delivery_claim_count"] == 0
            assert not value["response_surface"] and not value["response_plan"] and not value["generation"]
        if evidence_root:
            write_preflight_json(tmp_path, "checkpoint.json", {
                "stage": "ORIGINAL_GRAPH_COUNTERFACTUAL_CALLS_IN_PROGRESS" if number < 3 else "ORIGINAL_GRAPH_COUNTERFACTUAL_CALLS_ENDED",
                "completed_calls": number, "last_call_receipt": report, "free_dialogue_complete": 0})
    assert executions[0].context_identity == executions[1].context_identity
    assert executions[0].variables == executions[1].variables and executions[0].matches == executions[1].matches
    assert executions[0].outputs != executions[1].outputs
    assert reports[0]["response_surface"] != reports[1]["response_surface"]
    assert frozen == {path.relative_to(source).as_posix(): file_sha256(path)
                      for path in source.rglob("*") if path.is_file()}
    from pure_integer_ai.experiments.operation_program_call_audit import audit_saved_operation_counterfactuals
    audit = audit_saved_operation_counterfactuals(tmp_path, source)
    assert audit["source_write_count"] == audit["query_reexecution_count"] == audit["generation_reexecution_count"] == 0
    assert audit["qualified_semantic_operation_families"] == 0
    if evidence_root:
        write_preflight_json(Path(evidence_root), "operation_counterfactual_cold_audit.json", audit)
