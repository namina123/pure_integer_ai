"""Development integration of two admitted Core facts in one natural turn.

This builds a fresh test Core; it is not a Wave 1 production acceptance run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pure_integer_ai.experiments.conjoined_relation_generation_course import (
    compile_train_conjoined_generation,
)
from pure_integer_ai.cognition.shared.query_state import TERMINATION_ANSWER_CLOSED
from pure_integer_ai.experiments.conjoined_relation_question_course import (
    compile_train_conjoined_question,
)
from pure_integer_ai.experiments.conversation_typed_relation_bridge import (
    build_integer_w06_learning_runtime,
)
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.independent_relation_fact_source import (
    compile_independent_relation_fact_source,
)
from pure_integer_ai.experiments.integer_relation_connected_course import _seed
from pure_integer_ai.experiments.integer_relation_generation_bridge import (
    materialize_integer_relation_generation,
)
from pure_integer_ai.experiments.integer_relation_graph_course import (
    write_integer_relation_graph_course,
)
from pure_integer_ai.experiments.language_protocol_runtime import install_language_graph_protocols
from pure_integer_ai.experiments.natural_relation_preflight import (
    file_sha256, run_natural_relation_case, write_preflight_json,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.run_conversation_training import (
    dialogue_semantic_protocols,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
from pure_integer_ai.experiments.trained_input_discourse_conjunction_graph import (
    materialize_input_discourse_conjunction_graph,
)
from pure_integer_ai.experiments.trained_input_discourse_graph import (
    materialize_input_discourse_operator_graph,
)
from pure_integer_ai.experiments.trained_input_discourse_operation_graph import (
    materialize_input_discourse_operation_graph,
)
from pure_integer_ai.experiments.trained_multi_claim_composer import (
    TrainedMultiClaimComposer, materialize_multi_claim_composition_graph,
)
from pure_integer_ai.experiments.trained_relation_question_graph import (
    materialize_relation_question_graph,
)
from pure_integer_ai.storage.backend import SQLiteBackend


def _rows():
    return (
        {"anchor": "导致", "case_id": "shared-cause", "relation": "CAUSES",
         "sample_role": "support", "seed_id": "shared-cause", "split": "train",
         "surface": "青苔闸开启导致萤砂灯点亮。",
         "terms": [["青苔闸开启", "event", 202610130201],
                   ["萤砂灯点亮", "event", 202610130202]]},
        {"anchor": "之前", "case_id": "shared-before", "relation": "EVENT_BEFORE",
         "sample_role": "support", "seed_id": "shared-before", "split": "train",
         "surface": "萤砂灯点亮发生在蓝曜扉闭合之前。",
         "terms": [["萤砂灯点亮", "event", 202610130202],
                   ["蓝曜扉闭合", "event", 202610130203]]},
    )


def test_one_natural_turn_consumes_two_new_core_facts_and_delivers(tmp_path):
    evidence_root = os.environ.get("PIA_RELATION_VALIDATION_EVIDENCE_ROOT")
    if evidence_root:
        tmp_path = Path(evidence_root) / "development_artifacts"
        tmp_path.mkdir(exist_ok=False)
    rows = _rows()
    source = tmp_path / "data" / "ph2" / "train_facts.jsonl"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    teachers = compile_independent_relation_fact_source(source, tmp_path / "teacher_pack")
    course = tmp_path / "core_course.int.json"
    write_integer_relation_graph_course(course, teachers.course)
    core = tmp_path / "core.sqlite3"
    backend = SQLiteBackend(str(core))
    try:
        context = make_train_context(backend)
        _semantic, occurrence, span = dialogue_semantic_protocols()
        install_language_graph_protocols(
            context, occurrence_protocol=occurrence, span_protocol=span)
        learned = build_integer_w06_learning_runtime(
            backend, context, (course,), require_full_curriculum=False)
        assert learned.report().active_candidate_count == 2
        backend.commit()
    finally:
        backend.close()
    generation = materialize_integer_relation_generation(
        core, branch_key=(92044, 60), family_key=(92045, 60),
        representative_limit=2, require_full_active_coverage=True)
    assert generation.active_proposition_count == 2
    question = "青苔闸开启与萤砂灯点亮有什么关系？同时萤砂灯点亮发生在蓝曜扉闭合之前吗？"
    trained = compile_train_conjoined_question(
        tuple(_seed(row, index) for index, row in enumerate(rows, 1)), question,
        source_case_key=(92002, 60), question_operation_key=(92004, 60),
        conjunction_operation_key=(92004, 61), conjunction_surface="同时",
        source_identity=teachers.course.source_identity,
        license_identity=teachers.course.license_identity)
    materialize_relation_question_graph(
        trained.frames, tmp_path / "questions.sqlite3",
        source_identity=trained.source_identity,
        license_identity=trained.license_identity)
    materialize_input_discourse_operator_graph(
        (trained.operator,), tmp_path / "operators.sqlite3",
        source_identity=trained.source_identity,
        license_identity=trained.license_identity)
    materialize_input_discourse_operation_graph(
        (trained.contract,), tmp_path / "operation_contract.sqlite3",
        source_identity=trained.source_identity,
        license_identity=trained.license_identity,
        operators=(trained.operator,))
    materialize_input_discourse_conjunction_graph(
        (trained.conjunction,), tmp_path / "conjunction.sqlite3",
        source_identity=trained.source_identity,
        license_identity=trained.license_identity)
    seeds = tuple(_seed(row, index) for index, row in enumerate(rows, 1))
    variant = compile_train_conjoined_generation(
        seeds, "；".join(seed.surface for seed in seeds), separator="；",
        variant_id=1, source_identity=trained.source_identity,
        license_identity=trained.license_identity)
    materialize_multi_claim_composition_graph(
        (variant,), tmp_path / "composer.sqlite3")
    frozen_core_sha = file_sha256(core)
    with TrainedGenerationConnectorRuntime(core) as generator:
        composer = TrainedMultiClaimComposer(tmp_path / "composer.sqlite3", generator)
        with TrainedGraphQueryBridge(
                core, memory_database=tmp_path / "memory.sqlite3", session_id=92060,
                surface_generator=composer,
                relation_question_graph_database=tmp_path / "questions.sqlite3",
                input_discourse_graph_database=tmp_path / "operators.sqlite3",
                input_discourse_operation_graph_database=tmp_path / "operation_contract.sqlite3",
                input_discourse_conjunction_graph_database=tmp_path / "conjunction.sqlite3") as bridge:
            structure = bridge.prepare_input_structure(question)
            if evidence_root:
                write_preflight_json(tmp_path, "input_projection.json", structure.integer_trace())
                write_preflight_json(tmp_path, "source_audit.json", teachers.source_audit)
                write_preflight_json(tmp_path, "checkpoint.json", {
                    "stage": "BEFORE_NATURAL_QUERY", "scope": "DEVELOPMENT_NOT_WAVE1",
                    "active_core_count": 2, "core_sha256": file_sha256(core),
                    "source_sha256": file_sha256(source), "free_dialogue_complete": 0,
                })
            assert len(structure.relation_question_bindings) == 2
            report = run_natural_relation_case(
                bridge, question, tmp_path, "cases/001", case_ordinal=1,
                max_depth=24, max_nodes=512, max_edges=1024, max_reads=2048,
                max_case_wall_seconds=300)
            result = json.loads((tmp_path / "cases/001/trace.int.json").read_text("utf-8"))
            trace = GraphQueryTraceEvidence(result)
            counts = trace.counts()
            assert result["termination"] == TERMINATION_ANSWER_CLOSED
            assert result["response_plan"] and result["generation"]
            assert len(result["response_plan"]["claim_refs"]) == 2
            assert counts["memory_hop_count"] > 0
            assert counts["companion_hop_count"] > 0
            assert counts["operation_contract_count"] == 2
            assert result["response_surface"] != question
            assert report["delivery_claim_count"] == 2
            assert report["hypothesis_count"] > 0 and report["evidence_count"] > 0
            assert report["positive_chain_phase_coverage"] == 1
            # A full parse of the reversed question has no trained output
            # predicate order; it must not reuse the forward composition.
            reversed_question = "同时".join(reversed(question.split("同时")))
            partial_question = question + "外来后缀"
            boundary = []
            for ordinal, surface in enumerate((reversed_question, partial_question), 2):
                rejected = run_natural_relation_case(
                    bridge, surface, tmp_path, f"cases/{ordinal:03d}", case_ordinal=ordinal,
                    max_depth=24, max_nodes=512, max_edges=1024, max_reads=2048,
                    max_case_wall_seconds=300)
                boundary.append(rejected)
                rejected_trace = json.loads((tmp_path / f"cases/{ordinal:03d}/trace.int.json").read_text("utf-8"))
                assert rejected["termination"] != TERMINATION_ANSWER_CLOSED
                assert rejected["delivery_claim_count"] == 0
                assert not (rejected_trace.get("response_plan") or {}).get("claim_refs")
            assert file_sha256(core) == frozen_core_sha
            if evidence_root:
                write_preflight_json(tmp_path, "checkpoint.json", {
                    "stage": "TWO_FACT_NATURAL_TURN_DELIVERED_NOT_WAVE1",
                    "core_sha256": file_sha256(core), "call_receipt": report,
                    "boundary_receipts": boundary,
                    "query_count": 3, "free_dialogue_complete": 0,
                })
