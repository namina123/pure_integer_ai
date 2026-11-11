"""Fresh licensed Core, native composition and adversarial shared-role contracts."""
from __future__ import annotations

from pure_integer_ai.experiments.batch_case_storage import load_case_provenance

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.integer_operation_execution import READY, execute_integer_operation
from pure_integer_ai.cognition.shared.integer_operation_program import (
    UNEQUAL, IntegerOperationProgram, OperationAtom, OperationProjection,
)
from pure_integer_ai.cognition.shared.query_state import BindingEntry, QueryState, TERMINATION_ANSWER_CLOSED
from pure_integer_ai.cognition.understanding.query_input_structure import TrainedInputStructureProjector
from pure_integer_ai.cognition.understanding.query_structure_adapter import RelationSurfaceStructureAdapter
from pure_integer_ai.cognition.shared.representation_rendering import representation_parts
from pure_integer_ai.cognition.shared.query_state import EvidenceEntry
from pure_integer_ai.experiments.conversation_typed_relation_bridge import build_integer_w06_learning_runtime
from pure_integer_ai.experiments.generation_organization import ClosedGenerationEvidence, bind_query_evidence, build_multi_response_plan
from pure_integer_ai.experiments.independent_relation_fact_source import compile_independent_relation_fact_source, read_independent_relation_fact_source
from pure_integer_ai.experiments.integer_relation_generation_bridge import materialize_integer_relation_generation
from pure_integer_ai.experiments.integer_relation_graph_course import write_integer_relation_graph_course
from pure_integer_ai.experiments.language_protocol_runtime import install_language_graph_protocols
from pure_integer_ai.experiments.natural_relation_preflight import (
    file_sha256, run_natural_relation_case, write_preflight_json,
)
from pure_integer_ai.experiments.operation_request_call_audit import restore_saved_operation_request
from pure_integer_ai.experiments.operation_request_course import (
    compile_train_operation_request, materialize_operation_request_graph,
    read_operation_request_graph,
)
from pure_integer_ai.experiments.operation_composition_course import OperationCompositionWitness, compile_train_operation_composition
from pure_integer_ai.experiments.operation_composition_materialization import materialize_native_operation_composition, read_native_operation_composition_graph
from pure_integer_ai.experiments.operation_program_generation import generate_operation_response
from pure_integer_ai.experiments.operation_program_course import compile_train_operation_program
from pure_integer_ai.experiments.operation_program_query import OPERATION_EXECUTION_ROLE, read_operation_core_context, operation_plan_matches
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.run_conversation_training import dialogue_semantic_protocols
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_generation_connector_runtime import TrainedGenerationConnectorRuntime
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
from pure_integer_ai.experiments.trained_multi_claim_composer import (
    GRAPH_KIND, PART_CLAIM_PREDICATE,
)
from pure_integer_ai.experiments.trained_operation_composition import TrainedOperationCompositionRuntime
from pure_integer_ai.experiments.trained_operation_program_graph import materialize_operation_program_graph
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.trained_input_concept_links import restore_input_concept_links
from pure_integer_ai.experiments.trained_open_role_frames import restore_open_role_frames
from pure_integer_ai.experiments.saved_relation_memory_audit import audit_saved_relation_memory
from pure_integer_ai.experiments.integer_relation_graph_course import integer_relation_graph_course_to_w06_adapter
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack
from pure_integer_ai.storage.backend import SQLiteBackend


def _build(root, *, before_train=None):
    rows = (
        {"anchor": "属于", "case_id": "red-member", "relation": "MEMBER", "sample_role": "support",
         "seed_id": "red-member", "split": "train", "surface": "澄石甲属于红枝群。",
         "terms": [["澄石甲", "entity", 202610131101], ["红枝群", "set", 202610131102]]},
        {"anchor": "属于", "case_id": "blue-member", "relation": "MEMBER", "sample_role": "support",
         "seed_id": "blue-member", "split": "train", "surface": "澄石甲属于蓝叶群。",
         "terms": [["澄石甲", "entity", 202610131101], ["蓝叶群", "set", 202610131103]]},
    )
    source = root / "data/ph2/facts.jsonl"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    facts = compile_independent_relation_fact_source(source, root / "teachers")
    course_path = root / "course.int.json"
    write_integer_relation_graph_course(course_path, facts.course)
    if before_train is not None:
        before_train(source, facts, course_path)
    database = root / "core.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        context = make_train_context(backend)
        _, occurrence, span = dialogue_semantic_protocols()
        install_language_graph_protocols(context, occurrence_protocol=occurrence, span_protocol=span)
        learning = build_integer_w06_learning_runtime(backend, context, (course_path,), require_full_curriculum=False)
        assert learning.report().active_candidate_count == 2
        backend.commit()
    finally:
        backend.close()
    materialize_integer_relation_generation(database, branch_key=(92310, 1), family_key=(92310, 2),
        representative_limit=2, require_full_active_coverage=True)
    return _finish(root, facts)


def _finish(root, facts):
    """Resume the same unfinished syntax stage; never repeat Core fact learning."""
    database = root / "core.sqlite3"
    write_preflight_json(root, "syntax_continuation_started.json", {
        "stage": "BEFORE_NATIVE_SYNTAX_APPEND", "existing_core_sha256": file_sha256(database),
        "w06_fact_training_reexecution_count": 0, "natural_query_count": 0, "free_dialogue_complete": 0})
    with TrainedRelationGraphRuntime(database) as core, TrainedGenerationConnectorRuntime(database) as generator:
        active = core.active_surface_facts()
        keys = tuple(fact.proposition.stable_key() for fact in sorted(active, key=lambda item: item.evidence_surface))
        instruction = root / "composition.train.json"
        if not instruction.exists():
            instruction.write_bytes(canonical_json_bytes({"claim_keys": [list(key) for key in keys],
                "separator_values": [65292], "variant_id": 1, "owner": "teacher", "split": "train", "license_id": "CC0-1.0"}) + b"\n")
        compiled = compile_train_operation_composition(instruction, facts, core, generator)
    materialized = materialize_native_operation_composition(compiled, database, root / "composition.sqlite3")
    adapter = integer_relation_graph_course_to_w06_adapter(facts.course, require_full_curriculum=False)
    assert len(adapter.schemas) == 1
    schema = adapter.schemas[0]
    by_roles = tuple(slot.role for slot in schema.slots)
    assert len(by_roles) == 2
    atom1 = OperationAtom(schema, tuple((role, index) for index, role in enumerate(by_roles, 1)))
    atom2 = OperationAtom(schema, ((by_roles[0], 1), (by_roles[1], 3)))
    program = IntegerOperationProgram((92311, 1), (1, 2), (atom1, atom2), ((UNEQUAL, 2, 3),),
        (OperationProjection(1, by_roles), OperationProjection(2, by_roles)), realization_key=(GRAPH_KIND, 1))
    program_source = root / "program.train.json"
    program_source.write_bytes(canonical_json_bytes({"program_key": list(program.stable_key()),
        "owner": "teacher", "split": "train", "license_id": "CC0-1.0"}) + b"\n")
    witness = compile_train_operation_program(program_source, facts)
    materialize_operation_program_graph((witness,), root / "program.sqlite3")
    write_preflight_json(root, "checkpoint.json", {"stage": "NATIVE_SHARED_ROLE_COMPOSITION_OFFLINE_CONTRACT",
        "natural_query_count": 0, "materialization": materialized, "free_dialogue_complete": 0})
    return root, facts, compiled, witness


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    saved = os.environ.get("PIA_SAVED_NATIVE_COMPOSITION_SOURCE")
    if saved:
        root = Path(saved).resolve(strict=True)
        if (root / "composition.sqlite3").exists() or (root / "program.sqlite3").exists():
            raise ValueError("completed native composition cannot repeat syntax or program training")
        facts = read_independent_relation_fact_source(root / "data/ph2/facts.jsonl",
                                                      next((root / "teachers").rglob("manifest.json")).parent)
        return _finish(root, facts)
    evidence = os.environ.get("PIA_RELATION_VALIDATION_EVIDENCE_ROOT")
    root = Path(evidence) / "development_native_composition_artifacts" if evidence else tmp_path_factory.mktemp("native-composition")
    if evidence:
        root.mkdir(exist_ok=False)
    return _build(root)


def _actual_components(root, witness):
    core, generator = TrainedRelationGraphRuntime(root / "core.sqlite3"), TrainedGenerationConnectorRuntime(root / "core.sqlite3")
    facts = read_operation_core_context(core)
    first = facts[0]
    variables = tuple((variable, next(binding.filler for binding in first.definition.bindings if binding.role == role))
                      for role, variable in witness.program.atoms[0].role_variables)
    execution = execute_integer_operation(witness, (92312, 1), variables, facts)
    assert execution.status == READY and len(execution.outputs) == 2
    surfaces = {fact.proposition.stable_key(): fact for fact in core.active_surface_facts()}
    components = tuple((core.generation_input(surfaces[key].proposition), surfaces[key], None) for key, _ in execution.outputs)
    bindings = tuple(BindingEntry(binding.role.stable_key(), binding.filler.stable_key(), scope_key=fact.proposition_key)
                     for fact in facts for binding in fact.definition.bindings)
    entries = []
    for fact in facts:
        for key in fact.evidence_keys:
            fields = _unpack(key, 3, 6)
            entries.append(EvidenceEntry(fields[0], fields[1], key[0], key[1], key[2], *fields[2:]))
    entries.append(EvidenceEntry(witness.source.stable_key(), witness.program.operation_key,
        space=1, polarity=3, evidence_key=(92230, *execution.query_key), scope_key=execution.query_key,
        payload_key=execution.stable_key()))
    state = QueryState(execution.query_key, evidence_closed=1, generation_ready=1,
        best_candidate_key=execution.outputs[0][0],
        termination=TERMINATION_ANSWER_CLOSED, evidence=tuple(entries), bindings=(*bindings,
            BindingEntry(OPERATION_EXECUTION_ROLE, execution.stable_key(), scope_key=execution.query_key)))
    return core, generator, execution, components, state


def test_single_claim_rewrite_uses_trained_frame_and_keeps_current_claim_evidence(monkeypatch):
    evidence_root = os.environ.get("PIA_RELATION_VALIDATION_EVIDENCE_ROOT")
    if evidence_root is None:
        pytest.skip("requires an explicitly named K run root for Core training")
    root = Path(evidence_root).resolve()
    if root.drive.upper() != "K:" or root.exists():
        raise ValueError("single-claim integration requires one new named K identity")
    root.mkdir(parents=False, exist_ok=False)
    write_preflight_json(root, "identity.json", {
        "format": "PURE_INTEGER_SINGLE_CLAIM_REWRITE_INTEGRATION_V1",
        "identity": root.name,
        "source": "original fictional two-fact CC0 integration fixture",
        "license": "CC0-1.0", "training_started": 1,
        "production_authorized": 0, "free_dialogue_complete": 0,
    })
    write_preflight_json(root, "split_manifest.json", {
        "status": "FROZEN_TRAIN_ONLY_INTEGRATION_FIXTURE",
        "train_case_count": 2, "held_out_case_count": 0,
        "negative_case_count": 0, "conflict_case_count": 0,
        "held_out_read_by_preflight": 0, "free_dialogue_complete": 0,
    })
    write_preflight_json(root, "resource_budget.json", {
        "max_core_training_seconds": 300, "max_case_seconds": 30,
        "max_total_seconds": 600, "max_peak_working_set_bytes": 1073741824,
        "free_dialogue_complete": 0,
    })

    def freeze_source(source, facts, _course_path):
        manifest = next((root / "teachers").rglob("manifest.json"))
        write_preflight_json(root, "source_permission.json", {
            "source_identity": list(facts.course.source_identity),
            "license_identity": list(facts.course.license_identity),
            "license_id": "CC0-1.0", "owner": "teacher",
            "source_sha256": file_sha256(source),
            "teacher_manifest_sha256": file_sha256(manifest),
            "source_body_rows_added": 0, "whole_sentence_or_answer_table": 0,
            "free_dialogue_complete": 0,
        })

    root, facts, _prior_course, _prior_program = _build(root, before_train=freeze_source)
    source_path = root / "rewrite-composition.train.json"
    with TrainedRelationGraphRuntime(root / "core.sqlite3") as core, TrainedGenerationConnectorRuntime(
            root / "core.sqlite3") as relation_generator:
        chosen = core.active_surface_facts()[0]
        roles = tuple(binding.role for binding in chosen.bindings)
        source_path.write_bytes(canonical_json_bytes({
            "claim_keys": [list(chosen.proposition.stable_key())],
            "separator_values": [],
            "output_gaps": [[[], list(map(ord, "是")), list(map(ord, "的一名成员。"))]],
            "variant_id": 2, "owner": "teacher", "split": "train", "license_id": "CC0-1.0",
        }) + b"\n")
        rewrite_course = compile_train_operation_composition(source_path, facts, core, relation_generator)
    materialize_native_operation_composition(
        rewrite_course, root / "core.sqlite3", root / "rewrite-composition.sqlite3")

    schema = integer_relation_graph_course_to_w06_adapter(
        facts.course, require_full_curriculum=False).schemas[0]
    atom = OperationAtom(schema, tuple(sorted(
        ((role, index) for index, role in enumerate(roles, 1)),
        key=lambda item: item[0].stable_key())))
    program = IntegerOperationProgram(
        (92311, 2), (1, 2), (atom,), (), (OperationProjection(1, roles),),
        realization_key=(GRAPH_KIND, 2))
    program_path = root / "rewrite-program.train.json"
    program_path.write_bytes(canonical_json_bytes({
        "program_key": list(program.stable_key()), "owner": "teacher",
        "split": "train", "license_id": "CC0-1.0",
    }) + b"\n")
    witness = compile_train_operation_program(program_path, facts)
    materialize_operation_program_graph((witness,), root / "rewrite-program.sqlite3")

    with TrainedRelationGraphRuntime(root / "core.sqlite3") as core, TrainedGenerationConnectorRuntime(
            root / "core.sqlite3") as relation_generator:
        runtime = TrainedOperationCompositionRuntime(
            root / "rewrite-composition.sqlite3", relation_generator, core,
            root / "rewrite-program.sqlite3")
        calls = {"relation_generation": 0, "composition": 0, "plan": 0, "binding": 0}
        generate_relation = runtime.generate_relation
        def checked_generate_relation(source, fact):
            calls["relation_generation"] += 1
            try:
                generated = generate_relation(source, fact)
            except Exception as error:
                raise AssertionError(
                    f"base relation generation raised {type(error).__name__}: {error}") from error
            assert generated.connector is not None and generated.representations, (
                "base relation generation lacks connector or representations")
            return generated
        runtime.generate_relation = checked_generate_relation
        compose = runtime.generate_operation_composition
        def checked_compose(*args, **kwargs):
            calls["composition"] += 1
            try:
                return compose(*args, **kwargs)
            except Exception as error:
                raise AssertionError(
                    f"native operation realization rejected: {type(error).__name__}: {error}") from error
        runtime.generate_operation_composition = checked_compose
        from pure_integer_ai.experiments import operation_program_generation as operation_generation
        build_plan = operation_generation.build_multi_response_plan
        def checked_build_plan(*args, **kwargs):
            calls["plan"] += 1
            try:
                plan = build_plan(*args, **kwargs)
            except Exception as error:
                raise AssertionError(
                    f"ResponsePlan builder raised {type(error).__name__}: {error}") from error
            assert plan is not None, "single-claim ResponsePlan builder rejected the trained realization"
            return plan
        monkeypatch.setattr(operation_generation, "build_multi_response_plan", checked_build_plan)
        bind_plan = operation_generation.bind_query_evidence
        def checked_bind_plan(plan, current_state):
            calls["binding"] += 1
            try:
                return bind_plan(plan, current_state)
            except Exception as error:
                raise AssertionError(
                    f"QueryState evidence binding raised {type(error).__name__}: {error}") from error
        monkeypatch.setattr(operation_generation, "bind_query_evidence", checked_bind_plan)
        match_plan = operation_generation.operation_plan_matches
        def checked_match_plan(*args, **kwargs):
            matched = match_plan(*args, **kwargs)
            assert matched, "single-claim ResponsePlan does not match the integer projection"
            return matched
        monkeypatch.setattr(operation_generation, "operation_plan_matches", checked_match_plan)
        core_facts = read_operation_core_context(core)
        chosen_core = next(item for item in core_facts
                           if item.proposition_key == chosen.proposition.stable_key())
        variables = tuple((index, binding.filler)
                          for index, role in enumerate(roles, 1)
                          for binding in chosen.bindings if binding.role == role)
        query_key = (92312, 1)
        execution = execute_integer_operation(witness, query_key, variables, core_facts)
        assert execution.status == READY and tuple(key for key, _ in execution.outputs) == (
            chosen.proposition.stable_key(),)

        evidence = []
        used = {key for match in execution.matches for key in match}
        for fact in core_facts:
            if fact.proposition_key not in used:
                continue
            for key in fact.evidence_keys:
                fields = _unpack(key, 3, 6)
                evidence.append(EvidenceEntry(fields[0], fields[1], key[0], key[1], key[2], *fields[2:]))
        evidence.append(EvidenceEntry(witness.source.stable_key(), witness.program.operation_key,
            space=1, polarity=3, evidence_key=(92230, *query_key), scope_key=query_key,
            payload_key=execution.stable_key()))
        core_bindings = tuple(BindingEntry(
            binding.role.stable_key(), binding.filler.stable_key(),
            scope_key=chosen.proposition.stable_key())
            for binding in chosen.bindings)
        state = QueryState(query_key, evidence=tuple(evidence), termination=TERMINATION_ANSWER_CLOSED,
            evidence_closed=1, generation_ready=1, best_candidate_key=execution.outputs[0][0],
            bindings=(*core_bindings,
                BindingEntry(OPERATION_EXECUTION_ROLE, execution.stable_key(), scope_key=query_key)))
        from pure_integer_ai.experiments.operation_companion_query import consume_operation_companion
        state, _ = consume_operation_companion(core,
            {frame.proposition.stable_key(): frame for frame in core.active_surface_frames()},
            {fact.proposition.stable_key(): fact for fact in core.active_surface_facts()}, execution, state)
        assert execution.witness.program.realization_key == (GRAPH_KIND, 2)
        assert execution.outputs[0][0] == chosen.proposition.stable_key()
        assert set(execution.outputs[0][1]) == {
            binding.role.stable_key() for binding in chosen.bindings}
        variant = next(item for item in runtime.variants if item.variant_id == 2)
        assert variant.source_identity == execution.witness.source_identity
        assert variant.license_identity == execution.witness.license_identity
        assert tuple(part[2] for part in variant.parts
                     if part[0] == PART_CLAIM_PREDICATE) == (chosen.predicate.stable_key(),)
        try:
            ready, plan, generation = generate_operation_response(
                core, runtime, execution, state)
        except Exception as error:
            write_preflight_json(root, "operation_generation_attempt.json", {
                "status": "FAIL", "error_type": type(error).__name__,
                "error": str(error), "stage_calls": calls,
                "operation_execution": list(execution.stable_key()),
                "free_dialogue_complete": 0,
            })
            raise
        write_preflight_json(root, "operation_generation_attempt.json", {
            "status": "PLAN_RETURNED" if plan is not None else "EMPTY_PLAN",
            "stage_calls": calls,
            "operation_execution": list(execution.stable_key()),
            "surface_values": [] if generation is None else generation.get("surface_values", []),
            "free_dialogue_complete": 0,
        })
        assert plan is not None, f"empty operation plan; stage calls={calls}"
        assert plan is not None and operation_plan_matches(execution, bind_query_evidence(plan, ready))
        assert tuple(claim.stable_key() for claim in plan.claim_refs) == (chosen.proposition.stable_key(),)
        entity, group = (binding.surface for binding in chosen.bindings)
        assert plan.surface() == f"{entity}是{group}的一名成员。"
        assert generation["operation_truth_added"] == 0
        assert generation["native_composition_witness"]
        write_preflight_json(root, "operation_generation_receipt.json", {
            "status": "PASS", "stage_calls": calls,
            "operation_execution": list(execution.stable_key()),
            "claim_refs": [list(claim.stable_key()) for claim in plan.claim_refs],
            "surface_values": generation["surface_values"],
            "native_composition_witness": generation["native_composition_witness"],
            "operation_truth_added": 0, "free_dialogue_complete": 0,
        })
        relation_generator.close()
        core.close()

    source_rows = tuple(json.loads(line) for line in facts.source_path.read_text(
        encoding="utf-8").splitlines())
    seed = next(row for row in source_rows if all(
        binding.surface in {term[0] for term in row["terms"]}
        for binding in chosen.bindings))
    request_text = f"请把{entity}和{group}的已知关系换一种说法。"
    variable_by_role = {role: variable for role, variable in atom.role_variables}
    request_path = root / "rewrite-request.train.json"
    request_path.write_bytes(canonical_json_bytes({
        "question": request_text,
        "readers": [{"variable": variable_by_role[binding.role],
                     "seed_id": seed["seed_id"],
                     "role_key": list(binding.role.stable_key())}
                    for binding in chosen.bindings],
        "owner": "teacher", "split": "train", "license_id": "CC0-1.0",
    }) + b"\n")
    request_frame = compile_train_operation_request(request_path, facts, witness)
    materialize_operation_request_graph((request_frame,), root / "rewrite-requests.sqlite3")
    write_preflight_json(root, "rewrite_request_permission.json", {
        "source_identity": list(facts.course.source_identity),
        "license_identity": list(facts.course.license_identity),
        "license_id": "CC0-1.0", "owner": "teacher", "split": "train",
        "request_source_sha256": file_sha256(request_path),
        "request_frame_key": list(request_frame.stable_key()),
        "free_dialogue_complete": 0,
    })
    with TrainedRelationGraphRuntime(root / "core.sqlite3") as query_core, TrainedGenerationConnectorRuntime(
            root / "core.sqlite3") as query_relation_generator:
        query_generator = TrainedOperationCompositionRuntime(
            root / "rewrite-composition.sqlite3", query_relation_generator, query_core,
            root / "rewrite-program.sqlite3")
        with TrainedGraphQueryBridge(
                root / "core.sqlite3", memory_database=root / "ordinary-memory.sqlite3",
                session_id=92313, surface_generator=query_generator,
                operation_request_graph_database=root / "rewrite-requests.sqlite3",
                operation_program_graph_database=root / "rewrite-program.sqlite3") as bridge:
            report = run_natural_relation_case(
                bridge, request_text, root, "calls/ordinary-rewrite", case_ordinal=1,
                max_depth=24, max_nodes=512, max_edges=1024, max_reads=4096,
                max_case_wall_seconds=60)
    assert report["termination"] == TERMINATION_ANSWER_CLOSED
    assert report["response_surface"] == f"{entity}是{group}的一名成员。"
    assert report["delivery_claim_count"] == 1
    assert report["capture_present"] == report["positive_chain_phase_coverage"] == 1
    assert report["hypothesis_count"] > 0 and report["evidence_count"] > 0
    assert report["projection"]["operation_keys"] == [list(program.operation_key)]

    case_root = root / "calls/ordinary-rewrite"
    trace_value = json.loads((case_root / "trace.int.json").read_bytes())
    captured = load_case_provenance(case_root / "case_provenance.json")
    with TrainedRelationGraphRuntime(root / "core.sqlite3") as audit_core:
        surfaces = audit_core.active_surface_facts()
        projector = TrainedInputStructureProjector(
            RelationSurfaceStructureAdapter.from_facts(surfaces),
            concept_links=restore_input_concept_links(audit_core),
            open_role_frames=restore_open_role_frames(
                audit_core.active_surface_frames(), surfaces))
        frames = read_operation_request_graph(root / "rewrite-requests.sqlite3")
        prepared = restore_saved_operation_request(trace_value, report, projector, frames)
    cold_audit = audit_saved_relation_memory(
        root / "ordinary-memory.sqlite3", trace_value, report, prepared,
        captured=captured, require_operation_request_memory=True)
    assert cold_audit["memory_ohe"]["status"] == (
        "MATCHED_DURABLE_INTAKE_OHE_QUERYSTATE_AND_PLAN_REFERENCES")
    assert cold_audit["memory_request_binding_status"] == (
        "MATCHED_CURRENT_SOURCE_LOCAL_FULL_BINDING")
    assert cold_audit["delivery_proof"]["status"] == (
        "MATCHED_DURABLE_PROOF_CONTENT_TO_CAPTURED_PLAN")
    assert cold_audit["delivery_events"]["status"] == (
        "MATCHED_DURABLE_DELIVERY_EPISODE_AND_USES")
    write_preflight_json(root, "ordinary_rewrite_cold_audit.json", cold_audit)
    write_preflight_json(root, "ordinary_rewrite_receipt.json", {
        "status": "PASS_DEVELOPMENT_SINGLE_CLAIM_ONLY",
        "input_values": list(map(ord, request_text)),
        "operation_key": list(program.operation_key),
        "response_values": list(map(ord, report["response_surface"])),
        "query_wall_ns": report["wall_ns"], "query_cpu_ns": report["cpu_ns"],
        "memory_ohe_status": cold_audit["memory_ohe"]["status"],
        "request_memory_status": cold_audit["memory_request_binding_status"],
        "delivery_proof_status": cold_audit["delivery_proof"]["status"],
        "delivery_events_status": cold_audit["delivery_events"]["status"],
        "qualified_semantic_operation_families": 0,
        "held_out_novel_behavior_count": 0, "free_dialogue_complete": 0,
    })


def test_native_graph_restores_exact_source_template_and_claim_local_role_readers(prepared):
    root, _, compiled, _ = prepared
    before = {path.name: file_sha256(path) for path in (root / "core.sqlite3", root / "composition.sqlite3")}
    variants, witnesses, _ = read_native_operation_composition_graph(root / "composition.sqlite3")
    assert variants == (compiled.variant,) and witnesses == (compiled.witness,)
    assert OperationCompositionWitness.from_stable_key(compiled.witness.stable_key()) == compiled.witness
    assert len({role for _, role, _, _ in compiled.witness.readers}) == 2
    assert len({scoped for _, _, scoped, _ in compiled.witness.readers}) == 4
    assert before == {path.name: file_sha256(path) for path in (root / "core.sqlite3", root / "composition.sqlite3")}


def test_native_g_layers_generate_shared_subject_twice_and_keep_all_four_plan_slots(prepared):
    root, _, compiled, witness = prepared
    core, generator, execution, components, state = _actual_components(root, witness)
    try:
        runtime = TrainedOperationCompositionRuntime(root / "composition.sqlite3", generator, core, root / "program.sqlite3")
        generated = runtime.generate_operation_composition(components, state, execution=execution, variant_id=1)
        assert generated.trace[0] == 2 and len(generated.representations) == 7
        units = tuple(unit for item in generated.representations for unit in representation_parts(item)[1])
        assert units == tuple(map(ord, generated.surface))
        assert generated.surface.count("澄石甲") == 2
        evidences = tuple(ClosedGenerationEvidence(fact.bindings, generated, fact.proposition.stable_key(),
            fact.predicate.stable_key(), fact.source_hash) for _, fact, _ in components)
        plan = build_multi_response_plan(evidences, generated, claim_slot_contract=execution.outputs)
        assert plan is not None and len(plan.claim_refs) == 2 and len(plan.slot_sequence) == 4
        assert operation_plan_matches(execution, bind_query_evidence(plan, state))
        assert len(core.active_surface_facts()) == 2
        write_preflight_json(root, "native_generation_contract.json", {"status": "PASS_OFFLINE_NATIVE_COMPOSITION_ONLY",
            "surface_values": list(units), "execution_key": list(execution.stable_key()),
            "native_composition_witness": list(compiled.witness.stable_key()), "plan": plan.to_dict(),
            "new_core_claim_count": 0, "natural_query_count": 0, "free_dialogue_complete": 0})
    finally:
        generator.close(); core.close()


@pytest.mark.parametrize("field", ["owner", "split", "license_id", "claim_keys"])
def test_native_grammar_rejects_evaluator_or_unlicensed_and_unknown_claims(prepared, tmp_path, field):
    root, facts, _, _ = prepared
    row = json.loads((root / "composition.train.json").read_bytes())
    row[field] = {"owner": "evaluator", "split": "held_out", "license_id": "UNKNOWN", "claim_keys": [[1], [2]]}[field]
    path = tmp_path / "bad.train.json"
    path.write_bytes(canonical_json_bytes(row) + b"\n")
    with TrainedRelationGraphRuntime(root / "core.sqlite3") as core, TrainedGenerationConnectorRuntime(root / "core.sqlite3") as generator:
        with pytest.raises(ValueError):
            compile_train_operation_composition(path, facts, core, generator)


@pytest.mark.parametrize("field", ["bindings", "evidence"])
def test_native_composition_requires_same_query_proof_and_complete_current_premises(prepared, field):
    root, _, _, witness = prepared
    core, generator, execution, components, state = _actual_components(root, witness)
    try:
        runtime = TrainedOperationCompositionRuntime(root / "composition.sqlite3", generator, core, root / "program.sqlite3")
        changed = state.with_(**{field: ()})
        with pytest.raises(ValueError, match="proof|premise Evidence"):
            runtime.generate_operation_composition(components, changed, execution=execution, variant_id=1)
    finally:
        generator.close(); core.close()
