"""One new Core, independent licensed admission, then one ordinary-input session."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
import traceback

from pure_integer_ai.experiments.batch_phase_host import current_process_resource_endpoint
from pure_integer_ai.experiments.connected_relation_training_course import compile_relation_training_course
from pure_integer_ai.experiments.conversation_typed_relation_bridge import build_integer_w06_learning_runtime
from pure_integer_ai.experiments.independent_fact_preflight_contract import (
    BUDGET, INPUTS, QUESTIONS, SOURCE_FILES, SYNTAX, TEACHER,
    audit_prepared_question_carriers, authorized_fact_inputs, fact_input_source,
    inspect_independent_fact_contract, read_rows,
)
from pure_integer_ai.experiments.independent_fact_preflight_core_audit import verify_cold_admission_history
from pure_integer_ai.experiments.independent_fact_preflight_identity import (
    RUNNER, code_identity, tree_hashes, verify_frozen_files,
)
from pure_integer_ai.experiments.independent_relation_fact_admission import (
    ADMISSION_READY, admit_independent_relation_fact, grant_independent_fact_inputs,
    learn_independent_relation_fact_admission,
)
from pure_integer_ai.experiments.independent_relation_fact_source import (
    compile_independent_relation_fact_source, read_independent_relation_fact_source,
)
from pure_integer_ai.experiments.integer_relation_generation_bridge import materialize_integer_relation_generation
from pure_integer_ai.experiments.integer_relation_graph_course import write_integer_relation_graph_course
from pure_integer_ai.experiments.language_protocol_runtime import install_language_graph_protocols
from pure_integer_ai.experiments.natural_relation_preflight import (
    file_sha256, run_natural_relation_case, write_preflight_json,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.relation_runtime_phase import capture_relation_phases
from pure_integer_ai.experiments.run_conversation_training import dialogue_semantic_protocols
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_generation_connector_runtime import TrainedGenerationConnectorRuntime
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
from pure_integer_ai.experiments.trained_open_role_frames import restore_open_role_frames
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.trained_relation_question_graph import materialize_relation_question_graph
from pure_integer_ai.storage.backend import SQLiteBackend


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _regression_gate(project: Path, receipt: Path) -> dict:
    report = _json(receipt)
    identity = report["validation_identity"]
    if (report["suite"] != "independent_fact_preflight" or report["exit_code"] != 0
            or report["timed_out"] != 0 or report["code_unchanged"] != 1
            or identity.get("preflight_code_identity") != code_identity(project)
            or file_sha256(project / "scripts/validate_relation_memory_consumption.py")
               != identity["runner_sha256"]):
        raise ValueError("passing regression for the frozen preflight implementation is missing")
    verify_frozen_files(project, identity["test_file_sha256"])
    verify_frozen_files(project, identity["sample_file_sha256"])
    return {"path": str(receipt), "sha256": file_sha256(receipt),
            "scope": "DEVELOPMENT_REGRESSION_NOT_LANGUAGE_QUALIFICATION"}


def prepare(project: Path, root: Path, validation_receipt: Path) -> dict:
    """Freeze every owner and contract, without Core training or held-out queries."""
    gate = _regression_gate(project, validation_receipt)
    if shutil.disk_usage(root.parent).free < BUDGET["minimum_free_bytes"]:
        raise RuntimeError("K free space is below the frozen preflight budget")
    root.mkdir(exist_ok=False)
    try:
        hashes = {}
        for relative, original in SOURCE_FILES.items():
            source, target = project / original, root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[relative] = file_sha256(source)
            if file_sha256(target) != hashes[relative]:
                raise ValueError("frozen source byte copy changed")
        split = inspect_independent_fact_contract(root)
        matrix = {
            "admission": {"teacher_records": 9, "learnable_support": 5,
                          "disputed_or_refuted_unknown": 3, "ungranted_refuse": 1,
                          "cold_full_core_witness_required": 1},
            "held_out": {"denominator": 5, "ordinary_natural_input": 1,
                         "unique_composed_question": 1, "expected_direct_claim_only": 1,
                         "core_memory_companion_plan_generation_delivery": 1,
                         "cold_proof_events_current_ohe_join": 1},
            "failure": {"negative": 2, "conflict": 1, "unauthorized": 1,
                        "allowed": ["UNKNOWN", "CLARIFY", "REFUSE"], "factual_claim_count": 0},
            "wave1_qualification": "BLOCKED_ONE_OPERATION_AND_INCOMPLETE_NOVELTY_SPLITS",
            "multi_relation_reasoning": "UNKNOWN_NOT_PROVED_BY_SHARED_ENDPOINTS",
        }
        write_preflight_json(root, "run_identity.json", {
            "format": "INDEPENDENT_FACT_ADMISSION_PREFLIGHT_V1", "identity": root.name,
            "runner": RUNNER, "code_identity": code_identity(project),
            "input_sha256": hashes, "budget": BUDGET, "split": split,
            "regression_gate": gate, "acceptance_matrix": matrix,
            "maximum_real_gap": "INDEPENDENT_FACT_ADMISSION_TO_CURRENT_GRAPH_DELIVERY",
            "scope": "ADMISSION_AND_LANGUAGE_TRANSFER_PREFLIGHT_NOT_WAVE1",
            "canonical_used": 0, "parent_model_used": 0, "production_authorized": 0,
            "release_started": 0, "free_dialogue_complete": 0,
        })
        syntax = compile_relation_training_course(root / SYNTAX, root / QUESTIONS, root / "syntax_pack")
        teacher = compile_independent_relation_fact_source(root / TEACHER, root / "teacher_pack")
        write_integer_relation_graph_course(root / "syntax.int.json", syntax.train_course)
        write_integer_relation_graph_course(root / "teacher.int.json", teacher.course)
        graph = materialize_relation_question_graph(
            syntax.question_frames, root / "relation_questions.sqlite3",
            source_identity=syntax.train_course.source_identity,
            license_identity=syntax.train_course.license_identity)
        carriers = audit_prepared_question_carriers(
            syntax.question_frames, read_rows(root / INPUTS), teacher_course=teacher.course)
        write_preflight_json(root, "prepared_carriers.json", carriers)
        artifacts = tree_hashes(root, ("syntax_pack", "teacher_pack"))
        artifacts.update({relative: file_sha256(root / relative) for relative in (
            "syntax.int.json", "teacher.int.json", "relation_questions.sqlite3", "prepared_carriers.json")})
        prepared = {"status": "PREPARED_CONTRACT_PASS_NOT_TRAINED",
                    "syntax_source_audit": syntax.source_audit, "question_audit": syntax.question_audit,
                    "teacher_source_audit": teacher.source_audit, "question_graph": graph,
                    "teacher_pack_root": teacher.pack.pack_root.relative_to(root).as_posix(),
                    "artifact_sha256": artifacts, "split": split,
                    "core_training_count": 0, "query_execution_count": 0,
                    "free_dialogue_complete": 0}
        write_preflight_json(root, "prepared.json", prepared)
        write_preflight_json(root, "checkpoint.json", {
            "stage": "PREPARED_CONTRACT_PASS_NOT_TRAINED", "next_action": "launch_once_native_passive_wait",
            "prepared_sha256": file_sha256(root / "prepared.json"),
            "restart_same_identity_authorized": 0, "free_dialogue_complete": 0})
        return prepared
    except Exception:
        write_preflight_json(root, "prepare_failure.json", {
            "error": traceback.format_exc(), "core_training_count": 0,
            "query_execution_count": 0, "restart_authorized": 0, "free_dialogue_complete": 0})
        raise


def _context(backend):
    context = make_train_context(backend)
    _semantic, occurrence, span = dialogue_semantic_protocols()
    install_language_graph_protocols(context, occurrence_protocol=occurrence, span_protocol=span)
    return context


def execute(project: Path, root: Path) -> int:
    """Learn only independently admitted teachers, then consume evaluator text once."""
    identity, prepared = _json(root / "run_identity.json"), _json(root / "prepared.json")
    if identity["code_identity"] != code_identity(project):
        raise ValueError("code changed after preflight freeze")
    verify_frozen_files(root, identity["input_sha256"])
    verify_frozen_files(root, prepared["artifact_sha256"])
    if _regression_gate(project, Path(identity["regression_gate"]["path"])) != identity["regression_gate"]:
        raise ValueError("regression receipt changed after freeze")
    checkpoint_before = _json(root / "checkpoint.json")
    if (checkpoint_before["stage"] != "PREPARED_CONTRACT_PASS_NOT_TRAINED"
            or checkpoint_before["prepared_sha256"] != file_sha256(root / "prepared.json")
            or (root / "core.sqlite3").exists()):
        raise FileExistsError("preflight is one-shot and cannot resume a started training identity")
    started = time.perf_counter_ns()
    start_resource = current_process_resource_endpoint().to_value()
    with (root / "execution_started.json").open("xb") as stream:
        stream.write(canonical_json_bytes({"resource_endpoint": start_resource,
                     "restart_authorized": 0, "free_dialogue_complete": 0}) + b"\n")
    reports, admissions = [], []
    phase = "syntax_core_training"

    def checkpoint(stage, **values):
        endpoint = current_process_resource_endpoint().to_value()
        size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        record = {"stage": stage, "completed_case_count": len(reports),
                  "completed_admission_count": len(admissions),
                  "elapsed_ns": time.perf_counter_ns() - started, "output_bytes": size,
                  "resource_endpoint": endpoint, "resource_started": start_resource,
                  "peak_scope": "PROCESS_CUMULATIVE_NOT_PHASE_PEAK", **values,
                  "restart_same_identity_authorized": 0, "free_dialogue_complete": 0}
        write_preflight_json(root, "checkpoint.json", record)
        if (type(endpoint["process_peak_working_set_bytes"]) is not int
                or endpoint["process_peak_working_set_bytes"] > BUDGET["max_process_peak_bytes"]
                or size > BUDGET["max_output_bytes"] or record["elapsed_ns"] > BUDGET["max_wall_ns"]
                or shutil.disk_usage(root).free < BUDGET["minimum_free_bytes"]):
            raise RuntimeError("frozen preflight resource budget exceeded or unknown")
        return record

    try:
        core = root / "core.sqlite3"
        backend = SQLiteBackend(str(core))
        try:
            runtime = build_integer_w06_learning_runtime(
                backend, _context(backend), (root / "syntax.int.json",), require_full_curriculum=False)
            if runtime.report().active_candidate_count != 5:
                raise ValueError("syntax Core denominator did not close")
            backend.commit()
        finally:
            backend.close()
        write_preflight_json(root, "phase_resources/syntax_core_training.json", checkpoint(phase))
        with TrainedRelationGraphRuntime(core) as owner:
            frames = restore_open_role_frames(owner.active_surface_frames(), owner.active_surface_facts())
            syntax_claims = [list(item.definition.proposition.stable_key()) for item in owner.active_propositions()]
        write_preflight_json(root, "syntax_core_checkpoint.json", {
            "core_sha256": file_sha256(core), "claim_keys": syntax_claims,
            "open_frame_keys": [list(item.stable_key()) for item in frames], "free_dialogue_complete": 0})
        phase = "independent_fact_admission"
        teacher = read_independent_relation_fact_source(root / TEACHER, root / prepared["teacher_pack_root"])
        grant = grant_independent_fact_inputs(teacher, authorized_fact_inputs(root))
        backend = SQLiteBackend(str(core))
        try:
            context = _context(backend)
            for number, row in enumerate(read_rows(root / TEACHER), 1):
                admitted = admit_independent_relation_fact(row["surface"], fact_input_source(number), frames, grant=grant,
                    max_parse_steps=BUDGET["max_parse_steps"], max_witness_integers=BUDGET["max_witness_integers"])
                report = {"seed_id": row["seed_id"], "ordinal": number, "status": admitted.status,
                          "reason": admitted.reason, "input_source": list(admitted.source.stable_key()),
                          "input_values": list(map(ord, row["surface"])),
                          "parse_keys": [list(item.stable_key()) for item in admitted.parses],
                          "proposition_keys": [], "witness": []}
                if admitted.status == ADMISSION_READY:
                    report.update({"witness": list(admitted.stable_key()),
                                   "proposition_keys": [list(item.proposition_key) for item in admitted.course.records]})
                    learn_independent_relation_fact_admission(backend, context, admitted)
                    backend.commit()
                admissions.append(report)
                write_preflight_json(root, f"admissions/{number:03d}/receipt.json", report)
                write_preflight_json(root, f"admissions/{number:03d}/resources.json", checkpoint(
                    phase, last_admission_seed=row["seed_id"], next_admission_ordinal=number + 1 if number < 9 else None))
        finally:
            backend.close()
        counts = {status: sum(item["status"] == status for item in admissions)
                  for status in (ADMISSION_READY, "UNKNOWN", "CLARIFY", "REFUSE")}
        if counts != {ADMISSION_READY: 5, "UNKNOWN": 3, "CLARIFY": 0, "REFUSE": 1}:
            raise ValueError("independent admission contract did not close; do not continue generation")
        cold = verify_cold_admission_history(core, admissions)
        write_preflight_json(root, "admission_receipt.json", {
            "counts": counts, "cold_core": cold, "evaluator_read_count": 0,
            "question_truth_read_count": 0, "free_dialogue_complete": 0})
        phase = "generation_training"
        generation = materialize_integer_relation_generation(
            core, branch_key=(92044, 20261013), family_key=(92045, 20261013),
            representative_limit=32, require_full_active_coverage=True)
        core_sha = file_sha256(core)
        write_preflight_json(root, "training_receipt.json", {
            "active_proposition_count": generation.active_proposition_count,
            "connector_count": generation.connector_count, "realization_count": generation.realization_count,
            "core_sha256": core_sha, "admission_receipt_sha256": file_sha256(root / "admission_receipt.json"),
            "generation_from_active_core_only": 1, "free_dialogue_complete": 0})
        write_preflight_json(root, "phase_resources/generation_training.json", checkpoint(phase))
        phase = "natural_input_batch"
        cases = read_rows(root / INPUTS)
        with capture_relation_phases(root / "session_init_phase", run_identity=root.name,
                attempt_key="session_init", case_ordinal=0, required_phases=("memory_init",)):
            with TrainedGenerationConnectorRuntime(core) as generator, TrainedGraphQueryBridge(
                    core, memory_database=root / "memory.sqlite3", session_id=202610130044,
                    surface_generator=generator,
                    relation_question_graph_database=root / "relation_questions.sqlite3") as bridge:
                for number, row in enumerate(cases, 1):
                    report = run_natural_relation_case(bridge, row["input"], root, f"cases/{number:03d}",
                        max_case_wall_seconds=BUDGET["max_case_wall_seconds"], case_ordinal=number,
                        **{key: BUDGET[key] for key in ("max_depth", "max_nodes", "max_edges", "max_reads")})
                    reports.append({"case_id": row["case_id"], "split": row["split"], **report})
                    checkpoint(phase, last_completed_case_id=row["case_id"],
                               next_case_ordinal=number + 1 if number < len(cases) else None)
        phase = "complete_pending_independent_audit"
        verify_frozen_files(root, identity["input_sha256"])
        verify_frozen_files(root, prepared["artifact_sha256"])
        write_preflight_json(root, "receipt.json", {
            "format": "INDEPENDENT_FACT_PREFLIGHT_RECEIPT_V1", "status": "ENDED_PENDING_INDEPENDENT_AUDIT",
            "admission_counts": counts, "cases": reports,
            "core_unchanged_during_query": int(file_sha256(core) == core_sha),
            "code_unchanged": int(identity["code_identity"] == code_identity(project)),
            "frozen_input_unchanged": 1, "query_teacher_source_read_count": 0,
            "production_authorized": 0, "free_dialogue_complete": 0})
        checkpoint(phase, receipt_sha256=file_sha256(root / "receipt.json"), next_action="independent_cold_audit_only")
        print(json.dumps({"status": "ENDED_PENDING_INDEPENDENT_AUDIT", "case_count": len(reports)}, sort_keys=True))
        return 0
    except Exception:
        write_preflight_json(root, "failure.json", {
            "phase": phase, "completed_admission_count": len(admissions), "completed_case_count": len(reports),
            "completed_case_ids": [item["case_id"] for item in reports], "error": traceback.format_exc(),
            "restart_same_identity_authorized": 0, "free_dialogue_complete": 0})
        traceback.print_exc()
        return 1


__all__ = ["prepare", "execute"]
