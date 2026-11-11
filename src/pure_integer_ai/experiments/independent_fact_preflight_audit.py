"""Cold audit of original Core/Memory/delivery evidence; never rerun a query."""
from __future__ import annotations

from pure_integer_ai.experiments.batch_trace_storage import load_case_query_trace

from pure_integer_ai.experiments.batch_case_storage import load_case_provenance

import json
from pathlib import Path
from types import SimpleNamespace

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.query_state import (
    TERMINATION_CLARIFY_CONFLICT, TERMINATION_CLARIFY_MISSING_BINDING,
    TERMINATION_DIALOGUE_UNKNOWN, TERMINATION_NO_FRONTIER,
)
from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record
from pure_integer_ai.cognition.understanding.query_relation_questions import QUESTION_COMPOSED_FRAME_VERSION
from pure_integer_ai.experiments.factual_delivery_event_audit import verify_factual_delivery_events
from pure_integer_ai.experiments.factual_delivery_proof_audit import verify_factual_delivery_proof
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.independent_fact_preflight_contract import (
    INPUTS, LABELS, TEACHER, authorized_fact_inputs, read_rows,
)
from pure_integer_ai.experiments.independent_fact_preflight_core_audit import verify_cold_admission_history
from pure_integer_ai.experiments.independent_fact_preflight_identity import verify_frozen_files
from pure_integer_ai.experiments.independent_relation_fact_admission import (
    ADMISSION_READY, ADMISSION_WITNESS, grant_independent_fact_inputs,
)
from pure_integer_ai.experiments.independent_relation_fact_source import read_independent_relation_fact_source
from pure_integer_ai.experiments.integer_relation_graph_course import read_integer_relation_graph_course
from pure_integer_ai.experiments.natural_relation_preflight import file_sha256, write_preflight_json
from pure_integer_ai.experiments.query_memory_ohe_audit import verify_query_memory_ohe
from pure_integer_ai.experiments.trained_dialogue_memory_graph import _build_interaction_memory_intake
from pure_integer_ai.experiments.trained_response_delivery import read_delivery_proof, read_factual_evidence_ref
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD
from pure_integer_ai.storage.backend import SQLiteBackend


SAFE_TERMINATIONS = {TERMINATION_CLARIFY_CONFLICT, TERMINATION_CLARIFY_MISSING_BINDING,
                     TERMINATION_DIALOGUE_UNKNOWN, TERMINATION_NO_FRONTIER}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_admission_sources(source, prepared, admissions, facts):
    teacher = read_independent_relation_fact_source(source / TEACHER, source / prepared["teacher_pack_root"])
    if (teacher.source_audit != prepared["teacher_source_audit"]
            or teacher.course != read_integer_relation_graph_course(source / "teacher.int.json")):
        raise ValueError("physical independent teacher differs from frozen source audit or course")
    grant = grant_independent_fact_inputs(teacher, authorized_fact_inputs(source))
    frames = _json(source / "syntax_core_checkpoint.json")["open_frame_keys"]
    for report in admissions:
        if report["status"] != ADMISSION_READY:
            continue
        fields = unpack_record(tuple(report["witness"]), ADMISSION_WITNESS)
        record = facts[report["seed_id"]]
        if (len(fields) != 5 or fields[0] != tuple(report["input_source"])
                or fields[1] != grant.stable_key()
                or unpack_record(fields[2], 92041) != tuple(tuple(key) for key in frames)
                or unpack_record(fields[3], 92037) != tuple(tuple(key) for key in report["parse_keys"])
                or unpack_record(fields[4], 92038) != (record.stable_key(),)
                or report["input_values"] != list(record.surface_values)
                or report["proposition_keys"] != [list(record.proposition_key)]):
            raise ValueError("admission witness does not close independent source, permission or complete record")
    return verify_cold_admission_history(source / "core.sqlite3", admissions)


def _audit_case(source, number, report, label, facts, intake, *, max_state_members):
    trace_path = source / f"cases/{number:03d}/trace.int.json"
    trace_value = load_case_query_trace(trace_path)
    trace = GraphQueryTraceEvidence(trace_value)
    counts = trace.counts()
    if file_sha256(trace_path) != report.get("trace_descriptor_sha256", report["trace_sha256"]) or counts != report["trace_counts"]:
        raise ValueError("actual original trace differs from call receipt")
    if (report["phase_receipt_sha256"] != file_sha256(source / f"cases/{number:03d}/phase_attempt/receipt.json")
            or report["phase_boundary_status"] != "PASS"):
        raise ValueError("real production caller phase evidence is incomplete")
    events = intake.event_log
    owner = SourceRef.from_stable_key(tuple(report["input_source"])).owner
    access = MemoryAccessContext(owner.tenant_id, owner.user_id, owner.session_id)
    memory = SimpleNamespace(owner=owner, intake=intake)
    declarations = lambda ref: tuple(
        row for row in events.query(access=access, object_ref=ref) if row.event.is_declaration)
    refs = (report["observation_ref"], *report["hypothesis_refs"], *report["evidence_refs"])
    if any(len(declarations(MemoryObjectRef.from_stable_key(tuple(ref)))) != 1 for ref in refs):
        raise ValueError("original intake O/H/E declaration is absent or duplicated")
    result = {"case_id": report["case_id"], "split": report["split"], "termination": report["termination"],
              "trace_counts": counts, "response_surface": report["response_surface"]}
    if label["policy"] == "DIRECT_ONLY":
        if (report["split"] != "held_out" or report["capture_present"] != 1
                or report["positive_chain_phase_coverage"] != 1
                or any(counts[key] <= 0 for key in ("core_hop_count", "memory_hop_count", "companion_hop_count"))):
            raise ValueError("held-out did not close a real Core/Memory/Companion/delivery chain")
        captured = load_case_provenance(source / f"cases/{number:03d}/case_provenance.json")
        claim = facts[label["expected_seed_id"]].proposition_key
        if (captured["expected_claim"] != list(claim) or captured["plan_claims"] != [list(claim)]
                or report["projection"]["question_frame_versions"] != [QUESTION_COMPOSED_FRAME_VERSION]
                or report["hypothesis_count"] <= 0 or report["evidence_count"] <= 0):
            raise ValueError("held-out direct target, unseen question or current O/H/E differs")
        proof_reader = lambda key: read_delivery_proof(memory, ObjectIdentity.from_stable_key(key))
        evidence_reader = lambda key: read_factual_evidence_ref(memory, ObjectIdentity.from_stable_key(key))
        result["proof"] = verify_factual_delivery_proof(captured, proof_reader=proof_reader, evidence_reader=evidence_reader)
        result["delivery_events"] = verify_factual_delivery_events(
            captured, proof_reader=proof_reader, evidence_reader=evidence_reader,
            event_reader=lambda event_hash: events.read(event_hash, access=access), declaration_reader=declarations)
        result["memory_ohe"] = verify_query_memory_ohe(
            captured, declaration_reader=declarations, current_manifest_reader=intake.require_current_manifest,
            source_hash_reader=lambda ref: events.scoped_identities.registry.find(IDENTITY_SOURCE_RECORD, ref.stable_key()),
            max_state_members=max_state_members)
        result["status"] = "PASS_ADMITTED_NEW_FACT_UNSEEN_QUESTION_CURRENT_GRAPH_DELIVERY"
    else:
        if (report["split"] == "held_out" or report["termination"] not in SAFE_TERMINATIONS
                or report["delivery_claim_count"] or report["projection"]["question_bindings"]
                or (trace_value.get("response_plan") or {}).get("claim_refs", [])):
            raise ValueError("rejected or unsupported input acquired factual eligibility")
        result["status"] = "PASS_NO_FACTUAL_ELIGIBILITY"
    return result


def audit(source: Path, target: Path, *, max_state_members: int = 256,
          implementation_identity: dict | None = None, validation_receipt: dict | None = None) -> dict:
    """Audit naturally ended artifacts; incomplete calls remain UNKNOWN, never rerun."""
    completion = _json(source / "launcher_completion.json")
    if completion.get("completion_signal") != "NATIVE_PROCESS_EXIT":
        raise ValueError("original process completion is not independently observed")
    if type(max_state_members) is not int or not 0 < max_state_members <= 4096:
        raise ValueError("read-only state audit capacity must be a strict integer from 1 to 4096")
    target.mkdir(exist_ok=False)
    write_preflight_json(target, "audit_started.json", {
        "source_identity": source.name, "source_write_count": 0, "query_reexecution_count": 0,
        "max_state_members": max_state_members, "implementation_identity": implementation_identity,
        "validation_receipt": validation_receipt,
        "qualification": "AUDIT_IMPLEMENTATION_CAPACITY_NOT_ORIGINAL_TRAINING_BUDGET_CHANGE",
        "free_dialogue_complete": 0})
    hashes = {path.relative_to(source).as_posix(): file_sha256(path)
              for path in sorted(source.rglob("*")) if path.is_file()}
    try:
        identity, prepared = _json(source / "run_identity.json"), _json(source / "prepared.json")
        verify_frozen_files(source, identity["input_sha256"])
        verify_frozen_files(source, prepared["artifact_sha256"])
        checkpoint = _json(source / "checkpoint.json")
        inputs, labels = read_rows(source / INPUTS), read_rows(source / LABELS)
        label_by_id = {row["case_id"]: row for row in labels}
        teacher_course = read_integer_relation_graph_course(source / "teacher.int.json")
        facts = {"".join(map(chr, row.teacher_seed_values)): row for row in teacher_course.records}
        admissions = [_json(path) for path in sorted((source / "admissions").glob("*/receipt.json"))]
        cold_core = _audit_admission_sources(source, prepared, admissions, facts)
        calls = [_json(path) for path in sorted((source / "cases").glob("*/call_receipt.json"))]
        rows = []
        if (source / "memory.sqlite3-journal").exists() or (source / "memory.sqlite3-wal").exists():
            raise ValueError("interrupted Memory needs an independently hash-verified forensic recovery before audit")
        backend = SQLiteBackend(str(source / "memory.sqlite3"), read_only=True) if calls else None
        try:
            intake = _build_interaction_memory_intake(backend) if backend else None
            for number, expected in enumerate(inputs, 1):
                if number > len(calls):
                    rows.append({"case_id": expected["case_id"], "split": expected["split"],
                                 "status": "UNKNOWN_NO_COMPLETED_ORIGINAL_CALL"})
                    continue
                report = {**expected, **calls[number - 1]}
                if report["input_values"] != list(map(ord, expected["input"])):
                    raise ValueError("completed call is not the frozen ordinary input")
                try:
                    rows.append(_audit_case(source, number, report, label_by_id[expected["case_id"]], facts, intake,
                                            max_state_members=max_state_members))
                except (ValueError, RuntimeError) as error:
                    rows.append({"case_id": expected["case_id"], "split": expected["split"],
                                 "status": "FAIL_ORIGINAL_CALL_EVIDENCE", "reason": str(error)})
        finally:
            if backend:
                backend.close()
        receipt = _json(source / "receipt.json") if (source / "receipt.json").exists() else None
        complete = (completion["exit_code"] == 0 and checkpoint["stage"] == "complete_pending_independent_audit"
                    and receipt is not None and len(calls) == len(inputs))
        if complete and receipt["cases"] != [{"case_id": row["case_id"], "split": row["split"], **call}
                                             for row, call in zip(inputs, calls, strict=True)]:
            raise ValueError("batch receipt differs from the original completed calls")
        budget = identity["budget"]
        peak = completion.get("process_peak_working_set_bytes")
        elapsed = completion.get("elapsed_milliseconds")
        cpu_ticks = completion.get("process_cpu_ticks")
        resources = {"process_peak_working_set_bytes": peak,
                     "cpu_ns": None if cpu_ticks is None else cpu_ticks * 100,
                     "elapsed_ns": None if elapsed is None else elapsed * 10**6,
                     "output_bytes": sum(path.stat().st_size for path in source.rglob("*") if path.is_file()),
                     "peak_scope": "PROCESS_CUMULATIVE_NOT_CASE_PEAK",
                     "endpoint_scope": completion.get("resource_scope", "UNKNOWN")}
        resource_pass = (type(peak) is int and peak <= budget["max_process_peak_bytes"]
                         and type(cpu_ticks) is int
                         and type(elapsed) is int and elapsed * 10**6 <= budget["max_wall_ns"]
                         and resources["output_bytes"] <= budget["max_output_bytes"])
        passed = {split: sum(row["split"] == split and row["status"].startswith("PASS") for row in rows)
                  for split in identity["split"]["evaluation_case_counts"]}
        all_passed = (complete and len(admissions) == 9 and len(cold_core["claims"]) == 5
                      and sum(passed.values()) == len(inputs) and resource_pass
                      and receipt["core_unchanged_during_query"] == receipt["code_unchanged"] == 1)
        report = {"format": "INDEPENDENT_FACT_PREFLIGHT_AUDIT_V1",
                  "status": "PASS_NARROW_INDEPENDENT_FACT_LANGUAGE_TRANSFER" if all_passed else "FAIL_PREFLIGHT",
                  "source_identity": source.name, "source_sha256": hashes, "cold_core": cold_core,
                  "cases": rows, "passed_counts": passed,
                  "audit_contract": {"max_state_members": max_state_members,
                                     "implementation_identity": implementation_identity,
                                     "validation_receipt": validation_receipt,
                                     "post_execution_audit_capacity_change": int(max_state_members != 256)},
                  "denominators": identity["split"]["evaluation_case_counts"],
                  "resources": resources, "resource_budget_compliant": int(resource_pass),
                  "split_scope": identity["split"], "multi_relation_reasoning": "UNKNOWN",
                  "source_write_count": 0, "query_reexecution_count": 0,
                  "wave1_qualification": "BLOCKED_ONE_OPERATION_AND_INCOMPLETE_NOVELTY_SPLITS",
                  "production_authorized": 0, "free_dialogue_complete": 0}
        verify_frozen_files(source, hashes)
        write_preflight_json(target, "audit.json", report)
        return report
    except Exception as error:
        write_preflight_json(target, "audit_failure.json", {
            "error_type": type(error).__name__, "reason": str(error),
            "source_write_count": 0, "query_reexecution_count": 0, "free_dialogue_complete": 0})
        raise


__all__ = ["audit"]
