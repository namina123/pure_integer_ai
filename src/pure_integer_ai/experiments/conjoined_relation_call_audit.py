"""Cold-read exact Core/Memory/Plan/delivery joins for saved compound calls."""
from __future__ import annotations

from pure_integer_ai.experiments.batch_trace_storage import load_case_query_trace

from pure_integer_ai.experiments.batch_case_storage import load_case_provenance

import json
from pathlib import Path
from types import SimpleNamespace

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.experiments.factual_delivery_event_audit import verify_factual_delivery_events
from pure_integer_ai.experiments.factual_delivery_proof_audit import verify_factual_delivery_proof
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.natural_relation_preflight import file_sha256, write_preflight_json
from pure_integer_ai.experiments.query_memory_ohe_audit import verify_query_memory_ohe
from pure_integer_ai.experiments.trained_dialogue_memory_graph import _build_interaction_memory_intake
from pure_integer_ai.experiments.trained_multi_claim_composer import (
    PART_CLAIM_PREDICATE, PART_LITERAL, read_multi_claim_composition_graph,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.trained_relation_question_graph import read_relation_question_graph
from pure_integer_ai.experiments.trained_relation_question_graph import MODE_RELATION_PAIR_QUERY
from pure_integer_ai.experiments.conjoined_relation_question_course import MODE_CONJOINED_RELATION_QUERY
from pure_integer_ai.experiments.trained_response_delivery import (
    read_delivery_proof, read_factual_evidence_ref,
)
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD
from pure_integer_ai.storage.backend import SQLiteBackend


def _json(path):
    return json.loads(path.read_text("utf-8"))


def audit_saved_conjoined_call(source: Path, target: Path, *,
                              max_state_members: int = 1024) -> dict[str, object]:
    """Audit a completed development call without rerunning or altering it.

    Split novelty, source independence and multi-relation inference stay UNKNOWN;
    this proves only the actual caller and durable evidence joins named below.
    """
    source, target = source.resolve(strict=True), target.resolve()
    if (source.drive.upper() != "K:" or target.drive.upper() != "K:"
            or target.parent == Path("K:/") or target.is_relative_to(source)
            or source.is_relative_to(target)):
        raise ValueError("cold audit output must be separate from the completed source")
    target.mkdir(exist_ok=False)
    hashes = {path.relative_to(source).as_posix(): file_sha256(path)
              for path in sorted(source.rglob("*")) if path.is_file()}
    if any((source / f"memory.sqlite3{suffix}").exists() for suffix in ("-journal", "-wal")):
        raise ValueError("cold audit requires a completed Memory without recovery sidecars")
    checkpoint = _json(source / "checkpoint.json")
    if (checkpoint.get("stage") != "TWO_FACT_NATURAL_TURN_DELIVERED_NOT_WAVE1"
            or checkpoint.get("query_count") != 3
            or checkpoint.get("core_sha256") != file_sha256(source / "core.sqlite3")):
        raise ValueError("completed compound fixture checkpoint differs")
    case = source / "cases/001"
    report, captured = _json(case / "call_receipt.json"), load_case_provenance(case / "case_provenance.json")
    trace = GraphQueryTraceEvidence(load_case_query_trace(case / "trace.int.json"))
    if (report.get("trace_descriptor_sha256", report["trace_sha256"]) != file_sha256(case / "trace.int.json")
            or report["phase_receipt_sha256"] != file_sha256(case / "phase_attempt/receipt.json")
            or report["trace_counts"] != trace.counts()
            or report["delivery_claim_count"] != 2
            or report["phase_boundary_status"] != "PASS"
            or report["positive_chain_phase_coverage"] != 1):
        raise ValueError("saved compound call or production phase identity differs")
    claims = tuple(tuple(key) for key in captured["plan_claims"])
    if len(claims) != 2 or set(claims) != {
            tuple(key) for key in trace.trace["response_plan"]["claim_refs"]}:
        raise ValueError("compound plan claim identity differs")
    frames, question_meta = read_relation_question_graph(source / "questions.sqlite3")
    variants, generation_meta = read_multi_claim_composition_graph(source / "composer.sqlite3")
    if any(question_meta[key] != generation_meta[key]
           for key in ("source_identity", "license_identity")):
        raise ValueError("input and generation graph provenance differs")
    input_bindings = trace.trace["input_structure"].get("relation_question_bindings")
    if (not isinstance(input_bindings, list) or len(input_bindings) != 2
            or len(report["projection"]["question_bindings"]) != 2):
        raise ValueError("saved compound input lacks two scoped question bindings")
    restored_frames = {item.stable_key() for item in frames}
    binding_keys = []
    for encoded, original in zip(input_bindings, report["projection"]["question_bindings"]):
        if not isinstance(encoded, dict) or set(encoded) != {"integer_key_ref"}:
            raise ValueError("scoped binding is not an integer graph reference")
        key = trace.resolve_ref(encoded["integer_key_ref"], "scoped question binding")
        if key != tuple(original) or len(key) < 2 or key[0] != 92024:
            raise ValueError("saved scoped question binding differs from original projection")
        frame_size = key[1]
        if key[2:2 + frame_size] not in restored_frames:
            raise ValueError("scoped binding has no cold-trained source frame")
        binding_keys.append(key)
    contracts = trace.operation_contracts()
    if ({mode for _key, mode in contracts}
            != {MODE_RELATION_PAIR_QUERY, MODE_CONJOINED_RELATION_QUERY}
            or len(contracts) != 2 or len(set(binding_keys)) != 2):
        raise ValueError("compound operation contracts or scoped bindings differ")
    with TrainedRelationGraphRuntime(source / "core.sqlite3") as core:
        facts = {fact.proposition.stable_key(): fact for fact in core.active_surface_facts()}
        if any(key not in facts or not core.evidence_history(ObjectIdentity.from_stable_key(key))
               for key in claims):
            raise ValueError("compound claims lack current active Core Evidence")
        core_predicates = {key: facts[key].predicate.stable_key() for key in claims}
    # The plan's claim set may be canonical. The trained conjunction operands
    # carry semantic clause order; recover every member from its saved join.
    ordered_claims = []
    for row in trace.trace["generation"]["scope_conjunction_bindings"]:
        for ref in row[2]:
            key = trace.resolve_ref(ref["integer_key_ref"])
            if key not in ordered_claims:
                ordered_claims.append(key)
    if set(ordered_claims) != set(claims):
        raise ValueError("generation scope join differs from its plan claims")
    generation_contracts = tuple(tuple(part[2] for part in sorted(
        (part for part in variant.parts if part[0] == PART_CLAIM_PREDICATE),
        key=lambda part: part[1])) for variant in variants)
    if not generation_contracts or any(len(contract) != 2 for contract in generation_contracts):
        raise ValueError("compound generation has no complete trained predicate contract")
    predicates = tuple(core_predicates[key] for key in ordered_claims)
    selected = tuple(variant for variant, contract in zip(variants, generation_contracts)
                     if contract == predicates)
    if len(selected) != 1:
        raise ValueError("generation predicate order has no unique trained contract")
    units = []
    for part in selected[0].parts:
        if part[0] == PART_CLAIM_PREDICATE:
            continue
        if part[0] == PART_LITERAL:
            units.extend(part[1])
        else:
            units.extend(map(ord, facts[ordered_claims[part[1] - 1]].bindings[part[2] - 1].surface))
    if units != captured["output_values"]:
        raise ValueError("compound output differs from trained grammar and current Core roles")
    backend = SQLiteBackend(str(source / "memory.sqlite3"), read_only=True)
    try:
        intake = _build_interaction_memory_intake(backend)
        owner = SourceRef.from_stable_key(tuple(report["input_source"])).owner
        access = MemoryAccessContext(owner.tenant_id, owner.user_id, owner.session_id)
        memory = SimpleNamespace(owner=owner, intake=intake)
        events = intake.event_log
        declarations = lambda ref: tuple(row for row in events.query(
            access=access, object_ref=ref) if row.event.is_declaration)
        for key in (report["observation_ref"], *report["hypothesis_refs"], *report["evidence_refs"]):
            if len(declarations(MemoryObjectRef.from_stable_key(tuple(key)))) != 1:
                raise ValueError("compound input has missing or duplicate O/H/E declaration")
        proof_reader = lambda key: read_delivery_proof(memory, ObjectIdentity.from_stable_key(key))
        evidence_reader = lambda key: read_factual_evidence_ref(memory, ObjectIdentity.from_stable_key(key))
        proof = verify_factual_delivery_proof(
            captured, proof_reader=proof_reader, evidence_reader=evidence_reader)
        delivery = verify_factual_delivery_events(
            captured, proof_reader=proof_reader, evidence_reader=evidence_reader,
            event_reader=lambda event_hash: events.read(event_hash, access=access),
            declaration_reader=declarations)
        ohe = verify_query_memory_ohe(
            captured, declaration_reader=declarations,
            current_manifest_reader=intake.require_current_manifest,
            source_hash_reader=lambda ref: events.scoped_identities.registry.find(
                IDENTITY_SOURCE_RECORD, ref.stable_key()), max_state_members=max_state_members)
    finally:
        backend.close()
    boundaries = []
    for path in sorted((source / "cases").glob("*/call_receipt.json")):
        if path.parent == case:
            continue
        row = _json(path)
        value = load_case_query_trace(path.parent / "trace.int.json")
        if (row.get("trace_descriptor_sha256", row["trace_sha256"]) != file_sha256(path.parent / "trace.int.json")
                or row["termination"] == 1 or row["delivery_claim_count"]
                or (value.get("response_plan") or {}).get("claim_refs")):
            raise ValueError("saved boundary input acquired factual output")
        boundaries.append({"case": path.parent.name, "termination": row["termination"],
                           "trace_sha256": row["trace_sha256"]})
    if {row["case"] for row in boundaries} != {"002", "003"}:
        raise ValueError("compound fixture failure boundaries are incomplete")
    after = {path.relative_to(source).as_posix(): file_sha256(path)
             for path in sorted(source.rglob("*")) if path.is_file()}
    if hashes != after:
        raise ValueError("cold audit source bytes changed")
    result = {
        "status": "PASS_DEVELOPMENT_CONJOINED_CORE_MEMORY_PLAN_DELIVERY",
        "scope": "DEVELOPMENT_NOT_WAVE1", "trace_counts": trace.counts(),
        "proof": proof, "delivery_events": delivery, "memory_ohe": ohe,
        "source_sha256": hashes, "query_count": 1 + len(boundaries), "boundary_cases": boundaries,
        "question_frame_count": len(frames), "scoped_question_binding_count": len(binding_keys),
        "guarded_generation_contract_count": len(generation_contracts),
        "source_write_count": 0, "query_reexecution_count": 0,
        "held_out_novelty": "UNKNOWN_TRAIN_FIXTURE_ONLY",
        "multi_relation_inference": "UNKNOWN_JOINT_LOOKUP_ONLY",
        "source_independence": "UNKNOWN_DEVELOPMENT_TEACHERS",
        "free_dialogue_complete": 0,
    }
    write_preflight_json(target, "audit.json", result)
    return result


__all__ = ["audit_saved_conjoined_call"]
