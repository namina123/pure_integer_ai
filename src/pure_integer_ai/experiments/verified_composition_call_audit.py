"""Cold audit of an original derived call; no query or generation reexecution."""
from __future__ import annotations

from pure_integer_ai.experiments.batch_trace_storage import load_case_query_trace

from pure_integer_ai.experiments.batch_case_storage import load_case_provenance

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.cognition.shared.relation_composition_verifier import (
    COMPOSITION_VERIFICATION_ROLE, VerifiedRelationComposition, admit_verified_composition,
)
from pure_integer_ai.cognition.shared.typed_relation import ActiveSupportedRelationFact
from pure_integer_ai.cognition.understanding.query_input_structure import TrainedInputStructureProjector
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION, MEMORY_CATEGORY_OPEN_RELATION, frame_key,
)
from pure_integer_ai.cognition.understanding.query_relation_composition import match_pending_relation_questions
from pure_integer_ai.cognition.understanding.query_structure_adapter import RelationSurfaceStructureAdapter
from pure_integer_ai.experiments.connected_relation_question_course import compile_train_relation_question
from pure_integer_ai.experiments.factual_delivery_event_audit import verify_factual_delivery_events
from pure_integer_ai.experiments.factual_delivery_proof_audit import verify_factual_delivery_proof
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.independent_relation_fact_source import read_independent_relation_fact_source
from pure_integer_ai.experiments.integer_relation_connected_course import _FACT_FIELDS, _read_lines, _seed
from pure_integer_ai.experiments.natural_relation_preflight import file_sha256, write_preflight_json
from pure_integer_ai.experiments.pending_composition_call_audit import _decode_input, _verify_cold_candidate
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack, verify_query_memory_ohe
from pure_integer_ai.experiments.relation_composition_course import compile_train_relation_composition
from pure_integer_ai.experiments.trained_dialogue_memory_graph import _build_interaction_memory_intake
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime, _slot_ordinal,
)
from pure_integer_ai.experiments.trained_input_concept_links import restore_input_concept_links
from pure_integer_ai.experiments.trained_open_role_frames import restore_open_role_frames
from pure_integer_ai.experiments.trained_query_evidence import core_query_evidence, current_query_evidence
from pure_integer_ai.experiments.trained_relation_composition_graph import read_relation_composition_graph
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.trained_relation_question_graph import MODE_RELATION_PAIR_QUERY, read_relation_question_graph
from pure_integer_ai.experiments.trained_response_delivery import read_delivery_proof, read_factual_evidence_ref
from pure_integer_ai.experiments.trained_response_occurrence import build_response_occurrence_index, read_input_roles
from pure_integer_ai.experiments.verified_composition_generation import composition_generation_input
from pure_integer_ai.cognition.shared.representation_rendering import representation_parts
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD
from pure_integer_ai.storage.backend import SQLiteBackend


def _json(path):
    return json.loads(path.read_bytes())


def verify_saved_composition_trace(value, report, prepared, verified, core):
    """Independently compare full saved premise/support/proof/plan members."""
    trace = GraphQueryTraceEvidence(value)
    state = value["state_key"]
    claim, proof = verified.claim.stable_key(), verified.stable_key()
    if (value["termination"] != 1 or report["termination"] != 1
            or state["scalars"][6:9] != [0, 1, 0] or state["scalars"][15:] != [1, 1]
            or trace.resolve_ref(state["scalars"][11]) != claim
            or _decode_input(value["input_structure"], trace) != prepared.integer_trace()
            or trace.operation_contracts() != ((prepared.operation_keys[0], MODE_RELATION_PAIR_QUERY),)
            or tuple(report["input_values"]) != prepared.token_values
            or verified.query_key != trace.resolve_ref(state["query_ref"])):
        raise ValueError("saved derived call is not the exact closed current question")
    evidence = []
    for ref in state["evidence"]:
        key = trace.resolve_ref(ref)
        parts = _unpack(key, 3, 6)
        evidence.append(EvidenceEntry(parts[0], parts[1], key[0], key[1], key[2], *parts[2:]))
    if len(set(evidence)) != len(evidence):
        raise ValueError("saved derived Evidence duplicates")
    current = current_query_evidence(tuple(evidence))
    for premise, expected in zip(verified.pending.candidate.premises,
                                 verified.pending.premise_evidence_keys, strict=True):
        actual = tuple(sorted(entry.stable_key() for entry in current
            if entry.space == 1 and entry.hypothesis_key == premise.proposition.stable_key()
            and (entry.evidence_key[:1] == (1,) or entry.polarity != 3)))
        if actual != expected:
            raise ValueError("saved derived call omitted or changed exact Core premise Evidence")
    registry = core.context.scoped_identity_store.registry
    expected = core_query_evidence(verified.claim, verified.history, registry)
    actual = tuple(sorted(entry.stable_key() for entry in current
                          if entry.space == 1 and entry.hypothesis_key == claim))
    if actual != tuple(sorted(entry.stable_key() for entry in expected)):
        raise ValueError("derived support differs from independently restored H-05 history")
    marker = BindingEntry(COMPOSITION_VERIFICATION_ROLE, proof, space=1,
                          scope_key=verified.query_key).stable_key()
    if marker not in {trace.resolve_ref(ref) for ref in state["bindings"]}:
        raise ValueError("derived call has no exact query-local verification binding")
    planned = value["response_plan"]
    if (planned["claim_refs"] != [list(claim)] or report["delivery_claim_count"] != 1
            or value["generation"]["kind"] != 5
            or value["generation"]["core_claims_written"] != 0
            or not {entry.stable_key() for entry in evidence} <= {
                tuple(key) for key in planned["evidence_refs"]}):
        raise ValueError("derived output plan omits proof or adds a different claim")
    counts = trace.counts()
    if counts != report["trace_counts"] or any(counts[name] <= 0 for name in
            ("core_hop_count", "memory_hop_count", "companion_hop_count")):
        raise ValueError("derived call did not actually consume all three spaces")
    hops = {(item.proposition_key, item.predicate_key) for item in trace.hops() if item.space == 1}
    for premise in verified.pending.candidate.premises:
        atomic = core.generation_input(premise.proposition).proposition.definition
        if (premise.proposition.stable_key(), atomic.predicate.stable_key()) not in hops:
            raise ValueError("derived support premise has no actual Core consumption hop")
    return trace


def inspect_saved_verified_composition(source: Path):
    """Read completed original artifacts, including a failed regression postcondition."""
    source = source.resolve(strict=True)
    if source.drive.upper() != "K:" or source.parent == Path("K:/"):
        raise ValueError("saved derived audit needs an explicit K source")
    hashes = {path.relative_to(source).as_posix(): file_sha256(path)
              for path in sorted(source.rglob("*")) if path.is_file()}
    if any((source / ("memory.sqlite3" + suffix)).exists() for suffix in ("-wal", "-journal")):
        raise ValueError("derived source Memory is not cold")
    validation = _json(source.parent / "receipt.json")
    if validation["timed_out"] or validation["code_unchanged"] != 1:
        raise ValueError("original derived source timed out or changed code")
    case = source / "cases/001"
    report, value = _json(case / "call_receipt.json"), load_case_query_trace(case / "trace.int.json")
    if (report.get("trace_descriptor_sha256", report["trace_sha256"]) != file_sha256(case / "trace.int.json")
            or report["phase_receipt_sha256"] != file_sha256(case / "phase_attempt/receipt.json")
            or report["phase_boundary_status"] != "PASS"
            or report["positive_chain_phase_coverage"] != 1):
        raise ValueError("derived caller/production phases are incomplete or drifted")
    trace = GraphQueryTraceEvidence(value)
    verified = VerifiedRelationComposition.from_stable_key(trace.resolve_ref(
        value["generation"]["verified_composition"]["integer_key_ref"]))
    pending = verified.pending
    witnesses = read_relation_composition_graph(source / "rules.sqlite3")
    if witnesses != (pending.witness,):
        raise ValueError("derived proof has no exact cold integer rule authority")
    manifests = tuple((source / "teachers").rglob("manifest.json"))
    if len(manifests) != 1:
        raise ValueError("derived source has no unique licensed train teacher pack")
    facts = read_independent_relation_fact_source(source / "data/ph2/facts.jsonl", manifests[0].parent)
    if compile_train_relation_composition(source / "instruction.json", facts) != pending.witness:
        raise ValueError("derived physical rule/source/license changed")
    frames, metadata = read_relation_question_graph(source / "questions.sqlite3")
    if (metadata["source_identity"] != facts.course.source_identity
            or metadata["license_identity"] != facts.course.license_identity):
        raise ValueError("derived question permission differs from physical facts")
    syntax = _json(source / "question.train.json")
    if (syntax["owner"], syntax["split"], syntax["license_id"]) != ("teacher", "train", "CC0-1.0"):
        raise ValueError("derived syntax is not licensed train data")
    rows = _read_lines(source / "data/ph2/facts.jsonl", _FACT_FIELDS)
    seeds = tuple(_seed(row, number) for number, row in enumerate(rows, 1)
                  if row["seed_id"] == syntax["seed_id"])
    if len(seeds) != 1 or frames != (compile_train_relation_question(seeds[0], syntax["question"],
            source_case_key=tuple(syntax["source_case_key"]), operation_key=tuple(syntax["operation_key"])),):
        raise ValueError("derived physical question differs from the trained frame")
    with TrainedRelationGraphRuntime(source / "core.sqlite3") as core:
        _verify_cold_candidate(core, pending)
        current_facts = tuple(ActiveSupportedRelationFact(item.proposition.definition, item.snapshot,
            core.owner.learning.engine.definition(item.hypothesis))
            for premise in pending.candidate.premises for item in (core.generation_input(premise.proposition),))
        independent = admit_verified_composition(pending, verified.query_key, current_facts,
            pending.premise_evidence_keys, witnesses, declaration_version=verified.declaration_version)
        if independent != verified:
            raise ValueError("saved derived result differs from independent cold verification")
        surfaces = core.active_surface_facts()
        projector = TrainedInputStructureProjector(RelationSurfaceStructureAdapter.from_facts(surfaces),
            concept_links=restore_input_concept_links(core),
            open_role_frames=restore_open_role_frames(core.active_surface_frames(), surfaces))
        projected = projector.project(tuple(report["input_values"]))
        bindings = match_pending_relation_questions(frames, projected, (pending,))
        if len(bindings) != 1:
            raise ValueError("saved derived ordinary input has no unique learned question binding")
        prepared = replace(projected, operation_keys=(bindings[0].operation_key,),
                           pending_relation_composition_bindings=bindings)
        trace = verify_saved_composition_trace(value, report, prepared, verified, core)
        with TrainedGenerationConnectorRuntime(source / "core.sqlite3") as generator:
            bound, template, _, result_fact, _ = composition_generation_input(generator, verified, core)
            recovered = generator.connector_for(bound, branch=template.language_branch,
                                                 connector_identity=template.connector)
            values = recovered.connector.registry.values(template, bound)
            if any(item.source == recovered.connector.registry.value_protocol.proposition_source
                   for item in template.bindings):
                raise ValueError("derived generator contains a whole-proposition replay slot")
            routes = tuple(recovered.surface_for(item.filler) for item in values)
            if any(route.result.selected is None or len(route.result.options) != 1 for route in routes):
                raise ValueError("derived generator has ambiguous/unproved graph realization")
            representations = {item.slot: route.result.selected.value
                               for item, route in zip(values, routes, strict=True)}
            ordered = tuple(representations[item.slot] for item in sorted(template.slots,
                             key=lambda item: _slot_ordinal(item.slot)))
            actual_representations = tuple(trace.resolve_ref(item["integer_key_ref"])
                for item in value["generation"]["representations"])
            if tuple(item.stable_key() for item in ordered) != actual_representations:
                raise ValueError("derived emitted Representations differ from graph role/atom readers")
            output = tuple(unit for representation in ordered for unit in representation_parts(representation)[1])
            if (output != tuple(map(ord, value["response_surface"]))
                    or value["response_surface"] != report["response_surface"]):
                raise ValueError("derived output differs from learned graph representation sequence")
            saved_generation = trace.resolve_ref(value["generation"]["trace"]["integer_key_ref"])
            generation_parts = _unpack(saved_generation, 2, saved_generation[1])
            if saved_generation[0] != 2 or generation_parts[2:] != tuple(route.stable_key() for route in routes):
                raise ValueError("saved graph generation has different actual realization routes")
            slots = value["response_plan"]["slot_sequence"]
            if tuple((tuple(item["role"]), tuple(item["filler"])) for item in slots) != tuple(
                    (item.role.stable_key(), item.filler.stable_key()) for item in result_fact.bindings):
                raise ValueError("derived ResponsePlan result roles drifted")
    captured = load_case_provenance(case / "case_provenance.json")
    backend = SQLiteBackend(str(source / "memory.sqlite3"), read_only=True)
    try:
        intake = _build_interaction_memory_intake(backend)
        owner = SourceRef.from_stable_key(tuple(report["input_source"])).owner
        access = MemoryAccessContext(owner.tenant_id, owner.user_id, owner.session_id)
        events = intake.event_log
        declarations = lambda ref: tuple(row for row in events.query(access=access, object_ref=ref)
                                          if row.event.is_declaration)
        memory = SimpleNamespace(owner=owner, intake=intake, candidate_index=SimpleNamespace(access=access),
                                 occurrence_index=build_response_occurrence_index(events))
        proof_reader = lambda key: read_delivery_proof(memory, ObjectIdentity.from_stable_key(key))
        evidence_reader = lambda key: read_factual_evidence_ref(memory, ObjectIdentity.from_stable_key(key))
        proof_audit = verify_factual_delivery_proof(captured, proof_reader=proof_reader,
                                                  evidence_reader=evidence_reader)
        delivery = verify_factual_delivery_events(captured, proof_reader=proof_reader,
            evidence_reader=evidence_reader, event_reader=lambda key: events.read(key, access=access),
            declaration_reader=declarations)
        candidates = {frame_key(MEMORY_CANDIDATE_VERSION, (MEMORY_CATEGORY_OPEN_RELATION,), item.stable_key()): item
                      for item in prepared.open_relations}
        observation = MemoryObjectRef.from_stable_key(tuple(report["observation_ref"]))
        occurrences = []
        for key in report["hypothesis_refs"]:
            ref = MemoryObjectRef.from_stable_key(tuple(key))
            rows = declarations(ref)
            if len(rows) != 1:
                raise ValueError("derived input lacks an original unique H declaration")
            candidate = candidates.get(rows[0].event.payload.hypothesis.candidate_key)
            if candidate is not None:
                occurrences.extend(read_input_roles(memory, candidate, observation, ref, required=True))
        occurrences = tuple(sorted(occurrences, key=lambda item: item.stable_key()))
        if tuple(item.stable_key() for item in occurrences) != tuple(trace.resolve_ref(ref)
                for ref in value["input_role_occurrence_refs"]):
            raise ValueError("derived input Role occurrences differ from physical graph")
        ohe = verify_query_memory_ohe(captured, declaration_reader=declarations,
            current_manifest_reader=intake.require_current_manifest,
            source_hash_reader=lambda ref: events.scoped_identities.registry.find(
                IDENTITY_SOURCE_RECORD, ref.stable_key()), input_role_occurrences=occurrences)
    finally:
        backend.close()
    after = {path.relative_to(source).as_posix(): file_sha256(path)
             for path in sorted(source.rglob("*")) if path.is_file()}
    if hashes != after:
        raise ValueError("derived cold audit changed source bytes")
    return {"status": "PASS_DEVELOPMENT_QUERY_LOCAL_DERIVATION_CORE_OHE_PLAN_DELIVERY",
        "scope": "DEVELOPMENT_NOT_WAVE1", "original_regression_exit_code": validation["exit_code"],
        "original_regression_status_preserved": 1, "source_sha256": hashes,
        "trace_counts": trace.counts(), "derived_support_count": 1, "core_claims_written": 0,
        "request_local_h05_admission": 1, "premise_count": 2, "memory_ohe": ohe,
        "delivery_proof": proof_audit, "delivery_events": delivery, "query_count": 1,
        "source_write_count": 0, "query_reexecution_count": 0, "generation_reexecution_count": 0,
        "held_out_novelty": "UNKNOWN_NO_FOUR_WAY_COURSE", "semantic_behavior_count": 1,
        "resources": {key: report[key] for key in
            ("wall_ns", "cpu_ns", "resource_started", "resource_ended", "peak_scope")},
        "free_dialogue_complete": 0}


def audit_saved_verified_composition(source, target):
    source, target = Path(source).resolve(strict=True), Path(target).resolve()
    if (target.drive.upper() != "K:" or target.parent == Path("K:/")
            or target.is_relative_to(source) or source.is_relative_to(target)):
        raise ValueError("derived cold audit requires a separate named K target")
    target.mkdir(exist_ok=False)
    result = inspect_saved_verified_composition(source)
    write_preflight_json(target, "audit.json", result)
    return result
