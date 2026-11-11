"""Cold evidence audit of one unresolved composition; never execute a query."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

from pure_integer_ai.experiments.batch_trace_storage import load_case_query_trace

from pure_integer_ai.cognition.shared.query_state import EvidenceEntry
from pure_integer_ai.cognition.shared.relation_closure import ActiveRelationClosureConsumer
from pure_integer_ai.cognition.shared.relation_composition import PendingRelationComposition
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.understanding.query_input_structure import TrainedInputStructureProjector
from pure_integer_ai.cognition.understanding.query_relation_composition import match_pending_relation_questions
from pure_integer_ai.cognition.understanding.query_structure_adapter import RelationSurfaceStructureAdapter
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION, MEMORY_CATEGORY_OPEN_RELATION, frame_key,
)
from pure_integer_ai.experiments.connected_relation_question_course import compile_train_relation_question
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.independent_relation_fact_source import read_independent_relation_fact_source
from pure_integer_ai.experiments.integer_relation_connected_course import _FACT_FIELDS, _read_lines, _seed
from pure_integer_ai.experiments.natural_relation_preflight import file_sha256, write_preflight_json
from pure_integer_ai.experiments.ph2_dataset_contract import parse_canonical_json_bytes
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack, verify_query_memory_ohe_without_plan
from pure_integer_ai.experiments.relation_composition_course import compile_train_relation_composition
from pure_integer_ai.experiments.trained_dialogue_memory_graph import _build_interaction_memory_intake
from pure_integer_ai.experiments.trained_core_relation_composition import RelationCompositionBudget
from pure_integer_ai.experiments.trained_generation_connector_runtime import TrainedGenerationConnectorRuntime
from pure_integer_ai.experiments.trained_input_concept_links import restore_input_concept_links
from pure_integer_ai.experiments.trained_open_role_frames import restore_open_role_frames
from pure_integer_ai.experiments.trained_query_evidence import core_query_evidence, current_query_evidence
from pure_integer_ai.experiments.trained_relation_composition_graph import read_relation_composition_graph
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.trained_relation_question_graph import (
    MODE_RELATION_PAIR_QUERY, read_relation_question_graph,
)
from pure_integer_ai.experiments.trained_response_occurrence import (
    build_response_occurrence_index, read_input_roles,
)
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD
from pure_integer_ai.storage.backend import SQLiteBackend


def _json(path: Path):
    return json.loads(path.read_bytes())


def _decode_input(value, trace):
    if isinstance(value, dict):
        if set(value) == {"integer_key_ref"}:
            return list(trace.resolve_ref(value["integer_key_ref"]))
        return {key: _decode_input(item, trace) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_input(item, trace) for item in value]
    return value


def verify_pending_composition_trace(trace_value, report, pending, prepared):
    """Compare saved input, consumed full premises and the unverified proof."""
    trace = GraphQueryTraceEvidence(trace_value)
    state = trace.trace["state_key"]
    if (trace.trace["termination"] != 3 or report["termination"] != 3
            or state["scalars"][7] != 0 or state["scalars"][15] != 0
            or state["scalars"][16] != 3
            or trace.trace.get("response_plan") or trace.trace.get("generation")
            or trace.trace.get("response_surface") or report["delivery_claim_count"]
            or report["response_plan_present"] or report["generation_present"]):
        raise ValueError("pending composition acquired factual closure or output")
    if (_decode_input(trace.trace["input_structure"], trace) != prepared.integer_trace()
            or tuple(report["input_values"]) != prepared.token_values
            or trace.operation_contracts() != ((prepared.operation_keys[0], MODE_RELATION_PAIR_QUERY),)):
        raise ValueError("saved pending question is not the complete cold-trained input")
    query_key = trace.resolve_ref(state["query_ref"])
    evidence = []
    for ref in state["evidence"]:
        key = trace.resolve_ref(ref)
        source, hypothesis, member, scope, payload, supersedes = _unpack(key, 3, 6)
        evidence.append(EvidenceEntry(source, hypothesis, key[0], key[1], key[2],
                                      member, scope, payload, supersedes))
    if len(set(evidence)) != len(evidence):
        raise ValueError("saved pending QueryState duplicates Evidence")
    current = current_query_evidence(tuple(evidence))
    for premise, expected in zip(pending.candidate.premises,
                                 pending.premise_evidence_keys, strict=True):
        actual = tuple(sorted(entry.stable_key() for entry in current
            if entry.space == 1 and entry.hypothesis_key == premise.proposition.stable_key()
            and (entry.evidence_key[:1] == (1,) or entry.polarity != 3)))
        if actual != expected:
            raise ValueError("saved pending query omitted or changed exact Core premise Evidence")
    proof = pending.stable_key()
    expected_proof = EvidenceEntry(pending.witness.source.stable_key(), proof,
        space=1, polarity=3, trust=1, evidence_key=proof, scope_key=query_key, payload_key=proof)
    if tuple(entry for entry in evidence if entry.payload_key == proof) != (expected_proof,):
        raise ValueError("saved pending proof is missing or has promoted support")
    if trace.counts() != report["trace_counts"] or any(
            trace.counts()[field] <= 0 for field in (
                "core_hop_count", "memory_hop_count", "companion_hop_count")):
        raise ValueError("pending query did not consume all three production spaces")
    return trace, tuple(evidence)


def _verify_cold_candidate(core, pending):
    witness, candidate = pending.witness, pending.candidate
    consumer = ActiveRelationClosureConsumer(core.owner.semantic_graph, core.owner.candidate_graph,
        core.owner.relation_protocol, witness.schemas, engine=core.owner.learning.engine)
    sources = tuple(core.generation_input(item.proposition) for item in candidate.premises)
    for premise, source, expected in zip(candidate.premises, sources,
                                       pending.premise_evidence_keys, strict=True):
        fact = consumer.require_proposition(premise.proposition).as_algebra_fact()
        if (fact.snapshot != source.snapshot or fact.definition != source.proposition.definition
                or source.hypothesis != premise.hypothesis
                or source.snapshot.support_evidence_ids != premise.support_evidence_ids
                or not set(fact.candidate_definition.forming_sources)
                       <= set(witness.authorized_forming_sources)):
            raise ValueError("cold H-05/R-00 premise or forming authority differs")
        projected = current_query_evidence(core_query_evidence(
            premise.proposition, core.evidence_history(premise.proposition),
            core.context.scoped_identity_store.registry))
        if tuple(sorted(entry.stable_key() for entry in projected)) != expected:
            raise ValueError("pending proof does not retain full current cold Core Evidence")
    first, second = (source.proposition.definition for source in sources)

    def filler(definition, role):
        values = tuple(item.filler for item in definition.bindings if item.role == role)
        if len(values) != 1:
            raise ValueError("cold composition premise is not single-valued")
        return values[0]

    rule = witness.rule
    if (first.predicate != rule.first_relation or second.predicate != rule.second_relation
            or filler(first, rule.first_join_role) != filler(second, rule.second_join_role)
            or {item.role: item.filler for item in candidate.bindings} != {
                rule.result_input_role: filler(first, rule.first_input_role),
                rule.result_output_role: filler(second, rule.second_output_role)}
            or candidate.rule != rule.rule or any(item.ordinal != 0 for item in candidate.bindings)):
        raise ValueError("pending result does not follow the explicitly licensed two-role join")
    schemas = {schema.relation: schema for schema in witness.schemas}
    if candidate.schemas != tuple(schemas[item].schema for item in (
            rule.first_relation, rule.second_relation, rule.result_relation)):
        raise ValueError("pending proof schema sequence differs")
    schemas[rule.result_relation].validate_bindings(candidate.bindings)
    if any(item.definition.predicate == candidate.relation
           and item.definition.canonical_bindings() == candidate.bindings
           for item in core.active_propositions()):
        raise ValueError("pending conclusion is already an active teacher answer")


def audit_saved_pending_composition(source: Path, target: Path, *, max_state_members=256):
    """Restore completed immutable sources and receipts; writes only a separate audit."""
    source, target = source.resolve(strict=True), target.resolve()
    if (source.drive.upper() != "K:" or target.drive.upper() != "K:"
            or target.parent == Path("K:/") or target.is_relative_to(source)
            or source.is_relative_to(target)):
        raise ValueError("pending cold audit needs a separate named K output")
    target.mkdir(exist_ok=False)
    hashes = {path.relative_to(source).as_posix(): file_sha256(path)
              for path in sorted(source.rglob("*")) if path.is_file()}
    if any((source / ("memory.sqlite3" + suffix)).exists() for suffix in ("-wal", "-journal")):
        raise ValueError("pending cold audit requires completed Memory")
    checkpoint = _json(source / "checkpoint.json")
    if (checkpoint["stage"] != "CORE_TWO_RELATION_PENDING_CANDIDATE_UNKNOWN_NOT_WAVE1"
            or checkpoint["query_count"] != 1 or checkpoint["free_dialogue_complete"] != 0
            or checkpoint["core_sha256"] != file_sha256(source / "core.sqlite3")
            or checkpoint["rules_sha256"] != file_sha256(source / "rules.sqlite3")
            or checkpoint["question_sha256"] != file_sha256(source / "questions.sqlite3")
            or checkpoint["composition_budget"] != asdict(RelationCompositionBudget())):
        raise ValueError("pending checkpoint is not the frozen completed development call")
    validation = _json(source.parent / "receipt.json")
    if validation["exit_code"] or validation["timed_out"] or validation["code_unchanged"] != 1:
        raise ValueError("source validation failed or changed code during execution")
    pending = PendingRelationComposition.from_stable_key(tuple(checkpoint["candidate_key"]))
    witnesses = read_relation_composition_graph(source / "rules.sqlite3")
    if witnesses != (pending.witness,):
        raise ValueError("pending proof differs from the physical integer rule graph")
    manifests = tuple((source / "teachers").rglob("manifest.json"))
    if len(manifests) != 1:
        raise ValueError("pending fixture has no unique audited teacher pack")
    facts = read_independent_relation_fact_source(source / "data/ph2/facts.jsonl", manifests[0].parent)
    if compile_train_relation_composition(source / "instruction.json", facts) != pending.witness:
        raise ValueError("cold physical instruction/source/license differs")
    frames, metadata = read_relation_question_graph(source / "questions.sqlite3")
    if (metadata["source_identity"] != facts.course.source_identity
            or metadata["license_identity"] != facts.course.license_identity):
        raise ValueError("pending syntax permission differs from rule authority")
    raw = (source / "question.train.json").read_bytes()
    syntax = parse_canonical_json_bytes(raw[:-1], require_object=True)
    if (not raw.endswith(b"\n") or syntax["split"] != "train"
            or syntax["owner"] != "teacher" or syntax["license_id"] != "CC0-1.0"):
        raise ValueError("cold question syntax is not licensed physical train data")
    rows = _read_lines(source / "data/ph2/facts.jsonl", _FACT_FIELDS)
    seeds = tuple(_seed(row, number) for number, row in enumerate(rows, 1)
                  if row["seed_id"] == syntax["seed_id"])
    if len(seeds) != 1 or frames != (compile_train_relation_question(
            seeds[0], syntax["question"], source_case_key=tuple(syntax["source_case_key"]),
            operation_key=tuple(syntax["operation_key"])),):
        raise ValueError("cold source question does not witness the saved frame")
    case = source / "cases/001"
    report = _json(case / "call_receipt.json")
    trace_value = load_case_query_trace(case / "trace.int.json")
    if (checkpoint["call_receipt"] != report
            or report.get("trace_descriptor_sha256", report["trace_sha256"]) != file_sha256(case / "trace.int.json")
            or report["phase_receipt_sha256"] != file_sha256(case / "phase_attempt/receipt.json")
            or report["phase_boundary_status"] != "PASS" or report["positive_chain_phase_coverage"] != 0):
        raise ValueError("pending caller or production phase identity differs")
    with TrainedRelationGraphRuntime(source / "core.sqlite3") as core:
        _verify_cold_candidate(core, pending)
        surfaces = core.active_surface_facts()
        projector = TrainedInputStructureProjector(
            RelationSurfaceStructureAdapter.from_facts(surfaces),
            concept_links=restore_input_concept_links(core),
            open_role_frames=restore_open_role_frames(core.active_surface_frames(), surfaces))
        projected = projector.project(tuple(report["input_values"]))
        bindings = match_pending_relation_questions(frames, projected, (pending,))
        if len(bindings) != 1 or projected.token_values == tuple(map(ord, syntax["question"])):
            raise ValueError("pending query does not use a new endpoint binding")
        prepared = replace(projected, operation_keys=(bindings[0].operation_key,),
                           pending_relation_composition_bindings=bindings)
        trace, _evidence = verify_pending_composition_trace(trace_value, report, pending, prepared)
        hops = {(item.proposition_key, item.predicate_key, item.source_hash)
                for item in trace.hops() if item.space == 1}
        for premise in pending.candidate.premises:
            fact = next(item for item in surfaces if item.proposition == premise.proposition)
            if (premise.proposition.stable_key(), fact.predicate.stable_key(), fact.source_hash) not in hops:
                raise ValueError("cold premise lacks an actual Core expansion hop")
        with TrainedGenerationConnectorRuntime(
                source / "core.sqlite3", shared_relation_runtime=core) as generator:
            if (generator.template_count != len(surfaces)
                    or generator.template_count != checkpoint["premise_generation_template_count"]):
                raise ValueError("pending call lacked actual premise generation contracts")
            for fact in surfaces:
                generator.generation_template_for(fact)
    backend = SQLiteBackend(str(source / "memory.sqlite3"), read_only=True)
    try:
        intake = _build_interaction_memory_intake(backend)
        owner = SourceRef.from_stable_key(tuple(report["input_source"])).owner
        access = MemoryAccessContext(owner.tenant_id, owner.user_id, owner.session_id)
        events = intake.event_log
        declarations = lambda ref: tuple(row for row in events.query(
            access=access, object_ref=ref) if row.event.is_declaration)
        for key in (report["observation_ref"], *report["hypothesis_refs"], *report["evidence_refs"]):
            if len(declarations(MemoryObjectRef.from_stable_key(tuple(key)))) != 1:
                raise ValueError("pending input has no unique durable O/H/E declaration")
        memory = SimpleNamespace(owner=owner, intake=intake,
            candidate_index=SimpleNamespace(access=access),
            occurrence_index=build_response_occurrence_index(events))
        observation = MemoryObjectRef.from_stable_key(tuple(report["observation_ref"]))
        open_candidates = {frame_key(MEMORY_CANDIDATE_VERSION, (MEMORY_CATEGORY_OPEN_RELATION,),
                                     item.stable_key()): item for item in prepared.open_relations}
        occurrences = []
        for key in report["hypothesis_refs"]:
            ref = MemoryObjectRef.from_stable_key(tuple(key))
            hypothesis = declarations(ref)[0].event.payload.hypothesis
            candidate = open_candidates.get(hypothesis.candidate_key)
            if candidate is not None:
                occurrences.extend(read_input_roles(memory, candidate, observation, ref, required=True))
        occurrences = tuple(sorted(occurrences, key=lambda item: item.stable_key()))
        if tuple(item.stable_key() for item in occurrences) != tuple(
                trace.resolve_ref(ref) for ref in trace.trace["input_role_occurrence_refs"]):
            raise ValueError("saved input role occurrences differ from the durable graph")
        ohe = verify_query_memory_ohe_without_plan(
            trace_value, report, declaration_reader=declarations,
            current_manifest_reader=intake.require_current_manifest,
            source_hash_reader=lambda ref: events.scoped_identities.registry.find(
                IDENTITY_SOURCE_RECORD, ref.stable_key()), max_state_members=max_state_members,
            input_role_occurrences=occurrences)
    finally:
        backend.close()
    after = {path.relative_to(source).as_posix(): file_sha256(path)
             for path in sorted(source.rglob("*")) if path.is_file()}
    if hashes != after:
        raise ValueError("cold pending audit changed source bytes")
    result = {"status": "PASS_DEVELOPMENT_PENDING_COMPOSITION_CORE_OHE_UNKNOWN",
        "scope": "DEVELOPMENT_NOT_WAVE1", "query_count": 1, "source_sha256": hashes,
        "trace_counts": trace.counts(), "memory_ohe": ohe, "source_write_count": 0,
        "query_reexecution_count": 0, "premise_count": 2, "pending_proof_count": 1,
        "response_plan_count": 0, "generation_count": 0, "delivery_claim_count": 0,
        "held_out_novelty": "UNKNOWN_NO_FOUR_WAY_COURSE",
        "verified_derived_answer": "UNKNOWN_NO_INDEPENDENT_DERIVATION_EVIDENCE",
        "resources": {key: report[key] for key in (
            "wall_ns", "cpu_ns", "resource_started", "resource_ended", "peak_scope")},
        "free_dialogue_complete": 0}
    write_preflight_json(target, "audit.json", result)
    return result
