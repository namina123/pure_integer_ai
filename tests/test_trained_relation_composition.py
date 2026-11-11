"""Actual cold Core premises, integer rule records and fail-closed candidate use."""
from __future__ import annotations

from pure_integer_ai.experiments.batch_case_storage import (
    COMPACT_PROVENANCE_FORMAT, load_case_provenance,
)

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sqlite3

import pytest

from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord
from pure_integer_ai.cognition.shared.identity import SourceRef, minimal_instruction_identity
from pure_integer_ai.cognition.shared.query_state import QueryState, TERMINATION_ANSWER_CLOSED
from pure_integer_ai.cognition.shared.typed_relation import ActiveSupportedRelationFact
from pure_integer_ai.cognition.shared.relation_composition_verifier import (
    COMPOSITION_VERIFIED_V2, VerifiedRelationComposition, admit_verified_composition,
)
from pure_integer_ai.cognition.understanding.query_input_structure import PROJECTION_FILLER
from pure_integer_ai.cognition.understanding.query_relation_composition import match_pending_relation_questions
from pure_integer_ai.cognition.shared.relation_composition import (
    COMPOSITION_PENDING_PROOF, PendingRelationComposition, RelationCompositionWitness,
)
from pure_integer_ai.experiments.conversation_typed_relation_bridge import build_integer_w06_learning_runtime
from pure_integer_ai.experiments.connected_relation_question_course import compile_train_relation_question
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.independent_relation_fact_source import compile_independent_relation_fact_source
from pure_integer_ai.experiments.integer_relation_connected_course import _seed
from pure_integer_ai.experiments.integer_relation_graph_course import write_integer_relation_graph_course
from pure_integer_ai.experiments.integer_relation_generation_bridge import materialize_integer_relation_generation
from pure_integer_ai.experiments.language_protocol_runtime import install_language_graph_protocols
from pure_integer_ai.experiments.natural_relation_preflight import file_sha256, run_natural_relation_case, write_preflight_json
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.pending_composition_call_audit import verify_pending_composition_trace
from pure_integer_ai.experiments.relation_composition_course import compile_train_relation_composition
from pure_integer_ai.experiments.run_conversation_training import dialogue_semantic_protocols
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_core_relation_composition import (
    RelationCompositionBudget, pending_core_compositions,
    pending_composition_query_evidence, validate_relation_composition_provenance,
)
from pure_integer_ai.experiments.trained_generation_connector_runtime import TrainedGenerationConnectorRuntime
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge
from pure_integer_ai.experiments.trained_query_evidence import core_query_evidence
from pure_integer_ai.experiments.trained_relation_composition_graph import (
    PART_TABLE, materialize_relation_composition_graph, read_relation_composition_graph,
)
from pure_integer_ai.experiments.trained_relation_question_graph import materialize_relation_question_graph
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.experiments.verified_composition_query import consume_verified_composition
from pure_integer_ai.storage.backend import SQLiteBackend


def _rows(*, disconnected=False, conflict=False):
    return (
        {"anchor": "属于", "case_id": "member", "relation": "MEMBER",
         "sample_role": "conflict" if conflict else "support", "seed_id": "member", "split": "train",
         "surface": "晶羽甲属于青纹群。",
         "terms": [["晶羽甲", "entity", 202610130701], ["青纹群", "set", 202610130702]]},
        {"anchor": "包含于", "case_id": "subset", "relation": "SUBSET",
         "sample_role": "support", "seed_id": "subset", "split": "train",
         "surface": "青纹群包含于银纹群。",
         "terms": [["青纹群", "set", 202610130799 if disconnected else 202610130702],
                   ["银纹群", "set", 202610130703]]},
    )


def _build(root, *, disconnected=False, conflict=False, question_graph=False,
           generation=False):
    source = root / "data" / "ph2" / "facts.jsonl"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in _rows(
        disconnected=disconnected, conflict=conflict)))
    facts = compile_independent_relation_fact_source(source, root / "teachers")
    course = root / "course.int.json"
    write_integer_relation_graph_course(course, facts.course)
    database = root / "core.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        context = make_train_context(backend)
        _, occurrence, span = dialogue_semantic_protocols()
        install_language_graph_protocols(context, occurrence_protocol=occurrence, span_protocol=span)
        learning = build_integer_w06_learning_runtime(
            backend, context, (course,), require_full_curriculum=False)
        assert learning.report().active_candidate_count == (1 if conflict else 2)
        backend.commit()
    finally:
        backend.close()
    if generation:
        result = materialize_integer_relation_generation(
            database, branch_key=(92113, 1), family_key=(92113, 2),
            representative_limit=2, require_full_active_coverage=True)
        assert result.connector_count == result.active_proposition_count == 2
    instruction = root / "instruction.json"
    instruction.write_bytes(canonical_json_bytes({
        "instruction": [4, 7, 8, 3, 5, 6, 4, 7, 8],
        "license_id": "CC0-1.0", "owner": "teacher", "rule_id": [92110, 1], "split": "train",
    }) + b"\n")
    witness = compile_train_relation_composition(instruction, facts)
    graph = root / "rules.sqlite3"
    materialize_relation_composition_graph((witness,), graph)
    if question_graph:
        question_source = root / "question.train.json"
        question_source.write_bytes(canonical_json_bytes({
            "license_id": "CC0-1.0", "owner": "teacher", "split": "train",
            "question": "晶羽甲属于青纹群吗？", "source_case_key": [92112, 1],
            "operation_key": [92112, 2], "seed_id": "member",
        }) + b"\n")
        syntax = json.loads(question_source.read_bytes())
        frame = compile_train_relation_question(
            _seed(_rows()[0], 1), syntax["question"],
            source_case_key=tuple(syntax["source_case_key"]),
            operation_key=tuple(syntax["operation_key"]))
        assert frame is not None
        questions = root / "questions.sqlite3"
        materialize_relation_question_graph(
            (frame,), questions, source_identity=facts.course.source_identity,
            license_identity=facts.course.license_identity)
        return database, witness, graph, questions
    return database, witness, graph


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("core-composition")
    database, witness, graph = _build(root)
    return root, database, witness, graph


def test_licensed_instruction_and_graph_cold_restore_preserve_full_schemas(fixture):
    _, database, witness, graph = fixture
    before = file_sha256(graph), file_sha256(database)
    assert RelationCompositionWitness.from_stable_key(witness.stable_key()) == witness
    assert read_relation_composition_graph(graph) == (witness,)
    with TrainedRelationGraphRuntime(database) as core:
        pending = pending_core_compositions(core, read_relation_composition_graph(graph))
        assert len(pending) == 1
        candidate = pending[0].candidate
        assert PendingRelationComposition.from_stable_key(pending[0].stable_key()) == pending[0]
        with pytest.raises(ValueError):
            PendingRelationComposition.from_stable_key(pending[0].stable_key()[:-1])
        assert len(candidate.premises) == 2
        assert len({core.generation_input(item.proposition).proposition.definition.predicate
                    for item in candidate.premises}) == 2
        assert candidate.relation == witness.rule.result_relation
        assert not any(candidate.bindings == item.definition.canonical_bindings()
                       and candidate.relation == item.definition.predicate
                       for item in core.active_propositions())
        for premise in candidate.premises:
            source = core.generation_input(premise.proposition)
            assert premise.hypothesis == source.hypothesis
            assert premise.support_evidence_ids == source.snapshot.support_evidence_ids
    assert (file_sha256(graph), file_sha256(database)) == before


@pytest.mark.parametrize("mutation", ["no_rule", "foreign_source", "wrong_role"])
def test_rule_absence_permission_and_role_mutations_do_not_derive(fixture, mutation):
    _, database, witness, _ = fixture
    rules = (witness,)
    if mutation == "no_rule":
        rules = ()
    elif mutation == "foreign_source":
        source = witness.authorized_forming_sources[0]
        rules = (replace(witness, authorized_forming_sources=(
            SourceRef(source.source_kind, source.source_id + 1, source.document_id,
                      source.owner, source.versions),)),)
    else:
        rules = (replace(witness, rule=replace(witness.rule,
            second_join_role=witness.rule.second_output_role,
            second_output_role=witness.rule.second_join_role)),)
    with TrainedRelationGraphRuntime(database) as core:
        assert pending_core_compositions(core, rules) == ()


@pytest.mark.parametrize("disconnected,conflict", [(True, False), (False, True)])
def test_same_surface_with_distinct_identity_and_conflicted_premise_reject(tmp_path, disconnected, conflict):
    database, witness, _ = _build(tmp_path, disconnected=disconnected, conflict=conflict)
    with TrainedRelationGraphRuntime(database) as core:
        assert pending_core_compositions(core, (witness,)) == ()


def test_pending_proof_requires_exact_consumed_support_and_never_adds_support(fixture):
    _, database, witness, _ = fixture
    with TrainedRelationGraphRuntime(database) as core:
        pending = pending_core_compositions(core, (witness,))
        query = QueryState((92120, 1))
        assert pending_composition_query_evidence(pending, query) == ()
        groups = tuple(core_query_evidence(
            premise.proposition, core.evidence_history(premise.proposition),
            core.context.scoped_identity_store.registry)
            for premise in pending[0].candidate.premises)
        assert pending_composition_query_evidence(pending, query.with_(evidence=groups[0])) == ()
        entries = tuple(sorted((*groups[0], *groups[1]), key=lambda item: item.stable_key()))
        proof = pending_composition_query_evidence(pending, query.with_(evidence=entries))
        assert len(proof) == 1 and proof[0].polarity == 3
        assert proof[0].payload_key == pending[0].stable_key()
        support = next(item for item in entries if item.polarity == 1)
        record = EvidenceRecord.from_stable_key(support.payload_key)
        stale = replace(record, evidence_id=record.evidence_id + 100000)
        wrong = tuple(item if item != support else replace(item, payload_key=stale.stable_key())
                      for item in entries)
        assert pending_composition_query_evidence(pending, query.with_(evidence=wrong)) == ()
        changed_source = replace(record, source=replace(
            record.source, source_id=record.source.source_id + 1))
        changed_payload = replace(record, payload=(*record.payload, 1))
        for mutation in (changed_source, changed_payload):
            tampered = tuple(item if item != support else replace(
                item, payload_key=mutation.stable_key()) for item in entries)
            assert pending_composition_query_evidence(
                pending, query.with_(evidence=tampered)) == ()
        for field in ("source_ref", "scope_key", "evidence_key", "supersedes_key"):
            tampered_entry = replace(support, **{field: (*getattr(support, field), 1)})
            tampered = tuple(item if item != support else tampered_entry for item in entries)
            assert pending_composition_query_evidence(
                pending, query.with_(evidence=tampered)) == ()
        assert pending_composition_query_evidence(
            pending, query.with_(evidence=(*entries, support))) == ()
        projected = replace(support, polarity=3, evidence_key=(91569, 1),
                            payload_key=(91569, 1))
        assert pending_composition_query_evidence(
            pending, query.with_(evidence=(*entries, projected))) == proof
        for mutation in (replace(projected, polarity=2), replace(projected, polarity=1)):
            assert pending_composition_query_evidence(
                pending, query.with_(evidence=(*entries, mutation))) == ()


def test_composition_budgets_fail_before_eager_rule_cross_product(fixture):
    _, database, witness, _ = fixture
    second = replace(witness, rule=replace(witness.rule,
        rule=minimal_instruction_identity((92110, 3),
            versions=witness.rule.rule.versions)))
    with TrainedRelationGraphRuntime(database) as core:
        with pytest.raises(ValueError, match="witness budget"):
            pending_core_compositions(core, (witness, second),
                budget=RelationCompositionBudget(max_witnesses=1))
        with pytest.raises(ValueError, match="active fact budget"):
            pending_core_compositions(core, (witness,),
                budget=RelationCompositionBudget(max_active_facts=1))
        with pytest.raises(ValueError, match="rule pair budget"):
            pending_core_compositions(core, (witness, second),
                budget=RelationCompositionBudget(max_rule_pairs=1))
        with pytest.raises(ValueError, match="candidate budget"):
            pending_core_compositions(core, (witness, second),
                budget=RelationCompositionBudget(max_candidates=1))


@pytest.mark.parametrize("field", ["source_identity", "license_identity"])
def test_composition_question_provenance_requires_exact_permission(fixture, field):
    _, _, witness, _ = fixture
    metadata = {"source_identity": witness.source_identity,
                "license_identity": witness.license_identity}
    validate_relation_composition_provenance((witness,), metadata)
    metadata[field] = (*metadata[field], 1)
    with pytest.raises(ValueError, match="different source permission"):
        validate_relation_composition_provenance((witness,), metadata)


def test_physical_rule_digest_is_a_complete_integer_record(fixture):
    _, _, witness, _ = fixture
    assert len(witness.source_sha256) == 32
    with pytest.raises(ValueError, match="physical instruction digest"):
        replace(witness, source_sha256=witness.source_sha256[:-1])


def test_composition_graph_rejects_corrupt_integer_parts(tmp_path, fixture):
    _, _, witness, _ = fixture
    graph = tmp_path / "rules.sqlite3"
    materialize_relation_composition_graph((witness,), graph)
    with sqlite3.connect(graph) as connection:
        connection.execute(f"DELETE FROM {PART_TABLE} WHERE part_ordinal=3")
    with pytest.raises(RuntimeError, match="incomplete"):
        read_relation_composition_graph(graph)


def test_train_instruction_rejects_evaluator_source(tmp_path, fixture):
    root, _, _, _ = fixture
    facts = compile_independent_relation_fact_source(root / "data/ph2/facts.jsonl", tmp_path / "teachers")
    source = tmp_path / "instruction.json"
    source.write_bytes(canonical_json_bytes({
        "instruction": [4, 7, 8, 3, 5, 6, 4, 7, 8],
        "license_id": "CC0-1.0", "owner": "evaluator", "rule_id": [92110, 2], "split": "held_out",
    }) + b"\n")
    with pytest.raises(ValueError, match="train teacher"):
        compile_train_relation_composition(source, facts)


def test_natural_input_consumes_core_premises_but_unverified_conclusion_is_unknown(tmp_path):
    evidence_root = os.environ.get("PIA_RELATION_VALIDATION_EVIDENCE_ROOT")
    if evidence_root:
        tmp_path = Path(evidence_root) / "development_composition_artifacts"
        tmp_path.mkdir(exist_ok=False)
    database, witness, graph, questions = _build(
        tmp_path, question_graph=True, generation=True)
    frozen = file_sha256(database), file_sha256(graph), file_sha256(questions)
    with TrainedGenerationConnectorRuntime(database) as generator, TrainedGraphQueryBridge(
            database, memory_database=tmp_path / "memory.sqlite3", session_id=92160,
            surface_generator=generator,
            relation_composition_verification=False,
            relation_composition_graph_database=graph,
            relation_question_graph_database=questions) as bridge:
        assert len(bridge.pending_relation_compositions) == 1
        with TrainedRelationGraphRuntime(database) as core:
            for fact in core.active_surface_facts():
                generated = generator.generate_relation(
                    core.generation_input(fact.proposition), fact)
                assert generated.connector is not None and generated.representations
        surface = "晶羽甲属于银纹群吗？"
        prepared = bridge.prepare_input_structure(surface)
        assert len(prepared.pending_relation_composition_bindings) == 1
        for invalid in ("请问晶羽甲属于银纹群吗？", "晶羽甲属于银纹群吗？还有呢",
                        "银纹群属于晶羽甲吗？", "未知甲属于银纹群吗？"):
            try:
                other = bridge.prepare_input_structure(invalid)
            except ValueError:
                continue
            assert not other.pending_relation_composition_bindings
        assert match_pending_relation_questions(
            (), prepared, bridge.pending_relation_compositions) == ()
        assert match_pending_relation_questions(
            bridge.relation_question_frames, prepared, ()) == ()
        no_graph = object.__new__(TrainedGraphQueryBridge)
        no_graph.relation_question_frames = ()
        with pytest.raises(ValueError, match="no trained graph"):
            no_graph._attach_relation_question_binding(prepared)
        original = next(item for item in prepared.semantic_candidates
                        if item.projection_kind == PROJECTION_FILLER
                        and item.span_ref == prepared.pending_relation_composition_bindings[0].slot_span_refs[0]
                        and not item.concept_route_key)
        rival = replace(original, candidate_key=(
            *original.candidate_key[:-1], original.candidate_key[-1] + 1))
        ambiguous = replace(prepared, semantic_candidates=tuple(sorted(
            (*prepared.semantic_candidates, rival), key=lambda item: item.stable_key())))
        assert match_pending_relation_questions(
            bridge.relation_question_frames, ambiguous,
            bridge.pending_relation_compositions) == ()
        forged = replace(prepared.pending_relation_composition_bindings[0],
                         pending_key=(92107, 99))
        with pytest.raises(ValueError, match="not authorized"):
            bridge._attach_relation_question_binding(replace(
                prepared, pending_relation_composition_bindings=(forged,)))
        report = run_natural_relation_case(
            bridge, surface, tmp_path, "cases/001", case_ordinal=1,
            max_depth=24, max_nodes=512, max_edges=1024, max_reads=2048,
            max_case_wall_seconds=300)
        result = json.loads((tmp_path / "cases/001/trace.int.json").read_text("utf-8"))
        trace = GraphQueryTraceEvidence(result)
        proofs = tuple(item for item in result["evidence"]
                       if trace.resolve_ref(item["payload_ref"])
                       and trace.resolve_ref(item["payload_ref"])[0] == COMPOSITION_PENDING_PROOF)
        assert len(proofs) == 1 and proofs[0]["polarity"] == 3
        assert result["termination"] != TERMINATION_ANSWER_CLOSED
        assert not result["response_plan"] and not result["response_surface"]
        assert report["delivery_claim_count"] == 0
        assert report["hypothesis_count"] > 0 and report["evidence_count"] > 0
        assert trace.counts()["core_hop_count"] > 0 and trace.counts()["memory_hop_count"] > 0
        verify_pending_composition_trace(
            result, report, bridge.pending_relation_compositions[0], prepared)
        changed = deepcopy(result)
        premise_key = bridge.pending_relation_compositions[0].premise_evidence_keys[0][0]
        removed_ref = next(ref for ref in changed["state_key"]["evidence"]
                           if trace.resolve_ref(ref) == premise_key)
        changed["state_key"]["evidence"].remove(removed_ref)
        with pytest.raises(ValueError, match="exact Core premise"):
            verify_pending_composition_trace(
                changed, report, bridge.pending_relation_compositions[0], prepared)
        if evidence_root:
            write_preflight_json(tmp_path, "checkpoint.json", {
                "stage": "CORE_TWO_RELATION_PENDING_CANDIDATE_UNKNOWN_NOT_WAVE1",
                "core_sha256": file_sha256(database), "rules_sha256": file_sha256(graph),
                "candidate_key": list(bridge.pending_relation_compositions[0].stable_key()),
                "composition_budget": asdict(RelationCompositionBudget()),
                "question_sha256": file_sha256(questions),
                "premise_generation_template_count": generator.template_count,
                "call_receipt": report, "query_count": 1, "free_dialogue_complete": 0,
            })
    assert (file_sha256(database), file_sha256(graph), file_sha256(questions)) == frozen


def test_independent_join_verifier_rejects_unauthorized_roles_and_conclusion_refute(fixture):
    _, database, witness, _ = fixture
    with TrainedRelationGraphRuntime(database) as core:
        pending = pending_core_compositions(core, (witness,))[0]
        facts = tuple(ActiveSupportedRelationFact(source.proposition.definition, source.snapshot,
            core.owner.learning.engine.definition(source.hypothesis))
            for premise in pending.candidate.premises
            for source in (core.generation_input(premise.proposition),))
        verified = admit_verified_composition(pending, (92187, 1), facts,
            pending.premise_evidence_keys, (witness,))
        assert verified is not None
        assert VerifiedRelationComposition.from_stable_key(verified.stable_key()) == verified
        with pytest.raises(ValueError):
            VerifiedRelationComposition.from_stable_key(verified.stable_key()[:-1])
        assert verified.active.hypothesis in verified.active.decision.adopted_hypotheses
        assert len(verified.active.snapshot.support_evidence_ids) == 1
        assert verified.verifier_trace[-1] == 1
        assert any(entry.stance == 3 for entry in verified.history)
        assert any(entry.stance == 1 for entry in verified.history)
        assert admit_verified_composition(pending, (92187, 1), facts,
            pending.premise_evidence_keys, (witness,), conclusion_refuted=True) is None
        other = replace(witness, rule=replace(witness.rule, rule=minimal_instruction_identity(
            (92187, 2), versions=witness.rule.rule.versions)))
        assert admit_verified_composition(pending, (92187, 1), facts,
            pending.premise_evidence_keys, (other,)) is None
        assert admit_verified_composition(pending, (92187, 1), facts[::-1],
            pending.premise_evidence_keys, (witness,)) is None
        bindings = pending.candidate.bindings
        swapped = tuple(replace(binding, filler=bindings[1-index].filler)
                        for index, binding in enumerate(bindings))
        # MEMBER's typed schema also rejects placing a SetExpr in its Entity role.
        with pytest.raises(ValueError):
            candidate = replace(pending.candidate, bindings=swapped)
            admit_verified_composition(replace(pending, candidate=candidate), (92187, 1),
                facts, pending.premise_evidence_keys, (witness,))
        denied = replace(witness, authorized_forming_sources=(witness.source,))
        assert admit_verified_composition(replace(pending, witness=denied), (92187, 1),
            facts, pending.premise_evidence_keys, (denied,)) is None


def test_natural_derived_result_uses_current_core_h05_graph_generation_and_delivery(tmp_path):
    evidence_root = os.environ.get("PIA_RELATION_VALIDATION_EVIDENCE_ROOT")
    if evidence_root:
        tmp_path = Path(evidence_root) / "development_verified_composition_artifacts"
        tmp_path.mkdir(exist_ok=False)
    database, witness, graph, questions = _build(tmp_path, question_graph=True, generation=True)
    frozen = tuple(file_sha256(path) for path in (database, graph, questions))
    with TrainedGenerationConnectorRuntime(database) as generator, TrainedGraphQueryBridge(
            database, memory_database=tmp_path / "memory.sqlite3", session_id=92188,
            surface_generator=generator, relation_composition_graph_database=graph,
            relation_question_graph_database=questions) as bridge:
        surface = "晶羽甲属于银纹群吗？"
        prepared = bridge.prepare_input_structure(surface)
        assert len(prepared.pending_relation_composition_bindings) == 1
        pending = bridge.pending_relation_compositions[0]
        state = QueryState((92188, 1))
        assert consume_verified_composition(bridge.core_runtime, (witness,), pending,
            state, generator, RelationCompositionBudget()) == (state, None)
        premise_entries = tuple(entry for premise in pending.candidate.premises
            for entry in core_query_evidence(premise.proposition,
                bridge.core_runtime.evidence_history(premise.proposition),
                bridge.core_runtime.context.scoped_identity_store.registry))
        supplied = state.with_(evidence=premise_entries)
        assert consume_verified_composition(bridge.core_runtime, (witness,), pending,
            supplied, None, RelationCompositionBudget()) == (supplied, None)
        with pytest.raises(ValueError, match="verified proof budget"):
            consume_verified_composition(bridge.core_runtime, (witness,), pending,
                supplied, generator, RelationCompositionBudget(max_verified_proof_values=1))
        for invalid in ("银纹群属于晶羽甲吗？", "未知甲属于银纹群吗？",
                        "晶羽甲属于银纹群吗？还有呢"):
            assert not bridge.prepare_input_structure(invalid).pending_relation_composition_bindings
        report = run_natural_relation_case(bridge, surface, tmp_path, "cases/001", case_ordinal=1,
            max_depth=24, max_nodes=512, max_edges=1024, max_reads=2048,
            max_case_wall_seconds=300)
        result = json.loads((tmp_path / "cases/001/trace.int.json").read_text("utf-8"))
        assert result["termination"] == TERMINATION_ANSWER_CLOSED
        assert result["response_surface"] == "晶羽甲属于银纹群"
        saved = tmp_path / "cases/001/case_provenance.json"
        compact = json.loads(saved.read_bytes())
        assert compact["format"] == COMPACT_PROVENANCE_FORMAT
        assert "trace" not in compact["provenance"]
        restored = load_case_provenance(saved)
        assert GraphQueryTraceEvidence.from_integer_artifact(restored["trace"]).counts() == (
            GraphQueryTraceEvidence(result).counts())
        assert report["delivery_claim_count"] == 1
        assert result["generation"]["kind"] == 5
        trace = GraphQueryTraceEvidence(result)
        proof = trace.resolve_ref(result["generation"]["verified_composition"]["integer_key_ref"])
        assert proof[0] == COMPOSITION_VERIFIED_V2
        assert VerifiedRelationComposition.from_stable_key(proof).pending == pending
        assert result["generation"]["request_local_derivation"] == 1
        assert result["generation"]["core_claims_written"] == 0
        claims = result["response_plan"]["claim_refs"]
        assert len(claims) == 1
        assert tuple(claims[0]) not in bridge.core_fact_index
        assert report["positive_chain_phase_coverage"] == 1
        assert report["phase_boundary_status"] == "PASS"
        assert report["capture_present"] == 1
        assert all(report["trace_counts"][key] > 0 for key in
                   ("core_hop_count", "memory_hop_count", "companion_hop_count"))
        if evidence_root:
            write_preflight_json(tmp_path, "checkpoint.json", {
                "stage": "CORE_TWO_RELATION_QUERY_LOCAL_H05_DERIVED_OUTPUT_NOT_WAVE1",
                "core_sha256": frozen[0], "rules_sha256": frozen[1],
                "question_sha256": frozen[2], "query_count": 1, "call_receipt": report,
                "composition_budget": asdict(RelationCompositionBudget()),
                "derived_core_claim_count": 0, "free_dialogue_complete": 0,
            })
    assert tuple(file_sha256(path) for path in (database, graph, questions)) == frozen
