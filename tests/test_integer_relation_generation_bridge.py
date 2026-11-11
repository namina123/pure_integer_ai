"""Regression coverage for the same-database integer relation generation bridge."""
from __future__ import annotations

from pathlib import Path
import os
import json
from contextlib import nullcontext

from pure_integer_ai.experiments.relation_runtime_phase import CASE_PHASES, capture_relation_phases

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.query_state import TERMINATION_ANSWER_CLOSED
from pure_integer_ai.cognition.understanding.query_relation_questions import TrainedRelationQuestionFrame
from pure_integer_ai.experiments.batch_case_provenance import (
    capture_factual_case_provenance,
)
from pure_integer_ai.experiments.conversation_typed_relation_bridge import (
    build_integer_w06_learning_runtime,
)
from pure_integer_ai.experiments.integer_relation_generation_bridge import (
    materialize_integer_relation_generation,
)
from pure_integer_ai.experiments.natural_relation_preflight import run_natural_relation_case
from pure_integer_ai.experiments.integer_relation_graph_course import (
    integer_course_from_w06_adapter,
    write_integer_relation_graph_course,
)
from pure_integer_ai.experiments.factual_delivery_proof_audit import (
    verify_factual_delivery_proof,
)
from pure_integer_ai.experiments.factual_delivery_event_audit import (
    verify_factual_delivery_events,
)
from pure_integer_ai.experiments.query_memory_ohe_audit import verify_query_memory_ohe
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_w06_adapter import (
    adapt_w06_training_payload,
)
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload
from pure_integer_ai.experiments.run_conversation_training import (
    dialogue_semantic_protocols,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.trained_generation_connector_runtime import (
    TrainedGenerationConnectorRuntime,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    TrainedDialogueMemoryGraph,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.experiments.trained_relation_question_graph import (
    materialize_relation_question_graph,
)
from pure_integer_ai.experiments.trained_response_delivery import (
    read_delivery_proof,
    read_factual_evidence_ref,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "ph2_w06_dataset_artifacts" / "source_semantic_overlay_v1" / "packs"


def _authored_adapter():
    sources = []
    observations = []
    teachers = []
    for manifest_path in sorted(PACK_ROOT.glob(
            "AUTHORED_CC0_V1--CC0-1.0--*/manifest.json")):
        manifest = read_artifact_manifest(manifest_path)
        for identity in manifest.files:
            records = read_record_artifact(manifest_path.parent, identity)
            if identity.record_kind == RECORD_SOURCE_REF:
                sources.extend(records)
            elif identity.record_kind == RECORD_OBSERVATION:
                observations.extend(
                    item for item in records if item.split == "train")
            elif identity.record_kind == RECORD_TEACHER_EVIDENCE:
                teachers.extend(records)
    return adapt_w06_training_payload(W06TrainingPayload(
        tuple(sources), tuple(observations), tuple(teachers)))


def _audit_delivery_events(captured, memory):
    events = memory.intake.event_log
    access = memory.candidate_index.access
    return verify_factual_delivery_events(
        captured,
        proof_reader=lambda key: read_delivery_proof(
            memory, ObjectIdentity.from_stable_key(key)),
        evidence_reader=lambda key: read_factual_evidence_ref(
            memory, ObjectIdentity.from_stable_key(key)),
        event_reader=lambda event_hash: events.read(event_hash, access=access),
        declaration_reader=lambda ref: tuple(
            row for row in events.query(access=access, object_ref=ref)
            if row.event.is_declaration),
    )


def _audit_query_ohe(captured, memory):
    events = memory.intake.event_log
    access = memory.candidate_index.access
    return verify_query_memory_ohe(
        captured,
        declaration_reader=lambda ref: tuple(
            row for row in events.query(access=access, object_ref=ref)
            if row.event.is_declaration),
        current_manifest_reader=memory.intake.require_current_manifest,
        source_hash_reader=lambda source: events.scoped_identities.registry.find(
            IDENTITY_SOURCE_RECORD, source.stable_key()),
    )


def test_integer_relation_generation_closes_querybridge_delivery(tmp_path, monkeypatch):
    adapter = _authored_adapter()
    course = integer_course_from_w06_adapter(
        adapter,
        source_identity=(91660, 999, 20261008, 1),
        license_identity=(91661, 999, 1),
    )
    course_path = tmp_path / "relation-course.int.json"
    write_integer_relation_graph_course(course_path, course)
    database = tmp_path / "training.sqlite3"

    backend = SQLiteBackend(str(database))
    try:
        context = make_train_context(backend)
        _semantic, occurrence, span = dialogue_semantic_protocols()
        install_language_graph_protocols(
            context, occurrence_protocol=occurrence, span_protocol=span)
        runtime = build_integer_w06_learning_runtime(
            backend, context, (course_path,))
        backend.commit()
        assert runtime.report().candidate_count == 40
    finally:
        backend.close()

    materialized = materialize_integer_relation_generation(
        database,
        branch_key=(91662, 999, 1),
        family_key=(91663, 999, 1),
        representative_limit=14,
    )
    assert materialized.connector_count >= 5
    assert materialized.realization_count >= materialized.connector_count

    question_operation = (91680, 999, 1)
    source_identity = (91660, 999, 20261008, 1)
    license_identity = (91661, 999, 1)

    with TrainedRelationGraphRuntime(database) as relation:
        fact = next(item for item in relation.active_surface_facts()
                    if item.proposition.stable_key()
                    == materialized.proposition_keys[0])
        source = relation.generation_input(fact.proposition)
        query = "请确认：" + fact.evidence_surface
        # R50 起要求完整来源化语法，单独的操作词不能授权任意外围文字。
        # 本专项显式训练测试载体的完整间隔，仍由原输入覆盖门消费。
        frame = next(item for item in relation.active_surface_frames()
                     if item.proposition == fact.proposition)
        question_database = tmp_path / "question.sqlite3"
        materialize_relation_question_graph((TrainedRelationQuestionFrame(
            (92456, 999, 1), question_operation, fact.predicate.stable_key(),
            tuple(role.stable_key() for role in frame.roles),
            (tuple(map(ord, "请确认：" + fact.evidence_surface[:frame.envelope_start] + frame.gaps[0])),
             *(tuple(map(ord, gap)) for gap in frame.gaps[1:-1]),
             tuple(map(ord, frame.gaps[-1] + fact.evidence_surface[frame.envelope_end:])))),), question_database,
            source_identity=source_identity, license_identity=license_identity)
        with TrainedGenerationConnectorRuntime(database) as generation:
            direct = generation.generate_relation(source, fact)
            assert generation.template_count == materialized.connector_count
            assert direct.slot_count == 3
            assert len(direct.representations) == 3
            assert direct.surface

            phase_root = Path(os.environ.get("PIA_RELATION_VALIDATION_EVIDENCE_ROOT", str(tmp_path)))
            phase_scope = capture_relation_phases(
                    phase_root / "actual-runtime-phases", run_identity="development-mainline",
                    attempt_key="one", case_ordinal=1,
                    required_phases=("memory_init",)
                ) if phase_root.drive.upper() == "K:" else nullcontext(None)
            with phase_scope as phases, TrainedGraphQueryBridge(
                    database,
                    memory_database=tmp_path / "memory.sqlite3",
                    surface_generator=generation,
                    session_id=100809,
                    relation_question_graph_database=question_database,
            ) as bridge:
                if phases is not None:
                    receipts = []
                    acknowledge = bridge.acknowledge_response_delivery

                    def capture_delivery(surface):
                        """仅保留实际交付返回值，仍执行原生产方法一次。"""
                        receipt = acknowledge(surface)
                        receipts.append(receipt)
                        return receipt

                    monkeypatch.setattr(bridge, "acknowledge_response_delivery", capture_delivery)
                    call = run_natural_relation_case(
                        bridge, query, phase_root, "actual-natural-case",
                        max_depth=64, max_nodes=4096, max_edges=8192, max_reads=16384,
                        max_case_wall_seconds=600, case_ordinal=1)
                    assert call["phase_boundary_status"] == "PASS"
                    assert call["positive_chain_phase_coverage"] == 1
                    assert call["projection"]["operation_keys"] == [list(question_operation)]
                    result = json.loads((phase_root / "actual-natural-case/trace.int.json").read_bytes())
                    observation = MemoryObjectRef.from_stable_key(tuple(call["observation_ref"]))
                    input_source_key = tuple(call["input_source"])
                    assert len(receipts) == 1
                    delivered = receipts[0]
                else:
                    structure = bridge.prepare_input_structure(query)
                    appended = bridge.memory.append(
                        query, speaker_kind=1,
                        input_structure=bridge.active_memory_input_structure(structure))
                    observation = bridge.memory.intake.result_for_source(appended.source).observation_ref
                    input_source_key = appended.source.stable_key()
                    assert structure.operation_keys == (question_operation,)
                    assert observation is not None
                    result = bridge.query(
                        query, input_structure=structure, input_observation_ref=observation,
                        compact_trace=True, max_depth=64, max_nodes=4096, max_edges=8192,
                        max_reads=16384)
                    delivered = bridge.acknowledge_response_delivery(result["response_surface"])
                assert result["termination"] == TERMINATION_ANSWER_CLOSED
                assert result["response_plan"] is not None
                assert result["generation"] is not None
                output = result["response_surface"]
                assert isinstance(output, str) and output
                assert delivered is not None
                captured = capture_factual_case_provenance(
                    query_result=result,
                    query_values=tuple(map(ord, query)),
                    expected_claim=fact.proposition.stable_key(),
                    expected_operation=question_operation,
                    input_observation=observation.stable_key(),
                    input_source=input_source_key,
                    delivery_summary=delivered.summary(),
                )
                audited = verify_factual_delivery_proof(
                    captured,
                    proof_reader=lambda key: read_delivery_proof(
                        bridge.memory, ObjectIdentity.from_stable_key(key)),
                    evidence_reader=lambda key: read_factual_evidence_ref(
                        bridge.memory, ObjectIdentity.from_stable_key(key)),
                )
                assert audited["status"] == "MATCHED_DURABLE_PROOF_CONTENT_TO_CAPTURED_PLAN"
                assert audited["claim_count"] == delivered.claim_count == 1
                event_audited = _audit_delivery_events(captured, bridge.memory)
                assert event_audited["status"] == "MATCHED_DURABLE_DELIVERY_EPISODE_AND_USES"
                assert event_audited["used_memory_ref_count"] == len(delivered.uses)
                ohe_audited = _audit_query_ohe(captured, bridge.memory)
                assert ohe_audited["status"] == "MATCHED_DURABLE_INTAKE_OHE_QUERYSTATE_AND_PLAN_REFERENCES"
                # The direct connector includes its predicate slot; the
                # delivered factual plan contains the relation's semantic roles.
                assert audited["slot_count"] == delivered.slot_count == len(fact.bindings)
                assert {
                    (tuple(slot["role"]), tuple(slot["filler"]))
                    for slot in result["response_plan"]["slot_sequence"]
                } == {
                    (binding.role.stable_key(), binding.filler.stable_key())
                    for binding in fact.bindings
                }

    if phases is not None:
        assert phases.report["status"] == "PASS"
        assert phases.report["language_qualification"].startswith("UNKNOWN")

    # The proof must survive a fresh Memory owner, not merely the live
    # QueryBridge registry cache used to create it.
    with TrainedDialogueMemoryGraph(
            tmp_path / "memory.sqlite3", session_id=100809) as restored_memory:
        cold_audit = verify_factual_delivery_proof(
            captured,
            proof_reader=lambda key: read_delivery_proof(
                restored_memory, ObjectIdentity.from_stable_key(key)),
            evidence_reader=lambda key: read_factual_evidence_ref(
                restored_memory, ObjectIdentity.from_stable_key(key)),
        )
        assert cold_audit == audited
        assert _audit_delivery_events(captured, restored_memory) == event_audited
        assert _audit_query_ohe(captured, restored_memory) == ohe_audited
