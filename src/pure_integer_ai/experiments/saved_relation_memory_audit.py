"""Cold O/H/E, proof, Episode and Use checks for one original natural call."""
from __future__ import annotations

from types import SimpleNamespace

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION, MEMORY_CATEGORY_OPEN_RELATION, frame_key, memory_input_candidate_keys,
)
from pure_integer_ai.experiments.factual_delivery_event_audit import verify_factual_delivery_events
from pure_integer_ai.experiments.factual_delivery_proof_audit import verify_factual_delivery_proof
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.query_memory_ohe_audit import (
    QueryMemoryOheEvidenceError, verify_query_memory_ohe, verify_query_memory_ohe_without_plan,
    verify_query_memory_ohe_nonfactual,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import _build_interaction_memory_intake
from pure_integer_ai.experiments.trained_response_delivery import read_delivery_proof, read_factual_evidence_ref
from pure_integer_ai.experiments.trained_response_occurrence import build_response_occurrence_index, read_input_roles
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD
from pure_integer_ai.storage.backend import SQLiteBackend


def audit_saved_relation_memory(database, value, report, prepared, *, captured=None, max_state_members=4096,
                                require_operation_request_memory=False, report_incomplete_ohe=False,
                                max_ohe_declarations=256, reference_bridge=None):
    backend = SQLiteBackend(str(database), read_only=True)
    try:
        intake = _build_interaction_memory_intake(backend)
        owner = SourceRef.from_stable_key(tuple(report["input_source"])).owner
        access = MemoryAccessContext(owner.tenant_id, owner.user_id, owner.session_id)
        events = intake.event_log
        declarations = lambda ref: tuple(row for row in events.query(access=access, object_ref=ref)
                                          if row.event.is_declaration)
        memory = SimpleNamespace(owner=owner, intake=intake, candidate_index=SimpleNamespace(access=access),
                                 occurrence_index=build_response_occurrence_index(events),
                                 input_projector=None if reference_bridge is None else reference_bridge.input_projector)
        refs = (report["observation_ref"], *report["hypothesis_refs"], *report["evidence_refs"])
        if any(len(declarations(MemoryObjectRef.from_stable_key(tuple(ref)))) != 1 for ref in refs):
            raise ValueError("saved call lacks a unique original O/H/E declaration")
        candidates = {frame_key(MEMORY_CANDIDATE_VERSION, (MEMORY_CATEGORY_OPEN_RELATION,), item.stable_key()): item
                      for item in prepared.open_relations}
        observation = MemoryObjectRef.from_stable_key(tuple(report["observation_ref"]))
        occurrences = []
        relation_map = {}
        for key in report["hypothesis_refs"]:
            ref = MemoryObjectRef.from_stable_key(tuple(key))
            row = declarations(ref)[0]
            candidate = candidates.get(row.event.payload.hypothesis.candidate_key)
            if candidate is not None:
                relation_map[ref.stable_key()] = candidate
                occurrences.extend(read_input_roles(memory, candidate, observation, ref, required=True))
        occurrences = tuple(sorted(occurrences, key=lambda item: item.stable_key()))
        trace = GraphQueryTraceEvidence(value)
        has_reference = bool(value.get('reference_candidate_set_refs') or value.get('reference_resolution_refs'))
        if not has_reference and tuple(item.stable_key() for item in occurrences) != tuple(trace.resolve_ref(ref)
                for ref in value["input_role_occurrence_refs"]):
            raise ValueError("saved input Role occurrences differ from actual Memory graph")
        readers = dict(declaration_reader=declarations, current_manifest_reader=intake.require_current_manifest,
            source_hash_reader=lambda ref: events.scoped_identities.registry.find(IDENTITY_SOURCE_RECORD, ref.stable_key()),
            input_role_occurrences=occurrences, max_state_members=max_state_members,
            max_ohe_declarations=max_ohe_declarations)
        if reference_bridge is not None:
            readers['revision_cue_records'] = reference_bridge.revision_cue_records
        request_memory = {}
        reference_transition = {}
        complete_response_contexts = ()
        if require_operation_request_memory:
            from pure_integer_ai.experiments.operation_request_call_audit import verify_operation_request_memory
            request_memory = verify_operation_request_memory(prepared, value, report, declarations)
        try:
            from pure_integer_ai.experiments.saved_response_context_audit import read_saved_factual_response_contexts
            from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack
            input_role_keys = {item.stable_key() for item in occurrences}
            response_roles_consumed = False
            response_role_keys = set()
            for ref in value["state_key"]["evidence"]:
                key = trace.resolve_ref(ref)
                member = _unpack(key, 3, 6)[2]
                if key[0] == 2 and member[:1] == (91534,) and member not in input_role_keys:
                    response_roles_consumed = True
                    response_role_keys.add(member)
            if response_roles_consumed or value.get('response_context_refs'):
                try:
                    complete_response_contexts = read_saved_factual_response_contexts(memory, trace,
                        input_observation=observation, declaration_reader=declarations,
                        source_hash_reader=readers["source_hash_reader"],
                        input_candidate_keys=memory_input_candidate_keys(
                            prepared, session_id=owner.session_id, speaker_kind=1))
                    from pure_integer_ai.experiments.memory_provenance_inputs import consumed_response_contexts
                    readers["response_contexts"] = consumed_response_contexts(
                        complete_response_contexts, trace, response_role_keys)
                except ValueError as error:
                    raise QueryMemoryOheEvidenceError(str(error)) from error
            if has_reference:
                from pure_integer_ai.experiments.saved_reference_selection_audit import restore_saved_reference_selection
                selected_input, occurrences, relation_map, reference_selection = restore_saved_reference_selection(
                    memory, trace, report, prepared, occurrences, complete_response_contexts,
                    relation_map, reference_bridge=reference_bridge)
                readers['input_role_occurrences'] = occurrences
                from pure_integer_ai.experiments.saved_reference_context_audit import read_saved_reference_contexts
                sets, resolutions = read_saved_reference_contexts(memory, trace, report, prepared,
                    occurrences, complete_response_contexts, relation_map, reference_bridge=reference_bridge)
                readers['reference_contexts'] = complete_response_contexts
                readers['reference_candidate_sets'] = sets
                readers['reference_resolutions'] = resolutions
                from pure_integer_ai.experiments.reference_question_transition_audit import verify_reference_question_transition
                reference_transition = verify_reference_question_transition(trace, selected_input, resolutions,
                                                                            reference_bridge=reference_bridge)
                reference_transition['input_selection'] = reference_selection
            if any(route[0] == 6 for route in trace.memory_routes()):
                if reference_bridge is None:
                    raise QueryMemoryOheEvidenceError('reference read scope lacks its trained operation reader')
                from pure_integer_ai.experiments.memory_reference_read_scope import verify_reference_read_scope
                try:
                    reference_transition['memory_read_scope'] = verify_reference_read_scope(
                        memory, trace, complete_response_contexts, observation,
                        input_structure=reference_bridge.active_memory_input_structure(prepared),
                        question_frames=reference_bridge.relation_question_frames, declaration=lambda ref: declarations(ref)[0])
                except (ValueError, IndexError) as error:
                    raise QueryMemoryOheEvidenceError(str(error)) from error
            if captured is not None:
                ohe = verify_query_memory_ohe(captured, **readers,
                    core_runtime=None if reference_bridge is None else reference_bridge.core_runtime)
            elif value.get('response_plan'):
                ohe = verify_query_memory_ohe_nonfactual(value, report, **readers,
                    input_structure=prepared,
                    core_fact_index=None if reference_bridge is None else reference_bridge.core_fact_index,
                    relation_question_frames=() if reference_bridge is None else reference_bridge.relation_question_frames,
                    assertion_scope_frames=() if reference_bridge is None else reference_bridge.assertion_scope_frames,
                    core_runtime=None if reference_bridge is None else reference_bridge.core_runtime,
                    authorized_input=None if reference_bridge is None else reference_bridge.active_memory_input_structure(prepared))
            else:
                ohe = verify_query_memory_ohe_without_plan(value, report, **readers)
        except QueryMemoryOheEvidenceError as error:
            if not report_incomplete_ohe:
                raise
            ohe = {"status": "UNKNOWN_SAVED_QUERYSTATE_OHE_NOT_FULLY_AUDITED",
                   "failure_reason": str(error), "failure_type": type(error).__name__,
                   "free_dialogue_complete": 0}
        if captured is None:
            return {"memory_ohe": ohe, **request_memory, 'reference_transition':reference_transition}
        proof_reader = lambda key: read_delivery_proof(memory, ObjectIdentity.from_stable_key(key))
        evidence_reader = lambda key: read_factual_evidence_ref(memory, ObjectIdentity.from_stable_key(key))
        return {
            **request_memory,
            'reference_transition':reference_transition,
            "memory_ohe": ohe,
            "delivery_proof": verify_factual_delivery_proof(captured, proof_reader=proof_reader,
                                                           evidence_reader=evidence_reader),
            "delivery_events": verify_factual_delivery_events(captured, proof_reader=proof_reader,
                evidence_reader=evidence_reader, event_reader=lambda key: events.read(key, access=access),
            declaration_reader=declarations, max_memory_refs=max_ohe_declarations),
        }
    finally:
        backend.close()
