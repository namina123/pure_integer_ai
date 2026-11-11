"""独立 request 的冷输入、当前查询根和 source-local Memory 证据合同。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION, MEMORY_CATEGORY_OPERATION_REQUEST, MEMORY_TOPIC_PROOF_VERSION,
    _parts, frame_key, memory_operation_request_binding,
)
from pure_integer_ai.cognition.understanding.query_operation_requests import attach_operation_request
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.pending_composition_call_audit import _decode_input
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack
from pure_integer_ai.experiments.trained_input_discourse_operation_graph import MODE_OPERATION_GENERAL


def restore_saved_operation_request_projection(value, report, projector, frames, *, input_preparer=None):
    """从原输入冷恢复完整训练图投影，可复用生产 prepare；不调用 query 或生成。"""
    trace = GraphQueryTraceEvidence(value)
    values = tuple(report["input_values"])
    projected = projector.project(values) if input_preparer is None else input_preparer(values)
    prepared = attach_operation_request(frames, projected)
    decoded = _decode_input(value["input_structure"], trace)
    if decoded != prepared.integer_trace():
        from pure_integer_ai.cognition.understanding.query_operation_requests import operation_request_input_view
        prepared = operation_request_input_view(prepared)
    if (decoded != prepared.integer_trace()
            or trace.resolve_ref(value["query_ref"]) != prepared.source_ref
            or trace.operation_contracts() != tuple((key, MODE_OPERATION_GENERAL) for key in prepared.operation_keys)):
        raise ValueError("request saved input differs from its complete cold graph projection")
    if prepared.operation_request_bindings:
        request, = prepared.operation_request_bindings
        operation, key, query = request.operation_key, request.stable_key(), prepared.source_ref
        proof = (91672, 5, len(operation), *operation,
                 len(MODE_OPERATION_GENERAL), *MODE_OPERATION_GENERAL, 92322, len(key), *key)
        expected = EvidenceEntry(query, operation, space=3, polarity=3, trust=1,
                                 evidence_key=proof, scope_key=query, payload_key=proof)
        evidence = []
        for ref in value["state_key"]["evidence"]:
            row = trace.resolve_ref(ref)
            fields = _unpack(row, 3, 6)
            evidence.append(EvidenceEntry(fields[0], fields[1], row[0], row[1], row[2], *fields[2:]))
        if (expected not in evidence or BindingEntry((91672, 4), operation, space=3, scope_key=query).stable_key()
                not in tuple(trace.resolve_ref(ref) for ref in value["state_key"]["bindings"])):
            raise ValueError("request full variable proof was not consumed by the same QueryState")
    return prepared


def audit_operation_request_roots(value, prepared):
    """Report a saved scope gap explicitly; projection alone cannot admit roots."""
    trace = GraphQueryTraceEvidence(value)
    query = prepared.source_ref
    if query != trace.resolve_ref(value["state_key"]["query_ref"]):
        raise ValueError("request saved state has a foreign query scope")
    expanded, current = set(), set()
    for ref in value["state_key"]["roots"]:
        row = trace.resolve_ref(ref)
        fields = _unpack(row, 3, 3)
        if row[0] == 1 and row[1] == 2:
            expanded.add(fields[0])
            if fields[1:] == (query, query):
                current.add(fields[0])
    readings = tuple(reading[1] for request in prepared.operation_request_bindings for reading in request.readings)
    missing = [index + 1 for index, key in enumerate(readings) if key not in current]
    return {"request_core_root_status": "UNKNOWN_CURRENT_SCOPE_ROOTS_MISSING" if missing else "PASS",
            "request_reader_count": len(readings), "expanded_reader_count": sum(key in expanded for key in readings),
            "current_scoped_reader_count": sum(key in current for key in readings),
            "missing_current_reader_ordinals": missing}


def restore_saved_operation_request(value, report, projector, frames):
    prepared = restore_saved_operation_request_projection(value, report, projector, frames)
    if audit_operation_request_roots(value, prepared)["request_core_root_status"] != "PASS":
        raise ValueError("request variables lack actual expanded current Core roots")
    return prepared


def verify_operation_request_memory(prepared, value, report, declaration_reader):
    """请求完整 binding 必须有当前 O/H/E，UNKNOWN 证据不可升级为事实 support。"""
    trace = GraphQueryTraceEvidence(value)
    source = SourceRef.from_stable_key(tuple(report["input_source"]))
    requests = set(prepared.operation_request_bindings)
    selected = {}
    for key in report["hypothesis_refs"]:
        ref = MemoryObjectRef.from_stable_key(tuple(key))
        rows = declaration_reader(ref)
        if len(rows) != 1:
            raise ValueError("request Memory hypothesis declaration is not unique")
        hypothesis = rows[0].event.payload.hypothesis
        request = memory_operation_request_binding(hypothesis.candidate_key)
        if request is not None:
            if request not in requests or hypothesis.observation != source or request in selected:
                raise ValueError("request Memory hypothesis crosses the current binding/source")
            selected[request] = (ref, hypothesis.candidate_key)
    if set(selected) != requests:
        raise ValueError("request complete variables are missing from current Memory O/H/E")
    evidence = {trace.resolve_ref(ref) for ref in value["state_key"]["evidence"]}
    for request, (hypothesis, candidate) in selected.items():
        count = 0
        for key in report["evidence_refs"]:
            ref = MemoryObjectRef.from_stable_key(tuple(key))
            rows = declaration_reader(ref)
            if len(rows) != 1:
                raise ValueError("request Memory Evidence declaration is not unique")
            payload = rows[0].event.payload
            if payload.hypothesis_ref != hypothesis:
                continue
            detail = _parts(payload.detail, 91507)
            expected = frame_key(MEMORY_TOPIC_PROOF_VERSION, prepared.source_ref, request.stable_key())
            if (payload.source != source or payload.stance != 3 or payload.supersedes_ref is not None
                    or len(detail) != 3 or detail[0][-1:] != (MEMORY_CATEGORY_OPERATION_REQUEST,)
                    or detail[1:] != (candidate, expected)
                    or candidate != frame_key(MEMORY_CANDIDATE_VERSION, (MEMORY_CATEGORY_OPERATION_REQUEST,), request.stable_key())):
                raise ValueError("request Memory Evidence lost its complete variables/projection/source proof")
            if not any(_unpack(row, 3, 6)[4] == payload.stable_key() and row[0] == 2 and row[1] == 3
                       for row in evidence):
                raise ValueError("request Memory Evidence was not consumed in the original QueryState")
            count += 1
        if count != 1:
            raise ValueError("request lacks one exact current intake Evidence")
    return {"memory_request_binding_count": len(selected),
            "memory_request_binding_status": "MATCHED_CURRENT_SOURCE_LOCAL_FULL_BINDING",
            "request_memory_is_fact_support": 0}
