"""未落地请求的同次 QueryState 证明及仅 UNKNOWN 的生成资格。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record
from pure_integer_ai.cognition.understanding.query_operation_request_scope import (
    PENDING_REQUEST_SCOPE, match_pending_operation_request_scopes,
)
from pure_integer_ai.cognition.understanding.query_operation_requests import OperationRequestFrame


def pending_request_query_records(frames, structure):
    """只把完整当前语法作为 UNKNOWN 证明，保留其真实训练来源和查询作用域。"""
    records = []
    for scope in match_pending_operation_request_scopes(frames, structure):
        frame = OperationRequestFrame.from_stable_key(scope.frame_key)
        key = scope.stable_key()
        records.append((
            BindingEntry((PENDING_REQUEST_SCOPE, 1), key, space=3, scope_key=structure.source_ref),
            EvidenceEntry(frame.source.stable_key(), frame.operation_key, space=3, polarity=3,
                evidence_key=pack_record(PENDING_REQUEST_SCOPE, structure.source_ref, key),
                scope_key=structure.source_ref, payload_key=key)))
    return tuple(records)


def pending_request_context_allowed(state, structure, frames):
    """只有当前完整记录实际在 state 中且原事实/冲突/预算门通过，才允许无对象 UNKNOWN。"""
    if not any(item.role_key == (PENDING_REQUEST_SCOPE, 1) for item in state.bindings):
        return False
    from pure_integer_ai.experiments.trained_input_clarification import unresolved_role_context_allowed
    if state.query_key != structure.source_ref or not unresolved_role_context_allowed(state, structure):
        return False
    records = pending_request_query_records(frames, structure)
    expected_bindings = {binding for binding, _evidence in records}
    actual_bindings = {item for item in state.bindings if item.role_key == (PENDING_REQUEST_SCOPE, 1)}
    actual_evidence = {item for item in state.evidence
                       if item.payload_key[:1] == (PENDING_REQUEST_SCOPE,)}
    return bool(records and actual_bindings == expected_bindings
                and actual_evidence == {evidence for _binding, evidence in records})
