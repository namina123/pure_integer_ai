"""把完整作用域证明接入共享查询和非事实回应，不修改证据真值。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry, SPACE_DIALOGUE
from pure_integer_ai.cognition.understanding.query_assertion_scope import (
    SCOPE_NEGATED, SCOPE_POSSIBLE, SCOPE_QUOTED, SCOPE_QUERY_PROOF,
)
from pure_integer_ai.experiments.unknown_response_course import (
    RESPONSE_FEATURE_ASSERTION_NEGATED, RESPONSE_FEATURE_ASSERTION_POSSIBLE,
    RESPONSE_FEATURE_ASSERTION_QUOTED,
)


def assertion_scope_feature(structure) -> int:
    """只有一个完整训练绑定才贡献算子特征；不读取表面文本。"""
    bindings = getattr(structure, "assertion_scope_bindings", ())
    if len(bindings) != 1:
        return 0
    return {SCOPE_NEGATED: RESPONSE_FEATURE_ASSERTION_NEGATED,
            SCOPE_POSSIBLE: RESPONSE_FEATURE_ASSERTION_POSSIBLE,
            SCOPE_QUOTED: RESPONSE_FEATURE_ASSERTION_QUOTED}[bindings[0].mode]


def assertion_scope_query_records(structure, query_key):
    """在原 QueryState 登记完整当前绑定，语法资格只形成 UNKNOWN 证据。"""
    if len(structure.assertion_scope_bindings) != 1:
        return ()
    scope = structure.assertion_scope_bindings[0]
    proof = scope.stable_key()
    return ((BindingEntry((SCOPE_QUERY_PROOF,), proof, space=SPACE_DIALOGUE, scope_key=query_key),
             EvidenceEntry(scope.source_ref, proof, space=SPACE_DIALOGUE, polarity=3, trust=1,
                 evidence_key=proof, payload_key=scope.frame_key, scope_key=query_key)),)


def assertion_scope_response_allowed(structure, state) -> bool:
    """仅允许当前完整作用域的非事实生成；结构冲突和丢失证明仍关闭。"""
    records = assertion_scope_query_records(structure, state.query_key)
    return bool(records and not any(item.conflict_kept for item in state.bindings)
                and all(binding in state.bindings and evidence in state.evidence
                        for binding, evidence in records))
