"""更正立场的纯资格逻辑；只消费实际挂载并完整重匹配的训练作用域。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_REFUTE, EVIDENCE_SUPPORT
from pure_integer_ai.cognition.understanding.query_assertion_scope import (
    InputAssertionScopeBinding, SCOPE_BINDING_VERSION, SCOPE_NEGATED,
    TrainedAssertionScopeFrame, match_assertion_scope_frames,
)
from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record


def read_revision_scope(key, *, relation_key, source_ref):
    """无损恢复否定证明并核对当前关系、来源和规范记录；缺项一律拒绝。"""
    fields = unpack_record(key, SCOPE_BINDING_VERSION)
    if len(fields) < 4:
        raise ValueError("revision scope proof is incomplete")
    binding = InputAssertionScopeBinding(*fields[:3], tuple(fields[3:]))
    if (binding.stable_key() != key or binding.mode != SCOPE_NEGATED
            or binding.relation_key != relation_key or binding.source_ref != source_ref):
        raise ValueError("revision scope proof differs from the current negated relation")
    return binding


def revision_scope_stance(frames, structure):
    """独立重匹配完整输入；原无作用域支持保持，否定须有唯一物理证明。"""
    matched = match_assertion_scope_frames(frames, structure)
    attached = structure.assertion_scope_bindings
    if not attached and not matched:
        # 总匹配器对竞争返回空；不能把这种空误当成旧无作用域支持。
        if any(match_assertion_scope_frames((frame,), structure) for frame in frames):
            raise ValueError("revision has competing mounted scope interpretations")
        return EVIDENCE_SUPPORT, ()
    if len(matched) != 1 or attached != matched or matched[0].mode != SCOPE_NEGATED:
        raise ValueError("revision requires one mounted complete negated scope")
    return EVIDENCE_REFUTE, matched[0].stable_key()


def require_mounted_revision_scope(frames, key, *, relation_key, source_ref):
    """提交前重查完整训练成员身份，不能由外部预装证明或同形框架授权。"""
    binding = read_revision_scope(key, relation_key=relation_key, source_ref=source_ref)
    frame = TrainedAssertionScopeFrame.from_stable_key(binding.frame_key)
    if frame not in frames:
        raise ValueError("revision scope proof is outside the mounted model")
    return binding
