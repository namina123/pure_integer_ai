"""完整当前输入的角色竞争合同；保留原绑定，只授权无事实澄清。"""
from pure_integer_ai.cognition.shared.query_state import (
    BindingEntry, EvidenceEntry, ROOT_EMPTY, SPACE_CORE,
)
from pure_integer_ai.cognition.understanding.query_input_structure import PROJECTION_FILLER, STRUCTURE_ORDER_CONFLICT
from pure_integer_ai.cognition.understanding.query_memory_candidates import frame_key


def conflicting_input_filler_groups(semantic_fillers):
    """按完整命题、来源和角色分组，不合并同名或不同身份端点。"""
    groups = {}
    for item in semantic_fillers:
        key = (item.proposition_key, item.source_ref, item.role_key)
        groups.setdefault(key, set()).add(item.candidate_key)
    return frozenset(key for key, values in groups.items() if len(values) > 1)


def projection_conflict_keys_match(structure, query_key, binding_keys, evidence_keys):
    """独立重建全部保留竞争及Core投影证据；任何增删漂移均拒绝。"""
    if (structure is None or structure.source_ref != query_key
            or structure.relation_question_bindings or structure.operation_keys
            or structure.operation_request_bindings or structure.assertion_scope_bindings
            or structure.discourse_compositions or structure.discourse_gaps
            or structure.pending_relation_composition_bindings):
        return False
    fillers = tuple(item for item in structure.semantic_candidates
                    if item.projection_kind == PROJECTION_FILLER)
    groups = conflicting_input_filler_groups(fillers)
    expected_conflicts = {
        BindingEntry(item.role_key, item.candidate_key, space=SPACE_CORE,
                     scope_key=query_key, conflict_kept=1).stable_key()
        for item in fillers
        if (item.proposition_key, item.source_ref, item.role_key) in groups}
    expected_conflicts.update(
        BindingEntry(item.predicate_key, item.proposition_key, space=SPACE_CORE,
                     scope_key=query_key, conflict_kept=1).stable_key()
        for item in structure.relation_candidates if item.state == STRUCTURE_ORDER_CONFLICT)
    if not expected_conflicts or expected_conflicts != {key for key in binding_keys if key[-1]}:
        return False
    expected_evidence = set()
    for kind, members in ((1, fillers), (2, structure.relation_candidates)):
        for item in members:
            key = item.stable_key()
            expected_evidence.add(EvidenceEntry(item.source_ref, item.proposition_key,
                space=SPACE_CORE, polarity=3, trust=1,
                evidence_key=frame_key(91569, (kind,), key),
                scope_key=query_key, payload_key=key).stable_key())
    return expected_evidence == {key for key in evidence_keys if key[0] == SPACE_CORE}


def retained_projection_conflict_allowed(state, structure, *, restricted=False):
    """原绑定和证据不变；生成阶段额外要求既有受限澄清的唯一Core空根。"""
    if not projection_conflict_keys_match(structure, state.query_key,
            tuple(item.stable_key() for item in state.bindings),
            tuple(item.stable_key() for item in state.evidence)):
        return False
    if not restricted:
        return True
    core = tuple(root for root in state.roots if root.owner_space == SPACE_CORE)
    return (len(core) == 1 and core[0].status == ROOT_EMPTY
            and core[0].root_key == (92371, 0)
            and state.required_slots_open > 0 and not state.evidence_closed)
