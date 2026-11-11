"""从独立冷图恢复完整指代竞争和决策，禁止用当前正确答案替代历史成员。"""
from types import SimpleNamespace

from pure_integer_ai.cognition.shared.hypothesis_resolution import ResolverDecision
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.query_state import EvidenceEntry
from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError, _unpack
from pure_integer_ai.experiments.trained_query_evidence import core_query_evidence, current_query_evidence
from pure_integer_ai.experiments.trained_reference_resolution import TrainedCrossTurnReferenceRuntime
from pure_integer_ai.experiments.trained_response_occurrence import ReferenceCandidateSet


def take_reference_field(key, cursor):
    """严格读取一个长度前缀整数子记录，拒绝负数、越界和截断。"""
    if (type(key) is not tuple or any(type(value) is not int or value < 0 for value in key)
            or cursor >= len(key) or key[cursor] > len(key) - cursor - 1):
        raise QueryMemoryOheEvidenceError('reference packed field is invalid or truncated')
    end = cursor + 1 + key[cursor]
    return key[cursor + 1:end], end


def restore_reference_candidate_set(key, inputs, roles, contexts):
    """只允许来自当前输入发生及已恢复历史 occurrence 的精确成员和深度。"""
    fields = []
    for field in unpack_record(key, 91534):
        if len(field) % 2 or any(field[index] not in (0, 1) for index in range(0, len(field), 2)):
            raise QueryMemoryOheEvidenceError('reference signed field is invalid')
        fields.append(tuple(value if sign == 0 else -value - 1
            for sign, value in zip(field[::2], field[1::2], strict=True)))
    if len(fields) < 3 or fields[0] != (8,) or fields[1] not in inputs:
        raise QueryMemoryOheEvidenceError('reference candidate set lacks its current input role')
    antecedents, depths = [], []
    for field in fields[2:]:
        parts = unpack_record(field, 91534)
        if len(parts) != 2 or len(parts[1]) != 1 or parts[0] not in roles:
            raise QueryMemoryOheEvidenceError('reference antecedent is absent from the cold graph')
        antecedent, depth = roles[parts[0]], parts[1][0]
        if not any(antecedent in context.role_occurrences and context.context_depth == depth
                   for context in contexts):
            raise QueryMemoryOheEvidenceError('reference context depth differs from the cold graph')
        antecedents.append(antecedent)
        depths.append(depth)
    restored = ReferenceCandidateSet(inputs[fields[1]], tuple(antecedents), tuple(depths))
    if restored.stable_key() != key:
        raise QueryMemoryOheEvidenceError('reference candidate set does not round trip')
    complete = tuple(roles[role] for role in sorted(roles)
                     if roles[role].source_ref != restored.reference.source_ref)
    complete_depths = tuple(min(context.context_depth for context in contexts
                               if role in context.role_occurrences) for role in complete)
    if restored != ReferenceCandidateSet(restored.reference, complete, complete_depths):
        raise QueryMemoryOheEvidenceError('reference candidate set omits a cold antecedent or changes its depth')
    return restored


def reference_resolution_header(key):
    """恢复全部变长字段和 H-04 逻辑时间，未知协议及尾随数据失败关闭。"""
    if not key or key[0] != 91536:
        raise QueryMemoryOheEvidenceError('unregistered reference resolution version')
    candidate, cursor = take_reference_field(key, 1)
    if len(key) - cursor < 3:
        raise QueryMemoryOheEvidenceError('reference resolution header is truncated')
    required, status, count = key[cursor:cursor + 3]
    cursor += 3
    if count > 128:
        raise QueryMemoryOheEvidenceError('reference resolution exceeds candidate audit budget')
    for _ in range(count):
        _, cursor = take_reference_field(key, cursor)
    if cursor >= len(key) or key[cursor] > 4096:
        raise QueryMemoryOheEvidenceError('reference resolution exceeds evidence audit budget')
    evidence_count = key[cursor]
    cursor += 1
    for _ in range(evidence_count):
        _, cursor = take_reference_field(key, cursor)
    decision_key, cursor = take_reference_field(key, cursor)
    decision = ResolverDecision.from_stable_key(decision_key)
    _, cursor = take_reference_field(key, cursor)
    _, cursor = take_reference_field(key, cursor)
    if cursor < len(key):
        if key[cursor] != 91655:
            raise QueryMemoryOheEvidenceError('unregistered reference target support version')
        _, cursor = take_reference_field(key, cursor + 1)
    if cursor != len(key) or decision.stable_key() != decision_key:
        raise QueryMemoryOheEvidenceError('reference resolution has trailing or noncanonical data')
    return candidate, required, status, decision.timestamp_seq


def read_saved_reference_contexts(memory, trace, report, prepared, input_roles, contexts,
                                 relation_map, *, reference_bridge):
    """从当前完整输入、只读历史图和已消费 Core 证据重建全部保存决策；不运行查询。"""
    sets_refs = trace.trace.get('reference_candidate_set_refs', ())
    resolution_refs = trace.trace.get('reference_resolution_refs', ())
    if any(not isinstance(refs, (tuple, list)) or len(refs) > 64 for refs in (sets_refs, resolution_refs)):
        raise QueryMemoryOheEvidenceError('saved reference records exceed cold audit budget')
    if not sets_refs and not resolution_refs:
        return (), ()
    if reference_bridge is None:
        raise QueryMemoryOheEvidenceError('reference cold audit requires the original trained graph')
    projection = report.get('projection')
    if (not isinstance(projection, dict)
            or tuple(tuple(key) for key in projection.get('question_bindings', ()))
            != tuple(item.stable_key() for item in prepared.relation_question_bindings)):
        raise QueryMemoryOheEvidenceError('original reference question bindings differ from cold input')
    inputs = {role.stable_key(): role for role in input_roles}
    roles = {role.stable_key(): role for context in contexts for role in context.role_occurrences}
    sets = tuple(restore_reference_candidate_set(trace.resolve_ref(ref), inputs, roles, contexts)
                 for ref in sets_refs)
    if len({item.stable_key() for item in sets}) != len(sets):
        raise QueryMemoryOheEvidenceError('saved reference candidate set is duplicated')
    evidence = []
    for ref in trace.trace['state_key']['evidence']:
        key = trace.resolve_ref(ref)
        source, hypothesis, member, scope, payload, supersedes = _unpack(key, 3, 6)
        evidence.append(EvidenceEntry(source, hypothesis, *key[:3], member, scope, payload, supersedes))
    core = reference_bridge.core_runtime
    registry = core.context.scoped_identity_store.registry
    for proposition in {item.hypothesis_key for item in evidence
                        if item.space == 1 and item.evidence_key[:1] == (1,)}:
        expected = core_query_evidence(ObjectIdentity.from_stable_key(proposition),
                                       core.evidence_history(ObjectIdentity.from_stable_key(proposition)), registry)
        if {item for item in evidence if item.space == 1 and item.hypothesis_key == proposition
                and item.evidence_key[:1] == (1,)} != set(expected):
            raise QueryMemoryOheEvidenceError('reference Core Evidence differs from the original trained graph')
    core_support = frozenset(item.hypothesis_key for item in current_query_evidence(tuple(evidence))
                             if item.space == 1 and item.polarity == 1)
    resolver = TrainedCrossTurnReferenceRuntime(core, reference_bridge.input_projector, memory.occurrence_index)
    from pure_integer_ai.experiments.current_reference_scope import read_current_reference_scope
    reference_scope = read_current_reference_scope(memory,
        MemoryObjectRef.from_stable_key(tuple(report['observation_ref'])), contexts)
    owner = SimpleNamespace(_input_open_relations_by_hypothesis=relation_map)
    by_key = {item.stable_key(): item for item in sets}
    result = []
    for ref in resolution_refs:
        key = trace.resolve_ref(ref)
        candidate_key, required, status, timestamp = reference_resolution_header(key)
        if candidate_key not in by_key:
            raise QueryMemoryOheEvidenceError('resolution lacks its complete cold candidate set')
        candidate = by_key[candidate_key]
        source = SourceRef.from_stable_key(candidate.reference.source_ref[1:])
        if timestamp != source.document_id:
            raise QueryMemoryOheEvidenceError('reference decision logical sequence differs from its input')
        relation_frame, fillers = type(reference_bridge)._reference_relation_context(owner, candidate, prepared)
        objects = tuple(sorted({filler for _, filler in fillers}, key=lambda item: item.stable_key()))
        answer_role = ()
        binding = next((item for item in prepared.relation_question_bindings if item.answer_role_key), None)
        if binding is not None:
            parts = reference_bridge._relation_slot_question_parts(prepared, binding)
            if parts is not None and candidate.reference.role.stable_key() in parts[4] and not parts[0].is_reference_slot:
                answer_role = binding.answer_role_key
        restored = resolver.resolve(candidate, contexts, core_support=core_support, timestamp_seq=timestamp,
            current_objects=objects, relation_frame=relation_frame, current_role_fillers=fillers,
            relation_answer_role=answer_role, reference_scope=reference_scope)
        if restored.stable_key() != key or (restored.requires_resolution, restored.status) != (required, status):
            raise QueryMemoryOheEvidenceError('cold reference resolution differs from the full original decision')
        result.append(restored)
    if len({item.stable_key() for item in result}) != len(result):
        raise QueryMemoryOheEvidenceError('saved reference resolution is duplicated')
    return sets, tuple(result)
