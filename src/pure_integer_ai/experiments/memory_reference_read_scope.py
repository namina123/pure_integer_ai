"""显式指代操作只读取获来源角色授权的历史投影，完整候选账本不变。"""
from dataclasses import replace

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.memory_event import (
    INTAKE_DERIVED_HYPOTHESIS, INTAKE_DERIVED_OBSERVATION,
    MEMORY_OBJECT_HYPOTHESIS, HypothesisPayload, MemoryObjectRef,
    MEMORY_EVENT_HYPOTHESIS, ObservationPayload,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    frame_key, memory_candidate_category,
)
from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record


REFERENCE_READ_SCOPE_ROUTE = 6
_SOURCE_LOCAL_CATEGORIES = frozenset((1, 2, 3, 4, 8))
_PROTECTED_CATEGORIES = frozenset((5, 10, 11, 12))


def _reference_hypothesis(ref):
    """H引用已完整编码HypothesisKey；它不是摘要，也不替代实际消费的声明核验。"""
    if ref.object_kind != MEMORY_OBJECT_HYPOTHESIS:
        raise ValueError('reference read scope member is not a Hypothesis')
    hypothesis = HypothesisKey.from_stable_key(ref.object_key)
    if (hypothesis.stable_key() != ref.object_key or ref.owner != hypothesis.observation.owner
            or ref.versions != hypothesis.observation.versions):
        raise ValueError('reference read scope Hypothesis identity differs')
    return hypothesis


def _source_hypotheses(memory, source):
    """回读完整manifest和全部一等H键；无索引命中不能证明来源没有断言或操作。"""
    manifest = memory.intake.require_current_manifest(source)
    observations = tuple(item.object_ref for item in manifest.bindings
                         if item.binding_kind == INTAKE_DERIVED_OBSERVATION)
    refs = tuple(item.object_ref for item in manifest.bindings
                 if item.binding_kind == INTAKE_DERIVED_HYPOTHESIS)
    if manifest.source != source or len(observations) != 1 or not refs or len(set(refs)) != len(refs):
        raise ValueError('reference read scope lacks a complete source manifest')
    found = {}
    for ref in refs:
        hypothesis = _reference_hypothesis(ref)
        if (ref.owner != source.owner or ref.versions != source.versions
                or ref.memory_space != observations[0].memory_space
                or hypothesis.observation != source):
            raise ValueError('reference read scope source Hypothesis differs')
        found[ref.stable_key()] = memory_candidate_category(hypothesis.candidate_key)
    return observations[0].stable_key(), found


def _partition(memory, original, contexts, current, declaration):
    """按完整来源类别和所有context显式依赖分割，不以是否交付选择来源。"""
    current_payload = declaration(current).event.payload
    if not isinstance(current_payload, ObservationPayload):
        raise ValueError('reference read scope lacks its current source')
    current_source = current_payload.source
    required = {ref.stable_key() for context in contexts for ref in context.memory_refs()
                if ref.object_kind == MEMORY_OBJECT_HYPOTHESIS}
    if not required <= set(original):
        raise ValueError('reference read scope omits a context Hypothesis dependency')
    sources, kept, deferred = {}, set(), set()
    for key in original:
        ref = MemoryObjectRef.from_stable_key(key)
        source = _reference_hypothesis(ref).observation
        if (ref.owner != current.owner or ref.versions != current.versions
                or ref.memory_space != current.memory_space
                or source.source_kind != current_source.source_kind
                or source.source_id != current_source.source_id
                or source.document_id > current_source.document_id):
            raise ValueError('reference read scope crosses owner/version/space')
        if source not in sources:
            sources[source] = _source_hypotheses(memory, source)
        observation, categories = sources[source]
        if key not in categories:
            raise ValueError('reference read scope Hypothesis is outside its manifest')
        if (observation == current.stable_key() or key in required
                or categories[key] not in _SOURCE_LOCAL_CATEGORIES
                or _PROTECTED_CATEGORIES.intersection(categories.values())):
            kept.add(key)
        else:
            deferred.add(key)
    return tuple(sorted(kept)), tuple(sorted(deferred)), tuple(sorted(required)), sources


def retain_reference_role_dependencies(memory, structures, contexts, current, scope):
    """保留当前输入、显式context依赖和断言来源；仅延后无操作读取权的旧投影。"""
    if scope is None or not scope.context_keys:
        return structures
    if current is None or scope.observation != current:
        raise ValueError('reference read scope does not belong to the current Observation')
    if not set(scope.context_keys) <= {context.stable_key() for context in contexts}:
        raise ValueError('reference read scope lacks its complete delivered context')

    def declaration(ref):
        """仅当前恢复作用域内读取原唯一声明，禁止候选替代物理对象。"""
        rows = tuple(row for row in memory.intake.event_log.query(
            access=memory.candidate_index.access, object_ref=ref) if row.event.is_declaration)
        if len(rows) != 1:
            raise ValueError('reference read scope lacks one durable declaration')
        return rows[0]

    original = tuple(item.hypothesis_key for item in structures.hypotheses)
    kept, deferred, required, _sources = _partition(memory, original, contexts, current, declaration)
    if not deferred:
        return structures
    proof = frame_key(REFERENCE_READ_SCOPE_ROUTE, (1,), current.stable_key(), scope.stable_key(),
                      frame_key(1, *original), frame_key(2, *kept),
                      frame_key(3, *deferred), frame_key(4, *required))
    keep = set(kept)
    return replace(structures,
        hypotheses=tuple(item for item in structures.hypotheses if item.hypothesis_key in keep),
        evidence=tuple(item for item in structures.evidence if item.hypothesis_key in keep),
        match_routes=tuple(sorted((*structures.match_routes, proof))))


def verify_reference_read_scope(memory, trace, contexts, current, *, input_structure,
                                question_frames, declaration):
    """冷恢复作用域及完整来源，核对分割与实际消费；不重新查询或生成。"""
    proofs = tuple(route for route in trace.memory_routes() if route[0] == REFERENCE_READ_SCOPE_ROUTE)
    if not proofs:
        return {}
    from pure_integer_ai.experiments.current_reference_scope import read_current_reference_scope
    from pure_integer_ai.cognition.understanding.query_reference_context import extended_reference_context_required

    if len(proofs) != 1 or not extended_reference_context_required(input_structure, question_frames):
        raise ValueError('reference read scope requires one actual trained reference operation')
    fields = unpack_record(proofs[0], REFERENCE_READ_SCOPE_ROUTE)
    scope = read_current_reference_scope(memory, current, contexts)
    if (len(fields) != 7 or fields[0] != (1,) or fields[1] != current.stable_key()
            or scope is None or not scope.context_keys or fields[2] != scope.stable_key()):
        raise ValueError('reference read scope differs from the cold physical input scope')
    original, kept, deferred, required = (unpack_record(fields[i], i-2) for i in range(3,7))
    if any(items != tuple(sorted(set(items))) for items in (original, kept, deferred, required)):
        raise ValueError('reference read scope members are not canonical')
    expected = _partition(memory, original, contexts, current, declaration)
    if (kept, deferred, required) != expected[:3] or not deferred:
        raise ValueError('reference read scope omits or changes an authorized dependency')
    sources = expected[3]
    original_set = set(original)
    payload = declaration(current).event.payload
    if not isinstance(payload, ObservationPayload):
        raise ValueError('reference read scope lacks its physical current Observation')
    _observation, categories = _source_hypotheses(memory, payload.source)
    if not {key for key, category in categories.items() if category not in (6,7)} <= original_set:
        raise ValueError('reference read scope omitted a current Hypothesis')
    for route in trace.memory_routes():
        if route[0] == 1:
            _candidate, _shared, key = unpack_record(route, 1)
            hypothesis = _reference_hypothesis(MemoryObjectRef.from_stable_key(key))
            if memory_candidate_category(hypothesis.candidate_key) not in (6,7) and key not in original_set:
                raise ValueError('reference read scope omitted an original content match')
    consumed = {hop.proposition_key for hop in trace.hops()
                if hop.space == 2 and hop.predicate_key == (MEMORY_EVENT_HYPOTHESIS,)}
    if consumed != set(kept):
        raise ValueError('reference read scope differs from the complete consumed Hypotheses')
    return dict(status='MATCHED_PHYSICAL_REFERENCE_ROLE_READ_SCOPE',
                original_hypotheses=len(original),retained_hypotheses=len(kept),
                deferred_hypotheses=len(deferred),context_hypothesis_dependencies=len(required))
