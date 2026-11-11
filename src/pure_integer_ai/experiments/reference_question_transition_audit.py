"""冷审初始开放问式到完整目标采用及最终闭合输入的精确整数转换。"""
from dataclasses import replace

from pure_integer_ai.cognition.understanding.query_input_structure import (
    InputSemanticCandidate, InputSpan, InputSpanRelation, PROJECTION_FILLER, SPAN_SOURCE, STRUCTURE_CLOSED,
)
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError, _unpack
from pure_integer_ai.experiments.saved_reference_context_audit import take_reference_field


def _trace_key(trace, value):
    """恢复唯一 compact 键或完整整数序列，不接受不透明替身。"""
    if isinstance(value, dict) and set(value) == {'integer_key_ref'}:
        return trace.resolve_ref(value['integer_key_ref'])
    if isinstance(value, (list, tuple)) and all(type(item) is int and item >= 0 for item in value):
        return tuple(value)
    raise QueryMemoryOheEvidenceError('question transition requires complete integer keys')


def verify_reference_question_transition(trace, prepared, resolutions, *, reference_bridge):
    """重建目标采用的全部 InputStructure 差分，保持初始语法及事实证据独立。"""
    original = tuple(item.stable_key() for item in prepared.relation_question_bindings)
    members = tuple(trace.resolve_ref(ref) for ref in trace.trace['state_key']['evidence'])
    syntax = tuple(trace.resolve_ref(row['evidence_ref']) for row in trace.trace['evidence'])
    proven = tuple(any(key[-len(binding)-1:] == (len(binding), *binding) for key in syntax) for binding in original)
    if not all(proven):
        if any(proven):
            raise QueryMemoryOheEvidenceError('original question binding lacks its QueryState proof')
        from pure_integer_ai.experiments.unselected_reference_question_audit import verify_unselected_reference_questions
        return verify_unselected_reference_questions(trace, prepared, resolutions,
                                                     reference_bridge=reference_bridge)
    bindings = {trace.resolve_ref(ref) for ref in trace.trace['state_key']['bindings']}
    expected_input = prepared
    adopted = 0
    for key in members:
        source, hypothesis, member, scope, payload, supersedes = _unpack(key, 3, 6)
        if key[0] != 1 or member[:2] != (91599, 7):
            continue
        if len(payload) < 2 or payload[:2] != (91599, 7) or member != (91599, 7, *payload):
            raise QueryMemoryOheEvidenceError('target adoption record or marker is malformed')
        fields, cursor = [], 2
        for _ in range(payload[1]):
            field, cursor = take_reference_field(payload, cursor)
            fields.append(field)
        if cursor != len(payload) or len(fields) != 7:
            raise QueryMemoryOheEvidenceError('target adoption record has trailing or missing fields')
        matching = tuple(item for item in resolutions if item.stable_key() == fields[0])
        if (len(matching) != 1 or matching[0].adoption_support_key != fields[5]
                or matching[0].winner_target_key != fields[6] or matching[0].status != 3):
            raise QueryMemoryOheEvidenceError('target adoption differs from its full reference resolution')
        relation = tuple(item for item in prepared.relation_candidates if item.stable_key() == fields[3])
        opened = tuple(item for item in prepared.open_relations if item.stable_key() == fields[4])
        fact = reference_bridge.core_fact_index.get(fields[1])
        if (len(relation) != 1 or len(opened) != 1 or fact is None
                or fact.predicate.stable_key() != fields[2] or hypothesis != fields[1]
                or key[1:3] != (3, 1) or supersedes):
            raise QueryMemoryOheEvidenceError('target adoption differs from current input or Core identities')
        graph_input = reference_bridge.core_runtime.generation_input(fact.proposition)
        if (scope != graph_input.hypothesis.scope.stable_key()
                or source != (fact.source_hash, *graph_input.proposition.definition.source.stable_key())):
            raise QueryMemoryOheEvidenceError('target adoption source or scope differs from the trained graph')
        relation, opened = relation[0], opened[0]
        reference = matching[0].candidate_set.reference
        dynamic = tuple(item for item in opened.bindings if item.role == reference.role.stable_key()
                        and item.ordinal == reference.ordinal and item.values == reference.values
                        and item.source_ref == prepared.source_ref)
        fact_roles = tuple((ordinal, item) for ordinal, item in enumerate(fact.bindings)
                           if item.role == reference.role and item.filler.stable_key() == fields[6])
        if len(dynamic) != 1 or len(fact_roles) != 1:
            raise QueryMemoryOheEvidenceError('target adoption does not fill its exact dynamic Role')
        if (relation.proposition_key != fields[1] or relation.predicate_key != fields[2]
                or opened.frame.proposition != fields[1] or opened.frame.predicate != fields[2]
                or not any(entry[0] == 1 and entry[1] == 1 and _unpack(entry, 3, 6)[1] == fields[1]
                           for entry in members)):
            raise QueryMemoryOheEvidenceError('target adoption lacks its current positive Core evidence')
        from pure_integer_ai.cognition.shared.query_state import BindingEntry
        if not any(BindingEntry((91599, 6), payload, 1, scope).stable_key() == item for item in bindings):
            raise QueryMemoryOheEvidenceError('target adoption lacks its exact Core binding')
        source_spans = tuple(item for item in prepared.spans if item.start == dynamic[0].start
                             and item.end == dynamic[0].end and item.values == dynamic[0].values
                             and item.source_ref == prepared.source_ref and item.span_kind == SPAN_SOURCE)
        if len(source_spans) > 1:
            raise QueryMemoryOheEvidenceError('target adoption dynamic span is ambiguous')
        span = (source_spans[0] if source_spans else InputSpan(prepared.source_ref,
                dynamic[0].start, dynamic[0].end, SPAN_SOURCE, dynamic[0].values))
        ordinal, fact_role = fact_roles[0]
        projection = InputSemanticCandidate(fields[6], fact_role.filler.object_kind, PROJECTION_FILLER,
            span.ref_key, fields[1], fields[2], reference.role.stable_key(), relation.source_ref, ordinal)
        replacement = replace(relation, span_refs=tuple(sorted({*relation.span_refs, span.ref_key})),
            support_count=relation.required_count, role_coverage=len(fact.bindings), predicate_coverage=1,
            required_open=0, state=STRUCTURE_CLOSED, conflict_count=0)
        spans = tuple(sorted({*prepared.spans, span}, key=lambda item: item.stable_key()))
        precedes, contains = [], []
        for left in spans:
            for right in spans:
                if left == right:
                    continue
                if left.end <= right.start:
                    precedes.append(InputSpanRelation(left.ref_key, right.ref_key))
                if left.start <= right.start and right.end <= left.end and (left.start, left.end) != (right.start, right.end):
                    contains.append(InputSpanRelation(left.ref_key, right.ref_key))
        expected_input = replace(prepared, spans=spans,
            precedes=tuple(sorted(set(precedes), key=lambda item: item.stable_key())),
            contains=tuple(sorted(set(contains), key=lambda item: item.stable_key())),
            semantic_candidates=tuple(sorted({*prepared.semantic_candidates, projection}, key=lambda item: item.stable_key())),
            relation_candidates=tuple(sorted((replacement if item == relation else item
                for item in prepared.relation_candidates), key=lambda item: item.stable_key())),
            relation_question_bindings=tuple(sorted((replace(item, relation_key=replacement.stable_key())
                if item.relation_key == relation.stable_key() else item for item in prepared.relation_question_bindings),
                key=lambda item: item.stable_key())))
        adopted += 1
    if adopted > 1:
        raise QueryMemoryOheEvidenceError('multiple target adoptions are outside the current transition contract')
    final = trace.trace['input_structure']
    if (_trace_key(trace, final['stable_key']) != expected_input.stable_key()
            or tuple(_trace_key(trace, item) for item in final['relation_question_bindings'])
            != tuple(item.stable_key() for item in expected_input.relation_question_bindings)):
        raise QueryMemoryOheEvidenceError('final question/input differs from the exact target adoption transition')
    return {'initial_question_proofs':len(original), 'target_adoption_count':adopted,
            'full_input_transition_matches':1, 'query_call_count':0, 'generation_call_count':0}
