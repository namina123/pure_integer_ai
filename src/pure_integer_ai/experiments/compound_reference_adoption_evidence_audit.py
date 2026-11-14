"""按注册的七字段协议冷读复合Core目标采用证据，保持每个原子上限。"""
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.query_state import BindingEntry


def verify_compound_reference_adoption_evidence(value, trace, *, transition, resolutions,
        core_runtime, key_reader, rows_reader, max_state_members, error):
    """逐项核对完整转换见证、独立决策、物理来源和当前Binding；不执行生产者。"""
    from pure_integer_ai.experiments.reference_question_transition_audit import (
        ReferenceQuestionTransitionAudit, _trace_key,
    )

    def fields(record, prefix, count):
        """读取固定数量的非负严格整数长度帧，拒绝截断和尾随内容。"""
        cursor, parts = prefix, []
        for _ in range(count):
            if (cursor >= len(record) or type(record[cursor]) is not int
                    or record[cursor] < 0 or record[cursor] > len(record) - cursor - 1):
                raise error('compound Core adoption has invalid or truncated framing')
            size = record[cursor]
            cursor += 1
            parts.append(record[cursor:cursor + size])
            cursor += size
        if cursor != len(record):
            raise error('compound Core adoption has trailing fields')
        return tuple(parts)

    if (not isinstance(value, (tuple, list)) or tuple(value[:3]) != (1, 3, 1)
            or any(type(item) is not int for item in value[:3])
            or core_runtime is None or not isinstance(transition, ReferenceQuestionTransitionAudit)):
        raise error('compound Core adoption requires its cold transition and physical graph')
    value = tuple(value)
    source, claim, member, scope, payload, supersedes = fields(value, 3, 6)
    for part, label in ((source, 'source'), (claim, 'claim'), (scope, 'scope')):
        key_reader(part, 'compound Core adoption ' + label)
    if (supersedes or payload[:2] != (91599, 7)
            or any(type(item) is not int for item in payload[:2])
            or member != (91599, 7, *payload) or 7 > max_state_members):
        raise error('compound Core adoption has an unregistered marker or member budget')
    parts = fields(payload, 2, 7)
    for part in parts:
        key_reader(part, 'compound Core adoption atom')
    if (type(resolutions) is not tuple or len(resolutions) > max_state_members):
        raise error('compound Core adoption resolution budget differs')
    matching = tuple(item for item in resolutions if item.stable_key() == parts[0])
    if (len(matching) != 1 or matching[0].status != 3
            or matching[0].adoption_support_key != parts[5]
            or matching[0].winner_target_key != parts[6] or claim != parts[1]):
        raise error('compound Core adoption differs from its complete cold resolution')
    if (transition.report().get('target_adoption_count') != 1
            or transition.adoption_evidence_keys != (value,)
            or _trace_key(trace, trace.trace['input_structure']['stable_key']) != transition.final_input_key):
        raise error('compound Core adoption differs from its complete current input transition')
    physical = core_runtime.generation_input(ObjectIdentity.from_stable_key(claim))
    frames = tuple(frame for frame in core_runtime.active_surface_frames()
                   if frame.proposition.stable_key() == claim)
    if (len(frames) != 1 or physical.proposition.definition.proposition.stable_key() != claim
            or source != (frames[0].source_hash, *physical.proposition.definition.source.stable_key())
            or scope != physical.hypothesis.scope.stable_key()):
        raise error('compound Core adoption differs from the physical source or scope')
    expected = BindingEntry((91599, 6), payload, 1, scope).stable_key()
    bindings = rows_reader(trace.trace['state_key'].get('bindings'),
                           'compound Core adoption bindings', max_state_members)
    if sum(trace.resolve_ref(ref) == expected for ref in bindings) != 1:
        raise error('compound Core adoption lacks its unique current binding')
    return value
