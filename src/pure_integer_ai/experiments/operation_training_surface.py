"""在训练期独立来源上核验无正文模型的完整Span语言成员，不回填来源句。"""
from __future__ import annotations


def verify_operation_training_surface(record, fact, frame, source_hash):
    """逐段核对完整命题、来源、角色和载体；模型不需要保留或重构来源正文。"""
    values = record.surface_values
    if (source_hash is None or source_hash != fact.source_hash
            or fact.proposition.stable_key() != record.proposition_key
            or fact.predicate.stable_key() != record.predicate_key
            or frame.proposition != fact.proposition or frame.predicate != fact.predicate
            or frame.source_hash != source_hash
            or (fact.evidence_surface and tuple(map(ord, fact.evidence_surface)) != values)):
        raise ValueError('native composition source identity differs from its train teacher')
    endpoints = {tuple(row['endpoint_key']): row for item in record.endpoints for row in (item.to_value(),)}
    expected = []
    for item in record.role_bindings:
        row = item.to_value()
        endpoint = endpoints[tuple(row['filler_key'])]
        start, end = endpoint['start'], endpoint['end']
        expected.append((tuple(row['role_key']), tuple(row['filler_key']), start, end, values[start:end]))
    expected.sort(key=lambda row: (row[2], row[3], row[0]))
    actual = tuple((binding.role.stable_key(), binding.filler.stable_key(), binding.start, binding.end,
                    tuple(map(ord, binding.surface))) for binding in fact.bindings)
    if (actual != tuple(expected) or any(binding.source_hash != source_hash for binding in fact.bindings)
            or frame.roles != tuple(binding.role for binding in fact.bindings)
            or (fact.cue_start, fact.cue_end) != record.anchor_span[:2]
            or tuple(map(ord, fact.cue)) != values[fact.cue_start:fact.cue_end]):
        raise ValueError('native composition role/cue spans differ from its train teacher')
    start = min(fact.cue_start, *(binding.start for binding in fact.bindings))
    end = max(fact.cue_end, *(binding.end for binding in fact.bindings))
    cursor, gaps = start, []
    for binding in fact.bindings:
        if binding.start < cursor:
            raise ValueError('native composition train spans overlap')
        gaps.append(values[cursor:binding.start])
        cursor = binding.end
    gaps.append(values[cursor:end])
    if ((frame.envelope_start, frame.envelope_end) != (start, end)
            or tuple(tuple(map(ord, gap)) for gap in frame.gaps) != tuple(gaps)):
        raise ValueError('native composition frame intervals differ from its train teacher')
    if frame.carrier_prefix is None and frame.carrier_suffix is None:
        if not fact.evidence_surface:
            raise ValueError('body-free composition teacher requires complete learned carrier members')
    elif (type(frame.carrier_prefix) is not str or type(frame.carrier_suffix) is not str
            or tuple(map(ord, frame.carrier_prefix)) != values[:start]
            or tuple(map(ord, frame.carrier_suffix)) != values[end:]):
        raise ValueError('native composition carrier differs from its train teacher')
