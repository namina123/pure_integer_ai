"""Cold-restore compound generation Evidence with the unchanged atomic bounds."""
from collections.abc import Mapping

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.experiments.trained_query_evidence import dialogue_query_evidence
from pure_integer_ai.experiments.trained_relation_graph_runtime import GraphRelationGeneration


def verify_compound_dialogue_evidence(value, trace, *, core_runtime, key_reader,
                                     rows_reader, max_state_members, error):
    """Read every original member and compare the full canonical Evidence key.

    Only the registered single-claim, role-level generation format is accepted.
    The trained physical frame/source and the original trace are both required;
    an oversized opaque key never acquires an exemption from the atomic bound.
    """
    from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack

    def key(item, label):
        if isinstance(item, Mapping):
            if set(item) != {'integer_key_ref'}:
                raise error('compound Dialogue has an unregistered integer reference')
            item = trace.resolve_ref(item['integer_key_ref'])
        return key_reader(item, label)

    def rows(item, label):
        return rows_reader(item, label, max_state_members)

    generation = trace.trace.get('generation')
    plan = trace.trace.get('response_plan')
    if (core_runtime is None or not isinstance(generation, Mapping)
            or generation.get('kind') != 1 or not isinstance(plan, Mapping)
            or not isinstance(value, (list, tuple)) or tuple(value[:3]) != (3, 1, 1)):
        raise error('compound Dialogue requires the physical single-claim generation graph')
    source, claim, member, scope, payload, supersedes = _unpack(tuple(value), 3, 6)
    for part, label in ((source, 'source'), (claim, 'claim'), (member, 'member'), (scope, 'scope')):
        key(part, 'compound Dialogue ' + label)
    if supersedes:
        raise error('compound Dialogue cannot add a supersedes link')
    if tuple(key(item, 'compound Dialogue Plan claim') for item in rows(
            plan.get('claim_refs'), 'compound Dialogue Plan claims')) != (claim,):
        raise error('compound Dialogue claim differs from the current Plan')
    frame_key = key(generation.get('frame_proposition'), 'compound Dialogue frame')
    frames = tuple(frame for frame in core_runtime.active_surface_frames()
                   if frame.proposition.stable_key() == frame_key)
    if len(frames) != 1:
        raise error('compound Dialogue lacks one physical trained frame')
    frame, = frames
    original_trace = generation.get('trace')
    if isinstance(original_trace, Mapping):
        if set(original_trace) != {'integer_key_ref'}:
            raise error('compound Dialogue trace has an unregistered reference')
        original_trace = trace.resolve_ref(original_trace['integer_key_ref'])
    if (not isinstance(original_trace, (list, tuple)) or len(original_trace) < 2
            or type(original_trace[0]) is not int or original_trace[0] != 2
            or type(original_trace[1]) is not int):
        raise error('compound Dialogue has an unregistered generation trace')
    representations = tuple(ObjectIdentity.from_stable_key(key(item, 'compound Representation'))
        for item in rows(generation.get('representations'), 'compound Representations'))
    count = original_trace[1]
    if not representations or count != len(representations) + 2 or count > max_state_members:
        raise error('compound Dialogue trace member count differs or exceeds budget')
    cursor = 2
    for ordinal in range(count):
        if cursor >= len(original_trace) or type(original_trace[cursor]) is not int:
            raise error('compound Dialogue generation member is truncated')
        size = original_trace[cursor]
        cursor += 1
        if size <= 0 or size > len(original_trace) - cursor:
            raise error('compound Dialogue generation member length differs')
        key(original_trace[cursor:cursor + size], 'compound Dialogue generation atom')
        cursor += size
    if cursor != len(original_trace):
        raise error('compound Dialogue generation has trailing members')
    slot_count = generation.get('slot_count')
    if type(slot_count) is not int or not 0 < slot_count <= max_state_members:
        raise error('compound Dialogue slot count exceeds the original member budget')
    connector = ObjectIdentity.from_stable_key(key(generation.get('connector'), 'compound connector'))
    units = key(generation.get('surface_values'), 'compound generated units')
    if any(unit > 0x10ffff or 0xd800 <= unit <= 0xdfff for unit in units):
        raise error('compound Dialogue surface contains invalid Unicode')
    generated = GraphRelationGeneration(''.join(map(chr, units)), frame.proposition,
        frame.source_hash, slot_count, connector, representations, tuple(original_trace))
    expected = dialogue_query_evidence(ObjectIdentity.from_stable_key(claim), frame,
        core_runtime.generation_input(frame.proposition), generated).stable_key()
    if tuple(value) != expected:
        raise error('compound Dialogue differs from the complete physical frame and original generation')
    return expected
