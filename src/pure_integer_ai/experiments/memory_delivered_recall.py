"""把当前内容命中的历史输入连接到它自己的实际交付，不读取历史表面。"""
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, OBJECT_CONTEXT_SCOPE
from pure_integer_ai.cognition.shared.memory_event import (
    EpisodePayload, MemoryObjectRef, ObservationPayload, MEMORY_OBJECT_OBSERVATION,
)
from pure_integer_ai.cognition.shared.memory_record_identity import observation_reference_source
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record
from pure_integer_ai.experiments.memory_context_lineage import delivered_context_links
from pure_integer_ai.experiments.trained_memory_candidate_index import episode_input_route


DELIVERED_RECALL_ROUTE = 5


def verify_delivered_recall(input_row, episode_row, output_row, current_source, *, proof_reader):
    """核验同会话严格先后、实际输入/输出/动作和原有界proof；不授予事实真值。"""
    inp, ep, out = input_row.event, episode_row.event, output_row.event
    if (not inp.is_declaration or not ep.is_declaration or not out.is_declaration
            or not isinstance(inp.payload, ObservationPayload) or not isinstance(out.payload, ObservationPayload)
            or not isinstance(ep.payload, EpisodePayload)):
        raise ValueError('delivered recall requires original unique O/Episode/O declarations')
    before, after, episode = inp.payload.source, out.payload.source, ep.payload
    for ref, source in ((inp.object_ref, before), (out.object_ref, after)):
        if (ref.object_kind != MEMORY_OBJECT_OBSERVATION or ref.owner != current_source.owner
                or ref.versions != current_source.versions or ref.memory_space != inp.object_ref.memory_space
                or observation_reference_source(ref) != source
                or source.owner != current_source.owner or source.versions != current_source.versions
                or (source.source_kind, source.source_id) != (current_source.source_kind, current_source.source_id)):
            raise ValueError('delivered recall crosses owner/version/space/source')
    if not before.document_id < after.document_id < current_source.document_id:
        raise ValueError('delivered recall requires past actual input and output')
    for payload, speaker in ((inp.payload, 1), (out.payload, 2)):
        context = payload.context.value()
        if (not isinstance(context, ObjectIdentity) or context.object_kind != OBJECT_CONTEXT_SCOPE
                or context.owner != current_source.owner or context.versions != current_source.versions
                or context.components != (91501, current_source.owner.session_id, speaker)):
            raise ValueError('delivered recall speaker context differs from its graph')
    if (episode.input_observation_ref != inp.object_ref or episode.output_ref is None
            or episode.output_ref.value() != out.object_ref or episode.failure_ref is not None
            or ep.object_ref.owner != current_source.owner or ep.object_ref.versions != current_source.versions
            or ep.object_ref.memory_space != inp.object_ref.memory_space
            or episode.intent_ref is None or episode.selected_path_ref is None):
        raise ValueError('delivered recall Episode does not bind the actual input and output')
    proof_ref = episode.selected_path_ref.value()
    proof = unpack_record(proof_reader(proof_ref), 91530)
    if not proof or proof[0] not in ((3,), (4,)):
        return False
    if (len(proof) not in (5, 6) or proof[1] != before.stable_key()
            or proof[2] != inp.object_ref.stable_key()
            or proof[3] != episode.intent_ref.value().stable_key()):
        raise ValueError('delivered recall proof source/input/action differs')
    links = delivered_context_links(out.object_ref, out.payload, (episode,), proof_reader=proof_reader)
    if inp.object_ref not in tuple(link.value() for link in links):
        raise ValueError('delivered recall lacks the proven actual input edge')
    return True


def recall_delivered_observations(memory, matched, current_ref, *, declaration_reader):
    """按索引恢复所有合法处理结果为零深度内容锚点，完整历史声明保持。"""
    current_rows = declaration_reader(current_ref)
    if len(current_rows) != 1 or not isinstance(current_rows[0].event.payload, ObservationPayload):
        raise ValueError('delivered recall lacks the unique current Observation')
    current = current_rows[0].event.payload.source
    anchors = {}
    for row in matched:
        source = row.event.payload.source
        context = row.event.payload.context.value()
        if (source.document_id >= current.document_id or not isinstance(context, ObjectIdentity)
                or context.components != (91501, memory.owner.session_id, 1)):
            continue
        anchors[row.event.object_ref] = row
    found, routes = {}, set()
    from pure_integer_ai.experiments.trained_response_delivery import read_delivery_proof
    for item in memory.candidate_index.matching_events(tuple(episode_input_route(ref.stable_key()) for ref in anchors)):
        episode = item.event.payload
        if not isinstance(episode, EpisodePayload) or episode.input_observation_ref not in anchors:
            raise ValueError('Episode input index differs from its physical declaration')
        if episode.output_ref is None or not isinstance(episode.output_ref.value(), MemoryObjectRef):
            continue
        output = episode.output_ref.value()
        rows = declaration_reader(output)
        if len(rows) != 1:
            raise ValueError('delivered recall output is not a unique physical declaration')
        if verify_delivered_recall(anchors[episode.input_observation_ref], item, rows[0], current,
                proof_reader=lambda ref: read_delivery_proof(memory, ref)):
            found[output] = rows[0]
            routes.add(pack_record(DELIVERED_RECALL_ROUTE, episode.input_observation_ref.stable_key(),
                                  item.event.object_ref.stable_key(), output.stable_key()))
    return tuple(found[key] for key in sorted(found, key=lambda ref: ref.stable_key())), tuple(sorted(routes))
