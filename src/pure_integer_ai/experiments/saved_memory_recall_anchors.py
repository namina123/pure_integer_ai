"""从当前输入候选及原结构召回路由独立恢复历史 Observation 锚点。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.memory_event import (
    INTAKE_DERIVED_HYPOTHESIS, INTAKE_DERIVED_OBSERVATION,
    HypothesisPayload, MemoryObjectRef, ObservationPayload, EpisodePayload, MEMORY_EVENT_EPISODE,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import frame_key, memory_candidate_routes
from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError


def read_saved_recall_observation_anchors(memory, trace, *, input_observation,
                                         input_candidate_keys, declaration_reader):
    """只用完整当前候选与物理 H/O 声明授权零深度锚点，拒绝未来及跨域来源。"""
    def declaration(ref, payload_type):
        """每个锚点成员必须是同一 Memory 空间内的唯一原声明。"""
        rows = declaration_reader(ref)
        if (ref.owner != input_observation.owner or ref.versions != input_observation.versions
                or ref.memory_space != input_observation.memory_space
                or len(rows) != 1 or not rows[0].event.is_declaration
                or rows[0].event.object_ref != ref
                or not isinstance(rows[0].event.payload, payload_type)):
            raise QueryMemoryOheEvidenceError("saved recall anchor lacks a unique scoped declaration")
        return rows[0]

    current = declaration(input_observation, ObservationPayload).event.payload.source
    candidates = set(input_candidate_keys)
    anchors = {}
    for route in trace.memory_routes():
        if route[0] != 1:
            continue
        fields = unpack_record(route, 1)
        if len(fields) != 3:
            raise QueryMemoryOheEvidenceError("saved recall route is incomplete")
        candidate, shared, hypothesis_key = fields
        if candidate not in candidates or shared not in memory_candidate_routes(candidate):
            raise QueryMemoryOheEvidenceError("saved recall route is not authorized by the current input")
        hypothesis_ref = MemoryObjectRef.from_stable_key(hypothesis_key)
        hypothesis = declaration(hypothesis_ref, HypothesisPayload).event.payload.hypothesis
        source = hypothesis.observation
        if (source.owner != current.owner or source.versions != current.versions
                or source.source_kind != current.source_kind or source.source_id != current.source_id
                or source.document_id > current.document_id
                or shared not in memory_candidate_routes(hypothesis.candidate_key)):
            raise QueryMemoryOheEvidenceError("saved recall anchor has no current structural match or is from the future")
        manifest = memory.intake.require_current_manifest(source)
        observations = tuple(binding.object_ref for binding in manifest.bindings
                             if binding.binding_kind == INTAKE_DERIVED_OBSERVATION)
        if (len(observations) != 1 or not any(
                binding.binding_kind == INTAKE_DERIVED_HYPOTHESIS and binding.object_ref == hypothesis_ref
                for binding in manifest.bindings)):
            raise QueryMemoryOheEvidenceError("saved recall anchor differs from its original intake")
        row = declaration(observations[0], ObservationPayload)
        if row.event.payload.source != source or hypothesis.scope != row.event.scope:
            raise QueryMemoryOheEvidenceError("saved recall anchor source/scope differs from graph")
        anchors[observations[0]] = row
    delivered_routes = tuple(route for route in trace.memory_routes() if route[0] == 5)
    if delivered_routes:
        from pure_integer_ai.experiments.memory_delivered_recall import verify_delivered_recall
        from pure_integer_ai.experiments.trained_response_delivery import read_delivery_proof
        input_anchors = dict(anchors)
        actual = set()
        for route in delivered_routes:
            fields = unpack_record(route, 5)
            if len(fields) != 3:
                raise QueryMemoryOheEvidenceError('saved delivered recall route is incomplete')
            inp, episode, output = (MemoryObjectRef.from_stable_key(key) for key in fields)
            if inp not in input_anchors:
                raise QueryMemoryOheEvidenceError('delivered recall input has no current content match')
            erow = declaration(episode, EpisodePayload)
            orow = declaration(output, ObservationPayload)
            try:
                valid = verify_delivered_recall(input_anchors[inp], erow, orow, current,
                    proof_reader=lambda ref: read_delivery_proof(memory, ref))
            except ValueError as error:
                raise QueryMemoryOheEvidenceError(str(error)) from error
            if not valid:
                raise QueryMemoryOheEvidenceError('delivered recall has no registered bounded proof')
            anchors[output] = orow
            actual.add(route)
        expected = set()
        # 冷审独立枚举原Episode；生产只读有版本的稀疏索引，不能自证完整。
        for erow in memory.intake.event_log.query(access=memory.candidate_index.access,
                event_kind=MEMORY_EVENT_EPISODE):
            episode = erow.event.payload
            if (not erow.event.is_declaration or not isinstance(episode, EpisodePayload)
                    or episode.input_observation_ref not in input_anchors or episode.output_ref is None):
                continue
            output = episode.output_ref.value()
            if not isinstance(output, MemoryObjectRef):
                continue
            orow = declaration(output, ObservationPayload)
            if orow.event.payload.source.document_id >= current.document_id:
                continue
            inp = episode.input_observation_ref
            context = input_anchors[inp].event.payload.context.value()
            if getattr(context, 'components', ()) != (91501, memory.owner.session_id, 1):
                continue
            if verify_delivered_recall(input_anchors[inp], erow, orow, current,
                    proof_reader=lambda ref: read_delivery_proof(memory, ref)):
                expected.add(frame_key(5, inp.stable_key(), erow.event.object_ref.stable_key(), output.stable_key()))
        if actual != expected:
            raise QueryMemoryOheEvidenceError('saved delivered recall omitted or added an actual eligible output')
    return tuple(anchors[ref] for ref in sorted(anchors, key=lambda item: item.stable_key()))
