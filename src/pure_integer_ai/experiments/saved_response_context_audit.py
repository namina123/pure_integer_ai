"""从不可变 O/H/E、proof 与 Episode 恢复事实及通用非事实历史。"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from pure_integer_ai.cognition.shared.memory_event import (
    INTAKE_DERIVED_EVIDENCE, INTAKE_DERIVED_HYPOTHESIS, INTAKE_DERIVED_OBSERVATION,
    MEMORY_EVENT_EPISODE, EvidencePayload, HypothesisPayload, MemoryObjectRef, ObservationPayload,
)
from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    MemoryActiveStructures, MemoryEvidenceStructure, MemoryHypothesisStructure,
    MemoryObservationStructure, TrainedDialogueMemoryGraph,
)
from pure_integer_ai.experiments.trained_response_context import RESPONSE_CONTEXT_VERSION, restore_response_contexts


def read_saved_factual_response_contexts(memory, trace, *, input_observation,
                                       declaration_reader, source_hash_reader, input_candidate_keys=None):
    """保持原context接口；完整来源锚点由同一独立冷读内核恢复。"""
    return restore_saved_factual_response_contexts(memory, trace, input_observation=input_observation,
        declaration_reader=declaration_reader, source_hash_reader=source_hash_reader,
        input_candidate_keys=input_candidate_keys).contexts


@dataclass(frozen=True, slots=True)
class SavedResponseContextAudit:
    """完整冷图context和由当前输入匹配独立授权的来源O，不赋予内容真值。"""

    contexts: tuple
    recall_observations: tuple[MemoryObjectRef, ...]


def restore_saved_factual_response_contexts(memory, trace, *, input_observation,
                                          declaration_reader, source_hash_reader, input_candidate_keys=None):
    """逐项核验来源、深度及物理 proof/Episode，只恢复结构历史而不读取答案。"""
    refs = trace.trace.get("response_context_refs", ())
    if not isinstance(refs, (list, tuple)) or len(refs) > 32:
        raise QueryMemoryOheEvidenceError("saved response contexts exceed cold audit budget")
    if not refs and input_candidate_keys is None:
        return SavedResponseContextAudit((), ())
    keys = tuple(trace.resolve_ref(ref) for ref in refs)
    if len(set(keys)) != len(keys):
        raise QueryMemoryOheEvidenceError("saved response context is duplicated")
    parsed = []
    for key in keys:
        parts = unpack_record(key, RESPONSE_CONTEXT_VERSION)
        registered = ((len(parts) == 14 and parts[0] == (4,))
                      or (len(parts) == 10 and parts[0] == (2,)))
        if not registered or len(parts[-1]) != 1:
            raise QueryMemoryOheEvidenceError("saved response context format is unregistered")
        parsed.append(parts)

    def row(ref, expected):
        found = declaration_reader(ref)
        if len(found) != 1 or not isinstance(found[0].event.payload, expected):
            raise QueryMemoryOheEvidenceError("saved response context lacks a unique declaration")
        if (ref.owner != memory.owner or ref.memory_space != input_observation.memory_space
                or ref.versions != input_observation.versions):
            raise QueryMemoryOheEvidenceError("saved response context crosses owner/version/space")
        return found[0]

    input_row = row(input_observation, ObservationPayload)
    episodes = None

    def response_episodes_for(output):
        nonlocal episodes
        if episodes is None:
            episodes = memory.intake.event_log.query(
                access=memory.candidate_index.access, event_kind=MEMORY_EVENT_EPISODE)
        return tuple(item.event.payload for item in episodes
                     if item.event.is_declaration and item.event.payload.output_ref is not None
                     and item.event.payload.output_ref.value() == output)

    memory.response_episodes_for = response_episodes_for
    # This only follows durable Observation/context links. It does not run
    # matching, inference, the query executor, or a producer.
    closure_owner = SimpleNamespace(owner=memory.owner, intake=memory.intake,
        source_id=input_row.event.payload.source.source_id,
        _event_payload=TrainedDialogueMemoryGraph._event_payload,
        response_episodes_for=response_episodes_for)
    anchors = ()
    if input_candidate_keys is not None:
        from pure_integer_ai.experiments.saved_memory_recall_anchors import read_saved_recall_observation_anchors
        anchors = read_saved_recall_observation_anchors(memory, trace,
            input_observation=input_observation, input_candidate_keys=input_candidate_keys,
            declaration_reader=declaration_reader)
    recall_observations = tuple(item.event.object_ref for item in anchors)
    if not refs:
        return SavedResponseContextAudit((), recall_observations)
    _rows, depths, _sources, _routes = TrainedDialogueMemoryGraph._observation_closure(
        closure_owner, anchors, (input_observation,), memory.candidate_index.access,
        context_depth_limit=1)
    result = []
    for original, parts in zip(keys, parsed, strict=True):
        observation = MemoryObjectRef.from_stable_key(parts[1])
        hypothesis_ref = MemoryObjectRef.from_stable_key(parts[2])
        orow, hrow = row(observation, ObservationPayload), row(hypothesis_ref, HypothesisPayload)
        opayload, hypothesis = orow.event.payload, hrow.event.payload.hypothesis
        source = opayload.source
        source_hash = source_hash_reader(source)
        depth, = parts[-1]
        if (type(source_hash) is not int or source_hash <= 0
                or parts[3] != (source_hash, *source.stable_key())
                or parts[4] != orow.event.scope.stable_key()
                or hypothesis.observation != source or hypothesis.scope != orow.event.scope
                or depths.get(source) != depth):
            raise QueryMemoryOheEvidenceError("saved response source/scope/context depth differs from graph")
        manifest = memory.intake.require_current_manifest(source)
        if tuple(binding.object_ref for binding in manifest.bindings
                 if binding.binding_kind == INTAKE_DERIVED_OBSERVATION) != (observation,):
            raise QueryMemoryOheEvidenceError("saved response Observation differs from intake")
        hbindings = tuple(binding for binding in manifest.bindings
                          if binding.binding_kind == INTAKE_DERIVED_HYPOTHESIS
                          and binding.object_ref == hypothesis_ref)
        if len(hbindings) != 1:
            raise QueryMemoryOheEvidenceError("saved response Hypothesis differs from intake")
        evidence = []
        for binding in manifest.bindings:
            if (binding.binding_kind != INTAKE_DERIVED_EVIDENCE
                    or binding.lineage_key != hbindings[0].lineage_key):
                continue
            erow = row(binding.object_ref, EvidencePayload)
            payload = erow.event.payload
            if payload.hypothesis_ref != hypothesis_ref or payload.source != source:
                raise QueryMemoryOheEvidenceError("saved response Evidence differs from intake")
            evidence.append(MemoryEvidenceStructure(binding.object_ref.stable_key(),
                hypothesis_ref.stable_key(), (source_hash, *source.stable_key()),
                erow.event.scope.stable_key(), payload.stance, payload.detail, payload.stable_key(),
                () if payload.supersedes_ref is None else payload.supersedes_ref.stable_key()))
        evidence.sort(key=lambda item: item.evidence_key)
        context = opayload.context.value()
        if context.components[:2] != (91501, memory.owner.session_id) or len(context.components) != 3:
            raise QueryMemoryOheEvidenceError("saved response speaker context differs")
        ostructure = MemoryObservationStructure(observation.stable_key(), parts[3], parts[4],
            source_hash, source.document_id, context.components[2], (hypothesis_ref.stable_key(),),
            opayload.relation_occurrences, depth)
        hstructure = MemoryHypothesisStructure(observation.stable_key(), hypothesis_ref.stable_key(),
            hypothesis.hypothesis_kind, hypothesis.candidate_key, hypothesis.competition_key,
            parts[3], parts[4], tuple(item.evidence_key for item in evidence))
        restored = restore_response_contexts(memory, MemoryActiveStructures(
            (ostructure,), (hstructure,), tuple(evidence)))
        if len(restored) != 1 or restored[0].stable_key() != original:
            raise QueryMemoryOheEvidenceError("saved response context differs from proof/Episode graph")
        result.extend(restored)
    return SavedResponseContextAudit(tuple(result), recall_observations)
