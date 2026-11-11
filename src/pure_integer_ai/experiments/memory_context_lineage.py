"""区分已交付输出的实际输入边和完整采用证明依赖，不改写任何原声明。"""
from pure_integer_ai.cognition.shared.memory_event import (
    EpisodePayload, MemoryLinkedRef, MemoryObjectRef, MEMORY_OBJECT_OBSERVATION,
)
from pure_integer_ai.cognition.shared.memory_record_identity import observation_reference_source
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record


def _proof_authorizes_generic_input(episode, source, proof_reader):
    """旧通用输出可缺直接输入边，但完整原proof必须逐字段核对输入、act及采用集合。"""
    from pure_integer_ai.experiments.trained_response_delivery import DELIVERY_PROOF_IDENTITY
    if proof_reader is None or episode.selected_path_ref is None or episode.intent_ref is None:
        return False
    fields = unpack_record(proof_reader(episode.selected_path_ref.value()), DELIVERY_PROOF_IDENTITY)
    return (len(fields) in (5, 6) and fields[0] == (3,)
            and SourceRef.from_stable_key(fields[1]) == source
            and fields[2] == episode.input_observation_ref.stable_key()
            and fields[3] == episode.intent_ref.value().stable_key()
            and unpack_record(fields[4], 1) == tuple(ref.stable_key() for ref in episode.used_memory_refs))


def delivered_context_links(observation, payload, episodes, *, proof_reader=None):
    """唯一实际 Episode 只授权它的输入作为后继语境，完整证明依赖另由冷审核验。

    普通观察没有 Episode 时保留原边。所有原成员、Use/Outcome、Evidence
    和物理记录保持；返回值只是本次有界语境行走允许消费的派生边集合。
    """
    if not episodes:
        return payload.relation_occurrences
    if len(episodes) != 1 or not isinstance(episodes[0], EpisodePayload):
        raise ValueError('delivered context requires one complete physical Episode')
    episode, = episodes
    current, input_ref = payload.source, episode.input_observation_ref
    source = observation_reference_source(input_ref)
    links = set(payload.relation_occurrences)
    if (episode.output_ref is None or episode.output_ref.value() != observation
            or observation_reference_source(observation) != current
            or observation.object_kind != MEMORY_OBJECT_OBSERVATION
            or input_ref.owner != observation.owner or input_ref.versions != observation.versions
            or input_ref.memory_space != observation.memory_space
            or source.source_kind != current.source_kind or source.source_id != current.source_id
            or source.owner != current.owner or source.versions != current.versions
            or source.document_id >= current.document_id
            or episode.session_ref.owner != current.owner
            or episode.session_ref.versions != current.versions
            or episode.selected_path_ref is None or episode.selected_path_ref not in links
            or any(MemoryLinkedRef.memory(ref) not in links for ref in episode.used_memory_refs)):
        raise ValueError('delivered context input/proof/source differs from the original graph')
    input_link = MemoryLinkedRef.memory(input_ref)
    if input_link not in links and not _proof_authorizes_generic_input(episode, source, proof_reader):
        raise ValueError('delivered context input has no complete original proof membership')
    retained = tuple(link for link in payload.relation_occurrences
                     if not isinstance(link.value(), MemoryObjectRef) or link.value() == input_ref)
    return retained if input_link in links else (*retained, input_link)
