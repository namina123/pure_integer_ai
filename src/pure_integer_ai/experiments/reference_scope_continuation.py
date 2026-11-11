"""只沿已交付的无对象回应证明延续焦点，不按历史位置选择对象。"""
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT, OBJECT_ENTITY, OBJECT_EVENT, OBJECT_SET_EXPR, ObjectIdentity, SourceRef,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_OBSERVATION, EpisodePayload, MemoryObjectRef, ObservationPayload,
)
from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record
from pure_integer_ai.experiments.trained_response_context import DeliveredGenericResponseContext
from pure_integer_ai.experiments.trained_response_delivery import DELIVERY_PROOF_IDENTITY, read_delivery_proof
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity, authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import RELATION_PROPERTY, ROLE_PROPERTY_ATTRIBUTE


def neutral_input_witnesses(memory, payload):
    """概念可能是谓词或属性限定，须由实际来源成员证明而非类型名猜测。

    只有当前输入明确包含的命题/谓词与训练结构相符时，谓词和PROPERTY的
    ATTRIBUTE成员才不充当新焦点；实体/事件/集合及其它概念仍阻断延续。
    返回完整来源框架/成员作为可逆证明，不把属性名当作主体或新事实。
    """
    objects = tuple(link.value() for link in payload.relation_occurrences)
    keys = {value.stable_key() for value in objects if isinstance(value, ObjectIdentity)}
    targets = {value.stable_key() for value in objects if isinstance(value, ObjectIdentity)
               and value.object_kind in (OBJECT_CONCEPT, OBJECT_ENTITY, OBJECT_EVENT, OBJECT_SET_EXPR)}
    if any(value.object_kind in (OBJECT_ENTITY, OBJECT_EVENT, OBJECT_SET_EXPR)
           for value in objects if isinstance(value, ObjectIdentity)):
        return False, ()
    projector = getattr(memory, 'input_projector', None)
    witnesses = {}
    for frame in (*getattr(projector, 'open_role_frames', ()),
                  *getattr(projector, 'envelope_role_frames', ())):
        if frame.proposition in keys and frame.predicate in targets:
            witnesses.setdefault(frame.predicate, set()).add(frame.stable_key())
    predicate = authored_relation_identity(RELATION_PROPERTY).stable_key()
    attribute = authored_relation_role_identity(ROLE_PROPERTY_ATTRIBUTE).stable_key()
    for structure in getattr(projector, 'structures', ()):
        if structure.proposition_key not in keys or structure.predicate_key != predicate or predicate not in keys:
            continue
        for member in structure.members:
            if (member.candidate_key in targets and member.projection_kind == 1
                    and member.role_key == attribute):
                witnesses.setdefault(member.candidate_key, set()).add(pack_neutral_member(structure, member))
    if targets != witnesses.keys():
        return False, ()
    return True, tuple(sorted({key for members in witnesses.values() for key in members}))


def pack_neutral_member(structure, member):
    """限定词保留完整父结构和逐角色来源，不能只用概念键豁免。"""
    from pure_integer_ai.cognition.understanding.query_open_roles import pack_record
    return pack_record(91659, structure.stable_key(), member.stable_key())


def continuation_input(memory, context):
    """返回是否为中性无对象交付及其原输入；新端点使延续保持关闭。

    非通用或有对象的交付是作用域终点。通用交付必须具有完整kind3证明、
    唯一实际Episode及同会话递增输入/输出；损坏证据抛错而非借用历史。
    """
    if (not isinstance(context, DeliveredGenericResponseContext)
            or context.graph_surfaces or context.role_occurrences):
        return False, None, ()
    observation = context.input_observation
    output = SourceRef.from_stable_key(tuple(context.source_ref[1:]))
    rows = memory.intake.event_log.query(access=memory.candidate_index.access,
        event_kind=MEMORY_EVENT_OBSERVATION, object_ref=observation)
    if len(rows) != 1 or not isinstance(rows[0].event.payload, ObservationPayload):
        raise ValueError('focus continuation lacks its original input Observation')
    payload = rows[0].event.payload
    source = payload.source
    if (any(not isinstance(ref, MemoryObjectRef) or ref.owner != source.owner
            or ref.versions != source.versions or ref.memory_space != observation.memory_space
            for ref in (context.observation, context.hypothesis, observation))
            or context.proof_ref.owner != source.owner or context.proof_ref.versions != source.versions
            or output.owner != source.owner or output.versions != source.versions
            or output.source_kind != source.source_kind or output.source_id != source.source_id
            or output.document_id <= source.document_id):
        raise ValueError('focus continuation crosses source, owner, version, space or logical order')
    proof = unpack_record(read_delivery_proof(memory, context.proof_ref), DELIVERY_PROOF_IDENTITY)
    if (len(proof) not in (5, 6) or proof[0] != (3,)
            or proof[1] != source.stable_key() or proof[2] != observation.stable_key()
            or proof[3] != context.response_act.stable_key()
            or (len(proof) == 6 and unpack_record(proof[5], 2))):
        raise ValueError('focus continuation requires the exact bounded no-object delivery proof')
    episodes = memory.response_episodes_for(context.observation)
    if (len(episodes) != 1 or not isinstance(episodes[0], EpisodePayload)
            or episodes[0].input_observation_ref != observation
            or episodes[0].output_ref is None or episodes[0].output_ref.value() != context.observation
            or episodes[0].selected_path_ref is None
            or episodes[0].selected_path_ref.value() != context.proof_ref
            or episodes[0].intent_ref is None or episodes[0].intent_ref.value() != context.response_act
            or episodes[0].failure_ref is not None
            or episodes[0].round_seq != output.document_id
            or episodes[0].session_ref.owner != source.owner
            or episodes[0].session_ref.versions != source.versions
            or episodes[0].session_ref.local_id != source.owner.session_id):
        raise ValueError('focus continuation differs from its unique original delivered Episode')
    neutral, witnesses = neutral_input_witnesses(memory, payload)
    return True, observation if neutral else None, witnesses
