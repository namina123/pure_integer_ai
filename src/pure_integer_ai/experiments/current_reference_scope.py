"""从当前Observation的显式来源边恢复回应指向，历史召回成员保持完整。"""
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import OBJECT_CONTEXT_SCOPE, ObjectIdentity
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_OBSERVATION, MemoryObjectRef, ObservationPayload,
)
from pure_integer_ai.cognition.shared.semantic_object import context_scope_identity, semantic_source
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record


CURRENT_REFERENCE_SCOPE = 91658


@dataclass(frozen=True, slots=True)
class CurrentReferenceScope:
    """当前输入、来源化当前回应指向及其全部合格交付上下文。"""

    observation: MemoryObjectRef
    source_link: ObjectIdentity
    context_keys: tuple[tuple[int, ...], ...]
    continuation_keys: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        """作用域保存完整规范成员，不用候选排序或摘要选择历史对象。"""
        if (not isinstance(self.observation, MemoryObjectRef)
                or not isinstance(self.source_link, ObjectIdentity)
                or self.context_keys != tuple(sorted(set(self.context_keys)))
                or type(self.continuation_keys) is not tuple
                or len(self.continuation_keys) > 8
                or any(type(key) is not tuple or not key
                       or any(type(value) is not int or value < 0 for value in key)
                       for key in self.continuation_keys)):
            raise ValueError('current reference scope members are not canonical')
        self.stable_key()

    def stable_key(self):
        """完整记录可在其它语言以有序整数段及显式长度边界复原。"""
        fields = ((1,), self.observation.stable_key(), self.source_link.stable_key(),
                  pack_record(1, *self.context_keys))
        if self.continuation_keys:
            fields = ((2,), *fields[1:], pack_record(2, *self.continuation_keys))
        return pack_record(CURRENT_REFERENCE_SCOPE, *fields)

    def permits(self, context):
        """只核对显式作用域成员；不根据context深度或时间挑选对象。"""
        return context.stable_key() in self.context_keys


def _read_direct_scope(memory, observation, contexts):
    """只读当前声明及已核验交付图，拒绝跨来源、跨版本或多指向状态。

    未携带新来源边的既有声明返回None，保留其原版本行为。新来源边存在
    但未召回对应交付时返回空成员作用域，不能回退到历史窗口授权对象。
    """
    if observation is None:
        return None
    rows = memory.intake.event_log.query(access=memory.candidate_index.access,
        event_kind=MEMORY_EVENT_OBSERVATION, object_ref=observation)
    if len(rows) != 1 or not isinstance(rows[0].event.payload, ObservationPayload):
        raise ValueError('reference scope lacks one current Observation')
    payload = rows[0].event.payload
    links = tuple(link.value() for link in payload.relation_occurrences)
    scopes = tuple(link for link in links if isinstance(link, ObjectIdentity)
                   and link.object_kind == OBJECT_CONTEXT_SCOPE
                   and link.components[-2:] == (1, CURRENT_REFERENCE_SCOPE))
    if not scopes:
        return None
    if len(scopes) != 1:
        raise ValueError('current Observation has competing explicit reference scopes')
    scope, = scopes
    source = semantic_source(scope)
    current = payload.source
    if (scope != context_scope_identity(source, (CURRENT_REFERENCE_SCOPE,))
            or scope.owner != current.owner or scope.versions != current.versions
            or source.owner != current.owner or source.versions != current.versions
            or source.source_kind != current.source_kind or source.source_id != current.source_id
            or source.document_id >= current.document_id
            or observation.owner != current.owner or observation.versions != current.versions
            or context_scope_identity(source, (91543,)) not in links):
        raise ValueError('current reference scope differs from the original source context')
    matched = tuple(context for context in contexts
                    if tuple(context.source_ref[1:]) == source.stable_key())
    if any(context.proof_ref.owner != current.owner
           or context.proof_ref.versions != current.versions for context in matched):
        raise ValueError('current reference scope crosses a delivery proof boundary')
    return CurrentReferenceScope(observation, scope,
                                 tuple(sorted({context.stable_key() for context in matched})))


def read_current_reference_scope(memory, observation, contexts):
    """从显式来源链恢复对象作用域，通用无对象交付不吞掉原焦点。

    每步只沿原交付的输入来源边前进，全部历史候选保留；有限链缺证据、
    有新对象或超预算时返回空作用域，不能退回最近事实或提高召回深度。
    """
    direct = _read_direct_scope(memory, observation, contexts)
    if direct is None:
        return None
    from pure_integer_ai.experiments.reference_scope_continuation import continuation_input

    active, chain = direct, []
    by_key = {context.stable_key(): context for context in contexts}
    while active.context_keys:
        matched = tuple(by_key[key] for key in active.context_keys)
        if len(matched) != 1:
            break
        permitted, prior, witnesses = continuation_input(memory, matched[0])
        if not permitted:
            break
        if prior is None or len(chain) == 8:
            return CurrentReferenceScope(direct.observation, direct.source_link, (), tuple(chain))
        previous = _read_direct_scope(memory, prior, contexts)
        if previous is None:
            return CurrentReferenceScope(direct.observation, direct.source_link, (), tuple(chain))
        chain.append(pack_record(91659, matched[0].stable_key(),
                                 previous.observation.stable_key(), previous.source_link.stable_key(),
                                 pack_record(1, *witnesses)))
        active = previous
    return CurrentReferenceScope(direct.observation, direct.source_link,
                                 active.context_keys, tuple(chain))
