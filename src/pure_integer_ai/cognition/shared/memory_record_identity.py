"""Observation/Episode/Evidence 的一等记录身份；完整关系由唯一声明承载。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_OBSERVED, LogicalClockIdentity, LogicalTimestamp,
    ScopeIdentity, SCOPE_SESSION,
)


MEMORY_RECORD_IDENTITY = 91574
MEMORY_RECORD_IDENTITY_VERSION = 2
OBSERVATION_RECORD = 1
EPISODE_RECORD = 2
EVIDENCE_RECORD = 3
EVIDENCE_RECORD_OBSERVED = 1
EVIDENCE_RECORD_H00 = 2


def _field(values: tuple[int, ...]) -> tuple[int, ...]:
    """以严格整数长度前缀编码一个完整身份字段。"""
    if type(values) is not tuple or not values or any(type(value) is not int for value in values):
        raise ValueError("Memory record identity field must be a nonempty integer tuple")
    return len(values), *values


def _take(key: tuple[int, ...], cursor: int) -> tuple[tuple[int, ...], int]:
    """读取完整字段并拒绝负长度、空字段和截断。"""
    if cursor >= len(key) or key[cursor] <= 0 or cursor + 1 + key[cursor] > len(key):
        raise ValueError("Memory record identity field is truncated")
    end = cursor + 1 + key[cursor]
    return key[cursor + 1:end], end


def _header(key: tuple[int, ...], kind: int) -> None:
    """核验开放对象键的协议、版本、种类及严格整数类型。"""
    if (type(key) is not tuple or len(key) < 4
            or any(type(value) is not int for value in key)
            or key[:3] != (MEMORY_RECORD_IDENTITY, MEMORY_RECORD_IDENTITY_VERSION, kind)):
        raise ValueError("Memory record identity header is invalid")


def observation_record_key(source: SourceRef, observed_at: LogicalTimestamp) -> tuple[int, ...]:
    """用来源与完整 observed 时钟标识记录，不把关系载荷嵌进引用。"""
    if not isinstance(source, SourceRef) or not isinstance(observed_at, LogicalTimestamp):
        raise TypeError("Observation record identity requires source and logical timestamp")
    scope = observed_at.clock.scope
    if scope.owner != source.owner or scope.versions != source.versions:
        raise ValueError("Observation record identity source/timestamp owner or version drifted")
    return (MEMORY_RECORD_IDENTITY, MEMORY_RECORD_IDENTITY_VERSION, OBSERVATION_RECORD,
            *_field(source.stable_key()), *_field(observed_at.stable_key()))


def observation_record_source(key: tuple[int, ...]) -> SourceRef:
    """只解身份字段供来源定位；来源正文和成员仍须读取唯一声明。"""
    _header(key, OBSERVATION_RECORD)
    source_key, cursor = _take(key, 3)
    timestamp_key, cursor = _take(key, cursor)
    if cursor != len(key):
        raise ValueError("Observation record identity has trailing values")
    source = SourceRef.from_stable_key(source_key)
    timestamp = LogicalTimestamp.from_stable_key(timestamp_key)
    if observation_record_key(source, timestamp) != key:
        raise ValueError("Observation record identity fields drifted")
    return source


def episode_record_key(session: ScopeIdentity, round_seq: int,
                       created_at: LogicalTimestamp) -> tuple[int, ...]:
    """用 session、处理轮次与完整创建时钟标识独立 Episode 记录。"""
    if (not isinstance(session, ScopeIdentity) or session.scope_kind != SCOPE_SESSION
            or not isinstance(created_at, LogicalTimestamp)):
        raise TypeError("Episode record identity requires session and logical timestamp")
    if type(round_seq) is not int or round_seq < 0:
        raise ValueError("Episode record identity round must be a nonnegative integer")
    scope = created_at.clock.scope
    if scope.owner != session.owner or scope.versions != session.versions:
        raise ValueError("Episode record identity session/timestamp owner or version drifted")
    return (MEMORY_RECORD_IDENTITY, MEMORY_RECORD_IDENTITY_VERSION, EPISODE_RECORD,
            *_field(session.stable_key()), round_seq, *_field(created_at.stable_key()))


def evidence_record_key(observed_at: LogicalTimestamp,
                        compatibility_record_key: tuple[int, ...] = ()) -> tuple[int, ...]:
    """按原生时钟或 H-00 ledger/id 标识证据，排除成员及替代链的递归复制。"""
    if not isinstance(observed_at, LogicalTimestamp):
        raise TypeError("Evidence record identity requires a logical timestamp")
    if observed_at.clock.clock_kind != CLOCK_MEMORY_OBSERVED:
        raise ValueError("Evidence record identity requires the observed clock")
    if (type(compatibility_record_key) is not tuple
            or any(type(value) is not int for value in compatibility_record_key)):
        raise ValueError("Evidence compatibility record must be a strict integer tuple")
    header = MEMORY_RECORD_IDENTITY, MEMORY_RECORD_IDENTITY_VERSION, EVIDENCE_RECORD
    if not compatibility_record_key:
        return (*header, EVIDENCE_RECORD_OBSERVED, *_field(observed_at.stable_key()))
    record = EvidenceRecord.from_stable_key(compatibility_record_key)
    hypothesis = record.hypothesis
    expected = LogicalTimestamp(
        LogicalClockIdentity(hypothesis.scope, CLOCK_MEMORY_OBSERVED),
        record.timestamp_seq + 1,
    )
    if observed_at != expected:
        raise ValueError("Evidence compatibility record timestamp drifted")
    return (*header, EVIDENCE_RECORD_H00,
            *_field(hypothesis.hypothesis_kind), *_field(hypothesis.scope.stable_key()),
            *_field(hypothesis.observation.stable_key()), record.evidence_id)


def observation_reference_source(ref) -> SourceRef:
    """兼容 V1 全载荷和 V2 记录身份；身份来源不能代替声明证据。"""
    from pure_integer_ai.cognition.shared.memory_event import (
        MemoryObjectRef, MEMORY_OBJECT_OBSERVATION, ObservationPayload,
    )
    if not isinstance(ref, MemoryObjectRef) or ref.object_kind != MEMORY_OBJECT_OBSERVATION:
        raise TypeError("Observation source requires an Observation reference")
    source = (observation_record_source(ref.object_key)
              if ref.object_key[0] == MEMORY_RECORD_IDENTITY
              else ObservationPayload.from_stable_key(ref.object_key).source)
    if source.owner != ref.owner or source.versions != ref.versions:
        raise ValueError("Observation reference source owner/version drifted")
    return source
