"""A-05 来源准入、信任与独立来源簇的纯领域契约。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
    TypedRef,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


SOURCE_ADMISSION_ACCEPTED = 1
SOURCE_ADMISSION_REJECTED = 2
_SOURCE_ADMISSION_STATES = frozenset({
    SOURCE_ADMISSION_ACCEPTED,
    SOURCE_ADMISSION_REJECTED,
})
SOURCE_TRUST_PROTOCOL_VERSION = 1
_TEXT_HASHER = Hasher("pure_integer_ai.source_trust.text.v1")
_LICENSE_HASHER = Hasher("pure_integer_ai.source_trust.license.v1")


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验开放协议键为非空严格整数元组。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空严格整数元组")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长度稳定键增加长度边界。"""
    return len(value), *value


def _take(
        value: tuple[int, ...],
        cursor: int,
        *,
        label: str,
        allow_empty: bool = False,
        ) -> tuple[tuple[int, ...], int]:
    """从长度前缀整数流读取一个稳定键并返回新游标。"""
    if cursor >= len(value):
        raise ValueError(f"{label} 缺少长度")
    size = value[cursor]
    if type(size) is not int or size < 0 or (size == 0 and not allow_empty):
        raise ValueError(f"{label} 长度非法")
    end = cursor + 1 + size
    if end > len(value):
        raise ValueError(f"{label} 被截断")
    result = tuple(value[cursor + 1:end])
    if result:
        _strict_key(result, label=label)
    return result, end


def _typed_refs(
        value: tuple[TypedRef, ...], *, label: str,
        ) -> tuple[TypedRef, ...]:
    """核验图引用集合已经唯一稳定排序。"""
    if (not isinstance(value, tuple)
            or any(not isinstance(item, TypedRef) for item in value)):
        raise TypeError(f"{label} 必须是 TypedRef tuple")
    keys = tuple(item.stable_key() for item in value)
    if keys != tuple(sorted(set(keys))):
        raise ValueError(f"{label} 必须唯一稳定排序")
    return value


@dataclass(frozen=True)
class SourceTrustRequest:
    """一次尚未写入 SourceRecord 的来源准入请求。"""

    route_kind: ObjectIdentity
    source: SourceRef
    raw_text: str
    license_id: str
    batch_id: int
    trace: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验入口、来源、原文、许可、批次和调用 trace。"""
        if not isinstance(self.route_kind, ObjectIdentity):
            raise TypeError("来源准入 route_kind 必须是 ObjectIdentity")
        if not isinstance(self.source, SourceRef):
            raise TypeError("来源准入 source 必须是 SourceRef")
        if not isinstance(self.raw_text, str):
            raise TypeError("来源准入 raw_text 必须是字符串")
        if not isinstance(self.license_id, str) or not self.license_id:
            raise ValueError("来源准入必须声明非空 license_id")
        assert_int(self.batch_id, _where="SourceTrustRequest.batch_id")
        if type(self.batch_id) is not int or self.batch_id <= 0:
            raise ValueError("来源准入 batch_id 必须是正严格整数")
        if not isinstance(self.trace, tuple):
            raise TypeError("来源准入 trace 必须是 tuple")
        if self.trace:
            _strict_key(self.trace, label="SourceTrustRequest.trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回包含内容和许可指纹、但不泄漏原文的稳定请求键。"""
        text_hash = _TEXT_HASHER.h63(self.raw_text)
        license_hash = _LICENSE_HASHER.h63(self.license_id)
        return (
            SOURCE_TRUST_PROTOCOL_VERSION,
            *_packed(self.route_kind.stable_key()),
            *_packed(self.source.stable_key()),
            text_hash,
            len(self.raw_text),
            license_hash,
            len(self.license_id),
            self.batch_id,
            *_packed(self.trace),
        )

    @classmethod
    def source_from_stable_key(cls, key: tuple[int, ...]) -> SourceRef:
        """从不可逆请求身份中恢复并核验完整 SourceRef。"""
        key = _strict_key(key, label="SourceTrustRequest.stable_key")
        if key[0] != SOURCE_TRUST_PROTOCOL_VERSION:
            raise ValueError("来源准入请求协议版本未注册")
        route_key, cursor = _take(
            key, 1, label="SourceTrustRequest.route_kind")
        source_key, cursor = _take(
            key, cursor, label="SourceTrustRequest.source")
        ObjectIdentity.from_stable_key(route_key)
        source = SourceRef.from_stable_key(source_key)
        if cursor + 5 > len(key):
            raise ValueError("来源准入请求指纹字段被截断")
        text_hash, text_length, license_hash, license_length, batch_id = (
            key[cursor:cursor + 5])
        assert_int(
            text_hash,
            text_length,
            license_hash,
            license_length,
            batch_id,
            _where="SourceTrustRequest.stable_key",
        )
        if (any(type(item) is not int for item in (
                text_hash, text_length, license_hash, license_length, batch_id))
                or text_length < 0 or license_length <= 0 or batch_id <= 0):
            raise ValueError("来源准入请求指纹字段非法")
        _, cursor = _take(
            key,
            cursor + 5,
            label="SourceTrustRequest.trace",
            allow_empty=True,
        )
        if cursor != len(key):
            raise ValueError("来源准入请求稳定键含尾随字段")
        return source


@dataclass(frozen=True)
class SourceTrustAssessment:
    """policy 对完整来源请求给出的来源化、可回溯准入裁决。"""

    request_key: tuple[int, ...]
    policy_state_key: tuple[int, ...]
    decision: int
    source_cluster_key: tuple[int, ...]
    freshness_key: tuple[int, ...]
    source_kind_ref: TypedRef
    license_ref: TypedRef
    trust_ref: TypedRef
    reason_refs: tuple[TypedRef, ...]
    blocking_anomaly_refs: tuple[TypedRef, ...] = ()
    trace: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验裁决状态、开放键和所有图引用的结构不变量。"""
        _strict_key(self.request_key, label="SourceTrustAssessment.request_key")
        _strict_key(
            self.policy_state_key,
            label="SourceTrustAssessment.policy_state_key",
        )
        assert_int(self.decision, _where="SourceTrustAssessment.decision")
        if type(self.decision) is not int or self.decision not in _SOURCE_ADMISSION_STATES:
            raise ValueError("来源准入 decision 未注册")
        _strict_key(
            self.source_cluster_key,
            label="SourceTrustAssessment.source_cluster_key",
        )
        _strict_key(
            self.freshness_key,
            label="SourceTrustAssessment.freshness_key",
        )
        for label, ref in (
                ("source_kind_ref", self.source_kind_ref),
                ("license_ref", self.license_ref),
                ("trust_ref", self.trust_ref)):
            if not isinstance(ref, TypedRef):
                raise TypeError(f"SourceTrustAssessment.{label} 必须是 TypedRef")
        _typed_refs(self.reason_refs, label="SourceTrustAssessment.reason_refs")
        _typed_refs(
            self.blocking_anomaly_refs,
            label="SourceTrustAssessment.blocking_anomaly_refs",
        )
        if not self.reason_refs:
            raise ValueError("来源准入裁决必须提供至少一个图内 reason")
        if (self.decision == SOURCE_ADMISSION_ACCEPTED
                and self.blocking_anomaly_refs):
            raise ValueError("已接受来源不得携带阻断异常")
        if not isinstance(self.trace, tuple):
            raise TypeError("来源准入 assessment trace 必须是 tuple")
        if self.trace:
            _strict_key(self.trace, label="SourceTrustAssessment.trace")

    @property
    def accepted(self) -> bool:
        """返回该裁决是否允许来源进入 SourceRecord 写路径。"""
        return self.decision == SOURCE_ADMISSION_ACCEPTED

    def stable_key(self) -> tuple[int, ...]:
        """返回包含来源簇、时效与图内裁决依据的完整稳定键。"""
        result = [
            SOURCE_TRUST_PROTOCOL_VERSION,
            *_packed(self.request_key),
            *_packed(self.policy_state_key),
            self.decision,
            *_packed(self.source_cluster_key),
            *_packed(self.freshness_key),
            *_packed(self.source_kind_ref.stable_key()),
            *_packed(self.license_ref.stable_key()),
            *_packed(self.trust_ref.stable_key()),
            len(self.reason_refs),
        ]
        for ref in self.reason_refs:
            result.extend(_packed(ref.stable_key()))
        result.append(len(self.blocking_anomaly_refs))
        for ref in self.blocking_anomaly_refs:
            result.extend(_packed(ref.stable_key()))
        result.extend(_packed(self.trace))
        return tuple(result)

    @classmethod
    def from_stable_key(
            cls,
            key: tuple[int, ...],
            ) -> "SourceTrustAssessment":
        """从完整整数键恢复准入裁决、来源簇与全部图内依据。"""
        key = _strict_key(key, label="SourceTrustAssessment.stable_key")
        if key[0] != SOURCE_TRUST_PROTOCOL_VERSION:
            raise ValueError("来源准入 assessment 协议版本未注册")
        request_key, cursor = _take(
            key, 1, label="SourceTrustAssessment.request_key")
        policy_key, cursor = _take(
            key, cursor, label="SourceTrustAssessment.policy_state_key")
        if cursor >= len(key):
            raise ValueError("来源准入 assessment 缺少 decision")
        decision = key[cursor]
        cluster_key, cursor = _take(
            key, cursor + 1, label="SourceTrustAssessment.source_cluster_key")
        freshness_key, cursor = _take(
            key, cursor, label="SourceTrustAssessment.freshness_key")
        source_kind_key, cursor = _take(
            key, cursor, label="SourceTrustAssessment.source_kind_ref")
        license_key, cursor = _take(
            key, cursor, label="SourceTrustAssessment.license_ref")
        trust_key, cursor = _take(
            key, cursor, label="SourceTrustAssessment.trust_ref")
        if cursor >= len(key):
            raise ValueError("来源准入 assessment 缺少 reason 数量")
        reason_count = key[cursor]
        if type(reason_count) is not int or reason_count < 0:
            raise ValueError("来源准入 assessment reason 数量非法")
        reasons = []
        cursor += 1
        for index in range(reason_count):
            item, cursor = _take(
                key, cursor, label=f"SourceTrustAssessment.reason[{index}]")
            reasons.append(TypedRef.from_stable_key(item))
        if cursor >= len(key):
            raise ValueError("来源准入 assessment 缺少异常数量")
        anomaly_count = key[cursor]
        if type(anomaly_count) is not int or anomaly_count < 0:
            raise ValueError("来源准入 assessment 异常数量非法")
        anomalies = []
        cursor += 1
        for index in range(anomaly_count):
            item, cursor = _take(
                key, cursor, label=f"SourceTrustAssessment.anomaly[{index}]")
            anomalies.append(TypedRef.from_stable_key(item))
        trace, cursor = _take(
            key,
            cursor,
            label="SourceTrustAssessment.trace",
            allow_empty=True,
        )
        if cursor != len(key):
            raise ValueError("来源准入 assessment 稳定键含尾随字段")
        return cls(
            request_key,
            policy_key,
            decision,
            cluster_key,
            freshness_key,
            TypedRef.from_stable_key(source_kind_key),
            TypedRef.from_stable_key(license_key),
            TypedRef.from_stable_key(trust_key),
            tuple(reasons),
            tuple(anomalies),
            trace,
        )


class SourceTrustPolicy(Protocol):
    """把开放来源元数据映射为版本化准入和来源簇裁决。"""

    def assess(self, request: SourceTrustRequest) -> SourceTrustAssessment:
        """对尚未写入的完整来源请求给出确定性裁决。"""

    def state_key(self) -> tuple[int, ...]:
        """返回许可、来源 kind、时效和异常规则的版本化状态键。"""

    def clone_for_context(self, ctx: object) -> "SourceTrustPolicy":
        """为隔离上下文克隆只读 policy，并保持状态键不变。"""


__all__ = [
    "SOURCE_ADMISSION_ACCEPTED",
    "SOURCE_ADMISSION_REJECTED",
    "SOURCE_TRUST_PROTOCOL_VERSION",
    "SourceTrustAssessment",
    "SourceTrustPolicy",
    "SourceTrustRequest",
]
