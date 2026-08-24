"""B1 双学习平面的统一纯值输入协议。

这个模块只定义可跨语言重建的值和账本转换，不执行文本解析、图推理或回答生成。
同一个 :class:`LearningInputCapsule` 可以显式投影到 Core 或 Runtime Memory，但两
个账本分别绑定 scope、拥有独立生命周期，Memory 不会自动晋升为 Core。

JSON/document 形状只是交换层；权威身份是稳定整数键，规范记录由整数 tuple 表达。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


CAPSULE_SCHEMA = 1

# Projection kinds are protocol tags, not domain meanings supplied by the caller.
PROJECTION_CORE = 1
PROJECTION_RUNTIME = 2

STATUS_OBSERVED = 1
STATUS_LEARNED = 2
STATUS_REJECTED = 3
STATUS_CONFLICTED = 4
_STATUSES = frozenset({
    STATUS_OBSERVED, STATUS_LEARNED, STATUS_REJECTED, STATUS_CONFLICTED,
})

ADMISSION_ACCEPTED = 1
ADMISSION_DUPLICATE = 2
ADMISSION_CONFLICT = 3
ADMISSION_SCOPE_MISMATCH = 4
ADMISSION_REVISION_GAP = 5

EVENT_ASSERTION = 1
EVENT_REVISION = 2
EVENT_TOMBSTONE = 3
EVENT_CONFLICT = 4


class LearningInputCapsuleError(ValueError):
    """输入 capsule、投影或账本违反双平面边界。"""


def _ints(value: Any, *, label: str, empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, tuple) or (not empty and not value):
        raise LearningInputCapsuleError(
            f"{label} 必须是{'可空' if empty else '非空'}整数 tuple")
    if value:
        assert_int(*value, _where=label)
        if any(type(item) is not int for item in value):
            raise LearningInputCapsuleError(f"{label} 必须使用严格整数")
    return value


def _positive(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise LearningInputCapsuleError(f"{label} 必须是正严格整数")
    return value


def _digest(value: Any, *, label: str) -> tuple[int, ...]:
    result = _ints(value, label=label)
    if len(result) != hashlib.sha256().digest_size or any(
            item < 0 or item > 255 for item in result):
        raise LearningInputCapsuleError(
            f"{label} 必须是 32 个 0..255 整数")
    return result


def digest_bytes(value: bytes) -> tuple[int, ...]:
    """将原始 bytes 的 SHA-256 返回为可跨语言复现的整数 tuple。"""
    if not isinstance(value, bytes):
        raise TypeError("digest_bytes 只接受 bytes")
    return tuple(hashlib.sha256(value).digest())


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    result.extend((len(value), *value))


def _pack_many(result: list[int], values: tuple[tuple[int, ...], ...]) -> None:
    result.append(len(values))
    for value in values:
        _pack(result, value)


def _capsule_record(
        source_key: tuple[int, ...], scope_key: tuple[int, ...],
        version_key: tuple[int, ...], parent_version_key: tuple[int, ...],
        language: int, modality: int, raw_digest: tuple[int, ...],
        structural_units: tuple[tuple[int, ...], ...], authority_key: tuple[int, ...],
        license_id: str, split: int, delta_sequence: int,
        ) -> tuple[int, ...]:
    """Build the one canonical integer record used by all projections."""
    # License text is metadata at the boundary; its bytes are represented by the
    # digest so the record itself remains a pure integer stream.
    license_digest = digest_bytes(license_id.encode("utf-8"))
    result: list[int] = [CAPSULE_SCHEMA]
    for value in (source_key, scope_key, version_key, parent_version_key):
        _pack(result, value)
    result.extend((language, modality))
    _pack(result, raw_digest)
    _pack_many(result, structural_units)
    _pack(result, authority_key)
    _pack(result, license_digest)
    result.extend((split, delta_sequence))
    return tuple(result)


def _identity_digest(record: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(hashlib.sha256(encode_integer_tuple(record)).digest())


@dataclass(frozen=True, slots=True)
class LearningInputCapsule:
    """一次新资料输入的唯一、不可变、来源化表达。"""

    source: SourceRef
    scope: ScopeIdentity
    version_key: tuple[int, ...]
    parent_version_key: tuple[int, ...]
    language: int
    modality: int
    raw_content_digest: tuple[int, ...]
    structural_units: tuple[tuple[int, ...], ...]
    authority_key: tuple[int, ...]
    license_id: str
    split: int
    delta_sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceRef):
            raise TypeError("capsule.source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("capsule.scope 必须是 ScopeIdentity")
        if self.scope.owner != self.source.owner or self.scope.versions != self.source.versions:
            raise LearningInputCapsuleError("capsule source 与 scope owner/version 不一致")
        if self.scope.source is not None and self.scope.source != self.source:
            raise LearningInputCapsuleError("capsule scope.source 与 source 不一致")
        _ints(self.version_key, label="capsule.version_key")
        _ints(self.parent_version_key, label="capsule.parent_version_key", empty=True)
        _positive(self.language, label="capsule.language")
        _positive(self.modality, label="capsule.modality")
        _digest(self.raw_content_digest, label="capsule.raw_content_digest")
        if not isinstance(self.structural_units, tuple):
            raise LearningInputCapsuleError("capsule.structural_units 必须是 tuple")
        for index, unit in enumerate(self.structural_units):
            _ints(unit, label=f"capsule.structural_units[{index}]")
        _ints(self.authority_key, label="capsule.authority_key")
        if (not isinstance(self.license_id, str)
                or not self.license_id or self.license_id.strip() != self.license_id):
            raise LearningInputCapsuleError("capsule.license_id 必须是非空文本")
        _positive(self.split, label="capsule.split")
        _positive(self.delta_sequence, label="capsule.delta_sequence")

    @property
    def canonical_record(self) -> tuple[int, ...]:
        """返回唯一的规范整数记录，不含 Python 对象地址或隐式状态。"""
        return _capsule_record(
            self.source.stable_key(), self.scope.stable_key(),
            self.version_key, self.parent_version_key, self.language,
            self.modality, self.raw_content_digest, self.structural_units,
            self.authority_key, self.license_id, self.split, self.delta_sequence)

    @property
    def identity_key(self) -> tuple[int, ...]:
        """返回 capsule 内容身份；完整记录仍保留用于碰撞/回放核验。"""
        return _identity_digest(self.canonical_record)

    def to_document(self) -> dict[str, Any]:
        """返回严格 JSON 交换对象；canonical_record 用于回读交叉核验。"""
        return {
            "schema": CAPSULE_SCHEMA,
            "source_key": list(self.source.stable_key()),
            "scope_key": list(self.scope.stable_key()),
            "version_key": list(self.version_key),
            "parent_version_key": list(self.parent_version_key),
            "language": self.language,
            "modality": self.modality,
            "raw_content_digest": list(self.raw_content_digest),
            "structural_units": [list(item) for item in self.structural_units],
            "authority_key": list(self.authority_key),
            "license_id": self.license_id,
            "split": self.split,
            "delta_sequence": self.delta_sequence,
            "canonical_record": list(self.canonical_record),
            "identity_key": list(self.identity_key),
        }

    @classmethod
    def from_document(cls, value: Any) -> "LearningInputCapsule":
        """严格从交换对象恢复并核验完整 capsule。"""
        if not isinstance(value, dict):
            raise LearningInputCapsuleError("capsule document 必须是 object")
        expected = {
            "schema", "source_key", "scope_key", "version_key",
            "parent_version_key", "language", "modality", "raw_content_digest",
            "structural_units", "authority_key", "license_id", "split",
            "delta_sequence", "canonical_record", "identity_key",
        }
        if set(value) != expected or value.get("schema") != CAPSULE_SCHEMA:
            raise LearningInputCapsuleError("capsule document 字段或 schema 漂移")
        try:
            source = SourceRef.from_stable_key(tuple(value["source_key"]))
            scope = ScopeIdentity.from_stable_key(tuple(value["scope_key"]))
            capsule = cls(
                source, scope,
                tuple(value["version_key"]), tuple(value["parent_version_key"]),
                value["language"], value["modality"],
                tuple(value["raw_content_digest"]),
                tuple(tuple(item) for item in value["structural_units"]),
                tuple(value["authority_key"]), value["license_id"],
                value["split"], value["delta_sequence"],
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise LearningInputCapsuleError("capsule document typed 字段非法") from exc
        if tuple(value["canonical_record"]) != capsule.canonical_record:
            raise LearningInputCapsuleError("capsule canonical_record 漂移")
        if tuple(value["identity_key"]) != capsule.identity_key:
            raise LearningInputCapsuleError("capsule identity_key 漂移")
        return capsule


@dataclass(frozen=True, slots=True)
class CoreDelta:
    """由 capsule 显式投影出的核心增量；不携带 Runtime payload。"""

    base_state_identity: tuple[int, ...]
    capsule: LearningInputCapsule
    status: int = STATUS_OBSERVED
    graph_diff: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _ints(self.base_state_identity, label="core_delta.base_state_identity")
        if not isinstance(self.capsule, LearningInputCapsule):
            raise TypeError("core_delta.capsule 类型错误")
        if self.status not in _STATUSES:
            raise LearningInputCapsuleError("core_delta.status 未注册")
        _ints(self.graph_diff, label="core_delta.graph_diff", empty=True)

    @property
    def delta_identity(self) -> tuple[int, ...]:
        return self.capsule.identity_key

    def stable_key(self) -> tuple[int, ...]:
        result = [PROJECTION_CORE]
        _pack(result, self.base_state_identity)
        _pack(result, self.delta_identity)
        result.append(self.status)
        _pack(result, self.graph_diff)
        _pack(result, self.capsule.canonical_record)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class LearningReplayReceipt:
    """双平面投影共用的纯整数回放凭据。"""

    projection_kind: int
    input_identity: tuple[int, ...]
    output_identity: tuple[int, ...]
    status: int
    replay_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.projection_kind not in {PROJECTION_CORE, PROJECTION_RUNTIME}:
            raise LearningInputCapsuleError("receipt.projection_kind 未注册")
        _ints(self.input_identity, label="receipt.input_identity")
        _ints(self.output_identity, label="receipt.output_identity")
        if self.status not in _STATUSES:
            raise LearningInputCapsuleError("receipt.status 未注册")
        _ints(self.replay_key, label="receipt.replay_key")

    @classmethod
    def from_core_delta(
            cls, delta: CoreDelta, *, output_identity: tuple[int, ...],
            replay_key: tuple[int, ...],
            ) -> "LearningReplayReceipt":
        if not isinstance(delta, CoreDelta):
            raise TypeError("receipt core delta 类型错误")
        return cls(PROJECTION_CORE, delta.delta_identity, output_identity,
                   delta.status, replay_key)

    @classmethod
    def from_runtime_event(
            cls, event: RuntimeMemoryEvent, *, replay_key: tuple[int, ...],
            status: int = STATUS_OBSERVED,
            ) -> "LearningReplayReceipt":
        if not isinstance(event, RuntimeMemoryEvent):
            raise TypeError("receipt runtime event 类型错误")
        return cls(PROJECTION_RUNTIME, event.capsule.identity_key,
                   event.event_key, status, replay_key)

    def stable_key(self) -> tuple[int, ...]:
        result = [CAPSULE_SCHEMA, self.projection_kind, self.status]
        for value in (self.input_identity, self.output_identity, self.replay_key):
            _pack(result, value)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class CoreLearningState:
    """绑定一个 scope 的 Core 增量消费台账。"""

    scope_key: tuple[int, ...]
    base_state_identity: tuple[int, ...]
    consumed_item_ledger: tuple[tuple[int, ...], ...] = ()
    deltas: tuple[CoreDelta, ...] = ()

    def __post_init__(self) -> None:
        _ints(self.scope_key, label="core_state.scope_key")
        _ints(self.base_state_identity, label="core_state.base_state_identity")
        if not isinstance(self.consumed_item_ledger, tuple):
            raise LearningInputCapsuleError("core_state ledger 必须是 tuple")
        for index, key in enumerate(self.consumed_item_ledger):
            _ints(key, label=f"core_state.ledger[{index}]")
        if tuple(sorted(self.consumed_item_ledger)) != self.consumed_item_ledger:
            raise LearningInputCapsuleError("core_state ledger 必须稳定排序")
        if len(set(self.consumed_item_ledger)) != len(self.consumed_item_ledger):
            raise LearningInputCapsuleError("core_state ledger 不得重复")
        if not isinstance(self.deltas, tuple) or any(
                not isinstance(item, CoreDelta) for item in self.deltas):
            raise LearningInputCapsuleError("core_state deltas 类型错误")
        identities = tuple(item.delta_identity for item in self.deltas)
        if len(set(identities)) != len(identities):
            raise LearningInputCapsuleError("core_state delta identity 不得重复")
        for item in self.deltas:
            if item.capsule.scope.stable_key() != self.scope_key:
                raise LearningInputCapsuleError("core_state 存在跨 scope delta")
            if item.base_state_identity != self.base_state_identity:
                raise LearningInputCapsuleError("core_state delta 基线漂移")
            if item.delta_identity not in self.consumed_item_ledger:
                raise LearningInputCapsuleError("core_state delta 未进入 consumed ledger")


def consume_core_delta(
        state: CoreLearningState, delta: CoreDelta,
        ) -> tuple[CoreLearningState, int]:
    """只消费未见 delta；重复幂等，内容冲突 fail closed。"""
    if not isinstance(state, CoreLearningState) or not isinstance(delta, CoreDelta):
        raise TypeError("consume_core_delta 类型错误")
    if delta.capsule.scope.stable_key() != state.scope_key:
        return state, ADMISSION_SCOPE_MISMATCH
    if delta.base_state_identity != state.base_state_identity:
        return state, ADMISSION_CONFLICT
    existing = tuple(item for item in state.deltas
                     if item.delta_identity == delta.delta_identity)
    if existing:
        if any(item.stable_key() == delta.stable_key() for item in existing):
            return state, ADMISSION_DUPLICATE
        return state, ADMISSION_CONFLICT
    if delta.delta_identity in state.consumed_item_ledger:
        return state, ADMISSION_CONFLICT
    ledger = tuple(sorted((*state.consumed_item_ledger, delta.delta_identity)))
    next_state = CoreLearningState(
        state.scope_key, state.base_state_identity, ledger,
        (*state.deltas, delta))
    return next_state, ADMISSION_ACCEPTED


@dataclass(frozen=True, slots=True)
class RuntimeMemoryEvent:
    """Runtime Memory 的 append-only 事件；revision/tombstone 不覆盖旧事件。"""

    capsule: LearningInputCapsule
    memory_item_key: tuple[int, ...]
    event_kind: int = EVENT_ASSERTION
    revision: int = 1
    supersedes_event_key: tuple[int, ...] = ()
    tombstone: bool = False
    conflict_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.capsule, LearningInputCapsule):
            raise TypeError("runtime_event.capsule 类型错误")
        _ints(self.memory_item_key, label="runtime_event.memory_item_key")
        _positive(self.event_kind, label="runtime_event.event_kind")
        _positive(self.revision, label="runtime_event.revision")
        _ints(self.supersedes_event_key, label="runtime_event.supersedes_event_key", empty=True)
        _ints(self.conflict_key, label="runtime_event.conflict_key", empty=True)
        if type(self.tombstone) is not bool:
            raise LearningInputCapsuleError("runtime_event.tombstone 必须是 bool")
        if self.revision == 1 and self.supersedes_event_key:
            raise LearningInputCapsuleError("首个 runtime event 不得 supersede")
        if self.revision > 1 and not self.supersedes_event_key:
            raise LearningInputCapsuleError("revision event 必须 supersede 前一事件")
        if self.tombstone and self.event_kind != EVENT_TOMBSTONE:
            raise LearningInputCapsuleError("tombstone 必须使用 EVENT_TOMBSTONE")
        if self.event_kind == EVENT_CONFLICT and not self.conflict_key:
            raise LearningInputCapsuleError("冲突 event 必须保留 conflict_key")

    @property
    def event_key(self) -> tuple[int, ...]:
        result = [PROJECTION_RUNTIME, self.event_kind, self.revision,
                  1 if self.tombstone else 0]
        for value in (self.memory_item_key, self.supersedes_event_key,
                      self.conflict_key, self.capsule.identity_key):
            _pack(result, value)
        _pack(result, self.capsule.canonical_record)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class RuntimeMemoryState:
    """绑定一个 scope 的 Runtime Memory 事件账本。"""

    scope_key: tuple[int, ...]
    events: tuple[RuntimeMemoryEvent, ...] = ()

    def __post_init__(self) -> None:
        _ints(self.scope_key, label="runtime_state.scope_key")
        if not isinstance(self.events, tuple) or any(
                not isinstance(item, RuntimeMemoryEvent) for item in self.events):
            raise LearningInputCapsuleError("runtime_state.events 类型错误")
        if any(item.capsule.scope.stable_key() != self.scope_key for item in self.events):
            raise LearningInputCapsuleError("runtime_state 存在跨 scope event")
        event_keys = tuple(item.event_key for item in self.events)
        if len(set(event_keys)) != len(event_keys):
            raise LearningInputCapsuleError("runtime_state event 不得重复")


def append_runtime_event(
        state: RuntimeMemoryState, event: RuntimeMemoryEvent,
        ) -> tuple[RuntimeMemoryState, int]:
    """追加事件并保留重复/冲突证据；不覆盖旧版本。"""
    if not isinstance(state, RuntimeMemoryState) or not isinstance(event, RuntimeMemoryEvent):
        raise TypeError("append_runtime_event 类型错误")
    if event.capsule.scope.stable_key() != state.scope_key:
        return state, ADMISSION_SCOPE_MISMATCH
    same_item = tuple(item for item in state.events
                      if item.memory_item_key == event.memory_item_key)
    if any(item.event_key == event.event_key for item in same_item):
        return state, ADMISSION_DUPLICATE
    same_revision = tuple(item for item in same_item
                          if item.revision == event.revision)
    if same_revision:
        conflict = RuntimeMemoryEvent(
            event.capsule, event.memory_item_key, EVENT_CONFLICT,
            event.revision, event.supersedes_event_key, False,
            event.event_key)
        return RuntimeMemoryState(state.scope_key, (*state.events, conflict)), ADMISSION_CONFLICT
    max_revision = max((item.revision for item in same_item), default=0)
    if event.revision != max_revision + 1:
        return state, ADMISSION_REVISION_GAP
    latest = tuple(item for item in same_item if item.revision == max_revision)
    if same_item:
        # A competing same-revision event makes the chain ambiguous.  It is
        # preserved above, but no later revision may silently choose a branch.
        if len(latest) != 1 or event.supersedes_event_key != latest[0].event_key:
            return state, ADMISSION_CONFLICT
    return RuntimeMemoryState(state.scope_key, (*state.events, event)), ADMISSION_ACCEPTED


@dataclass(frozen=True, slots=True)
class PromotionRequest:
    """Memory -> Core 的显式晋升请求；没有此值就没有晋升路径。"""

    runtime_event_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    evidence_keys: tuple[tuple[int, ...], ...]
    authority_key: tuple[int, ...]
    replay_key: tuple[int, ...]
    consent_key: tuple[int, ...]

    def __post_init__(self) -> None:
        _ints(self.runtime_event_key, label="promotion.runtime_event_key")
        if not isinstance(self.source, SourceRef) or not isinstance(self.scope, ScopeIdentity):
            raise TypeError("promotion source/scope 类型错误")
        if self.scope.owner != self.source.owner or self.scope.versions != self.source.versions:
            raise LearningInputCapsuleError("promotion source/scope owner/version 不一致")
        if self.scope.source != self.source:
            raise LearningInputCapsuleError(
                "promotion 必须绑定带 source 的 scope，禁止只凭 owner 晋升")
        if not isinstance(self.evidence_keys, tuple) or not self.evidence_keys:
            raise LearningInputCapsuleError("promotion.evidence_keys 不得为空")
        for index, key in enumerate(self.evidence_keys):
            _ints(key, label=f"promotion.evidence_keys[{index}]")
        _ints(self.authority_key, label="promotion.authority_key")
        _ints(self.replay_key, label="promotion.replay_key")
        _ints(self.consent_key, label="promotion.consent_key")

    def stable_key(self) -> tuple[int, ...]:
        result: list[int] = [CAPSULE_SCHEMA]
        for value in (self.runtime_event_key, self.source.stable_key(),
                      self.scope.stable_key(), self.authority_key,
                      self.replay_key, self.consent_key):
            _pack(result, value)
        _pack_many(result, self.evidence_keys)
        return tuple(result)


__all__ = [
    "ADMISSION_ACCEPTED", "ADMISSION_CONFLICT", "ADMISSION_DUPLICATE",
    "ADMISSION_REVISION_GAP", "ADMISSION_SCOPE_MISMATCH", "CAPSULE_SCHEMA",
    "CoreDelta", "CoreLearningState", "EVENT_ASSERTION", "EVENT_CONFLICT",
    "EVENT_REVISION", "EVENT_TOMBSTONE", "LearningInputCapsule",
    "LearningInputCapsuleError", "LearningReplayReceipt", "PromotionRequest", "PROJECTION_CORE",
    "PROJECTION_RUNTIME", "RuntimeMemoryEvent", "RuntimeMemoryState",
    "STATUS_CONFLICTED", "STATUS_LEARNED", "STATUS_OBSERVED", "STATUS_REJECTED",
    "append_runtime_event", "consume_core_delta", "digest_bytes",
]
