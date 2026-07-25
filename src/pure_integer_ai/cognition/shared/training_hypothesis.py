"""H-00/H-04 与断奶训练 Core 历史之间的持久化适配。

训练历史和断奶后 Memory 使用同一领域事件接口，但物理表、协议身份和恢复入口完全
分离。PH2 因此可以恢复候选学习状态，而不创建或写入任何 Memory 对象。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EvidenceRecord,
    HypothesisEventSink,
    HypothesisKey,
    HypothesisLedger,
    HypothesisTransition,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ResolverDecision,
)
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.training_candidate_event import (
    TrainingCandidateEventIntegrityError,
    TrainingCandidateEventRecord,
    TrainingCandidateEventRecordStore,
    decode_integer_stream,
    encode_integer_stream,
)


TRAINING_HISTORY_HYPOTHESIS = 1
TRAINING_HISTORY_EVIDENCE = 2
TRAINING_HISTORY_TRANSITION = 3
TRAINING_HISTORY_DECISION = 4
_HISTORY_KINDS = frozenset({
    TRAINING_HISTORY_HYPOTHESIS,
    TRAINING_HISTORY_EVIDENCE,
    TRAINING_HISTORY_TRANSITION,
    TRAINING_HISTORY_DECISION,
})
_ENVELOPE_VERSION = 2
_PROTOCOL_HASHER = Hasher("training_hypothesis.protocol.v1")
_EVENT_HASHER = Hasher("training_hypothesis.event.v1")


class TrainingHypothesisHistoryError(RuntimeError):
    """Core 训练历史无法无损追加或恢复。"""


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键添加长度边界。"""
    return len(value), *value


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _take(
        values: tuple[int, ...], cursor: int, *, label: str,
        ) -> tuple[tuple[int, ...], int]:
    """从长度前缀整数流读取一个非空键段。"""
    if cursor >= len(values):
        raise TrainingHypothesisHistoryError(f"训练历史缺少 {label} 长度")
    size = values[cursor]
    cursor += 1
    if type(size) is not int or size <= 0 or cursor + size > len(values):
        raise TrainingHypothesisHistoryError(f"训练历史 {label} 长度非法")
    return values[cursor:cursor + size], cursor + size


@dataclass(frozen=True)
class TrainingHypothesisHistoryProtocol:
    """声明一个 Core 训练候选日志的协议、kind、scope 和 aggregate 来源。"""

    namespace: tuple[int, ...]
    hypothesis_kind: tuple[int, ...]
    aggregate_source: SourceRef
    aggregate_scope: ScopeIdentity

    def __post_init__(self) -> None:
        """核验协议边界完整且 aggregate scope 指向同一来源。"""
        _strict_key(self.namespace, label="training history namespace")
        _strict_key(
            self.hypothesis_kind,
            label="training history hypothesis kind",
        )
        if not isinstance(self.aggregate_source, SourceRef):
            raise TypeError("training history aggregate_source 类型错误")
        if not isinstance(self.aggregate_scope, ScopeIdentity):
            raise TypeError("training history aggregate_scope 类型错误")
        if self.aggregate_scope.source != self.aggregate_source:
            raise ValueError("training history scope 必须指向 aggregate source")

    def stable_key(self) -> tuple[int, ...]:
        """返回不依赖物理 backend 或运行时 space_id 的完整协议键。"""
        return (
            _ENVELOPE_VERSION,
            *_packed(self.namespace),
            *_packed(self.hypothesis_kind),
            *_packed(self.aggregate_source.stable_key()),
            *_packed(self.aggregate_scope.stable_key()),
        )

    def accepts(self, hypothesis: HypothesisKey) -> bool:
        """判断候选是否严格属于本协议，不进行 owner 或 scope 降级。"""
        return (
            isinstance(hypothesis, HypothesisKey)
            and hypothesis.hypothesis_kind == self.hypothesis_kind
            and hypothesis.observation == self.aggregate_source
            and hypothesis.scope == self.aggregate_scope
        )


@dataclass(frozen=True)
class TrainingHistoryEntry:
    """一条已回验协议和物理信封的训练领域事件。"""

    event_hash: int
    event_kind: int
    event_seq: int
    record_key: tuple[int, ...]


class TrainingCandidateHistoryLog:
    """绑定一个 Core 空间的训练候选 append-only 日志。"""

    def __init__(
            self,
            backend: StorageBackend,
            core_space_id: int,
            ) -> None:
        """绑定 backend 和 Core 空间，不创建 Memory 或 Companion facade。"""
        if not isinstance(backend, StorageBackend):
            raise TypeError("training history backend 类型错误")
        assert_int(core_space_id, _where="training history core_space_id")
        if type(core_space_id) is not int or core_space_id <= 0:
            raise ValueError("training history core_space_id 必须为正整数")
        self.backend = backend
        self.core_space_id = core_space_id
        self._records = TrainingCandidateEventRecordStore(backend)

    def append(
            self,
            protocol: TrainingHypothesisHistoryProtocol,
            event_kind: int,
            event_seq: int,
            record_key: tuple[int, ...],
            ) -> TrainingHistoryEntry:
        """把完整领域事件按版本化协议幂等追加到 Core 历史。"""
        if not isinstance(protocol, TrainingHypothesisHistoryProtocol):
            raise TypeError("training history protocol 类型错误")
        if type(event_kind) is not int or event_kind not in _HISTORY_KINDS:
            raise ValueError("training history event kind 未注册")
        assert_int(event_seq, _where="training history event_seq")
        if type(event_seq) is not int or event_seq < 0:
            raise ValueError("training history event_seq 必须为非负整数")
        record_key = _strict_key(record_key, label="training history record")
        protocol_key = protocol.stable_key()
        envelope = (
            _ENVELOPE_VERSION,
            *_packed(protocol_key),
            event_kind,
            event_seq,
            *_packed(record_key),
        )
        protocol_hash = _PROTOCOL_HASHER.h63(protocol_key) or 1
        event_hash = _EVENT_HASHER.h63(envelope) or 1
        encoded = encode_integer_stream(envelope)
        record = TrainingCandidateEventRecord(
            event_hash,
            self.core_space_id,
            protocol_hash,
            event_kind,
            event_seq,
            len(envelope),
            len(encoded),
        )
        self._records.add(record, encoded)
        return TrainingHistoryEntry(
            event_hash, event_kind, event_seq, record_key)

    def entries(
            self,
            protocol: TrainingHypothesisHistoryProtocol,
            *,
            event_kind: int | None = None,
            ) -> tuple[TrainingHistoryEntry, ...]:
        """读取并逐条回验完整协议、事件种类和信封顺序字段。"""
        if not isinstance(protocol, TrainingHypothesisHistoryProtocol):
            raise TypeError("training history protocol 类型错误")
        if (event_kind is not None
                and (type(event_kind) is not int
                     or event_kind not in _HISTORY_KINDS)):
            raise ValueError("training history event kind 未注册")
        protocol_key = protocol.stable_key()
        protocol_hash = _PROTOCOL_HASHER.h63(protocol_key) or 1
        result = []
        for record in self._records.query(
                space_id=self.core_space_id,
                protocol_hash=protocol_hash,
                event_kind=event_kind):
            envelope = decode_integer_stream(
                self._records.read_payload(record))
            if len(envelope) != record.original_size:
                raise TrainingCandidateEventIntegrityError(
                    "训练候选事件原始 payload 长度与信封漂移")
            expected_hash = _EVENT_HASHER.h63(envelope) or 1
            if expected_hash != record.event_hash:
                raise TrainingCandidateEventIntegrityError(
                    "训练候选 event_hash 与完整事件键不一致")
            stored_protocol, stored_kind, stored_seq, stored_key = (
                self._decode(envelope))
            if stored_protocol != protocol_key:
                raise TrainingCandidateEventIntegrityError(
                    "训练候选 protocol_hash 发生完整键碰撞")
            if stored_kind != record.event_kind:
                raise TrainingCandidateEventIntegrityError(
                    "训练候选事件种类与信封漂移")
            if stored_seq != record.event_seq:
                raise TrainingCandidateEventIntegrityError(
                    "训练候选事件逻辑序与信封漂移")
            result.append(TrainingHistoryEntry(
                record.event_hash,
                record.event_kind,
                record.event_seq,
                stored_key,
            ))
        return tuple(sorted(
            result,
            key=lambda item: (
                item.event_seq,
                item.event_kind,
                item.event_hash,
            ),
        ))

    @staticmethod
    def _decode(
            envelope: tuple[int, ...],
            ) -> tuple[tuple[int, ...], int, int, tuple[int, ...]]:
        """从固定版本 envelope 恢复协议、事件种类和领域记录键。"""
        if not envelope or envelope[0] != _ENVELOPE_VERSION:
            raise TrainingHypothesisHistoryError("训练历史 envelope 版本非法")
        protocol_key, cursor = _take(envelope, 1, label="protocol")
        if cursor >= len(envelope):
            raise TrainingHypothesisHistoryError("训练历史缺少事件种类")
        event_kind = envelope[cursor]
        cursor += 1
        if type(event_kind) is not int or event_kind not in _HISTORY_KINDS:
            raise TrainingHypothesisHistoryError("训练历史事件种类未注册")
        if cursor >= len(envelope):
            raise TrainingHypothesisHistoryError("训练历史缺少事件逻辑序")
        event_seq = envelope[cursor]
        cursor += 1
        if type(event_seq) is not int or event_seq < 0:
            raise TrainingHypothesisHistoryError("训练历史事件逻辑序非法")
        record_key, cursor = _take(envelope, cursor, label="record")
        if cursor != len(envelope):
            raise TrainingHypothesisHistoryError("训练历史 envelope 含尾随字段")
        return protocol_key, event_kind, event_seq, record_key


class TrainingHypothesisEventSink(HypothesisEventSink):
    """把 H-00/H-04 事件直接保存到训练 Core 历史。"""

    def __init__(
            self,
            history: TrainingCandidateHistoryLog,
            protocol: TrainingHypothesisHistoryProtocol,
            ) -> None:
        """绑定 Core 日志与唯一候选协议。"""
        if not isinstance(history, TrainingCandidateHistoryLog):
            raise TypeError("training hypothesis history 类型错误")
        if not isinstance(protocol, TrainingHypothesisHistoryProtocol):
            raise TypeError("training hypothesis protocol 类型错误")
        self.history = history
        self.protocol = protocol

    def append_hypothesis(self, hypothesis: HypothesisKey) -> None:
        """追加候选声明，并拒绝其他 aggregate 协议混入。"""
        self._require_hypothesis(hypothesis)
        self.history.append(
            self.protocol,
            TRAINING_HISTORY_HYPOTHESIS,
            0,
            hypothesis.stable_key(),
        )

    def append_evidence(self, evidence: EvidenceRecord) -> None:
        """追加完整 Evidence，不把 verifier 结果合成为标量。"""
        if not isinstance(evidence, EvidenceRecord):
            raise TypeError("training history Evidence 类型错误")
        self._require_hypothesis(evidence.hypothesis)
        self.history.append(
            self.protocol,
            TRAINING_HISTORY_EVIDENCE,
            evidence.timestamp_seq,
            evidence.stable_key(),
        )

    def append_transition(self, transition: HypothesisTransition) -> None:
        """追加 H-00 lifecycle 转换，并核验 replacement 仍属同一协议。"""
        if not isinstance(transition, HypothesisTransition):
            raise TypeError("training history transition 类型错误")
        self._require_hypothesis(transition.hypothesis)
        if transition.replacement is not None:
            self._require_hypothesis(transition.replacement)
        self.history.append(
            self.protocol,
            TRAINING_HISTORY_TRANSITION,
            transition.timestamp_seq,
            transition.stable_key(),
        )

    def append_decision(self, decision: ResolverDecision) -> None:
        """追加完整 H-04 决策链，不只保存 adopted 终态。"""
        if not isinstance(decision, ResolverDecision):
            raise TypeError("training history decision 类型错误")
        for candidate in decision.candidates:
            self._require_hypothesis(candidate.hypothesis)
        self.history.append(
            self.protocol,
            TRAINING_HISTORY_DECISION,
            decision.timestamp_seq,
            decision.stable_key(),
        )

    def hypotheses(self) -> tuple[HypothesisKey, ...]:
        """恢复本协议全部候选声明，重复身份只允许物理幂等。"""
        values = tuple(
            HypothesisKey.from_stable_key(item.record_key)
            for item in self.history.entries(
                self.protocol,
                event_kind=TRAINING_HISTORY_HYPOTHESIS,
            )
        )
        if len(set(values)) != len(values):
            raise TrainingHypothesisHistoryError(
                "训练历史含重复 Hypothesis 声明")
        for hypothesis in values:
            self._require_hypothesis(hypothesis)
        return tuple(sorted(values, key=HypothesisKey.stable_key))

    def load_ledger(self, *, attach_sink: bool) -> HypothesisLedger:
        """从 Core 历史重建 H-00 ledger，并可把后续追加重新绑定当前日志。"""
        if type(attach_sink) is not bool:
            raise TypeError("attach_sink 必须是严格 bool")
        ledger = HypothesisLedger()
        hypotheses = self.hypotheses()
        for hypothesis in hypotheses:
            ledger.register(hypothesis)
        evidence = [
            EvidenceRecord.from_stable_key(item.record_key)
            for item in self.history.entries(
                self.protocol,
                event_kind=TRAINING_HISTORY_EVIDENCE,
            )
        ]
        for item in self._evidence_topological(evidence):
            self._require_hypothesis(item.hypothesis)
            ledger.append_evidence(item)
        transitions = tuple(sorted(
            (HypothesisTransition.from_stable_key(item.record_key)
             for item in self.history.entries(
                 self.protocol,
                 event_kind=TRAINING_HISTORY_TRANSITION,
             )),
            key=lambda item: (item.timestamp_seq, item.event_id),
        ))
        for item in transitions:
            self._require_hypothesis(item.hypothesis)
            ledger.append_transition(item)
        return ledger.with_sink(self) if attach_sink else ledger

    def load_decisions(self) -> tuple[ResolverDecision, ...]:
        """恢复本协议完整 H-04 决策链，前驱和内容由 resolver 再次核验。"""
        decisions = tuple(sorted(
            (ResolverDecision.from_stable_key(item.record_key)
             for item in self.history.entries(
                 self.protocol,
                 event_kind=TRAINING_HISTORY_DECISION,
             )),
            key=lambda item: (item.timestamp_seq, item.decision_id),
        ))
        for decision in decisions:
            for candidate in decision.candidates:
                self._require_hypothesis(candidate.hypothesis)
        return decisions

    def state_key(self) -> tuple:
        """返回协议和当前 Core 训练历史的完整可比较状态。"""
        return (
            self.protocol.stable_key(),
            tuple((
                item.event_hash,
                item.event_kind,
                item.event_seq,
                item.record_key,
            ) for item in self.history.entries(self.protocol)),
        )

    def _require_hypothesis(self, hypothesis: HypothesisKey) -> None:
        """要求候选精确属于当前 kind、scope 和 aggregate 来源。"""
        if not self.protocol.accepts(hypothesis):
            raise TrainingHypothesisHistoryError(
                "H-00/H-04 事件不属于当前训练历史协议")

    @staticmethod
    def _evidence_topological(
            records: list[EvidenceRecord],
            ) -> tuple[EvidenceRecord, ...]:
        """按 supersede 依赖排序 Evidence，同层按逻辑序和 id 稳定排序。"""
        pending = {item.evidence_id: item for item in records}
        if len(pending) != len(records):
            raise TrainingHypothesisHistoryError(
                "训练历史含重复 evidence_id")
        emitted: set[int] = set()
        ordered: list[EvidenceRecord] = []
        while pending:
            ready = sorted(
                (item for item in pending.values()
                 if item.supersedes_evidence_id == 0
                 or item.supersedes_evidence_id in emitted),
                key=lambda item: (item.timestamp_seq, item.evidence_id),
            )
            if not ready:
                raise TrainingHypothesisHistoryError(
                    "训练 Evidence supersede 链存在孤儿或环")
            for item in ready:
                ordered.append(item)
                emitted.add(item.evidence_id)
                del pending[item.evidence_id]
        return tuple(ordered)


__all__ = [
    "TRAINING_HISTORY_DECISION",
    "TRAINING_HISTORY_EVIDENCE",
    "TRAINING_HISTORY_HYPOTHESIS",
    "TRAINING_HISTORY_TRANSITION",
    "TrainingCandidateHistoryLog",
    "TrainingHistoryEntry",
    "TrainingHypothesisEventSink",
    "TrainingHypothesisHistoryError",
    "TrainingHypothesisHistoryProtocol",
]
