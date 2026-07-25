"""把 M-07 resolved Memory 证据无损投影到 G-00 候选的共享协议。

本模块只保存纯整数 SourceRecord 身份和 Companion metadata，不携带原文，也不把
Memory Evidence 伪装成 H-00 ``EvidenceRecord``。目标命题绑定由调用方注入理由和
trace，实际采用与 Use 仍由 G-05 后续运行边界处理。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.memory_event import MemoryLinkedRef
from pure_integer_ai.cognition.shared.memory_resolver import (
    RESOLUTION_ORIGIN_MEMORY,
    MemorySourceTrace,
    ResolvedCandidate,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.source_record import SourceRecordStorage


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为开放整数键增加长度边界。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验来源和绑定 trace 使用非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


@dataclass(frozen=True)
class MemoryGenerationSource:
    """一条 Memory 来源分账及其可回查 SourceRecord 的纯整数身份。"""

    trace: MemorySourceTrace
    source_record_hash: int
    text_hash: int
    codepoint_count: int
    batch_id: int
    companion_type_hash: int
    companion_name_hash: int
    companion_assoc_id: int

    def __post_init__(self) -> None:
        """核验来源 trace 和断奶后完整 SourceRecord metadata。"""
        if not isinstance(self.trace, MemorySourceTrace):
            raise TypeError("Memory generation source trace 类型错误")
        values = (
            self.source_record_hash,
            self.text_hash,
            self.codepoint_count,
            self.batch_id,
            self.companion_type_hash,
            self.companion_name_hash,
            self.companion_assoc_id,
        )
        assert_int(*values, _where="MemoryGenerationSource")
        if any(type(value) is not int for value in values):
            raise ValueError("Memory generation source 必须使用严格整数")
        if (self.source_record_hash <= 0
                or self.text_hash < 0
                or self.codepoint_count < 0
                or self.batch_id <= 0
                or self.companion_type_hash <= 0
                or self.companion_name_hash <= 0
                or self.companion_assoc_id <= 0):
            raise ValueError("Memory generation source 缺少完整来源或 Companion 身份")

    @classmethod
    def from_record(
            cls,
            trace: MemorySourceTrace,
            record: SourceRecordStorage,
            ) -> "MemoryGenerationSource":
        """从同一 SourceRef 的完整 SourceRecord 建立无原文热路径身份。"""
        if not isinstance(trace, MemorySourceTrace):
            raise TypeError("Memory source trace 类型错误")
        if not isinstance(record, SourceRecordStorage):
            raise TypeError("SourceRecord 类型错误")
        if record.source_key != trace.source.stable_key():
            raise ValueError("SourceRecord 与 Memory 来源 trace 漂移")
        if not record.metadata_complete:
            raise ValueError("Memory generation 来源缺少完整许可和 Companion metadata")
        return cls(
            trace,
            record.source_hash,
            record.text_hash,
            record.codepoint_count,
            record.batch_id,
            record.companion_type_hash,
            record.companion_name_hash,
            record.companion_assoc_id,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回来源分账及 SourceRecord/Companion 的完整整数身份。"""
        return (
            *_packed(self.trace.stable_key()),
            self.source_record_hash,
            self.text_hash,
            self.codepoint_count,
            self.batch_id,
            self.companion_type_hash,
            self.companion_name_hash,
            self.companion_assoc_id,
        )


@dataclass(frozen=True)
class MemoryGenerationEvidence:
    """一个 resolved Memory 候选对当前一等命题的来源化支持或反驳。"""

    candidate: ResolvedCandidate
    target: BoundProposition
    binding_reason: ObjectIdentity
    binding_trace: tuple[int, ...]
    sources: tuple[MemoryGenerationSource, ...]

    def __post_init__(self) -> None:
        """核验 M-07 身份、命题绑定和 SourceRecord 分账一一对应。"""
        if not isinstance(self.candidate, ResolvedCandidate):
            raise TypeError("Memory generation candidate 类型错误")
        if self.candidate.origin_kind != RESOLUTION_ORIGIN_MEMORY:
            raise ValueError("Memory generation evidence 只接受 Memory 候选")
        if not isinstance(self.target, BoundProposition):
            raise TypeError("Memory generation target 必须是 BoundProposition")
        if (not isinstance(self.binding_reason, ObjectIdentity)
                or self.binding_reason.object_kind != OBJECT_MINIMAL_INSTRUCTION):
            raise ValueError("Memory generation binding reason 必须是 MinimalInstruction")
        _strict_key(self.binding_trace, label="Memory generation binding trace")
        if (not isinstance(self.sources, tuple)
                or any(not isinstance(item, MemoryGenerationSource)
                       for item in self.sources)):
            raise TypeError("Memory generation sources 类型错误")
        keys = tuple(item.stable_key() for item in self.sources)
        if not keys or keys != tuple(sorted(set(keys))):
            raise ValueError("Memory generation sources 必须非空、唯一且稳定有序")
        if tuple(item.trace for item in self.sources) != (
                self.candidate.memory_source_traces):
            raise ValueError("Memory generation SourceRecord 未逐项覆盖 M-07 来源分账")

    @property
    def state(self) -> LogicEvidenceState:
        """从真实 Memory 来源立场导出 support/refute 四态。"""
        return LogicEvidenceState(
            any(item.trace.stance == EVIDENCE_SUPPORT for item in self.sources),
            any(item.trace.stance == EVIDENCE_REFUTE for item in self.sources),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 resolved 候选、目标绑定和全部来源记录身份。"""
        result = [
            *_packed(self.candidate.stable_key()),
            *_packed(self.target.stable_key()),
            *_packed(self.binding_reason.stable_key()),
            *_packed(self.binding_trace),
            len(self.sources),
        ]
        for source in self.sources:
            result.extend(_packed(source.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class MemoryGenerationUseCommit:
    """一个已选 Memory evidence 对 A-10 processing 和 M-08 Use 的提交链接。"""

    candidate_key: tuple[int, ...]
    evidence_key: tuple[int, ...]
    processing_key: tuple[int, ...]
    use_ref: MemoryObjectRef

    def __post_init__(self) -> None:
        """核验候选、证据、处理 trace 和 Use 引用均完整可回查。"""
        _strict_key(self.candidate_key, label="Memory generation candidate key")
        _strict_key(self.evidence_key, label="Memory generation evidence key")
        _strict_key(self.processing_key, label="Memory generation processing key")
        if not isinstance(self.use_ref, MemoryObjectRef):
            raise TypeError("Memory generation use_ref 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回 G-01 候选、Memory evidence、A-10 与 M-08 的完整链接。"""
        return (
            *_packed(self.candidate_key),
            *_packed(self.evidence_key),
            *_packed(self.processing_key),
            *_packed(self.use_ref.stable_key()),
        )


@dataclass(frozen=True)
class MemoryGenerationCommitReport:
    """同次 G-01/G-03 后形成的零个或多个 Memory Use 提交报告。"""

    selection_key: tuple[int, ...]
    generation_key: tuple[int, ...]
    commits: tuple[MemoryGenerationUseCommit, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验报告绑定同次选择和生成，且每条提交互不重复。"""
        _strict_key(self.selection_key, label="Memory generation selection key")
        _strict_key(self.generation_key, label="Memory generation execution key")
        if (not isinstance(self.commits, tuple)
                or any(not isinstance(item, MemoryGenerationUseCommit)
                       for item in self.commits)):
            raise TypeError("Memory generation commits 类型错误")
        keys = tuple(item.stable_key() for item in self.commits)
        if len(set(keys)) != len(keys):
            raise ValueError("Memory generation commit 不得重复")
        _strict_key(self.trace, label="Memory generation commit trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回同次选择、生成、全部 Use 提交和 route trace。"""
        result = [
            *_packed(self.selection_key),
            *_packed(self.generation_key),
            len(self.commits),
        ]
        for commit in self.commits:
            result.extend(_packed(commit.stable_key()))
        result.extend(_packed(self.trace))
        return tuple(result)


@dataclass(frozen=True)
class MemoryGenerationOutcomeCommit:
    """一个 G-04 分维信号对 exact M-08 Use 的持久化结果链接。"""

    candidate_key: tuple[int, ...]
    use_ref: MemoryObjectRef
    outcome_kind: MemoryLinkedRef
    outcome_ref: MemoryLinkedRef
    signal_key: tuple[int, ...]
    event_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验候选、Use、分型引用和内容引用均可独立回查。"""
        _strict_key(self.candidate_key, label="Memory outcome candidate key")
        if not isinstance(self.use_ref, MemoryObjectRef):
            raise TypeError("Memory outcome use_ref 类型错误")
        if not isinstance(self.outcome_kind, MemoryLinkedRef):
            raise TypeError("Memory outcome kind 必须是一等引用")
        if not isinstance(self.outcome_ref, MemoryLinkedRef):
            raise TypeError("Memory outcome ref 必须是一等引用")
        _strict_key(self.signal_key, label="Memory outcome signal key")
        _strict_key(self.event_key, label="Memory outcome event key")

    def stable_key(self) -> tuple[int, ...]:
        """返回候选、Use、分维语义和持久化事件的完整链接。"""
        return (
            *_packed(self.candidate_key),
            *_packed(self.use_ref.stable_key()),
            *_packed(self.outcome_kind.stable_key()),
            *_packed(self.outcome_ref.stable_key()),
            *_packed(self.signal_key),
            *_packed(self.event_key),
        )


@dataclass(frozen=True)
class MemoryGenerationOutcomeReport:
    """同次选择、生成和 G-04 复核形成的逐 Use 分维 outcome 报告。"""

    selection_key: tuple[int, ...]
    generation_key: tuple[int, ...]
    postcheck_key: tuple[int, ...]
    outcomes: tuple[MemoryGenerationOutcomeCommit, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验上游内容引用完整，且同一 Use/outcome kind 不得重复。"""
        _strict_key(self.selection_key, label="Memory outcome selection key")
        _strict_key(self.generation_key, label="Memory outcome generation key")
        _strict_key(self.postcheck_key, label="Memory outcome postcheck key")
        if (not isinstance(self.outcomes, tuple)
                or any(not isinstance(item, MemoryGenerationOutcomeCommit)
                       for item in self.outcomes)):
            raise TypeError("Memory generation outcomes 类型错误")
        pairs = tuple(
            (item.use_ref.stable_key(), item.outcome_kind.stable_key())
            for item in self.outcomes
        )
        if len(set(pairs)) != len(pairs):
            raise ValueError("同一 Use 的同类 Memory outcome 不得重复")
        _strict_key(self.trace, label="Memory outcome report trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回三段上游引用、逐维 outcome 和桥接 trace。"""
        result = [
            *_packed(self.selection_key),
            *_packed(self.generation_key),
            *_packed(self.postcheck_key),
            len(self.outcomes),
        ]
        for outcome in self.outcomes:
            result.extend(_packed(outcome.stable_key()))
        result.extend(_packed(self.trace))
        return tuple(result)


__all__ = [
    "MemoryGenerationCommitReport",
    "MemoryGenerationEvidence",
    "MemoryGenerationOutcomeCommit",
    "MemoryGenerationOutcomeReport",
    "MemoryGenerationSource",
    "MemoryGenerationUseCommit",
]
