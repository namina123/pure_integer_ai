"""不绑定 Memory 物理表的 Hypothesis/Evidence 领域协议。

本模块把证据立场与候选生命周期拆开：前者由未被替代的 append-only Evidence
派生，后者只决定候选是否仍可被消费者采用。具体语言、关系和任务含义全部由调用方
注入的整数 kind/key 表达，本层不写死领域候选。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_HYPOTHESIS,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int

EVIDENCE_SUPPORT = 1
EVIDENCE_REFUTE = 2
EVIDENCE_UNKNOWN = 3
_EVIDENCE_STANCES = frozenset({
    EVIDENCE_SUPPORT,
    EVIDENCE_REFUTE,
    EVIDENCE_UNKNOWN,
})

EPISTEMIC_UNKNOWN = 1
EPISTEMIC_SUPPORTED = 2
EPISTEMIC_REFUTED = 3
EPISTEMIC_CONFLICTED = 4

LIFECYCLE_ACTIVE = 1
LIFECYCLE_SUPERSEDED = 2
LIFECYCLE_ARCHIVED = 3
_LIFECYCLE_STATES = frozenset({
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SUPERSEDED,
    LIFECYCLE_ARCHIVED,
})

_HYPOTHESIS_KEY_VERSION = 1
_TRANSITION_KEY_VERSION = 1


def _strict_tuple(value, *, where: str) -> tuple[int, ...]:
    """校验开放整数键，禁止字符串或 bool 混入稳定身份。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _positive(value: int, *, where: str) -> int:
    """校验元定义 kind 和事件 id 为严格正整数。"""
    assert_int(value, _where=where)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{where} 必须为严格正整数")
    return value


def _nonnegative(value: int, *, where: str) -> int:
    """校验逻辑序和可选引用为严格非负整数。"""
    assert_int(value, _where=where)
    if type(value) is not int or value < 0:
        raise ValueError(f"{where} 必须为非负整数")
    return value


@dataclass(frozen=True, order=True)
class HypothesisKey:
    """一个来源化候选的 kind、竞争组、scope 和稳定候选键。"""

    hypothesis_kind: tuple[int, ...]
    candidate_key: tuple[int, ...]
    competition_key: tuple[int, ...]
    scope: ScopeIdentity
    observation: SourceRef

    def __post_init__(self) -> None:
        _strict_tuple(
            self.hypothesis_kind, where="HypothesisKey.hypothesis_kind")
        _strict_tuple(self.candidate_key, where="HypothesisKey.candidate_key")
        _strict_tuple(
            self.competition_key, where="HypothesisKey.competition_key")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("HypothesisKey.scope 必须是 ScopeIdentity")
        if not isinstance(self.observation, SourceRef):
            raise TypeError("HypothesisKey.observation 必须是 SourceRef")
        if (self.scope.owner != self.observation.owner
                or self.scope.versions != self.observation.versions):
            raise ValueError("Hypothesis scope 与 observation owner/version 不一致")
        if (self.scope.source is not None
                and self.scope.source != self.observation):
            raise ValueError("来源化 scope 必须指向同一 observation")

    def stable_key(self) -> tuple[int, ...]:
        """返回可由训练状态或 Memory adapter 保存的完整整数键。"""
        scope_key = self.scope.stable_key()
        observation_key = self.observation.stable_key()
        return (
            _HYPOTHESIS_KEY_VERSION,
            len(self.hypothesis_kind),
            *self.hypothesis_kind,
            len(self.candidate_key),
            *self.candidate_key,
            len(self.competition_key),
            *self.competition_key,
            len(scope_key),
            *scope_key,
            len(observation_key),
            *observation_key,
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "HypothesisKey":
        """从完整整数键恢复候选，拒绝截断、尾随和错误版本。"""
        if not isinstance(key, tuple):
            raise TypeError("HypothesisKey 稳定键必须是整数 tuple")
        assert_int(*key, _where="HypothesisKey.stable_key")
        if any(type(value) is not int for value in key):
            raise ValueError("HypothesisKey 稳定键必须使用严格整数")
        if len(key) < 10 or key[0] != _HYPOTHESIS_KEY_VERSION:
            raise ValueError("HypothesisKey 稳定键版本或长度非法")

        cursor = 1

        def take_part(label: str) -> tuple[int, ...]:
            """按长度前缀读取一个非空键段，并推进外层游标。"""
            nonlocal cursor
            if cursor >= len(key):
                raise ValueError(f"HypothesisKey 缺少 {label} 长度")
            size = key[cursor]
            cursor += 1
            if size <= 0 or cursor + size > len(key):
                raise ValueError(f"HypothesisKey {label} 长度非法")
            part = key[cursor:cursor + size]
            cursor += size
            return part

        hypothesis_kind = take_part("hypothesis_kind")
        candidate_key = take_part("candidate_key")
        competition_key = take_part("competition_key")
        scope_key = take_part("scope")
        observation_key = take_part("observation")
        if cursor != len(key):
            raise ValueError("HypothesisKey 稳定键含尾随数据")
        return cls(
            hypothesis_kind,
            candidate_key,
            competition_key,
            ScopeIdentity.from_stable_key(scope_key),
            SourceRef.from_stable_key(observation_key),
        )

    def object_identity(self) -> ObjectIdentity:
        """把候选协议身份映射为一等 Hypothesis 对象身份。"""
        return ObjectIdentity(
            OBJECT_HYPOTHESIS,
            self.stable_key(),
            self.observation.owner,
            self.observation.versions,
        )


@dataclass(frozen=True, order=True)
class EvidenceRecord:
    """一条不可变证据事件；纠正通过 supersedes 链追加而非原地修改。"""

    evidence_id: int
    hypothesis: HypothesisKey
    stance: int
    reason_key: tuple[int, ...]
    source: SourceRef
    timestamp_seq: int
    payload: tuple[int, ...] = ()
    supersedes_evidence_id: int = 0

    def __post_init__(self) -> None:
        _positive(self.evidence_id, where="EvidenceRecord.evidence_id")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("EvidenceRecord.hypothesis 必须是 HypothesisKey")
        _positive(self.stance, where="EvidenceRecord.stance")
        if self.stance not in _EVIDENCE_STANCES:
            raise ValueError("EvidenceRecord.stance 未注册")
        _strict_tuple(self.reason_key, where="EvidenceRecord.reason_key")
        if not isinstance(self.source, SourceRef):
            raise TypeError("EvidenceRecord.source 必须是 SourceRef")
        _nonnegative(
            self.timestamp_seq, where="EvidenceRecord.timestamp_seq")
        if not isinstance(self.payload, tuple):
            raise TypeError("EvidenceRecord.payload 必须是整数 tuple")
        assert_int(*self.payload, _where="EvidenceRecord.payload")
        if any(type(value) is not int for value in self.payload):
            raise ValueError("EvidenceRecord.payload 必须使用严格整数")
        _nonnegative(
            self.supersedes_evidence_id,
            where="EvidenceRecord.supersedes_evidence_id",
        )
        if self.supersedes_evidence_id == self.evidence_id:
            raise ValueError("Evidence 不得 supersede 自身")

    def stable_key(self) -> tuple[int, ...]:
        """返回包含候选、来源、理由和逻辑序的完整整数事件键。"""
        hypothesis_key = self.hypothesis.stable_key()
        source_key = self.source.stable_key()
        return (
            self.evidence_id,
            len(hypothesis_key),
            *hypothesis_key,
            self.stance,
            len(self.reason_key),
            *self.reason_key,
            len(source_key),
            *source_key,
            self.timestamp_seq,
            self.supersedes_evidence_id,
            len(self.payload),
            *self.payload,
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "EvidenceRecord":
        """从完整事件键恢复 Evidence，拒绝截断、尾随和非法空理由。"""
        if not isinstance(key, tuple):
            raise TypeError("EvidenceRecord 稳定键必须是整数 tuple")
        assert_int(*key, _where="EvidenceRecord.stable_key")
        if any(type(value) is not int for value in key) or len(key) < 10:
            raise ValueError("EvidenceRecord 稳定键长度或类型非法")
        cursor = 1

        def take_part(label: str, *, allow_empty: bool = False) -> tuple[int, ...]:
            """读取一个长度前缀字段并推进 Evidence 外层游标。"""
            nonlocal cursor
            if cursor >= len(key):
                raise ValueError(f"EvidenceRecord 缺少 {label} 长度")
            size = key[cursor]
            cursor += 1
            if size < 0 or (size == 0 and not allow_empty):
                raise ValueError(f"EvidenceRecord {label} 长度非法")
            end = cursor + size
            if end > len(key):
                raise ValueError(f"EvidenceRecord {label} 被截断")
            result = key[cursor:end]
            cursor = end
            return result

        hypothesis_key = take_part("hypothesis")
        if cursor >= len(key):
            raise ValueError("EvidenceRecord 缺少 stance")
        stance = key[cursor]
        cursor += 1
        reason_key = take_part("reason_key")
        source_key = take_part("source")
        if cursor + 3 > len(key):
            raise ValueError("EvidenceRecord 时序或 supersede 字段被截断")
        timestamp_seq, supersedes_evidence_id = key[cursor:cursor + 2]
        cursor += 2
        payload = take_part("payload", allow_empty=True)
        if cursor != len(key):
            raise ValueError("EvidenceRecord 稳定键含尾随字段")
        return cls(
            key[0],
            HypothesisKey.from_stable_key(hypothesis_key),
            stance,
            reason_key,
            SourceRef.from_stable_key(source_key),
            timestamp_seq,
            payload,
            supersedes_evidence_id,
        )


@dataclass(frozen=True, order=True)
class HypothesisTransition:
    """候选生命周期的不可变转换事件，必须引用已有 Evidence 和理由。"""

    event_id: int
    hypothesis: HypothesisKey
    from_state: int
    to_state: int
    reason_evidence_id: int
    reason_key: tuple[int, ...]
    timestamp_seq: int
    replacement: HypothesisKey | None = None

    def __post_init__(self) -> None:
        _positive(self.event_id, where="HypothesisTransition.event_id")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("HypothesisTransition.hypothesis 必须是 HypothesisKey")
        for name, value in (
                ("from_state", self.from_state),
                ("to_state", self.to_state)):
            _positive(value, where=f"HypothesisTransition.{name}")
            if value not in _LIFECYCLE_STATES:
                raise ValueError(f"HypothesisTransition.{name} 未注册")
        if self.from_state == self.to_state:
            raise ValueError("生命周期转换前后状态不得相同")
        _positive(
            self.reason_evidence_id,
            where="HypothesisTransition.reason_evidence_id",
        )
        _strict_tuple(
            self.reason_key, where="HypothesisTransition.reason_key")
        _nonnegative(
            self.timestamp_seq, where="HypothesisTransition.timestamp_seq")
        if self.replacement is not None and not isinstance(
                self.replacement, HypothesisKey):
            raise TypeError("replacement 必须是 HypothesisKey 或 None")

    def stable_key(self) -> tuple[int, ...]:
        """返回生命周期转换、理由、时序和 replacement 的完整稳定键。"""
        hypothesis_key = self.hypothesis.stable_key()
        replacement_key = (
            () if self.replacement is None
            else self.replacement.stable_key()
        )
        return (
            _TRANSITION_KEY_VERSION,
            self.event_id,
            len(hypothesis_key),
            *hypothesis_key,
            self.from_state,
            self.to_state,
            self.reason_evidence_id,
            len(self.reason_key),
            *self.reason_key,
            self.timestamp_seq,
            len(replacement_key),
            *replacement_key,
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "HypothesisTransition":
        """从完整转换键恢复对象，拒绝错误版本、截断和尾随。"""
        if not isinstance(key, tuple):
            raise TypeError("HypothesisTransition 稳定键必须是整数 tuple")
        assert_int(*key, _where="HypothesisTransition.stable_key")
        if (any(type(value) is not int for value in key)
                or len(key) < 11
                or key[0] != _TRANSITION_KEY_VERSION):
            raise ValueError("HypothesisTransition 稳定键版本或长度非法")
        cursor = 2

        def take_part(label: str, *, allow_empty: bool = False) -> tuple[int, ...]:
            """读取一个长度前缀字段并推进转换外层游标。"""
            nonlocal cursor
            if cursor >= len(key):
                raise ValueError(f"HypothesisTransition 缺少 {label} 长度")
            size = key[cursor]
            cursor += 1
            if size < 0 or (size == 0 and not allow_empty):
                raise ValueError(f"HypothesisTransition {label} 长度非法")
            end = cursor + size
            if end > len(key):
                raise ValueError(f"HypothesisTransition {label} 被截断")
            result = key[cursor:end]
            cursor = end
            return result

        hypothesis_key = take_part("hypothesis")
        if cursor + 4 > len(key):
            raise ValueError("HypothesisTransition 状态或理由字段被截断")
        from_state, to_state, reason_evidence_id = key[cursor:cursor + 3]
        cursor += 3
        reason_key = take_part("reason_key")
        if cursor >= len(key):
            raise ValueError("HypothesisTransition 缺少 timestamp_seq")
        timestamp_seq = key[cursor]
        cursor += 1
        replacement_key = take_part("replacement", allow_empty=True)
        if cursor != len(key):
            raise ValueError("HypothesisTransition 稳定键含尾随字段")
        return cls(
            key[1],
            HypothesisKey.from_stable_key(hypothesis_key),
            from_state,
            to_state,
            reason_evidence_id,
            reason_key,
            timestamp_seq,
            None if not replacement_key else HypothesisKey.from_stable_key(
                replacement_key),
        )


@dataclass(frozen=True)
class HypothesisSnapshot:
    """由 append-only 事件派生的当前候选状态，不承担权威写入。"""

    hypothesis: HypothesisKey
    lifecycle: int
    epistemic_status: int
    support_evidence_ids: tuple[int, ...]
    refute_evidence_ids: tuple[int, ...]
    unknown_evidence_ids: tuple[int, ...]


class HypothesisEventSink(Protocol):
    """训练状态或未来 Memory adapter 可实现的 append-only 写边界。"""

    def append_hypothesis(self, hypothesis: HypothesisKey) -> None: ...
    def append_evidence(self, evidence: EvidenceRecord) -> None: ...
    def append_transition(self, transition: HypothesisTransition) -> None: ...


class HypothesisLedger:
    """在认知层聚合候选事件，并保持原始 Evidence 与转换历史不可变。"""

    def __init__(self, sink: HypothesisEventSink | None = None) -> None:
        self._sink = sink
        self._hypotheses: dict[HypothesisKey, int] = {}
        self._evidence: dict[int, EvidenceRecord] = {}
        self._superseded_evidence: dict[int, int] = {}
        self._transitions: dict[int, HypothesisTransition] = {}
        self._competition_members: dict[tuple, set[HypothesisKey]] = {}
        self._candidate_members: dict[tuple, set[HypothesisKey]] = {}
        self._evidence_by_hypothesis: dict[
            HypothesisKey, list[int]
        ] = {}
        self._transitions_by_hypothesis: dict[
            HypothesisKey, list[int]
        ] = {}

    def register(self, hypothesis: HypothesisKey) -> HypothesisKey:
        """幂等登记 active 候选；同竞争组可同时登记多个候选。"""
        if not isinstance(hypothesis, HypothesisKey):
            raise TypeError("register 需要 HypothesisKey")
        if hypothesis in self._hypotheses:
            return hypothesis
        if self._sink is not None:
            self._sink.append_hypothesis(hypothesis)
        self._hypotheses[hypothesis] = LIFECYCLE_ACTIVE
        competition = self._competition_index_key(hypothesis)
        self._competition_members.setdefault(
            competition, set()).add(hypothesis)
        candidate = self._candidate_index_key(hypothesis)
        self._candidate_members.setdefault(candidate, set()).add(hypothesis)
        self._evidence_by_hypothesis[hypothesis] = []
        self._transitions_by_hypothesis[hypothesis] = []
        return hypothesis

    def append_evidence(self, evidence: EvidenceRecord) -> EvidenceRecord:
        """追加证据并核验 supersede 目标属于同一候选且未被其他事件替代。"""
        if not isinstance(evidence, EvidenceRecord):
            raise TypeError("append_evidence 需要 EvidenceRecord")
        if evidence.hypothesis not in self._hypotheses:
            raise KeyError("Evidence 对应 Hypothesis 尚未登记")
        existing = self._evidence.get(evidence.evidence_id)
        if existing is not None:
            if existing != evidence:
                raise ValueError("同一 evidence_id 已绑定不同 Evidence")
            return existing
        old_id = evidence.supersedes_evidence_id
        if old_id:
            old = self._evidence.get(old_id)
            if old is None or old.hypothesis != evidence.hypothesis:
                raise ValueError("Evidence supersede 目标不存在或属于其他候选")
            if evidence.timestamp_seq < old.timestamp_seq:
                raise ValueError("Evidence supersede 的逻辑序不得早于旧事件")
            prior = self._superseded_evidence.get(old_id)
            if prior is not None and prior != evidence.evidence_id:
                raise ValueError("同一 Evidence 已被不同事件替代")
        if self._sink is not None:
            self._sink.append_evidence(evidence)
        self._evidence[evidence.evidence_id] = evidence
        self._evidence_by_hypothesis[
            evidence.hypothesis].append(evidence.evidence_id)
        if old_id:
            self._superseded_evidence[old_id] = evidence.evidence_id
        return evidence

    def append_transition(
            self, transition: HypothesisTransition) -> HypothesisTransition:
        """追加单向生命周期转换，不允许从 archived/superseded 回到 active。"""
        if not isinstance(transition, HypothesisTransition):
            raise TypeError("append_transition 需要 HypothesisTransition")
        existing = self._transitions.get(transition.event_id)
        if existing is not None:
            if existing != transition:
                raise ValueError("同一 event_id 已绑定不同转换")
            return existing
        current = self._hypotheses.get(transition.hypothesis)
        if current is None:
            raise KeyError("转换对应 Hypothesis 尚未登记")
        if current != transition.from_state:
            raise ValueError("转换 from_state 与当前生命周期不一致")
        evidence = self._evidence.get(transition.reason_evidence_id)
        if evidence is None or evidence.hypothesis != transition.hypothesis:
            raise ValueError("生命周期转换必须引用同一候选的已有 Evidence")
        if transition.timestamp_seq < evidence.timestamp_seq:
            raise ValueError("生命周期转换的逻辑序不得早于理由 Evidence")
        allowed = {
            (LIFECYCLE_ACTIVE, LIFECYCLE_SUPERSEDED),
            (LIFECYCLE_ACTIVE, LIFECYCLE_ARCHIVED),
            (LIFECYCLE_SUPERSEDED, LIFECYCLE_ARCHIVED),
        }
        if (transition.from_state, transition.to_state) not in allowed:
            raise ValueError("生命周期只允许 active→superseded/archived 或 superseded→archived")
        if transition.to_state == LIFECYCLE_SUPERSEDED:
            replacement = transition.replacement
            if replacement is None or replacement == transition.hypothesis:
                raise ValueError("superseded 转换必须指定不同 replacement")
            replacement_state = self._hypotheses.get(replacement)
            if replacement_state != LIFECYCLE_ACTIVE:
                raise ValueError("replacement 必须是已登记的 active 候选")
            if (
                    replacement.hypothesis_kind
                    != transition.hypothesis.hypothesis_kind
                    or replacement.competition_key
                    != transition.hypothesis.competition_key
                    or replacement.scope != transition.hypothesis.scope
                    or replacement.observation
                    != transition.hypothesis.observation):
                raise ValueError("replacement 必须属于同一竞争组")
        elif transition.replacement is not None:
            raise ValueError("archived 转换不得携带 replacement")
        if self._sink is not None:
            self._sink.append_transition(transition)
        self._transitions[transition.event_id] = transition
        self._transitions_by_hypothesis[
            transition.hypothesis].append(transition.event_id)
        self._hypotheses[transition.hypothesis] = transition.to_state
        return transition

    def snapshot(self, hypothesis: HypothesisKey) -> HypothesisSnapshot:
        """从未被替代的 Evidence 派生 epistemic 状态和独立生命周期。"""
        lifecycle = self._hypotheses.get(hypothesis)
        if lifecycle is None:
            raise KeyError("Hypothesis 尚未登记")
        active = [
            self._evidence[evidence_id]
            for evidence_id in self._evidence_by_hypothesis[hypothesis]
            if evidence_id not in self._superseded_evidence
        ]
        support = tuple(sorted(
            item.evidence_id for item in active
            if item.stance == EVIDENCE_SUPPORT))
        refute = tuple(sorted(
            item.evidence_id for item in active
            if item.stance == EVIDENCE_REFUTE))
        unknown = tuple(sorted(
            item.evidence_id for item in active
            if item.stance == EVIDENCE_UNKNOWN))
        if support and refute:
            epistemic = EPISTEMIC_CONFLICTED
        elif support:
            epistemic = EPISTEMIC_SUPPORTED
        elif refute:
            epistemic = EPISTEMIC_REFUTED
        else:
            epistemic = EPISTEMIC_UNKNOWN
        return HypothesisSnapshot(
            hypothesis,
            lifecycle,
            epistemic,
            support,
            refute,
            unknown,
        )

    def competition(
            self, hypothesis: HypothesisKey) -> tuple[HypothesisSnapshot, ...]:
        """返回同 scope、kind、observation 和竞争组中的全部互斥候选。"""
        if hypothesis not in self._hypotheses:
            raise KeyError("Hypothesis 尚未登记")
        matches = self._competition_members[
            self._competition_index_key(hypothesis)]
        return tuple(self.snapshot(key) for key in sorted(matches))

    def candidate_snapshots(
            self,
            candidate_key: tuple[int, ...],
            *,
            observation: SourceRef,
            scope: ScopeIdentity,
            hypothesis_kind: tuple[int, ...],
            ) -> tuple[HypothesisSnapshot, ...]:
        """按对象完整键、来源和 scope 返回全部匹配候选，不替调用方选择竞争组。"""
        _strict_tuple(candidate_key, where="candidate_snapshots.candidate_key")
        if not isinstance(observation, SourceRef):
            raise TypeError("candidate_snapshots observation 必须是 SourceRef")
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("candidate_snapshots scope 必须是 ScopeIdentity")
        _strict_tuple(
            hypothesis_kind,
            where="candidate_snapshots.hypothesis_kind",
        )
        index_key = (
            hypothesis_kind,
            candidate_key,
            scope,
            observation,
        )
        matches = self._candidate_members.get(index_key, ())
        return tuple(self.snapshot(item) for item in sorted(matches))

    def evidence_history(
            self, hypothesis: HypothesisKey) -> tuple[EvidenceRecord, ...]:
        """返回候选的完整 Evidence 历史，包括已被替代的旧事件。"""
        if hypothesis not in self._hypotheses:
            raise KeyError("Hypothesis 尚未登记")
        return tuple(sorted(
            (self._evidence[evidence_id]
             for evidence_id in self._evidence_by_hypothesis[hypothesis]),
            key=lambda item: (item.timestamp_seq, item.evidence_id),
        ))

    def transition_history(
            self, hypothesis: HypothesisKey
            ) -> tuple[HypothesisTransition, ...]:
        """返回候选的完整生命周期事件历史。"""
        if hypothesis not in self._hypotheses:
            raise KeyError("Hypothesis 尚未登记")
        return tuple(sorted(
            (self._transitions[event_id]
             for event_id in self._transitions_by_hypothesis[hypothesis]),
            key=lambda item: (item.timestamp_seq, item.event_id),
        ))

    def hypotheses(self) -> tuple[HypothesisKey, ...]:
        """返回当前 ledger 已登记的全部完整候选身份。"""
        return tuple(sorted(
            self._hypotheses,
            key=lambda item: item.stable_key(),
        ))

    @property
    def event_sink(self) -> HypothesisEventSink | None:
        """返回当前追加边界；无 sink 表示状态仅存在于本进程。"""
        return self._sink

    def with_sink(self, sink: HypothesisEventSink) -> "HypothesisLedger":
        """复制已恢复状态并绑定后续追加 sink，不重放任何历史事件。"""
        if any(not callable(getattr(sink, name, None)) for name in (
                "append_hypothesis", "append_evidence", "append_transition")):
            raise TypeError("sink 必须实现完整 HypothesisEventSink 协议")
        cloned = self.clone()
        cloned._sink = sink
        return cloned

    def clone(self) -> "HypothesisLedger":
        """复制纯领域状态，供评测或候选试算隔离后追加事件。"""
        cloned = HypothesisLedger()
        for hypothesis in sorted(self._hypotheses):
            cloned._hypotheses[hypothesis] = self._hypotheses[hypothesis]
        cloned._evidence = dict(self._evidence)
        cloned._superseded_evidence = dict(self._superseded_evidence)
        cloned._transitions = dict(self._transitions)
        cloned._competition_members = {
            key: set(value)
            for key, value in self._competition_members.items()
        }
        cloned._candidate_members = {
            key: set(value)
            for key, value in self._candidate_members.items()
        }
        cloned._evidence_by_hypothesis = {
            key: list(value)
            for key, value in self._evidence_by_hypothesis.items()
        }
        cloned._transitions_by_hypothesis = {
            key: list(value)
            for key, value in self._transitions_by_hypothesis.items()
        }
        return cloned

    def state_key(self) -> tuple:
        """返回完整不可变状态，用于评测隔离和确定性回归核验。"""
        return (
            tuple(sorted(self._hypotheses.items())),
            tuple(sorted(self._evidence.items())),
            tuple(sorted(self._superseded_evidence.items())),
            tuple(sorted(self._transitions.items())),
        )

    @staticmethod
    def _competition_index_key(hypothesis: HypothesisKey) -> tuple:
        """构造内存路由键；权威身份仍由 Hypothesis 完整字段保存。"""
        return (
            hypothesis.hypothesis_kind,
            hypothesis.competition_key,
            hypothesis.scope,
            hypothesis.observation,
        )

    @staticmethod
    def _candidate_index_key(hypothesis: HypothesisKey) -> tuple:
        """构造对象恢复索引键，避免只读 query 线性扫描全部候选。"""
        return (
            hypothesis.hypothesis_kind,
            hypothesis.candidate_key,
            hypothesis.scope,
            hypothesis.observation,
        )


__all__ = [
    "EPISTEMIC_CONFLICTED",
    "EPISTEMIC_REFUTED",
    "EPISTEMIC_SUPPORTED",
    "EPISTEMIC_UNKNOWN",
    "EVIDENCE_REFUTE",
    "EVIDENCE_SUPPORT",
    "EVIDENCE_UNKNOWN",
    "EvidenceRecord",
    "HypothesisEventSink",
    "HypothesisKey",
    "HypothesisLedger",
    "HypothesisSnapshot",
    "HypothesisTransition",
    "LIFECYCLE_ACTIVE",
    "LIFECYCLE_ARCHIVED",
    "LIFECYCLE_SUPERSEDED",
]
