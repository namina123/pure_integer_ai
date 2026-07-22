"""H-04 通用候选解析、来源分账和 append-only 决策历史。

本模块只理解 H-00 的生命周期与证据状态，不解释语言、结构或成本语义。领域 scorer
以成对关系提交比较结果；通用层只做保守合并，遇到反向、不可比或等价时保留多解。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_SUPPORTED,
    EPISTEMIC_UNKNOWN,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    HypothesisSnapshot,
    HypothesisTransition,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int

PREFERENCE_LEFT_BETTER = 1
PREFERENCE_RIGHT_BETTER = 2
PREFERENCE_EQUIVALENT = 3
PREFERENCE_INCOMPARABLE = 4
_PREFERENCES = frozenset({
    PREFERENCE_LEFT_BETTER,
    PREFERENCE_RIGHT_BETTER,
    PREFERENCE_EQUIVALENT,
    PREFERENCE_INCOMPARABLE,
})

RESOLUTION_ADOPTED = 1
RESOLUTION_RETAINED = 2
RESOLUTION_EXITED = 3
_RESOLUTION_ROLES = frozenset({
    RESOLUTION_ADOPTED,
    RESOLUTION_RETAINED,
    RESOLUTION_EXITED,
})

_DECISION_HASHER = Hasher("hypothesis_resolution.decision.v1")
_TRANSITION_HASHER = Hasher("hypothesis_resolution.transition.v1")
_RESOLUTION_VERSION = 1


def _integer_key(value, *, where: str) -> tuple[int, ...]:
    """校验调用方注入的开放整数协议键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _strict_int_tuple(value, *, where: str) -> tuple[int, ...]:
    """校验允许为空和负数的审计 payload。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{where} 必须是整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _stable_tuple(value, *, where: str) -> tuple[int, ...]:
    """校验反序列化入口收到严格整数 tuple。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{where} 必须是整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _take_packed(
        values: tuple[int, ...], cursor: int, *, where: str,
        ) -> tuple[tuple[int, ...], int]:
    """从长度前缀稳定键读取一个子键并返回新游标。"""
    if cursor >= len(values):
        raise ValueError(f"{where} 缺少长度")
    size = values[cursor]
    if size < 0 or cursor + 1 + size > len(values):
        raise ValueError(f"{where} 长度非法或内容被截断")
    start = cursor + 1
    return values[start:start + size], start + size


def _take_ids(
        values: tuple[int, ...], cursor: int, *, where: str,
        ) -> tuple[tuple[int, ...], int]:
    """读取一个数量前缀的非负整数 id 序列。"""
    items, cursor = _take_packed(values, cursor, where=where)
    if any(item <= 0 for item in items):
        raise ValueError(f"{where} 只能包含严格正整数 id")
    return items, cursor


def _snapshot_key(snapshot: HypothesisSnapshot) -> tuple[int, ...]:
    """把派生快照展开为不依赖对象 hash 的完整整数键。"""
    hypothesis = snapshot.hypothesis.stable_key()
    return (
        len(hypothesis),
        *hypothesis,
        snapshot.lifecycle,
        snapshot.epistemic_status,
        len(snapshot.support_evidence_ids),
        *snapshot.support_evidence_ids,
        len(snapshot.refute_evidence_ids),
        *snapshot.refute_evidence_ids,
        len(snapshot.unknown_evidence_ids),
        *snapshot.unknown_evidence_ids,
    )


def _snapshot_from_key(key: tuple[int, ...]) -> HypothesisSnapshot:
    """从完整整数键恢复一个 H-00 派生快照。"""
    key = _stable_tuple(key, where="HypothesisSnapshot.stable_key")
    hypothesis_key, cursor = _take_packed(
        key, 0, where="HypothesisSnapshot.hypothesis")
    if cursor + 2 > len(key):
        raise ValueError("HypothesisSnapshot 缺 lifecycle/epistemic")
    lifecycle, epistemic = key[cursor:cursor + 2]
    support, cursor = _take_ids(
        key, cursor + 2, where="HypothesisSnapshot.support")
    refute, cursor = _take_ids(
        key, cursor, where="HypothesisSnapshot.refute")
    unknown, cursor = _take_ids(
        key, cursor, where="HypothesisSnapshot.unknown")
    if cursor != len(key):
        raise ValueError("HypothesisSnapshot 稳定键含尾随数据")
    if lifecycle not in {
            LIFECYCLE_ACTIVE, LIFECYCLE_ARCHIVED, LIFECYCLE_SUPERSEDED}:
        raise ValueError("HypothesisSnapshot lifecycle 未注册")
    if epistemic not in {
            EPISTEMIC_UNKNOWN, EPISTEMIC_SUPPORTED,
            EPISTEMIC_REFUTED, EPISTEMIC_CONFLICTED}:
        raise ValueError("HypothesisSnapshot epistemic 未注册")
    active_ids = (*support, *refute, *unknown)
    if len(set(active_ids)) != len(active_ids):
        raise ValueError("HypothesisSnapshot active Evidence id 重复")
    return HypothesisSnapshot(
        HypothesisKey.from_stable_key(hypothesis_key),
        lifecycle,
        epistemic,
        support,
        refute,
        unknown,
    )


def _competition_key(hypothesis: HypothesisKey) -> tuple[int, ...]:
    """返回不含 candidate_key 的完整竞争边界身份。"""
    kind = hypothesis.hypothesis_kind
    competition = hypothesis.competition_key
    scope = hypothesis.scope.stable_key()
    observation = hypothesis.observation.stable_key()
    return (
        _RESOLUTION_VERSION,
        len(kind),
        *kind,
        len(competition),
        *competition,
        len(scope),
        *scope,
        len(observation),
        *observation,
    )


def _same_competition(left: HypothesisKey, right: HypothesisKey) -> bool:
    """按 H-00 完整边界判断两个候选能否直接竞争。"""
    return _competition_key(left) == _competition_key(right)


def _epistemic_status(
        support_ids: tuple[int, ...], refute_ids: tuple[int, ...],
        ) -> int:
    """从一个来源账户的 active 立场派生四态，不按频次投票。"""
    if support_ids and refute_ids:
        return EPISTEMIC_CONFLICTED
    if support_ids:
        return EPISTEMIC_SUPPORTED
    if refute_ids:
        return EPISTEMIC_REFUTED
    return EPISTEMIC_UNKNOWN


@dataclass(frozen=True)
class ResolverPreference:
    """一个 typed scorer 对规范化候选对给出的可审计关系。"""

    scorer_key: tuple[int, ...]
    left: HypothesisKey
    right: HypothesisKey
    preference: int
    payload: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _integer_key(self.scorer_key, where="ResolverPreference.scorer_key")
        if not isinstance(self.left, HypothesisKey):
            raise TypeError("ResolverPreference.left 必须是 HypothesisKey")
        if not isinstance(self.right, HypothesisKey):
            raise TypeError("ResolverPreference.right 必须是 HypothesisKey")
        if self.left == self.right or not _same_competition(
                self.left, self.right):
            raise ValueError("scorer 只能比较同一竞争组中的两个不同候选")
        if self.left.stable_key() > self.right.stable_key():
            raise ValueError("候选对必须按完整 Hypothesis 稳定键规范化")
        assert_int(self.preference, _where="ResolverPreference.preference")
        if self.preference not in _PREFERENCES:
            raise ValueError("ResolverPreference.preference 未注册")
        _strict_int_tuple(self.payload, where="ResolverPreference.payload")

    def stable_key(self) -> tuple[int, ...]:
        """展开 scorer、候选对、关系和完整 payload。"""
        left = self.left.stable_key()
        right = self.right.stable_key()
        return (
            len(self.scorer_key),
            *self.scorer_key,
            len(left),
            *left,
            len(right),
            *right,
            self.preference,
            len(self.payload),
            *self.payload,
        )

    @classmethod
    def from_stable_key(
            cls, key: tuple[int, ...],
            ) -> "ResolverPreference":
        """从完整稳定键恢复一个领域 scorer 偏好。"""
        key = _stable_tuple(key, where="ResolverPreference.stable_key")
        scorer, cursor = _take_packed(
            key, 0, where="ResolverPreference.scorer")
        left, cursor = _take_packed(
            key, cursor, where="ResolverPreference.left")
        right, cursor = _take_packed(
            key, cursor, where="ResolverPreference.right")
        if cursor >= len(key):
            raise ValueError("ResolverPreference 缺 preference")
        preference = key[cursor]
        payload, cursor = _take_packed(
            key, cursor + 1, where="ResolverPreference.payload")
        if cursor != len(key):
            raise ValueError("ResolverPreference 稳定键含尾随数据")
        return cls(
            scorer,
            HypothesisKey.from_stable_key(left),
            HypothesisKey.from_stable_key(right),
            preference,
            payload,
        )


class TypedResolverScorer(Protocol):
    """领域 scorer 的最小只读边界，通用层不解释比较 payload。"""

    scorer_key: tuple[int, ...]

    def preferences(
            self, hypotheses: tuple[HypothesisKey, ...],
            ) -> tuple[ResolverPreference, ...]: ...


@dataclass(frozen=True)
class ReplacementDirective:
    """调用方用已有定向反驳显式声明一个候选替代关系。"""

    rejected: HypothesisKey
    replacement: HypothesisKey
    reason_evidence_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.rejected, HypothesisKey):
            raise TypeError("ReplacementDirective.rejected 类型错误")
        if not isinstance(self.replacement, HypothesisKey):
            raise TypeError("ReplacementDirective.replacement 类型错误")
        if self.rejected == self.replacement or not _same_competition(
                self.rejected, self.replacement):
            raise ValueError("replacement 必须是同一竞争组中的不同候选")
        assert_int(
            self.reason_evidence_id,
            _where="ReplacementDirective.reason_evidence_id",
        )
        if type(self.reason_evidence_id) is not int or self.reason_evidence_id <= 0:
            raise ValueError("replacement 必须引用严格正整数 Evidence id")


@dataclass(frozen=True)
class ArchiveDirective:
    """调用方用已有定向反驳显式归档候选，但不声明替代者。"""

    rejected: HypothesisKey
    reason_evidence_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.rejected, HypothesisKey):
            raise TypeError("ArchiveDirective.rejected 类型错误")
        assert_int(
            self.reason_evidence_id,
            _where="ArchiveDirective.reason_evidence_id",
        )
        if type(self.reason_evidence_id) is not int or self.reason_evidence_id <= 0:
            raise ValueError("archive 必须引用严格正整数 Evidence id")


@dataclass(frozen=True)
class ResolverSourceAccount:
    """一个候选按完整 SourceRef 划分的当前 active Evidence 账户。"""

    source: SourceRef
    epistemic_status: int
    support_evidence_ids: tuple[int, ...]
    refute_evidence_ids: tuple[int, ...]
    unknown_evidence_ids: tuple[int, ...]

    def stable_key(self) -> tuple[int, ...]:
        """保存来源完整身份和三类 Evidence 引用，不压成计数。"""
        source = self.source.stable_key()
        return (
            len(source),
            *source,
            self.epistemic_status,
            len(self.support_evidence_ids),
            *self.support_evidence_ids,
            len(self.refute_evidence_ids),
            *self.refute_evidence_ids,
            len(self.unknown_evidence_ids),
            *self.unknown_evidence_ids,
        )

    @classmethod
    def from_stable_key(
            cls, key: tuple[int, ...],
            ) -> "ResolverSourceAccount":
        """从完整来源和 Evidence 分账键恢复一个账户。"""
        key = _stable_tuple(key, where="ResolverSourceAccount.stable_key")
        source, cursor = _take_packed(
            key, 0, where="ResolverSourceAccount.source")
        if cursor >= len(key):
            raise ValueError("ResolverSourceAccount 缺 epistemic_status")
        epistemic = key[cursor]
        support, cursor = _take_ids(
            key, cursor + 1, where="ResolverSourceAccount.support")
        refute, cursor = _take_ids(
            key, cursor, where="ResolverSourceAccount.refute")
        unknown, cursor = _take_ids(
            key, cursor, where="ResolverSourceAccount.unknown")
        if cursor != len(key):
            raise ValueError("ResolverSourceAccount 稳定键含尾随数据")
        if epistemic not in {
                EPISTEMIC_UNKNOWN, EPISTEMIC_SUPPORTED,
                EPISTEMIC_REFUTED, EPISTEMIC_CONFLICTED}:
            raise ValueError("ResolverSourceAccount epistemic 未注册")
        if len(set((*support, *refute, *unknown))) != (
                len(support) + len(refute) + len(unknown)):
            raise ValueError("ResolverSourceAccount Evidence id 重复")
        return cls(
            SourceRef.from_stable_key(source),
            epistemic,
            support,
            refute,
            unknown,
        )


@dataclass(frozen=True)
class ResolverCandidateTrace:
    """一个候选在本次解析前后状态、来源账和消费者投影。"""

    before: HypothesisSnapshot
    after: HypothesisSnapshot
    source_accounts: tuple[ResolverSourceAccount, ...]
    dominated_by: tuple[HypothesisKey, ...]
    role: int
    prior_role: int
    transition_event_id: int = 0

    def __post_init__(self) -> None:
        if self.before.hypothesis != self.after.hypothesis:
            raise ValueError("候选 trace 前后必须引用同一 Hypothesis")
        assert_int(
            self.role,
            self.prior_role,
            self.transition_event_id,
            _where="ResolverCandidateTrace",
        )
        if self.role not in _RESOLUTION_ROLES:
            raise ValueError("ResolverCandidateTrace.role 未注册")
        if self.prior_role not in {0, *_RESOLUTION_ROLES}:
            raise ValueError("ResolverCandidateTrace.prior_role 未注册")
        if self.transition_event_id < 0:
            raise ValueError("transition_event_id 不得为负")

    @property
    def hypothesis(self) -> HypothesisKey:
        """返回本条 trace 对应的完整候选身份。"""
        return self.after.hypothesis

    def stable_key(self) -> tuple[int, ...]:
        """展开前后快照、来源账、支配者和生命周期事件引用。"""
        before = _snapshot_key(self.before)
        after = _snapshot_key(self.after)
        accounts = tuple(
            item.stable_key() for item in self.source_accounts)
        dominated = tuple(
            item.stable_key() for item in self.dominated_by)
        return (
            len(before),
            *before,
            len(after),
            *after,
            len(accounts),
            *(value for account in accounts for value in (len(account), *account)),
            len(dominated),
            *(value for item in dominated for value in (len(item), *item)),
            self.role,
            self.prior_role,
            self.transition_event_id,
        )

    @classmethod
    def from_stable_key(
            cls, key: tuple[int, ...],
            ) -> "ResolverCandidateTrace":
        """从完整快照、来源账和角色键恢复候选决策轨迹。"""
        key = _stable_tuple(key, where="ResolverCandidateTrace.stable_key")
        before, cursor = _take_packed(
            key, 0, where="ResolverCandidateTrace.before")
        after, cursor = _take_packed(
            key, cursor, where="ResolverCandidateTrace.after")
        if cursor >= len(key):
            raise ValueError("ResolverCandidateTrace 缺 source account 数量")
        account_count = key[cursor]
        if account_count < 0:
            raise ValueError("ResolverCandidateTrace source account 数量非法")
        cursor += 1
        accounts: list[ResolverSourceAccount] = []
        for index in range(account_count):
            account, cursor = _take_packed(
                key, cursor,
                where=f"ResolverCandidateTrace.account[{index}]")
            accounts.append(ResolverSourceAccount.from_stable_key(account))
        if cursor >= len(key):
            raise ValueError("ResolverCandidateTrace 缺 dominated 数量")
        dominated_count = key[cursor]
        if dominated_count < 0:
            raise ValueError("ResolverCandidateTrace dominated 数量非法")
        cursor += 1
        dominated: list[HypothesisKey] = []
        for index in range(dominated_count):
            hypothesis, cursor = _take_packed(
                key, cursor,
                where=f"ResolverCandidateTrace.dominated[{index}]")
            dominated.append(HypothesisKey.from_stable_key(hypothesis))
        if cursor + 3 != len(key):
            raise ValueError("ResolverCandidateTrace 角色字段被截断或含尾随数据")
        return cls(
            _snapshot_from_key(before),
            _snapshot_from_key(after),
            tuple(accounts),
            tuple(dominated),
            key[cursor],
            key[cursor + 1],
            key[cursor + 2],
        )


@dataclass(frozen=True)
class ResolverDecision:
    """一次不可变解析事件；完整 trace 可独立核验 hash 碰撞。"""

    decision_id: int
    competition_key: tuple[int, ...]
    timestamp_seq: int
    previous_decision_id: int
    candidates: tuple[ResolverCandidateTrace, ...]
    preferences: tuple[ResolverPreference, ...]
    adopted_hypotheses: tuple[HypothesisKey, ...]

    def __post_init__(self) -> None:
        assert_int(
            self.decision_id,
            self.timestamp_seq,
            self.previous_decision_id,
            _where="ResolverDecision",
        )
        if self.decision_id <= 0 or self.timestamp_seq < 0:
            raise ValueError("decision id 必须为正且逻辑序不得为负")
        if self.previous_decision_id < 0:
            raise ValueError("previous_decision_id 不得为负")
        _integer_key(
            self.competition_key,
            where="ResolverDecision.competition_key",
        )

    def candidate(self, hypothesis: HypothesisKey) -> ResolverCandidateTrace:
        """按完整 Hypothesis 身份读取本次候选 trace。"""
        for candidate in self.candidates:
            if candidate.hypothesis == hypothesis:
                return candidate
        raise KeyError("Hypothesis 不属于本次 resolver 决策")

    def stable_key(self) -> tuple[int, ...]:
        """返回包含 decision id 的完整 append-only 事件键。"""
        identity = self.identity_key()
        return (self.decision_id, len(identity), *identity)

    def identity_key(self) -> tuple[int, ...]:
        """返回用于生成 decision id 的无摘要完整事件内容。"""
        candidates = tuple(item.stable_key() for item in self.candidates)
        preferences = tuple(item.stable_key() for item in self.preferences)
        adopted = tuple(item.stable_key() for item in self.adopted_hypotheses)
        return (
            _RESOLUTION_VERSION,
            len(self.competition_key),
            *self.competition_key,
            self.timestamp_seq,
            self.previous_decision_id,
            len(candidates),
            *(value for item in candidates for value in (len(item), *item)),
            len(preferences),
            *(value for item in preferences for value in (len(item), *item)),
            len(adopted),
            *(value for item in adopted for value in (len(item), *item)),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "ResolverDecision":
        """从完整 append-only 稳定键恢复 H-04 决策并重算 id。"""
        key = _stable_tuple(key, where="ResolverDecision.stable_key")
        if len(key) < 3 or key[1] != len(key) - 2:
            raise ValueError("ResolverDecision 外层 identity 长度非法")
        decision_id = key[0]
        identity = key[2:]
        if not identity or identity[0] != _RESOLUTION_VERSION:
            raise ValueError("ResolverDecision 版本未注册")
        competition, cursor = _take_packed(
            identity, 1, where="ResolverDecision.competition")
        if cursor + 2 > len(identity):
            raise ValueError("ResolverDecision 缺 timestamp/previous")
        timestamp_seq, previous_id = identity[cursor:cursor + 2]
        cursor += 2
        if cursor >= len(identity):
            raise ValueError("ResolverDecision 缺 candidate 数量")
        candidate_count = identity[cursor]
        if candidate_count <= 0:
            raise ValueError("ResolverDecision candidate 数量非法")
        cursor += 1
        candidates: list[ResolverCandidateTrace] = []
        for index in range(candidate_count):
            candidate, cursor = _take_packed(
                identity, cursor,
                where=f"ResolverDecision.candidate[{index}]")
            candidates.append(ResolverCandidateTrace.from_stable_key(candidate))
        if cursor >= len(identity):
            raise ValueError("ResolverDecision 缺 preference 数量")
        preference_count = identity[cursor]
        if preference_count < 0:
            raise ValueError("ResolverDecision preference 数量非法")
        cursor += 1
        preferences: list[ResolverPreference] = []
        for index in range(preference_count):
            preference, cursor = _take_packed(
                identity, cursor,
                where=f"ResolverDecision.preference[{index}]")
            preferences.append(ResolverPreference.from_stable_key(preference))
        if cursor >= len(identity):
            raise ValueError("ResolverDecision 缺 adopted 数量")
        adopted_count = identity[cursor]
        if adopted_count < 0:
            raise ValueError("ResolverDecision adopted 数量非法")
        cursor += 1
        adopted: list[HypothesisKey] = []
        for index in range(adopted_count):
            hypothesis, cursor = _take_packed(
                identity, cursor,
                where=f"ResolverDecision.adopted[{index}]")
            adopted.append(HypothesisKey.from_stable_key(hypothesis))
        if cursor != len(identity):
            raise ValueError("ResolverDecision identity 含尾随数据")
        decision = cls(
            decision_id,
            competition,
            timestamp_seq,
            previous_id,
            tuple(candidates),
            tuple(preferences),
            tuple(adopted),
        )
        if (_DECISION_HASHER.h63(decision.identity_key()) or 1) != decision_id:
            raise ValueError("ResolverDecision id 与完整内容不一致")
        return decision


class ResolverDecisionSink(Protocol):
    """H-04 决策的 append-only 持久化边界。"""

    def append_decision(self, decision: ResolverDecision) -> None:
        """幂等追加一个完整 ResolverDecision。"""
        ...


class HypothesisResolver:
    """在真正 H-00 ledger 上解析一个竞争组并保存可审计决策链。"""

    def __init__(
            self, ledger: HypothesisLedger,
            sink: ResolverDecisionSink | None = None,
            ) -> None:
        """绑定 ledger；resolver 只追加合法转换，不复制 Evidence。"""
        if not isinstance(ledger, HypothesisLedger):
            raise TypeError("ledger 必须是 HypothesisLedger")
        if sink is not None and not callable(
                getattr(sink, "append_decision", None)):
            raise TypeError("resolver sink 必须实现 append_decision")
        self.ledger = ledger
        self._sink = sink
        self._decisions: dict[int, ResolverDecision] = {}
        self._latest_by_competition: dict[tuple[int, ...], int] = {}

    def resolve(
            self, anchor: HypothesisKey, *, timestamp_seq: int,
            scorers: tuple[TypedResolverScorer, ...] = (),
            replacements: tuple[ReplacementDirective, ...] = (),
            archives: tuple[ArchiveDirective, ...] = (),
            commit: bool = True,
            ) -> ResolverDecision:
        """解析完整竞争组，并按显式指令或纯反驳状态执行原子退出。"""
        if not isinstance(anchor, HypothesisKey):
            raise TypeError("anchor 必须是 HypothesisKey")
        assert_int(timestamp_seq, _where="HypothesisResolver.timestamp_seq")
        if type(timestamp_seq) is not int or timestamp_seq < 0:
            raise ValueError("resolver 逻辑序必须为非负严格整数")
        if not isinstance(scorers, tuple):
            raise TypeError("scorers 必须是 typed scorer tuple")
        if not isinstance(replacements, tuple) or any(
                not isinstance(item, ReplacementDirective)
                for item in replacements):
            raise TypeError("replacements 只能包含 ReplacementDirective")
        if not isinstance(archives, tuple) or any(
                not isinstance(item, ArchiveDirective)
                for item in archives):
            raise TypeError("archives 只能包含 ArchiveDirective")
        replacement_targets = {
            item.rejected for item in replacements
        }
        archive_targets = {item.rejected for item in archives}
        if replacement_targets & archive_targets:
            raise ValueError("同一 rejected 不能同时 archive 和 replacement")
        if type(commit) is not bool:
            raise TypeError("commit 必须是 bool")
        if not commit:
            cloned_ledger = self.ledger.clone()
            cloned = self.clone(ledger=cloned_ledger)
            return cloned.resolve(
                anchor,
                timestamp_seq=timestamp_seq,
                scorers=scorers,
                replacements=replacements,
                archives=archives,
                commit=True,
            )

        before = self.ledger.competition(anchor)
        if not before:
            raise ValueError("resolver 竞争组不得为空")
        competition = _competition_key(anchor)
        directives, archive_directives = self._validated_directives(
            before, replacements, archives)
        projected = self._projected_snapshots(
            before, directives, archive_directives)
        eligible = self._eligible_hypotheses(projected)
        preferences = self._collect_preferences(eligible, scorers)
        dominated_by = self._dominance(eligible, preferences, len(scorers))
        adopted = tuple(
            hypothesis for hypothesis in eligible
            if not dominated_by[hypothesis]
        )
        transition_ids = self._exit_refuted(
            before,
            directives,
            archive_directives=archive_directives,
            timestamp_seq=timestamp_seq,
        )
        after = tuple(
            self.ledger.snapshot(item.hypothesis) for item in before)

        previous_id = self._latest_by_competition.get(competition, 0)
        previous = self._decisions.get(previous_id)
        if (previous is not None
                and previous.timestamp_seq == timestamp_seq
                and previous.preferences == preferences
                and previous.adopted_hypotheses == adopted
                and tuple(_snapshot_key(item) for item in after) == tuple(
                    _snapshot_key(item.after)
                    for item in previous.candidates)):
            return previous
        previous_roles = self._previous_roles(previous_id)
        candidate_traces = tuple(
            self._candidate_trace(
                old,
                current,
                adopted,
                dominated_by,
                previous_roles,
                transition_ids,
            )
            for old, current in zip(before, after, strict=True)
        )
        decision_without_id = ResolverDecision(
            1,
            competition,
            timestamp_seq,
            previous_id,
            candidate_traces,
            preferences,
            adopted,
        )
        decision_id = _DECISION_HASHER.h63(
            decision_without_id.identity_key()) or 1
        decision = ResolverDecision(
            decision_id,
            competition,
            timestamp_seq,
            previous_id,
            candidate_traces,
            preferences,
            adopted,
        )
        existing = self._decisions.get(decision_id)
        if existing is not None and existing != decision:
            raise ValueError("resolver decision hash 碰撞绑定了不同完整事件")
        if self._sink is not None:
            self._sink.append_decision(decision)
        self._decisions[decision_id] = decision
        self._latest_by_competition[competition] = decision_id
        return decision

    def decision_history(
            self, anchor: HypothesisKey) -> tuple[ResolverDecision, ...]:
        """返回一个完整竞争组的 append-only 决策历史。"""
        competition = _competition_key(anchor)
        return tuple(sorted(
            (item for item in self._decisions.values()
             if item.competition_key == competition),
            key=lambda item: (item.timestamp_seq, item.decision_id),
        ))

    @property
    def event_sink(self) -> ResolverDecisionSink | None:
        """返回当前决策追加边界；None 表示只保存在本进程。"""
        return self._sink

    @classmethod
    def from_history(
            cls, ledger: HypothesisLedger,
            decisions: tuple[ResolverDecision, ...], *,
            sink: ResolverDecisionSink | None = None,
            ) -> "HypothesisResolver":
        """从完整线性决策链恢复 resolver，并与 ledger 当前态双向核验。"""
        if not isinstance(ledger, HypothesisLedger):
            raise TypeError("from_history.ledger 类型错误")
        if (not isinstance(decisions, tuple)
                or any(not isinstance(item, ResolverDecision)
                       for item in decisions)):
            raise TypeError("decisions 必须是 ResolverDecision tuple")
        resolver = cls(ledger, sink=sink)
        if not decisions:
            return resolver
        by_id = {item.decision_id: item for item in decisions}
        if len(by_id) != len(decisions):
            raise ValueError("H-04 恢复历史包含重复 decision id")
        ledger_hypotheses = frozenset(ledger.hypotheses())
        grouped: dict[tuple[int, ...], list[ResolverDecision]] = {}
        for decision in decisions:
            if ResolverDecision.from_stable_key(
                    decision.stable_key()) != decision:
                raise ValueError("H-04 decision 稳定键无法无损往返")
            traces = tuple(item.hypothesis for item in decision.candidates)
            trace_set = frozenset(traces)
            if (not traces or len(trace_set) != len(traces)
                    or not trace_set <= ledger_hypotheses):
                raise ValueError("H-04 decision 候选未在恢复 ledger 唯一登记")
            if any(_competition_key(item) != decision.competition_key
                   for item in traces):
                raise ValueError("H-04 decision 候选越过竞争边界")
            if not set(decision.adopted_hypotheses) <= trace_set:
                raise ValueError("H-04 adopted 候选不属于 decision trace")
            if any({item.left, item.right} - trace_set
                   for item in decision.preferences):
                raise ValueError("H-04 preference 引用了 decision 外候选")
            grouped.setdefault(decision.competition_key, []).append(decision)

        for competition, history in grouped.items():
            group_ids = {item.decision_id for item in history}
            roots = [item for item in history
                     if item.previous_decision_id == 0]
            if len(roots) != 1:
                raise ValueError("H-04 每个竞争组必须有唯一决策链根")
            children: dict[int, ResolverDecision] = {}
            for decision in history:
                previous = decision.previous_decision_id
                if previous == 0:
                    continue
                if previous not in group_ids or previous in children:
                    raise ValueError("H-04 决策链缺前驱或出现分叉")
                children[previous] = decision
            ordered: list[ResolverDecision] = []
            current = roots[0]
            while True:
                ordered.append(current)
                next_value = children.get(current.decision_id)
                if next_value is None:
                    break
                current = next_value
            if len(ordered) != len(history):
                raise ValueError("H-04 决策链存在环或不可达事件")
            for prior, later in zip(ordered, ordered[1:]):
                if later.timestamp_seq < prior.timestamp_seq:
                    raise ValueError("H-04 决策逻辑序倒退")
            latest = ordered[-1]
            for trace in latest.candidates:
                if ledger.snapshot(trace.hypothesis) != trace.after:
                    raise ValueError("H-04 最新 decision 与 H-00 当前快照不一致")
            resolver._latest_by_competition[competition] = latest.decision_id
            for decision in ordered:
                resolver._decisions[decision.decision_id] = decision
        return resolver

    def clone(
            self, *, ledger: HypothesisLedger | None = None,
            ) -> "HypothesisResolver":
        """复制决策历史，并显式绑定调用方提供的克隆 ledger。"""
        target_ledger = self.ledger.clone() if ledger is None else ledger
        if not isinstance(target_ledger, HypothesisLedger):
            raise TypeError("clone.ledger 必须是 HypothesisLedger")
        cloned = HypothesisResolver(target_ledger)
        cloned._decisions = dict(self._decisions)
        cloned._latest_by_competition = dict(self._latest_by_competition)
        return cloned

    def state_key(self) -> tuple:
        """返回决策历史和各竞争组最新指针的完整可比较状态。"""
        return (
            tuple(
                self._decisions[key].stable_key()
                for key in sorted(self._decisions)
            ),
            tuple(sorted(self._latest_by_competition.items())),
        )

    def _validated_directives(
            self, snapshots: tuple[HypothesisSnapshot, ...],
            replacements: tuple[ReplacementDirective, ...],
            archives: tuple[ArchiveDirective, ...],
            ) -> tuple[
                dict[HypothesisKey, ReplacementDirective],
                dict[HypothesisKey, ArchiveDirective],
            ]:
        """核验 replacement/archive 边界、重放历史和 active refute。"""
        by_hypothesis = {item.hypothesis: item for item in snapshots}
        directives: dict[HypothesisKey, ReplacementDirective] = {}
        seen_replacements: set[HypothesisKey] = set()
        for directive in replacements:
            rejected = by_hypothesis.get(directive.rejected)
            replacement = by_hypothesis.get(directive.replacement)
            if rejected is None or replacement is None:
                raise ValueError("replacement directive 越过 resolver 竞争边界")
            if directive.rejected in seen_replacements:
                raise ValueError("同一 rejected 候选不得指定多个 replacement")
            seen_replacements.add(directive.rejected)
            if rejected.lifecycle == LIFECYCLE_SUPERSEDED:
                matching = tuple(
                    event for event in self.ledger.transition_history(
                        directive.rejected)
                    if (event.to_state == LIFECYCLE_SUPERSEDED
                        and event.reason_evidence_id
                        == directive.reason_evidence_id
                        and event.replacement == directive.replacement)
                )
                if not matching:
                    raise ValueError("已 superseded 候选与重放 directive 不一致")
                continue
            if rejected.lifecycle != LIFECYCLE_ACTIVE:
                raise ValueError("只有 active 候选可以被显式替代")
            if directive.reason_evidence_id not in rejected.refute_evidence_ids:
                raise ValueError("replacement 必须引用 rejected 的 active refute")
            if (replacement.lifecycle != LIFECYCLE_ACTIVE
                    or replacement.epistemic_status == EPISTEMIC_REFUTED):
                raise ValueError("replacement 必须是未被纯反驳的 active 候选")
            directives[directive.rejected] = directive

        archive_directives: dict[HypothesisKey, ArchiveDirective] = {}
        seen_archives: set[HypothesisKey] = set()
        for directive in archives:
            rejected = by_hypothesis.get(directive.rejected)
            if rejected is None:
                raise ValueError("archive directive 越过 resolver 竞争边界")
            if directive.rejected in seen_archives:
                raise ValueError("同一 rejected 候选不得重复指定 archive")
            seen_archives.add(directive.rejected)
            if rejected.lifecycle == LIFECYCLE_ARCHIVED:
                matching = tuple(
                    event for event in self.ledger.transition_history(
                        directive.rejected)
                    if (event.from_state == LIFECYCLE_ACTIVE
                        and event.to_state == LIFECYCLE_ARCHIVED
                        and event.reason_evidence_id
                        == directive.reason_evidence_id
                        and event.replacement is None)
                )
                if not matching:
                    raise ValueError("已 archived 候选与重放 directive 不一致")
                continue
            if rejected.lifecycle != LIFECYCLE_ACTIVE:
                raise ValueError("只有 active 候选可以被显式归档")
            if directive.reason_evidence_id not in rejected.refute_evidence_ids:
                raise ValueError("archive 必须引用 rejected 的 active refute")
            archive_directives[directive.rejected] = directive
        return directives, archive_directives

    def _exit_refuted(
            self, snapshots: tuple[HypothesisSnapshot, ...],
            directives: dict[HypothesisKey, ReplacementDirective], *,
            archive_directives: dict[HypothesisKey, ArchiveDirective],
            timestamp_seq: int,
            ) -> dict[HypothesisKey, int]:
        """先验证整批退出事件，再原子式提交到真实 ledger。"""
        planned: list[HypothesisTransition] = []
        for snapshot in snapshots:
            directive = directives.get(snapshot.hypothesis)
            archive_directive = archive_directives.get(snapshot.hypothesis)
            if snapshot.lifecycle != LIFECYCLE_ACTIVE:
                continue
            if (snapshot.epistemic_status != EPISTEMIC_REFUTED
                    and directive is None
                    and archive_directive is None):
                continue
            history = {
                item.evidence_id: item
                for item in self.ledger.evidence_history(snapshot.hypothesis)
            }
            if directive is not None:
                reason = history[directive.reason_evidence_id]
                replacement = directive.replacement
                to_state = LIFECYCLE_SUPERSEDED
            elif archive_directive is not None:
                reason = history[archive_directive.reason_evidence_id]
                replacement = None
                to_state = LIFECYCLE_ARCHIVED
            else:
                reason = max(
                    (history[item] for item in snapshot.refute_evidence_ids),
                    key=lambda item: (item.timestamp_seq, item.evidence_id),
                )
                replacement = None
                to_state = LIFECYCLE_ARCHIVED
            if reason.stance != EVIDENCE_REFUTE:
                raise ValueError("生命周期退出理由必须是 active refute Evidence")
            if timestamp_seq < reason.timestamp_seq:
                raise ValueError("resolver 逻辑序不得早于退出理由 Evidence")
            replacement_key = (
                () if replacement is None else replacement.stable_key())
            event_id = _TRANSITION_HASHER.h63((
                snapshot.hypothesis.stable_key(),
                reason.evidence_id,
                to_state,
                reason.reason_key,
                timestamp_seq,
                replacement_key,
            )) or 1
            planned.append(HypothesisTransition(
                event_id,
                snapshot.hypothesis,
                LIFECYCLE_ACTIVE,
                to_state,
                reason.evidence_id,
                reason.reason_key,
                timestamp_seq,
                replacement,
            ))
        validation_ledger = self.ledger.clone()
        for transition in planned:
            validation_ledger.append_transition(transition)
        transition_ids: dict[HypothesisKey, int] = {}
        for transition in planned:
            self.ledger.append_transition(transition)
            transition_ids[transition.hypothesis] = transition.event_id
        return transition_ids

    @staticmethod
    def _projected_snapshots(
            snapshots: tuple[HypothesisSnapshot, ...],
            directives: dict[HypothesisKey, ReplacementDirective],
            archive_directives: dict[HypothesisKey, ArchiveDirective],
            ) -> tuple[HypothesisSnapshot, ...]:
        """在零写状态下投影本次合法退出，供 scorer 先完成全部校验。"""
        projected: list[HypothesisSnapshot] = []
        for snapshot in snapshots:
            if snapshot.lifecycle != LIFECYCLE_ACTIVE:
                lifecycle = snapshot.lifecycle
            elif snapshot.hypothesis in directives:
                lifecycle = LIFECYCLE_SUPERSEDED
            elif snapshot.hypothesis in archive_directives:
                lifecycle = LIFECYCLE_ARCHIVED
            elif snapshot.epistemic_status == EPISTEMIC_REFUTED:
                lifecycle = LIFECYCLE_ARCHIVED
            else:
                lifecycle = snapshot.lifecycle
            projected.append(HypothesisSnapshot(
                snapshot.hypothesis,
                lifecycle,
                snapshot.epistemic_status,
                snapshot.support_evidence_ids,
                snapshot.refute_evidence_ids,
                snapshot.unknown_evidence_ids,
            ))
        return tuple(projected)

    @staticmethod
    def _eligible_hypotheses(
            snapshots: tuple[HypothesisSnapshot, ...],
            ) -> tuple[HypothesisKey, ...]:
        """优先返回 active supported；若不存在则保留 active unknown 多解。"""
        supported = tuple(
            item.hypothesis for item in snapshots
            if (item.lifecycle == LIFECYCLE_ACTIVE
                and item.epistemic_status == EPISTEMIC_SUPPORTED)
        )
        if supported:
            return tuple(sorted(supported, key=lambda item: item.stable_key()))
        unknown = tuple(
            item.hypothesis for item in snapshots
            if (item.lifecycle == LIFECYCLE_ACTIVE
                and item.epistemic_status == EPISTEMIC_UNKNOWN)
        )
        return tuple(sorted(unknown, key=lambda item: item.stable_key()))

    @staticmethod
    def _collect_preferences(
            eligible: tuple[HypothesisKey, ...],
            scorers: tuple[TypedResolverScorer, ...],
            ) -> tuple[ResolverPreference, ...]:
        """要求每个 scorer 完整覆盖候选对，拒绝部分关系制造隐式优先级。"""
        expected_pairs = {
            (eligible[left], eligible[right])
            for left in range(len(eligible))
            for right in range(left + 1, len(eligible))
        }
        all_preferences: list[ResolverPreference] = []
        scorer_keys: set[tuple[int, ...]] = set()
        for scorer in scorers:
            scorer_key = _integer_key(
                getattr(scorer, "scorer_key", None),
                where="TypedResolverScorer.scorer_key",
            )
            if scorer_key in scorer_keys:
                raise ValueError("同一次解析不得重复使用 scorer_key")
            scorer_keys.add(scorer_key)
            produced = scorer.preferences(eligible)
            if not isinstance(produced, tuple) or any(
                    not isinstance(item, ResolverPreference)
                    for item in produced):
                raise TypeError("typed scorer 必须返回 ResolverPreference tuple")
            by_pair: dict[
                tuple[HypothesisKey, HypothesisKey], ResolverPreference
            ] = {}
            for preference in produced:
                if preference.scorer_key != scorer_key:
                    raise ValueError("preference.scorer_key 与 scorer 不一致")
                pair = (preference.left, preference.right)
                if pair not in expected_pairs:
                    raise ValueError("typed scorer 返回了 eligible 集合外的候选对")
                if pair in by_pair:
                    raise ValueError("typed scorer 不得重复比较同一候选对")
                by_pair[pair] = preference
            if set(by_pair) != expected_pairs:
                raise ValueError("typed scorer 必须完整覆盖全部 eligible 候选对")
            all_preferences.extend(by_pair[pair] for pair in sorted(
                by_pair,
                key=lambda item: (
                    item[0].stable_key(), item[1].stable_key()),
            ))
        return tuple(sorted(
            all_preferences,
            key=lambda item: item.stable_key(),
        ))

    @staticmethod
    def _dominance(
            eligible: tuple[HypothesisKey, ...],
            preferences: tuple[ResolverPreference, ...],
            scorer_count: int,
            ) -> dict[HypothesisKey, tuple[HypothesisKey, ...]]:
        """仅在全部 scorer 同向或等价时建立支配，冲突和不可比均不裁决。"""
        dominated: dict[HypothesisKey, set[HypothesisKey]] = {
            hypothesis: set() for hypothesis in eligible
        }
        if scorer_count == 0:
            return {key: () for key in eligible}
        grouped: dict[
            tuple[HypothesisKey, HypothesisKey], list[int]
        ] = {}
        for preference in preferences:
            grouped.setdefault(
                (preference.left, preference.right), []).append(
                    preference.preference)
        for (left, right), relations in grouped.items():
            if (len(relations) != scorer_count
                    or PREFERENCE_INCOMPARABLE in relations):
                continue
            strict = {
                relation for relation in relations
                if relation != PREFERENCE_EQUIVALENT
            }
            if strict == {PREFERENCE_LEFT_BETTER}:
                dominated[right].add(left)
            elif strict == {PREFERENCE_RIGHT_BETTER}:
                dominated[left].add(right)
        return {
            key: tuple(sorted(value, key=lambda item: item.stable_key()))
            for key, value in dominated.items()
        }

    def _candidate_trace(
            self, before: HypothesisSnapshot, after: HypothesisSnapshot,
            adopted: tuple[HypothesisKey, ...],
            dominated_by: dict[HypothesisKey, tuple[HypothesisKey, ...]],
            previous_roles: dict[HypothesisKey, int],
            transition_ids: dict[HypothesisKey, int],
            ) -> ResolverCandidateTrace:
        """组合一个候选的状态变化、来源账户和消费者角色。"""
        if after.lifecycle != LIFECYCLE_ACTIVE:
            role = RESOLUTION_EXITED
        elif after.hypothesis in adopted:
            role = RESOLUTION_ADOPTED
        else:
            role = RESOLUTION_RETAINED
        return ResolverCandidateTrace(
            before,
            after,
            self._source_accounts(after),
            dominated_by.get(after.hypothesis, ()),
            role,
            previous_roles.get(after.hypothesis, 0),
            transition_ids.get(after.hypothesis, 0),
        )

    def _source_accounts(
            self, snapshot: HypothesisSnapshot,
            ) -> tuple[ResolverSourceAccount, ...]:
        """按完整 SourceRef 汇总当前 active Evidence 引用，不作多数投票。"""
        active_ids = frozenset((
            *snapshot.support_evidence_ids,
            *snapshot.refute_evidence_ids,
            *snapshot.unknown_evidence_ids,
        ))
        by_source: dict[SourceRef, list[EvidenceRecord]] = {}
        for evidence in self.ledger.evidence_history(snapshot.hypothesis):
            if evidence.evidence_id in active_ids:
                by_source.setdefault(evidence.source, []).append(evidence)
        accounts: list[ResolverSourceAccount] = []
        for source in sorted(
                by_source, key=lambda item: item.stable_key()):
            evidence = by_source[source]
            support = tuple(sorted(
                item.evidence_id for item in evidence
                if item.stance == EVIDENCE_SUPPORT))
            refute = tuple(sorted(
                item.evidence_id for item in evidence
                if item.stance == EVIDENCE_REFUTE))
            unknown = tuple(sorted(
                item.evidence_id for item in evidence
                if item.stance not in {EVIDENCE_SUPPORT, EVIDENCE_REFUTE}))
            accounts.append(ResolverSourceAccount(
                source,
                _epistemic_status(support, refute),
                support,
                refute,
                unknown,
            ))
        return tuple(accounts)

    def _previous_roles(self, decision_id: int) -> dict[HypothesisKey, int]:
        """读取上一决策的候选投影，供降级与替代链审计。"""
        if decision_id == 0:
            return {}
        previous = self._decisions.get(decision_id)
        if previous is None:
            raise ValueError("resolver 最新决策指针引用了缺失事件")
        return {
            candidate.hypothesis: candidate.role
            for candidate in previous.candidates
        }


__all__ = [
    "ArchiveDirective",
    "HypothesisResolver",
    "PREFERENCE_EQUIVALENT",
    "PREFERENCE_INCOMPARABLE",
    "PREFERENCE_LEFT_BETTER",
    "PREFERENCE_RIGHT_BETTER",
    "RESOLUTION_ADOPTED",
    "RESOLUTION_EXITED",
    "RESOLUTION_RETAINED",
    "ReplacementDirective",
    "ResolverCandidateTrace",
    "ResolverDecision",
    "ResolverPreference",
    "ResolverSourceAccount",
    "TypedResolverScorer",
]
