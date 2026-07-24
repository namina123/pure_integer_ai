"""M-09 记忆巩固、冲突观察、生命周期仲裁和物理放置提示。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ArchiveDirective,
    HypothesisResolver,
    ReplacementDirective,
    ResolverDecision,
    TypedResolverScorer,
)
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MemoryHypothesisAggregateIndex,
)
from pure_integer_ai.cognition.shared.memory_decay import (
    ActivationDecayAssessment,
    ActivationDecayPolicy,
    MemoryActivationSnapshot,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_USE,
    MEMORY_EVENT_USE_OUTCOME,
    MEMORY_OBJECT_EVIDENCE,
    MEMORY_OBJECT_HYPOTHESIS,
    MEMORY_OBJECT_USE,
    EvidencePayload,
    DerivationTransitionPayload,
    HypothesisPayload,
    MemoryEvent,
    MemoryObjectRef,
    RetentionTransitionPayload,
    UseOutcomePayload,
    UsePayload,
    MEMORY_EVENT_RETENTION,
    RETENTION_CONSOLIDATED,
    RETENTION_EPISODIC,
)
from pure_integer_ai.cognition.shared.memory_event_log import (
    MaterializedMemoryEvent,
    MemoryEventLog,
)
from pure_integer_ai.cognition.shared.memory_hypothesis import (
    MemoryHypothesisEventSink,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_RETENTION,
    LogicalClockIdentity,
    LogicalTimestamp,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.placement import TemperatureProfile
from pure_integer_ai.storage.memory_aggregate import (
    MemoryHypothesisAggregateRecord,
)
from pure_integer_ai.storage.storage_role import (
    StorageRoleRegistry,
)


_MAINTENANCE_PROTOCOL_VERSION = 1


def _key(
        value: tuple[int, ...], *, label: str,
        allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """校验 M-09 中开放的纯整数键。"""
    if not isinstance(value, tuple) or (not allow_empty and not value):
        raise ValueError(f"{label} 必须是{'可空' if allow_empty else '非空'}整数 tuple")
    if value:
        assert_int(*value, _where=label)
        if any(type(item) is not int for item in value):
            raise ValueError(f"{label} 必须使用严格整数")
    return value


def _component_key(component: object, *, label: str) -> tuple[int, ...]:
    """读取注入组件的版本身份。"""
    method = getattr(component, "state_key", None)
    if not callable(method):
        raise TypeError(f"{label} 缺少 state_key")
    return _key(method(), label=f"{label}.state_key")


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长整数键增加长度边界。"""
    return len(value), *value


@dataclass(frozen=True, order=True)
class MemoryConflictRatio:
    """不使用除法表达的冲突比例，分子分母均可回溯。"""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        """核验分数非负且零分母只能对应零分子。"""
        assert_int(
            self.numerator,
            self.denominator,
            _where="MemoryConflictRatio",
        )
        if (type(self.numerator) is not int
                or type(self.denominator) is not int
                or self.numerator < 0
                or self.denominator < 0
                or self.numerator > self.denominator
                or (self.denominator == 0 and self.numerator != 0)):
            raise ValueError("冲突比例分子分母非法")

    def at_most(self, numerator: int, denominator: int) -> bool:
        """用交叉乘法比较两个非负比例，不产生浮点数。"""
        assert_int(numerator, denominator, _where="MemoryConflictRatio.compare")
        if (type(numerator) is not int or type(denominator) is not int
                or numerator < 0 or denominator < 0
                or numerator > denominator):
            raise ValueError("待比较冲突比例非法")
        if denominator == 0:
            return self.numerator == 0
        if self.denominator == 0:
            return True
        return self.numerator * denominator <= numerator * self.denominator

    def stable_key(self) -> tuple[int, int]:
        """返回分子和分母。"""
        return self.numerator, self.denominator


@dataclass(frozen=True)
class MemoryEvidenceActivity:
    """一个未被替代的 Evidence 及其统一 Memory 时间线位置。"""

    reference: MemoryObjectRef
    payload: EvidencePayload
    timeline_seq: int

    def __post_init__(self) -> None:
        """核验 Evidence 引用、payload 和时间线一致。"""
        if (not isinstance(self.reference, MemoryObjectRef)
                or self.reference.object_kind != MEMORY_OBJECT_EVIDENCE):
            raise ValueError("MemoryEvidenceActivity reference 必须指向 Evidence")
        if not isinstance(self.payload, EvidencePayload):
            raise TypeError("MemoryEvidenceActivity payload 类型错误")
        if type(self.timeline_seq) is not int or self.timeline_seq <= 0:
            raise ValueError("Evidence timeline_seq 必须为正严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回 Evidence 完整身份和时间线。"""
        return (
            *_packed(self.reference.stable_key()),
            self.timeline_seq,
            *_packed(self.payload.stable_key()),
        )


@dataclass(frozen=True)
class MemoryUseOutcomeActivity:
    """一个精确 Use 的延迟 outcome 及其统一时间线位置。"""

    event_reference: MemoryObjectRef
    payload: UseOutcomePayload
    timeline_seq: int

    def __post_init__(self) -> None:
        """核验 outcome 事件引用和 target Use 对齐。"""
        if (not isinstance(self.event_reference, MemoryObjectRef)
                or self.event_reference.object_kind != MEMORY_OBJECT_USE):
            raise ValueError("UseOutcome activity reference 必须指向 Use")
        if not isinstance(self.payload, UseOutcomePayload):
            raise TypeError("UseOutcome activity payload 类型错误")
        if self.payload.target_ref != self.event_reference:
            raise ValueError("UseOutcome target 与事件引用漂移")
        if type(self.timeline_seq) is not int or self.timeline_seq <= 0:
            raise ValueError("UseOutcome timeline_seq 必须为正严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回 outcome 完整身份和时间线。"""
        return (
            self.timeline_seq,
            *_packed(self.payload.stable_key()),
        )


@dataclass(frozen=True)
class MemoryUseActivity:
    """一个实际使用 Hypothesis 的 Use 及其延迟结果集合。"""

    reference: MemoryObjectRef
    payload: UsePayload
    timeline_seq: int
    outcomes: tuple[MemoryUseOutcomeActivity, ...]

    def __post_init__(self) -> None:
        """核验 Use 目标、结果 target 和确定性排序。"""
        if (not isinstance(self.reference, MemoryObjectRef)
                or self.reference.object_kind != MEMORY_OBJECT_USE):
            raise ValueError("MemoryUseActivity reference 必须指向 Use")
        if not isinstance(self.payload, UsePayload):
            raise TypeError("MemoryUseActivity payload 类型错误")
        if self.payload.memory_ref.object_kind != MEMORY_OBJECT_HYPOTHESIS:
            raise ValueError("M-09 当前只维护 Hypothesis Use")
        if type(self.timeline_seq) is not int or self.timeline_seq <= 0:
            raise ValueError("Use timeline_seq 必须为正严格整数")
        if (not isinstance(self.outcomes, tuple)
                or any(not isinstance(item, MemoryUseOutcomeActivity)
                       for item in self.outcomes)):
            raise TypeError("Use outcomes 类型错误")
        if self.outcomes != tuple(sorted(
                self.outcomes,
                key=lambda item: (
                    item.timeline_seq, item.payload.stable_key()))):
            raise ValueError("Use outcomes 必须按 timeline 稳定排序")


@dataclass(frozen=True)
class MemoryMaintenanceSnapshot:
    """M-09 策略可读的完整 Hypothesis 活动、证据和冲突快照。"""

    hypothesis_ref: MemoryObjectRef
    hypothesis: HypothesisKey
    aggregate: MemoryHypothesisAggregateRecord
    as_of: LogicalTimestamp
    active_evidence: tuple[MemoryEvidenceActivity, ...]
    uses: tuple[MemoryUseActivity, ...]
    evidence_conflict_ratio: MemoryConflictRatio
    source_conflict_ratio: MemoryConflictRatio
    distinct_context_count: int

    def __post_init__(self) -> None:
        """核验快照的对象身份、时间线和活动集合边界。"""
        if (not isinstance(self.hypothesis_ref, MemoryObjectRef)
                or self.hypothesis_ref.object_kind != MEMORY_OBJECT_HYPOTHESIS):
            raise ValueError("maintenance snapshot 必须指向 Hypothesis")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("maintenance snapshot 缺少 HypothesisKey")
        if self.hypothesis_ref.object_key != self.hypothesis.stable_key():
            raise ValueError("Hypothesis 完整身份与 Memory ref 漂移")
        if not isinstance(
                self.aggregate, MemoryHypothesisAggregateRecord):
            raise TypeError("maintenance snapshot 缺少 aggregate")
        if not isinstance(self.as_of, LogicalTimestamp):
            raise TypeError("maintenance snapshot.as_of 类型错误")
        if self.as_of.seq < max(
                self.aggregate.created_seq,
                self.aggregate.last_observed_seq,
                self.aggregate.last_used_seq,
                ):
            raise ValueError("maintenance as_of 早于 aggregate 最近活动")
        if (not isinstance(self.active_evidence, tuple)
                or any(not isinstance(item, MemoryEvidenceActivity)
                       for item in self.active_evidence)):
            raise TypeError("maintenance active_evidence 类型错误")
        if self.active_evidence != tuple(sorted(
                self.active_evidence, key=lambda item: item.stable_key())):
            raise ValueError("maintenance active_evidence 必须稳定排序")
        if any(
                item.payload.hypothesis_ref != self.hypothesis_ref
                or item.timeline_seq > self.as_of.seq
                for item in self.active_evidence):
            raise ValueError("maintenance Evidence 越过目标或 as_of")
        if (not isinstance(self.uses, tuple)
                or any(not isinstance(item, MemoryUseActivity)
                       for item in self.uses)):
            raise TypeError("maintenance uses 类型错误")
        if self.uses != tuple(sorted(
                self.uses,
                key=lambda item: (
                    item.timeline_seq, item.reference.stable_key()))):
            raise ValueError("maintenance uses 必须稳定排序")
        if any(
                item.payload.memory_ref != self.hypothesis_ref
                or item.timeline_seq > self.as_of.seq
                or any(outcome.timeline_seq > self.as_of.seq
                       for outcome in item.outcomes)
                for item in self.uses):
            raise ValueError("maintenance Use 越过目标或 as_of")
        if not isinstance(
                self.evidence_conflict_ratio, MemoryConflictRatio):
            raise TypeError("evidence_conflict_ratio 类型错误")
        if not isinstance(self.source_conflict_ratio, MemoryConflictRatio):
            raise TypeError("source_conflict_ratio 类型错误")
        expected_evidence_ratio = MemoryConflictRatio(
            self.aggregate.contradict_count,
            self.aggregate.support_count + self.aggregate.contradict_count,
        )
        expected_source_ratio = MemoryConflictRatio(
            self.aggregate.contradict_source_count,
            self.aggregate.support_source_count
            + self.aggregate.contradict_source_count,
        )
        if (self.evidence_conflict_ratio != expected_evidence_ratio
                or self.source_conflict_ratio != expected_source_ratio):
            raise ValueError("maintenance 冲突比例与 aggregate 漂移")
        if type(self.distinct_context_count) is not int or self.distinct_context_count < 0:
            raise ValueError("distinct_context_count 必须为非负严格整数")
        contexts = {
            item.payload.context_key
            for item in self.uses
            if item.payload.context_key
        }
        if self.distinct_context_count != len(contexts):
            raise ValueError("distinct_context_count 与 Use 上下文漂移")

    def stable_key(self) -> tuple[int, ...]:
        """返回可复核当前维护输入的完整键。"""
        aggregate = self.aggregate
        return (
            _MAINTENANCE_PROTOCOL_VERSION,
            *_packed(self.hypothesis_ref.stable_key()),
            aggregate.hypothesis_hash,
            aggregate.created_seq,
            aggregate.last_observed_seq,
            aggregate.last_used_seq,
            aggregate.support_count,
            aggregate.contradict_count,
            aggregate.use_count,
            aggregate.retention_state,
            aggregate.lifecycle_state,
            aggregate.evidence_state,
            *_packed(self.as_of.stable_key()),
            len(self.active_evidence),
            *(value for item in self.active_evidence for value in (
                len(item.stable_key()), *item.stable_key())),
            len(self.uses),
            *(value for item in self.uses for value in (
                len(item.reference.stable_key()),
                *item.reference.stable_key(),
                item.timeline_seq,
                len(item.outcomes),
                *(value for outcome in item.outcomes for value in (
                    len(outcome.stable_key()), *outcome.stable_key())),
            )),
        )


@dataclass(frozen=True)
class MemoryRetentionDecision:
    """注入巩固策略的显式决定，不猜测领域阈值或真值。"""

    consolidate: bool
    reason_evidence_refs: tuple[MemoryObjectRef, ...]
    reason_key: tuple[int, ...]
    policy_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验决定、真实 Evidence 理由和策略身份。"""
        if type(self.consolidate) is not bool:
            raise TypeError("retention consolidate 必须是 bool")
        if (not isinstance(self.reason_evidence_refs, tuple)
                or any(
                    not isinstance(item, MemoryObjectRef)
                    or item.object_kind != MEMORY_OBJECT_EVIDENCE
                    for item in self.reason_evidence_refs)):
            raise TypeError("retention reason 必须是 Evidence ref tuple")
        if len(set(self.reason_evidence_refs)) != len(self.reason_evidence_refs):
            raise ValueError("retention reason Evidence 不得重复")
        if self.consolidate and not self.reason_evidence_refs:
            raise ValueError("巩固决定必须引用真实 Evidence")
        if not self.consolidate and self.reason_evidence_refs:
            raise ValueError("不巩固决定不得携带 Evidence 理由")
        _key(self.reason_key, label="retention reason_key")
        _key(self.policy_key, label="retention policy_key")


class MemoryRetentionPolicy(Protocol):
    """按完整快照决定是否巩固的注入协议。"""

    def state_key(self) -> tuple[int, ...]:
        """返回 kind-specific 规则和参数的稳定身份。"""
        ...

    def assess(
            self,
            snapshot: MemoryMaintenanceSnapshot,
            ) -> MemoryRetentionDecision:
        """返回决定，不直接写入 retention 或 Evidence。"""
        ...


@dataclass(frozen=True)
class MemoryPlacementHint:
    """仅供 K 线继续规划的物理位置候选，不执行迁移。"""

    object_key: tuple[int, ...]
    descriptor_key: tuple[int, ...]
    preferred_tier_key: tuple[int, ...]
    temperature_profile_key: tuple[int, ...]
    reason_key: tuple[int, ...]
    policy_key: tuple[int, ...]
    as_of_seq: int

    def __post_init__(self) -> None:
        """核验 hint 只携带完整 K-01 身份和非负时间线。"""
        _key(self.object_key, label="placement hint object_key")
        _key(self.descriptor_key, label="placement hint descriptor_key")
        _key(self.preferred_tier_key, label="placement hint tier_key")
        _key(self.temperature_profile_key, label="placement hint profile_key")
        _key(self.reason_key, label="placement hint reason_key")
        _key(self.policy_key, label="placement hint policy_key")
        if type(self.as_of_seq) is not int or self.as_of_seq <= 0:
            raise ValueError("placement hint as_of_seq 必须为正严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含文字语义的完整物理 hint 键。"""
        return (
            _MAINTENANCE_PROTOCOL_VERSION,
            *_packed(self.object_key),
            *_packed(self.descriptor_key),
            *_packed(self.preferred_tier_key),
            *_packed(self.temperature_profile_key),
            *_packed(self.reason_key),
            *_packed(self.policy_key),
            self.as_of_seq,
        )


class MemoryPlacementHintPolicy(Protocol):
    """按维护快照提出物理位置候选的注入协议。"""

    def state_key(self) -> tuple[int, ...]:
        """返回 hint 规则和参数的稳定身份。"""
        ...

    def hints(
            self,
            snapshot: MemoryMaintenanceSnapshot,
            ) -> tuple[MemoryPlacementHint, ...]:
        """返回候选 hint，不写事件、不改 manifest、不迁移数据。"""
        ...


@dataclass(frozen=True)
class MemoryMaintenanceAssessment:
    """一次无写入 M-09 维护评估的完整结果。"""

    snapshot: MemoryMaintenanceSnapshot
    activation: ActivationDecayAssessment
    retention: MemoryRetentionDecision
    placement_hints: tuple[MemoryPlacementHint, ...]


@dataclass(frozen=True)
class MemoryRetentionCommit:
    """一次巩固尝试的前后状态及可选 retention 事件。"""

    assessment: MemoryMaintenanceAssessment
    retention_event: MaterializedMemoryEvent | None
    after: MemoryHypothesisAggregateRecord


class MemoryMaintenanceService:
    """在 M-04、H-04 和 K-01 边界内执行 M-09 维护。"""

    def __init__(
            self,
            aggregates: MemoryHypothesisAggregateIndex,
            activation_policy: ActivationDecayPolicy,
            retention_policy: MemoryRetentionPolicy,
            placement_policy: MemoryPlacementHintPolicy,
            storage_roles: StorageRoleRegistry,
            temperature_profile: TemperatureProfile,
            ) -> None:
        """绑定同一 Memory 空间的策略、角色注册表和温层 profile。"""
        if not isinstance(aggregates, MemoryHypothesisAggregateIndex):
            raise TypeError("maintenance aggregates 类型错误")
        for label, component, method in (
                ("activation_policy", activation_policy, "assess"),
                ("retention_policy", retention_policy, "assess"),
                ("placement_policy", placement_policy, "hints")):
            if not callable(getattr(component, method, None)):
                raise TypeError(f"{label} 缺少 {method}")
            _component_key(component, label=label)
        if not isinstance(storage_roles, StorageRoleRegistry):
            raise TypeError("storage_roles 类型错误")
        if not isinstance(temperature_profile, TemperatureProfile):
            raise TypeError("temperature_profile 类型错误")
        self.aggregates = aggregates
        self.activation_policy = activation_policy
        self.retention_policy = retention_policy
        self.placement_policy = placement_policy
        self.storage_roles = storage_roles
        self.temperature_profile = temperature_profile

    @property
    def event_log(self) -> MemoryEventLog:
        """返回 M-09 绑定的 Memory 事件日志。"""
        return self.aggregates.event_log

    def state_key(self) -> tuple[int, ...]:
        """返回全部策略、空间、角色和温层身份。"""
        return (
            _MAINTENANCE_PROTOCOL_VERSION,
            *_packed(self.event_log.memory_space_identity.stable_key()),
            *_packed(_component_key(
                self.activation_policy, label="activation_policy")),
            *_packed(_component_key(
                self.retention_policy, label="retention_policy")),
            *_packed(_component_key(
                self.placement_policy, label="placement_policy")),
            *_packed(self.storage_roles.stable_key()),
            *_packed(self.temperature_profile.stable_key()),
        )

    def assess(
            self,
            hypothesis_ref: MemoryObjectRef,
            *,
            access: MemoryAccessContext,
            as_of: LogicalTimestamp | None = None,
            ) -> MemoryMaintenanceAssessment:
        """读取一个干净 Hypothesis，计算衰减、巩固和 placement hint。"""
        if (not isinstance(hypothesis_ref, MemoryObjectRef)
                or hypothesis_ref.object_kind != MEMORY_OBJECT_HYPOTHESIS):
            raise ValueError("M-09 assess 必须接收 Hypothesis ref")
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("M-09 assess 缺少 MemoryAccessContext")
        self.aggregates.require_hypothesis_clean(
            hypothesis_ref, access=access)
        aggregate = self.aggregates.read(hypothesis_ref, access=access)
        if aggregate is None:
            raise ValueError("M-09 assess 找不到可见 aggregate")
        current = self.event_log.timeline_watermark()
        if current is None:
            raise RuntimeError("M-09 assess 找不到 Memory timeline")
        if as_of is None:
            as_of = current
        if as_of != current:
            raise ValueError("M-09 as_of 必须等于当前 Memory timeline 水位")
        declaration = self._hypothesis_declaration(
            hypothesis_ref, access=access)
        hypothesis = declaration.event.payload.hypothesis
        snapshot = self._snapshot(
            hypothesis_ref, hypothesis, aggregate, as_of, access=access)
        activation_snapshot = MemoryActivationSnapshot(
            hypothesis, aggregate, as_of)
        activation = self.activation_policy.assess(activation_snapshot)
        expected_activation_key = _component_key(
            self.activation_policy, label="activation_policy")
        if (not isinstance(activation, ActivationDecayAssessment)
                or activation.policy_key != expected_activation_key):
            raise ValueError("activation policy 返回身份漂移")
        retention = self.retention_policy.assess(snapshot)
        expected_retention_key = _component_key(
            self.retention_policy, label="retention_policy")
        if (not isinstance(retention, MemoryRetentionDecision)
                or retention.policy_key != expected_retention_key):
            raise ValueError("retention policy 返回身份漂移")
        active_refs = {item.reference for item in snapshot.active_evidence}
        if not set(retention.reason_evidence_refs) <= active_refs:
            raise ValueError("retention 决定必须引用当前活动 Evidence")
        hints = self.placement_policy.hints(snapshot)
        if not isinstance(hints, tuple) or any(
                not isinstance(item, MemoryPlacementHint) for item in hints):
            raise TypeError("placement policy 必须返回 hint tuple")
        for hint in hints:
            self._validate_hint(hint, hypothesis_ref, as_of)
        if hints != tuple(sorted(hints, key=lambda item: item.stable_key())):
            raise ValueError("placement hints 必须按稳定键排序")
        return MemoryMaintenanceAssessment(
            snapshot, activation, retention, hints)

    def consolidate(
            self,
            hypothesis_ref: MemoryObjectRef,
            *,
            access: MemoryAccessContext,
            as_of: LogicalTimestamp | None = None,
            ) -> MemoryRetentionCommit:
        """按策略追加一次 retention 转换，绝不改写旧事件或生命周期。"""
        assessment = self.assess(
            hypothesis_ref, access=access, as_of=as_of)
        current = assessment.snapshot.aggregate
        if not assessment.retention.consolidate:
            return MemoryRetentionCommit(assessment, None, current)
        if current.retention_state == RETENTION_CONSOLIDATED:
            return MemoryRetentionCommit(assessment, None, current)
        if current.retention_state != RETENTION_EPISODIC:
            raise ValueError("未知 retention state 不允许 M-09 巩固")
        clock = self.event_log.scoped_identities.resume_clock(
            LogicalClockIdentity(
                assessment.snapshot.hypothesis.scope,
                CLOCK_MEMORY_RETENTION,
            ))
        changed_at = clock.advance()
        payload = RetentionTransitionPayload(
            hypothesis_ref,
            RETENTION_EPISODIC,
            RETENTION_CONSOLIDATED,
            assessment.retention.reason_evidence_refs,
            changed_at,
        )
        event = self.event_log.append(MemoryEvent(
            MEMORY_EVENT_RETENTION,
            hypothesis_ref,
            assessment.snapshot.hypothesis.scope,
            payload,
        ))
        after = self.aggregates.rebuild(hypothesis_ref, access=access)
        return MemoryRetentionCommit(assessment, event, after)

    def resolve_lifecycle(
            self,
            anchor: HypothesisKey,
            *,
            access: MemoryAccessContext,
            timestamp_seq: int,
            scorers: tuple[TypedResolverScorer, ...] = (),
            replacements: tuple[ReplacementDirective, ...] = (),
            archives: tuple[ArchiveDirective, ...] = (),
            ) -> ResolverDecision:
        """恢复完整 H-04 竞争组并经真实 Evidence 提交 archive/supersede。"""
        if not isinstance(anchor, HypothesisKey):
            raise TypeError("lifecycle anchor 必须是 HypothesisKey")
        if type(timestamp_seq) is not int or timestamp_seq < 0:
            raise ValueError("lifecycle timestamp_seq 必须为非负严格整数")
        candidate_records = self.aggregates.query(
            access=access,
            hypothesis_kind=anchor.hypothesis_kind,
            context=anchor.scope.stable_key(),
            observation_source=anchor.observation,
        )
        candidates = tuple(
            self._hypothesis_for_aggregate(item, access=access)
            for item in candidate_records
        )
        candidates = tuple(sorted(
            (item for item in candidates
             if item.competition_key == anchor.competition_key),
            key=lambda item: item.stable_key(),
        ))
        if anchor not in candidates:
            raise ValueError("lifecycle anchor 不在完整 Memory 竞争组")
        if not candidates:
            raise ValueError("lifecycle 竞争组为空")
        sink = MemoryHypothesisEventSink(self.event_log)
        indexed_events: dict[int, MaterializedMemoryEvent] = {}
        for hypothesis in candidates:
            ref = self._hypothesis_ref_for_key(hypothesis)
            self.aggregates.require_hypothesis_clean(
                ref, access=access)
            for entry in self.aggregates.events(ref, access=access):
                previous = indexed_events.get(entry.event_hash)
                if previous is not None and previous != entry:
                    raise ValueError("竞争组 event hash 命中不同事件")
                indexed_events[entry.event_hash] = entry
        event_tuple = tuple(sorted(
            indexed_events.values(),
            key=lambda item: (item.timeline.seq, item.event_hash),
        ))
        ledger = sink.load_indexed_ledger(
            hypotheses=candidates,
            events=event_tuple,
            attach_sink=True,
        )
        decisions = sink.load_indexed_decisions(
            hypotheses=candidates,
            events=event_tuple,
        )
        resolver = HypothesisResolver.from_history(
            ledger, decisions, sink=sink)
        decision = resolver.resolve(
            anchor,
            timestamp_seq=timestamp_seq,
            scorers=scorers,
            replacements=replacements,
            archives=archives,
        )
        for hypothesis in candidates:
            ref = self._hypothesis_ref_for_key(hypothesis)
            self.aggregates.rebuild(ref, access=access)
        return decision

    def _snapshot(
            self,
            hypothesis_ref: MemoryObjectRef,
            hypothesis: HypothesisKey,
            aggregate,
            as_of: LogicalTimestamp,
            *,
            access: MemoryAccessContext,
            ) -> MemoryMaintenanceSnapshot:
        """从 M-04 反向索引一次性构造活动 Evidence、Use 和 outcome。"""
        events = self.aggregates.events(hypothesis_ref, access=access)
        evidence_items = tuple(
            item for item in events
            if item.event.event_kind == MEMORY_EVENT_EVIDENCE
            and isinstance(item.event.payload, EvidencePayload)
        )
        evidence_refs = {item.event.object_ref for item in evidence_items}
        superseded_refs = {
            item.event.payload.supersedes_ref
            for item in evidence_items
            if item.event.payload.supersedes_ref is not None
        }
        inactive_derivations = {
            item.event.payload.target_ref
            for item in events
            if isinstance(item.event.payload, DerivationTransitionPayload)
            and item.event.payload.target_ref.object_kind == MEMORY_OBJECT_EVIDENCE
        }
        if not inactive_derivations <= evidence_refs:
            raise ValueError("Evidence derivation inactive target 越界")
        active = tuple(
            MemoryEvidenceActivity(
                item.event.object_ref,
                item.event.payload,
                item.timeline.seq,
            )
            for item in evidence_items
            if (item.event.object_ref not in superseded_refs
                and item.event.object_ref not in inactive_derivations)
        )
        uses: dict[MemoryObjectRef, tuple[MaterializedMemoryEvent, UsePayload]] = {}
        outcomes: dict[MemoryObjectRef, list[MemoryUseOutcomeActivity]] = {}
        for item in events:
            if (item.event.event_kind == MEMORY_EVENT_USE
                    and isinstance(item.event.payload, UsePayload)):
                if item.event.payload.memory_ref != hypothesis_ref:
                    raise ValueError("Use event 越过当前 Hypothesis")
                uses[item.event.object_ref] = item, item.event.payload
            elif (item.event.event_kind == MEMORY_EVENT_USE_OUTCOME
                  and isinstance(item.event.payload, UseOutcomePayload)):
                outcomes.setdefault(item.event.payload.target_ref, []).append(
                    MemoryUseOutcomeActivity(
                        item.event.object_ref,
                        item.event.payload,
                        item.timeline.seq,
                    ))
        use_activities = tuple(
            MemoryUseActivity(
                ref,
                payload,
                materialized.timeline.seq,
                tuple(sorted(
                    outcomes.get(ref, []),
                    key=lambda item: (
                        item.timeline_seq, item.payload.stable_key()))),
            )
            for ref, (materialized, payload) in sorted(
                uses.items(), key=lambda item: (
                    item[1][0].timeline.seq, item[0].stable_key()))
        )
        if not set(outcomes) <= set(uses):
            raise ValueError("UseOutcome 缺少当前 Hypothesis 的目标 Use")
        context_keys = {
            use.payload.context_key
            for use in use_activities
            if use.payload.context_key
        }
        evidence_denominator = (
            aggregate.support_count + aggregate.contradict_count)
        source_denominator = (
            aggregate.support_source_count + aggregate.contradict_source_count)
        return MemoryMaintenanceSnapshot(
            hypothesis_ref,
            hypothesis,
            aggregate,
            as_of,
            tuple(sorted(active, key=lambda item: item.stable_key())),
            use_activities,
            MemoryConflictRatio(
                aggregate.contradict_count, evidence_denominator),
            MemoryConflictRatio(
                aggregate.contradict_source_count, source_denominator),
            len(context_keys),
        )

    def _validate_hint(
            self,
            hint: MemoryPlacementHint,
            hypothesis_ref: MemoryObjectRef,
            as_of: LogicalTimestamp,
            ) -> None:
        """核验 hint 不越过对象、角色、温层或策略身份边界。"""
        if hint.object_key != hypothesis_ref.stable_key():
            raise ValueError("placement hint 不得指向其他 Memory 对象")
        self.storage_roles.get(hint.descriptor_key)
        if (hint.temperature_profile_key
                != self.temperature_profile.profile_key):
            raise ValueError("placement hint temperature profile 漂移")
        if not self.temperature_profile.has(hint.preferred_tier_key):
            raise ValueError("placement hint tier 不属于当前 profile")
        if hint.policy_key != _component_key(
                self.placement_policy, label="placement_policy"):
            raise ValueError("placement hint policy 身份漂移")
        if hint.as_of_seq != as_of.seq:
            raise ValueError("placement hint 时间线漂移")

    def _hypothesis_declaration(
            self,
            ref: MemoryObjectRef,
            *,
            access: MemoryAccessContext,
            ) -> MaterializedMemoryEvent:
        """恢复一个 Hypothesis 的唯一声明事件。"""
        events = self.event_log.query(
            access=access,
            event_kind=MEMORY_EVENT_HYPOTHESIS,
            object_ref=ref,
        )
        if len(events) != 1 or not isinstance(
                events[0].event.payload, HypothesisPayload):
            raise ValueError("Hypothesis 没有唯一声明")
        return events[0]

    def _hypothesis_for_aggregate(
            self,
            aggregate,
            *,
            access: MemoryAccessContext,
            ) -> HypothesisKey:
        """从 M-04 aggregate 恢复完整 HypothesisKey。"""
        ref = self.aggregates.hypothesis_ref_for_aggregate(
            aggregate, access=access)
        if ref is None:
            raise ValueError("aggregate 不可见")
        return self._hypothesis_declaration(ref, access=access).event.payload.hypothesis

    def _hypothesis_ref_for_key(
            self,
            hypothesis: HypothesisKey,
            ) -> MemoryObjectRef:
        """按完整 HypothesisKey 构造当前 Memory 空间引用。"""
        return MemoryObjectRef(
            self.event_log.memory_space_identity,
            hypothesis.observation.owner,
            hypothesis.observation.versions,
            MEMORY_OBJECT_HYPOTHESIS,
            hypothesis.stable_key(),
        )


__all__ = [
    "MemoryConflictRatio",
    "MemoryEvidenceActivity",
    "MemoryMaintenanceAssessment",
    "MemoryMaintenanceService",
    "MemoryMaintenanceSnapshot",
    "MemoryPlacementHint",
    "MemoryPlacementHintPolicy",
    "MemoryRetentionCommit",
    "MemoryRetentionDecision",
    "MemoryRetentionPolicy",
    "MemoryUseActivity",
    "MemoryUseOutcomeActivity",
]
