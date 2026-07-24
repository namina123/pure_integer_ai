"""H-00 HypothesisLedger 与 M-03 Memory 事件之间的持久化适配。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.hypothesis import (
    EvidenceRecord,
    HypothesisEventSink,
    HypothesisKey,
    HypothesisLedger,
    HypothesisTransition,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ResolverDecision,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_LIFECYCLE,
    MEMORY_EVENT_RESOLUTION,
    MEMORY_OBJECT_EVIDENCE,
    MEMORY_OBJECT_HYPOTHESIS,
    MEMORY_OBJECT_RESOLUTION,
    EvidencePayload,
    HypothesisPayload,
    LifecycleTransitionPayload,
    MemoryEvent,
    MemoryObjectRef,
    ResolutionPayload,
    RETENTION_EPISODIC,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_event_log import (
    MaterializedMemoryEvent,
    MemoryEventLog,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_CREATED,
    CLOCK_MEMORY_LIFECYCLE,
    CLOCK_MEMORY_OBSERVED,
    CLOCK_MEMORY_RESOLVED,
    LogicalClockIdentity,
    LogicalTimestamp,
)


class MemoryHypothesisAdapterError(RuntimeError):
    """H-00 兼容事件无法无损映射或重建。"""


class MemoryHypothesisEventSink(HypothesisEventSink):
    """把 H-00 候选、Evidence 和生命周期转换写入一个 MemoryEventLog。"""

    def __init__(self, event_log: MemoryEventLog) -> None:
        """绑定事件日志，并初始化仅作幂等核验的 H-00 引用缓存。"""
        if not isinstance(event_log, MemoryEventLog):
            raise TypeError("event_log 必须是 MemoryEventLog")
        self.event_log = event_log
        self._evidence_by_ledger_id: dict[
            tuple[tuple[int, ...], int], tuple[EvidenceRecord, MemoryObjectRef]
        ] = {}
        self._transition_by_ledger_id: dict[
            tuple[tuple[int, ...], int], HypothesisTransition
        ] = {}
        self._decision_by_ledger_id: dict[
            tuple[tuple[int, ...], int], ResolverDecision
        ] = {}
        self._hydrated_compatibility_boundaries: set[tuple[int, ...]] = set()
        self._hydrated_decision_boundaries: set[tuple[int, ...]] = set()

    def append_hypothesis(self, hypothesis: HypothesisKey) -> None:
        """幂等声明 EPISODIC/ACTIVE Hypothesis，并分配持久 created 时钟。"""
        if not isinstance(hypothesis, HypothesisKey):
            raise TypeError("append_hypothesis 需要 HypothesisKey")
        ref = self._hypothesis_ref(hypothesis)
        existing = self.event_log.query(
            access=self._access(hypothesis),
            event_kind=MEMORY_EVENT_HYPOTHESIS,
            object_ref=ref,
        )
        if existing:
            if len(existing) != 1:
                raise MemoryHypothesisAdapterError(
                    "Hypothesis 存在重复声明事件")
            payload = existing[0].event.payload
            if (not isinstance(payload, HypothesisPayload)
                    or payload.hypothesis != hypothesis):
                raise MemoryHypothesisAdapterError(
                    "Hypothesis 声明事件内容漂移")
            return
        clock = self.event_log.scoped_identities.resume_clock(
            LogicalClockIdentity(hypothesis.scope, CLOCK_MEMORY_CREATED))
        created_at = clock.advance()
        payload = HypothesisPayload(
            hypothesis,
            RETENTION_EPISODIC,
            LIFECYCLE_ACTIVE,
            created_at,
        )
        self.event_log.append(MemoryEvent(
            MEMORY_EVENT_HYPOTHESIS,
            ref,
            hypothesis.scope,
            payload,
        ))

    def append_evidence(self, evidence: EvidenceRecord) -> None:
        """把 H-00 Evidence 写成显式 legacy-reason Evidence，不丢原稳定键。"""
        if not isinstance(evidence, EvidenceRecord):
            raise TypeError("append_evidence 需要 EvidenceRecord")
        boundary = self._ledger_event_key(evidence.hypothesis, 1)[0]
        if boundary not in self._hydrated_compatibility_boundaries:
            self._hydrate_compatibility(evidence.hypothesis)
        cache_key = self._ledger_event_key(
            evidence.hypothesis, evidence.evidence_id)
        existing = self._evidence_by_ledger_id.get(cache_key)
        if existing is not None:
            if existing[0] != evidence:
                raise MemoryHypothesisAdapterError(
                    "同一 owner/evidence_id 已绑定不同 Evidence")
            return
        hypothesis_ref = self._require_hypothesis(evidence.hypothesis)
        supersedes_ref = None
        if evidence.supersedes_evidence_id:
            prior = self._evidence_by_ledger_id.get(self._ledger_event_key(
                evidence.hypothesis, evidence.supersedes_evidence_id))
            if prior is None or prior[0].hypothesis != evidence.hypothesis:
                raise MemoryHypothesisAdapterError(
                    "Evidence supersede 目标没有可恢复的同候选事件")
            supersedes_ref = prior[1]
        observed_at = LogicalTimestamp(
            LogicalClockIdentity(
                evidence.hypothesis.scope, CLOCK_MEMORY_OBSERVED),
            evidence.timestamp_seq + 1,
        )
        payload = EvidencePayload(
            hypothesis_ref,
            evidence.stance,
            None,
            evidence.reason_key,
            evidence.source,
            None,
            evidence.payload,
            supersedes_ref,
            observed_at,
            evidence.stable_key(),
        )
        ref = memory_object_ref(
            self.event_log.memory_space_identity,
            MEMORY_OBJECT_EVIDENCE,
            payload.stable_key(),
            owner=evidence.hypothesis.observation.owner,
            versions=evidence.hypothesis.observation.versions,
        )
        self.event_log.append(MemoryEvent(
            MEMORY_EVENT_EVIDENCE,
            ref,
            evidence.hypothesis.scope,
            payload,
        ))
        self._evidence_by_ledger_id[cache_key] = evidence, ref

    def append_transition(self, transition: HypothesisTransition) -> None:
        """把 H-00 生命周期转换写成引用理由 Evidence 的通用 lifecycle 事件。"""
        if not isinstance(transition, HypothesisTransition):
            raise TypeError("append_transition 需要 HypothesisTransition")
        boundary = self._ledger_event_key(transition.hypothesis, 1)[0]
        if boundary not in self._hydrated_compatibility_boundaries:
            self._hydrate_compatibility(transition.hypothesis)
        cache_key = self._ledger_event_key(
            transition.hypothesis, transition.event_id)
        existing = self._transition_by_ledger_id.get(cache_key)
        if existing is not None:
            if existing != transition:
                raise MemoryHypothesisAdapterError(
                    "同一 owner/transition event_id 已绑定不同转换")
            return
        target_ref = self._require_hypothesis(transition.hypothesis)
        reason = self._evidence_by_ledger_id.get(self._ledger_event_key(
            transition.hypothesis, transition.reason_evidence_id))
        if reason is None or reason[0].hypothesis != transition.hypothesis:
            raise MemoryHypothesisAdapterError(
                "生命周期转换理由 Evidence 不存在或属于其他候选")
        replacement_ref = None
        if transition.replacement is not None:
            replacement_ref = self._require_hypothesis(transition.replacement)
        changed_at = LogicalTimestamp(
            LogicalClockIdentity(
                transition.hypothesis.scope, CLOCK_MEMORY_LIFECYCLE),
            transition.timestamp_seq + 1,
        )
        payload = LifecycleTransitionPayload(
            target_ref,
            transition.from_state,
            transition.to_state,
            (reason[1],),
            replacement_ref,
            changed_at,
            transition.stable_key(),
        )
        self.event_log.append(MemoryEvent(
            MEMORY_EVENT_LIFECYCLE,
            target_ref,
            transition.hypothesis.scope,
            payload,
        ))
        self._transition_by_ledger_id[cache_key] = transition

    def append_decision(self, decision: ResolverDecision) -> None:
        """把 H-04 完整决策写成独立 Resolution 声明，不依赖图投影。"""
        if not isinstance(decision, ResolverDecision):
            raise TypeError("append_decision 需要 ResolverDecision")
        hypotheses = tuple(
            item.hypothesis for item in decision.candidates)
        if not hypotheses:
            raise MemoryHypothesisAdapterError("H-04 decision 不得没有候选")
        anchor = hypotheses[0]
        boundary = self._ledger_event_key(anchor, 1)[0]
        if boundary not in self._hydrated_decision_boundaries:
            self._hydrate_decisions(anchor)
        cache_key = self._ledger_event_key(anchor, decision.decision_id)
        existing = self._decision_by_ledger_id.get(cache_key)
        if existing is not None:
            if existing != decision:
                raise MemoryHypothesisAdapterError(
                    "同一 owner/decision_id 已绑定不同 H-04 决策")
            return
        refs = tuple(self._require_hypothesis(item) for item in hypotheses)
        resolved_at = LogicalTimestamp(
            LogicalClockIdentity(anchor.scope, CLOCK_MEMORY_RESOLVED),
            decision.timestamp_seq + 1,
        )
        payload = ResolutionPayload(
            decision.stable_key(), refs, resolved_at)
        resolution_ref = memory_object_ref(
            self.event_log.memory_space_identity,
            MEMORY_OBJECT_RESOLUTION,
            payload.identity_key(),
            owner=anchor.observation.owner,
            versions=anchor.observation.versions,
        )
        self.event_log.append(MemoryEvent(
            MEMORY_EVENT_RESOLUTION,
            resolution_ref,
            anchor.scope,
            payload,
        ))
        self._decision_by_ledger_id[cache_key] = decision

    def load_decisions(
            self, *, access: MemoryAccessContext,
            hypotheses: tuple[HypothesisKey, ...],
            ) -> tuple[ResolverDecision, ...]:
        """按精确候选集合恢复完整 H-04 决策链，拒绝跨协议混入。"""
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("load_decisions 必须提供 MemoryAccessContext")
        if (not isinstance(hypotheses, tuple)
                or any(not isinstance(item, HypothesisKey)
                       for item in hypotheses)
                or len(set(hypotheses)) != len(hypotheses)):
            raise TypeError("hypotheses 必须是无重复 HypothesisKey tuple")
        selected = frozenset(hypotheses)
        decisions: list[ResolverDecision] = []
        for entry in self.event_log.query(
                access=access, event_kind=MEMORY_EVENT_RESOLUTION):
            payload = entry.event.payload
            if not isinstance(payload, ResolutionPayload):
                raise MemoryHypothesisAdapterError(
                    "Resolution event 携带错误 payload")
            decision = ResolverDecision.from_stable_key(
                payload.decision_key)
            candidates = frozenset(
                item.hypothesis for item in decision.candidates)
            if not candidates & selected:
                continue
            if not candidates <= selected:
                raise MemoryHypothesisAdapterError(
                    "H-04 decision 横跨请求恢复的候选协议边界")
            decisions.append(decision)
        by_id = {item.decision_id: item for item in decisions}
        if len(by_id) != len(decisions):
            raise MemoryHypothesisAdapterError(
                "可见 Resolution 含重复 decision id")
        return tuple(sorted(
            decisions,
            key=lambda item: (
                item.competition_key,
                item.timestamp_seq,
                item.decision_id,
            ),
        ))

    def load_indexed_decisions(
            self,
            *,
            hypotheses: tuple[HypothesisKey, ...],
            events: tuple[MaterializedMemoryEvent, ...],
            ) -> tuple[ResolverDecision, ...]:
        """只从竞争组反向索引事件恢复 H-04 决策链。"""
        selected = self._validate_indexed_inputs(hypotheses, events)
        decisions: list[ResolverDecision] = []
        for entry in events:
            payload = entry.event.payload
            if not isinstance(payload, ResolutionPayload):
                continue
            decision = ResolverDecision.from_stable_key(payload.decision_key)
            candidates = frozenset(
                item.hypothesis for item in decision.candidates)
            if not candidates <= selected:
                raise MemoryHypothesisAdapterError(
                    "索引 Resolution 横跨请求竞争边界")
            anchor = decision.candidates[0].hypothesis
            cache_key = self._ledger_event_key(
                anchor, decision.decision_id)
            existing = self._decision_by_ledger_id.get(cache_key)
            if existing is not None and existing != decision:
                raise MemoryHypothesisAdapterError(
                    "索引同一 decision id 对应不同内容")
            self._decision_by_ledger_id[cache_key] = decision
            decisions.append(decision)
        by_id = {item.decision_id: item for item in decisions}
        if len(by_id) != len(decisions):
            raise MemoryHypothesisAdapterError(
                "索引 Resolution 含重复 decision id")
        for hypothesis in hypotheses:
            self._hydrated_decision_boundaries.add(
                self._ledger_event_key(hypothesis, 1)[0])
        return tuple(sorted(
            decisions,
            key=lambda item: (
                item.competition_key,
                item.timestamp_seq,
                item.decision_id,
            ),
        ))

    def hypotheses(
            self, *, access: MemoryAccessContext,
            ) -> tuple[HypothesisKey, ...]:
        """返回 ACL 可见的唯一 Hypothesis 声明，供领域协议先行过滤。"""
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("hypotheses 必须提供 MemoryAccessContext")
        hypothesis_entries = self.event_log.query(
            access=access, event_kind=MEMORY_EVENT_HYPOTHESIS)
        hypotheses = tuple(sorted(
            (entry.event.payload.hypothesis for entry in hypothesis_entries
             if isinstance(entry.event.payload, HypothesisPayload)),
            key=lambda item: item.stable_key(),
        ))
        if len(set(hypotheses)) != len(hypotheses):
            raise MemoryHypothesisAdapterError(
                "可见事件包含重复 Hypothesis 声明")
        return hypotheses

    def load_ledger(
            self, *, access: MemoryAccessContext,
            hypotheses: tuple[HypothesisKey, ...] | None = None,
            attach_sink: bool = False,
            ) -> HypothesisLedger:
        """按精确候选集合重建 H-00 ledger，可选择绑定当前 sink 供续写。"""
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("load_ledger 必须提供 MemoryAccessContext")
        if hypotheses is not None and (
                not isinstance(hypotheses, tuple)
                or any(not isinstance(item, HypothesisKey)
                       for item in hypotheses)):
            raise TypeError("hypotheses 必须是 HypothesisKey tuple 或 None")
        if type(attach_sink) is not bool:
            raise TypeError("attach_sink 必须是 bool")
        visible_hypotheses = self.hypotheses(access=access)
        if hypotheses is None:
            selected = visible_hypotheses
        else:
            if len(set(hypotheses)) != len(hypotheses):
                raise ValueError("load_ledger 不接受重复 Hypothesis")
            missing = set(hypotheses) - set(visible_hypotheses)
            if missing:
                raise MemoryHypothesisAdapterError(
                    "请求恢复的 Hypothesis 没有可见唯一声明")
            selected = tuple(sorted(
                hypotheses, key=lambda item: item.stable_key()))
        selected_set = frozenset(selected)
        evidence_entries = self.event_log.query(
            access=access, event_kind=MEMORY_EVENT_EVIDENCE)
        transition_entries = self.event_log.query(
            access=access, event_kind=MEMORY_EVENT_LIFECYCLE)

        ledger = HypothesisLedger()
        for hypothesis in selected:
            ledger.register(hypothesis)

        evidence: list[EvidenceRecord] = []
        for entry in evidence_entries:
            payload = entry.event.payload
            if (not isinstance(payload, EvidencePayload)
                    or not payload.compatibility_record_key):
                continue
            record = EvidenceRecord.from_stable_key(
                payload.compatibility_record_key)
            if record.hypothesis not in selected_set:
                continue
            self._validate_evidence_projection(payload, record)
            evidence.append(record)
        for record in self._evidence_topological(evidence):
            ledger.append_evidence(record)

        transitions: list[HypothesisTransition] = []
        for entry in transition_entries:
            payload = entry.event.payload
            if (not isinstance(payload, LifecycleTransitionPayload)
                    or not payload.compatibility_transition_key):
                continue
            transition = HypothesisTransition.from_stable_key(
                payload.compatibility_transition_key)
            if transition.hypothesis not in selected_set:
                continue
            self._validate_transition_projection(payload, transition)
            transitions.append(transition)
        for transition in sorted(
                transitions,
                key=lambda item: (item.timestamp_seq, item.event_id)):
            ledger.append_transition(transition)
        return ledger.with_sink(self) if attach_sink else ledger

    def load_indexed_ledger(
            self,
            *,
            hypotheses: tuple[HypothesisKey, ...],
            events: tuple[MaterializedMemoryEvent, ...],
            attach_sink: bool = False,
            ) -> HypothesisLedger:
        """只从调用方给出的竞争组索引事件重建 H-00 ledger。"""
        selected_set = self._validate_indexed_inputs(hypotheses, events)
        if type(attach_sink) is not bool:
            raise TypeError("attach_sink 必须是 bool")
        selected = tuple(sorted(
            hypotheses, key=lambda item: item.stable_key()))
        ledger = HypothesisLedger()
        for hypothesis in selected:
            ledger.register(hypothesis)

        evidence: list[EvidenceRecord] = []
        transitions: list[HypothesisTransition] = []
        for entry in events:
            payload = entry.event.payload
            if (isinstance(payload, EvidencePayload)
                    and payload.compatibility_record_key):
                record = EvidenceRecord.from_stable_key(
                    payload.compatibility_record_key)
                if record.hypothesis not in selected_set:
                    raise MemoryHypothesisAdapterError(
                        "索引 Evidence 横跨请求竞争边界")
                self._validate_evidence_projection(payload, record)
                cache_key = self._ledger_event_key(
                    record.hypothesis, record.evidence_id)
                existing = self._evidence_by_ledger_id.get(cache_key)
                if existing is not None and existing[0] != record:
                    raise MemoryHypothesisAdapterError(
                        "索引同一 evidence id 对应不同内容")
                self._evidence_by_ledger_id[cache_key] = (
                    record, entry.event.object_ref)
                evidence.append(record)
            elif (isinstance(payload, LifecycleTransitionPayload)
                  and payload.compatibility_transition_key):
                transition = HypothesisTransition.from_stable_key(
                    payload.compatibility_transition_key)
                if transition.hypothesis not in selected_set:
                    raise MemoryHypothesisAdapterError(
                        "索引 lifecycle 横跨请求竞争边界")
                self._validate_transition_projection(payload, transition)
                cache_key = self._ledger_event_key(
                    transition.hypothesis, transition.event_id)
                existing = self._transition_by_ledger_id.get(cache_key)
                if existing is not None and existing != transition:
                    raise MemoryHypothesisAdapterError(
                        "索引同一 transition id 对应不同内容")
                self._transition_by_ledger_id[cache_key] = transition
                transitions.append(transition)
        for record in self._evidence_topological(evidence):
            ledger.append_evidence(record)
        for transition in sorted(
                transitions,
                key=lambda item: (item.timestamp_seq, item.event_id)):
            ledger.append_transition(transition)
        for hypothesis in hypotheses:
            self._hydrated_compatibility_boundaries.add(
                self._ledger_event_key(hypothesis, 1)[0])
        return ledger.with_sink(self) if attach_sink else ledger

    @staticmethod
    def _validate_indexed_inputs(
            hypotheses: tuple[HypothesisKey, ...],
            events: tuple[MaterializedMemoryEvent, ...],
            ) -> frozenset[HypothesisKey]:
        """核验竞争组候选、索引事件类型和 event hash 唯一性。"""
        if (not isinstance(hypotheses, tuple)
                or not hypotheses
                or any(not isinstance(item, HypothesisKey)
                       for item in hypotheses)
                or len(set(hypotheses)) != len(hypotheses)):
            raise TypeError("hypotheses 必须是非空无重复 HypothesisKey tuple")
        if (not isinstance(events, tuple)
                or any(not isinstance(item, MaterializedMemoryEvent)
                       for item in events)):
            raise TypeError("events 必须是 MaterializedMemoryEvent tuple")
        hashes = tuple(item.event_hash for item in events)
        if len(set(hashes)) != len(hashes):
            raise ValueError("竞争组索引事件不得重复 event hash")
        return frozenset(hypotheses)

    def _hydrate_compatibility(self, hypothesis: HypothesisKey) -> None:
        """按 owner 可见范围加载已有 H-00 事件，供新 adapter 幂等续写。"""
        access = self._access(hypothesis)
        for entry in self.event_log.query(
                access=access, event_kind=MEMORY_EVENT_EVIDENCE):
            payload = entry.event.payload
            if (not isinstance(payload, EvidencePayload)
                    or not payload.compatibility_record_key):
                continue
            record = EvidenceRecord.from_stable_key(
                payload.compatibility_record_key)
            self._validate_evidence_projection(payload, record)
            key = self._ledger_event_key(
                record.hypothesis, record.evidence_id)
            existing = self._evidence_by_ledger_id.get(key)
            if existing is not None and existing[0] != record:
                raise MemoryHypothesisAdapterError(
                    "持久层同一 owner/evidence_id 对应不同记录")
            self._evidence_by_ledger_id[key] = record, entry.event.object_ref
        for entry in self.event_log.query(
                access=access, event_kind=MEMORY_EVENT_LIFECYCLE):
            payload = entry.event.payload
            if (not isinstance(payload, LifecycleTransitionPayload)
                    or not payload.compatibility_transition_key):
                continue
            transition = HypothesisTransition.from_stable_key(
                payload.compatibility_transition_key)
            self._validate_transition_projection(payload, transition)
            key = self._ledger_event_key(
                transition.hypothesis, transition.event_id)
            existing = self._transition_by_ledger_id.get(key)
            if existing is not None and existing != transition:
                raise MemoryHypothesisAdapterError(
                    "持久层同一 owner/transition id 对应不同记录")
            self._transition_by_ledger_id[key] = transition
        self._hydrated_compatibility_boundaries.add(
            self._ledger_event_key(hypothesis, 1)[0])

    def _hydrate_decisions(self, hypothesis: HypothesisKey) -> None:
        """加载同 owner 可见 Resolution，供 decision id 碰撞和幂等核验。"""
        for entry in self.event_log.query(
                access=self._access(hypothesis),
                event_kind=MEMORY_EVENT_RESOLUTION):
            payload = entry.event.payload
            if not isinstance(payload, ResolutionPayload):
                raise MemoryHypothesisAdapterError(
                    "Resolution event 携带错误 payload")
            decision = ResolverDecision.from_stable_key(
                payload.decision_key)
            anchor = decision.candidates[0].hypothesis
            key = self._ledger_event_key(anchor, decision.decision_id)
            existing = self._decision_by_ledger_id.get(key)
            if existing is not None and existing != decision:
                raise MemoryHypothesisAdapterError(
                    "持久层同一 owner/decision id 对应不同记录")
            self._decision_by_ledger_id[key] = decision
        self._hydrated_decision_boundaries.add(
            self._ledger_event_key(hypothesis, 1)[0])

    def _require_hypothesis(self, hypothesis: HypothesisKey) -> MemoryObjectRef:
        """返回已唯一声明的 Hypothesis 引用，缺失时拒绝隐式创建。"""
        ref = self._hypothesis_ref(hypothesis)
        entries = self.event_log.query(
            access=self._access(hypothesis),
            event_kind=MEMORY_EVENT_HYPOTHESIS,
            object_ref=ref,
        )
        if len(entries) != 1:
            raise MemoryHypothesisAdapterError(
                "H-00 事件写入前 Hypothesis 必须已有唯一声明")
        payload = entries[0].event.payload
        if (not isinstance(payload, HypothesisPayload)
                or payload.hypothesis != hypothesis):
            raise MemoryHypothesisAdapterError("Hypothesis 声明内容漂移")
        return ref

    def _hypothesis_ref(self, hypothesis: HypothesisKey) -> MemoryObjectRef:
        """把 H-00 完整候选键映射为当前 Memory 空间对象引用。"""
        return memory_object_ref(
            self.event_log.memory_space_identity,
            MEMORY_OBJECT_HYPOTHESIS,
            hypothesis.stable_key(),
            owner=hypothesis.observation.owner,
            versions=hypothesis.observation.versions,
        )

    @staticmethod
    def _access(hypothesis: HypothesisKey) -> MemoryAccessContext:
        """从候选 owner 生成等权读取上下文，不提升 visibility。"""
        owner = hypothesis.observation.owner
        return MemoryAccessContext(
            owner.tenant_id, owner.user_id, owner.session_id)

    @staticmethod
    def _ledger_event_key(hypothesis: HypothesisKey,
                          event_id: int) -> tuple[tuple[int, ...], int]:
        """用完整 aggregate 协议边界隔离不同 ledger 的局部事件 id。"""
        kind = hypothesis.hypothesis_kind
        scope = hypothesis.scope.stable_key()
        observation = hypothesis.observation.stable_key()
        return (
            (
                len(kind),
                *kind,
                len(scope),
                *scope,
                len(observation),
                *observation,
            ),
            event_id,
        )

    def _validate_evidence_projection(
            self, payload: EvidencePayload,
            record: EvidenceRecord) -> None:
        """逐字段核验 EvidencePayload 没有重解释 H-00 原记录。"""
        expected_hypothesis = self._hypothesis_ref(record.hypothesis)
        if (
                payload.hypothesis_ref != expected_hypothesis
                or payload.stance != record.stance
                or payload.signal_ref is not None
                or payload.legacy_reason_key != record.reason_key
                or payload.source != record.source
                or payload.episode_ref is not None
                or payload.detail != record.payload
                or payload.observed_at.seq != record.timestamp_seq + 1
                or payload.observed_at.clock.scope != record.hypothesis.scope
                or payload.observed_at.clock.clock_kind
                != CLOCK_MEMORY_OBSERVED):
            raise MemoryHypothesisAdapterError(
                "Memory Evidence 与 H-00 compatibility record 不一致")
        if record.supersedes_evidence_id == 0:
            if payload.supersedes_ref is not None:
                raise MemoryHypothesisAdapterError(
                    "H-00 Evidence 无 supersede 却带 Memory 目标")
        elif payload.supersedes_ref is None:
            raise MemoryHypothesisAdapterError(
                "H-00 Evidence supersede 目标在 Memory 中缺失")
        else:
            prior = self._compatibility_evidence_from_ref(
                payload.supersedes_ref, record.hypothesis)
            if (prior.evidence_id != record.supersedes_evidence_id
                    or prior.hypothesis != record.hypothesis):
                raise MemoryHypothesisAdapterError(
                    "Memory Evidence supersede 引用与 H-00 id 不一致")

    def _validate_transition_projection(
            self, payload: LifecycleTransitionPayload,
            transition: HypothesisTransition) -> None:
        """逐字段核验 lifecycle payload 没有改写 H-00 状态转换。"""
        replacement = (
            None if transition.replacement is None
            else self._hypothesis_ref(transition.replacement)
        )
        if (
                payload.target_ref != self._hypothesis_ref(
                    transition.hypothesis)
                or payload.from_state != transition.from_state
                or payload.to_state != transition.to_state
                or payload.replacement_ref != replacement
                or len(payload.reason_evidence_refs) != 1
                or payload.changed_at.seq != transition.timestamp_seq + 1
                or payload.changed_at.clock.scope != transition.hypothesis.scope
                or payload.changed_at.clock.clock_kind
                != CLOCK_MEMORY_LIFECYCLE):
            raise MemoryHypothesisAdapterError(
                "Memory lifecycle 与 H-00 compatibility transition 不一致")
        reason = self._compatibility_evidence_from_ref(
            payload.reason_evidence_refs[0], transition.hypothesis)
        if (reason.evidence_id != transition.reason_evidence_id
                or reason.hypothesis != transition.hypothesis):
            raise MemoryHypothesisAdapterError(
                "Memory lifecycle 理由引用与 H-00 evidence id 不一致")

    def _compatibility_evidence_from_ref(
            self, ref: MemoryObjectRef,
            hypothesis: HypothesisKey) -> EvidenceRecord:
        """按完整 Memory 引用恢复一条 H-00 兼容 Evidence，并核验唯一性。"""
        entries = self.event_log.query(
            access=self._access(hypothesis),
            event_kind=MEMORY_EVENT_EVIDENCE,
            object_ref=ref,
        )
        if len(entries) != 1:
            raise MemoryHypothesisAdapterError(
                "H-00 compatibility Evidence 引用没有唯一事件")
        payload = entries[0].event.payload
        if (not isinstance(payload, EvidencePayload)
                or not payload.compatibility_record_key):
            raise MemoryHypothesisAdapterError(
                "理由 Evidence 不是 H-00 compatibility 事件")
        return EvidenceRecord.from_stable_key(
            payload.compatibility_record_key)

    @staticmethod
    def _evidence_topological(
            records: list[EvidenceRecord]) -> tuple[EvidenceRecord, ...]:
        """按 supersedes 依赖拓扑排序 Evidence，同层再按逻辑序和 id 稳定排序。"""
        pending = {
            record.evidence_id: record
            for record in records
        }
        if len(pending) != len(records):
            raise MemoryHypothesisAdapterError(
                "可见 H-00 Evidence 含重复 evidence_id")
        emitted: set[int] = set()
        ordered: list[EvidenceRecord] = []
        while pending:
            ready = sorted(
                (record for record in pending.values()
                 if record.supersedes_evidence_id == 0
                 or record.supersedes_evidence_id in emitted),
                key=lambda item: (item.timestamp_seq, item.evidence_id),
            )
            if not ready:
                raise MemoryHypothesisAdapterError(
                    "H-00 Evidence supersede 链存在孤儿或环")
            for record in ready:
                ordered.append(record)
                emitted.add(record.evidence_id)
                del pending[record.evidence_id]
        return tuple(ordered)


__all__ = [
    "MemoryHypothesisAdapterError",
    "MemoryHypothesisEventSink",
]
