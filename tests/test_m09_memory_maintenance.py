"""M-09 记忆维护、衰减、生命周期和放置提示对抗测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ReplacementDirective,
)
from pure_integer_ai.cognition.shared.identity import (
    SourceRef,
)
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MEMORY_EVIDENCE_PROVISIONAL,
)
from pure_integer_ai.cognition.shared.memory_decay import (
    LinearTimelineDecayPolicy,
    MemoryActivationSnapshot,
    MemoryDecayScoreProvider,
    RetentionDecayCurve,
)
from pure_integer_ai.cognition.shared.memory_event import (
    EpisodePayload,
    MEMORY_EVENT_EPISODE,
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_LIFECYCLE,
    MEMORY_EVENT_RESOLUTION,
    MEMORY_EVENT_RETENTION,
    MEMORY_EVENT_USE,
    MEMORY_OBJECT_EPISODE,
    MEMORY_OBJECT_USE,
    MemoryEvent,
    MemoryLinkedRef,
    RETENTION_CONSOLIDATED,
    RETENTION_EPISODIC,
    UsePayload,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_hypothesis import (
    MemoryHypothesisEventSink,
)
from pure_integer_ai.cognition.shared.memory_maintenance import (
    MemoryMaintenanceService,
    MemoryPlacementHint,
    MemoryRetentionDecision,
)
from pure_integer_ai.cognition.shared.memory_overlay import (
    MemoryAccessContext,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_CREATED,
    CLOCK_MEMORY_USED,
    LogicalTimestamp,
)
from pure_integer_ai.experiments.memory_maintenance_runtime import (
    install_memory_maintenance_runtime,
)
from pure_integer_ai.experiments.memory_use_runtime import (
    install_memory_use_runtime,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    isolated_evaluation,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.placement import (
    TemperatureProfile,
    TemperatureTier,
)
from pure_integer_ai.storage.storage_role import (
    STORAGE_ACCESS_INDEXED_READ,
    STORAGE_ROLE_AUTHORITATIVE,
    StorageRoleDescriptor,
    StorageRoleRegistry,
)

from test_a10_attractor_state import (
    _close_query,
    _setup,
)
from test_m03_memory_event import (
    _append_observation,
    _core_refs,
    _scopes,
    _source,
    _timestamp,
)
from test_m08_memory_use import (
    _consume_head as _consume_memory_head,
    _query_time,
)


_ACCESS = MemoryAccessContext(1, 2, 3)
_DESCRIPTOR_KEY = (99010, 1)
_HOT_TIER = (99011, 1)
_COLD_TIER = (99011, 2)


class _RetentionPolicy:
    """测试用巩固门，Evidence 理由由测试显式注入。"""

    def __init__(self, consolidate: bool, reason_refs=()):
        """保存是否巩固以及候选理由，避免共享层猜测阈值。"""
        self.consolidate = consolidate
        self.reason_refs = tuple(reason_refs)

    def state_key(self):
        """返回策略模式和参数的稳定身份。"""
        return (
            99020,
            int(self.consolidate),
            len(self.reason_refs),
            *(value for item in self.reason_refs for value in item.stable_key()),
        )

    def assess(self, snapshot):
        """按测试注入结果返回 retention 决定。"""
        refs = self.reason_refs if self.consolidate else ()
        return MemoryRetentionDecision(
            self.consolidate,
            refs,
            (99021,),
            self.state_key(),
        )


class _PlacementPolicy:
    """测试用 K-01 hint 策略，只返回候选而不触碰存储。"""

    def state_key(self):
        """返回放置策略的稳定身份。"""
        return (99030, 1)

    def hints(self, snapshot):
        """为当前 Hypothesis 返回一个冷温层候选。"""
        return (MemoryPlacementHint(
            snapshot.hypothesis_ref.stable_key(),
            _DESCRIPTOR_KEY,
            _COLD_TIER,
            (99012, 1),
            (99031,),
            self.state_key(),
            snapshot.as_of.seq,
        ),)


class _BaseScoreProvider:
    """测试用稳定基础评分器，时间分量由 M-09 另行注入。"""

    def state_key(self):
        """返回基础评分器版本。"""
        return (99040, 1)

    def score(self, request, hypothesis, aggregate, sources):
        """忽略领域输入，只提供可审计的零基础分。"""
        del request, hypothesis, aggregate, sources
        from pure_integer_ai.cognition.shared.memory_resolver import (
            ActivationScore,
            ActivationScoreReason,
        )
        return ActivationScore(0, (ActivationScoreReason((99041,), 0),))


def _hypothesis(source: SourceRef, candidate: int) -> HypothesisKey:
    """构造同一来源、scope 和竞争组中的候选。"""
    return HypothesisKey(
        (99050,),
        (candidate,),
        (99051,),
        _document_scope(source),
        source,
    )


def _document_scope(source: SourceRef):
    """构造来源化 document scope，保持测试对象边界完整。"""
    from pure_integer_ai.cognition.shared.scope_identity import document_scope
    return document_scope(source)


def _hypothesis_ref(ctx, hypothesis: HypothesisKey):
    """从唯一 Hypothesis 声明恢复完整 Memory 对象引用。"""
    entries = ctx.memory_interact_events.query(
        access=_ACCESS,
        event_kind=MEMORY_EVENT_HYPOTHESIS,
    )
    matches = tuple(
        item for item in entries
        if item.event.payload.hypothesis == hypothesis
    )
    assert len(matches) == 1
    return matches[0].event.object_ref


def _evidence_ref(ctx, evidence_id: int):
    """按兼容 Evidence 记录的稳定 id 恢复 Memory Evidence 引用。"""
    entries = ctx.memory_interact_events.query(
        access=_ACCESS,
        event_kind=MEMORY_EVENT_EVIDENCE,
    )
    matches = tuple(
        item for item in entries
        if item.event.payload.compatibility_record_key[0] == evidence_id
    )
    assert len(matches) == 1
    return matches[0].event.object_ref


def _register_evidence(ctx, source, hypotheses, records):
    """通过 H-00 sink 写入候选和 Evidence，保持真实事件生产路径。"""
    ledger = HypothesisLedger(
        MemoryHypothesisEventSink(ctx.memory_interact_events))
    for hypothesis in hypotheses:
        ledger.register(hypothesis)
    for record in records:
        ledger.append_evidence(record)
    ctx.memory_interact_aggregates.rebuild_dirty(access=_ACCESS)


def _append_linked_episode(ctx, source, observation, memory_ref):
    """追加明确引用旧 Memory 版本的 Episode，供生命周期反例使用。"""
    session, _, episode_scope = _scopes(source)
    linked = MemoryLinkedRef.memory(memory_ref)
    payload = EpisodePayload(
        observation.event.object_ref,
        None,
        (linked,),
        linked,
        linked,
        (memory_ref,),
        (),
        None,
        1,
        session,
        _timestamp(episode_scope, CLOCK_MEMORY_CREATED, 2),
    )
    ref = memory_object_ref(
        ctx.memory_interact_events.memory_space_identity,
        MEMORY_OBJECT_EPISODE,
        payload.stable_key(),
        owner=source.owner,
        versions=source.versions,
    )
    return ctx.memory_interact_events.append(MemoryEvent(
        MEMORY_EVENT_EPISODE,
        ref,
        episode_scope,
        payload,
    ))


def _append_compatibility_use(ctx, source, episode, memory_ref):
    """追加一个旧协议 Use，验证 supersede 不会改写消费者引用。"""
    _, _, episode_scope = _scopes(source)
    payload = UsePayload(
        memory_ref,
        episode.event.object_ref,
        MemoryLinkedRef.core(_core_refs(ctx)[0]),
        None,
        _timestamp(episode_scope, CLOCK_MEMORY_USED),
    )
    ref = memory_object_ref(
        ctx.memory_interact_events.memory_space_identity,
        MEMORY_OBJECT_USE,
        payload.stable_key(),
        owner=source.owner,
        versions=source.versions,
    )
    return ctx.memory_interact_events.append(MemoryEvent(
        MEMORY_EVENT_USE,
        ref,
        episode_scope,
        payload,
    ))


def _service(ctx, *, retention_policy):
    """装配一套不含领域阈值的 M-09 注入策略。"""
    roles = StorageRoleRegistry()
    roles.register(StorageRoleDescriptor(
        _DESCRIPTOR_KEY,
        STORAGE_ROLE_AUTHORITATIVE,
        (STORAGE_ACCESS_INDEXED_READ,),
    ))
    profile = TemperatureProfile(
        (99012, 1),
        (TemperatureTier(_HOT_TIER, 0), TemperatureTier(_COLD_TIER, 1)),
    )
    activation = LinearTimelineDecayPolicy(
        (99013, 1),
        (
            RetentionDecayCurve(RETENTION_EPISODIC, 100, 1, 0),
            RetentionDecayCurve(RETENTION_CONSOLIDATED, 80, 2, 0),
        ),
        (99014,),
    )
    return MemoryMaintenanceService(
        ctx.memory_interact_aggregates,
        activation,
        retention_policy,
        _PlacementPolicy(),
        roles,
        profile,
    )


def test_timeline_seq_is_comparable_across_domain_scopes_and_tracks_use():
    """不同领域时钟的裸 seq 不可比较时，aggregate 仍使用统一物理 timeline。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source(source_id=99060)
        refs = _core_refs(ctx)
        hypothesis = _hypothesis(source, 1)
        _register_evidence(ctx, source, (hypothesis,), (
            EvidenceRecord(1, hypothesis, EVIDENCE_SUPPORT, (99063,), source, 1),
        ))
        hypothesis_ref = _hypothesis_ref(ctx, hypothesis)
        observation = _append_observation(ctx, source, refs)
        episode = _append_linked_episode(
            ctx,
            source,
            observation,
            hypothesis_ref,
        )
        use = _append_compatibility_use(
            ctx, source, episode, hypothesis_ref)
        ctx.memory_interact_aggregates.rebuild_dirty(access=_ACCESS)
        aggregate = ctx.memory_interact_aggregates.read(
            hypothesis_ref, access=_ACCESS)
        assert aggregate is not None
        use_entry = ctx.memory_interact_events.query(
            access=_ACCESS,
            event_kind=MEMORY_EVENT_USE,
            object_ref=use.event.object_ref,
        )[0]
        assert aggregate.last_used_seq == use_entry.timeline.seq
        assert aggregate.last_used_seq > 1
        assert aggregate.last_used_seq != use.event.payload.used_at.seq
    finally:
        backend.close()


def test_assess_and_placement_hint_have_no_write_or_clock_side_effect():
    """维护评估和 hint 只能读当前投影，不能写事件、Evidence 或物理迁移。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source(source_id=99070)
        hypothesis = _hypothesis(source, 1)
        _register_evidence(ctx, source, (hypothesis,), (
            EvidenceRecord(1, hypothesis, EVIDENCE_SUPPORT, (99071,), source, 1),
        ))
        ref = _hypothesis_ref(ctx, hypothesis)
        policy = _RetentionPolicy(False)
        service = _service(ctx, retention_policy=policy)
        before = backend.snapshot()
        watermark = ctx.memory_interact_events.timeline_watermark()
        events = ctx.memory_interact_events.query(access=_ACCESS)
        assessment = service.assess(ref, access=_ACCESS)
        commit = service.consolidate(ref, access=_ACCESS)
        assert assessment.placement_hints
        assert commit.retention_event is None
        assert backend.snapshot() == before
        assert ctx.memory_interact_events.timeline_watermark() == watermark
        assert ctx.memory_interact_events.query(access=_ACCESS) == events
    finally:
        backend.close()


def test_m07_decay_score_ages_both_retention_states_without_changing_facts():
    """统一 timeline 只降低 M-07 激活分，episodic 和 consolidated 均不免衰减。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source(source_id=99075)
        hypothesis = _hypothesis(source, 1)
        _register_evidence(ctx, source, (hypothesis,), (
            EvidenceRecord(1, hypothesis, EVIDENCE_SUPPORT, (99076,), source, 1),
        ))
        ref = _hypothesis_ref(ctx, hypothesis)
        evidence_ref = _evidence_ref(ctx, 1)
        service = _service(
            ctx,
            retention_policy=_RetentionPolicy(True, (evidence_ref,)),
        )
        provider = MemoryDecayScoreProvider(
            ctx.memory_interact_aggregates,
            _BaseScoreProvider(),
            service.activation_policy,
        )
        episodic = ctx.memory_interact_aggregates.read(ref, access=_ACCESS)
        assert episodic is not None
        before_events = tuple(
            item for item in ctx.memory_interact_events.query(access=_ACCESS)
            if item.event.object_ref == ref
        )
        first = provider.score(object(), hypothesis, episodic, ())
        HypothesisLedger(
            MemoryHypothesisEventSink(
                ctx.memory_interact_events)).register(_hypothesis(source, 2))
        aged = provider.score(object(), hypothesis, episodic, ())
        assert aged.value < first.value
        assert ctx.memory_interact_aggregates.read(ref, access=_ACCESS) == episodic

        commit = service.consolidate(ref, access=_ACCESS)
        assert commit.after.retention_state == RETENTION_CONSOLIDATED
        consolidated = provider.score(object(), hypothesis, commit.after, ())
        HypothesisLedger(
            MemoryHypothesisEventSink(
                ctx.memory_interact_events)).register(_hypothesis(source, 3))
        consolidated_aged = provider.score(
            object(), hypothesis, commit.after, ())
        assert consolidated_aged.value < consolidated.value
        assert ctx.memory_interact_aggregates.read(
            ref, access=_ACCESS) == commit.after
        after_events = tuple(
            item for item in ctx.memory_interact_events.query(access=_ACCESS)
            if item.event.object_ref == ref
            and item.event.event_kind != MEMORY_EVENT_RETENTION
        )
        assert after_events == before_events
    finally:
        backend.close()


def test_retention_must_reference_current_active_evidence_and_consolidated_still_decays():
    """旧 Evidence 被替代后不能作为巩固理由，consolidated 也仍按曲线衰减。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source(source_id=99080)
        hypothesis = _hypothesis(source, 1)
        _register_evidence(ctx, source, (hypothesis,), (
            EvidenceRecord(1, hypothesis, EVIDENCE_SUPPORT, (99081,), source, 1),
            EvidenceRecord(
                2, hypothesis, EVIDENCE_SUPPORT, (99082,), source, 2,
                supersedes_evidence_id=1,
            ),
        ))
        ref = _hypothesis_ref(ctx, hypothesis)
        old_evidence = _evidence_ref(ctx, 1)
        new_evidence = _evidence_ref(ctx, 2)
        bad_service = _service(
            ctx,
            retention_policy=_RetentionPolicy(True, (old_evidence,)),
        )
        with pytest.raises(ValueError, match="当前活动 Evidence"):
            bad_service.assess(ref, access=_ACCESS)
        with pytest.raises(ValueError, match="当前活动 Evidence"):
            bad_service.consolidate(ref, access=_ACCESS)

        service = _service(
            ctx,
            retention_policy=_RetentionPolicy(True, (new_evidence,)),
        )
        commit = service.consolidate(ref, access=_ACCESS)
        assert commit.retention_event is not None
        assert commit.after.retention_state == RETENTION_CONSOLIDATED
        assert commit.after.evidence_state == MEMORY_EVIDENCE_PROVISIONAL

        stale_assessment = service.assess(ref, access=_ACCESS)
        assert stale_assessment.activation.value < 80
        current = ctx.memory_interact_events.timeline_watermark()
        future = LogicalTimestamp(current.clock, current.seq + 3)
        future_snapshot = service.activation_policy.assess(
            MemoryActivationSnapshot(
                hypothesis,
                commit.after,
                future,
            ))
        assert future_snapshot.value < stale_assessment.activation.value
    finally:
        backend.close()


def test_past_as_of_is_rejected_until_historical_projection_exists():
    """当前 M-04 投影不是历史版本，维护接口不得伪装支持过去水位。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source(source_id=99090)
        hypothesis = _hypothesis(source, 1)
        _register_evidence(ctx, source, (hypothesis,), (
            EvidenceRecord(1, hypothesis, EVIDENCE_SUPPORT, (99091,), source, 1),
        ))
        ref = _hypothesis_ref(ctx, hypothesis)
        service = _service(ctx, retention_policy=_RetentionPolicy(False))
        HypothesisLedger(
            MemoryHypothesisEventSink(
                ctx.memory_interact_events)).register(_hypothesis(source, 2))
        current = ctx.memory_interact_events.timeline_watermark()
        assert current is not None and current.seq > 1
        with pytest.raises(ValueError, match="当前 Memory timeline 水位"):
            service.assess(
                ref,
                access=_ACCESS,
                as_of=LogicalTimestamp(current.clock, current.seq - 1),
            )
    finally:
        backend.close()


def test_lifecycle_reuses_h04_and_preserves_consolidated_old_use_and_episode(
        monkeypatch):
    """consolidated 候选仍可由真实 H-04 显式 supersede，旧消费者引用保持不变。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source(source_id=99100)
        refs = _core_refs(ctx)
        observation = _append_observation(ctx, source, refs)
        old = _hypothesis(source, 1)
        replacement = _hypothesis(source, 2)
        _register_evidence(ctx, source, (old, replacement), (
            EvidenceRecord(1, old, EVIDENCE_REFUTE, (99101,), source, 1),
            EvidenceRecord(2, replacement, EVIDENCE_SUPPORT, (99102,), source, 2),
        ))
        old_ref = _hypothesis_ref(ctx, old)
        replacement_ref = _hypothesis_ref(ctx, replacement)
        episode = _append_linked_episode(ctx, source, observation, old_ref)
        use = _append_compatibility_use(ctx, source, episode, old_ref)
        ctx.memory_interact_aggregates.rebuild_dirty(access=_ACCESS)

        consolidate = _service(
            ctx,
            retention_policy=_RetentionPolicy(
                True, (_evidence_ref(ctx, 1),)),
        )
        commit = consolidate.consolidate(old_ref, access=_ACCESS)
        assert commit.after.retention_state == RETENTION_CONSOLIDATED
        before_episode = episode.event.payload
        before_use = use.event.payload
        original_query = ctx.memory_interact_events.query

        def guarded_query(**kwargs):
            """禁止 H-04 增量路径退回按事件种类扫描全空间历史。"""
            if (kwargs.get("event_kind") in {
                    MEMORY_EVENT_HYPOTHESIS,
                    MEMORY_EVENT_EVIDENCE,
                    MEMORY_EVENT_LIFECYCLE,
                    MEMORY_EVENT_RESOLUTION,
                }
                    and kwargs.get("object_ref") is None):
                raise AssertionError("lifecycle 增量路径扫描了全事件历史")
            return original_query(**kwargs)

        monkeypatch.setattr(
            ctx.memory_interact_events, "query", guarded_query)

        decision = consolidate.resolve_lifecycle(
            old,
            access=_ACCESS,
            timestamp_seq=4,
            replacements=(ReplacementDirective(old, replacement, 1),),
        )
        assert decision.candidate(old).after.lifecycle == LIFECYCLE_SUPERSEDED
        old_after = ctx.memory_interact_aggregates.read(
            old_ref, access=_ACCESS)
        replacement_after = ctx.memory_interact_aggregates.read(
            replacement_ref, access=_ACCESS)
        assert old_after is not None
        assert old_after.lifecycle_state == LIFECYCLE_SUPERSEDED
        assert old_after.retention_state == RETENTION_CONSOLIDATED
        assert replacement_after is not None
        assert ctx.memory_interact_events.query(
            access=_ACCESS,
            event_kind=MEMORY_EVENT_LIFECYCLE,
            object_ref=old_ref,
        )
        assert episode.event.payload == before_episode
        assert use.event.payload == before_use
        assert before_episode.selected_path_ref == MemoryLinkedRef.memory(old_ref)
        assert before_use.memory_ref == old_ref
    finally:
        backend.close()


def test_use_outcome_marks_only_target_hypothesis_dirty():
    """延迟结果只能使精确 Use 所属 Hypothesis dirty，不能扩散到同空间候选。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    try:
        observation = _append_observation(ctx, source, _core_refs(ctx))
        memory_use = install_memory_use_runtime(ctx)
        state = attractor.resolve_and_activate(compilation, goals)
        trace = _consume_memory_head(attractor, source)
        use = memory_use.record_selection_use(
            trace,
            input_observation_ref=observation.event.object_ref,
            influence_kind=MemoryLinkedRef.core(_core_refs(ctx)[0]),
            used_at=_query_time(state, 2),
        )
        target_ref = use.use.event.payload.memory_ref
        ctx.memory_interact_aggregates.rebuild_dirty(access=_ACCESS)
        target = ctx.memory_interact_aggregates.read(target_ref, access=_ACCESS)
        assert target is not None
        other_refs = tuple(
            item.event.object_ref
            for item in ctx.memory_interact_events.query(
                access=_ACCESS,
                event_kind=MEMORY_EVENT_HYPOTHESIS,
            )
            if item.event.object_ref != target_ref
        )
        assert other_refs
        assert ctx.memory_interact_aggregates.store.list_dirty() == ()
        memory_use.record_outcome(
            use.use.event.object_ref,
            scope=state.scope,
            outcome_kind=MemoryLinkedRef.core(_core_refs(ctx)[1]),
            outcome_ref=None,
            observed_at=_query_time(state, 3),
        )
        dirty = ctx.memory_interact_aggregates.store.list_dirty()
        assert tuple(item.hypothesis_hash for item in dirty) == (
            target.hypothesis_hash,)
    finally:
        _close_query(ctx)
        backend.close()


def test_v06_clone_keeps_m09_retention_event_inside_evaluation_sandbox():
    """评测 clone 的巩固写入只能落在 clone Memory，不能污染宿主。"""
    setup = _setup(prefer_matching_document=False)
    backend, ctx, source, attractor, _, compilation, goals = setup
    try:
        install_memory_use_runtime(ctx)
        state = attractor.resolve_and_activate(compilation, goals)
        del state, attractor
        hypothesis_entries = ctx.memory_interact_events.query(
            access=_ACCESS,
            event_kind=MEMORY_EVENT_HYPOTHESIS,
        )
        evidence_ref = _evidence_ref(ctx, 1)
        evidence_entry = ctx.memory_interact_events.query(
            access=_ACCESS,
            event_kind=MEMORY_EVENT_EVIDENCE,
            object_ref=evidence_ref,
        )[0]
        target_ref = evidence_entry.event.payload.hypothesis_ref
        assert target_ref in tuple(
            item.event.object_ref for item in hypothesis_entries)
        host = install_memory_maintenance_runtime(
            ctx,
            _service(
                ctx,
                retention_policy=_RetentionPolicy(True, (evidence_ref,)),
            ),
        )
        _close_query(ctx)
        host_before = backend.snapshot()
        with isolated_evaluation(ctx, label="m09-retention") as eval_ctx:
            clone = eval_ctx.memory_maintenance_runtime
            assert clone is not host
            assert clone.state_key() == host.state_key()
            result = clone.consolidate(target_ref, access=_ACCESS)
            assert result.retention_event is not None
            assert eval_ctx.backend.snapshot() != host_before
            assert ctx.memory_interact_events.query(
                access=_ACCESS,
                event_kind=MEMORY_EVENT_RETENTION,
            ) == ()
        assert backend.snapshot() == host_before
    finally:
        if ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


__all__ = []
