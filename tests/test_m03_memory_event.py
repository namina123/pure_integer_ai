"""M-03 Memory 事件对象、H-00 恢复和 legacy 迁移对抗测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactSchema,
    FormalArtifact,
    artifact_identity,
)
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    HypothesisTransition,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    VISIBILITY_SESSION,
    concept_identity,
)
from pure_integer_ai.cognition.shared.memory_event import (
    ArtifactPayload,
    CapabilityPayload,
    EpisodePayload,
    MEMORY_EVENT_ARTIFACT,
    MEMORY_EVENT_CAPABILITY,
    MEMORY_EVENT_EPISODE,
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_LEGACY_IMPORT,
    MEMORY_EVENT_LIFECYCLE,
    MEMORY_EVENT_OBSERVATION,
    MEMORY_EVENT_RETENTION,
    MEMORY_EVENT_USE,
    MEMORY_OBJECT_ARTIFACT,
    MEMORY_OBJECT_CAPABILITY,
    MEMORY_OBJECT_EPISODE,
    MEMORY_OBJECT_OBSERVATION,
    MEMORY_OBJECT_USE,
    MemoryEvent,
    MemoryLinkedRef,
    ObservationPayload,
    RETENTION_CONSOLIDATED,
    RETENTION_EPISODIC,
    RetentionTransitionPayload,
    UsePayload,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_hypothesis import (
    MemoryHypothesisEventSink,
)
from pure_integer_ai.cognition.shared.memory_legacy import (
    append_legacy_import,
    legacy_experience_count_payload,
    legacy_memory_item_payload,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_CREATED,
    CLOCK_MEMORY_IMPORT,
    CLOCK_MEMORY_LIFECYCLE,
    CLOCK_MEMORY_OBSERVED,
    CLOCK_MEMORY_USED,
    LogicalClock,
    LogicalClockIdentity,
    document_scope,
    episode_scope,
    session_scope,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_HEADER_TABLE,
    IDENTITY_MEMORY_EVENT,
    IDENTITY_MEMORY_OBJECT,
    IDENTITY_PART_TABLE,
)
from pure_integer_ai.storage.discipline import AppendOnlyViolation
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_CHUNK_WIDTH,
    MEMORY_EVENT_PART_TABLE,
    MEMORY_EVENT_TABLE,
    MemoryEventIntegrityError,
)
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run


def _backend(kind: str):
    """为双后端契约测试创建独立存储。"""
    if kind == "dict":
        return DictBackend()
    if kind == "sqlite":
        return SQLiteBackend()
    raise ValueError(kind)


def _versions() -> VersionBundle:
    """构造非零版本，防止测试只覆盖 legacy 默认版本。"""
    return VersionBundle(
        CorpusVersion(1),
        ParserVersion(2),
        PrimitiveVersion(3),
        CurriculumVersion(4),
    )


def _owner(session_id: int = 3) -> OwnerScope:
    """构造 session 私有测试 owner。"""
    return OwnerScope(1, 2, session_id, VISIBILITY_SESSION)


def _source(owner: OwnerScope | None = None, *, document_id: int = 11,
            source_id: int = 10) -> SourceRef:
    """构造与 Memory owner/version 对齐的来源。"""
    return SourceRef(
        1,
        source_id,
        document_id,
        owner or _owner(),
        _versions(),
    )


def _scopes(source: SourceRef):
    """构造同 owner/version 的 session、document 和 episode scope。"""
    session = session_scope(
        source.owner.session_id,
        owner=source.owner,
        versions=source.versions,
    )
    document = document_scope(source)
    episode = episode_scope(1, parent=document)
    return session, document, episode


def _timestamp(scope, clock_kind: int, seq: int = 1):
    """在指定 scope 的完整时钟内构造测试逻辑时间。"""
    return LogicalClock(
        LogicalClockIdentity(scope, clock_kind), seq - 1).advance()


def _core_refs(ctx):
    """在 Core 中物化上下文、概念、结构和关系类型测试端点。"""
    ontology = ctx.graph_ontology
    context = ontology.materialize(concept_identity((301,)))
    concept = ontology.materialize(concept_identity((302,)))
    structure = ontology.materialize(concept_identity((303,)))
    proposition = ontology.materialize(concept_identity((304,)))
    relation = ontology.materialize(relation_concept_identity((305,)))
    return context, concept, structure, proposition, relation


def _append_observation(ctx, source: SourceRef, refs):
    """追加一个字段齐全的 Observation 并返回物化事件。"""
    _, _, episode = _scopes(source)
    context, concept, structure, proposition, relation = refs
    payload = ObservationPayload(
        source,
        MemoryLinkedRef.core(context),
        (concept,),
        (concept, structure),
        structure,
        (proposition,),
        (MemoryLinkedRef.core(relation),),
        _timestamp(episode, CLOCK_MEMORY_OBSERVED),
    )
    ref = memory_object_ref(
        ctx.memory_interact_events.memory_space_identity,
        MEMORY_OBJECT_OBSERVATION,
        payload.stable_key(),
        owner=source.owner,
        versions=source.versions,
    )
    return ctx.memory_interact_events.append(MemoryEvent(
        MEMORY_EVENT_OBSERVATION, ref, episode, payload))


def _append_episode(ctx, source: SourceRef, observation, refs):
    """追加一个带候选、选择、输出、结果和版本身份的 Episode。"""
    session, _, episode = _scopes(source)
    context, concept, structure, proposition, _ = refs
    payload = EpisodePayload(
        observation.event.object_ref,
        MemoryLinkedRef.core(context),
        (MemoryLinkedRef.core(concept), MemoryLinkedRef.core(structure)),
        MemoryLinkedRef.core(structure),
        MemoryLinkedRef.core(proposition),
        (observation.event.object_ref,),
        (MemoryLinkedRef.core(context),),
        None,
        7,
        session,
        _timestamp(episode, CLOCK_MEMORY_CREATED, 2),
    )
    ref = memory_object_ref(
        ctx.memory_interact_events.memory_space_identity,
        MEMORY_OBJECT_EPISODE,
        payload.stable_key(),
        owner=source.owner,
        versions=source.versions,
    )
    return ctx.memory_interact_events.append(MemoryEvent(
        MEMORY_EVENT_EPISODE, ref, episode, payload))


def _hypothesis(source: SourceRef, candidate: int) -> HypothesisKey:
    """构造同一竞争组内的 H-00 候选。"""
    return HypothesisKey(
        (41,),
        (candidate,),
        (42,),
        document_scope(source),
        source,
    )


def test_memory_sink_isolates_same_local_event_id_across_candidate_protocols():
    """同 owner 下不同 Hypothesis 协议的相同裸 Evidence id 不得互相污染。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source()
        first = _hypothesis(source, 1)
        second = HypothesisKey(
            (99,),
            (1,),
            (42,),
            document_scope(source),
            source,
        )
        sink = MemoryHypothesisEventSink(ctx.memory_interact_events)
        first_ledger = HypothesisLedger(sink)
        second_ledger = HypothesisLedger(sink)
        first_ledger.register(first)
        second_ledger.register(second)
        first_ledger.append_evidence(
            EvidenceRecord(7, first, EVIDENCE_SUPPORT, (1,), source, 1))
        second_ledger.append_evidence(
            EvidenceRecord(7, second, EVIDENCE_SUPPORT, (2,), source, 1))

        access = MemoryAccessContext(1, 2, 3)
        restored_first = sink.load_ledger(
            access=access, hypotheses=(first,))
        restored_second = sink.load_ledger(
            access=access, hypotheses=(second,))
        assert restored_first.evidence_history(first)[0].reason_key == (1,)
        assert restored_second.evidence_history(second)[0].reason_key == (2,)
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_observation_episode_and_explicit_use_are_separate(kind: str):
    """Episode 创建及 used_memory_refs 不得自动生成 Use；显式 Use 才增加事件。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        source = _source()
        refs = _core_refs(ctx)
        observation = _append_observation(ctx, source, refs)
        episode = _append_episode(ctx, source, observation, refs)
        access = MemoryAccessContext(1, 2, 3)

        assert ctx.memory_interact_events.query(
            access=access, event_kind=MEMORY_EVENT_OBSERVATION) == (observation,)
        assert ctx.memory_interact_events.query(
            access=access, event_kind=MEMORY_EVENT_EPISODE) == (episode,)
        assert ctx.memory_interact_events.query(
            access=access, event_kind=MEMORY_EVENT_USE) == ()

        _, _, episode_scope_value = _scopes(source)
        payload = UsePayload(
            observation.event.object_ref,
            episode.event.object_ref,
            MemoryLinkedRef.core(refs[0]),
            MemoryLinkedRef.core(refs[3]),
            _timestamp(episode_scope_value, CLOCK_MEMORY_USED, 3),
        )
        use_ref = memory_object_ref(
            ctx.memory_interact_events.memory_space_identity,
            MEMORY_OBJECT_USE,
            payload.stable_key(),
            owner=source.owner,
            versions=source.versions,
        )
        use = ctx.memory_interact_events.append(MemoryEvent(
            MEMORY_EVENT_USE, use_ref, episode_scope_value, payload))
        assert ctx.memory_interact_events.query(
            access=access, event_kind=MEMORY_EVENT_USE) == (use,)
        assert MemoryEvent.from_stable_key(
            use.event.stable_key()) == use.event
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_h00_rebuild_preserves_conflict_and_orthogonal_states(kind: str):
    """CONSOLIDATED、SUPERSEDED 和正负证据冲突可同时表达并完整重建。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        source = _source()
        old = _hypothesis(source, 1)
        replacement = _hypothesis(source, 2)
        sink = MemoryHypothesisEventSink(ctx.memory_interact_events)
        ledger = HypothesisLedger(sink)
        ledger.register(old)
        ledger.register(replacement)
        support = EvidenceRecord(
            101, old, EVIDENCE_SUPPORT, (501,), source, 1, (601,))
        refute = EvidenceRecord(
            102, old, EVIDENCE_REFUTE, (502,), source, 2, (602,))
        ledger.append_evidence(support)
        ledger.append_evidence(refute)
        ledger.append_transition(HypothesisTransition(
            201,
            old,
            LIFECYCLE_ACTIVE,
            LIFECYCLE_SUPERSEDED,
            refute.evidence_id,
            (701,),
            3,
            replacement,
        ))

        access = MemoryAccessContext(1, 2, 3)
        evidence_events = ctx.memory_interact_events.query(
            access=access, event_kind=MEMORY_EVENT_EVIDENCE)
        old_ref = ctx.memory_interact_events.query(
            access=access,
            event_kind=MEMORY_EVENT_HYPOTHESIS,
        )[0].event.object_ref
        evidence_refs = tuple(
            entry.event.object_ref for entry in evidence_events)
        retention = RetentionTransitionPayload(
            old_ref,
            RETENTION_EPISODIC,
            RETENTION_CONSOLIDATED,
            evidence_refs,
            _timestamp(
                old.scope, CLOCK_MEMORY_LIFECYCLE, 5),
        )
        ctx.memory_interact_events.append(MemoryEvent(
            MEMORY_EVENT_RETENTION,
            old_ref,
            old.scope,
            retention,
        ))

        rebuilt = MemoryHypothesisEventSink(
            ctx.memory_interact_events).load_ledger(access=access)
        snapshot = rebuilt.snapshot(old)
        assert snapshot.epistemic_status == EPISTEMIC_CONFLICTED
        assert snapshot.lifecycle == LIFECYCLE_SUPERSEDED
        assert snapshot.support_evidence_ids == (101,)
        assert snapshot.refute_evidence_ids == (102,)
        assert len(ctx.memory_interact_events.query(
            access=access, event_kind=MEMORY_EVENT_RETENTION)) == 1
        assert len(ctx.memory_interact_events.query(
            access=access, event_kind=MEMORY_EVENT_LIFECYCLE)) == 1
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_artifact_and_capability_keep_typed_contracts(kind: str):
    """Artifact/Capability 保存来源、schema、契约、证据引用和独立对象身份。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        source = _source()
        refs = _core_refs(ctx)
        observation = _append_observation(ctx, source, refs)
        _, _, episode = _scopes(source)
        artifact_kind = concept_identity(
            (801,), owner=source.owner, versions=source.versions)
        value_type = concept_identity(
            (802,), owner=source.owner, versions=source.versions)
        unit = concept_identity(
            (803,), owner=source.owner, versions=source.versions)
        schema = ArtifactSchema(value_type, unit)
        identity = artifact_identity(
            source, artifact_kind, schema, (804,), (9, 8, 7), episode)
        artifact = FormalArtifact(
            identity, artifact_kind, schema, source, (9, 8, 7), episode)
        artifact_payload = ArtifactPayload(
            artifact.identity,
            observation.event.object_ref,
            _timestamp(episode, CLOCK_MEMORY_CREATED, 4),
        )
        artifact_ref = memory_object_ref(
            ctx.memory_interact_events.memory_space_identity,
            MEMORY_OBJECT_ARTIFACT,
            artifact.identity.stable_key(),
            owner=source.owner,
            versions=source.versions,
        )
        artifact_event = ctx.memory_interact_events.append(MemoryEvent(
            MEMORY_EVENT_ARTIFACT,
            artifact_ref,
            episode,
            artifact_payload,
        ))
        capability_payload = CapabilityPayload(
            MemoryLinkedRef.core(refs[0]),
            artifact_ref,
            (901, 902, 903),
            (),
            _timestamp(episode, CLOCK_MEMORY_CREATED, 5),
        )
        capability_ref = memory_object_ref(
            ctx.memory_interact_events.memory_space_identity,
            MEMORY_OBJECT_CAPABILITY,
            capability_payload.stable_key(),
            owner=source.owner,
            versions=source.versions,
        )
        capability_event = ctx.memory_interact_events.append(MemoryEvent(
            MEMORY_EVENT_CAPABILITY,
            capability_ref,
            episode,
            capability_payload,
        ))
        access = MemoryAccessContext(1, 2, 3)
        assert ctx.memory_interact_events.query(
            access=access, event_kind=MEMORY_EVENT_ARTIFACT) == (
                artifact_event,)
        assert ctx.memory_interact_events.query(
            access=access, event_kind=MEMORY_EVENT_CAPABILITY) == (
                capability_event,)
    finally:
        backend.close()


def test_clock_identity_prevents_cross_scope_sequence_comparison():
    """相同裸 seq 在不同 owner/scope 中必须保留不同完整 timestamp identity。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        first_source = _source(_owner(3), document_id=11, source_id=10)
        second_source = _source(_owner(4), document_id=12, source_id=20)
        first_sink = MemoryHypothesisEventSink(ctx.memory_interact_events)
        first_ledger = HypothesisLedger(first_sink)
        second_sink = MemoryHypothesisEventSink(ctx.memory_interact_events)
        second_ledger = HypothesisLedger(second_sink)
        first = _hypothesis(first_source, 1)
        second = _hypothesis(second_source, 1)
        first_ledger.register(first)
        second_ledger.register(second)
        first_ledger.append_evidence(EvidenceRecord(
            1, first, EVIDENCE_SUPPORT, (1,), first_source, 7))
        second_ledger.append_evidence(EvidenceRecord(
            1, second, EVIDENCE_SUPPORT, (1,), second_source, 7))

        first_event = ctx.memory_interact_events.query(
            access=MemoryAccessContext(1, 2, 3),
            event_kind=MEMORY_EVENT_EVIDENCE,
        )[0].event
        second_event = ctx.memory_interact_events.query(
            access=MemoryAccessContext(1, 2, 4),
            event_kind=MEMORY_EVENT_EVIDENCE,
        )[0].event
        assert first_event.timestamp.seq == second_event.timestamp.seq == 8
        assert first_event.timestamp.clock != second_event.timestamp.clock
        assert first_event.timestamp.stable_key() != second_event.timestamp.stable_key()
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_legacy_import_does_not_create_new_semantics(kind: str):
    """旧 memory_item/experience_count 只生成 LEGACY_IMPORT，不生成 Hypothesis/Evidence/Use。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        source = _source()
        _, document, _ = _scopes(source)
        imported_at = _timestamp(document, CLOCK_MEMORY_IMPORT)
        memory_row = {
            "space_id": ctx.memory_read.space_id,
            "local_id": 1,
            "content_hash": 101,
            "status": 2,
            "session_id": None,
            "count": 3,
            "success_count": 2,
            "seg_type": 1,
            "info_ref_space": ctx.space_id,
            "info_ref_id": 9,
            "context_tag": 7,
            "round_id": 6,
        }
        experience_row = {
            "space_id": ctx.space_id,
            "local_id": 9,
            "ctx_code": 7,
            "speaker_code": 8,
            "base_freq": 10,
            "e_sn": 2,
            "e_tn": 3,
            "observe_tn": 4,
        }
        first = append_legacy_import(
            ctx.memory_read_events,
            legacy_memory_item_payload(memory_row, imported_at),
            scope=document,
            owner=source.owner,
            versions=source.versions,
        )
        second = append_legacy_import(
            ctx.memory_read_events,
            legacy_experience_count_payload(
                experience_row,
                _timestamp(document, CLOCK_MEMORY_IMPORT, 2),
            ),
            scope=document,
            owner=source.owner,
            versions=source.versions,
        )
        access = MemoryAccessContext(1, 2, 3)
        assert set(ctx.memory_read_events.query(
            access=access, event_kind=MEMORY_EVENT_LEGACY_IMPORT)) == {
                first, second}
        for event_kind in (
                MEMORY_EVENT_HYPOTHESIS,
                MEMORY_EVENT_EVIDENCE,
                MEMORY_EVENT_USE):
            assert ctx.memory_read_events.query(
                access=access, event_kind=event_kind) == ()
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_event_acl_append_only_and_corruption_fail_closed(kind: str):
    """跨 session 不可见，事件行不可改删，owner 漂移和重复行必须失败。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        event = _append_observation(ctx, _source(), _core_refs(ctx))
        assert ctx.memory_interact_events.query(
            access=MemoryAccessContext(1, 2, 99)) == ()
        assert ctx.memory_interact_events.read(
            event.event_hash,
            access=MemoryAccessContext(1, 2, 99),
        ) is None
        with pytest.raises(AppendOnlyViolation):
            backend.update(
                MEMORY_EVENT_TABLE,
                where={"event_hash": event.event_hash},
                set_={"owner_session_id": 99},
            )
        with pytest.raises(AppendOnlyViolation):
            backend.delete(
                MEMORY_EVENT_TABLE,
                where={"event_hash": event.event_hash},
            )

        snapshot = backend.snapshot()
        snapshot[MEMORY_EVENT_TABLE][0]["owner_session_id"] = 99
        backend.load_snapshot(snapshot)
        ctx.memory_interact_events.clear_runtime_caches()
        with pytest.raises(MemoryEventIntegrityError):
            ctx.memory_interact_events.query(
                access=MemoryAccessContext(1, 2, 99))
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_event_dump_load_roundtrip(kind: str, tmp_path):
    """正式 dump/load 后事件、对象、scope、时钟和 ACL 身份必须完整恢复。"""
    first_backend = _backend(kind)
    try:
        first = make_train_context(first_backend)
        expected = _append_observation(
            first, _source(), _core_refs(first))
        dump_run(
            first_backend,
            str(tmp_path),
            "m03_event",
            spaces=[
                first.space_id,
                first.memory_read.space_id,
                first.memory_interact.space_id,
            ],
            tables=DUMP_TABLES,
        )
    finally:
        first_backend.close()

    second_backend = _backend(kind)
    try:
        second = make_train_context(second_backend)
        assert load_run(
            second_backend, str(tmp_path), "m03_event") == [1, 2, 3]
        restored = second.memory_interact_events.query(
            access=MemoryAccessContext(1, 2, 3),
            event_kind=MEMORY_EVENT_OBSERVATION,
        )
        assert restored == (expected,)
        assert MEMORY_EVENT_TABLE in DUMP_TABLES
    finally:
        second_backend.close()


def test_evaluation_clone_rebuilds_event_log_without_owner_leak():
    """V-06 clone 重建事件 facade，宿主私有事件对评测 session 不可见。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        _append_observation(ctx, _source(), _core_refs(ctx))
        cloned_backend = clone_backend(backend)
        try:
            cloned = clone_train_context(ctx, cloned_backend, label="m03")
            eval_owner = cloned.scope_owner
            assert cloned.memory_interact_events is not ctx.memory_interact_events
            assert cloned.memory_interact_events.query(
                access=MemoryAccessContext(
                    eval_owner.tenant_id,
                    eval_owner.user_id,
                    eval_owner.session_id,
                ),
            ) == ()
            assert len(cloned.memory_interact_events.query(
                access=MemoryAccessContext(1, 2, 3))) == 1
            assert backend.count(MEMORY_EVENT_TABLE) == 1
            assert cloned_backend.count(MEMORY_EVENT_TABLE) == 1
        finally:
            cloned_backend.close()
    finally:
        backend.close()


def test_orphan_memory_reference_and_duplicate_declaration_fail_closed():
    """未声明引用和同对象不同声明事件不得进入 Memory。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source()
        refs = _core_refs(ctx)
        _, _, episode = _scopes(source)
        orphan = memory_object_ref(
            ctx.memory_interact_events.memory_space_identity,
            MEMORY_OBJECT_OBSERVATION,
            (999,),
            owner=source.owner,
            versions=source.versions,
        )
        payload = EpisodePayload(
            orphan,
            None,
            (),
            None,
            None,
            (),
            (),
            None,
            1,
            _scopes(source)[0],
            _timestamp(episode, CLOCK_MEMORY_CREATED),
        )
        episode_ref = memory_object_ref(
            ctx.memory_interact_events.memory_space_identity,
            MEMORY_OBJECT_EPISODE,
            payload.stable_key(),
            owner=source.owner,
            versions=source.versions,
        )
        with pytest.raises(MemoryEventIntegrityError, match="不存在"):
            ctx.memory_interact_events.append(MemoryEvent(
                MEMORY_EVENT_EPISODE, episode_ref, episode, payload))

        observation = _append_observation(ctx, source, refs)
        other_payload = replace(
            observation.event.payload,
            observed_at=_timestamp(episode, CLOCK_MEMORY_OBSERVED, 9),
        )
        with pytest.raises(ValueError, match="对象键"):
            MemoryEvent(
                MEMORY_EVENT_OBSERVATION,
                observation.event.object_ref,
                episode,
                other_payload,
            )
    finally:
        backend.close()


def test_private_event_can_reference_visible_global_source():
    """Memory owner 独立于来源 owner；私有事件可引用全局来源但反向不可见。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        private_owner = _owner()
        source = SourceRef(
            1, 77, 88, GLOBAL_OWNER_SCOPE, _versions())
        private_session = session_scope(
            3, owner=private_owner, versions=source.versions)
        private_episode = episode_scope(1, parent=private_session)
        refs = _core_refs(ctx)
        payload = ObservationPayload(
            source,
            MemoryLinkedRef.core(refs[0]),
            (refs[1],),
            (refs[1],),
            refs[2],
            (refs[3],),
            (),
            _timestamp(private_episode, CLOCK_MEMORY_OBSERVED),
        )
        ref = memory_object_ref(
            ctx.memory_interact_events.memory_space_identity,
            MEMORY_OBJECT_OBSERVATION,
            payload.stable_key(),
            owner=private_owner,
            versions=source.versions,
        )
        event = ctx.memory_interact_events.append(MemoryEvent(
            MEMORY_EVENT_OBSERVATION, ref, private_episode, payload))
        assert ctx.memory_interact_events.query(
            access=MemoryAccessContext(1, 2, 3)) == (event,)
        assert ctx.memory_interact_events.query(
            access=MemoryAccessContext()) == ()
    finally:
        backend.close()


def test_retention_and_lifecycle_histories_reject_competing_transitions():
    """同一对象不能重复巩固，也不能从过期 lifecycle from_state 分叉。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source()
        old = _hypothesis(source, 1)
        replacement = _hypothesis(source, 2)
        sink = MemoryHypothesisEventSink(ctx.memory_interact_events)
        ledger = HypothesisLedger(sink)
        ledger.register(old)
        ledger.register(replacement)
        evidence = EvidenceRecord(
            1, old, EVIDENCE_REFUTE, (1,), source, 1)
        ledger.append_evidence(evidence)
        access = MemoryAccessContext(1, 2, 3)
        old_ref = next(
            entry.event.object_ref
            for entry in ctx.memory_interact_events.query(
                access=access, event_kind=MEMORY_EVENT_HYPOTHESIS)
            if entry.event.payload.hypothesis == old
        )
        evidence_ref = ctx.memory_interact_events.query(
            access=access, event_kind=MEMORY_EVENT_EVIDENCE)[0].event.object_ref
        first_retention = RetentionTransitionPayload(
            old_ref,
            RETENTION_EPISODIC,
            RETENTION_CONSOLIDATED,
            (evidence_ref,),
            _timestamp(old.scope, CLOCK_MEMORY_LIFECYCLE, 2),
        )
        first_event = MemoryEvent(
            MEMORY_EVENT_RETENTION, old_ref, old.scope, first_retention)
        assert ctx.memory_interact_events.append(first_event) == (
            ctx.memory_interact_events.append(first_event))
        second_retention = replace(
            first_retention,
            changed_at=_timestamp(old.scope, CLOCK_MEMORY_LIFECYCLE, 3),
        )
        with pytest.raises(MemoryEventIntegrityError, match="from_state"):
            ctx.memory_interact_events.append(MemoryEvent(
                MEMORY_EVENT_RETENTION,
                old_ref,
                old.scope,
                second_retention,
            ))

        ledger.append_transition(HypothesisTransition(
            10,
            old,
            LIFECYCLE_ACTIVE,
            LIFECYCLE_SUPERSEDED,
            1,
            (2,),
            4,
            replacement,
        ))
        stale = replace(
            ctx.memory_interact_events.query(
                access=access,
                event_kind=MEMORY_EVENT_LIFECYCLE,
            )[0].event.payload,
            to_state=2,
            changed_at=_timestamp(old.scope, CLOCK_MEMORY_LIFECYCLE, 6),
        )
        with pytest.raises(MemoryEventIntegrityError, match="from_state"):
            ctx.memory_interact_events.append(MemoryEvent(
                MEMORY_EVENT_LIFECYCLE, old_ref, old.scope, stale))
    finally:
        backend.close()


def test_normalized_event_identity_does_not_duplicate_payload_parts():
    """object/event identity 必须由正规化记录恢复，禁止把 payload 重复展开到 identity_part。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        event = _append_observation(ctx, _source(), _core_refs(ctx))
        for identity_kind in (
                IDENTITY_MEMORY_OBJECT, IDENTITY_MEMORY_EVENT):
            assert backend.count(
                IDENTITY_HEADER_TABLE,
                where={"identity_kind": identity_kind}) == 0
            assert backend.count(
                IDENTITY_PART_TABLE,
                where={"identity_kind": identity_kind}) == 0
        expected_chunks = (
            len(event.event.payload.stable_key())
            + MEMORY_EVENT_CHUNK_WIDTH - 1
        ) // MEMORY_EVENT_CHUNK_WIDTH
        assert backend.count(MEMORY_EVENT_PART_TABLE) == expected_chunks
        assert expected_chunks == 10
    finally:
        backend.close()


@pytest.mark.parametrize("corruption", ["duplicate_chunk", "payload_value"])
def test_payload_chunk_corruption_fails_closed(corruption: str):
    """重复 chunk 或 payload 位漂移不得被 event/object 外部 identity 恢复器接受。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        _append_observation(ctx, _source(), _core_refs(ctx))
        snapshot = backend.snapshot()
        rows = snapshot[MEMORY_EVENT_PART_TABLE]
        if corruption == "duplicate_chunk":
            rows.append(dict(rows[0]))
        else:
            rows[0]["part_00"] += 1
        backend.load_snapshot(snapshot)
        ctx.memory_interact_events.clear_runtime_caches()
        with pytest.raises((MemoryEventIntegrityError, ValueError)):
            ctx.memory_interact_events.query(
                access=MemoryAccessContext(1, 2, 3))
    finally:
        backend.close()
