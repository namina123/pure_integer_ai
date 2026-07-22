"""从 Memory event 重建 Hypothesis 当前聚合与查询索引。

本模块不宣布事实真值，也不执行巩固、衰减或 Top-K 评分。它只维护可删除派生表，
并用 dirty Hypothesis 反向索引把增量重建限制在受影响对象。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    HypothesisKey,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.identity import (
    OwnerScope,
    SourceRef,
    VISIBILITY_GLOBAL,
    VISIBILITY_SESSION,
    VISIBILITY_TENANT,
    VISIBILITY_USER,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_OBJECT_EVIDENCE,
    MEMORY_OBJECT_HYPOTHESIS,
    RETENTION_EPISODIC,
    DerivationTransitionPayload,
    EvidencePayload,
    EpisodePayload,
    HypothesisPayload,
    LifecycleTransitionPayload,
    MemoryObjectRef,
    ObservationPayload,
    RetentionTransitionPayload,
    UsePayload,
)
from pure_integer_ai.cognition.shared.memory_event_log import (
    MaterializedMemoryEvent,
    MemoryEventLog,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_MEMORY_OBJECT,
    IDENTITY_SOURCE_RECORD,
)
from pure_integer_ai.storage.memory_aggregate import (
    MEMORY_HYPOTHESIS_AGGREGATE_TABLE,
    MEMORY_HYPOTHESIS_DIRTY_TABLE,
    MEMORY_HYPOTHESIS_EVENT_TABLE,
    MEMORY_HYPOTHESIS_SOURCE_TABLE,
    MemoryAggregateIntegrityError,
    MemoryAggregateStore,
    MemoryHypothesisAggregateRecord,
    MemoryHypothesisDirtyRecord,
    MemoryHypothesisEventIndexRecord,
    MemoryHypothesisSourceRecord,
)


MEMORY_EVIDENCE_PROVISIONAL = 1
MEMORY_EVIDENCE_CORROBORATED = 2
MEMORY_EVIDENCE_CONFLICTED = 3

_DERIVED_HASH_HYPOTHESIS_KIND = 1
_DERIVED_HASH_COMPETITION = 2
_DERIVED_HASH_CONTEXT = 3


class MemoryCorroborationPolicy(Protocol):
    """按开放 Hypothesis kind 判断统计支持是否达到可标注门。"""

    def is_corroborated(
            self,
            hypothesis: HypothesisKey,
            *,
            support_count: int,
            contradict_count: int,
            support_source_count: int,
            contradict_source_count: int,
            ) -> bool:
        """返回无冲突支持是否达到调用方注入的证据门。"""


class ConservativeCorroborationPolicy:
    """未配置按 kind 证据门时保持 provisional 的保守策略。"""

    def is_corroborated(
            self,
            hypothesis: HypothesisKey,
            *,
            support_count: int,
            contradict_count: int,
            support_source_count: int,
            contradict_source_count: int,
            ) -> bool:
        """拒绝默认升格，避免把单次或高频统计写成可靠性结论。"""
        del hypothesis, support_count, contradict_count
        del support_source_count, contradict_source_count
        return False


@dataclass(frozen=True)
class MemoryAggregateRebuildReport:
    """一次全量或 dirty 重建的确定性计数报告。"""

    scanned_event_count: int
    processed_hypothesis_count: int
    aggregate_count: int
    source_index_count: int

    def __post_init__(self) -> None:
        """校验报告只包含非负严格整数。"""
        for value in (
                self.scanned_event_count,
                self.processed_hypothesis_count,
                self.aggregate_count,
                self.source_index_count):
            if type(value) is not int or value < 0:
                raise ValueError("Memory aggregate rebuild 报告必须为非负严格整数")


class MemoryHypothesisAggregateIndex:
    """一个 Memory 空间的事件反向索引、dirty queue 和 aggregate facade。"""

    def __init__(
            self,
            event_log: MemoryEventLog,
            corroboration_policy: MemoryCorroborationPolicy | None = None,
            ) -> None:
        """绑定事件日志并安装新事件 listener；既有事件需显式全量重建。"""
        if not isinstance(event_log, MemoryEventLog):
            raise TypeError("event_log 必须是 MemoryEventLog")
        policy = corroboration_policy or ConservativeCorroborationPolicy()
        if not hasattr(policy, "is_corroborated"):
            raise TypeError("corroboration_policy 缺少 is_corroborated")
        self.event_log = event_log
        self.policy = policy
        self.store = MemoryAggregateStore(
            event_log.backend, event_log.memory_space_id)
        event_log.attach_append_listener(self.mark_event_dirty)

    def mark_event_dirty(self, materialized: MaterializedMemoryEvent) -> None:
        """为新事件建立 Hypothesis 反向索引并合并唯一 dirty 键。"""
        if not isinstance(materialized, MaterializedMemoryEvent):
            raise TypeError("mark_event_dirty 需要 MaterializedMemoryEvent")
        event = materialized.event
        access = self._owner_access(event.object_ref.owner)
        hypothesis_ref = self._hypothesis_for_event(event, access=access)
        if hypothesis_ref is None:
            return
        hypothesis_hash = self._hypothesis_hash(hypothesis_ref)
        owner_key = hypothesis_ref.owner.stable_key()
        self.store.index_event(MemoryHypothesisEventIndexRecord(
            self.event_log.memory_space_id,
            hypothesis_hash,
            materialized.event_hash,
            event.event_kind,
            materialized.object_hash,
            event.timestamp.seq,
            owner_key,
        ))
        self.store.enqueue_dirty(MemoryHypothesisDirtyRecord(
            self.event_log.memory_space_id,
            hypothesis_hash,
            materialized.event_hash,
            event.timestamp.seq,
            owner_key,
        ))

    def rebuild_all(self, *, access: MemoryAccessContext) -> MemoryAggregateRebuildReport:
        """扫描一次全部可见事件，重建可见派生表且不写回事件。"""
        self._require_access(access)
        self._delete_visible_derived(access)
        materialized_events = self.event_log.query(access=access)
        hypotheses: dict[int, MemoryObjectRef] = {}
        event_records: dict[int, list[MemoryHypothesisEventIndexRecord]] = {}
        for materialized in materialized_events:
            hypothesis_ref = self._hypothesis_for_event(
                materialized.event, access=access)
            if hypothesis_ref is None:
                continue
            hypothesis_hash = self._hypothesis_hash(hypothesis_ref)
            previous = hypotheses.get(hypothesis_hash)
            if previous is not None and previous != hypothesis_ref:
                raise MemoryAggregateIntegrityError(
                    "Hypothesis hash 命中不同完整引用")
            hypotheses[hypothesis_hash] = hypothesis_ref
            event_records.setdefault(hypothesis_hash, []).append(
                MemoryHypothesisEventIndexRecord(
                    self.event_log.memory_space_id,
                    hypothesis_hash,
                    materialized.event_hash,
                    materialized.event.event_kind,
                    materialized.object_hash,
                    materialized.event.timestamp.seq,
                    hypothesis_ref.owner.stable_key(),
                ))
        declared = {
            self._hypothesis_hash(item.event.object_ref)
            for item in materialized_events
            if item.event.event_kind == MEMORY_EVENT_HYPOTHESIS
        }
        if declared != set(hypotheses):
            raise MemoryAggregateIntegrityError(
                "可见 Hypothesis 反向索引缺声明或含孤儿事件")
        for hypothesis_hash, records in sorted(event_records.items()):
            self.store.replace_events(
                tuple(sorted(records, key=lambda item: (
                    item.event_seq, item.event_hash))),
                hypothesis_hash,
            )
        aggregates: list[MemoryHypothesisAggregateRecord] = []
        source_count = 0
        for hypothesis_hash in sorted(hypotheses):
            aggregate, sources = self._rebuild_hypothesis(
                hypotheses[hypothesis_hash], access=access)
            aggregates.append(aggregate)
            source_count += len(sources)
            self.store.delete_dirty(hypothesis_hash)
        return MemoryAggregateRebuildReport(
            len(materialized_events),
            len(hypotheses),
            len(aggregates),
            source_count,
        )

    def rebuild_dirty(
            self,
            *,
            access: MemoryAccessContext,
            limit: int | None = None,
            ) -> MemoryAggregateRebuildReport:
        """只重建 ACL 可见的 dirty Hypothesis，不扫描全事件表。"""
        self._require_access(access)
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("dirty rebuild limit 必须为正严格整数")
        dirty = tuple(
            item for item in self.store.list_dirty()
            if access.can_read(OwnerScope(*item.owner_key))
        )
        if limit is not None:
            dirty = dirty[:limit]
        source_count = 0
        for item in dirty:
            hypothesis_ref = self._hypothesis_ref_from_hash(
                item.hypothesis_hash)
            aggregate, sources = self._rebuild_hypothesis(
                hypothesis_ref, access=access)
            source_count += len(sources)
            self.store.delete_dirty(item.hypothesis_hash)
        return MemoryAggregateRebuildReport(
            0,
            len(dirty),
            len(dirty),
            source_count,
        )

    def read(
            self,
            hypothesis_ref: MemoryObjectRef,
            *,
            access: MemoryAccessContext,
            ) -> MemoryHypothesisAggregateRecord | None:
        """按完整 Hypothesis 引用读取可见 aggregate，拒绝 hash/owner 漂移。"""
        self._require_hypothesis_ref(hypothesis_ref)
        self._require_access(access)
        if not access.can_read(hypothesis_ref.owner):
            return None
        hypothesis_hash = self._hypothesis_hash(hypothesis_ref)
        record = self.store.read_aggregate(hypothesis_hash)
        if record is None:
            return None
        if record.owner_key != hypothesis_ref.owner.stable_key():
            raise MemoryAggregateIntegrityError("aggregate owner 与完整引用漂移")
        if self._hypothesis_ref_from_hash(hypothesis_hash) != hypothesis_ref:
            raise MemoryAggregateIntegrityError("aggregate hash 与完整引用漂移")
        return record

    def query(
            self,
            *,
            access: MemoryAccessContext,
            hypothesis_kind: tuple[int, ...] | None = None,
            context: tuple[int, ...] | None = None,
            evidence_state: int | None = None,
            lifecycle_state: int | None = None,
            retention_state: int | None = None,
            source: SourceRef | None = None,
            ) -> tuple[MemoryHypothesisAggregateRecord, ...]:
        """先按 owner 与可选 kind/context/status/source 索引过滤 aggregate。"""
        self._require_access(access)
        where: dict[str, int] = {"space_id": self.event_log.memory_space_id}
        if hypothesis_kind is not None:
            where["hypothesis_kind_hash"] = self.hypothesis_kind_hash(
                hypothesis_kind)
        if context is not None:
            where["context_hash"] = self.context_hash(context)
        for name, value in (
                ("evidence_state", evidence_state),
                ("lifecycle_state", lifecycle_state),
                ("retention_state", retention_state)):
            if value is not None:
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} 必须为正严格整数")
                where[name] = value
        source_candidates = None
        if source is not None:
            if not isinstance(source, SourceRef):
                raise TypeError("source 必须是 SourceRef")
            source_hash = self.source_hash(source)
            source_candidates = set()
            source_key = source.stable_key()
            source_rows = self.event_log.backend.select(
                MEMORY_HYPOTHESIS_SOURCE_TABLE,
                {
                    "space_id": self.event_log.memory_space_id,
                    "source_hash": source_hash,
                },
            )
            for row in source_rows:
                if (not access.can_read(OwnerScope(
                    row["owner_tenant_id"], row["owner_user_id"],
                    row["owner_session_id"], row["owner_visibility"],
                ))
                        or tuple(row[f"source_key_{index:02d}"]
                                 for index in range(11)) != source_key):
                    continue
                source_candidates.add(row["hypothesis_hash"])
        records: dict[int, MemoryHypothesisAggregateRecord] = {}
        for owner_where in self._owner_where(access):
            rows = self.event_log.backend.select(
                MEMORY_HYPOTHESIS_AGGREGATE_TABLE,
                {**where, **owner_where},
            )
            for row in rows:
                record = MemoryHypothesisAggregateRecord.from_row(row)
                if (source_candidates is not None
                        and record.hypothesis_hash not in source_candidates):
                    continue
                if hypothesis_kind is not None or context is not None:
                    hypothesis_ref = self._hypothesis_ref_from_hash(
                        record.hypothesis_hash)
                    declaration = self._declaration_event(
                        hypothesis_ref,
                        access=access,
                    ).event.payload
                    if not isinstance(declaration, HypothesisPayload):
                        raise MemoryAggregateIntegrityError(
                            "aggregate 查询目标声明 payload 漂移")
                    if (hypothesis_kind is not None
                            and declaration.hypothesis.hypothesis_kind
                            != hypothesis_kind):
                        continue
                    if (context is not None
                            and declaration.hypothesis.scope.stable_key()
                            != context):
                        continue
                previous = records.get(record.hypothesis_hash)
                if previous is not None and previous != record:
                    raise MemoryAggregateIntegrityError(
                        "query 命中重复漂移 aggregate")
                records[record.hypothesis_hash] = record
        return tuple(records[key] for key in sorted(records))

    def sources(
            self,
            hypothesis_ref: MemoryObjectRef,
            *,
            access: MemoryAccessContext,
            ) -> tuple[MemoryHypothesisSourceRecord, ...]:
        """返回一个可见 Hypothesis 的活动来源分账。"""
        self._require_hypothesis_ref(hypothesis_ref)
        self._require_access(access)
        if not access.can_read(hypothesis_ref.owner):
            return ()
        hypothesis_hash = self._hypothesis_hash(hypothesis_ref)
        records = tuple(
            MemoryHypothesisSourceRecord.from_row(row)
            for row in self.event_log.backend.select(
                MEMORY_HYPOTHESIS_SOURCE_TABLE,
                {
                    "space_id": self.event_log.memory_space_id,
                    "hypothesis_hash": hypothesis_hash,
                },
            )
        )
        return tuple(sorted(records, key=lambda item: (
            item.source_hash, item.stance)))

    def hypothesis_kind_hash(self, key: tuple[int, ...]) -> int:
        """计算开放 Hypothesis kind 的稳定派生索引 hash。"""
        return self._derived_hash(_DERIVED_HASH_HYPOTHESIS_KIND, key)

    def context_hash(self, key: tuple[int, ...]) -> int:
        """计算调用方提供 context 稳定键的派生索引 hash。"""
        return self._derived_hash(_DERIVED_HASH_CONTEXT, key)

    def source_hash(self, source: SourceRef) -> int:
        """计算与 SourceRecord identity namespace 一致的来源 hash。"""
        if not isinstance(source, SourceRef):
            raise TypeError("source 必须是 SourceRef")
        return self.event_log.scoped_identities.registry.identity_hash(
            IDENTITY_SOURCE_RECORD, source.stable_key())

    def _rebuild_hypothesis(
            self,
            hypothesis_ref: MemoryObjectRef,
            *,
            access: MemoryAccessContext,
            ) -> tuple[
                MemoryHypothesisAggregateRecord,
                tuple[MemoryHypothesisSourceRecord, ...],
            ]:
        """只读取反向索引关联事件，重放一个 Hypothesis 当前派生态。"""
        self._require_hypothesis_ref(hypothesis_ref)
        hypothesis_hash = self._hypothesis_hash(hypothesis_ref)
        index_records = self.store.list_events(hypothesis_hash)
        if not index_records:
            raise MemoryAggregateIntegrityError(
                "dirty Hypothesis 缺事件反向索引，需先全量 rebuild")
        events: list[MaterializedMemoryEvent] = []
        for index_record in index_records:
            materialized = self.event_log.read(
                index_record.event_hash, access=access)
            if materialized is None:
                raise MemoryAggregateIntegrityError(
                    "event index 指向当前 ACL 不可见事件")
            if self._hypothesis_for_event(
                    materialized.event, access=access) != hypothesis_ref:
                raise MemoryAggregateIntegrityError(
                    "event index 指向其他 Hypothesis")
            events.append(materialized)
        declarations = tuple(
            item for item in events
            if item.event.event_kind == MEMORY_EVENT_HYPOTHESIS
        )
        if len(declarations) != 1:
            raise MemoryAggregateIntegrityError(
                "Hypothesis aggregate 需要唯一声明事件")
        declaration = declarations[0].event
        payload = declaration.payload
        if (not isinstance(payload, HypothesisPayload)
                or declaration.object_ref != hypothesis_ref):
            raise MemoryAggregateIntegrityError("Hypothesis 声明 payload 漂移")

        evidence_items = tuple(
            item for item in events
            if isinstance(item.event.payload, EvidencePayload)
        )
        evidence_refs = {item.event.object_ref for item in evidence_items}
        superseded_refs = {
            item.event.payload.supersedes_ref
            for item in evidence_items
            if item.event.payload.supersedes_ref is not None
        }
        if not superseded_refs.issubset(evidence_refs):
            raise MemoryAggregateIntegrityError(
                "Evidence supersede 目标未进入同一 Hypothesis 反向索引")
        derivation_inactive_refs = {
            item.event.payload.target_ref
            for item in events
            if (isinstance(
                item.event.payload, DerivationTransitionPayload)
                and item.event.payload.target_ref.object_kind
                == MEMORY_OBJECT_EVIDENCE)
        }
        if not derivation_inactive_refs.issubset(evidence_refs):
            raise MemoryAggregateIntegrityError(
                "Derivation transition 指向其他 Hypothesis 的 Evidence")
        active_evidence = tuple(
            item for item in evidence_items
            if (item.event.object_ref not in superseded_refs
                and item.event.object_ref not in derivation_inactive_refs)
        )

        source_buckets: dict[
            tuple[tuple[int, ...], int], list[int]
        ] = {}
        source_by_key: dict[tuple[int, ...], SourceRef] = {}
        support_seqs: list[int] = []
        refute_seqs: list[int] = []
        unknown_seqs: list[int] = []
        for item in active_evidence:
            evidence = item.event.payload
            source = self._evidence_source(evidence, access=access)
            source_key = source.stable_key()
            source_by_key[source_key] = source
            source_buckets.setdefault(
                (source_key, evidence.stance), []).append(
                    evidence.observed_at.seq)
            if evidence.stance == EVIDENCE_SUPPORT:
                support_seqs.append(evidence.observed_at.seq)
            elif evidence.stance == EVIDENCE_REFUTE:
                refute_seqs.append(evidence.observed_at.seq)
            elif evidence.stance == EVIDENCE_UNKNOWN:
                unknown_seqs.append(evidence.observed_at.seq)
            else:
                raise MemoryAggregateIntegrityError("活动 Evidence stance 未注册")

        support_sources = {
            source_key for source_key, stance in source_buckets
            if stance == EVIDENCE_SUPPORT
        }
        refute_sources = {
            source_key for source_key, stance in source_buckets
            if stance == EVIDENCE_REFUTE
        }
        support_count = len(support_seqs)
        contradict_count = len(refute_seqs)
        if support_sources and refute_sources:
            evidence_state = MEMORY_EVIDENCE_CONFLICTED
        else:
            corroborated = self.policy.is_corroborated(
                payload.hypothesis,
                support_count=support_count,
                contradict_count=contradict_count,
                support_source_count=len(support_sources),
                contradict_source_count=len(refute_sources),
            )
            if type(corroborated) is not bool:
                raise MemoryAggregateIntegrityError(
                    "corroboration policy 必须返回严格 bool")
            evidence_state = (
                MEMORY_EVIDENCE_CORROBORATED
                if support_sources and corroborated
                else MEMORY_EVIDENCE_PROVISIONAL
            )

        retention = payload.initial_retention
        lifecycle = payload.initial_lifecycle
        use_seqs: list[int] = []
        state_events = sorted(events, key=lambda item: (
            item.event.timestamp.clock.stable_key(),
            item.event.timestamp.seq,
            item.event_hash,
        ))
        for item in state_events:
            event_payload = item.event.payload
            if isinstance(event_payload, RetentionTransitionPayload):
                if event_payload.from_state != retention:
                    raise MemoryAggregateIntegrityError(
                        "retention aggregate 重放不连续")
                retention = event_payload.to_state
            elif isinstance(event_payload, LifecycleTransitionPayload):
                if event_payload.from_state != lifecycle:
                    raise MemoryAggregateIntegrityError(
                        "lifecycle aggregate 重放不连续")
                lifecycle = event_payload.to_state
            elif (isinstance(event_payload, DerivationTransitionPayload)
                  and event_payload.target_ref == hypothesis_ref):
                if event_payload.from_state != lifecycle:
                    raise MemoryAggregateIntegrityError(
                        "derivation lifecycle aggregate 重放不连续")
                lifecycle = event_payload.to_state
            elif isinstance(event_payload, UsePayload):
                use_seqs.append(event_payload.used_at.seq)

        hypothesis = payload.hypothesis
        source_records: list[MemoryHypothesisSourceRecord] = []
        for (source_key, stance), seqs in sorted(source_buckets.items()):
            source_records.append(MemoryHypothesisSourceRecord(
                self.event_log.memory_space_id,
                hypothesis_hash,
                self.source_hash(source_by_key[source_key]),
                stance,
                min(seqs),
                max(seqs),
                len(seqs),
                hypothesis_ref.owner.stable_key(),
                source_by_key[source_key].stable_key(),
            ))
        aggregate = MemoryHypothesisAggregateRecord(
            self.event_log.memory_space_id,
            hypothesis_hash,
            hypothesis_ref.owner.stable_key(),
            self.hypothesis_kind_hash(hypothesis.hypothesis_kind),
            self._derived_hash(
                _DERIVED_HASH_COMPETITION, hypothesis.competition_key),
            self.context_hash(hypothesis.scope.stable_key()),
            self.source_hash(hypothesis.observation),
            payload.created_at.seq,
            max((*support_seqs, *refute_seqs, *unknown_seqs), default=0),
            max(support_seqs, default=0),
            max(refute_seqs, default=0),
            max(use_seqs, default=0),
            support_count,
            contradict_count,
            len(unknown_seqs),
            len(source_by_key),
            len(support_sources),
            len(refute_sources),
            len(use_seqs),
            retention,
            lifecycle,
            evidence_state,
        )
        self.store.replace_sources(tuple(source_records), hypothesis_hash)
        self.store.replace_aggregate(aggregate)
        return aggregate, tuple(source_records)

    def _evidence_source(
            self,
            payload: EvidencePayload,
            *,
            access: MemoryAccessContext,
            ) -> SourceRef:
        """恢复 Evidence 的直接来源，或沿 Episode→Observation 精确回溯。"""
        if payload.source is not None:
            return payload.source
        if payload.episode_ref is None:
            raise MemoryAggregateIntegrityError("Evidence 缺少可回溯来源")
        episode_event = self._declaration_event(
            payload.episode_ref, access=access)
        if not isinstance(episode_event.event.payload, EpisodePayload):
            raise MemoryAggregateIntegrityError("Evidence episode_ref 不是 Episode")
        observation_event = self._declaration_event(
            episode_event.event.payload.input_observation_ref,
            access=access,
        )
        observation = observation_event.event.payload
        if not isinstance(observation, ObservationPayload):
            raise MemoryAggregateIntegrityError(
                "Episode input_observation_ref 不是 Observation")
        return observation.source

    def _hypothesis_for_event(
            self,
            event,
            *,
            access: MemoryAccessContext,
            ) -> MemoryObjectRef | None:
        """返回事件影响的唯一 Hypothesis；无关事件返回空。"""
        payload = event.payload
        if isinstance(payload, HypothesisPayload):
            return event.object_ref
        if isinstance(payload, EvidencePayload):
            return payload.hypothesis_ref
        if isinstance(payload, (RetentionTransitionPayload,
                                LifecycleTransitionPayload)):
            return (
                payload.target_ref
                if payload.target_ref.object_kind == MEMORY_OBJECT_HYPOTHESIS
                else None
            )
        if isinstance(payload, DerivationTransitionPayload):
            return self._hypothesis_for_ref(
                payload.target_ref, access=access)
        if isinstance(payload, UsePayload):
            return self._hypothesis_for_ref(
                payload.memory_ref, access=access)
        return None

    def _hypothesis_for_ref(
            self,
            ref: MemoryObjectRef,
            *,
            access: MemoryAccessContext,
            ) -> MemoryObjectRef | None:
        """把 Hypothesis 或其 Evidence 引用归并到 Hypothesis。"""
        if ref.object_kind == MEMORY_OBJECT_HYPOTHESIS:
            return ref
        if ref.object_kind != MEMORY_OBJECT_EVIDENCE:
            return None
        event = self._declaration_event(ref, access=access).event
        payload = event.payload
        if not isinstance(payload, EvidencePayload):
            raise MemoryAggregateIntegrityError("Evidence 引用 payload 漂移")
        return payload.hypothesis_ref

    def _declaration_event(
            self,
            ref: MemoryObjectRef,
            *,
            access: MemoryAccessContext,
            ) -> MaterializedMemoryEvent:
        """按完整对象引用读取唯一声明事件。"""
        entries = self.event_log.query(access=access, object_ref=ref)
        declarations = tuple(
            item for item in entries
            if item.event.object_ref == ref and item.event.is_declaration
        )
        if len(declarations) != 1:
            raise MemoryAggregateIntegrityError("Memory 引用没有唯一声明事件")
        return declarations[0]

    def _delete_visible_derived(self, access: MemoryAccessContext) -> None:
        """删除当前 ACL 可见的派生行，不触碰其他 owner 或事件。"""
        hypothesis_hashes: set[int] = set()
        for table in (
                MEMORY_HYPOTHESIS_AGGREGATE_TABLE,
                MEMORY_HYPOTHESIS_SOURCE_TABLE,
                MEMORY_HYPOTHESIS_EVENT_TABLE,
                MEMORY_HYPOTHESIS_DIRTY_TABLE):
            rows = self.event_log.backend.select(
                table, {"space_id": self.event_log.memory_space_id})
            for row in rows:
                owner = OwnerScope(
                    row["owner_tenant_id"], row["owner_user_id"],
                    row["owner_session_id"], row["owner_visibility"],
                )
                if access.can_read(owner):
                    hypothesis_hashes.add(row["hypothesis_hash"])
        for hypothesis_hash in sorted(hypothesis_hashes):
            self.event_log.backend.delete(
                MEMORY_HYPOTHESIS_AGGREGATE_TABLE,
                {
                    "space_id": self.event_log.memory_space_id,
                    "hypothesis_hash": hypothesis_hash,
                },
            )
            self.store.delete_sources(hypothesis_hash)
            self.store.delete_events(hypothesis_hash)
            self.store.delete_dirty(hypothesis_hash)

    def _hypothesis_hash(self, ref: MemoryObjectRef) -> int:
        """计算并核验 Hypothesis 完整引用的 Memory object hash。"""
        self._require_hypothesis_ref(ref)
        registry = self.event_log.scoped_identities.registry
        value = registry.identity_hash(
            IDENTITY_MEMORY_OBJECT, ref.stable_key())
        existing = registry.find(IDENTITY_MEMORY_OBJECT, ref.stable_key())
        if existing != value:
            raise MemoryAggregateIntegrityError(
                "Hypothesis object identity 尚未完整登记")
        return value

    def _hypothesis_ref_from_hash(self, hypothesis_hash: int) -> MemoryObjectRef:
        """从 self-headed Memory identity 恢复完整 Hypothesis 引用。"""
        key = self.event_log.scoped_identities.registry.read_key(
            IDENTITY_MEMORY_OBJECT, hypothesis_hash)
        ref = MemoryObjectRef.from_stable_key(key)
        self._require_hypothesis_ref(ref)
        return ref

    def _derived_hash(self, tag: int, key: tuple[int, ...]) -> int:
        """为非权威检索键计算稳定 hash，完整语义仍从事件恢复。"""
        if (type(tag) is not int or tag <= 0 or not isinstance(key, tuple)
                or not key or any(type(value) is not int for value in key)):
            raise ValueError("derived index key 必须是非空严格整数 tuple")
        return self.event_log.scoped_identities.registry.identity_hash(
            IDENTITY_MEMORY_OBJECT, (tag, *key))

    def _require_hypothesis_ref(self, ref: MemoryObjectRef) -> None:
        """核验引用属于当前 Memory 空间且种类为 Hypothesis。"""
        if (not isinstance(ref, MemoryObjectRef)
                or ref.object_kind != MEMORY_OBJECT_HYPOTHESIS):
            raise ValueError("需要 Memory Hypothesis 引用")
        if ref.memory_space != self.event_log.memory_space_identity:
            raise ValueError("Hypothesis 引用属于其他 Memory 空间")

    @staticmethod
    def _require_access(access: MemoryAccessContext) -> None:
        """拒绝省略 ACL 上下文。"""
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("Memory aggregate 查询必须提供 MemoryAccessContext")

    @staticmethod
    def _owner_access(owner: OwnerScope) -> MemoryAccessContext:
        """从事件 owner 构造等权读取上下文，不提升 visibility。"""
        return MemoryAccessContext(
            owner.tenant_id, owner.user_id, owner.session_id)

    @staticmethod
    def _owner_where(access: MemoryAccessContext) -> tuple[dict[str, int], ...]:
        """生成 GLOBAL 到 SESSION 的精确 owner 预过滤条件。"""
        result: list[dict[str, int]] = [{
            "owner_visibility": VISIBILITY_GLOBAL,
            "owner_tenant_id": 0,
            "owner_user_id": 0,
            "owner_session_id": 0,
        }]
        if access.tenant_id:
            result.append({
                "owner_visibility": VISIBILITY_TENANT,
                "owner_tenant_id": access.tenant_id,
                "owner_user_id": 0,
                "owner_session_id": 0,
            })
        if access.user_id:
            result.append({
                "owner_visibility": VISIBILITY_USER,
                "owner_tenant_id": access.tenant_id,
                "owner_user_id": access.user_id,
                "owner_session_id": 0,
            })
        if access.session_id:
            result.append({
                "owner_visibility": VISIBILITY_SESSION,
                "owner_tenant_id": access.tenant_id,
                "owner_user_id": access.user_id,
                "owner_session_id": access.session_id,
            })
        return tuple(result)


__all__ = [
    "MEMORY_EVIDENCE_PROVISIONAL",
    "MEMORY_EVIDENCE_CORROBORATED",
    "MEMORY_EVIDENCE_CONFLICTED",
    "MemoryCorroborationPolicy",
    "ConservativeCorroborationPolicy",
    "MemoryAggregateRebuildReport",
    "MemoryHypothesisAggregateIndex",
]
