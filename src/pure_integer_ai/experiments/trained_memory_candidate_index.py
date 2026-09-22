"""从 Memory 权威事件增量派生结构候选索引；分页不截断历史。"""
from __future__ import annotations

from typing import Callable, Iterator

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.memory_event import (
    EvidencePayload,
    HypothesisPayload,
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_HYPOTHESIS,
)
from pure_integer_ai.cognition.shared.memory_event_log import (
    MaterializedMemoryEvent,
    MemoryEventLog,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.storage.backend import TYPE_INT, register_extension_table
from pure_integer_ai.storage.discipline import DISC_NONE
from pure_integer_ai.storage.memory_event import MemoryEventRecordStore


# 路由语义升级必须新建投影，旧 cursor 不能跳过历史；权威 O/H/E 与旧表均保留。
CANDIDATE_INDEX_TABLE = "dialogue_memory_candidate_index_v3"
CANDIDATE_CURSOR_TABLE = "dialogue_memory_candidate_cursor_v3"
_ROUTE_HASHER = Hasher("trained_memory_candidate.route.v1")
_SELECTOR_COLUMNS = (
    "space_id", "tenant_id", "user_id", "session_id", "source_kind", "source_id",
)
_PAGE_SIZE = 256


def evidence_target_route(hypothesis_key: tuple[int, ...]) -> tuple[int, ...]:
    """用完整 Hypothesis 引用定位所有后续 Evidence 事件。"""
    return 4, len(hypothesis_key), *hypothesis_key


# object-model: resource_owner; representation=runtime; interop=memory-candidate-index-v3
class MemoryCandidateIndex:
    """持有可丢弃物理索引；候选语义始终回读原事件核验。"""

    def __init__(
            self,
            event_log: MemoryEventLog,
            source: SourceRef,
            route_keys: Callable[[tuple[int, ...]], tuple[tuple[int, ...], ...]],
            ) -> None:
        """按完整 owner/source 分区注册索引，并从持久 cursor 接续事件尾。"""
        self.event_log = event_log
        self.backend = event_log.backend
        self.source = source
        self.route_keys = route_keys
        self.access = MemoryAccessContext(
            source.owner.tenant_id, source.owner.user_id, source.owner.session_id)
        self.selector = dict(zip(_SELECTOR_COLUMNS, (
            event_log.memory_space_id,
            source.owner.tenant_id, source.owner.user_id, source.owner.session_id,
            source.source_kind, source.source_id,
        )))
        self.records = MemoryEventRecordStore(self.backend)
        selector_schema = [(name, TYPE_INT) for name in _SELECTOR_COLUMNS]
        register_extension_table(
            self.backend, CANDIDATE_INDEX_TABLE,
            selector_schema + [(name, TYPE_INT) for name in (
                "row_seq", "timeline_seq", "route_ordinal", "route_hash",
                "event_hash")],
            discipline=DISC_NONE,
            indexes=[
                (*_SELECTOR_COLUMNS, "route_hash", "row_seq"),
                (*_SELECTOR_COLUMNS, "row_seq"),
            ],
            recovery_key=(*_SELECTOR_COLUMNS, "row_seq"),
        )
        register_extension_table(
            self.backend, CANDIDATE_CURSOR_TABLE,
            selector_schema + [("timeline_seq", TYPE_INT), ("row_count", TYPE_INT)],
            discipline=DISC_NONE,
            indexes=[_SELECTOR_COLUMNS], recovery_key=_SELECTOR_COLUMNS,
        )
        self._rebuild = False
        checkpoints = self.backend.select(CANDIDATE_CURSOR_TABLE, self.selector)
        if len(checkpoints) > 1:
            raise RuntimeError("Memory 候选索引 cursor 不唯一")
        if checkpoints:
            expected = checkpoints[0]["row_count"]
            actual = self.backend.count(CANDIDATE_INDEX_TABLE, self.selector)
            self._rebuild = expected != actual
        self.synchronize()

    def _owns_source(self, source: SourceRef) -> bool:
        """只接受当前完整 owner 与来源谱系；版本继续保留在事件内。"""
        return (source.owner == self.source.owner
                and source.source_kind == self.source.source_kind
                and source.source_id == self.source.source_id)

    def _event_routes(self, event: MaterializedMemoryEvent) -> tuple[tuple[int, ...], ...]:
        """从候选或 Evidence 的显式目标关系恢复可索引路由。"""
        payload = event.event.payload
        if isinstance(payload, HypothesisPayload):
            return (self.route_keys(payload.hypothesis.candidate_key)
                    if self._owns_source(payload.hypothesis.observation) else ())
        if isinstance(payload, EvidencePayload):
            parents = self.event_log.query(
                access=self.access, event_kind=MEMORY_EVENT_HYPOTHESIS,
                object_ref=payload.hypothesis_ref)
            if len(parents) != 1 or not isinstance(parents[0].event.payload, HypothesisPayload):
                raise RuntimeError("Memory Evidence 目标缺少唯一 Hypothesis 声明")
            if self._owns_source(parents[0].event.payload.hypothesis.observation):
                return (evidence_target_route(payload.hypothesis_ref.stable_key()),)
        return ()

    def synchronize(self) -> None:
        """分页重放尚未索引的事件；每页先物化路由再提交可恢复 cursor。"""
        checkpoints = self.backend.select(CANDIDATE_CURSOR_TABLE, self.selector)
        if len(checkpoints) > 1:
            raise RuntimeError("Memory 候选索引 cursor 不唯一")
        checkpoint = checkpoints[0] if checkpoints else None
        cursor = 0 if checkpoint is None or self._rebuild else checkpoint["timeline_seq"]
        count = 0 if checkpoint is None or self._rebuild else checkpoint["row_count"]
        while True:
            records = self.records.timeline_records_after(
                self.event_log.memory_space_id, cursor, limit=_PAGE_SIZE)
            if not records:
                break
            for record in records:
                if (record.event_kind not in {MEMORY_EVENT_HYPOTHESIS, MEMORY_EVENT_EVIDENCE}
                        or record.owner_key != self.source.owner.stable_key()):
                    continue
                materialized = self.event_log.read(record.event_hash, access=self.access)
                if materialized is None:
                    raise RuntimeError("Memory 候选索引不能越过不可见的候选事件")
                for ordinal, route in enumerate(self._event_routes(materialized)):
                    count += 1
                    row = {
                        **self.selector, "row_seq": count,
                        "timeline_seq": record.timeline_seq,
                        "route_ordinal": ordinal,
                        "route_hash": _ROUTE_HASHER.h63(route),
                        "event_hash": record.event_hash,
                    }
                    existing = self.backend.select(CANDIDATE_INDEX_TABLE, {
                        **self.selector, "row_seq": count})
                    if existing and existing != [row]:
                        raise RuntimeError("Memory 候选索引与权威事件重建结果冲突")
                    if not existing:
                        self.backend.insert(CANDIDATE_INDEX_TABLE, row)
            cursor = records[-1].timeline_seq
            # 重建时保留旧水位直到完整恢复，避免半次重建隐藏旧索引行。
            if not self._rebuild:
                self._checkpoint(cursor, count)
            self.records.clear_runtime_caches()
        self._checkpoint(cursor, count)
        self._rebuild = False

    def _checkpoint(self, cursor: int, count: int) -> None:
        """仅更新可重建投影的恢复点，权威 O/H/E 不受影响。"""
        values = {"timeline_seq": cursor, "row_count": count}
        if not self.backend.update(CANDIDATE_CURSOR_TABLE, self.selector, values):
            self.backend.insert(CANDIDATE_CURSOR_TABLE, {**self.selector, **values})

    def matching_events(
            self, routes: tuple[tuple[int, ...], ...],
            ) -> Iterator[MaterializedMemoryEvent]:
        """按结构路由分页定位候选并核验完整键；哈希碰撞不产生语义命中。"""
        self.synchronize()
        yielded: set[int] = set()
        for route in sorted(set(routes)):
            cursor = 0
            while True:
                rows = self.backend.select(
                    CANDIDATE_INDEX_TABLE,
                    {**self.selector, "route_hash": _ROUTE_HASHER.h63(route)},
                    where_gt={"row_seq": cursor}, order_by="row_seq", limit=_PAGE_SIZE)
                if not rows:
                    break
                for row in rows:
                    event = self.event_log.read(row["event_hash"], access=self.access)
                    if event is None:
                        raise RuntimeError("Memory 索引引用了不可见事件")
                    if event.timeline.seq != row["timeline_seq"]:
                        raise RuntimeError("Memory 索引 owner/source/事件时间线漂移")
                    actual_routes = self._event_routes(event)
                    ordinal = row["route_ordinal"]
                    if (not 0 <= ordinal < len(actual_routes)
                            or _ROUTE_HASHER.h63(actual_routes[ordinal]) != row["route_hash"]):
                        raise RuntimeError("Memory 索引路由与候选本体不一致")
                    if actual_routes[ordinal] == route and event.event_hash not in yielded:
                        yielded.add(event.event_hash)
                        yield event
                cursor = rows[-1]["row_seq"]


__all__ = ["MemoryCandidateIndex", "evidence_target_route"]
