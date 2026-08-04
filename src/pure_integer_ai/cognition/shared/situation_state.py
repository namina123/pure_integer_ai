"""MD-02 对既有 Memory event、WorkMemory 投影和 A-10 依赖的薄适配。"""
from __future__ import annotations

import json
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorContextUpdate,
    AttractorDependency,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.memory_event import MemoryEvent
from pure_integer_ai.cognition.shared.memory_event_log import (
    MaterializedMemoryEvent,
    MemoryEventLog,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_QUERY,
    ScopeIdentity,
)
from pure_integer_ai.cognition.shared.work_memory_content import (
    WorkMemoryContentItem,
    WorkMemoryContentStore,
)
from pure_integer_ai.cognition.shared.work_memory_discourse import (
    WorkMemoryDiscourseProjection,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


SITUATION_STATE_VERSION = 1
PROJECTION_KINDS = (
    "ADOPTED_CANDIDATE",
    "ENTITY",
    "EVENT",
    "GOAL",
    "OPEN_QUESTION",
    "PROPOSITION",
    "UNRESOLVED_STATE",
)

_ENTRY_KEY_DOMAIN = "situation.projection_entry.v1"
_STATE_KEY_DOMAIN = "situation.current_projection.v1"


class SituationStateError(ValueError):
    """篇章三件套的边界、依赖或局部重建不闭合。"""


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """要求开放身份为非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise SituationStateError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise SituationStateError(f"{where} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def _canonical_integer_bytes(value: tuple[int, ...]) -> bytes:
    """把整数稳定键编码成平台无关、无空白的审计字节。"""
    _strict_key(value, where="canonical integer key")
    return (json.dumps(
        list(value), ensure_ascii=True, separators=(",", ":")) + "\n").encode(
            "ascii")


def _normalized_dependencies(
        dependencies: tuple[AttractorDependency, ...],
        *,
        where: str,
        ) -> tuple[AttractorDependency, ...]:
    """要求 typed dependency 非空、唯一并按完整稳定键排序。"""
    if (not isinstance(dependencies, tuple) or not dependencies
            or any(not isinstance(item, AttractorDependency)
                   for item in dependencies)):
        raise SituationStateError(f"{where} 必须含 typed dependency")
    by_key = {item.stable_key(): item for item in dependencies}
    if len(by_key) != len(dependencies):
        raise SituationStateError(f"{where} 不得重复 dependency")
    return tuple(by_key[key] for key in sorted(by_key))


def _validate_source_scope(source: SourceRef, scope: ScopeIdentity) -> None:
    """要求当前投影属于精确 source/owner/version 的 query。"""
    if not isinstance(source, SourceRef):
        raise TypeError("situation source 必须是 SourceRef")
    if not isinstance(scope, ScopeIdentity):
        raise TypeError("situation scope 必须是 ScopeIdentity")
    if scope.scope_kind != SCOPE_QUERY:
        raise SituationStateError("CurrentSituationProjection 必须绑定 query scope")
    if (scope.source != source or scope.owner != source.owner
            or scope.versions != source.versions):
        raise SituationStateError("situation source/scope/owner/version 不一致")


@dataclass(frozen=True)
class SituationProjectionEntry:
    """当前投影中的一个稳定槽位，仅引用既有 WorkMemory 内容。"""

    projection_key: tuple[int, ...]
    projection_kind: str
    content_ref: tuple[int, ...]
    dependencies: tuple[AttractorDependency, ...]
    revision: int = 0

    def __post_init__(self) -> None:
        _strict_key(self.projection_key, where="situation projection key")
        if self.projection_kind not in PROJECTION_KINDS:
            raise SituationStateError("situation projection kind 非法")
        _strict_key(self.content_ref, where="situation content ref")
        object.__setattr__(self, "dependencies", _normalized_dependencies(
            self.dependencies, where="situation projection dependencies"))
        assert_int(self.revision, _where="SituationProjectionEntry.revision")
        if type(self.revision) is not int or self.revision < 0:
            raise SituationStateError("situation projection revision 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回槽位、当前内容引用、依赖和 revision 的完整键。"""
        result = [
            SITUATION_STATE_VERSION,
            *_packed(self.projection_key),
            PROJECTION_KINDS.index(self.projection_kind) + 1,
            *_packed(self.content_ref),
            self.revision,
            len(self.dependencies),
        ]
        for dependency in self.dependencies:
            result.extend(_packed(dependency.stable_key()))
        return tuple(result)

    def canonical_bytes(self) -> bytes:
        """返回可直接逐字节比较的规范投影字节。"""
        return _canonical_integer_bytes(self.stable_key())


@dataclass(frozen=True)
class SituationDependencyLink:
    """一个 A-10 typed dependency 到当前投影槽位的可重建索引行。"""

    dependency_key: tuple[int, ...]
    projection_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _strict_key(self.dependency_key, where="situation dependency key")
        if not isinstance(self.projection_keys, tuple) or not self.projection_keys:
            raise SituationStateError("dependency link 必须命中投影")
        for key in self.projection_keys:
            _strict_key(key, where="dependency projection key")
        if (self.projection_keys != tuple(sorted(self.projection_keys))
                or len(set(self.projection_keys)) != len(self.projection_keys)):
            raise SituationStateError("dependency 投影键必须稳定有序且唯一")

    def stable_key(self) -> tuple[int, ...]:
        """返回 dependency 与全部派生投影引用。"""
        result = [*_packed(self.dependency_key), len(self.projection_keys)]
        for key in self.projection_keys:
            result.extend(_packed(key))
        return tuple(result)


@dataclass(frozen=True)
class SituationDependencyIndex:
    """由当前投影重建的 dependency 索引，不拥有长期对象。"""

    links: tuple[SituationDependencyLink, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.links, tuple)
                or any(not isinstance(item, SituationDependencyLink)
                       for item in self.links)):
            raise TypeError("SituationDependencyIndex links 类型错误")
        keys = tuple(item.dependency_key for item in self.links)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise SituationStateError("dependency index 必须稳定有序且唯一")

    @classmethod
    def from_entries(
            cls,
            entries: tuple[SituationProjectionEntry, ...],
            ) -> "SituationDependencyIndex":
        """从当前投影完整重建索引，避免维护第二份权威本体。"""
        mapping: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
        for entry in entries:
            if not isinstance(entry, SituationProjectionEntry):
                raise TypeError("dependency index entry 类型错误")
            for dependency in entry.dependencies:
                mapping.setdefault(dependency.stable_key(), set()).add(
                    entry.projection_key)
        return cls(tuple(
            SituationDependencyLink(key, tuple(sorted(mapping[key])))
            for key in sorted(mapping)
        ))

    def affected(
            self,
            changed_dependencies: tuple[AttractorDependency, ...],
            ) -> tuple[tuple[int, ...], ...]:
        """返回仅由变化 dependency 命中的投影槽位。"""
        normalized = _normalized_dependencies(
            changed_dependencies, where="changed dependencies")
        changed = {item.stable_key() for item in normalized}
        result = {
            projection_key
            for link in self.links
            if link.dependency_key in changed
            for projection_key in link.projection_keys
        }
        return tuple(sorted(result))

    def stable_key(self) -> tuple[int, ...]:
        """返回完整可重建索引键。"""
        result = [SITUATION_STATE_VERSION, len(self.links)]
        for link in self.links:
            result.extend(_packed(link.stable_key()))
        return tuple(result)


class SituationEventLog:
    """对既有 ``MemoryEventLog`` 的 source/query 边界委托 facade。"""

    def __init__(
            self,
            event_log: MemoryEventLog,
            source: SourceRef,
            query_scope: ScopeIdentity,
            ) -> None:
        if not isinstance(event_log, MemoryEventLog):
            raise TypeError("SituationEventLog 必须复用 MemoryEventLog")
        _validate_source_scope(source, query_scope)
        self.event_log = event_log
        self.source = source
        self.query_scope = query_scope
        self.access = MemoryAccessContext(
            source.owner.tenant_id,
            source.owner.user_id,
            source.owner.session_id,
        )

    def _validate_event(self, event: MemoryEvent) -> None:
        """拒绝其他 source、owner 或 version 的事件进入当前 situation。"""
        if not isinstance(event, MemoryEvent):
            raise TypeError("situation append 必须提供 MemoryEvent")
        if (event.scope.source != self.source
                or event.scope.owner != self.source.owner
                or event.scope.versions != self.source.versions
                or event.object_ref.owner != self.source.owner
                or event.object_ref.versions != self.source.versions):
            raise SituationStateError("situation event source/owner/version 越权")

    def append(self, event: MemoryEvent) -> MaterializedMemoryEvent:
        """委托既有 append-only MemoryEventLog，不建立第二个事件存储。"""
        self._validate_event(event)
        return self.event_log.append(event)

    def read(self, event_hash: int) -> MaterializedMemoryEvent:
        """经既有 ACL 回读一个当前 situation 事件。"""
        assert_int(event_hash, _where="SituationEventLog.event_hash")
        if type(event_hash) is not int or event_hash <= 0:
            raise SituationStateError("situation event hash 必须为正严格整数")
        materialized = self.event_log.read(event_hash, access=self.access)
        if materialized is None:
            raise SituationStateError("situation event 不存在或不可见")
        self._validate_event(materialized.event)
        return materialized

    def require_materialized(
            self,
            materialized: MaterializedMemoryEvent,
            ) -> MaterializedMemoryEvent:
        """证明调用方提交的事件确实来自当前 backing event log。"""
        if not isinstance(materialized, MaterializedMemoryEvent):
            raise TypeError("revision event 必须是 MaterializedMemoryEvent")
        restored = self.read(materialized.event_hash)
        if restored != materialized:
            raise SituationStateError("revision event 与 backing log 不一致")
        return restored


@dataclass(frozen=True)
class SituationProjectionReplacement:
    """一个受影响投影槽位及其已来源化 WorkMemory 替换项。"""

    entry: SituationProjectionEntry
    content_item: WorkMemoryContentItem

    def __post_init__(self) -> None:
        if not isinstance(self.entry, SituationProjectionEntry):
            raise TypeError("replacement entry 类型错误")
        if not isinstance(self.content_item, WorkMemoryContentItem):
            raise TypeError("replacement content item 类型错误")
        if self.entry.content_ref != self.content_item.content_ref():
            raise SituationStateError("replacement entry 未引用对应 WorkMemory item")


@dataclass(frozen=True)
class SituationRebuildReceipt:
    """一次后文修正的精确 invalidation/rebuild 与零宿主写证据。"""

    revision_event_hash: int
    update_key: tuple[int, ...]
    invalidated_projection_keys: tuple[tuple[int, ...], ...]
    rebuilt_projection_keys: tuple[tuple[int, ...], ...]
    unaffected_projection_keys: tuple[tuple[int, ...], ...]
    preserved_event_hashes: tuple[int, ...]
    before_projection_ref: tuple[int, ...]
    after_projection_ref: tuple[int, ...]
    work_memory_write_count: int
    unaffected_bit_identical: int
    original_events_preserved: int
    host_learning_write_count: int

    def __post_init__(self) -> None:
        assert_int(
            self.revision_event_hash,
            self.work_memory_write_count,
            self.unaffected_bit_identical,
            self.original_events_preserved,
            self.host_learning_write_count,
            _where="SituationRebuildReceipt",
        )
        if type(self.revision_event_hash) is not int or self.revision_event_hash <= 0:
            raise SituationStateError("rebuild revision event hash 非法")
        _strict_key(self.update_key, where="rebuild update key")
        for label, keys in (
                ("invalidated", self.invalidated_projection_keys),
                ("rebuilt", self.rebuilt_projection_keys),
                ("unaffected", self.unaffected_projection_keys)):
            if not isinstance(keys, tuple):
                raise TypeError(f"rebuild {label} keys 必须是 tuple")
            for key in keys:
                _strict_key(key, where=f"rebuild {label} key")
            if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
                raise SituationStateError(f"rebuild {label} keys 未稳定去重")
        if (not self.invalidated_projection_keys
                or self.invalidated_projection_keys
                != self.rebuilt_projection_keys):
            raise SituationStateError("局部 invalidation 与 rebuild 槽位不闭合")
        if set(self.unaffected_projection_keys) & set(
                self.invalidated_projection_keys):
            raise SituationStateError("unaffected 与 invalidated 投影重叠")
        if (not isinstance(self.preserved_event_hashes, tuple)
                or not self.preserved_event_hashes
                or any(type(item) is not int or item <= 0
                       for item in self.preserved_event_hashes)
                or self.preserved_event_hashes
                != tuple(sorted(set(self.preserved_event_hashes)))):
            raise SituationStateError("preserved event hashes 必须稳定非空且唯一")
        for key in (self.before_projection_ref, self.after_projection_ref):
            _strict_key(key, where="rebuild projection ref")
        if self.work_memory_write_count != len(self.rebuilt_projection_keys):
            raise SituationStateError("WorkMemory write 数与 rebuild 槽位不闭合")
        if self.unaffected_bit_identical != 1:
            raise SituationStateError("未受影响投影必须 bit-identical")
        if self.original_events_preserved != 1:
            raise SituationStateError("原事件必须 append-only 保留")
        if self.host_learning_write_count != 0:
            raise SituationStateError("MD-02 不得产生 host learning write")

    def stable_key(self) -> tuple[int, ...]:
        """返回事件、局部差异、投影前后和零写事实。"""
        result = [
            SITUATION_STATE_VERSION,
            self.revision_event_hash,
            *_packed(self.update_key),
        ]
        for keys in (
                self.invalidated_projection_keys,
                self.rebuilt_projection_keys,
                self.unaffected_projection_keys):
            result.append(len(keys))
            for key in keys:
                result.extend(_packed(key))
        result.extend((len(self.preserved_event_hashes),
                       *self.preserved_event_hashes,
                       *_packed(self.before_projection_ref),
                       *_packed(self.after_projection_ref),
                       self.work_memory_write_count,
                       self.unaffected_bit_identical,
                       self.original_events_preserved,
                       self.host_learning_write_count))
        return tuple(result)


class CurrentSituationProjection:
    """由 WorkMemory 引用构成的 query 当前投影和可重建依赖索引。"""

    def __init__(
            self,
            events: SituationEventLog,
            work_memory: WorkMemoryContentStore,
            scope: ScopeIdentity,
            entries: tuple[SituationProjectionEntry, ...],
            ) -> None:
        if not isinstance(events, SituationEventLog):
            raise TypeError("CurrentSituationProjection events 类型错误")
        if not isinstance(work_memory, WorkMemoryContentStore):
            raise TypeError("CurrentSituationProjection 必须复用 WorkMemoryContentStore")
        _validate_source_scope(events.source, scope)
        if scope != events.query_scope:
            raise SituationStateError("projection 与 event facade query scope 不一致")
        self.events = events
        self.work_memory = work_memory
        self.scope = scope
        self.source = events.source
        self._entries = self._normalize_entries(entries)
        self._validate_backing_entries(self._entries, work_memory)
        self._dependency_index = SituationDependencyIndex.from_entries(
            self._entries)

    @classmethod
    def from_work_memory_discourse(
            cls,
            events: SituationEventLog,
            work_memory: WorkMemoryContentStore,
            scope: ScopeIdentity,
            discourse: WorkMemoryDiscourseProjection,
            *,
            kind_by_role: dict[ObjectIdentity, str],
            dependencies_by_content_ref: dict[
                tuple[int, ...], tuple[AttractorDependency, ...]],
            ) -> "CurrentSituationProjection":
        """把既有 G-02 投影薄映射为稳定槽位，不复制其内容本体。"""
        if not isinstance(discourse, WorkMemoryDiscourseProjection):
            raise TypeError("discourse 必须是 WorkMemoryDiscourseProjection")
        if not isinstance(kind_by_role, dict):
            raise TypeError("kind_by_role 必须是 dict")
        if not isinstance(dependencies_by_content_ref, dict):
            raise TypeError("dependencies_by_content_ref 必须是 dict")
        entries = []
        for ordinal, item in enumerate(discourse.items, start=1):
            kind = kind_by_role.get(item.role)
            content_ref = item.content_ref()
            dependencies = dependencies_by_content_ref.get(content_ref)
            if kind is None or dependencies is None:
                raise SituationStateError("G-02 item 缺 projection kind 或 dependency")
            entry_key = integer_tuple_fingerprint(
                (
                    *_packed(scope.stable_key()),
                    *_packed(discourse.discourse_key),
                    *_packed(discourse.proposition_key),
                    *_packed(item.role.stable_key()),
                    ordinal,
                ),
                domain=_ENTRY_KEY_DOMAIN,
            )
            entries.append(SituationProjectionEntry(
                entry_key, kind, content_ref, dependencies, 0))
        if set(kind_by_role) != {item.role for item in discourse.items}:
            raise SituationStateError("kind_by_role 含遗漏或额外 Role")
        if set(dependencies_by_content_ref) != {
                item.content_ref() for item in discourse.items}:
            raise SituationStateError("dependency map 含遗漏或额外 content ref")
        return cls(events, work_memory, scope, tuple(entries))

    @staticmethod
    def _normalize_entries(
            entries: tuple[SituationProjectionEntry, ...],
            ) -> tuple[SituationProjectionEntry, ...]:
        """要求当前投影非空、槽位唯一并稳定排序。"""
        if (not isinstance(entries, tuple) or not entries
                or any(not isinstance(item, SituationProjectionEntry)
                       for item in entries)):
            raise TypeError("CurrentSituationProjection entries 类型错误")
        keys = tuple(item.projection_key for item in entries)
        if len(set(keys)) != len(keys):
            raise SituationStateError("CurrentSituationProjection 槽位不得重复")
        refs = tuple(item.content_ref for item in entries)
        if len(set(refs)) != len(refs):
            raise SituationStateError("CurrentSituationProjection 内容引用不得重复")
        return tuple(sorted(entries, key=lambda item: item.projection_key))

    def _validate_backing_entries(
            self,
            entries: tuple[SituationProjectionEntry, ...],
            store: WorkMemoryContentStore,
            ) -> None:
        """证明每个当前槽位精确引用同 source 的 active WorkMemory item。"""
        active = {item.content_ref(): item for item in store.active()}
        for entry in entries:
            item = active.get(entry.content_ref)
            if item is None:
                raise SituationStateError("projection 引用了非 active WorkMemory item")
            if (item.source != self.source
                    or item.source.owner != self.source.owner
                    or item.source.versions != self.source.versions
                    or item.lifespan_scope.owner != self.source.owner
                    or item.lifespan_scope.versions != self.source.versions):
                raise SituationStateError("projection WorkMemory source/owner/version 越权")

    def entries(self) -> tuple[SituationProjectionEntry, ...]:
        """返回按稳定槽位排序的当前投影。"""
        return self._entries

    @property
    def dependency_index(self) -> SituationDependencyIndex:
        """返回从当前投影重建的只读 dependency 索引。"""
        return self._dependency_index

    def entry_bytes(self) -> dict[tuple[int, ...], bytes]:
        """返回逐槽位规范字节，供局部修正直接比较 bit identity。"""
        return {item.projection_key: item.canonical_bytes()
                for item in self._entries}

    def state_key(self) -> tuple[int, ...]:
        """返回 source/scope、当前引用及可重建索引的完整状态。"""
        result = [
            SITUATION_STATE_VERSION,
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            len(self._entries),
        ]
        for entry in self._entries:
            result.extend(_packed(entry.stable_key()))
        result.extend(_packed(self._dependency_index.stable_key()))
        return tuple(result)

    def state_ref(self) -> tuple[int, ...]:
        """返回固定长度当前投影内容引用。"""
        return integer_tuple_fingerprint(
            self.state_key(), domain=_STATE_KEY_DOMAIN)

    def apply_revision(
            self,
            update: AttractorContextUpdate,
            revision_event: MaterializedMemoryEvent,
            replacements: tuple[SituationProjectionReplacement, ...],
            *,
            preserved_event_hashes: tuple[int, ...],
            ) -> SituationRebuildReceipt:
        """按 A-10 dependency 只重建命中槽位，并保留原事件和其余字节。"""
        if not isinstance(update, AttractorContextUpdate):
            raise TypeError("situation update 必须复用 AttractorContextUpdate")
        if update.scope != self.scope:
            raise SituationStateError("situation update scope 越权")
        self.events.require_materialized(revision_event)
        if (not isinstance(preserved_event_hashes, tuple)
                or not preserved_event_hashes
                or any(type(item) is not int or item <= 0
                       for item in preserved_event_hashes)
                or tuple(sorted(set(preserved_event_hashes)))
                != preserved_event_hashes):
            raise SituationStateError("preserved event hashes 必须稳定非空且唯一")
        if revision_event.event_hash in preserved_event_hashes:
            raise SituationStateError("preserved events 必须是 revision 之前的事件")
        preserved_before = tuple(
            self.events.read(event_hash) for event_hash in preserved_event_hashes)

        affected = self._dependency_index.affected(
            update.changed_dependencies)
        if not affected:
            raise SituationStateError("situation revision 未命中任何 dependency")
        if (not isinstance(replacements, tuple)
                or any(not isinstance(item, SituationProjectionReplacement)
                       for item in replacements)):
            raise TypeError("situation replacements 类型错误")
        by_key = {item.entry.projection_key: item for item in replacements}
        if len(by_key) != len(replacements) or tuple(sorted(by_key)) != affected:
            raise SituationStateError("replacement 未精确覆盖局部 invalidation")

        before_entries = {item.projection_key: item for item in self._entries}
        before_bytes = self.entry_bytes()
        before_state_ref = self.state_ref()
        replacement_refs = set()
        for key in affected:
            old = before_entries[key]
            replacement = by_key[key]
            new = replacement.entry
            item = replacement.content_item
            if (new.projection_kind != old.projection_kind
                    or new.revision != old.revision + 1):
                raise SituationStateError("replacement kind 或 revision 不连续")
            if item.supersedes != (old.content_ref,):
                raise SituationStateError("replacement 必须只 supersede 命中的前项")
            if (item.source != self.source
                    or item.lifespan_scope.owner != self.source.owner
                    or item.lifespan_scope.versions != self.source.versions):
                raise SituationStateError("replacement WorkMemory source/owner/version 越权")
            if new.content_ref in replacement_refs:
                raise SituationStateError("replacement content ref 不得重复")
            replacement_refs.add(new.content_ref)

        preview = self.work_memory.clone()
        for key in affected:
            preview.put(by_key[key].content_item)
        preview_entries = tuple(
            by_key[item.projection_key].entry
            if item.projection_key in by_key else item
            for item in self._entries)
        preview_entries = self._normalize_entries(preview_entries)
        self._validate_backing_entries(preview_entries, preview)
        preview_index = SituationDependencyIndex.from_entries(preview_entries)

        unaffected = tuple(sorted(set(before_entries) - set(affected)))
        preview_bytes = {
            item.projection_key: item.canonical_bytes()
            for item in preview_entries}
        if any(before_bytes[key] != preview_bytes[key] for key in unaffected):
            raise SituationStateError("未受影响投影发生字节漂移")
        before_active = {
            item.content_ref(): item.stable_key()
            for item in self.work_memory.active()}
        preview_active = {
            item.content_ref(): item.stable_key()
            for item in preview.active()}
        unaffected_refs = {
            before_entries[key].content_ref for key in unaffected}
        if any(before_active[ref] != preview_active.get(ref)
               for ref in unaffected_refs):
            raise SituationStateError("未受影响 WorkMemory 内容发生字节漂移")

        preserved_after = tuple(
            self.events.read(event_hash) for event_hash in preserved_event_hashes)
        if preserved_after != preserved_before:
            raise SituationStateError("原 situation event 被 revision 改写")

        before_work_memory_state = self.work_memory.state_key()
        self.work_memory.commit_preview(
            preview,
            expected_state_key=before_work_memory_state,
        )
        self._entries = preview_entries
        self._dependency_index = preview_index
        return SituationRebuildReceipt(
            revision_event.event_hash,
            update.stable_key(),
            affected,
            affected,
            unaffected,
            preserved_event_hashes,
            before_state_ref,
            self.state_ref(),
            len(affected),
            1,
            1,
            0,
        )


__all__ = [
    "CurrentSituationProjection",
    "PROJECTION_KINDS",
    "SITUATION_STATE_VERSION",
    "SituationDependencyIndex",
    "SituationDependencyLink",
    "SituationEventLog",
    "SituationProjectionEntry",
    "SituationProjectionReplacement",
    "SituationRebuildReceipt",
    "SituationStateError",
]
