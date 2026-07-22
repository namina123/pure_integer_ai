"""Memory 不可变事件的身份、ACL、引用完整性和持久化 facade。

object/event identity 使用正规化外部恢复器，避免把同一 payload 重复展开到通用
identity_part；事件信封和固定宽度 payload chunk 仍是 append-only 权威记录。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_ARTIFACT,
    MEMORY_EVENT_CAPABILITY,
    MEMORY_EVENT_DERIVATION,
    MEMORY_EVENT_EPISODE,
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_INTAKE_MANIFEST,
    MEMORY_EVENT_LEGACY_IMPORT,
    MEMORY_EVENT_LIFECYCLE,
    MEMORY_EVENT_OBSERVATION,
    MEMORY_EVENT_PARSE_FAILURE,
    MEMORY_EVENT_RETENTION,
    MEMORY_EVENT_RESOLUTION,
    MEMORY_EVENT_USE,
    MEMORY_OBJECT_INTAKE_MANIFEST,
    RETENTION_CONSOLIDATED,
    RETENTION_EPISODIC,
    TIME_AXIS_CREATED,
    TIME_AXIS_OBSERVED,
    TIME_AXIS_USED,
    INTAKE_DERIVED_MANIFEST,
    EvidencePayload,
    DerivationTransitionPayload,
    HypothesisPayload,
    IntakeManifestPayload,
    MemoryEvent,
    MemoryObjectRef,
    LifecycleTransitionPayload,
    ObservationPayload,
    ParseFailurePayload,
    RetentionTransitionPayload,
    source_reparse_lineage_key,
    declaration_object_key,
    payload_from_stable_key,
)
from pure_integer_ai.cognition.shared.memory_overlay import (
    CoreIdentityCatalog,
    MemoryAccessContext,
    MemoryOverlayQueryError,
)
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.cognition.shared.hypothesis import (
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.identity import OwnerScope
from pure_integer_ai.storage.assertion_identity import (
    ExternalIdentityKey,
    IDENTITY_MEMORY_EVENT,
    IDENTITY_MEMORY_OBJECT,
)
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_TABLE,
    MemoryEventIntegrityError,
    MemoryEventRecord,
    MemoryEventRecordStore,
)
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_MEMORY,
    SpaceIdentity,
    SpaceRegistry,
)


_DECLARATION_EVENT_KINDS = frozenset({
    MEMORY_EVENT_OBSERVATION,
    MEMORY_EVENT_EPISODE,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_USE,
    MEMORY_EVENT_ARTIFACT,
    MEMORY_EVENT_CAPABILITY,
    MEMORY_EVENT_PARSE_FAILURE,
    MEMORY_EVENT_INTAKE_MANIFEST,
    MEMORY_EVENT_LEGACY_IMPORT,
    MEMORY_EVENT_RESOLUTION,
})


@dataclass(frozen=True)
class MaterializedMemoryEvent:
    """带物理 event/object hash 的已核验 Memory 事件。"""

    event_hash: int
    object_hash: int
    event: MemoryEvent


class _MemoryEventIdentityCodec:
    """从正规化事件信封和单份 payload 恢复 object/event 完整稳定键。"""

    def __init__(self, registry: SpaceRegistry,
                 scoped_identities: ScopedIdentityStore) -> None:
        """绑定同一 backend 的空间、scope/timestamp 和事件物理记录。"""
        if registry.backend is not scoped_identities.backend:
            raise ValueError("Memory event identity codec backend 不一致")
        self.registry = registry
        self.scoped_identities = scoped_identities
        self.records = MemoryEventRecordStore(registry.backend)

    @property
    def backend(self):
        """返回 codec 绑定的存储后端。"""
        return self.registry.backend

    def event(self, event_hash: int) -> MemoryEvent:
        """从固定信封、payload chunk、scope 和 timestamp 恢复完整事件。"""
        record = self.records.read(event_hash)
        payload = payload_from_stable_key(
            record.event_kind,
            self.records.read_payload(event_hash),
        )
        scope = self.scoped_identities.load_scope(record.scope_hash)
        timestamp = self.scoped_identities.load_timestamp(
            record.timestamp_hash)
        memory_space = SpaceIdentity(*record.memory_space_key)
        if self.registry.identity(record.space_id) != memory_space:
            raise MemoryEventIntegrityError("Memory event 空间注册身份漂移")
        owner = OwnerScope(*record.owner_key)
        if scope.owner != owner:
            raise MemoryEventIntegrityError("Memory event scope owner 与信封漂移")
        if payload.timestamp() != timestamp:
            raise MemoryEventIntegrityError("Memory event payload timestamp 与信封漂移")
        if record.event_kind in _DECLARATION_EVENT_KINDS:
            object_key = declaration_object_key(record.event_kind, payload)
            object_ref = MemoryObjectRef(
                memory_space,
                owner,
                scope.versions,
                record.object_kind,
                object_key,
            )
        else:
            object_ref = payload.target_ref
        event = MemoryEvent(record.event_kind, object_ref, scope, payload)
        if (event.object_ref.memory_space != memory_space
                or event.object_ref.owner != owner
                or event.object_ref.object_kind != record.object_kind
                or event.timestamp.seq != record.event_seq):
            raise MemoryEventIntegrityError(
                "Memory event payload 恢复结果与固定信封不一致")
        return event

    def event_external_key(self, event_hash: int) -> ExternalIdentityKey | None:
        """为 identity registry 提供 self-headed 完整事件键和索引元数据。"""
        rows = self.backend.select(
            MEMORY_EVENT_TABLE, where={"event_hash": event_hash})
        if not rows:
            return None
        if len(rows) != 1:
            raise MemoryEventIntegrityError("event_hash 存在重复正规化信封")
        record = MemoryEventRecord.from_row(rows[0])
        event = self.event(event_hash)
        return ExternalIdentityKey(
            event.stable_key(),
            parent_hash=record.object_hash,
            ordinal=record.event_seq,
        )

    def object_external_key(self, object_hash: int) -> ExternalIdentityKey | None:
        """从唯一对象声明事件恢复 self-headed MemoryObjectRef 完整键。"""
        rows = tuple(
            record for record in self.records.rows_for_object(object_hash)
            if record.event_kind in _DECLARATION_EVENT_KINDS)
        if not rows:
            return None
        if len(rows) != 1:
            raise MemoryEventIntegrityError("Memory object 存在重复声明信封")
        event = self.event(rows[0].event_hash)
        return ExternalIdentityKey(
            event.object_ref.stable_key(),
            parent_hash=rows[0].scope_hash,
            ordinal=event.object_ref.object_kind,
        )


class MemoryEventLog:
    """一个 Memory 空间的 append-only 事件写入、读取和 ACL 查询入口。"""

    def __init__(self, registry: SpaceRegistry, backend,
                 memory_space_id: int,
                 scoped_identities: ScopedIdentityStore,
                 core_identities: CoreIdentityCatalog,
                 append_listener: Callable[[MaterializedMemoryEvent], None] | None = None,
                 ) -> None:
        """绑定 Memory 空间、共享 identity registry 和只读 Core catalog。"""
        if registry.backend is not backend:
            raise ValueError("MemoryEventLog registry 与 backend 不一致")
        if scoped_identities.backend is not backend:
            raise ValueError("MemoryEventLog identity store 与 backend 不一致")
        if core_identities.backend is not backend:
            raise ValueError("MemoryEventLog Core catalog 与 backend 不一致")
        memory_identity = registry.identity(memory_space_id)
        if memory_identity.space_type != SPACE_TYPE_MEMORY:
            raise ValueError("MemoryEventLog 必须绑定 Memory 空间")
        self.registry = registry
        self.backend = backend
        self.memory_space_id = memory_space_id
        self.memory_space_identity = memory_identity
        self.scoped_identities = scoped_identities
        self.core_identities = core_identities
        self._records = MemoryEventRecordStore(backend)
        self._append_listener = append_listener
        codec = getattr(
            scoped_identities, "_memory_event_identity_codec", None)
        if codec is None:
            codec = _MemoryEventIdentityCodec(registry, scoped_identities)
            scoped_identities._memory_event_identity_codec = codec
            scoped_identities.registry.register_external_key_resolver(
                IDENTITY_MEMORY_OBJECT,
                codec.object_external_key,
                self_headed=True,
            )
            scoped_identities.registry.register_external_key_resolver(
                IDENTITY_MEMORY_EVENT,
                codec.event_external_key,
                self_headed=True,
            )
        elif (not isinstance(codec, _MemoryEventIdentityCodec)
              or codec.backend is not backend
              or codec.scoped_identities is not scoped_identities):
            raise ValueError("ScopedIdentityStore 已绑定不兼容的 Memory event codec")
        self._identity_codec = codec

    def attach_append_listener(
            self,
            listener: Callable[[MaterializedMemoryEvent], None],
            ) -> None:
        """安装一个事件追加后的派生投影 listener，重复安装必须显式相同。"""
        if not callable(listener):
            raise TypeError("append listener 必须可调用")
        if (self._append_listener is not None
                and self._append_listener is not listener):
            raise ValueError("Memory event log 已绑定其他 append listener")
        self._append_listener = listener

    def append(self, event: MemoryEvent) -> MaterializedMemoryEvent:
        """追加一个完整事件，先核验所有 Core/Memory 引用和对象声明唯一性。"""
        self._validate_event(event)
        scope_hash = self.scoped_identities.register_scope(event.scope)
        timestamp_hash = self.scoped_identities.register_timestamp(
            event.timestamp)
        clock_hash = self.scoped_identities.register_clock(
            event.timestamp.clock)
        registry = self.scoped_identities.registry
        object_key = event.object_ref.stable_key()
        event_key = event.stable_key()
        object_hash = registry.identity_hash(
            IDENTITY_MEMORY_OBJECT, object_key)
        event_hash = registry.identity_hash(IDENTITY_MEMORY_EVENT, event_key)
        object_identity = registry.find(IDENTITY_MEMORY_OBJECT, object_key)
        object_rows = self._records.rows_for_object(object_hash)
        declarations = tuple(
            row for row in object_rows
            if row.event_kind in _DECLARATION_EVENT_KINDS)
        if event.is_declaration:
            self._validate_declaration_state(
                event, object_hash, object_identity, object_rows, declarations)
        else:
            if object_identity != object_hash or len(declarations) != 1:
                raise MemoryEventIntegrityError(
                    "状态事件目标没有唯一、完整的对象声明")
        self._validate_memory_references(event)

        existing_event = registry.find(IDENTITY_MEMORY_EVENT, event_key)
        event_rows = self.backend.select(
            MEMORY_EVENT_TABLE, where={"event_hash": event_hash})
        if event_rows or existing_event is not None:
            if existing_event != event_hash or len(event_rows) != 1:
                raise MemoryEventIntegrityError(
                    "Memory event 身份和物理行存在半写或重复")
            restored = self._restore(event_hash)
            if restored.event != event:
                raise MemoryEventIntegrityError(
                    "同一 Memory event identity 命中不同内容")
            return restored
        self._validate_evidence_consistency(event)
        self._validate_derivation_consistency(event)
        self._validate_transition_append(event, object_hash)

        record = self._record_for(
            event_hash,
            object_hash,
            event,
            scope_hash=scope_hash,
            timestamp_hash=timestamp_hash,
            clock_hash=clock_hash,
        )
        payload_key = event.payload.stable_key()

        def write_event(_: int) -> None:
            """把已计算身份对应的正规化信封和单份 payload 写入物理表。"""
            self._records.add(record, payload_key)

        if object_identity is None:
            registered = registry.register_resolved(
                IDENTITY_MEMORY_OBJECT,
                object_key,
                parent_hash=scope_hash,
                ordinal=event.object_ref.object_kind,
                writer=write_event,
            )
            if registered != object_hash:
                raise MemoryEventIntegrityError("Memory object hash 登记漂移")

            def reject_duplicate_writer(_: int) -> None:
                """对象 writer 已写同一事件，event resolver 仍缺失说明物理半写。"""
                raise MemoryEventIntegrityError(
                    "Memory object 已写但 event 外部身份仍不可恢复")

            event_writer = reject_duplicate_writer
        else:
            event_writer = write_event
        registered_event = registry.register_resolved(
            IDENTITY_MEMORY_EVENT,
            event_key,
            parent_hash=object_hash,
            ordinal=event.timestamp.seq,
            writer=event_writer,
        )
        if registered_event != event_hash:
            raise MemoryEventIntegrityError("Memory event hash 登记漂移")
        restored = self._restore(event_hash)
        if self._append_listener is not None:
            self._append_listener(restored)
        return restored

    def read(self, event_hash: int, *,
             access: MemoryAccessContext) -> MaterializedMemoryEvent | None:
        """按显式 ACL 读取事件；不可见事件返回空且不泄露内容。"""
        self._require_access(access)
        record = self._records.read(event_hash)
        from pure_integer_ai.cognition.shared.identity import OwnerScope

        if not access.can_read(OwnerScope(*record.owner_key)):
            return None
        return self._restore(event_hash)

    def query(self, *, access: MemoryAccessContext,
              event_kind: int | None = None,
              object_kind: int | None = None,
              object_ref: MemoryObjectRef | None = None,
              ) -> tuple[MaterializedMemoryEvent, ...]:
        """按空间、ACL 和可选事件/对象条件读取稳定有序事件。"""
        self._require_access(access)
        object_hash = None
        if object_ref is not None:
            if not isinstance(object_ref, MemoryObjectRef):
                raise TypeError("object_ref 必须是 MemoryObjectRef")
            if object_ref.memory_space != self.memory_space_identity:
                return ()
            object_hash = self.scoped_identities.registry.identity_hash(
                IDENTITY_MEMORY_OBJECT, object_ref.stable_key())
        from pure_integer_ai.cognition.shared.identity import OwnerScope

        result: list[MaterializedMemoryEvent] = []
        for record in self._records.query(
                space_id=self.memory_space_id,
                event_kind=event_kind,
                object_kind=object_kind,
                object_hash=object_hash):
            if not access.can_read(OwnerScope(*record.owner_key)):
                continue
            result.append(self._restore(record.event_hash))
        result.sort(key=lambda item: item.event.stable_key())
        return tuple(result)

    def clear_runtime_caches(self) -> None:
        """清空物理信封缓存；完整 identity 缓存由共享 store 统一管理。"""
        self._records.clear_runtime_caches()

    def _validate_declaration_state(
            self, event: MemoryEvent, object_hash: int,
            object_identity: int | None,
            object_rows: tuple[MemoryEventRecord, ...],
            declarations: tuple[MemoryEventRecord, ...]) -> None:
        """核验一个不可变对象至多有一条声明，且身份与物理行同时存在。"""
        if object_rows and object_identity is None:
            raise MemoryEventIntegrityError("Memory object 行存在但完整身份缺失")
        if object_identity is not None and object_identity != object_hash:
            raise MemoryEventIntegrityError("Memory object identity hash 漂移")
        if object_identity is not None and not declarations:
            raise MemoryEventIntegrityError("Memory object 身份存在但声明事件缺失")
        if len(declarations) > 1:
            raise MemoryEventIntegrityError("Memory object 存在重复声明事件")
        if declarations:
            declared = self._restore(declarations[0].event_hash)
            if declared.event != event:
                raise MemoryEventIntegrityError(
                    "同一 Memory object 被不同声明事件复用")

    def _validate_event(self, event: MemoryEvent) -> None:
        """核验事件属于当前 facade，且 scope/source/timestamp owner 不漂移。"""
        if not isinstance(event, MemoryEvent):
            raise TypeError("event 必须是 MemoryEvent")
        if event.object_ref.memory_space != self.memory_space_identity:
            raise ValueError("Memory event 稳定空间与 facade 不一致")
        payload_source = getattr(event.payload, "source", None)
        if payload_source is not None:
            access = MemoryAccessContext(
                event.object_ref.owner.tenant_id,
                event.object_ref.owner.user_id,
                event.object_ref.owner.session_id,
            )
            if not access.can_read(payload_source.owner):
                raise ValueError("Memory event 引用了 owner 不可见的来源")
        for ref in event.payload.core_refs():
            self.core_identities.identity_of(ref)

    def _validate_memory_references(self, event: MemoryEvent) -> None:
        """核验所有 Memory 引用已有唯一声明且对事件 owner 可见。"""
        access = MemoryAccessContext(
            event.object_ref.owner.tenant_id,
            event.object_ref.owner.user_id,
            event.object_ref.owner.session_id,
        )
        for ref in event.payload.memory_refs():
            if ref == event.object_ref and not event.is_declaration:
                continue
            if not access.can_read(ref.owner):
                raise MemoryEventIntegrityError(
                    "Memory event 引用了 owner 不可见的对象")
            self._require_declared_object(ref)

    def _require_declared_object(self, ref: MemoryObjectRef) -> None:
        """跨 Memory 空间核验对象完整身份、唯一声明和稳定空间注册行。"""
        registry = self.scoped_identities.registry
        object_hash = registry.find(
            IDENTITY_MEMORY_OBJECT, ref.stable_key())
        if object_hash is None:
            raise MemoryEventIntegrityError("Memory 引用目标 identity 不存在")
        rows = self._records.rows_for_object(object_hash)
        declarations = tuple(
            row for row in rows
            if row.event_kind in _DECLARATION_EVENT_KINDS)
        if len(declarations) != 1:
            raise MemoryEventIntegrityError("Memory 引用目标没有唯一声明事件")
        row = declarations[0]
        if self.registry.identity(row.space_id) != ref.memory_space:
            raise MemoryEventIntegrityError("Memory 引用目标空间身份漂移")
        restored_key = registry.read_key(
            IDENTITY_MEMORY_OBJECT, object_hash)
        if MemoryObjectRef.from_stable_key(restored_key) != ref:
            raise MemoryEventIntegrityError("Memory 引用目标完整键漂移")

    def _declaration_event(self, ref: MemoryObjectRef) -> MemoryEvent:
        """不经递归 facade 恢复某对象的唯一声明事件。"""
        registry = self.scoped_identities.registry
        object_hash = registry.find(
            IDENTITY_MEMORY_OBJECT, ref.stable_key())
        if object_hash is None:
            raise MemoryEventIntegrityError("Memory 对象 identity 不存在")
        rows = tuple(
            row for row in self._records.rows_for_object(object_hash)
            if row.event_kind in _DECLARATION_EVENT_KINDS)
        if len(rows) != 1:
            raise MemoryEventIntegrityError("Memory 对象没有唯一声明事件")
        event = MemoryEvent.from_stable_key(registry.read_key(
            IDENTITY_MEMORY_EVENT, rows[0].event_hash))
        if event.object_ref != ref:
            raise MemoryEventIntegrityError("Memory 对象声明完整键漂移")
        return event

    def _validate_evidence_consistency(self, event: MemoryEvent) -> None:
        """核验 Evidence supersede 同候选、同时钟、非倒退且没有竞争替代。"""
        if not isinstance(event.payload, EvidencePayload):
            return
        payload = event.payload
        old_ref = payload.supersedes_ref
        if old_ref is None:
            return
        old_event = self._declaration_event(old_ref)
        if not isinstance(old_event.payload, EvidencePayload):
            raise MemoryEventIntegrityError(
                "Evidence supersede 目标不是 Evidence payload")
        old = old_event.payload
        if old.hypothesis_ref != payload.hypothesis_ref:
            raise MemoryEventIntegrityError(
                "Evidence 不得 supersede 其他 Hypothesis 的证据")
        if (old.observed_at.clock != payload.observed_at.clock
                or payload.observed_at.seq < old.observed_at.seq):
            raise MemoryEventIntegrityError(
                "Evidence supersede 时钟漂移或逻辑序倒退")
        superseders: list[MemoryObjectRef] = []
        registry = self.scoped_identities.registry
        for row in self.backend.select(
                MEMORY_EVENT_TABLE,
                where={"event_kind": MEMORY_EVENT_EVIDENCE}):
            candidate = MemoryEvent.from_stable_key(registry.read_key(
                IDENTITY_MEMORY_EVENT, row["event_hash"]))
            candidate_payload = candidate.payload
            if (isinstance(candidate_payload, EvidencePayload)
                    and candidate_payload.supersedes_ref == old_ref):
                superseders.append(candidate.object_ref)
        if superseders and event.object_ref not in superseders:
            raise MemoryEventIntegrityError(
                "同一 Evidence 已被其他事件替代")
        if len(set(superseders)) != len(superseders) or len(superseders) > 1:
            raise MemoryEventIntegrityError(
                "同一 Evidence 存在竞争 supersede 事件")

    def _validate_derivation_consistency(self, event: MemoryEvent) -> None:
        """核验 parser derivation 只能按两个 manifest 的同 lineage 绑定替代。"""
        payload = event.payload
        if not isinstance(payload, DerivationTransitionPayload):
            return
        prior_manifest = self._intake_manifest(payload.prior_source)
        replacement_manifest = self._intake_manifest(
            payload.replacement_source)
        prior_ref = prior_manifest[0]
        replacement_ref = replacement_manifest[0]
        if replacement_manifest[1].supersedes_manifest_ref != prior_ref:
            raise MemoryEventIntegrityError(
                "新 intake manifest 未显式指向 derivation 前驱")

        if payload.binding_kind == INTAKE_DERIVED_MANIFEST:
            if (event.object_ref != prior_ref
                    or payload.replacement_ref != replacement_ref
                    or payload.lineage_key
                    != source_reparse_lineage_key(payload.prior_source)):
                raise MemoryEventIntegrityError(
                    "manifest derivation 的目标、replacement 或谱系不一致")
            return

        old_bindings = tuple(
            item for item in prior_manifest[1].bindings
            if (item.binding_kind == payload.binding_kind
                and item.lineage_key == payload.lineage_key)
        )
        if (len(old_bindings) != 1
                or old_bindings[0].object_ref != payload.target_ref):
            raise MemoryEventIntegrityError(
                "derivation target 未由旧 manifest 的唯一 lineage 声明")
        if self._source_for_derived(payload.target_ref) != payload.prior_source:
            raise MemoryEventIntegrityError(
                "derivation target 声明的来源与旧 manifest 不一致")

        new_bindings = tuple(
            item for item in replacement_manifest[1].bindings
            if (item.binding_kind == payload.binding_kind
                and item.lineage_key == payload.lineage_key)
        )
        if payload.replacement_ref is None:
            if new_bindings:
                raise MemoryEventIntegrityError(
                    "归档 derivation 不得遗漏已有同 lineage replacement")
            return
        if (len(new_bindings) != 1
                or new_bindings[0].object_ref != payload.replacement_ref
                or self._source_for_derived(payload.replacement_ref)
                != payload.replacement_source):
            raise MemoryEventIntegrityError(
                "superseded derivation 未指向新 manifest 的唯一同 lineage 对象")

    def _intake_manifest(
            self, source,
            ) -> tuple[MemoryObjectRef, IntakeManifestPayload]:
        """按确定性对象键恢复当前 Memory 空间某来源的唯一 intake manifest。"""
        ref = MemoryObjectRef(
            self.memory_space_identity,
            source.owner,
            source.versions,
            MEMORY_OBJECT_INTAKE_MANIFEST,
            source.stable_key(),
        )
        try:
            event = self._declaration_event(ref)
        except MemoryEventIntegrityError as exc:
            raise MemoryEventIntegrityError(
                "derivation 前后来源必须各有唯一 intake manifest") from exc
        if (not isinstance(event.payload, IntakeManifestPayload)
                or event.payload.source != source):
            raise MemoryEventIntegrityError(
                "intake manifest 声明来源或 payload 漂移")
        return ref, event.payload

    def _source_for_derived(self, ref: MemoryObjectRef):
        """从 M-05 派生对象声明中恢复其精确 SourceRef。"""
        payload = self._declaration_event(ref).payload
        if isinstance(payload, ObservationPayload):
            return payload.source
        if isinstance(payload, HypothesisPayload):
            return payload.hypothesis.observation
        if isinstance(payload, EvidencePayload):
            if payload.source is None:
                raise MemoryEventIntegrityError(
                    "M-05 derivation 不接受只能沿 Episode 回溯的 Evidence")
            return payload.source
        if isinstance(payload, ParseFailurePayload):
            return payload.source
        raise MemoryEventIntegrityError("derivation target 不是 M-05 派生对象")

    def _validate_transition_append(
            self, event: MemoryEvent, object_hash: int) -> None:
        """写前核验 retention/lifecycle 当前态、时钟和单向事件链。"""
        retention, lifecycle, retention_clock, lifecycle_clock = (
            self._transition_state(object_hash))
        if isinstance(event.payload, RetentionTransitionPayload):
            payload = event.payload
            if payload.from_state != retention:
                raise MemoryEventIntegrityError(
                    "retention from_state 与当前事件历史不一致")
            if retention_clock is not None:
                raise MemoryEventIntegrityError("对象已经存在 retention 转换")
        elif isinstance(event.payload, (
                LifecycleTransitionPayload,
                DerivationTransitionPayload,
        )):
            payload = event.payload
            if payload.from_state != lifecycle:
                raise MemoryEventIntegrityError(
                    "lifecycle from_state 与当前事件历史不一致")
            if lifecycle_clock is not None:
                prior_clock, prior_seq = lifecycle_clock
                if payload.changed_at.clock != prior_clock:
                    raise MemoryEventIntegrityError(
                        "同一对象 lifecycle 不得跨 clock identity 排序")
                if payload.changed_at.seq <= prior_seq:
                    raise MemoryEventIntegrityError(
                        "lifecycle 逻辑序必须严格前进")

    def _transition_state(
            self, object_hash: int,
            ) -> tuple[int, int, tuple | None, tuple | None]:
        """从 append-only 状态事件重放当前双轴，仅用于写入完整性而非 M-04 聚合。"""
        retention = RETENTION_EPISODIC
        lifecycle = LIFECYCLE_ACTIVE
        retention_clock = None
        lifecycle_clock = None
        registry = self.scoped_identities.registry
        rows = self._records.rows_for_object(object_hash)
        for row in rows:
            if row.event_kind not in {
                    MEMORY_EVENT_RETENTION,
                    MEMORY_EVENT_LIFECYCLE,
                    MEMORY_EVENT_DERIVATION,
            }:
                continue
            event = MemoryEvent.from_stable_key(registry.read_key(
                IDENTITY_MEMORY_EVENT, row.event_hash))
            event_object_hash = registry.identity_hash(
                IDENTITY_MEMORY_OBJECT, event.object_ref.stable_key())
            if event_object_hash != object_hash:
                raise MemoryEventIntegrityError(
                    "状态事件信封 object_hash 与完整事件目标不一致")
            if isinstance(event.payload, RetentionTransitionPayload):
                payload = event.payload
                if retention_clock is not None:
                    raise MemoryEventIntegrityError(
                        "同一对象存在多条 retention 转换")
                if (payload.from_state != retention
                        or payload.to_state != RETENTION_CONSOLIDATED):
                    raise MemoryEventIntegrityError(
                        "retention 事件历史状态不连续")
                retention = payload.to_state
                retention_clock = (
                    payload.changed_at.clock, payload.changed_at.seq)
                continue
            if not isinstance(event.payload, (
                    LifecycleTransitionPayload,
                    DerivationTransitionPayload,
            )):
                raise MemoryEventIntegrityError(
                    "状态事件 kind 与 payload 类型不一致")
            payload = event.payload
            if payload.from_state != lifecycle:
                raise MemoryEventIntegrityError(
                    "lifecycle 事件历史状态不连续")
            if lifecycle_clock is not None:
                prior_clock, prior_seq = lifecycle_clock
                if (payload.changed_at.clock != prior_clock
                        or payload.changed_at.seq <= prior_seq):
                    raise MemoryEventIntegrityError(
                        "lifecycle 事件时钟漂移或逻辑序未前进")
            lifecycle = payload.to_state
            lifecycle_clock = (
                payload.changed_at.clock, payload.changed_at.seq)
        return retention, lifecycle, retention_clock, lifecycle_clock

    def _record_for(self, event_hash: int, object_hash: int,
                    event: MemoryEvent, *, scope_hash: int,
                    timestamp_hash: int, clock_hash: int
                    ) -> MemoryEventRecord:
        """把领域事件投影成固定查询信封和单一时间轴。"""
        seq = event.timestamp.seq
        created = seq if event.time_axis == TIME_AXIS_CREATED else 0
        observed = seq if event.time_axis == TIME_AXIS_OBSERVED else 0
        used = seq if event.time_axis == TIME_AXIS_USED else 0
        return MemoryEventRecord(
            event_hash,
            object_hash,
            self.memory_space_id,
            event.object_ref.memory_space.stable_key(),
            event.object_ref.owner.stable_key(),
            event.event_kind,
            event.object_ref.object_kind,
            len(event.payload.stable_key()),
            scope_hash,
            timestamp_hash,
            clock_hash,
            seq,
            created,
            observed,
            used,
        )

    def _restore(self, event_hash: int) -> MaterializedMemoryEvent:
        """从事件、对象、scope 和 timestamp 四类身份双向核验后恢复对象。"""
        record = self._records.read(event_hash)
        registry = self.scoped_identities.registry
        event_key = registry.read_key(IDENTITY_MEMORY_EVENT, event_hash)
        event = MemoryEvent.from_stable_key(event_key)
        object_key = registry.read_key(
            IDENTITY_MEMORY_OBJECT, record.object_hash)
        if MemoryObjectRef.from_stable_key(object_key) != event.object_ref:
            raise MemoryEventIntegrityError("Memory event 对象身份与信封不一致")
        if self.scoped_identities.load_scope(record.scope_hash) != event.scope:
            raise MemoryEventIntegrityError("Memory event scope hash 漂移")
        if self.scoped_identities.load_timestamp(
                record.timestamp_hash) != event.timestamp:
            raise MemoryEventIntegrityError("Memory event timestamp hash 漂移")
        if self.scoped_identities.load_clock(
                record.clock_hash) != event.timestamp.clock:
            raise MemoryEventIntegrityError("Memory event clock hash 漂移")
        expected = self._record_for(
            event_hash,
            record.object_hash,
            event,
            scope_hash=record.scope_hash,
            timestamp_hash=record.timestamp_hash,
            clock_hash=record.clock_hash,
        )
        if expected != record:
            raise MemoryEventIntegrityError("Memory event 信封与完整事件键不一致")
        self._transition_state(record.object_hash)
        self._validate_event(event)
        self._validate_memory_references(event)
        self._validate_evidence_consistency(event)
        return MaterializedMemoryEvent(event_hash, record.object_hash, event)

    @staticmethod
    def _require_access(access: MemoryAccessContext) -> None:
        """拒绝省略或伪造 ACL 查询上下文。"""
        if not isinstance(access, MemoryAccessContext):
            raise MemoryOverlayQueryError(
                "Memory event 查询必须显式提供 MemoryAccessContext")


__all__ = [
    "MaterializedMemoryEvent",
    "MemoryEventLog",
]
