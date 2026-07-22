"""Memory overlay 的 Core identity、关系所有权和可见性协议。

Memory 关系可以引用多个 Core 空间中的既有 typed identity，但不能通过进入 Memory
而物化或复制 Core 对象。关系自身拥有独立的 Memory 稳定空间、owner、scope 和完整
整数身份；查询必须携带显式访问上下文，不能用缺省值扩大可见范围。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    OBJECT_CONCEPT,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    TypedRef,
    VISIBILITY_GLOBAL,
    VISIBILITY_SESSION,
    VISIBILITY_TENANT,
    VISIBILITY_USER,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.scoped_persistence import (
    ScopedIdentityStore,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_MEMORY_OVERLAY,
)
from pure_integer_ai.storage.memory_overlay import (
    MEMORY_OVERLAY_TABLE,
    MemoryOverlayIntegrityError,
    MemoryOverlayRecord,
    MemoryOverlayRecordStore,
)
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_CORE,
    SPACE_TYPE_MEMORY,
    SpaceIdentity,
    SpaceRegistry,
)


OVERLAY_KEY_VERSION = 1
TYPED_REF_KEY_SIZE = 11


class CoreIdentityBoundaryError(RuntimeError):
    """Memory 试图使用未注册或非 Core typed identity。"""


class MemoryOverlayQueryError(RuntimeError):
    """Memory overlay 查询上下文或过滤条件非法。"""


def _strict_key(key: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """校验稳定键是非空严格整数元组。"""
    if not isinstance(key, tuple) or not key:
        raise ValueError(f"{where} 必须是非空整数元组")
    assert_int(*key, _where=where)
    if any(type(value) is not int for value in key):
        raise ValueError(f"{where} 必须使用严格整数")
    return key


def _take(key: tuple[int, ...], cursor: int, size: int,
          *, where: str) -> tuple[tuple[int, ...], int]:
    """从稳定键取固定长度片段，并在截断时 fail closed。"""
    end = cursor + size
    if cursor < 0 or end > len(key):
        raise ValueError(f"{where} 稳定键被截断")
    return key[cursor:end], end


@dataclass(frozen=True)
class MemoryAccessContext:
    """查询方的 tenant/user/session 身份，不携带可提升权限的 visibility。"""

    tenant_id: int = 0
    user_id: int = 0
    session_id: int = 0

    def __post_init__(self) -> None:
        """校验访问身份的层级关系，零值只表示未进入对应层级。"""
        assert_int(
            self.tenant_id, self.user_id, self.session_id,
            _where="MemoryAccessContext",
        )
        if any(type(value) is not int for value in self.stable_key()):
            raise ValueError("MemoryAccessContext 必须使用严格整数")
        if min(self.stable_key()) < 0:
            raise ValueError("MemoryAccessContext 身份不得为负数")
        if self.user_id and not self.tenant_id:
            raise ValueError("访问 user 必须同时携带 tenant")
        if self.session_id and (not self.tenant_id or not self.user_id):
            raise ValueError("访问 session 必须同时携带 tenant/user")

    def stable_key(self) -> tuple[int, int, int]:
        """返回访问上下文的稳定整数键。"""
        return self.tenant_id, self.user_id, self.session_id

    def can_read(self, owner: OwnerScope) -> bool:
        """判断当前访问身份是否可读取一个 owner scope 的关系。"""
        if owner.visibility == VISIBILITY_GLOBAL:
            return True
        if owner.visibility == VISIBILITY_TENANT:
            return self.tenant_id == owner.tenant_id and self.tenant_id > 0
        if owner.visibility == VISIBILITY_USER:
            return (
                self.tenant_id == owner.tenant_id
                and self.user_id == owner.user_id
                and self.user_id > 0
            )
        if owner.visibility == VISIBILITY_SESSION:
            return (
                self.tenant_id == owner.tenant_id
                and self.user_id == owner.user_id
                and self.session_id == owner.session_id
                and self.session_id > 0
            )
        raise MemoryOverlayQueryError("owner visibility 未注册")


@dataclass(frozen=True)
class MemoryOverlayRelation:
    """引用既有 Core typed identity 的 Memory-local overlay 关系。"""

    memory_space: SpaceIdentity
    owner: OwnerScope
    versions: VersionBundle
    predicate: TypedRef
    subject: TypedRef
    object_ref: TypedRef
    scope: ScopeIdentity
    provenance_kind: int
    epistemic_origin: int = 0
    content_version: int = 0
    qualifiers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """校验 Memory 空间、关系 owner、scope 和三个 Core 端点的类型契约。"""
        if not isinstance(self.memory_space, SpaceIdentity):
            raise TypeError("memory_space 必须是 SpaceIdentity")
        if self.memory_space.space_type != SPACE_TYPE_MEMORY:
            raise ValueError("MemoryOverlayRelation 必须绑定 Memory 空间")
        if not isinstance(self.owner, OwnerScope):
            raise TypeError("owner 必须是 OwnerScope")
        if not isinstance(self.versions, VersionBundle):
            raise TypeError("versions 必须是 VersionBundle")
        if not isinstance(self.predicate, TypedRef):
            raise TypeError("predicate 必须是 TypedRef")
        if self.predicate.object_kind != OBJECT_CONCEPT:
            raise ValueError("overlay predicate 必须是 Core 概念 typed ref")
        if not isinstance(self.subject, TypedRef):
            raise TypeError("subject 必须是 TypedRef")
        if not isinstance(self.object_ref, TypedRef):
            raise TypeError("object_ref 必须是 TypedRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        if self.scope.owner != self.owner:
            raise ValueError("overlay relation owner 与 scope owner 不一致")
        if self.scope.versions != self.versions:
            raise ValueError("overlay relation versions 与 scope 不一致")
        if not isinstance(self.qualifiers, tuple):
            raise TypeError("overlay qualifiers 必须是整数元组")
        assert_int(
            self.provenance_kind, self.epistemic_origin, self.content_version,
            *self.qualifiers,
            _where="MemoryOverlayRelation",
        )
        if self.provenance_kind <= 0:
            raise ValueError("provenance_kind 必须为正整数")
        if self.epistemic_origin < 0 or self.content_version < 0:
            raise ValueError("overlay relation 版本字段不得为负数")
        if any(type(value) is not int for value in self.qualifiers):
            raise ValueError("overlay qualifiers 必须使用严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回包含 Memory 空间、owner、端点、scope 和限定项的完整身份键。"""
        scope_key = self.scope.stable_key()
        return (
            OVERLAY_KEY_VERSION,
            *self.memory_space.stable_key(),
            *self.owner.stable_key(),
            *self.versions.stable_key(),
            *self.predicate.stable_key(),
            *self.subject.stable_key(),
            *self.object_ref.stable_key(),
            len(scope_key),
            *scope_key,
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            len(self.qualifiers),
            *self.qualifiers,
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "MemoryOverlayRelation":
        """从完整关系身份键恢复对象，拒绝截断、尾随和版本漂移。"""
        key = _strict_key(key, where="MemoryOverlayRelation.stable_key")
        if len(key) < 1 + 3 + 4 + 4 + 3 * TYPED_REF_KEY_SIZE + 1 + 4:
            raise ValueError("MemoryOverlayRelation 稳定键长度不足")
        if key[0] != OVERLAY_KEY_VERSION:
            raise ValueError("MemoryOverlayRelation 稳定键版本未注册")
        cursor = 1
        memory_key, cursor = _take(key, cursor, 3, where="memory_space")
        owner_key, cursor = _take(key, cursor, 4, where="owner")
        version_key, cursor = _take(key, cursor, 4, where="versions")
        predicate_key, cursor = _take(key, cursor, 11, where="predicate")
        subject_key, cursor = _take(key, cursor, 11, where="subject")
        object_key, cursor = _take(key, cursor, 11, where="object")
        if cursor >= len(key):
            raise ValueError("overlay scope 长度缺失")
        scope_size = key[cursor]
        cursor += 1
        if scope_size <= 0:
            raise ValueError("overlay scope 长度非法")
        scope_key, cursor = _take(key, cursor, scope_size, where="scope")
        if cursor + 4 > len(key):
            raise ValueError("overlay provenance 字段被截断")
        provenance_kind, epistemic_origin, content_version = key[cursor:cursor + 3]
        cursor += 3
        qualifier_size = key[cursor]
        cursor += 1
        if qualifier_size < 0:
            raise ValueError("overlay qualifier 长度非法")
        qualifiers, cursor = _take(
            key, cursor, qualifier_size, where="qualifiers")
        if cursor != len(key):
            raise ValueError("overlay 稳定键存在尾随字段")
        return cls(
            SpaceIdentity(*memory_key),
            OwnerScope(*owner_key),
            VersionBundle(
                CorpusVersion(version_key[0]),
                ParserVersion(version_key[1]),
                PrimitiveVersion(version_key[2]),
                CurriculumVersion(version_key[3]),
            ),
            TypedRef.from_stable_key(predicate_key),
            TypedRef.from_stable_key(subject_key),
            TypedRef.from_stable_key(object_key),
            ScopeIdentity.from_stable_key(scope_key),
            provenance_kind,
            epistemic_origin,
            content_version,
            qualifiers,
        )


@dataclass(frozen=True)
class MaterializedMemoryOverlay:
    """带物理身份 hash 的可见 Memory overlay 关系。"""

    identity_hash: int
    relation: MemoryOverlayRelation


class CoreIdentityCatalog:
    """跨多个 Core GraphOntology 只读解析 typed identity，不提供物化能力。"""

    def __init__(self, ontologies: Iterable[GraphOntology]) -> None:
        """登记同一 backend 上的 Core 空间，并拒绝 Memory/Companion 混入。"""
        items = tuple(ontologies)
        if not items:
            raise ValueError("CoreIdentityCatalog 至少需要一个 Core ontology")
        self._backend = items[0].backend
        self._by_space: dict[int, GraphOntology] = {}
        for ontology in items:
            if ontology.backend is not self._backend:
                raise ValueError("CoreIdentityCatalog 不得跨 backend 拼接")
            if ontology.space_identity.space_type != SPACE_TYPE_CORE:
                raise CoreIdentityBoundaryError("CoreIdentityCatalog 拒绝非 Core 空间")
            if ontology.space_id in self._by_space:
                raise ValueError("CoreIdentityCatalog 存在重复运行空间")
            self._by_space[ontology.space_id] = ontology

    @property
    def backend(self):
        """返回 Core identity catalog 绑定的后端。"""
        return self._backend

    def identity_of(self, ref: TypedRef):
        """只读恢复 Core ref 的权威 ObjectIdentity，不在缺失时自动物化。"""
        if not isinstance(ref, TypedRef):
            raise TypeError("Core ref 必须是 TypedRef")
        ontology = self._by_space.get(ref.space_id)
        if ontology is None:
            raise CoreIdentityBoundaryError(
                f"typed ref space_id={ref.space_id} 不属于已登记 Core")
        return ontology.identity_of(ref)

    def identity_hash_of(self, ref: TypedRef) -> int:
        """读取 Core ref 的稳定身份 hash，仍不创建任何图对象。"""
        ontology = self._by_space.get(ref.space_id)
        if ontology is None:
            raise CoreIdentityBoundaryError("typed ref 不属于已登记 Core")
        return ontology.identity_hash_of(ref)


class MemoryOverlay:
    """Memory overlay 的唯一写入和带 ACL 查询 facade。"""

    def __init__(self, registry: SpaceRegistry, backend,
                 memory_space_id: int,
                 scoped_identities: ScopedIdentityStore,
                 core_identities: CoreIdentityCatalog) -> None:
        """绑定一个 Memory 空间、scope registry 和只读 Core identity catalog。"""
        if registry.backend is not backend:
            raise ValueError("MemoryOverlay registry 与 backend 不一致")
        if scoped_identities.backend is not backend:
            raise ValueError("MemoryOverlay scope registry 与 backend 不一致")
        if core_identities.backend is not backend:
            raise ValueError("MemoryOverlay Core catalog 与 backend 不一致")
        memory_identity = registry.identity(memory_space_id)
        if memory_identity.space_type != SPACE_TYPE_MEMORY:
            raise ValueError("MemoryOverlay 必须绑定 Memory 空间")
        self.registry = registry
        self.backend = backend
        self.memory_space_id = memory_space_id
        self.memory_space_identity = memory_identity
        self.scoped_identities = scoped_identities
        self.core_identities = core_identities
        self._records = MemoryOverlayRecordStore(backend)

    def add(self, relation: MemoryOverlayRelation) -> MaterializedMemoryOverlay:
        """追加一条只引用 Core 的 overlay 关系，不写入 Core 图。"""
        self._validate_relation(relation)
        scope_hash = self.scoped_identities.register_scope(relation.scope)
        stable_key = relation.stable_key()
        identity_hash = self.scoped_identities.registry.identity_hash(
            IDENTITY_MEMORY_OVERLAY, stable_key)
        existing_rows = self.backend.select(
            MEMORY_OVERLAY_TABLE,
            where={"identity_hash": identity_hash},
        )
        existing_hash = self.scoped_identities.registry.find(
            IDENTITY_MEMORY_OVERLAY, stable_key)
        if existing_rows and existing_hash is None:
            raise MemoryOverlayIntegrityError(
                "overlay 物理行存在但完整 identity 缺失")
        if existing_hash is not None:
            restored = self._restore(existing_hash)
            if restored.relation != relation:
                raise MemoryOverlayIntegrityError(
                    "同一 overlay identity 命中不同关系内容")
            return restored
        self.scoped_identities.registry.register(
            IDENTITY_MEMORY_OVERLAY,
            stable_key,
            parent_hash=scope_hash,
        )
        record = self._record_for(
            identity_hash, relation, scope_hash=scope_hash)
        self._records.add(record)
        return self._restore(identity_hash)

    def read(self, identity_hash: int, *,
             access: MemoryAccessContext) -> MaterializedMemoryOverlay | None:
        """按显式访问上下文读取关系；不可见关系返回空而不泄露存在性。"""
        self._require_access(access)
        record = self._records.read(identity_hash)
        owner = OwnerScope(*record.owner_key)
        if not access.can_read(owner):
            return None
        return self._restore(record.identity_hash)

    def query(self, *, access: MemoryAccessContext,
              predicate: TypedRef | None = None,
              subject: TypedRef | None = None,
              object_ref: TypedRef | None = None,
              ) -> tuple[MaterializedMemoryOverlay, ...]:
        """按 owner ACL 和可选完整端点查询，缺访问上下文不得扩大可见范围。"""
        self._require_access(access)
        for ref in (predicate, subject, object_ref):
            if ref is not None:
                self.core_identities.identity_of(ref)
        result: list[MaterializedMemoryOverlay] = []
        for record in self._records.query(space_id=self.memory_space_id):
            if not access.can_read(OwnerScope(*record.owner_key)):
                continue
            entry = self._restore(record.identity_hash)
            relation = entry.relation
            if predicate is not None and relation.predicate != predicate:
                continue
            if subject is not None and relation.subject != subject:
                continue
            if object_ref is not None and relation.object_ref != object_ref:
                continue
            result.append(entry)
        result.sort(key=lambda item: item.relation.stable_key())
        return tuple(result)

    def clear_runtime_caches(self) -> None:
        """清空 overlay 物理缓存和 scope 之外的运行期对象，供恢复核验使用。"""
        self._records.clear_runtime_caches()

    def _validate_relation(self, relation: MemoryOverlayRelation) -> None:
        """核验 Memory 空间和三个端点确实是已存在的 Core identity。"""
        if not isinstance(relation, MemoryOverlayRelation):
            raise TypeError("relation 必须是 MemoryOverlayRelation")
        if relation.memory_space != self.memory_space_identity:
            raise ValueError("relation Memory 稳定空间与 facade 不一致")
        for ref in (relation.predicate, relation.subject, relation.object_ref):
            self.core_identities.identity_of(ref)

    def _record_for(self, identity_hash: int,
                    relation: MemoryOverlayRelation, *,
                    scope_hash: int) -> MemoryOverlayRecord:
        """把领域关系投影成可按 Memory 空间过滤的固定物理记录。"""
        return MemoryOverlayRecord(
            identity_hash,
            self.memory_space_id,
            relation.memory_space.stable_key(),
            relation.owner.stable_key(),
            relation.versions.stable_key(),
            relation.predicate.stable_key(),
            relation.subject.stable_key(),
            relation.object_ref.stable_key(),
            scope_hash,
            relation.provenance_kind,
            relation.epistemic_origin,
            relation.content_version,
        )

    def _restore(self, identity_hash: int) -> MaterializedMemoryOverlay:
        """从 identity registry 和 overlay 行双向核验并恢复完整关系。"""
        record = self._records.read(identity_hash)
        stable_key = self.scoped_identities.registry.read_key(
            IDENTITY_MEMORY_OVERLAY, identity_hash)
        relation = MemoryOverlayRelation.from_stable_key(stable_key)
        restored_scope = self.scoped_identities.load_scope(record.scope_hash)
        if restored_scope != relation.scope:
            raise MemoryOverlayIntegrityError("overlay scope hash 与关系身份不一致")
        expected = self._record_for(
            identity_hash, relation, scope_hash=record.scope_hash)
        if expected != record:
            raise MemoryOverlayIntegrityError("overlay 主记录与完整身份键不一致")
        self._validate_relation(relation)
        return MaterializedMemoryOverlay(identity_hash, relation)

    @staticmethod
    def _require_access(access: MemoryAccessContext) -> None:
        """拒绝省略、空值或错误类型的 ACL 查询上下文。"""
        if not isinstance(access, MemoryAccessContext):
            raise MemoryOverlayQueryError(
                "Memory overlay 查询必须显式提供 MemoryAccessContext")


__all__ = [
    "CoreIdentityBoundaryError",
    "CoreIdentityCatalog",
    "MaterializedMemoryOverlay",
    "MemoryAccessContext",
    "MemoryOverlay",
    "MemoryOverlayQueryError",
    "MemoryOverlayRelation",
]
