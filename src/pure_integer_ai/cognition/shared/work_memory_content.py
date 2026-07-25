"""A-02 工作记忆的来源化、有界内容状态。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_OCCURRENCE,
    OBJECT_ROLE,
    ObjectIdentity,
    SourceRef,
    TypedRef,
    VersionBundle,
    object_contracts_by_kind,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    SCOPE_EPISODE,
    SCOPE_GENERATION,
    SCOPE_QUERY,
    SCOPE_SESSION,
    ScopeIdentity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    SEMANTIC_OBJECT_KINDS,
    semantic_source,
    validate_semantic_identity,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_CONTENT_VERSION = 1
_ITEM_REFERENCE_DOMAIN = "work_memory.content_item.v1"
_SOURCE_KEY_SIZE = len(SourceRef(
    1, 1, 0, GLOBAL_OWNER_SCOPE, VersionBundle()).stable_key())
_LIFESPAN_SCOPE_KINDS = frozenset({
    SCOPE_SESSION,
    SCOPE_DOCUMENT,
    SCOPE_EPISODE,
    SCOPE_QUERY,
    SCOPE_GENERATION,
})
_PARENT_SCOPE_KIND = {
    SCOPE_EPISODE: SCOPE_DOCUMENT,
    SCOPE_QUERY: SCOPE_EPISODE,
    SCOPE_GENERATION: SCOPE_QUERY,
}


class WorkMemoryContentError(ValueError):
    """工作记忆内容违反角色、容量、来源或生命周期契约时抛出。"""


def _strict_positive(value: int, *, where: str) -> int:
    """校验协议容量是严格正整数。"""
    assert_int(value, _where=where)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{where} 必须是严格正整数")
    return value


def _strict_nonnegative(value: int, *, where: str) -> int:
    """校验无墙钟逻辑序是严格非负整数。"""
    assert_int(value, _where=where)
    if type(value) is not int or value < 0:
        raise ValueError(f"{where} 必须是严格非负整数")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """校验开放协议键或 trace 是非空严格整数元组。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键添加长度边界。"""
    return len(value), *value


def _descends_from(scope: ScopeIdentity, ancestor: ScopeIdentity) -> bool:
    """判断一个运行 scope 是否等于或递归位于指定祖先之下。"""
    current: ScopeIdentity | None = scope
    while current is not None:
        if current == ancestor:
            return True
        current = current.parent
    return False


@dataclass(frozen=True)
class WorkMemoryRoleDefinition:
    """注册一个开放 Role 的允许对象种类、寿命和 active 硬上限。"""

    role: ObjectIdentity
    allowed_object_kinds: tuple[int, ...]
    lifespan_scope_kind: int
    max_active: int

    def __post_init__(self) -> None:
        if (not isinstance(self.role, ObjectIdentity)
                or self.role.object_kind != OBJECT_ROLE):
            raise TypeError("WorkMemory Role 必须是一等 ObjectIdentity Role")
        validate_semantic_identity(self.role)
        if not isinstance(self.allowed_object_kinds, tuple) or not (
                self.allowed_object_kinds):
            raise ValueError("WorkMemory Role 必须声明允许的对象种类")
        assert_int(
            *self.allowed_object_kinds,
            _where="WorkMemoryRoleDefinition.allowed_object_kinds",
        )
        if (any(type(item) is not int or item <= 0
                for item in self.allowed_object_kinds)
                or len(set(self.allowed_object_kinds))
                != len(self.allowed_object_kinds)):
            raise ValueError("WorkMemory Role 对象种类必须是唯一正整数")
        contracts = object_contracts_by_kind()
        if any(
                item not in contracts
                or not contracts[item].authoritative_identity
                for item in self.allowed_object_kinds):
            raise ValueError("WorkMemory Role 只允许权威对象身份种类")
        if self.lifespan_scope_kind not in _LIFESPAN_SCOPE_KINDS:
            raise ValueError("WorkMemory Role lifespan 不是 A-09 活动 scope")
        _strict_positive(self.max_active, where="WorkMemory Role max_active")
        object.__setattr__(
            self, "allowed_object_kinds",
            tuple(sorted(self.allowed_object_kinds)),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 Role 身份、允许对象种类、寿命和容量的完整键。"""
        return (
            _CONTENT_VERSION,
            *_packed(self.role.stable_key()),
            len(self.allowed_object_kinds),
            *self.allowed_object_kinds,
            self.lifespan_scope_kind,
            self.max_active,
        )


@dataclass(frozen=True)
class WorkMemoryContentProtocol:
    """一次 WorkMemory 实例的开放角色表和 transient history 硬上限。"""

    roles: tuple[WorkMemoryRoleDefinition, ...]
    max_history: int

    def __post_init__(self) -> None:
        if (not isinstance(self.roles, tuple) or not self.roles
                or any(not isinstance(item, WorkMemoryRoleDefinition)
                       for item in self.roles)):
            raise TypeError("WorkMemory content protocol 必须含 Role 定义")
        role_keys = tuple(item.role.stable_key() for item in self.roles)
        if len(set(role_keys)) != len(role_keys):
            raise ValueError("WorkMemory content Role 不得重复注册")
        _strict_positive(self.max_history, where="WorkMemory max_history")
        object.__setattr__(self, "roles", tuple(sorted(
            self.roles, key=lambda item: item.role.stable_key())))

    def definition(self, role: ObjectIdentity) -> WorkMemoryRoleDefinition:
        """按完整 Role 身份返回定义，未注册时拒绝猜测字段语义。"""
        if not isinstance(role, ObjectIdentity):
            raise TypeError("WorkMemory content role 必须是 ObjectIdentity")
        matches = tuple(item for item in self.roles if item.role == role)
        if len(matches) != 1:
            raise WorkMemoryContentError("WorkMemory content Role 未注册")
        return matches[0]

    def stable_key(self) -> tuple[int, ...]:
        """返回角色定义和 transient history 容量的确定性键。"""
        result = [_CONTENT_VERSION, self.max_history, len(self.roles)]
        for role in self.roles:
            result.extend(_packed(role.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class WorkMemoryOccurrenceAnchor:
    """同时保存 occurrence 图引用、权威身份、来源和形成 scope。"""

    occurrence: TypedRef
    identity: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity

    def __post_init__(self) -> None:
        if (not isinstance(self.occurrence, TypedRef)
                or self.occurrence.object_kind != OBJECT_OCCURRENCE):
            raise TypeError("WorkMemory anchor 必须是一等 Occurrence 引用")
        if (not isinstance(self.identity, ObjectIdentity)
                or self.identity.object_kind != OBJECT_OCCURRENCE):
            raise TypeError("WorkMemory anchor 必须携带 Occurrence 图身份")
        if not isinstance(self.source, SourceRef):
            raise TypeError("WorkMemory anchor source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("WorkMemory anchor scope 必须是 ScopeIdentity")
        if (self.occurrence.owner != self.identity.owner
                or self.occurrence.versions != self.identity.versions):
            raise ValueError("WorkMemory anchor 图引用与图身份 owner/version 不一致")
        if (self.identity.owner != self.source.owner
                or self.identity.versions != self.source.versions):
            raise ValueError("WorkMemory anchor 图身份与 SourceRef 不一致")
        if (len(self.identity.components) != _SOURCE_KEY_SIZE + 3
                or self.identity.components[:_SOURCE_KEY_SIZE]
                != self.source.stable_key()):
            raise ValueError("WorkMemory anchor Occurrence 身份不属于 SourceRef")
        if (self.scope.source != self.source
                or self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions):
            raise ValueError("WorkMemory anchor scope 与 SourceRef 不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回 occurrence 图引用、图身份、来源和 scope 的完整键。"""
        return (
            _CONTENT_VERSION,
            *_packed(self.occurrence.stable_key()),
            *_packed(self.identity.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
        )


@dataclass(frozen=True)
class WorkMemoryContentItem:
    """一个来源化内容值及其显式 lifespan、逻辑序和替代前项。"""

    role: ObjectIdentity
    value: ObjectIdentity
    anchor: WorkMemoryOccurrenceAnchor
    lifespan_scope: ScopeIdentity
    logical_seq: int
    trace: tuple[int, ...]
    supersedes: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.role, ObjectIdentity)
                or self.role.object_kind != OBJECT_ROLE):
            raise TypeError("WorkMemory item role 必须是一等 Role")
        if not isinstance(self.value, ObjectIdentity):
            raise TypeError("WorkMemory item value 必须是 ObjectIdentity")
        if not isinstance(self.anchor, WorkMemoryOccurrenceAnchor):
            raise TypeError("WorkMemory item anchor 类型错误")
        if not isinstance(self.lifespan_scope, ScopeIdentity):
            raise TypeError("WorkMemory item lifespan_scope 类型错误")
        _strict_nonnegative(self.logical_seq, where="WorkMemory item logical_seq")
        _strict_key(self.trace, where="WorkMemory item trace")
        if not isinstance(self.supersedes, tuple):
            raise TypeError("WorkMemory item supersedes 必须是 tuple")
        for key in self.supersedes:
            _strict_key(key, where="WorkMemory item supersedes key")
        if len(set(self.supersedes)) != len(self.supersedes):
            raise ValueError("WorkMemory item 不得重复 supersede 同一前项")
        if self.value.object_kind in SEMANTIC_OBJECT_KINDS:
            validate_semantic_identity(self.value)
            if (self.value.object_kind != OBJECT_ROLE
                    and semantic_source(self.value) != self.anchor.source):
                raise ValueError("WorkMemory 来源化语义 value 与 occurrence 来源不一致")
        if (self.value.object_kind == OBJECT_OCCURRENCE
                and (len(self.value.components) != _SOURCE_KEY_SIZE + 3
                     or self.value.components[:_SOURCE_KEY_SIZE]
                     != self.anchor.source.stable_key())):
            raise ValueError("WorkMemory Occurrence value 与 anchor 来源不一致")
        object.__setattr__(self, "supersedes", tuple(sorted(self.supersedes)))

    @property
    def source(self) -> SourceRef:
        """返回形成当前内容项的原始来源。"""
        return self.anchor.source

    def stable_key(self) -> tuple[int, ...]:
        """返回内容值、来源锚、寿命、逻辑序、trace 和前项引用。"""
        result = [
            _CONTENT_VERSION,
            *_packed(self.role.stable_key()),
            *_packed(self.value.stable_key()),
            *_packed(self.anchor.stable_key()),
            *_packed(self.lifespan_scope.stable_key()),
            self.logical_seq,
            *_packed(self.trace),
            len(self.supersedes),
        ]
        for key in self.supersedes:
            result.extend(_packed(key))
        return tuple(result)

    def content_ref(self) -> tuple[int, ...]:
        """返回固定长度内容引用，避免 supersede 链递归放大稳定键。"""
        return integer_tuple_fingerprint(
            self.stable_key(), domain=_ITEM_REFERENCE_DOMAIN)


class WorkMemoryContentStore:
    """在 A-09 活动 scope 内维护有界、显式替代的 transient 内容。"""

    def __init__(self, protocol: WorkMemoryContentProtocol) -> None:
        """绑定不可变角色/容量协议并建立空生命周期状态。"""
        if not isinstance(protocol, WorkMemoryContentProtocol):
            raise TypeError("WorkMemoryContentStore 需要 content protocol")
        self.protocol = protocol
        self._active_scopes: dict[int, ScopeIdentity] = {}
        self._items: dict[tuple[int, ...], WorkMemoryContentItem] = {}
        self._superseded_by: dict[tuple[int, ...], tuple[int, ...]] = {}

    def open_scope(self, scope: ScopeIdentity) -> None:
        """按 A-09 父子顺序打开一个内容生命周期边界。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("WorkMemory content scope 必须是 ScopeIdentity")
        if scope.scope_kind not in _LIFESPAN_SCOPE_KINDS:
            raise WorkMemoryContentError("WorkMemory content 不接受该 scope kind")
        if scope.scope_kind in self._active_scopes:
            if self._active_scopes[scope.scope_kind] == scope:
                return
            raise WorkMemoryContentError("同 kind WorkMemory content scope 尚未关闭")
        if scope.scope_kind == SCOPE_SESSION:
            if self._active_scopes or self._items:
                raise WorkMemoryContentError("打开 session 前内容状态必须为空")
        elif scope.scope_kind == SCOPE_DOCUMENT:
            session = self._active_scopes.get(SCOPE_SESSION)
            if session is None:
                raise WorkMemoryContentError("打开 document 前缺少活动 session")
            if (scope.owner != session.owner
                    or scope.versions != session.versions):
                raise WorkMemoryContentError("document 与 session owner/version 不一致")
        else:
            parent_kind = _PARENT_SCOPE_KIND[scope.scope_kind]
            parent = self._active_scopes.get(parent_kind)
            if parent is None or scope.parent != parent:
                raise WorkMemoryContentError("WorkMemory content 子 scope 与活动父边界不一致")
        self._active_scopes[scope.scope_kind] = scope

    def close_scope(self, scope: ScopeIdentity) -> None:
        """关闭精确活动 scope，并清除仅属于该 lifespan 的全部历史。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("WorkMemory content scope 必须是 ScopeIdentity")
        current = self._active_scopes.get(scope.scope_kind)
        if current is None:
            return
        if current != scope:
            raise WorkMemoryContentError("关闭的 WorkMemory content scope 不是当前边界")
        child_kinds = tuple(
            kind for kind, parent_kind in _PARENT_SCOPE_KIND.items()
            if parent_kind == scope.scope_kind and kind in self._active_scopes)
        if child_kinds:
            raise WorkMemoryContentError("关闭 WorkMemory content scope 前仍有活动子边界")
        expired = {
            key for key, item in self._items.items()
            if item.lifespan_scope == scope
        }
        for key in expired:
            self._items.pop(key, None)
            self._superseded_by.pop(key, None)
        for old_key, new_key in tuple(self._superseded_by.items()):
            if new_key in expired:
                self._superseded_by.pop(old_key, None)
        self._active_scopes.pop(scope.scope_kind)

    def scope_for_role(self, role: ObjectIdentity) -> ScopeIdentity:
        """返回指定 Role 当前应使用的精确 lifespan scope。"""
        definition = self.protocol.definition(role)
        scope = self._active_scopes.get(definition.lifespan_scope_kind)
        if scope is None:
            raise WorkMemoryContentError("WorkMemory Role 对应 lifespan 尚未活动")
        return scope

    def put(self, item: WorkMemoryContentItem) -> WorkMemoryContentItem:
        """在首写前核验来源、scope、替代链和容量，再幂等追加内容项。"""
        if not isinstance(item, WorkMemoryContentItem):
            raise TypeError("WorkMemory content put 需要 WorkMemoryContentItem")
        definition = self.protocol.definition(item.role)
        if item.value.object_kind not in definition.allowed_object_kinds:
            raise WorkMemoryContentError("WorkMemory item value kind 未获 Role 允许")
        expected_scope = self._active_scopes.get(definition.lifespan_scope_kind)
        if expected_scope is None or item.lifespan_scope != expected_scope:
            raise WorkMemoryContentError("WorkMemory item lifespan 不是 Role 当前活动 scope")
        document = self._active_scopes.get(SCOPE_DOCUMENT)
        if document is None or document.source != item.source:
            raise WorkMemoryContentError("WorkMemory item 必须来自当前活动 document")
        if not _descends_from(item.anchor.scope, document):
            raise WorkMemoryContentError("WorkMemory item occurrence scope 不属于当前 document")
        if (item.lifespan_scope.owner != item.source.owner
                or item.lifespan_scope.versions != item.source.versions):
            raise WorkMemoryContentError("WorkMemory item lifespan owner/version 与来源不一致")
        item_ref = item.content_ref()
        replay = self._items.get(item_ref)
        if replay is not None:
            if replay != item:
                raise WorkMemoryContentError("WorkMemory item 内容引用碰撞")
            return replay
        if len(self._items) >= self.protocol.max_history:
            raise WorkMemoryContentError("WorkMemory transient history 容量已满")
        superseded = set(item.supersedes)
        for old_ref in superseded:
            old = self._items.get(old_ref)
            if old is None:
                raise WorkMemoryContentError("WorkMemory supersede 前项不存在")
            if old_ref in self._superseded_by:
                raise WorkMemoryContentError("WorkMemory supersede 前项已非 active")
            if old.role != item.role or old.lifespan_scope != item.lifespan_scope:
                raise WorkMemoryContentError("WorkMemory 只能替代同 Role、同 lifespan 前项")
            if item.logical_seq <= old.logical_seq:
                raise WorkMemoryContentError("WorkMemory 替代项逻辑序必须晚于前项")
        active_same_role = tuple(
            key for key, existing in self._items.items()
            if (existing.role == item.role
                and existing.lifespan_scope == item.lifespan_scope
                and key not in self._superseded_by
                and key not in superseded)
        )
        if len(active_same_role) + 1 > definition.max_active:
            raise WorkMemoryContentError("WorkMemory Role active 容量已满")
        self._items[item_ref] = item
        for old_ref in superseded:
            self._superseded_by[old_ref] = item_ref
        return item

    def active(
            self, *, role: ObjectIdentity | None = None,
            ) -> tuple[WorkMemoryContentItem, ...]:
        """返回全部或指定 Role 的 active 项，不施加最近项 winner。"""
        if role is not None:
            self.protocol.definition(role)
        values = tuple(
            item for key, item in self._items.items()
            if key not in self._superseded_by
            and (role is None or item.role == role)
        )
        return tuple(sorted(
            values,
            key=lambda item: (item.logical_seq, item.content_ref()),
        ))

    def history(
            self, *, role: ObjectIdentity | None = None,
            ) -> tuple[WorkMemoryContentItem, ...]:
        """返回当前存活 lifespan 内完整历史，含已被替代项。"""
        if role is not None:
            self.protocol.definition(role)
        values = tuple(
            item for item in self._items.values()
            if role is None or item.role == role
        )
        return tuple(sorted(
            values,
            key=lambda item: (item.logical_seq, item.content_ref()),
        ))

    def clone(self) -> "WorkMemoryContentStore":
        """复制独立 transient 状态，供评测或故障预演使用。"""
        cloned = WorkMemoryContentStore(self.protocol)
        cloned._active_scopes = dict(self._active_scopes)
        cloned._items = dict(self._items)
        cloned._superseded_by = dict(self._superseded_by)
        return cloned

    def state_key(self) -> tuple[int, ...]:
        """返回协议、活动 scope、内容历史和 supersede 投影的完整键。"""
        result = [
            _CONTENT_VERSION,
            *_packed(self.protocol.stable_key()),
            len(self._active_scopes),
        ]
        for kind, scope in sorted(self._active_scopes.items()):
            result.extend((kind, *_packed(scope.stable_key())))
        result.append(len(self._items))
        for key in sorted(self._items):
            result.extend(_packed(key))
            result.extend(_packed(self._items[key].stable_key()))
        result.append(len(self._superseded_by))
        for old_key, new_key in sorted(self._superseded_by.items()):
            result.extend(_packed(old_key))
            result.extend(_packed(new_key))
        return tuple(result)


__all__ = [
    "WorkMemoryContentError",
    "WorkMemoryContentItem",
    "WorkMemoryContentProtocol",
    "WorkMemoryContentStore",
    "WorkMemoryOccurrenceAnchor",
    "WorkMemoryRoleDefinition",
]
