"""运行 scope、逻辑时钟和断言的纯整数身份契约。

本模块只定义领域对象及稳定键，不负责持久化。具体语言规则、序类型和角色通过
``qualifiers`` 注入，不能在共享协议中写死。身份索引只是这些对象的可恢复投影，
不能替代图内命题、证据或结构概念。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    TypedRef,
    VersionBundle,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


SCOPE_DOCUMENT = 1
SCOPE_EPISODE = 2
SCOPE_QUERY = 3
SCOPE_GENERATION = 4
SCOPE_SESSION = 5

CLOCK_OBSERVATION = 1
CLOCK_EPISODE = 2
CLOCK_QUERY = 3
CLOCK_GENERATION = 4
CLOCK_MEMORY_CREATED = 5
CLOCK_MEMORY_OBSERVED = 6
CLOCK_MEMORY_USED = 7
CLOCK_MEMORY_LIFECYCLE = 8
CLOCK_MEMORY_IMPORT = 9
CLOCK_MEMORY_RESOLVED = 10

_SCOPE_KEY_VERSION = 1
_CLOCK_KEY_VERSION = 1
_TIMESTAMP_KEY_VERSION = 1
_ASSERTION_KEY_VERSION = 1
_OWNER_KEY_SIZE = 4
_VERSION_KEY_SIZE = 4
_SOURCE_KEY_SIZE = 3 + _OWNER_KEY_SIZE + _VERSION_KEY_SIZE
_TYPED_REF_KEY_SIZE = 3 + _OWNER_KEY_SIZE + _VERSION_KEY_SIZE


def _require_identity_int(value: int, *, where: str,
                          nonnegative: bool = False,
                          positive: bool = False) -> int:
    """校验身份字段是严格整数，并按需限制正负范围。"""
    if type(value) is not int:
        assert_int(value, _where=where)
        raise ValueError(f"{where} 必须为严格整数")
    if positive and value <= 0:
        raise ValueError(f"{where} 必须为正整数")
    if nonnegative and value < 0:
        raise ValueError(f"{where} 必须为非负整数")
    return value


def _require_integer_key(key: tuple[int, ...], *, where: str) -> None:
    """校验稳定键只含严格整数且非空。"""
    if not isinstance(key, tuple) or not key:
        raise ValueError(f"{where} 必须是非空整数元组")
    for index, value in enumerate(key):
        _require_identity_int(value, where=f"{where}[{index}]")


def _owner_from_key(key: tuple[int, ...]) -> OwnerScope:
    """从固定长度稳定键恢复 owner。"""
    if len(key) != _OWNER_KEY_SIZE:
        raise ValueError("OwnerScope 稳定键长度非法")
    return OwnerScope(key[0], key[1], key[2], key[3])


def _versions_from_key(key: tuple[int, ...]) -> VersionBundle:
    """从固定长度稳定键恢复版本束。"""
    if len(key) != _VERSION_KEY_SIZE:
        raise ValueError("VersionBundle 稳定键长度非法")
    return VersionBundle(
        CorpusVersion(key[0]),
        ParserVersion(key[1]),
        PrimitiveVersion(key[2]),
        CurriculumVersion(key[3]),
    )


def _source_from_key(key: tuple[int, ...]) -> SourceRef:
    """从固定长度稳定键恢复来源引用。"""
    if len(key) != _SOURCE_KEY_SIZE:
        raise ValueError("SourceRef 稳定键长度非法")
    return SourceRef(
        key[0], key[1], key[2],
        _owner_from_key(key[3:7]),
        _versions_from_key(key[7:11]),
    )


def _typed_ref_from_key(key: tuple[int, ...]) -> TypedRef:
    """从固定长度稳定键恢复分型图引用。"""
    if len(key) != _TYPED_REF_KEY_SIZE:
        raise ValueError("TypedRef 稳定键长度非法")
    return TypedRef(
        key[0], key[1], key[2],
        _owner_from_key(key[3:7]),
        _versions_from_key(key[7:11]),
    )


@dataclass(frozen=True, order=True)
class ScopeIdentity:
    """可嵌套的运行边界身份，显式携带 owner、版本和可选来源。"""

    scope_kind: int
    local_id: int
    owner: OwnerScope = GLOBAL_OWNER_SCOPE
    versions: VersionBundle = VersionBundle()
    source: SourceRef | None = None
    parent: "ScopeIdentity | None" = None

    def __post_init__(self) -> None:
        _require_identity_int(
            self.scope_kind, where="ScopeIdentity.scope_kind", positive=True)
        _require_identity_int(
            self.local_id, where="ScopeIdentity.local_id", nonnegative=True)
        if self.source is not None:
            if self.source.owner != self.owner:
                raise ValueError("ScopeIdentity.source owner 与 scope 不一致")
            if self.source.versions != self.versions:
                raise ValueError("ScopeIdentity.source versions 与 scope 不一致")
        if self.parent is not None:
            if self.parent.owner != self.owner:
                raise ValueError("子 scope 与 parent owner 不一致")
            if self.parent.versions != self.versions:
                raise ValueError("子 scope 与 parent versions 不一致")
            if (self.source is not None and self.parent.source is not None
                    and self.source != self.parent.source):
                raise ValueError("子 scope 与 parent source 不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回带长度边界的递归纯整数稳定键。"""
        source_key = () if self.source is None else self.source.stable_key()
        parent_key = () if self.parent is None else self.parent.stable_key()
        return (
            _SCOPE_KEY_VERSION,
            self.scope_kind,
            self.local_id,
            *self.owner.stable_key(),
            *self.versions.stable_key(),
            len(source_key),
            *source_key,
            len(parent_key),
            *parent_key,
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "ScopeIdentity":
        """从完整稳定键恢复 scope，遇截断或尾随字段时拒绝。"""
        _require_integer_key(key, where="ScopeIdentity.stable_key")
        fixed_size = 3 + _OWNER_KEY_SIZE + _VERSION_KEY_SIZE
        if len(key) < fixed_size + 2 or key[0] != _SCOPE_KEY_VERSION:
            raise ValueError("ScopeIdentity 稳定键版本或长度非法")
        cursor = fixed_size
        source_size = key[cursor]
        cursor += 1
        if source_size not in (0, _SOURCE_KEY_SIZE):
            raise ValueError("ScopeIdentity source 长度非法")
        source_end = cursor + source_size
        if source_end >= len(key):
            raise ValueError("ScopeIdentity source 键被截断")
        source = None if source_size == 0 else _source_from_key(key[cursor:source_end])
        cursor = source_end
        parent_size = key[cursor]
        cursor += 1
        if parent_size < 0 or cursor + parent_size != len(key):
            raise ValueError("ScopeIdentity parent 长度非法")
        parent = None if parent_size == 0 else cls.from_stable_key(
            key[cursor:cursor + parent_size])
        return cls(
            key[1], key[2],
            _owner_from_key(key[3:7]),
            _versions_from_key(key[7:11]),
            source,
            parent,
        )


def make_scope(scope_kind: int, local_id: int, *,
               owner: OwnerScope = GLOBAL_OWNER_SCOPE,
               versions: VersionBundle = VersionBundle(),
               source: SourceRef | None = None,
               parent: ScopeIdentity | None = None) -> ScopeIdentity:
    """构造开放类型的 scope；调用方注入类型，不依赖固定语义表。"""
    if parent is not None:
        if owner == GLOBAL_OWNER_SCOPE and parent.owner != GLOBAL_OWNER_SCOPE:
            owner = parent.owner
        if versions == VersionBundle() and parent.versions != VersionBundle():
            versions = parent.versions
        if source is None:
            source = parent.source
    return ScopeIdentity(scope_kind, local_id, owner, versions, source, parent)


def document_scope(source: SourceRef, *,
                   parent: ScopeIdentity | None = None) -> ScopeIdentity:
    """由来源记录构造文档 scope。"""
    return make_scope(
        SCOPE_DOCUMENT,
        source.document_id,
        owner=source.owner,
        versions=source.versions,
        source=source,
        parent=parent,
    )


def session_scope(local_id: int, *,
                  owner: OwnerScope = GLOBAL_OWNER_SCOPE,
                  versions: VersionBundle = VersionBundle(),
                  source: SourceRef | None = None) -> ScopeIdentity:
    """构造 session scope，作为一组 document/episode 的最外层边界。"""
    return make_scope(
        SCOPE_SESSION, local_id, owner=owner, versions=versions, source=source)


def episode_scope(local_id: int, *, parent: ScopeIdentity | None = None,
                  owner: OwnerScope = GLOBAL_OWNER_SCOPE,
                  versions: VersionBundle = VersionBundle(),
                  source: SourceRef | None = None) -> ScopeIdentity:
    """构造 episode scope，并在给定 parent 时继承隔离字段。"""
    return make_scope(
        SCOPE_EPISODE, local_id, owner=owner, versions=versions,
        source=source, parent=parent)


def query_scope(local_id: int, *, parent: ScopeIdentity | None = None,
                owner: OwnerScope = GLOBAL_OWNER_SCOPE,
                versions: VersionBundle = VersionBundle(),
                source: SourceRef | None = None) -> ScopeIdentity:
    """构造 query scope，并在给定 parent 时继承隔离字段。"""
    return make_scope(
        SCOPE_QUERY, local_id, owner=owner, versions=versions,
        source=source, parent=parent)


def generation_scope(local_id: int, *, parent: ScopeIdentity | None = None,
                     owner: OwnerScope = GLOBAL_OWNER_SCOPE,
                     versions: VersionBundle = VersionBundle(),
                     source: SourceRef | None = None) -> ScopeIdentity:
    """构造 generation scope，并在给定 parent 时继承隔离字段。"""
    return make_scope(
        SCOPE_GENERATION, local_id, owner=owner, versions=versions,
        source=source, parent=parent)


@dataclass(frozen=True, order=True)
class LogicalClockIdentity:
    """由 scope 拥有的某类逻辑时钟身份。"""

    scope: ScopeIdentity
    clock_kind: int

    def __post_init__(self) -> None:
        _require_identity_int(
            self.clock_kind, where="LogicalClockIdentity.clock_kind", positive=True)

    def stable_key(self) -> tuple[int, ...]:
        """返回包含完整 owner scope 的时钟键。"""
        scope_key = self.scope.stable_key()
        return (
            _CLOCK_KEY_VERSION,
            self.clock_kind,
            len(scope_key),
            *scope_key,
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "LogicalClockIdentity":
        """从完整稳定键恢复时钟身份。"""
        _require_integer_key(key, where="LogicalClockIdentity.stable_key")
        if len(key) < 4 or key[0] != _CLOCK_KEY_VERSION:
            raise ValueError("LogicalClockIdentity 稳定键版本或长度非法")
        scope_size = key[2]
        if scope_size <= 0 or 3 + scope_size != len(key):
            raise ValueError("LogicalClockIdentity scope 长度非法")
        return cls(ScopeIdentity.from_stable_key(key[3:]), key[1])


@dataclass(frozen=True, order=True)
class LogicalTimestamp:
    """完整逻辑时间戳；裸 seq 不能跨时钟比较身份。"""

    clock: LogicalClockIdentity
    seq: int

    def __post_init__(self) -> None:
        _require_identity_int(
            self.seq, where="LogicalTimestamp.seq", positive=True)

    def stable_key(self) -> tuple[int, ...]:
        """返回包含完整时钟身份的时间戳键。"""
        clock_key = self.clock.stable_key()
        return (
            _TIMESTAMP_KEY_VERSION,
            self.seq,
            len(clock_key),
            *clock_key,
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "LogicalTimestamp":
        """从完整稳定键恢复逻辑时间戳。"""
        _require_integer_key(key, where="LogicalTimestamp.stable_key")
        if len(key) < 4 or key[0] != _TIMESTAMP_KEY_VERSION:
            raise ValueError("LogicalTimestamp 稳定键版本或长度非法")
        clock_size = key[2]
        if clock_size <= 0 or 3 + clock_size != len(key):
            raise ValueError("LogicalTimestamp clock 长度非法")
        return cls(LogicalClockIdentity.from_stable_key(key[3:]), key[1])


class LogicalClock:
    """只在自身 identity 内单调推进和恢复的无墙钟逻辑时钟。"""

    def __init__(self, identity: LogicalClockIdentity,
                 current_seq: int = 0) -> None:
        _require_identity_int(
            current_seq, where="LogicalClock.current_seq", nonnegative=True)
        self.identity = identity
        self._current_seq = current_seq

    @property
    def current_seq(self) -> int:
        """返回当前序号，不推进时钟。"""
        return self._current_seq

    def advance(self, step: int = 1) -> LogicalTimestamp:
        """按正整数步长推进，并返回完整时间戳。"""
        _require_identity_int(step, where="LogicalClock.step", positive=True)
        self._current_seq += step
        return LogicalTimestamp(self.identity, self._current_seq)

    def restore(self, timestamp: LogicalTimestamp) -> None:
        """仅接受同一时钟且不倒退的恢复点。"""
        if timestamp.clock != self.identity:
            raise ValueError("LogicalClock 不得从其他 clock identity 恢复")
        if timestamp.seq < self._current_seq:
            raise ValueError("LogicalClock 恢复不得倒退")
        self._current_seq = timestamp.seq


@dataclass(frozen=True, order=True)
class AssertionIdentity:
    """可唯一寻址的关系断言身份，语义细节由开放整数限定项携带。"""

    relation_kind: int
    subject: TypedRef
    object: TypedRef
    scope: ScopeIdentity
    provenance_kind: int
    epistemic_origin: int = 0
    content_version: int = 0
    qualifiers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_identity_int(
            self.relation_kind, where="AssertionIdentity.relation_kind", positive=True)
        _require_identity_int(
            self.provenance_kind, where="AssertionIdentity.provenance_kind", positive=True)
        _require_identity_int(
            self.epistemic_origin, where="AssertionIdentity.epistemic_origin",
            nonnegative=True)
        _require_identity_int(
            self.content_version, where="AssertionIdentity.content_version",
            nonnegative=True)
        if not isinstance(self.qualifiers, tuple):
            raise ValueError("AssertionIdentity.qualifiers 必须是整数元组")
        for index, value in enumerate(self.qualifiers):
            _require_identity_int(
                value, where=f"AssertionIdentity.qualifiers[{index}]")

    def stable_key(self) -> tuple[int, ...]:
        """返回含端点类型、scope、来源、版本和限定项的无歧义稳定键。"""
        subject_key = self.subject.stable_key()
        object_key = self.object.stable_key()
        scope_key = self.scope.stable_key()
        return (
            _ASSERTION_KEY_VERSION,
            self.relation_kind,
            len(subject_key),
            *subject_key,
            len(object_key),
            *object_key,
            len(scope_key),
            *scope_key,
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            len(self.qualifiers),
            *self.qualifiers,
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "AssertionIdentity":
        """从完整稳定键恢复断言，拒绝任何截断、尾随或长度歧义。"""
        _require_integer_key(key, where="AssertionIdentity.stable_key")
        if len(key) < 2 or key[0] != _ASSERTION_KEY_VERSION:
            raise ValueError("AssertionIdentity 稳定键版本非法")
        cursor = 2

        def take_sized(label: str) -> tuple[int, ...]:
            nonlocal cursor
            if cursor >= len(key):
                raise ValueError(f"AssertionIdentity {label} 长度缺失")
            size = key[cursor]
            cursor += 1
            if size <= 0 or cursor + size > len(key):
                raise ValueError(f"AssertionIdentity {label} 长度非法")
            part = key[cursor:cursor + size]
            cursor += size
            return part

        subject_key = take_sized("subject")
        object_key = take_sized("object")
        scope_key = take_sized("scope")
        if cursor + 4 > len(key):
            raise ValueError("AssertionIdentity 尾部字段被截断")
        provenance_kind = key[cursor]
        epistemic_origin = key[cursor + 1]
        content_version = key[cursor + 2]
        qualifier_size = key[cursor + 3]
        cursor += 4
        if qualifier_size < 0 or cursor + qualifier_size != len(key):
            raise ValueError("AssertionIdentity qualifiers 长度非法")
        return cls(
            key[1],
            _typed_ref_from_key(subject_key),
            _typed_ref_from_key(object_key),
            ScopeIdentity.from_stable_key(scope_key),
            provenance_kind,
            epistemic_origin,
            content_version,
            key[cursor:],
        )


def concept_assertion(relation_kind: int,
                      subject_ref: tuple[int, int],
                      object_ref: tuple[int, int], *,
                      scope: ScopeIdentity,
                      provenance_kind: int,
                      epistemic_origin: int = 0,
                      content_version: int = 0,
                      qualifiers: tuple[int, ...] = ()) -> AssertionIdentity:
    """把现有概念节点端点投影为 scoped assertion，不声称其是语言原子。"""
    subject = TypedRef(
        OBJECT_CONCEPT, subject_ref[0], subject_ref[1],
        GLOBAL_OWNER_SCOPE, VersionBundle())
    object_ref_typed = TypedRef(
        OBJECT_CONCEPT, object_ref[0], object_ref[1],
        GLOBAL_OWNER_SCOPE, VersionBundle())
    return AssertionIdentity(
        relation_kind,
        subject,
        object_ref_typed,
        scope,
        provenance_kind,
        epistemic_origin,
        content_version,
        qualifiers,
    )


__all__ = [
    "AssertionIdentity",
    "CLOCK_EPISODE",
    "CLOCK_GENERATION",
    "CLOCK_OBSERVATION",
    "CLOCK_QUERY",
    "LogicalClock",
    "LogicalClockIdentity",
    "LogicalTimestamp",
    "SCOPE_DOCUMENT",
    "SCOPE_EPISODE",
    "SCOPE_GENERATION",
    "SCOPE_QUERY",
    "SCOPE_SESSION",
    "ScopeIdentity",
    "concept_assertion",
    "document_scope",
    "episode_scope",
    "generation_scope",
    "make_scope",
    "query_scope",
    "session_scope",
]
