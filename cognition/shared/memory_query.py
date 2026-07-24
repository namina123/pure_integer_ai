"""把当前 typed 输入编译为 Memory 检索请求。

本模块只负责把当前 query 的 occurrence、Span、语义对象、结构、意图和任务上下文
编码为可审计的 activation request。它不读取聚合行、不进行排序或仲裁、不写
Memory event，也不把上一 episode 的 reward、replay 或 WorkMemory 残留作为输入。
具体 query kind、输入角色和 Hypothesis kind 全部由调用方以一等
MinimalInstruction 身份注入。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_OCCURRENCE,
    OBJECT_SPAN,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
    TypedRef,
)
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MemoryHypothesisAggregateIndex,
)
from pure_integer_ai.cognition.shared.memory_overlay import (
    MemoryAccessContext,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    LogicalTimestamp,
    SCOPE_QUERY,
    ScopeIdentity,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_MEMORY,
    SpaceIdentity,
)


_QUERY_PROTOCOL_VERSION = 1
_QUERY_REFERENCE_TYPED = 1
_QUERY_REFERENCE_OBJECT = 2


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界，避免拼接歧义。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """校验调用方注入的开放整数 Hypothesis kind。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _require_instruction(value: ObjectIdentity, *, label: str) -> None:
    """要求 query 角色和 query kind 都是注入的一等最小指令。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")


def _require_object(value: ObjectIdentity, *, label: str) -> None:
    """校验领域上下文以一等对象而非裸枚举进入查询。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")


def _require_refs(
        values: tuple[TypedRef, ...], *, label: str,
        required_kind: int | None = None,
        ) -> tuple[TypedRef, ...]:
    """校验 typed 输入引用、可选对象种类和重复身份。"""
    if not isinstance(values, tuple) or any(
            not isinstance(item, TypedRef) for item in values):
        raise TypeError(f"{label} 必须是 TypedRef tuple")
    if required_kind is not None and any(
            item.object_kind != required_kind for item in values):
        raise ValueError(f"{label} 包含错误对象种类")
    stable_keys = tuple(item.stable_key() for item in values)
    if len(set(stable_keys)) != len(stable_keys):
        raise ValueError(f"{label} 不得重复同一 typed identity")
    return values


@dataclass(frozen=True)
class MemoryQueryReference:
    """一条 query 理由中的一等对象或 typed 图引用。"""

    value: ObjectIdentity | TypedRef

    def __post_init__(self) -> None:
        """拒绝裸整数、字符串和不可追溯的派生 hash。"""
        if not isinstance(self.value, (ObjectIdentity, TypedRef)):
            raise TypeError("Memory query reference 必须是一等对象或 TypedRef")

    def stable_key(self) -> tuple[int, ...]:
        """返回带引用种类标签的完整稳定键。"""
        if isinstance(self.value, TypedRef):
            tag = _QUERY_REFERENCE_TYPED
        else:
            tag = _QUERY_REFERENCE_OBJECT
        return tag, *_packed(self.value.stable_key())


@dataclass(frozen=True)
class MemoryQueryReason:
    """一个注入角色及其当前输入锚点，供后续 resolver 解释匹配。"""

    role: ObjectIdentity
    anchors: tuple[MemoryQueryReference, ...]

    def __post_init__(self) -> None:
        """要求理由有唯一角色和至少一个可回溯锚点。"""
        _require_instruction(self.role, label="MemoryQueryReason.role")
        if not isinstance(self.anchors, tuple) or not self.anchors:
            raise ValueError("MemoryQueryReason.anchors 必须为非空 tuple")
        if any(not isinstance(item, MemoryQueryReference)
               for item in self.anchors):
            raise TypeError("MemoryQueryReason.anchors 类型错误")
        keys = tuple(item.stable_key() for item in self.anchors)
        if len(set(keys)) != len(keys):
            raise ValueError("MemoryQueryReason.anchors 不得重复")

    def stable_key(self) -> tuple[int, ...]:
        """返回角色与有序锚点的完整键，保留 occurrence 输入顺序。"""
        result = [*_packed(self.role.stable_key()), len(self.anchors)]
        for anchor in self.anchors:
            result.extend(_packed(anchor.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class MemoryQueryRoles:
    """当前输入各字段在图中对应的注入式 query 角色。"""

    occurrence: ObjectIdentity
    span: ObjectIdentity
    semantic_object: ObjectIdentity
    structure: ObjectIdentity
    domain: ObjectIdentity
    speaker: ObjectIdentity
    intent: ObjectIdentity
    task: ObjectIdentity

    def __post_init__(self) -> None:
        """确保各角色均为不同的一等最小指令。"""
        values = self.values()
        for label, value in zip(
                ("occurrence", "span", "semantic_object", "structure",
                 "domain", "speaker", "intent", "task"), values):
            _require_instruction(value, label=f"MemoryQueryRoles.{label}")
        keys = tuple(item.stable_key() for item in values)
        if len(set(keys)) != len(keys):
            raise ValueError("Memory query 输入角色不得复用同一 MinimalInstruction")

    def values(self) -> tuple[ObjectIdentity, ...]:
        """按固定字段序返回所有协议角色，供定义和编译核验。"""
        return (
            self.occurrence,
            self.span,
            self.semantic_object,
            self.structure,
            self.domain,
            self.speaker,
            self.intent,
            self.task,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回所有输入角色的完整协议键。"""
        result = []
        for item in self.values():
            result.extend(_packed(item.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class MemoryCurrentQuery:
    """一次 query 的完整 typed 输入快照，不携带历史奖励或回放状态。"""

    scope: ScopeIdentity
    source: SourceRef
    logical_timestamp: LogicalTimestamp
    occurrences: tuple[TypedRef, ...]
    spans: tuple[TypedRef, ...]
    semantic_objects: tuple[TypedRef, ...]
    structures: tuple[TypedRef, ...]
    domain: ObjectIdentity
    intent: ObjectIdentity
    speaker: ObjectIdentity | None = None
    task: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        """核验 query 边界、来源、时钟和 typed 输入没有跨 owner/version 漂移。"""
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("MemoryCurrentQuery.scope 必须是 ScopeIdentity")
        if self.scope.scope_kind != SCOPE_QUERY:
            raise ValueError("MemoryCurrentQuery 必须绑定 query ScopeIdentity")
        if not isinstance(self.source, SourceRef):
            raise TypeError("MemoryCurrentQuery.source 必须是 SourceRef")
        if (self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions
                or self.scope.source != self.source):
            raise ValueError("Memory query scope 与当前来源 owner/version 不一致")
        if not isinstance(self.logical_timestamp, LogicalTimestamp):
            raise TypeError("MemoryCurrentQuery.logical_timestamp 类型错误")
        if self.logical_timestamp.clock.scope != self.scope:
            raise ValueError("Memory query 逻辑时钟必须属于当前 query scope")
        _require_refs(
            self.occurrences,
            label="MemoryCurrentQuery.occurrences",
            required_kind=OBJECT_OCCURRENCE,
        )
        _require_refs(
            self.spans,
            label="MemoryCurrentQuery.spans",
            required_kind=OBJECT_SPAN,
        )
        _require_refs(
            self.semantic_objects,
            label="MemoryCurrentQuery.semantic_objects",
        )
        _require_refs(
            self.structures,
            label="MemoryCurrentQuery.structures",
            required_kind=OBJECT_STRUCTURE_CONCEPT,
        )
        if not any((self.occurrences, self.spans, self.semantic_objects,
                    self.structures)):
            raise ValueError("Memory query 至少需要一个当前 typed 锚点")
        _require_object(self.domain, label="MemoryCurrentQuery.domain")
        _require_object(self.intent, label="MemoryCurrentQuery.intent")
        if self.speaker is not None:
            _require_object(self.speaker, label="MemoryCurrentQuery.speaker")
        if self.task is not None:
            _require_object(self.task, label="MemoryCurrentQuery.task")

    def anchors_by_role(
            self, roles: MemoryQueryRoles,
            ) -> tuple[tuple[ObjectIdentity, tuple[MemoryQueryReference, ...]], ...]:
        """把字段值投影为注入角色，不解释领域对象的具体语言意义。"""
        if not isinstance(roles, MemoryQueryRoles):
            raise TypeError("anchors_by_role 需要 MemoryQueryRoles")

        def references(values: tuple[ObjectIdentity | TypedRef, ...]
                       ) -> tuple[MemoryQueryReference, ...]:
            """将一组当前一等身份包装成可审计 query 锚点。"""
            return tuple(MemoryQueryReference(item) for item in values)

        return (
            (roles.occurrence, references(self.occurrences)),
            (roles.span, references(self.spans)),
            (roles.semantic_object, references(self.semantic_objects)),
            (roles.structure, references(self.structures)),
            (roles.domain, references((self.domain,))),
            (roles.speaker, () if self.speaker is None else references(
                (self.speaker,))),
            (roles.intent, references((self.intent,))),
            (roles.task, () if self.task is None else references((self.task,))),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回当前 query 全部输入的稳定键，用于确定性回归和 trace。"""
        result = [
            _QUERY_PROTOCOL_VERSION,
            *_packed(self.scope.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.logical_timestamp.stable_key()),
        ]
        for refs in (
                self.occurrences, self.spans, self.semantic_objects,
                self.structures):
            result.append(len(refs))
            for ref in refs:
                result.extend(_packed(ref.stable_key()))
        for value in (self.domain, self.intent, self.speaker, self.task):
            result.extend(_packed(
                () if value is None else MemoryQueryReference(value).stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class MemoryQueryDefinition:
    """一个由课程注册的 query kind、目标 Hypothesis kind、理由角色和预算。"""

    query_kind: ObjectIdentity
    hypothesis_kind: tuple[int, ...]
    reason_roles: tuple[ObjectIdentity, ...]
    budget: int

    def __post_init__(self) -> None:
        """拒绝未定义的 kind、空理由和非正预算。"""
        _require_instruction(
            self.query_kind, label="MemoryQueryDefinition.query_kind")
        _strict_key(
            self.hypothesis_kind,
            label="MemoryQueryDefinition.hypothesis_kind",
        )
        if not isinstance(self.reason_roles, tuple) or not self.reason_roles:
            raise ValueError("MemoryQueryDefinition.reason_roles 必须为非空 tuple")
        for role in self.reason_roles:
            _require_instruction(
                role, label="MemoryQueryDefinition.reason_role")
        keys = tuple(item.stable_key() for item in self.reason_roles)
        if len(set(keys)) != len(keys):
            raise ValueError("MemoryQueryDefinition.reason_roles 不得重复")
        assert_int(self.budget, _where="MemoryQueryDefinition.budget")
        if type(self.budget) is not int or self.budget <= 0:
            raise ValueError("MemoryQueryDefinition.budget 必须为正严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回注入定义的完整稳定键。"""
        result = [
            *_packed(self.query_kind.stable_key()),
            *_packed(self.hypothesis_kind),
            self.budget,
            len(self.reason_roles),
        ]
        for role in self.reason_roles:
            result.extend(_packed(role.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class MemoryQueryProtocol:
    """一组输入角色和 query 定义，所有领域含义都由调用方注入。"""

    roles: MemoryQueryRoles
    definitions: tuple[MemoryQueryDefinition, ...]

    def __post_init__(self) -> None:
        """核验定义只引用本协议角色，并按 query kind 规范化顺序。"""
        if not isinstance(self.roles, MemoryQueryRoles):
            raise TypeError("MemoryQueryProtocol.roles 类型错误")
        if not isinstance(self.definitions, tuple) or not self.definitions:
            raise ValueError("MemoryQueryProtocol.definitions 必须为非空 tuple")
        if any(not isinstance(item, MemoryQueryDefinition)
               for item in self.definitions):
            raise TypeError("MemoryQueryProtocol.definitions 类型错误")
        known_roles = set(self.roles.values())
        if any(not set(item.reason_roles).issubset(known_roles)
               for item in self.definitions):
            raise ValueError("Memory query 定义引用了未注册输入角色")
        kinds = tuple(item.query_kind.stable_key() for item in self.definitions)
        if len(set(kinds)) != len(kinds):
            raise ValueError("Memory query protocol 不得重复 query kind")
        object.__setattr__(self, "definitions", tuple(sorted(
            self.definitions,
            key=lambda item: item.query_kind.stable_key(),
        )))

    def stable_key(self) -> tuple[int, ...]:
        """返回角色与所有 query 定义的完整协议键。"""
        result = [
            _QUERY_PROTOCOL_VERSION,
            *_packed(self.roles.stable_key()),
            len(self.definitions),
        ]
        for definition in self.definitions:
            result.extend(_packed(definition.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class MemoryActivationRequest:
    """交给 M-07 resolver 或 A-10 agenda 的单类 Memory 激活请求。"""

    memory_space: SpaceIdentity
    access: MemoryAccessContext
    query_kind: ObjectIdentity
    hypothesis_kind: tuple[int, ...]
    scope: ScopeIdentity
    source: SourceRef
    logical_timestamp: LogicalTimestamp
    reasons: tuple[MemoryQueryReason, ...]
    budget: int

    def __post_init__(self) -> None:
        """校验 request 的 ACL、来源、时钟和理由均可完整审计。"""
        if not isinstance(self.memory_space, SpaceIdentity):
            raise TypeError("MemoryActivationRequest.memory_space 类型错误")
        if self.memory_space.space_type != SPACE_TYPE_MEMORY:
            raise ValueError("Memory activation request 必须指向 Memory 空间")
        if not isinstance(self.access, MemoryAccessContext):
            raise TypeError("MemoryActivationRequest.access 类型错误")
        _require_instruction(
            self.query_kind, label="MemoryActivationRequest.query_kind")
        _strict_key(
            self.hypothesis_kind,
            label="MemoryActivationRequest.hypothesis_kind",
        )
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("MemoryActivationRequest.scope 类型错误")
        if self.scope.scope_kind != SCOPE_QUERY:
            raise ValueError("Memory activation request 必须绑定 query scope")
        if not isinstance(self.source, SourceRef):
            raise TypeError("MemoryActivationRequest.source 类型错误")
        if (self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions
                or self.scope.source != self.source):
            raise ValueError("Memory activation request 的 scope/source 漂移")
        if not self.access.can_read(self.scope.owner):
            raise PermissionError("Memory query access 不可读取当前 owner")
        if not isinstance(self.logical_timestamp, LogicalTimestamp):
            raise TypeError("MemoryActivationRequest.logical_timestamp 类型错误")
        if self.logical_timestamp.clock.scope != self.scope:
            raise ValueError("Memory activation request 时钟不属于当前 query")
        if not isinstance(self.reasons, tuple) or not self.reasons:
            raise ValueError("Memory activation request 必须保留至少一个理由")
        if any(not isinstance(item, MemoryQueryReason) for item in self.reasons):
            raise TypeError("Memory activation request reasons 类型错误")
        role_keys = tuple(item.role.stable_key() for item in self.reasons)
        if len(set(role_keys)) != len(role_keys):
            raise ValueError("Memory activation request reasons 不得重复角色")
        assert_int(self.budget, _where="MemoryActivationRequest.budget")
        if type(self.budget) is not int or self.budget <= 0:
            raise ValueError("Memory activation request budget 必须为正严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回包含 ACL、输入边界、理由和预算的完整请求键。"""
        result = [
            _QUERY_PROTOCOL_VERSION,
            *_packed(self.memory_space.stable_key()),
            *_packed(self.access.stable_key()),
            *_packed(self.query_kind.stable_key()),
            *_packed(self.hypothesis_kind),
            *_packed(self.scope.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.logical_timestamp.stable_key()),
            self.budget,
            len(self.reasons),
        ]
        for reason in self.reasons:
            result.extend(_packed(reason.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class MemoryQueryCompilation:
    """一次当前输入编译得到的全部分型 activation request。"""

    current: MemoryCurrentQuery
    access: MemoryAccessContext
    memory_space: SpaceIdentity
    requests: tuple[MemoryActivationRequest, ...]

    def __post_init__(self) -> None:
        """核验所有输出请求共同绑定同一次输入、ACL 和 Memory 空间。"""
        if not isinstance(self.current, MemoryCurrentQuery):
            raise TypeError("MemoryQueryCompilation.current 类型错误")
        if not isinstance(self.access, MemoryAccessContext):
            raise TypeError("MemoryQueryCompilation.access 类型错误")
        if not isinstance(self.memory_space, SpaceIdentity):
            raise TypeError("MemoryQueryCompilation.memory_space 类型错误")
        if self.memory_space.space_type != SPACE_TYPE_MEMORY:
            raise ValueError("Memory query compilation 必须指向 Memory 空间")
        if not isinstance(self.requests, tuple) or any(
                not isinstance(item, MemoryActivationRequest)
                for item in self.requests):
            raise TypeError("MemoryQueryCompilation.requests 类型错误")
        kinds = tuple(item.query_kind.stable_key() for item in self.requests)
        if len(set(kinds)) != len(kinds):
            raise ValueError("Memory query compilation 不得重复 query kind")
        for request in self.requests:
            if (request.memory_space != self.memory_space
                    or request.access != self.access
                    or request.scope != self.current.scope
                    or request.source != self.current.source
                    or request.logical_timestamp != self.current.logical_timestamp):
                raise ValueError("Memory query compilation 含跨输入 request")

    def stable_key(self) -> tuple[int, ...]:
        """返回当前输入和所有输出 request 的完整确定性键。"""
        result = [
            _QUERY_PROTOCOL_VERSION,
            *_packed(self.current.stable_key()),
            *_packed(self.access.stable_key()),
            *_packed(self.memory_space.stable_key()),
            len(self.requests),
        ]
        for request in self.requests:
            result.extend(_packed(request.stable_key()))
        return tuple(result)


class MemoryQueryCompiler:
    """绑定 M-04 aggregate 目标空间的无写入 current-input query compiler。"""

    def __init__(
            self,
            aggregates: MemoryHypothesisAggregateIndex,
            protocol: MemoryQueryProtocol,
            ) -> None:
        """固定目标 aggregate 与注入协议；编译时不读取聚合或历史奖励。"""
        if not isinstance(aggregates, MemoryHypothesisAggregateIndex):
            raise TypeError("aggregates 必须是 MemoryHypothesisAggregateIndex")
        if not isinstance(protocol, MemoryQueryProtocol):
            raise TypeError("protocol 必须是 MemoryQueryProtocol")
        self.aggregates = aggregates
        self.protocol = protocol
        self.memory_space = aggregates.event_log.memory_space_identity

    def compile(
            self,
            current: MemoryCurrentQuery,
            *,
            access: MemoryAccessContext,
            ) -> MemoryQueryCompilation:
        """仅由当前 typed 输入编译分型请求，不读取聚合、WorkMemory 或历史事件。"""
        if not isinstance(current, MemoryCurrentQuery):
            raise TypeError("current 必须是 MemoryCurrentQuery")
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("access 必须是 MemoryAccessContext")
        if not access.can_read(current.scope.owner):
            raise PermissionError("Memory query access 不可读取当前 owner")
        anchors = dict(current.anchors_by_role(self.protocol.roles))
        requests = []
        for definition in self.protocol.definitions:
            reasons = []
            for role in definition.reason_roles:
                values = anchors[role]
                if not values:
                    break
                reasons.append(MemoryQueryReason(role, values))
            else:
                requests.append(MemoryActivationRequest(
                    self.memory_space,
                    access,
                    definition.query_kind,
                    definition.hypothesis_kind,
                    current.scope,
                    current.source,
                    current.logical_timestamp,
                    tuple(reasons),
                    definition.budget,
                ))
        return MemoryQueryCompilation(
            current,
            access,
            self.memory_space,
            tuple(requests),
        )

    def clone_for_aggregates(
            self,
            aggregates: MemoryHypothesisAggregateIndex,
            ) -> "MemoryQueryCompiler":
        """为 V-06 等独立上下文重绑同一 Memory identity 的 aggregate facade。"""
        if not isinstance(aggregates, MemoryHypothesisAggregateIndex):
            raise TypeError("aggregates 必须是 MemoryHypothesisAggregateIndex")
        if aggregates.event_log.memory_space_identity != self.memory_space:
            raise ValueError("Memory query compiler 不得克隆到其他 Memory 空间")
        return MemoryQueryCompiler(aggregates, self.protocol)

    def state_key(self) -> tuple[int, ...]:
        """返回绑定 Memory 空间和协议的不可变状态键，供隔离核验。"""
        return (
            _QUERY_PROTOCOL_VERSION,
            *_packed(self.memory_space.stable_key()),
            *_packed(self.protocol.stable_key()),
        )


__all__ = [
    "MemoryActivationRequest",
    "MemoryCurrentQuery",
    "MemoryQueryCompilation",
    "MemoryQueryCompiler",
    "MemoryQueryDefinition",
    "MemoryQueryProtocol",
    "MemoryQueryReason",
    "MemoryQueryReference",
    "MemoryQueryRoles",
]
