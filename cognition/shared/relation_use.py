"""PH2 Core 关系采用事件的来源化图协议与恢复 owner。

`use_key` 只在完整 query context 内承担幂等路由。实际采用的 Proposition、
Hypothesis、active Evidence、H-04 decision、消费者、用途和恢复状态均保存在
一等 Event 及其图拓扑中；PH3 Memory Use 不经过本模块。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    GraphStatement,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    SourceRef,
    TypedRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    event_identity,
    semantic_source,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int

_USE_EVENT_VERSION = 1
_USE_SNAPSHOT_VERSION = 1


class RelationUseIntegrityError(RuntimeError):
    """Core Use Event 缺边、竞争、篡改或来源元数据不一致。"""


def _strict_key(
        value: tuple[int, ...], *, where: str,
        allow_empty: bool = False) -> tuple[int, ...]:
    """核验开放整数键，并按调用点决定是否允许空键。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{where} 必须是整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长稳定键增加长度边界。"""
    return len(value), *value


def _take(
        values: tuple[int, ...], cursor: int, *, label: str,
        allow_empty: bool = False) -> tuple[tuple[int, ...], int]:
    """读取一个长度前缀字段并返回更新后的游标。"""
    if cursor >= len(values):
        raise RelationUseIntegrityError(f"Use 缺少 {label} 长度")
    size = values[cursor]
    cursor += 1
    if size < 0 or (size == 0 and not allow_empty):
        raise RelationUseIntegrityError(f"Use {label} 长度非法")
    end = cursor + size
    if end > len(values):
        raise RelationUseIntegrityError(f"Use {label} 被截断")
    return values[cursor:end], end


@dataclass(frozen=True)
class RelationUseContext:
    """一次关系采用所属的 query 来源、scope、消费者和用途。"""

    source: SourceRef
    scope: ScopeIdentity
    consumer: ObjectIdentity
    purpose: ObjectIdentity

    def __post_init__(self) -> None:
        """要求 query scope 明确绑定来源，消费者与用途保持一等身份。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("RelationUseContext.source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("RelationUseContext.scope 必须是 ScopeIdentity")
        if self.scope.source != self.source:
            raise ValueError("RelationUseContext.scope 必须绑定 query source")
        for label, identity in (
                ("consumer", self.consumer), ("purpose", self.purpose)):
            if not isinstance(identity, ObjectIdentity):
                raise TypeError(f"RelationUseContext.{label} 必须是 ObjectIdentity")

    def stable_key(self) -> tuple[int, ...]:
        """返回 query 来源、scope、消费者和用途的完整稳定键。"""
        return (
            *_pack(self.source.stable_key()),
            *_pack(self.scope.stable_key()),
            *_pack(self.consumer.stable_key()),
            *_pack(self.purpose.stable_key()),
        )

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "RelationUseContext":
        """从完整稳定键恢复 Use context，拒绝截断和尾随字段。"""
        key = _strict_key(key, where="RelationUseContext.stable_key")
        source_key, cursor = _take(key, 0, label="context source")
        scope_key, cursor = _take(key, cursor, label="context scope")
        consumer_key, cursor = _take(key, cursor, label="context consumer")
        purpose_key, cursor = _take(key, cursor, label="context purpose")
        if cursor != len(key):
            raise RelationUseIntegrityError("Use context 含尾随字段")
        return cls(
            SourceRef.from_stable_key(source_key),
            ScopeIdentity.from_stable_key(scope_key),
            ObjectIdentity.from_stable_key(consumer_key),
            ObjectIdentity.from_stable_key(purpose_key),
        )


@dataclass(frozen=True)
class RelationUseDefinition:
    """一条 Core relation Use Event 的完整领域定义。"""

    use_key: tuple[int, ...]
    context: RelationUseContext
    proposition: ObjectIdentity
    hypothesis: HypothesisKey
    evidence_keys: tuple[tuple[int, ...], ...]
    decision_key: tuple[int, ...]
    read_only_recovered: bool

    def __post_init__(self) -> None:
        """核验 Use 精确引用一个命题及其当前 active H-00/H-04 状态。"""
        _strict_key(self.use_key, where="RelationUseDefinition.use_key")
        if not isinstance(self.context, RelationUseContext):
            raise TypeError("RelationUseDefinition.context 类型错误")
        if (not isinstance(self.proposition, ObjectIdentity)
                or self.proposition.object_kind != OBJECT_PROPOSITION):
            raise ValueError("RelationUseDefinition.proposition 必须是 Proposition")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("RelationUseDefinition.hypothesis 类型错误")
        if not isinstance(self.evidence_keys, tuple) or not self.evidence_keys:
            raise ValueError("RelationUseDefinition 必须保存 active Evidence")
        checked = tuple(
            _strict_key(item, where=f"RelationUseDefinition.evidence[{index}]")
            for index, item in enumerate(self.evidence_keys)
        )
        if len(set(checked)) != len(checked):
            raise ValueError("RelationUseDefinition 不得重复 Evidence")
        _strict_key(self.decision_key, where="RelationUseDefinition.decision_key")
        if type(self.read_only_recovered) is not bool:
            raise TypeError("RelationUseDefinition.read_only_recovered 必须是 bool")
        object.__setattr__(self, "evidence_keys", checked)

    def route_key(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """返回由完整 context 和局部 use_key 组成的幂等路由。"""
        return self.context.stable_key(), self.use_key

    def stable_key(self) -> tuple[int, ...]:
        """返回 Use 的全部来源、采用对象和 H-00/H-04 归因。"""
        result = [
            *_pack(self.use_key),
            *_pack(self.context.stable_key()),
            *_pack(self.proposition.stable_key()),
            *_pack(self.hypothesis.stable_key()),
            len(self.evidence_keys),
        ]
        for key in self.evidence_keys:
            result.extend(_pack(key))
        result.extend(_pack(self.decision_key))
        result.append(1 if self.read_only_recovered else 0)
        return tuple(result)

    @classmethod
    def from_stable_key(cls, key: tuple[int, ...]) -> "RelationUseDefinition":
        """从 Event snapshot 恢复完整 Use 定义。"""
        key = _strict_key(key, where="RelationUseDefinition.stable_key")
        use_key, cursor = _take(key, 0, label="use key")
        context_key, cursor = _take(key, cursor, label="context")
        proposition_key, cursor = _take(key, cursor, label="proposition")
        hypothesis_key, cursor = _take(key, cursor, label="hypothesis")
        if cursor >= len(key):
            raise RelationUseIntegrityError("Use 缺少 Evidence 数量")
        evidence_count = key[cursor]
        cursor += 1
        if evidence_count <= 0:
            raise RelationUseIntegrityError("Use Evidence 数量非法")
        evidence: list[tuple[int, ...]] = []
        for index in range(evidence_count):
            item, cursor = _take(
                key, cursor, label=f"evidence[{index}]")
            evidence.append(item)
        decision_key, cursor = _take(key, cursor, label="decision")
        if cursor + 1 != len(key) or key[cursor] not in (0, 1):
            raise RelationUseIntegrityError("Use 恢复状态缺失或含尾随字段")
        return cls(
            use_key,
            RelationUseContext.from_stable_key(context_key),
            ObjectIdentity.from_stable_key(proposition_key),
            HypothesisKey.from_stable_key(hypothesis_key),
            tuple(evidence),
            decision_key,
            bool(key[cursor]),
        )


@dataclass(frozen=True)
class RelationUseGraphProtocol:
    """由课程注入的 Use Event predicate、状态和命名空间。"""

    event_snapshot: ObjectIdentity
    event_proposition: ObjectIdentity
    event_hypothesis: ObjectIdentity
    event_consumer: ObjectIdentity
    event_purpose: ObjectIdentity
    event_read_state: ObjectIdentity
    live_read_state: ObjectIdentity
    recovered_read_state: ObjectIdentity
    event_namespace_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """要求 predicate、状态均为互异 Concept，命名空间为开放整数键。"""
        identities = self.predicate_identities() + self.state_identities()
        if any(not isinstance(item, ObjectIdentity) for item in identities):
            raise TypeError("RelationUseGraphProtocol identity 类型错误")
        if any(item.object_kind != OBJECT_CONCEPT for item in identities):
            raise ValueError("RelationUseGraphProtocol identity 必须是 Concept")
        if len(set(identities)) != len(identities):
            raise ValueError("RelationUseGraphProtocol identity 必须互不相同")
        _strict_key(
            self.event_namespace_key,
            where="RelationUseGraphProtocol.event_namespace_key",
        )

    def predicate_identities(self) -> tuple[ObjectIdentity, ...]:
        """按协议槽位返回全部 Use Event predicate。"""
        return (
            self.event_snapshot,
            self.event_proposition,
            self.event_hypothesis,
            self.event_consumer,
            self.event_purpose,
            self.event_read_state,
        )

    def state_identities(self) -> tuple[ObjectIdentity, ObjectIdentity]:
        """返回 live-read 与 recovered-read 两个互异状态。"""
        return self.live_read_state, self.recovered_read_state

    def stable_key(self) -> tuple[int, ...]:
        """返回全部一等协议身份和 Event 命名空间。"""
        result: list[int] = []
        for identity in self.predicate_identities() + self.state_identities():
            result.extend(_pack(identity.stable_key()))
        result.extend(_pack(self.event_namespace_key))
        return tuple(result)


@dataclass(frozen=True)
class RelationUseWriteMetadata:
    """Core Use statement 的注入式来源元数据。"""

    provenance_kind: int
    epistemic_origin: int = 0
    content_version: int = 0
    qualifiers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验来源元数据为严格整数，且 provenance 为正。"""
        if not isinstance(self.qualifiers, tuple):
            raise TypeError("RelationUseWriteMetadata.qualifiers 必须是 tuple")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            *self.qualifiers,
            _where="RelationUseWriteMetadata",
        )
        if (type(self.provenance_kind) is not int
                or self.provenance_kind <= 0
                or type(self.epistemic_origin) is not int
                or self.epistemic_origin < 0
                or type(self.content_version) is not int
                or self.content_version < 0
                or any(type(item) is not int for item in self.qualifiers)):
            raise ValueError("RelationUseWriteMetadata 来源元数据非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回来源类型、认识论来源、内容版本和限定键。"""
        return (
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            *_pack(self.qualifiers),
        )


@dataclass(frozen=True)
class MaterializedRelationUse:
    """从 Core 图恢复的 Use 定义、Event、元数据和断言集合。"""

    definition: RelationUseDefinition
    event: ObjectIdentity
    event_ref: TypedRef
    metadata: RelationUseWriteMetadata
    assertion_hashes: tuple[int, ...]


class RelationUseGraph:
    """把完整 relation Use 保存为一等 Event，并从图恢复全部采用历史。"""

    def __init__(
            self, ontology: GraphOntology,
            protocol: RelationUseGraphProtocol) -> None:
        """绑定 Core 图并物化调用方注入的 Use 图协议身份。"""
        if not isinstance(ontology, GraphOntology):
            raise TypeError("RelationUseGraph.ontology 类型错误")
        if not isinstance(protocol, RelationUseGraphProtocol):
            raise TypeError("RelationUseGraph.protocol 类型错误")
        self.ontology = ontology
        self.protocol = protocol
        self._refs = {
            identity: ontology.materialize(identity)
            for identity in (
                *protocol.predicate_identities(),
                *protocol.state_identities(),
            )
        }

    def preflight_many(
            self, definitions: tuple[RelationUseDefinition, ...], *,
            metadata: RelationUseWriteMetadata) -> None:
        """整批零写核验幂等路由、已有 Event 完整性和来源元数据。"""
        self._validate_batch(definitions, metadata)
        routes = tuple(item.route_key() for item in definitions)
        if len(set(routes)) != len(routes):
            raise RelationUseIntegrityError("同批 Core Use 路由不得重复")
        for definition in definitions:
            event = self._event_identity(definition)
            if self.ontology.resolve(event) is not None:
                prior = self.read(event)
                if prior.definition != definition:
                    raise RelationUseIntegrityError(
                        "同一 Core Use 路由已绑定不同采用事实")
                if prior.metadata != metadata:
                    raise RelationUseIntegrityError(
                        "同一 Core Use 路由来源元数据不一致")
                continue

    def materialize_many(
            self, definitions: tuple[RelationUseDefinition, ...], *,
            metadata: RelationUseWriteMetadata,
            ) -> tuple[MaterializedRelationUse, ...]:
        """整批预检后幂等写入 Use 拓扑，snapshot 最后落图。"""
        self.preflight_many(definitions, metadata=metadata)
        for definition in definitions:
            event_identity_value = self._event_identity(definition)
            event = self.ontology.materialize(event_identity_value)
            proposition = self.ontology.materialize(definition.proposition)
            hypothesis = self.ontology.materialize(
                definition.hypothesis.object_identity())
            consumer = self.ontology.materialize(definition.context.consumer)
            purpose = self.ontology.materialize(definition.context.purpose)
            read_state = self._refs[
                self.protocol.recovered_read_state
                if definition.read_only_recovered
                else self.protocol.live_read_state
            ]
            ordinary = (
                (self.protocol.event_proposition, proposition),
                (self.protocol.event_hypothesis, hypothesis),
                (self.protocol.event_consumer, consumer),
                (self.protocol.event_purpose, purpose),
                (self.protocol.event_read_state, read_state),
            )
            for predicate, target in ordinary:
                self._relate(
                    predicate, event, target,
                    scope=definition.context.scope,
                    metadata=metadata,
                    qualifiers=metadata.qualifiers,
                )
            self._relate(
                self.protocol.event_snapshot,
                event,
                event,
                scope=definition.context.scope,
                metadata=metadata,
                qualifiers=self._snapshot_qualifiers(definition, metadata),
            )
        return tuple(self.read(self._event_identity(item)) for item in definitions)

    def read(self, event: ObjectIdentity) -> MaterializedRelationUse:
        """从 Event snapshot 和全部 typed 边双向恢复一次 Core Use。"""
        if (not isinstance(event, ObjectIdentity)
                or event.object_kind != OBJECT_EVENT):
            raise ValueError("RelationUseGraph.read 需要 Event 身份")
        event_ref = self.ontology.resolve(event)
        if event_ref is None:
            raise RelationUseIntegrityError("Core Use Event 尚未物化")
        snapshots = self.ontology.statements(
            predicate=self._refs[self.protocol.event_snapshot],
            subject=event_ref,
        )
        if (len(snapshots) != 1
                or snapshots[0].object != event_ref):
            raise RelationUseIntegrityError(
                "Core Use Event snapshot 必须是唯一自指 statement")
        definition, metadata = self._parse_snapshot(snapshots[0])
        if self._event_identity(definition) != event:
            raise RelationUseIntegrityError("Core Use Event 路由与 snapshot 不一致")
        expected = (
            (self.protocol.event_proposition, definition.proposition),
            (self.protocol.event_hypothesis,
             definition.hypothesis.object_identity()),
            (self.protocol.event_consumer, definition.context.consumer),
            (self.protocol.event_purpose, definition.context.purpose),
            (
                self.protocol.event_read_state,
                self.protocol.recovered_read_state
                if definition.read_only_recovered
                else self.protocol.live_read_state,
            ),
        )
        statements = [snapshots[0]]
        for predicate, target in expected:
            rows = self.ontology.statements(
                predicate=self._refs[predicate],
                subject=event_ref,
            )
            target_ref = self.ontology.resolve(target)
            if len(rows) != 1 or target_ref is None or rows[0].object != target_ref:
                raise RelationUseIntegrityError(
                    "Core Use Event typed 拓扑缺失或存在竞争端点")
            self._validate_statement(
                rows[0], definition.context.scope, metadata,
                metadata.qualifiers,
            )
            statements.append(rows[0])
        self._validate_statement(
            snapshots[0], definition.context.scope, metadata,
            self._snapshot_qualifiers(definition, metadata),
        )
        return MaterializedRelationUse(
            definition,
            event,
            event_ref,
            metadata,
            tuple(sorted(item.assertion_hash for item in statements)),
        )

    def history(self) -> tuple[MaterializedRelationUse, ...]:
        """按 Event 完整身份恢复当前协议命名空间内的全部 Use。"""
        rows = self.ontology.statements(
            predicate=self._refs[self.protocol.event_snapshot])
        events: dict[ObjectIdentity, ObjectIdentity] = {}
        for row in rows:
            identity = self.ontology.identity_of(row.subject)
            if identity.object_kind != OBJECT_EVENT:
                raise RelationUseIntegrityError("Use snapshot 主语不是 Event")
            namespace, _, _ = self._parse_event_route(identity)
            if namespace != self.protocol.event_namespace_key:
                raise RelationUseIntegrityError(
                    "Use snapshot Event 使用了其他协议命名空间")
            events[identity] = identity
        return tuple(
            self.read(events[key])
            for key in sorted(events, key=ObjectIdentity.stable_key)
        )

    def clone_for_ontology(self, ontology: GraphOntology) -> "RelationUseGraph":
        """在已复制的 Core 图上重建同一 Use 协议 facade。"""
        return RelationUseGraph(ontology, self.protocol)

    def _event_identity(self, definition: RelationUseDefinition) -> ObjectIdentity:
        """以来源、完整 context 和局部 use_key 构造一等 Event 路由身份。"""
        return event_identity(
            definition.context.source,
            (
                _USE_EVENT_VERSION,
                *_pack(self.protocol.event_namespace_key),
                *_pack(definition.context.stable_key()),
                *_pack(definition.use_key),
            ),
        )

    @staticmethod
    def _event_key(event: ObjectIdentity) -> tuple[int, ...]:
        """从来源化 Event 身份中取出开放声明键。"""
        source = semantic_source(event)
        cursor = 1 + len(source.stable_key())
        key, cursor = _take(
            event.components, cursor, label="event declaration key")
        if cursor != len(event.components):
            raise RelationUseIntegrityError("Relation Use Event 身份含尾随字段")
        return _strict_key(key, where="RelationUse Event key")

    def _parse_event_route(
            self, event: ObjectIdentity,
            ) -> tuple[tuple[int, ...], RelationUseContext, tuple[int, ...]]:
        """解析 Event 的协议命名空间、完整 context 和局部 use_key。"""
        key = self._event_key(event)
        if key[0] != _USE_EVENT_VERSION:
            raise RelationUseIntegrityError("Relation Use Event 版本未注册")
        namespace, cursor = _take(key, 1, label="event namespace")
        context_key, cursor = _take(key, cursor, label="event context")
        use_key, cursor = _take(key, cursor, label="event use key")
        if cursor != len(key):
            raise RelationUseIntegrityError("Relation Use Event 含尾随字段")
        context = RelationUseContext.from_stable_key(context_key)
        if semantic_source(event) != context.source:
            raise RelationUseIntegrityError("Relation Use Event 来源与 context 不一致")
        return namespace, context, use_key

    def _parse_snapshot(
            self, statement: GraphStatement,
            ) -> tuple[RelationUseDefinition, RelationUseWriteMetadata]:
        """从 snapshot qualifiers 恢复 Use 定义和外部来源限定键。"""
        qualifiers = statement.assertion.qualifiers
        if not qualifiers or qualifiers[0] != _USE_SNAPSHOT_VERSION:
            raise RelationUseIntegrityError("Core Use snapshot 版本未注册")
        definition_key, cursor = _take(
            qualifiers, 1, label="snapshot definition")
        metadata_qualifiers, cursor = _take(
            qualifiers, cursor, label="snapshot metadata qualifiers",
            allow_empty=True,
        )
        if cursor != len(qualifiers):
            raise RelationUseIntegrityError("Core Use snapshot 含尾随字段")
        assertion = statement.assertion
        definition = RelationUseDefinition.from_stable_key(definition_key)
        metadata = RelationUseWriteMetadata(
            assertion.provenance_kind,
            assertion.epistemic_origin,
            assertion.content_version,
            metadata_qualifiers,
        )
        return definition, metadata

    @staticmethod
    def _snapshot_qualifiers(
            definition: RelationUseDefinition,
            metadata: RelationUseWriteMetadata) -> tuple[int, ...]:
        """把完整 Use snapshot 与调用方限定键编码为一份图内载荷。"""
        return (
            _USE_SNAPSHOT_VERSION,
            *_pack(definition.stable_key()),
            *_pack(metadata.qualifiers),
        )

    def _relate(
            self, predicate: ObjectIdentity, subject: TypedRef,
            target: TypedRef, *, scope: ScopeIdentity,
            metadata: RelationUseWriteMetadata,
            qualifiers: tuple[int, ...]) -> GraphStatement:
        """按统一来源元数据追加一条 Use Event statement。"""
        return self.ontology.relate(
            self._refs[predicate],
            subject,
            target,
            scope=scope,
            provenance_kind=metadata.provenance_kind,
            epistemic_origin=metadata.epistemic_origin,
            content_version=metadata.content_version,
            qualifiers=qualifiers,
        )

    @staticmethod
    def _validate_statement(
            statement: GraphStatement, scope: ScopeIdentity,
            metadata: RelationUseWriteMetadata,
            qualifiers: tuple[int, ...]) -> None:
        """核验一条 Use statement 的 scope、来源和限定载荷。"""
        assertion = statement.assertion
        if (
                assertion.scope != scope
                or assertion.provenance_kind != metadata.provenance_kind
                or assertion.epistemic_origin != metadata.epistemic_origin
                or assertion.content_version != metadata.content_version
                or assertion.qualifiers != qualifiers):
            raise RelationUseIntegrityError("Core Use statement 来源元数据不一致")

    @staticmethod
    def _validate_batch(
            definitions: tuple[RelationUseDefinition, ...],
            metadata: RelationUseWriteMetadata) -> None:
        """核验批次只包含完整 Use 定义和单一来源元数据。"""
        if not isinstance(definitions, tuple) or not definitions or any(
                not isinstance(item, RelationUseDefinition)
                for item in definitions):
            raise TypeError("RelationUseGraph definitions 必须是非空定义 tuple")
        if not isinstance(metadata, RelationUseWriteMetadata):
            raise TypeError("RelationUseGraph metadata 类型错误")


class RelationUseOwner:
    """持有一个 Core Use 图协议，并把进程索引完全派生自图历史。"""

    def __init__(
            self, graph: RelationUseGraph,
            metadata: RelationUseWriteMetadata) -> None:
        """恢复既有 Use 历史，拒绝同一路由出现竞争定义。"""
        if not isinstance(graph, RelationUseGraph):
            raise TypeError("RelationUseOwner.graph 类型错误")
        if not isinstance(metadata, RelationUseWriteMetadata):
            raise TypeError("RelationUseOwner.metadata 类型错误")
        self.graph = graph
        self.metadata = metadata
        self._uses: dict[
            tuple[tuple[int, ...], tuple[int, ...]], MaterializedRelationUse
        ] = {}
        for item in graph.history():
            if item.metadata != metadata:
                raise RelationUseIntegrityError("恢复的 Core Use 来源元数据不一致")
            route = item.definition.route_key()
            prior = self._uses.get(route)
            if prior is not None and prior != item:
                raise RelationUseIntegrityError("恢复的 Core Use 路由发生竞争")
            self._uses[route] = item

    def append_many(
            self, definitions: tuple[RelationUseDefinition, ...],
            ) -> tuple[MaterializedRelationUse, ...]:
        """整批追加或幂等恢复 Use，并仅在写后核验成功时更新派生索引。"""
        materialized = self.graph.materialize_many(
            definitions, metadata=self.metadata)
        for item in materialized:
            self._uses[item.definition.route_key()] = item
        return materialized

    def history(self) -> tuple[MaterializedRelationUse, ...]:
        """按完整 Event 身份返回当前 owner 的全部恢复历史。"""
        return tuple(sorted(
            self._uses.values(),
            key=lambda item: item.event.stable_key(),
        ))

    def clone_for_ontology(self, ontology: GraphOntology) -> "RelationUseOwner":
        """从已复制 Core 图恢复独立 owner，不复制宿主可变字典。"""
        return RelationUseOwner(
            self.graph.clone_for_ontology(ontology),
            self.metadata,
        )

    def state_key(self) -> tuple:
        """返回协议、来源元数据和全部图内 Use 的确定状态。"""
        return (
            self.graph.protocol.stable_key(),
            self.metadata.stable_key(),
            tuple(item.definition.stable_key() for item in self.history()),
        )


__all__ = [
    "MaterializedRelationUse",
    "RelationUseContext",
    "RelationUseDefinition",
    "RelationUseGraph",
    "RelationUseGraphProtocol",
    "RelationUseIntegrityError",
    "RelationUseOwner",
    "RelationUseWriteMetadata",
]
