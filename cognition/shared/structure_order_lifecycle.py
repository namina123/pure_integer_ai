"""S-07 结构顺序约束的图内 append-only 生命周期。

生命周期状态、事件 kind 和 predicate 均由调用方以一等 Concept 注入。当前状态只由
来源化 Event 链投影，旧结构定义和旧事件不会因降级或替代而被删除。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import GraphStatement
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_EVENT,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    TypedRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    event_identity,
    semantic_source,
)
from pure_integer_ai.cognition.shared.structure_order import (
    MaterializedStructureOrderConstraint,
    StructureOrderGraph,
    StructureOrderTopologyError,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_EVENT_KEY_VERSION = 1
_SEMANTIC_IDENTITY_VERSION = 1
_SOURCE_KEY_SIZE = 11


class StructureOrderLifecycleError(RuntimeError):
    """生命周期事件拓扑、转换链或完整身份不一致。"""


def _integer_key(value: tuple[int, ...], *, where: str,
                 allow_empty: bool = False) -> tuple[int, ...]:
    """校验开放整数键，必要时允许空键表示无 resolver decision。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{where} 必须是整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长完整键增加长度前缀。"""
    return len(value), *value


def _take(
        values: tuple[int, ...], cursor: int, *, label: str,
        allow_empty: bool = False) -> tuple[tuple[int, ...], int]:
    """从事件键读取一个长度前缀段并返回新游标。"""
    if cursor >= len(values):
        raise ValueError(f"生命周期事件缺少 {label} 长度")
    size = values[cursor]
    cursor += 1
    if size < 0 or (size == 0 and not allow_empty):
        raise ValueError(f"生命周期事件 {label} 长度非法")
    if cursor + size > len(values):
        raise ValueError(f"生命周期事件 {label} 被截断")
    return values[cursor:cursor + size], cursor + size


@dataclass(frozen=True)
class StructureOrderLifecycleProtocol:
    """图 predicate、状态、事件 kind 和事件命名空间的注入协议。"""

    event_constraint: TypedRef
    event_kind: TypedRef
    event_from_state: TypedRef
    event_to_state: TypedRef
    event_hypothesis: TypedRef
    event_replacement: TypedRef
    inactive_state: ObjectIdentity
    active_state: ObjectIdentity
    superseded_state: ObjectIdentity
    promotion_kind: ObjectIdentity
    demotion_kind: ObjectIdentity
    supersede_kind: ObjectIdentity
    event_namespace_key: tuple[int, ...]

    def __post_init__(self) -> None:
        refs = self.predicate_refs()
        if any(not isinstance(item, TypedRef) for item in refs):
            raise TypeError("lifecycle predicate 必须全部是 TypedRef")
        if len({item.stable_key() for item in refs}) != len(refs):
            raise ValueError("lifecycle predicate 槽位必须互不相同")
        identities = self.state_identities() + self.kind_identities()
        if any(not isinstance(item, ObjectIdentity) for item in identities):
            raise TypeError("lifecycle state/kind 必须是 ObjectIdentity")
        if any(item.object_kind != OBJECT_CONCEPT for item in identities):
            raise ValueError("lifecycle state/kind 必须是一等 Concept")
        if len({item.stable_key() for item in identities}) != len(identities):
            raise ValueError("lifecycle state 和 event kind 必须互不相同")
        if len({item.owner for item in identities}) != 1:
            raise ValueError("lifecycle state/kind owner 必须一致")
        _integer_key(
            self.event_namespace_key,
            where="StructureOrderLifecycleProtocol.event_namespace_key",
        )

    def predicate_refs(self) -> tuple[TypedRef, ...]:
        """返回 Event 拓扑使用的全部 predicate。"""
        return (
            self.event_constraint,
            self.event_kind,
            self.event_from_state,
            self.event_to_state,
            self.event_hypothesis,
            self.event_replacement,
        )

    def state_identities(self) -> tuple[ObjectIdentity, ...]:
        """返回 inactive、active、superseded 三个图内状态。"""
        return (
            self.inactive_state,
            self.active_state,
            self.superseded_state,
        )

    def kind_identities(self) -> tuple[ObjectIdentity, ...]:
        """返回 promotion、demotion、supersede 三个图内事件 kind。"""
        return (
            self.promotion_kind,
            self.demotion_kind,
            self.supersede_kind,
        )

    def expected_kind(
            self, from_state: ObjectIdentity,
            to_state: ObjectIdentity) -> ObjectIdentity:
        """把通用生命周期转换映射到注入的一等事件 kind。"""
        if from_state == self.inactive_state and to_state == self.active_state:
            return self.promotion_kind
        if from_state == self.active_state and to_state == self.inactive_state:
            return self.demotion_kind
        if (from_state == self.active_state
                and to_state == self.superseded_state):
            return self.supersede_kind
        raise ValueError("结构顺序生命周期转换不合法")


@dataclass(frozen=True)
class StructureOrderLifecycleEvent:
    """一个保存完整 Evidence/decision provenance 的不可变图内转换。"""

    event: ObjectIdentity
    constraint: ObjectIdentity
    event_kind: ObjectIdentity
    from_state: ObjectIdentity
    to_state: ObjectIdentity
    hypothesis: HypothesisKey
    evidence_keys: tuple[tuple[int, ...], ...]
    decision_key: tuple[int, ...]
    timestamp_seq: int
    replacement: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        for name, value, kind in (
                ("event", self.event, OBJECT_EVENT),
                ("constraint", self.constraint, OBJECT_STRUCTURE_CONCEPT),
                ("event_kind", self.event_kind, OBJECT_CONCEPT),
                ("from_state", self.from_state, OBJECT_CONCEPT),
                ("to_state", self.to_state, OBJECT_CONCEPT)):
            if not isinstance(value, ObjectIdentity):
                raise TypeError(f"{name} 必须是 ObjectIdentity")
            if value.object_kind != kind:
                raise ValueError(f"{name} 对象类型不匹配")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("hypothesis 必须是 HypothesisKey")
        if not isinstance(self.evidence_keys, tuple) or not self.evidence_keys:
            raise ValueError("lifecycle event 必须保存非空 Evidence 完整键")
        checked_evidence = tuple(
            _integer_key(item, where=f"evidence_keys[{index}]")
            for index, item in enumerate(self.evidence_keys)
        )
        if len(set(checked_evidence)) != len(checked_evidence):
            raise ValueError("lifecycle event 不得重复 Evidence 键")
        _integer_key(self.decision_key, where="decision_key")
        assert_int(self.timestamp_seq, _where="timestamp_seq")
        if type(self.timestamp_seq) is not int or self.timestamp_seq < 0:
            raise ValueError("timestamp_seq 必须为非负严格整数")
        if self.replacement is not None:
            if not isinstance(self.replacement, ObjectIdentity):
                raise TypeError("replacement 必须是 ObjectIdentity")
            if self.replacement.object_kind != OBJECT_STRUCTURE_CONCEPT:
                raise ValueError("replacement 必须是 StructureConcept")
            if self.replacement == self.constraint:
                raise ValueError("constraint 不得 supersede 为自身")
        if semantic_source(self.event) != self.hypothesis.observation:
            raise ValueError("lifecycle Event 来源与 H-06 Hypothesis 不一致")
        if any(item.owner != self.constraint.owner for item in (
                self.event_kind, self.from_state, self.to_state)):
            raise ValueError("lifecycle state/kind 与 constraint owner 不一致")
        if self.constraint.owner != self.hypothesis.observation.owner:
            raise ValueError("constraint owner 与 aggregate source 不一致")


@dataclass(frozen=True)
class MaterializedStructureOrderLifecycleEvent:
    """从图中恢复的生命周期 Event 和全部 provenance statement。"""

    definition: StructureOrderLifecycleEvent
    event: TypedRef
    statements: tuple[GraphStatement, ...]


@dataclass(frozen=True)
class StructureOrderProjection:
    """一个 constraint 的当前状态、完整历史和可选 replacement。"""

    constraint: MaterializedStructureOrderConstraint
    state: ObjectIdentity
    history: tuple[MaterializedStructureOrderLifecycleEvent, ...]
    replacement: ObjectIdentity | None


class StructureOrderLifecycleGraph:
    """写入生命周期 Event，并从严格递增事件链投影当前 constraint 状态。"""

    def __init__(
            self, order_graph: StructureOrderGraph,
            protocol: StructureOrderLifecycleProtocol) -> None:
        if not isinstance(order_graph, StructureOrderGraph):
            raise TypeError("order_graph 必须是 StructureOrderGraph")
        if not isinstance(protocol, StructureOrderLifecycleProtocol):
            raise TypeError("protocol 必须是 StructureOrderLifecycleProtocol")
        self._order_graph = order_graph
        self._ontology = order_graph.ontology
        self.protocol = protocol
        self._validate_protocol_graph()

    @property
    def order_graph(self) -> StructureOrderGraph:
        """返回生命周期 facade 绑定的结构顺序图。"""
        return self._order_graph

    def make_event(
            self, constraint: ObjectIdentity, *,
            event_kind: ObjectIdentity,
            from_state: ObjectIdentity,
            to_state: ObjectIdentity,
            hypothesis: HypothesisKey,
            evidence_keys: tuple[tuple[int, ...], ...],
            decision_key: tuple[int, ...],
            timestamp_seq: int,
            replacement: ObjectIdentity | None = None,
            ) -> StructureOrderLifecycleEvent:
        """从完整转换内容构造来源化 Event 身份，不以 hash 替代内容。"""
        canonical_evidence = tuple(sorted(evidence_keys))
        event_key = self._event_key(
            constraint,
            event_kind=event_kind,
            from_state=from_state,
            to_state=to_state,
            hypothesis=hypothesis,
            evidence_keys=canonical_evidence,
            decision_key=decision_key,
            timestamp_seq=timestamp_seq,
            replacement=replacement,
        )
        definition = StructureOrderLifecycleEvent(
            event_identity(hypothesis.observation, event_key),
            constraint,
            event_kind,
            from_state,
            to_state,
            hypothesis,
            canonical_evidence,
            decision_key,
            timestamp_seq,
            replacement,
        )
        self._validate_event(definition)
        return definition

    def append(
            self, definition: StructureOrderLifecycleEvent, *,
            scope: ScopeIdentity, provenance_kind: int,
            epistemic_origin: int = 0, content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> MaterializedStructureOrderLifecycleEvent:
        """预检完整转换链和 Event 拓扑后幂等追加，不删除旧结构 statement。"""
        self._validate_event(definition)
        self._validate_metadata(
            definition,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )
        constraint_ref = self._ontology.resolve(definition.constraint)
        if constraint_ref is None:
            raise StructureOrderLifecycleError("constraint 尚未物化定义")
        constraint = self._order_graph.read_constraint(constraint_ref)
        if constraint.definition.hypothesis != definition.hypothesis:
            raise StructureOrderLifecycleError(
                "lifecycle Hypothesis 与 constraint provenance 不一致")

        existing_ref = self._ontology.resolve(definition.event)
        if existing_ref is not None:
            existing_statements = self._event_statements(existing_ref)
            if existing_statements:
                restored = self.read_event(existing_ref)
                if restored.definition != definition:
                    raise StructureOrderLifecycleError(
                        "同一 lifecycle Event 身份绑定了不同拓扑")
                self._require_replay_metadata(
                    restored,
                    scope=scope,
                    provenance_kind=provenance_kind,
                    epistemic_origin=epistemic_origin,
                    content_version=content_version,
                    qualifiers=qualifiers,
                )
                return restored

        projection = self.project(constraint_ref)
        if projection.state != definition.from_state:
            raise StructureOrderLifecycleError(
                "lifecycle from_state 与当前投影不一致")
        if projection.history and definition.timestamp_seq <= (
                projection.history[-1].definition.timestamp_seq):
            raise StructureOrderLifecycleError("lifecycle 逻辑序必须严格递增")
        if definition.replacement is not None:
            replacement_ref = self._ontology.resolve(definition.replacement)
            if replacement_ref is None:
                raise StructureOrderLifecycleError("replacement constraint 尚未定义")
            replacement_projection = self.project(replacement_ref)
            if replacement_projection.state != self.protocol.active_state:
                raise StructureOrderLifecycleError(
                    "replacement constraint 必须已处于 active 状态")
            replacement_definition = replacement_projection.constraint.definition
            if replacement_definition.structure != constraint.definition.structure:
                raise StructureOrderLifecycleError(
                    "replacement constraint 不得跨 StructureConcept")
            old_hypothesis = definition.hypothesis
            new_hypothesis = replacement_definition.hypothesis
            if (
                    old_hypothesis.hypothesis_kind
                    != new_hypothesis.hypothesis_kind
                    or old_hypothesis.competition_key
                    != new_hypothesis.competition_key
                    or old_hypothesis.scope != new_hypothesis.scope
                    or old_hypothesis.observation
                    != new_hypothesis.observation):
                raise StructureOrderLifecycleError(
                    "replacement constraint 必须来自同一 H-00 竞争组")

        event = self._ontology.materialize(definition.event)
        metadata = (
            scope,
            provenance_kind,
            epistemic_origin,
            content_version,
            qualifiers,
        )
        targets = (
            (self.protocol.event_constraint, definition.constraint),
            (self.protocol.event_kind, definition.event_kind),
            (self.protocol.event_from_state, definition.from_state),
            (self.protocol.event_to_state, definition.to_state),
            (self.protocol.event_hypothesis,
             definition.hypothesis.object_identity()),
        )
        for predicate, identity in targets:
            self._relate(
                predicate,
                event,
                self._ontology.materialize(identity),
                metadata,
            )
        if definition.replacement is not None:
            self._relate(
                self.protocol.event_replacement,
                event,
                self._ontology.materialize(definition.replacement),
                metadata,
            )
        restored = self.read_event(event)
        if restored.definition != definition:
            raise StructureOrderLifecycleError("写后恢复的 lifecycle Event 不一致")
        return restored

    def read_event(
            self, event: TypedRef,
            ) -> MaterializedStructureOrderLifecycleEvent:
        """从 Event identity 与六类图槽双向恢复完整生命周期事件。"""
        event_identity_value = self._ontology.identity_of(event)
        definition = self._parse_event_identity(event_identity_value)
        singleton_specs = (
            (self.protocol.event_constraint, definition.constraint,
             "event constraint"),
            (self.protocol.event_kind, definition.event_kind, "event kind"),
            (self.protocol.event_from_state, definition.from_state,
             "event from state"),
            (self.protocol.event_to_state, definition.to_state,
             "event to state"),
            (self.protocol.event_hypothesis,
             definition.hypothesis.object_identity(), "event Hypothesis"),
        )
        statements: list[GraphStatement] = []
        for predicate, expected, label in singleton_specs:
            target, group = self._single_target(predicate, event, label=label)
            if self._ontology.identity_of(target) != expected:
                raise StructureOrderLifecycleError(f"{label} 与 Event identity 不一致")
            statements.extend(group)
        replacement_targets, replacement_statements = self._multi_targets(
            self.protocol.event_replacement, event)
        if definition.replacement is None:
            if replacement_targets:
                raise StructureOrderLifecycleError(
                    "非 supersede Event 不得带 replacement")
        elif (
                len(replacement_targets) != 1
                or self._ontology.identity_of(replacement_targets[0])
                != definition.replacement):
            raise StructureOrderLifecycleError(
                "replacement 拓扑与 Event identity 不一致")
        statements.extend(replacement_statements)
        return MaterializedStructureOrderLifecycleEvent(
            definition,
            event,
            tuple(sorted(statements, key=lambda item: item.assertion_hash)),
        )

    def history(
            self, constraint: TypedRef,
            ) -> tuple[MaterializedStructureOrderLifecycleEvent, ...]:
        """读取一个 constraint 的完整事件历史并拒绝逻辑序歧义。"""
        identity = self._ontology.identity_of(constraint)
        if identity.object_kind != OBJECT_STRUCTURE_CONCEPT:
            raise ValueError("constraint 必须是 StructureConcept")
        links = self._ontology.statements(
            predicate=self.protocol.event_constraint,
            object_ref=constraint,
        )
        event_refs: dict[ObjectIdentity, TypedRef] = {}
        for link in links:
            event_identity_value = self._ontology.identity_of(link.subject)
            prior = event_refs.get(event_identity_value)
            if prior is not None and prior != link.subject:
                raise StructureOrderLifecycleError(
                    "同一 Event 身份映射到多个图节点")
            event_refs[event_identity_value] = link.subject
        restored = tuple(
            self.read_event(event_refs[key])
            for key in sorted(event_refs, key=ObjectIdentity.stable_key)
        )
        ordered = tuple(sorted(
            restored,
            key=lambda item: (
                item.definition.timestamp_seq,
                item.definition.event.stable_key(),
            ),
        ))
        timestamps = tuple(item.definition.timestamp_seq for item in ordered)
        if len(set(timestamps)) != len(timestamps):
            raise StructureOrderLifecycleError(
                "同一 constraint 存在相同逻辑序的竞争 lifecycle Event")
        return ordered

    def project(self, constraint: TypedRef) -> StructureOrderProjection:
        """顺序执行完整 Event 链，派生当前状态和 terminal replacement。"""
        materialized = self._order_graph.read_constraint(constraint)
        history = self.history(constraint)
        state = self.protocol.inactive_state
        replacement = None
        for item in history:
            event = item.definition
            if event.constraint != materialized.definition.constraint:
                raise StructureOrderLifecycleError(
                    "history 混入其他 constraint Event")
            if event.hypothesis != materialized.definition.hypothesis:
                raise StructureOrderLifecycleError(
                    "history Event Hypothesis 与 constraint 不一致")
            if event.from_state != state:
                raise StructureOrderLifecycleError(
                    "lifecycle Event 链 from_state 不连续")
            if event.event_kind != self.protocol.expected_kind(
                    event.from_state, event.to_state):
                raise StructureOrderLifecycleError(
                    "lifecycle Event kind 与转换不一致")
            if event.to_state == self.protocol.superseded_state:
                if event.replacement is None:
                    raise StructureOrderLifecycleError(
                        "supersede Event 缺少 replacement")
                replacement = event.replacement
            elif event.replacement is not None:
                raise StructureOrderLifecycleError(
                    "非 supersede Event 带有 replacement")
            state = event.to_state
        return StructureOrderProjection(
            materialized,
            state,
            history,
            replacement,
        )

    def active_constraints(
            self, structure: TypedRef,
            ) -> tuple[StructureOrderProjection, ...]:
        """返回结构当前 active 的 typed constraint，不读取旧序列或统计索引。"""
        materialized = self._order_graph.read_structure(structure)
        projections = tuple(
            self.project(item.constraint)
            for item in materialized.constraints
        )
        return tuple(sorted(
            (item for item in projections
             if item.state == self.protocol.active_state),
            key=lambda item: item.constraint.definition.constraint.stable_key(),
        ))

    def _validate_protocol_graph(self) -> None:
        """核验 predicate、状态和事件 kind 均为当前图内一等 Concept。"""
        all_predicates = (
            *self._order_graph.predicates.refs(),
            *self.protocol.predicate_refs(),
        )
        if len({item.stable_key() for item in all_predicates}) != len(
                all_predicates):
            raise ValueError("lifecycle predicate 不得复用结构顺序定义槽")
        for ref in self.protocol.predicate_refs():
            if self._ontology.identity_of(ref).object_kind != OBJECT_CONCEPT:
                raise ValueError("lifecycle predicate 必须是 Concept")
        for identity in (
                *self.protocol.state_identities(),
                *self.protocol.kind_identities()):
            ref = self._ontology.resolve(identity)
            if ref is None:
                raise ValueError("lifecycle state/kind 必须先物化到当前图")
            if self._ontology.identity_of(ref) != identity:
                raise ValueError("lifecycle state/kind 图身份不一致")

    def _validate_event(self, event: StructureOrderLifecycleEvent) -> None:
        """核验事件转换、owner、replacement 和完整 identity codec。"""
        if not isinstance(event, StructureOrderLifecycleEvent):
            raise TypeError("event 必须是 StructureOrderLifecycleEvent")
        expected_kind = self.protocol.expected_kind(
            event.from_state, event.to_state)
        if event.event_kind != expected_kind:
            raise ValueError("lifecycle event kind 与 from/to 状态不一致")
        known_identities = {
            *self.protocol.state_identities(),
            *self.protocol.kind_identities(),
        }
        if event.event_kind not in known_identities:
            raise ValueError("lifecycle event kind 未由当前协议注入")
        if event.from_state not in known_identities:
            raise ValueError("lifecycle from_state 未由当前协议注入")
        if event.to_state not in known_identities:
            raise ValueError("lifecycle to_state 未由当前协议注入")
        if event.to_state == self.protocol.superseded_state:
            if event.replacement is None:
                raise ValueError("supersede Event 必须指定 replacement")
        elif event.replacement is not None:
            raise ValueError("非 supersede Event 不得指定 replacement")
        expected_key = self._event_key(
            event.constraint,
            event_kind=event.event_kind,
            from_state=event.from_state,
            to_state=event.to_state,
            hypothesis=event.hypothesis,
            evidence_keys=event.evidence_keys,
            decision_key=event.decision_key,
            timestamp_seq=event.timestamp_seq,
            replacement=event.replacement,
        )
        expected_identity = event_identity(
            event.hypothesis.observation, expected_key)
        if event.event != expected_identity:
            raise ValueError("lifecycle Event identity 未完整编码转换内容")

    def _event_key(
            self, constraint: ObjectIdentity, *,
            event_kind: ObjectIdentity,
            from_state: ObjectIdentity,
            to_state: ObjectIdentity,
            hypothesis: HypothesisKey,
            evidence_keys: tuple[tuple[int, ...], ...],
            decision_key: tuple[int, ...],
            timestamp_seq: int,
            replacement: ObjectIdentity | None,
            ) -> tuple[int, ...]:
        """编码可逆事件声明键，所有 hash 之外的完整 provenance 均保留。"""
        replacement_key = (
            () if replacement is None else replacement.stable_key())
        return (
            _EVENT_KEY_VERSION,
            *_pack(self.protocol.event_namespace_key),
            *_pack(constraint.stable_key()),
            *_pack(event_kind.stable_key()),
            *_pack(from_state.stable_key()),
            *_pack(to_state.stable_key()),
            *_pack(hypothesis.stable_key()),
            len(evidence_keys),
            *(value for key in evidence_keys for value in _pack(key)),
            *_pack(decision_key),
            timestamp_seq,
            *_pack(replacement_key),
        )

    def _parse_event_identity(
            self, identity: ObjectIdentity) -> StructureOrderLifecycleEvent:
        """解析 Event 声明键并重建所有一等端点和 provenance 完整键。"""
        if identity.object_kind != OBJECT_EVENT:
            raise StructureOrderLifecycleError("lifecycle 引用不是 Event")
        source = semantic_source(identity)
        components = identity.components
        cursor = 1 + _SOURCE_KEY_SIZE
        declaration, cursor = _take(
            components, cursor, label="event declaration")
        if cursor != len(components):
            raise StructureOrderLifecycleError("Event identity 含尾随数据")
        values = declaration
        if not values or values[0] != _EVENT_KEY_VERSION:
            raise StructureOrderLifecycleError("lifecycle Event key 版本非法")
        cursor = 1
        namespace, cursor = _take(values, cursor, label="namespace")
        if namespace != self.protocol.event_namespace_key:
            raise StructureOrderLifecycleError("Event 不属于当前 lifecycle 协议")
        constraint_key, cursor = _take(values, cursor, label="constraint")
        kind_key, cursor = _take(values, cursor, label="event kind")
        from_key, cursor = _take(values, cursor, label="from state")
        to_key, cursor = _take(values, cursor, label="to state")
        hypothesis_key, cursor = _take(values, cursor, label="Hypothesis")
        if cursor >= len(values):
            raise StructureOrderLifecycleError("Event 缺少 Evidence 数量")
        evidence_count = values[cursor]
        cursor += 1
        if evidence_count <= 0:
            raise StructureOrderLifecycleError("Event Evidence 数量非法")
        evidence: list[tuple[int, ...]] = []
        for index in range(evidence_count):
            key, cursor = _take(
                values, cursor, label=f"Evidence[{index}]")
            evidence.append(key)
        decision_key, cursor = _take(values, cursor, label="decision")
        if cursor >= len(values):
            raise StructureOrderLifecycleError("Event 缺少 timestamp_seq")
        timestamp_seq = values[cursor]
        cursor += 1
        replacement_key, cursor = _take(
            values,
            cursor,
            label="replacement",
            allow_empty=True,
        )
        if cursor != len(values):
            raise StructureOrderLifecycleError("lifecycle Event key 含尾随数据")
        try:
            constraint = ObjectIdentity.from_stable_key(constraint_key)
            event_kind_value = ObjectIdentity.from_stable_key(kind_key)
            from_state = ObjectIdentity.from_stable_key(from_key)
            to_state = ObjectIdentity.from_stable_key(to_key)
            hypothesis = HypothesisKey.from_stable_key(hypothesis_key)
            replacement = (
                None if not replacement_key
                else ObjectIdentity.from_stable_key(replacement_key)
            )
        except (TypeError, ValueError) as exc:
            raise StructureOrderLifecycleError(
                "lifecycle Event 内嵌完整身份无法恢复") from exc
        if hypothesis.observation != source:
            raise StructureOrderLifecycleError(
                "Event SourceRef 与内嵌 Hypothesis 不一致")
        restored = StructureOrderLifecycleEvent(
            identity,
            constraint,
            event_kind_value,
            from_state,
            to_state,
            hypothesis,
            tuple(evidence),
            decision_key,
            timestamp_seq,
            replacement,
        )
        self._validate_event(restored)
        return restored

    def _event_statements(self, event: TypedRef) -> tuple[GraphStatement, ...]:
        """读取 Event 在当前协议六个槽中的全部 statement。"""
        return tuple(
            statement
            for predicate in self.protocol.predicate_refs()
            for statement in self._ontology.statements(
                predicate=predicate, subject=event)
        )

    def _single_target(
            self, predicate: TypedRef, subject: TypedRef, *, label: str,
            ) -> tuple[TypedRef, tuple[GraphStatement, ...]]:
        """读取一个唯一语义端点，允许同端点具有多来源 assertion。"""
        targets, statements = self._multi_targets(predicate, subject)
        if len(targets) != 1:
            raise StructureOrderLifecycleError(
                f"{label} 必须有唯一端点，实际 {len(targets)} 个")
        return targets[0], statements

    def _multi_targets(
            self, predicate: TypedRef, subject: TypedRef,
            ) -> tuple[tuple[TypedRef, ...], tuple[GraphStatement, ...]]:
        """按完整对象身份去重并稳定返回 Event 槽端点。"""
        statements = self._ontology.statements(
            predicate=predicate, subject=subject)
        targets: dict[ObjectIdentity, TypedRef] = {}
        for statement in statements:
            identity = self._ontology.identity_of(statement.object)
            prior = targets.get(identity)
            if prior is not None and prior != statement.object:
                raise StructureOrderLifecycleError(
                    "同一对象身份映射到多个图节点")
            targets[identity] = statement.object
        identities = tuple(sorted(targets, key=ObjectIdentity.stable_key))
        return (
            tuple(targets[item] for item in identities),
            tuple(sorted(statements, key=lambda item: item.assertion_hash)),
        )

    @staticmethod
    def _validate_metadata(
            event: StructureOrderLifecycleEvent, *,
            scope: ScopeIdentity, provenance_kind: int,
            epistemic_origin: int, content_version: int,
            qualifiers: tuple[int, ...]) -> None:
        """核验 Event assertion 绑定 aggregate scope 和开放整数来源字段。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        if scope != event.hypothesis.scope:
            raise ValueError("lifecycle Event 必须使用 H-06 aggregate scope")
        if not isinstance(qualifiers, tuple):
            raise TypeError("qualifiers 必须是整数 tuple")
        assert_int(
            provenance_kind,
            epistemic_origin,
            content_version,
            *qualifiers,
            _where="StructureOrderLifecycleGraph.append",
        )
        if type(provenance_kind) is not int or provenance_kind <= 0:
            raise ValueError("provenance_kind 必须为严格正整数")
        if type(epistemic_origin) is not int or epistemic_origin < 0:
            raise ValueError("epistemic_origin 必须为非负严格整数")
        if type(content_version) is not int or content_version < 0:
            raise ValueError("content_version 必须为非负严格整数")
        if any(type(item) is not int for item in qualifiers):
            raise ValueError("qualifiers 必须使用严格整数")

    @staticmethod
    def _require_replay_metadata(
            event: MaterializedStructureOrderLifecycleEvent, *,
            scope: ScopeIdentity, provenance_kind: int,
            epistemic_origin: int, content_version: int,
            qualifiers: tuple[int, ...]) -> None:
        """精确重放时要求每条 Event statement 具有同一本次来源元数据。"""
        expected = (
            scope,
            provenance_kind,
            epistemic_origin,
            content_version,
            qualifiers,
        )
        for statement in event.statements:
            assertion = statement.assertion
            actual = (
                assertion.scope,
                assertion.provenance_kind,
                assertion.epistemic_origin,
                assertion.content_version,
                assertion.qualifiers,
            )
            if actual != expected:
                raise StructureOrderLifecycleError(
                    "lifecycle Event 精确重放元数据不一致")

    def _relate(
            self, predicate: TypedRef, subject: TypedRef,
            object_ref: TypedRef,
            metadata: tuple[ScopeIdentity, int, int, int, tuple[int, ...]],
            ) -> GraphStatement:
        """用统一来源元数据追加一条 lifecycle statement。"""
        scope, provenance, epistemic, content_version, qualifiers = metadata
        return self._ontology.relate(
            predicate,
            subject,
            object_ref,
            scope=scope,
            provenance_kind=provenance,
            epistemic_origin=epistemic,
            content_version=content_version,
            qualifiers=qualifiers,
        )


__all__ = [
    "MaterializedStructureOrderLifecycleEvent",
    "StructureOrderLifecycleError",
    "StructureOrderLifecycleEvent",
    "StructureOrderLifecycleGraph",
    "StructureOrderLifecycleProtocol",
    "StructureOrderProjection",
]
