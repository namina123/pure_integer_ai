"""只读恢复训练图中的 LC-05 Event/Time 结构拓扑。

reader 只查询 ``GraphOntology.statements`` 和图对象完整身份，不读取课程、
SourceRecord 正文、occurrence 表层、posting 或 successor。所有边保留 statement
scope 中的完整 ``SourceRef``，共享 anchor 身份也不会丢失各自来源。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.cognition.shared.identity import ObjectIdentity


TOPOLOGY_READER_VERSION = 1

_EDGE_KIND_BY_NAME = {
    "frame_proposition": 1,
    "frame_event": 2,
    "frame_anchor": 3,
    "frame_interval": 4,
    "frame_aspect": 5,
    "frame_context": 6,
    "event_kind": 8,
    "candidate_kind": 9,
    "anchor_kind": 10,
    "interval_start": 11,
    "interval_end": 12,
    "aspect_object": 13,
    "aspect_anchor": 14,
    "aspect_interval": 15,
    "aspect_kind": 16,
    "revision": 18,
    "retention": 19,
    "epistemic_state": 20,
    "frame_connector": 21,
    "frame_input_structure": 22,
    "frame_relation_predicate": 23,
    "frame_role_binding": 24,
}
_FRAME_EDGE_KINDS = frozenset({
    1, 2, 3, 4, 5, 6, 9, 18, 19, 20, 21, 22, 23, 24,
})
_REVERSE_EDGE_KINDS = frozenset({
    1, 2, 3, 4, 5, 6, 18, 19, 21, 22, 23, 24,
})


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """校验一个非空严格非负整数稳定键。"""
    if (type(value) is not tuple or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError(f"{where} 必须是非空非负严格整数 tuple")
    return value


@dataclass(frozen=True, slots=True)
class EventTimeGraphEdge:
    """一条来源化 Event/Time 图边；``edge_kind=17`` 表示开放 narrative 边。"""

    edge_kind: int
    subject_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    object_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.edge_kind) is not int or self.edge_kind not in {
                *_EDGE_KIND_BY_NAME.values(), 17}:
            raise ValueError("Event/Time edge_kind 未注册")
        for name in ("subject_key", "predicate_key", "object_key", "scope_key"):
            _strict_key(getattr(self, name), where=f"EventTimeGraphEdge.{name}")
        _strict_key(self.source_ref, where="EventTimeGraphEdge.source_ref")
        if len(self.source_ref) != 11:
            raise ValueError("Event/Time edge 必须保留完整 SourceRef")

    def stable_key(self) -> tuple[int, ...]:
        """返回边类型、端点、来源和 scope 的无歧义整数键。"""
        values = [TOPOLOGY_READER_VERSION, self.edge_kind]
        for key in (
                self.subject_key, self.predicate_key, self.object_key,
                self.source_ref, self.scope_key):
            values.extend((len(key), *key))
        return tuple(values)

    def integer_trace(self) -> dict[str, object]:
        """返回不含任何来源正文的整数 trace。"""
        return {
            "edge_kind": self.edge_kind,
            "subject": list(self.subject_key),
            "predicate": list(self.predicate_key),
            "object": list(self.object_key),
            "source_ref": list(self.source_ref),
            "scope": list(self.scope_key),
        }

    @property
    def permits_reverse(self) -> bool:
        """只允许从结构成员回到 frame；类型概念不得反向枚举全部实例。"""
        return self.edge_kind in _REVERSE_EDGE_KINDS


@dataclass(frozen=True, slots=True)
class EventTimeGraphTopology:
    """从同一训练 SQLite 恢复的不可变 LC-05 拓扑快照。"""

    edges: tuple[EventTimeGraphEdge, ...]
    frame_keys: tuple[tuple[int, ...], ...]
    proposition_keys: tuple[tuple[int, ...], ...]
    event_state_keys: tuple[tuple[int, ...], ...]
    anchor_keys: tuple[tuple[int, ...], ...]
    interval_keys: tuple[tuple[int, ...], ...]
    aspect_keys: tuple[tuple[int, ...], ...]
    revision_keys: tuple[tuple[int, ...], ...]
    retention_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (type(self.edges) is not tuple
                or any(not isinstance(item, EventTimeGraphEdge) for item in self.edges)
                or self.edges != tuple(sorted(set(self.edges), key=lambda item: item.stable_key()))):
            raise ValueError("Event/Time edges 必须规范排序去重")
        for name in (
                "frame_keys", "proposition_keys", "event_state_keys", "anchor_keys",
                "interval_keys", "aspect_keys", "revision_keys", "retention_keys"):
            values = getattr(self, name)
            if (type(values) is not tuple or values != tuple(sorted(set(values)))
                    or any(type(item) is not tuple or not item for item in values)):
                raise ValueError(f"EventTimeGraphTopology.{name} 必须规范排序去重")

    @property
    def object_keys(self) -> tuple[tuple[int, ...], ...]:
        """返回可作为显式图输入的全部已恢复端点身份。"""
        return tuple(sorted({
            *(item.subject_key for item in self.edges),
            *(item.object_key for item in self.edges),
        }))

    @property
    def root_object_keys(self) -> tuple[tuple[int, ...], ...]:
        """返回允许显式入图的 frame 和一等 Event/Time 结构对象。"""
        return tuple(sorted({
            *self.frame_keys,
            *self.proposition_keys,
            *self.event_state_keys,
            *self.anchor_keys,
            *self.interval_keys,
            *self.aspect_keys,
            *self.revision_keys,
            *self.retention_keys,
        }))

    def stable_key(self) -> tuple[int, ...]:
        """返回对象分类和全部来源化边的整数快照。"""
        values = [TOPOLOGY_READER_VERSION]
        for group in (
                self.frame_keys, self.proposition_keys, self.event_state_keys,
                self.anchor_keys, self.interval_keys, self.aspect_keys,
                self.revision_keys, self.retention_keys):
            values.append(len(group))
            for key in group:
                values.extend((len(key), *key))
        values.append(len(self.edges))
        for edge in self.edges:
            key = edge.stable_key()
            values.extend((len(key), *key))
        return tuple(values)

    def integer_trace(self) -> dict[str, object]:
        """返回对象计数、身份和边来源，不暴露课程或来源正文。"""
        return {
            "reader_version": TOPOLOGY_READER_VERSION,
            "frame_keys": [list(item) for item in self.frame_keys],
            "proposition_keys": [list(item) for item in self.proposition_keys],
            "event_state_keys": [list(item) for item in self.event_state_keys],
            "anchor_keys": [list(item) for item in self.anchor_keys],
            "interval_keys": [list(item) for item in self.interval_keys],
            "aspect_keys": [list(item) for item in self.aspect_keys],
            "revision_keys": [list(item) for item in self.revision_keys],
            "retention_keys": [list(item) for item in self.retention_keys],
            "edges": [item.integer_trace() for item in self.edges],
        }


def _edge_from_statement(ontology: GraphOntology, edge_kind: int, statement) -> EventTimeGraphEdge:
    """从已核验 statement 恢复完整对象、来源和 scope 身份。"""
    source = statement.assertion.scope.source
    if source is None:
        raise ValueError("Event/Time statement scope 缺少 SourceRef")
    return EventTimeGraphEdge(
        edge_kind,
        ontology.identity_of(statement.subject).stable_key(),
        ontology.identity_of(statement.predicate).stable_key(),
        ontology.identity_of(statement.object).stable_key(),
        source.stable_key(),
        statement.assertion.scope.stable_key(),
    )


def read_event_time_topology(
        ontology: GraphOntology,
        predicate_identities: Mapping[str, ObjectIdentity],
        narrative_predicate_prefix: tuple[int, ...],
        ) -> EventTimeGraphTopology:
    """只读恢复 LC-05 固定结构边和 Event 间的开放 narrative 边。"""
    if not isinstance(ontology, GraphOntology):
        raise TypeError("ontology 必须是 GraphOntology")
    if not isinstance(predicate_identities, Mapping):
        raise TypeError("predicate_identities 必须是 Mapping")
    _strict_key(narrative_predicate_prefix, where="narrative_predicate_prefix")

    edges: set[EventTimeGraphEdge] = set()
    fixed_predicate_keys: set[tuple[int, ...]] = set()
    for name, edge_kind in sorted(_EDGE_KIND_BY_NAME.items(), key=lambda item: item[1]):
        identity = predicate_identities.get(name)
        if not isinstance(identity, ObjectIdentity):
            raise ValueError(f"Event/Time predicate 缺失: {name}")
        predicate_key = identity.stable_key()
        fixed_predicate_keys.add(predicate_key)
        predicate = ontology.resolve(identity)
        if predicate is None:
            continue
        for statement in ontology.statements(predicate=predicate):
            edges.add(_edge_from_statement(ontology, edge_kind, statement))

    event_keys = {
        item.object_key for item in edges if item.edge_kind == 2}
    for event_key in sorted(event_keys):
        event_ref = ontology.resolve(ObjectIdentity.from_stable_key(event_key))
        if event_ref is None:
            raise RuntimeError("Event/Time topology event identity 无法回读")
        for statement in ontology.statements(subject=event_ref):
            predicate = ontology.identity_of(statement.predicate)
            predicate_key = predicate.stable_key()
            if (predicate_key in fixed_predicate_keys
                    or predicate.components[:len(narrative_predicate_prefix)]
                    != narrative_predicate_prefix):
                continue
            object_key = ontology.identity_of(statement.object).stable_key()
            if object_key not in event_keys:
                raise ValueError("Event/Time narrative relation 指向非 Event/State 对象")
            edges.add(_edge_from_statement(ontology, 17, statement))

    ordered = tuple(sorted(edges, key=lambda item: item.stable_key()))

    def objects(edge_kind: int) -> tuple[tuple[int, ...], ...]:
        return tuple(sorted({item.object_key for item in ordered
                             if item.edge_kind == edge_kind}))

    return EventTimeGraphTopology(
        ordered,
        tuple(sorted({item.subject_key for item in ordered
                      if item.edge_kind in _FRAME_EDGE_KINDS})),
        objects(1), objects(2), objects(3), objects(4), objects(5),
        objects(18), objects(19),
    )


__all__ = [
    "TOPOLOGY_READER_VERSION",
    "EventTimeGraphEdge",
    "EventTimeGraphTopology",
    "read_event_time_topology",
]
