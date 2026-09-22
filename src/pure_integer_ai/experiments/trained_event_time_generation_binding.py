"""从 LC-05 拓扑恢复输入结构到既有 connector 的零复制绑定。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OBJECT_ROLE_BINDING,
    ObjectIdentity,
)
from pure_integer_ai.experiments.trained_event_time_topology import (
    EventTimeGraphEdge,
    EventTimeGraphTopology,
)


GENERATION_BINDING_VERSION = 1
GENERATION_BINDING_NAMESPACE = 91561


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """用长度边界编码一个完整整数键。"""
    return len(value), *value


def _one(
        edges: tuple[EventTimeGraphEdge, ...], edge_kind: int, *, label: str,
        ) -> EventTimeGraphEdge:
    """恢复唯一 frame 字段，拒绝缺失和竞争。"""
    matches = tuple(item for item in edges if item.edge_kind == edge_kind)
    if len(matches) != 1:
        raise ValueError(f"Event/Time generation frame {label} 不唯一")
    return matches[0]


@dataclass(frozen=True, slots=True)
class TrainedEventTimeGenerationBinding:
    """同一训练图内 frame、输入拓扑、事实、角色和 connector 的闭合关系。"""

    frame_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    connector_key: tuple[int, ...]
    input_structure_key: tuple[int, ...]
    context_key: tuple[int, ...]
    role_binding_keys: tuple[tuple[int, ...], ...]
    event_keys: tuple[tuple[int, ...], ...]
    anchor_keys: tuple[tuple[int, ...], ...]
    interval_keys: tuple[tuple[int, ...], ...]
    aspect_keys: tuple[tuple[int, ...], ...]
    revision_keys: tuple[tuple[int, ...], ...]
    retention_keys: tuple[tuple[int, ...], ...]
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    topology_edge_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """核验每个字段仍是完整对象键，且集合规范排序去重。"""
        scalar = (
            "frame_key", "proposition_key", "predicate_key", "connector_key",
            "input_structure_key", "context_key", "source_ref", "scope_key",
        )
        for name in scalar:
            value = getattr(self, name)
            if (type(value) is not tuple or not value
                    or any(type(item) is not int or item < 0 for item in value)):
                raise ValueError(f"Event/Time generation binding {name} 非法")
        collections = (
            "role_binding_keys", "event_keys", "anchor_keys", "interval_keys",
            "aspect_keys", "revision_keys", "retention_keys",
            "topology_edge_keys",
        )
        for name in collections:
            values = getattr(self, name)
            if (type(values) is not tuple or values != tuple(sorted(set(values)))
                    or any(type(item) is not tuple or not item for item in values)):
                raise ValueError(
                    f"Event/Time generation binding {name} 必须规范排序去重")
        if len(self.role_binding_keys) < 2 or len(self.event_keys) < 2:
            raise ValueError("Event/Time generation binding 缺少二元角色/Event")
        if ObjectIdentity.from_stable_key(
                self.proposition_key).object_kind != OBJECT_PROPOSITION:
            raise ValueError("Event/Time generation binding proposition 类型错误")
        if any(ObjectIdentity.from_stable_key(item).object_kind != OBJECT_ROLE_BINDING
               for item in self.role_binding_keys):
            raise ValueError("Event/Time generation binding RoleBinding 类型错误")
        if any(ObjectIdentity.from_stable_key(item).object_kind not in {
                OBJECT_EVENT, OBJECT_PROPOSITION} for item in self.event_keys):
            raise ValueError("Event/Time generation binding event 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回可恢复全部绑定对象和拓扑证据的整数记录。"""
        result = [GENERATION_BINDING_VERSION]
        for value in (
                self.frame_key, self.proposition_key, self.predicate_key,
                self.connector_key, self.input_structure_key, self.context_key,
                self.source_ref, self.scope_key):
            result.extend(_pack(value))
        for values in (
                self.role_binding_keys, self.event_keys, self.anchor_keys,
                self.interval_keys, self.aspect_keys, self.revision_keys,
                self.retention_keys, self.topology_edge_keys):
            result.append(len(values))
            for value in values:
                result.extend(_pack(value))
        return tuple(result)

    def scope_and_time_key(self) -> tuple[int, ...]:
        """提供 ResponsePlan 可直接追加的 Event/Time 限定记录。"""
        result = [GENERATION_BINDING_NAMESPACE, GENERATION_BINDING_VERSION]
        for value in (self.frame_key, self.source_ref, self.scope_key):
            result.extend(_pack(value))
        for values in (
                self.event_keys, self.anchor_keys, self.interval_keys,
                self.aspect_keys, self.revision_keys, self.retention_keys):
            result.append(len(values))
            for value in values:
                result.extend(_pack(value))
        return tuple(result)

    def integer_trace(self) -> dict[str, object]:
        """返回不含来源正文或表层的完整整数 trace。"""
        return {
            "frame": list(self.frame_key),
            "proposition": list(self.proposition_key),
            "predicate": list(self.predicate_key),
            "connector": list(self.connector_key),
            "input_structure": list(self.input_structure_key),
            "context": list(self.context_key),
            "role_bindings": [list(item) for item in self.role_binding_keys],
            "events": [list(item) for item in self.event_keys],
            "anchors": [list(item) for item in self.anchor_keys],
            "intervals": [list(item) for item in self.interval_keys],
            "aspects": [list(item) for item in self.aspect_keys],
            "revisions": [list(item) for item in self.revision_keys],
            "retentions": [list(item) for item in self.retention_keys],
            "source_ref": list(self.source_ref),
            "scope": list(self.scope_key),
            "topology_edges": [list(item) for item in self.topology_edge_keys],
            "scope_and_time": list(self.scope_and_time_key()),
            "stable_key": list(self.stable_key()),
        }


def _reachable_edges(
        topology: EventTimeGraphTopology,
        frame_key: tuple[int, ...],
        ) -> tuple[EventTimeGraphEdge, ...]:
    """从单个 frame 正向收集有限结构，不经共享 kind 反查其他实例。"""
    frame_set = set(topology.frame_keys)
    reached = {frame_key}
    pending = [frame_key]
    result: set[EventTimeGraphEdge] = set()
    while pending:
        subject = pending.pop()
        for edge in topology.edges:
            if edge.subject_key != subject:
                continue
            result.add(edge)
            target = edge.object_key
            if target in frame_set and target != frame_key:
                continue
            if target not in reached:
                reached.add(target)
                pending.append(target)
    return tuple(sorted(result, key=lambda item: item.stable_key()))


def read_event_time_generation_bindings(
        topology: EventTimeGraphTopology,
        ) -> tuple[TrainedEventTimeGenerationBinding, ...]:
    """只接受字段齐全的 generation frame；普通 LC-05 frame 保持非生成状态。"""
    if not isinstance(topology, EventTimeGraphTopology):
        raise TypeError("event-time generation reader 需要 topology")
    result = []
    for frame_key in topology.frame_keys:
        direct = tuple(item for item in topology.edges
                       if item.subject_key == frame_key)
        generation_edges = tuple(item for item in direct
                                 if item.edge_kind in {21, 22, 23, 24})
        if not generation_edges:
            continue
        if {item.edge_kind for item in generation_edges} != {21, 22, 23, 24}:
            raise ValueError("Event/Time generation frame 只有部分绑定字段")
        proposition = _one(direct, 1, label="proposition")
        context = _one(direct, 6, label="context")
        connector = _one(direct, 21, label="connector")
        input_structure = _one(direct, 22, label="input structure")
        predicate = _one(direct, 23, label="relation predicate")
        roles = tuple(sorted({item.object_key for item in direct
                              if item.edge_kind == 24}))
        events = tuple(sorted({item.object_key for item in direct
                               if item.edge_kind == 2}))
        required_edges = tuple(item for item in direct if item.edge_kind in {
            1, 2, 6, 21, 22, 23, 24,
        })
        sources = {item.source_ref for item in required_edges}
        scopes = {item.scope_key for item in required_edges}
        if len(sources) != 1 or len(scopes) != 1:
            raise ValueError("Event/Time generation frame source/scope 发生竞争")
        reachable = _reachable_edges(topology, frame_key)

        def objects(edge_kind: int) -> tuple[tuple[int, ...], ...]:
            return tuple(sorted({item.object_key for item in direct
                                 if item.edge_kind == edge_kind}))

        result.append(TrainedEventTimeGenerationBinding(
            frame_key,
            proposition.object_key,
            predicate.object_key,
            connector.object_key,
            input_structure.object_key,
            context.object_key,
            roles,
            events,
            objects(3),
            objects(4),
            objects(5),
            objects(18),
            objects(19),
            next(iter(sources)),
            next(iter(scopes)),
            tuple(item.stable_key() for item in reachable),
        ))
    return tuple(sorted(set(result), key=lambda item: item.stable_key()))


__all__ = [
    "GENERATION_BINDING_NAMESPACE",
    "GENERATION_BINDING_VERSION",
    "TrainedEventTimeGenerationBinding",
    "read_event_time_generation_bindings",
]
