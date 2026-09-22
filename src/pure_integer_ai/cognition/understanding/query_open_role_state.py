"""把开放角色解析接入同次 Core/Dialogue frontier，绝不提升成事实。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.query_state import (
    ROOT_PENDING, SPACE_CORE, SPACE_DIALOGUE,
    BindingEntry, EvidenceEntry, FrontierEntry, QueryAnchor, QueryRoot, QueryState, VisitedKey,
)
from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate, OpenRoleFrame,
)


@dataclass(frozen=True, slots=True)
class OpenRoleQueryNode:
    """一个有明确图来源的框架节点，或其来源化开放角色投影。"""

    space: int
    frame: OpenRoleFrame
    candidate: OpenRelationCandidate | None = None

    def __post_init__(self) -> None:
        if self.space not in {SPACE_CORE, SPACE_DIALOGUE}:
            raise ValueError("开放角色节点空间非法")
        if (self.space == SPACE_DIALOGUE) != (self.candidate is not None):
            raise ValueError("只有 Dialogue 节点携带当前输入解析")
        if self.candidate is not None and self.candidate.frame != self.frame:
            raise ValueError("开放角色节点框架不符")

    @property
    def key(self) -> tuple[int, ...]:
        """框架知识与用户输入假设分别使用不同完整身份。"""
        return self.frame.schema_key() if self.candidate is None else self.candidate.stable_key()

    @property
    def source_ref(self) -> tuple[int, ...]:
        """框架来源和当前输入来源均保留，不互相替代。"""
        return (self.frame.source_ref if self.candidate is None
                else (self.candidate.source_ref[1], *self.candidate.source_ref))

    def seed(self, weight: int) -> tuple[QueryRoot, QueryAnchor, FrontierEntry]:
        """在运行查询之前登记根与 frontier；不是 Core 失败后的后备链。"""
        return (
            QueryRoot(self.space, self.key, ROOT_PENDING, weight,
                      source_ref=self.source_ref),
            QueryAnchor(self.space, self.key, kind=3),
            FrontierEntry((self.space, *self.key), self.space, target_key=self.key,
                          root_key=self.key, source_ref=self.source_ref,
                          owner_weight=weight, required_slot_gain=1),
        )

    def expand(self, state: QueryState, edge: FrontierEntry) -> QueryState:
        """绑定真实输入区间并保留 UNKNOWN；框架证据只证明框架存在。"""
        if (edge.owner_space, edge.target_key) != (self.space, self.key):
            raise ValueError("开放角色扩展节点与 frontier 不符")
        bindings = list(state.bindings)
        if self.candidate is not None:
            bindings.extend(BindingEntry(item.role, item.stable_key(), space=self.space,
                                         scope_key=self.candidate.source_ref)
                            for item in self.candidate.bindings)
        evidence = EvidenceEntry(
            self.source_ref, self.key, space=self.space,
            polarity=1 if self.candidate is None else 3,
            evidence_key=self.key,
            payload_key=self.frame.stable_key() if self.candidate is None else self.key,
            scope_key=() if self.candidate is None else self.candidate.source_ref,
        )
        visited = VisitedKey(self.space, self.key, direction=edge.direction)
        return state.with_(
            depth=max(state.depth, edge.depth + 1),
            frontier=tuple(item for item in state.frontier if item != edge),
            bindings=tuple(sorted(set(bindings), key=lambda item: item.stable_key())),
            evidence=tuple(sorted({*state.evidence, evidence}, key=lambda item: item.stable_key())),
            visited=tuple(sorted({*state.visited, visited}, key=lambda item: item.stable_key())),
            node_count=state.node_count + 1, edge_count=state.edge_count + 1,
            read_count=state.read_count + 1,
        )


def open_role_query_nodes(candidates: tuple[OpenRelationCandidate, ...],
                          ) -> tuple[OpenRoleQueryNode, ...]:
    """保留每个并列解析与原框架，不以长度或首次命中裁决歧义。"""
    nodes = set()
    for candidate in candidates:
        # All-empty-gap frames are generic envelopes synthesized for an
        # unseen predicate.  They establish a Dialogue/Memory hypothesis but
        # carry no source-local Core syntax evidence, so do not create a Core
        # capability root that could be mistaken for a fact closure.
        if any(candidate.frame.gaps):
            nodes.add(OpenRoleQueryNode(SPACE_CORE, candidate.frame))
        nodes.add(OpenRoleQueryNode(SPACE_DIALOGUE, candidate.frame, candidate))
    return tuple(sorted(nodes, key=lambda item: (item.space, item.key)))
