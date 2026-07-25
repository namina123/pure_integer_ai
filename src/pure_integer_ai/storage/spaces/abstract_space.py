"""storage.spaces.abstract_space — 核心概念空间（§十五决策1）。

纯整数·无衰减底座·初始训练固化·结构不增长（训练完成后）·边参数可调（reward 调 strength/sn/tn）。
记忆→核心砍（来源分工+衰减性差异非信任梯度·自演化在记忆层闭合）。
本类是核心空间的语义标记 + 编址入口；节点/边经 NodeStore/EdgeStore 落表。
"""
from __future__ import annotations

from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.node_store import NodeStore
from pure_integer_ai.storage.spaces.registry import SPACE_TYPE_CORE, SpaceRegistry


class AbstractSpace:
    """核心概念空间（纯整数·无衰减底座）。

    训练期 observe 建图养洁净（§十二阶段1-2）；训练完成后结构固化不增长·边参数仍可由 reward 调。
    """

    def __init__(self, registry: SpaceRegistry, backend: StorageBackend,
                 space_id: int) -> None:
        self.registry = registry
        self.backend = backend
        self.space_id = space_id
        self.nodes = NodeStore(backend)

    @classmethod
    def create(cls, registry: SpaceRegistry, name: str) -> "AbstractSpace":
        sid = registry.register(SPACE_TYPE_CORE, name)
        return cls(registry, registry.backend, sid)

    def new_local_id(self) -> int:
        """本空间内自增 local_id（编址 (space_id, local_id)·决策1）。"""
        return self.backend.next_id(self.space_id)
