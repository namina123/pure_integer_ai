"""storage.spaces — 三空间（§十五决策1·从目的层重画）。

三空间（信任递减·职责分工·形式介质可不同）：
  伴随库(CompanionSpace)  最低优先度·非整数合法(TEXT)·sign=0 隔离·惰性挂载·可任意多个
  记忆空间(MemorySpace)   纯整数·两层物理分开(阅读/交互)·带衰减·自晋升 status·记忆→核心砍
  核心空间(AbstractSpace) 纯整数·无衰减底座·初始训练固化·结构不增长(训练后)·边参数可调

A5 space_name_hash：space 表存 type_hash/name_hash 整数·文本名入伴随库·守"文本不入核心"。
编址 (space_id, local_id) + 每空间 id_pool 自增（吸收保留）。
"""
from pure_integer_ai.storage.spaces.registry import (
    SpaceRegistry, SPACE_TYPE_CORE, SPACE_TYPE_MEMORY, SPACE_TYPE_COMPANION,
)
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.memory_space import (
    MemorySpace, STATUS_EXPERIENCE, STATUS_CONSOLIDATED,
)
from pure_integer_ai.storage.spaces.companion import CompanionSpace

__all__ = [
    "SpaceRegistry", "SPACE_TYPE_CORE", "SPACE_TYPE_MEMORY", "SPACE_TYPE_COMPANION",
    "AbstractSpace", "MemorySpace", "STATUS_EXPERIENCE", "STATUS_CONSOLIDATED",
    "CompanionSpace",
]
