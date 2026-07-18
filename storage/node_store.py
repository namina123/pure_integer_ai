"""storage.node_store — 概念节点 + def_array + assoc_table（决策3·扩列非重造）。

concept_node（决策3扩列·§7.7.1 路径 B 节点列回归 6 列）：
  (space_id, local_id, type, born_granularity, version_head, tier)
  - type ∈ {CONCEPT/WORD/CALC/TEMPLATE/OPERATOR}（吸收保留）
  - tier 维（=max 其边 tier·§十二第三视角⑤）
  - modality_marker 维撤回（迁 abstract_mark MARK_MODALITY·§7.7.1·守铁律 6 不污染节点列）
  - polarity 砍（核证无概念极性·极性在边方向+sn/tn）
  - 编址 (space_id, local_id) 留（决策1）

def_array（有序定义链·order_index 编序）/ assoc_table（无序关联）：决策4 序列符号范式留。
纯整数（_validate_row 拒 float·核心表无 str）。MUTABLE_MONOTONE（version_head 前移）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    StorageBackend, TYPE_INT, register_extension_table,
)

# ---- 节点 type 枚举（吸收保留·§十五决策3） ----
NODE_CONCEPT = 1
NODE_WORD = 2
NODE_CALC = 3
NODE_TEMPLATE = 4
NODE_OPERATOR = 5

# ---- 节点 tier（与边 tier 同·二级·§十二） ----
TIER_PRIMARY = 2
TIER_SHADOW = 1

# concept_node 列（决策3扩列·§7.7.1 路径 B：tier 留·modality_marker 迁 abstract_mark·polarity 砍）
CONCEPT_NODE_COLUMNS = [
    ("space_id", TYPE_INT),
    ("local_id", TYPE_INT),
    ("type", TYPE_INT),
    ("born_granularity", TYPE_INT),
    ("version_head", TYPE_INT),
    ("tier", TYPE_INT),
]
CONCEPT_NODE_INDEXES = [
    ("space_id", "local_id"),  # 主键端点
    ("space_id",),
]


def register_node_tables(backend: StorageBackend) -> None:
    """注册 concept_node / def_array / assoc_table（核心表·启动调一次）。"""
    backend.register_table(
        "concept_node", CONCEPT_NODE_COLUMNS,
        disc.DISC_MUTABLE_MONOTONE, CONCEPT_NODE_INDEXES, core=True,
    )
    # def_array：有序定义链（决策4 序列符号范式·order_index 编序）
    # perf round3（2026-07-13·profile n=4 坐实 84.2% wall）：read_role_seq 查
    # {space_id, local_id, ref_space_id}（graph_view.read_role_seq·generate 主热路径·348K 次/run）
    # 既有 (space_id,local_id,order_index) 因 order_index ∉ where 不满足 _covering_candidates 全覆盖判据
    # （backend.py:282 frozenset(cols)<=where_keys·无前缀匹配）→ _do_select 退全表扫（2.83ms/次）。
    # 加 (space_id,local_id,ref_space_id) 全覆盖 read_role_seq（桶 O(1) 查）+ (space_id,local_id) 覆盖
    # read_memory_sequence {space_id,local_id}（dispatch_slot:117）。bit-identical：索引只缩候选·_do_select
    # 仍按全 where 过滤·桶保插入序 → 同集同序同结果（backend.py docstring 明示·纯 perf 无 gate）。
    backend.register_table(
        "def_array",
        [("space_id", TYPE_INT), ("local_id", TYPE_INT),
         ("order_index", TYPE_INT), ("ref_space_id", TYPE_INT),
         ("ref_local_id", TYPE_INT)],
        disc.DISC_APPEND_ONLY,
        [("space_id", "local_id", "order_index"),    # order_index 序查（既有）
         ("space_id", "local_id", "ref_space_id"),   # perf: read_role_seq 全覆盖（解 84% 全表扫）
         ("space_id", "local_id")],                  # perf: read_memory_sequence 覆盖 + 前缀
        core=True,
    )
    # assoc_table：无序关联（决策4）
    backend.register_table(
        "assoc_table",
        [("space_id", TYPE_INT), ("local_id", TYPE_INT),
         ("ref_space_id", TYPE_INT), ("ref_local_id", TYPE_INT),
         ("kind", TYPE_INT)],
        disc.DISC_APPEND_ONLY,
        [("space_id", "local_id"), ("space_id", "local_id", "kind")], core=True,
    )


class NodeStore:
    """概念节点存储（经 backend 抽象·绝不写 raw SQL）。"""

    def __init__(self, backend: StorageBackend) -> None:
        self._b = backend

    def put(self, space_id: int, local_id: int, *, node_type: int,
            born_granularity: int = 0,
            tier: int = TIER_SHADOW) -> None:
        """插入概念节点（append-only·version_head 初始 0）。"""
        assert_int(space_id, local_id, node_type, born_granularity,
                   tier, _where="NodeStore.put")
        self._b.insert("concept_node", {
            "space_id": space_id, "local_id": local_id,
            "type": node_type, "born_granularity": born_granularity,
            "version_head": 0,
            "tier": tier,
        })

    def get(self, space_id: int, local_id: int) -> dict[str, Any] | None:
        rows = self._b.select("concept_node",
                              where={"space_id": space_id, "local_id": local_id},
                              limit=1)
        return rows[0] if rows else None

    def advance_version(self, space_id: int, local_id: int,
                        new_head: int) -> None:
        """version_head 前移（MUTABLE_MONOTONE·单调不降·新 head 须 ≥ 旧）。"""
        cur = self.get(space_id, local_id)
        if cur is None:
            raise KeyError(f"advance_version: 节点不存在 ({space_id},{local_id})")
        old = cur["version_head"]
        if new_head < old:
            raise disc.MonotoneViolation(
                f"version_head 须单调不降: old={old}, new={new_head}"
            )
        self._b.update("concept_node",
                       where={"space_id": space_id, "local_id": local_id},
                       set_={"version_head": new_head})

    def set_tier(self, space_id: int, local_id: int, new_tier: int) -> None:
        """节点 tier = max 其边 tier（MUTABLE_MONOTONE·只升不降·§十二⑤）。"""
        cur = self.get(space_id, local_id)
        if cur is None:
            raise KeyError(f"set_tier: 节点不存在 ({space_id},{local_id})")
        old = cur["tier"]
        if new_tier < old:
            raise disc.MonotoneViolation(
                f"tier 须单调不降: old={old}, new={new_tier}"
            )
        self._b.update("concept_node",
                       where={"space_id": space_id, "local_id": local_id},
                       set_={"tier": new_tier})
