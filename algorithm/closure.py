"""algorithm.closure — transitive_closure（按 edge_type 分发·闭包纯净性·CLOSURE 派生不存储）。

§十五决策9 + §8.1c-bis 闭包纯净性。传递闭包按 edge_type 分发：
  - 只**同 edge_type** 边参与闭包（闭包纯净性·不跨类型混闭）
  - purity_filter 可选：排除不进闭包的边（如 REFERS_TO 仅 PURE_ALIAS subtype 进纯同指闭包·
    喻称 METAPHOR / occurrence 不进·防闭包推出诗仙↔李太白式语义错位·§8.1c-bis 闭包失败模式B翻版）
  - **CLOSURE 派生不存储**：返回派生闭包边集（tag EDGE_CLOSURE）·调用方不写 edge 宽表
    （§十五 line159·闭包按需派生·不持久化膨胀）

闭包失败模式（§8.1c-bis·CAUSES 闭包·调用方守·本模块机械返回可达性）：
  A 缺中间节点 / B 边类型不纯断链 / C 方向错误放大 O(链²) / D confounder 误标固化
  → 本模块按 type 分发 + purity_filter 守 B（类型纯净）·A/C/D 由调用方（认知层）守源。

算法：BFS per source（稀疏图·纯整数）·O(|V|·|E|)·确定性（按节点自然序扩散）。
NodeRef = tuple[int, int] = (space_id, local_id)。
"""
from __future__ import annotations

from typing import Callable

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_types import EDGE_CLOSURE

NodeRef = tuple[int, int]
# edge dict：调用方传的边元数据（含 edge_type / subtype 等）·purity_filter 用
EdgeDict = dict
PurityFilter = Callable[[EdgeDict], bool]


def transitive_closure(
    edges: list[tuple[NodeRef, NodeRef, int, EdgeDict | None]],
    *,
    types: set[int],
    purity_filter: PurityFilter | None = None,
    include_direct: bool = False,
) -> set[tuple[NodeRef, NodeRef, int]]:
    """按 edge_type 分发传递闭包。

    edges : [(from, to, edge_type, edge_meta_dict|None), ...]
    types : 参与闭包的 edge_type 集合（按头分发·调用方传 {EDGE_REFERS_TO} 等）。
            **跨 type 不混闭**：每 type 各自算闭包（闭包纯净性）。
    purity_filter : 可选·edge_meta → bool（True=纯净进闭包·False=排除）。
            REFERS_TO 闭包用：lambda m: m.get("subtype")==SUBTYPE_PURE_ALIAS（喻称/occurrence 排除）。
    include_direct : True=返回含直接边·False=只返回派生（间接）闭包边。

    返回派生闭包边集 {(from, to, edge_type)}·tag 原 edge_type（不混 EDGE_CLOSURE·
    调用方知这是闭包·CLOSURE 派生不存储）。
    """
    for t in types:
        assert_int(t, _where="transitive_closure.types")
    # 按 edge_type 分组（同 type 才闭包）
    adj_by_type: dict[int, dict[NodeRef, list[NodeRef]]] = {t: {} for t in types}
    direct: set[tuple[NodeRef, NodeRef, int]] = set()
    for u, v, et, meta in edges:
        assert_int(et, _where="transitive_closure.edge_type")
        if et not in adj_by_type:
            continue  # 非 types 内的边不参与闭包
        if purity_filter is not None and meta is not None:
            if not purity_filter(meta):
                continue  # 不纯净·排除（闭包纯净性）
        adj_by_type[et].setdefault(u, []).append(v)
        direct.add((u, v, et))
    # 邻接排序（确定性·按节点自然序扩散）
    for et in adj_by_type:
        for k in adj_by_type[et]:
            adj_by_type[et][k] = sorted(adj_by_type[et][k])

    closure: set[tuple[NodeRef, NodeRef, int]] = set()
    for et, adj in adj_by_type.items():
        # BFS per source
        sources = sorted(adj.keys())
        for src in sources:
            visited: set[NodeRef] = set()
            queue: list[NodeRef] = list(adj[src])
            # 多层 BFS
            while queue:
                nxt: list[NodeRef] = []
                for w in queue:
                    if w == src:
                        continue  # 自环不进闭包（同指闭包自反性不显式存）
                    if w in visited:
                        continue
                    visited.add(w)
                    closure.add((src, w, et))
                    for x in adj.get(w, []):
                        if x not in visited:
                            nxt.append(x)
                # 保序（确定性）
                queue = sorted(set(nxt))
    if include_direct:
        return closure | direct
    # 只返回派生（间接）闭包：去掉直接边
    return closure - direct


def reachable(closure: set[tuple[NodeRef, NodeRef, int]],
              src: NodeRef, edge_type: int) -> set[NodeRef]:
    """从闭包集查 src 经 edge_type 可达的节点集（派生闭包查询）。"""
    return {v for (u, v, et) in closure if u == src and et == edge_type}


def closure_pure_refers_to(meta: EdgeDict) -> bool:
    """REFERS_TO 闭包纯净性 filter 样例（性质A 稳定同指进闭包·喻称/occurrence 排除）。

    供调用方参考·本模块不硬编码 subtype（subtype 语义在 storage/edge_store）。
    调用方应传：lambda m: m.get("subtype") == SUBTYPE_PURE_ALIAS。
    """
    from pure_integer_ai.storage.edge_store import SUBTYPE_PURE_ALIAS
    return meta.get("subtype") == SUBTYPE_PURE_ALIAS
