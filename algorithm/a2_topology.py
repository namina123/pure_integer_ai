"""algorithm.a2_topology — A2 拓扑分层（Kahn·按头分发·依赖 storage+crosscut）。

§十五决策9 A2 系统内自建。Kahn 算法：拓扑序 + 分层。
  - **按头分发**：调用方按 edge_type（头）过滤边再传入。A2 stepper（Stage 4）对每头各调一次
    Kahn·同头内拓扑序·跨头不混（PRECEDES 头与 CAUSES 头各自分层·不交叉）。
  - **分层**：layer[node] = max(layer[pred]) + 1（源节点 layer=0）。汇聚步进按层推进
    （PRECEDES AND 等所有前驱·CAUSES OR 任一 active·Stage 4 汇聚步进消费此 layer）。
  - **环检测**：Kahn 余留 in_degree>0 者 = 环节点（loop_closure_defect·不无限循环·诚实返回）。
    PRECEDES/T_STEP 是 A 类型 DAG·环 = 结构矛盾·报 cycle_nodes 给上层判定（不兜底删边）。

确定性：Kahn 队列按 node 排序（(space_id, local_id) 自然序）处理·bit-identical。
不查 backend·纯图操作（与 storage 解耦·调用方传边列表）。

NodeRef = tuple[int, int] = (space_id, local_id)。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int

NodeRef = tuple[int, int]


class TopologyResult:
    """Kahn 拓扑结果。

    order        : 拓扑序节点列表（环外·确定性）。
    layers       : node → layer（max(pred layer)+1·源=0）。环节点不在 layers。
    cycle_nodes  : 环内节点列表（in_degree 余留>0 者·空=无环）。
    """

    __slots__ = ("order", "layers", "cycle_nodes")

    def __init__(self, order: list[NodeRef], layers: dict[NodeRef, int],
                 cycle_nodes: list[NodeRef]) -> None:
        self.order = order
        self.layers = layers
        self.cycle_nodes = cycle_nodes

    @property
    def is_dag(self) -> bool:
        return not self.cycle_nodes

    def layer_of(self, node: NodeRef, default: int = -1) -> int:
        return self.layers.get(node, default)


def kahn_topo(edges: list[tuple[NodeRef, NodeRef]]) -> TopologyResult:
    """Kahn 拓扑序 + 分层（按头分发·调用方已按 edge_type 过滤）。

    edges : [(from, to), ...] 单头边集（PRECEDES 或 T_STEP 等某头）。
    返回 TopologyResult（order/layers/cycle_nodes）。环节点入 cycle_nodes·不无限循环。
    """
    # 收集节点 + 入度 + 邻接（确定性·按节点自然序）
    nodes: set[NodeRef] = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    sorted_nodes = sorted(nodes)  # (space_id, local_id) 自然序·bit-identical
    in_deg: dict[NodeRef, int] = {n: 0 for n in sorted_nodes}
    adj: dict[NodeRef, list[NodeRef]] = {n: [] for n in sorted_nodes}
    for u, v in edges:
        adj[u].append(v)
        in_deg[v] += 1
    # 邻接排序（保确定性·同 from 的 to 按自然序）
    for k in adj:
        adj[k] = sorted(adj[k])

    # Kahn：源队列按自然序处理（deque + 批量 sorted 入队保 bit-identical·2026-07-02 修 head 指针错位 bug）
    order: list[NodeRef] = []
    layers: dict[NodeRef, int] = {}
    from collections import deque
    queue: deque = deque(sorted(n for n in sorted_nodes if in_deg[n] == 0))
    for s in queue:
        layers[s] = 0
    while queue:
        u = queue.popleft()
        order.append(u)
        lu = layers[u]
        # 收集本节点处理后就绪的后继·批量 sorted 入队（保同层自然序·确定性 bit-identical）
        newly_ready: list[NodeRef] = []
        for v in adj[u]:
            in_deg[v] -= 1
            # 层 = max(前驱层)+1：前驱层在 layers 中（前驱已出队·已定层）
            nl = lu + 1
            if v not in layers or nl > layers[v]:
                layers[v] = nl
            if in_deg[v] == 0:
                newly_ready.append(v)
        for v in sorted(newly_ready):
            queue.append(v)
    # 环节点：未出队者（in_deg 仍 > 0）
    ordered_set = set(order)
    cycle_nodes = sorted(n for n in sorted_nodes if n not in ordered_set)
    return TopologyResult(order, layers, cycle_nodes)


def max_layer(result: TopologyResult) -> int:
    """最大层数（DAG 深度·环外）。空图返 -1。"""
    if not result.layers:
        return -1
    return max(result.layers.values())


def predecessors_by_layer(edges: list[tuple[NodeRef, NodeRef]],
                          result: TopologyResult) -> dict[NodeRef, list[NodeRef]]:
    """node → 前驱列表（按 layer 升序·同层按自然序）·汇聚步进（Stage 4）消费。

    PRECEDES AND 汇聚等所有前驱·CAUSES OR 任一 active·前驱列表供汇聚判定。
    """
    preds: dict[NodeRef, list[NodeRef]] = {}
    for u, v in edges:
        preds.setdefault(v, []).append(u)
    for v in preds:
        # 按 (layer, node) 排序·确定性
        preds[v] = sorted(preds[v], key=lambda n: (result.layers.get(n, -1), n))
    return preds
