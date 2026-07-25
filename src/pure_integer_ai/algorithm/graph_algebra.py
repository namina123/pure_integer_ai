"""algorithm.graph_algebra — 图代数层（纯整数图推理·后继推理与图代数设计 §九.3）。

**「用代数域+集合论逻辑表达简单逻辑」= 图代数**：全局性质（传递性·可达）从邻接派生·不存（§二 1-1 推不出整链→图代数登场）。
本层落 §九.3 graph_algebra（邻接矩阵幂/传递闭包/迹·整数精确·100万边可行·后继推理与图代数设计）。

首发：IS_A 祖先闭包。**SCC 凝聚 + 拓扑序传播 O(V+E)**——处理任意图（DAG 或含环）：
  ① Tarjan SCC：强连通分量（环节点互为祖先）凝成超节点。
  ② 凝聚图是 DAG → Kahn 拓扑序（parent-SCC before child-SCC）单遍祖先传播。
  ③ 展开：节点祖先 = 其 SCC 外部祖先 ∪ 同 SCC 其他成员（环互祖）。
远快于 `closure.transitive_closure` 的 BFS-per-source O(V·E)（307k 含环 IS_A 实测后者 >5min·本层 O(V+E)）。

**为何含环**：raw 数据（ConceptNet/ChineseSemanticKB 抽象）噪声成环（实测 abstract 163k 节点 134k 在环）。
proper subset 应无环但 raw 不净·§二 已期「链 A→B→C→A 成环」→ 图代数须处理（非降级 fallback）。

**bit-identical**：SCC 凝聚输出 == `closure.transitive_closure(include_direct=True)` 祖先集
（环节点互为祖先 + 共享外部祖先·同 BFS 可达性·DAG 退化为单节点 SCC 同前）。

铁律：纯整数（NodeRef 整数二元组·零浮点·设计「代数数域精确」）/ 确定性（Tarjan sorted starts + heapq·输出集合序无关 bit-identical）/ 单向依赖（L2 算法·仅读入参边·不环·同 closure.py 层位）。
诚实边界：图代数派生 ≠ 语义验真（闭包给可达祖先·真伪=#479 外部数据责任·同 IS_A 既有「非证明」）。
"""
from __future__ import annotations

import heapq

from pure_integer_ai.storage.edge_types import EDGE_IS_A

NodeRef = tuple[int, int]


def _tarjan_scc(nodes: set[NodeRef], adj: dict[NodeRef, list[NodeRef]]) -> list[list[NodeRef]]:
    """Tarjan 强连通分量（iterative·避递归栈溢·确定性 sorted starts）。

    返 list of SCC（each a list of NodeRef·同 SCC 内节点互达=环互祖）。
    """
    index: dict[NodeRef, int] = {}
    low: dict[NodeRef, int] = {}
    on_stack: dict[NodeRef, bool] = {}
    stack: list[NodeRef] = []
    sccs: list[list[NodeRef]] = []
    cnt = 0
    for start in sorted(nodes):
        if start in index:
            continue
        # work 栈帧：[node, next_neighbor_idx]（显式索引·可靠·避 iterator-resume 坑）
        work: list[list] = [[start, 0]]
        index[start] = low[start] = cnt; cnt += 1
        stack.append(start); on_stack[start] = True
        while work:
            frame = work[-1]
            node, ni = frame[0], frame[1]
            nbs = adj.get(node, ())
            if ni < len(nbs):
                frame[1] = ni + 1   # 推进邻居指针
                nb = nbs[ni]
                if nb not in index:
                    index[nb] = low[nb] = cnt; cnt += 1
                    stack.append(nb); on_stack[nb] = True
                    work.append([nb, 0])
                elif on_stack.get(nb):
                    low[node] = min(low[node], index[nb])   # 栈上邻居·Tarjan 用 index[nb]
            else:
                # node 所有邻居处理完·判 SCC 根
                if low[node] == index[node]:
                    comp: list[NodeRef] = []
                    while True:
                        w = stack.pop(); on_stack[w] = False; comp.append(w)
                        if w == node:
                            break
                    sccs.append(comp)
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])   # 回溯传 low
    return sccs


def break_back_edges(nodes: set[NodeRef], adj: dict[NodeRef, list[NodeRef]]
                     ) -> set[tuple[NodeRef, NodeRef]]:
    """DFS back-edge removal — 删每 SCC>1 内的环边（保 tree/forward/cross·只删 back-edge）→ 干净 DAG。

    per SCC>1：3-color DFS（WHITE/GRAY/BLACK）·back-edge = 指向当前栈上（GRAY）节点的边 → 删。
    显式索引栈帧（镜像 ``_tarjan_scc``·避 iterator-resume 坑）·sorted starts/neighbors = 确定性
    （同输入同删除集 → 同 cleaned DAG → bit-identical）。

    用途：数据 prep（ChineseSemanticKB 抽象 raw 含环噪声 → 干净 DAG·#1133·``scratch/clean_abstract_cycles`` 调）。
    **为何在 graph_algebra**：纯图变换（adj→删除集·无 I/O·无语义判定）·同 ``_tarjan_scc`` 为图原语·
    在包内可单测（``tests/test_abstract_cycle_clean`` 锁算法）·数据 prep pipeline（读 K:/·写 data_llm/）留 scratch。

    保证：删 back-edge 破所有环（任一有向环含 ≥1 DFS back-edge·CLRS）·corpus-relevant 子集（#1142）of DAG is DAG。
    诚实边界：删除集非最小 feedback arc set（NP-hard）·是确定性启发式（保 DAG·非最少删边·§六已认）。

    nodes/adj : 同 ``_tarjan_scc`` 入参（adj child→parents·caller 已 drop 自环 + sorted）。
    返 : 删除边集 ``{(child, parent), ...}``（caller 据此过滤·保留输入序）。
    """
    sccs = _tarjan_scc(nodes, adj)
    remove: set[tuple[NodeRef, NodeRef]] = set()
    for comp in sccs:
        if len(comp) < 2:
            continue
        members = set(comp)
        # induced sub-adj（within SCC·drop 自环·sorted 确定性）
        sub = {n: sorted(set(m for m in adj.get(n, ()) if m in members and m != n))
               for n in members}
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in members}
        for start in sorted(members):
            if color[start] != WHITE:
                continue
            work: list[list] = [[start, 0]]
            color[start] = GRAY
            while work:
                frame = work[-1]
                node, ni = frame[0], frame[1]
                nbs = sub.get(node, ())
                if ni < len(nbs):
                    frame[1] = ni + 1
                    nb = nbs[ni]
                    if color[nb] == GRAY:
                        remove.add((node, nb))   # back-edge（nb 在栈上）→ 删
                    elif color[nb] == WHITE:
                        color[nb] = GRAY
                        work.append([nb, 0])
                    # BLACK = cross/forward edge ·保（非环边）
                else:
                    color[node] = BLACK
                    work.pop()
    return remove


def isa_ancestor_map(edges: list[tuple[NodeRef, NodeRef, int, object | None]]
                     ) -> tuple[dict[NodeRef, set[NodeRef]], bool]:
    """IS_A 祖先闭包（SCC 凝聚 + 拓扑序传播·O(V+E)·处理任意图含环）。

    edges : IS_A 边 ``[(child, parent, EDGE_IS_A, meta|None), ...]``（from=child to=parent·同
      `closure.transitive_closure` 入参格式·caller 已按需 filter source）。
    返 ``(ancestor_map, fell_back)``：
      - ancestor_map ``{child: set(ancestors)}``（child 全部祖先·不含自身·仅含 ≥1 祖先的 child·
        同 `closure.transitive_closure(include_direct=True)` 祖先集·bit-identical·含环互祖）。
      - fell_back : 恒 False（SCC 凝聚处理任意图·无须 fallback·保留字段为 API 稳定/未来扩展）。

    bit-identical：SCC 凝聚输出同 closure（环互祖 + 共享外部·同 BFS 可达性·DAG 退化同单节点 SCC）。

    铁律：纯整数 / 确定性（Tarjan sorted starts + heapq·祖先集内容序无关）/ O(V+E)（远快于 BFS-per-source O(V·E)）。
    """
    # ① 稀疏邻接（child→parents）+ 节点集（确定性 sorted+dedup）
    adj: dict[NodeRef, list[NodeRef]] = {}
    nodes: set[NodeRef] = set()
    for child, parent, _et, _meta in edges:
        if child == parent:
            continue   # 自环不建（同 is_a.py:57·closure:80 src 自跳）
        adj.setdefault(child, []).append(parent)
        nodes.add(child)
        nodes.add(parent)
    for k in adj:
        adj[k] = sorted(set(adj[k]))   # 确定性（NodeRef 升序）

    if not nodes:
        return {}, False

    # ② Tarjan SCC（环凝超节点）
    sccs = _tarjan_scc(nodes, adj)
    scc_of: dict[NodeRef, int] = {}
    scc_members: list[set[NodeRef]] = []
    for i, comp in enumerate(sccs):
        members = set(comp)
        scc_members.append(members)
        for nd in comp:
            scc_of[nd] = i

    # ③ 凝聚图（DAG）：scc_adj[X]=X 的 parent-SCC 集（Y where X→Y·X≠Y）·scc_children[Y]=[X]
    n_scc = len(sccs)
    scc_adj: dict[int, set[int]] = {i: set() for i in range(n_scc)}
    scc_children: dict[int, list[int]] = {i: [] for i in range(n_scc)}
    for child, parents in adj.items():
        x = scc_of[child]
        for parent in parents:
            y = scc_of[parent]
            if x != y and y not in scc_adj[x]:
                scc_adj[x].add(y)
                scc_children[y].append(x)

    # ④ Kahn 拓扑序（parent-SCC Y before child-SCC X）·heapq scc_id 升序（确定性·输出集合序无关）
    parent_count = {i: len(scc_adj[i]) for i in range(n_scc)}
    heap = [i for i in range(n_scc) if parent_count[i] == 0]
    heapq.heapify(heap)
    scc_external: dict[int, set[NodeRef]] = {}   # scc_id → 外部祖先节点集（已全）
    while heap:
        x = heapq.heappop(heap)
        ext_x = scc_external.get(x, ())   # x 的外部祖先已全（所有 parent-SCC 已贡献）
        # 向 child-SCC 累积：x 的成员 + x 的外部祖先（皆 child 成员的祖先）
        for c in scc_children[x]:
            cur = scc_external.setdefault(c, set())
            cur |= scc_members[x]
            if ext_x:
                cur |= ext_x
            parent_count[c] -= 1
            if parent_count[c] == 0:
                heapq.heappush(heap, c)   # 所有 parent-SCC 已贡献 → c 外部祖先全 → 入队
    # 凝聚图是 DAG → 必处理完（无须环 fallback）

    # ⑤ 展开：节点祖先 = 其 SCC 外部祖先 ∪ 同 SCC 其他成员（环互祖·不含自身）
    ancestors: dict[NodeRef, set[NodeRef]] = {}
    for i, members in enumerate(scc_members):
        ext = scc_external.get(i, ())
        if len(members) == 1:
            nd = next(iter(members))
            if ext:
                ancestors[nd] = set(ext)
        else:
            others = members   # 减自身在循环内
            for nd in members:
                a = set(ext) | (others - {nd})
                if a:
                    ancestors[nd] = a
    return ancestors, False   # fell_back 恒 False（SCC 处理任意图）
