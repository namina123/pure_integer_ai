"""cognition.process.a2_stepper — 模块1 A2 拓扑分层步进调度（Kahn + 按头 AND/OR 分发）。

§十三C 按头分唯一不退化：PRECEDES AND（所有前驱到齐·结构汇聚·守 def→use 完备性）/
  CAUSES OR（任一 active 前驱可推·因果汇聚·选 strength/率高优先）。两头存在正是承载
  不同语义（统一 AND→CAUSES 死锁 / 统一 OR→PRECEDES 破缺）。

  A2_layer(subgraph_edges, active, head_types) -> (topo_layers, convergence, stepper)
    topo_layers  Kahn 拓扑层序（combined head 边·O(V+E)·环检测自然·层索引纯整）
    convergence  汇聚点 per-head in-degree 计数（PRECEDES 同 order_index 共前驱才算并行汇聚组）
    stepper      HeadStepper（按头分步进器·advance 返选定前驱边集·F1/S5/D2/D3 落盘）

  stepper.advance(node, head) -> list[EdgeRef] | BLOCKED
    AND（PRECEDES）= 全前驱边（所有前驱 active·到齐）/ OR（CAUSES）= 选中前驱的边（高优先）
    返选定边集（非 bool/chosen node·path.edges 存非派生·§十四DAG-path 契约）
    D3：predecessors 返前驱边列表（edge.to==node·edge.et==head）非节点

复用 algorithm.a2_topology.kahn_topo（决策9 自建·Kahn O(V+E)·纯整标准算法）。
确定性：Kahn 队列按节点自然序·tiebreak 稳定排序 ref·bit-identical。
诚实边界：汇聚语义不判语义因果（按边声明头类型分发非判"前驱真导致后继"·§十三C）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.algorithm.a2_topology import kahn_topo, TopologyResult
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES, EDGE_CAUSES
from pure_integer_ai.cognition.shared.types import ConceptRef, EdgeRef
from pure_integer_ai.cognition.process.effective_weight import edge_rate, is_unobserved
from pure_integer_ai.config import gates

NodeRef = ConceptRef


class _Blocked:
    """步进阻塞标记（CAUSES OR 零 active 前驱·正确停滞·§十三D）。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False   # 空语义·dag_path 用 `is BLOCKED` 判


BLOCKED = _Blocked()


def _edge_ref(e: dict[str, Any]) -> EdgeRef:
    """边行 → EdgeRef 5-tuple（space_id_from, local_id_from, space_id_to, local_id_to, edge_type）。"""
    return (e["space_id_from"], e["local_id_from"],
            e["space_id_to"], e["local_id_to"], e["edge_type"])


def _effective_weight_of(e: dict[str, Any]) -> int:
    """边行 effective_weight（item3 缺漏3 选路键·CAUSES=strength×rate·未观测=0·PRECEDES=1）。

    选路用动态 effective_weight（响应 reward）非静态 base_strength（reward 永不调·违 H4）。
    """
    from pure_integer_ai.cognition.process.effective_weight import effective_weight
    return effective_weight(e)


def _group_by_order_index(preds: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """PRECEDES 前驱按 order_index 分组（同 order_index 共前驱=并行汇聚组·§十三C）。

    None order_index 各自成组（不同序·顺序到达非汇聚）。
    """
    groups: dict[int, list[dict[str, Any]]] = {}
    singles: list[list[dict[str, Any]]] = []
    for e in preds:
        oi = e.get("order_index")
        if oi is None:
            singles.append([e])
        else:
            groups.setdefault(oi, []).append(e)
    return list(groups.values()) + singles


class HeadStepper:
    """按头分步进器（AND/OR 分发·advance 返选定前驱边集）。

    active 随步进增长（dag_path 主控调 add_active·初始 = 种子集 e）。
    """

    def __init__(self, head_types: set[int],
                 active: set[NodeRef],
                 pred_index: dict[tuple[NodeRef, int], list[dict[str, Any]]]) -> None:
        self.head_types = head_types
        self.active = active
        # _pred_index = pred_index（_build_pred_index 预建·与 _build_convergence 共享去重）的**排序副本**。
        # 排序按 from_ref（CAUSES advance min tiebreak 确定性 + predecessors 返序确定）。pred_index 未排序
        # （插入序）·此处 dict-comp 造新 list 不改共享原 dict·_build_convergence 读原样（conv 计数序无关）。
        # 同 pred_index 数据 + 同 sort key + stable-sort 同输入序 → 与原自建+排序逐位一致 bit-identical。
        self._pred_index: dict[tuple[NodeRef, int], list[dict[str, Any]]] = {
            k: sorted(v, key=lambda e: (e["space_id_from"], e["local_id_from"]))
            for k, v in pred_index.items()
        }

    def predecessors(self, node: NodeRef, head: int) -> list[dict[str, Any]]:
        """前驱边列表（D3·edge.to==node·edge.et==head·返边非节点）。"""
        return self._pred_index.get((node, head), [])

    def add_active(self, node: NodeRef) -> None:
        """步进到 node → 加入 active（dag_path 主控调·active 即"已到达"）。"""
        self.active.add(node)

    def advance(self, node: NodeRef, head: int) -> list[EdgeRef] | _Blocked:
        """按头步进·返选定前驱边集（AND=全前驱边 / OR=选中前驱的边）。

        AND 到齐语义 = 所有前驱 active（已到达·layer 序保证前驱先于后继）。
        """
        assert_int(head, _where="HeadStepper.advance")
        preds = self.predecessors(node, head)
        if not preds:
            return []   # 无前驱（源节点）·空选定边集·非阻塞
        if head == EDGE_PRECEDES:
            # S2 dead-end factor A 修（gate PRECEDES_OR_MODE）+ factor C 修（gate PRECEDES_OI_MODE·F2）：
            # （episode_loop 唯 production caller·code 走 verify_round 绕 dag_path）→ PRECEDES AND
            # （def→use 设计）production 死语义·language 重复词概念多前驱·AND 全 active 永不满足致 dead-end。
            # gate ON→OR（任一前驱 active 即推进·返全活跃前驱边保汇聚信息·非选 1）·OFF→AND（bit-identical）。
            # 若未来 code/structural 接 episode_loop·须改 Y 分域（code AND / language OR）。
            if getattr(gates, "PRECEDES_OR_MODE", False) or getattr(gates, "PRECEDES_OI_MODE", False):
                active_preds = [e for e in preds
                                if (e["space_id_from"], e["local_id_from"]) in self.active]
                if active_preds:
                    return [_edge_ref(e) for e in active_preds]
                return BLOCKED
            # AND（旧·def→use 完备性·gate OFF bit-identical）
            if all((e["space_id_from"], e["local_id_from"]) in self.active
                   for e in preds):
                return [_edge_ref(e) for e in preds]
            return BLOCKED
        if head == EDGE_CAUSES:
            # OR：任一 active 且 (rate>0 或未观测) 的前驱可推（因果汇聚·无因则无果正确停滞）
            # item3 缺漏2：未观测边（sn=tn=0）放行进 path.edges·R5 兜底给首次机会·已失败边（sn=0 tn>0）仍挡
            active_preds = [e for e in preds
                            if (e["space_id_from"], e["local_id_from"]) in self.active
                            and (edge_rate(e) > 0 or is_unobserved(e))]
            if not active_preds:
                return BLOCKED
            # 选 effective_weight 高优先（item3 缺漏3·base_strength 静态不响应 reward→改 effective_weight）
            # tiebreak：sn desc, ref 自然序（未观测边 effective_weight=0·退 sn/ref tiebreak）
            chosen = min(active_preds, key=lambda e: (
                -_effective_weight_of(e),
                -(e.get("sn", 0) or 0),
                (e["space_id_from"], e["local_id_from"])))
            return [_edge_ref(chosen)]
        # 其他头（T_STEP 等）首版按 AND 语义（结构序）
        if all((e["space_id_from"], e["local_id_from"]) in self.active for e in preds):
            return [_edge_ref(e) for e in preds]
        return BLOCKED


def _build_topo_layers(result: TopologyResult) -> list[list[NodeRef]]:
    """TopologyResult.layers → 按层分组的节点列表（层内自然序·层间层索引升序）。"""
    by_layer: dict[int, list[NodeRef]] = {}
    for node, layer in result.layers.items():
        by_layer.setdefault(layer, []).append(node)
    out: list[list[NodeRef]] = []
    for layer in sorted(by_layer):
        out.append(sorted(by_layer[layer]))
    return out


def _build_pred_index(subgraph_edges: list[dict[str, Any]],
                      head_types: set[int]) -> dict[tuple[NodeRef, int], list[dict[str, Any]]]:
    """per-head 前驱边索引 {(to_node, et): [edge, ...]}（插入序未排序·head_types 过滤）。

    perf 去重（legacy hot_zone._adj_cache 范式·cognition/process 层）：原 _build_convergence 与
    HeadStepper 各扫一遍 subgraph_edges 建**同结构** pred_index·现扫一次共享。**未排序**
    （= subgraph_edges 迭代序）·两消费者按需后处理：
      - _build_convergence 读原样（conv 计数序无关·_group_by_order_index 按 order_index 分区序无关）。
      - HeadStepper 排序自己的副本（CAUSES advance min tiebreak 需确定性 from_ref 序·不改本原 dict）。
    同 filter（et∈head_types）+ 同 key ((to,et)) + 同迭代序 → 与原两处各自建的 pred_index 逐位一致 →
    bit-identical（镜像 ancestor_map cache 同构数据共享范式）。
    """
    pred_index: dict[tuple[NodeRef, int], list[dict[str, Any]]] = {}
    for e in subgraph_edges:
        et = e["edge_type"]
        if et not in head_types:
            continue
        to = (e["space_id_to"], e["local_id_to"])
        pred_index.setdefault((to, et), []).append(e)
    return pred_index


def _build_convergence(pred_index: dict[tuple[NodeRef, int], list[dict[str, Any]]]) -> dict:
    """汇聚点识别（per-head in-degree·PRECEDES 同 order_index 共前驱才算并行汇聚组）。

    pred_index：_build_pred_index 预建（a2_layer/_oi 与 HeadStepper 共享·perf 去重免重扫）。
    """
    convergence: dict = {}
    for (node, head), preds in pred_index.items():
        if head == EDGE_PRECEDES:
            groups = _group_by_order_index(preds)
            conv = sum(1 for g in groups if len(g) >= 2)
        else:  # CAUSES·无 order_index 时序语义直接 count≥2
            conv = 1 if len(preds) >= 2 else 0
        if conv > 0:
            convergence[(node, head)] = (preds, conv)
    return convergence


def a2_layer(subgraph_edges: list[dict[str, Any]], active: set[NodeRef],
             head_types: set[int]) -> tuple[list[list[NodeRef]], dict, HeadStepper]:
    """A2 拓扑分层 + 汇聚识别 + 步进器构造。

    subgraph_edges：热区相关边集（卷一建好·audit_float==0·含各头）。
    active：初始 active 节点集（种子集 e·dag_path 主控会随步进增长）。
    head_types：按头分发的 edge_type 集合（{PRECEDES, CAUSES}·§十三C）。
    返回 (topo_layers, convergence, stepper)。
    """
    for e in subgraph_edges:
        assert_no_float(e["edge_type"], e.get("order_index") or 0,
                        _where="a2_layer.edge")
    # combined head 边 → Kahn 统一分层（跨头层序一致·环检测自然）
    combined: list[tuple[NodeRef, NodeRef]] = []
    for e in subgraph_edges:
        if e["edge_type"] not in head_types:
            continue
        combined.append(((e["space_id_from"], e["local_id_from"]),
                         (e["space_id_to"], e["local_id_to"])))
    result = kahn_topo(combined)
    topo_layers = _build_topo_layers(result)
    pred_index = _build_pred_index(subgraph_edges, head_types)
    convergence = _build_convergence(pred_index)
    stepper = HeadStepper(head_types, set(active), pred_index)
    return topo_layers, convergence, stepper


def _build_topo_layers_oi(subgraph_edges: list[dict[str, Any]],
                          head_types: set[int]) -> list[list[NodeRef]]:
    """F2 oi-first-occurrence sequence layering (drop Kahn; include cyclic nodes; v3).

    factor C solution: language PRECEDES concept cycles (token repeat); Kahn drops all
    cyclic + downstream nodes incl sink -> sink unreachable -> dag_path always DEAD_END.
    This fn layers nodes by first-occurrence order_index (each node visited once, acyclic
    by construction per design doc Sec 4); cyclic nodes enter layers at their first-occ
    out-edge oi, cycle not re-entered.

    first_occ gap detection (v3; fixes inter-seg oi incompatibility):
      out_ois/in_ois = oi of PRECEDES edges where node is from/to (CAUSES oi=None excluded).
      inter-seg PRECEDES oi = seg_order_base + i*TOKEN_CAP_OFFSET (huge; seg_order_base =
        total token count). last_token[i]'s only out-edge is inter-seg (huge) -> v2
        min(out-edge oi) gave huge first_occ -> last_token[i] visited late -> inter-seg
        chain reversed (struct_ref[i+1] visited before its pred last_token[i]) -> broken.
      v3: if out_ois and min(out_ois) <= max(in_ois)+1: first_occ = min(out_ois)
          (out-edge is temporal: intra/anchor); elif in_ois: first_occ = max(in_ois)+1
          (out-edge is inter-seg huge OR no out-edge: use arrival+1); else None (append
          last layer). -> last_token[i] first_occ = in_edge+1 = end-of-segment position;
          inter-seg chain forward.

    tiebreak normalized_max_in (v3; fixes anchor same-first_occ collision):
      anchor edge struct_ref[i]->token_0[i] oi = order_base_i; intra token_0->token_1 oi =
      order_base_i+0 = order_base_i (SAME). So struct_ref[i] and token_0[i] share
      first_occ = order_base_i. Need struct_ref[i] (from) before token_0[i] (to).
      norm_max_in = max([in_edge oi where oi <= first_occ+1], default=-1) (filters
      inter-seg huge in-edges; only temporal in-edges). struct_ref[i] (in-edge inter-seg
      huge > first_occ+1 -> norm=-1) before token_0[i] (in-edge anchor = first_occ ->
      norm=order_base_i). Layer-internal order = (first_occ, norm_max_in, ConceptRef).

    See doc/重来_F2_PRECEDES_oi遍历_设计_2026-07-09 Sec 3bis + 5bis (v3).
    """
    # Collect PRECEDES out/in oi per node (CAUSES order_index=None has no temporal oi).
    out_ois: dict[NodeRef, list[int]] = {}
    in_ois: dict[NodeRef, list[int]] = {}
    nodes: set[NodeRef] = set()
    for e in subgraph_edges:
        if e["edge_type"] != EDGE_PRECEDES:
            continue
        oi = e.get("order_index")
        if oi is None:
            continue   # PRECEDES should always have oi; defensive
        f = (e["space_id_from"], e["local_id_from"])
        t = (e["space_id_to"], e["local_id_to"])
        out_ois.setdefault(f, []).append(oi)
        in_ois.setdefault(t, []).append(oi)
        nodes.add(f)
        nodes.add(t)
    # Also include CAUSES-only nodes (no PRECEDES edge; rare in language; appended last).
    for e in subgraph_edges:
        if e["edge_type"] in head_types:
            nodes.add((e["space_id_from"], e["local_id_from"]))
            nodes.add((e["space_id_to"], e["local_id_to"]))

    entries: list[tuple[int, int, NodeRef]] = []   # (first_occ, norm_max_in, node)
    none_nodes: list[NodeRef] = []                  # no PRECEDES edge -> append last
    for node in nodes:
        outs = out_ois.get(node, [])
        ins = in_ois.get(node, [])
        max_in = max(ins) if ins else -1
        if outs and min(outs) <= max_in + 1:
            first_occ = min(outs)
        elif ins:
            first_occ = max_in + 1
        else:
            none_nodes.append(node)
            continue
        temporal_in = [oi for oi in ins if oi <= first_occ + 1]
        norm_max_in = max(temporal_in) if temporal_in else -1
        entries.append((first_occ, norm_max_in, node))
    entries.sort(key=lambda x: (x[0], x[1], x[2]))
    out: list[list[NodeRef]] = []
    cur_layer: list[NodeRef] = []
    cur_fo: int | None = None
    for fo, _nmi, node in entries:
        if cur_fo is None or fo == cur_fo:
            cur_layer.append(node)
            cur_fo = fo
        else:
            out.append(cur_layer)
            cur_layer = [node]
            cur_fo = fo
    if cur_layer:
        out.append(cur_layer)
    if none_nodes:
        none_nodes.sort()
        if out:
            out[-1].extend(none_nodes)
        else:
            out.append(none_nodes)
    return out


def a2_layer_oi(subgraph_edges: list[dict[str, Any]], active: set[NodeRef],
                head_types: set[int]) -> tuple[list[list[NodeRef]], dict, HeadStepper]:
    """F2 A2 oi-first-occurrence layering + convergence + stepper (drop Kahn; gate ON).

    Mirrors a2_layer signature; replaces Kahn layering with _build_topo_layers_oi
    (includes cyclic nodes). convergence + HeadStepper reuse a2_layer's (subgraph-based,
    independent of Kahn). Called by dag_path when PRECEDES_OI_MODE ON.
    """
    for e in subgraph_edges:
        assert_no_float(e["edge_type"], e.get("order_index") or 0,
                        _where="a2_layer_oi.edge")
    topo_layers = _build_topo_layers_oi(subgraph_edges, head_types)
    pred_index = _build_pred_index(subgraph_edges, head_types)
    convergence = _build_convergence(pred_index)
    stepper = HeadStepper(head_types, set(active), pred_index)
    return topo_layers, convergence, stepper


def has_cycle(topo_layers: list[list[NodeRef]],
              subgraph_edges: list[dict[str, Any]],
              head_types: set[int]) -> list[NodeRef]:
    """环检测（Kahn 余留 in_degree>0 者 = 环节点·loop_closure_defect）。

    返环节点列表（空=无环）。本版从 Kahn 重新算（a2_layer 不返 result·独立判定）。
    """
    combined: list[tuple[NodeRef, NodeRef]] = []
    for e in subgraph_edges:
        if e["edge_type"] not in head_types:
            continue
        combined.append(((e["space_id_from"], e["local_id_from"]),
                         (e["space_id_to"], e["local_id_to"])))
    result = kahn_topo(combined)
    return result.cycle_nodes
