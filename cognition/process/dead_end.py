"""cognition.process.dead_end — 模块6 死路检测（二条件·§十三D-E3·M4）。

死路二条件任一触发（① 已废·2026-07-02 设计缺漏修正）：
  ① ~~无后继节点（出度=0 且非 sink）~~ — 已删：dag_path_step 全拓扑层遍历·
     非 sink 叶子是正常分支终点非死路·① 误杀"sink 可达但叶子先被访"阻断 reward>0（致命7）。
     sink 不可达靠 dag_path_step 末行层尽返回 DEAD_END。
  ② 候选前驱全不 active（汇聚点 OR 语义·CAUSES 前驱 rate 全 0 或不可达·无因则无果）
  ③ 步数上限耗尽（step_budget·防超长·环靠 Kahn 检测非此）

死路 reward<0 → tn++（防塌柱② 真负通路·破永正防塌靠 failure 非负值）。
死路 vs 正常达 sink 边界 = 是否达 sink ∧ J4 闭合（J4 卷三真判·卷二占位 true）。

纯整数·确定性。诚实边界：死路是结构判定非语义判定（"无后继"是图结构非"推理无解"·
CAUSES OR 零 active 前驱是"无因则无果的正确行为"非错误·§十三D）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.cognition.shared.types import ConceptRef, IntentType
from pure_integer_ai.storage.edge_types import EDGE_CAUSES
from pure_integer_ai.cognition.process.effective_weight import edge_rate, is_unobserved
from pure_integer_ai.config import gates

NodeRef = ConceptRef


def out_degree(node: NodeRef, subgraph_edges: list[dict[str, Any]]) -> int:
    """节点出度（subgraph 内 from==node 的边数·含各头）。"""
    sid, lid = node
    n = 0
    for e in subgraph_edges:
        if e["space_id_from"] == sid and e["local_id_from"] == lid:
            n += 1
    return n


def causes_predecessor_edges(node: NodeRef,
                             subgraph_edges: list[dict[str, Any]]
                             ) -> list[dict[str, Any]]:
    """CAUSES 前驱边列表（edge.to==node · edge.et==CAUSES·D3 返前驱边非节点）。"""
    sid, lid = node
    return [e for e in subgraph_edges
            if e["edge_type"] == EDGE_CAUSES
            and e["space_id_to"] == sid and e["local_id_to"] == lid]


def is_dead_end(node: NodeRef, subgraph_edges: list[dict[str, Any]],
                intent: IntentType, active: set[NodeRef],
                path_len: int, step_budget: int,
                *, stepper: Any = None) -> bool:
    """死路三条件判定（D7·补 intent + active）。

    node          当前节点。
    subgraph_edges 热区相关边集（卷一建好·audit_float==0）。
    intent        意图（含 sink·终点判定）。
    active        当前 active 节点集（stepper.active·D7 补）。
    path_len      已步数（len(path.steps)）。
    step_budget   步数上限（段内拓扑层数 × SAFETY_FACTOR·M4·非全图固定 STEP_LIMIT）。
    stepper       可选 HeadStepper（perf round5·提供则用 stepper.predecessors O(1) 查替
                  causes_predecessor_edges 全扫·解每 node 全扫 O(n×m)）。bit-identical：② 结果
                  序无关（all_blocked = NOT(any 非阻塞前驱)·OR+break 优化·sorted 序与迭代序同判定）。
    """
    sink = intent.sink
    # ① 已废（2026-07-02·设计缺漏修正·全遍历下叶子非死路·sink 不可达靠 dag_path 层尽返回）
    _ = sink
    # ② 候选前驱全不 active（汇聚点 OR 语义·CAUSES 前驱率全 0 或不可达·无因则无果）
    # F2（PRECEDES_OI_MODE·factor C 修）：PRECEDES oi-first-occ 链是 path 主体·节点经 PRECEDES 已 active（已到达）·
    # 其 CAUSES 前驱若 backward（later token·§八代价1·访节点时未 active）·② 会误杀整 path 致 sink 不可达
    # （factor D·F2 暴露·531 实测：PRECEDES 530->531 在 path 但 CAUSES backward·② 杀 path）。
    # 设计 §八代价1 只说 backward CAUSES **丢**（不收集入 path.edges）·未说 ② **杀 path**。② 与 PRECEDES-链冲突。
    # gate ON -> 跳过 ②（path 靠 PRECEDES 链 + budget ③ + sink-check + 层尽·backward CAUSES 丢不致命）·
    # gate OFF -> 原 ②（bit-identical·② 单测 OI_MODE OFF 不受影响）。
    if not bool(getattr(gates, "PRECEDES_OI_MODE", False)):
        # perf round5：stepper.predecessors（_build_pred_index 预建 O(1) 查）替 causes_predecessor_edges 全扫。
        preds = (stepper.predecessors(node, EDGE_CAUSES)
                 if stepper is not None
                 else causes_predecessor_edges(node, subgraph_edges))
        if preds:
            all_blocked = True
            for e in preds:
                from_node = (e["space_id_from"], e["local_id_from"])
                # item3 缺漏2：未观测边（sn=tn=0）不算 blocked（冷启动给机会）·已失败边（sn=0 tn>0）算 blocked
                if from_node in active and (edge_rate(e) > 0 or is_unobserved(e)):
                    all_blocked = False
                    break
            if all_blocked:
                return True
    # ③ 步数上限耗尽（防超长·环靠 Kahn 检测非此）
    if path_len >= step_budget:
        return True
    return False
