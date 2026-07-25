"""#1133 测试：abstract IS_A cycle-cleaning 算法锁（graph_algebra.break_back_edges·DFS back-edge removal）。

cleaning = DFS back-edge removal（保 tree/forward/cross·只删 back-edge·CLRS）→ 干净 DAG。
``break_back_edges`` 是纯图原语（adj→删除集·无 I/O·无语义判定）·同居 graph_algebra 与 ``_tarjan_scc``·
被 ``scratch/clean_abstract_cycles``（数据 prep 胶水·读 K:/ raw → 写 data_llm/ 干净 DAG）调用。

本测锁算法正确性（人造环破 + 保结构 + 确定性 + 子集 of DAG is DAG）·反 theater（cleaning 真破环可验）。

测：
  AC1 3-node 环 A→B→C→A → 恰删 1 back-edge → DAG
  AC2 2-cycle A↔B + forward appendage B→C → 删 1·appendage 保·DAG
  AC3 已无环链 A→B→C → 删 0（DAG 上幂等）
  AC4 确定性：同输入两跑同删除集（bit-identical）
  AC5 多 SCC（两 disjoint 环 + forward 桥）→ 每环破 ≥1·桥保·DAG
  AC6 子集 of DAG is DAG（cleaned DAG 的 corpus-relevant 子集无环·#1142+#1133 交互 crux）

铁律：纯整数（NodeRef 整数二元组）/ 确定性（sorted starts/neighbors·bit-identical）/ 反 theater。
"""
from __future__ import annotations

from pure_integer_ai.algorithm.graph_algebra import _tarjan_scc, break_back_edges

# 节点（NodeRef = (space, local) 整数二元组）
A, B, C = (1, 1), (1, 2), (1, 3)
X, Y = (2, 1), (2, 2)
P, Q, R = (3, 1), (3, 2), (3, 3)


def _cyclic_scc_count(nodes, adj):
    """cyclic SCC（>1 节点）数——0 = DAG。"""
    sccs = _tarjan_scc(nodes, adj)
    return sum(1 for s in sccs if len(s) > 1)


def _apply_remove(nodes, adj, remove):
    """apply 删除集 → cleaned adj（保 nodes 集·孤立节点成 singleton）。"""
    radj = {}
    for n, nbs in adj.items():
        kept = [m for m in nbs if (n, m) not in remove]
        if kept:
            radj[n] = kept
    return radj


def test_ac1_three_cycle_breaks_to_dag():
    """3-node 环 A→B→C→A → 恰删 1 back-edge（非 2/3·保 tree/forward）→ cleaned 是 DAG。"""
    nodes = {A, B, C}
    adj = {A: [B], B: [C], C: [A]}
    assert _cyclic_scc_count(nodes, adj) == 1   # 一 SCC 含 3 节点（环）
    remove = break_back_edges(nodes, adj)
    assert len(remove) == 1, f"3-cycle 应恰删 1 back-edge·得 {remove}"
    assert remove.issubset({(A, B), (B, C), (C, A)}), "删除集只含实际环边"
    cleaned = _apply_remove(nodes, adj, remove)
    assert _cyclic_scc_count(nodes, cleaned) == 0, "cleaned 须无环（DAG）"


def test_ac2_two_cycle_plus_appendage_preserves_forward():
    """2-cycle A↔B + forward appendage B→C（C 不在环）→ 删 1·B→C 保·DAG。

    保 tree/forward 边（非全砍 intra-SCC）——appendage B→C 非 back-edge 不删。
    """
    nodes = {A, B, C}
    adj = {A: [B], B: [A, C]}   # A→B, B→A（环）+ B→C（forward·C 树叶）
    remove = break_back_edges(nodes, adj)
    assert len(remove) == 1, f"2-cycle 应恰删 1·得 {remove}"
    assert remove.issubset({(A, B), (B, A)}), "删除集只含环内边 A↔B"
    assert (B, C) not in remove, "forward appendage B→C 须保（非 back-edge）"
    cleaned = _apply_remove(nodes, adj, remove)
    assert _cyclic_scc_count(nodes, cleaned) == 0
    # C 仍可达（B→C 保）
    assert C in cleaned.get(B, []), "appendage B→C 保留"


def test_ac3_acyclic_is_idempotent():
    """已无环链 A→B→C → 删 0（DAG 上幂等·无 back-edge）。"""
    nodes = {A, B, C}
    adj = {A: [B], B: [C]}
    assert _cyclic_scc_count(nodes, adj) == 0
    remove = break_back_edges(nodes, adj)
    assert remove == set(), "acyclic 图应删 0（幂等）"


def test_ac4_determinism_two_runs_identical():
    """同输入两跑 → 同删除集（sorted starts/neighbors + 显式索引 → bit-identical）。"""
    nodes = {A, B, C, X, Y, P, Q, R}
    adj = {A: [B], B: [C], C: [A],   # 3-cycle
           X: [Y], Y: [X],            # 2-cycle
           P: [Q], Q: [R], R: [P]}    # 3-cycle
    r1 = break_back_edges(nodes, adj)
    r2 = break_back_edges(nodes, adj)
    assert r1 == r2, "确定性：同输入同删除集"
    cleaned = _apply_remove(nodes, adj, r1)
    assert _cyclic_scc_count(nodes, cleaned) == 0, "三 disjoint 环皆破 → DAG"


def test_ac5_multi_scc_forward_bridge_preserved():
    """两 disjoint 环 + forward 桥（Y→P）→ 每环破 ≥1·桥保·全局 DAG。"""
    nodes = {X, Y, P, Q, R}
    adj = {X: [Y], Y: [X, P],   # X↔Y 环 + Y→P 桥（forward）
           P: [Q], Q: [R], R: [P]}   # P→Q→R→P 环
    remove = break_back_edges(nodes, adj)
    assert (Y, P) not in remove, "forward 桥 Y→P 须保（跨 SCC·非 back-edge）"
    # 每环至少破 1（X↔Y 破 1·PQR 破 1）
    xy_edges = {(X, Y), (Y, X)}
    pqr_edges = {(P, Q), (Q, R), (R, P)}
    assert any(e in remove for e in xy_edges), "X↔Y 环须破 ≥1"
    assert any(e in remove for e in pqr_edges), "PQR 环须破 ≥1"
    cleaned = _apply_remove(nodes, adj, remove)
    assert _cyclic_scc_count(nodes, cleaned) == 0, "两环皆破 → 全局 DAG"


def test_ac6_subset_of_dag_is_dag():
    """cleaned DAG 的 corpus-relevant 子集（#1142 boot vocab 过滤）仍无环——子集 of DAG is DAG。

    决断3 crux：cycle-cleaned 全图是 DAG → 任一子集（corpus-relevant filter 后）亦 DAG →
    graph_algebra SCC 凝聚对 abstract 子集退化为单节点 SCC（环防御不触发）。
    """
    # 含环原图 → clean → DAG
    nodes = {A, B, C, X, Y}
    adj = {A: [B], B: [C], C: [A],   # 3-cycle
           X: [Y]}                    # X→Y 树
    remove = break_back_edges(nodes, adj)
    cleaned = _apply_remove(nodes, adj, remove)
    assert _cyclic_scc_count(nodes, cleaned) == 0
    # corpus-relevant 子集：只留 {A, B, X} 间边（drop C·drop Y·模拟 vocab 过滤）
    subset_nodes = {A, B, X}
    subset_adj = {n: [m for m in cleaned.get(n, []) if m in subset_nodes]
                  for n in subset_nodes}
    subset_adj = {n: m for n, m in subset_adj.items() if m}
    assert _cyclic_scc_count(subset_nodes, subset_adj) == 0, \
        "DAG 的任一子集须仍 DAG（子集 of DAG 性质·#1142+#1133 交互守）"
