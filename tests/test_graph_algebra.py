"""#1136 algorithm.graph_algebra.isa_ancestor_map 测试。

bit-identical 核证：DAG 下 isa_ancestor_map（拓扑序单遍 O(V+E)）输出 == closure.transitive_closure
（BFS-per-source O(V·E)）祖先集。环 → fallback closure（fell_back=True·守 bit-identical/正确）。
"""
from __future__ import annotations

from pure_integer_ai.algorithm.graph_algebra import isa_ancestor_map
from pure_integer_ai.algorithm.closure import transitive_closure
from pure_integer_ai.storage.edge_types import EDGE_IS_A


def _edges(pairs):
    """pairs [(child_id, parent_id)] → IS_A 边（from=child to=parent·space 0）。"""
    return [((0, c), (0, p), EDGE_IS_A, None) for c, p in pairs]


def _closure_map(edges):
    cl = transitive_closure(edges, types={EDGE_IS_A}, include_direct=True)
    m = {}
    for (c, a, _et) in cl:
        m.setdefault(c, set()).add(a)
    return m


def _eq(a, b):
    return {k: frozenset(v) for k, v in a.items()} == {k: frozenset(v) for k, v in b.items()}


def test_chain_bit_identical():
    edges = _edges([(1, 2), (2, 3)])   # 1→2→3（child→parent）
    amap, fb = isa_ancestor_map(edges)
    assert fb is False
    assert _eq(amap, _closure_map(edges))
    assert amap[(0, 1)] == {(0, 2), (0, 3)}
    assert amap[(0, 2)] == {(0, 3)}
    assert (0, 3) not in amap   # 根（无祖先）不入图


def test_diamond_bit_identical():
    # 猫(1)→动物(2)·狗(3)→动物(2)·动物(2)→生物(4)（diamond 汇聚）
    edges = _edges([(1, 2), (3, 2), (2, 4)])
    amap, fb = isa_ancestor_map(edges)
    assert fb is False
    assert _eq(amap, _closure_map(edges))
    assert amap[(0, 1)] == {(0, 2), (0, 4)}
    assert amap[(0, 3)] == {(0, 2), (0, 4)}
    assert amap[(0, 2)] == {(0, 4)}


def test_multi_parent_branching_bit_identical():
    # 深宽混合：5→{3,4}·3→{1,2}·4→2（多父 + 共享祖 + diamond）
    edges = _edges([(5, 3), (5, 4), (3, 1), (3, 2), (4, 2)])
    amap, fb = isa_ancestor_map(edges)
    assert fb is False
    assert _eq(amap, _closure_map(edges)), "多父 branching DAG bit-identical closure"


def test_cycle_scc_bit_identical():
    """2-cycle：SCC 凝聚处理（无须 fallback）·输出 bit-identical closure·环节点互祖。"""
    edges = _edges([(1, 2), (2, 1)])
    amap, fb = isa_ancestor_map(edges)
    assert fb is False   # SCC 凝聚处理环（凝聚图是 DAG·无须 fallback）
    assert _eq(amap, _closure_map(edges)), "SCC 凝聚环输出 bit-identical closure"
    assert amap[(0, 1)] == {(0, 2)} and amap[(0, 2)] == {(0, 1)}   # 互祖


def test_cycle_with_external_scc_bit_identical():
    """3-cycle + 外部祖先：SCC 互祖 + 共享外部·bit-identical closure。A→B→C→A + B→D(外部)。"""
    edges = _edges([(1, 2), (2, 3), (3, 1), (2, 4)])   # {1,2,3} 环·4 外部祖
    amap, fb = isa_ancestor_map(edges)
    assert fb is False
    assert _eq(amap, _closure_map(edges)), "3-cycle+外部 SCC bit-identical closure"
    # 1 的祖先 = 环内 {2,3} + 外部 {4}
    assert amap[(0, 1)] == {(0, 2), (0, 3), (0, 4)}


def test_empty_and_self_loop():
    amap, fb = isa_ancestor_map([])
    assert fb is False and amap == {}
    # 自环跳（同 is_a.py:57 / closure:80）
    edges = _edges([(1, 1), (1, 2)])
    amap, fb = isa_ancestor_map(edges)
    assert fb is False
    assert amap[(0, 1)] == {(0, 2)}
