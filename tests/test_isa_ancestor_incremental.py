"""#1115 增量 ancestor_map 测试（apply_isa_edge_to_map + build_isa_ancestor_map_with_index）。

bit-identical 核证：
  - DAG：逐边 apply_isa_edge_to_map == 全量 isa_ancestor_map（闭包单调 + 并集交换·序无关）。
  - 环：apply 返 False（环检测 fall back·守既有 SCC 契约·环走全量）。
  - desc_index 与 ancestor_map 双向一致（apply 维护正确）。
  - fresh copy：build_isa_ancestor_map_with_index 返深拷贝·mutate 不污染 backend gen-cache。

镜像 test_graph_algebra.py 既有环契约（SCC 凝聚·test_cycle_scc_bit_identical）。
"""
from __future__ import annotations

from pure_integer_ai.algorithm.graph_algebra import isa_ancestor_map
from pure_integer_ai.cognition.process.abstraction import (
    apply_isa_edge_to_map, build_isa_ancestor_map_with_index)
from pure_integer_ai.cognition.understanding.is_a import build_is_a_edge
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_types import EDGE_IS_A
from pure_integer_ai.storage.edge_store import EdgeStore, EPI_CUE


def _edges(pairs):
    """pairs [(child_id, parent_id)] → IS_A 边（from=child to=parent·space 0）。"""
    return [((0, c), (0, p), EDGE_IS_A, None) for c, p in pairs]


def _incremental_build(pairs):
    """逐边 apply·返 (ancestor_map, desc_index, any_cycle_false)。

    any_cycle_false = 是否有任一边 apply 返 False（环检测命中）。
    """
    amap: dict = {}
    didx: dict = {}
    any_cycle = False
    for c, p in pairs:
        if not apply_isa_edge_to_map(amap, didx, (0, c), (0, p)):
            any_cycle = True
    return amap, didx, any_cycle


def _eq(a, b):
    return {k: frozenset(v) for k, v in a.items()} == {k: frozenset(v) for k, v in b.items()}


# ===== DAG：增量 == 全量 =====

def test_incremental_chain_equals_full():
    pairs = [(1, 2), (2, 3)]   # 1→2→3
    amap, _didx, cyc = _incremental_build(pairs)
    full, _fb = isa_ancestor_map(_edges(pairs))
    assert cyc is False
    assert _eq(amap, full)
    assert amap[(0, 1)] == {(0, 2), (0, 3)}
    assert amap[(0, 2)] == {(0, 3)}


def test_incremental_diamond_equals_full():
    pairs = [(1, 2), (3, 2), (2, 4)]   # diamond 汇聚
    amap, _didx, cyc = _incremental_build(pairs)
    full, _fb = isa_ancestor_map(_edges(pairs))
    assert cyc is False
    assert _eq(amap, full)


def test_incremental_multi_parent_equals_full():
    pairs = [(5, 3), (5, 4), (3, 1), (3, 2), (4, 2)]   # 深宽混合
    amap, _didx, cyc = _incremental_build(pairs)
    full, _fb = isa_ancestor_map(_edges(pairs))
    assert cyc is False
    assert _eq(amap, full), "多父 branching DAG 增量 == 全量"


def test_incremental_batch_order_invariant():
    """同一边集·不同入序 apply·结果同（DAG 上序无关·LOW-3）。"""
    pairs = [(5, 3), (5, 4), (3, 1), (3, 2), (4, 2)]
    am1, _, _ = _incremental_build(pairs)
    am2, _, _ = _incremental_build(list(reversed(pairs)))
    assert _eq(am1, am2), "入序 vs 逆序·增量结果一致"
    full, _ = isa_ancestor_map(_edges(pairs))
    assert _eq(am1, full)


def test_incremental_add_to_existing_dag():
    """在已有 DAG 上追加边·增量 == 重建（模拟 observe 期间 build_is_a_edges 追加）。"""
    base = [(1, 2), (2, 3)]
    amap, didx, cyc = _incremental_build(base)
    assert cyc is False
    # 追加 (4→2)·4 的新祖先 = anc[2] ∪ {2} = {2,3}
    assert apply_isa_edge_to_map(amap, didx, (0, 4), (0, 2)) is True
    assert amap[(0, 4)] == {(0, 2), (0, 3)}
    # 追加 (5→1)·5 的新祖先 = anc[1] ∪ {1} = {1,2,3}
    assert apply_isa_edge_to_map(amap, didx, (0, 5), (0, 1)) is True
    assert amap[(0, 5)] == {(0, 1), (0, 2), (0, 3)}
    # 对比全量重建（base + 追加）
    full, _ = isa_ancestor_map(_edges(base + [(4, 2), (5, 1)]))
    assert _eq(amap, full), "追加边后增量 == 全量重建"


# ===== 环：apply 返 False（fall back）=====

def test_cycle_2_returns_false():
    """2-cycle (a→b)+(b→a)：第二条边 apply 返 False（闭环检测）。"""
    amap, didx = {}, {}
    assert apply_isa_edge_to_map(amap, didx, (0, 1), (0, 2)) is True    # 1→2 建
    assert apply_isa_edge_to_map(amap, didx, (0, 2), (0, 1)) is False   # 2→1 闭环→False
    # 第一边已建（amap[1]={2}）·第二边没改（无自身污染）
    assert amap[(0, 1)] == {(0, 2)}
    assert (0, 2) not in amap   # 2 的祖先没建（环边 fall back）


def test_cycle_3_returns_false():
    """3-cycle 1→2→3→1：第三边 3→1 闭环（1 是 3 的后代）→ False。"""
    amap, didx = {}, {}
    assert apply_isa_edge_to_map(amap, didx, (0, 1), (0, 2)) is True
    assert apply_isa_edge_to_map(amap, didx, (0, 2), (0, 3)) is True
    # 此时 anc[1]={2,3}·desc[2]={1}·desc[3]={1,2}·加 3→1：1 是 3 的后代（1 ∈ desc[3]? desc[3]={1,2} 含1）→ 闭环
    assert apply_isa_edge_to_map(amap, didx, (0, 3), (0, 1)) is False


def test_cycle_with_external_returns_false():
    """环 + 外部祖 A→B→C→A + B→D：C→A 闭环 False（D 仍进）。"""
    amap, didx = {}, {}
    assert apply_isa_edge_to_map(amap, didx, (0, 1), (0, 2)) is True    # A→B
    assert apply_isa_edge_to_map(amap, didx, (0, 2), (0, 4)) is True    # B→D（外部）
    assert apply_isa_edge_to_map(amap, didx, (0, 2), (0, 3)) is True    # B→C
    assert apply_isa_edge_to_map(amap, didx, (0, 3), (0, 1)) is False   # C→A 闭环
    # 外部 D 进了（amap[A] 含 B·D·C）
    assert (0, 4) in amap[(0, 1)]


# ===== 守卫 =====

def test_self_loop_noop():
    amap, didx = {(0, 1): {(0, 2)}}, {(0, 2): {(0, 1)}}
    assert apply_isa_edge_to_map(amap, didx, (0, 1), (0, 1)) is True   # child==parent → no-op
    assert amap == {(0, 1): {(0, 2)}}   # 不改
    assert didx == {(0, 2): {(0, 1)}}


def test_duplicate_edge_noop():
    """重复边（parent 已是 child 祖先）→ True 幂等 no-op·不改。"""
    amap, didx = {}, {}
    assert apply_isa_edge_to_map(amap, didx, (0, 1), (0, 2)) is True
    snap = ({k: set(v) for k, v in amap.items()}, {k: set(v) for k, v in didx.items()})
    assert apply_isa_edge_to_map(amap, didx, (0, 1), (0, 2)) is True   # 重复
    assert ({k: set(v) for k, v in amap.items()}, {k: set(v) for k, v in didx.items()}) == snap


# ===== desc_index 一致性 =====

def test_desc_index_consistent_with_ancestor_map():
    """desc_index[a] == {d : a ∈ ancestor_map[d]}（apply 维护双向一致）。"""
    pairs = [(5, 3), (5, 4), (3, 1), (3, 2), (4, 2)]
    amap, didx, cyc = _incremental_build(pairs)
    assert cyc is False
    # 反推 desc_index·对比 apply 维护的
    expected_didx: dict = {}
    for d, ancs in amap.items():
        for a in ancs:
            expected_didx.setdefault(a, set()).add(d)
    exp = {k: frozenset(v) for k, v in expected_didx.items()}
    got = {k: frozenset(v) for k, v in didx.items()}
    assert exp == got, "desc_index 与 ancestor_map 双向一致"


# ===== build_isa_ancestor_map_with_index：fresh copy + desc_index =====

def _seed_backend(pairs, space_id=0):
    """建 DictBackend + 注入 IS_A 边（from=child to=parent·via build_is_a_edge·bump isa_edge_generation）。"""
    from pure_integer_ai.storage import bootstrap
    backend = DictBackend()
    bootstrap(backend)
    es = EdgeStore(backend)
    for c, p in pairs:
        build_is_a_edge(es, (space_id, c), (space_id, p),
                        source=2, epistemic=EPI_CUE, space_id=space_id)
    return backend


def test_with_index_equals_full_and_desc_consistent():
    pairs = [(5, 3), (5, 4), (3, 1), (3, 2), (4, 2)]
    backend = _seed_backend(pairs)
    amap, didx = build_isa_ancestor_map_with_index(backend, space_id=0)
    full, _ = isa_ancestor_map(_edges(pairs))
    assert _eq(amap, full), "with_index ancestor_map == 全量"
    # desc_index 一致
    expected_didx: dict = {}
    for d, ancs in amap.items():
        for a in ancs:
            expected_didx.setdefault(a, set()).add(d)
    assert {k: frozenset(v) for k, v in didx.items()} == \
           {k: frozenset(v) for k, v in expected_didx.items()}


def test_with_index_fresh_copy_not_pollute_cache():
    """fresh copy：mutate with_index 返回的 map 不污染 backend gen-cache（审1 MED-3）。"""
    pairs = [(1, 2), (2, 3)]
    backend = _seed_backend(pairs)
    amap1, _didx1 = build_isa_ancestor_map_with_index(backend, space_id=0)
    # mutate fresh copy（模拟 observe 增量 apply）
    amap1[(0, 9)] = {(0, 99)}
    # 再取一次（走 backend gen-cache·应不受 fresh copy mutate 影响）
    amap2, _didx2 = build_isa_ancestor_map_with_index(backend, space_id=0)
    assert (0, 9) not in amap2, "fresh copy mutate 不污染 backend cache/gen-cache"
    assert _eq(amap2, {(0, 1): {(0, 2), (0, 3)}, (0, 2): {(0, 3)}})
