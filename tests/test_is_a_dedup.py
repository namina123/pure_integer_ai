"""#1115 修法 X 专项测：build_is_a_edges 双向 IS_A 矛盾源头去重。

验审1 MED-1：boot A→B + observe 系词反向 (B→A) → X skip observe 边（first-observed wins）→
backend 仅 boot 1 边（无环）→ ancestor_map 无 SCC>1（DAG）→ apply_isa_edge_to_map 增量不返
False（hoist 不失效·免整 space 退化全量重建）。

**bit-identical**：CI default（CUE_EXTRACTOR_MODE OFF·caller 传 is_a_pairs=[]）→ 循环 0 次 →
返 0（逐字现状）。本测直调 build_is_a_edges（gate 上游已解耦·验机制本身）。

设计档：doc/重来_observe性能_#1115_修法设计_2026-07-18.md §14。
"""
from __future__ import annotations

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_CONCEPTNET, EPI_STRUCTURED
from pure_integer_ai.storage.edge_types import EDGE_IS_A
from pure_integer_ai.cognition.understanding.is_a import build_is_a_edges, build_is_a_edge
from pure_integer_ai.cognition.process.abstraction import (
    build_isa_ancestor_map, apply_isa_edge_to_map, build_isa_ancestor_map_with_index)


def _make():
    backend = DictBackend()
    bootstrap(backend)
    es = EdgeStore(backend)
    return backend, es


def _isa_rows(backend, space=1):
    return backend.select("edge", where={"edge_type": EDGE_IS_A, "space_id_from": space})


def test_reverse_skip_boot_then_observe():
    """boot A→B（ConceptNet 结构化源①）· observe 系词反向 (B→A) → X skip·仅 boot 1 边。"""
    backend, es = _make()
    s = 1
    A, B = (s, 100), (s, 200)
    build_is_a_edge(es, A, B, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED, space_id=s)
    # observe 系词反向：child=B parent=A（B IsA A·与 boot A IsA B 反向矛盾）
    n = build_is_a_edges(es, [A, B], is_a_pairs=[(1, 0)], source=99, space_id=s)
    assert n == 0   # X skip observe 边
    rows = _isa_rows(backend, s)
    assert len(rows) == 1   # 仅 boot 边
    assert rows[0]["local_id_from"] == 100 and rows[0]["local_id_to"] == 200   # A→B 保留


def test_same_direction_builds():
    """boot A→B · observe 同向 (A→B) → 反向 (B→A) 不存在 → X 不 skip → 建 observe 边。"""
    backend, es = _make()
    s = 1
    A, B = (s, 100), (s, 200)
    build_is_a_edge(es, A, B, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED, space_id=s)
    n = build_is_a_edges(es, [A, B], is_a_pairs=[(0, 1)], source=99, space_id=s)   # child=A parent=B 同向
    assert n == 1   # 不 skip
    rows = _isa_rows(backend, s)
    assert len(rows) == 2   # boot + observe


def test_no_reverse_builds_cold():
    """无 boot · observe (A→B) → 反向空 → 建。"""
    backend, es = _make()
    s = 1
    A, B = (s, 100), (s, 200)
    n = build_is_a_edges(es, [A, B], is_a_pairs=[(0, 1)], source=99, space_id=s)
    assert n == 1


def test_empty_pairs_zero():
    """CI default（is_a_pairs=[]）→ 循环 0 次 → 返 0（逐字现状·bit-identical 心脏）。"""
    backend, es = _make()
    n = build_is_a_edges(es, [(1, 100), (1, 200)], is_a_pairs=[], source=99, space_id=1)
    assert n == 0
    assert _isa_rows(backend, 1) == []


def test_same_batch_first_observed_wins():
    """同段 is_a_pairs=[(A→B),(B→A)] → first 建·second 查反向命中 skip（first-observed wins·MED-2）。"""
    backend, es = _make()
    s = 1
    A, B = (s, 100), (s, 200)
    # (0,1)=A→B 先建·(1,0)=B→A 后查反向（A→B 已建于本段）→ skip
    n = build_is_a_edges(es, [A, B], is_a_pairs=[(0, 1), (1, 0)], source=99, space_id=s)
    assert n == 1
    rows = _isa_rows(backend, s)
    assert len(rows) == 1
    assert rows[0]["local_id_from"] == 100 and rows[0]["local_id_to"] == 200


def test_skip_yields_dag_ancestor_map_and_hoist_ok():
    """boot A→B + observe 反向 skip → ancestor_map 无环（DAG·SCC 全单节点）·
    apply_isa_edge_to_map 增量不返 False（hoist 不失效·审1 MED-1 核心）。"""
    backend, es = _make()
    s = 1
    A, B = (s, 100), (s, 200)
    build_is_a_edge(es, A, B, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED, space_id=s)
    build_is_a_edges(es, [A, B], is_a_pairs=[(1, 0)], source=99, space_id=s)   # 反向 skip
    # ancestor_map：A 的祖先含 B（A IsA B）·B 的祖先不含 A（无反向·无环·DAG）
    amap = build_isa_ancestor_map(backend, space_id=s)
    assert B in amap.get(A, set())
    assert A not in amap.get(B, set())
    # apply_isa_edge_to_map 在此 DAG 上加新边（不闭环）→ 返 True（增量成功·非环 fallback）
    fresh, didx = build_isa_ancestor_map_with_index(backend, space_id=s)
    C = (s, 300)
    ok = apply_isa_edge_to_map(fresh, didx, C, A)   # C IsA A（A 是 C 祖先·DAG 边）
    assert ok is True


def test_cross_seg_first_observed_wins():
    """段1 建 (A→B) · 段2 observe 反向 (B→A) → 段2 查反向命中段1 → skip（跨段 first-observed wins）。"""
    backend, es = _make()
    s = 1
    A, B = (s, 100), (s, 200)
    # 段1：observe 建 A→B（无 boot·反向空→建）
    n1 = build_is_a_edges(es, [A, B], is_a_pairs=[(0, 1)], source=99, space_id=s)
    assert n1 == 1
    # 段2：observe 反向 B→A → 查反向命中段1 的 A→B → skip
    n2 = build_is_a_edges(es, [A, B], is_a_pairs=[(1, 0)], source=99, space_id=s)
    assert n2 == 0
    rows = _isa_rows(backend, s)
    assert len(rows) == 1   # 仅段1 的 A→B


def test_multihop_cycle_skip_keeps_hoist_post_impl_low2():
    """§14.9 反转 caller 不变量（post-impl 审 LOW-2）：多跳环 apply 返 False·caller skip（不 invalidate）·hoist 保留。

    boot A→B→C（A IsA B IsA C）+ observe C→A（多跳反向环）：
    - apply_isa_edge_to_map(C, A) 环检测命中（A ∈ desc_index[C]·boot A IsA...C）→ 返 False。
    - §14.9 caller（observe.py:345）continue skip·不 apply·不 invalidate（hoist[space] 不置 None）。
    - hoist map 保留 boot（A 的祖先仍 B,C）·不含 C IsA A（环边 skip·A 不是 C 祖先）。

    锁不变量：未来若误改回 invalidate（hoist None）/ break（漏后续边）·本测 catch。
    注：caller continue 逻辑在 observe.py:345（直白 if not apply: continue）·本测验 apply 返 False + skip 语义
    （map 保留 boot 不含环边）·非 ObservePipeline e2e（核心不变量已锁）。
    """
    backend, es = _make()
    s = 1
    A, B, C = (s, 100), (s, 200), (s, 300)
    # boot 多跳链 A IsA B IsA C（A 的祖先是 B,C）
    build_is_a_edge(es, A, B, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED, space_id=s)
    build_is_a_edge(es, B, C, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED, space_id=s)
    # hoist 首建（with_index·含 boot）
    fresh, didx = build_isa_ancestor_map_with_index(backend, space_id=s)
    assert B in fresh.get(A, set()) and C in fresh.get(A, set())   # boot 传递闭包
    # observe C→A（多跳反向环）apply：环检测命中（A ∈ desc_index[C]）→ 返 False
    ok = apply_isa_edge_to_map(fresh, didx, C, A)
    assert ok is False   # 多跳环·apply 拒绝（审1 HIGH-1 环检测仍 active）
    # §14.9 caller skip：map 不含 C IsA A（环边未 apply）·boot 保留（未被污染）
    assert A not in fresh.get(C, set())   # C 的祖先不含 A（环边 skip）
    assert B in fresh.get(A, set()) and C in fresh.get(A, set())   # boot A:{B,C} 保留
