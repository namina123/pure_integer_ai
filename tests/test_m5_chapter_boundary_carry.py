"""#729 M5 真分页 + 长文本连贯 transient — 实施2 章边界 carry → prior_topic_refs 测试。

章边界（chapter_seq 变化点）触发 carry_to_workmem 写 prior_topic_refs（既有 stub 字段激活·
含 cap=k=16 FIFO 截断 + 去重）。caller 端 chap_filter
过滤排除新章首 unit（决断6）。五环闭环：读章边界→触发 carry→写 prior_topic_refs→
slot_dispatch collide_score 消化→选词受影响（实施3 e2e 验最后一环）。

设计文档：doc/重来_任务0729_M5真分页_长文本连贯_设计.md 决断4 + 决断6。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT, SUBTYPE_PURE_ALIAS
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.chapter_seq import register_chapter_seq, attach_chapter_seq
from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO
from pure_integer_ai.cognition.shared.types import (
    PathData, PathResult, ConceptRef,
    TERMINAL_REACHED_SINK, LANG_ZH,
)
from pure_integer_ai.cognition.shared.work_memory import (
    WorkMemory, PRODUCED_REFS_WINDOW, PRIOR_TOPIC_REFS_WINDOW,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.understanding.role_precedes import attach_role_seq
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.generate import generate_output


# ---- fixtures ----

@pytest.fixture
def core():
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    register_chapter_seq(b)
    yield b, sp.space_id, es, ci
    b.close()


def _dag(sid, *, sink=None, struct_unit_refs=None, topo_layers=None):
    return PathResult(
        path=PathData(edges=[], struct_unit_refs=struct_unit_refs or []),
        terminal=TERMINAL_REACHED_SINK, sink=sink,
        topo_layers=topo_layers or [], convergence={}, source=None,
    )


def _make_unit(b, sid, es, ci, name, chapter_seq, surface_map):
    """建 unit + word + role_seq[201] + REFERS_TO + chapter_seq attach。返 unit ref。"""
    u = ci.ensure(name, space_id=sid, tier=TIER_PRIMARY)
    w = ci.ensure(f"w_{name}", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, u, [201])
    es.add(space_id_from=sid, local_id_from=w[1], space_id_to=sid,
           local_id_to=u[1], edge_type=EDGE_REFERS_TO,
           strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
           subtype=SUBTYPE_PURE_ALIAS, sn=0, tn=0)
    attach_chapter_seq(b, ref=u, chapter_seq=chapter_seq, section_seq=1)
    surface_map[u] = name
    surface_map[w] = f"w_{name}"
    return u


# ============ add_prior_topic 单元测 ============

def test_add_prior_topic_dedup():
    """同 ref 两次 add → prior_topic_refs 单条（保序去重）。"""
    wm = WorkMemory()
    r = (0, 1)
    wm.add_prior_topic(r, window=16)
    wm.add_prior_topic(r, window=16)
    assert wm.prior_topic_refs == [r]


def test_add_prior_topic_fifo_cap():
    """window 满 del [:over] 截断保近期（末 window 个·最旧淘汰）。"""
    wm = WorkMemory()
    window = 5
    for i in range(window + 5):
        wm.add_prior_topic((0, i), window=window)
    assert len(wm.prior_topic_refs) == window
    assert wm.prior_topic_refs == [(0, i) for i in range(5, 10)]


def test_add_prior_topic_default_window_is_16():
    """add_prior_topic 不传 window → 用默认 PRIOR_TOPIC_REFS_WINDOW=16。"""
    assert PRIOR_TOPIC_REFS_WINDOW == 16
    wm = WorkMemory()
    for i in range(20):
        wm.add_prior_topic((0, i))
    assert len(wm.prior_topic_refs) == 16
    assert wm.prior_topic_refs == [(0, i) for i in range(4, 20)]


# ============ 章边界 carry e2e ============

def test_chapter_boundary_carry_writes_prior_topic_refs(core):
    """章边界（ch1→ch2）触发 carry → prior_topic_refs 含前章末 unit（五环闭环：读→触发→写）。"""
    b, sid, es, ci = core
    surface_map = {}
    u1 = _make_unit(b, sid, es, ci, "u1", chapter_seq=1, surface_map=surface_map)
    u2 = _make_unit(b, sid, es, ci, "u2", chapter_seq=1, surface_map=surface_map)
    u3 = _make_unit(b, sid, es, ci, "u3", chapter_seq=2, surface_map=surface_map)   # 章边界
    g = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
    dag = _dag(sid, sink=u3, struct_unit_refs=[u1, u2, u3], topo_layers=[[u1, u2, u3]])
    wm = WorkMemory()
    generate_output(dag, g, wm, LANG_ZH)
    # 章边界（u1/u2 ch1 → u3 ch2）触发 carry·chap_filter 过滤 ch1 → [u1, u2] 写 prior_topic_refs
    assert u1 in wm.prior_topic_refs
    assert u2 in wm.prior_topic_refs
    # 新章首 u3 不进 prior_topic_refs（决断6 排除）
    assert u3 not in wm.prior_topic_refs


def test_chapter_boundary_carry_excludes_new_chapter_unit(core):
    """决断6：章边界 carry 排除新章首 unit（已 append 进 parts·chap_filter 过滤 prev_chapter）。"""
    b, sid, es, ci = core
    surface_map = {}
    u1 = _make_unit(b, sid, es, ci, "u1", chapter_seq=1, surface_map=surface_map)
    u2 = _make_unit(b, sid, es, ci, "u2", chapter_seq=2, surface_map=surface_map)   # 章边界
    g = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
    dag = _dag(sid, sink=u2, struct_unit_refs=[u1, u2], topo_layers=[[u1, u2]])
    wm = WorkMemory()
    generate_output(dag, g, wm, LANG_ZH)
    # 前章 u1 进 prior_topic_refs·新章首 u2 排除
    assert wm.prior_topic_refs == [u1]
    assert u2 not in wm.prior_topic_refs


def test_no_chapter_seq_no_prior_topic_refs(core):
    """无 chapter_seq（表注册但 unit 无 attach）→ chap_no 恒 0 → 无边界 → prior_topic_refs 空。

    回归 bit-identical：现存测试零行为变。
    """
    b, sid, es, ci = core
    # unit 不 attach chapter_seq（chap_no 恒 0·无边界）
    u1 = ci.ensure("u1", space_id=sid, tier=TIER_PRIMARY)
    w1 = ci.ensure("w1", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, u1, [201])
    es.add(space_id_from=sid, local_id_from=w1[1], space_id_to=sid,
           local_id_to=u1[1], edge_type=EDGE_REFERS_TO,
           strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
           subtype=SUBTYPE_PURE_ALIAS, sn=0, tn=0)
    u2 = ci.ensure("u2", space_id=sid, tier=TIER_PRIMARY)
    w2 = ci.ensure("w2", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, u2, [202])
    es.add(space_id_from=sid, local_id_from=w2[1], space_id_to=sid,
           local_id_to=u2[1], edge_type=EDGE_REFERS_TO,
           strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
           subtype=SUBTYPE_PURE_ALIAS, sn=0, tn=0)
    surface_map = {u1: "u1", w1: "w1", u2: "u2", w2: "w2"}
    g = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
    dag = _dag(sid, sink=u2, struct_unit_refs=[u1, u2], topo_layers=[[u1, u2]])
    wm = WorkMemory()
    generate_output(dag, g, wm, LANG_ZH)
    # 无 chapter_seq → chap_no 恒 0 → 无边界 → prior_topic_refs 空
    assert wm.prior_topic_refs == []


def test_multi_chapter_prior_topic_refs_accumulate_and_cap(core):
    """多章累积 → prior_topic_refs cap=16 FIFO 截断保近期（最旧章末淘汰）。"""
    b, sid, es, ci = core
    surface_map = {}
    n_chapters = 20   # > 16 cap
    units = []
    for ch in range(1, n_chapters + 1):
        u = _make_unit(b, sid, es, ci, f"u{ch}", chapter_seq=ch, surface_map=surface_map)
        units.append(u)
    g = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
    dag = _dag(sid, sink=units[-1], struct_unit_refs=units,
               topo_layers=[units])
    wm = WorkMemory()
    generate_output(dag, g, wm, LANG_ZH)
    # 19 章边界 carry（ch1→ch2 ... ch19→ch20）·每章 1 unit·u1..u19 被 carry
    # （u20 是末章无后继边界不 carry）·prior_topic_refs cap=16 → 保 u4..u19（最旧 u1..u3 淘汰）
    assert len(wm.prior_topic_refs) == PRIOR_TOPIC_REFS_WINDOW
    assert wm.prior_topic_refs == units[3:19]   # u4..u19（index 3..18·16 个）
    assert units[0] not in wm.prior_topic_refs    # u1 最旧淘汰
    assert units[-1] not in wm.prior_topic_refs   # u20 末章不 carry
    assert units[3] in wm.prior_topic_refs        # u4 保留


def test_chapter_boundary_carry_bit_identical(core):
    """bit-identical：两独立跑同 dag → 同 prior_topic_refs + 同 produced_refs + 同 parts。"""
    b, sid, es, ci = core
    surface_map = {}
    units = []
    for ch in range(1, 6):
        u = _make_unit(b, sid, es, ci, f"u{ch}", chapter_seq=ch, surface_map=surface_map)
        units.append(u)
    g = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
    dag = _dag(sid, sink=units[-1], struct_unit_refs=units, topo_layers=[units])
    wm1 = WorkMemory()
    out1 = generate_output(dag, g, wm1, LANG_ZH)
    wm2 = WorkMemory()
    out2 = generate_output(dag, g, wm2, LANG_ZH)
    assert wm1.prior_topic_refs == wm2.prior_topic_refs
    assert wm1.produced_refs == wm2.produced_refs
    assert [p.words for p in out1.parts] == [p.words for p in out2.parts]


def test_chapter_boundary_carry_multi_unit_per_chapter(core):
    """多 unit/章：章边界 carry 整章 unit 子集（chap_filter 过滤 prev_chapter 全部 unit）。"""
    b, sid, es, ci = core
    surface_map = {}
    # 章 1：u1a, u1b, u1c · 章 2：u2a（章边界）
    u1a = _make_unit(b, sid, es, ci, "u1a", chapter_seq=1, surface_map=surface_map)
    u1b = _make_unit(b, sid, es, ci, "u1b", chapter_seq=1, surface_map=surface_map)
    u1c = _make_unit(b, sid, es, ci, "u1c", chapter_seq=1, surface_map=surface_map)
    u2a = _make_unit(b, sid, es, ci, "u2a", chapter_seq=2, surface_map=surface_map)
    g = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
    dag = _dag(sid, sink=u2a, struct_unit_refs=[u1a, u1b, u1c, u2a],
               topo_layers=[[u1a, u1b, u1c, u2a]])
    wm = WorkMemory()
    generate_output(dag, g, wm, LANG_ZH)
    # 章 1 全部 3 unit 进 prior_topic_refs·章 2 首 unit 排除
    assert u1a in wm.prior_topic_refs
    assert u1b in wm.prior_topic_refs
    assert u1c in wm.prior_topic_refs
    assert u2a not in wm.prior_topic_refs
    assert len(wm.prior_topic_refs) == 3


def test_mixed_chapter_seq_boundary_triggers_carry(core):
    """混合 chapter_seq（审2 P2-2）：unit1 无 chapter_seq（chap_no=0）→ unit2 chapter_seq=1（chap_no=1）
    ·0!=1 触发边界·chap_filter 过滤 chap_no==0 → [unit1] 写 prior_topic_refs。
    chap_no=0 是有效章号（无标记 unit 退化 0）·混合 case 边界正确触发。
    """
    b, sid, es, ci = core
    # unit1 不 attach chapter_seq（chap_no=0·无标记源退化）
    u1 = ci.ensure("u1", space_id=sid, tier=TIER_PRIMARY)
    w1 = ci.ensure("w1", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, u1, [201])
    es.add(space_id_from=sid, local_id_from=w1[1], space_id_to=sid,
           local_id_to=u1[1], edge_type=EDGE_REFERS_TO,
           strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
           subtype=SUBTYPE_PURE_ALIAS, sn=0, tn=0)
    # unit2 attach chapter_seq=1（chap_no=1·有标记）
    u2 = ci.ensure("u2", space_id=sid, tier=TIER_PRIMARY)
    w2 = ci.ensure("w2", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, u2, [202])
    es.add(space_id_from=sid, local_id_from=w2[1], space_id_to=sid,
           local_id_to=u2[1], edge_type=EDGE_REFERS_TO,
           strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
           subtype=SUBTYPE_PURE_ALIAS, sn=0, tn=0)
    attach_chapter_seq(b, ref=u2, chapter_seq=1, section_seq=1)
    surface_map = {u1: "u1", w1: "w1", u2: "u2", w2: "w2"}
    g = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
    dag = _dag(sid, sink=u2, struct_unit_refs=[u1, u2], topo_layers=[[u1, u2]])
    wm = WorkMemory()
    generate_output(dag, g, wm, LANG_ZH)
    # 混合 case：unit1 chap_no=0 → unit2 chap_no=1·0!=1 触发边界
    # chap_filter 过滤 chap_no==0 → [unit1]·写 prior_topic_refs
    assert u1 in wm.prior_topic_refs
    assert u2 not in wm.prior_topic_refs   # 新章首排除
    assert wm.prior_topic_refs == [u1]
