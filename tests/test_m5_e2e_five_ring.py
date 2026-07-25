"""#729 M5 真分页 + 长文本连贯 transient — 实施3 反 theater 256 触发 + e2e 五环闭环测。

反 theater：LAYER_UNIT_CAP 触发段满 carry 路径（决断5 缺口·test_exp_count:368 未测）。
e2e 五环闭环：读章边界→触发 carry→写 prior_topic_refs→slot_dispatch collide_score 消化→选词受影响。
  - with chapter_seq：章边界 carry 写 prior_topic_refs·U1（capped out of produced_refs）经
    prior_topic_refs 进 ctx_refs·collide_score(W2a) > collide_score(W2b) → 选 W2a
  - contrast 无 chapter_seq：prior_topic_refs 空·U1 不进 ctx·collide_score 持平 → tiebreak 选 W2b

设计文档：doc/重来_任务0729_M5真分页_长文本连贯_设计.md 决断4/5/6 + 实施3。
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
from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO, EDGE_COOCCURS
from pure_integer_ai.cognition.shared.types import (
    PathData, PathResult, ConceptRef,
    TERMINAL_REACHED_SINK, LANG_ZH, LANG_NONE,
    LINEAGE_CONCEPT_FILL,
)
from pure_integer_ai.cognition.shared.work_memory import (
    WorkMemory, PRODUCED_REFS_WINDOW, PRIOR_TOPIC_REFS_WINDOW,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.understanding.role_precedes import attach_role_seq
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result import generate as gen_mod
from pure_integer_ai.cognition.result.generate import generate_output, LAYER_UNIT_CAP


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


def _edge(es, sid, frm, to, et, *, strength=1, subtype=None):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid,
           local_id_to=to, edge_type=et, strength=strength,
           source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY, subtype=subtype, sn=0, tn=0)


# ============ 反 theater：LAYER_UNIT_CAP 段满 carry 触发路径（决断5 缺口） ============

def test_layer_unit_cap_triggers_carry_path(core, monkeypatch):
    """反 theater：LAYER_UNIT_CAP 触发段满 carry 路径（256 代理·monkeypatch LAYER_UNIT_CAP=5）。

    test_experience_count_chapter_seq.py:368 只测无 chapter_seq 无 carry·未测 256 触发。
    本测 monkeypatch LAYER_UNIT_CAP=5（避免建 256 unit fixture·性能）·建 6 unit → 段满触发。
    验 carry_to_workmem 被调·produced_refs cap=48 全程 append-only·输出 bit-identical。
    reload_next_layer 维持 no-op（决断1·真分页 Stage 6 接线·反 theater 标注）。
    """
    b, sid, es, ci = core
    n_units = 6   # > monkeypatched LAYER_UNIT_CAP=5
    units = []
    surface_map = {}
    for i in range(n_units):
        u = ci.ensure(f"u{i}", space_id=sid, tier=TIER_PRIMARY)
        w = ci.ensure(f"w{i}", space_id=sid, tier=TIER_PRIMARY)
        attach_role_seq(b, u, [201])
        _edge(es, sid, w[1], u[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
        units.append(u)
        surface_map[u] = f"u{i}"
        surface_map[w] = f"w{i}"
    # mock dispatch_slot 返固定词（性能·避免 collide_score 复杂度·本测只验 carry 触发）
    monkeypatch.setattr(gen_mod, "dispatch_slot",
                        lambda slot, dag, g, wm, tl: ("x", LINEAGE_CONCEPT_FILL))
    # monkeypatch LAYER_UNIT_CAP=5（256 代理·测段满触发路径）
    monkeypatch.setattr(gen_mod, "LAYER_UNIT_CAP", 5)
    # track carry_to_workmem 调用
    calls = []
    orig_carry = gen_mod.carry_to_workmem
    def track_carry(wm, parts):
        calls.append(len(parts))
        orig_carry(wm, parts)
    monkeypatch.setattr(gen_mod, "carry_to_workmem", track_carry)
    g = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
    dag = _dag(sid, sink=units[-1], struct_unit_refs=units, topo_layers=[units])
    wm = WorkMemory()
    out = generate_output(dag, g, wm, LANG_ZH)
    assert len(out.parts) == n_units
    # 段满 carry 触发（6 > 5·至少 1 次）
    assert len(calls) >= 1
    # produced_refs cap=48 全程 append-only（6 unit < 48 cap·全保留）
    assert len(wm.produced_refs) == n_units
    assert wm.produced_refs == units
    # reload 维持 no-op（决断1·无 reload 函数·produced_refs 不重置）


def test_layer_unit_cap_trigger_bit_identical(core, monkeypatch):
    """bit-identical：LAYER_UNIT_CAP 触发段满 carry 两独立跑 → 同 produced_refs + 同输出。"""
    b, sid, es, ci = core
    n_units = 7   # > monkeypatched LAYER_UNIT_CAP=5
    units = []
    surface_map = {}
    for i in range(n_units):
        u = ci.ensure(f"u{i}", space_id=sid, tier=TIER_PRIMARY)
        w = ci.ensure(f"w{i}", space_id=sid, tier=TIER_PRIMARY)
        attach_role_seq(b, u, [201])
        _edge(es, sid, w[1], u[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
        units.append(u)
        surface_map[u] = f"u{i}"
        surface_map[w] = f"w{i}"
    monkeypatch.setattr(gen_mod, "dispatch_slot",
                        lambda slot, dag, g, wm, tl: ("x", LINEAGE_CONCEPT_FILL))
    monkeypatch.setattr(gen_mod, "LAYER_UNIT_CAP", 5)
    g = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
    dag = _dag(sid, sink=units[-1], struct_unit_refs=units, topo_layers=[units])
    wm1 = WorkMemory()
    out1 = generate_output(dag, g, wm1, LANG_ZH)
    wm2 = WorkMemory()
    out2 = generate_output(dag, g, wm2, LANG_ZH)
    assert wm1.produced_refs == wm2.produced_refs
    assert [p.words for p in out1.parts] == [p.words for p in out2.parts]


# ============ e2e 五环闭环：章边界 carry → prior_topic_refs → collide_score → 选词 ============

def _build_two_chapter_fixture(b, sid, es, ci, *, with_chapter_seq: bool):
    """建两章 fixture：ch1=U1·ch2=U2..U51（50 unit 填 produced_refs cap=48 顶出 U1）。

    U51 候选 W2b（低 lid·tiebreak 优先）+ W2a（高 lid）·W2a COOCCURS U1（跨章 ctx）。
    with_chapter_seq=True：attach chapter_seq（章边界 carry·U1 进 prior_topic_refs）。
    with_chapter_seq=False：不 attach（无章边界·U1 不进 prior_topic_refs·被 produced_refs 顶出）。
    """
    surface_map = {}
    # ch1: U1 + W1
    u1 = ci.ensure("u1", space_id=sid, tier=TIER_PRIMARY)
    w1 = ci.ensure("w1", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, u1, [201])
    _edge(es, sid, w1[1], u1[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    surface_map[u1] = "u1"; surface_map[w1] = "w1"
    if with_chapter_seq:
        attach_chapter_seq(b, ref=u1, chapter_seq=1, section_seq=1)
    # ch2: U2..U50（49 unit 填 produced_refs cap=48·顶出 U1）+ U51（验选词）
    ch2_units = [u1]
    for i in range(2, 51):   # U2..U50
        u = ci.ensure(f"u{i}", space_id=sid, tier=TIER_PRIMARY)
        w = ci.ensure(f"w{i}", space_id=sid, tier=TIER_PRIMARY)
        attach_role_seq(b, u, [201])
        _edge(es, sid, w[1], u[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
        surface_map[u] = f"u{i}"; surface_map[w] = f"w{i}"
        if with_chapter_seq:
            attach_chapter_seq(b, ref=u, chapter_seq=2, section_seq=1)
        ch2_units.append(u)
    # U51（验选词）·两候选：W2b（低 lid·先建）+ W2a（高 lid·后建）
    u51 = ci.ensure("u51", space_id=sid, tier=TIER_PRIMARY)
    attach_role_seq(b, u51, [201])
    if with_chapter_seq:
        attach_chapter_seq(b, ref=u51, chapter_seq=2, section_seq=1)
    w2b = ci.ensure("w2b", space_id=sid, tier=TIER_PRIMARY)   # 低 lid（tiebreak 优先）
    w2a = ci.ensure("w2a", space_id=sid, tier=TIER_PRIMARY)   # 高 lid
    _edge(es, sid, w2b[1], u51[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    _edge(es, sid, w2a[1], u51[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    surface_map[u51] = "u51"; surface_map[w2b] = "w2b"; surface_map[w2a] = "w2a"
    # W2a COOCCURS U1（跨章 ctx·#729 章边界 carry 五环关键边）
    _edge(es, sid, w2a[1], u1[1], EDGE_COOCCURS, strength=1, subtype=None)
    ch2_units.append(u51)
    return ch2_units, u51, w2a, w2b, u1, surface_map


def test_e2e_five_ring_with_chapter_seq(core):
    """五环闭环（with chapter_seq）：章边界 carry U1 进 prior_topic_refs·U1 被 produced_refs
    顶出·经 prior_topic_refs 进 ctx_refs·collide_score(W2a)=1 > collide_score(W2b)=0 → 选 W2a。"""
    b, sid, es, ci = core
    units, u51, w2a, w2b, u1, surface_map = _build_two_chapter_fixture(
        b, sid, es, ci, with_chapter_seq=True)
    g = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
    dag = _dag(sid, sink=u51, struct_unit_refs=units, topo_layers=[units])
    wm = WorkMemory()
    out = generate_output(dag, g, wm, LANG_NONE)   # LANG_NONE 无 lang 偏好（纯 collide_score）
    # 五环验证：
    # ① 读章边界：U1 ch1 → U2 ch2（read_chapter_seq 变化点）
    # ② 触发 carry：章边界 carry_to_workmem 被调
    # ③ 写 prior_topic_refs：U1（ch1 唯一 unit）进 prior_topic_refs
    assert u1 in wm.prior_topic_refs
    # ④ produced_refs cap=48 顶出 U1（51 unit·末 48 = U4..U51·U1 不在）
    assert u1 not in wm.produced_refs
    assert len(wm.produced_refs) == PRODUCED_REFS_WINDOW
    # ⑤ slot_dispatch collide_score 消化 → 选词受影响：U51 槽选 W2a（collide_score=1 via U1）
    last_part = out.parts[-1]
    assert last_part.unit == u51
    assert last_part.words[0] == "w2a"   # W2a 胜（collide_score 1 > W2b 0）


def test_e2e_five_ring_contrast_no_chapter_seq(core):
    """对比（无 chapter_seq）：无章边界 carry·prior_topic_refs 空·U1 不进 ctx·
    collide_score(W2a)=0 = collide_score(W2b)=0 → tiebreak 选 W2b（低 lid）。"""
    b, sid, es, ci = core
    units, u51, w2a, w2b, u1, surface_map = _build_two_chapter_fixture(
        b, sid, es, ci, with_chapter_seq=False)
    g = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
    dag = _dag(sid, sink=u51, struct_unit_refs=units, topo_layers=[units])
    wm = WorkMemory()
    out = generate_output(dag, g, wm, LANG_NONE)
    # 无章边界 → 无 carry → prior_topic_refs 空
    assert wm.prior_topic_refs == []
    # U1 被 produced_refs cap=48 顶出
    assert u1 not in wm.produced_refs
    # collide_score(W2a)=0（U1 不在 ctx）= collide_score(W2b)=0 → tiebreak 选 W2b（低 lid）
    last_part = out.parts[-1]
    assert last_part.unit == u51
    assert last_part.words[0] == "w2b"   # W2b 胜（tiebreak 低 lid·无 prior_topic_refs boost）


def test_e2e_five_ring_bit_identical(core):
    """bit-identical：五环 e2e 两独立跑 → 同 prior_topic_refs + 同 produced_refs + 同选词。"""
    b, sid, es, ci = core
    units, u51, w2a, w2b, u1, surface_map = _build_two_chapter_fixture(
        b, sid, es, ci, with_chapter_seq=True)
    g = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
    dag = _dag(sid, sink=u51, struct_unit_refs=units, topo_layers=[units])
    wm1 = WorkMemory()
    out1 = generate_output(dag, g, wm1, LANG_NONE)
    wm2 = WorkMemory()
    out2 = generate_output(dag, g, wm2, LANG_NONE)
    assert wm1.prior_topic_refs == wm2.prior_topic_refs
    assert wm1.produced_refs == wm2.produced_refs
    assert [p.words for p in out1.parts] == [p.words for p in out2.parts]
