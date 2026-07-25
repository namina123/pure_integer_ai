"""#729 M5 真分页 + 长文本连贯 transient — 实施1 produced_refs FIFO cap 测试。

produced_refs 加 FIFO cap=N（PRODUCED_REFS_WINDOW=48）。
覆盖：
  - add_produced 保序去重 + FIFO 截断保近期
  - cap > 单段 unit 数守去重语义（本段已 append 不被截断）
  - generate_output e2e：>48 units → produced_refs capped at 48
  - bit-identical 两跑

设计文档：doc/重来_任务0729_M5真分页_长文本连贯_设计.md 决断3。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT, SUBTYPE_PURE_ALIAS
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO
from pure_integer_ai.cognition.shared.types import (
    PathData, PathResult, ConceptRef, OutputPart, RoleSlot,
    LINEAGE_CONCEPT_FILL, TERMINAL_REACHED_SINK, LANG_ZH,
)
from pure_integer_ai.cognition.shared.work_memory import (
    WorkMemory, DEFAULT_PRONOUN_WINDOW, PRODUCED_REFS_WINDOW,
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
    yield b, sp.space_id, es, ci
    b.close()


def _ref(sid, lid):
    return (sid, lid)


def _graph(b, *, surface_map=None):
    return ConceptGraph(b,
                       surface_of=(lambda r: surface_map.get(r)) if surface_map else None)


def _dag(sid, *, sink=None, edges=None, struct_unit_refs=None,
         topo_layers=None, terminal=TERMINAL_REACHED_SINK):
    return PathResult(
        path=PathData(edges=edges or [], struct_unit_refs=struct_unit_refs or []),
        terminal=terminal, sink=sink,
        topo_layers=topo_layers or [], convergence={}, source=None,
    )


# ============ add_produced 单元测 ============

def test_add_produced_dedup():
    """同 ref 两次 add → produced_refs 单条（保序去重）。"""
    wm = WorkMemory()
    r = (0, 1)
    wm.add_produced(r, window=48)
    wm.add_produced(r, window=48)
    assert wm.produced_refs == [r]


def test_add_produced_fifo_cap_truncates_oldest():
    """window 满 del [:over] 截断保近期（末 window 个·最旧淘汰）。"""
    wm = WorkMemory()
    window = 5
    for i in range(window + 5):   # 加 10 个·cap=5 → 淘汰最旧 5 个
        wm.add_produced((0, i), window=window)
    assert len(wm.produced_refs) == window
    # 末 5 个保留（i=5..9）·最旧 5 个（i=0..4）淘汰
    assert wm.produced_refs == [(0, i) for i in range(5, 10)]


def test_add_produced_default_window_is_48():
    """add_produced 不传 window → 用默认 PRODUCED_REFS_WINDOW=48。"""
    assert PRODUCED_REFS_WINDOW == 48
    wm = WorkMemory()
    for i in range(60):
        wm.add_produced((0, i))   # 默认 window=48
    assert len(wm.produced_refs) == 48
    # 末 48 个保留（i=12..59）·最旧 12 个淘汰
    assert wm.produced_refs == [(0, i) for i in range(12, 60)]


def test_add_produced_dedup_within_window_not_dropped():
    """cap > 单段 unit 数守去重语义：ref A + 47 others + ref A again → A 仍在（去重·不重复 append·不占额外位）。"""
    wm = WorkMemory()
    window = 48
    a = (0, 100)
    wm.add_produced(a, window=window)
    for i in range(47):   # 47 others → 共 48 个·满 cap 但未超
        wm.add_produced((0, i), window=window)
    wm.add_produced(a, window=window)   # 去重·不 append
    assert len(wm.produced_refs) == window   # 仍 48（A 去重未增）
    assert a in wm.produced_refs              # A 仍在


def test_add_produced_dropped_after_window():
    """ref A + 48 others（超 cap）→ A 被 FIFO 淘汰（最旧）。"""
    wm = WorkMemory()
    window = 48
    a = (0, 100)
    wm.add_produced(a, window=window)
    for i in range(window):   # 48 others → 共 49 个·超 cap 1 个 → 淘汰最旧 A
        wm.add_produced((0, i), window=window)
    assert len(wm.produced_refs) == window
    assert a not in wm.produced_refs   # A 被淘汰


def test_add_produced_preserves_order():
    """保序：append 时序确定（无 sort·无 hash 顺序）。"""
    wm = WorkMemory()
    refs = [(0, 5), (0, 3), (0, 1), (0, 4), (0, 2)]
    for r in refs:
        wm.add_produced(r, window=48)
    assert wm.produced_refs == refs   # 按 append 序·非自然序


def test_add_produced_bit_identical_two_runs():
    """bit-identical：两独立 WorkMemory 同序列 add → 同 produced_refs。"""
    seq = [(0, i) for i in range(60)]
    wm1 = WorkMemory()
    wm2 = WorkMemory()
    for r in seq:
        wm1.add_produced(r)
        wm2.add_produced(r)
    assert wm1.produced_refs == wm2.produced_refs


# ============ generate_output e2e：produced_refs cap 真生效 ============

def test_generate_output_produced_refs_cap_e2e(core):
    """generate_output 产 >48 units → workmem.produced_refs capped at 48（末 48 个保留）。

    验 cap 在生产路径真生效（非 theater）：generate.py:132 add_produced(unit) 真调·
    produced_refs 真截断。slot_dispatch collide_score 读截断后子集。
    """
    b, sid, es, ci = core
    n_units = 55   # > 48 cap
    units = []
    surface_map = {}
    for i in range(n_units):
        u = ci.ensure(f"u{i}", space_id=sid, tier=TIER_PRIMARY)
        w = ci.ensure(f"w{i}", space_id=sid, tier=TIER_PRIMARY)
        attach_role_seq(b, u, [201])   # 单槽 role_seq=201
        es.add(space_id_from=sid, local_id_from=w[1], space_id_to=sid,
               local_id_to=u[1], edge_type=EDGE_REFERS_TO,
               strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
               subtype=SUBTYPE_PURE_ALIAS, sn=0, tn=0)
        units.append(u)
        surface_map[u] = f"u{i}"
        surface_map[w] = f"w{i}"
    g = _graph(b, surface_map=surface_map)
    wm = WorkMemory()
    dag = _dag(sid, sink=units[-1], struct_unit_refs=units,
               topo_layers=[units])   # 单层 55 units
    out = generate_output(dag, g, wm, LANG_ZH)
    assert len(out.parts) == n_units
    # produced_refs capped at 48（末 48 个·最旧 7 个淘汰）
    assert len(wm.produced_refs) == PRODUCED_REFS_WINDOW
    # 末 48 个保留（units[7:]）·最旧 7 个淘汰（units[:7]）
    assert wm.produced_refs == units[7:]
    # 淘汰的最旧 unit 不在 produced_refs
    assert units[0] not in wm.produced_refs
    assert units[6] not in wm.produced_refs
    # 保留的末 unit 在 produced_refs
    assert units[-1] in wm.produced_refs


def test_generate_output_produced_refs_bit_identical_two_runs(core):
    """bit-identical：generate_output 两独立跑同 dag → 同 produced_refs + 同 parts。

    验 cap 截断确定性（两跑同 cap 同子集·同输出）。
    """
    b, sid, es, ci = core
    n_units = 55
    units = []
    surface_map = {}
    for i in range(n_units):
        u = ci.ensure(f"u{i}", space_id=sid, tier=TIER_PRIMARY)
        w = ci.ensure(f"w{i}", space_id=sid, tier=TIER_PRIMARY)
        attach_role_seq(b, u, [201])
        es.add(space_id_from=sid, local_id_from=w[1], space_id_to=sid,
               local_id_to=u[1], edge_type=EDGE_REFERS_TO,
               strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
               subtype=SUBTYPE_PURE_ALIAS, sn=0, tn=0)
        units.append(u)
        surface_map[u] = f"u{i}"
        surface_map[w] = f"w{i}"
    g = _graph(b, surface_map=surface_map)
    dag = _dag(sid, sink=units[-1], struct_unit_refs=units,
               topo_layers=[units])
    # 两独立跑（独立 WorkMemory + ConceptGraph）
    wm1 = WorkMemory()
    out1 = generate_output(dag, g, wm1, LANG_ZH)
    wm2 = WorkMemory()
    out2 = generate_output(dag, g, wm2, LANG_ZH)
    assert wm1.produced_refs == wm2.produced_refs
    assert len(out1.parts) == len(out2.parts)
    assert [p.words for p in out1.parts] == [p.words for p in out2.parts]
