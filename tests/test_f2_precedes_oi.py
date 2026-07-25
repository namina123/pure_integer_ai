"""F2 PRECEDES_OI_MODE 测试（gate ON 路径覆盖·反 theater·doc/重来_F2_PRECEDES_oi遍历_设计_2026-07-09）。

覆盖（设计 §十四/§十六 要求的 e2e·对抗审 H1 反 theater 缺口）：
  - _build_topo_layers_oi：first_occ gap 检测 + normalized_max_in tiebreak·含环节点（factor C 解）
  - a2_layer_oi：sink 在 topo_layers（sink 可达层）
  - is_dead_end ②：PRECEDES_OI_MODE ON 跳过 ②（factor D·backward CAUSES 不杀 PRECEDES 链）
  - dag_path_step OI_MODE：sink 可达（REACHED_SINK）
  - acyclic fix：seed（含显式 query-local 附加候选·有 in-edge）跳过 advance·
    path.edges 无环（generate.py:109 _path_acyclic 通过·不 crash）
  - bit-identical：gate OFF 走 Kahn（a2_layer）·既有测零回归（test_stage4 守）

gate 控制：每测 try/finally 翻 PRECEDES_OI_MODE ON + 复位（镜像生产 try/finally·单测 OFF 守回归）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES, EDGE_CAUSES
from pure_integer_ai.cognition.shared.types import (
    IntentType, ConceptRef, TERMINAL_REACHED_SINK, TERMINAL_DEAD_END, INTENT_QUESTION,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.process.a2_stepper import (
    a2_layer, a2_layer_oi, _build_topo_layers_oi, BLOCKED,
)
from pure_integer_ai.cognition.process.dead_end import is_dead_end
from pure_integer_ai.cognition.process.dag_path import dag_path_step
from pure_integer_ai.cognition.result.generate import _path_acyclic
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def core():
    b = DictBackend()
    bootstrap(b)
    from pure_integer_ai.storage.spaces.registry import SpaceRegistry
    from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    yield b, sp.space_id, es
    b.close()


def _edge(b, es, sid, frm, to, et, *, strength=1, order_index=None, sn=0, tn=0):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
           order_index=order_index, sn=sn, tn=tn)


def _ref(sid, lid):
    return (sid, lid)


def _rows(b):
    return b.select("edge")


@pytest.fixture
def oi_mode():
    """翻 PRECEDES_OI_MODE ON·测后复位（镜像生产 try/finally）。"""
    saved = gates.PRECEDES_OI_MODE
    gates.PRECEDES_OI_MODE = True
    yield gates.PRECEDES_OI_MODE
    gates.PRECEDES_OI_MODE = saved


# ============ _build_topo_layers_oi（first_occ gap 检测 + tiebreak）============

def test_build_topo_oi_intra_chain_order(core):
    """intra token 链 first_occ 严格递增·层序 = 文本序。"""
    b, sid, es = core
    # struct_ref(1) -> token_0(2) -> token_1(3) -> token_2(4)·intra oi=0,1
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES, order_index=0)   # anchor struct_ref->token_0
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=0)   # intra token_0->token_1 (oi=order_base+0=0)
    _edge(b, es, sid, 3, 4, EDGE_PRECEDES, order_index=1)   # intra token_1->token_2 (oi=1)
    layers = _build_topo_layers_oi(_rows(b), {EDGE_PRECEDES, EDGE_CAUSES})
    all_nodes = [n for layer in layers for n in layer]
    # 全 4 节点入层（含 struct_ref + tokens）
    assert set(all_nodes) == {_ref(sid, 1), _ref(sid, 2), _ref(sid, 3), _ref(sid, 4)}
    # token_0(2) first_occ=0·token_1(3) first_occ=1·token_2(4) 无出边 first_occ=入边+1=2
    # struct_ref(1) first_occ=0（anchor 出边 oi=0）·同 first_occ=0 与 token_0(2)
    # -> struct_ref(1) 与 token_0(2) 同层·tiebreak struct_ref(norm=-1) 先
    assert all_nodes[0] == _ref(sid, 1) or layers[0][0] == _ref(sid, 1)   # struct_ref 最先


def test_build_topo_oi_cyclic_nodes_included(core):
    """PRECEDES 概念环（token 重复·factor C）：环节点入层不被丢（Kahn 会丢）。"""
    b, sid, es = core
    # 环：2->3->4->2（token 2 重复出现致环）
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 3, 4, EDGE_PRECEDES, order_index=1)
    _edge(b, es, sid, 4, 2, EDGE_PRECEDES, order_index=2)
    layers = _build_topo_layers_oi(_rows(b), {EDGE_PRECEDES, EDGE_CAUSES})
    all_nodes = set(n for layer in layers for n in layer)
    # 全 3 环节点入层（Kahn 会全丢·入 cycle_nodes）
    assert all_nodes == {_ref(sid, 2), _ref(sid, 3), _ref(sid, 4)}


def test_build_topo_oi_inter_seg_forward(core):
    """inter-seg 边：last_token(i) first_occ=段末 < struct_ref(i+1) first_occ=段首·前向（v3 gap 检测）。"""
    b, sid, es = core
    # seg0: token_0(2)->last_token(3)·intra oi=0·seg1: struct_ref(4)->token_0(5) anchor oi=1
    # inter-seg: last_token(3)->struct_ref(4)·oi=巨(1000000)
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=0)           # intra seg0 (order_base_0=0)
    _edge(b, es, sid, 3, 4, EDGE_PRECEDES, order_index=1000000)     # inter-seg (巨 oi·seg_order_base+i*1<<20)
    _edge(b, es, sid, 4, 5, EDGE_PRECEDES, order_index=2)           # anchor seg1 (order_base_1=n_0=2)
    _edge(b, es, sid, 5, 6, EDGE_PRECEDES, order_index=2)           # intra seg1 (order_base_1+0=2)
    layers = _build_topo_layers_oi(_rows(b), {EDGE_PRECEDES, EDGE_CAUSES})
    order = [n for layer in layers for n in layer]
    # last_token(3) 出边 inter-seg 巨 > 入边(0)+1 -> first_occ=0+1=1（gap 检测·段末位置）
    # struct_ref(4) 出边 anchor oi=2·first_occ=2（段首·order_base_1）·> last_token(3)=1 -> 前向（v3 解 inter-seg）
    pos_3 = order.index(_ref(sid, 3))
    pos_4 = order.index(_ref(sid, 4))
    assert pos_3 < pos_4   # last_token(3) 先于 struct_ref(4)·inter-seg 链前向（active 传播序）


# ============ is_dead_end ② gate ON 跳过 ============

def test_is_dead_end_causes_skipped_oi_mode(core, oi_mode):
    """OI_MODE ON：节点有 CAUSES 前驱全不 active·② 不触发（factor D·PRECEDES 链不被杀）。"""
    b, sid, es = core
    # 3->2 (CAUSES unobserved)·4->2 (CAUSES unobserved)·节点 2 非 sink
    _edge(b, es, sid, 3, 2, EDGE_CAUSES)
    _edge(b, es, sid, 4, 2, EDGE_CAUSES)
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 99))
    # gate OFF 此处会 return True（test_dead_end_causes_preds_all_inactive）·gate ON 跳过 ②
    assert is_dead_end(_ref(sid, 2), _rows(b), intent, set(), 0, 100) is False


def test_is_dead_end_causes_active_gate_off(core):
    """gate OFF（默认）：② 仍触发（bit-identical·既有 test_dead_end_causes_preds_all_inactive 行为）。"""
    b, sid, es = core
    assert gates.PRECEDES_OI_MODE is False   # 默认 OFF
    _edge(b, es, sid, 3, 2, EDGE_CAUSES)
    _edge(b, es, sid, 4, 2, EDGE_CAUSES)
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 99))
    assert is_dead_end(_ref(sid, 2), _rows(b), intent, set(), 0, 100) is True


# ============ dag_path_step OI_MODE sink 可达 + acyclic fix ============

def test_dag_path_oi_reached_sink(core, oi_mode):
    """OI_MODE ON + PRECEDES 链 forward：sink 可达（REACHED_SINK·factor C 解）。"""
    b, sid, es = core
    # seed struct_ref(1) -> token(2) -> sink(3)·intra oi=0,1
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES, order_index=0)   # anchor
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=0)   # intra token->? (oi=0)
    _edge(b, es, sid, 2, 4, EDGE_PRECEDES, order_index=1)   # intra token->sink-ish
    # 简单：seed=1·sink=4·链 1->2->4
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 4))
    res = dag_path_step(_rows(b), [_ref(sid, 1)], wm, intent)
    assert res.terminal == TERMINAL_REACHED_SINK
    assert res.sink == _ref(sid, 4)


def test_dag_path_oi_acyclic_replay_seed(core, oi_mode):
    """★对抗审 REFUTED fix：seed（含 replay candidate·有 in-edge）+ PRECEDES 环场景·path.edges 无环。

    场景：S(seed·有 in-edge P->S) -> X -> P -> S 环。S 经 replay 进 active·访 S 时若 advance 收 P->S
    则 path.edges 含环 S->X->P->S（generate.py:109 _path_acyclic crash）。fix：seed 跳过 advance·不收 P->S。
    """
    b, sid, es = core
    S, X, P, sink = 1, 2, 3, 4
    # X->P (intra·oi=0)·P->S (inter-seg·oi=巨)·S->X (anchor·oi=2)·S->sink (oi=3)
    _edge(b, es, sid, X, P, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, P, S, EDGE_PRECEDES, order_index=1000000)   # inter-seg 巨
    _edge(b, es, sid, S, X, EDGE_PRECEDES, order_index=2)         # anchor
    _edge(b, es, sid, S, sink, EDGE_PRECEDES, order_index=3)
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, sink))
    # seeds=[S]·S 有 in-edge(P->S)，模拟 query-local resolver 显式增加候选。
    res = dag_path_step(_rows(b), [_ref(sid, S)], wm, intent)
    # sink 可达
    assert res.terminal == TERMINAL_REACHED_SINK
    assert res.sink == _ref(sid, sink)
    # ★ path.edges 无环（generate.py:109 _path_acyclic 通过·不 crash）·acyclic fix 生效
    assert _path_acyclic(res) is True
    # P->S 边不入 path.edges（S 是 seed·跳过 advance·不收其 pred 边）
    ps_edge = (_ref(sid, P)[0], _ref(sid, P)[1], _ref(sid, S)[0], _ref(sid, S)[1], EDGE_PRECEDES)
    assert ps_edge not in res.path.edges
    # S->X 出边入 path.edges（X 访时收·seed 的 out-edge 由后继访时收）
    sx_edge = (_ref(sid, S)[0], _ref(sid, S)[1], _ref(sid, X)[0], _ref(sid, X)[1], EDGE_PRECEDES)
    assert sx_edge in res.path.edges


def test_dag_path_oi_cyclic_graph_acyclic_path(core, oi_mode):
    """OI_MODE ON + PRECEDES 概念环：path.edges 无环（node-centric first-occ 每节点访一次·§四 acyclic）。"""
    b, sid, es = core
    # 环 2->3->4->2 + seed 1->2 + sink=4
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES, order_index=0)   # seed->环
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=1)
    _edge(b, es, sid, 3, 4, EDGE_PRECEDES, order_index=2)
    _edge(b, es, sid, 4, 2, EDGE_PRECEDES, order_index=3)   # 回边成环
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 4))
    res = dag_path_step(_rows(b), [_ref(sid, 1)], wm, intent)
    # path.edges 无环（_path_acyclic 通过）
    assert _path_acyclic(res) is True


# ============ bit-identical（gate OFF 走 Kahn·既有行为不变）============

def test_dag_path_gate_off_uses_kahn(core):
    """gate OFF（默认）：dag_path 走 a2_layer（Kahn）·既有行为（bit-identical·反 theater gate 守）。"""
    b, sid, es = core
    assert gates.PRECEDES_OI_MODE is False
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)
    _edge(b, es, sid, 2, 3, EDGE_CAUSES, sn=1, tn=0)
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 3))
    res = dag_path_step(_rows(b), [_ref(sid, 1)], wm, intent)
    assert res.terminal == TERMINAL_REACHED_SINK   # 既有行为（test_stage4 同款）
