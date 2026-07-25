"""旧 replay seed 迁移探针与 M-00 跨 episode 隔离测试。

反 theater e2e（照 S4 片3 模板 test_stage_s4_selection_pref_dock.py:766-844）：
- B 半真活：dag_path local_seeds 扩张 replay_candidates → path 变（非 theater）
- B 半过滤：replay 过滤当前 subgraph 外节点（避孤立 seed 污染 PR）
- 显式 query-local replay_candidates 仍可由路径层消费，供后续 Memory resolver 接线
- tri_space 不再依据上一 episode reward 写 replay/exclude
- episode_loop 即使获得语言正 reward 也不写 memory_item 或下一轮 seed

诚实边界：replay 作种子是弱有效（topo + PR 结构性变·非强偏好·stable≠correct·#479 墙）。
exclude_refs defer（sink 保护 + intent.sink 固定 + 粒度不匹配 三重阻断·无有效读法）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.memory_space import MemorySpace
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES, EDGE_CAUSES
from pure_integer_ai.cognition.shared.types import (
    InputPayload, IntentType, Episode, GMeta,
    TERMINAL_REACHED_SINK, TERMINAL_DEAD_END, REWARD_DEAD_END,
    INTENT_QUESTION, LANG_ZH, DOMAIN_TEXT,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.understanding.role_precedes import attach_role_seq
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.generate import generate_output
from pure_integer_ai.cognition.result.tri_space import tri_space_coordination
from pure_integer_ai.cognition.process.episode import episode_loop
from pure_integer_ai.cognition.process.dag_path import dag_path_step
from pure_integer_ai.crosscut.integer.rational import make


# ---- helpers ----

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


def _edge(es, sid, frm, to, et, *, strength=1, source=SOURCE_BARE_TEXT):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=source, tier=TIER_PRIMARY)


def _graph(b, *, surface_map=None):
    return ConceptGraph(b,
                       surface_of=(lambda r: surface_map.get(r)) if surface_map else None)


def _path_nodes(pr) -> set:
    """PathResult.path.edges 节点集（from/to tuple 前 4 元素）。"""
    nodes = set()
    for e in pr.path.edges:
        nodes.add((e[0], e[1]))   # from
        nodes.add((e[2], e[3]))   # to
    return nodes


def _episode(*, reward=1, terminal=TERMINAL_REACHED_SINK):
    return Episode(
        episode_id=0, run_id=1, reward=reward, terminal=terminal,
        pr_vector={1: make(1, 2)},
        judge_G4_active=False, judge_G2p_active=False, judge_G3a_active=False,
        judge_G3b_active=False, judge_G5_active=False,
        judge_veto_count=0, dead_end_count=0, vetoed=False,
    )


# ============ B 半：dag_path local_seeds 扩张（replay 真活·非 theater） ============

def test_dag_path_local_seeds_replay_expansion(core):
    """#728 B 半反 theater：dag_path local_seeds 扩张 replay_candidates·path 变（非 theater）。

    subgraph：seed→A→sink1（主支）+ sink2→B（独立支·sink2 非种子非 sink）。
    replay=[] → e_set={seed} → Kahn 分层从 seed 起 → sink2→B 支不入分层 → path 不含 sink2/B。
    replay=[sink2]（sink2 in subgraph）→ e_set={seed, sink2} → Kahn 从 seed+sink2 起 →
      sink2→B 支入分层 → path 含 sink2/B。
    **反 theater 牙**：replay 扩张 path 变（sink2 支加入）= local_seeds 扩张真活在决策路径·非 theater。
    """
    b, sid, es, ci = core
    seed = ci.ensure("seed", space_id=sid, tier=TIER_PRIMARY)
    A = ci.ensure("A", space_id=sid, tier=TIER_PRIMARY)
    sink1 = ci.ensure("sink1", space_id=sid, tier=TIER_PRIMARY)
    sink2 = ci.ensure("sink2", space_id=sid, tier=TIER_PRIMARY)
    B = ci.ensure("B", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, seed[1], A[1], EDGE_PRECEDES)
    _edge(es, sid, A[1], sink1[1], EDGE_CAUSES)
    _edge(es, sid, sink2[1], B[1], EDGE_PRECEDES)
    dag_edges = b.select("edge")
    intent = IntentType(type=INTENT_QUESTION, sink=sink1)

    # replay=[] → local_seeds=[seed] → path 不含 sink2 支
    wm1 = WorkMemory()
    pr1 = dag_path_step(dag_edges, [seed], wm1, intent, backend=b)
    nodes1 = _path_nodes(pr1)

    # replay=[sink2]（sink2 in subgraph）→ local_seeds=[seed, sink2] → path 含 sink2 支
    wm2 = WorkMemory()
    wm2.replay_candidates = [sink2]
    pr2 = dag_path_step(dag_edges, [seed], wm2, intent, backend=b)
    nodes2 = _path_nodes(pr2)

    # 反 theater 牙：replay 扩张 path 变（sink2 支加入）
    assert sink2 not in nodes1, "replay=[] path 不该含 sink2 支"
    assert sink2 in nodes2, "replay=[sink2] path 该含 sink2（local_seeds 扩张真活）"
    assert B in nodes2, "replay=[sink2] path 该含 B（sink2→B 边加入）"
    assert nodes1 != nodes2, "replay 扩张 path 变 = B 半真活非 theater"


def test_dag_path_replay_filter_outside_subgraph(core):
    """#728 B 半：replay 过滤当前 subgraph 外节点（避孤立 seed 污染 PR·Agent B 判决）。

    replay=[notin_subgraph_ref]（不在 subgraph 节点集）→ 过滤 → local_seeds == seeds →
      path 与 replay=[] bit-identical。
    """
    b, sid, es, ci = core
    seed = ci.ensure("seed", space_id=sid, tier=TIER_PRIMARY)
    A = ci.ensure("A", space_id=sid, tier=TIER_PRIMARY)
    sink1 = ci.ensure("sink1", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, seed[1], A[1], EDGE_PRECEDES)
    _edge(es, sid, A[1], sink1[1], EDGE_CAUSES)
    dag_edges = b.select("edge")
    intent = IntentType(type=INTENT_QUESTION, sink=sink1)

    wm1 = WorkMemory()
    pr1 = dag_path_step(dag_edges, [seed], wm1, intent, backend=b)

    wm2 = WorkMemory()
    wm2.replay_candidates = [(sid, 9999)]   # 不在 subgraph
    pr2 = dag_path_step(dag_edges, [seed], wm2, intent, backend=b)

    assert _path_nodes(pr1) == _path_nodes(pr2), \
        "replay 过滤 subgraph 外节点 → local_seeds == seeds → bit-identical"


def test_dag_path_replay_dedup_existing_seed(core):
    """#728 B 半：replay 含已有 seed → 不重复加（_seed_set 去重）。

    replay=[seed]（seed 已 in seeds）→ 不重复加 → local_seeds == [seed] → bit-identical。
    """
    b, sid, es, ci = core
    seed = ci.ensure("seed", space_id=sid, tier=TIER_PRIMARY)
    A = ci.ensure("A", space_id=sid, tier=TIER_PRIMARY)
    sink1 = ci.ensure("sink1", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, seed[1], A[1], EDGE_PRECEDES)
    _edge(es, sid, A[1], sink1[1], EDGE_CAUSES)
    dag_edges = b.select("edge")
    intent = IntentType(type=INTENT_QUESTION, sink=sink1)

    wm1 = WorkMemory()
    pr1 = dag_path_step(dag_edges, [seed], wm1, intent, backend=b)

    wm2 = WorkMemory()
    wm2.replay_candidates = [seed]   # 已 in seeds
    pr2 = dag_path_step(dag_edges, [seed], wm2, intent, backend=b)

    assert _path_nodes(pr1) == _path_nodes(pr2), \
        "replay 含已有 seed 去重 → local_seeds == seeds → bit-identical"


# ============ M-00：旧 reward replay 关闭 ============

def test_tri_space_keeps_reward_replay_empty(core):
    """M-00：协调入口不再根据 reward 或 memory_item 产生 replay。"""
    b, sid, es, ci = core
    reg = SpaceRegistry(b)
    ms = MemorySpace.create(reg, "mem_read")
    ms.put(1, content_hash=100, info_ref_space=sid, info_ref_id=100)
    ms.record_use(1, success=True)
    ep = _episode(reward=1)
    wm = WorkMemory()
    tri_space_coordination(ep, workmem=wm, memory_space=ms)
    assert wm.replay_candidates == []
    assert wm.exclude_refs == set()


def test_tri_space_ignores_removed_replay_gate(core):
    """M-00：旧配置是否存在不再改变 replay 关闭语义。"""
    b, sid, es, ci = core
    reg = SpaceRegistry(b)
    ms = MemorySpace.create(reg, "mem_read")
    ms.put(1, content_hash=100, info_ref_space=sid, info_ref_id=100)
    ms.record_use(1, success=True)   # success_rate=1
    ep = _episode(reward=1)
    wm = WorkMemory()
    tri_space_coordination(ep, workmem=wm, memory_space=ms)
    assert wm.replay_candidates == []


def test_tri_space_fresh_clear_per_episode(core):
    """M-00：协调入口清除旧 replay/exclude，且不以本次 reward 重建。"""
    b, sid, es, ci = core
    reg = SpaceRegistry(b)
    ms = MemorySpace.create(reg, "mem_read")
    ms.put(1, content_hash=100, info_ref_space=sid, info_ref_id=100)
    ms.record_use(1, success=True)
    ep = _episode(reward=1)
    wm = WorkMemory()
    old_ref = (sid, 8888)
    wm.replay_candidates = [old_ref]   # 预填旧值
    wm.exclude_refs.add(old_ref)
    tri_space_coordination(ep, workmem=wm, memory_space=ms)
    assert wm.replay_candidates == []
    assert wm.exclude_refs == set()


def test_tri_space_negative_reward_does_not_write_exclusions(core):
    """M-00：上一 episode 的负 reward 不得控制下一次召回。"""
    b, sid, es, ci = core
    reg = SpaceRegistry(b)
    ms = MemorySpace.create(reg, "mem_read")
    ms.put(1, content_hash=100, info_ref_space=sid, info_ref_id=100)
    ms.record_use(1, success=False)   # success_rate=0 < 1/2 = 负经验
    ep = _episode(reward=REWARD_DEAD_END, terminal=TERMINAL_DEAD_END)
    wm = WorkMemory()
    tri_space_coordination(ep, workmem=wm, memory_space=ms)
    assert wm.exclude_refs == set()


def test_tri_space_cannot_expand_next_dag_path(core):
    """M-00：memory_item 与 reward 不能经 tri_space 扩张下一条 DAG 路径。"""
    b, sid, es, ci = core
    # subgraph：seed→A→sink1（主支）+ sink2→B（独立支·sink2 非种子非 sink）
    seed = ci.ensure("seed", space_id=sid, tier=TIER_PRIMARY)
    A = ci.ensure("A", space_id=sid, tier=TIER_PRIMARY)
    sink1 = ci.ensure("sink1", space_id=sid, tier=TIER_PRIMARY)
    sink2 = ci.ensure("sink2", space_id=sid, tier=TIER_PRIMARY)
    B = ci.ensure("B", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, seed[1], A[1], EDGE_PRECEDES)
    _edge(es, sid, A[1], sink1[1], EDGE_CAUSES)
    _edge(es, sid, sink2[1], B[1], EDGE_PRECEDES)
    dag_edges = b.select("edge")

    # tri_space 写：reward>0 + memory_item info_ref_space=sid（core·同 dag_path subgraph space）
    # 模拟生产 reward_propagate 落点② 写（info_ref_space=sink[0]=core sid）
    reg = SpaceRegistry(b)
    ms = MemorySpace.create(reg, "mem_read")
    ms.put(1, content_hash=100, info_ref_space=sid, info_ref_id=sink2[1])   # info_ref=sink2（core space）
    ms.record_use(1, success=True)
    wm = WorkMemory()
    tri_space_coordination(_episode(reward=1), workmem=wm, memory_space=ms)
    assert wm.replay_candidates == []

    # dag_path 读 workmem.replay → local_seeds 扩张（sink2 in subgraph_nodes·core space 一致→命中）
    intent = IntentType(type=INTENT_QUESTION, sink=sink1)
    pr = dag_path_step(dag_edges, [seed], wm, intent, backend=b)
    nodes = _path_nodes(pr)
    assert sink2 not in nodes
    assert B not in nodes


def test_episode_loop_language_reward_does_not_seed_replay(core):
    """M-00：legacy 语言 episode 的 scalar reward 不写 Memory 或 replay。"""
    b, sid, es, ci = core
    seed = ci.ensure("seed", space_id=sid, tier=TIER_PRIMARY)
    A = ci.ensure("A", space_id=sid, tier=TIER_PRIMARY)
    sink1 = ci.ensure("sink1", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, seed[1], A[1], EDGE_PRECEDES)
    _edge(es, sid, A[1], sink1[1], EDGE_CAUSES)
    attach_role_seq(b, sink1, [1])
    dag_edges = b.select("edge")
    reg = SpaceRegistry(b)
    ms = MemorySpace.create(reg, "mem_read")
    g = _graph(b, surface_map={seed: "seed", A: "A", sink1: "sink1"})

    intent = IntentType(type=INTENT_QUESTION, sink=sink1)
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       domain=DOMAIN_TEXT, intent=intent, key_skeleton=[seed])
    wm = WorkMemory()

    out, ep = episode_loop(
        inp, dag_edges, [seed], wm, intent,
        generate_fn=lambda pr, w, i: generate_output(pr, g, w, LANG_ZH),
        judge_fn=lambda o, pr, i, w: (1, GMeta()),
        edge_store=es, backend=b, memory_read=ms,
        current_seq=1)

    assert ep.reward > 0, "judge reward=1"
    assert wm.replay_candidates == []
    assert b.select("memory_item", where={"space_id": ms.space_id}) == []


def test_episode_loop_replay_is_empty_without_memory_use(core):
    """没有 query-local Memory Use 时，episode 结束后 replay 保持为空。"""
    b, sid, es, ci = core
    seed = ci.ensure("seed", space_id=sid, tier=TIER_PRIMARY)
    A = ci.ensure("A", space_id=sid, tier=TIER_PRIMARY)
    sink1 = ci.ensure("sink1", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, seed[1], A[1], EDGE_PRECEDES)
    _edge(es, sid, A[1], sink1[1], EDGE_CAUSES)
    attach_role_seq(b, sink1, [1])
    dag_edges = b.select("edge")
    reg = SpaceRegistry(b)
    ms = MemorySpace.create(reg, "mem_read")
    g = _graph(b, surface_map={seed: "seed", A: "A", sink1: "sink1"})

    intent = IntentType(type=INTENT_QUESTION, sink=sink1)
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       domain=DOMAIN_TEXT, intent=intent, key_skeleton=[seed])
    wm = WorkMemory()

    out, ep = episode_loop(
        inp, dag_edges, [seed], wm, intent,
        generate_fn=lambda pr, w, i: generate_output(pr, g, w, LANG_ZH),
        judge_fn=lambda o, pr, i, w: (1, GMeta()),
        edge_store=es, backend=b, memory_read=ms)

    assert wm.replay_candidates == []


# ============ gate OFF bit-identical（两 fresh fixture 同输入→输出一致） ============

def _run_episode_once():
    """单跑 episode_loop（fresh fixture·gate OFF 默认）·返 output.words。"""
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    sid = sp.space_id
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    ms = MemorySpace.create(reg, "mem_read")
    seed = ci.ensure("seed", space_id=sid, tier=TIER_PRIMARY)
    A = ci.ensure("A", space_id=sid, tier=TIER_PRIMARY)
    sink1 = ci.ensure("sink1", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, seed[1], A[1], EDGE_PRECEDES)
    _edge(es, sid, A[1], sink1[1], EDGE_CAUSES)
    attach_role_seq(b, sink1, [1])
    dag_edges = b.select("edge")
    g = _graph(b, surface_map={seed: "seed", A: "A", sink1: "sink1"})
    intent = IntentType(type=INTENT_QUESTION, sink=sink1)
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                       domain=DOMAIN_TEXT, intent=intent, key_skeleton=[seed])
    wm = WorkMemory()
    out, ep = episode_loop(
        inp, dag_edges, [seed], wm, intent,
        generate_fn=lambda pr, w, i: generate_output(pr, g, w, LANG_ZH),
        judge_fn=lambda o, pr, i, w: (1, GMeta()),
        edge_store=es, backend=b, memory_read=ms)
    return [p.words for p in out.parts]


def test_gate_off_bit_identical_two_runs():
    """#728 gate OFF bit-identical：两 fresh fixture 同输入 → 输出一致（replay 永空·local_seeds == seeds）。

    gate OFF（默认）→ tri_space early-return → workmem.replay 永空 → dag_path local_seeds == seeds →
    两跑 bit-identical（守 1168 测零回归基线）。
    """
    out1 = _run_episode_once()
    out2 = _run_episode_once()
    assert out1 == out2, \
        "gate OFF 两跑 bit-identical（replay 永空·local_seeds == seeds·无跨 run 污染）"
