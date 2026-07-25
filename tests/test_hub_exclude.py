"""归一化半 A 功能词/hub 排除测试（doc/重来_归一化与功能词排除_设计_2026-07-08.md）。

hub_degree = COOCCURS 关联边总数（from+to 双向·绝对计数·read-time）·≥ THETA_HUB_DEGREE → is_hub。
3 live 消费/污染点 read-time 过滤（gate EXCLUDE_FUNCTION_MODE 守·default OFF·生产 try/finally 翻 ON）：
  collide_score（slot_dispatch caller·candidates+ctx_refs·解"分子是曾经"排序污染）
  _cooccurs_count（emergent_relation_signal·a/b hub→0·解伪产 REL_CAUSES 喂 reward）
  refers_occurrence（代词候选·解"他"→"曾经"语义层污染）

覆盖：
  - hub_degree 单元（from+to 双向计数 / θ 边界 / 冷启动 0 / 表未注册无 crash）
  - gate default OFF（守 CI bit-identical）
  - slot_dispatch collide_score 候选过滤改选 + ctx_refs 过滤 + 全 hub fallback 无 crash
  - _cooccurs_count hub→0（gate ON）/ 计数（gate OFF）
  - refers_occurrence hub 先行词过滤（gate ON 改选非 hub）
  - formal_train 生产 try/finally 翻 ON + finally 复位（反 theater）
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SUBTYPE_PURE_ALIAS
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.memory_space import MemorySpace
from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO, EDGE_COOCCURS, EDGE_PRECEDES
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.types import RoleSlot, LANG_NONE, LANG_ZH
from pure_integer_ai.cognition.shared.hub_detect import (
    HubDegreeState,
    hub_degree,
    is_hub,
    THETA_HUB_DEGREE,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.slot_dispatch import dispatch_slot
from pure_integer_ai.cognition.understanding.emergent_relation_signal import generate_emergent_hypotheses
from pure_integer_ai.cognition.understanding.refers_occurrence import resolve_pronoun_occurrence
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.config import gates


# ---- fixtures / helpers ----

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


def _edge(es, sid, frm, to, et, *, strength=1, subtype=None):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=4, tier=TIER_PRIMARY, subtype=subtype)


def _make_hub(es, sid, hub_lid, n=THETA_HUB_DEGREE):
    """建 n 条 COOCCURS 出边（hub→filler·filler lid 9000+ 避撞测试概念）使 hub_degree=≥n。"""
    for i in range(n):
        _edge(es, sid, hub_lid, 9000 + i, EDGE_COOCCURS)


def _graph(b, *, surface_map=None):
    return ConceptGraph(b,
                        surface_of=(lambda r: surface_map.get(r)) if surface_map else None)


def _dag(sid, *, sink=None):
    from pure_integer_ai.cognition.shared.types import PathResult, PathData, TERMINAL_REACHED_SINK
    return PathResult(path=PathData(edges=[], struct_unit_refs=[]),
                      terminal=TERMINAL_REACHED_SINK, sink=sink, topo_layers=[],
                      convergence={}, source=None)


# ============ hub_degree 单元 ============

def test_hub_degree_counts_from_and_to_bidirectional(core):
    """from-rows + to-rows 双向计数（不相交边集·无 double-count）。"""
    b, sid, es, ci = core
    # 4 条 from hub + 4 条 to hub = 8
    for i in range(4):
        _edge(es, sid, 100, 9000 + i, EDGE_COOCCURS)   # hub 作 from
        _edge(es, sid, 9000 + i, 100, EDGE_COOCCURS)   # hub 作 to（异边·lid 同但方向异）
    # from-rows: hub(100)→{9000..9003} = 4 · to-rows: {9000..9003}→hub(100) = 4 · 总 8
    assert hub_degree((sid, 100), es) == 8


def test_is_hub_threshold_boundary(core):
    """hub_degree ≥ THETA_HUB_DEGREE → True·< θ → False。"""
    b, sid, es, ci = core
    _make_hub(es, sid, 200, n=THETA_HUB_DEGREE)   # 恰 θ 条 → = θ → True（≥）
    assert hub_degree((sid, 200), es) == THETA_HUB_DEGREE
    assert is_hub((sid, 200), es) is True
    _make_hub(es, sid, 201, n=THETA_HUB_DEGREE - 1)   # θ-1 → False
    assert is_hub((sid, 201), es) is False


def test_is_hub_cold_start_no_cooccurs_returns_false(core):
    """无 COOCCURS 边 → hub_degree=0 < θ → False（冷启动退化·bit-identical OFF·无 crash）。"""
    b, sid, es, ci = core
    assert hub_degree((sid, 300), es) == 0
    assert is_hub((sid, 300), es) is False


def test_is_hub_missing_edge_table_no_crash():
    """edge 表未注册（bare backend 无 bootstrap）→ KeyError 容错→0→False（无 crash·审1 Q7）。"""
    b = DictBackend()   # 无 bootstrap → edge 表未注册
    es = EdgeStore(b)
    assert hub_degree((1, 1), es) == 0
    assert is_hub((1, 1), es) is False
    b.close()


def test_compute_hub_set_equals_is_hub_boundary(core):
    """compute_hub_set degree == is_hub 对所有 ref（审 P2-1·boundary pin·混合 from/to 恰 θ）。

    e2e 测用 n=9 > θ=8 不抓边界 divergence（若 compute_hub_set 漏 to 计数只算 from·n=9 仍 ≥8 过测）。
    本测 HUB 恰 θ=8（4 from + 4 to 混合）·NEAR θ-1=7·断言 compute_hub_set 返集 == {ref: is_hub}。
    """
    from pure_integer_ai.cognition.shared.hub_detect import compute_hub_set
    b, sid, es, ci = core
    # HUB=500：4 from（500→filler）+ 4 to（filler→500）= 8 = θ（混合·恰边界）
    for i in range(4):
        _edge(es, sid, 500, 9000 + i, EDGE_COOCCURS)
        _edge(es, sid, 9000 + i, 500, EDGE_COOCCURS)
    # NEAR=501：3 from + 4 to = 7 = θ-1 → 非 hub
    for i in range(3):
        _edge(es, sid, 501, 9100 + i, EDGE_COOCCURS)
    for i in range(4):
        _edge(es, sid, 9100 + i, 501, EDGE_COOCCURS)
    hub_set = compute_hub_set(es)
    assert (sid, 500) in hub_set, "4 from + 4 to = θ=8 → hub（混合 from/to·boundary）"
    assert (sid, 501) not in hub_set, "3 from + 4 to = 7 < θ → 非 hub"
    # 承重等价：compute_hub_set 返集 ⊇ {ref : is_hub(ref, es)}（所有相关 ref）
    refs = [(sid, 500), (sid, 501)] + [(sid, 9000 + i) for i in range(4)] + [(sid, 9100 + i) for i in range(4)]
    expected = {r for r in refs if is_hub(r, es)}
    assert hub_set >= expected, "compute_hub_set 须含所有 is_hub ref（degree 等价）"


def test_hub_degree_state_applies_writer_deltas_without_rescan(core, monkeypatch):
    """上下文状态首次扫图一次，后续 writer 增量直接更新 hub 集。"""
    b, sid, es, ci = core
    original_query_type = es.query_type
    scans = 0

    def counted_query_type(edge_type):
        """统计完整 COOCCURS 读取次数并保持原始返回值。"""
        nonlocal scans
        scans += 1
        return original_query_type(edge_type)

    monkeypatch.setattr(es, "query_type", counted_query_type)
    state = HubDegreeState(es)
    assert state.hub_set() == set()
    for index in range(THETA_HUB_DEGREE):
        target = (sid, 9300 + index)
        _edge(es, sid, 600, target[1], EDGE_COOCCURS)
        state.observe_cooccurs((sid, 600), target, 1)
    assert (sid, 600) in state.hub_set()
    assert scans == 1
    state.invalidate()
    assert (sid, 600) in state.hub_set()
    assert scans == 2


def test_concept_graph_hub_state_requires_explicit_round_invalidation(core):
    """ConceptGraph 的独立 EdgeStore 不再依赖失效不了的模块级版本缓存。"""
    b, sid, es, ci = core
    graph = ConceptGraph(b)
    assert (sid, 700) not in graph.hub_set()
    _make_hub(es, sid, 700)
    assert (sid, 700) not in graph.hub_set()
    graph.invalidate_ancestor_map()
    assert (sid, 700) in graph.hub_set()


# ============ gate default ============

def test_exclude_function_gate_default_off(monkeypatch):
    """EXCLUDE_FUNCTION_MODE default OFF（守 CI 回归·OFF = 3 点不过滤 bit-identical 现状）。

    直接断言模块属性（非 _flag helper）·若有人误改 default True 本测即抓（审 P2-1 加固）。
    """
    monkeypatch.delenv("PURE_INTEGER_AI_EXCLUDE_FUNCTION_MODE", raising=False)
    import pure_integer_ai.config.gates as g
    assert g.EXCLUDE_FUNCTION_MODE is False, \
        "EXCLUDE_FUNCTION_MODE 须 default OFF（守 CI===生产 bit-identical）"


# ============ slot_dispatch collide_score 候选过滤（caller 侧排除·决断 A3）============

def test_collide_score_candidate_filter_changes_selection(core):
    """gate ON → hub 候选排除 → 选非 hub；gate OFF → 共现高选 hub（解"分子是曾经"排序污染）。"""
    b, sid, es, ci = core
    C = ci.ensure("concept", space_id=sid, tier=TIER_PRIMARY)
    HUB = ci.ensure("hub", space_id=sid, tier=TIER_PRIMARY)        # lid = C+1 ish
    GOOD = ci.ensure("good", space_id=sid, tier=TIER_PRIMARY)
    CTX = ci.ensure("ctx", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, HUB[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    _edge(es, sid, GOOD[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    _edge(es, sid, HUB[1], CTX[1], EDGE_COOCCURS)   # HUB 与 ctx 共现（collide 高）
    _make_hub(es, sid, HUB[1])                      # HUB 成 hub（≥θ 边）
    g = _graph(b, surface_map={C: "c", HUB: "hub", GOOD: "good"})
    wm = WorkMemory()
    wm.prior_topic_refs = [CTX]

    saved = gates.EXCLUDE_FUNCTION_MODE
    try:
        # gate OFF：HUB 共现高 → 选 HUB
        gates.EXCLUDE_FUNCTION_MODE = False
        word_off, _ = dispatch_slot(RoleSlot(ref=C), _dag(sid), g, wm, LANG_NONE)
        assert word_off == "hub", f"gate OFF → 共现高选 hub·got {word_off}"
        # gate ON：HUB 过滤 → 选 GOOD
        gates.EXCLUDE_FUNCTION_MODE = True
        word_on, _ = dispatch_slot(RoleSlot(ref=C), _dag(sid), g, wm, LANG_NONE)
        assert word_on == "good", f"gate ON → hub 排除选 good·got {word_on}"
    finally:
        gates.EXCLUDE_FUNCTION_MODE = saved


def test_collide_score_ctx_refs_filtered_via_spy(core):
    """gate ON → ctx_refs 中 hub 被过滤（spy 证传入 collide_score 的 ctx_refs 不含 hub）。"""
    b, sid, es, ci = core
    C = ci.ensure("concept", space_id=sid, tier=TIER_PRIMARY)
    W = ci.ensure("w", space_id=sid, tier=TIER_PRIMARY)
    HUB = ci.ensure("hub", space_id=sid, tier=TIER_PRIMARY)
    GOOD = ci.ensure("good", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, W[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    _make_hub(es, sid, HUB[1])                      # HUB 成 hub
    g = _graph(b, surface_map={C: "c", W: "w"})
    wm = WorkMemory()
    wm.prior_topic_refs = [HUB, GOOD]               # ctx 含 hub + 非 hub
    captured: list[list] = []
    real = g.collide_score
    def spy(c, ctx_refs):
        captured.append(list(ctx_refs))
        return real(c, ctx_refs)
    g.collide_score = spy

    saved = gates.EXCLUDE_FUNCTION_MODE
    try:
        gates.EXCLUDE_FUNCTION_MODE = True
        dispatch_slot(RoleSlot(ref=C), _dag(sid), g, wm, LANG_NONE)
    finally:
        gates.EXCLUDE_FUNCTION_MODE = saved
    assert captured, "collide_score 须被调"
    # gate ON → ctx_refs 不含 HUB（hub 过滤）·含 GOOD
    flat = [r for ctx in captured for r in ctx]
    assert HUB not in flat, "gate ON → ctx_refs 须过滤 hub"
    assert GOOD in flat, "gate ON → ctx_refs 保留非 hub"


def test_collide_score_all_hub_candidates_fallback_no_crash(core):
    """全候选皆 hub → 过滤后空→fallback 保原 candidates（避 _stable_tiebreak([]) crash·stable≠correct）。"""
    b, sid, es, ci = core
    C = ci.ensure("concept", space_id=sid, tier=TIER_PRIMARY)
    H1 = ci.ensure("h1", space_id=sid, tier=TIER_PRIMARY)
    H2 = ci.ensure("h2", space_id=sid, tier=TIER_PRIMARY)
    _edge(es, sid, H1[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    _edge(es, sid, H2[1], C[1], EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS)
    _make_hub(es, sid, H1[1])                       # 两候选皆 hub
    _make_hub(es, sid, H2[1])
    g = _graph(b, surface_map={C: "c", H1: "h1", H2: "h2"})
    wm = WorkMemory()
    saved = gates.EXCLUDE_FUNCTION_MODE
    try:
        gates.EXCLUDE_FUNCTION_MODE = True
        word, _ = dispatch_slot(RoleSlot(ref=C), _dag(sid), g, wm, LANG_NONE)
        assert word in ("h1", "h2"), f"全 hub fallback 选原候选之一·got {word}"
    finally:
        gates.EXCLUDE_FUNCTION_MODE = saved


# ============ generate_emergent_hypotheses hub 跳过（caller 预计算·解伪产 REL_CAUSES 喂 reward）============

def test_generate_emergent_skips_hub_pair(core):
    """gate ON → hub pair（a 或 b 是 hub）被预计算 hub_set 跳过 → connector 不产 REL_CAUSES 假设。

    caller 预计算落点（决断 A3·perf 精化：O(unique refs) vs 每对 fresh 4 query·2026-07-08 实测 3-7× 灾难修复）。
    gate OFF → x-y COOCCURS≥MIN → 引发 connector 涌 REL_CAUSES。gate ON → x hub·(x,y) 跳过→不涌。
    """
    from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives
    from pure_integer_ai.storage.composes_attr import register_composes_attr
    b, sid, es, ci = core
    register_composes_attr(b)   # ensure_relation_primitives 写 ATTR_RELATION_PRIMITIVE 需此表
    ensure_relation_primitives(ci, b, space_id=sid)
    x = ci.ensure("雨", space_id=sid, tier=TIER_PRIMARY)
    w = ci.ensure("引发", space_id=sid, tier=TIER_PRIMARY)
    y = ci.ensure("洪水", space_id=sid, tier=TIER_PRIMARY)
    # 3× PRECEDES x→w→y + COOCCURS x-y（COOCCURS(x,y)=3 ≥ MIN → gate OFF 时 w 涌）
    for _ in range(3):
        _edge(es, sid, x[1], w[1], EDGE_PRECEDES)
        _edge(es, sid, w[1], y[1], EDGE_PRECEDES)
        _edge(es, sid, x[1], y[1], EDGE_COOCCURS)
    # 让 x 成 hub（x 已有 3 COOCCURS(x→y)·补 6 x→filler → hub_degree(x)=9 ≥ θ=8）
    for i in range(6):
        _edge(es, sid, x[1], 8000 + i, EDGE_COOCCURS)
    saved = gates.EXCLUDE_FUNCTION_MODE
    try:
        gates.EXCLUDE_FUNCTION_MODE = False
        hyps_off = generate_emergent_hypotheses(b, es, ci, space_id=sid, excluded_word_refs=set())
        assert any(h[0] == w for h in hyps_off), "gate OFF → 引发 connector 涌 REL_CAUSES"
        gates.EXCLUDE_FUNCTION_MODE = True
        hyps_on = generate_emergent_hypotheses(b, es, ci, space_id=sid, excluded_word_refs=set())
        assert not any(h[0] == w for h in hyps_on), \
            "gate ON → x hub·(x,y) 对预计算跳过·引发 不达 EMERGENT_CONNECTOR_MIN·不涌"
    finally:
        gates.EXCLUDE_FUNCTION_MODE = saved


# ============ refers_occurrence hub 先行词过滤（解"他"→"曾经"语义层污染）============

def test_refers_occurrence_hub_antecedent_filtered(core):
    """gate ON → hub 先行词（曾经）排除 → 选非 hub（小明）；OFF → 高近因选 hub。

    cengjing（hub·seg1 近因高）vs xiaoming（非 hub·seg0 近因低）。gate OFF 选 cengjing（污染）·
    gate ON 过滤 cengjing → 选 xiaoming（解"他"→"曾经"）。
    """
    b, sid, es, ci = core
    reg = SpaceRegistry(b)
    mem = MemorySpace.create(reg, "mem")
    xiaoming = ci.ensure("小明", space_id=sid, tier=TIER_PRIMARY)
    cengjing = ci.ensure("曾经", space_id=sid, tier=TIER_PRIMARY)
    _make_hub(es, sid, cengjing[1])                 # 曾经 成 hub
    wm = WorkMemory()
    wm.push_segment(0, [xiaoming])                  # seg0 近因低
    wm.push_segment(1, [cengjing])                  # seg1 近因高

    saved = gates.EXCLUDE_FUNCTION_MODE
    try:
        gates.EXCLUDE_FUNCTION_MODE = False
        ant_off = resolve_pronoun_occurrence(
            es, ci, "他", work_memory=wm, memory_space_id=mem.space_id, timestamp_seq=1)
        assert ant_off == cengjing, f"gate OFF → 高近因选 hub 曾经·got {ant_off}"

        gates.EXCLUDE_FUNCTION_MODE = True
        ant_on = resolve_pronoun_occurrence(
            es, ci, "他", work_memory=wm, memory_space_id=mem.space_id, timestamp_seq=2)
        assert ant_on == xiaoming, f"gate ON → hub 过滤选小明·got {ant_on}"
    finally:
        gates.EXCLUDE_FUNCTION_MODE = saved


# ============ formal_train 生产 try/finally 翻 ON（反 theater）============

def test_formal_train_flips_exclude_function_gate(tmp_path, monkeypatch):
    """formal_train 生产入口 try/finally 翻 EXCLUDE_FUNCTION_MODE ON + finally 复位（反 theater）。

    若 gate 非 try/finally 翻 ON·3 点永不过滤 → hub 污染未修 = theater。
    本测在 run_round_full 入口采 gates.EXCLUDE_FUNCTION_MODE·断言生产路径运行期间曾为 True
    （flip 生效）+ finally 复位（防泄漏·守后续测 bit-identical）。
    """
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    monkeypatch.setattr(gates, "TRAINING_MODE", True)   # 生产路径（stage_active_gates 读）
    seen: list[bool] = []
    from pure_integer_ai.experiments.formal_train import DefaultRoundRunner
    orig = DefaultRoundRunner.run_round_full
    def wrap(self, ctx, item, stage, rid):
        seen.append(bool(gates.EXCLUDE_FUNCTION_MODE))
        return orig(self, ctx, item, stage, rid)
    monkeypatch.setattr(DefaultRoundRunner, "run_round_full", wrap)

    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    corpus = [CollectedItem(tokens=["狗", "追", "兔子"])]
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="hub_flip",
                            rounds_per_stage=1)
    formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())

    assert seen, "formal_train 须至少调一次 run_round_full（采 gate 状态）"
    assert any(seen), "生产 try/finally 须翻 EXCLUDE_FUNCTION_MODE ON（否则 theater）"
    assert gates.EXCLUDE_FUNCTION_MODE is False, "finally 须复位 EXCLUDE_FUNCTION_MODE（saved_exclude_func）"
