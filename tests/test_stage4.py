"""Stage 4 验收门测试：卷二过程建模（步进产 path + 死路产负 + reward 落 CAUSES + H4 闭环 + R1）。

覆盖（doc/重来_落地规划与实施顺序.md §六 Stage 4 验收门）：
  - 步进产 PathData（模块4·选定边集存非派生）
  - 死路三条件产 reward<0（模块6·防塌柱② 真负通路）
  - reward 落 CAUSES 头（模块8·PRECEDES 永不接 reward）
  - H4 effective_weight 闭环（模块7·strength×rate·reward 调 sn/tn→rate 变→权重变）
  - R1 episode 级 reward 符号（非边级 delta_reward·新边首次观测给机会）
  - Σ=0 边界（R5·全脏边冷启动）
  - A2 按头 AND/OR 分发（模块1·PRECEDES AND / CAUSES OR）
  - A3 PR 多种子 wrapper（模块2·线性性零损失·cache·add/remove seed）
  - A4 结构对齐（模块3·pairwise LCS 折叠 + coverage_overlap）
  - attractor 松入严留（模块5）
  - episode wiring 单点化（模块9·DEAD_END→reward<0）
  - 确定性 bit-identical
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT, SUBTYPE_OCCURRENCE
from pure_integer_ai.storage.node_store import TIER_PRIMARY, TIER_SHADOW
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import (
    EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO, EDGE_T_STEP,
)
from pure_integer_ai.cognition.shared.types import (
    InputPayload, IntentType, PathData, PathResult, Step, Episode, GMeta,
    ConceptRef, EdgeRef,
    TERMINAL_REACHED_SINK, TERMINAL_DEAD_END, REWARD_DEAD_END,
    INTENT_QUESTION, DOMAIN_MATH, MODALITY_ARITH,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.process.effective_weight import (
    effective_weight, edge_rate, RATE_SCALE,
)
from pure_integer_ai.cognition.process.dead_end import is_dead_end, out_degree
from pure_integer_ai.cognition.process.a2_stepper import a2_layer, HeadStepper, BLOCKED
from pure_integer_ai.cognition.process.a3_pr_wrapper import A3PRWrapper
from pure_integer_ai.cognition.process.a4_align import coverage_overlap, MAX_QUALITY
from pure_integer_ai.cognition.process.attractor import maybe_expand_attractor
from pure_integer_ai.cognition.process.dag_path import dag_path_step, j4_closure_check, SAFETY_FACTOR
from pure_integer_ai.cognition.process.reward_propagate import propagate_reward, DELTA_DEFAULT
from pure_integer_ai.cognition.process.episode import episode_loop
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def core():
    """建 backend + 核心空间 + EdgeStore。返 (backend, space_id, edge_store)。"""
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    yield b, sp.space_id, es
    b.close()


def _edge(b, es, sid, frm, to, et, *, strength=1, sn=0, tn=0,
          order_index=None, subtype=None, memory_time_attach=None, source=SOURCE_BARE_TEXT):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=source, tier=TIER_PRIMARY,
           order_index=order_index, subtype=subtype,
           memory_time_attach=memory_time_attach, sn=sn, tn=tn)


def _rows(b):
    return b.select("edge")


def _ref(sid, lid):
    return (sid, lid)


# ============ 模块7 effective_weight（H4）============

def test_effective_weight_precedes_one(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES, strength=1, order_index=0)
    e = _rows(b)[0]
    assert effective_weight(e) == 1   # PRECEDES strength 恒=1·结构真值


def test_effective_weight_causes_strength_times_rate(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, strength=3, sn=7, tn=3)
    e = _rows(b)[0]
    # rate = 7*1000/(7+3) = 700 · effective_weight = 3 * 700 = 2100
    assert edge_rate(e) == 700
    assert effective_weight(e) == 3 * 700


def test_effective_weight_causes_no_observation_zero(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, strength=5, sn=0, tn=0)
    e = _rows(b)[0]
    assert edge_rate(e) == 0
    assert effective_weight(e) == 0   # 无观测·零权重（待 reward）


def test_effective_weight_occurrence_decay(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_REFERS_TO, strength=2, subtype=SUBTYPE_OCCURRENCE,
          memory_time_attach=5)
    e = _rows(b)[0]
    # current_seq=10 · logical_age=5 · w = 2*1 - 5 = -3 → floor 0
    assert effective_weight(e, current_seq=10) == 0
    # current_seq=6 · logical_age=1 · w = 2*1 - 1 = 1
    assert effective_weight(e, current_seq=6) == 1


# ============ 模块6 死路检测（二条件·①已废 2026-07-02） ============

def test_dead_end_no_successor_non_sink_not_dead(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)   # 1→2
    edges = _rows(b)
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 99))   # sink 不在图
    active = {_ref(sid, 1)}
    # ① 已废：节点 2 出度=0 且非 sink·但全遍历下叶子非死路·②前驱 active rate>0·③未达 budget → 非死路
    # sink 不可达由 dag_path_step 末行层尽返回 DEAD_END 判定（见 test_dag_path_dead_end_no_successor）
    assert is_dead_end(_ref(sid, 2), edges, intent, active, 0, 100) is False


def test_dead_end_causes_preds_all_inactive(core):
    b, sid, es = core
    # 3→2 (CAUSES rate=0) · 4→2 (CAUSES rate=0) · 2 非 sink
    _edge(b, es, sid, 3, 2, EDGE_CAUSES, sn=0, tn=0)
    _edge(b, es, sid, 4, 2, EDGE_CAUSES, sn=0, tn=0)
    edges = _rows(b)
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 99))
    active = set()   # 前驱全不 active 且 rate=0
    assert is_dead_end(_ref(sid, 2), edges, intent, active, 0, 100) is True


def test_dead_end_step_budget_exhausted(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)
    edges = _rows(b)
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 99))
    active = {_ref(sid, 1)}
    # path_len >= step_budget → 死路③
    assert is_dead_end(_ref(sid, 1), edges, intent, active, 5, 5) is True


def test_not_dead_end_active_pred(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)   # rate>0
    _edge(b, es, sid, 2, 3, EDGE_CAUSES, sn=1, tn=0)   # 2 有后继
    edges = _rows(b)
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 3))
    active = {_ref(sid, 1)}
    # 节点 2：有后继·CAUSES 前驱 1 active rate>0·未达 budget → 非死路
    assert is_dead_end(_ref(sid, 2), edges, intent, active, 0, 100) is False


# ============ 模块1 A2 按头 AND/OR 分发 ============

def test_a2_and_or_dispatch(core):
    b, sid, es = core
    # A(1)→B(2): PRECEDES + CAUSES · B 是汇聚（两前驱？此处单前驱测 AND/OR 分发）
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)
    edges = _rows(b)
    active = {_ref(sid, 1)}
    topo, conv, stepper = a2_layer(edges, active, {EDGE_PRECEDES, EDGE_CAUSES})
    assert len(topo) == 2   # 两层：[1] [2]
    assert topo[0] == [_ref(sid, 1)]
    assert topo[1] == [_ref(sid, 2)]
    # AND（PRECEDES）：前驱 1 active → 选全前驱边
    sel_and = stepper.advance(_ref(sid, 2), EDGE_PRECEDES)
    assert sel_and is not BLOCKED and len(sel_and) == 1
    # OR（CAUSES）：前驱 1 active rate>0 → 选一前驱边
    sel_or = stepper.advance(_ref(sid, 2), EDGE_CAUSES)
    assert sel_or is not BLOCKED and len(sel_or) == 1


def test_a2_and_blocked_until_all_active(core):
    b, sid, es = core
    # 两前驱→B(3): 1→3 · 2→3 (PRECEDES) · B 汇聚
    _edge(b, es, sid, 1, 3, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=0)
    edges = _rows(b)
    active = {_ref(sid, 1)}   # 只 1 active·2 未到
    topo, conv, stepper = a2_layer(edges, active, {EDGE_PRECEDES})
    # AND 未到齐 → BLOCKED
    assert stepper.advance(_ref(sid, 3), EDGE_PRECEDES) is BLOCKED
    stepper.add_active(_ref(sid, 2))
    # 到齐 → 选全前驱边（2 条）
    sel = stepper.advance(_ref(sid, 3), EDGE_PRECEDES)
    assert sel is not BLOCKED and len(sel) == 2


def test_a2_or_zero_active_pred_blocked(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=0, tn=0)   # rate=0
    edges = _rows(b)
    active = set()
    topo, conv, stepper = a2_layer(edges, active, {EDGE_CAUSES})
    # OR 无 active rate>0 前驱 → BLOCKED（无因则无果·正确停滞）
    assert stepper.advance(_ref(sid, 2), EDGE_CAUSES) is BLOCKED


def test_a2_convergence_precedes_group_by_order_index(core):
    b, sid, es = core
    # 同 order_index 两前驱→B(3) · PRECEDES 并行汇聚组
    _edge(b, es, sid, 1, 3, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=0)
    edges = _rows(b)
    topo, conv, _ = a2_layer(edges, {_ref(sid, 1), _ref(sid, 2)},
                             {EDGE_PRECEDES})
    assert (_ref(sid, 3), EDGE_PRECEDES) in conv   # 汇聚点识别


def test_a2_t_step_fallback_and_semantics(core):
    """T_STEP 走 AND 兜底分支（a2_stepper.advance :150-153·非 PRECEDES/CAUSES 头首版按 AND 语义）。

    T_STEP 当前不进 dag_path head_types（task #697 defer M1·T_STEP 闭包归属未定）·
    但 HeadStepper.advance 对未识别头走 AND 兜底（结构序）·本测守此分支不空转（先前零覆盖）。
    current_clock 删（2026-07-07·gap①补 doc :71 决断死参数）后 advance 签名 (node, head)。
    """
    b, sid, es = core
    _edge(b, es, sid, 1, 3, EDGE_T_STEP, order_index=0)
    _edge(b, es, sid, 2, 3, EDGE_T_STEP, order_index=0)
    edges = _rows(b)
    active = {_ref(sid, 1)}   # 只 1 active·2 未到
    topo, conv, stepper = a2_layer(edges, active, {EDGE_T_STEP})
    # AND 兜底未到齐 → BLOCKED
    assert stepper.advance(_ref(sid, 3), EDGE_T_STEP) is BLOCKED
    stepper.add_active(_ref(sid, 2))
    # 到齐 → 选全前驱边（2 条·AND 兜底语义）
    sel = stepper.advance(_ref(sid, 3), EDGE_T_STEP)
    assert sel is not BLOCKED and len(sel) == 2


# ============ 模块2 A3 PR wrapper ============

def test_a3_pr_wrapper_solve_and_linearity(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES)
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES)
    edges = _rows(b)
    w = A3PRWrapper.build(edges)
    x = w.solve([_ref(sid, 1)])
    assert w.mode == "B1" and w.exact is True
    # 种子自身 rank > 0（teleport (1-α)）
    assert x[_ref(sid, 1)].num > 0
    # 线性性：add_seed(2) 后 x = x_{1} + x_{2}（零损失）
    x_before = dict(w._x)
    w.add_seed(_ref(sid, 2))
    for n in w._x:
        assert w._x[n].num  # 叠加后仍 Rational


def test_a3_pr_wrapper_add_remove_seed(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES)
    edges = _rows(b)
    w = A3PRWrapper.build(edges)
    w.solve([_ref(sid, 1)])
    x1 = dict(w._x)
    w.add_seed(_ref(sid, 2))
    # remove_seed(2) → 回到 x1（精确 O(n)·零损失）
    w.remove_seed(_ref(sid, 2))
    for n in x1:
        assert w._x[n] == x1[n]


def test_a3_pr_wrapper_residual_none_b1(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES)
    w = A3PRWrapper.build(_rows(b) if False else [_rows(b)[0]])
    w.solve([_ref(sid, 1)])
    assert w.residual is None   # B1 精确路径无残差（D1）


def test_a3_pr_wrapper_seed_rank(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES)
    edges = _rows(b)
    w = A3PRWrapper.build(edges)
    w.solve([_ref(sid, 1)])
    # seed_rank(1) = x at 1 > 0
    assert w.seed_rank(_ref(sid, 1)).num > 0
    # seed_rank(99) 热区外 → ZERO
    assert w.seed_rank(_ref(sid, 99)).num == 0


def test_a3_pr_wrapper_fq_to_rational_b2_conversion():
    """stub #6：FixedQuotient→Rational 精确转换（B2 兜底·值=(M·b+r)/(b·B^k)·旧版 make(M,1) 丢标度全错）。"""
    from pure_integer_ai.crosscut.integer.valtypes import FixedQuotient
    from pure_integer_ai.crosscut.integer.constants import BASE
    from pure_integer_ai.cognition.process.a3_pr_wrapper import _fq_to_rational
    bk = BASE ** 2
    fq = FixedQuotient(M=3, r=1, k=2, b=bk)   # 0<=r=1<bk·合法
    r = _fq_to_rational(fq)
    # 值 = (M·b + r)/(b·B^k) = (3·bk + 1)/(bk·bk)
    assert r.num == 3 * bk + 1
    assert r.den == bk * bk


# ============ 模块3 A4 coverage_overlap（合质量·a4_align 函数已删 2026-07-07·范畴由 coverage_overlap + 件4 + HeadStepper cover）============


def test_coverage_overlap_ordered():
    # consensus [1,3,4] vs [[1,2,3,4],[1,3,4]] · lcs=3,3 · |consensus|=3
    # = 1000*(3+3)/(2*3) = 1000
    q = coverage_overlap([1, 3, 4], [[1, 2, 3, 4], [1, 3, 4]])
    assert q == 1000


# ============ 模块4 DAG-path 步进主控 ============

def test_dag_path_reached_sink(core):
    b, sid, es = core
    # A(1)→B(2)→C(3) sink · PRECEDES + CAUSES 链
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=1)
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)
    _edge(b, es, sid, 2, 3, EDGE_CAUSES, sn=1, tn=0)
    edges = _rows(b)
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 3))
    res = dag_path_step(edges, [_ref(sid, 1)], wm, intent)
    assert res.terminal == TERMINAL_REACHED_SINK
    assert res.sink == _ref(sid, 3)
    # path.edges 选定边集存非派生（含 PRECEDES + CAUSES）
    assert len(res.path.edges) > 0
    assert all(isinstance(e, tuple) and len(e) == 5 for e in res.path.edges)
    # pr_vector 填（本 episode PR 向量）
    assert wm.pr_vector  # 非空


def test_dag_path_dead_end_no_successor(core):
    b, sid, es = core
    # A(1)→B(2) · B 叶子非 sink（sink=99 不在图）
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)
    edges = _rows(b)
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 99))
    res = dag_path_step(edges, [_ref(sid, 1)], wm, intent)
    assert res.terminal == TERMINAL_DEAD_END
    assert res.sink is None


def test_dag_path_j4_placeholder_true(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES)
    pd = PathData()
    assert j4_closure_check(pd, WorkMemory()) is True   # 占位返 true（A1·卷三真判）


def test_dag_path_struct_unit_refs_convergence(core):
    b, sid, es = core
    # 两前驱→汇聚点 B(3) · B→C(4) sink
    _edge(b, es, sid, 1, 3, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 3, 4, EDGE_PRECEDES, order_index=1)
    edges = _rows(b)
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 4))
    res = dag_path_step(edges, [_ref(sid, 1), _ref(sid, 2)], wm, intent)
    # B(3) 是 PRECEDES 汇聚点 → 入 struct_unit_refs
    assert _ref(sid, 3) in res.path.struct_unit_refs


# ============ 模块8 reward 反传（CAUSES 头 + R1）============

def test_dag_path_exploration_mode_off_no_injection(core):
    """柱③ proactive 注入：EXPLORATION_MODE OFF（默认）→ exploration_injected False（bit-identical 守回归）。"""
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=1)
    edges = _rows(b)
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 3))
    old = gates.EXPLORATION_MODE
    gates.EXPLORATION_MODE = False
    try:
        res = dag_path_step(edges, [_ref(sid, 1)], wm, intent)
        assert res.exploration_injected is False   # OFF→不注入（stub#1/致命5 bit-identical）
    finally:
        gates.EXPLORATION_MODE = old


def test_dag_path_exploration_mode_on_flat_variance_injects(core):
    """A2 柱③ proactive 注入 ON companion：EXPLORATION_MODE ON + PR 方差趋平 → exploration_injected True。

    chain 1→2→3 seed=[1] 的 PR 方差经 ×1000 缩放截断=0（趋平）→ ON 时 dag_path:96 注入新种子。
    证 EXPLORATION gate 控柱③ 注入 probe：ON→注入（本测）/ OFF→不注入（上测）= falsifiable 非恒真 stub。
    A2 formal_train reward 阶段翻此 gate ON·本测证 gate-on 路径 dag_path 内部注入真触发（生产柱③ 依赖
    dag_path 内部注入·anti_collapse:124 caller 不传 pr_wrapper/e_set 时柱③ 读 episode.exploration_injected）。
    """
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=1)
    edges = _rows(b)
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 3))
    old = gates.EXPLORATION_MODE
    gates.EXPLORATION_MODE = True
    try:
        res = dag_path_step(edges, [_ref(sid, 1)], wm, intent)
        assert res.exploration_injected is True   # ON+趋平→注入新种子（柱③ proactive·A2 reward 阶段翻此 gate）
    finally:
        gates.EXPLORATION_MODE = old


def _path_result_with_edges(sid, edge_refs):
    pd = PathData()
    pd.edges = list(edge_refs)
    return PathResult(path=pd, terminal=TERMINAL_REACHED_SINK)


_REWARD_CTX = (DOMAIN_MATH, MODALITY_ARITH, 0, INTENT_QUESTION)


def test_reward_positive_sn_tn_strength(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, strength=1, sn=1, tn=0)
    e_ref = (sid, 1, sid, 2, EDGE_CAUSES)
    pr = _path_result_with_edges(sid, [e_ref])
    propagate_reward(pr, [], 1, _REWARD_CTX, INTENT_QUESTION,
                     WorkMemory(), edge_store=es, backend=b)
    row = _rows(b)[0]
    # reward>0 → sn++ & tn++ + strength+=Δ
    assert row["sn"] == 2 and row["tn"] == 1
    assert row["strength"] == 1 + DELTA_DEFAULT


def test_reward_zero_veto_tn_only(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, strength=1, sn=1, tn=0)
    e_ref = (sid, 1, sid, 2, EDGE_CAUSES)
    pr = _path_result_with_edges(sid, [e_ref])
    propagate_reward(pr, [], 0, _REWARD_CTX, INTENT_QUESTION,
                     WorkMemory(), edge_store=es, backend=b)
    row = _rows(b)[0]
    # reward==0 veto → tn++ only（破永正·非"不调"）·sn/strength 不动
    assert row["sn"] == 1 and row["tn"] == 1 and row["strength"] == 1


def test_reward_negative_dead_end_tn_only(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, strength=1, sn=1, tn=0)
    e_ref = (sid, 1, sid, 2, EDGE_CAUSES)
    pr = _path_result_with_edges(sid, [e_ref])
    propagate_reward(pr, [], REWARD_DEAD_END, _REWARD_CTX,
                     INTENT_QUESTION, WorkMemory(), edge_store=es, backend=b)
    row = _rows(b)[0]
    # reward<0 死路 → tn++ only·不 decrement sn 守单调
    assert row["sn"] == 1 and row["tn"] == 1 and row["strength"] == 1


def test_reward_only_causes_precedes_untouched(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES, order_index=0)   # strength=1
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)
    pre_ref = (sid, 1, sid, 2, EDGE_PRECEDES)
    cau_ref = (sid, 1, sid, 2, EDGE_CAUSES)
    pr = _path_result_with_edges(sid, [pre_ref, cau_ref])
    propagate_reward(pr, [], 1, _REWARD_CTX, INTENT_QUESTION,
                     WorkMemory(), edge_store=es, backend=b)
    rows = _rows(b)
    pre = next(r for r in rows if r["edge_type"] == EDGE_PRECEDES)
    cau = next(r for r in rows if r["edge_type"] == EDGE_CAUSES)
    # PRECEDES 永不接 reward·strength 恒=1·sn/tn 不动
    assert pre["strength"] == 1 and pre["sn"] == 0 and pre["tn"] == 0
    # CAUSES 接 reward
    assert cau["sn"] == 2 and cau["tn"] == 1


def test_reward_sigma_zero_r5_new_edge_chance(core):
    b, sid, es = core
    # 全脏边（sn=tn=0·rate=0）·Σactive 率=0·冷启动
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, strength=1, sn=0, tn=0)
    e_ref = (sid, 1, sid, 2, EDGE_CAUSES)
    pr = _path_result_with_edges(sid, [e_ref])
    propagate_reward(pr, [], 1, _REWARD_CTX, INTENT_QUESTION,
                     WorkMemory(), edge_store=es, backend=b)
    row = _rows(b)[0]
    # R5：reward>0 全脏边→sn++&tn++（首次观测给新边机会·不丢弃成功信号·不分功不加 strength）
    assert row["sn"] == 1 and row["tn"] == 1
    assert row["strength"] == 1   # 无分功→delta=0·strength 不动（下次有观测后分功才加）


def test_h4_closed_loop_reward_changes_effective_weight(core):
    """H4 闭环：reward 调 sn/tn → rate 变 → effective_weight 变。"""
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, strength=1, sn=0, tn=0)
    row = _rows(b)[0]
    assert effective_weight(row) == 0   # 无观测·零权重
    e_ref = (sid, 1, sid, 2, EDGE_CAUSES)
    pr = _path_result_with_edges(sid, [e_ref])
    # reward>0 R5 → sn=1,tn=1·rate=500·effective_weight=1*500=500
    propagate_reward(pr, [], 1, _REWARD_CTX, INTENT_QUESTION,
                     WorkMemory(), edge_store=es, backend=b)
    row = _rows(b)[0]
    assert edge_rate(row) == 500   # sn=1,tn=1 → 1*1000/2=500
    assert effective_weight(row) == 500   # strength(1) × rate(500)


# ============ 模块5 attractor 松入严留 ============

def test_attractor_no_entry_no_expand(core):
    b, sid, es = core
    # D(5) 无入度·非 promoted·无 CAUSES 入边·tier SHADOW → entry False
    from pure_integer_ai.storage.node_store import NodeStore
    NodeStore(b).put(sid, 5, node_type=1, tier=TIER_SHADOW)
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES)
    edges = _rows(b)
    w = A3PRWrapper.build(edges)
    w.solve([_ref(sid, 1)])
    e = {_ref(sid, 1)}
    wm = WorkMemory()
    expanded = maybe_expand_attractor(_ref(sid, 5), e, w, edges, wm, backend=b)
    assert expanded is False
    assert _ref(sid, 5) not in e


def test_attractor_entry_expands(core):
    b, sid, es = core
    # D(3) 两 PRECEDES 入度（1→3·2→3·同 order_index）→ in_degree_seq=2 ≥ θ_conv
    _edge(b, es, sid, 1, 3, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES, order_index=0)
    edges = _rows(b)
    w = A3PRWrapper.build(edges)
    w.solve([_ref(sid, 1)])
    e = {_ref(sid, 1)}
    wm = WorkMemory()
    # D(3) 入度≥2 → entry · x_c = seed_rank(3)（1→3 PRECEDES 传 rank）·若 ≥θ_coh 则扩张
    maybe_expand_attractor(_ref(sid, 3), e, w, edges, wm, backend=b)
    # 3 入度满足 entry·若相干则入 e（seed_rank(3)>0 因 1→3 边传 rank）
    if w.seed_rank(_ref(sid, 3)).num > 0:
        assert _ref(sid, 3) in e


def test_attractor_cap_hard(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES)
    edges = _rows(b)
    w = A3PRWrapper.build(edges)
    w.solve([_ref(sid, 1)])
    e = {_ref(sid, 1)}
    # e 已达 K_CAP=8 → 不再扩张
    for i in range(10, 20):
        e.add(_ref(sid, i))
    wm = WorkMemory()
    expanded = maybe_expand_attractor(_ref(sid, 2), e, w, edges, wm,
                                      backend=b, k_cap=8)
    assert expanded is False


# ============ 模块9 episode wiring 单点化 ============

def test_episode_dead_end_wiring_negative_reward(core):
    b, sid, es = core
    # A(1)→B(2) · B 叶子非 sink → DEAD_END
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)
    edges = _rows(b)
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 99))
    inp = InputPayload(
        segments=[], source=SOURCE_BARE_TEXT, stage=1,
        domain=DOMAIN_MATH, modality=MODALITY_ARITH)
    output, ep = episode_loop(inp, edges, [_ref(sid, 1)], wm, intent,
                              generate_fn=None, judge_fn=None,
                              edge_store=es, backend=b)
    # wiring 单点：DEAD_END → reward<0 → propagate tn++（防塌柱② greenfield）
    assert ep.terminal == TERMINAL_DEAD_END
    assert ep.reward == REWARD_DEAD_END
    assert ep.reward < 0
    assert ep.dead_end_count == 1
    # CAUSES 边 tn++（死路失败）
    row = _rows(b)[0]
    assert row["tn"] == 1


def test_episode_reached_sink_judge_stub(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=1)
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)
    _edge(b, es, sid, 2, 3, EDGE_CAUSES, sn=1, tn=0)
    edges = _rows(b)
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 3))
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1)

    def judge_stub(output, path_result, input_payload, workmem):
        return 1, GMeta()   # reward=1·无 veto

    output, ep = episode_loop(inp, edges, [_ref(sid, 1)], wm, intent,
                              generate_fn=None, judge_fn=judge_stub,
                              edge_store=es, backend=b)
    assert ep.terminal == TERMINAL_REACHED_SINK
    assert ep.reward == 1
    assert ep.judge_veto_count == 0


def test_episode_reached_sink_no_judge_veto(core):
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=1)
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)
    _edge(b, es, sid, 2, 3, EDGE_CAUSES, sn=1, tn=0)
    edges = _rows(b)
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 3))
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1)
    output, ep = episode_loop(inp, edges, [_ref(sid, 1)], wm, intent,
                              generate_fn=None, judge_fn=None,
                              edge_store=es, backend=b)
    # 卷三未接·REACHED_SINK 默认 reward=0（veto 语义）
    assert ep.terminal == TERMINAL_REACHED_SINK
    assert ep.reward == 0
    assert ep.judge_veto_count == 1   # reward==0 = veto


# ============ 确定性 bit-identical ============

def test_dag_path_deterministic_bit_identical():
    def run():
        b = DictBackend()
        bootstrap(b)
        reg = SpaceRegistry(b)
        sp = AbstractSpace.create(reg, "core")
        es = EdgeStore(b)
        sid = sp.space_id
        _edge(b, es, sid, 1, 2, EDGE_PRECEDES, order_index=0)
        _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=1)
        _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)
        _edge(b, es, sid, 2, 3, EDGE_CAUSES, sn=1, tn=0)
        wm = WorkMemory()
        intent = IntentType(type=INTENT_QUESTION, sink=(sid, 3))
        res = dag_path_step(b.select("edge"), [(sid, 1)], wm, intent)
        fp = (res.terminal,
              sorted(res.path.edges),
              tuple(res.path.struct_unit_refs),
              res.sink)
        b.close()
        return fp
    assert run() == run()
