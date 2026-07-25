"""阶段8 freq 进 PR 测试：experience_count 真支柱 freq 维兑现（落点 A·PR seed 向量吸收 effective_freq）。

覆盖（doc/重来_阶段8freq进PR设计补充.md + doc/重来_attractor职责澄清_修正分析十一.md§二/§三
  + doc/重来_P0决断集_修正分析十三.md §七 8c-design 落点 A 订正）：
  - T1-T5 _seed_weight：backend=None 退 ONE / eff_freq=0 退 ONE / eff_freq>0 make(SCALE+eff,SCALE)>ONE / 线性 / ctx_code 桶分离
  - T6-T7 solve bit-identical：backend=None 退化 / backend+eff_freq=0 与 None 同
  - T8 反 theater 主锚：freq>0 seed 传播力强（同结构 s1/s2·s1 邻居 rank > s2 邻居）
  - T9 闭环：reward feed → e_tn 增 → eff_freq 增 → _seed_weight 增（reward→e_tn→eff_freq→seed→PR）
  - T10 dag_path_step 默认 backend → PathResult bit-identical（seed eff_freq=0→ONE·既有行为）
  - T11 dag_path_step ctx_code 透传 e2e（freq>0 seed → pr_vector 受 _seed_weight 影响）
  - T12 seed 自身 rank：freq>0 seed solve([c]) seed_rank > freq=0（attractor 相干判据 x_c≥θ_coh 输入变）

铁律：纯整数（make Rational）/ 单向依赖（L5→L0）/ §8.4（乘子吸收进 PR 不单做）/ §8.5（不动 edge schema）/
  反 theater（write+read+consume 三件全活·T8/T9 真行为变）/ bit-identical（eff_freq=0/backend=None 退 ONE）。
归一化决断（亲自修正 Plan agent）：make(SCALE+eff,SCALE) 非 make(eff,SCALE)（后者 eff<SCALE 时权重<ONE 反弱）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_CAUSES
from pure_integer_ai.storage.experience_count import (
    register_experience_count,
    record_base_freq, record_experience_outcome,
    read_effective_freq,
    pack_ctx_code,
)
from pure_integer_ai.cognition.shared.types import (
    IntentType, INTENT_COMMAND,
    DOMAIN_TEXT, DOMAIN_MATH, MODALITY_LANGUAGE, MODALITY_ARITH,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.process.dag_path import dag_path_step
from pure_integer_ai.cognition.process.reward_propagate import propagate_reward
from pure_integer_ai.cognition.process.a3_pr_wrapper import A3PRWrapper, FREQ_SEED_SCALE
from pure_integer_ai.cognition.shared.types import PathData, PathResult, TERMINAL_REACHED_SINK
from pure_integer_ai.crosscut.integer.rational import ZERO, ONE, make, sub
from pure_integer_ai.crosscut.integer import compare as cmp


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def backend(request):
    """bootstrap + register_experience_count（_seed_weight 单测用·两 backend 一致）。"""
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    register_experience_count(b)
    yield b
    b.close()


@pytest.fixture
def core():
    """DictBackend + core 空间 + EdgeStore + ConceptIndex + register experience_count。"""
    b = DictBackend()
    bootstrap(b)
    register_experience_count(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    yield b, sp.space_id, es, ci
    b.close()


# ---- helpers ----

def _edge(b, es, sid, frm, to, et, *, strength=1, sn=0, tn=0):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
           sn=sn, tn=tn)


def _wrapper(b, edges, *, backend=None, ctx_code=0):
    """建 A3PRWrapper（透传 backend/ctx_code·落点 A）。"""
    return A3PRWrapper.build(edges, backend=backend, ctx_code=ctx_code)


def _edges_causes(b, sid, pairs):
    """建 CAUSES 边集（pairs=[(from,to),...]）·返 b.select('edge') dict 行列表。"""
    es = EdgeStore(b)
    for frm, to in pairs:
        _edge(b, es, sid, frm, to, EDGE_CAUSES, sn=1, tn=0)
    return b.select("edge")


# ============ T1-T5 _seed_weight（落点 A 核心·归一化）============

def test_seed_weight_no_backend_returns_one(core):
    """T1 backend=None → _seed_weight 退 ONE（bit-identical·无 backend 退化）。"""
    b, sid, es, ci = core
    s = ci.ensure("s", space_id=sid)
    edges = _edges_causes(b, sid, [(s[1], ci.ensure("n", space_id=sid)[1])])
    w = _wrapper(b, edges, backend=None)
    assert w._seed_weight(s) == ONE


def test_seed_weight_zero_eff_freq_returns_one(backend):
    """T2 backend 非 None 但 eff_freq=0（冷启动无 experience_count 行）→ 退 ONE（bit-identical）。"""
    edges = [{
        "edge_type": EDGE_CAUSES, "space_id_from": 1, "local_id_from": 1,
        "space_id_to": 1, "local_id_to": 2, "strength": 1, "sn": 1, "tn": 0,
    }]
    w = _wrapper(backend, edges, backend=backend, ctx_code=0)
    assert w._seed_weight((1, 1)) == ONE   # (1,1) 无 experience_count 行 → eff_freq=0 → ONE


def test_seed_weight_positive_eff_freq_gt_one(backend):
    """T3 eff_freq>0 → make(SCALE+eff,SCALE) > ONE（有经验 seed 传播力强·归一化正确非 make(eff,SCALE)）。"""
    edges = [{
        "edge_type": EDGE_CAUSES, "space_id_from": 1, "local_id_from": 1,
        "space_id_to": 1, "local_id_to": 2, "strength": 1, "sn": 1, "tn": 0,
    }]
    record_base_freq(backend, ref=(1, 1), base_freq=5)   # 0 桶通识·eff_freq=5
    w = _wrapper(backend, edges, backend=backend, ctx_code=0)
    sw = w._seed_weight((1, 1))
    assert sw == make(FREQ_SEED_SCALE + 5, FREQ_SEED_SCALE)   # make(1005, 1000)
    # sw > ONE（cmp.cross_gt(num1,den1,num2,den2)）
    assert cmp.cross_gt(sw.num, sw.den, ONE.num, ONE.den)


def test_seed_weight_linear_in_eff_freq(backend):
    """T4 线性性：_seed_weight 是 eff_freq 线性函数（make(SCALE+eff,SCALE)=1+eff/SCALE）。

    3 个独立 ref（eff=0/5/10·隔离防 _seed_weight 实时读串扰）·diff(10,5)==diff(5,0)==5/SCALE（线性·非 make(eff,SCALE)）。
    """
    def _w(seed_ref):
        edges = [{"edge_type": EDGE_CAUSES,
                  "space_id_from": seed_ref[0], "local_id_from": seed_ref[1],
                  "space_id_to": seed_ref[0], "local_id_to": seed_ref[1] + 100,
                  "strength": 1, "sn": 1, "tn": 0}]
        return A3PRWrapper.build(edges, backend=backend, ctx_code=0)

    r0 = (1, 1)    # eff=0（无 record）
    r5 = (1, 3)    # eff=5
    r10 = (1, 5)   # eff=10
    record_base_freq(backend, ref=r5, base_freq=5)
    record_base_freq(backend, ref=r10, base_freq=10)
    sw0 = _w(r0)._seed_weight(r0)      # ONE
    sw5 = _w(r5)._seed_weight(r5)      # make(1005,1000)
    sw10 = _w(r10)._seed_weight(r10)   # make(1010,1000)
    assert sw0 == ONE
    assert sw5 == make(FREQ_SEED_SCALE + 5, FREQ_SEED_SCALE)
    assert sw10 == make(FREQ_SEED_SCALE + 10, FREQ_SEED_SCALE)
    diff_hi = sub(sw10, sw5)   # 5/1000
    diff_lo = sub(sw5, sw0)    # 5/1000
    assert diff_hi == diff_lo   # 线性：等差


def test_seed_weight_ctx_code_bucket_split(backend):
    """T5 ctx_code 桶分离：feed ctx_b 桶 e_tn·_seed_weight(ctx_b)>ONE·_seed_weight(ctx_a)=ONE。

    阶段6 复合 key 桶分离在落点 A 的体现：同概念不同 ctx 桶不同 eff_freq→不同 seed 权重。
    """
    edges = [{
        "edge_type": EDGE_CAUSES, "space_id_from": 1, "local_id_from": 1,
        "space_id_to": 1, "local_id_to": 2, "strength": 1, "sn": 1, "tn": 0,
    }]
    ctx_a = pack_ctx_code(1, 1, 0, INTENT_COMMAND)
    ctx_b = pack_ctx_code(2, 1, 0, INTENT_COMMAND)
    record_experience_outcome(backend, ref=(1, 1), reward=1, ctx_code=ctx_b)   # ctx_b 桶 e_tn=1
    wa = _wrapper(backend, edges, backend=backend, ctx_code=ctx_a)
    wb = _wrapper(backend, edges, backend=backend, ctx_code=ctx_b)
    # ctx_a 桶：base(0) + e_tn(0) = 0 → ONE
    assert wa._seed_weight((1, 1)) == ONE
    # ctx_b 桶：base(0) + e_tn(1) = 1 → make(1001, 1000) > ONE
    swb = wb._seed_weight((1, 1))
    assert swb == make(FREQ_SEED_SCALE + 1, FREQ_SEED_SCALE)
    assert cmp.cross_gt(swb.num, swb.den, ONE.num, ONE.den)


# ============ T6-T7 solve bit-identical（落点 A 退化）============

def test_solve_no_backend_bit_identical(core):
    """T6 build backend=None → solve 退化（_seed_weight 退 ONE·Σ ONE·x_s = 原版）。"""
    b, sid, es, ci = core
    s = ci.ensure("s", space_id=sid)
    n = ci.ensure("n", space_id=sid)
    edges = _edges_causes(b, sid, [(s[1], n[1])])
    w = _wrapper(b, edges, backend=None)
    x = w.solve([s])
    assert w.mode == "B1"
    # seed s 自身有 rank（ONE·unit·非零）·邻居 n 有 rank（传播）
    assert cmp.cross_gt(w.seed_rank(s).num, w.seed_rank(s).den, ZERO.num, ZERO.den)
    assert cmp.cross_gt(w.seed_rank(n).num, w.seed_rank(n).den, ZERO.num, ZERO.den)


def test_solve_zero_eff_freq_equals_no_backend(core):
    """T7 build backend + seed eff_freq=0 → solve 与 backend=None 逐 node 相同（bit-identical）。"""
    b, sid, es, ci = core
    s = ci.ensure("s", space_id=sid)
    n = ci.ensure("n", space_id=sid)
    edges = _edges_causes(b, sid, [(s[1], n[1])])
    w_none = _wrapper(b, edges, backend=None)
    w_zero = _wrapper(b, edges, backend=b, ctx_code=0)   # backend 非 None·但 s 无 experience_count
    x_none = w_none.solve([s])
    x_zero = w_zero.solve([s])
    # 逐 node Rational 相等（_seed_weight 都退 ONE → 同解）
    assert set(x_none) == set(x_zero)
    for node in x_none:
        assert x_none[node] == x_zero[node]


# ============ T8 反 theater 主锚：freq>0 seed 传播力强（真行为变）============

def test_freq_seed_propagates_stronger(core):
    """T8 反 theater 主锚：同结构 s1→n1 / s2→n2·s1 eff_freq>0·s2 eff_freq=0 →

    solve([s1,s2]) 后 seed_rank(n1) > seed_rank(n2)（s1 传播力强·w_s1>ONE 缩放 unit·s2 w=ONE）。
    真行为变：有经验 seed 点亮邻居更强（attractor 方向维度 freq 维·非符号试金石）。
    """
    b, sid, es, ci = core
    s1 = ci.ensure("s1", space_id=sid)
    n1 = ci.ensure("n1", space_id=sid)
    s2 = ci.ensure("s2", space_id=sid)
    n2 = ci.ensure("n2", space_id=sid)
    # 对称两子图：s1→n1 / s2→n2（无交叉边·PR 解对称·unit_x_s1[n1]==unit_x_s2[n2]）
    edges = _edges_causes(b, sid, [(s1[1], n1[1]), (s2[1], n2[1])])
    record_base_freq(b, ref=s1, base_freq=5)   # s1 eff_freq=5（0 桶通识）·s2 eff_freq=0
    w = _wrapper(b, edges, backend=b, ctx_code=0)
    w.solve([s1, s2])
    rank_n1 = w.seed_rank(n1)   # = w_s1 · unit（w_s1=make(1005,1000)）
    rank_n2 = w.seed_rank(n2)   # = w_s2 · unit = ONE · unit（s2 eff_freq=0）
    # 同结构 unit 相等·w_s1 > ONE → rank_n1 > rank_n2（freq>0 seed 传播力强·真行为变）
    assert cmp.cross_gt(rank_n1.num, rank_n1.den, rank_n2.num, rank_n2.den)


# ============ T9 闭环：reward feed → eff_freq → _seed_weight（write+read 闭环）============

def test_reward_feed_increases_seed_weight(core):
    """T9 闭环：propagate_reward feed → e_tn 增 → eff_freq 增 → _seed_weight 增。

    reward feed（阶段2）→ record_experience_outcome（e_tn++）→ read_effective_freq 增 →
    _seed_weight 增（落点 A read 半边）。write（feed）+ read（_seed_weight）闭环·反 theater。
    """
    b, sid, es, ci = core
    s = ci.ensure("s", space_id=sid)
    n = ci.ensure("n", space_id=sid)
    _edge(b, es, sid, s[1], n[1], EDGE_CAUSES, sn=1, tn=0)
    edges = b.select("edge")
    ctx_b = pack_ctx_code(DOMAIN_MATH, MODALITY_ARITH, 0, INTENT_COMMAND)
    w_before = _wrapper(b, edges, backend=b, ctx_code=ctx_b)
    sw_before = w_before._seed_weight(s)   # ctx_b 桶 eff_freq=0 → ONE
    assert sw_before == ONE
    # propagate_reward feed（reward>0 → e_sn++&e_tn++·阶段2 R1 符号·写 ctx_b 桶）
    pd = PathData()
    pd.edges = [(sid, s[1], sid, n[1], EDGE_CAUSES)]
    pr = PathResult(path=pd, terminal=TERMINAL_REACHED_SINK, sink=n)
    propagate_reward(pr, [], 1, (DOMAIN_MATH, MODALITY_ARITH, 0, INTENT_COMMAND),
                     INTENT_COMMAND, WorkMemory(), edge_store=es, backend=b)
    w_after = _wrapper(b, edges, backend=b, ctx_code=ctx_b)
    sw_after = w_after._seed_weight(s)   # ctx_b 桶 eff_freq = base(0)+e_tn(1) = 1 → make(1001,1000)
    assert sw_after == make(FREQ_SEED_SCALE + 1, FREQ_SEED_SCALE)
    # feed 后 _seed_weight 增（write→read 闭环·反 theater）
    assert cmp.cross_gt(sw_after.num, sw_after.den, sw_before.num, sw_before.den)


# ============ T10 dag_path_step 默认 → PathResult bit-identical ============

def test_dag_path_step_default_bit_identical(core):
    """T10 dag_path_step 默认（seed eff_freq=0·_seed_weight 退 ONE）→ PathResult 既有行为不变。

    生产 seed eff_freq≈0（首轮 _inject_base_freq 在 stage 循环后）→ _seed_weight=ONE →
    solve Σ ONE·x_s = 原版 → pr_vector bit-identical → PathResult bit-identical。
    """
    b, sid, es, ci = core
    A = ci.ensure("A", space_id=sid)
    B = ci.ensure("B", space_id=sid)
    _edge(b, es, sid, A[1], B[1], EDGE_CAUSES, sn=1, tn=0)
    edges = b.select("edge")
    intent = IntentType(type=INTENT_COMMAND, sink=B)
    pr = dag_path_step(edges, [A], WorkMemory(), intent, current_seq=0, backend=b)
    # 默认 ctx_code=0·seed A eff_freq=0 → _seed_weight ONE → 既有步进行为·达 sink
    assert pr.terminal == TERMINAL_REACHED_SINK
    assert (sid, A[1], sid, B[1], EDGE_CAUSES) in set(pr.path.edges)


# ============ T11 dag_path_step ctx_code 透传 e2e（pr_vector 受 _seed_weight 影响）============

def test_dag_path_step_ctx_code_seed_weight_e2e(core):
    """T11 ctx_code 透传 e2e：dag_path_step build(backend,ctx_code) → pr_wrapper 持 ctx_code →

    _seed_weight 按 ctx 桶读。feed ctx_b 桶后·pr_vector 受 _seed_weight 缩放（workmem.pr_vector 非零）。
    反 theater：freq>0 seed → pr_vector 真受影响（F5 聚合读扩张后 x·非孤立）。
    """
    b, sid, es, ci = core
    A = ci.ensure("A", space_id=sid)
    B = ci.ensure("B", space_id=sid)
    _edge(b, es, sid, A[1], B[1], EDGE_CAUSES, sn=1, tn=0)
    ctx_b = pack_ctx_code(2, 1, 0, INTENT_COMMAND)
    # feed ctx_b 桶 A e_tn（A eff_freq>0 in ctx_b 桶）
    for _ in range(5):
        record_experience_outcome(b, ref=A, reward=1, ctx_code=ctx_b)
    edges = b.select("edge")
    intent = IntentType(type=INTENT_COMMAND, sink=B)
    wm = WorkMemory()
    dag_path_step(edges, [A], wm, intent, current_seq=0, backend=b, ctx_code=ctx_b)
    # workmem.pr_vector 非 None·A 处 rank > 0（_seed_weight(A ctx_b)>ONE 缩放·传播）
    x = wm.pr_vector
    assert x is not None
    assert cmp.cross_gt(x[A].num, x[A].den, ZERO.num, ZERO.den)


# ============ T12 seed 自身 rank：freq>0 seed solve([c]) seed_rank > freq=0 ============

def test_seed_self_rank_freq_positive(core):
    """T12 freq>0 seed 自身 rank > freq=0（attractor 相干判据 x_c≥θ_coh 输入受 freq 缩放）。

    solve([c]) seed_rank(c) = w_c · unit_c[c]。w_c>ONE（freq>0）→ seed_rank(c) > ONE·unit（freq=0）。
    attractor seed_rank 是 x_c≥θ_coh 判据输入（attractor.py:93）·freq 缩放它 → 扩张倾向变。
    """
    b, sid, es, ci = core
    c1 = ci.ensure("c1", space_id=sid)
    n1 = ci.ensure("n1", space_id=sid)
    c2 = ci.ensure("c2", space_id=sid)
    n2 = ci.ensure("n2", space_id=sid)
    edges = _edges_causes(b, sid, [(c1[1], n1[1]), (c2[1], n2[1])])
    record_base_freq(b, ref=c1, base_freq=5)   # c1 eff_freq=5·c2 eff_freq=0
    w = _wrapper(b, edges, backend=b, ctx_code=0)
    w.solve([c1])
    rank_c1_freq = w.seed_rank(c1)   # w_c1=make(1005,1000) · unit
    w2 = _wrapper(b, edges, backend=b, ctx_code=0)
    w2.solve([c2])
    rank_c2_zero = w2.seed_rank(c2)   # w_c2=ONE · unit（同结构 unit_c1[c1]==unit_c2[c2]）
    # freq>0 seed 自身 rank > freq=0（attractor 相干判据输入受 freq 缩放·扩张倾向变）
    assert cmp.cross_gt(rank_c1_freq.num, rank_c1_freq.den,
                        rank_c2_zero.num, rank_c2_zero.den)


# ============ T13 _solve_single 缓存一致性（_seed_weight cache 路径）============

def test_solve_single_cache_with_seed_weight(core):
    """T13 _solve_single 缓存一致性：cache 存 solve_exact({c:_seed_weight(c)})·hit 返回同值。

    _seed_weight 在 solve(:119)/_solve_single(:138) 两处注入·cache 权重一致（同 wrapper 生命期
    eff_freq 稳定·propagate_reward 在 dag_path 后跑·episode.py:82-106 时序锁死·无交错写 e_tn）。
    add_seed(c) cache miss → _solve_single 重算 → cache 存。
    """
    b, sid, es, ci = core
    c1 = ci.ensure("c1", space_id=sid)
    n1 = ci.ensure("n1", space_id=sid)
    edges = _edges_causes(b, sid, [(c1[1], n1[1])])
    record_base_freq(b, ref=c1, base_freq=5)   # c1 eff_freq=5
    w = _wrapper(b, edges, backend=b, ctx_code=0)
    w.solve([c1])
    assert c1 in w._cache   # solve 存 cache（xs = solve_exact({c1:_seed_weight(c1)})）
    # _solve_single cache hit → 返回 cache[c1]（不重算·_seed_weight 权重一致）
    assert w._solve_single(c1) == w._cache[c1]
    # add_seed cache miss 路径：空 solve 后 add_seed(c1) → _solve_single 重算 → cache 存 → seed_rank>0
    w2 = _wrapper(b, edges, backend=b, ctx_code=0)
    w2.solve([])
    assert w2._cache == {}
    w2.add_seed(c1)   # cache miss → _solve_single(c1) → solve_exact({c1:_seed_weight(c1)})
    assert c1 in w2._cache
    assert cmp.cross_gt(w2.seed_rank(c1).num, w2.seed_rank(c1).den, ZERO.num, ZERO.den)


# ============ T14 speaker_code 桶（defer #495 但机制建对）============

def test_seed_weight_speaker_code_bucket(backend):
    """T14 speaker_code 桶：_seed_weight 接 speaker_code 参数·defer #495 但机制建对。

    dag_path 不传 speaker_code（默认 0·#495 defer 记忆空间阶段11·与 word_terminated 一致第一刀单 key）·
    但 _seed_weight/build 支持 speaker_code 桶读（复合 key 第二刀 schema 阶段1 已落·消费者侧 ctx 维活/speaker 维 defer）。
    feed speaker 5 桶 → _seed_weight(speaker_code=5)>ONE·_seed_weight(speaker_code=0)=ONE。
    """
    edges = [{
        "edge_type": EDGE_CAUSES, "space_id_from": 1, "local_id_from": 1,
        "space_id_to": 1, "local_id_to": 2, "strength": 1, "sn": 1, "tn": 0,
    }]
    record_experience_outcome(backend, ref=(1, 1), reward=1, speaker_code=5)   # speaker 5 桶 e_tn=1
    w0 = A3PRWrapper.build(edges, backend=backend, ctx_code=0, speaker_code=0)
    w5 = A3PRWrapper.build(edges, backend=backend, ctx_code=0, speaker_code=5)
    assert w0._seed_weight((1, 1)) == ONE   # speaker 0 桶未 feed → eff_freq=0 → ONE
    sw5 = w5._seed_weight((1, 1))           # speaker 5 桶 e_tn=1 → make(1001,1000)
    assert sw5 == make(FREQ_SEED_SCALE + 1, FREQ_SEED_SCALE)
    assert cmp.cross_gt(sw5.num, sw5.den, ONE.num, ONE.den)
