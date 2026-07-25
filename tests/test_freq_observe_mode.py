"""方案3 tn路 observe_tn 测试（B4 β_arith 修法·STEP4 P1a·#892）。

覆盖（B4 频率观察修正 +
  doc/重来_纠偏轮_round2_任务文档_2026-07-10.md §五 B4）：
  - storage：record_experience_observe（sign-agnostic / 首次建行 / +=1 / 不碰 base/e_sn/e_tn /
    after-outcome 增量 / 表未注册 skip）+ read_effective_freq observe_mode（OFF=base+e_tn bit-identical /
    ON=base+observe_tn / cold-start 0 / 仅 observe_tn 不含 e_tn）
  - dag_path e2e：gate OFF 不写（bit-identical）/ gate ON 写 path 节点 observe_tn / once-per-node（双头不过计）
  - attractor e2e：gate OFF 不写 / gate ON 扩张节点写 observe_tn（_seed_weight freq 维真活）

β_arith 病：reward>0 episode 同比 e_sn++&e_tn++ → rate 塌缩·w_freq 概念间同·
observe_tn 决策时写（sign-agnostic·独立 episode reward 符号）替 e_tn 作 w_freq 源·跨 episode 分化。

铁律：纯整数 / MUTABLE_MONOTONE（observe_tn += 1）/ reward CAUSES-only（observe 不接 reward·概念维决策活动统计）/
bit-identical（gate OFF 零行为变）/ 不碰 base_freq append-only / storage gate-free（observe_mode caller 传参）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES, EDGE_CAUSES
from pure_integer_ai.storage.experience_count import (
    register_experience_count,
    record_base_freq, record_experience_outcome, record_experience_observe,
    read_experience_count, read_effective_freq,
)
from pure_integer_ai.cognition.shared.types import (
    IntentType, INTENT_COMMAND, TERMINAL_REACHED_SINK,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.process.dag_path import dag_path_step
from pure_integer_ai.cognition.process.attractor import maybe_expand_attractor
from pure_integer_ai.cognition.process.a3_pr_wrapper import A3PRWrapper


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def backend(request):
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    register_experience_count(b)
    yield b
    b.close()


@pytest.fixture
def gate_off():
    """守 gate FREQ_OBSERVE_MODE OFF + 用后复位（bit-identical 基线·不污染其他测）。"""
    saved = gates.FREQ_OBSERVE_MODE
    gates.FREQ_OBSERVE_MODE = False
    try:
        yield
    finally:
        gates.FREQ_OBSERVE_MODE = saved


@pytest.fixture
def gate_on():
    """翻 gate FREQ_OBSERVE_MODE ON + 用后复位。"""
    saved = gates.FREQ_OBSERVE_MODE
    gates.FREQ_OBSERVE_MODE = True
    try:
        yield
    finally:
        gates.FREQ_OBSERVE_MODE = saved


@pytest.fixture
def core():
    """建 backend + core 空间 + EdgeStore + register_experience_count。"""
    b = DictBackend()
    bootstrap(b)
    register_experience_count(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    yield b, sp.space_id, es
    b.close()


# ---- helpers ----

def _ref(sid, lid):
    return (sid, lid)


def _edge(b, es, sid, frm, to, et, *, strength=1, sn=0, tn=0, order_index=None):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
           order_index=order_index, sn=sn, tn=tn)


def _observe_tn(b, ref):
    """读 observe_tn 列（0 if 无行）。"""
    r = read_experience_count(b, ref)
    if r is None:
        return 0
    rows = b.select("experience_count", where={
        "space_id": ref[0], "local_id": ref[1], "ctx_code": 0, "speaker_code": 0}, limit=1)
    return rows[0]["observe_tn"] if rows else 0


# ============ record_experience_observe（storage 单测）============

def test_record_experience_observe_creates_row(backend):
    """首次：insert(base=0, e_sn=0, e_tn=0, observe_tn=1)·镜像 record_experience_outcome 首次。"""
    b = backend
    ref = (1, 10)
    record_experience_observe(b, ref=ref)
    base, sn, tn = read_experience_count(b, ref)
    assert (base, sn, tn) == (0, 0, 0)          # 不碰 base/e_sn/e_tn
    assert _observe_tn(b, ref) == 1


def test_record_experience_observe_increments(backend):
    """多次调 → observe_tn += 1（MUTABLE_MONOTONE·delta +1·跨决策累积）。"""
    b = backend
    ref = (1, 11)
    for _ in range(5):
        record_experience_observe(b, ref=ref)
    assert _observe_tn(b, ref) == 5
    base, sn, tn = read_experience_count(b, ref)
    assert (base, sn, tn) == (0, 0, 0)          # 累积只 observe_tn·其他列不变


def test_record_experience_observe_sign_agnostic_no_reward():
    """sign-agnostic：不接 reward 参数（与 record_experience_outcome 签名区别·β_arith 修法核心）。"""
    import inspect
    sig = inspect.signature(record_experience_observe)
    assert "reward" not in sig.parameters     # 不接 reward·独立 episode reward 符号
    assert "observe_tn" not in sig.parameters  # 非显式传值·内部 += 1


def test_record_experience_observe_does_not_touch_base_sn_tn(backend):
    """observe 不碰 base_freq/e_sn/e_tn（reward CAUSES-only 铁律·observe 是决策活动统计非 reward feed）。"""
    b = backend
    ref = (1, 12)
    record_base_freq(b, ref=ref, base_freq=100)       # base_freq=100
    record_experience_outcome(b, ref=ref, reward=5)   # e_sn=1, e_tn=1
    record_experience_observe(b, ref=ref)             # observe_tn=1·不碰其他
    record_experience_observe(b, ref=ref)             # observe_tn=2
    base, sn, tn = read_experience_count(b, ref)
    assert (base, sn, tn) == (100, 1, 1)              # base/e_sn/e_tn 不变
    assert _observe_tn(b, ref) == 2


def test_record_experience_observe_after_outcome_increments(backend):
    """异常顺序（reward feed 先建行 base=0）→ observe_tn += 1 on existing（诚实降级·base 留 0·同 record_base_freq 公约）。"""
    b = backend
    ref = (1, 13)
    record_experience_outcome(b, ref=ref, reward=5)   # 先建行 (base=0, e_sn=1, e_tn=1, observe_tn=0)
    record_experience_observe(b, ref=ref)             # observe_tn += 1 → 1
    base, sn, tn = read_experience_count(b, ref)
    assert (base, sn, tn) == (0, 1, 1)                # 诚实降级·base 留 0
    assert _observe_tn(b, ref) == 1


def test_record_experience_observe_table_unregistered_skip():
    """表未注册（bare fixture）→ KeyError 静默 skip（向后兼容·镜像 record_base_freq/record_experience_outcome）。"""
    b = DictBackend()   # 未 register_experience_count
    bootstrap(b)
    record_experience_observe(b, ref=(1, 14))   # 不崩
    assert b.select("experience_count", where=None) == [] \
        if "experience_count" in getattr(b, "_data", {}) else True
    b.close()


# ============ read_effective_freq observe_mode（storage 单测）============

def test_read_effective_freq_observe_mode_off_default(backend):
    """observe_mode 默认 False → base+e_tn（既有 bit-identical·gate OFF 路径）。"""
    b = backend
    ref = (1, 20)
    record_base_freq(b, ref=ref, base_freq=100)
    record_experience_outcome(b, ref=ref, reward=5)   # e_tn=1
    record_experience_observe(b, ref=ref)             # observe_tn=1
    assert read_effective_freq(b, ref) == 101             # default observe_mode=False → 100+1(e_tn)
    assert read_effective_freq(b, ref, observe_mode=False) == 101


def test_read_effective_freq_observe_mode_on(backend):
    """observe_mode=True → base+observe_tn（gate ON 路径·替 e_tn·β_arith 修法）。"""
    b = backend
    ref = (1, 21)
    record_base_freq(b, ref=ref, base_freq=100)
    record_experience_outcome(b, ref=ref, reward=5)   # e_tn=1
    record_experience_observe(b, ref=ref)             # observe_tn=1
    record_experience_observe(b, ref=ref)             # observe_tn=2
    assert read_effective_freq(b, ref, observe_mode=True) == 102  # 100+2(observe_tn)


def test_read_effective_freq_observe_mode_only_observe_tn(backend):
    """observe_mode=True 仅用 observe_tn·不含 e_tn（e_tn β_arith 塌缩弃用·pure sign-agnostic 源）。"""
    b = backend
    ref = (1, 22)
    record_base_freq(b, ref=ref, base_freq=10)
    record_experience_outcome(b, ref=ref, reward=5)   # e_tn=1
    record_experience_observe(b, ref=ref)             # observe_tn=1
    # observe_mode=True → 10+1(observe_tn)=11·非 10+1(e_tn)+1(observe_tn)=12
    assert read_effective_freq(b, ref, observe_mode=True) == 11
    # 对照 observe_mode=False → 10+1(e_tn)=11（此处 e_tn==observe_tn 巧合·下测分离）
    assert read_effective_freq(b, ref, observe_mode=False) == 11


def test_read_effective_freq_observe_mode_differentiates(backend):
    """β_arith 修法核心：两 concept 同 e_tn（rate 塌缩）但不同 observe_tn → observe_mode=True 分化·False 不分化。"""
    b = backend
    a, c = (1, 30), (1, 31)
    for ref in (a, c):
        record_base_freq(b, ref=ref, base_freq=0)
        record_experience_outcome(b, ref=ref, reward=5)   # 两 concept 同 e_tn=1（β_arith 塌缩）
    # a 被 observe 3 次·c 被 observe 1 次（决策活动分化）
    for _ in range(3):
        record_experience_observe(b, ref=a)
    record_experience_observe(b, ref=c)
    # observe_mode=False（gate OFF·既有）→ 两 concept 同 eff_freq=1（e_tn 塌缩·β_arith 病）
    assert read_effective_freq(b, a, observe_mode=False) == 1
    assert read_effective_freq(b, c, observe_mode=False) == 1
    # observe_mode=True（gate ON）→ 两 concept eff_freq 分化（observe_tn 3 vs 1·β_arith 修法）
    assert read_effective_freq(b, a, observe_mode=True) == 3
    assert read_effective_freq(b, c, observe_mode=True) == 1


def test_read_effective_freq_observe_mode_cold_start(backend):
    """冷启动：无行 → observe_mode 两态都 0（消费者按 0 处理）。"""
    b = backend
    assert read_effective_freq(b, (1, 99), observe_mode=False) == 0
    assert read_effective_freq(b, (1, 99), observe_mode=True) == 0


# ============ dag_path e2e（改点 6·add_active 后 observe_tn）============

def _causes_chain(core):
    """A(1)→B(2)→C(3) CAUSES 链·sink=C(3)·返 (b, sid, edges, intent)。"""
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)   # A→B
    _edge(b, es, sid, 2, 3, EDGE_CAUSES, sn=1, tn=0)   # B→C
    edges = b.select("edge")
    intent = IntentType(type=INTENT_COMMAND, sink=_ref(sid, 3))
    return b, sid, edges, intent


def test_dag_path_observe_gate_off_no_write(core, gate_off):
    """gate OFF → dag_path 不写 observe_tn（bit-identical·既有行为零变）。"""
    b, sid, edges, intent = _causes_chain(core)
    pr = dag_path_step(edges, [_ref(sid, 1)], WorkMemory(), intent,
                       current_seq=0, backend=b, theta_freq=1000)
    assert pr.terminal == TERMINAL_REACHED_SINK
    # 全 path 节点 observe_tn=0（无行或 0）
    for lid in (1, 2, 3):
        assert _observe_tn(b, _ref(sid, lid)) == 0


def test_dag_path_observe_gate_on_writes(core, gate_on):
    """gate ON → dag_path add_active 后写 path 节点 observe_tn（喂 word_terminated consumer）。"""
    b, sid, edges, intent = _causes_chain(core)
    pr = dag_path_step(edges, [_ref(sid, 1)], WorkMemory(), intent,
                       current_seq=0, backend=b, theta_freq=1000)
    assert pr.terminal == TERMINAL_REACHED_SINK
    # path-reached 节点（B·C 有入边→add_active）observe_tn=1·A(source 无入边) 可能 0
    # 核验：至少 B 或 C 有 observe_tn=1（机制真活·反 theater）
    observed = [_observe_tn(b, _ref(sid, lid)) for lid in (1, 2, 3)]
    assert max(observed) >= 1                    # 有 path 节点被 observe
    assert all(o <= 1 for o in observed)         # once-per-node（无过计）


def test_dag_path_observe_once_per_node_dual_head(core, gate_on):
    """双头节点（PRECEDES+CAUSES 同入）→ observe_tn += 1 仅一次（_node_activated flag·避 per-head 过计）。"""
    b, sid, es = core
    # P(4) 同时有 PRECEDES + CAUSES 入边（双头）·source S(1)
    _edge(b, es, sid, 1, 4, EDGE_CAUSES, sn=1, tn=0)       # S→P CAUSES
    _edge(b, es, sid, 1, 4, EDGE_PRECEDES, order_index=0)  # S→P PRECEDES（双头）
    _edge(b, es, sid, 4, 5, EDGE_CAUSES, sn=1, tn=0)       # P→X
    edges = b.select("edge")
    intent = IntentType(type=INTENT_COMMAND, sink=_ref(sid, 5))
    dag_path_step(edges, [_ref(sid, 1)], WorkMemory(), intent,
                  current_seq=0, backend=b, theta_freq=1000)
    # P(4) 双头成功但 observe_tn 仅 1（once-per-node·非 2）
    assert _observe_tn(b, _ref(sid, 4)) <= 1


# ============ attractor 不写 observe_tn（改点 7 移除·对抗审隐患 B 修）============

def _attractor_expand_setup(core):
    """两 PRECEDES 入边 c(3)·入度=2 ≥ θ_conv=2 → entry·返 (b, sid, edges, w, e)。"""
    b, sid, es = core
    _edge(b, es, sid, 1, 3, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=0)
    edges = b.select("edge")
    w = A3PRWrapper.build(edges, backend=b)
    w.solve([_ref(sid, 1), _ref(sid, 2)])
    e = {_ref(sid, 1), _ref(sid, 2)}
    return b, sid, edges, w, e


def test_attractor_does_not_write_observe(core, gate_on):
    """改点 7 移除：maybe_expand_attractor 不写 observe_tn（dag_path add_active 已覆盖 path-reached 节点·
    attractor 写致双写 +2 违 once-per-node·对抗审隐患 B 修）。attractor-expanded ⊆ path-reached·dag_path 写。"""
    b, sid, edges, w, e = _attractor_expand_setup(core)
    maybe_expand_attractor(_ref(sid, 3), e, w, edges, WorkMemory(), backend=b)
    # maybe_expand_attractor 无论是否扩张·都不写 observe_tn（dag_path 职责）
    assert _observe_tn(b, _ref(sid, 3)) == 0


def test_dag_path_attractor_on_no_double_write(core, gate_on):
    """隐患 B 修：ATTRACTOR_MODE + FREQ_OBSERVE_MODE 同时 ON（生产 formal_train 叠加）→
    path-reached+attractor-expanded 节点 observe_tn 仅 1（非 2·dag_path add_active 单写·attractor 不另写）。

    构造 c(3) 双 PRECEDES 入边（in_degree=2 ≥ θ_conv=2 → attractor entry+扩张）+ c(3)→X(CAUSES) sink=X·
    dag_path 路径达 c(3) → add_active 写 observe_tn=1·maybe_expand_attractor(c3) 扩张但不另写 → observe_tn=1 非 2。
    """
    b, sid, es = core
    _edge(b, es, sid, 1, 3, EDGE_PRECEDES, order_index=0)   # 入边 1（attractor in_degree）
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=0)   # 入边 2（in_degree=2 ≥ θ_conv=2 → entry）
    _edge(b, es, sid, 3, 4, EDGE_CAUSES, sn=1, tn=0)        # c(3)→X(4)·sink=X
    edges = b.select("edge")
    intent = IntentType(type=INTENT_COMMAND, sink=_ref(sid, 4))
    saved_attr = gates.ATTRACTOR_MODE
    gates.ATTRACTOR_MODE = True                              # 生产 formal_train 叠加 ATTRACTOR ON
    try:
        dag_path_step(edges, [_ref(sid, 1), _ref(sid, 2)], WorkMemory(), intent,
                      current_seq=0, backend=b, theta_freq=1000)
    finally:
        gates.ATTRACTOR_MODE = saved_attr
    # c(3) path-reached + attractor-expanded·observe_tn=1（dag_path 单写）非 2（无双写）
    assert _observe_tn(b, _ref(sid, 3)) == 1
