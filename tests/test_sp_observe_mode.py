"""方案3 tn路 sp_observe_tn 测试（B5 β_arith 修法·STEP4 P1b·#893）。

覆盖（doc/重来_纠偏轮_round2_任务文档_2026-07-10.md §五 B5）：
  - storage：record_selection_pref_cooccur 加写 sp_observe_tn（sign-agnostic / 首次建行 / +=1 /
    不碰 base/sp_sn / after-reward 增量 / 表未注册 skip）+ record_selection_pref_reward 不写 sp_observe_tn
    （reward 路不碰·显式 0 避 NULL corruption）+ read_selection_pref_count observe_mode（OFF=sp_tn
    bit-identical / ON=sp_observe_tn / cold-start None / 仅 sp_observe_tn 不含 sp_tn / 分化）+
    read_selection_pref_agg observe_mode（OFF/ON/分化/cold-start）
  - consumer e2e：_seed_weight sp_agg observe_mode（gate ON 读 sum_sp_observe_tn·PR 侧粗筛·对偶 B4 w_freq）+
    selection_pref_score observe_mode（gate ON 读 sp_observe_tn·生成侧精查·sp_sn 仍读诚实边界）

β_arith 病：sp_tn 混 observe（record_selection_pref_cooccur 段内共现 sign-agnostic）+ reward
  （record_selection_pref_reward episode 末 reward>0 sp_sn++&sp_tn++）→ reward>0 episode 同 concept_targets
  同比 sp_tn++ → rate 塌缩·sp_observe_tn 由 record_selection_pref_cooccur 同写（observe 路纯副本·sign-agnostic·
  独立 episode reward 符号）替 sp_tn 作 consumer 源·跨决策分化。

铁律：纯整数 / MUTABLE_MONOTONE（sp_observe_tn += 1）/ reward CAUSES-only（observe 不接 reward·独立表统计非 reward feed）/
bit-identical（gate OFF 零行为变·SP_OBSERVE_MODE 默认 OFF）/ 不碰 base_count append-only / storage gate-free（observe_mode caller 传参）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.selection_pref_count import (
    register_selection_pref_count,
    record_selection_pref_cooccur, record_selection_pref_reward,
    read_selection_pref_count, read_selection_pref_agg,
    SELECTION_PREF_COUNT_TABLE,
)
from pure_integer_ai.storage.experience_count import register_experience_count, record_base_freq
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.process.a3_pr_wrapper import A3PRWrapper, FREQ_SEED_SCALE, SP_SEED_SCALE
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.crosscut.integer.rational import ONE, make, mul, eq


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def backend(request):
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    register_selection_pref_count(b)
    register_experience_count(b)
    yield b
    b.close()


@pytest.fixture
def gate_off():
    """守 SP_OBSERVE_MODE OFF + 用后复位（bit-identical 基线）。"""
    saved = gates.SP_OBSERVE_MODE
    gates.SP_OBSERVE_MODE = False
    try:
        yield
    finally:
        gates.SP_OBSERVE_MODE = saved


@pytest.fixture
def gate_on():
    """翻 SP_OBSERVE_MODE ON + 用后复位。"""
    saved = gates.SP_OBSERVE_MODE
    gates.SP_OBSERVE_MODE = True
    try:
        yield
    finally:
        gates.SP_OBSERVE_MODE = saved


@pytest.fixture
def sp_env():
    """backend + core space + ConceptIndex（_seed_weight/selection_pref_score e2e 用）。"""
    b = DictBackend()
    bootstrap(b)
    register_selection_pref_count(b)
    register_experience_count(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    ci = ConceptIndex(b)
    yield b, sp.space_id, ci
    b.close()


# ---- helpers ----

def _ensure(ci, sid, surface):
    return ci.ensure(surface, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)


def _observe_tn(b, ref_a, ref_class):
    """读 sp_observe_tn 列（0 if 无行）。"""
    rows = b.select(SELECTION_PREF_COUNT_TABLE, where={
        "space_id_from": ref_a[0], "local_id_from": ref_a[1],
        "space_id_to": ref_class[0], "local_id_to": ref_class[1]}, limit=1)
    return rows[0]["sp_observe_tn"] if rows else 0


# ============ record_selection_pref_cooccur 加写 sp_observe_tn（storage 单测）============

def test_record_cooccur_creates_row_with_observe_tn(backend):
    """首次：insert(base=0, sp_sn=0, sp_tn=1, sp_observe_tn=1)·两列同写。"""
    b = backend
    a, c = (1, 10), (1, 11)
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c)
    assert read_selection_pref_count(b, a, c) == (0, 0, 1)   # 既有 3-tuple 不变
    assert _observe_tn(b, a, c) == 1


def test_record_cooccur_increments_observe_tn(backend):
    """多次调 → sp_observe_tn += 1 & sp_tn += 1（MUTABLE_MONOTONE·两列同写·delta +1）。"""
    b = backend
    a, c = (1, 12), (1, 13)
    for _ in range(5):
        record_selection_pref_cooccur(b, ref_a=a, ref_class=c)
    assert _observe_tn(b, a, c) == 5
    assert read_selection_pref_count(b, a, c) == (0, 0, 5)   # sp_tn 也=5（两列同写）


def test_record_cooccur_observe_tn_sign_agnostic():
    """sign-agnostic：record_selection_pref_cooccur 不接 reward（β_arith 修法核心·与 record_selection_pref_reward 区别）。"""
    import inspect
    sig = inspect.signature(record_selection_pref_cooccur)
    assert "reward" not in sig.parameters


def test_record_cooccur_does_not_touch_base_sn(backend):
    """observe 不碰 base_count/sp_sn（reward CAUSES-only·observe 是段内共现统计非 reward feed）。"""
    b = backend
    a, c = (1, 14), (1, 15)
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c)              # sp_tn=1, sp_observe_tn=1
    record_selection_pref_reward(b, ref_a=a, ref_class=c, reward=5)     # sp_sn=1, sp_tn=2（reward 路）
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c)              # sp_tn=3, sp_observe_tn=2
    base, sn, tn = read_selection_pref_count(b, a, c)
    assert base == 0
    assert sn == 1                                          # reward 路写·observe 不碰
    assert tn == 3                                          # sp_tn = 2 observe + 1 reward
    assert _observe_tn(b, a, c) == 2                       # sp_observe_tn = 2 observe only（reward 路不写）


def test_record_cooccur_after_reward_increments_observe(backend):
    """异常顺序（reward 先建行 sp_observe_tn=0）→ observe_tn += 1 on existing。"""
    b = backend
    a, c = (1, 16), (1, 17)
    record_selection_pref_reward(b, ref_a=a, ref_class=c, reward=5)     # 建行 sp_observe_tn=0
    assert _observe_tn(b, a, c) == 0
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c)              # sp_observe_tn += 1 → 1
    assert _observe_tn(b, a, c) == 1


def test_record_cooccur_table_unregistered_skip():
    """表未注册 → KeyError 静默 skip（向后兼容·镜像 record_base_freq 范式）。"""
    b = DictBackend()
    bootstrap(b)
    record_selection_pref_cooccur(b, ref_a=(1, 18), ref_class=(1, 19))   # 不崩
    b.close()


# ============ record_selection_pref_reward 不写 sp_observe_tn（storage 单测）============

def test_record_reward_does_not_write_observe_tn(backend):
    """reward 路不写 sp_observe_tn（reward>0 写 sp_sn+sp_tn·不碰 sp_observe_tn·显式 0 避 NULL corruption）。"""
    b = backend
    a, c = (1, 20), (1, 21)
    record_selection_pref_reward(b, ref_a=a, ref_class=c, reward=5)     # 建行 sp_sn=1, sp_tn=1, sp_observe_tn=0
    base, sn, tn = read_selection_pref_count(b, a, c)
    assert (base, sn, tn) == (0, 1, 1)
    assert _observe_tn(b, a, c) == 0                       # reward 路不写 sp_observe_tn
    record_selection_pref_reward(b, ref_a=a, ref_class=c, reward=5)     # 再次 reward → sp_tn++ & sp_sn++·sp_observe_tn 不变
    assert _observe_tn(b, a, c) == 0
    base, sn, tn = read_selection_pref_count(b, a, c)
    assert (base, sn, tn) == (0, 2, 2)


# ============ read_selection_pref_count observe_mode（storage 单测）============

def test_read_count_observe_mode_off_default(backend):
    """observe_mode 默认 False → 第 3 元=sp_tn（既有 bit-identical·gate OFF 路径）。"""
    b = backend
    a, c = (1, 30), (1, 31)
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c)              # sp_tn=1, sp_observe_tn=1
    record_selection_pref_reward(b, ref_a=a, ref_class=c, reward=5)     # sp_tn=2, sp_observe_tn=1
    assert read_selection_pref_count(b, a, c) == (0, 1, 2)              # default observe_mode=False → sp_tn=2
    assert read_selection_pref_count(b, a, c, observe_mode=False) == (0, 1, 2)


def test_read_count_observe_mode_on(backend):
    """observe_mode=True → 第 3 元=sp_observe_tn（gate ON 路径·替 sp_tn·β_arith 修法）。"""
    b = backend
    a, c = (1, 32), (1, 33)
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c)              # sp_observe_tn=1
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c)              # sp_observe_tn=2
    record_selection_pref_reward(b, ref_a=a, ref_class=c, reward=5)     # sp_tn=3, sp_observe_tn=2
    assert read_selection_pref_count(b, a, c, observe_mode=True) == (0, 1, 2)   # sp_observe_tn=2 非 sp_tn=3


def test_read_count_observe_mode_only_observe_tn(backend):
    """observe_mode=True 仅用 sp_observe_tn·不含 sp_tn 的 reward 路（pure sign-agnostic 源）。"""
    b = backend
    a, c = (1, 34), (1, 35)
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c)              # sp_observe_tn=1, sp_tn=1
    record_selection_pref_reward(b, ref_a=a, ref_class=c, reward=5)     # sp_tn=2, sp_observe_tn=1
    assert read_selection_pref_count(b, a, c, observe_mode=True) == (0, 1, 1)    # sp_observe_tn=1·非 sp_tn=2
    assert read_selection_pref_count(b, a, c, observe_mode=False) == (0, 1, 2)   # sp_tn=2


def test_read_count_observe_mode_differentiates(backend):
    """β_arith 修法核心：两 pair observe 次数不同 → observe_mode=True 分化 3:1·False 分化 2:1（reward 路 +1 拉平）。"""
    b = backend
    a1, c1 = (1, 40), (1, 41)
    a2, c2 = (1, 42), (1, 43)
    # 两 pair 各 1 次 reward（sp_sn=1, sp_tn=1, sp_observe_tn=0）
    record_selection_pref_reward(b, ref_a=a1, ref_class=c1, reward=5)
    record_selection_pref_reward(b, ref_a=a2, ref_class=c2, reward=5)
    # a1 被 observe 3 次·a2 被 observe 1 次（决策活动分化）
    for _ in range(3):
        record_selection_pref_cooccur(b, ref_a=a1, ref_class=c1)
    record_selection_pref_cooccur(b, ref_a=a2, ref_class=c2)
    # observe_mode=True → sp_observe_tn 分化 3 vs 1（纯 observe 路·sign-agnostic·β_arith 修法）
    assert read_selection_pref_count(b, a1, c1, observe_mode=True)[2] == 3
    assert read_selection_pref_count(b, a2, c2, observe_mode=True)[2] == 1
    # observe_mode=False → sp_tn 分化 4 vs 2（observe + reward·reward 路 +1 拉平·分化比 2:1 < 3:1）
    assert read_selection_pref_count(b, a1, c1, observe_mode=False)[2] == 4
    assert read_selection_pref_count(b, a2, c2, observe_mode=False)[2] == 2


def test_read_count_observe_mode_cold_start(backend):
    """冷启动：无行 → observe_mode 两态都 None。"""
    b = backend
    assert read_selection_pref_count(b, (1, 99), (1, 98), observe_mode=False) is None
    assert read_selection_pref_count(b, (1, 99), (1, 98), observe_mode=True) is None


# ============ read_selection_pref_agg observe_mode（storage 单测）============

def test_read_agg_observe_mode_off_default(backend):
    """observe_mode 默认 False → 第 3 元=sum_sp_tn（既有 bit-identical）。"""
    b = backend
    a = (1, 50)
    c1, c2 = (1, 51), (1, 52)
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c1)             # sp_tn=1, sp_observe_tn=1
    record_selection_pref_reward(b, ref_a=a, ref_class=c1, reward=5)    # sp_tn=2, sp_observe_tn=1
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c2)             # sp_tn=1, sp_observe_tn=1
    sum_base, sum_sn, sum_tn = read_selection_pref_agg(b, a)
    assert (sum_base, sum_sn, sum_tn) == (0, 1, 3)                      # sum_sp_tn = 2+1=3


def test_read_agg_observe_mode_on(backend):
    """observe_mode=True → 第 3 元=sum_sp_observe_tn（替 sum_sp_tn·避 reward 染色）。"""
    b = backend
    a = (1, 53)
    c1, c2 = (1, 54), (1, 55)
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c1)             # sp_observe_tn=1
    record_selection_pref_reward(b, ref_a=a, ref_class=c1, reward=5)    # sp_tn=2, sp_observe_tn=1
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c2)             # sp_observe_tn=1
    sum_base, sum_sn, sum_tn = read_selection_pref_agg(b, a, observe_mode=True)
    assert (sum_base, sum_sn, sum_tn) == (0, 1, 2)                      # sum_sp_observe_tn = 1+1=2（不含 reward 路 sp_tn）


def test_read_agg_observe_mode_differentiates(backend):
    """β_arith 修法：两 concept observe 次数不同 → observe_mode=True 分化 3:1·False 分化 2:1（reward 路 +1 拉平）。"""
    b = backend
    a1, a2 = (1, 60), (1, 61)
    c = (1, 62)
    record_selection_pref_reward(b, ref_a=a1, ref_class=c, reward=5)    # sp_tn=1, sp_observe_tn=0
    record_selection_pref_reward(b, ref_a=a2, ref_class=c, reward=5)    # sp_tn=1, sp_observe_tn=0
    for _ in range(3):
        record_selection_pref_cooccur(b, ref_a=a1, ref_class=c)         # sp_observe_tn=3, sp_tn=4
    record_selection_pref_cooccur(b, ref_a=a2, ref_class=c)             # sp_observe_tn=1, sp_tn=2
    assert read_selection_pref_agg(b, a1, observe_mode=True)[2] == 3
    assert read_selection_pref_agg(b, a2, observe_mode=True)[2] == 1
    assert read_selection_pref_agg(b, a1, observe_mode=False)[2] == 4
    assert read_selection_pref_agg(b, a2, observe_mode=False)[2] == 2


def test_read_agg_observe_mode_cold_start(backend):
    """冷启动：无行 → observe_mode 两态都 (0,0,0)。"""
    b = backend
    assert read_selection_pref_agg(b, (1, 99), observe_mode=False) == (0, 0, 0)
    assert read_selection_pref_agg(b, (1, 99), observe_mode=True) == (0, 0, 0)


# ============ consumer e2e：_seed_weight sp_agg observe_mode（PR 侧粗筛·对偶 B4 w_freq）============

def test_seed_weight_sp_observe_mode_gate_off_reads_sp_tn(sp_env, gate_off):
    """gate OFF → _seed_weight 读 sum_sp_tn（既有 bit-identical·sp_agg = sum_base + sum_sp_tn）。"""
    b, sid, ci = sp_env
    a = _ensure(ci, sid, "追")
    animal = _ensure(ci, sid, "动物")
    record_base_freq(b, ref=a, base_freq=2)                             # eff_freq=2
    for _ in range(3):
        record_selection_pref_cooccur(b, ref_a=a, ref_class=animal)     # sp_observe_tn=3, sp_tn=3
    record_selection_pref_reward(b, ref_a=a, ref_class=animal, reward=5)   # sp_tn=4, sp_observe_tn=3
    wrapper = A3PRWrapper.build([], backend=b)
    saved_dock = gates.SELECTION_PREF_DOCK_MODE
    gates.SELECTION_PREF_DOCK_MODE = True
    try:
        w = wrapper._seed_weight(a)
    finally:
        gates.SELECTION_PREF_DOCK_MODE = saved_dock
    # gate OFF sp_agg = 0 + 4 (sum_sp_tn) = 4
    w_freq = make(FREQ_SEED_SCALE + 2, FREQ_SEED_SCALE)
    w_sp = make(SP_SEED_SCALE + 4, SP_SEED_SCALE)
    assert eq(w, mul(w_freq, w_sp)), f"gate OFF sp_agg=sum_sp_tn=4·got {w}"


def test_seed_weight_sp_observe_mode_gate_on_reads_sp_observe_tn(sp_env, gate_on):
    """gate ON → _seed_weight 读 sum_sp_observe_tn（避 reward 染色·sp_agg = sum_base + sum_sp_observe_tn）。"""
    b, sid, ci = sp_env
    a = _ensure(ci, sid, "追")
    animal = _ensure(ci, sid, "动物")
    record_base_freq(b, ref=a, base_freq=2)
    for _ in range(3):
        record_selection_pref_cooccur(b, ref_a=a, ref_class=animal)     # sp_observe_tn=3
    record_selection_pref_reward(b, ref_a=a, ref_class=animal, reward=5)   # sp_tn=4, sp_observe_tn=3
    wrapper = A3PRWrapper.build([], backend=b)
    saved_dock = gates.SELECTION_PREF_DOCK_MODE
    gates.SELECTION_PREF_DOCK_MODE = True
    try:
        w = wrapper._seed_weight(a)
    finally:
        gates.SELECTION_PREF_DOCK_MODE = saved_dock
    # gate ON sp_agg = 0 + 3 (sum_sp_observe_tn) = 3（避 reward 路 +1）
    w_freq = make(FREQ_SEED_SCALE + 2, FREQ_SEED_SCALE)
    w_sp = make(SP_SEED_SCALE + 3, SP_SEED_SCALE)
    assert eq(w, mul(w_freq, w_sp)), f"gate ON sp_agg=sum_sp_observe_tn=3·got {w}"


# ============ consumer e2e：selection_pref_score observe_mode（生成侧精查）============

def test_selection_pref_score_observe_mode_gate_off_reads_sp_tn(sp_env, gate_off):
    """gate OFF → selection_pref_score 读 sp_sn+sp_tn（既有 bit-identical）。"""
    b, sid, ci = sp_env
    chase = _ensure(ci, sid, "追")
    mouse = _ensure(ci, sid, "老鼠")    # 无 IS_A·class_of(mouse)=mouse
    record_selection_pref_cooccur(b, ref_a=chase, ref_class=mouse)     # sp_observe_tn=1, sp_tn=1
    record_selection_pref_reward(b, ref_a=chase, ref_class=mouse, reward=5)   # sp_tn=2, sp_sn=1, sp_observe_tn=1
    g = ConceptGraph(b)
    score = g.selection_pref_score(mouse, [chase])     # Σ (sp_sn+sp_tn)(chase, class_of(mouse)=mouse)
    assert score == 3, f"gate OFF sp_sn+sp_tn=1+2=3·got {score}"


def test_selection_pref_score_observe_mode_gate_on_reads_sp_observe_tn(sp_env, gate_on):
    """gate ON → selection_pref_score 读 sp_sn+sp_observe_tn（替 sp_tn·避 reward 染色·sp_sn 仍读诚实边界）。"""
    b, sid, ci = sp_env
    chase = _ensure(ci, sid, "追")
    mouse = _ensure(ci, sid, "老鼠")
    record_selection_pref_cooccur(b, ref_a=chase, ref_class=mouse)     # sp_observe_tn=1
    record_selection_pref_reward(b, ref_a=chase, ref_class=mouse, reward=5)   # sp_tn=2, sp_sn=1, sp_observe_tn=1
    g = ConceptGraph(b)
    score = g.selection_pref_score(mouse, [chase])
    assert score == 2, f"gate ON sp_sn+sp_observe_tn=1+1=2（sp_tn=2 reward 路不读）·got {score}"
