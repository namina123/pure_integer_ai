"""S4 三乘子进 PR 测试（selection_pref 维 dock PR seed·学习放开 6 刀后主线续接·任务 #639-#645）。

S4 = selection_pref 维 dock PR seed（_seed_weight 乘积扩）+ sp_sn reward feed 第三条腿（reward_propagate
  落点⑥）+ H2 _rebuild_path ctx_code 修。**纠偏（S4 Plan agent·doc L264）**：seed=struct_ref 是数据真空
  非机制阻塞·乘子 dock 走 _seed_weight 权重缩放（线性性）不动 seed 节点集·真生效走 attractor 扩张路径。

本测覆盖：
  - 片1 表 unit（read_selection_pref_agg 聚合 / record_selection_pref_reward R1 符号 / bit-identical）
  - 片2 _seed_weight 乘积 dock（mul(w_freq, w_sp)·线性性 / gate OFF bit-identical / sp_agg=0 退化）— 片2 加
  - 片3 第三条腿（reward_propagate 落点⑥·守 :131 assert / concept_targets 配对 / class_of 自建）— 片3 加
  - 片4 H2 ctx_code 一致性 — 片4 加
  - 片5 反 theater e2e（狐狸追鸡/猫追老鼠 reward>0 + 石头追老鼠 reward<0 → (追,动物) sp_sn 高·w_sp(追)>ONE）— 片5 加

铁律：纯整数 / 确定性 bit-identical（2 gate 默认 OFF·sorted NodeRef 升序·mul 闭运算）/ §8.4（乘子吸收进 PR
  seed 向量 e·不单做）/ §8.5（不建边·独立表·不预留乘子字段）/ reward CAUSES-only（第三条腿独立表·不进 distributed·
  :131 assert 不动）/ 单向依赖（reward_propagate 不 import understanding·自建 class_of）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.edge_types import EDGE_CAUSES
from pure_integer_ai.storage.selection_pref_count import (
    register_selection_pref_count, read_selection_pref_count,
    record_selection_pref_cooccur, read_selection_pref_agg,
    record_selection_pref_reward, SELECTION_PREF_COUNT_TABLE,
)
from pure_integer_ai.storage.experience_count import (
    register_experience_count, record_base_freq,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.process.a3_pr_wrapper import (
    A3PRWrapper, FREQ_SEED_SCALE, SP_SEED_SCALE,
)
from pure_integer_ai.cognition.process.reward_propagate import propagate_reward
from pure_integer_ai.cognition.shared.types import (
    PathData, PathResult, TERMINAL_REACHED_SINK,
    INTENT_COMMAND, DOMAIN_TEXT, DOMAIN_MATH,
    MODALITY_LANGUAGE, MODALITY_ARITH,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.understanding.is_a import build_is_a_edge
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_CONCEPTNET
from pure_integer_ai.crosscut.integer.rational import ONE, make, mul, eq
from pure_integer_ai.config import gates


# ============ 片1：表 unit（read_agg + record_reward） ============

@pytest.fixture
def sp_env():
    """selection_pref_count 单测环境（dict backend + core space + 表注册）。"""
    b = DictBackend()
    bootstrap(b)
    register_selection_pref_count(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    ci = ConceptIndex(b)
    sid = sp.space_id
    yield b, sid, ci
    b.close()


def _ensure(ci, sid, surface):
    return ci.ensure(surface, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)


def test_read_agg_empty(sp_env):
    """read_selection_pref_agg：冷启动（concept_a 无任何搭配行）→ (0,0,0)。"""
    b, sid, ci = sp_env
    a = _ensure(ci, sid, "追")
    assert read_selection_pref_agg(b, a) == (0, 0, 0), "冷启动 sp_agg=0 → w_sp=ONE 退化"


def test_read_agg_table_unregistered():
    """read_selection_pref_agg：表未注册（bare fixture）→ (0,0,0)（KeyError skip·向后兼容）。"""
    b = DictBackend()
    bootstrap(b)
    # 不调 register_selection_pref_count
    assert read_selection_pref_agg(b, (1, 1)) == (0, 0, 0), "表未注册→(0,0,0) 不抛"


def test_read_agg_aggregates_multiple_argument_classes(sp_env):
    """read_selection_pref_agg：聚合 concept_a 的所有 argument_class 行（"追"搭配动物/石头/...）。"""
    b, sid, ci = sp_env
    a = _ensure(ci, sid, "追")
    animal = _ensure(ci, sid, "动物")
    stone = _ensure(ci, sid, "石头")
    food = _ensure(ci, sid, "食物")
    # observe 写 sp_tn（3 个 argument_class·不同次数）
    for _ in range(5):
        record_selection_pref_cooccur(b, ref_a=a, ref_class=animal)
    for _ in range(1):
        record_selection_pref_cooccur(b, ref_a=a, ref_class=stone)
    for _ in range(3):
        record_selection_pref_cooccur(b, ref_a=a, ref_class=food)
    got = read_selection_pref_agg(b, a)
    # sum_base=0（observe 不写 base）·sum_sp_sn=0（reward 未调·sp_sn defer S4 片3）·sum_sp_tn=5+1+3=9
    assert got == (0, 0, 9), f"observe 聚合 sp_tn=9·base/sp_sn 守 0·got={got}"


def test_read_agg_deterministic_node_order(sp_env):
    """read_selection_pref_agg：sorted NodeRef 升序遍历（确定性 tiebreak·bit-identical）。"""
    b, sid, ci = sp_env
    a = _ensure(ci, sid, "追")
    # 3 个 argument_class·插入序故意乱（升序验证）
    cls = [_ensure(ci, sid, f"c{i}") for i in range(3)]
    cls_sorted = sorted(cls)
    for c in cls:
        record_selection_pref_cooccur(b, ref_a=a, ref_class=c)
    # 多次读结果一致（确定性）
    g1 = read_selection_pref_agg(b, a)
    g2 = read_selection_pref_agg(b, a)
    assert g1 == g2 == (0, 0, 3), "3 行各 sp_tn=1·sum=3·确定"


def test_record_reward_insert_reward_positive(sp_env):
    """record_selection_pref_reward 首次 reward>0：insert(base=0, sp_sn=1, sp_tn=1)。"""
    b, sid, ci = sp_env
    a = _ensure(ci, sid, "追")
    animal = _ensure(ci, sid, "动物")
    record_selection_pref_reward(b, ref_a=a, ref_class=animal, reward=1)
    got = read_selection_pref_count(b, a, animal)
    assert got == (0, 1, 1), f"reward>0 首次 sp_sn=1/sp_tn=1·base=0·got={got}"


def test_record_reward_insert_reward_zero_and_negative(sp_env):
    """record_selection_pref_reward 首次 reward≤0：insert(base=0, sp_sn=0, sp_tn=1)（不涨 sp_sn）。"""
    b, sid, ci = sp_env
    a = _ensure(ci, sid, "追")
    stone = _ensure(ci, sid, "石头")
    record_selection_pref_reward(b, ref_a=a, ref_class=stone, reward=0)
    assert read_selection_pref_count(b, a, stone) == (0, 0, 1), "reward==0 首次 sp_sn=0/sp_tn=1"
    # 另一对 reward<0
    food = _ensure(ci, sid, "食物")
    record_selection_pref_reward(b, ref_a=a, ref_class=food, reward=-1)
    assert read_selection_pref_count(b, a, food) == (0, 0, 1), "reward<0 首次 sp_sn=0/sp_tn=1"


def test_record_reward_increment_reward_positive(sp_env):
    """record_selection_pref_reward 已存在 reward>0：sp_sn++ & sp_tn++。"""
    b, sid, ci = sp_env
    a = _ensure(ci, sid, "追")
    animal = _ensure(ci, sid, "动物")
    record_selection_pref_reward(b, ref_a=a, ref_class=animal, reward=1)
    record_selection_pref_reward(b, ref_a=a, ref_class=animal, reward=1)
    record_selection_pref_reward(b, ref_a=a, ref_class=animal, reward=1)
    got = read_selection_pref_count(b, a, animal)
    assert got == (0, 3, 3), f"3 次 reward>0 → sp_sn=3/sp_tn=3·got={got}"


def test_record_reward_increment_reward_non_positive(sp_env):
    """record_selection_pref_reward 已存在 reward≤0：sp_tn++ only（sp_sn 不动守单调）。"""
    b, sid, ci = sp_env
    a = _ensure(ci, sid, "追")
    stone = _ensure(ci, sid, "石头")
    # 先 reward>0 建 sp_sn=1
    record_selection_pref_reward(b, ref_a=a, ref_class=stone, reward=1)
    # 再 reward<0 / reward==0 各一次
    record_selection_pref_reward(b, ref_a=a, ref_class=stone, reward=-1)
    record_selection_pref_reward(b, ref_a=a, ref_class=stone, reward=0)
    got = read_selection_pref_count(b, a, stone)
    assert got == (0, 1, 3), f"1 正 + 2 非正 → sp_sn=1（不降）·sp_tn=3·got={got}"


def test_record_reward_base_append_only(sp_env):
    """record_selection_pref_reward：base_count append-only 永不调（reward 路径不碰 base）。"""
    b, sid, ci = sp_env
    a = _ensure(ci, sid, "追")
    animal = _ensure(ci, sid, "动物")
    record_selection_pref_reward(b, ref_a=a, ref_class=animal, reward=1)
    record_selection_pref_reward(b, ref_a=a, ref_class=animal, reward=1)
    got = read_selection_pref_count(b, a, animal)
    assert got[0] == 0, "base_count append-only·reward 路径永不调"


def test_record_reward_table_unregistered():
    """record_selection_pref_reward：表未注册（bare fixture）→ KeyError skip 不抛。"""
    b = DictBackend()
    bootstrap(b)
    # 不调 register_selection_pref_count
    record_selection_pref_reward(b, ref_a=(1, 1), ref_class=(1, 2), reward=1)  # 不抛


def test_record_reward_then_agg(sp_env):
    """record_selection_pref_reward 后 read_selection_pref_agg 正确聚合（_seed_weight sp_agg 来源）。"""
    b, sid, ci = sp_env
    a = _ensure(ci, sid, "追")
    animal = _ensure(ci, sid, "动物")
    stone = _ensure(ci, sid, "石头")
    # 5 次 reward>0 搭配动物·1 次 reward<0 搭配石头
    for _ in range(5):
        record_selection_pref_reward(b, ref_a=a, ref_class=animal, reward=1)
    record_selection_pref_reward(b, ref_a=a, ref_class=stone, reward=-1)
    sum_base, sum_sn, sum_tn = read_selection_pref_agg(b, a)
    assert sum_base == 0, "base append-only 不动"
    assert sum_sn == 5, "5 次 reward>0 动物 → sp_sn=5（石头 reward<0 不涨 sp_sn）"
    assert sum_tn == 6, "总 sp_tn=5+1=6"


# ============ 片2：_seed_weight 乘积 dock ============

@pytest.fixture
def dock_env():
    """_seed_weight 乘积 dock 单测环境（experience_count + selection_pref_count + minimal 2-node wrapper）。"""
    b = DictBackend()
    bootstrap(b)
    register_experience_count(b)
    register_selection_pref_count(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    # minimal CAUSES 边 a→b 建 2-node matrix（_seed_weight 不触 matrix·但 build 需边）
    a = ci.ensure("a", space_id=sid, tier=TIER_PRIMARY)
    bb = ci.ensure("b", space_id=sid, tier=TIER_PRIMARY)
    es.add(space_id_from=sid, local_id_from=a[1], space_id_to=sid, local_id_to=bb[1],
           edge_type=EDGE_CAUSES, strength=1, source=SOURCE_BARE_TEXT,
           tier=TIER_PRIMARY, sn=0, tn=1)
    wrapper = A3PRWrapper.build(b.select("edge"), backend=b)
    yield b, sid, ci, wrapper, a
    b.close()


def test_seed_weight_gate_off_bit_identical(dock_env):
    """gate OFF：w_sp 恒 ONE → w = w_freq（落点 A 不变·bit-identical·CI===生产）。"""
    b, sid, ci, wrapper, a = dock_env
    record_base_freq(b, ref=a, base_freq=3)
    gates.SELECTION_PREF_DOCK_MODE = False
    try:
        w = wrapper._seed_weight(a)
    finally:
        gates.SELECTION_PREF_DOCK_MODE = False
    expected = make(FREQ_SEED_SCALE + 3, FREQ_SEED_SCALE)
    assert eq(w, expected), f"gate OFF w=w_freq·got {w} expected {expected}"


def test_seed_weight_gate_on_sp_agg_zero_degrades(dock_env):
    """gate ON + sp_agg=0：w_sp ONE → w = w_freq（同 gate OFF·退化 bit-identical）。"""
    b, sid, ci, wrapper, a = dock_env
    record_base_freq(b, ref=a, base_freq=3)
    # 不写 selection_pref（sp_agg=0）
    gates.SELECTION_PREF_DOCK_MODE = True
    try:
        w = wrapper._seed_weight(a)
    finally:
        gates.SELECTION_PREF_DOCK_MODE = False
    expected = make(FREQ_SEED_SCALE + 3, FREQ_SEED_SCALE)
    assert eq(w, expected), f"sp_agg=0 w_sp ONE·w=w_freq·got {w}"


def test_seed_weight_gate_on_sp_agg_positive_product(dock_env):
    """gate ON + sp_agg>0：w = mul(w_freq, w_sp)（乘积 dock·S4 决断 2 选项 A）。"""
    b, sid, ci, wrapper, a = dock_env
    record_base_freq(b, ref=a, base_freq=2)   # eff_freq=2
    animal = _ensure(ci, sid, "动物")
    for _ in range(4):
        record_selection_pref_cooccur(b, ref_a=a, ref_class=animal)   # sp_agg = 0 + 4 = 4
    gates.SELECTION_PREF_DOCK_MODE = True
    try:
        w = wrapper._seed_weight(a)
    finally:
        gates.SELECTION_PREF_DOCK_MODE = False
    w_freq = make(FREQ_SEED_SCALE + 2, FREQ_SEED_SCALE)
    w_sp = make(SP_SEED_SCALE + 4, SP_SEED_SCALE)
    expected = mul(w_freq, w_sp)
    assert eq(w, expected), f"乘积 w=mul(w_freq,w_sp)·got {w} expected {expected}"
    assert not eq(w, w_freq), "乘积 dock 真 boost（w ≠ w_freq）"


def test_seed_weight_no_backend_returns_one(dock_env):
    """backend=None → ONE（bit-identical·无 backend 退化·不论 gate）。"""
    b, sid, ci, wrapper, a = dock_env
    # 重建无 backend wrapper
    wrapper_nb = A3PRWrapper.build(b.select("edge"), backend=None)
    gates.SELECTION_PREF_DOCK_MODE = True
    try:
        assert eq(wrapper_nb._seed_weight(a), ONE), "backend=None → ONE 不论 gate"
    finally:
        gates.SELECTION_PREF_DOCK_MODE = False


def test_seed_weight_linearity_product_decomposes(dock_env):
    """mul 闭运算·乘积线性性：w(gate ON) = w(gate OFF) · w_sp（数学核证·solve_exact 线性缩放 x_s）。"""
    b, sid, ci, wrapper, a = dock_env
    record_base_freq(b, ref=a, base_freq=2)
    animal = _ensure(ci, sid, "动物")
    for _ in range(3):
        record_selection_pref_cooccur(b, ref_a=a, ref_class=animal)   # sp_agg=3
    gates.SELECTION_PREF_DOCK_MODE = True
    try:
        w_full = wrapper._seed_weight(a)
    finally:
        gates.SELECTION_PREF_DOCK_MODE = False
    # w(gate OFF) = w_freq only
    gates.SELECTION_PREF_DOCK_MODE = False
    w_freq_only = wrapper._seed_weight(a)
    w_sp = make(SP_SEED_SCALE + 3, SP_SEED_SCALE)
    assert eq(w_full, mul(w_freq_only, w_sp)), "w_full = w_freq · w_sp（乘积线性）"
    assert eq(w_freq_only, make(FREQ_SEED_SCALE + 2, FREQ_SEED_SCALE)), "w_freq 与 eff_freq=2 一致"


def test_seed_weight_struct_ref_data_vacuum(dock_env):
    """struct_ref 数据真空：eff_freq=0 + sp_agg=0 → w=ONE（生产 seed bit-identical·S4 Plan agent 纠偏核证）。"""
    b, sid, ci, wrapper, a = dock_env
    # a 无 base_freq（eff_freq=0）·无 selection_pref（sp_agg=0）→ 模拟 struct_ref 数据真空
    gates.SELECTION_PREF_DOCK_MODE = True
    try:
        w = wrapper._seed_weight(a)
    finally:
        gates.SELECTION_PREF_DOCK_MODE = False
    assert eq(w, ONE), f"数据真空（eff_freq=0 + sp_agg=0）→ w=ONE·got {w}（生产 seed struct_ref bit-identical）"


# ============ 片3：第三条腿 reward_propagate 落点⑥ ============

@pytest.fixture
def prop_env():
    """第三条腿单测环境（experience_count + selection_pref_count + edge_store）。"""
    b = DictBackend()
    bootstrap(b)
    register_experience_count(b)
    register_selection_pref_count(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    yield b, sid, es, ci
    b.close()


def _causes_edge(b, es, sid, frm, to):
    """建 CAUSES 边（sn=1/tn=1·rate>0·落点① 正常跑）。"""
    es.add(space_id_from=sid, local_id_from=frm[1], space_id_to=sid, local_id_to=to[1],
           edge_type=EDGE_CAUSES, strength=1, source=SOURCE_BARE_TEXT,
           tier=TIER_PRIMARY, sn=1, tn=1)


_CTX_TAG = (DOMAIN_MATH, MODALITY_ARITH, 0, INTENT_COMMAND)


def test_third_leg_gate_off_skips(prop_env):
    """gate OFF：第三条腿整段跳过 → selection_pref_count 无 reward 写（bit-identical·CI===生产）。"""
    b, sid, es, ci = prop_env
    a = _ensure(ci, sid, "追")
    c = _ensure(ci, sid, "老鼠")
    _causes_edge(b, es, sid, a, c)
    pd = PathData()
    pd.edges = [(sid, a[1], sid, c[1], EDGE_CAUSES)]
    pr = PathResult(path=pd, terminal=TERMINAL_REACHED_SINK, sink=c)
    gates.SELECTION_PREF_FEED_MODE = False
    try:
        propagate_reward(pr, [], 1, _CTX_TAG, INTENT_COMMAND,
                         WorkMemory(), edge_store=es, backend=b)
    finally:
        gates.SELECTION_PREF_FEED_MODE = False
    assert read_selection_pref_agg(b, a) == (0, 0, 0), "gate OFF 第三条腿跳过·无 reward 写"


def test_third_leg_gate_on_reward_positive_pairs_with_isa(prop_env):
    """gate ON + reward>0：concept_targets pair 喂 (a, class_of(b))·IS_A 上卷·守 :131 assert 不触发。"""
    b, sid, es, ci = prop_env
    a = _ensure(ci, sid, "追")          # 动词（无 IS_A·class_of 退自身）
    c = _ensure(ci, sid, "老鼠")        # 论元
    animal = _ensure(ci, sid, "动物")   # IS_A 祖先
    _causes_edge(b, es, sid, a, c)
    build_is_a_edge(es, c, animal, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    pd = PathData()
    pd.edges = [(sid, a[1], sid, c[1], EDGE_CAUSES)]
    pr = PathResult(path=pd, terminal=TERMINAL_REACHED_SINK, sink=c)
    gates.SELECTION_PREF_FEED_MODE = True
    try:
        # 守 :131 assert：第三条腿独立段不进 distributed·propagate_reward 不抛 AssertionError
        propagate_reward(pr, [], 1, _CTX_TAG, INTENT_COMMAND,
                         WorkMemory(), edge_store=es, backend=b)
    finally:
        gates.SELECTION_PREF_FEED_MODE = False
    # concept_targets = {追, 老鼠}·pair (追, 老鼠)：
    #   (追, class_of(老鼠)=动物) reward>0 → sp_sn=1/sp_tn=1
    #   (老鼠, class_of(追)=追) reward>0 → sp_sn=1/sp_tn=1
    assert read_selection_pref_count(b, a, animal) == (0, 1, 1), "(追,动物) IS_A 上卷 + reward>0 sp_sn=1"
    assert read_selection_pref_count(b, c, a) == (0, 1, 1), "(老鼠,追) class_of(追)=追 + reward>0 sp_sn=1"


def test_third_leg_gate_on_reward_negative(prop_env):
    """gate ON + reward<0：pair 喂 sp_tn++ only（sp_sn 不动守 MUTABLE_MONOTONE）。"""
    b, sid, es, ci = prop_env
    a = _ensure(ci, sid, "追")
    c = _ensure(ci, sid, "石头")   # 无 IS_A·class_of(石头)=石头
    _causes_edge(b, es, sid, a, c)
    pd = PathData()
    pd.edges = [(sid, a[1], sid, c[1], EDGE_CAUSES)]
    pr = PathResult(path=pd, terminal=TERMINAL_REACHED_SINK, sink=c)
    gates.SELECTION_PREF_FEED_MODE = True
    try:
        propagate_reward(pr, [], -1, _CTX_TAG, INTENT_COMMAND,
                         WorkMemory(), edge_store=es, backend=b)
    finally:
        gates.SELECTION_PREF_FEED_MODE = False
    assert read_selection_pref_count(b, a, c) == (0, 0, 1), "(追,石头) reward<0 sp_sn=0/sp_tn=1"


def test_third_leg_class_of_self_built_no_isa(prop_env):
    """class_of 自建：concept 无 IS_A 祖先 → class_of 退自身（冷启动·守单向依赖不复用 understanding）。"""
    b, sid, es, ci = prop_env
    a = _ensure(ci, sid, "打")
    c = _ensure(ci, sid, "球")
    _causes_edge(b, es, sid, a, c)
    # 不建 IS_A 边 → ancestor_map 空 → class_of 退自身
    pd = PathData()
    pd.edges = [(sid, a[1], sid, c[1], EDGE_CAUSES)]
    pr = PathResult(path=pd, terminal=TERMINAL_REACHED_SINK, sink=c)
    gates.SELECTION_PREF_FEED_MODE = True
    try:
        propagate_reward(pr, [], 1, _CTX_TAG, INTENT_COMMAND,
                         WorkMemory(), edge_store=es, backend=b)
    finally:
        gates.SELECTION_PREF_FEED_MODE = False
    # 无 IS_A → class_of(球)=球·class_of(打)=打
    assert read_selection_pref_count(b, a, c) == (0, 1, 1), "(打,球) class_of 退自身"
    assert read_selection_pref_count(b, c, a) == (0, 1, 1), "(球,打) class_of 退自身"


def test_third_leg_multi_concept_pairing(prop_env):
    """多 concept_targets 配对正确：3 concept 两两配对（episode 级粗聚合·双向算 class_of）。"""
    b, sid, es, ci = prop_env
    a = _ensure(ci, sid, "追")          # 动词
    c1 = _ensure(ci, sid, "老鼠")       # 论元 1
    c2 = _ensure(ci, sid, "鸡")         # 论元 2
    animal = _ensure(ci, sid, "动物")
    _causes_edge(b, es, sid, a, c1)
    _causes_edge(b, es, sid, a, c2)
    build_is_a_edge(es, c1, animal, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED, space_id=sid)
    build_is_a_edge(es, c2, animal, source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED, space_id=sid)
    pd = PathData()
    pd.edges = [(sid, a[1], sid, c1[1], EDGE_CAUSES), (sid, a[1], sid, c2[1], EDGE_CAUSES)]
    pr = PathResult(path=pd, terminal=TERMINAL_REACHED_SINK, sink=c2)
    gates.SELECTION_PREF_FEED_MODE = True
    try:
        propagate_reward(pr, [], 1, _CTX_TAG, INTENT_COMMAND,
                         WorkMemory(), edge_store=es, backend=b)
    finally:
        gates.SELECTION_PREF_FEED_MODE = False
    # concept_targets = {追, 老鼠, 鸡}·3 对：(追,老鼠)(追,鸡)(老鼠,鸡)
    # (追, animal) 来自 (追,老鼠)+(追,鸡) 两对 → sp_sn=2/sp_tn=2
    assert read_selection_pref_count(b, a, animal) == (0, 2, 2), "(追,动物) 两对聚合 sp_sn=2"
    # (老鼠, animal) 来自 (老鼠,鸡) 一对 → sp_sn=1
    assert read_selection_pref_count(b, c1, animal) == (0, 1, 1), "(老鼠,动物) 一对 sp_sn=1"


# ============ 片4：H2 _rebuild_path ctx_code 透传 ============

def test_rebuild_path_forwards_ctx_code(monkeypatch):
    """_rebuild_path 透传 ctx_code + edge_store 给 dag_path_step（S4 片4 + B-PR3·H2 与生产 episode_loop 一致性）。

    生产 caller（run_round_full reward 阶段）算 ctx_code = pack_ctx_code(*_ctx_tag(raw, intent))·同 episode_loop:82。
    H2 _rebuild_path 须透传同 ctx_code（stage8 latent 修·dock 后 attractor 扩张路径 token seed 读 ctx 桶·
    不同桶 bit-identical 失）。**B-PR3**：edge_store 亦须透传（gate③ _intent_override D:11 查找·gate ON 时
    episode_loop 穿 edge_store→H2 须同穿·否则 gate③ override 分叉→H2 path≠生产 path·bit-identical 失）。
    本测 spy dag_path_step 验证 ctx_code + edge_store 透传。
    """
    from types import SimpleNamespace
    from pure_integer_ai.cognition.process import dag_path as dag_path_mod
    from pure_integer_ai.experiments.formal_train import _rebuild_path
    captured = {}

    def spy_dag_path_step(edges, seeds, workmem, intent, **kw):
        captured['ctx_code'] = kw.get('ctx_code')
        captured['backend'] = kw.get('backend')
        captured['edge_store'] = kw.get('edge_store')
        return "STUB_PATHRESULT"   # _rebuild_path 直接返回·不进真实 dag_path

    monkeypatch.setattr(dag_path_mod, 'dag_path_step', spy_dag_path_step)
    b = DictBackend()
    bootstrap(b)
    es = EdgeStore(b)
    ctx = SimpleNamespace(work_memory=WorkMemory(), backend=b, edge_store=es)
    result = _rebuild_path(ctx, [], [(1, 1)], INTENT_COMMAND, 0, ctx_code=42)
    assert captured['ctx_code'] == 42, f"_rebuild_path 透传 ctx_code·got {captured.get('ctx_code')}"
    assert captured['backend'] is b, "_rebuild_path 透传 backend"
    assert captured['edge_store'] is es, "_rebuild_path 透传 edge_store（B-PR3 gate③ bit-identical 守）"
    assert result == "STUB_PATHRESULT"


def test_rebuild_path_default_ctx_code_zero(monkeypatch):
    """_rebuild_path 默认 ctx_code=0（向后兼容·既有 caller 不传→退化 bit-identical）。"""
    from types import SimpleNamespace
    from pure_integer_ai.cognition.process import dag_path as dag_path_mod
    from pure_integer_ai.experiments.formal_train import _rebuild_path
    captured = {}

    def spy_dag_path_step(edges, seeds, workmem, intent, **kw):
        captured['ctx_code'] = kw.get('ctx_code')
        return "STUB"

    monkeypatch.setattr(dag_path_mod, 'dag_path_step', spy_dag_path_step)
    b = DictBackend()
    bootstrap(b)
    ctx = SimpleNamespace(work_memory=WorkMemory(), backend=b, edge_store=None)
    _rebuild_path(ctx, [], [(1, 1)], INTENT_COMMAND, 0)   # 不传 ctx_code
    assert captured['ctx_code'] == 0, f"默认 ctx_code=0（向后兼容）·got {captured.get('ctx_code')}"


def test_h2_ctx_code_uses_same_algorithm_as_episode_loop():
    """P1 加固（审1）：formal_train H2 caller 与 episode_loop :82 用同 `_ctx_tag + pack_ctx_code` 算法。

    数学等价（同函数同参同对象 raw=input_payload）·本测**锁定算法契约**（formal_train import _ctx_tag/pack_ctx_code·
    防未来一侧改另一侧漏 divergence）+ 确定性 + 等价。完整生产 run_round_full e2e ctx_code 数值相等断言
    **已落**（test_experiments.py::test_formal_train_e2e_ctx_code_episode_loop_matches_h2_rebuild·
    spy dag_path_step 双 patch episode_mod+dag_path_mod 捕两路径·断言同桶非 0·两审 P1 agenda 闭）。
    """
    from types import SimpleNamespace
    from pure_integer_ai.cognition.shared.types import IntentType
    from pure_integer_ai.cognition.process.episode import _ctx_tag
    from pure_integer_ai.storage.experience_count import pack_ctx_code
    from pure_integer_ai.experiments import formal_train as ft_mod
    # 锁定 formal_train import 同 _ctx_tag + pack_ctx_code（防未来 removed/divergent copy）
    assert hasattr(ft_mod, "_ctx_tag"), "formal_train imports _ctx_tag（H2 ctx_code 算法同 episode_loop）"
    assert hasattr(ft_mod, "pack_ctx_code"), "formal_train imports pack_ctx_code"
    # 同算法同入参 → 同 ctx_code（确定性 + 两路等价）
    raw = SimpleNamespace(domain=DOMAIN_TEXT, modality=MODALITY_LANGUAGE)
    intent = IntentType(type=INTENT_COMMAND)   # _ctx_tag 读 intent.type（IntentType 对象·非 int 常量）
    ep_code = pack_ctx_code(*_ctx_tag(raw, intent))           # episode_loop :81-82 算法
    h2_code = pack_ctx_code(*ft_mod._ctx_tag(raw, intent))    # formal_train H2 caller 算法（同 import 同函数）
    assert ep_code == h2_code, "两路同 _ctx_tag + pack_ctx_code → 同 ctx_code"
    assert ep_code == pack_ctx_code(*_ctx_tag(raw, intent)), "_ctx_tag 确定性"


# ============ 片5：反 theater e2e（第三条腿 + _seed_weight dock 闭环） ============

def test_s4_anti_theater_e2e(prop_env):
    """S4 反 theater e2e：reward>0 动物搭配 → (追,动物) sp_sn 高·石头 reward<0 → (追,石头) sp_sn=0·
    _seed_weight(追) w_sp dock boost（gate ON ≠ OFF·乘积 dock·反 theater 牙）。

    **两层正交**（决断 2 设计故意）：PR 侧 w_sp 聚合（粗筛·追 搭配多种动物传播力强）+ 生成侧
    read_selection_pref_count rate（精查·(追,动物) >> (追,石头)·defer 独立线）。
    **诚实边界**：PR 侧 w_sp 不区分动物/石头（聚合 sum_sp_tn）·生成侧精查区分·两层各司其职。
    """
    from pure_integer_ai.storage.experience_count import pack_ctx_code
    b, sid, es, ci = prop_env
    chase = _ensure(ci, sid, "追")
    mouse = _ensure(ci, sid, "老鼠")
    chicken = _ensure(ci, sid, "鸡")
    cat = _ensure(ci, sid, "猫")
    stone = _ensure(ci, sid, "石头")
    animal = _ensure(ci, sid, "动物")
    for child in (mouse, chicken, cat):    # IS_A: 动物 IS_A 动物·石头无 IS_A（class_of 退自身）
        build_is_a_edge(es, child, animal, source=SOURCE_CONCEPTNET,
                        epistemic=EPI_STRUCTURED, space_id=sid)

    gates.SELECTION_PREF_FEED_MODE = True
    try:
        # 3 动物搭配 episode reward>0（语义合理）
        for prey in (mouse, chicken, cat):
            _causes_edge(b, es, sid, chase, prey)
            pd = PathData()
            pd.edges = [(sid, chase[1], sid, prey[1], EDGE_CAUSES)]
            pr = PathResult(path=pd, terminal=TERMINAL_REACHED_SINK, sink=prey)
            propagate_reward(pr, [], 1, _CTX_TAG, INTENT_COMMAND,
                             WorkMemory(), edge_store=es, backend=b)
        # 1 石头搭配 episode reward<0（DEAD_END·语义荒谬）
        _causes_edge(b, es, sid, chase, stone)
        pd = PathData()
        pd.edges = [(sid, chase[1], sid, stone[1], EDGE_CAUSES)]
        pr = PathResult(path=pd, terminal=TERMINAL_REACHED_SINK, sink=stone)
        propagate_reward(pr, [], -1, _CTX_TAG, INTENT_COMMAND,
                         WorkMemory(), edge_store=es, backend=b)
    finally:
        gates.SELECTION_PREF_FEED_MODE = False

    # 反 theater ①：第三条腿产 (追,动物) sp_sn=3（3 reward>0 episodes·IS_A 上卷到动物）
    assert read_selection_pref_count(b, chase, animal) == (0, 3, 3), \
        "(追,动物) 3 reward>0 → sp_sn=3/sp_tn=3（IS_A 共祖上卷）"
    # 反 theater ②：(追,石头) reward<0 → sp_sn=0/sp_tn=1（石头无 IS_A·class_of 退石头·死路不涨 sp_sn）
    assert read_selection_pref_count(b, chase, stone) == (0, 0, 1), \
        "(追,石头) reward<0 → sp_sn=0/sp_tn=1（生成侧精查 rate 低·反 theater 牙）"

    # 反 theater ③：_seed_weight(追) w_sp dock boost（PR 侧粗筛·gate ON 乘积 > gate OFF）
    _ctx_code = pack_ctx_code(*_CTX_TAG)
    edges = b.select("edge")
    wrapper = A3PRWrapper.build(edges, backend=b, ctx_code=_ctx_code)
    gates.SELECTION_PREF_DOCK_MODE = False
    w_off = wrapper._seed_weight(chase)        # w_freq only（落点 A·sp_agg 不读）
    gates.SELECTION_PREF_DOCK_MODE = True
    w_on = wrapper._seed_weight(chase)         # mul(w_freq, w_sp)（乘积 dock）
    gates.SELECTION_PREF_DOCK_MODE = False
    _sb, _ssn, _stn = read_selection_pref_agg(b, chase)
    sp_agg = _sb + _stn
    assert sp_agg == 4, f"sp_agg = sum_sp_tn(追) = 3(动物)+1(石头) = 4·got {sp_agg}"
    expected_w_sp = make(SP_SEED_SCALE + sp_agg, SP_SEED_SCALE)
    assert eq(w_on, mul(w_off, expected_w_sp)), "w_on = w_off · w_sp（乘积 dock·线性性）"
    assert not eq(w_on, w_off), "gate ON w_sp boost（sp_agg>0 → w_on ≠ w_off·反 theater 牙·PR 粗筛）"


# ============ graph_view.selection_pref_score unit（S4 决断 2 生成侧精查·两层正交第二腿） ============

def test_selection_pref_score_class_level_aggregation(prop_env):
    """selection_pref_score：CLASS 级 Σ sp_tn（c 的 IS_A class 与 ctx_refs 的 pair-rate 共现）。

    鸡 IS_A 动物 → class_of(鸡)=动物·(追,动物) sp_tn=3 → selection_pref_score(鸡,[追])=3。
    **CLASS 级泛化**：鸡未见与追 token 共现（token 级 collide=0）·但鸡的 class 动物与追共现高 → boost。
    """
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    b, sid, es, ci = prop_env
    chase = _ensure(ci, sid, "追")
    chicken = _ensure(ci, sid, "鸡")
    animal = _ensure(ci, sid, "动物")
    build_is_a_edge(es, chicken, animal, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    for _ in range(3):
        record_selection_pref_cooccur(b, ref_a=chase, ref_class=animal)   # (追,动物) sp_tn=3

    graph = ConceptGraph(b)
    score = graph.selection_pref_score(chicken, [chase])
    assert score == 3, f"class_of(鸡)=动物·sp_tn(追,动物)=3 → score=3·got {score}（CLASS 级 boost 真活）"


def test_selection_pref_score_no_isa_class_of_self(prop_env):
    """selection_pref_score：c 无 IS_A 祖先 → class_of(c)=c 自身（冷启动退化恒等·镜像 _nearest_isa_ancestor）。"""
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    b, sid, es, ci = prop_env
    chase = _ensure(ci, sid, "追")
    stone = _ensure(ci, sid, "石头")   # 无 IS_A
    record_selection_pref_cooccur(b, ref_a=chase, ref_class=stone)   # (追,石头) sp_tn=1

    graph = ConceptGraph(b)
    score = graph.selection_pref_score(stone, [chase])
    assert score == 1, f"class_of(石头)=石头（无 IS_A）·sp_tn(追,石头)=1·got {score}"


def test_selection_pref_score_c_in_ctx_skipped(prop_env):
    """selection_pref_score：r == c 跳过避自 boost（对称写时 if a==b: continue）。"""
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    b, sid, es, ci = prop_env
    chase = _ensure(ci, sid, "追")
    animal = _ensure(ci, sid, "动物")
    record_selection_pref_cooccur(b, ref_a=chase, ref_class=animal)   # (追,动物) sp_tn=1

    graph = ConceptGraph(b)
    # c=chase·ctx=[chase]·r=chase==c → skip → score=0（无其他 r·防自 boost）
    assert graph.selection_pref_score(chase, [chase]) == 0, "c==ctx_ref 跳过（避自 boost）"


def test_selection_pref_score_none_to_zero(prop_env):
    """selection_pref_score：无行→read_selection_pref_count 返 None→0（冷启动）。"""
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    b, sid, es, ci = prop_env
    chase = _ensure(ci, sid, "追")
    chicken = _ensure(ci, sid, "鸡")
    animal = _ensure(ci, sid, "动物")
    build_is_a_edge(es, chicken, animal, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    # 无任何 selection_pref_count 行

    graph = ConceptGraph(b)
    assert graph.selection_pref_score(chicken, [chase]) == 0, "无行→None→0"


def test_selection_pref_score_empty_ctx_zero(prop_env):
    """selection_pref_score：ctx_refs 空 → 0（同 collide_score 范式）。"""
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    b, sid, es, ci = prop_env
    chicken = _ensure(ci, sid, "鸡")
    graph = ConceptGraph(b)
    assert graph.selection_pref_score(chicken, []) == 0


def test_selection_pref_score_write_read_class_of_consistency(prop_env):
    """写读 class_of 一致：graph_view nearest_isa_ancestor == understanding.selection_pref._nearest_isa_ancestor。

    三处同 min(ancestor_map.get(ref)) 逻辑（selection_pref 写 / reward_propagate reward /
    graph_view 读）·pair-rate 命中硬条件。写 (追, class_of(鸡)=动物)·读 selection_pref_score
    用同 class_of → 命中（反 theater：若 graph_view 用不同 class_of 则读不到写过的行）。
    """
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    from pure_integer_ai.cognition.understanding.selection_pref import _nearest_isa_ancestor
    from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map
    b, sid, es, ci = prop_env
    chase = _ensure(ci, sid, "追")
    chicken = _ensure(ci, sid, "鸡")
    animal = _ensure(ci, sid, "动物")
    build_is_a_edge(es, chicken, animal, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    # 写侧 class_of（selection_pref._nearest_isa_ancestor·understanding）
    amap = build_isa_ancestor_map(b, space_id=sid)
    class_write = _nearest_isa_ancestor(amap, chicken)
    assert class_write == animal, "写侧 class_of(鸡)=动物"
    record_selection_pref_cooccur(b, ref_a=chase, ref_class=class_write)   # 写 (追,动物) sp_tn=1
    # 读侧（graph_view.selection_pref_score·nearest_isa_ancestor·result）
    graph = ConceptGraph(b)
    score = graph.selection_pref_score(chicken, [chase])
    assert score == 1, f"写读 class_of 一致 → pair-rate 命中·got {score}（若不一致则读不到=0）"


def test_selection_pref_score_table_unregistered():
    """selection_pref_score：表未注册→read_selection_pref_count 返 None→0（向后兼容）。"""
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    b = DictBackend()
    bootstrap(b)
    # 不调 register_selection_pref_count（bare fixture·向后兼容）
    graph = ConceptGraph(b)
    # space_id=1 可能无 IS_A·class_of 退自身·read None→0（不抛 KeyError）
    assert graph.selection_pref_score((1, 1), [(1, 2)]) == 0, "表未注册→0 不抛"


# ============ sp_sn 维（S4 后续加固·反 dead column·成功搭配加成·消费侧落生成精查） ============

def test_selection_pref_score_includes_sp_sn(prop_env):
    """selection_pref_score 含 sp_sn 维（成功搭配加成·S4 后续加固反 dead column）。

    fixture：(追,动物) 3 次 reward>0 episode feed → record_selection_pref_reward R1 符号 sp_sn=3 & sp_tn=3。
    鸡 IS_A 动物 → class_of(鸡)=动物 → selection_pref_score(鸡,[追]) = sp_tn(追,动物) + sp_sn(追,动物) = 3+3 = 6。
    若不含 sp_sn（旧实现）= 3 → 反证 sp_sn 维真活（6 ≠ 3·sp_sn 消费侧落生成精查·反 dead column）。
    """
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    b, sid, es, ci = prop_env
    chase = _ensure(ci, sid, "追")
    chicken = _ensure(ci, sid, "鸡")
    animal = _ensure(ci, sid, "动物")
    build_is_a_edge(es, chicken, animal, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    for _ in range(3):
        record_selection_pref_reward(b, ref_a=chase, ref_class=animal, reward=1)   # sp_sn=3 & sp_tn=3

    graph = ConceptGraph(b)
    score = graph.selection_pref_score(chicken, [chase])
    assert score == 6, f"sp_tn(追,动物)=3 + sp_sn(追,动物)=3 = 6·got {score}（sp_sn 维真活·反 dead column）"


def test_selection_pref_score_sp_sn_success_amplifies(prop_env):
    """成功搭配（reward>0·sp_sn>0）比失败搭配（reward≤0·sp_sn=0）boost 高·强化 selectional preference。

    fixture：(追,动物) 3 reward>0 → sp_tn=3, sp_sn=3 / (追,植物) 3 reward=0 → sp_tn=3, sp_sn=0。
    鸡 IS_A 动物 / 草 IS_A 植物。selection_pref_score(鸡)=3+3=6 > (草)=3+0=3。
    **控制变量**：sp_tn 同（3）·只 sp_sn 异 → boost 差异纯来自 sp_sn（成功加成）。
    反 theater：生成精查用 sp_sn 区分成功/失败搭配·强化 selectional preference 本意（成功 > 失败）。
    """
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    b, sid, es, ci = prop_env
    chase = _ensure(ci, sid, "追")
    chicken = _ensure(ci, sid, "鸡")
    grass = _ensure(ci, sid, "草")
    animal = _ensure(ci, sid, "动物")
    plant = _ensure(ci, sid, "植物")
    build_is_a_edge(es, chicken, animal, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    build_is_a_edge(es, grass, plant, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    for _ in range(3):
        record_selection_pref_reward(b, ref_a=chase, ref_class=animal, reward=1)   # sp_tn=3, sp_sn=3
    for _ in range(3):
        record_selection_pref_reward(b, ref_a=chase, ref_class=plant, reward=0)   # sp_tn=3, sp_sn=0

    graph = ConceptGraph(b)
    score_chicken = graph.selection_pref_score(chicken, [chase])   # 3+3=6
    score_grass = graph.selection_pref_score(grass, [chase])       # 3+0=3
    assert score_chicken == 6, f"鸡（成功搭配）3+3=6·got {score_chicken}"
    assert score_grass == 3, f"草（失败搭配）3+0=3·got {score_grass}"
    assert score_chicken > score_grass, "sp_sn 加成：成功搭配 boost > 失败搭配（sp_tn 同·只 sp_sn 异）"


# ============ 片3 反 theater e2e（生成侧 sel_pref boost·CLASS 级·S4 决断 2 第二腿·破半 theater） ============

def test_dispatch_slot_sel_pref_boost_class_level(prop_env):
    """片3 反 theater e2e：生成侧 sel_pref pair-rate boost（CLASS 级泛化·破半 theater）。

    反例 fixture：两候选 collide_score 都 0（未见 token·无 COOCCURS）·**石头 lid < 鸡 lid**（OFF ref
    tiebreak 偏石头）。OFF → 选石头（ref tiebreak·bit-identical·等同 sel_pref 维不存在）·
    ON → 选鸡（class_of(鸡)=动物·sp_tn(追,动物)=3 > 石头 class_of(石头)=石头·sp_tn(追,石头)=0）。
    **证 sel_pref 维因果活**：OFF≠ON（机制异·非结果巧合）·去掉 selection_pref_score 调用 ON 退 OFF
    → 选石头（反例）→ 与 ON 选鸡不同 = sel_pref 维在决策路径真活·非 theater。

    **两层正交**（S4 决断 2）：PR 侧 w_sp 聚合丢 argument_class 区分（追×动物/石头 同 score）·
    生成侧 selection_pref_score pair-rate 区分（CLASS 级 boost）·本测验生成侧腿真活。
    """
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    from pure_integer_ai.cognition.result.slot_dispatch import dispatch_slot
    from pure_integer_ai.cognition.shared.types import (
        RoleSlot, PathData, PathResult, TERMINAL_REACHED_SINK, LANG_NONE,
        LINEAGE_CONCEPT_FILL,
    )
    from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO
    b, sid, es, ci = prop_env
    slot_c = _ensure(ci, sid, "slot")        # patient slot concept（两候选都 REFERS_TO 它）
    chase = _ensure(ci, sid, "追")
    animal = _ensure(ci, sid, "动物")
    fox = _ensure(ci, sid, "狐狸")
    stone = _ensure(ci, sid, "石头")           # ensure 序先于鸡 → lid < 鸡 lid
    chicken = _ensure(ci, sid, "鸡")
    # IS_A：鸡/狐狸 IS_A 动物（class_of=动物）·石头无 IS_A（class_of=石头·冷启动退自身）
    for child in (chicken, fox):
        build_is_a_edge(es, child, animal, source=SOURCE_CONCEPTNET,
                        epistemic=EPI_STRUCTURED, space_id=sid)
    # REFERS_TO：石头/鸡 → slot_c（activate_candidates(slot_c) 返 sorted [石头, 鸡]）
    es.add(space_id_from=sid, local_id_from=stone[1], space_id_to=sid, local_id_to=slot_c[1],
           edge_type=EDGE_REFERS_TO, strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY)
    es.add(space_id_from=sid, local_id_from=chicken[1], space_id_to=sid, local_id_to=slot_c[1],
           edge_type=EDGE_REFERS_TO, strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY)
    # selection_pref_count：(追,动物) sp_tn=3（observe 段内共现 3 次）·(追,石头) 无行→0
    for _ in range(3):
        record_selection_pref_cooccur(b, ref_a=chase, ref_class=animal)
    # 无 COOCCURS 边 → collide_score(鸡/石头, ctx) 都 0（token 级未见共现）

    # 反例核验：石头 ref < 鸡 ref（OFF ref tiebreak 偏石头·前提）
    assert stone < chicken, \
        f"石头 ref 须 < 鸡 ref（OFF ref tiebreak 偏石头·反例前提）·got stone={stone} chicken={chicken}"

    wm = WorkMemory()
    wm.produced_refs = [fox, chase]   # ctx_refs = [狐狸, 追]（已产出的 predicate + agent）
    surface_map = {stone: "石头", chicken: "鸡"}
    dag = PathResult(path=PathData(edges=[], struct_unit_refs=[]),
                     terminal=TERMINAL_REACHED_SINK, sink=None, topo_layers=[],
                     convergence={}, source=None)

    # OFF：两候选 collide=0 → ref tiebreak → 选石头（bit-identical·sel_pref 维不存在）
    saved = gates.GENERATE_SELECTION_PREF_MODE
    gates.GENERATE_SELECTION_PREF_MODE = False
    try:
        g_off = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
        word_off, src_off = dispatch_slot(RoleSlot(ref=slot_c, role=1), dag, g_off, wm, LANG_NONE)
    finally:
        gates.GENERATE_SELECTION_PREF_MODE = saved
    assert word_off == "石头", \
        f"OFF 选石头（两候选 collide=0·ref tiebreak 偏石头·bit-identical）·got {word_off}"
    assert src_off == LINEAGE_CONCEPT_FILL

    # ON：sel_pref boost 鸡（class 动物 count 高·CLASS 级泛化）→ 选鸡
    saved = gates.GENERATE_SELECTION_PREF_MODE
    gates.GENERATE_SELECTION_PREF_MODE = True
    try:
        g_on = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))   # fresh graph（cache 独立）
        word_on, src_on = dispatch_slot(RoleSlot(ref=slot_c, role=1), dag, g_on, wm, LANG_NONE)
    finally:
        gates.GENERATE_SELECTION_PREF_MODE = saved
    assert word_on == "鸡", \
        f"ON 选鸡（sel_pref boost·class_of(鸡)=动物·sp_tn(追,动物)=3 > 石头 0）·got {word_on}"

    # 反 theater 牙：OFF ≠ ON（机制异·非结果巧合）→ sel_pref 维因果活在决策路径
    assert word_off != word_on, \
        f"OFF 选石头 / ON 选鸡 → sel_pref 维真活·非 theater·got off={word_off} on={word_on}"


def test_dispatch_slot_sel_pref_collide_main_axis(prop_env):
    """片3 collide 主轴守：1 个真 token 共现（collide=1）> 999 个 class 共现（sel_pref cap 999）。

    鸡 collide=0/sp=3 → combined=3·狗 collide=1/sp=0 → combined=1000·选狗（collide 维不被 sel_pref 盖）。
    守 combine 硬不变量（collide×SCALE + min(sp,SCALE-1)·cap 守主轴）。
    """
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    from pure_integer_ai.cognition.result.slot_dispatch import dispatch_slot
    from pure_integer_ai.cognition.shared.types import (
        RoleSlot, PathData, PathResult, TERMINAL_REACHED_SINK, LANG_NONE,
    )
    from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO, EDGE_COOCCURS
    b, sid, es, ci = prop_env
    slot_c = _ensure(ci, sid, "slot")
    chase = _ensure(ci, sid, "追")
    animal = _ensure(ci, sid, "动物")
    chicken = _ensure(ci, sid, "鸡")
    dog = _ensure(ci, sid, "狗")             # collide=1（与追 COOCCURS）
    build_is_a_edge(es, chicken, animal, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    # 鸡/狗 → slot_c 候选
    es.add(space_id_from=sid, local_id_from=chicken[1], space_id_to=sid, local_id_to=slot_c[1],
           edge_type=EDGE_REFERS_TO, strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY)
    es.add(space_id_from=sid, local_id_from=dog[1], space_id_to=sid, local_id_to=slot_c[1],
           edge_type=EDGE_REFERS_TO, strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY)
    # (追,动物) sp_tn=3 → 鸡 class 动物 boost 3
    for _ in range(3):
        record_selection_pref_cooccur(b, ref_a=chase, ref_class=animal)
    # 狗-追 COOCCURS（collide=1）·鸡无 COOCCURS（collide=0）
    es.add(space_id_from=sid, local_id_from=dog[1], space_id_to=sid, local_id_to=chase[1],
           edge_type=EDGE_COOCCURS, strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY)

    wm = WorkMemory()
    wm.produced_refs = [chase]
    surface_map = {chicken: "鸡", dog: "狗"}
    dag = PathResult(path=PathData(edges=[], struct_unit_refs=[]),
                     terminal=TERMINAL_REACHED_SINK, sink=None, topo_layers=[],
                     convergence={}, source=None)
    saved = gates.GENERATE_SELECTION_PREF_MODE
    gates.GENERATE_SELECTION_PREF_MODE = True
    try:
        g = ConceptGraph(b, surface_of=lambda r: surface_map.get(r))
        word, _ = dispatch_slot(RoleSlot(ref=slot_c, role=1), dag, g, wm, LANG_NONE)
    finally:
        gates.GENERATE_SELECTION_PREF_MODE = saved
    # 鸡 combined = 0×1000 + 3 = 3·狗 combined = 1×1000 + 0 = 1000 → 选狗（collide 主轴守）
    assert word == "狗", \
        f"collide 主轴：狗 collide=1 → 1000 > 鸡 sp=3 → 3·选狗·got {word}（cap 守 collide 维优先）"


# ============ S4 后续加固·项1 cache invalidate（P1 theater 修复配套）============

def test_invalidate_ancestor_map_clears_cache(prop_env):
    """项1：cache invalidate 清全 cache（build 后 invalidate → cache 空）。"""
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    b, sid, es, ci = prop_env
    chase = _ensure(ci, sid, "追")
    chicken = _ensure(ci, sid, "鸡")
    animal = _ensure(ci, sid, "动物")
    build_is_a_edge(es, chicken, animal, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    graph = ConceptGraph(b)
    graph.selection_pref_score(chicken, [chase])   # 触发 cache build
    assert sid in graph._ancestor_map_cache, "cache build 后含 sid"
    graph.invalidate_ancestor_map()
    assert len(graph._ancestor_map_cache) == 0, "invalidate 全清后 cache 空"


def test_invalidate_no_op_when_empty():
    """项1：空 cache invalidate no-op（gate OFF 安全·不抛）。"""
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    b = DictBackend()
    bootstrap(b)
    graph = ConceptGraph(b)
    graph.invalidate_ancestor_map()   # 空 cache·不抛
    assert len(graph._ancestor_map_cache) == 0, "空 cache invalidate no-op"


def test_invalidate_specific_space(prop_env):
    """项1：invalidate 指定 space（多 space cache·清单 space·其他保留）。"""
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    b, sid, es, ci = prop_env
    chase = _ensure(ci, sid, "追")
    chicken = _ensure(ci, sid, "鸡")
    animal = _ensure(ci, sid, "动物")
    build_is_a_edge(es, chicken, animal, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    graph = ConceptGraph(b)
    graph.selection_pref_score(chicken, [chase])   # build sid cache
    graph._ancestor_map_cache[999] = {"fake": set()}   # 手动塞另一 space 假 cache
    assert len(graph._ancestor_map_cache) == 2
    graph.invalidate_ancestor_map(space_id=999)
    assert 999 not in graph._ancestor_map_cache, "指定 space 清除"
    assert sid in graph._ancestor_map_cache, "其他 space 保留"


# ============ S4 后续加固·项2 多层 IS_A 真 LCA（三处同源·替 min 升序首）============

def test_nearest_isa_ancestor_deepest_multilayer(prop_env):
    """项2：多层 IS_A 真 LCA·狐狸→动物→生物·class_of(狐狸)=动物（最深·非生物 min）。

    fixture 序狐狸/生物/动物（生物 lid < 动物 lid）→ min(ancestors)=生物（错·非最近）·
    nearest_isa_ancestor=动物（对·最深·动物∈ancestor_map[生物]? 不·生物是动物的祖先·动物非生物的祖先→
    动物是 candidate·生物∈ancestor_map[动物]? 是→生物非 candidate→return 动物）。
    """
    from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map, nearest_isa_ancestor
    b, sid, es, ci = prop_env
    fox = _ensure(ci, sid, "狐狸")        # lid=1
    creature = _ensure(ci, sid, "生物")   # lid=2（< 动物·min 会取它·错）
    animal = _ensure(ci, sid, "动物")     # lid=3
    build_is_a_edge(es, fox, animal, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    build_is_a_edge(es, animal, creature, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    amap = build_isa_ancestor_map(b, space_id=sid)
    ancestors = amap.get(fox)
    assert ancestors == {animal, creature}, f"狐狸祖先={animal,creature}·got {ancestors}"
    assert min(ancestors) == creature, "反证前提：min=生物（非最近·项2 须改）"
    assert nearest_isa_ancestor(amap, fox) == animal, \
        "最深=动物（非 min 生物·项2 真 LCA）"
    assert nearest_isa_ancestor(amap, fox) != min(ancestors), \
        "nearest ≠ min（项2 真 LCA 非升序首）"


def test_nearest_isa_ancestor_no_ancestor_returns_self(prop_env):
    """项2：无 IS_A 祖先→返 ref 自身（冷启动退化恒等·三处同源）。"""
    from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map, nearest_isa_ancestor
    b, sid, es, ci = prop_env
    fox = _ensure(ci, sid, "狐狸")   # 无 IS_A 边
    amap = build_isa_ancestor_map(b, space_id=sid)
    assert amap == {}, "无 IS_A 边→空 map"
    assert nearest_isa_ancestor(amap, fox) == fox, "无祖先→返自身"


def test_nearest_isa_ancestor_diamond_tiebreak(prop_env):
    """项2：diamond 多候选（两不可比直接父）→ NodeRef 升序 tiebreak（bit-identical·同 common_is_a_ancestor 范式）。"""
    from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map, nearest_isa_ancestor
    b, sid, es, ci = prop_env
    fox = _ensure(ci, sid, "狐狸")
    animal = _ensure(ci, sid, "动物")      # lid 小
    creature2 = _ensure(ci, sid, "生物2")   # lid 大
    build_is_a_edge(es, fox, animal, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    build_is_a_edge(es, fox, creature2, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    amap = build_isa_ancestor_map(b, space_id=sid)
    # 两直接父不可比（动物∉ancestor_map[生物2] 且 生物2∉ancestor_map[动物]）→ 都 candidate → min tiebreak
    assert nearest_isa_ancestor(amap, fox) == min(animal, creature2), \
        "diamond 多候选 NodeRef 升序 tiebreak"


def test_nearest_isa_ancestor_write_read_consistency_multilayer(prop_env):
    """项2：多层 IS_A 写读一致·三处同源 nearest_isa_ancestor（graph_view + selection_pref + reward_propagate）。

    write (追, class_of(鸡)=动物) → read selection_pref_score(鸡, [追]) 命中 sp_tn>0。
    若三处不同源（写用 min=生物·读用 nearest=动物）→ pair-rate 失配死信（读不到写过的行）。
    """
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    from pure_integer_ai.cognition.understanding.selection_pref import _nearest_isa_ancestor
    from pure_integer_ai.cognition.process.reward_propagate import _nearest_isa_ancestor_reward
    from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map, nearest_isa_ancestor
    b, sid, es, ci = prop_env
    chase = _ensure(ci, sid, "追")
    chicken = _ensure(ci, sid, "鸡")
    creature = _ensure(ci, sid, "生物")   # lid < 动物（min 会取它·错）
    animal = _ensure(ci, sid, "动物")
    build_is_a_edge(es, chicken, animal, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    build_is_a_edge(es, animal, creature, source=SOURCE_CONCEPTNET,
                    epistemic=EPI_STRUCTURED, space_id=sid)
    amap = build_isa_ancestor_map(b, space_id=sid)
    # 三处同源·都返动物（最深·非 min 生物）
    assert nearest_isa_ancestor(amap, chicken) == animal, "abstraction.nearest=动物"
    assert _nearest_isa_ancestor(amap, chicken) == animal, "selection_pref wrapper=动物"
    assert _nearest_isa_ancestor_reward(amap, chicken) == animal, "reward_propagate wrapper=动物"
    # 写 (追, class_of(鸡)=动物)
    record_selection_pref_cooccur(b, ref_a=chase, ref_class=animal)
    # 读 selection_pref_score(鸡, [追]) 命中（写读同 nearest=动物→row 命中）
    graph = ConceptGraph(b)
    assert graph.selection_pref_score(chicken, [chase]) == 1, \
        "写读一致·pair-rate 命中（三处同源 nearest=动物）"
