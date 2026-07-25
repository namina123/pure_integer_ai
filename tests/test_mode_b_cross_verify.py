"""Mode B POST-weaning 异算法 cross-verify 测试（#479 加强腿·统计学一致）。

片1 unit：cross_verify_pair 纯机制 5 路径（AGREE / DISAGREE / 空探针 / None-vacate / KeyError-vacate）。
片4 e2e：反 theater（OFF=0 / ON_agree=1 / ON_disagree=0 三态因果活·见文件后段）。

哲学定位（doc/重来_ModeB自洽设计补充.md §七 + mode_b_cross_verify.py docstring）：不追求 correctness 真墙·
只求统计学内一致。Mechanism Y（execute_composes_value 双路取值 + rational.eq·无 oracle·POST 可用）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_MATH
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
from pure_integer_ai.training.mode_b_cross_verify import cross_verify_pair, CrossVerifyResult
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.formal_train import make_train_context, DefaultRoundRunner
from pure_integer_ai.training.stages import STAGE3_REWARD
from pure_integer_ai.cognition.shared.types import (
    CodeSpec, MODALITY_ARITH, MODALITY_CODE, DOMAIN_MATH, DOMAIN_CODE, LANG_NONE, WEANING_POST,
)
from pure_integer_ai.storage.edge_store import SOURCE_MATH, SOURCE_CODE


# ---- e2e fixture：造带参树（arith_source_b）的算术 CollectedItem ----

def _arith_item_with_peer(src: str, src_b: str, specs) -> CollectedItem:
    """造算术域 CollectedItem + 参树 DSL（arith_source_b·异 shape·同函数第二表达）。"""
    return CollectedItem(
        modality=MODALITY_ARITH,
        domain=DOMAIN_MATH,
        lang=LANG_NONE,
        source=SOURCE_MATH,
        arith_source=src,
        arith_source_b=src_b,
        arith_specs=tuple(specs),
    )


def _code_item_with_peer(src: str, src_b: str, specs) -> CollectedItem:
    """造代码域 CollectedItem + 参树 Python 源码（code_source_b·异 shape·同函数第二表达·§施工序 1.2）。

    CODE 域 build_composes_from_source 支持 BinOp Add/Sub/Mult + Name + Constant + Return（doc §3.1）·
    fixture 用 n+n vs n*2（ADD vs MUL·同值异 shape）作 AGREE · n*2 vs n*3 作 DISAGREE。
    """
    return CollectedItem(
        modality=MODALITY_CODE,
        domain=DOMAIN_CODE,
        lang=LANG_NONE,
        source=SOURCE_CODE,
        code_source=src,
        code_source_b=src_b,
        code_specs=tuple(specs),
    )


# ---- fixture helper：建两棵独立 COMPOSES 树（同 space·异 root·镜像 test_stage9:542-559 范式） ----

def _build_pair(dsl_a: str, dsl_b: str | None = None):
    """建 backend + space + 两棵独立 COMPOSES 树（root_a / root_b）。

    dsl_b=None 时 root_b 不建 COMPOSES（plain concept·execute_composes_value 返 None·供 None-vacate 测）。
    返 (graph, root_a, root_b)。
    """
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    g = ConceptGraph(b)
    sid = sp.space_id
    root_a = ci.ensure("__seg_a", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith(dsl_a, concept_index=ci, edge_store=es,
                              backend=b, space_id=sid, source=SOURCE_MATH, root_ref=root_a)
    root_b = ci.ensure("__seg_b", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    if dsl_b is not None:
        build_composes_from_arith(dsl_b, concept_index=ci, edge_store=es,
                                  backend=b, space_id=sid, source=SOURCE_MATH, root_ref=root_b)
    return g, root_a, root_b


# ============ 片1 unit：cross_verify_pair 纯机制 5 路径 ============

def test_bit_identical_defaults():
    """片2 bit-identical 守：CollectedItem() 默认 arith_source_b/code_source_b is None + gate 默认 OFF。

    双 False 短路保证 POST 路径 reward=0 与现状 bit-identical（既有 CollectedItem 全无参树字段）。
    """
    ci = CollectedItem()
    assert ci.arith_source_b is None   # 默认 None·既有 CollectedItem 零影响
    assert ci.code_source_b is None    # 默认 None·CODE 域对称（§施工序 1.2）
    assert gates.MODE_B_CROSS_VERIFY_MODE is False   # 默认 OFF·守回归


def test_cross_verify_pair_agree():
    """AGREE：Sigma(1,n,i) 迭代 vs n*(n+1)/2 闭式·两探针 {5,10}·双路执行值相等→all_agree。

    异 shape（CTRL_WHILE 迭代 vs 直线 BinOp MUL+DIV）→ 异 builder 代码路径 → R6 真守。
    n=5→15 / n=10→55·两路都==→ n_valid=2 n_agree=2 all_agree=True。
    """
    g, root_a, root_b = _build_pair("lambda n: Sigma(1, n, i)", "lambda n: n * (n + 1) / 2")
    cv = cross_verify_pair(g, root_a, root_b, ((5,), (10,)))
    assert cv.n_probes == 2
    assert cv.n_valid == 2
    assert cv.n_agree == 2
    assert cv.all_agree is True


def test_cross_verify_pair_disagree():
    """DISAGREE：Sigma(1,n,i) 三角数 vs n*n 平方·n=5→15 vs 25·异 shape 异值→all_agree=False。

    n_valid=1（两路都执行成功）·n_agree=0（值不等）·all_agree=False。证机制真比对非 theater（异值真 veto）。
    """
    g, root_a, root_b = _build_pair("lambda n: Sigma(1, n, i)", "lambda n: n * n")
    cv = cross_verify_pair(g, root_a, root_b, ((5,),))
    assert cv.n_probes == 1
    assert cv.n_valid == 1
    assert cv.n_agree == 0
    assert cv.all_agree is False


def test_cross_verify_pair_empty_probes():
    """空探针：probes=()→n_probes=0 n_valid=0 all_agree=False（诚实 no-op·非 vacuous agree）。"""
    g, root_a, root_b = _build_pair("lambda n: Sigma(1, n, i)", "lambda n: n * (n + 1) / 2")
    cv = cross_verify_pair(g, root_a, root_b, ())
    assert cv.n_probes == 0
    assert cv.n_valid == 0
    assert cv.n_agree == 0
    assert cv.all_agree is False


def test_cross_verify_pair_none_vacate():
    """None-vacate：root_b 非 COMPOSES 根（plain concept·不 build）→ execute_composes_value 返 None
    → 该探针 vacate（不计 valid 不计 agree）→ n_valid=0 all_agree=False。

    镜像 vm_proof.py:77-78 root 非 COMPOSES 根→None 语义。证 None 诚实传播非 theater。
    """
    g, root_a, root_b = _build_pair("lambda n: Sigma(1, n, i)", dsl_b=None)   # root_b plain concept
    cv = cross_verify_pair(g, root_a, root_b, ((5,), (10,)))
    assert cv.n_probes == 2
    assert cv.n_valid == 0   # root_b 全 None → 全 vacate
    assert cv.n_agree == 0
    assert cv.all_agree is False


def test_cross_verify_pair_keyerror_vacate():
    """KeyError-vacate：root_b arity2（lambda x,y:x+y）·探针 (5,) 只绑 mv0→root_b LOAD mv1 unbound→KeyError
    → except 捕获→vb=None→vacate。root_a arity1 正常执行→va=value·但 vb=None→vacate→n_valid=0。

    证 arity 不匹配 KeyError 不外泄·诚实 vacate（漏洞 2 修·同 StepLimit→None 语义）。
    """
    g, root_a, root_b = _build_pair("lambda n: n + 1", "lambda x, y: x + y")
    cv = cross_verify_pair(g, root_a, root_b, ((5,),))
    assert cv.n_probes == 1
    assert cv.n_valid == 0   # root_b KeyError → vb=None → vacate
    assert cv.n_agree == 0
    assert cv.all_agree is False


# ============ 片4 e2e：反 theater（OFF=0 / ON_agree=1 / ON_disagree=0 三态因果活） ============
# 经 run_round_full MODALITY_ARITH + WEANING_POST·走 _run_verify_round POST 分支（formal_train:446-）。
# 三态全成立 → 证机制因果活非 theater（OFF 短路 + ON 真比对 + 异值真 veto）。

def test_post_cross_verify_off_bit_identical(monkeypatch):
    """OFF=0：gate OFF + arith_source_b 在→POST 短路 reward=0（与现状 bit-identical·双 False 短路第一项）。"""
    monkeypatch.setattr(gates, 'MODE_B_CROSS_VERIFY_MODE', False)
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST
    r = DefaultRoundRunner()
    item = _arith_item_with_peer("lambda n: Sigma(1, n, i)", "lambda n: n * (n + 1) / 2",
                                 [CodeSpec((5,), (15, 1)), CodeSpec((10,), (55, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 0          # gate OFF → 短路 → reward=0（bit-identical）
    assert res.episode.judge_G5_active is False


def test_post_cross_verify_on_agree(monkeypatch):
    """ON_agree=1：gate ON + AGREE（Sigma 迭代 vs n*(n+1)/2 闭式·异 shape）+ 探针 {5,10}→两路执行值
    都==15/55→all_agree→reward=1·g5_active=True（cross-verify 承重门 active）。"""
    monkeypatch.setattr(gates, 'MODE_B_CROSS_VERIFY_MODE', True)
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST
    r = DefaultRoundRunner()
    item = _arith_item_with_peer("lambda n: Sigma(1, n, i)", "lambda n: n * (n + 1) / 2",
                                 [CodeSpec((5,), (15, 1)), CodeSpec((10,), (55, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1          # 两路 agree → reward=1
    assert res.episode.judge_G5_active is True


def test_post_cross_verify_on_disagree(monkeypatch):
    """ON_disagree=0：gate ON + DISAGREE（Sigma 三角数 vs n*n 平方·异 shape 异值）+ 探针 {5}→15 vs 25
    不等→all_agree=False→reward=0·g5_active=True（cross-verify 真比对·异值真 veto·非 theater）。"""
    monkeypatch.setattr(gates, 'MODE_B_CROSS_VERIFY_MODE', True)
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST
    r = DefaultRoundRunner()
    item = _arith_item_with_peer("lambda n: Sigma(1, n, i)", "lambda n: n * n",
                                 [CodeSpec((5,), (15, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 0          # 15 vs 25 不等 → reward=0（异值真 veto）
    assert res.episode.judge_G5_active is True


def test_post_cross_verify_idempotent_rerun(monkeypatch):
    """漏洞 4 守门：同 item 跑两次（root_b 已建 COMPOSES 出边）→ query_from 守门 skip 重 build →
    第二次 reward 仍==1（不 corrupt 树·不重复加边·幂等）。"""
    monkeypatch.setattr(gates, 'MODE_B_CROSS_VERIFY_MODE', True)
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST
    r = DefaultRoundRunner()
    item = _arith_item_with_peer("lambda n: Sigma(1, n, i)", "lambda n: n * (n + 1) / 2",
                                 [CodeSpec((5,), (15, 1))])
    res1 = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    res2 = r.run_round_full(ctx, item, STAGE3_REWARD, 1)   # 同 root_b·query_from 守门 skip 重 build
    assert res1.episode.reward == 1
    assert res2.episode.reward == 1   # 幂等·不 corrupt


# ============ CODE 域 e2e：模态对称（§施工序 1.2·OFF=0 / ON_agree=1 / ON_disagree=0 / 幂等 四态） ============
# 经 run_round_full MODALITY_CODE + WEANING_POST·走 _run_verify_round POST 分支按模态分流（CODE 分支）。
# 证 corpus-agnostic：execute_composes_value + rational.eq 模态无关·CODE 域同函数异 shape 也获 reward=1。

def test_post_cross_verify_code_off_bit_identical(monkeypatch):
    """CODE OFF=0：gate OFF + code_source_b 在→POST 短路 reward=0（与现状 bit-identical·双 False 短路第一项·CODE 域对称）。"""
    monkeypatch.setattr(gates, 'MODE_B_CROSS_VERIFY_MODE', False)
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST
    r = DefaultRoundRunner()
    item = _code_item_with_peer("def f(n): return n + n", "def f(n): return n * 2",
                                [CodeSpec((5,), (10, 1)), CodeSpec((10,), (20, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 0          # gate OFF → 短路 → reward=0（bit-identical）
    assert res.episode.judge_G5_active is False


def test_post_cross_verify_code_on_agree(monkeypatch):
    """CODE ON_agree=1：gate ON + AGREE（n+n vs n*2·ADD vs MUL 异 shape·同函数 f(n)=2n）+ 探针 {5,10}→
    两路执行值都==10/20→all_agree→reward=1·g5_active=True（cross-verify 承重门 active·CODE 域对称 ARITH）。"""
    monkeypatch.setattr(gates, 'MODE_B_CROSS_VERIFY_MODE', True)
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST
    r = DefaultRoundRunner()
    item = _code_item_with_peer("def f(n): return n + n", "def f(n): return n * 2",
                                [CodeSpec((5,), (10, 1)), CodeSpec((10,), (20, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1          # 两路 agree → reward=1
    assert res.episode.judge_G5_active is True


def test_post_cross_verify_code_on_disagree(monkeypatch):
    """CODE ON_disagree=0：gate ON + DISAGREE（n*2 vs n*3·异 shape 异值）+ 探针 {5}→10 vs 15 不等→
    all_agree=False→reward=0·g5_active=True（cross-verify 真比对·异值真 veto·CODE 域对称·非 theater）。"""
    monkeypatch.setattr(gates, 'MODE_B_CROSS_VERIFY_MODE', True)
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST
    r = DefaultRoundRunner()
    item = _code_item_with_peer("def f(n): return n * 2", "def f(n): return n * 3",
                                [CodeSpec((5,), (10, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 0          # 10 vs 15 不等 → reward=0（异值真 veto）
    assert res.episode.judge_G5_active is True


def test_post_cross_verify_code_idempotent_rerun(monkeypatch):
    """CODE 漏洞 4 守门：同 item 跑两次（root_b 已建 COMPOSES 出边）→ query_from 守门 skip 重 build →
    第二次 reward 仍==1（不 corrupt 树·幂等·CODE 域对称 ARITH test_post_cross_verify_idempotent_rerun）。"""
    monkeypatch.setattr(gates, 'MODE_B_CROSS_VERIFY_MODE', True)
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST
    r = DefaultRoundRunner()
    item = _code_item_with_peer("def f(n): return n + n", "def f(n): return n * 2",
                                [CodeSpec((5,), (10, 1))])
    res1 = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    res2 = r.run_round_full(ctx, item, STAGE3_REWARD, 1)   # 同 root_b·query_from 守门 skip 重 build
    assert res1.episode.reward == 1
    assert res2.episode.reward == 1   # 幂等·不 corrupt
