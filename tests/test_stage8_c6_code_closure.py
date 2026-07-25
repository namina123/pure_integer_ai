"""Stage 8 验收门测试：C6 生产闭环 — 代码域训练管线 + vm_proof_fn 生产接线（#361 下游）。

覆盖（doc/重来_VM图灵完备与C6设计补充.md §4.5 + doc/重来_A3_代码域observe设计补充.md §二致命#2）：
  e2e 在 DefaultRoundRunner.run_round_full 层（非 vm_proof 单测——单测 test_stage7 L4 已覆盖）。
  - 涌现：code item → observe 建 COMPOSES → vm_proof_fn 验执行 vs spec → reward>0
  - 多 spec all-pass / any-fail
  - 反 theater 锚点：deadloop→0 / mismatch→0 / Mode B POST→0（证 reward 来自真 VM 执行 vs spec·非 stub）
  - observe-only 阶段 / len<2 绕过 / 无 spec 诚实跳过 / H2 排 code / pre_flight 兼容

核心架构决断：代码域独立 episode 路径（vm_proof_fn 直调绕 judge/generate/propagate）——
judge G5 路由物理上不成立（G2p 在 not reached_sink veto + J-sum 恒 0·代码域无词生成/无 key_skeleton/无 CAUSES）。
铁律：纯整数 / 确定性 bit-identical / fail-loud / 依赖单向向下 / 反 theater（reward>0 测试配 reward=0 锚点）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_CODE
from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
from pure_integer_ai.cognition.shared.types import (
    CodeSpec, MODALITY_CODE, DOMAIN_CODE, LANG_NONE,
    TERMINAL_REACHED_SINK, WEANING_PRE, WEANING_POST,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.formal_train import (
    make_train_context, DefaultRoundRunner, _h2_calibrate,
)
from pure_integer_ai.training.stages import STAGE1_SKELETON, STAGE2_CAUSES_ABS, STAGE3_REWARD


# ---- 代码语料 ----

_SUM_CODE = "def f(n):\n  acc=0\n  while n>0:\n    acc+=n\n    n+=-1\n  return acc"
_DEADLOOP_CODE = "def f(x):\n  while x > 0:\n    x = x"


def _code_item(code: str, specs, *, source: int = SOURCE_CODE) -> CollectedItem:
    """造代码域 CollectedItem（一段一函数 + 多测试用例 spec）。"""
    return CollectedItem(
        modality=MODALITY_CODE,
        domain=DOMAIN_CODE,
        lang=LANG_NONE,
        source=source,
        code_source=code,
        code_specs=tuple(specs),
    )


# ============ 涌现 + 反 theater 锚点 ============

def test_code_round_sum_emerges_reward_positive_e2e():
    """涌现门：code item（sum 函数 + spec f(5)=15）→ reward>0·G5 承重·ref=root·sink=root。

    证 C6 生产闭环 PRE 通：observe 建 COMPOSES → vm_proof_fn 执行 vs 独立 spec → reward>0 涌现。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _code_item(_SUM_CODE, [CodeSpec((5,), (15, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1, "sum f(5)=15 须 reward=1（vm_proof 验证通过）"
    assert res.episode.terminal == TERMINAL_REACHED_SINK
    assert res.episode.judge_G5_active is True   # PRE 有 spec·G5 承重
    assert res.episode.judge_veto_count == 0
    # dag_path.sink=root=struct_ref=COMPOSES 根（vm_proof_fn 定位用）
    assert res.dag_path is not None and res.dag_path.sink == res.episode.ref


def test_code_round_multi_spec_all_pass_e2e():
    """多 spec all-pass：f(5)=15 + f(10)=55 → reward=1。"""
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _code_item(_SUM_CODE, [CodeSpec((5,), (15, 1)), CodeSpec((10,), (55, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1


def test_code_round_multi_spec_any_fail_e2e():
    """反 theater 锚点①：多 spec any-fail（f(10)=999 错）→ reward=0·veto_count=1。"""
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _code_item(_SUM_CODE, [CodeSpec((5,), (15, 1)), CodeSpec((10,), (999, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 0, "任一 spec fail→reward=0（反 theater·不全员 pass）"
    assert res.episode.judge_veto_count == 1
    assert res.episode.vetoed is True


def test_code_round_mismatch_reward_zero_e2e():
    """反 theater 锚点②：mismatch（spec f(5)=14 错·真值 15）→ reward=0。

    证 reward 来自真 VM 执行 vs spec·非 stub 返 1。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _code_item(_SUM_CODE, [CodeSpec((5,), (14, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 0, "mismatch 须 reward=0（VM 执行 15≠spec 14）"


def test_code_round_deadloop_reward_zero_pre_e2e():
    """反 theater 锚点③：deadloop（while x>0: x=x·x 不变）→ StepLimit→None→reward=0（PRE 非 vacate）。

    R1 PRE：deadloop(None)→reward=0 诚实（不验过不给 reward）·非 POST vacate=1。
    证伪"死循环也 pass"theater。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _code_item(_DEADLOOP_CODE, [CodeSpec((1,), (0, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 0, "deadloop→None→PRE reward=0（非 vacate=1·反 theater）"
    assert res.episode.judge_G5_active is True   # PRE 仍 active（fail=veto）


def test_code_round_mode_b_post_reward_zero_no_vacuous_e2e():
    """反 theater 核心锚点：Mode B POST → reward=0·G5 不 active·不调 vm_proof（防 vacuous reward=1）。

    self_proof_check(POST, None)→1 vacate·若 POST 调 vm_proof 则 deadloop→None→vacate=1=vacuous reward。
    本路径 POST 不调 vm_proof·reward=0 诚实（Mode B re-derivation defer·无独立源）。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST   # 断奶后
    r = DefaultRoundRunner()
    # 用 sum 函数（PRE 本会 pass）·POST 须仍 reward=0（非 vacuous=1）
    item = _code_item(_SUM_CODE, [CodeSpec((5,), (15, 1))])
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 0, "POST 须 reward=0（不产 vacuous reward=1·Mode B defer）"
    assert res.episode.judge_G5_active is False


# ============ 阶段 / 路由 / 诚实跳过 ============

def test_code_round_observe_only_stage_no_episode():
    """observe-only 阶段（STAGE2）：无 episode·EDGE_COMPOSES 边已落盘。"""
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _code_item(_SUM_CODE, [CodeSpec((5,), (15, 1))])
    ep = r.run_round(ctx, item, STAGE2_CAUSES_ABS, 0)
    assert ep is None   # observe-only·无 episode
    # COMPOSES 边真落盘（observe MODALITY_CODE gate 建树）
    composes = b.select("edge", where={"edge_type": EDGE_COMPOSES})
    assert len(composes) >= 2


def test_code_round_single_segment_bypasses_len2_gate():
    """len<2 绕过：单 code 段→episode 产出（code 路径不查 len<2）·单语言段→None（对照）。

    formal_train.py:267 `len(struct_refs)<2` 是语言路径约束·code 单函数=1 struct_ref·
    vm_proof 直读 sink=root 不需 seed≠sink·路由分支在 len<2 之前接管。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    # 单 code 段→episode 产出
    code_res = r.run_round_full(ctx, _code_item(_SUM_CODE, [CodeSpec((5,), (15, 1))]),
                                 STAGE3_REWARD, 0)
    assert code_res.episode is not None
    # 单语言段（1 句）→None（语言路径 len<2 跳过·对照）
    lang_res = r.run_round_full(ctx, CollectedItem(tokens=["单句。"]),
                                STAGE3_REWARD, 0)
    assert lang_res.episode is None


def test_code_round_no_spec_honest_skip():
    """无 spec 诚实跳过：code_source 有但 code_specs=() → RoundResult()（无 episode·不伪造 reward=0）。

    无 spec 不能验证·诚实 observe-only·不产 reward=0 episode（区别于 mismatch 的 reward=0）。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _code_item(_SUM_CODE, [])   # 无 spec
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is None   # 无 spec→不验证→无 episode（诚实·非 reward=0）
    # observe 仍建了 COMPOSES 树
    assert len(b.select("edge", where={"edge_type": EDGE_COMPOSES})) >= 2


def test_code_round_no_code_source_honest_skip():
    """无 code_source：MODALITY_CODE 段无源码→_split 返空→RoundResult()（observe 无可建）。"""
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = CollectedItem(modality=MODALITY_CODE, domain=DOMAIN_CODE, lang=LANG_NONE,
                         source=SOURCE_CODE, code_source=None,
                         code_specs=(CodeSpec((1,), (1, 1)),))
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is None


# ============ H2 排 code ============

def test_h2_calibrate_skips_code_items():
    """H2 标定排 code：code-only corpus → _h2_calibrate 跳 code → 无 sample → 返 ctx.weights 原对象。

    code reward 经 vm_proof_fn 不用 JudgeWeights·且 code item 进 language judge()→G2p veto reward=0
    对齐 GT=1=垃圾标定污染 JudgeWeights。跳过 = `if not samples: return ctx.weights` 路径。
    """
    b = DictBackend()
    ctx = make_train_context(b)   # teacher=None
    r = DefaultRoundRunner()
    code_corpus = [_code_item(_SUM_CODE, [CodeSpec((5,), (15, 1))])]
    result = _h2_calibrate(ctx, code_corpus, r)
    # code 全跳过→samples 空→返 ctx.weights 原对象（若未跳·code sample 进 calibrate→返新 weights）
    assert result is ctx.weights


# ============ pre_flight / metrics 兼容 ============

def test_pre_flight_code_corpus_compatible():
    """pre_flight 兼容：code corpus → has_pos=True·metrics_signal=True·无崩·anti_collapse verified==0。

    code episode pr_vector={} → _anti_collapse_summary 跳过（空 pr_vector·诚实·code 不参与防塌验收）。
    """
    from pure_integer_ai.experiments.formal_train import pre_flight
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_code_item(_SUM_CODE, [CodeSpec((5,), (15, 1)), CodeSpec((10,), (55, 1))])]
    rep = pre_flight(ctx, corpus, rounds=2)
    # ① metrics_signal（graph_size>0·COMPOSES 节点）
    assert rep.metrics_signal is True
    assert rep.detail["graph_size"] > 0
    # ③ reward_gate_ok（has_pos：code reward=1>0）
    assert rep.reward_gate_ok is True
    assert rep.detail["has_pos_reward"] is True
    # anti_collapse：code pr_vector 空→verified==0（诚实·非崩）
    assert rep.detail["anti_collapse"]["verified"] == 0


# ============ 确定性 bit-identical ============

def test_code_round_bit_identical():
    """同 code item 两跑 bit-identical（reward/ref/terminal 一致）。"""
    b1 = DictBackend()
    ctx1 = make_train_context(b1)
    r = DefaultRoundRunner()
    item = _code_item(_SUM_CODE, [CodeSpec((5,), (15, 1)), CodeSpec((10,), (55, 1))])
    res1 = r.run_round_full(ctx1, item, STAGE3_REWARD, 0)
    b2 = DictBackend()
    ctx2 = make_train_context(b2)
    res2 = r.run_round_full(ctx2, item, STAGE3_REWARD, 0)
    assert res1.episode.reward == res2.episode.reward
    assert res1.episode.ref == res2.episode.ref
    assert res1.episode.terminal == res2.episode.terminal
    assert res1.dag_path.sink == res2.dag_path.sink
