"""P0 #1041 测试：统计层产出度量构造（构造①reader + 构造②reward truthiness 校准）。

承接 doc/重来_全局缺口重审_2026-07-14.md §5.1/§6 + doc/重来_统计层产出度量_设计_2026-07-14.md。
review-2 钉死：judge slot_fill_rate `if w:` 只判非空字符串 → 真词/__seg_* 同分 → reward 信号假（判据②③）。
#1040 已让 OutputPart.token_refs 携段真 token concept 序（gate OFF 空）·#1041 读它作 truthiness 解药。

机制（Path C 之后的产出度量腿）：
  TC1 output_word_ratio：Σ token_refs / Σ words ×1000（empty 0 / full 1000 / partial / repeat-safe / 多 part 聚合）。
  TC3 judge J4word gate ON truthiness 区分：同结构（同 words）真词 output reward > label output reward
     （review-2 钉死的解药·判据②③信号质量）。
  TC4 judge J4word gate OFF bit-identical：reward 不随 token_refs 变（gate OFF→J4word=0 主守·逐字现状）。
  TC5 JudgeWeights.w4 默认 1 + 向后兼容（JudgeWeights(1,1,1)→w4=1）+ H2 calibrate 不动 w4（空样本→w4=1）。
  TC6 metrics verified 真词感知 → **defer Phase2**（record_generate_round 服务 code+language·code 无 token_refs·
     token-aware verified 误杀 code 计数·须 modality-aware capability_exam·judge J4word 已 modality-safe 落）。

铁律：纯整数（ratio/reward/w4 全整·w4 进 assert_no_float/assert_int 守）/ 确定性 bit-identical（gate OFF 逐字现状）/
  反 theater（J4word 读 token_refs 真信号·非 truthiness 占位）。全统计层（判据②③信号质量·统计度量腿）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.cognition.shared.types import (
    OutputResult, OutputPart, InputPayload, IntentType,
    JudgeWeights, PathData, PathResult,
    STAGE_TRAINING, INTENT_QUESTION, DOMAIN_TEXT,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.judge import judge
from pure_integer_ai.cognition.result.output_measure import output_word_ratio
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.training.oracle import calibrate_weights


@pytest.fixture(autouse=True)
def _gate_reset():
    """每测前后复位 OUTPUT_WORD_REWARD_MODE（守测试隔离·防跨测泄漏）。"""
    saved = gates.OUTPUT_WORD_REWARD_MODE
    gates.OUTPUT_WORD_REWARD_MODE = False
    yield
    gates.OUTPUT_WORD_REWARD_MODE = saved


U = (1, 100)          # struct_ref
T0, T1, T2 = (2, 10), (2, 11), (2, 12)   # token concept refs


# ---- TC1 output_word_ratio（纯整·empty/full/partial/repeat-safe/多 part） ----

def test_tc1a_output_word_ratio_empty_token_refs_is_zero():
    """token_refs 全空（gate OFF 现状）→ ratio=0（bit-identical 次守）。"""
    out = OutputResult(parts=[OutputPart(unit=U, words=["a", "b", "c"])])
    assert output_word_ratio(out) == 0


def test_tc1b_output_word_ratio_full_coverage_is_thousand():
    """token_refs 与 words 等长（全真 token）→ ratio=1000。"""
    out = OutputResult(parts=[OutputPart(unit=U, words=["猫", "吃", "鱼"],
                                         token_refs=[T0, T1, T2])])
    assert output_word_ratio(out) == 1000


def test_tc1c_output_word_ratio_partial_coverage():
    """token_refs < words（部分槽退 unit 未入 token_refs·LOW-2 守）→ ratio=666（2/3×1000）。"""
    out = OutputResult(parts=[OutputPart(unit=U, words=["a", "b", "c"],
                                         token_refs=[T0, T1])])
    assert output_word_ratio(out) == 666


def test_tc1d_output_word_ratio_repeat_safe():
    """repeat-safe（#1040 重复 token 同 concept ref·每 position 一 token_ref）→ 全计不 dedup。"""
    out = OutputResult(parts=[OutputPart(unit=U, words=["的", "猫", "的", "鱼"],
                                         token_refs=[T0, T1, T0, T2])])
    assert output_word_ratio(out) == 1000   # 4 token_refs / 4 words


def test_tc1e_output_word_ratio_multi_part_aggregation():
    """多 part 聚合：part1 全真（3/3）+ part2 无（0/2）→ Σ=3 / Σ=5 → 600。"""
    out = OutputResult(parts=[
        OutputPart(unit=U, words=["a", "b", "c"], token_refs=[T0, T1, T2]),
        OutputPart(unit=(1, 200), words=["d", "e"]),    # 无 token_refs（gate OFF 段）
    ])
    assert output_word_ratio(out) == 600


def test_tc1f_output_word_ratio_no_words_is_zero():
    """空 output（无 words·冷启动）→ 0（防除零·不报假信号）。"""
    assert output_word_ratio(OutputResult()) == 0


def test_tc1g_output_word_ratio_clamps_to_thousand_when_invariant_broken():
    """防御性 clamp（LOW-1·纵深防御）：直接构造 token_refs>words 破不变量 → ratio clamp 1000（非 3000）。

    生产不变量 token_refs≤words（generate emitted_tokens 守）→ ratio 天然≤1000·clamp 守直接构造边界。
    """
    out = OutputResult(parts=[OutputPart(unit=U, words=["a"], token_refs=[T0, T1, T2])])
    assert output_word_ratio(out) == 1000, "clamp 守 token_refs>words 边界（防 J4word>1000 过度主导 reward）"


# ---- TC3+TC4 judge J4word（gate ON 区分 / gate OFF bit-identical） ----

def _judge_inputs(token_refs):
    """建 judge 输入（同结构·只 token_refs 变·DOMAIN_TEXT + INTENT_QUESTION 非因果→G3a/G3b/G5 skip）。"""
    out = OutputResult(parts=[OutputPart(unit=U, words=["猫", "吃", "鱼"],
                                         token_refs=list(token_refs))],
                       reached_sink=True)
    dag = PathResult(path=PathData(edges=[], struct_unit_refs=[U]),
                     topo_layers=[[U]], convergence={}, source=U, sink=U)
    inp = InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING,
                       domain=DOMAIN_TEXT,
                       intent=IntentType(type=INTENT_QUESTION, sink=U,
                                         is_causal_reasoning=False))
    g = ConceptGraph(DictBackend())
    wm = WorkMemory()
    return out, dag, inp, g, wm


def test_tc3_judge_j4word_gate_on_distinguishes_real_from_label():
    """**核心（review-2 钉死解药）**：gate ON·同结构（同 words）·真词 reward > label reward。

    J4word 读 token_refs 真信号 → reward 反映产出真词质量（判据②③信号质量·非 truthiness）。
    weights=(1,1,1,1)：reward = J2s(1000) + J4word（real=1000 / label=0）→ real 2000 > label 1000。
    """
    gates.OUTPUT_WORD_REWARD_MODE = True
    weights = JudgeWeights(1, 1, 1, 1)
    out_real, dag, inp, g, wm = _judge_inputs([T0, T1, T2])
    out_label, *_ = _judge_inputs([])           # 同结构·token_refs 空（label/__seg_* 现状）
    reward_real, gm_real = judge(out_real, dag, inp, g, weights, wm)
    reward_label, gm_label = judge(out_label, dag, inp, g, weights, wm)
    # truthiness 旧病：两 output 同 words → slot_fill_rate 同分。J4word 解药：真词 reward > label。
    assert reward_real > reward_label, "gate ON J4word 区分真词（review-2 钉死解药）"
    assert reward_real == 2000, "real: J2s(1000) + J4word(1000)"
    assert reward_label == 1000, "label: J2s(1000) + J4word(0)"
    assert not gm_real.G2p_vetoed and not gm_real.G4_vetoed   # reward>0 非否决


def test_tc4_judge_j4word_gate_off_bit_identical_insensitive():
    """gate OFF bit-identical：reward 不随 token_refs 变（J4word=0 主守→逐字现状）。

    两 output 同 words·异 token_refs → gate OFF reward 等同（J1/J2s/J3path 不读 token_refs·J4word=0）。
    """
    weights = JudgeWeights(1, 1, 1, 1)
    out_real, dag, inp, g, wm = _judge_inputs([T0, T1, T2])
    out_label, *_ = _judge_inputs([])
    reward_real, _ = judge(out_real, dag, inp, g, weights, wm)
    reward_label, _ = judge(out_label, dag, inp, g, weights, wm)
    assert reward_real == reward_label, "gate OFF reward 不随 token_refs 变（bit-identical 主守）"
    assert reward_real == 1000, "gate OFF: J2s(1000) + J4word(0)"


def test_tc4b_judge_j4word_gate_off_equals_w4_zero_formula():
    """gate OFF 等价于 w4=0 公式（双守核证）：同 output gate-ON-w4=0 reward == gate-OFF-w4=1 reward。

    证 gate 是 J4word 主守（非 w4）：gate OFF+w4=1 ⟺ gate ON+w4=0（都 J4word 贡献 0）。
    """
    out_real, dag, inp, g, wm = _judge_inputs([T0, T1, T2])
    gates.OUTPUT_WORD_REWARD_MODE = True
    r_w4_zero, _ = judge(out_real, dag, inp, g, JudgeWeights(1, 1, 1, 0), wm)   # gate ON w4=0
    gates.OUTPUT_WORD_REWARD_MODE = False
    r_w4_one, _ = judge(out_real, dag, inp, g, JudgeWeights(1, 1, 1, 1), wm)    # gate OFF w4=1
    assert r_w4_zero == r_w4_one, "gate OFF+w4=1 ⟺ gate ON+w4=0（双守·gate 主守非 w4）"


# ---- TC5 JudgeWeights.w4 默认 + H2 calibrate 不动 ----

def test_tc5a_weights_w4_default_one_and_backward_compat():
    """w4 默认 1·JudgeWeights(1,1,1) 向后兼容（既有构造零改·w4 落默认）。"""
    assert JudgeWeights().w4 == 1
    assert JudgeWeights(1, 1, 1).w4 == 1           # 既有位置构造（oracle.calibrate_weights 范式）
    assert JudgeWeights(w1=2, w2=3, w3=5).w4 == 1  # 既有 keyword 构造


def test_tc5b_calibrate_weights_leaves_w4_default():
    """H2 calibrate 不动 w4（oracle 网格搜 w1/w2/w3·`JudgeWeights(w1=,w2=,w3=)`→w4 落默认 1）。

    空样本 → 返 JudgeWeights() → w4=1。证标定结果不含 w4（bit-identical：weights 不变）。
    """
    result = calibrate_weights([], lambda s, **kw: (0, None), lambda s: 1)
    assert isinstance(result, JudgeWeights)
    assert result.w4 == 1, "calibrate 不动 w4（网格 w1/w2/w3·w4 落默认）"


def test_tc5c_judge_weights_w4_float_rejected_by_pure_integer_guard():
    """w4 进纯整数守（MEDIUM-1·review 钉死）：JudgeWeights(w4=1.5) → judge assert_no_float 抛（守铁律）。

    证 w4 与 w1/w2/w3 同守纯整数（防浮点 w4 经 w4·j4word 渗 reward 通道·reviewer-2 指出 float_w4*0=0.0
    即使 gate OFF 也破纯整数）。
    """
    out_real, dag, inp, g, wm = _judge_inputs([T0, T1, T2])
    with pytest.raises(AssertionError):
        judge(out_real, dag, inp, g, JudgeWeights(1, 1, 1, 1.5), wm)   # float w4 → assert_no_float 抛


# ---- TC6 metrics verified 真词感知 → defer Phase2（modality-aware） ----
# 构造① metrics 真词感知 defer：record_generate_round 同时服务 code+language task-driven·code generate
# 无 token_refs（语言 dispatch 才填）·token-aware verified 误杀 code 计数（formal_train flip 后 code 行
# generate_verified 归零·test_task_driven_metrics_downstream_reader 验）·须 modality-aware（capability_exam
# 知域·Phase2 接）。judge J4word（TC3/TC4）modality-safe（code token_refs 空→J4word=0 不扰 code reward）已落。
