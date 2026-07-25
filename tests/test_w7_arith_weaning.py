"""W7 算术机制闸与 V-00 来源隔离纠偏回归。

验（doc/重来_断奶阶段训练设计_2026-07-11.md W7）：
1. e2e：全旧 flag 开仍缺严格来源 ledger，必须由 D4 阻塞，不能再宣称 can_ween。
2. bit-identical：默认全 OFF → weaning_ready=False + blockers 全（7）·同 W6 既有行为零翻
3. 干预测试：关各 flag → 相应闸门不过（反 theater·每闸门真依赖机制非伪造）
4. 断点6 机制活：per-round series 8 entries·末4 verify flat → plateau True

**W7 = 7 断点修（Explore 5 + Plan agent 深挖 2）**：
- 断点1 retention backfill（eval 在 record_round 后·series baked 0·回填真值）
- 断点2 arith_bad D2（既有坏 corpus 产 veto·非扩算子族）
- 断点3 D3 路径 B（judge_source_independent_arith W3 建生产零调用·W7 接）
- 断点4 intervention/dependency 诚实标（teacher=None 架构事实·非 vacuous）
- 断点5 judge_self count_g5 口径（verify round G5_active 是 verify 门标志非 judge 排除）
- 断点6 per-round series（weaning.py:173 设计 per-run·实现 per-stage observe-only 0 混入 bug）
- 断点7 oov floor override（算术域 COMPOSES 直接 PRIMARY·FLOOR_OOV_PROMOTE 不适用）

**纠偏背景**：W7 讨论中 Claude 四轮窄化（浮点墙→eval→数值对拍→符号主线贬数值）·三路对抗（实证/设计忠诚/
反老毛病）综合→算术域多维共评（生产真活 vm_proof+structure_discover+op_confidence / opt-in Mode B cross_verify /
纸面 inline+L3 生产不 fire）·W7=六闸门接通非"自主符号关系"（AGENT.md D7）。

**V-00 纠偏**：旧 fixture 的 retention 恒 1000 只证明同族样本自洽；没有 provenance 独立性，
不能作为断奶 D4。其余算术机制测量继续保留。
"""
from __future__ import annotations

import pure_integer_ai.experiments.formal_train as _ft
from pure_integer_ai.experiments.run_weaning_train import run_weaning_arith


def _full_flags(tmp_path, **overrides):
    """全 flag 开 convenience（W7 can_ween=True 的完整 stack）。"""
    kw = dict(mix_bad_corpus=True, simulate_offline_eval=True, probe_holdout=2,
              weaning_round_series=True, calibrate_mode_b=True,
              run_dir=str(tmp_path / "w7"))
    kw.update(overrides)
    return run_weaning_arith(**kw)


def test_w7_legacy_holdout_cannot_ween_after_v00_audit(tmp_path):
    """全旧 flag 仍缺严格来源 ledger，不能再宣称算术域可断奶。"""
    r = _full_flags(tmp_path)
    assert r.weaning_ready is False
    assert r.weaning_blockers == ["D4_probe_set_disjoint"], (
        f"旧 W7 只应被严格来源隔离阻塞·got {r.weaning_blockers}")
    assert r.evaluation_strictly_isolated is False
    assert r.holdout_retention > 0, "holdout_retention 真值（eval 采·断点1 backfill 进 series）"


def test_w7_bit_identical_default_off(tmp_path):
    """★bit-identical：默认全 OFF → weaning_ready=False + 七闸门 blocker·同 W6 既有行为零翻。"""
    r = run_weaning_arith(run_dir=str(tmp_path / "w7bit"))   # 默认全 OFF
    assert r.weaning_ready is False, "默认 OFF 须 weaning_ready=False（bit-identical）"
    assert r.final_metrics.conduction_rate > 0, "W0 reward 闭环须不受 W7 影响"
    for b in ["D1_capability_plateau", "D1_retention_stable", "D2_neg_pathway_active",
              "D3_judge_source_independent", "D4_probe_set_disjoint",
              "D5_mode_b_prevalidated", "E2_independent_production"]:
        assert b in r.weaning_blockers, f"默认 OFF 须 {b} blocker·got {r.weaning_blockers}"


def test_w7_intervention_mix_bad_d2(tmp_path):
    """★干预：mix_bad_corpus=False → D2 不过（veto 来源 arith_bad·反 theater 真机制）。"""
    r = _full_flags(tmp_path, mix_bad_corpus=False)
    assert r.weaning_ready is False
    assert "D2_neg_pathway_active" in r.weaning_blockers, (
        "mix_bad=False→D2 须 blocker（arith_bad 产 veto·关掉则 square 全 reward=1 veto=0）")


def test_w7_intervention_simulate_eval(tmp_path):
    """★干预：simulate_offline_eval=False → D3+retention+E2 不过（路径 B/backfill/eval 三依赖此 flag）。"""
    r = _full_flags(tmp_path, simulate_offline_eval=False)
    assert r.weaning_ready is False
    assert "D3_judge_source_independent" in r.weaning_blockers, "simulate=False→D3 路径 B 不接"
    assert "D1_retention_stable" in r.weaning_blockers, "simulate=False→retention backfill 不调"
    assert "E2_independent_production" in r.weaning_blockers, "simulate=False→E2 eval 不跑"


def test_w7_intervention_round_series_plateau(tmp_path):
    """★干预：weaning_round_series=False → D1 plateau 不过（per-stage observe-only 0 混入·断点6 bug 复现）。"""
    r = _full_flags(tmp_path, weaning_round_series=False)
    assert r.weaning_ready is False
    assert "D1_capability_plateau" in r.weaning_blockers, (
        "round_series=False→D1 plateau 须 blocker（per-stage [0,0,X,X] 假跳变）")


def test_w7_intervention_calibrate_d5(tmp_path):
    """★干预：calibrate_mode_b=False → D5 不过（calibration 台账空）。"""
    r = _full_flags(tmp_path, calibrate_mode_b=False)
    assert r.weaning_ready is False
    assert "D5_mode_b_prevalidated" in r.weaning_blockers


def test_w7_intervention_probe_d4_e2(tmp_path):
    """★干预：probe_holdout=0 → D4+E2 不过（held-out 探针隔离 + eval 候选缺）。"""
    r = _full_flags(tmp_path, probe_holdout=0)
    assert r.weaning_ready is False
    assert "D4_probe_set_disjoint" in r.weaning_blockers
    assert "E2_independent_production" in r.weaning_blockers


def test_w7_per_round_series_8_entries(tmp_path):
    """★断点6 机制活：weaning_round_series=True → series 8 entries（per-round·4 stages×2 rounds）。
    末4 全 stage3/4 verify 同值 flat → plateau True（解 per-stage observe-only stages 产 0 混入 bug）。"""
    series_lens: list[int] = []
    orig = _ft.weaning_check

    def spy(history, **kw):
        series_lens.append(len(history))
        return orig(history, **kw)

    _ft.weaning_check = spy
    try:
        _full_flags(tmp_path)
    finally:
        _ft.weaning_check = orig
    assert series_lens and series_lens[0] == 8, (
        f"per-round series 须 8 entries（4 stages × rounds_per_stage=2）·got {series_lens}")
