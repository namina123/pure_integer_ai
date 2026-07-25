"""tests/test_w0_reward_active — W0 激活 formal_train 生产 reward 闭环测试（断奶训练首步）。

验（doc/重来_断奶阶段训练设计_2026-07-11.md W0）：
1. gate OFF（TRAINING_MODE 不翻）→ observe-only·零 episode·conduction_rate=0（bit-identical）
2. gate ON（TRAINING_MODE 翻）→ 算术域 stage3 reward>0 真流·conduction_rate>0（非 scratch·非 flat_floors 放水 reward）
3. 反 theater：gate ON conduction_rate > gate OFF conduction_rate（真激活·非 theater）
4. 诚实：weaning_ready=False + weaning_blockers 非空（D3/D4/D5/E2 永 False·theatrical·E2 真墙 #493）

**核心反"学不会"**：gate ON 算术域 reward>0 真流过 formal_train 生产路径（_run_verify_round vm_proof
all-pass → reward=1 → conduction_rate>0）·非 scratch 诊断脚本·非 flat_floors 放水 reward。
"""
from __future__ import annotations

from pure_integer_ai.training.stages import STAGE3_REWARD
from pure_integer_ai.experiments.run_weaning_train import run_weaning_arith


def test_w0_gate_off_observe_only(tmp_path):
    """gate OFF：TRAINING_MODE 不翻 → eff_stage=STAGE2 observe-only → 零 episode·conduction_rate=0（bit-identical）。

    formal_train.py:1947 eff_stage=stage if reward_active else STAGE2_CAUSES_ABS·TRAINING_MODE OFF →
    reward_active=False → eff_stage=STAGE2 → run_round_full:368 stage<STAGE3 返空 → 零 episode。
    """
    result = run_weaning_arith(rounds_per_stage=1, training_mode=False,
                               flat_floors=True, run_dir=str(tmp_path / "off"))
    m = result.final_metrics
    # StageMetrics 无 episode_count/reward_pos（在 RoundMetrics·metrics.py:54）·用 conduction_rate==0 证零 reward 信号
    assert m.conduction_rate == 0, \
        f"gate OFF conduction_rate=0（零 reward 信号·observe-only·bit-identical）·got {m.conduction_rate}"


def test_w0_gate_on_arith_reward_positive(tmp_path):
    """★gate ON：TRAINING_MODE 翻 → 算术域 stage3 _run_verify_round vm_proof all-pass → reward>0 真流。

    conduction_rate>0 + reward_pos>0（非 scratch 诊断脚本·非 flat_floors 放水 reward·vm_proof 真验执行值）。
    反"学不会"：reward>0 真流过 formal_train 生产路径。
    """
    result = run_weaning_arith(rounds_per_stage=1, training_mode=True,
                               flat_floors=True, run_dir=str(tmp_path / "on"))
    m = result.final_metrics
    assert STAGE3_REWARD in result.stages_completed, \
        f"gate ON 须跑到 STAGE3 reward（非 observe-only 降级）·got {result.stages_completed}"
    # StageMetrics 无 reward_pos/episode_count（在 RoundMetrics·metrics.py:54）·用 conduction_rate>0 证 reward>0 真流
    # conduction_rate = reward>0 episode 占比 ×1000（metrics.py:148）·>0 即算术域 vm_proof reward>0 真流过生产路径
    assert m.conduction_rate > 0, \
        f"算术域 reward>0 真流·conduction_rate>0（reward>0 episode 占比）·got {m.conduction_rate}"


def test_w0_gate_on_gt_gate_off(tmp_path):
    """反 theater：gate ON conduction_rate > gate OFF conduction_rate（真激活·非 theater）。

    gate OFF observe-only 零 reward·gate ON 算术域 vm_proof reward>0·差分证真激活非 theater。
    """
    off = run_weaning_arith(rounds_per_stage=1, training_mode=False,
                            flat_floors=True, run_dir=str(tmp_path / "off2"))
    on = run_weaning_arith(rounds_per_stage=1, training_mode=True,
                           flat_floors=True, run_dir=str(tmp_path / "on2"))
    assert on.final_metrics.conduction_rate > off.final_metrics.conduction_rate, \
        (f"gate ON conduction_rate({on.final_metrics.conduction_rate}) > "
         f"gate OFF({off.final_metrics.conduction_rate})（反 theater·真激活 reward 闭环）")


def test_w0_weaning_theatrical_honest(tmp_path):
    """诚实：weaning_ready=False + weaning_blockers 非空（D3/D4/D5/E2 永 False·theatrical·E2 真墙 #493）。

    W0 不 claim 断奶 PASS·weaning_ready=False（E2 执行条件依赖真训练 run·can_wean 永墙 #493）。
    weaning_blockers 诚实标注未过闸门（D3 裁判同源 / D4 无 probe_set / D5 无台账 / E2 永墙）。
    """
    result = run_weaning_arith(rounds_per_stage=1, training_mode=True,
                               flat_floors=True, run_dir=str(tmp_path / "wean"))
    assert result.weaning_ready is False, \
        "weaning_ready=False（E2 永墙 #493·诚实 theatrical·不伪造真断奶）"
    assert result.weaning_blockers, \
        f"weaning_blockers 非空（D3/D4/D5/E2 未过闸门诚实标注·不静默）·got {result.weaning_blockers}"
    # 审2 catch：显式断言 E2 in blockers（防 E2 被从 blockers 列表移除的退化·E2 永墙 #493 隔离验证）
    assert "E2_independent_production" in result.weaning_blockers, \
        f"weaning_blockers 须含 E2_independent_production（最硬闸门·真墙 #493）·got {result.weaning_blockers}"
