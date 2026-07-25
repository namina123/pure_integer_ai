"""tests/test_w2_mode_b_self_anchor — W2 算术域 Mode B cross-verify 自锚（E2 第三条件算术域就位）。

验（doc/重来_断奶阶段训练设计_2026-07-11.md W2）：
1. POST cross-verify reward>0（formal_train 全路径 POST·cross_verify_pair 两路独立 execute + rational.eq·
   all_agree·conduction_rate>0·episode reward==1 + judge_G5_active==True）
2. produced_without_teacher_anchor_arith 算术域第三条件就位（VM 执行值自锚非教师锚·e2_execution_ready 仍 False）
3. weaning_phase=PRE（默认）→ POST 短路 bit-identical（W0/W1 既有·gate 恢复 OFF·conduction_rate>0 PRE vm_proof）
4. POST 也绕 episode_loop·experience_count 空（β_arith 不染 POST cross-verify reward）

**W2 = §7.6 defer 的"formal_train 全路径 POST 跑"**（doc/重来_ModeB自洽设计补充.md §7.6 done cross_verify_pair
+ test_mode_b_cross_verify 三态 run_round_full 层·formal_train POST 跑 defer）。W2 mock weaning_phase=WEANING_POST
+ 翻 MODE_B_CROSS_VERIFY_MODE + 填 arith_source_b（Sigma(1,{p},{p}) 迭代 vs {p}*{p} 闭式·square n² 异 shape·R6 真守）
→ _run_verify_round POST 分支（formal_train.py:578）→ cross_verify_pair all_agree → reward=1 → E2 第三条件算术域就位。

**诚实边界**：mock POST 非真断奶（weaning_ready 仍 False·非 rep.ready 切换·E2 整体仍 False·teacher_offline
defer W6 / probe_input_novel defer W4·W2 只验第三条件算术域就位·非 E2 过）。DISAGREE 反 theater 既有
test_mode_b_cross_verify:206 覆盖（run_round_full 三态）·W2 不重造。
"""
from __future__ import annotations

from pure_integer_ai.experiments.run_weaning_train import run_weaning_arith
from pure_integer_ai.cognition.shared.types import WEANING_PRE, WEANING_POST
from pure_integer_ai.storage.experience_count import EXPERIENCE_COUNT_TABLE


def test_w2_post_cross_verify_reward_positive(tmp_path):
    """★POST cross-verify reward>0（formal_train 全路径 POST·all_agree·conduction_rate>0）。

    weaning_phase=POST → _run_verify_round 走 POST 分支（:578·cross_verify_pair 两路独立
    execute_composes_value + rational.eq）→ all_agree（Sigma(1,n,n) vs n*n·square n² 异 shape·R6 真守）
    → reward=1 → conduction_rate>0（POST reward 真流过生产路径·非 scratch·非 flat_floors 放水）。
    collect_episodes=True 验存在 cross-verify agree episode（reward==1 + judge_G5_active==True·
    cross-verify 承重门 active·非 gate OFF 短路 :624 g5_active=False）。
    """
    result, _backend = run_weaning_arith(
        rounds_per_stage=1, training_mode=True, flat_floors=True,
        run_dir=str(tmp_path / "post"), return_backend=True,
        weaning_phase=WEANING_POST)
    # POST cross-verify reward>0 真流（conduction_rate = reward>0 episode 占比 ×1000）
    assert result.final_metrics.conduction_rate > 0, (
        f"POST cross-verify 须 conduction_rate>0（all_agree→reward=1 真流过生产路径）"
        f"·got {result.final_metrics.conduction_rate}")
    # collect_episodes=True 验存在 cross-verify agree episode（reward==1 + judge_G5_active==True）
    assert result.episodes, "collect_episodes=True（W2 POST）须收集 episode"
    cv_agree_eps = [e for e in result.episodes if e.reward == 1 and e.judge_G5_active]
    assert cv_agree_eps, (
        "须存在 POST cross-verify agree episode（reward==1 + judge_G5_active==True·"
        "cross-verify 承重门 active·非 gate OFF 短路 g5_active=False）")


def test_w2_produced_without_teacher_anchor_arith_true():
    """★produced_without_teacher_anchor_arith 算术域第三条件就位（VM 执行值自锚非教师锚）。

    判据：cross_verify_ran（gate ON + source_b·:597 双条件）+ cv_all_agree（两路独立执行一致）。
    W2 只建判定 + 算术域能 True·e2_independent_production 仍 False（teacher_offline defer W6 /
    probe_input_novel defer W4）·e2_execution_ready() 仍 False·can_wean 永 False。W7 才接全 E2。
    """
    from pure_integer_ai.teacher.weaning_e2 import (
        produced_without_teacher_anchor_arith, e2_execution_ready, e2_independent_production)
    # 算术域 POST cross-verify 就位：cross_verify_ran=True + cv_all_agree=True → 第三条件 True
    assert produced_without_teacher_anchor_arith(cross_verify_ran=True, cv_all_agree=True) is True
    # 反 theater：cross_verify 未跑（gate OFF/source_b 缺短路）/ disagree → 第三条件 False
    assert produced_without_teacher_anchor_arith(cross_verify_ran=False, cv_all_agree=True) is False
    assert produced_without_teacher_anchor_arith(cross_verify_ran=True, cv_all_agree=False) is False
    # E2 整体仍 False（teacher_offline/probe_input_novel defer·W2 只第三条件算术域就位·非 E2 过）
    assert e2_execution_ready() is False
    assert e2_independent_production(teacher_offline=False, probe_input_novel=False,
                                     produced_without_teacher_anchor=True) is False
    # W7 才接全 E2：三条件全 True 才 True（算术域 W2+W4+W6 全就位后）
    assert e2_independent_production(teacher_offline=True, probe_input_novel=True,
                                     produced_without_teacher_anchor=True) is True


def test_w2_gate_off_bit_identical(tmp_path):
    """weaning_phase=PRE（默认）→ POST 短路·W0/W1 既有行为 bit-identical。

    PRE 不翻 MODE_B_CROSS_VERIFY_MODE + 不填 arith_source_b → _run_verify_round 走 PRE 分支
    （spec.expected vm_proof·非 POST cross-verify）。conduction_rate>0（PRE reward·W0 既有）+
    weaning_ready=False（E2 仍 False·mock POST 不偷渡）+ run 后 gate 恢复 OFF（try/finally 守 CI）。
    """
    from pure_integer_ai.config import gates
    assert gates.MODE_B_CROSS_VERIFY_MODE is False   # run 前 OFF（CI 默认）
    result = run_weaning_arith(
        rounds_per_stage=1, training_mode=True, flat_floors=True,
        run_dir=str(tmp_path / "pre"), return_backend=False,
        weaning_phase=WEANING_PRE)
    assert gates.MODE_B_CROSS_VERIFY_MODE is False, (
        "PRE 须不翻 MODE_B_CROSS_VERIFY_MODE·run 后恢复 OFF（try/finally 守 CI bit-identical）")
    # PRE 路径：conduction_rate>0（spec.expected vm_proof·W0/W1 既有·非 POST cross-verify）
    assert result.final_metrics.conduction_rate > 0
    # E2 仍 False（mock POST 不偷渡·weaning_ready 仍 False·W2 只在 weaning_phase=POST 时激活）
    assert not result.weaning_ready
    assert "E2_independent_production" in result.weaning_blockers


def test_w2_beta_arith_no_contam_post(tmp_path):
    """POST 也绕 episode_loop·experience_count 台账空（β_arith 不染 POST cross-verify reward）。

    POST _run_verify_round 同 PRE 绕 episode_loop（formal_train.py:375 早返·在 episode_loop :506 之前
    return）·不调 propagate_reward·不写 e_sn/e_tn。β_arith e_sn/e_tn rate 不染 POST cross-verify reward
    （直调 cross_verify_pair·reward=all_agree 非 experience rate）。镜像 W1 test_w1_beta_arith_no_contam_arith。
    """
    _result, backend = run_weaning_arith(
        rounds_per_stage=1, training_mode=True, flat_floors=True,
        run_dir=str(tmp_path / "beta"), return_backend=True,
        weaning_phase=WEANING_POST)
    rows = backend.select(EXPERIENCE_COUNT_TABLE, where={}, limit=100)
    assert len(rows) == 0, (
        f"experience_count 台账须空（POST _run_verify_round:375 绕 episode_loop 不调 propagate_reward·"
        f"不写 e_sn/e_tn·β_arith 不染 POST cross-verify reward·直调 cross_verify_pair reward=all_agree）"
        f"·got {len(rows)} 行")
