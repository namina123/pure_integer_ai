"""training.oracle — oracle 标定（B1-B4 占位校验 + H2 小批量权重标定·§十四 line1030/H2）。

  validate_b1_b4() —— 校验四项占位值与系统常量一致（B1 J3path 1:10 / B2 新种子 / B3 N=1000 / B4 逐个叠加）
  calibrate_weights(samples, judge_fn, teacher) -> JudgeWeights —— H2 小批量离线标定
  conduction_rate(samples, ...) -> int —— reward 导通率（×1000·阶段3 门控）

**H2 落盘（§十四 line1030·消解鸡生蛋）**：judge 权重靠 oracle 标定·但标定需 judge 跑过有输出+教师
ground-truth 对照 → "标定需 judge / judge 需权重"死锁。次序：**阶段3 先小批量离线标定权重
（教师 ground-truth 经录放层·纯整数网格最大化 agreement）→ 权重定后开全量 reward 反传**。
默认权重期 reward 不落 strength（防错权重调错 strength 污染图）。

  calibrate_weights：纯整数**全网格搜索** WEIGHT_GRID^3 最大化 agreement（judge reward>0 ⟺ 教师 GT=pass）。
    agreement = |{s : sign(judge_reward(s)) == teacher_gt(s)}| / |samples|
    网格：w_i ∈ WEIGHT_GRID（纯整· oracle 标定非硬编码·网格搜索非梯度）。
    标定集 = 小批量带教师标准答案样本（QA+推理题）·agreement 对齐教师 ground-truth（标定时依赖教师·
    运行时判据自锚输入不依赖教师·§十四 line1031·两时相不同非矛盾）。

**B1-B4（拍板清单·已占位·oracle 验后调）**：
  B1 J3path PRECEDES:CAUSES = 1:10（types.py J3_CAUSES_WEIGHT/J3_PRECEDES_WEIGHT·序边贡献一个数量级低）
  B2 seeded 探索 = 新种子（anti_collapse.inject_seeded_exploration·A3 linear 叠加零损失·非权重扰动）
  B3 收敛窗口 N=1000 episode（convergence.CONVERGENCE_WINDOW·断奶评估 window_rounds=4 runs 两窗口不混）
  B4 attractor 扩张逐个叠加（卷二模块5·守线性性零损失·非层末批量）

铁律：纯整数（权重/网格/agreement 全整·×1000）/ 不写死（网格+占位值·oracle 验后调·非硬编码单值）/
  外部只启发（教师 ground-truth 经录放层标定权重·运行时判据自锚输入·断奶后教师退场权重冻结）/ 不走外挂 LLM
  （标定用录放层录制好的 ground-truth·零 LLM）。依赖 cognition + teacher。
诚实边界：oracle 标定是经验调参非理论最优（网格有限·阈值经验）/ 权重断奶后冻结防自评膨胀（非语义保证）/
  agreement 是结构对齐非语义对（stable≠correct）。
"""
from __future__ import annotations

from typing import Callable, Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.cognition.shared.types import (
    JudgeWeights, J3_CAUSES_WEIGHT, J3_PRECEDES_WEIGHT,
)
from pure_integer_ai.cognition.result.convergence import CONVERGENCE_WINDOW

# H2 权重网格（纯整·oracle 标定搜索空间·非硬编码单值·非梯度）
WEIGHT_GRID: tuple[int, ...] = (1, 2, 3, 5, 8)
# reward 符号判据（judge reward > 0 = 正确·== 0 = veto·< 0 = 死路·agreement 对齐教师 GT pass/fail）
REWARD_POSITIVE = 0

_SCALE = 1000


def validate_b1_b4() -> None:
    """校验 B1-B4 占位值与系统常量一致（实施前置·防漂移）。

    raise AssertionError 若占位值被改未同步 oracle 登记员。oracle 验后调须同改此处+常量。
    """
    # B1：J3path PRECEDES:CAUSES = 1:10
    assert J3_CAUSES_WEIGHT == 10 and J3_PRECEDES_WEIGHT == 1, (
        f"B1 占位漂移：J3_CAUSES_WEIGHT={J3_CAUSES_WEIGHT} "
        f"J3_PRECEDES_WEIGHT={J3_PRECEDES_WEIGHT}·预期 10:1（oracle 验后调须同改）")
    # B2：seeded 探索 = 新种子（机制·非标量·stub #11 修：旧版仅注释·今校验机制存在可证伪）
    from pure_integer_ai.cognition.result.anti_collapse import inject_seeded_exploration
    assert callable(inject_seeded_exploration), (
        "B2 落点缺失：anti_collapse.inject_seeded_exploration（seeded 探索新种子机制）")
    # B3：收敛窗口 N=1000 episode
    assert CONVERGENCE_WINDOW == 1000, (
        f"B3 占位漂移：CONVERGENCE_WINDOW={CONVERGENCE_WINDOW}·预期 1000（oracle 验后调须同改）")
    # B4：attractor 扩张逐个叠加（机制·非标量·stub #11 修：旧版仅注释·今校验机制存在可证伪）
    from pure_integer_ai.cognition.process.attractor import maybe_expand_attractor
    assert callable(maybe_expand_attractor), (
        "B4 落点缺失：attractor.maybe_expand_attractor（逐个叠加扩张机制）")


# ---- H2 小批量权重标定 ----

# judge_fn 契约：(output, path, input, workmem, *, weights) -> (reward, GMeta)
# teacher_gt 契约：(sample) -> int（GT_PASS=1/GT_FAIL=0·教师 ground-truth 经录放层）
JudgeFn = Callable[..., tuple[int, Any]]
TeacherGT = Callable[[Any], int]


def _agreement(judge_fn: JudgeFn, teacher_gt: TeacherGT, samples: list[Any],
               weights: JudgeWeights) -> int:
    """agreement = 命中数（judge reward>0 ⟺ 教师 GT=pass·纯整计数）。

    teacher_gt 返 None（miss/退场）→ 跳过不参与对齐（stub #3·防 None 当 pass 凑 agreement）。
    同 samples 同 teacher_gt 确定性·跨权重一致跳过·hit 可比。
    """
    hit = 0
    for s in samples:
        gt = teacher_gt(s)
        if gt is None:
            continue   # miss GT·不参与对齐（防占位凑 agreement）
        reward, _ = judge_fn(s, weights=weights)
        judge_positive = reward > 0
        teacher_pass = gt != 0
        if judge_positive == teacher_pass:
            hit += 1
    return hit


def calibrate_weights(samples: list[Any], judge_fn: JudgeFn,
                      teacher_gt: TeacherGT) -> JudgeWeights:
    """H2 小批量离线标定 judge 权重（纯整数全网格最大化 agreement·§十四 H2）。

    **全网格搜索 WEIGHT_GRID^3**（三重嵌套遍历所有 (w1,w2,w3) 组合·纯整非梯度·**B4 诚实化**：
    原称"坐标上升剪枝"是错——实为全网格·坐标上升 perf 优化 defer）·选 agreement 最大的权重。
    断奶后冻结（防自评膨胀第一闸·§十四 line1031）。tiebreak：权重和最小（防堆量游戏）。
    """
    if not samples:
        return JudgeWeights()   # 空标定集→默认权重（1,1,1,1·w4 默认 1·#1041·网格不动 w4·oracle 验后调）
    best = JudgeWeights()
    best_hit = _agreement(judge_fn, teacher_gt, samples, best)
    best_sum = best.w1 + best.w2 + best.w3
    # 纯整数全网格 WEIGHT_GRID^3（遍历所有组合·坐标上升 perf 优化 defer·非剪枝）
    for w1 in WEIGHT_GRID:
        for w2 in WEIGHT_GRID:
            for w3 in WEIGHT_GRID:
                cand = JudgeWeights(w1=w1, w2=w2, w3=w3)
                hit = _agreement(judge_fn, teacher_gt, samples, cand)
                cand_sum = w1 + w2 + w3
                # 选 agreement 最大·tiebreak 权重和最小（防堆量）
                if hit > best_hit or (hit == best_hit and cand_sum < best_sum):
                    best, best_hit, best_sum = cand, hit, cand_sum
    return best


def conduction_rate(samples: list[Any], judge_fn: JudgeFn) -> int:
    """reward 导通率（×1000·阶段3 门控·图规模/reward 闭环导通度量）。

    = 1000 × |{s : reward > 0}| / |samples|。导通率=0=reward 闭环未导通禁进阶段4（守几百G不重训红线）。
    """
    if not samples:
        return 0
    conducted = 0
    for s in samples:
        reward, _ = judge_fn(s)
        if reward > 0:
            conducted += 1
    return (_SCALE * conducted) // len(samples)


def agreement_rate(samples: list[Any], judge_fn: JudgeFn,
                   teacher_gt: TeacherGT, weights: JudgeWeights) -> int:
    """agreement 率（×1000·标定质量度量·judge 与教师 ground-truth 对齐率）。"""
    if not samples:
        return 0
    return (_SCALE * _agreement(judge_fn, teacher_gt, samples, weights)) // len(samples)
