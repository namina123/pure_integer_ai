"""numeric.numeric_relatedness — 大常数−交叉积差值归一 · 跨轴=0 接地墙（依赖 crosscut）。

数值关联只在**同轴**内有意义：同轴两概念的定点值 a/b 与 c/d 的数值关联度经交叉积差值
归一到有理数 ∈ [0,1]；**跨轴 = 0**（接地墙：不同轴间无数值关联·§八限制空间·守门）。

公式（同轴·纯整数·零浮点）：
  diff = |a·d − b·c|              （交叉积差值·compare.cross_compare 的 |sign| 量化版）
  raw  = max(0, BIG − diff)       （大常数 − 差值·clamped ≥ 0）
  rel  = raw / BIG                （归一到 [0,1] 有理·返回 (num, den) 有理对）

BIG 是归一化大常数（默认 BASE**DEFAULT_K = 2^240；可配·oracle 起点非终值）。
diff ≥ BIG 时关联 = 0（差异过大视为无数值关联）。

【诚实边界】
  - 跨轴=0 是承重接地墙（设计稳定·硬约束）。
  - 同轴归一公式是 oracle 起点非终值：diff 的尺度依赖 b,d 量级，BIG 选值随 oracle 标定调；
    Stage 5 oracle 验后可调 BIG 或改归一（比序仍走 cross_compare 零误差）。
  - 不判语义相关（接地墙·只按轴内定点值数值距离）。

【诚实标注·零生产 caller（C1 设计决断 2026-07-03·doc/重来_ConceptNumeric数值轴设计决断.md）】
  本函数 **零生产 caller**（仅 test_stage0 显式值单测）。跨轴=0 接地墙在 v1 **诚实空转**：numeric_relatedness
  比 axis_id·v1 无 concept_numeric 数据→墙无对手→不触发。**诚实非纸面闭合**（不建无消费者数据·违最少边）。
  保留范式 = W2-B6（designed+tested+零 caller+阻塞于整层 defer）·待用途②激活（模态/reward）后·
  concept_numeric populate → 本函数自然有数据承重→跨轴=0 墙生效。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.integer.constants import BASE, DEFAULT_K
from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int

# 默认归一化大常数。oracle 起点非终值（Stage 5 可调）。
DEFAULT_BIG = BASE ** DEFAULT_K  # 2^240


def relatedness(a_num: int, a_den: int, b_num: int, b_den: int,
                axis_a: int, axis_b: int,
                big: int = DEFAULT_BIG) -> tuple[int, int]:
    """两概念定点值 a_num/a_den 与 b_num/b_den 的数值关联度（有理对 (num, den)）。

    跨轴（axis_a != axis_b）→ (0, 1)（接地墙·无数值关联）。
    同轴 → raw/BIG 归一到 [0,1]；diff ≥ BIG → (0, 1)。
    分母 a_den, b_den 须 > 0。
    """
    assert_no_float(a_num, a_den, b_num, b_den, axis_a, axis_b, big,
                    _where="numeric_relatedness")
    assert_int(a_num, a_den, b_num, b_den, axis_a, axis_b, big,
               _where="numeric_relatedness")
    if a_den <= 0 or b_den <= 0:
        raise ValueError(f"分母须为正: a_den={a_den}, b_den={b_den}")
    if big <= 0:
        raise ValueError(f"big 须为正: {big}")

    # 跨轴 = 0 接地墙（承重·硬约束）
    if axis_a != axis_b:
        return (0, 1)

    diff = a_num * b_den - b_num * a_den
    if diff < 0:
        diff = -diff
    if diff >= big:
        return (0, 1)
    return (big - diff, big)


def is_ground_wall(axis_a: int, axis_b: int) -> bool:
    """是否跨轴（接地墙触发·无数值关联）。"""
    return axis_a != axis_b
