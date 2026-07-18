"""numeric.concept_numeric — 概念在某轴的代分数定点值（依赖 crosscut）。

ConceptNumeric(space_id, local_id, axis_id, axis_symbol_id, M, k)：
  值 = M / B^k（定点有理·B=2^30·与 fixed_point 同构）
  axis_id       —— 哪个数值轴（TYPE_AXIS_*）
  axis_symbol_id —— 图即程序 opcode / 变量 / 参数 / 数值关联标记（symbol_domain 区分）
                   区分"图即程序（concept 是算子节点）"vs"数值关联（concept 有定点值）"

同一概念可在多轴上有定点值（一概念多抽象身份·§八限制空间）。
纯整数（assert_no_float 守入口）；M/B^k 真值经 fixed_point.to_rational_interval 取区间。

【诚实标注·零生产 caller（C1 设计决断 2026-07-03·doc/重来_ConceptNumeric数值轴设计决断.md）】
本 dataclass **零生产 caller**（grep 仅自身 + numeric/__init__ re-export）·**不建扩展表不 populate**。
两用途判决：
  - 用途①图即程序立即数值 → reboot 已由 composes_attr.ATTR_IMMEDIATE 服务（graph_compile 读
    immediate_of emit PUSH_IMM）·**非本 dataclass**。
  - 用途②数值关联轴值 → **layer-defer**：语言数值面空（§8.8·文字无外观 ARGB）+ 模态 defer（C10）
    + credit_sink 奖励涌现故意断线（防塌柱①）。v1 无墙内干净闭环。
保留（非删）范式 = W2-B6 `_build_metaphor_three_layer`：designed + tested + 零 caller + 阻塞于整层
defer → 保留 + 诚实标注·待模态扩展（2D ARGB/声强度）或 credit_sink reward 环激活。TYPE_AXIS_* 是
值类型枚举（元定义合法·非语义轴）·语义轴身份在 axis_symbol_id KIND_NUMERIC 维（C10 解）。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


@dataclass(frozen=True)
class ConceptNumeric:
    """概念在某轴的代分数定点值 M/B^k。"""

    space_id: int        # 三空间之一（核心/记忆/伴随）
    local_id: int        # 概念在 space 内的本地 id
    axis_id: int         # 数值轴（TYPE_AXIS_*）
    axis_symbol_id: int  # 图即程序 opcode / 变量 / 参数 / 数值关联标记
    M: int               # 定点整数（值 = M / B^k）
    k: int               # 精度位数（≥ 0）

    def __post_init__(self) -> None:
        assert_no_float(
            self.space_id, self.local_id, self.axis_id,
            self.axis_symbol_id, self.M, self.k,
            _where="ConceptNumeric",
        )
        assert_int(
            self.space_id, self.local_id, self.axis_id,
            self.axis_symbol_id, self.M, self.k,
            _where="ConceptNumeric",
        )
        if self.k < 0:
            raise ValueError(f"ConceptNumeric.k 须 ≥ 0，got k={self.k}")
