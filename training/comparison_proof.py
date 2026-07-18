"""training.comparison_proof — 比较声明验序器（刀 D·语言域形式 cue·构造性检查层）。

镜像 numeric_proof_fn_factory 范式·但验 **比较声明的算术序一致性**（cross_compare 交叉积·比序唯一零误差路径）·
非整数等式算术·非 PRECEDES DAG 无环·非 COMPOSES 执行值。

机制（Option A·比较声明不入图·瞬态闭包计算·同刀 A 时序 / 刀 B 数值边不入图）：
  - caller（_run_comparison_verify_round）收集 segments.comparison_claims（已解析纯整数 3-tuple·self-contained·
    无需 token→ConceptRef resolve·无需 backend query·比数值更简）→ factory 闭包捕获 claims
  - 内层逐声明 cross_compare(left,1,right,1)=sign(left−right)·比对 cmp_opcode（GT/LT/GE/LE）→ 全成立返 1 /
    任一违返 0 / 空返 None
  - 同 time_seq/numeric_proof_fn 闭包传（非持久化边/节点）·通道分离：reward 通道边入图 / self_proof_fn 通道闭包传

返值（守 SelfProofFn 协议·judge.py:49·三参数 output/dag_path/graph 占位·内层只用闭包 claims）：
  1 = 全比较声明比序一致（verified·构造性检查通过·如 "5 大于 3"·sign(5−3)=1>0 ✓）
  0 = 任一声明违反（mismatch·如 "3 不小于 5"·sign(3−5)=−1·GE 须 ≥0 → 违）
  None = 无声明（vacate·无比较声明可验·诚实退场·非 pass）

**诚实分层（构造性检查 ≠ 构造性验证·同刀 A/B）**：
  - 构造性检查 ✅：cross_compare 交叉积确定性零误差（纯整数·零浮点·零定点·assert_int 在提取期守）·验比较声明序一致
  - 构造性验证 ❌：左/右式数均 single-source（来自文本 cue 锚·非 R6 独立源）→ 非构造性验证
    （须 R6 独立源升验证·Layer0 标 SELF_PRODUCED·全自产不准停）
  - **为何用 cross_compare 非裸 sign(left−right)**：compare.py docstring "任何'比序'语义强制走本模块·禁止走定点近似"
    → 刀 D 守此铁律。cross_compare = 比序唯一零误差路径（交叉积·不计算商）·整数（den=1）是 Rational 特例·
    未来分数 operand（num/den）无缝扩（defer·本刀整数）。**给 cross_compare 首个真比较消费者**（既有 1 caller
    非比较用途·分层墙 §四缝1·反 theater：机制获真消费者）。
  - stable≠correct：比较声明序一致 ≠ 命题真（"5 大于 3" 算术对·但文本是否真在陈述此比较是语义层·#479 墙）

依赖方向：training(L7)→crosscut(L1) 向下（cross_compare/CMP_*）+ cognition(L5) 向下（judge SelfProofFn）·
lint 允许（镜像 numeric_proof.py）。judge(cognition L5) 不 import training·comparison_proof_fn 在 training
接线层建（守解耦）。

铁律：纯整数（claims 全 int·cross_compare 交叉积·零浮点·零除法）/ 确定性（逐声明比序·bit-identical）/
      fail-loud（任一声明违→0 非 pass·空→None 非 theater·未知 cmp→0 防静默放行）/ stable≠correct / 永不接 reward。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.integer.compare import cross_compare, CMP_GT, CMP_LT, CMP_GE, CMP_LE
from pure_integer_ai.cognition.result.judge import SelfProofFn


def comparison_proof_fn_factory(*, claims: list[tuple[int, int, int]]) -> SelfProofFn:
    """造比较声明验序器 fn（闭包捕获 claims·镜像 numeric_proof_fn_factory）。

    claims : 比较声明 list·每元 (left_num, cmp_opcode, right_num)·纯整数。
             cmp_opcode ∈ {CMP_GT, CMP_LT, CMP_GE, CMP_LE}（大于/小于/不小于/不大于·cue_words.comparison_op_of 识别）。
             caller（_run_comparison_verify_round）收集自 segments.comparison_claims（extract_comparison_claims 已解析）。

    返 SelfProofFn(output, dag_path, graph) -> int|None：
      1（全声明比序一致）/ 0（任一声明违反·如 "3 不小于 5"·3<5 GE 须≥0 违）/ None（claims 空 vacate·同 numeric 三态）。
      _run_comparison_verify_round 直调（绕 judge·reward=1 iff r==1）·self_proof_check 不经（镜像 _run_verify_round）。

    **构造性检查层**（诚实·非构造性验证·同刀 A/B）：cross_compare 确定性·非 R6 独立源·Layer0 标 SELF_PRODUCED。
    """
    # 防御性拷贝（对抗审 P2-3·镜像 numeric_proof）：闭包按引用捕获·防 caller 后续 mutation（claims.append）改 fn 行为。
    # 成本可忽略（浅拷贝 list·tuple 元素不可变）。
    claims = list(claims)

    def comparison_proof_fn(output: Any, dag_path: Any, graph: Any) -> int | None:
        # output/dag_path/graph 占位（守 SelfProofFn 协议三参数·内层只用闭包 claims·同 numeric_proof_fn）
        if not claims:
            return None   # 无比较声明 → vacate（诚实退场·非 pass·非 theater）
        for (left_num, cmp, right_num) in claims:
            # cross_compare(left,1,right,1) = sign(left·1 − 1·right) = sign(left − right)·整数（den=1）·零误差
            sign = cross_compare(left_num, 1, right_num, 1)
            if cmp == CMP_GT:
                ok = sign > 0     # 大于：left > right
            elif cmp == CMP_LT:
                ok = sign < 0     # 小于：left < right
            elif cmp == CMP_GE:
                ok = sign >= 0    # 不小于：left ≥ right
            elif cmp == CMP_LE:
                ok = sign <= 0    # 不大于：left ≤ right
            else:
                # 不可达（comparison_op_of 仅返 GT/LT/GE/LE）·防御：未知 cmp → 检查失败（非 pass·防静默放行）
                return 0
            if not ok:
                return 0   # 声明违反（如 "3 不小于 5"·3<5·GE 须≥0 → sign=−1 违）→ mismatch
        return 1   # 全声明比序一致

    return comparison_proof_fn
