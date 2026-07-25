"""training.numeric_proof — 数值等式验序器（刀 B·语言域形式 cue·构造性检查层）。

镜像 time_seq_proof_fn_factory 范式·但验 **数值等式声明的算术一致性**（直接整数算术）·
非 PRECEDES DAG 无环·非 COMPOSES 执行值。

机制（Option A·数值声明不入图·瞬态闭包计算·同刀 A 时序边不入图）：
  - caller（_run_numeric_verify_round）收集 segments.numeric_claims（已解析纯整数 4-tuple·self-contained·
    无需 token→ConceptRef resolve·无需 backend query·比时序更简）→ factory 闭包捕获 claims
  - 内层逐声明整数算术（ADD/SUB/MUL）计算 left op right·比对 result_num → 全声明成立返 1 / 任一违返 0 / 空返 None
  - 同 time_seq_proof_fn 闭包传（非持久化边/节点）·通道分离：reward 通道边入图 / self_proof_fn 通道闭包传

返值（守 SelfProofFn 协议·judge.py:49·三参数 output/dag_path/graph 占位·内层只用闭包 claims）：
  1 = 全数值声明算术一致（verified·构造性检查通过）
  0 = 任一声明违反（mismatch·如 "3 加 5 等于 9"·3+5=8≠9）
  None = 无声明（vacate·无数值等式可验·诚实退场·非 pass）

**诚实分层（构造性检查 ≠ 构造性验证·同刀 A）**：
  - 构造性检查 ✅：整数算术确定性可执行（+,-,×·纯整数·零浮点·assert_int 在提取期守）·验等式声明算术一致
  - 构造性验证 ❌：左式/右式数均 single-source（来自文本 cue 锚·非 R6 独立源）→ 非构造性验证
    （须 R6 独立源升验证·Layer0 标 SELF_PRODUCED·全自产不准停）
  - 为何直接整数算术非 execute_composes_value：平坦表达式 "3+5" 直接整数算术即可·建 COMPOSES 树调
    execute_composes_value 反而图污染（加 EDGE_COMPOSES 节点/边·干扰结构发现）+ 无验证增益（R6 仍缺·
    single-source 检查不论机制）。刀 A 已立"闭包非图"先例（时序边不入图）·刀 B 同理。execute_composes_value
    留给真构造性验证（arith 域 R6·expected 独立源·Mode B 已闭环·非本刀语言域 single-source 检查）。
  - stable≠correct：数值声明算术一致 ≠ 命题真（"3 加 5 等于 8" 算术对·但文本是否真在陈述此算式是语义层·#479 墙）

依赖方向：training(L7)→numeric(L1) 向下 + cognition(L5) 向下·lint 允许（镜像 time_seq_proof.py）。
judge(cognition L5) 不 import training·numeric_proof_fn 在 training 接线层建（守解耦）。

铁律：纯整数（claims 全 int·算术 +,-,× 整数保持·零浮点·零除法·OPCODE_DIV defer）/ 确定性
      （逐声明算术·bit-identical）/ fail-loud（任一声明违→0 非 pass·空→None 非 theater）/ stable≠correct / 永不接 reward。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.numeric.symbol_domain import OPCODE_ADD, OPCODE_SUB, OPCODE_MUL
from pure_integer_ai.cognition.result.judge import SelfProofFn


def numeric_proof_fn_factory(*, claims: list[tuple[int, int, int, int]]) -> SelfProofFn:
    """造数值等式验序器 fn（闭包捕获 claims·镜像 time_seq_proof_fn_factory）。

    claims : 数值等式声明 list·每元 (left_num, op_opcode, right_num, result_num)·纯整数。
             op_opcode ∈ {OPCODE_ADD, OPCODE_SUB, OPCODE_MUL}（除法 defer·arith_op_of 仅返此三）。
             caller（_run_numeric_verify_round）收集自 segments.numeric_claims（extract_numeric_claims 已解析）。

    返 SelfProofFn(output, dag_path, graph) -> int|None：
      1（全声明算术一致）/ 0（任一声明违反·如 3+5≠9）/ None（claims 空 vacate·同 time_seq 三态）。
      _run_numeric_verify_round 直调（绕 judge·reward=1 iff r==1）·self_proof_check 不经（镜像 _run_verify_round）。

    **构造性检查层**（诚实·非构造性验证·同刀 A）：整数算术确定性·非 R6 独立源·Layer0 标 SELF_PRODUCED。
    """
    # 防御性拷贝（对抗审 P2-3）：闭包按引用捕获·防 caller 后续 mutation（claims.append）改 fn 行为。
    # 成本可忽略（浅拷贝 list·tuple 元素不可变）。镜像 time_seq_proof_fn 同模式（其边集 list() 拷贝）。
    claims = list(claims)

    def numeric_proof_fn(output: Any, dag_path: Any, graph: Any) -> int | None:
        # output/dag_path/graph 占位（守 SelfProofFn 协议三参数·内层只用闭包 claims·同 time_seq_proof_fn）
        if not claims:
            return None   # 无数值声明 → vacate（诚实退场·非 pass·非 theater）
        for (left_num, op, right_num, result_num) in claims:
            if op == OPCODE_ADD:
                computed = left_num + right_num
            elif op == OPCODE_SUB:
                computed = left_num - right_num
            elif op == OPCODE_MUL:
                computed = left_num * right_num
            else:
                # 不可达（arith_op_of 仅返 ADD/SUB/MUL·除法 defer）·防御：未知 op → 检查失败（非 pass·防静默放行）
                return 0
            if computed != result_num:
                return 0   # 声明违反（如 3+5=8 但 result=9）→ mismatch
        return 1   # 全声明算术一致

    return numeric_proof_fn
