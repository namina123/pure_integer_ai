"""training.vm_proof — VM-proof fn 骨架（A3·C6-wiring·致命#2·依赖 vm+cognition+numeric）。

vm_proof_fn 执行学生自建 COMPOSES 程序·比对独立规格 (input_args → expected)·产 1/0/None：
  - 读 dag_path.sink（=struct_ref=COMPOSES 根·致命#3）的 COMPOSES 子树（ConceptGraph.read_composes_tree）
  - compile_graph 编译 → execute（固定 step_limit·catch StepLimitExceeded→None·R1 vacate）
  - 比 result == expected → 1（verified）/ 0（mismatch）
  - root 非 COMPOSES 根 / StepLimit → None（vacate·self_proof_check 按 weaning_phase 路由 #361）

**R6 独立源**：expected 来自手编/教师（独立源·非学生 COMPOSES 编译）→ 两路独立判同一事
  (input→result)·非 theater（C6 doc §5.2）。断奶后代码域 Mode B = vacate（单源诚实·待第二源）。

**生产接线（2026-07-03 C2 纠错·已落）**：旧文本"生产接线 defer / 仅单测可达"**过时**——vm_proof_fn
  **已生产可达**（formal_train.py:367 `_run_verify_round` PRE 路径直调·CodeSpec input_args/expected
  契约·绕 judge/generate/propagate·A3/C6-wiring #361 done）。原 build_judge_fn 注入意图物理上不成立
  （judge G2p veto + J-sum 恒 0·代码域无词生成/无 key_skeleton/无 CAUSES）→ 正解=代码域独立 episode
  路径直调（doc/重来_VM图灵完备与C6设计补充.md §4.5 + doc/重来_A3_代码域observe设计补充.md §一）。
  **真缺口 = 断奶后无 expected 独立源**（POST Mode B 无教师→vm_proof 无 expected·formal_train:376
  POST 不调 vm_proof 防 vacuous reward=1 theater·须 E1 第二独立源 #479·非 VM 未接线）。

依赖方向：training(L7)→vm(L3)+cognition(L5)+numeric(L1) 全向下·lint 允许。judge(cognition L5)
  不 import vm·vm_proof_fn 在 training 接线层建（守解耦·doc/重来_VM图灵完备与C6设计补充.md §5.5）。

铁律：纯整数（input_args/expected 全 int·Rational 经 make）/ 确定性（同输入同 PC-trace·exec_hash 跨
  宿主 R3 defer）/ fail-loud（StepLimit→None 非 pass·root 非根→None 非 theater）/ stable≠correct
  （VM 跑通≠对·靠独立 expected 验对错·非 VM 自证）。

**execute_composes_value（值暴露版）**：vm_proof_fn 抽出"编译+执行+预载 PARAM env"为独立帮手·返
  执行值 Rational（非 1/0 比对）。vm_proof_fn 复用它同比对（行为 bit-identical·换皮=好）。供**序列3-min
  验证半闭环**复用（formal_train caller 级 L7）：发现骨架绑识别 params 执行 == held-out 新输入执行值 →
  vm_proof 独立验泛化（识别=结构对齐·vm_proof=执行比对·两路独立·反 theater·§8.7）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer.rational import make
from pure_integer_ai.numeric.symbol_domain import make_variable
from pure_integer_ai.vm.vm_core import execute, StepLimitExceeded, DEFAULT_STEP_LIMIT
from pure_integer_ai.vm.graph_compile import compile_graph
from pure_integer_ai.cognition.result.judge import SelfProofFn


def execute_composes_value(graph: Any, root: Any,
                           param_values: tuple[tuple[int, int], ...]) -> Any | None:
    """编译 + 执行 COMPOSES 根 → 值（Rational）| None（vm_proof_fn 的**值暴露版**·同比对解耦）。

    读 root COMPOSES 子树（ConceptGraph.read_composes_tree）→ compile_graph → 预载 PARAM env
    （make_variable(i) ← param_values[i] 的 (num,den)）→ execute（固定 step_limit·catch
    StepLimitExceeded → None·R1 vacate）。**返执行值 Rational·非 1/0 比对**。

    **None** = root 非 COMPOSES 根（无子无算子属性）/ StepLimitExceeded（不挂·caller 判）。

    **复用**（序列3-min 验证半闭环·formal_train caller 级 L7）：发现骨架绑识别 params 执行 ==
    held-out 新输入执行值 → vm_proof 独立**重新执行**守门。识别 = 结构对齐（_align_walk 抽 PARAM）·
    vm_proof = VM 执行比对·两路独立计算。**诚实定位**（对抗审计·勿过判）：对**正确识别**·骨架与输入结构同构
    （同 opcode/固定位/arity·_align_walk 固定位值等门保）→ VM 把 LOAD mv_i(绑值 v) 与 PUSH_IMM v 等同执行 →
    同值是**构造性预期**（非惊奇交叉验证）。vm_proof 的真"牙"=抓获 PARAM 阅读序错位 / skeleton 编译发散 /
    shape 签名漏判的结构异配（probe 实证：SUB 错参 → -47≠43 不 verified）。重执行本身 = 真 READ+应用消费
    （非 theater·非死写·非自证标签）·牙真（守序错）·非"惊奇证伪"。

    param_values : PARAM 槽值序（DFS 阅读序·= make_variable index 序·与 inline arg_subst 契约一致）·
                   每元 (num, den) Rational。空序 = nullary（无 PARAM 槽·全立即数·如 held-out 立即数输入）。

    **None** = root 非 COMPOSES 根 / StepLimit。**raise KeyError**（非 None）= root 含 OPERAND/PARAM 叶但
    param_values 未绑（unbound LOAD）·caller 须预过滤。immediate 输入（loop1）=nullary 无 OPERAND 叶→不可达。
    operand-input 识别（序列2+）·_verify_generalization 喂 rec.input_probe_values（连续 slot-序·_align_extract 保证
    覆盖全部 input operand slot 含未用 slot·未用 slot never LOADed 故 harmless）→ 全绑·不可达 KeyError。

    铁律：纯整数（num/den 全 int·make 闭运算）/ 确定性（同输入同 PC-trace）/ fail-loud
          （StepLimit→None 非 pass·root 非根→None 非 theater）。
    """
    children_of, operator_of, operand_of, immediate_of, store_target_of = \
        graph.read_composes_tree(root)
    # root 非 COMPOSES 根（无属性无子）→ None（非代码域段·不伪造值）
    if not children_of and root not in operator_of:
        return None
    instrs = compile_graph(root, children_of, operator_of, operand_of,
                           immediate_of=immediate_of or None,
                           store_target_of=store_target_of or None)
    env = {make_variable(i): make(num, den) for i, (num, den) in enumerate(param_values)}
    try:
        return execute(instrs, env, step_limit=DEFAULT_STEP_LIMIT)
    except StepLimitExceeded:
        return None   # R1 vacate（不挂·caller 判 None）


def vm_proof_fn_factory(*, input_args: tuple[int, ...],
                        expected: tuple[int, int]) -> SelfProofFn:
    """造 VM-proof fn（闭包捕获独立规格 input_args/expected）。

    input_args : 函数参数值序（对应 FunctionDef.args 顺序·builder 分配 make_variable(0..n-1)）。
    expected   : (num, den) 期望返回值（独立源·手编/教师·R6 独立源非学生编译）。

    返 SelfProofFn(output, dag_path, graph) -> int|None：
      1/0/None（verified/mismatch/vacate）。self_proof_check 按 weaning_phase 路由 None（#361）。
    """
    for a in input_args:
        assert_int(a, _where="vm_proof_fn_factory.input_args")
    assert_int(expected[0], expected[1], _where="vm_proof_fn_factory.expected")
    assert_no_float(*input_args, expected[0], expected[1],
                    _where="vm_proof_fn_factory")
    if expected[1] == 0:
        raise ValueError("vm_proof_fn_factory: expected den 须非零")

    def vm_proof_fn(output: Any, dag_path: Any, graph: Any) -> int | None:
        # root = dag_path.sink（=struct_ref=COMPOSES 根·致命#3）
        root = getattr(dag_path, "sink", None)
        if root is None:
            return None   # 无 sink → vacate（非代码域 episode）
        # 执行学生 COMPOSES（execute_composes_value·值暴露版·input_args int → (arg,1) Rational 预载）
        result = execute_composes_value(
            graph, root, tuple((arg, 1) for arg in input_args))
        if result is None:
            return None   # root 非 COMPOSES 根 / StepLimit → vacate（R1·judge 3态路由 #361）
        # 比对独立规格（R6 独立源·非学生编译·非 theater）
        expected_r = make(expected[0], expected[1])
        return 1 if result == expected_r else 0

    return vm_proof_fn
