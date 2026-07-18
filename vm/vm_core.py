"""vm.vm_core — 图即程序执行（纯整数 ITER·栈机·step_limit 禁无限步）。

execute(instructions, env, *, step_limit) → Rational 结果。
栈机语义：
  LOAD  arg → push env[arg]（变量 symbol_id → Rational 值）
  STORE arg → pop → env[arg]
  ADD/SUB/MUL/DIV/EQ/LT/GT → pop b, pop a, push dispatch_binary(opcode, a, b)
  NOP → no-op
  A1 控制流（PC = 指令表整数下标·图灵完备·doc/重来_VM图灵完备与C6设计补充.md §二）：
  PUSH_IMM (num,den) → push make(num,den)（有理立即数·常量/循环界）
  JZ target → pop；零则 pc=target 否则 pc+1（条件分支·EQ/LT/GT 产 ONE/ZERO 喂此）
  JMP target → pc=target（无条件跳·循环回边在字节码非图·COMPOSES 仍 DAG）
  HALT → 显式终止返栈顶（落末尾隐式终止兼容旧直线程序·行为零变化）

step_limit：总步数上限·A1 后是**唯一终止界**（§九铁律·禁无限步·超 → StepLimitExceeded 不挂·
  JMP/JZ 回边经 step_limit 界·**非**结构性循环检测——后者会禁止迭代本身违图灵完备目的）。
纯整数：栈值全 Rational·DIV 走有理倒数精确·零浮点（assert_no_float 守入口）。

execute 不读 gate（primitive·测试/验机制/training 层 execute_composes_value 直接调）·gate DISPATCH_MODE 装饰位
零读取（机制不读 gate·无条件跑·cognition 永不调 VM 单向依赖守·故 gate 永无接线点读·OFF/ON 等价 bit-identical·见 gates.py 装饰位范式）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer.rational import Rational, ZERO, make, is_zero
from pure_integer_ai.numeric.symbol_domain import (
    OPCODE_NOP, OPCODE_LOAD, OPCODE_STORE,
    OPCODE_PUSH_IMM, OPCODE_JZ, OPCODE_JMP, OPCODE_HALT,
)
from pure_integer_ai.vm.graph_compile import Instruction
from pure_integer_ai.vm.dispatch import dispatch_binary, is_binary_opcode

DEFAULT_STEP_LIMIT = 1 << 16  # 65536 步上限（防病态长程序·代码域通常远小于此）


class StepLimitExceeded(RuntimeError):
    """执行超 step_limit（禁无限步·§九铁律·不挂）。"""


def execute(
    instructions: list[Instruction],
    env: dict[int, Rational],
    *,
    step_limit: int = DEFAULT_STEP_LIMIT,
) -> Rational:
    """执行指令序列·返回栈顶 Rational（空栈返 ZERO）。

    env：变量 symbol_id → Rational 值（LOAD 源 / STORE 目标）。env 被原地修改（STORE 写回）。
    step_limit：总执行步数上限（超 → StepLimitExceeded）。
    """
    assert_int(step_limit, _where="execute.step_limit")
    if step_limit < 0:
        raise ValueError(f"step_limit 须 ≥ 0: {step_limit}")
    stack: list[Rational] = []
    pc = 0
    n = len(instructions)
    steps = 0
    while pc < n:
        steps += 1
        if steps > step_limit:
            raise StepLimitExceeded(
                f"执行超 step_limit={step_limit}（at step {steps}, pc={pc}）"
            )
        instr = instructions[pc]
        op = instr.opcode
        # 守纯整数（opcode/args 是 symbol_id·入口 assert）
        assert_no_float(op, *instr.args, _where="execute.instr")

        # ---- 直线 opcode（pc += 1·行为与旧 for-walk 零变化） ----
        if op == OPCODE_NOP:
            pc += 1
            continue
        if op == OPCODE_LOAD:
            (var,) = instr.args
            if var not in env:
                raise KeyError(f"LOAD: 变量未绑定 {var!r}")
            stack.append(env[var])
            pc += 1
            continue
        if op == OPCODE_STORE:
            (var,) = instr.args
            if not stack:
                raise IndexError("STORE: 栈空")
            env[var] = stack.pop()
            pc += 1
            continue
        if is_binary_opcode(op):
            if len(stack) < 2:
                raise IndexError(f"二元算子栈不足: opcode={op}, stack_len={len(stack)}")
            b = stack.pop()
            a = stack.pop()
            stack.append(dispatch_binary(op, a, b))
            pc += 1
            continue

        # ---- A1 控制流（PC = 指令表整数下标·step_limit 唯一终止界·doc/重来_VM图灵完备与C6设计补充.md §二） ----
        if op == OPCODE_PUSH_IMM:
            (num, den) = instr.args
            if den == 0:
                raise ZeroDivisionError("PUSH_IMM: den=0")
            stack.append(make(num, den))
            pc += 1
            continue
        if op == OPCODE_JZ:
            (target,) = instr.args
            if not stack:
                raise IndexError("JZ: 栈空")
            if is_zero(stack.pop()):
                _check_jump_target(target, n, pc)
                pc = target
            else:
                pc += 1
            continue
        if op == OPCODE_JMP:
            (target,) = instr.args
            _check_jump_target(target, n, pc)
            pc = target
            continue
        if op == OPCODE_HALT:
            return stack[-1] if stack else ZERO   # 显式终止
        raise ValueError(f"execute: 未知 opcode {op!r}")
    # 落到末尾（无 HALT）= 隐式终止·兼容旧直线程序（行为零变化）
    return stack[-1] if stack else ZERO


def _check_jump_target(target: int, n: int, src_pc: int) -> None:
    """JMP/JZ 目标越界 fail-loud（非静默回绕·A1 控制流安全）。

    target ∈ [0, n]：n = 落末尾隐式终止（与 fall-off-end 一致·A2 控制流 skip-to-end 合法）。
    IF 无 else / WHILE 在程序末尾时 skip/exit 目标 = n（跳到末尾 = 终止）·非越界。
    """
    assert_int(target, _where="_check_jump_target.target")
    if not (0 <= target <= n):
        raise IndexError(
            f"跳转目标越界: target={target}, len={n}, src_pc={src_pc}"
        )
