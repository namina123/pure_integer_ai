"""vm.graph_compile — 沿 COMPOSES 编译图即程序（依赖 storage + crosscut）。

§九 2a 图即程序：COMPOSES 边（父→子）= 组合关系。父节点是算子（有 opcode）·子节点是操作数。
compile_graph 从 root 沿 COMPOSES 遍历 emit Instruction 序列：
  - 叶节点（无 COMPOSES 子）→ LOAD 其 operand 变量 symbol（或 PUSH_IMM 立即数·A2）
  - 算子节点（有 COMPOSES 子）→ 先递归 emit 各子节点·再 emit 该算子 opcode（后序·栈机消费）
  - STORE 节点（有子 + store_target_of）→ emit 值源子 + STORE 目标变量（A2·控制流体回写 env）
  - 控制流根（operator_of=CTRL_IF/IFELSE/WHILE）→ lower 成 JZ/JMP（A2·见下）

**限深环保护**（§九铁律·禁无限递归·术语脱钩 loop_closure_defect）：
  - max_depth：递归深度上限（超 → LoopClosureDefect·不静默截断）
  - path 栈：检测 COMPOSES 环（图即程序不允许环·环 → LoopClosureDefect）
  COMPOSES 是组合关系·组合环 = 结构矛盾（一个整体直接/间接包含自身）·必须报错非兜底。
  A2 控制流回边在**字节码**（JMP target=指令下标）非图·path/frozenset 看不到·COMPOSES 仍 DAG。

**首版语言接口留空**：compile_graph 只编译既有 COMPOSES 子图·不从高级语言生成图
（代码域 COMPOSES 随代码域阶段激活·§7.4·A3 真瓶颈）。调用方传 children_of（已按 EDGE_COMPOSES
过滤 + 确定性排序）·graph_compile 不查 backend（与 storage 解耦·纯图操作）。

**A2 控制流 lower（2026-07-02·doc/重来_VM图灵完备与C6设计补充.md §三）**：COMPOSES 根可挂
控制流编译指令（CTRL_IF/IFELSE/WHILE·compiler-internal sentinel·非 VM opcode·不入 _OPCODE_TABLE）·
_emit 识别后 lower 成 JZ/JMP·回边在字节码非图（COMPOSES 仍 DAG·LoopClosureDefect 不触发）·
子按 order_index 槽位序（0=COND/1=THEN-BODY/2=ELSE）caller 排。控制流体经 env
（LOAD/STORE loopvar·PUSH_IMM 常量）·状态走 env 非图。
A2 同时补 STORE 节点 + PUSH_IMM 常量叶 emit——图灵完备程序须能回写 env（loopvar）+ 自含常量·
否则 WHILE 不可终止测（无 mutation）+ 违"指令流即整程序"（doc §2.1）。

NodeRef = tuple[int, int] = (space_id, local_id)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.numeric.symbol_domain import (
    OPCODE_LOAD, OPCODE_STORE, OPCODE_PUSH_IMM, OPCODE_JZ, OPCODE_JMP,
)

# NodeRef = (space_id, local_id)
NodeRef = tuple[int, int]

DEFAULT_MAX_DEPTH = 64  # COMPOSES 嵌套深度上限（防病态深递归·代码域通常浅）


class LoopClosureDefect(RuntimeError):
    """COMPOSES 环 / 超限深 退化（loop_closure_defect·术语脱钩安全词）。

    图即程序的组合关系不允许环（整体不能包含自身）·超限深同样报错非截断。
    """


@dataclass(frozen=True)
class Instruction:
    """图即程序指令（纯整数·opcode + args 都是 symbol_id）。

    opcode ∈ OPCODE_*（symbol_domain·LOAD/STORE/PUSH_IMM/JZ/JMP/HALT/二元算子）。
    args：LOAD/STORE → (variable_symbol_id,)；PUSH_IMM → (num, den)；JZ/JMP → (target,)；
          二元算子 → ()（操作数从栈取）。
    """

    opcode: int
    args: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        assert_no_float(self.opcode, *self.args, _where="Instruction")
        assert_int(self.opcode, *self.args, _where="Instruction")


# ---- 控制流编译指令（A2·compiler-internal sentinel·doc/重来_VM图灵完备与C6设计补充.md §三） ----
# 这些是**编译指令**非 VM opcode：不入 symbol_domain._OPCODE_TABLE / 不经 dispatch_binary /
# vm_core.execute 不认。operator_of[node] 携带这些值时 _emit 走控制流分支·lower 成 JZ/JMP。
# 用负数避免与真实 opcode symbol_id（OPCODE_BASE|·≥ 2^60）及 operand variable symbol_id 碰撞。
CTRL_IF = -1        # IF：children=[COND, THEN]（槽位序·caller 按 order_index 排）
CTRL_IFELSE = -2    # IFELSE：children=[COND, THEN, ELSE]
CTRL_WHILE = -3     # WHILE：children=[COND, BODY]

_CTRL_TAGS = frozenset({CTRL_IF, CTRL_IFELSE, CTRL_WHILE})

# 跳转 target backpatch 占位（无效下标·backpatch 在同函数内同步完成·未 patch 则 execute 越界 fail-loud）
_PATCH_PENDING = -(1 << 30)

# order_index 缺省槽位（无显式槽位的子排末·组内仍按 NodeRef 确定性）
_MAX_SLOT = 1 << 30


def is_control_flow_tag(op: int) -> bool:
    """是否控制流编译指令（IF/IFELSE/WHILE·compiler-internal·非 VM opcode）。"""
    return op in _CTRL_TAGS


def compile_graph(
    root: NodeRef,
    children_of: dict[NodeRef, list[NodeRef]],
    operator_of: dict[NodeRef, int],
    operand_of: dict[NodeRef, int],
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    immediate_of: dict[NodeRef, tuple[int, int]] | None = None,
    store_target_of: dict[NodeRef, int] | None = None,
) -> list[Instruction]:
    """沿 COMPOSES emit 指令序列（控制流根 lower 成 JZ/JMP·doc §三）。

    children_of     : COMPOSES 邻接（调用方已按 EDGE_COMPOSES 过滤 + 确定性排序·控制流子按 order_index 槽位）。
    operator_of     : 算子节点 → opcode symbol_id·或控制流编译指令 CTRL_IF/IFELSE/WHILE（A2）。
    operand_of      : 叶节点 → 变量 symbol_id（无子节点者须在此·LOAD 源）。
    max_depth       : 递归深度上限（超 → LoopClosureDefect）。
    immediate_of    : 常量叶节点 → (num, den)（A2·emit PUSH_IMM·自含常量非 env 预载）。
    store_target_of : STORE 节点（有子·值源）→ 目标变量 symbol_id（A2·控制流体回写 env）。

    普通算子节点：后序 emit 子 + opcode。控制流根：lower 成 JZ/JMP（回边在字节码非图）。
    确定性：子节点 emit 顺序 = children_of[root] 列表顺序（调用方排序）+ backpatch 同步。
    """
    assert_int(max_depth, _where="compile_graph.max_depth")
    if max_depth < 0:
        raise ValueError(f"max_depth 须 ≥ 0: {max_depth}")
    instrs: list[Instruction] = []
    _emit(root, children_of, operator_of, operand_of, 0, max_depth, frozenset(),
          instrs, immediate_of, store_target_of)
    return instrs


def _emit(
    node: NodeRef,
    children_of: dict[NodeRef, list[NodeRef]],
    operator_of: dict[NodeRef, int],
    operand_of: dict[NodeRef, int],
    depth: int,
    max_depth: int,
    path: frozenset[NodeRef],
    out: list[Instruction],
    immediate_of: dict[NodeRef, tuple[int, int]] | None,
    store_target_of: dict[NodeRef, int] | None,
) -> None:
    # 环检测（path 栈·COMPOSES 不允许环·A2 不变：回边在字节码非图·此处仍只见 DAG）
    if node in path:
        raise LoopClosureDefect(f"COMPOSES 环检测到: {node}")
    if depth > max_depth:
        raise LoopClosureDefect(
            f"COMPOSES 嵌套超限 max_depth={max_depth} at {node}"
        )
    # 常量叶（PUSH_IMM·A2）——无子·在 leaf-LOAD 之前判
    if immediate_of is not None and node in immediate_of:
        num, den = immediate_of[node]
        out.append(Instruction(OPCODE_PUSH_IMM, (num, den)))
        return
    children = children_of.get(node, [])
    new_path = path | {node}
    if children:
        if store_target_of is not None and node in store_target_of:
            # STORE 节点：emit 值源子（栈顶为待存值）+ STORE 目标变量（无须 operator_of·A2·控制流体回写 env）
            for c in children:
                _emit(c, children_of, operator_of, operand_of, depth + 1, max_depth,
                      new_path, out, immediate_of, store_target_of)
            out.append(Instruction(OPCODE_STORE, (store_target_of[node],)))
            return
        op = operator_of.get(node)
        if op is None:
            raise ValueError(
                f"组合节点缺 opcode（operator_of）: {node}"
            )
        if is_control_flow_tag(op):
            # 控制流根（IF/IFELSE/WHILE 编译指令）→ lower 成 JZ/JMP·非后序·doc §3.2
            _emit_control_flow(
                op, node, children, children_of, operator_of, operand_of,
                depth, max_depth, new_path, out, immediate_of, store_target_of,
            )
        else:
            # 普通算子：后序 emit 子节点 + 算子 opcode（栈机消费）
            for c in children:
                _emit(c, children_of, operator_of, operand_of, depth + 1, max_depth,
                      new_path, out, immediate_of, store_target_of)
            out.append(Instruction(op, ()))
    else:
        var = operand_of.get(node)
        if var is None:
            raise ValueError(
                f"叶节点缺 operand symbol（operand_of）: {node}"
            )
        out.append(Instruction(OPCODE_LOAD, (var,)))


def _emit_control_flow(
    tag: int,
    node: NodeRef,
    children: list[NodeRef],
    children_of: dict[NodeRef, list[NodeRef]],
    operator_of: dict[NodeRef, int],
    operand_of: dict[NodeRef, int],
    depth: int,
    max_depth: int,
    path: frozenset[NodeRef],
    out: list[Instruction],
    immediate_of: dict[NodeRef, tuple[int, int]] | None,
    store_target_of: dict[NodeRef, int] | None,
) -> None:
    """控制流根 lower（IF/IFELSE/WHILE 编译指令 → JZ/JMP·doc/重来_VM图灵完备与C6设计补充.md §3.2）。

    children 已按 order_index 槽位序排（COND 在前·THEN/BODY 次·ELSE 末·caller 排）。
    跳转 target = 指令表整数下标·two-pass backpatch（占位→emit→patch）确定性 bit-identical。
    回边（WHILE 的 JMP loop_head）在字节码非图·COMPOSES 仍 DAG·LoopClosureDefect 不触发。
    """
    if tag == CTRL_IF:
        if len(children) != 2:
            raise ValueError(f"IF 须 2 子 [COND, THEN]·得 {len(children)} at {node}")
        cond, then = children
        _emit(cond, children_of, operator_of, operand_of, depth + 1, max_depth, path,
              out, immediate_of, store_target_of)
        jz = len(out)
        out.append(Instruction(OPCODE_JZ, (_PATCH_PENDING,)))   # JZ then_skip
        _emit(then, children_of, operator_of, operand_of, depth + 1, max_depth, path,
              out, immediate_of, store_target_of)
        out[jz] = Instruction(OPCODE_JZ, (len(out),))           # then_skip backpatch
    elif tag == CTRL_IFELSE:
        if len(children) != 3:
            raise ValueError(f"IFELSE 须 3 子 [COND, THEN, ELSE]·得 {len(children)} at {node}")
        cond, then, els = children
        _emit(cond, children_of, operator_of, operand_of, depth + 1, max_depth, path,
              out, immediate_of, store_target_of)
        jz = len(out)
        out.append(Instruction(OPCODE_JZ, (_PATCH_PENDING,)))   # JZ else_branch
        _emit(then, children_of, operator_of, operand_of, depth + 1, max_depth, path,
              out, immediate_of, store_target_of)
        jmp = len(out)
        out.append(Instruction(OPCODE_JMP, (_PATCH_PENDING,)))  # JMP end
        out[jz] = Instruction(OPCODE_JZ, (len(out),))           # else_branch backpatch
        _emit(els, children_of, operator_of, operand_of, depth + 1, max_depth, path,
              out, immediate_of, store_target_of)
        out[jmp] = Instruction(OPCODE_JMP, (len(out),))         # end backpatch
    elif tag == CTRL_WHILE:
        if len(children) != 2:
            raise ValueError(f"WHILE 须 2 子 [COND, BODY]·得 {len(children)} at {node}")
        cond, body = children
        loop_head = len(out)
        _emit(cond, children_of, operator_of, operand_of, depth + 1, max_depth, path,
              out, immediate_of, store_target_of)
        jz = len(out)
        out.append(Instruction(OPCODE_JZ, (_PATCH_PENDING,)))   # JZ loop_exit
        _emit(body, children_of, operator_of, operand_of, depth + 1, max_depth, path,
              out, immediate_of, store_target_of)
        out.append(Instruction(OPCODE_JMP, (loop_head,)))       # 回边在字节码·不在图
        out[jz] = Instruction(OPCODE_JZ, (len(out),))           # loop_exit backpatch
    else:  # 不可达（is_control_flow_tag 已筛）
        raise ValueError(f"_emit_control_flow: 非控制流标签 {tag!r} at {node}")


def compile_from_edges(
    root: NodeRef,
    composes_edges: list[tuple[NodeRef, NodeRef]],
    operator_of: dict[NodeRef, int],
    operand_of: dict[NodeRef, int],
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    sort_key: Callable[[NodeRef], tuple[int, int]] | None = None,
    order_index_of: dict[NodeRef, int] | None = None,
    immediate_of: dict[NodeRef, tuple[int, int]] | None = None,
    store_target_of: dict[NodeRef, int] | None = None,
) -> list[Instruction]:
    """便捷入口：从扁平 COMPOSES 边列表构建 children_of 再 compile。

    sort_key：子节点排序键（默认 (space_id, local_id)·确定性）。COMPOSES 边无序·须排序保 bit-identical emit。
    order_index_of：dict[NodeRef, int]——控制流根槽位序（0=COND/1=THEN-BODY/2=ELSE·doc §3.3）。
      提供时按 (order_index, NodeRef) 排（控制流子有位置语义·非自然序）·否则按 sort_key。
    immediate_of / store_target_of：透传 compile_graph（A2 常量叶 / STORE 节点）。
    """
    children_of: dict[NodeRef, list[NodeRef]] = {}
    for parent, child in composes_edges:
        children_of.setdefault(parent, []).append(child)
    if order_index_of is not None:
        sk: Callable[[NodeRef], tuple] = lambda r: (order_index_of.get(r, _MAX_SLOT), r)
    elif sort_key is None:
        sk = lambda r: r  # (space_id, local_id) 自然序
    else:
        sk = sort_key
    for k in children_of:
        children_of[k] = sorted(children_of[k], key=sk)
    return compile_graph(root, children_of, operator_of, operand_of,
                         max_depth=max_depth,
                         immediate_of=immediate_of, store_target_of=store_target_of)
