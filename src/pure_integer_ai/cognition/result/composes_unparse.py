"""cognition.result.composes_unparse — COMPOSES 程序子树 → 源码串序化器（#730 路径 W·代码模态）。

读 ConceptGraph.read_composes_tree 的 5 dict（children_of / operator_of / operand_of / immediate_of /
store_target_of）→ DFS 前序 → 按 ATTR kind 分派词形 → 拼接 Python-ish 源码串（函数 body 语句串）。

**#730 路径 W 用途**（代码模态 task-driven episode·Mode A 构造性·formal_train._run_task_driven_generate 代码模态
分派）：unparse_composes(struct_ref) → 源码串 → ast normalize == code_source normalize → verified
（skeleton 派生自 code_source·构造性必然·同 arith skeleton(args)==expected·formal_train.py:1944 范式·
**stable≠correct·结构对称非字面还原·非真生成·真生成须路径 X 跨模态 defer**）。

**写读序对称 bit-identical**（最大风险·task doc 风险 1）：复用 read_composes_tree 的 children_of order_index
排序键（graph_view.py read_composes_tree·(order_index[None→1<<30], space_id, local_id)）→ DFS 前序 ==
code_observe._seq 建树前序（code_observe 单一递归前序·先建当前节点再递归子·surface 序号 = 前序遍历计数器
code_observe.py:100）→ 两跑逐结构一致·同输入同输出。

**ATTR kind 分派**（CTRL_TAG 优先·守 read 优先级 graph_view.py:295-298：read 把 ATTR_CTRL_TAG 优先塞 operator_of）：
  - STORE 节点（node in store_target_of·operator_of 不含）：var{target_idx} = <value_expr>（Assign 形式）
  - CTRL 根（is_control_flow_tag(operator_of[node])·CTRL_IF/IFELSE/WHILE·负数 sentinel）：if/elif-else/while 词形 + 缩进体
  - HALT（OPCODE_HALT·Return 语句）：return <expr> / return
  - NOP 有子（OPCODE_NOP·SEQ 块·函数根 / then / else / while-body）：子语句序连
  - 二元算子（OPCODE_ADD/SUB/MUL/DIV/EQ/LT/GT·expr 上下文）：(<left> <op> <right>)·paren 透明
    （ast.parse 剥冗余括号·(a+b) 与 a+b 同 AST·结构保持）
  - OPERAND 叶（node in operand_of·无子）：var{index_of(sid)}（人造名·原始名丢失·决断 3 诚实标）
  - IMMEDIATE 叶（node in immediate_of·无子）：{num}（den 恒 1·code 模态 code_observe 不产 den>1·arith 有理 defer）
**非结构 kind 不消费**（反 theater·只读 read_composes_tree 5 结构 dict·ATTR_OPERATOR_DEF/ARITY/ORIGIN/SLOT_ROLE/
RELATION_PRIMITIVE 非结构 kind 不进 5 dict·read_composes_tree 忽略·消费非重生）。

铁律：纯整数（opcode/var index/immediate 全 int·assert_int 守·零浮点零除法）/ 确定性（DFS 前序 + read 排序键·
  bit-identical·sorted 无随机）/ 单向依赖（cognition=result L5·import storage(2)/numeric(1)/vm(3) 全向下·
  不调 training L7 execute_composes_value·序化纯读）/ §8.5 不碰 schema（纯读 composes_attr·零写盘）/
  不写死（opcode→词形通用映射表 _BINOP_WORDS·非硬编码特定程序）/ 限深环保护（_MAX_DEPTH·同 graph_compile）。
诚实边界：stable≠correct（结构对称非字面还原·变量名 var{index} 人造·原始名还原 defer observe 侧 index→name 表）/
  Mode A 构造性（unparse(__prog_*)==code_source 构造性必然·非真生成·真生成须路径 X defer）/ 仅白名单内单函数
  （For/Call/class/嵌套 def 须 VM 扩·决断 6 defer）/ 仅 CODE 模态（arith 走 execute 非序化·modality 参数路径 X 跟进）/
  AugAssign 歧义（code_observe 对 x=x+1 与 x+=1 产同构树·不可区分·统一 Assign 形式·code_source 须 Assign 形式）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.numeric.symbol_domain import (
    OPCODE_ADD, OPCODE_SUB, OPCODE_MUL, OPCODE_DIV, OPCODE_EQ, OPCODE_LT, OPCODE_GT,
    OPCODE_HALT, OPCODE_NOP, index_of,
)
from pure_integer_ai.vm.graph_compile import (
    is_control_flow_tag, CTRL_IF, CTRL_IFELSE, CTRL_WHILE, LoopClosureDefect,
)
from pure_integer_ai.cognition.shared.types import ConceptRef

# opcode → 代码模态二元算子词形（通用映射表·非硬编码特定程序·symmetric to code_observe._binop_opcode）
_BINOP_WORDS: dict[int, str] = {
    OPCODE_ADD: "+", OPCODE_SUB: "-", OPCODE_MUL: "*", OPCODE_DIV: "/",
    OPCODE_EQ: "==", OPCODE_LT: "<", OPCODE_GT: ">",
}

# 限深环保护（同 graph_compile.DEFAULT_MAX_DEPTH / read_composes_tree._COMPOSES_MAX_DEPTH·防病态深递归）。
# read_composes_tree 已 visited 防 COMPOSES 环·children_of 是 DAG·本守深递归栈爆 + 病态嵌套 fail-loud。
_MAX_DEPTH = 64


def unparse_composes(graph, root_ref: ConceptRef) -> str:
    """COMPOSES 子树（函数根 SEQ NOP）→ Python-ish body 源码串（#730 路径 W·代码模态）。

    graph    : ConceptGraph（read_composes_tree 读 5 dict·graph_view.py:265）。
    root_ref : 函数 struct_ref（= COMPOSES 根·code_observe.build 返的 SEQ NOP 根·子 = funcdef.body 顶层语句）。

    返 body 语句串（行 join "\n"·缩进感知·root 的子各一语句·控制流体缩进 +1 层）。
    **body-only**（剥 FunctionDef 包装）：Mode A 验 body 计算结构·code_observe 丢签名名（函数名 + arg 名）但保
    body 结构（arg→var index 映射·var index 0..n_args-1 = params·body Name 经 AST normalize 按位置对齐）。
    签名（函数名 + arg 列表）排除比（code_observe 部分丢·保 body·非验签名·诚实）。

    限深超 _MAX_DEPTH → LoopClosureDefect（fail-loud·不静默截断·同 graph_compile:139）。
    """
    assert_int(root_ref[0], root_ref[1], _where="unparse_composes.root_ref")
    (children_of, operator_of, operand_of,
     immediate_of, store_target_of) = graph.read_composes_tree(root_ref)
    lines: list[str] = []
    # root = 函数根 SEQ NOP·子 = 顶层语句（funcdef.body）·各 _emit_stmt·indent=0
    for stmt in children_of.get(root_ref, []):
        _emit_stmt(stmt, children_of, operator_of, operand_of,
                   immediate_of, store_target_of, 0, 0, lines)
    return "\n".join(lines)


def _emit_block(node: ConceptRef, children_of, operator_of, operand_of,
                immediate_of, store_target_of, indent: int, depth: int,
                lines: list[str]) -> None:
    """语句块：node 若 SEQ NOP（多语句）→ 子各 _emit_stmt·否则 node 自身单语句 _emit_stmt。

    控制流体（then/else/while-body）来自 code_observe._build_block（code_observe.py:121）：单语句直返·
    多语句返 SEQ NOP 容器。两态本函数统一处理（block 入口·indent 已 +1 由 caller）。
    """
    if depth > _MAX_DEPTH:
        raise LoopClosureDefect(f"composes_unparse: 嵌套超限 max_depth={_MAX_DEPTH} at {node}")
    children = children_of.get(node)
    op = operator_of.get(node)
    if op == OPCODE_NOP and children:
        # SEQ NOP 容器·多语句·子各一语句
        for c in children:
            _emit_stmt(c, children_of, operator_of, operand_of,
                       immediate_of, store_target_of, indent, depth + 1, lines)
    else:
        # 单语句块（node 自身是一语句·非 SEQ 容器）
        _emit_stmt(node, children_of, operator_of, operand_of,
                   immediate_of, store_target_of, indent, depth + 1, lines)


def _emit_stmt(node: ConceptRef, children_of, operator_of, operand_of,
               immediate_of, store_target_of, indent: int, depth: int,
               lines: list[str]) -> None:
    """语句节点 → 一行（append lines）·按 ATTR kind 分派（STORE / CTRL / HALT / NOP-block）。"""
    if depth > _MAX_DEPTH:
        raise LoopClosureDefect(f"composes_unparse: 嵌套超限 max_depth={_MAX_DEPTH} at {node}")
    pad = "    " * indent
    # STORE 节点（Assign 形式·node in store_target_of·operator_of 不含 STORE）
    if node in store_target_of:
        target_idx = index_of(store_target_of[node])
        children = children_of.get(node, [])
        if len(children) != 1:
            raise ValueError(f"序化器 STORE 须 1 子（值源）·得 {len(children)} at {node}")
        rhs = _emit_expr(children[0], children_of, operator_of, operand_of,
                         immediate_of, store_target_of, depth + 1)
        # 统一 Assign 形式（AugAssign x+=1 与 x=x+1 同构树不可区分·诚实边界·见模块 docstring）
        lines.append(f"{pad}var{target_idx} = {rhs}")
        return
    op = operator_of.get(node)
    # CTRL 根（IF/IFELSE/WHILE·负数 sentinel·is_control_flow_tag）
    if op is not None and is_control_flow_tag(op):
        children = children_of.get(node, [])
        _emit_ctrl(op, node, children, children_of, operator_of, operand_of,
                   immediate_of, store_target_of, indent, depth, lines)
        return
    # Return（OPCODE_HALT）
    if op == OPCODE_HALT:
        children = children_of.get(node, [])
        if children:
            rhs = _emit_expr(children[0], children_of, operator_of, operand_of,
                             immediate_of, store_target_of, depth + 1)
            lines.append(f"{pad}return {rhs}")
        else:
            lines.append(f"{pad}return")
        return
    # SEQ NOP 块（OPCODE_NOP 有子·嵌套块·非根·caller _emit_block 已处理多数·此处兜底直块根场景）
    if op == OPCODE_NOP:
        _emit_block(node, children_of, operator_of, operand_of,
                    immediate_of, store_target_of, indent, depth, lines)
        return
    # 叶节点 / 算子节点作语句（code_observe 不产裸 expr 语句·防御 fail-loud）
    raise ValueError(f"序化器：节点非支持语句形态 at {node} op={op}")


def _emit_ctrl(op: int, node: ConceptRef, children, children_of, operator_of, operand_of,
               immediate_of, store_target_of, indent: int, depth: int,
               lines: list[str]) -> None:
    """控制流根 → if / if-else / while 词形 + 缩进体（槽位序 0=COND / 1=THEN-BODY / 2=ELSE）。"""
    pad = "    " * indent
    if op == CTRL_IF:
        if len(children) != 2:
            raise ValueError(f"序化器 IF 须 2 子 [COND,THEN]·得 {len(children)} at {node}")
        cond, then_blk = children[0], children[1]
        cond_s = _emit_expr(cond, children_of, operator_of, operand_of,
                            immediate_of, store_target_of, depth + 1)
        lines.append(f"{pad}if {cond_s}:")
        _emit_block(then_blk, children_of, operator_of, operand_of,
                    immediate_of, store_target_of, indent + 1, depth + 1, lines)
    elif op == CTRL_IFELSE:
        if len(children) != 3:
            raise ValueError(f"序化器 IFELSE 须 3 子 [COND,THEN,ELSE]·得 {len(children)} at {node}")
        cond, then_blk, else_blk = children[0], children[1], children[2]
        cond_s = _emit_expr(cond, children_of, operator_of, operand_of,
                            immediate_of, store_target_of, depth + 1)
        lines.append(f"{pad}if {cond_s}:")
        _emit_block(then_blk, children_of, operator_of, operand_of,
                    immediate_of, store_target_of, indent + 1, depth + 1, lines)
        lines.append(f"{pad}else:")
        _emit_block(else_blk, children_of, operator_of, operand_of,
                    immediate_of, store_target_of, indent + 1, depth + 1, lines)
    elif op == CTRL_WHILE:
        if len(children) != 2:
            raise ValueError(f"序化器 WHILE 须 2 子 [COND,BODY]·得 {len(children)} at {node}")
        cond, body_blk = children[0], children[1]
        cond_s = _emit_expr(cond, children_of, operator_of, operand_of,
                            immediate_of, store_target_of, depth + 1)
        lines.append(f"{pad}while {cond_s}:")
        _emit_block(body_blk, children_of, operator_of, operand_of,
                    immediate_of, store_target_of, indent + 1, depth + 1, lines)
    else:  # 不可达（is_control_flow_tag 已筛 IF/IFELSE/WHILE）
        raise ValueError(f"序化器 _emit_ctrl：非控制流标签 {op} at {node}")


def _emit_expr(node: ConceptRef, children_of, operator_of, operand_of,
               immediate_of, store_target_of, depth: int) -> str:
    """表达式节点 → 词形串（IMM / VAR / 二元算子）·paren 包裹二元（透明·ast.parse 剥冗余括号）。"""
    if depth > _MAX_DEPTH:
        raise LoopClosureDefect(f"composes_unparse: 嵌套超限 max_depth={_MAX_DEPTH} at {node}")
    # IMMEDIATE 叶（常量·code 模态 den 恒 1·code_observe:202/219）
    if node in immediate_of:
        num, den = immediate_of[node]
        assert_int(num, den, _where="composes_unparse._emit_expr.immediate")
        if den != 1:
            # code 模态不产 den>1（code_observe 不建·arith 有理专用）·fail-loud 防 arith 误调
            raise ValueError(f"序化器 immediate den>1 非 code 模态（code den 恒 1·arith 有理 defer）: {num}/{den}")
        return str(num)
    # OPERAND 叶（变量·人造名 var{index}·原始名丢失·诚实边界）
    if node in operand_of:
        sid = operand_of[node]
        return f"var{index_of(sid)}"
    # 二元算子（OPCODE_ADD/SUB/MUL/DIV/EQ/LT/GT）
    op = operator_of.get(node)
    if op is not None and not is_control_flow_tag(op) and op not in (OPCODE_HALT, OPCODE_NOP):
        word = _BINOP_WORDS.get(op)
        if word is None:
            raise ValueError(f"序化器：opcode 无代码词形 {op} at {node}")
        children = children_of.get(node, [])
        if len(children) != 2:
            raise ValueError(f"序化器二元算子须 2 子·得 {len(children)} at {node}")
        left = _emit_expr(children[0], children_of, operator_of, operand_of,
                          immediate_of, store_target_of, depth + 1)
        right = _emit_expr(children[1], children_of, operator_of, operand_of,
                           immediate_of, store_target_of, depth + 1)
        return f"({left} {word} {right})"
    raise ValueError(f"序化器：节点非支持表达式形态 at {node} op={op}")
