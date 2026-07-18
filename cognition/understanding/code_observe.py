"""cognition.understanding.code_observe — 代码域(Python) AST → COMPOSES 程序建造者（A3·真瓶颈）。

打通 VM 图灵完备链上游：A1+A2 已让 VM 图灵完备但 production-inert（全仓 EDGE_COMPOSES 零建造者·
compile_graph 无 production caller·C6 无输入）。本模块把 Python AST → COMPOSES 树 +
operator_of/operand_of/immediate_of/store_target_of·落 EDGE_COMPOSES 边 + composes_attr 属性·
喂 compile_graph（经 ConceptGraph.read_composes_tree 读回）。

建造者产出契约（test_stage2.py 已固化）：(root, children_of, operator_of, operand_of,
immediate_of?, store_target_of?) → compile_graph。NodeRef=(space_id, local_id)。
双符号系统：图节点走 concept_index.ensure(surface, space_id)→ConceptRef（Hasher.h63 per-space
dedup）·VM 操作数走 make_variable(index)→axis_symbol_id（建造者自持 name→index map）。

支持子集白名单（doc/重来_A3_代码域observe设计补充.md §3.1）·超出 fail-loud UnsupportedConstruct：
  Module/FunctionDef body → SEQ（OPCODE_NOP 序连）
  Assign(单Name) → STORE·AugAssign → STORE(BinOp)·BinOp Add/Sub/Mult → 算子
  Compare Eq/Lt/Gt → 算子·UnaryOp(USub/UAdd on int 常量) → immediate 特例
  Constant(int/bool) → immediate·Name → operand·If → CTRL_IF/IFELSE·While → CTRL_WHILE
  Return(value) → HALT
  Div(`/`产float)/FloorDiv(`//`无FLOOR_DIV)/Mod/Pow/For/Call/ListComp/Lambda/Subscript/Attribute/
  Tuple多赋值/嵌套def → UnsupportedConstruct

铁律：纯整数（属性全 int·浮点字面量+`/`fail-loud 拒绝）/ 确定性（单一递归前序遍历·禁 ast.walk·
surface 序号+var index 共用遍历计数器·name→index 插入序 dict 禁 set→list）/ 核心无墙钟 /
不写死（AST 节点类型→opcode 通用映射·非硬编码特定程序）/ 最少边（复用 EDGE_COMPOSES·不增边类型）。
诚实边界：支持子集=算术+if/while+单赋值+一元负号+Return·真实 CodeSearchNet 函数多含 Call/For·
首版语料须手写或筛选子集·stable≠correct（VM 跑通≠对·靠独立 expected 验对错）。
"""
from __future__ import annotations

import ast
import sys

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.numeric.symbol_domain import (
    OPCODE_ADD, OPCODE_SUB, OPCODE_MUL, OPCODE_EQ, OPCODE_LT, OPCODE_GT,
    OPCODE_NOP, OPCODE_HALT, make_variable,
)
from pure_integer_ai.vm.graph_compile import CTRL_IF, CTRL_IFELSE, CTRL_WHILE
from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_OPERATOR
from pure_integer_ai.storage.composes_attr import (
    record_composes_attr, ATTR_OPERATOR, ATTR_CTRL_TAG, ATTR_OPERAND,
    ATTR_IMMEDIATE, ATTR_STORE_TARGET,
)
from pure_integer_ai.cognition.shared.types import ConceptRef


class UnsupportedConstruct(ValueError):
    """AST 构造超出支持子集（fail-loud·非静默跳过·非兜底）。

    代码域首版支持子集 = 算术+if/while+单赋值+一元负号+Return（doc §3.1）。
    超出子集 fail-loud 拒绝·诚实 scope（真实 CodeSearchNet 函数多含 Call/For·须筛选子集）。
    """


class _ComposesBuilder:
    """COMPOSES 树建造者（持 concept_index/edge_store/backend + 遍历状态）。

    单一递归前序遍历（_build_stmt/_build_expr 递归·先建当前节点再递归子）·
    surface 序号 + var index 共用此前序遍历计数器（确定性 bit-identical）。
    """

    def __init__(self, *, concept_index, edge_store, backend,
                 space_id: int, source: int, root_ref: ConceptRef) -> None:
        self._ci = concept_index
        self._es = edge_store
        self._b = backend
        self._space_id = space_id
        self._source = source
        self._root = root_ref
        self._seq = 0                          # AST 节点序号（前序遍历计数器）
        self._var_index: dict[str, int] = {}   # name→index 插入序 dict（禁 set→list）
        self._next_var = 0                     # 变量 index 计数器
        self._func_name = ""
        self._py_major = sys.version_info.major   # 钉死 Python 版本位（防跨版本 AST 字段序变）

    def build(self, funcdef: ast.FunctionDef) -> ConceptRef:
        """建函数体 COMPOSES 树·root=struct_ref（SEQ NOP 序连顶层语句）。"""
        self._func_name = funcdef.name
        # 参数先分配 var index（FunctionDef.args 顺序·统一 make_variable·E5 一致性）
        for arg in funcdef.args.args:
            self._var_sid(arg.arg)
        # root = struct_ref = SEQ NOP（函数根·序连顶层语句）
        record_composes_attr(self._b, ref=self._root, kind=ATTR_OPERATOR, int_a=OPCODE_NOP)
        for i, stmt in enumerate(funcdef.body):
            child = self._build_stmt(stmt)
            self._edge(self._root, child, i)
        return self._root

    # ---- 节点/变量/边原语 ----

    def _new_node(self, node_type: str) -> ConceptRef:
        """建 AST 节点 ConceptRef（surface 含 func_name+root_lid+前序序号+类型+py_major·per-space dedup）。

        root_lid（struct_ref local_id）隔离不同段的 AST 节点——同 func_name 不同函数体的
        节点不碰撞误合并（如 return True 与 return False 的 IMM 节点·doc §3.3）。
        """
        self._seq += 1
        surface = f"__ast_{self._func_name}_{self._root[1]}_{self._seq}_{node_type}_{self._py_major}"
        return self._ci.ensure(surface, space_id=self._space_id,
                               tier=TIER_PRIMARY, node_type=NODE_OPERATOR)

    def _var_sid(self, name: str) -> int:
        """变量名→make_variable(index) symbol_id（插入序 dict·首次出现分配·确定性）。"""
        if name not in self._var_index:
            self._var_index[name] = self._next_var
            self._next_var += 1
        return make_variable(self._var_index[name])

    def _edge(self, parent: ConceptRef, child: ConceptRef, order_index: int) -> None:
        """落 EDGE_COMPOSES 边（父→子·order_index=槽位序·source=SOURCE_*·epi=STRUCTURED）。"""
        self._es.add(space_id_from=parent[0], local_id_from=parent[1],
                     space_id_to=child[0], local_id_to=child[1],
                     edge_type=EDGE_COMPOSES, strength=1, source=self._source,
                     epistemic_origin=EPI_STRUCTURED, order_index=order_index)

    # ---- 语句 ----

    def _build_block(self, stmts: list[ast.stmt]) -> ConceptRef:
        """语句块 → 单语句直返 / 多语句 SEQ NOP 序连（test_stage2 WHILE body 样板）。"""
        if not stmts:
            raise UnsupportedConstruct("空语句块不支持")
        if len(stmts) == 1:
            return self._build_stmt(stmts[0])
        seq = self._new_node("SEQ")
        record_composes_attr(self._b, ref=seq, kind=ATTR_OPERATOR, int_a=OPCODE_NOP)
        for i, s in enumerate(stmts):
            self._edge(seq, self._build_stmt(s), i)
        return seq

    def _build_stmt(self, node: ast.stmt) -> ConceptRef:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise UnsupportedConstruct("Assign 须单 Name 目标（多赋值/Tuple 不支持）")
            target_sid = self._var_sid(node.targets[0].id)
            store = self._new_node("STORE")
            record_composes_attr(self._b, ref=store, kind=ATTR_STORE_TARGET, int_a=target_sid)
            self._edge(store, self._build_expr(node.value), 0)
            return store
        if isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                raise UnsupportedConstruct("AugAssign 须 Name 目标")
            op = self._binop_opcode(node.op)   # Add/Sub/Mul·Div 等拒绝
            target_sid = self._var_sid(node.target.id)
            store = self._new_node("STORE")
            record_composes_attr(self._b, ref=store, kind=ATTR_STORE_TARGET, int_a=target_sid)
            bnode = self._new_node("BINOP")
            record_composes_attr(self._b, ref=bnode, kind=ATTR_OPERATOR, int_a=op)
            self._edge(bnode, self._build_expr(node.target), 0)   # Name(target) 叶
            self._edge(bnode, self._build_expr(node.value), 1)
            self._edge(store, bnode, 0)
            return store
        if isinstance(node, ast.If):
            cond = self._build_expr(node.test)
            then_blk = self._build_block(node.body)
            if node.orelse:
                else_blk = self._build_block(node.orelse)
                ifnode = self._new_node("IFELSE")
                record_composes_attr(self._b, ref=ifnode, kind=ATTR_CTRL_TAG, int_a=CTRL_IFELSE)
                self._edge(ifnode, cond, 0)
                self._edge(ifnode, then_blk, 1)
                self._edge(ifnode, else_blk, 2)
            else:
                ifnode = self._new_node("IF")
                record_composes_attr(self._b, ref=ifnode, kind=ATTR_CTRL_TAG, int_a=CTRL_IF)
                self._edge(ifnode, cond, 0)
                self._edge(ifnode, then_blk, 1)
            return ifnode
        if isinstance(node, ast.While):
            if node.orelse:
                raise UnsupportedConstruct("While-else 不支持")
            cond = self._build_expr(node.test)
            body_blk = self._build_block(node.body)
            wnode = self._new_node("WHILE")
            record_composes_attr(self._b, ref=wnode, kind=ATTR_CTRL_TAG, int_a=CTRL_WHILE)
            self._edge(wnode, cond, 0)
            self._edge(wnode, body_blk, 1)
            return wnode
        if isinstance(node, ast.Return):
            rnode = self._new_node("RET")
            record_composes_attr(self._b, ref=rnode, kind=ATTR_OPERATOR, int_a=OPCODE_HALT)
            if node.value is not None:
                self._edge(rnode, self._build_expr(node.value), 0)
            return rnode
        raise UnsupportedConstruct(f"语句不支持: {type(node).__name__}")

    # ---- 表达式 ----

    def _build_expr(self, node: ast.expr) -> ConceptRef:
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):           # bool 先于 int（bool 是 int 子类·显式规范化）
                num = 1 if v else 0
            elif isinstance(v, int):
                num = v
            else:
                raise UnsupportedConstruct(
                    f"Constant 类型不支持（纯整数铁律·float/str/None 拒绝）: {type(v).__name__}")
            leaf = self._new_node("IMM")
            record_composes_attr(self._b, ref=leaf, kind=ATTR_IMMEDIATE, int_a=num, int_b=1)
            return leaf
        if isinstance(node, ast.Name):
            sid = self._var_sid(node.id)
            leaf = self._new_node("VAR")
            record_composes_attr(self._b, ref=leaf, kind=ATTR_OPERAND, int_a=sid)
            return leaf
        if isinstance(node, ast.UnaryOp):
            # 仅 USub/UAdd on int 常量 → immediate(±num,1) 特例（负数字面量是 UnaryOp 非 Constant）
            if (isinstance(node.op, (ast.USub, ast.UAdd))
                    and isinstance(node.operand, ast.Constant)
                    and isinstance(node.operand.value, int)
                    and not isinstance(node.operand.value, bool)):
                num = node.operand.value
                if isinstance(node.op, ast.USub):
                    num = -num
                leaf = self._new_node("IMM")
                record_composes_attr(self._b, ref=leaf, kind=ATTR_IMMEDIATE, int_a=num, int_b=1)
                return leaf
            raise UnsupportedConstruct("UnaryOp 仅支持 USub/UAdd on int 常量（Not 等不支持）")
        if isinstance(node, ast.BinOp):
            op = self._binop_opcode(node.op)
            bnode = self._new_node("BINOP")
            record_composes_attr(self._b, ref=bnode, kind=ATTR_OPERATOR, int_a=op)
            self._edge(bnode, self._build_expr(node.left), 0)
            self._edge(bnode, self._build_expr(node.right), 1)
            return bnode
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise UnsupportedConstruct("Compare 仅支持单比较（a<b<c 不支持）")
            op = self._compare_opcode(node.ops[0])
            cnode = self._new_node("CMP")
            record_composes_attr(self._b, ref=cnode, kind=ATTR_OPERATOR, int_a=op)
            self._edge(cnode, self._build_expr(node.left), 0)
            self._edge(cnode, self._build_expr(node.comparators[0]), 1)
            return cnode
        raise UnsupportedConstruct(f"表达式不支持: {type(node).__name__}")

    # ---- 运算符映射（通用·非硬编码特定程序） ----

    def _binop_opcode(self, op: ast.operator) -> int:
        if isinstance(op, ast.Add):
            return OPCODE_ADD
        if isinstance(op, ast.Sub):
            return OPCODE_SUB
        if isinstance(op, ast.Mult):
            return OPCODE_MUL
        # Div(`/`)产 float·FloorDiv(`//`)无 FLOOR_DIV（映 OPCODE_DIV 语义错）·Mod/Pow 无 opcode → 拒绝
        raise UnsupportedConstruct(
            f"BinOp 运算符不支持（/ 产 float·// 无 FLOOR_DIV·% ** 无 opcode）: {type(op).__name__}")

    def _compare_opcode(self, op: ast.cmpop) -> int:
        if isinstance(op, ast.Eq):
            return OPCODE_EQ
        if isinstance(op, ast.Lt):
            return OPCODE_LT
        if isinstance(op, ast.Gt):
            return OPCODE_GT
        raise UnsupportedConstruct(
            f"Compare 运算符不支持（LtE/GtE/NotEq/In/Is 不支持）: {type(op).__name__}")


def build_composes_from_source(code_source: str, *, concept_index, edge_store,
                               backend, space_id: int, source: int,
                               root_ref: ConceptRef) -> ConceptRef:
    """Python 源码 → COMPOSES 树（落 EDGE_COMPOSES 边 + composes_attr 属性）·返 root=struct_ref。

    code_source : Python 源码字符串（Segment.code_source·MODALITY_CODE）。
    source      : SOURCE_* 枚举（edge_store·代码域 SOURCE_CODE）。
    root_ref    : 该段 struct_ref（= 函数根 = COMPOSES 根·解致命#3·dag_path.sink=struct_ref=root）。

    首版取第一个 FunctionDef（一段一函数·doc §3.10）。超出支持子集 fail-loud UnsupportedConstruct。
    """
    assert_int(space_id, source, _where="build_composes_from_source")
    assert_no_float(space_id, source, _where="build_composes_from_source")
    try:
        tree = ast.parse(code_source)
    except SyntaxError as e:
        raise UnsupportedConstruct(f"源码语法错误: {e}") from e
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        raise UnsupportedConstruct("代码源须含 FunctionDef（首版一段一函数）")
    funcdef = funcs[0]   # 首版取第一个函数（多函数语料须分段·一段一函数）
    builder = _ComposesBuilder(concept_index=concept_index, edge_store=edge_store,
                               backend=backend, space_id=space_id,
                               source=source, root_ref=root_ref)
    return builder.build(funcdef)
