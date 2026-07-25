"""Stage 2 验收门测试：VM + 算法原语（决策9自建·PR零损失/A2无环/A4对齐/closure纯净）。

覆盖（doc/重来_落地规划与实施顺序.md §六 Stage 2 验收门）：
  - VM：graph_compile 沿 COMPOSES 后序 emit + 限深环保护 + dispatch + vm_core 栈机 step_limit
  - A2：Kahn 拓扑分层（无环序 + 层 + 环检测·确定性）
  - A3：PR 线性系统零损失（(I−αA^T)x=(1−α)e 残差=0）+ 线性性 x=Σx_s + B2 迭代收敛 + B1→B2 回退
  - A4：LCS 对齐 + pairwise 折叠
  - closure：按 edge_type 分发 + 纯净性 filter + CLOSURE 派生不存储
"""
from __future__ import annotations

import pytest

from pure_integer_ai.crosscut.integer.rational import Rational, ZERO, ONE, make
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.numeric.symbol_domain import (
    OPCODE_ADD, OPCODE_SUB, OPCODE_MUL, OPCODE_DIV, OPCODE_EQ, OPCODE_LT,
    OPCODE_GT, OPCODE_LOAD, OPCODE_STORE, OPCODE_NOP, make_variable,
    OPCODE_PUSH_IMM, OPCODE_JZ, OPCODE_JMP, OPCODE_HALT, symbol_to_opcode,
)
from pure_integer_ai.vm.graph_compile import (
    compile_graph, compile_from_edges, Instruction, LoopClosureDefect,
    CTRL_IF, CTRL_IFELSE, CTRL_WHILE, is_control_flow_tag,
)
from pure_integer_ai.vm.dispatch import (
    dispatch_binary, rdiv, reciprocal, calc_role, CALC_ROLE_ANALYTICAL,
    is_binary_opcode,
)
from pure_integer_ai.vm.vm_core import execute, StepLimitExceeded, DEFAULT_STEP_LIMIT
from pure_integer_ai.algorithm.a2_topology import kahn_topo, max_layer, predecessors_by_layer
from pure_integer_ai.algorithm.a3_personal_rank import (
    build_matrix, solve_exact, solve_exact_multi, solve_iterative,
    personal_rank, PRMatrix, PRSingular, DEFAULT_ALPHA_NUM, DEFAULT_ALPHA_DEN,
)
from pure_integer_ai.algorithm.a4_alignment import (
    lcs, lcs_score, pairwise_fold, alignment_matches,
)
from pure_integer_ai.algorithm.closure import transitive_closure, reachable
from pure_integer_ai.storage.edge_types import (
    EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO, EDGE_COMPOSES, EDGE_CLOSURE,
)
from pure_integer_ai.storage.edge_store import SUBTYPE_PURE_ALIAS, SUBTYPE_METAPHOR


# ============ VM ============

def test_dispatch_binary_rational_exact():
    a, b = make(1, 2), make(1, 3)
    assert rational.eq(dispatch_binary(OPCODE_ADD, a, b), make(5, 6))
    assert rational.eq(dispatch_binary(OPCODE_SUB, a, b), make(1, 6))
    assert rational.eq(dispatch_binary(OPCODE_MUL, a, b), make(1, 6))
    # DIV 有理倒数精确（非 fixed_point 近似）
    assert rational.eq(dispatch_binary(OPCODE_DIV, a, b), make(3, 2))
    assert rational.eq(dispatch_binary(OPCODE_EQ, a, make(2, 4)), ONE)
    assert rational.eq(dispatch_binary(OPCODE_EQ, a, b), ZERO)
    assert rational.eq(dispatch_binary(OPCODE_LT, b, a), ONE)   # 1/3 < 1/2
    assert rational.eq(dispatch_binary(OPCODE_GT, a, b), ONE)


def test_dispatch_div_by_zero():
    with pytest.raises(ZeroDivisionError):
        dispatch_binary(OPCODE_DIV, ONE, ZERO)


# ============ A1 控制流（图灵完备·doc/重来_VM图灵完备与C6设计补充.md §二） ============

def test_push_imm_rational():
    """PUSH_IMM(num,den) → make(num,den)·有理立即数（常量/循环界）。"""
    assert rational.eq(execute([Instruction(OPCODE_PUSH_IMM, (3, 2))], {}), make(3, 2))
    assert rational.eq(execute([Instruction(OPCODE_PUSH_IMM, (-3, 2))], {}), make(-3, 2))
    with pytest.raises(ZeroDivisionError):
        execute([Instruction(OPCODE_PUSH_IMM, (1, 0))], {})


def test_jz_jump_and_fallthrough():
    """JZ：栈顶零则跳 target·非零则 pc+1（JZ 消费条件·待存值须分离另压）。"""
    marker = make_variable(0)
    # 条件=零 → 跳过 STORE·env 无 marker
    execute([Instruction(OPCODE_PUSH_IMM, (1, 1)),     # 待存值 ONE
             Instruction(OPCODE_PUSH_IMM, (0, 1)),     # 条件 zero
             Instruction(OPCODE_JZ, (4,)),             # 零→跳 HALT(4)
             Instruction(OPCODE_STORE, (marker,)),     # 跳过
             Instruction(OPCODE_HALT, ())], {})
    # 条件=非零 → 落到 STORE·env[marker]=ONE
    env = {}
    execute([Instruction(OPCODE_PUSH_IMM, (1, 1)),     # 待存值 ONE
             Instruction(OPCODE_PUSH_IMM, (1, 1)),     # 条件 nonzero
             Instruction(OPCODE_JZ, (4,)),             # 非零→落到 STORE(3)
             Instruction(OPCODE_STORE, (marker,)),     # pop ONE→env[marker]
             Instruction(OPCODE_HALT, ())], env)
    assert rational.eq(env[marker], ONE)


def test_jmp_and_halt_skip_subsequent():
    """JMP 无条件跳·HALT 终止（被跳过/之后的指令不执行）。"""
    marker = make_variable(0)
    env = {}
    execute([Instruction(OPCODE_JMP, (2,)),            # 跳过 STORE
             Instruction(OPCODE_STORE, (marker,)),
             Instruction(OPCODE_HALT, ())], env)
    assert marker not in env                            # STORE 被跳过


def test_jump_target_out_of_range_raises():
    """跳转目标越界 fail-loud（_check_jump_target·非静默回绕）。"""
    with pytest.raises(IndexError):
        execute([Instruction(OPCODE_JMP, (5,))], {})   # len=1 target=5 越界
    with pytest.raises(IndexError):
        execute([Instruction(OPCODE_PUSH_IMM, (0, 1)),  # push zero
                 Instruction(OPCODE_JZ, (9,))], {})     # len=2 target=9·taken→越界


def test_loop_counter_sum_turing_complete():
    """循环计数 sum(1..5)=15 via JZ/JMP 回边——**图灵完备证明（迭代可用）**。"""
    i = make_variable(0)
    acc = make_variable(1)
    prog = [
        Instruction(OPCODE_PUSH_IMM, (1, 1)),   # 0: i=1
        Instruction(OPCODE_STORE, (i,)),
        Instruction(OPCODE_PUSH_IMM, (0, 1)),    # 2: acc=0
        Instruction(OPCODE_STORE, (acc,)),
        Instruction(OPCODE_LOAD, (i,)),          # 4: loop_head
        Instruction(OPCODE_PUSH_IMM, (6, 1)),
        Instruction(OPCODE_LT, ()),              # 6: i<6
        Instruction(OPCODE_JZ, (17,)),           # 7: i>=6 → exit(17)
        Instruction(OPCODE_LOAD, (acc,)),        # 8
        Instruction(OPCODE_LOAD, (i,)),          # 9
        Instruction(OPCODE_ADD, ()),             # 10: acc+i
        Instruction(OPCODE_STORE, (acc,)),       # 11: acc+=i
        Instruction(OPCODE_LOAD, (i,)),          # 12
        Instruction(OPCODE_PUSH_IMM, (1, 1)),    # 13
        Instruction(OPCODE_ADD, ()),             # 14: i+1
        Instruction(OPCODE_STORE, (i,)),         # 15: i+=1
        Instruction(OPCODE_JMP, (4,)),           # 16: → loop_head
        Instruction(OPCODE_LOAD, (acc,)),        # 17: exit
        Instruction(OPCODE_HALT, ()),            # 18
    ]
    result = execute(prog, {i: ZERO, acc: ZERO}, step_limit=1000)
    assert rational.eq(result, make(15, 1))


def test_loop_equivalence_with_closed_form():
    """闭式 n*(n+1)/2 vs 迭代 sum(1..n) 两路一致——C6 Mode B 自证交叉验证原语。"""
    n = 5
    closed = execute([
        Instruction(OPCODE_PUSH_IMM, (n, 1)),
        Instruction(OPCODE_PUSH_IMM, (n + 1, 1)),
        Instruction(OPCODE_MUL, ()),
        Instruction(OPCODE_PUSH_IMM, (2, 1)),
        Instruction(OPCODE_DIV, ()),
        Instruction(OPCODE_HALT, ()),
    ], {})
    i = make_variable(0)
    acc = make_variable(1)
    loop_prog = [
        Instruction(OPCODE_PUSH_IMM, (1, 1)),
        Instruction(OPCODE_STORE, (i,)),
        Instruction(OPCODE_PUSH_IMM, (0, 1)),
        Instruction(OPCODE_STORE, (acc,)),
        Instruction(OPCODE_LOAD, (i,)),
        Instruction(OPCODE_PUSH_IMM, (n + 1, 1)),
        Instruction(OPCODE_LT, ()),
        Instruction(OPCODE_JZ, (17,)),
        Instruction(OPCODE_LOAD, (acc,)),
        Instruction(OPCODE_LOAD, (i,)),
        Instruction(OPCODE_ADD, ()),
        Instruction(OPCODE_STORE, (acc,)),
        Instruction(OPCODE_LOAD, (i,)),
        Instruction(OPCODE_PUSH_IMM, (1, 1)),
        Instruction(OPCODE_ADD, ()),
        Instruction(OPCODE_STORE, (i,)),
        Instruction(OPCODE_JMP, (4,)),
        Instruction(OPCODE_LOAD, (acc,)),
        Instruction(OPCODE_HALT, ()),
    ]
    iterative = execute(loop_prog, {i: ZERO, acc: ZERO}, step_limit=1000)
    assert rational.eq(closed, iterative)   # 两独立路径一致
    assert rational.eq(closed, make(15, 1))


def test_step_limit_catches_infinite_loop():
    """JMP-self 死循环·step_limit 是 A1 后唯一终止界（fail-loud 不挂）。"""
    with pytest.raises(StepLimitExceeded):
        execute([Instruction(OPCODE_JMP, (0,))], {}, step_limit=100)


def test_control_opcodes_not_binary():
    """A1 控制流 opcode 禁走 dispatch_binary（is_binary_opcode 不含·负不变量）。"""
    for op in (OPCODE_PUSH_IMM, OPCODE_JZ, OPCODE_JMP, OPCODE_HALT):
        assert not is_binary_opcode(op)


def test_calc_role_analytical_first():
    assert calc_role(OPCODE_ADD) == CALC_ROLE_ANALYTICAL
    assert calc_role(OPCODE_DIV) == CALC_ROLE_ANALYTICAL


def test_graph_compile_postorder():
    """COMPOSES 后序 emit：叶 LOAD 在前·算子在后（栈机消费）。"""
    # tree: root=ADD(a, MUL(b, c)) → LOAD a, LOAD b, LOAD c, MUL, ADD
    root = (1, 1)
    a = (1, 2); b = (1, 3); c = (1, 4); mul = (1, 5)
    children_of = {root: [a, mul], mul: [b, c]}
    operator_of = {root: OPCODE_ADD, mul: OPCODE_MUL}
    operand_of = {a: make_variable(1), b: make_variable(2), c: make_variable(3)}
    instrs = compile_graph(root, children_of, operator_of, operand_of)
    opcodes = [i.opcode for i in instrs]
    assert opcodes == [OPCODE_LOAD, OPCODE_LOAD, OPCODE_LOAD, OPCODE_MUL, OPCODE_ADD]
    assert instrs[0].args == (make_variable(1),)


def test_graph_compile_cycle_detected():
    """COMPOSES 环 → LoopClosureDefect（不无限递归）。"""
    n1, n2 = (1, 1), (1, 2)
    children_of = {n1: [n2], n2: [n1]}  # 环
    operator_of = {n1: OPCODE_ADD, n2: OPCODE_ADD}
    operand_of = {}
    with pytest.raises(LoopClosureDefect):
        compile_graph(n1, children_of, operator_of, operand_of)


def test_graph_compile_max_depth():
    """超 max_depth → LoopClosureDefect（不静默截断）。"""
    n1, n2, n3 = (1, 1), (1, 2), (1, 3)
    children_of = {n1: [n2], n2: [n3]}
    operator_of = {n1: OPCODE_ADD, n2: OPCODE_ADD}
    operand_of = {n3: make_variable(0)}
    with pytest.raises(LoopClosureDefect):
        compile_graph(n1, children_of, operator_of, operand_of, max_depth=1)


def test_vm_execute_end_to_end():
    """compile → execute：(1/2 + (1/3 * 1/4)) 纯整数精确。"""
    root = (1, 1)
    a = (1, 2); b = (1, 3); c = (1, 4); mul = (1, 5)
    children_of = {root: [a, mul], mul: [b, c]}
    operator_of = {root: OPCODE_ADD, mul: OPCODE_MUL}
    operand_of = {a: make_variable(1), b: make_variable(2), c: make_variable(3)}
    instrs = compile_graph(root, children_of, operator_of, operand_of)
    env = {make_variable(1): make(1, 2), make_variable(2): make(1, 3), make_variable(3): make(1, 4)}
    result = execute(instrs, env)
    # 1/2 + 1/12 = 7/12
    assert rational.eq(result, make(7, 12))


def test_vm_execute_store_load():
    """STORE 写回 env·LOAD 再读·验证 env 原地修改。"""
    instrs = [
        Instruction(OPCODE_LOAD, (make_variable(1),)),   # push a
        Instruction(OPCODE_LOAD, (make_variable(2),)),   # push b
        Instruction(OPCODE_ADD, ()),                      # push a+b
        Instruction(OPCODE_STORE, (make_variable(3),)),   # pop → c
        Instruction(OPCODE_LOAD, (make_variable(3),)),    # push c
    ]
    env = {make_variable(1): make(1, 4), make_variable(2): make(1, 4)}
    result = execute(instrs, env)
    assert rational.eq(result, make(1, 2))
    assert rational.eq(env[make_variable(3)], make(1, 2))


def test_vm_execute_step_limit():
    """超 step_limit → StepLimitExceeded（禁无限步）。"""
    instrs = [Instruction(OPCODE_NOP, ()) for _ in range(DEFAULT_STEP_LIMIT + 1)]
    with pytest.raises(StepLimitExceeded):
        execute(instrs, {}, step_limit=10)


# ============ VM A2 控制流 compile（图灵完备 lower） ============

def test_control_flow_tags_not_vm_opcodes():
    """A2: CTRL_* 是编译指令非 VM opcode（不入 _OPCODE_TABLE / 不经 dispatch_binary）。"""
    for tag in (CTRL_IF, CTRL_IFELSE, CTRL_WHILE):
        assert is_control_flow_tag(tag)
        assert not is_binary_opcode(tag)
        with pytest.raises(KeyError):          # 不在 VM opcode 桥（compiler-internal sentinel）
            symbol_to_opcode(tag)


def test_compile_if_else_selects_branch():
    """A2: IFELSE compile → execute·cond 选 THEN/ELSE 值（balanced·栈净 1）。"""
    root, cond, then, els = (1, 1), (1, 2), (1, 3), (1, 4)
    x, y, a, b = (1, 5), (1, 6), (1, 7), (1, 8)
    vx, vy, va, vb = (make_variable(20), make_variable(21),
                      make_variable(22), make_variable(23))
    children_of = {root: [cond, then, els], cond: [x, y]}   # EQ(x, y)
    operator_of = {root: CTRL_IFELSE, cond: OPCODE_EQ}
    operand_of = {x: vx, y: vy, then: va, els: vb}          # then/els 为叶 → LOAD
    instrs = compile_graph(root, children_of, operator_of, operand_of)
    base = {va: make(10, 1), vb: make(20, 1)}
    # cond true (x==y) → THEN 值 a=10
    env_t = {**base, vx: make(3, 1), vy: make(3, 1)}
    assert rational.eq(execute(instrs, env_t, step_limit=100), make(10, 1))
    # cond false (x!=y) → ELSE 值 b=20
    env_f = {**base, vx: make(3, 1), vy: make(4, 1)}
    assert rational.eq(execute(instrs, env_f, step_limit=100), make(20, 1))


def test_compile_if_store_side_effect():
    """A2: IF compile → execute·cond 真 THEN 回写 R·假跳末尾 R 不变（skip-target=n 合法）。"""
    root, cond, then, const42 = (1, 1), (1, 2), (1, 3), (1, 4)
    vc, vr = make_variable(30), make_variable(31)
    children_of = {root: [cond, then], then: [const42]}
    operator_of = {root: CTRL_IF}
    operand_of = {cond: vc}
    immediate_of = {const42: (42, 1)}
    store_target_of = {then: vr}                    # then = STORE R ← 42
    instrs = compile_graph(root, children_of, operator_of, operand_of,
                           immediate_of=immediate_of, store_target_of=store_target_of)
    # cond 真 → R=42
    env_t = {vc: ONE, vr: ZERO}
    execute(instrs, env_t, step_limit=100)
    assert rational.eq(env_t[vr], make(42, 1))
    # cond 假 → JZ skip-to-end（target=n）·R 不变
    env_f = {vc: ZERO, vr: ZERO}
    execute(instrs, env_f, step_limit=100)
    assert rational.eq(env_f[vr], ZERO)


def test_compile_if_lowering_structure():
    """A2: IF lower → [LOAD cond, JZ(target=then 后), LOAD then]·backpatch 正确。"""
    root, cond, then = (1, 1), (1, 2), (1, 3)
    vc, vt = make_variable(40), make_variable(41)
    children_of = {root: [cond, then]}              # cond/then 叶
    operator_of = {root: CTRL_IF}
    operand_of = {cond: vc, then: vt}
    instrs = compile_graph(root, children_of, operator_of, operand_of)
    assert [i.opcode for i in instrs] == [OPCODE_LOAD, OPCODE_JZ, OPCODE_LOAD]
    assert instrs[1].args == (3,)                   # then_skip = 末尾 = n（跳 THEN 落末）


def test_compile_while_loop_count_up():
    """A2: WHILE compile → execute·while i<6: i+=1 → i 终值 6（lower + 回边 + STORE + PUSH_IMM）。"""
    root, cond, stmt_i = (1, 1), (1, 2), (1, 6)
    i_leaf, six, add_i, one = (1, 4), (1, 5), (1, 7), (1, 8)
    vi = make_variable(50)
    children_of = {
        root: [cond, stmt_i],      # WHILE[COND, BODY]·BODY = 单语句 i:=i+1
        cond: [i_leaf, six],       # LT(i, 6)
        stmt_i: [add_i],           # STORE i ← (i+1)
        add_i: [i_leaf, one],      # ADD(i, 1)
    }
    operator_of = {root: CTRL_WHILE, cond: OPCODE_LT, add_i: OPCODE_ADD}
    operand_of = {i_leaf: vi}
    immediate_of = {six: (6, 1), one: (1, 1)}
    store_target_of = {stmt_i: vi}
    instrs = compile_graph(root, children_of, operator_of, operand_of,
                           immediate_of=immediate_of, store_target_of=store_target_of)
    env = {vi: make(1, 1)}
    execute(instrs, env, step_limit=1000)
    assert rational.eq(env[vi], make(6, 1))


def test_compile_while_loop_sum_turing_complete():
    """A2 端到端图灵完备：compile WHILE+SEQ(NOP)+STORE+PUSH_IMM → sum(1..5)=15。"""
    root, cond, body = (1, 1), (1, 2), (1, 3)
    i, acc, six, one = (1, 4), (1, 5), (1, 6), (1, 7)
    add_acc, add_i, s_acc, s_i = (1, 8), (1, 9), (1, 10), (1, 11)
    vi, vacc = make_variable(60), make_variable(61)
    children_of = {
        root: [cond, body],
        cond: [i, six],            # LT(i, 6)
        body: [s_acc, s_i],        # SEQ（NOP operator 顺序胶水）
        add_acc: [acc, i],         # acc + i
        add_i: [i, one],           # i + 1
        s_acc: [add_acc],          # STORE acc ← (acc+i)
        s_i: [add_i],              # STORE i ← (i+1)
    }
    operator_of = {root: CTRL_WHILE, cond: OPCODE_LT, body: OPCODE_NOP,
                   add_acc: OPCODE_ADD, add_i: OPCODE_ADD}
    operand_of = {i: vi, acc: vacc}
    immediate_of = {six: (6, 1), one: (1, 1)}
    store_target_of = {s_acc: vacc, s_i: vi}
    instrs = compile_graph(root, children_of, operator_of, operand_of,
                           immediate_of=immediate_of, store_target_of=store_target_of)
    env = {vi: make(1, 1), vacc: make(0, 1)}
    execute(instrs, env, step_limit=1000)
    assert rational.eq(env[vacc], make(15, 1))      # 1+2+3+4+5


def test_compile_while_backedge_not_graph_cycle():
    """A2: WHILE 回边在字节码非图·COMPOSES 仍 DAG·LoopClosureDefect 永不触发。"""
    root, cond, stmt_i = (1, 1), (1, 2), (1, 6)
    i_leaf, six, add_i, one = (1, 4), (1, 5), (1, 7), (1, 8)
    vi = make_variable(70)
    children_of = {root: [cond, stmt_i], cond: [i_leaf, six],
                   stmt_i: [add_i], add_i: [i_leaf, one]}
    operator_of = {root: CTRL_WHILE, cond: OPCODE_LT, add_i: OPCODE_ADD}
    operand_of = {i_leaf: vi}
    immediate_of = {six: (6, 1), one: (1, 1)}
    store_target_of = {stmt_i: vi}
    instrs = compile_graph(root, children_of, operator_of, operand_of,
                           immediate_of=immediate_of, store_target_of=store_target_of)
    assert any(x.opcode == OPCODE_JMP for x in instrs)   # 回边
    assert any(x.opcode == OPCODE_JZ for x in instrs)    # 出口


def test_control_flow_arity_validation():
    """A2: 控制流根子数错 → ValueError（fail-loud·IF 须 2 子）。"""
    root, only = (1, 1), (1, 2)
    children_of = {root: [only]}                    # IF 须 [COND, THEN]·得 1
    operator_of = {root: CTRL_IF}
    operand_of = {only: make_variable(80)}
    with pytest.raises(ValueError):
        compile_graph(root, children_of, operator_of, operand_of)


def test_compile_from_edges_order_index_slots():
    """A2: order_index_of 槽位序（控制流子位置语义·覆盖自然序）。"""
    root, a, b, c = (1, 1), (1, 5), (1, 2), (1, 9)
    va, vb, vc = make_variable(90), make_variable(91), make_variable(92)
    # 自然序 (1,2)<(1,5)<(1,9) → [b,a,c]·order_index a=0,b=1,c=2 → [a,b,c]
    edges = [(root, a), (root, b), (root, c)]
    operator_of = {root: OPCODE_NOP}                # NOP 组合 = 顺序胶水（emit 子 + NOP）
    operand_of = {a: va, b: vb, c: vc}
    instrs = compile_from_edges(root, edges, operator_of, operand_of,
                                order_index_of={a: 0, b: 1, c: 2})
    loads = [i.args[0] for i in instrs if i.opcode == OPCODE_LOAD]
    assert loads == [va, vb, vc]                    # 按 order_index 非 NodeRef 自然序


# ============ A2 拓扑分层 ============

def test_kahn_topo_dag_layers():
    """Kahn 分层：layer[node]=max(pred)+1·源=0·无环。"""
    # 1→2, 1→3, 2→4, 3→4
    edges = [((1, 1), (1, 2)), ((1, 1), (1, 3)),
             ((1, 2), (1, 4)), ((1, 3), (1, 4))]
    res = kahn_topo(edges)
    assert res.is_dag
    assert res.cycle_nodes == []
    assert res.layers[(1, 1)] == 0
    assert res.layers[(1, 2)] == 1
    assert res.layers[(1, 3)] == 1
    assert res.layers[(1, 4)] == 2
    # 拓扑序：1 在 2/3 前·2/3 在 4 前
    pos = {n: i for i, n in enumerate(res.order)}
    assert pos[(1, 1)] < pos[(1, 2)] < pos[(1, 4)]
    assert pos[(1, 1)] < pos[(1, 3)] < pos[(1, 4)]
    assert max_layer(res) == 2


def test_kahn_topo_cycle_detected():
    """环 → cycle_nodes 非空·不无限循环。"""
    edges = [((1, 1), (1, 2)), ((1, 2), (1, 1))]  # 环
    res = kahn_topo(edges)
    assert not res.is_dag
    assert set(res.cycle_nodes) == {(1, 1), (1, 2)}


def test_kahn_topo_deterministic():
    """同输入两跑同序（bit-identical·队列按节点自然序）。"""
    edges = [((1, 3), (1, 4)), ((1, 1), (1, 2)), ((1, 2), (1, 4))]
    r1 = kahn_topo(edges)
    r2 = kahn_topo(edges)
    assert r1.order == r2.order
    assert r1.layers == r2.layers


def test_kahn_predecessors_by_layer():
    edges = [((1, 1), (1, 3)), ((1, 2), (1, 3))]
    res = kahn_topo(edges)
    preds = predecessors_by_layer(edges, res)
    assert preds[(1, 3)] == [(1, 1), (1, 2)]  # 同层按自然序


# ============ A3 PersonalRank ============

def _small_matrix():
    """3 节点：0→1(w1) 0→2(w1) 1→2(w2)·α=85/100。"""
    nodes = [(1, 1), (1, 2), (1, 3)]
    edges = [((1, 1), (1, 2), 1), ((1, 1), (1, 3), 1), ((1, 2), (1, 3), 2)]
    alpha = make(DEFAULT_ALPHA_NUM, DEFAULT_ALPHA_DEN)
    return build_matrix(nodes, edges, alpha)


def test_build_matrix_row_normalized():
    m = _small_matrix()
    # row 0 (node 1,1): 出边到 (1,2) 1/2 + (1,3) 1/2 = 1
    r0 = dict(m.rows[0])
    assert rational.eq(r0[1], make(1, 2))
    assert rational.eq(r0[2], make(1, 2))
    # row 1 (node 1,2): 出边到 (1,3) 2/2 = 1
    assert rational.eq(dict(m.rows[1])[2], ONE)
    # row 2 (node 1,3): dangling → 空
    assert m.rows[2] == []


def test_pr_solve_exact_residual_zero():
    """B1 精确：(I−αA^T)x = (1−α)e 残差 = 0（线性系统零损失）。"""
    m = _small_matrix()
    e = {(1, 1): ONE}
    x = solve_exact(m, e)
    # 验残差 (I−αA^T)x − (1−α)e = 0
    alpha = m.alpha
    one_minus_alpha = sub_one(alpha)
    for i, node_i in enumerate(m.nodes):
        # (I−αA^T)x[i] = x[i] − α·Σ_{j→i} A_ji·x[j]
        s = ZERO
        for j in range(m.n):
            for ii, aji in m.rows.get(j, []):
                if ii == i:
                    s = rational.add(s, rational.mul(aji, x[m.nodes[j]]))
        lhs = rational.sub(x[node_i], rational.mul(alpha, s))
        rhs = rational.mul(one_minus_alpha, e.get(node_i, ZERO))
        assert rational.eq(lhs, rhs), f"残差非零 at {node_i}: {lhs} vs {rhs}"


def test_pr_solve_exact_known_values():
    """手算验证：前向 PR·seed=node0·x[0]=1−α·x[1]=α/2·(1−α)。"""
    m = _small_matrix()
    x = solve_exact(m, {(1, 1): ONE})
    alpha = m.alpha
    one_minus_alpha = sub_one(alpha)
    assert rational.eq(x[(1, 1)], one_minus_alpha)
    # x[1] = α·A_01·x[0] = α·(1/2)·(1−α)
    assert rational.eq(x[(1, 2)], rational.mul(rational.mul(alpha, make(1, 2)), one_minus_alpha))


def test_pr_linearity_zero_loss():
    """线性性：x = Σ x_s（per-seed 解之和 == 合并 e 解）·A 固定零损失。"""
    m = _small_matrix()
    seeds = [(1, 1), (1, 2)]
    combined = solve_exact(m, {(1, 1): ONE, (1, 2): ONE})
    summed = solve_exact_multi(m, seeds)
    for node in m.nodes:
        assert rational.eq(combined[node], summed[node]), f"线性性失败 at {node}"


def test_pr_b2_iterative_converges_to_exact():
    """B2 迭代收敛到 B1 精确解附近（迭代截断累积 ~几 ulp·诚实标注近似非 1 ulp）。"""
    from pure_integer_ai.crosscut.integer.constants import BASE, DEFAULT_K
    m = _small_matrix()
    e = {(1, 1): ONE}
    x_exact = solve_exact(m, e)
    x_iter = solve_iterative(m, e, k=DEFAULT_K, max_iter=300)
    # B2 是定点迭代近似：每步 mul 截断 <1 ulp·累积 ~几十 ulp。
    # 验 |exact − M/B^k| < 64/B^k（64 ulp 容差·诚实反映迭代近似非精确）。
    bk = BASE ** DEFAULT_K
    TOL_ULP = 64
    for node in m.nodes:
        fq = x_iter[node]
        ex = x_exact[node]
        # |ex.num/ex.den − fq.M/bk| < TOL_ULP/bk
        # ⟺ |ex.num·bk − fq.M·ex.den| < TOL_ULP·ex.den
        diff_num = abs(ex.num * bk - fq.M * ex.den)
        assert diff_num < TOL_ULP * ex.den, (
            f"{node}: B2 偏离 exact 超容差: diff={diff_num/ex.den} ulp > {TOL_ULP}"
        )


def test_pr_personal_rank_b1():
    m = _small_matrix()
    res = personal_rank(m, [(1, 1)], mode="B1")
    assert res.mode == "B1" and res.exact is True
    alpha = m.alpha
    assert rational.eq(res.values[(1, 1)], sub_one(alpha))


def test_pr_personal_rank_b1_fallback_b2_on_singular():
    """B1 奇异（零权图·全 dangling）→ D1 落盘自动回退 B2。"""
    # 全 dangling：无正权出边 → A=0 → (I−αA)=I 非奇异 actually...
    # 真奇异需构造：用孤立节点（A=0）→ (I-αA)=I 可逆·不奇异。
    # 构造奇异：n=0 或... 实际 (I-αA) 对 α<1 + A 随机恒非奇异。
    # 故 PRSingular 路径用 monkeypatch solve_exact_multi 抛 PRSingular 验回退。
    import pure_integer_ai.algorithm.a3_personal_rank as pr
    m = _small_matrix()
    orig = pr.solve_exact_multi

    def boom(matrix, seeds):
        raise PRSingular("test")
    pr.solve_exact_multi = boom
    try:
        res = personal_rank(m, [(1, 1)], mode="B1")
        assert res.mode == "B2" and res.exact is False
    finally:
        pr.solve_exact_multi = orig


def test_pr_b3_lu_defer():
    """B3 LU defer（NotImplementedError·§十五决策9 B3）。"""
    m = _small_matrix()
    from pure_integer_ai.algorithm.a3_personal_rank import solve_lu
    with pytest.raises(NotImplementedError):
        solve_lu(m, {(1, 1): ONE})


# ============ A4 对齐 ============

def test_lcs_basic():
    assert lcs([1, 2, 3], [2, 3, 4]) == [2, 3]
    assert lcs([1, 2, 3], [4, 5, 6]) == []
    assert lcs([], [1, 2]) == []
    assert lcs([1, 2, 1, 2], [2, 1]) == [2, 1]


def test_lcs_score():
    assert lcs_score([1, 2, 3], [2, 3, 4]) == 2
    assert lcs_score([1, 2, 3], [1, 2, 3]) == 3


def test_lcs_deterministic():
    a, b = [3, 1, 2, 1, 3], [1, 3, 2, 1]
    assert lcs(a, b) == lcs(a, b)


def test_pairwise_fold():
    seqs = [[1, 2, 3, 4], [2, 3, 5], [1, 3, 4]]
    consensus, score = pairwise_fold(seqs)
    # 最长 [1,2,3,4] 为种子·fold [2,3,5]→[2,3]·fold [1,3,4]→[3]
    assert consensus == [3]
    assert score >= 0


def test_alignment_matches():
    pairs = alignment_matches([1, 2, 3], [2, 3, 4])
    # 匹配 2@位置1/0·3@位置2/1
    assert (1, 0) in pairs and (2, 1) in pairs


# ============ closure ============

def test_closure_by_type_dispatch():
    """按 edge_type 分发：同 type 闭包·不跨 type 混闭。"""
    edges = [
        ((1, 1), (1, 2), EDGE_PRECEDES, None),
        ((1, 2), (1, 3), EDGE_PRECEDES, None),
        # CAUSES 链
        ((1, 1), (1, 2), EDGE_CAUSES, None),
        ((1, 2), (1, 4), EDGE_CAUSES, None),
    ]
    cl = transitive_closure(edges, types={EDGE_PRECEDES, EDGE_CAUSES})
    # PRECEDES 派生：(1,1)→(1,3)
    assert ((1, 1), (1, 3), EDGE_PRECEDES) in cl
    # CAUSES 派生：(1,1)→(1,4)
    assert ((1, 1), (1, 4), EDGE_CAUSES) in cl
    # 不跨 type：(1,1)→(1,4) 不在 PRECEDES 闭包
    assert ((1, 1), (1, 4), EDGE_PRECEDES) not in cl


def test_closure_purity_filter_refers_to():
    """纯净性：REFERS_TO 仅 PURE_ALIAS 进纯同指闭包·喻称排除（闭包纯净性）。"""
    edges = [
        # 苹果 ↔ apple（PURE_ALIAS）进闭包
        ((1, 1), (1, 2), EDGE_REFERS_TO, {"subtype": SUBTYPE_PURE_ALIAS}),
        ((1, 2), (1, 3), EDGE_REFERS_TO, {"subtype": SUBTYPE_PURE_ALIAS}),
        # 诗仙 → 李白（METAPHOR 喻称）不进纯同指闭包
        ((1, 3), (1, 4), EDGE_REFERS_TO, {"subtype": SUBTYPE_METAPHOR}),
    ]
    cl = transitive_closure(
        edges, types={EDGE_REFERS_TO},
        purity_filter=lambda m: m.get("subtype") == SUBTYPE_PURE_ALIAS,
    )
    # 派生：(1,1)→(1,3)（经 PURE_ALIAS 链）
    assert ((1, 1), (1, 3), EDGE_REFERS_TO) in cl
    # 喻称断链：(1,3)→(1,4) 被 purity_filter 排除·(1,1)→(1,4) 不在闭包
    assert ((1, 1), (1, 4), EDGE_REFERS_TO) not in cl
    assert ((1, 3), (1, 4), EDGE_REFERS_TO) not in cl


def test_closure_derived_not_stored_no_direct():
    """CLOSURE 派生不存储：默认返回只含派生（间接）边·不含直接边。"""
    edges = [
        ((1, 1), (1, 2), EDGE_PRECEDES, None),
        ((1, 2), (1, 3), EDGE_PRECEDES, None),
    ]
    cl = transitive_closure(edges, types={EDGE_PRECEDES})
    # 直接边 (1,1)→(1,2) (1,2)→(1,3) 不在派生闭包
    assert ((1, 1), (1, 2), EDGE_PRECEDES) not in cl
    assert ((1, 2), (1, 3), EDGE_PRECEDES) not in cl
    # 派生 (1,1)→(1,3) 在
    assert ((1, 1), (1, 3), EDGE_PRECEDES) in cl


def test_closure_reachable_query():
    edges = [
        ((1, 1), (1, 2), EDGE_PRECEDES, None),
        ((1, 2), (1, 3), EDGE_PRECEDES, None),
        ((1, 3), (1, 4), EDGE_PRECEDES, None),
    ]
    cl = transitive_closure(edges, types={EDGE_PRECEDES}, include_direct=True)
    reach = reachable(cl, (1, 1), EDGE_PRECEDES)
    assert reach == {(1, 2), (1, 3), (1, 4)}


# ============ helpers ============

def sub_one(r: Rational) -> Rational:
    return rational.sub(ONE, r)
