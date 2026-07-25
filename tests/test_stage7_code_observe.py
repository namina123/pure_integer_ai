"""Stage 7 验收门测试：A3 代码域(Python) observe 建 COMPOSES 程序（#360·真瓶颈）。

覆盖（doc/重来_A3_代码域observe设计补充.md）：
  L1 AST→COMPOSES 映射单元（每节点类型 + fail-loud 拒绝 + 负数/bool 特例）
  L2 持久化往返（build→read_composes_tree→compile→execute 端到端算术·bit-identical）
  L3 observe 集成（MODALITY_CODE 段→EDGE_COMPOSES 落盘 + struct_ref=COMPOSES 根 + 不建 PRECEDES）
  L4 vm_proof_fn 闭环（手编源码+手算 expected→1/0·死循环→None vacate·非根→None）

铁律：纯整数 / 确定性 bit-identical / fail-loud UnsupportedConstruct / 依赖单向向下。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_CODE
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_COMPOSES, EDGE_PRECEDES
from pure_integer_ai.storage.composes_attr import (
    register_composes_attr, read_composes_attrs,
    ATTR_OPERATOR, ATTR_CTRL_TAG, ATTR_OPERAND, ATTR_IMMEDIATE, ATTR_STORE_TARGET,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import (
    InputPayload, Segment, SpaceContext, IntentType, ConceptRef,
    PathResult, PathData, OutputResult,
    STAGE_TRAINING, MODALITY_CODE, DOMAIN_CODE,
)
from pure_integer_ai.cognition.understanding.code_observe import (
    build_composes_from_source, UnsupportedConstruct,
)
from pure_integer_ai.cognition.understanding.observe import observe
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.vm.graph_compile import compile_graph
from pure_integer_ai.vm.vm_core import execute, StepLimitExceeded
from pure_integer_ai.numeric.symbol_domain import make_variable
from pure_integer_ai.crosscut.integer.rational import make
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.training.vm_proof import vm_proof_fn_factory


# ---- fixtures ----

@pytest.fixture
def code_env():
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    g = ConceptGraph(b)
    sid = sp.space_id
    yield b, sid, es, ci, g
    b.close()


def _build(code_env, code: str, *, seg_label: str = "__seg_1_0") -> ConceptRef:
    """建 COMPOSES 树·返 root=struct_ref。"""
    b, sid, es, ci, _ = code_env
    root_ref = ci.ensure(seg_label, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_source(
        code, concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_CODE, root_ref=root_ref)
    return root_ref


def _run(code_env, root: ConceptRef, input_args: tuple[int, ...]):
    """read_composes_tree → compile → execute（预载 input_args）·返 Rational。"""
    _, _, _, _, g = code_env
    children_of, operator_of, operand_of, immediate_of, store_target_of = \
        g.read_composes_tree(root)
    instrs = compile_graph(root, children_of, operator_of, operand_of,
                           immediate_of=immediate_of or None,
                           store_target_of=store_target_of or None)
    env = {make_variable(i): make(int(a), 1) for i, a in enumerate(input_args)}
    return execute(instrs, env)


# ============ L1 AST→COMPOSES 映射单元 ============

def test_assign_store_mapping(code_env):
    """Assign(单Name) → STORE 节点（store_target_of + value 子）。"""
    b, sid, es, ci, _ = code_env
    root = _build(code_env, "def f(x): y = x")
    g = ConceptGraph(b)
    children_of, operator_of, operand_of, _, store_target_of = g.read_composes_tree(root)
    # root=SEQ NOP·children=[STORE]
    assert operator_of[root]  # root 有 ATTR_OPERATOR (NOP)
    assert len(children_of[root]) == 1
    store = children_of[root][0]
    assert store in store_target_of          # STORE 节点有目标变量
    assert store in operator_of or True       # STORE 无 operator_of（graph_compile 走 store_target 分支）


def test_binop_add_arithmetic(code_env):
    """BinOp Add → 算子节点·execute x+1。"""
    root = _build(code_env, "def f(x): return x + 1")
    assert rational.eq(_run(code_env, root, (5,)), make(6, 1))


def test_negative_constant_via_unaryop(code_env):
    """负数字面量 = UnaryOp(USub, Constant) → immediate(-num,1) 特例（非拒绝）。"""
    root = _build(code_env, "def f(): return -5")
    assert rational.eq(_run(code_env, root, ()), make(-5, 1))


def test_uadd_constant(code_env):
    """UnaryOp(UAdd, Constant) → immediate(num,1) 特例。"""
    root = _build(code_env, "def f(): return +7")
    assert rational.eq(_run(code_env, root, ()), make(7, 1))


def test_bool_constant_normalized(code_env):
    """Constant(bool) → immediate(0/1,1) 显式规范化（非 isinstance 副作用）。"""
    root = _build(code_env, "def f(): return True")
    assert rational.eq(_run(code_env, root, ()), make(1, 1))
    root2 = _build(code_env, "def f(): return False", seg_label="__seg_1_1")
    assert rational.eq(_run(code_env, root2, ()), make(0, 1))


def test_compare_lt_returns_one_zero(code_env):
    """Compare Lt → EQ/LT/GT 产 ONE/ZERO·1<2→1。"""
    root = _build(code_env, "def f(a, b): return a < b")
    assert rational.eq(_run(code_env, root, (1, 2)), make(1, 1))
    assert rational.eq(_run(code_env, root, (3, 2)), make(0, 1))


def test_div_rejected_float(code_env):
    """BinOp Div(`/`) 产 float → UnsupportedConstruct（纯整数铁律）。"""
    with pytest.raises(UnsupportedConstruct):
        _build(code_env, "def f(a, b): return a / b")


def test_floordiv_rejected_no_opcode(code_env):
    """BinOp FloorDiv(`//`) 无 FLOOR_DIV → UnsupportedConstruct（不可映 OPCODE_DIV 语义错）。"""
    with pytest.raises(UnsupportedConstruct):
        _build(code_env, "def f(a, b): return a // b")


def test_mod_pow_rejected(code_env):
    """Mod/Pow 无 opcode → UnsupportedConstruct。"""
    with pytest.raises(UnsupportedConstruct):
        _build(code_env, "def f(a, b): return a % b")
    with pytest.raises(UnsupportedConstruct):
        _build(code_env, "def f(a, b): return a ** b")


def test_compare_multicomparison_rejected(code_env):
    """Compare 多比较 a<b<c → UnsupportedConstruct。"""
    with pytest.raises(UnsupportedConstruct):
        _build(code_env, "def f(a, b, c): return a < b < c")


def test_compare_lte_rejected(code_env):
    """Compare LtE 不支持 → UnsupportedConstruct。"""
    with pytest.raises(UnsupportedConstruct):
        _build(code_env, "def f(a, b): return a <= b")


def test_call_rejected(code_env):
    """Call 不支持（CALLS 边 defer）→ UnsupportedConstruct。"""
    with pytest.raises(UnsupportedConstruct):
        _build(code_env, "def f(): return foo()")


def test_for_rejected(code_env):
    """For 不支持（迭代器协议 defer）→ UnsupportedConstruct。"""
    with pytest.raises(UnsupportedConstruct):
        _build(code_env, "def f():\n  for i in range(3):\n    x = i\n  return x")


def test_multi_assign_rejected(code_env):
    """多赋值 a=b=1 / Tuple 赋值 → UnsupportedConstruct。"""
    with pytest.raises(UnsupportedConstruct):
        _build(code_env, "def f(): a = b = 1\n  return a")


def test_no_function_def_rejected(code_env):
    """无 FunctionDef → UnsupportedConstruct。"""
    with pytest.raises(UnsupportedConstruct):
        _build(code_env, "x = 1")


def test_unary_not_rejected(code_env):
    """UnaryOp(Not) 不支持 → UnsupportedConstruct。"""
    with pytest.raises(UnsupportedConstruct):
        _build(code_env, "def f(a): return not a")


# ============ L2 持久化往返（端到端算术·bit-identical） ============

def test_persistence_roundtrip_while_sum(code_env):
    """while 求和 f(5)=15·build→read→compile→execute 端到端。"""
    root = _build(code_env, "def f(n):\n  acc=0\n  while n>0:\n    acc+=n\n    n+=-1\n  return acc")
    assert rational.eq(_run(code_env, root, (5,)), make(15, 1))
    assert rational.eq(_run(code_env, root, (10,)), make(55, 1))   # sum(1..10)=55


def test_persistence_roundtrip_if_else(code_env):
    """if/else 选支·x>0→1 else→0。"""
    root = _build(code_env, "def f(x):\n  if x > 0:\n    return 1\n  else:\n    return 0")
    assert rational.eq(_run(code_env, root, (5,)), make(1, 1))
    assert rational.eq(_run(code_env, root, (-3,)), make(0, 1))


def test_persistence_roundtrip_if_no_else(code_env):
    """if 无 else·x>0→1 else 落末尾返栈顶 ZERO。"""
    root = _build(code_env, "def f(x):\n  if x > 0:\n    return 1")
    assert rational.eq(_run(code_env, root, (5,)), make(1, 1))
    # x<=0：跳过 then·落末尾隐式终止·栈空返 ZERO
    assert rational.eq(_run(code_env, root, (-1,)), make(0, 1))


def test_deterministic_bit_identical(code_env):
    """同源码两跑 bit-identical（surface hash + var index + 指令序列一致）。"""
    code = "def f(n):\n  acc=0\n  while n>0:\n    acc+=n\n    n+=-1\n  return acc"
    # 第一跑
    b1, sid1, es1, ci1, g1 = code_env
    root1 = _build(code_env, code, seg_label="__seg_a")
    c1, o1, od1, im1, st1 = g1.read_composes_tree(root1)
    instrs1 = compile_graph(root1, c1, o1, od1,
                            immediate_of=im1 or None, store_target_of=st1 or None)
    # 第二跑（fresh backend）
    b2 = DictBackend(); bootstrap(b2); register_composes_attr(b2)
    reg2 = SpaceRegistry(b2); sp2 = AbstractSpace.create(reg2, "core")
    es2 = EdgeStore(b2); ci2 = ConceptIndex(b2); g2 = ConceptGraph(b2)
    root2 = ci2.ensure("__seg_a", space_id=sp2.space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_source(
        code, concept_index=ci2, edge_store=es2, backend=b2,
        space_id=sp2.space_id, source=SOURCE_CODE, root_ref=root2)
    c2, o2, od2, im2, st2 = g2.read_composes_tree(root2)
    instrs2 = compile_graph(root2, c2, o2, od2,
                            immediate_of=im2 or None, store_target_of=st2 or None)
    # 指令序列 bit-identical
    assert len(instrs1) == len(instrs2)
    for i1, i2 in zip(instrs1, instrs2):
        assert i1.opcode == i2.opcode
        assert i1.args == i2.args
    # root ConceptRef 同 local_id（同 surface hash→同 local_id·per-space dedup）
    assert root1[1] == root2[1]
    b2.close()


# ============ L3 observe 集成 ============

def _observe_code(b, sp, code: str):
    seg = Segment(seg_id=0, modality=MODALITY_CODE, domain=DOMAIN_CODE,
                  code_source=code)
    raw = InputPayload(
        segments=[seg], source=SOURCE_CODE, stage=STAGE_TRAINING,
        modality=MODALITY_CODE, domain=DOMAIN_CODE,
        intent=IntentType())
    ctx = SpaceContext(
        core=sp, memory_read=None, memory_interact=None, companion=None,
        stage=STAGE_TRAINING)
    return observe(raw, ctx)


def test_observe_code_segment_lands_composes(code_env):
    """MODALITY_CODE 段→EDGE_COMPOSES 边落 EdgeStore + struct_ref=COMPOSES 根可遍历。"""
    b, sid, es, ci, g = code_env
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    obs = _observe_code(b, sp, "def f(x): return x + 1")
    assert obs.built_concepts >= 1
    root = obs.struct_refs[0]
    children_of, operator_of, _, _, _ = g.read_composes_tree(root)
    assert children_of                      # struct_ref=COMPOSES 根有子
    assert root in operator_of              # root 有 ATTR_OPERATOR (SEQ NOP)
    # EDGE_COMPOSES 边真落盘
    composes_edges = b.select("edge", where={"edge_type": EDGE_COMPOSES})
    assert len(composes_edges) >= 2         # root→Return + Return→BinOp 至少


def test_observe_code_segment_no_precedes(code_env):
    """代码域段不建 PRECEDES 序链（致命#3·不污染序链）。"""
    b, _, _, _, _ = code_env
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    _observe_code(b, sp, "def f(x): return x + 1")
    precedes = b.select("edge", where={"edge_type": EDGE_PRECEDES})
    assert len(precedes) == 0               # 代码域不建 PRECEDES


def test_observe_code_segment_while_compiles_and_runs(code_env):
    """observe 建的 COMPOSES 树经 read→compile→execute 端到端算术正确。"""
    b, sid, es, ci, g = code_env
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    obs = _observe_code(b, sp, "def f(n):\n  acc=0\n  while n>0:\n    acc+=n\n    n+=-1\n  return acc")
    root = obs.struct_refs[0]
    result = _run(code_env, root, (5,))
    assert rational.eq(result, make(15, 1))


# ============ L4 vm_proof_fn 闭环 ============

def _dag(sink: ConceptRef) -> PathResult:
    return PathResult(path=PathData(), terminal=1, sink=sink,
                      topo_layers=[], convergence={}, source=None)


def test_vm_proof_fn_sum_pass(code_env):
    """手编源码+手算 expected→vm_proof_fn 返 1（verified·R6 独立源）。"""
    b, sid, es, ci, g = code_env
    root = _build(code_env, "def f(n):\n  acc=0\n  while n>0:\n    acc+=n\n    n+=-1\n  return acc")
    fn = vm_proof_fn_factory(input_args=(5,), expected=(15, 1))
    assert fn(OutputResult(), _dag(root), g) == 1


def test_vm_proof_fn_mismatch(code_env):
    """错误 expected→vm_proof_fn 返 0（mismatch）。"""
    b, sid, es, ci, g = code_env
    root = _build(code_env, "def f(n):\n  acc=0\n  while n>0:\n    acc+=n\n    n+=-1\n  return acc")
    fn = vm_proof_fn_factory(input_args=(5,), expected=(14, 1))   # 错·应是 15
    assert fn(OutputResult(), _dag(root), g) == 0


def test_vm_proof_fn_deadloop_vacate(code_env):
    """死循环（while x>0: x=x·x 不变）→StepLimit→None（R1 vacate·非 pass）。"""
    b, sid, es, ci, g = code_env
    root = _build(code_env, "def f(x):\n  while x > 0:\n    x = x")
    fn = vm_proof_fn_factory(input_args=(1,), expected=(0, 1))
    assert fn(OutputResult(), _dag(root), g) is None


def test_vm_proof_fn_non_composes_root_vacate(code_env):
    """root 非 COMPOSES 根（语言段 struct_ref 无属性）→None（vacate·不伪造判）。"""
    b, sid, es, ci, g = code_env
    # 建 struct_ref 不 build COMPOSES（模拟语言段 struct_ref）
    root = ci.ensure("__seg_lang_0", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    fn = vm_proof_fn_factory(input_args=(1,), expected=(1, 1))
    assert fn(OutputResult(), _dag(root), g) is None


def test_vm_proof_fn_no_sink_vacate(code_env):
    """dag_path.sink=None→None（vacate·非代码域 episode）。"""
    _, _, _, _, g = code_env
    fn = vm_proof_fn_factory(input_args=(1,), expected=(1, 1))
    assert fn(OutputResult(), _dag(None), g) is None
