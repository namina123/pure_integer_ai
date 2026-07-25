"""Stage 9 验收门测试：结构发现最小闭环（§八序列1·doc/重来_结构发现设计补充.md）。

结构发现 = 系统生成核心 + 多模态根基（§〇/§五）。本段 = 最小闭环（序列1）：多样本 COMPOSES 程序 →
等长结构对齐 → 抽共性骨架（固定位 + PARAM 槽）→ 落 struct_ref+COMPOSES（ATTR_ORIGIN=discovered）→
复用既有 inline+β+vm_proof / coverage_overlap 消费。

覆盖（反 theater 四断言散布各测）：
  SD1 抽骨架单元（两样本→骨架 arity>0·ATTR_ORIGIN 落盘·固定位/相异位分类·异构/越界→None）
  SD2 骨架自验（直接 vm_proof：PARAM 槽绑 input_args → 执行 → 值·骨架是合法程序）
  SD3 inline 消费（register+Call+β → 复现样本 + 泛化新值·下游消费非摆设）
  SD4 coverage_overlap 识别消费（shape_signature 新样本同形 → coverage=1000·认出结构）
  SD5 ATTR_ORIGIN 不传播（inline 嫁接是消费非重生·_STRUCTURAL_KINDS 不含 ATTR_ORIGIN）
  SD6 fail-loud 边界（异构结构/异 opcode/异 body/internal sid 数异/<2样本 → None·loop1 scope 诚实·operand 序列2 已支持·ctrl/store S3 现产骨架）
  SD7 确定性（两独立 backend 同结构/arity/执行 bit-identical）

铁律：纯整数 / 确定性 bit-identical / fail-loud（None 非静默错骨架）/ 依赖单向向下 /
  反 theater（进容器+真抽骨架+vm_proof真验+下游消费·四断言）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_MATH
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT, NodeStore
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
from pure_integer_ai.storage.composes_attr import (
    register_composes_attr, read_composes_attrs,
    ATTR_ORIGIN, ORIGIN_DISCOVERED, ATTR_OPERATOR_DEF,
)
from pure_integer_ai.storage.op_confidence import (
    register_op_confidence, record_op_outcome, read_op_confidence, OP_CONFIDENCE_TABLE,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import ConceptRef, MODALITY_ARITH, DOMAIN_MATH, LANG_NONE
from pure_integer_ai.cognition.understanding.arith_observe import (
    build_composes_from_arith, register_arith_operator,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.process.structure_discover import (
    discover_skeleton, shape_signature, SkeletonResult,
    auto_discover_operators, recognize_operators, Recognition,
    load_discovered_operators, probe_arity,
    route_samples_for_discovery, _collect_slot_lcas, _collect_cue_sig,
    _normalize_abstract_sig,
)
from pure_integer_ai.cognition.process.a4_align import coverage_overlap, MAX_QUALITY
from pure_integer_ai.vm.graph_compile import compile_graph
from pure_integer_ai.vm.vm_core import execute
from pure_integer_ai.numeric.symbol_domain import make_variable
from pure_integer_ai.crosscut.integer.rational import make
from pure_integer_ai.crosscut.integer import rational


# ---- fixtures（镜像 test_stage9_arith_observe·同 arith_env 范式） ----

@pytest.fixture
def disc_env():
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


def _build(disc_env, src: str, *, seg_label: str) -> ConceptRef:
    """建 COMPOSES 树·返 root=struct_ref（样本程序）。"""
    b, sid, es, ci, _ = disc_env
    root_ref = ci.ensure(seg_label, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith(
        src, concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_MATH, root_ref=root_ref)
    return root_ref


def _run(disc_env, root: ConceptRef, input_args: tuple[int, ...]):
    """read_composes_tree → compile → execute（预载 input_args）·返 Rational。"""
    _, _, _, _, g = disc_env
    children_of, operator_of, operand_of, immediate_of, store_target_of = \
        g.read_composes_tree(root)
    instrs = compile_graph(root, children_of, operator_of, operand_of,
                           immediate_of=immediate_of or None,
                           store_target_of=store_target_of or None)
    env = {make_variable(i): make(int(a), 1) for i, a in enumerate(input_args)}
    return execute(instrs, env)


# ============ SD1 抽骨架单元（两样本 → 骨架·反 theater：进容器+真抽骨架） ============

def test_discover_two_samples_extracts_skeleton_arity2(disc_env):
    """两样本 ADD(IMM3,IMM5) + ADD(IMM4,IMM7) → 骨架 ADD(PARAM0,PARAM1)·arity=2。

    反 theater 断言①进容器：ATTR_ORIGIN=discovered 落盘 + read_composes_tree 非空。
    反 theater 断言②真抽骨架：arity=2（两相异立即数位真参数化·非空壳复制）。
    """
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda: 3 + 5", seg_label="__seg_a")
    s2 = _build(disc_env, "lambda: 4 + 7", seg_label="__seg_b")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="add_two")
    assert res is not None
    assert res.arity == 2, "两相异立即数位 → arity=2（真参数化）"
    # ①进容器：ATTR_ORIGIN=discovered 落在骨架 root
    attrs = read_composes_attrs(b, res.skeleton_ref)
    assert attrs.get(ATTR_ORIGIN) == (ORIGIN_DISCOVERED, 0), "骨架 root 须标 ATTR_ORIGIN=discovered"
    # ①进容器：骨架有 COMPOSES 子树（NOP-root + ADD 体 + 2 PARAM 叶）
    children_of, operator_of, _, _, _ = g.read_composes_tree(res.skeleton_ref)
    assert children_of, "骨架须有 COMPOSES 子树（非空壳）"
    assert res.skeleton_ref in operator_of, "骨架 root 须是算子节点（NOP struct_ref）"


def test_discover_fixed_position_kept_identical_immediates(disc_env):
    """相同立即数位→固定位（FIXED·非 PARAM）：lambda:3+5 + lambda:3+7 → 骨架 arity=1（位0=3 固定·位1=PARAM）。

    位0 两样本同值 3→FIXED IMM(3)·位1 异值(5,7)→PARAM_0。arity=1。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 3 + 5", seg_label="__seg_a")
    s2 = _build(disc_env, "lambda: 3 + 7", seg_label="__seg_b")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="add_c")
    assert res is not None and res.arity == 1, "位0 固定·仅位1 参数化 → arity=1"


def test_discover_nested_structure_alignment(disc_env):
    """嵌套结构对齐：(3+5)*2 vs (4+7)*3 → 骨架 MUL(ADD(PARAM0,PARAM1),PARAM2)·arity=3。

    证对齐处理任意深度算子树（非扁平）·固定位/相异位按结构位置分类。
    执行断言（防 PARAM 序 bug）：mul3(3,5,2) == (3+5)*2 == 16·mul3(4,7,3) == (4+7)*3 == 33。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: (3 + 5) * 2", seg_label="__seg_a")
    s2 = _build(disc_env, "lambda: (4 + 7) * 3", seg_label="__seg_b")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="muladd")
    assert res is not None and res.arity == 3, "三相异立即数位 → arity=3"
    # 直接 vm_proof：PARAM 阅读序 (3,5,2) → MUL(ADD(3,5),2) = 16（BFS 层序 bug 会得 MUL(ADD(5,2),3)=21）
    assert rational.eq(_run(disc_env, res.skeleton_ref, (3, 5, 2)), make(16, 1)), (
        "PARAM 须按 DFS 阅读序赋 sid·(3,5,2) → (3+5)*2=16（BFS 层序会错成 21）")
    assert rational.eq(_run(disc_env, res.skeleton_ref, (4, 7, 3)), make(33, 1))


def test_discover_noncommutative_nested_param_reading_order(disc_env):
    """**回归测试**（对抗正确性审计揪的致命 bug）：非交换 + 非均匀深度 → PARAM 须按 DFS 阅读序。

    bug：PARAM sid 按 BFS 层序分配 ≠ inline arg_subst 按 AST 位置序·非交换算子(SUB)必触发静默错值。
    lambda:100-10-1 = SUB(SUB(100,10),1)·BFS 叶序=[1,100,10]（深2优先）·阅读序=[100,10,1]。
    BFS bug 骨架=SUB(SUB(mv1,mv2),mv0)·inline sub3(100,10,1) → (10-1)-100 = -91（期 89）。
    DFS 修后骨架=SUB(SUB(mv0,mv1),mv2)·sub3(100,10,1) → (100-10)-1 = 89 ✓。
    两路独立断言：直接 vm_proof + inline Call β·都得 89（非交换·序错必露馅）。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 100 - 10 - 1", seg_label="__seg_a")
    s2 = _build(disc_env, "lambda: 200 - 20 - 2", seg_label="__seg_b")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="sub3")
    assert res is not None and res.arity == 3
    # ① 直接 vm_proof：阅读序 (100,10,1) → (100-10)-1 = 89（BFS bug 得 (10-1)-100=-91）
    assert rational.eq(_run(disc_env, res.skeleton_ref, (100, 10, 1)), make(89, 1)), (
        "PARAM 须 DFS 阅读序·(100,10,1)→(100-10)-1=89（BFS 层序 bug 得 -91）")
    # ② inline 消费：sub3(100,10,1) == 直接 (100-10)-1 == 89（β-归约 arg_subst AST 位置序）
    register_arith_operator(b, ci, "sub3", res.skeleton_ref, arity=res.arity)
    root_ref = _build(disc_env, "lambda: sub3(100, 10, 1)", seg_label="__seg_use")
    root_direct = _build(disc_env, "lambda: 100 - 10 - 1", seg_label="__seg_direct")
    assert _run(disc_env, root_ref, ()) == _run(disc_env, root_direct, ()) == make(89, 1), (
        "inline β 须按阅读序填槽·sub3(100,10,1)=(100-10)-1=89（非交换算子验序真契约）")
    # 泛化：sub3(50,5,2) = (50-5)-2 = 43
    root_gen = _build(disc_env, "lambda: sub3(50, 5, 2)", seg_label="__seg_gen")
    assert rational.eq(_run(disc_env, root_gen, ()), make(43, 1))


# ============ SD2 骨架自验（直接 vm_proof·骨架是合法参数化程序） ============

def test_skeleton_direct_vm_proof_binds_params(disc_env):
    """骨架 PARAM 槽=make_variable(0..arity-1)·直接 vm_proof：input_args 绑参 → 执行 → 值。

    反 theater 断言③vm_proof真验：发现的骨架是合法可执行程序（ADD(PARAM0,PARAM1)·
    input_args=(10,20)→30·=(3,5)→8 复现样本0）。PARAM sid 与 vm_proof 绑参约定一致（零摩擦）。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 3 + 5", seg_label="__seg_a")
    s2 = _build(disc_env, "lambda: 4 + 7", seg_label="__seg_b")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="add_two")
    assert res is not None
    # 绑参执行：skel(10,20)=30·skel(3,5)=8（复现样本0）·skel(4,7)=11（复现样本1）
    assert rational.eq(_run(disc_env, res.skeleton_ref, (10, 20)), make(30, 1))
    assert rational.eq(_run(disc_env, res.skeleton_ref, (3, 5)), make(8, 1)), "骨架泛化含样本0"
    assert rational.eq(_run(disc_env, res.skeleton_ref, (4, 7)), make(11, 1)), "骨架泛化含样本1"


# ============ SD3 inline 消费（register+Call+β → 复现+泛化·下游消费非摆设） ============

def test_skeleton_inline_consumer_reproduces_and_generalizes(disc_env):
    """inline 消费：register 骨架为算子 → Call 引用 → β-归约 → vm_proof（复现样本 + 泛化新值）。

    反 theater 断言④下游消费：发现的骨架被 inline 进新程序（非仅自验·非摆设）。
    add_two=发现骨架(arity2)·lambda:add_two(3,5)→inline→ADD(IMM3,IMM5)→8 复现样本0·
    lambda:add_two(10,20)→30 泛化新值·对比直接段同果。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 3 + 5", seg_label="__seg_a")
    s2 = _build(disc_env, "lambda: 4 + 7", seg_label="__seg_b")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="add_two")
    assert res is not None
    register_arith_operator(b, ci, "add_two", res.skeleton_ref, arity=res.arity)
    # inline 复现样本0：add_two(3,5) == 直接 3+5 == 8
    root_ref = _build(disc_env, "lambda: add_two(3, 5)", seg_label="__seg_use_a")
    root_direct = _build(disc_env, "lambda: 3 + 5", seg_label="__seg_direct_a")
    assert _run(disc_env, root_ref, ()) == _run(disc_env, root_direct, ()) == make(8, 1)
    # inline 泛化新值：add_two(10,20) == 直接 10+20 == 30
    root_gen = _build(disc_env, "lambda: add_two(10, 20)", seg_label="__seg_use_gen")
    assert rational.eq(_run(disc_env, root_gen, ()), make(30, 1))


def test_skeleton_inline_vm_proof_closed_loop(disc_env):
    """vm_proof 闭环：inline 引用发现骨架段 → vm_proof_fn 验 → reward=1·mismatch→0（反 theater）。"""
    from pure_integer_ai.training.vm_proof import vm_proof_fn_factory
    from pure_integer_ai.cognition.shared.types import PathResult, PathData, OutputResult, TERMINAL_REACHED_SINK
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda: 3 + 5", seg_label="__seg_a")
    s2 = _build(disc_env, "lambda: 4 + 7", seg_label="__seg_b")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="add_two")
    register_arith_operator(b, ci, "add_two", res.skeleton_ref, arity=res.arity)
    root_ref = _build(disc_env, "lambda: add_two(6, 8)", seg_label="__seg_use")
    dag = PathResult(path=PathData(), terminal=TERMINAL_REACHED_SINK, sink=root_ref)
    # pass：add_two(6,8)=14
    assert vm_proof_fn_factory(input_args=(), expected=(14, 1))(OutputResult(), dag, g) == 1
    # 反 theater：mismatch expected=15→0（证 reward 来自真 VM 执行·非 stub）
    assert vm_proof_fn_factory(input_args=(), expected=(15, 1))(OutputResult(), dag, g) == 0


# ============ SD4 coverage_overlap 识别消费（shape_signature → 认出结构） ============

def test_skeleton_coverage_overlap_recognizes_new_sample(disc_env):
    """coverage_overlap 识别：新样本同算子形状 → shape_signature 等价 → coverage=1000（认出结构）。

    第二个下游消费（§8.7）：发现的骨架可被 coverage_overlap 命中识别新样本（"认出语言/结构"）。
    新样本 lambda:6+9 与骨架 ADD(_,_) 同形状（NOP,ADD,叶,叶）→ coverage=MAX_QUALITY。
    异形状样本 lambda:6*9 → coverage<1000（MUL≠ADD·不认作同结构）。
    """
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda: 3 + 5", seg_label="__seg_a")
    s2 = _build(disc_env, "lambda: 4 + 7", seg_label="__seg_b")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="add_two")
    skel_sig = shape_signature(g, res.skeleton_ref)
    # 新样本同形状（ADD）→ 识别
    new_add = _build(disc_env, "lambda: 6 + 9", seg_label="__seg_new_add")
    new_sig = shape_signature(g, new_add)
    assert new_sig == skel_sig, "同形状（NOP,ADD,叶,叶）签名须等价"
    assert coverage_overlap(new_sig, [skel_sig]) == MAX_QUALITY, "同形 → coverage=1000 认出"
    # 异形状样本（MUL）→ 不识别
    new_mul = _build(disc_env, "lambda: 6 * 9", seg_label="__seg_new_mul")
    mul_sig = shape_signature(g, new_mul)
    assert mul_sig != skel_sig, "异形状（MUL）签名须不同"
    assert coverage_overlap(mul_sig, [skel_sig]) < MAX_QUALITY, "异形 → coverage<1000 不认作同结构"


# ============ SD5 ATTR_ORIGIN 不传播（inline 是消费非重生） ============

def test_attr_origin_not_propagated_on_inline(disc_env):
    """inline 嫁接的节点不带 ATTR_ORIGIN（_STRUCTURAL_KINDS 不含它→消费非重生）。

    骨架 root 标 discovered·但 inline 嫁接到引用段的 fresh 节点是 USE·不应被标 discovered
    （否则消费点伪造成"新发现"·反 theater 失败）。_deep_copy_subtree 只复制 _STRUCTURAL_KINDS。
    """
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda: 3 + 5", seg_label="__seg_a")
    s2 = _build(disc_env, "lambda: 4 + 7", seg_label="__seg_b")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="add_two")
    register_arith_operator(b, ci, "add_two", res.skeleton_ref, arity=res.arity)
    root_ref = _build(disc_env, "lambda: add_two(3, 5)", seg_label="__seg_use")
    # 遍历引用段 COMPOSES 子树·断言无节点带 ATTR_ORIGIN（inline 嫁接是消费非重生）
    children_of, _, _, _, _ = g.read_composes_tree(root_ref)
    visited: set[ConceptRef] = set()
    queue = [root_ref]
    origin_count = 0
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        if ATTR_ORIGIN in read_composes_attrs(b, node):
            origin_count += 1
        queue.extend(children_of.get(node, []))
    assert origin_count == 0, "inline 嫁接节点不应带 ATTR_ORIGIN（消费非重生）"


# ============ SD6 fail-loud 边界（异构/越界/<2样本 → None·loop1 scope 诚实） ============

def test_discover_heterogeneous_opcode_returns_none(disc_env):
    """异构结构（root 体 opcode 不同：ADD vs MUL）→ None（无共性骨架·变长锚对齐=序列2 defer）。"""
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 3 + 5", seg_label="__seg_add")
    s2 = _build(disc_env, "lambda: 3 * 5", seg_label="__seg_mul")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="x")
    assert res is None, "异构 opcode（ADD vs MUL）→ 无共性骨架"


def test_discover_heterogeneous_shape_returns_none(disc_env):
    """异构形状（子数不同：ADD(IMM,IMM) vs ADD(ADD(IMM,IMM),IMM)）→ None。"""
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 3 + 5", seg_label="__seg_flat")
    s2 = _build(disc_env, "lambda: 1 + 2 + 3", seg_label="__seg_nested")   # ADD(ADD(1,2),3)
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="x")
    assert res is None, "异构形状（叶 vs 内部节点同位）→ 无共性骨架"


def test_discover_operand_bearing_produces_skeleton(disc_env):
    """序列2：operand-bearing 样本（lambda n: n+3·含 OPERAND 叶 n）→ 骨架（非 None·operand 已支持）。

    `n+3` + `n+5`：leaf0 operand n（两样本同 sid mv0→同槽 PARAM_0=变量同一性）·leaf1 立即数 3/5 异→PARAM_1。
    → ADD[PARAM_0, PARAM_1] arity 2（n 参数化 + 常数位参数化·= λn,c. n+c）。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda n: n + 3", seg_label="__seg_p1")
    s2 = _build(disc_env, "lambda n: n + 5", seg_label="__seg_p2")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="addnc")
    assert res is not None and res.arity == 2, "operand+异立即数 → arity 2（PARAM_0=n·PARAM_1=常数位）"


def test_discover_ctrl_store_bearing_produces_skeleton(disc_env):
    """S3 ctrl/store-迭代骨架（doc/重来_S3S4迭代机制设计 §三-bis）：Sigma（CTRL_WHILE+STORE）→ **产骨架**（非 None）。

    翻向（原 loop1 defer→现支持）：两异参名 Sigma(1,n,i) 样本 → discover_skeleton 成功·arity=1（n=PARAM·acc/idx=internal alpha）。
    internal sid alpha-重命名到 make_variable(arity+k)·避 PARAM 区·镜像 _deep_copy_subtree。反 theater：vm_proof 真验值。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda n: Sigma(1, n, i)", seg_label="__seg_s1")
    s2 = _build(disc_env, "lambda m: Sigma(1, m, i)", seg_label="__seg_s2")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="sigma")
    assert res is not None, "ctrl/store-bearing（Sigma）→ S3 须产骨架（非 defer）"
    assert res.arity == 1, "Sigma(1,n,i) arity=1（n=PARAM·acc/idx=internal·非参数）"


def test_discover_single_sample_returns_none(disc_env):
    """<2 样本 → None（须 ≥2 对齐·单样本无对齐对象）。"""
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 3 + 5", seg_label="__seg_a")
    res = discover_skeleton([s1], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="x")
    assert res is None


def test_discover_identical_samples_arity_zero_valid(disc_env):
    """两相同样本 → arity=0（无相异位·全固定位·退化但合法·结构被复制）。"""
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 3 + 5", seg_label="__seg_a")
    s2 = _build(disc_env, "lambda: 3 + 5", seg_label="__seg_b")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="copy")
    assert res is not None and res.arity == 0, "相同样本 → arity=0（结构复制）"
    # 直接 vm_proof：骨架=ADD(IMM3,IMM5)·无参·执行=8
    assert rational.eq(_run(disc_env, res.skeleton_ref, ()), make(8, 1))


# ============ SD7 确定性（两独立 backend bit-identical） ============

def test_discover_bit_identical():
    """同两样本两独立 backend 跑 → arity + 骨架执行结果 bit-identical（确定性）。"""

    def _discover_and_run():
        b = DictBackend(); bootstrap(b); register_composes_attr(b)
        reg = SpaceRegistry(b); sp = AbstractSpace.create(reg, "core")
        es = EdgeStore(b); ci = ConceptIndex(b)
        sid = sp.space_id
        s1 = ci.ensure("__seg_a", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        build_composes_from_arith("lambda: 3 + 5", concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH, root_ref=s1)
        s2 = ci.ensure("__seg_b", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        build_composes_from_arith("lambda: 4 + 7", concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH, root_ref=s2)
        res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                space_id=sid, source=SOURCE_MATH, skeleton_label="add_two")
        assert res is not None
        g = ConceptGraph(b)
        ch, op, nd, imm, st = g.read_composes_tree(res.skeleton_ref)
        instrs = compile_graph(res.skeleton_ref, ch, op, nd,
                               immediate_of=imm or None, store_target_of=st or None)
        val = execute(instrs, {make_variable(0): make(10, 1), make_variable(1): make(20, 1)})
        return res.arity, val

    a1, v1 = _discover_and_run()
    a2, v2 = _discover_and_run()
    assert a1 == a2 == 2
    assert v1 == v2 == make(30, 1)


# ============ 序列6-min 生产触发（auto_discover_operators + formal_train·de-theater 序列1·§八.6） ============
#
# 序列1（discover_skeleton）零生产 caller = theater（§8.7 line306）。序列6-min 给它真生产 caller：
# formal_train 触发器（内容哈希独立根·绕 observe 多程序撞 struct_ref）→ auto_discover_operators
# （group by shape + discover_skeleton + register_arith_operator）→ 注册名经 _try_inline_learned 真 consumption。
# 反 theater 断言散布：①真生产 caller（formal_train/_discover_arith_operators 调）②真抽骨架+注册
# ③inline+vm_proof 真消费（非死写）④<2/异形→无发现（非绿测试掩零学习）⑤幂等+bit-identical。

def _arith_item(arith_src: str):
    """造算术域 CollectedItem（lambda 记号·formal_train/_discover_arith_operators 喂料）。"""
    from pure_integer_ai.experiments.collection import CollectedItem
    return CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                         source=SOURCE_MATH, arith_source=arith_src)


def test_auto_discover_same_shape_registers_operator(disc_env):
    """序列6-min 核心：两同形立即数程序 → auto_discover 抽骨架+注册（真生产机制·de-theater）。

    lambda:5*5 + lambda:6*6 → MUL[PARAM,PARAM] arity2 → 注册 __op_disc_{tag}。
    反 theater ①真抽骨架（arity=2）②真注册（name 节点 ATTR_OPERATOR_DEF 指向 skeleton）③sample_count=2。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__p_a")
    s2 = _build(disc_env, "lambda: 6 * 6", seg_label="__p_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    assert len(ops) == 1
    op = ops[0]
    assert op.arity == 2, "MUL 两相异立即数位 → arity=2"
    assert op.sample_count == 2
    # ②真注册：name 节点 ATTR_OPERATOR_DEF 落·指向 skeleton_ref
    name_ref = ci.lookup(op.name, sid)
    assert name_ref is not None, "注册名须落盘可查"
    name_attrs = read_composes_attrs(b, name_ref)
    assert (name_attrs[ATTR_OPERATOR_DEF][0],
            name_attrs[ATTR_OPERATOR_DEF][1]) == op.skeleton_ref
    # ①骨架 ATTR_ORIGIN=discovered
    assert read_composes_attrs(b, op.skeleton_ref).get(ATTR_ORIGIN) == (ORIGIN_DISCOVERED, 0)


def test_auto_discover_heterogeneous_shapes_no_discovery(disc_env):
    """反 theater 锚点：异形程序（MUL vs ADD·不同 shape_signature）→ 无一组≥2 → 空发现。"""
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__p_a")   # MUL 形
    s2 = _build(disc_env, "lambda: 3 + 7", seg_label="__p_b")   # ADD 形（异形）
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    assert ops == [], "异形 → 无同形组≥2 → 空发现"


def test_auto_discover_single_program_no_discovery(disc_env):
    """反 theater 锚点：单程序（<K=2）→ 无发现（K=MIN_DISCOVER_SAMPLES 元定义常量）。"""
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__p_a")
    ops = auto_discover_operators([s1], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    assert ops == []


def test_auto_discover_dedups_identical_programs(disc_env):
    """去重：同一程序两份（同根）→ 只算一份样本 → <K → 无发现（非把重复当多样本）。"""
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__p_a")
    ops = auto_discover_operators([s1, s1], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    assert ops == [], "同根重复须去重为单样本 → <K → 无发现"


def test_auto_discover_idempotent_across_calls(disc_env):
    """幂等：两次调 auto_discover 同语料 → 第二次 lookup 门 skip·不重抽不撞名（跨 run/续训 bit-identical）。"""
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__p_a")
    s2 = _build(disc_env, "lambda: 6 * 6", seg_label="__p_b")
    roots = [s1, s2]
    ops1 = auto_discover_operators(roots, concept_index=ci, edge_store=es, backend=b,
                                   space_id=sid, source=SOURCE_MATH)
    ops2 = auto_discover_operators(roots, concept_index=ci, edge_store=es, backend=b,
                                   space_id=sid, source=SOURCE_MATH)
    assert len(ops1) == 1
    assert ops2 == [], "第二次须幂等 skip（同形已注册·lookup 门·不重抽）"
    # name 仍唯一指向原 skeleton（无新骨架生成）
    name_ref = ci.lookup(ops1[0].name, sid)
    assert (read_composes_attrs(b, name_ref)[ATTR_OPERATOR_DEF][0],
            read_composes_attrs(b, name_ref)[ATTR_OPERATOR_DEF][1]) == ops1[0].skeleton_ref


def test_auto_discovered_operator_inlined_and_executed(disc_env):
    """反 theater 核心：注册算子经 inline+β+vm_proof 真复现（非死写·§8.7 序列6=被复用注册）。

    发现 MUL[PARAM,PARAM](arity2) 后·新程序 lambda:<name>(7,7) Call 路径嫁接骨架 → execute → 49。
    证写入的注册名被 _try_inline_learned 真 read 消费（落 struct_ref 非摆设）。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__p_a")
    s2 = _build(disc_env, "lambda: 6 * 6", seg_label="__p_b")
    op = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                 space_id=sid, source=SOURCE_MATH)[0]
    # 消费程序：引用发现的算子（Call 路径→inline+β·实参 7,7 进 PARAM 槽）
    consumer_root = _build(disc_env, f"lambda: {op.name}(7, 7)", seg_label="__consumer")
    result = _run(disc_env, consumer_root, input_args=())   # nullary·值经 β 进槽
    assert rational.eq(result, make(49, 1)), f"inline 消费须复现 7*7=49·得 {result}"


def test_auto_discover_bit_identical():
    """确定性：两独立 backend 同语料 → 同发现（name/arity bit-identical·Hasher 固定种子）。"""

    def _disc():
        b = DictBackend(); bootstrap(b); register_composes_attr(b)
        reg = SpaceRegistry(b); sp = AbstractSpace.create(reg, "core")
        es = EdgeStore(b); ci = ConceptIndex(b)
        sid = sp.space_id
        env = (b, sid, es, ci, None)
        s1 = _build(env, "lambda: 5 * 5", seg_label="__p_a")
        s2 = _build(env, "lambda: 6 * 6", seg_label="__p_b")
        ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                      space_id=sid, source=SOURCE_MATH)
        return ops

    ops1 = _disc()
    ops2 = _disc()
    assert len(ops1) == len(ops2) == 1
    assert ops1[0].name == ops2[0].name, "Hasher 固定种子 → 同形同 name（跨 run bit-identical）"
    assert ops1[0].arity == ops2[0].arity == 2


# ---- formal_train 生产触发器 e2e（真生产 caller·非 test-only helper）----

def test_discover_arith_operators_trigger_via_train_context():
    """生产触发器 _discover_and_recognize_arith_operators（make_train_context）→ 算术语料 → DiscoveredOperator。

    证 de-theater：discover_skeleton 在生产 TrainContext（非 test fixture）里被调·产物落生产 backend。
    2 样本同形==2 → 全发现·无 held-out → recognitions=[]（序列3 识别须 ≥3 留 held-out）。
    """
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_arith_operators
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_arith_item("lambda: 5 * 5"), _arith_item("lambda: 6 * 6")]
    ops, recs, gen = _discover_and_recognize_arith_operators(ctx, corpus)
    assert len(ops) == 1
    op = ops[0]
    assert op.arity == 2
    assert recs == [], "2 样本同形==2 → 全发现无 held-out → 无可识别（诚实·序列3 须 ≥3 留 held-out）"
    assert gen.total_held_out == 0 and gen.verified == 0, "无 held-out → 空泛化汇总"
    # 骨架真落生产 backend（ATTR_ORIGIN=discovered）
    assert read_composes_attrs(b, op.skeleton_ref).get(ATTR_ORIGIN) == (ORIGIN_DISCOVERED, 0)


def test_discover_arith_operators_content_hash_root_idempotent():
    """生产触发器幂等：同语料两次跑 → 内容哈希根 skip 已建（不复制边）+ lookup 门不重抽。

    证跨 run/续训 bit-identical：__disc_src_{h63} 已建有 COMPOSES 出边 → build 跳·auto_discover lookup 门跳。
    """
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_arith_operators
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_arith_item("lambda: 5 * 5"), _arith_item("lambda: 6 * 6")]
    ops1, _, _gen = _discover_and_recognize_arith_operators(ctx, corpus)
    ops2, recs2, _gen2 = _discover_and_recognize_arith_operators(ctx, corpus)
    assert len(ops1) == 1
    assert ops2 == [] and recs2 == [], "第二次须幂等（根 skip + lookup 门·不重抽不复制边·识别纯读）"


def test_formal_train_wires_discovered_operators(tmp_path):
    """formal_train e2e：算术语料 → FormalTrainResult.discovered_operators 非空（生产接线·反 theater ①真 caller）。

    发现算子真落生产 backend（skeleton ATTR_ORIGIN·非仅返回值）·证 formal_train 主入口真触发发现。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    b = DictBackend()
    corpus = [_arith_item("lambda: 5 * 5"), _arith_item("lambda: 6 * 6")]
    config = FormalTrainConfig(run_dir=str(tmp_path), run_id="disc_e2e",
                               rounds_per_stage=1)
    result = formal_train(config, corpus, backend=b)
    assert len(result.discovered_operators) == 1, "formal_train 须触发算子发现（生产 caller）"
    op = result.discovered_operators[0]
    assert op.arity == 2
    # 反 theater：骨架真落生产 backend（非仅返回值摆设）
    assert read_composes_attrs(b, op.skeleton_ref).get(ATTR_ORIGIN) == (ORIGIN_DISCOVERED, 0)


# ============ 序列3-min 识别消费（recognize_operators + formal_train held-out·生产期 READ·§八.3）===========
#
# 序列6-min 留的缺口：生产训练 loop 不引用 __op_disc_* → "存进去没人读" theater（§8.7）。序列3-min 补：
# recognize_operators 让发现骨架在生产期被真读（read_composes_tree + DFS 前序对齐 + PARAM 抽值 + 固定位值等）
# → 识别 held-out 新输入（非发现集→真泛化非循环）。caller vm_proof 验识别绑定（骨架绑参==新输入值）。
# 反 theater 断言散布：①held-out 新输入被识（非发现集）②params 真抽（DFS 阅读序）③固定位值等（非纯形状指纹）
# ④vm_proof 验识别绑定（骨架读+应用复现新输入值·READ 消费铁证）⑤异形/无算子→不识（非绿测试掩零）。


def test_recognize_heldout_input_extracts_params(disc_env):
    """序列3-min 核心：发现骨架 → 识别 held-out 新输入 → 抽 PARAM 绑定（READ 消费·§8.7 让结构被读）。

    发现 5*5,6*6 → MUL[PARAM,PARAM] arity2·识别 held-out 7*7,8*8 → params (7,7),(8,8)。
    反 theater ①held-out（7,8 非发现集{5,6}→真泛化非循环）②params 真抽。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__d_a")
    s2 = _build(disc_env, "lambda: 6 * 6", seg_label="__d_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    assert len(ops) == 1
    held_a = _build(disc_env, "lambda: 7 * 7", seg_label="__h_a")   # held-out 新输入
    held_b = _build(disc_env, "lambda: 8 * 8", seg_label="__h_b")
    recs = recognize_operators([held_a, held_b], discovered_operators=ops,
                               backend=b, space_id=sid)
    assert len(recs) == 2, "两 held-out 同形输入须都被识别"
    pv = {rec.param_values for rec in recs}
    assert pv == {((7, 1), (7, 1)), ((8, 1), (8, 1))}, f"params 须 (7,7),(8,8)·得 {pv}"
    assert all(rec.operator_name == ops[0].name for rec in recs)
    assert all(rec.arity == 2 for rec in recs)


def test_recognize_fixed_position_value_must_match(disc_env):
    """反 theater ③固定位值等（非纯形状指纹）：骨架 ADD[PARAM,IMM3]·识 7+3（param 7）·不识 7+4（fixed 3≠4）。

    shape 同（ADD[叶,叶]）但固定位 IMM3 值异（7+4 的 4≠3）→ _align_walk 固定位值等门拒→不识别。
    证识别非纯形状指纹（比 coverage_overlap shape-only 更精）·须值对齐。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 5 + 3", seg_label="__d_a")
    s2 = _build(disc_env, "lambda: 6 + 3", seg_label="__d_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    assert len(ops) == 1 and ops[0].arity == 1, "ADD[PARAM,IMM3] → arity=1"
    match = _build(disc_env, "lambda: 7 + 3", seg_label="__m")      # 固定位 3 同 → 识别
    nomatch = _build(disc_env, "lambda: 7 + 4", seg_label="__nm")   # 固定位 3≠4 → 不识
    recs_ok = recognize_operators([match], discovered_operators=ops, backend=b, space_id=sid)
    recs_no = recognize_operators([nomatch], discovered_operators=ops, backend=b, space_id=sid)
    assert len(recs_ok) == 1 and recs_ok[0].param_values == ((7, 1),), "固定位同→识别·param=(7,)"
    assert recs_no == [], "固定位值异（3≠4）→ 不识别（非纯形状指纹）"


def test_recognize_wrong_shape_not_recognized(disc_env):
    """反 theater ⑤锚点：异形输入（骨架 MUL·输入 ADD）→ shape 不同 → 不识别。"""
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__d_a")
    s2 = _build(disc_env, "lambda: 6 * 6", seg_label="__d_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    add_input = _build(disc_env, "lambda: 7 + 7", seg_label="__add")   # ADD 形（骨架 MUL）
    recs = recognize_operators([add_input], discovered_operators=ops, backend=b, space_id=sid)
    assert recs == [], "异形（ADD vs MUL 骨架）→ 不识别"


def test_recognize_params_vm_proof_verified(disc_env):
    """反 theater ④vm_proof 验识别绑定（READ 消费铁证）：骨架绑识别 params 执行 == held-out 新输入执行值。

    发现 MUL 骨架·识别 held-out 7*7 → params (7,7)·骨架(7,7)=49 == 新输入 7*7=49。
    证发现骨架被读+应用到新输入复现其值（非仅形状匹配·真计算消费）。caller 级 vm_proof（本模块 L5 不调 L7）。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__d_a")
    s2 = _build(disc_env, "lambda: 6 * 6", seg_label="__d_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    held = _build(disc_env, "lambda: 7 * 7", seg_label="__h")
    rec = recognize_operators([held], discovered_operators=ops, backend=b, space_id=sid)[0]
    input_args = tuple(p[0] for p in rec.param_values)   # PARAM sid=make_variable(0..arity-1)
    skel_val = _run(disc_env, ops[0].skeleton_ref, input_args)
    input_val = _run(disc_env, held, ())                  # held-out 输入 nullary 执行
    assert rational.eq(skel_val, input_val), "骨架绑识别 params 须 == 新输入值（READ 消费复现）"
    assert rational.eq(input_val, make(49, 1)), "新输入 7*7 须 == 49"


def test_recognize_no_discovered_operators_returns_empty(disc_env):
    """反 theater ⑤锚点：无已学骨架（discovered=[]）→ 无可识别→空（不伪造识别）。"""
    b, sid, es, ci, _ = disc_env
    inp = _build(disc_env, "lambda: 7 * 7", seg_label="__i")
    recs = recognize_operators([inp], discovered_operators=[], backend=b, space_id=sid)
    assert recs == [], "无已学骨架 → 无可识别"


def test_recognize_dedups_duplicate_input_roots(disc_env):
    """去重：同输入根两份 → 只识别一次（保序确定·非把重复当多识别）。"""
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__d_a")
    s2 = _build(disc_env, "lambda: 6 * 6", seg_label="__d_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    held = _build(disc_env, "lambda: 7 * 7", seg_label="__h")
    recs = recognize_operators([held, held], discovered_operators=ops, backend=b, space_id=sid)
    assert len(recs) == 1, "同根重复须去重为单识别"


def test_recognize_nested_sub_reading_order_vm_proof(disc_env):
    """识别 PARAM 阅读序回归（非交换 SUB·parallel discover 的 sub3 回归测试的识别侧）。

    发现 (10-3)-2,(20-5)-3 → SUB(SUB(P0,P1),P2) arity3·识别 held-out (100-10)-1 → params (100,10,1)
    ·骨架(100,10,1)=(100-10)-1=89 == 新输入=89。DFS 前序阅读序对齐（BFS 层序会抽错序）。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 10 - 3 - 2", seg_label="__d_a")
    s2 = _build(disc_env, "lambda: 20 - 5 - 3", seg_label="__d_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    assert len(ops) == 1 and ops[0].arity == 3
    held = _build(disc_env, "lambda: 100 - 10 - 1", seg_label="__h")
    rec = recognize_operators([held], discovered_operators=ops, backend=b, space_id=sid)[0]
    assert rec.param_values == ((100, 1), (10, 1), (1, 1)), (
        "params 须 DFS 阅读序 (100,10,1)·非 BFS 层序")
    input_args = tuple(p[0] for p in rec.param_values)
    skel_val = _run(disc_env, ops[0].skeleton_ref, input_args)
    input_val = _run(disc_env, held, ())
    assert rational.eq(skel_val, input_val), "骨架绑参须 == 新输入值"
    assert rational.eq(input_val, make(89, 1)), "新输入 (100-10)-1 须 == 89（阅读序对齐）"


# ---- formal_train 生产 READ 触发 e2e（held-out split·真生产 caller）----

def test_discover_and_recognize_arith_split_heldout(disc_env):
    """生产触发器 held-out split：4 同形 → 发现身首 2·识别 held-out 余 2（真泛化非循环·§八.3）。

    反 theater：识别 params 是 held-out {7,7},{8,8}（非发现集 {5,5},{6,6}）→ 证 per-shape 留 held-out
    真接通·识别新输入非循环 theater。
    """
    from pure_integer_ai.experiments.formal_train import (
        make_train_context, _discover_and_recognize_arith_operators)
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_arith_item(f"lambda: {n} * {n}") for n in (5, 6, 7, 8)]
    ops, recs, gen = _discover_and_recognize_arith_operators(ctx, corpus)
    assert len(ops) == 1, "4 同形 → 发现身 1 骨架"
    assert len(recs) == 2, "held-out 余 2 须被识别"
    pv = {rec.param_values for rec in recs}
    assert pv == {((7, 1), (7, 1)), ((8, 1), (8, 1))}, (
        f"识别须 held-out (7,7),(8,8)（非发现集 (5,5),(6,6)）·得 {pv}")
    # 验证半闭环：两 held-out 都 vm_proof 验过（骨架(7,7)=49==7*7·(8,8)=64==8*8）→ 泛化率 1000
    assert gen.total_held_out == 2 and gen.recognized == 2 and gen.verified == 2, (
        f"两 held-out 须都识别+vm_proof 验·得 {gen}")
    assert gen.rate_permille == 1000, "全验 → 泛化率 1000"


def test_discover_and_recognize_idempotent():
    """生产触发器幂等：4 同形语料两次跑 → 第二次 discovered=[] (lookup 门) + recognitions=[] (无发现无可识)。"""
    from pure_integer_ai.experiments.formal_train import (
        make_train_context, _discover_and_recognize_arith_operators)
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_arith_item(f"lambda: {n} * {n}") for n in (5, 6, 7, 8)]
    ops1, recs1, gen1 = _discover_and_recognize_arith_operators(ctx, corpus)
    ops2, recs2, gen2 = _discover_and_recognize_arith_operators(ctx, corpus)
    assert len(ops1) == 1 and len(recs1) == 2
    assert gen1.verified == 2, "首次跑须验两 held-out"
    assert ops2 == [] and recs2 == [], "第二次须幂等（根 skip + lookup 门·识别纯读）"
    assert gen2.verified == 0, "第二次无发现/识别 → 空泛化汇总"


def test_formal_train_wires_recognitions(tmp_path):
    """formal_train e2e：4 同形算术语料 → FormalTrainResult.recognitions 非空（生产 READ 接线·反 theater）。

    证 formal_train 主入口真触发 held-out 识别（序列3-min 生产期 READ 消费·解序列6-min 留的 theater）。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    b = DictBackend()
    corpus = [_arith_item(f"lambda: {n} * {n}") for n in (5, 6, 7, 8)]
    config = FormalTrainConfig(run_dir=str(tmp_path), run_id="rec_e2e", rounds_per_stage=1)
    result = formal_train(config, corpus, backend=b)
    assert len(result.discovered_operators) == 1, "formal_train 须触发算子发现"
    assert len(result.recognitions) == 2, "formal_train 须触发 held-out 识别（生产 READ）"
    pv = {rec.param_values for rec in result.recognitions}
    assert pv == {((7, 1), (7, 1)), ((8, 1), (8, 1))}, f"识别 held-out params·得 {pv}"


# ============ 序列3-min 验证半闭环（_verify_generalization + execute_composes_value·§8.7 反 theater）===========
#
# 序列3-min 识别留的边界：recognitions terminal（写 result.recognitions·生产 loop 不读）。验证半闭环补——
# _verify_generalization 对每个识别做 caller 级 vm_proof（execute_composes_value）：骨架绑识别 params 执行
# == held-out 新输入执行值（识别=结构对齐·vm_proof=执行比对·两路独立方法判同一事 = 反 theater 铁证）。
# verified/total_held_out = 泛化率（学到的能力覆盖多少新输入·量化"学到能力"）。生成侧洗净循环 defer。
# 反 theater 断言：①vm_proof 真执行（值暴露·非 stub）②两路独立（结构对齐 vs 执行比对）③mismatch 可检出
# （错参→错值→不 verified·证明 vm_proof 有牙）④泛化率量化（verified/total）⑤非交换 SUB 阅读序验。


def test_execute_composes_value_exposes_value(disc_env):
    """execute_composes_value（vm_proof 值暴露版）：COMPOSES 根 → 编译执行 → Rational 值（非 1/0 比对）。

    反 theater ①vm_proof 真执行：nullary 程序 7*7 → 49（值暴露·证非 stub 恒真）。
    """
    from pure_integer_ai.training.vm_proof import execute_composes_value
    b, sid, es, ci, g = disc_env
    nullary = _build(disc_env, "lambda: 7 * 7", seg_label="__ev_a")
    v = execute_composes_value(g, nullary, ())
    assert rational.eq(v, make(49, 1)), f"nullary 7*7 须 → 49·得 {v}"


def test_execute_composes_value_binds_params_skeleton(disc_env):
    """execute_composes_value 绑参执行骨架：发现 MUL(P0,P1) 骨架·绑 (7,1),(7,1) → 49。

    证 PARAM 槽 make_variable(i) ← param_values[i] 绑参约定（与 inline arg_subst 契约一致）。
    """
    from pure_integer_ai.training.vm_proof import execute_composes_value
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__ev_b1")
    s2 = _build(disc_env, "lambda: 6 * 6", seg_label="__ev_b2")
    op = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                 space_id=sid, source=SOURCE_MATH)[0]
    v = execute_composes_value(g, op.skeleton_ref, ((7, 1), (7, 1)))
    assert rational.eq(v, make(49, 1)), f"骨架绑 (7,7) 须 → 49·得 {v}"


def test_execute_composes_value_mismatch_detectable(disc_env):
    """反 theater ③mismatch 可检出：骨架绑**错参** (7,8) → 56 ≠ 输入 7*7=49（vm_proof 有牙）。

    证 vm_proof 非恒真 stub：错参产错值·与输入值不等 → 不计 verified（若识别抽错参·vm_proof 必抓获）。
    """
    from pure_integer_ai.training.vm_proof import execute_composes_value
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__ev_c1")
    s2 = _build(disc_env, "lambda: 6 * 6", seg_label="__ev_c2")
    op = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                 space_id=sid, source=SOURCE_MATH)[0]
    v_wrong = execute_composes_value(g, op.skeleton_ref, ((7, 1), (8, 1)))   # 错参 7,8 → 56
    v_input = execute_composes_value(g, _build(disc_env, "lambda: 7 * 7", seg_label="__ev_cin"), ())
    assert rational.eq(v_wrong, make(56, 1)), "错参 (7,8) → 56"
    assert not rational.eq(v_wrong, v_input), "错参值 56 ≠ 输入 49 → mismatch 可检出（vm_proof 有牙）"


def test_verify_generalization_noncommutative_sub():
    """反 theater ⑤非交换 SUB 阅读序验：发现 (10-3)-2,(20-5)-3 → 识别+验 held-out (100-10)-1 → 89。

    验证半闭环在非交换算子上稳（PARAM 阅读序对齐·骨架(100,10,1)=(100-10)-1=89 == 输入 89）。
    两路独立：识别 _align_walk 抽 params (100,10,1)·vm_proof 执行比对 → 同值 89。
    """
    from pure_integer_ai.experiments.formal_train import (
        make_train_context, _discover_and_recognize_arith_operators)
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_arith_item(f"lambda: {x} - {y} - {z}")
              for x, y, z in ((10, 3, 2), (20, 5, 3), (100, 10, 1), (200, 20, 2))]
    ops, recs, gen = _discover_and_recognize_arith_operators(ctx, corpus)
    assert len(ops) == 1 and len(recs) == 2, "4 同形 SUB → 发现身 1·识别 held-out 2"
    # held-out (100-10)-1=89·(200-20)-2=178·都须 vm_proof 验过（阅读序对齐·非交换）
    assert gen.verified == 2, f"两 held-out 须都 vm_proof 验（非交换阅读序）·得 verified={gen.verified}"
    assert gen.rate_permille == 1000
    pv = {rec.param_values for rec in recs}
    assert pv == {((100, 1), (10, 1), (1, 1)), ((200, 1), (20, 1), (2, 1))}, (
        f"held-out params 须阅读序 (100,10,1)/(200,20,2)·得 {pv}")


def test_verify_generalization_two_independent_methods_agree(disc_env):
    """反 theater ②两路独立：识别（结构对齐 _align_walk）与 vm_proof（执行比对）判同一事且同果。

    识别抽 params 经结构对齐·vm_proof 独立执行骨架绑参比对输入值——两路方法独立·同果 = 反 theater
    铁证（非循环·识别非自证）。square: 骨架(7,7)=49（执行）== 输入 7*7=49（执行）·识别 params (7,7)（结构）。
    """
    from pure_integer_ai.training.vm_proof import execute_composes_value
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__ti_a")
    s2 = _build(disc_env, "lambda: 6 * 6", seg_label="__ti_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    held = _build(disc_env, "lambda: 7 * 7", seg_label="__ti_h")
    rec = recognize_operators([held], discovered_operators=ops, backend=b, space_id=sid)[0]
    # 路 A：识别结构对齐抽 params
    assert rec.param_values == ((7, 1), (7, 1)), "结构对齐抽 (7,7)"
    # 路 B：vm_proof 独立执行比对
    v_skel = execute_composes_value(g, ops[0].skeleton_ref, rec.param_values)
    v_input = execute_composes_value(g, held, ())
    assert rational.eq(v_skel, v_input) and rational.eq(v_input, make(49, 1)), (
        "两路独立同果：骨架(7,7)=49 == 输入 7*7=49")


def test_formal_train_wires_generalization(tmp_path):
    """formal_train e2e：4 同形算术语料 → FormalTrainResult.generalization 非空（验证半闭环生产接线）。

    证 formal_train 主入口真触发 vm_proof 验泛化（识别产物 recognitions 被真消费·解 terminal 边界·反 theater）。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    b = DictBackend()
    corpus = [_arith_item(f"lambda: {n} * {n}") for n in (5, 6, 7, 8)]
    config = FormalTrainConfig(run_dir=str(tmp_path), run_id="gen_e2e", rounds_per_stage=1)
    result = formal_train(config, corpus, backend=b)
    assert result.generalization is not None, "formal_train 须产 generalization 汇总"
    gen = result.generalization
    assert gen.total_held_out == 2 and gen.recognized == 2 and gen.verified == 2, (
        f"两 held-out 须都识别+vm_proof 验·得 {gen}")
    assert gen.rate_permille == 1000, "全验 → 泛化率 1000"


# ============ 序列2 operand 对应 + 变量同一性（让发现/识别产有意义算子 square/double） ============

def test_discover_square_variable_identity_arity1(disc_env):
    """序列2 核心：operand 样本 lambda x:x*x + lambda y:y*y → square 骨架 arity 1（变量同一性）。

    两 OPERAND 叶同 sid（x=mv0·两出现同 sid）→ 同槽 PARAM_0 复用 → MUL[PARAM_0,PARAM_0] arity 1。
    对比 loop1 退化：立即数 5*5/6*6 → mul arity 2（相异槽·无同一性）。
    """
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda x: x * x", seg_label="__sq_a")
    s2 = _build(disc_env, "lambda y: y * y", seg_label="__sq_b")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="square")
    assert res is not None and res.arity == 1, "x*x 两 OPERAND 同 sid → 同槽 → arity 1 = square"


def test_discover_square_inline_reuse_vm_proof(disc_env):
    """序列2 反 theater：发现的 square 骨架注册后可被 inline 复用（Call+β·L1.5）·vm_proof 复现+泛化。

    square=λx.x*x 注册 → lambda: square(7) inline β → 7*7=49 == 直接 7*7=49 · square(9)→81 泛化新值。
    """
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda x: x * x", seg_label="__sq_a")
    s2 = _build(disc_env, "lambda y: y * y", seg_label="__sq_b")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="square")
    register_arith_operator(b, ci, "square", res.skeleton_ref, arity=res.arity)
    root_use = _build(disc_env, "lambda: square(7)", seg_label="__sq_use")
    root_direct = _build(disc_env, "lambda: 7 * 7", seg_label="__sq_dir")
    assert _run(disc_env, root_use, ()) == _run(disc_env, root_direct, ()) == make(49, 1), (
        "square(7) inline β → 7*7=49 复现样本")
    root_gen = _build(disc_env, "lambda: square(9)", seg_label="__sq_gen")
    assert rational.eq(_run(disc_env, root_gen, ()), make(81, 1)), "square(9)→81 泛化新值"


def test_recognize_square_immediate_input_vm_proof(disc_env):
    """序列2 READ 消费：square 骨架识别 held-out 立即数 7*7 → params (7,) arity 1·vm_proof 49==49。

    骨架 MUL[PARAM_0,PARAM_0]·input 7*7：leaf0 slot0→7·leaf1 slot0(同槽)→7 须等 ✓。params=(7,)。
    """
    from pure_integer_ai.training.vm_proof import execute_composes_value
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda x: x * x", seg_label="__sq_a")
    s2 = _build(disc_env, "lambda y: y * y", seg_label="__sq_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    assert ops[0].arity == 1
    held = _build(disc_env, "lambda: 7 * 7", seg_label="__sq_h")
    rec = recognize_operators([held], discovered_operators=ops, backend=b, space_id=sid)[0]
    assert rec.param_values == ((7, 1),), "square 识别 7*7 → params (7,) arity 1"
    v_skel = execute_composes_value(g, ops[0].skeleton_ref, rec.param_values)
    v_input = execute_composes_value(g, held, ())
    assert v_skel is not None and rational.eq(v_skel, v_input) and rational.eq(v_input, make(49, 1))


def test_recognize_square_rejects_seven_times_eight(disc_env):
    """序列2 变量同一性牙真：square 拒 7*8（同槽值须等·第二叶 8≠已绑 7）。

    mul 会认 7*8=mul(7,8)·square 不认（slot0 第二叶 8≠首叶 7）—— 差异实证变量同一性非恒真 stub。
    """
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda x: x * x", seg_label="__sq_a")
    s2 = _build(disc_env, "lambda y: y * y", seg_label="__sq_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    held = _build(disc_env, "lambda: 7 * 8", seg_label="__sq_78")
    assert recognize_operators([held], discovered_operators=ops, backend=b, space_id=sid) == [], (
        "square 拒 7*8（slot0 第二叶 8≠首叶 7·变量同一性）")


def test_discover_double_variable_identity(disc_env):
    """序列2：double=λx.x+x（ADD 两 OPERAND 同 sid）arity 1·认 7+7→(7,)·拒 7+8。"""
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda x: x + x", seg_label="__db_a")
    s2 = _build(disc_env, "lambda y: y + y", seg_label="__db_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    assert ops[0].arity == 1, "x+x 两 OPERAND 同 sid → arity 1 = double"
    held_ok = _build(disc_env, "lambda: 7 + 7", seg_label="__db_h7")
    rec = recognize_operators([held_ok], discovered_operators=ops, backend=b, space_id=sid)[0]
    assert rec.param_values == ((7, 1),), "double 认 7+7 → (7,)"
    held_no = _build(disc_env, "lambda: 7 + 8", seg_label="__db_78")
    assert recognize_operators([held_no], discovered_operators=ops, backend=b, space_id=sid) == [], (
        "double 拒 7+8（slot0 第二叶 8≠7）")


def test_discover_cross_sample_split_rejected(disc_env):
    """序列2 cross-sample 一致性门（禁止拆分）：sample0=square(x*x) + sample1=mul(a,b:a*b) → None。

    sample0 两叶同 sid（slot0·square）·sample1 两叶异 sid（mv0,mv1）→ sample1 在 sample0 同槽位(slot0)
    出现两不同 sid → 拆 sample0 变量 → 非实例 → _NoSkeleton。（mul 非 square 实例）
    """
    b, sid, es, ci, g = disc_env
    s_sq = _build(disc_env, "lambda x: x * x", seg_label="__sp_sq")
    s_mul = _build(disc_env, "lambda p, q: p * q", seg_label="__sp_mul")
    res = discover_skeleton([s_sq, s_mul], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="split")
    assert res is None, "square+mul → mul 拆 square 变量 → 非实例 → None"


def test_discover_cross_sample_collapse_allowed(disc_env):
    """序列2 cross-sample 一致性门（允许坍缩）：sample0=mul(a,b:a*b) + sample1=square(x*x) → mul arity 2。

    sample0 两叶异 sid（slot0,slot1·mul）·sample1 两叶同 sid（mv0·square）→ square 是 mul 实例
    （两槽绑同变量·坍缩）→ 通过。骨架=mul arity 2（sample0 模板）。
    """
    b, sid, es, ci, g = disc_env
    s_mul = _build(disc_env, "lambda p, q: p * q", seg_label="__cl_mul")
    s_sq = _build(disc_env, "lambda x: x * x", seg_label="__cl_sq")
    res = discover_skeleton([s_mul, s_sq], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_MATH, skeleton_label="collapse")
    assert res is not None and res.arity == 2, "mul+square → square 是 mul 实例（坍缩）→ mul arity 2"


def test_recognize_square_operand_input_probe_verified(disc_env):
    """operand-input 识别核心（探针值执行比对）：square 骨架识 operand 输入 λz:z*z → is_operand_input=True·探针 vm_proof 验。

    骨架 MUL[PARAM_0,PARAM_0]（arity1）·input λz:z*z=MUL[OP_z,OP_z]：skeleton slot0 两叶对齐 input operand slot0（同 z·变量同一性一致）。
    探针 z=_PROBE_VALUES[0]=2：骨架(2)=4 == input(2)=4。Recognition.is_operand_input=True·operand_binding=(0,)·param_values=派生探针值 ((2,1),)。
    within-run operand 闭环：operand 语料 held-out（参数化输入）现可识别（序列2 defer 项收口）。
    """
    from pure_integer_ai.training.vm_proof import execute_composes_value
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda x: x * x", seg_label="__oi_a")
    s2 = _build(disc_env, "lambda y: y * y", seg_label="__oi_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    held_op = _build(disc_env, "lambda z: z * z", seg_label="__oi_h")
    rec = recognize_operators([held_op], discovered_operators=ops, backend=b, space_id=sid)[0]
    assert rec.is_operand_input, "operand 输入须 is_operand_input=True"
    assert rec.operand_binding == (0,), f"skeleton slot0 → input slot0·binding=(0,)·得 {rec.operand_binding}"
    # 探针 vm_proof：骨架绑派生探针 == input 绑反演探针（_PROBE_VALUES[0]=2·z=2）
    probe = rec.param_values[0][0]
    assert probe == 2, f"首探针须 _PROBE_VALUES[0]=2·得 {probe}"
    v_skel = execute_composes_value(g, ops[0].skeleton_ref, rec.param_values)
    v_input = execute_composes_value(g, held_op, (rec.param_values[0],))   # input_arity=1·z=探针[0]（反演自 binding）
    assert v_skel is not None and v_input is not None
    assert rational.eq(v_skel, make(4, 1)) and rational.eq(v_input, make(4, 1)), (
        f"探针 z=2：骨架(2)={probe*probe}=4 == input(2)=4（operand-input 识别+探针验）")


def test_formal_train_wires_square_discovery(tmp_path):
    """序列2 formal_train 生产发现（WRITE de-theater）：纯 operand 语料 → formal_train 真发现 square（arity 1）。

    纯 operand 语料（x*x/y*y/z*z·同 shape [MUL,LEAF,LEAF]·同变量同一性）→ 同组 ≥3 → 发现首2 → square arity 1
    （非退化 mul arity 2）。证 formal_train 主入口真触发 auto_discover 从 operand 语料抽 square 骨架（生产期 WRITE）。
    **诚实（对抗审计 Finding #2）**：纯 operand 语料避混类 shape 组序依赖（混 operand+立即数→首K 跨类→_NoSkeleton·
    序决定 discover/recognize 分配=脆弱）。held-out operand 识别+泛化见 test_formal_train_operand_corpus_generalization。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    b = DictBackend()
    corpus = [_arith_item(s) for s in ("lambda x: x * x", "lambda y: y * y", "lambda z: z * z")]
    config = FormalTrainConfig(run_dir=str(tmp_path), run_id="sq_disc_e2e", rounds_per_stage=1)
    result = formal_train(config, corpus, backend=b)
    arities = {op.arity for op in result.discovered_operators}
    assert 1 in arities, f"formal_train 须从 operand 语料发现 arity 1（square·变量同一性）·得 {arities}"


# ============ operand-input 识别（探针值执行比对·补序列2 operand READ 闭环·within-run 收口） ============
#
# 序列2 让发现产 operand 算子（square）·序列3-min 让识别 immediate 实例（7*7）·operand 输入（λz:z*z）识别原 defer
# （_align_walk 骨架 PARAM 遇 input OPERAND 叶→False）→ operand 语料 held-out 识别率恒 0。本步补 operand-input 识别：
# _align_walk operand 分支（变量同一性结构等价判定=真牙）+ 探针值执行验证（复用 execute_composes_value·input 探针纯从
# Recognition 字段反演）。Recognition 加 is_operand_input/operand_binding。within-run operand 闭环收口。
# 反 theater 断言：①operand 输入被识（is_operand_input=True）②变量同一性牙（拒 λp,q:p*q 为 square）③探针 vm_proof 验
# （骨架绑探针==input 绑反演探针）④坍缩允（mul 识 square 为 mul(z,z)）⑤混合 input 兼容（λz:z+3）⑥formal_train 泛化率>0。


def test_recognize_square_rejects_two_distinct_operands(disc_env):
    """反 theater ②变量同一性牙真：square 骨架拒 λp,q:p*q（两异 operand 对齐同 skeleton slot0→冲突）。

    square 两叶同槽 slot0·input p*q 两叶异 operand（p=slot0·q=slot1）→ skeleton slot0 首对 p(slot0)·次对 q(slot1)·
    冲突→拒。对比 mul 识 p*q（异槽·无冲突）·差异实证 operand-input 变量同一性牙真（非恒真 stub）。
    """
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda x: x * x", seg_label="__ab_a")
    s2 = _build(disc_env, "lambda y: y * y", seg_label="__ab_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    held = _build(disc_env, "lambda p, q: p * q", seg_label="__ab_h")   # 两异 operand（注：a 是保留名·用 p,q）
    assert recognize_operators([held], discovered_operators=ops, backend=b, space_id=sid) == [], (
        "square 拒 p*q（两异 operand 对齐同 slot0·变量同一性冲突）")


def test_recognize_mul_collapses_square_operand_input(disc_env):
    """反 theater ④坍缩允许：mul 骨架识 operand 输入 λz:z*z 为 mul(z,z)（两 skeleton slot→同 input slot·坍缩）。

    mul=λp,q:p*q arity2·input λz:z*z 两叶同 operand(z=slot0)→skeleton slot0→input slot0·skeleton slot1→input slot0
    （不同 skeleton slot→同 input slot=坍缩·合法·square 是 mul 实例 mul(z,z)）。operand_binding=(0,0)·探针 z=2：mul(2,2)=4==input(2)=4。
    """
    from pure_integer_ai.training.vm_proof import execute_composes_value
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda p, q: p * q", seg_label="__cm_a")
    s2 = _build(disc_env, "lambda m, n: m * n", seg_label="__cm_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    assert ops[0].arity == 2, "p*q/m*n → mul arity 2"
    held = _build(disc_env, "lambda z: z * z", seg_label="__cm_h")
    rec = recognize_operators([held], discovered_operators=ops, backend=b, space_id=sid)[0]
    assert rec.is_operand_input and rec.operand_binding == (0, 0), (
        f"坍缩：两 skeleton slot→input slot0·binding=(0,0)·得 {rec.operand_binding}")
    assert rec.param_values == ((2, 1), (2, 1)), f"坍缩探针：两 slot←探针[0]=2·得 {rec.param_values}"
    v_skel = execute_composes_value(g, ops[0].skeleton_ref, rec.param_values)         # mul(2,2)=4
    v_input = execute_composes_value(g, held, (rec.param_values[0],))                 # λz:z*z(2)=4
    assert v_skel is not None and v_input is not None and rational.eq(v_skel, v_input) and rational.eq(v_input, make(4, 1)), (
        "坍缩探针 z=2：mul(2,2)=4 == λz:z*z(2)=4")


def test_recognize_operand_input_unused_param_no_crash(disc_env):
    """**回归测**（对抗审计 F1·必修 bug）：input λp,q:q*q（p 声明未用·q 用两次）→ mul 识·不崩·探针验。

    bug：_verify_generalization 旧反演 input_probe={1:...}（slot0 缺·p 未用）·range(max+1)=range(2) 访 input_probe[0]
    → KeyError 冒泡崩 formal_train。修：Recognition 增 input_probe_values（连续含未用 slot）·_verify 直接用·消除反演洞。
    mul skeleton arity2 识 λp,q:q*q：operand_binding=(1,1)（两 skeleton slot→input slot1=q）·input_arity=2·
    input_probe_values=((探针0),(探针1)) 连续·execute 绑 make_variable(0=p未用,1=q) 全位·q=探针1=3 → q*q=9 == mul(3,3)=9。
    """
    from pure_integer_ai.training.vm_proof import execute_composes_value
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda p, q: p * q", seg_label="__up_a")
    s2 = _build(disc_env, "lambda m, n: m * n", seg_label="__up_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    held = _build(disc_env, "lambda p, q: q * q", seg_label="__up_h")   # p 未用·仅 q=slot1
    rec = recognize_operators([held], discovered_operators=ops, backend=b, space_id=sid)[0]
    assert rec.is_operand_input and rec.operand_binding == (1, 1), (
        f"两 skeleton slot→input slot1(q)·binding=(1,1)·得 {rec.operand_binding}")
    assert len(rec.input_probe_values) == 2, (
        f"input_arity=2（p+q）·连续探针元组长度 2（含未用 slot p）·得 {rec.input_probe_values}")
    # 探针：q=slot1=探针[1]=3 → q*q=9 == mul 探针 params（两 slot←探针[1]=3）=mul(3,3)=9
    v_skel = execute_composes_value(g, ops[0].skeleton_ref, rec.param_values)
    v_input = execute_composes_value(g, held, rec.input_probe_values)   # 直接用连续探针·无 KeyError
    assert v_skel is not None and v_input is not None and rational.eq(v_skel, v_input) and rational.eq(v_input, make(9, 1)), (
        f"未用首参 p·q=3：mul(3,3)=9 == λp,q:q*q(q=3)=9·无 KeyError 崩（F1 回归）")


def test_recognize_mixed_operand_immediate_input(disc_env):
    """反 theater ⑤混合 input（operand + immediate）兼容：骨架 ADD[PARAM,IMM3] 识 λz:z+3·探针 z=2→5。

    skeleton PARAM_0 位对齐 operand z（operand_binding）·IMM_3 固定位对齐 input IMM_3（值等）。value/operand 两 dict 分位。
    拒 λz:z+5（fixed 3≠5·固定位值等门）。探针 z=2：ADD(2,3)=5 == λz:z+3(2)=5。
    """
    from pure_integer_ai.training.vm_proof import execute_composes_value
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda: 5 + 3", seg_label="__mx_a")
    s2 = _build(disc_env, "lambda: 6 + 3", seg_label="__mx_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    assert ops[0].arity == 1, "5+3/6+3 → ADD[PARAM,IMM3] arity1"
    held_ok = _build(disc_env, "lambda z: z + 3", seg_label="__mx_ok")    # operand z + fixed 3
    rec = recognize_operators([held_ok], discovered_operators=ops, backend=b, space_id=sid)[0]
    assert rec.is_operand_input and rec.operand_binding == (0,), "混合：PARAM 位 operand·binding=(0,)"
    v_skel = execute_composes_value(g, ops[0].skeleton_ref, rec.param_values)        # ADD(2,3)=5
    v_input = execute_composes_value(g, held_ok, (rec.param_values[0],))             # λz:z+3(2)=5
    assert v_skel is not None and v_input is not None and rational.eq(v_skel, v_input) and rational.eq(v_input, make(5, 1)), (
        "混合探针 z=2：ADD(2,3)=5 == λz:z+3(2)=5")
    held_no = _build(disc_env, "lambda z: z + 5", seg_label="__mx_no")    # fixed 3≠5
    assert recognize_operators([held_no], discovered_operators=ops, backend=b, space_id=sid) == [], (
        "混合 input fixed 位值异（3≠5）→ 拒（固定位值等门·operand 输入亦适用）")


def test_recognize_operand_input_renamed_variable(disc_env):
    """operand-input 变量重命名识别：square 骨架识 λw:w*w（任意变量名·同结构等价）·探针验。

    operand-input 识别按结构等价（变量名无关·operand slot 对齐）·非字面 sid 比。λw:w*w 同 square 结构→识·探针 w=2→4。
    """
    from pure_integer_ai.training.vm_proof import execute_composes_value
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda x: x * x", seg_label="__rn_a")
    s2 = _build(disc_env, "lambda y: y * y", seg_label="__rn_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    held = _build(disc_env, "lambda w: w * w", seg_label="__rn_h")   # 不同变量名 w·同结构
    rec = recognize_operators([held], discovered_operators=ops, backend=b, space_id=sid)[0]
    assert rec.is_operand_input and rec.operand_binding == (0,), "变量名无关·结构等价→识"
    v_skel = execute_composes_value(g, ops[0].skeleton_ref, rec.param_values)
    v_input = execute_composes_value(g, held, (rec.param_values[0],))
    assert v_skel is not None and v_input is not None and rational.eq(v_skel, v_input) and rational.eq(v_input, make(4, 1))


def test_formal_train_operand_corpus_generalization(tmp_path):
    """反 theater ⑥formal_train within-run operand 闭环：纯 operand 语料 → 发现 square + 识别 held-out operand + vm_proof 验泛化。

    纯 operand 语料（x*x/y*y/z*z·同 shape）→ 发现身 2（x*x,y*y）·held-out z*z（operand 输入）现被识别 + 探针 vm_proof 验
    → 泛化率 1000（序列2 defer 项收口·operand 语料泛化率从 0 升有意义·within-run operand 闭环）。证 formal_train 主入口真接通
    operand READ 消费（识别 held-out 参数化输入 + 验证）·非 theater。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    b = DictBackend()
    corpus = [_arith_item(s) for s in ("lambda x: x * x", "lambda y: y * y", "lambda z: z * z")]
    config = FormalTrainConfig(run_dir=str(tmp_path), run_id="sq_gen_e2e", rounds_per_stage=1)
    result = formal_train(config, corpus, backend=b)
    assert 1 in {op.arity for op in result.discovered_operators}, "须发现 square arity1"
    assert len(result.recognitions) == 1, "held-out z*z（operand 输入）须被识别"
    assert result.recognitions[0].is_operand_input, "operand 输入识别须标 is_operand_input"
    gen = result.generalization
    assert gen is not None and gen.total_held_out == 1 and gen.recognized == 1, (
        f"held-out 1 须被识别·得 {gen}")
    assert gen.verified == 1, f"operand 输入探针 vm_proof 须验过·得 verified={gen.verified}"
    assert gen.rate_permille == 1000, "全验 → 泛化率 1000"


# ============ 序列7 跨 run READ 闭环（dump/load + load_discovered_operators·§八序列7）===========
#
# within-run 闭环（序列1-6+operand-input）已收口。序列7 闭**跨 run 环**——run N 发现算子 → dump →
# run N+1 resume load → 识别**全新** held-out 命中载入算子 → vm_proof 验泛化。修两基础缺口：
# Gap1 composes_attr 未 dump（formal_train dump_tables 默认含它）·Gap2 _id_pool 未 restore（load_run rebaseline）。
# 反 theater 断言：①跨 run 识别（载入算子被 READ）②composes_attr payload round-trip（read_composes_tree 重建）
# ③id_pool 不撞（续训新分配高于已载）④无 load 不识别（control·证识别因载入非 within-run 假象）。


def test_load_discovered_operators_reconstructs_from_graph(disc_env):
    """序列7 机制③：load_discovered_operators 从已载图重建发现算子列表（纯读·名重派生确定）。

    auto_discover 注册 mul → load_discovered_operators 扫 ATTR_OPERATOR_DEF + ATTR_ARITY +
    ATTR_ORIGIN==DISCOVERED → 重派生 name=_shape_name(sig, arity, abstract_sig)（arith abstract_sig=()
    经 _collect_slot_lcas 重建·B6 Bug 1 修）→ 同骨架/arity/名。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__ld_a")
    s2 = _build(disc_env, "lambda: 6 * 6", seg_label="__ld_b")
    ops = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    assert len(ops) == 1
    loaded = load_discovered_operators(b, space_id=sid)
    assert len(loaded) == 1, f"load 须重建发现算子·得 {len(loaded)}"
    assert loaded[0].skeleton_ref == ops[0].skeleton_ref, "同骨架 struct_ref"
    assert loaded[0].arity == ops[0].arity == 2, "arity 还原"
    assert loaded[0].name == ops[0].name, "名重派生一致（_shape_name 固定种子确定）"


def test_load_discovered_operators_fresh_backend_empty(disc_env):
    """序列7：fresh backend（无 composes_attr）→ load_discovered_operators 返 []（诚实空·不伪造）。"""
    b, sid, _, _, _ = disc_env
    assert load_discovered_operators(b, space_id=sid) == [], "fresh 图无发现算子 → []"


def test_load_discovered_operators_excludes_non_discovered_origin(disc_env):
    """序列7：ATTR_ORIGIN != DISCOVERED（observer BUILT / 手注册）→ 不载入（仅持久化识别发现算子）。

    register_arith_operator 在 build_composes 根上（ORIGIN_BUILT 或无 ORIGIN）→ load 须排除。
    """
    b, sid, es, ci, _ = disc_env
    root = _build(disc_env, "lambda: 3 + 5", seg_label="__ld_hand")   # observer 建造（非 discovered）
    register_arith_operator(b, ci, "__hand_op", root, arity=2)
    loaded = load_discovered_operators(b, space_id=sid)
    assert loaded == [], "手注册（非 discovered）算子须排除·不载入"


def test_composes_attr_round_trips_through_dump_load(tmp_path):
    """序列7 Gap1：dump_run 含 composes_attr → load_run 还原 → read_composes_tree 重建 5 dict（payload 不丢）。

    反 theater ②：composes_attr payload（opcode/operand/immediate）round-trip·read_composes_tree 非空。
    """
    from pure_integer_ai.training.cursor import dump_run, load_run, DUMP_TABLES
    from pure_integer_ai.storage.composes_attr import COMPOSES_ATTR_TABLE
    b1 = DictBackend(); bootstrap(b1); register_composes_attr(b1)
    reg = SpaceRegistry(b1); sp = AbstractSpace.create(reg, "core"); sid = sp.space_id
    es = EdgeStore(b1); ci = ConceptIndex(b1)
    root = ci.ensure("__rt", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith("lambda: 3 + 5", concept_index=ci, edge_store=es, backend=b1,
                              space_id=sid, source=SOURCE_MATH, root_ref=root)
    tables = DUMP_TABLES + (COMPOSES_ATTR_TABLE,)   # formal_train 默认 dump 表集
    dump_run(b1, str(tmp_path), "r", spaces=[sid], tables=tables)
    b2 = DictBackend(); bootstrap(b2); register_composes_attr(b2)
    load_run(b2, str(tmp_path), "r")
    g2 = ConceptGraph(b2)
    children_of, operator_of, _, _, _ = g2.read_composes_tree(root)
    assert children_of, "composes_attr 还原后 read_composes_tree 须重建拓扑（payload 不丢）"
    assert root in operator_of, "算子 opcode 须还原"


def test_load_run_rebases_id_pool_no_collision(tmp_path):
    """序列7 Gap2：load_run 后 next_id 高于已载 max local_id（修 latent 续训 id-collision bug）。

    反 theater ③：续训新分配不撞已载节点（无 rebase → next_id=1 撞·DictBackend 静默 dup corrupt）。
    """
    from pure_integer_ai.training.cursor import dump_run, load_run
    b1 = DictBackend(); bootstrap(b1)
    reg = SpaceRegistry(b1); sp = AbstractSpace.create(reg, "core"); sid = sp.space_id
    ns = NodeStore(b1)
    for lid in range(1, 6):   # 占用 local_id 1..5（已载节点）
        ns.put(sid, lid, node_type=1, tier=TIER_PRIMARY)
    dump_run(b1, str(tmp_path), "r", spaces=[sid])
    b2 = DictBackend(); bootstrap(b2)
    load_run(b2, str(tmp_path), "r")
    nid = b2.next_id(sid)
    assert nid == 6, f"load 后 next_id 须=6（高于已载 max 5·不撞）·得 {nid}（撞已载=latent bug 未修）"


def test_cross_run_resume_recognizes_loaded_operator(tmp_path):
    """序列7 跨 run READ 闭环 e2e：run N 发现 mul → dump → run N+1 --resume load → 识别全新 7*7 + vm_proof 验。

    反 theater ①跨 run READ：载入 mul 被 recognize_operators 真读·识别 run N 未见的 7*7（非发现集→真泛化·非循环）。
    学到能力持久 + 跨 run 可用（闭合"学习累积"环·侧证"能从语料学到能力"）。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    # Run N：发现 mul（5*5/6*6 立即数样本）→ dump（含 composes_attr·config 默认）
    b1 = DictBackend()
    corpus_N = [_arith_item("lambda: 5 * 5"), _arith_item("lambda: 6 * 6")]
    res_N = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="runN",
                                           rounds_per_stage=1), corpus_N, backend=b1)
    assert len(res_N.discovered_operators) == 1, "run N 须发现 mul"
    # Run N+1：fresh backend + --resume load runN → 识别全新 held-out 7*7（非 run N 发现集{5,6}）
    b2 = DictBackend()
    corpus_N1 = [_arith_item("lambda: 7 * 7")]
    res_N1 = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="runN1",
                                            resume=True, base_run_id="runN",
                                            rounds_per_stage=1), corpus_N1, backend=b2)
    assert len(res_N1.recognitions) == 1, f"载入 mul 须识别 7*7·得 {len(res_N1.recognitions)}"
    rec = res_N1.recognitions[0]
    assert rec.param_values == ((7, 1), (7, 1)), f"mul(7,7) 绑定·得 {rec.param_values}"
    gen = res_N1.generalization
    assert gen is not None and gen.verified == 1 and gen.rate_permille == 1000, (
        f"vm_proof 须验 mul(7,7)=49==7*7·得 {gen}")
    assert res_N1.discovered_operators == [], "run N+1 不重发现（载入同形不 re-discover·单样本 <K）"


def test_cross_run_no_recognition_without_load(tmp_path):
    """序列7 control：fresh run（无 resume load）单样本 7*7 → 无识别（无载入算子·<K 不发现不识别）。

    反 theater ④：证 test_cross_run_resume 的识别因 **load 载入**算子·非 within-run 假象。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    b = DictBackend()
    corpus = [_arith_item("lambda: 7 * 7")]   # 单样本·无载入算子
    res = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="ctrl",
                                         rounds_per_stage=1), corpus, backend=b)
    assert res.recognitions == [], "无 load + 单样本 → 无识别（control）"
    assert res.discovered_operators == [], "<K 样本 → 不发现"


def test_load_run_rebases_id_pool_memory_space(tmp_path):
    """序列7 Gap2 Finding2 回归：memory_item 自分配 local_id·load_run rebaseline 须覆盖（不止 concept_node）。

    memory space 载 memory_item local_id 1..5（零 concept_node）→ 逐行跟踪 space_id+local_id →
    next_id=6（修 Finding2·原 concept_node-only 扫描漏 memory space → next_id=1 撞已载 corrupt）。
    """
    from pure_integer_ai.training.cursor import dump_run, load_run
    from pure_integer_ai.storage.spaces.memory_space import MemorySpace
    b1 = DictBackend(); bootstrap(b1)
    reg = SpaceRegistry(b1); mem = MemorySpace.create(reg, "mem_read"); msid = mem.space_id
    for lid in range(1, 6):
        mem.put(lid, content_hash=lid * 7, session_id=1)   # memory_item 自分配位 1..5
    dump_run(b1, str(tmp_path), "r", spaces=[msid])
    b2 = DictBackend(); bootstrap(b2)
    load_run(b2, str(tmp_path), "r")
    nid = b2.next_id(msid)
    assert nid == 6, f"memory space load 后 next_id 须=6（Finding2：逐行跟踪覆盖 memory_item 自分配）·得 {nid}"


# ---- Half B（§八.7②·Finding1 真修）：跨 run arity 进名 + 同形异 arity 发现不抑制 ----
# 反 theater 断言：①square(arity1) 载入后 mul(arity2) 同形异 arity **仍被发现**（非 sig-only 路由吞）
#   ②probe_arity==discover_skeleton.arity 不变量（drift 防线·twin 不漂移）③幂等不重 build（无 orphan 骨架）
#   ④载入同 (sig,arity) 不 re-discover（识别 held-out·守幂等）。

def _disc_skeleton_count(b, sid) -> int:
    """数 ATTR_ORIGIN==DISCOVERED 的骨架 root 数（验幂等不重 build·无 orphan）。"""
    from pure_integer_ai.storage.composes_attr import COMPOSES_ATTR_TABLE
    rows = b.select(COMPOSES_ATTR_TABLE, where={"space_id": sid, "kind": ATTR_ORIGIN})
    return sum(1 for r in rows if r["int_a"] == ORIGIN_DISCOVERED)


def test_shape_name_arity_distinguishes_same_shape_operators():
    """Half B 机制①：同 shape_signature 异 arity → 异名（arity 进 hash·square(1)≠mul(2)）。

    square=λx:x*x（operand arity1）与 mul=λa,b:a*b（operand arity2）shape_signature 皆 (MUL,LEAF,LEAF)
    （叶统一 _LEAF_SIG·不改）·但 _shape_name(sig,arity) arity 进名 → 异名 → 无碰撞。
    """
    def _disc_op(srcs):
        b = DictBackend(); bootstrap(b); register_composes_attr(b)
        reg = SpaceRegistry(b); sp = AbstractSpace.create(reg, "core"); sid = sp.space_id
        es = EdgeStore(b); ci = ConceptIndex(b); g = ConceptGraph(b)
        env = (b, sid, es, ci, g)
        roots = [_build(env, s, seg_label=f"__sn_{i}") for i, s in enumerate(srcs)]
        ops = auto_discover_operators(roots, concept_index=ci, edge_store=es, backend=b,
                                      space_id=sid, source=SOURCE_MATH)
        assert len(ops) == 1, f"{srcs} 须发现 1 算子·得 {len(ops)}"
        sig = tuple(shape_signature(g, ops[0].skeleton_ref))
        return ops[0].name, ops[0].arity, sig

    sq_name, sq_arity, sq_sig = _disc_op(["lambda x: x * x", "lambda y: y * y"])
    mul_name, mul_arity, mul_sig = _disc_op(["lambda p, q: p * q", "lambda m, n: m * n"])
    assert sq_arity == 1 and mul_arity == 2, "square arity1·mul arity2"
    assert sq_sig == mul_sig, f"同 shape_signature（叶统一）·得 {sq_sig} vs {mul_sig}"
    assert sq_name != mul_name, (
        f"同形异 arity 须异名（arity 进 hash）·square={sq_name} mul={mul_name}")


def test_probe_arity_matches_discover_skeleton(disc_env):
    """Half B 机制② drift 防线：probe_arity(samples) == discover_skeleton(samples).arity 全语料族。

    faithful twin 不变量——两函数同 DFS + 同槽规则 + 同一致性门。覆盖 immediate/operand/mixed/heterogeneous/<K。
    任何漂移（probe!=discover）此测失败 → 名(用 probe arity) 与骨架(用 discover arity) 错配 → inline 崩。
    """
    b, sid, es, ci, _ = disc_env
    _counter = [0]

    def _probe_vs_disc(srcs):
        _counter[0] += 1
        roots = [_build(disc_env, s, seg_label=f"__pd_{_counter[0]}_{i}")
                 for i, s in enumerate(srcs)]
        probed = probe_arity(b, roots)
        result = discover_skeleton(roots, concept_index=ci, edge_store=es, backend=b,
                                   space_id=sid, source=SOURCE_MATH,
                                   skeleton_label=f"__pd_skel_{_counter[0]}")
        disc_arity = result.arity if result is not None else None
        assert probed == disc_arity, (
            f"drift！probe={probed} != discover={disc_arity} @ {srcs}（twin 须等）")

    _probe_vs_disc(["lambda: 5 * 5", "lambda: 6 * 6"])          # immediate mul arity2
    _probe_vs_disc(["lambda: 5 + 3", "lambda: 6 + 3"])          # immediate add arity1（次位 fixed）
    _probe_vs_disc(["lambda: 5 + 3", "lambda: 6 + 7"])          # immediate add arity2（两皆异）
    _probe_vs_disc(["lambda x: x * x", "lambda y: y * y"])      # operand square arity1
    _probe_vs_disc(["lambda p, q: p * q", "lambda m, n: m * n"])  # operand mul arity2
    _probe_vs_disc(["lambda x: x * x", "lambda: 5 * 5"])        # mixed operand+immediate 同形 → None
    _probe_vs_disc(["lambda: 5 * 5", "lambda: 3 + 7"])          # heterogeneous MUL vs ADD → None
    _probe_vs_disc(["lambda x: x * x", "lambda p, q: p * q"])   # operand square vs mul 混 → None（拆分冲突）
    assert probe_arity(b, [_build(disc_env, "lambda: 5 * 5", seg_label="__pd_ltK")]) is None  # <K


def test_auto_discover_idempotent_no_orphan_rebuild(disc_env):
    """Half B 机制③：幂等 pre-check（probe→名→lookup）→ 第二次 auto_discover **不重 build**（无 orphan 骨架）。

    守幂等不重 build：原"先 build 再查"会留 orphan 骨架（ATTR_ORIGIN=discovered 但无 name 指）。
    现 probe 先（纯读）→ lookup 命中 → skip discover_skeleton → ATTR_ORIGIN==DISCOVERED 骨架数不增。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build(disc_env, "lambda: 5 * 5", seg_label="__ir_a")
    s2 = _build(disc_env, "lambda: 6 * 6", seg_label="__ir_b")
    roots = [s1, s2]
    ops1 = auto_discover_operators(roots, concept_index=ci, edge_store=es, backend=b,
                                   space_id=sid, source=SOURCE_MATH)
    assert len(ops1) == 1
    count_after_first = _disc_skeleton_count(b, sid)
    assert count_after_first == 1, f"首次发现 1 骨架·得 {count_after_first}"
    ops2 = auto_discover_operators(roots, concept_index=ci, edge_store=es, backend=b,
                                   space_id=sid, source=SOURCE_MATH)
    assert ops2 == [], "第二次幂等 skip（同 (sig,arity) 已注册·不重抽）"
    count_after_second = _disc_skeleton_count(b, sid)
    assert count_after_second == count_after_first == 1, (
        f"幂等不重 build：骨架数不增（无 orphan）·第二次后={count_after_second}·须==1")


def test_cross_run_resume_discovers_different_arity_operator(tmp_path):
    """Half B 核心 e2e（§八.7②·Finding1 真修）：run N square(arity1) 载入后 run N+1 mul(arity2) **仍被发现**。

    无 Half B（sig-only 路由）：mul 同形 → 送识别 → square 拒（两异 operand）→ mul 不发现 = 丢。
    有 Half B（(sig,arity) 路由）：probe mul arity2 → (sig,2) ∉ existing{(sig,1)} → 新算子 → discover mul 异名。
    反 theater：跨 run 学习累积——run N 学 square·run N+1 独立学 mul（同形异 arity 不互抑）+ held-out vm_proof 验。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    # Run N：operand square（λx:x*x/λy:y*y·arity1）→ 发现 + dump
    b1 = DictBackend()
    corpus_N = [_arith_item("lambda x: x * x"), _arith_item("lambda y: y * y")]
    res_N = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="runN",
                                           rounds_per_stage=1), corpus_N, backend=b1)
    assert len(res_N.discovered_operators) == 1, "run N 须发现 square"
    square = res_N.discovered_operators[0]
    assert square.arity == 1, "square operand arity1"
    # Run N+1：fresh backend + --resume load square + mul 语料（同形异 arity·arity2）
    b2 = DictBackend()
    corpus_N1 = [_arith_item("lambda p, q: p * q"),
                 _arith_item("lambda m, n: m * n"),
                 _arith_item("lambda u, v: u * v")]   # 3 mul 样本：发现首 2·held-out 第 3
    res_N1 = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="runN1",
                                            resume=True, base_run_id="runN",
                                            rounds_per_stage=1), corpus_N1, backend=b2)
    assert len(res_N1.discovered_operators) == 1, (
        f"run N+1 须发现 mul（同形异 arity·Half B 不抑制）·得 {len(res_N1.discovered_operators)}")
    mul = res_N1.discovered_operators[0]
    assert mul.arity == 2, "mul operand arity2"
    assert mul.name != square.name, (
        f"square/mul 须异名（arity 进 hash）·square={square.name} mul={mul.name}")
    assert len(res_N1.recognitions) == 1, f"held-out mul 须被识·得 {len(res_N1.recognitions)}"
    gen = res_N1.generalization
    assert gen is not None and gen.verified == 1 and gen.rate_permille == 1000, (
        f"vm_proof 须验 mul(probe)=input(probe)·泛化率 1000·得 {gen}")


def test_cross_run_resume_same_arity_not_rediscovered(tmp_path):
    """Half B 幂等（机制④）：载入同 (sig,arity) 算子 → 不 re-discover（识别 held-out·守幂等）。

    run N square(arity1) 载入 → run N+1 再喂 square 样本（同 arity）→ (sig,1) ∈ existing_keys →
    全送识别（held-out）·**不重发现**（discovered_operators 空）·识别载入 square + vm_proof 验。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    b1 = DictBackend()
    corpus_N = [_arith_item("lambda x: x * x"), _arith_item("lambda y: y * y")]
    res_N = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="runN",
                                           rounds_per_stage=1), corpus_N, backend=b1)
    square_name = res_N.discovered_operators[0].name
    # Run N+1：再喂 square 样本（同 arity1·变量重命名 w*w / u*u·同结构）
    b2 = DictBackend()
    corpus_N1 = [_arith_item("lambda w: w * w"), _arith_item("lambda u: u * u")]
    res_N1 = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="runN1",
                                            resume=True, base_run_id="runN",
                                            rounds_per_stage=1), corpus_N1, backend=b2)
    assert res_N1.discovered_operators == [], (
        "载入同 (sig,arity) → 不 re-discover（守幂等·识别 held-out）")
    assert len(res_N1.recognitions) == 2, f"两 square held-out 须识·得 {len(res_N1.recognitions)}"
    assert all(r.operator_name == square_name for r in res_N1.recognitions), "皆命中载入 square"
    gen = res_N1.generalization
    assert gen is not None and gen.verified == 2 and gen.rate_permille == 1000, (
        f"两 held-out 皆 vm_proof 验·泛化率 1000·得 {gen}")


# ============ B6 Bug 2+3 路由（聚类前置·route_samples_for_discovery·2026-07-06）============
#
# arith 路由 bit-identical 守卫：arith 首样本无 CONCEPT_LEAF → _cluster_by_lca 单簇 slot_lcas=None →
# abstract_sig=() → 路由键 (sig,arity,()) 与原 (sig,arity) 等价（arith abstract_sig 恒 ()）。lang 多簇
# 见 test_stage12。Bug 2 = existing_keys 加 abstract_sig 维；Bug 3 = per-cluster held-out 切分。


def _existing_keys_from_ops_arith(disc_env, ops):
    """镜像 formal_train existing_keys 构造（B6 Bug 2·abstract_sig 维 + §十八 6a-3 cue_sig 第4维）·返 (existing_keys, existing_sigs)。"""
    b, sid, es, ci, g = disc_env
    existing_keys: set = set()
    existing_sigs: set = set()
    for op in ops:
        op_sig = tuple(shape_signature(g, op.skeleton_ref))
        op_asig = _normalize_abstract_sig(_collect_slot_lcas(b, g, op.skeleton_ref))
        op_cue = _normalize_abstract_sig(_collect_cue_sig(b, g, op.skeleton_ref))   # §十八 6a-3：cue_sig 第4维（gate OFF 全 ()→bit-identical）
        existing_keys.add((op_sig, op.arity, op_asig, op_cue))
        existing_sigs.add(op_sig)
    return existing_keys, existing_sigs


def test_route_arith_existing_empty_discovers_k(disc_env):
    """arith 路由 bit-identical：existing 空 → 2 样本（≥K）全 discover·recognize 空（K=2 恰满·无 held-out）。

    arith 首 sample 无 CONCEPT_LEAF → 单簇 None → abstract_sig=() → 行为同原 (sig,arity) 路由。
    """
    b, sid, es, ci, g = disc_env
    s1 = _build(disc_env, "lambda: 3 + 5", seg_label="__route_a1")
    s2 = _build(disc_env, "lambda: 4 + 7", seg_label="__route_a2")
    discover_roots, recognize_roots = route_samples_for_discovery(
        b, g, [s1, s2], existing_keys=set(), existing_sigs=set(), space_id=sid)
    assert set(discover_roots) == {s1, s2}, f"existing 空·两样本 ≥K → 全 discover·得 {discover_roots}"
    assert recognize_roots == [], f"K=2 恰满无 held-out → recognize 空·得 {recognize_roots}"


def test_route_arith_loaded_abstract_sig_empty_matches_recognize(disc_env):
    """arith 路由 bit-identical + Bug 2 守：载入 add 算子（abstract_sig=()·无 CONCEPT_LEAF）→
    新同 (sig,arity) add 样本簇 abstract_sig=() ∈ existing_keys → 全 recognize（不 re-discover·守幂等）。

    路由键 (sig,2,()) 与原 (sig,2) 等价·arith abstract_sig 恒 ()·bit-identical。
    """
    b, sid, es, ci, g = disc_env
    # 载入：先发现 add 算子（arity=2·两相异立即数位）
    disc_s1 = _build(disc_env, "lambda: 3 + 5", seg_label="__route_load1")
    disc_s2 = _build(disc_env, "lambda: 4 + 7", seg_label="__route_load2")
    loaded = auto_discover_operators(
        [disc_s1, disc_s2], concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_MATH)
    assert len(loaded) == 1, f"前置：发现 1 add 算子·得 {len(loaded)}"
    add_op = loaded[0]
    add_asig = _normalize_abstract_sig(_collect_slot_lcas(b, g, add_op.skeleton_ref))
    assert add_asig == (), f"arith 算子 abstract_sig 恒 ()（无 CONCEPT_LEAF）·得 {add_asig}"
    existing_keys, existing_sigs = _existing_keys_from_ops_arith(disc_env, loaded)
    # 新同 (sig,arity) add 样本（变量重命名·同结构）
    s3 = _build(disc_env, "lambda: 6 + 9", seg_label="__route_new1")
    s4 = _build(disc_env, "lambda: 1 + 2", seg_label="__route_new2")
    discover_roots, recognize_roots = route_samples_for_discovery(
        b, g, [s3, s4], existing_keys=existing_keys, existing_sigs=existing_sigs, space_id=sid)
    assert discover_roots == [], f"载入 add abstract_sig=() 已在 existing → 不 re-discover·得 {discover_roots}"
    assert set(recognize_roots) == {s3, s4}, f"两 held-out 全识别·得 {recognize_roots}"


# ============ §8.7-洗 洗净循环反馈半闭环（op_confidence 台账 + recognize 择优·2026-07-03）============
#
# 反 theater 半环：_verify_generalization 验结果写算子置信度（op_confidence sn/tn）→ recognize_operators
# 择优读（滤 tested-never-verified=洗净）·解 recognitions terminal。生成侧全环（generate.py 读置信度）defer。
#
# 诚实边界（关键·影响 e2e 设计）：正确算术算子 **aligns⟹verifies**（构造性必然——树结构定执行·sid 标签仅
# 重标槽·swap PARAM 等价非坏算子）·故 sn==0 滤除在生产算术不可达。sn==0 真正守的是 **vm_proof-None 失败
# 模式**（编译发散/StepLimit）·本测模拟之（record_op_outcome verified=False）验机制活·非构造真坏算子（不可构造）。


def test_op_confidence_record_read_monotone(disc_env):
    """§8.7-洗 存储：record_op_outcome 累积 sn/tn/strength·MUTABLE_MONOTONE（sn/strength 单调·tn 失败计数）·read 回读。"""
    b, sid, es, ci, _ = disc_env
    register_op_confidence(b)
    ref = (sid, 12345)
    assert read_op_confidence(b, ref) is None, "冷启动无行→None"
    # verified → insert (sn=1, tn=1, strength=DEFAULT+1=2)
    record_op_outcome(b, ref=ref, verified=True)
    assert read_op_confidence(b, ref) == (1, 1, 2), f"首次 verified·得 {read_op_confidence(b, ref)}"
    # fail → tn++ only（sn 不降·守单调）
    record_op_outcome(b, ref=ref, verified=False)
    assert read_op_confidence(b, ref) == (1, 2, 2), f"fail tn++·sn 不降·得 {read_op_confidence(b, ref)}"
    # verified again → sn=2 tn=3 strength=3
    record_op_outcome(b, ref=ref, verified=True)
    assert read_op_confidence(b, ref) == (2, 3, 3), f"再 verified·得 {read_op_confidence(b, ref)}"
    # 未注册 ref → None（冷启动）
    assert read_op_confidence(b, (sid, 99999)) is None, "无行→None"


def test_op_confidence_dump_load_roundtrip(tmp_path):
    """§8.7-洗 跨 run dump/load bit-identical（"学习累积"环·同序列7 composes_attr 模式）。"""
    from pure_integer_ai.training.cursor import dump_run, load_run, DUMP_TABLES
    b1 = DictBackend(); bootstrap(b1); register_op_confidence(b1)
    reg = SpaceRegistry(b1); sp = AbstractSpace.create(reg, "core"); sid = sp.space_id
    ref = (sid, 100)
    record_op_outcome(b1, ref=ref, verified=True)
    record_op_outcome(b1, ref=ref, verified=True)
    record_op_outcome(b1, ref=ref, verified=False)
    before = read_op_confidence(b1, ref)
    assert before == (2, 3, 3), f"sn=2 tn=3 strength=3·得 {before}"
    dump_run(b1, str(tmp_path), "runN", spaces=[sid],
             tables=DUMP_TABLES + (OP_CONFIDENCE_TABLE,))
    b2 = DictBackend(); bootstrap(b2); register_op_confidence(b2)
    load_run(b2, str(tmp_path), "runN")
    after = read_op_confidence(b2, ref)
    assert after == before, f"跨 run round-trip bit-identical·得 {after} ≠ {before}"


def test_recognize_selects_by_confidence_washes_out_failed(disc_env):
    """§8.7-洗 + 刀2 件6 多解析：返全列按 rate 降序·tested-never-verified(sn==0)滤除·冷启动全返（BFS 序）。

    mul(arity2)+square(arity1) 同 shape (MUL,LEAF,LEAF)（叶统一 _LEAF_SIG·Half B 异名但同 sig）→ 同候选池·
    两都 align `7*7`（mul(7,7)=square(7)=49）。**刀2 多解析**：冷启动两都 align → 返全列 [sq, mul]（BFS 序·
    候选序 square 先·stable sort 同率保 BFS）·非旧 aligning[0] 单选。洗净（sn==0 滤）后 square 滤除 → 仅 mul。
    **诚实边界**：正确算术算子 aligns⟹verifies（构造性必然）·sn==0 滤除在生产算术不可达·本测**模拟**
    vm_proof-None 失败模式（record_op_outcome verified=False）验机制活·非构造真坏算子（不可构造）。
    反 theater 锚点：冷启动返全列 [sq,mul]（多解析）·写置信度后 square 滤除→仅 mul（洗净行为真变）。
    """
    b, sid, es, ci, _ = disc_env
    register_op_confidence(b)
    mul_roots = [_build(disc_env, "lambda: 5 * 5", seg_label="__wc_ma"),
                 _build(disc_env, "lambda: 6 * 6", seg_label="__wc_mb")]
    sq_roots = [_build(disc_env, "lambda x: x * x", seg_label="__wc_sa"),
                _build(disc_env, "lambda y: y * y", seg_label="__wc_sb")]
    mul_ops = auto_discover_operators(mul_roots, concept_index=ci, edge_store=es, backend=b,
                                      space_id=sid, source=SOURCE_MATH)
    sq_ops = auto_discover_operators(sq_roots, concept_index=ci, edge_store=es, backend=b,
                                     space_id=sid, source=SOURCE_MATH)
    assert len(mul_ops) == 1 and len(sq_ops) == 1, "mul arity2 + square arity1 各发现 1"
    mul_op, sq_op = mul_ops[0], sq_ops[0]
    held = _build(disc_env, "lambda: 7 * 7", seg_label="__wc_held")
    all_ops = [sq_op, mul_op]   # 候选序 square 先
    # 刀2 冷启动（无置信度）→ 返全列 [sq, mul]（两都 align 7*7·stable sort 同率保 BFS 序·候选序 square 先）
    recs_cold = recognize_operators([held], discovered_operators=all_ops, backend=b, space_id=sid)
    assert len(recs_cold) == 2, f"刀2 多解析：square+mul 两都 align 7*7 → 返全列 2·得 {recs_cold}"
    assert recs_cold[0].operator_name == sq_op.name, f"BFS 序 square 在前·得 {recs_cold}"
    assert recs_cold[1].operator_name == mul_op.name, f"mul 在后·得 {recs_cold}"
    # 模拟 vm_proof-None 失败：square tested-never-verified(sn=0)·mul verified(sn>0)
    record_op_outcome(b, ref=sq_op.name_ref, verified=False)   # square (0,1,1) → 滤
    record_op_outcome(b, ref=mul_op.name_ref, verified=True)   # mul (1,1,2) rate=1000
    # 洗净：square sn==0 滤除 → 仅 mul（返多路不含坏算子·即便 square 候选序首）
    recs = recognize_operators([held], discovered_operators=all_ops, backend=b, space_id=sid)
    assert len(recs) == 1 and recs[0].operator_name == mul_op.name, (
        f"square tested-never-verified 滤除→仅 mul（洗净行为真变·反 theater）·得 {recs}")


def test_formal_train_writes_operator_confidence(tmp_path):
    """§8.7-洗 生产接线：formal_train _verify_generalization 把 vm_proof 验结果写算子置信度（解 recognitions terminal）。

    语料 5*5/6*6/7*7（3 同形≥K+1）→ 发现首 2（mul）·held-out 7*7 识别+vm_proof 验 → mul.name_ref 置信度
    sn=1 tn=1（verified）。证识别产物去 terminal（confidence 被写非死写）·反 theater 半环。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    b = DictBackend()
    corpus = [_arith_item("lambda: 5 * 5"), _arith_item("lambda: 6 * 6"),
              _arith_item("lambda: 7 * 7")]
    res = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="runC",
                                         rounds_per_stage=1), corpus, backend=b)
    assert len(res.discovered_operators) == 1, "mul 发现"
    op = res.discovered_operators[0]
    conf = read_op_confidence(b, op.name_ref)
    assert conf is not None, "置信度须被写（识别产物去 terminal）"
    sn, tn, _st = conf
    assert sn == 1 and tn == 1, f"mul held-out 7*7 验 1 次·sn=1 tn=1·得 ({sn},{tn})"
    assert res.generalization is not None and res.generalization.verified == 1, "held-out 验 1"


def test_cross_run_op_confidence_persistence(tmp_path):
    """§8.7-洗 跨 run 置信度持久：run N 写置信度→dump→run N+1 resume load（空语料·无新写）→置信度纯 round-trip。

    "学习累积"环：run N 验结果不丢·run N+1 可读。证 op_confidence 经 dump_tables round-trip（formal_train
    生产路径·非仅存储单测）。空语料 run N+1 确保无新写·纯验持久化。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    corpus = [_arith_item("lambda: 5 * 5"), _arith_item("lambda: 6 * 6"),
              _arith_item("lambda: 7 * 7")]
    b1 = DictBackend()
    res_N = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="runD",
                                           rounds_per_stage=1), corpus, backend=b1)
    op = res_N.discovered_operators[0]
    conf_N = read_op_confidence(b1, op.name_ref)
    assert conf_N is not None and conf_N[0] >= 1, "run N 须写置信度"
    # run N+1 resume + 空语料（无新识别/验·纯 round-trip 验持久化）
    b2 = DictBackend()
    formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="runD1",
                                   resume=True, base_run_id="runD",
                                   rounds_per_stage=1), [], backend=b2)
    conf_N1 = read_op_confidence(b2, op.name_ref)
    assert conf_N1 == conf_N, f"跨 run 置信度纯 round-trip bit-identical·得 {conf_N1} ≠ {conf_N}"


# ============ §8.7-全 生成侧全环·task-driven L8 episode（外真半·补半环缺·墙内现可达）===========
#
# 半环（done·§8.7-洗）测自洽：skeleton(recognized_params)==input()（学生==学生·传递必然对正确算子）。
# task-driven 测外真泛化：skeleton(新 task args)==expected（学生==教师独立源·非传递·须骨架是正确抽象）。
# 6 步：任务(input_args,expected) → 选算子(arity+置信度择优) → 执行骨架 → 外真验 vs expected → 写 op_confidence
#       → 打包 OutputResult → metrics generate_verified。反 theater 4 锚点 + 双计防护 + 单候选边界。


def _task_item(specs):
    """造任务 CollectedItem（arith_specs·无 arith_source·纯任务规格·R6 独立源·非学生编译）。"""
    from pure_integer_ai.experiments.collection import CollectedItem
    from pure_integer_ai.cognition.shared.types import CodeSpec
    return CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                         source=SOURCE_MATH, arith_specs=tuple(specs))


def _gen_ctx():
    """make_train_context（注册 composes_attr + op_confidence·task-driven 需两者）+ disc_env 元组 + ctx。"""
    from pure_integer_ai.experiments.formal_train import make_train_context
    b = DictBackend()
    ctx = make_train_context(b)
    env = (b, ctx.space_id, ctx.edge_store, ctx.concept_index, ctx.concept_graph)
    return ctx, b, env


def test_task_driven_single_candidate_constructive():
    """§8.7-全 单候选构造性边界（诚实边界①）：mul 发现(5*5/6*6 立即数) + task(7,8)→56 → 选 mul → 56 → 外真验过。

    单候选时 skeleton(args)==expected 构造性必过（信号薄·同半环）·证 6 步机制活：选算子+执行+外真验+
    写置信度+打包 OutputResult。mul.name_ref 置信度 sn=1 tn=1（task-driven 写）。episode.output.parts 非空。
    """
    from pure_integer_ai.experiments.formal_train import _run_task_driven_generate
    from pure_integer_ai.cognition.shared.types import CodeSpec
    ctx, b, env = _gen_ctx()
    sid, es, ci = ctx.space_id, ctx.edge_store, ctx.concept_index
    ops = auto_discover_operators(
        [_build(env, "lambda: 5 * 5", seg_label="__tg_m1"),
         _build(env, "lambda: 6 * 6", seg_label="__tg_m2")],
        concept_index=ci, edge_store=es, backend=b, space_id=sid, source=SOURCE_MATH)
    assert len(ops) == 1 and ops[0].arity == 2
    mul_op = ops[0]
    task = _task_item([CodeSpec((7, 8), (56, 1))])
    eps, summary = _run_task_driven_generate(ctx, [task], ops)
    assert summary.total_tasks == 1 and summary.selected == 1 and summary.verified == 1, (
        f"单候选 mul(7,8)=56==56 外真验过·得 {summary}")
    # 写置信度（task-driven 写·与半环同表）
    conf = read_op_confidence(b, mul_op.name_ref)
    assert conf is not None and conf[0] == 1 and conf[1] == 1, f"mul sn=1 tn=1·得 {conf}"
    # 打包 OutputResult 非空（反 theater ③下游读者锚）
    assert len(eps) == 1
    out = eps[0].output
    assert out.parts and out.parts[0].words == ["56/1"], f"产出值 56/1·得 {out.parts[0].words}"
    assert eps[0].reward == 1


def test_task_driven_multi_candidate_wash():
    """§8.7-全 多候选洗净环（反 theater ①行为真变 + ④拒坏选好）：add+mul 发现 + 4 个 mul-fitting task →

    冷启 add 首选失败(add sn=0)→mul 择优胜出(add 滤·mul 验)·generate_verified=3/4。add(7,8)=15≠56 失败·
    mul 验过。add tested-never-verified(sn==0)滤除·mul 择优。all_ops=[add,mul]（add 语料序先）·冷启 tie→
    BFS 序 add 首选→失败→mul 胜出=自然洗净环（读置信度有行为效应·非首匹配兜底）。
    """
    from pure_integer_ai.experiments.formal_train import _run_task_driven_generate
    from pure_integer_ai.cognition.shared.types import CodeSpec
    ctx, b, env = _gen_ctx()
    sid, es, ci = ctx.space_id, ctx.edge_store, ctx.concept_index
    # add 语料先（→ discovered=[add,mul]·all_ops=[add,mul]·冷启 tie add 首选）
    add_ops = auto_discover_operators(
        [_build(env, "lambda: 3 + 5", seg_label="__tw_a1"),
         _build(env, "lambda: 4 + 7", seg_label="__tw_a2")],
        concept_index=ci, edge_store=es, backend=b, space_id=sid, source=SOURCE_MATH)
    mul_ops = auto_discover_operators(
        [_build(env, "lambda: 5 * 5", seg_label="__tw_m1"),
         _build(env, "lambda: 6 * 6", seg_label="__tw_m2")],
        concept_index=ci, edge_store=es, backend=b, space_id=sid, source=SOURCE_MATH)
    assert len(add_ops) == 1 and len(mul_ops) == 1
    add_op, mul_op = add_ops[0], mul_ops[0]
    all_ops = [add_op, mul_op]   # add 先（冷启 tie→BFS add 首选）
    # 4 个 mul-fitting task（add 不 fit·mul fit）·distinct input_args
    specs = [CodeSpec((7, 8), (56, 1)), CodeSpec((3, 4), (12, 1)),
             CodeSpec((5, 6), (30, 1)), CodeSpec((2, 9), (18, 1))]
    eps, summary = _run_task_driven_generate(ctx, [_task_item(specs)], all_ops)
    assert summary.total_tasks == 4 and summary.selected == 4, f"4 task 全选到·得 {summary}"
    assert summary.verified == 3, f"task1 add 失败·task2-4 mul 验过→verified=3·得 {summary}"
    # 拒坏选好：add tested-never-verified(sn=0)·mul verified(sn=3)
    add_conf = read_op_confidence(b, add_op.name_ref)
    mul_conf = read_op_confidence(b, mul_op.name_ref)
    assert add_conf is not None and add_conf[0] == 0 and add_conf[1] == 1, (
        f"add task1 失败 sn=0 tn=1·得 {add_conf}")
    assert mul_conf is not None and mul_conf[0] == 3 and mul_conf[1] == 3, (
        f"mul task2-4 验过 sn=3 tn=3·得 {mul_conf}")
    # 行为真变：task1 选 add(产 15·不验→parts=[]·reward 0)·task2 选 mul(产 12·验→提交·reward 1)
    assert eps[0].reward == 0 and eps[0].output.parts == [], (
        f"task1 冷启选 add→15 不验→不提交产出（parts=[]·审计 F1）·得 {eps[0].output.parts}")
    assert eps[1].reward == 1 and eps[1].output.parts[0].words == ["12/1"], (
        f"task2 选 mul→12 验过→提交产出·得 {eps[1].output.parts[0].words}")


def test_task_driven_behavior_changes_with_confidence():
    """§8.7-全 反 theater ①行为真变（crisp）：读置信度择优有行为效应——mul 冷启首选产 56·置 mul sn=0→选 add 产 15。

    两独立 backend（sn 单调不可降·须 fresh 各设）：Run A fresh mul 冷启→选 mul→56 验过。Run B fresh 后
    手注 mul sn=0（record_op_outcome False）→mul 滤→选 add→15 不验。产出真变=读置信度有行为效应（非装饰读取）。
    """
    from pure_integer_ai.experiments.formal_train import _run_task_driven_generate
    from pure_integer_ai.cognition.shared.types import CodeSpec
    task = _task_item([CodeSpec((7, 8), (56, 1))])

    def setup():
        ctx, b, env = _gen_ctx()
        sid, es, ci = ctx.space_id, ctx.edge_store, ctx.concept_index
        mul_ops = auto_discover_operators(
            [_build(env, "lambda: 5 * 5", seg_label="__bc_m1"),
             _build(env, "lambda: 6 * 6", seg_label="__bc_m2")],
            concept_index=ci, edge_store=es, backend=b, space_id=sid, source=SOURCE_MATH)
        add_ops = auto_discover_operators(
            [_build(env, "lambda: 3 + 5", seg_label="__bc_a1"),
             _build(env, "lambda: 4 + 7", seg_label="__bc_a2")],
            concept_index=ci, edge_store=es, backend=b, space_id=sid, source=SOURCE_MATH)
        return ctx, b, [mul_ops[0], add_ops[0]]   # [mul, add]（冷启 tie mul 首选）

    # Run A：fresh·mul 冷启→选 mul→56 验→提交
    ctxA, _bA, opsA = setup()
    epsA, sumA = _run_task_driven_generate(ctxA, [task], opsA)
    assert sumA.verified == 1 and epsA[0].output.parts[0].words == ["56/1"], (
        f"Run A mul 冷启首选→56 验过→提交·得 {sumA}/{epsA[0].output.parts}")
    # Run B：fresh 后手注 mul sn=0→mul 滤→选 add→15 不验→不提交（parts=[]）
    ctxB, bB, opsB = setup()
    record_op_outcome(bB, ref=opsB[0].name_ref, verified=False)   # mul sn=0 tn=1→滤
    epsB, sumB = _run_task_driven_generate(ctxB, [task], opsB)
    assert sumB.verified == 0 and epsB[0].output.parts == [], (
        f"Run B mul sn=0 滤→选 add→15 不验→不提交（parts=[]·审计 F1）·得 {sumB}/{epsB[0].output.parts}")
    # 行为真变：Run A 提交 56/1·Run B 不提交（parts=[]）·读置信度择优有行为效应（非装饰读取）
    assert epsA[0].output.parts != epsB[0].output.parts, "产出须真变（提交 vs 不提交）"


def test_task_driven_external_truth_new_args():
    """§8.7-全 反 theater ②新消费（审计降级·Mode A 构造性·非"外真非传递"）：skeleton(新 task args) vs expected·
    新 args 非学习输入→产新值非记忆复现。

    mul 从 5*5/6*6 学（立即数 5,6）·task(7,8)→56·skeleton(7,8)=56==56 验过（Mode A 构造性必然·expected=正确答案）。
    产出 56 ≠ 学习输入值 25(5*5)/36(6*6)→新 args 产新值·非记忆复现。**审计 F2**：skeleton 派生自程序·
    故 skeleton(args)==expected 构造性必然（传递经 skeleton 起源）·**非"非传递外真"**·真非传递留 Mode B（异算法）。
    task 无输入程序（生成姿态 call 算子为函数·非半环识别姿态 align 程序）。
    """
    from pure_integer_ai.experiments.formal_train import _run_task_driven_generate
    from pure_integer_ai.cognition.shared.types import CodeSpec
    ctx, _b, env = _gen_ctx()
    sid, es, ci = ctx.space_id, ctx.edge_store, ctx.concept_index
    ops = auto_discover_operators(
        [_build(env, "lambda: 5 * 5", seg_label="__et_m1"),
         _build(env, "lambda: 6 * 6", seg_label="__et_m2")],
        concept_index=ci, edge_store=es, backend=ctx.backend, space_id=sid, source=SOURCE_MATH)
    task = _task_item([CodeSpec((7, 8), (56, 1))])
    eps, summary = _run_task_driven_generate(ctx, [task], ops)
    assert summary.verified == 1
    val_str = eps[0].output.parts[0].words[0]
    assert val_str == "56/1", f"skeleton(7,8)=56（Mode A 构造性）·得 {val_str}"
    # 新 args 产新值·非学习输入记忆（5*5=25·6*6=36·task=56·三者相异）= 新 args 泛化探针真增量
    assert val_str not in ("25/1", "36/1"), "产出须是新 args 的新值·非学习输入记忆复现"
    # task 无输入程序（生成姿态）·非半环识别姿态（align 输入程序）·但 skeleton 起源仍是程序（Mode A 构造性）
    assert task.arith_source is None, "task 输入无程序（生成姿态 call 算子为函数）"


def test_task_driven_metrics_downstream_reader(tmp_path):
    """§8.7-全 反 theater ③下游读者：formal_train e2e → metrics jsonl 有 generate 行·generate_verified 真读 episode。

    OutputResult.parts 非空 → metrics generate_verified 计数真读（非死写）。stage=0 行独有 generate 字段非 0。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    from pure_integer_ai.cognition.shared.types import CodeSpec
    import json
    import os
    b = DictBackend()
    corpus = [_arith_item("lambda: 5 * 5"), _arith_item("lambda: 6 * 6"),
              _task_item([CodeSpec((7, 8), (56, 1))])]
    res = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="genM",
                                         rounds_per_stage=1), corpus, backend=b)
    assert res.generate is not None and res.generate.verified == 1, (
        f"generate.verified==1·得 {res.generate}")
    # metrics jsonl 有 generate 行（stage=0·generate_verified>0）
    mpath = os.path.join(str(tmp_path), "genM", "metrics.jsonl")
    with open(mpath, encoding="utf-8") as fh:
        lines = [json.loads(ln) for ln in fh if ln.strip()]
    gen_lines = [ln for ln in lines if ln.get("generate_verified", 0) > 0]
    assert len(gen_lines) == 1, f"须有 1 行 generate_verified>0·得 {len(gen_lines)} 行"
    assert gen_lines[0]["generate_verified"] == 1 and gen_lines[0]["generate_total"] == 1
    assert gen_lines[0]["stage"] == 0, "generate 行 stage=0（disambiguate）"


def test_task_driven_no_double_count_complementary(tmp_path):
    """§8.7-全 诚实边界④ 防双计：task 输入(spec.input_args)≠recognize held-out(程序)·两路置信度互补非重复计。

    语料 5*5/6*6/7*7（3 同形≥K+1）→ mul 发现首 2 + held-out 7*7 识别+自洽验（半环·sn+1）。
    task(8,9)→72 → mul skeleton(8,9)=72==72 外真验（本环·sn+1）。mul sn=2（1 自洽+1 外真·互补非双计）。
    task 输入(8,9)≠held-out(7*7 程序)·两姿态不同输入非同试验。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    from pure_integer_ai.cognition.shared.types import CodeSpec
    b = DictBackend()
    corpus = [_arith_item("lambda: 5 * 5"), _arith_item("lambda: 6 * 6"),
              _arith_item("lambda: 7 * 7"),
              _task_item([CodeSpec((8, 9), (72, 1))])]
    res = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="genDC",
                                         rounds_per_stage=1), corpus, backend=b)
    assert len(res.discovered_operators) == 1, "mul 发现"
    mul_op = res.discovered_operators[0]
    # 半环自洽验（7*7 held-out·generalization）+ 本环外真验（task 8*9·generate）各 1
    assert res.generalization is not None and res.generalization.verified == 1, (
        f"held-out 7*7 自洽验 1·得 {res.generalization}")
    assert res.generate is not None and res.generate.verified == 1, (
        f"task 8*9 外真验 1·得 {res.generate}")
    # mul 置信度 sn=2（1 自洽 + 1 外真·互补非双计·同表累积）
    conf = read_op_confidence(b, mul_op.name_ref)
    assert conf is not None and conf[0] == 2 and conf[1] == 2, (
        f"mul sn=2 tn=2（自洽+外真互补）·得 {conf}")
    # task 输入(8,9)≠recognize held-out(7*7 程序)·两路不同输入非同试验
    assert res.generate.total_tasks == 1, "task 输入是 spec.input_args(8,9) 非 held-out 程序"


def test_formal_train_wires_generate(tmp_path):
    """§8.7-全 生产接线：formal_train 主入口触发 task-driven generate·result.generate 非空（反 theater 真 caller）。

    算术语料 + task → result.generate.verified 非零·mul.name_ref 置信度被 task-driven 写（非死写）。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    from pure_integer_ai.cognition.shared.types import CodeSpec
    b = DictBackend()
    corpus = [_arith_item("lambda: 5 * 5"), _arith_item("lambda: 6 * 6"),
              _task_item([CodeSpec((7, 8), (56, 1))])]
    res = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="genW",
                                         rounds_per_stage=1), corpus, backend=b)
    assert res.generate is not None, "formal_train 须触发 task-driven generate"
    assert res.generate.total_tasks == 1 and res.generate.verified == 1, (
        f"task(7,8)→56 mul 验过·得 {res.generate}")
    op = res.discovered_operators[0]
    conf = read_op_confidence(b, op.name_ref)
    assert conf is not None and conf[0] >= 1, "task-driven 须写置信度（非死写）"


def test_task_driven_bit_identical(tmp_path):
    """§8.7-全 确定性：同语料两次 formal_train → result.generate bit-identical + 置信度同（rate 排序稳定 tiebreak BFS）。"""
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    from pure_integer_ai.cognition.shared.types import CodeSpec

    def run(rid):
        b = DictBackend()
        corpus = [_arith_item("lambda: 5 * 5"), _arith_item("lambda: 6 * 6"),
                  _arith_item("lambda: 3 + 5"), _arith_item("lambda: 4 + 7"),
                  _task_item([CodeSpec((7, 8), (56, 1)), CodeSpec((2, 3), (6, 1))])]
        res = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id=rid,
                                             rounds_per_stage=1), corpus, backend=b)
        confs = tuple(sorted(
            (op.name, read_op_confidence(b, op.name_ref))
            for op in res.discovered_operators))
        return (res.generate.total_tasks, res.generate.selected, res.generate.verified,
                confs)

    r1 = run("bi1")
    r2 = run("bi2")
    assert r1 == r2, f"两次跑须 bit-identical·{r1} ≠ {r2}"


def test_task_driven_metrics_reads_parts_not_reward(tmp_path):
    """§8.7-全 反 theater ③ regression guard（审计 F1 必修）：metrics generate_verified 须读 e.output.parts·非 e.reward。

    构造 reward=1 但 parts=[] 的 episode（"验信号与产出脱钩"反例）→ generate_verified 须=0（读 parts）。
    若 metrics 读 reward 会=1（落 §8.7-洗-证伪 candidate (Z) theater·parts 死写无消费者）·本测守 metrics 真读 parts。
    对照：parts 非空 → generate_verified=1。conduction_rate 用 reward_pos（与 parts 解耦·冗余双保险）。
    """
    import os
    from pure_integer_ai.experiments.metrics import MetricsCollector
    from pure_integer_ai.cognition.shared.types import Episode, OutputResult, OutputPart
    mc = MetricsCollector(os.path.join(str(tmp_path), "reg", "metrics.jsonl"))
    # reward=1 但 parts=[] → 须不计 generate_verified（metrics 读 parts 非 reward）
    ep = Episode(episode_id=0, run_id=0, reward=1, output=OutputResult(parts=[]))
    m = mc.record_generate_round(0, [ep])
    mc.close()
    assert m.generate_verified == 0, (
        "reward=1 但 parts=[] → generate_verified 须=0（读 parts·F1 regression guard·防 (Z) theater）")
    assert m.reward_pos == 1, "conduction_rate 用 reward_pos=1（与 parts 解耦·冗余双保险）"
    # 对照：parts 非空 + reward=1 → generate_verified=1
    mc2 = MetricsCollector(os.path.join(str(tmp_path), "reg2", "metrics.jsonl"))
    ep2 = Episode(episode_id=0, run_id=0, reward=1,
                  output=OutputResult(parts=[OutputPart(unit=(0, 0), words=["56/1"])]))
    m2 = mc2.record_generate_round(0, [ep2])
    mc2.close()
    assert m2.generate_verified == 1, "parts 非空 → generate_verified=1（metrics 真读 parts）"


# ============ Task #475 ConceptIndex._index 重建（跨 run identity·载入算子可 inline·§8.7-idx）===========
#
# _index in-memory run-scoped·load_run 不重建 → 载入算子不可 inline + observe 续训后建重复概念点（latent corrupt）。
# 修：concept_identity 扩展表持久化 (space_id,local_id,content_hash)·ConceptIndex lazy per-space 重建。


def test_concept_identity_record_load_roundtrip(tmp_path):
    """Task #475 存储：record_concept_identity 写（幂等）→ dump → load → load_space_identity 还原 hash→local_id。"""
    from pure_integer_ai.storage.concept_identity import (
        register_concept_identity, record_concept_identity, load_space_identity,
        CONCEPT_IDENTITY_TABLE)
    from pure_integer_ai.cognition.shared.concept_index import content_hash
    from pure_integer_ai.training.cursor import dump_run, load_run, DUMP_TABLES
    b1 = DictBackend(); bootstrap(b1); register_concept_identity(b1)
    reg = SpaceRegistry(b1); sp = AbstractSpace.create(reg, "core"); sid = sp.space_id
    ch = content_hash("__roundtrip_test__")
    record_concept_identity(b1, space_id=sid, local_id=42, content_hash=ch)
    assert load_space_identity(b1, sid) == {ch: 42}, "写后读"
    record_concept_identity(b1, space_id=sid, local_id=42, content_hash=ch)   # 幂等：同 (space,local_id) skip
    dump_run(b1, str(tmp_path), "rRT", spaces=[sid],
             tables=DUMP_TABLES + (CONCEPT_IDENTITY_TABLE,))
    b2 = DictBackend(); bootstrap(b2); register_concept_identity(b2)
    load_run(b2, str(tmp_path), "rRT")
    assert load_space_identity(b2, sid) == {ch: 42}, "跨 run roundtrip bit-identical"


def test_concept_index_lookup_finds_loaded_concept(tmp_path):
    """Task #475 core fix：run N ensure 写 concept_identity → dump → fresh ConceptIndex lazy 重建 →
    lookup 命中载入概念点（非 None）。无 fix 则 _index 空不重建→lookup 返 None。"""
    from pure_integer_ai.experiments.formal_train import make_train_context
    from pure_integer_ai.training.cursor import dump_run, load_run, DUMP_TABLES
    from pure_integer_ai.storage.concept_identity import CONCEPT_IDENTITY_TABLE
    b1 = DictBackend(); ctx1 = make_train_context(b1)
    sid1 = ctx1.space_id
    ref = ctx1.concept_index.ensure("__loaded_op", space_id=sid1, tier=TIER_PRIMARY,
                                    node_type=NODE_CONCEPT)
    dump_run(b1, str(tmp_path), "rA", spaces=[sid1],
             tables=DUMP_TABLES + (CONCEPT_IDENTITY_TABLE,))
    # fresh backend + load + fresh ConceptIndex（_index 空·lazy 重建在 lookup 时触发）
    b2 = DictBackend(); ctx2 = make_train_context(b2)
    load_run(b2, str(tmp_path), "rA")
    found = ctx2.concept_index.lookup("__loaded_op", sid1)   # lazy 重建命中载入
    assert found == ref, f"fresh ConceptIndex lazy 重建须命中载入概念点·得 {found} ≠ {ref}"


def test_concept_index_observe_dedup_after_load(tmp_path):
    """Task #475 latent corrupt fix：续训 ensure 同 surface 须 dedup 命中载入 local_id（非建重复）。

    无 fix：_index 空→ensure 建新 local_id→同身份两节点→图碎裂。修后：lazy 重建命中→dedup。
    """
    from pure_integer_ai.experiments.formal_train import make_train_context
    from pure_integer_ai.training.cursor import dump_run, load_run, DUMP_TABLES
    from pure_integer_ai.storage.concept_identity import CONCEPT_IDENTITY_TABLE
    b1 = DictBackend(); ctx1 = make_train_context(b1)
    sid1 = ctx1.space_id
    lid_A = ctx1.concept_index.ensure("__dedup_surf", space_id=sid1, tier=TIER_PRIMARY,
                                      node_type=NODE_CONCEPT)[1]
    dump_run(b1, str(tmp_path), "rD", spaces=[sid1],
             tables=DUMP_TABLES + (CONCEPT_IDENTITY_TABLE,))
    b2 = DictBackend(); ctx2 = make_train_context(b2)
    load_run(b2, str(tmp_path), "rD")
    lid_redo = ctx2.concept_index.ensure("__dedup_surf", space_id=sid1, tier=TIER_PRIMARY,
                                         node_type=NODE_CONCEPT)[1]
    assert lid_redo == lid_A, (
        f"续训 ensure 同 surface 须 dedup 命中 lid_A={lid_A}·非建重复 lid={lid_redo}（latent corrupt 修复）")


def test_concept_index_bare_fixture_backward_compat(disc_env):
    """Task #475 向后兼容：disc_env 未注册 concept_identity → ensure/lookup 仍工作（内存 _index·best-effort skip 持久化·不崩）。"""
    b, sid, es, ci, _ = disc_env
    # disc_env 只注册 composes_attr·无 concept_identity → record_concept_identity best-effort skip
    ref = ci.ensure("__bare_test", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    found = ci.lookup("__bare_test", sid)
    assert found == ref, "bare fixture（无 concept_identity）ensure/lookup 仍须工作（向后兼容）"
    from pure_integer_ai.storage.concept_identity import load_space_identity
    assert load_space_identity(b, sid) == {}, "表未注册 → 返空 dict（向后兼容·不崩）"


def test_concept_index_lazy_rebuild_once(disc_env):
    """Task #475 lazy 效率：_ensure_space_loaded per space once·多次 access 同 space 不重扫（_loaded_spaces 稳定）。"""
    b, sid, es, ci, _ = disc_env
    ci.ensure("__lz_a", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    assert ci._loaded_spaces == {sid}, "首次 ensure 触发 lazy load·_loaded_spaces={sid}"
    ci.ensure("__lz_b", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    ci.lookup("__lz_a", sid)
    assert ci._loaded_spaces == {sid}, "同 space 多次 access 不重扫（per-space once）"


def test_cross_run_loaded_operator_inlines(tmp_path):
    """Task #475 headline e2e（载入算子可 inline）：run N 发现 square → dump（concept_identity）→
    run N+1 load → fresh ConceptIndex lazy 重建 → 引用载入 square 建 consumer → _try_inline_learned lookup
    命中载入 name_ref → inline+β → vm_proof square(5)=25。**_index 重建闭环铁证 = 序列7 载入算子可 inline。**
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig, make_train_context
    from pure_integer_ai.training.cursor import load_run
    from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
    from pure_integer_ai.training.vm_proof import execute_composes_value
    # Run N：发现 square（operand 样本）→ dump（formal_train 默认 dump_tables 含 concept_identity）
    b1 = DictBackend()
    corpus_N = [_arith_item("lambda x: x * x"), _arith_item("lambda y: y * y")]
    res_N = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="runIN",
                                           rounds_per_stage=1), corpus_N, backend=b1)
    assert len(res_N.discovered_operators) == 1, "run N 须发现 square"
    sq_name = res_N.discovered_operators[0].name
    # Run N+1：make_train_context（bootstrap+core space）→ load_run（载入 runIN 数据·含 concept_identity）
    # → fresh ConceptIndex lazy 重建 → 引用载入 square 建 consumer → inline+β
    b2 = DictBackend()
    ctx2 = make_train_context(b2)
    load_run(b2, str(tmp_path), "runIN")
    sid2 = ctx2.space_id   # "core" 确定名 → == runIN 的 core space_id
    consumer = ctx2.concept_index.ensure("__cons_inl", space_id=sid2, tier=TIER_PRIMARY,
                                         node_type=NODE_CONCEPT)
    build_composes_from_arith(f"lambda n: {sq_name}(n)", concept_index=ctx2.concept_index,
                              edge_store=ctx2.edge_store, backend=b2, space_id=sid2,
                              source=SOURCE_MATH, root_ref=consumer)
    # _try_inline_learned(sq_name) lookup 命中载入 name_ref（lazy 重建）→ 嫁接载入 square skeleton → n*n
    v = execute_composes_value(ctx2.concept_graph, consumer, ((5, 1),))
    assert v is not None and rational.eq(v, rational.make(25, 1)), (
        f"载入 square 须 inline → square(5)=25（_index 重建闭环）·得 {v}")


# ============ Task #476 同类样本分组（序列2 另半·within-run 混合 shape 组·§八.5 诚实边界②）===========
#
# 同 shape_signature 异 operand 结构（square(x*x) hint=1 vs mul(a*b) hint=2）原合并一组→probe_arity
# cross-sample 门 None→不发现。修：(shape_sig, operand_arity_hint) 分组·分离异构·各同质组独立发现。


def test_operand_arity_hint_values(disc_env):
    """Task #476 helper：_operand_arity_hint = distinct operand sid 数（grouping 键·非最终 arity）。

    square(x*x·两 OPERAND 同 sid)→1 / mul-operand(a*b·两 OPERAND 异 sid)→2 / 立即数(5*5·无 OPERAND)→0。
    """
    from pure_integer_ai.cognition.process.structure_discover import _operand_arity_hint
    _, _, _, _, g = disc_env
    sq = _build(disc_env, "lambda x: x * x", seg_label="__h_sq")
    mul_op = _build(disc_env, "lambda p, q: p * q", seg_label="__h_mul")
    imm = _build(disc_env, "lambda: 5 * 5", seg_label="__h_imm")
    assert _operand_arity_hint(g, sq) == 1, "square 两 OPERAND 同 sid → hint=1"
    assert _operand_arity_hint(g, mul_op) == 2, "mul 两 OPERAND 异 sid → hint=2"
    assert _operand_arity_hint(g, imm) == 0, "立即数无 OPERAND → hint=0"


def test_mixed_square_mul_discovers_both(disc_env):
    """Task #476 headline：within-run 混合 square+mul 语料 → (sig,hint) 分组 → 两者各独立发现。

    无 fix：同 shape_signature 合并一组→probe_arity cross-sample 门 None→皆不发现。
    修后：(MUL_sig,hint=1)=[x*x,y*y]→square arity1·(MUL_sig,hint=2)=[a*b,c*d]→mul arity2·各发现。
    """
    b, sid, es, ci, _ = disc_env
    sq_roots = [_build(disc_env, "lambda x: x * x", seg_label="__mx_s1"),
                _build(disc_env, "lambda y: y * y", seg_label="__mx_s2")]
    mul_roots = [_build(disc_env, "lambda p, q: p * q", seg_label="__mx_m1"),
                 _build(disc_env, "lambda r, s: r * s", seg_label="__mx_m2")]
    ops = auto_discover_operators(sq_roots + mul_roots, concept_index=ci, edge_store=es,
                                  backend=b, space_id=sid, source=SOURCE_MATH)
    arities = sorted(op.arity for op in ops)
    assert arities == [1, 2], f"混合 square+mul 须各独立发现（arity 1+2）·得 {arities}（无 fix 则空）"


def test_mixed_operand_immediate_discovers_both(disc_env):
    """Task #476 同理：operand square + 立即数 mul 混合 → hint 分离（1 vs 0）→ 各独立发现。

    无 fix：同 shape 合并→probe None→皆不发现。修后：分离→square(hint=1)+mul-immediate(hint=0) 各发现。
    """
    b, sid, es, ci, _ = disc_env
    sq_roots = [_build(disc_env, "lambda x: x * x", seg_label="__mi_s1"),
                _build(disc_env, "lambda y: y * y", seg_label="__mi_s2")]
    imm_roots = [_build(disc_env, "lambda: 5 * 5", seg_label="__mi_i1"),
                 _build(disc_env, "lambda: 6 * 6", seg_label="__mi_i2")]
    ops = auto_discover_operators(sq_roots + imm_roots, concept_index=ci, edge_store=es,
                                  backend=b, space_id=sid, source=SOURCE_MATH)
    arities = sorted(op.arity for op in ops)
    assert arities == [1, 2], (
        f"operand square(arity1) + 立即数 mul(arity2) 须各独立发现·得 {arities}（无 fix 则空）")


def test_homogeneous_grouping_unchanged(disc_env):
    """Task #476 回归：同质组（纯立即数 mul）hint 分组后行为不变（仍发现 mul arity2·bit-identical）。"""
    b, sid, es, ci, _ = disc_env
    roots = [_build(disc_env, "lambda: 5 * 5", seg_label="__hg_1"),
             _build(disc_env, "lambda: 6 * 6", seg_label="__hg_2")]
    ops = auto_discover_operators(roots, concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, source=SOURCE_MATH)
    assert len(ops) == 1 and ops[0].arity == 2, "同质立即数 mul 仍发现 arity2（hint 分组不改同质行为）"


def test_formal_train_mixed_corpus_discovers_both(tmp_path):
    """Task #476 生产 e2e：formal_train 混合 square+mul 语料 → result.discovered_operators 含两者。

    formal_train per-(sig,hint) 分组 + held-out split·混合语料两者各独立发现（无 fix 则 probe None→空）。
    """
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
    b = DictBackend()
    corpus = [_arith_item("lambda x: x * x"), _arith_item("lambda y: y * y"),
              _arith_item("lambda p, q: p * q"), _arith_item("lambda r, s: r * s")]
    res = formal_train(FormalTrainConfig(run_dir=str(tmp_path), run_id="mixE2e",
                                         rounds_per_stage=1), corpus, backend=b)
    arities = sorted(op.arity for op in res.discovered_operators)
    assert arities == [1, 2], (
        f"formal_train 混合 square+mul 语料须各独立发现·得 {arities}（无 fix 则空）")
