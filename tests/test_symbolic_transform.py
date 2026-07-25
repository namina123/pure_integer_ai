"""符号数学能力扩展测试（Phase 1·_align_walk 子树绑定·doc/重来_符号数学能力扩展设计_2026-07-15 §八-bis.5）。

机制：_align_walk 加 subtree_binding 参数（默认 None·**None 守 bit-identical**）·PARAM 槽遇 input
复合子树（ATTR_OPERATOR 节点）→绑 slot→子树根（第4路·解符号变换命门：d/dx 的 VAR 须绑 x+1 子树·
原 _align_walk:1299 拒）。既有 recognize_operators/_align_extract caller 不传 subtree_binding（默认 None）
→第4路 inactive→零行为变。

TC1 bit-identical None 守：subtree_binding=None → PARAM 槽遇复合子树 → return False（既有行为·零翻）。
TC2 子树绑定真活：pattern Mul(PARAM,IMM2) vs input Mul(x+1,2) → subtree_binding={0: Add 子树根}·match True。
TC3 变量同一性：pattern Mul(PARAM,PARAM) vs input Mul(x+1,x+1) 两异 ConceptRef (x+1) → 同槽异子树→return False。
TC4a 既有 IMMEDIATE 路不扰：subtree_binding={} · PARAM 遇 IMMEDIATE 叶 → value_binding（1st 路·非 4th）。
TC4b 既有 OPERAND 路不扰：subtree_binding={} · PARAM 遇 OPERAND 叶 → operand_binding（2nd 路·非 4th）。

铁律：纯整数 / bit-identical（None 守·既有 caller 零行为变）/ 反 theater（子树绑定真活·非 stub）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_MATH
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.process.structure_discover import _align_walk
from pure_integer_ai.experiments.formal_train import make_train_context, _run_task_driven_generate
from pure_integer_ai.training.symbolic_transform import (
    register_transform_rule, load_transform_rule, apply_transform, induce_transform_rule)
from pure_integer_ai.training.vm_proof import execute_composes_value
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.cognition.shared.types import (
    MODALITY_ARITH, DOMAIN_MATH, LANG_NONE, TERMINAL_REACHED_SINK,
    VERIFY_SOURCE_SELF_PRODUCED, TransformSpec, TransformHeldOut)


def _build_two(exprs):
    """同一 ctx 建 2 lambda COMPOSES 树·返 (ctx, [root_a, root_b])。同 backend 让 _align_walk 读两树。"""
    ctx = make_train_context(DictBackend())
    roots = []
    for e in exprs:
        r = ctx.concept_index.ensure(f"__seg_{e}", space_id=ctx.space_id,
                                     tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        build_composes_from_arith(e, concept_index=ctx.concept_index,
                                  edge_store=ctx.edge_store, backend=ctx.backend,
                                  space_id=ctx.space_id, source=SOURCE_MATH, root_ref=r)
        roots.append(r)
    return ctx, roots


def _children(backend, root):
    """读 root 的 children_of（read_composes_tree·按 order_index 排）。"""
    g = ConceptGraph(backend)
    ch, _op, _operand, _imm, _st = g.read_composes_tree(root)
    return ch


# ---- TC1 bit-identical None 守（subtree_binding=None → 复合子树遇 PARAM 槽 → return False） ----

def test_tc1_none_guard_composite_rejected():
    """subtree_binding=None（默认）→ 第4路 inactive → PARAM 槽遇 input 复合子树（Add）→ return False。
    既有 recognize_operators/_align_extract caller 不传 subtree_binding → 零行为变（bit-identical）。"""
    ctx, (pat, inp) = _build_two(["lambda x: x*2", "lambda x: (x+1)*2"])
    sk_ch = _children(ctx.backend, pat)   # pattern: NOP→MUL(operand_0, imm_2)
    in_ch = _children(ctx.backend, inp)   # input:  NOP→MUL(Add(x,1), imm_2)
    ok = _align_walk(ctx.backend, sk_ch, in_ch, pat, inp,
                     {}, {}, {}, 0, ancestor_map=None, subtree_binding=None)
    assert ok is False, (
        "None 守：subtree_binding=None → 第4路 inactive → PARAM 槽遇复合子树 Add → return False"
        "（既有行为·bit-identical·既有 caller 零翻）")


# ---- TC2 子树绑定真活（pattern Mul(PARAM,IMM2) vs input Mul(x+1,2) → 绑 Add 子树） ----

def test_tc2_subtree_binding_binds_composite():
    """subtree_binding={} → 第4路 active → PARAM 槽遇 input 复合子树 Add → 绑 slot 0→Add 子树根·match True。
    解符号变换命门：d/dx 的 VAR 须绑 x+1 子树·原 :1299 拒·现第4路绑。"""
    ctx, (pat, inp) = _build_two(["lambda x: x*2", "lambda x: (x+1)*2"])
    sk_ch = _children(ctx.backend, pat)
    in_ch = _children(ctx.backend, inp)
    subtree_binding: dict = {}
    ok = _align_walk(ctx.backend, sk_ch, in_ch, pat, inp,
                     {}, {}, {}, 0, ancestor_map=None, subtree_binding=subtree_binding)
    assert ok is True, "子树绑定：PARAM 槽遇复合子树 Add → 绑定→match True（第4路真活）"
    # subtree_binding[0] 应 = input MUL 的 child 0 = Add(x,1) 子树根
    in_mul = in_ch[inp][0]              # input NOP→child 0 = MUL
    in_add = in_ch[in_mul][0]           # input MUL→child 0 = Add(x,1)
    assert 0 in subtree_binding, "slot 0 已绑"
    assert subtree_binding[0] == in_add, (
        "subtree_binding[0] = input Add(x,1) 子树根（复合子树绑定·非叶·非 IMM 值）")


# ---- TC3 变量同一性（pattern Mul(PARAM,PARAM) vs input Mul(x+1,x+1) 两异 Add → 同槽异子树→拒） ----

def test_tc3_variable_identity_rejects_distinct_subtrees():
    """pattern Mul(operand_0, operand_0)（同 sid·同 slot 0）vs input Mul(Add_1, Add_2)（两异 ConceptRef
    (x+1)·build 各 fresh）→ slot 0 先绑 Add_1·第二位 Add_2≠Add_1 → 变量同一性拒→return False。
    对称既有 value/operand/concept 三 binding 防御（square 两叶同槽须同值/同 operand slot/同 token）。"""
    ctx, (pat, inp) = _build_two(["lambda x: x*x", "lambda x: (x+1)*(x+1)"])
    sk_ch = _children(ctx.backend, pat)
    in_ch = _children(ctx.backend, inp)
    subtree_binding: dict = {}
    ok = _align_walk(ctx.backend, sk_ch, in_ch, pat, inp,
                     {}, {}, {}, 0, ancestor_map=None, subtree_binding=subtree_binding)
    assert ok is False, (
        "变量同一性：同 slot 0 两异 Add 子树（Add_1≠Add_2·build 各 fresh ConceptRef）→ 拒"
        "（Pow(VAR,VAR) 两 VAR 同槽须绑同子树·对称既有三 binding）")


# ---- TC4a 既有 IMMEDIATE 路不扰（PARAM 遇 IMMEDIATE 叶 → value_binding·非 4th） ----

def test_tc4a_immediate_path_unchanged():
    """subtree_binding={} · PARAM 槽遇 input IMMEDIATE 叶 → 1st 路 value_binding（既有·非 4th 子树绑定）。
    subtree_binding 保持空（4th 路未误激活·既有 IMMEDIATE 路优先·bit-identical）。"""
    ctx, (pat, inp) = _build_two(["lambda x: x*2", "lambda: 5*2"])   # input nullary 5*2
    sk_ch = _children(ctx.backend, pat)
    in_ch = _children(ctx.backend, inp)
    value_binding: dict = {}
    subtree_binding: dict = {}
    ok = _align_walk(ctx.backend, sk_ch, in_ch, pat, inp,
                     value_binding, {}, {}, 0, ancestor_map=None, subtree_binding=subtree_binding)
    assert ok is True, "IMMEDIATE 路：PARAM 遇 IMM 叶 → value_binding·match True（既有·bit-identical）"
    assert 0 in value_binding, "value_binding[0] = 5（1st 路·IMMEDIATE 值绑定）"
    assert value_binding[0] == (5, 1), "value_binding[0] = (5,1)（input 5）"
    assert 0 not in subtree_binding, "subtree_binding 未激活（4th 路不误扰 IMMEDIATE 路）"


# ---- TC4b 既有 OPERAND 路不扰（subtree_binding=None·既有 caller·PARAM 遇 OPERAND 叶 → operand_binding） ----

def test_tc4b_operand_path_unchanged():
    """subtree_binding=None（既有 recognize_operators caller）· PARAM 遇 OPERAND 叶 → operand_binding
    （既有 2nd 路·bit-identical·零行为变）。注：符号模式（subtree_binding={}）下 operand→subtree_binding
    （Phase 2b 改动·d/dx operand base 需要·TC9/TC10 验）·非既有 caller 范围。"""
    ctx, (pat, inp) = _build_two(["lambda x: x*2", "lambda y: y*2"])   # input operand y*2
    sk_ch = _children(ctx.backend, pat)
    in_ch = _children(ctx.backend, inp)
    operand_binding: dict = {}
    ok = _align_walk(ctx.backend, sk_ch, in_ch, pat, inp,
                     {}, operand_binding, {}, 0, ancestor_map=None, subtree_binding=None)
    assert ok is True, "OPERAND 路（None 守）：PARAM 遇 OPERAND 叶 → operand_binding·match True（既有·bit-identical）"
    assert 0 in operand_binding, "operand_binding[0] = input operand slot（2nd 路·既有·subtree_binding=None 守）"


# ============================================================
# Phase 2 测试（符号变换规则存储 + apply·doc/重来_符号数学能力扩展设计 §八-bis）
#
# 机制：register_transform_rule（name→LHS+RHS struct_ref·ATTR_TRANSFORM_LHS/RHS）+ apply_transform
# （_align_walk LHS 匹配 input→subtree/value 绑定→_deep_copy_subtree β-替换 RHS→输出）。
# Phase 2 scope：纯替换（无 Pow/无值算术）·测试分配律 a*(b+c)→a*b+a*c。
#
# TC5 register+load round-trip：ATTR_TRANSFORM_LHS/RHS 存盘+读回。
# TC6 register 幂等+冲突 fail-loud：同 (name,lhs,rhs) 重注册不变·异 fail-loud。
# TC7 apply 分配律 cross-verify：register 分配律·apply (x+1)*((y-1)+3)→(x+1)*(y-1)+(x+1)*3·执行 input==output 验正确。
# TC8 apply 非匹配→None：input 结构不符 LHS→None（诚实 skip·非 theater）。
# ============================================================

def _build_many(exprs):
    """同一 ctx 建 N lambda COMPOSES 树·返 (ctx, {expr: root})。"""
    ctx = make_train_context(DictBackend())
    roots = {}
    for e in exprs:
        r = ctx.concept_index.ensure(f"__seg_{e}", space_id=ctx.space_id,
                                     tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        build_composes_from_arith(e, concept_index=ctx.concept_index,
                                  edge_store=ctx.edge_store, backend=ctx.backend,
                                  space_id=ctx.space_id, source=SOURCE_MATH, root_ref=r)
        roots[e] = r
    return ctx, roots


# ---- TC5 register+load round-trip（ATTR_TRANSFORM_LHS/RHS 存盘+读回） ----

def test_tc5_register_load_roundtrip():
    """register_transform_rule 存 ATTR_TRANSFORM_LHS/RHS·load_transform_rule 读回 (lhs,rhs) 对。"""
    ctx, roots = _build_many(["lambda p,q,r: p*(q+r)", "lambda p,q,r: p*q+p*r"])
    lhs_ref = roots["lambda p,q,r: p*(q+r)"]
    rhs_ref = roots["lambda p,q,r: p*q+p*r"]
    rule_ref = register_transform_rule(ctx.backend, ctx.concept_index, "distrib",
                                       lhs_ref, rhs_ref, space_id=ctx.space_id)
    loaded = load_transform_rule(ctx.backend, rule_ref)
    assert loaded is not None, "load_transform_rule 返 (lhs,rhs)·非 None（规则已注册）"
    assert loaded[0] == lhs_ref, "load lhs == 注册 lhs"
    assert loaded[1] == rhs_ref, "load rhs == 注册 rhs"
    # 非规则节点 load → None
    assert load_transform_rule(ctx.backend, lhs_ref) is None, "非规则节点 load → None（无 ATTR_TRANSFORM_*）"


# ---- TC6 register 幂等+冲突 fail-loud ----

def test_tc6_register_idempotent_and_conflict():
    """同 (name,lhs,rhs) 重注册幂等不变·异 (lhs,rhs) 重名 fail-loud（拒歧义·反 theater）。"""
    ctx, roots = _build_many(["lambda p,q,r: p*(q+r)", "lambda p,q,r: p*q+p*r",
                              "lambda p,q,r: p+q+r"])
    lhs_ref = roots["lambda p,q,r: p*(q+r)"]
    rhs_ref = roots["lambda p,q,r: p*q+p*r"]
    other_ref = roots["lambda p,q,r: p+q+r"]
    rule_ref = register_transform_rule(ctx.backend, ctx.concept_index, "distrib",
                                       lhs_ref, rhs_ref, space_id=ctx.space_id)
    # 幂等：同 (name, lhs, rhs) 重注册 → 同 name_ref·不变
    rule_ref2 = register_transform_rule(ctx.backend, ctx.concept_index, "distrib",
                                        lhs_ref, rhs_ref, space_id=ctx.space_id)
    assert rule_ref == rule_ref2, "幂等：同 (name,lhs,rhs) 重注册返同 name_ref"
    # 冲突：同 name 异 (lhs,rhs) → ValueError fail-loud
    with pytest.raises(ValueError, match="重名冲突"):
        register_transform_rule(ctx.backend, ctx.concept_index, "distrib",
                                lhs_ref, other_ref, space_id=ctx.space_id)


# ---- TC7 apply 分配律 cross-verify（执行 input==output 验正确） ----

def test_tc7_apply_distributivity_cross_verify():
    """register 分配律 a*(b+c)→a*b+a*c·apply input (x+1)*((y-1)+3)→输出·执行 input==output 验正确。

    LHS Mul(PARAM_a, Add(PARAM_b, PARAM_c))·input Mul(Add(x,1), Add(Sub(y,1), 3))：
    PARAM_a→Add(x,1) 子树·PARAM_b→Sub(y,1) 子树·PARAM_c→3 IMM 值。
    RHS Add(Mul(a,b), Mul(a,c))→β-替换（a 重复 fresh-copy）→(x+1)*(y-1)+(x+1)*3。
    cross-verify：执行 input vs output 在采样点 (x,y) 等价（分配律正确·stable≠correct·统计验非 truth）。
    """
    ctx, roots = _build_many([
        "lambda p,q,r: p*(q+r)",        # LHS 模式
        "lambda p,q,r: p*q+p*r",        # RHS 模板
        "lambda x,y: (x+1)*((y-1)+3)",  # input（a=(x+1)·b=(y-1)·c=3）
    ])
    lhs_ref = roots["lambda p,q,r: p*(q+r)"]
    rhs_ref = roots["lambda p,q,r: p*q+p*r"]
    input_ref = roots["lambda x,y: (x+1)*((y-1)+3)"]
    rule_ref = register_transform_rule(ctx.backend, ctx.concept_index, "distrib",
                                       lhs_ref, rhs_ref, space_id=ctx.space_id)

    output_ref = apply_transform(ctx.backend, ctx.concept_index, ctx.edge_store,
                                 space_id=ctx.space_id, source=SOURCE_MATH,
                                 rule_name_ref=rule_ref, input_ref=input_ref)
    assert output_ref is not None, "apply 产输出（LHS 匹配 input·PARAM a/b 子树 + c 值绑定）"

    # cross-verify：执行 input vs output 在采样点 (x,y) 等价
    graph = ConceptGraph(ctx.backend)
    for x, y in [(2, 5), (3, 2), (0, 7), (-1, 4)]:
        params = ((x, 1), (y, 1))
        v_in = execute_composes_value(graph, input_ref, params)
        v_out = execute_composes_value(graph, output_ref, params)
        assert v_in is not None and v_out is not None, f"执行非 None（x={x},y={y}）"
        assert rational.eq(v_in, v_out), (
            f"分配律 cross-verify：input==output @ (x={x},y={y})"
            f"（a*(b+c)=a*b+a*c·统计验正确·stable≠correct 非 truth）")


# ---- TC8 apply 非匹配→None（LHS 结构不符 input→诚实 skip·非 theater） ----

def test_tc8_apply_non_matching_returns_none():
    """input 结构不符 LHS（LHS Mul(PARAM,Add(PARAM,PARAM))·input Mul(x,y) 第二操作数非 Add）→
    _align_walk 不匹配→apply 返 None（诚实 skip·非伪造输出·反 theater）。"""
    ctx, roots = _build_many([
        "lambda p,q,r: p*(q+r)",
        "lambda p,q,r: p*q+p*r",
        "lambda x,y: x*y",   # input：Mul(operand_x, operand_y)·第二操作数 operand 非 Add
    ])
    lhs_ref = roots["lambda p,q,r: p*(q+r)"]
    rhs_ref = roots["lambda p,q,r: p*q+p*r"]
    input_ref = roots["lambda x,y: x*y"]
    rule_ref = register_transform_rule(ctx.backend, ctx.concept_index, "distrib",
                                       lhs_ref, rhs_ref, space_id=ctx.space_id)
    output_ref = apply_transform(ctx.backend, ctx.concept_index, ctx.edge_store,
                                 space_id=ctx.space_id, source=SOURCE_MATH,
                                 rule_name_ref=rule_ref, input_ref=input_ref)
    assert output_ref is None, (
        "LHS 不匹配 input（Mul 第二操作数 operand 非 Add）→ None（诚实 skip·非 theater）")


# ============================================================
# Phase 2b 测试（Pow + 常量折叠·d/dx 命门·doc/重来_符号数学能力扩展设计 §八-bis.3/4）
#
# 机制：OPCODE_POW_PATTERN（Pow 节点·pattern-level）+ _eval_rhs 部分求值（值算术折叠 SUB(n,1)→n-1
# + Pow lower concrete 指数→MUL + 子树/operand/concept 绑定全→subtree_binding）。
# d/dx 规则：LHS Pow(b,n)·RHS n*Pow(b, n-1)。input Pow(x,k)→ PARAM_b=x(operand→subtree)·PARAM_n=k(IMM→value)。
# 应用：折叠 n-1 → lower Pow(b, n-1) concrete → 输出 n*b^(n-1)（Pow 全 lower·可执行）。
#
# TC9 d/dx Pow(x,2)→2x：register d/dx·apply→Mul(2,x)·执行 output==2x 验正确。
# TC10 d/dx Pow(y,5)→5y⁴：apply→Mul(5, Pow(y,4))→lower→Mul(5,y*y*y*y)·执行 output==5*y⁴ 验正确。
# ============================================================

def _register_ddx(ctx):
    """建 + 注册 d/dx 规则（LHS Pow(b,n)·RHS n*Pow(b,n-1)）·返 rule_ref。"""
    roots = {}
    for e in ["lambda b,n: Pow(b,n)", "lambda b,n: n * Pow(b, n-1)"]:
        r = ctx.concept_index.ensure(f"__seg_{e}", space_id=ctx.space_id,
                                     tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        build_composes_from_arith(e, concept_index=ctx.concept_index,
                                  edge_store=ctx.edge_store, backend=ctx.backend,
                                  space_id=ctx.space_id, source=SOURCE_MATH, root_ref=r)
        roots[e] = r
    return register_transform_rule(ctx.backend, ctx.concept_index, "ddx",
                                   roots["lambda b,n: Pow(b,n)"],
                                   roots["lambda b,n: n * Pow(b, n-1)"],
                                   space_id=ctx.space_id)


# ---- TC9 d/dx Pow(x,2)→2x（常量折叠 SUB(2,1)=1 + Pow lower exp=1→base + operand x→subtree） ----

def test_tc9_ddx_pow_x_2_to_2x():
    """d/dx Pow(x,2)→2x。PARAM_b=x（operand 叶→subtree_binding·符号模式）·PARAM_n=2（IMM→value_binding）。
    _eval_rhs：Sub(PARAM_n,1)→IMM(2)-IMM(1)=IMM(1)（常量折叠）→Pow(x,1)→lower exp=1→base x→Mul(2,x)。
    cross-verify：执行 output vs expected 2*x @ 采样点等价（d/dx(x²)=2x 统计验正确）。"""
    ctx = make_train_context(DictBackend())
    rule_ref = _register_ddx(ctx)
    # input Pow(x,2)（Pow 节点·不执行·仅匹配）
    input_ref = ctx.concept_index.ensure("__seg_in_ddx2", space_id=ctx.space_id,
                                         tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith("lambda x: Pow(x,2)", concept_index=ctx.concept_index,
                              edge_store=ctx.edge_store, backend=ctx.backend,
                              space_id=ctx.space_id, source=SOURCE_MATH, root_ref=input_ref)
    output_ref = apply_transform(ctx.backend, ctx.concept_index, ctx.edge_store,
                                 space_id=ctx.space_id, source=SOURCE_MATH,
                                 rule_name_ref=rule_ref, input_ref=input_ref)
    assert output_ref is not None, "d/dx apply 产输出（Pow 匹配 + 常量折叠 + Pow lower）"
    # expected 2*x（无 Pow·可执行）
    exp_ref = ctx.concept_index.ensure("__seg_exp_2x", space_id=ctx.space_id,
                                       tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith("lambda x: 2*x", concept_index=ctx.concept_index,
                              edge_store=ctx.edge_store, backend=ctx.backend,
                              space_id=ctx.space_id, source=SOURCE_MATH, root_ref=exp_ref)
    graph = ConceptGraph(ctx.backend)
    for x in [0, 1, 3, -2, 5]:
        v_out = execute_composes_value(graph, output_ref, ((x, 1),))
        v_exp = execute_composes_value(graph, exp_ref, ((x, 1),))
        assert v_out is not None, f"output 可执行（Pow 全 lower·无 OPCODE_POW_PATTERN）@ x={x}"
        assert rational.eq(v_out, v_exp), (
            f"d/dx(x²)=2x cross-verify：output==2x @ x={x}（得 {v_out.num}/{v_out.den}·期 {v_exp.num}/{v_exp.den}）")


# ---- TC10 d/dx Pow(y,5)→5y⁴（常量折叠 SUB(5,1)=4 + Pow lower exp=4→MUL chain） ----

def test_tc10_ddx_pow_y_5_to_5y4():
    """d/dx Pow(y,5)→5y⁴。PARAM_b=y·PARAM_n=5。_eval_rhs：Sub(5,1)=4→Pow(y,4)→lower exp=4→MUL(y,y,y,y)→Mul(5,y*y*y*y)。
    cross-verify：执行 output vs expected 5*y⁴ @ 采样点等价。"""
    ctx = make_train_context(DictBackend())
    rule_ref = _register_ddx(ctx)
    input_ref = ctx.concept_index.ensure("__seg_in_ddx5", space_id=ctx.space_id,
                                         tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith("lambda y: Pow(y,5)", concept_index=ctx.concept_index,
                              edge_store=ctx.edge_store, backend=ctx.backend,
                              space_id=ctx.space_id, source=SOURCE_MATH, root_ref=input_ref)
    output_ref = apply_transform(ctx.backend, ctx.concept_index, ctx.edge_store,
                                 space_id=ctx.space_id, source=SOURCE_MATH,
                                 rule_name_ref=rule_ref, input_ref=input_ref)
    assert output_ref is not None, "d/dx apply 产输出（Pow(y,5) 匹配 + 折叠 + Pow lower）"
    # expected 5*y**4（_build_pow 展开 y**4→MUL chain·可执行）
    exp_ref = ctx.concept_index.ensure("__seg_exp_5y4", space_id=ctx.space_id,
                                       tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith("lambda y: 5*y**4", concept_index=ctx.concept_index,
                              edge_store=ctx.edge_store, backend=ctx.backend,
                              space_id=ctx.space_id, source=SOURCE_MATH, root_ref=exp_ref)
    graph = ConceptGraph(ctx.backend)
    for y in [1, 2, 3, -1]:
        v_out = execute_composes_value(graph, output_ref, ((y, 1),))
        v_exp = execute_composes_value(graph, exp_ref, ((y, 1),))
        assert v_out is not None, f"output 可执行（Pow 全 lower）@ y={y}"
        assert rational.eq(v_out, v_exp), (
            f"d/dx(y⁵)=5y⁴ cross-verify：output==5y⁴ @ y={y}（得 {v_out.num}/{v_out.den}·期 {v_exp.num}/{v_exp.den}）")


# ============================================================
# Phase 3 测试（formal_train task-driven 集成·doc §八-bis.7）
#
# TC11 gate ON 集成：CollectedItem.transform_specs（d/dx 教师 + 2 held-out）→ _run_task_driven_generate
#   → register+apply+cross-verify → 独立 verified episode（reward=1·SELF_PRODUCED·weaning-safe）。
# TC12 gate OFF bit-identical：gate OFF → 无 transform episode（既有行为零翻·bit-identical）。
# ============================================================

def _ddx_item():
    """CollectedItem + d/dx transform_spec（教师陈述规则 + 2 held-out 验证对）。"""
    spec = TransformSpec(
        rule_name="ddx_pow_p3",
        lhs_source="lambda b,n: Pow(b,n)",
        rhs_source="lambda b,n: n * Pow(b, n-1)",
        held_out=(TransformHeldOut("lambda x: Pow(x,2)", "lambda x: 2*x"),
                  TransformHeldOut("lambda y: Pow(y,3)", "lambda y: 3*y*y")),
    )
    return CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                         source=SOURCE_MATH, transform_specs=(spec,))


def test_tc11_formal_train_integration_gate_on():
    """gate ON：CollectedItem.transform_specs（d/dx 教师 + held-out Pow(x,2)→2x + Pow(y,3)→3y²）→
    _run_task_driven_generate → register 规则 + apply held-out + cross-verify 执行等价通过 → 独立 verified episode。
    weaning-safe 决断 A（独立 task-driven episode·不替换 vm_proof·不碎 W7）·verify_source SELF_PRODUCED。"""
    saved = gates.SYMBOLIC_TRANSFORM_MODE
    gates.SYMBOLIC_TRANSFORM_MODE = True
    try:
        ctx = make_train_context(DictBackend())
        episodes, summary = _run_task_driven_generate(ctx, [_ddx_item()], all_ops=[])
        assert summary.total_tasks >= 1, "total_tasks 计 transform_spec（d/dx 规则）"
        assert summary.verified >= 1, (
            "d/dx 规则 cross-verify 通过（Pow(x,2)→2x + Pow(y,3)→3y² @ 探针）→ verified episode")
        # episode SELF_PRODUCED（守"全自产不准停"·反 theater）
        _xform_eps = [e for e in episodes
                      if e.reward == 1 and e.terminal == TERMINAL_REACHED_SINK
                      and e.verify_source == VERIFY_SOURCE_SELF_PRODUCED]
        assert len(_xform_eps) >= 1, (
            "≥1 SELF_PRODUCED verified episode（规则应用+cross-verify single-source·不准驱动停止·反 theater）")
    finally:
        gates.SYMBOLIC_TRANSFORM_MODE = saved


def test_tc12_gate_off_bit_identical():
    """gate OFF：transform_specs 不消费 → 无 transform episode（既有行为零翻·bit-identical）。
    同断桥 Phase A/B TC2/TC6 范式（gate OFF→episodes 无 transform 产物）。"""
    saved = gates.SYMBOLIC_TRANSFORM_MODE
    gates.SYMBOLIC_TRANSFORM_MODE = False
    try:
        ctx = make_train_context(DictBackend())
        # arith item 无 arith_specs/code_source/transform_specs 消费（gate OFF）→ arith task loop 无匹配
        # transform_specs gate OFF 不进 → 无 transform episode。episodes 应为空（MODALITY_ARITH item 无 specs）。
        episodes, summary = _run_task_driven_generate(ctx, [_ddx_item()], all_ops=[])
        assert episodes == [], (
            "gate OFF → transform_specs 不消费 → 无 episode（MODALITY_ARITH item 无 arith_specs·"
            "gate OFF 全路径不产·bit-identical）")
    finally:
        gates.SYMBOLIC_TRANSFORM_MODE = saved


def test_tc13_malformed_spec_graceful_skip():
    """对抗审 Finding 1 修：malformed spec（Pow(x,0)→n-1=-1 负指数→_lower_pow raise UnsupportedConstruct）
    → try/except 守→graceful skip 此 spec（不 abort run·不产 episode）·后续 valid spec 仍处理。
    mirror code_unparse :3753-3759 范式（单 spec 异常不崩整个 run）。"""
    saved = gates.SYMBOLIC_TRANSFORM_MODE
    gates.SYMBOLIC_TRANSFORM_MODE = True
    try:
        ctx = make_train_context(DictBackend())
        # spec1 malformed（Pow(x,0)→0·n=0→n-1=-1→_lower_pow 负指数 raise）+ spec2 valid（Pow(x,2)→2x）
        bad_spec = TransformSpec(
            "ddx_bad_p0", "lambda b,n: Pow(b,n)", "lambda b,n: n * Pow(b, n-1)",
            (TransformHeldOut("lambda x: Pow(x,0)", "lambda x: 0"),))
        good_spec = TransformSpec(
            "ddx_good_p2", "lambda b,n: Pow(b,n)", "lambda b,n: n * Pow(b, n-1)",
            (TransformHeldOut("lambda x: Pow(x,2)", "lambda x: 2*x"),))
        item = CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                             source=SOURCE_MATH, transform_specs=(bad_spec, good_spec))
        episodes, summary = _run_task_driven_generate(ctx, [item], all_ops=[])
        # 两 spec 都计入 total_tasks（各 register+验 任务）·malformed spec1 graceful skip（不 abort）
        assert summary.total_tasks >= 2, "total_tasks 计两 spec（malformed 不 abort·仍计数）"
        # good_spec2 cross-verify 通过 → verified（malformed spec1 不计 verified·graceful skip）
        assert summary.verified >= 1, (
            "good_spec verified·malformed spec graceful skip（try/except 守·不 abort run·反 theater 不伪造）")
    finally:
        gates.SYMBOLIC_TRANSFORM_MODE = saved


def test_tc14_curriculum_multi_rule_s5_s8():
    """S5-S8 课程语料（doc §五-bis.4）：多规则 transform_specs（分配律 S5 + d/dx 幂规则 S6 + 交换律 S5）
    → formal_train gate ON → 学全→多 verified episode（每规则一 episode·"断奶前广学符号+计算"机制验）。
    诚实：本测验课程机制（多规则 formal_train 学全）·非完整"几十-上百轮"regimen（那是持续训练过程·非单测）。"""
    saved = gates.SYMBOLIC_TRANSFORM_MODE
    gates.SYMBOLIC_TRANSFORM_MODE = True
    try:
        ctx = make_train_context(DictBackend())
        specs = (
            # S5 分配律 a*(b+c) → a*b+a*c（重排·输出同构·PARAM 绑子树）
            TransformSpec("distrib", "lambda p,q,r: p*(q+r)", "lambda p,q,r: p*q+p*r",
                          (TransformHeldOut("lambda x: (x+1)*(x+2)", "lambda x: (x+1)*x+(x+1)*2"),)),
            # S6 d/dx 幂规则 Pow(b,n) → n*Pow(b,n-1)（算术构造·PARAM 值算术折叠+Pow lower）
            TransformSpec("ddx_pow", "lambda b,n: Pow(b,n)", "lambda b,n: n*Pow(b,n-1)",
                          (TransformHeldOut("lambda x: Pow(x,2)", "lambda x: 2*x"),
                           TransformHeldOut("lambda y: Pow(y,4)", "lambda y: 4*y*y*y")),),
            # S5 乘法交换律 a*b → b*a（重排·验证 operand→subtree_binding 双向）
            TransformSpec("mul_comm", "lambda p,q: p*q", "lambda p,q: q*p",
                          (TransformHeldOut("lambda x: (x+1)*2", "lambda x: 2*(x+1)"),)),
        )
        item = CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                             source=SOURCE_MATH, transform_specs=specs)
        episodes, summary = _run_task_driven_generate(ctx, [item], all_ops=[])
        assert summary.total_tasks >= 3, "total_tasks 计 3 课程规则（广学）"
        assert summary.verified >= 3, (
            "3 课程规则全 cross-verify 通过 → ≥3 verified episode（分配律+d/dx+交换律·断奶前广学符号+计算机制验）")
    finally:
        gates.SYMBOLIC_TRANSFORM_MODE = saved




# ============================================================
# 自归纳测（single-pair symbolic induction·doc §八-bis.2）
#
# TC15 归纳交换律：从符号例 (p*q, q*p) 归纳规则 Mul(a,b)→Mul(b,a)·apply (x+1)*2→2*(x+1) cross-verify。
# TC16 归纳 d/dx：从符号例 (Pow(b,n), n*Pow(b,n-1)) 归纳规则·apply Pow(x,2)→2x cross-verify（单对符号例归纳通用规则）。
# ============================================================

def test_tc15_induce_commutativity():
    """自归纳交换律：符号例 (lambda p,q: p*q, lambda p,q: q*p) → 归纳规则 Mul(a,b)→Mul(b,a)。
    operand p,q→PARAM_0,1·LHS=Mul(PARAM_0,PARAM_1)·RHS=Mul(PARAM_1,PARAM_0)。apply (x+1)*2→2*(x+1) cross-verify。"""
    ctx, roots = _build_many(["lambda p,q: p*q", "lambda p,q: q*p", "lambda x: (x+1)*2", "lambda x: 2*(x+1)"])
    rule_ref = induce_transform_rule(ctx.backend, ctx.concept_index, ctx.edge_store,
                                     space_id=ctx.space_id, source=SOURCE_MATH, name="mul_comm_ind",
                                     input_ref=roots["lambda p,q: p*q"], output_ref=roots["lambda p,q: q*p"])
    out_ref = apply_transform(ctx.backend, ctx.concept_index, ctx.edge_store,
                              space_id=ctx.space_id, source=SOURCE_MATH,
                              rule_name_ref=rule_ref, input_ref=roots["lambda x: (x+1)*2"])
    assert out_ref is not None, "归纳的交换律 apply 产输出"
    graph = ConceptGraph(ctx.backend)
    for x in [0, 1, 3, -2]:
        vo = execute_composes_value(graph, out_ref, ((x, 1),))
        ve = execute_composes_value(graph, roots["lambda x: 2*(x+1)"], ((x, 1),))
        assert vo is not None and rational.eq(vo, ve), (
            f"归纳交换律 cross-verify：(x+1)*2→2*(x+1) @ x={x}（自归纳产通用规则）")


def test_tc16_induce_ddx_from_symbolic_pair():
    """自归纳 d/dx：符号例 (Pow(b,n), n*Pow(b,n-1)) → 归纳规则·apply Pow(x,2)→2x。
    单对符号例归纳**通用规则**（PARAM 泛化·Pow(b,n) 的 b,n→PARAM·n-1 立即数 1 fixed）。
    cross-verify Pow(x,2)→2x @ 多点·证自归纳 d/dx 正确（vs teacher-stated TC9·自归纳是更纯学习模式）。"""
    ctx, roots = _build_many([
        "lambda b,n: Pow(b,n)", "lambda b,n: n*Pow(b,n-1)",
        "lambda x: Pow(x,2)", "lambda x: 2*x"])
    rule_ref = induce_transform_rule(ctx.backend, ctx.concept_index, ctx.edge_store,
                                     space_id=ctx.space_id, source=SOURCE_MATH, name="ddx_ind",
                                     input_ref=roots["lambda b,n: Pow(b,n)"],
                                     output_ref=roots["lambda b,n: n*Pow(b,n-1)"])
    out_ref = apply_transform(ctx.backend, ctx.concept_index, ctx.edge_store,
                              space_id=ctx.space_id, source=SOURCE_MATH,
                              rule_name_ref=rule_ref, input_ref=roots["lambda x: Pow(x,2)"])
    assert out_ref is not None, "归纳的 d/dx apply 产输出（Pow lower + 常量折叠）"
    graph = ConceptGraph(ctx.backend)
    for x in [0, 1, 3, -2, 5]:
        vo = execute_composes_value(graph, out_ref, ((x, 1),))
        ve = execute_composes_value(graph, roots["lambda x: 2*x"], ((x, 1),))
        assert vo is not None and rational.eq(vo, ve), (
            f"归纳 d/dx cross-verify：Pow(x,2)→2x @ x={x}（单对符号例归纳通用 d/dx 规则）")
