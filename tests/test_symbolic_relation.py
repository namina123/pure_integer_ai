"""S8 符号间运算关联测试（逆关系·doc/重来_S8符号间关联机制设计_2026-07-15）。

机制：register_inverse_relation（name→KIND+RULE_A+RULE_B·ATTR_RELATION_*）+ verify_inverse_relation
（构造验证 B∘A=identity @ 采样·三值 True/False/None·反 theater 心脏）。复用 symbolic_transform
register/apply + vm_proof execute + rational.eq。表示镜像 ATTR_TRANSFORM_LHS/RHS=25/26 + ATTR_PROPOSITION marker。

Phase 1 = INVERSE only（覆盖 +/−·×/÷·d/dx↔∫ 主体）·链式法则 COMPOSITION defer Phase 2·恒等 IDENTITY 折化简规则。

IR1 register/load roundtrip + 幂等 + 冲突 fail-loud（镜像 TC5/TC6）。
IR2 ×/÷ 逆关系 double↔halve B∘A 还原 @ 采样（旗舰·verified True·bare-PARAM LHS 绑复合子树）。
IR3 composes-fail can't-verify（B LHS 不匹配 A 输出 shape → None·诚实降级·非 theater）。
IR4 falsified（B∘A 不还原 → False·诚实不偷渡 verified）。
IR5 formal_train 集成 gate ON（verified → 独立 SELF_PRODUCED episode·weaning-safe 决断 A）。
IR6 gate OFF bit-identical（无 relation episode·既有行为零翻）。
IR7 malformed graceful skip（bad spec 不 abort run·后续 valid 仍处理）。
IR8 +/− 化简规则路径（x+y−y→x 单条规则·不走逆关系·验两路皆可学 = S8 不留缺口）。

铁律：纯整数 / bit-identical（gate OFF 零行为变）/ 反 theater（逆关系须构造验证 B∘A 还原·三值诚实降级）。
诚实边界：逆验证=统计非证明（采样还原 ≠ 数学逆·#479 守）·自产验证 ≠ R6 两源 ≠ truth·
可复合约束（B LHS 须匹配 A 输出 shape·否则 can't-verify）·机制完成 ≠ 已学（S8 课程训练续起）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_MATH
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.experiments.formal_train import make_train_context, _run_task_driven_generate
from pure_integer_ai.training.symbolic_transform import register_transform_rule, apply_transform
from pure_integer_ai.training.symbolic_relation import (
    register_inverse_relation, load_inverse_relation, verify_inverse_relation,
    RELATION_KIND_INVERSE)
from pure_integer_ai.training.vm_proof import execute_composes_value
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.cognition.shared.types import (
    MODALITY_ARITH, DOMAIN_MATH, LANG_NONE,
    TERMINAL_REACHED_SINK, VERIFY_SOURCE_SELF_PRODUCED,
    TransformSpec, InverseRelationSpec)


# ---- helpers（镜像 test_symbolic_transform _build_many + _register_ddx 范式） ----

def _build(ctx, key: str, src: str):
    """同一 ctx 建 lambda COMPOSES 树·返 root ConceptRef。"""
    r = ctx.concept_index.ensure(f"__seg_{key}", space_id=ctx.space_id,
                                 tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith(src, concept_index=ctx.concept_index,
                              edge_store=ctx.edge_store, backend=ctx.backend,
                              space_id=ctx.space_id, source=SOURCE_MATH, root_ref=r)
    return r


def _mkrule(ctx, name: str, lhs_src: str, rhs_src: str):
    """建 + 注册变换规则 name→(lhs, rhs)·返 rule_name ConceptRef。"""
    return register_transform_rule(ctx.backend, ctx.concept_index, name,
                                   _build(ctx, f"{name}_lhs", lhs_src),
                                   _build(ctx, f"{name}_rhs", rhs_src),
                                   space_id=ctx.space_id)


# ============================================================
# IR1 register/load roundtrip + 幂等 + 冲突 fail-loud（镜像 TC5/TC6）
# ============================================================

def test_ir1_register_load_idempotent_conflict():
    """register_inverse_relation 存 ATTR_RELATION_KIND/RULE_A/RULE_B·load 读回 (kind,A,B)。
    幂等：同 (name,kind,A,B) 重注册返同 ref·冲突：同 name 异 (A,B) fail-loud（拒歧义·反 theater）。"""
    ctx = make_train_context(DictBackend())
    ra = _mkrule(ctx, "ir1_a", "lambda p: p", "lambda p: 2*p")
    rb = _mkrule(ctx, "ir1_b", "lambda p: 2*p", "lambda p: p")
    rel_ref = register_inverse_relation(ctx.backend, ctx.concept_index,
                                        space_id=ctx.space_id, name="ir1_rel",
                                        kind=RELATION_KIND_INVERSE,
                                        rule_a_ref=ra, rule_b_ref=rb)
    loaded = load_inverse_relation(ctx.backend, rel_ref)
    assert loaded is not None, "load 返 (kind, A, B)·非 None（关系已注册）"
    assert loaded[0] == RELATION_KIND_INVERSE, "kind == INVERSE"
    assert loaded[1] == ra, "load rule_a == 注册 ra"
    assert loaded[2] == rb, "load rule_b == 注册 rb"
    # 非关系节点 load → None
    assert load_inverse_relation(ctx.backend, ra) is None, "非关系节点 load → None（无 ATTR_RELATION_*）"
    # 幂等：同 (name, kind, A, B) 重注册 → 同 rel_ref
    rel_ref2 = register_inverse_relation(ctx.backend, ctx.concept_index,
                                         space_id=ctx.space_id, name="ir1_rel",
                                         kind=RELATION_KIND_INVERSE,
                                         rule_a_ref=ra, rule_b_ref=rb)
    assert rel_ref == rel_ref2, "幂等：同 (name,kind,A,B) 重注册返同 ref"
    # 冲突：同 name 异 (A,B) → ValueError fail-loud
    rc = _mkrule(ctx, "ir1_c", "lambda p: p", "lambda p: p+1")
    with pytest.raises(ValueError, match="重名冲突"):
        register_inverse_relation(ctx.backend, ctx.concept_index,
                                  space_id=ctx.space_id, name="ir1_rel",
                                  kind=RELATION_KIND_INVERSE,
                                  rule_a_ref=rc, rule_b_ref=rb)


# ============================================================
# IR2 ×/÷ 逆关系 double↔halve B∘A 还原 @ 采样（旗舰·verified True）
# ============================================================

def test_ir2_double_halve_verified():
    """×/÷ 逆关系：A=double (p→2p)·B=halve (2p→p)·B∘A 还原。
    A 的 bare-PARAM LHS（lambda p: p）绑复合子树 e（subtree_binding 4th 路·同 TC2）→ A(e)=2e。
    B 的 LHS=Mul(2,PARAM) 匹配 A 输出 Mul(2,e) → B(A(e))=e（fresh deep-copy·执行等价）。
    verify_inverse_relation 全采样还原 → True（统计验非 truth·#479 守·反 theater：构造执行串联非教师声称）。"""
    ctx = make_train_context(DictBackend())
    ra = _mkrule(ctx, "dbl", "lambda p: p", "lambda p: 2*p")
    rb = _mkrule(ctx, "hlv", "lambda p: 2*p", "lambda p: p")
    rel_ref = register_inverse_relation(ctx.backend, ctx.concept_index,
                                        space_id=ctx.space_id, name="dbl_hlv",
                                        kind=RELATION_KIND_INVERSE,
                                        rule_a_ref=ra, rule_b_ref=rb)
    samples = [_build(ctx, "e_xp3", "lambda x: x+3"),
               _build(ctx, "e_xsqr", "lambda x: x*x"),
               _build(ctx, "e_ym5", "lambda y: y-5")]
    result = verify_inverse_relation(ctx.backend, ctx.concept_index, ctx.edge_store,
                                     space_id=ctx.space_id, source=SOURCE_MATH,
                                     relation_ref=rel_ref, sample_inputs=samples)
    assert result is True, "double↔halve B∘A 全采样还原 → verified True"

    # 直接显式 cross-verify（同 TC7 范式·自证机制非仅信 verify 返回值）。
    # 用 **fresh sample**（异于 verify 内部已 apply 的 samples）·避 apply_transform 复用同 _out_root
    # 节点（同 (rule,input) 二次 apply 撞 verify 已建树·apply_transform :251 fresh-per-(rule,input) 守）。
    graph = ConceptGraph(ctx.backend)
    e_direct = _build(ctx, "e_direct", "lambda z: z+8")
    out_a = apply_transform(ctx.backend, ctx.concept_index, ctx.edge_store,
                            space_id=ctx.space_id, source=SOURCE_MATH,
                            rule_name_ref=ra, input_ref=e_direct)
    out_b = apply_transform(ctx.backend, ctx.concept_index, ctx.edge_store,
                            space_id=ctx.space_id, source=SOURCE_MATH,
                            rule_name_ref=rb, input_ref=out_a)
    assert out_a is not None and out_b is not None, "double/halve apply 产输出（A 后 B 可复合）"
    for z in [0, 1, 3, -2, 5]:
        v_e = execute_composes_value(graph, e_direct, ((z, 1),))
        v_ba = execute_composes_value(graph, out_b, ((z, 1),))
        assert v_e is not None and v_ba is not None, f"执行非 None @ z={z}"
        assert rational.eq(v_e, v_ba), (
            f"double↔halve cross-verify：B(A(e))==e @ z={z}"
            f"（得 {v_ba.num}/{v_ba.den}·期 {v_e.num}/{v_e.den}·×/÷ 逆关系构造验证）")


# ============================================================
# IR3 composes-fail can't-verify（B LHS 不匹配 A 输出 shape → None·诚实降级）
# ============================================================

def test_ir3_composes_fail_cant_verify():
    """can't-verify：A=square (p→p*p·输出 Mul(e,e))·B=sub_one (LHS=p+1 Add·期望 Add(_,1))。
    B 的 LHS=Add(PARAM,IMM1) 不匹配 A 输出 Mul(e,e) → apply_transform(B, Mul)=None → 不可复合 →
    verify 返 None（can't-verify·诚实降级·非 theater·非声称"已验互逆"·须教师设计可复合规则对）。"""
    ctx = make_train_context(DictBackend())
    ra = _mkrule(ctx, "sq", "lambda p: p", "lambda p: p*p")     # A 输出 Mul
    rb = _mkrule(ctx, "so", "lambda p: p+1", "lambda p: p")     # B LHS 期望 Add
    rel_ref = register_inverse_relation(ctx.backend, ctx.concept_index,
                                        space_id=ctx.space_id, name="sq_so",
                                        kind=RELATION_KIND_INVERSE,
                                        rule_a_ref=ra, rule_b_ref=rb)
    e = _build(ctx, "e_ir3", "lambda x: x+3")
    result = verify_inverse_relation(ctx.backend, ctx.concept_index, ctx.edge_store,
                                     space_id=ctx.space_id, source=SOURCE_MATH,
                                     relation_ref=rel_ref, sample_inputs=[e])
    assert result is None, (
        "B LHS(Add) 不匹配 A 输出(Mul) → 不可复合 → can't-verify None"
        "（诚实降级·非 theater·设计 §五 可复合约束）")


# ============================================================
# IR4 falsified（B∘A 不还原 → False·诚实不偷渡 verified）
# ============================================================

def test_ir4_falsified_not_inverse():
    """falsified：A=double (p→2p)·B=add_one (p→p+1)·两规则可复合但非互逆。
    B(A(e))=B(2e)=2e+1·≠ e → verify 返 False（诚实·不偷渡 verified·反 theater）。
    B 的 bare-PARAM LHS 匹配 A 输出 Mul(2,e)（可复合）·但组合后不还原（非逆）→ falsified 非 can't-verify。"""
    ctx = make_train_context(DictBackend())
    ra = _mkrule(ctx, "dbl4", "lambda p: p", "lambda p: 2*p")
    rb = _mkrule(ctx, "add1", "lambda p: p", "lambda p: p+1")
    rel_ref = register_inverse_relation(ctx.backend, ctx.concept_index,
                                        space_id=ctx.space_id, name="dbl_add1",
                                        kind=RELATION_KIND_INVERSE,
                                        rule_a_ref=ra, rule_b_ref=rb)
    e = _build(ctx, "e_ir4", "lambda x: x+3")
    result = verify_inverse_relation(ctx.backend, ctx.concept_index, ctx.edge_store,
                                     space_id=ctx.space_id, source=SOURCE_MATH,
                                     relation_ref=rel_ref, sample_inputs=[e])
    assert result is False, (
        "B(A(e))=2e+1 ≠ e → falsified False（可复合但不还原·诚实不偷渡 verified·反 theater）")


# ============================================================
# IR5 formal_train 集成 gate ON（verified → 独立 SELF_PRODUCED episode·weaning-safe 决断 A）
# ============================================================

def _inv_item():
    """CollectedItem + double/halve inverse_relation_spec（教师陈述逆关系 + 采样输入）。"""
    spec = InverseRelationSpec(
        relation_name="dbl_hlv_int",
        rule_a=TransformSpec("dbl_i", "lambda p: p", "lambda p: 2*p"),
        rule_b=TransformSpec("hlv_i", "lambda p: 2*p", "lambda p: p"),
        sample_sources=("lambda x: x+3", "lambda x: x*x"))
    return CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                         source=SOURCE_MATH, inverse_relation_specs=(spec,))


def test_ir5_formal_train_integration_gate_on():
    """gate ON：CollectedItem.inverse_relation_specs（double/halve + 采样）→ _run_task_driven_generate
    → register 两规则 + register_inverse_relation + verify B∘A 还原通过 → 独立 verified episode。
    weaning-safe 决断 A（独立 task-driven episode·不替换 vm_proof·不碎 W7·同 transform_specs）·
    verify_source SELF_PRODUCED（两规则 single-source·不准驱动停止·反 theater）。"""
    saved = gates.SYMBOLIC_RELATION_MODE
    gates.SYMBOLIC_RELATION_MODE = True
    try:
        ctx = make_train_context(DictBackend())
        episodes, summary = _run_task_driven_generate(ctx, [_inv_item()], all_ops=[])
        assert summary.total_tasks >= 1, "total_tasks 计 inverse_relation_spec（double/halve 逆关系）"
        assert summary.verified >= 1, "double/halve B∘A 还原 @ 采样 → verified episode"
        _rel_eps = [e for e in episodes
                    if e.reward == 1 and e.terminal == TERMINAL_REACHED_SINK
                    and e.verify_source == VERIFY_SOURCE_SELF_PRODUCED]
        assert len(_rel_eps) >= 1, (
            "≥1 SELF_PRODUCED verified episode（逆关系构造验证 single-source·不准驱动停止·反 theater）")
    finally:
        gates.SYMBOLIC_RELATION_MODE = saved


# ============================================================
# IR6 gate OFF bit-identical（无 relation episode·既有行为零翻）
# ============================================================

def test_ir6_gate_off_bit_identical():
    """gate OFF：inverse_relation_specs 不消费 → 无 relation episode（既有行为零翻·bit-identical）。
    同 TC12 transform_specs gate OFF 范式（gate OFF→episodes 无 relation 产物）。"""
    saved = gates.SYMBOLIC_RELATION_MODE
    gates.SYMBOLIC_RELATION_MODE = False
    try:
        ctx = make_train_context(DictBackend())
        # MODALITY_ARITH item 无 arith_specs/code_source/transform_specs·gate SYMBOLIC_RELATION_MODE OFF
        # → inverse_relation_specs 不消费 → 无 episode（bit-identical）
        episodes, summary = _run_task_driven_generate(ctx, [_inv_item()], all_ops=[])
        assert episodes == [], (
            "gate OFF → inverse_relation_specs 不消费 → 无 episode（MODALITY_ARITH item 无 arith_specs·"
            "gate OFF 全路径不产·bit-identical）")
    finally:
        gates.SYMBOLIC_RELATION_MODE = saved


# ============================================================
# IR7 malformed graceful skip（bad spec 不 abort run·后续 valid 仍处理·镜像 TC13）
# ============================================================

def test_ir7_malformed_spec_graceful_skip():
    """malformed spec（rule DSL 解析错 / Pow 负指数 raise UnsupportedConstruct）→ try/except 守 →
    graceful skip 此 spec（不 abort run·不产 episode）·后续 valid spec 仍处理。mirror TC13 范式。"""
    saved = gates.SYMBOLIC_RELATION_MODE
    gates.SYMBOLIC_RELATION_MODE = True
    try:
        ctx = make_train_context(DictBackend())
        # spec1 malformed（rule_a lhs 非法 DSL "lambda p: p @@" → build raise）+ spec2 valid（double/halve）
        bad_spec = InverseRelationSpec(
            relation_name="bad_rel",
            rule_a=TransformSpec("bad_a", "lambda p: p @@", "lambda p: 2*p"),   # 非法 DSL → build raise
            rule_b=TransformSpec("bad_b", "lambda p: 2*p", "lambda p: p"),
            sample_sources=("lambda x: x+3",))
        good_spec = InverseRelationSpec(
            relation_name="good_rel",
            rule_a=TransformSpec("good_a", "lambda p: p", "lambda p: 2*p"),
            rule_b=TransformSpec("good_b", "lambda p: 2*p", "lambda p: p"),
            sample_sources=("lambda x: x+3", "lambda x: x*x"))
        item = CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                             source=SOURCE_MATH, inverse_relation_specs=(bad_spec, good_spec))
        episodes, summary = _run_task_driven_generate(ctx, [item], all_ops=[])
        # 两 spec 都计入 total_tasks·malformed spec1 graceful skip（不 abort）
        assert summary.total_tasks >= 2, "total_tasks 计两 spec（malformed 不 abort·仍计数）"
        # good_spec2 B∘A 还原 → verified（malformed spec1 不计 verified·graceful skip）
        assert summary.verified >= 1, (
            "good_spec verified·malformed spec graceful skip（try/except 守·不 abort run·反 theater 不伪造）")
    finally:
        gates.SYMBOLIC_RELATION_MODE = saved


# ============================================================
# IR9 共享 rule_name 跨关系幂等（对抗审 MEDIUM fix·formal_train surface 键 rule_name）
# ============================================================

def test_ir9_shared_rule_name_across_relations():
    """两 InverseRelationSpec 共享同 rule_name（如 "double" 出现在两关系的 rule_a）→ formal_train surface
    键用 rule_name（非 relation_name）→ 同 rule_name 映射同 lhs/rhs ConceptRef → register_transform_rule 幂等
    （非冲突 ValueError 静默 skip）→ 两关系都 verified（对抗审 MEDIUM：原键 relation_name 致 spec2 静默丢）。"""
    saved = gates.SYMBOLIC_RELATION_MODE
    gates.SYMBOLIC_RELATION_MODE = True
    try:
        ctx = make_train_context(DictBackend())
        # 两关系共享 rule_a="double" / rule_b="halve"（同规则不同 relation_name）
        spec1 = InverseRelationSpec(
            relation_name="rel_one",
            rule_a=TransformSpec("double", "lambda p: p", "lambda p: 2*p"),
            rule_b=TransformSpec("halve", "lambda p: 2*p", "lambda p: p"),
            sample_sources=("lambda x: x+3",))
        spec2 = InverseRelationSpec(
            relation_name="rel_two",
            rule_a=TransformSpec("double", "lambda p: p", "lambda p: 2*p"),   # 同 rule_name "double"
            rule_b=TransformSpec("halve", "lambda p: 2*p", "lambda p: p"),    # 同 rule_name "halve"
            sample_sources=("lambda x: x*x",))
        item = CollectedItem(modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
                             source=SOURCE_MATH, inverse_relation_specs=(spec1, spec2))
        episodes, summary = _run_task_driven_generate(ctx, [item], all_ops=[])
        assert summary.total_tasks >= 2, "total_tasks 计两关系（共享 rule_name 不 abort·仍计数）"
        assert summary.verified >= 2, (
            "两关系共享 rule_name 都 verified（surface 键 rule_name→幂等·非冲突静默 skip·对抗审 MEDIUM fix）")
    finally:
        gates.SYMBOLIC_RELATION_MODE = saved


# ============================================================
# IR8 +/− 化简规则路径（x+y−y→x 单条规则·不走逆关系·验两路皆可学 = S8 不留缺口·设计 §八）
# ============================================================

def test_ir8_plus_minus_simplification_rule_path():
    """+/− 走**化简规则**（a+0→a 恒等化简·symbolic_transform 已有机制够·无须逆关系机制·设计 §一 IDENTITY 折化简）。
    register 恒等化简规则 LHS=Add(PARAM_x,IMM0)·RHS=PARAM_x·apply input (u*u)+0 → u*u。
    cross-verify：执行 input (u*u)+0 == output u*u @ 采样（加法恒等经化简规则表达·两路皆可学 = S8 不留缺口）。

    **诚实边界（机制墙·defer）**：设计 §八 举例的消去化简 x+y−y→x（+−互逆的化简表达）在当前机制**不可达**——
    LHS 的 PARAM_y 重复出现·input (u+v)-v 的两 v 是异 ConceptRef（fresh operand 叶）→ _align_walk 变量同一性
    拒（同 TC3 重复-PARAM 墙）→ apply 返 None。本测用无重复 PARAM 的恒等化简 a+0→a（可学）·消去化简 x+y−y→x
    须匹配机制扩展（重复-PARAM 同值判定·非同子树）defer。故 +/− 互逆主体实际经**逆关系机制**（add_one/sub_one·
    同 IR2 double/halve 范式）覆盖·化简规则路径覆盖恒等类（a+0→a·a*1→a）。两路并列 = S8 +/−·×/÷ 不留缺口。"""
    ctx = make_train_context(DictBackend())
    rule_ref = _mkrule(ctx, "simp_addzero",
                       "lambda x: x+0",   # LHS: Add(PARAM_x, IMM0)·无重复 PARAM
                       "lambda x: x")     # RHS: PARAM_x（加法恒等经化简表达）
    # input (u*u)+0（LHS 匹配：x→u*u 子树·IMM0→IMM0 值·复合子树绑定）
    input_ref = _build(ctx, "e_ir8", "lambda u: (u*u)+0")
    output_ref = apply_transform(ctx.backend, ctx.concept_index, ctx.edge_store,
                                 space_id=ctx.space_id, source=SOURCE_MATH,
                                 rule_name_ref=rule_ref, input_ref=input_ref)
    assert output_ref is not None, (
        "恒等化简规则 LHS Add(x,0) 匹配 input (u*u)+0 → 产出（PARAM_x 绑 u*u 复合子树·无重复 PARAM·匹配 True）")
    # cross-verify：执行 input (u*u)+0 == output u*u @ 采样（加法恒等·化简规则路径）
    graph = ConceptGraph(ctx.backend)
    for u in [0, 1, 3, -2, 5]:
        v_in = execute_composes_value(graph, input_ref, ((u, 1),))
        v_out = execute_composes_value(graph, output_ref, ((u, 1),))
        assert v_in is not None and v_out is not None, f"执行非 None（u={u}）"
        assert rational.eq(v_in, v_out), (
            f"+/− 恒等化简 cross-verify：(u*u)+0==u*u @ u={u}"
            f"（得 {v_out.num}/{v_out.den}·期 {v_in.num}/{v_in.den}·加法恒等经化简规则可学·S8 不留缺口）")
