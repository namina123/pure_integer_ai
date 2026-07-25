"""training.symbolic_transform — 符号变换规则存储 + 应用（符号数学扩展 Phase 2·纯替换子集）。

权威设计 = doc/重来_符号数学能力扩展设计_2026-07-15 §八-bis。

变换规则 = (LHS 模式, RHS 模板) + PARAM 槽对齐（make_variable(slot)·教师陈述模板 lambda 同 arg 序保证）。
- register_transform_rule: 教师陈述规则（name→LHS+RHS struct_ref·composes_attr ATTR_TRANSFORM_LHS/RHS·
  范式同 register_arith_operator 的 ATTR_OPERATOR_DEF）。
- apply_transform: LHS 匹配 input（_align_walk subtree_binding 绑子树 + value_binding 绑值）→
  RHS β-替换（_deep_copy_subtree·PARAM 槽→子树/fresh IMM 叶）→ 输出表达式。

**Phase 2 scope（纯替换·无 Pow / 无值算术）**：PARAM 绑子树（subtree_binding）+ 绑 IMM 值（value_binding）·
β-替换 RHS（_deep_copy_subtree 复用零改）。测试分配律 a*(b+c)→a*b+a*c（PARAM a/b 绑子树·c 绑值·a 重复 β-fresh-copy）。
**Phase 2b 待加**（doc §八-bis.3/4）：OPCODE_POW_PATTERN + Pow lower（concrete 整数指数→MUL）+ 值算术 VM 求值
（d/dx n-1·PARAM 值参与算术构造新子树·部分求值 walker）。Phase 2 不涉·诚实 scope。

铁律：纯整数（全 int/ConceptRef·无浮点）/ bit-identical（apply 纯函数·gate 在 formal_train 集成层 Phase 3·
本模块无 gate）/ 单向依赖（L7 training→L5 cognition _align_walk + L7 arith_observe _deep_copy_subtree·向下·
不向上）/ 反 theater（LHS 不匹配→None 诚实 skip·非 stub·非声称"已学"）/ 不写死（规则是数据·教师陈述模板·
非硬编码 d/dx→formula·PARAM=自由度涌现·同 discover_skeleton）/ 闭项守（fail_on_external=True·RHS 须全 PARAM 绑定）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.numeric.symbol_domain import (
    make_variable, index_of,
    OPCODE_ADD, OPCODE_SUB, OPCODE_MUL, OPCODE_DIV, OPCODE_POW_PATTERN, OPCODE_NOP,
)
from pure_integer_ai.storage.composes_attr import (
    record_composes_attr, read_composes_attrs,
    ATTR_TRANSFORM_LHS, ATTR_TRANSFORM_RHS,
    ATTR_OPERATOR, ATTR_OPERAND, ATTR_IMMEDIATE,
)
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.cognition.shared.types import ConceptRef


def _is_imm(backend, ref: ConceptRef) -> bool:
    """ref 是 IMMEDIATE 叶（ATTR_IMMEDIATE 在）·常量折叠判定用。"""
    return ATTR_IMMEDIATE in read_composes_attrs(backend, ref)


def _lower_pow(backend, builder, base_ref: ConceptRef, exp_val: tuple[int, int]) -> ConceptRef:
    """Pow(base, concrete exp) → MUL chain（lower·doc §八-bis.4）。

    exp 须非负整数（exp_den==1·exp_num>=0）·否则 fail-loud（负/分指数 defer·C7 墙外）。
    exp=0 → IMM(1)（VM 约定 a^0=1）·exp=1 → base·exp>=2 → 左结合 MUL chain（DAG 共享 base·执行 OK·
    算术 base 无 STORE 副作用·复用 _build_pow:422-441 范式）。
    """
    from pure_integer_ai.cognition.understanding.arith_observe import UnsupportedConstruct
    exp_num, exp_den = exp_val
    if exp_den != 1 or exp_num < 0:
        raise UnsupportedConstruct(
            f"Pow lower 须非负整数指数·得 {exp_num}/{exp_den}（负/分指数 defer）")
    k = exp_num
    if k == 0:
        return builder._imm_leaf(1, 1)
    if k == 1:
        return base_ref
    result = base_ref
    for _ in range(k - 1):
        mul_node = builder._new_node("BINOP")
        record_composes_attr(backend, ref=mul_node, kind=ATTR_OPERATOR, int_a=OPCODE_MUL)
        builder._edge(mul_node, result, 0)
        builder._edge(mul_node, base_ref, 1)   # DAG 共享 base（算术无 STORE·执行 OK）
        result = mul_node
    return result


def _rebuild_op(backend, builder, opcode: int, kids: list) -> ConceptRef:
    """重建算子节点（opcode + kids·常量折叠未命中时保留结构·如混合算术 Mul(IMM,子树)）。"""
    node = builder._new_node("POW" if opcode == OPCODE_POW_PATTERN else "BINOP")
    record_composes_attr(backend, ref=node, kind=ATTR_OPERATOR, int_a=opcode)
    for i, k in enumerate(kids):
        builder._edge(node, k, i)
    return node


def _eval_rhs(backend, builder, *, rhs_root: ConceptRef,
              subtree_binding: dict, value_binding: dict) -> ConceptRef:
    """递归部分求值 RHS（doc §八-bis.3·Phase 2b 核心·单 pass 替换+折叠+lower）。

    每节点：
      - PARAM operand 叶 → subtree_binding→_deep_copy_subtree 子树·value_binding→fresh IMM 叶·未绑→fail-loud（闭项）。
      - IMM 叶 → fresh IMM 拷贝。
      - 无属性叶（concept/token）→ _deep_copy_subtree 拷贝。
      - Pow（OPCODE_POW_PATTERN）→ 递归求值 base/exp·exp 成 IMM（concrete）→ _lower_pow MUL·exp 变量→重建 Pow。
      - 算术 opcode（ADD/SUB/MUL/DIV）→ 递归求值子·两子皆 IMM → rational 求值→fresh IMM（常量折叠·如 SUB(n,1)→n-1）·
        混合 → _rebuild_op 保留结构。
      - 其他 opcode → _rebuild_op 保留。

    返求值后 ConceptRef（新树·PARAM 全替换·值算术折叠·Pow concrete lower·可 VM 执行）。
    """
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    from pure_integer_ai.cognition.understanding.arith_observe import UnsupportedConstruct
    graph = ConceptGraph(backend)
    children_of, _, _, _, _ = graph.read_composes_tree(rhs_root)
    attrs = read_composes_attrs(backend, rhs_root)

    # PARAM operand 叶 → 替换
    if ATTR_OPERAND in attrs:
        slot = index_of(attrs[ATTR_OPERAND][0])
        if slot in subtree_binding:
            return builder._deep_copy_subtree(subtree_binding[slot],
                                               param_subst=None, fail_on_external=False)
        if slot in value_binding:
            num, den = value_binding[slot]
            return builder._imm_leaf(num, den)
        raise UnsupportedConstruct(
            f"RHS PARAM slot {slot} 未绑定（闭项违例·LHS 未匹配此 PARAM→反 theater 防半绑残缺输出）")

    # IMM 叶 → fresh 拷贝
    if ATTR_IMMEDIATE in attrs:
        num, den = attrs[ATTR_IMMEDIATE]
        return builder._imm_leaf(num, den)

    # 无属性叶（concept/token）→ deep copy
    if not attrs:
        return builder._deep_copy_subtree(rhs_root, param_subst=None, fail_on_external=False)

    # 算子节点
    if ATTR_OPERATOR in attrs:
        op = attrs[ATTR_OPERATOR][0]
        kids = children_of.get(rhs_root, [])
        eval_kids = [_eval_rhs(backend, builder, rhs_root=k,
                               subtree_binding=subtree_binding, value_binding=value_binding)
                     for k in kids]

        # exp 为具体值时，把 Pow 降为可执行结构。
        if op == OPCODE_POW_PATTERN:
            base_ref, exp_ref = eval_kids[0], eval_kids[1]
            exp_attrs = read_composes_attrs(backend, exp_ref)
            if ATTR_IMMEDIATE in exp_attrs:
                return _lower_pow(backend, builder, base_ref, exp_attrs[ATTR_IMMEDIATE])
            return _rebuild_op(backend, builder, OPCODE_POW_PATTERN, [base_ref, exp_ref])

        # 算术 opcode → 折叠 if 两子皆 IMM
        if op in (OPCODE_ADD, OPCODE_SUB, OPCODE_MUL, OPCODE_DIV) and len(eval_kids) == 2:
            if _is_imm(backend, eval_kids[0]) and _is_imm(backend, eval_kids[1]):
                v0 = read_composes_attrs(backend, eval_kids[0])[ATTR_IMMEDIATE]
                v1 = read_composes_attrs(backend, eval_kids[1])[ATTR_IMMEDIATE]
                r0, r1 = rational.make(v0[0], v0[1]), rational.make(v1[0], v1[1])
                if op == OPCODE_ADD:
                    res = rational.add(r0, r1)
                elif op == OPCODE_SUB:
                    res = rational.sub(r0, r1)
                elif op == OPCODE_MUL:
                    res = rational.mul(r0, r1)
                else:   # OPCODE_DIV
                    if r1.num == 0:
                        raise UnsupportedConstruct("常量折叠 DIV by zero")
                    res = rational.div(r0, r1)
                return builder._imm_leaf(res.num, res.den)
            return _rebuild_op(backend, builder, op, eval_kids)   # 混合 → 保留

        # 其他 opcode（EQ/LT/GT/NOP/...）→ 重建保留
        return _rebuild_op(backend, builder, op, eval_kids)

    return builder._deep_copy_subtree(rhs_root, param_subst=None, fail_on_external=False)


def register_transform_rule(backend, concept_index, name: str,
                            lhs_ref: ConceptRef, rhs_ref: ConceptRef, *,
                            space_id: int) -> ConceptRef:
    """注册符号变换规则 name→(LHS, RHS) struct_ref（教师陈述模板·数据驱动·非硬编码）。

    在 lhs_ref 同 space 建 name 概念点·挂 ATTR_TRANSFORM_LHS（lhs_ref）+ ATTR_TRANSFORM_RHS（rhs_ref）。
    镜像 register_arith_operator 范式（ATTR_OPERATOR_DEF+ATTR_ARITY·composes_attr 存 struct_ref 作 (int_a=sid,int_b=lid)）。
    幂等：同 name 同 (lhs,rhs) 重注册不变·重映射不同 (lhs,rhs) fail-loud（拒歧义）。

    返 name 节点 ConceptRef（规则身份·apply_transform 查表键）。
    """
    assert_int(space_id, _where="register_transform_rule.space_id")
    from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
    name_ref = concept_index.ensure(name, space_id=space_id,
                                    tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    existing = read_composes_attrs(backend, name_ref)
    prev_lhs = existing.get(ATTR_TRANSFORM_LHS)
    if prev_lhs is not None:
        prev_rhs = existing.get(ATTR_TRANSFORM_RHS)
        if prev_lhs != (lhs_ref[0], lhs_ref[1]) or prev_rhs != (rhs_ref[0], rhs_ref[1]):
            raise ValueError(
                f"变换规则重名冲突: {name!r} 已映射→LHS{prev_lhs}/RHS{prev_rhs}"
                f"·试图重映射→LHS{(lhs_ref[0], lhs_ref[1])}/RHS{(rhs_ref[0], rhs_ref[1])}"
                f"（fail-loud 拒歧义·同义重注册须同 (lhs,rhs)）")
        return name_ref   # 幂等：同 (name, lhs, rhs) 重注册不变
    record_composes_attr(backend, ref=name_ref, kind=ATTR_TRANSFORM_LHS,
                         int_a=lhs_ref[0], int_b=lhs_ref[1])
    record_composes_attr(backend, ref=name_ref, kind=ATTR_TRANSFORM_RHS,
                         int_a=rhs_ref[0], int_b=rhs_ref[1])
    return name_ref


def load_transform_rule(backend, name_ref: ConceptRef):
    """读变换规则→(lhs_ref, rhs_ref) | None（非规则/无 ATTR→None·caller 判）。"""
    attrs = read_composes_attrs(backend, name_ref)
    lhs_attr = attrs.get(ATTR_TRANSFORM_LHS)
    rhs_attr = attrs.get(ATTR_TRANSFORM_RHS)
    if lhs_attr is None or rhs_attr is None:
        return None
    return ((lhs_attr[0], lhs_attr[1]), (rhs_attr[0], rhs_attr[1]))


def apply_transform(backend, concept_index, edge_store, *, space_id: int,
                    source: int, rule_name_ref: ConceptRef,
                    input_ref: ConceptRef):
    """应用符号变换规则：LHS 匹配 input → RHS β-替换 → 输出表达式。

    返 output ConceptRef | None（LHS 不匹配 / 非规则 / 无 PARAM 绑定→None·诚实 skip·非 theater）。

    机制（doc §八-bis.3 部分求值·Phase 2 纯替换子集）：
      1. _align_walk(LHS, input, subtree_binding={}, value_binding={}) → 绑 PARAM 槽
         （复合子树→subtree_binding·IMM 值→value_binding·operand/concept 既有路 Phase 2 不处理→无绑→闭项守拒）。
      2. param_subst = {make_variable(slot): subtree | fresh IMM 叶}
         （subtree_binding→子树 ConceptRef·value_binding→_imm_leaf 造 fresh IMM 叶）。
      3. _deep_copy_subtree(RHS, param_subst, fail_on_external=True) → β-替换 RHS PARAM 槽→绑定
         （复用 arith_observe._deep_copy_subtree·零改·PARAM operand 叶→fresh 子树拷贝·重复 PARAM 各 fresh 无别名）。

    Phase 2b 待加：Pow lower + 值算术 VM 求值（d/dx n-1·PARAM 值参与算术构造新子树·部分求值 walker）。
    """
    assert_int(space_id, source, _where="apply_transform")
    from pure_integer_ai.cognition.process.structure_discover import _align_walk
    from pure_integer_ai.cognition.understanding.arith_observe import _ArithBuilder
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph

    rule = load_transform_rule(backend, rule_name_ref)
    if rule is None:
        return None   # 非变换规则
    lhs_ref, rhs_ref = rule

    graph = ConceptGraph(backend)
    sk_children, _, _, _, _ = graph.read_composes_tree(lhs_ref)
    in_children, _, _, _, _ = graph.read_composes_tree(input_ref)

    value_binding: dict[int, tuple[int, int]] = {}
    subtree_binding: dict[int, ConceptRef] = {}
    if not _align_walk(backend, sk_children, in_children, lhs_ref, input_ref,
                       value_binding, {}, {}, 0, ancestor_map=None,
                       subtree_binding=subtree_binding):
        return None   # LHS 不匹配 input → 规则不适用（诚实·非 theater）

    if not subtree_binding and not value_binding:
        return None   # 无 PARAM 绑定到 input 内容（LHS 无 PARAM 或全 operand/concept·Phase 2 不处理）→ 无变换意义

    # 部分求值 RHS（_eval_rhs·单 pass 替换+常量折叠+Pow lower·doc §八-bis.3·Phase 2b 核心）：
    # subtree_binding→子树 β-替换·value_binding→IMM 叶·值算术折叠（如 SUB(n,1)→n-1）·Pow concrete→MUL lower。
    # 闭项守：RHS 全 PARAM 须绑定·未绑→_eval_rhs PARAM 分支 UnsupportedConstruct fail-loud（反 theater 防半绑残缺）。
    # **fresh output root**（unique per rule+input）：_ArithBuilder._new_node surface 用 root_lid+seq·若复用 rhs_ref
    # 则多次 apply 同规则时 seq restart→surface 碰撞→ensure dedup 复用前次节点→ATTR_OPERAND 幂等不覆盖→
    # 残留前次 sid→execute 取错值（多 held-out 序贯碰撞·对抗审级 bug）。fresh _out_root per (rule,input) 避碰·确定性。
    from pure_integer_ai.storage.node_store import NODE_CONCEPT, TIER_PRIMARY
    _out_root = concept_index.ensure(
        f"__xform_apply_{rule_name_ref[0]}_{rule_name_ref[1]}_{input_ref[0]}_{input_ref[1]}",
        space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    builder = _ArithBuilder(concept_index=concept_index, edge_store=edge_store,
                            backend=backend, space_id=space_id, source=source,
                            root_ref=_out_root)
    return _eval_rhs(backend, builder, rhs_root=rhs_ref,
                     subtree_binding=subtree_binding, value_binding=value_binding)


# ---- 自归纳（single-pair symbolic induction·doc §八-bis.2 简单重排自归纳·2026-07-15） ----

def _collect_operand_sids(backend, root, children_of) -> list[int]:
    """DFS 前序收集 operand 叶 sid（ATTR_OPERAND·阅读序）。caller 去重→slot 映射。"""
    attrs = read_composes_attrs(backend, root)
    if ATTR_OPERAND in attrs:
        return [attrs[ATTR_OPERAND][0]]
    if ATTR_IMMEDIATE in attrs:
        return []
    if ATTR_OPERATOR in attrs:
        sids: list[int] = []
        for child in children_of.get(root, []):
            sids.extend(_collect_operand_sids(backend, child, children_of))
        return sids
    return []   # ctrl/store/concept 叶无 operand


def _clone_induced(backend, builder, root, children_of, sid_to_slot):
    """克隆 COMPOSES 子树为变换模板：operand 叶→PARAM(slot)·immediate→fixed clone·operator→递归子。

    PARAM 转换让归纳出的 LHS/RHS 泛化（operand=变量→通配 PARAM·匹配时绑子树/值）。
    operand sid 须在 sid_to_slot（output 含 input 未出现的 operand→fail-loud·守闭项一致性）。
    """
    from pure_integer_ai.cognition.understanding.arith_observe import UnsupportedConstruct
    attrs = read_composes_attrs(backend, root)
    if ATTR_OPERAND in attrs:
        sid = attrs[ATTR_OPERAND][0]
        if sid not in sid_to_slot:
            raise UnsupportedConstruct(
                f"induce: operand sid {sid} 未在 input（output 含 input 未出现变量·闭项违例）")
        return builder._var_leaf(make_variable(sid_to_slot[sid]))   # PARAM operand 叶
    if ATTR_IMMEDIATE in attrs:
        num, den = attrs[ATTR_IMMEDIATE]
        return builder._imm_leaf(num, den)   # 立即数叶→fixed clone（常量·非 PARAM）
    if ATTR_OPERATOR in attrs:
        op = attrs[ATTR_OPERATOR][0]
        node = builder._new_node("IND")
        record_composes_attr(backend, ref=node, kind=ATTR_OPERATOR, int_a=op)
        for i, child in enumerate(children_of.get(root, [])):
            builder._edge(node, _clone_induced(backend, builder, child, children_of, sid_to_slot), i)
        return node
    # ctrl/store/概念叶 → deep copy（simple induction scope·arith 表达式不涉·防御保留）
    return builder._deep_copy_subtree(root, param_subst=None, fail_on_external=False)


def induce_transform_rule(backend, concept_index, edge_store, *, space_id: int,
                          source: int, name: str,
                          input_ref: ConceptRef, output_ref: ConceptRef) -> ConceptRef:
    """自归纳（single-pair symbolic induction·doc §八-bis.2）：从一符号例 (input, output) 归纳变换规则。

    机制：input/output 是符号表达式（lambda args = 变量）→ operand 变量成 PARAM 槽（通配·匹配时绑子树/值）
    ·immediate 成 fixed（常量）·input clone→LHS 模式·output clone→RHS 模板·PARAM 按 operand sid 对齐。
    单对符号归纳直接产**通用规则**（PARAM 泛化·如 Pow(b,n)→n*Pow(b,n-1) 从一符号例归纳·应用 Pow(y,5)→5y⁴）。

    比 teacher-stated（TransformSpec 分述 LHS+RHS）更纯学习模式：教师给一符号例·系统归纳规则。
    **诚实边界**：single-pair（一例归纳）·非 multi-pair 反统一（跨例泛化规则结构·hard defer）。
    单对归纳泛化力来自 PARAM（符号例本身已用变量表达通则·系统提取为 PARAM 规则）。

    返 rule_name_ref（register_transform_rule·同 teacher-stated 路径存）。
    """
    assert_int(space_id, source, _where="induce_transform_rule")
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    from pure_integer_ai.cognition.understanding.arith_observe import _ArithBuilder
    from pure_integer_ai.storage.node_store import NODE_CONCEPT, TIER_PRIMARY as _IND_TIER
    graph = ConceptGraph(backend)
    _TIER = _IND_TIER
    in_children, _, _, _, _ = graph.read_composes_tree(input_ref)
    out_children, _, _, _, _ = graph.read_composes_tree(output_ref)
    # 收集 input operand sid（DFS 阅读序·distinct 首遇→slot）
    sid_to_slot: dict[int, int] = {}
    for sid in _collect_operand_sids(backend, input_ref, in_children):
        if sid not in sid_to_slot:
            sid_to_slot[sid] = len(sid_to_slot)
    # build LHS（clone input·operand→PARAM）+ RHS（clone output·operand→PARAM·同 sid_to_slot）
    lhs_root = concept_index.ensure(f"__ind_lhs_{name}", space_id=space_id,
                                    tier=_TIER, node_type=NODE_CONCEPT)
    rhs_root = concept_index.ensure(f"__ind_rhs_{name}", space_id=space_id,
                                    tier=_TIER, node_type=NODE_CONCEPT)
    builder = _ArithBuilder(concept_index=concept_index, edge_store=edge_store,
                            backend=backend, space_id=space_id, source=source, root_ref=lhs_root)
    record_composes_attr(backend, ref=lhs_root, kind=ATTR_OPERATOR, int_a=OPCODE_NOP)
    _in_body = in_children[input_ref][0] if in_children.get(input_ref) else input_ref
    builder._edge(lhs_root, _clone_induced(backend, builder, _in_body, in_children, sid_to_slot), 0)
    record_composes_attr(backend, ref=rhs_root, kind=ATTR_OPERATOR, int_a=OPCODE_NOP)
    _out_body = out_children[output_ref][0] if out_children.get(output_ref) else output_ref
    builder._edge(rhs_root, _clone_induced(backend, builder, _out_body, out_children, sid_to_slot), 0)
    return register_transform_rule(backend, concept_index, name, lhs_root, rhs_root,
                                   space_id=space_id)
