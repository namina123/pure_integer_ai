"""training.symbolic_relation — 符号间运算关联（S8 逆关系·doc/重来_S8符号间关联机制设计_2026-07-15）。

权威设计 = doc/重来_S8符号间关联机制设计_2026-07-15。

符号数学扩展（symbolic_transform）让系统学**单条变换规则**。本模块在其上加**规则间关联**层——
逆关系 = 两条变换规则 (A, B) + 构造性验证 B∘A = identity @ 采样（反 theater 心脏）。

- register_inverse_relation: 教师陈述逆关系（name→KIND+RULE_A+RULE_B·composes_attr ATTR_RELATION_*·
  范式同 register_transform_rule ATTR_TRANSFORM_LHS/RHS 存 struct_ref + ATTR_PROPOSITION marker 范式）。
- verify_inverse_relation: 构造验证 B∘A 还原（apply A→apply B→执行比对 @ 采样·三值 True/False/None）。
  verified=True（全采样还原）/ falsified=False（任一不还原·诚实不偷渡 verified）/ can't-verify=None
  （不可复合·B 的 LHS 不匹配 A 的输出 shape·或规则非变换规则·诚实降级·同刀C 三值逻辑）。
- load_inverse_relation: 读逆关系→(kind, rule_a_ref, rule_b_ref) | None。

**Phase 1 scope（INVERSE only·覆盖 +/−·×/÷·d/dx↔∫ 逆关系主体）**：链式法则（COMPOSITION·派生新规则
A∘B→C）defer Phase 2（独立派生机制）·恒等（IDENTITY·a+0=a）折入化简规则（symbolic_transform 已有机制够·
+/−×/÷ 走化简规则亦可·本逆关系机制主战场=d/dx↔∫ 这类两条独立变换规则互逆）。

铁律：纯整数（kind/rule refs/采样点/rational 全 int/ConceptRef·assert_int 守）/ bit-identical（verify 纯函数·
gate 在 formal_train 集成层·本模块无 gate）/ 单向依赖（L7 training→L7 symbolic_transform apply_transform +
L7 vm_proof execute_composes_value + L5 cognition ConceptGraph/read_composes_attrs·向下·不向上·不环）/
反 theater（逆关系须构造验证 B∘A 还原·can't-verify/falsified 诚实降级·不偷渡 verified·**非教师声称互逆就 verified**）/
不写死（关系是教师陈述数据·非硬编码"+ −互逆"·code 是通用验证+存储机制·同 register_transform_rule）/
闭项守（apply 返 None→skip/降级·不可复合 can't-verify·三值逻辑·malformed caller try/except 守）。

诚实边界：① 逆验证=统计非证明（B∘A 还原 @ 有限采样 ≠ 数学逆·#479 守·humans 代值验亦此）·
② 自产验证 ≠ R6 两源 ≠ truth（两规则 single-source 教师·self-consistency 非构造性验证非 truth）·
③ 可复合约束（B LHS 须匹配 A 输出 shape·否则 can't-verify·须教师设计可复合规则对）·
④ Phase 1=INVERSE only（链式 defer·恒等折化简）·⑤ 机制完成 ≠ 已学（具体互逆对须 S8 课程训练）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.composes_attr import (
    record_composes_attr, read_composes_attrs,
    ATTR_RELATION_KIND, ATTR_RELATION_RULE_A, ATTR_RELATION_RULE_B,
    ATTR_OPERAND, ATTR_IMMEDIATE, ATTR_OPERATOR,
    RELATION_KIND_INVERSE,
)
from pure_integer_ai.cognition.shared.types import ConceptRef

# 小素数探针（B∘A 还原采样点·per-slot·arity≤6·同 symbolic_transform formal_train _XFORM_PROBES）。
_REL_PROBES = (2, 3, 5, 7, 11, 13)


def _arity_of(backend, root: ConceptRef) -> int:
    """数表达式树的 distinct operand sid 数（= arity·探针选 slot 数用）。

    DFS 收集 ATTR_OPERAND 的 int_a（operand sid）·set 去重（同 param 多次出现算 1·如 x*x arity=1）。
    build_composes_from_arith 按 lambda arg 序分配 make_variable(0..n-1)·故 distinct sid 数 = arity·
    slot 序 = 0..arity-1·range(arity) 探针覆盖全 slot（同 formal_train AST-parse arity 范式·但从树算非 parse）。
    """
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    graph = ConceptGraph(backend)
    children_of, _, _, _, _ = graph.read_composes_tree(root)
    sids: set[int] = set()

    def _walk(node: ConceptRef) -> None:
        attrs = read_composes_attrs(backend, node)
        if ATTR_OPERAND in attrs:
            sids.add(attrs[ATTR_OPERAND][0])
            return
        if ATTR_IMMEDIATE in attrs:
            return
        if ATTR_OPERATOR in attrs:
            for c in children_of.get(node, []):
                _walk(c)
            return
        # 无属性叶（concept/token）→ 无 operand

    _walk(root)
    return len(sids)


def register_inverse_relation(backend, concept_index, *, space_id: int,
                              name: str, kind: int,
                              rule_a_ref: ConceptRef, rule_b_ref: ConceptRef) -> ConceptRef:
    """注册运算间逆关系 name→(KIND, RULE_A, RULE_B)（教师陈述·数据驱动·非硬编码）。

    在 rule_a_ref 同 space 建 name 概念点·挂 ATTR_RELATION_KIND（kind）+ ATTR_RELATION_RULE_A（rule_a_ref）
    + ATTR_RELATION_RULE_B（rule_b_ref）。镜像 register_transform_rule 幂等+冲突 fail-loud 范式
    （ATTR_TRANSFORM_LHS/RHS 存 struct_ref 作 (int_a=sid,int_b=lid)）+ ATTR_PROPOSITION=11 marker 范式
    （KIND 挂 marker int_a=RELATION_KIND_*·int_b=0）。

    幂等：同 name 同 (kind, ruleA, ruleB) 重注册不变·重映射不同 fail-loud（拒歧义·同 register_transform_rule）。
    返 name 节点 ConceptRef（关系身份·verify_inverse_relation/load 查表键）。
    """
    assert_int(space_id, kind, _where="register_inverse_relation")
    from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
    name_ref = concept_index.ensure(name, space_id=space_id,
                                    tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    existing = read_composes_attrs(backend, name_ref)
    prev_kind = existing.get(ATTR_RELATION_KIND)
    if prev_kind is not None:
        prev_a = existing.get(ATTR_RELATION_RULE_A)
        prev_b = existing.get(ATTR_RELATION_RULE_B)
        if (prev_kind[0] != kind
                or prev_a != (rule_a_ref[0], rule_a_ref[1])
                or prev_b != (rule_b_ref[0], rule_b_ref[1])):
            raise ValueError(
                f"逆关系重名冲突: {name!r} 已映射→KIND{prev_kind[0]}/A{prev_a}/B{prev_b}"
                f"·试图重映射→KIND{kind}/A{(rule_a_ref[0], rule_a_ref[1])}"
                f"/B{(rule_b_ref[0], rule_b_ref[1])}"
                f"（fail-loud 拒歧义·同义重注册须同 (kind,A,B)）")
        return name_ref   # 幂等：同 (name, kind, A, B) 重注册不变
    record_composes_attr(backend, ref=name_ref, kind=ATTR_RELATION_KIND, int_a=kind, int_b=0)
    record_composes_attr(backend, ref=name_ref, kind=ATTR_RELATION_RULE_A,
                         int_a=rule_a_ref[0], int_b=rule_a_ref[1])
    record_composes_attr(backend, ref=name_ref, kind=ATTR_RELATION_RULE_B,
                         int_a=rule_b_ref[0], int_b=rule_b_ref[1])
    return name_ref


def load_inverse_relation(backend, relation_ref: ConceptRef):
    """读逆关系→(kind, rule_a_ref, rule_b_ref) | None（非关系/无 ATTR→None·caller 判）。"""
    attrs = read_composes_attrs(backend, relation_ref)
    kind_attr = attrs.get(ATTR_RELATION_KIND)
    a_attr = attrs.get(ATTR_RELATION_RULE_A)
    b_attr = attrs.get(ATTR_RELATION_RULE_B)
    if kind_attr is None or a_attr is None or b_attr is None:
        return None
    return (kind_attr[0], (a_attr[0], a_attr[1]), (b_attr[0], b_attr[1]))


def verify_inverse_relation(backend, concept_index, edge_store, *, space_id: int,
                            source: int, relation_ref: ConceptRef,
                            sample_inputs) -> bool | None:
    """构造验证逆关系 B∘A = identity @ 采样（反 theater 心脏·doc §五）。

    返三值（同刀C 三值逻辑·honest 降级）：
      - True  = verified（全采样输入 B∘A 还原原值·统计验非 truth·#479 守）
      - False = falsified（至少一可验采样 B∘A 不还原·诚实不偷渡 verified）
      - None  = can't-verify（不可复合·B 的 LHS 不匹配 A 的输出 shape·或关系非 INVERSE·或不可执行·
                诚实降级·非 theater·非声称"已验互逆"）

    机制（复用 symbolic_transform.apply_transform + vm_proof.execute_composes_value + rational.eq）：
      1. 读关系→(kind, ruleA, ruleB)·kind≠INVERSE → None（Phase 1 唯一 kind·COMPOSITION defer Phase 2）。
      2. 对每个采样输入 e（ConceptRef·已建 COMPOSES 树）：
         a. out_a = apply_transform(ruleA, e) → None（A 不匹配 e·e 不在 A 域）→ skip 此样本。
         b. out_b = apply_transform(ruleB, out_a) → None（**B LHS 不匹配 A 输出**）→ 返 None（can't-verify）。
         c. 执行 e 与 out_b @ 探针（per-slot 小素数）→ rational 值·rational.eq 全点成立 → 还原此样本。
            执行 None（含未 lower 的 Pow pattern 等）→ 返 None（can't-verify·诚实降级）。
         d. 任一采样 rational.eq 失败 → 返 False（falsified·诚实·不偷渡 verified）。
      3. 全采样还原 → True·无可验样本（全 skip·e 全不在 A 域）→ None（can't-verify·诚实·非声称 verified）。

    **反 theater**：关联 verified = 系统执行两规则串联还原·非教师声称。can't-verify/falsified 都诚实
    （不偷渡 verified）·stable≠correct（采样还原 ≠ 数学逆证明·#479 守）。
    **可复合约束**：逆验证要求两规则可复合（B LHS 匹配 A 输出 shape）·d/dx↔∫ 须 ∫ LHS 设计匹配 d/dx
    输出形·不可复合 → can't-verify（须教师设计可复合规则对·非 bug·非 theater）。
    """
    assert_int(space_id, source, _where="verify_inverse_relation")
    from pure_integer_ai.training.symbolic_transform import apply_transform
    from pure_integer_ai.training.vm_proof import execute_composes_value
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    from pure_integer_ai.crosscut.integer import rational

    rel = load_inverse_relation(backend, relation_ref)
    if rel is None:
        return None   # 非逆关系节点
    kind, rule_a_ref, rule_b_ref = rel
    if kind != RELATION_KIND_INVERSE:
        return None   # Phase 1 唯一 kind（COMPOSITION defer Phase 2）

    graph = ConceptGraph(backend)
    tested = 0
    for e_ref in sample_inputs:
        out_a = apply_transform(backend, concept_index, edge_store,
                                space_id=space_id, source=source,
                                rule_name_ref=rule_a_ref, input_ref=e_ref)
        if out_a is None:
            continue   # e 不在 A 域（A LHS 不匹配 e）→ skip 此样本（e 非合法输入·非 can't-verify）
        out_b = apply_transform(backend, concept_index, edge_store,
                                space_id=space_id, source=source,
                                rule_name_ref=rule_b_ref, input_ref=out_a)
        if out_b is None:
            return None   # **B LHS 不匹配 A 输出**→不可复合→can't-verify（诚实降级·非 theater）
        # 执行 e 与 B(A(e)) @ 探针 → 还原比对（stable≠correct·统计验非 truth·#479 守）
        _arity = _arity_of(backend, e_ref)
        if _arity > len(_REL_PROBES):
            continue   # 此样本 arity 超探针覆盖（>6）→ skip 此样本（input-specific·同 out_a None·防 IndexError·
                       # 对抗审 LOW：非关系级 can't-verify·他样本仍验·全 skip 时 tested==0 自然 None）
        _probes = tuple((_REL_PROBES[i], 1) for i in range(_arity))
        v_e = execute_composes_value(graph, e_ref, _probes)
        v_ba = execute_composes_value(graph, out_b, _probes)
        if v_e is None or v_ba is None:
            return None   # 不可执行（如含未 lower 的 Pow pattern·StepLimit）→ can't-verify（诚实降级）
        if not rational.eq(v_e, v_ba):
            return False   # 此采样 B∘A 不还原 → falsified（诚实·不偷渡 verified）
        tested += 1
    if tested == 0:
        return None   # 无可验样本（全 skip·e 全不在 A 域）→ can't-verify（诚实·非声称 verified）
    return True
