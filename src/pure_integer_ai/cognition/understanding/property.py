"""cognition.understanding.property — G1 reification + #774 PROPERTY 命题建边。

设计 doc/重来_G1reification_774PROPERTY_设计_2026-07-09.md（fork 分析 §5.3 实施 ready·选 b 避 generate 改动）。

**命题节点 reification（G1）**：属性命题 = (subject, attr_type, value) 三元的载体节点。
  - 命题节点身份 = (subject, attr_type) 对·确定性 surface ``__prop_{subj_sid}_{subj_lid}_{attr_sid}_{attr_lid}``·
    concept_index.ensure 去重（同 subject 同 attr_type 多 claim 命中同命题节点）。
  - ATTR_PROPOSITION=11 标记（composes_attr·int_a=int_b=0·仅标记·同 ATTR_ORIGIN/ATTR_RELATION_PRIMITIVE 范式·
    非结构 kind·read_composes_tree 忽略·_STRUCTURAL_KINDS 不含·inline 不传播）。
  - value = EDGE_PROPERTY 出边（命题节点→value 概念·core space·非 memory_space）。

**#774 builder**：在命题节点建 PROPERTY 出边（value）。fork 分析 §3.2/§3.3 根因——PROPERTY 语义在 token 主语
但 G3b 读 produced struct_ref（struct_ref 无 PROPERTY 边=theater）。命题节点承载三元后·同(subject,attr_type)
多 value 聚同节点 → G3b ``len(PROPERTY out-edges)>1`` 精确判矛盾·无假矛盾（异 subject / 异 attr_type 不聚）。

**G3b 真消费者（反 theater）**：has_value_claim=True（property_claims 非空·gate PROPOSITION_MODE ON 时）激活 G3b·
命题节点真有 PROPERTY 出边（本 builder 建）·G3b 全局扫真判·非空集永返 1（fork §四选项 B theater 反例）。

铁律：纯整数（命题 ref + PROPERTY 整边 + ATTR_PROPOSITION 标记·零浮点·assert_int 守）/ 确定性（surface 编码
bit-identical + concept_index.ensure 去重 + query_from 幂等）/ 不写死（cue 词表元定义·builder 只机制）/ reification
给表达力非验证力（命题 truth=#479 墙·G3b 只判结构矛盾层a·语义真对立层b/c #479 truth 关切 W2·**非 W1 D 物理接地墙**·provisional/可废止对立 E3 覆写+E4 推理引擎可达·只 definitive truth 撞 #479·defer）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import EdgeStore, DEFAULT_STRENGTH
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_types import EDGE_PROPERTY
from pure_integer_ai.storage.composes_attr import (
    record_composes_attr, ATTR_PROPOSITION,
    ATTR_PROP_SUBJ, ATTR_PROP_ATTR, ATTR_PROP_POLMOD, ATTR_PROP_INTENSITY,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex


def build_property_edges(edge_store: EdgeStore, concept_index: ConceptIndex,
                         backend, refs: list[tuple[int, int]],
                         *, property_claims: list[tuple[int, int, int, int, int, int]],
                         source: int, space_id: int,
                         default_attr_ref: tuple[int, int] | None = None) -> int:
    """G1+#774 命题建边（property_claims 6-tuple → 命题节点 + PROPERTY value 出边·P0.3 扩 polarity/modality 进 surface）。

    property_claims：``(subject_idx, attr_type_idx, value_idx, _reserved, polarity, modality)`` 6-int tuple·
    经 refs 解析为概念 ref。attr_type_idx<0（"具有 Z"领属模式·无 attr_type）→ default_attr_ref 非 None 时用之
    作 attr_type（STEP5 PR3 possess un-defer·REL_PROPERTY ConceptRef 作默认 attr_type·补命题身份
    (subject,REL_PROPERTY,value)·G3b 消费）·default_attr_ref=None 时 skip（既有行为 bit-identical·
    设计 §六诚实边界·无 attr_type 无命题身份）。polarity(0/1)+modality(0-4) 进 surface 后缀（pol/mod>0 才加·
    default 0=既有命题 bit-identical·P0.3 扩展·B1 否定/B2 情态填值·防御 claim[4]/[5] 缺省 0 向后兼容 4-tuple）。
    返建边数（命题节点 ensure 幂等不计·PROPERTY 边计）。

    每条 的...是 claim：
      ① 命题节点身份 = (subject, attr_type)·确定性 surface ``__prop_{subj}_{attr}``·concept_index.ensure 去重。
      ② ATTR_PROPOSITION=11 标记（record_composes_attr 幂等·同节点已标 skip）。
      ③ value 出边（PROPERTY·命题节点→value·core space·source=caller raw.source·epistemic_origin=None·
         property claim 是 bare text 断言非 structured/cue/llm 认识论源·同 refers_occurrence pronoun PROPERTY 范式）。
         query_from 幂等 skip（同命题节点同 value 重 claim 不重复建边·EdgeStore.add append-only 不去重）。

    **命题节点不进 dag_path/structure_units**（无 role_seq·非 struct_unit_refs·generate structure_units 不收）·
    判断层载体非路径层载体·零 J1/J2/J3 path 计算扰动（设计 §二.3 选 b 理由）。

    诚实边界：reification 给表达力非验证力（命题"猫的颜色是黑"是否真=#479 墙·G3b 只判结构矛盾层a·
    同(subject,attr_type)多值=CONTRADICTED·语义真对立层b/c #479 truth 关切 W2·**非 W1 D 物理接地墙**·provisional/可废止对立 E3 覆写+E4 推理引擎可达·只 definitive truth 撞 #479·defer·judge.py:22 既有 defer 注）。
    """
    assert_int(source, space_id, _where="build_property_edges.args")
    n = 0
    for claim in property_claims:
        subj_idx, attr_idx, val_idx = claim[0], claim[1], claim[2]
        polarity = claim[4] if len(claim) > 4 else 0   # P0.3 命题节点扩展·default 0 向后兼容 4-tuple
        modality = claim[5] if len(claim) > 5 else 0
        intensity_num = claim[6] if len(claim) > 6 else 1   # #1134 程度 augment·default 1/1 向后兼容 ≤6-tuple（degree 副词缩放命题值强度·Rational·非 float）
        intensity_den = claim[7] if len(claim) > 7 else 1
        assert_int(polarity, modality, intensity_num, intensity_den,
                   _where="build_property_edges.pol_mod_intensity")   # 纯整数诺（防 float 静默 corrupt surface·agent 审建议）
        assert 0 <= polarity <= 1, f"polarity 编码 ∈{{0=肯定,1=否定}}·got {polarity}"
        assert 0 <= modality <= 4, f"modality 编码 ∈{{0=实然,1=□必然,2=◇可能,3=道义必然,4=道义可能}}·got {modality}"
        assert intensity_den > 0 and intensity_num > 0, \
            f"intensity 正缩放（num>0·den>0·Rational·很=2/1·稍=2/5）·got {intensity_num}/{intensity_den}"
        if subj_idx < 0 or val_idx < 0 or subj_idx >= len(refs) or val_idx >= len(refs):
            continue   # 越界守（refs 切片映射错位 fail-soft·不崩·同 build_causes 索引消费）
        subj = refs[subj_idx]
        val = refs[val_idx]
        if attr_idx < 0:
            if default_attr_ref is None:
                continue   # 领属模式（"具有 Z"）无 attr_type·无 default_attr_ref·缺命题身份·skip（既有 bit-identical）
            attr = default_attr_ref   # STEP5 PR3 possess un-defer·REL_PROPERTY 作默认 attr_type·补命题身份
        else:
            attr = refs[attr_idx] if attr_idx < len(refs) else None
            if attr is None:
                continue
        # ① 命题节点身份 = (subject, attr_type, polarity, modality)·确定性 surface·concept_index.ensure 去重
        # P0.3 pol/mod 进 surface 后缀（pol/mod>0 才加·pol/mod=0 surface 字面不变=既有命题 bit-identical）
        prop_surface = f"__prop_{subj[0]}_{subj[1]}_{attr[0]}_{attr[1]}"
        if polarity or modality:
            prop_surface = f"{prop_surface}_{polarity}_{modality}"
        if not (intensity_num == 1 and intensity_den == 1):
            prop_surface = f"{prop_surface}_i{intensity_num}_{intensity_den}"   # #1134 intensity 后缀（≠1/1·_i 前缀避 pol/mod 后缀歧义·default 1/1 无后缀 bit-identical）
        prop_ref = concept_index.ensure(
            prop_surface, space_id=space_id,
            tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        # ② ATTR_PROPOSITION=11 标记（record_composes_attr 幂等·同 (ref,kind) 已标 skip）
        record_composes_attr(backend, ref=prop_ref,
                             kind=ATTR_PROPOSITION, int_a=0, int_b=0)
        # ②-bis 命题 identity 结构存（STEP6 PR3·G3b 模态对当跨节点按 (subj,attr) 分组用·解 ref→surface defer
        # ·node_store 无 surface 列·命题节点 (subj,attr,pol,mod) 亦结构存于 composes_attr·additive 非结构 kind）
        record_composes_attr(backend, ref=prop_ref,
                             kind=ATTR_PROP_SUBJ, int_a=subj[0], int_b=subj[1])
        record_composes_attr(backend, ref=prop_ref,
                             kind=ATTR_PROP_ATTR, int_a=attr[0], int_b=attr[1])
        record_composes_attr(backend, ref=prop_ref,
                             kind=ATTR_PROP_POLMOD, int_a=polarity, int_b=modality)
        # #1134 程度 intensity 结构存（平行 ATTR_PROP_POLMOD·命题值强度可查·default 1/1·degree 副词缩放·Rational num/den·非结构 kind）
        record_composes_attr(backend, ref=prop_ref,
                             kind=ATTR_PROP_INTENSITY, int_a=intensity_num, int_b=intensity_den)
        # ③ value 出边（PROPERTY·幂等 skip·同命题节点同 value 重 claim 不重复建边）
        existing = edge_store.query_from(prop_ref[0], prop_ref[1],
                                         edge_type=EDGE_PROPERTY)
        already = any(
            row.get("space_id_to") == val[0]
            and row.get("local_id_to") == val[1]
            for row in existing
        )
        if already:
            continue   # 同命题节点同 value 已建→skip（幂等·observe 跨段重 claim / 多轮重 observe 不 corrupt）
        if prop_ref == val:
            continue   # 自环不建（命题节点→自身 value·防御·同 build_causes _insert a==b 守）
        edge_store.add(
            space_id_from=prop_ref[0], local_id_from=prop_ref[1],
            space_id_to=val[0], local_id_to=val[1],
            edge_type=EDGE_PROPERTY, strength=DEFAULT_STRENGTH,
            source=source, epistemic_origin=None,   # bare text 断言·非 structured/cue/llm 认识论源
            tier=TIER_PRIMARY,
        )
        n += 1
    return n
