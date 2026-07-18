"""cognition.process.fixed_position_detect — exposure-driven CONCEPT-leaf 固定位检测（§十八 condition 3·dormant 基建）。

§十八 cue-保留结构抽象的 condition 3（对抗审 PASS-WITH-CONDITIONS 裁决的最小 sound 第一片·2026-07-17）：
给定一组结构同构样本（caller 分组·如 auto_discover_operators 按 shape_signature）·DFS 阅读序（镜像
_SkeletonBuilder.build CONCEPT_LEAF 的 _concept_slot_idx·build():388 `_seen` DAG 复用 + :472 子序 range）
收 CONCEPT-leaf token ConceptRef·按位对齐·同位同 token 跨样本一致 → 固定位（闭类原语候选·token 身份保留·
是/使 可分）·否则 → 变量位（PARAM content 槽·苹果/水果 异）。

**exposure-driven·无 frozenset**（D6 #2·裁决命门·禁 {是→COPULA} 硬编码）："fixed" 从跨样本 token-identity
一致性涌现·非词表硬挂。token 身份 = ConceptRef（concept_index 按 surface 去重 → 同 token 同 ref·"是" 跨样本
同 ref = 固定候选）。原语 TYPE 赋值（是→TYPE_COPULA 泛化·闭类原语 auto-extend）是**下游** D:11 readback
教师种子（§十八 point 4）·**非本函数**·本函数只判 fixed/variable + 报 token 身份。

**condition 1+2 已满足**（symbol_types.py TYPE_COPULA=13 / TYPE_CAUSES=15 / TYPE_ATTR_MARKER=16·登记不激活·
⊥ REL_* 关系命名空间·D6 #1 合法像 OPCODE_*）·本模块**只补 condition 3 检测器**·非冗余原语 type vocab。

**dormant 诚实**（同 A/B/C/D.1 dormant-infra 范式）：无 production consumer。condition 6 (c) wiring =
Phase E 热代码手术（shape_signature 编码固定位 token 身份 + CONCEPT_LEAF 分支保留固定位原语 vs 参化变量位·
复合键 (primitive-pattern, arity, position-dominance)·裁决 condition 6）·defer focused session。
**本检测器单独不解决**"是/使 同 shape_signature 合并坍缩"——须 (c) wiring 把固定位 token 身份反馈进分组/
形状签名（固定位 token 身份入键）才分离 是/使 簇。本检测器是 (c) wiring 的**必要组件**（per-position 固定/
变量分类）·非独立解。gate 待 caller wire 时加（无 production caller = bit-identical·同未调函数）。

铁律：纯整数（ConceptRef 整·零浮点）/ 确定性（DFS reading order + sorted children + 唯一 mode 判·bit-identical）/
  不写死（exposure-driven·无 frozenset·无 word→type 映射）/ 单向依赖（process→result+storage 向下读·
  同 structure_discover）/ bit-identical（纯读无写·无 production caller = 零行为变）。
"""
from __future__ import annotations

from typing import NamedTuple

from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.shared.types import ConceptRef

MIN_DETECT_SAMPLES = 2   # 触发固定位检测的最小样本数（< 2 无跨样本一致性→返 []·caller fallback 全参化·镜像 MIN_DISCOVER_SAMPLES）


class FixedConceptPosition(NamedTuple):
    """CONCEPT-leaf DFS-position 固定位检测结果（slot 对齐 _concept_slot_idx·build() PARAM 槽序）。"""
    slot: int                         # CONCEPT-leaf DFS 阅读序（= _concept_slot_idx·对齐 build() PARAM 槽）
    is_fixed: bool                    # True = 固定位（闭类原语候选·唯一 mode 跨样本一致达阈值）
    token_ref: ConceptRef | None      # is_fixed=True 时 = 一致 token ConceptRef（是/使）·False 时 None
    agreement: int                    # 此位 mode token 的样本数（is_fixed=False 时仍报 max-count·informational）
    total: int                        # 此位样本总数（= len(samples)）


def _concept_leaf_sequence(children_of, operator_of, operand_of, immediate_of,
                           store_target_of, root):
    """DFS 前序 → CONCEPT-leaf ConceptRef 阅读序（= _concept_slot_idx 对齐 PARAM 槽）。

    镜像 _SkeletonBuilder.build：:472 `for slot in range(cc0)` 子序（children_of 已按 order_index 排）+
    :388 `_seen` DAG 复用（共享 concept leaf 首遇记一次 = 一槽）。

    CONCEPT leaf = 叶（不在 children_of·无 COMPOSES 子）+ 无结构 attr（operator/operand/immediate/
    store_target 全无 = `_is_concept_leaf` 的 `not attrs` 判据在 read_composes_tree 5-dict 投影）。
    """
    out: list[ConceptRef] = []
    seen: set[ConceptRef] = set()
    stack: list[ConceptRef] = [root]
    while stack:
        node = stack.pop()
        if node in seen:
            continue   # DAG 共享·首遇已记（镜像 build():388 `_seen` 复用同一槽）
        seen.add(node)
        if (node not in children_of
                and node not in operator_of
                and node not in operand_of
                and node not in immediate_of
                and node not in store_target_of):
            out.append(node)   # CONCEPT leaf（无子无属性）·叶无子可递归
            continue
        for child in reversed(children_of.get(node, [])):   # 逆序压栈·保 reading order pop（首子先出）
            if child not in seen:
                stack.append(child)
    return out


def detect_fixed_concept_positions(graph: ConceptGraph, samples,
                                   *, min_agreement_count: int | None = None):
    """exposure-driven CONCEPT-leaf 固定位检测（§十八 condition 3·dormant·pure read·无写）。

    samples : 结构同构的 COMPOSES 程序树根 ConceptRef 列表（caller 分组·如 auto_discover_operators 按
              shape_signature·discover_skeleton 同构门已保证 concept-leaf spine 一致）。
    min_agreement_count : 固定位判阈值（同位某 token 出现 ≥ 此数 且 为唯一 mode → 固定）。
              None（默认）= unanimous（全样本同 token）·最保守·bit-identical。
              (c) wiring caller 可放宽（嘈杂语料 是 5/6 → 阈值 5）。

    返：
      - samples < MIN_DETECT_SAMPLES → []（无跨样本一致性·caller 须 fallback 参化全部 = 现状行为）。
      - 各 sample CONCEPT-leaf 序**等长**（spine 同构）→ list[FixedConceptPosition]（per position·slot 对齐
        _concept_slot_idx）：unanimous / 达阈值唯一 mode → fixed（token_ref = mode）·否则 variable（None）。
      - 各 sample CONCEPT-leaf 序**异长**（spine 非同构·discover_skeleton 同构门不应放过·防御 malformed）→ None。

    **确定性 bit-identical**：mode 判 = max count + 唯一性（`len(modes) == 1`）·不依赖 Counter.most_common
    tie-break（counts dict 插入序确定·mode 唯一即无歧义·ties→非 fixed）。DFS reading order + sorted children 锁序。

    **反 theater**：无 frozenset·无 word→type 映射·"fixed" 纯跨样本 token-identity 统计涌现。原语 TYPE 赋值
    下游 D:11（非此函数）。无 production caller → bit-identical（gate 待 (c) wiring caller 时加）。
    """
    k = len(samples)
    if k < MIN_DETECT_SAMPLES:
        return []   # 单样本无跨样本一致性·caller fallback 全参化（现状行为）
    # 每 sample DFS 收 CONCEPT-leaf 阅读序
    sequences: list[list[ConceptRef]] = []
    for root in samples:
        (children_of, operator_of, operand_of,
         immediate_of, store_target_of) = graph.read_composes_tree(root)
        sequences.append(_concept_leaf_sequence(
            children_of, operator_of, operand_of, immediate_of, store_target_of, root))
    # spine 同构门：等长（discover_skeleton 同构门已保证·此处防御 malformed 输入）
    length = len(sequences[0])
    for seq in sequences[1:]:
        if len(seq) != length:
            return None   # CONCEPT-leaf spine 异构 → 非同簇（caller 须分簇）
    threshold = k if min_agreement_count is None else min_agreement_count
    out: list[FixedConceptPosition] = []
    for slot in range(length):
        tokens_at = [seq[slot] for seq in sequences]
        counts: dict[ConceptRef, int] = {}
        for t in tokens_at:
            counts[t] = counts.get(t, 0) + 1
        max_count = max(counts.values()) if counts else 0
        modes = [t for t, c in counts.items() if c == max_count]   # 插入序确定·ties 多 mode
        if max_count >= threshold and len(modes) == 1:
            out.append(FixedConceptPosition(slot=slot, is_fixed=True,
                                            token_ref=modes[0],
                                            agreement=max_count, total=k))
        else:
            out.append(FixedConceptPosition(slot=slot, is_fixed=False,
                                            token_ref=None,
                                            agreement=max_count, total=k))
    return out
