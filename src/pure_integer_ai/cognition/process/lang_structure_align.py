"""cognition.process.lang_structure_align — 钥匙①件4 变长 LCS wrapper（语言独有·arith 无对应）。

变长语言样本 COMPOSES 根 → consensus 锚位 LCS 折叠（pairwise_fold）→ 每样本产**等长对齐** token
concept_ref 序列（锚位=原 token concept_ref·锚间段=**段首 token concept_ref**·空段=统一占位）→ caller
建对齐独立根喂 discover_skeleton（破同子数门拒变长·structure_discover.py 同子数门 line 339-340）。

机制（doc/重来_钥匙①语言结构发现机制设计_修正分析七.md §三件4 + §六 D2 弱化门）：
  - consensus anchors = 全样 LCS（algorithm.a4_alignment.pairwise_fold·保序·确定性 tiebreak）。
  - 每样本 alignment_matches(consensus, sample_lid) → 锚位位置 → 切段（段前/锚间/段后·共 |anchors|+1 段）。
  - 等长 = 2·|anchors|+1（段 slot + 锚位 交替 + 末段 slot）。
  - **段 slot = 段首 token concept_ref**（cross-sample 异词同槽 = PARAM 泛化牙·件2 D2 弱化门极致·
    "猫追狗"/"大猫追大狗" 段前 slot GAP/"大" 异词同槽 = 语言泛化·空段才占位）。
  - 空段 slot = 占位 concept_ref（统一 surface __lang_align_gap__·全样同·DAG 共享同槽）。
  - 退化（<MIN_DISCOVER_SAMPLES / consensus 空 / 样本未全匹配 consensus）→ None（caller 走原同长路径·
    bit-identical·变长不发现 = 诚实不纸面闭合）。

铁律：纯整数（token local_id 喂 LCS）/ 单向依赖（L5→L2 a4_alignment + L5→L0 graph_view·
不 import structure_discover 守 wrapper 不反向消费 discover）/ 不写死（锚位=LCS 共识·非硬编码词性/位置）/
§8.1c（LCS 结构对齐·wrapper 零边建造·仅占位 ensure 幂等·非语义判据·不撞三死刑）/ 最少边
（复用 pairwise_fold/alignment_matches·不重写 LCS DP）/ bit-identical（pairwise_fold/alignment_matches
确定性 tiebreak·种子序 (-len,tuple) 确定·占位 surface 确定）。

诚实边界（defer·非偷懒）：
  - 段内多 token 取段首·丢段内结构（句内异质 span 嵌合 = 件6 defer·doc §五）。
  - 全变长 roots 一起 fold·混杂时 consensus 退化（按 LCS 矩阵预聚类远期 defer·首版诚实返 None）。
  - 抽象级 shape_signature（IS_A LCA 上卷）= 第二刀·件4 是词级（D3 待用）。
"""
from __future__ import annotations

from pure_integer_ai.algorithm.a4_alignment import pairwise_fold, alignment_matches
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.storage.node_store import NODE_CONCEPT, TIER_PRIMARY

# 占位 surface（空段 slot·全样同 ref·DAG 共享同槽·concept_index 幂等 ensure 不膨胀）
_GAP_SURFACE = "__lang_align_gap__"
# consensus 至少 1 锚位（空 consensus = 无共识 → 退化 None）
_MIN_ANCHORS = 1


def align_variable_lang_sequences(
    graph: ConceptGraph,
    roots: list[ConceptRef],
    *,
    concept_index,
    space_id: int,
    min_samples: int = 2,
) -> list[list[ConceptRef]] | None:
    """变长语言 COMPOSES 根 → consensus LCS 折叠 → 每样本等长对齐 token concept_ref 序列。

    参数：
      graph         : ConceptGraph（read_composes_tree 读 token 子树·按 order_index 保序）。
      roots         : 变长语言样本的 COMPOSES 根 ConceptRef list（NOP SEQ·children = token 叶）。
      concept_index : ConceptIndex（占位 __lang_align_gap__ 幂等 ensure）。
      space_id      : 概念空间 id。
      min_samples   : 最小样本数（默认 2·同 MIN_DISCOVER_SAMPLES）。

    返 list[list[ConceptRef]]（每样本等长 = 2·|consensus|+1·段 slot + 锚位 交替）或 None：
      - <min_samples 样本 / 任 root 无 token 子树 / consensus 空 / 任样本未全匹配 consensus → None。
    纯算法 + 占位 ensure（幂等）。caller 建对齐独立根 + COMPOSES 序喂 discover_skeleton（破同子数门）。
    """
    if len(roots) < min_samples:
        return None
    # 读每 root 的 token concept_ref 序列 + local_id 序列（read_composes_tree children_of 按 order_index 保序）
    token_refs: list[list[ConceptRef]] = []
    token_lids: list[list[int]] = []
    for root in roots:
        children_of = graph.read_composes_tree(root)[0]
        toks = list(children_of.get(root, []))
        if not toks:
            return None    # 空 token 子树退化
        token_refs.append(toks)
        token_lids.append([t[1] for t in toks])    # local_id 喂 LCS（纯整）
    # consensus anchors = 全样 LCS（pairwise_fold·保序·种子序 (-len,tuple) 确定）
    consensus, _score = pairwise_fold(token_lids)
    if len(consensus) < _MIN_ANCHORS:
        return None    # 无共识退化（consensus 空）
    # 占位 concept_ref（幂等 ensure·全样同·空段 slot）
    gap_ref = concept_index.ensure(
        _GAP_SURFACE, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    # 每样本产等长对齐 concept_ref 序列
    aligned: list[list[ConceptRef]] = []
    for toks_ref, toks_lid in zip(token_refs, token_lids):
        pairs = alignment_matches(consensus, toks_lid)
        if len(pairs) != len(consensus):
            return None    # 不变量断言（构造上恒真：consensus=全样 LCS·每样本必含全 consensus·pairs 长度==consensus·防御性 guard）
        seq: list[ConceptRef] = []
        prev_pos = -1    # 前一锚位位置（段前 slot = [prev_pos+1, anchor_pos-1]）
        for (_a_pos, pos_b) in pairs:
            seg_start = prev_pos + 1
            # 段前 slot：段非空取段首 token concept_ref（cross-sample 异词同槽=PARAM 泛化牙）·空段占位
            seq.append(toks_ref[seg_start] if seg_start < pos_b else gap_ref)
            seq.append(toks_ref[pos_b])    # 锚位 token（consensus LCS 共识·全样同 token）
            prev_pos = pos_b
        # 末段 slot
        seg_start = prev_pos + 1
        seq.append(toks_ref[seg_start] if seg_start < len(toks_ref) else gap_ref)
        aligned.append(seq)
    return aligned
