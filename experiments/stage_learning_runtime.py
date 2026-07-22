"""训练阶段的基础频次注入和候选晋升 runtime。"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.cognition.shared.types import (
    ConceptRef,
    MODALITY_ARITH,
    MODALITY_CODE,
    WEANING_PRE,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.storage.edge_store import SUBTYPE_OCCURRENCE
from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO
from pure_integer_ai.training.promote import promote_edge, promote_report

def _inject_base_freq(ctx: TrainContext, corpus: list[CollectedItem]) -> None:
    """阶段2 通识 base_freq 注入（experience_count·断奶前·observe 后 lookup 命中 token concept）。

    镜像 edge_store base_strength 范式（建概念时一次性写通识先验·reward 不调·断奶后退场镜像
    EPI_LLM_CONFIRM）。只注 token concept（surface=token 文本）·不注段 struct_ref（observe seg_label
    `__seg_/__prog_` 结构标签非 token·无频次语义）/ code / arith source。不改 observe /
    normalize_to_concept / ConceptIndex.ensure（保 observe 4 入口公共原语契约 §十一缺口#3）。
    Counter 返 int 纯整数合规·first-write-wins 幂等·sorted 确定序 bit-identical。
    """
    from collections import Counter
    from pure_integer_ai.storage.experience_count import record_base_freq
    if ctx.weaning_phase != WEANING_PRE:
        return   # 断奶后退场（镜像 EPI_LLM_CONFIRM·断奶后新概念无 base_freq 只靠 exp 自积累）
    surface_freq = Counter()
    for item in corpus:
        if item.modality in (MODALITY_CODE, MODALITY_ARITH) or not item.tokens:
            continue   # 代码/算术非 token concept / 空 tokens 跳过
        for tok in item.tokens:
            if tok:
                surface_freq[tok] += 1
    for surface, freq in sorted(surface_freq.items()):   # sorted bit-identical
        ref = ctx.concept_index.lookup(surface, ctx.space_id)
        if ref is None:
            continue   # lookup miss：未建 concept（结构标签/未 observe token）·不建·诚实 skip
        record_base_freq(ctx.backend, ref=ref, base_freq=int(freq))

def _promote_eligible(ctx: TrainContext, teacher: Any) -> tuple[int, int]:
    """阶段4 promote 三重（SHADOW→PRIMARY·频次/reward/定义·§十三防塌C4）。

    扫 SHADOW 边 → promote_report 判 eligible → promote_edge flip tier。
    返 (promote_count, oov_promote_count)。oov_promote = 概念点新晋 PRIMARY 计数。
    """
    from pure_integer_ai.storage.node_store import TIER_SHADOW
    promote_count = 0
    oov_promote = 0
    promoted_concepts: set[ConceptRef] = set()
    rows = ctx.backend.select("edge", where={"tier": TIER_SHADOW})
    for row in rows:
        # occurrence 是每次语篇事件的短期事实，不是可晋升的长期关系边；其
        # 兼容行允许同五元端点多次出现，不能交给五元 EdgeRef 晋升接口猜测。
        if (row.get("edge_type") == EDGE_REFERS_TO
                and row.get("subtype") == SUBTYPE_OCCURRENCE):
            continue
        ref = (row["space_id_from"], row["local_id_from"],
               row["space_id_to"], row["local_id_to"], row["edge_type"])
        rep = promote_report(ctx.edge_store, ref, teacher=teacher, backend=ctx.backend)
        if not rep["eligible"]:
            continue
        if promote_edge(ctx.edge_store, ctx.node_store, ref, teacher=teacher,
                        backend=ctx.backend):
            promote_count += 1
            for c in ((ref[0], ref[1]), (ref[2], ref[3])):
                if c not in promoted_concepts:
                    promoted_concepts.add(c)
                    oov_promote += 1
    if promote_count > 0:
        # D:11 PRIMARY 候选和 cue_rel_of 都有 ConceptGraph 读缓存；晋升后必须同进程可见。
        ctx.concept_graph.invalidate_ancestor_map(ctx.space_id)
    return promote_count, oov_promote

__all__ = [
    "_inject_base_freq",
    "_promote_eligible",
]
