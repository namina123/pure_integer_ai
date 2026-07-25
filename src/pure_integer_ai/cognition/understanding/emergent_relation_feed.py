"""cognition.understanding.emergent_relation_feed — 刀4 子环3 鸡生蛋破解（experience_count 概念维验证）。

**鸡生蛋问题**：非-cue 词"引发"无 CAUSES 边（cue_extractor 不认它·不建含"引发"的 CAUSES 边）→
不在 reward_propagate concept_targets（主集来自 causes_edges 端点·reward_propagate.py:168-175）→
无 e_sn/e_tn → 子环4 _experience_ok 永远 False → 死锁。

**破解**：collect_emergent_word_concepts_for_feed 扫 D:11 SHADOW 候选边·返 from 端 word concept refs·
reward_propagate concept_targets 扩展加入这些 refs（gate EMERGENT_RELATION_FEED_MODE ON）·
让"引发"被经验 feed·子环4 _experience_ok 验证。

**守 reward CAUSES-only 铁律**（防塌柱①·不可破）：
  - causes_edges 列表（reward_propagate.py:112-113）不变·D:11 边**永不进** distributed /
    record_episode_result（edge sn/tn 写）。
  - 只 word concept ref 进 concept_targets（experience_count **概念维对偶表**·设计本意·
    storage/experience_count.py:1-15 明示"概念维对偶聚合·非新机制·多落一笔"·非 edge reward 多头）。
  - reward_propagate.py:131 CAUSES-only assert 不动（distributed 仍全 CAUSES）。

铁律：纯整数（ConceptRef/REL_*/tier 全 int）/ 确定性（set 迭代无序但 caller sorted·bit-identical）/
  §8.1c（概念维 feed 非 edge reward·不破 CAUSES-only）/ 单向依赖（L4→L0 storage）。
"""
from __future__ import annotations

from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.composes_attr import read_composes_attrs, ATTR_RELATION_PRIMITIVE
from pure_integer_ai.storage.node_store import TIER_SHADOW
from pure_integer_ai.cognition.shared.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.cognition.shared.relation_primitives import REL_CAUSES


def collect_emergent_word_concepts_for_feed(backend: StorageBackend, *,
                                            rel_kind_filter: int = REL_CAUSES,
                                            tier_filter: int = TIER_SHADOW,
                                            ) -> set[tuple[int, int]]:
    """扫 D:11 边（tier=tier_filter）·target 读 ATTR_RELATION_PRIMITIVE==rel_kind_filter
    ·返 from 端 word concept refs（涌现假设候选·须被 feed experience_count 才能验证）。

    caller：reward_propagate.py 落点① concept_targets 扩展（gate EMERGENT_RELATION_FEED_MODE ON）。
    返 set[ConceptRef]（from 端 word concept）·caller update 进 concept_targets·
    后续 sorted 迭代 + record_experience_outcome feed e_sn/e_tn。

    **tier_filter=TIER_SHADOW**（默认）：只 feed 未晋升候选（已 promote PRIMARY 走 _reward_ok 通路·
    不重复 feed）。**rel_kind_filter=REL_CAUSES**（默认·首版只涌 REL_CAUSES）。

    表未注册（bare fixture）→ KeyError 静默返空 set（向后兼容·镜像 record_experience_outcome 范式）。
    """
    out: set[tuple[int, int]] = set()
    try:
        rows = backend.select("edge", where={"edge_type": EDGE_RELATION_SIGNAL})
    except KeyError:
        return out   # edge 表未注册（bare fixture）·向后兼容
    for r in rows:
        if r.get("tier") != tier_filter:
            continue
        target_ref = (r["space_id_to"], r["local_id_to"])
        attrs = read_composes_attrs(backend, target_ref)
        kind = attrs.get(ATTR_RELATION_PRIMITIVE, (0, 0))[0]
        if kind != rel_kind_filter:
            continue   # target REL_* kind 不匹配（脏边/教师种子异 kind）·skip
        out.add((r["space_id_from"], r["local_id_from"]))
    return out
