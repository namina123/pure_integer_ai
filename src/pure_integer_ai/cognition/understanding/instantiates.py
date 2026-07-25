"""cognition.understanding.instantiates — EDGE_INSTANTIATES 建边（结构一等化 Phase A·§十三-bis A.1）。

EDGE_INSTANTIATES(=15·C9-bis C:15) = 表层段 __seg_ struct_ref → discovered skeleton_ref
（"此表层实例化此结构"·layer1 结构一等化图对象）。**替维度桥 ATTR_SKELETON_BINDING 注解**
（composes_attr kind=24·维度桥 #1066-1075 effect-dormant·reader 只写 last_dim_skeleton 零消费=死信）。

**关联在图中**：真边替 dormant 注解（铁律"关联在图中"·对应/绑定是图对象非节点属性 tag）。

**来源分层（honest）**：skeleton 由 auto_discover_operators 从语言样本归纳（structure_discover.py:1104）·
binding 纯结构（surface↔structure·**无 cue 内容·无关系型信息**）→
source=SOURCE_DERIVED（discovery 派生）·epistemic_origin=EPI_STRUCTURED（结构绑定·非 cue·非 LLM）。
**审2 APPROVE A.1**：INSTANTIATES 边 honest EPI_STRUCTURED（非 cue·非 mislabel·pure 结构绑定）。
对照 A.2 REALIZES（关系型全来自 `_CUE_WORDS`=cue→relation 换装命门·审2 REJECT·DEFER Phase D 外源 oracle）。

**幂等**：query_from 查 struct_ref 已有同 (skeleton_ref, EDGE_INSTANTIATES) 边则 skip（mirror
bootstrap_is_a_edges 幂等范式·EdgeStore.add append-only 不去重·跨 round re-observe / 多轮训练不 corrupt）。

**测度**：结构边·strength 恒 1·不接 reward（effective_weight 只认 {PRECEDES,CAUSES,REFERS_TO}·
INSTANTIATES 不内·M9 非学习对象·同 IS_A/MEREOLOGY 结构边纪律）。

**bit-identical**：caller（observe·COMPOSES_COMBINE_MODE gate 后）门控·gate OFF→零边→逐字现状。

铁律：纯整数（ConceptRef + EDGE_INSTANTIATES 整边·零浮点）/ 不写死（skeleton 来自 discovery·本函数只机制非语义）/
  单向依赖（L4 understanding→L1 storage 向下）/ bit-identical（gate OFF 零副作用 + query_from 幂等）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import EdgeStore, EPI_STRUCTURED, SOURCE_DERIVED
from pure_integer_ai.storage.edge_types import EDGE_INSTANTIATES
from pure_integer_ai.storage.node_store import TIER_PRIMARY

INSTANTIATES_STRENGTH = 1   # 结构边·恒 1·不接 reward（M9 非学习对象·同 IS_A 结构边纪律）


def build_instantiates_edge(edge_store: EdgeStore,
                            struct_ref: tuple[int, int],
                            skeleton_ref: tuple[int, int],
                            *, space_id: int) -> int:
    """EDGE_INSTANTIATES 建边（struct_ref → skeleton_ref·"此表层实例化此结构"）。

    幂等：query_from 查 struct_ref 已有同 (skeleton_ref, EDGE_INSTANTIATES) 边则 skip
    （mirror bootstrap_is_a_edges·跨 round re-observe 不 corrupt）。自环不建（struct≠skeleton）。
    返建边数（0=skip/自环·1=建）。
    """
    assert_int(space_id, _where="build_instantiates_edge.space_id")
    if struct_ref == skeleton_ref:
        return 0
    existing = edge_store.query_from(struct_ref[0], struct_ref[1], edge_type=EDGE_INSTANTIATES)
    already = any(
        row.get("space_id_to") == skeleton_ref[0]
        and row.get("local_id_to") == skeleton_ref[1]
        for row in existing
    )
    if already:
        return 0   # 同 (struct→skeleton, INSTANTIATES) 已建→skip（幂等·EdgeStore.add append-only 不去重）
    edge_store.add(
        space_id_from=struct_ref[0], local_id_from=struct_ref[1],
        space_id_to=skeleton_ref[0], local_id_to=skeleton_ref[1],
        edge_type=EDGE_INSTANTIATES, strength=INSTANTIATES_STRENGTH,
        source=SOURCE_DERIVED, epistemic_origin=EPI_STRUCTURED,
        order_index=None, role=None,   # INSTANTIATES 无时序/槽位语义
        tier=TIER_PRIMARY,
    )
    return 1
