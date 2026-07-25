"""cognition.understanding.realizes — EDGE_REALIZES 建边（对应图对象·Phase D §十六-bis D.1）。

EDGE_REALIZES(=27·C9-bis B:27) = skeleton（发现的骨架）→ reified relation-type 节点（__REL_SUBSET__ 等）
（"此结构实现此逻辑关系"·layer3 对应的图对象）。承 Phase A（INSTANTIATES surface→skeleton）·续到 logic。

**option (b) sound 来源**（Phase D §十六-bis·解 Phase A审2 REJECT 命门）：skeleton 从**通用文本独立发现**
（auto_discover_operators·不 partition 发现输入）·之后 iff skeleton 的 forming-sample token-pair（ordered）
**命中外源 EDGE_IS_A**（SOURCE_CONCEPTNET/SOURCE_CHINESE_KB + EPI_STRUCTURED·boot 已种）→ 写 REALIZES。
**oracle-pair-match 定 IS_A·非读 `_CUE_WORDS`**·**禁 pairs→文本渲染**（Cue 经渲染文本泄漏 = 命门隐蔽复现）。
labeled bed（同 boot IS_A）·学习 claim 严禁前置·验 floor Phase F·consumer Phase E。

**关联在图中**：REALIZES 真边 = 对应图对象（structure→logic·承 INSTANTIATES surface→structure）。

**幂等**：query_from 查 skeleton 已有同 (rel-type, REALIZES) 边则 skip（mirror build_instantiates_edge /
bootstrap_is_a_edges·EdgeStore.add append-only 不去重·跨 round re-discover 不 corrupt）。

**测度**：结构对应边·strength 恒 1·不接 reward（effective_weight 只认 {PRECEDES,CAUSES,REFERS_TO}·REALIZES 不内·
M9 非学习对象·同 INSTANTIATES/IS_A 结构边纪律）。

**bit-identical**：caller（structure_discover.label_realizes_is_a·REALIZES_MODE gate 后）门控·gate OFF→零边→逐字现状。

铁律：纯整数（ConceptRef + EDGE_REALIZES 整边·零浮点）/ 不写死（rel-type 来自外源 oracle·本函数只机制）/
  单向依赖（L4 understanding→L1 storage 向下）/ bit-identical（gate OFF 零副作用 + query_from 幂等）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import EdgeStore, EPI_STRUCTURED, SOURCE_DERIVED
from pure_integer_ai.storage.edge_types import EDGE_REALIZES
from pure_integer_ai.storage.node_store import TIER_PRIMARY

REALIZES_STRENGTH = 1   # 结构对应边·恒 1·不接 reward（M9 非学习对象·同 INSTANTIATES 纪律）


def build_realizes_edge(edge_store: EdgeStore,
                        skeleton_ref: tuple[int, int],
                        rel_type_ref: tuple[int, int],
                        *, space_id: int) -> int:
    """EDGE_REALIZES 建边（skeleton → relation-type 节点·"此结构实现此逻辑关系"·Phase D §十六-bis D.1）。

    幂等：query_from 查 skeleton 已有同 (rel_type_ref, EDGE_REALIZES) 边则 skip
    （mirror build_instantiates_edge·跨 round re-discover 不 corrupt）。自环不建（skeleton≠rel-type）。
    source=SOURCE_DERIVED（skeleton 发现派生·option-b oracle 标 IS_A·非 cue）·epistemic=EPI_STRUCTURED（honest labeled bed）。
    返建边数（0=skip/自环·1=建）。
    """
    assert_int(space_id, _where="build_realizes_edge.space_id")
    if skeleton_ref == rel_type_ref:
        return 0
    existing = edge_store.query_from(skeleton_ref[0], skeleton_ref[1], edge_type=EDGE_REALIZES)
    already = any(
        row.get("space_id_to") == rel_type_ref[0]
        and row.get("local_id_to") == rel_type_ref[1]
        for row in existing
    )
    if already:
        return 0   # 同 (skeleton→rel-type, REALIZES) 已建→skip（幂等·EdgeStore.add append-only 不去重）
    edge_store.add(
        space_id_from=skeleton_ref[0], local_id_from=skeleton_ref[1],
        space_id_to=rel_type_ref[0], local_id_to=rel_type_ref[1],
        edge_type=EDGE_REALIZES, strength=REALIZES_STRENGTH,
        source=SOURCE_DERIVED, epistemic_origin=EPI_STRUCTURED,
        order_index=None, role=None,   # REALIZES 无时序/槽位语义
        tier=TIER_PRIMARY,
    )
    return 1
