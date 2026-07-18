"""cognition.shared.relation_primitives — 关系原语 first-class NODE_CONCEPT（L0 元定义层·刀3 件1）。

关系作 first-class NODE_CONCEPT 节点（⊂/∈/= /因果/mereology/属性/类似）·为件2 涌现学习/件5 选择倾向/
件8 词→概念提供晋升目标框架。D:11 EDGE_RELATION_SIGNAL 连词→关系概念·ATTR_RELATION_PRIMITIVE=10 标记。

**元定义层固化·非语义规则**（§九铁律承认 enum 例外·同 cue_words / OPCODE_* / ORIGIN_*·reward 不调·
断奶前后不变）。REL_* 是关系类型的元定义命名空间·非语义规则。§8.8：种关系概念 = 非层次链复活·
非 META_* 复活·与 σ代数+typed edges 三腿并存（关系概念是被 typed edge D:11 引用的 first-class 节点·非替代）。

位于 cognition/shared（L0）·import storage.composes_attr/storage.node_store 跨层向下合规·非 re-export
（同 cognition/shared/edge_types.py 范式）。

铁律：纯整数（REL_* int/ConceptRef 整/D:11 整）/ 确定性（稳定 surface hash bit-identical）/
  不写死（REL_* enum=meta定义例外·非语义规则）/ 单向依赖（L0 依赖 storage 向下）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.composes_attr import record_composes_attr, ATTR_RELATION_PRIMITIVE
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT

# ---- REL_* 枚举（关系原语类型·meta定义·刀3 件1） ----
REL_SUBSET = 1      # ⊂ proper subset（A⊂B·IS_A 同义·类属）
REL_MEMBER = 2      # ∈ 成员（元素属于集合·mereology 的 membership 面）
REL_EQUAL = 3       # = 相等（共指归 REFERS_TO·本原语留框架）
REL_CAUSES = 4      # 因果（CAUSES 同义·因果域）
REL_PRECEDES = 5    # 前驱（时序/序·PRECEDES 同义）
REL_MEREOLOGY = 6   # 部分-整体（part-of·mereology）
REL_PROPERTY = 7    # 属性（has-property·PROPERTY 同义）
REL_SIMILAR = 8     # 相似（类似·非等同）

# 稳定 surface（content_hash dedup·跨 run identity·bit-identical）
_REL_SURFACE: dict[int, str] = {
    REL_SUBSET: "__REL_SUBSET__",
    REL_MEMBER: "__REL_MEMBER__",
    REL_EQUAL: "__REL_EQUAL__",
    REL_CAUSES: "__REL_CAUSES__",
    REL_PRECEDES: "__REL_PRECEDES__",
    REL_MEREOLOGY: "__REL_MEREOLOGY__",
    REL_PROPERTY: "__REL_PROPERTY__",
    REL_SIMILAR: "__REL_SIMILAR__",
}


def ensure_relation_primitives(concept_index, backend: StorageBackend, *,
                               space_id: int) -> dict[int, tuple[int, int]]:
    """ensure 全部 REL_* first-class NODE_CONCEPT 节点 + ATTR_RELATION_PRIMITIVE=10 标记。

    每 REL_*：concept_index.ensure(REL_SURFACE[kind], NODE_CONCEPT, TIER_PRIMARY) → ref
    + record_composes_attr(backend, ref, kind=ATTR_RELATION_PRIMITIVE, int_a=kind)。
    返 {rel_kind: ConceptRef}（caller record_word_concept 用·target 解析）。

    **幂等**（ConceptIndex.ensure 同 hash 返既有 tier 单调升 + record_composes_attr 同 (ref,kind) skip）→
    每 boot 调安全（resume 跨 run / 重复 boot 不 corrupt）。

    与刀0 bootstrap_is_a_edges 的差异：本函数**无条件 ensure 全部原语**（元定义层常驻·类 OPCODE_*）·
    非"有数据才 ensure"。刀3 关系原语是 first-class 框架节点·应常驻（boot 种 D:11 边前先建目标）。

    backend 显式传（镜像刀0 bootstrap_is_a_edges 接 concept_index+edge_store 范式·record_composes_attr
    需 backend·不触 ConceptIndex 私有 _b）。
    """
    assert_int(space_id, _where="ensure_relation_primitives.space_id")
    out: dict[int, tuple[int, int]] = {}
    for kind, surface in _REL_SURFACE.items():
        ref = concept_index.ensure(surface, space_id=space_id,
                                   tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        record_composes_attr(backend, ref=ref,
                             kind=ATTR_RELATION_PRIMITIVE, int_a=kind)
        out[kind] = ref
    return out
