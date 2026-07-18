"""cognition.understanding.mereology — MEREOLOGY 部分-整体 建边（T-L1d·客观序 gap 补·EDGE_MEREOLOGY=25）。

MEREOLOGY = part-of typed 边（X 是 Y 的一部分·part→whole 有向）。**异 IS_A**（child→parent proper subset）：
部分-整体≠子集（车轮是汽车的一部分·非车轮⊂汽车·车轮不是一种汽车）。首版 cue_words 误把 REL_MEREOLOGY
折入 IS_A_CUE（`_REL_KIND_TO_CUE_TYPE[REL_MEREOLOGY]=IS_A_CUE`·首版简化）→ gate ON 时 部分-整体文本
被建成 IsA 边=语义误路由（doc/重来_语言域断奶客观序_2026-07-15 §三 T-L1d"潜在语义误路由"）。本构造器
+ EDGE_MEREOLOGY=25 + cue 路由修正（REL_MEREOLOGY→MEREOLOGY_CUE）解此误路由·独立 typed edge 守语义正交。

构造器 = 把合法来源落成 EDGE_MEREOLOGY 边（建边接线·非"算关系"·镜像 is_a.py 范式）。

**来源分层（§8.1c-bis 同构·epistemic_origin 守认识论来源·禁裸共现）**：
  ① 结构化源（ConceptNet PartOf / WordNet meronym 有向三元组·part part-of whole·照搬不反转·M1）→ EPI_STRUCTURED
  ② 部分 cue + 句法位置提取（defer·X 是 Y 的一部分·observe-time·boot loader 是首版主路径·同 is_a/causes）
  ③ 断奶前 LLM 教师确认（teacher·歧义·defer）→ EPI_LLM_CONFIRM
  ④ mereology part-of 预序闭包（A part-of B ∧ B part-of C ⊢ A part-of C·**Phase B §十四-bis done**·
     build_mereology_ancestor_map_external·abstraction.py·镜像 IS_A external·本构造器只建直边）

**测度编码（M9·镜像 IS_A）**：MEREOLOGY **不接 reward 反传**（effective_weight:82 assert 只认
  {PRECEDES,CAUSES,REFERS_TO}·MEREOLOGY 不在内·同 IS_A）。测度走**初始 strength 静态**·非 sn/tn：
  - 经验来源（ConceptNet 车轮 part-of 汽车）→ initial_strength = MEREOLOGY_STRENGTH_EMPIRICAL（测度1−ε·同 IS_A）
  - sn/tn 建 0/0·不再动（MEREOLOGY 非学习对象·镜像 IS_A 非学习对象纪律）。

**方向（M1）**：from=part(A/部分) to=whole(B/整体)·ConceptNet PartOf = part PartOf whole 照搬不反转。

**PR 邻接安全（镜像 IS_A）**：MEREOLOGY 是结构边（非 PR 邻接头·用静态 base_strength·不调 effective_weight）·
  dag_path head_types={PRECEDES,CAUSES} + a3_pr_wrapper 双过滤挡 MEREOLOGY 入 PR·effective_weight:82 assert
  拒 MEREOLOGY（同 IS_A=10 不在 {PRECEDES,CAUSES,REFERS_TO}）·无须改 effective_weight。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import (
    EdgeStore, EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM, SOURCE_CONCEPTNET,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_types import EDGE_MEREOLOGY

# MEREOLOGY 初始测度（M9·strength 非 sn/tn·镜像 IS_A·oracle 标 meta 常量·非硬编码语义规则）
# 经验 MEREOLOGY（测度1−ε·车轮 part-of 汽车·ConceptNet·允许零测度反例·新观察可打破）= 900（同 IS_A 经验·初值可调）
# 无 DEFINITIONAL 变体（异 IS_A 三角形⊂多边形逻辑定义性·mereology 无经典逻辑定义性·全经验）→ 不引 RATE_SCALE
MEREOLOGY_STRENGTH_EMPIRICAL = 900


def build_mereology_edge(edge_store: EdgeStore,
                         part: tuple[int, int], whole: tuple[int, int],
                         *, source: int, epistemic: int, space_id: int,
                         initial_strength: int = MEREOLOGY_STRENGTH_EMPIRICAL) -> int:
    """MEREOLOGY part-of 建边（part A part-of whole B·from=part to=whole）。

    epistemic ∈ {EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM}（认识论来源·禁裸共现·镜像 build_is_a_edge:54）。
    initial_strength：初始测度（M9·静态·默认经验测度1−ε）。
    自环不建（part≠whole·同 build_is_a_edge:57）。返建边数。
    """
    assert epistemic in (EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM), \
        "MEREOLOGY 必须有认识论来源·禁裸共现"
    assert_int(initial_strength, _where="build_mereology_edge.initial_strength")
    if part == whole:
        return 0
    edge_store.add(
        space_id_from=part[0], local_id_from=part[1],
        space_id_to=whole[0], local_id_to=whole[1],
        edge_type=EDGE_MEREOLOGY, strength=initial_strength,
        source=source, epistemic_origin=epistemic,
        order_index=None, role=None,   # MEREOLOGY 无 order_index 时序语义（同 IS_A）
        tier=TIER_PRIMARY,
    )
    return 1


def bootstrap_mereology_edges(concept_index, edge_store: EdgeStore,
                              surface_pairs: list[tuple[str, str]],
                              *, space_id: int,
                              source: int = SOURCE_CONCEPTNET,
                              epistemic: int = EPI_STRUCTURED,
                              initial_strength: int = MEREOLOGY_STRENGTH_EMPIRICAL) -> int:
    """MEREOLOGY 批量 boot 种边（T-L1d·surface pairs → ensure → build·镜像 bootstrap_is_a_edges 范式）。

    入参是 surface 文本对（part_surface, whole_surface）·caller 不依赖语料 token 切片（boot 时种边·早于 observe）。
    默认来源① ConceptNet PartOf（source=SOURCE_CONCEPTNET·epistemic=EPI_STRUCTURED）。
    **幂等 skip 按源细化**（镜像 bootstrap_is_a_edges:128-137）：query_from 查同 (part,whole,EDGE_MEREOLOGY,source)
    已建则 skip。

    **无文件零副作用硬守（bit-identical·P0·镜像 bootstrap_is_a_edges:119-120）**：surface_pairs 空 → 立即 return 0·
    **绝不调 concept_index.ensure / query_from / build**（无 ZERO_AI_LOCAL_DIR → resolve 返 [] → 图与不接 T-L1d bit-identical）。

    每 (part_surface, whole_surface)：concept_index.ensure 两 ref（TIER_PRIMARY·NODE_CONCEPT）→
    query_from 幂等 skip → build_mereology_edge（part==whole 跳·:58 已守）。返建边数。

    **用途**（T-L1d）：formal_train boot 段调·让 mereology 边 boot 时种（外部 ConceptNet PartOf 源·E10·EPI_STRUCTURED）。
    闭包 defer（mereology 传递闭包须独立设计·closure 首版仅 EDGE_IS_A）。

    铁律：纯整数（ConceptRef + EDGE_MEREOLOGY 整边·零浮点）/ 不写死（surface 来自外部文件·本函数只机制非语义）/
      §8.1c（来源① EPI_STRUCTURED 合规）/ bit-identical（空 pairs 零副作用 + query_from 幂等）。
    诚实边界：surface 真伪 = 外部数据责任（接地墙）·MEREOLOGY 不接 reward（effective_weight:82 assert 只认
      PRECEDES/CAUSES/REFERS_TO·MEREOLOGY 不内·同 IS_A·boot 边无 reward 误判风险）。
    """
    if not surface_pairs:
        return 0   # P0·无文件零副作用硬守（不调 ensure/query_from/build·CI/生产 default bit-identical）
    assert_int(space_id, source, initial_strength, _where="bootstrap_mereology_edges.args")
    n = 0
    for part_surf, whole_surf in surface_pairs:
        part_ref = concept_index.ensure(
            part_surf, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        whole_ref = concept_index.ensure(
            whole_surf, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        # 幂等 skip（按源细化·镜像 bootstrap_is_a_edges:128-137）：query_from 查 part 已有同 (whole,source) MEREOLOGY 边。
        existing = edge_store.query_from(part_ref[0], part_ref[1], edge_type=EDGE_MEREOLOGY)
        already = any(
            row.get("space_id_to") == whole_ref[0]
            and row.get("local_id_to") == whole_ref[1]
            and row.get("source") == source
            for row in existing
        )
        if already:
            continue   # 同源同三元组已建→skip（幂等·resume 跨 run / 重复 boot 不 corrupt·EdgeStore.add 不去重）
        n += build_mereology_edge(edge_store, part_ref, whole_ref,
                                  source=source, epistemic=epistemic, space_id=space_id,
                                  initial_strength=initial_strength)
    return n
