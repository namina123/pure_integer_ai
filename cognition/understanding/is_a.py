"""cognition.understanding.is_a — IS_A proper subset 建边（§8.1b·致命3 构造器·D1 落盘）。

IS_A = proper subset (A⊂B) typed 边（§8.1b 砍相等变体·共指归 REFERS_TO）。
构造器 = 把合法来源落成 EDGE_IS_A 边（建边接线·非"算关系"·测度定位见 §8.1a）。

**来源分层（§8.1c-bis 同构·epistemic_origin 守认识论来源·禁裸共现）**：
  ① 结构化源（ConceptNet IsA 有向三元组·child IsA parent·照搬不反转·M1）→ EPI_STRUCTURED
  ② 系词 + 句法位置提取（cue_extractor·"X 是一种 Y"·紧邻 token）→ EPI_CUE
  ③ 断奶前 LLM 教师确认（teacher.confirm_is_a·歧义如番茄水果/蔬菜）→ EPI_LLM_CONFIRM
  ④ proper subset 传递闭包（algorithm/closure types={EDGE_IS_A}·派生不存·非此构造器）
  ⑤ staging（伴随 sign=0·非此构造器）

**测度编码（M9·doc:1421）**：IS_A **不接 reward 反传**（effective_weight:82 assert 只认
  {PRECEDES,CAUSES,REFERS_TO}·IS_A 不在内）。测度走**初始 strength + promote/tier 升**·非 sn/tn：
  - 定义性来源（元定义/三角形⊂多边形）→ initial_strength = IS_A_STRENGTH_DEFINITIONAL（测度1）
  - 经验来源（ConceptNet 苹果 IsA 水果）→ initial_strength = IS_A_STRENGTH_EMPIRICAL（测度1−ε）
  - sn/tn 建 0/0·不再动（IS_A 非学习对象·删 sn/tn/is_unobserved 类比）。

**方向（§8.1b point2·M1）**：from=child(A/子集) to=parent(B/超集)·ConceptNet IsA = child IsA parent
  照搬不反转。M1② 方向校验首版靠来源方向标注（结构化源带方向）/ 句法规则（系词提取）·
  跨源冲突检测 defer 给闭包纯净性（C9-bis 闭包隔离）。

**PR 邻接安全**：IS_A 是结构边（非 PR 邻接头·用静态 base_strength·不调 effective_weight）·
  dag_path.py:82 head_types={PRECEDES,CAUSES} + a3_pr_wrapper.py:74 双过滤挡 IS_A 入 PR·无须改 effective_weight。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import (
    EdgeStore, EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM, SOURCE_CONCEPTNET,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_types import EDGE_IS_A
from pure_integer_ai.cognition.process.effective_weight import RATE_SCALE

# IS_A 初始测度（M9·strength 非 sn/tn·oracle 标 meta 常量·非硬编码语义规则）
# 定义性 IS_A（测度1·逻辑事实·三角形⊂多边形）= RATE_SCALE(1000)
IS_A_STRENGTH_DEFINITIONAL = RATE_SCALE
# 经验 IS_A（测度1−ε·苹果⊂水果·允许零测度反例·新观察可打破）= 900（初值·oracle 标可调）
IS_A_STRENGTH_EMPIRICAL = 900


def build_is_a_edge(edge_store: EdgeStore,
                    child: tuple[int, int], parent: tuple[int, int],
                    *, source: int, epistemic: int, space_id: int,
                    initial_strength: int = IS_A_STRENGTH_EMPIRICAL) -> int:
    """IS_A proper subset 建边（child A⊂B parent·from=child to=parent）。

    epistemic ∈ {EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM}（认识论来源·禁裸共现）。
    initial_strength：初始测度（M9·定义性=IS_A_STRENGTH_DEFINITIONAL / 经验=EMPIRICAL·
      默认经验测度1−ε·结构化源 caller 按源类型传）。
    自环不建（child≠parent）。返建边数。
    """
    assert epistemic in (EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM), \
        "IS_A 必须有认识论来源·禁裸共现"
    assert_int(initial_strength, _where="build_is_a_edge.initial_strength")
    if child == parent:
        return 0
    edge_store.add(
        space_id_from=child[0], local_id_from=child[1],
        space_id_to=parent[0], local_id_to=parent[1],
        edge_type=EDGE_IS_A, strength=initial_strength,
        source=source, epistemic_origin=epistemic,
        order_index=None, role=None,   # IS_A 无 order_index 时序语义
        tier=TIER_PRIMARY,
    )
    return 1


def has_reverse_isa_edge(edge_store: EdgeStore, child: tuple[int, int],
                         parent: tuple[int, int]) -> bool:
    """查反向 IS_A 边 (parent→child) 是否存在（#1115 §14 双向矛盾检测）。

    proper subset 偏序不允许 A⊂B ∧ B⊂A（A≠B 矛盾）。建 (child→parent) 前查反向 (parent→child)：
    query_from(parent) 返 from=parent 的 IS_A 边·查 to==child（反向边存在 ⟺ parent IsA child）。
    build_is_a_edges（写 backend）+ observe hoist apply（写 ancestor_map）**共用此 helper**·
    两处必须 sync——否则 build skip 了 backend 边·apply 仍 apply 原始 parsed.is_a_pairs → 环检测
    命中 → hoist None（实测 n=50 130s 复发·build_is_a_edges skip 无效因 apply 不同步）。

    返 True = 反向已存在（双向矛盾·caller 须 skip）。纯读 L0·单向依赖向下。
    """
    _reverse = edge_store.query_from(parent[0], parent[1], edge_type=EDGE_IS_A)
    return any(row.get("space_id_to") == child[0] and row.get("local_id_to") == child[1]
               for row in _reverse)


def build_is_a_edges(edge_store: EdgeStore, refs: list[tuple[int, int]],
                     *, is_a_pairs: list[tuple[int, int]],
                     source: int, space_id: int,
                     initial_strength: int = IS_A_STRENGTH_EMPIRICAL) -> int:
    """IS_A 批量建边（observe 调·接 Segment.is_a_pairs·来源② 系词提取）。

    is_a_pairs：(child_idx, parent_idx) token index 对·经 refs 解析为概念 ref。
    source/initial_strength 由 caller 按来源定（来源② = SOURCE_BARE_TEXT / 经验测度）。
    返建边数。

    **#1115 §14 双向矛盾源头去重**：建 (child→parent) 前查反向边 (parent→child)·有则 skip
    （proper subset 偏序不允许 A⊂B ∧ B⊂A·A≠B 数学矛盾）。observe 系词（来源② EPI_CUE）可能
    提取与既有（boot 结构化源① / 前序 observe）反向矛盾的边 → 落盘成环 → ancestor_map SCC 凝聚
    → hoist 增量环检测失效（apply_isa_edge_to_map 返 False → 整 space 退化全量重建·n=50 实测 127s）。

    **skip 语义 = first-observed direction wins**（时序+认识论启发式·非图论判定）：boot/前序先建者胜·
    observe 后到反向者弃。图论无法判矛盾对边哪向"对"（只知矛盾）·X 选先到方向（boot 结构化源①
    EPI_STRUCTURED 可信 > observe 系词源② EPI_CUE 噪声）·被弃边不落盘无 silent corruption。接受
    "弃可能对的 observe 边"风险换 DAG 不变量（ancestor_map 无环·hoist 增量全程生效）。

    **bit-identical**：CI default（CUE_EXTRACTOR_MODE OFF）is_a_pairs=[] → 循环 0 次 → 反向查不触发
    → 逐字现状。gate ON 既有测试逐个核证 SAFE（test_existential_proof reversed e2e 唯一反向场景·
    reward 走 build_isa_ancestor_map_external 外部图·observe EPI_CUE 边本被双 filter 滤·X 前后同）。
    boot 残留环（ConceptNet raw 噪声 / 跨源）X 不拦·由 §7 apply_isa_edge_to_map 兜底（hoist 失效但
    正确性守·退化 gen-cache SCC）·详 doc §14.6。
    """
    n = 0
    for ci, pi in is_a_pairs:
        if ci < 0 or ci >= len(refs) or pi < 0 or pi >= len(refs):
            continue   # 越界 index 跳（守确定性·切片重映射错时保守跳）
        child, parent = refs[ci], refs[pi]
        # #1115 §14：双向 IS_A 矛盾源头去重（first-observed wins·保偏序 DAG·免环致 hoist 退化）。
        # build_is_a_edges（写 backend）+ observe apply block（写 hoist）共用 has_reverse_isa_edge·
        # 两处必须 sync（否则 build skip 了 backend 边·apply 仍 apply → 环检测命中 → hoist None）。
        if has_reverse_isa_edge(edge_store, child, parent):
            continue   # 反向 (parent IsA child) 已存在 → 双向矛盾·skip 此边
        n += build_is_a_edge(edge_store, child, parent,
                             source=source, epistemic=EPI_CUE, space_id=space_id,
                             initial_strength=initial_strength)
    return n


def bootstrap_is_a_edges(concept_index, edge_store: EdgeStore,
                         surface_pairs: list[tuple[str, str]],
                         *, space_id: int,
                         source: int = SOURCE_CONCEPTNET,
                         epistemic: int = EPI_STRUCTURED,
                         initial_strength: int = IS_A_STRENGTH_EMPIRICAL) -> int:
    """IS_A 批量 boot 种边（刀0·surface pairs → ensure → build·解锁 Interp2 生产路径）。

    与 `build_is_a_edges`（observe caller·token index 对·EPI_CUE）的差异：
      - 入参是 surface 文本对（非 token index）·caller 不依赖语料 token 切片（boot 时种边·早于 observe）。
      - 默认来源① ConceptNet（source=SOURCE_CONCEPTNET·epistemic=EPI_STRUCTURED·is_a.py:54 assert 白名单）。
      - **幂等 skip 按源细化**：query_from 查同 (child,parent,EDGE_IS_A,source) 已建则 skip（不挡 observe
        EPI_CUE 路径·同源同三元组才 skip·镜像 formal_train.py:1358 幂等范式）。

    **无文件零副作用硬守（bit-identical·P0）**：surface_pairs 空 → 立即 return 0·**绝不调
    concept_index.ensure / query_from / build**（无 ZERO_AI_LOCAL_DIR → resolve 返 [] → 图与不接刀0 bit-identical）。

    每 (child_surface, parent_surface)：concept_index.ensure 两 ref（TIER_PRIMARY·NODE_CONCEPT）→
    query_from 幂等 skip → build_is_a_edge（child==parent 跳·:57 已守）。返建边数。

    **用途**（刀0）：formal_train boot 段（make_train_context 后·lang 发现前）调·让 ancestor_map 非空 →
    S3 第二刀 Interp2 LCA 聚类真火（当前机制活数据空·冷启动 no-op）·held-out "狐狸追鸡"（狐狸⊂动物）
    命中动物类骨架（当前词级零交集必拒）。详 doc/重来_学习放开整合设计_纠偏纠偏.md §5 刀0。

    铁律：纯整数（ConceptRef + EDGE_ISA 整边·零浮点）/ 不写死（surface 来自外部文件·本函数只机制非语义）/
      §8.1c（来源① EPI_STRUCTURED 合规·is_a.py:54 assert 允许）/ bit-identical（空 pairs 零副作用 + query_from 幂等）。
    诚实边界：surface 真伪 = 外部数据责任（接地墙）·IS_A 不接 reward（effective_weight:82 assert 只认
      PRECEDES/CAUSES/REFERS_TO·IS_A 不内·boot 边无 reward 误判风险）。
    """
    if not surface_pairs:
        return 0   # P0·无文件零副作用硬守（不调 ensure/query_from/build·CI/生产 default bit-identical）
    assert_int(space_id, source, initial_strength, _where="bootstrap_is_a_edges.args")
    n = 0
    for child_surf, parent_surf in surface_pairs:
        child_ref = concept_index.ensure(
            child_surf, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        parent_ref = concept_index.ensure(
            parent_surf, space_id=space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        # 幂等 skip（按源细化·不挡 observe EPI_CUE 路径·风险4）：query_from 查 child 已有同 (parent,source) IS_A 边。
        existing = edge_store.query_from(child_ref[0], child_ref[1], edge_type=EDGE_IS_A)
        already = any(
            row.get("space_id_to") == parent_ref[0]
            and row.get("local_id_to") == parent_ref[1]
            and row.get("source") == source
            for row in existing
        )
        if already:
            continue   # 同源同三元组已建→skip（幂等·resume 跨 run / 重复 boot 不 corrupt·EdgeStore.add 不去重）
        n += build_is_a_edge(edge_store, child_ref, parent_ref,
                             source=source, epistemic=epistemic, space_id=space_id,
                             initial_strength=initial_strength)
    return n
