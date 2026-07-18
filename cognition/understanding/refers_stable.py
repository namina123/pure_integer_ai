"""cognition.understanding.refers_stable — 模块4 REFERS_TO 性质A 稳定同指（含喻称拆三层）。

性质A = 稳定定义性同指（apple↔苹果·李白↔李太白·纯任意能指）→ 核心稳定边·无衰减·进纯同指闭包。
来源（§8.1c-bis 同解形状不同失败形状）：
  ① 结构化源（Wikidata QID per-concept / WordNet synset / lemmatizer 形态归一）
  ② 语言线索 + 句法（同位语/也称/又名/即/aka/定义式）
  ③ 断奶前 LLM 教师（多 sense 消歧带 context）
  ④ 传递闭包（前提边类型纯净）

**喻称拆三层**（§十一#2-bis M8·最硬·污染闭包）：
  组合性喻称（诗仙↔李白）不能落 flat REFERS_TO——落 flat 会闭包推出诗仙↔李太白语义错位
  （§8.1c-bis 闭包失败模式B 翻版）+ 丢属性语义。判定 = 表层非组合（结构判·墙内）+ 组合义不对
  所指断言属性（语义判·#479 W2 truth 墙·**非 D 物理接地**）。组合性 → 拆三层：范畴节点入核心 + PROPERTY/IS_A 边 + 喻称
  REFERS_TO 子类型（METAPHOR·不进纯同指闭包）。

裸共现禁直接落边（§十一#2-bis·过粗）·staging 候选落伴随 sign=0 非此。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import (
    EdgeStore, DEFAULT_STRENGTH, EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM,
    SUBTYPE_PURE_ALIAS, SUBTYPE_METAPHOR,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.cognition.shared.edge_types import EDGE_REFERS_TO, EDGE_PROPERTY, EDGE_IS_A
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex

# 喻称判定：组合义是否对所指断言属性 = 语义判 #479 W2 truth 墙（**非 D 物理接地**）。首版**保守**：默认非组合（PURE_ALIAS），
# 仅当调用方显式标 metaphor=True（预处理/教师标注）时拆三层。
#
# **B6 诚实降级（2026-07-03·design-未决 C 类·非纯 theater）**：
# _build_metaphor_three_layer 机制已实现 + test_stage3:318 测（metaphor=True+"诗仙"验三层）·
# **但生产零 caller 触发**（observe:215 / refers_to:89 全 metaphor=False 默认 PURE_ALIAS）→
# 喻称边生产永不建。原 docstring 称"断奶后新遇无标注 → 保守落三层"是 **over-claim 未落 code**。
# 保守默认须**设计决断**（非盲接·防 over-trigger 真 alias 失闭包）：
#   ① weaning-phase-aware 默认（pre=PURE_ALIAS 教师标 / post=conservative 三层）OR gate 控制
#     （METAPHOR_CONSERVATIVE_MODE default OFF·生产 post-weaning 翻·守 bit-identical）
#   ② detect_surface_composition 墙内结构判 hook（表层非组合=喻称候选·**未实现**）——
#     无此 hook 的保守默认是盲三层（post-weaning 全 unannotated→三层·真 alias 亦失闭包·代价）
#   ③ 保守 vs PURE_ALIAS 权衡（误落 flat=闭包语义错位 worst / 盲三层=真 alias 失闭包 less bad）
# 随喻称保守策略设计 pass 落（同 cue_extractor 元定义 + weaning gate 决断范式）。


def build_refers_stable_edge(edge_store: EdgeStore, concept_index: ConceptIndex,
                             a: tuple[int, int], b: tuple[int, int],
                             *, epistemic: int, space_id: int,
                             metaphor: bool = False,
                             surface_form_a: str | None = None) -> int:
    """性质A 稳定同指建边。

    epistemic ∈ {EPI_STRUCTURED(①), EPI_CUE(②), EPI_LLM_CONFIRM(③)}——性质A 必须有认识论来源·禁裸共现。
    metaphor=True → 喻称拆三层（范畴节点 + PROPERTY 边 + 喻称子类型不进闭包）。
    metaphor=False → 纯任意能指·PURE_ALIAS·进纯同指闭包。
    返回建边数。
    """
    assert epistemic in (EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM), \
        "性质A 必须有认识论来源·禁裸共现"
    if a == b:
        return 0
    if metaphor:
        return _build_metaphor_three_layer(edge_store, concept_index, a, b,
                                           epistemic=epistemic, space_id=space_id,
                                           surface_form_a=surface_form_a)
    # 纯任意能指 → 稳定 REFERS_TO·PURE_ALIAS·进纯同指闭包
    edge_store.add(
        space_id_from=a[0], local_id_from=a[1],
        space_id_to=b[0], local_id_to=b[1],
        edge_type=EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS,
        strength=DEFAULT_STRENGTH, source=6,  # SOURCE_DERIVED
        epistemic_origin=epistemic, tier=TIER_PRIMARY,
    )
    return 1


def _build_metaphor_three_layer(edge_store: EdgeStore, concept_index: ConceptIndex,
                                a: tuple[int, int], b: tuple[int, int],
                                *, epistemic: int, space_id: int,
                                surface_form_a: str | None) -> int:
    """喻称拆三层（§十一#2-bis M8）。

    ① 范畴节点入核心（surface_form_a 的概念点）
    ② 属性边保留语义（PROPERTY 或 IS_A·a → 范畴节点）
    ③ 喻称 REFERS_TO 子类型（METAPHOR·不进纯同指闭包·closure_dispatch 跳过）
    """
    n = 0
    # ① 范畴节点（surface_form_a 的概念点·a 的字面范畴）
    if surface_form_a is not None:
        cat_ref = concept_index.ensure(surface_form_a, space_id=space_id,
                                       tier=TIER_PRIMARY)
        # ② 属性边 a → 范畴节点（保留语义·不丢属性）
        edge_store.add(
            space_id_from=a[0], local_id_from=a[1],
            space_id_to=cat_ref[0], local_id_to=cat_ref[1],
            edge_type=EDGE_PROPERTY, strength=DEFAULT_STRENGTH,
            source=6, epistemic_origin=epistemic, tier=TIER_PRIMARY,
        )
        n += 1
    # ③ 喻称 REFERS_TO 子类型·不进纯同指闭包
    edge_store.add(
        space_id_from=a[0], local_id_from=a[1],
        space_id_to=b[0], local_id_to=b[1],
        edge_type=EDGE_REFERS_TO, subtype=SUBTYPE_METAPHOR,
        strength=DEFAULT_STRENGTH, source=6,
        epistemic_origin=epistemic, tier=TIER_PRIMARY,
    )
    n += 1
    return n
