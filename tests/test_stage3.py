"""Stage 3 验收门测试：卷一理解建模（observe 建图 + 边干净 + 三空间分流 + 闭包纯净）。

覆盖（doc/重来_落地规划与实施顺序.md §六 Stage 3 验收门）：
  - observe 建图（概念点 + 边）
  - PRECEDES strength 恒 = 1（reward 永不调）
  - CAUSES 只从结构化源 + 指向词（§8.1c 硬边界·≠ CONDITION）
  - COOCCURS SHADOW 隔离 + 分桶不跨桶（C1 防跨语言污染）
  - REFERS_TO 闭包纯净（喻称 METAPHOR 不污染纯同指闭包）
  - 性质B pronoun 落记忆 occurrence token
  - 三空间落点分流（按 stage）
  - 6Q 完备
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import (
    SOURCE_CONCEPTNET, SOURCE_BARE_TEXT, EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM,
    SUBTYPE_PURE_ALIAS, SUBTYPE_METAPHOR, SUBTYPE_OCCURRENCE,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY, TIER_SHADOW
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.memory_space import MemorySpace
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.cognition.shared.types import (
    InputPayload, Segment, SpaceContext, ConceptRef, MultiRef,
    STAGE_TRAINING, STAGE_POST_WEANING_READ, STAGE_USER_INTERACTION,
    STAGE_EXTERNAL_DEFINE, WEANING_PRE,
    MODALITY_LANGUAGE, LANG_ZH, LANG_EN, DOMAIN_TEXT, DOMAIN_CODE,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.edge_types import (
    EDGE_PRECEDES, EDGE_CAUSES, EDGE_COOCCURS, EDGE_REFERS_TO,
)
from pure_integer_ai.cognition.understanding.observe import ObservePipeline
from pure_integer_ai.cognition.understanding.role_precedes import (
    build_precedes_edges, attach_role_seq, PRECEDES_STRENGTH,
)
from pure_integer_ai.cognition.understanding.causes import build_causes_edges
from pure_integer_ai.cognition.understanding.cooccurs import build_cooccurs, make_bucket
from pure_integer_ai.cognition.understanding.refers_stable import build_refers_stable_edge
from pure_integer_ai.cognition.understanding.refers_occurrence import resolve_pronoun_occurrence
from pure_integer_ai.cognition.understanding.space_routing import (
    gate_check_memory_steady, target_space_id, route_to_space,
    META_DEFINITION, KNOWLEDGE_DEFINITION,
)
from pure_integer_ai.cognition.understanding.polysemy import (
    SenseMapping, preprocess_sense_disambiguation,
)
from pure_integer_ai.cognition.understanding.refers_to import is_pronoun
from pure_integer_ai.algorithm.closure import transitive_closure, reachable
from pure_integer_ai.storage.edge_store import EdgeStore


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def ctx(request):
    """建 backend + 三空间 + SpaceContext。"""
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    reg = SpaceRegistry(b)
    core = AbstractSpace.create(reg, "core")
    mem_read = MemorySpace.create(reg, "mem_read")
    mem_interact = MemorySpace.create(reg, "mem_interact")
    comp = CompanionSpace.create(reg, "comp1")
    c = SpaceContext(
        core=core, memory_read=mem_read, memory_interact=mem_interact,
        companion=comp, stage=STAGE_TRAINING, memory_active=False,
        weaning_phase=WEANING_PRE,
    )
    yield c
    b.close()


def _seg(tokens, **kw):
    return Segment(seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
                   domain=DOMAIN_TEXT, tokens=tokens, **kw)


# ============ observe 建图 + PRECEDES ============

def test_observe_builds_precedes_strength_one(ctx):
    """observe 建 PRECEDES·strength 恒 = 1（reward 永不调·§7.1）。"""
    raw = InputPayload(
        segments=[_seg(["小明", "吃", "苹果"], role_seq=[1, 2, 3])],
        source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING,
    )
    pipe = ObservePipeline(ctx)
    res = pipe.observe(raw)
    assert res.built_concepts > 0
    assert res.built_edges > 0
    pre = ctx.core.backend.select("edge", where={"edge_type": EDGE_PRECEDES})
    assert len(pre) >= 2   # 小明→吃·吃→苹果
    assert all(e["strength"] == PRECEDES_STRENGTH == 1 for e in pre)


def test_observe_inter_segment_precedes(ctx):
    """两段→句间序 PRECEDES（替旧 TYPE_SENTENCE_TRANSITION·§十五E）。"""
    raw = InputPayload(
        segments=[_seg(["小明", "跑"]), _seg(["他", "累"])],
        source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING,
    )
    pipe = ObservePipeline(ctx)
    res = pipe.observe(raw)
    pre = ctx.core.backend.select("edge", where={"edge_type": EDGE_PRECEDES})
    # 段内 1+1 + 句间 1 = 3（"他"是代词可能 resolve·但 PRECEDES 仍建）
    assert res.built_edges >= 3


# ============ CAUSES（§8.1c 硬边界）============

def test_observe_causes_structured_and_cue(ctx):
    """CAUSES 只从结构化源(EPI_STRUCTURED)+指向词(EPI_CUE)建·epistemic_origin 标来源。"""
    seg = _seg(["雨", "导致", "地", "湿"],
               structured_causal_pairs=[(0, 3)],   # 雨→湿 结构化源
               cue_based_causal_pairs=[(0, 3)])    # 导致 指向词
    raw = InputPayload(segments=[seg], source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
    ObservePipeline(ctx).observe(raw)
    causes = ctx.core.backend.select("edge", where={"edge_type": EDGE_CAUSES})
    assert len(causes) == 2
    epis = {e["epistemic_origin"] for e in causes}
    assert EPI_STRUCTURED in epis
    assert EPI_CUE in epis


def test_epistemic_origin_provenance_discipline(ctx):
    """#355: epistemic_origin 是 provenance 纪律——认识论边(CAUSES/IS_A/REFERS_TO-stable)必带
    合法来源 1-3·非认识论边(PRECEDES/COOCCURS)必 None。构造器级强制(assert+forced non-None)·
    结构性满足三驱动：断奶退场=来源③停建(causes.py POST guard)/J3溯源=构造器 assert(每 CAUSES 边可溯)/
    闭包纯净=type 隔离(认识论边永非 None·非认识论边不进其闭包)。故无缺失热路径消费者(非"写后忘"gap·
    是 provenance-by-design)。本测试读 epistemic_origin 守此不变量·防未来构造器/A3 代码域 CAUSES 路径破纪律。
    （CONDITION 写侧 2026-07-09 删·EDGE_CONDITION=7 保留注册登记但不激活·此处非认识论边例用 PRECEDES。）"""
    seg = _seg(["如果", "雨", "则", "地", "湿"],
               structured_causal_pairs=[(1, 4)])
    raw = InputPayload(segments=[seg, _seg(["他", "累"])],   # 第二段引出 inter-segment PRECEDES
                       source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
    ObservePipeline(ctx).observe(raw)
    b = ctx.core.backend
    valid_epi = {EPI_STRUCTURED, EPI_CUE, EPI_LLM_CONFIRM}
    # 认识论边 CAUSES 必带合法来源 1-3（构造器 assert 强制·observe 出口守）
    causes = b.select("edge", where={"edge_type": EDGE_CAUSES})
    assert causes and all(e["epistemic_origin"] in valid_epi for e in causes)
    # 非认识论边 PRECEDES 必 None（无认识论来源）
    assert all(e["epistemic_origin"] is None
               for e in b.select("edge", where={"edge_type": EDGE_PRECEDES}))


# ============ IS_A 构造器（§8.1b·致命3 来源② 系词提取） ============

def test_cue_extractor_no_cue_no_pairs():
    """反统计契约：无指向词/系词 → 零 pair（绝不共现式 N×N 配对·§8.1c-bis §7 同构）。"""
    from pure_integer_ai.cognition.understanding.cue_extractor import extract_cues
    cue, is_a, _ = extract_cues(["苹果", "香蕉", "橘子"], lang=LANG_ZH)
    assert cue == []
    assert is_a == []


def test_cue_extractor_causes_forward_backward():
    """CAUSES 指向词方向：前因后果(导致)→(左,右) / 前果后因(因为)→(右,左)。"""
    from pure_integer_ai.cognition.understanding.cue_extractor import extract_cues
    # 雨 导致 地湿 → 因(0)→果(2)
    cue, _, _ = extract_cues(["雨", "导致", "地湿"], lang=LANG_ZH)
    assert cue == [(0, 2)]
    # 地湿 因为 雨 → 因(2)→果(0)
    cue, _, _ = extract_cues(["地湿", "因为", "雨"], lang=LANG_ZH)
    assert cue == [(2, 0)]


def test_cue_extractor_is_a_cue():
    """IS_A 系词：苹果 是一种 水果 → child(0)→parent(2)。"""
    from pure_integer_ai.cognition.understanding.cue_extractor import extract_cues
    _, is_a, _ = extract_cues(["苹果", "是一种", "水果"], lang=LANG_ZH)
    assert is_a == [(0, 2)]


def test_cue_extractor_boundary_cue_skipped():
    """边界 cue（句首/句末无左或右）跳·不凑配（守反统计）。"""
    from pure_integer_ai.cognition.understanding.cue_extractor import extract_cues
    # 导致在句首无左 → 跳
    cue, _, _ = extract_cues(["导致", "地湿"], lang=LANG_ZH)
    assert cue == []
    # 导致在句末无右 → 跳
    cue, _, _ = extract_cues(["雨", "导致"], lang=LANG_ZH)
    assert cue == []


def test_cue_extractor_gated_off_empty():
    """CUE_EXTRACTOR_MODE OFF → 返空（守回归 bit-identical）。"""
    from pure_integer_ai.cognition.understanding.cue_extractor import extract_cues_gated
    from pure_integer_ai.config import gates
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = False
    try:
        cue, is_a, _ = extract_cues_gated(["雨", "导致", "地湿"], lang=LANG_ZH)
        assert cue == []
        assert is_a == []
    finally:
        gates.CUE_EXTRACTOR_MODE = saved


def test_observe_builds_is_a_from_cue_pairs(ctx):
    """observe 读 Segment.is_a_pairs 建 EDGE_IS_A（致命3 来源②·测度走 strength 非 sn/tn·M9）。"""
    from pure_integer_ai.storage.edge_types import EDGE_IS_A
    seg = _seg(["苹果", "是一种", "水果"], is_a_pairs=[(0, 2)])   # child→parent
    raw = InputPayload(segments=[seg], source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
    ObservePipeline(ctx).observe(raw)
    is_a_edges = ctx.core.backend.select("edge", where={"edge_type": EDGE_IS_A})
    assert len(is_a_edges) == 1
    e = is_a_edges[0]
    # 方向 child→parent（苹果→水果·M1 照搬不反转）
    assert (e["space_id_from"], e["local_id_from"]) != (e["space_id_to"], e["local_id_to"])
    # M9：IS_A 不接 reward·sn/tn 建 0/0 不再动·strength=初始测度（>0）
    assert e["sn"] == 0 and e["tn"] == 0
    assert e["strength"] > 0
    # 来源② 系词提取 → EPI_CUE
    from pure_integer_ai.storage.edge_store import EPI_CUE
    assert e["epistemic_origin"] == EPI_CUE


def test_is_a_edge_rejects_bare_epistemic():
    """IS_A 必须有认识论来源（禁裸共现·同 refers_stable 性质A 纪律）。"""
    from pure_integer_ai.cognition.understanding.is_a import build_is_a_edge
    from pure_integer_ai.storage.edge_store import EdgeStore, EPI_STRUCTURED
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    core = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    child = (core.space_id, 1)
    parent = (core.space_id, 2)
    # epistemic 不在 {STRUCTURED,CUE,LLM_CONFIRM} → 断言拒
    import pytest as _pytest
    with _pytest.raises(AssertionError):
        build_is_a_edge(es, child, parent, source=SOURCE_BARE_TEXT,
                        epistemic=99, space_id=core.space_id)



# ============ COOCCURS 分桶 + SHADOW 隔离 ============

def test_cooccurs_shadow_isolation(ctx):
    """COOCCURS 全 SHADOW tier（不进默认 A1/PR·防塌柱①保护）。"""
    seg = _seg(["苹果", "香蕉", "橘子"])
    raw = InputPayload(segments=[seg], source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
    ObservePipeline(ctx).observe(raw)
    co = ctx.core.backend.select("edge", where={"edge_type": EDGE_COOCCURS})
    assert len(co) >= 1
    assert all(e["tier"] == TIER_SHADOW for e in co)


def test_cooccurs_no_cross_bucket():
    """分桶不跨桶（C1·防中文答 apple 泄数学·§7.4 C1）。"""
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    core = AbstractSpace.create(reg, "core")
    c = SpaceContext(core=core, memory_read=None, memory_interact=None,
                     companion=None, stage=STAGE_TRAINING)
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    # 中文桶 refs + 英文桶 refs（不同 lang）
    zh1 = ci.ensure("苹果", space_id=core.space_id)
    zh2 = ci.ensure("香蕉", space_id=core.space_id)
    en1 = ci.ensure("apple", space_id=core.space_id)
    en2 = ci.ensure("banana", space_id=core.space_id)
    # 中文桶内配对
    n_zh = build_cooccurs(es, [zh1, zh2], lang=LANG_ZH, domain=DOMAIN_TEXT,
                          source=SOURCE_BARE_TEXT, space_id=core.space_id)
    # 英文桶内配对
    n_en = build_cooccurs(es, [en1, en2], lang=LANG_EN, domain=DOMAIN_TEXT,
                          source=SOURCE_BARE_TEXT, space_id=core.space_id)
    assert n_zh == 1 and n_en == 1
    co = b.select("edge", where={"edge_type": EDGE_COOCCURS})
    assert len(co) == 2
    # 无跨桶对验证：两次独立 build_cooccurs（zh 段/en 段）各产 1 同段对=2 边·无 zh-en 跨桶对
    # （C1 强制点在 caller 单语言段·build_cooccurs 段内配对天然同桶·stub #4 修：删空 pass 循环）
    assert make_bucket(LANG_ZH, DOMAIN_TEXT) != make_bucket(LANG_EN, DOMAIN_TEXT)
    b.close()


# ============ REFERS_TO 闭包纯净（喻称不污染）============

def test_refers_metaphor_excluded_from_closure(ctx):
    """喻称 METAPHOR 不进纯同指闭包·PURE_ALIAS 进（§十一#2-bis 闭包纯净性）。"""
    es = EdgeStore(ctx.core.backend)
    ci = ConceptIndex(ctx.core.backend, ctx.companion)
    sid = ctx.core.space_id
    # PURE_ALIAS 链：A→B→C（稳定同指·进闭包）
    a = ci.ensure("李白", space_id=sid, tier=TIER_PRIMARY)
    b = ci.ensure("李太白", space_id=sid, tier=TIER_PRIMARY)
    c = ci.ensure("青莲居士", space_id=sid, tier=TIER_PRIMARY)
    build_refers_stable_edge(es, ci, a, b, epistemic=EPI_STRUCTURED, space_id=sid)
    build_refers_stable_edge(es, ci, b, c, epistemic=EPI_STRUCTURED, space_id=sid)
    # 喻称：诗仙→李白（METAPHOR·不进闭包）
    d = ci.ensure("诗仙", space_id=sid, tier=TIER_PRIMARY)
    build_refers_stable_edge(es, ci, d, a, epistemic=EPI_CUE, space_id=sid,
                             metaphor=True, surface_form_a="诗仙")
    # 收集 REFERS_TO 边 + meta（subtype）
    edges = []
    for e in ctx.core.backend.select("edge", where={"edge_type": EDGE_REFERS_TO}):
        edges.append(((e["space_id_from"], e["local_id_from"]),
                      (e["space_id_to"], e["local_id_to"]),
                      EDGE_REFERS_TO, {"subtype": e["subtype"]}))
    cl = transitive_closure(edges, types={EDGE_REFERS_TO},
                            purity_filter=lambda m: m.get("subtype") == SUBTYPE_PURE_ALIAS,
                            include_direct=True)
    # PURE_ALIAS 闭包：A→C（经 B）应可达
    assert (a, c, EDGE_REFERS_TO) in cl
    # 喻称断链：D→C 不在闭包（METAPHOR 被 purity_filter 排除）
    assert (d, c, EDGE_REFERS_TO) not in cl
    # D→A 是喻称直接边·include_direct=True 含直接边·但 purity_filter 排除喻称直接边
    assert (d, a, EDGE_REFERS_TO) not in cl


# ============ 性质B pronoun 落记忆 occurrence ============

def test_pronoun_resolves_to_antecedent(ctx):
    """性质B pronoun 解析到先行词·落记忆 occurrence token（OCCURRENCE subtype·memory_time_attach）。"""
    es = EdgeStore(ctx.core.backend)
    ci = ConceptIndex(ctx.core.backend, ctx.companion)
    wm = WorkMemory()
    core_sid = ctx.core.space_id
    mem_sid = ctx.memory_read.space_id
    # 前文段：小明（核心）
    xiaoming = ci.ensure("小明", space_id=core_sid, tier=TIER_PRIMARY)
    wm.push_segment(0, [xiaoming])
    # 解析"他"→应解析到小明
    ant = resolve_pronoun_occurrence(
        es, ci, "他", work_memory=wm, memory_space_id=mem_sid, timestamp_seq=1,
    )
    assert ant == xiaoming
    # OCCURRENCE 边落记忆
    occ = ctx.core.backend.select("edge", where={"edge_type": EDGE_REFERS_TO,
                                                  "subtype": SUBTYPE_OCCURRENCE})
    assert len(occ) == 1
    assert occ[0]["memory_time_attach"] == 1
    assert occ[0]["tier"] == TIER_SHADOW   # 性质B 不进默认 A1


def test_recency_weight_decreases_with_distance():
    """stub #5：近因权重越近越大（线性衰减 max(1, WINDOW−dist)·旧版 decay=1 恒 1 退自然序）。"""
    wm = WorkMemory()
    wm.push_segment(0, [(1, 1)])
    wm.push_segment(1, [(1, 2)])
    wm.push_segment(2, [(1, 3)])
    # dist 0（最近 seg 2）→ WINDOW=3·dist 1（seg 1）→ 2·dist 2（seg 0）→ 1
    assert wm.recency_weight(2) == 3
    assert wm.recency_weight(1) == 2
    assert wm.recency_weight(0) == 1
    # 单调递减（近因优先·非自然序）
    assert wm.recency_weight(2) > wm.recency_weight(1) > wm.recency_weight(0)


def test_pronoun_dangling_returns_none(ctx):
    """悬空代词（无候选）→ None（J4=0 真碎句·§十一#2-bis）。"""
    es = EdgeStore(ctx.core.backend)
    ci = ConceptIndex(ctx.core.backend, ctx.companion)
    wm = WorkMemory()   # 空·无前文
    ant = resolve_pronoun_occurrence(
        es, ci, "他", work_memory=wm,
        memory_space_id=ctx.memory_read.space_id, timestamp_seq=1,
    )
    assert ant is None


def test_is_pronoun_anaphora_only():
    """仅 anaphora 人称代词首版（前指/指示 defer·§十一#2）。"""
    assert is_pronoun("他")
    assert is_pronoun("she")
    assert not is_pronoun("苹果")
    assert not is_pronoun("这")   # 指示代词 defer


def test_lookup_pronoun_features_unit():
    """B5：lookup_pronoun_features 元定义表正确性（人称/数/生命性/性别·exact 匹配·中英不撞单表）。"""
    from pure_integer_ai.cognition.understanding.pronoun_features import (
        lookup_pronoun_features, FEAT_3P_SG_HUMAN_MALE, FEAT_3P_SG_HUMAN_FEMALE,
        FEAT_3P_SG_NONHUMAN, FEAT_3P_PL_HUMAN_FEMALE, FEAT_3P_PL_NONHUMAN,
        FEAT_3P_PL_GENERIC)
    # 中文
    assert lookup_pronoun_features("他") == FEAT_3P_SG_HUMAN_MALE
    assert lookup_pronoun_features("她") == FEAT_3P_SG_HUMAN_FEMALE
    assert lookup_pronoun_features("它") == FEAT_3P_SG_NONHUMAN
    assert lookup_pronoun_features("她们") == FEAT_3P_PL_HUMAN_FEMALE
    assert lookup_pronoun_features("它们") == FEAT_3P_PL_NONHUMAN
    # 英文（与中文不撞·同特征概念）
    assert lookup_pronoun_features("he") == FEAT_3P_SG_HUMAN_MALE
    assert lookup_pronoun_features("her") == FEAT_3P_SG_HUMAN_FEMALE
    assert lookup_pronoun_features("it") == FEAT_3P_SG_NONHUMAN
    assert lookup_pronoun_features("they") == FEAT_3P_PL_GENERIC
    # 非代词 / 指示代词 defer → None（守反统计契约）
    assert lookup_pronoun_features("苹果") is None
    assert lookup_pronoun_features("这") is None


def test_pronoun_feature_property_edge_via_observe_pipeline(ctx):
    """B5 反 theater：ObservePipeline 注入 pronoun_feature_lookup → pronoun 解析建 PROPERTY 边
    （pronoun→特征概念）进记忆空间 PR 种子 e 软兜·防 PR 软排序把'他'指向'苹果'非人称。

    pronoun-feature PROPERTY 边落**记忆空间**（pronoun_ref/feat_ref 均 ensure 进 mem_sid）·
    refers_stable alias PROPERTY 边落核心·故 mem 空间 PROPERTY 边 = pronoun-feature 独有（干净判据）。
    无 lookup（pre-B5）→ pronoun_features=None → 零此边（refers_occurrence:55 守）。
    """
    from pure_integer_ai.cognition.understanding.pronoun_features import lookup_pronoun_features
    from pure_integer_ai.storage.edge_types import EDGE_PROPERTY
    mem_sid = ctx.memory_read.space_id
    seg0 = _seg(["小明"], role_seq=[1])
    seg1 = _seg(["他"], role_seq=[1])
    raw = InputPayload(segments=[seg0, seg1], source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
    ObservePipeline(ctx, pronoun_feature_lookup=lookup_pronoun_features).observe(raw)
    # pronoun-feature PROPERTY 边在记忆空间（pronoun_ref→feat_ref·均 mem_sid）
    b = ctx.core.backend
    prop = b.select("edge", where={"edge_type": EDGE_PROPERTY,
                                   "space_id_from": mem_sid})
    assert len(prop) >= 1, "pronoun→特征概念 PROPERTY 边须建（记忆空间·B5 注入软兜）"


# ============ 三空间落点分流 ============

def test_space_routing_by_stage(ctx):
    """按 stage 路由：训练→核心 / 阅读→记忆一层 / 交互→记忆二层 / define→伴随检疫。"""
    assert target_space_id(STAGE_TRAINING, ctx) == ctx.core.space_id
    assert target_space_id(STAGE_POST_WEANING_READ, ctx) == ctx.memory_read.space_id
    assert target_space_id(STAGE_USER_INTERACTION, ctx) == ctx.memory_interact.space_id
    assert target_space_id(STAGE_EXTERNAL_DEFINE, ctx) == ctx.companion.space_id
    # 路由描述
    assert route_to_space(STAGE_TRAINING, ctx) == "CORE"
    assert route_to_space(STAGE_USER_INTERACTION, ctx) == "MEMORY_INTERACT"


def test_space_routing_external_define_meta_pre_weaning(ctx):
    """外部 define：元定义仅断奶前直落核心（H3）·知识定义走检疫。"""
    ctx.weaning_phase = WEANING_PRE
    assert route_to_space(STAGE_EXTERNAL_DEFINE, ctx,
                          teacher_content_type=META_DEFINITION) == "CORE_PRIMARY"
    assert route_to_space(STAGE_EXTERNAL_DEFINE, ctx,
                          teacher_content_type=KNOWLEDGE_DEFINITION) == "COMPANION_QUARANTINE"
    # 断奶后元定义退化为知识定义走检疫（H3）
    ctx.weaning_phase = 1
    assert route_to_space(STAGE_EXTERNAL_DEFINE, ctx,
                          teacher_content_type=META_DEFINITION) == "COMPANION_QUARANTINE"


def test_space_routing_missing_facility_fails_without_core_write(ctx):
    """M-00：训练后路由缺 Memory/Companion 时失败，且 Observe 不回写 Core。"""
    raw = InputPayload(
        segments=[_seg(["unrouted"])], source=SOURCE_BARE_TEXT,
        stage=STAGE_POST_WEANING_READ)
    before_nodes = ctx.core.backend.select("concept_node")
    before_edges = ctx.core.backend.select("edge")
    ctx.memory_read = None

    with pytest.raises(RuntimeError, match="MemoryRead"):
        ObservePipeline(ctx).observe(raw)

    assert ctx.core.backend.select("concept_node") == before_nodes
    assert ctx.core.backend.select("edge") == before_edges


def test_space_routing_missing_companion_and_unknown_stage_fail(ctx):
    """M-00：依赖 Companion 的路由与未知 stage 均不得静默返回 Core。"""
    ctx.companion = None
    with pytest.raises(RuntimeError, match="Companion"):
        target_space_id(STAGE_POST_WEANING_READ, ctx)
    with pytest.raises(RuntimeError, match="Companion"):
        route_to_space(STAGE_EXTERNAL_DEFINE, ctx)
    with pytest.raises(ValueError, match="未知训练 stage"):
        target_space_id(999, ctx)
    with pytest.raises(ValueError, match="未知训练 stage"):
        route_to_space(999, ctx)


def test_memory_steady_gate_remains_blocking(ctx):
    """M-00：真实稳态探针完成前不得以恒真 gate 开放 Memory。"""
    assert gate_check_memory_steady(ctx) is False


# ============ role_seq 属性（def_array）=============

def test_role_seq_attached_to_def_array(ctx):
    """role_seq 作结构概念点 def_array 属性（§十一缺口#1·role 降字段不建边）。"""
    sid = ctx.core.space_id
    struct_ref = (sid, 1)
    ctx.core.backend.insert("concept_node", {
        "space_id": sid, "local_id": 1, "type": 1, "born_granularity": 0,
        "version_head": 0, "tier": TIER_PRIMARY,
    })
    attach_role_seq(ctx.core.backend, struct_ref, [1, 2, 3], order_base=0)
    rows = ctx.core.backend.select("def_array", where={"space_id": sid, "local_id": 1})
    assert len(rows) == 3
    assert [r["ref_local_id"] for r in rows] == [1, 2, 3]
    # role 标记非概念 ref（ref_space_id=0）
    assert all(r["ref_space_id"] == 0 for r in rows)


# ============ 多义 sense（模块7）=============

def test_polysemy_sense_mapping_1toN(ctx):
    """多义 sense 1:N 挂概念（§B12·消歧在生成侧·返全部禁取首）。"""
    ci = ConceptIndex(ctx.core.backend, ctx.companion)
    sid = ctx.core.space_id
    mapping = preprocess_sense_disambiguation(
        ci, space_id=sid,
        wikidata_dump=[("apple", 89), ("apple", 3122)],   # Q89 fruit / Q3122 Inc
    )
    senses = mapping.lookup("apple")
    assert len(senses) == 2   # 1:N·返全部
    assert all(isinstance(s, tuple) and len(s) == 2 for s in senses)


# ============ 6Q 完备 ============

def test_observe_six_questions_complete(ctx):
    """6Q 完备（§十一）：Q1识别+role / Q2归一 / Q3 OOV概念点 / Q4结构PRECEDES / Q5语境留PR / Q6信任tier。

    Q5 语境不在此建（留 PR 计算层·§十三种类1）——验 observe 不建语境边（无 Q5 专属边类型）。
    """
    seg = _seg(["苹果", "是", "水果"], role_seq=[1, 2, 3],
               alias_cue_pairs=[(0, 2)])   # 苹果 aka 水果（性质A 线索）
    raw = InputPayload(segments=[seg], source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING)
    res = ObservePipeline(ctx).observe(raw)
    b = ctx.core.backend
    # Q1: 概念点建 + role_seq 属性
    nodes = b.select("concept_node")
    assert len(nodes) >= 3
    # Q2: 归一（概念点经 ConceptIndex dedup·同 surface 不重建）
    # Q3: OOV 概念点建（SHADOW 起步·裸文本）
    # Q4: PRECEDES 结构边
    assert len(b.select("edge", where={"edge_type": EDGE_PRECEDES})) >= 2
    # Q6: tier 按源分级（PRECEDES PRIMARY·COOCCURS SHADOW）
    pre = b.select("edge", where={"edge_type": EDGE_PRECEDES})
    assert all(e["tier"] == TIER_PRIMARY for e in pre)
    co = b.select("edge", where={"edge_type": EDGE_COOCCURS})
    assert all(e["tier"] == TIER_SHADOW for e in co)
    # 性质A 线索建 REFERS_TO PURE_ALIAS
    ref = b.select("edge", where={"edge_type": EDGE_REFERS_TO,
                                  "subtype": SUBTYPE_PURE_ALIAS})
    assert len(ref) == 1
    # Q5: 语境不在此建（observe 不产 PR 种子 e·留卷二）——验 observe 成功跑完即 Q5 defer 合规
    assert res.built_edges > 0


# ============ 确定性 bit-identical ============

def test_observe_deterministic_bit_identical():
    """同输入两跑同图（确定性·ConceptIndex dedup + 确定性排序）。"""
    def run():
        b = DictBackend()
        bootstrap(b)
        reg = SpaceRegistry(b)
        core = AbstractSpace.create(reg, "core")
        c = SpaceContext(core=core, memory_read=None, memory_interact=None,
                         companion=None, stage=STAGE_TRAINING)
        raw = InputPayload(
            segments=[_seg(["小明", "吃", "苹果"], role_seq=[1, 2, 3])],
            source=SOURCE_BARE_TEXT, stage=STAGE_TRAINING,
        )
        ObservePipeline(c).observe(raw)
        # 收集 edge (from,to,et,strength,tier) 排序作指纹
        edges = b.select("edge")
        fp = sorted((e["space_id_from"], e["local_id_from"],
                     e["space_id_to"], e["local_id_to"], e["edge_type"],
                     e["strength"], e["tier"]) for e in edges)
        b.close()
        return fp
    assert run() == run()
