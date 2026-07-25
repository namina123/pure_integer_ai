"""Phase E §十八 condition 3 测试（exposure-driven CONCEPT-leaf 固定位检测·dormant 基建）。

detect_fixed_concept_positions：给定结构同构样本簇·DFS 阅读序收 CONCEPT-leaf token·按位对齐·
跨样本同 token 一致 → 固定位（闭类原语候选·是/使 可分）·否则 → 变量位（PARAM content）。

反 theater：exposure-driven·无 frozenset·无 word→type 映射·"fixed" 纯跨样本 token-identity 统计涌现。
对齐：DFS 阅读序镜像 _SkeletonBuilder.build _concept_slot_idx（DAG _seen 复用 + 子序 range）。

铁律：纯整数 / 确定性 bit-identical（mode 唯一判·不依赖 Counter tie-break）/ 不写死 / 纯读无写·无 production caller。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT, EPI_STRUCTURED
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.composes_attr import record_composes_attr, ATTR_OPERATOR
from pure_integer_ai.cognition.process.fixed_position_detect import (
    detect_fixed_concept_positions, FixedConceptPosition, MIN_DETECT_SAMPLES,
)
from tests.test_experiments import make_train_context


def _ensure(ctx, surface):
    return ctx.concept_index.ensure(surface, space_id=ctx.space_id)


def _build_token_tree(ctx, root, tokens):
    """root（ATTR_OPERATOR·internal）→ token concept leaves（无 attr=CONCEPT_LEAF·reading 序 order_index）。"""
    record_composes_attr(ctx.backend, ref=root, kind=ATTR_OPERATOR, int_a=0, int_b=0)
    for ti, tok in enumerate(tokens):
        ctx.edge_store.add(
            space_id_from=root[0], local_id_from=root[1],
            space_id_to=tok[0], local_id_to=tok[1],
            edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED, order_index=ti, tier=TIER_PRIMARY,
        )


# ============ 核心：unanimous 固定位识别 ============

def test_detect_unanimous_fixed_cue():
    """★ 是 同位跨样本一致 → slot1 固定位（闭类原语候选）·content 词（苹果/猫·水果/动物）变量位。

    [苹果 是 水果] + [猫 是 动物] → slot0 var·slot1 FIXED(是)·slot2 var。
    """
    ctx = make_train_context(DictBackend())
    shi = _ensure(ctx, "是")
    r1 = _ensure(ctx, "__disc_isA_1")
    r2 = _ensure(ctx, "__disc_isA_2")
    _build_token_tree(ctx, r1, [_ensure(ctx, "苹果"), shi, _ensure(ctx, "水果")])
    _build_token_tree(ctx, r2, [_ensure(ctx, "猫"), shi, _ensure(ctx, "动物")])
    pos = detect_fixed_concept_positions(ctx.concept_graph, [r1, r2])
    assert len(pos) == 3, "3 CONCEPT-leaf 位"
    assert [p.is_fixed for p in pos] == [False, True, False], "★ slot1 是 固定位·slot0/2 content 变量位"
    assert pos[1].token_ref == shi, "固定位 token = 是（ConceptRef）"
    assert pos[1].agreement == 2 and pos[1].total == 2, "unanimous·2/2 同意"
    assert pos[0].token_ref is None and pos[2].token_ref is None, "变量位 token_ref=None"


def test_detect_distinguishes_cues():
    """★ 是 vs 使 同位异 token → slot1 变量位（无固定位）= 是/使 仍可经 token 身份区分（(c) wiring 反馈分组后）。

    [苹果 是 水果] + [火 使 水] → 全变量位（是/使 在 slot1 异·无 unanimous mode）。
    本检测器单独不分离 是/使 簇（须 (c) wiring 固定位 token 身份入分组键）·但正确报 slot1 非 unanimous。
    """
    ctx = make_train_context(DictBackend())
    r1 = _ensure(ctx, "__disc_a")
    r2 = _ensure(ctx, "__disc_b")
    _build_token_tree(ctx, r1, [_ensure(ctx, "苹果"), _ensure(ctx, "是"), _ensure(ctx, "水果")])
    _build_token_tree(ctx, r2, [_ensure(ctx, "火"), _ensure(ctx, "使"), _ensure(ctx, "水")])
    pos = detect_fixed_concept_positions(ctx.concept_graph, [r1, r2])
    assert [p.is_fixed for p in pos] == [False, False, False], "是/使 异→slot1 非 unanimous→变量位（dormant 诚实）"


# ============ 边界：样本数 / spine 同构 ============

def test_detect_single_sample_empty():
    """单样本（< MIN_DETECT_SAMPLES）→ []（无跨样本一致性·caller fallback 全参化=现状）。"""
    ctx = make_train_context(DictBackend())
    r1 = _ensure(ctx, "__disc_solo")
    _build_token_tree(ctx, r1, [_ensure(ctx, "苹果"), _ensure(ctx, "是"), _ensure(ctx, "水果")])
    assert detect_fixed_concept_positions(ctx.concept_graph, [r1]) == [], "单样本→[]"
    assert detect_fixed_concept_positions(ctx.concept_graph, []) == [], "空→[]"
    assert MIN_DETECT_SAMPLES == 2


def test_detect_inconsistent_spine_none():
    """CONCEPT-leaf 序异长（spine 非同构·discover_skeleton 同构门不应放过·防御）→ None。"""
    ctx = make_train_context(DictBackend())
    r1 = _ensure(ctx, "__disc_3")
    r2 = _ensure(ctx, "__disc_4")
    _build_token_tree(ctx, r1, [_ensure(ctx, "苹果"), _ensure(ctx, "是"), _ensure(ctx, "水果")])            # 3 叶
    _build_token_tree(ctx, r2, [_ensure(ctx, "苹果"), _ensure(ctx, "是"), _ensure(ctx, "甜"), _ensure(ctx, "的")])  # 4 叶
    assert detect_fixed_concept_positions(ctx.concept_graph, [r1, r2]) is None, "spine 异长→None（防御）"


# ============ DAG 对齐（镜像 build() _seen 复用） ============

def test_detect_dag_shared_concept_leaf_once():
    """★ DAG 共享 concept leaf（"猫 是 猫"两猫同 ref）首遇记一次 = 一槽（镜像 build():388 _seen 复用）。

    [猫 是 猫] + [狗 是 狗] → 序 [猫,是] + [狗,是]（共享叶去重）·slot0 var·slot1 FIXED(是)。
    若检测器不去重（记 [猫,是,猫]）则 slot2 会是 phantom 固定位（错）·本测钉死去重对齐。
    """
    ctx = make_train_context(DictBackend())
    mao = _ensure(ctx, "猫"); shi = _ensure(ctx, "是"); gou = _ensure(ctx, "狗")
    r1 = _ensure(ctx, "__disc_dag1")
    r2 = _ensure(ctx, "__disc_dag2")
    _build_token_tree(ctx, r1, [mao, shi, mao])   # 猫 出现两次（同 ref·DAG 共享）
    _build_token_tree(ctx, r2, [gou, shi, gou])
    pos = detect_fixed_concept_positions(ctx.concept_graph, [r1, r2])
    assert len(pos) == 2, "★ DAG 共享去重→2 位（猫/是）非 3（phantom slot2 消除）"
    assert [p.is_fixed for p in pos] == [False, True], "slot0(猫/狗)var·slot1(是)FIXED"
    assert pos[1].token_ref == shi


# ============ 阈值参数 + 确定性 ============

def test_detect_threshold_relaxed():
    """min_agreement_count 放宽：是 2/3 + 使 1/3·unanimous(3) → 变量·阈值 2 → FIXED(是)。"""
    ctx = make_train_context(DictBackend())
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    r1 = _ensure(ctx, "__t1"); r2 = _ensure(ctx, "__t2"); r3 = _ensure(ctx, "__t3")
    _build_token_tree(ctx, r1, [_ensure(ctx, "苹果"), shi])
    _build_token_tree(ctx, r2, [_ensure(ctx, "猫"), shi])
    _build_token_tree(ctx, r3, [_ensure(ctx, "火"), shi3])
    pos_unan = detect_fixed_concept_positions(ctx.concept_graph, [r1, r2, r3])
    assert pos_unan[1].is_fixed is False, "unanimous(3)·是 2/3 → 变量位"
    pos_relax = detect_fixed_concept_positions(ctx.concept_graph, [r1, r2, r3], min_agreement_count=2)
    assert pos_relax[1].is_fixed is True, "阈值 2·是 2/3 唯一 mode → FIXED"
    assert pos_relax[1].token_ref == shi
    assert pos_relax[1].agreement == 2 and pos_relax[1].total == 3


def test_detect_tie_not_fixed():
    """mode 并列（是 2/2 出现在两 token... 不可能·同位 2 样本各一 token）→ 用 3 样本 是是使不可 tie。
    实测 tie：2 样本 slot0 = 是/使（各 1·max=1·两 mode）→ 非 fixed（歧义）。"""
    ctx = make_train_context(DictBackend())
    r1 = _ensure(ctx, "__x1"); r2 = _ensure(ctx, "__x2")
    _build_token_tree(ctx, r1, [_ensure(ctx, "是")])
    _build_token_tree(ctx, r2, [_ensure(ctx, "使")])
    pos = detect_fixed_concept_positions(ctx.concept_graph, [r1, r2])
    assert pos[0].is_fixed is False, "tie（是/使 各 1·两 mode）→ 非 fixed（确定性·不依赖 Counter tie-break）"


def test_detect_deterministic_bit_identical():
    """两次调同输入 → 逐字同结果（确定性·bit-identical 守卫）。"""
    ctx = make_train_context(DictBackend())
    r1 = _ensure(ctx, "__d1"); r2 = _ensure(ctx, "__d2")
    _build_token_tree(ctx, r1, [_ensure(ctx, "苹果"), _ensure(ctx, "是"), _ensure(ctx, "水果")])
    _build_token_tree(ctx, r2, [_ensure(ctx, "猫"), _ensure(ctx, "是"), _ensure(ctx, "动物")])
    a = detect_fixed_concept_positions(ctx.concept_graph, [r1, r2])
    b = detect_fixed_concept_positions(ctx.concept_graph, [r1, r2])
    assert a == b, "确定性：两次调逐字同"
    assert all(isinstance(p, FixedConceptPosition) for p in a)


def test_detect_no_concept_leaves():
    """纯算子/立即数树（无 CONCEPT leaf）→ []（length=0·无固定位可检）。"""
    ctx = make_train_context(DictBackend())
    # root 仅 ATTR_OPERATOR 无子（无 CONCEPT leaf）
    r1 = _ensure(ctx, "__op1")
    r2 = _ensure(ctx, "__op2")
    record_composes_attr(ctx.backend, ref=r1, kind=ATTR_OPERATOR, int_a=0, int_b=0)
    record_composes_attr(ctx.backend, ref=r2, kind=ATTR_OPERATOR, int_a=0, int_b=0)
    assert detect_fixed_concept_positions(ctx.concept_graph, [r1, r2]) == [], "无 CONCEPT leaf→[]"
