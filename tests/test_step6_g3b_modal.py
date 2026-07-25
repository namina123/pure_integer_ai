"""tests.test_step6_g3b_modal — B2 G3b 模态对当扩展 + 命题 identity 结构化测试套件（STEP6 PR3）。

P0.3 modality 进命题 surface·PR2 填 modality 值·PR3 扩 G3b 判模态对当矛盾：
  - 命题 identity 结构存（ATTR_PROP_SUBJ=19/ATTR_PROP_ATTR=20/ATTR_PROP_POLMOD=21·解 ref→surface defer）
  - graph_view.iter_proposition_identity → [(prop_ref, subj, attr, pol, mod)]
  - judge.counterfactual_value_check 层a-extended 模态对当（T 公理形式层·非 #479 truth·gate MODALITY_MODE）

模态方阵（_modal_contradiction·T 公理 □p→p + □>◇ + Ought(p)→¬May(¬p)）：
  矛盾：□p+¬p / □p+◇¬p / □p+□¬p / 道义必然p+道义可能¬p / 道义必然p+道义必然¬p
  不矛盾：◇p+◇¬p / 断言p+断言¬p（B1 对立）/ 断言p+◇¬p / 跨风味 / □p 单独

诚实边界：T 公理形式层墙内（构造性检查·非 truth·情态比命题多一口气）·实质情态真值 W2/W1 defer·
bit-identical（MODALITY_MODE OFF → modality=0 → 无 □ claim → 模态检查 inert → 既有层a）。
"""
from __future__ import annotations

import pytest
from pure_integer_ai.config import gates
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import LANG_ZH, ConceptRef
from pure_integer_ai.cognition.understanding.property import build_property_edges
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.judge import (
    counterfactual_value_check, _modal_contradiction,
)


@pytest.fixture
def prop_env():
    """build_property_edges + G3b 单测环境（dict backend·core space·composes_attr 已注册·同 test_g1_proposition）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    yield b, sid, es, ci
    b.close()


@pytest.fixture
def modality_on():
    """翻 MODALITY_MODE ON·测后复位（G3b 模态对当激活）。"""
    saved = gates.MODALITY_MODE
    gates.MODALITY_MODE = True
    yield
    gates.MODALITY_MODE = saved


def _ensure(ci, sid, surface):
    return ci.ensure(surface, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)


# ============ 件① _modal_contradiction 单元（T 公理模态方阵） ============

def test_modal_contradiction_box_p_and_neg_p():
    """□p(mod=1,pol=0) + ¬p(mod=0,pol=1) → True（T 公理 □p→p·□p 与 ¬p 矛盾）。"""
    assert _modal_contradiction([(0, 1), (1, 0)]) is True


def test_modal_contradiction_box_p_and_possibly_not_p():
    """□p(mod=1,pol=0) + ◇¬p(mod=2,pol=1) → True（□>◇·□p→¬◇¬p·矛盾）。"""
    assert _modal_contradiction([(0, 1), (1, 2)]) is True


def test_modal_contradiction_box_p_and_box_not_p():
    """□p(mod=1,pol=0) + □¬p(mod=1,pol=1) → True（两 □ 对立·矛盾）。"""
    assert _modal_contradiction([(0, 1), (1, 1)]) is True


def test_modal_contradiction_deontic_necessity_and_possibility():
    """道义必然p(mod=3,pol=0) + 道义可能¬p(mod=4,pol=1) → True（Ought(p)+May(¬p)·矛盾）。"""
    assert _modal_contradiction([(0, 3), (1, 4)]) is True


def test_modal_contradiction_deontic_necessity_both():
    """道义必然p(mod=3,pol=0) + 道义必然¬p(mod=3,pol=1) → True（两 Ought 对立·矛盾）。"""
    assert _modal_contradiction([(0, 3), (1, 3)]) is True


def test_modal_contradiction_possibly_both_no_contradiction():
    """◇p(mod=2,pol=0) + ◇¬p(mod=2,pol=1) → False（两可能·兼容·非矛盾）。"""
    assert _modal_contradiction([(0, 2), (1, 2)]) is False


def test_modal_contradiction_assertoric_opposition_no_contradiction():
    """断言p(mod=0,pol=0) + 断言¬p(mod=0,pol=1) → False（B1 对立·无 □·非矛盾）。"""
    assert _modal_contradiction([(0, 0), (1, 0)]) is False


def test_modal_contradiction_assertoric_and_possibly_no_contradiction():
    """断言p(mod=0,pol=0) + ◇¬p(mod=2,pol=1) → False（◇¬p 与 p 兼容·无 □·非矛盾）。"""
    assert _modal_contradiction([(0, 0), (1, 2)]) is False


def test_modal_contradiction_cross_flavor_no_contradiction():
    """跨风味：□p(mod=1,pol=0·认识) + 道义必然¬p(mod=3,pol=1·道义) → False（不同风味·不矛盾）。"""
    assert _modal_contradiction([(0, 1), (1, 3)]) is False


def test_modal_contradiction_box_p_alone_no_contradiction():
    """□p(mod=1,pol=0) 单独 → False（无反极性·非矛盾）。"""
    assert _modal_contradiction([(0, 1)]) is False


def test_modal_contradiction_empty_no_contradiction():
    """空 claims → False（无 claim·非矛盾）。"""
    assert _modal_contradiction([]) is False


def test_modal_contradiction_box_p_and_box_p_same_pol_no_contradiction():
    """两 □p(mod=1,pol=0) 同极性 → False（同 claim 重复·非矛盾）。"""
    assert _modal_contradiction([(0, 1), (0, 1)]) is False


def test_modal_contradiction_box_not_p_and_p_symmetric():
    """□¬p(mod=1,pol=1) + p(mod=0,pol=0) → True（对称·T 公理 □¬p→¬p·与 p 矛盾）。"""
    assert _modal_contradiction([(1, 1), (0, 0)]) is True


def test_modal_contradiction_box_p_and_possibly_p_same_pol_no_contradiction():
    """□p(mod=1,pol=0) + ◇p(mod=2,pol=0) 同极性 → False（□p 蕴涵 ◇p·兼容·非矛盾）。"""
    assert _modal_contradiction([(0, 1), (0, 2)]) is False


# ============ 件② 命题 identity 结构存（ATTR_PROP_*） ============

def test_proposition_identity_recorded(prop_env):
    """build_property_edges 建命题节点 → record ATTR_PROP_SUBJ/ATTR_PROP_ATTR/ATTR_PROP_POLMOD（identity 结构存）。"""
    from pure_integer_ai.storage.composes_attr import read_composes_attrs, ATTR_PROP_SUBJ, ATTR_PROP_ATTR, ATTR_PROP_POLMOD
    b, sid, es, ci = prop_env
    猫, 颜色, 黑 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑")
    build_property_edges(es, ci, b, [猫, 颜色, 黑],
                         property_claims=[(0, 1, 2, 0, 0, 1)],   # mod=1 必然
                         source=SOURCE_BARE_TEXT, space_id=sid)
    prop_ref = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}_0_1", sid)
    assert prop_ref is not None
    attrs = read_composes_attrs(b, prop_ref)
    assert attrs.get(ATTR_PROP_SUBJ) == (猫[0], 猫[1]), "ATTR_PROP_SUBJ 存 subj ref"
    assert attrs.get(ATTR_PROP_ATTR) == (颜色[0], 颜色[1]), "ATTR_PROP_ATTR 存 attr ref"
    assert attrs.get(ATTR_PROP_POLMOD) == (0, 1), "ATTR_PROP_POLMOD 存 (pol=0, mod=1)"


def test_iter_proposition_identity_returns_groups(prop_env):
    """iter_proposition_identity → [(prop_ref, subj, attr, pol, mod)]（G3b 模态对当分组用）。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 黑, 白 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑"), _ensure(ci, sid, "白")
    # 同(subj,attr) 不同 modality：□p(mod=1) + ◇¬p(mod=2,pol=1)
    build_property_edges(es, ci, b, [猫, 颜色, 黑, 白],
                         property_claims=[(0, 1, 2, 0, 0, 1),   # 猫的颜色必然是黑·mod=1 pol=0
                                          (0, 1, 3, 0, 1, 2)],  # 猫的颜色可能[不]是白·mod=2 pol=1
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    identity = graph.iter_proposition_identity()
    assert len(identity) == 2, "两命题节点（mod=1/mod=2 异 surface）"
    # 同 (subj,attr) 分组
    subjs = {(subj, attr) for _prop, subj, attr, _pol, _mod in identity}
    assert subjs == {(猫, 颜色)}, "两节点同 (subj,attr)·异 (pol,mod)·模态对当分组基础"


# ============ 件③ G3b 模态对当 e2e（counterfactual_value_check·gate MODALITY_MODE） ============

def test_g3b_modal_box_p_and_neg_p_contradicted(prop_env, modality_on):
    """e2e：□p(必然是黑) + ¬p(不是黑) 同(subj,attr,val) → G3b 返 0（T 公理矛盾·层a-extended）。

    同 value=黑·异 (pol,mod) → 跨节点同 (subj,attr,val) 分组·□(黑)+¬(黑) → T 公理矛盾。
    """
    b, sid, es, ci = prop_env
    猫, 颜色, 黑 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑")
    build_property_edges(es, ci, b, [猫, 颜色, 黑],
                         property_claims=[(0, 1, 2, 0, 0, 1),   # 必然是黑·mod=1 pol=0
                                          (0, 1, 2, 0, 1, 0)],  # 不是黑·mod=0 pol=1（同 value=黑）
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 0, "□(黑) + ¬(黑) 同值 → T 公理矛盾 → G3b 返 0"


def test_g3b_modal_box_p_and_neg_different_value_no_contradiction(prop_env, modality_on):
    """★对抗审 catch：□(黑) + ¬(白) 异值 → G3b 返 1（不同命题·兼容·黑蕴涵非白·非矛盾）。

    value 维度防假阳性：分组 key 含 val·□(黑)∈group(黑) / ¬(白)∈group(白)→不同组→不矛盾。
    无 val 分组会假阳性 over-veto（□(黑)+¬(白) 误判矛盾）。
    """
    b, sid, es, ci = prop_env
    猫, 颜色, 黑, 白 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑"), _ensure(ci, sid, "白")
    build_property_edges(es, ci, b, [猫, 颜色, 黑, 白],
                         property_claims=[(0, 1, 2, 0, 0, 1),   # 必然是黑·mod=1 pol=0
                                          (0, 1, 3, 0, 1, 0)],  # 不是白·mod=0 pol=1（异 value=白）
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 1, (
        "□(黑)+¬(白) 异值 → 不同命题·兼容 → G3b 返 1（value 维度防假阳性·非 over-veto）")


def test_g3b_modal_possibly_both_no_contradiction(prop_env, modality_on):
    """e2e：◇p(可能是黑) + ◇¬p(可能不是白) 同(subj,attr) → G3b 返 1（两可能·兼容·非矛盾）。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 黑, 白 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑"), _ensure(ci, sid, "白")
    build_property_edges(es, ci, b, [猫, 颜色, 黑, 白],
                         property_claims=[(0, 1, 2, 0, 0, 2),   # 可能是黑·mod=2 pol=0
                                          (0, 1, 3, 0, 1, 2)],  # 可能不是白·mod=2 pol=1
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 1, "◇p + ◇¬p → 兼容 → G3b 返 1"


def test_g3b_modal_assertoric_opposition_no_contradiction(prop_env, modality_on):
    """e2e：断言p(是黑) + 断言¬p(不是白) 同(subj,attr) → G3b 返 1（B1 对立·无 □·非矛盾）。

    核证 B1 不破坏：断言对立（mod=0）非矛盾·须有 □(mod=1) 才触发 T 公理矛盾。
    """
    b, sid, es, ci = prop_env
    猫, 颜色, 黑, 白 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑"), _ensure(ci, sid, "白")
    build_property_edges(es, ci, b, [猫, 颜色, 黑, 白],
                         property_claims=[(0, 1, 2, 0, 0, 0),   # 是黑·mod=0 pol=0
                                          (0, 1, 3, 0, 1, 0)],  # 不是白·mod=0 pol=1
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 1, "断言p+断言¬p → B1 对立·非矛盾 → G3b 返 1"


def test_g3b_modal_box_p_and_box_not_p_contradicted(prop_env, modality_on):
    """e2e：□p(必然是黑) + □¬p(必然不是黑) 同(subj,attr,val) → G3b 返 0（两 □ 对立·矛盾）。

    注：两 claim 同 value=黑·异 pol/mod·异命题节点（surface _0_1 vs _1_1）·跨节点 □p+□¬p → 矛盾。
    """
    b, sid, es, ci = prop_env
    猫, 颜色, 黑 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑")
    build_property_edges(es, ci, b, [猫, 颜色, 黑],
                         property_claims=[(0, 1, 2, 0, 0, 1),   # 必然是黑·mod=1 pol=0
                                          (0, 1, 2, 0, 1, 1)],  # 必然不是黑·mod=1 pol=1
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 0, "□p + □¬p → 两 □ 对立 → G3b 返 0"


def test_g3b_modal_deontic_contradiction(prop_env, modality_on):
    """e2e：道义必然p(必须是黑) + 道义可能¬p(可以不是黑) 同(subj,attr,val) → G3b 返 0（Ought+May¬p 矛盾）。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 黑 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑")
    build_property_edges(es, ci, b, [猫, 颜色, 黑],
                         property_claims=[(0, 1, 2, 0, 0, 3),   # 必须是黑·mod=3 pol=0
                                          (0, 1, 2, 0, 1, 4)],  # 可以不是黑·mod=4 pol=1（同 value=黑）
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 0, "道义必然(黑)+道义可能¬(黑) 同值 → 矛盾 → G3b 返 0"


def test_g3b_modal_cross_flavor_no_contradiction(prop_env, modality_on):
    """e2e：跨风味 □p(认识必然是黑) + 道义必然¬p(必须不是白) 同(subj,attr) → G3b 返 1（跨风味不矛盾）。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 黑, 白 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑"), _ensure(ci, sid, "白")
    build_property_edges(es, ci, b, [猫, 颜色, 黑, 白],
                         property_claims=[(0, 1, 2, 0, 0, 1),   # 必然是黑·mod=1 pol=0（认识）
                                          (0, 1, 3, 0, 1, 3)],  # 必须不是白·mod=3 pol=1（道义）
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 1, "跨风味 → 不矛盾 → G3b 返 1"


# ============ 件④ bit-identical（MODALITY_MODE OFF → 模态检查 inert·既有层a） ============

def test_g3b_modal_gate_off_inert_bit_identical(prop_env):
    """★bit-identical：MODALITY_MODE OFF + □p+¬p 同(subj,attr) → G3b 返 1（模态检查不跑·既有层a·bit-identical）。

    gate OFF → modality 对当检查不跑（_modal_contradiction_in_graph 不调）→ □p+¬p 跨节点不判矛盾 → 返 1。
    同(subj,attr) 异(pol,mod) 各 1 value → 层a 多值不触发 → 返 1（既有行为零变）。
    """
    saved = gates.MODALITY_MODE
    gates.MODALITY_MODE = False
    try:
        b, sid, es, ci = prop_env
        猫, 颜色, 黑, 白 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑"), _ensure(ci, sid, "白")
        build_property_edges(es, ci, b, [猫, 颜色, 黑, 白],
                             property_claims=[(0, 1, 2, 0, 0, 1),   # 必然是黑·mod=1 pol=0
                                              (0, 1, 3, 0, 1, 0)],  # 不是白·mod=0 pol=1
                             source=SOURCE_BARE_TEXT, space_id=sid)
        graph = ConceptGraph(b)
        assert counterfactual_value_check(None, None, graph) == 1, (
            "MODALITY_MODE OFF → 模态检查 inert → □p+¬p 不判矛盾 → 返 1（bit-identical·既有层a）")
    finally:
        gates.MODALITY_MODE = saved


def test_g3b_layer_a_multi_value_still_works(prop_env, modality_on):
    """G3b 既有层a 多值矛盾不受 PR3 影响：同命题节点（同 pol/mod）多 value → G3b 返 0。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 黑, 白 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑"), _ensure(ci, sid, "白")
    # 同命题节点（pol=0/mod=0）两 value（黑+白）→ 层a 多值矛盾
    build_property_edges(es, ci, b, [猫, 颜色, 黑, 白],
                         property_claims=[(0, 1, 2, 0, 0, 0, 1, 1),   # 猫的颜色是黑·mod=0 pol=0·#1134 intensity 1/1 default
                                          (0, 1, 3, 0, 0, 0, 1, 1)],  # 猫的颜色是白·mod=0 pol=0（同节点多值）·intensity 1/1
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 0, "同节点多 value → 层a 多值矛盾 → G3b 返 0（既有·PR3 不影响）"


# ============ 件⑤ 守墙标注（T 公理形式层·非 truth） ============

def test_g3b_modal_wall_annotation_formal_not_truth():
    """守墙：G3b 模态对当是 T 公理形式层（构造性检查·非 truth）·实质情态真值 W2/W1 defer。

    核证 _modal_contradiction_in_graph docstring 标注"T 公理形式层墙内"+"实质情态真值 defer"（诚实边界）。
    """
    import inspect
    import pure_integer_ai.cognition.result.judge as judge
    src = inspect.getsource(judge._modal_contradiction_in_graph)
    assert "T 公理" in src or "形式层" in src, "_modal_contradiction_in_graph 须标 T 公理形式层墙内"
    assert "W2" in src or "W1" in src or "defer" in src, "须标实质情态真值 W2/W1 defer"
    assert "truth" in src or "#479" in src, "须标非 truth / #479 墙"
