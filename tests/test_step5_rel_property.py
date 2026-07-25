"""STEP5 PR3 REL_PROPERTY possess un-defer 测试（任务 #897）。

REL_PROPERTY 词级种子（具有/has·closed-class）+ is_property_possess_cue D:11 readback 第二源 +
build_property_edges default_attr_ref un-defer（REL_PROPERTY 作默认 attr_type·领属命题补身份）。
消费者 = G3b counterfactual_value_check（live·扫命题节点 PROPERTY 出边判同(subject,attr_type)多值矛盾）。

**反 theater 两路证**：
  (i) seeded closed-class '具有'（_PROPERTY_POSSESS_CUE frozenset 既有 + _REL_LEXICAL_CUE D:11 种子）：
      gate ON/OFF frozenset 第一源均命中（D:11 冗余但建边）
  (ii) fixture 注入开放变体 '拥有'→REL_PROPERTY D:11 边（SOURCE_TEACHER 模拟教师晋升）：
       gate ON is_property_possess_cue True + extract_property_claims 产 (0,-1,2,...) +
       build_property_edges(default_attr_ref) 建命题节点 + PROPERTY 边·gate OFF False + default_attr_ref=None skip

**不写死守卫**（D6）：_REL_LEXICAL_CUE 只 closed-class 核心（具有/has）·开放变体（拥有/possesses）零硬编码。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_TEACHER, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import (
    register_composes_attr, read_composes_attrs, ATTR_PROPOSITION,
)
from pure_integer_ai.storage.edge_types import EDGE_PROPERTY
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
from pure_integer_ai.cognition.shared.relation_primitives import (
    ensure_relation_primitives, REL_PROPERTY,
)
from pure_integer_ai.cognition.understanding.cue_words import (
    is_property_possess_cue, _PROPERTY_POSSESS_CUE,
)
from pure_integer_ai.cognition.understanding.word_concept_signal import (
    bootstrap_word_concept_signals, record_word_concept, _REL_LEXICAL_CUE,
)
from pure_integer_ai.cognition.understanding.property import build_property_edges
from pure_integer_ai.cognition.understanding.cue_extractor import extract_property_claims
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def prop_env():
    """PR3 单测环境（dict backend·core space·composes_attr·boot 种 REL_* D:11 边含 具有）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    bootstrap_word_concept_signals(ci, es, b, space_id=sid, langs={LANG_ZH, LANG_EN})
    yield b, sid, es, ns, ci
    b.close()


def _inject_rel_d11(ci, es, b, sid, word_surface, rel_kind):
    """fixture 注入 D:11 边（word→REL_* ref·SOURCE_TEACHER·模拟教师晋升·非 frozenset）。"""
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    rel_ref = rel_refs[rel_kind]
    return record_word_concept(ci, es, word_surface, rel_ref,
                               space_id=sid, source=SOURCE_TEACHER)


# ============ build_property_edges default_attr_ref un-defer ============

def test_build_property_edges_possess_undefer_builds_node(prop_env):
    """possess claim (attr_idx<0) + default_attr_ref → 建命题节点 + PROPERTY 边（un-defer·G3b 消费输入）。
    '猫 具有 黑色' → claim (0,-1,2,0,0,0) → default_attr_ref=REL_PROPERTY → 命题节点 __prop_{猫}_{REL_PROPERTY}。"""
    b, sid, es, ns, ci = prop_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    rel_prop_ref = rel_refs[REL_PROPERTY]
    subj = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    val = ci.ensure("黑色", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    refs = [subj, None, val]   # refs[0]=subj·refs[2]=val（refs[1] 占位·attr_idx=-1 不用）
    # possess claim: (subj_idx=0, attr_idx=-1, val_idx=2, reserved=0, pol=0, mod=0)
    claims = [(0, -1, 2, 0, 0, 0, 1, 1)]
    n = build_property_edges(es, ci, b, refs, property_claims=claims,
                             source=SOURCE_BARE_TEXT, space_id=sid,
                             default_attr_ref=rel_prop_ref)
    assert n == 1, "possess un-defer 建 PROPERTY 边"
    # 命题节点 __prop_{猫}_{REL_PROPERTY} 建出 + ATTR_PROPOSITION 标记
    prop_surface = f"__prop_{subj[0]}_{subj[1]}_{rel_prop_ref[0]}_{rel_prop_ref[1]}"
    prop_ref = ci.lookup(prop_surface, sid)
    assert prop_ref is not None, "possess 命题节点建出（REL_PROPERTY 作默认 attr_type）"
    attrs = read_composes_attrs(b, prop_ref)
    assert ATTR_PROPOSITION in attrs, "命题节点 ATTR_PROPOSITION=11 标记"
    # PROPERTY 出边 命题节点→黑色（G3b 消费输入）
    out = es.query_from(prop_ref[0], prop_ref[1], edge_type=EDGE_PROPERTY)
    assert any(r["space_id_to"] == val[0] and r["local_id_to"] == val[1] for r in out), \
        "PROPERTY 边 命题节点→黑色（G3b 扫读输入）"


def test_build_property_edges_default_attr_none_skip(prop_env):
    """default_attr_ref=None + possess claim (attr_idx<0) → skip（既有行为 bit-identical）。"""
    b, sid, es, ns, ci = prop_env
    subj = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    val = ci.ensure("黑色", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    refs = [subj, None, val]
    claims = [(0, -1, 2, 0, 0, 0, 1, 1)]
    n = build_property_edges(es, ci, b, refs, property_claims=claims,
                             source=SOURCE_BARE_TEXT, space_id=sid,
                             default_attr_ref=None)   # 既有行为
    assert n == 0, "default_attr_ref=None·possess skip（既有 bit-identical）"


def test_build_property_edges_attr_type_window_unchanged(prop_env):
    """的...是 窗口（attr_idx>=0）不受 default_attr_ref 影响（既有命题身份完整·bit-identical）。"""
    b, sid, es, ns, ci = prop_env
    subj = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    attr = ci.ensure("颜色", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    val = ci.ensure("黑", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    refs = [subj, attr, val]
    claims = [(0, 1, 2, 0, 0, 0, 1, 1)]   # attr_idx=1（的...是 窗口）
    n_with = build_property_edges(es, ci, b, refs, property_claims=claims,
                                  source=SOURCE_BARE_TEXT, space_id=sid,
                                  default_attr_ref=ci.ensure("__REL_PROPERTY__", space_id=sid,
                                                             tier=TIER_PRIMARY, node_type=NODE_CONCEPT))
    # 的...是 窗口 attr_idx>=0 → 用 refs[1]=attr（颜色）·非 default_attr_ref·命题节点 __prop_{猫}_{颜色}
    prop_surface = f"__prop_{subj[0]}_{subj[1]}_{attr[0]}_{attr[1]}"
    assert ci.lookup(prop_surface, sid) is not None, "的...是 窗口用 attr_type=颜色（非 default_attr_ref）"


# ============ (i) seeded closed-class '具有' ============

def test_possess_cue_seeded_frozenset_first_source(prop_env):
    """(i) seeded '具有' gate ON/OFF 均 True（_PROPERTY_POSSESS_CUE frozenset 第一源·D:11 冗余但建边）。"""
    b, sid, es, ns, ci = prop_env
    saved = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = False
    try:
        assert is_property_possess_cue("具有", LANG_ZH, backend=b, edge_store=es,
                                       space_id=sid, concept_index=ci) is True
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    try:
        assert is_property_possess_cue("具有", LANG_ZH, backend=b, edge_store=es,
                                       space_id=sid, concept_index=ci) is True
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved


# ============ (ii) fixture 注入开放变体 '拥有'（D:11 唯一源·反 theater） ============

def test_possess_cue_d11_readback_nontournament_word(prop_env):
    """(ii) fixture 注入 '拥有'→REL_PROPERTY D:11 边→gate ON is_property_possess_cue True·gate OFF False。"""
    b, sid, es, ns, ci = prop_env
    _inject_rel_d11(ci, es, b, sid, "拥有", REL_PROPERTY)
    saved = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = False
    try:
        assert is_property_possess_cue("拥有", LANG_ZH, backend=b, edge_store=es,
                                       space_id=sid, concept_index=ci) is False, \
            "gate OFF '拥有' 非 frozenset·readback 关→False"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    try:
        assert is_property_possess_cue("拥有", LANG_ZH, backend=b, edge_store=es,
                                       space_id=sid, concept_index=ci) is True, \
            "gate ON D:11 readback '拥有'→REL_PROPERTY→True"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved


def test_possess_d11_drives_property_build(prop_env):
    """(ii) 端到端：'拥有' 经 D:11 readback 驱动 extract_property_claims 产 possess claim +
    build_property_edges(default_attr_ref) 建命题节点。'猫 拥有 黑色' → claim (0,-1,2,0,0,0) → 命题节点。"""
    b, sid, es, ns, ci = prop_env
    _inject_rel_d11(ci, es, b, sid, "拥有", REL_PROPERTY)
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    rel_prop_ref = rel_refs[REL_PROPERTY]
    # ensure 词概念（extract_property_claims 不 ensure·只返 idx·caller 须先 ensure）
    ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    ci.ensure("拥有", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    ci.ensure("黑色", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    tokens = ["猫", "拥有", "黑色"]
    saved = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    try:
        claims_on = extract_property_claims(tokens, lang=LANG_ZH,
                                            backend=b, edge_store=es,
                                            space_id=sid, concept_index=ci)
        assert (0, -1, 2, 0, 0, 0, 1, 1) in claims_on, \
            "gate ON '拥有' 经 D:11 readback→possess claim (0,-1,2,0,0,0)"
        # build：default_attr_ref=REL_PROPERTY → 命题节点建出
        refs = [ci.lookup("猫", sid), None, ci.lookup("黑色", sid)]
        n = build_property_edges(es, ci, b, refs, property_claims=claims_on,
                                 source=SOURCE_BARE_TEXT, space_id=sid,
                                 default_attr_ref=rel_prop_ref)
        assert n >= 1, "possess un-defer 建命题节点 PROPERTY 边"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved
    # gate OFF：'拥有' 不识别→无 possess claim
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = False
    try:
        claims_off = extract_property_claims(tokens, lang=LANG_ZH,
                                             backend=b, edge_store=es,
                                             space_id=sid, concept_index=ci)
        assert claims_off == [], "gate OFF '拥有' 不识别→无 possess claim（反 theater）"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved


# ============ 不写死守卫（D6·开放变体零硬编码） ============

def test_no_hardcode_open_variants_property():
    """不写死守卫：_REL_LEXICAL_CUE 只 closed-class 核心（具有/has）·
    开放变体（拥有/possesses）零硬编码·走 D:11 教师晋升（测试 fixture 注入）。"""
    zh = _REL_LEXICAL_CUE.get(LANG_ZH, {})
    assert "具有" in zh, "closed-class 核心 '具有' 在 _REL_LEXICAL_CUE"
    assert "拥有" not in zh, "开放变体 '拥有' 不硬编码（走 D:11 教师晋升）"
    en = _REL_LEXICAL_CUE.get(LANG_EN, {})
    assert "has" in en and "possesses" not in en
    # _PROPERTY_POSSESS_CUE frozenset 既有（具有/有/has/have）·开放变体不在
    zh_possess = _PROPERTY_POSSESS_CUE.get(LANG_ZH, frozenset())
    assert "具有" in zh_possess and "拥有" not in zh_possess
