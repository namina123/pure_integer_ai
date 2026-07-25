"""STEP5 PR1 REL_EQUAL D:11 readback 测试（任务 #895）。

REL_EQUAL → ARITH_EQUALS_CUE 映射 + _REL_LEXICAL_CUE 加 等于/equals closed-class 种子。
消费者 = extract_numeric_claims → numeric_proof_fn（live·reward=1 iff 算术一致）。

**反 theater 两路证**：
  (i) seeded closed-class "等于"：gate ON/OFF 均 ARITH_EQUALS_CUE（frozenset 第一源先命中·D:11 冗余但建边）
  (ii) fixture 注入开放变体 "等同于"→REL_EQUAL D:11 边（record_word_concept SOURCE_TEACHER·模拟教师晋升）：
       gate ON cue_type_of 返 ARITH_EQUALS_CUE + extract_numeric_claims 命中 + numeric_proof_fn reward=1·
       gate OFF 返 None（D:11 唯一源·行为差可观测）。

**不写死守卫**（D6·用户强调）：开放变体（等同于）零硬编码·仅测试 fixture 注入。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_TEACHER
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
from pure_integer_ai.cognition.shared.relation_primitives import (
    ensure_relation_primitives, REL_EQUAL,
)
from pure_integer_ai.cognition.understanding.cue_words import (
    cue_type_of, ARITH_EQUALS_CUE, _CUE_WORDS, _REL_KIND_TO_CUE_TYPE,
)
from pure_integer_ai.cognition.understanding.word_concept_signal import (
    bootstrap_word_concept_signals, record_word_concept, lookup_word_concept,
    _REL_LEXICAL_CUE,
)
from pure_integer_ai.cognition.understanding.cue_extractor import extract_numeric_claims
from pure_integer_ai.cognition.understanding.cue_words import arith_op_of
from pure_integer_ai.numeric.symbol_domain import OPCODE_ADD
from pure_integer_ai.training.numeric_proof import numeric_proof_fn_factory
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def rel_equal_env():
    """PR1 单测环境（dict backend·core space·composes_attr 注册）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    # boot 种 D:11 边（含 STEP5 新增 等于→REL_EQUAL）
    bootstrap_word_concept_signals(ci, es, b, space_id=sid, langs={LANG_ZH, LANG_EN})
    yield b, sid, es, ns, ci
    b.close()


# ============ (i) seeded closed-class "等于" ============

def test_rel_equal_seed_bootstrapped(rel_equal_env):
    """boot 种 等于→REL_EQUAL D:11 边（lookup_word_concept 返 REL_EQUAL）。"""
    b, sid, es, ns, ci = rel_equal_env
    等于 = ci.lookup("等于", sid)
    assert 等于 is not None, "boot 种了 等于 概念"
    rels = lookup_word_concept(b, es, 等于, space_id=sid, tier_filter=TIER_PRIMARY)
    kinds = [k for _r, k in rels]
    assert REL_EQUAL in kinds, "等于→REL_EQUAL D:11 PRIMARY 边建出"


def test_rel_equal_seeded_frozenset_first_source(rel_equal_env):
    """(i) seeded '等于' gate ON/OFF 均 ARITH_EQUALS_CUE（frozenset 第一源先命中·D:11 冗余但无害）。"""
    b, sid, es, ns, ci = rel_equal_env
    # gate OFF：纯 frozenset 第一源
    saved = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = False
    try:
        ct_off = cue_type_of("等于", LANG_ZH, backend=b, edge_store=es,
                             space_id=sid, concept_index=ci)
        assert ct_off == ARITH_EQUALS_CUE, "gate OFF frozenset 第一源 命中 等于→ARITH_EQUALS_CUE"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved
    # gate ON：第一源仍先命中（D:11 冗余）
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    try:
        ct_on = cue_type_of("等于", LANG_ZH, backend=b, edge_store=es,
                            space_id=sid, concept_index=ci)
        assert ct_on == ARITH_EQUALS_CUE, "gate ON frozenset 第一源先命中（D:11 冗余但无害）"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved


def test_rel_equal_mapping_in_table():
    """_REL_KIND_TO_CUE_TYPE 含 REL_EQUAL→ARITH_EQUALS_CUE 映射。"""
    assert _REL_KIND_TO_CUE_TYPE.get(REL_EQUAL) == ARITH_EQUALS_CUE


# ============ (ii) fixture 注入开放变体 "等同于"（D:11 唯一源·反 theater） ============

def _inject_teacher_d11(b, es, ci, sid, word_surface, rel_kind):
    """fixture 注入 D:11 边（SOURCE_TEACHER·模拟教师晋升·非 frozenset·非生产 boot 种子）。"""
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    rel_ref = rel_refs[rel_kind]
    return record_word_concept(ci, es, word_surface, rel_ref,
                               space_id=sid, source=SOURCE_TEACHER)


def test_rel_equal_d11_readback_nontournament_word(rel_equal_env):
    """(ii) fixture 注入 '等同于'→REL_EQUAL D:11 边→gate ON cue_type_of 返 ARITH_EQUALS_CUE·
    gate OFF 返 None（D:11 唯一源·行为差可观测·反 theater）。"""
    b, sid, es, ns, ci = rel_equal_env
    _inject_teacher_d11(b, es, ci, sid, "等同于", REL_EQUAL)
    # gate OFF：'等同于' 不在 frozenset·readback 关→None
    saved = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = False
    try:
        ct_off = cue_type_of("等同于", LANG_ZH, backend=b, edge_store=es,
                             space_id=sid, concept_index=ci)
        assert ct_off is None, "gate OFF '等同于' 非 frozenset·readback 关→None"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved
    # gate ON：D:11 readback 命中 REL_EQUAL→ARITH_EQUALS_CUE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    try:
        ct_on = cue_type_of("等同于", LANG_ZH, backend=b, edge_store=es,
                            space_id=sid, concept_index=ci)
        assert ct_on == ARITH_EQUALS_CUE, "gate ON D:11 readback '等同于'→REL_EQUAL→ARITH_EQUALS_CUE"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved


def test_rel_equal_d11_drives_numeric_proof(rel_equal_env):
    """(ii) 端到端：'等同于' 经 D:11 readback 驱动 extract_numeric_claims + numeric_proof_fn reward=1。
    '3 加 5 等同于 8' → claim (3, ADD, 5, 8) → numeric_proof_fn 3+5==8→1。"""
    b, sid, es, ns, ci = rel_equal_env
    _inject_teacher_d11(b, es, ci, sid, "等同于", REL_EQUAL)
    tokens = ["3", "加", "5", "等同于", "8"]
    saved = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    # gate ON：readback 命中→extract_numeric_claims 产 claim
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    try:
        claims_on = extract_numeric_claims(tokens, lang=LANG_ZH,
                                           backend=b, edge_store=es,
                                           space_id=sid, concept_index=ci)
        assert (3, OPCODE_ADD, 5, 8) in claims_on, \
            "gate ON '等同于' 经 D:11 readback 作 ARITH_EQUALS_CUE 锚→claim (3,ADD,5,8) 产出"
        fn = numeric_proof_fn_factory(claims=claims_on)
        assert fn(None, None, None) == 1, "numeric_proof_fn 3+5==8→reward=1（构造性检查通过）"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved
    # gate OFF：'等同于' 不识别→无 ARITH_EQUALS_CUE 锚→无 claim
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = False
    try:
        claims_off = extract_numeric_claims(tokens, lang=LANG_ZH,
                                            backend=b, edge_store=es,
                                            space_id=sid, concept_index=ci)
        assert claims_off == [], "gate OFF '等同于' 不识别→无 claim（行为差可观测·反 theater）"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved


def test_rel_equal_arith_op_frozenset_unchanged(rel_equal_env):
    """'加' 走 frozenset 第一源（arith_op_of·非 D:11·STEP5 PR1 不改 operator 路）。"""
    b, sid, es, ns, ci = rel_equal_env
    assert arith_op_of("加", LANG_ZH) == OPCODE_ADD, "'加' frozenset 第一源不变"


# ============ 不写死守卫（D6·开放变体零硬编码） ============

def test_no_hardcode_open_variants_rel_equal():
    """不写死守卫：开放变体 '等同于' 零硬编码（不在 _REL_LEXICAL_CUE / _CUE_WORDS 生产 frozenset）。
    仅 closed-class 核心 '等于' 在种子表。开放变体走 D:11 教师晋升（测试 fixture 注入）。"""
    zh_rel = _REL_LEXICAL_CUE.get(LANG_ZH, {})
    assert "等于" in zh_rel, "closed-class 核心 '等于' 在 _REL_LEXICAL_CUE"
    assert "等同于" not in zh_rel, "开放变体 '等同于' 不硬编码（走 D:11 教师晋升）"
    # _CUE_WORDS ARITH_EQUALS_CUE frozenset
    zh_cue = _CUE_WORDS.get(LANG_ZH, {})
    assert "等于" in zh_cue.get(ARITH_EQUALS_CUE, frozenset()), "'等于' 在 _CUE_WORDS"
    assert "等同于" not in zh_cue.get(ARITH_EQUALS_CUE, frozenset()), \
        "开放变体 '等同于' 不硬编码进 _CUE_WORDS"
    # EN 同理
    en_rel = _REL_LEXICAL_CUE.get(LANG_EN, {})
    assert "equals" in en_rel and "equivalent_to" not in en_rel
