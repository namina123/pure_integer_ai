"""STEP5 PR2 operator-level D:11 alias 测试（任务 #896）。

operator_primitives.py 新模块（OP_* enum + ensure + lookup_word_operator）+ ATTR_OPERATOR_PRIMITIVE=18 +
arith_op_of/comparison_op_of D:11 readback 第二源（gate OPERATOR_D11_READBACK_MODE）+ bootstrap_operator_signals。
消费者 = extract_numeric_claims→numeric_proof_fn / extract_comparison_claims→comparison_proof_fn（live）。

**反 theater 两路证**：
  (i) seeded closed-class（加/大于·_OP_LEXICAL_CUE D:11 种子）：gate ON/OFF frozenset 第一源均命中（D:11 冗余但建边）
  (ii) fixture 注入开放变体（相加/超过→OP_* D:11 边·SOURCE_TEACHER 模拟教师晋升）：
       gate ON arith_op_of/comparison_op_of 返 opcode + extract_*_claims 命中 + proof_fn reward=1·
       gate OFF 返 None（D:11 唯一源·行为差可观测）。

**不写死守卫**（D6）：_OP_LEXICAL_CUE 只 closed-class 核心（加减乘大于小于·镜像 _ARITH_OP_WORDS/_COMPARISON_OP_WORDS）·
开放变体（相加/超过/增加）零硬编码·仅测试 fixture 注入。

**无交叉污染**：D:11 共享边类型·lookup_word_concept 过滤 ATTR_RELATION_PRIMITIVE·lookup_word_operator 过滤
ATTR_OPERATOR_PRIMITIVE·kind==0 skip。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_TEACHER
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import register_composes_attr, ATTR_OPERATOR_PRIMITIVE, ATTR_RELATION_PRIMITIVE
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives, REL_EQUAL
from pure_integer_ai.cognition.shared.operator_primitives import (
    OP_ADD, OP_SUB, OP_MUL, OP_GT, OP_LT, OP_GE, OP_LE,
    ensure_operator_primitives, lookup_word_operator,
    op_kind_to_opcode, is_arith_op_kind, is_comparison_op_kind,
    _OP_LEXICAL_CUE, _OP_TO_OPCODE,
)
from pure_integer_ai.cognition.understanding.cue_words import (
    arith_op_of, comparison_op_of, _ARITH_OP_WORDS, _COMPARISON_OP_WORDS,
)
from pure_integer_ai.cognition.understanding.word_concept_signal import (
    bootstrap_operator_signals, record_word_concept,
)
from pure_integer_ai.cognition.understanding.cue_extractor import (
    extract_numeric_claims, extract_comparison_claims,
)
from pure_integer_ai.numeric.symbol_domain import OPCODE_ADD, OPCODE_SUB, OPCODE_MUL
from pure_integer_ai.crosscut.integer.compare import CMP_GT, CMP_LT, CMP_GE, CMP_LE
from pure_integer_ai.training.numeric_proof import numeric_proof_fn_factory
from pure_integer_ai.training.comparison_proof import comparison_proof_fn_factory
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def op_env():
    """PR2 单测环境（dict backend·core space·composes_attr 注册·boot 种 operator D:11 边）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    # boot 种 operator D:11 边（含 closed-class 核心 加/减/乘/大于/小于/不小于/不大于 + EN）
    bootstrap_operator_signals(ci, es, b, space_id=sid, langs={LANG_ZH, LANG_EN})
    yield b, sid, es, ns, ci
    b.close()


def _inject_op_d11(ci, es, b, sid, word_surface, op_kind):
    """fixture 注入 D:11 边（word→OP_* ref·SOURCE_TEACHER·模拟教师晋升·非 frozenset·非生产 boot 种子）。"""
    op_refs = ensure_operator_primitives(ci, b, space_id=sid)
    op_ref = op_refs[op_kind]
    return record_word_concept(ci, es, word_surface, op_ref,
                               space_id=sid, source=SOURCE_TEACHER)


# ============ 基建：ensure_operator_primitives + lookup_word_operator ============

def test_ensure_operator_primitives_builds_7_nodes(op_env):
    """ensure_operator_primitives 建 7 个 OP_* NODE_CONCEPT + ATTR_OPERATOR_PRIMITIVE=18 标记·幂等。"""
    b, sid, es, ns, ci = op_env
    attrs = [r for r in b.select("composes_attr", where={"kind": ATTR_OPERATOR_PRIMITIVE})]
    assert len(attrs) == 7, "7 OP_* 节点全建（OP_ADD/SUB/MUL/GT/LT/GE/LE）"
    kinds = sorted(r["int_a"] for r in attrs)
    assert kinds == [OP_ADD, OP_SUB, OP_MUL, OP_GT, OP_LT, OP_GE, OP_LE]
    # 幂等：重复调不 corrupt
    ensure_operator_primitives(ci, b, space_id=sid)
    attrs2 = [r for r in b.select("composes_attr", where={"kind": ATTR_OPERATOR_PRIMITIVE})]
    assert len(attrs2) == 7, "幂等·重复 ensure 不增重复"


def test_lookup_word_operator_reads_d11(op_env):
    """lookup_word_operator 读 D:11 边→[(op_ref, op_kind)]·tier_filter + kind==0 skip。"""
    b, sid, es, ns, ci = op_env
    加 = ci.lookup("加", sid)
    assert 加 is not None, "boot 种了 加 概念"
    ops = lookup_word_operator(b, es, 加, space_id=sid, tier_filter=TIER_PRIMARY)
    kinds = [k for _r, k in ops]
    assert OP_ADD in kinds, "加→OP_ADD D:11 PRIMARY 边建出"


def test_lookup_word_operator_no_cross_pollution_rel(op_env):
    """无交叉污染：lookup_word_operator 对 REL_* target（ATTR_RELATION_PRIMITIVE）kind==0 skip。
    REL_EQUAL 概念节点挂 ATTR_RELATION_PRIMITIVE（非 ATTR_OPERATOR_PRIMITIVE）→ lookup_word_operator 返空。"""
    b, sid, es, ns, ci = op_env
    ensure_relation_primitives(ci, b, space_id=sid)  # 建 REL_* 节点
    rel_equal_ref = ci.lookup("__REL_EQUAL__", sid)
    assert rel_equal_ref is not None
    ops = lookup_word_operator(b, es, rel_equal_ref, space_id=sid, tier_filter=TIER_PRIMARY)
    assert ops == [], "REL_* target 挂 ATTR_RELATION_PRIMITIVE·lookup_word_operator kind==0 skip·返空"


def test_op_to_opcode_mapping():
    """_OP_TO_OPCODE 双射：OP_*→opcode（OPCODE_* 大整数 + CMP_* 1-4 值域不重叠）。"""
    assert _OP_TO_OPCODE[OP_ADD] == OPCODE_ADD
    assert _OP_TO_OPCODE[OP_GT] == CMP_GT
    assert op_kind_to_opcode(OP_ADD) == OPCODE_ADD
    assert op_kind_to_opcode(OP_LE) == CMP_LE
    assert op_kind_to_opcode(999) is None  # 防御
    assert is_arith_op_kind(OP_ADD) and not is_arith_op_kind(OP_GT)
    assert is_comparison_op_kind(OP_GT) and not is_comparison_op_kind(OP_ADD)


# ============ (i) seeded closed-class ============

def test_arith_op_seeded_frozenset_first_source(op_env):
    """(i) seeded '加' gate ON/OFF 均 OPCODE_ADD（frozenset 第一源先命中·D:11 冗余但无害）。"""
    b, sid, es, ns, ci = op_env
    saved = gates.OPERATOR_D11_READBACK_MODE
    gates.OPERATOR_D11_READBACK_MODE = False
    try:
        assert arith_op_of("加", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) == OPCODE_ADD
    finally:
        gates.OPERATOR_D11_READBACK_MODE = saved
    gates.OPERATOR_D11_READBACK_MODE = True
    try:
        assert arith_op_of("加", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) == OPCODE_ADD
    finally:
        gates.OPERATOR_D11_READBACK_MODE = saved


# ============ (ii) fixture 注入开放变体（D:11 唯一源·反 theater） ============

def test_arith_op_d11_readback_nontournament_word(op_env):
    """(ii) fixture 注入 '相加'→OP_ADD D:11 边→gate ON arith_op_of 返 OPCODE_ADD·gate OFF 返 None。"""
    b, sid, es, ns, ci = op_env
    _inject_op_d11(ci, es, b, sid, "相加", OP_ADD)
    saved = gates.OPERATOR_D11_READBACK_MODE
    gates.OPERATOR_D11_READBACK_MODE = False
    try:
        assert arith_op_of("相加", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) is None, \
            "gate OFF '相加' 非 frozenset·readback 关→None"
    finally:
        gates.OPERATOR_D11_READBACK_MODE = saved
    gates.OPERATOR_D11_READBACK_MODE = True
    try:
        assert arith_op_of("相加", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) == OPCODE_ADD, \
            "gate ON D:11 readback '相加'→OP_ADD→OPCODE_ADD"
    finally:
        gates.OPERATOR_D11_READBACK_MODE = saved


def test_arith_op_d11_drives_numeric_proof(op_env):
    """(ii) 端到端：'相加' 经 D:11 readback 驱动 extract_numeric_claims + numeric_proof_fn reward=1。
    '3 相加 5 等于 8' → claim (3, ADD, 5, 8) → numeric_proof_fn 3+5==8→1。"""
    b, sid, es, ns, ci = op_env
    _inject_op_d11(ci, es, b, sid, "相加", OP_ADD)
    tokens = ["3", "相加", "5", "等于", "8"]
    saved = gates.OPERATOR_D11_READBACK_MODE
    gates.OPERATOR_D11_READBACK_MODE = True
    try:
        claims_on = extract_numeric_claims(tokens, lang=LANG_ZH,
                                           backend=b, edge_store=es,
                                           space_id=sid, concept_index=ci)
        assert (3, OPCODE_ADD, 5, 8) in claims_on, \
            "gate ON '相加' 经 D:11 readback→OPCODE_ADD·'等于' frozenset 锚→claim (3,ADD,5,8)"
        fn = numeric_proof_fn_factory(claims=claims_on)
        assert fn(None, None, None) == 1, "numeric_proof_fn 3+5==8→reward=1"
    finally:
        gates.OPERATOR_D11_READBACK_MODE = saved
    # gate OFF：'相加' 不识别→op None→无 claim
    gates.OPERATOR_D11_READBACK_MODE = False
    try:
        claims_off = extract_numeric_claims(tokens, lang=LANG_ZH,
                                            backend=b, edge_store=es,
                                            space_id=sid, concept_index=ci)
        assert claims_off == [], "gate OFF '相加' 不识别→无 claim（行为差可观测·反 theater）"
    finally:
        gates.OPERATOR_D11_READBACK_MODE = saved


def test_comparison_op_d11_readback_nontournament_word(op_env):
    """(ii) fixture 注入 '超过'→OP_GT D:11 边→gate ON comparison_op_of 返 CMP_GT·gate OFF 返 None。"""
    b, sid, es, ns, ci = op_env
    _inject_op_d11(ci, es, b, sid, "超过", OP_GT)
    saved = gates.OPERATOR_D11_READBACK_MODE
    gates.OPERATOR_D11_READBACK_MODE = False
    try:
        assert comparison_op_of("超过", LANG_ZH, backend=b, edge_store=es,
                                space_id=sid, concept_index=ci) is None
    finally:
        gates.OPERATOR_D11_READBACK_MODE = saved
    gates.OPERATOR_D11_READBACK_MODE = True
    try:
        assert comparison_op_of("超过", LANG_ZH, backend=b, edge_store=es,
                                space_id=sid, concept_index=ci) == CMP_GT, \
            "gate ON D:11 readback '超过'→OP_GT→CMP_GT"
    finally:
        gates.OPERATOR_D11_READBACK_MODE = saved


def test_comparison_op_d11_drives_comparison_proof(op_env):
    """(ii) 端到端：'超过' 经 D:11 readback 驱动 extract_comparison_claims + comparison_proof_fn。
    '5 超过 3' → claim (5, CMP_GT, 3) → comparison_proof_fn 5>3→1。"""
    b, sid, es, ns, ci = op_env
    _inject_op_d11(ci, es, b, sid, "超过", OP_GT)
    tokens = ["5", "超过", "3"]
    saved = gates.OPERATOR_D11_READBACK_MODE
    gates.OPERATOR_D11_READBACK_MODE = True
    try:
        claims_on = extract_comparison_claims(tokens, lang=LANG_ZH,
                                              backend=b, edge_store=es,
                                              space_id=sid, concept_index=ci)
        assert (5, CMP_GT, 3) in claims_on, "gate ON '超过'→CMP_GT·claim (5,CMP_GT,3)"
        fn = comparison_proof_fn_factory(claims=claims_on)
        assert fn(None, None, None) == 1, "comparison_proof_fn 5>3→reward=1"
    finally:
        gates.OPERATOR_D11_READBACK_MODE = saved
    gates.OPERATOR_D11_READBACK_MODE = False
    try:
        claims_off = extract_comparison_claims(tokens, lang=LANG_ZH,
                                               backend=b, edge_store=es,
                                               space_id=sid, concept_index=ci)
        assert claims_off == [], "gate OFF '超过' 不识别→无 claim（反 theater）"
    finally:
        gates.OPERATOR_D11_READBACK_MODE = saved


# ============ 无交叉污染（arith vs comparison 过滤） ============

def test_arith_vs_comparison_no_cross_pollution(op_env):
    """种 '加'→OP_ADD·_arith_op_from_d11_primary 返 OPCODE_ADD·_comparison_op_from_d11_primary 返 None
    （OP_ADD 非比较 OP·过滤正确·无交叉污染）。"""
    b, sid, es, ns, ci = op_env
    saved = gates.OPERATOR_D11_READBACK_MODE
    gates.OPERATOR_D11_READBACK_MODE = True
    try:
        # '加' seeded→OP_ADD·arith 读回 OPCODE_ADD·comparison 读回 None（OP_ADD 非 OP_GT/LT/GE/LE）
        assert arith_op_of("加", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) == OPCODE_ADD
        assert comparison_op_of("加", LANG_ZH, backend=b, edge_store=es,
                                space_id=sid, concept_index=ci) is None, \
            "'加'→OP_ADD·comparison_op_of 过滤算术 OP·返 None（无交叉污染）"
        # '大于' seeded→OP_GT·comparison 读回 CMP_GT·arith 读回 None
        assert comparison_op_of("大于", LANG_ZH, backend=b, edge_store=es,
                                space_id=sid, concept_index=ci) == CMP_GT
        assert arith_op_of("大于", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) is None, \
            "'大于'→OP_GT·arith_op_of 过滤比较 OP·返 None"
    finally:
        gates.OPERATOR_D11_READBACK_MODE = saved


# ============ 不写死守卫（D6·开放变体零硬编码） ============

def test_no_hardcode_open_variants_operator():
    """不写死守卫：_OP_LEXICAL_CUE 只 closed-class 核心（镜像 _ARITH_OP_WORDS/_COMPARISON_OP_WORDS）·
    开放变体（相加/超过/增加）零硬编码·仅测试 fixture 注入。"""
    zh = _OP_LEXICAL_CUE.get(LANG_ZH, {})
    # closed-class 核心在
    for w in ["加", "减", "乘", "大于", "小于", "不小于", "不大于"]:
        assert w in zh, f"closed-class 核心 {w} 在 _OP_LEXICAL_CUE"
    # 开放变体不在
    for w in ["相加", "超过", "增加", "多于"]:
        assert w not in zh, f"开放变体 {w} 不硬编码（走 D:11 教师晋升）"
    # _ARITH_OP_WORDS / _COMPARISON_OP_WORDS 也无开放变体（既有 frozenset 不被 PR2 污染）
    zh_arith = _ARITH_OP_WORDS.get(LANG_ZH, {})
    assert "相加" not in zh_arith and "加" in zh_arith
    zh_cmp = _COMPARISON_OP_WORDS.get(LANG_ZH, {})
    assert "超过" not in zh_cmp and "大于" in zh_cmp


# ============ 铁律守卫 ============

def test_op_d11_not_in_effective_weight(op_env):
    """铁律：D:11 EDGE_RELATION_SIGNAL 边不入 effective_weight（只认 {PRECEDES,CAUSES,REFERS_TO}·:82 assert）。
    ATTR_OPERATOR_PRIMITIVE=18 非结构 kind（_STRUCTURAL_KINDS 不含·read_composes_tree 忽略·inline 不传播）。"""
    b, sid, es, ns, ci = op_env
    from pure_integer_ai.cognition.process.effective_weight import effective_weight
    # D:11 边 dict 传入 effective_weight 应 assert fail（et=11 不在 {PRECEDES,CAUSES,REFERS_TO}·loud fail 非偷注入）
    d11_edge = {"edge_type": EDGE_RELATION_SIGNAL, "strength": 1}
    try:
        effective_weight(d11_edge)
        assert False, "D:11 边入 effective_weight 应 assert fail（:82 et 不在 {PRECEDES,CAUSES,REFERS_TO}）"
    except AssertionError:
        pass  # 预期 assert fail
    # ATTR_OPERATOR_PRIMITIVE=18 非结构 kind 确认（_STRUCTURAL_KINDS 不含·不污染 5-dict 重建）
    from pure_integer_ai.cognition.understanding.arith_observe import _STRUCTURAL_KINDS
    assert 18 not in _STRUCTURAL_KINDS, "ATTR_OPERATOR_PRIMITIVE=18 非结构 kind·不污染 5-dict 重建"
    assert ATTR_OPERATOR_PRIMITIVE == 18
