"""#940 否定词 D:11 readback 第二源测试（镜像 test_modal_d11_readback.py 范式·符号域先天变体）。

symbol_types.py 激活 ensure_symbol_types（shadow→活）+ _NEGATION_LEXICAL_CUE + lookup_word_negation +
is_negation_cue D:11 readback 第二源（gate NEGATION_D11_READBACK_MODE）+ bootstrap_negation_signals。

**关键差异（否定 vs modal·D6）**：否定=**符号域先天**（TYPE_NEGATION=12·¬ 先天冻结·同 operator·非 modal 抽象空间）·
故只挂 ATTR_SYMBOL_TYPE=17·**不挂 abstract_mark**（异 modal 双挂 ATTR_MODAL_KIND+MARK_MODAL_KIND）·
复用既有 TYPE_NEGATION+ATTR_SYMBOL_TYPE·**不新增 ATTR/MARK 编号**（STOP+D6 双合规）。

**反 theater 两路证**：
  (i) seeded closed-class（不/没·_NEGATION_LEXICAL_CUE D:11 种子）：gate ON/OFF frozenset 第一源均命中（D:11 冗余但建边）
  (ii) fixture 注入开放变体（未必→TYPE_NEGATION D:11 边·SOURCE_TEACHER 模拟教师晋升）：
       gate ON is_negation_cue 返 True + extract_property_claims 否定窗口填 polarity=1·gate OFF 返 False（D:11 唯一源·行为差可观测）。

**不写死守卫**（D6）：_NEGATION_LEXICAL_CUE 只 closed-class 核心（不/没/非/无·镜像 _NEGATION_CUES）·
开放变体（未必/绝非/谈不上/休想）零硬编码·仅测试 fixture 注入。

**无交叉污染**：D:11 共享边类型·lookup_word_concept 过滤 ATTR_RELATION_PRIMITIVE·lookup_word_operator 过滤
ATTR_OPERATOR_PRIMITIVE·lookup_word_modality 过滤 ATTR_MODAL_KIND·lookup_word_negation 过滤
ATTR_SYMBOL_TYPE int_a==TYPE_NEGATION·互不交叉。

**否定 D:11 readback 语义**：¬ 概念先天不可学（TYPE_NEGATION 冻结·同 OP_*）·D:11 readback 意义=否定词
文字 alias 可学习（教师晋升新否定词如未必/绝非）·非概念可学（异 modal 的 modal_kind concept 可学）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_TEACHER
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import (
    register_composes_attr, ATTR_SYMBOL_TYPE, ATTR_RELATION_PRIMITIVE, ATTR_OPERATOR_PRIMITIVE, ATTR_MODAL_KIND,
)
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import LANG_ZH
from pure_integer_ai.cognition.shared.symbol_types import (
    TYPE_NEGATION, ensure_symbol_types, lookup_word_negation, _NEGATION_LEXICAL_CUE,
)
from pure_integer_ai.cognition.understanding.cue_words import is_negation_cue, _NEGATION_CUES
from pure_integer_ai.cognition.understanding.word_concept_signal import (
    bootstrap_negation_signals, record_word_concept,
)
from pure_integer_ai.cognition.understanding.cue_extractor import extract_property_claims
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def neg_env():
    """#940 单测环境（dict backend·core space·composes_attr 注册·boot 种 negation D:11 边）。
    否定=符号域先天·**不**注册 abstract_mark（异 modal·D6 符号域非抽象空间）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)   # ATTR_SYMBOL_TYPE=17 标记表（否定不挂 abstract_mark·无需 register_abstract_mark）
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    # boot 种 negation D:11 边（含 closed-class 核心 不/没/非/无）
    bootstrap_negation_signals(ci, es, b, space_id=sid, langs={LANG_ZH})
    yield b, sid, es, ns, ci
    b.close()


def _inject_negation_d11(ci, es, b, sid, word_surface):
    """fixture 注入 D:11 边（word→TYPE_NEGATION ref·SOURCE_TEACHER·模拟教师晋升·非 frozenset·非生产 boot 种子）。"""
    type_refs = ensure_symbol_types(ci, b, space_id=sid)
    neg_ref = type_refs[TYPE_NEGATION]
    return record_word_concept(ci, es, word_surface, neg_ref,
                               space_id=sid, source=SOURCE_TEACHER)


# ============ 基建：ensure_symbol_types（激活）+ lookup_word_negation ============

def test_ensure_symbol_types_builds_negation_node(neg_env):
    """ensure_symbol_types 建 TYPE_NEGATION NODE_CONCEPT + ATTR_SYMBOL_TYPE=17 标记（int_a=12）·幂等。"""
    b, sid, es, ns, ci = neg_env
    attrs = [r for r in b.select("composes_attr", where={"kind": ATTR_SYMBOL_TYPE})]
    assert len(attrs) == 1, "1 TYPE_NEGATION 节点（单一·否定无种类）"
    assert attrs[0]["int_a"] == TYPE_NEGATION, "ATTR_SYMBOL_TYPE int_a=TYPE_NEGATION=12"
    # 幂等：重复调不 corrupt
    ensure_symbol_types(ci, b, space_id=sid)
    attrs2 = [r for r in b.select("composes_attr", where={"kind": ATTR_SYMBOL_TYPE})]
    assert len(attrs2) == 1, "幂等·重复 ensure 不增重复"


def test_lookup_word_negation_reads_d11(neg_env):
    """lookup_word_negation 读 D:11 边→True（命中 TYPE_NEGATION target）·tier_filter 守。"""
    b, sid, es, ns, ci = neg_env
    不 = ci.lookup("不", sid)
    assert 不 is not None, "boot 种了 不 概念"
    assert lookup_word_negation(b, es, 不, space_id=sid, tier_filter=TIER_PRIMARY) is True, \
        "不→TYPE_NEGATION D:11 PRIMARY 边建出·readback True"


def test_lookup_word_negation_no_cross_pollution_rel_op_modal(neg_env):
    """无交叉污染：lookup_word_negation 对 REL_*/OP_*/MODAL target（无 ATTR_SYMBOL_TYPE 或 int_a≠12）返 False。"""
    b, sid, es, ns, ci = neg_env
    from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives
    from pure_integer_ai.cognition.shared.operator_primitives import ensure_operator_primitives
    from pure_integer_ai.cognition.shared.modal_primitives import ensure_modal_primitives
    ensure_relation_primitives(ci, b, space_id=sid)
    ensure_operator_primitives(ci, b, space_id=sid)
    ensure_modal_primitives(ci, b, space_id=sid)
    rel_equal_ref = ci.lookup("__REL_EQUAL__", sid)
    assert rel_equal_ref is not None
    assert lookup_word_negation(b, es, rel_equal_ref, space_id=sid, tier_filter=TIER_PRIMARY) is False, \
        "REL_* target 无 ATTR_SYMBOL_TYPE int_a=12·lookup_word_negation 返 False"
    op_add_ref = ci.lookup("__OP_ADD__", sid)
    assert op_add_ref is not None
    assert lookup_word_negation(b, es, op_add_ref, space_id=sid, tier_filter=TIER_PRIMARY) is False, \
        "OP_* target 无 ATTR_SYMBOL_TYPE int_a=12·lookup_word_negation 返 False"
    modal_box_ref = ci.lookup("__MODAL_BOX_NECESSITY__", sid)
    assert modal_box_ref is not None
    assert lookup_word_negation(b, es, modal_box_ref, space_id=sid, tier_filter=TIER_PRIMARY) is False, \
        "MODAL_KIND target 无 ATTR_SYMBOL_TYPE int_a=12·lookup_word_negation 返 False"


# ============ (i) seeded closed-class ============

def test_negation_seeded_frozenset_first_source(neg_env):
    """(i) seeded '不' gate ON/OFF 均 True（frozenset 第一源先命中·D:11 冗余但无害）。"""
    b, sid, es, ns, ci = neg_env
    saved = gates.NEGATION_D11_READBACK_MODE
    gates.NEGATION_D11_READBACK_MODE = False
    try:
        assert is_negation_cue("不", LANG_ZH, backend=b, edge_store=es,
                               space_id=sid, concept_index=ci) is True
    finally:
        gates.NEGATION_D11_READBACK_MODE = saved
    gates.NEGATION_D11_READBACK_MODE = True
    try:
        assert is_negation_cue("不", LANG_ZH, backend=b, edge_store=es,
                               space_id=sid, concept_index=ci) is True
    finally:
        gates.NEGATION_D11_READBACK_MODE = saved


# ============ (ii) fixture 注入开放变体（D:11 唯一源·反 theater） ============

def test_negation_d11_readback_open_variant(neg_env):
    """(ii) fixture 注入 '未必'→TYPE_NEGATION D:11 边→gate ON is_negation_cue 返 True·gate OFF 返 False。"""
    b, sid, es, ns, ci = neg_env
    _inject_negation_d11(ci, es, b, sid, "未必")
    saved = gates.NEGATION_D11_READBACK_MODE
    gates.NEGATION_D11_READBACK_MODE = False
    try:
        assert is_negation_cue("未必", LANG_ZH, backend=b, edge_store=es,
                               space_id=sid, concept_index=ci) is False, \
            "gate OFF '未必' 非 frozenset·readback 关→False"
    finally:
        gates.NEGATION_D11_READBACK_MODE = saved
    gates.NEGATION_D11_READBACK_MODE = True
    try:
        assert is_negation_cue("未必", LANG_ZH, backend=b, edge_store=es,
                               space_id=sid, concept_index=ci) is True, \
            "gate ON D:11 readback '未必'→TYPE_NEGATION→True"
    finally:
        gates.NEGATION_D11_READBACK_MODE = saved


def test_negation_d11_drives_property_claim(neg_env):
    """(ii) 端到端：'未必' 经 D:11 readback 驱动 extract_property_claims 否定窗口填 polarity=1。
    '猫 的 颜色 未必 是 黑' → gate ON 有 pol=1 claim·gate OFF '未必' 不识别→无 pol=1（行为差可观测·反 theater）。"""
    b, sid, es, ns, ci = neg_env
    _inject_negation_d11(ci, es, b, sid, "未必")
    tokens = ["猫", "的", "颜色", "未必", "是", "黑"]
    saved = gates.NEGATION_D11_READBACK_MODE
    # gate ON：'未必' 经 D:11 readback→is_negation_cue True→否定窗口 pol=1
    gates.NEGATION_D11_READBACK_MODE = True
    try:
        claims_on = extract_property_claims(tokens, lang=LANG_ZH,
                                            negation_on=True, modality_on=False,
                                            backend=b, edge_store=es,
                                            space_id=sid, concept_index=ci)
        pol_values = {c[4] for c in claims_on}
        assert 1 in pol_values, "gate ON '未必' 经 D:11 readback→否定窗口 pol=1"
    finally:
        gates.NEGATION_D11_READBACK_MODE = saved
    # gate OFF：'未必' 不识别→非否定窗口·无 pol=1
    gates.NEGATION_D11_READBACK_MODE = False
    try:
        claims_off = extract_property_claims(tokens, lang=LANG_ZH,
                                             negation_on=True, modality_on=False,
                                             backend=b, edge_store=es,
                                             space_id=sid, concept_index=ci)
        pol_values_off = {c[4] for c in claims_off}
        assert 1 not in pol_values_off, "gate OFF '未必' 不识别→无 pol=1（行为差可观测·反 theater）"
    finally:
        gates.NEGATION_D11_READBACK_MODE = saved


def test_is_negation_cue_d11_readback(neg_env):
    """is_negation_cue 透传 4 参→D:11 readback（gate ON 时非 frozenset 否定词亦判）。"""
    b, sid, es, ns, ci = neg_env
    _inject_negation_d11(ci, es, b, sid, "绝非")
    saved = gates.NEGATION_D11_READBACK_MODE
    gates.NEGATION_D11_READBACK_MODE = False
    try:
        assert is_negation_cue("绝非", LANG_ZH, backend=b, edge_store=es,
                               space_id=sid, concept_index=ci) is False, \
            "gate OFF '绝非' 非 frozenset·readback 关→False"
    finally:
        gates.NEGATION_D11_READBACK_MODE = saved
    gates.NEGATION_D11_READBACK_MODE = True
    try:
        assert is_negation_cue("绝非", LANG_ZH, backend=b, edge_store=es,
                               space_id=sid, concept_index=ci) is True, \
            "gate ON D:11 readback '绝非'→TYPE_NEGATION→True"
    finally:
        gates.NEGATION_D11_READBACK_MODE = saved


# ============ 无交叉污染（negation vs REL_*/OP_*/MODAL 过滤） ============

def test_negation_vs_rel_op_modal_no_cross_pollution(neg_env):
    """种 '不'→TYPE_NEGATION·lookup_word_negation 返 True·REL_*/OP_*/MODAL target kind==0/≠12 skip。
    lookup_word_concept/operator/modality 对 TYPE_NEGATION target 亦 skip（ATTR_SYMBOL_TYPE 非 REL/OP/MODAL 标记）。"""
    b, sid, es, ns, ci = neg_env
    from pure_integer_ai.cognition.understanding.word_concept_signal import lookup_word_concept
    from pure_integer_ai.cognition.shared.operator_primitives import lookup_word_operator
    from pure_integer_ai.cognition.shared.modal_primitives import lookup_word_modality
    不 = ci.lookup("不", sid)
    saved = gates.NEGATION_D11_READBACK_MODE
    gates.NEGATION_D11_READBACK_MODE = True
    try:
        # '不' seeded→TYPE_NEGATION·negation 读回 True·REL/OP/MODAL 读回空
        assert is_negation_cue("不", LANG_ZH, backend=b, edge_store=es,
                               space_id=sid, concept_index=ci) is True
        rels = lookup_word_concept(b, es, 不, space_id=sid, tier_filter=TIER_PRIMARY)
        assert rels == [], "TYPE_NEGATION target 挂 ATTR_SYMBOL_TYPE·lookup_word_concept kind==0 skip·返空"
        ops = lookup_word_operator(b, es, 不, space_id=sid, tier_filter=TIER_PRIMARY)
        assert ops == [], "TYPE_NEGATION target·lookup_word_operator kind==0 skip·返空"
        mods = lookup_word_modality(b, es, 不, space_id=sid, tier_filter=TIER_PRIMARY)
        assert mods == [], "TYPE_NEGATION target·lookup_word_modality kind==0 skip·返空"
    finally:
        gates.NEGATION_D11_READBACK_MODE = saved


# ============ 不写死守卫（D6·开放变体零硬编码） ============

def test_no_hardcode_open_variants_negation():
    """不写死守卫：_NEGATION_LEXICAL_CUE 只 closed-class 核心（镜像 _NEGATION_CUES）·
    开放变体（未必/绝非/谈不上/休想）零硬编码·仅测试 fixture 注入。"""
    zh = _NEGATION_LEXICAL_CUE.get(LANG_ZH, frozenset())
    # closed-class 核心在
    for w in ["不", "没", "非", "无"]:
        assert w in zh, f"closed-class 核心 {w} 在 _NEGATION_LEXICAL_CUE"
    # 开放变体不在
    for w in ["未必", "绝非", "谈不上", "休想", "勿", "莫"]:
        assert w not in zh, f"开放变体 {w} 不硬编码（走 D:11 教师晋升）"
    # _NEGATION_CUES 既有 frozenset 也无开放变体（第一源不被 #940 污染）
    zh_cues = _NEGATION_CUES.get(LANG_ZH, frozenset())
    assert "未必" not in zh_cues and "不" in zh_cues


# ============ 铁律守卫 ============

def test_negation_d11_not_in_effective_weight(neg_env):
    """铁律：D:11 EDGE_RELATION_SIGNAL 边不入 effective_weight（只认 {PRECEDES,CAUSES,REFERS_TO}·:82 assert）。
    ATTR_SYMBOL_TYPE=17 非结构 kind（_STRUCTURAL_KINDS 不含·read_composes_tree 忽略·inline 不传播）。"""
    b, sid, es, ns, ci = neg_env
    from pure_integer_ai.cognition.process.effective_weight import effective_weight
    # D:11 边 dict 传入 effective_weight 应 assert fail
    d11_edge = {"edge_type": EDGE_RELATION_SIGNAL, "strength": 1}
    try:
        effective_weight(d11_edge)
        assert False, "D:11 边入 effective_weight 应 assert fail"
    except AssertionError:
        pass  # 预期 assert fail
    # ATTR_SYMBOL_TYPE=17 非结构 kind 确认
    from pure_integer_ai.cognition.understanding.arith_observe import _STRUCTURAL_KINDS
    assert 17 not in _STRUCTURAL_KINDS, "ATTR_SYMBOL_TYPE=17 非结构 kind·不污染 5-dict 重建"
    assert ATTR_SYMBOL_TYPE == 17


# ============ bit-identical gate OFF ============

def test_negation_d11_gate_off_bit_identical(neg_env):
    """bit-identical：gate OFF is_negation_cue 退化纯 frozenset _NEGATION_CUES·既有 caller 无 4 参亦退化。
    4 参全 None + gate OFF → 纯 frozenset（既有行为·bit-identical）。"""
    b, sid, es, ns, ci = neg_env
    saved = gates.NEGATION_D11_READBACK_MODE
    gates.NEGATION_D11_READBACK_MODE = False
    try:
        # 4 参全 None + gate OFF → 纯 frozenset（既有两参调用兼容）
        assert is_negation_cue("不", LANG_ZH) is True
        assert is_negation_cue("没", LANG_ZH) is True
        assert is_negation_cue("非否定词", LANG_ZH) is False
    finally:
        gates.NEGATION_D11_READBACK_MODE = saved


# ============ D6 + STOP 合规（否定符号域先天·不挂 abstract_mark·复用 ATTR_SYMBOL_TYPE） ============

def test_negation_d6_no_abstract_mark():
    """D6：否定=符号域先天（TYPE_NEGATION=12）·ensure_symbol_types **不挂 abstract_mark**（异 modal 双挂
    ATTR_MODAL_KIND+MARK_MODAL_KIND）·只挂 ATTR_SYMBOL_TYPE。symbol_types 模块不 import set_mark。"""
    from pure_integer_ai.cognition.shared import symbol_types
    # symbol_types 模块不挂 abstract_mark（否定符号域先天·不挂 D6 归属·异 modal）
    assert not hasattr(symbol_types, "set_mark"), \
        "symbol_types 不挂 abstract_mark（否定符号域先天·异 modal 双挂 ATTR+MARK）"
    # 对比 modal_primitives 挂 set_mark（D6 抽象空间·双挂）
    from pure_integer_ai.cognition.shared import modal_primitives
    assert hasattr(modal_primitives, "set_mark"), "对比：modal_primitives 挂 set_mark（抽象空间 D6 归属）"


def test_negation_stop_reuse_no_new_numbers():
    """STOP+D6 合规：复用 TYPE_NEGATION=12 + ATTR_SYMBOL_TYPE=17·**不新增** ATTR/MARK/TYPE 编号。
    否定 D:11 readback 零新增编号（异 modal 新增 ATTR_MODAL_KIND=22 + MARK_MODAL_KIND=5）。"""
    from pure_integer_ai.cognition.shared import symbol_types
    # TYPE_NEGATION=12 已存在（STEP3·非 #940 新增）
    assert symbol_types.TYPE_NEGATION == 12
    # ATTR_SYMBOL_TYPE=17 已存在（STEP3 通用标记·复用·非新增）
    assert ATTR_SYMBOL_TYPE == 17
    # #940 未新增任何 ATTR_*（无 ATTR_NEGATION_CUE 之类）
    from pure_integer_ai.storage import composes_attr as ca
    assert not hasattr(ca, "ATTR_NEGATION_CUE"), "#940 不新增 ATTR_NEGATION_CUE·复用 ATTR_SYMBOL_TYPE"
    assert not hasattr(ca, "ATTR_NEGATION"), "无 ATTR_NEGATION（doc:193 ¬ 走命题 surface polarity·不建 ATTR）"
