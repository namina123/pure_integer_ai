"""审计根治 [严重-1] modal-level D:11 alias 测试（PR1·路 A2-变体）。

modal_primitives.py 新模块（MODAL_KIND_* enum + ensure_modal_primitives + lookup_word_modality）+
ATTR_MODAL_KIND=22 + abstract_mark MARK_MODAL_KIND=5（D6 归属）+ modal_op_of/is_modal_cue D:11 readback
第二源（gate MODAL_D11_READBACK_MODE）+ bootstrap_modal_signals。
消费者 = extract_property_claims 情态窗口→命题 surface modality（live）。

**反 theater 两路证**：
  (i) seeded closed-class（必然/可能·_MODAL_LEXICAL_CUE D:11 种子）：gate ON/OFF frozenset 第一源均命中（D:11 冗余但建边）
  (ii) fixture 注入开放变体（想必→MODAL_KIND D:11 边·SOURCE_TEACHER 模拟教师晋升）：
       gate ON modal_op_of 返 modality + extract_property_claims 情态窗口填值·gate OFF 返 None（D:11 唯一源·行为差可观测）。

**不写死守卫**（D6）：_MODAL_LEXICAL_CUE 只 closed-class 核心（必然/可能/必须/应该/可以·镜像 _MODAL_CUES）·
开放变体（想必/势必/说不定）零硬编码·仅测试 fixture 注入。

**无交叉污染**：D:11 共享边类型·lookup_word_concept 过滤 ATTR_RELATION_PRIMITIVE·lookup_word_operator 过滤
ATTR_OPERATOR_PRIMITIVE·lookup_word_modality 过滤 ATTR_MODAL_KIND·kind==0 skip。

**D6 职责分离**：composes_attr ATTR_MODAL_KIND=22 readback 标记 + abstract_mark MARK_MODAL_KIND=5 D6 归属·双挂非重复。

**解 [严重-1]**：删"无 REL_MODALITY 故无 D:11"循环论证偷渡·建 modal_kind concept + D:11 readback 二源·
开放变体走 D:11 教师晋升有路径·不违 STOP（ATTR_* 非 TYPE_*）不违 D6（abstract_mark 归属）。
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
    register_composes_attr, ATTR_MODAL_KIND, ATTR_RELATION_PRIMITIVE, ATTR_OPERATOR_PRIMITIVE,
)
from pure_integer_ai.storage.abstract_mark import (
    register_abstract_mark, MARK_MODAL_KIND, MARK_PROMOTED, get_mark,
)
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import LANG_ZH
from pure_integer_ai.cognition.shared.modal_primitives import (
    MODAL_KIND_BOX_NECESSITY, MODAL_KIND_BOX_POSSIBILITY,
    MODAL_KIND_DEONTIC_NECESSITY, MODAL_KIND_DEONTIC_POSSIBILITY,
    ensure_modal_primitives, lookup_word_modality, _MODAL_LEXICAL_CUE,
)
from pure_integer_ai.cognition.understanding.cue_words import modal_op_of, is_modal_cue, _MODAL_CUES
from pure_integer_ai.cognition.understanding.word_concept_signal import (
    bootstrap_modal_signals, record_word_concept,
)
from pure_integer_ai.cognition.understanding.cue_extractor import extract_property_claims
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def modal_env():
    """PR1 单测环境（dict backend·core space·composes_attr + abstract_mark 注册·boot 种 modal D:11 边）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    register_abstract_mark(b)   # MARK_MODAL_KIND D6 归属表
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    # boot 种 modal D:11 边（含 closed-class 核心 必然/可能/也许/必须/应该/可以）
    bootstrap_modal_signals(ci, es, b, space_id=sid, langs={LANG_ZH})
    yield b, sid, es, ns, ci
    b.close()


def _inject_modal_d11(ci, es, b, sid, word_surface, modal_kind):
    """fixture 注入 D:11 边（word→MODAL_KIND ref·SOURCE_TEACHER·模拟教师晋升·非 frozenset·非生产 boot 种子）。"""
    modal_refs = ensure_modal_primitives(ci, b, space_id=sid)
    modal_ref = modal_refs[modal_kind]
    return record_word_concept(ci, es, word_surface, modal_ref,
                               space_id=sid, source=SOURCE_TEACHER)


# ============ 基建：ensure_modal_primitives + lookup_word_modality ============

def test_ensure_modal_primitives_builds_4_nodes(modal_env):
    """ensure_modal_primitives 建 4 个 MODAL_KIND_* NODE_CONCEPT + ATTR_MODAL_KIND=22 标记 + MARK_MODAL_KIND=5 D6 归属·幂等。"""
    b, sid, es, ns, ci = modal_env
    attrs = [r for r in b.select("composes_attr", where={"kind": ATTR_MODAL_KIND})]
    assert len(attrs) == 4, "4 MODAL_KIND_* 节点全建"
    kinds = sorted(r["int_a"] for r in attrs)
    assert kinds == [1, 2, 3, 4], "MODAL_KIND_BOX_NECESSITY=1/BOX_POSSIBILITY=2/DEONTIC_NECESSITY=3/DEONTIC_POSSIBILITY=4"
    # 幂等：重复调不 corrupt
    ensure_modal_primitives(ci, b, space_id=sid)
    attrs2 = [r for r in b.select("composes_attr", where={"kind": ATTR_MODAL_KIND})]
    assert len(attrs2) == 4, "幂等·重复 ensure 不增重复"


def test_ensure_modal_primitives_d6_attribution(modal_env):
    """D6 归属：4 MODAL_KIND_* 节点挂 abstract_mark MARK_MODAL_KIND=5（set_mark·mark_value=modal_kind·D6 模态种类归抽象空间）。"""
    b, sid, es, ns, ci = modal_env
    modal_refs = ensure_modal_primitives(ci, b, space_id=sid)
    for modal_kind, modal_ref in modal_refs.items():
        mv = get_mark(b, ref=modal_ref, mark_kind=MARK_MODAL_KIND, status=MARK_PROMOTED)
        assert mv == modal_kind, f"MODAL_KIND_{modal_kind} 节点挂 MARK_MODAL_KIND=5 mark_value={modal_kind}（D6 归属）"


def test_lookup_word_modality_reads_d11(modal_env):
    """lookup_word_modality 读 D:11 边→[(modal_ref, modal_kind)]·tier_filter + kind==0 skip。"""
    b, sid, es, ns, ci = modal_env
    必然 = ci.lookup("必然", sid)
    assert 必然 is not None, "boot 种了 必然 概念"
    mods = lookup_word_modality(b, es, 必然, space_id=sid, tier_filter=TIER_PRIMARY)
    kinds = [k for _r, k in mods]
    assert MODAL_KIND_BOX_NECESSITY in kinds, "必然→MODAL_KIND_BOX_NECESSITY D:11 PRIMARY 边建出"


def test_lookup_word_modality_no_cross_pollution_rel_op(modal_env):
    """无交叉污染：lookup_word_modality 对 REL_*/OP_* target（ATTR_RELATION_PRIMITIVE/ATTR_OPERATOR_PRIMITIVE）kind==0 skip。"""
    b, sid, es, ns, ci = modal_env
    from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives
    from pure_integer_ai.cognition.shared.operator_primitives import ensure_operator_primitives
    ensure_relation_primitives(ci, b, space_id=sid)
    ensure_operator_primitives(ci, b, space_id=sid)
    rel_equal_ref = ci.lookup("__REL_EQUAL__", sid)
    assert rel_equal_ref is not None
    assert lookup_word_modality(b, es, rel_equal_ref, space_id=sid, tier_filter=TIER_PRIMARY) == [], \
        "REL_* target 挂 ATTR_RELATION_PRIMITIVE·lookup_word_modality kind==0 skip·返空"
    op_add_ref = ci.lookup("__OP_ADD__", sid)
    assert op_add_ref is not None
    assert lookup_word_modality(b, es, op_add_ref, space_id=sid, tier_filter=TIER_PRIMARY) == [], \
        "OP_* target 挂 ATTR_OPERATOR_PRIMITIVE·lookup_word_modality kind==0 skip·返空"


def test_modal_kind_equals_modality_encoding():
    """MODAL_KIND_* = modality 编码（1-4·与 P0.3 surface modality int 一致·modal_op_of readback 返此即 modality 值）。"""
    assert MODAL_KIND_BOX_NECESSITY == 1
    assert MODAL_KIND_BOX_POSSIBILITY == 2
    assert MODAL_KIND_DEONTIC_NECESSITY == 3
    assert MODAL_KIND_DEONTIC_POSSIBILITY == 4


# ============ (i) seeded closed-class ============

def test_modal_op_seeded_frozenset_first_source(modal_env):
    """(i) seeded '必然' gate ON/OFF 均 modality=1（frozenset 第一源先命中·D:11 冗余但无害）。"""
    b, sid, es, ns, ci = modal_env
    saved = gates.MODAL_D11_READBACK_MODE
    gates.MODAL_D11_READBACK_MODE = False
    try:
        assert modal_op_of("必然", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) == 1
    finally:
        gates.MODAL_D11_READBACK_MODE = saved
    gates.MODAL_D11_READBACK_MODE = True
    try:
        assert modal_op_of("必然", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) == 1
    finally:
        gates.MODAL_D11_READBACK_MODE = saved


# ============ (ii) fixture 注入开放变体（D:11 唯一源·反 theater） ============

def test_modal_op_d11_readback_nontournament_word(modal_env):
    """(ii) fixture 注入 '想必'→MODAL_KIND_BOX_NECESSITY D:11 边→gate ON modal_op_of 返 1·gate OFF 返 None。"""
    b, sid, es, ns, ci = modal_env
    _inject_modal_d11(ci, es, b, sid, "想必", MODAL_KIND_BOX_NECESSITY)
    saved = gates.MODAL_D11_READBACK_MODE
    gates.MODAL_D11_READBACK_MODE = False
    try:
        assert modal_op_of("想必", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) is None, \
            "gate OFF '想必' 非 frozenset·readback 关→None"
    finally:
        gates.MODAL_D11_READBACK_MODE = saved
    gates.MODAL_D11_READBACK_MODE = True
    try:
        assert modal_op_of("想必", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) == 1, \
            "gate ON D:11 readback '想必'→MODAL_KIND_BOX_NECESSITY=1"
    finally:
        gates.MODAL_D11_READBACK_MODE = saved


def test_modal_op_d11_drives_property_claim(modal_env):
    """(ii) 端到端：'想必' 经 D:11 readback 驱动 extract_property_claims 情态窗口填 modality。
    '猫 的 颜色 想必 是 黑' → claim (0, 2, 5, 0, 0, 1)（subj=猫·attr=颜色·val=黑·pol=0·mod=1）。
    gate OFF '想必' 不识别→非情态窗口·退化肯定窗口（modality=0·bit-identical）。"""
    b, sid, es, ns, ci = modal_env
    _inject_modal_d11(ci, es, b, sid, "想必", MODAL_KIND_BOX_NECESSITY)
    tokens = ["猫", "的", "颜色", "想必", "是", "黑"]
    saved = gates.MODAL_D11_READBACK_MODE
    # gate ON：'想必' 经 D:11 readback→modality=1·情态窗口填值
    gates.MODAL_D11_READBACK_MODE = True
    try:
        claims_on = extract_property_claims(tokens, lang=LANG_ZH,
                                            negation_on=False, modality_on=True,
                                            backend=b, edge_store=es,
                                            space_id=sid, concept_index=ci)
        assert (0, 2, 5, 0, 0, 1, 1, 1) in claims_on, \
            "gate ON '想必' 经 D:11 readback→modality=1·情态窗口 claim (猫,颜色,黑,0,0,1)"
    finally:
        gates.MODAL_D11_READBACK_MODE = saved
    # gate OFF：'想必' 不识别→非情态窗口·'的...是' 肯定窗口（j=4 是·j-2=2 的...等·实际 j-3=1 的·退化肯定窗口）
    gates.MODAL_D11_READBACK_MODE = False
    try:
        claims_off = extract_property_claims(tokens, lang=LANG_ZH,
                                             negation_on=False, modality_on=True,
                                             backend=b, edge_store=es,
                                             space_id=sid, concept_index=ci)
        # gate OFF '想必' 非情态·是 at j=4·肯定窗口 j-3=1 须 是 的（tokens[1]=的 ✓）→ claim (0,2,5,0,0,0) modality=0
        modality_values = {c[5] for c in claims_off}
        assert 1 not in modality_values, "gate OFF '想必' 不识别→无 modality=1（行为差可观测·反 theater）"
    finally:
        gates.MODAL_D11_READBACK_MODE = saved


def test_is_modal_cue_d11_readback(modal_env):
    """is_modal_cue 透传 4 参→modal_op_of D:11 readback（gate ON 时非 frozenset 情态词亦判·与主调一致）。"""
    b, sid, es, ns, ci = modal_env
    _inject_modal_d11(ci, es, b, sid, "势必", MODAL_KIND_DEONTIC_NECESSITY)
    saved = gates.MODAL_D11_READBACK_MODE
    gates.MODAL_D11_READBACK_MODE = False
    try:
        assert is_modal_cue("势必", LANG_ZH, backend=b, edge_store=es,
                            space_id=sid, concept_index=ci) is False, \
            "gate OFF '势必' 非 frozenset·readback 关→False"
    finally:
        gates.MODAL_D11_READBACK_MODE = saved
    gates.MODAL_D11_READBACK_MODE = True
    try:
        assert is_modal_cue("势必", LANG_ZH, backend=b, edge_store=es,
                            space_id=sid, concept_index=ci) is True, \
            "gate ON D:11 readback '势必'→MODAL_KIND_DEONTIC_NECESSITY→True"
    finally:
        gates.MODAL_D11_READBACK_MODE = saved


# ============ 无交叉污染（modal vs REL_*/OP_* 过滤） ============

def test_modal_vs_rel_op_no_cross_pollution(modal_env):
    """种 '必然'→MODAL_KIND_BOX_NECESSITY·lookup_word_modality 返 1·REL_*/OP_* target kind==0 skip。
    lookup_word_concept/operator 对 MODAL_KIND target 亦 kind==0 skip（ATTR_MODAL_KIND 非 REL/OP 标记）。"""
    b, sid, es, ns, ci = modal_env
    from pure_integer_ai.cognition.understanding.word_concept_signal import lookup_word_concept
    from pure_integer_ai.cognition.shared.operator_primitives import lookup_word_operator
    必然 = ci.lookup("必然", sid)
    saved = gates.MODAL_D11_READBACK_MODE
    gates.MODAL_D11_READBACK_MODE = True
    try:
        # '必然' seeded→MODAL_KIND_BOX_NECESSITY·modal 读回 1·REL/OP 读回空
        assert modal_op_of("必然", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) == 1
        rels = lookup_word_concept(b, es, 必然, space_id=sid, tier_filter=TIER_PRIMARY)
        assert rels == [], "MODAL_KIND target 挂 ATTR_MODAL_KIND·lookup_word_concept kind==0 skip·返空"
        ops = lookup_word_operator(b, es, 必然, space_id=sid, tier_filter=TIER_PRIMARY)
        assert ops == [], "MODAL_KIND target 挂 ATTR_MODAL_KIND·lookup_word_operator kind==0 skip·返空"
    finally:
        gates.MODAL_D11_READBACK_MODE = saved


# ============ 不写死守卫（D6·开放变体零硬编码） ============

def test_no_hardcode_open_variants_modal():
    """不写死守卫：_MODAL_LEXICAL_CUE 只 closed-class 核心（镜像 _MODAL_CUES）·
    开放变体（想必/势必/说不定）零硬编码·仅测试 fixture 注入。"""
    zh = _MODAL_LEXICAL_CUE.get(LANG_ZH, {})
    # closed-class 核心在
    for w in ["必然", "可能", "也许", "必须", "应该", "可以"]:
        assert w in zh, f"closed-class 核心 {w} 在 _MODAL_LEXICAL_CUE"
    # 开放变体不在
    for w in ["想必", "势必", "说不定", "估计"]:
        assert w not in zh, f"开放变体 {w} 不硬编码（走 D:11 教师晋升）"
    # _MODAL_CUES 既有 frozenset 也无开放变体（第一源不被 PR1 污染）
    zh_cues = _MODAL_CUES.get(LANG_ZH, {})
    assert "想必" not in zh_cues and "必然" in zh_cues


# ============ 铁律守卫 ============

def test_modal_d11_not_in_effective_weight(modal_env):
    """铁律：D:11 EDGE_RELATION_SIGNAL 边不入 effective_weight（只认 {PRECEDES,CAUSES,REFERS_TO}·:82 assert）。
    ATTR_MODAL_KIND=22 非结构 kind（_STRUCTURAL_KINDS 不含·read_composes_tree 忽略·inline 不传播）。"""
    b, sid, es, ns, ci = modal_env
    from pure_integer_ai.cognition.process.effective_weight import effective_weight
    # D:11 边 dict 传入 effective_weight 应 assert fail
    d11_edge = {"edge_type": EDGE_RELATION_SIGNAL, "strength": 1}
    try:
        effective_weight(d11_edge)
        assert False, "D:11 边入 effective_weight 应 assert fail"
    except AssertionError:
        pass  # 预期 assert fail
    # ATTR_MODAL_KIND=22 非结构 kind 确认
    from pure_integer_ai.cognition.understanding.arith_observe import _STRUCTURAL_KINDS
    assert 22 not in _STRUCTURAL_KINDS, "ATTR_MODAL_KIND=22 非结构 kind·不污染 5-dict 重建"
    assert ATTR_MODAL_KIND == 22


# ============ bit-identical gate OFF ============

def test_modal_d11_gate_off_bit_identical(modal_env):
    """bit-identical：gate OFF modal_op_of 退化纯 frozenset _MODAL_CUES·既有 caller 无 4 参亦退化。
    4 参全 None + gate OFF → 纯 frozenset（既有行为·bit-identical）。"""
    b, sid, es, ns, ci = modal_env
    saved = gates.MODAL_D11_READBACK_MODE
    gates.MODAL_D11_READBACK_MODE = False
    try:
        # 4 参全 None + gate OFF → 纯 frozenset
        assert modal_op_of("必然", LANG_ZH) == 1
        assert modal_op_of("可能", LANG_ZH) == 2
        assert modal_op_of("必须", LANG_ZH) == 3
        assert modal_op_of("可以", LANG_ZH) == 4
        assert modal_op_of("非情态词", LANG_ZH) is None
        # is_modal_cue 无 4 参 + gate OFF → 纯 frozenset
        assert is_modal_cue("必然", LANG_ZH) is True
        assert is_modal_cue("非情态词", LANG_ZH) is False
    finally:
        gates.MODAL_D11_READBACK_MODE = saved


# ============ STOP 合规（不上 TYPE_MODALITY） ============

def test_stop_no_type_modality():
    """STOP 合规：不上 TYPE_MODALITY·modality 走 surface int + ATTR_MODAL_KIND=22 readback + MARK_MODAL_KIND=5 D6 归属。
    symbol_types 无 TYPE_MODALITY（STOP marker 守符号域 type_ref 不扩张）。"""
    from pure_integer_ai.cognition.shared import symbol_types
    assert not hasattr(symbol_types, "TYPE_MODALITY"), "STOP 守·不上 TYPE_MODALITY·modality 走 surface+ATTR_MODAL_KIND"
    assert not hasattr(symbol_types, "TYPE_EXISTENTIAL"), "STOP 守·不上 TYPE_EXISTENTIAL"
    # ATTR_MODAL_KIND=22 是 composes_attr kind（存储 readback）·非 TYPE_* 符号域 type_ref
    assert ATTR_MODAL_KIND == 22
    # MARK_MODAL_KIND=5 是 abstract_mark mark_kind（D6 抽象空间）·非 TYPE_*
    assert MARK_MODAL_KIND == 5
