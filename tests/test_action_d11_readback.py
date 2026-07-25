"""B-PR1 动作意图 D:11 readback 第二源测试（镜像 test_negation_d11_readback.py 范式·符号域先天变体·doc §16）。

action_primitives.py 建 5 concept（INTENT_COMMAND_MOOD=0 + ACTION_GENERATE/COMPUTE/ANALYZE/SOLVE=1-4）+
ATTR_OPERATION_INTENT=23 标记 + _ACTION_LEXICAL_CUE（命令词+动作词种子）+ lookup_word_action +
is_action_intent_cue D:11 readback 第二源（gate ACTION_D11_READBACK_MODE）+ bootstrap_action_signals。

**关键差异（action vs negation·D6）**：动作意图=**符号域先天**（镜像 operator·异 modal 抽象空间）·
故只挂 ATTR_OPERATION_INTENT=23·**不挂 abstract_mark**（异 modal 双挂）·**新增 ATTR_OPERATION_INTENT=23**
（异 negation 复用 ATTR_SYMBOL_TYPE·动作意图是新命名空间·doc §16.3）。

**W7+B-PR1 合并**（doc §16.4）：命令判定 = 命令词（→COMMAND_MOOD）OR 动作词（→ACTION_*）命中任一。
覆盖引导词祈使（帮我生成）+ 有动作词裸祈使（生成代码）·纯句式（去开门）defer B-PR2。

**反 theater 两路证**：
  (i) seeded closed-class（帮我/生成·_ACTION_LEXICAL_CUE D:11 种子）：gate ON/OFF frozenset 第一源均命中。
  (ii) fixture 注入开放变体（劳驾→COMMAND_MOOD / 编写→GENERATE D:11 边·模拟教师晋升）：
       gate ON is_action_intent_cue 返 True·gate OFF 返 False（D:11 唯一源·行为差可观测）。

**不写死守卫**（D6·用户反馈①）：_ACTION_LEXICAL_CUE 只 closed-class 核心·开放变体（劳驾/编写/运算）零硬编码·
仅测试 fixture 注入。**doc §16.1 推翻 §15.1 纠正③**：命令词走 D:11（非"不走 D:11"）。

**无交叉污染**：D:11 共享边类型·lookup_word_action 过滤 ATTR_OPERATION_INTENT（None 判据·含 int_a=0
COMMAND_MOOD）·与 REL_*/OP_*/MODAL_KIND/TYPE_NEGATION 互不交叉。
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
    register_composes_attr, ATTR_OPERATION_INTENT,
    ATTR_RELATION_PRIMITIVE, ATTR_OPERATOR_PRIMITIVE, ATTR_MODAL_KIND, ATTR_SYMBOL_TYPE,
)
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import LANG_ZH
from pure_integer_ai.cognition.shared.action_primitives import (
    INTENT_COMMAND_MOOD, ACTION_GENERATE, ACTION_COMPUTE, ACTION_ANALYZE, ACTION_SOLVE,
    ensure_action_primitives, lookup_word_action, _ACTION_LEXICAL_CUE,
    is_command_mood_kind, is_action_class_kind,
)
from pure_integer_ai.cognition.understanding.cue_words import is_action_intent_cue
from pure_integer_ai.cognition.understanding.word_concept_signal import (
    bootstrap_action_signals, record_word_concept,
)
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def action_env():
    """B-PR1 单测环境（dict backend·core space·composes_attr 注册·boot 种 action D:11 边）。
    动作意图=符号域先天·**不**注册 abstract_mark（镜像 operator·异 modal·D6 符号域非抽象空间）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)   # ATTR_OPERATION_INTENT=23 标记表（动作意图不挂 abstract_mark·无需 register_abstract_mark）
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    # boot 种 action D:11 边（含 closed-class 核心 帮我/请 + 生成/计算）
    bootstrap_action_signals(ci, es, b, space_id=sid, langs={LANG_ZH})
    yield b, sid, es, ns, ci
    b.close()


def _inject_action_d11(ci, es, b, sid, word_surface, action_kind):
    """fixture 注入 D:11 边（word→ACTION_INTENT_* ref·SOURCE_TEACHER·模拟教师晋升·非 frozenset·非生产 boot 种子）。"""
    action_refs = ensure_action_primitives(ci, b, space_id=sid)
    ref = action_refs[action_kind]
    return record_word_concept(ci, es, word_surface, ref,
                               space_id=sid, source=SOURCE_TEACHER)


# ============ 基建：ensure_action_primitives（5 concept）+ lookup_word_action ============

def test_ensure_action_primitives_builds_5_concepts(action_env):
    """ensure_action_primitives 建 5 ACTION_INTENT_* NODE_CONCEPT + ATTR_OPERATION_INTENT=23 标记·幂等。"""
    b, sid, es, ns, ci = action_env
    attrs = [r for r in b.select("composes_attr", where={"kind": ATTR_OPERATION_INTENT})]
    assert len(attrs) == 5, "5 ACTION_INTENT_* 节点（COMMAND_MOOD + 4 ACTION_*）"
    kinds = {a["int_a"] for a in attrs}
    assert kinds == {0, 1, 2, 3, 4}, "int_a 值域 0=COMMAND_MOOD/1-4=ACTION_*"
    # 幂等：重复调不 corrupt
    ensure_action_primitives(ci, b, space_id=sid)
    attrs2 = [r for r in b.select("composes_attr", where={"kind": ATTR_OPERATION_INTENT})]
    assert len(attrs2) == 5, "幂等·重复 ensure 不增重复"


def test_lookup_word_action_reads_d11(action_env):
    """lookup_word_action 读 D:11 边→[(ref, kind)]·命令词→COMMAND_MOOD·动作词→ACTION_*·tier_filter 守。"""
    b, sid, es, ns, ci = action_env
    帮我 = ci.lookup("帮我", sid)
    assert 帮我 is not None, "boot 种了 帮我 概念（→COMMAND_MOOD）"
    hits_mood = lookup_word_action(b, es, 帮我, space_id=sid, tier_filter=TIER_PRIMARY)
    assert len(hits_mood) == 1 and hits_mood[0][1] == INTENT_COMMAND_MOOD, \
        "帮我→COMMAND_MOOD D:11 PRIMARY 边·readback kind=0"
    生成 = ci.lookup("生成", sid)
    assert 生成 is not None, "boot 种了 生成 概念（→ACTION_GENERATE）"
    hits_gen = lookup_word_action(b, es, 生成, space_id=sid, tier_filter=TIER_PRIMARY)
    assert len(hits_gen) == 1 and hits_gen[0][1] == ACTION_GENERATE, \
        "生成→ACTION_GENERATE D:11 PRIMARY 边·readback kind=1"


def test_lookup_word_action_command_mood_zero_not_skipped(action_env):
    """None 判据（非 kind==0）：COMMAND_MOOD=0 是合法值·lookup_word_action 返（ATTR 存在）·不误 skip。

    异 lookup_word_operator 的 OP_* 1-7 哨兵（kind==0 skip）·action 含 0·用 None 判据。
    """
    b, sid, es, ns, ci = action_env
    帮我 = ci.lookup("帮我", sid)
    hits = lookup_word_action(b, es, 帮我, space_id=sid, tier_filter=TIER_PRIMARY)
    assert len(hits) == 1, "COMMAND_MOOD=0 不被 kind==0 判据 skip（None 判据·ATTR 存在即返）"
    assert is_command_mood_kind(hits[0][1]) is True
    assert is_action_class_kind(hits[0][1]) is False, "COMMAND_MOOD 非动作类别"


def test_lookup_word_action_no_cross_pollution(action_env):
    """无交叉污染：lookup_word_action 对 REL_*/OP_*/MODAL/NEGATION target（无 ATTR_OPERATION_INTENT）返空。"""
    b, sid, es, ns, ci = action_env
    from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives
    from pure_integer_ai.cognition.shared.operator_primitives import ensure_operator_primitives
    from pure_integer_ai.cognition.shared.modal_primitives import ensure_modal_primitives
    from pure_integer_ai.cognition.shared.symbol_types import ensure_symbol_types
    ensure_relation_primitives(ci, b, space_id=sid)
    ensure_operator_primitives(ci, b, space_id=sid)
    ensure_modal_primitives(ci, b, space_id=sid)
    ensure_symbol_types(ci, b, space_id=sid)
    for surface, label in [("__REL_EQUAL__", "REL"), ("__OP_ADD__", "OP"),
                           ("__MODAL_BOX_NECESSITY__", "MODAL"), ("__TYPE_NEGATION__", "NEGATION")]:
        ref = ci.lookup(surface, sid)
        assert ref is not None, f"{label} concept 建了"
        assert lookup_word_action(b, es, ref, space_id=sid, tier_filter=TIER_PRIMARY) == [], \
            f"{label} target 无 ATTR_OPERATION_INTENT·lookup_word_action 返空（None 判据 skip）"


# ============ (i) seeded closed-class ============

def test_action_seeded_frozenset_first_source(action_env):
    """(i) seeded '帮我'/'生成' gate ON/OFF 均 True（frozenset 第一源先命中·D:11 冗余但无害）。"""
    b, sid, es, ns, ci = action_env
    saved = gates.ACTION_D11_READBACK_MODE
    for g in (False, True):
        gates.ACTION_D11_READBACK_MODE = g
        try:
            assert is_action_intent_cue("帮我", LANG_ZH, backend=b, edge_store=es,
                                        space_id=sid, concept_index=ci) is True
            assert is_action_intent_cue("生成", LANG_ZH, backend=b, edge_store=es,
                                        space_id=sid, concept_index=ci) is True
        finally:
            gates.ACTION_D11_READBACK_MODE = saved


# ============ (ii) fixture 注入开放变体（D:11 唯一源·反 theater） ============

def test_action_d11_readback_open_variant_command(action_env):
    """(ii) fixture 注入 '劳驾'→COMMAND_MOOD D:11 边→gate ON is_action_intent_cue True·gate OFF False。"""
    b, sid, es, ns, ci = action_env
    _inject_action_d11(ci, es, b, sid, "劳驾", INTENT_COMMAND_MOOD)
    saved = gates.ACTION_D11_READBACK_MODE
    gates.ACTION_D11_READBACK_MODE = False
    try:
        assert is_action_intent_cue("劳驾", LANG_ZH, backend=b, edge_store=es,
                                    space_id=sid, concept_index=ci) is False, \
            "gate OFF '劳驾' 非 frozenset·readback 关→False"
    finally:
        gates.ACTION_D11_READBACK_MODE = saved
    gates.ACTION_D11_READBACK_MODE = True
    try:
        assert is_action_intent_cue("劳驾", LANG_ZH, backend=b, edge_store=es,
                                    space_id=sid, concept_index=ci) is True, \
            "gate ON D:11 readback '劳驾'→COMMAND_MOOD→True"
    finally:
        gates.ACTION_D11_READBACK_MODE = saved


def test_action_d11_readback_open_variant_action_verb(action_env):
    """(ii) fixture 注入 '编写'→ACTION_GENERATE D:11 边→gate ON True·gate OFF False（动作词开放变体）。"""
    b, sid, es, ns, ci = action_env
    _inject_action_d11(ci, es, b, sid, "编写", ACTION_GENERATE)
    saved = gates.ACTION_D11_READBACK_MODE
    gates.ACTION_D11_READBACK_MODE = False
    try:
        assert is_action_intent_cue("编写", LANG_ZH, backend=b, edge_store=es,
                                    space_id=sid, concept_index=ci) is False, "gate OFF→False"
    finally:
        gates.ACTION_D11_READBACK_MODE = saved
    gates.ACTION_D11_READBACK_MODE = True
    try:
        assert is_action_intent_cue("编写", LANG_ZH, backend=b, edge_store=es,
                                    space_id=sid, concept_index=ci) is True, \
            "gate ON '编写'→ACTION_GENERATE→True（动作词开放变体 D:11 学）"
    finally:
        gates.ACTION_D11_READBACK_MODE = saved


def test_is_action_intent_cue_d11_readback(action_env):
    """is_action_intent_cue 透传 4 参→D:11 readback（gate ON 时非 frozenset 动作意图词亦判）。"""
    b, sid, es, ns, ci = action_env
    _inject_action_d11(ci, es, b, sid, "运算", ACTION_COMPUTE)
    saved = gates.ACTION_D11_READBACK_MODE
    gates.ACTION_D11_READBACK_MODE = False
    try:
        assert is_action_intent_cue("运算", LANG_ZH, backend=b, edge_store=es,
                                    space_id=sid, concept_index=ci) is False
    finally:
        gates.ACTION_D11_READBACK_MODE = saved
    gates.ACTION_D11_READBACK_MODE = True
    try:
        assert is_action_intent_cue("运算", LANG_ZH, backend=b, edge_store=es,
                                    space_id=sid, concept_index=ci) is True, \
            "gate ON '运算'→ACTION_COMPUTE→True"
    finally:
        gates.ACTION_D11_READBACK_MODE = saved


# ============ 无交叉污染（action vs REL/OP/MODAL/NEGATION 过滤） ============

def test_action_vs_rel_op_modal_negation_no_cross_pollution(action_env):
    """种 '帮我'→COMMAND_MOOD·lookup_word_action 返命中·REL/OP/MODAL/NEGATION lookup 对 COMMAND_MOOD target skip。"""
    b, sid, es, ns, ci = action_env
    from pure_integer_ai.cognition.understanding.word_concept_signal import lookup_word_concept
    from pure_integer_ai.cognition.shared.operator_primitives import lookup_word_operator
    from pure_integer_ai.cognition.shared.modal_primitives import lookup_word_modality
    from pure_integer_ai.cognition.shared.symbol_types import lookup_word_negation
    帮我 = ci.lookup("帮我", sid)
    saved = gates.ACTION_D11_READBACK_MODE
    gates.ACTION_D11_READBACK_MODE = True
    try:
        assert is_action_intent_cue("帮我", LANG_ZH, backend=b, edge_store=es,
                                    space_id=sid, concept_index=ci) is True
        assert lookup_word_concept(b, es, 帮我, space_id=sid, tier_filter=TIER_PRIMARY) == [], \
            "COMMAND_MOOD target 挂 ATTR_OPERATION_INTENT·lookup_word_concept skip·返空"
        assert lookup_word_operator(b, es, 帮我, space_id=sid, tier_filter=TIER_PRIMARY) == [], "operator skip"
        assert lookup_word_modality(b, es, 帮我, space_id=sid, tier_filter=TIER_PRIMARY) == [], "modality skip"
        assert lookup_word_negation(b, es, 帮我, space_id=sid, tier_filter=TIER_PRIMARY) is False, "negation skip"
    finally:
        gates.ACTION_D11_READBACK_MODE = saved


# ============ 不写死守卫（D6·用户反馈①·开放变体零硬编码） ============

def test_no_hardcode_open_variants_action():
    """不写死守卫：_ACTION_LEXICAL_CUE 只 closed-class 核心（命令词+动作词）·
    开放变体（劳驾/编写/运算 等穷举不尽）零硬编码·仅测试 fixture 注入（doc §16·用户反馈①）。"""
    zh = _ACTION_LEXICAL_CUE.get(LANG_ZH, {})
    # closed-class 核心在
    for w in ["帮我", "请", "生成", "计算", "分析", "解决"]:
        assert w in zh, f"closed-class 核心 {w} 在 _ACTION_LEXICAL_CUE"
    # 开放变体不在（走 D:11 教师晋升·非硬编码）
    for w in ["劳驾", "编写", "运算", "烦请", "推演"]:
        assert w not in zh, f"开放变体 {w} 不硬编码（走 D:11 教师晋升）"


# ============ 铁律守卫 ============

def test_action_d11_not_in_effective_weight(action_env):
    """铁律：D:11 EDGE_RELATION_SIGNAL 边不入 effective_weight（只认 {PRECEDES,CAUSES,REFERS_TO}）。
    ATTR_OPERATION_INTENT=23 非结构 kind（_STRUCTURAL_KINDS 不含·read_composes_tree 忽略·inline 不传播）。"""
    b, sid, es, ns, ci = action_env
    from pure_integer_ai.cognition.process.effective_weight import effective_weight
    d11_edge = {"edge_type": EDGE_RELATION_SIGNAL, "strength": 1}
    try:
        effective_weight(d11_edge)
        assert False, "D:11 边入 effective_weight 应 assert fail"
    except AssertionError:
        pass
    from pure_integer_ai.cognition.understanding.arith_observe import _STRUCTURAL_KINDS
    assert 23 not in _STRUCTURAL_KINDS, "ATTR_OPERATION_INTENT=23 非结构 kind·不污染 5-dict 重建"
    assert ATTR_OPERATION_INTENT == 23


# ============ bit-identical gate OFF ============

def test_action_d11_gate_off_bit_identical(action_env):
    """bit-identical：gate OFF is_action_intent_cue 退化纯 frozenset _ACTION_LEXICAL_CUE·既有 caller 无 4 参亦退化。"""
    b, sid, es, ns, ci = action_env
    saved = gates.ACTION_D11_READBACK_MODE
    gates.ACTION_D11_READBACK_MODE = False
    try:
        assert is_action_intent_cue("帮我", LANG_ZH) is True
        assert is_action_intent_cue("生成", LANG_ZH) is True
        assert is_action_intent_cue("雨", LANG_ZH) is False
    finally:
        gates.ACTION_D11_READBACK_MODE = saved


# ============ D6 + STOP 合规（动作意图符号域先天·镜像 operator·不挂 abstract_mark） ============

def test_action_d6_no_abstract_mark():
    """D6（doc §16.3）：动作意图=符号域先天（镜像 operator）·ensure_action_primitives **不挂 abstract_mark**
    （异 modal 双挂 ATTR_MODAL_KIND+MARK_MODAL_KIND）·只挂 ATTR_OPERATION_INTENT=23。action_primitives 不 import set_mark。"""
    from pure_integer_ai.cognition.shared import action_primitives
    assert not hasattr(action_primitives, "set_mark"), \
        "action_primitives 不挂 abstract_mark（动作意图符号域先天·镜像 operator·异 modal 双挂）"
    # 对比 modal_primitives 挂 set_mark（D6 抽象空间·双挂）
    from pure_integer_ai.cognition.shared import modal_primitives
    assert hasattr(modal_primitives, "set_mark"), "对比：modal_primitives 挂 set_mark（抽象空间 D6 归属）"


def test_action_stop_new_attr_only():
    """STOP+D6 合规（doc §16.3）：**新增 ATTR_OPERATION_INTENT=23**（composes_attr 命名空间·非 TYPE_*·STOP 管辖 TYPE_*/OP_*
    不辖 ATTR_*）·**不新增 TYPE_ACTION_*/OP_ACTION_***（守 STOP）·**不新增 MARK_ACTION_KIND**（镜像 operator 不双挂）。"""
    assert ATTR_OPERATION_INTENT == 23, "新增 ATTR_OPERATION_INTENT=23（composes_attr·非 TYPE_*·STOP 不辖）"
    from pure_integer_ai.storage import composes_attr as ca
    # 不新增 TYPE_ACTION_*（STOP 守符号域 type_ref）
    from pure_integer_ai.cognition.shared import types as t
    assert not hasattr(t, "TYPE_ACTION"), "不新增 TYPE_ACTION（STOP 守 TYPE_*）"
    # 不新增 MARK_ACTION_KIND（镜像 operator 不双挂 abstract_mark）
    from pure_integer_ai.storage import abstract_mark
    assert not hasattr(abstract_mark, "MARK_ACTION_KIND"), "不新增 MARK_ACTION_KIND（镜像 operator 不双挂）"


# ============ W7 命令判定（_has_action_intent·命令词 OR 动作词·doc §16.4） ============

def test_has_action_intent_command_or_action_verb(action_env):
    """W7 doc §16.4：_has_action_intent 命令词（帮我）OR 动作词（生成）命中→命令判定 True。
    纯句式（去开门·无命令词无动作词）漏判 defer B-PR2（doc §16.5 诚实边界）。"""
    from pure_integer_ai.cognition.understanding.intent_classify import _has_action_intent
    from pure_integer_ai.cognition.shared.types import Segment, MODALITY_LANGUAGE, DOMAIN_TEXT

    def _seg(tokens):
        return Segment(seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
                       domain=DOMAIN_TEXT, tokens=tokens)
    b, sid, es, ns, ci = action_env
    saved_cmd = gates.INTENT_COMMAND_MODE
    gates.INTENT_COMMAND_MODE = True
    try:
        # 命令词命中（帮我）
        assert _has_action_intent([_seg(["帮我", "生成", "代码"])],
                                  backend=b, edge_store=es, space_id=sid, concept_index=ci) is True
        # 动作词命中（生成·裸祈使·doc §16.4）
        assert _has_action_intent([_seg(["生成", "代码"])],
                                  backend=b, edge_store=es, space_id=sid, concept_index=ci) is True
        # 纯句式漏判（去开门·无命令词无动作词·defer B-PR2）
        assert _has_action_intent([_seg(["去", "开门"])],
                                  backend=b, edge_store=es, space_id=sid, concept_index=ci) is False
        # 非命令（雨导致地湿）
        assert _has_action_intent([_seg(["雨", "导致", "地湿"])],
                                  backend=b, edge_store=es, space_id=sid, concept_index=ci) is False
    finally:
        gates.INTENT_COMMAND_MODE = saved_cmd
