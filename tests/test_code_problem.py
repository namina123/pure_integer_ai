"""language→code 桥测试（doc/重来_语言通用接地 §七-bis piece 2·条件句接地 + code 执行）。

piece 2 = 语言域主攻第二刀（语言嵌入代码·控制流）·结构→结构·vm_proof 验·不撞 D墙。
泛化 piece 1（[num op num] 纯表达式）到 if-else 控制流。
覆盖：① 条件真分支返 then ② 条件假分支返 else ③ 小于比较 ④ 负控（非 8 元/结构 cue 缺/数字未接地→None）
⑤ 冷启动 None ⑥ EN（greater_than）。

铁律：纯整数 / 确定性 bit-identical / 不写死（数字数据驱动 + 比较/条件 cue 元定义种子）/ 关联在图中（数字 PURE_ALIAS 图边）/ 反 theater（vm_proof 执行 if/Compare/Return）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.storage.concept_correspondence import register_concept_correspondence
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
from pure_integer_ai.cognition.understanding.number_grounding import bootstrap_number_grounding
from pure_integer_ai.cognition.understanding.code_problem import code_problem_value
from pure_integer_ai.cognition.understanding.cue_words import (
    cond_keyword_of, cue_type_of, _COND_IF, _COND_THEN, _COND_ELSE)
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.crosscut.integer.rational import make


@pytest.fixture
def env():
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    register_concept_correspondence(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    yield b, sp.space_id, es, ci
    b.close()


_ZH_FACTS = [("零", 1, 0), ("一", 1, 1), ("二", 1, 2), ("三", 1, 3), ("四", 1, 4),
             ("五", 1, 5), ("六", 1, 6), ("七", 1, 7), ("八", 1, 8), ("九", 1, 9), ("十", 1, 10)]


# ============ cond_keyword_of：旧字面 compatibility 回归 ============

def test_cond_keyword_of_seed():
    """未注入图 runtime 时，旧条件字面仍可作为显式兼容路径读取。"""
    assert cond_keyword_of("如果", LANG_ZH) == _COND_IF
    assert cond_keyword_of("那么", LANG_ZH) == _COND_THEN
    assert cond_keyword_of("否则", LANG_ZH) == _COND_ELSE
    assert cond_keyword_of("if", LANG_EN) == _COND_IF
    assert cond_keyword_of("苹果", LANG_ZH) is None, "非条件结构词→None"
    assert cond_keyword_of("大于", LANG_ZH) is None, "比较词非条件结构词→None（独立两族）"
    # E1（2 对抗审）：§九隔离不变量锁死——条件结构词不入 _CUE_WORDS·cue_type_of 返 None（零 extract_cues 污染·bit-identical）
    assert cue_type_of("如果", LANG_ZH) is None, "如果不入 _CUE_WORDS（独立两族·bit-identical·防回归）"
    assert cue_type_of("那么", LANG_ZH) is None
    assert cue_type_of("否则", LANG_ZH) is None


# ============ code_problem_value：端到端 vm_proof（控制流·反 theater） ============

def test_code_problem_conditional_true(env):
    """如果三大于二那么一否则零 → 1（3>2 真→then 分支返 1·vm_proof 真执行 if/Compare/Return）。"""
    b, sid, es, ci = env
    bootstrap_number_grounding(ci, es, b, _ZH_FACTS, space_id=sid)

    def cp(toks):
        return code_problem_value(toks, concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, lang=LANG_ZH)
    assert rational.eq(cp(["如果", "三", "大于", "二", "那么", "一", "否则", "零"]), make(1, 1)), "3>2 真→返 1"


def test_code_problem_conditional_false_branch(env):
    """如果二大于三那么一否则零 → 0（2>3 假→else 分支返 0·vm_proof 真执行假分支·非 parser 猜）。"""
    b, sid, es, ci = env
    bootstrap_number_grounding(ci, es, b, _ZH_FACTS, space_id=sid)

    def cp(toks):
        return code_problem_value(toks, concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, lang=LANG_ZH)
    assert rational.eq(cp(["如果", "二", "大于", "三", "那么", "一", "否则", "零"]), make(0, 1)), "2>3 假→返 0"


def test_code_problem_less_than(env):
    """如果二小于三那么一否则零 → 1（2<3 真·小于比较 CMP_LT→OPCODE_LT·vm_proof）。"""
    b, sid, es, ci = env
    bootstrap_number_grounding(ci, es, b, _ZH_FACTS, space_id=sid)

    def cp(toks):
        return code_problem_value(toks, concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, lang=LANG_ZH)
    assert rational.eq(cp(["如果", "二", "小于", "三", "那么", "一", "否则", "零"]), make(1, 1)), "2<3 真→返 1"
    # 假分支：三小于二 → 0
    assert rational.eq(cp(["如果", "三", "小于", "二", "那么", "一", "否则", "零"]), make(0, 1)), "3<2 假→返 0"


def test_code_problem_equality(env):
    """如果三等于三那么一否则零 → 1（3==3 真·CMP_EQ→源码 `==`→code_observe ast.Eq→OPCODE_EQ→vm_proof 真执行 Compare(Eq)）。
    piece 2.1 比较族补全：等于/EQ 关系（piece 2 显式 defer 的等式一支·补既有比较维度·非 pivot 所指"广度扩"新支）。
    反 theater：假分支三等于二→0（vm_proof 真执行 else·非 parser 猜）。"""
    b, sid, es, ci = env
    bootstrap_number_grounding(ci, es, b, _ZH_FACTS, space_id=sid)

    def cp(toks):
        return code_problem_value(toks, concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, lang=LANG_ZH)
    assert rational.eq(cp(["如果", "三", "等于", "三", "那么", "一", "否则", "零"]), make(1, 1)), "3==3 真→返 1"
    # 假分支：三等于二 → 0（3!=2·else 分支返 0·vm_proof 真执行假分支·非 parser 猜）
    assert rational.eq(cp(["如果", "三", "等于", "二", "那么", "一", "否则", "零"]), make(0, 1)), "3==2 假→返 0"


def test_code_problem_negative_controls(env):
    """负控：非 8 元 / 结构 cue 缺（无如果/那么/否则）/ 数字词未接地 / GE-LE defer → None（守反统计契约·不凑配）。"""
    b, sid, es, ci = env
    bootstrap_number_grounding(ci, es, b, _ZH_FACTS, space_id=sid)

    def cp(toks):
        return code_problem_value(toks, concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, lang=LANG_ZH)
    # 非 8 元
    assert cp(["如果", "三", "大于", "二", "那么", "一"]) is None, "非 8 元→None"
    # 结构 cue 缺（首词非如果）
    assert cp(["三", "三", "大于", "二", "那么", "一", "否则", "零"]) is None, "首词非如果→None"
    # 数字词未接地（苹果非数字词）
    assert cp(["如果", "苹果", "大于", "二", "那么", "一", "否则", "零"]) is None, "左操作数未接地→None"
    # GE/LE defer（不小于→CMP_GE·_CMP_TO_SYM 无 GE→None·code_observe 不支持 Ge/Le）
    assert cp(["如果", "三", "不小于", "二", "那么", "一", "否则", "零"]) is None, "GE defer→None"


def test_code_problem_cold_start_none(env):
    """冷启动（未 bootstrap·无图边）→ None（数字词未接地·resolve_number_word 走图无 PURE_ALIAS 边）。"""
    b, sid, es, ci = env
    val = code_problem_value(["如果", "三", "大于", "二", "那么", "一", "否则", "零"],
                             concept_index=ci, edge_store=es, backend=b,
                             space_id=sid, lang=LANG_ZH)
    assert val is None, "冷启动（未 bootstrap）→ None（数字词未接地）"


def test_code_problem_en(env):
    """EN：if three greater_than two then one else zero = 1（EN facts + EN 条件/比较 cue·跨语言控制流）·vm_proof 验。"""
    b, sid, es, ci = env
    en_facts = [("zero", 2, 0), ("one", 2, 1), ("two", 2, 2), ("three", 2, 3)]
    bootstrap_number_grounding(ci, es, b, en_facts, space_id=sid)
    val = code_problem_value(["if", "three", "greater_than", "two", "then", "one", "else", "zero"],
                             concept_index=ci, edge_store=es, backend=b,
                             space_id=sid, lang=LANG_EN)
    assert rational.eq(val, make(1, 1)), "if three greater_than two then one else zero = 1（EN·控制流 vm_proof）"
    # EN 等式（piece 2.1）：if three equals three then one else zero = 1（CMP_EQ·EN equals cue·刀B ARITH_EQUALS_CUE 单源·vm_proof）
    val_eq = code_problem_value(["if", "three", "equals", "three", "then", "one", "else", "zero"],
                                concept_index=ci, edge_store=es, backend=b,
                                space_id=sid, lang=LANG_EN)
    assert rational.eq(val_eq, make(1, 1)), "if three equals three then one else zero = 1（EN 等式·vm_proof）"
