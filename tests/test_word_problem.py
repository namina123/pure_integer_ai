"""language→arith 桥测试（doc/重来_语言通用接地 §六 piece 1·数字词接地 + word-problem 解析）。

piece 1 = 语言域主攻首刀（语言嵌入算数）·结构→结构·vm_proof 验·不撞 D墙。
覆盖：① bootstrap 建图边（词↔整数 PURE_ALIAS + CORR_NUMERIC 值·关联在图中）② 空 facts 零副作用（bit-identical）
③ resolve_number_word（图遍历读值）④ word_problem_value 端到端（三加二→5 等·vm_proof）⑤ 负控（非数字词/非二元→None）。

铁律：纯整数 / 确定性 bit-identical / 不写死（数据驱动 facts）/ 关联在图中（PURE_ALIAS 图边）/ 反 theater（vm_proof 验值）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.edge_store import EdgeStore, SUBTYPE_PURE_ALIAS
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.storage.concept_correspondence import (
    register_concept_correspondence, load_numeric)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.edge_types import EDGE_REFERS_TO
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
from pure_integer_ai.cognition.understanding.number_grounding import (
    bootstrap_number_grounding, resolve_number_word)
from pure_integer_ai.cognition.understanding.word_problem import word_problem_value
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


# ============ bootstrap：关联在图中（PURE_ALIAS 图边 + CORR_NUMERIC 值） ============

def test_bootstrap_builds_graph_edges(env):
    """bootstrap：facts → 整数概念 __int_{value}（CORR_NUMERIC 值）+ 词 PURE_ALIAS→整数概念（图边）。

    关联在图中（命门）：词↔整数 = PURE_ALIAS 图边（可遍历）·非旁侧表。三+three 共指 __int_3（跨词 dedup）。
    """
    b, sid, es, ci = env
    n = bootstrap_number_grounding(ci, es, b, _ZH_FACTS, space_id=sid)
    assert n > 0, "建边数>0（每 fact 双向 = 最多 2×11）"
    int3 = ci.ensure("__int_3", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    assert load_numeric(b, space_id=sid, local_id=int3[1]) == 3, "__int_3 CORR_NUMERIC=3（值在概念）"
    int10 = ci.ensure("__int_10", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    assert load_numeric(b, space_id=sid, local_id=int10[1]) == 10, "__int_10 值=10"
    san = ci.ensure("三", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    fwd = es.query_from(san[0], san[1], edge_type=EDGE_REFERS_TO)
    assert any(r.get("subtype") == SUBTYPE_PURE_ALIAS and r.get("local_id_to") == int3[1] for r in fwd), \
        "三 —PURE_ALIAS 图边→ __int_3（关联在图中）"


def test_bootstrap_empty_no_side_effects(env):
    """空 facts → return 0 + 零副作用（bit-identical 硬守·镜像 bootstrap_alias_edges 短路）。

    可观测验证（非脆弱表计数）：空 facts 短路后 __int_3 概念未建（lookup None）+ 三 未接地（resolve None）。
    """
    b, sid, es, ci = env
    assert bootstrap_number_grounding(ci, es, b, [], space_id=sid) == 0
    assert ci.lookup("__int_3", sid) is None, "空 facts 不建 __int_3（短路未 ensure）"
    assert resolve_number_word(ci, es, b, "三", space_id=sid) is None, "空 facts 不接地（无图边）"


# ============ resolve_number_word：图遍历读值（关联在图中·读侧） ============

def test_resolve_number_word_via_graph(env):
    """resolve_number_word：词→PURE_ALIAS 图遍历→整数概念→CORR_NUMERIC（关联在图中读侧）。"""
    b, sid, es, ci = env
    bootstrap_number_grounding(ci, es, b, _ZH_FACTS, space_id=sid)
    assert resolve_number_word(ci, es, b, "三", space_id=sid) == 3
    assert resolve_number_word(ci, es, b, "十", space_id=sid) == 10
    assert resolve_number_word(ci, es, b, "苹果", space_id=sid) is None, "非数字词→None（无 PURE_ALIAS→整数接地）"


# ============ word_problem_value：端到端 vm_proof（结构→结构·反 theater） ============

def test_word_problem_arith(env):
    """word_problem_value：[num,op,num]→数字接地（图）+算子 cue→arith 树→vm_proof 真·反 theater。

    三加二=5 / 十减三=7 / 二乘三=6 / 五加五=10（vm_proof 真执行·非 parser 算）。
    """
    b, sid, es, ci = env
    bootstrap_number_grounding(ci, es, b, _ZH_FACTS, space_id=sid)

    def wp(toks):
        return word_problem_value(toks, concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, lang=LANG_ZH)
    assert rational.eq(wp(["三", "加", "二"]), make(5, 1)), "三加二=5"
    assert rational.eq(wp(["十", "减", "三"]), make(7, 1)), "十减三=7"
    assert rational.eq(wp(["二", "乘", "三"]), make(6, 1)), "二乘三=6"
    assert rational.eq(wp(["五", "加", "五"]), make(10, 1)), "五加五=10"


def test_word_problem_negative_controls(env):
    """负控：非数字词 / 非二元 / 未 bootstrap（冷启动）→ None（守反统计契约·不凑配）。"""
    b, sid, es, ci = env
    bootstrap_number_grounding(ci, es, b, _ZH_FACTS, space_id=sid)

    def wp(toks):
        return word_problem_value(toks, concept_index=ci, edge_store=es, backend=b,
                                  space_id=sid, lang=LANG_ZH)
    assert wp(["苹果", "加", "二"]) is None, "非数字词→None（未接地）"
    assert wp(["三", "加"]) is None, "非二元（2 token）→None"
    assert wp(["三", "加", "二", "减", "一"]) is None, "非二元（5 token·多步 defer）→None"


def test_word_problem_cold_start_none(env):
    """冷启动（未 bootstrap·无图边）→ word_problem_value None（数字词未接地·关联未在图中）。"""
    b, sid, es, ci = env
    val = word_problem_value(["三", "加", "二"], concept_index=ci, edge_store=es,
                             backend=b, space_id=sid, lang=LANG_ZH)
    assert val is None, "冷启动（未 bootstrap）→ None（resolve_number_word 走图无 PURE_ALIAS 边）"


def test_word_problem_en(env):
    """EN：one plus two = 3（EN facts + EN lang·镜像 ZH·跨语言接地）·vm_proof 验。"""
    b, sid, es, ci = env
    en_facts = [("one", 2, 1), ("two", 2, 2), ("three", 2, 3)]
    bootstrap_number_grounding(ci, es, b, en_facts, space_id=sid)
    val = word_problem_value(["one", "plus", "two"], concept_index=ci, edge_store=es,
                             backend=b, space_id=sid, lang=LANG_EN)
    assert rational.eq(val, make(3, 1)), "one plus two = 3（EN·arith_op_of first source plus→ADD）"
