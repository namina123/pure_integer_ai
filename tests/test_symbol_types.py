"""STEP3 symbol_types.py 命名层测试（#890·doc/重来_符号域修正分析_2026-07-10.md §五 STEP 3）。

STEP3 = 符号空间 type_ref 先天分类 canonical 命名层（D6 对齐）。TYPE_* enum 元定义固化 + ensure_symbol_types
镜像 ensure_relation_primitives（建 first-class NODE_CONCEPT + ATTR_SYMBOL_TYPE=17 标记）+ shadow 空挂载 defer。

**STEP3 验收判据**：
  - TYPE_PROPOSITION 别名 ATTR_PROPOSITION=11（既有·命题节点挂此·非本模块 ensure 建）。
  - ensure_symbol_types 建 TYPE_NEGATION first-class NODE_CONCEPT（lookup 命中 + ATTR_SYMBOL_TYPE=17 标记 int_a=TYPE_NEGATION）。
  - 幂等（调 2 次同 ref·composes_attr 不重复）。
  - _TYPE_SURFACE 稳定（content_hash dedup·bit-identical）。
  - 不建 ATTR_NEGATION（doc:193 ¬ 走命题 surface polarity）。

**bit-identical**：enum + 函数 ship 不调用 = AST 级零变·shadow 空挂载 defer。

铁律：纯整数 / 确定性 bit-identical / 不写死（TYPE_* enum=meta定义例外·非语义规则）/ 单向依赖（L0 依赖 storage 向下）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import (
    register_composes_attr, read_composes_attrs, ATTR_PROPOSITION, ATTR_SYMBOL_TYPE,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.symbol_types import (
    ensure_symbol_types,
    TYPE_PROPOSITION, TYPE_NEGATION, TYPE_COPULA, TYPE_CMP, TYPE_CAUSES, TYPE_ATTR_MARKER,
    _TYPE_SURFACE,
)


# ---- fixtures ----

@pytest.fixture
def sym_env():
    """符号类型单测环境（dict backend·core space·composes_attr 注册）。镜像 test_g1_proposition prop_env。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)   # ATTR_SYMBOL_TYPE=17 标记表
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    yield b, sid, es, ci
    b.close()


# ============ unit：TYPE_* enum（canonical 命名层·D6 对齐） ============

def test_type_proposition_alias_attr_proposition():
    """TYPE_PROPOSITION 别名 ATTR_PROPOSITION=11（既有·命题节点挂此·G1 reification·非本模块 ensure 建）。"""
    assert TYPE_PROPOSITION == 11, "TYPE_PROPOSITION = 11"
    assert ATTR_PROPOSITION == 11, "ATTR_PROPOSITION = 11"
    assert TYPE_PROPOSITION == ATTR_PROPOSITION, "TYPE_PROPOSITION 别名 ATTR_PROPOSITION（同值 11）"


def test_type_negation_value():
    """TYPE_NEGATION=12（¬ 先天分类·符号空间 type_ref·非抽象空间 abstract_mark）。"""
    assert TYPE_NEGATION == 12, "TYPE_NEGATION = 12"


def test_type_registered_not_active_distinct():
    """登记不激活 TYPE_* 各异（defer 范式·同 EDGE_CALLS·消费者出现才 ensure）。"""
    values = [TYPE_PROPOSITION, TYPE_NEGATION, TYPE_COPULA, TYPE_CMP, TYPE_CAUSES, TYPE_ATTR_MARKER]
    assert values == [11, 12, 13, 14, 15, 16], "TYPE_* 11-16 序列"
    assert len(set(values)) == 6, "6 TYPE_* 各异"


def test_type_surface_stable():
    """_TYPE_SURFACE 稳定 surface（content_hash dedup·跨 run identity·bit-identical）。"""
    assert _TYPE_SURFACE[TYPE_NEGATION] == "__TYPE_NEGATION__", "TYPE_NEGATION 稳定 surface"


def test_type_surface_only_negation():
    """只建消费者需要的 surface（TYPE_NEGATION）·TYPE_PROPOSITION/登记不激活的不建。"""
    assert set(_TYPE_SURFACE.keys()) == {TYPE_NEGATION}, \
        "只 TYPE_NEGATION 建 surface（TYPE_PROPOSITION 走 __prop_*·登记不激活的不建）"


def test_type_negation_not_attr_negation():
    """doc:193 ¬ 走命题 surface polarity·不建 ATTR_NEGATION marker。

    TYPE_NEGATION(12) 是符号空间 type_ref 先天分类·非 ATTR_NEGATION。
    ATTR_NEGATION 在 composes_attr 不存在（doc:193 明示不建·¬ 走 surface polarity _1_0 后缀）。
    """
    import pure_integer_ai.storage.composes_attr as ca
    assert not hasattr(ca, "ATTR_NEGATION"), \
        "doc:193 ¬ 走命题 surface polarity·不建 ATTR_NEGATION marker"
    assert ATTR_SYMBOL_TYPE == 17, "ATTR_SYMBOL_TYPE=17（通用符号类型标记·非 ATTR_NEGATION 专用）"


def test_attr_symbol_type_value():
    """ATTR_SYMBOL_TYPE=17（通用符号类型标记·int_a=TYPE_*·非结构 kind）。"""
    assert ATTR_SYMBOL_TYPE == 17, "ATTR_SYMBOL_TYPE = 17"


# ============ unit：ensure_symbol_types（TYPE_NEGATION NODE_CONCEPT + 标记 + 幂等） ============

def test_ensure_symbol_types_builds_negation_node(sym_env):
    """ensure -> {TYPE_NEGATION: ref}·ref 有效·composes_attr 有 ATTR_SYMBOL_TYPE=17 行（int_a=TYPE_NEGATION）。"""
    b, sid, es, ci = sym_env
    type_refs = ensure_symbol_types(ci, b, space_id=sid)
    assert TYPE_NEGATION in type_refs, "TYPE_NEGATION 在返 map"
    assert len(type_refs) == 1, "只建 TYPE_NEGATION（消费者需要的）"
    ref = type_refs[TYPE_NEGATION]
    assert ref is not None and ref != (0, 0), "TYPE_NEGATION -> 有效 ConceptRef"
    # ATTR_SYMBOL_TYPE=17 标记·int_a=TYPE_NEGATION
    attrs = read_composes_attrs(b, ref)
    assert ATTR_SYMBOL_TYPE in attrs, "TYPE_NEGATION 节点有 ATTR_SYMBOL_TYPE=17 标记"
    assert attrs[ATTR_SYMBOL_TYPE] == (TYPE_NEGATION, 0), \
        "ATTR_SYMBOL_TYPE int_a=TYPE_NEGATION int_b=0"


def test_ensure_symbol_types_lookup_hits(sym_env):
    """ensure 后 concept_index.lookup(stable surface) 命中（NODE_CONCEPT first-class·跨 run identity 基础）。"""
    b, sid, es, ci = sym_env
    ensure_symbol_types(ci, b, space_id=sid)
    ref = ci.lookup("__TYPE_NEGATION__", sid)
    assert ref is not None, "lookup(__TYPE_NEGATION__) 命中（TYPE_NEGATION first-class NODE_CONCEPT）"


def test_ensure_symbol_types_idempotent(sym_env):
    """幂等：调两次 -> concept_node 不增 + composes_attr ATTR_SYMBOL_TYPE 行不增（resume/重 boot 不 corrupt）。"""
    b, sid, es, ci = sym_env
    refs1 = ensure_symbol_types(ci, b, space_id=sid)
    nodes_after_1 = len(b.select("concept_node", where=None))
    attrs_after_1 = len([r for r in b.select("composes_attr", where={"kind": ATTR_SYMBOL_TYPE})])
    refs2 = ensure_symbol_types(ci, b, space_id=sid)   # 二次调
    nodes_after_2 = len(b.select("concept_node", where=None))
    attrs_after_2 = len([r for r in b.select("composes_attr", where={"kind": ATTR_SYMBOL_TYPE})])
    assert nodes_after_1 == nodes_after_2, "二次 ensure 不增 concept_node（ConceptIndex.ensure 同 hash dedup）"
    assert attrs_after_1 == attrs_after_2, "二次 ensure 不增 ATTR_SYMBOL_TYPE 行（record_composes_attr 幂等 skip）"
    assert refs1[TYPE_NEGATION] == refs2[TYPE_NEGATION], "二次 ensure 同 ref（幂等返既有）"


def test_ensure_symbol_types_not_proposition_node(sym_env):
    """ensure 不建 TYPE_PROPOSITION boot 种（命题节点 observe build_property_edges 建 __prop_*·挂 ATTR_PROPOSITION=11）。"""
    b, sid, es, ci = sym_env
    ensure_symbol_types(ci, b, space_id=sid)
    # __TYPE_PROPOSITION__ 不存在（命题节点走 __prop_{subj}_{attr} surface·非 boot 种）
    ref = ci.lookup("__TYPE_PROPOSITION__", sid)
    assert ref is None, "不建 __TYPE_PROPOSITION__ boot 种（命题节点 observe 建 __prop_*）"
    # ATTR_PROPOSITION=11 标记零行（本函数不建命题节点）
    prop_attrs = [r for r in b.select("composes_attr", where={"kind": ATTR_PROPOSITION})]
    assert len(prop_attrs) == 0, "ensure_symbol_types 不建 ATTR_PROPOSITION 节点（命题节点走 observe）"


def test_ensure_symbol_types_space_id_assert():
    """assert_int 守 space_id（纯整数铁律·非 int -> AssertionError）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    ci = ConceptIndex(b)
    with pytest.raises(AssertionError):
        ensure_symbol_types(ci, b, space_id=1.5)   # 浮点 -> assert_int 拒
    b.close()
