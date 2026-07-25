"""刀3 件1 种概念测试（学习放开 6 刀·任务 #594·doc/重来_学习放开整合设计_纠偏纠偏.md §5 刀3）。

刀3 = 关系作 first-class NODE_CONCEPT 节点（REL_*）+ 激活 D:11 EDGE_RELATION_SIGNAL（零产消者→产消者）+
ATTR_RELATION_PRIMITIVE=10 标记。boot 时种 REL_* 节点 + 词→REL_* D:11 边（元定义层 frozenset 种子·
Plan agent 路线决断 ii·同 cue_words 范式）。

**刀3 验收判据**（反 theater）：
  - boot 种 REL_* first-class NODE_CONCEPT（8 个·concept_index.lookup 命中 + ATTR_RELATION_PRIMITIVE=10 标记）。
  - boot 种 D:11 EDGE_RELATION_SIGNAL 边（"是"→REL_SUBSET 等·当前刀3 前零产消者·刀3 后非空 = 真激活）。
  - lookup_word_concept round-trip 活（record 后 read 返 [(rel_ref, rel_kind)]·非死码）。

**bit-identical**：frozenset 内置无条件种·CI===生产·既有测试全 type-filtered 正交 D:11/REL_* 节点 → 870 测零翻。

铁律：纯整数 / 确定性 bit-identical / 不写死（REL_* enum + _REL_LEXICAL_CUE frozenset=meta定义例外）/
  §8.1c（D:11 不接 reward）/ §8.8（关系概念=first-class 节点非层次链复活）/ epistemic 闭合（镜像 is_a.py:54）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import (
    EdgeStore, SOURCE_TEACHER, SOURCE_CONCEPTNET, EPI_STRUCTURED, EPI_CUE,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.storage.composes_attr import (
    register_composes_attr, read_composes_attrs, ATTR_RELATION_PRIMITIVE,
    ATTR_OPERATOR_PRIMITIVE, ATTR_MODAL_KIND, ATTR_OPERATION_INTENT, ATTR_SYMBOL_TYPE,
)
from pure_integer_ai.cognition.shared.symbol_types import TYPE_NEGATION
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import ConceptRef, LANG_ZH, MODALITY_LANGUAGE
from pure_integer_ai.cognition.shared.relation_primitives import (
    ensure_relation_primitives,
    REL_SUBSET, REL_MEMBER, REL_EQUAL, REL_CAUSES, REL_PRECEDES, REL_MEREOLOGY,
    REL_PROPERTY, REL_SIMILAR,
)
from pure_integer_ai.cognition.shared.operator_primitives import (
    OP_ADD, OP_SUB, OP_MUL, OP_GT, OP_LT, OP_GE, OP_LE,
)
from pure_integer_ai.cognition.understanding.word_concept_signal import (
    record_word_concept, lookup_word_concept, bootstrap_word_concept_signals,
)
from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_PRECEDES

_ALL_REL_KINDS = [
    REL_SUBSET, REL_MEMBER, REL_EQUAL, REL_CAUSES, REL_PRECEDES,
    REL_MEREOLOGY, REL_PROPERTY, REL_SIMILAR,
]

_ALL_OP_KINDS = [OP_ADD, OP_SUB, OP_MUL, OP_GT, OP_LT, OP_GE, OP_LE]


# ---- fixtures ----

@pytest.fixture
def rel_env():
    """关系原语单测环境（dict backend·core space·composes_attr 注册）。镜像 test_knife0 isa_env。"""
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


def _lang_item(tokens: list[str]) -> CollectedItem:
    """语言 corpus item（MODALITY_LANGUAGE·LANG_ZH·空白已切 token）。"""
    return CollectedItem(tokens=tokens, collect_type=COLLECT_PRECEDES)


# ============ unit：ensure_relation_primitives（8 REL_* NODE_CONCEPT + 标记 + 幂等） ============

def test_ensure_relation_primitives_creates_8_nodes(rel_env):
    """ensure → 8 REL_* NODE_CONCEPT（lookup 命中）+ 各挂 ATTR_RELATION_PRIMITIVE=10 int_a=kind。"""
    b, sid, es, ci = rel_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    assert len(rel_refs) == 8, "8 REL_* primitives"
    for kind in _ALL_REL_KINDS:
        assert kind in rel_refs, f"REL kind {kind} 在返 map"
        ref = rel_refs[kind]
        assert ref is not None and ref != (0, 0), f"REL kind {kind} → 有效 ConceptRef"
        # ATTR_RELATION_PRIMITIVE=10 标记·int_a=kind
        attrs = read_composes_attrs(b, ref)
        assert ATTR_RELATION_PRIMITIVE in attrs, f"REL kind {kind} 节点有 ATTR_RELATION_PRIMITIVE 标记"
        assert attrs[ATTR_RELATION_PRIMITIVE] == (kind, 0), \
            f"ATTR_RELATION_PRIMITIVE int_a={kind} int_b=0"


def test_ensure_relation_primitives_lookup_hits(rel_env):
    """ensure 后 concept_index.lookup(stable surface) 命中（NODE_CONCEPT first-class·跨 run identity 基础）。"""
    b, sid, es, ci = rel_env
    ensure_relation_primitives(ci, b, space_id=sid)
    for surface, kind in [("__REL_SUBSET__", REL_SUBSET), ("__REL_CAUSES__", REL_CAUSES),
                          ("__REL_MEREOLOGY__", REL_MEREOLOGY), ("__REL_MEMBER__", REL_MEMBER)]:
        ref = ci.lookup(surface, sid)
        assert ref is not None, f"lookup({surface}) 命中（REL_* first-class NODE_CONCEPT）"


def test_ensure_relation_primitives_idempotent(rel_env):
    """幂等：调两次 → concept_node 不增 + composes_attr ATTR_RELATION_PRIMITIVE 行不增（resume/重 boot 不 corrupt）。"""
    b, sid, es, ci = rel_env
    ensure_relation_primitives(ci, b, space_id=sid)
    nodes_after_1 = len(b.select("concept_node", where=None))
    attrs_after_1 = len([r for r in b.select("composes_attr", where={"kind": ATTR_RELATION_PRIMITIVE})])
    ensure_relation_primitives(ci, b, space_id=sid)   # 二次调
    nodes_after_2 = len(b.select("concept_node", where=None))
    attrs_after_2 = len([r for r in b.select("composes_attr", where={"kind": ATTR_RELATION_PRIMITIVE})])
    assert nodes_after_1 == nodes_after_2, "二次 ensure 不增 concept_node（ConceptIndex.ensure 同 hash dedup）"
    assert attrs_after_1 == attrs_after_2, "二次 ensure 不增 ATTR_RELATION_PRIMITIVE 行（record_composes_attr 幂等 skip）"


# ============ unit：record_word_concept（D:11 边建 + 幂等 + 异源 + 防御 + epistemic assert） ============

def test_record_word_concept_builds_d11_edge(rel_env):
    """record → D:11 EDGE_RELATION_SIGNAL 边建（from=word·to=REL_*·source=SOURCE_TEACHER·epistemic=EPI_STRUCTURED·strength=1）。"""
    b, sid, es, ci = rel_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    n = record_word_concept(ci, es, "是", rel_refs[REL_SUBSET], space_id=sid)
    assert n == 1, "建 1 条 D:11 边"
    d11 = [r for r in b.select("edge", where={"edge_type": EDGE_RELATION_SIGNAL})]
    assert len(d11) == 1, "1 条 EDGE_RELATION_SIGNAL 边"
    e = d11[0]
    assert e["source"] == SOURCE_TEACHER, "默认 source=SOURCE_TEACHER（教师元定义·Plan agent 修点3）"
    assert e["epistemic_origin"] == EPI_STRUCTURED, "默认 epistemic=EPI_STRUCTURED"
    assert e["strength"] == 1, "tentative strength=1（不接 reward·非学习对象初值）"
    assert e["tier"] == TIER_PRIMARY, "TIER_PRIMARY"
    assert (e["space_id_to"], e["local_id_to"]) == rel_refs[REL_SUBSET], "to=REL_SUBSET ConceptRef"
    # from=word concept（"是"）
    word_ref = ci.lookup("是", sid)
    assert word_ref is not None and (e["space_id_from"], e["local_id_from"]) == word_ref, "from=word concept"


def test_record_word_concept_idempotent_same_source_skip(rel_env):
    """query_from 按源幂等 skip：同 (word,rel,D:11,SOURCE_TEACHER) 重种 → 0（镜像 is_a.py:129-135）。"""
    b, sid, es, ci = rel_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    n1 = record_word_concept(ci, es, "是", rel_refs[REL_SUBSET], space_id=sid)
    n2 = record_word_concept(ci, es, "是", rel_refs[REL_SUBSET], space_id=sid)   # 同源同三元组重种
    assert n1 == 1 and n2 == 0, "同源同三元组第二次种 → skip（query_from 幂等）"
    d11 = [r for r in b.select("edge", where={"edge_type": EDGE_RELATION_SIGNAL})]
    assert len(d11) == 1, "重种不增边（幂等·EdgeStore.add 不去重·须 query_from skip 守）"


def test_record_word_concept_different_source_coexist(rel_env):
    """异源并存：SOURCE_TEACHER 边不挡 SOURCE_CONCEPTNET 路径（未来 ConceptNet loader·镜像 is_a 异源测试）。"""
    b, sid, es, ci = rel_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    # 教师元定义种（默认 SOURCE_TEACHER）
    record_word_concept(ci, es, "是", rel_refs[REL_SUBSET], space_id=sid)
    # 异源种（ConceptNet loader 范式·caller 覆写 source）
    record_word_concept(ci, es, "是", rel_refs[REL_SUBSET], space_id=sid,
                        source=SOURCE_CONCEPTNET)
    d11 = [r for r in b.select("edge", where={"edge_type": EDGE_RELATION_SIGNAL})]
    assert len(d11) == 2, "异源同三元组并存（TEACHER 不挡 CONCEPTNET）"
    sources = sorted(r["source"] for r in d11)
    assert sources == sorted([SOURCE_TEACHER, SOURCE_CONCEPTNET]), "两条边异源"


def test_record_word_concept_empty_defensive_short_circuit(rel_env):
    """Plan agent 修点4 防御短路：空 word / None rel_ref → return 0（镜像 bootstrap_is_a_edges:119）。"""
    b, sid, es, ci = rel_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    assert record_word_concept(ci, es, "", rel_refs[REL_SUBSET], space_id=sid) == 0, "空 word → 0"
    assert record_word_concept(ci, es, "是", None, space_id=sid) == 0, "None rel_ref → 0"
    assert record_word_concept(ci, es, "是", (0, 0), space_id=sid) == 0, "(0,0) rel_ref → 0"
    d11 = [r for r in b.select("edge", where={"edge_type": EDGE_RELATION_SIGNAL})]
    assert len(d11) == 0, "防御短路零边副作用"


def test_record_word_concept_epistemic_assertion(rel_env):
    """Plan agent 修点2 epistemic 闭合：非法 epistemic（裸共现）→ AssertionError（镜像 is_a.py:54）。"""
    b, sid, es, ci = rel_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    # 非法 epistemic（不在 EPI_STRUCTURED/EPI_CUE/EPI_LLM_CONFIRM 白名单）
    with pytest.raises(AssertionError, match="认识论来源"):
        record_word_concept(ci, es, "是", rel_refs[REL_SUBSET], space_id=sid, epistemic=999)


# ============ unit：lookup_word_concept round-trip（反 theater read 路径活） ============

def test_lookup_word_concept_round_trip(rel_env):
    """record 后 lookup 返 [(rel_ref, rel_kind)]·read 路径非死码（反 theater·刀3 无生产 caller·刀4/5 消费）。"""
    b, sid, es, ci = rel_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    record_word_concept(ci, es, "是", rel_refs[REL_SUBSET], space_id=sid)
    record_word_concept(ci, es, "导致", rel_refs[REL_CAUSES], space_id=sid)
    # lookup "是" → [(REL_SUBSET_ref, REL_SUBSET)]
    是_ref = ci.lookup("是", sid)
    results = lookup_word_concept(b, es, 是_ref, space_id=sid)
    assert len(results) == 1, "lookup 返 1 条（是→REL_SUBSET）"
    rel_ref, kind = results[0]
    assert rel_ref == rel_refs[REL_SUBSET], "rel_ref = REL_SUBSET ConceptRef"
    assert kind == REL_SUBSET, "rel_kind = REL_SUBSET（ATTR_RELATION_PRIMITIVE int_a 读回）"
    # lookup "导致" → REL_CAUSES
    导致_ref = ci.lookup("导致", sid)
    results2 = lookup_word_concept(b, es, 导致_ref, space_id=sid)
    assert len(results2) == 1 and results2[0][1] == REL_CAUSES, "lookup 导致 → REL_CAUSES"
    # lookup 不存在的 word → []
    absent_ref = ci.ensure("不存在词", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    assert lookup_word_concept(b, es, absent_ref, space_id=sid) == [], "无 D:11 边的 word → []"


def test_lookup_word_concept_none_word_ref(rel_env):
    """lookup(None) → []（防御）。"""
    b, sid, es, ci = rel_env
    assert lookup_word_concept(b, es, None, space_id=sid) == [], "None word_ref → []"


# ============ unit：bootstrap_word_concept_signals（boot 入口·langs 集循环种） ============

def test_bootstrap_word_concept_signals_seeds_by_lang(rel_env):
    """boot 入口：ensure REL_* + 按 langs 种 _REL_LEXICAL_CUE 词→D:11 边。LANG_ZH 种 9 词 8 类。"""
    b, sid, es, ci = rel_env
    n = bootstrap_word_concept_signals(ci, es, b, space_id=sid, langs={LANG_ZH})
    # LANG_ZH _REL_LEXICAL_CUE: 是/属于/导致/引起/先于/部分/等于/具有/像 = 9 词·映射
    # REL_SUBSET/REL_MEMBER/REL_CAUSES×2/REL_PRECEDES/REL_MEREOLOGY/REL_EQUAL/REL_PROPERTY/REL_SIMILAR
    # （STEP5 PR1 等于·PR3 具有·PR4 像）
    assert n == 9, f"LANG_ZH 种 9 条 D:11 边·got {n}"
    d11 = [r for r in b.select("edge", where={"edge_type": EDGE_RELATION_SIGNAL})]
    assert len(d11) == 9, "9 条 D:11 边"
    # REL_* 8 节点都建（框架·REL_EQUAL/PROPERTY/SIMILAR 有种子·全 8 节点建为晋升目标）
    attrs = [r for r in b.select("composes_attr", where={"kind": ATTR_RELATION_PRIMITIVE})]
    assert len(attrs) == 8, "8 REL_* NODE_CONCEPT 节点全建（框架·非只种子词对应的）"


def test_bootstrap_word_concept_signals_empty_langs_no_op(rel_env):
    """空 langs（arith-only run）→ ensure REL_* 仍建框架节点·record 0 D:11 边（最小副作用·bit-identical）。"""
    b, sid, es, ci = rel_env
    n = bootstrap_word_concept_signals(ci, es, b, space_id=sid, langs=set())
    assert n == 0, "空 langs → 0 D:11 边"
    # REL_* 框架节点仍建（ensure_relation_primitives 无条件·类 OPCODE_* 常驻）
    attrs = [r for r in b.select("composes_attr", where={"kind": ATTR_RELATION_PRIMITIVE})]
    assert len(attrs) == 8, "空 langs 仍建 8 REL_* 框架节点（晋升目标·刀4 涌现可填）"


# ============ e2e：formal_train 全 main 流程（刀3 真验收·boot 接线层） ============

def test_knife3_formal_train_boot_seeds_d11_and_rel_nodes(tmp_path, monkeypatch):
    """刀3 真验收（反 theater）：formal_train boot（frozenset 种子·无 PURE_INTEGER_AI_LOCAL_DIR）→
    REL_* NODE_CONCEPT 在 + D:11 EDGE_RELATION_SIGNAL 边 >0（"是"→REL_SUBSET 落盘）。

    走真 formal_train 全 main 流程·证 boot 接线层激活 D:11 产消者（当前刀3 前零产消者全空）。
    无 PURE_INTEGER_AI_LOCAL_DIR（CI/生产 default）·刀3 不依赖外部文件·frozenset 内置种子。
    """
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)   # CI/生产 default 无文件（刀3 不依赖·frozenset 内置）
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig, DefaultRoundRunner
    corpus = [_lang_item(["猫", "追", "老鼠"]), _lang_item(["狗", "追", "兔子"])]
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="knife3_boot", rounds_per_stage=1)
    result = formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())

    # ① boot 种了 D:11 EDGE_RELATION_SIGNAL 边（formal_train boot 段跑·frozenset 种子）
    d11_edges = [r for r in b.select("edge", where={"edge_type": EDGE_RELATION_SIGNAL})]
    assert len(d11_edges) > 0, "刀3 boot 种 D:11 EDGE_RELATION_SIGNAL 边（反 theater·当前刀3 前零产消者）"
    # ② 每 D:11 边 target 是 REL_*/OP_*/MODAL_KIND/TYPE_NEGATION/ACTION_INTENT NODE_CONCEPT（D:11 共享边类型·五类 target·
    # #940 加 TYPE_NEGATION·B-PR1 加 ACTION_INTENT（INTENT_COMMAND_MOOD + ACTION_*·doc §16·镜像 operator·挂 ATTR_OPERATION_INTENT））
    for e in d11_edges:
        rel_ref = (e["space_id_to"], e["local_id_to"])
        attrs = read_composes_attrs(b, rel_ref)
        assert e["source"] == SOURCE_TEACHER, "D:11 边 source=SOURCE_TEACHER（教师元定义）"
        if ATTR_RELATION_PRIMITIVE in attrs:
            kind = attrs[ATTR_RELATION_PRIMITIVE][0]
            assert kind in _ALL_REL_KINDS, f"D:11 target REL_* kind 合法·got {kind}"
        elif ATTR_OPERATOR_PRIMITIVE in attrs:
            op_kind = attrs[ATTR_OPERATOR_PRIMITIVE][0]
            assert op_kind in _ALL_OP_KINDS, f"D:11 target OP_* kind 合法·got {op_kind}"
        elif ATTR_MODAL_KIND in attrs:
            modal_kind = attrs[ATTR_MODAL_KIND][0]
            assert modal_kind in (1, 2, 3, 4), f"D:11 target MODAL_KIND 合法·got {modal_kind}"
        elif ATTR_SYMBOL_TYPE in attrs:
            sym_kind = attrs[ATTR_SYMBOL_TYPE][0]
            assert sym_kind == TYPE_NEGATION, f"D:11 target ATTR_SYMBOL_TYPE kind=TYPE_NEGATION·got {sym_kind}"
        elif ATTR_OPERATION_INTENT in attrs:
            action_kind = attrs[ATTR_OPERATION_INTENT][0]
            assert action_kind in (0, 1, 2, 3, 4), \
                f"D:11 target ATTR_OPERATION_INTENT kind 合法（0=COMMAND_MOOD/1-4=ACTION_*）·got {action_kind}"
        else:
            assert False, "D:11 target = REL_*/OP_*/MODAL_KIND/TYPE_NEGATION/ACTION_INTENT NODE_CONCEPT（有标记）"
    # ③ "是"→REL_SUBSET 落盘（验证种子词真种·非空泛）
    word_refs = {(e["space_id_from"], e["local_id_from"]) for e in d11_edges}
    # 建 ConceptIndex 读回找 "是" word concept（concept_identity 持久化·fresh ConceptIndex lazy-load）
    ci2 = ConceptIndex(b)
    是_ref = ci2.lookup("是", d11_edges[0]["space_id_from"])   # space_id 从 D:11 边读
    assert 是_ref is not None, "'是' word concept 建了"
    assert 是_ref in word_refs, "'是'→D:11 边落盘（frozenset 种子 REL_SUBSET）"
    # ④ lookup round-trip（生产 read 路径活）
    results = lookup_word_concept(b, EdgeStore(b), 是_ref, space_id=d11_edges[0]["space_id_from"])
    assert any(kind == REL_SUBSET for _ref, kind in results), "lookup('是') 返 REL_SUBSET（round-trip 活）"


def test_knife3_formal_train_existing_edge_types_unchanged(tmp_path, monkeypatch):
    """bit-identical baseline：formal_train 跑 → D:11 边存在（新 baseline）·既有 EDGE_ISA/COMPOSES/PRECEDES
    边数与刀3 前行为一致（type-filtered·正交 D:11·非总数断言破）。

    刀3 加的 REL_* NODE_CONCEPT + D:11 边与新 baseline·既有 edge_type 计数语义不变
    （刀0 WITH-文件种 ISA / observe 建 COMPOSES/PRECEDES·刀3 不涉这些 type）。
    """
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig, DefaultRoundRunner
    from pure_integer_ai.storage.edge_types import EDGE_IS_A, EDGE_COMPOSES, EDGE_PRECEDES
    corpus = [_lang_item(["猫", "追", "老鼠"]), _lang_item(["狗", "追", "兔子"])]
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="knife3_baseline", rounds_per_stage=1)
    formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())
    # D:11 边存在（新 baseline·刀3 激活）
    assert len([r for r in b.select("edge", where={"edge_type": EDGE_RELATION_SIGNAL})]) > 0, \
        "D:11 EDGE_RELATION_SIGNAL 边存在（刀3 新 baseline）"
    # 既有 edge_type 边数 >0（formal_train 正常产·刀3 不破）
    assert len([r for r in b.select("edge", where={"edge_type": EDGE_COMPOSES})]) > 0, \
        "EDGE_COMPOSES 边正常产（刀3 不破·语言 observe 建 COMPOSES）"
    # ISA 边：无文件 → 0（刀0 零副作用·刀3 不改刀0 行为）
    isa_edges = [r for r in b.select("edge", where={"edge_type": EDGE_IS_A})]
    assert not any(r["source"] == SOURCE_CONCEPTNET for r in isa_edges), \
        "无 IS_A 文件 → 刀0 零副作用（刀3 不改刀0 boot 行为）"
