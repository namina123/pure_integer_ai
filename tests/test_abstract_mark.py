"""§七实现层 modality_subspace 片1 测试：abstract_mark 扩展表（§7.4 L212 + §7.7.1 路径 B）。

覆盖（doc/重来_大工程实施顺序_2026-07-06.md §2.1）：
  - schema/discipline/core=False（DISC_NONE·status flip 非单调）
  - register 幂等
  - set_mark insert / 幂等（同 status 不写）/ status flip（PENDING→PROMOTED→ARCHIVED）
  - get_marks（全 / 按 kind / 按 status）/ get_mark 单值便捷读
  - query_nodes_by_mark（维度正向查）
  - query_intersection（多维 set 交集·非 Venn·单维退化·空 marks）
  - bare fixture 向后兼容（表未注册→skip/[]）
  - bit-identical（两跑一致·sorted 输出）
  - dump_tables 含 abstract_mark（片1 接线·跨 run 还原）

铁律：纯整数 / DISC_NONE（status flip·core=False）/ core=False（不污染节点列·守铁律 6）/
确定性（set_mark query-then-upsert 幂等 + query_intersection sorted·bit-identical）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap, discipline as disc
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.node_store import NodeStore, NODE_CONCEPT, TIER_SHADOW, TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.edge_types import EDGE_REFERS_TO
from pure_integer_ai.storage.abstract_mark import (
    ABSTRACT_MARK_TABLE, register_abstract_mark,
    MARK_MODALITY, MARK_LANG, MARK_DOMAIN,
    MARK_PENDING, MARK_PROMOTED, MARK_ARCHIVED,
    set_mark, get_marks, get_mark,
    query_nodes_by_mark, query_intersection,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.types import (
    RoleSlot, PathResult, PathData, TERMINAL_REACHED_SINK, LINEAGE_CONCEPT_FILL,
    LANG_ZH, LANG_EN, LANG_NONE,
)
from pure_integer_ai.cognition.understanding.refers_to import normalize_to_concept
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.slot_dispatch import dispatch_slot
from pure_integer_ai.training.cursor import dump_run, load_run
from pure_integer_ai.experiments.formal_train import FormalTrainConfig


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def backend(request):
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    register_abstract_mark(b)
    yield b
    b.close()


# ============ schema / register ============

def test_schema_discipline_core_false(backend):
    """schema 5 列 + DISC_NONE（status flip 非单调）+ core=False（不污染节点列·守铁律 6）。"""
    meta = backend._tables[ABSTRACT_MARK_TABLE]
    assert meta["core"] is False
    assert meta["discipline"] == disc.DISC_NONE
    assert meta["columns"] == ["space_id", "local_id", "mark_kind",
                               "mark_value", "status"]


def test_register_idempotent():
    """register 幂等（重复调不报错·同 experience_count 范式）。"""
    b = DictBackend()
    bootstrap(b)
    register_abstract_mark(b)
    register_abstract_mark(b)   # 二次·幂等
    assert ABSTRACT_MARK_TABLE in b._tables
    b.close()


# ============ set_mark ============

def test_set_mark_insert_new(backend):
    """set_mark 新行 insert（默认 status=PROMOTED）。"""
    set_mark(backend, ref=(1, 10), mark_kind=MARK_MODALITY, mark_value=3)
    rows = backend.select(ABSTRACT_MARK_TABLE,
                          where={"space_id": 1, "local_id": 10})
    assert len(rows) == 1
    assert rows[0]["mark_kind"] == MARK_MODALITY
    assert rows[0]["mark_value"] == 3
    assert rows[0]["status"] == MARK_PROMOTED


def test_set_mark_idempotent_same_status(backend):
    """set_mark 同 (kind,value,status) 幂等（零写·bit-identical）。"""
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1)
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1)   # 同 status
    rows = backend.select(ABSTRACT_MARK_TABLE,
                          where={"space_id": 1, "local_id": 10})
    assert len(rows) == 1   # 不增行


def test_set_mark_status_flip(backend):
    """set_mark status flip（PENDING→PROMOTED→ARCHIVED·DISC_NONE 允许·非单调）。"""
    set_mark(backend, ref=(1, 10), mark_kind=MARK_DOMAIN, mark_value=2,
             status=MARK_PENDING)
    assert get_mark(backend, ref=(1, 10), mark_kind=MARK_DOMAIN,
                    status=MARK_PENDING) == 2
    set_mark(backend, ref=(1, 10), mark_kind=MARK_DOMAIN, mark_value=2,
             status=MARK_PROMOTED)   # flip → PROMOTED
    rows = backend.select(ABSTRACT_MARK_TABLE,
                          where={"space_id": 1, "local_id": 10,
                                 "mark_kind": MARK_DOMAIN, "mark_value": 2})
    assert len(rows) == 1   # 仍单行（update 非 insert）
    assert rows[0]["status"] == MARK_PROMOTED
    set_mark(backend, ref=(1, 10), mark_kind=MARK_DOMAIN, mark_value=2,
             status=MARK_ARCHIVED)   # flip → ARCHIVED
    rows = backend.select(ABSTRACT_MARK_TABLE,
                          where={"space_id": 1, "local_id": 10,
                                 "mark_kind": MARK_DOMAIN, "mark_value": 2})
    assert rows[0]["status"] == MARK_ARCHIVED


def test_set_mark_multi_kind_same_node(backend):
    """同节点多 kind 共存（modality + lang + domain 三标记独立行）。"""
    set_mark(backend, ref=(1, 10), mark_kind=MARK_MODALITY, mark_value=1)
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=2)
    set_mark(backend, ref=(1, 10), mark_kind=MARK_DOMAIN, mark_value=3)
    marks = get_marks(backend, ref=(1, 10))
    assert len(marks) == 3
    kinds = {m[0] for m in marks}
    assert kinds == {MARK_MODALITY, MARK_LANG, MARK_DOMAIN}


def test_set_mark_multi_value_same_kind(backend):
    """同 kind 多 value 共存（多 lang 词形·PK 前四列区分）。"""
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1)   # ZH
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=2)   # EN
    marks = get_marks(backend, ref=(1, 10), mark_kind=MARK_LANG)
    assert len(marks) == 2
    values = {m[1] for m in marks}
    assert values == {1, 2}


# ============ get_marks / get_mark ============

def test_get_marks_filter_by_kind(backend):
    """get_marks 按 kind 过滤。"""
    set_mark(backend, ref=(1, 10), mark_kind=MARK_MODALITY, mark_value=1)
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1)
    set_mark(backend, ref=(1, 10), mark_kind=MARK_DOMAIN, mark_value=1)
    lang_only = get_marks(backend, ref=(1, 10), mark_kind=MARK_LANG)
    assert len(lang_only) == 1
    assert lang_only[0][0] == MARK_LANG


def test_get_marks_filter_by_status(backend):
    """get_marks 按 status 过滤（PENDING 不进 PROMOTED 默认读）。"""
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1,
             status=MARK_PENDING)
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=2,
             status=MARK_PROMOTED)
    promoted = get_marks(backend, ref=(1, 10), mark_kind=MARK_LANG,
                         status=MARK_PROMOTED)
    assert len(promoted) == 1
    assert promoted[0][1] == 2
    pending = get_marks(backend, ref=(1, 10), mark_kind=MARK_LANG,
                        status=MARK_PENDING)
    assert len(pending) == 1
    assert pending[0][1] == 1


def test_get_mark_single_value(backend):
    """get_mark 单值便捷读（无→None）。"""
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1)
    assert get_mark(backend, ref=(1, 10), mark_kind=MARK_LANG) == 1
    assert get_mark(backend, ref=(1, 10), mark_kind=MARK_DOMAIN) is None


def test_get_marks_empty_node(backend):
    """get_marks 无行→[]（冷启动）。"""
    assert get_marks(backend, ref=(1, 999)) == []
    assert get_mark(backend, ref=(1, 999), mark_kind=MARK_LANG) is None


# ============ query_nodes_by_mark ============

def test_query_nodes_by_mark(backend):
    """按维度正向查节点（跨节点同 mark）。"""
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1)
    set_mark(backend, ref=(1, 20), mark_kind=MARK_LANG, mark_value=1)
    set_mark(backend, ref=(1, 30), mark_kind=MARK_LANG, mark_value=2)   # 异 lang
    nodes = query_nodes_by_mark(backend, mark_kind=MARK_LANG, mark_value=1)
    assert nodes == [(1, 10), (1, 20)]   # 不含 (1,30)
    assert query_nodes_by_mark(backend, mark_kind=MARK_LANG,
                               mark_value=2) == [(1, 30)]


def test_query_nodes_by_mark_status_filter(backend):
    """query_nodes_by_mark 默认 PROMOTED（PENDING/ARCHIVED 不进）。"""
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1,
             status=MARK_PROMOTED)
    set_mark(backend, ref=(1, 20), mark_kind=MARK_LANG, mark_value=1,
             status=MARK_PENDING)
    set_mark(backend, ref=(1, 30), mark_kind=MARK_LANG, mark_value=1,
             status=MARK_ARCHIVED)
    nodes = query_nodes_by_mark(backend, mark_kind=MARK_LANG, mark_value=1)
    assert nodes == [(1, 10)]   # 仅 PROMOTED
    pending = query_nodes_by_mark(backend, mark_kind=MARK_LANG, mark_value=1,
                                  status=MARK_PENDING)
    assert pending == [(1, 20)]


# ============ query_intersection ============

def test_query_intersection_multi_dim(backend):
    """多维相交（同节点 MARK_LANG=1 AND MARK_DOMAIN=1·set 交集·非 Venn）。"""
    # (1,10) 有 LANG=ZH + DOMAIN=TEXT → 命中
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1)
    set_mark(backend, ref=(1, 10), mark_kind=MARK_DOMAIN, mark_value=1)
    # (1,20) 只有 LANG=ZH → 不命中（缺 DOMAIN=TEXT）
    set_mark(backend, ref=(1, 20), mark_kind=MARK_LANG, mark_value=1)
    # (1,30) 只有 DOMAIN=TEXT → 不命中（缺 LANG=ZH）
    set_mark(backend, ref=(1, 30), mark_kind=MARK_DOMAIN, mark_value=1)
    common = query_intersection(backend, marks=[(MARK_LANG, 1), (MARK_DOMAIN, 1)])
    assert common == [(1, 10)]


def test_query_intersection_single_dim_degenerate(backend):
    """单 mark = query_nodes_by_mark 退化。"""
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1)
    set_mark(backend, ref=(1, 20), mark_kind=MARK_LANG, mark_value=1)
    common = query_intersection(backend, marks=[(MARK_LANG, 1)])
    assert common == [(1, 10), (1, 20)]


def test_query_intersection_empty_marks(backend):
    """空 marks → []。"""
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1)
    assert query_intersection(backend, marks=[]) == []


def test_query_intersection_three_dim(backend):
    """三维相交（modality AND lang AND domain）。"""
    set_mark(backend, ref=(1, 10), mark_kind=MARK_MODALITY, mark_value=1)
    set_mark(backend, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1)
    set_mark(backend, ref=(1, 10), mark_kind=MARK_DOMAIN, mark_value=1)
    set_mark(backend, ref=(1, 20), mark_kind=MARK_MODALITY, mark_value=1)
    set_mark(backend, ref=(1, 20), mark_kind=MARK_LANG, mark_value=1)
    # (1,20) 缺 DOMAIN → 不命中
    common = query_intersection(backend, marks=[(MARK_MODALITY, 1),
                                               (MARK_LANG, 1), (MARK_DOMAIN, 1)])
    assert common == [(1, 10)]


# ============ bare fixture 向后兼容 ============

def test_set_mark_table_not_registered_skip():
    """表未注册（bare fixture）→ KeyError 静默 skip（向后兼容·observe 热路径不崩）。"""
    b = DictBackend()
    bootstrap(b)
    # 不 register_abstract_mark
    set_mark(b, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1)   # 不 raise
    b.close()


def test_get_marks_table_not_registered_empty():
    """表未注册→get_marks/get_mark/query_* 返空（向后兼容·同 read_op_confidence 范式）。"""
    b = DictBackend()
    bootstrap(b)
    assert get_marks(b, ref=(1, 10)) == []
    assert get_mark(b, ref=(1, 10), mark_kind=MARK_LANG) is None
    assert query_nodes_by_mark(b, mark_kind=MARK_LANG, mark_value=1) == []
    assert query_intersection(b, marks=[(MARK_LANG, 1)]) == []
    b.close()


# ============ bit-identical ============

def test_bit_identical_two_runs():
    """两跑同插入同输出（set_mark query-then-upsert 幂等 + query_intersection sorted）。"""
    def _run() -> tuple[list, list, list]:
        b = DictBackend()
        bootstrap(b)
        register_abstract_mark(b)
        # 乱序插入（验 sorted 守确定序）
        set_mark(b, ref=(1, 30), mark_kind=MARK_LANG, mark_value=1)
        set_mark(b, ref=(1, 10), mark_kind=MARK_LANG, mark_value=1)
        set_mark(b, ref=(1, 20), mark_kind=MARK_LANG, mark_value=1)
        set_mark(b, ref=(1, 10), mark_kind=MARK_DOMAIN, mark_value=1)
        # 同 status 重复（验幂等不增行）
        set_mark(b, ref=(1, 30), mark_kind=MARK_LANG, mark_value=1)
        nodes = query_nodes_by_mark(b, mark_kind=MARK_LANG, mark_value=1)
        common = query_intersection(b, marks=[(MARK_LANG, 1), (MARK_DOMAIN, 1)])
        marks10 = get_marks(b, ref=(1, 10))
        b.close()
        return nodes, common, marks10
    r1 = _run()
    r2 = _run()
    assert r1 == r2
    # 守 sorted 确定序（非插入序 30,10,20）
    assert r1[0] == [(1, 10), (1, 20), (1, 30)]
    assert r1[1] == [(1, 10)]   # 仅 (1,10) 双 mark


# ============ dump_tables 接线 ============

def test_dump_tables_contains_abstract_mark():
    """FormalTrainConfig.dump_tables 含 abstract_mark（片1 接线·跨 run 还原）。"""
    assert ABSTRACT_MARK_TABLE in FormalTrainConfig.dump_tables


# ============ 片2 迁移守卫：concept_node 7→6 列（路径 B） ============

def test_concept_node_schema_six_columns():
    """concept_node 6 列·modality_marker 已迁 abstract_mark（§7.7.1 路径 B·守铁律 6）。"""
    b = DictBackend()
    bootstrap(b)
    meta = b._tables["concept_node"]
    assert "modality_marker" not in meta["columns"]
    assert meta["columns"] == ["space_id", "local_id", "type",
                               "born_granularity", "version_head", "tier"]
    b.close()


def test_node_store_put_no_modality_marker_param():
    """NodeStore.put 删 modality_marker 参数（传该参 raise TypeError·签名固化）。"""
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    ns = NodeStore(b)
    ns.put(sp.space_id, 1, node_type=NODE_CONCEPT)   # 无 modality_marker·正常
    row = ns.get(sp.space_id, 1)
    assert "modality_marker" not in row   # 列已删
    assert row["tier"] == TIER_SHADOW   # 默认 TIER_SHADOW=1·守其余列正常
    with pytest.raises(TypeError):
        ns.put(sp.space_id, 2, node_type=NODE_CONCEPT, modality_marker=1)   # 参已删
    b.close()


def test_concept_node_dump_load_roundtrip_six_columns(tmp_path):
    """B7 dump/load round-trip：6 列 concept_node 跨 run 还原（窗口期零生产 dump·守 schema 变更）。

    dump_run 序列化 concept_node 全行（无 modality_marker）·load_run 还原·
    新 backend select 出的行无 modality_marker 键（6 列 schema 一致）。
    abstract_mark 同 round-trip（dump_tables 接线）。
    """
    # 源 backend：建 concept_node + abstract_mark 行
    b1 = SQLiteBackend(":memory:")
    bootstrap(b1)
    register_abstract_mark(b1)
    reg = SpaceRegistry(b1)
    sp = AbstractSpace.create(reg, "core")
    ns = NodeStore(b1)
    ns.put(sp.space_id, 1, node_type=NODE_CONCEPT, tier=2)
    set_mark(b1, ref=(sp.space_id, 1), mark_kind=MARK_LANG, mark_value=1)
    # dump（concept_node + abstract_mark）
    run_dir = str(tmp_path)
    dump_run(b1, run_dir, "run1", spaces=[sp.space_id],
             tables=("concept_node", ABSTRACT_MARK_TABLE))
    # 目标 backend：同 schema·load
    b2 = SQLiteBackend(":memory:")
    bootstrap(b2)
    register_abstract_mark(b2)
    load_run(b2, run_dir, "run1")
    rows = b2.select("concept_node", where={"space_id": sp.space_id})
    assert len(rows) == 1
    assert "modality_marker" not in rows[0]   # 6 列·无 modality_marker（B7 守）
    assert rows[0]["tier"] == 2
    # abstract_mark 也 round-trip
    assert get_mark(b2, ref=(sp.space_id, 1), mark_kind=MARK_LANG) == 1
    b1.close()
    b2.close()


def test_concept_point_no_mark_modality_bit_identical():
    """bit-identical：概念点 modality_marker=0（未设）→ 无 abstract_mark MARK_MODALITY → get_mark None。

    §7.7.1 决断 1：概念点从未被标模态（observe/ensure 从未传非 0·cognition/ 零真消费读）·
    迁移语义：modality_marker=0（vestigial）→ 不挂 mark → get_mark(MARK_MODALITY) None ≡ 现状。
    完全等价·bit-identical。未来标模态须显式 set_mark（首版无 caller）。
    """
    b = DictBackend()
    bootstrap(b)
    register_abstract_mark(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    ns = NodeStore(b)
    ns.put(sp.space_id, 1, node_type=NODE_CONCEPT)   # 概念点（无模态标·≡ 旧 modality_marker=0）
    # 读侧契约：无 mark = 未标 modality（非 LANGUAGE=1·未设）
    assert get_mark(b, ref=(sp.space_id, 1), mark_kind=MARK_MODALITY) is None
    assert get_marks(b, ref=(sp.space_id, 1)) == []
    b.close()


# ============ 片3 接线：lang 挂词形 NODE_WORD（解 target_lang 缺口） ============

def test_normalize_to_concept_tags_mark_lang():
    """normalize_to_concept OOV NODE_WORD 挂 MARK_LANG（set_mark 写侧·§7.7.1 片3）。"""
    b = DictBackend()
    bootstrap(b)
    register_abstract_mark(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b, None)
    wzh = normalize_to_concept("zhong", concept_index=ci, edge_store=es,
                                space_id=sp.space_id, source=SOURCE_BARE_TEXT,
                                backend=b, lang=LANG_ZH)
    wen = normalize_to_concept("en", concept_index=ci, edge_store=es,
                                space_id=sp.space_id, source=SOURCE_BARE_TEXT,
                                backend=b, lang=LANG_EN)
    assert get_mark(b, ref=wzh, mark_kind=MARK_LANG) == LANG_ZH
    assert get_mark(b, ref=wen, mark_kind=MARK_LANG) == LANG_EN
    b.close()


def test_normalize_to_concept_no_lang_none_skips_mark():
    """lang=LANG_NONE（非语言模态）→ 不挂 MARK_LANG（守门）·backend=None → skip（bare fixture）。"""
    b = DictBackend()
    bootstrap(b)
    register_abstract_mark(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b, None)
    w = normalize_to_concept("tok", concept_index=ci, edge_store=es,
                              space_id=sp.space_id, source=SOURCE_BARE_TEXT,
                              backend=b, lang=LANG_NONE)   # 非语言
    assert get_mark(b, ref=w, mark_kind=MARK_LANG) is None   # 不挂
    b.close()


def test_lang_of_reads_abstract_mark_production_path():
    """片3 反 theater 核心：lang_of 生产路径（无注入）读 abstract_mark MARK_LANG。

    normalize_to_concept 建 ZH/EN 词形 → ConceptGraph 不注入 lang_of →
    graph_view.lang_of Option A 读 abstract_mark（WZH=ZH/WEN=EN/概念=None）·
    dispatch_slot target_lang 偏好对词形候选生效（解 C1 缺口·反 theater 真消费）。
    """
    b = DictBackend()
    bootstrap(b)
    register_abstract_mark(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b, None)
    # 概念 C + 两异 lang 词形（OOV NODE_WORD·set_mark MARK_LANG）
    C = ci.ensure("concept", space_id=sp.space_id, tier=TIER_PRIMARY)
    WZH = normalize_to_concept("zhong", concept_index=ci, edge_store=es,
                                space_id=sp.space_id, source=SOURCE_BARE_TEXT,
                                backend=b, lang=LANG_ZH)
    WEN = normalize_to_concept("en", concept_index=ci, edge_store=es,
                                space_id=sp.space_id, source=SOURCE_BARE_TEXT,
                                backend=b, lang=LANG_EN)
    # REFERS_TO 词形→概念（activate_candidates 反向）
    for w in (WZH, WEN):
        es.add(space_id_from=sp.space_id, local_id_from=w[1],
               space_id_to=sp.space_id, local_id_to=C[1],
               edge_type=EDGE_REFERS_TO, strength=1, source=SOURCE_BARE_TEXT,
               tier=TIER_PRIMARY)
    # 生产 ConceptGraph：surface_of 注入（modality_serialize 渲染用）·lang_of 不注入（读 abstract_mark）
    surfaces = {C: "c", WZH: "zhong", WEN: "en"}
    g = ConceptGraph(b, surface_of=lambda r: surfaces.get(r))
    # lang_of 读 abstract_mark（无注入·生产路径）
    assert g.lang_of(WZH) == LANG_ZH
    assert g.lang_of(WEN) == LANG_EN
    assert g.lang_of(C) is None   # 概念点无 mark ≡ 既有注入式 None（bit-identical）
    # dispatch_slot target_lang 偏好对词形候选生效（解 C1 缺口）
    dag = PathResult(path=PathData(edges=[], struct_unit_refs=[]),
                     terminal=TERMINAL_REACHED_SINK, sink=C,
                     topo_layers=[[C]], convergence={}, source=None)
    word_zh, src_zh = dispatch_slot(RoleSlot(ref=C), dag, g, WorkMemory(), LANG_ZH)
    assert word_zh == "zhong"   # target=ZH 选 WZH（lang_of 读 MARK_LANG 真过滤）
    assert src_zh == LINEAGE_CONCEPT_FILL
    word_en, _ = dispatch_slot(RoleSlot(ref=C), dag, g, WorkMemory(), LANG_EN)
    assert word_en == "en"   # target=EN 选 WEN
    b.close()
