"""#1133 测试：抽象→EDGE_IS_A 泛化 boot（ChineseSemanticKB 抽象关系库·source=SOURCE_CHINESE_KB·复用 bootstrap_is_a_edges）。

抽象 = IS_A 泛化（#1133 纠偏·撤回 EDGE_ABSTRACT·abstraction.py+EDGE_IS_A+bootstrap_is_a_edges 全在）。
本测验证：resolve_abstract_facts loader + bootstrap_is_a_edges(source=SOURCE_CHINESE_KB) → EDGE_IS_A 边·
**异 ConceptNet provenance**（build_isa_ancestor_map_external 刀C 验证 filter CONCEPTNET·abstract 边不污染）。

测：
  AB1 resolve_abstract_facts 读文件返 (child,parent) 对（复用 is_a 格式·E5 graceful）
  AB2 bootstrap_is_a_edges(source=CHINESE_KB) 建 EDGE_IS_A + provenance source=9（异 CONCEPTNET=1）
  AB3 无文件 → resolve [] → 零 EDGE_IS_A（CI bit-identical）
  AB4 provenance 隔离：abstract IS_A 边（source=CHINESE_KB）不进 build_isa_ancestor_map_external（filter CONCEPTNET）

铁律：纯整数 / 确定性 bit-identical / 反 theater。
"""
from __future__ import annotations

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.edge_types import EDGE_IS_A
from pure_integer_ai.storage.edge_store import SOURCE_CONCEPTNET, SOURCE_CHINESE_KB
from pure_integer_ai.cognition.understanding.is_a import bootstrap_is_a_edges
from pure_integer_ai.cognition.process.abstraction import (
    build_isa_ancestor_map_external, build_isa_ancestor_map,
)
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_NONE
from pure_integer_ai.experiments.collection import resolve_abstract_facts


def test_ab1_resolve_abstract_facts_reads_file(tmp_path):
    """resolve_abstract_facts 读 abstract_facts_zh.txt → (child,parent) 对（复用 is_a 格式·E5 graceful）。"""
    p = tmp_path / "abstract_facts_zh.txt"
    p.write_text("# 注释\n\n苹果 水果\n坏行\n跑步 运动\n", encoding="utf-8")
    pairs = resolve_abstract_facts(LANG_ZH, local_dir=str(tmp_path))
    assert pairs == [("苹果", "水果"), ("跑步", "运动")]
    # 缺文件 / 无映射 lang → []
    assert resolve_abstract_facts(LANG_ZH, local_dir=str(tmp_path / "nope")) == []
    assert resolve_abstract_facts(LANG_NONE, local_dir=str(tmp_path)) == []


def test_ab2_bootstrap_isa_provenance_chinese_kb():
    """bootstrap_is_a_edges(source=SOURCE_CHINESE_KB) 建 EDGE_IS_A·provenance source=9（异 CONCEPTNET=1）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    n = bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store,
                             [("苹果", "水果")], space_id=sid, source=SOURCE_CHINESE_KB)
    assert n == 1
    apple = ctx.concept_index.ensure("苹果", space_id=sid)
    rows = ctx.edge_store.query_from(apple[0], apple[1], edge_type=EDGE_IS_A)
    assert len(rows) == 1
    assert rows[0]["source"] == SOURCE_CHINESE_KB   # 9·provenance 准（异 CONCEPTNET=1）


def test_ab3_no_file_zero_edges_bit_identical():
    """无 PURE_INTEGER_AI_LOCAL_DIR / 无 abstract_facts 文件 → resolve_abstract_facts [] → 零新 EDGE_IS_A（CI bit-identical）。"""
    import os
    assert "PURE_INTEGER_AI_LOCAL_DIR" not in os.environ or not os.environ["PURE_INTEGER_AI_LOCAL_DIR"]
    assert resolve_abstract_facts(LANG_ZH) == []


def test_ab4_provenance_isolation_external_graph():
    """abstract IS_A 边（source=CHINESE_KB）不进 build_isa_ancestor_map_external（刀C 验证 filter source=CONCEPTNET+EPI_STRUCTURED）。

    反 MED-1 式污染：若 abstract 误盖 SOURCE_CONCEPTNET → 会污染刀C 外部源验证图（自证闭环风险）。
    本测证 abstract stamp CHINESE_KB → 外部图 filter 排除 → 祖先图空（隔离成立）。
    """
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    # 种 abstract IS_A 边（source=CHINESE_KB·异 CONCEPTNET）
    bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store,
                         [("苹果", "水果")], space_id=sid, source=SOURCE_CHINESE_KB)
    # 刀C 外部源验证图（filter source=SOURCE_CONCEPTNET）应排除 abstract 边 → 空
    ext_map = build_isa_ancestor_map_external(b, space_id=sid)
    apple = ctx.concept_index.ensure("苹果", space_id=sid)
    assert ext_map.get(apple) is None or len(ext_map.get(apple, set())) == 0, \
        "abstract IS_A 边（CHINESE_KB）不应进 CONCEPTNET-filter 外部验证图（provenance 隔离）"


def test_ab5_ancestor_map_includes_cyclecleaned_abstract():
    """build_isa_ancestor_map **含** abstract IS_A（#1133 DONE·stopgap 已撤·cycle-cleaned DAG·enrich LCA）。

    原 stopgap（排除 SOURCE_CHINESE_KB abstract raw·giant-SCC 循环噪声）已**撤**——#1133 cycle-cleaning
    （scratch/clean_abstract_cycles.py·DFS back-edge removal·306998 raw → 300233 DAG·0 环）+ #1142 语料相关过滤
    （boot 时 abstract 只留 corpus-relevant）→ abstract 子集是干净 DAG（子集 of DAG is DAG）。graph_algebra
    （SCC 凝聚 O(V+E)·#1136）处理。故 abstract IS_A（苹果→水果·CHINESE_KB·干净 DAG 对）**重进 closure**·
    与 ConceptNet IS_A（猫→动物）共同 enrich ancestor_map 驱动 LCA。
    **bit-identical 守**：CI 无 abstract_facts → 无 CHINESE_KB 边 → 同旧（此测注入 CHINESE_KB 边验新行为）。
    """
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store, [("猫", "动物")],
                         space_id=sid, source=SOURCE_CONCEPTNET)
    bootstrap_is_a_edges(ctx.concept_index, ctx.edge_store, [("苹果", "水果")],
                         space_id=sid, source=SOURCE_CHINESE_KB)
    amap = build_isa_ancestor_map(b, space_id=sid)
    cat = ctx.concept_index.ensure("猫", space_id=sid)
    animal = ctx.concept_index.ensure("动物", space_id=sid)
    apple = ctx.concept_index.ensure("苹果", space_id=sid)
    fruit = ctx.concept_index.ensure("水果", space_id=sid)
    assert animal in amap.get(cat, set()), "ConceptNet IS_A（猫→动物）进 ancestor_map（驱动 LCA）"
    # #1133 DONE：abstract IS_A（苹果→水果·CHINESE_KB·cycle-cleaned 干净 DAG 对）**重进 closure**
    assert fruit in amap.get(apple, set()), \
        "abstract IS_A（苹果→水果·CHINESE_KB·stopgap 已撤·cycle-cleaned）重进 ancestor_map（enrich LCA）"
    assert apple in amap, "苹果（abstract IS_A·重进 closure）在 ancestor_map（#1133 stopgap 已撤）"
