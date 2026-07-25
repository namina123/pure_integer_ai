"""T-L1c 测试：SIMILAR 近义对称 EDGE_SIMILAR=24 boot-side loader+bootstrap（#1132·镜像 test_antonym.py）。

EDGE_SIMILAR 机制全在（STEP5 PR4 #898·observe-side build_similar_edges + dispatch_slot reader）·
本测补 boot-side（resolve_similar_facts loader + bootstrap_similar_edges）·镜像 antonym 范式。

测：
  S1 bootstrap round-trip（单边 a→b EDGE_SIMILAR + strength=1 结构真值 + source/epistemic·镜像 antonym A1）
  S2 空 pairs 短路零副作用（spy 验零 ensure/insert·bit-identical 硬守·镜像 antonym A2）
  S3 自环跳（a==b 不建·镜像 antonym A3）
  S4 无文件零边 bit-identical（无 similar_facts → resolve [] → 零 EDGE_SIMILAR·镜像 antonym A4）
  S5 resolve_similar_facts 缺文件返空 + load_similar_facts E5 graceful 错行 skip（镜像 antonym A5）
  S6 幂等（重复 bootstrap 不建重复边·query_from skip·镜像 antonym 幂等）
  S7 语义正交（EDGE_SIMILAR≠EDGE_ANTONYM·不在 {PRECEDES,CAUSES,REFERS_TO}·结构边·镜像 antonym A8）

铁律：纯整数（ConceptRef/EDGE_SIMILAR）/ 确定性 bit-identical / 反 theater（机制真活·非死列表）。
"""
from __future__ import annotations

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.edge_types import (
    EDGE_SIMILAR, EDGE_ANTONYM, EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO,
)
from pure_integer_ai.storage.edge_store import SOURCE_CONCEPTNET, SOURCE_CHINESE_KB, EPI_STRUCTURED
from pure_integer_ai.cognition.understanding.similar import (
    bootstrap_similar_edges, build_similar_edge,
)
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_NONE
from pure_integer_ai.experiments.collection import resolve_similar_facts, load_similar_facts_file


def _has_similar(edge_store, a, b):
    """a→b EDGE_SIMILAR 边是否存在（query_from a·查 to==b 且 edge_type==SIMILAR）。"""
    rows = edge_store.query_from(a[0], a[1], edge_type=EDGE_SIMILAR)
    return any(r.get("space_id_to") == b[0] and r.get("local_id_to") == b[1] for r in rows)


def test_s1_bootstrap_roundtrip():
    """bootstrap_similar_edges 建 (开心,高兴) → 单边 开心→高兴 EDGE_SIMILAR + strength=1 + source/epistemic。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    n = bootstrap_similar_edges(ctx.concept_index, ctx.edge_store,
                                [("开心", "高兴")], space_id=sid, source=SOURCE_CHINESE_KB)
    assert n == 1
    kaixin = ctx.concept_index.ensure("开心", space_id=sid)
    gxing = ctx.concept_index.ensure("高兴", space_id=sid)
    assert _has_similar(ctx.edge_store, kaixin, gxing), "开心→高兴 EDGE_SIMILAR（单边）"
    # strength 恒=1 结构真值 + source provenance + epistemic EPI_STRUCTURED
    rows = ctx.edge_store.query_from(kaixin[0], kaixin[1], edge_type=EDGE_SIMILAR)
    assert rows[0]["strength"] == 1
    assert rows[0]["source"] == SOURCE_CHINESE_KB
    assert rows[0]["epistemic_origin"] == EPI_STRUCTURED


def test_s2_empty_pairs_short_circuit_zero_side_effects():
    """空 pairs → return 0·绝不调 ensure/query_from/build（bit-identical 硬守·镜像 bootstrap_antonym_edges:99-100）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    edges_before = ctx.backend.count("edge")
    n = bootstrap_similar_edges(ctx.concept_index, ctx.edge_store, [], space_id=sid)
    assert n == 0
    assert ctx.backend.count("edge") == edges_before   # 零新边
    assert ctx.backend.count("edge", where={"edge_type": EDGE_SIMILAR}) == 0


def test_s3_self_loop_skipped():
    """自环（a==b）不建（词非自身近义·build_similar_edge 早跳）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    n = bootstrap_similar_edges(ctx.concept_index, ctx.edge_store,
                                [("一样", "一样")], space_id=sid, source=SOURCE_CONCEPTNET)
    assert n == 0
    assert ctx.backend.count("edge", where={"edge_type": EDGE_SIMILAR}) == 0


def test_s4_no_file_zero_edges_bit_identical():
    """无 PURE_INTEGER_AI_LOCAL_DIR / 无 similar_facts 文件 → resolve_similar_facts [] → 零 EDGE_SIMILAR 边（CI bit-identical）。"""
    import os
    assert "PURE_INTEGER_AI_LOCAL_DIR" not in os.environ or not os.environ["PURE_INTEGER_AI_LOCAL_DIR"]
    assert resolve_similar_facts(LANG_ZH) == []
    b = DictBackend(); ctx = make_train_context(b)
    assert ctx.backend.count("edge", where={"edge_type": EDGE_SIMILAR}) == 0


def test_s5_resolve_similar_facts_missing_file_returns_empty(tmp_path):
    """缺文件 → resolve_similar_facts 返 []（E5 graceful·镜像 resolve_antonym_facts）。"""
    assert resolve_similar_facts(LANG_ZH, local_dir=str(tmp_path)) == []
    # LANG_NONE 无映射 → []
    assert resolve_similar_facts(LANG_NONE, local_dir=str(tmp_path)) == []


def test_s5b_load_similar_facts_graceful_skips_bad_lines(tmp_path):
    """load_similar_facts_file：注释/空行/错行/自环 skip·合法对保留（E5 graceful·镜像 load_antonym_facts_file）。"""
    p = tmp_path / "similar_facts_zh.txt"
    p.write_text("# 注释\n\n开心 高兴\n坏行\n重复 重复\n快乐 愉快\n", encoding="utf-8")
    pairs = load_similar_facts_file(str(p))
    assert pairs == [("开心", "高兴"), ("快乐", "愉快")]


def test_s6_idempotent_no_duplicate():
    """重复 bootstrap 同对 → query_from 幂等 skip·不建重复边（resume 跨 run 不 corrupt）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    pairs = [("开心", "高兴")]
    n1 = bootstrap_similar_edges(ctx.concept_index, ctx.edge_store, pairs,
                                 space_id=sid, source=SOURCE_CHINESE_KB)
    n2 = bootstrap_similar_edges(ctx.concept_index, ctx.edge_store, pairs,
                                 space_id=sid, source=SOURCE_CHINESE_KB)
    assert n1 == 1 and n2 == 0   # 第二次幂等 skip
    assert ctx.backend.count("edge", where={"edge_type": EDGE_SIMILAR}) == 1


def test_s7_semantic_orthogonality():
    """EDGE_SIMILAR ≠ EDGE_ANTONYM·不在 reward 反传头集 {PRECEDES,CAUSES,REFERS_TO}（结构边·镜像 antonym A8）。"""
    assert EDGE_SIMILAR != EDGE_ANTONYM
    assert EDGE_SIMILAR not in (EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO)
    # build_similar_edge epistemic 白名单守（禁裸共现·镜像 build_antonym_edge）
    b = DictBackend(); ctx = make_train_context(b)
    import pytest
    with pytest.raises(AssertionError):
        build_similar_edge(ctx.edge_store, (0, 1), (0, 2),
                           source=SOURCE_CONCEPTNET, epistemic=999, space_id=ctx.space_id)
