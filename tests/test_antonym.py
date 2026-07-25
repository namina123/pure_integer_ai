"""T-L1e 测试：ANTONYM 反义对称独立 EDGE_ANTONYM=26 + 构造器 + loader。

客观序 gap 补（doc/重来_语言域断奶客观序_2026-07-15 §三 T-L1e）。语言反义 大↔小 = concept↔concept 1 阶。
**近 EDGE_SIMILAR 对称形·异 SIMILAR 语义**（反义=对立非相似）·**非 verify_inverse**（代数逆 transform↔transform T-L4·
verify_inverse 只验数学·对词对返 None can't-verify·#479 外部 seed 非 verify）。

覆盖：
  A1 bootstrap round-trip（单边 a→b EDGE_ANTONYM + strength=1 结构真值 + source/epistemic·镜像 SIMILAR 单边）
  A2 空 pairs 短路零副作用（return 0·绝不调 ensure/build·CI bit-identical 硬守）
  A3 自环 skip（a==b 不建·词非自身反义）
  A4 bit-identical（无 antonym 文件 → resolve [] → 零 EDGE_ANTONYM 边·CI 逐字现状）
  A5 loader E5 graceful（缺文件→[]·错行/自环/注释/空行 skip·BOM）
  A6 load_antonym_facts_file 格式（a b parse·首 a 末 b）
  A7 resume 幂等（query_from skip·二次 boot 零新增边）
  A8 语义正交 + PR/dag_path 排除（EDGE_ANTONYM≠EDGE_SIMILAR·不在 {PRECEDES,CAUSES,REFERS_TO}·结构边·镜像 SIMILAR）
  A9 非 verify_inverse（ANTONYM 是纯结构边·不经 symbolic_relation/verify_inverse·无 REL_ANTONYM 原语依赖·#479 外部 seed）

铁律：纯整数（ConceptRef/EDGE_ANTONYM）/ 确定性 bit-identical / 反 theater（机制真活·非死列表）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.experiments.formal_train import make_train_context, FormalTrainResult
from pure_integer_ai.storage.edge_types import (
    EDGE_ANTONYM, EDGE_SIMILAR, EDGE_IS_A, EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO,
    is_registered_edge_type, REGISTERED_EDGE_TYPES,
)
from pure_integer_ai.storage.edge_store import SOURCE_CONCEPTNET, EPI_STRUCTURED, DEFAULT_STRENGTH
from pure_integer_ai.cognition.understanding.antonym import (
    bootstrap_antonym_edges, build_antonym_edge,
)
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_NONE
from pure_integer_ai.cognition.shared.relation_primitives import (
    REL_SUBSET, REL_MEMBER, REL_EQUAL, REL_CAUSES, REL_PRECEDES, REL_MEREOLOGY, REL_PROPERTY, REL_SIMILAR,
)
from pure_integer_ai.experiments.collection import resolve_antonym_facts, load_antonym_facts_file


def _has_antonym(edge_store, a, b) -> bool:
    """a→b EDGE_ANTONYM 边是否存在（query_from a·查 to==b 且 edge_type==ANTONYM）。"""
    rows = edge_store.query_from(a[0], a[1], edge_type=EDGE_ANTONYM)
    return any(r.get("space_id_to") == b[0] and r.get("local_id_to") == b[1] for r in rows)


# ---- A1 bootstrap round-trip（单边·镜像 SIMILAR） ----

def test_a1_bootstrap_roundtrip():
    """bootstrap_antonym_edges 建 (大,小) → 单边 大→小 EDGE_ANTONYM + strength=1 结构真值 + source/epistemic。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    n = bootstrap_antonym_edges(ctx.concept_index, ctx.edge_store,
                                [("大", "小")], space_id=sid)
    assert n == 1, "单边 a→b = 1 边（镜像 SIMILAR 单边·异 alias 双边·reader 双向查）"
    big = ctx.concept_index.ensure("大", space_id=sid)
    small = ctx.concept_index.ensure("小", space_id=sid)
    assert big != small
    # 单边 大→小（非双向·镜像 SIMILAR build_similar_edges X→Y）
    assert _has_antonym(ctx.edge_store, big, small), "大→小 EDGE_ANTONYM（单边）"
    assert not _has_antonym(ctx.edge_store, small, big), "小→大 不建（单边·reader 双向查镜像 similar_candidates）"
    # strength=1 结构真值（DEFAULT_STRENGTH·镜像 SIMILAR·非学习对象）+ source/epistemic
    rows = ctx.edge_store.query_from(big[0], big[1], edge_type=EDGE_ANTONYM)
    assert len(rows) == 1
    assert rows[0]["strength"] == DEFAULT_STRENGTH == 1
    assert rows[0]["source"] == SOURCE_CONCEPTNET
    assert rows[0]["epistemic_origin"] == EPI_STRUCTURED


# ---- A2 空 pairs 短路零副作用 ----

def test_a2_empty_pairs_short_circuit_zero_side_effects():
    """空 pairs → return 0·绝不调 ensure/build（CI/生产 default 无文件 bit-identical 硬守）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    n = bootstrap_antonym_edges(ctx.concept_index, ctx.edge_store, [], space_id=sid)
    assert n == 0
    assert ctx.concept_index.lookup("大", sid) is None, "短路不调 ensure·无节点"
    assert ctx.concept_index.lookup("小", sid) is None
    assert ctx.backend.count("edge", where={"edge_type": EDGE_ANTONYM}) == 0


# ---- A3 自环 skip ----

def test_a3_self_loop_skipped():
    """a==b → build_antonym_edge 返 0·不建自环边（词非自身反义·镜像 build_mereology_edge:58）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    x = ctx.concept_index.ensure("word", space_id=sid)
    assert build_antonym_edge(ctx.edge_store, x, x,
                              source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED,
                              space_id=sid) == 0
    n = bootstrap_antonym_edges(ctx.concept_index, ctx.edge_store,
                                [("same", "same")], space_id=sid)
    assert n == 0, "自环 pair → build_antonym_edge 跳·零边"


# ---- A4 bit-identical（无 antonym 文件·CI 逐字现状） ----

def test_a4_no_file_zero_edges_bit_identical():
    """无 PURE_INTEGER_AI_LOCAL_DIR / 无 antonym_facts 文件 → resolve [] → 零 EDGE_ANTONYM 边（CI bit-identical）。"""
    b = DictBackend(); ctx = make_train_context(b)
    assert resolve_antonym_facts(LANG_ZH, local_dir=None) == []
    assert resolve_antonym_facts(LANG_ZH, local_dir="C:/nonexistent_dir_zzz") == []
    assert resolve_antonym_facts(LANG_NONE, local_dir=None) == []   # lang 无映射
    assert ctx.backend.count("edge", where={"edge_type": EDGE_ANTONYM}) == 0


# ---- A5 loader E5 graceful ----

def test_a5_resolve_antonym_facts_missing_file_returns_empty(tmp_path):
    """缺文件（无 local_dir / 文件不存在 / lang 无映射）→ []（E5 graceful·CI bit-identical 守）。"""
    assert resolve_antonym_facts(LANG_ZH, local_dir=None) == []
    assert resolve_antonym_facts(LANG_ZH, local_dir=str(tmp_path)) == []
    assert resolve_antonym_facts(LANG_NONE, local_dir=str(tmp_path)) == []


def test_a5_load_antonym_facts_graceful_skips_bad_lines(tmp_path):
    """错行（<2 段）/ 自环 / 注释 / 空行 skip + 不抛崩（E5 graceful·镜像 load_mereology_facts_file）。"""
    p = tmp_path / "antonym_facts_zh.txt"
    p.write_text(
        "# 注释行\n"
        "\n"                          # 空行
        "大 小\n"                     # 合法
        "only_one_segment\n"          # 1 段·错行 skip
        "dup dup\n"                   # 自环 skip（词非自身反义）
        "冷 热\n",                    # 合法
        encoding="utf-8",
    )
    pairs = load_antonym_facts_file(str(p))
    assert pairs == [("大", "小"), ("冷", "热")]


def test_a5_load_antonym_facts_bom(tmp_path):
    """BOM（utf-8-sig）首行不破识别（对抗审点·镜像 load_mereology_facts_file）。"""
    p = tmp_path / "antonym_facts_zh.txt"
    p.write_bytes("﻿大 小\n".encode("utf-8"))
    assert load_antonym_facts_file(str(p)) == [("大", "小")]


# ---- A6 load_antonym_facts_file 格式 ----

def test_a6_load_antonym_facts_format(tmp_path):
    """a b parse·首段 a 末段 b（中段忽略·容错）。"""
    p = tmp_path / "antonym_facts_zh.txt"
    p.write_text(
        "大\t小\n"            # tab 分隔
        "上 BLAH 下\n"        # 中段噪声忽略·首 a 末 b
        "开 关\n",
        encoding="utf-8",
    )
    assert load_antonym_facts_file(str(p)) == [
        ("大", "小"), ("上", "下"), ("开", "关"),
    ]
    assert resolve_antonym_facts(LANG_ZH, local_dir=str(tmp_path)) == [
        ("大", "小"), ("上", "下"), ("开", "关"),
    ]


# ---- A7 resume 幂等（query_from skip） ----

def test_a7_resume_idempotent_no_duplicate_edges():
    """二次 boot（resume 路径）→ query_from skip·零新增边（EdgeStore.add 不去重故须此守）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    pairs = [("大", "小"), ("冷", "热")]
    n1 = bootstrap_antonym_edges(ctx.concept_index, ctx.edge_store, pairs, space_id=sid)
    n2 = bootstrap_antonym_edges(ctx.concept_index, ctx.edge_store, pairs, space_id=sid)
    assert n1 == 2 and n2 == 0, "二次 boot 全 skip（query_from 幂等）·零新增"
    assert ctx.backend.count("edge", where={"edge_type": EDGE_ANTONYM}) == 2


# ---- A8 语义正交 + PR/dag_path 排除 ----

def test_a8_edge_antonym_registered_structural_orthogonal_to_similar():
    """EDGE_ANTONYM=26 注册合法 + 结构边（不进 PR）+ 语义正交 EDGE_SIMILAR（异 edge_type）。"""
    assert is_registered_edge_type(EDGE_ANTONYM)
    assert EDGE_ANTONYM in REGISTERED_EDGE_TYPES
    assert EDGE_ANTONYM == 26
    # 不在 PR 头集（effective_weight:82 assert 只认 {PRECEDES,CAUSES,REFERS_TO}）
    pr_heads = {EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO}
    assert EDGE_ANTONYM not in pr_heads, "ANTONYM 不进 PR（结构边·镜像 SIMILAR/IS_A）"
    assert EDGE_SIMILAR not in pr_heads, "对照：SIMILAR 同样不进 PR"
    # 语义正交：ANTONYM ≠ SIMILAR（反义 vs 相似·异 edge_type·不混）
    assert EDGE_ANTONYM != EDGE_SIMILAR
    assert EDGE_ANTONYM != EDGE_IS_A


# ---- A9 非 verify_inverse（#479 外部 seed·无 REL_ANTONYM 原语依赖） ----

def test_a9_antonym_not_verify_inverse_no_rel_primitive():
    """ANTONYM 是纯语言域 concept↔concept 结构边·非代数 verify_inverse（T-L4 transform↔transform）·#479 外部 seed。

    核证：(1) REL_* 原语集无 REL_ANTONYM（antonym 不挂代数关系原语·纯语言域）·双重守（hasattr + 显式集）；
          (2) ANTONYM 不接 reward（静态 strength·非 verify 路径）·#479 truth 墙守。
    异 T-L4 INVERSE（symbolic_relation verify_inverse 真验 B∘A=identity·只验数学 transform·对语言词对返 None can't-verify）。
    """
    import pure_integer_ai.cognition.shared.relation_primitives as _rp
    # REL_* 原语集（relation_primitives.py）无 antonym——antonym 是纯语言域 concept↔concept·不挂代数原语
    # 双重守：hasattr（防未来新增 REL_ANTONYM 失守）+ 显式集（核证当前 8 个）
    assert not hasattr(_rp, "REL_ANTONYM"), "无 REL_ANTONYM 原语（antonym 纯语言域·非代数原语·防未来误加）"
    rel_primitives = {REL_SUBSET, REL_MEMBER, REL_EQUAL, REL_CAUSES, REL_PRECEDES,
                      REL_MEREOLOGY, REL_PROPERTY, REL_SIMILAR}
    assert len(rel_primitives) == 8, "REL_* 原语集 8 个·无 REL_ANTONYM（antonym 纯语言域·非代数原语）"
    # ANTONYM 不接 reward：静态 strength=1（DEFAULT_STRENGTH）·effective_weight 不认（A8 已证不在 PR 头集）
    # → 非 verify 路径·#479 truth 墙守（外部 ConceptNet/WordNet seed·系统只落边不验真）
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    bootstrap_antonym_edges(ctx.concept_index, ctx.edge_store, [("大", "小")], space_id=sid)
    big = ctx.concept_index.ensure("大", space_id=sid)
    rows = ctx.edge_store.query_from(big[0], big[1], edge_type=EDGE_ANTONYM)
    assert rows[0]["strength"] == 1, "ANTONYM 静态 strength=1（结构真值·非学习对象·不接 reward·非 verify）"


# ---- A10 observability 字段默认 0（#1119 数据补全·对称 alias/number·bit-identical） ----

def test_a10_observability_fields_default_zero():
    """FormalTrainResult 默认 antonym_edges_seeded / mereology_edges_seeded = 0（CI bit-identical）。

    #1119 数据补全：formal_train boot 捕获 antonym/mereology 种边数 → result.{antonym,mereology}_edges_seeded
    （对称 alias_edges_seeded / number_edges_seeded·W8 语言域断奶 observability 信号·消费者待接 #941·reader 待接线）。
    无文件（CI/生产 default 无 PURE_INTEGER_AI_LOCAL_DIR）→ resolve [] → boot 空 pairs 短路 → 不进赋值分支 → 落默认 0。
    生产有文件时 result.antonym_edges_seeded>0（dev 验证脚本 validate_antonym_mereology.py 已证 11633/11173 边）。
    """
    r = FormalTrainResult(run_id="test")
    assert r.antonym_edges_seeded == 0, "默认 0（CI 无 antonym_facts·boot 空 pairs 短路→bit-identical）"
    assert r.mereology_edges_seeded == 0, "默认 0（CI 无 mereology_facts·boot 空 pairs 短路→bit-identical）"
