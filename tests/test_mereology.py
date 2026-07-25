"""T-L1d 测试：MEREOLOGY 部分-整体独立 EDGE_MEREOLOGY=25 + 构造器 + loader + cue 路由修正。

客观序 gap 补（doc/重来_语言域断奶客观序_2026-07-15 §三 T-L1d"潜在语义误路由"）。
解 cue_words 首版 REL_MEREOLOGY 折入 IS_A_CUE（部分-整体被建成 IsA 边=语义误路由）·独立 typed edge 守语义正交。
**MEREOLOGY ≠ IS_A**（部分-整体 ≠ 子集·车轮 part-of 汽车·非车轮⊂汽车）。

覆盖：
  M1 bootstrap round-trip（part→whole EDGE_MEREOLOGY + ensure + 静态 strength + source/epistemic）
  M2 空 pairs 短路零副作用（return 0·绝不调 ensure/build·CI bit-identical 硬守）
  M3 自环 skip（part==whole 不建边）
  M4 bit-identical（无 mereology 文件 → resolve [] → 零 EDGE_MEREOLOGY 边·CI 逐字现状）
  M5 loader E5 graceful（缺文件→[]·错行/自环/注释/空行 skip·BOM）
  M6 load_mereology_facts_file 格式（part whole parse·首段 part 末段 whole·中段忽略）
  M7 resume 幂等（query_from skip·二次 boot 零新增边·EdgeStore.add 不去重故须此守）
  M8 cue 路由修正（REL_MEREOLOGY→MEREOLOGY_CUE 非 IS_A_CUE·部分 不入 _CUE_WORDS frozenset·gate OFF bit-identical）
  M9 PR/dag_path 排除（EDGE_MEREOLOGY 不在 {PRECEDES,CAUSES,REFERS_TO}·结构边·不接 reward·镜像 IS_A）

铁律：纯整数（ConceptRef/EDGE_MEREOLOGY）/ 确定性 bit-identical / 反 theater（机制真活·非死列表）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.edge_types import (
    EDGE_MEREOLOGY, EDGE_IS_A, EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO,
    is_registered_edge_type, REGISTERED_EDGE_TYPES,
)
from pure_integer_ai.storage.edge_store import SOURCE_CONCEPTNET, EPI_STRUCTURED
from pure_integer_ai.cognition.understanding.mereology import (
    bootstrap_mereology_edges, build_mereology_edge, MEREOLOGY_STRENGTH_EMPIRICAL,
)
from pure_integer_ai.cognition.understanding.cue_words import (
    _REL_KIND_TO_CUE_TYPE, MEREOLOGY_CUE, IS_A_CUE, _CUE_WORDS, cue_type_of,
)
from pure_integer_ai.cognition.shared.relation_primitives import REL_MEREOLOGY
from pure_integer_ai.cognition.shared.types import LANG_ZH
from pure_integer_ai.experiments.collection import resolve_mereology_facts, load_mereology_facts_file


def _has_mereo(edge_store, part, whole) -> bool:
    """part→whole EDGE_MEREOLOGY 边是否存在（query_from part·查 to==whole 且 edge_type==MEREOLOGY）。"""
    rows = edge_store.query_from(part[0], part[1], edge_type=EDGE_MEREOLOGY)
    return any(r.get("space_id_to") == whole[0] and r.get("local_id_to") == whole[1]
               for r in rows)


# ---- M1 bootstrap round-trip ----

def test_m1_bootstrap_roundtrip():
    """bootstrap_mereology_edges 建 (车轮,汽车) → part→whole EDGE_MEREOLOGY + 静态 strength + source/epistemic。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    n = bootstrap_mereology_edges(ctx.concept_index, ctx.edge_store,
                                  [("车轮", "汽车")], space_id=sid)
    assert n == 1, "单向 part→whole = 1 边（异 alias 双向）"
    wheel = ctx.concept_index.ensure("车轮", space_id=sid)   # dedup 命中既有 ref
    car = ctx.concept_index.ensure("汽车", space_id=sid)
    assert wheel != car, "两不同 local_id 概念节点"
    # part→whole 有向（非双向·异 alias）
    assert _has_mereo(ctx.edge_store, wheel, car), "车轮→汽车 EDGE_MEREOLOGY（part→whole）"
    assert not _has_mereo(ctx.edge_store, car, wheel), "汽车→车轮 不建（单向·异 alias 双向）"
    # 静态 strength + source/epistemic（M9 不接 reward·镜像 IS_A）
    rows = ctx.edge_store.query_from(wheel[0], wheel[1], edge_type=EDGE_MEREOLOGY)
    assert len(rows) == 1
    assert rows[0]["strength"] == MEREOLOGY_STRENGTH_EMPIRICAL
    assert rows[0]["source"] == SOURCE_CONCEPTNET
    assert rows[0]["epistemic_origin"] == EPI_STRUCTURED


# ---- M2 空 pairs 短路零副作用 ----

def test_m2_empty_pairs_short_circuit_zero_side_effects():
    """空 pairs → return 0·绝不调 ensure/build（CI/生产 default 无文件 bit-identical 硬守）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    n = bootstrap_mereology_edges(ctx.concept_index, ctx.edge_store,
                                  [], space_id=sid)
    assert n == 0
    # 短路在 ensure 之前·无节点建（lookup 任一 surface 皆 None）
    assert ctx.concept_index.lookup("车轮", sid) is None, "短路不调 ensure·无节点"
    assert ctx.concept_index.lookup("汽车", sid) is None
    # 零 EDGE_MEREOLOGY 边
    assert ctx.backend.count("edge", where={"edge_type": EDGE_MEREOLOGY}) == 0


# ---- M3 自环 skip ----

def test_m3_self_loop_skipped():
    """part==whole → build_mereology_edge 返 0·不建自环边（无义·镜像 build_is_a_edge:57）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    x = ctx.concept_index.ensure("thing", space_id=sid)
    # build_mereology_edge 自环守（part==whole）
    assert build_mereology_edge(ctx.edge_store, x, x,
                                source=SOURCE_CONCEPTNET, epistemic=EPI_STRUCTURED,
                                space_id=sid) == 0
    # bootstrap 层 self-loop pair（同 surface ensure 同 ref）→ build 自环守跳
    n = bootstrap_mereology_edges(ctx.concept_index, ctx.edge_store,
                                  [("same", "same")], space_id=sid)
    assert n == 0, "自环 pair → build_mereology_edge 跳·零边"


# ---- M4 bit-identical（无 mereology 文件·CI 逐字现状） ----

def test_m4_no_file_zero_edges_bit_identical():
    """无 PURE_INTEGER_AI_LOCAL_DIR / 无 mereology_facts 文件 → resolve [] → 零 EDGE_MEREOLOGY 边（CI bit-identical）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    # resolve_mereology_facts 无 local_dir → []
    assert resolve_mereology_facts(LANG_ZH, local_dir=None) == []
    # boot 不调 bootstrap（_pairs 空）→ 零边
    assert ctx.backend.count("edge", where={"edge_type": EDGE_MEREOLOGY}) == 0


# ---- M5 loader E5 graceful ----

def test_m5_resolve_mereology_facts_missing_file_returns_empty(tmp_path):
    """缺文件（无 local_dir / 文件不存在 / lang 无映射）→ []（E5 graceful·CI bit-identical 守）。"""
    assert resolve_mereology_facts(LANG_ZH, local_dir=None) == []   # 无 local_dir
    assert resolve_mereology_facts(LANG_ZH, local_dir=str(tmp_path)) == []   # 无文件
    from pure_integer_ai.cognition.shared.types import LANG_NONE
    assert resolve_mereology_facts(LANG_NONE, local_dir=str(tmp_path)) == []   # lang 无映射


def test_m5_load_mereology_facts_graceful_skips_bad_lines(tmp_path):
    """错行（<2 段）/ 自环 / 注释 / 空行 skip + 不抛崩（E5 graceful·镜像 load_is_a_facts_file）。"""
    p = tmp_path / "mereology_facts_zh.txt"
    p.write_text(
        "# 注释行\n"
        "\n"                          # 空行
        "车轮 汽车\n"                 # 合法
        "only_one_segment\n"          # 1 段·错行 skip
        "dup dup\n"                   # 自环 skip
        "手指 手\n",                  # 合法
        encoding="utf-8",
    )
    pairs = load_mereology_facts_file(str(p))
    assert pairs == [("车轮", "汽车"), ("手指", "手")]


def test_m5_load_mereology_facts_bom(tmp_path):
    """BOM（utf-8-sig）首行不破识别（对抗审点·镜像 load_is_a_facts_file）。"""
    p = tmp_path / "mereology_facts_zh.txt"
    p.write_bytes("﻿车轮 汽车\n".encode("utf-8"))
    assert load_mereology_facts_file(str(p)) == [("车轮", "汽车")]


# ---- M6 load_mereology_facts_file 格式 ----

def test_m6_load_mereology_facts_format(tmp_path):
    """part whole parse·首段 part 末段 whole·中段忽略（容错·whole 须在末·干净 ConceptNet PartOf 对）。"""
    p = tmp_path / "mereology_facts_zh.txt"
    p.write_text(
        "车轮\t汽车\n"                # tab 分隔·干净对
        "方向盘 BLAH 汽车\n"          # 中段噪声忽略·首 part(方向盘) 末 whole(汽车)
        "叶子 树\n",                  # 干净对
        encoding="utf-8",
    )
    assert load_mereology_facts_file(str(p)) == [
        ("车轮", "汽车"),
        ("方向盘", "汽车"),   # 中段噪声忽略·首=方向盘 末=汽车
        ("叶子", "树"),
    ]
    # resolve_mereology_facts 经 local_dir + lang 后缀读同文件
    assert resolve_mereology_facts(LANG_ZH, local_dir=str(tmp_path)) == [
        ("车轮", "汽车"), ("方向盘", "汽车"), ("叶子", "树"),
    ]


# ---- M7 resume 幂等（query_from skip） ----

def test_m7_resume_idempotent_no_duplicate_edges():
    """二次 boot（resume 路径·load_run 已还原边）→ query_from skip·零新增边（EdgeStore.add 不去重故须此守）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    pairs = [("车轮", "汽车"), ("手指", "手")]
    n1 = bootstrap_mereology_edges(ctx.concept_index, ctx.edge_store, pairs, space_id=sid)
    n2 = bootstrap_mereology_edges(ctx.concept_index, ctx.edge_store, pairs, space_id=sid)
    assert n1 == 2 and n2 == 0, "二次 boot 全 skip（query_from 幂等）·零新增"
    # 边总数仍是 2（无重复堆叠）
    assert ctx.backend.count("edge", where={"edge_type": EDGE_MEREOLOGY}) == 2


# ---- M8 cue 路由修正（REL_MEREOLOGY→MEREOLOGY_CUE·非 IS_A_CUE） ----

def test_m8_cue_routing_fix_mereology_not_isa():
    """REL_MEREOLOGY → MEREOLOGY_CUE（非 IS_A_CUE）·解首版误路由（部分-整体被建成 IsA 边=语义错）。"""
    assert _REL_KIND_TO_CUE_TYPE[REL_MEREOLOGY] == MEREOLOGY_CUE, \
        "REL_MEREOLOGY 路由到 MEREOLOGY_CUE（非 IS_A_CUE·解误路由）"
    assert MEREOLOGY_CUE != IS_A_CUE, "MEREOLOGY_CUE 独立 cue_type（≠IS_A_CUE）"
    assert MEREOLOGY_CUE == 8


def test_m8b_bufen_not_in_cue_words_frozenset_bit_identical():
    """部分 不入 _CUE_WORDS frozenset（首源不认）→ gate OFF cue_type_of(部分) 返 None·bit-identical。

    bit-identical 关键：部分 仅经 D:11 readback 第二源（gate EMERGENT_RELATION_CUE_READBACK_MODE ON）识别·
    gate OFF（CI/生产 default）首源 frozenset 不含 部分 → cue_type_of 返 None → 零行为变。
    """
    # 部分 不在任何 lang 的 _CUE_WORDS frozenset 值集中
    for lang_set in _CUE_WORDS.values():
        for cue_type, words in lang_set.items():
            assert "部分" not in words, f"部分 不入 _CUE_WORDS[{cue_type}]（首源不认·gate OFF bit-identical）"
            assert "part" not in words
    # gate OFF（默认）cue_type_of(部分) → None（首源不命中·第二源 gate-off）
    assert cue_type_of("部分", LANG_ZH) is None
    assert cue_type_of("part", LANG_ZH) is None


# ---- M9 PR/dag_path 排除（结构边·不接 reward·镜像 IS_A） ----

def test_m9_edge_mereology_registered_and_structural():
    """EDGE_MEREOLOGY=25 注册合法 + 不在 PR 头集 {PRECEDES,CAUSES,REFERS_TO}（结构边·不接 reward·镜像 IS_A）。"""
    # 注册合法（C9-bis 完备性检查 #1）
    assert is_registered_edge_type(EDGE_MEREOLOGY)
    assert EDGE_MEREOLOGY in REGISTERED_EDGE_TYPES
    assert EDGE_MEREOLOGY == 25
    # 不在 PR 头集（effective_weight:82 assert 只认 {PRECEDES,CAUSES,REFERS_TO}）
    pr_heads = {EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO}
    assert EDGE_MEREOLOGY not in pr_heads, "MEREOLOGY 不进 PR（结构边·镜像 IS_A）"
    assert EDGE_IS_A not in pr_heads, "对照：IS_A 同样不进 PR"
    # MEREOLOGY ≠ IS_A（语义正交·不同 edge_type）
    assert EDGE_MEREOLOGY != EDGE_IS_A
