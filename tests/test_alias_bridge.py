"""P0b 测试：跨语言/同义 PURE_ALIAS 桥（alias_bridge.bootstrap_alias_edges）+ activate_candidates 自包含 fix。

纠偏回合 round2 地基尾刀（doc/重来_设计偏离审查_2026-07-14.md C 偏离·plan velvet-juggling-garden）。
让 apple/苹果 合一到同一抽象概念（PURE_ALIAS 等价类）·生成按 target_lang 选对词形。

身份模型 = Model A（词形↔词形双向 PURE_ALIAS·铁律"永不合并节点"合规）·非 Model C（同 local_id·违铁律）。

覆盖：
  AB1 bootstrap round-trip（双向 PURE_ALIAS + MARK_LANG + correspondence）
  AB2 空 pairs 短路零副作用（return 0·绝不调 ensure/build·CI bit-identical 硬守）
  AB3 activate_candidates 自包含（PURE_ALIAS 在→{self,alias}·双向对称）
  AB4 bit-identical gate-OFF（无 PURE_ALIAS→[self]·OCCURRENCE 边不触发自包含·逐字现状）
  AB5 target_lang 机制（activate_candidates 双词形 + lang_of 区分 EN/ZH + surface_of 产真字）
  AB6 loader E5 graceful（缺文件→[]·错行/未知 lang/自环 skip·BOM）
  AB7 load_alias_facts_file 格式（4 段 parse·en/zh→LANG_* 映射）
  AB8 resume 幂等（query_from skip·二次 boot 零新增边·EdgeStore.add 不去重故须此守）

铁律：纯整数（local_id/lang/mark_kind）/ 确定性 bit-identical / 反 theater（机制真活·非死列表）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.cognition.shared.edge_types import EDGE_REFERS_TO
from pure_integer_ai.storage.edge_store import SUBTYPE_PURE_ALIAS, SUBTYPE_OCCURRENCE, EPI_STRUCTURED
from pure_integer_ai.storage.abstract_mark import get_mark, MARK_LANG
from pure_integer_ai.storage.concept_correspondence import (
    load_correspondence, CORR_ORDINAL,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.understanding.alias_bridge import bootstrap_alias_edges
from pure_integer_ai.crosscut.integer.unicode_codec import encode
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
from pure_integer_ai.experiments.collection import resolve_alias_facts, load_alias_facts_file


@pytest.fixture(autouse=True)
def _gate_off():
    """每测前后复位 ORDINAL_SURFACE_MODE（守测试隔离·防 AB5 toggle 跨测泄漏）。"""
    saved = gates.ORDINAL_SURFACE_MODE
    gates.ORDINAL_SURFACE_MODE = False
    yield
    gates.ORDINAL_SURFACE_MODE = saved


def _has_pure_alias(edge_store, a, b) -> bool:
    """a→b PURE_ALIAS 边是否存在（query_from a·查 to==b 且 subtype==PURE_ALIAS）。"""
    rows = edge_store.query_from(a[0], a[1], edge_type=EDGE_REFERS_TO)
    return any(r.get("space_id_to") == b[0] and r.get("local_id_to") == b[1]
               and r.get("subtype") == SUBTYPE_PURE_ALIAS for r in rows)


# ---- AB1 bootstrap round-trip ----

def test_ab1_bootstrap_roundtrip():
    """bootstrap_alias_edges 建 (apple,EN,苹果,ZH) → 双向 PURE_ALIAS + MARK_LANG + correspondence。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    n = bootstrap_alias_edges(ctx.concept_index, ctx.edge_store, ctx.backend,
                              [("apple", LANG_EN, "苹果", LANG_ZH)],
                              space_id=sid)
    assert n == 2, "双向 PURE_ALIAS = 2 边"
    apple = ctx.concept_index.ensure("apple", space_id=sid)   # dedup 命中既有 ref
    pingguo = ctx.concept_index.ensure("苹果", space_id=sid)
    assert apple != pingguo, "Model A：两不同 local_id 词形节点（非同 id·铁律合规）"
    # 双向 PURE_ALIAS
    assert _has_pure_alias(ctx.edge_store, apple, pingguo), "apple→苹果 PURE_ALIAS"
    assert _has_pure_alias(ctx.edge_store, pingguo, apple), "苹果→apple PURE_ALIAS（对称）"
    # MARK_LANG 各自 lang（dispatch_slot target_lang 偏好读 lang_of）
    assert get_mark(ctx.backend, ref=apple, mark_kind=MARK_LANG) == LANG_EN
    assert get_mark(ctx.backend, ref=pingguo, mark_kind=MARK_LANG) == LANG_ZH
    # correspondence（P0a hook·ensure 写码点）
    assert load_correspondence(ctx.backend, space_id=sid, local_id=apple[1],
                               corr_kind=CORR_ORDINAL) == encode("apple")
    assert load_correspondence(ctx.backend, space_id=sid, local_id=pingguo[1],
                               corr_kind=CORR_ORDINAL) == (33529, 26524)


# ---- AB2 空 pairs 短路零副作用 ----

def test_ab2_empty_pairs_short_circuit_zero_side_effects():
    """空 pairs → return 0·绝不调 ensure/set_mark/build（CI/生产 default 无文件 bit-identical 硬守）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    n = bootstrap_alias_edges(ctx.concept_index, ctx.edge_store, ctx.backend,
                              [], space_id=sid)
    assert n == 0
    # 短路在 ensure 之前·无节点建（lookup 任一 surface 皆 None）
    assert ctx.concept_index.lookup("apple", sid) is None, "短路不调 ensure·无节点"
    assert ctx.concept_index.lookup("苹果", sid) is None
    # 零 PURE_ALIAS 边
    assert ctx.backend.count("edge", where={"edge_type": EDGE_REFERS_TO}) == 0


# ---- AB3 activate_candidates 自包含 ----

def test_ab3_activate_candidates_self_inclusion():
    """PURE_ALIAS 边在 → activate_candidates 返 {self, alias}（双向对称）·self 补回。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    bootstrap_alias_edges(ctx.concept_index, ctx.edge_store, ctx.backend,
                          [("apple", LANG_EN, "苹果", LANG_ZH)], space_id=sid)
    apple = ctx.concept_index.ensure("apple", space_id=sid)
    pingguo = ctx.concept_index.ensure("苹果", space_id=sid)
    g = ConceptGraph(b)
    # 双向对称：两端 activate_candidates 都得 {self, 对方}
    assert g.activate_candidates(apple) == sorted([apple, pingguo])
    assert g.activate_candidates(pingguo) == sorted([apple, pingguo])


# ---- AB4 bit-identical gate-OFF（逐字现状） ----

def test_ab4_bit_identical_no_pure_alias_is_baseline():
    """无 PURE_ALIAS 边 → activate_candidates 退化 `if not cands` 现状：lone 词返 [self]。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    lone = ctx.concept_index.ensure("loneword", space_id=sid)
    g = ConceptGraph(b)
    # 无 REFERS_TO 候选 → [self] fallback（逐字现状·bit-identical）
    assert g.activate_candidates(lone) == [lone]


def test_ab4b_occurrence_edge_does_not_trigger_self_inclusion():
    """OCCURRENCE REFERS_TO 边（代词性质B·subtype≠PURE_ALIAS）→ 不触发自包含·逐字现状。

    bit-identical 关键：PURE_ALIAS gate 精准·OCCURRENCE 边不误触发 self 补回。
    """
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    target = ctx.concept_index.ensure("target", space_id=sid)
    other = ctx.concept_index.ensure("other", space_id=sid)
    # 手建一条 OCCURRENCE REFERS_TO 边 other→target（非 PURE_ALIAS·镜像代词性质B）
    ctx.edge_store.add(space_id_from=other[0], local_id_from=other[1],
                       space_id_to=target[0], local_id_to=target[1],
                       edge_type=EDGE_REFERS_TO, subtype=SUBTYPE_OCCURRENCE,
                       source=6, epistemic_origin=EPI_STRUCTURED)
    g = ConceptGraph(b)
    # 有 OCCURRENCE 候选但非 PURE_ALIAS → self 不补回 → 只返 {other}（逐字现状·bit-identical）
    cands = g.activate_candidates(target)
    assert cands == [other], "OCCURRENCE 边不触发自包含·逐字现状"
    assert target not in cands


def test_ab4c_concept_node_excludes_self_under_pure_alias():
    """NODE_CONCEPT（抽象概念）+ PURE_ALIAS 词形候选 → self 排除（词形代表概念·承重既有语义）。

    回归守卫：node_type 门控精准——从抽象概念派发时（test_stage5 C / test_hub_exclude C / test_m5 u）
    concept 自身 excluded·其词形候选优先。仅 NODE_WORD（词形·observed token）派发时才 include self。
    """
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    # C = NODE_CONCEPT（ensure 默认 node_type=1）·WZH/WEN 词形 PURE_ALIAS→C（既有 concept+词形 模型）
    C = ctx.concept_index.ensure("concept_c", space_id=sid)
    WZH = ctx.concept_index.ensure("zhword", space_id=sid)
    WEN = ctx.concept_index.ensure("enword", space_id=sid)
    ctx.edge_store.add(space_id_from=WZH[0], local_id_from=WZH[1],
                       space_id_to=C[0], local_id_to=C[1],
                       edge_type=EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS,
                       source=6, epistemic_origin=EPI_STRUCTURED)
    ctx.edge_store.add(space_id_from=WEN[0], local_id_from=WEN[1],
                       space_id_to=C[0], local_id_to=C[1],
                       edge_type=EDGE_REFERS_TO, subtype=SUBTYPE_PURE_ALIAS,
                       source=6, epistemic_origin=EPI_STRUCTURED)
    g = ConceptGraph(b)
    # C 是 NODE_CONCEPT → self 排除 → 只返词形 {WEN, WZH}（C 不在候选·词形优先·既有语义不变）
    cands = g.activate_candidates(C)
    assert cands == sorted([WEN, WZH]), "NODE_CONCEPT 派发 self 排除·词形候选优先"
    assert C not in cands


# ---- AB5 target_lang 机制 ----

def test_ab5_target_lang_mechanism():
    """activate_candidates 返双词形 + lang_of 区分 EN/ZH + surface_of 产真字（dispatch_slot 依此选）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    bootstrap_alias_edges(ctx.concept_index, ctx.edge_store, ctx.backend,
                          [("apple", LANG_EN, "苹果", LANG_ZH)], space_id=sid)
    apple = ctx.concept_index.ensure("apple", space_id=sid)
    pingguo = ctx.concept_index.ensure("苹果", space_id=sid)
    g = ConceptGraph(b)
    # lang_of 区分两词形（dispatch_slot target_lang 过滤的输入）
    assert g.lang_of(apple) == LANG_EN
    assert g.lang_of(pingguo) == LANG_ZH
    # surface_of（gate ON）产真字——dispatch_slot 选 apple 时产 "apple"·选苹果时产 "苹果"
    gates.ORDINAL_SURFACE_MODE = True
    g2 = ConceptGraph(b)
    assert g2.surface_of(apple) == "apple"
    assert g2.surface_of(pingguo) == "苹果"
    gates.ORDINAL_SURFACE_MODE = False


# ---- AB6 loader E5 graceful ----

def test_ab6_resolve_alias_facts_missing_file_returns_empty(tmp_path):
    """缺文件（无 PURE_INTEGER_AI_LOCAL_DIR / 文件不存在）→ []（E5 graceful·CI bit-identical 守）。"""
    # 无 local_dir → []
    assert resolve_alias_facts(local_dir=None) == []
    # local_dir 存在但无 alias_facts.txt → []
    assert resolve_alias_facts(local_dir=str(tmp_path)) == []


def test_ab6_load_alias_facts_graceful_skips_bad_lines(tmp_path):
    """错行（≠4 段）/ 未知 lang 码 / 自环 / 注释 / 空行 skip + 不抛崩（E5 graceful）。"""
    p = tmp_path / "alias_facts.txt"
    p.write_text(
        "# 注释行\n"
        "\n"                          # 空行
        "apple en 苹果 zh\n"          # 合法（tab/空格混合 split）
        "only three segments\n"       # 3 段·错行 skip
        "cat en 猫 jp\n"              # 未知 lang 码 jp → skip
        "dup zh dup zh\n"             # 自环（同 surface 同 lang）skip
        "x en y en\n",                # 合法
        encoding="utf-8",
    )
    pairs = load_alias_facts_file(str(p))
    assert pairs == [("apple", LANG_EN, "苹果", LANG_ZH),
                     ("x", LANG_EN, "y", LANG_EN)]


def test_ab6_load_alias_facts_bom(tmp_path):
    """BOM（utf-8-sig）首行不破识别（对抗审点·镜像 load_is_a_facts_file）。"""
    p = tmp_path / "alias_facts.txt"
    p.write_bytes("﻿apple en 苹果 zh\n".encode("utf-8"))
    assert load_alias_facts_file(str(p)) == [("apple", LANG_EN, "苹果", LANG_ZH)]


# ---- AB7 load_alias_facts_file 格式 ----

def test_ab7_load_alias_facts_format_and_lang_mapping(tmp_path):
    """4 段 parse + en/zh→LANG_* 映射（types.py:35-36 LANG_ZH=1/LANG_EN=2）。"""
    p = tmp_path / "alias_facts.txt"
    p.write_text(
        "apple\ten\t苹果\tzh\n"
        "李白\tzh\t李太白\tzh\n",
        encoding="utf-8",
    )
    assert load_alias_facts_file(str(p)) == [
        ("apple", LANG_EN, "苹果", LANG_ZH),
        ("李白", LANG_ZH, "李太白", LANG_ZH),
    ]
    # resolve_alias_facts 经 local_dir 读同文件
    assert resolve_alias_facts(local_dir=str(tmp_path)) == [
        ("apple", LANG_EN, "苹果", LANG_ZH),
        ("李白", LANG_ZH, "李太白", LANG_ZH),
    ]


# ---- AB8 resume 幂等（query_from skip） ----

def test_ab8_resume_idempotent_no_duplicate_edges():
    """二次 boot（resume 路径·load_run 已还原边）→ query_from skip·零新增边（EdgeStore.add 不去重故须此守）。"""
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    pairs = [("apple", LANG_EN, "苹果", LANG_ZH)]
    n1 = bootstrap_alias_edges(ctx.concept_index, ctx.edge_store, ctx.backend,
                               pairs, space_id=sid)
    n2 = bootstrap_alias_edges(ctx.concept_index, ctx.edge_store, ctx.backend,
                               pairs, space_id=sid)
    assert n1 == 2 and n2 == 0, "二次 boot 全 skip（双向 query_from 幂等）·零新增"
    # 边总数仍是 2（无重复堆叠）
    apple = ctx.concept_index.ensure("apple", space_id=sid)
    pingguo = ctx.concept_index.ensure("苹果", space_id=sid)
    assert _has_pure_alias(ctx.edge_store, apple, pingguo)
    assert _has_pure_alias(ctx.edge_store, pingguo, apple)
    # REFERS_TO 边总数 = 2（无重复）
    assert ctx.backend.count("edge", where={"edge_type": EDGE_REFERS_TO}) == 2
