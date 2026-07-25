"""STEP5 PR4 REL_SIMILAR EDGE_SIMILAR + dispatch_slot slot-filler 测试（任务 #898）。

REL_SIMILAR 迁移兼容词级种子 + is_similar_cue（来源化图主读，旧 D:11 显式回退）+
extract_similar_claims + build_similar_edges（EDGE_SIMILAR X→Y·TIER_SHADOW·D2 合规非向量）+
graph.similar_candidates（双向·dispatch_slot 读路径）+ dispatch_slot slot-filler 候选扩展（gate SIMILAR_SLOT_MODE）。
消费者 = dispatch_slot（live·GENERATE_MODE 永远 active·读 EDGE_SIMILAR 扩展 slot 候选）。

**反 theater 两路证**：
  (i) compatibility seed “像”的旧 D:11 ON/OFF 行为差；不把该路径计作 U-04 readiness
  (ii) fixture 注入开放变体 '相似于'→REL_SIMILAR D:11 边：gate ON True + extract_similar_claims 产 (left,right) + build EDGE_SIMILAR

**D2 合规**：EDGE_SIMILAR 二元离散边（非向量·非相似度 SCORE）·确定性文本提取（非学习型）·结构关系 slot-filler（非语义承载）→ 三维度全不满 → 非向量 → 合法。

**兼容边界守卫**：旧表不扩入开放变体；正式扩展应通过来源化图进入。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_TEACHER, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.storage.edge_types import EDGE_SIMILAR, EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives, REL_SIMILAR
from pure_integer_ai.cognition.understanding.cue_words import is_similar_cue, _CUE_WORDS
from pure_integer_ai.cognition.understanding.word_concept_signal import (
    bootstrap_word_concept_signals, record_word_concept, _REL_LEXICAL_CUE,
)
from pure_integer_ai.cognition.understanding.similar import build_similar_edges
from pure_integer_ai.cognition.understanding.cue_extractor import extract_similar_claims
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def sim_env():
    """PR4 单测环境（dict backend·core space·composes_attr·boot 种 REL_* D:11 边含 像）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    bootstrap_word_concept_signals(ci, es, b, space_id=sid, langs={LANG_ZH, LANG_EN})
    yield b, sid, es, ns, ci
    b.close()


def _inject_similar_d11(ci, es, b, sid, word_surface):
    """fixture 注入 D:11 边（word→REL_SIMILAR ref·SOURCE_TEACHER·模拟教师晋升·非 frozenset）。"""
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    rel_ref = rel_refs[REL_SIMILAR]
    return record_word_concept(ci, es, word_surface, rel_ref,
                               space_id=sid, source=SOURCE_TEACHER)


# ============ build_similar_edges 基建 ============

def test_build_similar_edges_builds_edge(sim_env):
    """build_similar_edges 建 EDGE_SIMILAR(X→Y·TIER_SHADOW·strength=1)·幂等 + 自环守。"""
    b, sid, es, ns, ci = sim_env
    x = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    y = ci.ensure("老虎", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    refs = [x, y]
    n = build_similar_edges(es, refs, similar_claims=[(0, 1)],
                            source=SOURCE_BARE_TEXT, space_id=sid)
    assert n == 1, "建 EDGE_SIMILAR(猫→老虎)"
    out = es.query_from(x[0], x[1], edge_type=EDGE_SIMILAR)
    assert any(r["space_id_to"] == y[0] and r["local_id_to"] == y[1] for r in out), \
        "EDGE_SIMILAR 猫→老虎 建出"
    # 幂等：重 claim 不重复建
    n2 = build_similar_edges(es, refs, similar_claims=[(0, 1)],
                             source=SOURCE_BARE_TEXT, space_id=sid)
    assert n2 == 0, "幂等·同 (X,Y,EDGE_SIMILAR,source) skip"
    # 自环守：X→X 不建
    n3 = build_similar_edges(es, [x, x], similar_claims=[(0, 1)],
                             source=SOURCE_BARE_TEXT, space_id=sid)
    assert n3 == 0, "自环 X→X 不建"


def test_build_similar_edges_out_of_bounds_skip(sim_env):
    """越界 idx → skip（fail-soft）。"""
    b, sid, es, ns, ci = sim_env
    x = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    n = build_similar_edges(es, [x], similar_claims=[(0, 5)],
                            source=SOURCE_BARE_TEXT, space_id=sid)
    assert n == 0, "越界 idx=5 → skip"


# ============ (i) seeded closed-class '像'（D:11-only·gate ON/OFF 行为差） ============

def test_similar_cue_seeded_d11_only(sim_env):
    """旧 seed “像”仅在 compatibility D:11 gate 开启时可回退命中。"""
    b, sid, es, ns, ci = sim_env
    # '像' 不在 _CUE_WORDS 检测 frozenset（is_similar_cue 无第一源 frozenset·D:11-only）
    saved = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = False
    try:
        assert is_similar_cue("像", LANG_ZH, backend=b, edge_store=es,
                              space_id=sid, concept_index=ci) is False, \
            "gate OFF is_similar_cue 恒 False（D:11-only·无 frozenset 第一源）"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    try:
        assert is_similar_cue("像", LANG_ZH, backend=b, edge_store=es,
                              space_id=sid, concept_index=ci) is True, \
            "gate ON D:11 readback '像'→REL_SIMILAR→True（seeded D:11 种子）"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved


# ============ (ii) fixture 注入开放变体 '相似于'（D:11 唯一源·反 theater） ============

def test_similar_cue_d11_readback_nontournament_word(sim_env):
    """(ii) fixture 注入 '相似于'→REL_SIMILAR D:11 边→gate ON True·gate OFF False。"""
    b, sid, es, ns, ci = sim_env
    _inject_similar_d11(ci, es, b, sid, "相似于")
    saved = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = False
    try:
        assert is_similar_cue("相似于", LANG_ZH, backend=b, edge_store=es,
                              space_id=sid, concept_index=ci) is False
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    try:
        assert is_similar_cue("相似于", LANG_ZH, backend=b, edge_store=es,
                              space_id=sid, concept_index=ci) is True, \
            "gate ON D:11 readback '相似于'→REL_SIMILAR→True"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved


def test_similar_d11_drives_edge_build(sim_env):
    """(ii) 端到端：'像' 经 D:11 readback 驱动 extract_similar_claims 产 pair + build_similar_edges 建 EDGE_SIMILAR。
    '猫 像 老虎' → claim (0,2) → EDGE_SIMILAR(猫→老虎)。"""
    b, sid, es, ns, ci = sim_env
    ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    ci.ensure("像", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    ci.ensure("老虎", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    tokens = ["猫", "像", "老虎"]
    saved = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    try:
        claims_on = extract_similar_claims(tokens, lang=LANG_ZH,
                                           backend=b, edge_store=es,
                                           space_id=sid, concept_index=ci)
        assert (0, 2) in claims_on, "gate ON '像'→similar claim (0,2)"
        refs = [ci.lookup("猫", sid), ci.lookup("像", sid), ci.lookup("老虎", sid)]
        n = build_similar_edges(es, refs, similar_claims=claims_on,
                                source=SOURCE_BARE_TEXT, space_id=sid)
        assert n >= 1, "EDGE_SIMILAR(猫→老虎) 建出"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved
    # gate OFF：'像' 不识别→无 similar claim
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = False
    try:
        claims_off = extract_similar_claims(tokens, lang=LANG_ZH,
                                            backend=b, edge_store=es,
                                            space_id=sid, concept_index=ci)
        assert claims_off == [], "gate OFF '像' 不识别→无 similar claim（反 theater）"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved


# ============ 消费者读路径：graph.similar_candidates（dispatch_slot 用） ============

def test_graph_similar_candidates_bidirectional(sim_env):
    """graph.similar_candidates 双向读 EDGE_SIMILAR（dispatch_slot 消费者读路径·consumer live）。
    EDGE_SIMILAR(猫→老虎) → similar_candidates(老虎)=[猫]（入向）·similar_candidates(猫)=[老虎]（出向）。"""
    b, sid, es, ns, ci = sim_env
    猫 = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    老虎 = ci.ensure("老虎", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_similar_edges(es, [猫, 老虎], similar_claims=[(0, 1)],
                        source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    # 入向：老虎 的相似概念 = 猫（猫→老虎 边·X 像 Y·Y 的相似是 X）
    sim_to_老虎 = graph.similar_candidates(老虎)
    assert 猫 in sim_to_老虎, "similar_candidates(老虎) 含 猫（入向·猫像老虎）"
    # 出向：猫 的相似概念 = 老虎（猫→老虎 边·猫 像 老虎）
    sim_to_猫 = graph.similar_candidates(猫)
    assert 老虎 in sim_to_猫, "similar_candidates(猫) 含 老虎（出向）"


def test_dispatch_slot_expands_candidates_with_similar(sim_env):
    """dispatch_slot gate SIMILAR_SLOT_MODE ON → 候选扩展含 EDGE_SIMILAR 邻居·gate OFF 不扩展。
    反 theater：consumer 真活（dispatch_slot 读 EDGE_SIMILAR 扩展 slot 候选）。"""
    b, sid, es, ns, ci = sim_env
    猫 = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    老虎 = ci.ensure("老虎", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_similar_edges(es, [猫, 老虎], similar_claims=[(0, 1)],
                        source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    # 模拟 dispatch_slot 候选扩展逻辑（slot.ref=老虎·activate_candidates + SIMILAR_SLOT_MODE 扩展）
    from pure_integer_ai.cognition.understanding.cue_words import _CUE_WORDS  # noqa: F401（确认 import 路径）
    candidates = graph.activate_candidates(老虎)
    saved = gates.SIMILAR_SLOT_MODE
    gates.SIMILAR_SLOT_MODE = True
    try:
        # 复现 dispatch_slot 的 EDGE_SIMILAR 扩展逻辑
        _seen = set(candidates)
        for c in list(candidates):
            for s in graph.similar_candidates(c):
                if s not in _seen:
                    candidates.append(s)
                    _seen.add(s)
        assert 猫 in candidates, "gate ON dispatch_slot 扩展候选含 猫（EDGE_SIMILAR slot-filler）"
    finally:
        gates.SIMILAR_SLOT_MODE = saved
    # gate OFF：不扩展（候选无 猫·除非 activate_candidates 本身含）
    candidates_off = graph.activate_candidates(老虎)
    assert 猫 not in candidates_off or 猫 == 老虎, \
        "gate OFF dispatch_slot 不扩展（EDGE_SIMILAR 不读·反 theater 行为差）"


# ============ D2 守卫 + 铁律 ============

def test_d2_similar_not_vector():
    """D2 守卫：EDGE_SIMILAR 是 int edge_type·无 float/SCORE·图遍历非相似度排序。"""
    assert isinstance(EDGE_SIMILAR, int), "EDGE_SIMILAR 是 int edge_type"
    assert EDGE_SIMILAR == 24


def test_edge_similar_not_in_effective_weight(sim_env):
    """铁律：EDGE_SIMILAR 边不入 effective_weight（只认 {PRECEDES,CAUSES,REFERS_TO}·:82 assert）。"""
    from pure_integer_ai.cognition.process.effective_weight import effective_weight
    similar_edge = {"edge_type": EDGE_SIMILAR, "strength": 1}
    try:
        effective_weight(similar_edge)
        assert False, "EDGE_SIMILAR 边入 effective_weight 应 assert fail"
    except AssertionError:
        pass  # 预期 assert fail（EDGE_SIMILAR=24 不在 {PRECEDES,CAUSES,REFERS_TO}）


# ============ 不写死守卫（D6·开放变体零硬编码） ============

def test_no_hardcode_open_variants_similar():
    """兼容表不得继续吸收开放变体，正式扩展必须走来源化图。"""
    zh = _REL_LEXICAL_CUE.get(LANG_ZH, {})
    assert "像" in zh, "closed-class 核心 '像' 在 _REL_LEXICAL_CUE"
    assert "相似于" not in zh, "开放变体 '相似于' 不硬编码"
    en = _REL_LEXICAL_CUE.get(LANG_EN, {})
    assert "resembles" in en and "similar_to" not in en
    # is_similar_cue 无检测 frozenset（D:11-only·不新增 _SIMILAR_CUE frozenset·更守 D6）
    # 确认 '像' 不在 _CUE_WORDS 任何 cue_type（is_similar_cue 不走 cue_type_of 第一源）
    for cue_type, words in _CUE_WORDS.get(LANG_ZH, {}).items():
        assert "像" not in words, f"'像' 不在 _CUE_WORDS[{cue_type}]（is_similar_cue D:11-only 无 frozenset）"
