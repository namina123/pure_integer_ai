"""P0 #1040 测试：generate-dispatch 主缺口修复（slot.ref 派发 token concept + ctx_refs token 级）。

承接 doc/重来_全局缺口重审_2026-07-14.md §1（主缺口 runtime 钉死：generate 产 121584 words·100% `__seg_*`
label·0% 真词·候选恒=1）。机制 = Path C 存储（def_array·repeat-safe·非 PRECEDES walk）：

  TC1 read_token_seq 存储 read：struct_ref def_array ref_space_id!=0 → token concept 序（order_index 序）+
     **repeat-safe**（重复 token "的"跨 position 共享 concept ref·每 position 一行→完整序·证伪 walk dedup bug）。
  TC2 generate gate ON 产真词：slot.ref=token → surface_of(token) → P0a 码点真字（"猫/吃/鱼"）。
     gate OFF 逐字现状：slot.ref=struct_ref → surface_of(struct_ref) → "__seg_*" label（主缺口复现）。
  TC3 bit-identical gate-OFF：OFF 路径 slot.ref=struct_ref + ctx=unit + token_refs 空 → 既有行为零回归。
  TC4 ctx token 级：gate ON produced_refs/prior_topic 含 token concept（非 unit）→ collide/sel_pref/pronoun
     同空间·解错节点（唯一消费方 slot_dispatch）。
  TC5 P0b alias 多候选 + target_lang：apple↔苹果 PURE_ALIAS + MARK_LANG → activate_candidates 多候选 →
     dispatch_slot 按 target_lang 选词形（ZH→苹果 / EN→apple·surface_of 产真字）。P0b 经 #1040 接通真活。

铁律：纯整数（refs/order_index/lang 全整）/ 确定性 bit-identical / 反 theater（机制真产真词·非 label 占位）。
全统计层（判据①编码接地 + ⑤跨语言汇聚）·非 truth/非 can_ween。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.node_store import NODE_WORD, NODE_CONCEPT
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.cognition.understanding.role_precedes import (
    build_struct_anchor, build_precedes_edges, attach_role_seq, attach_token_seq,
)
from pure_integer_ai.cognition.understanding.alias_bridge import bootstrap_alias_edges
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.generate import generate_output
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.types import (
    PathResult, PathData, LANG_NONE, LANG_ZH, LANG_EN,
)


@pytest.fixture(autouse=True)
def _gate_reset():
    """每测前后复位两 gate（守测试隔离·防跨测泄漏）。"""
    saved_disp = gates.DISPATCH_TOKEN_CHAIN_MODE
    saved_surf = gates.ORDINAL_SURFACE_MODE
    gates.DISPATCH_TOKEN_CHAIN_MODE = False
    gates.ORDINAL_SURFACE_MODE = False
    yield
    gates.DISPATCH_TOKEN_CHAIN_MODE = saved_disp
    gates.ORDINAL_SURFACE_MODE = saved_surf


def _build_seg(ctx, struct_label, token_surfaces, *, order_base=0, with_token_seq=True):
    """建一段：struct_ref(NODE_CONCEPT) + tokens(NODE_WORD) + PRECEDES 锚+序链 + role_seq + token_seq(def_array)。

    镜像 observe 语言段建边（build_struct_anchor + build_precedes_edges + attach_role_seq + attach_token_seq）。
    ensure 自动写 correspondence（P0a hook）。with_token_seq=False 模拟 gate OFF（observe 不写 token_seq）。"""
    sid = ctx.space_id
    struct_ref = ctx.concept_index.ensure(struct_label, space_id=sid,
                                          node_type=NODE_CONCEPT)
    tokens = [ctx.concept_index.ensure(t, space_id=sid, node_type=NODE_WORD)
              for t in token_surfaces]
    build_struct_anchor(ctx.edge_store, struct_ref, tokens[0],
                        source=SOURCE_BARE_TEXT, space_id=sid, order_base=order_base)
    build_precedes_edges(ctx.edge_store, tokens,
                         source=SOURCE_BARE_TEXT, space_id=sid, order_base=order_base)
    attach_role_seq(ctx.backend, struct_ref, list(range(len(tokens))),
                    order_base=order_base)
    if with_token_seq:
        attach_token_seq(ctx.backend, struct_ref, tokens, order_base=order_base)
    return struct_ref, tokens


def _path(struct_ref, *, sink=None):
    """最小 PathResult（topo_layers=[[struct_ref]] + struct_unit_refs=[struct_ref]）。
    sink=struct_ref 时验 reached_sink（reward=0 瀑布的根·G2p 读）。"""
    return PathResult(path=PathData(edges=[], struct_unit_refs=[struct_ref]),
                      topo_layers=[[struct_ref]], convergence={}, source=struct_ref,
                      sink=sink)


# ---- TC1 read_token_seq 存储 read（repeat-safe·证伪 walk dedup bug） ----

def test_tc1a_read_token_seq_returns_stored_tokens_in_order():
    """struct_ref def_array token_seq → [t0,t1,t2]（order_index 序）。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, tokens = _build_seg(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    g = ConceptGraph(b)
    assert g.read_token_seq(struct_ref) == tokens


def test_tc1b_read_token_seq_repeat_safe_repeated_token():
    """**repeat-safe（核心·证伪 walk bug）**：重复 token（"的"跨 position 共享 concept ref）→
    存储每 position 一行 → read 返完整 [的,猫,的,鱼]·非 walk dedup 漏 token。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, tokens = _build_seg(ctx, "__seg_0_0", ["的", "猫", "的", "鱼"])
    g = ConceptGraph(b)
    seq = g.read_token_seq(struct_ref)
    assert len(seq) == 4, "repeat-safe：4 position 4 token（的 出现两次同 concept ref 不 dedup）"
    assert seq[0] == seq[2], "两处'的'同 concept ref（content_hash dedup）"
    assert seq == tokens, "完整序 [的,猫,的,鱼]"


def test_tc1c_no_token_seq_returns_empty():
    """无 token_seq（gate OFF observe 不写 / code/arith）→ []。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, _tokens = _build_seg(ctx, "__seg_0_0", ["猫", "吃", "鱼"],
                                     with_token_seq=False)
    g = ConceptGraph(b)
    assert g.read_token_seq(struct_ref) == []


# ---- TC2 generate gate ON 产真词 / gate OFF 逐字现状 ----

def test_tc2a_generate_gate_on_produces_real_words():
    """gate ON：slot.ref=token → surface_of(token) → P0a 码点真字（"猫/吃/鱼"·非 __seg_*）。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, _tokens = _build_seg(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    g = ConceptGraph(b)
    gates.DISPATCH_TOKEN_CHAIN_MODE = True
    gates.ORDINAL_SURFACE_MODE = True
    out = generate_output(_path(struct_ref), g, WorkMemory(), LANG_NONE)
    assert out.parts[0].words == ["猫", "吃", "鱼"], "gate ON 产真词（主缺口修复）"


def test_tc2b_generate_gate_off_is_baseline_seg_label():
    """gate OFF：slot.ref=struct_ref → surface_of(struct_ref) → "__seg_*" label（主缺口复现·逐字现状）。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, _tokens = _build_seg(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    g = ConceptGraph(b)
    gates.ORDINAL_SURFACE_MODE = True   # surface ON 才显 __seg_* label（CI 两 gate OFF 产 #ref 占位）
    # DISPATCH_TOKEN_CHAIN_MODE OFF（fixture 默认）
    out = generate_output(_path(struct_ref), g, WorkMemory(), LANG_NONE)
    assert out.parts[0].words == ["__seg_0_0", "__seg_0_0", "__seg_0_0"], \
        "gate OFF 主缺口：struct_ref label 非 token 真词"


# ---- TC3 bit-identical gate-OFF（token_refs 空 + ctx unit·既有行为零回归） ----

def test_tc3_gate_off_token_refs_empty_and_ctx_unit():
    """gate OFF：OutputPart.token_refs 空（默认）+ produced_refs 含 unit（非 token）→ 既有行为零回归。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, _tokens = _build_seg(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    g = ConceptGraph(b)
    wm = WorkMemory()
    out = generate_output(_path(struct_ref), g, wm, LANG_NONE)
    assert out.parts[0].token_refs == [], "gate OFF token_refs 空（默认·bit-identical）"
    assert wm.produced_refs == [struct_ref], "gate OFF ctx=unit（逐字现状·非 token）"


# ---- TC4 ctx token 级（gate ON） ----

def test_tc4_gate_on_ctx_is_token_level():
    """gate ON：produced_refs 含 token concept（非 unit）→ collide/sel_pref/pronoun 同空间·解错节点。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, tokens = _build_seg(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    g = ConceptGraph(b)
    gates.DISPATCH_TOKEN_CHAIN_MODE = True
    gates.ORDINAL_SURFACE_MODE = True
    wm = WorkMemory()
    out = generate_output(_path(struct_ref), g, wm, LANG_NONE)
    assert out.parts[0].token_refs == tokens, "gate ON part 携 token concept 序"
    assert struct_ref not in wm.produced_refs, "gate ON ctx 不含 unit（token 级）"
    for t in tokens:
        assert t in wm.produced_refs, f"gate ON ctx 含 token {t}"


# ---- TC6 reached_sink（reward=0 瀑布的根·G2p 读·review-2 LOW-1 补） ----

def test_tc6_gate_on_reached_sink_true_when_sink_unit_produced():
    """gate ON：sink struct_ref 有 token_seq → 不被 `if not token_seq: continue` 跳过 → 进 produced →
    reached_sink=True（旧 walk 返 [] → 跳 sink → reached_sink=False → G2p veto reward=0·Path C 解）。"""
    b = DictBackend(); ctx = make_train_context(b)
    struct_ref, _tokens = _build_seg(ctx, "__seg_0_0", ["猫", "吃", "鱼"])
    g = ConceptGraph(b)
    gates.DISPATCH_TOKEN_CHAIN_MODE = True
    gates.ORDINAL_SURFACE_MODE = True
    out = generate_output(_path(struct_ref, sink=struct_ref), g, WorkMemory(), LANG_NONE)
    assert out.reached_sink is True, "gate ON sink(struct_ref) produced → reached_sink=True（reward>0 前置）"


# ---- TC5 P0b alias 多候选 + target_lang（P0b 经 #1040 接通真活） ----

def test_tc5_alias_multicandidate_target_lang_selects_wordform():
    """apple↔苹果 PURE_ALIAS + MARK_LANG → activate_candidates 多候选 → target_lang 选词形（ZH→苹果/EN→apple）。

    P0b alias 桥经 #1040 接通真活：slot.ref=token(apple concept) → activate_candidates 返 {apple,苹果} →
    dispatch_slot 按 target_lang 选。无 #1040 时 slot.ref=struct_ref → activate_candidates 触不到 word↔word PURE_ALIAS。
    """
    b = DictBackend(); ctx = make_train_context(b)
    sid = ctx.space_id
    bootstrap_alias_edges(ctx.concept_index, ctx.edge_store, ctx.backend,
                         [("apple", LANG_EN, "苹果", LANG_ZH)], space_id=sid)
    # 一段只含 apple token（alias 候选 = 苹果）
    struct_ref, tokens = _build_seg(ctx, "__seg_0_0", ["apple"])
    g = ConceptGraph(b)
    gates.DISPATCH_TOKEN_CHAIN_MODE = True
    gates.ORDINAL_SURFACE_MODE = True
    out_zh = generate_output(_path(struct_ref), g, WorkMemory(), LANG_ZH)
    assert out_zh.parts[0].words == ["苹果"], "ZH target 选苹果"
    out_en = generate_output(_path(struct_ref), ConceptGraph(b), WorkMemory(), LANG_EN)
    assert out_en.parts[0].words == ["apple"], "EN target 选 apple"
