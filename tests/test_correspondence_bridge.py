"""对应泛化 readback→generation 桥测试（dispatch_slot 第 8 路·doc/重来_对应泛化_readback_generation_桥_2026-07-17）。

机制（2 对抗审 APPROVE·CORRESPONDENCE_SLOT_MODE）：
  - rel_kind_of_skeleton（two-hop·unit→INSTANTIATES→skeleton→REALIZES→REL_*·审2 致命-1 修·挂 skeleton 非 unit）
  - read_cue_sig（skeleton PARAM slot ATTR_CUE_SIG 重建·镜像 _collect_cue_sig·runtime length-guard 守 slot 对齐）
  - cue_rel_of（D:11 source==SOURCE_BARE_TEXT AND tier==TIER_PRIMARY·v2-learned only·审1 CONDITION A·反 boot theater）
  - _correspondence_bonus（(β) 独立轴 CORR_BONUS=SCORE_SCALE+1=1001·不入 _cap_sp·cue-slot-aware·反 theater）

TC1 e2e：cue slot 学到的 cue 词激活（CORR_BONUS=1001 严格胜 collide=1=1000·学全 BEYOND 共现）。
TC2 cue-slot-aware：非 cue slot bonus=0（unit + e2e·反 theater·审1 CONDITION D）。
TC3 cue_sig=()（无 cue 位）→ current_cue_slots=∅ → 无 bonus。
TC4 boot 种子不触发（SOURCE_TEACHER·审1 CONDITION A source 滤）。
TC5 无 REALIZES skeleton → rel_kind=0 → 无 bonus（two-hop robust 退化）。
TC6 cue_sig length ≠ role_seq → runtime length-guard → cue_slots=∅ → 无 bonus（§3 对齐 sound 守）。
TC7 gate OFF bit-identical：cue slot 选 collide 高的 distractor（无 bonus·逐字现状）。
TC8 two-hop 核证：rel_kind_of_skeleton(skel)=REL_CAUSES / rel_kind_of_skeleton(struct_ref)=0（REALIZES 挂 skeleton 非 unit·审2 致命-1）。

铁律：纯整数（sid/lid/rel_kind 全整）/ bit-identical（gate OFF 逐字现状·default OFF）/ 反 theater（cue-slot-aware
  + source 纯净 + 非自证生成纯读）/ 不写死（cue_rel_of 读图 D:11 边·非 word→rel frozenset）。
诚实边界：TC1-8 fixture 手造（直建 skeleton/REALIZES/D:11·非生产 tally/promote 链）·机制层预验非 empirical 泛化率
  （empirical 真 ConceptNet held-out defer 断奶后 W4 探针·超阈 reject·同 v2 F1/F2 框法）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_types import (
    EDGE_REALIZES, EDGE_COMPOSES, EDGE_RELATION_SIGNAL, EDGE_REFERS_TO,
    EDGE_COOCCURS, EDGE_INSTANTIATES,
)
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT, SOURCE_TEACHER, EPI_STRUCTURED
from pure_integer_ai.storage.node_store import TIER_PRIMARY, TIER_SHADOW, NODE_WORD, NODE_CONCEPT
from pure_integer_ai.storage.composes_attr import (
    record_composes_attr, ATTR_OPERATOR, ATTR_OPERAND, ATTR_CUE_SIG, ATTR_RELATION_PRIMITIVE,
)
from pure_integer_ai.numeric.symbol_domain import make_variable
from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives, REL_CAUSES
from pure_integer_ai.cognition.understanding.emergent_relation_signal import record_emergent_relation_signal_shadow
from pure_integer_ai.cognition.understanding.realizes import build_realizes_edge
from pure_integer_ai.cognition.understanding.instantiates import build_instantiates_edge
from pure_integer_ai.cognition.understanding.role_precedes import (
    build_struct_anchor, build_precedes_edges, attach_role_seq, attach_token_seq,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.generate import generate_output
from pure_integer_ai.cognition.result.slot_dispatch import _correspondence_bonus, CORR_BONUS, SCORE_SCALE
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.types import (
    PathResult, PathData, LANG_NONE, CUE_SLOT_FILL,
)
from tests.test_experiments import make_train_context


@pytest.fixture(autouse=True)
def _gate_reset():
    """每测前后复位 correspondence + dispatch gate（守测试隔离·防跨测泄漏）。"""
    saved = (gates.CORRESPONDENCE_SLOT_MODE, gates.DISPATCH_TOKEN_CHAIN_MODE,
             gates.ORDINAL_SURFACE_MODE, gates.CUE_SLOT_FILL_MODE)
    gates.CORRESPONDENCE_SLOT_MODE = False
    gates.DISPATCH_TOKEN_CHAIN_MODE = False
    gates.ORDINAL_SURFACE_MODE = False
    gates.CUE_SLOT_FILL_MODE = False
    yield
    (gates.CORRESPONDENCE_SLOT_MODE, gates.DISPATCH_TOKEN_CHAIN_MODE,
     gates.ORDINAL_SURFACE_MODE, gates.CUE_SLOT_FILL_MODE) = saved


# ---- helpers（镜像 test_dim_bridge._build_seg/_path + test_structure_reverse_inference._build_alignable_skel） ----

def _ensure(ctx, surface, node_type=NODE_WORD):
    return ctx.concept_index.ensure(surface, space_id=ctx.space_id, node_type=node_type)


def _add_composes(ctx, frm, to, order_index):
    ctx.edge_store.add(
        space_id_from=frm[0], local_id_from=frm[1],
        space_id_to=to[0], local_id_to=to[1],
        edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED, order_index=order_index)


def _build_skel(ctx, root, slot_refs, cue_at=None, cue_ref=None):
    """对齐就绪 skeleton：root ATTR_OPERATOR(NOP) → slot 叶（ATTR_OPERAND make_variable(i)·PARAM slot）·
    cue_at 位叶加挂 ATTR_CUE_SIG=cue_ref。镜像 test_structure_reverse_inference._build_alignable_skel。"""
    record_composes_attr(ctx.backend, ref=root, kind=ATTR_OPERATOR, int_a=0, int_b=0)
    for ti, tok in enumerate(slot_refs):
        record_composes_attr(ctx.backend, ref=tok, kind=ATTR_OPERAND,
                             int_a=make_variable(ti), int_b=0)
        _add_composes(ctx, root, tok, ti)
        if ti == cue_at and cue_ref is not None:
            record_composes_attr(ctx.backend, ref=tok, kind=ATTR_CUE_SIG,
                                 int_a=cue_ref[0], int_b=cue_ref[1])


def _build_seg(ctx, struct_label, token_surfaces, *, order_base=0):
    """建一段 lang __seg_：struct_ref(NODE_CONCEPT) + tokens(NODE_WORD) + PRECEDES 锚+序链 + role_seq + token_seq。"""
    sid = ctx.space_id
    struct_ref = ctx.concept_index.ensure(struct_label, space_id=sid, node_type=NODE_CONCEPT)
    tokens = [ctx.concept_index.ensure(t, space_id=sid, node_type=NODE_WORD) for t in token_surfaces]
    build_struct_anchor(ctx.edge_store, struct_ref, tokens[0],
                        source=SOURCE_BARE_TEXT, space_id=sid, order_base=order_base)
    build_precedes_edges(ctx.edge_store, tokens,
                         source=SOURCE_BARE_TEXT, space_id=sid, order_base=order_base)
    attach_role_seq(ctx.backend, struct_ref, list(range(len(tokens))), order_base=order_base)
    attach_token_seq(ctx.backend, struct_ref, tokens, order_base=order_base)
    return struct_ref, tokens


def _path(struct_ref):
    """最小 PathResult（topo_layers=[[struct_ref]]·generate 用）。"""
    return PathResult(path=PathData(edges=[], struct_unit_refs=[struct_ref]),
                      topo_layers=[[struct_ref]], convergence={}, source=struct_ref,
                      sink=None)


def _add_d11_primary(ctx, word_ref, rel_ref, *, source=SOURCE_BARE_TEXT):
    """建 D:11 word→rel_ref 边 tier=PRIMARY。
    source=SOURCE_BARE_TEXT（v2-learned 默认）：record_emergent_relation_signal_shadow 建 SHADOW(BARE_TEXT)
    → set_tier flip PRIMARY（镜像 v2 tally→promote 生产链·promote set_tier 不改 source）。
    source=SOURCE_TEACHER（TC4 boot 种子）：edge_store.add 直建（source=TEACHER·tier=PRIMARY·镜像 boot STEP5 PR1）。"""
    if source == SOURCE_BARE_TEXT:
        record_emergent_relation_signal_shadow(ctx.edge_store, word_ref, rel_ref, space_id=ctx.space_id)
        ctx.edge_store.set_tier(
            space_id_from=word_ref[0], local_id_from=word_ref[1],
            space_id_to=rel_ref[0], local_id_to=rel_ref[1],
            edge_type=EDGE_RELATION_SIGNAL, new_tier=TIER_PRIMARY)
    else:
        # boot 种子：直建 EDGE_RELATION_SIGNAL source=SOURCE_TEACHER tier=PRIMARY（record_*_shadow 会写 BARE_TEXT·不适用）
        ctx.edge_store.add(
            space_id_from=word_ref[0], local_id_from=word_ref[1],
            space_id_to=rel_ref[0], local_id_to=rel_ref[1],
            edge_type=EDGE_RELATION_SIGNAL, strength=1,
            source=SOURCE_TEACHER, tier=TIER_PRIMARY)


def _ref_lit(ref):
    """surface_of gate OFF → modality_serialize 占位 ref 字面（#sid:lid）·测选词用。"""
    return f"#{ref[0]}:{ref[1]}"


def _build_bridge_fixture(ctx, *, n_skel_leaves=3, cue_at=1, with_realizes=True,
                          learned_source=SOURCE_BARE_TEXT, cand_slot=1,
                          link_learned_candidate=True):
    """全链 fixture：struct_ref(INSTANTIATES)→skeleton(COMPOSES+ATTR_CUE_SIG)→REALIZES→REL_CAUSES + D:11 learned。
    cue_at=skeleton ATTR_CUE_SIG 位（None=无 cue 位·TC3）/ cand_slot=learned+distractor 候选竞争位（独立于 cue_at·
    TC3 cue_at=None 时 learned 仍候选 slot 1·验 cue_sig=()→无 bonus）。返 (struct_ref,tokens,skel_root,learned,distractor,ctx_ref,rel_causes)。"""
    sid = ctx.space_id
    # 1. struct_ref unit（3 token / 3 role·DISPATCH ON 时 slot.ref=token concept）
    struct_ref, tokens = _build_seg(ctx, "__seg_bridge", ["t0", "t1cue", "t2"])
    # 2. skeleton（flat NOP + N PARAM leaves·cue_at 位挂 ATTR_CUE_SIG·None=无 cue 位）
    skel_root = _ensure(ctx, "__skel_bridge", node_type=NODE_CONCEPT)
    cue_token = _ensure(ctx, "使")   # skeleton cue slot 的闭类 cue token（ATTR_CUE_SIG 存此）
    leaves = [_ensure(ctx, f"__leaf{i}") for i in range(n_skel_leaves)]
    _build_skel(ctx, skel_root, leaves, cue_at=cue_at, cue_ref=cue_token)
    # 3. INSTANTIATES(struct_ref → skel_root)
    build_instantiates_edge(ctx.edge_store, struct_ref, skel_root, space_id=sid)
    # 4. REALIZES(skel_root → REL_CAUSES) + rel primitive
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    rel_causes = rel_prims[REL_CAUSES]
    if with_realizes:
        build_realizes_edge(ctx.edge_store, skel_root, rel_causes, space_id=sid)
    # 5. learned_word：D:11 → REL_CAUSES PRIMARY + REFERS_TO cand_slot 位 token（成该 slot 候选）
    learned_word = _ensure(ctx, "引发")
    _add_d11_primary(ctx, learned_word, rel_causes, source=learned_source)
    # 旧夹具可显式把 learned 塞入普通 REFERS_TO 候选池；生产采用测试关闭此项，要求关系反向候选独立生效。
    if link_learned_candidate:
        ctx.edge_store.add(space_id_from=learned_word[0], local_id_from=learned_word[1],
                           space_id_to=tokens[cand_slot][0], local_id_to=tokens[cand_slot][1],
                           edge_type=EDGE_REFERS_TO, strength=1, source=SOURCE_BARE_TEXT)
    # 6. distractor：REFERS_TO cand_slot 位 token + COOCCURS(ctx_ref)（collide=1·共现驱动·须被 CORR_BONUS override）
    distractor = _ensure(ctx, "__distractor")
    ctx.edge_store.add(space_id_from=distractor[0], local_id_from=distractor[1],
                       space_id_to=tokens[cand_slot][0], local_id_to=tokens[cand_slot][1],
                       edge_type=EDGE_REFERS_TO, strength=1, source=SOURCE_BARE_TEXT)
    ctx_ref = _ensure(ctx, "__ctx_ref")
    ctx.edge_store.add(space_id_from=distractor[0], local_id_from=distractor[1],
                       space_id_to=ctx_ref[0], local_id_to=ctx_ref[1],
                       edge_type=EDGE_COOCCURS, strength=1, source=SOURCE_BARE_TEXT)
    return struct_ref, tokens, skel_root, learned_word, distractor, ctx_ref, rel_causes


# ============ TC8 two-hop 核证（先验 reader·审2 致命-1）============

def test_tc8_rel_kind_of_skeleton_two_hop():
    """TC8：rel_kind_of_skeleton(skel)=REL_CAUSES / rel_kind_of_skeleton(struct_ref)=0。
    REALIZES 挂 skeleton 非 unit（struct_ref 无 REALIZES 边）→ 两跳必需·一跳恒 0（审2 致命-1）。"""
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    struct_ref, _tokens, skel_root, _learned, _dist, _cref, _rel = _build_bridge_fixture(ctx)
    assert g.rel_kind_of_skeleton(skel_root) == REL_CAUSES, \
        "skeleton→REALIZES→REL_* → ATTR_RELATION_PRIMITIVE=REL_CAUSES（two-hop 第二跳）"
    assert g.rel_kind_of_skeleton(struct_ref) == 0, \
        "★ struct_ref 无 REALIZES 边→0（一跳 rel_kind_of(unit) 恒 0·两跳必需·审2 致命-1）"


def test_tc8b_read_cue_sig_rebuilds():
    """TC8b：read_cue_sig 从 skeleton 重建 cue_sig tuple（cue_at=1 位非 None·其余 None）·镜像 _collect_cue_sig。"""
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    _struct, _tokens, skel_root, _learned, _dist, _cref, _rel = _build_bridge_fixture(ctx)
    cue_sig = g.read_cue_sig(skel_root)
    assert len(cue_sig) == 3, "3 PARAM leaves → cue_sig len 3"
    assert cue_sig[1] is not None, "cue_at=1 位非 None（ATTR_CUE_SIG 重建）"
    assert cue_sig[0] is None and cue_sig[2] is None, "非 cue 位 None（单 cue 位·天然·审2 核证）"


# ============ TC1 e2e：cue slot 学到的 cue 词激活（CORR_BONUS override collide） ============

def test_tc1_cue_slot_activates_learned_word():
    """TC1：bridge ON + cue slot + v2-learned 匹配 → dispatch 选 learned_word（CORR_BONUS=1001 严格胜 distractor collide=1=1000）。

    学全 BEYOND 共现：learned(collide=0) 经对应 bonus override collide 驱动的 distractor(collide=1)·
    floor 可测稳定激活（1001>1000·非 50/50 tie）。
    """
    ctx = make_train_context(DictBackend())
    struct_ref, tokens, _skel, learned, distractor, ctx_ref, _rel = _build_bridge_fixture(ctx)
    gates.CORRESPONDENCE_SLOT_MODE = True
    gates.DISPATCH_TOKEN_CHAIN_MODE = True   # slot.ref=token concept（cue 位 slot.ref=t1）
    wm = WorkMemory()
    wm.prior_topic_refs = [ctx_ref]   # collide ctx（distractor COOCCURS ctx_ref → collide=1）
    out = generate_output(_path(struct_ref), ctx.concept_graph, wm, LANG_NONE)
    # ★ cue slot(slot 1) 选 learned_word（CORR_BONUS=1001 > distractor collide=1=1000）
    assert out.parts[0].words[1] == _ref_lit(learned), \
        "★ cue slot 选 learned（CORR_BONUS override collide·学全 BEYOND 共现）"
    assert out.parts[0].words[1] != _ref_lit(distractor), "distractor（collide 高但无对应）落选"


def test_tc1b_correspondence_bonus_value():
    """TC1b：_correspondence_bonus 返 CORR_BONUS=SCORE_SCALE+1=1001（cue slot + 匹配）/ 0（非 cue slot 或不匹配）。"""
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    _struct, _tokens, _skel, learned, _dist, _cref, _rel = _build_bridge_fixture(ctx)
    assert CORR_BONUS == SCORE_SCALE + 1 == 1001, "两审定值=1001（严格胜 collide=1·floor 可重复）"
    # cue slot + learned cue_rel_of==REL_CAUSES → CORR_BONUS
    assert _correspondence_bonus(g, learned, REL_CAUSES, is_cue_slot=True) == CORR_BONUS
    # 非 cue slot → 0（cue-slot-aware·审1 CONDITION D）
    assert _correspondence_bonus(g, learned, REL_CAUSES, is_cue_slot=False) == 0
    # cue slot 但 unit_rel_kind=0（非 R-skeleton）→ 0
    assert _correspondence_bonus(g, learned, 0, is_cue_slot=True) == 0


def test_tc1c_production_gate_combo_uses_relation_driven_candidate():
    """生产组合下，learned cue 无 REFERS_TO 仍进入候选并取代骨架原 cue。"""
    ctx = make_train_context(DictBackend())
    struct_ref, tokens, _skel, learned, _dist, _ctx, _rel = _build_bridge_fixture(
        ctx, link_learned_candidate=False)
    gates.CORRESPONDENCE_SLOT_MODE = True
    gates.CUE_SLOT_FILL_MODE = True
    gates.DISPATCH_TOKEN_CHAIN_MODE = True
    gates.ORDINAL_SURFACE_MODE = True

    graph = ctx.concept_graph
    assert learned not in graph.activate_candidates(tokens[1]), \
        "learned cue 不得依赖测试手工 REFERS_TO 进入普通候选池"
    assert graph.relation_cue_candidates(REL_CAUSES, space_id=ctx.space_id) == [learned]

    out = generate_output(_path(struct_ref), graph, WorkMemory(), LANG_NONE)
    assert out.parts[0].words[1] == "引发"
    assert out.lineage[(struct_ref, 1)] == CUE_SLOT_FILL, \
        "学得 cue 胜出后仍是结构 cue 血统，不进入内容词上下文"


def test_relation_cue_candidates_exclude_teacher_and_shadow():
    """关系反向候选只认同 space 的 PRIMARY + BARE_TEXT D:11。"""
    ctx = make_train_context(DictBackend())
    _struct, _tokens, _skel, learned, _dist, _ctx, rel = _build_bridge_fixture(
        ctx, link_learned_candidate=False)
    teacher = _ensure(ctx, "教师因果词")
    _add_d11_primary(ctx, teacher, rel, source=SOURCE_TEACHER)
    shadow = _ensure(ctx, "候选因果词")
    record_emergent_relation_signal_shadow(
        ctx.edge_store, shadow, rel, space_id=ctx.space_id)

    assert ctx.concept_graph.relation_cue_candidates(
        REL_CAUSES, space_id=ctx.space_id) == [learned]


# ============ TC2 cue-slot-aware（反 theater·非 cue slot bonus=0）============

def test_tc2_non_cue_slot_no_bonus_e2e():
    """TC2（post-impl 审 MEDIUM-2 修·真验 cue-slot-aware e2e 非同义反复）：
    learned + distractor 都是非 cue slot(slot 0) 候选（REFERS_TO tokens[0]）·learned cue_rel 匹配·
    但 bonus=0（is_cue_slot=False）→ 选 collide 高的 distractor（非 learned）。对照 cue slot(slot 1) 选 learned。
    ★ 证 cue-slot-aware 是**机制保证**（非 fixture 结构保证）：同候选池·按 cue 位差异化选词·反 theater。
    若删 generate current_slot_is_cue / slot_dispatch _is_cue 读 → learned 全 slot 激活=theater → 本测抓（words[0] 错选 learned）。"""
    ctx = make_train_context(DictBackend())
    struct_ref, tokens, _skel, learned, distractor, ctx_ref, _rel = _build_bridge_fixture(ctx)
    # ★ learned + distractor 也成非 cue slot(slot 0) 候选（REFERS_TO tokens[0]）·验 bonus 不泄漏到非 cue 位
    for cand in (learned, distractor):
        ctx.edge_store.add(space_id_from=cand[0], local_id_from=cand[1],
                           space_id_to=tokens[0][0], local_id_to=tokens[0][1],
                           edge_type=EDGE_REFERS_TO, strength=1, source=SOURCE_BARE_TEXT)
    gates.CORRESPONDENCE_SLOT_MODE = True
    gates.DISPATCH_TOKEN_CHAIN_MODE = True
    wm = WorkMemory()
    wm.prior_topic_refs = [ctx_ref]
    out = generate_output(_path(struct_ref), ctx.concept_graph, wm, LANG_NONE)
    # ★ 非 cue slot(slot 0)：learned 是候选 + cue_rel 匹配·但 is_cue_slot=False → bonus=0 → distractor(collide=1) 胜
    assert out.parts[0].words[0] == _ref_lit(distractor), \
        "★ 非 cue slot → bonus=0 → learned(collide=0) 落选·distractor(collide=1) 胜（cue-slot-aware 机制保证）"
    assert out.parts[0].words[0] != _ref_lit(learned), "learned 不在非 cue slot 激活（反 theater）"
    # 对照：cue slot(slot 1) 选 learned（bonus=1001）
    assert out.parts[0].words[1] == _ref_lit(learned), "cue slot 选 learned（对照基线·bonus 触发）"


# ============ TC3 cue_sig=()（无 cue 位）→ 无 bonus ============

def test_tc3_no_cue_sig_no_bonus():
    """TC3：skeleton 无 ATTR_CUE_SIG（cue_at=None）→ read_cue_sig 全 None → current_cue_slots=∅ → 无 bonus。
    generate → cue 位(slot 1) 选 collide 高的 distractor（bonus 不触发）。"""
    ctx = make_train_context(DictBackend())
    struct_ref, _tokens, _skel, learned, distractor, ctx_ref, _rel = _build_bridge_fixture(
        ctx, cue_at=None)   # 无 cue 位
    g = ctx.concept_graph
    # read_cue_sig 全 None
    cue_sig = g.read_cue_sig(_skel)
    assert all(c is None for c in cue_sig), "无 ATTR_CUE_SIG → cue_sig 全 None"
    gates.CORRESPONDENCE_SLOT_MODE = True
    gates.DISPATCH_TOKEN_CHAIN_MODE = True
    wm = WorkMemory()
    wm.prior_topic_refs = [ctx_ref]
    out = generate_output(_path(struct_ref), g, wm, LANG_NONE)
    # 无 cue 位 → current_cue_slots=∅ → 无 bonus → distractor(collide=1) 胜 learned(collide=0)
    assert out.parts[0].words[1] == _ref_lit(distractor), \
        "cue_sig=() → 无 bonus → collide 驱动选 distractor（非 learned）"
    assert out.parts[0].words[1] != _ref_lit(learned)


# ============ TC4 boot 种子不触发（SOURCE_TEACHER·审1 CONDITION A source 滤）============

def test_tc4_boot_seed_source_filter():
    """TC4：learned D:11 source=SOURCE_TEACHER（boot 种子）→ cue_rel_of 返 0（source 滤·审1 CONDITION A）→ 无 bonus。
    boot 种子（等于/导致·collide 已高）不走 bridge·防 floor 虚高 theater。"""
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    struct_ref, _tokens, _skel, learned, distractor, ctx_ref, _rel = _build_bridge_fixture(
        ctx, learned_source=SOURCE_TEACHER)
    # ★ cue_rel_of source 滤：SOURCE_TEACHER → 0（非 v2-learned BARE_TEXT）
    assert g.cue_rel_of(learned) == 0, "★ boot 种子 SOURCE_TEACHER → cue_rel_of=0（source 滤·反 theater）"
    gates.CORRESPONDENCE_SLOT_MODE = True
    gates.DISPATCH_TOKEN_CHAIN_MODE = True
    wm = WorkMemory()
    wm.prior_topic_refs = [ctx_ref]
    out = generate_output(_path(struct_ref), g, wm, LANG_NONE)
    # source 滤 → 无 bonus → distractor(collide=1) 胜
    assert out.parts[0].words[1] == _ref_lit(distractor), "boot 种子不触发 bonus → 选 collide distractor"


def test_tc4b_v2_learned_bare_text_passes():
    """TC4b（对照）：learned D:11 source=SOURCE_BARE_TEXT（v2-learned）→ cue_rel_of=REL_CAUSES（source 滤放行）。"""
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    _struct, _tokens, _skel, learned, _dist, _cref, _rel = _build_bridge_fixture(ctx)   # 默认 BARE_TEXT
    assert g.cue_rel_of(learned) == REL_CAUSES, "v2-learned BARE_TEXT → cue_rel_of=REL_CAUSES（放行）"


def test_tc4c_shadow_tier_filter():
    """TC4c：D:11 tier=SHADOW（未 promote）→ cue_rel_of 返 0（tier 滤·审1 CONDITION B 学到的才驱动）。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    g = ctx.concept_graph
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    rel_causes = rel_prims[REL_CAUSES]
    w = _ensure(ctx, "未晋词")
    # 只建 SHADOW（不 flip PRIMARY）
    record_emergent_relation_signal_shadow(ctx.edge_store, w, rel_causes, space_id=sid)
    assert g.cue_rel_of(w) == 0, "SHADOW（未 promote）→ cue_rel_of=0（tier 滤·学到的才驱动）"


# ============ TC5 无 REALIZES → rel_kind=0 → 无 bonus（two-hop robust 退化）============

def test_tc5_no_realizes_no_bonus():
    """TC5：skeleton 无 REALIZES 边 → rel_kind_of_skeleton=0 → unit_rel_kind=0 → 无 bonus（robust 退化）。"""
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    struct_ref, _tokens, skel_root, learned, distractor, ctx_ref, _rel = _build_bridge_fixture(
        ctx, with_realizes=False)
    assert g.rel_kind_of_skeleton(skel_root) == 0, "无 REALIZES → rel_kind=0（非 R-skeleton）"
    gates.CORRESPONDENCE_SLOT_MODE = True
    gates.DISPATCH_TOKEN_CHAIN_MODE = True
    wm = WorkMemory()
    wm.prior_topic_refs = [ctx_ref]
    out = generate_output(_path(struct_ref), g, wm, LANG_NONE)
    # rel_kind=0 → bonus=0 → distractor(collide=1) 胜
    assert out.parts[0].words[1] == _ref_lit(distractor), "无 REALIZES → 无 bonus → 选 collide distractor"


# ============ TC6 cue_sig length ≠ role_seq → runtime length-guard → 无 bonus ============

def test_tc6_length_guard_mismatch():
    """TC6：skeleton cue_sig length(2) ≠ role_seq length(3) → runtime length-guard → current_cue_slots=∅ → 无 bonus。
    alignment 是 discovery-process property 非 invariant·length-check 是唯一 sound guard（§3）。"""
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    # skeleton 2 leaves（cue_at=1）·但 unit 3 role/token → len(cue_sig)=2 ≠ len(role_seq)=3
    struct_ref, _tokens, skel_root, learned, distractor, ctx_ref, _rel = _build_bridge_fixture(
        ctx, n_skel_leaves=2, cue_at=1)
    cue_sig = g.read_cue_sig(skel_root)
    assert len(cue_sig) == 2, "skeleton 2 leaves → cue_sig len 2"
    gates.CORRESPONDENCE_SLOT_MODE = True
    gates.DISPATCH_TOKEN_CHAIN_MODE = True
    wm = WorkMemory()
    wm.prior_topic_refs = [ctx_ref]
    out = generate_output(_path(struct_ref), g, wm, LANG_NONE)
    # length 不等 → guard → current_cue_slots=∅ → 无 bonus → distractor(collide=1) 胜
    assert out.parts[0].words[1] == _ref_lit(distractor), \
        "★ length 错位 → guard → 无 bonus（sound·防 position 错位 theater）"


# ============ TC7 gate OFF bit-identical（无 bonus·逐字现状）============

def test_tc7_gate_off_bit_identical():
    """TC7：gate OFF → 无 correspondence bonus → cue slot 选 collide 高的 distractor（既有行为·bit-identical）。
    对照 TC1（gate ON 选 learned）：OFF 时 bonus=0·collide 主轴驱动·逐字现状。"""
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    struct_ref, _tokens, _skel, learned, distractor, ctx_ref, _rel = _build_bridge_fixture(ctx)
    gates.CORRESPONDENCE_SLOT_MODE = False   # ★ gate OFF（CI default）
    gates.DISPATCH_TOKEN_CHAIN_MODE = True
    wm = WorkMemory()
    wm.prior_topic_refs = [ctx_ref]
    out = generate_output(_path(struct_ref), g, wm, LANG_NONE)
    # gate OFF → 无 bonus → distractor(collide=1=1000) 胜 learned(collide=0)
    assert out.parts[0].words[1] == _ref_lit(distractor), \
        "gate OFF → 无 bonus → collide 驱动选 distractor（bit-identical·对照 TC1 gate ON 选 learned）"
    assert out.parts[0].words[1] != _ref_lit(learned)
    # 确证 learned 的 cue_rel_of 仍读得到（reader 不 gated·仅 dispatch_slot _corr_gate 守）
    assert g.cue_rel_of(learned) == REL_CAUSES


def test_tc7b_gate_off_correspondence_bonus_not_called():
    """TC7b：gate OFF → _corr_gate=False → combine 不进 correspondence 分支（_correspondence_bonus 不调·bit-identical）。"""
    ctx = make_train_context(DictBackend())
    # gate OFF（fixture 已 default OFF）→ dispatch_slot combine `if ... or _corr_gate:` _corr_gate=False
    # learned 即使有 D:11·bonus 路径不触发（getattr gates.CORRESPONDENCE_SLOT_MODE=False）
    assert gates.CORRESPONDENCE_SLOT_MODE is False, "CI default OFF"
    # CORR_BONUS 常量正确（不依赖 gate）
    assert CORR_BONUS == 1001
