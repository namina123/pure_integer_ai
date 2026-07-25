"""floor 端到端下游激活率测试（断奶 critical path 第 2 件·反 theater 首版机制层预验·doc/重来_floor_端到端下游激活率_2026-07-17）。

机制（2 对抗审 APPROVE-WITH-CONDITIONS·FLOOR_ACTIVATION_MODE）：
  - measure_floor_activation（读侧后验重导 cue slot 激活率·镜像 generate.py stash·纯读·审1 MEDIUM-2 用 role_seq）
  - verdict gate-gated conjunct（审1 严重-1·gate OFF→floor_conjunct=True→bit-identical·守 SW2/9/16）
  - floor_overrides bug 修（语言域 oov_promote 结构性低·审1 MEDIUM-3）

FC1 held-out cue slot 学到的对应词激活（activation≥阈·C-vs-L 真判别）。
FC2 闭包-only C（无 D:11 tally）→ cue_rel_of=0 → activation=0（攻证①·闭包偷不过）。
FC3 measured-guard：空探针（total=0）→ measured=False（防 stub-0 vacuous）。
FC4 false-positive 臂：distractor（无对应）cue slot → false_positive 高（specificity 硬闸·攻证③）。
FC6 floor_overrides 修：语言域 oov_promote override → floors_met 不恒卡（competence 守）。
FC7 gate-gated conjunct bit-identical：gate OFF→statistical_ready 不受 floor 影响 / gate ON+未测→False（审1 严重-1）。
FC8 read_cue_sig 镜像 _collect_cue_sig + measure role_seq length-guard（审2 LOW-3 + 审1 MEDIUM-2）。

piece 3 orchestrator（doc/重来_floor_orchestrator_piece3_2026-07-17·§9 锁定·drop generate recognition measure）：
FC9  orchestrator e2e + ★S1 load-bearing：sample-disjoint held-out（同训练 shape）→ S1 all_ops 搜填 map
     → INSTANTIATES fire → FloorActivation.measured=True（recognition：observed cue 位 token D:11）。
FC10 gate OFF bit-identical：FLOOR_ACTIVATION_MODE OFF→orchestrator 不调→floor 三参数 default→statistical_ready
     不受影响（守 SW2/9/16 既有过·与 FC7 互补·FC7 测 weaning 层 / FC10 测 orchestrator 入口）。
FC11 probe_holdout=0 早返：无 probe_corpus 或 probe_set_disjoint=False → FloorActivation(measured=False)
     early-return（审2 LOW-3 defense-in-depth + measured-guard·防 stub-0）。

铁律：纯整数 / bit-identical（gate OFF default·floor_conjunct gate-gated True）/ 反 theater（C-vs-L 判别 +
measured-guard + false-positive 臂 + source 滤断闭包 + 非自证生成纯读）/ 不写死（cue_rel_of 读图 D:11 边）。
诚实边界：FC1-8 fixture 手造（直建 skeleton/REALIZES/D:11/OutputResult·非生产 tally/promote/orchestrator 链）·
机制层预验非 empirical 泛化率（empirical 真 ConceptNet held-out defer W4）·FC9 orchestrator e2e（含 S1 fix）。
"""
from __future__ import annotations

import copy

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_types import EDGE_COMPOSES, EDGE_RELATION_SIGNAL, EDGE_INSTANTIATES
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import NODE_WORD, NODE_CONCEPT, TIER_PRIMARY
from pure_integer_ai.storage.composes_attr import (
    record_composes_attr, ATTR_OPERATOR, ATTR_OPERAND, ATTR_CUE_SIG,
    ATTR_ORIGIN, ORIGIN_DISCOVERED,
)
from pure_integer_ai.numeric.symbol_domain import make_variable
from pure_integer_ai.cognition.shared.relation_primitives import ensure_relation_primitives, REL_CAUSES
from pure_integer_ai.cognition.understanding.emergent_relation_signal import record_emergent_relation_signal_shadow
from pure_integer_ai.cognition.understanding.realizes import build_realizes_edge
from pure_integer_ai.cognition.understanding.instantiates import build_instantiates_edge
from pure_integer_ai.cognition.understanding.role_precedes import attach_role_seq
from pure_integer_ai.cognition.understanding.arith_observe import register_arith_operator
from pure_integer_ai.cognition.process.structure_discover import (
    _collect_cue_sig, shape_signature, _shape_name, _normalize_abstract_sig,
    _collect_slot_lcas, load_discovered_operators,
)
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.floor_measure import measure_floor_activation
from pure_integer_ai.cognition.shared.types import OutputResult, OutputPart, FloorActivation
from pure_integer_ai.teacher.weaning import (
    WeaningMetrics, language_statistical_weaning_check,
    LANG_FLOOR_ACTIVATION, LANG_FLOOR_FALSE_POS, METRIC_OOV_PROMOTE,
)
from tests.test_experiments import make_train_context


@pytest.fixture(autouse=True)
def _gate_reset():
    """每测前后复位 FLOOR_ACTIVATION_MODE（守测试隔离·防跨测泄漏）。"""
    saved = gates.FLOOR_ACTIVATION_MODE
    gates.FLOOR_ACTIVATION_MODE = False
    yield
    gates.FLOOR_ACTIVATION_MODE = saved


# ---- helpers（镜像 test_correspondence_bridge fixture builders）----

def _ensure(ctx, surface, node_type=NODE_WORD):
    return ctx.concept_index.ensure(surface, space_id=ctx.space_id, node_type=node_type)


def _add_composes(ctx, frm, to, order_index):
    ctx.edge_store.add(
        space_id_from=frm[0], local_id_from=frm[1],
        space_id_to=to[0], local_id_to=to[1],
        edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_BARE_TEXT,
        order_index=order_index)


def _build_skel(ctx, root, slot_refs, cue_at=None, cue_ref=None):
    """对齐就绪 skeleton：root ATTR_OPERATOR(NOP) → slot 叶（ATTR_OPERAND make_variable(i)·PARAM slot）·
    cue_at 位叶加挂 ATTR_CUE_SIG=cue_ref。镜像 test_correspondence_bridge._build_skel。"""
    record_composes_attr(ctx.backend, ref=root, kind=ATTR_OPERATOR, int_a=0, int_b=0)
    for ti, tok in enumerate(slot_refs):
        record_composes_attr(ctx.backend, ref=tok, kind=ATTR_OPERAND,
                             int_a=make_variable(ti), int_b=0)
        _add_composes(ctx, root, tok, ti)
        if ti == cue_at and cue_ref is not None:
            record_composes_attr(ctx.backend, ref=tok, kind=ATTR_CUE_SIG,
                                 int_a=cue_ref[0], int_b=cue_ref[1])


def _build_floor_fixture(ctx, *, n_leaves=3, cue_at=1, with_realizes=True,
                         with_learned_d11=True, n_roles=3):
    """held-out R-skeleton fixture：struct_ref(role_seq) + INSTANTIATES→skeleton(cue_at) + REALIZES→REL_CAUSES
    + 可选 D:11 learned(→REL_CAUSES BARE_TEXT PRIMARY)。返 (struct_ref, skel_root, learned_word, rel_causes)。"""
    sid = ctx.space_id
    struct_ref = _ensure(ctx, "__seg_floor", node_type=NODE_CONCEPT)
    attach_role_seq(ctx.backend, struct_ref, list(range(n_roles)))
    skel_root = _ensure(ctx, "__skel_floor", node_type=NODE_CONCEPT)
    cue_token = _ensure(ctx, "使")
    leaves = [_ensure(ctx, f"__leaf{i}") for i in range(n_leaves)]
    _build_skel(ctx, skel_root, leaves, cue_at=cue_at, cue_ref=cue_token)
    build_instantiates_edge(ctx.edge_store, struct_ref, skel_root, space_id=sid)
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    rel_causes = rel_prims[REL_CAUSES]
    if with_realizes:
        build_realizes_edge(ctx.edge_store, skel_root, rel_causes, space_id=sid)
    learned_word = _ensure(ctx, "引发")
    if with_learned_d11:
        record_emergent_relation_signal_shadow(ctx.edge_store, learned_word, rel_causes, space_id=sid)
        ctx.edge_store.set_tier(
            space_id_from=learned_word[0], local_id_from=learned_word[1],
            space_id_to=rel_causes[0], local_id_to=rel_causes[1],
            edge_type=EDGE_RELATION_SIGNAL, new_tier=TIER_PRIMARY)
    return struct_ref, skel_root, learned_word, rel_causes


def _out(unit, token_refs):
    """单 part OutputResult（token_refs=generate 产的段真 token·测 measure 读侧重导）。"""
    return OutputResult(parts=[OutputPart(unit=unit, token_refs=list(token_refs))])


def _wm(rounds, *, conduction=800, realizes=700, judge_self=800, oov=200,
        intervention=100, retention=900, dependency=100):
    return WeaningMetrics(rounds=rounds, conduction_rate=conduction, realizes_rate=realizes,
                          judge_self_rate=judge_self, oov_promote_rate=oov,
                          intervention_rate=intervention, holdout_retention=retention,
                          dependency=dependency)


def _passing_history():
    return [_wm(1, intervention=100), _wm(2, intervention=90),
            _wm(3, intervention=80), _wm(4, intervention=70)]


def _sw_check(history=None, **kw):
    """默认全过调用（5 锚点 + 2 前置 + measured 全 True）·kw 覆盖单项。镜像 test_statistical_weaning._check。"""
    defaults = dict(encoding_grounded=True, crosslingual_seeded=True, probe_set_disjoint=True,
                    neg_pathway_active=True, teacher_present=True, fadeout_measured=True,
                    heldout_measured=True, heldout_generalization_permille=700)
    defaults.update(kw)
    return language_statistical_weaning_check(history if history is not None else _passing_history(), **defaults)


# ============ FC1 held-out cue slot 学到的对应词激活（C-vs-L 真判别）============

def test_fc1_learned_activates():
    """FC1：held-out R-skeleton cue slot(slot 1) 选学到的对应词(learned·D:11→REL_CAUSES)→ activation=1000≥500。
    C-vs-L 真判别：学到的 W 在 cue slot 正确激活（cue_rel_of(learned)==rel_kind）。"""
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    struct_ref, _skel, learned, _rel = _build_floor_fixture(ctx)
    t0, t2 = _ensure(ctx, "t0"), _ensure(ctx, "t2")
    fa = measure_floor_activation(g, _out(struct_ref, [t0, learned, t2]))
    assert fa.measured is True and fa.total == 1, "1 cue slot(slot 1) 计 total"
    assert fa.activated == 1, "★ learned cue_rel_of==REL_CAUSES → activated"
    assert fa.activation_permille == 1000 >= LANG_FLOOR_ACTIVATION, "激活率 1000≥500（C-vs-L 真判别）"
    assert fa.false_positive_permille == 0 <= LANG_FLOOR_FALSE_POS


# ============ FC2 闭包-only C（无 D:11）→ activation=0（攻证①·闭包偷不过）============

def test_fc2_closure_only_zero():
    """FC2：闭包-only C（无 D:11 tally）→ cue_rel_of=0 → activation=0（攻证①·boot 闭包不能偷激活）。
    measured=True（total>0·测了未达阈）但 activation=0 → floor 不过。"""
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    struct_ref, _skel, _learned, _rel = _build_floor_fixture(ctx, with_learned_d11=False)
    bare_word = _ensure(ctx, "bare_no_d11")
    t0, t2 = _ensure(ctx, "t0"), _ensure(ctx, "t2")
    assert g.cue_rel_of(bare_word) == 0, "无 D:11 → cue_rel_of=0（闭包-only C）"
    fa = measure_floor_activation(g, _out(struct_ref, [t0, bare_word, t2]))
    assert fa.measured is True, "total=1>0 → measured（测了）"
    assert fa.activated == 0 and fa.activation_permille == 0, "★ activation=0（闭包偷不过·C-vs-L）"
    assert fa.false_positive_permille == 1000, "cue slot 选了 cue_rel_of=0≠rel_kind → false_positive"


# ============ FC3 measured-guard：空探针 → measured=False（防 stub-0）============

def test_fc3_measured_guard_empty():
    """FC3：空 output（无 parts·not-run / 无 held-out cue slot）→ total=0 → measured=False → anchor_pf 不过。
    防 stub-0 vacuous（空探针与「测了未达阈」不可区分·measured-guard 守）。"""
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    fa = measure_floor_activation(g, OutputResult(parts=[]))
    assert fa.measured is False and fa.total == 0, "空探针 → measured=False（stub-0 guard）"
    assert fa.activation_permille == 0   # max(0,1) 守除零


# ============ FC4 false-positive 臂：distractor（无对应）cue slot ============

def test_fc4_false_positive_arm():
    """FC4：distractor（高 collide·无对应 D:11）在 cue slot → false_positive 高（specificity 硬闸·攻证③）。
    floor false_positive_permille ≤ LANG_FLOOR_FALSE_POS 守——distractor 误激活会超阈不过 floor。"""
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    struct_ref, _skel, _learned, _rel = _build_floor_fixture(ctx, with_learned_d11=False)
    distractor = _ensure(ctx, "__distractor")
    t0, t2 = _ensure(ctx, "t0"), _ensure(ctx, "t2")
    fa = measure_floor_activation(g, _out(struct_ref, [t0, distractor, t2]))
    assert fa.activated == 0, "distractor 无对应 → 不 activated"
    assert fa.false_positive_permille == 1000 > LANG_FLOOR_FALSE_POS, "★ distractor 误激活 → false_positive 超阈（攻证③）"


# ============ FC6 floor_overrides bug 修（语言域 oov_promote 结构性低）============

def test_fc6_floor_overrides_fix():
    """FC6：语言域 oov_promote 结构性低（止血① 后）→ 无 override 卡 floors_met / override {OOV:0} 放行。
    审1 MEDIUM-3：override 移除结构性域错·competence 三量（conduction/realizes/judge_self）守。"""
    hist = [_wm(1, oov=0, intervention=100), _wm(2, oov=0, intervention=90),
            _wm(3, oov=0, intervention=80), _wm(4, oov=0, intervention=70)]
    # 无 override：oov=0 < FLOOR_OOV_PROMOTE=100 → floors_met False
    rep_no_ov = _sw_check(hist)
    assert rep_no_ov.floors_met is False, "oov=0 无 override → floors_met False（结构性域错）"
    # override {OOV:0}：移除 oov 卡阈 → floors_met True（competence 三量仍守）
    rep_ov = _sw_check(hist, floor_overrides={METRIC_OOV_PROMOTE: 0})
    assert rep_ov.floors_met is True, "★ override {OOV:0} → floors_met True（floor_overrides bug 修）"


# ============ FC7 gate-gated conjunct bit-identical（审1 严重-1 核心）============

def test_fc7_gate_off_bit_identical_and_gate_on_guard():
    """FC7（审1 严重-1 核心）：gate-gated floor_conjunct 守 bit-identical + gate ON 守 measured。
    (a) gate OFF：floor 三参数 default（未传）→ floor_conjunct=True → statistical_ready 不受 floor 影响（SW2 基线 True）。
    (b) gate ON + floor_measured=False（orchestrator 未跑/未测）→ floor_conjunct=False → statistical_ready False（守）。
    (c) gate ON + floor_measured=True + activation≥阈 + fp≤阈 → floor_conjunct=True → statistical_ready True。"""
    # (a) gate OFF（fixture default）+ 不传 floor 参数 → default False/0/0 → floor_conjunct=True → 仍 True（bit-identical）
    assert gates.FLOOR_ACTIVATION_MODE is False, "CI default OFF"
    rep_a = _sw_check()
    assert rep_a.statistical_ready is True, "★ gate OFF → floor_conjunct=True → SW2 基线 True（bit-identical·审1 严重-1 守）"
    # (b) gate ON + floor 未测（default False）→ floor_conjunct=False → statistical_ready False
    gates.FLOOR_ACTIVATION_MODE = True
    rep_b = _sw_check()   # floor 参数 default False/0/0
    assert rep_b.statistical_ready is False, "★ gate ON + floor_measured=False → floor_conjunct=False → 不过（measured-guard）"
    # (c) gate ON + floor 全过 → statistical_ready True
    rep_c = _sw_check(floor_measured=True, floor_activation_permille=700,
                      floor_false_positive_permille=100)
    assert rep_c.statistical_ready is True, "gate ON + floor 全过（activation 700≥500·fp 100≤200）→ True"


# ============ FC8 read_cue_sig 镜像 + measure role_seq length-guard（审2 LOW-3 + 审1 MEDIUM-2）============

def test_fc8_read_cue_sig_mirror_and_length_guard():
    """FC8：(a) graph.read_cue_sig(skel) == _collect_cue_sig(backend, graph, skel) 逐 ref 一致（审2 LOW-3·防 read/write 分歧）。
    (b) measure role_seq length-guard（审1 MEDIUM-2）：cue_sig len≠role_seq → skip（不计 total·防 accumulation 错位漏计）。"""
    ctx = make_train_context(DictBackend())
    g = ctx.concept_graph
    # (a) read_cue_sig 镜像 _collect_cue_sig
    _struct, skel_root, _learned, _rel = _build_floor_fixture(ctx, n_leaves=3, cue_at=1, n_roles=3)
    assert g.read_cue_sig(skel_root) == _collect_cue_sig(ctx.backend, g, skel_root), \
        "★ read_cue_sig == _collect_cue_sig 逐 ref 一致（审2 LOW-3）"
    # (b) length-guard：skeleton 2 leaves（cue_sig len 2）≠ role_seq len 3 → measure skip（total=0 → measured=False）
    struct_m, _skel_m, _lw, _rl = _build_floor_fixture(ctx, n_leaves=2, cue_at=1, n_roles=3)
    t0, learned_m, t2 = _ensure(ctx, "t0"), _ensure(ctx, "lw"), _ensure(ctx, "t2")
    fa = measure_floor_activation(g, _out(struct_m, [t0, learned_m, t2]))
    assert fa.total == 0 and fa.measured is False, \
        "★ cue_sig len(2)≠role_seq len(3) → length-guard skip（审1 MEDIUM-2·防错位漏计）"


# ============ piece 3 orchestrator 测试（doc/重来_floor_orchestrator_piece3_2026-07-17 §9）============
#
# FC9/10/11 测生产 orchestrator `_measure_floor_pass`（formal_train.py）·drop generate·recognition measure。
# fixture 策略：手建 training-registered skeleton（NOP + token 叶 + ATTR_CUE_SIG + REALIZES + ATTR_ORIGIN=DISCOVERED
# + register_arith_operator）+ D:11 SHADOW→PRIMARY for cue word·mirror FC1 _build_floor_fixture 范式·加注册。
# held-out CollectedItem（probe_corpus）·同 shape_signature（3 tokens）→ S1 scenario（auto_discover 幂等 skip）。

# 生产 gate 集（formal_train try/finally 共翻 + FLOOR_ACTIVATION_MODE env-gated·本测显式翻 ON·守还原）
_PROD_GATE_NAMES = (
    "FLOOR_ACTIVATION_MODE",       # floor orchestrator env-gated（同 STATISTICAL_WEANING_MODE·非 try/finally）
    "COMPOSES_COMBINE_MODE",       # INSTANTIATES fire（observe.py:204-208）
    "DISPATCH_TOKEN_CHAIN_MODE",   # attach_token_seq（observe.py:285·token_refs 源）
    "REALIZES_MODE",               # label_realizes_is_a/causes + rel_kind_of_skeleton 读
    "CUE_CLUSTER_MODE",            # ATTR_CUE_SIG 写（discover_skeleton）+ read_cue_sig 读
    "ORACLE_PROMOTE_MODE",         # tally→SHADOW（训练期·本测手建 D:11 替）
    "CORRESPONDENCE_SLOT_MODE",    # 桥第 8 路（floor 不读 dispatch·但生产共翻·守真路径）
    "STRUCT_REF_CONTENT_HASH_MODE",  # __seg_{stage}_{h63(tokens)} 确定性 struct_ref（FC9 find held-out struct_ref）
)


@pytest.fixture
def _prod_gates():
    """翻生产 gate 集 ON·测后还原（守测试隔离·镜像 formal_train try/finally 范式）。"""
    saved = {name: getattr(gates, name) for name in _PROD_GATE_NAMES}
    for name in _PROD_GATE_NAMES:
        setattr(gates, name, True)
    try:
        yield
    finally:
        for name, v in saved.items():
            setattr(gates, name, v)


def _register_lang_skeleton(ctx, skel_root: tuple, arity: int = 3) -> tuple:
    """把手建 skeleton 注册成 DiscoveredOperator（ATTR_ORIGIN=DISCOVERED + register_arith_operator）·
    使 load_discovered_operators 找到它（S1 all_ops 搜源）。返 name_ref。"""
    record_composes_attr(ctx.backend, ref=skel_root, kind=ATTR_ORIGIN, int_a=ORIGIN_DISCOVERED)
    sig = tuple(shape_signature(ctx.concept_graph, skel_root))
    abstract_sig = _normalize_abstract_sig(_collect_slot_lcas(ctx.backend, ctx.concept_graph, skel_root))
    cue_sig = _normalize_abstract_sig(_collect_cue_sig(ctx.backend, ctx.concept_graph, skel_root))
    name = _shape_name(sig, arity, abstract_sig, cue_sig)
    name_ref = register_arith_operator(ctx.backend, ctx.concept_index, name, skel_root, arity=arity)
    return name_ref


# ============ FC9 orchestrator e2e + S1 load-bearing（审1 严重）============

def test_fc9_orchestrator_e2e_s1_same_shape(_prod_gates):
    """FC9（★ 审1 严重 S1 load-bearing）：orchestrator e2e — sample-disjoint held-out（**同训练 shape**）
    → S1 fix 搜 all_ops 填 lang_skeleton_by_item[id(held_out)] → observe fire INSTANTIATES → measure_floor_activation
    measured=True（recognition：held-out cue 位 observed input token 携 training-learned D:11 匹配 skeleton rel）。

    S1 真实场景（非 novel-shape）：held-out 与训练**同 shape_signature** → auto_discover_operators 幂等 skip
    （structure_discover.py:1366-1368·lookup+ATTR_OPERATOR_DEF）→ this-call `discovered` 空 → 旧路径（仅 this-call）
    map 不填 → observe 不 fire INSTANTIATES → read_instantiates=None → measured=False silent veto。
    **S1 修**：搜 all_ops = training_lang_ops（load_discovered_operators）+ newly_discovered·training-registered
    skeleton 命中 shape → map 填 → INSTANTIATES fire。本测**必复现 S1 场景**（同 shape·非 novel）·否则隐藏 S1 bug。
    """
    from pure_integer_ai.experiments.formal_train import _measure_floor_pass
    from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_PRECEDES
    from pure_integer_ai.numeric.symbol_domain import OPCODE_NOP
    from pure_integer_ai.crosscut.determinism.hasher import Hasher
    from pure_integer_ai.cognition.shared.types import STAGE_TRAINING

    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    g = ctx.concept_graph

    # Training-registered skeleton S：root ATTR_OPERATOR=OPCODE_NOP（同 orchestrator·使 shape_signature 匹配 held-out root）
    # + 叶 ATTR_OPERAND=make_variable(i)（PARAM slot·discover_skeleton 建·read_cue_sig 读 ATTR_CUE_SIG 须此标记）+
    # ATTR_CUE_SIG on slot 1 cue leaf（cue 拆簇拆位·_collect_cue_sig 镜像）。
    skel_root = _ensure(ctx, "__skel_fc9", node_type=NODE_CONCEPT)
    cue_token = _ensure(ctx, "使")
    leaf0 = _ensure(ctx, "__fc9_leaf0")
    leaf2 = _ensure(ctx, "__fc9_leaf2")
    record_composes_attr(ctx.backend, ref=skel_root, kind=ATTR_OPERATOR, int_a=OPCODE_NOP, int_b=0)
    for ti, leaf in enumerate([leaf0, cue_token, leaf2]):
        record_composes_attr(ctx.backend, ref=leaf, kind=ATTR_OPERAND,
                             int_a=make_variable(ti), int_b=0)
        _add_composes(ctx, skel_root, leaf, ti)
        if ti == 1:   # slot 1 = cue·挂 ATTR_CUE_SIG
            record_composes_attr(ctx.backend, ref=leaf, kind=ATTR_CUE_SIG,
                                 int_a=cue_token[0], int_b=cue_token[1])
    # 健全性：shape_signature(S) == orchestrator-建 root（OPCODE_NOP + 3 叶）·shape 匹配是 S1 命中地基
    sig_s = tuple(shape_signature(g, skel_root))
    assert sig_s[0] == OPCODE_NOP and len(sig_s) == 4, "S shape = [OPCODE_NOP, -1×3]（与 orchestrator 同）"

    # REALIZES→REL_CAUSES（label_realizes_causes 镜像·oracle 直建·外源 ConceptNet·sound 无 D:11 写）。
    rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
    rel_causes = rel_prims[REL_CAUSES]
    build_realizes_edge(ctx.edge_store, skel_root, rel_causes, space_id=sid)
    # 注册 S 为 DiscoveredOperator（ATTR_ORIGIN=DISCOVERED + register_arith_operator）·使 load_discovered_operators 命中
    _register_lang_skeleton(ctx, skel_root, arity=3)
    # D:11 for cue word "使" → REL_CAUSES（SHADOW + PRIMARY·mirror FC1 _build_floor_fixture·training tally→promote 产物）
    record_emergent_relation_signal_shadow(ctx.edge_store, cue_token, rel_causes, space_id=sid)
    ctx.edge_store.set_tier(
        space_id_from=cue_token[0], local_id_from=cue_token[1],
        space_id_to=rel_causes[0], local_id_to=rel_causes[1],
        edge_type=EDGE_RELATION_SIGNAL, new_tier=TIER_PRIMARY)
    assert g.cue_rel_of(cue_token) == REL_CAUSES, "★ training-learned D:11 使→REL_CAUSES（C-vs-L 判别基础）"
    assert g.rel_kind_of_skeleton(skel_root) == REL_CAUSES, "S REALIZES→REL_CAUSES"

    # Sanity：load_discovered_operators 命中 S（S1 all_ops 源）
    loaded = load_discovered_operators(ctx.backend, space_id=sid)
    assert any(op.skeleton_ref == skel_root for op in loaded), "load_discovered_operators 命中 S"

    # ★ S1 scenario：held-out CollectedItem **同 shape**（3 tokens·cue "使" 在 slot 1·与训练 skeleton 同形）。
    # auto_discover_operators 幂等 skip（同 shape 已注册）→ newly_discovered 空 → S1 修搜 all_ops（含 S）填 map。
    held = CollectedItem(
        tokens=["__fc9_held0", "使", "__fc9_held2"],   # slot 1 = "使"（cue·与训练同）
        role_seq=[1, 1, 1],
        collect_type=COLLECT_PRECEDES, source=SOURCE_BARE_TEXT,
    )
    ctx.probe_corpus = [held]
    ctx.probe_set_disjoint = True   # W4 D4（anchor_heldout 守·本测 defense-in-depth）

    backend_before = ctx.backend.snapshot()
    work_memory_before = copy.deepcopy(ctx.work_memory)
    fa = _measure_floor_pass(ctx, ctx.backend, ctx.concept_graph)

    # V-06：S1 搜索和 INSTANTIATES 观察只允许发生在评测沙箱，正式 WorkMemory/图不得残留。
    assert ctx.backend.snapshot() == backend_before, \
        "held-out floor 评测不得写正式图、身份表或统计表"
    assert ctx.work_memory == work_memory_before, \
        "held-out floor 评测不得残留正式 WorkMemory"

    # measure_floor_activation 端到端（recognition·observed cue 位 token D:11）。
    assert fa.measured is True, "★ FloorActivation.measured=True（recognition 机制层预验）"
    assert fa.total >= 1, "至少 1 cue slot（slot 1·使）计 total"
    assert fa.activated >= 1, "★ cue slot 选 使·cue_rel_of==REL_CAUSES==rel_kind → activated"
    assert fa.activation_permille == 1000 >= LANG_FLOOR_ACTIVATION, \
        "★ activation=1000≥500（C-vs-L 真判别·training-learned D:11 在 held-out cue slot 正确激活）"


# ============ FC10 gate OFF bit-identical（守 SW2/9/16 + orchestrator 入口 gate-gated）============

def test_fc10_gate_off_bit_identical_orchestrator_noop():
    """FC10：FLOOR_ACTIVATION_MODE OFF → orchestrator 不调（caller gate-gated `if gate: _measure_floor_pass`）·
    floor 三参 default（False/0/0）→ weaning.py floor_conjunct gate-gated True（FC7 已测 weaning 层）→
    statistical_ready 不受 floor 影响。本测核 orchestrator **入口** gate-gated 行为（FC7 测 weaning 层·互补）。

    mirror FC7 (a)：gate OFF（fixture default）→ language_statistical_weaning_check 不传 floor 参 →
    default False/0/0 → floor_conjunct gate-gated True → statistical_ready 仍 True（SW2 基线 bit-identical）。
    """
    # fixture default：FLOOR_ACTIVATION_MODE OFF（_gate_reset 守·CI default）
    assert gates.FLOOR_ACTIVATION_MODE is False, "CI default OFF"
    # 模拟 formal_train statistical 路径 gate OFF：不调 orchestrator·传 default floor 三参
    _floor = FloorActivation()   # default measured=False/0/0（gate OFF→orchestrator 不调）
    rep = _sw_check(
        floor_measured=_floor.measured,
        floor_activation_permille=_floor.activation_permille,
        floor_false_positive_permille=_floor.false_positive_permille,
    )
    assert rep.statistical_ready is True, \
        "★ gate OFF → floor_conjunct gate-gated True → SW2 基线 True（bit-identical·守 SW2/9/16）"


# ============ FC11 probe_holdout=0 / probe_set_disjoint=False 早返（审2 LOW-3 + measured-guard）============

def test_fc11_no_probe_corpus_early_return(_prod_gates):
    """FC11（审2 LOW-3 + measured-guard）：probe_corpus 空（probe_holdout=0 default）→ orchestrator
    早返 FloorActivation(measured=False)（defense-in-depth·anchor_heldout 已守 sample-disjoint）。
    防止 vacuous activation（无 held-out→无 measure·measured=False 守 stub-0）。"""
    from pure_integer_ai.experiments.formal_train import _measure_floor_pass

    ctx = make_train_context(DictBackend())
    # (a) probe_corpus 空（probe_holdout=0 default·W4）
    ctx.probe_corpus = []
    ctx.probe_set_disjoint = True   # 即使 disjoint 标·无 held-out 仍早返
    fa_a = _measure_floor_pass(ctx, ctx.backend, ctx.concept_graph)
    assert fa_a.measured is False and fa_a.activation_permille == 0, \
        "★ (a) 无 probe_corpus → 早返 FloorActivation(measured=False)（measured-guard）"

    # (b) probe_set_disjoint=False（probe_holdout=0 default 或隔离失败·defense-in-depth）
    held = type("Held", (), {"modality": 1, "tokens": ["a", "b"], "source": SOURCE_BARE_TEXT,
                             "lang": 1, "domain": 1})()
    ctx.probe_corpus = [held]
    ctx.probe_set_disjoint = False
    fa_b = _measure_floor_pass(ctx, ctx.backend, ctx.concept_graph)
    assert fa_b.measured is False and fa_b.total == 0, \
        "★ (b) probe_set_disjoint=False → 早返（审2 LOW-3 defense-in-depth）"

    # (c) probe_corpus 仅含非语言模态（arith/code）→ lang_probe 空 → 早返
    from pure_integer_ai.cognition.shared.types import MODALITY_ARITH
    ctx.probe_corpus = [type("Held", (), {"modality": MODALITY_ARITH, "tokens": [],
                                          "source": SOURCE_BARE_TEXT, "lang": 0, "domain": 3})()]
    ctx.probe_set_disjoint = True
    fa_c = _measure_floor_pass(ctx, ctx.backend, ctx.concept_graph)
    assert fa_c.measured is False, \
        "★ (c) 无 lang_probe（全非语言模态或 tokens 空）→ 早返（measured-guard）"


# ============ FC12 COMPOSES OFF bit-identical 直守（审1 MEDIUM-1·大路 doc/重来_对应机制生产激活 §7）============

# 大路生产 try/finally 把 COMPOSES_COMBINE_MODE 翻 ON。本测把 COMPOSES **单独** OFF（其余生产 gate 全 ON）
# 直守关键 invariant：COMPOSES 是让 INSTANTIATES 边出现的**唯一因素**——OFF 则 observe.py:204 不建边 +
# formal_train:1748 held-out map 不填 → read_instantiates=None → measured=False。FC9（COMPOSES ON）是对照（fire）。
# 审1 MEDIUM-1：原档 §7 过声称 TC2+FC10 守此（TC2 守 toy-单-item-不聚簇·FC10 不测 COMPOSES）→ 本测补 unguarded gap。
_FC12_GATE_NAMES = tuple(n for n in _PROD_GATE_NAMES if n != "COMPOSES_COMBINE_MODE")


def test_fc12_composes_off_zero_instantiates_bit_identical():
    """FC12（★ 审1 MEDIUM-1·大路 bit-identical 直守）：COMPOSES_COMBINE_MODE 单独 OFF（其余生产 gate 全 ON）+
    FC9 同款 held-out（同训练 shape）→ held-out map 不填（formal_train:1748 gate OFF）+ observe 不建 INSTANTIATES
    （observe.py:204 gate OFF）→ read_instantiates=None → FloorActivation.measured=False。

    与 FC9（COMPOSES ON→INSTANTIATES fire / measured=True）构成干净 A/B：唯一变量 = COMPOSES·
    证"翻 COMPOSES 是让边出现的唯一因素"（gate OFF→零 INSTANTIATES·bit-identical 关键边界直守·非 toy-不聚簇间接）。
    直测受 gate 控制的真实代码点（observe.py:204 写侧 + formal_train:1748 held-out map 写侧）·
    非 capability_exam/formal_train 生产 flip（后者 monkeypatch 绕 flip 困难·orchestrator 路径同 gate 同 site·代表性强）。
    """
    from pure_integer_ai.experiments.formal_train import _measure_floor_pass
    from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_PRECEDES
    from pure_integer_ai.numeric.symbol_domain import OPCODE_NOP
    from pure_integer_ai.crosscut.determinism.hasher import Hasher
    from pure_integer_ai.cognition.shared.types import STAGE_TRAINING

    # 其余生产 gate 全 ON·COMPOSES 单独 OFF（_gate_reset autouse 已守 FLOOR_ACTIVATION_MODE OFF·本测显式翻）
    saved = {name: getattr(gates, name) for name in _PROD_GATE_NAMES}
    for name in _FC12_GATE_NAMES:           # 全 ON（除 COMPOSES）
        setattr(gates, name, True)
    gates.COMPOSES_COMBINE_MODE = False     # ★ 唯一 OFF 变量
    try:
        assert gates.COMPOSES_COMBINE_MODE is False, "★ COMPOSES 单独 OFF（唯一变量）"

        ctx = make_train_context(DictBackend())
        sid = ctx.space_id
        g = ctx.concept_graph

        # FC9 同款 fixture：training-registered skeleton S（OPCODE_NOP + 3 叶 + cue_sig slot 1）+ REALIZES→REL_CAUSES
        skel_root = _ensure(ctx, "__skel_fc12", node_type=NODE_CONCEPT)
        cue_token = _ensure(ctx, "使")
        leaf0 = _ensure(ctx, "__fc12_leaf0")
        leaf2 = _ensure(ctx, "__fc12_leaf2")
        record_composes_attr(ctx.backend, ref=skel_root, kind=ATTR_OPERATOR, int_a=OPCODE_NOP, int_b=0)
        for ti, leaf in enumerate([leaf0, cue_token, leaf2]):
            record_composes_attr(ctx.backend, ref=leaf, kind=ATTR_OPERAND,
                                 int_a=make_variable(ti), int_b=0)
            _add_composes(ctx, skel_root, leaf, ti)
            if ti == 1:
                record_composes_attr(ctx.backend, ref=leaf, kind=ATTR_CUE_SIG,
                                     int_a=cue_token[0], int_b=cue_token[1])
        rel_prims = ensure_relation_primitives(ctx.concept_index, ctx.backend, space_id=sid)
        rel_causes = rel_prims[REL_CAUSES]
        build_realizes_edge(ctx.edge_store, skel_root, rel_causes, space_id=sid)
        _register_lang_skeleton(ctx, skel_root, arity=3)
        record_emergent_relation_signal_shadow(ctx.edge_store, cue_token, rel_causes, space_id=sid)
        ctx.edge_store.set_tier(
            space_id_from=cue_token[0], local_id_from=cue_token[1],
            space_id_to=rel_causes[0], local_id_to=rel_causes[1],
            edge_type=EDGE_RELATION_SIGNAL, new_tier=TIER_PRIMARY)
        # 健全性：training 侧学到的 D:11 + skeleton REALIZES 都在（与 FC9 同·排除"fixture 不够"混淆）
        assert g.cue_rel_of(cue_token) == REL_CAUSES, "training-learned D:11 使→REL_CAUSES（与 FC9 同·非混淆）"
        assert g.rel_kind_of_skeleton(skel_root) == REL_CAUSES, "S REALIZES→REL_CAUSES"

        # FC9 同款 held-out（同 shape·cue "使" slot 1）
        held = CollectedItem(
            tokens=["__fc12_held0", "使", "__fc12_held2"],
            role_seq=[1, 1, 1],
            collect_type=COLLECT_PRECEDES, source=SOURCE_BARE_TEXT,
        )
        ctx.probe_corpus = [held]
        ctx.probe_set_disjoint = True

        fa = _measure_floor_pass(ctx, ctx.backend, ctx.concept_graph)

        # ★ invariant 1：held-out map 不填（formal_train:1748 _held_out_discovery_tally_free gate OFF→skel_by_item_new 空）
        assert held.document_scope_hash not in {
            key[0] for key in ctx.work_memory.lang_skeleton_by_item
        }, \
            "★ COMPOSES OFF → held-out map 不填（formal_train:1748 gate OFF·FC9 ON 时此断言反向）"
        # ★ invariant 2：observe 不建 INSTANTIATES（observe.py:204 gate OFF）→ read_instantiates=None
        held_struct_label = f"__seg_{STAGE_TRAINING}_{Hasher('observe.seg.v1').h63(held.tokens)}"
        held_struct = ctx.concept_index.lookup(held_struct_label, sid)
        if held_struct is not None:
            assert g.read_instantiates(held_struct) is None, \
                "★ COMPOSES OFF → observe.py:204 不建 INSTANTIATES→read_instantiates=None（FC9 ON 时 == skel_root）"
        # ★ invariant 3：全图零 EDGE_INSTANTIATES 边（bit-identical 关键边界直守）
        inst_rows = [row for row in ctx.backend.select("edge", where={"edge_type": EDGE_INSTANTIATES})]
        assert len(inst_rows) == 0, \
            f"★ COMPOSES OFF → 全图零 INSTANTIATES 边（got {len(inst_rows)}·bit-identical 关键边界）"
        # ★ invariant 4：floor 不测（无 INSTANTIATES→read_instantiates=None→floor_measure skip→measured=False）
        assert fa.measured is False, \
            "★ COMPOSES OFF → measured=False（无 INSTANTIATES→floor 不测·与 FC9 ON→measured=True 反向）"
    finally:
        for name, v in saved.items():
            setattr(gates, name, v)
