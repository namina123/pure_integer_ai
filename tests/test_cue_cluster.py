"""Phase E §十八 condition 6a-1 测试（cue 子聚类 dormant 基建）。

_cluster_by_cue（gate-gated sustainable-split 子聚类）+ _shape_name cue_sig（sentinel 编码·bit-identical）+
_collect_cue_sig（镜像 _collect_slot_lcas·读 ATTR_CUE_SIG·load 名重建）。

反 theater：exposure-driven·无 frozenset·"sustainable" 从闭类 cue 重复涌现（是×4/使×4 各≥K 可持续·
内容词各 1 不可持续）·关系 label 走外源 oracle 非读 cue（label_realizes·§十八 condition 6 复合键·禁单 primitive 单射）。
铁律：纯整数 / bit-identical（gate OFF·cue_sig=()→名同今）/ 确定性（sorted roots + sorted 子簇）/ dormant（无 production caller）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_types import EDGE_COMPOSES
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT, EPI_STRUCTURED
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.composes_attr import (
    record_composes_attr, ATTR_OPERATOR, ATTR_OPERAND, ATTR_CUE_SIG,
)
from pure_integer_ai.cognition.process.structure_discover import (
    _cluster_by_cue, _collect_cue_sig, _shape_name,
)
from tests.test_experiments import make_train_context


@pytest.fixture(autouse=True)
def _gate_reset():
    saved = gates.CUE_CLUSTER_MODE
    gates.CUE_CLUSTER_MODE = False
    yield
    gates.CUE_CLUSTER_MODE = saved


def _ensure(ctx, surface):
    return ctx.concept_index.ensure(surface, space_id=ctx.space_id)


def _build_token_tree(ctx, root, tokens):
    """root（ATTR_OPERATOR·internal）→ token concept leaves（无 attr=CONCEPT_LEAF·reading 序）。"""
    record_composes_attr(ctx.backend, ref=root, kind=ATTR_OPERATOR, int_a=0, int_b=0)
    for ti, tok in enumerate(tokens):
        ctx.edge_store.add(
            space_id_from=root[0], local_id_from=root[1],
            space_id_to=tok[0], local_id_to=tok[1],
            edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED, order_index=ti, tier=TIER_PRIMARY,
        )


# ============ _cluster_by_cue（sustainable-split 子聚类） ============

def test_cluster_by_cue_gate_off_no_split():
    """gate OFF → [(roots, ())]·不拆·bit-identical（即使 是×2/使×2 可持续拆）。"""
    ctx = make_train_context(DictBackend())
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    r1 = _ensure(ctx, "__c1"); r2 = _ensure(ctx, "__c2"); r3 = _ensure(ctx, "__c3"); r4 = _ensure(ctx, "__c4")
    _build_token_tree(ctx, r1, [_ensure(ctx, "苹果"), shi])
    _build_token_tree(ctx, r2, [_ensure(ctx, "猫"), shi])
    _build_token_tree(ctx, r3, [_ensure(ctx, "火"), shi3])
    _build_token_tree(ctx, r4, [_ensure(ctx, "铁"), shi3])
    out = _cluster_by_cue(ctx.backend, ctx.concept_graph, [r1, r2, r3, r4])
    assert out == [([r1, r2, r3, r4], ())], "gate OFF→单簇 cue_sig=()"


def test_cluster_by_cue_sustainable_split():
    """★ gate ON + 是×2/使×2（slot1 各≥K）→ 拆 2 子簇·cue_sig[slot1]=是/使·非拆位 None。"""
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    r1 = _ensure(ctx, "__c1"); r2 = _ensure(ctx, "__c2"); r3 = _ensure(ctx, "__c3"); r4 = _ensure(ctx, "__c4")
    _build_token_tree(ctx, r1, [_ensure(ctx, "苹果"), shi])
    _build_token_tree(ctx, r2, [_ensure(ctx, "猫"), shi])
    _build_token_tree(ctx, r3, [_ensure(ctx, "火"), shi3])
    _build_token_tree(ctx, r4, [_ensure(ctx, "铁"), shi3])
    out = _cluster_by_cue(ctx.backend, ctx.concept_graph, [r1, r2, r3, r4])
    assert len(out) == 2, "拆 是/使 两子簇"
    (a_roots, a_sig), (b_roots, b_sig) = out
    assert a_sig[1] in (shi, shi3) and b_sig[1] in (shi, shi3), "cue_sig[slot1]=cue token"
    assert a_sig[1] != b_sig[1], "两子簇异 cue"
    assert a_sig[0] is None and b_sig[0] is None, "非拆位 None（内容词 slot0 不进 cue_sig）"
    # 是子簇={r1,r2}·使子簇={r3,r4}（按 cue token 排序·哪在前由 ConceptRef 定·此处断内容正确非序）
    shi_cluster_roots = a_roots if a_sig[1] == shi else b_roots
    shi3_cluster_roots = b_roots if a_sig[1] == shi else a_roots
    assert sorted(shi_cluster_roots) == sorted([r1, r2]), "是子簇=是-samples"
    assert sorted(shi3_cluster_roots) == sorted([r3, r4]), "使子簇=使-samples"


def test_cluster_by_cue_k2_mixed_no_split():
    """★ K=2 是×1/使×1（总 2 < 2K=4·无 sustainable）→ [(roots, ())]·诚实（样本太少·名不变 bit-identical）。"""
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    r1 = _ensure(ctx, "__m1"); r2 = _ensure(ctx, "__m2")
    _build_token_tree(ctx, r1, [_ensure(ctx, "苹果"), shi])
    _build_token_tree(ctx, r2, [_ensure(ctx, "火"), shi3])
    out = _cluster_by_cue(ctx.backend, ctx.concept_graph, [r1, r2])
    assert out == [([r1, r2], ())], "K=2 mixed → 不拆（每 cue 1<K·sustainable 须每子簇≥K）·cue_sig=()"


def test_cluster_by_cue_all_same_cue_no_split():
    """全同 cue（是×4·slot1 单组 <2 组）→ 不拆·cue_sig=()（无 ≥2 组）。"""
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    shi = _ensure(ctx, "是")
    rs = [_ensure(ctx, f"__s{i}") for i in range(4)]
    for r in rs:
        _build_token_tree(ctx, r, [_ensure(ctx, f"tok_{r[1]}"), shi])
    out = _cluster_by_cue(ctx.backend, ctx.concept_graph, rs)
    assert out == [(sorted(rs), ())], "全同 cue→单组→不拆"


def test_cluster_by_cue_arith_no_concept_leaf_no_split():
    """无 CONCEPT_LEAF（纯算子树·无 token 子）→ 不拆·cue_sig=()。"""
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    r1 = _ensure(ctx, "__op1"); r2 = _ensure(ctx, "__op2")
    for r in (r1, r2):
        record_composes_attr(ctx.backend, ref=r, kind=ATTR_OPERATOR, int_a=0, int_b=0)
    out = _cluster_by_cue(ctx.backend, ctx.concept_graph, [r1, r2])
    assert out == [([r1, r2], ())], "无 CONCEPT_LEAF→不拆"


def test_cluster_by_cue_deterministic_sorted():
    """两次调 + 乱序输入 → 同果（sorted roots + sorted 子簇·bit-identical）。"""
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    rs = []
    for i, cue in enumerate([shi, shi, shi3, shi3]):
        r = _ensure(ctx, f"__d{i}")
        _build_token_tree(ctx, r, [_ensure(ctx, f"t{i}"), cue])
        rs.append(r)
    a = _cluster_by_cue(ctx.backend, ctx.concept_graph, rs)
    b = _cluster_by_cue(ctx.backend, ctx.concept_graph, list(reversed(rs)))
    assert a == b, "乱序输入同果（sorted·确定）"


def test_cluster_by_cue_then_distinct_names():
    """集成：是×2/使×2 拆 → 两子簇 cue_sig → _shape_name 异名（破 cue 坍缩·condition 6a 核心）。"""
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    r1 = _ensure(ctx, "__e1"); r2 = _ensure(ctx, "__e2"); r3 = _ensure(ctx, "__e3"); r4 = _ensure(ctx, "__e4")
    _build_token_tree(ctx, r1, [_ensure(ctx, "苹果"), shi])
    _build_token_tree(ctx, r2, [_ensure(ctx, "猫"), shi])
    _build_token_tree(ctx, r3, [_ensure(ctx, "火"), shi3])
    _build_token_tree(ctx, r4, [_ensure(ctx, "铁"), shi3])
    out = _cluster_by_cue(ctx.backend, ctx.concept_graph, [r1, r2, r3, r4])
    assert len(out) == 2
    sig = (0, -1, -1); arity = 2; asig = ()
    names = {_shape_name(sig, arity, asig, cue_sig) for _roots, cue_sig in out}
    assert len(names) == 2, "两子簇异名（是/使 独立骨架·破 (c) 命门）"


# ============ _shape_name cue_sig（sentinel 编码·bit-identical） ============

def test_shape_name_cue_sig_empty_bit_identical():
    """★ cue_sig=() → 名 == 不传（不加 payload·bit-identical·守跨 run resume）。"""
    sig = (1, -1, -1); arity = 2
    asig = ((1, 5),)
    n_default = _shape_name(sig, arity, asig)          # cue_sig 默认 ()
    n_explicit = _shape_name(sig, arity, asig, ())     # 显式 ()
    assert n_default == n_explicit, "默认 () == 显式 ()"
    assert n_default.startswith("__op_disc_")


def test_shape_name_cue_sig_nonempty_differs():
    """cue_sig 非 () → 名变（异名）。"""
    sig = (1, -1, -1); arity = 2; asig = ()
    n_base = _shape_name(sig, arity, asig)                       # cue_sig=()
    n_cue = _shape_name(sig, arity, asig, (None, (1, 9)))        # slot1=是
    assert n_cue != n_base, "cue_sig 非()→异名"


def test_shape_name_cue_distinguishes_shi_vs_shi3():
    """★ cue_sig 是 vs 使 → 异名（破 (c) 命门·是/使 各产独立骨架）。"""
    sig = (1, -1, -1); arity = 2; asig = ()
    n_shi = _shape_name(sig, arity, asig, (None, (1, 9)))     # slot1=是
    n_shi3 = _shape_name(sig, arity, asig, (None, (1, 10)))   # slot1=使
    assert n_shi != n_shi3, "是 vs 使 → 异名"


def test_shape_name_sentinel_isolates_abstract_and_cue():
    """★ sentinel 隔离：abs=(X,) cue=() vs abs=() cue=(X,) → 异名（避裸 NL+cue 拆碰撞·_CUE_SIG_SEP=-2）。"""
    sig = (1, -1, -1); arity = 2
    X = (1, 7)
    n_abs_only = _shape_name(sig, arity, (X,), ())     # abstract_sig=(X,)·cue_sig=()
    n_cue_only = _shape_name(sig, arity, (), (X,))     # abstract_sig=()·cue_sig=(X,)
    assert n_abs_only != n_cue_only, "sentinel 隔离·abstract 与 cue 不撞名"


def test_shape_name_all_none_cue_normalizes_empty():
    """cue_sig 全 None → _normalize 归一 () → 名 == cue_sig=()（bit-identical·防 (0,0) 泄漏）。"""
    sig = (1, -1, -1); arity = 2; asig = ()
    n_empty = _shape_name(sig, arity, asig, ())
    n_all_none = _shape_name(sig, arity, asig, (None, None))
    assert n_empty == n_all_none, "全 None cue 归一 () → 名同 cue_sig=()"


# ============ _collect_cue_sig（镜像 _collect_slot_lcas·读 ATTR_CUE_SIG） ============

def _build_skel_tree(ctx, root, slot_refs, cue_at=None, cue_ref=None):
    """skeleton-like 树：root ATTR_OPERATOR → slot 叶（ATTR_OPERAND make_variable(i)）·可选 cue_at slot 挂 ATTR_CUE_SIG。"""
    record_composes_attr(ctx.backend, ref=root, kind=ATTR_OPERATOR, int_a=0, int_b=0)
    for ti, tok in enumerate(slot_refs):
        ctx.edge_store.add(
            space_id_from=root[0], local_id_from=root[1],
            space_id_to=tok[0], local_id_to=tok[1],
            edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED, order_index=ti, tier=TIER_PRIMARY,
        )
        record_composes_attr(ctx.backend, ref=tok, kind=ATTR_OPERAND, int_a=ti, int_b=0)
    if cue_at is not None and cue_ref is not None:
        record_composes_attr(ctx.backend, ref=slot_refs[cue_at],
                             kind=ATTR_CUE_SIG, int_a=cue_ref[0], int_b=cue_ref[1])


def test_collect_cue_sig_reads_attr():
    """★ skeleton slot1 ATTR_CUE_SIG=是 → 读回 (None, 是)（slot 序·对齐 build CONCEPT_LEAF）。"""
    ctx = make_train_context(DictBackend())
    root = _ensure(ctx, "__skel")
    slot0 = _ensure(ctx, "__s0"); slot1 = _ensure(ctx, "__s1")
    shi = _ensure(ctx, "是")
    _build_skel_tree(ctx, root, [slot0, slot1], cue_at=1, cue_ref=shi)
    cue = _collect_cue_sig(ctx.backend, ctx.concept_graph, root)
    assert cue == (None, shi), "slot0 无 ATTR_CUE_SIG→None·slot1→是"


def test_collect_cue_sig_no_attr_all_none():
    """无 ATTR_CUE_SIG → 全 None（→ _normalize 归一 ()·load 名同今 bit-identical）。"""
    ctx = make_train_context(DictBackend())
    root = _ensure(ctx, "__skel2")
    slot0 = _ensure(ctx, "__s0"); slot1 = _ensure(ctx, "__s1")
    _build_skel_tree(ctx, root, [slot0, slot1])
    cue = _collect_cue_sig(ctx.backend, ctx.concept_graph, root)
    assert cue == (None, None), "无 ATTR_CUE_SIG→全 None"


def test_collect_cue_sig_round_trip_name():
    """★ build 名（cue_sig 非()）== load 名（_collect_cue_sig 重建喂 _shape_name）·修 cue_sig 版 B6 Bug 1。"""
    ctx = make_train_context(DictBackend())
    sig = (1, -1, -1); arity = 2; asig = ()
    shi = _ensure(ctx, "是")
    build_name = _shape_name(sig, arity, asig, (None, shi))   # BUILD 端 cue_sig
    root = _ensure(ctx, "__skel_rt")
    slot0 = _ensure(ctx, "__rt0"); slot1 = _ensure(ctx, "__rt1")
    _build_skel_tree(ctx, root, [slot0, slot1], cue_at=1, cue_ref=shi)
    loaded_cue = _collect_cue_sig(ctx.backend, ctx.concept_graph, root)
    load_name = _shape_name(sig, arity, asig, loaded_cue)
    assert build_name == load_name, "build 名 == load 名（_collect_cue_sig 重建 cue_sig）"
    assert loaded_cue == (None, shi)


# ============ post-impl 审补测（5 盲点·impl 已正确·guard 防回归） ============

def test_cluster_by_cue_multi_sustainable_lowest_slot_wins():
    """多 sustainable 位（slot0 是/使 + slot1 苹果/猫 都可持续）→ 取首个（最低 slot0）·单拆·slot1 不进 cue_sig。"""
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    pg = _ensure(ctx, "苹果"); mao = _ensure(ctx, "猫")
    rs = [_ensure(ctx, f"__u{i}") for i in range(4)]
    _build_token_tree(ctx, rs[0], [shi, pg])
    _build_token_tree(ctx, rs[1], [shi, mao])
    _build_token_tree(ctx, rs[2], [shi3, pg])
    _build_token_tree(ctx, rs[3], [shi3, mao])
    out = _cluster_by_cue(ctx.backend, ctx.concept_graph, rs)
    assert len(out) == 2, "拆 slot0（是/使）"
    for _roots, cue_sig in out:
        assert cue_sig[0] in (shi, shi3), "cue_sig[slot0]=cue（首位拆）"
        assert cue_sig[1] is None, "slot1 不进 cue_sig（单拆·取首 sustainable 位）"


def test_cluster_by_cue_partial_sustainable_singleton_remainder():
    """是×2/使×2/凭×1（凭<K）→ 修法 A：big=[(是,2),(使,2)]≥2 拆·凭单例合并 cue_sig=() 余簇。
    3 子簇：是-sub cue_sig=(None,是)·使-sub cue_sig=(None,使)·余簇[凭] cue_sig=()。
    原 stray-veto(all≥K 含单例永不触发)是 bug——docstring 形式判据"≥2 组各≥K"与语义例子(是×4/使×4 +
    苹果/猫 各 1 不阻断)自相矛盾·修法 A 对齐语义意图(big-groups)·2 审 APPROVE·审1 F1 确认此测必破须更新。"""
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使"); ping = _ensure(ctx, "凭")
    rs = [_ensure(ctx, f"__p{i}") for i in range(5)]
    cues = [shi, shi, shi3, shi3, ping]
    for r, c in zip(rs, cues):
        _build_token_tree(ctx, r, [_ensure(ctx, f"t{r[1]}"), c])
    out = _cluster_by_cue(ctx.backend, ctx.concept_graph, rs)
    assert len(out) == 3, "拆 3 子簇（是-sub + 使-sub + 凭余簇）"
    # 按 cue_sig[1] 分类（None=余簇 cue_sig=()）
    found: dict = {}
    for roots, cue_sig in out:
        found.setdefault(cue_sig[1], []).extend(roots)
    assert shi in found and shi3 in found and None in found, "是-sub + 使-sub + 余簇 cue_sig=()"
    assert sorted(found[shi]) == sorted([rs[0], rs[1]]), "是-sub=是 samples"
    assert sorted(found[shi3]) == sorted([rs[2], rs[3]]), "使-sub=使 samples"
    assert sorted(found[None]) == [rs[4]], "余簇=凭 singleton（cue_sig=()·走 uncued discover·同修法前）"


def test_cluster_by_cue_three_slots_length():
    """3-slot 骨架 → cue_sig 长度 3·非拆位 None（验 per-position 长度一致）。"""
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    rs = [_ensure(ctx, f"__v{i}") for i in range(4)]
    _build_token_tree(ctx, rs[0], [_ensure(ctx, "a"), shi, _ensure(ctx, "x")])
    _build_token_tree(ctx, rs[1], [_ensure(ctx, "b"), shi, _ensure(ctx, "y")])
    _build_token_tree(ctx, rs[2], [_ensure(ctx, "c"), shi3, _ensure(ctx, "z")])
    _build_token_tree(ctx, rs[3], [_ensure(ctx, "d"), shi3, _ensure(ctx, "w")])
    out = _cluster_by_cue(ctx.backend, ctx.concept_graph, rs)
    assert len(out) == 2
    for _roots, cue_sig in out:
        assert len(cue_sig) == 3, "3-slot cue_sig 长度 3"
        assert cue_sig[1] in (shi, shi3) and cue_sig[0] is None and cue_sig[2] is None


def test_collect_cue_sig_no_cue_round_trip_bit_identical():
    """★ 无 cue skeleton：BUILD 名（cue_sig=()）== LOAD 名（_collect_cue_sig 全 None→()）·跨 run bit-identical 不变。"""
    ctx = make_train_context(DictBackend())
    sig = (1, -1, -1); arity = 2; asig = ()
    build_name = _shape_name(sig, arity, asig)   # cue_sig=() 默认
    root = _ensure(ctx, "__skel_nc")
    slot0 = _ensure(ctx, "__nc0"); slot1 = _ensure(ctx, "__nc1")
    _build_skel_tree(ctx, root, [slot0, slot1])   # 无 ATTR_CUE_SIG
    loaded_cue = _collect_cue_sig(ctx.backend, ctx.concept_graph, root)
    assert loaded_cue == (None, None), "无 ATTR_CUE_SIG→全 None"
    load_name = _shape_name(sig, arity, asig, loaded_cue)
    assert build_name == load_name, "★ 无 cue：BUILD==LOAD 名（_normalize 全 None→()·bit-identical·跨 run 不变）"


def test_shape_name_abstract_and_cue_both_nonempty():
    """abs=(X,) + cue=(Y,) → 与仅 abs / 仅 cue 三路异名（sentinel + 多维区分）。"""
    sig = (1, -1, -1); arity = 2
    X = (1, 7); Y = (1, 9)
    n_abs_only = _shape_name(sig, arity, (X,), ())
    n_cue_only = _shape_name(sig, arity, (), (Y,))
    n_both = _shape_name(sig, arity, (X,), (Y,))
    assert len({n_abs_only, n_cue_only, n_both}) == 3, "abs-only / cue-only / both 三路异名"


# ============ 6a-2 e2e（auto_discover gate ON 产 是/使 异骨架·破 cue 坍缩） ============

def test_auto_discover_gate_off_collapses_e2e():
    """gate OFF·是×2/使×2（same shape）→ auto_discover 产 1 骨架（坍缩·现状 bit-identical）。"""
    from pure_integer_ai.cognition.process.structure_discover import auto_discover_operators
    ctx = make_train_context(DictBackend())
    # gate OFF（fixture 默认）
    sid = ctx.space_id
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    roots = []
    for i, (content, cue) in enumerate([("苹果", shi), ("猫", shi), ("火", shi3), ("铁", shi3)]):
        r = ctx.concept_index.ensure(f"__off_{i}", space_id=sid)
        _build_token_tree(ctx, r, [_ensure(ctx, content), cue])
        roots.append(r)
    discovered = auto_discover_operators(
        roots, concept_index=ctx.concept_index, edge_store=ctx.edge_store,
        backend=ctx.backend, space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(discovered) == 1, "gate OFF→1 骨架（是/使 坍缩·bit-identical 现状）"


def test_auto_discover_gate_on_splits_shi_shi3_e2e():
    """★ e2e：gate ON·是×2/使×2（same shape）→ auto_discover 产 2 异名骨架（破 cue 坍缩）·各 cue slot 挂 ATTR_CUE_SIG。"""
    from pure_integer_ai.cognition.process.structure_discover import auto_discover_operators
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    roots = []
    for i, (content, cue) in enumerate([("苹果", shi), ("猫", shi), ("火", shi3), ("铁", shi3)]):
        r = ctx.concept_index.ensure(f"__on_{i}", space_id=sid)
        _build_token_tree(ctx, r, [_ensure(ctx, content), cue])
        roots.append(r)
    discovered = auto_discover_operators(
        roots, concept_index=ctx.concept_index, edge_store=ctx.edge_store,
        backend=ctx.backend, space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(discovered) == 2, "★ gate ON→是/使 拆 2 骨架（破 cue 坍缩）"
    assert len({op.name for op in discovered}) == 2, "两骨架异名"
    # 各骨架 cue slot（slot1）挂 ATTR_CUE_SIG·_collect_cue_sig 读回 (None, 是/使)
    cues = {_collect_cue_sig(ctx.backend, ctx.concept_graph, op.skeleton_ref) for op in discovered}
    assert cues == {(None, shi), (None, shi3)}, "★ 两骨架 cue_sig=(None,是)/(None,使)·ATTR_CUE_SIG 落盘可读回"


def test_auto_discover_gate_on_load_round_trip_e2e():
    """★ gate ON build → load 名匹配（_collect_cue_sig 重建 cue_sig·修 cue_sig 版 B6 Bug 1）。"""
    from pure_integer_ai.cognition.process.structure_discover import auto_discover_operators, load_discovered_operators
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    roots = []
    for i, (content, cue) in enumerate([("梨", shi), ("桃", shi), ("烟", shi3), ("灰", shi3)]):
        r = ctx.concept_index.ensure(f"__rt_{i}", space_id=sid)
        _build_token_tree(ctx, r, [_ensure(ctx, content), cue])
        roots.append(r)
    built = auto_discover_operators(
        roots, concept_index=ctx.concept_index, edge_store=ctx.edge_store,
        backend=ctx.backend, space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(built) == 2
    built_names = {op.name for op in built}
    # load 重建（模拟 resume）→ 名须匹配 build 名（ATTR_CUE_SIG 经 _collect_cue_sig 重建 cue_sig）
    loaded = load_discovered_operators(ctx.backend, space_id=sid)
    loaded_names = {op.name for op in loaded}
    assert built_names == loaded_names, "★ build 名 == load 名（cue_sig 经 _collect_cue_sig 重建·修 B6 Bug 1）"


# ============ 6a-3 e2e（route 4-tuple 键 + recognize cue 感知·闭命门1+2） ============

def test_route_cue_key_distinguishes_resume():
    """★ 闭命门1：gate ON·resume 载 是-skeleton（仅）·喂 是×2/使×2 新样本 →
    是 recognize（cue 匹配）·使 discover（4-tuple cue 异键·不丢）。3-tuple 键会使 使 误判已载→全 recognize→丢。"""
    from pure_integer_ai.cognition.process.structure_discover import (
        auto_discover_operators, route_samples_for_discovery,
        _collect_slot_lcas, _normalize_abstract_sig, shape_signature)
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    # run N：是×2/使×2 → auto_discover 建 是-op + 使-op（sustainable split）
    rn = []
    for i, (c, cue) in enumerate([("苹果", shi), ("猫", shi), ("火", shi3), ("铁", shi3)]):
        r = ctx.concept_index.ensure(f"__rn_{i}", space_id=sid)
        _build_token_tree(ctx, r, [_ensure(ctx, c), cue])
        rn.append(r)
    built = auto_discover_operators(
        rn, concept_index=ctx.concept_index, edge_store=ctx.edge_store,
        backend=ctx.backend, space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(built) == 2
    # 取 是-skeleton（cue_sig slot1=是）·resume 模拟"是 已载·使 未载"
    shi_op = next(op for op in built
                  if _collect_cue_sig(ctx.backend, ctx.concept_graph, op.skeleton_ref)[1] == shi)
    op_sig = tuple(shape_signature(ctx.concept_graph, shi_op.skeleton_ref))
    op_asig = _normalize_abstract_sig(_collect_slot_lcas(ctx.backend, ctx.concept_graph, shi_op.skeleton_ref))
    op_cue = _normalize_abstract_sig(_collect_cue_sig(ctx.backend, ctx.concept_graph, shi_op.skeleton_ref))
    assert op_cue != (), "是-skeleton cue_sig 非()（gate ON 写 ATTR_CUE_SIG）"
    existing_keys = {(op_sig, shi_op.arity, op_asig, op_cue)}
    existing_sigs = {op_sig}
    # run N+1：喂 是×2/使×2 新内容样本（梨/桃=是·烟/灰=使）
    rnp = []
    for i, (c, cue) in enumerate([("梨", shi), ("桃", shi), ("烟", shi3), ("灰", shi3)]):
        r = ctx.concept_index.ensure(f"__rnp_{i}", space_id=sid)
        _build_token_tree(ctx, r, [_ensure(ctx, c), cue])
        rnp.append(r)
    discover_roots, recognize_roots = route_samples_for_discovery(
        ctx.backend, ctx.concept_graph, rnp,
        existing_keys=existing_keys, existing_sigs=existing_sigs, space_id=sid)
    # ★ 是（梨,桃）cue 匹配 是-skeleton → recognize·使（烟,灰）cue 异键 → discover（不丢·闭命门1）
    assert len(discover_roots) == 2, "★ 使 新样本进 discover（4-tuple cue 键不共享·闭命门1·3-tuple 会使丢）"
    assert len(recognize_roots) == 2, "是 新样本进 recognize（cue 匹配 是-skeleton）"


def test_route_cue_key_gate_off_bit_identical_3tuple_equiv():
    """gate OFF·resume 载 op（cue_sig=()）·喂 是×2/使×2 → cue 不拆→单簇 cue_sig=()·
    键 (sig,arity,(),())·4-tuple 第4维恒 ()·与 3-tuple 路由行为逐字等价·bit-identical。"""
    from pure_integer_ai.cognition.process.structure_discover import (
        auto_discover_operators, route_samples_for_discovery,
        _collect_slot_lcas, _normalize_abstract_sig, shape_signature)
    ctx = make_train_context(DictBackend())
    # gate OFF（fixture 默认）→ 是/使 坍缩 1 骨架（cue_sig=()）
    sid = ctx.space_id
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    rn = []
    for i, (c, cue) in enumerate([("苹果", shi), ("猫", shi), ("火", shi3), ("铁", shi3)]):
        r = ctx.concept_index.ensure(f"__bo_{i}", space_id=sid)
        _build_token_tree(ctx, r, [_ensure(ctx, c), cue])
        rn.append(r)
    built = auto_discover_operators(
        rn, concept_index=ctx.concept_index, edge_store=ctx.edge_store,
        backend=ctx.backend, space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(built) == 1, "gate OFF→1 坍缩骨架"
    op = built[0]
    op_sig = tuple(shape_signature(ctx.concept_graph, op.skeleton_ref))
    op_asig = _normalize_abstract_sig(_collect_slot_lcas(ctx.backend, ctx.concept_graph, op.skeleton_ref))
    op_cue = _normalize_abstract_sig(_collect_cue_sig(ctx.backend, ctx.concept_graph, op.skeleton_ref))
    assert op_cue == (), "gate OFF→cue_sig=()（无 ATTR_CUE_SIG）"
    existing_keys = {(op_sig, op.arity, op_asig, op_cue)}   # 4-tuple 第4维 ()
    existing_sigs = {op_sig}
    rnp = []
    for i, (c, cue) in enumerate([("梨", shi), ("桃", shi), ("烟", shi3), ("灰", shi3)]):
        r = ctx.concept_index.ensure(f"__bop_{i}", space_id=sid)
        _build_token_tree(ctx, r, [_ensure(ctx, c), cue])
        rnp.append(r)
    discover_roots, recognize_roots = route_samples_for_discovery(
        ctx.backend, ctx.concept_graph, rnp,
        existing_keys=existing_keys, existing_sigs=existing_sigs, space_id=sid)
    # gate OFF：全坍缩 1 簇 cue_sig=() → 键匹配 → 全 recognize（bit-identical·同 3-tuple 行为）
    assert len(discover_roots) == 0, "gate OFF→cue 不拆→键匹配→全 recognize（无 discover）"
    assert len(recognize_roots) == 4, "全 4 样本 recognize（4-tuple 第4维恒()·等价 3-tuple）"


def test_recognize_cue_aware_shi_input():
    """★ 闭命门2：gate ON·是-skeleton + 使-skeleton·recognize '梨 是' → 仅 是-skeleton 命中（使-skeleton cue 拒）。
    cue-blind（6a-3 前）使 两骨架都命中→2 Recognition（双计·假阳）。"""
    from pure_integer_ai.cognition.process.structure_discover import (
        auto_discover_operators, recognize_operators)
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    roots = []
    for i, (c, cue) in enumerate([("苹果", shi), ("猫", shi), ("火", shi3), ("铁", shi3)]):
        r = ctx.concept_index.ensure(f"__rc_{i}", space_id=sid)
        _build_token_tree(ctx, r, [_ensure(ctx, c), cue])
        roots.append(r)
    discovered = auto_discover_operators(
        roots, concept_index=ctx.concept_index, edge_store=ctx.edge_store,
        backend=ctx.backend, space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(discovered) == 2
    # held-out 输入：梨 是（cue=是·新内容 token）
    inp = ctx.concept_index.ensure("__inp_shi", space_id=sid)
    _build_token_tree(ctx, inp, [_ensure(ctx, "梨"), shi])
    recs = recognize_operators([inp], discovered_operators=discovered,
                               backend=ctx.backend, space_id=sid)
    assert len(recs) == 1, "★ 仅 是-skeleton 命中（使-skeleton ATTR_CUE_SIG 拒·闭命门2·破 cue-blind 双计）"
    matched = next(op for op in discovered if op.name == recs[0].operator_name)
    assert _collect_cue_sig(ctx.backend, ctx.concept_graph, matched.skeleton_ref)[1] == shi, "命中 是-skeleton"


def _add_isa(ctx, child, parent):
    """加 EDGE_IS_A child→parent（build_isa_ancestor_map 读全 EDGE_IS_A·任意 source）。"""
    from pure_integer_ai.storage.edge_types import EDGE_IS_A
    from pure_integer_ai.storage.edge_store import SOURCE_CONCEPTNET
    ctx.edge_store.add(
        space_id_from=child[0], local_id_from=child[1],
        space_id_to=parent[0], local_id_to=parent[1],
        edge_type=EDGE_IS_A, strength=1, source=SOURCE_CONCEPTNET,
        epistemic_origin=EPI_STRUCTURED, tier=TIER_PRIMARY,
    )


def test_auto_discover_multi_lca_cue_cross():
    """多 LCA 类（动物/物质）× cue（是/使）交叉 → 同 LCA 簇内 cue 拆（动物-是 ≠ 动物-使）。
    验 LCA 聚类（外层 _cluster_by_lca）+ cue 子聚类（内层 _cluster_by_cue）双层交互·非坍缩。
    是/使 共祖 词（否则 LCA 已按 cue 分离→cue 拆不触发·测须 共祖才命中 cue 拆路径）。"""
    from pure_integer_ai.cognition.process.structure_discover import (
        auto_discover_operators, _collect_slot_lcas, _normalize_abstract_sig)
    ctx = make_train_context(DictBackend())
    gates.CUE_CLUSTER_MODE = True
    sid = ctx.space_id
    shi = _ensure(ctx, "是"); shi3 = _ensure(ctx, "使")
    dongwu = _ensure(ctx, "动物"); wuzhi = _ensure(ctx, "物质"); ci = _ensure(ctx, "词")
    # IS_A：内容词→类（动物/物质）·是/使→词（共祖·使 cue 拆在 LCA 簇内触发）
    for tok in ["猫", "狗", "兔", "鼠"]:
        _add_isa(ctx, _ensure(ctx, tok), dongwu)
    for tok in ["石头", "沙", "烟", "灰"]:
        _add_isa(ctx, _ensure(ctx, tok), wuzhi)
    _add_isa(ctx, shi, ci); _add_isa(ctx, shi3, ci)
    # 8 样本：动物{猫,狗=是·兔,鼠=使}·物质{石头,沙=是·烟,灰=使}·内容 token 各 1（slot0 不可持续→cue 拆落 slot1）
    roots = []
    for i, (c, cue) in enumerate([("猫", shi), ("狗", shi), ("兔", shi3), ("鼠", shi3),
                                  ("石头", shi), ("沙", shi), ("烟", shi3), ("灰", shi3)]):
        r = ctx.concept_index.ensure(f"__ml_{i}", space_id=sid)
        _build_token_tree(ctx, r, [_ensure(ctx, c), cue])
        roots.append(r)
    discovered = auto_discover_operators(
        roots, concept_index=ctx.concept_index, edge_store=ctx.edge_store,
        backend=ctx.backend, space_id=sid, source=SOURCE_BARE_TEXT)
    # 收集 (abstract_sig, cue_sig) 对
    pairs = set()
    for op in discovered:
        asig = _normalize_abstract_sig(_collect_slot_lcas(ctx.backend, ctx.concept_graph, op.skeleton_ref))
        csig = _normalize_abstract_sig(_collect_cue_sig(ctx.backend, ctx.concept_graph, op.skeleton_ref))
        pairs.add((asig, csig))
    # ★ 动物 LCA 簇内 cue 拆：同 asig（含 动物）异 csig（是/使）≥2 → 证明 LCA 内 cue 拆非坍缩
    dongwu_pairs = [p for p in pairs if dongwu in p[0]]
    assert len(dongwu_pairs) >= 2, f"动物 LCA 簇内 cue 拆（是/使 异 cue_sig）·got {dongwu_pairs}"
    # 物质同理
    wuzhi_pairs = [p for p in pairs if wuzhi in p[0]]
    assert len(wuzhi_pairs) >= 2, f"物质 LCA 簇内 cue 拆·got {wuzhi_pairs}"
    # 动物-是 与 动物-使 异名（cue_sig 异·_shape_name 异名）→ 各独立骨架
    dongwu_cues = {csig for asig, csig in dongwu_pairs}
    assert any(c[1] == shi for c in dongwu_cues if len(c) >= 2), "动物-是 存在"
    assert any(c[1] == shi3 for c in dongwu_cues if len(c) >= 2), "动物-使 存在"


