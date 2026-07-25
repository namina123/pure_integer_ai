"""Stage 12 验收门测试：钥匙①语言结构发现·分片第一刀（件1+件2+D3·S3·doc/重来_钥匙①语言结构发现机制设计_修正分析七.md）。

钥匙①=语言结构发现=arith discover_skeleton 的语言对偶。本 session 分片=表示层地基（件1 语言 observe
建 COMPOSES 序 + 件2 _is_concept_leaf 无属性叶 + build/_probe_walk CONCEPT_LEAF 分支全参化）+ 抽象层
基建（D3 IS_A LCA 查询·第二刀前置）。caller defer（formal_train 语言发现触发·下 session）。

覆盖：
  件1：语言段 observe 建 COMPOSES（NOP root + token 叶=无属性 + 出边 order_index）/ 幂等 / read round-trip /
       concept_ref 稳定（within-sample 同一性地基）
  件2：_is_concept_leaf（无属性叶）/ discover 两语言样本→骨架（全 PARAM 无固定位）/ within-sample 同 ref
       同槽（"猫追猫"）/ cross-sample 异 ref 同槽（语言泛化牙·D2 弱化门极致）/ drift（probe==discover）
  D3：build_isa_ancestor_map（IS_A 祖先集）/ common_is_a_ancestor（LCA·diamond tiebreak·方向·无交集 None）
  反 theater：裸概念点无 COMPOSES→None vs 有→骨架（真行为变·非纸面）

铁律：纯整数 / 确定性 bit-identical / 单向依赖 / 反 theater（件1+件2 e2e 真行为变·D3 独立单测·caller defer）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import (
    EdgeStore, SOURCE_BARE_TEXT, SOURCE_CONCEPTNET, EPI_STRUCTURED,
)
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.spaces.memory_space import MemorySpace
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.storage.edge_types import EDGE_COMPOSES, EDGE_IS_A, EDGE_REALIZES
from pure_integer_ai.storage.composes_attr import (
    register_composes_attr, record_composes_attr, read_composes_attrs,
    ATTR_OPERATOR, ATTR_OPERAND, ATTR_ORIGIN, ORIGIN_DISCOVERED, ATTR_SLOT_ROLE,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import (
    ConceptRef, InputPayload, Segment, SpaceContext,
    STAGE_TRAINING, MODALITY_LANGUAGE, WEANING_PRE, LANG_ZH, DOMAIN_TEXT,
)
from pure_integer_ai.cognition.understanding.observe import ObservePipeline
from pure_integer_ai.cognition.understanding.is_a import build_is_a_edge, EPI_CUE
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.process.structure_discover import (
    discover_skeleton, probe_arity, _is_concept_leaf,
    auto_discover_operators, _shape_name, _normalize_abstract_sig,
    load_discovered_operators, route_samples_for_discovery, _collect_slot_lcas,
    _collect_cue_sig, shape_signature,
)
from pure_integer_ai.cognition.process.abstraction import (
    build_isa_ancestor_map, common_is_a_ancestor, set_lca,
)
from pure_integer_ai.numeric.symbol_domain import OPCODE_NOP


# ---- fixtures ----

@pytest.fixture
def disc_env():
    """手建语言 COMPOSES 样本环境（镜像 test_stage9 disc_env·dict backend）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    g = ConceptGraph(b)
    sid = sp.space_id
    yield b, sid, es, ci, g
    b.close()


@pytest.fixture(params=["dict", "sqlite"])
def ctx(request):
    """ObservePipeline 环境（件1 e2e·镜像 test_stage3 ctx）。"""
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    reg = SpaceRegistry(b)
    core = AbstractSpace.create(reg, "core")
    mem_read = MemorySpace.create(reg, "mem_read")
    mem_interact = MemorySpace.create(reg, "mem_interact")
    comp = CompanionSpace.create(reg, "comp1")
    c = SpaceContext(
        core=core, memory_read=mem_read, memory_interact=mem_interact,
        companion=comp, stage=STAGE_TRAINING, memory_active=False,
        weaning_phase=WEANING_PRE,
    )
    yield c
    b.close()


def _seg(tokens, **kw):
    """Segment helper（镜像 test_stage3 _seg）。"""
    return Segment(seg_id=0, modality=MODALITY_LANGUAGE, lang=LANG_ZH,
                   domain=DOMAIN_TEXT, tokens=tokens, **kw)


def _build_lang(disc_env, tokens: list[str], *, seg_label: str) -> ConceptRef:
    """手建语言段 COMPOSES 树（镜像件1 observe 建·NOP root + token 叶=无属性·order_index=token 序位）。

    token 叶 = concept_index.ensure 的 ConceptRef（不挂 attr·"无属性叶"判定·件2 _is_concept_leaf）。
    返 root=struct_ref（NOP SEQ 容器）。
    """
    b, sid, es, ci, _ = disc_env
    root_ref = ci.ensure(seg_label, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    record_composes_attr(b, ref=root_ref, kind=ATTR_OPERATOR, int_a=OPCODE_NOP)
    for ti, tok in enumerate(tokens):
        tok_ref = ci.ensure(tok, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        es.add(space_id_from=root_ref[0], local_id_from=root_ref[1],
               space_id_to=tok_ref[0], local_id_to=tok_ref[1],
               edge_type=EDGE_COMPOSES, strength=1, source=SOURCE_BARE_TEXT,
               epistemic_origin=EPI_STRUCTURED, order_index=ti)
    return root_ref


# ============ concept_ref 稳定（件2 within-sample 同槽地基） ============

def test_concept_ref_stable_within_sample(disc_env):
    """concept_ref 稳定：同 token 同 space→同 ConceptRef（within-sample 同一性地基·concept_index 幂等·件2 同槽守）。"""
    b, sid, es, ci, _ = disc_env
    r1 = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    r2 = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    assert r1 == r2, "同 token 同 space → 同 ConceptRef（concept_index 幂等·件2 within-sample 同槽地基）"


# ============ 件2：_is_concept_leaf + build CONCEPT_LEAF 分支（全参化） ============

def test_is_concept_leaf_empty_attrs():
    """件2 _is_concept_leaf：空属性 dict=语言 token 叶（无 OPERATOR/CTRL/OPERAND/IMMEDIATE/STORE）。"""
    assert _is_concept_leaf({}) is True
    assert _is_concept_leaf({1: (0, 0)}) is False   # 有 ATTR_OPERATOR→非概念叶
    assert _is_concept_leaf({4: (3, 1)}) is False    # 有 ATTR_IMMEDIATE→非概念叶


def test_discover_two_lang_samples_skeleton_all_param(disc_env):
    """件2：两语言样本 → 骨架·全 PARAM 槽无固定位（语言 token 永远参化）。arity=distinct sample0 ref 数。

    "猫追狗"+"鸡咬鸭"：sample0 [猫,追,狗] distinct ref=3 → arity=3（全 PARAM·无固定位）。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build_lang(disc_env, ["猫", "追", "狗"], seg_label="__seg_a")
    s2 = _build_lang(disc_env, ["鸡", "咬", "鸭"], seg_label="__seg_b")
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_BARE_TEXT, skeleton_label="svo")
    assert res is not None, "两语言样本须抽骨架（件1+件2 e2e·反 theater 真行为变）"
    assert res.arity == 3, "sample0 3 distinct token ref → arity=3（全 PARAM 无固定位）"
    attrs = read_composes_attrs(b, res.skeleton_ref)
    assert attrs.get(ATTR_ORIGIN) == (ORIGIN_DISCOVERED, 0), "骨架 root 标 ATTR_ORIGIN=discovered"


def test_discover_lang_within_sample_same_ref_same_slot(disc_env):
    """件2 within-sample 同 ref→同槽（DAG 共享：'猫追猫' 两猫同 ConceptRef→同 PARAM 槽·arity=2 非 3）。"""
    b, sid, es, ci, _ = disc_env
    s1 = _build_lang(disc_env, ["猫", "追", "猫"], seg_label="__seg_a")   # 两猫同 ref
    s2 = _build_lang(disc_env, ["狗", "咬", "狗"], seg_label="__seg_b")   # 两狗同 ref
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_BARE_TEXT, skeleton_label="self")
    assert res is not None and res.arity == 2, "两猫同 ref→DAG 共享同槽·arity=2（猫 slot0·追 slot1）"


def test_discover_lang_cross_sample_diff_ref_allowed(disc_env):
    """件2 cross-sample 全放开（D2 弱化门极致）：'猫追狗'+'鸡咬鸭' 同 slot 允许异 ref（语言泛化牙）。

    sample0 slot0=猫·sample1 同位=鸡（异 ref）→允许（语言 PARAM=词槽·开放词表·非形参实例）。
    若用 operand cross-sample 一致性门（拆分拒）会 raise _NoSkeleton·语言全放开故通过。
    """
    b, sid, es, ci, _ = disc_env
    s1 = _build_lang(disc_env, ["猫", "追", "狗"], seg_label="__seg_a")
    s2 = _build_lang(disc_env, ["鸡", "咬", "鸭"], seg_label="__seg_b")
    # cross-sample 全位异 ref（猫≠鸡·追≠咬·狗≠鸭）·语言允许（operand 门会拒·CONCEPT_LEAF 全放开）
    res = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_BARE_TEXT, skeleton_label="xver")
    assert res is not None and res.arity == 3, "cross-sample 全位异 ref 允许（语言泛化牙·D2 弱化门极致）"


def test_probe_arity_matches_discover_lang(disc_env):
    """件2 drift 防线：probe_arity == discover_skeleton.arity（语言 CONCEPT_LEAF 语料·twin 不漂移）。"""
    b, sid, es, ci, _ = disc_env
    cases = [
        (["猫", "追", "狗"], ["鸡", "咬", "鸭"]),    # arity=3
        (["猫", "追", "猫"], ["狗", "咬", "狗"]),    # arity=2（两同 ref）
        (["跑"], ["跳"]),                            # arity=1
    ]
    for toks0, toks1 in cases:
        s1 = _build_lang(disc_env, toks0, seg_label="__seg_a")
        s2 = _build_lang(disc_env, toks1, seg_label="__seg_b")
        probed = probe_arity(b, [s1, s2])
        disc = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                                 space_id=sid, source=SOURCE_BARE_TEXT,
                                 skeleton_label="drift")
        disc_arity = disc.arity if disc is not None else None
        assert probed == disc_arity, (
            f"drift！probe={probed} != discover={disc_arity} @ {toks0}/{toks1}（twin 须等）")


def test_no_composes_no_skeleton(disc_env):
    """反 theater：裸概念点（无 COMPOSES 子树）→ discover_skeleton None（非伪骨架）。

    对照件1+件2 e2e（有 COMPOSES→骨架）。无属性叶但无 COMPOSES 边=root 是孤立概念点非程序根。
    """
    b, sid, es, ci, _ = disc_env
    bare = ci.ensure("__bare_concept__", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    # 不建 COMPOSES 出边（裸概念点·非语言段 observe 产物）
    res = discover_skeleton([bare, bare], concept_index=ci, edge_store=es, backend=b,
                            space_id=sid, source=SOURCE_BARE_TEXT, skeleton_label="bare")
    # bare 无 COMPOSES 子树→read_composes_tree 空 children→build 越界（root 无属性非算子）→None
    assert res is None, "裸概念点无 COMPOSES → None（反 theater·非伪骨架）"


# ============ D3：IS_A 抽象层 LCA 查询基建（第二刀前置） ============

def test_build_isa_ancestor_map(disc_env):
    """D3 build_isa_ancestor_map：IS_A 链→祖先集（A⊂B⊂C → A 祖先={B,C}·transitive_closure 激活）。"""
    b, sid, es, ci, _ = disc_env
    A = ci.ensure("A", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    B = ci.ensure("B", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    C = ci.ensure("C", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_is_a_edge(es, A, B, source=SOURCE_BARE_TEXT, epistemic=EPI_CUE, space_id=sid)  # A⊂B
    build_is_a_edge(es, B, C, source=SOURCE_BARE_TEXT, epistemic=EPI_CUE, space_id=sid)  # B⊂C
    amap = build_isa_ancestor_map(b, space_id=sid)
    assert B in amap.get(A, set()), "A⊂B → B 是 A 祖先"
    assert C in amap.get(A, set()), "A⊂B⊂C → C 是 A 祖先（传递闭包·transitive_closure 激活）"
    assert C in amap.get(B, set()), "B⊂C → C 是 B 祖先"


def test_common_is_a_ancestor_lca(disc_env):
    """D3 common_is_a_ancestor：diamond 最近共同祖先（B1⊂C·B2⊂C → LCA(B1,B2)=C·最具体）。"""
    b, sid, es, ci, _ = disc_env
    B1 = ci.ensure("B1", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    B2 = ci.ensure("B2", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    C = ci.ensure("C", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_is_a_edge(es, B1, C, source=SOURCE_BARE_TEXT, epistemic=EPI_CUE, space_id=sid)  # B1⊂C
    build_is_a_edge(es, B2, C, source=SOURCE_BARE_TEXT, epistemic=EPI_CUE, space_id=sid)  # B2⊂C
    amap = build_isa_ancestor_map(b, space_id=sid)
    lca = common_is_a_ancestor(B1, B2, amap)
    assert lca == C, "B1 B2 共同祖先=C（最近·最具体·LCA 标准非最抽象）"


def test_common_is_a_ancestor_diamond_tiebreak(disc_env):
    """D3 diamond 多 LCA 候选（不可比）→ NodeRef 升序 tiebreak（bit-identical·非'唯一正确'）。

    B1⊂C·B1⊂D·B2⊂C·B2⊂D → common(B1,B2)={C,D}·C D 不可比（都最深）→ min(NodeRef)。
    """
    b, sid, es, ci, _ = disc_env
    B1 = ci.ensure("B1", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    B2 = ci.ensure("B2", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    C = ci.ensure("C", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    D = ci.ensure("D", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    for child in (B1, B2):
        build_is_a_edge(es, child, C, source=SOURCE_BARE_TEXT, epistemic=EPI_CUE, space_id=sid)
        build_is_a_edge(es, child, D, source=SOURCE_BARE_TEXT, epistemic=EPI_CUE, space_id=sid)
    amap = build_isa_ancestor_map(b, space_id=sid)
    lca = common_is_a_ancestor(B1, B2, amap)
    assert lca == min(C, D), "diamond 不可比 → NodeRef 升序 tiebreak（确定性 bit-identical）"


def test_common_is_a_ancestor_no_common(disc_env):
    """D3 无共同祖先→None（两 ref 无 IS_A 抽象交集·上卷到顶无果）。"""
    b, sid, es, ci, _ = disc_env
    X = ci.ensure("X", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    Y = ci.ensure("Y", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    Z = ci.ensure("Z", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_is_a_edge(es, X, Z, source=SOURCE_BARE_TEXT, epistemic=EPI_CUE, space_id=sid)  # X⊂Z（Y 孤立）
    amap = build_isa_ancestor_map(b, space_id=sid)
    assert common_is_a_ancestor(X, Y, amap) is None, "X Y 无共同祖先 → None"


# ============ 件5：recognize 语言识别（concept_binding·READ 消费） ============

def test_recognize_lang_input_concept_binding(disc_env):
    """件5：discover 语言 SVO 骨架 → recognize 新语言 input → concept_binding（token concept_ref·钥匙① READ 消费）。"""
    from pure_integer_ai.cognition.process.structure_discover import (
        auto_discover_operators, recognize_operators)
    b, sid, es, ci, g = disc_env
    # 发现 SVO 骨架（"猫追狗"+"鸡咬鸭"·同结构异词·cross-sample 全放开=语言泛化牙）
    s1 = _build_lang(disc_env, ["猫", "追", "狗"], seg_label="__seg_a")
    s2 = _build_lang(disc_env, ["鸡", "咬", "鸭"], seg_label="__seg_b")
    discovered = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es,
                                         backend=b, space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(discovered) == 1 and discovered[0].arity == 3, "两 SVO 样本 → 1 骨架 arity=3（全 PARAM）"
    # recognize 新语言 input（"鱼吃虫"·held-out·非发现集→真泛化非循环）
    s3 = _build_lang(disc_env, ["鱼", "吃", "虫"], seg_label="__seg_c")
    recs = recognize_operators([s3], discovered_operators=discovered,
                               backend=b, space_id=sid)
    assert len(recs) == 1, "新语言 input 命中 SVO 骨架"
    rec = recs[0]
    assert rec.is_concept_input, "语言识别须 is_concept_input=True"
    assert len(rec.concept_binding) == 3, "concept_binding 3 slot（全 PARAM）"
    fish = ci.ensure("鱼", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    eat = ci.ensure("吃", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    bug = ci.ensure("虫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    assert rec.concept_binding[0] == fish, "slot0 → 鱼"
    assert rec.concept_binding[1] == eat, "slot1 → 吃"
    assert rec.concept_binding[2] == bug, "slot2 → 虫"


def test_recognize_lang_variable_identity_tooth(disc_env):
    """件5 变量同一性牙：'猫追猫'骨架(arity=2) → '鸡跑鸡'命中(同槽同 ref) / '鸡跑鸭'拒(同槽异 ref)。

    skeleton slot0 两叶同 sid（猫 DAG 共享）·input 位0/位2 须同 ref（变量同一性）。
    """
    from pure_integer_ai.cognition.process.structure_discover import (
        auto_discover_operators, recognize_operators)
    b, sid, es, ci, g = disc_env
    s1 = _build_lang(disc_env, ["猫", "追", "猫"], seg_label="__seg_a")   # 两猫同 ref·arity=2
    s2 = _build_lang(disc_env, ["狗", "咬", "狗"], seg_label="__seg_b")
    discovered = auto_discover_operators([s1, s2], concept_index=ci, edge_store=es,
                                         backend=b, space_id=sid, source=SOURCE_BARE_TEXT)
    assert discovered[0].arity == 2, "两同 ref → arity=2（DAG 共享同槽）"
    # "鸡跑鸡"（两鸡同 ref·同槽）→ 命中
    s3 = _build_lang(disc_env, ["鸡", "跑", "鸡"], seg_label="__seg_c")
    recs = recognize_operators([s3], discovered_operators=discovered,
                               backend=b, space_id=sid)
    assert len(recs) == 1, "两鸡同 ref → 命中（同槽同 ref）"
    chick = ci.ensure("鸡", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    assert recs[0].concept_binding[0] == chick, "slot0 → 鸡（两鸡同槽）"
    # "鸡跑鸭"（鸡鸭异 ref·同槽冲突）→ 拒（变量同一性牙）
    s4 = _build_lang(disc_env, ["鸡", "跑", "鸭"], seg_label="__seg_d")
    recs4 = recognize_operators([s4], discovered_operators=discovered,
                                backend=b, space_id=sid)
    assert len(recs4) == 0, "鸡鸭异 ref → 拒（同 skeleton slot 两异 ref·变量同一性牙）"


# ============ _run_lang caller 生产 e2e（反 theater·caller 真跑·件1 落点修正） ============

def _lang_item(tokens):
    """造语言域 CollectedItem（formal_train/_discover_and_recognize_lang_structures 喂料·镜像 _arith_item）。"""
    from pure_integer_ai.experiments.collection import CollectedItem
    return CollectedItem(modality=MODALITY_LANGUAGE, domain=DOMAIN_TEXT, lang=LANG_ZH,
                         source=SOURCE_BARE_TEXT, tokens=list(tokens))


def test_discover_lang_structures_trigger_via_train_context():
    """生产触发器 _discover_and_recognize_lang_structures（make_train_context）→ 语言语料 → DiscoveredOperator。

    证 caller 真跑（反 theater·件1 落点修正·observe 不建=A6 不冲突）：discover_skeleton 语言样本在生产
    TrainContext 被调·内容哈希独立根 __disc_lang_ + 建 COMPOSES + 抽骨架·产物落生产 backend。
    2 样本同形==2 → 全发现·无 held-out → recognitions=[]。
    """
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_lang_structures
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_lang_item(["猫", "追", "狗"]), _lang_item(["鸡", "咬", "鸭"])]
    ops, recs, gen = _discover_and_recognize_lang_structures(ctx, corpus)
    assert len(ops) == 1, "两 SVO 样本 → 1 骨架（caller 真跑·反 theater）"
    assert ops[0].arity == 3, "SVO 全 PARAM·arity=3"
    assert recs == [], "2 样本同形==2 → 全发现无 held-out → 无可识别"
    assert gen.total_held_out == 0 and gen.verified == 0, "无 held-out → 空泛化（语言 verified=0 钥匙③墙）"
    assert read_composes_attrs(b, ops[0].skeleton_ref).get(ATTR_ORIGIN) == (ORIGIN_DISCOVERED, 0), (
        "骨架真落生产 backend（ATTR_ORIGIN=discovered）")


def test_discover_lang_structures_held_out_recognize_concept_binding():
    """3 样本同形 → 发现已首 2 + 识别余 1（held-out·concept_binding·真泛化非循环·生产 caller 反 theater）。"""
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_lang_structures
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_lang_item(["猫", "追", "狗"]), _lang_item(["鸡", "咬", "鸭"]),
              _lang_item(["鱼", "吃", "虫"])]
    ops, recs, gen = _discover_and_recognize_lang_structures(ctx, corpus)
    assert len(ops) == 1, "3 同形 → 发现已首 2"
    assert len(recs) == 1, "held-out 1（第三个·识别 concept_binding·真泛化非循环）"
    assert recs[0].is_concept_input and len(recs[0].concept_binding) == 3, "语言识别 is_concept_input + 3 slot"
    assert gen.total_held_out == 1 and gen.recognized == 1 and gen.verified == 0, (
        "held-out 1·命中 1·verified=0（语言不可 vm_proof·钥匙③墙·诚实）")


def test_existing_operator_gets_realizes_when_oracle_arrives_later(monkeypatch):
    """跨课程阶段：先发现骨架、后 boot oracle，历史骨架须幂等补标 REALIZES。"""
    from pure_integer_ai.config import gates
    from pure_integer_ai.experiments.formal_train import (
        make_train_context, _discover_and_recognize_lang_structures,
    )

    monkeypatch.setattr(gates, "REALIZES_MODE", True)
    monkeypatch.setattr(gates, "CUE_CLUSTER_MODE", True)
    b = DictBackend()
    train_ctx = make_train_context(b)
    corpus = [
        _lang_item(["猫", "是", "动物"]),
        _lang_item(["狗", "是", "生物"]),
    ]
    ops, _, _ = _discover_and_recognize_lang_structures(train_ctx, corpus)
    assert len(ops) == 1
    assert not train_ctx.edge_store.query_from(
        ops[0].skeleton_ref[0], ops[0].skeleton_ref[1], edge_type=EDGE_REALIZES)

    child = train_ctx.concept_index.lookup("猫", train_ctx.space_id)
    parent = train_ctx.concept_index.lookup("动物", train_ctx.space_id)
    assert child is not None and parent is not None
    train_ctx.edge_store.add(
        space_id_from=child[0], local_id_from=child[1],
        space_id_to=parent[0], local_id_to=parent[1],
        edge_type=EDGE_IS_A, strength=1, source=SOURCE_CONCEPTNET,
        epistemic_origin=EPI_STRUCTURED)

    new_ops, _, _ = _discover_and_recognize_lang_structures(
        train_ctx, [_lang_item(["鱼", "是", "生物"])],
        existing_operators=ops)

    assert new_ops == []
    assert train_ctx.edge_store.query_from(
        ops[0].skeleton_ref[0], ops[0].skeleton_ref[1], edge_type=EDGE_REALIZES)


# ============ S7 相0：vm_proof 降级对偶（教师标定比对·钥匙③完备性核心） ============

def test_phase0_expected_skeleton_verification():
    """S7 相0：教师标 expected_skeleton → recognize 命中骨架 ref==expected → expected_verified + op_confidence sn++。

    跨 run（run1 discover SVO 骨架·run2 教师标 expected_skeleton=该骨架·recognize held-out 比对）。
    反 theater：op_confidence 写 → recognize_operators 择优读（语言识别·半环闭合）。
    """
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_lang_structures
    from pure_integer_ai.storage.op_confidence import read_op_confidence
    b = DictBackend()
    ctx = make_train_context(b)
    # run1：discover 两 SVO 样本 → 骨架
    corpus1 = [_lang_item(["猫", "追", "狗"]), _lang_item(["鸡", "咬", "鸭"])]
    ops1, _, _ = _discover_and_recognize_lang_structures(ctx, corpus1)
    assert len(ops1) == 1
    svo_skeleton = ops1[0].skeleton_ref
    # run2：held-out SVO + 教师标 expected_skeleton=SVO 骨架
    heldout = _lang_item(["鱼", "吃", "虫"])
    heldout.expected_skeleton = svo_skeleton
    _, recs, gen = _discover_and_recognize_lang_structures(ctx, [heldout], existing_operators=ops1)
    assert len(recs) == 1, "held-out 命中 SVO 骨架"
    assert gen.expected_verified == 1, "相0 比对命中（骨架 ref==expected_skeleton）"
    # op_confidence sn++（相0 verified·反 theater 半环：recognize 择优读）
    conf = read_op_confidence(b, ops1[0].name_ref)
    assert conf is not None and conf[0] >= 1, "op_confidence sn>=1（相0 教师标定 verified）"


def test_phase0_expected_skeleton_mismatch_not_verified():
    """S7 相0 不命中：expected_skeleton 指向他骨架·recognize 命中不同→expected_verified=0·op_confidence tn++。"""
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_lang_structures
    from pure_integer_ai.storage.op_confidence import read_op_confidence
    b = DictBackend()
    ctx = make_train_context(b)
    corpus1 = [_lang_item(["猫", "追", "狗"]), _lang_item(["鸡", "咬", "鸭"])]
    ops1, _, _ = _discover_and_recognize_lang_structures(ctx, corpus1)
    # held-out 标 expected_skeleton=错误 ref（(99,99) 不存在）→ 命中骨架 ref != expected → 不 verified
    heldout = _lang_item(["鱼", "吃", "虫"])
    heldout.expected_skeleton = (99, 99)
    _, recs, gen = _discover_and_recognize_lang_structures(ctx, [heldout], existing_operators=ops1)
    assert len(recs) == 1, "held-out 仍命中 SVO（recognize 结构对齐）"
    assert gen.expected_verified == 0, "相0 比对不命中（骨架 ref != expected (99,99)）"
    conf = read_op_confidence(b, ops1[0].name_ref)
    assert conf is not None and conf[0] == 0 and conf[1] >= 1, "op_confidence tn++ sn=0（相0 fail）"


def test_phase0_weaning_post_retires():
    """S7 相0 POST 退场：WEANING_POST 不调相0（防 vacuous 命中 theater·镜像 vm_proof）·expected_verified=0。"""
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_lang_structures
    from pure_integer_ai.cognition.shared.types import WEANING_POST
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST   # 断奶后·教师退场
    corpus1 = [_lang_item(["猫", "追", "狗"]), _lang_item(["鸡", "咬", "鸭"])]
    ops1, _, _ = _discover_and_recognize_lang_structures(ctx, corpus1)
    heldout = _lang_item(["鱼", "吃", "虫"])
    heldout.expected_skeleton = ops1[0].skeleton_ref
    _, recs, gen = _discover_and_recognize_lang_structures(ctx, [heldout], existing_operators=ops1)
    assert gen.expected_verified == 0, "POST 退场·相0 教师路径不调（防 vacuous theater）"


def test_phase0_no_expected_skeleton_degrades():
    """S7 相0 退化：无 expected_skeleton（默认 None）→ 相0 不比对·expected_verified=0·bit-identical。"""
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_lang_structures
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_lang_item(["猫", "追", "狗"]), _lang_item(["鸡", "咬", "鸭"]),
              _lang_item(["鱼", "吃", "虫"])]
    _, _, gen = _discover_and_recognize_lang_structures(ctx, corpus)
    assert gen.expected_verified == 0, "无 expected_skeleton → 相0 不比对（退化 bit-identical）"


# ============ S7 相1：held-out 命中率判据化（渐近·钥匙③完备性核心） ============

def test_phase1_lang_rate_permille():
    """S7 相1：lang_rate_permille = recognized*1000//total_held_out（渐近命中率判据·区别算术 verified 口径）。"""
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_lang_structures
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_lang_item(["猫", "追", "狗"]), _lang_item(["鸡", "咬", "鸭"]),
              _lang_item(["鱼", "吃", "虫"])]
    _, _, gen = _discover_and_recognize_lang_structures(ctx, corpus)
    # 3 同形 SVO·发现首 2·识别余 1（held-out）→ recognized=1·total=1·lang_rate=1000
    assert gen.total_held_out == 1 and gen.recognized == 1
    assert gen.lang_rate_permille == 1000, "相1 命中率 1000（1/1·渐近判据）"
    assert gen.rate_permille == 0, "算术域 rate_permille（verified 口径）仍 0（语言不可 vm_proof）"


def test_phase1_op_confidence_multi_run_stability():
    """S7 相1 多 run 稳定性：op_confidence sn/tn 跨 run 累积（MUTABLE_MONOTONE）·recognize 择优读（阻断稳定）。

    多 run = 概念阻断稳定性（Plan agent 相1 形态 β）：op_confidence 跨 run 累积·渐近稳定·钥匙③判据。
    """
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_lang_structures
    from pure_integer_ai.storage.op_confidence import read_op_confidence
    b = DictBackend()
    ctx = make_train_context(b)
    # run1：discover SVO 骨架
    corpus1 = [_lang_item(["猫", "追", "狗"]), _lang_item(["鸡", "咬", "鸭"])]
    ops1, _, _ = _discover_and_recognize_lang_structures(ctx, corpus1)
    svo_skeleton = ops1[0].skeleton_ref
    # run2：held-out + 教师标 → 相0 比对 → op_confidence sn=1
    heldout1 = _lang_item(["鱼", "吃", "虫"])
    heldout1.expected_skeleton = svo_skeleton
    _discover_and_recognize_lang_structures(ctx, [heldout1], existing_operators=ops1)
    conf2 = read_op_confidence(b, ops1[0].name_ref)
    assert conf2 is not None and conf2[0] == 1, "run2 相0 verified sn=1"
    # run3：再 held-out + 教师标 → op_confidence sn=2 累积（MUTABLE_MONOTONE·多 run 稳定）
    heldout2 = _lang_item(["鸟", "抓", "虫"])
    heldout2.expected_skeleton = svo_skeleton
    _discover_and_recognize_lang_structures(ctx, [heldout2], existing_operators=ops1)
    conf3 = read_op_confidence(b, ops1[0].name_ref)
    assert conf3 is not None and conf3[0] == 2, "run3 相0 累积 sn=2（多 run 稳定性·阻断稳定）"


# ============ 件4：变长 LCS wrapper（lang_structure_align·破同子数门·语言独有） ============

def test_align_variable_consensus_single_anchor_equal_length(disc_env):
    """件4：变长 roots → pairwise_fold consensus（追）→ 等长对齐序列（2·|anchors|+1=3）。

    "猫追"/"狗追小猫"/"猪追大鸭" → consensus=[追]（LCS 共识·全样公共）→ 等长 3（段前 slot + 追 + 末段 slot）。
    """
    from pure_integer_ai.cognition.process.lang_structure_align import align_variable_lang_sequences
    b, sid, es, ci, g = disc_env
    r1 = _build_lang(disc_env, ["猫", "追"], seg_label="__v1")
    r2 = _build_lang(disc_env, ["狗", "追", "小", "猫"], seg_label="__v2")
    r3 = _build_lang(disc_env, ["猪", "追", "大", "鸭"], seg_label="__v3")
    aligned = align_variable_lang_sequences(g, [r1, r2, r3], concept_index=ci, space_id=sid)
    assert aligned is not None, "变长 roots → consensus 非空 → 对齐序列"
    assert len(aligned) == 3, "3 roots → 3 对齐序列"
    assert all(len(seq) == 3 for seq in aligned), "consensus=[追] 1 锚 → 等长 2·1+1=3"
    # 锚位（index 1）= 追 ref（全样同·LCS 共识）
    zhui_ref = ci.ensure("追", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    assert all(seq[1] == zhui_ref for seq in aligned), "锚位=追（consensus LCS 共识·全样同）"


def test_align_variable_no_consensus_returns_none(disc_env):
    """件4 退化：无公共 token → consensus 空 → None（诚实不发现·非 theater 纸面闭合）。"""
    from pure_integer_ai.cognition.process.lang_structure_align import align_variable_lang_sequences
    b, sid, es, ci, g = disc_env
    r1 = _build_lang(disc_env, ["猫", "狗"], seg_label="__v1")
    r2 = _build_lang(disc_env, ["鸡", "鸭"], seg_label="__v2")   # 无公共 token
    aligned = align_variable_lang_sequences(g, [r1, r2], concept_index=ci, space_id=sid)
    assert aligned is None, "无共识 → None（退化·caller roots 不变·变长不发现）"


def test_align_variable_below_min_samples_returns_none(disc_env):
    """件4 退化：<min_samples（1 root）→ None。"""
    from pure_integer_ai.cognition.process.lang_structure_align import align_variable_lang_sequences
    b, sid, es, ci, g = disc_env
    r1 = _build_lang(disc_env, ["猫", "追"], seg_label="__v1")
    aligned = align_variable_lang_sequences(g, [r1], concept_index=ci, space_id=sid)
    assert aligned is None, "<min_samples → None"


def test_align_variable_segment_first_token_slot_not_uniform_gap(disc_env):
    """件4 方案 B（择优）：段 slot=段首 token concept_ref（cross-sample 异词同槽=PARAM 泛化牙）·空段才占位。

    非统一占位（保泛化牙·段前真词异→同槽 PARAM·件2 D2 弱化门极致）。
    "猫追狗"/"大猫追大狗" → consensus=[猫,追,狗] → 等长 7。
    "猫追狗" 段前空→GAP·"大猫追大狗" 段前"大"（段首 token·非占位）。
    """
    from pure_integer_ai.cognition.process.lang_structure_align import align_variable_lang_sequences
    b, sid, es, ci, g = disc_env
    r1 = _build_lang(disc_env, ["猫", "追", "狗"], seg_label="__v1")
    r2 = _build_lang(disc_env, ["大", "猫", "追", "大", "狗"], seg_label="__v2")
    aligned = align_variable_lang_sequences(g, [r1, r2], concept_index=ci, space_id=sid)
    assert aligned is not None
    assert all(len(seq) == 7 for seq in aligned), "consensus=[猫,追,狗] 3 锚 → 等长 2·3+1=7"
    gap_ref = ci.ensure("__lang_align_gap__", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    da_ref = ci.ensure("大", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    # 段前 slot（seq[0]）：'猫追狗' 段前空→GAP·'大猫追大狗' 段前='大'（段首 token·方案 B）
    assert aligned[0][0] == gap_ref, "'猫追狗' 段前空 → GAP 占位"
    assert aligned[1][0] == da_ref, "'大猫追大狗' 段前='大' 段首 token（方案 B·非统一占位·保泛化牙）"


def test_discover_variable_lang_via_wrapper_e2e():
    """件4 e2e（反 theater·生产 caller 真跑）：变长语料 → caller 触发 wrapper → 对齐根 → discover 骨架。

    "猫追"/"狗追小猫"/"猪追大鸭"（length set {2,4}·变长）→ caller 检测变长 → wrapper consensus [追]
    → 等长对齐根（3 句等长 3）→ discover 首 2 + recognize held-out 1。
    无 wrapper 则 length set 多值 → 分散各组 → 变长不发现（对照·wrapper 真行为变）。
    """
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_lang_structures
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_lang_item(["猫", "追"]), _lang_item(["狗", "追", "小", "猫"]),
              _lang_item(["猪", "追", "大", "鸭"])]
    ops, recs, gen = _discover_and_recognize_lang_structures(ctx, corpus)
    assert len(ops) == 1, "变长 3 句 → wrapper 对齐 → 1 骨架（反 theater·真发现·非 theater）"
    assert ops[0].arity == 3, "对齐 [slot,追,slot] → distinct sample0={slot0,追,GAP}=3"
    assert len(recs) == 1, "3 句 → discover 首 2 + recognize held-out 1（concept_binding·真泛化非循环）"
    assert recs[0].is_concept_input and len(recs[0].concept_binding) == 3, (
        "held-out 变长识别 concept_binding 3 slot（语言识别产物）")


def test_variable_lang_same_length_unchanged_bit_identical():
    """件4 bit-identical：同长语料 → length set 单一 → 不触发 wrapper → 走原路径（既有行为零改）。

    守 S3 第二片 21 测 + 既有 caller 同长行为：wrapper 是变长前置门·同长完全不经 wrapper。
    """
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_lang_structures
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_lang_item(["猫", "追", "狗"]), _lang_item(["鸡", "咬", "鸭"])]   # 同长 3
    ops, recs, gen = _discover_and_recognize_lang_structures(ctx, corpus)
    assert len(ops) == 1 and ops[0].arity == 3, "同长 SVO → 原路径 discover（wrapper 不触发·bit-identical）"
    assert recs == [], "2 同长样本 → 全发现无 held-out（原路径·零改）"


# ============ S3 第二刀 Interp2：抽象级 LCA 聚类 + ATTR_SLOT_ROLE（真抽象对撞） ============
#
# Interp2（设计字面·doc/重来_钥匙①语言结构发现机制设计_修正分析七.md §四/§八 line 175）：
# 同 shape_sig 语言样本按 PARAM slot 的 IS_A LCA 上卷类分桶。"猫追老鼠/狗追兔子"（动物类）≠
# "石头追墙/砖追地"（非生物类）→ 产异骨架·破 D2 弱化门全 PARAM collapse。ATTR_SLOT_ROLE=9 挂 skeleton
# CONCEPT slot fresh 节点。冷启动（bare NL 无 IS_A）→ ancestor_map 空 → 走当前路径（bit-identical 零行为变）。


def _inject_isa(disc_env, child_tok: str, parent_ref: ConceptRef):
    """注入合规 IS_A 边（child_tok ⊂ parent_ref·EPI_CUE 系词路径·测试用）。"""
    b, sid, es, ci, _ = disc_env
    child = ci.ensure(child_tok, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_is_a_edge(es, child, parent_ref, source=SOURCE_BARE_TEXT, epistemic=EPI_CUE, space_id=sid)
    return child


def _read_slot_roles(b, g, skeleton_ref: ConceptRef) -> list[tuple[int, int] | None]:
    """读 skeleton 全 CONCEPT slot 的 ATTR_SLOT_ROLE（DFS 阅读序·None=无类约束·未写 attr）。

    返 list[ConceptRef|None]·slot 序 = DFS CONCEPT_LEAF 首遇序（与 build _concept_slot_idx 对齐）。
    fresh CONCEPT slot 节点有 ATTR_OPERAND（PARAM sid·非算子节点）·有 ATTR_SLOT_ROLE 则记值·无则 None。
    """
    children_of = g.read_composes_tree(skeleton_ref)[0]
    roles: list[tuple[int, int] | None] = []
    visited: set[ConceptRef] = set()

    def _dfs(node: ConceptRef) -> None:
        if node in visited:
            return
        visited.add(node)
        attrs = read_composes_attrs(b, node)
        # fresh CONCEPT slot 节点 = 有 ATTR_OPERAND（PARAM sid）·非算子（无 ATTR_OPERATOR）
        if ATTR_OPERAND in attrs and ATTR_OPERATOR not in attrs:
            if ATTR_SLOT_ROLE in attrs:
                r = attrs[ATTR_SLOT_ROLE]
                roles.append((r[0], r[1]))
            else:
                roles.append(None)   # 无 LCA（通配/裸 NL）
        for child in children_of.get(node, []):
            _dfs(child)

    _dfs(skeleton_ref)
    return roles


def test_set_lca_includes_self_single_token(disc_env):
    """set_lca 单 token = 自身（closure 含自身·最深=token 自身·非其 IS_A 祖先）。"""
    b, sid, es, ci, _ = disc_env
    猫 = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_is_a_edge(es, 猫, 动物, source=SOURCE_BARE_TEXT, epistemic=EPI_CUE, space_id=sid)
    amap = build_isa_ancestor_map(b, space_id=sid)
    assert set_lca([猫], amap) == 猫, "单 token LCA = 自身（closure 含自身·最深=token）"


def test_set_lca_diamond_deepest_no_drift(disc_env):
    """set_lca 三 token → 动物（非生物）·解 pairwise-reduce drift。

    猫/狗/狐狸 均 IS_A 动物·动物 IS_A 生物。pairwise reduce LCA(LCA(猫,狗)=动物,狐狸) 把动物当后代查
    → anc[动物]∩anc[狐狸]={生物}→生物（错）。set_lca 用 closure 含自身·交集={动物,生物}→最深=动物（正确）。
    """
    b, sid, es, ci, _ = disc_env
    猫 = _inject_isa(disc_env, "猫", ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT))
    狗 = _inject_isa(disc_env, "狗", ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT))
    狐狸 = _inject_isa(disc_env, "狐狸", ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT))
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    生物 = ci.ensure("生物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_is_a_edge(es, 动物, 生物, source=SOURCE_BARE_TEXT, epistemic=EPI_CUE, space_id=sid)
    amap = build_isa_ancestor_map(b, space_id=sid)
    assert set_lca([猫, 狗, 狐狸], amap) == 动物, "三 token 集合 LCA=动物（解 pairwise-drift·非生物）"


def test_set_lca_no_common_returns_none(disc_env):
    """set_lca 无共同祖先 → None（猫 IS_A 动物 vs 石头 IS_A 非生物·closure 交空）。"""
    b, sid, es, ci, _ = disc_env
    猫 = _inject_isa(disc_env, "猫", ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT))
    石头 = _inject_isa(disc_env, "石头", ci.ensure("非生物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT))
    amap = build_isa_ancestor_map(b, space_id=sid)
    assert set_lca([猫, 石头], amap) is None, "猫/石头 无共同祖先 → None"


def test_set_lca_empty_list_returns_none(disc_env):
    """set_lca 空 token 列表 → None（防御）。"""
    b, sid, es, ci, _ = disc_env
    amap = build_isa_ancestor_map(b, space_id=sid)
    assert set_lca([], amap) is None, "空 tokens → None"


def test_set_lca_repeated_token_returns_self(disc_env):
    """set_lca 重复同 token → 自身（closure∩closure=closure·最深=token 自身）。

    slot1=[追,追]（簇内两样本同 verb·无 IS_A）→ LCA=追（自身）·对应对抗审 L2 盲区补。
    """
    b, sid, es, ci, _ = disc_env
    追 = ci.ensure("追", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    amap = build_isa_ancestor_map(b, space_id=sid)   # 追 无 IS_A·anc[追]=空
    assert set_lca([追, 追], amap) == 追, "重复同 token → 自身（closure={追}∩{追}={追}·最深=追）"


def test_set_lca_distinct_tokens_no_isa_returns_none(disc_env):
    """set_lca 异 token 无 IS_A → None（closure={a}∩{b}={}）。

    对抗审 L2 盲区补：无 IS_A 时异 token 不聚类（守"抽象类须 IS_A 涌现"非词级强行归并）。
    """
    b, sid, es, ci, _ = disc_env
    猫 = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    狗 = ci.ensure("狗", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    amap = build_isa_ancestor_map(b, space_id=sid)   # 无 IS_A 边
    assert set_lca([猫, 狗], amap) is None, "异 token 无 IS_A → None（closure 交空·不强行归并）"


def test_normalize_abstract_sig_all_none_to_empty():
    """_normalize_abstract_sig：全 None → () 守 bit-identical（arith/裸 NL 同名）。"""
    assert _normalize_abstract_sig(()) == ()
    assert _normalize_abstract_sig((None, None, None)) == (), "全 None → ()（归一·守跨 run resume）"
    # 真 LCA ref 保留
    ref = (5, 7)
    assert _normalize_abstract_sig((ref, None, ref)) == (ref, None, ref), "真 LCA ref → 原样保留"


def test_shape_name_abstract_sig_bit_identical(disc_env):
    """_shape_name：abstract_sig=()/全None/缺省 → 同名（bit-identical·守幂等+跨 run）；真 LCA ref → 异名。"""
    b, sid, es, ci, _ = disc_env
    sig = (10, -1, -1, -1)   # NOP + 3 leaves
    arity = 3
    n_default = _shape_name(sig, arity)                      # 缺省 abstract_sig=()
    n_empty = _shape_name(sig, arity, ())                    # 显式 ()
    n_all_none = _shape_name(sig, arity, (None, None, None)) # 全 None
    assert n_default == n_empty == n_all_none, "缺省/()/全None → 同名（bit-identical·守跨 run resume + 幂等门）"
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    n_with_role = _shape_name(sig, arity, (动物, None, 动物))  # 真 LCA ref
    assert n_with_role != n_default, "真 LCA ref → 异名（Interp2 真行为变·破 D2 弱化门 collapse）"


def test_interp2_abstract_collision_two_skeletons(disc_env):
    """Interp2 e2e + 刀1 多级共存：4 SVO 样本·两类 IS_A（动物 vs 非生物）→ LCA 聚类产 2 类级骨架 + 1 词例级骨架 = 3 骨架。

    猫追老鼠/狗追兔子（动物类）+ 石头追墙/砖追地（非生物类）·slot1 全=追（同 verb·单 token LCA=追自身）。
    cluster A: slot0=动物·slot1=追·slot2=动物；cluster B: slot0=非生物·slot1=追·slot2=非生物。
    set_lca([猫,石头])=None → 不 join → 2 类级簇·各 ≥K=2 → 2 类级骨架·ATTR_SLOT_ROLE 写盘。
    **刀1 件3 多级共存**：存在 ≥K 类级簇 → 追加词例级簇 (4 roots, None) → 产 1 词例级骨架（abstract_sig=()·无
    ATTR_SLOT_ROLE·名 = has_isa=False 单簇路径之名·与类级异名不撞）→ 共 3 骨架（2 类级 + 1 词例级）。
    """
    b, sid, es, ci, g = disc_env
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    非生物 = ci.ensure("非生物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    for tok in ["猫", "狗", "老鼠", "兔子"]:
        _inject_isa(disc_env, tok, 动物)
    for tok in ["石头", "砖", "墙", "地"]:
        _inject_isa(disc_env, tok, 非生物)
    s1 = _build_lang(disc_env, ["猫", "追", "老鼠"], seg_label="__seg_a1")
    s2 = _build_lang(disc_env, ["狗", "追", "兔子"], seg_label="__seg_a2")
    s3 = _build_lang(disc_env, ["石头", "追", "墙"], seg_label="__seg_b1")
    s4 = _build_lang(disc_env, ["砖", "追", "地"], seg_label="__seg_b2")
    discovered = auto_discover_operators(
        [s1, s2, s3, s4], concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(discovered) == 3, "刀1 多级共存：2 类级（动物/非生物·LCA 聚类）+ 1 词例级（全 roots 无 LCA 约束）= 3 骨架"
    assert all(op.arity == 3 for op in discovered), "三骨架均 arity=3（同 shape·同 arity·异 abstract_sig）"
    assert len({op.name for op in discovered}) == 3, "三骨架异名（abstract_sig 进 hash·2 类级 + 1 词例级）"
    # ATTR_SLOT_ROLE 写盘：类级骨架 slot0/slot2 有 LCA（动物 或 非生物）·slot1=追（单 token LCA=追自身）·
    # 词例级骨架全 None（无 LCA 约束·PARAM 接受任何 token·_align_walk:1124）。
    追 = ci.ensure("追", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    role_sets = []
    for op in discovered:
        roles = _read_slot_roles(b, g, op.skeleton_ref)
        assert len(roles) == 3, f"skeleton 3 CONCEPT slot（DFS 序）·got {roles}"
        role_sets.append(tuple(roles))
    # 两类级骨架 slot 角色集 = (动物,追,动物) 与 (非生物,追,非生物)·词例级骨架 = (None,None,None)（刀1 多级共存）
    expected = {(动物, 追, 动物), (非生物, 追, 非生物), (None, None, None)}
    assert set(role_sets) == expected, f"slot 角色集 = 动物类/非生物类（类级）+ 全 None（词例级）·got {role_sets}"


def test_load_discovered_operators_reconstructs_abstract_sig(disc_env):
    """B6 Bug 1 修：LOAD 经 _collect_slot_lcas 重建 abstract_sig → 名 == BUILD 名。

    场景（镜像 test_interp2_abstract_collision_two_skeletons）：4 SVO 样本·两类 IS_A（动物 vs 非生物）
    → BUILD 产 2 类级骨架（同 (sig,arity) 异 abstract_sig）+ 1 词例级骨架（abstract_sig=()）= 3 骨架。

    修前：LOAD `name=_shape_name(sig, arity)` 缺 abstract_sig → 两类级骨架（同 sig 同 arity）撞同 name
    → `op_by_name` 字面覆盖（后赢）→ 验证用错 skeleton。
    修后：LOAD `name=_shape_name(sig, arity, _collect_slot_lcas(...))` → 名含 abstract_sig → 异名 → 不撞。

    同 backend LOAD（镜像 test_stage9:1188 既有范式·auto_discover 写 ATTR_OPERATOR_DEF 后 load 直读·
    免 dump/load 重型设施）·纯读 LOAD 代码路径（_collect_slot_lcas 从 skeleton ATTR_SLOT_ROLE 重建）。
    """
    b, sid, es, ci, g = disc_env
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    非生物 = ci.ensure("非生物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    for tok in ["猫", "狗", "老鼠", "兔子"]:
        _inject_isa(disc_env, tok, 动物)
    for tok in ["石头", "砖", "墙", "地"]:
        _inject_isa(disc_env, tok, 非生物)
    s1 = _build_lang(disc_env, ["猫", "追", "老鼠"], seg_label="__ld_sig_a1")
    s2 = _build_lang(disc_env, ["狗", "追", "兔子"], seg_label="__ld_sig_a2")
    s3 = _build_lang(disc_env, ["石头", "追", "墙"], seg_label="__ld_sig_b1")
    s4 = _build_lang(disc_env, ["砖", "追", "地"], seg_label="__ld_sig_b2")
    discovered = auto_discover_operators(
        [s1, s2, s3, s4], concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(discovered) == 3, "前置：2 类级（动物/非生物·异 abstract_sig）+ 1 词例级 = 3 骨架"
    build_names = {op.name: op for op in discovered}
    assert len(build_names) == 3, "前置：三骨架异名（abstract_sig 进 hash）"

    # LOAD 重建：名 == BUILD 名（核心断言·证 Bug 1 修）。
    loaded = load_discovered_operators(b, space_id=sid)
    assert len(loaded) == 3, f"load 须重建 3 算子·得 {len(loaded)}"
    loaded_names = {op.name: op for op in loaded}
    assert len(loaded_names) == 3, (
        f"load 名须全异（_collect_slot_lcas 重建 abstract_sig 进名）·"
        f"修前两类级骨架撞同 name 致 len≠3·得 {len(loaded_names)}"
    )
    assert set(loaded_names) == set(build_names), (
        f"load 名集合 == BUILD 名集合（abstract_sig 经 ATTR_SLOT_ROLE 重建对齐）·"
        f"loaded={set(loaded_names)} build={set(build_names)}"
    )
    # name → skeleton_ref 映射一致（防 op_by_name 字面覆盖致 name 指错骨架）。
    for op in discovered:
        assert loaded_names[op.name].skeleton_ref == op.skeleton_ref, (
            f"load 名 {op.name!r} → skeleton_ref 须 == BUILD（_verify_generalization op_by_name 用对骨架）"
        )


def test_interp2_collapse_when_no_majority(disc_env):
    """Interp2 兜底（修改点 B）：2 样本异抽象类·无簇 ≥K → 退化当前行为（合并 1 簇 slot_lcas=None·不写 ATTR_SLOT_ROLE）。

    猫追老鼠（动物类）+ 石头追墙（非生物类）·K=2。聚类：cluster1=[猫追老鼠]·cluster2=[石头追墙]·皆 <K=2
    → 兜底合并 → 1 骨架·abstract_sig=()（名同今）·无 ATTR_SLOT_ROLE。守"至少产 1 骨架"不劣化当前。
    """
    b, sid, es, ci, g = disc_env
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    非生物 = ci.ensure("非生物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    _inject_isa(disc_env, "猫", 动物)
    _inject_isa(disc_env, "石头", 非生物)
    s1 = _build_lang(disc_env, ["猫", "追", "老鼠"], seg_label="__seg_c1")
    s2 = _build_lang(disc_env, ["石头", "追", "墙"], seg_label="__seg_c2")
    discovered = auto_discover_operators(
        [s1, s2], concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(discovered) == 1, "兜底：2 异类样本无簇≥K → 合并 1 骨架（退化当前·不劣化）"
    roles = _read_slot_roles(b, g, discovered[0].skeleton_ref)
    assert all(r is None for r in roles), "兜底 slot_lcas=None → 不写 ATTR_SLOT_ROLE（absence=无类约束）"


def test_interp2_bit_identical_no_isa_unchanged(disc_env):
    """Interp2 bit-identical：裸 NL（无 IS_A）→ ancestor_map 空 → 跳过聚类 → 1 骨架·无 ATTR_SLOT_ROLE·名同今。

    守既有行为零改：bare NL 语料第二刀 no-op（诚实边界·D3 caller 激活但数据空）。
    """
    b, sid, es, ci, g = disc_env
    # 不注入任何 IS_A（ancestor_map 空）
    s1 = _build_lang(disc_env, ["猫", "追", "狗"], seg_label="__seg_d1")
    s2 = _build_lang(disc_env, ["鸡", "咬", "鸭"], seg_label="__seg_d2")
    discovered = auto_discover_operators(
        [s1, s2], concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(discovered) == 1, "裸 NL → 1 骨架（跳过聚类·单组一骨架·bit-identical）"
    # 名 = 原 _shape_name(sig, arity)（abstract_sig=() 归一）
    from pure_integer_ai.cognition.process.structure_discover import shape_signature
    sig = tuple(shape_signature(g, s1))
    expected_name = _shape_name(sig, 3)
    assert discovered[0].name == expected_name, "裸 NL 名 = 原 Interp1（abstract_sig 全 None 归一 ()）"
    roles = _read_slot_roles(b, g, discovered[0].skeleton_ref)
    assert all(r is None for r in roles), "裸 NL 不写 ATTR_SLOT_ROLE（无 IS_A·LCA 全 None）"


def test_probe_arity_matches_discover_interp2_clusters(disc_env):
    """Interp2 drift 防线：聚类不改 arity·probe_arity == discover_skeleton.arity（含 IS_A 聚类语料）。

    镜像 test_probe_arity_matches_discover_lang·扩覆盖 Interp2 IS_A 注入场景（聚类后簇内 arity 不变）。
    """
    b, sid, es, ci, _ = disc_env
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    for tok in ["猫", "狗", "老鼠", "兔子"]:
        _inject_isa(disc_env, tok, 动物)
    s1 = _build_lang(disc_env, ["猫", "追", "老鼠"], seg_label="__seg_e1")
    s2 = _build_lang(disc_env, ["狗", "追", "兔子"], seg_label="__seg_e2")
    probed = probe_arity(b, [s1, s2])
    disc = discover_skeleton([s1, s2], concept_index=ci, edge_store=es, backend=b,
                             space_id=sid, source=SOURCE_BARE_TEXT, skeleton_label="drift2")
    disc_arity = disc.arity if disc is not None else None
    assert probed == disc_arity == 3, "Interp2 聚类不改 arity（drift 防线·probe==discover）"


# ============ 片2：recognize 抽象匹配（held-out 词级零交集 + IS_A 上卷命中 = 反 theater 牙） ============


def test_interp2_recognize_held_out_abstract_hit(disc_env):
    """片2 反 theater 牙 + 刀2 件6 多解析：held-out '狐狸追鸡'（狐狸/鸡 词级未学·狐狸 IS_A 动物）双 align。

    Interp2 抽象对撞真行为变·非纸面：词级零交集（狐狸≠猫/狗·鸡≠老鼠/兔子·词级 ref 等价必拒）·
    仅靠 IS_A LCA 上卷（狐狸⊂动物 == skeleton slot0 ATTR_SLOT_ROLE=动物）→ 抽象命中。
    对照：非生物类骨架 slot0=非生物·狐狸不可达 → 不命中（不进 aligning）。
    **刀2 件6 多解析**：狐狸追鸡双 align——动物类骨架（ATTR_SLOT_ROLE 抽象命中）+ 词例级骨架（刀1 追加·
    无 ATTR_SLOT_ROLE·PARAM loose 兜底·_align_walk:1124）→ recognize 返全列 2 Recognition（同 input_root·
    rate 都 0 cold-start·stable sort 保 BFS 序·类级 append 末尾前→类级在前·词例级在后）。concept_binding[0]=狐狸
    （input token ref 真记录·两 rec 都记）。反 theater：真两级描述（类级抽象 + 词例级 loose）·非伪造。
    """
    from pure_integer_ai.cognition.process.structure_discover import recognize_operators
    b, sid, es, ci, g = disc_env
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    非生物 = ci.ensure("非生物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    for tok in ["猫", "狗", "老鼠", "兔子"]:
        _inject_isa(disc_env, tok, 动物)
    for tok in ["石头", "砖", "墙", "地"]:
        _inject_isa(disc_env, tok, 非生物)
    # 发现集：4 SVO·两类（动物/非生物）→ 2 骨架
    discover_roots = [
        _build_lang(disc_env, ["猫", "追", "老鼠"], seg_label="__seg_h1"),
        _build_lang(disc_env, ["狗", "追", "兔子"], seg_label="__seg_h2"),
        _build_lang(disc_env, ["石头", "追", "墙"], seg_label="__seg_h3"),
        _build_lang(disc_env, ["砖", "追", "地"], seg_label="__seg_h4"),
    ]
    discovered = auto_discover_operators(
        discover_roots, concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(discovered) == 3, "前置（刀1 多级共存）：4 样本两类 → 2 类级 + 1 词例级 = 3 骨架"
    # held-out 注入 IS_A（狐狸/鸡 ⊂ 动物·词级未学但抽象可达）
    狐狸 = _inject_isa(disc_env, "狐狸", 动物)
    鸡 = _inject_isa(disc_env, "鸡", 动物)
    held_out = _build_lang(disc_env, ["狐狸", "追", "鸡"], seg_label="__seg_held")
    recs = recognize_operators([held_out], discovered_operators=discovered,
                               backend=b, space_id=sid)
    # 刀2 件6 多解析：返全列 2（动物类抽象命中 + 词例级 loose 兜底·非生物类抽象拒不入 aligning）
    assert len(recs) == 2, f"刀2 多解析：狐狸追鸡双 align（类级抽象 + 词例级 loose）→ 返 2·得 {len(recs)}"
    # 定位动物类骨架 + 词例级骨架（slot0 ATTR_SLOT_ROLE 区分）
    animal_op = None
    word_inst_op = None
    for op in discovered:
        roles = _read_slot_roles(b, g, op.skeleton_ref)
        if roles and roles[0] == 动物:
            animal_op = op
        elif roles and all(r is None for r in roles):
            word_inst_op = op
    assert animal_op is not None, "前置：动物类骨架存在"
    assert word_inst_op is not None, "前置：词例级骨架存在（刀1 多级共存）"
    # recs[0] = 动物类（BFS 序类级在前·stable sort 保·反 theater 择类级优先）
    rec = recs[0]
    assert rec.operator_name == animal_op.name, f"recs[0]=动物类骨架（抽象匹配精准）·得 {rec.operator_name}"
    # concept_binding[0]=狐狸·concept_binding[2]=鸡（input token ref 真记录·slot1=追）
    狐狸_ref = ci.ensure("狐狸", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    鸡_ref = ci.ensure("鸡", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    追_ref = ci.ensure("追", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    assert rec.concept_binding[0] == 狐狸_ref, "类级 concept_binding[0]=狐狸（input token ref）"
    assert rec.concept_binding[2] == 鸡_ref, "类级 concept_binding[2]=鸡（input token ref）"
    assert rec.concept_binding[1] == 追_ref, "类级 concept_binding[1]=追（slot1 LCA=追自身·inp==lca）"
    # recs[1] = 词例级（loose 兜底·ATTR_SLOT_ROLE 全 None·反 theater 真两级描述对照）
    word_inst = recs[1]
    assert word_inst.operator_name == word_inst_op.name, (
        f"recs[1]=词例级骨架（loose 兜底·PARAM 接受任何 token）·得 {word_inst.operator_name}")
    assert word_inst.concept_binding[0] == 狐狸_ref, "词例级 concept_binding[0]=狐狸（loose 兜底）"
    assert word_inst.concept_binding[2] == 鸡_ref, "词例级 concept_binding[2]=鸡（loose 兜底）"


def test_interp2_recognize_rejects_non_isa_held_out(disc_env):
    """片2 对照 + 刀1 多级共存：held-out 无 IS_A 关系 → 不命中类级骨架（抽象拒）·但命中词例级骨架（loose 兜底）。

    动物类/非生物类骨架 slot0 有 ATTR_SLOT_ROLE=动物/非生物·held-out '桌子追椅子'（桌子/椅子 无 IS_A·词级亦未学）
    → 类级抽象拒（lca_ref not in anc[桌子]·且 动物≠桌子）+ 词级 ref 等价拒（桌子≠猫）。
    **刀1 后**：词例级骨架（无 ATTR_SLOT_ROLE·PARAM 接受任何 token·_align_walk:1124）loose 兜底命中桌子追椅子
    → recognize 命中词例级骨架（非类级）。冷启动 aligning：类级全拒·仅词例级 align → aligning[0]=词例级。
    新对照语义（刀1 多级）：命中词例级（loose）不命中类级（抽象拒）= 多级共存诚实边界·非"全拒"旧单级语义。
    """
    from pure_integer_ai.cognition.process.structure_discover import recognize_operators
    b, sid, es, ci, g = disc_env
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    非生物 = ci.ensure("非生物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    for tok in ["猫", "狗", "老鼠", "兔子"]:
        _inject_isa(disc_env, tok, 动物)
    for tok in ["石头", "砖", "墙", "地"]:
        _inject_isa(disc_env, tok, 非生物)
    discover_roots = [
        _build_lang(disc_env, ["猫", "追", "老鼠"], seg_label="__seg_r1"),
        _build_lang(disc_env, ["狗", "追", "兔子"], seg_label="__seg_r2"),
        _build_lang(disc_env, ["石头", "追", "墙"], seg_label="__seg_r3"),
        _build_lang(disc_env, ["砖", "追", "地"], seg_label="__seg_r4"),
    ]
    discovered = auto_discover_operators(
        discover_roots, concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(discovered) == 3, "前置（刀1）：4 样本两类 → 2 类级 + 1 词例级 = 3 骨架"
    # held-out 桌子追椅子：不注入 IS_A（桌子/椅子 无抽象关系）
    held_out = _build_lang(disc_env, ["桌子", "追", "椅子"], seg_label="__seg_rej")
    recs = recognize_operators([held_out], discovered_operators=discovered,
                               backend=b, space_id=sid)
    assert len(recs) == 1, "刀1：桌子追椅子命中词例级骨架（loose 兜底·无类约束）·非 0（旧单级语义）·非类级"
    rec = recs[0]
    # 命中的须是词例级骨架（ATTR_SLOT_ROLE 全 None）·非动物/非生物类级骨架（抽象拒·不可达）
    matched_op = next(op for op in discovered if op.name == rec.operator_name)
    roles = _read_slot_roles(b, g, matched_op.skeleton_ref)
    assert all(r is None for r in roles), \
        f"命中词例级骨架（ATTR_SLOT_ROLE 全 None）·非类级（抽象拒·got {roles}）"


# ============ 刀1 件4 可变父：动词槽 IS_A LCA 上卷（破动词常量锚·doc §5 刀1 point 2） ============


def test_knife1_variable_parent_verb_slot_lca_rollup(disc_env):
    """刀1 件4 真测：≥2 异动词（追/咬 IS_A 动作）→ 类级骨架 slot1 ATTR_SLOT_ROLE=动作（动词槽 IS_A LCA 上卷·非追/咬自身）。

    **破"动词常量锚"窄框**：动词槽与名词槽对称走 IS_A LCA（ATTR_SLOT_ROLE=9·非词性 enum 硬编码·§8.1c 合规·
    沿合规 IS_A 边 LCA climb 非建造）。簇内 slot1 token 集 [追,咬]·set_lca([追,咬]) under ancestor_map →
    anc[追]∩anc[咬]（含自身 closure）= {动作, ...} 共同 → 最深=动作 → slot_lcas[1]=动作 → ATTR_SLOT_ROLE 写盘。

    对照既有 `test_interp2_abstract_collision_two_skeletons`：slot1 全=追（同动词）·set_lca([追,追])=追自身·
    未真测"动词上卷到动作类"（件4 核心）。本测异动词 set_lca([追,咬])=动作 真验动词槽可变父。

    **刀1 多级共存**：异动词两类样本 → 1 类级簇（slot_lcas=[动物,动作,动物]·≥K=2）+ 追加词例级簇 = 2 骨架。
    类级 slot1=动作（非追/咬）·slot0/slot2=动物·词例级全 None。
    """
    b, sid, es, ci, g = disc_env
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    动作 = ci.ensure("动作", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    for tok in ["猫", "狗", "老鼠", "兔子"]:
        _inject_isa(disc_env, tok, 动物)
    for tok in ["追", "咬"]:
        _inject_isa(disc_env, tok, 动作)
    s1 = _build_lang(disc_env, ["猫", "追", "老鼠"], seg_label="__seg_v1")
    s2 = _build_lang(disc_env, ["狗", "咬", "兔子"], seg_label="__seg_v2")
    discovered = auto_discover_operators(
        [s1, s2], concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_BARE_TEXT)
    assert len(discovered) == 2, "刀1 多级共存：异动词两样本 → 1 类级 + 1 词例级 = 2 骨架"
    assert all(op.arity == 3 for op in discovered), "两骨架均 arity=3（同 shape）"
    # 找类级骨架（ATTR_SLOT_ROLE 非 None）·验动词槽 LCA 上卷到动作（非追/咬自身·件4 真验）
    class_op = None
    word_op = None
    for op in discovered:
        roles = _read_slot_roles(b, g, op.skeleton_ref)
        if any(r is not None for r in roles):
            class_op = op
        else:
            word_op = op
    assert class_op is not None, "类级骨架存在（异动词 IS_A 动作 → 聚类成类级簇）"
    assert word_op is not None, "词例级骨架存在（刀1 追加·多级共存）"
    追 = ci.ensure("追", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    咬 = ci.ensure("咬", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    roles = _read_slot_roles(b, g, class_op.skeleton_ref)
    assert roles == [动物, 动作, 动物], (
        f"动词槽 LCA 上卷到 动作（非追/咬自身·破动词常量锚·件4 核心）·"
        f"slot0=动物·slot1=动作·slot2=动物·got {roles}")
    # 对照：slot1 须非追/咬自身（真上卷·非单 token LCA）
    assert roles[1] == 动作 and roles[1] != 追 and roles[1] != 咬, (
        f"slot1=动作（异动词共同祖先）·非追/咬自身·got {roles[1]}")


# ============ 刀6 件7 sense 多义管线修通（片4 反 theater e2e·clone aligning_root 选 sense） ============

def test_knife6_sense_clone_disambiguation_e2e():
    """刀6 片4 反 theater 牙：held-out "狐狸追老鼠" clone 动物老鼠/鼠标·动物类骨架命中动物老鼠·不命中鼠标。

    MultiRef 管线修通验证：sense_candidates boot 种（老鼠→动物老鼠/鼠标）+ IS_A boot 种（动物类/鼠标物品）·
    caller 建 COMPOSES 用首 sense ref（动物老鼠·IS_A 上卷）→ discover 建动物类骨架（slot ATTR_SLOT_ROLE=动物）→
    recognize held-out clone（首 sense 动物老鼠 root + 鼠标 clone root）→ 动物老鼠命中（IS_A 共祖）·
    鼠标不命中（IS_A 物品非动物·ATTR_SLOT_ROLE 抽象拒）→ recognized==1 distinct origin（clone 不双计）。

    **诚实边界**：结构选优（IS_A 共祖）≠ 语义消歧（#479 墙·定义权归教师·共现也无法区分时撞墙·stable≠correct）。
    **bit-identical**：gate OFF（无 boot 种）→ 无 clone → 等同现状。
    """
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_lang_structures
    from pure_integer_ai.storage.sense_candidates import bootstrap_sense_candidates
    from pure_integer_ai.cognition.understanding.is_a import bootstrap_is_a_edges
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    from pure_integer_ai.config import gates
    b = DictBackend()
    ctx = make_train_context(b)
    sid = ctx.space_id
    ci = ctx.concept_index
    g = ConceptGraph(b)
    # boot 种 sense_candidates（老鼠→动物老鼠/鼠标·caller COMPOSES 首 sense=NodeRef 升序首）
    bootstrap_sense_candidates(b, ci, [("老鼠", ["动物老鼠", "鼠标"])], space_id=sid)
    # boot 种 IS_A（动物类 + 鼠标物品·Interp2 ATTR_SLOT_ROLE 上卷到动物/物品）
    bootstrap_is_a_edges(ci, ctx.edge_store, [
        ("动物老鼠", "动物"), ("猫", "动物"), ("狗", "动物"), ("兔子", "动物"), ("狐狸", "动物"),
        ("鼠标", "物品"),
    ], space_id=sid)
    # corpus: 3 同形 SVO·前 2 discover（建骨架·caller COMPOSES 老鼠位=首 sense 动物老鼠）·
    # 第 3 recognize（held-out "狐狸追老鼠"·老鼠位 clone 动物老鼠/鼠标）
    corpus = [_lang_item(["猫", "追", "老鼠"]),
              _lang_item(["狗", "追", "兔子"]),
              _lang_item(["狐狸", "追", "老鼠"])]
    saved = gates.SENSE_LOOKUP_MODE
    gates.SENSE_LOOKUP_MODE = True
    try:
        ops, recs, gen = _discover_and_recognize_lang_structures(ctx, corpus)
    finally:
        gates.SENSE_LOOKUP_MODE = saved
    # 前置：discover 建骨架（动物类·caller COMPOSES 首 sense ref·IS_A 上卷）
    assert len(ops) >= 1, f"discover 建骨架（动物类 + 词例级）·got {len(ops)}"
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    动物老鼠 = ci.ensure("动物老鼠", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    鼠标 = ci.ensure("鼠标", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    # 定位动物类骨架（slot2 ATTR_SLOT_ROLE=动物）·刀1 多级共存另有词例级（slot2=None）
    animal_op = None
    for op in ops:
        roles = _read_slot_roles(b, g, op.skeleton_ref)
        if roles and len(roles) >= 3 and roles[2] == 动物:
            animal_op = op
            break
    assert animal_op is not None, f"前置：动物类骨架存在（slot2=动物）·ops_roles={[_read_slot_roles(b, g, op.skeleton_ref) for op in ops]}"
    # 反 theater 牙：动物类骨架命中的 recs·concept_binding[2] 只含动物老鼠·不含鼠标
    # （动物老鼠 IS_A 动物 → 命中 slot2=动物；鼠标 IS_A 物品非动物 → ATTR_SLOT_ROLE 抽象拒·不命中动物类·
    #   词例级 loose 兜底让鼠标命中是 OK·非动物类结构选优·#479 墙·stable≠correct 非语义消歧）
    animal_recs = [rec for rec in recs if rec.operator_name == animal_op.name]
    animal_slot2 = {rec.concept_binding[2] for rec in animal_recs
                    if rec.is_concept_input and len(rec.concept_binding) >= 3}
    assert 动物老鼠 in animal_slot2, f"动物类骨架命中动物老鼠 sense（IS_A 共祖）·slot2={animal_slot2}"
    assert 鼠标 not in animal_slot2, (
        f"动物类骨架不命中鼠标 sense（IS_A 物品·ATTR_SLOT_ROLE 抽象拒·结构选优非语义消歧·#479 墙）·slot2={animal_slot2}")
    # distinct origin 防双计（clone root → 原 root·recognized 计 distinct·守 lang_rate≤1000）
    assert gen.recognized == 1, f"distinct origin=1（held-out 原 root·clone 不双计）·got {gen.recognized}"
    assert gen.total_held_out == 1, f"held-out=1（clone 是 sense 候选扩展·不增计数）·got {gen.total_held_out}"


def test_knife6_sense_clone_bit_identical_gate_off():
    """刀6 片4 bit-identical：gate OFF（无 sense_candidates boot）→ caller COMPOSES ensure(tok) 原路径·无 clone·
    recognized 计 distinct input_root（刀2 范式）·等同刀5 后现状零行为变。

    退化链 5 步（plan 决断 5）：gate OFF → sense_cands=[] → caller ensure(tok) → 无 clone → 退化。
    """
    from pure_integer_ai.experiments.formal_train import make_train_context, _discover_and_recognize_lang_structures
    from pure_integer_ai.config import gates
    b = DictBackend()
    ctx = make_train_context(b)
    # 不 boot 种 sense_candidates（gate OFF·无 sense_facts）·IS_A 也空（bare）
    corpus = [_lang_item(["猫", "追", "狗"]),
              _lang_item(["鸡", "咬", "鸭"]),
              _lang_item(["鱼", "吃", "虫"])]
    saved = gates.SENSE_LOOKUP_MODE
    gates.SENSE_LOOKUP_MODE = False
    try:
        ops, recs, gen = _discover_and_recognize_lang_structures(ctx, corpus)
    finally:
        gates.SENSE_LOOKUP_MODE = saved
    # 等同 test_discover_lang_structures_held_out_recognize_concept_binding（gate OFF 退化·零行为变）
    assert len(ops) == 1, "3 同形 → 发现已首 2（gate OFF 退化·无 sense 影响）"
    assert len(recs) == 1, "held-out 1（第三·无 clone·原路径）"
    assert gen.recognized == 1 and gen.total_held_out == 1, "distinct=1·held_out=1（无 clone）"


# ============ B6 Bug 2+3 路由（聚类前置·route_samples_for_discovery·2026-07-06）============
#
# 反 theater：Bug 2（existing_keys 缺 abstract_sig·resume 同 (sig,arity) 异 abstract_sig 新样本静默丢）+
# Bug 3（cluster-blind held-out·混合簇前 K 横跨簇致每簇 <K 不发现）经 route_samples_for_discovery 修。
# 聚类前置：路由级 _cluster_by_lca 决定 discover/recognize 归属·auto_discover 级聚类决定产骨架·两层幂等。


def _existing_keys_from_ops_lang(disc_env, ops):
    """镜像 formal_train existing_keys 构造（B6 Bug 2·abstract_sig 维 + §十八 6a-3 cue_sig 第4维）·返 (existing_keys, existing_sigs)。"""
    b, sid, es, ci, g = disc_env
    existing_keys: set = set()
    existing_sigs: set = set()
    for op in ops:
        op_sig = tuple(shape_signature(g, op.skeleton_ref))
        op_asig = _normalize_abstract_sig(_collect_slot_lcas(b, g, op.skeleton_ref))
        op_cue = _normalize_abstract_sig(_collect_cue_sig(b, g, op.skeleton_ref))   # §十八 6a-3：cue_sig 第4维（gate OFF 全 ()→bit-identical）
        existing_keys.add((op_sig, op.arity, op_asig, op_cue))
        existing_sigs.add(op_sig)
    return existing_keys, existing_sigs


def test_route_lang_no_isa_bit_identical_single_cluster(disc_env):
    """裸 NL（无 IS_A）路由 bit-identical：has_isa=False → 单簇 slot_lcas=None → abstract_sig=() →
    existing 空 → 2 样本 ≥K 全 discover·recognize 空。行为同原 (sig,arity) 路由。"""
    b, sid, es, ci, g = disc_env
    s1 = _build_lang(disc_env, ["猫", "追", "狗"], seg_label="__route_ni1")
    s2 = _build_lang(disc_env, ["鸡", "咬", "鸭"], seg_label="__route_ni2")
    discover_roots, recognize_roots = route_samples_for_discovery(
        b, g, [s1, s2], existing_keys=set(), existing_sigs=set(), space_id=sid)
    assert set(discover_roots) == {s1, s2}, f"无 IS_A 单簇·两样本 ≥K 全 discover·得 {discover_roots}"
    assert recognize_roots == [], f"K=2 恰满无 held-out·得 {recognize_roots}"


def test_route_lang_multi_cluster_per_cluster_heldout(disc_env):
    """**Bug 3 修（per-cluster held-out）**：4 SVO·两类 IS_A（动物/非生物）·existing 空 →
    每簇各 ≥K=2 → **每簇独立首 K discover**（discover=4·非 cluster-blind 整组首 K=2）·recognize 空。

    原路由 cluster-blind：grp_sorted[:2] 全送 discover（可能仅动物类）·grp_sorted[2:] 全送 recognize·
    非生物类永不发现。修后按簇路由：动物簇(2)+非生物簇(2) 各首 K → discover=4（两簇各 2）。
    """
    b, sid, es, ci, g = disc_env
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    非生物 = ci.ensure("非生物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    for tok in ["猫", "狗", "老鼠", "兔子"]:
        _inject_isa(disc_env, tok, 动物)
    for tok in ["石头", "砖", "墙", "地"]:
        _inject_isa(disc_env, tok, 非生物)
    s1 = _build_lang(disc_env, ["猫", "追", "老鼠"], seg_label="__route_b3_a1")
    s2 = _build_lang(disc_env, ["狗", "追", "兔子"], seg_label="__route_b3_a2")
    s3 = _build_lang(disc_env, ["石头", "追", "墙"], seg_label="__route_b3_b1")
    s4 = _build_lang(disc_env, ["砖", "追", "地"], seg_label="__route_b3_b2")
    discover_roots, recognize_roots = route_samples_for_discovery(
        b, g, [s1, s2, s3, s4], existing_keys=set(), existing_sigs=set(), space_id=sid)
    # Bug 3 核心断言：每簇各首 K=2 → discover=4（两簇都进 discover·非 cluster-blind 整组首 K=2）
    assert len(discover_roots) == 4, (
        f"Bug 3 per-cluster：动物簇(2)+非生物簇(2) 各 ≥K → 全 discover=4（原 cluster-blind 仅 2）·得 {discover_roots}")
    assert set(discover_roots) == {s1, s2, s3, s4}, f"4 样本皆 discover·得 {discover_roots}"
    # 顺序断言（bit-identical 守·对抗审 P1-1）：discover_roots = 簇创建序（_cluster_by_lca sort by NodeRef）
    # × 簇内 sorted。NodeRef 创建序 s1<s2<s3<s4 → 簇1(动物:s1,s2) 先建·簇2(非生物:s3,s4) 后建 → [s1,s2,s3,s4]。
    # 守未来 route 簇循环顺序漂移（如 dict 迭代序变）致 sample0 漂移破 bit-identical。
    assert discover_roots == [s1, s2, s3, s4], (
        f"顺序 bit-identical：簇创建序（NodeRef sort）× 簇内 sorted → [s1,s2,s3,s4]·得 {discover_roots}")
    assert recognize_roots == [], f"两簇各恰 K=2 无 held-out·得 {recognize_roots}"


def test_route_lang_resume_new_abstract_sig_discovered(disc_env):
    """**Bug 2 修（existing_keys 加 abstract_sig）**：载入动物类骨架（abstract_sig=(动物,追,动物)）→
    新同 (sig,arity) 异 abstract_sig 非生物类样本 → 按簇 abstract_sig 路由→**发现**（不送 recognize 静默丢）。

    原路由 (sig,arity) 键：非生物类同 (sig,arity=3) ∈ existing_keys → 全送 recognize → 不命中动物类骨架
    → 静默丢 → 非生物类本轮不发现（resume 渐失覆盖）。修后 (sig,arity,abstract_sig) 键：非生物类
    abstract_sig=(非生物,追,非生物) ∉ existing → discover。
    """
    b, sid, es, ci, g = disc_env
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    非生物 = ci.ensure("非生物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    for tok in ["猫", "狗", "老鼠", "兔子"]:
        _inject_isa(disc_env, tok, 动物)
    for tok in ["石头", "砖", "墙", "地"]:
        _inject_isa(disc_env, tok, 非生物)
    # 载入：先发现动物类骨架（+ 词例级·刀1 多级共存）
    a1 = _build_lang(disc_env, ["猫", "追", "老鼠"], seg_label="__route_b2_load1")
    a2 = _build_lang(disc_env, ["狗", "追", "兔子"], seg_label="__route_b2_load2")
    loaded = auto_discover_operators(
        [a1, a2], concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_BARE_TEXT)
    existing_keys, existing_sigs = _existing_keys_from_ops_lang(disc_env, loaded)
    # 新非生物类样本（同 (sig,arity=3)·异 abstract_sig）
    s3 = _build_lang(disc_env, ["石头", "追", "墙"], seg_label="__route_b2_new1")
    s4 = _build_lang(disc_env, ["砖", "追", "地"], seg_label="__route_b2_new2")
    discover_roots, recognize_roots = route_samples_for_discovery(
        b, g, [s3, s4], existing_keys=existing_keys, existing_sigs=existing_sigs, space_id=sid)
    # Bug 2 核心断言：非生物类 abstract_sig ∉ existing → discover（原路由误判"已载"全送 recognize）
    assert set(discover_roots) == {s3, s4}, (
        f"Bug 2：非生物类异 abstract_sig → discover（原 (sig,arity) 路由误送 recognize 静默丢）·得 {discover_roots}")
    assert recognize_roots == [], f"非生物类未载 → 不识别 → recognize 空·得 {recognize_roots}"


def test_route_lang_resume_same_abstract_sig_recognize(disc_env):
    """**Bug 2 正向守**：载入动物类骨架 → 新同 abstract_sig 动物类样本（狐狸/鸡·词级未学但 IS_A 动物）→
    簇 abstract_sig=(动物,追,动物) ∈ existing → 全 recognize（守幂等不 re-discover·跨 run 泛化）。"""
    b, sid, es, ci, g = disc_env
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    for tok in ["猫", "狗", "老鼠", "兔子", "狐狸", "狼", "鸡", "羊"]:
        _inject_isa(disc_env, tok, 动物)
    # 载入：先发现动物类骨架
    a1 = _build_lang(disc_env, ["猫", "追", "老鼠"], seg_label="__route_same_load1")
    a2 = _build_lang(disc_env, ["狗", "追", "兔子"], seg_label="__route_same_load2")
    loaded = auto_discover_operators(
        [a1, a2], concept_index=ci, edge_store=es, backend=b,
        space_id=sid, source=SOURCE_BARE_TEXT)
    existing_keys, existing_sigs = _existing_keys_from_ops_lang(disc_env, loaded)
    # 新动物类样本（词级未学·同 abstract_sig=(动物,追,动物)）
    s3 = _build_lang(disc_env, ["狐狸", "追", "鸡"], seg_label="__route_same_new1")
    s4 = _build_lang(disc_env, ["狼", "追", "羊"], seg_label="__route_same_new2")
    discover_roots, recognize_roots = route_samples_for_discovery(
        b, g, [s3, s4], existing_keys=existing_keys, existing_sigs=existing_sigs, space_id=sid)
    # 同 abstract_sig → recognize（守幂等·不 re-discover）
    assert discover_roots == [], f"同 abstract_sig 已载 → 不 re-discover·得 {discover_roots}"
    assert set(recognize_roots) == {s3, s4}, f"两 held-out 全识别·得 {recognize_roots}"
