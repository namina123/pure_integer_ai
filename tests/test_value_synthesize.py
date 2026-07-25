"""相1 G-PR1 算术归纳合成测试（doc §三/§二十·synthesize_value 行为匹配搜索 + verify round gate 接线）。

片1 unit：synthesize_value 纯机制 4 路径（AGREE / DISAGREE 牙 / held-out 泛化 / binding 搜索）。
片2 e2e：verify round gate 接线反 theater（OFF 构造性 reward=1 / ON 空池合成 reward=0·行为差可观测·非 theater）。

哲学定位（doc §20）：行为匹配=执行一致非意图正确（stable≠correct·#479 墙）·归纳合成=搜骨架池找行为匹配
（跨 item 骨架=泛化信号·非构造性 verify）·DISAGREE 牙（pool 无匹配返空·诚实 reward=0·非伪造）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_MATH
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
from pure_integer_ai.cognition.process.structure_discover import DiscoveredOperator
from pure_integer_ai.training.value_synthesize import synthesize_value
from pure_integer_ai.training.vm_proof import execute_composes_value
from pure_integer_ai.crosscut.integer import rational
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.formal_train import make_train_context, DefaultRoundRunner
from pure_integer_ai.training.stages import STAGE3_REWARD
from pure_integer_ai.cognition.shared.types import (
    CodeSpec, MODALITY_ARITH, DOMAIN_MATH, LANG_NONE, WEANING_PRE,
)


# ---- fixture helper：建单骨架 COMPOSES 树（mirror test_mode_b_cross_verify:68 _build_pair / test_stage9:68 _build） ----

def _build_skeleton(dsl: str, seg_label: str = "__seg_test"):
    """建 backend + space + 单 COMPOSES 骨架树·返 (graph, skeleton_ref, backend, space_id)。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "test")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    g = ConceptGraph(b)
    sid = sp.space_id
    root = ci.ensure(seg_label, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith(dsl, concept_index=ci, edge_store=es, backend=b,
                              space_id=sid, source=SOURCE_MATH, root_ref=root)
    return g, root, b, sid


def _op(skeleton_ref, arity: int, name: str = "__op_test") -> DiscoveredOperator:
    """造 DiscoveredOperator（单测用·name_ref=(0,0) 默认·synthesize_value 不读 name_ref）。"""
    return DiscoveredOperator(
        name=name, skeleton_ref=skeleton_ref, arity=arity,
        sample_count=1, name_ref=(0, 0))


def _arith_item(src: str, specs) -> CollectedItem:
    """造算术域 CollectedItem（arith_source + arith_specs·e2e verify round 用）。"""
    return CollectedItem(
        modality=MODALITY_ARITH, domain=DOMAIN_MATH, lang=LANG_NONE,
        source=SOURCE_MATH, arith_source=src, arith_specs=tuple(specs))


# ============ 片1 unit：synthesize_value 纯机制 4 路径 ============

def test_synthesize_agree_square():
    """AGREE：骨架池含 square（arity1）·spec={f(2)=4,f(3)=9}→synthesize_value 命中 square（identity binding (0,)）。"""
    g, square_ref, _b, _sid = _build_skeleton("lambda n: n * n")
    pool = [_op(square_ref, arity=1)]
    specs = (CodeSpec((2,), (4, 1)), CodeSpec((3,), (9, 1)))
    matches = synthesize_value(g, pool, specs)
    assert len(matches) == 1
    assert matches[0][0] == square_ref     # 命中 square 骨架
    assert matches[0][1] == (0,)           # identity binding（arity1·PARAM_0←input[0]）


def test_synthesize_disagree_no_match():
    """DISAGREE 牙（反 theater 核心）：骨架池只含 square·spec={f(2)=5}→square(2)=4≠5→返空（非伪造·诚实）。"""
    g, square_ref, _b, _sid = _build_skeleton("lambda n: n * n")
    pool = [_op(square_ref, arity=1)]
    specs = (CodeSpec((2,), (5, 1)),)   # square(2)=4≠5（行为不匹配）
    matches = synthesize_value(g, pool, specs)
    assert matches == []   # 无行为匹配→空（DISAGREE 牙·非 theater）


def test_synthesize_held_out_generalization():
    """held-out 泛化（测试级·防凑数骨架）：spec={f(5)=25}→搜到 square·test 自调 execute 验 held-out probe f(7)=49。

    held-out 是测试机制（测 synthesize_value soundness·§20.1 决断4）·非 production 机制。
    production 全 specs 搜索（教师 specs 即 ground truth）·held-out 在此由 test 独立 execute 验。
    """
    g, square_ref, _b, _sid = _build_skeleton("lambda n: n * n")
    pool = [_op(square_ref, arity=1)]
    specs = (CodeSpec((5,), (25, 1)),)   # 仅 1 spec 搜索
    matches = synthesize_value(g, pool, specs)
    assert len(matches) == 1
    assert matches[0][0] == square_ref
    # held-out probe（test 自调 execute·验搜到的骨架泛化·非凑数）
    v = execute_composes_value(g, square_ref, ((7, 1),))
    assert v is not None and rational.eq(v, rational.make(49, 1))   # square(7)=49（held-out ✓）


def test_synthesize_binding_search_same_arg():
    """binding 搜索：MUL 骨架（arity2）·spec={f(3)=9}（1 input）→绑定(PARAM_0=3,PARAM_1=3)变量同一性→MUL(3,3)=9 命中。

    arity2 != n_args1 → level2 enumerate_bindings(2,1)=[(0,0)]·binding(0,0)→PARAM_0←3,PARAM_1←3。
    证 PARAM 绑定搜索（非直接 arity 匹配）真能解变量同一性案例（§相1.4 测4）。
    """
    g, mul_ref, _b, _sid = _build_skeleton("lambda x, y: x * y")
    pool = [_op(mul_ref, arity=2)]
    specs = (CodeSpec((3,), (9, 1)),)   # 1 input·期望 9=MUL(3,3)
    matches = synthesize_value(g, pool, specs)
    assert len(matches) == 1
    assert matches[0][0] == mul_ref
    assert matches[0][1] == (0, 0)     # 同一性绑定（两 PARAM 都←input[0]=3）


def test_synthesize_empty_pool_or_specs():
    """边界守：空池 / 空 specs → 返空（无匹配诚实·caller 守·防御·不 crash）。"""
    g, square_ref, _b, _sid = _build_skeleton("lambda n: n * n")
    assert synthesize_value(g, [], (CodeSpec((2,), (4, 1)),)) == []          # 空池
    assert synthesize_value(g, [_op(square_ref, 1)], ()) == []                # 空 specs


# ============ 片2 e2e：verify round gate 接线反 theater（OFF 构造性 / ON 空池合成·行为差可观测） ============
# 经 run_round_full MODALITY_ARITH + WEANING_PRE·走 _run_verify_round PRE 分支（formal_train）。
# 两态行为差（OFF=1 构造性 / ON=0 空池合成）→ 证 gate ON 合成分支真活（非 theater·DISAGREE 牙）。

def test_verify_round_gate_off_constructive(monkeypatch):
    """gate OFF：verify round PRE 走既有 vm_proof 构造性 verify·square item + 匹配 specs → reward=1（构造性必然·bit-identical）。"""
    monkeypatch.setattr(gates, 'VALUE_SYNTHESIZE_MODE', False)
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_PRE
    item = _arith_item("lambda n: n * n", [CodeSpec((2,), (4, 1)), CodeSpec((3,), (9, 1))])
    res = DefaultRoundRunner().run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1   # 构造性 verify pass（gate OFF 既有 vm_proof 路径不变）


def test_verify_round_gate_on_empty_pool_disagree(monkeypatch):
    """gate ON：verify round PRE 走合成分支·空池（fresh ctx 无 discovered operator）→ synthesize_value 返空 → reward=0（DISAGREE 牙·诚实·非伪造）。

    证 gate ON 合成分支真活：同 item 同 specs·gate OFF 构造性 reward=1 vs gate ON 空池 reward=0·
    行为差可观测（非 theater）·DISAGREE 牙在 integration 层真活（pool 无行为匹配→诚实 0）。
    """
    monkeypatch.setattr(gates, 'VALUE_SYNTHESIZE_MODE', True)
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_PRE
    item = _arith_item("lambda n: n * n", [CodeSpec((2,), (4, 1)), CodeSpec((3,), (9, 1))])
    res = DefaultRoundRunner().run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 0   # 空池→无行为匹配→reward=0（gate ON 合成分支·DISAGREE 牙）


def test_verify_round_gate_on_match_root_redirect(monkeypatch):
    """gate ON 匹配正常路径（审1 finding2 fix-now 补）：pool 含 square·item 源是 identity（lambda n: n）+ square specs →
    synthesis 搜池命中 square → reward=1 + episode.ref=square_ref（搜索产物·**非** item 自观察 struct_ref）。

    证 root/sink 重指真活（Episode.ref=合成骨架·反 theater 正路径·非构造性 struct_refs[0]）。
    gate OFF 同 item 构造性 verify identity(2)=2≠4→reward=0（区分合成 vs 构造性·行为差可观测）。
    monkeypatch load_discovered_operators 控池（避 DISCOVERED attr 注册 complexity·integration path 真跑）。
    """
    from pure_integer_ai.experiments import round_runtime
    monkeypatch.setattr(gates, 'VALUE_SYNTHESIZE_MODE', True)
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_PRE
    # 在 ctx.space 建一个 square 骨架（合成搜索目标·seg label ≠ item 自观察 struct_ref）
    square_ref = ctx.concept_index.ensure(
        "__seg_synth_pool", space_id=ctx.space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith("lambda n: n * n", concept_index=ctx.concept_index,
                              edge_store=ctx.edge_store, backend=ctx.backend,
                              space_id=ctx.space_id, source=SOURCE_MATH, root_ref=square_ref)
    # monkeypatch load_discovered_operators 返 [square op]（控池·integration path 真跑·非 mock synthesize_value）
    monkeypatch.setattr(round_runtime, "load_discovered_operators",
                        lambda backend, *, space_id: [_op(square_ref, arity=1)])
    # item 源是 identity（自观察 struct_ref ≠ square_ref）+ square specs（gate OFF 构造性 identity(2)=2≠4→reward=0）
    item = _arith_item("lambda n: n", [CodeSpec((2,), (4, 1)), CodeSpec((3,), (9, 1))])
    res = DefaultRoundRunner().run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.reward == 1          # synthesis 命中 square（行为匹配全 specs·非构造性）
    assert res.episode.ref == square_ref    # 合成产物（root 重指·非 identity struct_refs[0]）


def test_value_synthesize_gate_default_off():
    """gate 默认 OFF（守 CI 回归·OFF → verify round 既有 vm_proof 路径 → 1909 bit-identical）。"""
    assert gates.VALUE_SYNTHESIZE_MODE is False
