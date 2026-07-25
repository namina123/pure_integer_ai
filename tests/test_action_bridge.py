"""断桥 Phase A 测试（P2 G-PR2/3 cross-path·doc/重来_断桥设计refinement_2026-07-15）。

机制（weaning-safe 决断 A）：CollectedItem.action_specs（教师标 I/O 例·数据驱动非硬编码）→
_run_task_driven_generate 调 synthesize_value **联合匹配**全 specs（PbE·一动作多 I/O 例共定一骨架·反 per-spec 碎
·审2 F4）→ 独立 task-driven episode（**不替换 vm_proof verify round·不碎 W7**·反 VALUE_SYNTHESIZE 翻 ON 教训）。
断桥 cross-path：language/action item 经 action_specs 跨路径喂 arith 骨架池合成（**spec→synthesis**·intent 分类
=Phase B 动态构造器·Phase A 教师标 specs 已含 intent 语义·审2 F1/F2/F3）。

TC1 gate ON joint-match：language item + action_specs(2 square I/O 例) + pool 含 square → 联合命中 → 1 verified
   episode（联合匹配·非 per-spec 2 episode·审2 F4）+ words=实际产出值（re-execute·非 expected·审1 MEDIUM-1）。
TC2 gate OFF bit-identical：ACTION_BRIDGE_MODE OFF → action_specs 不消费 → 无 episode（既有行为零翻·episodes==[]）。
TC3 gate ON no-match：pool 空池 → synthesize 无匹配 → 无 episode（诚实 continue·同 arith no-viable）。
TC4 gate ON divergent-specs joint-reject：2 specs 共指不同骨架（square 满足 spec1 不满足 spec2）→ 联合无匹配
   → 无 episode（联合匹配强制一骨架满足全 specs·反 per-spec 各自命中·审2 F4）。

铁律：纯整数 / bit-identical（gate OFF 逐字现状·action_specs 默认 () CI 零数据）/ 反 theater（gate ON+联合匹配
才产 episode·独立 task-driven episode·weaning-safe 不触 verify round）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_MATH
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.cognition.understanding.arith_observe import build_composes_from_arith
from pure_integer_ai.cognition.process.structure_discover import DiscoveredOperator
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.formal_train import make_train_context, _run_task_driven_generate
from pure_integer_ai.cognition.shared.types import (CodeSpec, MODALITY_LANGUAGE, DOMAIN_TEXT, LANG_ZH,
    VERIFY_SOURCE_SELF_PRODUCED)


def _op(skeleton_ref, arity: int, name: str = "__op_test") -> DiscoveredOperator:
    """造 DiscoveredOperator（单测用·name_ref=(0,0)·synthesize_value 不读 name_ref·mirror test_value_synthesize:55）。"""
    return DiscoveredOperator(name=name, skeleton_ref=skeleton_ref, arity=arity,
                              sample_count=1, name_ref=(0, 0))


def _action_item(action_specs) -> CollectedItem:
    """language/action CollectedItem + action_specs（跨路径断桥用·非 arith_specs·避 arith task loop 双计）。"""
    return CollectedItem(modality=MODALITY_LANGUAGE, domain=DOMAIN_TEXT, lang=LANG_ZH,
                         source=SOURCE_MATH, action_specs=tuple(action_specs))


@pytest.fixture(autouse=True)
def _gate_reset():
    """每测前后复位 ACTION_BRIDGE_MODE + ACTION_BRIDGE_CUE_MODE（守测试隔离）。"""
    saved_a = gates.ACTION_BRIDGE_MODE
    saved_b = gates.ACTION_BRIDGE_CUE_MODE
    gates.ACTION_BRIDGE_MODE = False
    gates.ACTION_BRIDGE_CUE_MODE = False
    yield
    gates.ACTION_BRIDGE_MODE = saved_a
    gates.ACTION_BRIDGE_CUE_MODE = saved_b


def _square_ctx(monkeypatch):
    """make_train_context + 建 square 骨架 + monkeypatch load_discovered_operators 返 [square op]。"""
    from pure_integer_ai.experiments import task_generation_runtime
    b = DictBackend()
    ctx = make_train_context(b)
    square_ref = ctx.concept_index.ensure(
        "__seg_synth_pool", space_id=ctx.space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith("lambda n: n * n", concept_index=ctx.concept_index,
                              edge_store=ctx.edge_store, backend=ctx.backend,
                              space_id=ctx.space_id, source=SOURCE_MATH, root_ref=square_ref)
    monkeypatch.setattr(task_generation_runtime, "load_discovered_operators",
                        lambda backend, *, space_id: [_op(square_ref, arity=1)])
    return ctx, square_ref


# ---- TC1 gate ON joint-match（联合匹配全 specs → 1 verified episode·words=实际产出值） ----

def test_tc1_gate_on_joint_match_produces_episode(monkeypatch):
    """gate ON：language item + action_specs(2 square I/O 例) + pool 含 square → 联合命中 → 1 verified episode。

    联合匹配（审2 F4）：2 specs 共指 square（2→4 + 3→9）→ synthesize_value 联合验证全 specs 命中 square → **1**
    episode（非 per-spec 2 episode）。words=实际产出值（re-execute spec[0]=2→4·非 expected·审1 MEDIUM-1）。
    """
    ctx, square_ref = _square_ctx(monkeypatch)
    gates.ACTION_BRIDGE_MODE = True
    item = _action_item([CodeSpec((2,), (4, 1)), CodeSpec((3,), (9, 1))])
    episodes, summary = _run_task_driven_generate(ctx, [item], all_ops=[])
    bridge_eps = [e for e in episodes if e.ref == square_ref]
    assert len(bridge_eps) == 1, "联合匹配 2 specs 共指 square → 1 verified episode（非 per-spec 2）"
    assert bridge_eps[0].reward == 1, "命中→reward=1"
    assert not bridge_eps[0].vetoed, "命中→not vetoed"
    assert bridge_eps[0].output.parts[0].words == ["4/1"], "words=实际产出值（re-execute spec[0] square(2)=4·非 expected）"
    assert summary.verified >= 1, "summary verified 计断桥联合命中"


# ---- TC2 gate OFF bit-identical（action_specs 不消费 → 无 episode·episodes==[]） ----

def test_tc2_gate_off_no_bridge_episode(monkeypatch):
    """gate OFF：action_specs 不消费 → 无 episode（episodes==[]·language item 无 arith/code episode·既有行为零翻·
    bit-identical·CI action_specs 默认 () 亦同·审2 L3 加强：assert episodes 非仅 bridge_eps）。"""
    ctx, square_ref = _square_ctx(monkeypatch)
    # gate OFF（fixture 默认）
    item = _action_item([CodeSpec((2,), (4, 1))])
    episodes, _summary = _run_task_driven_generate(ctx, [item], all_ops=[])
    assert episodes == [], "gate OFF → 无 episode（language item + gate OFF → arith/code/断桥 全不产·bit-identical）"


# ---- TC3 gate ON no-match（pool 空池 → 无匹配 → 无 episode·诚实 continue） ----

def test_tc3_gate_on_no_match_no_episode(monkeypatch):
    """gate ON：action_specs(square) 但 pool 空池 → synthesize 无匹配 → 无 episode（诚实 continue·同 arith no-viable）。"""
    from pure_integer_ai.experiments import task_generation_runtime
    b = DictBackend()
    ctx = make_train_context(b)
    monkeypatch.setattr(task_generation_runtime, "load_discovered_operators",
                        lambda backend, *, space_id: [])   # 空池
    gates.ACTION_BRIDGE_MODE = True
    item = _action_item([CodeSpec((2,), (4, 1))])
    episodes, summary = _run_task_driven_generate(ctx, [item], all_ops=[])
    assert episodes == [], "空池无匹配 → 无 episode（诚实·不伪造·同 arith no-viable continue）"
    assert summary.total_tasks >= 1, "total_tasks 计 action_spec（搜了 pool·虽无匹配）"
    assert summary.verified == 0, "无匹配→verified=0"


# ---- TC4 gate ON divergent-specs joint-reject（联合匹配强制一骨架满足全 specs·反 per-spec） ----

def test_tc4_gate_on_divergent_specs_joint_reject(monkeypatch):
    """gate ON：2 specs 共指矛盾（square(2)=4 满足 spec1·square(2)=4≠8 不满足 spec2）→ 联合无匹配 → 无 episode。

    联合匹配（审2 F4）：synthesize_value 要求一骨架满足**全** specs·spec1=(2→4)+spec2=(2→8) 共指矛盾（square 满足
    spec1 不满足 spec2）→ 无单骨架满足全 → 无匹配 → 无 episode。反 per-spec（per-spec 会 spec1 命中 square 产 1
    episode·碎联合语义）。pool=[square]·square(2)=4。
    """
    ctx, square_ref = _square_ctx(monkeypatch)
    gates.ACTION_BRIDGE_MODE = True
    item = _action_item([CodeSpec((2,), (4, 1)), CodeSpec((2,), (8, 1))])   # spec1 square(2)=4 ✓·spec2 square(2)=4≠8 ✗
    episodes, summary = _run_task_driven_generate(ctx, [item], all_ops=[])
    assert episodes == [], "联合匹配：specs 共指矛盾（无单骨架满足全）→ 无 episode（反 per-spec 各自命中）"
    assert summary.total_tasks >= 1, "total_tasks 计 item（搜了 pool·虽联合无匹配）"
    assert summary.verified == 0, "联合无匹配→verified=0"


# ============================================================
# 断桥 Phase B 片1 测试（动态构造器 cue→spec·doc/重来_断桥设计refinement_2026-07-15 §Phase B 片1）
#
# 机制：无教师 action_specs 时·从 language text cues 动态构造 spec：CollectedItem.numeric_claims_flat
# （刀B observe 期 flatten·4-tuple `(left,op,right,result)`）→ CodeSpec 隐 op（input_args=(left,right)·
# expected=(result,1)·**op 隐藏**=synthesize 找算子非刀B 验算子·真合成）→ synthesize_value 联合匹配 →
# 独立 task-driven episode（weaning-safe 决断 A·不替换 vm_proof·不碎 W7）。
#
# TC5 gate ON cue→spec joint-match：2 ADD claims（op 隐）→ 联合命中 ADD 骨架 → 1 episode·words=实际产出值。
# TC6 gate OFF bit-identical：ACTION_BRIDGE_CUE_MODE OFF → numeric_claims_flat 不消费 → 无 episode。
# TC7 gate ON no-match：pool 空池 → 无匹配 → 无 episode（诚实 continue）。
# TC8 gate ON mixed-operator joint-reject：2 claims 隐含不同算子（2+3=5 ADD·6-2=4 SUB）→ 联合无单算子 → 无 episode。
# TC9 gate ON op-hidden synthesis：pool=[ADD, MUL]·2 ADD claims → synthesize 命中 ADD（MUL 不满足）→ 验隐 op 真合成
#    （spec 不携 op·synthesize 据 I/O 找对算子·非刀B 用 op 验证）。
# ============================================================

_OP_IGNORED = 0   # cue→spec 隐 op：Phase B block 用 c[0]/c[2]/c[3]·c[1](op) 不读（synthesize 找算子非验算子）·此值任意


def _add_ctx(monkeypatch, *, with_mul: bool = False):
    """make_train_context + 建 ADD（arity=2）骨架 [+ 可选 MUL] + monkeypatch load_discovered_operators。"""
    from pure_integer_ai.experiments import task_generation_runtime
    b = DictBackend()
    ctx = make_train_context(b)
    add_ref = ctx.concept_index.ensure(
        "__seg_add_op", space_id=ctx.space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
    build_composes_from_arith("lambda x, y: x + y", concept_index=ctx.concept_index,
                              edge_store=ctx.edge_store, backend=ctx.backend,
                              space_id=ctx.space_id, source=SOURCE_MATH, root_ref=add_ref)
    pool = [_op(add_ref, arity=2)]
    mul_ref = None
    if with_mul:
        mul_ref = ctx.concept_index.ensure(
            "__seg_mul_op", space_id=ctx.space_id, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)
        build_composes_from_arith("lambda x, y: x * y", concept_index=ctx.concept_index,
                                  edge_store=ctx.edge_store, backend=ctx.backend,
                                  space_id=ctx.space_id, source=SOURCE_MATH, root_ref=mul_ref)
        pool.append(_op(mul_ref, arity=2))
    monkeypatch.setattr(task_generation_runtime, "load_discovered_operators",
                        lambda backend, *, space_id: pool)
    return ctx, add_ref, mul_ref


def _cue_item(numeric_claims_flat) -> CollectedItem:
    """language CollectedItem + numeric_claims_flat（断桥 Phase B cue→spec 用·刀B observe 期 flatten 产物模拟）。"""
    return CollectedItem(modality=MODALITY_LANGUAGE, domain=DOMAIN_TEXT, lang=LANG_ZH,
                         source=SOURCE_MATH, numeric_claims_flat=tuple(numeric_claims_flat))


# ---- TC5 gate ON cue→spec joint-match（2 ADD claims → 联合命中 ADD → 1 episode·words=实际值） ----

def test_tc5_gate_on_cue_spec_joint_match(monkeypatch):
    """gate ON：language item + numeric_claims_flat[(2,_,3,5),(4,_,1,5)]（2 ADD claims·op 隐）+ pool=[ADD] →
    联合命中 ADD（2+3=5 ✓·4+1=5 ✓）→ 1 verified episode·words=实际产出值（re-execute spec[0] ADD(2,3)=5·非 expected）。"""
    ctx, add_ref, _ = _add_ctx(monkeypatch)
    gates.ACTION_BRIDGE_CUE_MODE = True
    item = _cue_item([(2, _OP_IGNORED, 3, 5), (4, _OP_IGNORED, 1, 5)])
    episodes, summary = _run_task_driven_generate(ctx, [item], all_ops=[])
    cue_eps = [e for e in episodes if e.ref == add_ref]
    assert len(cue_eps) == 1, "联合匹配 2 ADD claims 共指 ADD → 1 verified episode"
    assert cue_eps[0].reward == 1, "命中→reward=1"
    assert not cue_eps[0].vetoed, "命中→not vetoed"
    assert cue_eps[0].output.parts[0].words == ["5/1"], "words=实际产出值（re-execute spec[0] ADD(2,3)=5·非 expected）"
    assert cue_eps[0].verify_source == VERIFY_SOURCE_SELF_PRODUCED, (
        "审2 LOW-1：cue-derived spec.expected 来自 text cues（single-source·非 R6 外部源·同刀B SELF_PRODUCED）"
        "→ SELF_PRODUCED 守'全自产不准停'（text-derived synthesis 不准驱动停止决策·反 theater）·非 Phase A 教师标 EXTERNAL")
    assert summary.verified >= 1, "summary verified 计 cue→spec 联合命中"


# ---- TC6 gate OFF bit-identical（numeric_claims_flat 不消费 → 无 episode·episodes==[]） ----

def test_tc6_gate_off_cue_no_episode(monkeypatch):
    """gate OFF：numeric_claims_flat 不消费 → 无 episode（episodes==[]·bit-identical·CI numeric_claims_flat 默认 () 亦同）。"""
    ctx, add_ref, _ = _add_ctx(monkeypatch)
    # gate OFF（fixture 默认）
    item = _cue_item([(2, _OP_IGNORED, 3, 5)])
    episodes, _summary = _run_task_driven_generate(ctx, [item], all_ops=[])
    assert episodes == [], "gate OFF → 无 episode（language item + gate OFF → arith/code/断桥 全不产·bit-identical）"


# ---- TC7 gate ON no-match（pool 空池 → 无匹配 → 无 episode·诚实 continue） ----

def test_tc7_gate_on_cue_no_match_no_episode(monkeypatch):
    """gate ON：numeric_claims_flat(ADD claims) 但 pool 空池 → synthesize 无匹配 → 无 episode（诚实 continue·同 Phase A TC3）。"""
    from pure_integer_ai.experiments import task_generation_runtime
    b = DictBackend()
    ctx = make_train_context(b)
    monkeypatch.setattr(task_generation_runtime, "load_discovered_operators",
                        lambda backend, *, space_id: [])   # 空池
    gates.ACTION_BRIDGE_CUE_MODE = True
    item = _cue_item([(2, _OP_IGNORED, 3, 5)])
    episodes, summary = _run_task_driven_generate(ctx, [item], all_ops=[])
    assert episodes == [], "空池无匹配 → 无 episode（诚实·不伪造·同 arith no-viable continue）"
    assert summary.total_tasks >= 1, "total_tasks 计 cue item（搜了 pool·虽无匹配）"
    assert summary.verified == 0, "无匹配→verified=0"


# ---- TC8 gate ON mixed-operator joint-reject（2 claims 隐含不同算子 → 联合无单算子 → 无 episode） ----

def test_tc8_gate_on_mixed_operator_joint_reject(monkeypatch):
    """gate ON：numeric_claims_flat[(2,_,3,5),(6,_,2,4)]（claim1 隐 ADD 2+3=5·claim2 隐 SUB 6-2=4）+ pool=[ADD] →
    联合无匹配（ADD 满足 claim1 2+3=5 ✓·不满足 claim2 6+2=8≠4 ✗）→ 无 episode。反 per-claim 各自命中（碎联合语义）。
    op 隐：claim2 的 result=4 驱动匹配（非 op）·ADD 给 8≠4 → 联合拒。"""
    ctx, add_ref, _ = _add_ctx(monkeypatch)
    gates.ACTION_BRIDGE_CUE_MODE = True
    item = _cue_item([(2, _OP_IGNORED, 3, 5), (6, _OP_IGNORED, 2, 4)])   # claim1 ADD 2+3=5 ✓·claim2 ADD 6+2=8≠4 ✗
    episodes, summary = _run_task_driven_generate(ctx, [item], all_ops=[])
    assert episodes == [], "联合匹配：claims 隐含不同算子（无单骨架满足全）→ 无 episode（反 per-claim 各自命中）"
    assert summary.total_tasks >= 1, "total_tasks 计 cue item（搜了 pool·虽联合无匹配）"
    assert summary.verified == 0, "联合无匹配→verified=0"


# ---- TC9 gate ON op-hidden synthesis（pool=[ADD,MUL]·2 ADD claims → 命中 ADD 非 MUL·验隐 op 真合成） ----

def test_tc9_gate_on_op_hidden_synthesis_picks_correct_operator(monkeypatch):
    """gate ON：pool=[ADD, MUL]·numeric_claims_flat[(2,_,3,5),(4,_,1,5)]（2 ADD claims·op 隐）→ synthesize 命中 ADD
    （ADD: 2+3=5 ✓·4+1=5 ✓·MUL: 2*3=6 ✗）→ episode.ref==add_ref（非 mul_ref）·验隐 op 真合成：
    spec 不携 op·synthesize 据 I/O 找对算子 ADD（非刀B 用 op 验证）·op 字段被忽略（_OP_IGNORED 任意值同效）。"""
    ctx, add_ref, mul_ref = _add_ctx(monkeypatch, with_mul=True)
    gates.ACTION_BRIDGE_CUE_MODE = True
    item = _cue_item([(2, _OP_IGNORED, 3, 5), (4, _OP_IGNORED, 1, 5)])
    episodes, summary = _run_task_driven_generate(ctx, [item], all_ops=[])
    cue_eps = [e for e in episodes if e.ref in (add_ref, mul_ref)]
    assert len(cue_eps) == 1, "联合命中一算子 → 1 episode"
    assert cue_eps[0].ref == add_ref, "隐 op 真合成：据 I/O 命中 ADD（2+3=5·4+1=5）·非 MUL（2*3=6≠5）"
    assert cue_eps[0].reward == 1, "命中→reward=1"
    assert cue_eps[0].output.parts[0].words == ["5/1"], "words=实际产出值（re-execute ADD(2,3)=5）"


def test_typed_generation_owner_disables_legacy_language_bridges(monkeypatch):
    """typed owner 下语言 action/cue 不得在 finalize 另产 legacy 标量 episode。"""
    ctx, _add_ref, _ = _add_ctx(monkeypatch)
    ctx.language_generation_runtime = object()
    gates.ACTION_BRIDGE_MODE = True
    gates.ACTION_BRIDGE_CUE_MODE = True
    item = _action_item((CodeSpec((2, 3), (5, 1)),))
    item.numeric_claims_flat = ((2, _OP_IGNORED, 3, 5),)

    episodes, summary = _run_task_driven_generate(
        ctx,
        [item],
        all_ops=[],
    )

    assert episodes == []
    assert summary.total_tasks == 0
    assert summary.selected == 0
    assert summary.verified == 0
