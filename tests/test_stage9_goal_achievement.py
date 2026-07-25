"""阶段9 目标达成判据测试：attractor 第一本职"目标达成"=控制环收敛判据（达 sink 判覆盖率主导）。

覆盖（doc/重来_目标达成判据设计补充.md + legacy doc/吸引子动力学与目标达成.md §二.1/§六 + plan）：
  - T1 覆盖率停止：达 sink + 路径已到达节点(stepper.active) 覆盖目标骨架≥阈值 → REACHED_SINK（真达成）
  - T2 反 theater 主锚：达 sink 但骨架未覆盖够（key 含路径外元素）→ DEAD_END（走到 sink 不=达成·真行为变）
  - T3 bit-identical：key_skeleton=None/threshold=0 退化 → 达 sink REACHED_SINK（既有行为·零 break）
  - T4 COMMAND 路由：type=COMMAND 覆盖够 → REACHED_SINK（QUESTION/COMMAND 都活）
  - T5 STATEMENT 不判达成：type=STATEMENT sink=None → 达 sink 判定不进 → 层尽 DEAD_END（现状·设计原意）
  - T6 e2e reward 反映达成：episode_loop 覆盖够 → REACHED_SINK reward≥0；覆盖不足 → DEAD_END reward<0
  - T7 ctx_code 透传：ctx_code 非 0·判据仍活（覆盖够达成）
  - T8 lint clean（python -m pure_integer_ai.crosscut.guards.lint·命令验证非 pytest）

铁律：纯整数（coverage_overlap ×1000 集合覆盖）/ 单向依赖（dag_path→a4_align 同层 cognition.process·judge:38 先例）/
  §8.1c（结构重合非统计标签·LCS 集合覆盖）/ 反 theater（达 sink 判覆盖·T2 覆盖不足 DEAD_END 真行为变·判据影响 terminal→reward）/
  bit-identical（threshold=0/key_skeleton=None 退化·T3·782 既有测零 break 根）。

★关键决断（偏离 plan 字面·有技术理由·动工期定）：
  1. 已到达源 = stepper.active（非 plan struct_unit_refs·后者只 AND 汇聚点·不含 seed/sink/普通链节点·
     用它会 break key_skeleton=[sink] 既有 e2e）。active = dag_path:190 "active 即已到达" 精确语义。
  2. 达 sink 判覆盖·非 plan"OR 并列"/"每-node 提前停"/"层尽判"：
     - OR 并列与反 theater 主锚矛盾（OR 走到 sink 就达成·主锚要求覆盖不足不达成）。
     - 每-node 提前停：source(active={seed}) 过早 50% 触发 + path 未含 CAUSES 边→reward feed 断（propagate 只 feed path.edges CAUSES）。
     - 层尽判：走过头（sink 后节点）触发 attractor K_CAP_SOFT 溢出 min(Rational) 阶段8 latent。
     - 达 sink 判 = 既有终止点·path 含到 sink 全步进边（feed 完整）+ 不走过头（避 latent）+ 主锚成立。
  3. ordered=False（active 无序 set·集合覆盖·与 judge J1 有序 parts 不同源不同序）。
  4. bit-identical 守卫 = coverage_threshold>0 启用（默认 0·既有测试全不传→全退化→零 break）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_CAUSES, EDGE_PRECEDES
from pure_integer_ai.storage.experience_count import register_experience_count, pack_ctx_code
from pure_integer_ai.cognition.shared.types import (
    InputPayload, IntentType, INTENT_QUESTION, INTENT_COMMAND, INTENT_STATEMENT,
    DOMAIN_TEXT, MODALITY_LANGUAGE,
    TERMINAL_REACHED_SINK, TERMINAL_DEAD_END, REWARD_DEAD_END,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.process.dag_path import dag_path_step


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def backend(request):
    """两 backend 一致性（覆盖率判据 backend 无关·纯 stepper.active 集合运算）。"""
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    register_experience_count(b)
    yield b
    b.close()


@pytest.fixture
def core():
    """DictBackend + core 空间 + EdgeStore + ConceptIndex + register experience_count。"""
    b = DictBackend()
    bootstrap(b)
    register_experience_count(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    yield b, sp.space_id, es, ci
    b.close()


# ---- helpers ----

def _edge(b, es, sid, frm, to, et, *, strength=1, sn=0, tn=0):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
           sn=sn, tn=tn)


def _input(sid, key_skeleton):
    return InputPayload(segments=[], source=SOURCE_BARE_TEXT, stage=1,
                        domain=DOMAIN_TEXT, key_skeleton=list(key_skeleton))


# ============ T1 覆盖率停止（达 sink + 覆盖够 → REACHED_SINK）============

def test_goal_coverage_reached_sink(backend):
    """T1 达 sink + 路径已到达节点覆盖全目标骨架≥阈值 → REACHED_SINK（真达成）。

    A→B→C PRECEDES 链·seeds=[A] sink=C key_skeleton=[A,B,C] threshold=900。
    路径 A→B→C·stepper.active={A,B,C}·覆盖 key{A,B,C}=1000×3/3=1000≥900 → REACHED_SINK。
    attractor 第一本职"目标达成"落地（控制环收敛判据）。
    """
    b = backend
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    sid = sp.space_id
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    A = ci.ensure("A", space_id=sid)
    B = ci.ensure("B", space_id=sid)
    C = ci.ensure("C", space_id=sid)
    _edge(b, es, sid, A[1], B[1], EDGE_PRECEDES)
    _edge(b, es, sid, B[1], C[1], EDGE_PRECEDES)
    edges = b.select("edge")
    intent = IntentType(type=INTENT_QUESTION, sink=C)
    res = dag_path_step(edges, [A], WorkMemory(), intent, backend=b, ctx_code=0,
                        key_skeleton=[A, B, C], coverage_threshold=900)
    assert res.terminal == TERMINAL_REACHED_SINK
    assert res.sink == C


# ============ T2 反 theater 主锚（达 sink 覆盖不足 → DEAD_END·真行为变）============

def test_goal_coverage_insufficient_dead_end(backend):
    """T2 反 theater 主锚：达 sink 但骨架未覆盖够 → DEAD_END（走到 sink 不=达成·判据真区分）。

    A→B→C 链·seeds=[A] sink=C·key_skeleton=[A,X,C]（X 不在路径·目标骨架有未到达部分）threshold=900。
    路径 A→B→C·active={A,B,C}·覆盖 key{A,X,C}=1000×|{A,X,C}∩{A,B,C}|/3=1000×2//3=666<900 → DEAD_END。
    对照 T1：同路径同 sink·key 多一未到达 X 即不达成（判据真活·非只 sink 节点量化）。
    这是 attractor 第一本职的核心——做成事(覆盖目标骨架)≠走到点(达 sink)。
    """
    b = backend
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    sid = sp.space_id
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    A = ci.ensure("A", space_id=sid)
    B = ci.ensure("B", space_id=sid)
    C = ci.ensure("C", space_id=sid)
    X = (sid, 999)   # 虚构 ConceptRef（不在图·不在路径·key_skeleton 含它=目标骨架有未到达部分）
    _edge(b, es, sid, A[1], B[1], EDGE_PRECEDES)
    _edge(b, es, sid, B[1], C[1], EDGE_PRECEDES)
    edges = b.select("edge")
    intent = IntentType(type=INTENT_QUESTION, sink=C)
    res = dag_path_step(edges, [A], WorkMemory(), intent, backend=b, ctx_code=0,
                        key_skeleton=[A, X, C], coverage_threshold=900)
    assert res.terminal == TERMINAL_DEAD_END   # 走到 sink=C 但 X 未覆盖→不达成


# ============ T3 bit-identical（退化 → 既有 sink 判定·零 break）============

def test_bit_identical_default_threshold_zero(core):
    """T3 bit-identical：key_skeleton=None/threshold=0 退化 → 达 sink REACHED_SINK（既有行为）。

    默认 key_skeleton=None/coverage_threshold=0 → 不启用覆盖判据 → 退化既有 sink 节点判定（达 sink ∧ J4）。
    既有 dag_path 行为不变（782 既有测零 break 的根·生产 formal_train 显式传 threshold 才启用）。
    """
    b, sid, es, ci = core
    A = ci.ensure("A", space_id=sid)
    B = ci.ensure("B", space_id=sid)
    _edge(b, es, sid, A[1], B[1], EDGE_PRECEDES)
    edges = b.select("edge")
    intent = IntentType(type=INTENT_QUESTION, sink=B)
    # 默认（不传 key_skeleton/coverage_threshold）→ 退化 sink 判定
    res_default = dag_path_step(edges, [A], WorkMemory(), intent, backend=b, ctx_code=0)
    assert res_default.terminal == TERMINAL_REACHED_SINK
    # 显式 threshold=0 也退化（即使 key_skeleton 非空·threshold=0 不启用）
    res_zero = dag_path_step(edges, [A], WorkMemory(), intent, backend=b, ctx_code=0,
                             key_skeleton=[A, B], coverage_threshold=0)
    assert res_zero.terminal == TERMINAL_REACHED_SINK
    assert res_zero.sink == B


# ============ T4 COMMAND 路由（QUESTION/COMMAND 都活）============

def test_command_route_coverage(core):
    """T4 COMMAND 路由：type=COMMAND 达 sink + 覆盖够 → REACHED_SINK（Q/C 都活·STATEMENT 不活见 T5）。"""
    b, sid, es, ci = core
    A = ci.ensure("A", space_id=sid)
    B = ci.ensure("B", space_id=sid)
    _edge(b, es, sid, A[1], B[1], EDGE_PRECEDES)
    edges = b.select("edge")
    intent = IntentType(type=INTENT_COMMAND, sink=B)
    res = dag_path_step(edges, [A], WorkMemory(), intent, backend=b, ctx_code=0,
                        key_skeleton=[A, B], coverage_threshold=900)
    assert res.terminal == TERMINAL_REACHED_SINK


# ============ T5 STATEMENT 不判达成（设计原意·保持现状）============

def test_statement_no_goal_coverage(core):
    """T5 STATEMENT 不判达成：type=STATEMENT sink=None → 达 sink 判定不进 → 层尽 DEAD_END（现状）。

    STATEMENT sink=None → 达 sink 判定 (intent.sink is not None) 不进 → 覆盖判据 Q/C 分流不活 →
    走完层尽 DEAD_END。陈述不步进取证（types.py:208 设计原意）·S1 不改步进只加判据·保持现状。
    """
    b, sid, es, ci = core
    A = ci.ensure("A", space_id=sid)
    B = ci.ensure("B", space_id=sid)
    _edge(b, es, sid, A[1], B[1], EDGE_PRECEDES)
    edges = b.select("edge")
    intent = IntentType(type=INTENT_STATEMENT, sink=None)
    res = dag_path_step(edges, [A], WorkMemory(), intent, backend=b, ctx_code=0,
                        key_skeleton=[A, B], coverage_threshold=900)
    assert res.terminal == TERMINAL_DEAD_END   # STATEMENT 不判达成·层尽 DEAD_END（现状）


# ============ T6 e2e reward 反映达成（判据影响 terminal→reward）============

def test_episode_reward_reflects_goal_achievement(core):
    """T6 e2e：episode_loop 覆盖够 → REACHED_SINK reward≥0；覆盖不足 → DEAD_END reward<0。

    反 theater 真行为变：判据落 dag_path 终点（影响 terminal）→ episode reward（DEAD_END reward<0 /
    REACHED_SINK judge_fn=None reward=0）·不只落在 judge J1 量化。generate/judge None（REACHED_SINK
    reward=0 veto 语义·DEAD_END reward=-1·episode.py 两半边）。
    """
    from pure_integer_ai.cognition.process.episode import episode_loop
    b, sid, es, ci = core
    A = ci.ensure("A", space_id=sid)
    B = ci.ensure("B", space_id=sid)
    _edge(b, es, sid, A[1], B[1], EDGE_PRECEDES)
    edges = b.select("edge")
    # 覆盖够：key=[A,B] threshold=900 → 达 B active={A,B} 覆盖1000 → REACHED_SINK reward=0
    inp_ok = _input(sid, [A, B])
    intent_ok = IntentType(type=INTENT_QUESTION, sink=B)
    _out_ok, ep_ok = episode_loop(inp_ok, edges, [A], WorkMemory(), intent_ok,
                                  generate_fn=None, judge_fn=None,
                                  edge_store=es, backend=b, coverage_threshold=900)
    assert ep_ok.terminal == TERMINAL_REACHED_SINK
    assert ep_ok.reward >= 0
    # 覆盖不足：key=[A,X,B] X 不在路径 threshold=900 → DEAD_END reward=-1
    X = (sid, 999)
    inp_bad = _input(sid, [A, X, B])
    intent_bad = IntentType(type=INTENT_QUESTION, sink=B)
    _out_bad, ep_bad = episode_loop(inp_bad, edges, [A], WorkMemory(), intent_bad,
                                    generate_fn=None, judge_fn=None,
                                    edge_store=es, backend=b, coverage_threshold=900)
    assert ep_bad.terminal == TERMINAL_DEAD_END
    assert ep_bad.reward == REWARD_DEAD_END   # <0


# ============ T7 ctx_code 透传（判据不破 ctx_code 路径）============

def test_ctx_code_passes_through(core):
    """T7 ctx_code 非 0·判据仍活（覆盖够达成·ctx_code 透传不破判据·与 word_terminated 同链）。"""
    b, sid, es, ci = core
    A = ci.ensure("A", space_id=sid)
    B = ci.ensure("B", space_id=sid)
    _edge(b, es, sid, A[1], B[1], EDGE_PRECEDES)
    edges = b.select("edge")
    intent = IntentType(type=INTENT_QUESTION, sink=B)
    ctx = pack_ctx_code(DOMAIN_TEXT, MODALITY_LANGUAGE, 0, INTENT_QUESTION)
    res = dag_path_step(edges, [A], WorkMemory(), intent, backend=b, ctx_code=ctx,
                        key_skeleton=[A, B], coverage_threshold=900)
    assert res.terminal == TERMINAL_REACHED_SINK   # ctx_code 非 0·判据仍活


# ============ T8 边界阈值（整除边界 + oracle 上限守·Agent1 F2）============

def test_threshold_boundary_full_coverage(core):
    """T8 边界阈值：threshold=1000（恰好满覆盖）→ REACHED_SINK；threshold=1001（>满分）→ DEAD_END。

    守整除边界（满覆盖 1000×N/N=1000·阈值=1000 边界含）+ oracle 上限
    （COVERAGE_THRESHOLD 须 ≤1000·>1000 让所有达成变 DEAD_END·防 oracle 误设）。
    生产 COVERAGE_THRESHOLD=500 远<1000 安全。
    """
    b, sid, es, ci = core
    A = ci.ensure("A", space_id=sid)
    B = ci.ensure("B", space_id=sid)
    _edge(b, es, sid, A[1], B[1], EDGE_PRECEDES)
    edges = b.select("edge")
    intent = IntentType(type=INTENT_QUESTION, sink=B)
    # threshold=1000：满覆盖（active={A,B}·key=[A,B]·覆盖 1000×2/2=1000）→ REACHED_SINK（边界含）
    res_1000 = dag_path_step(edges, [A], WorkMemory(), intent, backend=b, ctx_code=0,
                             key_skeleton=[A, B], coverage_threshold=1000)
    assert res_1000.terminal == TERMINAL_REACHED_SINK
    # threshold=1001：>满分 1000·即使满覆盖 1000 也<1001 → DEAD_END（oracle 上限守·防误设）
    res_1001 = dag_path_step(edges, [A], WorkMemory(), intent, backend=b, ctx_code=0,
                             key_skeleton=[A, B], coverage_threshold=1001)
    assert res_1001.terminal == TERMINAL_DEAD_END
