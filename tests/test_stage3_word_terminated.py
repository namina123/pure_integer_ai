"""阶段3 第一消费者测试：attractor 越界②归位（删第四支）+ dag_path skip word_terminated
（experience_count 第一消费者·真行为变）。

覆盖（doc/重来_阶段3第一消费者设计补充.md + doc/重来_experience_count落地设计指引.md §八-bis
  + doc/重来_架构漂移审计_多维理解到因果单维.md §五越界②）：
  - T1-T5 单测 word_terminated：sink 保护 / gate① freq（backend None + 足 + 表未注册）/ hook 覆写
  - T6 反 theater 主锚：通识词 W 出边（W→X）不进 path.edges（对照 theta_freq 高=进·低=不进·真行为变）
  - T7 PRECEDES AND 汇聚点前驱被 skip → 非 REACHED_SINK（诚实降级·非 bug）
  - T8 默认 THETA_FREQ=1000 保守：小累积 e_tn 不 fire（守未来语料撞穿）
  - T9 默认参数 bit-identical：dag_path_step 不传 backend → eff_freq=0 → 不 fire → 正常 path
  - T10 attractor 删第四支回归：CAUSES success 不再冒充"已知"（仅 success 的 node 不扩张）

铁律：纯整数 / 单向依赖 / 不污染 concept_node 核心 / reward CAUSES-only 真墙 / sink 保护防误伤。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES, EDGE_CAUSES
from pure_integer_ai.storage.experience_count import (
    register_experience_count,
    record_base_freq, record_experience_outcome,
    read_effective_freq,
)
from pure_integer_ai.cognition.shared.types import (
    IntentType, INTENT_QUESTION, INTENT_COMMAND,
    TERMINAL_REACHED_SINK, TERMINAL_DEAD_END,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.process.dag_path import (
    dag_path_step, word_terminated, THETA_FREQ,
)
from pure_integer_ai.cognition.process.attractor import maybe_expand_attractor
from pure_integer_ai.cognition.process.a3_pr_wrapper import A3PRWrapper


# ---- fixtures ----

@pytest.fixture
def core():
    """建 backend + core 空间 + EdgeStore + register_experience_count。返 (backend, space_id, edge_store)。"""
    b = DictBackend()
    bootstrap(b)
    register_experience_count(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    yield b, sp.space_id, es
    b.close()


# ---- helpers ----

def _ref(sid, lid):
    return (sid, lid)


def _edge(b, es, sid, frm, to, et, *, strength=1, sn=0, tn=0, order_index=None):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
           order_index=order_index, sn=sn, tn=tn)


def _edge_in(pr, sid, frm, to, et):
    """pr.path.edges 含 (sid,frm,sid,to,et) 5-tuple？"""
    return (sid, frm, sid, to, et) in set(pr.path.edges)


# ============ T1-T5 单测 word_terminated（三 0/1 gate + sink 保护） ============

def test_word_terminated_sink_protect(core):
    """T1 sink 保护：c==intent.sink → 永不终止（达 sink 语义·防断路径·即便 base_freq 巨高）。"""
    b, sid, es = core
    record_base_freq(b, ref=_ref(sid, 5), base_freq=9999)   # sink 高频
    wm = WorkMemory()
    intent = IntentType(type=INTENT_COMMAND, sink=_ref(sid, 5))
    assert word_terminated(_ref(sid, 5), wm, b, intent=intent, theta_freq=5) is False


def test_word_terminated_no_backend(core):
    """T2 gate① freq 不足：backend=None → eff_freq=0 < theta_freq → False（守既有不传 backend 调用方 bit-identical）。"""
    b, sid, es = core
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 99))
    assert word_terminated(_ref(sid, 1), wm, None, intent=intent, theta_freq=1000) is False


def test_word_terminated_freq_gate_pass(core):
    """T3 gate① freq 足：register + record_base_freq(W,10) + theta_freq=5 → eff_freq=10≥5·三 gate 全 1 → True。"""
    b, sid, es = core
    record_base_freq(b, ref=_ref(sid, 1), base_freq=10)
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 99))
    assert word_terminated(_ref(sid, 1), wm, b, intent=intent, theta_freq=5) is True


def test_word_terminated_override_hooks(core, monkeypatch):
    """T4 gate②/③ hook 覆写：_ctx_override=1 或 _intent_override=1 → False（验 hook 机制建对·消费者后续阶段接）。"""
    b, sid, es = core
    record_base_freq(b, ref=_ref(sid, 1), base_freq=10)   # eff_freq=10 ≥ theta_freq=5·freq gate 过
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 99))
    import pure_integer_ai.cognition.process.dag_path as dp
    # gate② 语境覆写关停
    monkeypatch.setattr(dp, "_ctx_override", lambda c, wm: 1)
    assert word_terminated(_ref(sid, 1), wm, b, intent=intent, theta_freq=5) is False
    # gate③ 操作意图覆写关停（ctx 恢复 0）
    monkeypatch.setattr(dp, "_ctx_override", lambda c, wm: 0)
    monkeypatch.setattr(dp, "_intent_override", lambda c, i, wm, **kw: 1)
    assert word_terminated(_ref(sid, 1), wm, b, intent=intent, theta_freq=5) is False


def test_word_terminated_table_not_registered():
    """T5 backend 非 None 但 experience_count 表未注册 → read_effective_freq KeyError 兜底返 0 → False。"""
    b = DictBackend()
    bootstrap(b)   # 不 register_experience_count
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(1, 99))
    assert word_terminated((1, 1), wm, b, intent=intent, theta_freq=5) is False


# ============ T6 反 theater 主锚（真行为变） ============

def test_dag_path_skips_word_terminated_out_edges(core):
    """T6 反 theater 主锚：通识词 W 的出边（W→X）不进 path.edges（真行为变·§八-bis）。

    对照：theta_freq 高（eff_freq=10 < 1000·不 fire）→ W→X 进 path·达 sink。
    主锚：theta_freq 低（eff_freq=10 ≥ 5·fire）→ W skip → W not active → X 的 advance
          CAUSES 前驱 W 缺失 BLOCKED → W→X 不进 path·非 REACHED_SINK。

    注：path.edges 按"node 遍历时选定其入边"逆向构建·skip W → W 的入边（A→W）也不进 path
       （W 完全不处理）·权威断言是 W 出边 W→X 不进 path。
    """
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)   # A(1)→W(2)
    _edge(b, es, sid, 2, 3, EDGE_CAUSES, sn=1, tn=0)   # W(2)→X(3)
    record_base_freq(b, ref=_ref(sid, 2), base_freq=10)   # W 通识高频·eff_freq=10
    edges = b.select("edge")
    intent = IntentType(type=INTENT_COMMAND, sink=_ref(sid, 3))   # sink=X(3)·sink 保护不 skip X
    # 对照：theta_freq=1000（默认）·eff_freq=10 < 1000 → 不 fire → W→X 进 path·达 sink
    pr_hi = dag_path_step(edges, [_ref(sid, 1)], WorkMemory(), intent,
                          current_seq=0, backend=b, theta_freq=1000)
    assert _edge_in(pr_hi, sid, 2, 3, EDGE_CAUSES) is True
    assert pr_hi.terminal == TERMINAL_REACHED_SINK
    # 主锚：theta_freq=5·eff_freq=10 ≥ 5 → word_terminated(W) fire → W skip → W→X 不进 path
    pr_lo = dag_path_step(edges, [_ref(sid, 1)], WorkMemory(), intent,
                          current_seq=0, backend=b, theta_freq=5)
    assert _edge_in(pr_lo, sid, 2, 3, EDGE_CAUSES) is False   # W 出边不进 path（真行为变）
    assert pr_lo.terminal != TERMINAL_REACHED_SINK   # W skip 断路径·未达 sink


def test_dag_path_default_theta_freq_fires_on_high_base_freq(core):
    """T6c 生产默认 theta_freq fire 路径：record_base_freq(W,1000) + 默认 THETA_FREQ=1000 → W skip → W→X 不进 path。

    覆盖生产链路（默认参数·非人为压低 theta_freq）·验 base_freq≥THETA_FREQ 时真 fire。
    """
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)   # A→W
    _edge(b, es, sid, 2, 3, EDGE_CAUSES, sn=1, tn=0)   # W→X
    record_base_freq(b, ref=_ref(sid, 2), base_freq=1000)   # W 通识极高频·eff_freq=1000 ≥ THETA_FREQ
    edges = b.select("edge")
    intent = IntentType(type=INTENT_COMMAND, sink=_ref(sid, 3))
    # 默认 theta_freq=THETA_FREQ=1000·eff_freq=1000 ≥ 1000 → fire → W→X 不进 path
    pr = dag_path_step(edges, [_ref(sid, 1)], WorkMemory(), intent, current_seq=0, backend=b)
    assert _edge_in(pr, sid, 2, 3, EDGE_CAUSES) is False
    assert pr.terminal != TERMINAL_REACHED_SINK


# ============ T7 PRECEDES AND 汇聚点前驱被 skip → 诚实降级 ============

def test_dag_path_word_terminated_precedes_deadlock(core):
    """T7 PRECEDES AND 汇聚点前驱被 skip → 非 REACHED_SINK（诚实降级·非 bug·T7 覆盖）。

    seed=[S]·S→A(CAUSES)·S→W(CAUSES)·[A,W]→P(PRECEDES AND·同 order_index 汇聚)·P→X(CAUSES)·sink=X。
    W 通识 skip → P 的 PRECEDES AND 前驱不齐（W not active）→ P 永不 active → X 死路。
    首版接受此语义（通识词本不参与路径·阶段4 可加结构关键节点白名单）。
    """
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)            # S(1)→A(2)
    _edge(b, es, sid, 1, 3, EDGE_CAUSES, sn=1, tn=0)            # S(1)→W(3)
    _edge(b, es, sid, 2, 4, EDGE_PRECEDES, order_index=0)       # A(2)→P(4) PRECEDES
    _edge(b, es, sid, 3, 4, EDGE_PRECEDES, order_index=0)       # W(3)→P(4) PRECEDES（同 oi·AND 汇聚）
    _edge(b, es, sid, 4, 5, EDGE_CAUSES, sn=1, tn=0)            # P(4)→X(5)
    record_base_freq(b, ref=_ref(sid, 3), base_freq=10)         # W(3) 通识
    edges = b.select("edge")
    intent = IntentType(type=INTENT_COMMAND, sink=_ref(sid, 5))
    pr = dag_path_step(edges, [_ref(sid, 1)], WorkMemory(), intent,
                       current_seq=0, backend=b, theta_freq=5)
    # W skip → P 的 PRECEDES AND 前驱不齐 → P 永不 active → X 死路 → 非 REACHED_SINK
    assert pr.terminal != TERMINAL_REACHED_SINK


# ============ T8 默认 THETA_FREQ=1000 保守（负锚·守未来语料撞穿） ============

def test_word_terminated_default_theta_freq_conservative(core):
    """T8 默认 THETA_FREQ=1000 保守：小累积 e_tn 不 fire（bit-identical·守未来语料撞穿阈值）。

    模拟多 round reward feed（R1 符号·reward>0 → e_sn++/e_tn++）→ e_tn=5·eff_freq=5 < 默认 1000 → 不 fire。
    """
    b, sid, es = core
    for _ in range(5):
        record_experience_outcome(b, ref=_ref(sid, 2), reward=1)   # R1: reward>0 → sn++/tn++
    assert read_effective_freq(b, _ref(sid, 2)) == 5   # base_freq(0) + e_tn(5) = 5
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=_ref(sid, 99))
    # 默认 theta_freq=THETA_FREQ=1000·eff_freq=5 < 1000 → 不 fire
    assert word_terminated(_ref(sid, 2), wm, b, intent=intent) is False


# ============ T9 默认参数 bit-identical（dag_path_step 不传 backend） ============

def test_dag_path_step_default_no_backend_bit_identical(core):
    """T9 dag_path_step 默认参数（不传 backend）→ eff_freq=0 → word_terminated 不 fire → 正常 path（bit-identical）。"""
    b, sid, es = core
    _edge(b, es, sid, 1, 2, EDGE_CAUSES, sn=1, tn=0)
    _edge(b, es, sid, 2, 3, EDGE_CAUSES, sn=1, tn=0)
    edges = b.select("edge")
    intent = IntentType(type=INTENT_COMMAND, sink=_ref(sid, 3))
    pr = dag_path_step(edges, [_ref(sid, 1)], WorkMemory(), intent, current_seq=0)   # 默认无 backend
    # backend=None → eff_freq=0 → 不 fire → 正常 path·达 sink
    assert _edge_in(pr, sid, 2, 3, EDGE_CAUSES) is True
    assert pr.terminal == TERMINAL_REACHED_SINK


# ============ T10 attractor 删第四支回归（越界②归位） ============

def test_attractor_no_causes_success_entry():
    """T10 attractor 越界②归位：删第四支后·CAUSES success 不再冒充"已知"。

    构造 c(5) 仅有 CAUSES 入边高 success rate（原第四支触发源）·无入度/promoted/tier →
    删第四支后 entry=False → 不扩张（验漂移越界②归位）。
    """
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    sid = sp.space_id
    _edge(b, es, sid, 2, 5, EDGE_CAUSES, sn=10, tn=0)   # X(2)→c(5) rate=1000（高 success·原第四支触发源）
    edges = b.select("edge")
    w = A3PRWrapper.build(edges)
    w.solve([_ref(sid, 2)])
    e = {_ref(sid, 2)}
    # c(5) 仅有 CAUSES success（原第四支）·删后 entry 靠入度/promoted/tier·c(5) 全无 → 不扩张
    expanded = maybe_expand_attractor(_ref(sid, 5), e, w, edges, WorkMemory(), backend=b)
    assert expanded is False
    assert _ref(sid, 5) not in e


def test_attractor_entry_degree_still_works():
    """T10b 回归：删第四支不影响入度 entry（D(3) 入度≥θ_conv=2 → 扩张·test_stage4:532 同语义）。"""
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    sid = sp.space_id
    # 两 PRECEDES/T_STEP 入边 → c(3) 入度=2 ≥ θ_conv=2 → entry（第一支）
    _edge(b, es, sid, 1, 3, EDGE_PRECEDES, order_index=0)
    _edge(b, es, sid, 2, 3, EDGE_PRECEDES, order_index=0)
    edges = b.select("edge")
    w = A3PRWrapper.build(edges)
    w.solve([_ref(sid, 1), _ref(sid, 2)])
    e = {_ref(sid, 1), _ref(sid, 2)}
    maybe_expand_attractor(_ref(sid, 3), e, w, edges, WorkMemory(), backend=b)
    # 入度 entry 触发·若相干则入 e（删第四支不影响）
    if w.seed_rank(_ref(sid, 3)).num > 0:
        assert _ref(sid, 3) in e
