"""阶段6 复合 key 第二刀测试：experience_count 复合 key 启用·解锁反讽固化层基建①。

覆盖（doc/重来_阶段6复合key设计补充.md + doc/重来_experience_count落地设计指引.md §点2/§点4
  + doc/重来_三把钥匙会师_含义命中结构发现attractor方案探_修正分析四.md §九）：
  - T1-T4 pack_ctx_code：位打包纯整数 / 全 0 退化 / 边界 255 / 维独立不污染
  - T5-T7 read_effective_freq 桶分离：ctx=0 退化同行读 / ctx≠0 base (0,0) e_tn ctx / ctx≠0 冷启动 base=0
  - T8-T9 feed 透传：propagate_reward feed 到非 0 桶 / 0 桶未污染（防混淆分桶主锚）
  - T10-T11 gate① 按桶读 e2e（反 theater 主锚·真行为变）：同概念两 ctx 不同终止 / dag_path_step ctx_code 透传
  - T12 bit-identical：dag_path_step 默认 ctx_code=0 退化（与阶段3 T6c 同语义）
  - T13 speaker_code 恒 0（defer #495 记忆空间层·桶列留值空）

铁律：纯整数（位打包 << / |）/ 单向依赖（L5→L0）/ 不污染 concept_node 核心 / §8.1c（计数测度分桶非结构语义标签）/
  反 theater（write+read+gate① 三件全活·T10/T11 真行为变）/ bit-identical（默认 0 桶退化）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_CAUSES
from pure_integer_ai.storage.experience_count import (
    register_experience_count,
    record_base_freq, record_experience_outcome,
    read_experience_count, read_effective_freq,
    pack_ctx_code, DEFAULT_CTX_CODE,
)
from pure_integer_ai.cognition.shared.types import (
    PathData, PathResult, IntentType,
    INTENT_QUESTION, INTENT_COMMAND,
    DOMAIN_TEXT, DOMAIN_MATH, MODALITY_LANGUAGE, MODALITY_ARITH,
    TERMINAL_REACHED_SINK,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.process.dag_path import dag_path_step, word_terminated
from pure_integer_ai.cognition.process.reward_propagate import propagate_reward


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def backend(request):
    """bootstrap + register_experience_count（pack_ctx_code / 桶分离单测用）。"""
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    register_experience_count(b)
    yield b
    b.close()


@pytest.fixture
def core():
    """建 backend + core 空间 + EdgeStore + ConceptIndex + register experience_count。

    返 (backend, space_id, edge_store, concept_index)——建 concept + 边 + propagate / dag_path e2e。
    """
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

def _edge(b, es, sid, frm, to, et, *, strength=1, sn=0, tn=0, order_index=None):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
           order_index=order_index, sn=sn, tn=tn)


def _path_result(edges, *, sink=None):
    pd = PathData()
    pd.edges = list(edges)
    return PathResult(path=pd, terminal=TERMINAL_REACHED_SINK, sink=sink)


def _edge_in(pr, sid, frm, to, et):
    """pr.path.edges 含 (sid,frm,sid,to,et) 5-tuple？"""
    return (sid, frm, sid, to, et) in set(pr.path.edges)


# ============ T1-T4 pack_ctx_code 位打包纯整数 ============

def test_pack_ctx_code_dimensions():
    """T1 ctx_tag 四维位打包纯整数（domain<<24 | modality<<16 | task<<8 | intent_type·bit-identical）。"""
    assert pack_ctx_code(2, 6, 0, INTENT_COMMAND) == (2 << 24) | (6 << 16) | (0 << 8) | INTENT_COMMAND
    assert pack_ctx_code(1, 1, 0, INTENT_QUESTION) == (1 << 24) | (1 << 16) | 1


def test_pack_ctx_code_all_zero_default():
    """T2 全 0 → 0 = DEFAULT_CTX_CODE（单 key 退化·第一刀恒 0·向后兼容）。"""
    assert pack_ctx_code(0, 0, 0, 0) == 0
    assert pack_ctx_code(0, 0, 0, 0) == DEFAULT_CTX_CODE


def test_pack_ctx_code_boundary_255():
    """T3 各维 255 边界不溢出（8 bit·不污染相邻维）。"""
    val = pack_ctx_code(255, 255, 255, 255)
    assert val == (255 << 24) | (255 << 16) | (255 << 8) | 255


def test_pack_ctx_code_dimensions_independent():
    """T4 维独立：domain=1 其余 0 → 高 byte=1·低 byte 全 0（验位分离·防混淆）。"""
    val = pack_ctx_code(1, 0, 0, 0)
    assert (val >> 24) & 0xFF == 1
    assert (val >> 16) & 0xFF == 0
    assert (val >> 8) & 0xFF == 0
    assert val & 0xFF == 0


# ============ T5-T7 read_effective_freq 桶分离（守通识基线·§点4）============

def test_read_effective_freq_default_zero_bit_identical(backend):
    """T5 ctx=0 退化：base+e_tn 同一 (0,0) 行 = 同行读 bit-identical（一次 select·既有路径零退化）。"""
    record_base_freq(backend, ref=(1, 10), base_freq=7)
    record_experience_outcome(backend, ref=(1, 10), reward=1)
    assert read_effective_freq(backend, (1, 10)) == 8   # base(7) + e_tn(1)·0 桶同行读


def test_read_effective_freq_bucket_split_base_zero_bucket(backend):
    """T6 ctx≠0：base 从 (0,0) 通识桶·e_tn 从当前 ctx 桶（守通识基线·§点4·防混淆频次）。"""
    ctx_b = pack_ctx_code(1, 1, 0, INTENT_QUESTION)
    record_base_freq(backend, ref=(1, 10), base_freq=7)   # 0 桶通识
    record_experience_outcome(backend, ref=(1, 10), reward=1, ctx_code=ctx_b)   # ctx_b 桶经验
    assert read_effective_freq(backend, (1, 10), ctx_code=ctx_b) == 8   # base(7 0桶) + e_tn(1 ctx桶)


def test_read_effective_freq_ctx_no_base_cold_start(backend):
    """T7 ctx≠0 且 (0,0) 无 base（冷启动·该概念无通识先验）→ base=0 + e_tn(ctx)（诚实降级）。"""
    ctx_b = pack_ctx_code(1, 1, 0, INTENT_QUESTION)
    record_experience_outcome(backend, ref=(1, 10), reward=1, ctx_code=ctx_b)   # 只 ctx_b 桶
    assert read_effective_freq(backend, (1, 10), ctx_code=ctx_b) == 1   # base(0) + e_tn(1)


# ============ T8-T9 feed 透传（write 半边·反 theater）============

def test_propagate_reward_feeds_ctx_bucket(core):
    """T8 propagate_reward feed 到 ctx_code 非 0 桶（e_sn/e_tn 增·防混淆分桶主锚）。"""
    b, sid, es, ci = core
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    ctx_code = pack_ctx_code(DOMAIN_MATH, MODALITY_ARITH, 0, INTENT_QUESTION)
    propagate_reward(pr, [], 1, (DOMAIN_MATH, MODALITY_ARITH, 0, INTENT_QUESTION),
                     INTENT_QUESTION, WorkMemory(), edge_store=es, backend=b)
    assert read_experience_count(b, a, ctx_code=ctx_code) == (0, 1, 1)   # ctx 桶 e_sn/e_tn 增
    assert read_experience_count(b, c, ctx_code=ctx_code) == (0, 1, 1)


def test_propagate_reward_zero_bucket_unpolluted(core):
    """T9 防 0 桶污染：feed 到 ctx 桶·0 桶未 feed（read None·防混淆分桶主锚·反 theater）。"""
    b, sid, es, ci = core
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    propagate_reward(pr, [], 1, (DOMAIN_MATH, MODALITY_ARITH, 0, INTENT_QUESTION),
                     INTENT_QUESTION, WorkMemory(), edge_store=es, backend=b)
    assert read_experience_count(b, a) is None   # 0 桶未污染（feed 只写 ctx 桶）
    assert read_experience_count(b, c) is None


# ============ T10-T11 gate① 按桶读 e2e（反 theater 主锚·真行为变）============

def test_word_terminated_per_ctx_bucket(core):
    """T10 反 theater 主锚：同概念 W 两 ctx 桶不同 e_tn → word_terminated 按当前 ctx 桶判。

    W 通识 base_freq=10（0 桶）·ctx_A 桶 e_tn=0·ctx_B 桶 e_tn=100·theta_freq=50：
      ctx_A eff = base(10) + e_tn_A(0) = 10 < 50 → False
      ctx_B eff = base(10) + e_tn_B(100) = 110 ≥ 50 → True
    同概念不同 ctx 不同终止（gate① 按桶读·真行为变·非纸面闭合）。
    """
    b, sid, es, ci = core
    W = ci.ensure("W", space_id=sid)
    ctx_a = pack_ctx_code(1, 1, 0, INTENT_QUESTION)    # domain=1
    ctx_b = pack_ctx_code(2, 1, 0, INTENT_QUESTION)    # domain=2（不同 ctx 桶）
    record_base_freq(b, ref=W, base_freq=10)           # 0 桶通识
    for _ in range(100):
        record_experience_outcome(b, ref=W, reward=1, ctx_code=ctx_b)   # ctx_B 桶累积 100
    wm = WorkMemory()
    intent = IntentType(type=INTENT_QUESTION, sink=(sid, 999))   # sink≠W·sink 保护不触发
    # ctx_A：base(10) + e_tn(0) = 10 < 50 → False（通识基线 + ctx_A 无经验增量）
    assert word_terminated(W, wm, b, intent=intent, theta_freq=50, ctx_code=ctx_a) is False
    # ctx_B：base(10) + e_tn(100) = 110 ≥ 50 → True（ctx_B 经验增量推过阈值）
    assert word_terminated(W, wm, b, intent=intent, theta_freq=50, ctx_code=ctx_b) is True


def test_dag_path_step_ctx_code_per_bucket_e2e(core):
    """T11 反 theater e2e：dag_path_step ctx_code 透传·同图不同 ctx 不同 path。

    A→W→X（CAUSES）·W 通识 base_freq=10（0 桶）·ctx_B 桶 e_tn=100·theta_freq=50：
      ctx_A → W eff=10 < 50 → 不 skip → W→X 进 path·达 sink
      ctx_B → W eff=110 ≥ 50 → W skip → W→X 不进 path·非达 sink
    dag_path_step 按 ctx_code 桶读 effective_freq·真行为变（gate① 按桶读）。
    """
    b, sid, es, ci = core
    A = ci.ensure("A", space_id=sid)
    W = ci.ensure("W", space_id=sid)
    X = ci.ensure("X", space_id=sid)
    _edge(b, es, sid, A[1], W[1], EDGE_CAUSES, sn=1, tn=0)
    _edge(b, es, sid, W[1], X[1], EDGE_CAUSES, sn=1, tn=0)
    record_base_freq(b, ref=W, base_freq=10)
    ctx_a = pack_ctx_code(1, 1, 0, INTENT_COMMAND)
    ctx_b = pack_ctx_code(2, 1, 0, INTENT_COMMAND)
    for _ in range(100):
        record_experience_outcome(b, ref=W, reward=1, ctx_code=ctx_b)
    edges = b.select("edge")
    intent = IntentType(type=INTENT_COMMAND, sink=X)
    # ctx_A：W eff=10 < 50 → 不 skip → W→X 进 path·达 sink
    pr_a = dag_path_step(edges, [A], WorkMemory(), intent, current_seq=0, backend=b,
                         theta_freq=50, ctx_code=ctx_a)
    assert _edge_in(pr_a, sid, W[1], X[1], EDGE_CAUSES) is True
    assert pr_a.terminal == TERMINAL_REACHED_SINK
    # ctx_B：W eff=110 ≥ 50 → W skip → W→X 不进 path·非达 sink
    pr_b = dag_path_step(edges, [A], WorkMemory(), intent, current_seq=0, backend=b,
                         theta_freq=50, ctx_code=ctx_b)
    assert _edge_in(pr_b, sid, W[1], X[1], EDGE_CAUSES) is False
    assert pr_b.terminal != TERMINAL_REACHED_SINK


# ============ T12 bit-identical（dag_path_step 默认 ctx_code=0 退化）============

def test_dag_path_step_default_ctx_code_zero_bit_identical(core):
    """T12 dag_path_step 默认 ctx_code=0 退化（与阶段3 T6c 同语义·退化单 key bit-identical）。"""
    b, sid, es, ci = core
    A = ci.ensure("A", space_id=sid)
    W = ci.ensure("W", space_id=sid)
    X = ci.ensure("X", space_id=sid)
    _edge(b, es, sid, A[1], W[1], EDGE_CAUSES, sn=1, tn=0)
    _edge(b, es, sid, W[1], X[1], EDGE_CAUSES, sn=1, tn=0)
    record_base_freq(b, ref=W, base_freq=1000)   # 0 桶·≥ THETA_FREQ=1000
    edges = b.select("edge")
    intent = IntentType(type=INTENT_COMMAND, sink=X)
    pr = dag_path_step(edges, [A], WorkMemory(), intent, current_seq=0, backend=b)   # 默认 ctx_code=0
    # 退化：base(1000) + e_tn(0) = 1000 ≥ THETA_FREQ → W skip → W→X 不进 path（与阶段3 T6c 同语义）
    assert _edge_in(pr, sid, W[1], X[1], EDGE_CAUSES) is False


# ============ T13 speaker_code 恒 0（defer #495 记忆空间层）============

def test_speaker_code_default_zero_defer(core):
    """T13 speaker_code 恒 0（defer #495 记忆空间层·桶列留值空·全 codebase 无 speaker 来源）。"""
    b, sid, es, ci = core
    a = ci.ensure("apple", space_id=sid)
    c = ci.ensure("cherry", space_id=sid)
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    propagate_reward(pr, [], 1, (DOMAIN_MATH, MODALITY_ARITH, 0, INTENT_QUESTION),
                     INTENT_QUESTION, WorkMemory(), edge_store=es, backend=b)
    ctx_code = pack_ctx_code(DOMAIN_MATH, MODALITY_ARITH, 0, INTENT_QUESTION)
    # feed 后 speaker_code=0 桶（默认·defer #495·propagate_reward 不传 speaker_code）
    assert read_experience_count(b, a, ctx_code=ctx_code, speaker_code=0) == (0, 1, 1)
    # speaker_code≠0 桶无行（未 feed·defer #495 记忆空间层）
    assert read_experience_count(b, a, ctx_code=ctx_code, speaker_code=99) is None
