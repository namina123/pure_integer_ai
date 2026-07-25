"""阶段2 feed + 通识管线测试：experience_count 概念维 feed（2a）+ base_freq 注入（2b）+
reward CAUSES-only 真墙 assertion（2d）+ 三写函数兜底一致（C）+ e2e 反 theater。

覆盖（doc/重来_阶段2feed与通识管线设计补充.md + doc/重来_experience_count落地设计指引.md §点3 feed/§点4 两源）：
  - 2a feed：propagate_reward 落点① 概念维对偶（CAUSES 边端点 + sink + struct_unit_refs·set 去重·R1 符号）
  - 2b 注入：_inject_base_freq（Counter 预扫 + lookup + record_base_freq·weaning_pre 守卫·断奶后退场）
  - 2d assertion：reward feed distributed 须全 CAUSES（防塌柱①·防未来 edit 静默引入非 CAUSES reward）
  - C 一致性：record_experience_outcome 表未注册兜底（镜像 record_base_freq·bit-identical 硬前置）
  - e2e 反 theater：注入 base_freq + feed e_tn → read_effective_freq == base_freq + e_tn（消费者真存在）
  - 铁律：纯整数 / 单向依赖 / 不污染 concept_node 核心 / reward CAUSES-only 真墙
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES, EDGE_CAUSES
from pure_integer_ai.storage.experience_count import (
    register_experience_count,
    record_base_freq, record_experience_outcome,
    read_experience_count, read_effective_freq, pack_ctx_code,
)
from pure_integer_ai.cognition.shared.types import (
    PathData, PathResult, INTENT_QUESTION,
    TERMINAL_REACHED_SINK, MODALITY_LANGUAGE, MODALITY_CODE, MODALITY_ARITH,
    DOMAIN_MATH, WEANING_PRE, WEANING_POST,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.process.reward_propagate import propagate_reward
from pure_integer_ai.experiments.formal_train import make_train_context, _inject_base_freq
from pure_integer_ai.experiments.collection import CollectedItem


# ---- fixtures ----

@pytest.fixture(params=["dict", "sqlite"])
def backend(request):
    """bootstrap + register_experience_count（2a feed / 一致性测试用）。"""
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    register_experience_count(b)
    yield b
    b.close()


@pytest.fixture
def core():
    """建 backend + core 空间 + EdgeStore + ConceptIndex + register experience_count。

    返 (backend, space_id, edge_store, concept_index)——建 concept + 边 + propagate。
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

def _concept(ci, sid, surface):
    return ci.ensure(surface, space_id=sid)


def _edge(b, es, sid, frm, to, et, *, strength=1, sn=0, tn=0, order_index=None):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=et, strength=strength, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY,
           order_index=order_index, sn=sn, tn=tn)


def _path_result(edges, *, sink=None):
    pd = PathData()
    pd.edges = list(edges)
    return PathResult(path=pd, terminal=TERMINAL_REACHED_SINK, sink=sink)


# scalar reward 只在具有独立自证锚的兼容域测试；语言模态在 M-00 后必须零持久写。
_FEED_CTX_TAG = (DOMAIN_MATH, MODALITY_ARITH, 0, INTENT_QUESTION)
_FEED_CTX_CODE = pack_ctx_code(*_FEED_CTX_TAG)


def _prop(pr, reward, es, b):
    propagate_reward(pr, [], reward, _FEED_CTX_TAG, INTENT_QUESTION,
                     WorkMemory(), edge_store=es, backend=b)


# ============ 2a 概念维 feed（落点① 对偶）============

def test_feed_positive_e_sn_e_tn(core):
    """reward>0 → CAUSES 边两端点概念 e_sn=1 e_tn=1（参与即成功·episode 级）。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    _prop(pr, 1, es, b)
    assert read_experience_count(b, a, ctx_code=_FEED_CTX_CODE) == (0, 1, 1)   # ctx 桶 base=0 e_sn=1 e_tn=1
    assert read_experience_count(b, c, ctx_code=_FEED_CTX_CODE) == (0, 1, 1)


def test_feed_set_dedup_same_concept_once(core):
    """同 episode 同 concept 只 feed 一次（set 去重·A→B + A→C·A 重复端点 → A e_tn=1 非 2）。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    bb = _concept(ci, sid, "banana")
    cc = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], bb[1], EDGE_CAUSES, sn=1, tn=0)
    _edge(b, es, sid, a[1], cc[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, bb[1], EDGE_CAUSES),
                       (sid, a[1], sid, cc[1], EDGE_CAUSES)])
    _prop(pr, 1, es, b)
    # A 是两条边的共同 from 端点·set 去重 → 只 feed 一次
    assert read_experience_count(b, a, ctx_code=_FEED_CTX_CODE) == (0, 1, 1)


def test_feed_r1_reward_zero_tn_only(core):
    """reward==0 → e_tn++ only（e_sn 不动·破永正·非"不调"）。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    _prop(pr, 0, es, b)
    assert read_experience_count(b, a, ctx_code=_FEED_CTX_CODE) == (0, 0, 1)   # e_sn=0 e_tn=1


def test_feed_r1_reward_negative_tn_only(core):
    """reward<0 死路 → e_tn++ only（e_sn 不降·率自然降·守单调）。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    _prop(pr, -5, es, b)
    assert read_experience_count(b, a, ctx_code=_FEED_CTX_CODE) == (0, 0, 1)


def test_feed_sink_and_struct_unit_refs_in_target(core):
    """sink + struct_unit_refs 进主集（含未在边端点出现的 concept 也被 feed）。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    sink_concept = _concept(ci, sid, "sink_target")        # 不在边端点
    struct_concept = _concept(ci, sid, "struct_unit")       # 不在边端点
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)],
                      sink=sink_concept)
    pr.path.struct_unit_refs = [struct_concept]
    _prop(pr, 1, es, b)
    assert read_experience_count(b, sink_concept, ctx_code=_FEED_CTX_CODE) == (0, 1, 1)
    assert read_experience_count(b, struct_concept, ctx_code=_FEED_CTX_CODE) == (0, 1, 1)


def test_feed_effective_freq_read(core):
    """feed 后 read_effective_freq == base_freq + e_tn（反 theater·消费者真读）。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    record_base_freq(b, ref=a, base_freq=7)   # 通识先验
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    _prop(pr, 1, es, b)
    # base_freq=7(0桶通识) + e_tn=1(ctx桶) = 8（桶分离·阶段6）
    assert read_effective_freq(b, a, ctx_code=_FEED_CTX_CODE) == 8


def test_feed_table_unregistered_no_crash():
    """表未注册（bootstrap only·不 register）→ propagate 不崩·read_effective_freq 返 0（bit-identical 硬前置）。"""
    b = DictBackend()
    bootstrap(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    sid = sp.space_id
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    _prop(pr, 1, es, b)   # feed 段 try/except 兜底 skip·不崩
    assert read_effective_freq(b, a) == 0
    assert read_experience_count(b, a) is None


def test_feed_empty_causes_edges_no_feed(core):
    """path 全 PRECEDES → causes_edges 空 → 概念维 feed 不执行·不崩。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], EDGE_PRECEDES, order_index=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_PRECEDES)])
    _prop(pr, 1, es, b)   # causes_edges 空·概念维 feed 跳过
    assert read_experience_count(b, a) is None


def test_feed_reward_negative_sink_struct_tn_only(core):
    """reward<0 + sink/struct_unit_refs → 这些 concept 也 feed·e_tn++ only（e_sn 不动·覆盖洞 MED-1）。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    sink_concept = _concept(ci, sid, "sink_target")
    struct_concept = _concept(ci, sid, "struct_unit")
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=sink_concept)
    pr.path.struct_unit_refs = [struct_concept]
    _prop(pr, -3, es, b)
    # reward<0 → 所有 feed 的 concept（边端点 + sink + struct）e_tn++ only·e_sn=0
    assert read_experience_count(b, sink_concept, ctx_code=_FEED_CTX_CODE) == (0, 0, 1)
    assert read_experience_count(b, struct_concept, ctx_code=_FEED_CTX_CODE) == (0, 0, 1)
    assert read_experience_count(b, a, ctx_code=_FEED_CTX_CODE) == (0, 0, 1)


def test_feed_dedup_sink_overlaps_edge_endpoint(core):
    """sink 恰等于某边 to 端点 → set 去重只 feed 一次（e_tn=1 非 2·覆盖洞 MED-2·跨源重叠）。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    # sink = c（恰是边 to 端点）·c 既从边端点进 concept_targets·又从 sink 进·set 去重
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)], sink=c)
    _prop(pr, 1, es, b)
    assert read_experience_count(b, c, ctx_code=_FEED_CTX_CODE) == (0, 1, 1)   # 只 feed 一次（非 e_tn=2）


# ============ 2b 通识 base_freq 注入（_inject_base_freq）============

def test_inject_base_freq_weaning_pre():
    """weaning_pre + corpus token 频次 → concept 行 base_freq==频次（镜像 edge_store base_strength）。"""
    b = DictBackend()
    bootstrap(b)
    ctx = make_train_context(b)
    sid = ctx.space_id
    apple = ctx.concept_index.ensure("apple", space_id=sid)
    pear = ctx.concept_index.ensure("pear", space_id=sid)
    corpus = [CollectedItem(tokens=["apple", "apple", "pear"], modality=MODALITY_LANGUAGE)]
    _inject_base_freq(ctx, corpus)
    assert read_experience_count(b, apple) == (2, 0, 0)   # base_freq=2
    assert read_experience_count(b, pear) == (1, 0, 0)


def test_inject_base_freq_post_no_inject():
    """weaning_post（断奶后）→ 不注入（镜像 EPI_LLM_CONFIRM 退场·断奶后新概念无 base_freq）。"""
    b = DictBackend()
    bootstrap(b)
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST
    ctx.concept_index.ensure("apple", space_id=ctx.space_id)
    corpus = [CollectedItem(tokens=["apple", "apple"], modality=MODALITY_LANGUAGE)]
    _inject_base_freq(ctx, corpus)   # POST 守卫→直接 return
    rows = b.select("experience_count", where=None)
    assert rows == []   # 无注入


def test_inject_base_freq_lookup_miss_skip():
    """lookup miss（token 未 observe 建 concept）→ skip（不建·保 observe 4 入口公共原语契约）。"""
    b = DictBackend()
    bootstrap(b)
    ctx = make_train_context(b)
    ctx.concept_index.ensure("apple", space_id=ctx.space_id)   # 只建 apple
    # corpus 含 pear（未建）+ apple（已建）
    corpus = [CollectedItem(tokens=["apple", "pear", "pear"], modality=MODALITY_LANGUAGE)]
    _inject_base_freq(ctx, corpus)
    apple_ref = ctx.concept_index.lookup("apple", ctx.space_id)
    assert read_experience_count(b, apple_ref) == (1, 0, 0)   # apple 注入
    # pear 未建 concept·lookup miss·不注入·不建 concept
    assert ctx.concept_index.lookup("pear", ctx.space_id) is None


def test_inject_base_freq_idempotent_first_write_wins():
    """调两次 _inject_base_freq → base_freq 不变（first-write-wins·record_base_freq 幂等）。"""
    b = DictBackend()
    bootstrap(b)
    ctx = make_train_context(b)
    ctx.concept_index.ensure("apple", space_id=ctx.space_id)
    corpus = [CollectedItem(tokens=["apple", "apple"], modality=MODALITY_LANGUAGE)]
    _inject_base_freq(ctx, corpus)
    _inject_base_freq(ctx, corpus)   # 二次·幂等 skip
    apple_ref = ctx.concept_index.lookup("apple", ctx.space_id)
    assert read_experience_count(b, apple_ref) == (2, 0, 0)


def test_inject_base_freq_skip_code_arith_modality():
    """MODALITY_CODE / MODALITY_ARITH item 跳过（代码/算术非 token concept·走 COMPOSES 非 token 频次）。"""
    b = DictBackend()
    bootstrap(b)
    ctx = make_train_context(b)
    ctx.concept_index.ensure("foo", space_id=ctx.space_id)
    corpus = [
        CollectedItem(tokens=["foo", "foo"], modality=MODALITY_CODE),
        CollectedItem(tokens=["foo"], modality=MODALITY_ARITH),
    ]
    _inject_base_freq(ctx, corpus)
    rows = b.select("experience_count", where=None)
    assert rows == []   # code/arith 跳过·无注入


def test_inject_base_freq_sorted_bit_identical():
    """两次跑同 corpus → experience_count 表完全一致（sorted 确定序·bit-identical）。"""
    def _run():
        b = DictBackend()
        bootstrap(b)
        ctx = make_train_context(b)
        for tok in ["zebra", "apple", "mango"]:
            ctx.concept_index.ensure(tok, space_id=ctx.space_id)
        corpus = [CollectedItem(tokens=["zebra", "apple", "mango", "apple"],
                                modality=MODALITY_LANGUAGE)]
        _inject_base_freq(ctx, corpus)
        return sorted([(r["space_id"], r["local_id"], r["base_freq"])
                       for r in b.select("experience_count", where=None)])

    assert _run() == _run()


# ============ 2d reward CAUSES-only 真墙 assertion ============

def test_assertion_blocks_non_causes_in_distributed(core, monkeypatch):
    """monkeypatch _distribute_by_rate 返 PRECEDES → assertion fire（防未来 edit 静默引入非 CAUSES reward）。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    cau_ref = (sid, a[1], sid, c[1], EDGE_CAUSES)
    pre_ref = (sid, a[1], sid, c[1], EDGE_PRECEDES)
    pr = _path_result([cau_ref])
    # 模拟未来 edit：_distribute_by_rate 引入非 CAUSES 边进 distributed
    import pure_integer_ai.cognition.process.reward_propagate as rp
    monkeypatch.setattr(rp, "_distribute_by_rate",
                        lambda items, reward: [(pre_ref, 0)])
    with pytest.raises(AssertionError):
        _prop(pr, 1, es, b)


def test_assertion_normal_path_no_fire(core):
    """正常 CAUSES path → assertion 不 fire（回归·distributed 全 CAUSES）。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    _prop(pr, 1, es, b)   # 不抛
    assert read_experience_count(b, a, ctx_code=_FEED_CTX_CODE) == (0, 1, 1)


# ============ C 一致性（三写函数兜底）============

def test_record_experience_outcome_table_unregistered_no_crash():
    """record_experience_outcome 表未注册 → 不崩（兜底·镜像 record_base_freq·bit-identical 硬前置）。"""
    b = DictBackend()
    bootstrap(b)   # 不 register experience_count
    record_experience_outcome(b, ref=(1, 10), reward=1)   # 不崩·静默 skip
    record_experience_outcome(b, ref=(1, 10), reward=-1)  # reward≤0 也不崩
    assert read_experience_count(b, (1, 10)) is None


def test_three_writes_unregistered_consistent():
    """三写函数（record_base_freq/read_experience_count/record_experience_outcome）表未注册都 None/skip 一致。"""
    b = DictBackend()
    bootstrap(b)
    # 三写函数都不崩
    record_base_freq(b, ref=(1, 10), base_freq=5)
    record_experience_outcome(b, ref=(1, 10), reward=1)
    assert read_experience_count(b, (1, 10)) is None
    assert read_effective_freq(b, (1, 10)) == 0


# ============ e2e 反 theater 主锚 ============

def test_e2e_inject_feed_effective_freq_consumer():
    """e2e：注入 base_freq + reward feed e_tn → read_effective_freq == base_freq + e_tn（消费者真存在·反 theater）。"""
    b = DictBackend()
    bootstrap(b)
    ctx = make_train_context(b)
    sid = ctx.space_id
    apple = ctx.concept_index.ensure("apple", space_id=sid)
    banana = ctx.concept_index.ensure("banana", space_id=sid)
    # 2b 通识注入
    corpus = [CollectedItem(tokens=["apple", "apple", "banana"], modality=MODALITY_LANGUAGE)]
    _inject_base_freq(ctx, corpus)
    assert read_effective_freq(b, apple) == 2    # 冷启动 effective_freq=base_freq（0 桶·ctx 桶无 e_tn）
    assert read_effective_freq(b, banana) == 1
    # 建 CAUSES 边 apple→banana + 2a feed（_FEED_CTX_CODE 非 0 桶）
    _edge(b, ctx.edge_store, sid, apple[1], banana[1], EDGE_CAUSES, sn=1, tn=0)
    pr = _path_result([(sid, apple[1], sid, banana[1], EDGE_CAUSES)])
    _prop(pr, 1, ctx.edge_store, b)
    # 阶段6 桶分离：base_freq 在 0 桶（通识注入·context-agnostic）·e_sn/e_tn 在 _FEED_CTX_CODE 桶（feed）
    assert read_experience_count(b, apple) == (2, 0, 0)              # 0 桶：base_freq=2·e_sn/e_tn=0
    assert read_experience_count(b, apple, ctx_code=_FEED_CTX_CODE) == (0, 1, 1)   # ctx 桶：feed e_sn/e_tn
    # read_effective_freq 桶分离：base(0桶 2) + e_tn(ctx桶 1) = 3（守通识基线·反 theater·阶段6）
    assert read_effective_freq(b, apple, ctx_code=_FEED_CTX_CODE) == 3
    assert read_effective_freq(b, banana, ctx_code=_FEED_CTX_CODE) == 2   # base(0桶 1) + e_tn(ctx桶 1)


# ============ 回归：propagate_reward 既有行为不破 ============

def test_regression_edge_level_sn_tn_strength_unchanged(core):
    """回归：边级落点① sn/tn/strength 行为不变（概念维 feed 是加段·不改边级）。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], EDGE_CAUSES, strength=1, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    _prop(pr, 1, es, b)
    edge_row = b.select("edge", where={"edge_type": EDGE_CAUSES})[0]
    # 边级 sn++&tn++ + strength+=Δ（既有行为·test_stage4 范式）
    assert edge_row["sn"] == 2 and edge_row["tn"] == 1
    assert edge_row["strength"] == 2   # 1 + DELTA_DEFAULT(1)
