"""止血 #1146 测试（methodology doc §五·reward 非 frame·CAUSES edge reward 写按域过滤）。

语言域 reward 结构性 theater（judge G5 vacated·无 correctness 锚）→ episode 全 dead-end → reward≤0 →
reward_propagate 落点① CAUSES tn++ 在惩罚唯一 reward-active 边（有害）。止血：reward-illegitimate 域
（DOMAIN_TEXT / DOMAIN_BARE·判据 = 不在 shared.REWARD_LEGITIMATE_DOMAINS·与 judge G5 激活域同源）剔出
落点① edge 写（sn/tn/strength）。gate CAUSES_REWARD_DOMAIN_FILTER_MODE（default OFF 守 CI bit-identical）。

覆盖（镜像 test_experience_count_feed core fixture + helpers）：
  H1 gate OFF bit-identical：语言 reward<0 → CAUSES tn++（既有行为·逐字现状·gate OFF 条件恒 False）
  H2 止血语言死路：gate ON 语言 reward<0 → CAUSES tn 不动（停惩罚·核心命门）
  H3 止血语言正例：gate ON 语言 reward>0 → CAUSES sn/tn/strength 全不动（reward 从语言完全退场·edge 写）
  H4 概念维对偶保留：gate ON 语言 reward<0 → experience_count e_tn++ 仍执行（概念维对偶非 edge reward 多头）
  H5 算术死路不滤：gate ON 算术(DOMAIN_MATH) reward<0 → CAUSES tn++（reward-legitimate·学习 intact）
  H6 算术正例 intact：gate ON 算术 reward>0 → CAUSES sn/tn/strength 涨（reward-legitimate 正常学习）
  H7 bare 也滤：gate ON DOMAIN_BARE reward<0 → CAUSES tn 不动（reward-illegitimate 非仅语言）
  H8 gate 默认 OFF：module-level gate is False（锁 CI bit-identical 硬前置）

铁律：纯整数 / bit-identical（gate OFF 逐字现状）/ 不写死（域判据 = shared frozenset·非字面 if domain==1）/
  反 theater（止血停有害 tn++·非"reward 流通"修死胡同 bug·reward 非 frame）/ 概念维对偶保留（experience_count）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_CAUSES
from pure_integer_ai.storage.experience_count import (
    register_experience_count, record_experience_outcome,
    read_experience_count, pack_ctx_code,
)
from pure_integer_ai.cognition.shared.types import (
    PathData, PathResult, INTENT_QUESTION,
    TERMINAL_REACHED_SINK, DOMAIN_TEXT, DOMAIN_MATH, DOMAIN_BARE,
    MODALITY_LANGUAGE, MODALITY_ARITH,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.process.reward_propagate import propagate_reward
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def core():
    """建 backend + core 空间 + EdgeStore + ConceptIndex + register experience_count。

    返 (backend, space_id, edge_store, concept_index)——建 concept + CAUSES 边 + propagate。
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


def _edge(b, es, sid, frm, to, *, strength=1, sn=0, tn=0):
    es.add(space_id_from=sid, local_id_from=frm, space_id_to=sid, local_id_to=to,
           edge_type=EDGE_CAUSES, strength=strength, source=SOURCE_BARE_TEXT,
           tier=TIER_PRIMARY, sn=sn, tn=tn)


def _path_result(edges, *, sink=None):
    pd = PathData()
    pd.edges = list(edges)
    return PathResult(path=pd, terminal=TERMINAL_REACHED_SINK, sink=sink)


def _edge_row(b):
    return b.select("edge", where={"edge_type": EDGE_CAUSES})[0]


def _prop(pr, reward, ctx_tag, es, b):
    propagate_reward(pr, [], reward, ctx_tag, INTENT_QUESTION,
                     WorkMemory(), edge_store=es, backend=b)


# ctx_tag (domain, modality, task, intent_type)·镜像 episode._ctx_tag 真实结构。
_LANG_CTX = (DOMAIN_TEXT, MODALITY_LANGUAGE, 0, INTENT_QUESTION)
_ARITH_CTX = (DOMAIN_MATH, MODALITY_ARITH, 0, INTENT_QUESTION)
_BARE_CTX = (DOMAIN_BARE, MODALITY_LANGUAGE, 0, INTENT_QUESTION)
_LANG_CTX_CODE = pack_ctx_code(*_LANG_CTX)


# ============ H1 gate OFF bit-identical（既有行为·逐字现状）============

def test_h1_language_deadend_never_updates_edge(core):
    """M-00：旧域过滤 gate 关闭时，语言 scalar reward 仍不能更新长期边。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    _prop(pr, -5, _LANG_CTX, es, b)
    row = _edge_row(b)
    assert row["tn"] == 0 and row["sn"] == 1


# ============ H2 止血语言死路（核心命门·停 tn++ 惩罚）============

def test_h2_gate_on_language_deadend_no_tn(core):
    """★gate ON：语言 reward<0 → CAUSES tn 不动（止血·停惩罚唯一 reward-active 边·methodology §二铁事实3）。

    gate ON + DOMAIN_TEXT 不在 REWARD_LEGITIMATE_DOMAINS → _skip=True → 落点① edge 写跳过 → tn 留 0。
    """
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    saved = gates.CAUSES_REWARD_DOMAIN_FILTER_MODE
    gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = True
    try:
        _prop(pr, -5, _LANG_CTX, es, b)
    finally:
        gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = saved
    row = _edge_row(b)
    assert row["tn"] == 0 and row["sn"] == 1 and row["strength"] == 1, \
        f"止血：语言 dead-end 不再 tn++（停惩罚）·got sn={row['sn']} tn={row['tn']} strength={row['strength']}"


# ============ H3 止血语言正例（reward 从语言完全退场·edge 写全跳）============

def test_h3_gate_on_language_positive_no_edge_write(core):
    """gate ON：语言 reward>0 → CAUSES sn/tn/strength 全不动（reward 从语言 edge 写完全退场·§五降级）。

    非仅死路——语言域 reward>0 也不写 edge（reward 非 frame·CAUSES 走刀 constructive-check 不接 strength）。
    """
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], strength=1, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    saved = gates.CAUSES_REWARD_DOMAIN_FILTER_MODE
    gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = True
    try:
        _prop(pr, 1, _LANG_CTX, es, b)
    finally:
        gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = saved
    row = _edge_row(b)
    assert row["sn"] == 1 and row["tn"] == 0 and row["strength"] == 1, \
        f"止血：语言正例 reward>0 也不写 edge（reward 完全退场）·got sn={row['sn']} tn={row['tn']} strength={row['strength']}"


# ============ H4 概念维对偶保留（experience_count 非 edge reward 多头）============

def test_h4_language_scalar_does_not_update_concept_aggregate(core):
    """M-00：语言 scalar reward 不得通过概念聚合旁路巩固长期对象。"""
    b, sid, es, ci = core
    a = _concept(ci, sid, "apple")
    c = _concept(ci, sid, "cherry")
    _edge(b, es, sid, a[1], c[1], sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    saved = gates.CAUSES_REWARD_DOMAIN_FILTER_MODE
    gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = True
    try:
        _prop(pr, -3, _LANG_CTX, es, b)
    finally:
        gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = saved
    assert read_experience_count(b, a, ctx_code=_LANG_CTX_CODE) is None
    assert read_experience_count(b, c, ctx_code=_LANG_CTX_CODE) is None


# ============ H5 算术死路不滤（reward-legitimate·学习 intact）============

def test_h5_gate_on_arith_deadend_tn_increments(core):
    """gate ON：算术(DOMAIN_MATH) reward<0 → CAUSES tn++（reward-legitimate·死路惩罚是正当学习信号）。

    算术 dead-end tn++ = 正当（错误路径应惩罚 CAUSES·与 correctness 对齐经 vm_proof）·止血不滤 reward-legitimate 域。
    """
    b, sid, es, ci = core
    a = _concept(ci, sid, "three")
    c = _concept(ci, sid, "five")
    _edge(b, es, sid, a[1], c[1], sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    saved = gates.CAUSES_REWARD_DOMAIN_FILTER_MODE
    gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = True
    try:
        _prop(pr, -5, _ARITH_CTX, es, b)
    finally:
        gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = saved
    row = _edge_row(b)
    assert row["tn"] == 1, \
        f"算术 reward-legitimate·dead-end tn++ 正当学习·不滤·got tn={row['tn']}"


# ============ H6 算术正例 intact（reward-legitimate 正常学习）============

def test_h6_gate_on_arith_positive_edge_write(core):
    """gate ON：算术 reward>0 → CAUSES sn++&tn++ + strength+=Δ（reward-legitimate 正常学习 intact）。

    止血只影响 reward-illegitimate 域（语言/bare）·算术 reward 学习链完全不变（反 theater·非"全域停 reward"）。
    """
    b, sid, es, ci = core
    a = _concept(ci, sid, "three")
    c = _concept(ci, sid, "five")
    _edge(b, es, sid, a[1], c[1], strength=1, sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    saved = gates.CAUSES_REWARD_DOMAIN_FILTER_MODE
    gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = True
    try:
        _prop(pr, 1, _ARITH_CTX, es, b)
    finally:
        gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = saved
    row = _edge_row(b)
    assert row["sn"] == 2 and row["tn"] == 1 and row["strength"] == 2, \
        f"算术 reward>0 正常学习 intact（sn++&tn++ + strength+=Δ）·got sn={row['sn']} tn={row['tn']} strength={row['strength']}"


# ============ H7 bare 也滤（reward-illegitimate 非仅语言）============

def test_h7_gate_on_bare_deadend_no_tn(core):
    """gate ON：DOMAIN_BARE reward<0 → CAUSES tn 不动（reward-illegitimate 非仅语言·bare 同滤）。

    判据 = 不在 REWARD_LEGITIMATE_DOMAINS（{CODE,MATH}）·DOMAIN_BARE=4 不在 → 同滤（不写死"仅 DOMAIN_TEXT"）。
    """
    b, sid, es, ci = core
    a = _concept(ci, sid, "alpha")
    c = _concept(ci, sid, "beta")
    _edge(b, es, sid, a[1], c[1], sn=1, tn=0)
    pr = _path_result([(sid, a[1], sid, c[1], EDGE_CAUSES)])
    saved = gates.CAUSES_REWARD_DOMAIN_FILTER_MODE
    gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = True
    try:
        _prop(pr, -5, _BARE_CTX, es, b)
    finally:
        gates.CAUSES_REWARD_DOMAIN_FILTER_MODE = saved
    row = _edge_row(b)
    assert row["tn"] == 0, \
        f"DOMAIN_BARE reward-illegitimate 同滤（判据=不在合法域集·非写死语言）·got tn={row['tn']}"


# ============ H8 gate 默认 OFF（锁 CI bit-identical 硬前置）============

def test_h8_gate_default_off():
    """module-level gate 默认 False（_flag("PURE_INTEGER_AI_CAUSES_REWARD_DOMAIN_FILTER_MODE", False)·CI 无 env）。

    兼验本文件 H2-H7 的 try/finally 真复位（若任一漏 restore·此断言失败锁穿漏）·gate OFF = bit-identical 硬前置。
    """
    assert gates.CAUSES_REWARD_DOMAIN_FILTER_MODE is False, \
        "gate 默认 OFF（CI bit-identical·生产 formal_train try/finally 翻 ON·测内 try/finally 复位）"
