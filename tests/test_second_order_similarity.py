"""Phase C 二阶相似读侧测试（§十五-bis·second_order_similarity + _second_order_bonus）。

C.1 second_order_similarity（read-side Jaccard·hub 滤·scaled-int·对称 cache）。
C.2 _second_order_bonus（per-candidate max over ctx_refs·mirror _pronoun_bonus·cap）。
反 theater：read-side 不存边·确定性 computation 非"学得"·同 collide_score 一阶 exposure signal。

铁律：纯整数（scaled-int //·禁 float）/ bit-identical（gate OFF·_second_order_bonus 由 caller inline 门控）/ 对称（Jaccard(a,b)==Jaccard(b,a)）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.config import gates
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_types import EDGE_COOCCURS
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_SHADOW
from pure_integer_ai.cognition.result.graph_view import ConceptGraph, _SIM_SCALE
from pure_integer_ai.cognition.result.slot_dispatch import _second_order_bonus
from tests.test_experiments import make_train_context


@pytest.fixture(autouse=True)
def _gate_reset():
    """每测前后复位 SIMILAR_SECOND_ORDER_MODE（守测试隔离）。"""
    saved = gates.SIMILAR_SECOND_ORDER_MODE
    gates.SIMILAR_SECOND_ORDER_MODE = False
    yield
    gates.SIMILAR_SECOND_ORDER_MODE = saved


def _cooccur(ctx, a, b):
    """加一条 COOCCURS 边 a→b（strength=1·_cooccur_neighbors 双向查故单向写够）。"""
    ctx.edge_store.add(
        space_id_from=a[0], local_id_from=a[1],
        space_id_to=b[0], local_id_to=b[1],
        edge_type=EDGE_COOCCURS, strength=1, source=SOURCE_BARE_TEXT,
        tier=TIER_SHADOW,
    )


# ============ C.1 second_order_similarity（read-side Jaccard） ============

def test_c1_jaccard_basic():
    """基本 Jaccard：a={x,y} b={y,z}·∩={y} ∪={x,y,z}·score=1000*1//3=333。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    a = ctx.concept_index.ensure("a", space_id=sid)
    b = ctx.concept_index.ensure("b", space_id=sid)
    x = ctx.concept_index.ensure("x", space_id=sid)
    y = ctx.concept_index.ensure("y", space_id=sid)
    z = ctx.concept_index.ensure("z", space_id=sid)
    _cooccur(ctx, a, x); _cooccur(ctx, a, y); _cooccur(ctx, b, y); _cooccur(ctx, b, z)
    assert ctx.concept_graph.second_order_similarity(a, b) == _SIM_SCALE * 1 // 3, \
        "Jaccard ∩{y}/∪{x,y,z}=333"


def test_c1_empty_neighbors_zero():
    """空邻集→0（冷启动·守确定性·无证据非 theater·无 ZeroDivision）。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    a = ctx.concept_index.ensure("a", space_id=sid)
    b = ctx.concept_index.ensure("b", space_id=sid)
    assert ctx.concept_graph.second_order_similarity(a, b) == 0, "空并集→0"


def test_c1_identical_neighbors_full():
    """同邻集→_SIM_SCALE（同外延最相似·Jaccard=1）。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    a = ctx.concept_index.ensure("a", space_id=sid)
    b = ctx.concept_index.ensure("b", space_id=sid)
    x = ctx.concept_index.ensure("x", space_id=sid)
    _cooccur(ctx, a, x); _cooccur(ctx, b, x)
    assert ctx.concept_graph.second_order_similarity(a, b) == _SIM_SCALE, "∩{x}/∪{x}=1000"


def test_c1_symmetric():
    """对称：Jaccard(a,b)==Jaccard(b,a)（对称 cache key·Phase C §十五-bis）。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    a = ctx.concept_index.ensure("a", space_id=sid)
    b = ctx.concept_index.ensure("b", space_id=sid)
    x = ctx.concept_index.ensure("x", space_id=sid)
    y = ctx.concept_index.ensure("y", space_id=sid)
    z = ctx.concept_index.ensure("z", space_id=sid)
    _cooccur(ctx, a, x); _cooccur(ctx, a, y); _cooccur(ctx, b, y); _cooccur(ctx, b, z)
    g = ctx.concept_graph
    assert g.second_order_similarity(a, b) == g.second_order_similarity(b, a), "对称（cache key sorted）"


def test_c1_hub_filtered():
    """hub 滤：hub（degree≥θ=8）从邻集排除·否则 co-occur-with-all 主宰交集塌相似度。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    a = ctx.concept_index.ensure("a", space_id=sid)
    b = ctx.concept_index.ensure("b", space_id=sid)
    hub = ctx.concept_index.ensure("hub", space_id=sid)
    x = ctx.concept_index.ensure("x", space_id=sid)
    _cooccur(ctx, a, hub); _cooccur(ctx, b, hub); _cooccur(ctx, a, x)
    for i in range(8):   # hub co-occur 8+ → degree≥8 → hub
        hn = ctx.concept_index.ensure(f"hn{i}", space_id=sid)
        _cooccur(ctx, hub, hn)
    g = ctx.concept_graph
    assert hub in g.hub_set(), "hub 达 degree≥8（θ=8）"
    # 滤 hub：a 邻={hub,x}→滤{x}·b 邻={hub}→滤{}·∩={}→0（hub 不主宰·否则 ∩{hub}/∪{hub,x}=500）
    assert g.second_order_similarity(a, b) == 0, "hub 滤后 b 无非-hub 邻→0"


# ============ C.2 _second_order_bonus（per-candidate·mirror _pronoun_bonus） ============

def test_c2_bonus_per_candidate_max():
    """_second_order_bonus per-candidate max over ctx_refs（c-r1 sim=1000 > c-r2 sim=0·max=1000）。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    c = ctx.concept_index.ensure("c", space_id=sid)
    r1 = ctx.concept_index.ensure("r1", space_id=sid)
    r2 = ctx.concept_index.ensure("r2", space_id=sid)
    x = ctx.concept_index.ensure("x", space_id=sid)
    _cooccur(ctx, c, x); _cooccur(ctx, r1, x)   # c,r1 同邻{x}·sim=1000·r2 无邻·sim=0
    g = ctx.concept_graph
    assert _second_order_bonus(g, c, [r1, r2]) == _SIM_SCALE, "max(c,r1=1000, c,r2=0)=1000"


def test_c2_bonus_skip_self():
    """_second_order_bonus skip c==r（避自 boost·镜像 _pronoun_bonus / selection_pref_score）。"""
    ctx = make_train_context(DictBackend())
    sid = ctx.space_id
    c = ctx.concept_index.ensure("c", space_id=sid)
    x = ctx.concept_index.ensure("x", space_id=sid)
    _cooccur(ctx, c, x)
    g = ctx.concept_graph
    assert _second_order_bonus(g, c, [c]) == 0, "skip c==r·避自 boost（c 自身不进 max）"
