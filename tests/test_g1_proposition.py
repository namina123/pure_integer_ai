"""G1 reification + #774 PROPERTY 测试（命题节点承载 subject/attr_type/value 三元·G3b 真消费者）。

设计 doc/重来_G1reification_774PROPERTY_设计_2026-07-09.md（fork 分析 §5.3 实施 ready·选 b 避 generate 改动）。

**G1+#774 验收判据**：
  - cue 提取（的...是 / 具有）+ cue_type_of 不污染（是/仍返 None·守 extract_cues bit-identical）。
  - 命题节点身份去重（同(subject,attr_type) 聚同节点）+ PROPERTY value 出边 + ATTR_PROPOSITION 标记。
  - G3b 全局扫命题节点：同(subject,attr_type)多值=CONTRADICTED(0)·异 subject/异 attr_type=无假矛盾(1)。
  - 反 theater（has_value_claim 真活激活 G3b + 命题节点真有 PROPERTY 出边·非空集永返 1）。
  - bit-identical（gate OFF = 无命题节点/PROPERTY/has_value_claim·既有行为零变）。
  - attr_type 缺省（具有模式）skip（首版 defer）。

铁律：纯整数 / 确定性 bit-identical / 不写死（cue 词表元定义·builder 只机制）/ 不纸面闭合（G3b 真消费者）。
诚实边界：reification 给表达力非验证力（命题 truth=#479 墙·G3b 只判结构矛盾层a·语义真对立层b/c D 墙 defer）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_types import EDGE_PROPERTY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.composes_attr import (
    register_composes_attr, read_composes_attrs, ATTR_PROPOSITION,
)
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.understanding.cue_words import (
    cue_type_of, is_property_attr_marker, is_property_value_copula, is_property_possess_cue,
    is_negation_cue,
)
from pure_integer_ai.cognition.understanding.cue_extractor import (
    extract_property_claims, extract_property_claims_gated,
)
from pure_integer_ai.cognition.understanding.property import build_property_edges
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.cognition.result.judge import counterfactual_value_check
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def prop_env():
    """build_property_edges 单测环境（dict backend·core space·composes_attr 已注册）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)   # 命题节点 ATTR_PROPOSITION 标记表（make_train_context 范式）
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    yield b, sid, es, ci
    b.close()


@pytest.fixture
def proposition_on():
    """翻 PROPOSITION_MODE ON·测后复位（镜像生产 formal_train try/finally）。"""
    saved = gates.PROPOSITION_MODE
    gates.PROPOSITION_MODE = True
    yield
    gates.PROPOSITION_MODE = saved


def _ensure(ci, sid, surface):
    return ci.ensure(surface, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)


# ============ unit：cue helpers（独立于 cue_type_of·防 是/的 污染 extract_cues） ============

def test_is_property_attr_marker():
    """属性标记 的 识别（ZH·exact·EN defer）。"""
    assert is_property_attr_marker("的", LANG_ZH) is True
    assert is_property_attr_marker("是", LANG_ZH) is False   # 是 是值系词非属性标记
    assert is_property_attr_marker("猫", LANG_ZH) is False
    assert is_property_attr_marker("的", LANG_EN) is False   # EN 's defer


def test_is_property_value_copula():
    """值系词 是 识别（ZH·exact）。"""
    assert is_property_value_copula("是", LANG_ZH) is True
    assert is_property_value_copula("的", LANG_ZH) is False
    assert is_property_value_copula("是一种", LANG_ZH) is False   # IS_A 多字 token 不混


def test_is_property_possess_cue():
    """领属 cue 具有/有/has 识别（ZH+EN）。"""
    assert is_property_possess_cue("具有", LANG_ZH) is True
    assert is_property_possess_cue("有", LANG_ZH) is True
    assert is_property_possess_cue("has", LANG_EN) is True
    assert is_property_possess_cue("have", LANG_EN) is True
    assert is_property_possess_cue("是", LANG_ZH) is False


def test_cue_type_of_not_polluted_by_property_tokens():
    """★bit-identical 硬守：是/的 不入 cue_type_of（防污染 extract_cues 邻居判·改变 CAUSES/IS_A 提取）。

    是/的 太通用·若 cue_type_of 返非 None → extract_cues 把 是/当 cue 跳过配对 → 非 bit-identical。
    属性命题检测走独立 helpers（is_property_*·非 cue_type_of）·cue_type_of 对 是/的 仍返 None。
    """
    assert cue_type_of("是", LANG_ZH) is None, "是 不可入 cue_type_of（防 extract_cues 邻居判污染）"
    assert cue_type_of("的", LANG_ZH) is None, "的 不可入 cue_type_of（同上）"
    assert cue_type_of("具有", LANG_ZH) is None, "具有 不可入 cue_type_of（独立 helper）"


# ============ unit：extract_property_claims（的...是 / 具有 固定窗口） ============

def test_extract_property_claims_de_pattern():
    """的...是 正：猫 的 颜色 是 黑 → (0, 2, 4, 0)（subject=猫·attr_type=颜色·value=黑）。"""
    tokens = ["猫", "的", "颜色", "是", "黑"]
    claims = extract_property_claims(tokens, lang=LANG_ZH)
    assert claims == [(0, 2, 4, 0, 0, 0, 1, 1)], "的...是 窗口：subject=的-3·attr_type=是-1·value=是+1·P0.3 pol/mod=0 default"


def test_extract_property_claims_possess_pattern():
    """领属 正：猫 具有 黑 → (0, -1, 2, 0)（attr_type=-1 缺省·首版 defer·build skip）。"""
    tokens = ["猫", "具有", "黑"]
    claims = extract_property_claims(tokens, lang=LANG_ZH)
    assert claims == [(0, -1, 2, 0, 0, 0, 1, 1)], "领属模式 attr_type=-1·P0.3 pol/mod=0 default"


def test_extract_property_claims_boundary_short():
    """边界：是 在句首无 subject/attr → 跳（守反统计·不凑配）。"""
    assert extract_property_claims(["是", "黑"], lang=LANG_ZH) == [], "是 at j=0·j-3<0·跳"
    assert extract_property_claims(["猫", "的", "是"], lang=LANG_ZH) == [], "是 at j=2·value j+1 越界·跳"


def test_extract_property_claims_no_attr_marker():
    """非 的...是：X 是 Y（无 的 attr marker）→ 跳（是 可能是 IS_A/其他用法·守 的...是 固定窗口）。"""
    assert extract_property_claims(["猫", "是", "黑"], lang=LANG_ZH) == [], "无 的 at j-2·非属性窗口·跳"


def test_extract_property_claims_cue_token_rejected():
    """subject/attr/value 自身是 cue token → 跳（守反统计·不配 cue token·同 extract_cues 邻居判）。"""
    # X=的(subject 是 cue)·的 的 颜色 是 黑 → subject=的 是 cue·跳
    assert extract_property_claims(["的", "的", "颜色", "是", "黑"], lang=LANG_ZH) == []


def test_extract_property_claims_gated_off_returns_empty():
    """gate OFF → 返 []（bit-identical 守回归·PROPOSITION_MODE 非 CUE_EXTRACTOR_MODE）。"""
    saved = gates.PROPOSITION_MODE
    gates.PROPOSITION_MODE = False
    try:
        assert extract_property_claims_gated(["猫", "的", "颜色", "是", "黑"], lang=LANG_ZH) == []
    finally:
        gates.PROPOSITION_MODE = saved


def test_extract_property_claims_gated_on(proposition_on):
    """gate ON → 真提取（的...是 模式）。"""
    assert extract_property_claims_gated(["猫", "的", "颜色", "是", "黑"], lang=LANG_ZH) == [(0, 2, 4, 0, 0, 0, 1, 1)]


# ============ unit：build_property_edges（命题节点 + PROPERTY value 出边 + ATTR_PROPOSITION 标记） ============

def test_build_property_edges_builds_prop_node_and_edge(prop_env):
    """的...是 claim → 命题节点（ATTR_PROPOSITION）+ PROPERTY value 出边。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 黑 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑")
    refs = [猫, 颜色, 黑]
    n = build_property_edges(es, ci, b, refs,
                             property_claims=[(0, 1, 2, 0)],
                             source=SOURCE_BARE_TEXT, space_id=sid)
    assert n == 1, "建 1 条 PROPERTY value 出边"
    # 命题节点 = (猫, 颜色)·确定性 surface
    prop_ref = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}", sid)
    assert prop_ref is not None, "命题节点须 ensure 建"
    # ATTR_PROPOSITION=11 标记
    attrs = read_composes_attrs(b, prop_ref)
    assert ATTR_PROPOSITION in attrs, "命题节点须 ATTR_PROPOSITION 标记"
    # PROPERTY 出边 命题节点→黑
    prop_edges = [r for r in b.select("edge", where={
        "edge_type": EDGE_PROPERTY, "space_id_from": prop_ref[0], "local_id_from": prop_ref[1]})]
    assert len(prop_edges) == 1, "1 条 PROPERTY value 出边"
    assert (prop_edges[0]["space_id_to"], prop_edges[0]["local_id_to"]) == 黑, "value=黑"


def test_build_property_edges_dedup_same_subject_attr(prop_env):
    """同(subject,attr_type) 两不同 value → 同命题节点·2 PROPERTY 边（G3b 判矛盾的基础）。"""
    b, sid, es, ci = prop_env
    猫 = _ensure(ci, sid, "猫")
    颜色 = _ensure(ci, sid, "颜色")
    黑, 白 = _ensure(ci, sid, "黑"), _ensure(ci, sid, "白")
    refs = [猫, 颜色, 黑, 白]
    n = build_property_edges(es, ci, b, refs,
                             property_claims=[(0, 1, 2, 0), (0, 1, 3, 0)],   # 猫的颜色是黑 + 猫的颜色是白
                             source=SOURCE_BARE_TEXT, space_id=sid)
    assert n == 2, "同命题节点 2 不同 value → 2 PROPERTY 边"
    # 同一命题节点（身份去重）
    prop_ref = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}", sid)
    prop_edges = [r for r in b.select("edge", where={
        "edge_type": EDGE_PROPERTY, "space_id_from": prop_ref[0], "local_id_from": prop_ref[1]})]
    assert len(prop_edges) == 2, "同命题节点 2 PROPERTY value 出边"
    targets = {(r["space_id_to"], r["local_id_to"]) for r in prop_edges}
    assert targets == {黑, 白}, "两 value 出边"


def test_build_property_edges_idempotent_same_value(prop_env):
    """幂等：同命题节点同 value 重 claim → skip（query_from·observe 跨段/多轮重 observe 不 corrupt）。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 黑 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑")
    refs = [猫, 颜色, 黑]
    n1 = build_property_edges(es, ci, b, refs, property_claims=[(0, 1, 2, 0)],
                              source=SOURCE_BARE_TEXT, space_id=sid)
    n2 = build_property_edges(es, ci, b, refs, property_claims=[(0, 1, 2, 0)],
                              source=SOURCE_BARE_TEXT, space_id=sid)   # 同 claim 重种
    assert n1 == 1 and n2 == 0, "同命题节点同 value 第二次 → skip（幂等）"


def test_build_property_edges_attr_type_missing_skipped(prop_env):
    """attr_type<0（领属 具有 模式）→ skip（首版 defer·无命题身份·设计 §六）。"""
    b, sid, es, ci = prop_env
    猫, 黑 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "黑")
    refs = [猫, 黑]
    n = build_property_edges(es, ci, b, refs,
                             property_claims=[(0, -1, 1, 0)],   # 猫 具有 黑·attr_type=-1
                             source=SOURCE_BARE_TEXT, space_id=sid)
    assert n == 0, "领属模式 attr_type<0 → skip（首版 defer·不建命题节点）"
    assert b.select("edge", where={"edge_type": EDGE_PROPERTY}) == [], "领属 skip → 零 PROPERTY 边"


def test_build_property_edges_different_attr_distinct_nodes(prop_env):
    """异 attr_type → 异命题节点（无假矛盾·fork §3.3）：猫的颜色是黑 + 猫的数量是四 → 2 节点各 1 边。"""
    b, sid, es, ci = prop_env
    猫 = _ensure(ci, sid, "猫")
    颜色, 数量 = _ensure(ci, sid, "颜色"), _ensure(ci, sid, "数量")
    黑, 四 = _ensure(ci, sid, "黑"), _ensure(ci, sid, "四")
    refs = [猫, 颜色, 黑, 数量, 四]
    n = build_property_edges(es, ci, b, refs,
                             property_claims=[(0, 1, 2, 0), (0, 3, 4, 0)],   # 颜色是黑 + 数量是四
                             source=SOURCE_BARE_TEXT, space_id=sid)
    assert n == 2
    prop_color = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}", sid)
    prop_count = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{数量[0]}_{数量[1]}", sid)
    assert prop_color != prop_count, "异 attr_type → 异命题节点"


# ============ unit：iter_proposition_nodes（graph_view·G3b 全局扫用） ============

def test_iter_proposition_nodes_reads_attr(prop_env):
    """build 后 iter_proposition_nodes 返命题节点 ref 列表（确定性序）。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 黑 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑")
    build_property_edges(es, ci, b, [猫, 颜色, 黑],
                         property_claims=[(0, 1, 2, 0)],
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    props = graph.iter_proposition_nodes()
    assert len(props) == 1, "1 命题节点"
    assert props[0] == ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}", sid)


def test_iter_proposition_nodes_empty_when_no_prop_nodes(prop_env):
    """无命题节点 → 返 []（gate OFF 生产路径·G3b 扫空返 1）。"""
    b, sid, es, ci = prop_env
    graph = ConceptGraph(b)
    assert graph.iter_proposition_nodes() == []


def test_iter_proposition_nodes_keyerror_bare_backend():
    """表未注册（bare fixture·未 register_composes_attr）→ KeyError try/except → []（向后兼容）。"""
    b = DictBackend()
    bootstrap(b)   # 不调 register_composes_attr
    graph = ConceptGraph(b)
    assert graph.iter_proposition_nodes() == [], "表未注册 → []（同 read_chapter_seq 范式）"
    b.close()


# ============ unit：counterfactual_value_check G3b（全局扫命题节点·反 theater 真消费者） ============

def test_g3b_same_subject_attr_multiple_values_contradicted(prop_env):
    """G3b 正：同(subject,attr_type) 多值 → 0（CONTRADICTED·fork §3.2）。"""
    b, sid, es, ci = prop_env
    猫 = _ensure(ci, sid, "猫")
    颜色 = _ensure(ci, sid, "颜色")
    黑, 白 = _ensure(ci, sid, "黑"), _ensure(ci, sid, "白")
    build_property_edges(es, ci, b, [猫, 颜色, 黑, 白],
                         property_claims=[(0, 1, 2, 0), (0, 1, 3, 0)],
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 0, "同对多值=结构值冲突→CONTRADICTED"


def test_g3b_different_attr_no_false_contradiction(prop_env):
    """G3b 无假矛盾（fork §3.3）：异 attr_type → 1（猫的颜色=黑 + 猫的数量=四 非冲突）。"""
    b, sid, es, ci = prop_env
    猫 = _ensure(ci, sid, "猫")
    颜色, 数量 = _ensure(ci, sid, "颜色"), _ensure(ci, sid, "数量")
    黑, 四 = _ensure(ci, sid, "黑"), _ensure(ci, sid, "四")
    build_property_edges(es, ci, b, [猫, 颜色, 黑, 数量, 四],
                         property_claims=[(0, 1, 2, 0), (0, 3, 4, 0)],
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 1, "异 attr_type 非冲突→无假矛盾"


def test_g3b_different_subject_no_false_contradiction(prop_env):
    """G3b 无假矛盾（fork §3.2）：异 subject → 1（猫的颜色=白 + 狗的颜色=白 非冲突）。"""
    b, sid, es, ci = prop_env
    猫, 狗 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "狗")
    颜色 = _ensure(ci, sid, "颜色")
    白 = _ensure(ci, sid, "白")
    build_property_edges(es, ci, b, [猫, 狗, 颜色, 白],
                         property_claims=[(0, 2, 3, 0), (1, 2, 3, 0)],   # 猫的颜色=白 + 狗的颜色=白
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 1, "异 subject 非冲突→无假矛盾"


def test_g3b_single_value_no_conflict(prop_env):
    """G3b 单值命题 → 1（无冲突·反 theater：非空集永返 1）。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 黑 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑")
    build_property_edges(es, ci, b, [猫, 颜色, 黑],
                         property_claims=[(0, 1, 2, 0)],
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 1


def test_g3b_empty_proposition_nodes_returns_one(prop_env):
    """G3b 无命题节点（gate OFF 生产路径）→ 1（既有行为 bit-identical·G3b 旧 part.unit 扫永空=返1）。"""
    b, sid, es, ci = prop_env
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 1, "空集永返 1（bit-identical·同旧行为）"


# ============ P0.3 命题节点扩展（polarity+modality 进 surface·B1 否定/B2 情态前置） ============

def test_build_property_edges_pol_mod_zero_bit_identical(prop_env):
    """P0.3 bit-identical：pol=0/mod=0 → surface 无后缀 = 既有命题（4-tuple 防御 + 6-tuple pol=0 等价）。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 黑 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑")
    refs = [猫, 颜色, 黑]
    # 4-tuple（既有格式·build 防御 claim[4]/[5] 缺省 0）
    n4 = build_property_edges(es, ci, b, refs,
                              property_claims=[(0, 1, 2, 0)],
                              source=SOURCE_BARE_TEXT, space_id=sid)
    prop_ref_4 = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}", sid)
    # 6-tuple pol=0/mod=0（P0.3 新格式）
    n6 = build_property_edges(es, ci, b, refs,
                              property_claims=[(0, 1, 2, 0, 0, 0, 1, 1)],
                              source=SOURCE_BARE_TEXT, space_id=sid)
    prop_ref_6 = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}", sid)
    assert n4 == 1 and n6 == 0, "4-tuple 建·6-tuple pol=0/mod=0 同节点幂等 skip（surface 无后缀）"
    assert prop_ref_4 is not None and prop_ref_4 == prop_ref_6, "pol=0/mod=0 surface 无后缀 = 同命题节点（bit-identical）"


def test_build_property_edges_polarity_distinct_node(prop_env):
    """P0.3 polarity 进 surface：'A是红'(pol=0) vs 'A不是红'(pol=1) → 异命题节点（G3b 各判·对立非矛盾）。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 红 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "红")
    refs = [猫, 颜色, 红]
    n = build_property_edges(es, ci, b, refs,
                             property_claims=[(0, 1, 2, 0, 0, 0),   # 猫的颜色是红·pol=0
                                              (0, 1, 2, 0, 1, 0)],  # 猫的颜色[不是]红·pol=1
                             source=SOURCE_BARE_TEXT, space_id=sid)
    assert n == 2, "pol=0 vs pol=1 → 异命题节点各 1 PROPERTY 边"
    prop_aff = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}", sid)            # pol=0 无后缀
    prop_neg = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}_1_0", sid)        # pol=1 后缀
    assert prop_aff is not None and prop_neg is not None, "pol=0/pol=1 各建独立命题节点"
    assert prop_aff != prop_neg, "pol 进 surface → 异节点（G3b 各判多值·对立非矛盾）"


def test_build_property_edges_modality_distinct_node(prop_env):
    """P0.3 modality 进 surface：'A必然是B'(mod=1) vs 'A可能是B'(mod=2) → 异命题节点。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 红 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "红")
    refs = [猫, 颜色, 红]
    build_property_edges(es, ci, b, refs,
                         property_claims=[(0, 1, 2, 0, 0, 1),   # mod=1 必然□
                                          (0, 1, 2, 0, 0, 2)],  # mod=2 可能◇
                         source=SOURCE_BARE_TEXT, space_id=sid)
    prop_nec = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}_0_1", sid)   # mod=1
    prop_pos = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}_0_2", sid)   # mod=2
    assert prop_nec is not None and prop_pos is not None, "mod=1/mod=2 各建独立命题节点"
    assert prop_nec != prop_pos, "modality 进 surface → 异节点（模态对当判定基础）"


def test_g3b_polarity_not_false_contradiction(prop_env):
    """P0.4 利好：'A是红'(pol=0) + 'A不是红'(pol=1) → 异节点各 1 值 → G3b 不判矛盾（对立非矛盾·返 1）。

    G3b 判据一行不改（P0.4 利好）：同(subj,attr)不同 pol 自动成异节点·各判多值·
    'A是红'+'A不是红' 两节点各 1 值 → 非'同节点多值矛盾' → G3b 返 1（正确·对立非结构矛盾）。
    """
    b, sid, es, ci = prop_env
    猫, 颜色, 红 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "红")
    build_property_edges(es, ci, b, [猫, 颜色, 红],
                         property_claims=[(0, 1, 2, 0, 0, 0),   # 猫的颜色是红·pol=0
                                          (0, 1, 2, 0, 1, 0)],  # 猫的颜色不是红·pol=1
                         source=SOURCE_BARE_TEXT, space_id=sid)
    graph = ConceptGraph(b)
    assert counterfactual_value_check(None, None, graph) == 1, "pol=0/pol=1 异节点各 1 值 → 非矛盾（对立）→ 1"


# ============ B1 否定 polarity（否定 cue 抽取 + pol=1 命题·P0.3 pol 进 surface 填值） ============

def test_is_negation_cue():
    """B1 否定词识别（ZH 不/没/非/无 + EN not/no/never·exact·独立 helper·不入 cue_type_of）。"""
    assert is_negation_cue("不", LANG_ZH) is True
    assert is_negation_cue("没", LANG_ZH) is True
    assert is_negation_cue("非", LANG_ZH) is True
    assert is_negation_cue("无", LANG_ZH) is True
    assert is_negation_cue("是", LANG_ZH) is False   # 是 是值系词非否定
    assert is_negation_cue("not", LANG_EN) is True
    assert is_negation_cue("never", LANG_EN) is True
    assert is_negation_cue("is", LANG_EN) is False


def test_extract_property_claims_negation_on():
    """B1 正：'猫 的 颜色 不 是 黑' → (0, 2, 4, 0, 1, 0)（pol=1·否定窗口·不 at 是-1·attr=颜色·subj=猫·value=黑）。"""
    tokens = ["猫", "的", "颜色", "不", "是", "黑"]
    claims = extract_property_claims(tokens, lang=LANG_ZH, negation_on=True)
    assert claims == [(0, 2, 5, 0, 1, 0, 1, 1)], "否定窗口 pol=1·value=黑 at j+1=5（不 占位窗口偏移 1）"


def test_extract_property_claims_negation_off_bit_identical():
    """★bit-identical：negation_on=False（NEGATION_MODE OFF 默认）→ '不是'走既有肯定窗口错位 skip（既有行为零变）。

    '猫 的 颜色 不 是 黑'·negation_on=False → 既有窗口 是 at j=4·的 须 at j-2=2=颜色（非 的）→ skip → []。
    """
    tokens = ["猫", "的", "颜色", "不", "是", "黑"]
    claims = extract_property_claims(tokens, lang=LANG_ZH)   # negation_on default False
    assert claims == [], "negation_on=False → 既有肯定窗口·'不是'错位 skip（bit-identical·NEGATION_MODE OFF 默认）"


def test_extract_property_claims_negation_affirmation_unchanged():
    """B1 肯定窗口不受 negation_on 影响：'猫 的 颜色 是 黑' → pol=0（无否定词·negation_on 无效）。"""
    tokens = ["猫", "的", "颜色", "是", "黑"]
    assert extract_property_claims(tokens, lang=LANG_ZH, negation_on=False) == [(0, 2, 4, 0, 0, 0, 1, 1)]
    assert extract_property_claims(tokens, lang=LANG_ZH, negation_on=True) == [(0, 2, 4, 0, 0, 0, 1, 1)], "无否定词·pol=0 不受 negation_on 影响"


def test_extract_property_claims_negation_boundary():
    """B1 否定边界：subj/attr/的 不足 → skip（守反统计·不凑配）。"""
    assert extract_property_claims(["不", "是", "黑"], lang=LANG_ZH, negation_on=True) == [], "否定窗口 j-4<0·跳"
    assert extract_property_claims(["猫", "的", "不", "是", "黑"], lang=LANG_ZH, negation_on=True) == [], "否定窗口 j-4<0（猫 at 0·subj=j-4=-1）·跳"


def test_extract_property_claims_negation_cue_token_rejected():
    """B1 subject/attr/value 自身是否定词 → skip（守反统计·_is_property_cue_token 扩含否定词）。"""
    # '不 的 颜色 不 是 黑'（subj=不 是 cue·跳）
    assert extract_property_claims(["不", "的", "颜色", "不", "是", "黑"], lang=LANG_ZH, negation_on=True) == [], "subj=不 是 cue·跳"


def test_extract_property_claims_gated_negation_off():
    """gate NEGATION_MODE OFF（default）+ PROPOSITION_MODE ON → negation_on=False·'不是'错位 skip（bit-identical）。"""
    saved_prop = gates.PROPOSITION_MODE
    saved_neg = gates.NEGATION_MODE
    gates.PROPOSITION_MODE = True
    gates.NEGATION_MODE = False
    try:
        assert extract_property_claims_gated(["猫", "的", "颜色", "不", "是", "黑"], lang=LANG_ZH) == []
    finally:
        gates.PROPOSITION_MODE = saved_prop
        gates.NEGATION_MODE = saved_neg


def test_extract_property_claims_gated_negation_on():
    """gate NEGATION_MODE ON + PROPOSITION_MODE ON → pol=1 否定命题提取。"""
    saved_prop = gates.PROPOSITION_MODE
    saved_neg = gates.NEGATION_MODE
    gates.PROPOSITION_MODE = True
    gates.NEGATION_MODE = True
    try:
        assert extract_property_claims_gated(["猫", "的", "颜色", "不", "是", "黑"], lang=LANG_ZH) == [(0, 2, 5, 0, 1, 0, 1, 1)]
    finally:
        gates.PROPOSITION_MODE = saved_prop
        gates.NEGATION_MODE = saved_neg
