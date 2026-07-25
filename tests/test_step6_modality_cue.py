"""tests.test_step6_modality_cue — B2 情态 cue 抽取 + modality 填值测试套件（STEP6 PR2）。

P0.3 已把 modality(0-4) 进命题 surface `__prop_{subj}_{attr}_{pol}_{mod}`（done）。PR2 填 modality 值：
  - cue_words：_MODAL_CUES {必然:1,可能:2,也许:2,必须:3,应该:3,可以:4} + modal_op_of + is_modal_cue
  - cue_extractor：extract_property_claims 情态窗口（modal at j-1·同否定几何 offset+1·modal 优先于 negation·互斥）
  - gates：MODALITY_MODE（三级 gate·PROPOSITION_MODE 守整体/NEGATION_MODE 守否定/MODALITY_MODE 守情态）

窗口："X 的 Y [必然] 是 Z" → (subj_idx, attr_idx, val_idx, 0, pol=0, modality)·命题 surface `__prop_..._0_{mod}`。

诚实边界：T 公理形式层墙内（构造性检查·非 truth·情态比命题多一口气）·实质情态真值（认识/规范 W2+动力 W1）defer·
D6 closed-class 情态副词·开放变体（想必/势必/说不定）走 D:11 教师晋升（审计根治 [严重-1]·MODAL_D11_READBACK_MODE·见 test_modal_d11_readback.py）·情态+否定复合窗口 defer。
"""
from __future__ import annotations

import pytest
from pure_integer_ai.config import gates
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.node_store import TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import LANG_ZH, LANG_EN
from pure_integer_ai.cognition.understanding.cue_words import (
    modal_op_of, is_modal_cue, is_negation_cue,
)
from pure_integer_ai.cognition.understanding.cue_extractor import (
    extract_property_claims, extract_property_claims_gated,
)
from pure_integer_ai.cognition.understanding.property import build_property_edges


# ============ 件① cue helpers（modal_op_of / is_modal_cue·独立于 cue_type_of） ============

def test_modal_op_of_zh():
    """ZH 情态词 → modality 编码（0-4·P0.3 surface 后缀 _{pol}_{mod}）。"""
    assert modal_op_of("必然", LANG_ZH) == 1   # □ 必然（认识）
    assert modal_op_of("可能", LANG_ZH) == 2   # ◇ 可能（认识）
    assert modal_op_of("也许", LANG_ZH) == 2   # ◇ 可能（同义）
    assert modal_op_of("必须", LANG_ZH) == 3   # 道义必然
    assert modal_op_of("应该", LANG_ZH) == 3   # 道义必然（should 弱义务·首版归道义必然）
    assert modal_op_of("可以", LANG_ZH) == 4   # 道义可能（permission）


def test_modal_op_of_non_modal_returns_none():
    """非情态词 → None（exact 匹配·不命中零填值·守反统计）。"""
    assert modal_op_of("猫", LANG_ZH) is None
    assert modal_op_of("是", LANG_ZH) is None
    assert modal_op_of("不", LANG_ZH) is None   # 否定词非情态（modal/negation 互斥·不同 helper）
    assert modal_op_of("想必", LANG_ZH) is None   # 开放变体 defer·不在 frozenset（D6）


def test_is_modal_cue():
    """is_modal_cue = modal_op_of is not None（邻居判用·配对两端不取 modal token）。"""
    assert is_modal_cue("必然", LANG_ZH) is True
    assert is_modal_cue("可能", LANG_ZH) is True
    assert is_modal_cue("猫", LANG_ZH) is False
    assert is_modal_cue("不", LANG_ZH) is False   # 否定词非情态


def test_modal_cue_independent_of_cue_type_of():
    """★bit-identical 硬守：必然/可能 不入 cue_type_of（防污染 extract_cues 邻居判·同 是/的/不 范式）。

    情态词太通用若入 cue_type_of → extract_cues 把 必然 当 cue 跳过配对 → 非 bit-identical。
    情态检测走独立 helper（is_modal_cue·非 cue_type_of）·cue_type_of 对 必然/可能 仍返 None。
    """
    from pure_integer_ai.cognition.understanding.cue_words import cue_type_of
    assert cue_type_of("必然", LANG_ZH) is None
    assert cue_type_of("可能", LANG_ZH) is None


# ============ 件② 情态窗口提取（extract_property_claims·modal at j-1·offset+1 同否定几何） ============

def test_extract_property_claims_modal_on_biran():
    """B2 正：'猫 的 颜色 必然 是 黑' → (0, 2, 5, 0, 0, 1)（mod=1·情态窗口·必然 at 是-1·attr=颜色·subj=猫·value=黑）。"""
    tokens = ["猫", "的", "颜色", "必然", "是", "黑"]
    claims = extract_property_claims(tokens, lang=LANG_ZH, modality_on=True)
    assert claims == [(0, 2, 5, 0, 0, 1, 1, 1)], "情态窗口 mod=1·value=黑 at j+1=5（必然 占位窗口偏移 1·同否定几何）"


def test_extract_property_claims_modal_on_keneng():
    """B2：'猫 的 颜色 可能 是 黑' → mod=2（◇可能）。"""
    tokens = ["猫", "的", "颜色", "可能", "是", "黑"]
    claims = extract_property_claims(tokens, lang=LANG_ZH, modality_on=True)
    assert claims == [(0, 2, 5, 0, 0, 2, 1, 1)]


def test_extract_property_claims_modal_on_deontic():
    """B2：必须→mod=3 / 可以→mod=4（道义）。"""
    assert extract_property_claims(["猫", "的", "颜色", "必须", "是", "黑"],
                                   lang=LANG_ZH, modality_on=True) == [(0, 2, 5, 0, 0, 3, 1, 1)]
    assert extract_property_claims(["猫", "的", "颜色", "可以", "是", "黑"],
                                   lang=LANG_ZH, modality_on=True) == [(0, 2, 5, 0, 0, 4, 1, 1)]


def test_extract_property_claims_modal_off_bit_identical():
    """★bit-identical：modality_on=False（MODALITY_MODE OFF 默认）→ '必然是'走既有肯定窗口错位 skip（既有行为零变）。

    '猫 的 颜色 必然 是 黑'·modality_on=False → 既有窗口 是 at j=4·的 须 at j-2=2=颜色（非 的）→ skip → []。
    """
    tokens = ["猫", "的", "颜色", "必然", "是", "黑"]
    claims = extract_property_claims(tokens, lang=LANG_ZH)   # modality_on default False
    assert claims == [], "modality_on=False → 既有肯定窗口·'必然是'错位 skip（bit-identical·MODALITY_MODE OFF 默认）"


def test_extract_property_claims_modal_affirmation_unchanged():
    """B2 肯定窗口不受 modality_on 影响：'猫 的 颜色 是 黑' → pol=0/mod=0（无情态词·modality_on 无效）。"""
    tokens = ["猫", "的", "颜色", "是", "黑"]
    assert extract_property_claims(tokens, lang=LANG_ZH, modality_on=False) == [(0, 2, 4, 0, 0, 0, 1, 1)]
    assert extract_property_claims(tokens, lang=LANG_ZH, modality_on=True) == [(0, 2, 4, 0, 0, 0, 1, 1)], "无情态词·mod=0 不受 modality_on 影响"


def test_extract_property_claims_modal_boundary():
    """B2 情态边界：subj/attr/的 不足 → skip（守反统计·不凑配·同否定边界）。"""
    assert extract_property_claims(["必然", "是", "黑"], lang=LANG_ZH, modality_on=True) == [], "情态窗口 j-4<0·跳"
    assert extract_property_claims(["猫", "的", "必然", "是", "黑"], lang=LANG_ZH, modality_on=True) == [], "情态窗口 j-4<0（猫 at 0·subj=j-4=-1）·跳"


def test_extract_property_claims_modal_no_de_copula_skip():
    """B2：j-3 非 的 → skip（守 的...是 固定窗口·同否定范式）。"""
    tokens = ["猫", "颜色", "必然", "是", "黑"]   # 无 的·j-3=颜色 非 的
    assert extract_property_claims(tokens, lang=LANG_ZH, modality_on=True) == []


def test_extract_property_claims_modal_cue_token_rejected():
    """B2 subject/attr/value 自身是情态词 → skip（守反统计·_is_property_cue_token 扩含情态词）。"""
    # '必然 的 颜色 必然 是 黑'（subj=必然 是 cue·跳）
    assert extract_property_claims(["必然", "的", "颜色", "必然", "是", "黑"],
                                   lang=LANG_ZH, modality_on=True) == [], "subj=必然 是 cue·跳"


def test_extract_property_claims_modal_priority_over_negation():
    """B2 modal 与 negation 同槽 j-1·互斥·先查 modal（modal 优先·首版 modal-only 窗口·复合 defer）。

    '猫 的 颜色 必然 是 黑'·modality_on=True + negation_on=True → modal 命中（必然 是情态词非否定词）→ mod=1 pol=0。
    modal 与 negation 是不同词集·互斥·先查 modal 不影响否定词识别（不 是情态词→走否定分支）。
    """
    tokens = ["猫", "的", "颜色", "必然", "是", "黑"]
    claims = extract_property_claims(tokens, lang=LANG_ZH, modality_on=True, negation_on=True)
    assert claims == [(0, 2, 5, 0, 0, 1, 1, 1)], "modal 优先·必然→mod=1 pol=0（非否定窗口）"


def test_extract_property_claims_negation_still_works_with_modality_on():
    """B2 modality_on 不破坏 B1 否定窗口：'猫 的 颜色 不 是 黑'·modality_on=True+negation_on=True → pol=1 mod=0。"""
    tokens = ["猫", "的", "颜色", "不", "是", "黑"]
    claims = extract_property_claims(tokens, lang=LANG_ZH, modality_on=True, negation_on=True)
    assert claims == [(0, 2, 5, 0, 1, 0, 1, 1)], "不 是否定词非情态→走否定分支·pol=1 mod=0"


def test_extract_property_claims_gated_modality_off():
    """gate MODALITY_MODE OFF（default）+ PROPOSITION_MODE ON → modality_on=False·'必然是'错位 skip（bit-identical）。"""
    saved_prop = gates.PROPOSITION_MODE
    saved_mod = gates.MODALITY_MODE
    gates.PROPOSITION_MODE = True
    gates.MODALITY_MODE = False
    try:
        assert extract_property_claims_gated(["猫", "的", "颜色", "必然", "是", "黑"], lang=LANG_ZH) == []
    finally:
        gates.PROPOSITION_MODE = saved_prop
        gates.MODALITY_MODE = saved_mod


def test_extract_property_claims_gated_modality_on():
    """gate MODALITY_MODE ON + PROPOSITION_MODE ON → mod=1 情态命题提取。"""
    saved_prop = gates.PROPOSITION_MODE
    saved_mod = gates.MODALITY_MODE
    gates.PROPOSITION_MODE = True
    gates.MODALITY_MODE = True
    try:
        assert extract_property_claims_gated(["猫", "的", "颜色", "必然", "是", "黑"], lang=LANG_ZH) == [(0, 2, 5, 0, 0, 1, 1, 1)]
    finally:
        gates.PROPOSITION_MODE = saved_prop
        gates.MODALITY_MODE = saved_mod


# ============ 件③ e2e（cue → build → surface·modality 进命题节点） ============

@pytest.fixture
def prop_env():
    """build_property_edges 单测环境（dict backend·core space·composes_attr 已注册·同 test_g1_proposition）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    yield b, sid, es, ci
    b.close()


def _ensure(ci, sid, surface):
    return ci.ensure(surface, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)


def test_modal_claim_builds_distinct_surface(prop_env):
    """e2e：情态 claim (mod=1) → build_property_edges 建命题节点 surface `__prop_..._0_1`（P0.3 mod 进 surface）。

    mod=0（实然）无后缀·mod>0 加 _0_{mod} 后缀·异 modality 异节点（模态对当判定基础·PR3 G3b 用）。
    """
    b, sid, es, ci = prop_env
    猫, 颜色, 黑 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑")
    refs = [猫, 颜色, 黑]
    build_property_edges(es, ci, b, refs,
                         property_claims=[(0, 1, 2, 0, 0, 1)],   # mod=1 必然□
                         source=SOURCE_BARE_TEXT, space_id=sid)
    prop_nec = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}_0_1", sid)
    assert prop_nec is not None, "mod=1 命题节点 surface `__prop_..._0_1` 建出"


def test_modal_claim_distinct_from_assertoric(prop_env):
    """e2e：'A必然是B'(mod=1) vs 'A是B'(mod=0) → 异命题节点（modality 进 surface·G3b 各判·模态对当基础）。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 黑 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑")
    refs = [猫, 颜色, 黑]
    build_property_edges(es, ci, b, refs,
                         property_claims=[(0, 1, 2, 0, 0, 0),   # mod=0 实然
                                          (0, 1, 2, 0, 0, 1)],  # mod=1 必然□
                         source=SOURCE_BARE_TEXT, space_id=sid)
    prop_assert = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}", sid)        # mod=0 无后缀
    prop_nec = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}_0_1", sid)       # mod=1 后缀
    assert prop_assert is not None and prop_nec is not None
    assert prop_assert != prop_nec, "mod=0 vs mod=1 → 异节点（modality 进 surface·G3b 各判·模态对当基础）"


def test_modal_claim_distinct_modality_values(prop_env):
    """e2e：必然(mod=1) vs 可能(mod=2) → 异命题节点（不同 modality 异节点）。"""
    b, sid, es, ci = prop_env
    猫, 颜色, 黑 = _ensure(ci, sid, "猫"), _ensure(ci, sid, "颜色"), _ensure(ci, sid, "黑")
    refs = [猫, 颜色, 黑]
    build_property_edges(es, ci, b, refs,
                         property_claims=[(0, 1, 2, 0, 0, 1),   # mod=1 必然□
                                          (0, 1, 2, 0, 0, 2)],  # mod=2 可能◇
                         source=SOURCE_BARE_TEXT, space_id=sid)
    prop_nec = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}_0_1", sid)
    prop_pos = ci.lookup(f"__prop_{猫[0]}_{猫[1]}_{颜色[0]}_{颜色[1]}_0_2", sid)
    assert prop_nec is not None and prop_pos is not None
    assert prop_nec != prop_pos, "mod=1/mod=2 → 异节点"


# ============ 件④ D6 不写死守卫 ============

def test_d6_modal_cues_closed_class_only():
    """D6 守卫：_MODAL_CUES ZH 只 closed-class {必然,可能,也许,必须,应该,可以}·开放变体零硬编码。"""
    from pure_integer_ai.cognition.understanding.cue_words import _MODAL_CUES
    zh_modal = _MODAL_CUES[LANG_ZH]
    # closed-class 核心 6 词（含同义 也许→2）
    expected = {"必然": 1, "可能": 2, "也许": 2, "必须": 3, "应该": 3, "可以": 4}
    assert zh_modal == expected, f"ZH 情态 closed-class 只 6 词·got {zh_modal}"
    # 开放变体不在 frozenset（走 D:11 教师晋升·审计根治 [严重-1] 已建 MODAL_D11_READBACK_MODE·见 test_modal_d11_readback.py）
    for open_word in ("想必", "势必", "说不定", "大概", "也许可能"):
        assert open_word not in zh_modal, f"开放变体 {open_word} 须零硬编码（D6）"


def test_modal_op_of_has_d11_readback_signature():
    """签名核证：modal_op_of 有 D:11 readback 4 参（审计根治 [严重-1]·路 A2-变体·镜像 arith_op_of）。
    无 REL_MODALITY（modal 非符号域二元关系·走 modal_kind concept+ATTR_MODAL_KIND·非 REL_MODALITY）。
    **行为验证**（gate ON/OFF 差分 + 开放变体 D:11 教师晋升）在 test_modal_d11_readback.py·本测只验签名。"""
    import inspect
    import pure_integer_ai.cognition.shared.relation_primitives as rp
    # REL_* 枚举无 REL_MODALITY（modal 非二元关系·走 modal_kind concept+ATTR_MODAL_KIND=22+MARK_MODAL_KIND=5·非 REL_*）
    assert not hasattr(rp, "REL_MODALITY"), "无 REL_MODALITY（modal 非符号域二元关系·走 modal_kind concept）"
    # 审计根治 [严重-1]：modal_op_of 有 D:11 readback 4 参（路 A2-变体·镜像 arith_op_of·解换名字写死）
    sig = inspect.signature(modal_op_of)
    assert "backend" in sig.parameters and "edge_store" in sig.parameters, (
        "审计根治后 modal_op_of 有 D:11 readback 4 参（路 A2-变体·镜像 arith_op_of·解 [严重-1] 换名字写死）")
    assert "space_id" in sig.parameters and "concept_index" in sig.parameters, "4 参全"


# ============ 件⑤ 守墙标注（T 公理形式层·非 truth） ============

def test_modal_wall_annotation_formal_not_truth():
    """守墙：T 公理形式层墙内（构造性检查·非 truth）·实质情态真值（认识/规范 W2+动力 W1）defer。

    核证 modal_op_of docstring 标注"T 公理形式层墙内"+"实质情态真值 defer"（诚实边界·非纸面闭合）。
    """
    import inspect
    import pure_integer_ai.cognition.understanding.cue_words as cw
    src = inspect.getsource(cw.modal_op_of)
    assert "T 公理" in src or "形式层" in src, "modal_op_of 须标 T 公理形式层墙内"
    assert "W2" in src or "W1" in src or "defer" in src, "modal_op_of 须标实质情态真值 W2/W1 defer"
