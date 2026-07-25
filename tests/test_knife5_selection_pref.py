"""刀5 件5 选择倾向共现统计测试（学习放开 6 刀·任务 #596·doc/重来_学习放开整合设计_纠偏纠偏.md §5 刀5 + 概念空间改造方案 §十）。

刀5 件5 地基 = selection_pref_count 统计表 + build_selection_pref_count builder + observe 接线 +
  read API。PR 软加权 dock seed defer S4·D:13 边 defer（守 role_scheme）·sp_sn reward feed defer。

**件5 验收判据**（反 theater·"石头追老鼠" stats count 区分·非 PR 偏置）：
  - 训练"狐狸追鸡/猫追老鼠/狗追猫"多次 → (追, LCA(动物类)=动物) sp_tn 高。
  - 偶现"石头追老鼠"一次 → (追, 石头) sp_tn 低（石头无动物 IS_A 祖先·class=石头自身）。
  - 验收：read(追, 动物).sp_tn >> read(追, 石头).sp_tn → 反 theater 判据成立。

**诚实边界**：地基 only（PR 软加权/sp_sn feed/D:13/predicate 写时识别全 defer）·反 theater 用 stats
  count 区分非 PR 偏置·stable≠correct（"吃猫"数据见过就高 count 接地墙外）。

铁律：纯整数 / 确定性 bit-identical（SELECTION_PREF_MODE 默认 OFF·sorted/NodeRef 升序）/ 不写死
  （IS_A LCA 结构查询·emergent_role 涌现非词性）/ §8.1c（统计表非关系边）/ §8.5（不建边）/ reward
  CAUSES-only（observe 写 sp_tn only·sp_sn defer）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_TEACHER, EPI_STRUCTURED
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import EDGE_IS_A
from pure_integer_ai.storage.composes_attr import register_composes_attr
from pure_integer_ai.storage.selection_pref_count import (
    register_selection_pref_count, read_selection_pref_count,
    record_selection_pref_cooccur, SELECTION_PREF_COUNT_TABLE,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import LANG_ZH
from pure_integer_ai.cognition.understanding.selection_pref import (
    build_selection_pref_count, _nearest_isa_ancestor,
)
from pure_integer_ai.cognition.process.abstraction import build_isa_ancestor_map
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def sp_env():
    """件5 单测环境（dict backend·core space·composes_attr + selection_pref_count 注册）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    register_selection_pref_count(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    yield b, sid, es, ns, ci
    b.close()


def _ensure_word(ci, sid, surface):
    return ci.ensure(surface, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)


def _add_is_a(es, child, parent, sid):
    """建 IS_A 边（教师种·EPI_STRUCTURED·供 build_isa_ancestor_map 读）。"""
    es.add(space_id_from=child[0], local_id_from=child[1],
           space_id_to=parent[0], local_id_to=parent[1],
           edge_type=EDGE_IS_A, strength=1, source=SOURCE_TEACHER,
           epistemic_origin=EPI_STRUCTURED, tier=TIER_PRIMARY)


# ============ unit：selection_pref_count 表 read/record ============

def test_record_cooccur_inserts_and_increments(sp_env):
    """record_selection_pref_cooccur：首次 insert(sp_tn=1)·再次 sp_tn++。base/sp_sn 守 0（defer）。"""
    b, sid, es, ns, ci = sp_env
    a = _ensure_word(ci, sid, "追")
    c = _ensure_word(ci, sid, "动物")
    assert read_selection_pref_count(b, a, c) is None, "首次前无行"
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c)
    record_selection_pref_cooccur(b, ref_a=a, ref_class=c)
    got = read_selection_pref_count(b, a, c)
    assert got == (0, 0, 2), "sp_tn=2·base/sp_sn 守 0（首版 defer）"


def test_read_unregistered_table_returns_none():
    """表未注册 → read 返 None（向后兼容·同 read_experience_count 范式）。"""
    b = DictBackend()
    bootstrap(b)   # 不调 register_selection_pref_count
    a = (1, 1)
    c = (1, 2)
    assert read_selection_pref_count(b, a, c) is None, "表未注册→None"


def test_record_unregistered_table_skips_silently():
    """表未注册 → record 静默 skip（向后兼容·不抛）。"""
    b = DictBackend()
    bootstrap(b)
    record_selection_pref_cooccur(b, ref_a=(1, 1), ref_class=(1, 2))   # 不抛


# ============ unit：_nearest_isa_ancestor ============

def test_nearest_isa_ancestor_no_ancestors_returns_self(sp_env):
    """无 IS_A 祖先 → 返 ref 自身（冷启动退化恒等·bit-identical）。"""
    b, sid, es, ns, ci = sp_env
    石头 = _ensure_word(ci, sid, "石头")
    amap = build_isa_ancestor_map(b, space_id=sid)   # 无 IS_A 边
    assert _nearest_isa_ancestor(amap, 石头) == 石头, "无祖先→自身"


def test_nearest_isa_ancestor_returns_deepest(sp_env):
    """有 IS_A 祖先 → 最近祖先最深（nearest_isa_ancestor·非 NodeRef 升序首 min·S4 项2）。"""
    b, sid, es, ns, ci = sp_env
    狐狸 = _ensure_word(ci, sid, "狐狸")
    动物 = _ensure_word(ci, sid, "动物")
    生物 = _ensure_word(ci, sid, "生物")
    _add_is_a(es, 狐狸, 动物, sid)
    _add_is_a(es, 动物, 生物, sid)
    amap = build_isa_ancestor_map(b, space_id=sid)
    # 狐狸 祖先 = {动物, 生物}·最近祖先 = 动物（最深·nearest_isa_ancestor·非 min 生物）
    got = _nearest_isa_ancestor(amap, 狐狸)
    assert got in (动物, 生物), "返祖先之一"
    assert got == 动物, "最近祖先=动物（最深·非 min 生物·S4 项2）"


# ============ unit：build_selection_pref_count（gate + 双向记录） ============

def test_build_selection_pref_count_gate_off_returns_zero(sp_env):
    """SELECTION_PREF_MODE OFF → 返 0·不写表（守回归 bit-identical）。"""
    b, sid, es, ns, ci = sp_env
    追 = _ensure_word(ci, sid, "追")
    老鼠 = _ensure_word(ci, sid, "老鼠")
    saved = gates.SELECTION_PREF_MODE
    gates.SELECTION_PREF_MODE = False
    try:
        n = build_selection_pref_count(b, [追, 老鼠], space_id=sid, lang=LANG_ZH)
        assert n == 0, "gate OFF → 0"
        assert read_selection_pref_count(b, 追, 老鼠) is None, "gate OFF → 不写表"
    finally:
        gates.SELECTION_PREF_MODE = saved


def test_build_selection_pref_count_records_both_directions(sp_env):
    """gate ON：段内配对双向记录 (a, class_of(b)) 与 (b, class_of(a))·各 sp_tn++。"""
    b, sid, es, ns, ci = sp_env
    追 = _ensure_word(ci, sid, "追")
    老鼠 = _ensure_word(ci, sid, "老鼠")
    动物 = _ensure_word(ci, sid, "动物")
    _add_is_a(es, 老鼠, 动物, sid)
    saved = gates.SELECTION_PREF_MODE
    gates.SELECTION_PREF_MODE = True
    try:
        n = build_selection_pref_count(b, [追, 老鼠], space_id=sid, lang=LANG_ZH)
        # 1 配对 × 2 双向 = 2 记录
        assert n == 2, f"双向记录 2·got {n}"
        # (追, class_of(老鼠)=动物) sp_tn=1
        assert read_selection_pref_count(b, 追, 动物) == (0, 0, 1), "(追, 动物) +1"
        # (老鼠, class_of(追)=追[无祖先自身]) sp_tn=1
        assert read_selection_pref_count(b, 老鼠, 追) == (0, 0, 1), "(老鼠, 追) +1（追 无 IS_A·class=自身）"
    finally:
        gates.SELECTION_PREF_MODE = saved


# ============ 反 theater e2e：石头追老鼠 stats count 区分 ============

def test_knife5_selection_pref_stone_chase_mouse_anomaly(sp_env):
    """件5 反 theater：训练"狐狸追鸡/猫追老鼠/狗追猫"多次 → (追, 动物) sp_tn 高·
    偶现"石头追老鼠"一次 → (追, 石头) sp_tn 低（石头无动物 IS_A 祖先）。
    验收：read(追, 动物).sp_tn >> read(追, 石头).sp_tn → 选择倾向异常判据成立（非 PR 偏置·defer S4）。
    """
    b, sid, es, ns, ci = sp_env
    # ensure words
    狐狸 = _ensure_word(ci, sid, "狐狸")
    追 = _ensure_word(ci, sid, "追")
    鸡 = _ensure_word(ci, sid, "鸡")
    猫 = _ensure_word(ci, sid, "猫")
    老鼠 = _ensure_word(ci, sid, "老鼠")
    狗 = _ensure_word(ci, sid, "狗")
    石头 = _ensure_word(ci, sid, "石头")
    动物 = _ensure_word(ci, sid, "动物")
    # IS_A 类：动物类（狐狸/鸡/猫/老鼠/狗 IS_A 动物）·石头无 IS_A（class=自身）·追无 IS_A（class=自身·动词位）
    for w in (狐狸, 鸡, 猫, 老鼠, 狗):
        _add_is_a(es, w, 动物, sid)

    saved = gates.SELECTION_PREF_MODE
    gates.SELECTION_PREF_MODE = True
    try:
        # 训练：动物类追动物类 ×3 each（3 种 pattern × 3 次）
        for _ in range(3):
            build_selection_pref_count(b, [狐狸, 追, 鸡], space_id=sid, lang=LANG_ZH)
            build_selection_pref_count(b, [猫, 追, 老鼠], space_id=sid, lang=LANG_ZH)
            build_selection_pref_count(b, [狗, 追, 猫], space_id=sid, lang=LANG_ZH)
        # 偶现异常：石头（无生命）追老鼠 ×1
        build_selection_pref_count(b, [石头, 追, 老鼠], space_id=sid, lang=LANG_ZH)
    finally:
        gates.SELECTION_PREF_MODE = saved

    # 反 theater 验收：(追, 动物) 高 vs (追, 石头) 低
    got_animal = read_selection_pref_count(b, 追, 动物)
    got_stone = read_selection_pref_count(b, 追, 石头)
    assert got_animal is not None, "(追, 动物) 行存在（训练数据充分）"
    assert got_stone is not None, "(追, 石头) 行存在（异常偶现 1 次）"
    sp_tn_animal = got_animal[2]
    sp_tn_stone = got_stone[2]
    assert sp_tn_animal > sp_tn_stone, \
        f"反 theater：(追,动物) sp_tn={sp_tn_animal} > (追,石头) sp_tn={sp_tn_stone}"
    assert sp_tn_stone == 1, f"(追,石头) 仅 1 次（异常偶现）·got {sp_tn_stone}"
    assert sp_tn_animal >= 9, \
        f"(追,动物) ≥9（3 pattern × 3 次 × 每 pattern 2 个动物类配对含追）·got {sp_tn_animal}"


def test_build_selection_pref_count_dedup_same_pair(sp_env):
    """同段内 a==b 跳（segment_cooccurrence_pairs i<j·a≠b 保证·但显式验守）。"""
    b, sid, es, ns, ci = sp_env
    追 = _ensure_word(ci, sid, "追")
    saved = gates.SELECTION_PREF_MODE
    gates.SELECTION_PREF_MODE = True
    try:
        # 单 token 段：无配对
        n = build_selection_pref_count(b, [追], space_id=sid, lang=LANG_ZH)
        assert n == 0, "单 token 无配对"
        # 空 段
        assert build_selection_pref_count(b, [], space_id=sid, lang=LANG_ZH) == 0
    finally:
        gates.SELECTION_PREF_MODE = saved


# ============ e2e：observe 真路径（observe.py:220 build_selection_pref_count 接线） ============

def test_knife5_observe_writes_selection_pref(tmp_path, monkeypatch):
    """e2e：observe 真路径（formal_train run_round_full → observe → build_selection_pref_count）
    写 selection_pref_count 表 sp_tn·防 observe 接线 future regression 静默（RISK-B 审1）。

    直调 run_round_full（绕 metric gate·SELECTION_PREF_MODE ON 模拟生产入口）·验 observe.py:220
    接线真工作（self.backend 真路径下 sp_tn 写入 selection_pref_count 表）。
    """
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    from pure_integer_ai.experiments.formal_train import make_train_context, DefaultRoundRunner
    from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_PRECEDES
    from pure_integer_ai.training.stages import STAGE3_REWARD
    b = DictBackend()
    ctx = make_train_context(b)
    runner = DefaultRoundRunner()
    saved = gates.SELECTION_PREF_MODE
    gates.SELECTION_PREF_MODE = True
    try:
        item = CollectedItem(tokens=["狐狸", "追", "鸡"], collect_type=COLLECT_PRECEDES)
        runner.run_round(ctx, item, STAGE3_REWARD, 0)
    finally:
        gates.SELECTION_PREF_MODE = saved
    # observe 真路径写 sp_tn（IS_A ancestor_map 空→class_of=自身·token 级配对·但 sp_tn 已写）
    rows = b.select(SELECTION_PREF_COUNT_TABLE)
    assert len(rows) > 0, \
        "observe 真路径写 selection_pref_count（段内 狐狸/追/鸡 三 token i<j 配对·sp_tn 写·RISK-B 防接线 regression）"
    assert all(int(r["sp_tn"]) >= 1 for r in rows), "每行 sp_tn ≥ 1"
    assert all(int(r["sp_sn"]) == 0 for r in rows), "sp_sn 守 0（reward feed defer S4）"
