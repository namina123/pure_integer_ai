"""刀4 件2 验证下涌现关系学习测试（学习放开 6 刀·任务 #595·初心核心·doc/重来_学习放开整合设计_纠偏纠偏.md §5 刀4）。

刀4 = 涌现假设生成器（PRECEDES 链 connector 定位）+ D:11 SHADOW 落边 + experience_count 概念维验证 +
  promote 双轨（experience 主导 + teacher 加分）+ cue_type_of D:11 readback。

**刀4 验收判据**（反 theater·"引发"真涌现词·NOT 在固化件）：
  - 冷启动 cue_type_of("引发") = None（frozenset 无·D:11 无 PRIMARY 边）。
  - observe "X 引发 Y" 反复 → 生成器 PRECEDES 链定位"引发"connector → 假设 引发→REL_CAUSES → D:11 SHADOW 落边。
  - experience feed（reward>0 episode ×N）→ e_sn/e_tn 达标 → promote PRIMARY（experience 主导·无教师）。
  - 第二轮 cue_type_of("引发", readback) 返 CAUSES_CUE_FORWARD（非 None·涌现学习得证）。
  - 反 theater 牙：SHADOW 未 promote 前 readback 返 None（未验证不注入）。

**诚实边界**（#479 墙·构造性非真独立源）：experience 验证 = reward 构造性必然（judge 来自教师 GT）·
  非真独立源验证（#479 defer·断奶后内生判据缺）。

铁律：纯整数 / 确定性 bit-identical / 不写死（观察签名·签名→REL_CAUSES 元定义 enum 例外）/ §8.1c
  （staging SHADOW→验证→promote 非共现直落）/ §8.8（关系概念=first-class 节点）/ epistemic 闭合
  （record_word_concept assert 不动·SHADOW 走独立构造器绕开）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT, SOURCE_TEACHER, EPI_STRUCTURED
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, TIER_SHADOW, NODE_CONCEPT
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.edge_types import (
    EDGE_RELATION_SIGNAL, EDGE_PRECEDES, EDGE_COOCCURS, EDGE_CAUSES,
)
from pure_integer_ai.storage.composes_attr import register_composes_attr, read_composes_attrs, ATTR_RELATION_PRIMITIVE
from pure_integer_ai.storage.experience_count import (
    register_experience_count, record_experience_outcome, read_experience_count,
)
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import LANG_ZH
from pure_integer_ai.cognition.shared.relation_primitives import (
    ensure_relation_primitives, REL_CAUSES, REL_SUBSET,
)
from pure_integer_ai.cognition.understanding.word_concept_signal import record_word_concept
from pure_integer_ai.cognition.understanding.cue_words import cue_type_of, CAUSES_CUE_FORWARD
from pure_integer_ai.cognition.understanding.emergent_relation_signal import (
    record_emergent_relation_signal_shadow,
    generate_emergent_hypotheses,
    EMERGENT_COOCCURS_MIN,
)
from pure_integer_ai.cognition.understanding.emergent_relation_feed import (
    collect_emergent_word_concepts_for_feed,
)
from pure_integer_ai.training.promote import (
    promote_edge, promote_report, _experience_ok, _definition_ok_d11,
    PROMOTE_EXP_FREQ_MIN,
)
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_PRECEDES


# ---- fixtures ----

@pytest.fixture
def emerg_env():
    """涌现关系单测环境（dict backend·core space·composes_attr + experience_count 注册）。"""
    b = DictBackend()
    bootstrap(b)
    register_composes_attr(b)
    register_experience_count(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    yield b, sid, es, ns, ci
    b.close()


def _lang_item(tokens: list[str]) -> CollectedItem:
    """语言 corpus item（LANG_ZH·空白已切 token）。"""
    return CollectedItem(tokens=tokens, collect_type=COLLECT_PRECEDES)


def _ensure_word(ci, sid, surface):
    return ci.ensure(surface, space_id=sid, tier=TIER_PRIMARY, node_type=NODE_CONCEPT)


# ============ unit：record_emergent_relation_signal_shadow（子环2 SHADOW 落边） ============

def test_record_shadow_builds_d11_shadow_edge(emerg_env):
    """落 D:11 SHADOW 边：tier=SHADOW·source=SOURCE_BARE_TEXT·epistemic=None·strength=1（三重隔离·镜像 COOCCURS）。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    w = _ensure_word(ci, sid, "引发")
    n = record_emergent_relation_signal_shadow(es, w, rel_refs[REL_CAUSES], space_id=sid)
    assert n == 1, "建 1 条 D:11 SHADOW 边"
    e = es.get(space_id_from=w[0], local_id_from=w[1],
               space_id_to=rel_refs[REL_CAUSES][0], local_id_to=rel_refs[REL_CAUSES][1],
               edge_type=EDGE_RELATION_SIGNAL)
    assert e is not None, "D:11 边建"
    assert e["tier"] == TIER_SHADOW, "tier=TIER_SHADOW（staging 隔离）"
    assert e["source"] == SOURCE_BARE_TEXT, "source=SOURCE_BARE_TEXT（涌现·非教师）"
    assert e["epistemic_origin"] is None, "epistemic_origin=None（伴随检疫·绕 record_word_concept assert）"
    assert e["strength"] == 1, "strength=1（不接 reward·非学习对象初值）"
    assert e["sn"] == 0 and e["tn"] == 0, "sn/tn=0（不接 reward 反传）"


def test_record_shadow_idempotent(emerg_env):
    """query_from 按 source 幂等 skip：同 (word, rel, D:11, BARE_TEXT) 重种 → 0。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    w = _ensure_word(ci, sid, "引发")
    n1 = record_emergent_relation_signal_shadow(es, w, rel_refs[REL_CAUSES], space_id=sid)
    n2 = record_emergent_relation_signal_shadow(es, w, rel_refs[REL_CAUSES], space_id=sid)
    assert n1 == 1 and n2 == 0, "第二次种 → skip（query_from 幂等）"
    d11 = [r for r in b.select("edge", where={"edge_type": EDGE_RELATION_SIGNAL})]
    assert len(d11) == 1, "重种不增边"


def test_record_shadow_coexists_with_teacher_seed(emerg_env):
    """SHADOW（BARE_TEXT）与教师种子（TEACHER/PRIMARY）异源并存：同 (word,rel,D:11) 两源两条边。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    w = _ensure_word(ci, sid, "引发")
    record_word_concept(ci, es, "引发", rel_refs[REL_CAUSES], space_id=sid)  # 教师 PRIMARY 种子
    record_emergent_relation_signal_shadow(es, w, rel_refs[REL_CAUSES], space_id=sid)  # 涌现 SHADOW
    d11 = [r for r in b.select("edge", where={"edge_type": EDGE_RELATION_SIGNAL})]
    assert len(d11) == 2, "异源并存（TEACHER PRIMARY + BARE_TEXT SHADOW）"
    tiers = sorted(r["tier"] for r in d11)
    assert tiers == sorted([TIER_PRIMARY, TIER_SHADOW]), "一 PRIMARY 一 SHADOW"


def test_record_shadow_defensive_short_circuit(emerg_env):
    """防御短路：(0,0)/None word_ref/rel_ref → 0（无副作用）。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    assert record_emergent_relation_signal_shadow(es, (0, 0), rel_refs[REL_CAUSES], space_id=sid) == 0
    assert record_emergent_relation_signal_shadow(es, None, rel_refs[REL_CAUSES], space_id=sid) == 0
    w = _ensure_word(ci, sid, "引发")
    assert record_emergent_relation_signal_shadow(es, w, (0, 0), space_id=sid) == 0
    assert record_emergent_relation_signal_shadow(es, w, None, space_id=sid) == 0
    d11 = [r for r in b.select("edge", where={"edge_type": EDGE_RELATION_SIGNAL})]
    assert len(d11) == 0, "防御短路零边副作用"


# ============ unit：generate_emergent_hypotheses（子环1 PRECEDES 链 connector 定位） ============

def _build_x_w_y_pattern(es, ci, sid, x_surf, w_surf, y_surf, n_times):
    """模拟 observe n 段 "x w y"：PRECEDES x→w→y + COOCCURS 三角（含 (x,y)）·返 (x,w,y) refs。"""
    x = _ensure_word(ci, sid, x_surf)
    w = _ensure_word(ci, sid, w_surf)
    y = _ensure_word(ci, sid, y_surf)
    for _ in range(n_times):
        es.add(space_id_from=x[0], local_id_from=x[1], space_id_to=w[0], local_id_to=w[1],
               edge_type=EDGE_PRECEDES, strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY)
        es.add(space_id_from=w[0], local_id_from=w[1], space_id_to=y[0], local_id_to=y[1],
               edge_type=EDGE_PRECEDES, strength=1, source=SOURCE_BARE_TEXT, tier=TIER_PRIMARY)
        es.add(space_id_from=x[0], local_id_from=x[1], space_id_to=y[0], local_id_to=y[1],
               edge_type=EDGE_COOCCURS, strength=1, source=SOURCE_BARE_TEXT, tier=TIER_SHADOW)
    return x, w, y


def test_generate_hypotheses_detects_connector(emerg_env):
    """PRECEDES 链 connector 定位：3× "雨 引发 洪水" → 引发 有 pred={雨} succ={洪水}·
    COOCCURS(雨,洪水)=3 ≥ MIN ∧ 无 CAUSES → 假设 引发→REL_CAUSES。"""
    b, sid, es, ns, ci = emerg_env
    ensure_relation_primitives(ci, b, space_id=sid)
    雨, 引发, 洪水 = _build_x_w_y_pattern(es, ci, sid, "雨", "引发", "洪水", 3)
    hyps = generate_emergent_hypotheses(b, es, ci, space_id=sid, excluded_word_refs=set())
    assert len(hyps) == 1, f"涌 1 假设（引发 connector）·got {len(hyps)}"
    w_ref, rel_kind, rel_ref = hyps[0]
    assert w_ref == 引发, "候选 = 引发（connector）"
    assert rel_kind == REL_CAUSES, "rel_kind = REL_CAUSES（首版只涌 REL_CAUSES）"


def test_generate_hypotheses_excludes_cue_words(emerg_env):
    """C9-bis §D：'导致'在 excluded_word_refs（cue 固化词）→ 不涌（防 reward 调固化件）。"""
    b, sid, es, ns, ci = emerg_env
    ensure_relation_primitives(ci, b, space_id=sid)
    雨, 导致, 洪水 = _build_x_w_y_pattern(es, ci, sid, "雨", "导致", "洪水", 3)
    hyps = generate_emergent_hypotheses(b, es, ci, space_id=sid, excluded_word_refs={导致})
    assert len(hyps) == 0, "'导致'被排除（C9-bis §D 候选池排除清单）·不涌"


def test_generate_hypotheses_low_cooccurs_no_emerge(emerg_env):
    """COOCCURS(x,y) < MIN（仅 2 次）→ 不涌（防单次噪声·签名 s1 未达）。"""
    b, sid, es, ns, ci = emerg_env
    ensure_relation_primitives(ci, b, space_id=sid)
    _build_x_w_y_pattern(es, ci, sid, "雨", "引发", "洪水", 2)  # 2 < EMERGENT_COOCCURS_MIN=3
    hyps = generate_emergent_hypotheses(b, es, ci, space_id=sid, excluded_word_refs=set())
    assert len(hyps) == 0, f"COOCCURS < {EMERGENT_COOCCURS_MIN} → 不涌（防噪声）"


def test_generate_hypotheses_skips_if_causes_exists(emerg_env):
    """有 CAUSES(x,y)（已确认因果）→ w 不再是涌现候选（签名 s3=0·无确认 CAUSES 未达）。"""
    b, sid, es, ns, ci = emerg_env
    ensure_relation_primitives(ci, b, space_id=sid)
    雨, 引发, 洪水 = _build_x_w_y_pattern(es, ci, sid, "雨", "引发", "洪水", 3)
    # 手建 CAUSES(雨→洪水)（已确认因果·非涌现）
    es.add(space_id_from=雨[0], local_id_from=雨[1], space_id_to=洪水[0], local_id_to=洪水[1],
           edge_type=EDGE_CAUSES, strength=1, source=SOURCE_TEACHER, tier=TIER_PRIMARY,
           epistemic_origin=EPI_STRUCTURED)
    hyps = generate_emergent_hypotheses(b, es, ci, space_id=sid, excluded_word_refs=set())
    assert len(hyps) == 0, "有 CAUSES(雨,洪水) → 引发 不再涌现候选（已确认非涌现）"


def test_generate_hypotheses_skips_existing_d11(emerg_env):
    """w 已有 D:11 边（已种/已涌现）→ 不重复涌（generator 排除已有 D:11 的词）。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    雨, 引发, 洪水 = _build_x_w_y_pattern(es, ci, sid, "雨", "引发", "洪水", 3)
    # 预先种 引发→REL_CAUSES D:11 边（教师种子）
    record_word_concept(ci, es, "引发", rel_refs[REL_CAUSES], space_id=sid)
    hyps = generate_emergent_hypotheses(b, es, ci, space_id=sid, excluded_word_refs=set())
    assert all(h[0] != 引发 for h in hyps), "引发 已有 D:11 → 不重复涌"


# ============ unit：collect_emergent_word_concepts_for_feed（子环3 鸡生蛋破解） ============

def test_collect_feed_returns_shadow_d11_words(emerg_env):
    """D:11 SHADOW 边指向 REL_CAUSES → 返 from word refs（reward_propagate concept_targets 扩展用）。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    w1 = _ensure_word(ci, sid, "引发")
    w2 = _ensure_word(ci, sid, "促使")
    record_emergent_relation_signal_shadow(es, w1, rel_refs[REL_CAUSES], space_id=sid)
    record_emergent_relation_signal_shadow(es, w2, rel_refs[REL_CAUSES], space_id=sid)
    # 教师 PRIMARY 种子（不应被收——tier_filter=SHADOW）
    w3 = _ensure_word(ci, sid, "导致")
    record_word_concept(ci, es, "导致", rel_refs[REL_CAUSES], space_id=sid)
    refs = collect_emergent_word_concepts_for_feed(b)
    assert w1 in refs and w2 in refs, "D:11 SHADOW 候选 word refs 被收"
    assert w3 not in refs, "教师 PRIMARY 种子不被收（tier_filter=TIER_SHADOW·只 feed 未晋升）"


def test_collect_feed_filters_rel_kind(emerg_env):
    """rel_kind_filter=REL_CAUSES 只收指向 REL_CAUSES 的 SHADOW 边（异 kind 不收）。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    w_causes = _ensure_word(ci, sid, "引发")
    w_subset = _ensure_word(ci, sid, "喻")  # 假设涌为 REL_SUBSET
    record_emergent_relation_signal_shadow(es, w_causes, rel_refs[REL_CAUSES], space_id=sid)
    record_emergent_relation_signal_shadow(es, w_subset, rel_refs[REL_SUBSET], space_id=sid)
    refs = collect_emergent_word_concepts_for_feed(b, rel_kind_filter=REL_CAUSES)
    assert w_causes in refs, "REL_CAUSES SHADOW 候选被收"
    assert w_subset not in refs, "REL_SUBSET SHADOW 不被收（rel_kind_filter 过滤）"


# ============ unit：_experience_ok + _definition_ok_d11（子环4 双轨闸） ============

def test_experience_ok_threshold(emerg_env):
    """_experience_ok：e_sn/(e_sn+e_tn) ≥ 1/2 ∧ (e_sn+e_tn) ≥ PROMOTE_EXP_FREQ_MIN → True。
    注：reward>0 feed 使 e_sn++&e_tn++（双计·同 _reward_ok 范式）·故 1 feed → total=2·2 feed → total=4。
    """
    b, sid, es, ns, ci = emerg_env
    w = _ensure_word(ci, sid, "引发")
    # 冷启动（无 feed）→ False
    assert not _experience_ok(b, w), "冷启动 e_sn/e_tn=0 → False"
    # feed 1 次（reward>0）→ e_sn=1 e_tn=1 total=2 < PROMOTE_EXP_FREQ_MIN=3 → False
    record_experience_outcome(b, ref=w, reward=1)
    assert not _experience_ok(b, w), f"total=2 < {PROMOTE_EXP_FREQ_MIN} → False"
    # 第 2 次 → e_sn=2 e_tn=2 total=4 ≥ 3 ∧ rate=2/4=1/2 → True
    record_experience_outcome(b, ref=w, reward=1)
    assert _experience_ok(b, w), "total=4 ≥ 3 ∧ rate=1/2 → True"


def test_experience_ok_low_success_rate(emerg_env):
    """e_sn/(e_tn) < 1/2（多失败）→ False（守 success_rate 闸·防全 0 假晋）。"""
    b, sid, es, ns, ci = emerg_env
    w = _ensure_word(ci, sid, "引发")
    # 1 成功 3 失败 → e_sn=1 e_tn=4 → 1/4 < 1/2 → False
    record_experience_outcome(b, ref=w, reward=1)
    for _ in range(3):
        record_experience_outcome(b, ref=w, reward=0)
    got = read_experience_count(b, w)
    assert got == (0, 1, 4), f"e_sn=1 e_tn=4·got {got}"
    assert not _experience_ok(b, w), "e_sn/e_tn=1/4 < 1/2 → False"


def test_definition_ok_d11_teacher_path(emerg_env):
    """_definition_ok_d11 teacher 加分：teacher.confirm_causes → True（断奶前·软 ∨ 项）。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    w = _ensure_word(ci, sid, "引发")
    record_emergent_relation_signal_shadow(es, w, rel_refs[REL_CAUSES], space_id=sid)
    ref = (w[0], w[1], rel_refs[REL_CAUSES][0], rel_refs[REL_CAUSES][1], EDGE_RELATION_SIGNAL)

    class _StubTeacher:
        def confirm_causes(self, a, b):
            return a == w
    assert _definition_ok_d11(es, ref, _StubTeacher()), "teacher.confirm_causes → True（加分项）"
    assert not _definition_ok_d11(es, ref, None), "无 teacher + 无同词 PRIMARY 种子 → False"


def test_definition_ok_d11_seed_anchor(emerg_env):
    """_definition_ok_d11 同词种子锚：from 节点已有 PRIMARY D:11 边（同词教师种子）→ True。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    w = _ensure_word(ci, sid, "引发")
    # 同词教师 PRIMARY 种子（REL_SUBSET）
    record_word_concept(ci, es, "引发", rel_refs[REL_SUBSET], space_id=sid)
    # 涌现 SHADOW（REL_CAUSES）—— 同词不同 rel
    record_emergent_relation_signal_shadow(es, w, rel_refs[REL_CAUSES], space_id=sid)
    ref = (w[0], w[1], rel_refs[REL_CAUSES][0], rel_refs[REL_CAUSES][1], EDGE_RELATION_SIGNAL)
    assert _definition_ok_d11(es, ref, None), "同词已有 PRIMARY D:11 种子 → True（种子锚加分）"


# ============ unit：promote_edge D:11 双轨（experience 主导 + teacher 加分） ============

def test_promote_edge_d11_experience_path(emerg_env):
    """D:11 promote experience 主导：SHADOW + e_sn/e_tn 达标 → flip PRIMARY（无教师·初心对齐）。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    w = _ensure_word(ci, sid, "引发")
    record_emergent_relation_signal_shadow(es, w, rel_refs[REL_CAUSES], space_id=sid)
    ref = (w[0], w[1], rel_refs[REL_CAUSES][0], rel_refs[REL_CAUSES][1], EDGE_RELATION_SIGNAL)
    # 未 feed → promote 失败
    assert not promote_edge(es, ns, ref, backend=b), "未 feed e_sn=0 → promote 失败"
    # feed 3 次 reward>0
    for _ in range(3):
        record_experience_outcome(b, ref=w, reward=1)
    assert promote_edge(es, ns, ref, backend=b), "experience 达标 → promote 成功（无教师·experience 主导）"
    e = es.get(space_id_from=ref[0], local_id_from=ref[1],
               space_id_to=ref[2], local_id_to=ref[3], edge_type=EDGE_RELATION_SIGNAL)
    assert e["tier"] == TIER_PRIMARY, "D:11 边 tier=PRIMARY（flip）"


def test_promote_edge_d11_no_backend_no_promote(emerg_env):
    """backend None → D:11 不 promote（safe degradation·D:11 验证须 backend 读 experience_count）。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    w = _ensure_word(ci, sid, "引发")
    record_emergent_relation_signal_shadow(es, w, rel_refs[REL_CAUSES], space_id=sid)
    ref = (w[0], w[1], rel_refs[REL_CAUSES][0], rel_refs[REL_CAUSES][1], EDGE_RELATION_SIGNAL)
    for _ in range(3):
        record_experience_outcome(b, ref=w, reward=1)
    assert not promote_edge(es, ns, ref), "backend=None → 不 promote（safe degradation）"


def test_promote_edge_d11_teacher_bonus_path(emerg_env):
    """D:11 promote teacher 加分：teacher.confirm_causes → promote（experience 未达标也可·断奶前稳）。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    w = _ensure_word(ci, sid, "引发")
    record_emergent_relation_signal_shadow(es, w, rel_refs[REL_CAUSES], space_id=sid)
    ref = (w[0], w[1], rel_refs[REL_CAUSES][0], rel_refs[REL_CAUSES][1], EDGE_RELATION_SIGNAL)

    class _StubTeacher:
        def confirm_causes(self, a, b):
            return a == w
    # experience 未 feed（e_sn=0）但 teacher 确认 → promote（双轨 ∨ teacher 加分）
    assert promote_edge(es, ns, ref, teacher=_StubTeacher(), backend=b), \
        "teacher 加分 ∨ → promote（experience 未达·断奶前稳）"


def test_promote_report_d11_diagnosis(emerg_env):
    """promote_report D:11 路径：返 experience + definition 诊断键·eligible 与 promote_edge 一致。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    w = _ensure_word(ci, sid, "引发")
    record_emergent_relation_signal_shadow(es, w, rel_refs[REL_CAUSES], space_id=sid)
    ref = (w[0], w[1], rel_refs[REL_CAUSES][0], rel_refs[REL_CAUSES][1], EDGE_RELATION_SIGNAL)
    rep = promote_report(es, ref, backend=b)
    assert rep["eligible"] is False, "未 feed → not eligible"
    assert rep["experience"] is False, "experience 诊断键"
    assert rep["freq"] is False and rep["reward"] is False, "D:11 N/A freq/reward"
    for _ in range(3):
        record_experience_outcome(b, ref=w, reward=1)
    rep2 = promote_report(es, ref, backend=b)
    assert rep2["eligible"] is True and rep2["experience"] is True, "feed 后 eligible + experience 达"


# ============ unit：cue_type_of D:11 readback（决断5 反 theater 关键） ============

def test_cue_type_of_readback_gate_off_degrades(emerg_env):
    """gate OFF → cue_type_of 只走 frozenset·退化·'引发' 返 None（bit-identical 守回归）。"""
    b, sid, es, ns, ci = emerg_env
    saved = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = False
    try:
        # '导致' 在 frozenset → 命中（第一源）
        assert cue_type_of("导致", LANG_ZH) == CAUSES_CUE_FORWARD, "'导致' frozenset 命中"
        # '引发' 不在 frozenset → None（gate OFF 不读 D:11）
        assert cue_type_of("引发", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) is None, \
            "gate OFF → '引发' 退化 None（不读 D:11）"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved


def test_cue_type_of_readback_cold_start_none(emerg_env):
    """gate ON + 冷启动（'引发' 未概念化/无 D:11 PRIMARY）→ None。"""
    b, sid, es, ns, ci = emerg_env
    saved = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    try:
        # '引发' 未概念化 → None
        assert cue_type_of("引发", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) is None, "未概念化 → None"
        # 概念化但无 D:11 PRIMARY 边 → None
        w = _ensure_word(ci, sid, "引发")
        assert cue_type_of("引发", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) is None, "无 D:11 PRIMARY → None"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved


def test_cue_type_of_readback_shadow_not_injected(emerg_env):
    """反 theater 牙：D:11 SHADOW（未 promote）→ readback 返 None（未验证不注入·tier_filter=PRIMARY）。"""
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    w = _ensure_word(ci, sid, "引发")
    record_emergent_relation_signal_shadow(es, w, rel_refs[REL_CAUSES], space_id=sid)  # SHADOW
    saved = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    try:
        assert cue_type_of("引发", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) is None, \
            "D:11 SHADOW 未 promote → readback None（未验证不注入·反 theater）"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved


# ============ integration：刀4 反 theater 完整闭环（手动驱动各步·证机制端到端） ============

def test_knife4_emergence_full_loop(emerg_env):
    """刀4 反 theater 完整闭环（手动驱动各步·证机制端到端）：
    建图（PRECEDES/COOCCURS）→ 生成器涌'引发'→REL_CAUSES → SHADOW 落边 →
    反 theater（readback 仍 None）→ experience feed → promote PRIMARY → readback 返非 None。

    诚实边界：experience feed 用 record_experience_outcome 手动注入（模拟 reward>0 episode）·
    非真 formal_train reward 通路（构造性·#479 边界）·真 formal_train 通路见 test_knife4_formal_train_smoke。
    """
    b, sid, es, ns, ci = emerg_env
    rel_refs = ensure_relation_primitives(ci, b, space_id=sid)
    雨, 引发, 洪水 = _build_x_w_y_pattern(es, ci, sid, "雨", "引发", "洪水", 3)

    saved_readback = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    try:
        # 阶段A 冷启动：cue_type_of('引发') = None
        assert cue_type_of("引发", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) is None, \
            "阶段A 冷启动 cue_type_of('引发') = None"
        # 生成器涌'引发'（excluded 不含引发）
        hyps = generate_emergent_hypotheses(b, es, ci, space_id=sid, excluded_word_refs=set())
        assert len(hyps) == 1 and hyps[0][0] == 引发, "生成器涌 '引发'→REL_CAUSES"
        # SHADOW 落边
        n = record_emergent_relation_signal_shadow(es, 引发, rel_refs[REL_CAUSES], space_id=sid)
        assert n == 1, "D:11 SHADOW 边落"
        # 反 theater：未 promote 前 readback 仍 None
        assert cue_type_of("引发", LANG_ZH, backend=b, edge_store=es,
                           space_id=sid, concept_index=ci) is None, \
            "SHADOW 未 promote → readback None（反 theater·未验证不注入）"
        # 阶段B experience feed（模拟 reward>0 episode ×3）
        for _ in range(3):
            record_experience_outcome(b, ref=引发, reward=1)
        # 阶段C promote（experience 主导·无教师）
        ref = (引发[0], 引发[1], rel_refs[REL_CAUSES][0], rel_refs[REL_CAUSES][1], EDGE_RELATION_SIGNAL)
        assert promote_edge(es, ns, ref, backend=b), "experience 达标 → promote PRIMARY"
        # 阶段D 反 theater 验收：promote 后 cue_type_of('引发') 返非 None
        ct = cue_type_of("引发", LANG_ZH, backend=b, edge_store=es,
                         space_id=sid, concept_index=ci)
        assert ct == CAUSES_CUE_FORWARD, \
            "阶段D promote 后 cue_type_of('引发') = CAUSES_CUE_FORWARD（涌现学习得证）"
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved_readback


# ============ e2e：formal_train 生产通路（涌现钩子真跑·D:11 SHADOW 边产消者激活） ============

def test_knife4_formal_train_emergence_hook_runs(tmp_path, monkeypatch):
    """e2e smoke：run_round_full reward 阶段（stage3+·HYPOTHESIS_MODE ON）涌现钩子真跑 →
    D:11 SHADOW 边（source=SOURCE_BARE_TEXT）产消者激活（刀4 前 D:11 仅教师 PRIMARY 种子·刀4 后 BARE_TEXT SHADOW 边非空）。

    直调 run_round_full（绕 metric gate·formal_train 全 main 流程受 stage1 metric 阻 stage3·smoke 直测 hook）。
    诚实边界：reward/promote 通路由 test_knife4_emergence_full_loop 覆盖·本测只验钩子在 reward 路径真跑。
    """
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    from pure_integer_ai.experiments.formal_train import make_train_context, DefaultRoundRunner
    from pure_integer_ai.training.stages import STAGE3_REWARD
    b = DictBackend()
    ctx = make_train_context(b)
    runner = DefaultRoundRunner()
    saved_hyp = gates.EMERGENT_RELATION_HYPOTHESIS_MODE
    gates.EMERGENT_RELATION_HYPOTHESIS_MODE = True   # 模拟 formal_train 生产入口翻 gate
    try:
        # 3× run_round_full（每 item observe 一段 → 第 3 次 COOCCURS(雨,洪水)=3 ≥ MIN → 钩子涌"引发"）
        for i in range(3):
            runner.run_round(ctx, _lang_item(["雨", "引发", "洪水"]), STAGE3_REWARD, i)
    finally:
        gates.EMERGENT_RELATION_HYPOTHESIS_MODE = saved_hyp
    # 涌现钩子跑 → D:11 SHADOW 边（source=BARE_TEXT）落盘（"引发"→REL_CAUSES）
    d11_bare = [r for r in b.select("edge", where={"edge_type": EDGE_RELATION_SIGNAL})
                if r["source"] == SOURCE_BARE_TEXT]
    assert len(d11_bare) > 0, \
        "刀4 涌现钩子跑 → D:11 SHADOW 边（BARE_TEXT）产消者激活（反 theater·当前刀4 前零 BARE_TEXT 产消者）"
    assert all(r["tier"] == TIER_SHADOW for r in d11_bare), "涌现 D:11 边 tier=SHADOW（staging）"
    assert all(r["epistemic_origin"] is None for r in d11_bare), "epistemic_origin=None（伴随检疫）"


def test_knife4_formal_train_gates_restored(tmp_path, monkeypatch):
    """bit-identical：formal_train finally 守 HYPOTHESIS/FEED gate 回归 OFF（CI/生产 default）。
    直调 formal_train 全 main（即使 metric gate 阻 stage3·finally 仍守 gate 回归）。"""
    monkeypatch.delenv("PURE_INTEGER_AI_LOCAL_DIR", raising=False)
    from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig, DefaultRoundRunner
    corpus = [_lang_item(["雨", "引发", "洪水"]) for _ in range(3)]
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="knife4_gate_restore", rounds_per_stage=1)
    assert gates.EMERGENT_RELATION_HYPOTHESIS_MODE is False, "默认 OFF"
    assert gates.EMERGENT_RELATION_FEED_MODE is False, "默认 OFF"
    formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())
    assert gates.EMERGENT_RELATION_HYPOTHESIS_MODE is False, "finally 回归 OFF"
    assert gates.EMERGENT_RELATION_FEED_MODE is False, "finally 回归 OFF"
