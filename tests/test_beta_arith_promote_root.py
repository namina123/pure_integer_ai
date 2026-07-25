"""审计根治 [严重-2] β_arith 根治测试（PR2·路径1 + 生成侧 cap）。

β_arith 病：reward>0 episode 同比 e_sn++/e_tn++（或 sp_sn++/sp_tn++）→ success_rate = N/(N+N) = 1/2
恰达标 promote 闸（或生成侧 score 塌缩）·promote 不该 promote 的 D:11 SHADOW 边。

PR2 根治（路径1·不违 promote 三重②）：
  - experience_count.read_experience_count 加 observe_mode 参（observe_tn 替 e_tn）
  - promote._experience_ok 内部读 gates.FREQ_OBSERVE_MODE（total = e_sn + observe_tn·rate 不恒 1/2·缓解 β_arith）
    _experience_ok 是 D:11 专用闸（非三重② _reward_ok 边级）·路径1 不违三重②
  - graph_view.selection_pref_score gate SP_OBSERVE_MODE ON 时 cap min(sp_sn, observe_tn) + observe_tn
    （守 sp_sn ≤ observe_tn 子集防膨胀·sp_observe_tn sign-agnostic 不同比 sp_sn 增）

**核心反 β_arith 测**：reward>0 episode 同比 e_sn++/e_tn++（β_arith）但 observe_tn 独立（决策时写·非 episode）→
gate OFF rate=1/2 恰达标（β_arith 病）·gate ON rate<1/2 不达标（β_arith 缓解）。

**诚实边界**：β_arith 缓解非根治·e_sn 仍 reward-feed 染·rate 缓解非恒 1/2（gate OFF 退化既有 bit-identical）。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_TEACHER, EPI_STRUCTURED
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY
from pure_integer_ai.storage.spaces.registry import SpaceRegistry
from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
from pure_integer_ai.storage.experience_count import (
    register_experience_count, read_experience_count,
    record_experience_outcome, record_experience_observe,
)
from pure_integer_ai.storage.selection_pref_count import (
    register_selection_pref_count, record_selection_pref_reward,
    record_selection_pref_cooccur,
)
from pure_integer_ai.storage.edge_types import EDGE_IS_A
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.training.promote import _experience_ok, PROMOTE_EXP_FREQ_MIN
from pure_integer_ai.config import gates


# ---- fixtures ----

@pytest.fixture
def beta_env():
    """PR2 单测环境（dict backend·core space·experience_count + selection_pref_count 注册）。"""
    b = DictBackend()
    bootstrap(b)
    register_experience_count(b)
    register_selection_pref_count(b)
    reg = SpaceRegistry(b)
    sp = AbstractSpace.create(reg, "core")
    es = EdgeStore(b)
    ns = NodeStore(b)
    ci = ConceptIndex(b)
    sid = sp.space_id
    yield b, sid, es, ns, ci
    b.close()


# ============ read_experience_count observe_mode ============

def test_read_experience_count_observe_mode_switches_tn(beta_env):
    """read_experience_count 加 observe_mode 参：False 返 e_tn·True 返 observe_tn（base_freq/e_sn 不变）。"""
    b, sid, es, ns, ci = beta_env
    ref = ci.ensure("概念", space_id=sid, tier=TIER_PRIMARY)
    # reward>0 episode × 3 → e_sn=3, e_tn=3（β_arith 同比）
    for _ in range(3):
        record_experience_outcome(b, ref=ref, reward=1)
    # observe 决策 × 5 → observe_tn=5（独立·非 episode）
    for _ in range(5):
        record_experience_observe(b, ref=ref)
    # observe_mode=False → e_tn=3
    got_off = read_experience_count(b, ref, observe_mode=False)
    assert got_off == (0, 3, 3), "observe_mode=False 返 (base=0, e_sn=3, e_tn=3)"
    # observe_mode=True → observe_tn=5（e_sn 不变=3）
    got_on = read_experience_count(b, ref, observe_mode=True)
    assert got_on == (0, 3, 5), "observe_mode=True 返 (base=0, e_sn=3, observe_tn=5)"
    # 默认 observe_mode=False（bit-identical·既有 caller 不传参退化）
    got_default = read_experience_count(b, ref)
    assert got_default == (0, 3, 3), "默认 observe_mode=False（既有 bit-identical）"


# ============ _experience_ok 反 β_arith（核心） ============

def test_experience_ok_gate_off_beta_arith_passes(beta_env):
    """β_arith 病重现：gate OFF·reward>0 episode 同比 e_sn++/e_tn++ → rate=1/2 恰达标 promote 闸。
    e_sn=3, e_tn=3 → total=6 ≥ PROMOTE_EXP_FREQ_MIN=3·rate=3/6=1/2 → _experience_ok=True（β_arith 病）。"""
    b, sid, es, ns, ci = beta_env
    ref = ci.ensure("概念", space_id=sid, tier=TIER_PRIMARY)
    for _ in range(3):
        record_experience_outcome(b, ref=ref, reward=1)   # e_sn=3, e_tn=3
    saved = gates.FREQ_OBSERVE_MODE
    gates.FREQ_OBSERVE_MODE = False
    try:
        assert _experience_ok(b, ref) is True, \
            "gate OFF β_arith：rate=3/6=1/2 恰达标（β_arith 病·promote 不该 promote 的）"
    finally:
        gates.FREQ_OBSERVE_MODE = saved


def test_experience_ok_observe_mode_avoids_beta_arith(beta_env):
    """★核心反 β_arith：gate ON·observe_tn 独立（决策时写·5 次 > reward>0 episode 3 次）→
    rate = e_sn/(e_sn+observe_tn) = 3/8 < 1/2 → _experience_ok=False（β_arith 缓解·不 promote）。
    observe_tn 不同比 e_sn 增（决策频繁的 concept observe_tn 高·rate 降）·rate 不恒 1/2。"""
    b, sid, es, ns, ci = beta_env
    ref = ci.ensure("概念", space_id=sid, tier=TIER_PRIMARY)
    for _ in range(3):
        record_experience_outcome(b, ref=ref, reward=1)   # e_sn=3, e_tn=3（β_arith 同比）
    for _ in range(5):
        record_experience_observe(b, ref=ref)             # observe_tn=5（独立·决策多于 reward>0 episode）
    saved = gates.FREQ_OBSERVE_MODE
    gates.FREQ_OBSERVE_MODE = True
    try:
        assert _experience_ok(b, ref) is False, \
            "gate ON β_arith 缓解：rate=3/8<1/2 不达标（observe_tn=5 独立·rate 不恒 1/2）"
    finally:
        gates.FREQ_OBSERVE_MODE = saved


def test_experience_ok_observe_mode_passes_when_observe_comparable(beta_env):
    """gate ON observe_tn 与 e_sn 相当时仍通过：e_sn=4, observe_tn=4 → rate=4/8=1/2 达标
    （observe_tn 与 e_sn 相当·非 β_arith 膨胀·机制描述非判定合法性）。"""
    b, sid, es, ns, ci = beta_env
    ref = ci.ensure("概念", space_id=sid, tier=TIER_PRIMARY)
    for _ in range(4):
        record_experience_outcome(b, ref=ref, reward=1)   # e_sn=4
    for _ in range(4):
        record_experience_observe(b, ref=ref)             # observe_tn=4
    saved = gates.FREQ_OBSERVE_MODE
    gates.FREQ_OBSERVE_MODE = True
    try:
        assert _experience_ok(b, ref) is True, \
            "gate ON observe_tn 与 e_sn 相当：rate=4/8=1/2 达标（非 β_arith 膨胀）"
    finally:
        gates.FREQ_OBSERVE_MODE = saved


def test_experience_ok_observe_tn_zero_gate_on_strict(beta_env):
    """observe_tn=0 假阳性守卫（对抗审 catch·已修）：concept 未在 FREQ_OBSERVE_MODE ON 期 path-reach·observe_tn=0 但
    e_sn>0 → 未修时 rate=e_sn/e_sn=1.0 假阳性 promote（gate ON 比 gate OFF 更宽松·反缓解）。
    修后：tn==0 守卫退 False（对称 SP cap min(sp_sn,0)+0=0 天然防护·_experience_ok 显式守卫·防假阳性 promote）。"""
    b, sid, es, ns, ci = beta_env
    ref = ci.ensure("概念", space_id=sid, tier=TIER_PRIMARY)
    for _ in range(3):
        record_experience_outcome(b, ref=ref, reward=1)   # e_sn=3, e_tn=3, observe_tn=0（未 observe）
    saved = gates.FREQ_OBSERVE_MODE
    gates.FREQ_OBSERVE_MODE = True
    try:
        # observe_tn=0 守卫：gate ON tn==0 退 False（防假阳性 promote·对称 SP cap）
        assert _experience_ok(b, ref) is False, \
            "observe_tn=0 守卫：gate ON tn==0 退 False（对称 SP cap·防假阳性 promote）"
    finally:
        gates.FREQ_OBSERVE_MODE = saved


def test_experience_ok_observe_tn_less_than_e_sn_boundary(beta_env):
    """observe_tn < e_sn 边界（对抗审 catch·设计诚实边界）：gate ON 时 rate=e_sn/(e_sn+observe_tn) > 1/2·
    反放宽（gate ON 比 gate OFF 更宽松）。β_arith 缓解依赖 observe_tn > e_sn 假设（决策多于 reward>0 episode）·
    observe_tn < e_sn 时（concept_targets 扩展非 path 节点）mitigation 退化·诚实标注非根治。"""
    b, sid, es, ns, ci = beta_env
    ref = ci.ensure("概念", space_id=sid, tier=TIER_PRIMARY)
    for _ in range(5):
        record_experience_outcome(b, ref=ref, reward=1)  # e_sn=5, e_tn=5
    for _ in range(2):
        record_experience_observe(b, ref=ref)            # observe_tn=2 < e_sn=5
    saved = gates.FREQ_OBSERVE_MODE
    # gate OFF: rate=5/10=1/2 达标（β_arith 病）
    gates.FREQ_OBSERVE_MODE = False
    try:
        assert _experience_ok(b, ref) is True, "gate OFF rate=5/10=1/2 达标（β_arith 病）"
    finally:
        gates.FREQ_OBSERVE_MODE = saved
    # gate ON: rate=5/7≈0.71 > 1/2 达标（observe_tn<e_sn 反放宽·mitigation 退化·诚实边界非根治）
    gates.FREQ_OBSERVE_MODE = True
    try:
        assert _experience_ok(b, ref) is True, \
            "gate ON observe_tn=2<e_sn=5 rate=5/7>1/2 达标（反放宽·mitigation 退化·诚实边界非根治）"
    finally:
        gates.FREQ_OBSERVE_MODE = saved


def test_experience_ok_gate_off_bit_identical(beta_env):
    """bit-identical：gate OFF _experience_ok 退化既有（e_sn/(e_sn+e_tn)·既有行为）。"""
    b, sid, es, ns, ci = beta_env
    ref = ci.ensure("概念", space_id=sid, tier=TIER_PRIMARY)
    record_experience_outcome(b, ref=ref, reward=1)
    record_experience_outcome(b, ref=ref, reward=0)   # e_sn=1, e_tn=2
    saved = gates.FREQ_OBSERVE_MODE
    gates.FREQ_OBSERVE_MODE = False
    try:
        # e_sn=1, e_tn=2 → total=3 ≥ 3·rate=1/3 < 1/2 → False（既有行为）
        assert _experience_ok(b, ref) is False, "gate OFF 既有：rate=1/3<1/2 不达标"
    finally:
        gates.FREQ_OBSERVE_MODE = saved


# ============ selection_pref_score cap sp_sn ============

def test_selection_pref_score_cap_sp_sn(beta_env):
    """生成侧 cap：gate SP_OBSERVE_MODE ON 时 score = min(sp_sn, observe_tn) + observe_tn（cap sp_sn ≤ observe_tn 防膨胀）。
    β_arith 场景：sp_sn=5, sp_tn=5（5 reward>0 episode 同比）·sp_observe_tn=2（2 决策·独立）。
    gate OFF: score = sp_sn + sp_tn = 10（β_arith 膨胀）。
    gate ON: score = min(5,2) + 2 = 4（cap·防 sp_sn 膨胀超 observe_tn 子集）。"""
    b, sid, es, ns, ci = beta_env
    猫 = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY)
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY)
    r = ci.ensure("语境词", space_id=sid, tier=TIER_PRIMARY)
    # 建 IS_A 边 猫→动物（selection_pref_score 读 class_of(猫)=nearest_isa_ancestor=动物）
    es.add(space_id_from=猫[0], local_id_from=猫[1],
           space_id_to=动物[0], local_id_to=动物[1], edge_type=EDGE_IS_A,
           strength=1, source=SOURCE_TEACHER, epistemic_origin=EPI_STRUCTURED,
           tier=TIER_PRIMARY)
    # β_arith：5 reward>0 episode → sp_sn=5, sp_tn=5（同比膨胀）
    for _ in range(5):
        record_selection_pref_reward(b, ref_a=r, ref_class=动物, reward=1)
    # observe：2 决策 → sp_observe_tn=2（独立·非 reward episode）
    for _ in range(2):
        record_selection_pref_cooccur(b, ref_a=r, ref_class=动物)
    graph = ConceptGraph(b)
    graph.invalidate_ancestor_map(sid)   # 清 cache·确保 build 含新 IS_A
    # gate OFF: score = sp_sn + sp_tn = 5 + 7 = 12（sp_sn=5·sp_tn=5reward+2cooccur=7·β_arith 膨胀）
    saved = gates.SP_OBSERVE_MODE
    gates.SP_OBSERVE_MODE = False
    try:
        score_off = graph.selection_pref_score(猫, [r])
        assert score_off == 12, f"gate OFF 既有 sp_sn+sp_tn=12（sp_sn=5+sp_tn=7=5reward+2cooccur·β_arith 膨胀）·got {score_off}"
    finally:
        gates.SP_OBSERVE_MODE = saved
    # gate ON: score = min(sp_sn, observe_tn) + observe_tn = min(5,2) + 2 = 4（cap 防膨胀）
    gates.SP_OBSERVE_MODE = True
    try:
        score_on = graph.selection_pref_score(猫, [r])
        assert score_on == 4, f"gate ON cap min(5,2)+2=4（cap sp_sn ≤ observe_tn 防膨胀）·got {score_on}"
    finally:
        gates.SP_OBSERVE_MODE = saved


def test_selection_pref_score_cap_no_harm_when_balanced(beta_env):
    """cap 无害：sp_sn ≤ observe_tn 时 cap = sp_sn（min 不截断·保留成功加成）。
    sp_sn=2, sp_observe_tn=5 → gate ON score = min(2,5)+5 = 7（cap 不截断·sp_sn ≤ observe_tn 子集正常）。"""
    b, sid, es, ns, ci = beta_env
    猫 = ci.ensure("猫", space_id=sid, tier=TIER_PRIMARY)
    动物 = ci.ensure("动物", space_id=sid, tier=TIER_PRIMARY)
    r = ci.ensure("语境词", space_id=sid, tier=TIER_PRIMARY)
    es.add(space_id_from=猫[0], local_id_from=猫[1],
           space_id_to=动物[0], local_id_to=动物[1], edge_type=EDGE_IS_A,
           strength=1, source=SOURCE_TEACHER, epistemic_origin=EPI_STRUCTURED,
           tier=TIER_PRIMARY)
    # sp_sn=2, sp_tn=2（2 reward>0）·sp_observe_tn=5（5 决策·observe 多）
    for _ in range(2):
        record_selection_pref_reward(b, ref_a=r, ref_class=动物, reward=1)
    for _ in range(5):
        record_selection_pref_cooccur(b, ref_a=r, ref_class=动物)
    graph = ConceptGraph(b)
    graph.invalidate_ancestor_map(sid)
    saved = gates.SP_OBSERVE_MODE
    gates.SP_OBSERVE_MODE = True
    try:
        score_on = graph.selection_pref_score(猫, [r])
        # gate ON: min(sp_sn=2, observe_tn=5) + observe_tn=5 = 2 + 5 = 7（cap 不截断·sp_sn ≤ observe_tn）
        assert score_on == 7, f"cap 无害：min(2,5)+5=7（sp_sn ≤ observe_tn·cap 不截断）·got {score_on}"
    finally:
        gates.SP_OBSERVE_MODE = saved


# ============ promote 三重② 不违（_experience_ok 是 D:11 专用闸非 _reward_ok） ============

def test_experience_ok_is_d11_not_reward_ok():
    """守卫：_experience_ok 是 D:11 专用闸（D:11 边不接 reward 走 experience 对偶）·非 promote 三重② _reward_ok（边级 sn/(sn+tn)）。
    路径1 改 _experience_ok 不动 _reward_ok·不违 promote 三重② reward G5 硬条件。"""
    from pure_integer_ai.training.promote import _experience_ok, _reward_ok
    # _experience_ok 与 _reward_ok 是不同函数（D:11 专用 vs 边级三重②）
    assert _experience_ok is not _reward_ok, "_experience_ok（D:11 专用）≠ _reward_ok（三重② 边级）"
    # _experience_ok 读 experience_count（D:11 对偶）·_reward_ok 读 edge row sn/tn（边级）
    import inspect
    exp_src = inspect.getsource(_experience_ok)
    assert "read_experience_count" in exp_src, "_experience_ok 读 experience_count（D:11 概念维对偶）"
    assert "FREQ_OBSERVE_MODE" in exp_src, "_experience_ok 内部读 gates.FREQ_OBSERVE_MODE（审计根治 [严重-2]）"
