"""tests.test_statistical_weaning — 语言域统计层断奶判定（#1143·5判据 formal gate·非 can_ween）。

验 language_statistical_weaning_check 反 theater 5 锚点（plateau/floor/withhold D2/fadeout/held-out）+
2 前置 + measured-guards（防 stub-0 vacuous 过·2审 HIGH-1/3）+ D2 硬 gate（防 permissive·HIGH-2）。
复用 weaning_check D1 机器·独立 verdict。

铁律：纯整数 / bit-identical（纯函数）/ 反 theater（5 锚点 veto·measured-guard·excluded_gates 复述）。
"""
from __future__ import annotations

from pure_integer_ai.teacher.weaning import (
    WeaningMetrics,
    language_statistical_weaning_check,
    StatisticalWeaningReport,
    WEANING_WINDOW_ROUNDS,
    LANG_HOLDOUT_FLOOR,
    METRIC_CONDUCTION,
)


def _wm(rounds: int, *, conduction: int = 800, realizes: int = 700,
        judge_self: int = 800, oov: int = 200,
        intervention: int = 100, retention: int = 900, dependency: int = 100) -> WeaningMetrics:
    """构造单 round WeaningMetrics（默认 D1-passing 口径·语言域校准值）。"""
    return WeaningMetrics(
        rounds=rounds, conduction_rate=conduction, realizes_rate=realizes,
        judge_self_rate=judge_self, oov_promote_rate=oov,
        intervention_rate=intervention, holdout_retention=retention,
        dependency=dependency,
    )


def _passing_history() -> list[WeaningMetrics]:
    """4-round D1-passing history（plateau + floor + intervention↓ + retention + dependency low）。"""
    return [_wm(1, intervention=100), _wm(2, intervention=90),
            _wm(3, intervention=80), _wm(4, intervention=70)]


def _check(history=None, **kw) -> StatisticalWeaningReport:
    """默认全过调用（5 锚点 + 2 前置 + measured 全 True）·kw 覆盖单项验 veto。"""
    defaults = dict(
        encoding_grounded=True, crosslingual_seeded=True, probe_set_disjoint=True,
        neg_pathway_active=True,            # 锚点3 D2 硬 gate
        teacher_present=True,               # 锚点4 教师在场（测 measured fadeout 路径）
        fadeout_measured=True,              # 锚点4 measured-guard
        heldout_measured=True,              # 锚点5 measured-guard
        heldout_generalization_permille=700,  # ≥ LANG_HOLDOUT_FLOOR=500
    )
    defaults.update(kw)
    return language_statistical_weaning_check(
        history if history is not None else _passing_history(), **defaults)


# SW1: empty history → False（不足窗口·诚实）
def test_sw1_empty_history_false():
    rep = language_statistical_weaning_check(
        [], encoding_grounded=True, crosslingual_seeded=True, probe_set_disjoint=True,
        neg_pathway_active=True, fadeout_measured=True, heldout_measured=True,
        heldout_generalization_permille=700)
    assert rep.statistical_ready is False
    assert rep.enough_window is False
    assert isinstance(rep, StatisticalWeaningReport)


# SW2: full pass → True（5 锚点 + 2 前置 + measured 全过）
def test_sw2_full_pass_true():
    rep = _check()
    assert rep.statistical_ready is True
    assert rep.enough_window is True
    assert all(rep.plateaued.values())
    assert rep.floors_met is True
    assert rep.intervention_decreasing is True
    assert rep.dependency_low is True


# SW3: plateau fail（once-met/仍在升·反 theater 锚点1）→ False
def test_sw3_plateau_fail_false():
    hist = [_wm(1, conduction=800, intervention=100), _wm(2, conduction=850, intervention=90),
            _wm(3, conduction=900, intervention=80), _wm(4, conduction=950, intervention=70)]
    assert _check(hist).statistical_ready is False
    assert _check(hist).plateaued[METRIC_CONDUCTION] is False


# SW4: floor fail（全-0 假平台·反 theater 锚点2）→ False
def test_sw4_floor_fail_false():
    hist = [_wm(r, conduction=10, realizes=10, judge_self=10, oov=0) for r in range(1, 5)]
    assert _check(hist).statistical_ready is False
    assert _check(hist).floors_met is False


# SW5: D2 fail（permissive-degenerate·无拒奖·反 theater 锚点3·2审 HIGH-2）→ False
def test_sw5_d2_permissive_fail_false():
    assert _check(neg_pathway_active=False).statistical_ready is False


# SW6: fadeout unmeasured（stub-0 vacuous·反 theater 锚点4·2审 HIGH-1）→ False
# 即便 history 的 intervention 系列看似下降·fadeout_measured=False → 不过（防未建测量 vacuous 过）
def test_sw6_fadeout_unmeasured_fail_false():
    assert _check(fadeout_measured=False).statistical_ready is False


# SW7: fadeout fail（intervention 升·即便 measured·反 theater 锚点4）→ False
def test_sw7_fadeout_rising_fail_false():
    hist = [_wm(1, intervention=50), _wm(2, intervention=80),
            _wm(3, intervention=100), _wm(4, intervention=120)]
    rep = _check(hist, fadeout_measured=True)
    assert rep.statistical_ready is False
    assert rep.intervention_decreasing is False


# SW8: held-out unmeasured（arith-only stub·反 theater 锚点5·2审 HIGH-3）→ False
def test_sw8_heldout_unmeasured_fail_false():
    assert _check(heldout_measured=False).statistical_ready is False


# SW9: held-out below floor（泛化不足·反 theater 锚点5）→ False
def test_sw9_heldout_below_floor_fail_false():
    assert _check(heldout_generalization_permille=LANG_HOLDOUT_FLOOR - 1).statistical_ready is False
    assert _check(heldout_generalization_permille=LANG_HOLDOUT_FLOOR).statistical_ready is True


# SW10: held-out probe split fail（死记·D4）→ False
def test_sw10_probe_disjoint_fail_false():
    assert _check(probe_set_disjoint=False).statistical_ready is False


# SW11: 前置 fail（encoding/crosslingual）→ False
def test_sw11_precondition_fail_false():
    assert _check(encoding_grounded=False).statistical_ready is False
    assert _check(crosslingual_seeded=False).statistical_ready is False


# SW12: excluded_gates 明示（E2/D5/D3 复述·反 theater 防「can_ween 减腿」松动）
def test_sw12_excluded_gates_present():
    rep = _check()
    assert len(rep.excluded_gates) == 3
    joined = " ".join(rep.excluded_gates)
    assert "E2" in joined and "D5" in joined and "D3" in joined


# SW13: bit-identical（纯函数·同输入同输出·无 gate/env 依赖）
def test_sw13_pure_deterministic():
    r1 = _check()
    r2 = _check()
    assert r1.statistical_ready == r2.statistical_ready
    assert r1.plateaued == r2.plateaued
    assert r1.floors_met == r2.floors_met


# SW14: 不足窗口（<WEANING_WINDOW_ROUNDS）→ False（多轮稳定前置）
def test_sw14_insufficient_window_false():
    rep = _check(_passing_history()[:3])
    assert rep.statistical_ready is False
    assert rep.enough_window is False


# SW15: P1b formal_train 接线（gate STATISTICAL_WEANING_MODE ON → result 字段 populated·wiring 不崩）
# arith corpus（teacher=None·无 lang held-out）→ heldout_measured=False + fadeout_measured=False
# → statistical_ready=False（诚实·非 theater）·但 report 已 populated（wiring 通）。
def test_sw15_formal_train_hook_wired():
    from pure_integer_ai.config import gates
    from pure_integer_ai.experiments.run_weaning_train import run_weaning_arith
    saved = gates.STATISTICAL_WEANING_MODE
    gates.STATISTICAL_WEANING_MODE = True
    try:
        result = run_weaning_arith(rounds_per_stage=2, training_mode=True, flat_floors=True)
        assert result.statistical_weaning_ready is False   # arith 无 lang held-out + fadeout 未建测量
        assert result.statistical_weaning_report is not None
        assert result.statistical_weaning_report.fadeout_measured is False   # P2 未建 intervention 聚合
    finally:
        gates.STATISTICAL_WEANING_MODE = saved


# SW16: 无教师（teacher_present=False）→ fadeout 锚结构性满足（无教师=无依赖·平行 arith vm_proof 无教师自锚
# ·= 用户「断奶后自主学习」目标：从语料学·无教师）·competence 由他锚（plateau/floor/D2/held-out）守。
def test_sw16_no_teacher_structural_independence():
    # 无教师 + 即便 fadeout_measured=False → anchor_fadeout=True（结构性独立·非 vacuous）
    rep = _check(teacher_present=False, fadeout_measured=False)
    assert rep.statistical_ready is True
    # 对照：有教师 + fadeout_measured=False → 不过（measured-guard·防 stub-0 vacuous·2审 HIGH-1）
    rep2 = _check(teacher_present=True, fadeout_measured=False)
    assert rep2.statistical_ready is False
