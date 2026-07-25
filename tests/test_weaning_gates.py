"""断奶闸门 D1-D5/E2 测试（#358 完整实现·六闸门·非布尔阈值）。

doc/重来_断奶闸门D1-D5_E2设计补充.md 权威设计。覆盖：
  D1 双规定曲线方向性（替 abs 对称病根）+ 依赖度
  D2 负通路活跃硬前置
  D3 source_id 裁判源独立 + 集合不相交
  D4 probe_set 隔离 + 版本化
  D5 预验台账 + calibration_set + Mode B 最小预验
  E2 教师下线独立产出骨架（执行条件未就位·永 False）

关键纠错验证：weaning.py 原 _max_recent_increment 用 abs() 对称平台（涨落都算非平台）违 D1 方向性·
是 weaning10/10 全 DEF_REPLAY 伪影病根。本测验证两规定曲线用交叉积方向性判（单调降∧无回升 / 后窗不回升）·
非 abs 对称。
"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.teacher.weaning import (
    weaning_check, WeaningMetrics,
    FLOOR_INTERVENTION, FLOOR_RETENTION, FLOOR_DEPENDENCY,
    _intervention_decreasing, _retention_stable,
)
from pure_integer_ai.teacher.source_independence import sources_disjoint
from pure_integer_ai.teacher.probe_set import ProbeSet, make_probe_set, is_disjoint
from pure_integer_ai.teacher.weaning_calibration import (
    register_weaning_calibration, record_calibration,
    false_pass_rate, mode_b_agreement, mode_b_prevalidated,
)
from pure_integer_ai.teacher.weaning_e2 import e2_independent_production, e2_execution_ready
from pure_integer_ai.teacher.recordable_teacher import RecordableLLMTeacher, MODE_OFF
from pure_integer_ai.training.stages import build_judge_fn
from pure_integer_ai.cognition.shared.types import JudgeWeights, WEANING_PRE


@pytest.fixture(params=["dict", "sqlite"])
def backend(request):
    b = DictBackend() if request.param == "dict" else SQLiteBackend(":memory:")
    bootstrap(b)
    register_weaning_calibration(b)
    return b


def _m(rounds, *, cond=0, real=0, judge=0, oov=0, interv=0, reten=0, dep=0):
    return WeaningMetrics(rounds=rounds, conduction_rate=cond, realizes_rate=real,
                          judge_self_rate=judge, oov_promote_rate=oov,
                          intervention_rate=interv, holdout_retention=reten,
                          dependency=dep)


def _gates_true():
    return dict(neg_pathway_active=True, judge_source_independent=True,
                probe_set_disjoint=True, mode_b_prevalidated=True, e2_passed=True)


# ============ D1 方向性（替 abs 对称病根） ============

def test_intervention_decreasing_monotone_drop_passes():
    """曲线① 单调降∧无回升∧降至阈值以下 → True。
    [500,400,300,200] 单调非增·latest=200≤FLOOR_INTERVENTION(200) → 通过。"""
    assert _intervention_decreasing([500, 400, 300, 200]) is True


def test_intervention_decreasing_rise_rejected():
    """曲线① 有回升 → False（方向性·替 abs 对称病根·abs 会把回升当非平台误挡健康降）。"""
    assert _intervention_decreasing([500, 400, 450, 200]) is False   # 450 回升


def test_intervention_decreasing_above_threshold_rejected():
    """曲线① 单调降但未降至阈值以下 → False（防高介入率平台假断奶）。"""
    assert _intervention_decreasing([800, 700, 600, 500]) is False   # latest 500 > 200


def test_intervention_decreasing_abs_symmetry_artifact():
    """病根验证：[500,200,200,200] 健康单调降——abs 对称会算 max|inc|=300 误判'非平台'挡断奶·
    方向性判正确通过（单调非增∧降至阈值以下）。此即 weaning10/10 全 DEF_REPLAY 伪影病根纠错。"""
    # abs 逻辑：max(|−300|,0,0)=300 ≥ THETA_PLATEAU → 误判非平台 → 误挡断奶
    # 方向性逻辑：单调非增 ∧ latest=200≤200 → 通过
    assert _intervention_decreasing([500, 200, 200, 200]) is True


def test_intervention_decreasing_insufficient_window():
    assert _intervention_decreasing([200, 200, 200]) is False   # < WEANING_WINDOW_ROUNDS(4)


def test_retention_stable_plateau_passes():
    """曲线② 后窗不回升∧达下限 → True。
    [800,750,720,710]：early sum=1550·late sum=1430·late≤early（不回升）∧ latest 710≥700 → 通过。"""
    assert _retention_stable([800, 750, 720, 710]) is True


def test_retention_stable_rising_rejected():
    """曲线② 后窗回升（仍在升=未稳态） → False。"""
    assert _retention_stable([700, 710, 750, 800]) is False   # late(1550) > early(1410)


def test_retention_stable_below_floor_rejected():
    """曲线② 不回升但保持率低于下限 → False（防低保持率平台假断奶）。"""
    assert _retention_stable([600, 590, 580, 570]) is False   # latest 570 < 700


def test_retention_stable_insufficient_window():
    assert _retention_stable([800, 800, 800]) is False


# ============ D1 依赖度 + weaning_check 集成 ============

def test_weaning_dependency_low_floor():
    """D1 依赖度 ≤ FLOOR_DEPENDENCY(300) → dependency_low。latest=120 → True。"""
    hist = [_m(1, dep=200), _m(2, dep=180), _m(3, dep=160), _m(4, dep=120)]
    rep = weaning_check(hist)
    assert rep.dependency_low is True


def test_weaning_dependency_high_blocks():
    """依赖度高于阈值 → dependency_low False → ready False（even if all gates true）。"""
    hist = [
        _m(1, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=400),
        _m(2, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=400),
        _m(3, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=400),
        _m(4, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=400),
    ]
    rep = weaning_check(hist, **_gates_true())
    assert rep.dependency_low is False
    assert rep.ready is False


def test_weaning_intervention_not_decreasing_blocks():
    """D1 曲线① 方向性未满足 → ready False（even if all gates true + 4 能力指标平台）。"""
    hist = [
        _m(1, cond=600, real=400, judge=600, oov=100, interv=300, reten=710, dep=100),
        _m(2, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=100),
        _m(3, cond=600, real=400, judge=600, oov=100, interv=250, reten=710, dep=100),  # 回升
        _m(4, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=100),
    ]
    rep = weaning_check(hist, **_gates_true())
    assert rep.intervention_decreasing is False
    assert rep.ready is False


# ============ D2 负通路活跃硬前置 ============

def test_d2_neg_pathway_active_hard_prerequisite():
    """D2·负通路不活跃 → ready False（even if D1 + D3-D5/E2 all true）。
    防 reward 永正趋平伪满足（负 reward=0 即塌非收敛）。"""
    hist = [
        _m(1, cond=600, real=400, judge=600, oov=100, interv=300, reten=800, dep=100),
        _m(2, cond=600, real=400, judge=600, oov=100, interv=200, reten=750, dep=100),
        _m(3, cond=600, real=400, judge=600, oov=100, interv=200, reten=720, dep=100),
        _m(4, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=100),
    ]
    gates = _gates_true()
    gates["neg_pathway_active"] = False
    rep = weaning_check(hist, **gates)
    assert rep.ready is False


# ============ D3 source_id 裁判源独立 ============

def test_sources_disjoint_basic():
    """D3·裁判源与训练教师源集合不相交。"""
    assert sources_disjoint({1, 2}, {3, 4}) is True
    assert sources_disjoint({1, 2}, {2, 3}) is False   # 2 同源
    assert sources_disjoint(set(), {1}) is True
    assert sources_disjoint({1}, set()) is True


def test_teacher_has_source_id():
    """RecordableLLMTeacher 带 source_id 字段（D3 贯通）。"""
    t = RecordableLLMTeacher(DictBackend(), mode=MODE_OFF, source_id=42)
    assert t.source_id == 42
    t0 = RecordableLLMTeacher(DictBackend(), mode=MODE_OFF)
    assert t0.source_id == 0   # 默认 0


def test_build_judge_fn_judge_equals_teacher_blocks_d3():
    """D3·当前裁判=训练教师本尊（build_judge_fn 绑 teacher.judge_ground_truth）→
    同源→sources_disjoint False→judge_source_independent False（诚实挡假断奶·独立裁判分离待工程）。"""
    b = DictBackend()
    bootstrap(b)
    from pure_integer_ai.teacher.recordable_teacher import register_recording_table
    register_recording_table(b)
    teacher = RecordableLLMTeacher(b, mode=MODE_OFF, source_id=7)
    jf = build_judge_fn(None, JudgeWeights(1, 1, 1), teacher=teacher,
                        weaning_phase=WEANING_PRE)
    assert jf.judge_source_independent is False   # 裁判=教师本尊·同源


def test_build_judge_fn_independent_judge_source_passes_d3():
    """D3·caller 传独立 judge_source_id（与 teacher.source_id 不相交）→
    judge_source_independent True（独立裁判分离后自然通过·闸门先就位挡假断奶）。"""
    b = DictBackend()
    bootstrap(b)
    from pure_integer_ai.teacher.recordable_teacher import register_recording_table
    register_recording_table(b)
    teacher = RecordableLLMTeacher(b, mode=MODE_OFF, source_id=7)
    jf = build_judge_fn(None, JudgeWeights(1, 1, 1), teacher=teacher,
                        weaning_phase=WEANING_PRE, judge_source_id=99)
    assert jf.judge_source_independent is True   # 99 与 7 不相交


def test_d3_judge_source_independent_hard_prerequisite():
    """D3·裁判源不独立 → ready False（even if D1+D2+D4+D5+E2 all true）。"""
    hist = [
        _m(1, cond=600, real=400, judge=600, oov=100, interv=300, reten=800, dep=100),
        _m(2, cond=600, real=400, judge=600, oov=100, interv=200, reten=750, dep=100),
        _m(3, cond=600, real=400, judge=600, oov=100, interv=200, reten=720, dep=100),
        _m(4, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=100),
    ]
    gates = _gates_true()
    gates["judge_source_independent"] = False
    rep = weaning_check(hist, **gates)
    assert rep.ready is False


# ============ D4 probe_set 隔离 + 版本化 ============

def test_probe_set_disjoint_passes():
    """D4·探针集∩训练集=∅ → is_disjoint True。"""
    ps = make_probe_set(version=1, refs=[(1, 10), (1, 11)])
    assert is_disjoint(ps, {(1, 20), (1, 30)}) is True


def test_probe_set_overlap_blocks():
    """D4·探针集∩训练集≠∅（泄漏） → is_disjoint False → can_wean False。"""
    ps = make_probe_set(version=1, refs=[(1, 10), (1, 11)])
    assert is_disjoint(ps, {(1, 11), (1, 30)}) is False   # (1,11) 泄漏


def test_probe_set_versioned_frozen():
    """D4·探针集版本化 + frozenset 不可变（bit-identical 可复现）。"""
    ps = make_probe_set(version=2, refs=[(1, 1), (1, 2)])
    assert ps.version == 2
    assert isinstance(ps.probe_refs, frozenset)
    with pytest.raises((AttributeError, Exception)):
        ps.version = 3   # frozen dataclass 不可变


def test_d4_probe_set_disjoint_hard_prerequisite():
    """D4·探针集未隔离 → ready False（even if D1+D2+D3+D5+E2 all true）。"""
    hist = [
        _m(1, cond=600, real=400, judge=600, oov=100, interv=300, reten=800, dep=100),
        _m(2, cond=600, real=400, judge=600, oov=100, interv=200, reten=750, dep=100),
        _m(3, cond=600, real=400, judge=600, oov=100, interv=200, reten=720, dep=100),
        _m(4, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=100),
    ]
    gates = _gates_true()
    gates["probe_set_disjoint"] = False
    rep = weaning_check(hist, **gates)
    assert rep.ready is False


# ============ D5 预验台账 + Mode B 最小预验 ============

def test_calibration_record_and_false_pass_rate(backend):
    """D5·记预验台账 + false_pass_rate 有理对（纯整禁浮点）。
    mode_a=fail(0) ∧ mode_b=success(1) → 误报。"""
    record_calibration(backend, round_id=1, mode_a_pass=0, mode_b_pass=1)  # 误报
    record_calibration(backend, round_id=1, mode_a_pass=1, mode_b_pass=1)  # 一致
    record_calibration(backend, round_id=1, mode_a_pass=0, mode_b_pass=0)  # 同错
    p, q = false_pass_rate(backend)
    assert (p, q) == (1, 3)   # 1 误报 / 3 总


def test_calibration_mode_b_agreement(backend):
    """D5·mode_b_agreement = count(a==b==success) / count(a==success)。"""
    record_calibration(backend, round_id=1, mode_a_pass=1, mode_b_pass=1)
    record_calibration(backend, round_id=1, mode_a_pass=1, mode_b_pass=0)
    record_calibration(backend, round_id=1, mode_a_pass=0, mode_b_pass=1)
    p, q = mode_b_agreement(backend)
    assert (p, q) == (1, 2)   # 1 一致 / 2 mode_a success


def test_mode_b_prevalidated_empty_is_false(backend):
    """D5·无 calibration 记录 → mode_b_prevalidated False（诚实·未预验）。"""
    assert mode_b_prevalidated(backend) is False


def test_mode_b_prevalidated_passes_on_stable_high(backend):
    """D5·Mode B 通过率后窗不回升∧达下限 → prevalidated True。
    4 轮·每轮 Mode B 通过率高且不回升（VM 单路径 re-derivation·墙内弱·下限 500）。"""
    # 轮 1-4·每轮 10 样本·mode_b 通过 8/10=800（高·不回升·≥FLOOR_MODE_B 500）
    for rid in (1, 2, 3, 4):
        for _ in range(8):
            record_calibration(backend, round_id=rid, mode_a_pass=1, mode_b_pass=1)
        for _ in range(2):
            record_calibration(backend, round_id=rid, mode_a_pass=1, mode_b_pass=0)
    assert mode_b_prevalidated(backend) is True


def test_mode_b_prevalidated_rising_rejected(backend):
    """D5·Mode B 通过率仍在升（未稳态） → prevalidated False。"""
    # 轮 1-4 通过率递增（仍在升=未稳态·不可断奶）
    for rid, passes in [(1, 2), (2, 4), (3, 6), (4, 8)]:  # 200/400/600/800 递增
        for _ in range(passes):
            record_calibration(backend, round_id=rid, mode_a_pass=1, mode_b_pass=1)
        for _ in range(10 - passes):
            record_calibration(backend, round_id=rid, mode_a_pass=1, mode_b_pass=0)
    assert mode_b_prevalidated(backend) is False


def test_d5_mode_b_prevalidated_hard_prerequisite():
    """D5·Mode B 未预验 → ready False（even if D1+D2+D3+D4+E2 all true）。
    防 Mode A→B 切换能力断崖假断奶。"""
    hist = [
        _m(1, cond=600, real=400, judge=600, oov=100, interv=300, reten=800, dep=100),
        _m(2, cond=600, real=400, judge=600, oov=100, interv=200, reten=750, dep=100),
        _m(3, cond=600, real=400, judge=600, oov=100, interv=200, reten=720, dep=100),
        _m(4, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=100),
    ]
    gates = _gates_true()
    gates["mode_b_prevalidated"] = False
    rep = weaning_check(hist, **gates)
    assert rep.ready is False


# ============ E2 教师下线独立产出骨架 ============

def test_e2_independent_production_all_conditions():
    """E2·三执行条件全就位 → True。"""
    assert e2_independent_production(
        teacher_offline=True, probe_input_novel=True,
        produced_without_teacher_anchor=True) is True


def test_e2_independent_production_any_missing():
    """E2·任一执行条件缺 → False。"""
    assert e2_independent_production(
        teacher_offline=False, probe_input_novel=True,
        produced_without_teacher_anchor=True) is False
    assert e2_independent_production(
        teacher_offline=True, probe_input_novel=False,
        produced_without_teacher_anchor=True) is False
    assert e2_independent_production(
        teacher_offline=True, probe_input_novel=True,
        produced_without_teacher_anchor=False) is False


def test_e2_execution_ready_always_false_currently():
    """E2·执行条件当前永 False（无真训练 run·gate 全 OFF·诚实·不伪造真断奶）。"""
    assert e2_execution_ready() is False


def test_e2_hard_prerequisite_blocks_ready():
    """E2·最硬闸门未过 → ready False（even if D1-D5 all true）。
    当前 E2 执行条件未就位 → can_wean 永 False（诚实声明当前断奶 theatrical）。"""
    hist = [
        _m(1, cond=600, real=400, judge=600, oov=100, interv=300, reten=800, dep=100),
        _m(2, cond=600, real=400, judge=600, oov=100, interv=200, reten=750, dep=100),
        _m(3, cond=600, real=400, judge=600, oov=100, interv=200, reten=720, dep=100),
        _m(4, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=100),
    ]
    gates = _gates_true()
    gates["e2_passed"] = False   # E2 当前永 False
    rep = weaning_check(hist, **gates)
    assert rep.ready is False
    assert rep.e2_passed is False


# ============ 集成：六闸门全过 ============

def test_all_six_gates_pass_ready_true():
    """六闸门全过 → ready True（D1×3 + D2 + D3 + D4 + D5 + E2）。"""
    hist = [
        _m(1, cond=100, real=100, judge=100, oov=20, interv=800, reten=600, dep=200),
        _m(2, cond=594, real=395, judge=595, oov=95, interv=500, reten=800, dep=180),
        _m(3, cond=597, real=398, judge=598, oov=98, interv=400, reten=750, dep=160),
        _m(4, cond=599, real=399, judge=599, oov=99, interv=300, reten=720, dep=140),
        _m(5, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=120),
    ]
    rep = weaning_check(hist, **_gates_true())
    assert rep.ready is True


def test_empty_history_not_ready():
    rep = weaning_check([], **_gates_true())
    assert rep.ready is False


# ============ 断奶点真驱动消费（formal_train 接线） ============

def test_weaning_blockers_lists_all_failed_gates():
    """formal_train._weaning_blockers：ready=False 时列全 9 未过闸门（诚实标注·不静默）。
    验证 D1×4(能力平台+曲线①+曲线②+依赖度)+D2+D3+D4+D5+E2 全覆盖。"""
    from pure_integer_ai.experiments.formal_train import _weaning_blockers
    rep = weaning_check([_m(1, dep=400)])   # 全闸门 False（窗口不足 + dep 高 + 全闸门 False）
    blockers = _weaning_blockers(rep)
    expected = {
        "D1_capability_plateau", "D1_intervention_decreasing",
        "D1_retention_stable", "D1_dependency_low",
        "D2_neg_pathway_active", "D3_judge_source_independent",
        "D4_probe_set_disjoint", "D5_mode_b_prevalidated",
        "E2_independent_production",
    }
    assert set(blockers) == expected


def test_weaning_blockers_empty_when_ready():
    """ready=True → 无 blocker（六闸门全过）。"""
    from pure_integer_ai.experiments.formal_train import _weaning_blockers
    hist = [
        _m(1, cond=100, real=100, judge=100, oov=20, interv=800, reten=600, dep=200),
        _m(2, cond=594, real=395, judge=595, oov=95, interv=500, reten=800, dep=180),
        _m(3, cond=597, real=398, judge=598, oov=98, interv=400, reten=750, dep=160),
        _m(4, cond=599, real=399, judge=599, oov=99, interv=300, reten=720, dep=140),
        _m(5, cond=600, real=400, judge=600, oov=100, interv=200, reten=710, dep=120),
    ]
    rep = weaning_check(hist, **_gates_true())
    assert rep.ready is True
    assert _weaning_blockers(rep) == []

