"""#726 片2 反 theater 锚点 + 反向回归 + mutation 敏感性测试（§D 第二三层）。

doc/重来_任务0726_反theater片2.md（4 Explore + 2 对抗审 P0/P1 修订）。

覆盖：
  - run_anti_theater_anchor：3 锚点 e2e（corpus 层注入·陷阱1·验期望维度判 FAIL·非死写 PASS）
  - run_reverse_regression：8 维度（6 regressable + 2 NE 守恒·①④注入诱因·P0-3）
  - run_capability_exam anti_theater=True/False（ABORT 态 P0-4 + 零回归）
  - mutation 敏感性（5 case·mut1-3 THRESH 常量 / mut4-5 helper monkeypatch·P1-1·陷阱3）

铁律：纯整数 / bit-identical（to_json list[dict] sort·P0-5）/ 不纸面闭合（真跑真断言）/ stable≠correct。
"""
from __future__ import annotations

import json

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig, FormalTrainResult, DefaultRoundRunner,
)
from pure_integer_ai.training.stages import StageMetrics
from pure_integer_ai.experiments.capability_exam import (
    run_capability_exam, run_anti_theater_anchor, run_reverse_regression,
    project_dimensions, AnchorCheck, ReverseRegressionCase,
    DIM_CONCEPT, DIM_STRUCTURE, DIM_COMPUTE,
    DIM_LONG_TEXT, DIM_LONG_CODE, DIM_THREE_RING,
    DIM_INTENT, DIM_MEMORY,
    STATUS_PASS, STATUS_FAIL, STATUS_NE, STATUS_ABORT,
    THRESH_CAUSES_COV, THRESH_RATE_PERMILLE, THRESH_STRENGTH_DELTA,
)
from tests.test_experiments import _causal_multi_sent_item


# ============ helper ============

@pytest.fixture
def _training_mode():
    """formal_train 生产路径须 TRAINING_MODE=True（reward 阶段）·测完恢复。"""
    from pure_integer_ai.config import gates
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    yield saved
    gates.TRAINING_MODE = saved


# ============ 锚点（§D 第二层·3 e2e） ============

def test_run_anti_theater_anchor_3_anchors(tmp_path, _training_mode):
    """3 锚点 e2e 跑·全 passed=True（验 fixture 真造出期望 FAIL·非死写 PASS）。

    锚点1 anchor_arith_no_heldout：nullary 算术 2 样本 → held_out=0 → rate=0 → ③FAIL（P0-1）。
    锚点2 anchor_arith_all_wrong：arith_specs 全错 → Mode A task-driven verified=0 → ⑤FAIL（纠正3）。
    锚点3 anchor_no_causes_lang：无 cue 无 CAUSES → delta=0 + collapse 全 0 → ⑥⑦FAIL（纠正2 e2e 半）。
    """
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "anchor"), run_id="anchor_root")
    anchors = run_anti_theater_anchor(
        cfg, lambda: DictBackend(), runner=DefaultRoundRunner())

    assert len(anchors) == 3
    names = {a.name for a in anchors}
    assert names == {"anchor_arith_no_heldout", "anchor_arith_all_wrong", "anchor_no_causes_lang"}
    for a in anchors:
        assert isinstance(a, AnchorCheck)
        # STEP2 #889：anchor_arith_all_wrong expected=NE（⑤取严）·其余 expected=FAIL
        _exp = STATUS_NE if a.name == "anchor_arith_all_wrong" else STATUS_FAIL
        assert a.expected_status == _exp
        assert a.passed is True, (
            f"锚点 {a.name} 未 passed：actual={a.actual_status}·evidence={a.evidence}")


def test_anti_theater_anchor_to_dict_sort_bit_identical(_training_mode, tmp_path):
    """AnchorCheck.to_dict 序列化 bit-identical（key 固定序·evidence sort·#726 P0-5）。"""
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "anchor_bi"), run_id="bi")
    a1 = run_anti_theater_anchor(cfg, lambda: DictBackend(), runner=DefaultRoundRunner())
    a2 = run_anti_theater_anchor(cfg, lambda: DictBackend(), runner=DefaultRoundRunner())
    d1 = [a.to_dict() for a in sorted(a1, key=lambda a: a.name)]
    d2 = [a.to_dict() for a in sorted(a2, key=lambda a: a.name)]
    # evidence 可能含 metrics 实测值（run_id 等归一化后一致）·sort 后比
    s1 = json.dumps(d1, sort_keys=True, ensure_ascii=False)
    s2 = json.dumps(d2, sort_keys=True, ensure_ascii=False)
    assert s1 == s2, "两跑锚点 to_dict 不一致·违 bit-identical"


# ============ 反向回归（§D 第三层·8 维度） ============

def test_run_reverse_regression_8_cases():
    """反向回归 8 维度·全 passed=True（判据可证伪 + NE 守恒）。

    5 regressable（②③⑥⑦⑧）+ 3 ne_conservation（①④⑤·STEP2 #889 ⑤取严 NE·注入诱因·P0-3）。
    ⑥精确联立（三柱 ok + delta=0 → FAIL·测 strength_delta>0 那条腿·纠正2）。
    """
    cases = run_reverse_regression()
    assert len(cases) == 8
    ne_cases = [c for c in cases if c.category == "ne_conservation"]
    reg_cases = [c for c in cases if c.category == "regressable"]
    assert len(ne_cases) == 3
    assert len(reg_cases) == 5
    assert {c.dim for c in ne_cases} == {DIM_CONCEPT, DIM_LONG_TEXT, DIM_LONG_CODE}
    assert {c.dim for c in reg_cases} == {
        DIM_STRUCTURE, DIM_COMPUTE,
        DIM_THREE_RING, DIM_INTENT, DIM_MEMORY}
    for c in cases:
        assert isinstance(c, ReverseRegressionCase)
        assert c.passed is True, (
            f"反向回归 {c.dim}（{c.category}）未 passed："
            f"expected={c.expected_status} actual={c.actual_status}·evidence={c.evidence}")


def test_reverse_regression_ne_conservation_injects_bait():
    """NE 守恒①④：注入非 NE 诱因（graph_size>0 + 全 PASS 字段）→ 断言仍 NE（P0-3·非同义反复）。

    防未来偷偷塞判据：若改①读 graph_size 判 PASS·graph_size>0 会让①变非 NE·此测抓到。
    """
    cases = run_reverse_regression()
    c1 = [c for c in cases if c.dim == DIM_CONCEPT][0]
    c4 = [c for c in cases if c.dim == DIM_LONG_TEXT][0]
    # 诱因应含 graph_size>0 / causes_cov 等看似该 PASS 的字段
    assert "graph_size=100" in c1.bad_fixture or "graph_size" in c1.bad_fixture
    assert c1.expected_status == STATUS_NE
    assert c1.actual_status == STATUS_NE
    assert c4.expected_status == STATUS_NE
    assert c4.actual_status == STATUS_NE


def test_reverse_regression_six_exact_linkage():
    """⑥精确联立反例：三柱全 ok + delta=0 → FAIL（测 strength_delta>0 那条腿·P0-1/纠正2）。"""
    cases = run_reverse_regression()
    c6 = [c for c in cases if c.dim == DIM_THREE_RING][0]
    assert "三柱全 ok" in c6.bad_fixture or "pillar" in c6.bad_fixture
    assert "delta=0" in c6.bad_fixture or "strength_delta_total=0" in c6.bad_fixture
    assert c6.expected_status == STATUS_FAIL
    assert c6.actual_status == STATUS_FAIL


# ============ run_capability_exam anti_theater 接入（P0-4 ABORT） ============

def test_run_capability_exam_anti_theater_true(tmp_path, _training_mode):
    """anti_theater=True 完整跑·anti_theater_passed=True（锚点+反向回归全过）·无 ABORT。"""
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "at"), run_id="at1")
    report = run_capability_exam(
        cfg, [_causal_multi_sent_item()],
        backend=b, runner=DefaultRoundRunner(),
        anti_theater=True, backend_factory=lambda: DictBackend())

    assert report.anti_theater_passed is True
    # 无 ABORT 维度（全过）
    abort_dims = [d for d in report.dimensions.values() if d.status == STATUS_ABORT]
    assert abort_dims == []
    # anti_theater_anchor/reverse_regression 是 list[dict]（非 placeholder）
    assert len(report.anti_theater_anchor) == 3
    assert len(report.reverse_regression) == 8
    # to_json 序列化（P0-5）
    j = report.to_json()
    assert j["anti_theater_passed"] is True
    assert len(j["anti_theater_anchor"]) == 3
    assert len(j["reverse_regression"]) == 8
    # summary 不含 ABORT（全过）
    assert "ABORT" not in report.summary


def test_run_capability_exam_anti_theater_false_placeholder(tmp_path, _training_mode):
    """anti_theater=False（默认）·anti_theater_anchor/reverse_regression 是 placeholder list[dict]（零回归）。"""
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "atoff"), run_id="atoff1")
    report = run_capability_exam(
        cfg, [_causal_multi_sent_item()],
        backend=b, runner=DefaultRoundRunner())   # anti_theater 默认 False

    assert report.anti_theater_passed is True   # 默认 True（未跑锚点·无失败）
    # placeholder list[dict]（含 _status key·标 anti_theater=False）
    assert len(report.anti_theater_anchor) == 1
    assert "_status" in report.anti_theater_anchor[0]
    assert len(report.reverse_regression) == 1
    assert "_status" in report.reverse_regression[0]
    # to_json placeholder 序列化
    j = report.to_json()
    assert j["anti_theater_passed"] is True
    assert j["anti_theater_anchor"] == [dict(sorted(d.items())) for d in report.anti_theater_anchor]


def test_anti_theater_true_requires_backend_factory(tmp_path, _training_mode):
    """anti_theater=True 但 backend_factory=None → ValueError（每锚点须独立 backend）。"""
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "aterr"), run_id="aterr1")
    with pytest.raises(ValueError, match="backend_factory"):
        run_capability_exam(
            cfg, [_causal_multi_sent_item()],
            backend=b, runner=DefaultRoundRunner(),
            anti_theater=True)   # 无 backend_factory


def test_anti_theater_abort_on_failed_anchor(tmp_path, _training_mode, monkeypatch):
    """ABORT 态（P0-4）：monkeypatch 让锚点失败 → 失败维度升 ABORT + anti_theater_passed=False。

    保诊断信息：只升失败锚点对应维度（非全维度升）。
    """
    # monkeypatch run_anti_theater_anchor 返一个 passed=False 的锚点（anchor_arith_no_heldout → ③）
    def _fake_anchors(config, backend_factory, runner=None):
        return [AnchorCheck(
            name="anchor_arith_no_heldout",
            injected="fake mismatch",
            expected_status=STATUS_FAIL,
            actual_status=STATUS_PASS,   # 实际 PASS（判据失效·应 FAIL 却 PASS）
            passed=False,
            evidence=["fake：③实际 PASS·判据失效"],
        )]
    from pure_integer_ai.experiments import capability_exam as ce
    monkeypatch.setattr(ce, "run_anti_theater_anchor", _fake_anchors)

    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "abort"), run_id="abort1")
    report = run_capability_exam(
        cfg, [_causal_multi_sent_item()],
        backend=b, runner=DefaultRoundRunner(),
        anti_theater=True, backend_factory=lambda: DictBackend())

    # anti_theater_passed=False（有失败锚点）
    assert report.anti_theater_passed is False
    # ③计算 升 ABORT（只升失败锚点对应维度·P0-4）
    assert report.dimensions[DIM_COMPUTE].status == STATUS_ABORT
    # 其他维度不升 ABORT（保诊断信息）
    other_abort = [d for n, d in report.dimensions.items()
                   if d.status == STATUS_ABORT and n != DIM_COMPUTE]
    assert other_abort == []
    # summary 含 ABORT 前缀 + 计数
    assert "ABORT" in report.summary
    assert report.summary.startswith("ABORT[")
    assert "1 ABORT" in report.summary


def test_anti_theater_abort_on_failed_regression(tmp_path, _training_mode, monkeypatch):
    """ABORT 态（反向回归失败）：monkeypatch 让 ②反向回归失败 → ②升 ABORT。"""
    def _fake_regression():
        return [ReverseRegressionCase(
            dim=DIM_STRUCTURE,
            category="regressable",
            bad_fixture="fake",
            expected_status=STATUS_FAIL,
            actual_status=STATUS_PASS,   # 判据失效
            passed=False,
            evidence=["fake：②实际 PASS·判据失效"],
        )]
    from pure_integer_ai.experiments import capability_exam as ce
    monkeypatch.setattr(ce, "run_reverse_regression", _fake_regression)

    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "abort_reg"), run_id="abort_reg1")
    report = run_capability_exam(
        cfg, [_causal_multi_sent_item()],
        backend=b, runner=DefaultRoundRunner(),
        anti_theater=True, backend_factory=lambda: DictBackend())

    assert report.anti_theater_passed is False
    assert report.dimensions[DIM_STRUCTURE].status == STATUS_ABORT


def test_run_capability_exam_anti_theater_to_json_bit_identical(tmp_path, _training_mode):
    """anti_theater=True 完整 to_json 双跑 bit-identical（对抗审 2 P1-b·补完整链守）。

    既有 test_anti_theater_anchor_to_dict_sort_bit_identical 只守锚点 to_dict·本测守完整
    run_capability_exam(anti_theater=True) 的 to_json（含 dimensions/strength_delta/anchor/regression）。
    """
    def run_once(tag: str):
        b = DictBackend()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / f"bi_{tag}"), run_id=f"bi_{tag}")
        rep = run_capability_exam(
            cfg, [_causal_multi_sent_item()],
            backend=b, runner=DefaultRoundRunner(),
            anti_theater=True, backend_factory=lambda: DictBackend())
        j = rep.to_json()
        j["run_id"] = "NORMALIZED"   # 归一化 run_id（两跑不同）
        return j

    j1 = run_once("1")
    j2 = run_once("2")
    assert json.dumps(j1, sort_keys=True, ensure_ascii=False) == json.dumps(
        j2, sort_keys=True, ensure_ascii=False), (
        "anti_theater=True 两跑 to_json 不一致·违 bit-identical")


def test_anti_theater_abort_summary_names_dim(tmp_path, _training_mode, monkeypatch):
    """ABORT summary 维度名出现（对抗审 1 盲点1·补名断言非仅 startswith）。

    若 _ANCHOR_DIMS 映射错（如 anchor_arith_no_heldout→②）·summary 仍 startswith ABORT[·但名错。
    本测断言期望维度名（③计算）真出现在 summary。
    """
    def _fake_anchors(config, backend_factory, runner=None):
        return [AnchorCheck(
            name="anchor_arith_no_heldout",
            injected="fake", expected_status=STATUS_FAIL,
            actual_status=STATUS_PASS, passed=False, evidence=["fake"])]
    from pure_integer_ai.experiments import capability_exam as ce
    monkeypatch.setattr(ce, "run_anti_theater_anchor", _fake_anchors)

    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "abort_name"), run_id="abort_name1")
    report = run_capability_exam(
        cfg, [_causal_multi_sent_item()],
        backend=b, runner=DefaultRoundRunner(),
        anti_theater=True, backend_factory=lambda: DictBackend())

    # ③计算 名出现在 summary（DIM_ORDER 序拼接·"③计算" 是 DIM_COMPUTE 全名）
    assert "③计算" in report.summary, (
        f"ABORT 维度名未出现 summary：{report.summary}")
    # ③计算 维度升 ABORT
    assert report.dimensions[DIM_COMPUTE].status == STATUS_ABORT


# ============ mutation 敏感性（§D 陷阱3·5 case·P1-1 helper monkeypatch） ============

def test_mutation_mut1_thresh_causes_cov(monkeypatch):
    """mut1：THRESH_CAUSES_COV=0 → ②cov=0 仍 PASS（判据失效）→ harness 抓到（baseline FAIL ≠ mutant PASS）。

    simulated mutation（monkeypatch 常量·非改源码·效果等价·工程量小·诚实标）。
    """
    from pure_integer_ai.experiments import capability_exam as ce
    fake = FormalTrainResult(
        run_id="m1", final_metrics=StageMetrics(graph_size=100, causes_coverage=0))
    # baseline：cov=0 < 500 → FAIL
    baseline = project_dimensions(fake, strength_delta_total=5, backend=None)
    assert baseline[DIM_STRUCTURE].status == STATUS_FAIL
    # mutant：THRESH_CAUSES_COV=0 → cov=0>=0 → PASS
    monkeypatch.setattr(ce, "THRESH_CAUSES_COV", 0)
    mutant = project_dimensions(fake, strength_delta_total=5, backend=None)
    assert mutant[DIM_STRUCTURE].status == STATUS_PASS
    # harness 抓到：baseline ≠ mutant
    assert baseline[DIM_STRUCTURE].status != mutant[DIM_STRUCTURE].status


def test_mutation_mut2_thresh_rate_permille(monkeypatch):
    """mut2：THRESH_RATE_PERMILLE=0 → ③rate=0 仍 PASS（判据失效）→ harness 抓到。"""
    from pure_integer_ai.experiments import capability_exam as ce
    from pure_integer_ai.experiments.formal_train import GeneralizationSummary
    fake = FormalTrainResult(
        run_id="m2", final_metrics=StageMetrics(graph_size=100, causes_coverage=600),
        generalization=GeneralizationSummary(total_held_out=2, recognized=0, verified=0))
    baseline = project_dimensions(fake, strength_delta_total=5, backend=None)
    assert baseline[DIM_COMPUTE].status == STATUS_FAIL   # rate=0<500
    monkeypatch.setattr(ce, "THRESH_RATE_PERMILLE", 0)
    mutant = project_dimensions(fake, strength_delta_total=5, backend=None)
    assert mutant[DIM_COMPUTE].status == STATUS_PASS   # rate=0>=0
    assert baseline[DIM_COMPUTE].status != mutant[DIM_COMPUTE].status


def test_mutation_mut3_intent_status(monkeypatch):
    """mut3：_intent_status 改 >=0（delta=0 仍 PASS·判据失效）→ 反向回归⑦抓到。

    ⑦初心 status 派生提成 _intent_status helper（P1-1 范式·bit-identical 纯重构）·
    monkeypatch 该 helper 模拟"判据失效"。⑦反向回归 baseline（delta=0→FAIL·passed=True）。
    """
    from pure_integer_ai.experiments import capability_exam as ce
    # baseline：反向回归⑦ passed=True（delta=0→FAIL）
    baseline_cases = run_reverse_regression()
    baseline_c7 = [c for c in baseline_cases if c.dim == DIM_INTENT][0]
    assert baseline_c7.passed is True
    # mutant：_intent_status 改 >=0（delta=0 仍 PASS）
    monkeypatch.setattr(ce, "_intent_status",
                        lambda delta, rp: STATUS_PASS if delta >= 0 else STATUS_FAIL)
    mutant_cases = run_reverse_regression()
    mutant_c7 = [c for c in mutant_cases if c.dim == DIM_INTENT][0]
    # ⑦变 PASS（应 FAIL）→ harness 抓到
    assert mutant_c7.actual_status == STATUS_PASS
    assert mutant_c7.passed is False


def test_mutation_mut4_concept_status_ne_break(monkeypatch):
    """mut4：_concept_status 返 STATUS_PASS（删 NE 分支模拟）→ ①NE 守恒破 → 反向回归①抓到（P0-3 真牙）。

    验 NE 守恒测不是同义反复：注入诱因 + monkeypatch helper → ①变非 NE → 测试抓到。
    """
    from pure_integer_ai.experiments import capability_exam as ce
    # baseline：反向回归① passed=True（①仍 NE·注入诱因不改）
    baseline_cases = run_reverse_regression()
    baseline_c1 = [c for c in baseline_cases if c.dim == DIM_CONCEPT][0]
    assert baseline_c1.passed is True
    assert baseline_c1.actual_status == STATUS_NE
    # mutant：_concept_status 返 STATUS_PASS（模拟未来偷偷塞判据）
    monkeypatch.setattr(ce, "_concept_status", lambda graph_size: STATUS_PASS)
    mutant_cases = run_reverse_regression()
    mutant_c1 = [c for c in mutant_cases if c.dim == DIM_CONCEPT][0]
    # ①变 PASS·NE 守恒破·harness 抓到（passed=False）
    assert mutant_c1.actual_status == STATUS_PASS
    assert mutant_c1.passed is False


def test_mutation_mut5_three_ring_or(monkeypatch):
    """mut5：_three_ring_status 改 or（pillars_all_ok OR delta>0）→ ⑥联立破 → 反向回归⑥抓到。

    验⑥精确联立测有真牙：三柱 ok + delta=0 → mutant 下 or → PASS（非 FAIL）→ 测试抓到。
    """
    from pure_integer_ai.experiments import capability_exam as ce
    monkeypatch.setattr(
        ce, "_three_ring_status",
        lambda pillars_all_ok, delta, dead_states=(): STATUS_PASS if (pillars_all_ok or delta > 0) else STATUS_FAIL)
    cases = run_reverse_regression()
    c6 = [c for c in cases if c.dim == DIM_THREE_RING][0]
    # ⑥精确联立 fixture：三柱 ok + delta=0 → mutant or → PASS（应 FAIL）→ 抓到
    assert c6.actual_status == STATUS_PASS
    assert c6.passed is False

def test_mutation_mut6_three_ring_dead_g(monkeypatch):
    """mut6：_three_ring_status 删 dead-G门前置 -> ⑥偷渡 PASS -> 抓到（STEP2 #889）。

    验 dead-G门前置行敏感：g_dead_override 注入 G3a/G3b DEAD_LEAK + 三柱 ok + delta>0·
    baseline ⑥ FAIL（前置·D5-enforcing）·mutant 删前置 -> ⑥ PASS（应 FAIL·偷渡）-> 抓到。
    **STEP2 #889 stale 修正**：真实 _DIM_G_DEAD G3a/G3b ALIVE（M1片2+G1+#774 落地）·
    mut6 用 g_dead_override 注入 DEAD_LEAK 验前置行敏感（模拟未来退化）。
    """
    from pure_integer_ai.experiments import capability_exam as ce
    from pure_integer_ai.experiments.formal_train import FormalTrainResult
    from pure_integer_ai.training.stages import StageMetrics
    from pure_integer_ai.experiments.capability_exam import (
        G_DOOR_G4, G_DOOR_G2P, G_DOOR_G3A, G_DOOR_G3B, G_DOOR_G5,
        G_ALIVE, G_DEAD_LEAK, G_DEAD_DESIGN)
    fake = FormalTrainResult(
        run_id="mut6",
        final_metrics=StageMetrics(graph_size=100, causes_coverage=600),
        collapse_summary={"pillar1_ok": 1, "pillar2_ok": 1, "pillar3_ok": 1},
    )
    # 注入 G3a/G3b DEAD_LEAK（模拟未来退化·真实 _DIM_G_DEAD G3a/G3b ALIVE）
    _g_dead_leak = {DIM_THREE_RING: {G_DOOR_G4: G_ALIVE, G_DOOR_G2P: G_ALIVE,
                                     G_DOOR_G3A: G_DEAD_LEAK, G_DOOR_G3B: G_DEAD_LEAK, G_DOOR_G5: G_DEAD_DESIGN}}
    # baseline：⑥ FAIL（dead G门 DEAD_LEAK 前置·D5-enforcing）
    baseline = project_dimensions(fake, strength_delta_total=5, backend=None, g_dead_override=_g_dead_leak)
    assert baseline[DIM_THREE_RING].status == STATUS_FAIL, (
        "⑥三环 baseline 须 FAIL（dead-G门前置·注入 G3a/G3b DEAD_LEAK·D5-enforcing）")
    # mutant：删 dead-G门前置（lambda 不用 dead_states·返 PASS if pillars and delta>0）
    monkeypatch.setattr(
        ce, "_three_ring_status",
        lambda pillars_all_ok, delta, dead_states=(): STATUS_PASS if (pillars_all_ok and delta > 0) else STATUS_FAIL)
    mutant = project_dimensions(fake, strength_delta_total=5, backend=None, g_dead_override=_g_dead_leak)
    # ⑥变 PASS（应 FAIL·D5-enforcing 破·偷渡）-> 抓到
    assert mutant[DIM_THREE_RING].status == STATUS_PASS, (
        "mutant 删 dead-G门前置 -> ⑥ 须 PASS（三柱 ok + delta>0·前置失效偷渡）")
    assert baseline[DIM_THREE_RING].status != mutant[DIM_THREE_RING].status


def test_mutation_mut7_long_code_ne(monkeypatch):
    """mut7：_long_code_status 删 generate_literal_ne 前置 -> ⑤偷渡 Mode A -> 抓到（STEP2 #889）。

    验 ⑤ NE 守恒行敏感：generate 非 None + Mode A PASS + generate 字面 NE·
    baseline ⑤ NE（取严·D5-enforcing）·mutant 删前置 -> ⑤ PASS（应 NE·偷渡）-> 抓到。
    """
    from pure_integer_ai.experiments import capability_exam as ce
    from pure_integer_ai.experiments.formal_train import FormalTrainResult, GenerateSummary
    from pure_integer_ai.training.stages import StageMetrics
    fake = FormalTrainResult(
        run_id="mut7",
        final_metrics=StageMetrics(graph_size=100, causes_coverage=600),
        collapse_summary={"pillar1_ok": 1, "pillar2_ok": 1, "pillar3_ok": 1},
        generate=GenerateSummary(total_tasks=2, selected=2, verified=2),   # Mode A PASS
    )
    # baseline：⑤ NE（取严·generate 字面零测·D5-enforcing）
    baseline = project_dimensions(fake, strength_delta_total=5, backend=None)
    assert baseline[DIM_LONG_CODE].status == STATUS_NE, (
        "⑤长代码 baseline 须 NE（取严·generate 字面零测·D5-enforcing）")
    # mutant：删 generate_literal_ne 前置（lambda 直接返 mode_a_status）
    monkeypatch.setattr(
        ce, "_long_code_status",
        lambda generate_literal_ne, mode_a_status: mode_a_status)
    mutant = project_dimensions(fake, strength_delta_total=5, backend=None)
    # ⑤变 PASS（应 NE·D5-enforcing 破·Mode A 偷渡）-> 抓到
    assert mutant[DIM_LONG_CODE].status == STATUS_PASS, (
        "mutant 删 generate_literal_ne 前置 -> ⑤ 须 PASS（Mode A PASS 偷渡）")
    assert baseline[DIM_LONG_CODE].status != mutant[DIM_LONG_CODE].status


# ============ I-新：pre_flight 接通反 theater 生产自检（#726 片2 闭环） ============

def test_pre_flight_anti_theater_self_check_triggered(tmp_path, _training_mode):
    """I-新：pre_flight 传 config+backend_factory -> 触发反 theater 自我考核（层2锚点+层3反向回归）。

    闭环"旗标对自身失效残留"（D6 病更深层）：anti_theater 机制在测试中真活但生产 caller 从未触发->
    pre_flight 放量门接通生产自检。caller 传 config+backend_factory -> anti_theater_triggered=True +
    anti_theater_ok 真判（锚点+反向回归全过->True）+ detail 含 anchor/regression list。
    对照：不传 config/backend_factory -> anti_theater_triggered=False（bit-identical passthrough·守既有测）。
    """
    from pure_integer_ai.experiments.formal_train import (
        pre_flight, make_train_context, FormalTrainConfig, DefaultRoundRunner)
    from pure_integer_ai.storage.backend import DictBackend
    from tests.test_experiments import _multi_sent_item

    # 传 config+backend_factory -> 触发自检（独立 backend 跑锚点·不污染 ctx.backend）
    b1 = DictBackend()
    ctx1 = make_train_context(b1)
    corpus = [_multi_sent_item() for _ in range(5)]
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "pf_at"), run_id="pf_at")
    rep = pre_flight(ctx1, corpus, rounds=5, runner=DefaultRoundRunner(),
                     config=cfg, backend_factory=lambda: DictBackend())
    assert rep.detail["anti_theater_triggered"] is True
    assert rep.anti_theater_ok is True   # 正确系统：3 锚点+8 反向回归全过
    assert "anti_theater_anchor" in rep.detail
    assert len(rep.detail["anti_theater_anchor"]) == 3   # 3 锚点（arith_no_heldout/arith_all_wrong/no_causes_lang）
    assert "anti_theater_regression" in rep.detail
    assert len(rep.detail["anti_theater_regression"]) == 8   # 8 维反向回归
    # anti_theater_ok=True 不阻塞 passed（6 项 + anti_theater 全过 -> passed=True）
    assert rep.passed is True

    # 对照：不传 config/backend_factory -> skip（bit-identical passthrough·既有 9+ caller 路径）
    b2 = DictBackend()
    ctx2 = make_train_context(b2)
    rep2 = pre_flight(ctx2, [_multi_sent_item() for _ in range(5)],
                      rounds=5, runner=DefaultRoundRunner())
    assert rep2.detail["anti_theater_triggered"] is False
    assert rep2.anti_theater_ok is True   # 未触发 passthrough（默认 True·不阻塞 passed）
    assert "anti_theater_note" in rep2.detail


# ============ I-新：pre_flight 接通反 theater 生产自检（#726 片2 闭环） ============
