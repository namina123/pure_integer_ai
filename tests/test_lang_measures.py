"""P0 #1041 Phase2 测试：统计层产出度量构造③④（capability_exam lang_measures 消费者接线）。

承接 doc/重来_统计层产出度量_设计_2026-07-14.md + doc/重来_全局缺口重审_2026-07-14.md §6。
Phase1（构造①②·judge J4word truthiness 校准）已 done·本测覆盖 Phase2（构造③④·capability_exam 接线）。

构造③（判据④泛化消费者）：lang_rate_permille 前 observability-only 无消费者（formal_train:1967 自标"非闭环消费者"）·
  本测核证 capability_exam project_lang_measures 真消费 result.lang_generalization → CapabilityReport.lang_measures。
构造④（判据⑤跨语言汇聚度量）：result.alias_edges_seeded（P0b 桥 boot 种边数·formal_train boot 捕获·0=CI 无 alias_facts）
  → lang_measures。P0b 收敛对称性由桥设计结构保证（双向 seed + activate_candidates 自包含·test_alias_bridge AB3/AB5 验）。

机制：
  TC1 project_lang_measures ③：读 lang_generalization → lang_rate_permille（property 真算 recognized/total×1000）
     + total_held_out + recognized（判据④ deferred 闭环消费者）。
  TC2 project_lang_measures ④：读 alias_edges_seeded → pure_alias_edges_seeded（P0b 桥信号）。
  TC3 lang_generalization None → ③退化 -1/0（理论不发生·formal_train:2216 总赋值·纵深防御）。
  TC4 CapabilityReport.to_json 含 lang_measures（additive·既有字段 dimensions/summary/g_attribution/layer0 不变）。
  TC5 FormalTrainResult.alias_edges_seeded 默认 0（向后兼容·CI bit-identical）。
  TC6 e2e run_capability_exam → report.lang_measures 真填（lang_rate int + edges_seeded=0·CI 无 alias_facts）
     + 两跑 lang_measures 一致（bit-identical）+ FOOTNOTE_LANG_MEASURES 在 footnotes（反 theater·signal 非 criterion）。

铁律：纯整数（lang_measures 全整·project_lang_measures assert_int 守）/ bit-identical（additive 字段·不改
  dimensions/summary/g_attribution/layer0·既有测零回归）/ 反 theater（lang_measures=observability signal·
  FOOTNOTE_LANG_MEASURES 标非 criterion·防读成断奶判据）。全统计层（判据④⑤·非 can_ween/truth 决策层）。
"""
from __future__ import annotations

import json

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.experiments.formal_train import (
    FormalTrainResult, GeneralizationSummary, FormalTrainConfig, DefaultRoundRunner,
)
from pure_integer_ai.experiments.capability_exam import (
    project_lang_measures, CapabilityReport, run_capability_exam,
)
from pure_integer_ai.experiments.collection import CollectedItem, COLLECT_PRECEDES, SOURCE_BARE_TEXT
from tests.test_experiments import _causal_multi_sent_item, flat_floors


# ---- TC1 project_lang_measures ③（lang_rate deferred 闭环消费者） ----

def test_tc1_project_lang_measures_reads_lang_generalization():
    """③判据④：project_lang_measures 读 result.lang_generalization → lang_rate_permille 真算 property。

    lang_rate_permille = recognized*1000//max(total_held_out,1)（formal_train:1962·相1 recognize 结构对齐口径）。
    前 observability-only 无消费者（formal_train:1967 自标）·本测核证 capability_exam 真消费（deferred 闭环）。
    """
    lg = GeneralizationSummary(total_held_out=4, recognized=3, verified=0)
    result = FormalTrainResult(run_id="t1")
    result.lang_generalization = lg
    measures = project_lang_measures(result)
    assert measures["lang_rate_permille"] == 750, "3*1000//4=750（property 真算·判据④）"
    assert measures["lang_total_held_out"] == 4
    assert measures["lang_recognized"] == 3
    assert measures["pure_alias_edges_seeded"] == 0, "默认 result 无 alias boot → 0"


def test_tc1b_project_lang_measures_lang_rate_zero_when_no_heldout():
    """③边界：total_held_out=0 → lang_rate_permille=0（防除零·冷启动不报假信号·max(0,1) 守）。"""
    lg = GeneralizationSummary(total_held_out=0, recognized=0)
    result = FormalTrainResult(run_id="t1b")
    result.lang_generalization = lg
    assert project_lang_measures(result)["lang_rate_permille"] == 0


# ---- TC2 project_lang_measures ④（pure_alias_edges_seeded·P0b 桥信号） ----

def test_tc2_project_lang_measures_reads_alias_edges_seeded():
    """④判据⑤：project_lang_measures 读 result.alias_edges_seeded → pure_alias_edges_seeded（P0b 桥 boot 种边数）。

    P0b 收敛对称性由桥设计结构保证（双向 seed + activate_candidates 自包含·test_alias_bridge AB3/AB5 验）·
    runtime 对 seeded pairs 同义反复→故测 edges_seeded（桥执行信号·0=无 alias_facts→CI 无信号）。
    """
    result = FormalTrainResult(run_id="t2")
    result.lang_generalization = GeneralizationSummary(total_held_out=2, recognized=1)
    result.alias_edges_seeded = 6   # P0b boot 种 3 对 × 双向 = 6 边
    measures = project_lang_measures(result)
    assert measures["pure_alias_edges_seeded"] == 6, "P0b 桥 boot 种边数（判据⑤ observability）"
    assert measures["lang_rate_permille"] == 500   # 1*1000//2


# ---- TC3 lang_generalization None → ③退化（纵深防御） ----

def test_tc3_project_lang_measures_none_lang_generalization_defaults():
    """③退化：lang_generalization None → lang_rate=-1/total=0/recognized=0（理论不发生·formal_train:2216 总赋值·纵深防御）。

    getattr alias_edges_seeded 默认 0（旧 result 兼容·守纯整数）。返 dict 全 int（assert_int 守）。
    """
    result = FormalTrainResult(run_id="t3")   # lang_generalization 默认 None
    measures = project_lang_measures(result)
    assert measures["lang_rate_permille"] == -1, "None→NE 标记 -1（无测·纵深防御）"
    assert measures["lang_total_held_out"] == 0
    assert measures["lang_recognized"] == 0
    assert measures["pure_alias_edges_seeded"] == 0
    # 全 int 守（project_lang_measures 内 assert_int 已守·此处复核）
    assert all(isinstance(v, int) for v in measures.values())


# ---- TC4 CapabilityReport.to_json additive（既有字段不变） ----

def test_tc4_capability_report_to_json_has_lang_measures_additive():
    """TC4 additive：to_json 含 lang_measures（sorted dict）·既有字段 dimensions/summary/g_attribution/layer0 不变。

    bit-identical 守：lang_measures 是 additive 字段（同 layer0_attribution 范式）·不改既有 key 集。
    """
    report = CapabilityReport(
        run_id="t4",
        lang_measures={"lang_rate_permille": 750, "pure_alias_edges_seeded": 0,
                       "lang_total_held_out": 4, "lang_recognized": 3},
    )
    j = report.to_json()
    # lang_measures 序列化（sorted keys）
    assert j["lang_measures"] == {
        "lang_rate_permille": 750, "lang_recognized": 3,
        "lang_total_held_out": 4, "pure_alias_edges_seeded": 0}, "sorted dict 序列化"
    # 既有字段仍 in（additive 不删）
    for key in ("run_id", "summary", "dimensions", "strength_delta", "g_attribution",
                "layer0_attribution", "fixture_size_note", "footnotes"):
        assert key in j, f"additive 守：既有字段 {key} 不被 lang_measures 删"


# ---- TC5 FormalTrainResult.alias_edges_seeded 默认 0（向后兼容） ----

def test_tc5_formal_train_result_alias_edges_seeded_default_zero():
    """TC5 向后兼容：FormalTrainResult() 默认 alias_edges_seeded=0（既有构造零改·CI 无 alias_facts→bit-identical）。"""
    r = FormalTrainResult(run_id="t5")
    assert r.alias_edges_seeded == 0, "默认 0（CI 无 alias_facts·boot 空 pairs 短路→不赋值→落默认）"


# ---- TC6 e2e run_capability_exam → lang_measures 真填 + bit-identical ----

def test_tc6_run_capability_exam_lang_measures_populated(tmp_path, flat_floors):
    """TC6 e2e：run_capability_exam → report.lang_measures 真填（lang_rate int + edges_seeded=0·CI 无 alias_facts）。

    反 theater：lang_measures=observability signal·FOOTNOTE_LANG_MEASURES 在 footnotes（标非 criterion·防读成断奶判据）。
    bit-identical：两跑 lang_measures 一致（确定性·同输入同输出）。
    """
    from pure_integer_ai.config import gates
    saved = gates.TRAINING_MODE

    def run_once(run_id: str):
        gates.TRAINING_MODE = True
        try:
            b = DictBackend()
            cfg = FormalTrainConfig(run_dir=str(tmp_path / run_id), run_id=run_id)
            return run_capability_exam(
                cfg, [_causal_multi_sent_item()],
                backend=b, runner=DefaultRoundRunner())
        finally:
            gates.TRAINING_MODE = saved

    report = run_once("lm1")
    # lang_measures 真填（4 key·全 int）
    assert isinstance(report.lang_measures, dict)
    expected_keys = {"lang_rate_permille", "lang_total_held_out",
                     "lang_recognized", "pure_alias_edges_seeded"}
    assert set(report.lang_measures.keys()) == expected_keys
    assert all(isinstance(v, int) for v in report.lang_measures.values())
    # CI 无 alias_facts → edges_seeded=0（boot 空 pairs 短路·bit-identical 守）
    assert report.lang_measures["pure_alias_edges_seeded"] == 0
    # lang_rate 是 int（toy corpus 产某值·total_held_out=0 时 0·非 None）
    assert isinstance(report.lang_measures["lang_rate_permille"], int)
    # 反 theater：FOOTNOTE_LANG_MEASURES 在 footnotes（signal 非 criterion 标）
    assert any("lang_measures" in fn for fn in report.footnotes), (
        "FOOTNOTE_LANG_MEASURES 必在 footnotes（反 theater·防 observability 被读成断奶判据）")
    # to_json 含 lang_measures
    assert "lang_measures" in report.to_json()

    # bit-identical：两跑 lang_measures 一致
    report2 = run_once("lm2")
    assert report.lang_measures == report2.lang_measures, (
        "两跑 lang_measures 不一致·违 bit-identical（确定性）")


# ---- TC7 e2e 非0 alias 链（M2·对抗审钉死·闭合 formal_train:2159 接线回归缝） ----

def test_tc7_run_capability_exam_alias_facts_e2e_seeds_edges(tmp_path, flat_floors, monkeypatch):
    """TC7 e2e（M2·对抗审钉死）：alias_facts.txt → boot → result.alias_edges_seeded → report.lang_measures > 0。

    闭合 formal_train:2159 接线回归缝（wrong field / if-block never firing / return dropped 会被此测抓）。
    注入 PURE_INTEGER_AI_LOCAL_DIR（tmp dir + alias_facts.txt·apple/苹果 跨语言对）→ run_capability_exam →
    edges_seeded>0（boot 种双向 PURE_ALIAS·1 对 = 2 边）。
    TC6 验 CI 零路径（无文件→0）·TC7 验非零路径（有文件→boot 真种边→报告真读）。
    """
    from pure_integer_ai.config import gates
    # 造 alias_facts.txt（apple/苹果 跨语言对·格式 surface_a lang_a surface_b lang_b·whitespace split）
    local_dir = tmp_path / "alias_local"
    local_dir.mkdir()
    (local_dir / "alias_facts.txt").write_text("apple en 苹果 zh\n", encoding="utf-8")
    monkeypatch.setenv("PURE_INTEGER_AI_LOCAL_DIR", str(local_dir))

    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        b = DictBackend()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "cap7"), run_id="cap7")
        # 语料须含 alias surface（语料相关 KB 过滤·doc/重来_语料相关KB过滤_2026-07-16：boot filter 只留
        # ≥1 surface 在语料 vocab 的 pair·苹果 在语料 → apple↔苹果 留 → boot 种边）。_causal_multi_sent_item
        # tokens=[x,y。,z,w。] 不含苹果 → 会被过滤 → 此处用含苹果语料验真接线（含 corpus-relevant 层）。
        _alias_corpus = [CollectedItem(
            tokens=["苹果", "是", "水果"], collect_type=COLLECT_PRECEDES, source=SOURCE_BARE_TEXT)]
        report = run_capability_exam(
            cfg, _alias_corpus,
            backend=b, runner=DefaultRoundRunner())
    finally:
        gates.TRAINING_MODE = saved
    # boot 种 1 对 = 双向 2 边 → edges_seeded > 0（闭合 formal_train:2159 接线·resolve→boot→result→report 全链）
    assert report.lang_measures["pure_alias_edges_seeded"] > 0, (
        "alias_facts 注入→resolve→bootstrap→result.alias_edges_seeded→report.lang_measures 全链活·"
        "edges_seeded>0（M2 e2e 闭合接线回归缝·TC6 零路径互补）")
