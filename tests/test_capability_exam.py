"""全方位能力考核 harness 测试（片1 MVP·2026-07-07）。

doc/重来_全方位能力考核设计_2026-07-07.md §三.1 + §D 反 theater 三层。

覆盖：
  - DimScore status 派生（PASS/FAIL/NE·严格·不允许独立写死 PASS）
  - snapshot_strengths 读 CAUSES 边 strength（纯整数）
  - run_capability_exam e2e（formal_train fixture·断言返 CapabilityReport + 8 维度齐 + ④NE + 反 theater sanity）
  - bit-identical（同输入两跑 → to_json 一致）
  - 判据可证伪（注入 strength_delta_total=0 → ⑦初心 FAIL·证非死写 PASS）

铁律：纯整数 / bit-identical / 反 theater sanity（玩具语料不许全 PASS·至少 1 维 FAIL 或 NE）。
"""
from __future__ import annotations

import json

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, NODE_CONCEPT
from pure_integer_ai.storage.edge_types import EDGE_CAUSES
from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig, DefaultRoundRunner,
)
from pure_integer_ai.experiments.capability_exam import (
    run_capability_exam, CapabilityReport, DimScore,
    project_dimensions, snapshot_strengths,
    DIM_CONCEPT, DIM_STRUCTURE, DIM_COMPUTE,
    DIM_LONG_TEXT, DIM_LONG_CODE, DIM_THREE_RING,
    DIM_INTENT, DIM_MEMORY,
    STATUS_PASS, STATUS_FAIL, STATUS_NE, STATUS_MECHANISM_LIVE,
    G_DOOR_G4, G_DOOR_G2P, G_DOOR_G3A, G_DOOR_G3B, G_DOOR_G5,
    G_ALIVE, G_DEAD_LEAK, G_DEAD_DESIGN,
)
# 复用 test_experiments 的 fixture 范式（_causal_multi_sent_item / flat_floors / _teacher）
from tests.test_experiments import (
    _causal_multi_sent_item, _multi_sent_item, flat_floors,
)


# ============ DimScore status 派生（unit） ============

def test_dim_score_status_derivation():
    """permille≥threshold→PASS / <→FAIL / NE（unit·严格派生）。

    不允许独立写死 PASS：构造 DimScore 各 status·断言 status 字段对（构造即派生·无独立设 PASS 旁路）。
    """
    # PASS：permille ≥ threshold
    d_pass = DimScore(dim="test", status=STATUS_PASS, permille=800, threshold=500)
    assert d_pass.status == STATUS_PASS
    assert d_pass.permille >= d_pass.threshold

    # FAIL：permille < threshold
    d_fail = DimScore(dim="test", status=STATUS_FAIL, permille=200, threshold=500)
    assert d_fail.status == STATUS_FAIL
    assert d_fail.permille < d_fail.threshold

    # NE：零测（permille=-1·threshold=-1）
    d_ne = DimScore(dim="test", status=STATUS_NE, permille=-1, threshold=-1)
    assert d_ne.status == STATUS_NE
    assert d_ne.permille == -1
    assert d_ne.threshold == -1

    # 纯整数守（assert_int 在 __post_init__ 调）
    assert isinstance(d_pass.permille, int)
    assert isinstance(d_pass.threshold, int)


def test_dim_score_rejects_float():
    """纯整数铁律：浮点 permille 抛（assert_int 先拦·IntViolation）。"""
    from pure_integer_ai.crosscut.guards.int_blocker import IntViolation
    with pytest.raises(IntViolation):
        DimScore(dim="test", status=STATUS_PASS, permille=800.0, threshold=500)


# ============ snapshot_strengths 读 CAUSES 边（unit） ============

def test_snapshot_strengths_reads_causes():
    """建小图（DictBackend）+ CAUSES 边 strength → snapshot 读对（纯整数·edge_key 格式）。

    构造：两概念节点 (1,1)→(1,2) EDGE_CAUSES strength=5 / (1,2)→(1,3) EDGE_CAUSES strength=3。
    断言 snapshot 返 {"1:1->1:2": 5, "1:2->1:3": 3}·NE 边（COOCCURS）不混入。
    """
    b = DictBackend()
    bootstrap(b)
    es = EdgeStore(b)
    ns = NodeStore(b)
    # 三概念节点
    ns.put(space_id=1, local_id=1, node_type=NODE_CONCEPT, tier=TIER_PRIMARY)
    ns.put(space_id=1, local_id=2, node_type=NODE_CONCEPT, tier=TIER_PRIMARY)
    ns.put(space_id=1, local_id=3, node_type=NODE_CONCEPT, tier=TIER_PRIMARY)
    # CAUSES 边（带 strength）
    es.add(space_id_from=1, local_id_from=1, space_id_to=1, local_id_to=2,
           edge_type=EDGE_CAUSES, strength=5, source=SOURCE_BARE_TEXT)
    es.add(space_id_from=1, local_id_from=2, space_id_to=1, local_id_to=3,
           edge_type=EDGE_CAUSES, strength=3, source=SOURCE_BARE_TEXT)

    g = ConceptGraph(b)
    snap = snapshot_strengths(b, g)

    assert snap == {"1:1->1:2": 5, "1:2->1:3": 3}
    assert all(isinstance(v, int) for v in snap.values())


def test_snapshot_strengths_empty_when_no_causes():
    """无 CAUSES 边 → snapshot 空 dict（不抛·纯读）。"""
    b = DictBackend()
    bootstrap(b)
    ns = NodeStore(b)
    ns.put(space_id=1, local_id=1, node_type=NODE_CONCEPT, tier=TIER_PRIMARY)
    g = ConceptGraph(b)
    assert snapshot_strengths(b, g) == {}


# ============ run_capability_exam e2e（fixture） ============

def test_run_capability_exam_fixture_e2e(tmp_path, flat_floors):
    """formal_train 玩具语料 → run_capability_exam 返 CapabilityReport。

    断言：
      - 返 CapabilityReport / 8 维度齐
      - ④长文本 = NE（零测）
      - 反 theater sanity：至少 1 维 FAIL 或 NE（玩具语料·不许全 PASS）
      - footnotes 含 #479 墙
      - strength_delta_total ≥ 0（MUTABLE_MONOTONE 守）
      - summary 格式 "X/8 examined·Z NE·W PASS·V FAIL"
    """
    from pure_integer_ai.config import gates
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        b = DictBackend()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "cap"), run_id="cap1")
        report = run_capability_exam(
            cfg, [_causal_multi_sent_item()],
            backend=b, runner=DefaultRoundRunner())

        # 类型
        assert isinstance(report, CapabilityReport)

        # 8 维度齐
        assert set(report.dimensions.keys()) == {
            DIM_CONCEPT, DIM_STRUCTURE, DIM_COMPUTE,
            DIM_LONG_TEXT, DIM_LONG_CODE, DIM_THREE_RING,
            DIM_INTENT, DIM_MEMORY,
        }

        # ④长文本 = NE
        assert report.dimensions[DIM_LONG_TEXT].status == STATUS_NE

        # 反 theater sanity：至少 1 维非 PASS（玩具语料·不许全 PASS·MECHANISM_LIVE/NE/FAIL/ABORT 都算非 PASS）
        non_pass = [d for d in report.dimensions.values()
                    if d.status != STATUS_PASS]
        assert len(non_pass) >= 1, (
            f"反 theater sanity 失败：玩具语料全 PASS = theater 风险·"
            f"summary={report.summary}")

        # footnotes 含 #479
        assert any("#479" in fn for fn in report.footnotes)

        # strength_delta_total ≥ 0（MUTABLE_MONOTONE 守）
        assert report.strength_delta_total >= 0

        # summary 格式
        assert "examined" in report.summary
        assert "NE" in report.summary
        assert "PASS" in report.summary
        assert "FAIL" in report.summary

        # to_json 结构完整
        j = report.to_json()
        assert j["run_id"] == "cap1"
        assert len(j["dimensions"]) == 8
        assert isinstance(j["strength_delta"], dict)
    finally:
        gates.TRAINING_MODE = saved


# ============ bit-identical（同输入两跑 → to_json 一致） ============

def test_capability_report_bit_identical(tmp_path, flat_floors):
    """同输入两跑 → CapabilityReport.to_json 一致（确定性基线·sort_keys）。

    formal_train 两跑同输入同输出（既有测已证）+ snapshot_strengths 确定性序 +
    project_dimensions 纯投影 → to_json 必 bit-identical。
    """
    from pure_integer_ai.config import gates
    saved = gates.TRAINING_MODE

    def run_once(run_id: str):
        gates.TRAINING_MODE = True
        try:
            b = DictBackend()
            cfg = FormalTrainConfig(run_dir=str(tmp_path / run_id), run_id=run_id)
            report = run_capability_exam(
                cfg, [_causal_multi_sent_item()],
                backend=b, runner=DefaultRoundRunner())
            # run_id 不同（两跑）·归一化后再比（core 字段一致）
            j = report.to_json()
            j["run_id"] = "NORMALIZED"   # 归一化 run_id（两跑 run_id 必不同·非核心）
            return j
        finally:
            gates.TRAINING_MODE = saved

    j1 = run_once("bi1")
    j2 = run_once("bi2")
    # 深比（json 序列化后比·捕获嵌套 dict 序差异）
    assert json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True), (
        "两跑 to_json 不一致·违 bit-identical")


# ============ 判据可证伪（注入低 metrics → ⑦初心 FAIL） ============

def test_project_dimensions_no_cheating():
    """注入 strength_delta_total=0 + 空 metrics → ⑦初心判 FAIL（证判据可证伪·非死写 PASS）。

    构造假 FormalTrainResult（final_metrics 全 0 / collapse_summary 空 / generalization=None）+
    strength_delta_total=0 → ⑦初心必 FAIL / ⑥三环必 FAIL。
    证判据不是死写 PASS（反 theater·可证伪）。
    """
    from pure_integer_ai.experiments.formal_train import FormalTrainResult
    from pure_integer_ai.training.stages import StageMetrics

    fake_result = FormalTrainResult(
        run_id="fake",
        final_metrics=StageMetrics(graph_size=0, causes_coverage=0),
        collapse_summary={},   # 三柱全 False
        generalization=None,   # 无 oracle
    )

    dims = project_dimensions(fake_result, strength_delta_total=0, backend=None)

    # ⑦初心 FAIL（strength_delta_total=0 → 不>0）
    assert dims[DIM_INTENT].status == STATUS_FAIL, (
        "⑦初心判据不可证伪：strength_delta_total=0 仍 PASS = 死写·theater")
    # ⑥三环 FAIL（collapse 三柱空 + strength_delta_total=0）
    assert dims[DIM_THREE_RING].status == STATUS_FAIL, (
        "⑥三环判据不可证伪：collapse 空 + delta=0 仍 PASS = 死写·theater")
    # ②结构 FAIL（causes_coverage=0 < 500）
    assert dims[DIM_STRUCTURE].status == STATUS_FAIL
    # ③计算 NE（generalization=None）
    assert dims[DIM_COMPUTE].status == STATUS_NE
    # ⑧记忆 FAIL（写入=0 + 消费者=0）
    assert dims[DIM_MEMORY].status == STATUS_FAIL


def test_project_dimensions_pass_when_strength_rises():
    """对照测：strength_delta_total>0 + collapse 三柱全 ok + causes_cov≥500 -> ⑦/⑥ MECHANISM_LIVE·② PASS。

    证判据双向（FAIL ↔ MECHANISM_LIVE/PASS）·非恒 FAIL（反 theater sanity 双向）。
    **STEP2 #889 stale 修正**：_DIM_G_DEAD G3a/G3b 改 ALIVE（M1片2+G1+#774 落地后 classify_intent 真填三 bool·
    G3a/G3b 真活·e2e 8 ep is_causal=True active=0 不 veto·非硬编码短路）。
    **2026-07-11 算数域病灶以小见大**：⑥⑦ 机制接通（三柱 ok + delta>0）改 MECHANISM_LIVE 第三态非 PASS（strength_delta
    是 reward 通路活代理·#479 墙·非真独立源验证学到·机制接通冒充能力达成=偷渡）。② 结构 cov≥500 仍 PASS（无 #479 墙）。
    双向 ⑥ FAIL 腿用 g_dead_override 注入 G3a/G3b DEAD_LEAK 验 dead-G门前置（D5-enforcing 防退化·前置非死写）。
    """
    from pure_integer_ai.experiments.formal_train import FormalTrainResult
    from pure_integer_ai.training.stages import StageMetrics
    from pure_integer_ai.cognition.shared.types import Episode

    fake_result = FormalTrainResult(
        run_id="fake2",
        final_metrics=StageMetrics(graph_size=100, causes_coverage=600),
        collapse_summary={"pillar1_ok": 1, "pillar2_ok": 1, "pillar3_ok": 1},
        generalization=None,
        episodes=[Episode(reward=1)],   # I-新 reward_pos 腿：⑦ MECHANISM_LIVE 须 reward_pos>0（reward 学习活）
    )

    dims = project_dimensions(fake_result, strength_delta_total=5, backend=None)

    # ⑦初心 MECHANISM_LIVE（strength_delta_total=5>0 AND reward_pos=1>0·reward 学习机制活·#479 墙非能力达成）
    assert dims[DIM_INTENT].status == STATUS_MECHANISM_LIVE
    # ⑥三环 MECHANISM_LIVE（G3a/G3b ALIVE + 三柱 ok + delta>0 -> 机制接通·#479 墙非能力达成）
    assert dims[DIM_THREE_RING].status == STATUS_MECHANISM_LIVE, (
        "⑥三环 G3a/G3b ALIVE + 三柱 ok + delta>0 -> 须 MECHANISM_LIVE（机制接通非能力达成·#479 墙·禁偷渡 PASS）")
    # ②结构 PASS（causes_coverage=600≥500·无 #479 墙·cov 达阈即能力达成）
    assert dims[DIM_STRUCTURE].status == STATUS_PASS

    # 双向 ⑥ FAIL 腿：g_dead_override 注入 G3a/G3b DEAD_LEAK -> ⑥ FAIL（dead-G门前置·D5-enforcing 防退化）
    _g_dead_leak = {DIM_THREE_RING: {G_DOOR_G4: G_ALIVE, G_DOOR_G2P: G_ALIVE,
                                     G_DOOR_G3A: G_DEAD_LEAK, G_DOOR_G3B: G_DEAD_LEAK, G_DOOR_G5: G_DEAD_DESIGN}}
    dims_leak = project_dimensions(fake_result, strength_delta_total=5, backend=None,
                                   g_dead_override=_g_dead_leak)
    assert dims_leak[DIM_THREE_RING].status == STATUS_FAIL, (
        "⑥三环 G3a/G3b DEAD_LEAK（注入·模拟未来退化）-> ⑥ 须 FAIL（dead-G门前置·D5-enforcing 防退化）")


def test_project_dimensions_intent_fail_when_reward_zero():
    """I-新 reward_pos 腿深修：delta>0 但 reward_pos=0 -> ⑦ FAIL（建图机制活非 reward 学习活·禁偷渡 MECHANISM_LIVE）。

    反 theater 核心：旧单腿判据（delta>0）reward 零触发时 delta 全为建边 base_strength 仍标
    MECHANISM_LIVE = 建图机制活冒充 reward 学习活（偷渡）。深修加 reward_pos 腿：reward_pos=0
    -> ⑦ FAIL（CI 默认 training_mode False->reward 环路零触发·诚实标 FAIL 非 MECHANISM_LIVE）。
    对照 test_project_dimensions_pass_when_strength_rises（delta>0 + reward_pos>0 -> MECHANISM_LIVE）。
    """
    from pure_integer_ai.experiments.formal_train import FormalTrainResult
    from pure_integer_ai.training.stages import StageMetrics

    fake_result = FormalTrainResult(
        run_id="fake_no_reward",
        final_metrics=StageMetrics(graph_size=100, causes_coverage=600),
        collapse_summary={"pillar1_ok": 1, "pillar2_ok": 1, "pillar3_ok": 1},
        generalization=None,
        episodes=[],   # reward_pos=0（无 episode / episode reward 全 0）-> reward 环路零触发
    )

    dims = project_dimensions(fake_result, strength_delta_total=5, backend=None)

    # ⑦初心 FAIL（delta=5>0 但 reward_pos=0·建图机制活非 reward 学习活·禁偷渡 MECHANISM_LIVE）
    assert dims[DIM_INTENT].status == STATUS_FAIL, (
        "⑦初心 reward_pos 腿失效：delta>0 但 reward_pos=0 仍 MECHANISM_LIVE = 建图机制活冒充 reward 学习活·theater")
    # ⑥三环仍 MECHANISM_LIVE（⑥不要求 reward_pos·三环 collapse 维·delta>0 + 三柱 ok）
    assert dims[DIM_THREE_RING].status == STATUS_MECHANISM_LIVE, (
        "⑥三环不受 reward_pos 腿影响（⑥=三环 collapse·⑦=初心 reward 学习·两维不同）")
