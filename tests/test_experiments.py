"""Stage 7 验收门测试：experiments 实验入口层（formal_train + metrics + dump_to_sqlite + collection）。

覆盖（doc/重来_落地规划与实施顺序.md §一 experiments + §十二 E5/E7 + 五类收集）：
  - metrics 同源 D2 jsonl + StageMetrics snapshot + 断奶 series + 纯整数×1000 + 确定性
  - dump_to_sqlite 便携 SQLite 导出（backend→sqlite / dump→sqlite）·确定性有序
  - collection 五类收集 + LocalDirSource(PURE_INTEGER_AI_LOCAL_DIR 首选 E10) + E5 graceful 降级 + bit-identical
  - formal_train 五阶段编排 + --resume cursor E8 + E4 覆盖率 + H2 标定 + 终 dump + 度量门控
  - pre_flight E7 放量门 5 验收项
  - DefaultRoundRunner 真接线（observe + episode_loop + build_judge_fn）
  - 确定性 bit-identical（两跑同输入同输出 / dump 跨 run / metrics jsonl 行序确定）
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from tests.boundary_fixtures import attach_boundary_fixture

from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.edge_store import EdgeStore, SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, TIER_SHADOW
from pure_integer_ai.storage.edge_types import EDGE_CAUSES, EDGE_COOCCURS
from pure_integer_ai.cognition.shared.types import (
    Episode, JudgeWeights, IntentType, INTENT_QUESTION,
    TERMINAL_REACHED_SINK, TERMINAL_DEAD_END, WEANING_PRE,
)
from pure_integer_ai.teacher.recordable_teacher import (
    RecordableLLMTeacher, register_recording_table,
    MODE_RECORD, MODE_REPLAY, CONTENT_META_DEFINITION,
    KIND_DEFINE, GT_PASS,
)
from pure_integer_ai.training.cursor import dump_run, load_run, CursorState
from pure_integer_ai.training.stages import (
    STAGES, STAGE1_SKELETON, STAGE2_CAUSES_ABS, STAGE3_REWARD,
    STAGE4_PROMOTE_WEAN, STAGE5_MULTIMODAL, SKIPPABLE_STAGES,
)
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.metrics import MetricsCollector, RoundMetrics
from pure_integer_ai.experiments.dump_to_sqlite import (
    dump_to_sqlite, export_run_dump_to_sqlite,
)
from pure_integer_ai.experiments.collection import (
    CollectedItem, CollectionReport, LocalDirSource, InMemorySource,
    collect_corpus, source_dist_from_report,
    COLLECT_CAUSES, COLLECT_PRECEDES, COLLECT_COOCCURS, COLLECT_ABSTRACT,
)
from pure_integer_ai.experiments.formal_train import (
    make_train_context, DefaultRoundRunner, RoundResult,
    formal_train, pre_flight, FormalTrainConfig, TrainContext,
    PreFlightReport,
    PRE_FLIGHT_ROUNDS, H2_CALIB_BATCH,
)


# ---- helpers ----

def _ep(reward: int, *, terminal=TERMINAL_REACHED_SINK, vetoed=False,
        dead_end=False, g5=False) -> Episode:
    """造确定性 Episode（StubRunner 用·纯整数）。"""
    return Episode(
        episode_id=0, run_id=0, reward=reward, terminal=terminal,
        judge_veto_count=1 if vetoed else 0,
        dead_end_count=1 if dead_end else 0,
        judge_G5_active=g5,
    )


class StubRunner:
    """确定性 stub runner（orchestration 测试用·不依赖 cognition 集成）。

    stage<STAGE3 → None（observe-only）·stage>=STAGE3 → 造 Episode（reward 按 item.tokens 决定）。
    """

    def __init__(self, *, reward=1):
        self.reward = reward

    def run_round(self, ctx, item, stage, round_id):
        if stage < STAGE3_REWARD:
            return None
        return _ep(self.reward if item.tokens else 0)

    def run_round_full(self, ctx, item, stage, round_id):
        if stage < STAGE3_REWARD:
            return RoundResult()
        ep = self.run_round(ctx, item, stage, round_id)
        return RoundResult(episode=ep, output=None, dag_path=None)


def _corpus(n=3):
    """确定性测试语料（CollectedItem 列表）。"""
    return [CollectedItem(tokens=[f"t{i}_{j}" for j in range(3)],
                          role_seq=[1, 2, 3],
                          collect_type=COLLECT_PRECEDES,
                          source=SOURCE_BARE_TEXT)
            for i in range(n)]


def _multi_sent_item():
    """多句段结构化语料（role_seq 填·句末标点切 ≥2 段·串 struct_ref 链·reward 阶段可产 part）。

    用于端到端 reward>0 门（破 intent 退化致命1+generate 产空致命6+dead_end①误杀致命7 集群）。
    """
    return attach_boundary_fixture(CollectedItem(
        tokens=["a", "b。", "c", "d。"],
        role_seq=[1, 1, 1, 1],
        collect_type=COLLECT_PRECEDES,
        source=SOURCE_BARE_TEXT,
    ), cut_after=(2,))


def _causal_multi_sent_item():
    """带因果对的多句段语料（item3 反馈腿 e2e·CAUSES 边进 path.edges→sn/tn 变→strength 变）。

    两句段·段内带 causal_pairs（token index 对·observe 建 CAUSES 边 token→token）。
    struct_ref 锚边把 active 传到 token→CAUSES OR 选中→进 path.edges→propagate 调 sn/tn。
    """
    return attach_boundary_fixture(CollectedItem(
        tokens=["x", "y。", "z", "w。"],
        role_seq=[1, 1, 1, 1],
        causal_pairs=[(0, 1)],   # 段内 x→y CAUSES（_split_item 重映射到段内 index）
        collect_type=COLLECT_CAUSES,
        source=SOURCE_BARE_TEXT,
    ), cut_after=(2,))


@pytest.fixture
def flat_floors(monkeypatch):
    """orchestration 测试用：度量门控阈值置 0（StubRunner 不建图·绕过图规模门·测编排非阈值）。"""
    from pure_integer_ai.training import stages as _st
    monkeypatch.setattr(_st, "FLOOR_GRAPH_SIZE_S1", 0)
    monkeypatch.setattr(_st, "FLOOR_CAUSES_COV_S2", 0)
    monkeypatch.setattr(_st, "FLOOR_CONDUCTION_S3", 0)
    monkeypatch.setattr(_st, "FLOOR_PROMOTE_S4", 0)


def _teacher(backend, mode=MODE_RECORD):
    """假教师（录放层·确定性·不调真 LLM）。"""
    def llm_call(kind, args):
        if kind == KIND_DEFINE:
            return {"kind": KIND_DEFINE, "content_type": CONTENT_META_DEFINITION,
                    "text": f"def_{args[2]}", "response_int": 0}
        return {"kind": kind, "content_type": CONTENT_META_DEFINITION,
                "text": None, "response_int": GT_PASS}
    return RecordableLLMTeacher(backend, mode=mode, llm_call=llm_call)


# ============ metrics 同源 D2 jsonl ============

def test_metrics_record_writes_jsonl(tmp_path):
    p = str(tmp_path / "metrics.jsonl")
    with MetricsCollector(p) as mc:
        mc.record_round(0, STAGE1_SKELETON, [_ep(1), _ep(0, vetoed=True)],
                        graph_size=10, causes_coverage=200,
                        promote_count=0, oov_promote_count=0,
                        source_counts={SOURCE_BARE_TEXT: 5})
    lines = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    assert len(lines) == 1
    r = lines[0]
    assert r["round_id"] == 0
    assert r["stage"] == STAGE1_SKELETON
    assert r["graph_size"] == 10
    # conduction_rate = 1/2 ×1000 = 500
    assert r["conduction_rate"] == 500
    assert r["source_dist"] == {"4": 5}   # SOURCE_BARE_TEXT=4 → str key


def test_metrics_snapshot_and_weaning_series(tmp_path):
    p = str(tmp_path / "m.jsonl")
    with MetricsCollector(p) as mc:
        for rid in range(6):
            mc.record_round(rid, STAGE3_REWARD, [_ep(1), _ep(1)],
                            graph_size=10 + rid, causes_coverage=500,
                            promote_count=0, oov_promote_count=0)
        snap = mc.snapshot()
        assert snap.graph_size == 15
        assert snap.conduction_rate == 1000   # 2/2 ×1000
        series = mc.weaning_series()
        assert len(series) == 6
        assert series[-1].rounds == 5
        assert series[-1].conduction_rate == 1000


def test_metrics_zero_episodes_no_false_signal(tmp_path):
    """冷启动无 episode → 率=0 不报假信号（防除零·诚实占位）。"""
    p = str(tmp_path / "m.jsonl")
    with MetricsCollector(p) as mc:
        mc.record_round(0, STAGE1_SKELETON, [], graph_size=0,
                        causes_coverage=0, promote_count=0, oov_promote_count=0)
        snap = mc.snapshot()
        assert snap.conduction_rate == 0
        assert snap.graph_size == 0


def test_metrics_bit_identical(tmp_path):
    """两跑同输入 → jsonl bit-identical。"""
    def run():
        p = str(tmp_path / f"m_{id}.jsonl")
        with MetricsCollector(p) as mc:
            mc.record_round(0, STAGE1_SKELETON, [_ep(1), _ep(0, vetoed=True)],
                            graph_size=10, causes_coverage=200,
                            promote_count=1, oov_promote_count=1,
                            source_counts={1: 2, 4: 3})
        return open(p, encoding="utf-8").read()
    id = 0
    a = run()
    id = 1
    b = run()
    assert a == b


# ============ dump_to_sqlite 便携导出 ============

def _seed_graph(backend):
    bootstrap(backend)
    register_recording_table(backend)
    es = EdgeStore(backend)
    ns = NodeStore(backend)
    from pure_integer_ai.storage.spaces.registry import SpaceRegistry
    from pure_integer_ai.storage.spaces.abstract_space import AbstractSpace
    from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
    reg = SpaceRegistry(backend)
    sp = AbstractSpace.create(reg, "core")
    ci = ConceptIndex(backend)
    a = ci.ensure("a", space_id=sp.space_id, tier=TIER_PRIMARY)
    b = ci.ensure("b", space_id=sp.space_id, tier=TIER_PRIMARY)
    es.add(space_id_from=sp.space_id, local_id_from=a[1],
           space_id_to=sp.space_id, local_id_to=b[1],
           edge_type=EDGE_CAUSES, strength=1, source=SOURCE_BARE_TEXT,
           tier=TIER_PRIMARY, sn=1, tn=2)
    return sp.space_id


def test_dump_to_sqlite_portable(tmp_path):
    b = DictBackend()
    sid = _seed_graph(b)
    out = str(tmp_path / "portable.sqlite")
    dump_to_sqlite(b, out)
    assert os.path.exists(out)
    # 读回 SQLite 验证行数（便携 artifact 可查询·bootstrap 注册表元数据）
    rb = SQLiteBackend(out)
    bootstrap(rb)
    nodes = rb.select("concept_node", where=None)
    edges = rb.select("edge", where=None)
    assert len(nodes) == 2
    assert len(edges) == 1
    rb.close()


def test_dump_to_sqlite_deterministic(tmp_path):
    """两跑同 backend → sqlite 文件内容 bit-identical（有序 insert）。"""
    def run(tag):
        b = DictBackend()
        _seed_graph(b)
        out = str(tmp_path / f"p_{tag}.sqlite")
        dump_to_sqlite(b, out)
        return open(out, "rb").read()
    assert run(0) == run(1)


def test_export_run_dump_to_sqlite(tmp_path):
    """per-space dump 文件 → SQLite 文件（便携 artifact·读 cursor.dump_run 产物）。"""
    run_dir = str(tmp_path / "runs")
    b = DictBackend()
    sid = _seed_graph(b)
    dump_run(b, run_dir, "run_1", spaces=[sid])
    out = str(tmp_path / "export.sqlite")
    export_run_dump_to_sqlite(run_dir, "run_1", out)
    rb = SQLiteBackend(out)
    bootstrap(rb)
    assert len(rb.select("concept_node", where=None)) == 2
    rb.close()


# ============ collection 五类收集 ============

def test_local_dir_source_reads_txt(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "a.txt").write_text("hello world\n\nfoo bar baz", encoding="utf-8")
    (d / "b.txt").write_text("alpha beta", encoding="utf-8")
    (d / "ignore.md").write_text("skip", encoding="utf-8")
    src = LocalDirSource(local_dir=str(d))
    assert src.available() is True
    items = src.collect()
    # 2 txt 文件·a.txt 2 段 + b.txt 1 段 = 3 items
    assert len(items) == 3
    assert all(it.collect_type == COLLECT_PRECEDES for it in items)
    assert items[0].tokens == ["hello", "world"]


def test_local_dir_source_unavailable_graceful():
    src = LocalDirSource(local_dir=None)
    # 无 PURE_INTEGER_AI_LOCAL_DIR env（测试环境）→ unavailable
    import os as _os
    saved = _os.environ.pop("PURE_INTEGER_AI_LOCAL_DIR", None)
    try:
        src2 = LocalDirSource(local_dir=None)
        assert src2.available() is False
        assert src2.collect() == []
    finally:
        if saved is not None:
            _os.environ["PURE_INTEGER_AI_LOCAL_DIR"] = saved


def test_collect_corpus_e5_graceful():
    """单源失败不破坏训练（E5·失败源显式记录不静默吞错）。"""
    class BoomSource:
        def name(self): return "boom"
        def available(self): return True
        def collect(self): raise RuntimeError("SDK 挂了")
    class UnavailSource:
        def name(self): return "unavail"
        def available(self): return False
        def collect(self): return []
    good = InMemorySource([CollectedItem(tokens=["x"], collect_type=COLLECT_PRECEDES)])
    report = collect_corpus([good, BoomSource(), UnavailSource()])
    assert report.total == 1   # good 源的 1 条
    assert len(report.failed_sources) == 2
    assert any("boom" in s for s in report.failed_sources)
    assert any("unavail" in s for s in report.failed_sources)


def test_source_dist_audit():
    items = [
        CollectedItem(tokens=["a"], source=SOURCE_BARE_TEXT, collect_type=COLLECT_PRECEDES),
        CollectedItem(tokens=["b"], source=SOURCE_BARE_TEXT, collect_type=COLLECT_COOCCURS),
        CollectedItem(tokens=["c"], source=1, collect_type=COLLECT_CAUSES),  # SOURCE_CONCEPTNET
    ]
    report = collect_corpus([InMemorySource(items)])
    dist = source_dist_from_report(report)
    assert dist[SOURCE_BARE_TEXT] == 2
    assert dist[1] == 1


def test_local_dir_bit_identical(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "a.txt").write_text("one two\n\nthree four", encoding="utf-8")
    src = LocalDirSource(local_dir=str(d))
    a = [it.tokens for it in src.collect()]
    b = [it.tokens for it in src.collect()]
    assert a == b


# ============ DefaultRoundRunner 真接线 ============

def test_default_runner_observe_only_stage1():
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    ep = r.run_round(ctx, _corpus(1)[0], STAGE1_SKELETON, 0)
    assert ep is None   # observe-only
    assert len(b.select("concept_node", where=None)) > 0
    assert len(b.select("edge", where=None)) > 0


def test_default_runner_episode_stage3():
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    # 多句段语料（单段无 struct_ref 链→reward 阶段产不出 part→跳过·故须多句段）
    item = _multi_sent_item()
    # 先 observe 建图
    r.run_round(ctx, item, STAGE1_SKELETON, 0)
    # stage3 跑 episode
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 1)
    assert res.episode is not None
    assert res.dag_path is not None
    assert res.output is not None
    assert res.episode.reward >= 0   # judge 产 ≥0


def test_default_runner_episode_reward_positive_e2e():
    """端到端门：多句段结构化语料 → reward>0 涌现。

    证伪 intent 退化"reward 恒 0"声称（破致命1 sink=seed + 致命6 generate 产空 + 致命7 dead_end①误杀
    三断点集群）。多段 observe 串 struct_ref 链 → seed=首/sink=末 struct_ref → 达 sink → generate 产 part
    → reached_sink=True → judge J1+J2s>0。strength 变化属 item 3 反馈腿·此门只卡 reward>0。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    res = r.run_round_full(ctx, _multi_sent_item(), STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.episode.terminal == TERMINAL_REACHED_SINK
    assert res.episode.reward > 0, "reward>0 须涌现（证伪 reward 恒 0·堵绿测试掩零学习）"


def test_emergent_role_fallback_unblocks_fatal6_empty_role_seq_e2e():
    """缺口#1·致命6 闭合 e2e：空 role_seq 多句段语料（emergent_role 未预填）→ observe 兜底填
    role_seq（冷启动全 SUBJECT）→ generate 产 part → reached_sink=True → reward>0。

    证伪"role_seq 空致 generate continue 跳过→reward 腿空转"致命6：emergent_role 主导度闸
    的 SUBJECT 兜底（doc line580⑤）让空 role_seq 段也能闭合 reward 腿（退化态·全 SUBJECT）。
    主导度闸+混合桶精化（位置桶涌现）在 test_emergent_role.py 单元测·此门只卡致命6 闭合。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    # 空 role_seq 多句段（emergent_role 兜底填·解致命6）
    item = attach_boundary_fixture(CollectedItem(
        tokens=["a", "b。", "c", "d。"],
        role_seq=[],                      # 空·emergent_role 兜底
        collect_type=COLLECT_PRECEDES,
        source=SOURCE_BARE_TEXT,
    ), cut_after=(2,))
    res = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res.episode is not None
    assert res.output is not None
    assert len(res.output.parts) > 0, "空 role_seq 经 emergent_role 兜底须产 part（解致命6）"
    assert res.episode.terminal == TERMINAL_REACHED_SINK
    assert res.episode.reward > 0, "空 role_seq 兜底后 reward>0 须涌现（致命6 闭合）"


def test_language_causes_path_does_not_receive_scalar_reward_e2e():
    """语言 CAUSES 可进入本次路径，但两轮 scalar reward 不得改写边统计或强度。"""
    from pure_integer_ai.cognition.process.dag_path import dag_path_step
    from pure_integer_ai.cognition.shared.types import IntentType, INTENT_QUESTION
    b = DictBackend()
    ctx = make_train_context(b)
    r = DefaultRoundRunner()
    item = _causal_multi_sent_item()
    # 第一轮：observe 建图 + reward episode（ATTRACTOR ON·CAUSES 边进 path.edges·R5 兜底 sn/tn）
    res1 = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res1.episode is not None
    # path.edges 须含 CAUSES 边（证 active 传到 token + 冷启动放行·非全 PRECEDES）
    causes_in_path = [e for e in res1.dag_path.path.edges if e[4] == EDGE_CAUSES]
    assert causes_in_path, "path.edges 须含 CAUSES 边（反馈腿输入半边·破 active 传不到 token）"
    # M-00：路径存在不代表语言 scalar 可以持久化为 CAUSES 学习信号。
    causes_rows = [row for row in b.select("edge", where={"edge_type": EDGE_CAUSES})]
    assert all(row["sn"] == 0 and row["tn"] == 0 for row in causes_rows)
    strength_after_r1 = {row["local_id_from"]: row["strength"] for row in causes_rows}
    # 第二轮仍可得到本次 episode 评分，但不得把它变成跨 episode 的持久方向。
    res2 = r.run_round_full(ctx, item, STAGE3_REWARD, 1)
    assert res2.episode is not None
    assert res2.episode.reward > 0
    causes_rows2 = [row for row in b.select("edge", where={"edge_type": EDGE_CAUSES})]
    strength_after_r2 = {row["local_id_from"]: row["strength"] for row in causes_rows2}
    assert strength_after_r2 == strength_after_r1
    assert all(row["sn"] == 0 and row["tn"] == 0 for row in causes_rows2)


def test_formal_train_strength_changes_e2e(tmp_path, flat_floors):
    """硬债#3：顶层 formal_train() TRAINING_MODE=ON 全链 e2e（补 run_round_full 绕过的覆盖缺口）。

    现有反馈腿 e2e（test_feedback_loop_...）走 run_round_full 单 round 子路径·绕过 gate 二分。
    本测调顶层 formal_train()（五阶段 + stage_active_gates gate 二分 + 度量门控）·
    DefaultRoundRunner（run_round=run_round_full.episode·同 episode_loop 路径·line239-241）+
    _causal_multi_sent_item → STAGE3 reward → CAUSES 边进 path → propagate 调 sn/tn·
    证生产配置（TRAINING_MODE=ON）下控制环真闭合·破"测试绿≠生产绿"（e2e 走 run_round_full 绕过 gate 二分）。
    strength 两轮变由 test_feedback_loop 同 episode_loop 路径已证·此处 sn/tn 变为 control loop 通的充分信号。
    STAGE4 断奶闸门 D3/D4/D5/E2 永 False（weaning_e2 defer·诚实 theatrical·非断奶就绪）。
    """
    b = DictBackend()
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        cfg = FormalTrainConfig(
            run_dir=str(tmp_path / "runs"), run_id="debt3",
            rounds_per_stage=2)
        result = formal_train(cfg, [_causal_multi_sent_item()],
                              backend=b, runner=DefaultRoundRunner())
        # ① 顶层 formal_train TRAINING_MODE=ON 须跑到 STAGE3 reward（gate 二分未降 observe-only）
        assert STAGE3_REWARD in result.stages_completed, \
            "顶层 formal_train TRAINING_MODE=ON 须跑到 STAGE3 reward（gate 二分·非 observe-only 降级）"
        # ② observe 建 CAUSES 边（_causal_multi_sent_item causal_pairs·observe 侧不受止血影响·控制环未绕过）
        causes_rows = [row for row in b.select("edge", where={"edge_type": EDGE_CAUSES})]
        assert causes_rows, "observe 须建 CAUSES 边（_causal_multi_sent_item causal_pairs）"
        # ③ 止血 #1146（methodology §五·reward 非 frame）：语言域 CAUSES edge reward 写退场——生产 formal_train
        # 翻 CAUSES_REWARD_DOMAIN_FILTER_MODE ON → propagate_reward 落点① 跳过语言 CAUSES edge sn/tn/strength 写。
        # pre-hemostasis 此处 sn/tn>0：toy _causal_multi_sent_item 在生产 gate 栈（OI/OR/hotzone/ATTRACTOR 全 ON）
        # reaches sink → reward>0 arm → sn++&tn++&strength+=Δ（**非 dead-end 惩罚臂**·§一「语言 episode 全 dead-end」
        # 描述真语料 n=656 at-scale·非 toy 4-token item）·hemostasis 后全 0（停**所有**语言 CAUSES edge 写·两臂均退场·
        # §五降级·证生产生效·非 vacuous）。止血机制精测（edge-write skip + concept-dual 保留）在 test_causes_reward_domain_filter H2/H3/H4。
        assert all(row["sn"] == 0 and row["tn"] == 0 for row in causes_rows), \
            "止血 #1146：语言域 CAUSES edge 无 reward 写（sn/tn 全 0·reward 退场·methodology reward 非 frame）"
        # ③ STAGE4 断奶闸门永 False（weaning_e2 defer·诚实 theatrical·非断奶就绪）
        assert result.weaning_ready is False, \
            "weaning_ready 须 False（D3/D4/D5/E2 永 False·诚实 theatrical）"
        assert result.weaning_blockers, \
            "weaning_blockers 须非空（D3/D4/D5/E2 永未就位·诚实标注断奶阻断）"
    finally:
        gates.TRAINING_MODE = saved


def test_formal_train_e2e_ctx_code_episode_loop_matches_h2_rebuild(tmp_path, flat_floors,
                                                                    monkeypatch):
    """S4 片4 P1 agenda 闭（两审共识·test_stage_s4_selection_pref_dock.py:512 原 defer）。

    H2 _rebuild_path（formal_train.py:391+574·**函数内** import dag_path_step）与生产 episode_loop
    （episode.py:81-87·**模块级** import dag_path_step :33）必同 ctx_code 桶（同 raw+intent+_ctx_tag+
    pack_ctx_code·BY CONSTRUCTION）·否则 stage8 latent dock 后 attractor 扩张路径 token seed eff_freq
    读 ctx 桶 bit-identical 失。原守（test_stage_s4_selection_pref_dock.py:508 算法契约锁 import）+
    数学等价 + spy 默认值测。**本测加生产 e2e 数值相等断言**（spy dag_path_step wrap real·capture 两路径
    ctx_code·断言同桶非 0）。

    **双 patch 必要**（防测假过）：episode.py:33 模块级 import → patch episode_mod.dag_path_step；
    formal_train _rebuild_path :574 函数内 import → patch dag_path_mod.dag_path_step。两绑定独立·
    单 patch 只捕一路·漏另一路 = 测假过（捕不到 divergence）。
    """
    from pure_integer_ai.cognition.process import dag_path as dag_path_mod
    from pure_integer_ai.cognition.process import episode as episode_mod
    b = DictBackend()
    captured: list[int] = []
    real_step = dag_path_mod.dag_path_step   # patch 前捕 original（wrap real·返真 PathResult·不破 flow）

    def spy(edges, seeds, workmem, intent, **kw):
        captured.append(kw.get("ctx_code", 0))
        return real_step(edges, seeds, workmem, intent, **kw)

    monkeypatch.setattr(dag_path_mod, "dag_path_step", spy)    # _rebuild_path 函数内 import 捕
    monkeypatch.setattr(episode_mod, "dag_path_step", spy)     # episode_loop 模块级 import 捕

    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="ctxcode_e2e",
                                rounds_per_stage=2)
        result = formal_train(cfg, [_causal_multi_sent_item()],
                              backend=b, runner=DefaultRoundRunner())
        assert STAGE3_REWARD in result.stages_completed, \
            "须达 STAGE3 reward（episode_loop + H2 _rebuild_path 真跑 dag_path_step）"
    finally:
        gates.TRAINING_MODE = saved

    # 反 theater 牙：spy 真捕获（dag_path_step 真被调两路·非空 list·非 monkeypatch 假过）
    assert len(captured) >= 2, \
        f"dag_path_step 须被调多次（episode_loop + H2 _rebuild_path·每 round 两调）·got {len(captured)}"
    # 核心①：ctx_code 非 0（真 CAUSES 路径·_ctx_tag+pack_ctx_code 产非 0 桶·非默认 0 退化占位）
    nonzero = [c for c in captured if c != 0]
    assert nonzero, \
        "ctx_code 须非 0（_causal_multi_sent_item CAUSES 路径真跑·非默认 0 占位）"
    # 核心②：两路径同桶（episode_loop == H2 _rebuild_path·homogeneous corpus 同 domain/modality/task/intent.type）
    assert len(set(nonzero)) == 1, \
        f"所有非 0 ctx_code 须全等（episode_loop==H2 _rebuild_path·homogeneous corpus 同桶）·got {set(nonzero)}"


def test_formal_train_d2_uses_convergence_helper_not_inline(tmp_path, flat_floors,
                                                             monkeypatch):
    """B2 反 theater：formal_train D2 neg_pathway_active 须调 convergence.neg_pathway_active_from
    （单点源·非内联重复）。monkeypatch helper→True 时 D2_neg_pathway_active blocker 须消失·
    证明 formal_train 用 helper（若内联则 monkeypatch 无效·blocker 不变=theater 暴露）。

    baseline：toy 语料无 judge_veto/dead_end → neg_pathway False → D2 blocker 在。
    patch：helper→True → D2 blocker 须消失（formal_train 用 helper 才受 monkeypatch 影响）。
    """
    from pure_integer_ai.cognition.result import convergence
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        # baseline：toy 无负通路 → D2 blocker 须在
        b1 = DictBackend()
        cfg1 = FormalTrainConfig(run_dir=str(tmp_path / "runs1"), run_id="b2base",
                                 rounds_per_stage=2)
        res_base = formal_train(cfg1, [_causal_multi_sent_item()],
                                backend=b1, runner=DefaultRoundRunner())
        assert STAGE4_PROMOTE_WEAN in res_base.stages_completed, \
            "须达 STAGE4 断奶检查（flat_floors + TRAINING_MODE）"
        assert "D2_neg_pathway_active" in res_base.weaning_blockers, \
            "baseline: toy 语料无 judge_veto/dead_end → neg_pathway False → D2 blocker 须在"

        # patch：helper→True → formal_train 用 helper 则 D2 blocker 须消失
        monkeypatch.setattr(convergence, "neg_pathway_active_from",
                            lambda eps: True)
        b2 = DictBackend()
        cfg2 = FormalTrainConfig(run_dir=str(tmp_path / "runs2"), run_id="b2patch",
                                 rounds_per_stage=2)
        res_patch = formal_train(cfg2, [_causal_multi_sent_item()],
                                 backend=b2, runner=DefaultRoundRunner())
        assert "D2_neg_pathway_active" not in res_patch.weaning_blockers, \
            "monkeypatch helper→True 须使 D2 blocker 消失（证明 formal_train 用 helper 非内联·反 theater）"
    finally:
        gates.TRAINING_MODE = saved


def test_formal_train_wires_pronoun_feature_lookup_e2e(tmp_path, flat_floors, monkeypatch):
    """B5 反 theater：formal_train 生产 observe 注入 pronoun_feature_lookup（formal_train:276）。

    monkeypatch spy 监听 lookup_pronoun_features 调用·formal_train 跑含代词语料 →
    若 formal_train:276 真接线则 spy 被调（"他" in calls）·若 theater（不传 lookup）则 spy 零调→测 FAIL。
    消费侧（PROPERTY 边建）由 test_stage3.test_pronoun_feature_property_edge_via_observe_pipeline 证。
    """
    from pure_integer_ai.cognition.understanding import pronoun_features as pf
    calls: list[str] = []
    orig = pf.lookup_pronoun_features

    def spy(tok):
        calls.append(tok)
        return orig(tok)
    monkeypatch.setattr(pf, "lookup_pronoun_features", spy)
    b = DictBackend()
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="b5",
                                rounds_per_stage=2)
        # 两段：小明。他。→ seg1 含代词"他"·observe 经 normalize→is_pronoun→调 lookup
        item = CollectedItem(
            tokens=["小明", "。", "他", "。"],
            role_seq=[1, 1, 1, 1],
            collect_type=COLLECT_PRECEDES, source=SOURCE_BARE_TEXT,
        )
        formal_train(cfg, [item], backend=b, runner=DefaultRoundRunner())
        assert "他" in calls, \
            "formal_train 须调 lookup_pronoun_features（formal_train:276 接线·反 theater）"
    finally:
        gates.TRAINING_MODE = saved


def test_mode_b_post_weaning_language_scalar_is_not_persisted_e2e():
    """断奶态可计算本次结构评分，但不得把语言 scalar 写回长期 CAUSES。"""
    from pure_integer_ai.cognition.shared.types import WEANING_POST
    from pure_integer_ai.storage.edge_types import EDGE_CAUSES
    b = DictBackend()
    ctx = make_train_context(b)
    ctx.weaning_phase = WEANING_POST   # 断奶态（self_proof_fn=None·G5 vacate pass=1）
    r = DefaultRoundRunner()
    item = _causal_multi_sent_item()
    res1 = r.run_round_full(ctx, item, STAGE3_REWARD, 0)
    assert res1.episode is not None
    assert res1.episode.reward > 0
    strength_r1 = {row["local_id_from"]: row["strength"]
                   for row in b.select("edge", where={"edge_type": EDGE_CAUSES})}
    r.run_round_full(ctx, item, STAGE3_REWARD, 1)
    strength_r2 = {row["local_id_from"]: row["strength"]
                   for row in b.select("edge", where={"edge_type": EDGE_CAUSES})}
    assert strength_r2 == strength_r1
    assert all(
        row["sn"] == 0 and row["tn"] == 0
        for row in b.select("edge", where={"edge_type": EDGE_CAUSES}))


def test_cue_extractor_produces_causes_and_is_a_from_bare_text_e2e():
    """致命3 e2e：CUE_EXTRACTOR ON → 裸文本自产 CAUSES(导致)+IS_A(是一种) 边（破输入侧 D 墙）。

    证伪"裸文本产不出 CAUSES/IS_A"声称：cue_extractor 句法锚定 → Segment 字段 → observe 建边。
    反统计契约守：只指向词/系词命中产 pair（无 cue 部分零边）·非共现式 N×N。
    """
    from pure_integer_ai.storage.edge_types import EDGE_CAUSES, EDGE_IS_A
    from pure_integer_ai.experiments.formal_train import _split_item_to_segments
    from pure_integer_ai.storage.edge_store import EPI_CUE
    # 裸文本：两句·一句带导致(CAUSES)·一句带是一种(IS_A)·句末标点切两段
    item = CollectedItem(
        tokens=["雨", "导致", "地湿", "。", "苹果", "是一种", "水果", "。"],
        role_seq=[1, 1, 1, 1, 1, 1, 1, 1],
        collect_type=COLLECT_PRECEDES,
        source=SOURCE_BARE_TEXT,
    )
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = True
    try:
        segs = _split_item_to_segments(item)
        # 段内 cue 提取（句末标点切两段·段内 index 重映射）
        all_cue = [p for s in segs for p in s.cue_based_causal_pairs]
        all_is_a = [p for s in segs for p in s.is_a_pairs]
        assert all_cue, "cue_extractor 须自产 CAUSES pair（破 cue_pairs 无人喂）"
        assert all_is_a, "cue_extractor 须自产 IS_A pair"

        # observe 建边（run_round_full 走 _split→observe→EDGE_CAUSES/EDGE_IS_A）
        b = DictBackend()
        ctx = make_train_context(b)
        r = DefaultRoundRunner()
        r.run_round_full(ctx, item, STAGE2_CAUSES_ABS, 0)
        causes = [row for row in b.select("edge", where={"edge_type": EDGE_CAUSES})]
        is_a = [row for row in b.select("edge", where={"edge_type": EDGE_IS_A})]
        assert any(row["epistemic_origin"] == EPI_CUE for row in causes), \
            "裸文本自产 CAUSES 须标 EPI_CUE（来源② 指向词锚定）"
        assert is_a, "裸文本须自产 IS_A 边（破 IS_A 全仓无建边）"
        assert all(row["epistemic_origin"] == EPI_CUE for row in is_a), \
            "裸文本自产 IS_A 须标 EPI_CUE（来源② 系词锚定）"
    finally:
        gates.CUE_EXTRACTOR_MODE = saved


def test_cue_extractor_off_no_causes_is_a_from_bare_text_e2e():
    """gate OFF → 裸文本不自产 CAUSES/IS_A（bit-identical 守回归·无 cue 命中边）。"""
    from pure_integer_ai.storage.edge_types import EDGE_CAUSES, EDGE_IS_A
    item = CollectedItem(
        tokens=["雨", "导致", "地湿", "。", "苹果", "是一种", "水果", "。"],
        role_seq=[1, 1, 1, 1, 1, 1, 1, 1],
        collect_type=COLLECT_PRECEDES,
        source=SOURCE_BARE_TEXT,
    )
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = False
    try:
        b = DictBackend()
        ctx = make_train_context(b)
        r = DefaultRoundRunner()
        r.run_round_full(ctx, item, STAGE2_CAUSES_ABS, 0)
        # gate OFF → 无 EPI_CUE CAUSES（裸文本不自产）·无 IS_A
        cue_causes = [row for row in b.select("edge", where={"edge_type": EDGE_CAUSES})
                      if row.get("epistemic_origin") == 2]
        is_a = b.select("edge", where={"edge_type": EDGE_IS_A})
        assert not cue_causes, "gate OFF 须无自产 CAUSES（守回归）"
        assert not is_a, "gate OFF 须无 IS_A（守回归）"
    finally:
        gates.CUE_EXTRACTOR_MODE = saved


def test_formal_train_cue_self_produces_ca_no_hand_annotation_e2e(tmp_path, flat_floors):
    """A1 生产 e2e：顶层 formal_train()（**不手动翻 CUE gate**）→ 裸文本带 cue 词·无手注 causal_pairs
    → observe 自产 EPI_CUE CAUSES 边 + 进 reward 环 sn/tn 变（破致命3 残留·堵 test_mode_b 手注掩盖）。

    生产入口 formal_train 自动翻 CUE_EXTRACTOR_MODE ON（A1）→ 断奶后无教师手注亦能自产 CAUSES。
    区别 test_cue_extractor_produces_..._e2e（run_round_full 直调 + 手动翻 gate）：本测走顶层 formal_train
    生产路径·**不手动翻 gate**·证生产配置（TRAINING_MODE=ON）下裸文本自产（非单测旁路·堵手注掩盖）。
    自产 EPI_CUE CAUSES 与手注 CAUSES 走同一 _insert_causes（causes.py·仅 epistemic_origin 异）→
    reward/反传语义等价·故进 reward 环 sn/tn 变（H4 反馈腿·同 test_formal_train_strength_changes 模式）。
    """
    from pure_integer_ai.storage.edge_store import EPI_CUE
    # 裸文本带 cue 词·**无手注 causal_pairs**·两句段（≥2 struct_refs 让 reward 阶段 seed/sink 链跑）
    item = CollectedItem(
        tokens=["雨", "导致", "地湿", "。", "风", "导致", "树摇", "。"],
        role_seq=[1, 1, 1, 1, 1, 1, 1, 1],
        collect_type=COLLECT_PRECEDES,
        source=SOURCE_BARE_TEXT,
    )
    b = DictBackend()
    saved_training = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        cfg = FormalTrainConfig(
            run_dir=str(tmp_path / "runs"), run_id="a1_cue",
            rounds_per_stage=2)
        result = formal_train(cfg, [item], backend=b, runner=DefaultRoundRunner())
        # ① 生产 observe 自产 EPI_CUE CAUSES（A1 接线·裸文本无手注·破致命3 残留·断奶后语言域 CAUSES 源）
        assert STAGE3_REWARD in result.stages_completed, \
            "formal_train TRAINING_MODE=ON 须跑到 STAGE3 reward（gate 二分·非 observe-only 降级）"
        causes = [row for row in b.select("edge", where={"edge_type": EDGE_CAUSES})]
        cue_causes = [row for row in causes if row["epistemic_origin"] == EPI_CUE]
        assert cue_causes, \
            "生产 formal_train 须自产 EPI_CUE CAUSES（A1 接线·裸文本无手注·破致命3 残留）"
        # ② 止血 #1146（methodology §五·reward 非 frame）：自产 EPI_CUE CAUSES 是语言域 → edge reward 写退场
        # （sn/tn 全 0·pre-hemostasis 此处 sn/tn>0：toy cue item 生产 gate 栈 reaches sink → reward>0 arm →
        # sn++&tn++&strength+=Δ·**非 dead-end 惩罚臂**·§一 dead-end 描述真语料 n=656 at-scale·非 toy·hemostasis 后
        # 全 0·两臂均退场·§五降级·证生产生效·非 vacuous）。
        # 自产 CAUSES 与手注 CAUSES 在 observe 建图等价（①已证 EPI_CUE 边存在）·reward 退场是 methodology 域级决断
        # （语言 reward 非 frame·CAUSES 掌握走刀 constructive-check 不接 strength）·非"自产 CAUSES 不被消费"。
        assert all(row["sn"] == 0 and row["tn"] == 0 for row in cue_causes), \
            "止血 #1146：自产语言 EPI_CUE CAUSES edge 无 reward 写（sn/tn 全 0·reward 退场·reward 非 frame）"
    finally:
        gates.TRAINING_MODE = saved_training


# ============ formal_train 五阶段编排 ============

def test_formal_train_observe_only_when_training_off(tmp_path):
    """TRAINING_MODE OFF 时只观察，未过 gate 的阶段不得伪记完成。"""
    b = DictBackend()
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = False
    try:
        cfg = FormalTrainConfig(
            run_dir=str(tmp_path / "runs"), run_id="r1",
            rounds_per_stage=2)
        result = formal_train(cfg, _corpus(3), backend=b, runner=DefaultRoundRunner())
        assert result.stages_requested == list(STAGES)
        assert result.stages_completed == []
        assert result.dump_spaces  # 终 dump 产出
        # observe-only → 无 episode → conduction_rate=0（无假信号）·但图建了
        assert result.final_metrics.conduction_rate == 0
        assert result.final_metrics.graph_size > 0
    finally:
        gates.TRAINING_MODE = saved


def test_formal_train_reward_phase_when_training_on(tmp_path, flat_floors):
    """TRAINING_MODE ON → stage3+ 跑 episode（StubRunner 产 reward=1·conduction>0）。"""
    b = DictBackend()
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        cfg = FormalTrainConfig(
            run_dir=str(tmp_path / "runs"), run_id="r2",
            rounds_per_stage=2, telemetry_clock_ns=time.perf_counter_ns)
        result = formal_train(cfg, _corpus(3), backend=b, runner=StubRunner(reward=1))
        # StubRunner stage3 产 reward=1 episode → conduction_rate=1000
        assert result.final_metrics.conduction_rate == 1000
        # 4 个实际训练阶段 × 2 rounds × 3 items，机器指标须反映真实编排放大。
        assert result.execution.formal_train_calls == 1
        assert result.execution.stage_batch_calls == 8
        assert result.execution.stage_item_runs == 24
        assert result.execution.h2_item_runs == 0
        assert result.execution.graph_dump_calls == 1
        assert result.execution.total_elapsed_ns > 0
        execution_path = os.path.join(cfg.run_dir, "r2", "execution.json")
        with open(execution_path, encoding="utf-8") as f:
            execution_payload = json.load(f)
        assert execution_payload["execution"]["stage_item_runs"] == 24
        assert execution_payload["language_structure"]["routing"]["input_roots"] == 3
        assert execution_payload["language_structure"]["tally"]["calls"] == 1
        assert execution_payload["language_structure"]["state"] == {
            "operators_total": 1,
            "operators_new": 1,
            "cue_bearing_operators": 0,
            "realizes_operators": 0,
            "realizes_cue_operators": 0,
            "d11_shadow_edges": 0,
            "d11_primary_edges": 0,
        }
        # 终 dump 产出 per-space 文件
        assert os.path.exists(os.path.join(cfg.run_dir, "r2",
                                           f"space_{result.dump_spaces[0]}.dump"))
    finally:
        gates.TRAINING_MODE = saved


def test_formal_train_can_run_one_explicit_training_stage(tmp_path, flat_floors):
    """关系课程只跑当前训练相位，不再隐式嵌套完整四阶段。"""
    cfg = FormalTrainConfig(
        run_dir=str(tmp_path / "runs"), run_id="stage2-only",
        rounds_per_stage=2, active_training_stages=(STAGE2_CAUSES_ABS,),
        persist_graph_dump=False)

    result = formal_train(
        cfg, _corpus(3), backend=DictBackend(), runner=StubRunner(reward=1))

    assert result.stages_requested == [STAGE2_CAUSES_ABS]
    assert result.stages_completed == [STAGE2_CAUSES_ABS]
    assert result.execution.training_stages == (STAGE2_CAUSES_ABS,)
    assert result.execution.stage_batch_calls == 2
    assert result.execution.stage_item_runs == 6
    assert result.execution.graph_dump_calls == 0
    assert result.dump_spaces == []


def test_formal_train_promotes_novel_causes_cue_end_to_end(
        tmp_path, flat_floors, monkeypatch):
    """真实顶层链：外部 CAUSES 骨架使新 cue 晋升，并改变生产生成 winner。"""
    from pure_integer_ai.cognition.result.generate import generate_output
    from pure_integer_ai.cognition.result.graph_view import ConceptGraph
    from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
    from pure_integer_ai.cognition.shared.relation_primitives import REL_CAUSES
    from pure_integer_ai.cognition.shared.types import (
        LANG_ZH, LANG_NONE, PathResult, PathData, CUE_SLOT_FILL,
    )
    from pure_integer_ai.cognition.shared.work_memory import WorkMemory
    from pure_integer_ai.cognition.understanding.cue_words import (
        CAUSES_CUE_FORWARD, cue_type_of,
    )
    from pure_integer_ai.cognition.understanding.instantiates import (
        build_instantiates_edge,
    )
    from pure_integer_ai.cognition.understanding.role_precedes import (
        build_struct_anchor, build_precedes_edges, attach_role_seq,
        attach_token_seq,
    )
    from pure_integer_ai.experiments import formal_train as formal_train_module
    from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
    from pure_integer_ai.storage.node_store import NODE_CONCEPT, NODE_WORD

    corpus_tokens = [
        ["火", "导致", "烟"],
        ["雨", "导致", "湿"],
        ["a1", "引发", "b1"],
        ["a2", "引发", "b2"],
        ["a3", "引发", "b3"],
        ["a4", "引发", "b4"],
        ["a5", "引发", "b5"],
    ]
    corpus = [
        CollectedItem(
            tokens=tokens, role_seq=[1, 2, 3],
            collect_type=COLLECT_PRECEDES, source=SOURCE_BARE_TEXT)
        for tokens in corpus_tokens
    ]
    monkeypatch.setattr(gates, "TRAINING_MODE", True)
    monkeypatch.setattr(
        formal_train_module, "resolve_causes_facts",
        lambda lang: [("火", "烟"), ("雨", "湿")])

    backend = DictBackend()
    cfg = FormalTrainConfig(
        run_dir=str(tmp_path / "runs"), run_id="lang-cue-e2e",
        rounds_per_stage=1,
        active_training_stages=(STAGE4_PROMOTE_WEAN,),
        curriculum_active_relations=frozenset({"causes"}),
        curriculum_boot_relations=frozenset({"causes"}),
        persist_graph_dump=False)

    result = formal_train(cfg, corpus, backend=backend, runner=StubRunner())

    routing = result.lang_generalization.routing_stats
    tally = result.lang_generalization.tally_stats
    assert routing is not None and tally is not None
    assert routing.input_roots == 7
    assert routing.cue_clusters == 2
    assert routing.discover_samples == 4
    assert routing.recognize_samples == 3
    assert result.lang_generalization.recognized == 3
    assert tally.realizes_skeletons == 1
    assert tally.distinct_matches_added == 3
    assert tally.shadow_edges_added == 1
    state = result.lang_generalization.structure_state
    assert state is not None
    assert state.operators_total == 2
    assert state.operators_new == 2
    assert state.cue_bearing_operators == 2
    assert state.realizes_operators == 1
    assert state.realizes_cue_operators == 1
    assert state.d11_shadow_edges == 0
    assert state.d11_primary_edges == 1

    space_id = result.discovered_operators[0].skeleton_ref[0]
    concept_index = ConceptIndex(backend)
    word_ref = concept_index.lookup("引发", space_id)
    assert word_ref is not None
    edge_store = EdgeStore(backend)
    rows = edge_store.query_from(
        word_ref[0], word_ref[1], edge_type=EDGE_RELATION_SIGNAL)
    assert len(rows) == 1
    assert rows[0]["tier"] == TIER_PRIMARY
    assert rows[0]["source"] == SOURCE_BARE_TEXT
    assert ConceptGraph(backend).cue_rel_of(word_ref) == REL_CAUSES

    monkeypatch.setattr(gates, "EMERGENT_RELATION_CUE_READBACK_MODE", True)
    assert cue_type_of(
        "引发", LANG_ZH, backend=backend, edge_store=edge_store,
        space_id=space_id, concept_index=concept_index) == CAUSES_CUE_FORWARD

    # 生成采用：构造一个绑定到已确认 CAUSES skeleton 的新 unit，不给“引发”补 REFERS_TO。
    graph = ConceptGraph(backend)
    causes_skeleton = next(
        op.skeleton_ref for op in result.discovered_operators
        if graph.rel_kind_of_skeleton(op.skeleton_ref) == REL_CAUSES
        and any(cue is not None for cue in graph.read_cue_sig(op.skeleton_ref))
    )
    unit = concept_index.ensure(
        "__seg_learned_causes_generation", space_id=space_id,
        node_type=NODE_CONCEPT)
    target_tokens = [
        concept_index.ensure(surface, space_id=space_id, node_type=NODE_WORD)
        for surface in ("火", "导致", "湿")
    ]
    build_struct_anchor(
        edge_store, unit, target_tokens[0], source=SOURCE_BARE_TEXT,
        space_id=space_id, order_base=0)
    build_precedes_edges(
        edge_store, target_tokens, source=SOURCE_BARE_TEXT,
        space_id=space_id, order_base=0)
    attach_role_seq(backend, unit, [1, 2, 3], order_base=0)
    attach_token_seq(backend, unit, target_tokens, order_base=0)
    build_instantiates_edge(
        edge_store, unit, causes_skeleton, space_id=space_id)

    assert word_ref not in graph.activate_candidates(target_tokens[1]), \
        "生成采用不得依赖手工 REFERS_TO"
    assert graph.relation_cue_candidates(
        REL_CAUSES, space_id=space_id) == [word_ref]

    path = PathResult(
        path=PathData(edges=[], struct_unit_refs=[unit]),
        topo_layers=[[unit]], convergence={}, source=unit, sink=None)
    monkeypatch.setattr(gates, "DISPATCH_TOKEN_CHAIN_MODE", True)
    monkeypatch.setattr(gates, "ORDINAL_SURFACE_MODE", True)
    monkeypatch.setattr(gates, "CUE_SLOT_FILL_MODE", True)
    monkeypatch.setattr(gates, "SLOT_LCA_CONSTRAINT_MODE", True)
    monkeypatch.setattr(gates, "EXCLUDE_FUNCTION_MODE", True)

    monkeypatch.setattr(gates, "CORRESPONDENCE_SLOT_MODE", False)
    without_learning = generate_output(
        path, graph, WorkMemory(), LANG_NONE)
    assert without_learning.parts[0].words[1] == "导致"

    monkeypatch.setattr(gates, "CORRESPONDENCE_SLOT_MODE", True)
    with_learning = generate_output(
        path, graph, WorkMemory(), LANG_NONE)
    assert with_learning.parts[0].words[1] == "引发"
    assert with_learning.lineage[(unit, 1)] == CUE_SLOT_FILL


def test_formal_train_rejects_unknown_training_stage(tmp_path):
    cfg = FormalTrainConfig(
        run_dir=str(tmp_path / "runs"), run_id="invalid-stage",
        active_training_stages=(999,))

    with pytest.raises(ValueError, match="unknown stage"):
        formal_train(cfg, _corpus(1), backend=DictBackend())


def test_formal_train_dump_resume_load_bit_identical(tmp_path, flat_floors):
    """--resume：load 终 dump + cursor stage-skip·跨 run 续训 bit-identical（E1/E8）。"""
    run_dir = str(tmp_path / "runs")
    # run 1：完整训练产终 dump（DefaultRunner 真 observe 建图→dump 有行）
    b1 = DictBackend()
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        cfg1 = FormalTrainConfig(run_dir=run_dir, run_id="run1", rounds_per_stage=2)
        formal_train(cfg1, _corpus(3), backend=b1, runner=DefaultRoundRunner())
    finally:
        gates.TRAINING_MODE = saved
    assert os.path.exists(os.path.join(run_dir, "run1"))
    # run 2：resume from run1 终 dump
    b2 = DictBackend()
    gates.TRAINING_MODE = True
    try:
        cfg2 = FormalTrainConfig(run_dir=run_dir, run_id="run2",
                                 resume=True, base_run_id="run1", rounds_per_stage=2)
        result = formal_train(cfg2, _corpus(3), backend=b2, runner=DefaultRoundRunner())
        # load_run 把 run1 终 dump 载入 b2 → 图非空
        assert len(b2.select("concept_node", where=None)) > 0
        # cursor stage-skip：skippable 阶段（1/2）跳过
        assert STAGE1_SKELETON in result.stages_skipped
    finally:
        gates.TRAINING_MODE = saved


def test_formal_train_replay_coverage_gate(tmp_path, flat_floors):
    """续训前置 replay 覆盖率未达标 → 禁续训（E4·防 miss→None 静默降级）。"""
    run_dir = str(tmp_path / "runs")
    b1 = DictBackend()
    register_recording_table(b1)
    gates.TRAINING_MODE = True
    saved_t = gates.TEACHER_MODE
    gates.TEACHER_MODE = True
    try:
        cfg1 = FormalTrainConfig(run_dir=run_dir, run_id="run1", rounds_per_stage=2)
        formal_train(cfg1, _corpus(3), backend=b1, runner=StubRunner(reward=1),
                     teacher=_teacher(b1, MODE_RECORD))
    finally:
        gates.TRAINING_MODE = False
        gates.TEACHER_MODE = saved_t
    # run 2：resume·replay_needed 含未录制 key → 覆盖率未达 → RuntimeError
    b2 = DictBackend()
    register_recording_table(b2)
    gates.TRAINING_MODE = True
    gates.TEACHER_MODE = True
    try:
        cfg2 = FormalTrainConfig(run_dir=run_dir, run_id="run2",
                                 resume=True, base_run_id="run1",
                                 replay_needed=[(KIND_DEFINE, (("nope", 999),))],
                                 rounds_per_stage=2)
        with pytest.raises(RuntimeError, match="replay 覆盖率未达标"):
            formal_train(cfg2, _corpus(3), backend=b2, runner=StubRunner(reward=1),
                         teacher=_teacher(b2, MODE_REPLAY))
    finally:
        gates.TRAINING_MODE = False
        gates.TEACHER_MODE = saved_t


# ============ pre_flight E7 放量门 ============

def test_pre_flight_passes_with_signal(tmp_path):
    """E7 pre-flight：有度量信号 → 5 验收项过·允许放量。"""
    b = DictBackend()
    ctx = make_train_context(b)
    # 多句段语料（单段跳过 episode·pre_flight 须有 episode 才验 reward gate）
    corpus = [_multi_sent_item() for _ in range(5)]
    report = pre_flight(ctx, corpus, rounds=5, runner=DefaultRoundRunner())
    # DefaultRunner observe 建图 → graph_size>0 → metrics_signal
    assert report.metrics_signal is True
    assert report.mem_ok is True
    assert report.cursor_resume_ok is True
    # reward_gate_ok：多段 episode reward>0 涌现 → gate 生效（证伪 reward 恒 0）
    assert report.reward_gate_ok is True
    assert report.detail["has_pos_reward"] is True
    assert report.replay_coverage_ok is True   # 无教师→放行
    # S12：collapse_ok（柱③ 探索压力·reward 阶段 EXPLORATION_MODE ON→dag_path 内注入→柱③ OK）
    assert report.collapse_ok is True
    assert report.detail["collapse_degraded"] is False   # 有 PR 验过·非退化
    assert report.passed is True


def test_pre_flight_cursor_resume_skips_completed():
    """E8：cursor resume 跳已完成 skippable·非 skippable 保留。"""
    b = DictBackend()
    ctx = make_train_context(b)
    report = pre_flight(ctx, [], rounds=1, runner=StubRunner(reward=1))
    assert report.cursor_resume_ok is True
    # STAGE1 已完成 skippable → 跳过·STAGE3 non-skippable → 保留
    assert STAGE1_SKELETON not in report.detail["cursor_todo"]
    assert STAGE3_REWARD in report.detail["cursor_todo"]


def test_pre_flight_replay_coverage_with_teacher(tmp_path):
    """E4：教师 replay 覆盖率校验（未录制 key → 不过）。"""
    b = DictBackend()
    register_recording_table(b)
    ctx = make_train_context(b, teacher=_teacher(b, MODE_REPLAY))
    needed = [(KIND_DEFINE, (("nope", 1),))]
    report = pre_flight(ctx, _corpus(2), rounds=2,
                        runner=DefaultRoundRunner(), replay_needed=needed)
    assert report.replay_coverage_ok is False
    assert report.passed is False   # replay 覆盖率未达 → 禁放量


def test_pre_flight_reward_gate_falsifiable_no_signal():
    """stub ③：reward_gate_ok 可证伪——空语料无 episode → 无 veto/pos → False（删 cond>=0 恒真尾）。"""
    b = DictBackend()
    ctx = make_train_context(b)
    report = pre_flight(ctx, [], rounds=1, runner=DefaultRoundRunner())
    assert report.reward_gate_ok is False   # 空→无信号→False（可证伪·非恒真）


def test_pre_flight_runs_anti_collapse_verify():
    """致命5：pre_flight 生产 caller 跑 anti_collapse_verify·detail["anti_collapse"] 有汇总（非零 caller）。"""
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_multi_sent_item() for _ in range(3)]
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        report = pre_flight(ctx, corpus, rounds=3, runner=DefaultRoundRunner())
        ac = report.detail["anti_collapse"]
        # 汇总结构在（caller 真跑·柱①②③ falsifiable 各计）
        assert "verified" in ac and "pillar1_ok" in ac and "pillar3_ok" in ac
        assert ac["verified"] > 0   # 有 episode 验过（致命5 零 caller 修·三柱验收真跑）
    finally:
        gates.TRAINING_MODE = saved


def test_pre_flight_collapse_ok_degraded_no_pr_vector():
    """S12：空语料→无 episode→无 PR 向量→collapse_ok 退化放行 True + detail 标退化。

    verified=0（dag_path 未跑·无 PR 可验）→ 非趋平退化信号·由 metrics_signal/reward_gate 门先拦。
    """
    b = DictBackend()
    ctx = make_train_context(b)
    report = pre_flight(ctx, [], rounds=1, runner=DefaultRoundRunner())
    assert report.collapse_ok is True              # 退化放行（无 PR 可验）
    assert report.detail["collapse_degraded"] is True
    assert report.detail["anti_collapse"]["verified"] == 0


def test_pre_flight_collapse_ok_blocks_on_pillar3_fail(monkeypatch):
    """S12 反 theater 牙：柱③ 失守（pillar3_ok < verified）→ collapse_ok=False → passed=False。

    构造 _anti_collapse_summary 返 1/2 柱③ 失守（趋平且注入失败）= 趋平退化信号 → 放量阻塞。
    用 monkeypatch 替 preflight runtime 的 _anti_collapse_summary（pre_flight 内按名引用）。
    """
    import pure_integer_ai.experiments.preflight_runtime as preflight_runtime
    def fake_summary(eps):
        return {"verified": 2, "total": len(eps),
                "pillar1_ok": 2, "pillar2_ok": 2, "pillar3_ok": 1,   # 1/2 柱③ 失守
                "low_variance": 2}
    monkeypatch.setattr(preflight_runtime, "_anti_collapse_summary", fake_summary)
    b = DictBackend()
    ctx = make_train_context(b)
    corpus = [_multi_sent_item() for _ in range(3)]
    report = pre_flight(ctx, corpus, rounds=3, runner=DefaultRoundRunner())
    assert report.collapse_ok is False
    assert report.detail["collapse_degraded"] is False
    assert report.passed is False   # collapse_ok=False → 禁放量（即便其他门过）


def test_preflight_report_passed_requires_collapse_ok():
    """S12：PreFlightReport.passed 含 collapse_ok——collapse_ok=False → passed=False（5 项全 True 也不放）。"""
    r = PreFlightReport(metrics_signal=True, mem_ok=True, reward_gate_ok=True,
                        replay_coverage_ok=True, cursor_resume_ok=True,
                        collapse_ok=False)
    assert r.passed is False
    r.collapse_ok = True
    assert r.passed is True


# ============ E7 pre_flight 接通 formal_train() 生产主入口（S12 follow-up·破纸面闭合） ============

def test_formal_train_preflight_off_bit_identical(tmp_path, flat_floors):
    """基线确定性：config.pre_flight=False（默认）跑两次 → result + backend 终态全等（976 测零翻基线）。"""
    corpus = [_multi_sent_item() for _ in range(5)]

    def _run(rid):
        b = DictBackend()
        cfg = FormalTrainConfig(run_dir=str(tmp_path / rid), run_id=rid,
                                rounds_per_stage=2)
        res = formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())
        return b, res

    b1, r1 = _run("off1")
    b2, r2 = _run("off2")
    assert b1.snapshot() == b2.snapshot(), "OFF 两次 backend 须 bit-identical（确定性基线）"
    assert dict(b1._id_pool) == dict(b2._id_pool)
    assert r1.stages_completed == r2.stages_completed
    assert r1.pre_flight_report is None and r2.pre_flight_report is None


def test_formal_train_preflight_on_equiv_off_bit_identical(tmp_path, flat_floors):
    """★核心·守 rollback 完备：同 corpus OFF 一跑 + ON 一跑 → 终态 backend 全表 + _id_pool 全等。

    ON 路径多走 pre_flight trial（建图 + 写 op_confidence/experience_count 等 MUTABLE_MONOTONE
    递增计数器·非幂等）+ rollback 5 状态。若 rollback 漏任一状态→ON 终态≠OFF→爆→补。
    漏 backend._data→双计 / 漏 _id_pool→local_id 偏移 / 漏 _index→dedup 撞已删 / 漏 work_memory→
    pronoun 回溯看 trial 残段。等价 = rollback 完备的构造性证明。
    """
    corpus = [_multi_sent_item() for _ in range(5)]

    # OFF 路径
    b_off = DictBackend()
    cfg_off = FormalTrainConfig(run_dir=str(tmp_path / "off"), run_id="off",
                                rounds_per_stage=2)
    res_off = formal_train(cfg_off, corpus, backend=b_off, runner=DefaultRoundRunner())

    # ON 路径（pre_flight=True·trial 也跑 5 轮 = 全 corpus·rollback 后 stage loop 重跑同 corpus）
    b_on = DictBackend()
    cfg_on = FormalTrainConfig(run_dir=str(tmp_path / "on"), run_id="on",
                                rounds_per_stage=2, pre_flight=True,
                                pre_flight_rounds=5)
    res_on = formal_train(cfg_on, corpus, backend=b_on, runner=DefaultRoundRunner())

    # pre_flight 真跑 + 真过 + 真回滚（report 填·非 None）
    assert res_on.pre_flight_report is not None
    assert res_on.pre_flight_report.passed is True

    # ★ bit-identical 等价（rollback 完备构造性证明）
    assert b_off.snapshot() == b_on.snapshot(), \
        "ON 终态 backend 须 ≡ OFF（rollback 5 状态完备·trial 副作用清零）"
    assert dict(b_off._id_pool) == dict(b_on._id_pool), \
        "_id_pool 须等（rollback 含 next_id 水位·防 local_id 偏移）"
    # result 学习产出等
    assert res_off.stages_completed == res_on.stages_completed
    assert res_off.weights == res_on.weights
    assert res_off.final_metrics == res_on.final_metrics


def test_formal_train_preflight_on_passes_and_rollback_clean(tmp_path, flat_floors):
    """ON 路径 smoke：pre_flight passed + stages 跑到 STAGE3 + rollback 后无 trial 残留节点。

    rollback clean 验：trial 期建的 concept_node 须在 rollback 后清除（load_snapshot 恢复 _data）。
    multi_sent × 5 trial 建 trial-only struct_ref/token 节点·rollback 后终态 == OFF（无残）。
    """
    corpus = [_multi_sent_item() for _ in range(5)]
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="smoke",
                            rounds_per_stage=2, pre_flight=True, pre_flight_rounds=5)
    res = formal_train(cfg, corpus, backend=b, runner=DefaultRoundRunner())
    assert res.pre_flight_report is not None
    assert res.pre_flight_report.passed is True
    assert STAGE3_REWARD in res.stages_completed


def test_formal_train_preflight_on_equiv_off_training_mode_on(tmp_path, flat_floors):
    """P1 加固（对抗审1 B 项）：TRAINING_MODE=True 路径 ON≡OFF 等价·守 work_memory rollback 回归。

    默认 OFF 路径 stage loop observe-only·不读 work_memory produced_refs/pr_vector 残段→
    work_memory rollback（deepcopy）漏不会显于 backend 差异（correct-by-construction 但无回归守）。
    TRAINING_MODE=True 路径 stage loop 跑 episode_loop（reward·读 work_memory produced_refs 续接 / pr_vector）→
    若 work_memory rollback 漏（如 deepcopy 误改浅拷）→ trial 残段污染 stage loop → backend 差异 → 测爆。
    用 _causal_multi_sent_item（带 CAUSES·TRAINING_MODE ON 产 reward>0 + path.edges 含 CAUSES）。
    """
    corpus = [_causal_multi_sent_item() for _ in range(3)]
    saved = gates.TRAINING_MODE
    gates.TRAINING_MODE = True
    try:
        b_off = DictBackend()
        cfg_off = FormalTrainConfig(run_dir=str(tmp_path / "off"), run_id="off",
                                    rounds_per_stage=2)
        formal_train(cfg_off, corpus, backend=b_off, runner=DefaultRoundRunner())

        b_on = DictBackend()
        cfg_on = FormalTrainConfig(run_dir=str(tmp_path / "on"), run_id="on",
                                   rounds_per_stage=2, pre_flight=True,
                                   pre_flight_rounds=3)
        res_on = formal_train(cfg_on, corpus, backend=b_on, runner=DefaultRoundRunner())
        assert res_on.pre_flight_report is not None
        assert res_on.pre_flight_report.passed is True
        assert b_off.snapshot() == b_on.snapshot(), \
            "TRAINING_MODE ON 路径 ON≡OFF（守 work_memory rollback 回归·episode_loop 真消费 work_memory 残段）"
        assert dict(b_off._id_pool) == dict(b_on._id_pool)
    finally:
        gates.TRAINING_MODE = saved


def test_formal_train_preflight_fail_blocks_release(tmp_path, flat_floors, monkeypatch):
    """反 theater 牙：pre_flight 返 passed=False → formal_train raise RuntimeError（真阻塞·禁放量）。

    monkeypatch formal_train 模块级 pre_flight → 返全 False 报告（passed=False）·
    formal_train 须 raise（非只记 detail 继续跑）·6 项放量门生产真读。
    """
    import pure_integer_ai.experiments.formal_train as ft
    monkeypatch.setattr(ft, "pre_flight",
                        lambda *a, **k: PreFlightReport())   # 全默认 False → passed=False
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="fail",
                            rounds_per_stage=2, pre_flight=True, pre_flight_rounds=3)
    with pytest.raises(RuntimeError, match="pre_flight 放量门失败"):
        formal_train(cfg, [_multi_sent_item() for _ in range(3)],
                     backend=b, runner=DefaultRoundRunner())


def test_formal_train_preflight_collapse_pillar3_fail_blocks(tmp_path, flat_floors, monkeypatch):
    """6 项 gate 之一 fail（柱③ 失守）→ pre_flight passed=False → formal_train raise。

    monkeypatch _anti_collapse_summary 返 pillar3_ok<verified（复用 :880 fake 范式）·
    证 6 项放量门（含 S12 collapse_ok）经 formal_train caller 真生效·非 theater。
    """
    import pure_integer_ai.experiments.preflight_runtime as preflight_runtime
    def fake_summary(eps):
        return {"verified": 2, "total": len(eps),
                "pillar1_ok": 2, "pillar2_ok": 2, "pillar3_ok": 1,
                "low_variance": 2}
    monkeypatch.setattr(preflight_runtime, "_anti_collapse_summary", fake_summary)
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="col3",
                            rounds_per_stage=2, pre_flight=True, pre_flight_rounds=3)
    with pytest.raises(RuntimeError, match="pre_flight 放量门失败"):
        formal_train(cfg, [_multi_sent_item() for _ in range(3)],
                     backend=b, runner=DefaultRoundRunner())


def test_formal_train_preflight_mem_budget_zero_blocks(tmp_path, flat_floors, monkeypatch):
    """②mem_ok fail：PRE_FLIGHT_MEM_BUDGET_PER_ROUND=0 + 非空 corpus（peak>0）→ mem_ok=False → raise。

    证 mem_ok 代理 gate 真活（非永真）·经 formal_train caller 真生效。
    """
    import pure_integer_ai.experiments.preflight_runtime as preflight_runtime
    monkeypatch.setattr(preflight_runtime, "PRE_FLIGHT_MEM_BUDGET_PER_ROUND", 0)
    b = DictBackend()
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="mem",
                            rounds_per_stage=2, pre_flight=True, pre_flight_rounds=3)
    with pytest.raises(RuntimeError, match="pre_flight 放量门失败"):
        formal_train(cfg, [_multi_sent_item() for _ in range(3)],
                     backend=b, runner=DefaultRoundRunner())


def test_sqlite_backend_snapshot_load_roundtrip():
    """SQLiteBackend snapshot/load_snapshot roundtrip（对称 DictBackend · §施工序 4.2-1 解 pre_flight NotImplementedError）。

    建表 + insert → snapshot → mutate（insert/update/delete）→ load_snapshot → 数据恢复 == snapshot 时状态。
    证 SQLite snapshot 真活·`with conn` 事务 rollback 后 trial 副作用清零（pre_flight rollback 语义）。
    """
    from pure_integer_ai.storage.discipline import DISC_NONE
    b = SQLiteBackend(":memory:")
    b.register_table("t1", [("id", "INT"), ("v", "INT")], DISC_NONE)
    b.register_table("t2", [("k", "INT"), ("n", "INT")], DISC_NONE)
    b.insert("t1", {"id": 1, "v": 10})
    b.insert("t1", {"id": 2, "v": 20})
    b.insert("t2", {"k": 7, "n": 5})
    snap = b.snapshot()
    assert set(snap.keys()) == {"t1", "t2"}
    assert snap["t1"] == [{"id": 1, "v": 10}, {"id": 2, "v": 20}]
    assert snap["t2"] == [{"k": 7, "n": 5}]
    # trial 期 mutate（insert + update + delete）
    b.insert("t1", {"id": 3, "v": 30})
    b.update("t1", {"id": 1}, {"v": 999})
    b.delete("t2", {"k": 7})
    # rollback（load_snapshot · 事务原子）
    b.load_snapshot(snap)
    after = b.snapshot()
    assert after["t1"] == [{"id": 1, "v": 10}, {"id": 2, "v": 20}]   # trial insert/update 清零
    assert after["t2"] == [{"k": 7, "n": 5}]   # trial delete 清零
    b.close()


def test_formal_train_preflight_sqlite_supported(tmp_path, flat_floors):
    """SQLiteBackend snapshot 已实现 → formal_train pre_flight 不再 raise NotImplementedError（§施工序 4.2-1）。

    解原 DictBackend-only 限制（snapshot/load_snapshot 接口齐·hasattr 守通过·进 trial 路径）。trial 6 项
    验收可能 RuntimeError fail（corpus 不达标·非 snapshot 缺）·本测只核 NotImplementedError 不再抛。
    """
    b = SQLiteBackend(":memory:")
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="sqlite",
                            rounds_per_stage=2, pre_flight=True, pre_flight_rounds=2)
    try:
        formal_train(cfg, _corpus(2), backend=b, runner=DefaultRoundRunner())
    except NotImplementedError as e:
        pytest.fail(f"SQLiteBackend snapshot 已实现·不应 raise NotImplementedError: {e}")
    except RuntimeError:
        pass   # pre_flight 6 项验收可能 fail（corpus 不达标）·非 snapshot 缺·允许
    b.close()


def test_formal_train_preflight_sqlite_on_equiv_off_bit_identical(tmp_path, flat_floors):
    """★SQLite 对称 :984 DictBackend ON≡OFF 守·证 SQLite snapshot/rollback 完备（§施工序 4.2-1·审2 P1-1 采纳）。

    SQLite 与 DictBackend 物理路径不同（_do_insert 列序 / NULL 处理 / rowid 序 / `with conn` 事务）·
    DictBackend 等价测不蕴含 SQLite 等价·须独立守测。ON 路径多走 pre_flight trial + rollback 5 状态·
    若 SQLite load_snapshot 漏任一（列序错位 / rowid 偏移 / 事务半提交）→ ON 终态≠OFF → 爆。
    """
    corpus = [_multi_sent_item() for _ in range(5)]
    # OFF 路径
    b_off = SQLiteBackend(":memory:")
    cfg_off = FormalTrainConfig(run_dir=str(tmp_path / "sqlite_off"), run_id="sqlite_off",
                                rounds_per_stage=2)
    formal_train(cfg_off, corpus, backend=b_off, runner=DefaultRoundRunner())
    # ON 路径（pre_flight=True·trial 跑 5 轮 = 全 corpus·rollback 后 stage loop 重跑同 corpus）
    b_on = SQLiteBackend(":memory:")
    cfg_on = FormalTrainConfig(run_dir=str(tmp_path / "sqlite_on"), run_id="sqlite_on",
                               rounds_per_stage=2, pre_flight=True, pre_flight_rounds=5)
    res_on = formal_train(cfg_on, corpus, backend=b_on, runner=DefaultRoundRunner())
    assert res_on.pre_flight_report is not None
    assert res_on.pre_flight_report.passed is True
    # ★ bit-identical 等价（SQLite rollback 完备构造性证明·列序/rowid/事务对称 DictBackend）
    assert b_off.snapshot() == b_on.snapshot(), \
        "SQLite ON 终态须 ≡ OFF（snapshot/rollback 完备·列序/rowid/事务对称 DictBackend）"
    assert dict(b_off._id_pool) == dict(b_on._id_pool)
    b_off.close()
    b_on.close()


# ============ H2 标定 ============

def test_h2_calibrate_runs_with_teacher(tmp_path):
    """H2：阶段3 教师在位 → 小批量标定权重（不崩·产 JudgeWeights）。"""
    b = DictBackend()
    register_recording_table(b)
    saved_t = gates.TEACHER_MODE
    gates.TEACHER_MODE = True
    gates.TRAINING_MODE = True
    try:
        cfg = FormalTrainConfig(run_dir=str(tmp_path / "runs"), run_id="rh2",
                                rounds_per_stage=2)
        result = formal_train(cfg, _corpus(4), backend=b,
                              runner=DefaultRoundRunner(),
                              teacher=_teacher(b, MODE_RECORD))
        # H2 标定后 weights 是 JudgeWeights（网格搜索结果）
        assert isinstance(result.weights, JudgeWeights)
    finally:
        gates.TEACHER_MODE = saved_t
        gates.TRAINING_MODE = False
