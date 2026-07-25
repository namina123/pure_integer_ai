"""W4 兼容探针尾切及 V-00 严格 D4 纠偏回归。

验（doc/重来_断奶阶段训练设计_2026-07-11.md W4）：
1. e2e：run_weaning_arith(probe_holdout=2) 会排除末 2 样本并证明精确签名不重复，
   但缺 dedup/provenance cluster，D4 必须继续阻塞。
2. bit-identical：默认 probe_holdout=0 → 不切 → ctx.probe_set_disjoint=False → D4 blocker（既有测零翻）
3. versioning：probe_set.version 派生自 run_id（确定性·bit-identical 可复现）+ frozen + 新 run_id 新版本
4. holdout_retention track：formal_train record_round 调用传 ctx.holdout_retention（spy 验·非省略·W4 接线）+
   record_round→WeaningMetrics 真通（非硬编 0·接受非零值·真度量 defer W6）

**V-00 纠偏**：旧 W4 的集合 API 保留为兼容诊断；严格来源隔离由 evaluation_plan 承担。
"""
from __future__ import annotations

import dataclasses

import pytest

from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.config import gates
from pure_integer_ai.training import stages
from pure_integer_ai.training.stages import STAGE3_REWARD
from pure_integer_ai.experiments.formal_train import formal_train, FormalTrainConfig
from pure_integer_ai.experiments.metrics import MetricsCollector
from pure_integer_ai.experiments.collection import load_arith_corpus
from pure_integer_ai.experiments.run_weaning_train import run_weaning_arith, _derive_probe_version
from pure_integer_ai.teacher.probe_set import ref_from_signature


def test_w4_probe_holdout_is_content_diagnostic_only(tmp_path):
    """尾切只证明精确内容不重复，缺 provenance 时必须保留 D4 blocker。

    formal_train 主入口切末 2 样本作 held-out probe（shadow corpus=前 10 training·probe 不进 boot/discovery/
    H2/stage/generate/base_freq 全部下游），精确签名集合确实不相交；但同源改写与来源独立性
    无法由该集合证明，所以 V-00 严格 D4 不得通过。
    """
    result, _backend = run_weaning_arith(probe_holdout=2, return_backend=True,
                                         run_dir=str(tmp_path / "w4e2e"))
    assert "D4_probe_set_disjoint" in result.weaning_blockers, (
        f"旧尾切缺 provenance，必须保留 D4 blocker·got {result.weaning_blockers}")
    assert result.evaluation_strictly_isolated is False
    assert not result.weaning_ready
    # probe_set expose（result.probe_set 非 None·version 派生自 run_id）
    assert result.probe_set is not None
    assert result.probe_set.version == _derive_probe_version("w0_arith")
    # anti-leak：探针 item（末 2）arith_source ∉ 训练集（前 10）arith_source（真 held-out·反泄漏）
    corpus = load_arith_corpus()
    training_sigs = {it.arith_source for it in corpus[:-2]}
    probe_sigs = {it.arith_source for it in corpus[-2:]}
    assert probe_sigs.isdisjoint(training_sigs), "探针 arith_source 须 ∉ 训练集（真 held-out·反泄漏）"
    # probe_set.probe_refs 与重建 training refs 不相交（e2e 验 is_disjoint 真·非 theater）
    probe_refs = {ref_from_signature(s) for s in probe_sigs}
    training_refs = {ref_from_signature(s) for s in training_sigs}
    assert probe_refs.isdisjoint(training_refs), "probe_refs 须与 training_refs 不相交（D4 真隔离）"


def test_w4_gate_off_bit_identical(tmp_path):
    """★bit-identical：默认 probe_holdout=0 → 不切 → D4 blocker（既有测零翻）。

    默认 probe_holdout=0 → _split_holdout 返 (corpus,[]) → if probe_corpus 不执行 → ctx.probe_set_disjoint
    保持 TrainContext 默认 False → 路径 B 读 False（同原硬编·零翻）→ D4_probe_set_disjoint in blockers。
    W0 既有 reward 闭环不受影响（conduction_rate>0）。
    """
    result = run_weaning_arith(run_dir=str(tmp_path / "w4pre"))   # 默认 probe_holdout=0
    # D4 blocker（默认不切→ctx.probe_set_disjoint=False→路径 B 读 False·同原硬编·零翻）
    assert "D4_probe_set_disjoint" in result.weaning_blockers, (
        f"默认 probe_holdout=0 须 D4 blocker（bit-identical）·blockers={result.weaning_blockers}")
    assert not result.weaning_ready
    assert result.probe_set is None    # 默认不切→无 probe_set
    # W0 既有 reward 闭环不受 W4 影响（conduction_rate>0·vm_proof 自锚 reward 真流）
    assert result.final_metrics.conduction_rate > 0, "W0 reward 闭环须不受 W4 影响"


def test_w4_probe_versioning_deterministic(tmp_path):
    """★versioning：probe_set.version 派生自 run_id（确定性·bit-identical 可复现）+ frozen + 新 run_id 新版本。

    守几百 G 不重训红线（新 run_id 新探针版本·同 run_id 同版本·bit-identical 可复现）。
    ProbeSet frozen dataclass（不可变·版本化稳定）。
    """
    r1 = run_weaning_arith(probe_holdout=2, run_dir=str(tmp_path / "w4v1"))
    r2 = run_weaning_arith(probe_holdout=2, run_dir=str(tmp_path / "w4v2"))
    assert r1.probe_set is not None and r2.probe_set is not None
    # 同 run_id（"w0_arith"）→ 同 version（确定性·bit-identical 可复现·守几百 G 不重训）
    assert r1.probe_set.version == r2.probe_set.version == _derive_probe_version("w0_arith")
    # ProbeSet frozen（frozen dataclass·setattr raises·版本化稳定不可变）
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        r1.probe_set.version = 999
    # 新 run_id → 新 version（_derive_probe_version 直接验·run_weaning_arith run_id 硬编无法换）
    assert _derive_probe_version("w0_arith") != _derive_probe_version("other_run")


def test_w4_holdout_retention_track(tmp_path):
    """★holdout_retention track：record_round→WeaningMetrics 真通（非硬编 0）+ formal_train 接线（spy 验）。

    ① record_round→WeaningMetrics track 真通（非零值流过·机制存在·改原 record_round 不传=永 0）。
    ② formal_train record_round 调用传 ctx.holdout_retention kwarg（spy 验·W4 接线·非省略·默认 0 守 bit-identical）。
    **诚实**：track 通畅·非真 vm_proof 度量（真度量 defer W6 模拟退场 eval·naive fresh-compile 恒 1000 theater）。
    D1 曲线② retention_stable 达标 defer W7（retention 真值 W6 采·W7 消费）。
    """
    # ① record_round→WeaningMetrics track 真通（非零值流过·机制·非硬编 0）
    mc = MetricsCollector(str(tmp_path / "w4track" / "metrics.jsonl"))
    mc.record_round(0, STAGE3_REWARD, [], graph_size=10, causes_coverage=5,
                    promote_count=1, oov_promote_count=0, holdout_retention=800)
    ws = mc.weaning_series()
    assert ws[-1].holdout_retention == 800, (
        "record_round 须传 holdout_retention→WeaningMetrics（track 真通·非硬编 0）")
    mc.close()

    # ② formal_train record_round 调用传 holdout_retention kwarg（spy 验·W4 接线·非省略）
    seen = []

    class _SpyMC(MetricsCollector):
        def record_round(self, *a, **kw):
            seen.append(kw.get("holdout_retention", "MISSING"))
            return super().record_round(*a, **kw)

    spy = _SpyMC(str(tmp_path / "w4spy" / "metrics.jsonl"))
    cfg = FormalTrainConfig(run_dir=str(tmp_path / "w4spy"), run_id="w4spy",
                            rounds_per_stage=1, probe_holdout=2)
    corpus = load_arith_corpus()
    backend = DictBackend()
    saved_tm = gates.TRAINING_MODE
    saved_floors = (stages.FLOOR_GRAPH_SIZE_S1, stages.FLOOR_CAUSES_COV_S2,
                    stages.FLOOR_CONDUCTION_S3, stages.FLOOR_PROMOTE_S4)
    gates.TRAINING_MODE = True
    try:
        stages.FLOOR_GRAPH_SIZE_S1 = 0
        stages.FLOOR_CAUSES_COV_S2 = 0
        stages.FLOOR_CONDUCTION_S3 = 0
        stages.FLOOR_PROMOTE_S4 = 0
        formal_train(cfg, corpus, backend=backend, metrics=spy)
    finally:
        gates.TRAINING_MODE = saved_tm
        (stages.FLOOR_GRAPH_SIZE_S1, stages.FLOOR_CAUSES_COV_S2,
         stages.FLOOR_CONDUCTION_S3, stages.FLOOR_PROMOTE_S4) = saved_floors
    spy.close()
    assert seen, "formal_train 须调 record_round（跑了 stage）"
    assert "MISSING" not in seen, (
        "formal_train record_round 调用须传 holdout_retention kwarg（W4 接线·非省略）")
    assert all(v == 0 for v in seen), (
        "默认 ctx.holdout_retention=0（W6 模拟退场 eval 才驱动真值·bit-identical）")
