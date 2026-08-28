"""V-06 只读评测与训练状态隔离协议测试。"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.types import FloorActivation, JudgeWeights
from pure_integer_ai.experiments.capability_exam import run_capability_exam
from pure_integer_ai.experiments.collection import (
    COLLECT_PRECEDES,
    CollectedItem,
    load_arith_corpus,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    EvaluationIsolationError,
    clone_backend,
    isolated_evaluation,
)
from pure_integer_ai.experiments.formal_train import (
    DefaultRoundRunner,
    FormalTrainConfig,
    _h2_calibrate,
    _measure_floor_pass,
    _run_calibration_phase,
    _run_simulated_offline_eval,
    make_train_context,
    pre_flight,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.node_store import NODE_CONCEPT, TIER_PRIMARY, NodeStore
from pure_integer_ai.teacher.weaning_calibration import (
    CALIBRATION_TABLE,
    mode_b_prevalidated,
)


def _context_state(ctx):
    """提取 V-06 要求保持不变的宿主状态。"""
    backend_attrs = {
        name: copy.deepcopy(getattr(ctx.backend, name))
        for name in ("_id_pool", "_isa_edge_gen", "_legacy_observe_timestamp_seq")
        if hasattr(ctx.backend, name)
    }
    identity_cache = (
        copy.deepcopy(ctx.concept_index._index),
        copy.deepcopy(ctx.concept_index._loaded_spaces),
        copy.deepcopy(ctx.scoped_identity_store._scope_hashes),
        copy.deepcopy(ctx.scoped_identity_store._clock_hashes),
        copy.deepcopy(ctx.scoped_identity_store._timestamp_hashes),
        copy.deepcopy(ctx.scoped_identity_store._assertion_hashes),
    )
    teacher_calls = getattr(ctx.teacher, "call_count", None)
    return (
        ctx.backend.snapshot(),
        backend_attrs,
        identity_cache,
        copy.deepcopy(ctx.work_memory),
        teacher_calls,
    )


def _language_item() -> CollectedItem:
    """构造可让 pre-flight 产生图和 reward 信号的多句语言样本。"""
    return CollectedItem(
        tokens=["甲", "导致", "乙。", "乙", "影响", "丙。"],
        role_seq=[1, 1, 1, 1, 1, 1],
        collect_type=COLLECT_PRECEDES,
        source=SOURCE_BARE_TEXT,
    )


def _arith_eval_items() -> list[CollectedItem]:
    """构造带独立参树的平方样本，供 calibration 和 offline eval 使用。"""
    items: list[CollectedItem] = []
    for item, parameter in zip(load_arith_corpus()[:2], ("b", "c")):
        items.append(replace(
            item,
            arith_source_b=f"lambda {parameter}: Sigma(1, {parameter}, {parameter})",
        ))
    return items


@pytest.mark.parametrize("backend_type", [DictBackend, SQLiteBackend])
def test_clone_backend_copies_schema_indexes_data_and_watermarks(backend_type):
    """Dict/SQLite 沙箱复制完整 schema、索引、数据和水位，但不共享可变状态。"""
    source = backend_type()
    try:
        ctx = make_train_context(source)
        ctx.concept_index.ensure("宿主节点", space_id=ctx.space_id)
        source_before = source.snapshot()
        clone = clone_backend(source)
        try:
            assert clone.snapshot() == source_before
            assert clone._tables == source._tables
            NodeStore(clone).put(
                ctx.space_id,
                999_999,
                node_type=NODE_CONCEPT,
                tier=TIER_PRIMARY,
            )
            clone._id_pool[ctx.space_id] = 999_999
            assert source.snapshot() == source_before
            assert source._id_pool != clone._id_pool
        finally:
            clone.close()
    finally:
        source.close()


def test_persistent_sqlite_clone_stays_beside_source_and_cleans_up(tmp_path):
    """大图评测 clone 使用源库同盘临时 SQLite，关闭后只删除该临时文件。"""
    source_path = tmp_path / "training.sqlite3"
    source = SQLiteBackend(str(source_path), performance_mode="bulk")
    clone = None
    clone_path = None
    try:
        ctx = make_train_context(source)
        ctx.concept_index.ensure("宿主节点", space_id=ctx.space_id)
        source.commit()
        clone = clone_backend(source)
        database_rows = clone._conn.execute("PRAGMA database_list").fetchall()
        clone_path = next(
            item[2] for item in database_rows if item[1] == "main")
        assert clone_path
        assert Path(clone_path).resolve().parent == tmp_path.resolve()
        assert Path(clone_path).is_file()
        assert clone.snapshot() == source.snapshot()
    finally:
        if clone is not None:
            clone.close()
        source.close()
    assert clone_path is not None
    assert not Path(clone_path).exists()
    assert source_path.is_file()


def test_sqlite_clone_preserves_uncommitted_visible_state(tmp_path):
    """通用 clone 在未提交事务中使用 SQLite bytes 退路，不丢当前可见状态。"""
    source = SQLiteBackend(str(tmp_path / "uncommitted.sqlite3"))
    clone = None
    try:
        ctx = make_train_context(source)
        source.commit()
        ctx.concept_index.ensure("未提交节点", space_id=ctx.space_id)
        assert source._conn.in_transaction
        clone = clone_backend(source)
        assert clone.snapshot() == source.snapshot()
        assert clone._conn.execute("PRAGMA database_list").fetchone()[2] == ""
    finally:
        if clone is not None:
            clone.close()
        source.close()


def test_isolated_evaluation_uses_independent_nested_owner_and_scope():
    """嵌套评测各自获得独立 owner/session，退出后父沙箱和正式上下文均不变。"""
    ctx = make_train_context(DictBackend())
    host_before = _context_state(ctx)
    with isolated_evaluation(ctx, label="outer") as outer:
        assert outer.scope_owner != ctx.scope_owner
        assert outer.work_memory.active_session_scope is not None
        assert outer.work_memory.active_session_scope.owner == outer.scope_owner
        outer_before = _context_state(outer)
        with isolated_evaluation(outer, label="inner") as inner:
            assert inner.scope_owner != outer.scope_owner
            assert inner.work_memory.active_session_scope.owner == inner.scope_owner
            inner.node_store.put(
                inner.space_id, 800_001,
                node_type=NODE_CONCEPT, tier=TIER_PRIMARY)
        assert _context_state(outer) == outer_before
    assert _context_state(ctx) == host_before


def test_evaluator_exception_discards_sandbox_writes_and_preserves_host():
    """评测中断时沙箱写入被丢弃，正式图、水位和 WorkMemory 保持原样。"""
    ctx = make_train_context(DictBackend())
    host_before = _context_state(ctx)
    with pytest.raises(RuntimeError, match="故意中断"):
        with isolated_evaluation(ctx, label="exception") as eval_ctx:
            eval_ctx.node_store.put(
                eval_ctx.space_id, 800_002,
                node_type=NODE_CONCEPT, tier=TIER_PRIMARY)
            eval_ctx.backend._id_pool[eval_ctx.space_id] = 800_002
            raise RuntimeError("故意中断")
    assert _context_state(ctx) == host_before


def test_evaluator_direct_host_write_is_detected():
    """故意绕过沙箱写正式 backend 时，协议必须显式失败而非静默通过。"""
    ctx = make_train_context(DictBackend())
    with pytest.raises(EvaluationIsolationError, match="正式训练状态"):
        with isolated_evaluation(ctx, label="host_write"):
            ctx.node_store.put(
                ctx.space_id, 800_003,
                node_type=NODE_CONCEPT, tier=TIER_PRIMARY)


def test_h2_returns_weights_without_mutating_host_or_teacher(monkeypatch):
    """H2 只回传 JudgeWeights，沙箱教师计数和图写不得进入正式上下文。"""
    import pure_integer_ai.experiments.evaluation_runtime as evaluation_runtime

    class CountingTeacher:
        """提供 H2 所需接口并暴露可检测的调用计数。"""

        def __init__(self, backend):
            self._b = backend
            self.call_count = 0
            self.source_id = 7

        def judge_ground_truth(self, output, dag_path, graph):
            """记录沙箱调用并返回确定的通过真值。"""
            self.call_count += 1
            return 1

    backend = DictBackend()
    teacher = CountingTeacher(backend)
    ctx = make_train_context(backend, teacher=teacher)
    host_before = _context_state(ctx)

    def fake_h2(eval_ctx, corpus, runner, *, execution=None):
        """模拟 H2 的必要局部写入和教师调用。"""
        eval_ctx.node_store.put(
            eval_ctx.space_id, 800_004,
            node_type=NODE_CONCEPT, tier=TIER_PRIMARY)
        eval_ctx.teacher.judge_ground_truth(None, None, eval_ctx.concept_graph)
        return JudgeWeights(w1=2, w2=3, w3=4, w4=5)

    monkeypatch.setattr(evaluation_runtime, "_h2_calibrate_impl", fake_h2)
    weights = _h2_calibrate(ctx, [_language_item()], DefaultRoundRunner())
    assert weights == JudgeWeights(w1=2, w2=3, w3=4, w4=5)
    assert _context_state(ctx) == host_before


def test_calibration_commits_only_dedicated_ledger_rows():
    """calibration 只提交专属台账，其余图、身份、Memory 和水位保持不变。"""
    ctx = make_train_context(DictBackend())
    before = ctx.backend.snapshot()
    rows = _run_calibration_phase(ctx, _arith_eval_items(), ctx.backend)
    after = ctx.backend.snapshot()
    assert rows
    assert after[CALIBRATION_TABLE] != before[CALIBRATION_TABLE]
    assert {
        table: table_rows
        for table, table_rows in after.items()
        if table != CALIBRATION_TABLE
    } == {
        table: table_rows
        for table, table_rows in before.items()
        if table != CALIBRATION_TABLE
    }
    assert mode_b_prevalidated(ctx.backend) is True


def test_offline_eval_returns_signal_without_committing_probe_graph():
    """offline eval 返回保持率和 E2 信号，但不提交探针观察产生的图状态。"""
    ctx = make_train_context(DictBackend())
    ctx.probe_corpus = _arith_eval_items()
    ctx.probe_set_disjoint = True
    host_before = _context_state(ctx)
    retention, passed = _run_simulated_offline_eval(ctx, [], ctx.backend)
    assert retention == 1000
    assert passed is True
    assert _context_state(ctx) == host_before


def test_floor_wrapper_preserves_real_result_and_discards_local_writes(monkeypatch):
    """floor wrapper 保留测量信号，同时丢弃 orchestrator 的局部建图。"""
    import pure_integer_ai.experiments.evaluation_runtime as evaluation_runtime

    ctx = make_train_context(DictBackend())
    host_before = _context_state(ctx)

    def fake_floor(eval_ctx, backend, graph):
        """模拟 floor 在沙箱内建图后得到非零测量。"""
        eval_ctx.node_store.put(
            eval_ctx.space_id, 800_005,
            node_type=NODE_CONCEPT, tier=TIER_PRIMARY)
        return FloorActivation(
            activation_permille=1000,
            false_positive_permille=0,
            measured=True,
            total=1,
            activated=1,
        )

    monkeypatch.setattr(evaluation_runtime, "_measure_floor_pass_impl", fake_floor)
    result = _measure_floor_pass(ctx, ctx.backend, ctx.concept_graph)
    assert result.measured is True and result.activation_permille == 1000
    assert _context_state(ctx) == host_before


def test_pre_flight_is_read_only_and_still_returns_multidimensional_report():
    """课程放量门在沙箱内运行，宿主零写且报告保留各验收维度。"""
    ctx = make_train_context(DictBackend())
    host_before = _context_state(ctx)
    report = pre_flight(
        ctx,
        [_language_item(), _language_item()],
        rounds=2,
        runner=DefaultRoundRunner(),
    )
    assert report.metrics_signal is True
    assert "anti_collapse" in report.detail
    assert "cursor_todo" in report.detail
    assert _context_state(ctx) == host_before


def test_capability_exam_is_read_only_and_does_not_persist_cursor(tmp_path):
    """统一能力考核只返回报告，不改变 backend schema/data，也不落权威训练 dump。"""
    backend = DictBackend()
    backend_before = backend.snapshot()
    config = FormalTrainConfig(
        run_dir=str(tmp_path),
        run_id="v06_capability",
        rounds_per_stage=1,
    )
    report = run_capability_exam(
        config,
        [_language_item()],
        backend=backend,
        runner=DefaultRoundRunner(),
        flat_floors=True,
    )
    assert len(report.dimensions) == 8
    assert backend.snapshot() == backend_before
    assert backend._tables == {}
    assert not list(tmp_path.rglob("*.dump"))
