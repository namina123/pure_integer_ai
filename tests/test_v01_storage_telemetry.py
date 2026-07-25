"""V-01 存储遥测的语义隔离、后端一致性和作用域分桶测试。"""
from __future__ import annotations

import contextvars
import json
from types import SimpleNamespace

import pytest

from pure_integer_ai.storage import bootstrap, discipline as disc
from pure_integer_ai.storage.backend import (
    DictBackend,
    SQLiteBackend,
    TYPE_INT,
)
from pure_integer_ai.storage.telemetry import (
    BackendTelemetryCollector,
    collect_backend_telemetry,
    record_diagnostic_event,
    suppress_backend_telemetry,
    telemetry_scope,
)
from pure_integer_ai.storage.occurrence import (
    OccurrenceStorageRecord,
    OccurrenceStore,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    concept_assertion,
    document_scope,
)
from pure_integer_ai.cognition.shared.scoped_persistence import (
    ScopedIdentityStore,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
)
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.training.stages import STAGE2_CAUSES_ABS


class _IncrementingSource:
    """为墙钟和工作集测试提供严格递增的确定性整数。"""

    def __init__(self, step: int) -> None:
        self.value = 0
        self.step = step

    def __call__(self) -> int:
        """返回下一次严格整数采样。"""
        self.value += self.step
        return self.value


class _QueryOnlyRunner:
    """在 round scope 内发起一条可定位查询，不写认知状态。"""

    def run_round(self, ctx, item, stage, round_id):
        """读取 edge 行数并保持 observe-only 返回契约。"""
        ctx.backend.count("edge")
        return None


def _backend_factories():
    """返回 V-01 必须保持计数一致的两个正式后端工厂。"""
    return (DictBackend, SQLiteBackend)


def _new_backend(factory):
    """建立允许覆盖完整 CRUD 路径的非核心测试表。"""
    backend = factory()
    backend.register_table(
        "telemetry_test",
        [("key", TYPE_INT), ("value", TYPE_INT)],
        disc.DISC_MUTABLE_MONOTONE,
        [("key",)],
        core=False,
    )
    return backend


def _exercise_backend(backend) -> None:
    """以固定顺序执行可跨后端比较的存储操作。"""
    backend.insert("telemetry_test", {"key": 1, "value": 10})
    backend.insert("telemetry_test", {"key": 2, "value": 20})
    assert backend.select("telemetry_test", {"key": 1}) == [
        {"key": 1, "value": 10},
    ]
    assert backend.count("telemetry_test") == 2
    assert backend.update(
        "telemetry_test", {"key": 2}, {"value": 21}) == 1
    assert backend.delete("telemetry_test", {"key": 1}) == 1


def _operation_totals(payload):
    """把 JSON 列表投影为便于断言的操作计数字典。"""
    return {
        row["operation"]: (row["calls"], row["rows"], row["failures"])
        for row in payload["operation_totals"]
    }


@pytest.mark.parametrize("factory", _backend_factories())
def test_telemetry_enabled_does_not_change_backend_snapshot(factory):
    """开启遥测前后执行同一操作序，canonical 后端状态必须完全一致。"""
    without = _new_backend(factory)
    with_telemetry = _new_backend(factory)
    try:
        _exercise_backend(without)
        collector = BackendTelemetryCollector()
        with collect_backend_telemetry(collector):
            _exercise_backend(with_telemetry)
        assert with_telemetry.snapshot() == without.snapshot()
    finally:
        without.close()
        with_telemetry.close()


def test_dict_and_sqlite_report_identical_operation_counts():
    """Dict/SQLite 对同一抽象操作必须给出相同调用数和影响行数。"""
    payloads = []
    for factory in _backend_factories():
        backend = _new_backend(factory)
        collector = BackendTelemetryCollector()
        try:
            with collect_backend_telemetry(collector):
                with telemetry_scope(
                        caller="training",
                        query="unit_query",
                        source_key=(1, 2, 3),
                        occurrence_key=(4, 5),
                        assertion_key=(6, 7),
                        scope_key=(8, 9),
                        stage=2,
                        round_id=3,
                        item_index=4):
                    _exercise_backend(backend)
            payloads.append(collector.to_json())
        finally:
            backend.close()

    expected = {
        "insert": (2, 2, 0),
        "select": (1, 1, 0),
        "count": (1, 2, 0),
        "update": (1, 1, 0),
        "delete": (1, 1, 0),
    }
    assert _operation_totals(payloads[0]) == expected
    assert _operation_totals(payloads[1]) == expected
    assert payloads[0] == payloads[1]


def test_nested_and_independent_contexts_do_not_leak_scopes():
    """嵌套 scope 精确复位，独立 Context 不继承活动采集器。"""
    backend = _new_backend(DictBackend)
    collector = BackendTelemetryCollector()
    try:
        with collect_backend_telemetry(collector):
            with telemetry_scope(caller="training", source_key=(10,)):
                backend.count("telemetry_test")
                with telemetry_scope(
                        caller="evaluation:probe",
                        query="probe",
                        evaluation=True):
                    backend.count("telemetry_test")
                backend.count("telemetry_test")

                isolated = contextvars.Context()
                isolated.run(backend.count, "telemetry_test")

        payload = collector.to_json()
        assert _operation_totals(payload) == {"count": (3, 0, 0)}
        callers = {
            row["value"]: _operation_totals({
                "operation_totals": [
                    {
                        "operation": operation["operation"],
                        "calls": operation["calls"],
                        "rows": operation["rows"],
                        "failures": operation["failures"],
                    }
                    for operation in row["operations"]
                ],
            })
            for row in payload["by_dimension"]["caller"]
        }
        assert callers["training"]["count"] == (2, 0, 0)
        assert callers["evaluation:probe"]["count"] == (1, 0, 0)
        evaluations = {
            row["value"]: sum(op["calls"] for op in row["operations"])
            for row in payload["by_dimension"]["evaluation"]
        }
        assert evaluations == {False: 2, True: 1}
    finally:
        backend.close()


def test_suppression_excludes_diagnostic_queries():
    """诊断自身的表规模查询不能污染被测后端计数。"""
    backend = _new_backend(DictBackend)
    collector = BackendTelemetryCollector()
    try:
        with collect_backend_telemetry(collector):
            backend.count("telemetry_test")
            with suppress_backend_telemetry():
                backend.count("telemetry_test")
        assert _operation_totals(collector.to_json()) == {
            "count": (1, 0, 0),
        }
    finally:
        backend.close()


def test_diagnostic_events_keep_scope_without_backend_operations():
    """纯 CPU 热点也必须能按当前 query scope 统计调用次数。"""
    collector = BackendTelemetryCollector()
    with collect_backend_telemetry(collector):
        with telemetry_scope(caller="training", query="pure_hotspot"):
            record_diagnostic_event("hotspot.example")
            record_diagnostic_event("hotspot.example", 2)

    payload = collector.to_json()
    assert payload["event_totals"] == [{
        "kind": "hotspot.example",
        "count": 3,
    }]
    query = next(
        row for row in payload["by_dimension"]["query"]
        if row["value"] == "pure_hotspot"
    )
    assert query["operations"] == []
    assert query["events"] == [{
        "kind": "hotspot.example",
        "count": 3,
    }]


def test_formal_train_reports_stage_scope_growth_and_separate_file(tmp_path):
    """正式训练遥测能定位 item 查询，并把阶段和表增长写入独立报告。"""
    clock = _IncrementingSource(10)
    working_set = _IncrementingSource(100)
    config = FormalTrainConfig(
        run_dir=str(tmp_path),
        run_id="v01_formal",
        rounds_per_stage=1,
        active_training_stages=(STAGE2_CAUSES_ABS,),
        persist_graph_dump=False,
        telemetry_clock_ns=clock,
        telemetry_enabled=True,
        telemetry_working_set_bytes=working_set,
    )
    backend = DictBackend()
    item = CollectedItem(tokens=["甲", "乙"])
    item.diagnostic_candidates = SimpleNamespace(candidates=(1, 2))
    result = formal_train(
        config,
        [item],
        backend=backend,
        runner=_QueryOnlyRunner(),
    )

    assert len(result.execution.stages) == 1
    stage = result.execution.stages[0]
    assert stage.stage == STAGE2_CAUSES_ABS
    assert stage.item_count == 1
    assert stage.candidate_count == 2
    assert stage.elapsed_ns > 0
    assert result.execution.peak_working_set_bytes > 0
    assert any(
        row["table"] == "edge"
        for row in result.execution.run_table_growth)

    backend_payload = result.execution.backend_telemetry
    assert backend_payload is not None
    assert backend_payload["candidate_totals"] == [{
        "kind": "diagnostic_candidates",
        "observations": 1,
        "candidates": 2,
    }]
    round_scopes = [
        row for row in backend_payload["scopes"]
        if row["scope"]["query"] == "round_item"
    ]
    assert len(round_scopes) == 1
    assert round_scopes[0]["scope"]["source"] is not None
    assert round_scopes[0]["scope"]["scope"] is not None
    assert any(
        operation["operation"] == "count"
        and operation["table"] == "edge"
        and operation["calls"] == 1
        for operation in round_scopes[0]["operations"]
    )

    execution_path = tmp_path / "v01_formal" / "execution.json"
    with open(execution_path, encoding="utf-8") as file:
        restored = json.load(file)
    assert restored["execution"]["backend_telemetry"] == backend_payload
    assert restored["execution"]["graph_dump_calls"] == 0
    assert not list(tmp_path.rglob("*.dump"))


def test_formal_train_telemetry_toggle_preserves_canonical_snapshot(tmp_path):
    """顶层训练只因诊断开关不同，最终后端快照仍必须 bit-identical。"""
    snapshots = []
    for enabled in (False, True):
        backend = DictBackend()
        config = FormalTrainConfig(
            run_dir=str(tmp_path / ("on" if enabled else "off")),
            run_id="same_run",
            rounds_per_stage=1,
            active_training_stages=(STAGE2_CAUSES_ABS,),
            persist_graph_dump=False,
            telemetry_enabled=enabled,
            telemetry_clock_ns=_IncrementingSource(10),
            telemetry_working_set_bytes=_IncrementingSource(100),
        )
        formal_train(
            config,
            [CollectedItem(tokens=["甲", "乙"])],
            backend=backend,
            runner=_QueryOnlyRunner(),
        )
        snapshots.append(backend.snapshot())
    assert snapshots[0] == snapshots[1]


def test_evaluation_calls_are_bucketed_away_from_training():
    """同一采集器内的正式查询和评测沙箱查询必须落入不同 caller 桶。"""
    ctx = make_train_context(DictBackend())
    collector = BackendTelemetryCollector()
    host_before = ctx.backend.snapshot()
    with collect_backend_telemetry(collector):
        with telemetry_scope(caller="training", query="training"):
            ctx.backend.count("edge")
        with isolated_evaluation(ctx, label="probe") as eval_ctx:
            eval_ctx.backend.count("edge")

    payload = collector.to_json()
    caller_rows = {
        row["value"]: row
        for row in payload["by_dimension"]["caller"]
    }
    assert "training" in caller_rows
    assert "evaluation:probe" in caller_rows
    assert sum(
        operation["calls"]
        for operation in caller_rows["training"]["operations"]
    ) == 1
    assert sum(
        operation["calls"]
        for operation in caller_rows["evaluation:probe"]["operations"]
    ) >= 1
    assert ctx.backend.snapshot() == host_before


def test_occurrence_and_assertion_writers_attach_identity_buckets():
    """真实 occurrence 与 assertion writer 必须给后端操作附加可定位身份。"""
    backend = DictBackend()
    bootstrap(backend)
    collector = BackendTelemetryCollector()
    source = SourceRef(
        1,
        2,
        3,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )
    scope = document_scope(source)
    assertion = concept_assertion(
        1,
        (1, 1),
        (1, 2),
        scope=scope,
        provenance_kind=1,
    )
    with collect_backend_telemetry(collector):
        ScopedIdentityStore(backend).register_assertion(assertion)
        OccurrenceStore(backend).add(OccurrenceStorageRecord(
            1, 10, 11, 12,
            0, 1, 0,
            0, 0, 0,
            0,
            0, 0, 0,
        ))

    payload = collector.to_json()
    assertion_rows = payload["by_dimension"]["assertion"]
    occurrence_rows = payload["by_dimension"]["occurrence"]
    assert any(row["value"] == list(assertion.stable_key())
               for row in assertion_rows)
    assert any(row["value"] == [1, 10] for row in occurrence_rows)
    queries = {
        row["value"] for row in payload["by_dimension"]["query"]
    }
    assert "assertion.register" in queries
    assert "occurrence.add" in queries
    assert "occurrence.read" in queries
