"""B-03 课程范围和执行遥测模块边界测试。"""
from __future__ import annotations

import json
import contextvars
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.train_execution import (
    FormalTrainExecutionStats,
    TelemetryClock,
    execution_payload,
    save_execution_metrics,
)
from pure_integer_ai.experiments.train_scope import resolve_train_scope
from pure_integer_ai.config import gates
from pure_integer_ai.training import stages
from pure_integer_ai.training.stages import (
    STAGE1_SKELETON,
    StageMetrics,
)
from pure_integer_ai.cognition.shared.types import (
    DOMAIN_TEXT,
    LANG_ZH,
    MODALITY_ARITH,
    MODALITY_LANGUAGE,
    Segment,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.verification_dispatch import (
    VERIFY_ROUTE_COMPOSES,
    VERIFY_ROUTE_NUMERIC,
    VERIFY_ROUTE_OCCURRENCE_ORDER,
    select_verification_routes,
)


def test_train_scope_resolves_stage_and_boot_delta():
    scope = resolve_train_scope(
        known_stages=(1, 2, 3),
        requested_stages=(2,),
        active_relations=frozenset({"alias", "causes"}),
        boot_relations=frozenset({"causes"}),
    )
    assert scope.training_stages == (2,)
    assert scope.relation_enabled("causes") is True
    assert scope.relation_enabled("alias") is False


def test_train_scope_rejects_duplicate_and_unknown_stage():
    with pytest.raises(ValueError, match="duplicates"):
        resolve_train_scope(
            known_stages=(1, 2), requested_stages=(1, 1),
            active_relations=None, boot_relations=None)
    with pytest.raises(ValueError, match="unknown stage"):
        resolve_train_scope(
            known_stages=(1, 2), requested_stages=(3,),
            active_relations=None, boot_relations=None)


def test_telemetry_clock_is_optional_and_integer_only():
    assert TelemetryClock().now_ns() == 0
    values = iter((10, 25))
    clock = TelemetryClock(lambda: next(values))
    assert clock.now_ns() == 10
    assert clock.now_ns() == 25


def test_execution_payload_does_not_import_formal_train_types(tmp_path):
    stats = FormalTrainExecutionStats(stage_item_runs=7, training_stages=(2,))
    routing = SimpleNamespace(to_json=lambda: {"calls": 1})
    summary = SimpleNamespace(
        total_held_out=3,
        recognized=2,
        verified=0,
        expected_verified=1,
        routing_stats=routing,
        tally_stats=None,
        structure_state=None,
    )
    payload = execution_payload(
        run_id="scope-test", execution=stats, language_summary=summary)
    assert payload["execution"]["stage_item_runs"] == 7
    assert payload["language_structure"]["routing"] == {"calls": 1}

    path = save_execution_metrics(
        run_dir=str(tmp_path), run_id="scope-test",
        execution=stats, language_summary=summary)
    with open(path, encoding="utf-8") as file:
        restored = json.load(file)
    assert restored == payload


def test_gate_overrides_are_nested_and_context_local():
    baseline = gates.CUE_EXTRACTOR_MODE
    outer = gates.push_gate_overrides({"CUE_EXTRACTOR_MODE": not baseline})
    try:
        assert gates.CUE_EXTRACTOR_MODE is (not baseline)
        isolated = contextvars.Context()
        assert isolated.run(
            lambda: gates.CUE_EXTRACTOR_MODE) is baseline
        inner = gates.push_gate_overrides({"CUE_EXTRACTOR_MODE": baseline})
        try:
            assert gates.CUE_EXTRACTOR_MODE is baseline
        finally:
            gates.reset_gate_overrides(inner)
        assert gates.CUE_EXTRACTOR_MODE is (not baseline)
    finally:
        gates.reset_gate_overrides(outer)
    assert gates.CUE_EXTRACTOR_MODE is baseline


def test_gate_overrides_reject_unknown_or_non_boolean_values():
    with pytest.raises(AttributeError):
        gates.push_gate_overrides({"NOT_A_REAL_GATE": True})
    with pytest.raises(TypeError, match="bool"):
        gates.push_gate_overrides({"CUE_EXTRACTOR_MODE": 1})


def test_stage_floor_overrides_are_nested_and_context_local():
    baseline = stages.FLOOR_GRAPH_SIZE_S1
    metrics = StageMetrics(graph_size=max(baseline - 1, 0))
    assert stages.stage_metric_gate(STAGE1_SKELETON, metrics) is False
    outer = stages.push_stage_floor_overrides({"FLOOR_GRAPH_SIZE_S1": 0})
    try:
        assert stages.stage_metric_gate(STAGE1_SKELETON, metrics) is True
        isolated = contextvars.Context()
        assert isolated.run(
            stages.stage_metric_gate, STAGE1_SKELETON, metrics) is False
        inner = stages.push_stage_floor_overrides({
            "FLOOR_GRAPH_SIZE_S1": baseline + 1,
        })
        try:
            assert stages.stage_metric_gate(STAGE1_SKELETON, metrics) is False
        finally:
            stages.reset_stage_floor_overrides(inner)
        assert stages.stage_metric_gate(STAGE1_SKELETON, metrics) is True
    finally:
        stages.reset_stage_floor_overrides(outer)
    assert stages.stage_metric_gate(STAGE1_SKELETON, metrics) is False


def test_stage_floor_overrides_reject_invalid_values():
    with pytest.raises(AttributeError):
        stages.push_stage_floor_overrides({"NOT_A_REAL_FLOOR": 0})
    with pytest.raises(TypeError, match="非负 int"):
        stages.push_stage_floor_overrides({"FLOOR_GRAPH_SIZE_S1": True})
    with pytest.raises(TypeError, match="非负 int"):
        stages.push_stage_floor_overrides({"FLOOR_GRAPH_SIZE_S1": -1})


def test_verification_dispatch_returns_all_applicable_context_routes():
    item = CollectedItem(
        modality=MODALITY_LANGUAGE,
        domain=DOMAIN_TEXT,
        lang=LANG_ZH,
        source=1,
    )
    segment = Segment(
        seg_id=0,
        modality=MODALITY_LANGUAGE,
        domain=DOMAIN_TEXT,
        lang=LANG_ZH,
        numeric_claims=[("1", "1", "1")],
        precedes_pairs=[(0, 1)],
    )
    with gates.gate_overrides({
            "NUMERIC_PROOF_MODE": True,
            "TIME_SEQ_PROOF_MODE": True,
            }):
        assert select_verification_routes(item, [segment]) == (
            VERIFY_ROUTE_NUMERIC,
            VERIFY_ROUTE_OCCURRENCE_ORDER,
        )
    with gates.gate_overrides({
            "NUMERIC_PROOF_MODE": False,
            "TIME_SEQ_PROOF_MODE": True,
            }):
        assert select_verification_routes(item, [segment]) == (
            VERIFY_ROUTE_OCCURRENCE_ORDER,
        )


def test_verification_dispatch_routes_composes_modalities_without_claims():
    item = CollectedItem(
        modality=MODALITY_ARITH,
        domain=DOMAIN_TEXT,
        lang=LANG_ZH,
        source=1,
    )
    assert select_verification_routes(item, []) == (VERIFY_ROUTE_COMPOSES,)
