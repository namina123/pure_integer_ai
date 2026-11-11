from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.integer_relation_connected_course import (
    compile_connected_course,
)
from pure_integer_ai.experiments.integer_relation_wave1_course_gate import (
    Wave1CourseGateError,
    audit_integer_relation_wave1_course,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/ph2/authored_connected_wave1_facts_v1.jsonl.sample"
CASES = ROOT / "data/ph2/authored_connected_wave1_cases_v1.jsonl.sample"


@pytest.fixture
def compilation(tmp_path):
    return compile_connected_course(SOURCE, CASES, tmp_path)


def test_full_wave1_gate_rejects_structural_only_connected_course(compilation):
    report = audit_integer_relation_wave1_course(compilation)

    assert compilation.topology_admission["status"] == "PASS_STRUCTURAL_ADMISSION_ONLY"
    assert report["status"] == "BLOCKED_COURSE_COVERAGE"
    assert report["case_counts"] == {
        "train": 21, "held_out": 20, "negative": 1, "conflict": 1,
    }
    assert report["independent_novel_held_out_graph_count"] >= 20
    assert {"INSUFFICIENT_NEGATIVE_CASES", "INSUFFICIENT_CONFLICT_CASES",
            "TRAIN_MISSING_CAUSE", "HELD_OUT_MISSING_CAUSE",
            "TRAIN_MISSING_PROPERTY", "HELD_OUT_MISSING_PROPERTY",
            "TRAIN_OPERATION_HAS_SINGLE_INPUT"} <= set(report["issues"])
    assert report["train_operation_count"] == 8
    assert sorted(row["distinct_input_count"]
                  for row in report["train_operation_example_counts"]) == [
                      1, 1, 2, 2, 3, 3, 4, 5]
    assert report["production_authorized"] == 0
    assert report["free_dialogue_complete"] == 0


def test_wave1_gate_rejects_assignment_case_drift(compilation):
    first = replace(compilation.assignments[0], case_key=(92002, 999))
    broken = replace(compilation, assignments=(first, *compilation.assignments[1:]))

    with pytest.raises(Wave1CourseGateError, match="drifted"):
        audit_integer_relation_wave1_course(broken)


def test_wave1_gate_rejects_unqualified_source(compilation):
    broken = replace(compilation, source_audit={"status": "UNKNOWN"})

    with pytest.raises(Wave1CourseGateError, match="source"):
        audit_integer_relation_wave1_course(broken)
