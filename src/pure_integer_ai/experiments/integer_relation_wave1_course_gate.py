"""Plan-level Wave 1 course gate, distinct from graph topology admission."""
from __future__ import annotations

from pure_integer_ai.experiments.integer_relation_connected_course import (
    ConnectedCourseCompilation,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_RELATION_PROFILES,
)


_SPLIT_CODES = {"train": 1, "held_out": 2, "negative": 3, "conflict": 4}
_REQUIRED_RELATION_AXES = {
    "event_time": "PRECEDES",
    "cause": "CAUSES",
    "property": "PROPERTY",
    "reference_topic": "PURE_ALIAS_REFERS",
    "comparison_quantity": "SIMILAR_ANTONYM",
}
_MIN_CASES_PER_SPLIT = 20
_MIN_DISTINCT_OPERATIONS = 8
_MIN_TRAIN_EXAMPLES_PER_OPERATION = 2
_MIN_INDEPENDENT_NOVEL_HELD_OUT = 20


class Wave1CourseGateError(ValueError):
    """Course identities or admission evidence cannot be trusted."""


def _substages(records: tuple) -> tuple[str, ...]:
    by_kind = {
        profile.relation_kind: profile.substage_key
        for profile in W06_RELATION_PROFILES.values()
    }
    try:
        return tuple(sorted({by_kind[record.relation_kind] for record in records}))
    except KeyError as error:
        raise Wave1CourseGateError("unregistered relation kind") from error


def audit_integer_relation_wave1_course(
        compilation: ConnectedCourseCompilation,
        ) -> dict[str, object]:
    """Reject a narrow course even when its 20 held-out topologies are novel.

    This gate audits course coverage only. It cannot prove operation behavior,
    online language grounding, or a production QueryState path.
    """
    if not isinstance(compilation, ConnectedCourseCompilation):
        raise TypeError("connected compilation is required")
    if (compilation.train_course.split != "train"
            or set(compilation.evaluation_courses) != set(_SPLIT_CODES) - {"train"}
            or compilation.source_audit.get("status")
            != "PASS_SOURCE_AND_FOUR_WAY_ISOLATION_ONLY"):
        raise Wave1CourseGateError("source or four-way ownership is unqualified")
    courses = {
        "train": compilation.train_course,
        **compilation.evaluation_courses,
    }
    counts = {split: len(courses[split].cases) for split in _SPLIT_CODES}
    actual_keys = {
        split: {(case.case_key if split == "train" else case.assignment.case_key)
                for case in courses[split].cases}
        for split in _SPLIT_CODES
    }
    expected_keys = {split: set() for split in _SPLIT_CODES}
    train_operations: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
    for assignment in compilation.assignments:
        split = next((name for name, code in _SPLIT_CODES.items()
                      if code == assignment.split_code), None)
        if split is None or assignment.case_key in expected_keys[split]:
            raise Wave1CourseGateError("case split or identity is invalid")
        expected_keys[split].add(assignment.case_key)
        if split == "train":
            train_operations.setdefault(assignment.operation_key, set()).add(
                assignment.input_values)
    if expected_keys != actual_keys:
        raise Wave1CourseGateError("integer assignment/course case keys drifted")
    if any(courses[split].split != split for split in _SPLIT_CODES):
        raise Wave1CourseGateError("course split identity drifted")

    train_stages = set(_substages(compilation.train_course.records))
    held_stages = set(_substages(compilation.evaluation_courses["held_out"].records))
    novel_count = compilation.topology_admission.get(
        "independent_novel_held_out_graph_count")
    topology_pass = (
        compilation.topology_admission.get("status")
        == "PASS_STRUCTURAL_ADMISSION_ONLY"
        and type(novel_count) is int
        and novel_count >= _MIN_INDEPENDENT_NOVEL_HELD_OUT
    )
    issues = []
    for split, count in counts.items():
        if count < _MIN_CASES_PER_SPLIT:
            issues.append(f"INSUFFICIENT_{split.upper()}_CASES")
    if not topology_pass:
        issues.append("TOPOLOGY_NOVELTY_UNQUALIFIED")
    for axis, substage in _REQUIRED_RELATION_AXES.items():
        if substage not in train_stages:
            issues.append(f"TRAIN_MISSING_{axis.upper()}")
        if substage not in held_stages:
            issues.append(f"HELD_OUT_MISSING_{axis.upper()}")
    if len(train_operations) < _MIN_DISTINCT_OPERATIONS:
        issues.append("INSUFFICIENT_TRAIN_OPERATIONS")
    for operation, inputs in train_operations.items():
        if len(inputs) < _MIN_TRAIN_EXAMPLES_PER_OPERATION:
            issues.append("TRAIN_OPERATION_HAS_SINGLE_INPUT")
    return {
        "format": "PURE_INTEGER_RELATION_WAVE1_COURSE_GATE_V1",
        "status": "BLOCKED_COURSE_COVERAGE" if issues else "PASS_COVERAGE_ONLY",
        "issues": sorted(set(issues)),
        "case_counts": counts,
        "minimum_cases_per_split": _MIN_CASES_PER_SPLIT,
        "train_operation_count": len(train_operations),
        "train_operation_example_counts": [
            {"operation_key": list(key), "distinct_input_count": len(values)}
            for key, values in sorted(train_operations.items())
        ],
        "required_relation_substages": dict(_REQUIRED_RELATION_AXES),
        "train_substages": sorted(train_stages),
        "held_out_substages": sorted(held_stages),
        "independent_novel_held_out_graph_count": novel_count,
        "minimum_independent_novel_held_out": _MIN_INDEPENDENT_NOVEL_HELD_OUT,
        "operation_surface_qualification": "UNKNOWN_not_proven_by_case_labels",
        "semantic_behavior_qualification": "UNKNOWN_requires_production_trace",
        "topic_quantity_specific_qualification": "UNKNOWN_not_established_by_W06_relation_substages",
        "production_authorized": 0,
        "free_dialogue_complete": 0,
    }


__all__ = ["Wave1CourseGateError", "audit_integer_relation_wave1_course"]
