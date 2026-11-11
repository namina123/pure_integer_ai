from __future__ import annotations

import json
from dataclasses import replace

import pytest

from pure_integer_ai.experiments.integer_relation_graph_split_course import (
    IntegerRelationGraphSplitCase,
    IntegerRelationGraphSplitCourse,
    IntegerRelationGraphSplitCourseError,
    build_integer_graph_novelty_splits,
    read_integer_relation_graph_split_course,
    validate_integer_relation_graph_split_courses,
    write_integer_relation_graph_split_course,
)
from tests.test_integer_graph_novelty import _chain, _fork, _record, _event


OWNER = (91662, 999, 20261011, 1)
SOURCE = (91660, 999, 20261011, 1)
LICENSE = (91661, 999, 1)


def _course(split: str, records, text: str, ordinal: int):
    rows = tuple(records)
    case = IntegerRelationGraphSplitCase(
        (1, ordinal), tuple(map(ord, text)), ((12, ordinal),),
        tuple(row.observation_key for row in rows),
    )
    return IntegerRelationGraphSplitCourse(
        split, OWNER, SOURCE, LICENSE, (case,),
        tuple(sorted(rows, key=lambda row: row.stable_key())),
    )


def _courses():
    return {
        "train": _course("train", _chain(100, 100), "训练关系", 1),
        "held_out": _course("held_out", _fork(200, 200), "未见关系", 2),
        "negative": _course(
            "negative", (_record(300, _event(300, 1), _event(300, 2)),),
            "无证据", 3),
        "conflict": _course(
            "conflict", (_record(400, _event(400, 1), _event(400, 2)),),
            "冲突", 4),
    }


def test_split_course_round_trip_and_real_topology_admission(tmp_path):
    courses = _courses()
    restored = {}
    for split, course in courses.items():
        path = tmp_path / f"{split}.int.json"
        digest = write_integer_relation_graph_split_course(path, course)
        assert len(digest) == 64
        restored[split] = read_integer_relation_graph_split_course(path)
        assert restored[split] == course

    report = validate_integer_relation_graph_split_courses(
        restored, minimum_novel_held_out=1, minimum_relation_families=1)
    assert report["split_course_contract"] == "PASS"
    assert report["status"] == "PASS_STRUCTURAL_ADMISSION_ONLY"
    assert report["novel_held_out_case_count"] == 1
    assert report["free_dialogue_complete"] == 0
    assert report["label_owner_qualification"].startswith("UNKNOWN")
    assert report["unauthorized_qualification"].startswith("UNKNOWN")

    serialized = (tmp_path / "train.int.json").read_text(encoding="ascii")
    assert "raw_text" not in serialized
    assert '"surface":' not in serialized
    assert '"answer"' not in serialized


@pytest.mark.parametrize("field,value", [
    ("split_code", True), ("case_count", True), ("record_count", True),
    ("free_dialogue_complete", False),
])
def test_cold_split_metadata_requires_strict_integers(tmp_path, field, value):
    course = _course(
        "train", (_record(100, _event(100, 1), _event(100, 2)),), "训练", 1)
    path = tmp_path / "train.int.json"
    write_integer_relation_graph_split_course(path, course)
    payload = json.loads(path.read_bytes())
    payload[field] = value
    path.write_bytes((json.dumps(payload, sort_keys=True,
                                separators=(",", ":")) + "\n").encode("ascii"))
    with pytest.raises(IntegerRelationGraphSplitCourseError):
        read_integer_relation_graph_split_course(path)


def test_cold_split_does_not_sort_away_record_order_drift(tmp_path):
    path = tmp_path / "train.int.json"
    write_integer_relation_graph_split_course(path, _courses()["train"])
    payload = json.loads(path.read_bytes())
    payload["records"].reverse()
    path.write_bytes((json.dumps(payload, sort_keys=True,
                                separators=(",", ":")) + "\n").encode("ascii"))
    with pytest.raises(IntegerRelationGraphSplitCourseError, match="sorted"):
        read_integer_relation_graph_split_course(path)


def test_split_course_rejects_unowned_or_repeated_record(tmp_path):
    course = _courses()["train"]
    with pytest.raises(IntegerRelationGraphSplitCourseError, match="unique"):
        duplicate = replace(
            course.cases[0],
            record_keys=course.cases[0].record_keys
            + course.cases[0].record_keys[:1],
        )
        IntegerRelationGraphSplitCourse(
            course.split, course.owner_identity, course.source_identity,
            course.license_identity, (duplicate,), course.records,
        )

    path = tmp_path / "train.int.json"
    write_integer_relation_graph_split_course(path, course)
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["cases"][0]["record_keys"][0] = [999, 999]
    path.write_bytes((json.dumps(
        payload, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")) + "\n").encode("ascii"))
    with pytest.raises(IntegerRelationGraphSplitCourseError, match="reference"):
        read_integer_relation_graph_split_course(path)


def test_split_course_join_rejects_identity_drift_and_case_collision():
    courses = _courses()
    courses["held_out"] = replace(
        courses["held_out"], owner_identity=(999, 1, 1))
    with pytest.raises(IntegerRelationGraphSplitCourseError, match="owner_identity"):
        build_integer_graph_novelty_splits(courses)

    courses = _courses()
    courses["held_out"] = replace(
        courses["held_out"],
        cases=(replace(courses["held_out"].cases[0], case_key=(1, 1)),),
    )
    with pytest.raises(IntegerRelationGraphSplitCourseError, match="cross split"):
        build_integer_graph_novelty_splits(courses)

    courses = _courses()
    courses["train"] = replace(courses["train"], split="held_out")
    with pytest.raises(IntegerRelationGraphSplitCourseError, match="name drift"):
        build_integer_graph_novelty_splits(courses)


def test_split_course_requires_all_four_splits():
    courses = _courses()
    del courses["negative"]
    with pytest.raises(IntegerRelationGraphSplitCourseError, match="all four"):
        build_integer_graph_novelty_splits(courses)
