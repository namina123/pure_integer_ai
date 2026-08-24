from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    learn_relation_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_marker_evidence_learning import (
    RelationMarkerEvidenceLearningError,
    learn_relation_marker_evidence_model,
    project_marker_relation_value,
    relation_marker_evidence_sha256,
)


_ROOT = Path(__file__).resolve().parents[1]
_RELATION_COURSES = (
    _ROOT / "data/ph2/dlg_raw_public_relation_evidence_v1.jsonl.sample",
    _ROOT / "data/ph2/dlg_raw_public_relation_evidence_v2.jsonl.sample",
)
_MARKER_COURSE = _ROOT / (
    "data/ph2/dlg_raw_public_relation_marker_evidence_v1.jsonl.sample")
_MARKER_COURSE_V2 = _ROOT / (
    "data/ph2/dlg_raw_public_relation_marker_evidence_v2.jsonl.sample")


def test_marker_model_learns_tail_boundary_and_replays_as_integers() -> None:
    relation = learn_relation_evidence_model(_RELATION_COURSES)
    marker = learn_relation_marker_evidence_model((_MARKER_COURSE,))
    replay = learn_relation_marker_evidence_model((_MARKER_COURSE,))
    assert marker.case_count == 4
    assert relation_marker_evidence_sha256(marker) == relation_marker_evidence_sha256(replay)
    assert all(type(value) is int for value in marker.canonical_record())
    assert project_marker_relation_value(
        relation, marker, "该事项由谁负责？", "档案说明，由某机构负责。",
    ) == ("某机构", 6, 9)
    assert project_marker_relation_value(
        relation, marker, "该项数量是多少？", "清单记录，该项数量为某数。",
    ) == ("某数", 10, 12)


def test_marker_projection_fails_closed_for_multiple_candidates() -> None:
    relation = learn_relation_evidence_model(_RELATION_COURSES)
    marker = learn_relation_marker_evidence_model((_MARKER_COURSE,))
    assert project_marker_relation_value(
        relation, marker, "该项数量是多少？",
        "清单记录，该项数量为某数。统计资料显示，该项数量为另一数。",
    ) is None


def test_marker_v2_independent_tail_and_overlapping_marker_fail_closed() -> None:
    relation = learn_relation_evidence_model(_RELATION_COURSES)
    marker = learn_relation_marker_evidence_model(
        (_MARKER_COURSE, _MARKER_COURSE_V2))
    records = [json.loads(line) for line in _MARKER_COURSE_V2.read_text(
        encoding="utf-8").splitlines() if json.loads(line)["split"] == "train"]
    agent = records[0]
    quantity = records[3]
    overlapping = records[2]
    assert project_marker_relation_value(
        relation, marker, agent["question"]["question_surface"],
        agent["evidence"]["evidence_surface"],
    ) == ("某单位", 6, 9)
    assert project_marker_relation_value(
        relation, marker, quantity["question"]["question_surface"],
        quantity["evidence"]["evidence_surface"],
    ) == ("某数", 9, 11)
    assert project_marker_relation_value(
        relation, marker, overlapping["question"]["question_surface"],
        overlapping["evidence"]["evidence_surface"],
    ) is None


def test_marker_course_rejects_bad_span_and_heldout_only(tmp_path: Path) -> None:
    records = [json.loads(line) for line in _MARKER_COURSE.read_text(
        encoding="utf-8").splitlines()]
    records[0]["roles"][1]["start"] += 1
    bad_span = tmp_path / "bad-span.jsonl"
    bad_span.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                  for item in records) + "\n", encoding="utf-8")
    with pytest.raises(RelationMarkerEvidenceLearningError):
        learn_relation_marker_evidence_model((bad_span,))

    heldout = tmp_path / "heldout.jsonl"
    for item in records:
        item["split"] = "heldout"
    heldout.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                  for item in records) + "\n", encoding="utf-8")
    with pytest.raises(RelationMarkerEvidenceLearningError):
        learn_relation_marker_evidence_model((heldout,))


def test_marker_course_rejects_unknown_split(tmp_path: Path) -> None:
    record = json.loads(_MARKER_COURSE.read_text(encoding="utf-8").splitlines()[0])
    record["split"] = "negative"
    path = tmp_path / "invalid-split.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(RelationMarkerEvidenceLearningError):
        learn_relation_marker_evidence_model((path,))
