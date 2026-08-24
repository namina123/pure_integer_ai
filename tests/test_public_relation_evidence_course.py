"""公开关系-证据课程的隔离和整数训练输入回归。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)


COURSE = Path(__file__).resolve().parents[1] / (
    "data/ph2/dlg_raw_public_relation_evidence_v1.jsonl.sample")


def test_relation_evidence_course_is_public_surface_only() -> None:
    pack = load_dialogue_training_pack((COURSE,))
    assert pack.split_counts == (("train", 9), ("heldout", 1), ("negative", 0))
    assert all(type(value) is int for case in pack.cases for value in case.integer_record)
    surfaces = {surface for case in pack.cases for surface in case.surfaces}
    assert "该条目又称作什么？" in surfaces
    assert "该项还有什么名称？" in surfaces
