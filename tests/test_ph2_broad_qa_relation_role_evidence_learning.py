from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    learn_relation_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_role_evidence_learning import (
    RelationRoleEvidenceLearningError,
    learn_relation_role_evidence_model,
    project_qualified_relation_value,
    relation_role_evidence_sha256,
)


_ROOT = Path(__file__).resolve().parents[1]
_RELATION_COURSES = (
    _ROOT / "data/ph2/dlg_raw_public_relation_evidence_v1.jsonl.sample",
    _ROOT / "data/ph2/dlg_raw_public_relation_evidence_v2.jsonl.sample",
)
_ROLE_COURSE = _ROOT / "data/ph2/dlg_raw_public_relation_role_evidence_v1.jsonl.sample"


def test_role_model_learns_public_shape_and_projects_unique_qualified_item() -> None:
    relation = learn_relation_evidence_model(_RELATION_COURSES)
    role = learn_relation_role_evidence_model((_ROLE_COURSE,))
    assert role.case_count == 3
    assert relation_role_evidence_sha256(role) == relation_role_evidence_sha256(
        learn_relation_role_evidence_model((_ROLE_COURSE,)))
    value = project_qualified_relation_value(
        relation, role,
        "\u56db\u5ddd\u5730\u533a\uff0c\u5c06\u4e94\u6708\u827e\u79f0\u4e4b\u4e3a\u4ec0\u4e48\uff1f",
        "\u827e\uff08\u540d\u533b\u522b\u5f55\u3001\u53f0\u6e7e\u690d\u7269\u5fd7\uff09\uff0c\u767d\u84bf\uff08\u56db\u5ddd\uff09\uff0c\u767d\u827e\uff08\u6d59\u6c5f\uff09",
        anchor_text="\u4e94\u6708\u827e",
    )
    assert value is not None
    assert value[0] == "\u767d\u84bf\uff08\u56db\u5ddd\uff09"
    assert value[1] >= 0 and value[2] > value[1]


def test_role_projection_fails_closed_for_ambiguous_or_unqualified_input() -> None:
    relation = learn_relation_evidence_model(_RELATION_COURSES)
    role = learn_relation_role_evidence_model((_ROLE_COURSE,))
    assert project_qualified_relation_value(
        relation, role,
        "\u56db\u5ddd\u5730\u533a\uff0c\u5c06\u4e94\u6708\u827e\u79f0\u4e4b\u4e3a\u4ec0\u4e48\uff1f",
        "\u767d\u84bf\uff08\u56db\u5ddd\uff09\uff0c\u767d\u827e\uff08\u56db\u5ddd\uff09",
        anchor_text="\u4e94\u6708\u827e",
    ) is None
    assert project_qualified_relation_value(
        relation, role,
        "\u67d0\u9879\u662f\u4ec0\u4e48\uff1f",
        "\u767d\u84bf\uff08\u56db\u5ddd\uff09",
        anchor_text="\u67d0\u9879",
    ) is None


def test_role_course_rejects_non_train_split(tmp_path: Path) -> None:
    source = _ROLE_COURSE.read_bytes().replace(b'"split":"heldout"', b'"split":"negative"')
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(source)
    with pytest.raises(RelationRoleEvidenceLearningError):
        learn_relation_role_evidence_model((path,))
