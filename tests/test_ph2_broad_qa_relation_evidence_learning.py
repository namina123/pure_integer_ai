from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    BroadQaRelationEvidenceLearningError,
    learn_relation_evidence_model,
    relation_evidence_sha256,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_projection import (
    project_relation_evidence_value,
)
from pure_integer_ai.experiments import ph2_broad_qa_query as broad_query


_ROOT = Path(__file__).resolve().parents[1]
_COURSE = _ROOT / "data/ph2/dlg_raw_public_relation_evidence_v1.jsonl.sample"
_COURSE_V2 = _ROOT / "data/ph2/dlg_raw_public_relation_evidence_v2.jsonl.sample"
_ALIAS_QUERY = "\u8be5\u9879\u53c8\u79f0\u4f5c\u4ec0\u4e48\uff1f"
_ALIAS_EVIDENCE = "\u8be5\u9879\uff08\u67d0\u5730\uff09\u53c8\u79f0\u67d0\u540d\u79f0\u3002"


def test_relation_evidence_model_uses_train_only_and_is_integer_replayable() -> None:
    model = learn_relation_evidence_model((_COURSE,))
    assert model.case_count == 9
    assert {item.case_count for item in model.families} == {2, 5}
    assert all(type(value) is int for value in model.canonical_record())
    assert relation_evidence_sha256(model) == relation_evidence_sha256(
        learn_relation_evidence_model((_COURSE,)))
    assert model.evidence_bonus(_ALIAS_QUERY, _ALIAS_EVIDENCE) > 0


def test_relation_evidence_model_rejects_extra_fields(tmp_path: Path) -> None:
    record = {
        "evidence": {"evidence_surface": "\u67d0\u540d\u79f0\u3002"},
        "family": "PUBLIC_RELATION_EVIDENCE_ALIAS",
        "item_id": "extra-field",
        "license_id": "CC0-1.0",
        "question": {
            "question_surface": _ALIAS_QUERY,
            "typed_intent": "ASK_ENTITY",
        },
        "source_identity": "TEST",
        "split": "train",
        "answer": "\u4e0d\u5e94\u8fdb\u5165\u6a21\u578b",
    }
    path = tmp_path / "extra.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(BroadQaRelationEvidenceLearningError):
        learn_relation_evidence_model((path,))


def test_relation_evidence_model_fails_closed_for_heldout_only_and_unknown_query(
        tmp_path: Path) -> None:
    heldout = {
        "evidence": {"evidence_surface": "\u67d0\u540d\u79f0\u3002"},
        "family": "PUBLIC_RELATION_EVIDENCE_ALIAS",
        "item_id": "heldout-only",
        "license_id": "CC0-1.0",
        "question": {
            "question_surface": _ALIAS_QUERY,
            "typed_intent": "ASK_ENTITY",
        },
        "source_identity": "TEST",
        "split": "heldout",
    }
    path = tmp_path / "heldout.jsonl"
    path.write_text(json.dumps(heldout, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(BroadQaRelationEvidenceLearningError):
        learn_relation_evidence_model((path,))
    model = learn_relation_evidence_model((_COURSE,))
    assert model.evidence_bonus("\u672a\u77e5\u95ee\u6cd5\u662f\u4ec0\u4e48\uff1f", _ALIAS_EVIDENCE) == 0


def test_relation_evidence_model_can_rank_shape_without_answer_mapping() -> None:
    model = learn_relation_evidence_model((_COURSE,))
    question = "\u56db\u5ddd\u5730\u533a\uff0c\u5c06\u67d0\u9879\u79f0\u4e4b\u4e3a\u4ec0\u4e48\uff1f"
    terms = set(broad_query._script_terms("\u56db\u5ddd\u5730\u533a\uff0c\u5c06\n\u79f0\u4e4b\u4e3a\n\uff1f"))
    windows = broad_query._rank_evidence_windows(
        terms,
        "\u5206\u5e03\u4e8e\u591a\u5730\u3002",
        ("ENTITY",),
        {},
        relation_evidence_model=model,
        relation_question=question,
    )
    alias_windows = broad_query._rank_evidence_windows(
        terms,
        _ALIAS_EVIDENCE,
        ("ENTITY",),
        {},
        relation_evidence_model=model,
        relation_question=question,
    )
    assert alias_windows[0][0][0] > windows[0][0][0]


def test_relation_evidence_model_combines_independent_public_source_course() -> None:
    model = learn_relation_evidence_model((_COURSE, _COURSE_V2))
    assert model.case_count == 16
    assert {item.family for item in model.families} == {
        "PUBLIC_RELATION_EVIDENCE_AGENT",
        "PUBLIC_RELATION_EVIDENCE_ALIAS",
        "PUBLIC_RELATION_EVIDENCE_QUANTITY",
    }
    assert model.evidence_bonus(
        "\u5730\u65b9\u4e0a\u628a\u8be5\u7269\u53eb\u4f5c\u4ec0\u4e48\uff1f",
        "\u5730\u65b9\u8d44\u6599\u5c06\u8be5\u7269\u8bb0\u4f5c\u67d0\u540d\u79f0\u3002",
    ) > 0


def test_relation_evidence_projection_returns_only_unique_source_substring() -> None:
    model = learn_relation_evidence_model((_COURSE, _COURSE_V2))
    projection = project_relation_evidence_value(
        model,
        "\u56db\u5ddd\u5730\u533a\uff0c\u5c06\u67d0\u9879\u79f0\u4e4b\u4e3a\u4ec0\u4e48\uff1f",
        "\u767d\u84bf\uff08\u56db\u5ddd\uff09\uff0c\u767d\u827e\uff08\u6d59\u6c5f\uff09",
    )
    assert projection is not None
    assert projection.value == "\u767d\u84bf\uff08\u56db\u5ddd\uff09"
    assert projection.canonical_record()
    assert all(type(value) is int for value in projection.canonical_record())
    assert project_relation_evidence_value(
        model,
        "\u56db\u5ddd\u5730\u533a\uff0c\u5c06\u67d0\u9879\u79f0\u4e4b\u4e3a\u4ec0\u4e48\uff1f",
        "\u767d\u84bf\uff08\u56db\u5ddd\uff09\uff0c\u767d\u827e\uff08\u56db\u5ddd\uff09",
    ) is None
    assert project_relation_evidence_value(
        model,
        "\u67d0\u9879\u4e3a\u4ec0\u4e48\u53d1\u751f\uff1f",
        "\u767d\u84bf\uff08\u56db\u5ddd\uff09",
        ("CAUSE",),
    ) is None
