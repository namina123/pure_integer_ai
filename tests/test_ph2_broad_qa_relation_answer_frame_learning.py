from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_relation_answer_frame_learning import (
    RelationAnswerFrameLearningError,
    learn_relation_answer_frame_model,
    relation_answer_frame_sha256,
    render_relation_answer_frame,
)


_ROOT = Path(__file__).resolve().parents[1]
_COURSE = _ROOT / (
    "data/ph2/dlg_raw_public_relation_answer_frame_v1.jsonl.sample")


def test_relation_answer_frame_is_cross_source_and_integer_replayable() -> None:
    model = learn_relation_answer_frame_model((_COURSE,))
    replay = learn_relation_answer_frame_model((_COURSE,))
    assert model.case_count == 6
    assert relation_answer_frame_sha256(model) == relation_answer_frame_sha256(replay)
    assert all(type(value) is int for value in model.canonical_record())
    assert render_relation_answer_frame(
        model, "PUBLIC_RELATION_EVIDENCE_ALIAS", "五月艾", "白蒿（四川）",
    ) == "五月艾又称白蒿（四川）。"
    assert render_relation_answer_frame(
        model, "PUBLIC_RELATION_EVIDENCE_QUANTITY", "某事项", "12",
    ) == "某事项数量为12。"


def test_relation_answer_frame_unknown_family_fails_closed() -> None:
    model = learn_relation_answer_frame_model((_COURSE,))
    assert render_relation_answer_frame(model, "UNKNOWN", "某项", "某值") is None
    assert render_relation_answer_frame(model, None, "某项", "某值") is None


def test_relation_answer_frame_rejects_single_source_or_bad_span(tmp_path: Path) -> None:
    records = [json.loads(line) for line in _COURSE.read_text(
        encoding="utf-8").splitlines()]
    records = records[:1] + [dict(records[0], item_id="only-source-2")]
    single = tmp_path / "single.jsonl"
    single.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                  for item in records) + "\n", encoding="utf-8")
    with pytest.raises(RelationAnswerFrameLearningError):
        learn_relation_answer_frame_model((single,))

    bad = json.loads(_COURSE.read_text(encoding="utf-8").splitlines()[0])
    bad["roles"][1]["start"] = 3
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(RelationAnswerFrameLearningError):
        learn_relation_answer_frame_model((path,))
