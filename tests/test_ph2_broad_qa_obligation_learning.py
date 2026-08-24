"""typed obligation learner 的确定性、隔离和 fail-closed 测试。"""
from __future__ import annotations

import json

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_obligation_learning import (
    BroadQaObligationLearningError,
    learn_typed_obligations,
    typed_obligation_sha256,
)


def _path(tmp_path):
    path = tmp_path / "typed.jsonl"
    rows = (
        {"item_id": "time-a", "split": "train", "question":
         {"question_surface": "设施何时启用？", "typed_intent": "ASK_EVENT_TIME"}},
        {"item_id": "time-b", "split": "train", "question":
         {"question_surface": "项目何时开放？", "typed_intent": "ASK_EVENT_TIME"}},
        {"item_id": "heldout", "split": "heldout", "question":
         {"question_surface": "设施何时关闭？", "typed_intent": "ASK_EVENT_TIME"}},
    )
    path.write_bytes(b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows))
    return path


def test_typed_obligation_is_integer_deterministic_and_train_only(tmp_path):
    model = learn_typed_obligations((_path(tmp_path),))
    assert model.case_count == 2
    assert model.answer_kinds("新设施何时开放？") == ("TIME",)
    assert model.answer_kinds("新设施什么时候开放？") == ()
    assert typed_obligation_sha256(model) == typed_obligation_sha256(model)
    assert all(type(value) is int for value in model.canonical_record())


def test_typed_obligation_fails_closed_without_typed_train(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text(
        '{"item_id":"x","split":"heldout","question":'
        '{"question_surface":"设施何时启用？"}}\n', encoding="utf-8")
    with pytest.raises(BroadQaObligationLearningError, match="typed_intent"):
        learn_typed_obligations((path,))
