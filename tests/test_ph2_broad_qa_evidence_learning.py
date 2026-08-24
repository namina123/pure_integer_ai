"""公开课程证据词权重 learner 的确定性与边界测试。"""
from __future__ import annotations

import json

import pytest

from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_evidence_learning import (
    BroadQaEvidenceLearningError,
    evidence_learning_sha256,
    learn_evidence_term_weights,
)


def _pack(tmp_path):
    path = tmp_path / "course.jsonl"
    rows = (
        {
            "item_id": "course-a",
            "question_surface": "河水上涨的原因是什么？",
            "answer_surface": "需要查找原因证据。",
            "split": "train",
        },
        {
            "item_id": "course-b",
            "question_surface": "路面结冰的原因是什么？",
            "answer_surface": "需要查找原因证据。",
            "split": "train",
        },
        {
            "item_id": "heldout",
            "question_surface": "桥梁何时启用？",
            "answer_surface": "不进入训练词权重。",
            "split": "heldout",
        },
    )
    path.write_bytes(b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                   sort_keys=True).encode("utf-8") + b"\n"
        for row in rows))
    return load_dialogue_training_pack((path,))


def test_learner_is_integer_deterministic_and_train_only(tmp_path):
    pack = _pack(tmp_path)
    model = learn_evidence_term_weights(pack)
    assert model.case_count == 2
    assert model.weights == tuple(sorted(model.weights))
    assert model.for_query(("c:原因", "c:河水"))
    assert "c:桥梁" not in dict(model.weights)
    assert evidence_learning_sha256(model) == evidence_learning_sha256(
        learn_evidence_term_weights(pack))
    assert all(type(weight) is int and weight > 0
               for _, weight in model.weights)
    assert all(value >= 0 for value in model.canonical_record())


def test_learner_rejects_empty_train_course(tmp_path):
    path = tmp_path / "heldout.jsonl"
    path.write_text(
        '{"item_id":"x","question_surface":"桥梁何时启用？",'
        '"split":"heldout"}\n', encoding="utf-8")
    pack = load_dialogue_training_pack((path,))
    with pytest.raises(BroadQaEvidenceLearningError, match="train course"):
        learn_evidence_term_weights(pack)


def test_model_rejects_noncanonical_weights():
    from pure_integer_ai.experiments.ph2_broad_qa_evidence_learning import (
        LearnedEvidenceTermWeights,
    )
    with pytest.raises(BroadQaEvidenceLearningError):
        LearnedEvidenceTermWeights(
            (("c:x", 1), ("c:x", 2)), "0" * 64, 1)
