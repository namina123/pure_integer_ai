"""外部证据预测/评分的标签隔离与 span 完整性测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_runtime import (
    predict_external_evidence,
    score_external_evidence,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _pack(tmp_path: Path) -> tuple[Path, Path, Path]:
    """构造一题可答的 questions/labels 输入。"""
    question = {
        "context": "甲城位于北方。乙城位于南方。",
        "context_sha256": "",
        "format_version": 1,
        "item_id": "a" * 64,
        "license_id": "CC-BY-SA-4.0",
        "question": "甲城位于哪里？",
        "record_kind": "PH2_BROAD_QA_EXTERNAL_QUESTION_V1",
        "source_key": "CMRC2018", "source_partition": "dev",
        "source_question_id": "q1", "source_revision": "r1",
        "split": "dev", "title": "甲城",
        "upstream_url": "https://example.test/source",
    }
    import hashlib
    question["context_sha256"] = hashlib.sha256(
        question["context"].encode()).hexdigest()
    label = {
        "format_version": 1, "gold_answers": ["北方"],
        "item_id": question["item_id"],
        "record_kind": "PH2_BROAD_QA_EXTERNAL_LABEL_V1", "split": "dev",
    }
    qpath, lpath = tmp_path / "questions.jsonl", tmp_path / "labels.jsonl"
    qpath.write_bytes(canonical_json_line(question))
    lpath.write_bytes(canonical_json_line(label))
    return qpath, lpath, tmp_path / "predictions.jsonl"


def test_predict_does_not_read_labels_and_score_requires_exact_inventory(
        tmp_path: Path) -> None:
    """预测可独立运行，评分严格要求 questions/predictions/labels 同一题集。"""
    questions, labels, predictions = _pack(tmp_path)
    report = predict_external_evidence(
        questions, predictions_path=predictions)
    assert report["prediction_count"] == 1
    prediction = json.loads(predictions.read_text(encoding="utf-8"))
    assert "gold_answers" not in prediction
    aggregate = score_external_evidence(
        questions, predictions, labels,
        aggregate_path=tmp_path / "aggregate.json", scope="DEVELOPMENT")
    assert aggregate["status"] == "PASS"
    assert aggregate["evidence_hit_count"] == 1


def test_score_rejects_tampered_prediction_span(tmp_path: Path) -> None:
    """预测证据被改写或 hash 不一致时不得计入有效引用。"""
    questions, labels, predictions = _pack(tmp_path)
    predict_external_evidence(questions, predictions_path=predictions)
    value = json.loads(predictions.read_text(encoding="utf-8"))
    value["evidence_text"] = "伪造"
    predictions.write_bytes(canonical_json_line(value))
    aggregate = score_external_evidence(
        questions, predictions, labels,
        aggregate_path=tmp_path / "aggregate.json", scope="DEVELOPMENT")
    assert aggregate["status"] == "FAIL"
    assert aggregate["citation_valid_count"] == 0


def test_prediction_and_aggregate_are_not_overwritable(tmp_path: Path) -> None:
    """重复发布必须拒绝，避免悄悄替换正式证据。"""
    questions, labels, predictions = _pack(tmp_path)
    predict_external_evidence(questions, predictions_path=predictions)
    with pytest.raises(Exception, match="禁止覆盖"):
        predict_external_evidence(questions, predictions_path=predictions)
    aggregate_path = tmp_path / "aggregate.json"
    score_external_evidence(
        questions, predictions, labels,
        aggregate_path=aggregate_path, scope="DEVELOPMENT")
    with pytest.raises(Exception, match="禁止覆盖"):
        score_external_evidence(
            questions, predictions, labels,
            aggregate_path=aggregate_path, scope="DEVELOPMENT")
