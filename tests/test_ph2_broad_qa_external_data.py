"""外部中文问答来源校验、标题隔离和不可覆盖冻结测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    ExternalQaSourceFile,
    freeze_external_source_pack,
    load_external_qa_sources,
    select_external_source_pack,
)


def _write_json(path: Path, value: object) -> str:
    """写入紧凑测试来源并返回真实 SHA-256。"""
    payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _cmrc_source(tmp_path: Path) -> ExternalQaSourceFile:
    """构造含 float、坏答案和空标题异常的 CMRC 来源。"""
    path = tmp_path / "cmrc.json"
    value = [
        {
            "context_id": "c1", "context_text": "甲城位于北方。",
            "title": "甲城", "qas": [
                {"query_id": "q1", "query_text": "甲城位于哪里？",
                 "answers": ["北方"]},
                {"query_id": "q2", "query_text": "数值？",
                 "answers": [16.6]},
                {"query_id": "q3", "query_text": "缺失？",
                 "answers": ["南方"]},
            ],
        },
        {
            "context_id": "c2", "context_text": "无标题上下文。",
            "title": "", "qas": [
                {"query_id": "q4", "query_text": "什么？",
                 "answers": ["上下文"]},
            ],
        },
    ]
    sha256 = _write_json(path, value)
    return ExternalQaSourceFile(
        "CMRC2018", "test", "CMRC2018", path, sha256, "revision-a",
        "CC-BY-SA-4.0", "https://example.test/cmrc")


def _drcd_source(tmp_path: Path) -> ExternalQaSourceFile:
    """构造一条合法及一条越界 span 的 DRCD 来源。"""
    path = tmp_path / "drcd.json"
    value = {
        "version": "1.0", "data": [{
            "id": "a1", "title": "乙城", "paragraphs": [{
                "id": "p1", "context": "乙城位于南方。", "qas": [
                    {"id": "q5", "question": "乙城位于哪里？",
                     "answers": [
                         {"id": "1", "text": "南方", "answer_start": 4}
                     ]},
                    {"id": "q6", "question": "坏坐标？", "answers": [
                        {"id": "1", "text": "南方", "answer_start": -1}
                    ]},
                ],
            }],
        }]}
    sha256 = _write_json(path, value)
    return ExternalQaSourceFile(
        "DRCD", "test", "DRCD", path, sha256, "revision-b",
        "CC-BY-SA-3.0", "https://example.test/drcd")


def test_external_load_excludes_and_reports_schema_anomalies(
        tmp_path: Path) -> None:
    """float、坏答案、空标题及坏 span 均排除且逐类分账。"""
    items, report = load_external_qa_sources(
        (_cmrc_source(tmp_path), _drcd_source(tmp_path)))
    assert len(items) == 2
    assert report["anomalies"] == {
        "ANSWER_NOT_IN_CONTEXT_QUESTION": 1,
        "EMPTY_TITLE_QUESTION": 1,
        "INVALID_SPAN_QUESTION": 1,
        "NON_STRING_ANSWER_QUESTION": 1,
    }


def test_external_load_rejects_source_hash_drift(tmp_path: Path) -> None:
    """来源内容在冻结 commit 外变化时 fail closed。"""
    source = _cmrc_source(tmp_path)
    source.path.write_text("[]", encoding="utf-8")
    with pytest.raises(BroadQaExternalDataError, match="SHA"):
        load_external_qa_sources((source,))


def test_title_domain_split_is_deterministic_and_freeze_is_label_isolated(
        tmp_path: Path) -> None:
    """同标题不跨 split，question 文件不带答案且目录禁止覆盖。"""
    base_items, report = load_external_qa_sources(
        (_cmrc_source(tmp_path), _drcd_source(tmp_path)))
    expanded = []
    for source_item in base_items:
        for ordinal in range(400):
            expanded.append(source_item.__class__(
                hashlib.sha256(
                    f"{source_item.item_id}:{ordinal}".encode()).hexdigest(),
                source_item.source_key, source_item.source_partition,
                source_item.source_revision, f"q-{ordinal}",
                f"{source_item.title}-{ordinal // 2}", source_item.context,
                source_item.question, source_item.gold_answers,
                source_item.license_id, source_item.upstream_url,
            ))
    first = select_external_source_pack(
        expanded, dev_per_source=10, held_out_per_source=10)
    second = select_external_source_pack(
        reversed(expanded), dev_per_source=10, held_out_per_source=10)
    assert first == second
    assert not ({item.title_key for item in first["dev"]}
                & {item.title_key for item in first["held_out"]})
    target = tmp_path / "frozen"
    frozen = freeze_external_source_pack(
        first, target_dir=target, source_report=report)
    assert frozen["status"] == "FROZEN_NOT_RUN"
    questions = (target / "held_out.questions.jsonl").read_text(
        encoding="utf-8")
    labels = (target / "held_out.labels.jsonl").read_text(encoding="utf-8")
    assert "gold_answers" not in questions
    assert "gold_answers" in labels
    with pytest.raises(BroadQaExternalDataError, match="已存在"):
        freeze_external_source_pack(
            first, target_dir=target, source_report=report)
