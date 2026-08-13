"""来源归纳 feasibility audit 的机械边界测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_decision import (
    SOURCE_INFERENCE_REVIEW_WORKSHEET_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_feasibility import (
    audit_source_inference_feasibility,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _worksheet(path: Path) -> Path:
    """写入直命中、两段可拼与不可达三个固定 item。"""
    records = (
        ("1" * 64, "EXTRACTIVE_CANDIDATE", ["甲乙"], ["甲乙"]),
        ("2" * 64, "NON_EXTRACTIVE_REVIEW", ["甲中乙"], ["甲乙"]),
        ("3" * 64, "NON_EXTRACTIVE_REVIEW", ["甲"], ["甲乙"]),
    )
    path.write_bytes(b"".join(canonical_json_line({
        "allowed_decisions": [
            "EXTRACTIVE", "SOURCE_DERIVABLE", "SOURCE_CONFLICT", "REJECT"],
        "assignment": assignment,
        "decision": "UNREVIEWED",
        "format_version": 1,
        "gold_answers": gold,
        "item_id": item_id,
        "old_context_answer_snippets": [],
        "question": "问题",
        "record_kind": SOURCE_INFERENCE_REVIEW_WORKSHEET_RECORD_KIND,
        "review_context_sha256": "a" * 64,
        "terminal_passages": [{"text": text} for text in passages],
        "terminal_title": "页面",
        "terminal_wikitext_sha256": "b" * 64,
    }) for item_id, assignment, passages, gold in records))
    return path


def test_feasibility_audit_writes_no_semantic_decisions(
        tmp_path: Path) -> None:
    """机械可拼接库存不能被写成四态 decision 或能力 PASS。"""
    worksheet = _worksheet(tmp_path / "worksheet.jsonl")
    target = tmp_path / "audit"
    report = audit_source_inference_feasibility(
        worksheet, target_dir=target)
    assert report["normalized_direct_hit_count"] == 1
    assert report["segment_count_distribution"] == {
        "1": 1, "2": 1, "UNREACHABLE": 1}
    assert report["semantic_decisions_written"] == 0
    assert report["status"] == "BLOCKED_NO_PREBOUND_RULE_PACK"
    records = tuple(json.loads(line) for line in (
        target / "feasibility.records.jsonl").read_text(
            encoding="utf-8").splitlines())
    assert [item["minimum_source_segment_count"] for item in records] == [
        1, 2, None]
    assert all("decision" not in item for item in records)


def test_prebound_rule_pack_changes_readiness_not_semantic_labels(
        tmp_path: Path) -> None:
    """预绑定规则包只解除机械阻塞，不自动生成可推导标签。"""
    worksheet = _worksheet(tmp_path / "worksheet.jsonl")
    report = audit_source_inference_feasibility(
        worksheet,
        target_dir=tmp_path / "audit",
        prebound_rule_pack_sha256s=("c" * 64,),
    )
    assert report["status"] == "READY_FOR_INDEPENDENT_RULE_VALIDATION"
    assert report["semantic_decisions_written"] == 0
    with pytest.raises(BroadQaExternalDataError, match="已存在|输出"):
        audit_source_inference_feasibility(
            worksheet,
            target_dir=tmp_path / "audit",
            prebound_rule_pack_sha256s=("c" * 64,),
        )


def test_reviewed_or_noncanonical_worksheet_is_rejected(
        tmp_path: Path) -> None:
    """审计只消费原始未审 worksheet，不接受回填过的语义结果。"""
    worksheet = _worksheet(tmp_path / "worksheet.jsonl")
    values = [json.loads(line) for line in worksheet.read_text(
        encoding="utf-8").splitlines()]
    values[0]["decision"] = "EXTRACTIVE"
    worksheet.write_bytes(b"".join(canonical_json_line(item) for item in values))
    with pytest.raises(BroadQaExternalDataError, match="worksheet 漂移"):
        audit_source_inference_feasibility(
            worksheet, target_dir=tmp_path / "audit")
