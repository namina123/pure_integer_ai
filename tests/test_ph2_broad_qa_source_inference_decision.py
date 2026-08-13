"""来源内归纳 decision ledger、worksheet 和发布门测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_decision import (
    BroadQaSourceInferenceDecision,
    BroadQaSourceInferenceDecisionLedger,
    parse_source_inference_decision_ledger,
    publish_source_inference_review_worksheet,
    validate_source_inference_decision_ledger,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_family import (
    SOURCE_INFERENCE_ROSTER_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_review import (
    SOURCE_INFERENCE_DOSSIER_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _sha_text(value: str) -> str:
    """返回测试文本的 SHA-256。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _files(root: Path):
    """构造一个抽取项和一个非抽取项的固定 roster/dossier。"""
    first_id = "1" * 64
    second_id = "2" * 64
    roster_records = (
        {
            "assignment": "EXTRACTIVE_CANDIDATE",
            "format_version": 1,
            "item_id": first_id,
            "question_sha256": _sha_text("抽取问题"),
            "record_kind": SOURCE_INFERENCE_ROSTER_RECORD_KIND,
            "source_alignment_status": "SOURCE_ALIGNED",
            "source_key": "CMRC2018",
            "terminal_page_id": 1,
            "terminal_revision_id": 11,
            "title_key": "抽取页",
        },
        {
            "assignment": "NON_EXTRACTIVE_REVIEW",
            "format_version": 1,
            "item_id": second_id,
            "question_sha256": _sha_text("非抽取问题"),
            "record_kind": SOURCE_INFERENCE_ROSTER_RECORD_KIND,
            "source_alignment_status": "GOLD_ABSENT_FROM_TERMINAL_REVISION",
            "source_key": "DRCD",
            "terminal_page_id": 2,
            "terminal_revision_id": 22,
            "title_key": "非抽取页",
        },
    )
    terminal_first = _sha_text("抽取终页")
    terminal_second = _sha_text("非抽取终页")
    review_second = _sha_text("旧上下文")
    dossier_records = (
        {
            "assignment": "EXTRACTIVE_CANDIDATE",
            "format_version": 1,
            "item_id": first_id,
            "record_kind": SOURCE_INFERENCE_DOSSIER_RECORD_KIND,
            "review_source": {
                "context": "抽取答案位于旧上下文",
                "context_sha256": _sha_text("抽取答案位于旧上下文"),
                "gold_answers": ["抽取答案"],
                "question": "抽取问题",
                "source_key": "CMRC2018",
            },
            "roster_commitment": {"title_key": "抽取页"},
            "terminal_source": {
                "passages": [],
                "plain_text": "终页包含抽取答案。",
                "title": "抽取页",
                "wikitext_sha256": terminal_first,
            },
        },
        {
            "assignment": "NON_EXTRACTIVE_REVIEW",
            "format_version": 1,
            "item_id": second_id,
            "record_kind": SOURCE_INFERENCE_DOSSIER_RECORD_KIND,
            "review_source": {
                "context": "旧上下文",
                "context_sha256": review_second,
                "gold_answers": ["旧答案"],
                "question": "非抽取问题",
                "source_key": "DRCD",
            },
            "roster_commitment": {"title_key": "非抽取页"},
            "terminal_source": {
                "passages": [],
                "plain_text": "终页只有不同表述。",
                "title": "非抽取页",
                "wikitext_sha256": terminal_second,
            },
        },
    )
    roster = root / "roster.jsonl"
    roster.write_bytes(b"".join(
        canonical_json_line(item) for item in roster_records))
    dossier = root / "dossier.jsonl"
    dossier.write_bytes(b"".join(
        canonical_json_line(item) for item in dossier_records))
    return (
        roster,
        dossier,
        first_id,
        second_id,
        terminal_first,
        terminal_second,
        review_second,
    )


def _ledger(
        roster: Path,
        dossier: Path,
        first_id: str,
        second_decision: BroadQaSourceInferenceDecision,
        terminal_first: str,
        ) -> BroadQaSourceInferenceDecisionLedger:
    """构造完整覆盖两个 item 的规范 ledger。"""
    first = BroadQaSourceInferenceDecision(
        first_id,
        "EXTRACTIVE",
        "EXACT_TERMINAL_SUPPORT",
        (terminal_first,),
        None,
        _sha_text("抽取审阅记录"),
    )
    return BroadQaSourceInferenceDecisionLedger(
        hashlib.sha256(roster.read_bytes()).hexdigest(),
        hashlib.sha256(dossier.read_bytes()).hexdigest(),
        tuple(sorted((first, second_decision), key=lambda item: item.item_id)),
    )


def test_ledger_round_trip_and_reject_validation(tmp_path: Path) -> None:
    """规范 ledger 可逐字回读，REJECT 保持来源不足而不伪造能力。"""
    (roster, dossier, first_id, second_id, terminal_first,
     terminal_second, _) = _files(tmp_path)
    reject = BroadQaSourceInferenceDecision(
        second_id,
        "REJECT",
        "INSUFFICIENT_SOURCE_SUPPORT",
        (terminal_second,),
        None,
        _sha_text("来源不足"),
    )
    ledger = _ledger(
        roster, dossier, first_id, reject, terminal_first)
    restored = parse_source_inference_decision_ledger(
        ledger.canonical_bytes())
    assert restored == ledger
    report = validate_source_inference_decision_ledger(
        restored, roster_path=roster, dossier_path=dossier)
    assert report["decision_counts"] == {
        "EXTRACTIVE": 1,
        "SOURCE_DERIVABLE": 0,
        "SOURCE_CONFLICT": 0,
        "REJECT": 1,
    }
    assert report["verified_inference_record_count"] == 0


def test_conflict_requires_both_distinct_source_commitments(
        tmp_path: Path) -> None:
    """冲突层必须同时保留旧上下文和终页的不同承诺。"""
    (roster, dossier, first_id, second_id, terminal_first,
     terminal_second, review_second) = _files(tmp_path)
    conflict = BroadQaSourceInferenceDecision(
        second_id,
        "SOURCE_CONFLICT",
        "DISTINCT_SOURCE_ASSERTION_CONFLICT",
        tuple(sorted((review_second, terminal_second))),
        None,
        _sha_text("来源版本冲突"),
    )
    ledger = _ledger(
        roster, dossier, first_id, conflict, terminal_first)
    report = validate_source_inference_decision_ledger(
        ledger, roster_path=roster, dossier_path=dossier)
    assert report["decision_counts"]["SOURCE_CONFLICT"] == 1

    with pytest.raises(BroadQaExternalDataError, match="至少两个"):
        BroadQaSourceInferenceDecision(
            second_id,
            "SOURCE_CONFLICT",
            "DISTINCT_SOURCE_ASSERTION_CONFLICT",
            (terminal_second,),
            None,
            _sha_text("错误冲突"),
        )


def test_derivable_requires_an_actual_verified_inference_record(
        tmp_path: Path) -> None:
    """只有 SHA 文本而没有规范 inference record 文件时不得进入可推导层。"""
    (roster, dossier, first_id, second_id, terminal_first,
     terminal_second, _) = _files(tmp_path)
    derivable = BroadQaSourceInferenceDecision(
        second_id,
        "SOURCE_DERIVABLE",
        "AUDITABLE_SOURCE_DERIVATION",
        (terminal_second,),
        "f" * 64,
        _sha_text("待验证推导"),
    )
    ledger = _ledger(
        roster, dossier, first_id, derivable, terminal_first)
    with pytest.raises(BroadQaExternalDataError, match="来源边界"):
        validate_source_inference_decision_ledger(
            ledger, roster_path=roster, dossier_path=dossier)


def test_ledger_requires_exact_roster_coverage(tmp_path: Path) -> None:
    """少一条或增加未冻结 item 都不得形成 reviewed ledger。"""
    (roster, dossier, first_id, _, terminal_first,
     _, _) = _files(tmp_path)
    only = BroadQaSourceInferenceDecision(
        first_id,
        "EXTRACTIVE",
        "EXACT_TERMINAL_SUPPORT",
        (terminal_first,),
        None,
        _sha_text("只有一条"),
    )
    ledger = BroadQaSourceInferenceDecisionLedger(
        hashlib.sha256(roster.read_bytes()).hexdigest(),
        hashlib.sha256(dossier.read_bytes()).hexdigest(),
        (only,),
    )
    with pytest.raises(BroadQaExternalDataError, match="精确覆盖"):
        validate_source_inference_decision_ledger(
            ledger, roster_path=roster, dossier_path=dossier)


def test_worksheet_is_unreviewed_and_non_overwritable(tmp_path: Path) -> None:
    """worksheet 只整理来源，不预写任何四态 decision。"""
    _, dossier, *_ = _files(tmp_path)
    target = tmp_path / "worksheet.jsonl"
    report = publish_source_inference_review_worksheet(
        dossier, target_path=target)
    assert report["item_count"] == 2
    assert report["review_decisions_written"] == 0
    records = tuple(map(json.loads, target.read_text(
        encoding="utf-8").splitlines()))
    assert all(item["decision"] == "UNREVIEWED" for item in records)
    assert all(item["allowed_decisions"] == [
        "EXTRACTIVE", "SOURCE_DERIVABLE", "SOURCE_CONFLICT", "REJECT"]
        for item in records)
    with pytest.raises(BroadQaExternalDataError, match="输出或预算"):
        publish_source_inference_review_worksheet(
            dossier, target_path=target)
