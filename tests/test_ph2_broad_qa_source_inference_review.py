"""来源内归纳 review dossier 的物化边界测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_family import (
    SOURCE_INFERENCE_REVIEW_PAYLOAD_KIND,
    SOURCE_INFERENCE_ROSTER_KIND,
    SOURCE_INFERENCE_ROSTER_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_review import (
    SOURCE_INFERENCE_DOSSIER_RECORD_KIND,
    publish_source_inference_review_dossier,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_source_inference_review as review,
)


def _sha(path: Path) -> str:
    """返回小型测试 artifact 的 SHA-256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(root: Path):
    """构造一条冻结 roster、payload 和 manifest。"""
    item_id = "1" * 64
    question = "示例页是什么？"
    context = "旧来源上下文。"
    roster = root / "review.roster.jsonl"
    roster.write_bytes(canonical_json_line({
        "assignment": "NON_EXTRACTIVE_REVIEW",
        "format_version": 1,
        "item_id": item_id,
        "question_sha256": hashlib.sha256(
            question.encode("utf-8")).hexdigest(),
        "record_kind": SOURCE_INFERENCE_ROSTER_RECORD_KIND,
        "source_alignment_status": "GOLD_ABSENT_FROM_TERMINAL_REVISION",
        "source_key": "CMRC2018",
        "terminal_page_id": 10,
        "terminal_revision_id": 20,
        "title_key": "示例页",
    }))
    payload = root / "review.payload.jsonl"
    payload.write_bytes(canonical_json_line({
        "assignment": "NON_EXTRACTIVE_REVIEW",
        "context": context,
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "format_version": 1,
        "gold_answers": ["旧答案"],
        "item_id": item_id,
        "license_id": "CC-BY-SA-4.0",
        "question": question,
        "record_kind": SOURCE_INFERENCE_REVIEW_PAYLOAD_KIND,
        "source_key": "CMRC2018",
        "source_partition": "train",
        "source_question_id": "q-1",
        "source_revision": "revision",
        "terminal_page_id": 10,
        "terminal_revision_id": 20,
        "title": "示例页",
        "title_key": "示例页",
        "upstream_url": "https://example.test/source",
    }))
    manifest = root / "manifest.json"
    manifest.write_bytes(canonical_json_line({
        "artifact_kind": SOURCE_INFERENCE_ROSTER_KIND,
        "artifacts": [
            {"role": "review_roster_without_labels", "sha256": _sha(roster)},
            {"role": "private_development_review_payload", "sha256": _sha(payload)},
        ],
        "format_version": 1,
        "status": "FROZEN_UNREVIEWED_NOT_RUN",
    }))
    selection = root / "selection.json"
    selection.write_text("selection", encoding="utf-8")
    xml = root / "source.xml.bz2"
    xml.write_bytes(b"x")
    return manifest, roster, payload, selection, xml


def test_dossier_materializes_full_source_without_writing_decisions(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """终页全文、passage 和来源身份写入 dossier，但 decision/query 计数为零。"""
    manifest, roster, payload, selection, xml = _inputs(tmp_path)
    selected_page = SimpleNamespace(page_id=10)
    selection_value = SimpleNamespace(
        selected_pages=(selected_page,),
        source_key="ZHWIKIPEDIA_20260701",
        snapshot_id="zhwiki-20260701",
        xml_compressed_size_bytes=1,
        xml_local_sha256="a" * 64,
        sha256=lambda: "b" * 64,
    )
    inspection = SimpleNamespace(
        page_id=10,
        revision_id=20,
        title="示例页",
        timestamp="2026-07-01T00:00:00Z",
        contributor_json='{"id":1,"name":"tester"}',
        redirect_title=None,
        wikitext="首段正文。\n\n次段正文。",
        text_sha256=hashlib.sha256(
            "首段正文。\n\n次段正文。".encode("utf-8")).hexdigest(),
    )
    passage = SimpleNamespace(
        ordinal=1,
        raw_start=0,
        raw_end=5,
        raw_sha256="c" * 64,
        text="首段正文。",
        text_sha256="d" * 64,
        section_title="",
    )
    monkeypatch.setattr(
        review, "read_broad_qa_target_selection", lambda _path: selection_value)
    monkeypatch.setattr(
        review, "iter_broad_qa_selected_page_inspections",
        lambda *args, **kwargs: iter((inspection,)))
    monkeypatch.setattr(
        review, "project_broad_qa_plain_text", lambda _value: "首段正文。 次段正文。")
    monkeypatch.setattr(
        review, "project_broad_qa_passages", lambda *args, **kwargs: (passage,))

    target = tmp_path / "dossier"
    report = publish_source_inference_review_dossier(
        run_root=tmp_path,
        roster_manifest_path=manifest,
        roster_path=roster,
        review_payload_path=payload,
        terminal_selection_path=selection,
        xml_path=xml,
        target_dir=target,
        worker_count=4,
    )
    assert report["dossier_record_count"] == 1
    assert report["terminal_page_count"] == 1
    assert report["production_query_runs"] == 0
    assert report["review_decisions_written"] == 0
    assert report["status"] == "MATERIALIZED_UNREVIEWED_NOT_RUN"
    record = json.loads((target / "review.dossier.jsonl").read_text(
        encoding="utf-8"))
    assert record["record_kind"] == SOURCE_INFERENCE_DOSSIER_RECORD_KIND
    assert record["terminal_source"]["wikitext"] == inspection.wikitext
    assert record["terminal_source"]["passages"][0]["text"] == "首段正文。"
    assert record["review_source"]["gold_answers"] == ["旧答案"]


def test_dossier_rejects_roster_commitment_drift_and_overwrite(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """roster artifact 被改写或 target 已存在时必须失败关闭。"""
    manifest, roster, payload, selection, xml = _inputs(tmp_path)
    roster.write_bytes(roster.read_bytes() + b"\n")
    with pytest.raises(BroadQaExternalDataError, match="commitment"):
        publish_source_inference_review_dossier(
            run_root=tmp_path,
            roster_manifest_path=manifest,
            roster_path=roster,
            review_payload_path=payload,
            terminal_selection_path=selection,
            xml_path=xml,
            target_dir=tmp_path / "dossier",
        )

    target = tmp_path / "exists"
    target.mkdir()
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_source_inference_review_dossier(
            run_root=tmp_path,
            roster_manifest_path=manifest,
            roster_path=roster,
            review_payload_path=payload,
            terminal_selection_path=selection,
            xml_path=xml,
            target_dir=target,
        )
