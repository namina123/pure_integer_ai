"""来源归纳训练 dossier 的物化与严格回读测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_training import (
    SOURCE_INFERENCE_TRAINING_KIND,
    SOURCE_INFERENCE_TRAINING_PAYLOAD_KIND,
    SOURCE_INFERENCE_TRAINING_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_training_dossier import (
    SOURCE_INFERENCE_TRAINING_DOSSIER_RECORD_KIND,
    publish_source_inference_training_dossier,
    read_source_inference_training_dossier,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments import (
    ph2_broad_qa_source_inference_training_dossier as training_dossier,
)


def _sha(path: Path) -> str:
    """返回测试 artifact 的 SHA-256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terminal_source() -> dict[str, object]:
    """构造已由共享内核核验的终页来源。"""
    wikitext = "终页正文。"
    plain_text = "终页正文。"
    passage_text = "终页正文。"
    return {
        "attribution": "Wikipedia contributors",
        "contributor": {"id": 1, "name": "tester"},
        "license_id": "CC-BY-SA-4.0",
        "page_id": 10,
        "passages": [{
            "ordinal": 1,
            "raw_end": 5,
            "raw_sha256": hashlib.sha256(
                wikitext.encode("utf-8")).hexdigest(),
            "raw_start": 0,
            "section_title": "",
            "text": passage_text,
            "text_sha256": hashlib.sha256(
                passage_text.encode("utf-8")).hexdigest(),
        }],
        "plain_text": plain_text,
        "plain_text_sha256": hashlib.sha256(
            plain_text.encode("utf-8")).hexdigest(),
        "revision_id": 20,
        "revision_timestamp": "2026-07-01T00:00:00Z",
        "snapshot_id": "zhwiki-20260701",
        "source_url": "https://example.test/page",
        "title": "示例页",
        "wikitext": wikitext,
        "wikitext_sha256": hashlib.sha256(
            wikitext.encode("utf-8")).hexdigest(),
    }


def _inputs(root: Path):
    """构造一条冻结且尚未被 learner 读取的训练样本。"""
    item_id = "1" * 64
    question = "示例页是什么？"
    context = "旧来源上下文。"
    roster = root / "train.roster.jsonl"
    roster.write_bytes(canonical_json_line({
        "format_version": 1,
        "item_id": item_id,
        "question_sha256": hashlib.sha256(
            question.encode("utf-8")).hexdigest(),
        "record_kind": SOURCE_INFERENCE_TRAINING_RECORD_KIND,
        "source_alignment_status": "GOLD_ABSENT_FROM_TERMINAL_REVISION",
        "source_key": "CMRC2018",
        "terminal_page_id": 10,
        "terminal_revision_id": 20,
        "title_key": "示例页",
        "training_assignment": "NON_EXTRACTIVE_DISCOVERY",
    }))
    payload = root / "train.payload.jsonl"
    payload.write_bytes(canonical_json_line({
        "context": context,
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "format_version": 1,
        "gold_answers": ["答案"],
        "item_id": item_id,
        "license_id": "CC-BY-SA-4.0",
        "question": question,
        "record_kind": SOURCE_INFERENCE_TRAINING_PAYLOAD_KIND,
        "source_key": "CMRC2018",
        "source_partition": "train",
        "source_question_id": "q-1",
        "source_revision": "revision",
        "terminal_page_id": 10,
        "terminal_revision_id": 20,
        "title": "示例页",
        "title_key": "示例页",
        "training_assignment": "NON_EXTRACTIVE_DISCOVERY",
        "upstream_url": "https://example.test/source",
    }))
    manifest = root / "manifest.json"
    manifest.write_bytes(canonical_json_line({
        "artifact_kind": SOURCE_INFERENCE_TRAINING_KIND,
        "artifacts": [
            {
                "role": "training_roster_without_semantic_labels",
                "sha256": _sha(roster),
            },
            {
                "role": "training_payload_frozen_before_learner_read",
                "sha256": _sha(payload),
            },
        ],
        "format_version": 1,
        "learner_read_count_at_freeze": 0,
        "semantic_labels_written": 0,
        "status": "FROZEN_NOT_READ_NOT_LEARNED",
    }))
    selection = root / "selection.json"
    selection.write_text("selection", encoding="utf-8")
    xml = root / "source.xml.bz2"
    xml.write_bytes(b"x")
    return manifest, roster, payload, selection, xml


def test_training_dossier_materializes_without_learning(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """完整来源可回读，但 learner、规则、标签和生产查询计数均为零。"""
    manifest, roster, payload, selection, xml = _inputs(tmp_path)
    selection_value = SimpleNamespace(
        sha256=lambda: "e" * 64,
        xml_local_sha256="f" * 64,
        xml_compressed_size_bytes=1,
    )
    monkeypatch.setattr(
        training_dossier, "read_broad_qa_target_selection",
        lambda _path: selection_value)
    monkeypatch.setattr(
        training_dossier, "materialize_terminal_sources",
        lambda *args, **kwargs: {10: _terminal_source()})

    target = tmp_path / "training-dossier"
    report = publish_source_inference_training_dossier(
        run_root=tmp_path,
        roster_manifest_path=manifest,
        roster_path=roster,
        training_payload_path=payload,
        terminal_selection_path=selection,
        xml_path=xml,
        target_dir=target,
    )
    assert report["dossier_record_count"] == 1
    assert report["learner_read_count"] == 0
    assert report["rules_written"] == 0
    assert report["semantic_labels_written"] == 0
    assert report["production_query_runs"] == 0
    assert report["status"] == "MATERIALIZED_UNREAD_UNLEARNED"
    values = read_source_inference_training_dossier(
        target / "training.dossier.jsonl")
    assert values[0]["record_kind"] == (
        SOURCE_INFERENCE_TRAINING_DOSSIER_RECORD_KIND)
    assert values[0]["terminal_source"]["wikitext"] == "终页正文。"
    assert values[0]["training_source"]["gold_answers"] == ["答案"]


def test_training_dossier_rejects_commitment_drift_and_overwrite(
        tmp_path: Path,
        ) -> None:
    """冻结输入被改写或输出已存在时必须失败关闭。"""
    manifest, roster, payload, selection, xml = _inputs(tmp_path)
    roster.write_bytes(roster.read_bytes() + b"\n")
    with pytest.raises(BroadQaExternalDataError, match="commitment"):
        publish_source_inference_training_dossier(
            run_root=tmp_path,
            roster_manifest_path=manifest,
            roster_path=roster,
            training_payload_path=payload,
            terminal_selection_path=selection,
            xml_path=xml,
            target_dir=tmp_path / "training-dossier",
        )
    target = tmp_path / "exists"
    target.mkdir()
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_source_inference_training_dossier(
            run_root=tmp_path,
            roster_manifest_path=manifest,
            roster_path=roster,
            training_payload_path=payload,
            terminal_selection_path=selection,
            xml_path=xml,
            target_dir=target,
        )


def test_training_dossier_reader_rejects_inner_hash_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """规范重写单条正文仍不能绕过内部 context/passage/hash 承诺。"""
    manifest, roster, payload, selection, xml = _inputs(tmp_path)
    selection_value = SimpleNamespace(
        sha256=lambda: "e" * 64,
        xml_local_sha256="f" * 64,
        xml_compressed_size_bytes=1,
    )
    monkeypatch.setattr(
        training_dossier, "read_broad_qa_target_selection",
        lambda _path: selection_value)
    monkeypatch.setattr(
        training_dossier, "materialize_terminal_sources",
        lambda *args, **kwargs: {10: _terminal_source()})
    target = tmp_path / "training-dossier"
    publish_source_inference_training_dossier(
        run_root=tmp_path,
        roster_manifest_path=manifest,
        roster_path=roster,
        training_payload_path=payload,
        terminal_selection_path=selection,
        xml_path=xml,
        target_dir=target,
    )
    dossier = target / "training.dossier.jsonl"
    value = json.loads(dossier.read_bytes())
    value["terminal_source"]["passages"][0]["text"] = "篡改正文"
    dossier.write_bytes(canonical_json_line(value))
    with pytest.raises(BroadQaExternalDataError, match="passage hash"):
        read_source_inference_training_dossier(dossier)
