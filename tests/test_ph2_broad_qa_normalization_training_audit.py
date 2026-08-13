"""来源归纳 normalization TRAIN 对比证据审计测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_training_audit import (
    NORMALIZATION_TRAINING_AUDIT_KIND,
    NORMALIZATION_TRAINING_AUDIT_STATUS,
    audit_normalization_training_records,
    publish_normalization_training_audit,
    read_normalization_training_audit,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_training_audit as audit_module,
)


def _dossier(
        item_id: str,
        *,
        wikitext: str,
        gold: str,
        passage: bool,
        ) -> dict[str, object]:
    """构造 audit 纯函数所需的最小来源记录。"""
    passages = []
    if passage:
        passages.append({
            "ordinal": 1,
            "raw_end": len(wikitext),
            "raw_sha256": hashlib.sha256(
                wikitext.encode("utf-8")).hexdigest(),
            "raw_start": 0,
        })
    return {
        "item_id": item_id,
        "terminal_source": {
            "passages": passages,
            "wikitext": wikitext,
        },
        "training_source": {
            "gold_answers": [gold],
            "source_key": "TEST_SOURCE",
        },
    }


def _census(item_id: str, signal: str) -> dict[str, object]:
    """构造只允许路由、不允许变成标签的机械 census。"""
    return {
        "item_id": item_id,
        "mechanical_signal_state": signal,
    }


def test_audit_finds_mechanical_inventory_but_stays_contrastive_blocked(
        ) -> None:
    """对齐与 rewrite 库存不能冒充非等价标签或规则 Evidence。"""
    dossier = (
        _dossier("1" * 64, wikitext="畫蘭", gold="画兰", passage=True),
        _dossier("2" * 64, wikitext="原文", gold="原文", passage=True),
        _dossier("3" * 64, wikitext="頁外", gold="页外", passage=False),
        _dossier("4" * 64, wikitext="无答案", gold="目标", passage=True),
    )
    census = (
        _census("1" * 64, "MECHANICAL_SUPPORT_SIGNAL"),
        _census("2" * 64, "MECHANICAL_COUNTER_SIGNAL"),
        _census("3" * 64, "MECHANICAL_SUPPORT_SIGNAL"),
        _census("4" * 64, "UNDETERMINED"),
    )
    records, report = audit_normalization_training_records(dossier, census)
    assert len(records) == 4
    assert report["aligned_item_count"] == 3
    assert report["passage_aligned_item_count"] == 2
    assert report["rewrite_pair_count"] == 2
    assert report["contrastive_refute_evidence_count"] == 0
    assert report["semantic_non_equivalence_label_count"] == 0
    assert report["status"] == NORMALIZATION_TRAINING_AUDIT_STATUS


def test_audit_result_does_not_treat_routing_signal_as_semantic_label() -> None:
    """交换机械路由状态不改变对齐、rewrite 或 BLOCKED 结论。"""
    dossier = (
        _dossier("1" * 64, wikitext="畫蘭", gold="画兰", passage=True),
        _dossier("2" * 64, wikitext="原文", gold="原文", passage=True),
    )
    census_a = (
        _census("1" * 64, "MECHANICAL_SUPPORT_SIGNAL"),
        _census("2" * 64, "MECHANICAL_COUNTER_SIGNAL"),
    )
    census_b = (
        _census("1" * 64, "MECHANICAL_COUNTER_SIGNAL"),
        _census("2" * 64, "MECHANICAL_SUPPORT_SIGNAL"),
    )
    records_a, report_a = audit_normalization_training_records(
        dossier, census_a)
    records_b, report_b = audit_normalization_training_records(
        dossier, census_b)
    for name in (
            "aligned_item_count", "contrastive_refute_evidence_count",
            "passage_aligned_item_count", "rewrite_pair_count",
            "semantic_non_equivalence_label_count", "status"):
        assert report_a[name] == report_b[name]
    assert report_a["routing_signal_counts"] == report_b[
        "routing_signal_counts"]
    assert [item["routing_signal_state"] for item in records_a] != [
        item["routing_signal_state"] for item in records_b]


def test_publisher_reader_are_append_only_and_tamper_evident(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """正式边界只读 TRAIN，禁止覆盖并严格拒绝 manifest 漂移。"""
    dossier = (
        _dossier("1" * 64, wikitext="畫蘭", gold="画兰", passage=True),
        _dossier("2" * 64, wikitext="原文", gold="原文", passage=True),
    )
    census = (
        _census("1" * 64, "MECHANICAL_SUPPORT_SIGNAL"),
        _census("2" * 64, "MECHANICAL_COUNTER_SIGNAL"),
    )
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    monkeypatch.setattr(
        audit_module,
        "read_source_inference_learning_protocol",
        lambda path: {"manifest_sha256": "a" * 64},
    )

    def read_slice(*, protocol_dir, access_role, operator_family):
        assert Path(protocol_dir) == protocol.resolve()
        assert access_role == "LEARNER"
        assert operator_family == "NORMALIZATION_EQUIVALENCE"
        return dossier, census

    monkeypatch.setattr(
        audit_module, "read_source_inference_learning_slice", read_slice)
    target = tmp_path / "audit"
    report = publish_normalization_training_audit(
        run_root=tmp_path,
        protocol_dir=protocol,
        target_dir=target,
    )
    manifest, records = read_normalization_training_audit(target)
    assert report["manifest_sha256"] == manifest["manifest_sha256"]
    assert manifest["artifact_kind"] == NORMALIZATION_TRAINING_AUDIT_KIND
    assert manifest["validation_payload_read_count"] == 0
    assert manifest["rules_written"] == 0
    assert len(records) == 2
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_training_audit(
            run_root=tmp_path,
            protocol_dir=protocol,
            target_dir=target,
        )

    manifest_path = target / "manifest.json"
    value = json.loads(manifest_path.read_bytes())
    value["rules_written"] = 1
    manifest_path.write_bytes(canonical_json_line(value))
    with pytest.raises(BroadQaExternalDataError, match="manifest 漂移"):
        read_normalization_training_audit(target)


def test_reader_rejects_aggregate_record_mismatch(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """即使重算文件 SHA，伪造 aggregate 也不能通过严格回读。"""
    dossier = (
        _dossier("1" * 64, wikitext="畫蘭", gold="画兰", passage=True),
        _dossier("2" * 64, wikitext="原文", gold="原文", passage=True),
    )
    census = (
        _census("1" * 64, "MECHANICAL_SUPPORT_SIGNAL"),
        _census("2" * 64, "MECHANICAL_COUNTER_SIGNAL"),
    )
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    monkeypatch.setattr(
        audit_module,
        "read_source_inference_learning_protocol",
        lambda path: {"manifest_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        audit_module,
        "read_source_inference_learning_slice",
        lambda **kwargs: (dossier, census),
    )
    target = tmp_path / "audit"
    publish_normalization_training_audit(
        run_root=tmp_path,
        protocol_dir=protocol,
        target_dir=target,
    )
    manifest_path = target / "manifest.json"
    value = json.loads(manifest_path.read_bytes())
    value["passage_aligned_item_count"] -= 1
    manifest_path.write_bytes(canonical_json_line(value))
    with pytest.raises(
            BroadQaExternalDataError, match="aggregate/records 漂移"):
        read_normalization_training_audit(target)
