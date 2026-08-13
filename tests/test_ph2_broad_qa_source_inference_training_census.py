"""来源归纳训练 operator 机械盘点的边界测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_training_census import (
    audit_source_inference_training_operator_census,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_training_dossier import (
    SOURCE_INFERENCE_TRAINING_DOSSIER_KIND,
    SOURCE_INFERENCE_TRAINING_DOSSIER_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _sha(path: Path) -> str:
    """返回测试 artifact 的 SHA-256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
        ordinal: int,
        *,
        source_key: str,
        wikitext: str,
        passage: str,
        gold: str,
        ) -> dict[str, object]:
    """构造一条具有终页字节与 gold 的未学习 dossier record。"""
    item_id = format(ordinal, "064x")
    title = f"示例页{ordinal}"
    question = "问题？"
    context = "旧上下文"
    return {
        "format_version": 1,
        "item_id": item_id,
        "record_kind": SOURCE_INFERENCE_TRAINING_DOSSIER_RECORD_KIND,
        "roster_commitment": {
            "question_sha256": hashlib.sha256(
                question.encode("utf-8")).hexdigest(),
            "source_alignment_status": "SOURCE_ALIGNED",
            "title_key": title,
        },
        "terminal_source": {
            "attribution": "Wikipedia contributors",
            "contributor": {"id": 1},
            "license_id": "CC-BY-SA-4.0",
            "page_id": ordinal,
            "passages": [{
                "ordinal": 1,
                "raw_end": len(passage),
                "raw_sha256": hashlib.sha256(
                    wikitext[:len(passage)].encode("utf-8")).hexdigest(),
                "raw_start": 0,
                "section_title": "",
                "text": passage,
                "text_sha256": hashlib.sha256(
                    passage.encode("utf-8")).hexdigest(),
            }],
            "plain_text": wikitext,
            "plain_text_sha256": hashlib.sha256(
                wikitext.encode("utf-8")).hexdigest(),
            "revision_id": 100 + ordinal,
            "revision_timestamp": "2026-07-01T00:00:00Z",
            "snapshot_id": "zhwiki-20260701",
            "source_url": "https://example.test/page",
            "title": title,
            "wikitext": wikitext,
            "wikitext_sha256": hashlib.sha256(
                wikitext.encode("utf-8")).hexdigest(),
        },
        "training_assignment": "EXTRACTIVE_REFERENCE",
        "training_source": {
            "context": context,
            "context_sha256": hashlib.sha256(
                context.encode("utf-8")).hexdigest(),
            "gold_answers": [gold],
            "license_id": "CC-BY-SA-4.0",
            "question": question,
            "source_key": source_key,
            "source_partition": "train",
            "source_question_id": f"q-{ordinal}",
            "source_revision": "revision",
            "title": title,
            "upstream_url": "https://example.test/source",
        },
    }


def _inputs(root: Path):
    """构造包含支持、反例和不可判定机械信号的 dossier。"""
    dossier = root / "training.dossier.jsonl"
    records = (
        _record(
            1,
            source_key="CMRC2018",
            wikitext="公元2020年，示例（别名甲）、成员乙。",
            passage="示例（别名甲）、成员乙。",
            gold="别名甲",
        ),
        _record(
            2,
            source_key="DRCD",
            wikitext="甲（旧称）、乙、丙。",
            passage="甲乙丙",
            gold="乙",
        ),
    )
    dossier.write_bytes(b"".join(canonical_json_line(item) for item in records))
    manifest = root / "manifest.json"
    manifest.write_bytes(canonical_json_line({
        "artifact_kind": SOURCE_INFERENCE_TRAINING_DOSSIER_KIND,
        "dossier_bytes": dossier.stat().st_size,
        "dossier_record_count": 2,
        "dossier_sha256": _sha(dossier),
        "format_version": 1,
        "learner_read_count": 0,
        "rules_written": 0,
        "semantic_labels_written": 0,
        "status": "MATERIALIZED_UNREAD_UNLEARNED",
    }))
    return manifest, dossier


def test_training_census_reports_mechanical_signals_without_labels(
        tmp_path: Path,
        ) -> None:
    """每题每 family 有机械三态记录，但没有 operator 预分配或规则写入。"""
    manifest, dossier = _inputs(tmp_path)
    target = tmp_path / "census"
    report = audit_source_inference_training_operator_census(
        run_root=tmp_path,
        dossier_manifest_path=manifest,
        dossier_path=dossier,
        target_dir=target,
    )
    assert report["item_count"] == 2
    assert report["record_count"] == 12
    assert report["learner_read_count"] == 0
    assert report["operator_preassigned_count"] == 0
    assert report["rules_written"] == 0
    assert report["semantic_labels_written"] == 0
    assert report["status"] == "MECHANICAL_CENSUS_ONLY_NOT_LEARNED"
    assert report["operator_family_state_counts"][
        "PARENTHETICAL_EXPANSION"] == {
            "MECHANICAL_SUPPORT_SIGNAL": 1,
            "MECHANICAL_COUNTER_SIGNAL": 1,
            "UNDETERMINED": 0,
        }
    assert report["operator_family_state_counts"][
        "EXPLICIT_UNIT_ERA_FORMAT_MAPPING"] == {
            "MECHANICAL_SUPPORT_SIGNAL": 0,
            "MECHANICAL_COUNTER_SIGNAL": 0,
            "UNDETERMINED": 2,
        }
    records = tuple(map(json.loads, (
        target / "operator-census.records.jsonl").read_text(
            encoding="utf-8").splitlines()))
    assert all(value["semantic_label_written"] == 0 for value in records)
    assert all(value["rules_written"] == 0 for value in records)


def test_training_census_rejects_tamper_overwrite_and_escape(
        tmp_path: Path,
        ) -> None:
    """dossier 漂移、覆盖和逃逸 run root 均失败关闭。"""
    manifest, dossier = _inputs(tmp_path)
    dossier.write_bytes(dossier.read_bytes() + b"\n")
    with pytest.raises(BroadQaExternalDataError, match="commitment"):
        audit_source_inference_training_operator_census(
            run_root=tmp_path,
            dossier_manifest_path=manifest,
            dossier_path=dossier,
            target_dir=tmp_path / "census",
        )
    target = tmp_path / "exists"
    target.mkdir()
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        audit_source_inference_training_operator_census(
            run_root=tmp_path,
            dossier_manifest_path=manifest,
            dossier_path=dossier,
            target_dir=target,
        )
    with pytest.raises(BroadQaExternalDataError, match="run root"):
        audit_source_inference_training_operator_census(
            run_root=tmp_path,
            dossier_manifest_path=manifest,
            dossier_path=dossier,
            target_dir=tmp_path.parent / "escaped-census",
        )
