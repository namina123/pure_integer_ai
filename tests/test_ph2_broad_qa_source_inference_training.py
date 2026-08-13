"""独立来源归纳训练 roster 的冻结边界测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    ExternalQaItem,
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_joint_eval import (
    JOINT_TARGET_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_alignment import (
    SOURCE_ALIGNMENT_CANDIDATE_KIND,
    SOURCE_ALIGNMENT_CENSUS_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_family import (
    SOURCE_INFERENCE_ROSTER_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_training import (
    SOURCE_INFERENCE_OPERATOR_FAMILIES,
    SOURCE_INFERENCE_TRAINING_ASSIGNMENTS,
    SOURCE_INFERENCE_TRAINING_EXCLUSION_KIND,
    SOURCE_INFERENCE_TRAINING_PAYLOAD_KIND,
    SOURCE_INFERENCE_TRAINING_RECORD_KIND,
    freeze_source_inference_training_roster,
    read_source_inference_training_exclusions,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


_STATUSES = (
    "SOURCE_ALIGNED",
    "GOLD_PRESENT_OUTSIDE_PROJECTED_PASSAGES",
    "GOLD_ONLY_IN_RAW_WIKITEXT",
    "GOLD_ABSENT_FROM_TERMINAL_REVISION",
)


def _item(ordinal: int, *, source: str, status: str) -> ExternalQaItem:
    """构造稳定、标题唯一的官方问题。"""
    marker = f"{source}:{status}:{ordinal}"
    return ExternalQaItem(
        hashlib.sha256(marker.encode("ascii")).hexdigest(),
        source,
        "train",
        "revision",
        f"q-{ordinal}",
        f"标题{marker}",
        f"来源正文{marker}",
        f"标题{marker}的问题是什么？",
        (f"答案{marker}",),
        "CC-BY-SA-4.0",
        "https://example.test/source",
    )


def _candidate(item: ExternalQaItem) -> dict[str, object]:
    """导出 source-alignment candidate。"""
    return {
        "format_version": 1,
        "gold_answers": list(item.gold_answers),
        "item_id": item.item_id,
        "license_id": item.license_id,
        "record_kind": SOURCE_ALIGNMENT_CANDIDATE_KIND,
        "source_key": item.source_key,
        "source_partition": item.source_partition,
        "source_question_id": item.source_question_id,
        "source_revision": item.source_revision,
        "title": item.title,
        "title_key": item.title_key,
        "upstream_url": item.upstream_url,
    }


def _census(
        item: ExternalQaItem,
        ordinal: int,
        status: str,
        ) -> dict[str, object]:
    """导出具有终页身份的 census record。"""
    return {
        "format_version": 1,
        "item_id": item.item_id,
        "record_kind": SOURCE_ALIGNMENT_CENSUS_RECORD_KIND,
        "source_key": item.source_key,
        "status": status,
        "terminal_page_id": ordinal,
        "terminal_revision_id": 1000 + ordinal,
        "terminal_title": item.title,
        "title_key": item.title_key,
    }


def _inputs(root: Path):
    """构造双来源、四机械桶及两种既有消费排除。"""
    items = []
    census = []
    ordinal = 0
    for source in ("CMRC2018", "DRCD"):
        for status in _STATUSES:
            for _ in range(4):
                ordinal += 1
                item = _item(ordinal, source=source, status=status)
                items.append(item)
                census.append(_census(item, ordinal, status))
    candidates_path = root / "candidates.jsonl"
    candidates_path.write_bytes(b"".join(
        canonical_json_line(_candidate(item)) for item in items))
    census_path = root / "census.jsonl"
    census_path.write_bytes(b"".join(
        canonical_json_line(record) for record in census))

    consumed_item = items[0]
    consumed_target = root / "consumed-source-targets.jsonl"
    consumed_target.write_bytes(canonical_json_line({
        "format_version": 1,
        "record_kind": JOINT_TARGET_KIND,
        "surfaces": [consumed_item.title],
        "title_key": consumed_item.title_key,
    }))
    discovery_item = items[1]
    discovery_roster = root / "discovery.roster.jsonl"
    discovery_roster.write_bytes(canonical_json_line({
        "assignment": "EXTRACTIVE_CANDIDATE",
        "format_version": 1,
        "item_id": discovery_item.item_id,
        "question_sha256": hashlib.sha256(
            discovery_item.question.encode("utf-8")).hexdigest(),
        "record_kind": SOURCE_INFERENCE_ROSTER_RECORD_KIND,
        "source_alignment_status": "SOURCE_ALIGNED",
        "source_key": discovery_item.source_key,
        "terminal_page_id": 2,
        "terminal_revision_id": 1002,
        "title_key": discovery_item.title_key,
    }))
    return (
        tuple(items), candidates_path, census_path,
        consumed_target, discovery_roster,
    )


def _freeze(root: Path, *, target: Path | None = None):
    """以每来源每桶两个样本冻结测试总体。"""
    items, candidates, census, consumed, discovery = _inputs(root)
    report = freeze_source_inference_training_roster(
        items,
        run_root=root,
        candidates_path=candidates,
        source_census_path=census,
        consumed_source_target_paths=(consumed,),
        excluded_discovery_roster_paths=(discovery,),
        target_dir=target or root / "training-v1",
        source_report={"accepted_question_count": len(items)},
        extractive_per_source=2,
        passage_coverage_per_source=2,
        raw_wikitext_per_source=2,
        non_extractive_per_source=2,
    )
    return report, items


def test_training_roster_freezes_before_learning_and_balances_sources(
        tmp_path: Path) -> None:
    """训练总体必须双来源等额，且不得预写 operator 或语义 decision。"""
    report, items = _freeze(tmp_path)
    target = tmp_path / "training-v1"
    assert report["question_count"] == report["title_count"] == 16
    assert report["source_counts"] == {"CMRC2018": 8, "DRCD": 8}
    assert report["assignment_counts"] == {
        assignment: 4 for assignment in SOURCE_INFERENCE_TRAINING_ASSIGNMENTS}
    assert report["learner_read_count_at_freeze"] == 0
    assert report["semantic_labels_written"] == 0
    assert report["status"] == "FROZEN_NOT_READ_NOT_LEARNED"
    assert report["operator_discovery_contract"] == {
        "allowed_operator_families": list(SOURCE_INFERENCE_OPERATOR_FAMILIES),
        "defeater_required": 1,
        "negative_examples_required": 1,
        "operator_family_preassigned_count": 0,
        "positive_examples_required": 1,
        "rule_evidence_record_required": 1,
        "scope_required": 1,
        "unlisted_operator_allowed": 0,
    }

    roster_text = (target / "train.roster.jsonl").read_text(encoding="utf-8")
    assert "gold_answers" not in roster_text
    assert "context" not in roster_text
    assert "operator" not in roster_text
    assert "decision" not in roster_text
    roster = tuple(map(json.loads, roster_text.splitlines()))
    assert all(
        value["record_kind"] == SOURCE_INFERENCE_TRAINING_RECORD_KIND
        for value in roster)

    payload = tuple(map(json.loads, (target / "train.payload.jsonl").read_text(
        encoding="utf-8").splitlines()))
    assert all(
        value["record_kind"] == SOURCE_INFERENCE_TRAINING_PAYLOAD_KIND
        for value in payload)
    assert {value["item_id"] for value in payload} == {
        value["item_id"] for value in roster}
    excluded_ids = {items[0].item_id, items[1].item_id}
    assert not excluded_ids & {value["item_id"] for value in roster}


def test_training_exclusion_inventory_enforces_item_and_title_disjointness(
        tmp_path: Path) -> None:
    """未来 evaluation 必须能严格回读训练 item/title 双排除清单。"""
    report, _ = _freeze(tmp_path)
    path = tmp_path / "training-v1" / "evaluation-exclusion.inventory.jsonl"
    values = read_source_inference_training_exclusions(path)
    assert len(values) == 16
    assert len({item_id for item_id, _ in values}) == 16
    assert len({title_key for _, title_key in values}) == 16
    assert report["future_evaluation_contract"] == {
        "exclusion_inventory_required": 1,
        "item_overlap_allowed": 0,
        "title_overlap_allowed": 0,
        "evaluation_freeze_after_rule_pack_only": 1,
    }
    raw = tuple(map(json.loads, path.read_text(encoding="utf-8").splitlines()))
    assert all(value["record_kind"] == SOURCE_INFERENCE_TRAINING_EXCLUSION_KIND
               for value in raw)

    raw[1]["title_key"] = raw[0]["title_key"]
    path.write_bytes(b"".join(canonical_json_line(value) for value in raw))
    with pytest.raises(BroadQaExternalDataError, match="exclusion 漂移"):
        read_source_inference_training_exclusions(path)


def test_training_freeze_is_non_overwritable_and_stays_in_run_root(
        tmp_path: Path) -> None:
    """冻结目标不得覆盖，输入输出不得逃逸显式 run root。"""
    _freeze(tmp_path)
    with pytest.raises(BroadQaExternalDataError, match="路径边界非法"):
        _freeze(tmp_path)

    other = tmp_path / "other"
    other.mkdir()
    items, candidates, census, consumed, discovery = _inputs(other)
    with pytest.raises(BroadQaExternalDataError, match="run root"):
        freeze_source_inference_training_roster(
            items,
            run_root=other,
            candidates_path=candidates,
            source_census_path=census,
            consumed_source_target_paths=(consumed,),
            excluded_discovery_roster_paths=(discovery,),
            target_dir=tmp_path.parent / "escaped-training",
            source_report={"accepted_question_count": len(items)},
            extractive_per_source=1,
            passage_coverage_per_source=1,
            raw_wikitext_per_source=1,
            non_extractive_per_source=1,
        )
