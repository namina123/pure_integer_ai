"""来源内归纳未消费 review roster 的机械冻结测试。"""
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
    SOURCE_INFERENCE_REVIEW_PAYLOAD_KIND,
    SOURCE_INFERENCE_ROSTER_RECORD_KIND,
    freeze_source_inference_review_roster,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _item(ordinal: int, *, source: str) -> ExternalQaItem:
    """构造稳定标题唯一的官方来源问题。"""
    marker = f"{source}:{ordinal}"
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
    """导出 source-alignment 候选输入。"""
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
    """导出带终页身份的 source-alignment census 输入。"""
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
    """在 run root 内构造双来源、双机械 bucket 和一个已消费标题。"""
    items = []
    census = []
    ordinal = 0
    for source in ("CMRC2018", "DRCD"):
        for status in (
                "SOURCE_ALIGNED",
                "GOLD_ABSENT_FROM_TERMINAL_REVISION"):
            for _ in range(4):
                ordinal += 1
                item = _item(ordinal, source=source)
                items.append(item)
                census.append(_census(item, ordinal, status))
    ignored = _item(99, source="CMRC2018")
    items.append(ignored)
    census.append(_census(
        ignored, 99, "GOLD_PRESENT_OUTSIDE_PROJECTED_PASSAGES"))

    candidates_path = root / "candidates.jsonl"
    candidates_path.write_bytes(b"".join(
        canonical_json_line(_candidate(item)) for item in items))
    census_path = root / "census.jsonl"
    census_path.write_bytes(b"".join(
        canonical_json_line(record) for record in census))
    consumed_path = root / "consumed-source-targets.jsonl"
    consumed = items[0]
    consumed_path.write_bytes(canonical_json_line({
        "format_version": 1,
        "record_kind": JOINT_TARGET_KIND,
        "surfaces": [consumed.title],
        "title_key": normalize_external_text(consumed.title),
    }))
    return tuple(items), candidates_path, census_path, consumed_path


def test_roster_freezes_mechanical_buckets_without_semantic_labels(
        tmp_path: Path) -> None:
    """冻结器只发布待审 assignment，不提前宣判可推导或冲突。"""
    items, candidates, census, consumed = _inputs(tmp_path)
    target = tmp_path / "source-inference-roster-v1"
    report = freeze_source_inference_review_roster(
        items,
        run_root=tmp_path,
        candidates_path=candidates,
        source_census_path=census,
        consumed_source_target_paths=(consumed,),
        target_dir=target,
        source_report={"accepted_question_count": len(items)},
        extractive_quota=4,
        non_extractive_review_quota=8,
        target_stratum_quota=4,
    )

    assert report["question_count"] == report["title_count"] == 12
    assert report["assignment_counts"] == {
        "EXTRACTIVE_CANDIDATE": 4,
        "NON_EXTRACTIVE_REVIEW": 8,
    }
    assert report["source_assignment_counts"] == {
        "CMRC2018": {
            "EXTRACTIVE_CANDIDATE": 2,
            "NON_EXTRACTIVE_REVIEW": 4,
        },
        "DRCD": {
            "EXTRACTIVE_CANDIDATE": 2,
            "NON_EXTRACTIVE_REVIEW": 4,
        },
    }
    roster_payload = (target / "review.roster.jsonl").read_text(
        encoding="utf-8")
    assert "gold_answers" not in roster_payload
    assert "context" not in roster_payload
    assert "SOURCE_DERIVABLE" not in roster_payload
    assert "SOURCE_CONFLICT" not in roster_payload
    roster = tuple(map(json.loads, roster_payload.splitlines()))
    assert all(item["record_kind"] == SOURCE_INFERENCE_ROSTER_RECORD_KIND
               for item in roster)
    assert len({item["title_key"] for item in roster}) == 12

    review = tuple(map(json.loads, (target / "review.payload.jsonl").read_text(
        encoding="utf-8").splitlines()))
    assert all(item["record_kind"] == SOURCE_INFERENCE_REVIEW_PAYLOAD_KIND
               for item in review)
    assert {item["item_id"] for item in review} == {
        item["item_id"] for item in roster}
    assert all(item["gold_answers"] and item["context"] for item in review)
    assert report["decision_ledger_contract"] == {
        "allowed_decisions": [
            "EXTRACTIVE", "SOURCE_DERIVABLE", "SOURCE_CONFLICT", "REJECT"],
        "exact_roster_coverage_required": 1,
        "extractive_requires_source_aligned": 1,
        "source_conflict_requires_distinct_source_commitments": 1,
        "source_derivable_requires_inference_record_sha256": 1,
        "unreviewed_item_selection_forbidden": 1,
    }


def test_roster_is_non_overwritable_and_paths_cannot_escape_root(
        tmp_path: Path) -> None:
    """已冻结目标不得覆盖，输入输出也不得逃逸 run root。"""
    items, candidates, census, consumed = _inputs(tmp_path)
    target = tmp_path / "target"
    arguments = dict(
        items=items,
        run_root=tmp_path,
        candidates_path=candidates,
        source_census_path=census,
        consumed_source_target_paths=(consumed,),
        target_dir=target,
        source_report={"accepted_question_count": len(items)},
        extractive_quota=2,
        non_extractive_review_quota=4,
        target_stratum_quota=2,
    )
    freeze_source_inference_review_roster(**arguments)
    with pytest.raises(BroadQaExternalDataError, match="边界非法"):
        freeze_source_inference_review_roster(**arguments)

    outside = tmp_path.parent / "outside-roster"
    with pytest.raises(BroadQaExternalDataError, match="run root"):
        freeze_source_inference_review_roster(
            **{**arguments, "target_dir": outside})


def test_target_stratum_quota_requires_enough_review_denominator(
        tmp_path: Path) -> None:
    """可推导与冲突目标配额不能超过预冻结非抽取 review 分母。"""
    items, candidates, census, consumed = _inputs(tmp_path)
    with pytest.raises(BroadQaExternalDataError, match="配额边界"):
        freeze_source_inference_review_roster(
            items,
            run_root=tmp_path,
            candidates_path=candidates,
            source_census_path=census,
            consumed_source_target_paths=(consumed,),
            target_dir=tmp_path / "target",
            source_report={"accepted_question_count": len(items)},
            extractive_quota=2,
            non_extractive_review_quota=4,
            target_stratum_quota=3,
        )
