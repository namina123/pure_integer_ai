"""来源版本对齐 census 与新联合 family 冻结合同测试。"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaSelectedPage,
    BroadQaTargetSelectionManifest,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    ExternalQaItem,
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_joint_eval import (
    JOINT_ALIAS_KIND,
    JOINT_TARGET_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_selection import (
    read_broad_qa_target_selection,
)
from pure_integer_ai.experiments.ph2_broad_qa_source import (
    BroadQaSourceInspection,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_aligned_family import (
    derive_source_aligned_runtime_sources,
    freeze_source_aligned_joint_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_alignment import (
    SOURCE_ALIGNED_STATUS,
    SOURCE_ALIGNMENT_CANDIDATE_KIND,
    build_source_alignment_census,
    freeze_source_alignment_candidates,
    read_source_alignment_candidates,
)
from pure_integer_ai.experiments import ph2_broad_qa_source_alignment as alignment
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _selected(ordinal: int, page_id: int, title: str) -> BroadQaSelectedPage:
    """构造 rank 严格递增的合成终页坐标。"""
    rank = f"{ordinal:064x}"
    return BroadQaSelectedPage(
        ordinal, rank, title, hashlib.sha256(title.encode()).hexdigest(),
        page_id, ordinal, ordinal * 10, ordinal * 10 + 9)


def _selection() -> BroadQaTargetSelectionManifest:
    """构造三个非重定向终页的目标 selection。"""
    pages = (
        _selected(1, 2, "投影外页"),
        _selected(2, 3, "无答案页"),
        _selected(3, 4, "对齐页"),
    )
    return BroadQaTargetSelectionManifest(
        "ZHWIKIPEDIA_20260701", "synthetic", "a" * 64, "b" * 64,
        "c" * 40, "d" * 64, 100, 100, 3, "e" * 64, pages, ())


def _candidate(
        item_id: str, title: str, answer: str, source: str = "CMRC2018",
        ) -> dict[str, object]:
    """构造不携带问题/context 的 census 候选记录。"""
    return {
        "format_version": 1,
        "gold_answers": [answer],
        "item_id": item_id,
        "license_id": "CC-BY-SA-4.0",
        "record_kind": SOURCE_ALIGNMENT_CANDIDATE_KIND,
        "source_key": source,
        "source_partition": "train",
        "source_question_id": item_id[:8],
        "source_revision": "revision",
        "title": title,
        "title_key": normalize_external_text(title),
        "upstream_url": "https://example.test/source",
    }


def _alias(
        title: str, *, page_id: int | None, revision_id: int | None,
        ) -> dict[str, object]:
    """构造 resolved 或 missing alias record。"""
    common = {
        "chain": [title],
        "format_version": 1,
        "original_surfaces": [title],
        "record_kind": JOINT_ALIAS_KIND,
        "title_key": normalize_external_text(title),
    }
    if page_id is None:
        return {
            **common,
            "failure_code": "SOURCE_TITLE_NOT_IN_SNAPSHOT",
            "status": "MISSING",
        }
    return {
        **common,
        "status": "RESOLVED",
        "terminal_page_id": page_id,
        "terminal_revision_id": revision_id,
        "terminal_title": title,
        "terminal_title_key": normalize_external_text(title),
    }


def test_source_alignment_census_separates_all_source_states(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """census 必须区分来源缺失、终页缺失、投影缺失和真实对齐。"""
    candidates = tmp_path / "candidates.jsonl"
    candidate_values = (
        _candidate("1" * 64, "缺失页", "甲"),
        _candidate("2" * 64, "投影外页", "终局答案"),
        _candidate("3" * 64, "无答案页", "不存在答案"),
        _candidate("4" * 64, "对齐页", "李冰"),
    )
    candidates.write_bytes(b"".join(
        canonical_json_line(item) for item in candidate_values))
    aliases = tmp_path / "aliases.jsonl"
    aliases.write_bytes(b"".join(canonical_json_line(item) for item in (
        _alias("缺失页", page_id=None, revision_id=None),
        _alias("投影外页", page_id=2, revision_id=1002),
        _alias("无答案页", page_id=3, revision_id=1003),
        _alias("对齐页", page_id=4, revision_id=1004),
    )))
    late_answer = "\n\n".join(
        [f"第{ordinal}段只有普通说明文字。" for ordinal in range(1, 13)]
        + ["第十三段包含终局答案。"])
    inspections = (
        BroadQaSourceInspection(
            1, "投影外页", 2, 1002, "2026-07-01T00:00:00Z", "{}",
            "a" * 64, None, late_answer),
        BroadQaSourceInspection(
            2, "无答案页", 3, 1003, "2026-07-01T00:00:00Z", "{}",
            "b" * 64, None, "这里只包含其他事实。"),
        BroadQaSourceInspection(
            3, "对齐页", 4, 1004, "2026-07-01T00:00:00Z", "{}",
            "c" * 64, None, "都江堰由李冰主持修建。"),
    )
    monkeypatch.setattr(
        "pure_integer_ai.experiments.ph2_broad_qa_source_alignment."
        "iter_broad_qa_selected_page_inspections",
        lambda *args, **kwargs: iter(inspections))
    census = tmp_path / "census.jsonl"
    manifest = tmp_path / "census-manifest.json"
    report = build_source_alignment_census(
        candidates, aliases, _selection(), xml_path=tmp_path / "unused.xml",
        census_path=census, manifest_path=manifest, worker_count=1)
    assert report["candidate_count"] == 4
    assert report["source_aligned_count"] == 1
    assert report["source_aligned_ppm"] == 250_000
    assert report["status_counts"] == {
        "GOLD_ABSENT_FROM_TERMINAL_REVISION": 1,
        "GOLD_PRESENT_OUTSIDE_PROJECTED_PASSAGES": 1,
        "SOURCE_ALIAS_MISSING": 1,
        SOURCE_ALIGNED_STATUS: 1,
    }
    statuses = {
        value["item_id"]: value["status"]
        for value in map(json.loads, census.read_text(
            encoding="utf-8").splitlines())
    }
    assert statuses["2" * 64] == "GOLD_PRESENT_OUTSIDE_PROJECTED_PASSAGES"
    assert statuses["4" * 64] == SOURCE_ALIGNED_STATUS


@pytest.mark.parametrize(("raw_hit", "full_hit", "projected_hit", "status"), (
    (False, False, False, "GOLD_ABSENT_FROM_TERMINAL_REVISION"),
    (True, False, False, "GOLD_ONLY_IN_RAW_WIKITEXT"),
    (False, True, False, "GOLD_PRESENT_OUTSIDE_PROJECTED_PASSAGES"),
    (True, True, False, "GOLD_PRESENT_OUTSIDE_PROJECTED_PASSAGES"),
    (False, False, True, "PASSAGE_PROJECTION_DIVERGES_FROM_FULL_PAGE"),
    (True, False, True, "PASSAGE_PROJECTION_DIVERGES_FROM_FULL_PAGE"),
    (False, True, True, SOURCE_ALIGNED_STATUS),
    (True, True, True, SOURCE_ALIGNED_STATUS),
))
def test_source_alignment_status_is_total_and_explicit(
        raw_hit: bool, full_hit: bool, projected_hit: bool,
        status: str) -> None:
    """三层来源覆盖的八种组合必须稳定落入显式状态。"""
    assert alignment._source_alignment_status(
        raw_hit=raw_hit, full_text_hit=full_hit,
        projected_passage_hit=projected_hit) == status


def _item(source: str, title: str, ordinal: int) -> ExternalQaItem:
    """构造 source-aligned family 选择所需的官方来源 item。"""
    item_id = hashlib.sha256(
        f"{source}:{title}:{ordinal}".encode()).hexdigest()
    return ExternalQaItem(
        item_id, source, "train", "revision", f"q-{ordinal}", title,
        f"{title}由李冰主持修建。", f"谁主持修建{title}？", ("李冰",),
        "CC-BY-SA-4.0", "https://example.test/source")


def test_candidate_pack_excludes_consumed_titles_and_binds_sources(
        tmp_path: Path) -> None:
    """完整候选不得复用旧标题，manifest 必须绑定排除来源文件。"""
    consumed = tmp_path / "consumed.jsonl"
    consumed.write_bytes(canonical_json_line({"consumed": True}))
    items = (
        _item("CMRC2018", "旧标题", 1),
        _item("CMRC2018", "新标题", 2),
        ExternalQaItem(
            "f" * 64, "DRCD", "train", "revision", "q-3", "无锚标题",
            "上下文含有李冰。", "谁主持修建工程？", ("李冰",),
            "CC-BY-SA-4.0", "https://example.test/source"),
    )
    report = freeze_source_alignment_candidates(
        items, excluded_title_keys=(normalize_external_text("旧标题"),),
        excluded_title_source_paths=(consumed,),
        target_dir=tmp_path / "candidates", source_report={"accepted": 3})
    assert report["candidate_count"] == 1
    assert report["source_target_count"] == 1
    assert report["excluded_title_count"] == 1
    assert report["excluded_title_sources"] == [{
        "sha256": hashlib.sha256(consumed.read_bytes()).hexdigest(),
    }]
    values = read_source_alignment_candidates(
        tmp_path / "candidates" / "candidates.jsonl")
    assert tuple(item["title"] for item in values) == ("新标题",)


def _titles_by_split() -> dict[str, str]:
    """寻找一对由既有标题桶规则稳定分到两个 split 的标题。"""
    result = {}
    for ordinal in range(1000):
        title = f"来源标题{ordinal}"
        digest = hashlib.sha256(
            normalize_external_text(title).encode()).digest()
        split = "dev" if int.from_bytes(digest[:4], "big") % 5 < 2 else "held_out"
        result.setdefault(split, title)
        if set(result) == {"dev", "held_out"}:
            return result
    raise AssertionError("无法构造 split 标题")


def test_source_aligned_family_uses_only_frozen_aligned_population(
        tmp_path: Path) -> None:
    """新 family 只能消费 census 中 SOURCE_ALIGNED 的 item。"""
    titles = _titles_by_split()
    items = tuple(
        _item(source, titles[split], ordinal)
        for ordinal, (source, split) in enumerate((
            ("CMRC2018", "dev"), ("CMRC2018", "held_out"),
            ("DRCD", "dev"), ("DRCD", "held_out")), start=1))
    candidates = tmp_path / "candidates.jsonl"
    candidate_records = tuple(
        _candidate(item.item_id, item.title, "李冰", item.source_key)
        for item in items)
    candidates.write_bytes(b"".join(
        canonical_json_line(item) for item in candidate_records))
    census = tmp_path / "census.jsonl"
    census.write_bytes(b"".join(canonical_json_line({
        "format_version": 1,
        "item_id": item.item_id,
        "record_kind": "PH2_BROAD_QA_SOURCE_ALIGNMENT_CENSUS_RECORD_V1",
        "source_key": item.source_key,
        "status": SOURCE_ALIGNED_STATUS,
        "terminal_page_id": ordinal,
        "terminal_revision_id": 1000 + ordinal,
        "terminal_title": item.title,
        "title_key": item.title_key,
    }) for ordinal, item in enumerate(items, start=1)))
    candidate_manifest = tmp_path / "candidate-manifest.json"
    candidate_manifest.write_bytes(canonical_json_line({"candidate_count": 4}))
    census_manifest = tmp_path / "census-manifest.json"
    census_manifest.write_bytes(canonical_json_line({
        "candidates_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
        "census_sha256": hashlib.sha256(census.read_bytes()).hexdigest(),
        "source_aligned_count": 4,
    }))
    report = freeze_source_aligned_joint_pack(
        items, candidates_path=candidates, census_path=census,
        census_manifest_path=census_manifest,
        candidate_manifest_path=candidate_manifest,
        target_dir=tmp_path / "pack", source_report={"accepted": 4},
        dev_per_source=1, held_out_per_source=1)
    assert report["population_candidate_count"] == 4
    assert report["population_source_aligned_count"] == 4
    assert report["population_source_aligned_ppm"] == 1_000_000
    assert report["splits"]["dev"]["question_count"] == 2
    questions = (tmp_path / "pack" / "held_out.questions.jsonl").read_text(
        encoding="utf-8")
    assert "gold_answers" not in questions
    assert "expected_title_key" not in questions
    labels = (tmp_path / "pack" / "held_out.labels.jsonl").read_text(
        encoding="utf-8")
    assert "gold_answers" in labels


def test_runtime_source_derivation_is_subset_deduplicated_and_idempotent(
        tmp_path: Path) -> None:
    """family alias 子集须保留每个标题，并对共享终页只收录一次。"""
    targets = tmp_path / "targets.jsonl"
    targets.write_bytes(b"".join(canonical_json_line({
        "format_version": 1,
        "record_kind": JOINT_TARGET_KIND,
        "surfaces": [title],
        "title_key": normalize_external_text(title),
    }) for title in ("甲标题", "乙标题")))
    population_aliases = tmp_path / "population-aliases.jsonl"
    population_aliases.write_bytes(b"".join(canonical_json_line({
        "chain": [title, "对齐页"],
        "format_version": 1,
        "original_surfaces": [title],
        "record_kind": JOINT_ALIAS_KIND,
        "status": "RESOLVED",
        "terminal_page_id": 4,
        "terminal_revision_id": 1004,
        "terminal_title": "对齐页",
        "terminal_title_key": normalize_external_text("对齐页"),
        "title_key": normalize_external_text(title),
    }) for title in ("甲标题", "乙标题")))
    aliases = tmp_path / "aliases.jsonl"
    selection_path = tmp_path / "selection.json"
    manifest = tmp_path / "manifest.json"
    population = _selection()
    report = derive_source_aligned_runtime_sources(
        targets, population_aliases, population,
        aliases_path=aliases, terminal_selection_path=selection_path,
        manifest_path=manifest)
    assert report["alias_count"] == 2
    assert report["terminal_page_count"] == 1
    selected = read_broad_qa_target_selection(selection_path)
    assert tuple(item.page_id for item in selected.selected_pages) == (4,)
    before = (aliases.read_bytes(), selection_path.read_bytes(),
              manifest.read_bytes())
    repeated = derive_source_aligned_runtime_sources(
        targets, population_aliases, population,
        aliases_path=aliases, terminal_selection_path=selection_path,
        manifest_path=manifest)
    assert repeated == report
    assert before == (aliases.read_bytes(), selection_path.read_bytes(),
                      manifest.read_bytes())
