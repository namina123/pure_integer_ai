"""未消费来源对齐问题的交互维度 census 合同测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    ExternalQaItem,
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_interactive_family import (
    DIMENSION_ARTIFACT_PATH,
    DIMENSION_ARTIFACT_SHA256,
    build_interactive_dimension_census,
    classify_interactive_dimension,
    freeze_interactive_development_pack,
    load_interactive_dimension_rules,
)
from pure_integer_ai.experiments.ph2_broad_qa_interactive_eval import (
    REFUSAL_PROBE_PATH,
    load_refusal_probes,
    publish_interactive_refusal_report,
)
from pure_integer_ai.experiments import ph2_broad_qa_interactive_eval as ieval
from pure_integer_ai.experiments.ph2_broad_qa_joint_eval import (
    JOINT_TARGET_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_alignment import (
    SOURCE_ALIGNED_STATUS,
    SOURCE_ALIGNMENT_CANDIDATE_KIND,
    SOURCE_ALIGNMENT_CENSUS_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


INTERACTIVE_RECEIPT_PATH = Path(
    "data/ph2/broad_qa_interactive_development_receipt_v1.json")


def _item(ordinal: int, title: str, question: str) -> ExternalQaItem:
    """构造只供维度归类使用的官方来源问题。"""
    item_id = hashlib.sha256(f"item:{ordinal}".encode()).hexdigest()
    return ExternalQaItem(
        item_id, "CMRC2018" if ordinal % 2 else "DRCD", "train",
        "revision", f"q-{ordinal}", title, f"{title}的来源正文。",
        question, ("来源答案",), "CC-BY-SA-4.0",
        "https://example.test/source")


def _candidate(item: ExternalQaItem) -> dict[str, object]:
    """构造与官方问题同身份的来源对齐候选。"""
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


def _source_census(
        item: ExternalQaItem, ordinal: int, *, aligned: bool = True,
        ) -> dict[str, object]:
    """构造来源对齐或明确不对齐的 census 记录。"""
    common = {
        "format_version": 1,
        "item_id": item.item_id,
        "record_kind": SOURCE_ALIGNMENT_CENSUS_RECORD_KIND,
        "source_key": item.source_key,
        "status": (
            SOURCE_ALIGNED_STATUS if aligned
            else "GOLD_ABSENT_FROM_TERMINAL_REVISION"),
        "title_key": item.title_key,
        "terminal_page_id": ordinal,
        "terminal_revision_id": 1000 + ordinal,
        "terminal_title": item.title,
    }
    return common


def test_dimension_rules_are_canonical_and_sha_frozen(tmp_path: Path) -> None:
    """公开 CC0 规则须规范可回读，任何语义改写都被拒绝。"""
    rules = load_interactive_dimension_rules()
    assert rules["primary_dimension_order"] == [
        "CAUSE", "COMPARISON", "TIME", "QUANTITY", "RELATION"]
    tampered = dict(rules)
    tampered["comparison_surfaces"] = [*rules["comparison_surfaces"], "胜于"]
    target = tmp_path / "tampered.json"
    target.write_bytes(canonical_json_line(tampered))
    with pytest.raises(BroadQaExternalDataError, match="漂移"):
        load_interactive_dimension_rules(target)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(DIMENSION_ARTIFACT_PATH.read_bytes().rstrip())
    with pytest.raises(BroadQaExternalDataError, match="漂移"):
        load_interactive_dimension_rules(noncanonical)


@pytest.mark.parametrize(("question", "dimension"), (
    ("为什么甲比乙更早完成？", "CAUSE"),
    ("甲和乙相比哪个更早完成？", "COMPARISON"),
    ("甲何时完成？", "TIME"),
    ("甲有多长？", "QUANTITY"),
    ("谁主持修建甲？", "RELATION"),
))
def test_dimension_classification_uses_frozen_priority(
        question: str, dimension: str) -> None:
    """重叠问式必须按因果、比较、时间、数量、关系的冻结顺序归类。"""
    assert classify_interactive_dimension(question) == dimension


def test_census_excludes_consumed_and_closes_all_counts(
        tmp_path: Path) -> None:
    """只统计剩余 SOURCE_ALIGNED，且总体、来源和维度计数闭合。"""
    items = (
        _item(1, "因果页", "为什么形成该现象？"),
        _item(2, "比较页", "甲和乙有什么不同？"),
        _item(3, "时间页", "该事件何时发生？"),
        _item(4, "数量页", "这座桥有多长？"),
        _item(5, "关系页", "谁主持修建该工程？"),
        _item(6, "已消费页", "谁发现该遗址？"),
        _item(7, "未对齐页", "谁记录该事件？"),
    )
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_bytes(b"".join(
        canonical_json_line(_candidate(item)) for item in items))
    source_census = tmp_path / "source-census.jsonl"
    source_census.write_bytes(b"".join(
        canonical_json_line(_source_census(
            item, ordinal, aligned=ordinal != 7))
        for ordinal, item in enumerate(items, start=1)))
    consumed = tmp_path / "consumed-source-targets.jsonl"
    consumed.write_bytes(canonical_json_line({
        "format_version": 1,
        "record_kind": JOINT_TARGET_KIND,
        "surfaces": ["已消费页"],
        "title_key": normalize_external_text("已消费页"),
    }))
    census = tmp_path / "interactive-census.jsonl"
    manifest = tmp_path / "interactive-manifest.json"
    report = build_interactive_dimension_census(
        items, candidates_path=candidates,
        source_census_path=source_census,
        consumed_source_target_paths=(consumed,), census_path=census,
        manifest_path=manifest, source_report={"accepted_question_count": 7})

    assert report["candidate_count"] == 7
    assert report["consumed_title_count"] == 1
    assert report["excluded_consumed_item_count"] == 1
    assert report["remaining_source_aligned_count"] == 5
    assert report["dimension_counts"] == {
        "CAUSE": 1, "COMPARISON": 1, "TIME": 1,
        "QUANTITY": 1, "RELATION": 1,
    }
    records = tuple(map(json.loads, census.read_text(
        encoding="utf-8").splitlines()))
    assert {record["title_key"] for record in records} == {
        item.title_key for item in items[:5]}
    assert sum(
        sum(counts.values()) for counts in report["per_source"].values()
    ) == report["remaining_source_aligned_count"]
    assert json.loads(manifest.read_text(encoding="utf-8"))[
        "census_sha256"] == hashlib.sha256(census.read_bytes()).hexdigest()
    with pytest.raises(BroadQaExternalDataError, match="输出边界"):
        build_interactive_dimension_census(
            items, candidates_path=candidates,
            source_census_path=source_census,
            consumed_source_target_paths=(consumed,), census_path=census,
            manifest_path=manifest,
            source_report={"accepted_question_count": 7})


def test_development_pack_is_dimension_balanced_and_title_disjoint(
        tmp_path: Path) -> None:
    """开发集须五维等额、可用来源均衡且 questions 不泄漏标签。"""
    questions = {
        "CAUSE": "为什么形成该现象？",
        "COMPARISON": "甲和乙有什么不同？",
        "TIME": "该事件何时发生？",
        "QUANTITY": "这座桥有多长？",
        "RELATION": "谁主持修建该工程？",
    }
    items = []
    records = []
    ordinal = 0
    for dimension, question in questions.items():
        sources = ("CMRC2018",) if dimension == "CAUSE" else (
            "CMRC2018", "DRCD")
        for source in sources:
            for local in range(2):
                ordinal += 1
                item = _item(
                    ordinal, f"{dimension}-{source}-{local}", question)
                if item.source_key != source:
                    item = ExternalQaItem(
                        item.item_id, source, item.source_partition,
                        item.source_revision, item.source_question_id,
                        item.title, item.context, item.question,
                        item.gold_answers, item.license_id, item.upstream_url)
                items.append(item)
                records.append({
                    "dimension": dimension,
                    "format_version": 1,
                    "item_id": item.item_id,
                    "question_sha256": hashlib.sha256(
                        item.question.encode("utf-8")).hexdigest(),
                    "record_kind": (
                        "PH2_BROAD_QA_INTERACTIVE_DIMENSION_CENSUS_RECORD_V1"),
                    "source_key": source,
                    "terminal_page_id": ordinal,
                    "terminal_revision_id": 1000 + ordinal,
                    "title_key": item.title_key,
                })
    census = tmp_path / "census.jsonl"
    census.write_bytes(b"".join(
        canonical_json_line(record) for record in records))
    census_manifest = tmp_path / "census-manifest.json"
    census_manifest.write_bytes(canonical_json_line({
        "artifact_kind": "PH2_BROAD_QA_INTERACTIVE_DIMENSION_CENSUS_V1",
        "census_sha256": hashlib.sha256(census.read_bytes()).hexdigest(),
        "remaining_source_aligned_count": len(records),
        "rules_sha256": DIMENSION_ARTIFACT_SHA256,
    }))
    target = tmp_path / "pack"
    report = freeze_interactive_development_pack(
        items, census_path=census, census_manifest_path=census_manifest,
        target_dir=target, source_report={"accepted_question_count": len(items)},
        dimension_quota=2)

    assert report["question_count"] == 10
    assert report["title_count"] == 10
    assert report["dimension_counts"] == {
        dimension: 2 for dimension in questions}
    assert report["source_dimension_counts"]["CMRC2018"]["CAUSE"] == 2
    for dimension in ("COMPARISON", "TIME", "QUANTITY", "RELATION"):
        assert report["source_dimension_counts"]["CMRC2018"][dimension] == 1
        assert report["source_dimension_counts"]["DRCD"][dimension] == 1
    question_text = (target / "dev.questions.jsonl").read_text(
        encoding="utf-8")
    assert "gold_answers" not in question_text
    assert "expected_title_key" not in question_text
    assert len((target / "dev.dimensions.jsonl").read_text(
        encoding="utf-8").splitlines()) == 10
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        freeze_interactive_development_pack(
            items, census_path=census,
            census_manifest_path=census_manifest, target_dir=target,
            source_report={"accepted_question_count": len(items)},
            dimension_quota=2)


def test_refusal_probes_are_sha_frozen_and_report_is_non_overwritable(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """原创 UNKNOWN/CLARIFY 探针须锁定字节、逐项过门且禁止覆盖。"""
    probes = load_refusal_probes()
    expected = {item["question"]: item["expected_status"] for item in probes}
    database = tmp_path / "database.sqlite3"
    connection = sqlite3.connect(str(database))
    connection.close()

    def fake_query(_connection: sqlite3.Connection, question: str):
        """按已冻结开发预期返回最小查询结果结构。"""
        return SimpleNamespace(
            status=expected[question], candidate_document_count=3,
            matched_term_count=2)

    monkeypatch.setattr(ieval, "query_broad_qa", fake_query)
    report_path = tmp_path / "report.json"
    report = publish_interactive_refusal_report(
        database, report_path=report_path)
    assert report["status"] == "PASS"
    assert report["passed_count"] == report["probe_count"] == 4
    assert report["boundary"] == (
        "AUTHORED_DEVELOPMENT_REGRESSION_NOT_FORMAL_HELD_OUT")
    with pytest.raises(BroadQaExternalDataError, match="禁止覆盖"):
        publish_interactive_refusal_report(
            database, report_path=report_path)
    tampered = tmp_path / "tampered.json"
    value = json.loads(REFUSAL_PROBE_PATH.read_text(encoding="utf-8"))
    value["probes"][0]["expected_status"] = "CLARIFY"
    tampered.write_bytes(canonical_json_line(value))
    with pytest.raises(BroadQaExternalDataError, match="漂移"):
        load_refusal_probes(tampered)


def test_interactive_development_receipt_is_canonical_and_bounded() -> None:
    """公开开发 receipt 必须规范、非 formal，且不携带第三方 payload。"""
    payload = INTERACTIVE_RECEIPT_PATH.read_bytes()
    value = json.loads(payload)
    assert payload == canonical_json_line(value)
    assert value["artifact_kind"] == (
        "PH2_BROAD_QA_INTERACTIVE_DEVELOPMENT_RECEIPT_V1")
    assert value["scope"] == "DEVELOPMENT_NON_FORMAL"
    assert value["boundary"] == (
        "DEVELOPMENT_ONLY_SURFACE_BUCKETS_NOT_SEMANTIC_UNDERSTANDING")
    assert value["question_count"] == 100
    assert value["source_page_gold_coverage_count"] == 100
    assert value["total_evidence_hit_count"] == 87
    assert value["refusal_probe_passed_count"] == 4
    assert not ({
        "questions", "labels", "predictions", "contexts", "page_text",
        "local_paths", "run_root", "database_path",
    } & set(value))
    serialized = payload.decode("utf-8").casefold()
    separator = chr(92)
    assert all(f"{drive}:{separator}" not in serialized for drive in ("k", "d"))


def test_interactive_public_artifacts_are_packaged() -> None:
    """交互规则、拒答探针和紧凑 receipt 必须进入 wheel 清单。"""
    import tomllib

    value = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    files = set(value["tool"]["setuptools"]["data-files"][
        "share/pure_integer_ai/data/ph2"])
    assert {
        "data/ph2/broad_qa_interactive_dimensions_v1.json",
        "data/ph2/broad_qa_interactive_refusal_probes_v1.json",
        "data/ph2/broad_qa_interactive_development_receipt_v1.json",
    } <= files
