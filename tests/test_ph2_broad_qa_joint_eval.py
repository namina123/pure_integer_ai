"""自然标题锚定联合检索评测的冻结、合库和评分测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaSelectedPage,
    BroadQaTargetSelectionManifest,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    ExternalQaItem,
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_index import broad_qa_terms
from pure_integer_ai.experiments.ph2_broad_qa_interactive_eval import (
    publish_interactive_dimension_report,
)
from pure_integer_ai.experiments.ph2_broad_qa_interactive_family import (
    INTERACTIVE_DIMENSION_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_joint_eval import (
    JOINT_ALIAS_KIND,
    JOINT_QUESTION_KIND,
    augment_broad_qa_index,
    freeze_joint_source_pack,
    predict_joint_retrieval,
    resolve_joint_source_aliases,
    score_joint_retrieval,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_broad_qa_query import query_broad_qa
from pure_integer_ai.experiments.ph2_broad_qa_query import (
    _cover_explicit_number_evidence_windows,
    _expand_structural_evidence_windows,
)
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


def _item(source: str, ordinal: int, title: str) -> ExternalQaItem:
    """构造自然包含标题的稳定外部问题。"""
    item_id = hashlib.sha256(
        f"{source}:{ordinal}:{title}".encode()).hexdigest()
    return ExternalQaItem(
        item_id, source, "train", "revision", f"q-{ordinal}", title,
        f"{title}由李冰主持修建。", f"谁主持修建{title}？", ("李冰",),
        "CC-BY-SA-4.0", "https://example.test/source")


def _database(
        path: Path,
        pages: tuple[tuple[int, str, str | tuple[str, ...]], ...],
        ) -> None:
    """构造兼容生产查询 schema 的小型整数 posting 数据库。"""
    connection = sqlite3.connect(str(path))
    connection.executescript("""
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE document(
            doc_id INTEGER PRIMARY KEY,title TEXT NOT NULL,
            page_id INTEGER NOT NULL UNIQUE,revision_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,contributor_json TEXT NOT NULL,
            text_sha256 TEXT NOT NULL);
        CREATE TABLE passage(
            passage_id INTEGER PRIMARY KEY,doc_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,raw_start INTEGER NOT NULL,
            raw_end INTEGER NOT NULL,raw_sha256 TEXT NOT NULL,
            text TEXT NOT NULL,text_sha256 TEXT NOT NULL,
            section_title TEXT NOT NULL);
        CREATE TABLE posting(
            term TEXT PRIMARY KEY,document_frequency INTEGER NOT NULL,
            passage_deltas BLOB NOT NULL);
        CREATE INDEX passage_doc ON passage(doc_id,ordinal);
        CREATE INDEX document_title ON document(title);
    """)
    postings = {}
    passage_count = 0
    for doc_id, (page_id, title, text_value) in enumerate(pages, start=1):
        texts = (text_value,) if isinstance(text_value, str) else text_value
        document_text = "\n".join(texts)
        text_sha = hashlib.sha256(document_text.encode()).hexdigest()
        connection.execute(
            "INSERT INTO document VALUES(?,?,?,?,?,?,?)",
            (doc_id, title, page_id, 1000 + page_id,
             "2026-07-01T00:00:00Z",
             '{"kind":"registered","user_id":7,"username":"测试"}',
             text_sha))
        raw_start = 0
        for ordinal, text in enumerate(texts, start=1):
            passage_id = doc_id * 128 + ordinal
            passage_sha = hashlib.sha256(text.encode()).hexdigest()
            connection.execute(
                "INSERT INTO passage VALUES(?,?,?,?,?,?,?,?,?)",
                (passage_id, doc_id, ordinal, raw_start,
                 raw_start + len(text), passage_sha, text,
                 passage_sha, ""))
            for term in set(broad_qa_terms(title)) | set(broad_qa_terms(text)):
                postings.setdefault(term, []).append(passage_id)
            raw_start += len(text) + 1
            passage_count += 1
    for term, values in sorted(postings.items()):
        prior = 0
        deltas = []
        for value in values:
            deltas.append(value - prior)
            prior = value
        connection.execute(
            "INSERT INTO posting VALUES(?,?,?)",
            (term, len(values), encode_integer_tuple(tuple(deltas))))
    metadata = {
        "accepted_page_count": str(len(pages)),
        "index_schema_version": "1", "license_id": "CC-BY-SA-4.0",
        "passage_count": str(passage_count), "selection_sha256": "a" * 64,
        "snapshot_id": "synthetic", "source_key": "ZHWIKIPEDIA_20260701",
        "term_count": str(len(postings)),
    }
    connection.executemany(
        "INSERT INTO metadata VALUES(?,?)", sorted(metadata.items()))
    connection.commit()
    connection.execute("VACUUM")
    connection.close()


def test_joint_freeze_excludes_prior_titles_and_keeps_labels_separate(
        tmp_path: Path) -> None:
    """successor family 排除旧问题与 target 标题域，且不泄漏标签。"""
    prior = tmp_path / "prior.jsonl"
    prior.write_bytes(canonical_json_line({
        "context": "旧标题内容", "context_sha256": "a" * 64,
        "format_version": 1, "item_id": "f" * 64,
        "license_id": "CC-BY-SA-4.0", "question": "旧问题",
        "record_kind": "PH2_BROAD_QA_EXTERNAL_QUESTION_V1",
        "source_key": "CMRC2018", "source_partition": "dev",
        "source_question_id": "old", "source_revision": "old",
        "split": "dev", "title": "旧标题",
        "upstream_url": "https://example.test/old",
    }))
    prior_targets = tmp_path / "prior-targets.jsonl"
    prior_targets.write_bytes(canonical_json_line({
        "format_version": 1,
        "record_kind": "PH2_BROAD_QA_JOINT_SOURCE_TARGET_V1",
        "surfaces": ["前代标题"],
        "title_key": normalize_external_text("前代标题"),
    }))
    items = []
    for source in ("CMRC2018", "DRCD"):
        items.extend(_item(source, ordinal, f"标题{source}{ordinal}")
                     for ordinal in range(80))
        items.append(_item(source, 999, "旧标题"))
        items.append(_item(source, 1000, "前代标题"))
    report = freeze_joint_source_pack(
        items, prior_question_paths=(prior,),
        prior_source_target_paths=(prior_targets,),
        target_dir=tmp_path / "pack",
        source_report={"accepted_question_count": len(items)},
        dev_per_source=5, held_out_per_source=5)
    assert report["artifact_kind"].endswith("_V2")
    assert report["excluded_prior_question_title_count"] == 1
    assert report["excluded_prior_source_target_title_count"] == 1
    assert report["excluded_prior_title_count"] == 2
    assert report["excluded_prior_source_target_files"] == [{
        "sha256": hashlib.sha256(prior_targets.read_bytes()).hexdigest(),
    }]
    questions = (tmp_path / "pack" / "held_out.questions.jsonl").read_text(
        encoding="utf-8")
    labels = (tmp_path / "pack" / "held_out.labels.jsonl").read_text(
        encoding="utf-8")
    assert "gold_answers" not in questions
    assert "expected_title_key" not in questions
    assert "gold_answers" in labels
    assert normalize_external_text("旧标题") not in labels
    assert normalize_external_text("前代标题") not in labels


def test_alias_resolution_restores_requested_key_from_exact_surface(
        tmp_path: Path, monkeypatch) -> None:
    """命中页标题字形不同于规范键时，仍须恢复原始请求键。"""
    from pure_integer_ai.experiments.ph2_broad_qa_source import (
        BroadQaSourceInspection,
    )
    from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
        MediaWikiDumpSnapshotManifest,
    )

    selected = BroadQaSelectedPage(
        1, "1" * 64, "橢球座標系",
        hashlib.sha256("橢球座標系".encode()).hexdigest(), 7, 1, 0, 1)
    selection = BroadQaTargetSelectionManifest(
        "ZHWIKIPEDIA_20260701", "synthetic", "a" * 64, "b" * 64,
        "c" * 40, "d" * 64, 1, 1, 1, "e" * 64, (selected,), ())
    source_targets = {
        normalize_external_text("椭球坐标系"): (
            "椭球坐标系", "橢球座標系"),
    }
    inspection = BroadQaSourceInspection(
        1, "橢球座標系", 7, 1007, "2026-07-01T00:00:00Z", "{}",
        "f" * 64, None, "正文")
    monkeypatch.setattr(
        "pure_integer_ai.experiments.ph2_broad_qa_joint_eval."
        "iter_broad_qa_selected_page_inspections",
        lambda *args, **kwargs: iter((inspection,)))
    snapshot = object.__new__(MediaWikiDumpSnapshotManifest)
    terminal, report = resolve_joint_source_aliases(
        snapshot, selection, source_targets,
        snapshot_manifest_sha256="a" * 64,
        index_path=tmp_path / "unused-index",
        xml_path=tmp_path / "unused-xml",
        alias_path=tmp_path / "aliases.jsonl", worker_count=1)
    assert report["resolved_count"] == 1
    assert terminal.selected_pages[0].title == "橢球座標系"
    record = json.loads((tmp_path / "aliases.jsonl").read_text(
        encoding="utf-8"))
    assert record["title_key"] == normalize_external_text("椭球坐标系")


def test_joint_augmentation_prediction_and_scoring_close_end_to_end(
        tmp_path: Path) -> None:
    """随机基库可追加目标页，并以检索、证据和引用联合通过。"""
    base = tmp_path / "base.sqlite3"
    extra = tmp_path / "extra.sqlite3"
    combined = tmp_path / "combined.sqlite3"
    _database(base, ((1, "干扰页", "干扰页包含无关的水利工程说明。"),))
    _database(extra, ((2, "都江堰", (
        "都江堰是一项古代水利工程，主要用于防洪。",
        "这项工程由李冰主持修建，后来持续发挥作用。",
    )),))
    aliases = tmp_path / "aliases.jsonl"
    aliases.write_bytes(canonical_json_line({
        "chain": ["都江堰"], "format_version": 1,
        "original_surfaces": ["都江堰"], "record_kind": JOINT_ALIAS_KIND,
        "status": "RESOLVED", "terminal_page_id": 2,
        "terminal_revision_id": 1002, "terminal_title": "都江堰",
        "terminal_title_key": normalize_external_text("都江堰"),
        "title_key": normalize_external_text("都江堰"),
    }))
    report = augment_broad_qa_index(
        base, extra, output_database_path=combined,
        base_expected_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        target_selection_sha256="b" * 64, alias_path=aliases)
    assert report["added_document_count"] == 1
    question = tmp_path / "questions.jsonl"
    label = tmp_path / "labels.jsonl"
    prediction = tmp_path / "predictions.jsonl"
    item_id = "c" * 64
    question.write_bytes(canonical_json_line({
        "format_version": 1, "item_id": item_id,
        "license_id": "CC-BY-SA-4.0", "question": "谁主持修建都江堰？",
        "record_kind": JOINT_QUESTION_KIND, "source_key": "CMRC2018",
        "source_partition": "dev", "source_question_id": "q1",
        "source_revision": "r1", "split": "dev",
        "upstream_url": "https://example.test/source",
    }))
    label.write_bytes(canonical_json_line({
        "expected_title_key": normalize_external_text("都江堰"),
        "format_version": 1, "gold_answers": ["李冰"], "item_id": item_id,
        "record_kind": "PH2_BROAD_QA_JOINT_LABEL_V1", "split": "dev",
    }))
    predict_joint_retrieval(
        question, combined, predictions_path=prediction)
    selected = BroadQaSelectedPage(
        1, "1" * 64, "都江堰",
        hashlib.sha256("都江堰".encode()).hexdigest(), 2, 1, 0, 1)
    selection = BroadQaTargetSelectionManifest(
        "ZHWIKIPEDIA_20260701", "synthetic", "a" * 64, "b" * 64,
        "c" * 40, "d" * 64, 1, 1, 1, "e" * 64, (selected,), ())
    prediction_value = json.loads(prediction.read_text(encoding="utf-8"))
    assert len(prediction_value["result"]["citations"]) == 2
    assert sum(
        "李冰" in citation["selected_text"]
        for citation in prediction_value["result"]["citations"]
    ) == 1
    aggregate = score_joint_retrieval(
        question, prediction, label, selection, combined,
        alias_path=aliases, aggregate_path=tmp_path / "aggregate.json",
        scope="DEVELOPMENT")
    assert aggregate["status"] == "PASS"
    assert aggregate["recall_at_20_count"] == 1
    assert aggregate["evidence_hit_count"] == 1
    assert aggregate["source_page_gold_coverage_count"] == 1
    assert aggregate["conditional_evidence_hit_ppm"] == 1_000_000
    dimensions = tmp_path / "dimensions.jsonl"
    dimensions.write_bytes(canonical_json_line({
        "dimension": "RELATION", "format_version": 1,
        "item_id": item_id,
        "record_kind": INTERACTIVE_DIMENSION_RECORD_KIND,
    }))
    dimension_report = publish_interactive_dimension_report(
        questions_path=question, labels_path=label,
        dimensions_path=dimensions, predictions_path=prediction,
        aggregate_path=tmp_path / "aggregate.json", aliases_path=aliases,
        database_path=combined, report_path=tmp_path / "dimension-report.json")
    assert dimension_report["global_counts"] == {
        "answer_count": 1,
        "citation_valid_count": 1,
        "evidence_hit_count": 1,
        "recall_at_20_count": 1,
        "source_page_gold_coverage_count": 1,
        "top1_source_hit_count": 1,
    }
    assert dimension_report["per_dimension"]["RELATION"][
        "evidence_hit_ppm"] == 1_000_000
    assert dimension_report["boundary"] == (
        "DEVELOPMENT_SURFACE_BUCKETS_NOT_SEMANTIC_UNDERSTANDING")
    assert dimension_report["failure_counts"] == {}
    assert dimension_report["status_counts"] == {"ANSWER": 1}


def test_production_query_prefers_unknown_for_explicit_fiction_and_clarifies(
        tmp_path: Path) -> None:
    """同一真实索引上，虚构限定须优先 UNKNOWN，真实多义词仍 CLARIFY。"""
    database = tmp_path / "database.sqlite3"
    _database(database, ((2, "亚马逊河", (
        "亚马逊：\n* 亚马逊河：南美洲河流。\n* 亚马逊公司：电子商务公司。",
        "亚马逊公司成立于二十世纪九十年代。",
    )),))
    connection = sqlite3.connect(str(database))
    try:
        fictional = query_broad_qa(connection, "虚构的亚马逊成立于何时？")
        ambiguous = query_broad_qa(connection, "亚马逊是什么？")
    finally:
        connection.close()
    assert fictional.status == "UNKNOWN"
    assert ambiguous.status == "CLARIFY"


def test_page_evidence_prefers_distinct_passages_before_duplicate_windows(
        tmp_path: Path) -> None:
    """同页高分段不得以重叠窗口挤掉其他 passage 的证据。"""
    database = tmp_path / "database.sqlite3"
    _database(database, ((2, "都江堰", (
        "都江堰水利工程用于防洪。都江堰水利工程用于灌溉。"
        "都江堰水利工程历史悠久。都江堰水利工程影响深远。",
        "这项工程由李冰主持修建。",
        "都江堰位于四川。",
        "都江堰至今仍在发挥作用。",
    )),))
    connection = sqlite3.connect(str(database))
    try:
        from pure_integer_ai.experiments.ph2_broad_qa_query import (
            retrieve_broad_qa_candidates,
        )
        _, trace = retrieve_broad_qa_candidates(
            connection, "谁主持修建都江堰？")
    finally:
        connection.close()
    passage_ids = tuple(
        item.passage_id for item in trace.evidence_candidates)
    assert len(passage_ids) == 4
    assert len(set(passage_ids)) == 4


def test_explicit_number_coverage_uses_exact_same_page_window_only() -> None:
    """缺失数字可替换低优先窗口，但不得把 2014 当作 014。"""
    def item(score: int, passage_id: int, text: str):
        """构造只含选择器所需列的稳定窗口。"""
        row = (passage_id, 0, len(text), "a" * 64, text)
        return ((score, 1, 0, -1, 0), -passage_id, row, text)

    answer = item(50, 1, "反驱逐舰驱逐舰的代表为空想级。")
    exact = item(49, 1, "在1920年代，这类舰艇被称为反驱逐舰驱逐舰。")
    distractors = tuple(
        item(score, passage_id, f"无关证据{passage_id}。")
        for score, passage_id in ((40, 2), (30, 3), (20, 4)))
    ranked = [answer, exact, *distractors]
    selected = (answer, *distractors)
    covered = _cover_explicit_number_evidence_windows(
        ranked, selected, explicit_numbers=frozenset({"1920"}), limit=4)
    assert exact in covered
    assert answer in covered
    assert len(covered) == 4

    near = item(49, 1, "2014年，球队迁入新球场。")
    unchanged = _cover_explicit_number_evidence_windows(
        [answer, near, *distractors], selected,
        explicit_numbers=frozenset({"014"}), limit=4)
    assert unchanged == selected


def test_structural_expansion_stays_in_passage_and_adds_required_context(
        ) -> None:
    """因果问式和前向指代只扩展同 passage 的连续候选窗口。"""
    def item(score: int, passage_id: int, passage: str, text: str):
        """构造带真实 passage 文本的稳定窗口。"""
        row = (passage_id, 0, len(passage), "a" * 64, passage)
        return ((score, 1, 0, -1, 0), -passage_id, row, text)

    cause_passage = "该经济体长期稳定。其金融体系安全，使该国成为避风港。"
    cause = item(50, 1, cause_passage, "该经济体长期稳定。")
    cause_context = item(49, 1, cause_passage, cause_passage)
    unrelated = item(60, 2, "因为天气晴朗。", "因为天气晴朗。")
    expanded = _expand_structural_evidence_windows(
        [unrelated, cause, cause_context], (cause,),
        answer_kinds=("CAUSE",))
    assert expanded == (cause_context,)

    reference_passage = "该门同时有无性及有性形态。前者常见于幼生寄主。"
    reference = item(50, 3, reference_passage, "前者常见于幼生寄主。")
    reference_context = item(49, 3, reference_passage, reference_passage)
    expanded_reference = _expand_structural_evidence_windows(
        [reference, reference_context], (reference,), answer_kinds=())
    assert expanded_reference == (reference_context,)


def test_joint_augmentation_can_extend_an_existing_alias_index(
        tmp_path: Path) -> None:
    """链式增量须保留旧 alias 表，并把新页和新 alias 追加进去。"""
    base = tmp_path / "base.sqlite3"
    first = tmp_path / "first.sqlite3"
    second = tmp_path / "second.sqlite3"
    first_extra = tmp_path / "first-extra.sqlite3"
    second_extra = tmp_path / "second-extra.sqlite3"
    _database(base, ((1, "基页", "基页包含基础说明。"),))
    _database(first_extra, ((2, "甲工程", "甲工程由甲主持。"),))
    _database(second_extra, ((3, "乙工程", "乙工程由乙主持。"),))

    def alias(path: Path, page_id: int, title: str) -> None:
        """写入一条解析完成的合成 alias。"""
        path.write_bytes(canonical_json_line({
            "chain": [title], "format_version": 1,
            "original_surfaces": [title], "record_kind": JOINT_ALIAS_KIND,
            "status": "RESOLVED", "terminal_page_id": page_id,
            "terminal_revision_id": 1000 + page_id,
            "terminal_title": title,
            "terminal_title_key": normalize_external_text(title),
            "title_key": normalize_external_text(title),
        }))

    first_alias = tmp_path / "first-alias.jsonl"
    second_alias = tmp_path / "second-alias.jsonl"
    alias(first_alias, 2, "甲工程")
    alias(second_alias, 3, "乙工程")
    augment_broad_qa_index(
        base, first_extra, output_database_path=first,
        base_expected_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        target_selection_sha256="1" * 64, alias_path=first_alias)
    report = augment_broad_qa_index(
        first, second_extra, output_database_path=second,
        base_expected_sha256=hashlib.sha256(first.read_bytes()).hexdigest(),
        target_selection_sha256="2" * 64, alias_path=second_alias)
    assert report["added_document_count"] == 1
    connection = sqlite3.connect(str(second))
    try:
        assert connection.execute(
            "SELECT surface FROM alias ORDER BY surface").fetchall() == [
                ("乙工程",), ("甲工程",)]
        assert connection.execute(
            "SELECT COUNT(*) FROM alias_term").fetchone()[0] > 0
        assert connection.execute(
            "SELECT page_id FROM document ORDER BY page_id").fetchall() == [
                (1,), (2,), (3,)]
    finally:
        connection.close()


def test_joint_score_rejects_one_tampered_chain_citation(
        tmp_path: Path) -> None:
    """证据链任一 selected span 被篡改时，整条 ANSWER 引用必须失效。"""
    base = tmp_path / "base.sqlite3"
    extra = tmp_path / "extra.sqlite3"
    combined = tmp_path / "combined.sqlite3"
    _database(base, ((1, "干扰页", "与问题无关。"),))
    _database(extra, ((2, "都江堰", (
        "都江堰用于防洪。", "工程由李冰主持修建。")),))
    aliases = tmp_path / "aliases.jsonl"
    aliases.write_bytes(canonical_json_line({
        "chain": ["都江堰"], "format_version": 1,
        "original_surfaces": ["都江堰"], "record_kind": JOINT_ALIAS_KIND,
        "status": "RESOLVED", "terminal_page_id": 2,
        "terminal_revision_id": 1002, "terminal_title": "都江堰",
        "terminal_title_key": normalize_external_text("都江堰"),
        "title_key": normalize_external_text("都江堰"),
    }))
    augment_broad_qa_index(
        base, extra, output_database_path=combined,
        base_expected_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        target_selection_sha256="b" * 64, alias_path=aliases)
    item_id = "d" * 64
    question = tmp_path / "questions.jsonl"
    label = tmp_path / "labels.jsonl"
    prediction = tmp_path / "predictions.jsonl"
    question.write_bytes(canonical_json_line({
        "format_version": 1, "item_id": item_id,
        "license_id": "CC-BY-SA-4.0", "question": "谁主持修建都江堰？",
        "record_kind": JOINT_QUESTION_KIND, "source_key": "CMRC2018",
        "source_partition": "dev", "source_question_id": "q2",
        "source_revision": "r1", "split": "dev",
        "upstream_url": "https://example.test/source",
    }))
    label.write_bytes(canonical_json_line({
        "expected_title_key": normalize_external_text("都江堰"),
        "format_version": 1, "gold_answers": ["李冰"], "item_id": item_id,
        "record_kind": "PH2_BROAD_QA_JOINT_LABEL_V1", "split": "dev",
    }))
    predict_joint_retrieval(question, combined, predictions_path=prediction)
    value = json.loads(prediction.read_text(encoding="utf-8"))
    value["result"]["citations"][1]["selected_text"] = "伪造证据"
    prediction.write_bytes(canonical_json_line(value))
    selected = BroadQaSelectedPage(
        1, "1" * 64, "都江堰",
        hashlib.sha256("都江堰".encode()).hexdigest(), 2, 1, 0, 1)
    selection = BroadQaTargetSelectionManifest(
        "ZHWIKIPEDIA_20260701", "synthetic", "a" * 64, "b" * 64,
        "c" * 40, "d" * 64, 1, 1, 1, "e" * 64, (selected,), ())
    aggregate = score_joint_retrieval(
        question, prediction, label, selection, combined,
        alias_path=aliases, aggregate_path=tmp_path / "aggregate.json",
        scope="DEVELOPMENT")
    assert aggregate["status"] == "FAIL"
    assert aggregate["answer_citation_valid_count"] == 0
    assert aggregate["evidence_hit_count"] == 0


def test_joint_score_separates_gold_absent_from_snapshot(
        tmp_path: Path) -> None:
    """旧基准金答案不在冻结终页时，必须从算法失败中独立分账。"""
    base = tmp_path / "base.sqlite3"
    extra = tmp_path / "extra.sqlite3"
    combined = tmp_path / "combined.sqlite3"
    _database(base, ((1, "干扰页", "与问题无关。"),))
    _database(extra, ((2, "都江堰", (
        "都江堰是一项古代水利工程。",
        "当前冻结修订只记录其防洪与灌溉作用。",
    )),))
    aliases = tmp_path / "aliases.jsonl"
    aliases.write_bytes(canonical_json_line({
        "chain": ["都江堰"], "format_version": 1,
        "original_surfaces": ["都江堰"], "record_kind": JOINT_ALIAS_KIND,
        "status": "RESOLVED", "terminal_page_id": 2,
        "terminal_revision_id": 1002, "terminal_title": "都江堰",
        "terminal_title_key": normalize_external_text("都江堰"),
        "title_key": normalize_external_text("都江堰"),
    }))
    augment_broad_qa_index(
        base, extra, output_database_path=combined,
        base_expected_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        target_selection_sha256="b" * 64, alias_path=aliases)
    item_id = "e" * 64
    question = tmp_path / "questions.jsonl"
    label = tmp_path / "labels.jsonl"
    prediction = tmp_path / "predictions.jsonl"
    question.write_bytes(canonical_json_line({
        "format_version": 1, "item_id": item_id,
        "license_id": "CC-BY-SA-4.0", "question": "谁主持修建都江堰？",
        "record_kind": JOINT_QUESTION_KIND, "source_key": "CMRC2018",
        "source_partition": "dev", "source_question_id": "q3",
        "source_revision": "r1", "split": "dev",
        "upstream_url": "https://example.test/source",
    }))
    label.write_bytes(canonical_json_line({
        "expected_title_key": normalize_external_text("都江堰"),
        "format_version": 1, "gold_answers": ["李冰"], "item_id": item_id,
        "record_kind": "PH2_BROAD_QA_JOINT_LABEL_V1", "split": "dev",
    }))
    predict_joint_retrieval(question, combined, predictions_path=prediction)
    selected = BroadQaSelectedPage(
        1, "1" * 64, "都江堰",
        hashlib.sha256("都江堰".encode()).hexdigest(), 2, 1, 0, 1)
    selection = BroadQaTargetSelectionManifest(
        "ZHWIKIPEDIA_20260701", "synthetic", "a" * 64, "b" * 64,
        "c" * 40, "d" * 64, 1, 1, 1, "e" * 64, (selected,), ())
    aggregate = score_joint_retrieval(
        question, prediction, label, selection, combined,
        alias_path=aliases, aggregate_path=tmp_path / "aggregate.json",
        scope="DEVELOPMENT")
    assert aggregate["status"] == "FAIL"
    assert aggregate["source_page_gold_coverage_count"] == 0
    assert aggregate["conditional_evidence_hit_ppm"] == 0
    assert aggregate["evidence_hit_count"] == 0
    assert aggregate["failure_counts"] == {
        "SOURCE_GOLD_ABSENT_FROM_SNAPSHOT": 1,
    }
