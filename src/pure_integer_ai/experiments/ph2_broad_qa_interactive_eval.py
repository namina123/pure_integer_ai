"""发布交互开发集的问式分账与拒答/澄清回归报告。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_interactive_family import (
    INTERACTIVE_DIMENSION_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_joint_eval import (
    JOINT_AGGREGATE_KIND,
    JOINT_ALIAS_KIND,
    JOINT_LABEL_KIND,
    JOINT_PREDICTION_KIND,
    JOINT_QUESTION_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_query import query_broad_qa
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


REPOSITORY = Path(__file__).resolve().parents[3]
REFUSAL_PROBE_RELATIVE_PATH = Path(
    "data/ph2/broad_qa_interactive_refusal_probes_v1.json")
REFUSAL_PROBE_PATH = REPOSITORY / REFUSAL_PROBE_RELATIVE_PATH
REFUSAL_PROBE_SHA256 = (
    "a5c51f68fdfd6caf3f4d04d921e25370bb953a0746d1fa5a1b3daad839b80178")
DIMENSION_REPORT_KIND = "PH2_BROAD_QA_INTERACTIVE_DIMENSION_REPORT_V1"
REFUSAL_REPORT_KIND = "PH2_BROAD_QA_INTERACTIVE_REFUSAL_REPORT_V1"
_DIMENSIONS = ("CAUSE", "COMPARISON", "TIME", "QUANTITY", "RELATION")


def _sha256_file(path: Path) -> str:
    """流式计算输入、数据库或报告 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path, kind: str) -> tuple[dict[str, object], ...]:
    """回读规范 JSONL 并核验 record kind 与唯一身份。"""
    if not path.is_file():
        raise BroadQaExternalDataError("interactive report input 缺失")
    values = []
    identities = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                identity = (value.get("item_id", value.get("title_key"))
                            if isinstance(value, dict) else None)
                if (not line.endswith("\n") or not isinstance(value, dict)
                        or value.get("record_kind") != kind
                        or not isinstance(identity, str) or not identity
                        or identity in identities):
                    raise BroadQaExternalDataError(
                        f"interactive report record 漂移: {line_number}")
                identities.add(identity)
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("interactive report JSONL 非法") from error
    if not values:
        raise BroadQaExternalDataError("interactive report JSONL 为空")
    return tuple(values)


def _publish(path: Path, value: dict[str, object]) -> dict[str, object]:
    """不可覆盖地发布 canonical JSON，并返回报告 SHA。"""
    if path.exists():
        raise BroadQaExternalDataError("interactive report 禁止覆盖")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_line(value))
    return {**value, "report_sha256": _sha256_file(path)}


def _ppm(numerator: int, denominator: int) -> int:
    """以整数百万分率表达开发结果。"""
    return 0 if denominator == 0 else numerator * 1_000_000 // denominator


def publish_interactive_dimension_report(
        *, questions_path: str | Path, labels_path: str | Path,
        dimensions_path: str | Path, predictions_path: str | Path,
        aggregate_path: str | Path, aliases_path: str | Path,
        database_path: str | Path, report_path: str | Path,
        ) -> dict[str, object]:
    """强回验开发 aggregate 后，按公开问式表面主桶分账。"""
    question_file = Path(questions_path).resolve()
    label_file = Path(labels_path).resolve()
    dimension_file = Path(dimensions_path).resolve()
    prediction_file = Path(predictions_path).resolve()
    aggregate_file = Path(aggregate_path).resolve()
    alias_file = Path(aliases_path).resolve()
    database_file = Path(database_path).resolve()
    questions = _read_jsonl(question_file, JOINT_QUESTION_KIND)
    labels = _read_jsonl(label_file, JOINT_LABEL_KIND)
    dimensions = _read_jsonl(
        dimension_file, INTERACTIVE_DIMENSION_RECORD_KIND)
    predictions = _read_jsonl(prediction_file, JOINT_PREDICTION_KIND)
    aliases = _read_jsonl(alias_file, JOINT_ALIAS_KIND)
    mappings = tuple(
        {value["item_id"]: value for value in values}
        for values in (questions, labels, dimensions, predictions))
    inventory = set(mappings[0])
    if any(set(mapping) != inventory for mapping in mappings[1:]):
        raise BroadQaExternalDataError("interactive report inventory 漂移")
    alias_by_key = {value["title_key"]: value for value in aliases}
    try:
        aggregate_payload = aggregate_file.read_bytes()
        aggregate = json.loads(aggregate_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("interactive aggregate 非法") from error
    if (canonical_json_line(aggregate) != aggregate_payload
            or aggregate.get("artifact_kind") != JOINT_AGGREGATE_KIND
            or aggregate.get("scope") != "DEVELOPMENT"
            or aggregate.get("question_count") != len(inventory)
            or aggregate.get("questions_sha256") != _sha256_file(question_file)
            or aggregate.get("labels_sha256") != _sha256_file(label_file)
            or aggregate.get("predictions_sha256")
            != _sha256_file(prediction_file)
            or aggregate.get("database_sha256") != _sha256_file(database_file)
            or aggregate.get("alias_sha256") != _sha256_file(alias_file)):
        raise BroadQaExternalDataError("interactive aggregate commitment 漂移")
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    dimension_failures: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    global_failures: Counter[str] = Counter()
    global_statuses: Counter[str] = Counter()
    database = sqlite3.connect(f"file:{database_file}?mode=ro", uri=True)
    try:
        for item_id in sorted(inventory):
            question = mappings[0][item_id]
            label = mappings[1][item_id]
            dimension_record = mappings[2][item_id]
            prediction = mappings[3][item_id]
            dimension = dimension_record.get("dimension")
            if dimension not in _DIMENSIONS:
                raise BroadQaExternalDataError(
                    "interactive report dimension 漂移")
            expected_key = label.get("expected_title_key")
            alias = alias_by_key.get(expected_key)
            if (alias is None or alias.get("status") != "RESOLVED"
                    or prediction.get("source_key") != question.get("source_key")
                    or prediction.get("split") != "dev"):
                raise BroadQaExternalDataError(
                    "interactive report source binding 漂移")
            expected_page_id = alias["terminal_page_id"]
            normalized_gold = tuple(
                normalize_external_text(gold)
                for gold in label.get("gold_answers", ())
                if isinstance(gold, str) and gold)
            if not normalized_gold:
                raise BroadQaExternalDataError(
                    "interactive report gold inventory 漂移")
            page_projected = database.execute(
                "SELECT 1 FROM document WHERE page_id=?", (expected_page_id,)
            ).fetchone() is not None
            source_covered = page_projected and any(
                gold in normalize_external_text(row[0])
                for row in database.execute("""
                    SELECT p.text FROM passage AS p
                    JOIN document AS d ON d.doc_id=p.doc_id
                    WHERE d.page_id=? ORDER BY p.ordinal
                    """, (expected_page_id,))
                for gold in normalized_gold)
            candidates = prediction.get("candidates")
            result = prediction.get("result")
            if not isinstance(candidates, list) or not isinstance(result, dict):
                raise BroadQaExternalDataError(
                    "interactive report prediction schema 漂移")
            recall = any(
                value.get("page_id") == expected_page_id
                for value in candidates if isinstance(value, dict))
            top1 = bool(candidates) and candidates[0].get(
                "page_id") == expected_page_id
            answer = result.get("status") == "ANSWER"
            status = str(result.get("status"))
            citations = result.get("citations")
            selected_evidence = []
            citation_validity = []
            if answer and isinstance(citations, list) and 1 <= len(citations) <= 4:
                for citation in citations:
                    if not isinstance(citation, dict):
                        citation_validity.append(False)
                        continue
                    row = database.execute("""
                        SELECT p.text,d.title,d.revision_id
                        FROM passage AS p JOIN document AS d ON d.doc_id=p.doc_id
                        WHERE d.page_id=? AND d.revision_id=?
                          AND p.raw_start=? AND p.raw_end=? AND p.raw_sha256=?
                        """, (
                            citation.get("page_id"), citation.get("revision_id"),
                            citation.get("evidence_raw_start"),
                            citation.get("evidence_raw_end"),
                            citation.get("evidence_raw_sha256"),
                        )).fetchone()
                    selected_text = citation.get("selected_text")
                    valid_item = (
                        row is not None
                        and row[0] == citation.get("evidence_text")
                        and row[1] == citation.get("title")
                        and row[2] == citation.get("revision_id")
                        and isinstance(selected_text, str)
                        and selected_text and selected_text in row[0])
                    citation_validity.append(valid_item)
                    if valid_item:
                        selected_evidence.append(selected_text)
            valid = (bool(citation_validity) and all(citation_validity)
                     and result.get("answer") == "\n".join(selected_evidence))
            source_correct = (
                valid and all(citation.get("page_id") == expected_page_id
                              for citation in citations))
            evidence_hit = source_correct and any(
                gold in normalize_external_text("\n".join(selected_evidence))
                for gold in normalized_gold)
            values = counters[str(dimension)]
            values["question_count"] += 1
            values["recall_at_20_count"] += int(recall)
            values["top1_source_hit_count"] += int(top1)
            values["answer_count"] += int(answer)
            values["citation_valid_count"] += int(valid)
            values["evidence_hit_count"] += int(evidence_hit)
            values["source_page_gold_coverage_count"] += int(source_covered)
            global_counts["recall_at_20_count"] += int(recall)
            global_counts["top1_source_hit_count"] += int(top1)
            global_counts["answer_count"] += int(answer)
            global_counts["citation_valid_count"] += int(valid)
            global_counts["evidence_hit_count"] += int(evidence_hit)
            global_counts["source_page_gold_coverage_count"] += int(
                source_covered)
            global_statuses[status] += 1
            failure = None
            if not page_projected:
                failure = "SOURCE_PAGE_NOT_PROJECTED"
            elif not source_covered:
                failure = "SOURCE_GOLD_ABSENT_FROM_SNAPSHOT"
            elif not recall:
                failure = "RETRIEVAL_MISS_AT_20"
            elif not top1:
                failure = "TOP1_SOURCE_MISS"
            elif not answer:
                failure = "NON_ANSWER"
            elif not valid:
                failure = "CITATION_INVALID"
            elif not evidence_hit:
                failure = "GOLD_NOT_IN_EVIDENCE"
            if failure is not None:
                dimension_failures[str(dimension)][failure] += 1
                global_failures[failure] += 1
    finally:
        database.close()
    expected_global = {
        "recall_at_20_count": aggregate.get("recall_at_20_count"),
        "top1_source_hit_count": aggregate.get("top1_source_hit_count"),
        "answer_count": aggregate.get("answer_count"),
        "citation_valid_count": aggregate.get("answer_citation_valid_count"),
        "evidence_hit_count": aggregate.get("evidence_hit_count"),
        "source_page_gold_coverage_count": aggregate.get(
            "source_page_gold_coverage_count"),
    }
    expected_failures = aggregate.get("failure_counts")
    expected_statuses = aggregate.get("status_counts")
    if (dict(global_counts) != expected_global
            or dict(sorted(global_failures.items())) != expected_failures
            or dict(sorted(global_statuses.items())) != expected_statuses):
        raise BroadQaExternalDataError(
            "interactive dimension/global aggregate 不一致")
    per_dimension = {}
    for dimension in _DIMENSIONS:
        values = counters[dimension]
        total = values["question_count"]
        answers = values["answer_count"]
        per_dimension[dimension] = {
            **dict(values),
            "citation_valid_ppm": _ppm(
                values["citation_valid_count"], answers),
            "evidence_hit_ppm": _ppm(values["evidence_hit_count"], total),
            "failure_counts": dict(sorted(
                dimension_failures[dimension].items())),
            "recall_at_20_ppm": _ppm(values["recall_at_20_count"], total),
            "source_page_gold_coverage_ppm": _ppm(
                values["source_page_gold_coverage_count"], total),
            "conditional_evidence_hit_ppm": _ppm(
                values["evidence_hit_count"],
                values["source_page_gold_coverage_count"]),
            "top1_source_hit_ppm": _ppm(
                values["top1_source_hit_count"], total),
        }
    report = {
        "aggregate_sha256": _sha256_file(aggregate_file),
        "artifact_kind": DIMENSION_REPORT_KIND,
        "boundary": (
            "DEVELOPMENT_SURFACE_BUCKETS_NOT_SEMANTIC_UNDERSTANDING"),
        "database_sha256": _sha256_file(database_file),
        "dimensions_sha256": _sha256_file(dimension_file),
        "format_version": 1,
        "failure_counts": expected_failures,
        "global_counts": expected_global,
        "per_dimension": per_dimension,
        "question_count": len(inventory),
        "status": aggregate["status"],
        "status_counts": expected_statuses,
    }
    return _publish(Path(report_path).resolve(), report)


def load_refusal_probes(
        path: str | Path = REFUSAL_PROBE_PATH,
        ) -> tuple[dict[str, str], ...]:
    """严格加载公开 CC0 UNKNOWN/CLARIFY 开发探针。"""
    source = Path(path).resolve()
    try:
        payload = source.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("interactive refusal probes 非法") from error
    if (_sha256_file(source) != REFUSAL_PROBE_SHA256
            or canonical_json_line(value) != payload
            or not isinstance(value, dict)
            or set(value) != {
                "artifact_kind", "format_version", "language", "license_id",
                "probes", "source_identity"}
            or value["artifact_kind"]
            != "PH2_BROAD_QA_INTERACTIVE_REFUSAL_PROBES_V1"
            or value["format_version"] != 1 or value["language"] != "zh"
            or value["license_id"] != "CC0-1.0"
            or not isinstance(value["probes"], list)
            or not value["probes"]):
        raise BroadQaExternalDataError("interactive refusal probes 漂移")
    probes = []
    identities = set()
    for item in value["probes"]:
        if (not isinstance(item, dict)
                or set(item) != {"expected_status", "probe_id", "question"}
                or item["expected_status"] not in {"UNKNOWN", "CLARIFY"}
                or not isinstance(item["probe_id"], str)
                or not isinstance(item["question"], str)
                or not item["probe_id"] or not item["question"]
                or item["probe_id"] in identities):
            raise BroadQaExternalDataError(
                "interactive refusal probe record 漂移")
        identities.add(item["probe_id"])
        probes.append(item)
    return tuple(probes)


def publish_interactive_refusal_report(
        database_path: str | Path,
        *, report_path: str | Path,
        probes_path: str | Path = REFUSAL_PROBE_PATH,
        ) -> dict[str, object]:
    """在生产查询路径上运行原创 UNKNOWN/CLARIFY 开发回归。"""
    database_file = Path(database_path).resolve()
    probes_file = Path(probes_path).resolve()
    probes = load_refusal_probes(probes_file)
    results = []
    connection = sqlite3.connect(f"file:{database_file}?mode=ro", uri=True)
    try:
        for item in probes:
            result = query_broad_qa(connection, item["question"])
            results.append({
                "actual_status": result.status,
                "candidate_document_count": result.candidate_document_count,
                "expected_status": item["expected_status"],
                "matched_term_count": result.matched_term_count,
                "passed": int(result.status == item["expected_status"]),
                "probe_id": item["probe_id"],
                "question_sha256": hashlib.sha256(
                    item["question"].encode("utf-8")).hexdigest(),
            })
    finally:
        connection.close()
    passed = sum(item["passed"] for item in results)
    report = {
        "artifact_kind": REFUSAL_REPORT_KIND,
        "boundary": "AUTHORED_DEVELOPMENT_REGRESSION_NOT_FORMAL_HELD_OUT",
        "database_sha256": _sha256_file(database_file),
        "format_version": 1,
        "passed_count": passed,
        "probe_count": len(results),
        "probes_sha256": _sha256_file(probes_file),
        "results": results,
        "status": "PASS" if passed == len(results) else "FAIL",
    }
    return _publish(Path(report_path).resolve(), report)


__all__ = [
    "DIMENSION_REPORT_KIND",
    "REFUSAL_PROBE_PATH",
    "REFUSAL_PROBE_SHA256",
    "REFUSAL_REPORT_KIND",
    "load_refusal_probes",
    "publish_interactive_dimension_report",
    "publish_interactive_refusal_report",
]
