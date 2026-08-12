"""复跑公开 10k 广域问答开发探针，并可从冻结来源重算引用。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaSelectedPage,
    BroadQaSelectionManifest,
)
from pure_integer_ai.experiments.ph2_broad_qa_query import query_broad_qa
from pure_integer_ai.experiments.ph2_broad_qa_selection import (
    read_broad_qa_selection,
)
from pure_integer_ai.experiments.ph2_broad_qa_source import (
    iter_broad_qa_candidate_pages,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


REPOSITORY = Path(__file__).resolve().parents[1]
QUESTION_PATH = REPOSITORY / "data/ph2/broad_qa_dev_questions_v1.json"
QUESTION_SHA256 = "fe1b5f8ca9ce9904936442604e7c5a901f1ecebc4781c5f19e9c08392958ad93"


def _sha256_path(path: Path) -> str:
    """流式计算大 artifact SHA，避免随索引规模线性占用内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _positive(value: str) -> int:
    """把命令行重复次数解析为正严格整数。"""
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _questions() -> tuple[str, ...]:
    """严格加载公开 CC0 开发问题，拒绝 envelope、字节或 SHA 漂移。"""
    payload = QUESTION_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != QUESTION_SHA256:
        raise RuntimeError("broad QA dev question artifact SHA 漂移")
    value = json.loads(payload.decode("utf-8"))
    keys = {
        "artifact_kind", "format_version", "license_id", "questions",
        "scope", "source_identity",
    }
    if (not isinstance(value, dict) or set(value) != keys
            or value["artifact_kind"] != "PH2_BROAD_QA_DEV_QUESTIONS_V1"
            or value["format_version"] != 1
            or value["license_id"] != "CC0-1.0"
            or value["scope"] != "DEVELOPMENT_VERTICAL_PROBE_NOT_HELD_OUT"
            or value["source_identity"]
            != "AUTHORED_CC0_BROAD_QA_DEV_QUESTIONS_V1"
            or not isinstance(value["questions"], list)
            or len(value["questions"]) != 24
            or len(set(value["questions"])) != 24
            or any(not isinstance(item, str) or not item.strip()
                   for item in value["questions"])
            or canonical_json_bytes(value) + b"\n" != payload):
        raise RuntimeError("broad QA dev question artifact 非规范")
    return tuple(value["questions"])


def _percentile(values: list[int], numerator: int, denominator: int) -> int:
    """按 nearest-rank 返回非空整数延迟序列的百分位。"""
    ordered = sorted(values)
    ordinal = max(
        0,
        (len(ordered) * numerator + denominator - 1) // denominator - 1,
    )
    return ordered[ordinal]


def _source_pages(results, *, selection_path: Path, xml_path: Path):
    """只解压实际 ANSWER 涉及的冻结 block，恢复原始页面。"""
    page_ids = {
        result.page_id for result in results if result.status == "ANSWER"
    }
    source = read_broad_qa_selection(selection_path)
    originals = tuple(
        page for page in source.selected_pages if page.page_id in page_ids)
    selected = tuple(
        BroadQaSelectedPage(
            ordinal,
            page.rank_sha256,
            page.title,
            page.title_sha256,
            page.page_id,
            page.index_line_number,
            page.compressed_block_offset,
            page.compressed_block_end_offset,
        )
        for ordinal, page in enumerate(originals, start=1)
    )
    subset = BroadQaSelectionManifest(
        source.source_key,
        source.snapshot_id,
        source.snapshot_manifest_sha256,
        source.index_local_sha256,
        source.index_upstream_sha1,
        source.xml_local_sha256,
        source.xml_compressed_size_bytes,
        source.index_entry_count,
        len(selected),
        selected,
    )
    pages = {
        page.page_id: page for page in iter_broad_qa_candidate_pages(
            subset, xml_path=xml_path, worker_count=4)
    }
    return source, selected, pages


def _citation_checks(result, page) -> dict[str, bool]:
    """从原始 Wikitext 重算 ANSWER 的 span、hash 和来源身份。"""
    span_valid = (
        page is not None
        and 0 <= result.evidence_raw_start < result.evidence_raw_end
        <= len(page.wikitext)
    )
    raw = (
        page.wikitext[result.evidence_raw_start:result.evidence_raw_end]
        if span_valid else None
    )
    return {
        "answer_in_evidence": result.answer in result.evidence_text,
        "page_found": page is not None,
        "raw_sha256_equal": (
            raw is not None
            and hashlib.sha256(raw.encode("utf-8")).hexdigest()
            == result.evidence_raw_sha256
        ),
        "revision_equal": (
            page is not None and page.revision_id == result.revision_id),
        "revision_timestamp_equal": (
            page is not None
            and page.timestamp == result.revision_timestamp),
        "span_valid": span_valid,
        "title_equal": page is not None and page.title == result.title,
        "contributor_equal": (
            page is not None
            and page.contributor_json == result.contributor_json),
        "url_equal": result.source_url == (
            "https://zh.wikipedia.org/w/index.php?curid="
            f"{result.page_id}&oldid={result.revision_id}"
        ),
    }


def main() -> int:
    """执行固定问题、重复 warm 查询和可选的 snapshot 引用审计。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--repeat", type=_positive, default=3)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--xml", type=Path)
    args = parser.parse_args()
    if (args.selection is None) != (args.xml is None):
        parser.error("--selection and --xml must be provided together")

    questions = _questions()
    database = args.database.resolve()
    connection = sqlite3.connect(
        "file:" + database.as_posix() + "?mode=ro", uri=True)
    query_broad_qa(connection, "预热查询并不存在的限定对象")
    results = []
    latencies = []
    for repetition in range(args.repeat):
        for question in questions:
            started = time.perf_counter_ns()
            result = query_broad_qa(connection, question)
            elapsed_ns = time.perf_counter_ns() - started
            latencies.append(elapsed_ns)
            if repetition == 0:
                results.append(result)
                print(json.dumps(
                    {**result.to_dict(), "elapsed_ns": elapsed_ns},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ))
    connection.close()

    audit_failures = None
    answer_page_count = None
    selection_sha256 = None
    source_block_count = None
    if args.selection is not None:
        source, selected, pages = _source_pages(
            results,
            selection_path=args.selection.resolve(),
            xml_path=args.xml.resolve(),
        )
        failures = []
        for result in results:
            if result.status != "ANSWER":
                continue
            checks = _citation_checks(result, pages.get(result.page_id))
            if not all(checks.values()):
                failures.append({"checks": checks, "question": result.question})
        audit_failures = failures
        answer_page_count = len(pages)
        selection_sha256 = source.sha256()
        source_block_count = len({
            (page.compressed_block_offset, page.compressed_block_end_offset)
            for page in selected
        })

    aggregate = {
        "answer_count": sum(item.status == "ANSWER" for item in results),
        "answer_page_count": answer_page_count,
        "citation_audit_failure_count": (
            None if audit_failures is None else len(audit_failures)),
        "citation_audit_failures": audit_failures,
        "clarify_count": sum(item.status == "CLARIFY" for item in results),
        "conflict_count": sum(item.status == "CONFLICT" for item in results),
        "database_sha256": _sha256_path(database),
        "question_artifact_sha256": QUESTION_SHA256,
        "question_count": len(results),
        "query_count": len(latencies),
        "repeat_count": args.repeat,
        "scope": "DEVELOPMENT_VERTICAL_PROBE_NOT_HELD_OUT",
        "selection_sha256": selection_sha256,
        "source_block_count": source_block_count,
        "unknown_count": sum(item.status == "UNKNOWN" for item in results),
        "warm_max_ns": max(latencies),
        "warm_p50_ns": _percentile(latencies, 50, 100),
        "warm_p95_ns": _percentile(latencies, 95, 100),
    }
    print(json.dumps(
        {"aggregate": aggregate},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ))
    return int(audit_failures is not None and bool(audit_failures))


if __name__ == "__main__":
    raise SystemExit(main())
