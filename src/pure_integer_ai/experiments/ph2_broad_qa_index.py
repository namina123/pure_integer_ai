"""构建来源约束广域问答 V0 的紧凑整数段落索引。"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import re
import sqlite3
import time

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaSelectionManifest,
    BroadQaTargetSelectionManifest,
)
from pure_integer_ai.experiments.ph2_broad_qa_source import (
    iter_broad_qa_candidate_pages,
    project_broad_qa_passages,
)
from pure_integer_ai.experiments.v02_run_store import HostProcessMemory
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


INDEX_SCHEMA_VERSION = 1
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


# object-model: exception
class BroadQaIndexError(RuntimeError):
    """广域问答索引路径、schema、事务或身份发生漂移。"""


def broad_qa_terms(text: str) -> tuple[str, ...]:
    """生成确定性的中文二/三元组与 ASCII 单词稀疏特征。"""
    if not isinstance(text, str):
        raise TypeError("broad QA term text 必须是字符串")
    values: set[str] = set()
    for match in _CJK_RE.finditer(text):
        sequence = match.group(0)
        if len(sequence) == 1:
            values.add("c:" + sequence)
        for width in (2, 3):
            values.update(
                "c:" + sequence[index:index + width]
                for index in range(max(0, len(sequence) - width + 1))
            )
    values.update("w:" + item.group(0).casefold()
                  for item in _WORD_RE.finditer(text))
    return tuple(sorted(values))


def _delta(values: list[int]) -> tuple[int, ...]:
    """把严格递增 posting id 列表转为正整数 delta tuple。"""
    result = []
    prior = 0
    for value in values:
        if type(value) is not int or value <= prior:
            raise BroadQaIndexError("broad QA posting 不是严格递增整数")
        result.append(value - prior)
        prior = value
    return tuple(result)


def _database_sha256(path: Path) -> str:
    """以固定块计算已关闭 SQLite artifact 的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _memory_sample() -> tuple[int, int]:
    """读取当前工作集与进程峰值；宿主不支持时保留零证据。"""
    values = HostProcessMemory()()
    return (
        int(values.get("current_working_set_bytes", 0)),
        int(values.get("process_peak_working_set_bytes", 0)),
    )


def build_broad_qa_index(
        selection: BroadQaSelectionManifest,
        *,
        xml_path: str | Path,
        database_path: str | Path,
        accepted_page_limit: int | None,
        worker_count: int = 1,
        ) -> dict[str, object]:
    """流式读取主空间页，写文档/证据表并生成 delta-varint postings。"""
    if not isinstance(selection, (
            BroadQaSelectionManifest, BroadQaTargetSelectionManifest)):
        raise TypeError("broad QA selection 类型错误")
    target = Path(database_path).resolve()
    if target.exists():
        raise BroadQaIndexError("broad QA database 禁止覆盖")
    target.parent.mkdir(parents=True, exist_ok=True)
    unique_blocks = {
        (item.compressed_block_offset, item.compressed_block_end_offset)
        for item in selection.selected_pages
    }
    compressed_bytes_read = sum(end - start for start, end in unique_blocks)
    started_ns = time.perf_counter_ns()
    rss_before_bytes, peak_before_bytes = _memory_sample()
    source_started_ns = time.perf_counter_ns()
    eligible_page_count = 0
    connection = sqlite3.connect(str(target))
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE document(
                doc_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                page_id INTEGER NOT NULL UNIQUE,
                revision_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                contributor_json TEXT NOT NULL,
                text_sha256 TEXT NOT NULL
            );
            CREATE TABLE passage(
                passage_id INTEGER PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                raw_start INTEGER NOT NULL,
                raw_end INTEGER NOT NULL,
                raw_sha256 TEXT NOT NULL,
                text TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                section_title TEXT NOT NULL
            );
            CREATE TABLE posting(
                term TEXT PRIMARY KEY,
                document_frequency INTEGER NOT NULL,
                passage_deltas BLOB NOT NULL
            );
            CREATE INDEX passage_doc ON passage(doc_id, ordinal);
            CREATE INDEX document_title ON document(title);
            CREATE TABLE candidate(
                selection_ordinal INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                page_id INTEGER NOT NULL UNIQUE,
                revision_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                contributor_json TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                wikitext TEXT NOT NULL
            );
        """)
        for page in iter_broad_qa_candidate_pages(
                selection, xml_path=xml_path, worker_count=worker_count):
            eligible_page_count += 1
            connection.execute(
                "INSERT INTO candidate VALUES(?,?,?,?,?,?,?,?)",
                (page.ordinal, page.title, page.page_id, page.revision_id,
                 page.timestamp, page.contributor_json, page.text_sha256,
                 page.wikitext),
            )
        source_elapsed_ns = max(1, time.perf_counter_ns() - source_started_ns)

        projection_started_ns = time.perf_counter_ns()
        accepted = 0
        for row in connection.execute("""
                SELECT selection_ordinal,title,page_id,revision_id,timestamp,
                       contributor_json,text_sha256,wikitext
                FROM candidate ORDER BY selection_ordinal
                """):
            (selection_ordinal, title, page_id, revision_id, timestamp,
             contributor_json, text_sha256, wikitext) = row
            passages = project_broad_qa_passages(wikitext)
            if not passages:
                continue
            accepted += 1
            connection.execute(
                "INSERT INTO document VALUES(?,?,?,?,?,?,?)",
                (selection_ordinal, title, page_id, revision_id, timestamp,
                 contributor_json, text_sha256),
            )
            for passage in passages:
                passage_id = selection_ordinal * 128 + passage.ordinal
                connection.execute(
                    "INSERT INTO passage VALUES(?,?,?,?,?,?,?,?,?)",
                    (passage_id, selection_ordinal, passage.ordinal,
                     passage.raw_start, passage.raw_end, passage.raw_sha256,
                     passage.text, passage.text_sha256,
                     passage.section_title),
                )
            if (accepted_page_limit is not None
                    and accepted >= accepted_page_limit):
                break
        projection_elapsed_ns = max(
            1, time.perf_counter_ns() - projection_started_ns)
        if accepted_page_limit is not None and accepted != accepted_page_limit:
            raise BroadQaIndexError(
                "broad QA candidate 不足: "
                f"requested={accepted_page_limit}, "
                f"main_nonredirect={eligible_page_count}, projected={accepted}")
        if accepted == 0:
            raise BroadQaIndexError("broad QA 没有可投影页面")
        connection.execute("DROP TABLE candidate")

        postings_started_ns = time.perf_counter_ns()
        postings: dict[str, list[int]] = defaultdict(list)
        passage_count = 0
        for row in connection.execute("""
                SELECT p.passage_id,d.title,p.section_title,p.text
                FROM passage AS p JOIN document AS d ON d.doc_id=p.doc_id
                ORDER BY p.passage_id
                """):
            passage_id, title, section_title, text = row
            passage_count += 1
            terms = set(broad_qa_terms(title))
            terms.update(broad_qa_terms(section_title))
            terms.update(broad_qa_terms(text))
            for term in terms:
                postings[term].append(passage_id)
        for term in sorted(postings):
            values = postings[term]
            connection.execute(
                "INSERT INTO posting VALUES(?,?,?)",
                (term, len(values), encode_integer_tuple(_delta(values))),
            )
        postings_elapsed_ns = max(
            1, time.perf_counter_ns() - postings_started_ns)
        metadata = {
            "accepted_page_count": str(accepted),
            "index_schema_version": str(INDEX_SCHEMA_VERSION),
            "license_id": "CC-BY-SA-4.0",
            "passage_count": str(passage_count),
            "selection_sha256": selection.sha256(),
            "snapshot_id": selection.snapshot_id,
            "source_key": selection.source_key,
            "term_count": str(len(postings)),
        }
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)", sorted(metadata.items()))
        finalize_started_ns = time.perf_counter_ns()
        connection.commit()
        connection.execute("VACUUM")
        finalize_elapsed_ns = max(
            1, time.perf_counter_ns() - finalize_started_ns)
    except Exception:
        connection.close()
        raise
    finally:
        if connection:
            connection.close()
    rss_after_bytes, peak_after_bytes = _memory_sample()
    total_elapsed_ns = max(1, time.perf_counter_ns() - started_ns)
    return {
        "accepted_page_count": accepted,
        "candidate_page_count": len(selection.selected_pages),
        "compressed_block_count": len(unique_blocks),
        "compressed_bytes_read": compressed_bytes_read,
        "database_bytes": target.stat().st_size,
        "database_sha256": _database_sha256(target),
        "eligible_page_count": eligible_page_count,
        "finalize_elapsed_ns": finalize_elapsed_ns,
        "passage_count": passage_count,
        "postings_elapsed_ns": postings_elapsed_ns,
        "projection_elapsed_ns": projection_elapsed_ns,
        "process_peak_working_set_bytes": max(
            peak_before_bytes, peak_after_bytes),
        "rss_after_bytes": rss_after_bytes,
        "rss_before_bytes": rss_before_bytes,
        "selection_sha256": selection.sha256(),
        "source_elapsed_ns": source_elapsed_ns,
        "term_count": len(postings),
        "total_elapsed_ns": total_elapsed_ns,
        "worker_count": worker_count,
    }


__all__ = [
    "BroadQaIndexError",
    "INDEX_SCHEMA_VERSION",
    "broad_qa_terms",
    "build_broad_qa_index",
]
