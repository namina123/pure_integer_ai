"""在紧凑整数 postings 上执行有界检索和来源约束回答。"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import re
import sqlite3

from opencc import OpenCC

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaResult,
    WIKIPEDIA_ATTRIBUTION,
)
from pure_integer_ai.experiments.ph2_broad_qa_index import broad_qa_terms
from pure_integer_ai.experiments.ph2_broad_qa_question_slots import (
    load_broad_qa_question_slots,
)
from pure_integer_ai.storage.integer_codec import decode_integer_tuple


_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?|[^。！？!?\n]+$")
_CJK_SEQUENCE_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+")
_NUMBER_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)?")
_TO_SIMPLIFIED = OpenCC("t2s")
_TO_TRADITIONAL = OpenCC("s2t")


# object-model: exception
class BroadQaQueryError(RuntimeError):
    """索引 schema、posting、查询预算或来源身份发生漂移。"""


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    """读取并核验运行时所需的完整 metadata 键。"""
    values = dict(connection.execute("SELECT key,value FROM metadata"))
    expected = {
        "accepted_page_count", "index_schema_version", "license_id",
        "passage_count", "selection_sha256", "snapshot_id", "source_key",
        "term_count",
    }
    if (set(values) != expected or values["index_schema_version"] != "1"
            or values["license_id"] != "CC-BY-SA-4.0"
            or values["source_key"] != "ZHWIKIPEDIA_20260701"):
        raise BroadQaQueryError("broad QA metadata/schema 漂移")
    return values


def _restore_postings(payload: bytes) -> tuple[int, ...]:
    """把正 delta varint 流恢复为严格递增 passage id。"""
    deltas = decode_integer_tuple(payload)
    result = []
    current = 0
    for value in deltas:
        if value <= 0:
            raise BroadQaQueryError("broad QA posting delta 非正")
        current += value
        result.append(current)
    return tuple(result)


@lru_cache(maxsize=16384)
def _script_terms(text: str) -> frozenset[str]:
    """生成原文、简体和繁体三种离散特征并集。"""
    values = set(broad_qa_terms(text))
    values.update(broad_qa_terms(_TO_SIMPLIFIED.convert(text)))
    values.update(broad_qa_terms(_TO_TRADITIONAL.convert(text)))
    return frozenset(values)


def _best_sentence(question_terms: set[str], text: str) -> str:
    """从已选证据段中选取覆盖问题特征最多的完整短句。"""
    candidates = tuple(
        (ordinal, item.group(0).strip())
        for ordinal, item in enumerate(_SENTENCE_RE.finditer(text))
        if item.group(0).strip()
    )
    if not candidates:
        return text[:280].strip()
    ranked = sorted(
        candidates,
        key=lambda item: (
            -len(question_terms.intersection(_script_terms(item[1]))),
            item[0],
            len(item[1]),
            item,
        ),
    )
    if not question_terms.intersection(_script_terms(ranked[0][1])):
        adjacent = "".join(item[1] for item in candidates[:2])
        return adjacent if len(adjacent) <= 360 else adjacent[:360].rstrip()
    answer = ranked[0][1]
    return answer if len(answer) <= 360 else answer[:360].rstrip()


def _best_sentence_overlap(question_terms: set[str], text: str) -> int:
    """返回段内单句对关系/属性问题特征的最大覆盖数。"""
    return max((
        len(question_terms.intersection(_script_terms(item.group(0))))
        for item in _SENTENCE_RE.finditer(text)
        if item.group(0).strip()
    ), default=0)


def _answer_kind_bonus(answer_kinds: tuple[str, ...], text: str) -> int:
    """用问式槽类型优先满足显式值形态的证据句。"""
    bonus = 0
    if "QUANTITY" in answer_kinds and _NUMBER_RE.search(text):
        bonus += 8_000_000
    if "TIME" in answer_kinds and _NUMBER_RE.search(text):
        bonus += 4_000_000
    return bonus


def _structural_residue_penalty(text: str) -> int:
    """对 parser 未消除的分类/表格结构残留施加确定性降权。"""
    stripped = text.lstrip()
    if stripped.startswith("Category:") or stripped.startswith("{|"):
        return 12_000_000
    return 0


def _title_span(question: str, title: str) -> tuple[int, int] | None:
    """在原文或简繁标准化问题中定位完整页面标题。"""
    candidates = (
        (question, title),
        (_TO_SIMPLIFIED.convert(question), _TO_SIMPLIFIED.convert(title)),
        (_TO_TRADITIONAL.convert(question), _TO_TRADITIONAL.convert(title)),
    )
    for surface, expected in candidates:
        start = surface.find(expected)
        if start >= 0:
            return start, start + len(expected)
    return None


def _missing_strong_constraint(
        question: str,
        *,
        title: str,
        candidate_terms: set[str],
        slot_free_question: str,
        candidate_text: str,
        ) -> bool:
    """审计标题前实体限定和显式数字，允许关系问式自由改写。"""
    span = _title_span(slot_free_question, title)
    if span is None:
        return False
    prefix = slot_free_question[:span[0]]
    for match in _CJK_SEQUENCE_RE.finditer(prefix):
        sequence = match.group(0)
        if len(sequence) < 2:
            continue
        terms = _script_terms(sequence)
        if not terms.intersection(candidate_terms):
            return True
    candidate_numbers = set(_NUMBER_RE.findall(candidate_text))
    return any(
        value not in candidate_numbers
        for value in _NUMBER_RE.findall(slot_free_question)
    )


def _is_ambiguous_list(text: str) -> bool:
    """识别一个证据段内列举多个同名候选的消歧页。"""
    return text.count("*") >= 2 and ("：" in text or ":" in text)


def query_broad_qa(
        connection: sqlite3.Connection,
        question: str,
        *,
        max_query_terms: int = 24,
        max_candidate_passages: int = 20,
        max_posting_visits: int = 500_000,
        ) -> BroadQaResult:
    """以稀有 term postings 缩小候选，并从 top-K 页面投影可引用回答。"""
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("broad QA connection 类型错误")
    if (not isinstance(question, str) or not question.strip()
            or len(question) > 2048):
        raise BroadQaQueryError("broad QA question 不能为空")
    if (type(max_query_terms) is not int or not 1 <= max_query_terms <= 128
            or type(max_candidate_passages) is not int
            or not 1 <= max_candidate_passages <= 200
            or type(max_posting_visits) is not int
            or not 1 <= max_posting_visits <= 2_000_000):
        raise BroadQaQueryError("broad QA query budget 非法")
    metadata = _metadata(connection)
    question_slots = load_broad_qa_question_slots()
    answer_kinds = question_slots.answer_kinds(question)
    slot_free_question = question_slots.strip_slots(question)
    raw_terms = tuple(sorted(_script_terms(slot_free_question)))
    if len(raw_terms) > 512:
        raise BroadQaQueryError("broad QA 原始 query term 超预算")
    if not raw_terms:
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, metadata["snapshot_id"], metadata["license_id"], 0, 0,
        )
    placeholders = ",".join("?" for _ in raw_terms)
    rows = tuple(connection.execute(
        "SELECT term,document_frequency,passage_deltas FROM posting "
        f"WHERE term IN ({placeholders})",
        raw_terms,
    ))
    rows = tuple(sorted(rows, key=lambda item: (item[1], item[0])))
    rows = rows[:max_query_terms]
    if not rows:
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, metadata["snapshot_id"], metadata["license_id"], 0, 0,
        )
    scores: dict[int, int] = defaultdict(int)
    matched: dict[int, int] = defaultdict(int)
    total_passages = int(metadata["passage_count"])
    posting_visits = 0
    for _, frequency, payload in rows:
        if posting_visits + frequency > max_posting_visits:
            raise BroadQaQueryError("broad QA posting visit 超预算")
        weight = max(1, 1_000_000 // frequency)
        restored = _restore_postings(payload)
        if len(restored) != frequency:
            raise BroadQaQueryError("broad QA posting frequency 漂移")
        posting_visits += len(restored)
        for passage_id in restored:
            scores[passage_id] += weight
            matched[passage_id] += 1
    ranked_ids = tuple(
        item[0] for item in sorted(
            scores.items(), key=lambda item: (-item[1], -matched[item[0]], item[0])
        )[:max_candidate_passages]
    )
    if not ranked_ids:
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, metadata["snapshot_id"], metadata["license_id"],
            len(rows), 0,
        )
    candidate_rows = []
    sentence_question = slot_free_question
    for passage_id in ranked_ids:
        row = connection.execute("""
            SELECT p.passage_id,p.raw_start,p.raw_end,p.raw_sha256,p.text,
                   d.doc_id,d.title,d.page_id,d.revision_id,d.timestamp,
                   d.contributor_json
            FROM passage AS p JOIN document AS d ON d.doc_id=p.doc_id
            WHERE p.passage_id=?
        """, (passage_id,)).fetchone()
        if row is None:
            raise BroadQaQueryError("broad QA posting 指向缺失 passage")
        passage_terms = _script_terms(row[4])
        title_terms = _script_terms(row[6])
        overlap = len(set(raw_terms).intersection(passage_terms))
        title_overlap = len(set(raw_terms).intersection(title_terms))
        exact_title_score = (
            20_000_000 if _title_span(question, row[6]) is not None else 0)
        sentence_surface = sentence_question
        title_span = _title_span(sentence_surface, row[6])
        if title_span is not None:
            sentence_surface = (
                sentence_surface[:title_span[0]] + "\n"
                + sentence_surface[title_span[1]:])
        sentence_terms = _script_terms(sentence_surface)
        sentence_overlap = _best_sentence_overlap(sentence_terms, row[4])
        best_sentence = _best_sentence(sentence_terms, row[4])
        score = (scores[passage_id] + exact_title_score
                 + title_overlap * 2_000_000 + overlap * 10_000
                 + sentence_overlap * 2_000_000
                 + _answer_kind_bonus(answer_kinds, best_sentence)
                 - _structural_residue_penalty(row[4]))
        candidate_rows.append((score, matched[passage_id], row))
    candidate_rows.sort(key=lambda item: (-item[0], -item[1], item[2][0]))
    best_score, best_match_count, best = candidate_rows[0]
    document_count = len({item[2][5] for item in candidate_rows})
    if best_match_count < 2 or best_score <= max(1, 1_000_000 // total_passages):
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, metadata["snapshot_id"], metadata["license_id"],
            len(rows), document_count,
        )
    candidate_terms = set(_script_terms(best[4]))
    candidate_terms.update(_script_terms(best[6]))
    if _missing_strong_constraint(
            question, title=best[6], candidate_terms=candidate_terms,
            slot_free_question=slot_free_question, candidate_text=best[4]):
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, metadata["snapshot_id"], metadata["license_id"],
            len(rows), document_count,
        )
    if _is_ambiguous_list(best[4]):
        return BroadQaResult(
            "CLARIFY", question, None, None, None, None, None, None, None,
            None, None, metadata["snapshot_id"], metadata["license_id"],
            len(rows), document_count,
        )
    if (_title_span(question, best[6]) is None
            and len(candidate_rows) > 1
            and candidate_rows[1][2][5] != best[5]
            and candidate_rows[1][0] * 100 >= best_score * 95):
        return BroadQaResult(
            "CLARIFY", question, None, None, None, None, None, None, None,
            None, None, metadata["snapshot_id"], metadata["license_id"],
            len(rows), document_count,
        )
    # 没有页面标题锚点时，关系词共现不足以证明回答对象，必须拒答。
    if _title_span(slot_free_question, best[6]) is None:
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, metadata["snapshot_id"], metadata["license_id"],
            len(rows), document_count,
        )
    sentence_question = slot_free_question
    span = _title_span(sentence_question, best[6])
    if span is not None:
        sentence_question = (
            sentence_question[:span[0]] + "\n" + sentence_question[span[1]:])
    question_terms = _script_terms(sentence_question)
    answer = _best_sentence(question_terms, best[4])
    source_url = (
        "https://zh.wikipedia.org/w/index.php?curid="
        f"{best[7]}&oldid={best[8]}"
    )
    return BroadQaResult(
        "ANSWER", question, answer, best[6], best[7], best[8], best[4],
        best[1], best[2], best[3], source_url, metadata["snapshot_id"],
        metadata["license_id"], len(rows), document_count,
        best[9], best[10], WIKIPEDIA_ATTRIBUTION,
    )


__all__ = ["BroadQaQueryError", "query_broad_qa"]
