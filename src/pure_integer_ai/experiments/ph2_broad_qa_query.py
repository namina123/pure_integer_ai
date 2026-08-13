"""在紧凑整数 postings 上执行有界检索和来源约束回答。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import re
import sqlite3

from opencc import OpenCC

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaEvidenceCitation,
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
_NON_REAL_ENTITY_RE = re.compile(
    r"(?:不存在|虚构|假想|杜撰|幻想|架空)(?:的|之)?")
_TO_SIMPLIFIED = OpenCC("t2s")
_TO_TRADITIONAL = OpenCC("s2t")
_MAX_EVIDENCE_CITATIONS = 4


# object-model: exception
class BroadQaQueryError(RuntimeError):
    """索引 schema、posting、查询预算或来源身份发生漂移。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaRetrievalCandidate:
    """记录检索排序中的一条来源绑定 passage 候选。"""

    score: int
    matched_term_count: int
    passage_id: int
    raw_start: int
    raw_end: int
    raw_sha256: str
    text: str
    doc_id: int
    title: str
    query_anchor_title: str
    page_id: int
    revision_id: int
    revision_timestamp: str
    contributor_json: str
    selected_text: str = ""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaRetrievalTrace:
    """记录一次有界候选查询的资源计数与来源身份。"""

    snapshot_id: str
    license_id: str
    matched_query_term_count: int
    posting_visit_count: int
    candidate_document_count: int
    total_passage_count: int
    evidence_candidates: tuple[BroadQaRetrievalCandidate, ...] = ()


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


def _sentence_shape_bonus(answer_kinds: tuple[str, ...], text: str) -> int:
    """以公开问式类型给证据句施加整数形态偏置。"""
    bonus = 0
    if "QUANTITY" in answer_kinds and _NUMBER_RE.search(text):
        bonus += 4
    if "TIME" in answer_kinds and _NUMBER_RE.search(text):
        bonus += 3
    if "CAUSE" in answer_kinds and re.search(r"因为|由于|因而|因此|为此", text):
        bonus += 3
    if "LOCATION" in answer_kinds and re.search(
            r"位于|位在|地处|来自|居住|發源|发源|首都|省|市|县|區|区", text):
        bonus += 2
    if "TYPE" in answer_kinds and re.search(r"是|屬於|属于|指|稱為|称为", text):
        bonus += 2
    if "MANNER" in answer_kinds and re.search(r"通過|通过|使用|采用|以|由", text):
        bonus += 2
    if "ENTITY" in answer_kinds and re.search(r"由|為|为|担任|任命|作者", text):
        bonus += 1
    return bonus


def _best_sentence(
        question_terms: set[str],
        text: str,
        answer_kinds: tuple[str, ...] = (),
        ) -> str:
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
            -_sentence_shape_bonus(answer_kinds, item[1]),
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


def _best_evidence_window(
        question_terms: set[str], text: str,
        answer_kinds: tuple[str, ...],
        term_weights: dict[str, int],
        ) -> tuple[int, str]:
    """在一个 passage 内选择最多三句的整数加权证据窗口。"""
    ranked = _rank_evidence_windows(
        question_terms, text, answer_kinds, term_weights)
    if not ranked:
        return 0, text[:360].strip()
    return ranked[0][0][0], ranked[0][1]


def _rank_evidence_windows(
        question_terms: set[str], text: str,
        answer_kinds: tuple[str, ...],
        term_weights: dict[str, int],
        ) -> tuple[tuple[tuple[int, int, int, int, int], str], ...]:
    """对 passage 内全部 1 至 3 句精确窗口做确定性整数排序。"""
    sentences = tuple(
        item for item in _SENTENCE_RE.finditer(text)
        if item.group(0).strip())
    if not sentences:
        return ()
    ranked = []
    for start in range(len(sentences)):
        for width in (1, 2, 3):
            if start + width > len(sentences):
                continue
            window = text[
                sentences[start].start():sentences[start + width - 1].end()
            ].strip()
            overlap = question_terms.intersection(_script_terms(window))
            rare_score = sum(term_weights.get(term, 1) for term in overlap)
            ranked.append(((
                rare_score, len(overlap),
                _sentence_shape_bonus(answer_kinds, window),
                -width, -start), window))
    return tuple(sorted(ranked, reverse=True))


def _select_diverse_evidence_windows(
        ranked_windows: list[tuple[
            tuple[int, int, int, int, int], int, tuple, str]],
        *, limit: int,
        ) -> tuple[tuple[
            tuple[int, int, int, int, int], int, tuple, str], ...]:
    """先跨 passage 扩散，再补非重叠窗口，最后才允许重叠。"""
    if type(limit) is not int or limit <= 0:
        raise BroadQaQueryError("broad QA evidence window limit 非法")
    selected = []
    selected_identities = set()
    selected_passages = set()
    selected_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)

    def adopt(item: tuple[
            tuple[int, int, int, int, int], int, tuple, str]) -> None:
        """登记一个已排序窗口及其 passage 内字符范围。"""
        _, _, page_row, selected_text = item
        passage_id = int(page_row[0])
        identity = (passage_id, selected_text)
        selected.append(item)
        selected_identities.add(identity)
        selected_passages.add(passage_id)
        start = page_row[4].find(selected_text)
        if start >= 0:
            selected_ranges[passage_id].append(
                (start, start + len(selected_text)))

    for item in ranked_windows:
        passage_id = int(item[2][0])
        if passage_id in selected_passages:
            continue
        adopt(item)
        if len(selected) == limit:
            return tuple(selected)
    for item in ranked_windows:
        passage_id = int(item[2][0])
        selected_text = item[3]
        identity = (passage_id, selected_text)
        if identity in selected_identities:
            continue
        start = item[2][4].find(selected_text)
        if start < 0:
            continue
        end = start + len(selected_text)
        if any(start < prior_end and prior_start < end
               for prior_start, prior_end in selected_ranges[passage_id]):
            continue
        adopt(item)
        if len(selected) == limit:
            return tuple(selected)
    for item in ranked_windows:
        passage_id = int(item[2][0])
        identity = (passage_id, item[3])
        if identity in selected_identities:
            continue
        adopt(item)
        if len(selected) == limit:
            break
    return tuple(selected)


def _cover_explicit_number_evidence_windows(
        ranked_windows: list[tuple[
            tuple[int, int, int, int, int], int, tuple, str]],
        selected_windows: tuple[tuple[
            tuple[int, int, int, int, int], int, tuple, str], ...],
        *, explicit_numbers: frozenset[str], limit: int,
        ) -> tuple[tuple[
            tuple[int, int, int, int, int], int, tuple, str], ...]:
    """用同页高排窗口精确补齐问题数字，不猜测近似或算术关系。"""
    if (type(limit) is not int or limit <= 0
            or not isinstance(explicit_numbers, frozenset)
            or any(not isinstance(item, str) or not item
                   for item in explicit_numbers)
            or len(selected_windows) > limit):
        raise BroadQaQueryError("broad QA explicit number coverage 输入非法")
    selected = list(selected_windows)

    def identity(item: tuple[
            tuple[int, int, int, int, int], int, tuple, str],
            ) -> tuple[int, str]:
        """返回窗口在一个 passage 内的稳定去重身份。"""
        return int(item[2][0]), item[3]

    def covered(values: list[tuple[
            tuple[int, int, int, int, int], int, tuple, str]],
            ) -> frozenset[str]:
        """返回当前证据逐字覆盖的问题显式数字。"""
        observed = set()
        for item in values:
            observed.update(_NUMBER_RE.findall(item[3]))
        return frozenset(observed.intersection(explicit_numbers))

    for number in sorted(explicit_numbers):
        current_coverage = covered(selected)
        if number in current_coverage:
            continue
        identities = {identity(item) for item in selected}
        option = next((
            item for item in ranked_windows
            if identity(item) not in identities
            and number in _NUMBER_RE.findall(item[3])
        ), None)
        if option is None:
            continue
        if len(selected) < limit:
            selected.append(option)
            continue
        for index in range(len(selected) - 1, -1, -1):
            proposal = [
                item for ordinal, item in enumerate(selected)
                if ordinal != index
            ]
            proposal.append(option)
            if current_coverage.issubset(covered(proposal)):
                selected = proposal
                break
    return tuple(selected)


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
    explicit_scope = re.search(
        r"(?:上的|中的|位于|位在|来自|来自于|所在|發源於|发源于|居住於|居住于)",
        prefix)
    if explicit_scope is None:
        prefix = ""
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


def _has_explicit_non_real_entity(
        question: str, *, slot_free_question: str,
        candidate_title: str,
        ) -> bool:
    """显式虚构/不存在限定未被页面标题锚定时，保持 UNKNOWN。"""
    if _title_span(slot_free_question, candidate_title) is not None:
        return False
    return _NON_REAL_ENTITY_RE.search(slot_free_question) is not None


def select_broad_qa_evidence_sentence(question: str, context: str) -> str:
    """从调用方给定的来源上下文中确定性选择一条完整证据句。"""
    if (not isinstance(question, str) or not question.strip()
            or not isinstance(context, str) or not context.strip()):
        raise BroadQaQueryError("broad QA question/context 不能为空")
    slots = load_broad_qa_question_slots()
    surface = slots.strip_slots(question)
    return _best_sentence(
        set(_script_terms(surface)), context, slots.answer_kinds(question))


def retrieve_broad_qa_candidates(
        connection: sqlite3.Connection,
        question: str,
        *,
        max_query_terms: int = 24,
        max_candidate_passages: int = 20,
        max_posting_visits: int = 500_000,
        ) -> tuple[tuple[BroadQaRetrievalCandidate, ...], BroadQaRetrievalTrace]:
    """返回有界 passage 排名与资源轨迹，不执行回答、澄清或拒答门。"""
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
        return (), BroadQaRetrievalTrace(
            metadata["snapshot_id"], metadata["license_id"], 0, 0, 0,
            int(metadata["passage_count"]))
    placeholders = ",".join("?" for _ in raw_terms)
    all_rows = tuple(connection.execute(
        "SELECT term,document_frequency,passage_deltas FROM posting "
        f"WHERE term IN ({placeholders})",
        raw_terms,
    ))
    all_rows = tuple(sorted(all_rows, key=lambda item: (item[1], item[0])))
    rows = all_rows[:max_query_terms]
    if not rows:
        return (), BroadQaRetrievalTrace(
            metadata["snapshot_id"], metadata["license_id"], 0, 0, 0,
            int(metadata["passage_count"]))
    scores: dict[int, int] = defaultdict(int)
    matched: dict[int, int] = defaultdict(int)
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
    # 标题/alias 是来源合同的一部分；精确出现在问题中时，补入对应页面，
    # 防止低频关系词截断 max_query_terms 后把正确页排除。
    alias_table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='alias'"
    ).fetchone() is not None
    alias_term_exists = connection.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name='alias_term'"
    ).fetchone() is not None
    anchor_matches: dict[int, str] = {}
    if alias_term_exists:
        alias_rows = tuple(connection.execute(
            "SELECT surface,doc_id,COUNT(*) AS matched_count "
            "FROM alias_term WHERE term IN (" + placeholders + ") "
            "GROUP BY surface,doc_id "
            "ORDER BY matched_count DESC,LENGTH(surface) DESC,surface,doc_id "
            "LIMIT ?",
            (*raw_terms, max_candidate_passages * 32),
        ))
        for surface, doc_id, _ in alias_rows:
            if _title_span(question, surface) is None:
                continue
            prior = anchor_matches.get(int(doc_id))
            if prior is None or (-len(surface), surface) < (-len(prior), prior):
                anchor_matches[int(doc_id)] = surface
    anchor_passage_ids = []
    for doc_id, _ in sorted(
            anchor_matches.items(), key=lambda item: (
                -len(item[1]), item[1], item[0])):
        passages = tuple(
            int(row[0]) for row in connection.execute(
                "SELECT passage_id FROM passage WHERE doc_id=? ", (doc_id,)))
        if passages:
            anchor_passage_ids.append(max(
                passages,
                key=lambda passage_id: (
                    scores.get(passage_id, 0),
                    matched.get(passage_id, 0), -passage_id)))
    regular_ids = tuple(
        item[0] for item in sorted(
            scores.items(),
            key=lambda item: (-item[1], -matched[item[0]], item[0])))
    ranked_ids = tuple(dict.fromkeys(
        (*anchor_passage_ids, *regular_ids)))[:max_candidate_passages]
    candidate_rows = []
    term_weights = {
        term: max(1, 1_000_000 // frequency)
        for term, frequency, _ in all_rows}
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
        aliases = tuple(
            item[0] for item in connection.execute(
                "SELECT surface FROM alias WHERE doc_id=? ORDER BY surface",
                (row[5],))
        ) if alias_table_exists else ()
        title_surfaces = (row[6],) + aliases
        matching_titles = tuple(
            title for title in title_surfaces
            if _title_span(question, title) is not None)
        query_anchor_title = (
            sorted(matching_titles, key=lambda item: (-len(item), item))[0]
            if matching_titles else row[6])
        passage_terms = _script_terms(row[4])
        title_terms = frozenset().union(*(
            _script_terms(title) for title in title_surfaces))
        overlap = len(set(raw_terms).intersection(passage_terms))
        title_overlap = len(set(raw_terms).intersection(title_terms))
        exact_title_score = (
            1_000_000_000 + len(query_anchor_title) * 1_000_000
            if matching_titles else 0)
        sentence_surface = slot_free_question
        title_span = _title_span(sentence_surface, query_anchor_title)
        if title_span is not None:
            sentence_surface = (
                sentence_surface[:title_span[0]] + "\n"
                + sentence_surface[title_span[1]:])
        sentence_terms = _script_terms(sentence_surface)
        sentence_overlap = _best_sentence_overlap(sentence_terms, row[4])
        best_sentence = _best_sentence(
            sentence_terms, row[4], answer_kinds)
        score = (scores[passage_id] + exact_title_score
                 + title_overlap * 2_000_000 + overlap * 10_000
                 + sentence_overlap * 2_000_000
                 + _answer_kind_bonus(answer_kinds, best_sentence)
                 - _structural_residue_penalty(row[4]))
        candidate_rows.append(BroadQaRetrievalCandidate(
            score, matched[passage_id], row[0], row[1], row[2], row[3],
            row[4], row[5], row[6], query_anchor_title, row[7], row[8],
            row[9], row[10]))
    candidate_rows.sort(key=lambda item: (
        -item.score, -item.matched_term_count, item.passage_id))
    evidence_candidates: tuple[BroadQaRetrievalCandidate, ...] = ()
    if candidate_rows:
        # 页面已由标题/alias 锚定后，仅在该页面的有限段落内重选证据，
        # 避免把“找对页、选错段”误判为全库检索失败。
        page_candidate = candidate_rows[0]
        page_rows = tuple(connection.execute("""
            SELECT p.passage_id,p.raw_start,p.raw_end,p.raw_sha256,p.text,
                   d.doc_id,d.title,d.page_id,d.revision_id,d.timestamp,
                   d.contributor_json
            FROM passage AS p JOIN document AS d ON d.doc_id=p.doc_id
            WHERE d.doc_id=? ORDER BY p.passage_id
        """, (page_candidate.doc_id,)))
        page_terms = _script_terms(slot_free_question)
        anchor_span = _title_span(slot_free_question,
                                  page_candidate.query_anchor_title)
        if anchor_span is not None:
            surface = (slot_free_question[:anchor_span[0]] + "\n"
                       + slot_free_question[anchor_span[1]:])
            page_terms = _script_terms(surface)
        ranked_windows = []
        for page_row in page_rows:
            for window_score, selected_text in _rank_evidence_windows(
                    set(page_terms), page_row[4], answer_kinds, term_weights):
                ranked_windows.append((
                    window_score, -int(page_row[0]), page_row, selected_text))
        ranked_windows.sort(reverse=True)
        page_candidates = []
        selected_windows = _select_diverse_evidence_windows(
            ranked_windows, limit=_MAX_EVIDENCE_CITATIONS)
        selected_windows = _cover_explicit_number_evidence_windows(
            ranked_windows, selected_windows,
            explicit_numbers=frozenset(
                _NUMBER_RE.findall(slot_free_question)),
            limit=_MAX_EVIDENCE_CITATIONS)
        for window_score, _, page_row, selected_text in selected_windows:
            page_candidates.append(BroadQaRetrievalCandidate(
                page_candidate.score + window_score[0],
                max(page_candidate.matched_term_count,
                    window_score[1]),
                page_row[0], page_row[1], page_row[2], page_row[3], page_row[4],
                page_row[5], page_row[6], page_candidate.query_anchor_title,
                page_row[7], page_row[8], page_row[9], page_row[10],
                selected_text))
        evidence_candidates = tuple(page_candidates)
        page_candidate = page_candidates[0]
        candidate_rows[0] = page_candidate
    document_count = len({item.doc_id for item in candidate_rows})
    return tuple(candidate_rows), BroadQaRetrievalTrace(
        metadata["snapshot_id"], metadata["license_id"], len(rows),
        posting_visits, document_count, int(metadata["passage_count"]),
        evidence_candidates)


def answer_broad_qa_candidates(
        question: str,
        candidates: tuple[BroadQaRetrievalCandidate, ...],
        trace: BroadQaRetrievalTrace,
        ) -> BroadQaResult:
    """对一次已完成的候选检索执行回答、澄清和拒答门。"""
    if (not isinstance(question, str) or not question.strip()
            or not isinstance(candidates, tuple)
            or any(not isinstance(item, BroadQaRetrievalCandidate)
                   for item in candidates)
            or not isinstance(trace, BroadQaRetrievalTrace)):
        raise BroadQaQueryError("broad QA candidate resolution 输入非法")
    question_slots = load_broad_qa_question_slots()
    slot_free_question = question_slots.strip_slots(question)
    if not candidates:
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, trace.snapshot_id, trace.license_id,
            trace.matched_query_term_count, 0,
        )
    best = candidates[0]
    exact_title_anchor = (
        _title_span(slot_free_question, best.query_anchor_title) is not None)
    if ((best.matched_term_count < 2 and not exact_title_anchor)
            or best.score <= max(1, 1_000_000 // trace.total_passage_count)):
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, trace.snapshot_id, trace.license_id,
            trace.matched_query_term_count, trace.candidate_document_count,
        )
    evidence_candidates = trace.evidence_candidates or (best,)
    candidate_text = "\n".join(
        item.selected_text or item.text for item in evidence_candidates)
    candidate_terms = set(_script_terms(candidate_text))
    candidate_terms.update(_script_terms(best.title))
    candidate_terms.update(_script_terms(best.query_anchor_title))
    if _missing_strong_constraint(
            question, title=best.query_anchor_title,
            candidate_terms=candidate_terms,
            slot_free_question=slot_free_question,
            candidate_text=candidate_text):
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, trace.snapshot_id, trace.license_id,
            trace.matched_query_term_count, trace.candidate_document_count,
        )
    if _has_explicit_non_real_entity(
            question, slot_free_question=slot_free_question,
            candidate_title=best.query_anchor_title):
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, trace.snapshot_id, trace.license_id,
            trace.matched_query_term_count, trace.candidate_document_count,
        )
    if _is_ambiguous_list(best.text) and not exact_title_anchor:
        return BroadQaResult(
            "CLARIFY", question, None, None, None, None, None, None, None,
            None, None, trace.snapshot_id, trace.license_id,
            trace.matched_query_term_count, trace.candidate_document_count,
        )
    if (_title_span(question, best.query_anchor_title) is None
            and len(candidates) > 1
            and candidates[1].doc_id != best.doc_id
            and candidates[1].score * 100 >= best.score * 95):
        return BroadQaResult(
            "CLARIFY", question, None, None, None, None, None, None, None,
            None, None, trace.snapshot_id, trace.license_id,
            trace.matched_query_term_count, trace.candidate_document_count,
        )
    # 没有页面标题锚点时，关系词共现不足以证明回答对象，必须拒答。
    if _title_span(slot_free_question, best.query_anchor_title) is None:
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, trace.snapshot_id, trace.license_id,
            trace.matched_query_term_count, trace.candidate_document_count,
        )
    sentence_question = slot_free_question
    span = _title_span(sentence_question, best.query_anchor_title)
    if span is not None:
        sentence_question = (
            sentence_question[:span[0]] + "\n" + sentence_question[span[1]:])
    question_terms = _script_terms(sentence_question)
    answer_kinds = question_slots.answer_kinds(question)
    citations = []
    answer_parts = []
    for evidence in evidence_candidates[:_MAX_EVIDENCE_CITATIONS]:
        window = evidence.selected_text
        if not window:
            _, window = _best_evidence_window(
                question_terms, evidence.text, answer_kinds, {})
        answer_parts.append(window)
        source_url = (
            "https://zh.wikipedia.org/w/index.php?curid="
            f"{evidence.page_id}&oldid={evidence.revision_id}")
        citations.append(BroadQaEvidenceCitation(
            evidence.title, evidence.page_id, evidence.revision_id,
            evidence.text, evidence.raw_start, evidence.raw_end,
            evidence.raw_sha256, source_url, trace.snapshot_id,
            trace.license_id, evidence.revision_timestamp,
            evidence.contributor_json, WIKIPEDIA_ATTRIBUTION, window))
    answer = "\n".join(answer_parts)
    source_url = (
        "https://zh.wikipedia.org/w/index.php?curid="
        f"{best.page_id}&oldid={best.revision_id}"
    )
    return BroadQaResult(
        "ANSWER", question, answer, best.title, best.page_id,
        best.revision_id, best.text, best.raw_start, best.raw_end,
        best.raw_sha256, source_url, trace.snapshot_id, trace.license_id,
        trace.matched_query_term_count, trace.candidate_document_count,
        best.revision_timestamp, best.contributor_json,
        WIKIPEDIA_ATTRIBUTION, tuple(citations),
    )


def query_broad_qa(
        connection: sqlite3.Connection,
        question: str,
        *,
        max_query_terms: int = 24,
        max_candidate_passages: int = 20,
        max_posting_visits: int = 500_000,
        ) -> BroadQaResult:
    """以稀有 term postings 缩小候选，并从 top-K 页面投影可引用回答。"""
    candidates, trace = retrieve_broad_qa_candidates(
        connection, question,
        max_query_terms=max_query_terms,
        max_candidate_passages=max_candidate_passages,
        max_posting_visits=max_posting_visits)
    return answer_broad_qa_candidates(question, candidates, trace)


__all__ = [
    "answer_broad_qa_candidates",
    "BroadQaRetrievalCandidate",
    "BroadQaRetrievalTrace",
    "BroadQaQueryError",
    "query_broad_qa",
    "retrieve_broad_qa_candidates",
    "select_broad_qa_evidence_sentence",
]
