"""在紧凑整数 postings 上执行有界检索和来源约束回答。"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import re
import sqlite3
from typing import Callable, Iterable

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaEvidenceCitation,
    BroadQaResult,
    WIKIPEDIA_ATTRIBUTION,
)
from pure_integer_ai.experiments.ph2_broad_qa_index import broad_qa_terms
from pure_integer_ai.experiments.ph2_broad_qa_question_slots import (
    load_broad_qa_question_slots,
)
from pure_integer_ai.experiments.ph2_broad_qa_obligation_learning import (
    LearnedTypedObligation,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    LearnedRelationEvidenceModel,
)


# 语言变体不是查询算法的内置知识。调用方可注入图读取器，返回同一
# 表面的其他已证实表示；没有读取器时只使用原始表面，保持 fail-closed。
SurfaceVariantProvider = Callable[[str], Iterable[str]]
_MAX_SURFACE_VARIANTS = 64


_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?|[^。！？!?\n]+$")
_CJK_SEQUENCE_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_NUMBER_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)?")
_NON_REAL_ENTITY_RE = re.compile(
    r"(?:不存在|虚构|假想|杜撰|幻想|架空)(?:的|之)?")
_EXPLICIT_SCOPE_RE = re.compile(
    r"(?:上的|中的|位于|位在|来自|来自于|所在|"
    r"發源於|发源于|居住於|居住于)")
_CAUSE_EVIDENCE_RE = re.compile(
    r"因为|由於|由于|因而|因此|為此|为此|導致|导致|使得|"
    r"(?:^|[。！？!?])\s*因|"
    r"(?:使|令到?)[^。！？!?\n]{0,80}(?:成為|成为|變成|变成)")
_BACKWARD_REFERENCE_RE = re.compile(
    r"^(?:前者|後者|后者|上述|以上)")
_EVENT_REFERENCE_RE = re.compile(
    r"^(?:此(?:舉|举|事|行為|行为)|"
    r"(?:這|这)(?:一(?:舉|举|事|行為|行为|做法)|"
    r"導致|导致|使得|令到?))")
_MAX_EVIDENCE_CITATIONS = 4


def _surface_variants(
        text: str,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> tuple[str, ...]:
    """返回原始表面与调用方从图中提供的候选表示。"""
    if not isinstance(text, str):
        raise TypeError("surface 必须是字符串")
    values = [text]
    if surface_variant_provider is None:
        return (text,)
    if not callable(surface_variant_provider):
        raise TypeError("surface_variant_provider 必须是可调用对象")
    variants = surface_variant_provider(text)
    if isinstance(variants, str):
        raise TypeError("surface_variant_provider 不得直接返回字符串")
    try:
        for variant in variants:
            if not isinstance(variant, str) or not variant:
                raise TypeError("surface variant 必须是非空字符串")
            if variant not in values:
                values.append(variant)
            if len(values) > _MAX_SURFACE_VARIANTS:
                raise ValueError("surface variant 数量超过预算")
    except TypeError:
        raise
    except ValueError:
        raise
    except Exception as error:
        raise TypeError("surface_variant_provider 返回值不可迭代") from error
    return tuple(values)

# 长问句常在真正问题前增加“根据资料……回答”或“从时间线看”这类
# 回答侧指令。它们是 discourse framing，不是页面实体或答案属性；只在
# 行首、遇到明确分隔标点时剥离，避免吞掉实体内部的限定。
_QUERY_INSTRUCTION_PREFIX_PATTERNS = (
    re.compile(
        r"^\s*(?:\u8bf7(?:\u4f60)?\s*)?(?:\u53ea\s*)?"
        r"(?:\u4f9d\u636e|\u6839\u636e|\u57fa\u4e8e|\u6309\u7167)\s*"
        r"[^\uFF0C,\u3002\uFF01!\uFF1F?\uFF1A:]{1,160}"
        r"(?:\u56de\u7b54|\u8bf4\u660e|\u544a\u8bc9\u6211|\u4f5c\u7b54)?"
        r"\s*[\uFF0C,\uFF1A:]\s*"),
    re.compile(
        r"^\s*\u4ece[^\uFF0C,\u3002\uFF01!\uFF1F?\uFF1A:]{1,160}"
        r"(?:\u770b|\u6765\u770b|\u800c\u8a00)\s*[\uFF0C,\uFF1A:]\s*"),
)


# object-model: exception
class BroadQaQueryError(RuntimeError):
    """索引 schema、posting、查询预算或来源身份发生漂移。"""


def has_explicit_non_real_constraint(question: str) -> bool:
    """返回问题是否携带明确的不存在/虚构实体限定。"""
    if not isinstance(question, str):
        raise TypeError("broad QA question 类型错误")
    return _NON_REAL_ENTITY_RE.search(question) is not None


# object-model: runtime cache; representation=bounded map; not persisted
class BroadQaQueryCache:
    """绑定单个只读 SQLite 连接的有界问答结果缓存。

    缓存只接受无 learned overlay 的静态 broad index 查询；调用方显式传入
    实例即可启用，默认 query API 保持每次完整检索，避免可变数据库被误缓存。
    """

    __slots__ = (
        "_connection", "_entries", "_limit", "_metadata_values",
        "_schema_tables",
    )

    def __init__(self, connection: sqlite3.Connection, *, limit: int = 128):
        if not isinstance(connection, sqlite3.Connection):
            raise TypeError("BroadQaQueryCache connection 类型错误")
        if type(limit) is not int or not 1 <= limit <= 4096:
            raise ValueError("BroadQaQueryCache limit 非法")
        self._connection = connection
        self._entries: OrderedDict[tuple[object, ...], BroadQaResult] = (
            OrderedDict())
        self._limit = limit
        self._metadata_values: dict[str, str] | None = None
        self._schema_tables: frozenset[str] | None = None

    def get(self, key: tuple[object, ...]) -> BroadQaResult | None:
        if not isinstance(key, tuple):
            raise TypeError("BroadQaQueryCache key 必须是 tuple")
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key: tuple[object, ...], value: BroadQaResult) -> None:
        if not isinstance(key, tuple) or not isinstance(value, BroadQaResult):
            raise TypeError("BroadQaQueryCache entry 类型错误")
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._limit:
            self._entries.popitem(last=False)

    def owns(self, connection: sqlite3.Connection) -> bool:
        """拒绝将一个连接的结果误用于另一数据库。"""
        return connection is self._connection

    def prepare(self) -> None:
        """在请求计时前读取只读 release 的元数据与可选 schema。"""
        if self._metadata_values is not None:
            return
        metadata_values = _metadata(self._connection)
        schema_tables = frozenset(
            str(row[0]) for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('alias','alias_term')")
        )
        self._metadata_values = metadata_values
        self._schema_tables = schema_tables

    def _retrieval_context(
            self,
            ) -> tuple[dict[str, str], frozenset[str]]:
        """返回当前连接已经核验的静态检索上下文。"""
        self.prepare()
        if self._metadata_values is None or self._schema_tables is None:
            raise BroadQaQueryError("broad QA query cache 未完成准备")
        return self._metadata_values, self._schema_tables


def _strip_query_instruction_prefix(surface: str) -> str:
    """剥离行首回答指令，保留问题实体、属性和显式限定。"""
    if not isinstance(surface, str):
        raise TypeError("broad QA query surface 类型错误")
    current = surface
    for pattern in _QUERY_INSTRUCTION_PREFIX_PATTERNS:
        candidate = pattern.sub("", current, count=1)
        if candidate != current:
            return candidate
    return current


def _query_surface(
        question: str, slots,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> str:
    """统一生成检索、标题锚定和回答门使用的 discourse-free surface。"""
    return _strip_query_instruction_prefix(
        slots.strip_slots(question, surface_variant_provider))


def _answer_kinds(
        question: str, slots,
        learned_typed_obligation: LearnedTypedObligation | None,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> tuple[str, ...]:
    """合并静态问式与公开课程 learned obligation；冲突时保持静态结果。"""
    static = slots.answer_kinds(question, surface_variant_provider)
    if learned_typed_obligation is None:
        return static
    if not isinstance(learned_typed_obligation, LearnedTypedObligation):
        raise BroadQaQueryError("learned typed obligation 类型错误")
    learned = learned_typed_obligation.answer_kinds(question)
    if static:
        # 已有公开问式合同优先；learned obligation 不能把“哪一年”这类
        # 已解析时间问式叠加成 ENTITY，避免训练侧证据排序污染可靠路径。
        return static
    return learned


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


def _read_unsigned_canonical(data: bytes, cursor: int) -> tuple[int, int]:
    """读取一个规范 unsigned varint，避免通用 codec 的往返重编码。"""
    start = cursor
    value = 0
    shift = 0
    while True:
        if cursor >= len(data):
            raise BroadQaQueryError("broad QA posting varint 被截断")
        byte = data[cursor]
        cursor += 1
        value |= (byte & 127) << shift
        if byte < 128:
            break
        shift += 7
    encoded_size = cursor - start
    minimum_size = max(1, (value.bit_length() + 6) // 7)
    if encoded_size != minimum_size:
        raise BroadQaQueryError("broad QA posting varint 非规范")
    return value, cursor


def _decode_postings(payload: bytes) -> tuple[int, ...]:
    """把正 delta varint 流恢复为严格递增 passage id。

    posting payload 已由整数 codec 约束为正 delta；这里保留计数、最短
    varint、尾随字节和正值校验，但直接解析 zigzag 数值，避免
    ``decode_integer_tuple`` 为每个 posting 再分配 tuple 并完整重编码。
    """
    if not isinstance(payload, bytes) or not payload:
        raise BroadQaQueryError("broad QA posting payload 非空 bytes")
    size, cursor = _read_unsigned_canonical(payload, 0)
    result = []
    current = 0
    for _ in range(size):
        unsigned, cursor = _read_unsigned_canonical(payload, cursor)
        # 正 delta 的 zigzag 编码只能是偶数；奇数表示负整数。
        if unsigned & 1:
            raise BroadQaQueryError("broad QA posting delta 非正")
        value = unsigned // 2
        if value <= 0:
            raise BroadQaQueryError("broad QA posting delta 非正")
        current += value
        result.append(current)
    if cursor != len(payload):
        raise BroadQaQueryError("broad QA posting payload 存在尾随字节")
    return tuple(result)


@lru_cache(maxsize=2048)
def _restore_small_postings(payload: bytes) -> tuple[int, ...]:
    """缓存小 posting 的规范解码结果，限制长会话常驻工作集。"""
    return _decode_postings(payload)


def _restore_postings(payload: bytes) -> tuple[int, ...]:
    """恢复 posting；大 payload 直接解码，避免按条数缓存大整数 tuple。"""
    if isinstance(payload, bytes) and len(payload) <= 512:
        return _restore_small_postings(payload)
    return _decode_postings(payload)


@lru_cache(maxsize=16384)
def _raw_terms(text: str) -> frozenset[str]:
    """缓存单一原始表面的离散特征。"""
    return frozenset(broad_qa_terms(text))


def _script_terms(
        text: str,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> frozenset[str]:
    """生成原始表面及图解析器提供的候选表示的离散特征并集。"""
    if surface_variant_provider is None:
        return _raw_terms(text)
    values = set()
    for surface in _surface_variants(text, surface_variant_provider):
        values.update(broad_qa_terms(surface))
    return frozenset(values)


@lru_cache(maxsize=65536)
def _cached_script_term_overlap(
        question_terms: frozenset[str], surfaces: tuple[str, ...]
        ) -> frozenset[str]:
    """缓存一次问题词项集合与候选句的交集扫描。"""
    matched: set[str] = set()

    def scan(surface: str) -> None:
        for sequence_match in _CJK_SEQUENCE_RE.finditer(surface):
            sequence = sequence_match.group(0)
            if len(sequence) == 1:
                term = "c:" + sequence
                if term in question_terms:
                    matched.add(term)
            for width in (2, 3):
                for index in range(max(0, len(sequence) - width + 1)):
                    term = "c:" + sequence[index:index + width]
                    if term in question_terms:
                        matched.add(term)
        for word_match in _WORD_RE.finditer(surface):
            term = "w:" + word_match.group(0).casefold()
            if term in question_terms:
                matched.add(term)

    for surface in surfaces:
        scan(surface)
    return frozenset(matched)


def _script_term_overlap(
        question_terms: set[str] | frozenset[str], text: str,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> int:
    """只扫描与问题相关的词项，返回与规范脚本并集的交集大小。

    ``_script_terms`` 需要为整段候选文本构造全部二/三元组集合；回答侧
    实际只需要它与当前问题的交集。这里复用同一 CJK/ASCII 窗口规则，
    但直接把命中的词项写入小集合，避免候选句上的大集合分配。
    """
    if not isinstance(question_terms, (set, frozenset)) \
            or not isinstance(text, str):
        raise TypeError("script term overlap 输入类型错误")
    surfaces = _surface_variants(text, surface_variant_provider)
    if surface_variant_provider is None:
        surfaces = (text,)
    return len(_cached_script_term_overlap(
        frozenset(question_terms), surfaces))


def _sentence_shape_bonus(answer_kinds: tuple[str, ...], text: str) -> int:
    """以公开问式类型给证据句施加整数形态偏置。"""
    bonus = 0
    if "QUANTITY" in answer_kinds and _NUMBER_RE.search(text):
        bonus += 4
    if "TIME" in answer_kinds and _NUMBER_RE.search(text):
        bonus += 3
    if "TIME" in answer_kinds and re.search(
            r"建成通车|启用|开通|成立于|出生|逝世|发生于|举行于|开工建设|工程开工|完工|发布",
            text):
        # 时间问式优先选择明确事件谓词，避免把同页奖项/分类年份当作答案。
        bonus += 12
    if "CAUSE" in answer_kinds and re.search(r"因为|由于|因而|因此|为此", text):
        bonus += 3
    if "CAUSE" in answer_kinds and re.search(
            r"称为|稱為|称作|稱作|又称|又稱|得名|命名", text):
        bonus += 10
    if "LOCATION" in answer_kinds and re.search(
            r"位于|位在|地处|来自|居住|發源|发源|首都|省|市|县|區|区", text):
        bonus += 2
    if "TYPE" in answer_kinds and re.search(r"是|屬於|属于|指|稱為|称为", text):
        bonus += 2
    if "MANNER" in answer_kinds and re.search(r"通過|通过|使用|采用|以|由", text):
        bonus += 2
    if "ENTITY" in answer_kinds and re.search(r"由|為|为|担任|任命|作者", text):
        bonus += 1
    if "ENTITY" in answer_kinds and re.search(
            r"作者|作家|撰写|撰寫|所作|創作|创作|最后一位|最後一位|最多次|获得|獲得",
            text):
        bonus += 12
    if "ENTITY" in answer_kinds and re.search(
            r"又称|又稱|俗名|别名|別名|称为|稱為|命名|简称|簡稱",
            text):
        bonus += 12
    if "ENTITY" in answer_kinds and re.search(
            r"原作|原著|入圍次數最多|入围次数最多|提名次數最多|提名次数最多",
            text):
        bonus += 14
    if "TYPE" in answer_kinds and re.search(
            r"又称|又稱|俗名|别名|別名|称为|稱為|命名|简称|簡稱",
            text):
        bonus += 12
    return bonus


def _sentence_priority_bonus(answer_kinds: tuple[str, ...], text: str) -> int:
    """为明确事件谓词提供高位整数优先级，避免同页年份噪声抢答。"""
    if "TIME" in answer_kinds and re.search(
            r"\|\s*(?:date|日期|时间|時間)\s*=", text):
        # 事件页的 infobox 日期是结构化时间证据；仅在时间问式中恢复，
        # 其他问式仍对结构残片保持降权。
        return 320_000_000
    if "QUANTITY" in answer_kinds and _NUMBER_RE.search(text):
        # Quantity questions must prefer an explicit amount over a nearby
        # descriptive sentence that merely repeats the entity name.
        return 130_000_000
    if "TIME" in answer_kinds and re.search(
            r"每年|每月|每周|周末|通常在|通常于|季度|日期|日举行|月举行",
            text):
        # Recurring schedules are time evidence even without an event verb.
        return 130_000_000
    if "TIME" in answer_kinds and re.search(
            r"建成通车|启用|开通|成立于|出生|逝世|发生于|举行于|开工建设|工程开工|完工|发布",
            text):
        return 100_000_000
    if "TIME" in answer_kinds and re.search(
            r"追溯|来源|建立于|改名|更改|改为|改成|现名|現名|博士|学位|學位|毕业|畢業",
            text):
        return 90_000_000
    if "QUANTITY" in answer_kinds and re.search(
            r"公顷|公頃|平方公里|平方千米|面积|面積|占地|佔地|比例|百分之|％|%",
            text):
        return 95_000_000
    if "QUANTITY" in answer_kinds and re.search(
            r"长达|長達|高达|高達|身长|身長|体重|公里|千米|厘米|公分|米",
            text):
        return 75_000_000
    if "MANNER" in answer_kinds and re.search(
            r"按下它|转换到大写模式|英文字母都预设为大写|作用是",
            text):
        return 120_000_000
    if "MANNER" in answer_kinds and re.search(
            r"按下|切换|转换到|用于|用来|操作|采用|通过",
            text):
        return 90_000_000
    return 0


def _best_sentence(
        question_terms: set[str],
        text: str,
        answer_kinds: tuple[str, ...] = (),
        surface_variant_provider: SurfaceVariantProvider | None = None,
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
            -_script_term_overlap(
                question_terms, item[1], surface_variant_provider),
            -_sentence_shape_bonus(answer_kinds, item[1]),
            item[0],
            len(item[1]),
            item,
        ),
    )
    if _script_term_overlap(
            question_terms, ranked[0][1], surface_variant_provider) == 0:
        adjacent = "".join(item[1] for item in candidates[:2])
        return adjacent if len(adjacent) <= 360 else adjacent[:360].rstrip()
    answer = ranked[0][1]
    return answer if len(answer) <= 360 else answer[:360].rstrip()


def _best_sentence_overlap(
        question_terms: set[str], text: str,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> int:
    """返回段内单句对关系/属性问题特征的最大覆盖数。"""
    return max((
        _script_term_overlap(
            question_terms, item.group(0), surface_variant_provider)
        for item in _SENTENCE_RE.finditer(text)
        if item.group(0).strip()
    ), default=0)


def _best_evidence_window(
        question_terms: set[str], text: str,
        answer_kinds: tuple[str, ...],
        term_weights: dict[str, int],
        relation_evidence_model: LearnedRelationEvidenceModel | None = None,
        relation_question: str | None = None,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> tuple[int, str]:
    """在一个 passage 内选择最多三句的整数加权证据窗口。"""
    ranked = _rank_evidence_windows(
        question_terms, text, answer_kinds, term_weights,
        relation_evidence_model=relation_evidence_model,
        relation_question=relation_question,
        surface_variant_provider=surface_variant_provider)
    if not ranked:
        return 0, text[:360].strip()
    return ranked[0][0][0], ranked[0][1]


def _rank_evidence_windows(
        question_terms: set[str], text: str,
        answer_kinds: tuple[str, ...],
        term_weights: dict[str, int],
        *,
        relation_evidence_model: LearnedRelationEvidenceModel | None = None,
        relation_question: str | None = None,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> tuple[tuple[tuple[int, int, int, int, int], str], ...]:
    """对 passage 内全部 1 至 3 句精确窗口做确定性整数排序。"""
    sentences = tuple(
        item for item in _SENTENCE_RE.finditer(text)
        if item.group(0).strip())
    if not sentences:
        return ()
    if relation_evidence_model is not None and not isinstance(
            relation_evidence_model, LearnedRelationEvidenceModel):
        raise BroadQaQueryError("relation evidence model 类型错误")
    if relation_evidence_model is not None and (
            not isinstance(relation_question, str) or not relation_question.strip()):
        raise BroadQaQueryError("relation evidence question 缺失")
    ranked = []
    for start in range(len(sentences)):
        for width in (1, 2, 3):
            if start + width > len(sentences):
                continue
            window = text[
                sentences[start].start():sentences[start + width - 1].end()
            ].strip()
            overlap_terms = _cached_script_term_overlap(
                frozenset(question_terms), _surface_variants(
                    window, surface_variant_provider))
            rare_score = sum(
                term_weights.get(term, 1) for term in overlap_terms)
            # 形态优先级只能放大已经覆盖问题特征的窗口；否则一个
            # “公里/年份”邻句不能凭自身单位把无关证据抬到首位。
            if overlap_terms:
                rare_score += _sentence_priority_bonus(answer_kinds, window)
            if relation_evidence_model is not None:
                rare_score += relation_evidence_model.evidence_bonus(
                    relation_question, window)
            # Category/table carriers may contain every title and answer-kind
            # token while carrying no readable proposition. Keep them citable,
            # but let a complete sentence win when both windows cover the same
            # query terms.
            rare_score -= _structural_residue_penalty(window)
            ranked.append(((
                rare_score, len(overlap_terms),
                _sentence_shape_bonus(answer_kinds, window),
                -width, -start), window))
    return tuple(sorted(ranked, reverse=True))


def _validate_learned_term_weights(
        values: Iterable[tuple[str, int]] | None,
        ) -> dict[str, int]:
    """校验训练模型的整数投影；不接受任意运行时字符串规则。"""
    if values is None:
        return {}
    try:
        entries = tuple(values)
    except TypeError as error:
        raise BroadQaQueryError("learned evidence weights 不可迭代") from error
    result = {}
    for item in entries:
        if (not isinstance(item, tuple) or len(item) != 2
                or not isinstance(item[0], str) or not item[0]
                or type(item[1]) is not int or not 0 < item[1] <= 5_000_000):
            raise BroadQaQueryError("learned evidence weights 非法")
        if item[0] in result:
            raise BroadQaQueryError("learned evidence weights 重复")
        result[item[0]] = item[1]
    return result


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


def _expand_structural_evidence_windows(
        ranked_windows: list[tuple[
            tuple[int, int, int, int, int], int, tuple, str]],
        selected_windows: tuple[tuple[
            tuple[int, int, int, int, int], int, tuple, str], ...],
        *, answer_kinds: tuple[str, ...],
        ) -> tuple[tuple[
            tuple[int, int, int, int, int], int, tuple, str], ...]:
    """用同 passage 连续窗口补齐显式因果或事件回指所需邻句。"""
    if (not isinstance(answer_kinds, tuple)
            or any(not isinstance(item, str) or not item
                   for item in answer_kinds)):
        raise BroadQaQueryError("broad QA structural expansion 输入非法")
    selected = list(selected_windows)
    for index, current in enumerate(selected):
        passage_id = int(current[2][0])
        current_text = current[3]
        other_identities = {
            (int(item[2][0]), item[3])
            for ordinal, item in enumerate(selected)
            if ordinal != index
        }
        needs_cause = (
            "CAUSE" in answer_kinds
            and _CAUSE_EVIDENCE_RE.search(current_text) is None)
        needs_reference = _BACKWARD_REFERENCE_RE.search(
            current_text.lstrip()) is not None
        needs_event_reference = (
            "CAUSE" in answer_kinds
            and _EVENT_REFERENCE_RE.search(current_text.lstrip()) is not None)
        if not needs_cause and not needs_reference and not needs_event_reference:
            continue
        current_start = current[2][4].find(current_text)
        if current_start < 0:
            continue
        current_end = current_start + len(current_text)
        option = next((
            item for item in ranked_windows
            if int(item[2][0]) == passage_id
            and (int(item[2][0]), item[3]) not in other_identities
            and item[2][4].find(item[3]) >= 0
            and item[2][4].find(item[3]) <= current_start
            and (item[2][4].find(item[3]) + len(item[3])) >= current_end
            and item[3] != current_text
            and (not needs_cause
                 or _CAUSE_EVIDENCE_RE.search(item[3]) is not None)
            and (not (needs_reference or needs_event_reference)
                 or item[2][4].find(item[3]) < current_start)
        ), None)
        if option is not None:
            selected[index] = option
    return tuple(selected)


def _remove_redundant_evidence_candidates(
        candidates: tuple[BroadQaRetrievalCandidate, ...],
        ) -> tuple[BroadQaRetrievalCandidate, ...]:
    """删除完全重复或被另一条引用逐字覆盖的证据窗口。"""
    if any(not isinstance(item, BroadQaRetrievalCandidate)
           for item in candidates):
        raise BroadQaQueryError("broad QA evidence compression 输入非法")
    retained = []
    for ordinal, candidate in enumerate(candidates):
        text = candidate.selected_text or candidate.text
        redundant = any(
            candidate.page_id == other.page_id
            and candidate.revision_id == other.revision_id
            and (text == other_text and other_ordinal < ordinal
                 or (len(other_text) > len(text) and text in other_text))
            for other_ordinal, other in enumerate(candidates)
            for other_text in (other.selected_text or other.text,)
        )
        if not redundant:
            retained.append(candidate)
    return tuple(retained)


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
    if (stripped.startswith((
            "Category:", "category:", "{|", "{{", "===", "====",
            "# ", "* ", "File:", "Image:"))
            or re.match(r"^\|[^。！？!?\n]{0,80}=", stripped)):
        return 250_000_000
    return 0


def _title_span(
        question: str, title: str,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> tuple[int, int] | None:
    """在原文中定位页面标题及其图证实的表面变体。

    只让 provider 变换 ``title``，因此返回的 span 始终属于调用方的
    原始问题坐标；这使得后续移除标题、保留限定时不依赖变体长度相等。
    """
    for expected in _surface_variants(title, surface_variant_provider):
        start = question.find(expected)
        if start >= 0:
            return start, start + len(expected)
    return None


def _title_spans(
        question: str, title: str,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> tuple[tuple[int, int, int], ...]:
    """枚举标题变体在原始问题中的全部确定性位置。"""
    spans = set()
    for variant, expected in enumerate(
            _surface_variants(title, surface_variant_provider)):
        if not expected:
            continue
        start = question.find(expected)
        while start >= 0:
            spans.add((variant, start, start + len(expected)))
            start = question.find(expected, start + 1)
    return tuple(sorted(spans))


def _dominated_title_surfaces(
        question: str, surfaces: frozenset[str],
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> frozenset[str]:
    """找出仅作为更长候选标题子串出现的短标题 surface。"""
    spans_by_surface = {
        surface: _title_spans(
            question, surface, surface_variant_provider)
        for surface in surfaces
    }
    dominated = set()
    for short_surface, short_spans in spans_by_surface.items():
        if not short_spans:
            continue
        if all(any(
                long_variant == short_variant
                and long_start <= short_start
                and short_end <= long_end
                and long_end - long_start > short_end - short_start
                for long_surface, long_spans in spans_by_surface.items()
                if long_surface != short_surface
                for long_variant, long_start, long_end in long_spans)
                for short_variant, short_start, short_end in short_spans):
            dominated.add(short_surface)
    return frozenset(dominated)


def _page_evidence_surface(
        slot_free_question: str, anchor_span: tuple[int, int],
        answer_kinds: tuple[str, ...],
        ) -> str:
    """移除标题后，为带前置实体作用域的数量问式保留关系焦点。"""
    prefix = slot_free_question[:anchor_span[0]]
    suffix = slot_free_question[anchor_span[1]:]
    if _uses_scoped_quantity_focus(prefix, suffix, answer_kinds):
        return suffix
    return prefix + "\n" + suffix


def _uses_scoped_quantity_focus(
        prefix: str, suffix: str, answer_kinds: tuple[str, ...],
        ) -> bool:
    """判断数量属性是否应脱离标题前实体作用域独立排序。"""
    return (
        "QUANTITY" in answer_kinds
        and bool(suffix.strip())
        and prefix.rstrip().endswith(("的", "之"))
    )


def _missing_strong_constraint(
        question: str,
        *,
        title: str,
        candidate_terms: set[str],
        slot_free_question: str,
        candidate_text: str,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> bool:
    """审计标题前实体限定和显式数字，允许关系问式自由改写。"""
    span = _title_span(
        slot_free_question, title, surface_variant_provider)
    if span is None:
        return False
    prefix = slot_free_question[:span[0]]
    explicit_scope = _EXPLICIT_SCOPE_RE.search(prefix)
    if explicit_scope is None:
        prefix = ""
    for match in _CJK_SEQUENCE_RE.finditer(prefix):
        sequence = match.group(0)
        if len(sequence) < 2:
            continue
        terms = _script_terms(sequence, surface_variant_provider)
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
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> bool:
    """显式虚构/不存在限定未被页面标题锚定时，保持 UNKNOWN。"""
    if _title_span(
            slot_free_question, candidate_title,
            surface_variant_provider) is not None:
        return False
    return _NON_REAL_ENTITY_RE.search(slot_free_question) is not None


def _exact_document_title_matches(
        connection: sqlite3.Connection,
        slot_free_question: str,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> dict[int, str]:
    """从通用定义问式解析标题，并查询实际 document 索引。"""
    title_hint = None
    for suffix in ("是\n？", "是\n?", "是？", "是?"):
        if slot_free_question.endswith(suffix):
            candidate = slot_free_question[:-len(suffix)].strip()
            if candidate and "\n" not in candidate:
                title_hint = candidate
                break
    if title_hint is None:
        return {}
    variants = _surface_variants(title_hint, surface_variant_provider)
    placeholders = ",".join("?" for _ in variants)
    rows = connection.execute(
        "SELECT doc_id,title FROM document "
        f"WHERE title IN ({placeholders})",
        variants,
    ).fetchall()
    return {int(doc_id): str(title) for doc_id, title in rows}


def has_exact_broad_qa_title(
        connection: sqlite3.Connection,
        question: str,
        *,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> bool:
    """判断定义问式是否精确锚定 broad index 中的真实页面标题。"""
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("broad QA connection 类型错误")
    if not isinstance(question, str) or not question.strip():
        raise BroadQaQueryError("broad QA question 不能为空")
    slots = load_broad_qa_question_slots()
    return bool(_exact_document_title_matches(
        connection,
        _query_surface(question, slots, surface_variant_provider),
        surface_variant_provider))


def _anchor_matches(
        connection: sqlite3.Connection,
        raw_terms: tuple[str, ...],
        slot_free_question: str,
        max_candidate_passages: int,
        schema_tables: frozenset[str] | None = None,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> tuple[bool, bool, dict[int, str]]:
    """读取标题/别名锚点，供快速路径和常规路径共享。"""
    if not raw_terms:
        return False, False, {}
    placeholders = ",".join("?" for _ in raw_terms)
    if schema_tables is None:
        schema_tables = frozenset(
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('alias','alias_term')")
        )
    alias_table_exists = "alias" in schema_tables
    alias_term_exists = "alias_term" in schema_tables
    matches: dict[int, str] = {}
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
            if _title_span(
                    slot_free_question, surface,
                    surface_variant_provider) is None:
                continue
            prior = matches.get(int(doc_id))
            if prior is None or (-len(surface), surface) < (-len(prior), prior):
                matches[int(doc_id)] = surface
    # v14 保留紧凑 document 表但可能省略 alias_term。通用单实体
    # ``X 是什么`` 问式先做语义标题查询，不把标题或回答写死在代码中。
    if not matches:
        matches = _exact_document_title_matches(
            connection, slot_free_question, surface_variant_provider)
    return alias_table_exists, alias_term_exists, matches


def select_broad_qa_evidence_sentence(
        question: str, context: str,
        *,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> str:
    """从调用方给定的来源上下文中确定性选择一条完整证据句。"""
    if (not isinstance(question, str) or not question.strip()
            or not isinstance(context, str) or not context.strip()):
        raise BroadQaQueryError("broad QA question/context 不能为空")
    slots = load_broad_qa_question_slots()
    surface = _query_surface(question, slots, surface_variant_provider)
    return _best_sentence(
        set(_script_terms(surface, surface_variant_provider)), context,
        slots.answer_kinds(question, surface_variant_provider),
        surface_variant_provider)


def broad_qa_answer_shape_bonus(question: str, evidence_text: str) -> int:
    """公开投影问式槽与证据形态的既有纯整数相容分。

    该函数不查询知识库、不创建答案，只复用冻结 CC0 问式课程和既有证据
    形态合同，供其他来源 passage 检索器保持同一回答侧排序语义。
    """
    if (not isinstance(question, str) or not question.strip()
            or not isinstance(evidence_text, str) or not evidence_text.strip()):
        raise BroadQaQueryError("broad QA answer shape 输入不能为空")
    slots = load_broad_qa_question_slots()
    return _sentence_shape_bonus(slots.answer_kinds(question), evidence_text)


def retrieve_broad_qa_candidates(
        connection: sqlite3.Connection,
        question: str,
        *,
        max_query_terms: int = 24,
        max_candidate_passages: int = 20,
        max_posting_visits: int = 500_000,
        learned_evidence_term_weights: Iterable[tuple[str, int]] | None = None,
        learned_typed_obligation: LearnedTypedObligation | None = None,
        learned_relation_evidence_model: LearnedRelationEvidenceModel | None = None,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        fast_path: bool = False,
        _metadata_values: dict[str, str] | None = None,
        _schema_tables: frozenset[str] | None = None,
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
    if type(fast_path) is not bool:
        raise TypeError("broad QA fast_path 必须是严格 bool")
    if (_metadata_values is None) != (_schema_tables is None):
        raise BroadQaQueryError("broad QA 预载检索上下文不完整")
    metadata = (
        _metadata(connection)
        if _metadata_values is None else _metadata_values)
    if learned_relation_evidence_model is not None and not isinstance(
            learned_relation_evidence_model, LearnedRelationEvidenceModel):
        raise BroadQaQueryError("relation evidence model 类型错误")
    learned_weights = _validate_learned_term_weights(
        learned_evidence_term_weights)
    question_slots = load_broad_qa_question_slots()
    answer_kinds = _answer_kinds(
        question, question_slots, learned_typed_obligation,
        surface_variant_provider)
    slot_free_question = _query_surface(
        question, question_slots, surface_variant_provider)
    fast_exact_matches = (
        _exact_document_title_matches(
            connection, slot_free_question, surface_variant_provider)
        if fast_path else {})
    if (fast_path and not fast_exact_matches
            and has_explicit_non_real_constraint(slot_free_question)):
        # 显式非真实限定构成完整负约束；严格档仍执行完整检索，快速档可在
        # 查询真实标题后、解码无关 posting 前拒答，不削弱 UNKNOWN 安全边界。
        return (), BroadQaRetrievalTrace(
            metadata["snapshot_id"], metadata["license_id"], 0, 0, 0,
            int(metadata["passage_count"]))
    raw_terms = (
        () if fast_exact_matches
        else tuple(sorted(_script_terms(
            slot_free_question, surface_variant_provider))))
    if len(raw_terms) > 512:
        raise BroadQaQueryError("broad QA 原始 query term 超预算")
    if not raw_terms and not fast_exact_matches:
        return (), BroadQaRetrievalTrace(
            metadata["snapshot_id"], metadata["license_id"], 0, 0, 0,
            int(metadata["passage_count"]))
    placeholders = ",".join("?" for _ in raw_terms)
    alias_table_exists = False
    alias_term_exists = False
    anchor_matches: dict[int, str] = {}
    if fast_exact_matches:
        alias_table_exists = bool(
            _schema_tables and "alias" in _schema_tables)
        alias_term_exists = bool(
            _schema_tables and "alias_term" in _schema_tables)
        anchor_matches = fast_exact_matches
    elif fast_path:
        alias_table_exists, alias_term_exists, anchor_matches = _anchor_matches(
            connection, raw_terms, slot_free_question, max_candidate_passages,
            _schema_tables, surface_variant_provider)
    if anchor_matches:
        # 精确标题/别名已经确定页面，无需解码全部 posting；后续仍对来源
        # passage 排序并执行回答门，这里只在显式快速档去除无关全局候选。
        all_rows = ()
        rows = ()
    else:
        all_rows = tuple(connection.execute(
            "SELECT term,document_frequency,passage_deltas FROM posting "
            f"WHERE term IN ({placeholders})",
            raw_terms,
        ))
        all_rows = tuple(sorted(all_rows, key=lambda item: (item[1], item[0])))
        rows = all_rows[:max_query_terms]
    if not rows and not anchor_matches:
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
    if not fast_path:
        alias_table_exists, alias_term_exists, anchor_matches = _anchor_matches(
            connection, raw_terms, slot_free_question, max_candidate_passages,
            _schema_tables, surface_variant_provider)
    anchor_passage_ids = []
    anchor_passages_by_doc: dict[int, tuple[int, ...]] = {}
    if anchor_matches:
        anchor_doc_ids = tuple(sorted(anchor_matches))
        anchor_placeholders = ",".join("?" for _ in anchor_doc_ids)
        if fast_path:
            anchor_rows = connection.execute(
                "SELECT doc_id,MIN(passage_id) FROM passage "
                f"WHERE doc_id IN ({anchor_placeholders}) GROUP BY doc_id",
                anchor_doc_ids,
            ).fetchall()
        else:
            anchor_rows = connection.execute(
                "SELECT doc_id,passage_id FROM passage "
                f"WHERE doc_id IN ({anchor_placeholders})",
                anchor_doc_ids,
            ).fetchall()
        grouped_anchor: dict[int, list[int]] = defaultdict(list)
        for doc_id, passage_id in anchor_rows:
            grouped_anchor[int(doc_id)].append(int(passage_id))
        anchor_passages_by_doc = {
            doc_id: tuple(values) for doc_id, values in grouped_anchor.items()
        }
    for doc_id, _ in sorted(
            anchor_matches.items(), key=lambda item: (
                -len(item[1]), item[1], item[0])):
        passages = anchor_passages_by_doc.get(doc_id, ())
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
    candidate_inputs = []
    term_weights = {
        term: max(1, 1_000_000 // frequency)
        for term, frequency, _ in all_rows}
    for term, weight in learned_weights.items():
        if term in term_weights:
            term_weights[term] += weight
    # release 内候选 passage 与 alias 不可变；用两次有界批量查询替代逐
    # passage/doc SQL，排序仍严格遵循 ranked_ids 与规范 surface 顺序。
    candidate_rows_by_id = {}
    if ranked_ids:
        id_placeholders = ",".join("?" for _ in ranked_ids)
        candidate_rows = connection.execute(f"""
            SELECT p.passage_id,p.raw_start,p.raw_end,p.raw_sha256,p.text,
                   d.doc_id,d.title,d.page_id,d.revision_id,d.timestamp,
                   d.contributor_json
            FROM passage AS p JOIN document AS d ON d.doc_id=p.doc_id
            WHERE p.passage_id IN ({id_placeholders})
        """, ranked_ids).fetchall()
        candidate_rows_by_id = {int(row[0]): row for row in candidate_rows}
    aliases_by_doc: dict[int, tuple[str, ...]] = {}
    if alias_table_exists and candidate_rows_by_id:
        doc_ids = tuple(sorted({int(row[5]) for row in candidate_rows_by_id.values()}))
        doc_placeholders = ",".join("?" for _ in doc_ids)
        alias_rows = connection.execute(
            "SELECT doc_id,surface FROM alias "
            f"WHERE doc_id IN ({doc_placeholders}) ORDER BY doc_id,surface",
            doc_ids,
        ).fetchall()
        grouped: dict[int, list[str]] = defaultdict(list)
        for doc_id, surface in alias_rows:
            grouped[int(doc_id)].append(surface)
        aliases_by_doc = {
            doc_id: tuple(values) for doc_id, values in grouped.items()
        }
    for passage_id in ranked_ids:
        row = candidate_rows_by_id.get(passage_id)
        if row is None:
            raise BroadQaQueryError("broad QA posting 指向缺失 passage")
        aliases = aliases_by_doc.get(int(row[5]), ())
        title_surfaces = (row[6],) + aliases
        if fast_path and int(row[5]) in anchor_matches:
            matching_titles = (anchor_matches[int(row[5])],)
        else:
            matching_titles = tuple(
                title for title in title_surfaces
                if _title_span(
                        slot_free_question, title,
                        surface_variant_provider) is not None)
        candidate_inputs.append((row, title_surfaces, matching_titles))
    dominated_titles = (
        frozenset() if fast_path and len(candidate_inputs) <= 1 else
        _dominated_title_surfaces(
            question,
            frozenset(
                title
                for _, _, matching_titles in candidate_inputs
                for title in matching_titles),
            surface_variant_provider,
        )
    )
    candidate_rows = []
    for row, title_surfaces, matching_titles in candidate_inputs:
        scoring_titles = tuple(
            title for title in matching_titles
            if title not in dominated_titles)
        query_anchor_title = (
            sorted(scoring_titles, key=lambda item: (-len(item), item))[0]
            if scoring_titles else (
                sorted(matching_titles, key=lambda item: (-len(item), item))[0]
                if matching_titles else row[6]))
        if (fast_path and len(anchor_matches) == 1
                and len(candidate_inputs) == 1 and matching_titles):
            candidate_rows.append(BroadQaRetrievalCandidate(
                1_000_000_000 + len(query_anchor_title) * 1_000_000,
                matched[passage_id], row[0], row[1], row[2], row[3],
                row[4], row[5], row[6], query_anchor_title, row[7], row[8],
                row[9], row[10]))
            continue
        title_terms = frozenset().union(*(
            _script_terms(title, surface_variant_provider)
            for title in title_surfaces))
        raw_term_set = set(raw_terms)
        overlap = _script_term_overlap(
            raw_term_set, row[4], surface_variant_provider)
        title_overlap = len(raw_term_set.intersection(title_terms))
        exact_title_score = (
            1_000_000_000 + len(query_anchor_title) * 1_000_000
            if scoring_titles else 0)
        sentence_surface = slot_free_question
        title_span = _title_span(
            sentence_surface, query_anchor_title,
            surface_variant_provider)
        if title_span is not None:
            sentence_surface = (
                sentence_surface[:title_span[0]] + "\n"
                + sentence_surface[title_span[1]:])
        sentence_terms = _script_terms(
            sentence_surface, surface_variant_provider)
        sentence_overlap = _best_sentence_overlap(
            sentence_terms, row[4], surface_variant_provider)
        best_sentence = _best_sentence(
            sentence_terms, row[4], answer_kinds,
            surface_variant_provider)
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
        if fast_path:
            # 快速档已按 MIN(passage_id) 取得首段，直接复用同一规范行，
            # 避免重复 SQL；严格档保留整页扫描及原有完整度。
            page_rows = (candidate_rows_by_id[page_candidate.passage_id],)
        else:
            page_rows = tuple(connection.execute("""
                SELECT p.passage_id,p.raw_start,p.raw_end,p.raw_sha256,p.text,
                       d.doc_id,d.title,d.page_id,d.revision_id,d.timestamp,
                       d.contributor_json
                FROM passage AS p JOIN document AS d ON d.doc_id=p.doc_id
                WHERE d.doc_id=? ORDER BY p.passage_id
            """, (page_candidate.doc_id,)))
        page_terms = _script_terms(
            slot_free_question, surface_variant_provider)
        anchor_span = _title_span(
            slot_free_question, page_candidate.query_anchor_title,
            surface_variant_provider)
        scoped_quantity_focus = False
        if anchor_span is not None:
            scoped_quantity_focus = _uses_scoped_quantity_focus(
                slot_free_question[:anchor_span[0]],
                slot_free_question[anchor_span[1]:], answer_kinds)
            surface = _page_evidence_surface(
                slot_free_question, anchor_span, answer_kinds)
            page_terms = _script_terms(surface, surface_variant_provider)
        ranked_windows = []
        for page_row in page_rows:
            for window_score, selected_text in _rank_evidence_windows(
                set(page_terms), page_row[4], answer_kinds, term_weights,
                relation_evidence_model=learned_relation_evidence_model,
                relation_question=slot_free_question,
                surface_variant_provider=surface_variant_provider):
                if (scoped_quantity_focus
                        and _title_span(
                            selected_text,
                            page_candidate.query_anchor_title,
                            surface_variant_provider) is not None):
                    window_score = (
                        window_score[0], window_score[1],
                        window_score[2] + 1, window_score[3], window_score[4])
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
        selected_windows = _expand_structural_evidence_windows(
            ranked_windows, selected_windows, answer_kinds=answer_kinds)
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
        *,
        learned_typed_obligation: LearnedTypedObligation | None = None,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        ) -> BroadQaResult:
    """对一次已完成的候选检索执行回答、澄清和拒答门。"""
    if (not isinstance(question, str) or not question.strip()
            or not isinstance(candidates, tuple)
            or any(not isinstance(item, BroadQaRetrievalCandidate)
                   for item in candidates)
            or not isinstance(trace, BroadQaRetrievalTrace)):
        raise BroadQaQueryError("broad QA candidate resolution 输入非法")
    question_slots = load_broad_qa_question_slots()
    slot_free_question = _query_surface(
        question, question_slots, surface_variant_provider)
    if not candidates:
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, trace.snapshot_id, trace.license_id,
            trace.matched_query_term_count, 0,
        )
    best = candidates[0]
    exact_title_anchor = (
        _title_span(
            slot_free_question, best.query_anchor_title,
            surface_variant_provider) is not None)
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
    candidate_terms = set(_script_terms(
        candidate_text, surface_variant_provider))
    candidate_terms.update(_script_terms(
        best.title, surface_variant_provider))
    candidate_terms.update(_script_terms(
        best.query_anchor_title, surface_variant_provider))
    if _missing_strong_constraint(
            question, title=best.query_anchor_title,
            candidate_terms=candidate_terms,
            slot_free_question=slot_free_question,
            candidate_text=candidate_text,
            surface_variant_provider=surface_variant_provider):
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, trace.snapshot_id, trace.license_id,
            trace.matched_query_term_count, trace.candidate_document_count,
        )
    if _has_explicit_non_real_entity(
        question, slot_free_question=slot_free_question,
            candidate_title=best.query_anchor_title,
            surface_variant_provider=surface_variant_provider):
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
    if (_title_span(
            slot_free_question, best.query_anchor_title,
            surface_variant_provider) is None
            and len(candidates) > 1
            and candidates[1].doc_id != best.doc_id
            and candidates[1].score * 100 >= best.score * 95):
        return BroadQaResult(
            "CLARIFY", question, None, None, None, None, None, None, None,
            None, None, trace.snapshot_id, trace.license_id,
            trace.matched_query_term_count, trace.candidate_document_count,
        )
    # 没有页面标题锚点时，关系词共现不足以证明回答对象，必须拒答。
    if _title_span(
            slot_free_question, best.query_anchor_title,
            surface_variant_provider) is None:
        return BroadQaResult(
            "UNKNOWN", question, None, None, None, None, None, None, None,
            None, None, trace.snapshot_id, trace.license_id,
            trace.matched_query_term_count, trace.candidate_document_count,
        )
    sentence_question = slot_free_question
    span = _title_span(
        sentence_question, best.query_anchor_title, surface_variant_provider)
    if span is not None:
        sentence_question = (
            sentence_question[:span[0]] + "\n" + sentence_question[span[1]:])
    question_terms = _script_terms(
        sentence_question, surface_variant_provider)
    answer_kinds = _answer_kinds(
        question, question_slots, learned_typed_obligation,
        surface_variant_provider)
    evidence_candidates = _remove_redundant_evidence_candidates(
        evidence_candidates)
    primary_evidence = evidence_candidates[0]
    citations = []
    answer_parts = []
    for evidence in evidence_candidates[:_MAX_EVIDENCE_CITATIONS]:
        window = evidence.selected_text
        if not window:
            _, window = _best_evidence_window(
                question_terms, evidence.text, answer_kinds, {},
                surface_variant_provider=surface_variant_provider)
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
        f"{primary_evidence.page_id}&oldid={primary_evidence.revision_id}"
    )
    return BroadQaResult(
        "ANSWER", question, answer, primary_evidence.title,
        primary_evidence.page_id, primary_evidence.revision_id,
        primary_evidence.text, primary_evidence.raw_start,
        primary_evidence.raw_end, primary_evidence.raw_sha256, source_url,
        trace.snapshot_id, trace.license_id,
        trace.matched_query_term_count, trace.candidate_document_count,
        primary_evidence.revision_timestamp,
        primary_evidence.contributor_json,
        WIKIPEDIA_ATTRIBUTION, tuple(citations),
    )


def query_broad_qa(
        connection: sqlite3.Connection,
        question: str,
        *,
        max_query_terms: int = 24,
        max_candidate_passages: int = 20,
        max_posting_visits: int = 500_000,
        learned_evidence_term_weights: Iterable[tuple[str, int]] | None = None,
        learned_typed_obligation: LearnedTypedObligation | None = None,
        learned_relation_evidence_model: LearnedRelationEvidenceModel | None = None,
        query_cache: BroadQaQueryCache | None = None,
        surface_variant_provider: SurfaceVariantProvider | None = None,
        fast_path: bool = False,
        ) -> BroadQaResult:
    """以稀有 term postings 缩小候选，并从 top-K 页面投影可引用回答。"""
    if query_cache is not None:
        if not isinstance(query_cache, BroadQaQueryCache):
            raise TypeError("query_cache 类型错误")
        if not query_cache.owns(connection):
            raise ValueError("query_cache 未绑定当前 SQLite 连接")
    if type(fast_path) is not bool:
        raise TypeError("broad QA fast_path 必须是严格 bool")
    cacheable = (
        query_cache is not None
        and surface_variant_provider is None
        and learned_evidence_term_weights is None
        and learned_typed_obligation is None
        and learned_relation_evidence_model is None
    )
    cache_key = (
        question, max_query_terms, max_candidate_passages, max_posting_visits,
        fast_path)
    if cacheable:
        cached = query_cache.get(cache_key)
        if cached is not None:
            return cached
    metadata_values = None
    schema_tables = None
    if query_cache is not None:
        metadata_values, schema_tables = query_cache._retrieval_context()
    candidates, trace = retrieve_broad_qa_candidates(
        connection, question,
        max_query_terms=max_query_terms,
        max_candidate_passages=max_candidate_passages,
        max_posting_visits=max_posting_visits,
        learned_evidence_term_weights=learned_evidence_term_weights,
        learned_typed_obligation=learned_typed_obligation,
        learned_relation_evidence_model=learned_relation_evidence_model,
        surface_variant_provider=surface_variant_provider,
        fast_path=fast_path,
        _metadata_values=metadata_values,
        _schema_tables=schema_tables)
    result = answer_broad_qa_candidates(
        question, candidates, trace,
        learned_typed_obligation=learned_typed_obligation,
        surface_variant_provider=surface_variant_provider)
    if cacheable:
        query_cache.put(cache_key, result)
    return result


__all__ = [
    "answer_broad_qa_candidates",
    "broad_qa_answer_shape_bonus",
    "BroadQaRetrievalCandidate",
    "BroadQaRetrievalTrace",
    "BroadQaQueryCache",
    "BroadQaQueryError",
    "has_exact_broad_qa_title",
    "query_broad_qa",
    "has_explicit_non_real_constraint",
    "retrieve_broad_qa_candidates",
    "select_broad_qa_evidence_sentence",
]
