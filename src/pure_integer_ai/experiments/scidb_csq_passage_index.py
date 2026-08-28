"""构建并查询 CSQ train-only 来源知识段索引。

索引只发布课程中结构化的 ``知识点`` 文本。train 题干仅投影为二/三元稀疏
检索特征，原题文本、选项、最终答案和解题过程均不写入数据库；完整题目不作为
键，也不存在题目到完整答案映射，因此该 artifact 是来源文档索引而不是固定
问答表。held-out 仅形成不可逆摘要和计数，绝不进入索引。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_index import broad_qa_terms
from pure_integer_ai.experiments.ph2_broad_qa_query import (
    SurfaceVariantProvider,
    broad_qa_answer_shape_bonus,
    has_explicit_non_real_constraint,
)
from pure_integer_ai.experiments.ph2_broad_qa_question_slots import (
    load_broad_qa_question_slots,
)
from pure_integer_ai.storage.integer_codec import (
    decode_integer_tuple,
    encode_integer_tuple,
)


ARTIFACT_KIND = "PURE_INTEGER_AI_SCIDB_CSQ_PASSAGE_INDEX_V2"
INDEX_SCHEMA_VERSION = 2
COURSE_FORMAT = "PURE_INTEGER_AI_SCIDB_CSQ_COURSE_V1"
LICENSE_ID = "CC-BY-4.0"
ATTRIBUTION = (
    "Liu, Zhi; Li, Dong; Long, Taotao; Wen, Chaodong; "
    "Peng, Xian; Guo, Jiaxin"
)
_CONTEXT_FIELDS = (
    ("学段：", "grade"),
    ("领域：", "theme"),
    ("主题：", "category"),
    ("知识点：", "knowledge"),
    ("提示：", "hint"),
    ("科学技能：", "skills"),
)
_MANIFEST_NAME = "scidb_csq_passage_index_manifest.json"
_DATABASE_NAME = "scidb_csq_passage_index.sqlite3"
_CJK_SEQUENCE_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+")


# object-model: exception
class ScidbCsqPassageIndexError(ValueError):
    """CSQ passage 输入、来源、schema 或查询边界发生漂移。"""


def _sha256_bytes(payload: bytes) -> str:
    """计算有限 byte sequence 的 SHA-256 小写十六进制身份。"""
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    """流式计算既有文件 SHA-256，避免大 artifact 整体常驻内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_line(value: object) -> bytes:
    """输出排序键、紧凑 UTF-8 且单换行结尾的规范 JSON。"""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    """核验外部承诺为规范小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef"
                   for character in value)):
        raise ScidbCsqPassageIndexError(f"{label} 非法")
    return value


def _require_k_path(path: Path, *, label: str) -> Path:
    """确保大规模课程和索引位于唯一训练盘 K:。"""
    resolved = path.resolve()
    if resolved.drive.upper() != "K:":
        raise ScidbCsqPassageIndexError(f"{label} 必须位于 K 盘")
    return resolved


def _parse_context(surface: object) -> dict[str, str]:
    """按课程 V1 冻结的六行标签合同恢复结构字段。"""
    if not isinstance(surface, str):
        raise ScidbCsqPassageIndexError("CSQ context_surface 类型非法")
    lines = surface.split("\n")
    result: dict[str, str] = {}
    field_index = -1
    values: list[str] = []
    for line in lines:
        next_index = field_index + 1
        if (next_index < len(_CONTEXT_FIELDS)
                and line.startswith(_CONTEXT_FIELDS[next_index][0])):
            if field_index >= 0:
                prior_key = _CONTEXT_FIELDS[field_index][1]
                prior_value = "\n".join(values).strip()
                if not prior_value:
                    raise ScidbCsqPassageIndexError(
                        f"CSQ context_surface {prior_key} 为空")
                result[prior_key] = prior_value
            field_index = next_index
            values = [line[len(_CONTEXT_FIELDS[field_index][0]):]]
            continue
        if field_index < 0:
            raise ScidbCsqPassageIndexError("CSQ context_surface 首字段漂移")
        if any(line.startswith(prefix) for prefix, _key in _CONTEXT_FIELDS):
            raise ScidbCsqPassageIndexError("CSQ context_surface 标签顺序漂移")
        values.append(line)
    if field_index != len(_CONTEXT_FIELDS) - 1:
        missing = _CONTEXT_FIELDS[field_index + 1][0]
        raise ScidbCsqPassageIndexError(
            f"CSQ context_surface 缺少 {missing}")
    final_key = _CONTEXT_FIELDS[field_index][1]
    final_value = "\n".join(values).strip()
    if not final_value:
        raise ScidbCsqPassageIndexError(
            f"CSQ context_surface {final_key} 为空")
    result[final_key] = final_value
    return result


def _source_record_sha256(sample_id: object, doi: str,
                          record_id: int) -> str:
    """从课程稳定 sample identity 恢复上游规范记录摘要。"""
    prefix = f"{doi}:{record_id}:"
    if not isinstance(sample_id, str) or not sample_id.startswith(prefix):
        raise ScidbCsqPassageIndexError("CSQ sample_id 与来源记录不一致")
    return _require_sha256(sample_id[len(prefix):], label="source record SHA-256")


def _source_ref(record: dict[str, object]) -> tuple[int, ...]:
    """核验课程携带的 LC-16 十一整数来源引用。"""
    value = record.get("source_ref_key")
    if (not isinstance(value, list) or len(value) != 11
            or any(type(item) is not int or item < 0 for item in value)):
        raise ScidbCsqPassageIndexError("CSQ source_ref_key 非法")
    result = tuple(value)
    if (result[0] != record.get("source_kind")
            or result[1] != record.get("source_id")
            or result[2] != record.get("source_record_id")):
        raise ScidbCsqPassageIndexError("CSQ SourceRef 与来源字段漂移")
    return result


def _delta(values: Iterable[int]) -> tuple[int, ...]:
    """把严格递增 passage id 编成正整数 delta 序列。"""
    result = []
    prior = 0
    for value in values:
        if type(value) is not int or value <= prior:
            raise ScidbCsqPassageIndexError("CSQ posting id 非严格递增")
        result.append(value - prior)
        prior = value
    return tuple(result)


def _restore_posting(payload: bytes) -> tuple[int, ...]:
    """恢复规范正 delta posting，并拒绝非正或非递增结果。"""
    deltas = decode_integer_tuple(payload)
    restored = []
    cursor = 0
    for value in deltas:
        if value <= 0:
            raise ScidbCsqPassageIndexError("CSQ posting delta 必须为正")
        cursor += value
        restored.append(cursor)
    return tuple(restored)


def _ordered_focus_term(
        query_surface: str,
        frequencies: dict[str, int],
        *,
        maximum_document_frequency: int = 512,
        ) -> str | None:
    """选择表达顺序中最早出现、正文可辨识的稀疏主题 term。

    不登记词表或实体类型；中文按原字符位置枚举二元组，ASCII 按完整单词。
    超过 document-frequency 上限的通用 term 不能成为锚。
    """
    if (not isinstance(query_surface, str)
            or not isinstance(frequencies, dict)
            or type(maximum_document_frequency) is not int
            or maximum_document_frequency <= 0):
        raise ScidbCsqPassageIndexError("CSQ focus term 输入非法")
    ordered: list[tuple[int, str]] = []
    for match in _CJK_SEQUENCE_RE.finditer(query_surface):
        sequence = match.group(0)
        for offset in range(max(0, len(sequence) - 1)):
            ordered.append((match.start() + offset,
                            "c:" + sequence[offset:offset + 2]))
    ordered.extend(
        (match.start(), "w:" + match.group(0).casefold())
        for match in _ASCII_WORD_RE.finditer(query_surface)
    )
    for _position, term in sorted(ordered, key=lambda item: (item[0], item[1])):
        frequency = frequencies.get(term)
        if frequency is not None and frequency <= maximum_document_frequency:
            return term
    return None


def _normalize_causal_predicate_surface(surface: str) -> str:
    """移除因果谓词句末的结构助词，避免其 n-gram 冒充知识主题。"""
    if not isinstance(surface, str):
        raise ScidbCsqPassageIndexError("CSQ causal predicate surface 非法")
    end = len(surface)
    while end > 0 and (surface[end - 1].isspace()
                       or surface[end - 1] in "?？!！。"):
        end -= 1
    if end > 0 and surface[end - 1] == "的":
        return surface[:end - 1] + surface[end:]
    return surface


def _leading_topic_term(
        surface: str, frequencies: dict[str, int], *,
        maximum_document_frequency: int = 512,
        ) -> str | None:
    """要求因果问式首个语言片段具有正文证据，禁止仅凭通用谓词作答。"""
    if (not isinstance(surface, str) or not isinstance(frequencies, dict)
            or type(maximum_document_frequency) is not int
            or maximum_document_frequency <= 0):
        raise ScidbCsqPassageIndexError("CSQ leading topic 输入非法")
    cjk = _CJK_SEQUENCE_RE.search(surface)
    if cjk is not None:
        sequence = cjk.group(0)
        candidates = tuple(
            "c:" + sequence[:width]
            for width in (3, 2) if len(sequence) >= width)
    else:
        ascii_word = _ASCII_WORD_RE.search(surface)
        candidates = (() if ascii_word is None else (
            "w:" + ascii_word.group(0).casefold(),))
    for term in candidates:
        frequency = frequencies.get(term)
        if frequency is not None and frequency <= maximum_document_frequency:
            return term
    return None


def _parse_course_record(raw_line: bytes) -> dict[str, object]:
    """严格回读一条无 BOM、无首尾空白的课程 JSON object。"""
    if not raw_line or raw_line != raw_line.strip():
        raise ScidbCsqPassageIndexError("CSQ 课程行存在空白漂移")
    try:
        value = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScidbCsqPassageIndexError("CSQ 课程不是规范 UTF-8 JSONL") from error
    if (not isinstance(value, dict)
            or value.get("format") != COURSE_FORMAT
            or value.get("license_id") != LICENSE_ID
            or value.get("split") not in {"train", "heldout"}):
        raise ScidbCsqPassageIndexError("CSQ 课程身份或 split 漂移")
    return value


def build_scidb_csq_passage_index(
        *,
        course_path: str | Path,
        artifact_root: str | Path,
        expected_course_sha256: str,
        expected_source_sha256: str,
        require_k_drive: bool = True,
        ) -> dict[str, object]:
    """把 CSQ train 知识点构建为来源约束 document/passage/posting artifact。

    构建过程按课程行流式读取；只在内存保留紧凑 posting id 列表。数据库和
    manifest 均禁止覆盖，manifest 最后发布。held-out 的题目只进入有域分离的
    SHA-256 累积，不进入任何可查询表。
    """
    course = Path(course_path).resolve()
    root = Path(artifact_root).resolve()
    if require_k_drive:
        course = _require_k_path(course, label="CSQ course")
        root = _require_k_path(root, label="CSQ passage artifact root")
    if not course.is_file():
        raise ScidbCsqPassageIndexError("CSQ course 不存在")
    if root.exists():
        raise FileExistsError(root)
    expected_course = _require_sha256(
        expected_course_sha256, label="expected course SHA-256")
    expected_source = _require_sha256(
        expected_source_sha256, label="expected source SHA-256")
    actual_course = _sha256_path(course)
    if actual_course != expected_course:
        raise ScidbCsqPassageIndexError(
            f"CSQ course SHA-256 不匹配: {actual_course}")
    root.mkdir(parents=True, exist_ok=False)
    partial = root / (_DATABASE_NAME + ".partial")
    database = root / _DATABASE_NAME
    manifest_path = root / _MANIFEST_NAME
    connection = sqlite3.connect(str(partial))
    postings: dict[str, list[int]] = defaultdict(list)
    question_postings: dict[str, list[int]] = defaultdict(list)
    train_count = 0
    heldout_count = 0
    domain_train: dict[str, int] = defaultdict(int)
    domain_heldout: dict[str, int] = defaultdict(int)
    heldout_digest = hashlib.sha256(
        b"PURE-INTEGER-AI/SCIDB-CSQ-HELDOUT-QUESTIONS/V1\0")
    doi = None
    dataset_url = None
    source_kind = None
    source_id = None
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE document(
                doc_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                source_kind INTEGER NOT NULL,
                source_id INTEGER NOT NULL,
                source_record_id INTEGER NOT NULL UNIQUE,
                source_ref BLOB NOT NULL,
                source_record_sha256 TEXT NOT NULL,
                course_raw_start INTEGER NOT NULL,
                course_raw_end INTEGER NOT NULL,
                course_raw_sha256 TEXT NOT NULL,
                theme TEXT NOT NULL,
                category TEXT NOT NULL,
                grade TEXT NOT NULL,
                source_url TEXT NOT NULL,
                license_id TEXT NOT NULL,
                attribution TEXT NOT NULL
            );
            CREATE TABLE passage(
                passage_id INTEGER PRIMARY KEY,
                doc_id INTEGER NOT NULL UNIQUE,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                text_sha256 TEXT NOT NULL
            );
            CREATE TABLE posting(
                term TEXT PRIMARY KEY,
                document_frequency INTEGER NOT NULL,
                passage_deltas BLOB NOT NULL
            );
            CREATE TABLE question_posting(
                term TEXT PRIMARY KEY,
                document_frequency INTEGER NOT NULL,
                passage_deltas BLOB NOT NULL
            );
            CREATE INDEX document_theme ON document(theme, doc_id);
            CREATE INDEX passage_doc ON passage(doc_id);
        """)
        cursor = 0
        with course.open("rb") as stream:
            for line_ordinal, line in enumerate(stream, start=1):
                if not line.endswith(b"\n") or line.endswith(b"\n\n"):
                    raise ScidbCsqPassageIndexError(
                        f"CSQ 课程第 {line_ordinal} 行换行非法")
                raw_line = line[:-1]
                raw_start = cursor
                raw_end = cursor + len(raw_line)
                cursor += len(line)
                record = _parse_course_record(raw_line)
                context = _parse_context(record.get("context_surface"))
                record_source_sha = _require_sha256(
                    record.get("source_sha256"), label="course source SHA-256")
                if record_source_sha != expected_source:
                    raise ScidbCsqPassageIndexError("CSQ source SHA-256 漂移")
                record_id = record.get("source_record_id")
                if type(record_id) is not int or record_id <= 0:
                    raise ScidbCsqPassageIndexError("CSQ source_record_id 非法")
                current_doi = record.get("source_dataset_doi")
                current_url = record.get("source_dataset_url")
                current_kind = record.get("source_kind")
                current_source_id = record.get("source_id")
                if (not isinstance(current_doi, str) or not current_doi
                        or not isinstance(current_url, str)
                        or not current_url.startswith("https://")
                        or type(current_kind) is not int or current_kind <= 0
                        or type(current_source_id) is not int
                        or current_source_id <= 0):
                    raise ScidbCsqPassageIndexError("CSQ 来源身份非法")
                current_identity = (
                    current_doi, current_url, current_kind, current_source_id)
                frozen_identity = (doi, dataset_url, source_kind, source_id)
                if doi is None:
                    doi, dataset_url, source_kind, source_id = current_identity
                elif current_identity != frozen_identity:
                    raise ScidbCsqPassageIndexError("CSQ 课程混入多来源身份")
                if record["split"] == "heldout":
                    question = record.get("question_surface")
                    if not isinstance(question, str) or not question.strip():
                        raise ScidbCsqPassageIndexError("CSQ held-out question 非法")
                    heldout_digest.update(record_id.to_bytes(8, "big"))
                    heldout_digest.update(
                        hashlib.sha256(question.encode("utf-8")).digest())
                    heldout_count += 1
                    domain_heldout[context["theme"]] += 1
                    continue
                train_count += 1
                doc_id = train_count
                knowledge = context["knowledge"]
                question = record.get("question_surface")
                if not isinstance(question, str) or not question.strip():
                    raise ScidbCsqPassageIndexError("CSQ train question 非法")
                question_stem, separator, _options = question.partition("\n选项：")
                if not separator or not question_stem.strip():
                    raise ScidbCsqPassageIndexError(
                        "CSQ train question/options 结构漂移")
                source_ref = _source_ref(record)
                title = f"CSQ {context['category']} #{record_id}"
                connection.execute(
                    "INSERT INTO document VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (doc_id, title, current_kind, current_source_id, record_id,
                     encode_integer_tuple(source_ref),
                     _source_record_sha256(
                         record.get("sample_id"), current_doi, record_id),
                     raw_start, raw_end, _sha256_bytes(raw_line),
                     context["theme"], context["category"], context["grade"],
                     current_url, LICENSE_ID, ATTRIBUTION),
                )
                connection.execute(
                    "INSERT INTO passage VALUES(?,?,?,?,?)",
                    (doc_id, doc_id, 1, knowledge,
                     _sha256_bytes(knowledge.encode("utf-8"))),
                )
                # 知识正文 posting 决定候选准入；train 题干只进入独立、受限
                # 的辅助 posting。两者都不保存完整题目、选项、答案或过程。
                terms = broad_qa_terms(
                    f"{context['theme']}\n{context['category']}\n{knowledge}")
                question_terms = broad_qa_terms(question_stem.strip())
                for term in terms:
                    postings[term].append(doc_id)
                for term in question_terms:
                    question_postings[term].append(doc_id)
                domain_train[context["theme"]] += 1
        if train_count == 0 or heldout_count == 0:
            raise ScidbCsqPassageIndexError("CSQ train/held-out 不完整")
        for term in sorted(postings):
            passage_ids = postings[term]
            connection.execute(
                "INSERT INTO posting VALUES(?,?,?)",
                (term, len(passage_ids),
                 encode_integer_tuple(_delta(passage_ids))),
            )
        for term in sorted(question_postings):
            passage_ids = question_postings[term]
            connection.execute(
                "INSERT INTO question_posting VALUES(?,?,?)",
                (term, len(passage_ids),
                 encode_integer_tuple(_delta(passage_ids))),
            )
        metadata = {
            "artifact_kind": ARTIFACT_KIND,
            "attribution": ATTRIBUTION,
            "course_sha256": actual_course,
            "heldout_count": str(heldout_count),
            "heldout_question_sha256": heldout_digest.hexdigest(),
            "index_schema_version": str(INDEX_SCHEMA_VERSION),
            "license_id": LICENSE_ID,
            "passage_count": str(train_count),
            "source_dataset_doi": str(doi),
            "source_dataset_url": str(dataset_url),
            "source_sha256": expected_source,
            "term_count": str(len(postings)),
            "question_feature_term_count": str(len(question_postings)),
            "train_count": str(train_count),
        }
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)", sorted(metadata.items()))
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    partial.replace(database)
    database_sha = _sha256_path(database)
    manifest = {
        "artifact_kind": ARTIFACT_KIND,
        "attribution": ATTRIBUTION,
        "course": {
            "path": course.name,
            "sha256": actual_course,
        },
        "database": {
            "bytes": database.stat().st_size,
            "path": _DATABASE_NAME,
            "sha256": database_sha,
        },
        "domain_heldout_counts": [
            [key, domain_heldout[key]] for key in sorted(domain_heldout)],
        "domain_train_counts": [
            [key, domain_train[key]] for key in sorted(domain_train)],
        "heldout_count": heldout_count,
        "heldout_question_sha256": heldout_digest.hexdigest(),
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "license_id": LICENSE_ID,
        "source": {
            "dataset_url": dataset_url,
            "doi": doi,
            "source_id": source_id,
            "source_kind": source_kind,
            "source_sha256": expected_source,
        },
        "term_count": len(postings),
        "question_feature_term_count": len(question_postings),
        "train_count": train_count,
    }
    manifest_payload = _canonical_line(manifest)
    manifest_path.write_bytes(manifest_payload)
    return {
        "database_bytes": database.stat().st_size,
        "database_path": database.as_posix(),
        "database_sha256": database_sha,
        "heldout_count": heldout_count,
        "heldout_question_sha256": heldout_digest.hexdigest(),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _sha256_bytes(manifest_payload),
        "term_count": len(postings),
        "question_feature_term_count": len(question_postings),
        "train_count": train_count,
    }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ScidbCsqPassageResult:
    """一次稀疏来源查询的值结果和显式读取预算计数。"""

    status: str
    surface: str | None
    source_title: str | None
    source_url: str | None
    source_ref: tuple[int, ...] | None
    license_id: str | None
    attribution: str | None
    matched_term_count: int
    posting_visit_count: int
    candidate_document_count: int
    confidence_permille: int


# object-model: resource_owner; representation=class; interop=pending
class ScidbCsqPassageRuntime:
    """唯一持有只读 SQLite 连接的 CSQ passage 查询资源 owner。"""

    __slots__ = ("connection", "manifest", "root", "_closed")

    def __init__(
            self, artifact_root: str | Path, *, verify_database_sha256: bool = True,
            require_k_drive: bool = True,
            ) -> None:
        """回读 manifest、物理 artifact 和 metadata 后打开只读连接。"""
        if type(verify_database_sha256) is not bool:
            raise TypeError("verify_database_sha256 必须是严格 bool")
        root = Path(artifact_root).resolve()
        if require_k_drive:
            root = _require_k_path(root, label="CSQ passage artifact root")
        manifest_path = root / _MANIFEST_NAME
        database = root / _DATABASE_NAME
        if not manifest_path.is_file() or not database.is_file():
            raise ScidbCsqPassageIndexError("CSQ passage artifact 不完整")
        payload = manifest_path.read_bytes()
        try:
            manifest = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ScidbCsqPassageIndexError("CSQ passage manifest 不可回读") from error
        if (not isinstance(manifest, dict)
                or manifest.get("artifact_kind") != ARTIFACT_KIND
                or manifest.get("index_schema_version") != INDEX_SCHEMA_VERSION
                or manifest.get("license_id") != LICENSE_ID
                or _canonical_line(manifest) != payload):
            raise ScidbCsqPassageIndexError("CSQ passage manifest 漂移")
        database_spec = manifest.get("database")
        if (not isinstance(database_spec, dict)
                or database_spec.get("path") != _DATABASE_NAME
                or database_spec.get("bytes") != database.stat().st_size):
            raise ScidbCsqPassageIndexError("CSQ passage database identity 漂移")
        expected_sha = _require_sha256(
            database_spec.get("sha256"), label="passage database SHA-256")
        if verify_database_sha256 and _sha256_path(database) != expected_sha:
            raise ScidbCsqPassageIndexError("CSQ passage database SHA-256 不匹配")
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if (metadata.get("artifact_kind") != ARTIFACT_KIND
                or metadata.get("index_schema_version")
                != str(INDEX_SCHEMA_VERSION)
                or metadata.get("license_id") != LICENSE_ID
                or metadata.get("course_sha256")
                != manifest["course"]["sha256"]
                or metadata.get("source_sha256")
                != manifest["source"]["source_sha256"]
                or metadata.get("heldout_question_sha256")
                != manifest["heldout_question_sha256"]):
            connection.close()
            raise ScidbCsqPassageIndexError("CSQ passage metadata 漂移")
        self.connection = connection
        self.manifest = manifest
        self.root = root
        self._closed = False

    def close(self) -> None:
        """幂等关闭唯一 SQLite 资源。"""
        if not self._closed:
            self.connection.close()
            self._closed = True

    def __enter__(self) -> "ScidbCsqPassageRuntime":
        """返回已核验的只读 runtime。"""
        return self

    def __exit__(self, _type: object, _value: object,
                 _traceback: object) -> None:
        """离开作用域时关闭只读数据库。"""
        self.close()

    def query(
            self,
            question: str,
            *,
            max_query_terms: int = 32,
            max_candidate_documents: int = 32,
            max_posting_visits: int = 250_000,
            minimum_confidence_permille: int = 220,
            minimum_margin_permille: int = 1000,
            surface_variant_provider: SurfaceVariantProvider | None = None,
            ) -> ScidbCsqPassageResult:
        """以稀有 n-gram posting 检索知识段，并用整数覆盖/间隔门拒答。"""
        if self._closed:
            raise ScidbCsqPassageIndexError("CSQ passage runtime 已关闭")
        if not isinstance(question, str) or not question.strip():
            raise ScidbCsqPassageIndexError("CSQ passage question 不能为空")
        limits = (
            (max_query_terms, 1, 128, "max_query_terms"),
            (max_candidate_documents, 1, 256, "max_candidate_documents"),
            (max_posting_visits, 1, 2_000_000, "max_posting_visits"),
            (minimum_confidence_permille, 1, 1000,
             "minimum_confidence_permille"),
            (minimum_margin_permille, 1000, 5000,
             "minimum_margin_permille"),
        )
        for value, lower, upper, label in limits:
            if type(value) is not int or not lower <= value <= upper:
                raise ScidbCsqPassageIndexError(f"{label} 非法")
        if has_explicit_non_real_constraint(question):
            return ScidbCsqPassageResult(
                "UNKNOWN", None, None, None, None, None, None,
                0, 0, 0, 0)
        question_slots = load_broad_qa_question_slots()
        query_surface = question_slots.strip_slots(
            question, surface_variant_provider)
        causal_question = "CAUSE" in question_slots.answer_kinds(
            question, surface_variant_provider)
        if causal_question:
            query_surface = _normalize_causal_predicate_surface(query_surface)
        terms = broad_qa_terms(query_surface)
        if not terms:
            return ScidbCsqPassageResult(
                "UNKNOWN", None, None, None, None, None, None,
                0, 0, 0, 0)
        placeholders = ",".join("?" for _ in terms)
        rows = self.connection.execute(
            "SELECT term,document_frequency,passage_deltas FROM posting "
            f"WHERE term IN ({placeholders})",
            terms,
        ).fetchall()
        rows.sort(key=lambda item: (int(item[1]), str(item[0])))
        rows = rows[:max_query_terms]
        if len(rows) < 2:
            return ScidbCsqPassageResult(
                "UNKNOWN", None, None, None, None, None, None,
                len(rows), 0, 0, 0)
        body_scores: dict[int, int] = defaultdict(int)
        body_matches: dict[int, int] = defaultdict(int)
        term_frequencies = {
            str(term): int(frequency) for term, frequency, _payload in rows}
        focus_term = (
            _leading_topic_term(query_surface, term_frequencies)
            if causal_question else _ordered_focus_term(
                query_surface, term_frequencies))
        if causal_question and focus_term is None:
            return ScidbCsqPassageResult(
                "UNKNOWN", None, None, None, None, None, None,
                len(rows), 0, 0, 0)
        predicate_segments = tuple(
            segment.strip() for segment in query_surface.split("\n")
            if segment.strip())
        predicate_surface = predicate_segments[-1] if predicate_segments else ""
        tail_surface = predicate_surface[
            (len(predicate_surface) * 2) // 3:]
        tail_terms = frozenset(
            term for term in broad_qa_terms(tail_surface)
            if term in term_frequencies and term_frequencies[term] <= 512)
        if not tail_terms:
            tail_surface = predicate_surface[len(predicate_surface) // 2:]
            tail_terms = frozenset(
                term for term in broad_qa_terms(tail_surface)
                if term in term_frequencies
                and term_frequencies[term] <= 512)
        focus_passage_ids: frozenset[int] | None = None
        tail_matches: dict[int, int] = defaultdict(int)
        posting_visits = 0
        total_weight = 0
        for _term, frequency, payload in rows:
            frequency = int(frequency)
            if posting_visits + frequency > max_posting_visits:
                return ScidbCsqPassageResult(
                    "UNKNOWN", None, None, None, None, None, None,
                    len(rows), posting_visits, len(body_scores), 0)
            ids = _restore_posting(bytes(payload))
            if len(ids) != frequency:
                raise ScidbCsqPassageIndexError("CSQ posting frequency 漂移")
            # 单个偶然稀有 n-gram 不得支配完整查询；正文覆盖的多特征
            # 一致性优先于孤立 IDF 峰值。
            weight = min(100_000, max(1, 1_000_000 // frequency))
            total_weight += weight
            posting_visits += frequency
            if _term == focus_term:
                focus_passage_ids = frozenset(ids)
            for passage_id in ids:
                body_scores[passage_id] += weight
                body_matches[passage_id] += 1
                if _term in tail_terms:
                    tail_matches[passage_id] += 1
        question_rows = self.connection.execute(
            "SELECT term,document_frequency,passage_deltas "
            "FROM question_posting "
            f"WHERE term IN ({placeholders})",
            terms,
        ).fetchall()
        question_rows.sort(key=lambda item: (int(item[1]), str(item[0])))
        question_scores: dict[int, int] = defaultdict(int)
        question_matches: dict[int, int] = defaultdict(int)
        for _term, frequency, payload in question_rows[:max_query_terms]:
            frequency = int(frequency)
            if posting_visits + frequency > max_posting_visits:
                break
            ids = _restore_posting(bytes(payload))
            if len(ids) != frequency:
                raise ScidbCsqPassageIndexError(
                    "CSQ question posting frequency 漂移")
            # 题干特征只能微调已由正文命中的候选；单项和累计贡献都受
            # 正文分数上限约束，稀有措辞不能再次压过知识内容。
            weight = min(50_000, max(1, 1_000_000 // frequency))
            posting_visits += frequency
            for passage_id in ids:
                if body_matches.get(passage_id, 0) >= 2:
                    question_scores[passage_id] += weight
                    question_matches[passage_id] += 1
        scores = {
            passage_id: (
                body_scores[passage_id]
                + min(question_scores.get(passage_id, 0),
                      body_scores[passage_id] * 2)
                + body_matches[passage_id] * 10_000
                + min(question_matches.get(passage_id, 0),
                      body_matches[passage_id]) * 50_000
            )
            for passage_id in body_scores
            if body_matches[passage_id] >= 2
            and (focus_passage_ids is None
                 or passage_id in focus_passage_ids)
            and (not tail_terms or tail_matches[passage_id] >= 1)
        }
        preliminary_limit = min(256, max_candidate_documents * 4)
        preliminary_ids = tuple(
            passage_id for passage_id, _score in sorted(
                scores.items(),
                key=lambda item: (
                    -item[1], -body_matches[item[0]], item[0]),
            )[:preliminary_limit]
        )
        if not preliminary_ids or total_weight <= 0:
            return ScidbCsqPassageResult(
                "UNKNOWN", None, None, None, None, None, None,
                len(rows), posting_visits, 0, 0)
        candidate_placeholders = ",".join("?" for _ in preliminary_ids)
        candidate_rows = self.connection.execute(f"""
            SELECT p.passage_id,p.text,p.text_sha256,d.title,d.source_ref,
                   d.source_url,d.license_id,d.attribution
            FROM passage AS p JOIN document AS d ON d.doc_id=p.doc_id
            WHERE p.passage_id IN ({candidate_placeholders})
        """, preliminary_ids).fetchall()
        by_id = {int(row[0]): row for row in candidate_rows}
        for passage_id in preliminary_ids:
            row = by_id.get(passage_id)
            if row is None:
                raise ScidbCsqPassageIndexError(
                    "CSQ posting 指向缺失 passage")
            scores[passage_id] += (
                min(3, broad_qa_answer_shape_bonus(question, str(row[1])))
                * 25_000)
        ranked_ids = tuple(sorted(
            preliminary_ids,
            key=lambda passage_id: (
                -scores[passage_id], -body_matches[passage_id], passage_id),
        )[:max_candidate_documents])
        # 相同知识点的多个来源记录折叠成一个语义候选，避免重复样本制造
        # 虚假的歧义；保留排序最前、来源 id 最小的一条引用。
        grouped = {}
        for passage_id in ranked_ids:
            row = by_id.get(passage_id)
            if row is None:
                raise ScidbCsqPassageIndexError("CSQ posting 指向缺失 passage")
            key = str(row[2])
            prior = grouped.get(key)
            candidate = (
                scores[passage_id], body_matches[passage_id], passage_id, row)
            if prior is None or (-candidate[0], -candidate[1], candidate[2]) < (
                    -prior[0], -prior[1], prior[2]):
                grouped[key] = candidate
        ranked = sorted(grouped.values(), key=lambda item: (
            -item[0], -item[1], item[2]))
        best_score, best_matches, best_id, best = ranked[0]
        effective_matches = (
            best_matches
            + min(question_matches.get(best_id, 0), best_matches))
        confidence = min(1000, (effective_matches * 1000) // len(rows))
        margin_ok = (
            len(ranked) == 1
            or best_score * 1000 >= ranked[1][0] * minimum_margin_permille)
        # 两个独立正文 n-gram 是候选准入下限；实际生产门还要求正文覆盖
        # confidence。把下限固定为 2 允许问法改写，安全性不依赖题干特征。
        minimum_matches = 2
        if (confidence < minimum_confidence_permille
                or best_matches < minimum_matches or not margin_ok):
            return ScidbCsqPassageResult(
                "UNKNOWN", None, None, None, None, None, None,
                len(rows), posting_visits, len(grouped), confidence)
        source_ref = decode_integer_tuple(bytes(best[4]))
        if len(source_ref) != 11 or any(value < 0 for value in source_ref):
            raise ScidbCsqPassageIndexError("CSQ document SourceRef 损坏")
        return ScidbCsqPassageResult(
            "ANSWER", str(best[1]), str(best[3]), str(best[5]), source_ref,
            str(best[6]), str(best[7]), len(rows), posting_visits,
            len(grouped), confidence)


__all__ = [
    "ARTIFACT_KIND",
    "ATTRIBUTION",
    "LICENSE_ID",
    "ScidbCsqPassageIndexError",
    "ScidbCsqPassageResult",
    "ScidbCsqPassageRuntime",
    "build_scidb_csq_passage_index",
]
