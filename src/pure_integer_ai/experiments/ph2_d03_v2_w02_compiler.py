"""PH2-D03-V2 W-02 正式资料的流式编译和物理 owner 隔离。"""
from __future__ import annotations

import bz2
from dataclasses import dataclass
import gzip
import hashlib
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import sqlite3
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from typing import Any, Iterator

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorResourceBudget,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2PhysicalRoots,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import validate_v2_record
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    W02_COMPILE_FREEZE_PATH,
    W02_LAYOUTS,
    W02_SPLITS,
    W02_STAGE_KEY,
    W02CompileFreeze,
    W02CompileFreezeError,
    W02CompilePlan,
    W02FileFreeze,
    build_w02_code_freeze,
    formal_w02_compile_plan,
    publish_w02_compile_freeze,
    publish_w02_first_run_guard,
    w02_candidate_contract_value,
    w02_file_freeze_commitment,
    w02_first_run_guard_value,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    MediaWikiPageError,
    MediaWikiScanBudget,
    parse_mediawiki_page,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    read_mediawiki_dump_snapshot,
)
from pure_integer_ai.experiments.ph2_source_pack_contract import (
    stable_source_pack_key,
)
from pure_integer_ai.experiments.ph2_ud_gsdsimp_adapter import (
    COMMIT_SHA1 as UD_COMMIT_SHA1,
    NODE_RANGE,
    NODE_WORD,
    REPOSITORY_URL as UD_REPOSITORY_URL,
    ConlluSentence,
    iter_ud_conllu_sentences,
)


W02_FT00_REPORT_PATH = (
    "data/ph2/manifests/d03_v2/ph2_d03_v2_ft00_release_gate_v1.json")
W02_UD_SNAPSHOT_PATH = "data/ph2/manifests/ud_zh_gsdsimp_r2_18.git_snapshot.json"
W02_WIKTIONARY_SNAPSHOT_PATH = (
    "data/ph2/manifests/zhwiktionary_20260701.multistream_snapshot.json")
W02_COMPILER_VERSION = "PH2-D03-V2-W02-COMPILER-V1"
W02_AUTHORED_GENERATOR_VERSION = "PH2-D03-V2-W02-AUTHORED-OOV-V1"
W02_EVALUATOR_VERSION = 2
W02_ROOT_DIRECTORIES = {
    "CANDIDATE_TRAIN_ROOT": "candidate-train",
    "TEACHER_TRAIN_ROOT": "teacher-train",
    "DEV_CALIBRATION_ROOT": "dev-calibration",
    "SHADOW_AUDIT_ROOT": "shadow-audit",
    "PRIVATE_EVALUATOR_ROOT": "private-evaluator",
    "EXPOSURE_LEDGER_ROOT": "exposure-ledger",
}
W02_CARRIER_KINDS = (
    "plain_text", "markdown", "html", "source_code", "mathematics",
    "table", "document_container", "quotation_embedding",
    "ocr_asr_transcript",
)
_SOURCE_IDS = {
    "AUTHORED_CC0": 1,
    "UD_ZH_GSDSIMP_R2_18": 2,
    "ZHWIKTIONARY_20260701": 3,
}
_SPLIT_IDS = {name: index for index, name in enumerate(W02_SPLITS, start=1)}
_TRAIN_NONCE = b"PH2-D03-V2-W02-PUBLIC-TRAIN-FAMILY-V1"
_AUTHORED_POOLS = (
    "龘麤鱻灥爩靐齉馫纞虋",
    "甲乙丙丁戊己庚辛壬癸",
    "qzxvkjwybcdfghmnp",
    "23456789",
)


class W02CompilerError(W02CompileFreezeError):
    """W-02 raw、record、spool 或正式发布无法满足冻结合同。"""


def _sha256_file(path: Path) -> tuple[int, str]:
    """流式计算文件大小和 SHA-256。"""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _hash_value(value: object) -> str:
    """返回规范 JSON 值摘要。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_sha(value: object, *, where: str) -> str:
    """校验小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02CompilerError(f"{where} SHA-256 非法")
    return value


def _safe_repository_file(root: Path, relative: str) -> Path:
    """解析公开仓内文件并拒绝逃逸和链接。"""
    pure = PurePosixPath(relative)
    if (pure.is_absolute() or pure.as_posix() != relative
            or any(part in {"", ".", ".."} for part in pure.parts)):
        raise W02CompilerError("W-02 公开输入路径非法")
    target = (root / Path(*pure.parts)).resolve()
    if not target.is_relative_to(root) or not target.is_file() or target.is_symlink():
        raise W02CompilerError("W-02 公开输入文件缺失或为链接")
    return target


def _key(source_key: str, kind: int, split: str, ordinal: int, *tail: int) -> StableRecordKey:
    """形成可排序的完整整数 identity，不以摘要替代记录身份。"""
    source_id = _SOURCE_IDS[source_key]
    split_id = _SPLIT_IDS[split]
    values = (2, 2, source_id, kind, split_id, ordinal, *tail)
    if ordinal <= 0 or any(type(item) is not int or item <= 0 for item in values):
        raise W02CompilerError("W-02 stable key 分量非法")
    return StableRecordKey(values)


def _dataset_key(source_key: str) -> StableRecordKey:
    """返回来源独立的 v2 dataset key。"""
    return StableRecordKey((2, 2, _SOURCE_IDS[source_key], 1))


def _artifact_key(source_key: str) -> StableRecordKey:
    """返回 W-02 来源 pack 的 artifact key。"""
    return StableRecordKey((2, 2, _SOURCE_IDS[source_key], 2))


def _teacher_owner_key() -> StableRecordKey:
    """返回 W-02 teacher record owner 的整数身份。"""
    return StableRecordKey((2, 2, 90, 1))


def _evaluator_owner_key() -> StableRecordKey:
    """返回 W-02 evaluator record owner 的整数身份。"""
    return StableRecordKey((2, 2, 90, 2))


def _dimension_key(name: str) -> StableRecordKey:
    """把预注册维度名映射为稳定整数 key。"""
    return stable_source_pack_key("ph2-d03-v2-w02-dimension", name)


def _carrier_serialization(kind: str, surface: str) -> tuple[str, int, int, str, str]:
    """为九载体生成可逆 raw serialization 和内容 span。"""
    if kind == "plain_text":
        prefix, suffix, root_kind, child_kind = "", "", "text", "text_span"
    elif kind == "markdown":
        prefix, suffix, root_kind, child_kind = "## 词项\n\n", "\n", "document", "paragraph"
    elif kind == "html":
        prefix, suffix, root_kind, child_kind = "<p>", "</p>", "element", "text_node"
    elif kind == "source_code":
        prefix, suffix, root_kind, child_kind = "term = ", "", "module", "expression"
    elif kind == "mathematics":
        prefix, suffix, root_kind, child_kind = "\\operatorname{", "}", "formula", "operator_name"
    elif kind == "table":
        prefix, suffix, root_kind, child_kind = "| 词项 |\n|---|\n| ", " |", "table", "cell"
    elif kind == "document_container":
        prefix, suffix, root_kind, child_kind = "词项: ", "\n", "document", "field"
    elif kind == "quotation_embedding":
        prefix, suffix, root_kind, child_kind = "“", "”", "quotation", "quoted_text"
    elif kind == "ocr_asr_transcript":
        prefix, suffix, root_kind, child_kind = "[片段1] ", "", "transcript", "segment"
    else:
        raise W02CompilerError("W-02 carrier kind 未注册")
    raw = prefix + surface + suffix
    return raw, len(prefix), len(prefix) + len(surface), root_kind, child_kind


def _carrier_payload(
        source_key: str,
        split: str,
        ordinal: int,
        carrier_kind: str,
        surface: str,
        *,
        source_identity: str,
        ) -> CanonicalJsonObject:
    """保留载体节点、父子边、span 和原始 serialization。"""
    raw, start, end, root_kind, child_kind = _carrier_serialization(
        carrier_kind, surface)
    root_key = _key(source_key, 80, split, ordinal, 1)
    child_key = _key(source_key, 80, split, ordinal, 2)
    return CanonicalJsonObject.from_value({
        "carrier": {
            "carrier_kind": carrier_kind,
            "edges": [{
                "attributes": {"ordered": 1},
                "edge_kind": "contains",
                "source_node_key": root_key.to_list(),
                "target_node_key": child_key.to_list(),
            }],
            "nodes": [{
                "attributes": {"source_identity": source_identity},
                "node_key": root_key.to_list(),
                "node_kind": root_kind,
                "parent_node_key": None,
                "span_end": len(raw),
                "span_start": 0,
            }, {
                "attributes": {"language_content": 1},
                "node_key": child_key.to_list(),
                "node_kind": child_kind,
                "parent_node_key": root_key.to_list(),
                "span_end": end,
                "span_start": start,
            }],
            "raw_text_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "root_node_keys": [root_key.to_list()],
        },
        "language_payload": {
            "carrier_serialization": raw,
            "content_span_end": end,
            "content_span_start": start,
            "source_identity": source_identity,
            "surface": surface,
            "surface_sha256": hashlib.sha256(surface.encode("utf-8")).hexdigest(),
        },
    })


def _source_record(
        source_key: str,
        split: str,
        ordinal: int,
        *,
        snapshot_id: str,
        revision_id: str,
        official_url: str,
        source_identity: str,
        upstream_checksum: str,
        local_sha256: str,
        license_id: str,
        attribution: str,
        locator_kind: str,
        locator_value: str,
        span_end: int,
        source_ordinal: int | None = None,
        ) -> SourceRefRecord:
    """建立带文档/实体/source cluster 的严格 v2 SourceRef。"""
    record_ordinal = ordinal if source_ordinal is None else source_ordinal
    stable = _key(source_key, 10, split, record_ordinal)
    cluster = _key(source_key, 50, split, record_ordinal)
    return SourceRefRecord(
        2, 2, 2, _dataset_key(source_key), _artifact_key(source_key), stable,
        source_key, snapshot_id, revision_id, official_url, source_identity,
        upstream_checksum, local_sha256, license_id, "PUBLIC", attribution, 2,
        CanonicalJsonObject.from_value({
            "document_cluster_key": _key(
                source_key, 51, split, record_ordinal).to_list(),
            "entity_graph_cluster_key": _key(
                source_key, 52, split, record_ordinal).to_list(),
            "locator_kind": locator_kind,
            "locator_value": locator_value,
            "span_end": max(1, span_end),
            "span_start": 0,
        }),
        _SPLIT_IDS[split] * 1_000_000 + record_ordinal,
        cluster,
    )


def _observation_record(
        source_key: str,
        split: str,
        ordinal: int,
        source: SourceRefRecord,
        *,
        carrier_kind: str,
        surface: str,
        family_ordinal: int,
        sample_role: str,
        perturbation_kind: str,
        ) -> ObservationRecord:
    """建立不携带 expected label 的 W-02 Observation。"""
    stable = _key(source_key, 20, split, ordinal)
    return ObservationRecord(
        2, 2, 2, _dataset_key(source_key), _artifact_key(source_key), stable,
        W02_STAGE_KEY, "FT01-W02-FORMAL-FOUNDATION-V1", split, "zh",
        carrier_kind, source.stable_key, source.license_id,
        _key(source_key, 60, split, family_ordinal),
        _key(source_key, 61, split, family_ordinal),
        _key(source_key, 62, split, family_ordinal),
        _key(source_key, 63, split, family_ordinal),
        "forming" if split == "train" else "evaluator",
        sample_role, "typed_carrier",
        _carrier_payload(
            source_key, split, ordinal, carrier_kind, surface,
            source_identity=source.source_identity),
        perturbation_kind, None, (),
        _SPLIT_IDS[split] * 1_000_000 + ordinal,
    )


def _owner_record(
        source_key: str,
        split: str,
        ordinal: int,
        source: SourceRefRecord,
        observation: ObservationRecord,
        expected: dict[str, Any],
        *,
        dimension_name: str,
        ) -> TeacherEvidenceRecord | EvaluatorLabelRecord:
    """train 生成冻结 Evidence，其余 split 生成 evaluator-only label。"""
    payload = CanonicalJsonObject.from_value(expected)
    if split == "train":
        return TeacherEvidenceRecord(
            2, 2, 2, _dataset_key(source_key), _artifact_key(source_key),
            _key(source_key, 30, split, ordinal), observation.stable_key,
            "W02_FORMAL_FOUNDATION_EVIDENCE_V2", payload, source.stable_key,
            W02_STAGE_KEY, 3, _teacher_owner_key(),
        )
    return EvaluatorLabelRecord(
        2, 2, 2, _dataset_key(source_key), _artifact_key(source_key),
        _key(source_key, 40, split, ordinal), observation.stable_key,
        _dimension_key(dimension_name), "TRUE", payload, 1,
        W02_EVALUATOR_VERSION, W02_STAGE_KEY, _evaluator_owner_key(),
    )


def _dimension(source_key: str, ordinal: int) -> str:
    """在来源内稳定轮转四个 bearing 与 generation 硬合取。"""
    values = (
        "W-02-V2-BOUNDARY-WITHDRAWAL",
        "W-02-V2-MULTI-CANDIDATE",
        "W-02-V2-NEW-CONTENT-MORPHOLOGY",
        "W-02-V2-OOV",
        "W-02-V2-GENERATION-HARD-CONJUNCT",
    )
    offset = _SOURCE_IDS[source_key] - 1
    return values[(ordinal + offset - 1) % len(values)]


def _ud_surface_rows(sentence: ConlluSentence) -> tuple[tuple[str, int, int], ...]:
    """按 range 覆盖和 Space gap 把 CoNLL-U 表层 token 对齐到原句。"""
    ranges = {
        row.node_id.major: row for row in sentence.rows
        if row.node_id.kind == NODE_RANGE
    }
    words = {
        row.node_id.major: row for row in sentence.rows
        if row.node_id.kind == NODE_WORD
    }
    sequence: list[str] = []
    major = 1
    while major <= len(words):
        ranged = ranges.get(major)
        if ranged is not None:
            sequence.append(ranged.form)
            major = ranged.node_id.tail + 1
        else:
            sequence.append(words[major].form)
            major += 1
    cursor = 0
    aligned: list[tuple[str, int, int]] = []
    for form in sequence:
        start = sentence.text.find(form, cursor)
        if start < 0 or sentence.text[cursor:start].strip():
            raise W02CompilerError("W-02 UD token 无法按序对齐原句")
        end = start + len(form)
        aligned.append((form, start, end))
        cursor = end
    if sentence.text[cursor:].strip():
        raise W02CompilerError("W-02 UD token 未覆盖原句尾部")
    return tuple(aligned)


def _ud_expected(sentence: ConlluSentence, carrier_kind: str) -> dict[str, Any]:
    """形成 UD 来源可追溯的边界和词形 Evidence。"""
    boundaries = _ud_surface_rows(sentence)
    words = [row for row in sentence.rows if row.node_id.kind == NODE_WORD]
    return {
        "boundary_spans": [
            {"end": end, "form": form, "start": start}
            for form, start, end in boundaries
        ],
        "carrier_kind": carrier_kind,
        "definitive_truth_authoritative": 0,
        "dimension_scope": "TOKEN_BOUNDARY_AND_ANNOTATED_MORPHOLOGY",
        "morphology": [{
            "feats": [[name, value] for name, value in row.feats],
            "form": row.form,
            "lemma": row.lemma,
            "node_id": row.node_id.to_list(),
            "upos": row.upos,
        } for row in words],
        "source_annotation": "UD_CHINESE_GSDSIMP_R2_18",
    }


def _unicode_expected(surface: str, carrier_kind: str) -> dict[str, Any]:
    """形成可逆 code point、组合类和规范化边界 Evidence。"""
    units = []
    boundaries = [0]
    for index, char in enumerate(surface):
        combining = unicodedata.combining(char)
        if index and combining == 0:
            boundaries.append(index)
        units.append({
            "category": unicodedata.category(char),
            "code_point": ord(char),
            "combining_class": combining,
            "surface": char,
        })
    boundaries.append(len(surface))
    return {
        "carrier_kind": carrier_kind,
        "code_point_units": units,
        "definitive_truth_authoritative": 0,
        "grapheme_candidate_boundaries": boundaries,
        "nfc": unicodedata.normalize("NFC", surface),
        "nfkc": unicodedata.normalize("NFKC", surface),
        "source_annotation": "UNICODE_STANDARD_LIBRARY_DETERMINISTIC",
    }


def _mediawiki_upstream_checksum(value: str) -> str:
    """将 Wikimedia base36 revision 标识规范绑定为 schema 可接受的 SHA-256。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise W02CompilerError("W-02 MediaWiki upstream revision 标识非法")
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _authored_units(nonce: bytes, split: str, family_ordinal: int) -> tuple[str, ...]:
    """由隔离 nonce 确定性生成未登录连续串的隐藏分段。"""
    digest = hashlib.sha256(
        nonce + b"\x00" + split.encode("ascii") + b"\x00"
        + family_ordinal.to_bytes(8, "big")).digest()
    unit_count = 2 + digest[0] % 3
    units: list[str] = []
    cursor = 1
    for unit_index in range(unit_count):
        pool = _AUTHORED_POOLS[digest[cursor % len(digest)] % len(_AUTHORED_POOLS)]
        cursor += 1
        length = 1 + digest[cursor % len(digest)] % 3
        cursor += 1
        chars = []
        for _ in range(length):
            chars.append(pool[digest[cursor % len(digest)] % len(pool)])
            cursor += 1
        if unit_index == unit_count - 1 and digest[cursor % len(digest)] % 4 == 0:
            chars.append("\u0301")
            cursor += 1
        units.append("".join(chars))
    units.append(f"z{family_ordinal:x}")
    return tuple(units)


def _authored_expected(
        units: tuple[str, ...],
        carrier_kind: str,
        raw: str,
        content_start: int,
        content_end: int,
        ) -> dict[str, Any]:
    """形成 OOV 分段、载体内容 span 和精确重建目标。"""
    boundaries = [0]
    for unit in units:
        boundaries.append(boundaries[-1] + len(unit))
    surface = "".join(units)
    return {
        "carrier_content_span": [content_start, content_end],
        "carrier_kind": carrier_kind,
        "definitive_truth_authoritative": 0,
        "generation_target": raw,
        "oov_boundaries": boundaries,
        "oov_units": list(units),
        "surface": surface,
    }


def _record_surface_sha(observation: ObservationRecord) -> str:
    """从已校验 Observation 取表层摘要，不暴露表层到公开 freeze。"""
    return str(observation.typed_payload.to_value()["language_payload"]["surface_sha256"])


def _sort_key(key: StableRecordKey) -> str:
    """把正整数 tuple 编为保持数值字典序的 SQLite key。"""
    return ".".join(f"{item:020d}" for item in key.components)


class _W02Spool:
    """用 SQLite 有界归并 owner 文件，避免把正式记录全量放入 Python。"""

    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(path)
        self._connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE records(
                layout_key TEXT NOT NULL,
                sort_key TEXT NOT NULL,
                stable_key TEXT NOT NULL,
                payload BLOB NOT NULL,
                license_id TEXT NOT NULL,
                PRIMARY KEY(layout_key, stable_key)
            ) WITHOUT ROWID;
            CREATE INDEX records_order ON records(layout_key, sort_key);
            CREATE TABLE global_records(
                stable_key TEXT PRIMARY KEY,
                payload BLOB NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE clusters(
                dimension TEXT NOT NULL,
                cluster_key TEXT NOT NULL,
                split TEXT NOT NULL,
                PRIMARY KEY(dimension, cluster_key)
            ) WITHOUT ROWID;
            CREATE TABLE private_cases(
                stable_key TEXT PRIMARY KEY,
                split TEXT NOT NULL,
                surface_sha256 TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE private_labels(
                stable_key TEXT PRIMARY KEY,
                split TEXT NOT NULL,
                dimension_key TEXT NOT NULL,
                expected_payload_sha256 TEXT NOT NULL,
                expected_state TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE private_clusters(
                dimension TEXT NOT NULL,
                cluster_key TEXT NOT NULL,
                split TEXT NOT NULL,
                PRIMARY KEY(dimension, cluster_key)
            ) WITHOUT ROWID;
            CREATE TABLE surfaces(
                surface_sha256 TEXT PRIMARY KEY,
                split TEXT NOT NULL,
                family_key TEXT NOT NULL
            ) WITHOUT ROWID;
        """)
        self._pending = 0

    def close(self) -> None:
        """提交并关闭 spool。"""
        self._connection.commit()
        self._connection.close()

    def _global(self, record: object) -> bytes:
        """校验 record 并拒绝稳定 key 复用为不同内容。"""
        validate_v2_record(record.to_dict())
        payload = canonical_json_line(record.to_dict())
        stable = canonical_json_bytes(record.stable_key.to_list()).decode("ascii")
        row = self._connection.execute(
            "SELECT payload FROM global_records WHERE stable_key=?", (stable,)).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO global_records(stable_key,payload) VALUES(?,?)",
                (stable, payload))
        elif bytes(row[0]) != payload:
            raise W02CompilerError("W-02 stable key 被不同 record 复用")
        return payload

    def _layout(self, layout_key: str, record: object, payload: bytes) -> None:
        """把同一 immutable record 投影到一个物理 owner 文件。"""
        stable = canonical_json_bytes(record.stable_key.to_list()).decode("ascii")
        license_id = getattr(record, "license_id", None)
        if license_id is None:
            license_id = getattr(record, "license_partition", None)
        if license_id is None:
            license_id = "CC0-1.0" if record.dataset_key == _dataset_key(
                "AUTHORED_CC0") else "CC-BY-SA-4.0"
        row = self._connection.execute(
            "SELECT payload FROM records WHERE layout_key=? AND stable_key=?",
            (layout_key, stable)).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO records VALUES(?,?,?,?,?)",
                (layout_key, _sort_key(record.stable_key), stable, payload, license_id))
        elif bytes(row[0]) != payload:
            raise W02CompilerError("W-02 owner 文件内 stable key 内容漂移")

    def _cluster(self, dimension: str, key: StableRecordKey, split: str) -> None:
        """强制任一来源/文档/内容/template/shape cluster 不跨 split。"""
        encoded = canonical_json_bytes(key.to_list()).decode("ascii")
        row = self._connection.execute(
            "SELECT split FROM clusters WHERE dimension=? AND cluster_key=?",
            (dimension, encoded)).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO clusters VALUES(?,?,?)", (dimension, encoded, split))
        elif row[0] != split:
            raise W02CompilerError("W-02 cluster 跨 split")
        if split in {"held_out", "adversarial", "wall"}:
            self._connection.execute(
                "INSERT OR IGNORE INTO private_clusters VALUES(?,?,?)",
                (dimension, encoded, split))

    def _surface(self, observation: ObservationRecord, family_key: StableRecordKey) -> None:
        """拒绝同一表层或同族被分配到不同 split。"""
        digest = _record_surface_sha(observation)
        family = canonical_json_bytes(family_key.to_list()).decode("ascii")
        row = self._connection.execute(
            "SELECT split,family_key FROM surfaces WHERE surface_sha256=?", (digest,)).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO surfaces VALUES(?,?,?)", (digest, observation.split, family))
        elif row != (observation.split, family):
            raise W02CompilerError("W-02 表层或 family 跨 split/身份复用")

    def add(
            self,
            source: SourceRefRecord,
            observation: ObservationRecord,
            owner_record: TeacherEvidenceRecord | EvaluatorLabelRecord,
            ) -> None:
        """原子登记一个 SourceRef + Observation + owner record 小闭环。"""
        source_payload = self._global(source)
        observation_payload = self._global(observation)
        owner_payload = self._global(owner_record)
        if observation.source_ref_key != source.stable_key:
            raise W02CompilerError("W-02 Observation SourceRef 引用漂移")
        if owner_record.observation_key != observation.stable_key:
            raise W02CompilerError("W-02 owner record Observation 引用漂移")
        span = source.source_span.to_value()
        clusters = (
            ("source", source.source_cluster_key),
            ("document", StableRecordKey(tuple(span["document_cluster_key"]))),
            ("entity", StableRecordKey(tuple(span["entity_graph_cluster_key"]))),
            ("dedup", observation.dedup_cluster_key),
            ("content", observation.content_group_key),
            ("template", observation.template_group_key),
            ("shape", observation.shape_group_key),
        )
        for dimension, key in clusters:
            self._cluster(dimension, key, observation.split)
        self._surface(observation, observation.content_group_key)
        split = observation.split
        if split == "train":
            if not isinstance(owner_record, TeacherEvidenceRecord):
                raise W02CompilerError("W-02 train 缺 TeacherEvidence")
            for layout in ("CANDIDATE_SOURCE", "TEACHER_SOURCE", "SHADOW_SOURCE"):
                self._layout(layout, source, source_payload)
            for layout in ("CANDIDATE_TRAIN_OBSERVATION", "SHADOW_TRAIN_OBSERVATION"):
                self._layout(layout, observation, observation_payload)
            self._layout("TEACHER_TRAIN_EVIDENCE", owner_record, owner_payload)
        elif split == "dev":
            if not isinstance(owner_record, EvaluatorLabelRecord):
                raise W02CompilerError("W-02 dev 缺 EvaluatorLabel")
            for layout in ("DEV_SOURCE", "SHADOW_SOURCE"):
                self._layout(layout, source, source_payload)
            self._layout("DEV_OBSERVATION", observation, observation_payload)
            self._layout("SHADOW_DEV_OBSERVATION", observation, observation_payload)
            self._layout("DEV_LABEL", owner_record, owner_payload)
        else:
            if not isinstance(owner_record, EvaluatorLabelRecord):
                raise W02CompilerError("W-02 private split 缺 EvaluatorLabel")
            suffix = {"held_out": "HELD_OUT", "adversarial": "ADVERSARIAL", "wall": "WALL"}[split]
            self._layout("PRIVATE_SOURCE", source, source_payload)
            self._layout(f"PRIVATE_{suffix}_OBSERVATION", observation, observation_payload)
            self._layout(f"PRIVATE_{suffix}_LABEL", owner_record, owner_payload)
            stable = canonical_json_bytes(observation.stable_key.to_list()).decode("ascii")
            self._connection.execute(
                "INSERT INTO private_cases VALUES(?,?,?)",
                (stable, split, _record_surface_sha(observation)))
            label_stable = canonical_json_bytes(owner_record.stable_key.to_list()).decode("ascii")
            dimension = canonical_json_bytes(owner_record.dimension_key.to_list()).decode("ascii")
            self._connection.execute(
                "INSERT INTO private_labels VALUES(?,?,?,?,?)",
                (label_stable, split, dimension,
                 owner_record.expected_payload.sha256(), owner_record.expected_state))
        self._pending += 1
        if self._pending >= 512:
            self._connection.commit()
            self._pending = 0

    def _rows(self, query: str, parameters: tuple[object, ...] = ()) -> list[dict[str, object]]:
        """把小型承诺索引投影为规范 object；不返回 private payload。"""
        cursor = self._connection.execute(query, parameters)
        names = [item[0] for item in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor]

    def private_commitments(self, nonce_commitment: str) -> tuple[str, str, str]:
        """只用 key、摘要和 cluster 形成 case/label/cluster 承诺。"""
        cases = self._rows(
            "SELECT stable_key,split,surface_sha256 FROM private_cases ORDER BY stable_key")
        labels = self._rows(
            "SELECT stable_key,split,dimension_key,expected_payload_sha256,expected_state "
            "FROM private_labels ORDER BY stable_key")
        clusters = self._rows(
            "SELECT dimension,cluster_key,split FROM private_clusters "
            "ORDER BY dimension,cluster_key")
        return (
            _hash_value({"nonce_commitment": nonce_commitment, "rows": cases}),
            _hash_value(labels),
            _hash_value(clusters),
        )

    def write_files(self, roots: V2PhysicalRoots) -> tuple[W02FileFreeze, ...]:
        """按 layout 顺序写 deterministic gzip 并返回双 hash 身份。"""
        root_by_key = {
            "CANDIDATE_TRAIN_ROOT": roots.candidate_train,
            "TEACHER_TRAIN_ROOT": roots.teacher_train,
            "DEV_CALIBRATION_ROOT": roots.dev_calibration,
            "SHADOW_AUDIT_ROOT": roots.shadow_audit,
            "PRIVATE_EVALUATOR_ROOT": roots.private_evaluator,
        }
        freezes: list[W02FileFreeze] = []
        self._connection.commit()
        for layout_key, (root_key, record_kind, split, relative) in W02_LAYOUTS.items():
            root = root_by_key[root_key]
            target = root / Path(*PurePosixPath(relative).parts)
            if target.exists():
                raise W02CompilerError("W-02 owner payload 禁止覆盖")
            target.parent.mkdir(parents=True, exist_ok=True)
            content_digest = hashlib.sha256()
            content_size = 0
            count = 0
            first_key: tuple[int, ...] | None = None
            last_key: tuple[int, ...] | None = None
            with target.open("xb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                    cursor = self._connection.execute(
                        "SELECT stable_key,payload FROM records "
                        "WHERE layout_key=? ORDER BY sort_key", (layout_key,))
                    for stable_text, payload_value in cursor:
                        payload = bytes(payload_value)
                        parsed = parse_canonical_json_bytes(
                            stable_text.encode("ascii"), require_object=False)
                        if not isinstance(parsed, list):
                            raise W02CompilerError("W-02 spool stable key 损坏")
                        key = tuple(parsed)
                        if last_key is not None and key <= last_key:
                            raise W02CompilerError("W-02 spool key 未严格递增")
                        if first_key is None:
                            first_key = key
                        last_key = key
                        content_digest.update(payload)
                        content_size += len(payload)
                        count += 1
                        stream.write(payload)
            if count <= 0 or first_key is None or last_key is None:
                raise W02CompilerError("W-02 owner payload 为空")
            transport_size, transport_sha = _sha256_file(target)
            licenses = tuple(
                row[0] for row in self._connection.execute(
                    "SELECT DISTINCT license_id FROM records WHERE layout_key=? "
                    "ORDER BY CASE license_id WHEN 'CC0-1.0' THEN 1 ELSE 2 END",
                    (layout_key,)))
            freezes.append(W02FileFreeze(
                layout_key, root_key, record_kind, split, count, content_size,
                content_digest.hexdigest(), transport_size, transport_sha,
                first_key, last_key, licenses,
            ))
        return tuple(freezes)


def _ud_snapshot(
        repository_root: Path,
        workspace_root: Path,
        ) -> tuple[dict[str, Any], str, dict[str, Path]]:
    """回读 UD snapshot 并逐文件核对正式 raw identity。"""
    manifest_path = _safe_repository_file(repository_root, W02_UD_SNAPSHOT_PATH)
    payload = manifest_path.read_bytes()
    if not payload.endswith(b"\n"):
        raise W02CompilerError("W-02 UD snapshot 缺规范尾换行")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if not isinstance(value, dict) or value.get("source_key") != "UD_ZH_GSDSIMP_R2_18":
        raise W02CompilerError("W-02 UD snapshot 身份漂移")
    if (value.get("commit_sha1") != UD_COMMIT_SHA1
            or value.get("license_id") != "CC-BY-SA-4.0"
            or value.get("redistribution_policy") != "PUBLIC"):
        raise W02CompilerError("W-02 UD revision/license 漂移")
    files: dict[str, Path] = {}
    raw_root = (workspace_root / "ph2_dataset_raw" / "UD_ZH_GSDSIMP_R2_18").resolve()
    if not raw_root.is_dir():
        raise W02CompilerError("W-02 UD raw root 缺失")
    for item in value.get("files", []):
        if not isinstance(item, dict) or item.get("file_kind") != "conllu":
            continue
        split = str(item.get("split"))
        relative = str(item.get("relative_path"))
        target = (raw_root / relative).resolve()
        if not target.is_relative_to(raw_root) or not target.is_file():
            raise W02CompilerError("W-02 UD raw path 缺失或逃逸")
        size, digest = _sha256_file(target)
        if size != item.get("size_bytes") or digest != item.get("local_sha256"):
            raise W02CompilerError("W-02 UD raw identity 漂移")
        files[split] = target
    if set(files) != {"train", "dev", "held_out"}:
        raise W02CompilerError("W-02 UD split 文件不完整")
    return value, hashlib.sha256(payload).hexdigest(), files


def _compile_ud(
        spool: _W02Spool,
        plan: W02CompilePlan,
        snapshot: dict[str, Any],
        files: dict[str, Path],
        ) -> dict[str, int]:
    """完整消费 UD 三 split，并形成句级边界/词形闭环。"""
    expected = next(item for item in plan.source_counts
                    if item.source_key == "UD_ZH_GSDSIMP_R2_18")
    file_meta = {
        str(item["split"]): item for item in snapshot["files"]
        if item.get("file_kind") == "conllu"
    }
    counts: dict[str, int] = {}
    text_splits: dict[str, str] = {}
    for split in ("train", "dev", "held_out"):
        count = 0
        meta = file_meta[split]
        for ordinal, sentence in enumerate(iter_ud_conllu_sentences(files[split]), start=1):
            count += 1
            surface_sha = hashlib.sha256(sentence.text.encode("utf-8")).hexdigest()
            prior = text_splits.setdefault(surface_sha, split)
            if prior != split:
                raise W02CompilerError("W-02 UD 相同句面跨 split")
            source = _source_record(
                "UD_ZH_GSDSIMP_R2_18", split, ordinal,
                snapshot_id=f"ud-zh-gsdsimp-{snapshot['tag']}",
                revision_id=str(snapshot["commit_sha1"]),
                official_url=UD_REPOSITORY_URL,
                source_identity=f"{split}:{sentence.sent_id}",
                upstream_checksum="sha1:" + str(meta["git_blob_sha1"]),
                local_sha256=str(meta["local_sha256"]),
                license_id="CC-BY-SA-4.0",
                attribution="Universal Dependencies contributors; Chinese GSDSimp r2.18",
                locator_kind="sentence", locator_value=sentence.sent_id,
                span_end=len(sentence.text),
            )
            observation = _observation_record(
                "UD_ZH_GSDSIMP_R2_18", split, ordinal, source,
                carrier_kind="plain_text", surface=sentence.text,
                family_ordinal=ordinal, sample_role=(
                    "support" if split == "train" else "read_only_probe"),
                perturbation_kind="NONE" if split == "train" else "UPSTREAM_HELD_OUT",
            )
            owner = _owner_record(
                "UD_ZH_GSDSIMP_R2_18", split, ordinal, source, observation,
                _ud_expected(sentence, "plain_text"),
                dimension_name=_dimension("UD_ZH_GSDSIMP_R2_18", ordinal),
            )
            spool.add(source, observation, owner)
        if count != expected.count(split):
            raise W02CompilerError("W-02 UD sentence count 与正式定额不一致")
        counts[split] = count
    return counts


def _local_name(tag: str) -> str:
    """移除 MediaWiki XML namespace。"""
    return tag.rsplit("}", 1)[-1]


def _wiktionary_split(title: str) -> tuple[str, int, bytes]:
    """按 coarsest shape bucket 先分簇再 split，禁止按记录随机切。"""
    normalized = unicodedata.normalize("NFC", title)
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    shape_bucket = int.from_bytes(digest[:4], "big") % 65_536 + 1
    partition = int.from_bytes(hashlib.sha256(
        shape_bucket.to_bytes(4, "big")).digest()[:8], "big") % 100
    if partition < 70:
        split = "train"
    elif partition < 80:
        split = "dev"
    elif partition < 95:
        split = "held_out"
    else:
        split = "adversarial"
    return split, shape_bucket, digest


def _compile_wiktionary(
        spool: _W02Spool,
        plan: W02CompilePlan,
        repository_root: Path,
        workspace_root: Path,
        ) -> tuple[str, dict[str, int], int]:
    """从正式 bzip2 raw 顺序扫描，按 cluster quota 编制词形/Unicode pack。"""
    manifest_path = _safe_repository_file(
        repository_root, W02_WIKTIONARY_SNAPSHOT_PATH)
    manifest_payload = manifest_path.read_bytes()
    manifest = read_mediawiki_dump_snapshot(manifest_path)
    if (manifest.source_key != "ZHWIKTIONARY_20260701"
            or manifest.license_id != "CC-BY-SA-4.0"
            or manifest.redistribution_policy != "PUBLIC"
            or manifest.release_eligible != 1):
        raise W02CompilerError("W-02 Wiktionary snapshot 未获 release 资格")
    raw_files = [item for item in manifest.raw_files if item.role == "XML"]
    if len(raw_files) != 1:
        raise W02CompilerError("W-02 Wiktionary XML raw identity 非唯一")
    raw = raw_files[0]
    raw_root = (workspace_root / "ph2_dataset_raw").resolve()
    raw_path = (raw_root / Path(*PurePosixPath(raw.raw_relative_path).parts)).resolve()
    if not raw_path.is_relative_to(raw_root) or not raw_path.is_file():
        raise W02CompilerError("W-02 Wiktionary raw 路径缺失或逃逸")
    size, digest = _sha256_file(raw_path)
    if size != raw.compressed_size_bytes or digest != raw.local_sha256:
        raise W02CompilerError("W-02 Wiktionary compressed raw identity 漂移")
    expected_row = next(item for item in plan.source_counts
                        if item.source_key == "ZHWIKTIONARY_20260701")
    quotas = {split: expected_row.count(split) for split in W02_SPLITS}
    if quotas["wall"] != 0:
        raise W02CompilerError("W-02 Wiktionary wall 必须独立于自然来源")
    counts = {split: 0 for split in W02_SPLITS}
    seen_shape_split: dict[int, str] = {}
    budget = MediaWikiScanBudget(
        max_pages=manifest.final_parser_report.page_count,
        max_xml_events=manifest.final_parser_report.xml_event_count,
        max_text_bytes_per_page=manifest.final_parser_report.max_page_text_bytes,
        max_templates_per_page=1,
        max_template_depth=1,
    )
    scanned = 0
    with bz2.open(raw_path, "rb") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if _local_name(element.tag) != "page":
                continue
            scanned += 1
            try:
                page = parse_mediawiki_page(
                    element, source_key=manifest.source_key,
                    extract_templates=False, budget=budget)
            except MediaWikiPageError as error:
                element.clear()
                if error.code == "NON_MAIN_NAMESPACE":
                    continue
                raise W02CompilerError(
                    f"W-02 Wiktionary page 解析失败: {error.code}") from error
            split, shape_bucket, title_digest = _wiktionary_split(page.title)
            prior = seen_shape_split.setdefault(shape_bucket, split)
            if prior != split:
                raise W02CompilerError("W-02 Wiktionary shape cluster 跨 split")
            if counts[split] >= quotas[split]:
                element.clear()
                continue
            ordinal = counts[split] + 1
            source = _source_record(
                "ZHWIKTIONARY_20260701", split, ordinal,
                snapshot_id=manifest.snapshot_id,
                revision_id=str(page.revision_id),
                official_url=(
                    f"https://zh.wiktionary.org/?curid={page.page_id}"
                    f"&oldid={page.revision_id}"),
                source_identity=f"page:{page.page_id}:revision:{page.revision_id}",
                upstream_checksum=_mediawiki_upstream_checksum(page.upstream_sha1),
                local_sha256=raw.local_sha256,
                license_id="CC-BY-SA-4.0",
                attribution=manifest.attribution_policy,
                locator_kind="page", locator_value=str(page.page_id),
                span_end=len(page.title),
            )
            observation = _observation_record(
                "ZHWIKTIONARY_20260701", split, ordinal, source,
                carrier_kind="plain_text", surface=page.title,
                family_ordinal=shape_bucket,
                sample_role="support" if split == "train" else "read_only_probe",
                perturbation_kind=(
                    "NONE" if split == "train" else
                    "NEW_TITLE_SHAPE" if split != "adversarial" else
                    "UNICODE_NORMALIZATION_ADVERSARIAL"),
            )
            expected = _unicode_expected(page.title, "plain_text")
            expected["page_title_sha256"] = title_digest.hex()
            owner = _owner_record(
                "ZHWIKTIONARY_20260701", split, ordinal, source, observation,
                expected,
                dimension_name=_dimension("ZHWIKTIONARY_20260701", ordinal),
            )
            spool.add(source, observation, owner)
            counts[split] += 1
            element.clear()
            if all(counts[split_name] == quotas[split_name]
                   for split_name in ("train", "dev", "held_out", "adversarial")):
                break
    if counts != quotas:
        raise W02CompilerError("W-02 Wiktionary cluster quota 未闭合")
    return hashlib.sha256(manifest_payload).hexdigest(), counts, scanned


def _compile_authored(
        spool: _W02Spool,
        plan: W02CompilePlan,
        private_nonce: bytes,
        ) -> tuple[str, dict[str, int]]:
    """用 train/private 独立 nonce 生成九载体 OOV 与边界 family。"""
    if not isinstance(private_nonce, bytes) or len(private_nonce) < 32:
        raise W02CompilerError("W-02 private nonce 至少 256 bit")
    nonce_commitment = hashlib.sha256(private_nonce).hexdigest()
    expected_row = next(item for item in plan.source_counts
                        if item.source_key == "AUTHORED_CC0")
    counts: dict[str, int] = {}
    for split in W02_SPLITS:
        quota = expected_row.count(split)
        family_count = quota // len(W02_CARRIER_KINDS)
        nonce = _TRAIN_NONCE if split == "train" else private_nonce
        observation_ordinal = 0
        for family_ordinal in range(1, family_count + 1):
            units = _authored_units(nonce, split, family_ordinal)
            surface = "".join(units)
            source_sha = hashlib.sha256(canonical_json_bytes({
                "family_ordinal": family_ordinal,
                "generator": W02_AUTHORED_GENERATOR_VERSION,
                "split": split,
                "surface_sha256": hashlib.sha256(surface.encode("utf-8")).hexdigest(),
            })).hexdigest()
            source = _source_record(
                "AUTHORED_CC0", split, family_ordinal,
                snapshot_id=W02_AUTHORED_GENERATOR_VERSION,
                revision_id="1",
                official_url="https://creativecommons.org/publicdomain/zero/1.0/",
                source_identity=f"authored-family:{_SPLIT_IDS[split]}:{family_ordinal}",
                upstream_checksum="sha256:" + source_sha,
                local_sha256=source_sha,
                license_id="CC0-1.0",
                attribution="Pure Integer AI contributors; dedicated under CC0-1.0",
                locator_kind="record", locator_value=str(family_ordinal),
                span_end=len(surface), source_ordinal=family_ordinal,
            )
            for carrier_kind in W02_CARRIER_KINDS:
                observation_ordinal += 1
                observation = _observation_record(
                    "AUTHORED_CC0", split, observation_ordinal, source,
                    carrier_kind=carrier_kind, surface=surface,
                    family_ordinal=family_ordinal,
                    sample_role=(
                        "support" if split == "train" else
                        "refute" if split == "adversarial" else
                        "read_only_probe"),
                    perturbation_kind={
                        "train": "NONE",
                        "dev": "NOVEL_TEMPLATE_FAMILY",
                        "held_out": "OOV_UNSEEN_FAMILY",
                        "adversarial": "BOUNDARY_TRAP",
                        "wall": "INDEPENDENT_WALL_OOV",
                    }[split],
                )
                raw, start, end, _, _ = _carrier_serialization(carrier_kind, surface)
                owner = _owner_record(
                    "AUTHORED_CC0", split, observation_ordinal, source, observation,
                    _authored_expected(units, carrier_kind, raw, start, end),
                    dimension_name=_dimension("AUTHORED_CC0", observation_ordinal),
                )
                spool.add(source, observation, owner)
        if observation_ordinal != quota:
            raise W02CompilerError("W-02 authored 九载体 quota 未闭合")
        counts[split] = observation_ordinal
    snapshot_commitment = _hash_value({
        "carrier_kinds": list(W02_CARRIER_KINDS),
        "generator_version": W02_AUTHORED_GENERATOR_VERSION,
        "private_nonce_commitment": nonce_commitment,
        "train_nonce_commitment": hashlib.sha256(_TRAIN_NONCE).hexdigest(),
    })
    return snapshot_commitment, counts


def _physical_roots(staging: Path) -> V2PhysicalRoots:
    """建立六个 sibling 真实目录并交由 FT00 firewall 复核。"""
    paths = {
        key: staging / name for key, name in W02_ROOT_DIRECTORIES.items()
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    return V2PhysicalRoots.from_paths(
        paths["CANDIDATE_TRAIN_ROOT"], paths["TEACHER_TRAIN_ROOT"],
        paths["DEV_CALIBRATION_ROOT"], paths["SHADOW_AUDIT_ROOT"],
        paths["PRIVATE_EVALUATOR_ROOT"], paths["EXPOSURE_LEDGER_ROOT"],
    )


def _assert_output_boundary(
        repository_root: Path,
        workspace_root: Path,
        formal_root: Path,
        ) -> None:
    """要求正式数据在公开 Git 外且不覆盖已有 family。"""
    if formal_root.exists():
        raise W02CompilerError("W-02 formal root 已存在，禁止覆盖或复用")
    if (formal_root == repository_root or formal_root.is_relative_to(repository_root)
            or repository_root.is_relative_to(formal_root)):
        raise W02CompilerError("W-02 formal root 必须位于公开 Git 外")
    if not workspace_root.is_dir() or not repository_root.is_dir():
        raise W02CompilerError("W-02 workspace/repository root 缺失")
    if not repository_root.is_relative_to(workspace_root):
        raise W02CompilerError("W-02 public repository 不在工程 workspace 内")
    public_manifest = repository_root / Path(
        *PurePosixPath(W02_COMPILE_FREEZE_PATH).parts)
    if public_manifest.exists():
        raise W02CompilerError("W-02 public freeze 已存在，禁止重编正式 family")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02CompileResult:
    """一次正式 compile/freeze 的结果，不投影 private root 内容。"""

    freeze: W02CompileFreeze
    formal_root: Path
    source_observation_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    wiktionary_pages_scanned: int


def compile_formal_w02_stage(
        repository_root: str | Path,
        workspace_root: str | Path,
        formal_root: str | Path,
        *,
        private_nonce: bytes | None = None,
        ) -> W02CompileResult:
    """编译并冻结唯一正式 W-02 pack；本函数不运行 Candidate 或 evaluator。"""
    repository = Path(repository_root).resolve()
    workspace = Path(workspace_root).resolve()
    output = Path(formal_root).resolve()
    _assert_output_boundary(repository, workspace, output)
    plan = formal_w02_compile_plan()
    nonce = private_nonce if private_nonce is not None else secrets.token_bytes(32)
    if not isinstance(nonce, bytes) or len(nonce) < 32:
        raise W02CompilerError("W-02 private nonce 非法")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    moved = False
    spool: _W02Spool | None = None
    try:
        roots = _physical_roots(staging)
        work = staging / "compiler-work"
        work.mkdir()
        spool = _W02Spool(work / "records.sqlite3")
        authored_snapshot, authored_counts = _compile_authored(spool, plan, nonce)
        ud_snapshot, ud_snapshot_sha, ud_files = _ud_snapshot(
            repository, workspace)
        ud_counts = _compile_ud(spool, plan, ud_snapshot, ud_files)
        wiktionary_snapshot_sha, wiktionary_counts, scanned = _compile_wiktionary(
            spool, plan, repository, workspace)
        nonce_commitment = hashlib.sha256(nonce).hexdigest()
        case_commitment, label_commitment, cluster_commitment = (
            spool.private_commitments(nonce_commitment))
        files = spool.write_files(roots)
        pack_commitment = w02_file_freeze_commitment(files)
        private_payload_commitment = _hash_value([
            item.to_dict() for item in files
            if item.root_key == "PRIVATE_EVALUATOR_ROOT"
        ])
        code_files, code_freeze_sha = build_w02_code_freeze(repository)
        candidate_contract = w02_candidate_contract_value(
            plan, files, code_freeze_sha256=code_freeze_sha)
        candidate_contract_sha = _hash_value(candidate_contract)
        guard_value = w02_first_run_guard_value(
            candidate_contract_sha256=candidate_contract_sha,
            code_freeze_sha256=code_freeze_sha,
            pack_commitment=pack_commitment,
        )
        guard_sha = publish_w02_first_run_guard(
            roots.candidate_train, guard_value)
        ft00_path = _safe_repository_file(repository, W02_FT00_REPORT_PATH)
        _, ft00_sha = _sha256_file(ft00_path)
        resource = V2EvaluatorResourceBudget(
            512, 9_000_000, 536_870_912, 300_000, 100_000, 4)
        freeze = W02CompileFreeze(
            ft00_sha,
            (
                ("AUTHORED_CC0", authored_snapshot),
                ("UD_ZH_GSDSIMP_R2_18", ud_snapshot_sha),
                ("ZHWIKTIONARY_20260701", wiktionary_snapshot_sha),
            ),
            plan, files, code_files, code_freeze_sha, pack_commitment,
            candidate_contract_sha, guard_sha, private_payload_commitment,
            case_commitment, label_commitment, cluster_commitment, resource,
        )
        write_immutable_json(freeze.to_dict(), staging / "freeze.public.json")
        write_immutable_json({
            "artifact_kind": "PH2_D03_V2_W02_PRIVATE_SEED_VAULT",
            "format_version": 1,
            "generator_version": W02_AUTHORED_GENERATOR_VERSION,
            "nonce_hex": nonce.hex(),
            "nonce_sha256": nonce_commitment,
            "stage_key": W02_STAGE_KEY,
            "status": "SEALED_BEFORE_CANDIDATE_RUN",
        }, roots.private_evaluator / "family-seed.vault.json")
        write_immutable_json({
            "artifact_kind": "PH2_D03_V2_W02_COMPILE_STATE",
            "compiler_version": W02_COMPILER_VERSION,
            "formal_private_evaluation_runs": 0,
            "formal_training_runs": 0,
            "format_version": 1,
            "private_payload_reads": 0,
            "source_observation_counts": {
                "AUTHORED_CC0": authored_counts,
                "UD_ZH_GSDSIMP_R2_18": ud_counts,
                "ZHWIKTIONARY_20260701": wiktionary_counts,
            },
            "stage_key": W02_STAGE_KEY,
            "status": "COMPILE_FREEZE_COMPLETE",
            "teacher_calls": 0,
            "wiktionary_pages_scanned": scanned,
        }, staging / "compile.state.json")
        spool.close()
        spool = None
        shutil.rmtree(work)
        os.replace(staging, output)
        moved = True
        publish_w02_compile_freeze(freeze, repository)
        counts = (
            ("AUTHORED_CC0", tuple((split, authored_counts[split]) for split in W02_SPLITS)),
            ("UD_ZH_GSDSIMP_R2_18", tuple((split, ud_counts.get(split, 0)) for split in W02_SPLITS)),
            ("ZHWIKTIONARY_20260701", tuple((split, wiktionary_counts[split]) for split in W02_SPLITS)),
        )
        return W02CompileResult(freeze, output, counts, scanned)
    finally:
        if spool is not None:
            spool.close()
        if not moved and staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "W02_AUTHORED_GENERATOR_VERSION",
    "W02_CARRIER_KINDS",
    "W02_COMPILER_VERSION",
    "W02_ROOT_DIRECTORIES",
    "W02CompileResult",
    "W02CompilerError",
    "compile_formal_w02_stage",
]
