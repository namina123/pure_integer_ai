"""把 ChineseSemanticKB 只读记录转换为版本化课程候选产物。

本模块只处理课程数据，不连接 Core、Memory 或正式训练 backend。关系和分类标签保持为
manifest 注入的开放整数键；词典条目只形成词形、关系候选或模式候选，不在此处裁决词义、
结构角色和关系真值。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator

from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    GLOBAL_OWNER_SCOPE,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.chinese_semantic_kb_adapter import (
    ADAPTER_VERSION as SOURCE_ADAPTER_VERSION,
    DATASET_NAME,
    PARSER_DECIMAL_TAB,
    PARSER_DOCUMENT,
    PARSER_RELATION_MARKER,
    PARSER_SURFACE_LINE,
    PARSER_SYMMETRIC_AT,
    ChineseKBRecord,
    ChineseSemanticKBAdapter,
    DatasetAnomaly,
    DatasetSourceSpan,
    PROFILES,
    parse_decimal_rational,
)
from pure_integer_ai.experiments.data_manifest import (
    ManifestIntegrityError,
    RawDatasetManifest,
    sha256_file,
)

COURSE_ADAPTER_VERSION = 1
COURSE_VERSION = 1

SPLIT_UNASSIGNED = 0
SPLIT_TRAIN = 1
SPLIT_DEV = 2
SPLIT_HELD_OUT = 3
_VALID_SPLITS = frozenset({SPLIT_TRAIN, SPLIT_DEV, SPLIT_HELD_OUT})

KIND_WORD_FORM = "word_form"
KIND_RELATION_CANDIDATE = "relation_candidate"
KIND_PATTERN_CANDIDATE = "pattern_candidate"
KIND_ANOMALY = "anomaly"
_ARTIFACT_PATHS = {
    KIND_WORD_FORM: "word_forms.jsonl.gz",
    KIND_RELATION_CANDIDATE: "relation_candidates.jsonl.gz",
    KIND_PATTERN_CANDIDATE: "pattern_candidates.jsonl.gz",
    KIND_ANOMALY: "anomalies.jsonl.gz",
}

_COURSE_HASHER = Hasher("ChineseSemanticKB.curriculum.v1")


def _strict_nonnegative(value: int, *, where: str) -> int:
    """校验协议整数并拒绝 bool 和负数。"""
    assert_int(value, _where=where)
    if type(value) is not int or value < 0:
        raise ValueError(f"{where} 必须为非负严格整数")
    return value


def _positive_hash(value: Any) -> int:
    """返回可进入 SourceRef 和课程键的稳定非零 63 位整数。"""
    result = _COURSE_HASHER.h63(value)
    return result if result > 0 else 1


def _integer_tuple(value: Any, *, where: str,
                   allow_empty: bool = False) -> tuple[int, ...]:
    """把 JSON 列表或 tuple 严格还原为整数 tuple。"""
    if not isinstance(value, (list, tuple)):
        raise ManifestIntegrityError(f"{where} 必须是整数序列")
    out = tuple(value)
    if not allow_empty and not out:
        raise ManifestIntegrityError(f"{where} 不能为空")
    for item in out:
        _strict_nonnegative(item, where=where)
    return out


def _signed_integer_tuple(value: Any, *, where: str) -> tuple[int, ...]:
    """严格还原允许负值的整数 tuple，供有理数分子等协议字段使用。"""
    if not isinstance(value, (list, tuple)) or not value:
        raise ManifestIntegrityError(f"{where} 必须是非空整数序列")
    out = tuple(value)
    assert_int(*out, _where=where)
    if any(type(item) is not int for item in out):
        raise ManifestIntegrityError(f"{where} 必须使用严格整数")
    return out


@dataclass(frozen=True, order=True)
class CourseSplitPolicy:
    """记录可重放的整数切分阈值，避免把比例藏在构建代码中。"""

    version: int
    train_permille: int
    dev_permille: int
    held_out_permille: int

    def __post_init__(self) -> None:
        for name, value in (
                ("version", self.version),
                ("train_permille", self.train_permille),
                ("dev_permille", self.dev_permille),
                ("held_out_permille", self.held_out_permille)):
            _strict_nonnegative(value, where=f"CourseSplitPolicy.{name}")
        if self.version <= 0:
            raise ValueError("CourseSplitPolicy.version 必须为正")
        if (self.train_permille + self.dev_permille
                + self.held_out_permille != 1000):
            raise ValueError("课程切分比例之和必须为 1000")

    def assign(self, dedup_cluster_id: int) -> int:
        """按稳定 cluster id 分配 train/dev/held-out，不读取行序或墙钟。"""
        _strict_nonnegative(
            dedup_cluster_id,
            where="CourseSplitPolicy.dedup_cluster_id",
        )
        bucket = dedup_cluster_id % 1000
        if bucket < self.train_permille:
            return SPLIT_TRAIN
        if bucket < self.train_permille + self.dev_permille:
            return SPLIT_DEV
        return SPLIT_HELD_OUT

    def to_dict(self) -> dict[str, int]:
        """转换为课程 manifest 的规范 JSON 对象。"""
        return {
            "dev_permille": self.dev_permille,
            "held_out_permille": self.held_out_permille,
            "train_permille": self.train_permille,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CourseSplitPolicy":
        """从课程 manifest 恢复切分协议。"""
        return cls(
            value["version"],
            value["train_permille"],
            value["dev_permille"],
            value["held_out_permille"],
        )


@dataclass(frozen=True, order=True)
class CourseArtifactManifest:
    """一个不可变 gzip JSONL 课程文件的内容清单。"""

    record_kind: str
    relative_path: str
    sha256: str
    size_bytes: int
    record_count: int

    def __post_init__(self) -> None:
        if self.record_kind not in _ARTIFACT_PATHS:
            raise ValueError("未知课程 artifact 类型")
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("课程 artifact 路径必须是安全相对路径")
        digest = self.sha256.lower()
        if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest):
            raise ValueError("课程 artifact SHA-256 非法")
        object.__setattr__(self, "sha256", digest)
        _strict_nonnegative(self.size_bytes, where="CourseArtifactManifest.size")
        _strict_nonnegative(
            self.record_count, where="CourseArtifactManifest.count")

    def to_dict(self) -> dict[str, Any]:
        """转换为课程 manifest 的规范 JSON 对象。"""
        return {
            "record_count": self.record_count,
            "record_kind": self.record_kind,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CourseArtifactManifest":
        """从课程 manifest 恢复 artifact 清单。"""
        return cls(
            str(value["record_kind"]),
            str(value["relative_path"]),
            str(value["sha256"]),
            value["size_bytes"],
            value["record_count"],
        )


@dataclass(frozen=True, order=True)
class CategoryCourseSummary:
    """单个外部类别的产物计数和真实 provenance cluster 数。"""

    category: str
    category_key: tuple[int, ...]
    provenance_cluster_count: int
    counts: tuple[tuple[str, int, int, int, int], ...]

    def __post_init__(self) -> None:
        if not self.category:
            raise ValueError("课程类别不能为空")
        if self.category != "provenance" and not self.category_key:
            raise ValueError("词典课程类别必须携带 manifest 注入键")
        _integer_tuple(
            self.category_key,
            where="CategoryCourseSummary.category_key",
            allow_empty=self.category == "provenance",
        )
        _strict_nonnegative(
            self.provenance_cluster_count,
            where="CategoryCourseSummary.provenance_cluster_count",
        )
        kinds: set[str] = set()
        for kind, unassigned, train, dev, held_out in self.counts:
            if kind not in _ARTIFACT_PATHS or kind in kinds:
                raise ValueError("类别计数包含未知或重复产物类型")
            kinds.add(kind)
            for value in (unassigned, train, dev, held_out):
                _strict_nonnegative(
                    value, where="CategoryCourseSummary.counts")
            if ((kind == KIND_ANOMALY) != (unassigned > 0)
                    and unassigned > 0):
                raise ValueError("只有异常产物允许 unassigned 计数")

    def to_dict(self) -> dict[str, Any]:
        """转换为按类别和 split 可审计的 JSON 对象。"""
        return {
            "category": self.category,
            "category_key": list(self.category_key),
            "counts": [
                {
                    "dev": dev,
                    "held_out": held_out,
                    "record_kind": kind,
                    "train": train,
                    "unassigned": unassigned,
                }
                for kind, unassigned, train, dev, held_out in self.counts
            ],
            "provenance_cluster_count": self.provenance_cluster_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CategoryCourseSummary":
        """从课程 manifest 恢复类别摘要。"""
        counts = tuple(
            (
                str(item["record_kind"]),
                item["unassigned"],
                item["train"],
                item["dev"],
                item["held_out"],
            )
            for item in value["counts"]
        )
        category = str(value["category"])
        return cls(
            category,
            _integer_tuple(
                value["category_key"],
                where="CategoryCourseSummary.category_key",
                allow_empty=category == "provenance",
            ),
            value["provenance_cluster_count"],
            counts,
        )


@dataclass(frozen=True)
class ChineseSemanticKBCurriculumManifest:
    """D-01 课程产物、源 manifest、切分策略和类别统计的冻结清单。"""

    dataset_name: str
    dataset_version: str
    source_manifest_sha256: str
    source_adapter_version: int
    source_parser_version: int
    course_adapter_version: int
    course_version: int
    license_id: str
    split_policy: CourseSplitPolicy
    artifacts: tuple[CourseArtifactManifest, ...]
    categories: tuple[CategoryCourseSummary, ...]

    def __post_init__(self) -> None:
        if self.dataset_name != DATASET_NAME or not self.dataset_version:
            raise ValueError("课程 manifest 数据集身份不匹配")
        digest = self.source_manifest_sha256.lower()
        if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest):
            raise ValueError("源 manifest SHA-256 非法")
        object.__setattr__(self, "source_manifest_sha256", digest)
        for name, value in (
                ("source_adapter_version", self.source_adapter_version),
                ("source_parser_version", self.source_parser_version),
                ("course_adapter_version", self.course_adapter_version),
                ("course_version", self.course_version)):
            _strict_nonnegative(value, where=name)
            if value <= 0:
                raise ValueError(f"{name} 必须为正")
        if not self.license_id:
            raise ValueError("课程 manifest 必须继承源许可声明")
        object.__setattr__(self, "artifacts", tuple(sorted(self.artifacts)))
        object.__setattr__(self, "categories", tuple(sorted(
            self.categories, key=lambda item: item.category)))
        kinds = [item.record_kind for item in self.artifacts]
        if set(kinds) != set(_ARTIFACT_PATHS) or len(kinds) != len(set(kinds)):
            raise ValueError("课程 manifest 必须恰好包含四类产物")
        category_names = [item.category for item in self.categories]
        expected_categories = {profile.category for profile in PROFILES}
        if (len(category_names) != len(set(category_names))
                or set(category_names) != expected_categories):
            raise ValueError("课程 manifest 类别摘要重复、缺失或多余")

    def to_dict(self) -> dict[str, Any]:
        """转换为 bit-identical JSON 所需的稳定对象。"""
        return {
            "artifacts": [item.to_dict() for item in self.artifacts],
            "categories": [item.to_dict() for item in self.categories],
            "course_adapter_version": self.course_adapter_version,
            "course_version": self.course_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "license_id": self.license_id,
            "source_adapter_version": self.source_adapter_version,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_parser_version": self.source_parser_version,
            "split_policy": self.split_policy.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        """返回 UTF-8、排序键和紧凑分隔符的规范 JSON。"""
        text = json.dumps(
            self.to_dict(), ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        )
        return (text + "\n").encode("utf-8")

    def sha256(self) -> str:
        """返回不依赖输出目录的课程 manifest 内容摘要。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def artifact(self, record_kind: str) -> CourseArtifactManifest:
        """按类型读取唯一 artifact，缺失或重复时 fail closed。"""
        matches = [item for item in self.artifacts
                   if item.record_kind == record_kind]
        if len(matches) != 1:
            raise ManifestIntegrityError("课程 artifact 不唯一或缺失")
        return matches[0]

    @classmethod
    def from_dict(cls, value: dict[str, Any]
                  ) -> "ChineseSemanticKBCurriculumManifest":
        """从已解析 JSON 恢复课程 manifest。"""
        return cls(
            str(value["dataset_name"]),
            str(value["dataset_version"]),
            str(value["source_manifest_sha256"]),
            value["source_adapter_version"],
            value["source_parser_version"],
            value["course_adapter_version"],
            value["course_version"],
            str(value["license_id"]),
            CourseSplitPolicy.from_dict(value["split_policy"]),
            tuple(CourseArtifactManifest.from_dict(item)
                  for item in value["artifacts"]),
            tuple(CategoryCourseSummary.from_dict(item)
                  for item in value["categories"]),
        )


@dataclass(frozen=True, order=True)
class WordFormCourseItem:
    """可见词形候选；surface 只构造 Representation，不等于语言原子。"""

    category: str
    category_key: tuple[int, ...]
    surface: str
    codepoints: tuple[int, ...]
    split: int
    dedup_cluster_id: int
    provenance_cluster_id: int
    source_ref: SourceRef
    span: DatasetSourceSpan


@dataclass(frozen=True, order=True)
class RelationCourseCandidate:
    """带来源的二元外部关系候选，不是已确认图关系。"""

    category: str
    category_key: tuple[int, ...]
    fields: tuple[str, str]
    split: int
    dedup_cluster_id: int
    provenance_cluster_id: int
    source_ref: SourceRef
    span: DatasetSourceSpan


@dataclass(frozen=True, order=True)
class PatternCourseCandidate:
    """表层模式候选；整句、A-不-A 和 stoplist 项均不冒充单词或动作。"""

    category: str
    category_key: tuple[int, ...]
    fields: tuple[str, ...]
    rational: tuple[int, int] | None
    split: int
    dedup_cluster_id: int
    provenance_cluster_id: int
    source_ref: SourceRef
    span: DatasetSourceSpan


@dataclass(frozen=True, order=True)
class CourseAnomaly:
    """保留来源定位和原始字节哈希的隔离事件。"""

    category: str
    category_key: tuple[int, ...]
    kind: str
    raw_sha256: str
    detail: str
    provenance_cluster_id: int
    source_ref: SourceRef
    span: DatasetSourceSpan


class _GzipJsonlWriter:
    """以固定 gzip header 和规范 JSON 写 bit-identical 流。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.record_count = 0
        self._raw = path.open("xb")
        self._gzip = gzip.GzipFile(
            filename="", mode="wb", fileobj=self._raw, mtime=0)

    def write(self, value: dict[str, Any]) -> None:
        """写入一条排序键紧凑 JSON，并累计记录数。"""
        payload = json.dumps(
            value, ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        self._gzip.write(payload)
        self.record_count += 1

    def close(self) -> None:
        """完成 gzip trailer 并关闭底层独占文件。"""
        self._gzip.close()
        self._raw.close()


def _source_ref_key(source_ref: SourceRef) -> list[int]:
    """把 SourceRef 编码为完整稳定整数键。"""
    return list(source_ref.stable_key())


def _source_ref_from_key(value: Any) -> SourceRef:
    """从课程行恢复 SourceRef，并拒绝截断或尾随字段。"""
    key = _integer_tuple(value, where="course.source_ref")
    if len(key) != 11:
        raise ManifestIntegrityError("课程 SourceRef 稳定键长度非法")
    owner = OwnerScope(key[3], key[4], key[5], key[6])
    versions = VersionBundle(
        CorpusVersion(key[7]),
        ParserVersion(key[8]),
        PrimitiveVersion(key[9]),
        CurriculumVersion(key[10]),
    )
    return SourceRef(key[0], key[1], key[2], owner, versions)


def _span_value(span: DatasetSourceSpan) -> list[Any]:
    """把来源 span 编码为紧凑稳定列表。"""
    return [
        span.relative_path,
        span.line_number,
        span.byte_start,
        span.byte_end,
    ]


def _span_from_value(value: Any) -> DatasetSourceSpan:
    """从课程行严格恢复来源 span。"""
    if not isinstance(value, list) or len(value) != 4:
        raise ManifestIntegrityError("课程 span 格式非法")
    return DatasetSourceSpan(str(value[0]), value[1], value[2], value[3])


def _row_base(category: str, category_key: tuple[int, ...], split: int,
              dedup_cluster_id: int, provenance_cluster_id: int,
              source_ref: SourceRef,
              span: DatasetSourceSpan) -> dict[str, Any]:
    """构造所有课程候选共享的来源、cluster 和 split 字段。"""
    return {
        "category": category,
        "category_key": list(category_key),
        "dedup_cluster_id": dedup_cluster_id,
        "provenance_cluster_id": provenance_cluster_id,
        "source_ref": _source_ref_key(source_ref),
        "span": _span_value(span),
        "split": split,
    }


class _CourseBuilder:
    """单遍消费 adapter 事件并写四类 D-01 课程产物。"""

    def __init__(self, source_manifest: RawDatasetManifest,
                 raw_root: str | Path, split_policy: CourseSplitPolicy) -> None:
        self.source_manifest = source_manifest
        self.adapter = ChineseSemanticKBAdapter(source_manifest, raw_root)
        self.raw_root = Path(raw_root).resolve()
        self.split_policy = split_policy
        self._profiles = {
            profile.relative_path: profile for profile in PROFILES
        }
        self._items = {
            item.relative_path: item for item in source_manifest.files
        }
        self._source_manifest_sha256 = source_manifest.sha256()
        self._source_kind = self._single_binding("dataset_source_kind")
        self._category_keys = {
            profile.category: source_manifest.binding(
                f"category:{profile.category}")
            for profile in PROFILES
            if profile.parser_kind != PARSER_DOCUMENT
        }
        self._versions = VersionBundle(
            CorpusVersion(_positive_hash((
                "corpus", self._source_manifest_sha256))),
            ParserVersion(source_manifest.parser_version),
            PrimitiveVersion(0),
            CurriculumVersion(COURSE_VERSION),
        )
        self._source_ids = {
            path: _positive_hash((
                "provenance",
                source_manifest.dataset_name,
                source_manifest.dataset_version,
                self._source_manifest_sha256,
                path,
                item.sha256,
            ))
            for path, item in self._items.items()
        }
        self._seen_word_forms: set[tuple[str, str]] = set()
        self._counts: dict[str, dict[str, dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int)))
        self._provenance_clusters: dict[str, set[int]] = defaultdict(set)

    def build(self, output_dir: str | Path
              ) -> ChineseSemanticKBCurriculumManifest:
        """在 raw root 外原子生成新课程目录，拒绝覆盖任何既有版本。"""
        output = Path(output_dir).resolve()
        if output.is_relative_to(self.raw_root):
            raise ManifestIntegrityError("课程产物不得写入原始数据目录")
        if output.exists():
            raise ManifestIntegrityError("课程版本目录已存在，禁止覆盖")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(
            prefix=f".{output.name}.building-", dir=output.parent))
        try:
            manifest = self._write_staging(staging)
            os.replace(staging, output)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _write_staging(
            self, staging: Path) -> ChineseSemanticKBCurriculumManifest:
        """写完四类 artifact 后计算哈希，并最后提交课程 manifest。"""
        writers = {
            kind: _GzipJsonlWriter(staging / relative_path)
            for kind, relative_path in _ARTIFACT_PATHS.items()
        }
        try:
            for event in self.adapter.iter_events():
                if isinstance(event, ChineseKBRecord):
                    self._consume_record(event, writers)
                else:
                    self._consume_anomaly(event, writers[KIND_ANOMALY])
        finally:
            for writer in writers.values():
                writer.close()

        artifacts = tuple(
            CourseArtifactManifest(
                kind,
                _ARTIFACT_PATHS[kind],
                sha256_file(writer.path),
                writer.path.stat().st_size,
                writer.record_count,
            )
            for kind, writer in sorted(writers.items())
        )
        manifest = ChineseSemanticKBCurriculumManifest(
            self.source_manifest.dataset_name,
            self.source_manifest.dataset_version,
            self._source_manifest_sha256,
            self.source_manifest.adapter_version,
            self.source_manifest.parser_version,
            COURSE_ADAPTER_VERSION,
            COURSE_VERSION,
            self.source_manifest.license_id,
            self.split_policy,
            artifacts,
            self._summaries(),
        )
        (staging / "manifest.json").write_bytes(manifest.canonical_bytes())
        return manifest

    def _consume_record(
            self, record: ChineseKBRecord,
            writers: dict[str, _GzipJsonlWriter]) -> None:
        """把一条有效来源记录路由为词形及关系或模式候选。"""
        if record.parser_kind == PARSER_DOCUMENT:
            return
        category_key = self._category_keys[record.category]
        source_ref, provenance_cluster_id = self._provenance(record.span)
        surfaces = (record.fields[:1]
                    if record.parser_kind == PARSER_DECIMAL_TAB
                    else record.fields)
        for surface in surfaces:
            self._write_word_form(
                record.category,
                category_key,
                surface,
                source_ref,
                provenance_cluster_id,
                record.span,
                writers[KIND_WORD_FORM],
            )
        if record.parser_kind in {
                PARSER_RELATION_MARKER, PARSER_SYMMETRIC_AT}:
            self._write_relation(
                record,
                category_key,
                source_ref,
                provenance_cluster_id,
                writers[KIND_RELATION_CANDIDATE],
            )
        elif record.parser_kind in {
                PARSER_DECIMAL_TAB, PARSER_SURFACE_LINE}:
            self._write_pattern(
                record,
                category_key,
                source_ref,
                provenance_cluster_id,
                writers[KIND_PATTERN_CANDIDATE],
            )
        else:
            raise ManifestIntegrityError(
                f"D-01 未支持 parser_kind: {record.parser_kind}")

    def _write_word_form(
            self, category: str, category_key: tuple[int, ...], surface: str,
            source_ref: SourceRef, provenance_cluster_id: int,
            span: DatasetSourceSpan, writer: _GzipJsonlWriter) -> None:
        """每类别每 surface 只写一次词形候选，并保留首次来源。"""
        seen_key = (category, surface)
        if seen_key in self._seen_word_forms:
            return
        self._seen_word_forms.add(seen_key)
        dedup_cluster_id = _positive_hash(("word_form", surface))
        split = self.split_policy.assign(dedup_cluster_id)
        row = _row_base(
            category, category_key, split, dedup_cluster_id,
            provenance_cluster_id, source_ref, span,
        )
        row.update({
            "codepoints": [ord(character) for character in surface],
            "surface": surface,
        })
        writer.write(row)
        self._count(category, KIND_WORD_FORM, split, provenance_cluster_id)

    def _write_relation(
            self, record: ChineseKBRecord, category_key: tuple[int, ...],
            source_ref: SourceRef, provenance_cluster_id: int,
            writer: _GzipJsonlWriter) -> None:
        """写带来源的二元关系候选，不把 category key 当作已确认 predicate。"""
        if len(record.fields) != 2:
            raise ManifestIntegrityError("关系候选必须恰有两个 surface")
        profile = self._profiles[record.span.relative_path]
        canonical_fields = (tuple(sorted(record.fields))
                            if profile.symmetric else record.fields)
        dedup_cluster_id = _positive_hash((
            "relation", category_key, canonical_fields))
        split = self.split_policy.assign(dedup_cluster_id)
        row = _row_base(
            record.category, category_key, split, dedup_cluster_id,
            provenance_cluster_id, source_ref, record.span,
        )
        row["fields"] = list(record.fields)
        writer.write(row)
        self._count(
            record.category, KIND_RELATION_CANDIDATE,
            split, provenance_cluster_id,
        )

    def _write_pattern(
            self, record: ChineseKBRecord, category_key: tuple[int, ...],
            source_ref: SourceRef, provenance_cluster_id: int,
            writer: _GzipJsonlWriter) -> None:
        """写表层模式候选；程度值仅保存精确有理数，不执行类别语义。"""
        rational = (parse_decimal_rational(record.fields[1])
                    if record.parser_kind == PARSER_DECIMAL_TAB else None)
        fields = record.fields[:1]
        dedup_cluster_id = _positive_hash((
            "pattern", category_key, fields, rational))
        split = self.split_policy.assign(dedup_cluster_id)
        row = _row_base(
            record.category, category_key, split, dedup_cluster_id,
            provenance_cluster_id, source_ref, record.span,
        )
        row.update({
            "fields": list(fields),
            "rational": list(rational) if rational is not None else None,
        })
        writer.write(row)
        self._count(
            record.category, KIND_PATTERN_CANDIDATE,
            split, provenance_cluster_id,
        )

    def _consume_anomaly(
            self, anomaly: DatasetAnomaly,
            writer: _GzipJsonlWriter) -> None:
        """把所有 D-00 异常写入隔离表，不转成词形或训练候选。"""
        profile = self._profiles[anomaly.span.relative_path]
        category_key = self._category_keys.get(profile.category, ())
        source_ref, provenance_cluster_id = self._provenance(anomaly.span)
        writer.write({
            "category": profile.category,
            "category_key": list(category_key),
            "detail": anomaly.detail,
            "kind": anomaly.kind,
            "provenance_cluster_id": provenance_cluster_id,
            "raw_sha256": anomaly.raw_sha256,
            "source_ref": _source_ref_key(source_ref),
            "span": _span_value(anomaly.span),
        })
        self._count(
            profile.category, KIND_ANOMALY,
            SPLIT_UNASSIGNED, provenance_cluster_id,
        )

    def _provenance(
            self, span: DatasetSourceSpan) -> tuple[SourceRef, int]:
        """把数据集文件作为真实来源簇，行号只作为 document 定位。"""
        source_id = self._source_ids[span.relative_path]
        return (
            SourceRef(
                self._source_kind,
                source_id,
                span.line_number,
                GLOBAL_OWNER_SCOPE,
                self._versions,
            ),
            source_id,
        )

    def _count(self, category: str, kind: str, split: int,
               provenance_cluster_id: int) -> None:
        """累计 manifest 所需的类别/split 数和真实来源簇集合。"""
        self._counts[category][kind][split] += 1
        self._provenance_clusters[category].add(provenance_cluster_id)

    def _summaries(self) -> tuple[CategoryCourseSummary, ...]:
        """生成所有 profile 的稳定摘要，零计数类别也显式保留。"""
        summaries: list[CategoryCourseSummary] = []
        for profile in sorted(PROFILES, key=lambda item: item.category):
            category = profile.category
            counts = []
            for kind in sorted(_ARTIFACT_PATHS):
                by_split = self._counts[category][kind]
                if kind == KIND_ANOMALY:
                    total = by_split[SPLIT_UNASSIGNED]
                    counts.append((kind, total, 0, 0, 0))
                else:
                    counts.append((
                        kind,
                        0,
                        by_split[SPLIT_TRAIN],
                        by_split[SPLIT_DEV],
                        by_split[SPLIT_HELD_OUT],
                    ))
            summaries.append(CategoryCourseSummary(
                category,
                self._category_keys.get(category, ()),
                len(self._provenance_clusters[category]),
                tuple(counts),
            ))
        return tuple(summaries)

    def _single_binding(self, name: str) -> int:
        """读取要求为单整数的 manifest 绑定，避免静默截断开放键。"""
        values = self.source_manifest.binding(name)
        if len(values) != 1 or values[0] <= 0:
            raise ManifestIntegrityError(f"manifest 绑定 {name!r} 必须是单个正整数")
        return values[0]


def build_curriculum_artifacts(
        source_manifest: RawDatasetManifest,
        raw_root: str | Path,
        output_dir: str | Path,
        *,
        split_policy: CourseSplitPolicy,
        ) -> ChineseSemanticKBCurriculumManifest:
    """从已核验 D-00C manifest 原子生成 D-01 课程目录。"""
    if source_manifest.dataset_name != DATASET_NAME:
        raise ManifestIntegrityError("D-01 源 manifest 不是 ChineseSemanticKB")
    if source_manifest.adapter_version != SOURCE_ADAPTER_VERSION:
        raise ManifestIntegrityError("D-01 源 adapter version 不匹配")
    return _CourseBuilder(
        source_manifest, raw_root, split_policy).build(output_dir)


def read_curriculum_manifest(
        path: str | Path) -> ChineseSemanticKBCurriculumManifest:
    """严格读取课程 manifest，并拒绝非对象 JSON 根。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ManifestIntegrityError("课程 manifest 根必须是 JSON 对象")
    return ChineseSemanticKBCurriculumManifest.from_dict(value)


def verify_curriculum_artifacts(
        manifest: ChineseSemanticKBCurriculumManifest,
        course_root: str | Path) -> None:
    """逐文件核验课程 artifact 的路径、大小和 SHA-256。"""
    root = Path(course_root).resolve()
    for item in manifest.artifacts:
        path = (root / item.relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ManifestIntegrityError(
                f"课程 artifact 缺失或越界: {item.relative_path}")
        if path.stat().st_size != item.size_bytes:
            raise ManifestIntegrityError(
                f"课程 artifact 大小变化: {item.relative_path}")
        if sha256_file(path) != item.sha256:
            raise ManifestIntegrityError(
                f"课程 artifact 哈希变化: {item.relative_path}")


def _verify_artifact_item(root: Path, item: CourseArtifactManifest) -> Path:
    """核验单个 artifact 的安全路径、大小和内容摘要并返回绝对路径。"""
    path = (root / item.relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ManifestIntegrityError(
            f"课程 artifact 缺失或越界: {item.relative_path}")
    if path.stat().st_size != item.size_bytes:
        raise ManifestIntegrityError(
            f"课程 artifact 大小变化: {item.relative_path}")
    if sha256_file(path) != item.sha256:
        raise ManifestIntegrityError(
            f"课程 artifact 哈希变化: {item.relative_path}")
    return path


def _read_jsonl_rows(
        root: Path,
        item: CourseArtifactManifest) -> Iterator[dict[str, Any]]:
    """在前后完整性核验之间严格流式读取 UTF-8 gzip JSONL。"""
    path = _verify_artifact_item(root, item)
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="strict",
                       newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ManifestIntegrityError(
                        f"课程 JSONL 第 {line_number} 行损坏") from error
                if not isinstance(value, dict):
                    raise ManifestIntegrityError("课程 JSONL 行必须是 JSON 对象")
                yield value
    except (OSError, UnicodeError) as error:
        raise ManifestIntegrityError("课程 gzip/UTF-8 内容损坏") from error
    finally:
        _verify_artifact_item(root, item)


def _parse_shared_row(value: dict[str, Any]
                      ) -> tuple[str, tuple[int, ...], int, int, int,
                                 SourceRef, DatasetSourceSpan]:
    """解析词形、关系和模式候选共用字段。"""
    category = str(value["category"])
    category_key = _integer_tuple(
        value["category_key"], where="course.category_key")
    split = value["split"]
    if type(split) is not int or split not in _VALID_SPLITS:
        raise ManifestIntegrityError("课程候选 split 非法")
    dedup_cluster_id = _strict_nonnegative(
        value["dedup_cluster_id"], where="course.dedup_cluster_id")
    provenance_cluster_id = _strict_nonnegative(
        value["provenance_cluster_id"],
        where="course.provenance_cluster_id",
    )
    return (
        category,
        category_key,
        split,
        dedup_cluster_id,
        provenance_cluster_id,
        _source_ref_from_key(value["source_ref"]),
        _span_from_value(value["span"]),
    )


class ChineseSemanticKBCurriculum:
    """核验 D-01 产物后按类型、类别和 split 流式读取课程项。"""

    def __init__(self, manifest: ChineseSemanticKBCurriculumManifest,
                 course_root: str | Path) -> None:
        if manifest.dataset_name != DATASET_NAME:
            raise ManifestIntegrityError("课程 manifest 数据集不匹配")
        if manifest.course_adapter_version != COURSE_ADAPTER_VERSION:
            raise ManifestIntegrityError("课程 adapter version 不匹配")
        if manifest.course_version != COURSE_VERSION:
            raise ManifestIntegrityError("课程 version 不匹配")
        verify_curriculum_artifacts(manifest, course_root)
        self.manifest = manifest
        self.course_root = Path(course_root).resolve()

    def iter_word_forms(
            self, *, category: str | None = None,
            split: int | None = None) -> Iterator[WordFormCourseItem]:
        """流式读取词形候选，并按可选类别和 split 过滤。"""
        for value in self._rows(KIND_WORD_FORM):
            shared = _parse_shared_row(value)
            if not self._matches(shared[0], shared[2], category, split):
                continue
            surface = str(value["surface"])
            codepoints = _integer_tuple(
                value["codepoints"], where="word_form.codepoints")
            if tuple(ord(character) for character in surface) != codepoints:
                raise ManifestIntegrityError("词形 surface 与 codepoints 不一致")
            yield WordFormCourseItem(
                shared[0], shared[1], surface, codepoints,
                shared[2], shared[3], shared[4], shared[5], shared[6],
            )

    def iter_relation_candidates(
            self, *, category: str | None = None,
            split: int | None = None) -> Iterator[RelationCourseCandidate]:
        """流式读取二元关系候选，不把候选提升为图事实。"""
        for value in self._rows(KIND_RELATION_CANDIDATE):
            shared = _parse_shared_row(value)
            if not self._matches(shared[0], shared[2], category, split):
                continue
            fields = value["fields"]
            if (not isinstance(fields, list) or len(fields) != 2
                    or any(not isinstance(field, str) or not field
                           for field in fields)):
                raise ManifestIntegrityError("关系候选 fields 非法")
            yield RelationCourseCandidate(
                shared[0], shared[1], (fields[0], fields[1]),
                shared[2], shared[3], shared[4], shared[5], shared[6],
            )

    def iter_pattern_candidates(
            self, *, category: str | None = None,
            split: int | None = None) -> Iterator[PatternCourseCandidate]:
        """流式读取表层模式候选和可选精确有理数负载。"""
        for value in self._rows(KIND_PATTERN_CANDIDATE):
            shared = _parse_shared_row(value)
            if not self._matches(shared[0], shared[2], category, split):
                continue
            raw_fields = value["fields"]
            if (not isinstance(raw_fields, list) or not raw_fields
                    or any(not isinstance(field, str) or not field
                           for field in raw_fields)):
                raise ManifestIntegrityError("模式候选 fields 非法")
            raw_rational = value["rational"]
            rational = None
            if raw_rational is not None:
                rational_values = _signed_integer_tuple(
                    raw_rational, where="pattern.rational")
                if len(rational_values) != 2 or rational_values[1] <= 0:
                    raise ManifestIntegrityError("模式候选有理数非法")
                rational = (rational_values[0], rational_values[1])
            yield PatternCourseCandidate(
                shared[0], shared[1], tuple(raw_fields), rational,
                shared[2], shared[3], shared[4], shared[5], shared[6],
            )

    def iter_anomalies(
            self, *, category: str | None = None) -> Iterator[CourseAnomaly]:
        """流式读取隔离异常；异常永不进入词形或候选迭代器。"""
        for value in self._rows(KIND_ANOMALY):
            row_category = str(value["category"])
            if category is not None and row_category != category:
                continue
            category_key = _integer_tuple(
                value["category_key"],
                where="anomaly.category_key",
                allow_empty=row_category == "provenance",
            )
            yield CourseAnomaly(
                row_category,
                category_key,
                str(value["kind"]),
                str(value["raw_sha256"]),
                str(value["detail"]),
                _strict_nonnegative(
                    value["provenance_cluster_id"],
                    where="anomaly.provenance_cluster_id",
                ),
                _source_ref_from_key(value["source_ref"]),
                _span_from_value(value["span"]),
            )

    def verify_record_counts(self) -> None:
        """完整解压四类产物并核对 manifest 记录数。"""
        for item in self.manifest.artifacts:
            count = sum(1 for _ in self._rows(item.record_kind))
            if count != item.record_count:
                raise ManifestIntegrityError(
                    f"课程 artifact 记录数变化: {item.relative_path}")

    def _rows(self, record_kind: str) -> Iterator[dict[str, Any]]:
        """在每次流读取前后核验指定 artifact。"""
        item = self.manifest.artifact(record_kind)
        yield from _read_jsonl_rows(self.course_root, item)

    @staticmethod
    def _matches(row_category: str, row_split: int,
                 category: str | None, split: int | None) -> bool:
        """执行不改变产物顺序的类别和 split 过滤。"""
        if split is not None and split not in _VALID_SPLITS:
            raise ValueError("课程 split 过滤值非法")
        return ((category is None or row_category == category)
                and (split is None or row_split == split))


__all__ = [
    "COURSE_ADAPTER_VERSION",
    "COURSE_VERSION",
    "CategoryCourseSummary",
    "ChineseSemanticKBCurriculum",
    "ChineseSemanticKBCurriculumManifest",
    "CourseAnomaly",
    "CourseArtifactManifest",
    "CourseSplitPolicy",
    "KIND_ANOMALY",
    "KIND_PATTERN_CANDIDATE",
    "KIND_RELATION_CANDIDATE",
    "KIND_WORD_FORM",
    "PatternCourseCandidate",
    "RelationCourseCandidate",
    "SPLIT_DEV",
    "SPLIT_HELD_OUT",
    "SPLIT_TRAIN",
    "WordFormCourseItem",
    "build_curriculum_artifacts",
    "read_curriculum_manifest",
    "verify_curriculum_artifacts",
]
