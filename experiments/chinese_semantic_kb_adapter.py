"""ChineseSemanticKB 原始快照的只读扫描、manifest 和记录适配。"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from math import gcd
from pathlib import Path
from typing import Iterator

from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.data_manifest import (
    ManifestBinding,
    ManifestIntegrityError,
    RawDatasetManifest,
    RawFileManifest,
    open_verified_binary,
    sha256_file,
    verify_manifest,
)

DATASET_NAME = "ChineseSemanticKB"
ADAPTER_VERSION = 1
PARSER_VERSION = 3
LICENSE_ID = "NOASSERTION-README-SHARING-NOT-RECOMMENDED"

PARSER_DOCUMENT = "document"
PARSER_RELATION_MARKER = "relation_marker"
PARSER_SYMMETRIC_AT = "symmetric_at"
PARSER_DECIMAL_TAB = "decimal_tab"
PARSER_SURFACE_LINE = "surface_line"

_BINDING_HASHER = Hasher("ChineseSemanticKB.manifest.binding.v1")


@dataclass(frozen=True, order=True)
class ChineseKBProfile:
    """一个参与文件的外部格式，不承载核心语义或语言规则。"""

    relative_path: str
    parser_kind: str
    category: str
    relation_marker: str = ""
    symmetric: bool = False


PROFILES: tuple[ChineseKBProfile, ...] = (
    ChineseKBProfile("README.md", PARSER_DOCUMENT, "provenance"),
    ChineseKBProfile(
        "dict/程度副词.txt", PARSER_DECIMAL_TAB, "degree_adverb"),
    ChineseKBProfile(
        "dict/抽象关系库.txt", PARSER_RELATION_MARKER,
        "abstraction", "抽象"),
    ChineseKBProfile(
        "dict/反义关系库.txt", PARSER_SYMMETRIC_AT,
        "antonym", symmetric=True),
    ChineseKBProfile(
        "dict/否定词.txt", PARSER_SURFACE_LINE, "negation"),
    ChineseKBProfile(
        "dict/简称关系库.txt", PARSER_RELATION_MARKER,
        "abbreviation", "简称"),
    ChineseKBProfile(
        "dict/节日时间词.txt", PARSER_SURFACE_LINE, "festival_time"),
    ChineseKBProfile(
        "dict/量比词.txt", PARSER_SURFACE_LINE, "ratio_term"),
    ChineseKBProfile(
        "dict/情态词.txt", PARSER_SURFACE_LINE, "modality"),
    ChineseKBProfile(
        "dict/数量介词.txt", PARSER_SURFACE_LINE, "quantity_preposition"),
    ChineseKBProfile(
        "dict/停用词.txt", PARSER_SURFACE_LINE, "stoplist"),
    ChineseKBProfile(
        "dict/同义关系库.txt", PARSER_RELATION_MARKER,
        "synonym", "同义", symmetric=True),
    ChineseKBProfile(
        "dict/修饰副词.txt", PARSER_SURFACE_LINE, "modifier_adverb"),
)


@dataclass(frozen=True, order=True)
class DatasetSourceSpan:
    """原始记录的可回溯文件、行号和半开 byte span。"""

    relative_path: str
    line_number: int
    byte_start: int
    byte_end: int

    def __post_init__(self) -> None:
        assert_int(
            self.line_number,
            self.byte_start,
            self.byte_end,
            _where="DatasetSourceSpan",
        )
        if self.line_number <= 0 or self.byte_start < 0:
            raise ValueError("来源行号必须为正且 byte 起点不得为负")
        if self.byte_end <= self.byte_start:
            raise ValueError("来源 byte span 必须非空")


@dataclass(frozen=True, order=True)
class ChineseKBRecord:
    """通过格式校验且去重后的外部记录。"""

    category: str
    parser_kind: str
    fields: tuple[str, ...]
    span: DatasetSourceSpan


@dataclass(frozen=True, order=True)
class DatasetAnomaly:
    """被确定性隔离的坏行、重复键或规范化警告。"""

    kind: str
    span: DatasetSourceSpan
    raw_sha256: str
    detail: str = ""


@dataclass(frozen=True)
class FileScanReport:
    """单文件可重放的记录数、异常数和异常明细。"""

    relative_path: str
    record_count: int
    anomalies: tuple[DatasetAnomaly, ...]

    @property
    def anomaly_count(self) -> int:
        """返回异常事件总数，同一行可有规范化警告和有效记录。"""
        return len(self.anomalies)


def parse_decimal_rational(value: str) -> tuple[int, int]:
    """把十进制文本精确转换为最简有理数，禁止经过 float。"""
    if not isinstance(value, str):
        raise TypeError("十进制输入必须是字符串")
    text = value.strip()
    if not text:
        raise ValueError("十进制输入不能为空")
    sign = 1
    if text[0] in "+-":
        if text[0] == "-":
            sign = -1
        text = text[1:]
    if not text:
        raise ValueError("十进制输入缺少数字")
    if text.count(".") > 1:
        raise ValueError("十进制输入含多个小数点")
    whole, separator, fraction = text.partition(".")
    if not whole:
        whole = "0"
    if not whole.isdecimal() or (separator and not fraction.isdecimal()):
        raise ValueError("十进制输入含非法字符")
    denominator = 10 ** len(fraction) if separator else 1
    numerator = sign * (int(whole) * denominator + (
        int(fraction) if fraction else 0))
    divisor = gcd(abs(numerator), denominator)
    return numerator // divisor, denominator // divisor


def _binding(label: str) -> tuple[int, ...]:
    """把 adapter 边界的开放标签压成稳定非零整数键。"""
    value = _BINDING_HASHER.h63(label)
    return (value if value != 0 else 1,)


def manifest_bindings(
        unicode_sequence_family: tuple[int, ...]) -> tuple[ManifestBinding, ...]:
    """构造 D-01/L-01 所需的开放整数键，并复用 UCD 表示族。"""
    if not isinstance(unicode_sequence_family, tuple) or not unicode_sequence_family:
        raise ValueError("unicode_sequence_family 必须是非空整数元组")
    assert_int(
        *unicode_sequence_family,
        _where="ChineseSemanticKB.unicode_sequence_family",
    )
    bindings = [
        ManifestBinding("unicode_sequence_family", unicode_sequence_family),
        ManifestBinding("language_branch", _binding("language:zh")),
        ManifestBinding("dataset_source_kind", _binding("dataset_source")),
        ManifestBinding("word_inventory_relation", _binding("word_inventory")),
        ManifestBinding("atom_representation_relation", _binding("atom_representation")),
        ManifestBinding("atom_sense_relation", _binding("atom_sense")),
        ManifestBinding("sense_concept_relation", _binding("sense_concept")),
        ManifestBinding("course_scope_kind", _binding("course_scope")),
        ManifestBinding("provenance_kind", _binding("provenance")),
        ManifestBinding("epistemic_origin", _binding("external_lexical_source")),
    ]
    bindings.extend(
        ManifestBinding(f"category:{profile.category}", _binding(
            f"category:{profile.category}"))
        for profile in PROFILES if profile.parser_kind != PARSER_DOCUMENT
    )
    return tuple(bindings)


def _profile_by_path() -> dict[str, ChineseKBProfile]:
    """按相对路径建立唯一 profile 查找表。"""
    return {profile.relative_path: profile for profile in PROFILES}


def _source_span(item: RawFileManifest, line_number: int,
                 byte_start: int, raw_line: bytes) -> DatasetSourceSpan:
    """为当前原始行构造半开 byte span。"""
    return DatasetSourceSpan(
        item.relative_path,
        line_number,
        byte_start,
        byte_start + len(raw_line),
    )


def _anomaly(kind: str, span: DatasetSourceSpan,
             raw_line: bytes, detail: str = "") -> DatasetAnomaly:
    """用原始字节哈希记录异常，不复制整行内容到报告。"""
    return DatasetAnomaly(kind, span, sha256(raw_line).hexdigest(), detail)


def _parse_fields(profile: ChineseKBProfile,
                  text: str) -> tuple[str, ...]:
    """按 manifest profile 解析一行，具体语义留给 D-01 课程层。"""
    if profile.parser_kind in {PARSER_DOCUMENT, PARSER_SURFACE_LINE}:
        return (text,)
    if profile.parser_kind == PARSER_SYMMETRIC_AT:
        fields = tuple(part.strip() for part in text.split("@"))
        if len(fields) != 2 or any(not field for field in fields):
            raise ValueError("@ 关系行必须恰有两个非空字段")
        return fields
    if profile.parser_kind == PARSER_RELATION_MARKER:
        try:
            columns = next(csv.reader([text], strict=True))
        except csv.Error as error:
            raise ValueError("关系行不是合法 CSV") from error
        if len(columns) == 3 and columns[1].strip() == profile.relation_marker:
            endpoints = [columns[0].strip(), columns[2].strip()]
            if any(not field for field in endpoints):
                raise ValueError("关系标记行必须恰有两个非空端点")
            return tuple(endpoints)
        markers = [index for index, value in enumerate(columns)
                   if value.strip() == profile.relation_marker]
        if len(markers) != 1:
            raise ValueError("关系行必须恰有一个指定 marker")
        marker_index = markers[0]
        if len(columns) == 3:
            endpoints = [value.strip() for index, value in enumerate(columns)
                         if index != marker_index]
        elif 0 < marker_index < len(columns) - 1:
            endpoints = [
                ",".join(columns[:marker_index]).strip(),
                ",".join(columns[marker_index + 1:]).strip(),
            ]
        else:
            raise ValueError("关系行端点含逗号且 marker 位置无法唯一切分")
        if len(endpoints) != 2 or any(not field for field in endpoints):
            raise ValueError("关系标记行必须恰有两个非空端点")
        return tuple(endpoints)
    if profile.parser_kind == PARSER_DECIMAL_TAB:
        fields = tuple(part.strip() for part in text.split("\t"))
        if len(fields) != 2 or any(not field for field in fields):
            raise ValueError("程度行必须恰有 surface 和十进制值")
        parse_decimal_rational(fields[1])
        return fields
    raise ValueError(f"未支持 parser_kind: {profile.parser_kind}")


def _record_key(profile: ChineseKBProfile,
                fields: tuple[str, ...]) -> tuple[str, ...]:
    """构造文件内精确去重键；对声明对称的外部关系消除方向差异。"""
    if profile.symmetric and len(fields) == 2:
        return tuple(sorted(fields))
    return fields


def iter_file_events(
        raw_root: str | Path, item: RawFileManifest,
        profile: ChineseKBProfile) -> Iterator[ChineseKBRecord | DatasetAnomaly]:
    """流式产出有效记录和异常事件，不修改原始文件。"""
    seen: set[tuple[str, ...]] = set()
    byte_start = 0
    with open_verified_binary(raw_root, item) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            span = _source_span(item, line_number, byte_start, raw_line)
            byte_start += len(raw_line)
            try:
                decoded = raw_line.decode(item.encoding, errors="strict")
            except UnicodeDecodeError as error:
                yield _anomaly(
                    "decode_error", span, raw_line,
                    f"byte {error.start}:{error.end}")
                continue
            text = decoded.rstrip("\r\n")
            if line_number == 1 and text.startswith("\ufeff"):
                yield _anomaly("unexpected_bom", span, raw_line)
                text = text.removeprefix("\ufeff")
            if not text.strip():
                if profile.parser_kind != PARSER_DOCUMENT:
                    yield _anomaly("blank_record", span, raw_line)
                continue
            normalized = text.strip()
            if normalized != text:
                yield _anomaly("outer_whitespace", span, raw_line)
            if "\x00" in normalized:
                yield _anomaly("nul_character", span, raw_line)
                continue
            try:
                fields = _parse_fields(profile, normalized)
            except ValueError as error:
                yield _anomaly("malformed_record", span, raw_line, str(error))
                continue
            key = _record_key(profile, fields)
            if profile.parser_kind != PARSER_DOCUMENT and key in seen:
                yield _anomaly("duplicate_key", span, raw_line)
                continue
            seen.add(key)
            yield ChineseKBRecord(
                profile.category, profile.parser_kind, fields, span)


def scan_file(raw_root: str | Path, item: RawFileManifest,
              profile: ChineseKBProfile) -> FileScanReport:
    """完整扫描单文件并返回唯一有效记录数和异常明细。"""
    record_count = 0
    anomalies: list[DatasetAnomaly] = []
    for event in iter_file_events(raw_root, item, profile):
        if isinstance(event, ChineseKBRecord):
            record_count += 1
        else:
            anomalies.append(event)
    return FileScanReport(
        item.relative_path, record_count, tuple(anomalies))


def build_manifest(raw_root: str | Path, *, dataset_version: str,
                   unicode_sequence_family: tuple[int, ...]
                   ) -> tuple[RawDatasetManifest, tuple[FileScanReport, ...]]:
    """扫描固定参与文件并构造不写 raw root 的 ChineseSemanticKB manifest。"""
    root = Path(raw_root).resolve()
    files: list[RawFileManifest] = []
    reports: list[FileScanReport] = []
    for profile in PROFILES:
        path = (root / profile.relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ManifestIntegrityError(
                f"ChineseSemanticKB 参与文件缺失: {profile.relative_path}")
        provisional = RawFileManifest(
            profile.relative_path,
            sha256_file(path),
            path.stat().st_size,
            "utf-8",
            "MARKDOWN" if profile.parser_kind == PARSER_DOCUMENT else "TEXT",
            profile.parser_kind,
            DATASET_NAME,
            profile.category,
        )
        report = scan_file(root, provisional, profile)
        reports.append(report)
        files.append(RawFileManifest(
            provisional.relative_path,
            provisional.sha256,
            provisional.size_bytes,
            provisional.encoding,
            provisional.file_format,
            provisional.parser_kind,
            provisional.property_namespace,
            provisional.property_name,
            report.record_count,
            report.anomaly_count,
        ))
    manifest = RawDatasetManifest(
        DATASET_NAME,
        dataset_version,
        ADAPTER_VERSION,
        PARSER_VERSION,
        LICENSE_ID,
        tuple(files),
        manifest_bindings(unicode_sequence_family),
    )
    return manifest, tuple(reports)


class ChineseSemanticKBAdapter:
    """核验 manifest 后按文件顺序流式读取唯一有效记录。"""

    def __init__(self, manifest: RawDatasetManifest,
                 raw_root: str | Path) -> None:
        if manifest.dataset_name != DATASET_NAME:
            raise ManifestIntegrityError("manifest 不是 ChineseSemanticKB")
        if manifest.adapter_version != ADAPTER_VERSION:
            raise ManifestIntegrityError("ChineseSemanticKB adapter version 不匹配")
        if manifest.parser_version != PARSER_VERSION:
            raise ManifestIntegrityError("ChineseSemanticKB parser version 不匹配")
        verify_manifest(manifest, raw_root)
        expected = _profile_by_path()
        actual = {item.relative_path: item for item in manifest.files}
        if set(actual) != set(expected):
            raise ManifestIntegrityError("ChineseSemanticKB 参与文件集合变化")
        for relative_path, profile in expected.items():
            item = actual[relative_path]
            if (item.parser_kind != profile.parser_kind
                    or item.property_name != profile.category):
                raise ManifestIntegrityError(
                    f"ChineseSemanticKB 文件 profile 不匹配: {relative_path}")
        self.manifest = manifest
        self.raw_root = Path(raw_root).resolve()
        self._items = actual
        self._profiles = expected

    def iter_events(self, *, category: str | None = None
                    ) -> Iterator[ChineseKBRecord | DatasetAnomaly]:
        """按 manifest 路径顺序读取全部或指定类别的记录与异常。"""
        for relative_path in sorted(self._items):
            profile = self._profiles[relative_path]
            if category is not None and profile.category != category:
                continue
            yield from iter_file_events(
                self.raw_root, self._items[relative_path], profile)

    def verify_scan_counts(self) -> None:
        """重扫并核对 manifest 中每文件记录数和异常数。"""
        for relative_path in sorted(self._items):
            item = self._items[relative_path]
            report = scan_file(
                self.raw_root, item, self._profiles[relative_path])
            if (report.record_count != item.record_count
                    or report.anomaly_count != item.anomaly_count):
                raise ManifestIntegrityError(
                    f"ChineseSemanticKB 扫描统计变化: {relative_path}")


__all__ = [
    "ADAPTER_VERSION",
    "ChineseKBProfile",
    "ChineseKBRecord",
    "ChineseSemanticKBAdapter",
    "DATASET_NAME",
    "DatasetAnomaly",
    "DatasetSourceSpan",
    "FileScanReport",
    "LICENSE_ID",
    "PARSER_VERSION",
    "PROFILES",
    "build_manifest",
    "iter_file_events",
    "manifest_bindings",
    "parse_decimal_rational",
    "scan_file",
]
