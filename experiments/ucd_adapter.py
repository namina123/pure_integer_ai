"""Unicode Character Database 只读 manifest 构建和版本化属性适配器。"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
import zipfile

from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.data_manifest import (
    ManifestBinding,
    ManifestIntegrityError,
    RawDatasetManifest,
    RawFileManifest,
    open_verified_text,
    sha256_file,
    verify_manifest,
)


PARSER_ARCHIVE = "archive"
PARSER_DOCUMENT = "document"
PARSER_LICENSE = "license"
PARSER_UCD_README = "ucd_readme"
PARSER_UNICODE_DATA = "unicode_data"
PARSER_ENUMERATED_RANGE = "enumerated_range"
PARSER_BINARY_RANGE = "binary_range"

BINDING_UNICODE_SEQUENCE_FAMILY = "unicode_sequence_family"
BINDING_EXTERNAL_PROPERTY_RELATION = "external_property_relation"
BINDING_UCD_PROVENANCE_KIND = "ucd_provenance_kind"
BINDING_UCD_EPISTEMIC_ORIGIN = "ucd_epistemic_origin"
BINDING_UCD_SCOPE_KIND = "ucd_scope_kind"

_SOURCE_HASHER = Hasher("pure_integer_ai.ucd.source.v1")
_TEXT_HASHER = Hasher("pure_integer_ai.ucd.property.v1")


@dataclass(frozen=True, order=True)
class UcdParseAnomaly:
    """可确定排序的 UCD 坏行或范围冲突。"""

    relative_path: str
    line_number: int
    reason: str


@dataclass(frozen=True, order=True)
class UcdPropertyRecord:
    """某码点在指定 Unicode 版本和来源文件中的外部属性。"""

    unicode_version: tuple[int, int, int]
    codepoint: int
    namespace: str
    property_name: str
    value: str
    source_hash: int

    def integer_key(self, *, parser_version: int,
                    sequence_index: int) -> tuple[int, ...]:
        """把边界文本属性压成进入核心的完整整数 evidence 键。"""
        if parser_version < 0 or sequence_index < 0:
            raise ValueError("parser_version 和 sequence_index 必须非负")
        return (
            *self.unicode_version,
            parser_version,
            self.source_hash,
            _TEXT_HASHER.h63(self.namespace),
            _TEXT_HASHER.h63(self.property_name),
            _TEXT_HASHER.h63(self.value),
            self.codepoint,
            sequence_index,
        )


@dataclass(frozen=True, order=True)
class _RangeValue:
    """闭区间及其属性值。"""

    start: int
    end: int
    value: str


class _ValueRangeIndex:
    """不重叠闭区间的确定性二分查询索引。"""

    def __init__(self, ranges: tuple[_RangeValue, ...], *,
                 default: str | None) -> None:
        ordered = tuple(sorted(ranges, key=lambda item: (
            item.start, item.end, item.value)))
        previous_end = -1
        for item in ordered:
            if item.start < 0 or item.end < item.start or item.end > 0x10FFFF:
                raise ManifestIntegrityError("UCD 范围越界")
            if item.start <= previous_end:
                raise ManifestIntegrityError("UCD 同属性范围重叠")
            previous_end = item.end
        self._ranges = ordered
        self._starts = tuple(item.start for item in ordered)
        self._default = default

    def lookup(self, codepoint: int) -> str | None:
        """返回命中范围的值，未命中时返回文件声明的默认值。"""
        index = bisect_right(self._starts, codepoint) - 1
        if index >= 0:
            item = self._ranges[index]
            if codepoint <= item.end:
                return item.value
        return self._default


@dataclass(frozen=True)
class _PropertyIndex:
    """一个来源文件中的单个外部属性索引。"""

    namespace: str
    property_name: str
    source_hash: int
    values: _ValueRangeIndex


@dataclass(frozen=True)
class _SourceSpec:
    """官方 UCD 文件的解析协议声明。"""

    relative_path: str
    encoding: str
    file_format: str
    parser_kind: str
    namespace: str = ""
    property_name: str = ""


_OFFICIAL_SOURCE_SPECS: tuple[_SourceSpec, ...] = (
    _SourceSpec("downloads/UCD.zip", "", "ZIP", PARSER_ARCHIVE),
    _SourceSpec(
        "downloads/UAX29-tr29-47.html", "utf-8", "HTML",
        PARSER_DOCUMENT),
    _SourceSpec(
        "downloads/license.txt", "utf-8", "TEXT", PARSER_LICENSE),
    _SourceSpec("source/ReadMe.txt", "utf-8", "UCD-TEXT", PARSER_UCD_README),
    _SourceSpec(
        "source/UnicodeData.txt", "utf-8", "UCD-SEMICOLON",
        PARSER_UNICODE_DATA, "UCD", "General_Category"),
    _SourceSpec(
        "source/Scripts.txt", "utf-8", "UCD-RANGE",
        PARSER_ENUMERATED_RANGE, "UCD", "Script"),
    _SourceSpec(
        "source/auxiliary/GraphemeBreakProperty.txt", "utf-8", "UCD-RANGE",
        PARSER_ENUMERATED_RANGE, "UAX29", "Grapheme_Cluster_Break"),
    _SourceSpec(
        "source/auxiliary/SentenceBreakProperty.txt", "utf-8", "UCD-RANGE",
        PARSER_ENUMERATED_RANGE, "UAX29", "Sentence_Break"),
    _SourceSpec(
        "source/PropList.txt", "utf-8", "UCD-RANGE",
        PARSER_BINARY_RANGE, "UCD-PropList"),
    _SourceSpec(
        "source/emoji/emoji-data.txt", "utf-8", "UCD-RANGE",
        PARSER_BINARY_RANGE, "UTS51"),
)


def _parse_codepoint_range(value: str) -> tuple[int, int]:
    """解析 UCD 的单码点或十六进制闭区间。"""
    parts = value.strip().split("..")
    if len(parts) == 1:
        start = end = int(parts[0], 16)
    elif len(parts) == 2:
        start, end = int(parts[0], 16), int(parts[1], 16)
    else:
        raise ValueError("码点范围字段非法")
    if start < 0 or end < start or end > 0x10FFFF:
        raise ValueError("码点范围越界")
    return start, end


def _scan_text_records(path: Path, encoding: str) -> tuple[int, int]:
    """统计普通文本的非空行，并确定性报告解码异常。"""
    try:
        with path.open("r", encoding=encoding, errors="strict", newline="") as handle:
            return sum(1 for line in handle if line.strip()), 0
    except UnicodeError:
        return 0, 1


def _parse_range_lines(lines: list[str], *, relative_path: str,
                       parser_kind: str, namespace: str,
                       property_name: str
                       ) -> tuple[tuple[_PropertyIndex, ...],
                                  tuple[UcdParseAnomaly, ...], int]:
    """解析枚举型或二值 UCD range 文件，并生成属性索引。"""
    grouped: dict[str, list[_RangeValue]] = {}
    default: str | None = None
    anomalies: list[UcdParseAnomaly] = []
    records = 0
    for line_number, raw_line in enumerate(lines, 1):
        stripped = raw_line.strip()
        if stripped.startswith("# @missing:"):
            try:
                missing = stripped.split(":", 1)[1].split("#", 1)[0]
                _, default = [part.strip() for part in missing.split(";", 1)]
            except (ValueError, IndexError):
                anomalies.append(UcdParseAnomaly(
                    relative_path, line_number, "missing 声明非法"))
            continue
        content = raw_line.split("#", 1)[0].strip()
        if not content:
            continue
        try:
            range_text, value = [part.strip() for part in content.split(";", 1)]
            start, end = _parse_codepoint_range(range_text)
            if not value:
                raise ValueError("属性值为空")
            key = property_name if parser_kind == PARSER_ENUMERATED_RANGE else value
            stored_value = value if parser_kind == PARSER_ENUMERATED_RANGE else "Y"
            grouped.setdefault(key, []).append(
                _RangeValue(start, end, stored_value))
            records += 1
        except (ValueError, IndexError) as error:
            anomalies.append(UcdParseAnomaly(
                relative_path, line_number, str(error)))

    indexes: list[_PropertyIndex] = []
    source_hash = 0
    for key in sorted(grouped):
        ranges = tuple(sorted(grouped[key]))
        previous_end = -1
        for item in ranges:
            if item.start <= previous_end:
                anomalies.append(UcdParseAnomaly(
                    relative_path, 0, f"属性 {key} 范围重叠"))
                break
            previous_end = item.end
        if anomalies:
            continue
        indexes.append(_PropertyIndex(
            namespace,
            key,
            source_hash,
            _ValueRangeIndex(
                ranges,
                default=default if parser_kind == PARSER_ENUMERATED_RANGE else None,
            ),
        ))
    return tuple(indexes), tuple(sorted(anomalies)), records


def _parse_unicode_data_lines(
        lines: list[str], *, relative_path: str
        ) -> tuple[tuple[_RangeValue, ...], tuple[UcdParseAnomaly, ...], int]:
    """解析 UnicodeData 的普通行和 First/Last 压缩范围。"""
    ranges: list[_RangeValue] = []
    anomalies: list[UcdParseAnomaly] = []
    pending: tuple[int, str, int] | None = None
    records = 0
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip("\r\n")
        if not line:
            continue
        fields = line.split(";")
        if len(fields) != 15:
            anomalies.append(UcdParseAnomaly(
                relative_path, line_number, "UnicodeData 字段数不是 15"))
            continue
        try:
            codepoint = int(fields[0], 16)
            name = fields[1]
            category = fields[2]
            if name.endswith(", First>"):
                if pending is not None:
                    raise ValueError("嵌套 First 范围")
                pending = (codepoint, category, line_number)
            elif name.endswith(", Last>"):
                if pending is None or pending[1] != category:
                    raise ValueError("Last 没有匹配 First")
                ranges.append(_RangeValue(pending[0], codepoint, category))
                pending = None
            else:
                if pending is not None:
                    raise ValueError("First 后缺少相邻 Last")
                ranges.append(_RangeValue(codepoint, codepoint, category))
            records += 1
        except ValueError as error:
            anomalies.append(UcdParseAnomaly(
                relative_path, line_number, str(error)))
    if pending is not None:
        anomalies.append(UcdParseAnomaly(
            relative_path, pending[2], "First 范围没有 Last"))
    ordered = tuple(sorted(ranges))
    previous_end = -1
    for item in ordered:
        if item.start <= previous_end:
            anomalies.append(UcdParseAnomaly(
                relative_path, 0, "UnicodeData 范围重叠"))
            break
        previous_end = item.end
    return ordered, tuple(sorted(anomalies)), records


def _scan_source(spec: _SourceSpec, raw_root: Path
                 ) -> tuple[RawFileManifest, tuple[UcdParseAnomaly, ...]]:
    """只读扫描一个官方来源并生成内容指纹和解析统计。"""
    path = raw_root / spec.relative_path
    if not path.is_file():
        raise ManifestIntegrityError(f"UCD 原始文件缺失: {spec.relative_path}")
    digest_before = sha256_file(path)
    if spec.parser_kind == PARSER_ARCHIVE:
        try:
            with zipfile.ZipFile(path, "r") as archive:
                record_count = len(archive.infolist())
            anomalies: tuple[UcdParseAnomaly, ...] = ()
        except zipfile.BadZipFile:
            record_count = 0
            anomalies = (UcdParseAnomaly(
                spec.relative_path, 0, "ZIP 归档损坏"),)
    else:
        try:
            lines = path.read_text(
                encoding=spec.encoding, errors="strict").splitlines(keepends=True)
        except UnicodeError:
            lines = []
            anomalies = (UcdParseAnomaly(
                spec.relative_path, 0, "文本解码失败"),)
            record_count = 0
        else:
            if spec.parser_kind == PARSER_UNICODE_DATA:
                _, anomalies, record_count = _parse_unicode_data_lines(
                    lines, relative_path=spec.relative_path)
            elif spec.parser_kind in {
                    PARSER_ENUMERATED_RANGE, PARSER_BINARY_RANGE}:
                _, anomalies, record_count = _parse_range_lines(
                    lines,
                    relative_path=spec.relative_path,
                    parser_kind=spec.parser_kind,
                    namespace=spec.namespace,
                    property_name=spec.property_name,
                )
            else:
                record_count, anomaly_count = _scan_text_records(
                    path, spec.encoding)
                anomalies = () if anomaly_count == 0 else (
                    UcdParseAnomaly(spec.relative_path, 0, "文本解码失败"),)
    digest_after = sha256_file(path)
    if digest_after != digest_before:
        raise ManifestIntegrityError(
            f"扫描期间原始文件变化: {spec.relative_path}")
    return RawFileManifest(
        spec.relative_path,
        digest_after,
        path.stat().st_size,
        spec.encoding,
        spec.file_format,
        spec.parser_kind,
        spec.namespace,
        spec.property_name,
        record_count,
        len(anomalies),
    ), anomalies


def build_official_ucd_manifest(
        raw_root: str | Path, *, unicode_version: str,
        adapter_version: int, parser_version: int,
        unicode_sequence_family_key: tuple[int, ...],
        external_property_relation_key: tuple[int, ...],
        provenance_kind: int,
        epistemic_origin: int,
        scope_kind: int,
        ) -> tuple[RawDatasetManifest, tuple[UcdParseAnomaly, ...]]:
    """扫描官方 UCD/UAX 快照并构造不含绝对路径的只读 manifest。"""
    root = Path(raw_root).resolve()
    manifests: list[RawFileManifest] = []
    anomalies: list[UcdParseAnomaly] = []
    for spec in _OFFICIAL_SOURCE_SPECS:
        file_manifest, file_anomalies = _scan_source(spec, root)
        manifests.append(file_manifest)
        anomalies.extend(file_anomalies)
    readme = (root / "source/ReadMe.txt").read_text(
        encoding="utf-8", errors="strict")
    if f"Version {unicode_version} of the Unicode Standard" not in readme:
        anomalies.append(UcdParseAnomaly(
            "source/ReadMe.txt", 0, "ReadMe Unicode 版本不匹配"))
    manifest = RawDatasetManifest(
        "unicode-ucd-uax29",
        unicode_version,
        adapter_version,
        parser_version,
        "Unicode-3.0",
        tuple(manifests),
        (
            ManifestBinding(
                BINDING_UNICODE_SEQUENCE_FAMILY,
                unicode_sequence_family_key),
            ManifestBinding(
                BINDING_EXTERNAL_PROPERTY_RELATION,
                external_property_relation_key),
            ManifestBinding(BINDING_UCD_PROVENANCE_KIND, (provenance_kind,)),
            ManifestBinding(BINDING_UCD_EPISTEMIC_ORIGIN, (epistemic_origin,)),
            ManifestBinding(BINDING_UCD_SCOPE_KIND, (scope_kind,)),
        ),
    )
    return manifest, tuple(sorted(anomalies))


class UcdReadOnlyAdapter:
    """从已核验 manifest 构建按码点查询的版本化 UCD 属性索引。"""

    def __init__(self, raw_root: str | Path,
                 manifest: RawDatasetManifest) -> None:
        verify_manifest(manifest, raw_root)
        if manifest.dataset_name != "unicode-ucd-uax29":
            raise ManifestIntegrityError("manifest 不是 UCD/UAX29 数据集")
        if any(item.anomaly_count for item in manifest.files):
            raise ManifestIntegrityError("UCD manifest 含解析异常，拒绝正式加载")
        self._root = Path(raw_root).resolve()
        self.manifest = manifest
        self.unicode_version = self._parse_version(manifest.dataset_version)
        self._indexes = self._load_indexes()

    @staticmethod
    def _parse_version(value: str) -> tuple[int, int, int]:
        """把三段 Unicode 版本文本转换为严格整数元组。"""
        parts = value.split(".")
        if len(parts) != 3 or any(not part.isdigit() for part in parts):
            raise ManifestIntegrityError("Unicode 版本必须为三段非负整数")
        return int(parts[0]), int(parts[1]), int(parts[2])

    def _load_indexes(self) -> tuple[_PropertyIndex, ...]:
        """重新解析所有属性文件，并要求统计与 manifest 完全一致。"""
        indexes: list[_PropertyIndex] = []
        for item in self.manifest.files:
            if item.parser_kind not in {
                    PARSER_UNICODE_DATA,
                    PARSER_ENUMERATED_RANGE,
                    PARSER_BINARY_RANGE}:
                continue
            with open_verified_text(self._root, item) as handle:
                lines = list(handle)
            if item.parser_kind == PARSER_UNICODE_DATA:
                ranges, anomalies, record_count = _parse_unicode_data_lines(
                    lines, relative_path=item.relative_path)
                parsed = (_PropertyIndex(
                    item.property_namespace,
                    item.property_name,
                    _SOURCE_HASHER.h63(item.sha256),
                    _ValueRangeIndex(ranges, default="Cn"),
                ),)
            else:
                parsed, anomalies, record_count = _parse_range_lines(
                    lines,
                    relative_path=item.relative_path,
                    parser_kind=item.parser_kind,
                    namespace=item.property_namespace,
                    property_name=item.property_name,
                )
                source_hash = _SOURCE_HASHER.h63(item.sha256)
                parsed = tuple(_PropertyIndex(
                    entry.namespace,
                    entry.property_name,
                    source_hash,
                    entry.values,
                ) for entry in parsed)
            if anomalies or record_count != item.record_count:
                raise ManifestIntegrityError(
                    f"UCD 解析统计变化: {item.relative_path}")
            indexes.extend(parsed)
        return tuple(sorted(indexes, key=lambda item: (
            item.namespace, item.property_name, item.source_hash)))

    def properties_for(self, codepoint: int) -> tuple[UcdPropertyRecord, ...]:
        """返回某 Unicode scalar 的全部已配置外部属性，不推断语言或结构作用。"""
        if (type(codepoint) is not int or codepoint < 0
                or codepoint > 0x10FFFF
                or 0xD800 <= codepoint <= 0xDFFF):
            raise ValueError("codepoint 不是 Unicode scalar value")
        records: list[UcdPropertyRecord] = []
        for index in self._indexes:
            value = index.values.lookup(codepoint)
            if value is None:
                continue
            records.append(UcdPropertyRecord(
                self.unicode_version,
                codepoint,
                index.namespace,
                index.property_name,
                value,
                index.source_hash,
            ))
        return tuple(sorted(records))


__all__ = [
    "BINDING_EXTERNAL_PROPERTY_RELATION",
    "BINDING_UCD_EPISTEMIC_ORIGIN",
    "BINDING_UCD_PROVENANCE_KIND",
    "BINDING_UCD_SCOPE_KIND",
    "BINDING_UNICODE_SEQUENCE_FAMILY",
    "UcdParseAnomaly",
    "UcdPropertyRecord",
    "UcdReadOnlyAdapter",
    "build_official_ucd_manifest",
]
