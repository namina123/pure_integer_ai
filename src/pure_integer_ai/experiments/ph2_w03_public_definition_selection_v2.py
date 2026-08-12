"""FT30 Wiktionary 公开定义覆盖扩展的确定性标题选择合同。"""
from __future__ import annotations

import bz2
from dataclasses import dataclass
import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    iter_multistream_index,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    MediaWikiDumpSnapshotManifest,
)


FT30_SELECTION_FORMAT = "PH2_FT30_PUBLIC_DEFINITION_SELECTION"
FT30_SELECTION_VERSION = 2
FT30_SELECTION_RULE = "SNAPSHOT_NUL_TITLE_SHA256_LENGTH_STRATA_V1"
FT30_TITLE_FILTER = "CJK_UNIFIED_IDEOGRAPHS_U4E00_U9FFF"
FT30_MAX_SELECTED_TITLES = 32
FT30_STRATA = (
    ("L1", 1, 1, 8),
    ("L2_4", 2, 4, 8),
    ("L5_8", 5, 8, 8),
    ("L9_16", 9, 16, 8),
)


# object-model: exception
class FT30PublicDefinitionSelectionError(RuntimeError):
    """索引身份、选择规则或冻结 manifest 发生漂移。"""


def _sha256_path(path: Path) -> str:
    """以固定块大小计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _raw_file(manifest: MediaWikiDumpSnapshotManifest, role: str):
    """返回 snapshot 中指定角色的唯一 raw file。"""
    values = tuple(item for item in manifest.raw_files if item.role == role)
    if len(values) != 1:
        raise FT30PublicDefinitionSelectionError(
            f"FT30 snapshot {role} raw file 非唯一")
    return values[0]


def _safe_path(root: Path, relative_path: str) -> Path:
    """解析 raw 相对路径并拒绝目录逃逸。"""
    path = (root / Path(*relative_path.split("/"))).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FT30PublicDefinitionSelectionError("FT30 raw path 缺失或逃逸")
    return path


def _is_selected_title_form(title: str) -> bool:
    """仅接受由基本 CJK 统一汉字组成的非空标题。"""
    return bool(title) and all("\u4e00" <= character <= "\u9fff"
                               for character in title)


def _stratum_for_length(length: int) -> str | None:
    """按冻结字符长度边界返回标题分层。"""
    for name, minimum, maximum, _ in FT30_STRATA:
        if minimum <= length <= maximum:
            return name
    return None


def _selection_sha256(snapshot_id: str, title: str) -> str:
    """计算公开选择排名使用的 snapshot/title 稳定摘要。"""
    return hashlib.sha256(
        (snapshot_id + "\0" + title).encode("utf-8")).hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FT30SelectedTitle:
    """一个由 index 身份和稳定排名冻结的目标页坐标。"""

    stratum: str
    title: str
    title_length: int
    title_sha256: str
    selection_sha256: str
    page_id: int
    index_line_number: int
    compressed_block_offset: int
    compressed_block_end_offset: int

    def __post_init__(self) -> None:
        names = {item[0] for item in FT30_STRATA}
        if self.stratum not in names:
            raise FT30PublicDefinitionSelectionError("FT30 title stratum 非法")
        if (not isinstance(self.title, str)
                or not _is_selected_title_form(self.title)
                or self.title_length != len(self.title)
                or _stratum_for_length(self.title_length) != self.stratum):
            raise FT30PublicDefinitionSelectionError("FT30 title 字段漂移")
        expected_title_sha = hashlib.sha256(
            self.title.encode("utf-8")).hexdigest()
        if self.title_sha256 != expected_title_sha:
            raise FT30PublicDefinitionSelectionError("FT30 title SHA 漂移")
        for value in (self.selection_sha256, self.title_sha256):
            if (not isinstance(value, str) or len(value) != 64
                    or any(character not in "0123456789abcdef"
                           for character in value)):
                raise FT30PublicDefinitionSelectionError("FT30 SHA 非法")
        if (type(self.page_id) is not int or self.page_id <= 0
                or type(self.index_line_number) is not int
                or self.index_line_number <= 0
                or type(self.compressed_block_offset) is not int
                or self.compressed_block_offset < 0
                or type(self.compressed_block_end_offset) is not int
                or self.compressed_block_end_offset
                <= self.compressed_block_offset):
            raise FT30PublicDefinitionSelectionError("FT30 index 坐标非法")

    def to_dict(self) -> dict[str, object]:
        """导出规范 JSON 值。"""
        return {
            "compressed_block_end_offset": self.compressed_block_end_offset,
            "compressed_block_offset": self.compressed_block_offset,
            "index_line_number": self.index_line_number,
            "page_id": self.page_id,
            "selection_sha256": self.selection_sha256,
            "stratum": self.stratum,
            "title": self.title,
            "title_length": self.title_length,
            "title_sha256": self.title_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "FT30SelectedTitle":
        """从字段精确的 JSON object 恢复目标页坐标。"""
        keys = {
            "compressed_block_end_offset", "compressed_block_offset",
            "index_line_number", "page_id", "selection_sha256", "stratum",
            "title", "title_length", "title_sha256",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise FT30PublicDefinitionSelectionError(
                "FT30 selected title 字段集合漂移")
        return cls(
            value["stratum"], value["title"], value["title_length"],
            value["title_sha256"], value["selection_sha256"],
            value["page_id"], value["index_line_number"],
            value["compressed_block_offset"],
            value["compressed_block_end_offset"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FT30PublicDefinitionSelectionManifest:
    """绑定 snapshot/index、选择规则、排除基线与全部目标坐标。"""

    source_key: str
    snapshot_id: str
    snapshot_manifest_relative_path: str
    snapshot_manifest_sha256: str
    index_raw_relative_path: str
    index_compressed_size_bytes: int
    index_local_sha256: str
    index_upstream_sha1: str
    base_selection_manifest_relative_path: str
    base_selection_manifest_sha256: str
    excluded_title_sha256s: tuple[str, ...]
    index_entry_count: int
    eligible_title_count: int
    eligible_counts_by_stratum: tuple[tuple[str, int], ...]
    selected_titles: tuple[FT30SelectedTitle, ...]

    def __post_init__(self) -> None:
        if self.source_key != "ZHWIKTIONARY_20260701":
            raise FT30PublicDefinitionSelectionError("FT30 source_key 漂移")
        for name in (
                "snapshot_id", "snapshot_manifest_relative_path",
                "index_raw_relative_path",
                "base_selection_manifest_relative_path"):
            value = getattr(self, name)
            if (not isinstance(value, str) or not value
                    or "\\" in value or value.startswith(("/", "../"))):
                raise FT30PublicDefinitionSelectionError(
                    f"FT30 {name} 非安全文本")
        for name in (
                "snapshot_manifest_sha256", "index_local_sha256",
                "base_selection_manifest_sha256"):
            value = getattr(self, name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(character not in "0123456789abcdef"
                           for character in value)):
                raise FT30PublicDefinitionSelectionError(
                    f"FT30 {name} 非法")
        if (not isinstance(self.index_upstream_sha1, str)
                or len(self.index_upstream_sha1) != 40
                or any(character not in "0123456789abcdef"
                       for character in self.index_upstream_sha1)):
            raise FT30PublicDefinitionSelectionError("FT30 index SHA-1 非法")
        for name in (
                "index_compressed_size_bytes", "index_entry_count",
                "eligible_title_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise FT30PublicDefinitionSelectionError(
                    f"FT30 {name} 非正整数")
        if (tuple(sorted(set(self.excluded_title_sha256s)))
                != self.excluded_title_sha256s):
            raise FT30PublicDefinitionSelectionError(
                "FT30 excluded title SHA 非规范")
        expected_strata = tuple((item[0], item[3]) for item in FT30_STRATA)
        actual_strata = tuple(
            (name, sum(selected.stratum == name
                       for selected in self.selected_titles))
            for name, _ in expected_strata)
        if actual_strata != expected_strata:
            raise FT30PublicDefinitionSelectionError("FT30 strata quota 未满足")
        eligible_names = tuple(item[0] for item in FT30_STRATA)
        if (tuple(name for name, _ in self.eligible_counts_by_stratum)
                != eligible_names
                or sum(count for _, count in self.eligible_counts_by_stratum)
                != self.eligible_title_count):
            raise FT30PublicDefinitionSelectionError(
                "FT30 eligible strata 计数漂移")
        if (len(self.selected_titles) != FT30_MAX_SELECTED_TITLES
                or len({item.title for item in self.selected_titles})
                != len(self.selected_titles)):
            raise FT30PublicDefinitionSelectionError(
                "FT30 selected title 数量或唯一性漂移")
        ordered = tuple(sorted(
            self.selected_titles,
            key=lambda item: (
                eligible_names.index(item.stratum),
                item.selection_sha256, item.title, item.page_id),
        ))
        if ordered != self.selected_titles:
            raise FT30PublicDefinitionSelectionError(
                "FT30 selected title 顺序漂移")
        for item in self.selected_titles:
            if item.selection_sha256 != _selection_sha256(
                    self.snapshot_id, item.title):
                raise FT30PublicDefinitionSelectionError(
                    "FT30 selection SHA 漂移")
            if item.title_sha256 in self.excluded_title_sha256s:
                raise FT30PublicDefinitionSelectionError(
                    "FT30 expansion 重新选择了 v1 title")

    def to_dict(self) -> dict[str, object]:
        """导出规范 manifest JSON 值。"""
        return {
            "artifact_kind": FT30_SELECTION_FORMAT,
            "base_selection_manifest_relative_path": (
                self.base_selection_manifest_relative_path),
            "base_selection_manifest_sha256": (
                self.base_selection_manifest_sha256),
            "eligible_counts_by_stratum": {
                key: value for key, value in self.eligible_counts_by_stratum},
            "eligible_title_count": self.eligible_title_count,
            "excluded_title_sha256s": list(self.excluded_title_sha256s),
            "format_version": 1,
            "index_compressed_size_bytes": self.index_compressed_size_bytes,
            "index_entry_count": self.index_entry_count,
            "index_local_sha256": self.index_local_sha256,
            "index_raw_relative_path": self.index_raw_relative_path,
            "index_upstream_sha1": self.index_upstream_sha1,
            "max_selected_titles": FT30_MAX_SELECTED_TITLES,
            "selected_titles": [item.to_dict() for item in self.selected_titles],
            "selection_rule": FT30_SELECTION_RULE,
            "selection_version": FT30_SELECTION_VERSION,
            "snapshot_id": self.snapshot_id,
            "snapshot_manifest_relative_path": (
                self.snapshot_manifest_relative_path),
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "source_key": self.source_key,
            "strata": [
                {
                    "maximum_title_length": maximum,
                    "minimum_title_length": minimum,
                    "quota": quota,
                    "stratum": name,
                }
                for name, minimum, maximum, quota in FT30_STRATA
            ],
            "title_filter": FT30_TITLE_FILTER,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 manifest 字节。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回完整规范 manifest SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "FT30PublicDefinitionSelectionManifest":
        """从字段精确的 JSON object 恢复选择 manifest。"""
        keys = {
            "artifact_kind", "base_selection_manifest_relative_path",
            "base_selection_manifest_sha256", "eligible_counts_by_stratum",
            "eligible_title_count", "excluded_title_sha256s", "format_version",
            "index_compressed_size_bytes", "index_entry_count",
            "index_local_sha256", "index_raw_relative_path",
            "index_upstream_sha1", "max_selected_titles", "selected_titles",
            "selection_rule", "selection_version", "snapshot_id",
            "snapshot_manifest_relative_path", "snapshot_manifest_sha256",
            "source_key", "strata", "title_filter",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise FT30PublicDefinitionSelectionError(
                "FT30 selection manifest 字段集合漂移")
        if (value["artifact_kind"] != FT30_SELECTION_FORMAT
                or value["format_version"] != 1
                or value["selection_version"] != FT30_SELECTION_VERSION
                or value["selection_rule"] != FT30_SELECTION_RULE
                or value["title_filter"] != FT30_TITLE_FILTER
                or value["max_selected_titles"] != FT30_MAX_SELECTED_TITLES):
            raise FT30PublicDefinitionSelectionError(
                "FT30 selection manifest 合同漂移")
        expected_strata = [
            {
                "maximum_title_length": maximum,
                "minimum_title_length": minimum,
                "quota": quota,
                "stratum": name,
            }
            for name, minimum, maximum, quota in FT30_STRATA
        ]
        if value["strata"] != expected_strata:
            raise FT30PublicDefinitionSelectionError("FT30 strata 定义漂移")
        eligible = value["eligible_counts_by_stratum"]
        if not isinstance(eligible, dict) or set(eligible) != {
                item[0] for item in FT30_STRATA}:
            raise FT30PublicDefinitionSelectionError(
                "FT30 eligible strata 字段漂移")
        if (not isinstance(value["excluded_title_sha256s"], list)
                or not isinstance(value["selected_titles"], list)):
            raise FT30PublicDefinitionSelectionError(
                "FT30 selection manifest list 字段非法")
        return cls(
            value["source_key"], value["snapshot_id"],
            value["snapshot_manifest_relative_path"],
            value["snapshot_manifest_sha256"],
            value["index_raw_relative_path"],
            value["index_compressed_size_bytes"],
            value["index_local_sha256"], value["index_upstream_sha1"],
            value["base_selection_manifest_relative_path"],
            value["base_selection_manifest_sha256"],
            tuple(value["excluded_title_sha256s"]),
            value["index_entry_count"], value["eligible_title_count"],
            tuple((name, eligible[name]) for name, *_ in FT30_STRATA),
            tuple(FT30SelectedTitle.from_dict(item)
                  for item in value["selected_titles"]),
        )


def build_ft30_public_definition_selection(
        manifest: MediaWikiDumpSnapshotManifest,
        *,
        raw_root: str | Path,
        snapshot_manifest_relative_path: str,
        snapshot_manifest_sha256: str,
        base_selection_manifest_relative_path: str,
        base_selection_manifest_sha256: str,
        excluded_titles: tuple[str, ...],
        ) -> FT30PublicDefinitionSelectionManifest:
    """单遍扫描 index，按冻结分层排名选择标题并保存 block 终点。"""
    if not isinstance(manifest, MediaWikiDumpSnapshotManifest):
        raise TypeError("FT30 snapshot manifest 类型错误")
    if (manifest.source_key != "ZHWIKTIONARY_20260701"
            or manifest.project != "zhwiktionary"):
        raise FT30PublicDefinitionSelectionError("FT30 只接受冻结 Wiktionary")
    if (not isinstance(excluded_titles, tuple)
            or any(not isinstance(item, str) or not item
                   for item in excluded_titles)
            or len(set(excluded_titles)) != len(excluded_titles)):
        raise FT30PublicDefinitionSelectionError("FT30 excluded titles 非法")
    raw = Path(raw_root).resolve()
    index_file = _raw_file(manifest, "INDEX")
    index_path = _safe_path(raw, index_file.raw_relative_path)
    if (index_path.stat().st_size != index_file.compressed_size_bytes
            or _sha256_path(index_path) != index_file.local_sha256):
        raise FT30PublicDefinitionSelectionError("FT30 index size/SHA 漂移")
    excluded_sha = tuple(sorted(
        hashlib.sha256(item.encode("utf-8")).hexdigest()
        for item in excluded_titles))
    excluded_set = set(excluded_titles)
    candidates: dict[str, list[tuple[str, str, int, int, int]]] = {
        item[0]: [] for item in FT30_STRATA}
    eligible_counts = {item[0]: 0 for item in FT30_STRATA}
    offsets: list[int] = []
    prior_offset = -1
    index_count = 0
    with bz2.open(index_path, "rb") as stream:
        for entry in iter_multistream_index(stream):
            index_count += 1
            if entry.offset != prior_offset:
                offsets.append(entry.offset)
                prior_offset = entry.offset
            if not _is_selected_title_form(entry.title):
                continue
            stratum = _stratum_for_length(len(entry.title))
            if stratum is None or entry.title in excluded_set:
                continue
            eligible_counts[stratum] += 1
            rank = _selection_sha256(manifest.snapshot_id, entry.title)
            values = candidates[stratum]
            values.append((
                rank, entry.title, entry.page_id, entry.line_number,
                entry.offset,
            ))
            quota = next(item[3] for item in FT30_STRATA
                         if item[0] == stratum)
            if len(values) > quota * 4:
                values.sort()
                del values[quota:]
    offset_end = {
        offset: (
            offsets[index + 1]
            if index + 1 < len(offsets)
            else _raw_file(manifest, "XML").compressed_size_bytes)
        for index, offset in enumerate(offsets)
    }
    selected = []
    for name, _, _, quota in FT30_STRATA:
        values = sorted(candidates[name])[:quota]
        if len(values) != quota:
            raise FT30PublicDefinitionSelectionError(
                f"FT30 stratum {name} 不足 quota")
        for rank, title, page_id, line_number, offset in values:
            selected.append(FT30SelectedTitle(
                name,
                title,
                len(title),
                hashlib.sha256(title.encode("utf-8")).hexdigest(),
                rank,
                page_id,
                line_number,
                offset,
                offset_end[offset],
            ))
    return FT30PublicDefinitionSelectionManifest(
        manifest.source_key,
        manifest.snapshot_id,
        snapshot_manifest_relative_path,
        snapshot_manifest_sha256,
        index_file.raw_relative_path,
        index_file.compressed_size_bytes,
        index_file.local_sha256,
        index_file.upstream_sha1,
        base_selection_manifest_relative_path,
        base_selection_manifest_sha256,
        excluded_sha,
        index_count,
        sum(eligible_counts.values()),
        tuple((name, eligible_counts[name]) for name, *_ in FT30_STRATA),
        tuple(selected),
    )


def write_ft30_public_definition_selection(
        manifest: FT30PublicDefinitionSelectionManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等发布 canonical selection manifest。"""
    if not isinstance(manifest, FT30PublicDefinitionSelectionManifest):
        raise TypeError("FT30 selection manifest 类型错误")
    target = Path(path).resolve()
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FT30PublicDefinitionSelectionError(
                "FT30 selection manifest 已存在且字节不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise FT30PublicDefinitionSelectionError(
            "FT30 selection manifest 无法发布") from error
    return target


def read_ft30_public_definition_selection(
        path: str | Path,
        ) -> FT30PublicDefinitionSelectionManifest:
    """严格回读 canonical selection manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise FT30PublicDefinitionSelectionError(
                "FT30 selection manifest 换行非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        manifest = FT30PublicDefinitionSelectionManifest.from_dict(value)
    except FT30PublicDefinitionSelectionError:
        raise
    except OSError as error:
        raise FT30PublicDefinitionSelectionError(
            "FT30 selection manifest 缺失或不可读") from error
    if manifest.canonical_bytes() != payload:
        raise FT30PublicDefinitionSelectionError(
            "FT30 selection manifest 非规范字节")
    return manifest


__all__ = [
    "FT30_MAX_SELECTED_TITLES",
    "FT30_SELECTION_FORMAT",
    "FT30_SELECTION_RULE",
    "FT30_SELECTION_VERSION",
    "FT30_STRATA",
    "FT30PublicDefinitionSelectionError",
    "FT30PublicDefinitionSelectionManifest",
    "FT30SelectedTitle",
    "build_ft30_public_definition_selection",
    "read_ft30_public_definition_selection",
    "write_ft30_public_definition_selection",
]
