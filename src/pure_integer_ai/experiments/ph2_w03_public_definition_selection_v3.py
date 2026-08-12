"""FT31 Wiktionary 公开定义规模切片的确定性标题选择合同。"""
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
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v2 import (
    FT30SelectedTitle,
)


FT31_SELECTION_FORMAT = "PH2_FT31_PUBLIC_DEFINITION_SELECTION"
FT31_SELECTION_VERSION = 3
FT31_SELECTION_RULE = "SNAPSHOT_NUL_TITLE_SHA256_LENGTH_STRATA_V1"
FT31_TITLE_FILTER = "CJK_UNIFIED_IDEOGRAPHS_U4E00_U9FFF"
FT31_MAX_SELECTED_TITLES = 256
FT31_STRATA = (
    ("L1", 1, 1, 64),
    ("L2_4", 2, 4, 64),
    ("L5_8", 5, 8, 64),
    ("L9_16", 9, 16, 64),
)


# object-model: exception
class FT31PublicDefinitionSelectionError(RuntimeError):
    """索引身份、规模选择规则或冻结 manifest 发生漂移。"""


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
        raise FT31PublicDefinitionSelectionError(
            f"FT31 snapshot {role} raw file 非唯一")
    return values[0]


def _safe_path(root: Path, relative_path: str) -> Path:
    """解析 raw 相对路径并拒绝目录逃逸。"""
    path = (root / Path(*relative_path.split("/"))).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FT31PublicDefinitionSelectionError("FT31 raw path 缺失或逃逸")
    return path


def _is_selected_title_form(title: str) -> bool:
    """仅接受由基本 CJK 统一汉字组成的非空标题。"""
    return bool(title) and all("\u4e00" <= item <= "\u9fff" for item in title)


def _stratum_for_length(length: int) -> str | None:
    """按冻结字符长度边界返回标题分层。"""
    for name, minimum, maximum, _ in FT31_STRATA:
        if minimum <= length <= maximum:
            return name
    return None


def _selection_sha256(snapshot_id: str, title: str) -> str:
    """计算与 FT30 相同的公开稳定排名摘要。"""
    return hashlib.sha256(
        (snapshot_id + "\0" + title).encode("utf-8")).hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FT31PublicDefinitionSelectionManifest:
    """绑定 snapshot/index、FT26+FT30 排除集与 256 个目标坐标。"""

    source_key: str
    snapshot_id: str
    snapshot_manifest_relative_path: str
    snapshot_manifest_sha256: str
    index_raw_relative_path: str
    index_compressed_size_bytes: int
    index_local_sha256: str
    index_upstream_sha1: str
    predecessor_selection_relative_paths: tuple[str, ...]
    predecessor_selection_sha256s: tuple[str, ...]
    excluded_title_sha256s: tuple[str, ...]
    index_entry_count: int
    eligible_title_count: int
    eligible_counts_by_stratum: tuple[tuple[str, int], ...]
    selected_titles: tuple[FT30SelectedTitle, ...]

    def __post_init__(self) -> None:
        if self.source_key != "ZHWIKTIONARY_20260701":
            raise FT31PublicDefinitionSelectionError("FT31 source_key 漂移")
        paths = (
            self.snapshot_id,
            self.snapshot_manifest_relative_path,
            self.index_raw_relative_path,
            *self.predecessor_selection_relative_paths,
        )
        if any(not isinstance(item, str) or not item or "\\" in item
               or item.startswith(("/", "../")) for item in paths):
            raise FT31PublicDefinitionSelectionError("FT31 路径或 identity 非法")
        hashes = (
            self.snapshot_manifest_sha256,
            self.index_local_sha256,
            *self.predecessor_selection_sha256s,
            *self.excluded_title_sha256s,
        )
        if any(not isinstance(item, str) or len(item) != 64
               or any(character not in "0123456789abcdef"
                      for character in item) for item in hashes):
            raise FT31PublicDefinitionSelectionError("FT31 SHA-256 非法")
        if (len(self.predecessor_selection_relative_paths) != 2
                or len(self.predecessor_selection_sha256s) != 2
                or tuple(sorted(set(self.excluded_title_sha256s)))
                != self.excluded_title_sha256s):
            raise FT31PublicDefinitionSelectionError("FT31 predecessor 非规范")
        if (not isinstance(self.index_upstream_sha1, str)
                or len(self.index_upstream_sha1) != 40
                or any(item not in "0123456789abcdef"
                       for item in self.index_upstream_sha1)):
            raise FT31PublicDefinitionSelectionError("FT31 index SHA-1 非法")
        for name in (
                "index_compressed_size_bytes", "index_entry_count",
                "eligible_title_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise FT31PublicDefinitionSelectionError(
                    f"FT31 {name} 非正整数")
        expected_counts = tuple((item[0], item[3]) for item in FT31_STRATA)
        actual_counts = tuple(
            (name, sum(item.stratum == name for item in self.selected_titles))
            for name, _ in expected_counts)
        if actual_counts != expected_counts:
            raise FT31PublicDefinitionSelectionError("FT31 strata quota 未满足")
        if (tuple(name for name, _ in self.eligible_counts_by_stratum)
                != tuple(item[0] for item in FT31_STRATA)
                or sum(count for _, count in self.eligible_counts_by_stratum)
                != self.eligible_title_count):
            raise FT31PublicDefinitionSelectionError("FT31 eligible 计数漂移")
        if (len(self.selected_titles) != FT31_MAX_SELECTED_TITLES
                or len({item.title for item in self.selected_titles})
                != FT31_MAX_SELECTED_TITLES):
            raise FT31PublicDefinitionSelectionError("FT31 title 数量漂移")
        names = tuple(item[0] for item in FT31_STRATA)
        ordered = tuple(sorted(
            self.selected_titles,
            key=lambda item: (
                names.index(item.stratum), item.selection_sha256,
                item.title, item.page_id),
        ))
        if ordered != self.selected_titles:
            raise FT31PublicDefinitionSelectionError("FT31 title 顺序漂移")
        excluded = set(self.excluded_title_sha256s)
        for item in self.selected_titles:
            if item.selection_sha256 != _selection_sha256(
                    self.snapshot_id, item.title):
                raise FT31PublicDefinitionSelectionError("FT31 排名 SHA 漂移")
            if item.title_sha256 in excluded:
                raise FT31PublicDefinitionSelectionError("FT31 重选历史 title")

    def to_dict(self) -> dict[str, object]:
        """导出规范 manifest JSON 值。"""
        return {
            "artifact_kind": FT31_SELECTION_FORMAT,
            "eligible_counts_by_stratum": dict(
                self.eligible_counts_by_stratum),
            "eligible_title_count": self.eligible_title_count,
            "excluded_title_sha256s": list(self.excluded_title_sha256s),
            "format_version": 1,
            "index_compressed_size_bytes": self.index_compressed_size_bytes,
            "index_entry_count": self.index_entry_count,
            "index_local_sha256": self.index_local_sha256,
            "index_raw_relative_path": self.index_raw_relative_path,
            "index_upstream_sha1": self.index_upstream_sha1,
            "max_selected_titles": FT31_MAX_SELECTED_TITLES,
            "predecessor_selection_relative_paths": list(
                self.predecessor_selection_relative_paths),
            "predecessor_selection_sha256s": list(
                self.predecessor_selection_sha256s),
            "selected_titles": [item.to_dict() for item in self.selected_titles],
            "selection_rule": FT31_SELECTION_RULE,
            "selection_version": FT31_SELECTION_VERSION,
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
                for name, minimum, maximum, quota in FT31_STRATA
            ],
            "title_filter": FT31_TITLE_FILTER,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 manifest 字节。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回规范 manifest SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "FT31PublicDefinitionSelectionManifest":
        """从字段精确的 JSON object 恢复选择 manifest。"""
        keys = {
            "artifact_kind", "eligible_counts_by_stratum",
            "eligible_title_count", "excluded_title_sha256s", "format_version",
            "index_compressed_size_bytes", "index_entry_count",
            "index_local_sha256", "index_raw_relative_path",
            "index_upstream_sha1", "max_selected_titles",
            "predecessor_selection_relative_paths",
            "predecessor_selection_sha256s", "selected_titles",
            "selection_rule", "selection_version", "snapshot_id",
            "snapshot_manifest_relative_path", "snapshot_manifest_sha256",
            "source_key", "strata", "title_filter",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise FT31PublicDefinitionSelectionError("FT31 manifest 字段漂移")
        if (value["artifact_kind"] != FT31_SELECTION_FORMAT
                or value["format_version"] != 1
                or value["selection_version"] != FT31_SELECTION_VERSION
                or value["selection_rule"] != FT31_SELECTION_RULE
                or value["title_filter"] != FT31_TITLE_FILTER
                or value["max_selected_titles"] != FT31_MAX_SELECTED_TITLES):
            raise FT31PublicDefinitionSelectionError("FT31 manifest 合同漂移")
        expected_strata = [
            {
                "maximum_title_length": maximum,
                "minimum_title_length": minimum,
                "quota": quota,
                "stratum": name,
            }
            for name, minimum, maximum, quota in FT31_STRATA
        ]
        eligible = value["eligible_counts_by_stratum"]
        if (value["strata"] != expected_strata
                or not isinstance(eligible, dict)
                or set(eligible) != {item[0] for item in FT31_STRATA}
                or not all(isinstance(value[name], list) for name in (
                    "excluded_title_sha256s",
                    "predecessor_selection_relative_paths",
                    "predecessor_selection_sha256s", "selected_titles"))):
            raise FT31PublicDefinitionSelectionError("FT31 manifest 列表漂移")
        return cls(
            value["source_key"], value["snapshot_id"],
            value["snapshot_manifest_relative_path"],
            value["snapshot_manifest_sha256"],
            value["index_raw_relative_path"],
            value["index_compressed_size_bytes"],
            value["index_local_sha256"], value["index_upstream_sha1"],
            tuple(value["predecessor_selection_relative_paths"]),
            tuple(value["predecessor_selection_sha256s"]),
            tuple(value["excluded_title_sha256s"]),
            value["index_entry_count"], value["eligible_title_count"],
            tuple((name, eligible[name]) for name, *_ in FT31_STRATA),
            tuple(FT30SelectedTitle.from_dict(item)
                  for item in value["selected_titles"]),
        )


def build_ft31_public_definition_selection(
        manifest: MediaWikiDumpSnapshotManifest,
        *,
        raw_root: str | Path,
        snapshot_manifest_relative_path: str,
        snapshot_manifest_sha256: str,
        predecessor_selection_relative_paths: tuple[str, str],
        predecessor_selection_sha256s: tuple[str, str],
        excluded_titles: tuple[str, ...],
        ) -> FT31PublicDefinitionSelectionManifest:
    """单遍扫描 index，按冻结分层排名选择 256 个新标题。"""
    if not isinstance(manifest, MediaWikiDumpSnapshotManifest):
        raise TypeError("FT31 snapshot manifest 类型错误")
    if (manifest.source_key != "ZHWIKTIONARY_20260701"
            or manifest.project != "zhwiktionary"):
        raise FT31PublicDefinitionSelectionError("FT31 只接受冻结 Wiktionary")
    if (not isinstance(excluded_titles, tuple)
            or any(not isinstance(item, str) or not item
                   for item in excluded_titles)
            or len(set(excluded_titles)) != len(excluded_titles)):
        raise FT31PublicDefinitionSelectionError("FT31 excluded titles 非法")
    raw = Path(raw_root).resolve()
    index_file = _raw_file(manifest, "INDEX")
    index_path = _safe_path(raw, index_file.raw_relative_path)
    if (index_path.stat().st_size != index_file.compressed_size_bytes
            or _sha256_path(index_path) != index_file.local_sha256):
        raise FT31PublicDefinitionSelectionError("FT31 index size/SHA 漂移")
    excluded_set = set(excluded_titles)
    excluded_sha = tuple(sorted(
        hashlib.sha256(item.encode("utf-8")).hexdigest()
        for item in excluded_titles))
    candidates: dict[str, list[tuple[str, str, int, int, int]]] = {
        item[0]: [] for item in FT31_STRATA}
    eligible_counts = {item[0]: 0 for item in FT31_STRATA}
    offsets = []
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
            quota = next(item[3] for item in FT31_STRATA
                         if item[0] == stratum)
            if len(values) > quota * 4:
                values.sort()
                del values[quota:]
    xml_size = _raw_file(manifest, "XML").compressed_size_bytes
    offset_end = {
        offset: offsets[index + 1] if index + 1 < len(offsets) else xml_size
        for index, offset in enumerate(offsets)
    }
    selected = []
    for name, _, _, quota in FT31_STRATA:
        values = sorted(candidates[name])[:quota]
        if len(values) != quota:
            raise FT31PublicDefinitionSelectionError(
                f"FT31 stratum {name} 不足 quota")
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
    return FT31PublicDefinitionSelectionManifest(
        manifest.source_key,
        manifest.snapshot_id,
        snapshot_manifest_relative_path,
        snapshot_manifest_sha256,
        index_file.raw_relative_path,
        index_file.compressed_size_bytes,
        index_file.local_sha256,
        index_file.upstream_sha1,
        predecessor_selection_relative_paths,
        predecessor_selection_sha256s,
        excluded_sha,
        index_count,
        sum(eligible_counts.values()),
        tuple((name, eligible_counts[name]) for name, *_ in FT31_STRATA),
        tuple(selected),
    )


def write_ft31_public_definition_selection(
        manifest: FT31PublicDefinitionSelectionManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等发布规范 selection manifest。"""
    if not isinstance(manifest, FT31PublicDefinitionSelectionManifest):
        raise TypeError("FT31 selection manifest 类型错误")
    target = Path(path).resolve()
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FT31PublicDefinitionSelectionError(
                "FT31 selection manifest 已存在且字节不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise FT31PublicDefinitionSelectionError(
            "FT31 selection manifest 无法发布") from error
    return target


def read_ft31_public_definition_selection(
        path: str | Path,
        ) -> FT31PublicDefinitionSelectionManifest:
    """严格回读规范 selection manifest。"""
    try:
        payload = Path(path).resolve().read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise FT31PublicDefinitionSelectionError("FT31 manifest 换行非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        manifest = FT31PublicDefinitionSelectionManifest.from_dict(value)
    except FT31PublicDefinitionSelectionError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise FT31PublicDefinitionSelectionError(
            "FT31 selection manifest 缺失或不可读") from error
    if manifest.canonical_bytes() != payload:
        raise FT31PublicDefinitionSelectionError("FT31 manifest 非规范字节")
    return manifest


__all__ = [
    "FT31_MAX_SELECTED_TITLES",
    "FT31_SELECTION_FORMAT",
    "FT31_SELECTION_RULE",
    "FT31_SELECTION_VERSION",
    "FT31_STRATA",
    "FT31PublicDefinitionSelectionError",
    "FT31PublicDefinitionSelectionManifest",
    "build_ft31_public_definition_selection",
    "read_ft31_public_definition_selection",
    "write_ft31_public_definition_selection",
]
