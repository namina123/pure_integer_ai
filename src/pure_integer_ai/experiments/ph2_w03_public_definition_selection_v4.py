"""FT33 Wiktionary public-definition 512-title deterministic selection."""
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


FT33_SELECTION_FORMAT = "PH2_FT33_PUBLIC_DEFINITION_SELECTION"
FT33_SELECTION_VERSION = 4
FT33_SELECTION_RULE = "SNAPSHOT_NUL_TITLE_SHA256_LENGTH_STRATA_V1"
FT33_TITLE_FILTER = "CJK_UNIFIED_IDEOGRAPHS_U4E00_U9FFF"
FT33_MAX_SELECTED_TITLES = 512
FT33_EXCLUDED_TITLE_COUNT = 293
FT33_STRATA = (
    ("L1", 1, 1, 128),
    ("L2_4", 2, 4, 128),
    ("L5_8", 5, 8, 128),
    ("L9_16", 9, 16, 128),
)


# object-model: exception
class FT33PublicDefinitionSelectionError(RuntimeError):
    """Snapshot, predecessor, ranking, or canonical bytes drifted."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _raw_file(manifest: MediaWikiDumpSnapshotManifest, role: str):
    values = tuple(item for item in manifest.raw_files if item.role == role)
    if len(values) != 1:
        raise FT33PublicDefinitionSelectionError(
            f"FT33 snapshot {role} raw file is not unique")
    return values[0]


def _safe_path(root: Path, relative_path: str) -> Path:
    path = (root / Path(*relative_path.split("/"))).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FT33PublicDefinitionSelectionError(
            "FT33 raw path is missing or escaped")
    return path


def _is_selected_title_form(title: str) -> bool:
    return bool(title) and all("\u4e00" <= item <= "\u9fff" for item in title)


def _stratum_for_length(length: int) -> str | None:
    for name, minimum, maximum, _ in FT33_STRATA:
        if minimum <= length <= maximum:
            return name
    return None


def _selection_sha256(snapshot_id: str, title: str) -> str:
    return hashlib.sha256(
        (snapshot_id + "\0" + title).encode("utf-8")).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(item in "0123456789abcdef" for item in value)
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FT33PublicDefinitionSelectionManifest:
    """Bind the snapshot, three predecessors, exclusions, and 512 pages."""

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
            raise FT33PublicDefinitionSelectionError("FT33 source key drifted")
        paths = (
            self.snapshot_id,
            self.snapshot_manifest_relative_path,
            self.index_raw_relative_path,
            *self.predecessor_selection_relative_paths,
        )
        if any(
                not isinstance(item, str) or not item or "\\" in item
                or item.startswith(("/", "../")) for item in paths):
            raise FT33PublicDefinitionSelectionError(
                "FT33 path or identity is invalid")
        hashes = (
            self.snapshot_manifest_sha256,
            self.index_local_sha256,
            *self.predecessor_selection_sha256s,
            *self.excluded_title_sha256s,
        )
        if any(not _valid_sha256(item) for item in hashes):
            raise FT33PublicDefinitionSelectionError("FT33 SHA-256 is invalid")
        if (
            len(self.predecessor_selection_relative_paths) != 3
            or len(self.predecessor_selection_sha256s) != 3
            or len(self.excluded_title_sha256s) != FT33_EXCLUDED_TITLE_COUNT
            or tuple(sorted(set(self.excluded_title_sha256s)))
            != self.excluded_title_sha256s
        ):
            raise FT33PublicDefinitionSelectionError(
                "FT33 predecessor or exclusion inventory drifted")
        if (
            not isinstance(self.index_upstream_sha1, str)
            or len(self.index_upstream_sha1) != 40
            or any(item not in "0123456789abcdef"
                   for item in self.index_upstream_sha1)
        ):
            raise FT33PublicDefinitionSelectionError("FT33 index SHA-1 is invalid")
        for name in (
                "index_compressed_size_bytes", "index_entry_count",
                "eligible_title_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise FT33PublicDefinitionSelectionError(
                    f"FT33 {name} is not a positive integer")
        names = tuple(item[0] for item in FT33_STRATA)
        expected_counts = tuple((item[0], item[3]) for item in FT33_STRATA)
        actual_counts = tuple(
            (name, sum(item.stratum == name for item in self.selected_titles))
            for name, _ in expected_counts)
        if actual_counts != expected_counts:
            raise FT33PublicDefinitionSelectionError(
                "FT33 stratum quotas drifted")
        if (
            tuple(name for name, _ in self.eligible_counts_by_stratum) != names
            or any(type(count) is not int or count < 128
                   for _, count in self.eligible_counts_by_stratum)
            or sum(count for _, count in self.eligible_counts_by_stratum)
            != self.eligible_title_count
        ):
            raise FT33PublicDefinitionSelectionError(
                "FT33 eligible counts drifted")
        if (
            len(self.selected_titles) != FT33_MAX_SELECTED_TITLES
            or len({item.title for item in self.selected_titles})
            != FT33_MAX_SELECTED_TITLES
        ):
            raise FT33PublicDefinitionSelectionError(
                "FT33 selected-title inventory drifted")
        ordered = tuple(sorted(
            self.selected_titles,
            key=lambda item: (
                names.index(item.stratum), item.selection_sha256,
                item.title, item.page_id,
            ),
        ))
        if ordered != self.selected_titles:
            raise FT33PublicDefinitionSelectionError(
                "FT33 selected-title order drifted")
        excluded = set(self.excluded_title_sha256s)
        for item in self.selected_titles:
            if (
                item.selection_sha256
                != _selection_sha256(self.snapshot_id, item.title)
                or item.title_sha256 in excluded
            ):
                raise FT33PublicDefinitionSelectionError(
                    "FT33 selected title rank or exclusion drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": FT33_SELECTION_FORMAT,
            "eligible_counts_by_stratum": dict(
                self.eligible_counts_by_stratum),
            "eligible_title_count": self.eligible_title_count,
            "excluded_title_count": FT33_EXCLUDED_TITLE_COUNT,
            "excluded_title_sha256s": list(self.excluded_title_sha256s),
            "format_version": 1,
            "index_compressed_size_bytes": self.index_compressed_size_bytes,
            "index_entry_count": self.index_entry_count,
            "index_local_sha256": self.index_local_sha256,
            "index_raw_relative_path": self.index_raw_relative_path,
            "index_upstream_sha1": self.index_upstream_sha1,
            "max_selected_titles": FT33_MAX_SELECTED_TITLES,
            "predecessor_selection_relative_paths": list(
                self.predecessor_selection_relative_paths),
            "predecessor_selection_sha256s": list(
                self.predecessor_selection_sha256s),
            "selected_titles": [item.to_dict() for item in self.selected_titles],
            "selection_rule": FT33_SELECTION_RULE,
            "selection_version": FT33_SELECTION_VERSION,
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
                for name, minimum, maximum, quota in FT33_STRATA
            ],
            "title_filter": FT33_TITLE_FILTER,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "FT33PublicDefinitionSelectionManifest":
        keys = {
            "artifact_kind", "eligible_counts_by_stratum",
            "eligible_title_count", "excluded_title_count",
            "excluded_title_sha256s", "format_version",
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
            raise FT33PublicDefinitionSelectionError(
                "FT33 manifest fields drifted")
        if (
            value["artifact_kind"] != FT33_SELECTION_FORMAT
            or value["format_version"] != 1
            or value["selection_version"] != FT33_SELECTION_VERSION
            or value["selection_rule"] != FT33_SELECTION_RULE
            or value["title_filter"] != FT33_TITLE_FILTER
            or value["max_selected_titles"] != FT33_MAX_SELECTED_TITLES
            or value["excluded_title_count"] != FT33_EXCLUDED_TITLE_COUNT
        ):
            raise FT33PublicDefinitionSelectionError(
                "FT33 manifest contract drifted")
        expected_strata = [
            {
                "maximum_title_length": maximum,
                "minimum_title_length": minimum,
                "quota": quota,
                "stratum": name,
            }
            for name, minimum, maximum, quota in FT33_STRATA
        ]
        eligible = value["eligible_counts_by_stratum"]
        list_fields = (
            "excluded_title_sha256s",
            "predecessor_selection_relative_paths",
            "predecessor_selection_sha256s", "selected_titles",
        )
        if (
            value["strata"] != expected_strata
            or not isinstance(eligible, dict)
            or set(eligible) != {item[0] for item in FT33_STRATA}
            or not all(isinstance(value[name], list) for name in list_fields)
        ):
            raise FT33PublicDefinitionSelectionError(
                "FT33 manifest lists drifted")
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
            tuple((name, eligible[name]) for name, *_ in FT33_STRATA),
            tuple(FT30SelectedTitle.from_dict(item)
                  for item in value["selected_titles"]),
        )


def build_ft33_public_definition_selection(
        manifest: MediaWikiDumpSnapshotManifest,
        *,
        raw_root: str | Path,
        snapshot_manifest_relative_path: str,
        snapshot_manifest_sha256: str,
        predecessor_selection_relative_paths: tuple[str, str, str],
        predecessor_selection_sha256s: tuple[str, str, str],
        excluded_titles: tuple[str, ...],
        ) -> FT33PublicDefinitionSelectionManifest:
    """Scan only the index and select 128 new titles per fixed stratum."""
    if not isinstance(manifest, MediaWikiDumpSnapshotManifest):
        raise TypeError("FT33 snapshot manifest type mismatch")
    if (
        manifest.source_key != "ZHWIKTIONARY_20260701"
        or manifest.project != "zhwiktionary"
    ):
        raise FT33PublicDefinitionSelectionError(
            "FT33 accepts only the frozen Wiktionary snapshot")
    if (
        not isinstance(excluded_titles, tuple)
        or len(excluded_titles) != FT33_EXCLUDED_TITLE_COUNT
        or any(not isinstance(item, str) or not item for item in excluded_titles)
        or len(set(excluded_titles)) != FT33_EXCLUDED_TITLE_COUNT
    ):
        raise FT33PublicDefinitionSelectionError(
            "FT33 excluded-title inventory is invalid")
    raw = Path(raw_root).resolve()
    index_file = _raw_file(manifest, "INDEX")
    index_path = _safe_path(raw, index_file.raw_relative_path)
    if (
        index_path.stat().st_size != index_file.compressed_size_bytes
        or _sha256_path(index_path) != index_file.local_sha256
    ):
        raise FT33PublicDefinitionSelectionError("FT33 index size/SHA drifted")
    excluded_set = set(excluded_titles)
    excluded_sha = tuple(sorted(
        hashlib.sha256(item.encode("utf-8")).hexdigest()
        for item in excluded_titles))
    candidates: dict[str, list[tuple[str, str, int, int, int]]] = {
        item[0]: [] for item in FT33_STRATA}
    eligible_counts = {item[0]: 0 for item in FT33_STRATA}
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
            values = candidates[stratum]
            values.append((
                _selection_sha256(manifest.snapshot_id, entry.title),
                entry.title, entry.page_id, entry.line_number, entry.offset,
            ))
            quota = next(item[3] for item in FT33_STRATA
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
    for name, _, _, quota in FT33_STRATA:
        values = sorted(candidates[name])[:quota]
        if len(values) != quota:
            raise FT33PublicDefinitionSelectionError(
                f"FT33 stratum {name} has fewer than {quota} titles")
        for rank, title, page_id, line_number, offset in values:
            selected.append(FT30SelectedTitle(
                name, title, len(title),
                hashlib.sha256(title.encode("utf-8")).hexdigest(), rank,
                page_id, line_number, offset, offset_end[offset],
            ))
    return FT33PublicDefinitionSelectionManifest(
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
        tuple((name, eligible_counts[name]) for name, *_ in FT33_STRATA),
        tuple(selected),
    )


def write_ft33_public_definition_selection(
        manifest: FT33PublicDefinitionSelectionManifest,
        path: str | Path,
        ) -> Path:
    """Publish canonical bytes exclusively or accept an identical file."""
    if not isinstance(manifest, FT33PublicDefinitionSelectionManifest):
        raise TypeError("FT33 selection manifest type mismatch")
    target = Path(path).resolve()
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FT33PublicDefinitionSelectionError(
                "FT33 selection already exists with different bytes")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise FT33PublicDefinitionSelectionError(
            "FT33 selection cannot be published") from error
    return target


def read_ft33_public_definition_selection(
        path: str | Path,
        ) -> FT33PublicDefinitionSelectionManifest:
    """Strictly recover a canonical FT33 selection manifest."""
    try:
        payload = Path(path).resolve().read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise FT33PublicDefinitionSelectionError(
                "FT33 selection newline drifted")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        manifest = FT33PublicDefinitionSelectionManifest.from_dict(value)
    except FT33PublicDefinitionSelectionError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise FT33PublicDefinitionSelectionError(
            "FT33 selection is missing or unreadable") from error
    if manifest.canonical_bytes() != payload:
        raise FT33PublicDefinitionSelectionError(
            "FT33 selection bytes are not canonical")
    return manifest


__all__ = [
    "FT33_EXCLUDED_TITLE_COUNT",
    "FT33_MAX_SELECTED_TITLES",
    "FT33_SELECTION_FORMAT",
    "FT33_SELECTION_RULE",
    "FT33_SELECTION_VERSION",
    "FT33_STRATA",
    "FT33PublicDefinitionSelectionError",
    "FT33PublicDefinitionSelectionManifest",
    "build_ft33_public_definition_selection",
    "read_ft33_public_definition_selection",
    "write_ft33_public_definition_selection",
]
