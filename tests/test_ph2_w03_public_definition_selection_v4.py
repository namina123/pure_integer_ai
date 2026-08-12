"""FT33 public-definition v4 deterministic selection and chain tests."""
from __future__ import annotations

import bz2
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    MediaWikiDumpSnapshotManifest,
)
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v4 import (
    FT33PublicDefinitionSelectionError,
    FT33_STRATA,
    build_ft33_public_definition_selection,
    read_ft33_public_definition_selection,
    write_ft33_public_definition_selection,
)
from pure_integer_ai.experiments.ph2_w03_public_definition_source_catalog_v4 import (
    W03PublicDefinitionSourceCatalogV4Error,
    recover_ft33_public_definition_predecessor_titles,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SELECTION = REPOSITORY / (
    "data/ph2/manifests/ft33_w03_public_definition_selection_v4.json")
SELECTION_SHA256 = (
    "5032e3079eefbc0c6b602d913eccbc7f89f0c8c94466c1e11c8014b770f35092")


@dataclass(frozen=True)
class _Raw:
    """Minimal compressed-file identity required by the selector."""

    role: str
    raw_relative_path: str
    compressed_size_bytes: int
    local_sha256: str
    upstream_sha1: str


def _fake_snapshot(
        tmp_path: Path,
        ) -> tuple[MediaWikiDumpSnapshotManifest, tuple[str, ...]]:
    """Create 202 candidates per stratum and 293 valid exclusions."""
    groups = []
    bases = (0x4e00, 0x5100, 0x5400, 0x5700)
    lengths = (1, 2, 5, 9)
    for base, length in zip(bases, lengths):
        values = []
        for ordinal in range(202):
            prefix = chr(base + ordinal)
            values.append(prefix + chr(base + 300) * (length - 1))
        groups.append(tuple(values))
    excluded = tuple(
        item for group in groups for item in group[-74:]
    )[:293]
    lines = []
    page_id = 1
    offset = 0
    for values in groups:
        for title in values:
            lines.append(f"{offset}:{page_id}:{title}\n")
            page_id += 1
            offset += 10
    index = tmp_path / "index.txt.bz2"
    index.write_bytes(bz2.compress("".join(lines).encode("utf-8")))
    xml = tmp_path / "dump.xml.bz2"
    xml.write_bytes(b"x" * (offset + 10))
    raw_files = (
        _Raw(
            "XML", "dump.xml.bz2", xml.stat().st_size,
            hashlib.sha256(xml.read_bytes()).hexdigest(),
            hashlib.sha1(xml.read_bytes()).hexdigest(),
        ),
        _Raw(
            "INDEX", "index.txt.bz2", index.stat().st_size,
            hashlib.sha256(index.read_bytes()).hexdigest(),
            hashlib.sha1(index.read_bytes()).hexdigest(),
        ),
    )
    snapshot = object.__new__(MediaWikiDumpSnapshotManifest)
    object.__setattr__(snapshot, "source_key", "ZHWIKTIONARY_20260701")
    object.__setattr__(snapshot, "project", "zhwiktionary")
    object.__setattr__(snapshot, "snapshot_id", "synthetic-ft33")
    object.__setattr__(snapshot, "raw_files", raw_files)
    return snapshot, excluded


def test_v4_selection_is_512_stratified_ranked_and_excludes_293(tmp_path):
    """Selection freezes 128 titles per stratum before page parsing."""
    snapshot, excluded = _fake_snapshot(tmp_path)
    manifest = build_ft33_public_definition_selection(
        snapshot,
        raw_root=tmp_path,
        snapshot_manifest_relative_path="data/ph2/manifests/snapshot.json",
        snapshot_manifest_sha256="a" * 64,
        predecessor_selection_relative_paths=(
            "ft26.json", "ft30.json", "ft31.json"),
        predecessor_selection_sha256s=("b" * 64, "c" * 64, "d" * 64),
        excluded_titles=excluded,
    )
    assert len(manifest.selected_titles) == 512
    assert len(manifest.excluded_title_sha256s) == 293
    assert {item.title for item in manifest.selected_titles}.isdisjoint(excluded)
    for name, _, _, quota in FT33_STRATA:
        selected = [item for item in manifest.selected_titles
                    if item.stratum == name]
        assert len(selected) == quota == 128
        assert [item.selection_sha256 for item in selected] == sorted(
            item.selection_sha256 for item in selected)


def test_v4_manifest_is_canonical_and_tamper_evident(tmp_path):
    """Canonical publication and recovery reject quota or count drift."""
    snapshot, excluded = _fake_snapshot(tmp_path)
    manifest = build_ft33_public_definition_selection(
        snapshot,
        raw_root=tmp_path,
        snapshot_manifest_relative_path="snapshot.json",
        snapshot_manifest_sha256="a" * 64,
        predecessor_selection_relative_paths=(
            "ft26.json", "ft30.json", "ft31.json"),
        predecessor_selection_sha256s=("b" * 64, "c" * 64, "d" * 64),
        excluded_titles=excluded,
    )
    target = tmp_path / "selection-v4.json"
    write_ft33_public_definition_selection(manifest, target)
    assert read_ft33_public_definition_selection(target) == manifest
    target.write_bytes(target.read_bytes().replace(
        b'"excluded_title_count":293', b'"excluded_title_count":292'))
    with pytest.raises(FT33PublicDefinitionSelectionError):
        read_ft33_public_definition_selection(target)


def test_v4_public_manifest_recomputes_all_three_predecessors() -> None:
    """The published manifest binds the exact 5+32+256 title chain."""
    raw = SELECTION.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SELECTION_SHA256
    manifest = read_ft33_public_definition_selection(SELECTION)
    titles = recover_ft33_public_definition_predecessor_titles(
        REPOSITORY, manifest)
    assert len(titles) == len(set(titles)) == 293
    assert len(manifest.selected_titles) == 512
    assert set(titles).isdisjoint(item.title for item in manifest.selected_titles)


def test_v4_chain_rejects_a_changed_intermediate_commitment() -> None:
    """Matching terminal bytes cannot bypass FT30/FT31 exclusion checks."""
    manifest = read_ft33_public_definition_selection(SELECTION)
    changed = replace(
        manifest,
        excluded_title_sha256s=tuple(sorted((
            *manifest.excluded_title_sha256s[:-1],
            "f" * 64,
        ))),
    )
    with pytest.raises(
            W03PublicDefinitionSourceCatalogV4Error,
            match="excluded-title commitment"):
        recover_ft33_public_definition_predecessor_titles(
            REPOSITORY, changed)
