"""FT31 公开定义规模切片选择器的确定性合同测试。"""
from __future__ import annotations

import bz2
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    MediaWikiDumpSnapshotManifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v2 import (
    read_ft30_public_definition_selection,
    write_ft30_public_definition_selection,
)
from pure_integer_ai.experiments.ph2_w03_public_definition_source_catalog_v3 import (
    W03PublicDefinitionSourceCatalogV3Error,
    verify_ft31_public_definition_predecessors,
)
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v3 import (
    FT31PublicDefinitionSelectionError,
    FT31_STRATA,
    build_ft31_public_definition_selection,
    read_ft31_public_definition_selection,
    write_ft31_public_definition_selection,
)


@dataclass(frozen=True)
class _Raw:
    """选择器所需的最小压缩文件身份。"""

    role: str
    raw_relative_path: str
    compressed_size_bytes: int
    local_sha256: str
    upstream_sha1: str


def _fake_snapshot(tmp_path: Path) -> MediaWikiDumpSnapshotManifest:
    """构造每层 66 个候选和两个历史排除标题。"""
    groups = []
    bases = (0x4e00, 0x5100, 0x5400, 0x5700)
    lengths = (1, 2, 5, 9)
    for base, length in zip(bases, lengths):
        values = []
        for ordinal in range(66):
            prefix = chr(base + ordinal)
            values.append(prefix + chr(base + 100) * (length - 1))
        groups.append(tuple(values))
    excluded = tuple(group[-1] for group in groups)
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
        _Raw("XML", "dump.xml.bz2", xml.stat().st_size,
             hashlib.sha256(xml.read_bytes()).hexdigest(),
             hashlib.sha1(xml.read_bytes()).hexdigest()),
        _Raw("INDEX", "index.txt.bz2", index.stat().st_size,
             hashlib.sha256(index.read_bytes()).hexdigest(),
             hashlib.sha1(index.read_bytes()).hexdigest()),
    )
    snapshot = object.__new__(MediaWikiDumpSnapshotManifest)
    object.__setattr__(snapshot, "source_key", "ZHWIKTIONARY_20260701")
    object.__setattr__(snapshot, "project", "zhwiktionary")
    object.__setattr__(snapshot, "snapshot_id", "synthetic-ft31")
    object.__setattr__(snapshot, "raw_files", raw_files)
    object.__setattr__(snapshot, "excluded", excluded)
    return snapshot


def test_v3_selection_is_256_stratified_ranked_and_excludes_history(tmp_path):
    """每层 64、同一稳定排名和历史标题排除全部冻结。"""
    snapshot = _fake_snapshot(tmp_path)
    manifest = build_ft31_public_definition_selection(
        snapshot,
        raw_root=tmp_path,
        snapshot_manifest_relative_path="data/ph2/manifests/snapshot.json",
        snapshot_manifest_sha256="a" * 64,
        predecessor_selection_relative_paths=("ft26.json", "ft30.json"),
        predecessor_selection_sha256s=("b" * 64, "c" * 64),
        excluded_titles=snapshot.excluded,
    )
    assert len(manifest.selected_titles) == 256
    assert {item.title for item in manifest.selected_titles}.isdisjoint(
        snapshot.excluded)
    for name, _, _, quota in FT31_STRATA:
        selected = [item for item in manifest.selected_titles
                    if item.stratum == name]
        assert len(selected) == quota
        assert [item.selection_sha256 for item in selected] == sorted(
            item.selection_sha256 for item in selected)


def test_v3_manifest_round_trip_is_canonical_and_tamper_evident(tmp_path):
    """发布、回读与 predecessor 篡改均保持 fail closed。"""
    snapshot = _fake_snapshot(tmp_path)
    manifest = build_ft31_public_definition_selection(
        snapshot,
        raw_root=tmp_path,
        snapshot_manifest_relative_path="snapshot.json",
        snapshot_manifest_sha256="a" * 64,
        predecessor_selection_relative_paths=("ft26.json", "ft30.json"),
        predecessor_selection_sha256s=("b" * 64, "c" * 64),
        excluded_titles=snapshot.excluded,
    )
    target = tmp_path / "selection-v3.json"
    write_ft31_public_definition_selection(manifest, target)
    assert read_ft31_public_definition_selection(target) == manifest
    target.write_bytes(target.read_bytes().replace(
        b'"max_selected_titles":256', b'"max_selected_titles":255'))
    with pytest.raises(FT31PublicDefinitionSelectionError):
        read_ft31_public_definition_selection(target)


def test_v3_predecessor_chain_recomputes_the_complete_exclusion_set(
        tmp_path: Path,
        ) -> None:
    """Matching predecessor SHAs cannot hide a changed historical title set."""
    repository = Path(__file__).resolve().parents[1]
    manifest_root = tmp_path / "data/ph2/manifests"
    manifest_root.mkdir(parents=True)
    ft26_relative = (
        "data/ph2/manifests/ft26_w03_public_sense_source_selection_v1.json")
    ft30_relative = (
        "data/ph2/manifests/ft30_w03_public_definition_selection_v2.json")
    ft31 = read_ft31_public_definition_selection(
        repository
        / "data/ph2/manifests/ft31_w03_public_definition_selection_v3.json")
    ft26_value = json.loads(
        (repository / ft26_relative).read_text(encoding="utf-8"))
    title_slice = next(
        item for item in ft26_value["source_slices"]
        if item["source_key"] == "ZHWIKTIONARY_20260701")
    title_slice["selections"][0] = "新"
    ft26_target = tmp_path / ft26_relative
    ft26_target.write_bytes(canonical_json_line(ft26_value))
    ft26_sha256 = hashlib.sha256(ft26_target.read_bytes()).hexdigest()

    ft30 = replace(
        read_ft30_public_definition_selection(repository / ft30_relative),
        base_selection_manifest_sha256=ft26_sha256,
    )
    ft30_target = tmp_path / ft30_relative
    write_ft30_public_definition_selection(ft30, ft30_target)
    changed = replace(
        ft31,
        predecessor_selection_sha256s=(
            ft26_sha256,
            hashlib.sha256(ft30_target.read_bytes()).hexdigest(),
        ),
    )
    with pytest.raises(
            W03PublicDefinitionSourceCatalogV3Error,
            match="excluded-title commitment"):
        verify_ft31_public_definition_predecessors(tmp_path, changed)
