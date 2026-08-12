"""FT30 公开定义标题选择器的确定性合同测试。"""
from __future__ import annotations

import bz2
from dataclasses import dataclass
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    MediaWikiDumpSnapshotManifest,
)
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v2 import (
    FT30PublicDefinitionSelectionError,
    FT30_STRATA,
    build_ft30_public_definition_selection,
    read_ft30_public_definition_selection,
    write_ft30_public_definition_selection,
)


@dataclass(frozen=True)
class _Raw:
    """选择器所需的最小压缩文件身份。"""

    role: str
    raw_relative_path: str
    compressed_size_bytes: int
    local_sha256: str
    upstream_sha1: str


def _fake_snapshot(tmp_path: Path) -> tuple[MediaWikiDumpSnapshotManifest, Path]:
    """构造不解析 XML 的合成 snapshot 与 multistream index。"""
    titles = (
        ("甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬"),
        ("春夏", "秋冬", "天地", "日月", "山川", "草木", "风雨", "星辰", "排除"),
        ("公开定义词条", "长距离记忆", "记忆动力学", "语言学习能力", "结构标记体", "代码引用链", "因果泛化式", "数据来源表", "模板审计项"),
        ("这是一个公开定义标题", "这是第二个定义标题", "这是第三个定义标题", "这是第四个定义标题", "这是第五个定义标题", "这是第六个定义标题", "这是第七个定义标题", "这是第八个定义标题", "排除长标题"),
    )
    lines = []
    page_id = 1
    offset = 0
    for values in titles:
        for title in values:
            lines.append(f"{offset}:{page_id}:{title}\n")
            page_id += 1
            offset += 10
    index = tmp_path / "index.txt.bz2"
    index.write_bytes(bz2.compress("".join(lines).encode("utf-8")))
    xml = tmp_path / "dump.xml.bz2"
    xml.write_bytes(b"x" * (offset + 10))
    index_sha = hashlib.sha256(index.read_bytes()).hexdigest()
    index_sha1 = hashlib.sha1(index.read_bytes()).hexdigest()
    raw_files = (
        _Raw("XML", "dump.xml.bz2", xml.stat().st_size, hashlib.sha256(xml.read_bytes()).hexdigest(), hashlib.sha1(xml.read_bytes()).hexdigest()),
        _Raw("INDEX", "index.txt.bz2", index.stat().st_size, index_sha, index_sha1),
    )
    snapshot = object.__new__(MediaWikiDumpSnapshotManifest)
    object.__setattr__(snapshot, "source_key", "ZHWIKTIONARY_20260701")
    object.__setattr__(snapshot, "project", "zhwiktionary")
    object.__setattr__(snapshot, "snapshot_id", "synthetic-ft30")
    object.__setattr__(snapshot, "raw_files", raw_files)
    return snapshot, index


def test_selection_is_stratified_hash_ranked_and_excludes_v1(tmp_path):
    """四层配额、snapshot/title 排名和 v1 标题排除均冻结。"""
    snapshot, _ = _fake_snapshot(tmp_path)
    excluded = ("排除", "排除长标题")
    manifest = build_ft30_public_definition_selection(
        snapshot,
        raw_root=tmp_path,
        snapshot_manifest_relative_path="data/ph2/manifests/snapshot.json",
        snapshot_manifest_sha256="a" * 64,
        base_selection_manifest_relative_path="data/ph2/manifests/base.json",
        base_selection_manifest_sha256="b" * 64,
        excluded_titles=excluded,
    )
    assert len(manifest.selected_titles) == 32
    assert {item.title for item in manifest.selected_titles}.isdisjoint(excluded)
    for name, _, _, quota in FT30_STRATA:
        selected = [item for item in manifest.selected_titles if item.stratum == name]
        assert len(selected) == quota
        assert [item.selection_sha256 for item in selected] == sorted(
            item.selection_sha256 for item in selected)
    assert all(item.compressed_block_end_offset > item.compressed_block_offset
               for item in manifest.selected_titles)


def test_selection_manifest_round_trip_is_canonical_and_tamper_evident(tmp_path):
    """manifest 写入、回读和篡改检测保持字节级确定性。"""
    snapshot, _ = _fake_snapshot(tmp_path)
    manifest = build_ft30_public_definition_selection(
        snapshot,
        raw_root=tmp_path,
        snapshot_manifest_relative_path="snapshot.json",
        snapshot_manifest_sha256="a" * 64,
        base_selection_manifest_relative_path="base.json",
        base_selection_manifest_sha256="b" * 64,
        excluded_titles=("排除", "排除长标题"),
    )
    target = tmp_path / "selection.json"
    write_ft30_public_definition_selection(manifest, target)
    assert read_ft30_public_definition_selection(target) == manifest
    target.write_bytes(target.read_bytes().replace(b'"eligible_title_count":34', b'"eligible_title_count":35'))
    with pytest.raises(FT30PublicDefinitionSelectionError):
        read_ft30_public_definition_selection(target)
