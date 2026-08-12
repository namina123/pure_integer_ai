"""从冻结 Wikipedia multistream index 选择广域问答页面坐标。"""
from __future__ import annotations

import bz2
import hashlib
import heapq
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaSelectedPage,
    BroadQaSelectionManifest,
    parse_selection_manifest,
)
from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    iter_multistream_index,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    MediaWikiDumpSnapshotManifest,
)


# object-model: exception
class BroadQaSelectionError(RuntimeError):
    """Wikipedia 来源、索引、排名或发布身份发生漂移。"""


def _raw_file(manifest: MediaWikiDumpSnapshotManifest, role: str):
    """返回 snapshot 指定角色的唯一压缩文件身份。"""
    values = tuple(item for item in manifest.raw_files if item.role == role)
    if len(values) != 1:
        raise BroadQaSelectionError(f"broad QA {role} raw identity 非唯一")
    return values[0]


def _sha256_path(path: Path) -> str:
    """以固定读取块计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _rank(snapshot_id: str, page_id: int, title: str) -> str:
    """计算只依赖公开 index 身份的稳定页面排名。"""
    return hashlib.sha256(
        f"{snapshot_id}\0{page_id}\0{title}".encode("utf-8"),
    ).hexdigest()


def build_broad_qa_selection(
        manifest: MediaWikiDumpSnapshotManifest,
        *,
        index_path: str | Path,
        snapshot_manifest_sha256: str,
        requested_page_count: int,
        ) -> BroadQaSelectionManifest:
    """单遍扫描冻结 index，选择全局稳定排名最小的页面坐标。"""
    if not isinstance(manifest, MediaWikiDumpSnapshotManifest):
        raise TypeError("broad QA snapshot manifest 类型错误")
    if (manifest.source_key != "ZHWIKIPEDIA_20260701"
            or manifest.project != "zhwiki"):
        raise BroadQaSelectionError("broad QA 只接受冻结中文 Wikipedia")
    if type(requested_page_count) is not int or requested_page_count <= 0:
        raise BroadQaSelectionError("requested page count 必须是正严格整数")
    index = Path(index_path).resolve()
    if not index.is_file():
        raise BroadQaSelectionError("broad QA index 缺失")
    index_file = _raw_file(manifest, "INDEX")
    xml_file = _raw_file(manifest, "XML")
    if (index.stat().st_size != index_file.compressed_size_bytes
            or _sha256_path(index) != index_file.local_sha256):
        raise BroadQaSelectionError("broad QA index size/SHA 漂移")
    heap: list[tuple[int, str, int, int, int]] = []
    offsets: list[int] = []
    prior_offset = -1
    index_count = 0
    with bz2.open(index, "rb") as stream:
        for entry in iter_multistream_index(stream):
            index_count += 1
            if entry.offset != prior_offset:
                offsets.append(entry.offset)
                prior_offset = entry.offset
            rank = _rank(manifest.snapshot_id, entry.page_id, entry.title)
            value = (-int(rank, 16), entry.title, entry.page_id,
                     entry.line_number, entry.offset)
            if len(heap) < requested_page_count:
                heapq.heappush(heap, value)
            elif -value[0] < -heap[0][0]:
                heapq.heapreplace(heap, value)
    if len(heap) != requested_page_count:
        raise BroadQaSelectionError("broad QA index 不足选择数量")
    ends = {
        offset: offsets[ordinal + 1]
        if ordinal + 1 < len(offsets) else xml_file.compressed_size_bytes
        for ordinal, offset in enumerate(offsets)
    }
    ranked = sorted(
        ((format(-item[0], "064x"), *item[1:]) for item in heap),
        key=lambda item: (item[0], item[2], item[1]),
    )
    selected = tuple(
        BroadQaSelectedPage(
            ordinal,
            rank,
            title,
            hashlib.sha256(title.encode("utf-8")).hexdigest(),
            page_id,
            line_number,
            offset,
            ends[offset],
        )
        for ordinal, (rank, title, page_id, line_number, offset)
        in enumerate(ranked, start=1)
    )
    return BroadQaSelectionManifest(
        manifest.source_key,
        manifest.snapshot_id,
        snapshot_manifest_sha256,
        index_file.local_sha256,
        index_file.upstream_sha1,
        xml_file.local_sha256,
        xml_file.compressed_size_bytes,
        index_count,
        requested_page_count,
        selected,
    )


def write_broad_qa_selection(
        manifest: BroadQaSelectionManifest,
        path: str | Path,
        ) -> Path:
    """独占发布选择 manifest，已有相同字节时保持幂等。"""
    if not isinstance(manifest, BroadQaSelectionManifest):
        raise TypeError("broad QA selection manifest 类型错误")
    target = Path(path).resolve()
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise BroadQaSelectionError("broad QA selection 已存在且字节不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)
    return target


def read_broad_qa_selection(path: str | Path) -> BroadQaSelectionManifest:
    """严格读取并复核规范选择 manifest。"""
    try:
        return parse_selection_manifest(Path(path).resolve().read_bytes())
    except BroadQaSelectionError:
        raise
    except Exception as error:
        raise BroadQaSelectionError("broad QA selection 不可读") from error


def derive_broad_qa_selection_prefix(
        parent: BroadQaSelectionManifest,
        *,
        requested_page_count: int,
        ) -> BroadQaSelectionManifest:
    """从已封存稳定 selection 取排名前缀，不重扫 index 或读取正文。"""
    if not isinstance(parent, BroadQaSelectionManifest):
        raise TypeError("broad QA parent selection manifest 类型错误")
    if (type(requested_page_count) is not int
            or requested_page_count <= 0
            or requested_page_count > parent.requested_page_count):
        raise BroadQaSelectionError("broad QA prefix count 非法")
    pages = parent.selected_pages[:requested_page_count]
    if tuple(item.ordinal for item in pages) != tuple(
            range(1, requested_page_count + 1)):
        raise BroadQaSelectionError("broad QA prefix ordinal 漂移")
    return BroadQaSelectionManifest(
        parent.source_key,
        parent.snapshot_id,
        parent.snapshot_manifest_sha256,
        parent.index_local_sha256,
        parent.index_upstream_sha1,
        parent.xml_local_sha256,
        parent.xml_compressed_size_bytes,
        parent.index_entry_count,
        requested_page_count,
        pages,
    )


def profile_broad_qa_selection(
        manifest: BroadQaSelectionManifest,
        ) -> dict[str, int | str]:
    """在不读取 XML 的前提下汇总候选压缩块和读取字节预算。"""
    if not isinstance(manifest, BroadQaSelectionManifest):
        raise TypeError("broad QA selection manifest 类型错误")
    blocks: dict[tuple[int, int], int] = {}
    for item in manifest.selected_pages:
        key = (item.compressed_block_offset,
               item.compressed_block_end_offset)
        blocks[key] = blocks.get(key, 0) + 1
    compressed_bytes = sum(end - start for start, end in blocks)
    return {
        "candidate_page_count": len(manifest.selected_pages),
        "compressed_block_count": len(blocks),
        "compressed_bytes_read": compressed_bytes,
        "compressed_xml_touch_ppm": (
            compressed_bytes * 1_000_000
            // manifest.xml_compressed_size_bytes
        ),
        "maximum_candidates_per_block": max(blocks.values()),
        "reused_candidate_block_count": sum(
            1 for count in blocks.values() if count > 1),
        "selection_sha256": manifest.sha256(),
    }


__all__ = [
    "BroadQaSelectionError",
    "build_broad_qa_selection",
    "derive_broad_qa_selection_prefix",
    "profile_broad_qa_selection",
    "read_broad_qa_selection",
    "write_broad_qa_selection",
]
