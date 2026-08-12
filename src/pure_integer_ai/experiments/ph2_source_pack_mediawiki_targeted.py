"""从冻结 MediaWiki multistream snapshot 定位少量指定页面。"""
from __future__ import annotations

import bz2
from bisect import bisect_right
import hashlib
import io
from pathlib import Path
import xml.etree.ElementTree as ET

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    MediaWikiPageError,
    MediaWikiScanBudget,
    iter_multistream_index,
    parse_mediawiki_page,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    MediaWikiDumpSnapshotManifest,
)
from pure_integer_ai.experiments.ph2_source_pack_contract import (
    SourceObservationSeed,
)
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v2 import (
    FT30PublicDefinitionSelectionManifest,
    FT30SelectedTitle,
)
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v3 import (
    FT31PublicDefinitionSelectionManifest,
)
from pure_integer_ai.experiments.ph2_w03_public_definition_selection_v4 import (
    FT33PublicDefinitionSelectionManifest,
)


# object-model: exception
class TargetedMediaWikiSourceError(RuntimeError):
    """目标页、压缩 block 或冻结来源身份不一致。"""


def _local_name(tag: str) -> str:
    """去除 ElementTree namespace。"""
    return tag.rsplit("}", 1)[-1]


def _raw_file(manifest: MediaWikiDumpSnapshotManifest, role: str):
    """返回 snapshot 中指定角色的唯一 raw file。"""
    values = tuple(item for item in manifest.raw_files if item.role == role)
    if len(values) != 1:
        raise TargetedMediaWikiSourceError(
            f"MediaWiki snapshot {role} raw file 非唯一")
    return values[0]


def _safe_raw_path(root: Path, relative_path: str) -> Path:
    """解析 raw 相对路径并拒绝缺失或目录逃逸。"""
    path = (root / Path(*relative_path.split("/"))).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise TargetedMediaWikiSourceError("MediaWiki raw path 缺失或逃逸")
    return path


def _digest_pair(path: Path) -> tuple[str, str]:
    """单遍计算压缩文件 SHA-256 与 SHA-1。"""
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            sha256.update(block)
            sha1.update(block)
    return sha256.hexdigest(), sha1.hexdigest()


def verify_targeted_mediawiki_raw_identities(
        manifest: MediaWikiDumpSnapshotManifest,
        *,
        raw_root: str | Path,
        ) -> tuple[tuple[str, str, str], ...]:
    """复核 XML/INDEX 的本地 SHA-256 与上游 SHA-1，不解析页面。"""
    if not isinstance(manifest, MediaWikiDumpSnapshotManifest):
        raise TypeError("MediaWiki snapshot manifest 类型错误")
    root = Path(raw_root).resolve()
    results = []
    for role in ("XML", "INDEX"):
        raw_file = _raw_file(manifest, role)
        path = _safe_raw_path(root, raw_file.raw_relative_path)
        if path.stat().st_size != raw_file.compressed_size_bytes:
            raise TargetedMediaWikiSourceError(
                f"MediaWiki {role} raw size 漂移")
        local_sha256, upstream_sha1 = _digest_pair(path)
        if (local_sha256 != raw_file.local_sha256
                or upstream_sha1 != raw_file.upstream_sha1):
            raise TargetedMediaWikiSourceError(
                f"MediaWiki {role} 双 hash 漂移")
        results.append((role, local_sha256, upstream_sha1))
    return tuple(results)


def _index_targets(
        index_path: Path,
        titles: tuple[str, ...],
        ) -> tuple[dict[str, tuple[int, int]], tuple[int, ...]]:
    """完整验证 index，并返回目标 title 的 page id/压缩 offset。"""
    requested = set(titles)
    found: dict[str, tuple[int, int]] = {}
    offsets: list[int] = []
    prior_offset = -1
    with bz2.open(index_path, "rb") as stream:
        for entry in iter_multistream_index(stream):
            if entry.offset != prior_offset:
                offsets.append(entry.offset)
                prior_offset = entry.offset
            if entry.title in requested:
                if entry.title in found:
                    raise TargetedMediaWikiSourceError(
                        "MediaWiki index 目标 title 重复")
                found[entry.title] = (entry.page_id, entry.offset)
    missing = tuple(title for title in titles if title not in found)
    if missing:
        raise TargetedMediaWikiSourceError(
            f"MediaWiki index 缺少目标页: {missing!r}")
    return found, tuple(offsets)


def _page_text(page: ET.Element) -> str:
    """恢复目标 page 唯一 revision text。"""
    values = tuple(
        element for element in page.iter()
        if _local_name(element.tag) == "text"
    )
    if len(values) != 1:
        raise TargetedMediaWikiSourceError("MediaWiki page text 数量非法")
    return values[0].text or ""


def _block_pages(payload: bytes) -> tuple[ET.Element, ...]:
    """把独立 bzip2 multistream block 恢复成 page 元素。"""
    stripped = payload.replace(b"</mediawiki>", b"")
    wrapped = b"<mediawiki>" + stripped + b"</mediawiki>"
    try:
        root = ET.parse(io.BytesIO(wrapped)).getroot()
    except ET.ParseError as error:
        raise TargetedMediaWikiSourceError(
            "MediaWiki multistream block XML 非法") from error
    return tuple(
        element for element in root
        if _local_name(element.tag) == "page"
    )


def _length_bucket(size_bytes: int) -> str:
    """按冻结整数边界标注目标页长度。"""
    if size_bytes <= 1024:
        return "SHORT"
    if size_bytes <= 16 * 1024:
        return "MEDIUM"
    return "LONG"


def targeted_mediawiki_source_seeds(
        manifest: MediaWikiDumpSnapshotManifest,
        *,
        raw_root: str | Path,
        titles: tuple[str, ...],
        split: str = "train",
        ) -> tuple[SourceObservationSeed, ...]:
    """按 index block 读取指定主空间页，不遍历完整 XML 页面流。"""
    if not isinstance(manifest, MediaWikiDumpSnapshotManifest):
        raise TypeError("MediaWiki snapshot manifest 类型错误")
    if (not isinstance(titles, tuple) or not titles
            or any(not isinstance(item, str) or not item
                   or item.strip() != item for item in titles)
            or len(set(titles)) != len(titles)):
        raise TargetedMediaWikiSourceError("MediaWiki 目标 title 非法")
    if split not in {"train", "dev", "held_out"}:
        raise TargetedMediaWikiSourceError("MediaWiki 目标 split 非法")
    raw_root_path = Path(raw_root).resolve()
    xml_file = _raw_file(manifest, "XML")
    index_file = _raw_file(manifest, "INDEX")
    xml_path = _safe_raw_path(raw_root_path, xml_file.raw_relative_path)
    index_path = _safe_raw_path(raw_root_path, index_file.raw_relative_path)
    for raw_file, path in ((xml_file, xml_path), (index_file, index_path)):
        if path.stat().st_size != raw_file.compressed_size_bytes:
            raise TargetedMediaWikiSourceError("MediaWiki raw size 漂移")
    targets, offsets = _index_targets(index_path, titles)
    by_offset: dict[int, tuple[ET.Element, ...]] = {}
    with xml_path.open("rb") as stream:
        for offset in sorted({targets[title][1] for title in titles}):
            position = bisect_right(offsets, offset)
            next_offset = (
                offsets[position] if position < len(offsets)
                else xml_path.stat().st_size)
            if next_offset <= offset:
                raise TargetedMediaWikiSourceError(
                    "MediaWiki block offset 非递增")
            stream.seek(offset)
            compressed = stream.read(next_offset - offset)
            try:
                payload = bz2.decompress(compressed)
            except OSError as error:
                raise TargetedMediaWikiSourceError(
                    "MediaWiki 目标 block 解压失败") from error
            by_offset[offset] = _block_pages(payload)
    budget = MediaWikiScanBudget(
        max_pages=max(64, len(titles) * 32),
        max_xml_events=max(4096, len(titles) * 2048),
        max_text_bytes_per_page=64 * 1024 * 1024,
        max_templates_per_page=1,
        max_template_depth=1,
    )
    domain = "lexicon" if manifest.project == "zhwiktionary" else "encyclopedia"
    genre = "dictionary_entry" if manifest.project == "zhwiktionary" else "article"
    seeds = []
    for ordinal, title in enumerate(titles, start=1):
        expected_page_id, offset = targets[title]
        matches = []
        for page in by_offset[offset]:
            try:
                record = parse_mediawiki_page(
                    page,
                    source_key=manifest.source_key,
                    extract_templates=False,
                    budget=budget,
                )
            except MediaWikiPageError as error:
                if error.code == "NON_MAIN_NAMESPACE":
                    continue
                raise TargetedMediaWikiSourceError(
                    f"MediaWiki 目标页解析失败: {error.code}") from error
            if record.page_id == expected_page_id and record.title == title:
                matches.append((record, _page_text(page)))
        if len(matches) != 1:
            raise TargetedMediaWikiSourceError(
                "MediaWiki block 未唯一恢复目标页")
        record, text = matches[0]
        length = _length_bucket(record.text_size_bytes)
        axes = {
            "code_switch": "UNASSESSED",
            "dialect": "UNASSESSED",
            "domain": domain,
            "era": manifest.dump_date,
            "genre": genre,
            "language": "zh",
            "length": length,
            "register": "UNASSESSED",
            "script_orthography": "ZH_WIKIMEDIA_RAW",
            "source": manifest.source_key,
            "source_document_cluster": f"page:{record.page_id}",
        }
        contributor = record.contributor.to_value()
        raw_observation = {
            "contributor": contributor,
            "page_id": record.page_id,
            "redirect_title": record.redirect_title,
            "revision_id": record.revision_id,
            "text": text,
            "timestamp": record.timestamp,
            "title": record.title,
        }
        seeds.append(SourceObservationSeed(
            f"page-{record.page_id}-revision-{record.revision_id}",
            split,
            "zh",
            "mediawiki-raw-wikitext",
            (
                f"{xml_file.raw_relative_path}#page={record.page_id}"
                f";revision={record.revision_id}"
            ),
            "sha1:" + xml_file.upstream_sha1,
            xml_file.local_sha256,
            CanonicalJsonObject.from_value({
                "compressed_block_offset": offset,
                "compressed_raw_relative_path": xml_file.raw_relative_path,
                "contributor": contributor,
                "index_local_sha256": index_file.local_sha256,
                "index_raw_relative_path": index_file.raw_relative_path,
                "namespace_id": record.namespace_id,
                "page_id": record.page_id,
                "revision_id": record.revision_id,
                "text_sha256": record.text_sha256,
                "timestamp": record.timestamp,
            }),
            CanonicalJsonObject.from_value(raw_observation),
            CanonicalJsonObject.from_value(axes),
            ("page", record.page_id),
            ("text", record.text_sha256),
            ("page", record.page_id, record.revision_id),
            (
                "page", record.page_id,
                "redirect" if record.redirect_title else "article",
                domain,
            ),
            (
                "page", record.page_id,
                "length", length,
                "line_count", text.count("\n") + 1,
            ),
            tuple(
                value
                for key in sorted(axes)
                for value in (key, axes[key])
            ),
            "support" if split == "train" else "read_only_probe",
            "NONE",
            ordinal,
        ))
    return tuple(seeds)


def _selected_pages_from_blocks(
        xml_path: Path,
        selected: tuple[FT30SelectedTitle, ...],
        ) -> dict[tuple[int, str], tuple[ET.Element, int, int]]:
    """只解压 manifest 命中的 block，并恢复所选 page 元素。"""
    blocks: dict[tuple[int, int], tuple[ET.Element, ...]] = {}
    with xml_path.open("rb") as stream:
        for start, end in sorted({
                (item.compressed_block_offset,
                 item.compressed_block_end_offset)
                for item in selected}):
            if end > xml_path.stat().st_size:
                raise TargetedMediaWikiSourceError(
                    "FT30 compressed block 超出 XML 边界")
            stream.seek(start)
            compressed = stream.read(end - start)
            if len(compressed) != end - start:
                raise TargetedMediaWikiSourceError(
                    "FT30 compressed block 读取不完整")
            try:
                payload = bz2.decompress(compressed)
            except OSError as error:
                raise TargetedMediaWikiSourceError(
                    "FT30 目标 block 解压失败") from error
            blocks[(start, end)] = _block_pages(payload)
    result: dict[tuple[int, str], tuple[ET.Element, int, int]] = {}
    for target in selected:
        block_key = (
            target.compressed_block_offset,
            target.compressed_block_end_offset,
        )
        matches = []
        for page in blocks[block_key]:
            page_ids = tuple(
                element for element in page
                if _local_name(element.tag) == "id")
            titles = tuple(
                element for element in page
                if _local_name(element.tag) == "title")
            if (len(page_ids) == len(titles) == 1
                    and page_ids[0].text == str(target.page_id)
                    and titles[0].text == target.title):
                matches.append(page)
        if len(matches) != 1:
            raise TargetedMediaWikiSourceError(
                "FT30 block 未唯一恢复 manifest 目标页")
        key = target.page_id, target.title
        if key in result:
            raise TargetedMediaWikiSourceError("FT30 目标页重复")
        result[key] = (matches[0], *block_key)
    return result


def targeted_mediawiki_source_seeds_from_selection_v2(
        manifest: MediaWikiDumpSnapshotManifest,
        selection: (
            FT30PublicDefinitionSelectionManifest
            | FT31PublicDefinitionSelectionManifest
            | FT33PublicDefinitionSelectionManifest),
        *,
        raw_root: str | Path,
        selection_manifest_relative_path: str,
        selection_manifest_sha256: str,
        split: str = "train",
        max_text_bytes_per_page: int = 2 * 1024 * 1024,
        max_templates_per_page: int = 4096,
        max_template_depth: int = 64,
        ) -> tuple[SourceObservationSeed, ...]:
    """按冻结 v2 坐标直接读 block，禁止在正式构建中重扫 index。"""
    if (not isinstance(manifest, MediaWikiDumpSnapshotManifest)
            or not isinstance(selection, (
                FT30PublicDefinitionSelectionManifest,
                FT31PublicDefinitionSelectionManifest,
                FT33PublicDefinitionSelectionManifest))):
        raise TypeError("FT30 MediaWiki manifest/selection 类型错误")
    if (selection.source_key != manifest.source_key
            or selection.snapshot_id != manifest.snapshot_id):
        raise TargetedMediaWikiSourceError("FT30 selection/snapshot 身份漂移")
    if split not in {"train", "dev", "held_out"}:
        raise TargetedMediaWikiSourceError("FT30 目标 split 非法")
    for name, value in (
            ("max_text_bytes_per_page", max_text_bytes_per_page),
            ("max_templates_per_page", max_templates_per_page),
            ("max_template_depth", max_template_depth)):
        if type(value) is not int or value <= 0:
            raise TargetedMediaWikiSourceError(f"FT30 {name} 非正整数")
    raw = Path(raw_root).resolve()
    xml_file = _raw_file(manifest, "XML")
    index_file = _raw_file(manifest, "INDEX")
    xml_path = _safe_raw_path(raw, xml_file.raw_relative_path)
    index_path = _safe_raw_path(raw, index_file.raw_relative_path)
    if (xml_path.stat().st_size != xml_file.compressed_size_bytes
            or index_path.stat().st_size != index_file.compressed_size_bytes
            or selection.index_raw_relative_path
            != index_file.raw_relative_path
            or selection.index_compressed_size_bytes
            != index_file.compressed_size_bytes
            or selection.index_local_sha256 != index_file.local_sha256
            or selection.index_upstream_sha1 != index_file.upstream_sha1):
        raise TargetedMediaWikiSourceError("FT30 raw/selection identity 漂移")
    pages = _selected_pages_from_blocks(
        xml_path, selection.selected_titles)
    budget = MediaWikiScanBudget(
        max_pages=len(selection.selected_titles),
        max_xml_events=len(selection.selected_titles) * 2048,
        max_text_bytes_per_page=max_text_bytes_per_page,
        max_templates_per_page=max_templates_per_page,
        max_template_depth=max_template_depth,
    )
    seeds = []
    for ordinal, target in enumerate(selection.selected_titles, start=1):
        page, block_start, block_end = pages[(target.page_id, target.title)]
        try:
            record = parse_mediawiki_page(
                page,
                source_key=manifest.source_key,
                extract_templates=True,
                budget=budget,
            )
        except MediaWikiPageError as error:
            raise TargetedMediaWikiSourceError(
                f"FT30 目标页解析失败: {error.code}") from error
        text = _page_text(page)
        length = _length_bucket(record.text_size_bytes)
        axes = {
            "code_switch": "UNASSESSED",
            "dialect": "UNASSESSED",
            "domain": "lexicon",
            "era": manifest.dump_date,
            "genre": "dictionary_entry",
            "language": "zh",
            "length": length,
            "register": "UNASSESSED",
            "script_orthography": "ZH_WIKIMEDIA_RAW",
            "source": manifest.source_key,
            "source_document_cluster": f"page:{record.page_id}",
            "title_length_stratum": target.stratum,
        }
        contributor = record.contributor.to_value()
        raw_observation = {
            "contributor": contributor,
            "page_id": record.page_id,
            "redirect_title": record.redirect_title,
            "revision_id": record.revision_id,
            "text": text,
            "timestamp": record.timestamp,
            "title": record.title,
        }
        source_span = {
            "compressed_block_end_offset": block_end,
            "compressed_block_offset": block_start,
            "compressed_raw_relative_path": xml_file.raw_relative_path,
            "contributor": contributor,
            "index_line_number": target.index_line_number,
            "index_local_sha256": index_file.local_sha256,
            "index_raw_relative_path": index_file.raw_relative_path,
            "namespace_id": record.namespace_id,
            "page_id": record.page_id,
            "revision_id": record.revision_id,
            "selection_manifest_relative_path": (
                selection_manifest_relative_path),
            "selection_manifest_sha256": selection_manifest_sha256,
            "selection_sha256": target.selection_sha256,
            "text_sha256": record.text_sha256,
            "timestamp": record.timestamp,
            "title_length": target.title_length,
            "title_length_stratum": target.stratum,
            "title_sha256": target.title_sha256,
        }
        seeds.append(SourceObservationSeed(
            f"page-{record.page_id}-revision-{record.revision_id}",
            split,
            "zh",
            "mediawiki-raw-wikitext",
            (
                f"{xml_file.raw_relative_path}#page={record.page_id}"
                f";revision={record.revision_id}"
            ),
            "sha1:" + xml_file.upstream_sha1,
            xml_file.local_sha256,
            CanonicalJsonObject.from_value(source_span),
            CanonicalJsonObject.from_value(raw_observation),
            CanonicalJsonObject.from_value(axes),
            ("page", record.page_id),
            ("text", record.text_sha256),
            ("page", record.page_id, record.revision_id),
            (
                "page", record.page_id,
                "redirect" if record.redirect_title else "article",
                "lexicon",
            ),
            (
                "page", record.page_id,
                "length", length,
                "line_count", text.count("\n") + 1,
                "template_count", len(record.templates),
            ),
            tuple(
                value
                for key in sorted(axes)
                for value in (key, axes[key])
            ),
            "support" if split == "train" else "read_only_probe",
            "NONE",
            ordinal,
        ))
    return tuple(seeds)


__all__ = [
    "TargetedMediaWikiSourceError",
    "targeted_mediawiki_source_seeds",
    "targeted_mediawiki_source_seeds_from_selection_v2",
    "verify_targeted_mediawiki_raw_identities",
]
