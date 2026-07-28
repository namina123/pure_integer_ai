"""D-02 MediaWiki 正式 snapshot 的有界 raw Observation 取样边界。"""
from __future__ import annotations

import bz2
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    MediaWikiPageError,
    MediaWikiScanBudget,
    parse_mediawiki_page,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    MediaWikiDumpSnapshotManifest,
)
from pure_integer_ai.experiments.ph2_source_pack_contract import (
    SourceObservationSeed,
)


class SourcePackMediaWikiError(RuntimeError):
    """MediaWiki 有界取样缺页、路径漂移或 page/raw text 不一致。"""


def _local_name(tag: str) -> str:
    """去除 ElementTree namespace。"""
    return tag.rsplit("}", 1)[-1]


def _raw_text(page: ET.Element) -> str:
    """返回 page 唯一 revision 主 slot 的原始 wikitext。"""
    values = [
        element for element in page.iter()
        if _local_name(element.tag) == "text"
    ]
    if len(values) != 1:
        raise SourcePackMediaWikiError("MediaWiki page raw text 数量非法")
    return values[0].text or ""


def _length_bucket(size_bytes: int) -> str:
    """用冻结整数边界给有界样本标长度轴。"""
    if size_bytes <= 1024:
        return "SHORT"
    if size_bytes <= 16 * 1024:
        return "MEDIUM"
    return "LONG"


def _xml_raw_file(manifest: MediaWikiDumpSnapshotManifest):
    """返回正式 snapshot 中唯一 XML raw file。"""
    values = [item for item in manifest.raw_files if item.role == "XML"]
    if len(values) != 1:
        raise SourcePackMediaWikiError("MediaWiki snapshot XML raw file 非唯一")
    return values[0]


def bounded_mediawiki_source_seeds(
        manifest: MediaWikiDumpSnapshotManifest,
        *,
        raw_root: str | Path,
        limit: int = 4,
        max_scanned_pages: int = 10_000,
        ) -> tuple[SourceObservationSeed, ...]:
    """从正式压缩 raw 开头取固定主空间页，不重跑全量 scan。"""
    if not isinstance(manifest, MediaWikiDumpSnapshotManifest):
        raise SourcePackMediaWikiError("MediaWiki snapshot manifest 类型错误")
    if type(limit) is not int or limit < 2:
        raise SourcePackMediaWikiError("MediaWiki source pack limit 至少为 2")
    if type(max_scanned_pages) is not int or max_scanned_pages < limit:
        raise SourcePackMediaWikiError("MediaWiki max_scanned_pages 非法")
    raw_file = _xml_raw_file(manifest)
    root = Path(raw_root).resolve()
    raw_path = (root / Path(*raw_file.raw_relative_path.split("/"))).resolve()
    if not raw_path.is_relative_to(root) or not raw_path.is_file():
        raise SourcePackMediaWikiError("MediaWiki source pack raw path 缺失或逃逸")
    if raw_path.stat().st_size != raw_file.compressed_size_bytes:
        raise SourcePackMediaWikiError("MediaWiki source pack raw size 漂移")
    budget = MediaWikiScanBudget(
        max_pages=max_scanned_pages,
        max_xml_events=max_scanned_pages * 64,
        max_text_bytes_per_page=64 * 1024 * 1024,
        max_templates_per_page=1,
        max_template_depth=1,
    )
    selected: list[tuple[Any, str]] = []
    scanned_pages = 0
    with bz2.open(raw_path, "rb") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if _local_name(element.tag) != "page":
                continue
            scanned_pages += 1
            if scanned_pages > max_scanned_pages:
                raise SourcePackMediaWikiError("MediaWiki 有界取样超 page 预算")
            try:
                record = parse_mediawiki_page(
                    element,
                    source_key=manifest.source_key,
                    extract_templates=False,
                    budget=budget,
                )
            except MediaWikiPageError as error:
                if error.code == "NON_MAIN_NAMESPACE":
                    element.clear()
                    continue
                raise SourcePackMediaWikiError(
                    f"MediaWiki 有界 page 解析失败: {error.code}") from error
            selected.append((record, _raw_text(element)))
            element.clear()
            if len(selected) == limit:
                break
    if len(selected) != limit:
        raise SourcePackMediaWikiError("MediaWiki 有界取样未取得足够主空间页")
    boundary = limit // 2
    domain = "lexicon" if manifest.project == "zhwiktionary" else "encyclopedia"
    genre = "dictionary_entry" if manifest.project == "zhwiktionary" else "article"
    seeds: list[SourceObservationSeed] = []
    for ordinal, (record, text) in enumerate(selected, start=1):
        split = "train" if ordinal <= boundary else "held_out"
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
                f"{raw_file.raw_relative_path}#page={record.page_id}"
                f";revision={record.revision_id}"
            ),
            "sha1:" + raw_file.upstream_sha1,
            raw_file.local_sha256,
            CanonicalJsonObject.from_value({
                "compressed_raw_relative_path": raw_file.raw_relative_path,
                "contributor": contributor,
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


__all__ = [
    "SourcePackMediaWikiError",
    "bounded_mediawiki_source_seeds",
]
