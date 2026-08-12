"""按冻结坐标恢复 Wikipedia 页面并投影可引用证据段。"""
from __future__ import annotations

import bz2
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
from itertools import groupby
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import wikitextparser

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaPassage,
    BroadQaSelectedPage,
    BroadQaSelectionManifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    MediaWikiPageError,
    MediaWikiScanBudget,
    parse_mediawiki_page,
)


_PARAGRAPH_RE = re.compile(r"\S(?:.*?\S)?(?=\n[ \t]*\n|\Z)", re.DOTALL)


# object-model: exception
class BroadQaSourceError(RuntimeError):
    """压缩块、页面身份、Wikitext 或来源 span 发生漂移。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceCandidate:
    """携带选择序号、主空间来源身份和尚未投影的原始 Wikitext。"""

    ordinal: int
    title: str
    page_id: int
    revision_id: int
    timestamp: str
    contributor_json: str
    text_sha256: str
    wikitext: str


def _local_name(tag: str) -> str:
    """移除 ElementTree namespace 前缀。"""
    return tag.rsplit("}", 1)[-1]


def _block_pages(payload: bytes) -> tuple[ET.Element, ...]:
    """把单个 multistream bzip2 payload 包装为可解析页面集合。"""
    stripped = payload.replace(b"</mediawiki>", b"")
    try:
        root = ET.parse(io.BytesIO(
            b"<mediawiki>" + stripped + b"</mediawiki>")).getroot()
    except ET.ParseError as error:
        raise BroadQaSourceError("broad QA multistream block XML 非法") from error
    return tuple(item for item in root if _local_name(item.tag) == "page")


def _page_text(page: ET.Element) -> str:
    """恢复页面唯一 revision text 并拒绝结构漂移。"""
    values = tuple(
        item for item in page.iter() if _local_name(item.tag) == "text")
    if len(values) != 1:
        raise BroadQaSourceError("broad QA page text 数量非法")
    return values[0].text or ""


def _page_identity(page: ET.Element) -> tuple[int, str]:
    """只读页面直属 title/id，避免解析同 block 内的无关正文。"""
    titles = tuple(
        item.text or "" for item in page if _local_name(item.tag) == "title")
    identifiers = tuple(
        item.text or "" for item in page if _local_name(item.tag) == "id")
    if len(titles) != 1 or len(identifiers) != 1 or not titles[0]:
        raise BroadQaSourceError("broad QA page identity 结构非法")
    try:
        page_id = int(identifiers[0])
    except ValueError as error:
        raise BroadQaSourceError("broad QA page id 非整数") from error
    if page_id <= 0:
        raise BroadQaSourceError("broad QA page id 非正")
    return page_id, titles[0]


def _plain_text(raw: str) -> str:
    """用成熟 Wikitext parser 生成确定性的证据显示投影。"""
    try:
        value = wikitextparser.parse(raw).plain_text(
            replace_templates=True,
            replace_parser_functions=True,
            replace_parameters=True,
            replace_tags=True,
            replace_external_links=True,
            replace_wikilinks=True,
            unescape_html_entities=True,
            replace_bolds_and_italics=True,
            replace_tables=True,
        )
    except Exception as error:
        raise BroadQaSourceError("broad QA Wikitext projection 失败") from error
    lines = tuple(" ".join(item.split()) for item in value.splitlines())
    return " ".join(item for item in lines if item).strip()


def project_broad_qa_passages(
        wikitext: str,
        *,
        max_passages: int = 12,
        max_projection_characters: int = 1200,
        ) -> tuple[BroadQaPassage, ...]:
    """按 section/空行分段并保留每段原始 Wikitext 精确 span。"""
    if not isinstance(wikitext, str):
        raise TypeError("broad QA wikitext 必须是字符串")
    if (type(max_passages) is not int or not 1 <= max_passages <= 128
            or type(max_projection_characters) is not int
            or not 128 <= max_projection_characters <= 16384):
        raise BroadQaSourceError("broad QA passage budget 非法")
    try:
        document = wikitextparser.parse(wikitext)
        sections = tuple(document.sections)
    except Exception as error:
        raise BroadQaSourceError("broad QA Wikitext section parse 失败") from error
    passages: list[BroadQaPassage] = []
    for section in sections:
        section_start, section_end = section.span
        contents = section.contents
        if not section.string.endswith(contents):
            raise BroadQaSourceError("broad QA section content span 不可恢复")
        content_start = section_end - len(contents)
        title = "" if section.title is None else " ".join(section.title.split())
        for match in _PARAGRAPH_RE.finditer(contents):
            raw_start = content_start + match.start()
            raw_end = content_start + match.end()
            raw = wikitext[raw_start:raw_end]
            projected = _plain_text(raw)
            if len(projected) < 8:
                continue
            if len(projected) > max_projection_characters:
                projected = projected[:max_projection_characters].rstrip()
            passages.append(BroadQaPassage(
                len(passages) + 1,
                raw_start,
                raw_end,
                hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                projected,
                hashlib.sha256(projected.encode("utf-8")).hexdigest(),
                title,
            ))
            if len(passages) >= max_passages:
                return tuple(passages)
    return tuple(passages)


def _block_candidates(
        compressed: bytes,
        selected_pages: tuple[BroadQaSelectedPage, ...],
        source_key: str,
        ) -> tuple[tuple[int, ...], tuple[BroadQaSourceCandidate, ...]]:
    """在一个 worker 内解压单块并恢复其中全部目标候选。"""
    targets = {(item.page_id, item.title): item for item in selected_pages}
    if len(targets) != len(selected_pages):
        raise BroadQaSourceError("broad QA block target identity 重复")
    try:
        pages = _block_pages(bz2.decompress(compressed))
    except OSError as error:
        raise BroadQaSourceError("broad QA block 解压失败") from error
    budget = MediaWikiScanBudget(
        max_pages=1024,
        max_xml_events=1_000_000,
        max_text_bytes_per_page=2 * 1024 * 1024,
        max_templates_per_page=1,
        max_template_depth=1,
    )
    seen = []
    candidates = []
    for page in pages:
        selected = targets.get(_page_identity(page))
        if selected is None:
            continue
        seen.append(selected.ordinal)
        try:
            record = parse_mediawiki_page(
                page,
                source_key=source_key,
                extract_templates=False,
                budget=budget,
            )
        except MediaWikiPageError as error:
            if error.code == "NON_MAIN_NAMESPACE":
                continue
            raise BroadQaSourceError(
                f"broad QA page 解析失败: {error.code}") from error
        wikitext = _page_text(page)
        if not wikitext.strip() or record.redirect_title:
            continue
        contributor = canonical_json_bytes(
            record.contributor.to_value()).decode("utf-8")
        candidates.append(BroadQaSourceCandidate(
            selected.ordinal,
            record.title,
            record.page_id,
            record.revision_id,
            record.timestamp,
            contributor,
            record.text_sha256,
            wikitext,
        ))
    return tuple(seen), tuple(candidates)


def iter_broad_qa_candidate_pages(
        selection: BroadQaSelectionManifest,
        *,
        xml_path: str | Path,
        worker_count: int = 1,
        ):
    """有界并行解压 block，并按压缩 offset 原序产出候选。"""
    if not isinstance(selection, BroadQaSelectionManifest):
        raise TypeError("broad QA selection 类型错误")
    xml = Path(xml_path).resolve()
    if (not xml.is_file()
            or xml.stat().st_size != selection.xml_compressed_size_bytes):
        raise BroadQaSourceError("broad QA XML 缺失或 size 漂移")
    if type(worker_count) is not int or worker_count not in {1, 2, 4}:
        raise BroadQaSourceError("broad QA worker count 只能为 1/2/4")
    seen_ordinals: set[int] = set()
    ordered = sorted(
        selection.selected_pages,
        key=lambda item: (
            item.compressed_block_offset,
            item.compressed_block_end_offset,
            item.ordinal,
        ),
    )
    grouped = tuple(
        (key, tuple(selected_group))
        for key, selected_group in groupby(
            ordered,
            key=lambda item: (
                item.compressed_block_offset,
                item.compressed_block_end_offset,
            ),
        )
    )

    def consume(result):
        """按提交顺序核对 inventory 并转交有效候选。"""
        seen, candidates = result
        for ordinal in seen:
            if ordinal in seen_ordinals:
                raise BroadQaSourceError("broad QA page identity 非唯一")
            seen_ordinals.add(ordinal)
        yield from candidates

    with xml.open("rb") as handle, ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="broad-qa-block") as executor:
        pending: deque[Future] = deque()
        for key, selected_pages in grouped:
            handle.seek(key[0])
            compressed = handle.read(key[1] - key[0])
            if len(compressed) != key[1] - key[0]:
                raise BroadQaSourceError("broad QA block 读取不完整")
            pending.append(executor.submit(
                _block_candidates, compressed, selected_pages,
                selection.source_key))
            if len(pending) >= worker_count * 2:
                yield from consume(pending.popleft().result())
        while pending:
            yield from consume(pending.popleft().result())
    if len(seen_ordinals) != len(selection.selected_pages):
        raise BroadQaSourceError("broad QA index/XML page inventory 漂移")


__all__ = [
    "BroadQaSourceError",
    "BroadQaSourceCandidate",
    "iter_broad_qa_candidate_pages",
    "project_broad_qa_passages",
]
