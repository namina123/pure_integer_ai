"""物化来源 dossier 共用的终页证据内核。

本模块只恢复冻结页面并投影原始 Wikitext、纯文本和 passage 坐标。它不知道
review、training、decision、operator 或 mastery，也不写任何语义标签。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_source import (
    iter_broad_qa_selected_page_inspections,
    project_broad_qa_passages,
    project_broad_qa_plain_text,
)


def passage_evidence_record(value) -> dict[str, object]:
    """把 passage 投影为保留原始坐标和双 SHA 的规范字段。"""
    return {
        "ordinal": value.ordinal,
        "raw_end": value.raw_end,
        "raw_sha256": value.raw_sha256,
        "raw_start": value.raw_start,
        "section_title": value.section_title,
        "text": value.text,
        "text_sha256": value.text_sha256,
    }


def materialize_terminal_sources(
        selection,
        *,
        required_page_revisions: dict[int, int],
        xml_path: str | Path,
        worker_count: int,
        inspection_reader: Callable[..., Iterable[object]] = (
            iter_broad_qa_selected_page_inspections),
        plain_text_projector: Callable[[str], str] = (
            project_broad_qa_plain_text),
        passage_projector: Callable[..., Iterable[object]] = (
            project_broad_qa_passages),
        ) -> dict[int, dict[str, object]]:
    """恢复指定终页并核验 revision，返回无阶段语义的完整来源记录。"""
    if (not isinstance(required_page_revisions, dict)
            or not required_page_revisions
            or any(type(page_id) is not int or page_id <= 0
                   or type(revision_id) is not int or revision_id <= 0
                   for page_id, revision_id
                   in required_page_revisions.items())):
        raise BroadQaExternalDataError("terminal dossier page/revision 非法")
    page_ids = set(required_page_revisions)
    selected_pages = tuple(
        item for item in selection.selected_pages if item.page_id in page_ids)
    if {item.page_id for item in selected_pages} != page_ids:
        raise BroadQaExternalDataError("terminal dossier page 不在 selection")
    inspections = tuple(inspection_reader(
        selected_pages,
        xml_path=Path(xml_path).resolve(),
        source_key=selection.source_key,
        xml_compressed_size_bytes=selection.xml_compressed_size_bytes,
        worker_count=worker_count,
    ))
    inspection_by_id = {item.page_id: item for item in inspections}
    if (set(inspection_by_id) != page_ids
            or len(inspection_by_id) != len(inspections)
            or any(item.redirect_title for item in inspections)):
        raise BroadQaExternalDataError(
            "terminal dossier inspection inventory 漂移")

    terminal_sources = {}
    for page_id, inspection in inspection_by_id.items():
        if inspection.revision_id != required_page_revisions[page_id]:
            raise BroadQaExternalDataError("terminal dossier revision 漂移")
        wikitext_sha256 = hashlib.sha256(
            inspection.wikitext.encode("utf-8")).hexdigest()
        if wikitext_sha256 != inspection.text_sha256:
            raise BroadQaExternalDataError("terminal dossier Wikitext hash 漂移")
        plain_text = plain_text_projector(inspection.wikitext)
        passages = tuple(passage_projector(
            inspection.wikitext,
            max_passages=128,
            max_projection_characters=16384,
        ))
        terminal_sources[page_id] = {
            "attribution": "Wikipedia contributors",
            "contributor": json.loads(inspection.contributor_json),
            "license_id": "CC-BY-SA-4.0",
            "page_id": inspection.page_id,
            "passages": [passage_evidence_record(item) for item in passages],
            "plain_text": plain_text,
            "plain_text_sha256": hashlib.sha256(
                plain_text.encode("utf-8")).hexdigest(),
            "revision_id": inspection.revision_id,
            "revision_timestamp": inspection.timestamp,
            "snapshot_id": selection.snapshot_id,
            "source_url": (
                "https://zh.wikipedia.org/w/index.php?curid="
                f"{inspection.page_id}&oldid={inspection.revision_id}"),
            "title": inspection.title,
            "wikitext": inspection.wikitext,
            "wikitext_sha256": wikitext_sha256,
        }
    return terminal_sources


__all__ = [
    "materialize_terminal_sources",
    "passage_evidence_record",
]
