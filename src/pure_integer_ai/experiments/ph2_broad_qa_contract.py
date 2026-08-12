"""来源约束广域问答 V0 的选择、段落和结果值合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)


SELECTION_KIND = "PH2_BROAD_QA_SELECTION_V1"
SELECTION_RULE = "SNAPSHOT_PAGE_TITLE_SHA256_LOWEST_V1"
RESULT_STATUSES = ("ANSWER", "CLARIFY", "UNKNOWN", "CONFLICT")
WIKIPEDIA_ATTRIBUTION = "Wikipedia contributors"


# object-model: exception
class BroadQaContractError(ValueError):
    """广域问答结构、身份或规范字节不满足冻结合同。"""


def _sha256(value: str, *, label: str) -> str:
    """核验小写 SHA-256 文本并原样返回。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaContractError(f"{label} 不是小写 SHA-256")
    return value


def _positive(value: int, *, label: str) -> int:
    """核验协议字段为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise BroadQaContractError(f"{label} 必须是正严格整数")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSelectedPage:
    """记录仅由冻结 index 和稳定排名确定的页面压缩坐标。"""

    ordinal: int
    rank_sha256: str
    title: str
    title_sha256: str
    page_id: int
    index_line_number: int
    compressed_block_offset: int
    compressed_block_end_offset: int

    def __post_init__(self) -> None:
        """核验页面坐标、稳定排名和标题承诺的值不变量。"""
        _positive(self.ordinal, label="selected ordinal")
        _sha256(self.rank_sha256, label="selection rank")
        _sha256(self.title_sha256, label="selection title")
        _positive(self.page_id, label="selection page_id")
        _positive(self.index_line_number, label="selection index line")
        if (not isinstance(self.title, str) or not self.title
                or self.title.strip() != self.title):
            raise BroadQaContractError("selection title 非法")
        if (type(self.compressed_block_offset) is not int
                or type(self.compressed_block_end_offset) is not int
                or self.compressed_block_offset < 0
                or self.compressed_block_end_offset
                <= self.compressed_block_offset):
            raise BroadQaContractError("selection block 坐标非法")
        expected = hashlib.sha256(self.title.encode("utf-8")).hexdigest()
        if expected != self.title_sha256:
            raise BroadQaContractError("selection title SHA 漂移")

    def to_dict(self) -> dict[str, object]:
        """导出规范 JSON 值。"""
        return {
            "compressed_block_end_offset": self.compressed_block_end_offset,
            "compressed_block_offset": self.compressed_block_offset,
            "index_line_number": self.index_line_number,
            "ordinal": self.ordinal,
            "page_id": self.page_id,
            "rank_sha256": self.rank_sha256,
            "title": self.title,
            "title_sha256": self.title_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "BroadQaSelectedPage":
        """从字段精确的 JSON object 恢复页面坐标。"""
        keys = {
            "compressed_block_end_offset", "compressed_block_offset",
            "index_line_number", "ordinal", "page_id", "rank_sha256",
            "title", "title_sha256",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise BroadQaContractError("selection page 字段漂移")
        return cls(
            value["ordinal"], value["rank_sha256"], value["title"],
            value["title_sha256"], value["page_id"],
            value["index_line_number"], value["compressed_block_offset"],
            value["compressed_block_end_offset"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSelectionManifest:
    """绑定冻结 Wikipedia/index 和未看正文的稳定页面选择。"""

    source_key: str
    snapshot_id: str
    snapshot_manifest_sha256: str
    index_local_sha256: str
    index_upstream_sha1: str
    xml_local_sha256: str
    xml_compressed_size_bytes: int
    index_entry_count: int
    requested_page_count: int
    selected_pages: tuple[BroadQaSelectedPage, ...]

    def __post_init__(self) -> None:
        """核验 snapshot 身份、选择数量、唯一性和稳定排名顺序。"""
        if (self.source_key != "ZHWIKIPEDIA_20260701"
                or not isinstance(self.snapshot_id, str)
                or not self.snapshot_id):
            raise BroadQaContractError("selection snapshot identity 漂移")
        for label, value in (
                ("snapshot manifest", self.snapshot_manifest_sha256),
                ("index local", self.index_local_sha256),
                ("xml local", self.xml_local_sha256)):
            _sha256(value, label=label)
        if (not isinstance(self.index_upstream_sha1, str)
                or len(self.index_upstream_sha1) != 40
                or any(item not in "0123456789abcdef"
                       for item in self.index_upstream_sha1)):
            raise BroadQaContractError("selection index SHA-1 非法")
        _positive(self.xml_compressed_size_bytes, label="selection XML size")
        _positive(self.index_entry_count, label="selection index count")
        _positive(self.requested_page_count, label="selection requested count")
        if (not isinstance(self.selected_pages, tuple)
                or len(self.selected_pages) != self.requested_page_count
                or tuple(item.ordinal for item in self.selected_pages)
                != tuple(range(1, self.requested_page_count + 1))
                or len({item.page_id for item in self.selected_pages})
                != self.requested_page_count):
            raise BroadQaContractError("selection page inventory 非规范")
        ranks = tuple(item.rank_sha256 for item in self.selected_pages)
        if ranks != tuple(sorted(ranks)):
            raise BroadQaContractError("selection rank 顺序漂移")

    def to_dict(self) -> dict[str, object]:
        """导出规范 manifest JSON 值。"""
        return {
            "artifact_kind": SELECTION_KIND,
            "format_version": 1,
            "index_entry_count": self.index_entry_count,
            "index_local_sha256": self.index_local_sha256,
            "index_upstream_sha1": self.index_upstream_sha1,
            "requested_page_count": self.requested_page_count,
            "selected_pages": [item.to_dict() for item in self.selected_pages],
            "selection_rule": SELECTION_RULE,
            "snapshot_id": self.snapshot_id,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "source_key": self.source_key,
            "xml_compressed_size_bytes": self.xml_compressed_size_bytes,
            "xml_local_sha256": self.xml_local_sha256,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 manifest 字节。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回规范 manifest SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "BroadQaSelectionManifest":
        """从字段精确的 JSON object 恢复选择 manifest。"""
        keys = {
            "artifact_kind", "format_version", "index_entry_count",
            "index_local_sha256", "index_upstream_sha1",
            "requested_page_count", "selected_pages", "selection_rule",
            "snapshot_id", "snapshot_manifest_sha256", "source_key",
            "xml_compressed_size_bytes", "xml_local_sha256",
        }
        if (not isinstance(value, dict) or set(value) != keys
                or value["artifact_kind"] != SELECTION_KIND
                or value["format_version"] != 1
                or value["selection_rule"] != SELECTION_RULE
                or not isinstance(value["selected_pages"], list)):
            raise BroadQaContractError("selection manifest 字段漂移")
        return cls(
            value["source_key"], value["snapshot_id"],
            value["snapshot_manifest_sha256"],
            value["index_local_sha256"], value["index_upstream_sha1"],
            value["xml_local_sha256"], value["xml_compressed_size_bytes"],
            value["index_entry_count"], value["requested_page_count"],
            tuple(BroadQaSelectedPage.from_dict(item)
                  for item in value["selected_pages"]),
        )


def parse_selection_manifest(payload: bytes) -> BroadQaSelectionManifest:
    """严格回读单换行结尾且逐字节规范的选择 manifest。"""
    if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise BroadQaContractError("selection manifest 换行非法")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    manifest = BroadQaSelectionManifest.from_dict(value)
    if manifest.canonical_bytes() != payload:
        raise BroadQaContractError("selection manifest 不是规范字节")
    return manifest


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaPassage:
    """保留原始 Wikitext 段落 span 与确定性显示投影。"""

    ordinal: int
    raw_start: int
    raw_end: int
    raw_sha256: str
    text: str
    text_sha256: str
    section_title: str

    def __post_init__(self) -> None:
        """核验证据段原文范围、显示投影和两类内容承诺。"""
        _positive(self.ordinal, label="passage ordinal")
        if (type(self.raw_start) is not int or type(self.raw_end) is not int
                or self.raw_start < 0 or self.raw_end <= self.raw_start):
            raise BroadQaContractError("passage raw span 非法")
        _sha256(self.raw_sha256, label="passage raw")
        _sha256(self.text_sha256, label="passage text")
        if (not isinstance(self.text, str) or not self.text
                or not isinstance(self.section_title, str)):
            raise BroadQaContractError("passage text/section 非法")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaResult:
    """表达回答、澄清或拒答，并绑定可核验的来源证据。"""

    status: str
    question: str
    answer: str | None
    title: str | None
    page_id: int | None
    revision_id: int | None
    evidence_text: str | None
    evidence_raw_start: int | None
    evidence_raw_end: int | None
    evidence_raw_sha256: str | None
    source_url: str | None
    snapshot_id: str
    license_id: str
    matched_term_count: int
    candidate_document_count: int
    revision_timestamp: str | None = None
    contributor_json: str | None = None
    attribution: str | None = None

    def __post_init__(self) -> None:
        """核验状态、计数和 ANSWER 完整来源归属的不变量。"""
        if self.status not in RESULT_STATUSES:
            raise BroadQaContractError("broad QA status 非法")
        if not isinstance(self.question, str) or not self.question.strip():
            raise BroadQaContractError("broad QA question 非法")
        if (not isinstance(self.snapshot_id, str) or not self.snapshot_id
                or self.license_id != "CC-BY-SA-4.0"):
            raise BroadQaContractError("broad QA source boundary 漂移")
        if (type(self.matched_term_count) is not int
                or type(self.candidate_document_count) is not int
                or self.matched_term_count < 0
                or self.candidate_document_count < 0):
            raise BroadQaContractError("broad QA query counters 非法")
        evidence = self.status == "ANSWER"
        required = (
            self.answer, self.title, self.page_id, self.revision_id,
            self.evidence_text, self.evidence_raw_start,
            self.evidence_raw_end, self.evidence_raw_sha256, self.source_url,
            self.revision_timestamp, self.contributor_json, self.attribution,
        )
        if evidence and any(item is None for item in required):
            raise BroadQaContractError("ANSWER 缺少来源证据")
        if not evidence and any(item is not None for item in required):
            raise BroadQaContractError("非 ANSWER 不得携带伪证据")
        if evidence:
            try:
                contributor = json.loads(self.contributor_json)
            except (TypeError, json.JSONDecodeError) as error:
                raise BroadQaContractError("ANSWER contributor 非法") from error
            if (not isinstance(contributor, dict)
                    or not isinstance(self.revision_timestamp, str)
                    or not self.revision_timestamp
                    or self.attribution != WIKIPEDIA_ATTRIBUTION):
                raise BroadQaContractError("ANSWER 归属身份不完整")

    def to_dict(self) -> dict[str, object]:
        """导出稳定的外部 JSON 结果。"""
        return {
            "answer": self.answer,
            "candidate_document_count": self.candidate_document_count,
            "citation": None if self.status != "ANSWER" else {
                "evidence_raw_end": self.evidence_raw_end,
                "evidence_raw_sha256": self.evidence_raw_sha256,
                "evidence_raw_start": self.evidence_raw_start,
                "evidence_text": self.evidence_text,
                "attribution": self.attribution,
                "contributor": json.loads(self.contributor_json),
                "license_id": self.license_id,
                "page_id": self.page_id,
                "revision_id": self.revision_id,
                "revision_timestamp": self.revision_timestamp,
                "snapshot_id": self.snapshot_id,
                "source_url": self.source_url,
                "title": self.title,
            },
            "matched_term_count": self.matched_term_count,
            "question": self.question,
            "status": self.status,
        }


__all__ = [
    "BroadQaContractError",
    "BroadQaPassage",
    "BroadQaResult",
    "BroadQaSelectedPage",
    "BroadQaSelectionManifest",
    "RESULT_STATUSES",
    "SELECTION_KIND",
    "SELECTION_RULE",
    "WIKIPEDIA_ATTRIBUTION",
    "parse_selection_manifest",
]
