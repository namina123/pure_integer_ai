"""中文 Wikimedia multistream XML/index 的页簇流式 parser 与预算。"""
from __future__ import annotations

import bz2
import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
)


SOURCE_KEYS = frozenset({"ZHWIKIPEDIA_20260701", "ZHWIKTIONARY_20260701"})
_INTEGER_RE = re.compile(r"0|[1-9][0-9]*")
_POSITIVE_RE = re.compile(r"[1-9][0-9]*")
_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
)
_INDEX_ENTITY_RE = re.compile(r"&(amp|lt|gt|quot|apos);")
_INDEX_ENTITIES = {
    "amp": "&",
    "apos": "'",
    "gt": ">",
    "lt": "<",
    "quot": '"',
}


class MediaWikiAdapterError(RuntimeError):
    """multistream XML/index、页身份、预算或 snapshot hash 不一致。"""


class MediaWikiPageError(MediaWikiAdapterError):
    """单页可原子隔离的 revision、text 或 template 错误。"""

    def __init__(self, code: str, message: str) -> None:
        """保存稳定 anomaly code 与可读说明。"""
        super().__init__(message)
        self.code = code


def _text(value: Any, *, where: str) -> str:
    """要求非空且无首尾空白文本。"""
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise MediaWikiPageError("BAD_TEXT", f"{where} 必须是非空文本")
    return value


def _integer_text(value: Any, *, where: str, positive: bool) -> int:
    """从十进制文本恢复无前导零整数，绝不走浮点。"""
    if not isinstance(value, str):
        raise MediaWikiPageError("BAD_INTEGER", f"{where} 必须是整数文本")
    pattern = _POSITIVE_RE if positive else _INTEGER_RE
    if not pattern.fullmatch(value):
        raise MediaWikiPageError("BAD_INTEGER", f"{where} 整数文本非法")
    return int(value)


def _local_name(tag: str) -> str:
    """移除 ElementTree 展开的 XML namespace。"""
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    """返回指定 local name 的直接子元素。"""
    return [child for child in element if _local_name(child.tag) == name]


def _one_child(
        element: ET.Element,
        name: str,
        *,
        required: bool = True,
        ) -> ET.Element | None:
    """读取最多一个直接子元素，重复或缺失时失败。"""
    values = _children(element, name)
    if len(values) > 1 or (required and not values):
        raise MediaWikiPageError(
            "BAD_XML_CARDINALITY", f"{name} 元素数量非法")
    return values[0] if values else None


def _child_text(
        element: ET.Element,
        name: str,
        *,
        required: bool = True,
        allow_empty: bool = False,
        ) -> str:
    """读取唯一直接子元素文本并执行空值边界。"""
    child = _one_child(element, name, required=required)
    if child is None:
        return ""
    text = child.text or ""
    if not allow_empty:
        return _text(text, where=name)
    return text


@dataclass(frozen=True, order=True)
class MultistreamIndexEntry:
    """一个 `offset:pageid:title` index 行。"""

    offset: int
    page_id: int
    title: str
    line_number: int

    def __post_init__(self) -> None:
        if type(self.offset) is not int or self.offset < 0:
            raise MediaWikiAdapterError("multistream offset 非法")
        if type(self.page_id) is not int or self.page_id <= 0:
            raise MediaWikiAdapterError("multistream page id 非法")
        if (not isinstance(self.title, str) or not self.title
                or self.title.strip() != self.title):
            raise MediaWikiAdapterError("multistream title 非法")
        if type(self.line_number) is not int or self.line_number <= 0:
            raise MediaWikiAdapterError("multistream line number 非法")


def parse_multistream_index_line(
        line: bytes,
        *,
        line_number: int,
        ) -> MultistreamIndexEntry:
    """严格解析 index 行，title 中后续冒号保持原样。"""
    try:
        text = line.decode("utf-8").rstrip("\r\n")
    except UnicodeError as error:
        raise MediaWikiAdapterError("multistream index 非 UTF-8") from error
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise MediaWikiAdapterError("multistream index 列数非法")
    try:
        offset = _integer_text(parts[0], where="index offset", positive=False)
        page_id = _integer_text(parts[1], where="index page id", positive=True)
    except MediaWikiPageError as error:
        raise MediaWikiAdapterError("multistream index 整数非法") from error
    raw_title = parts[2]
    pieces: list[str] = []
    cursor = 0
    for match in _INDEX_ENTITY_RE.finditer(raw_title):
        if "&" in raw_title[cursor:match.start()]:
            raise MediaWikiAdapterError("multistream index title 实体非法")
        pieces.append(raw_title[cursor:match.start()])
        pieces.append(_INDEX_ENTITIES[match.group(1)])
        cursor = match.end()
    if "&" in raw_title[cursor:]:
        raise MediaWikiAdapterError("multistream index title 实体非法")
    pieces.append(raw_title[cursor:])
    title = "".join(pieces)
    return MultistreamIndexEntry(offset, page_id, title, line_number)


def iter_multistream_index(stream: BinaryIO) -> Iterator[MultistreamIndexEntry]:
    """逐行读取 index，验证顺序、唯一 page id 和终止换行。"""
    last_offset = -1
    last_page_id = 0
    line_number = 0
    while True:
        line = stream.readline()
        if not line:
            break
        line_number += 1
        if not line.endswith(b"\n"):
            raise MediaWikiAdapterError("multistream index 末行缺换行")
        entry = parse_multistream_index_line(line, line_number=line_number)
        if entry.offset < last_offset or entry.page_id <= last_page_id:
            raise MediaWikiAdapterError("multistream index 顺序或 page id 非单调")
        last_offset = entry.offset
        last_page_id = entry.page_id
        yield entry
    if line_number == 0:
        raise MediaWikiAdapterError("multistream index 为空")


@dataclass(frozen=True)
class MediaWikiTemplateSpan:
    """一个完整平衡模板的名称、字符区间与原文 hash。"""

    name: str
    start: int
    end: int
    raw_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """导出 template provenance。"""
        return {
            "end": self.end,
            "name": self.name,
            "raw_sha256": self.raw_sha256,
            "start": self.start,
        }


def _template_name(raw: str) -> str:
    """在 top-level 首个分隔符前恢复模板名，不展开模板语义。"""
    inner = raw[2:-2]
    stack: list[int] = []
    index = 0
    boundary = len(inner)
    while index < len(inner):
        if inner.startswith("{{{", index):
            stack.append(3)
            index += 3
        elif inner.startswith("{{", index):
            stack.append(2)
            index += 2
        elif inner.startswith("}}}", index) and stack and stack[-1] == 3:
            stack.pop()
            index += 3
        elif inner.startswith("}}", index) and stack and stack[-1] == 2:
            stack.pop()
            index += 2
        elif inner[index] == "|" and not stack:
            boundary = index
            break
        else:
            index += 1
    name = inner[:boundary].strip()
    if not name:
        raise MediaWikiPageError("BAD_TEMPLATE_NAME", "template name 非法")
    return name


def extract_balanced_templates(
        text: str,
        *,
        max_templates: int,
        max_depth: int,
        ) -> tuple[MediaWikiTemplateSpan, ...]:
    """扫描平衡双/三花括号，返回 top-level 模板而不猜标签。"""
    if (type(max_templates) is not int or max_templates <= 0
            or type(max_depth) is not int or max_depth <= 0):
        raise MediaWikiPageError("BAD_TEMPLATE_BUDGET", "template 预算非法")
    stack: list[tuple[int, int]] = []
    templates: list[MediaWikiTemplateSpan] = []
    index = 0
    while index < len(text):
        if text.startswith("{{{", index):
            stack.append((3, index))
            index += 3
        elif text.startswith("{{", index):
            stack.append((2, index))
            if len(stack) > max_depth:
                raise MediaWikiPageError(
                    "TEMPLATE_DEPTH_BUDGET", "template depth 超预算")
            index += 2
        elif (text.startswith("}}}", index)
                and stack and stack[-1][0] == 3):
            stack.pop()
            index += 3
        elif text.startswith("}}", index):
            if not stack or stack[-1][0] != 2:
                raise MediaWikiPageError(
                    "UNBALANCED_TEMPLATE", "双花括号关闭不匹配")
            _, start = stack.pop()
            end = index + 2
            if not stack:
                raw = text[start:end]
                templates.append(MediaWikiTemplateSpan(
                    _template_name(raw),
                    start,
                    end,
                    hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                ))
                if len(templates) > max_templates:
                    raise MediaWikiPageError(
                        "TEMPLATE_COUNT_BUDGET", "template count 超预算")
            index = end
        else:
            index += 1
    if stack:
        raise MediaWikiPageError("UNBALANCED_TEMPLATE", "template 未关闭")
    return tuple(templates)


@dataclass(frozen=True)
class MediaWikiScanBudget:
    """流式 scan 的确定性整数预算，不使用墙钟阈值。"""

    max_pages: int
    max_xml_events: int
    max_text_bytes_per_page: int
    max_templates_per_page: int
    max_template_depth: int

    def __post_init__(self) -> None:
        for name, value in (
                ("max_pages", self.max_pages),
                ("max_xml_events", self.max_xml_events),
                ("max_text_bytes_per_page", self.max_text_bytes_per_page),
                ("max_templates_per_page", self.max_templates_per_page),
                ("max_template_depth", self.max_template_depth)):
            if type(value) is not int or value <= 0:
                raise MediaWikiAdapterError(f"{name} 必须是正严格整数")


@dataclass(frozen=True)
class MediaWikiPageRecord:
    """主命名空间页面的完整 revision/text/template provenance。"""

    source_key: str
    page_id: int
    title: str
    namespace_id: int
    redirect_title: str
    revision_id: int
    parent_revision_id: int
    timestamp: str
    contributor: CanonicalJsonObject
    model: str
    format_name: str
    text_sha256: str
    text_size_bytes: int
    text_xml_attributes: CanonicalJsonObject
    upstream_sha1: str
    templates: tuple[MediaWikiTemplateSpan, ...]

    def to_event_dict(self) -> dict[str, Any]:
        """导出不含整页原文、但由 text hash 完整绑定的页级事件。"""
        return {
            "cluster_id": f"{self.source_key}:page:{self.page_id}",
            "contributor": self.contributor.to_value(),
            "definitive_truth_authoritative": 0,
            "format": self.format_name,
            "model": self.model,
            "namespace_id": self.namespace_id,
            "page_id": self.page_id,
            "parent_revision_id": self.parent_revision_id,
            "redirect_title": self.redirect_title,
            "revision_id": self.revision_id,
            "source_key": self.source_key,
            "templates": [item.to_dict() for item in self.templates],
            "text_sha256": self.text_sha256,
            "text_size_bytes": self.text_size_bytes,
            "text_xml_attributes": self.text_xml_attributes.to_value(),
            "timestamp": self.timestamp,
            "title": self.title,
            "upstream_sha1": self.upstream_sha1,
        }


def _page_identity(page: ET.Element) -> tuple[int, str, int]:
    """先恢复 page id/title/ns，供 index 对齐和 anomaly 定位。"""
    title = _child_text(page, "title")
    namespace_id = _integer_text(
        _child_text(page, "ns"), where="page.ns", positive=False)
    page_id = _integer_text(
        _child_text(page, "id"), where="page.id", positive=True)
    return page_id, title, namespace_id


def _revision_content(revision: ET.Element) -> tuple[str, str, ET.Element, str]:
    """兼容 direct revision 与 slots/main 两种 Wikimedia XML 结构。"""
    slots = _one_child(revision, "slots", required=False)
    if slots is None:
        model = _child_text(revision, "model")
        format_name = _child_text(revision, "format")
        text = _one_child(revision, "text")
        sha1 = _child_text(revision, "sha1")
        assert text is not None
        return model, format_name, text, sha1
    slot_values = _children(slots, "slot")
    main_slots = [
        slot for slot in slot_values
        if slot.attrib.get("role", "main") == "main"
    ]
    if len(main_slots) != 1:
        raise MediaWikiPageError("BAD_SLOT", "main slot 数量非法")
    main = main_slots[0]
    model = _child_text(main, "model")
    format_name = _child_text(main, "format")
    text = _one_child(main, "text")
    sha1 = _child_text(main, "sha1")
    assert text is not None
    return model, format_name, text, sha1


def _contributor(revision: ET.Element) -> CanonicalJsonObject:
    """保留 username/id、IP 或 deleted contributor 的来源字段。"""
    element = _one_child(revision, "contributor")
    assert element is not None
    if "deleted" in element.attrib:
        if list(element):
            raise MediaWikiPageError(
                "BAD_CONTRIBUTOR", "deleted contributor 不得有子字段")
        return CanonicalJsonObject.from_value({"deleted": 1})
    username = _children(element, "username")
    ids = _children(element, "id")
    ips = _children(element, "ip")
    if len(username) == len(ids) == 1 and not ips:
        user_id = _integer_text(
            ids[0].text or "", where="contributor.id", positive=False)
        return CanonicalJsonObject.from_value({
            "kind": "registered",
            "user_id": user_id,
            "username": _text(username[0].text or "", where="username"),
        })
    if len(ips) == 1 and not username and not ids:
        return CanonicalJsonObject.from_value({
            "ip": _text(ips[0].text or "", where="contributor.ip"),
            "kind": "ip",
        })
    raise MediaWikiPageError("BAD_CONTRIBUTOR", "contributor 字段组合非法")


def parse_mediawiki_page(
        page: ET.Element,
        *,
        source_key: str,
        extract_templates: bool,
        budget: MediaWikiScanBudget,
        ) -> MediaWikiPageRecord:
    """解析一个主命名空间 page，保留最新 revision 和文本 provenance。"""
    if source_key not in SOURCE_KEYS:
        raise MediaWikiPageError("BAD_SOURCE", "MediaWiki source_key 非法")
    page_id, title, namespace_id = _page_identity(page)
    if namespace_id != 0:
        raise MediaWikiPageError("NON_MAIN_NAMESPACE", "只消费主命名空间")
    revisions = _children(page, "revision")
    if len(revisions) != 1:
        raise MediaWikiPageError("BAD_REVISION_COUNT", "page revision 数量非法")
    revision = revisions[0]
    revision_id = _integer_text(
        _child_text(revision, "id"), where="revision.id", positive=True)
    parent_text = _child_text(
        revision, "parentid", required=False, allow_empty=True)
    parent_id = 0 if not parent_text else _integer_text(
        parent_text, where="revision.parentid", positive=True)
    timestamp = _child_text(revision, "timestamp")
    if not _TIMESTAMP_RE.fullmatch(timestamp):
        raise MediaWikiPageError("BAD_TIMESTAMP", "revision timestamp 非 UTC")
    contributor = _contributor(revision)
    model, format_name, text_element, sha1 = _revision_content(revision)
    if model != "wikitext" or format_name != "text/x-wiki":
        raise MediaWikiPageError("BAD_CONTENT_MODEL", "非 wikitext 主 slot")
    text = text_element.text or ""
    text_bytes = text.encode("utf-8")
    if len(text_bytes) > budget.max_text_bytes_per_page:
        raise MediaWikiPageError("TEXT_BUDGET", "单页文本超预算")
    redirect = _one_child(page, "redirect", required=False)
    redirect_title = ""
    if redirect is not None:
        if set(redirect.attrib) != {"title"}:
            raise MediaWikiPageError("BAD_REDIRECT", "redirect 字段非法")
        redirect_title = _text(redirect.attrib["title"], where="redirect.title")
    templates = ()
    if extract_templates:
        templates = extract_balanced_templates(
            text,
            max_templates=budget.max_templates_per_page,
            max_depth=budget.max_template_depth,
        )
    attributes = {
        key: value for key, value in sorted(text_element.attrib.items())
    }
    return MediaWikiPageRecord(
        source_key,
        page_id,
        title,
        namespace_id,
        redirect_title,
        revision_id,
        parent_id,
        timestamp,
        contributor,
        model,
        format_name,
        hashlib.sha256(text_bytes).hexdigest(),
        len(text_bytes),
        CanonicalJsonObject.from_value(attributes),
        _text(sha1, where="revision.sha1"),
        templates,
    )


@dataclass(frozen=True)
class MediaWikiScanReport:
    """一次完整 XML/index 流式扫描的确定性计数与 event hash。"""

    source_key: str
    xml_sha256: str
    index_sha256: str
    page_count: int
    main_namespace_count: int
    skipped_namespace_count: int
    valid_page_count: int
    anomaly_count: int
    anomaly_codes: CanonicalJsonObject
    anomaly_evidence: CanonicalJsonObject
    text_size_bytes: int
    template_count: int
    max_page_text_bytes: int
    max_template_count: int
    xml_event_count: int
    work_unit_count: int
    full_eof_verified: int
    event_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """导出 scan report。"""
        return {
            "anomaly_codes": self.anomaly_codes.to_value(),
            "anomaly_count": self.anomaly_count,
            "anomaly_evidence": self.anomaly_evidence.to_value()["items"],
            "event_sha256": self.event_sha256,
            "full_eof_verified": self.full_eof_verified,
            "index_sha256": self.index_sha256,
            "main_namespace_count": self.main_namespace_count,
            "max_page_text_bytes": self.max_page_text_bytes,
            "max_template_count": self.max_template_count,
            "page_count": self.page_count,
            "skipped_namespace_count": self.skipped_namespace_count,
            "source_key": self.source_key,
            "template_count": self.template_count,
            "text_size_bytes": self.text_size_bytes,
            "valid_page_count": self.valid_page_count,
            "work_unit_count": self.work_unit_count,
            "xml_event_count": self.xml_event_count,
            "xml_sha256": self.xml_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MediaWikiScanReport":
        """从规范 snapshot object 恢复 scan report。"""
        return cls(
            str(value["source_key"]),
            str(value["xml_sha256"]),
            str(value["index_sha256"]),
            value["page_count"],
            value["main_namespace_count"],
            value["skipped_namespace_count"],
            value["valid_page_count"],
            value["anomaly_count"],
            CanonicalJsonObject.from_value(dict(value["anomaly_codes"])),
            CanonicalJsonObject.from_value({
                "items": list(value["anomaly_evidence"]),
            }),
            value["text_size_bytes"],
            value["template_count"],
            value["max_page_text_bytes"],
            value["max_template_count"],
            value["xml_event_count"],
            value["work_unit_count"],
            value["full_eof_verified"],
            str(value["event_sha256"]),
        )


def _counter_add(counter: dict[str, int], key: str) -> None:
    """向异常计数器增加一次。"""
    counter[key] = counter.get(key, 0) + 1


def _sha256_path(path: Path) -> str:
    """以固定块大小流式计算压缩文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class _HashingReader:
    """在 ElementTree 读取时同步计算原始 stream SHA-256。"""

    def __init__(self, stream: BinaryIO) -> None:
        """包装二进制流并初始化 digest。"""
        self._stream = stream
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        """读取数据并更新 digest。"""
        data = self._stream.read(size)
        self._digest.update(data)
        return data

    def readline(self, size: int = -1) -> bytes:
        """逐行读取 index 并更新 digest。"""
        data = self._stream.readline(size)
        self._digest.update(data)
        return data

    def hexdigest(self) -> str:
        """返回截至 EOF 的原始 stream hash。"""
        return self._digest.hexdigest()


def scan_mediawiki_streams(
        xml_stream: BinaryIO,
        index_stream: BinaryIO,
        *,
        source_key: str,
        extract_templates: bool,
        budget: MediaWikiScanBudget,
        ) -> MediaWikiScanReport:
    """按 XML page 与 index 一一对齐扫描，页结束即 clear 保持有界内存。"""
    if source_key not in SOURCE_KEYS:
        raise MediaWikiAdapterError("MediaWiki source_key 非法")
    hashing_xml = _HashingReader(xml_stream)
    hashing_index = _HashingReader(index_stream)
    index_entries = iter_multistream_index(hashing_index)
    page_count = 0
    main_count = 0
    skipped_count = 0
    valid_count = 0
    text_size = 0
    template_count = 0
    max_text = 0
    max_templates = 0
    xml_events = 0
    anomaly_codes: dict[str, int] = {}
    anomaly_evidence: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    try:
        iterator = ET.iterparse(hashing_xml, events=("start", "end"))
        for event, element in iterator:
            xml_events += 1
            if xml_events > budget.max_xml_events:
                raise MediaWikiAdapterError("MediaWiki XML event 超预算")
            if event != "end" or _local_name(element.tag) != "page":
                continue
            page_count += 1
            if page_count > budget.max_pages:
                raise MediaWikiAdapterError("MediaWiki page count 超预算")
            try:
                index_entry = next(index_entries)
            except StopIteration as error:
                raise MediaWikiAdapterError("MediaWiki index 比 XML 短")
            try:
                page_id, title, namespace_id = _page_identity(element)
            except MediaWikiPageError as error:
                raise MediaWikiAdapterError("MediaWiki page identity 非法") from error
            if (index_entry.page_id != page_id
                    or index_entry.title != title):
                raise MediaWikiAdapterError("MediaWiki XML/index page identity 不一致")
            if namespace_id != 0:
                skipped_count += 1
                element.clear()
                continue
            main_count += 1
            try:
                record = parse_mediawiki_page(
                    element,
                    source_key=source_key,
                    extract_templates=extract_templates,
                    budget=budget,
                )
            except MediaWikiPageError as error:
                _counter_add(anomaly_codes, error.code)
                anomaly_evidence.append({
                    "code": error.code,
                    "page_id": page_id,
                    "title_sha256": hashlib.sha256(
                        title.encode("utf-8")).hexdigest(),
                })
                element.clear()
                continue
            valid_count += 1
            text_size += record.text_size_bytes
            template_count += len(record.templates)
            max_text = max(max_text, record.text_size_bytes)
            max_templates = max(max_templates, len(record.templates))
            digest.update(canonical_json_line(record.to_event_dict()))
            element.clear()
    except ET.ParseError as error:
        raise MediaWikiAdapterError("MediaWiki XML 损坏") from error
    try:
        next(index_entries)
    except StopIteration:
        pass
    else:
        raise MediaWikiAdapterError("MediaWiki index 比 XML 长")
    return MediaWikiScanReport(
        source_key=source_key,
        xml_sha256=hashing_xml.hexdigest(),
        index_sha256=hashing_index.hexdigest(),
        page_count=page_count,
        main_namespace_count=main_count,
        skipped_namespace_count=skipped_count,
        valid_page_count=valid_count,
        anomaly_count=len(anomaly_evidence),
        anomaly_codes=CanonicalJsonObject.from_value(anomaly_codes),
        anomaly_evidence=CanonicalJsonObject.from_value({
            "items": anomaly_evidence,
        }),
        text_size_bytes=text_size,
        template_count=template_count,
        max_page_text_bytes=max_text,
        max_template_count=max_templates,
        xml_event_count=xml_events,
        work_unit_count=xml_events + page_count + text_size + template_count,
        full_eof_verified=1,
        event_sha256=digest.hexdigest(),
    )


def scan_mediawiki_sample(
        xml_path: str | Path,
        index_path: str | Path,
        *,
        source_key: str,
        extract_templates: bool,
        budget: MediaWikiScanBudget,
        ) -> MediaWikiScanReport:
    """扫描未压缩极小 sample，供 adapter T0 和预算探针使用。"""
    try:
        with Path(xml_path).open("rb") as xml_stream:
            with Path(index_path).open("rb") as index_stream:
                return scan_mediawiki_streams(
                    xml_stream,
                    index_stream,
                    source_key=source_key,
                    extract_templates=extract_templates,
                    budget=budget,
                )
    except OSError as error:
        raise MediaWikiAdapterError("MediaWiki sample 无法读取") from error


def scan_mediawiki_bz2(
        xml_path: str | Path,
        index_path: str | Path,
        *,
        source_key: str,
        extract_templates: bool,
        budget: MediaWikiScanBudget,
        expected_xml_sha256: str,
        expected_index_sha256: str,
        ) -> MediaWikiScanReport:
    """核压缩文件 hash 后流式解压 XML/index，支持 concatenated bzip2 stream。"""
    xml_file = Path(xml_path)
    index_file = Path(index_path)
    try:
        xml_sha256 = _sha256_path(xml_file)
        index_sha256 = _sha256_path(index_file)
    except OSError as error:
        raise MediaWikiAdapterError("MediaWiki bz2 文件无法读取") from error
    if (xml_sha256 != expected_xml_sha256
            or index_sha256 != expected_index_sha256):
        raise MediaWikiAdapterError("MediaWiki bz2 SHA-256 不匹配")
    try:
        with bz2.open(xml_file, "rb") as xml_stream:
            with bz2.open(index_file, "rb") as index_stream:
                return scan_mediawiki_streams(
                    xml_stream,
                    index_stream,
                    source_key=source_key,
                    extract_templates=extract_templates,
                    budget=budget,
                )
    except (OSError, EOFError) as error:
        raise MediaWikiAdapterError("MediaWiki bz2 解压失败") from error


__all__ = [
    "MediaWikiAdapterError",
    "MediaWikiPageError",
    "MediaWikiPageRecord",
    "MediaWikiScanBudget",
    "MediaWikiScanReport",
    "MediaWikiTemplateSpan",
    "MultistreamIndexEntry",
    "SOURCE_KEYS",
    "extract_balanced_templates",
    "iter_multistream_index",
    "parse_mediawiki_page",
    "parse_multistream_index_line",
    "scan_mediawiki_bz2",
    "scan_mediawiki_sample",
    "scan_mediawiki_streams",
]
