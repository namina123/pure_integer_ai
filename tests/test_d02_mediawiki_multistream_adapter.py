"""D-02 Wikimedia multistream page cluster、模板与预算 adapter T0。"""
from __future__ import annotations

import bz2
import hashlib
import io
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    MediaWikiAdapterError,
    MediaWikiPageError,
    MediaWikiScanBudget,
    MediaWikiScanReport,
    extract_balanced_templates,
    iter_multistream_index,
    parse_mediawiki_page,
    parse_multistream_index_line,
    scan_mediawiki_bz2,
    scan_mediawiki_sample,
    scan_mediawiki_streams,
)


WIKTIONARY_XML = Path(
    "data/ph2/zhwiktionary_20260701_multistream_v1.xml.schema.sample")
WIKTIONARY_INDEX = Path(
    "data/ph2/zhwiktionary_20260701_multistream_index_v1.txt.schema.sample")
WIKIPEDIA_XML = Path(
    "data/ph2/zhwiki_20260701_multistream_v1.xml.schema.sample")
WIKIPEDIA_INDEX = Path(
    "data/ph2/zhwiki_20260701_multistream_index_v1.txt.schema.sample")


def _budget(**overrides) -> MediaWikiScanBudget:
    """构造小批 sample 使用的确定性整数预算。"""
    values = {
        "max_pages": 20,
        "max_template_depth": 8,
        "max_templates_per_page": 20,
        "max_text_bytes_per_page": 4096,
        "max_xml_events": 1000,
    }
    values.update(overrides)
    return MediaWikiScanBudget(**values)


def _local_name(tag: str) -> str:
    """返回测试中 ElementTree tag 的 local name。"""
    return tag.rsplit("}", 1)[-1]


def _first_main_page(path: Path):
    """从 sample 返回首个主命名空间 page。"""
    root = ET.fromstring(path.read_bytes())
    for element in root.iter():
        if _local_name(element.tag) != "page":
            continue
        ns = next(child for child in element if _local_name(child.tag) == "ns")
        if ns.text == "0":
            return element
    raise AssertionError("sample 缺少主命名空间 page")


def _generated_streams(page_count: int) -> tuple[io.BytesIO, io.BytesIO]:
    """生成同构 page/index 流，用于 2/4/8 线性预算探针。"""
    pages: list[str] = []
    index_lines: list[str] = []
    for page_id in range(1, page_count + 1):
        pages.append(
            f"<page><title>页{page_id}</title><ns>0</ns><id>{page_id}</id>"
            f"<revision><id>{1000 + page_id}</id>"
            "<timestamp>2026-07-01T00:00:00Z</timestamp>"
            "<contributor><username>用户</username><id>1</id></contributor>"
            "<model>wikitext</model><format>text/x-wiki</format>"
            "<text xml:space=\"preserve\">固定文本</text>"
            "<sha1>syntheticsha1</sha1></revision></page>"
        )
        index_lines.append(f"0:{page_id}:页{page_id}\n")
    xml = ("<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.11/\">"
           + "".join(pages) + "</mediawiki>").encode("utf-8")
    index = "".join(index_lines).encode("utf-8")
    return io.BytesIO(xml), io.BytesIO(index)


def test_index_parser_keeps_title_colons_and_rejects_float_or_bad_order():
    """offset/page id 只走整数，title 后续冒号不被误拆。"""
    entry = parse_multistream_index_line(
        b"100:7:Talk:Example\n", line_number=1)
    assert (entry.offset, entry.page_id, entry.title) == (
        100, 7, "Talk:Example")
    escaped = parse_multistream_index_line(
        b"101:8:c.&amp;f\n", line_number=2)
    assert escaped.title == "c.&f"
    for bad in (b"1.0:7:title\n", b"01:7:title\n", b"1:07:title\n"):
        with pytest.raises(MediaWikiAdapterError, match="整数"):
            parse_multistream_index_line(bad, line_number=1)
    with pytest.raises(MediaWikiAdapterError, match="实体"):
        parse_multistream_index_line(
            b"1:7:bad&copy;title\n", line_number=1)
    with pytest.raises(MediaWikiAdapterError, match="单调"):
        list(iter_multistream_index(io.BytesIO(
            b"0:2:b\n0:1:a\n")))


def test_balanced_template_scanner_keeps_nested_provenance_and_limits():
    """只返回 top-level 模板，嵌套、hash、深度和未关闭都可审计。"""
    text = "前{{外层|x={{内层|y}}}}后{{第二|z}}"
    templates = extract_balanced_templates(
        text, max_templates=2, max_depth=4)
    assert [item.name for item in templates] == ["外层", "第二"]
    assert all(text[item.start:item.end].startswith("{{") for item in templates)
    assert templates[0].raw_sha256 == hashlib.sha256(
        text[templates[0].start:templates[0].end].encode("utf-8")).hexdigest()
    commented = extract_balanced_templates(
        "{{pdc-decl-pronoun<!--\n-->|x}}",
        max_templates=1,
        max_depth=2,
    )
    assert commented[0].name == "pdc-decl-pronoun<!--\n-->"
    with pytest.raises(MediaWikiPageError, match="未关闭"):
        extract_balanced_templates("{{坏模板", max_templates=2, max_depth=2)
    with pytest.raises(MediaWikiPageError, match="count"):
        extract_balanced_templates(text, max_templates=1, max_depth=4)


def test_wiktionary_sample_is_page_clustered_and_bad_template_is_atomic():
    """词典只收主空间；坏模板整页 anomaly，不从残片猜词义标签。"""
    report = scan_mediawiki_sample(
        WIKTIONARY_XML,
        WIKTIONARY_INDEX,
        source_key="ZHWIKTIONARY_20260701",
        extract_templates=True,
        budget=_budget(),
    )
    assert report.page_count == 3
    assert report.main_namespace_count == 2
    assert report.skipped_namespace_count == 1
    assert report.valid_page_count == 1
    assert report.anomaly_count == 1
    assert report.anomaly_codes.to_value() == {"UNBALANCED_TEMPLATE": 1}
    assert report.template_count == 2
    assert report.max_template_count == 2


def test_scan_report_round_trip_preserves_anomaly_evidence():
    """规范 scan report 可逐字段恢复，页级异常证据不得在封存时丢失。"""
    report = scan_mediawiki_sample(
        WIKTIONARY_XML,
        WIKTIONARY_INDEX,
        source_key="ZHWIKTIONARY_20260701",
        extract_templates=True,
        budget=_budget(),
    )
    restored = MediaWikiScanReport.from_dict(report.to_dict())
    assert restored == report
    assert restored.anomaly_evidence == report.anomaly_evidence


def test_wikipedia_sample_supports_slots_redirect_and_zero_truth_authority():
    """百科 direct/slot 两结构都保留，redirect 与 truth 边界显式。"""
    report = scan_mediawiki_sample(
        WIKIPEDIA_XML,
        WIKIPEDIA_INDEX,
        source_key="ZHWIKIPEDIA_20260701",
        extract_templates=False,
        budget=_budget(),
    )
    assert report.page_count == 3
    assert report.valid_page_count == 2
    assert report.skipped_namespace_count == 1
    assert report.anomaly_count == 0
    page = _first_main_page(WIKIPEDIA_XML)
    record = parse_mediawiki_page(
        page,
        source_key="ZHWIKIPEDIA_20260701",
        extract_templates=False,
        budget=_budget(),
    )
    event = record.to_event_dict()
    assert event["cluster_id"] == "ZHWIKIPEDIA_20260701:page:10"
    assert event["definitive_truth_authoritative"] == 0
    assert event["revision_id"] == 201
    assert event["parent_revision_id"] == 200
    assert event["contributor"]["kind"] == "registered"


def test_xml_index_identity_and_extra_line_fail_closed():
    """page id/title 必须与 index 一一对应，短/长 index 均不能继续。"""
    xml = WIKIPEDIA_XML.read_bytes()
    bad_identity = WIKIPEDIA_INDEX.read_bytes().replace(
        "窗户".encode(), "门口".encode(), 1)
    with pytest.raises(MediaWikiAdapterError, match="identity"):
        scan_mediawiki_streams(
            io.BytesIO(xml),
            io.BytesIO(bad_identity),
            source_key="ZHWIKIPEDIA_20260701",
            extract_templates=False,
            budget=_budget(),
        )
    for index in (
            WIKIPEDIA_INDEX.read_bytes().splitlines(keepends=True)[0],
            WIKIPEDIA_INDEX.read_bytes() + b"3000:13:extra\n"):
        with pytest.raises(MediaWikiAdapterError, match="index"):
            scan_mediawiki_streams(
                io.BytesIO(xml),
                io.BytesIO(index),
                source_key="ZHWIKIPEDIA_20260701",
                extract_templates=False,
                budget=_budget(),
            )


def test_page_event_text_is_hash_bound_and_does_not_embed_full_text():
    """页事件保存 text hash/size/provenance，不把整页误当结构标签。"""
    page = _first_main_page(WIKTIONARY_XML)
    record = parse_mediawiki_page(
        page,
        source_key="ZHWIKTIONARY_20260701",
        extract_templates=True,
        budget=_budget(),
    )
    event = record.to_event_dict()
    assert event["page_id"] == 1
    assert event["revision_id"] == 101
    assert event["templates"][0]["name"] == "漢語詞"
    assert "text" not in event
    assert len(event["text_sha256"]) == 64
    assert event["text_xml_attributes"]


def test_conversion_script_contributor_zero_id_is_preserved():
    """Wikimedia conversion script 的注册 contributor id 0 是合法 provenance。"""
    page = _first_main_page(WIKTIONARY_XML)
    for element in page.iter():
        if _local_name(element.tag) == "id" and element.text == "11":
            element.text = "0"
    record = parse_mediawiki_page(
        page,
        source_key="ZHWIKTIONARY_20260701",
        extract_templates=True,
        budget=_budget(),
    )
    assert record.contributor.to_value() == {
        "kind": "registered",
        "user_id": 0,
        "username": "示例用户",
    }


def test_integer_budgets_stop_pages_events_text_and_templates():
    """页、XML event、单页文本和模板预算分别 fail closed。"""
    for budget in (_budget(max_pages=2), _budget(max_xml_events=10)):
        with pytest.raises(MediaWikiAdapterError):
            scan_mediawiki_sample(
                WIKTIONARY_XML,
                WIKTIONARY_INDEX,
                source_key="ZHWIKTIONARY_20260701",
                extract_templates=True,
                budget=budget,
            )
    text_limited = scan_mediawiki_sample(
        WIKTIONARY_XML,
        WIKTIONARY_INDEX,
        source_key="ZHWIKTIONARY_20260701",
        extract_templates=True,
        budget=_budget(max_text_bytes_per_page=4),
    )
    assert text_limited.anomaly_codes.to_value() == {"TEXT_BUDGET": 2}
    template_limited = scan_mediawiki_sample(
        WIKTIONARY_XML,
        WIKTIONARY_INDEX,
        source_key="ZHWIKTIONARY_20260701",
        extract_templates=True,
        budget=_budget(max_templates_per_page=1),
    )
    assert template_limited.anomaly_codes.to_value() == {
        "TEMPLATE_COUNT_BUDGET": 1,
        "UNBALANCED_TEMPLATE": 1,
    }


def test_bz2_double_pass_is_stable_and_bad_hash_or_truncation_is_rejected(
        tmp_path,
        ):
    """压缩 sample 双遍一致，压缩 hash 与完整 bzip2 尾都必须成立。"""
    xml_path = tmp_path / "sample.xml.bz2"
    index_path = tmp_path / "sample-index.txt.bz2"
    xml_path.write_bytes(bz2.compress(WIKIPEDIA_XML.read_bytes()))
    index_path.write_bytes(bz2.compress(WIKIPEDIA_INDEX.read_bytes()))
    xml_hash = hashlib.sha256(xml_path.read_bytes()).hexdigest()
    index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
    first = scan_mediawiki_bz2(
        xml_path,
        index_path,
        source_key="ZHWIKIPEDIA_20260701",
        extract_templates=False,
        budget=_budget(),
        expected_xml_sha256=xml_hash,
        expected_index_sha256=index_hash,
    )
    second = scan_mediawiki_bz2(
        xml_path,
        index_path,
        source_key="ZHWIKIPEDIA_20260701",
        extract_templates=False,
        budget=_budget(),
        expected_xml_sha256=xml_hash,
        expected_index_sha256=index_hash,
    )
    assert first == second
    with pytest.raises(MediaWikiAdapterError, match="SHA-256"):
        scan_mediawiki_bz2(
            xml_path,
            index_path,
            source_key="ZHWIKIPEDIA_20260701",
            extract_templates=False,
            budget=_budget(),
            expected_xml_sha256="0" * 64,
            expected_index_sha256=index_hash,
        )
    xml_path.write_bytes(xml_path.read_bytes()[:-8])
    with pytest.raises(MediaWikiAdapterError, match="解压"):
        scan_mediawiki_bz2(
            xml_path,
            index_path,
            source_key="ZHWIKIPEDIA_20260701",
            extract_templates=False,
            budget=_budget(),
            expected_xml_sha256=hashlib.sha256(
                xml_path.read_bytes()).hexdigest(),
            expected_index_sha256=index_hash,
        )


def test_two_four_eight_page_probe_has_linear_work_and_constant_live_bounds():
    """2/4/8 页的增量严格线性，单页 live bounds 不随总量增长。"""
    reports = []
    for count in (2, 4, 8):
        xml_stream, index_stream = _generated_streams(count)
        reports.append(scan_mediawiki_streams(
            xml_stream,
            index_stream,
            source_key="ZHWIKIPEDIA_20260701",
            extract_templates=False,
            budget=_budget(),
        ))
    delta_two = reports[1].work_unit_count - reports[0].work_unit_count
    delta_four = reports[2].work_unit_count - reports[1].work_unit_count
    assert delta_four == 2 * delta_two
    assert [report.page_count for report in reports] == [2, 4, 8]
    assert len({report.max_page_text_bytes for report in reports}) == 1
    assert all(report.anomaly_count == 0 for report in reports)


def test_samples_have_frozen_hashes_and_are_not_claimed_as_upstream_bytes():
    """四个 schema sample hash 固定，明确只验证格式而非 dated dump 身份。"""
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            WIKTIONARY_XML, WIKTIONARY_INDEX, WIKIPEDIA_XML, WIKIPEDIA_INDEX)
    }
    assert hashes == {
        "zhwiki_20260701_multistream_index_v1.txt.schema.sample": (
            "c93491fad69b420c83fc767a369bece38840ef61e9297a7cde833219d3baa490"),
        "zhwiki_20260701_multistream_v1.xml.schema.sample": (
            "4ca65e0ad09c5126c6eefce2481a1a58fa27658556661a87dd8a439535a0592c"),
        "zhwiktionary_20260701_multistream_index_v1.txt.schema.sample": (
            "f77ac692d04d7f56b5d4e4dfe540044b5bfeea3348b96eaf380a76629f037a1c"),
        "zhwiktionary_20260701_multistream_v1.xml.schema.sample": (
            "78d68a0fa092f88948731f318cedff2af896566be9f8c55c6f010ec9a70efe3b"),
    }
