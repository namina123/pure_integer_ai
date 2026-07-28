"""D-02 闸后 ConceptNet 5.7.0 raw、许可分区和纯整数 parser T0。"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_conceptnet_adapter import (
    ConceptNetAdapterError,
    ENDPOINT_CONCEPT,
    decimal_text_to_ratio,
    parse_conceptnet_assertion,
    parse_conceptnet_endpoint,
    read_conceptnet_sample,
    scan_conceptnet_gzip,
)
from pure_integer_ai.experiments.ph2_raw_snapshot import (
    HttpCaptureMetadata,
    RawSnapshotError,
    read_http_capture,
)


BY_PATH = Path("data/ph2/conceptnet_5_7_0_cc_by_4_0_zh_v1.csv.sample")
BY_SA_PATH = Path(
    "data/ph2/conceptnet_5_7_0_cc_by_sa_4_0_zh_v1.csv.sample")
MANIFEST_PATH = Path(
    "data/ph2/manifests/conceptnet_5_7_0.raw_snapshot.json")


def _gzip(path: Path, payload: bytes) -> str:
    """写 deterministic gzip 测试输入并返回压缩 SHA-256。"""
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(path: Path, index: int = 0) -> str:
    """读取 sample 的指定一行。"""
    return path.read_text(encoding="utf-8").splitlines()[index]


def test_samples_are_physically_license_partitioned_and_preserve_sources():
    """两个 sample 文件各只含一种许可，sources/dataset/weight 均可恢复。"""
    by = read_conceptnet_sample(BY_PATH)
    by_sa = read_conceptnet_sample(BY_SA_PATH)
    assert len(by) == 2 and len(by_sa) == 2
    assert {item.license_partition for item in by} == {"CC-BY-4.0"}
    assert {item.license_partition for item in by_sa} == {"CC-BY-SA-4.0"}
    assert by[1].weight_text == "2.0"
    assert (by[1].weight_numerator, by[1].weight_denominator) == (2, 1)
    assert len(by[1].sources) == 2
    assert by_sa[1].start.suffix == ("a",)
    assert all(item.start.language == "zh" and item.end.language == "zh"
               for item in by + by_sa)


def test_uri_relation_and_authority_boundaries_are_explicit():
    """assertion URI 必须匹配三元组，外部 relation 不冒充项目关系或真值。"""
    assertion = parse_conceptnet_assertion(_line(BY_SA_PATH), line_number=65800)
    assert assertion.start.kind == ENDPOINT_CONCEPT
    assert assertion.relation == "/r/Antonym"
    assert assertion.assertion_uri == (
        "/a/[/r/Antonym/,/c/zh/上/,/c/zh/下/]")
    exported = assertion.to_dict()
    assert exported["project_relation_authoritative"] == 0
    assert exported["definitive_truth_authoritative"] == 0
    bad = _line(BY_SA_PATH).replace(
        "/a/[/r/Antonym/", "/a/[/r/Synonym/", 1)
    with pytest.raises(ConceptNetAdapterError, match="三元组"):
        parse_conceptnet_assertion(bad, line_number=1)


def test_decimal_weight_is_minimal_integer_ratio_without_binary_float():
    """十进制 weight 只走整数约分，0、负数、指数和坏小数均拒绝。"""
    assert decimal_text_to_ratio("1.0") == (1, 1)
    assert decimal_text_to_ratio("2.50") == (5, 2)
    assert decimal_text_to_ratio("0.125") == (1, 8)
    for bad in ("0", "0.0", "-1.0", ".5", "1.", "1e-3", "01.0"):
        with pytest.raises(ConceptNetAdapterError, match="weight"):
            decimal_text_to_ratio(bad)


def test_endpoint_keeps_language_term_suffix_and_opaque_external_uri():
    """concept URI 不丢 POS/sense suffix，非 concept endpoint 不猜语言。"""
    concept = parse_conceptnet_endpoint("/c/zh/上/a/example")
    assert concept.to_dict() == {
        "kind": 1,
        "language": "zh",
        "suffix": ["a", "example"],
        "term": "上",
        "uri": "/c/zh/上/a/example",
    }
    external = parse_conceptnet_endpoint("https://example.invalid/item")
    assert external.kind == 2
    assert external.language == ""
    with pytest.raises(ConceptNetAdapterError, match="URI"):
        parse_conceptnet_endpoint("not-a-uri")
    trailing = parse_conceptnet_endpoint("/c/cs/apostrophe/n/wp/")
    assert trailing.suffix == ("n", "wp")


def test_external_url_assertion_uses_conceptnet_component_slashes():
    """ExternalURL 在 assertion URI 内补前导 slash，列值仍保留原始 URL。"""
    line = (
        "/a/[/r/ExternalURL/,/c/aa/a/,/http://en.wiktionary.org/wiki/a/]\t"
        "/r/ExternalURL\t/c/aa/a\thttp://en.wiktionary.org/wiki/a\t"
        '{"dataset": "/d/wiktionary/en", "license": "cc:by-sa/4.0", '
        '"sources": [{"contributor": "/s/resource/wiktionary/en"}], '
        '"weight": 0.25}'
    )
    assertion = parse_conceptnet_assertion(line, line_number=1)
    assert assertion.end.uri == "http://en.wiktionary.org/wiki/a"
    assert assertion.weight_numerator == 1
    assert assertion.weight_denominator == 4


def test_bad_license_weight_metadata_and_sources_fail_closed():
    """坏许可、quoted weight、重复 key/source 和坏 JSON 不形成 assertion。"""
    base = _line(BY_PATH)
    variants = (
        base.replace("cc:by/4.0", "cc:zero/1.0"),
        base.replace('"weight": 1.0', '"weight": "1.0"'),
        base.replace('"weight": 1.0}', '"weight": 1.0, "weight": 2.0}'),
        base.replace(
            '"sources": [{"activity": "/s/activity/ptt/petgame", '
            '"contributor": "/s/contributor/petgame/pet_20310"}]',
            '"sources": [{"activity": "/s/activity/ptt/petgame", '
            '"contributor": "/s/contributor/petgame/pet_20310"}, '
            '{"activity": "/s/activity/ptt/petgame", '
            '"contributor": "/s/contributor/petgame/pet_20310"}]',
        ),
        base[:-1],
    )
    for variant in variants:
        with pytest.raises(ConceptNetAdapterError):
            parse_conceptnet_assertion(variant, line_number=1)


def test_gzip_scan_splits_license_hashes_and_is_double_pass_stable(tmp_path):
    """同一 gzip 双遍稳定，并分别形成 BY/BY-SA count 与 event hash。"""
    path = tmp_path / "sample.csv.gz"
    payload = BY_SA_PATH.read_bytes() + BY_PATH.read_bytes()
    digest = _gzip(path, payload)
    first = scan_conceptnet_gzip(
        path,
        expected_compressed_sha256=digest,
        complete_snapshot=True,
    )
    second = scan_conceptnet_gzip(
        path,
        expected_compressed_sha256=digest,
        complete_snapshot=True,
    )
    assert first == second
    assert first.line_count == first.assertion_count == 4
    assert first.anomaly_count == 0
    assert first.zh_endpoint_count == first.zh_zh_count == 4
    assert first.license_counts.to_value() == {
        "CC-BY-4.0": 2,
        "CC-BY-SA-4.0": 2,
    }
    assert len(set(first.license_event_sha256.to_value().values())) == 2
    assert first.target_relation_counts.to_value()["/r/Antonym"]["zh_zh"] == 2


def test_scan_quarantines_bad_row_and_rejects_order_hash_and_truncation(tmp_path):
    """坏行只记 anomaly；坏序、坏压缩 hash 和截断均不能冒充完整快照。"""
    good_payload = BY_SA_PATH.read_bytes() + BY_PATH.read_bytes()
    damaged = tmp_path / "damaged.gz"
    damaged_digest = _gzip(damaged, good_payload + b"bad\trow\n")
    report = scan_conceptnet_gzip(
        damaged,
        expected_compressed_sha256=damaged_digest,
        complete_snapshot=False,
    )
    assert report.assertion_count == 4
    assert report.anomaly_count == 1
    assert report.anomaly_codes.to_value() == {"BAD_ROW": 1}
    with pytest.raises(ConceptNetAdapterError, match="完整快照"):
        scan_conceptnet_gzip(
            damaged,
            expected_compressed_sha256=damaged_digest,
            complete_snapshot=True,
        )

    reversed_path = tmp_path / "reversed.gz"
    reversed_digest = _gzip(
        reversed_path, BY_PATH.read_bytes() + BY_SA_PATH.read_bytes())
    reversed_report = scan_conceptnet_gzip(
        reversed_path,
        expected_compressed_sha256=reversed_digest,
        complete_snapshot=False,
    )
    assert reversed_report.anomaly_codes.to_value() == {"BAD_ORDER": 2}
    with pytest.raises(ConceptNetAdapterError, match="SHA-256"):
        scan_conceptnet_gzip(
            reversed_path,
            expected_compressed_sha256="0" * 64,
            complete_snapshot=False,
        )
    truncated = tmp_path / "truncated.gz"
    truncated.write_bytes(reversed_path.read_bytes()[:-8])
    with pytest.raises(ConceptNetAdapterError, match="gzip"):
        scan_conceptnet_gzip(
            truncated,
            expected_compressed_sha256=hashlib.sha256(
                truncated.read_bytes()).hexdigest(),
            complete_snapshot=False,
        )


def test_http_capture_records_explicit_utc_without_changing_old_manifests(tmp_path):
    """新 snapshot 可记录显式 UTC；空值保持旧 manifest 规范字节兼容。"""
    headers = tmp_path / "headers"
    headers.write_bytes(
        b"HTTP/1.1 200 OK\r\n"
        b"Date: Sun, 26 Jul 2026 02:13:10 GMT\r\n"
        b"Last-Modified: Wed, 03 Jul 2019 15:47:25 GMT\r\n"
        b"ETag: \"etag\"\r\n"
        b"Content-Type: application/x-gzip\r\n"
        b"Content-Length: 4\r\n\r\n"
    )
    captured = read_http_capture(
        headers,
        response_url="https://example.invalid/data.gz",
        captured_at_utc="2026-07-26T02:22:52.9910333Z",
    )
    assert captured.to_dict()["captured_at_utc"] == (
        "2026-07-26T02:22:52.9910333Z")
    old_shape = HttpCaptureMetadata(
        200,
        "https://example.invalid/data.gz",
        "Sun, 26 Jul 2026 02:13:10 GMT",
        "Wed, 03 Jul 2019 15:47:25 GMT",
        '"etag"',
        "application/x-gzip",
        4,
        "1" * 64,
    )
    assert "captured_at_utc" not in old_shape.to_dict()
    with pytest.raises(RawSnapshotError, match="UTC ISO-8601"):
        HttpCaptureMetadata(
            200,
            "https://example.invalid/data.gz",
            "Sun, 26 Jul 2026 02:13:10 GMT",
            "Wed, 03 Jul 2019 15:47:25 GMT",
            '"etag"',
            "application/x-gzip",
            4,
            "1" * 64,
            "2026-07-26 02:22:52",
        )


def test_repository_manifest_freezes_actual_full_scan_and_two_partitions():
    """正式 raw 引用冻结全量 hash/统计/许可分区，不把 498 MB 放入 Git。"""
    from pure_integer_ai.experiments.ph2_raw_snapshot import (
        read_raw_snapshot_manifest,
    )

    manifest = read_raw_snapshot_manifest(MANIFEST_PATH)
    assert manifest.sha256() == (
        "36195b2eee75bc3ef96722ffe3ed67de58e70c5067c00d4369db687dc3893e75")
    assert manifest.source_key == "CONCEPTNET_5_7_0"
    assert manifest.raw_sha256 == (
        "accd65fe94038584295574ddc26e1500c1919c8c4532bf771811cafd0948af7e")
    assert manifest.raw_size_bytes == 497963447
    assert manifest.http.captured_at_utc == "2026-07-26T02:22:52.9910333Z"
    assert manifest.license_status == "MATCH"
    assert manifest.redistribution_policy == "PUBLIC"
    report = manifest.parser_report.to_value()
    assert report["line_count"] == 34074917
    assert report["assertion_count"] == 34074915
    assert report["anomaly_count"] == 2
    assert report["anomaly_codes"] == {"BAD_ASSERTION_URI": 1, "BAD_URI": 1}
    assert report["scan_passes"] == 2
    assert report["full_eof_verified"] == 1
    assert report["license_counts"] == {
        "CC-BY-4.0": 2859296,
        "CC-BY-SA-4.0": 31215619,
    }
    assert report["zh_zh_count"] == 491882
    assert report["target_relation_counts"]["/r/Causes"]["zh_zh"] == 70048
