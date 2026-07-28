"""D-02 闸后 CC_CEDICT_20260725 raw snapshot、parser 与许可冲突 T0。"""
from __future__ import annotations

import gzip
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_cc_cedict_adapter import (
    EXPECTED_LICENSE_ID,
    OBSERVED_LICENSE_ID,
    CcCedictAdapterError,
    audit_cc_cedict_license,
    parse_cc_cedict_entry,
    read_cc_cedict_sample,
    scan_cc_cedict_gzip,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_raw_snapshot import (
    HttpCaptureMetadata,
    RawSnapshotError,
    ThirdPartyRawSnapshotManifest,
    read_http_capture,
    read_raw_snapshot_manifest,
    verify_raw_snapshot,
    write_raw_snapshot_manifest,
)


SAMPLE_PATH = Path(
    "data/ph2/cc_cedict_20260725_header_and_rows_v1.txt.sample")
MANIFEST_PATH = Path(
    "data/ph2/manifests/cc_cedict_20260725.raw_snapshot.json")


def _gzip(path: Path, payload: bytes) -> str:
    """写 deterministic gzip 测试输入并返回压缩 SHA-256。"""
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _headers(path: Path, *, length: int) -> None:
    """写 curl dump-header 风格的最终 200 响应。"""
    path.write_bytes((
        "HTTP/1.1 200 OK\r\n"
        "Date: Sun, 26 Jul 2026 01:15:12 GMT\r\n"
        "Last-Modified: Sat, 25 Jul 2026 09:16:04 GMT\r\n"
        "ETag: \"3c8214-6576bf0415885\"\r\n"
        f"Content-Length: {length}\r\n"
        "Content-Type: application/x-gzip\r\n\r\n"
    ).encode("ascii"))


def _manifest(
        raw: Path,
        headers: Path,
        report,
        ) -> ThirdPartyRawSnapshotManifest:
    """构造当前许可冲突的 snapshot manifest。"""
    return ThirdPartyRawSnapshotManifest(
        1,
        "CC_CEDICT_20260725",
        "20260725",
        "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz",
        "CC_CEDICT_20260725/cedict.txt.gz",
        hashlib.sha256(raw.read_bytes()).hexdigest(),
        raw.stat().st_size,
        "gzip",
        "UTF-8",
        1,
        1,
        read_http_capture(
            headers,
            response_url=(
                "https://www.mdbg.net/chinese/export/cedict/"
                "cedict_1_0_ts_utf-8_mdbg.txt.gz"),
        ),
        EXPECTED_LICENSE_ID,
        OBSERVED_LICENSE_ID,
        "https://creativecommons.org/licenses/by-sa/4.0/",
        "CONFLICT",
        "BLOCKED",
        0,
        "LICENSE_PARTITION_MISMATCH",
        "CC-CEDICT published by MDBG",
        report.declared_entry_count,
        report.entry_count,
        report.anomaly_count,
        report.event_sha256,
        CanonicalJsonObject.from_value(report.to_dict()),
    )


def test_sample_restores_entries_and_never_marks_english_gloss_authoritative():
    """极小公开 sample 保留繁简/读音/gloss，但 gloss 权威位固定为零。"""
    entries = read_cc_cedict_sample(SAMPLE_PATH)
    assert len(entries) == 8
    assert entries[0].traditional == "110"
    assert entries[2].traditional == "11區"
    assert entries[2].simplified == "11区"
    assert entries[6].glosses == (
        "computers, communications, and consumer electronics",
        "China Compulsory Certificate (CCC)",
    )
    assert all(item.to_dict()["english_gloss_authoritative"] == 0
               for item in entries)


def test_repository_snapshot_manifest_freezes_actual_full_scan_and_blocker():
    """仓库只引用 raw hash/统计，不收录 3.9 MB 原始内容或伪造可发布 pack。"""
    manifest = read_raw_snapshot_manifest(MANIFEST_PATH)
    assert manifest.sha256() == (
        "7c148fd121d90aab616b8fb804631cd92f3b1d522c4776ff2cd1a3c3036886fc")
    assert manifest.raw_sha256 == (
        "c745acaa8d549e6fd3a6cadadf5481c018eef0a0e3dbb2c704c3969c9f1685d3")
    assert manifest.raw_size_bytes == 3965460
    assert manifest.declared_record_count == 124732
    assert manifest.parsed_record_count == 124732
    assert manifest.anomaly_count == 0
    assert manifest.parser_report.to_value()["terminal_newline_present"] == 0
    assert manifest.license_status == "CONFLICT"
    assert manifest.redistribution_policy == "BLOCKED"
    assert manifest.release_eligible == 0
    assert manifest.blocker_code == "LICENSE_PARTITION_MISMATCH"


def test_actual_header_license_conflicts_with_frozen_document_partition():
    """实际 `#! license=...4.0/` 不得被静默标成文档中的 3.0。"""
    lines = SAMPLE_PATH.read_text(encoding="utf-8").splitlines()
    metadata = tuple(sorted(
        line[3:].split("=", 1) for line in lines if line.startswith("#! ")))
    audit = audit_cc_cedict_license(metadata)
    assert audit.expected_license_id == "CC-BY-SA-3.0"
    assert audit.observed_license_id == "CC-BY-SA-4.0"
    assert audit.status == "CONFLICT"
    assert audit.blocker_code == "LICENSE_PARTITION_MISMATCH"


def test_parser_rejects_bad_shape_empty_gloss_and_outer_whitespace():
    """坏行精确失败，不猜 traditional/simplified/pinyin/gloss。"""
    with pytest.raises(CcCedictAdapterError, match="格式非法"):
        parse_cc_cedict_entry("甲 甲 /missing pinyin/", line_number=1)
    with pytest.raises(CcCedictAdapterError, match="gloss 为空"):
        parse_cc_cedict_entry("甲 甲 [jia3] /a//", line_number=1)
    with pytest.raises(CcCedictAdapterError, match="首尾空白"):
        parse_cc_cedict_entry(" 甲 甲 [jia3] /a/", line_number=1)


def test_gzip_scan_is_double_pass_stable_and_counts_anomaly(tmp_path):
    """同一 gzip 双遍事件摘要一致；坏行只进入 anomaly，不污染好行。"""
    sample = SAMPLE_PATH.read_bytes()
    gzip_path = tmp_path / "sample.gz"
    digest = _gzip(gzip_path, sample)
    first = scan_cc_cedict_gzip(
        gzip_path,
        expected_compressed_sha256=digest,
        complete_snapshot=False,
    )
    second = scan_cc_cedict_gzip(
        gzip_path,
        expected_compressed_sha256=digest,
        complete_snapshot=False,
    )
    assert first == second
    assert first.entry_count == 8
    assert first.anomaly_count == 0
    assert first.declared_entry_count == 124732
    assert first.terminal_newline_present == 1
    assert first.license_audit.status == "CONFLICT"

    damaged = sample + b"bad row\n"
    bad_path = tmp_path / "bad.gz"
    bad_digest = _gzip(bad_path, damaged)
    report = scan_cc_cedict_gzip(
        bad_path,
        expected_compressed_sha256=bad_digest,
        complete_snapshot=False,
    )
    assert report.entry_count == 8
    assert report.anomaly_count == 1
    assert report.event_sha256 != first.event_sha256


def test_complete_snapshot_requires_declared_count_and_zero_anomaly(tmp_path):
    """sample 不得冒充完整快照；完整模式必须与 header entries 精确一致。"""
    gzip_path = tmp_path / "sample.gz"
    digest = _gzip(gzip_path, SAMPLE_PATH.read_bytes())
    with pytest.raises(CcCedictAdapterError, match="完整快照"):
        scan_cc_cedict_gzip(
            gzip_path,
            expected_compressed_sha256=digest,
            complete_snapshot=True,
        )


def test_bad_compressed_hash_truncated_gzip_and_midread_change_fail(tmp_path):
    """压缩 hash 与 gzip 完整性均 fail-closed。"""
    gzip_path = tmp_path / "sample.gz"
    digest = _gzip(gzip_path, SAMPLE_PATH.read_bytes())
    with pytest.raises(CcCedictAdapterError, match="SHA-256"):
        scan_cc_cedict_gzip(
            gzip_path,
            expected_compressed_sha256="0" * 64,
            complete_snapshot=False,
        )
    truncated = tmp_path / "truncated.gz"
    truncated.write_bytes(gzip_path.read_bytes()[:-8])
    with pytest.raises(CcCedictAdapterError, match="gzip"):
        scan_cc_cedict_gzip(
            truncated,
            expected_compressed_sha256=hashlib.sha256(
                truncated.read_bytes()).hexdigest(),
            complete_snapshot=False,
        )
    assert hashlib.sha256(gzip_path.read_bytes()).hexdigest() == digest


def test_http_capture_and_snapshot_manifest_round_trip(tmp_path):
    """raw byte、response headers、许可冲突和 parser 摘要共同规范封存。"""
    raw_root = tmp_path / "raw"
    raw = raw_root / "CC_CEDICT_20260725" / "cedict.txt.gz"
    raw.parent.mkdir(parents=True)
    digest = _gzip(raw, SAMPLE_PATH.read_bytes())
    headers = tmp_path / "response.headers"
    _headers(headers, length=raw.stat().st_size)
    report = scan_cc_cedict_gzip(
        raw,
        expected_compressed_sha256=digest,
        complete_snapshot=False,
    )
    manifest = _manifest(raw, headers, report)
    output = tmp_path / "manifests" / "snapshot.json"
    write_raw_snapshot_manifest(manifest, output)
    restored = read_raw_snapshot_manifest(output)
    assert restored == manifest
    assert restored.sha256() == hashlib.sha256(output.read_bytes()).hexdigest()
    verify_raw_snapshot(restored, raw_root)
    assert restored.license_status == "CONFLICT"
    assert restored.release_eligible == 0
    assert restored.redistribution_policy == "BLOCKED"


def test_snapshot_manifest_rejects_bad_size_match_and_overwrite(tmp_path):
    """HTTP size、许可裁决和不可覆盖边界不能被手工放宽。"""
    http = HttpCaptureMetadata(
        200,
        "https://example.invalid/data.gz",
        "Sun, 26 Jul 2026 01:15:12 GMT",
        "Sat, 25 Jul 2026 09:16:04 GMT",
        "\"etag\"",
        "application/x-gzip",
        10,
        "1" * 64,
    )
    base = ThirdPartyRawSnapshotManifest(
        1, "SOURCE", "v1", "https://example.invalid/data.gz",
        "SOURCE/data.gz", "2" * 64, 10, "gzip", "UTF-8", 1, 1,
        http, "CC-BY-SA-3.0", "CC-BY-SA-4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/",
        "CONFLICT", "BLOCKED", 0, "LICENSE_PARTITION_MISMATCH",
        "attribution", 1, 1, 0, "3" * 64,
        CanonicalJsonObject.from_value({"probe": 1}),
    )
    with pytest.raises(RawSnapshotError, match="Content-Length"):
        replace(base, raw_size_bytes=9)
    with pytest.raises(RawSnapshotError, match="MATCH 许可"):
        replace(base, license_status="MATCH", redistribution_policy="PUBLIC",
                release_eligible=1, blocker_code="")
    path = tmp_path / "snapshot.json"
    write_raw_snapshot_manifest(base, path)
    path.write_bytes(canonical_json_line({"bad": 1}))
    with pytest.raises(RawSnapshotError, match="内容不同"):
        write_raw_snapshot_manifest(base, path)


def test_http_capture_rejects_missing_or_duplicate_required_header(tmp_path):
    """HTTP 获取证据缺字段或重复字段时不形成 snapshot。"""
    missing = tmp_path / "missing.headers"
    missing.write_bytes(
        b"HTTP/1.1 200 OK\r\n"
        b"Date: Sun, 26 Jul 2026 01:15:12 GMT\r\n\r\n")
    with pytest.raises(RawSnapshotError, match="缺少"):
        read_http_capture(missing, response_url="https://example.invalid/data")
    duplicate = tmp_path / "duplicate.headers"
    _headers(duplicate, length=1)
    payload = duplicate.read_bytes().replace(
        b"Content-Length: 1\r\n",
        b"Content-Length: 1\r\nContent-Length: 1\r\n",
    )
    duplicate.write_bytes(payload)
    with pytest.raises(RawSnapshotError, match="重复"):
        read_http_capture(duplicate, response_url="https://example.invalid/data")
