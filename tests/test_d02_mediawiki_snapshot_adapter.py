"""D-02 MediaWiki dated snapshot contract and destructive T0."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    MediaWikiScanReport,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    ATTRIBUTION_POLICY,
    WIKIPEDIA_ATTRIBUTION_POLICY,
    MediaWikiSnapshotAcquisition,
    MediaWikiSnapshotError,
    build_mediawiki_dump_snapshot,
    read_mediawiki_dump_snapshot,
    verify_mediawiki_dump_snapshot,
    write_mediawiki_dump_snapshot,
)


SOURCE_KEY = "ZHWIKTIONARY_20260701"
PROJECT = "zhwiktionary"
DUMP_DATE = "20260701"
MANIFEST_PATH = Path(
    "data/ph2/manifests/zhwiktionary_20260701.multistream_snapshot.json")
WIKIPEDIA_MANIFEST_PATH = Path(
    "data/ph2/manifests/zhwikipedia_20260701.multistream_snapshot.json")


def _scan_report(
        *,
        source_key: str = SOURCE_KEY,
        event_sha256: str | None = None,
        ) -> MediaWikiScanReport:
    text_size = 5
    page_count = 3
    xml_events = 7
    template_count = 0
    return MediaWikiScanReport(
        source_key,
        hashlib.sha256(b"xml-content").hexdigest(),
        hashlib.sha256(b"index-content").hexdigest(),
        page_count,
        2,
        1,
        2,
        0,
        CanonicalJsonObject.from_value({}),
        CanonicalJsonObject.from_value({"items": []}),
        text_size,
        template_count,
        4,
        0,
        xml_events,
        xml_events + page_count + text_size + template_count,
        1,
        event_sha256 or hashlib.sha256(b"events").hexdigest(),
    )


def _write_headers(path: Path) -> None:
    path.write_bytes(
        b"HTTP/1.1 200 OK\r\n"
        b"Date: Sun, 26 Jul 2026 09:16:54 GMT\r\n"
        b"Set-Cookie: private-cookie\r\n"
        b"X-Client-IP: 192.0.2.1\r\n"
        b"X-Proxy-Token: private-proxy\r\n\r\n"
    )


def _fixture(
        tmp_path: Path,
        *,
        source_key: str = SOURCE_KEY,
        project: str = PROJECT,
        ):
    root = tmp_path / "raw"
    prefix = Path(source_key)
    directory = root / prefix
    directory.mkdir(parents=True)
    xml_name = (
        f"{project}-20260701-pages-articles-multistream.xml.bz2")
    index_name = (
        f"{project}-20260701-pages-articles-multistream-index.txt.bz2")
    xml = directory / xml_name
    index = directory / index_name
    xml.write_bytes(b"synthetic-compressed-xml")
    index.write_bytes(b"synthetic-compressed-index")

    def raw_identity(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
            "sha1": hashlib.sha1(payload, usedforsecurity=False).hexdigest(),
            "size": len(payload),
            "url": f"/{project}/{DUMP_DATE}/{path.name}",
        }

    dumpstatus = directory / "dumpstatus.json"
    dumpstatus.write_text(json.dumps({
        "jobs": {
            "articlesmultistreamdumprecombine": {
                "files": {
                    xml_name: raw_identity(xml),
                    index_name: raw_identity(index),
                },
                "status": "done",
                "updated": "2026-07-06 11:23:25",
            },
        },
        "version": "0.8",
    }, separators=(",", ":")), encoding="utf-8")
    checksum = directory / f"{project}-20260701-sha1sums.txt"
    checksum.write_text(
        f"{raw_identity(xml)['sha1']}  {xml_name}\n"
        f"{raw_identity(index)['sha1']}  {index_name}\n",
        encoding="utf-8",
    )
    header_names = (
        "dumpstatus.json.headers.txt",
        "sha1sums.txt.headers.txt",
        f"{xml_name}.download.headers.txt",
        f"{index_name}.download.headers.txt",
    )
    for name in header_names:
        _write_headers(directory / name)
    report = _scan_report(source_key=source_key)
    reports = (
        directory / "final.pass-1.report.json",
        directory / "final.pass-2.report.json",
    )
    for path in reports:
        path.write_bytes(canonical_json_line(report.to_dict()))
    rel = prefix.as_posix()
    acquisition = MediaWikiSnapshotAcquisition(
        f"{rel}/dumpstatus.json",
        f"{rel}/dumpstatus.json.headers.txt",
        f"{rel}/{project}-20260701-sha1sums.txt",
        f"{rel}/sha1sums.txt.headers.txt",
        f"{rel}/{xml_name}",
        f"{rel}/{xml_name}.download.headers.txt",
        f"{rel}/{index_name}",
        f"{rel}/{index_name}.download.headers.txt",
        (f"{rel}/{reports[0].name}", f"{rel}/{reports[1].name}"),
    )
    manifest = build_mediawiki_dump_snapshot(
        raw_root=root,
        source_key=source_key,
        project=project,
        dump_date=DUMP_DATE,
        snapshot_id="synthetic-double-pass-v1",
        acquisition=acquisition,
    )
    return root, directory, acquisition, manifest


def test_snapshot_round_trip_verify_and_nonoverwrite(tmp_path):
    """Manifest is canonical, raw-verifiable, idempotent, and immutable."""
    raw_root, _, _, manifest = _fixture(tmp_path)
    output = tmp_path / "snapshot.json"
    write_mediawiki_dump_snapshot(manifest, output)
    restored = read_mediawiki_dump_snapshot(output)
    assert restored == manifest
    verify_mediawiki_dump_snapshot(restored, raw_root=raw_root)
    write_mediawiki_dump_snapshot(manifest, output)
    output.write_bytes(canonical_json_line({"damaged": 1}))
    with pytest.raises(MediaWikiSnapshotError, match="different content"):
        write_mediawiki_dump_snapshot(manifest, output)


def test_dumpstatus_and_project_checksum_must_agree(tmp_path):
    """Two official checksum sources cannot silently disagree."""
    raw_root, directory, acquisition, _ = _fixture(tmp_path)
    checksum = directory / "zhwiktionary-20260701-sha1sums.txt"
    lines = checksum.read_text(encoding="utf-8").splitlines()
    lines[0] = "0" * 40 + lines[0][40:]
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(MediaWikiSnapshotError, match="disagree"):
        build_mediawiki_dump_snapshot(
            raw_root=raw_root, source_key=SOURCE_KEY, project=PROJECT,
            dump_date=DUMP_DATE, snapshot_id="bad", acquisition=acquisition)


@pytest.mark.parametrize("field", ["size", "sha1"])
def test_bad_upstream_size_or_sha1_is_rejected(tmp_path, field):
    """Official metadata cannot authorize bytes with a wrong size or SHA-1."""
    raw_root, directory, acquisition, _ = _fixture(tmp_path)
    path = directory / "dumpstatus.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    files = value["jobs"]["articlesmultistreamdumprecombine"]["files"]
    xml = files[
        "zhwiktionary-20260701-pages-articles-multistream.xml.bz2"]
    if field == "size":
        xml["size"] += 1
    else:
        xml["sha1"] = "f" * 40
        checksum = directory / "zhwiktionary-20260701-sha1sums.txt"
        lines = checksum.read_text(encoding="utf-8").splitlines()
        lines[0] = "f" * 40 + lines[0][40:]
        checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(MediaWikiSnapshotError, match="size or upstream SHA-1"):
        build_mediawiki_dump_snapshot(
            raw_root=raw_root, source_key=SOURCE_KEY, project=PROJECT,
            dump_date=DUMP_DATE, snapshot_id="bad", acquisition=acquisition)


def test_bad_local_sha256_or_header_hash_is_rejected_on_verify(tmp_path):
    """Verification recomputes local bytes and every saved header hash."""
    raw_root, directory, _, manifest = _fixture(tmp_path)
    xml = manifest.raw_files[0]
    bad_raw = replace(manifest, raw_files=(
        replace(xml, local_sha256="0" * 64), manifest.raw_files[1]))
    with pytest.raises(MediaWikiSnapshotError, match="compressed raw identity"):
        verify_mediawiki_dump_snapshot(bad_raw, raw_root=raw_root)
    directory.joinpath("dumpstatus.json.headers.txt").write_bytes(b"changed")
    with pytest.raises(MediaWikiSnapshotError, match="official evidence"):
        verify_mediawiki_dump_snapshot(manifest, raw_root=raw_root)


def test_mismatched_or_damaged_saved_reports_are_rejected(tmp_path):
    """Both preserved EOF reports must be canonical and object-equal."""
    raw_root, directory, acquisition, _ = _fixture(tmp_path)
    second = directory / "final.pass-2.report.json"
    second.write_bytes(canonical_json_line(_scan_report(
        event_sha256="1" * 64).to_dict()))
    with pytest.raises(MediaWikiSnapshotError, match="reports disagree"):
        build_mediawiki_dump_snapshot(
            raw_root=raw_root, source_key=SOURCE_KEY, project=PROJECT,
            dump_date=DUMP_DATE, snapshot_id="bad", acquisition=acquisition)
    second.write_bytes(b'{"damaged":1}\n\n')
    with pytest.raises(MediaWikiSnapshotError, match="report"):
        build_mediawiki_dump_snapshot(
            raw_root=raw_root, source_key=SOURCE_KEY, project=PROJECT,
            dump_date=DUMP_DATE, snapshot_id="bad", acquisition=acquisition)


def test_license_attribution_and_external_truth_are_fixed(tmp_path):
    """Bad licensing, attribution, or truth authority fails construction."""
    _, _, _, manifest = _fixture(tmp_path)
    with pytest.raises(MediaWikiSnapshotError, match="license or attribution"):
        replace(manifest, license_id="UNKNOWN")
    with pytest.raises(MediaWikiSnapshotError, match="license or attribution"):
        replace(manifest, attribution_policy="Wiktionary")
    with pytest.raises(MediaWikiSnapshotError, match="definitive truth"):
        replace(manifest, definitive_truth_authoritative=1)
    assert manifest.attribution_policy == ATTRIBUTION_POLICY


def test_wikipedia_uses_project_specific_attribution(tmp_path):
    """Wikipedia cannot inherit Wiktionary contributor attribution."""
    _, _, _, manifest = _fixture(
        tmp_path,
        source_key="ZHWIKIPEDIA_20260701",
        project="zhwiki",
    )
    assert manifest.attribution_policy == WIKIPEDIA_ATTRIBUTION_POLICY
    assert "Wikipedia contributors" in manifest.attribution_policy
    assert "Wiktionary contributors" not in manifest.attribution_policy
    with pytest.raises(MediaWikiSnapshotError, match="license or attribution"):
        replace(manifest, attribution_policy=ATTRIBUTION_POLICY)


def test_sensitive_headers_are_hash_bound_but_not_serialized(tmp_path):
    """Cookie, proxy, client data, and private absolute paths stay out."""
    _, _, _, manifest = _fixture(tmp_path)
    serialized = manifest.canonical_bytes().decode("utf-8").casefold()
    assert "private-cookie" not in serialized
    assert "private-proxy" not in serialized
    assert "192.0.2.1" not in serialized
    assert str(tmp_path).casefold() not in serialized
    value = manifest.to_dict()
    value["proxy"] = "http://proxy.invalid:8080"
    path = tmp_path / "bad-sensitive.json"
    path.write_bytes(canonical_json_line(value))
    with pytest.raises(MediaWikiSnapshotError, match="fields are not exact"):
        read_mediawiki_dump_snapshot(path)


def test_path_escape_and_duplicate_report_path_fail_closed(tmp_path):
    """Acquisition paths must be portable and reports independently saved."""
    with pytest.raises(MediaWikiSnapshotError, match="safe POSIX"):
        MediaWikiSnapshotAcquisition(
            "../dumpstatus.json", "headers", "checksum", "headers2",
            "xml", "xml.headers", "index", "index.headers", ("r1", "r2"))
    with pytest.raises(MediaWikiSnapshotError, match="repeat"):
        MediaWikiSnapshotAcquisition(
            "dumpstatus.json", "headers", "checksum", "headers2",
            "xml", "xml.headers", "index", "index.headers", ("r1", "r1"))


def test_verify_does_not_invoke_mediawiki_parser(tmp_path, monkeypatch):
    """Snapshot verification consumes reports and hashes, not a third scan."""
    import pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter as adapter

    raw_root, _, _, manifest = _fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("third parser scan was invoked")

    monkeypatch.setattr(adapter, "scan_mediawiki_bz2", forbidden)
    verify_mediawiki_dump_snapshot(manifest, raw_root=raw_root)


def test_repository_manifest_freezes_full_double_pass_evidence():
    """Published snapshot carries actual July dump and full-EOF evidence."""
    manifest = read_mediawiki_dump_snapshot(MANIFEST_PATH)
    report = manifest.final_parser_report
    assert manifest.sha256() == (
        "9d0c82e39719a8084eb5bd672ba984589952874ee248c04816a00c7be20f2fdc")
    assert manifest.source_key == SOURCE_KEY
    assert manifest.license_id == "CC-BY-SA-4.0"
    assert manifest.double_pass_equal == 1
    assert report.full_eof_verified == 1
    assert report.page_count == 3_191_659
    assert report.main_namespace_count == 2_674_506
    assert report.valid_page_count == 2_674_143
    assert report.anomaly_codes.to_value() == {"UNBALANCED_TEMPLATE": 363}
    assert report.event_sha256 == (
        "47374ce883d60fbcf021fd999d693dea3232636ce68c243beae30370a51d2eb2")
    assert {item.local_sha256 for item in manifest.raw_files} == {
        "91272454ea18aaf24b2ba199428a1c23af5a0af6c8c4f2c1600ffa399c59ea80",
        "c7f950e5b802ed5700e678a306a2f98984639ff8f0f9bea5b19de5c12302f8fd",
    }


def test_repository_wikipedia_manifest_freezes_full_double_pass_evidence():
    """Wikipedia 正式 snapshot 固定独立来源、许可和双遍 EOF 身份。"""
    manifest = read_mediawiki_dump_snapshot(WIKIPEDIA_MANIFEST_PATH)
    report = manifest.final_parser_report
    assert manifest.sha256() == (
        "0e81569aaf6cf9cb688b41da27d5eff19707153ee5c74bf9bf362f34427869dd")
    assert manifest.source_key == "ZHWIKIPEDIA_20260701"
    assert manifest.project == "zhwiki"
    assert manifest.license_id == "CC-BY-SA-4.0"
    assert manifest.attribution_policy == WIKIPEDIA_ATTRIBUTION_POLICY
    assert manifest.double_pass_equal == 1
    assert report.full_eof_verified == 1
    assert report.page_count == 4_924_196
    assert report.main_namespace_count == 2_989_957
    assert report.valid_page_count == 2_989_957
    assert report.anomaly_codes.to_value() == {}
    assert report.event_sha256 == (
        "0aee92a9d99ae5d69631cef104543dd400696fa0f99d58f37f1d8a15969270ad")
    assert {item.local_sha256 for item in manifest.raw_files} == {
        "0847ccf01280578fc7414fe0c5bfeb8602b56898498331366f53946a07875a6c",
        "e87989682dbce323cc47b3d9a9e122776ab27504d2939488059524c6566fb25c",
    }
    assert {item.sha256 for item in manifest.parser_reports} == {
        "8e03c5e898fb4b06455f2f09941f068e37c8191d59f676af733649b0d92a0971",
    }
