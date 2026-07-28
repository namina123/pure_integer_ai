"""D-02 Wikidata fixed-revision parser、snapshot 与反向破坏 T0。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
    canonical_json_line,
)
from pure_integer_ai.experiments.ph2_wikidata_adapter import (
    WikidataAdapterError,
    parse_wikidata_statement,
    scan_wikidata_entity_bytes,
)
from pure_integer_ai.experiments.ph2_wikidata_allowlist import (
    read_wikidata_allowlist,
)
from pure_integer_ai.experiments.ph2_wikidata_snapshot import (
    WikidataEntitySnapshot,
    WikidataHttpCapture,
    WikidataRevisionSnapshotManifest,
    WikidataSnapshotError,
    read_wikidata_http_capture,
    read_wikidata_revision_snapshot,
    verify_wikidata_revision_snapshot,
    write_wikidata_revision_snapshot,
)


ALLOWLIST_PATH = Path("data/ph2/wikidata_revision_v1_allowlist_v2.json")
MANIFEST_PATH = Path(
    "data/ph2/manifests/wikidata_revision_v1.pinned_snapshot.json")


def _rules():
    """返回 v2 逐项冻结的 property rules。"""
    return read_wikidata_allowlist(ALLOWLIST_PATH).properties


def _rule(property_id: str):
    """按 property id 返回一个冻结 rule。"""
    return {item.property_id: item for item in _rules()}[property_id]


def _value_snak(
        property_id: str,
        *,
        datatype: str = "wikibase-item",
        value=None,
        ) -> dict:
    """构造一个带 datavalue 的 Wikidata snak。"""
    if value is None:
        value = {"entity-type": "item", "id": "Q89", "numeric-id": 89}
    return {
        "datatype": datatype,
        "datavalue": {"type": "wikibase-entityid", "value": value},
        "property": property_id,
        "snaktype": "value",
    }


def _statement(*, datatype: str = "wikibase-item") -> dict:
    """构造同时含 qualifier、reference 和 rank 的 P31 statement。"""
    return {
        "id": "Q313$statement-1",
        "mainsnak": _value_snak("P31", datatype=datatype),
        "qualifiers": {
            "P580": [{
                "datatype": "time",
                "datavalue": {
                    "type": "time",
                    "value": {
                        "after": 0,
                        "before": 0,
                        "calendarmodel": "http://www.wikidata.org/entity/Q1985727",
                        "precision": 9,
                        "time": "+2020-00-00T00:00:00Z",
                        "timezone": 0,
                    },
                },
                "hash": "qualifier-hash",
                "property": "P580",
                "snaktype": "value",
            }],
        },
        "qualifiers-order": ["P580"],
        "rank": "preferred",
        "references": [{
            "hash": "reference-hash",
            "snaks": {
                "P854": [{
                    "datatype": "url",
                    "datavalue": {
                        "type": "string",
                        "value": "https://example.invalid/source",
                    },
                    "property": "P854",
                    "snaktype": "value",
                }],
            },
            "snaks-order": ["P854"],
        }],
        "type": "statement",
    }


def _entity_payload(statement: dict, *, revision: int = 7) -> bytes:
    """构造一个最小但完整的 EntityData JSON。"""
    return canonical_json_bytes({
        "entities": {
            "Q313": {
                "aliases": {
                    "zh": [{"language": "zh", "value": "太白"}],
                },
                "claims": {"P31": [statement]},
                "descriptions": {
                    "zh": {"language": "zh", "value": "太阳系行星"},
                },
                "id": "Q313",
                "labels": {
                    "zh": {"language": "zh", "value": "金星"},
                },
                "lastrevid": revision,
                "modified": "2026-07-26T00:00:00Z",
                "ns": 0,
                "pageid": 313,
                "sitelinks": {},
                "title": "Q313",
                "type": "item",
            },
        },
    })


def _headers(path: Path) -> None:
    """写入含敏感非公开字段的合法 chunked HTTP 响应头。"""
    path.write_bytes(
        b"HTTP/1.1 200 OK\r\n"
        b"date: Sun, 26 Jul 2026 07:42:18 GMT\r\n"
        b"last-modified: Thu, 23 Jul 2026 22:43:17 GMT\r\n"
        b"content-type: application/json; charset=UTF-8\r\n"
        b"set-cookie: private-one\r\n"
        b"set-cookie: private-two\r\n"
        b"x-client-ip: 192.0.2.1\r\n"
        b"transfer-encoding: chunked\r\n\r\n"
    )


def test_statement_preserves_qualifier_rank_reference_and_order():
    """完整 payload 必须保留 qualifier/rank/reference，不能只抽 mainsnak。"""
    parsed = parse_wikidata_statement(
        _statement(), qid="Q313", property_rule=_rule("P31"))
    payload = parsed.payload.to_value()
    assert parsed.rank == "preferred"
    assert parsed.qualifier_snak_count == 1
    assert parsed.reference_count == 1
    assert parsed.reference_snak_count == 1
    assert payload["qualifiers-order"] == ["P580"]
    assert payload["references"][0]["snaks-order"] == ["P854"]
    assert payload["mainsnak"]["datavalue"]["value"]["numeric-id"] == 89
    assert parsed.snaktype_counts.to_value() == {"value": 3}


@pytest.mark.parametrize("snaktype", ["somevalue", "novalue"])
def test_nonvalue_snaktype_and_deprecated_rank_are_retained(snaktype):
    """somevalue/novalue 与 deprecated 只保留为非正证据，不得猜值。"""
    statement = _statement()
    statement["mainsnak"] = {
        "datatype": "wikibase-item",
        "property": "P31",
        "snaktype": snaktype,
    }
    statement["rank"] = "deprecated"
    parsed = parse_wikidata_statement(
        statement, qid="Q313", property_rule=_rule("P31"))
    assert parsed.rank == "deprecated"
    assert parsed.payload.to_value()["mainsnak"]["snaktype"] == snaktype
    assert "datavalue" not in parsed.payload.to_value()["mainsnak"]


def test_raw_decimal_is_tagged_as_text_without_binary_float():
    """上游坐标小数保留原词元，不进入 Python float 或规范整数域。"""
    statement = _statement()
    statement["qualifiers"] = {
        "P625": [{
            "datatype": "globe-coordinate",
            "datavalue": {
                "type": "globecoordinate",
                "value": {
                    "altitude": None,
                    "globe": "http://www.wikidata.org/entity/Q2",
                    "latitude": "DECIMAL_SLOT",
                    "longitude": 0,
                    "precision": "PRECISION_SLOT",
                },
            },
            "property": "P625",
            "snaktype": "value",
        }],
    }
    statement["qualifiers-order"] = ["P625"]
    payload = _entity_payload(statement).replace(
        b'"DECIMAL_SLOT"', b'1.25').replace(
        b'"PRECISION_SLOT"', b'0.01')
    report = scan_wikidata_entity_bytes(
        payload,
        expected_qid="Q313",
        expected_revision=7,
        property_rules=_rules(),
    )
    assert report.anomaly_count == 0
    assert report.datatype_counts.to_value()["globe-coordinate"] == 1


def test_bad_main_datatype_and_unknown_nested_datatype_are_atomic_anomalies():
    """坏 mainsnak 类型和未知 qualifier datatype 只隔离 statement。"""
    bad_main = scan_wikidata_entity_bytes(
        _entity_payload(_statement(datatype="string")),
        expected_qid="Q313",
        expected_revision=7,
        property_rules=_rules(),
    )
    assert bad_main.selected_statement_count == 1
    assert bad_main.valid_statement_count == 0
    assert bad_main.anomaly_codes.to_value() == {"BAD_MAIN_DATATYPE": 1}

    statement = _statement()
    statement["qualifiers"]["P580"][0]["datatype"] = "future-datatype"
    unknown = scan_wikidata_entity_bytes(
        _entity_payload(statement),
        expected_qid="Q313",
        expected_revision=7,
        property_rules=_rules(),
    )
    assert unknown.anomaly_codes.to_value() == {"UNKNOWN_DATATYPE": 1}


def test_duplicate_json_key_wrong_qid_and_wrong_revision_fail_closed():
    """重复 key、错误实体身份和错误 revision 都是文件级停线。"""
    payload = _entity_payload(_statement())
    duplicate = payload.replace(
        b'"lastrevid":7', b'"lastrevid":7,"lastrevid":7')
    with pytest.raises(WikidataAdapterError, match="重复"):
        scan_wikidata_entity_bytes(
            duplicate,
            expected_qid="Q313",
            expected_revision=7,
            property_rules=_rules(),
        )
    with pytest.raises(WikidataAdapterError, match="QID"):
        scan_wikidata_entity_bytes(
            payload,
            expected_qid="Q312",
            expected_revision=7,
            property_rules=_rules(),
        )
    with pytest.raises(WikidataAdapterError, match="revision"):
        scan_wikidata_entity_bytes(
            payload,
            expected_qid="Q313",
            expected_revision=8,
            property_rules=_rules(),
        )


def test_http_capture_exposes_only_safe_fields_and_binds_full_header_hash(
        tmp_path,
        ):
    """cookie/client IP 不进 manifest，但原始 headers 仍由 SHA-256 绑定。"""
    path = tmp_path / "response.headers"
    _headers(path)
    capture = read_wikidata_http_capture(
        path,
        response_url=(
            "https://www.wikidata.org/wiki/Special:EntityData/"
            "Q313.json?revision=7"
        ),
        captured_at_utc="2026-07-26T07:42:18.7420258Z",
    )
    exported = capture.to_dict()
    assert capture.response_headers_sha256 == hashlib.sha256(
        path.read_bytes()).hexdigest()
    assert "cookie" not in str(exported).casefold()
    assert "client" not in str(exported).casefold()
    assert "192.0.2.1" not in str(exported)


def test_snapshot_round_trip_verify_and_nonoverwrite(tmp_path):
    """单实体 synthetic manifest 可恢复、重扫并拒绝 raw/manifest 覆盖。"""
    raw_root = tmp_path / "raw"
    relative = "WIKIDATA_REVISION_V1/Q313.revision-7.json"
    raw = raw_root / Path(*relative.split("/"))
    raw.parent.mkdir(parents=True)
    raw.write_bytes(_entity_payload(_statement()))
    report = scan_wikidata_entity_bytes(
        raw.read_bytes(),
        expected_qid="Q313",
        expected_revision=7,
        property_rules=_rules(),
    )
    http = WikidataHttpCapture(
        200,
        "https://www.wikidata.org/wiki/Special:EntityData/Q313.json?revision=7",
        "Sun, 26 Jul 2026 07:42:18 GMT",
        "Thu, 23 Jul 2026 22:43:17 GMT",
        "application/json; charset=UTF-8",
        "1" * 64,
        "2026-07-26T07:42:18.7420258Z",
    )
    entity = WikidataEntitySnapshot(
        "Q313", 7, "train", "celestial-alias",
        ("ALIAS", "LABEL_ALIAS_DESCRIPTION"),
        relative, report.raw_sha256, report.raw_size_bytes, http, report,
    )
    manifest = WikidataRevisionSnapshotManifest(
        1,
        "WIKIDATA_REVISION_V1",
        "synthetic",
        2,
        read_wikidata_allowlist(ALLOWLIST_PATH).sha256(),
        1,
        1,
        "CC0-1.0",
        "https://www.wikidata.org/wiki/Wikidata:Licensing",
        "Wikidata contributors; fixed EntityData revisions retained",
        "PUBLIC",
        (entity,),
        CanonicalJsonObject.from_value({"entity_count": 1}),
    )
    output = tmp_path / "snapshot.json"
    write_wikidata_revision_snapshot(manifest, output)
    restored = read_wikidata_revision_snapshot(output)
    assert restored == manifest
    verify_wikidata_revision_snapshot(
        restored, raw_root=raw_root, property_rules=_rules())
    raw.write_bytes(raw.read_bytes() + b"\n")
    with pytest.raises(WikidataSnapshotError, match="identity"):
        verify_wikidata_revision_snapshot(
            restored, raw_root=raw_root, property_rules=_rules())
    output.write_bytes(canonical_json_line({"damaged": 1}))
    with pytest.raises(WikidataSnapshotError, match="内容不同"):
        write_wikidata_revision_snapshot(manifest, output)


def test_repository_manifest_freezes_actual_revisions_counts_and_hashes():
    """正式 manifest 直接冻结 11 revision、190 statement 与零异常。"""
    manifest = read_wikidata_revision_snapshot(MANIFEST_PATH)
    assert manifest.sha256() == (
        "b9b22a3673d5861dc41936356db5fa5ae57c7ea6e7d1972015b6c561865b5048")
    assert manifest.allowlist_revision == 2
    assert manifest.allowlist_sha256 == (
        "4dcdff27653e38544f02aea38b5e3623b27d632bb134c42b6b4a65b682d986f6")
    assert len(manifest.entities) == 11
    assert {item.qid: item.revision for item in manifest.entities} == {
        "Q89": 2521731013,
        "Q312": 2521239321,
        "Q313": 2521855466,
        "Q361": 2521225334,
        "Q362": 2522179884,
        "Q446": 2522147522,
        "Q1420": 2522562333,
        "Q5113": 2522566131,
        "Q25364": 2522253375,
        "Q82069695": 2517702049,
        "Q84263196": 2521485810,
    }
    aggregate = manifest.aggregate_report.to_value()
    assert aggregate["entity_count"] == 11
    assert aggregate["statement_count"] == 3720
    assert aggregate["selected_statement_count"] == 190
    assert aggregate["valid_statement_count"] == 190
    assert aggregate["anomaly_count"] == 0
    assert aggregate["qualifier_snak_count"] == 24
    assert aggregate["reference_count"] == 70
    assert aggregate["reference_snak_count"] == 141
    assert aggregate["rank_counts"] == {
        "deprecated": 2, "normal": 187, "preferred": 1,
    }
    assert aggregate["snaktype_counts"] == {"somevalue": 1, "value": 354}
    serialized = manifest.canonical_bytes().decode("utf-8").casefold()
    assert "set-cookie" not in serialized
    assert "x-client-ip" not in serialized


def test_manifest_rejects_noncanonical_bytes_and_bad_url_binding(tmp_path):
    """多余空白和 QID/revision 不一致的 URL 均不得恢复为正式 snapshot。"""
    path = tmp_path / "manifest.json"
    path.write_bytes(MANIFEST_PATH.read_bytes() + b"\n")
    with pytest.raises(WikidataSnapshotError):
        read_wikidata_revision_snapshot(path)
    manifest = read_wikidata_revision_snapshot(MANIFEST_PATH)
    first = manifest.entities[0]
    bad_http = replace(first.http, response_url=first.http.response_url.replace(
        first.qid, "Q999"))
    with pytest.raises(WikidataSnapshotError, match="URL/revision"):
        replace(first, http=bad_http)
